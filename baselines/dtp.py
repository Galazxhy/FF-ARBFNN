"""DTP (Difference Target Propagation) baseline.
Each layer learns a feedback network that propagates target values
from higher layers to lower layers, replacing gradient chains.

Reference: Sato et al., "Fixed-Weight Difference Target Propagation," AAAI, 2023.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from baselines.base_model import BaseLocalLearningModel


class DTPLayer(nn.Module):
    """One DTP layer: forward mapping (F_i) + feedback mapping (G_i)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fwd = nn.Linear(in_dim, out_dim, bias=True)  # F_i
        self.fbk = nn.Linear(out_dim, in_dim, bias=True)   # G_i

    def forward_pass(self, x):
        return F.relu(self.fwd(x))

    def feedback_pass(self, h):
        """Map upper target back to lower target."""
        return self.fbk(h)


class DTPModel(BaseLocalLearningModel):
    """Difference Target Propagation network.

    Training:
    1. Forward pass to get activations h_i
    2. Set top target t_L from labels
    3. Propagate targets downward: t_i = h_i + G_i(t_{i+1} - h_{i+1})
    4. Update forward weights: minimize ||h_i - t_i||^2
    5. Update feedback weights: minimize ||G_i(h_{i+1}) - (t_i - h_i)||^2

    The "difference" in DTP: targets are propagated as corrections (differences)
    rather than absolute values.
    """

    def __init__(self, input_dim, num_classes, hidden_dims):
        super().__init__(input_dim, num_classes, hidden_dims)
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(DTPLayer(prev_dim, h_dim))
            prev_dim = h_dim

        # Top layer: maps last hidden to class logits
        self.top_forward = nn.Linear(hidden_dims[-1], num_classes, bias=True)
        self.top_feedback = nn.Linear(num_classes, hidden_dims[-1], bias=True)

    def _layer_norm(self, z, eps=1e-8):
        return z / (torch.sqrt(torch.mean(z**2, dim=-1, keepdim=True)) + eps)

    def _compute_targets(self, activations, class_labels):
        """Compute target values for each layer via DTP.

        activations: list [h_0(input), h_1, h_2, ..., h_L(last_hidden)]
        Returns: targets [t_0, t_1, ..., t_L] where t_i is the target for h_i
        """
        n_layers = len(self.layers)
        targets = [None] * (n_layers + 1)  # +1 for input

        # Top target: one-hot label encoding
        top_target = F.one_hot(class_labels, num_classes=self.num_classes).float()

        # Get last hidden -> class prediction
        h_top = activations[-1]
        top_pred = self.top_forward(h_top)
        top_diff = top_target - top_pred

        # Target for last hidden layer
        targets[-1] = h_top + 0.1 * self.top_feedback(top_diff)

        # Propagate down
        for i in reversed(range(n_layers)):
            h_i = activations[i]
            h_next = activations[i + 1]
            t_next = targets[i + 1]

            # Difference target propagation
            diff = t_next - h_next
            targets[i] = h_i + 0.1 * self.layers[i].feedback_pass(diff)

        return targets, top_target, top_pred

    def _forward_layers(self, z):
        activations = [z]
        for layer in self.layers:
            h = layer.forward_pass(activations[-1])
            activations.append(h)
        for h in activations[1:]:
            yield h

    def forward(self, inputs, labels):
        """DTP training with target propagation."""
        device = inputs["pos_sample"].device
        batch_size = inputs["pos_sample"].shape[0]

        x = inputs["pos_sample"].reshape(batch_size, -1)
        x = self._layer_norm(x)
        class_labels = labels["class_labels"]

        # Forward pass: collect activations
        activations = [x]
        for layer in self.layers:
            h = layer.forward_pass(activations[-1])
            h_norm = self._layer_norm(h)
            activations.append(h_norm)

        # Compute targets via DTP
        targets, top_target, top_pred = self._compute_targets(
            activations, class_labels
        )

        scalar_outputs = {"Loss": torch.zeros(1, device=device)}
        total_loss = torch.tensor(0.0, device=device)

        # Forward weight losses (minimize ||h_i - t_i||^2 for each layer)
        for i, layer in enumerate(self.layers):
            h_i = activations[i + 1]
            t_i = targets[i + 1]
            forward_loss = F.mse_loss(h_i, t_i.detach())
            scalar_outputs[f"forward_loss_layer_{i}"] = forward_loss
            total_loss = total_loss + forward_loss

        # Top layer loss
        top_loss = F.cross_entropy(top_pred, class_labels)
        top_acc = (top_pred.argmax(1) == class_labels).float().mean()
        total_loss = total_loss + top_loss

        scalar_outputs["Loss"] = total_loss
        scalar_outputs["top_accuracy"] = top_acc.item()

        # Feedback weight losses (inverse training)
        for i, layer in enumerate(self.layers):
            h_i = activations[i + 1]
            t_i = targets[i + 1]
            h_prev = activations[i]

            # Feedback should map: G_i(h_i) ≈ (t_prev - h_prev) for inverse
            feedback_pred = layer.feedback_pass(h_i)
            feedback_target = (targets[i] - activations[i]).detach()
            feedback_loss = F.mse_loss(feedback_pred, feedback_target)
            scalar_outputs[f"feedback_loss_layer_{i}"] = feedback_loss
            total_loss = total_loss + 0.1 * feedback_loss

            scalar_outputs[f"num_neurons_layer_{i}"] = torch.tensor(
                h_i.shape[-1], dtype=torch.float32
            )

        scalar_outputs["Loss"] = total_loss

        return scalar_outputs

    def forward_downstream(self, inputs, labels, scalar_outputs=None):
        """Task head on DTP hidden representations."""
        if scalar_outputs is None:
            scalar_outputs = {
                "classification_loss": torch.zeros(
                    1, device=inputs["natrual_sample"].device
                )
            }

        z = inputs["natrual_sample"]
        z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z)

        all_layer_outputs = []
        with torch.no_grad():
            for layer in self.layers:
                h = layer.forward_pass(z)
                h_norm = self._layer_norm(h)
                all_layer_outputs.append(h_norm)
                z = h_norm

        concat_output = torch.cat(all_layer_outputs, dim=-1)
        output = self.fc_out(concat_output.detach())
        output = output - torch.max(output, dim=-1, keepdim=True)[0]

        classification_loss = self.classification_loss(
            output, labels["class_labels"]
        )
        accuracy = self.get_accuracy(output.data, labels["class_labels"])

        scalar_outputs["output"] = output
        scalar_outputs["classification_loss"] = classification_loss
        scalar_outputs["classification_accuracy"] = accuracy.item()

        return scalar_outputs
