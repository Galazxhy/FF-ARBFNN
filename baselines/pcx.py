"""PCX (Predictive Coding) baseline.
Predictive coding network with iterative inference dynamics.
Each layer predicts the activity of the next layer; errors flow locally.

Reference: Pinchetti et al., "Benchmarking Predictive Coding Networks -- Made Simple," 2024.
Code: github.com/liukidar/pcax
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from baselines.base_model import BaseLocalLearningModel


class PCLayer(nn.Module):
    """One PC layer with forward weights (W_f) and backward weights (W_b)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_f = nn.Linear(in_dim, out_dim, bias=True)  # forward prediction
        self.W_b = nn.Linear(out_dim, in_dim, bias=True)  # backward projection

    def forward(self, x):
        """Forward prediction: given lower activity, predict upper activity."""
        return self.W_f(x)

    def backward(self, x_upper):
        """Backward projection: given upper activity, project to lower."""
        return self.W_b(x_upper)


class PCXModel(BaseLocalLearningModel):
    """Predictive Coding network for classification.

    Training has two phases:
    1. Inference: iterate activities to minimize prediction errors (free phase)
    2. Learning: update weights using converged errors

    For efficiency, we use a fixed number of inference steps.
    """

    def __init__(self, input_dim, num_classes, hidden_dims, inference_steps=10):
        super().__init__(input_dim, num_classes, hidden_dims)
        self.inference_steps = inference_steps

        self.pc_layers = nn.ModuleList()
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.pc_layers.append(PCLayer(prev_dim, h_dim))
            prev_dim = h_dim

        # Top layer predicts class logits
        self.top_layer = nn.Linear(hidden_dims[-1], num_classes, bias=True)

    def _layer_norm(self, z, eps=1e-8):
        return z / (torch.sqrt(torch.mean(z**2, dim=-1, keepdim=True)) + eps)

    def _infer_activities(self, x, class_labels=None):
        """Iteratively settle activities to minimize prediction errors."""
        batch_size = x.shape[0]
        device = x.device

        # Initialize activities
        activities = [x]
        for layer in self.pc_layers:
            with torch.no_grad():
                a_next = F.relu(layer(activities[-1]))
            activities.append(a_next)

        # Initialize top-level target
        if class_labels is not None:
            top_target = F.one_hot(class_labels, num_classes=self.num_classes).float()
        else:
            top_target = torch.zeros(batch_size, self.num_classes, device=device)

        # Iterative refinement
        for _ in range(self.inference_steps):
            # Bottom-up pass (compute predictions)
            predictions = [activities[0]]
            for i, layer in enumerate(self.pc_layers):
                pred = layer(predictions[-1])
                predictions.append(pred)

            # Top-down pass (compute errors and update activities)
            # Top layer error
            top_pred = self.top_layer(activities[-1])
            if class_labels is not None:
                top_error = top_target - top_pred
                # Update top activity
                activities[-1] = activities[-1] + 0.01 * torch.mm(
                    top_error, self.top_layer.weight
                )

            # Layer-wise errors (top-down)
            for i in reversed(range(len(self.pc_layers))):
                # Prediction error for this layer
                error = activities[i + 1] - predictions[i + 1]
                # Update activity
                activities[i + 1] = activities[i + 1] + 0.01 * error
                # Propagate error downward
                if i > 0:
                    back_error = self.pc_layers[i].backward(error)
                    activities[i] = activities[i] + 0.01 * F.relu(back_error)

        return activities

    def _forward_layers(self, z):
        activities = self._infer_activities(z, class_labels=None)
        for a in activities[1:]:  # skip input
            yield a

    def forward(self, inputs, labels):
        """PC training: inference with label target + weight update from errors."""
        device = inputs["pos_sample"].device
        batch_size = inputs["pos_sample"].shape[0]

        x = inputs["pos_sample"].reshape(batch_size, -1)
        x = self._layer_norm(x)
        class_labels = labels["class_labels"]

        # Inference phase (with class labels as top target)
        activities = self._infer_activities(x, class_labels)

        # Learning phase (compute weight updates from converged errors)
        scalar_outputs = {"Loss": torch.zeros(1, device=device)}

        # Compute prediction errors and losses
        total_loss = torch.tensor(0.0, device=device)
        for i, layer in enumerate(self.pc_layers):
            pred = layer(activities[i])
            target = activities[i + 1].detach()
            layer_loss = F.mse_loss(pred, target)

            scalar_outputs[f"loss_layer_{i}"] = layer_loss
            scalar_outputs[f"pc_error_layer_{i}"] = layer_loss.item()
            total_loss = total_loss + layer_loss

            scalar_outputs[f"num_neurons_layer_{i}"] = torch.tensor(
                activities[i + 1].shape[-1], dtype=torch.float32
            )

        # Top layer loss
        top_pred = self.top_layer(activities[-1])
        top_target = F.one_hot(class_labels, num_classes=self.num_classes).float()
        top_loss = F.mse_loss(top_pred, top_target)
        top_acc = (top_pred.argmax(1) == class_labels).float().mean()
        total_loss = total_loss + top_loss

        scalar_outputs["Loss"] = total_loss
        scalar_outputs["top_accuracy"] = top_acc.item()

        return scalar_outputs

    def forward_downstream(self, inputs, labels, scalar_outputs=None):
        """Downstream classification using converged PC activities."""
        if scalar_outputs is None:
            scalar_outputs = {
                "classification_loss": torch.zeros(
                    1, device=inputs["natrual_sample"].device
                )
            }

        z = inputs["natrual_sample"]
        z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z)

        with torch.no_grad():
            activities = self._infer_activities(z, class_labels=None)

        all_layer_outputs = [
            self._layer_norm(a) for a in activities[1:]
        ]
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
