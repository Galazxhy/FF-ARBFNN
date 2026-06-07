"""CaFo (Cascaded Forward) baseline.
Cascaded forward blocks with per-layer predictors.
No negative samples needed — each block directly outputs a label distribution.

Reference: Zhao et al., "The Cascaded Forward algorithm for neural network training,"
Pattern Recognition, 2024. Code: github.com/Graph-ZKY/CaFo
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from baselines.base_model import BaseLocalLearningModel


class CaFoBlock(nn.Module):
    """One cascaded forward block: Linear + ReLU + LayerNorm + predictor."""

    def __init__(self, in_dim, out_dim, num_classes, alpha=0.5):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        self.predictor = nn.Linear(out_dim, num_classes, bias=True)
        self.alpha = alpha

    def forward(self, x):
        h = F.relu(self.linear(x))
        h_norm = h / (torch.sqrt(torch.mean(h**2, dim=-1, keepdim=True)) + 1e-8)
        logits = self.predictor(h_norm)
        return h_norm, logits


class CaFo(BaseLocalLearningModel):
    """Cascaded Forward network.
    Each block produces its own class prediction via a local predictor.
    The final prediction is a weighted sum of all block predictions.
    """

    def __init__(self, input_dim, num_classes, hidden_dims, alpha=0.5):
        super().__init__(input_dim, num_classes, hidden_dims)
        self.alpha = alpha
        self.blocks = nn.ModuleList()
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.blocks.append(CaFoBlock(prev_dim, h_dim, num_classes, alpha))
            prev_dim = h_dim

    def _layer_norm(self, z, eps=1e-8):
        return z / (torch.sqrt(torch.mean(z**2, dim=-1, keepdim=True)) + eps)

    def _forward_layers(self, z):
        for block in self.blocks:
            h, _ = block(z)
            yield h
            z = h

    def forward(self, inputs, labels):
        """CaFo local training: each block's predictor is trained with
        cross-entropy on true labels. No negative samples needed.
        """
        device = inputs["pos_sample"].device
        batch_size = inputs["pos_sample"].shape[0]

        # Use pos_sample as the actual data (CaFo doesn't need neg samples)
        z = inputs["pos_sample"].reshape(batch_size, -1)
        z = self._layer_norm(z)
        class_labels = labels["class_labels"]

        scalar_outputs = {"Loss": torch.zeros(1, device=device)}
        all_logits = []

        for idx, block in enumerate(self.blocks):
            h_norm, logits = block(z)
            all_logits.append(logits)

            # Local predictor loss
            pred_loss = F.cross_entropy(logits, class_labels)
            pred_acc = (logits.argmax(1) == class_labels).float().mean()

            scalar_outputs[f"loss_layer_{idx}"] = pred_loss
            scalar_outputs[f"ff_accuracy_layer_{idx}"] = pred_acc.item()
            scalar_outputs["Loss"] += pred_loss

            z = h_norm.detach()  # gradient isolation between blocks

        # Weighted ensemble prediction
        ensemble_logits = torch.stack(all_logits).mean(dim=0)
        ensemble_acc = (ensemble_logits.argmax(1) == class_labels).float().mean()
        scalar_outputs["ensemble_accuracy"] = ensemble_acc.item()

        return scalar_outputs

    def forward_downstream(self, inputs, labels, scalar_outputs=None):
        """For CaFo, downstream uses the last block's hidden state + all predictors."""
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
        all_logits = []
        with torch.no_grad():
            for block in self.blocks:
                h_norm, logits = block(z)
                all_layer_outputs.append(h_norm)
                all_logits.append(logits)
                z = h_norm

        # Concatenate for task head
        concat_output = torch.cat(all_layer_outputs, dim=-1)
        output = self.fc_out(concat_output.detach())
        output = output - torch.max(output, dim=-1, keepdim=True)[0]

        # Also add ensemble prediction from local predictors
        ensemble_logits = torch.stack(all_logits).mean(dim=0)
        output = output + 0.1 * ensemble_logits  # slight boost from local predictors

        classification_loss = self.classification_loss(
            output, labels["class_labels"]
        )
        accuracy = self.get_accuracy(output.data, labels["class_labels"])

        scalar_outputs["output"] = output
        scalar_outputs["classification_loss"] = classification_loss
        scalar_outputs["classification_accuracy"] = accuracy.item()

        return scalar_outputs
