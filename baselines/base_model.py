"""Base class for all local learning baseline models."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import utils
from config import config as original_config


class BaseLocalLearningModel(nn.Module):
    """Abstract base for local learning models.
    Subclasses implement: forward_local() for layer-wise training,
    and forward_downstream() for task head training.
    """

    def __init__(self, input_dim, num_classes, hidden_dims, out_features=None):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        if out_features is None:
            self.out_features = num_classes

        # Task head: linear classifier on concatenated layer outputs
        total_hidden = sum(hidden_dims)
        self.fc_out = nn.Linear(total_hidden, num_classes, bias=True)
        self.classification_loss = nn.CrossEntropyLoss()

    def _layer_norm(self, z, eps=1e-8):
        return z / (torch.sqrt(torch.mean(z**2, dim=-1, keepdim=True)) + eps)

    def get_accuracy(self, output, target):
        with torch.no_grad():
            prediction = torch.argmax(output, dim=1)
            return (prediction == target).sum() / target.shape[0]

    def forward_downstream(self, inputs, labels, scalar_outputs=None):
        """Task head forward pass. Uses 'natrual_sample' (keeping original typo for compatibility)."""
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
            for layer_output in self._forward_layers(z):
                normalized = self._layer_norm(layer_output)
                all_layer_outputs.append(normalized)

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

    def _forward_layers(self, z):
        """Generator that yields layer outputs one by one. Override in subclasses."""
        raise NotImplementedError

    def forward(self, inputs, labels):
        """Full forward pass. Override in subclasses."""
        raise NotImplementedError
