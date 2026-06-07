"""InfoPro baseline.
Gradient-isolated modules trained with Information Propagation loss.
Each module maximizes I(h; y) while minimizing I(h; x_prev) via an
information bottleneck surrogate.

Reference: Wang et al., "InfoPro: Locally Supervised Deep Learning by
Maximizing Information Propagation," IJCV, 2024.
Code: github.com/blackfeather-wang/InfoPro-Pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from baselines.base_model import BaseLocalLearningModel


class InfoProModule(nn.Module):
    """One InfoPro module: Linear + ReLU + local classifier + reconstruction head."""

    def __init__(self, in_dim, out_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        # Local classifier (maximizes I(h; y))
        self.local_classifier = nn.Linear(out_dim, num_classes, bias=True)
        # Reconstruction head (regularizes I(h; x) — info bottleneck)
        self.reconstructor = nn.Linear(out_dim, in_dim, bias=True)

    def forward(self, x):
        h = F.relu(self.linear(x))
        return h

    def classify(self, h):
        return self.local_classifier(h)

    def reconstruct(self, h):
        return self.reconstructor(h)


class InfoPro(BaseLocalLearningModel):
    """InfoPro: locally supervised with information propagation.

    Each module is trained with:
    L = L_classify + beta * L_reconstruct
    where:
    - L_classify = CE(local_classifier(h), y) — maximizes I(h; y)
    - L_reconstruct = MSE(reconstruct(h), x) — regularizes I(h; x)
    """

    def __init__(self, input_dim, num_classes, hidden_dims, beta=0.1):
        super().__init__(input_dim, num_classes, hidden_dims)
        self.beta = beta
        self.info_blocks = nn.ModuleList()
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.info_blocks.append(InfoProModule(prev_dim, h_dim, num_classes))
            prev_dim = h_dim

    def _layer_norm(self, z, eps=1e-8):
        return z / (torch.sqrt(torch.mean(z**2, dim=-1, keepdim=True)) + eps)

    def _forward_layers(self, z):
        for module in self.info_blocks:
            h = module(z)
            yield h
            z = h

    def forward(self, inputs, labels):
        """InfoPro local training with classification + reconstruction loss."""
        device = inputs["pos_sample"].device
        batch_size = inputs["pos_sample"].shape[0]

        z = inputs["pos_sample"].reshape(batch_size, -1)
        z = self._layer_norm(z)
        class_labels = labels["class_labels"]

        scalar_outputs = {"Loss": torch.zeros(1, device=device)}
        prev_input = z

        for idx, module in enumerate(self.info_blocks):
            h = module(prev_input)
            h_norm = self._layer_norm(h)

            # Local classification loss (maximize I(h; y))
            logits = module.classify(h_norm)
            cls_loss = F.cross_entropy(logits, class_labels)
            cls_acc = (logits.argmax(1) == class_labels).float().mean()

            # Reconstruction loss (regularize I(h; x_prev))  — information bottleneck
            recon = module.reconstruct(h_norm)
            recon_loss = F.mse_loss(recon, prev_input)

            total_loss = cls_loss + self.beta * recon_loss

            scalar_outputs[f"loss_layer_{idx}"] = total_loss
            scalar_outputs[f"cls_acc_layer_{idx}"] = cls_acc.item()
            scalar_outputs[f"recon_loss_layer_{idx}"] = recon_loss.item()
            scalar_outputs["Loss"] += total_loss

            prev_input = h_norm.detach()  # gradient isolation

            # Store per-layer neuron count placeholder
            scalar_outputs[f"num_neurons_layer_{idx}"] = torch.tensor(
                h.shape[-1], dtype=torch.float32
            )

        return scalar_outputs

    def forward_downstream(self, inputs, labels, scalar_outputs=None):
        """Task head on concatenated InfoPro representations."""
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
            for module in self.info_blocks:
                h = module(z)
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
