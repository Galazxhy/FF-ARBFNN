"""FF-FNN baselines: original FF loss and SymBa loss.
Supports two loss types:
- "original": asymmetric BCE loss with theta threshold (Hinton 2022)
- "symba": symmetric loss log(1+exp(G_neg-G_pos)) without theta (Lee & Song 2023)

Use loss_type="original" for the FF-FNN baseline in the paper.
Use loss_type="symba" for the improved FF+SymBa baseline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from baselines.base_model import BaseLocalLearningModel


class FFLayer(nn.Module):
    """Standard FF layer: Linear."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x):
        return self.linear(x)


class FFModel(BaseLocalLearningModel):
    """FF-FNN with configurable loss type.
    Architecture: multiple Linear layers, each trained with local FF/SymBa loss.
    """

    def __init__(self, input_dim, num_classes, hidden_dims, loss_type="original", theta=200):
        super().__init__(input_dim, num_classes, hidden_dims)
        self.loss_type = loss_type
        self.theta = theta
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(FFLayer(prev_dim, h_dim))
            prev_dim = h_dim

        if loss_type == "original":
            self.ff_loss_fn = nn.BCEWithLogitsLoss()

    def _original_loss(self, z_pos, z_neg):
        """Original FF loss: BCE on (sum of squares - theta)."""
        batch_size = z_pos.shape[0]
        z = torch.cat([z_pos, z_neg], dim=0)
        sum_sq = torch.sum(z**2, dim=-1)
        logits = sum_sq - self.theta
        labels = torch.zeros(z.shape[0], device=z.device)
        labels[:batch_size] = 1.0  # positive samples
        return self.ff_loss_fn(logits, labels)

    def _symba_loss(self, z_pos, z_neg):
        """Symmetric Ba loss: log(1 + exp(G_neg - G_pos)). No theta needed."""
        G_pos = torch.sum(z_pos**2, dim=-1)
        G_neg = torch.sum(z_neg**2, dim=-1)
        loss = torch.mean(torch.log(1 + torch.exp(G_neg - G_pos)))
        return loss

    def _get_ff_accuracy(self, z_pos, z_neg):
        """Binary accuracy: can we tell pos from neg by goodness?"""
        G_pos = torch.sum(z_pos**2, dim=-1)
        G_neg = torch.sum(z_neg**2, dim=-1)
        correct = (G_pos > G_neg).sum().item()
        total = z_pos.shape[0]
        return correct / total

    def _forward_layers(self, z):
        for layer in self.layers:
            h = layer(z)
            yield h
            z = h

    def forward(self, inputs, labels):
        scalar_outputs = {"Loss": torch.zeros(1, device=inputs["pos_sample"].device)}

        z_pos = inputs["pos_sample"].reshape(inputs["pos_sample"].shape[0], -1)
        z_neg = inputs["neg_sample"].reshape(inputs["neg_sample"].shape[0], -1)
        z_pos = self._layer_norm(z_pos)
        z_neg = self._layer_norm(z_neg)

        for idx, layer in enumerate(self.layers):
            h_pos = layer(z_pos)
            h_neg = layer(z_neg)

            if self.loss_type == "original":
                ff_loss = self._original_loss(h_pos, h_neg)
            else:
                ff_loss = self._symba_loss(h_pos, h_neg)

            ff_accuracy = self._get_ff_accuracy(h_pos, h_neg)

            scalar_outputs[f"loss_layer_{idx}"] = ff_loss
            scalar_outputs[f"ff_accuracy_layer_{idx}"] = ff_accuracy
            scalar_outputs[f"num_neurons_layer_{idx}"] = torch.tensor(
                h_pos.shape[-1], dtype=torch.float32
            )
            scalar_outputs["Loss"] += ff_loss

            z_pos = self._layer_norm(h_pos.detach())
            z_neg = self._layer_norm(h_neg.detach())

        return scalar_outputs


# Backward-compatible alias
FFSymBa = FFModel  # default uses original loss; pass loss_type="symba" for SymBa
