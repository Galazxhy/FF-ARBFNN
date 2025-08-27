"""
Author: Galazxhy galazxhy@163.com
Date: 2025-08-09 12:10:30
LastEditors: Galazxhy galazxhy@163.com
LastEditTime: 2025-08-09 12:10:31
FilePath: /SORBF/model/FF_RBF.py
Description:

Copyright (c) 2025 by Astroyd, All Rights Reserved.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from model.layer import RBFLayer
from model.utils import get_accuracy
from torchviz import make_dot


class FF_RBF(nn.Module):
    def __init__(self, out_features=4):
        super(FF_RBF, self).__init__()

        self.rbf_embeddings = nn.ModuleList([])
        for i in range(len(config.init_num_centers) - 1):
            self.rbf_embeddings.append(
                RBFLayer(config.init_num_centers[i + 1], config.init_num_centers[i]),
            )

        channels_for_classification_loss = sum(
            [
                config.init_num_centers[i + 1]
                for i in range(len(config.init_num_centers) - 1)
            ]
        )

        self.fc_out = nn.Linear(
            channels_for_classification_loss, out_features, bias=False
        )
        self.classification_weight = nn.Parameter(
            torch.randn(channels_for_classification_loss, out_features)
        )
        self.bias = nn.Parameter(torch.randn(1, out_features))
        self.act_fn = ReLU_full_grad()

        self.ff_loss = nn.BCEWithLogitsLoss()
        self.classification_loss = nn.CrossEntropyLoss()
        self.dup_neurons = 0
        self.del_neurons = 0
        self.organized = False

    def fc_out(self, input):
        return torch.mm(input, self.classification_weight) + self.bias

    def add_weight(self, indices):
        """
        Add Neurons
        """
        best_weights = nn.Parameter(
            torch.index_select(self.classification_weight.detach(), 0, indices)
        )
        if (
            best_weights.shape[0] != 0
            and self.classification_weight.shape[0] < config.max_neurons
        ):
            self.classification_weight = nn.Parameter(
                torch.cat(
                    [self.classification_weight.detach(), best_weights],
                    dim=0,
                )
            )
            return True
        return False

    def del_weight(self, indices):
        """
        Delete Neurons
        """
        if indices.shape[0] != 0:
            mask = torch.ones(self.classification_weight.size(0), dtype=torch.bool)
            mask[indices] = False
            self.classification_weight = nn.Parameter(
                self.classification_weight.detach()[mask]
            )
            return True
        return False

    def _layer_norm(self, z, eps=1e-8):
        return z / (torch.sqrt(torch.mean(z**2, dim=-1, keepdim=True)) + eps)
        # return (z - z.mean(dim=1, keepdim=True)) / (z.std(dim=1, keepdim=True) + eps)

    def _calc_ff_loss(self, z, labels):
        sum_of_squares = torch.sum(z**2, dim=-1)

        logits = sum_of_squares - config.theta
        # print(logits)
        ff_loss = self.ff_loss(logits, labels.float())

        with torch.no_grad():
            ff_accuracy = (
                torch.sum((torch.sigmoid(logits) > 0.5) == labels) / z.shape[0]
            ).item()
        return ff_loss, ff_accuracy

    def self_organize(self, inputs):
        with torch.no_grad():
            z = torch.cat([inputs["pos_sample"], inputs["neg_sample"]], dim=0)
            z = z.reshape(z.shape[0], -1)
            z = self._layer_norm(z)
            for i, layer in enumerate(self.rbf_embeddings):
                hid = layer(z)

                sum_of_squares = torch.sum(hid**2, dim=-1)
                g_pos = sum_of_squares[: config.batch_size]
                g_neg = sum_of_squares[config.batch_size :]

                lgt_pos = g_pos - config.theta
                lgt_neg = g_neg - config.theta

                add_pos_mask = torch.sigmoid(lgt_pos) > 0.9
                add_neg_mask = torch.sigmoid(lgt_neg) < 0.1

                add_mask = add_pos_mask & add_neg_mask
                self.dup_neurons = add_mask.sum()
                # print(self.dup_neurons)

                layer.addNeurons(torch.where(add_mask)[0])
                added = self.add_weight(torch.where(add_mask)[0])
                self.organized = self.organized or added

                del_pos_mask = torch.sigmoid(lgt_pos) < 0.4
                del_neg_mask = torch.sigmoid(lgt_neg) > 0.6

                del_mask = del_pos_mask & del_neg_mask
                self.del_neurons = del_mask.sum()
                # print(self.del_neurons)

                layer.delNeurons(torch.where(del_mask)[0])
                deled = self.del_weight(torch.where(del_mask)[0])
                self.organized = self.organized or deled

                # print(self.dup_neurons, self.del_neurons)

                z = hid
                z = self._layer_norm(z)

    def forward(self, inputs, labels):
        self.dup_neurons = 0
        self.del_neurons = 0
        self.organized = False
        scalar_outputs = {
            "Loss": torch.zeros(1, device=torch.device(config.device)),
            "Dup_neurons": torch.zeros(1, device=torch.device(config.device)),
            "Del_neurons": torch.zeros(1, device=torch.device(config.device)),
        }

        z = torch.cat([inputs["pos_sample"], inputs["neg_sample"]], dim=0)
        posneg_labels = torch.ones(z.shape[0], device=torch.device(config.device))
        posneg_labels[config.batch_size :] = 0

        z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z)
        for idx, layer in enumerate(self.rbf_embeddings):
            z = layer(z)

            ff_loss, ff_accuracy = self._calc_ff_loss(z, posneg_labels)
            scalar_outputs[f"loss_layer_{idx}"] = ff_loss
            scalar_outputs[f"ff_accuracy_layer_{idx}"] = ff_accuracy
            scalar_outputs["Loss"] += ff_loss
            # g = make_dot(z)
            # g.render(filename=f"Graph {idx}", view=False)
            z = z.detach()
            # self._self_organize(layer, z)
            scalar_outputs[f"num_neurons_layer_{idx}"] = layer.getNeuronNum()

            z = self._layer_norm(z)

        scalar_outputs = self.forward_downstream_classification_model(
            inputs, labels, scalar_outputs=scalar_outputs
        )

        return scalar_outputs

    def forward_downstream_classification_model(
        self, inputs, labels, scalar_outputs=None
    ):
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=torch.device(config.device))
            }

        z = inputs["natrual_sample"]
        z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z)

        input_classification_model = []

        with torch.no_grad():
            for idx, layer in enumerate(self.rbf_embeddings):
                z = layer(z)
                z = self._layer_norm(z)

                if idx >= 0:
                    input_classification_model.append(z)

        input_classification_model = torch.concat(input_classification_model, dim=-1)
        output = self.fc_out(input_classification_model.detach())
        # output = output - torch.max(output, dim=-1, keepdim=True)[0]

        classification_loss = self.classification_loss(output, labels["class_labels"])
        classification_accuracy = get_accuracy(output.data, labels["class_labels"])

        scalar_outputs["output"] = output
        scalar_outputs["Loss"] += classification_loss

        scalar_outputs["classification_loss"] = classification_loss
        scalar_outputs["classification_accuracy"] = classification_accuracy

        return scalar_outputs


class ReLU_full_grad(torch.autograd.Function):
    """ReLU activation function that passes through the gradient irrespective of its input value."""

    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()
