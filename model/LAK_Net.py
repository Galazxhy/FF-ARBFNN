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

        self.layers = nn.ModuleList([])
        for i in range(len(config.init_num_centers) - 1):
            self.layers.append(
                RBFLayer(
                    config.init_num_centers[i + 1],
                    config.init_num_centers[i],
                    config.init_num_centers[i + 1],
                )
            )

        channels_for_classification_loss = sum(
            [
                config.init_num_centers[i + 1]
                for i in range(len(config.init_num_centers) - 1)
            ]
        )

        self.fc_out = nn.Linear(
            channels_for_classification_loss, out_features, bias=True
        )

        # self.act_fn = nn.ReLU()

        self.ff_loss = nn.BCEWithLogitsLoss()
        self.classification_loss = nn.CrossEntropyLoss()
        self.organized = False

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

    def embedding(self, inputs):
        z = torch.cat([inputs["pos_sample"], inputs["neg_sample"]], dim=0)
        z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z)

        pos_embeddings = []
        neg_embeddings = []

        for idx, layer in enumerate(self.layers):
            rbf_out = layer(z)
            z = layer.mapping(rbf_out)

            if idx >= 0:
                pos_embeddings.append(z[: z.shape[0] // 2])
                neg_embeddings.append(z[z.shape[0] // 2 :])

        pos_embeddings = torch.concat(pos_embeddings, dim=-1)
        neg_embeddings = torch.concat(neg_embeddings, dim=-1)

        return pos_embeddings, neg_embeddings

    def forward(self, inputs, labels):
        self.organized = False
        scalar_outputs = {"Loss": torch.zeros(1, device=torch.device(config.device))}

        z = torch.cat([inputs["pos_sample"], inputs["neg_sample"]], dim=0)
        posneg_labels = torch.ones(z.shape[0], device=torch.device(config.device))
        posneg_labels[config.batch_size :] = 0

        z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z)
        for idx, layer in enumerate(self.layers):
            # print(z.shape)
            rbf_out = layer(z)
            z = layer.mapping(rbf_out)

            ff_loss, ff_accuracy = self._calc_ff_loss(z, posneg_labels)
            scalar_outputs[f"loss_layer_{idx}"] = ff_loss
            scalar_outputs[f"ff_accuracy_layer_{idx}"] = ff_accuracy
            scalar_outputs["Loss"] += ff_loss
            # g = make_dot(z)
            # g.render(filename=f"Graph {idx}", view=False)
            z = z.detach()
            rbf_out = rbf_out.detach()
            organized, dup_neurons, del_neurons = layer.adjustStruc(rbf_out.detach())
            self.organized = self.organized or organized

            scalar_outputs[f"num_neurons_layer_{idx}"] = layer.getNeuronNum()
            scalar_outputs[f"dup_neurons_layer_{idx}"] = dup_neurons
            scalar_outputs[f"del_neurons_layer_{idx}"] = del_neurons

            z = self._layer_norm(z)

        # scalar_outputs = self.forward_downstream_classification_model(
        #     inputs, labels, scalar_outputs=scalar_outputs
        # )

        return scalar_outputs

    def forward_downstream_classification_model(
        self, inputs, labels, scalar_outputs=None
    ):
        if scalar_outputs is None:
            scalar_outputs = {
                "classification_loss": torch.zeros(
                    1, device=torch.device(config.device)
                )
            }

        z = inputs["natrual_sample"]
        z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z)

        input_classification_model = []

        with torch.no_grad():
            for idx, layer in enumerate(self.layers):
                z = layer.mapping(layer(z))
                z = self._layer_norm(z)

                if idx >= 0:
                    input_classification_model.append(z)

        input_classification_model = torch.concat(input_classification_model, dim=-1)
        output = self.fc_out(input_classification_model.detach())
        output = output - torch.max(output, dim=-1, keepdim=True)[0]

        classification_loss = self.classification_loss(output, labels["class_labels"])
        classification_accuracy = get_accuracy(output.data, labels["class_labels"])

        scalar_outputs["output"] = output
        scalar_outputs["classification_loss"] += classification_loss
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
