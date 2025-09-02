"""
Author: Galazxhy galazxhy@163.com
Date: 2025-08-27 18:02:43
LastEditors: Galazxhy galazxhy@163.com
LastEditTime: 2025-08-27 18:02:43
FilePath: /SORBF/model/layer.py
Description:

Copyright (c) 2025 by Astroyd, All Rights Reserved.
"""

"""
Author: Galazxhy galazxhy@163.com
Date: 2025-08-26 12:46:13
LastEditors: Galazxhy galazxhy@163.com
LastEditTime: 2025-08-26 12:46:14
FilePath: /SORBF/model/layer.py
Description:

Copyright (c) 2025 by Astroyd, All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from model import utils


class RBFLayer(nn.Module):
    def __init__(self, num_centers, dim_centers, out_features):
        super(RBFLayer, self).__init__()
        """
        :param centers: shape=[center_num,data_dim]
        :param n_out:
        """
        self.num_centers = num_centers
        self.centers = nn.Parameter(torch.randn((num_centers, dim_centers)))
        self.width = nn.Parameter(torch.randn(num_centers))
        self.outweight = nn.Parameter(torch.randn((num_centers, out_features)))
        self.bias = nn.Parameter(torch.rand(1, out_features))

    def kernel_fun(self, x):
        # print(x.shape, self.centers.shape, self.width.shape)
        centers = self.centers.repeat(
            x.shape[0], 1, 1
        )  # (batch_size, num_centers, num_features)
        widths = self.width.unsqueeze(0).repeat(
            x.shape[0], 1
        )  # (batch_size, num_centers)
        input_g = x.unsqueeze(1).repeat(
            1, self.centers.shape[0], 1
        )  # (batch_size, num_centers, num_features)
        distance_g = ((centers - input_g).pow(2)).mean(
            2, keepdim=False
        )  # (batch_size, num_centers)
        result_g = torch.exp(-distance_g / (2 * widths.pow(2)))
        # result_g = widths.pow(2) / distance_g
        # (batch_size, num_centers)

        return result_g

    def addNeurons(self, indices):
        best_centers = nn.Parameter(
            torch.index_select(self.centers.clone().detach(), 0, indices)
            + 0.1 * torch.randn((len(indices), 1)).to(torch.device(config.device))
        )
        best_width = nn.Parameter(
            torch.index_select(self.width.clone().detach(), 0, indices)
        )
        if indices.shape[0] != 0 and self.centers.shape[0] < 2 * self.num_centers:
            self.centers = nn.Parameter(
                torch.cat(
                    [self.centers.detach(), best_centers],
                    dim=0,
                )
            )
            self.width = nn.Parameter(
                torch.cat([self.width.detach(), best_width], dim=0)
            )
            return True
        return False

    def delNeurons(self, indices):
        if indices.shape[0] != 0:
            mask = torch.ones(self.centers.size(0), dtype=torch.bool)
            mask[indices] = False
            self.centers = nn.Parameter(self.centers.detach()[mask])
            self.width = nn.Parameter(self.width.detach()[mask])
            return True
        return False

    def addWeight(self, indices):
        best_weights = nn.Parameter(
            torch.index_select(self.outweight.detach(), 0, indices)
        )
        if (
            best_weights.shape[0] != 0
            and self.outweight.shape[0] < 2 * self.num_centers
        ):
            self.outweight = nn.Parameter(
                torch.cat(
                    [self.outweight.detach(), best_weights],
                    dim=0,
                )
            )
            return True
        return False

    def delWeight(self, indices):
        if indices.shape[0] != 0:
            mask = torch.ones(self.outweight.size(0), dtype=torch.bool)
            mask[indices] = False
            self.outweight = nn.Parameter(self.outweight.detach()[mask])
            return True
        return False

    def adjustStruc(self, z):
        z_pos = z[: config.batch_size]
        z_neg = z[config.batch_size :]
        organized = False
        g_pos = torch.sum(z_pos, dim=0) / self.centers.shape[0] * config.batch_size
        g_neg = torch.sum(z_neg, dim=0) / self.centers.shape[0] * config.batch_size

        lgt_pos = g_pos - config.theta
        lgt_neg = g_neg - config.theta

        # print(F.softmax(lgt_pos, dim=0), F.softmax(lgt_neg, dim=0))
        add_pos_mask = F.gumbel_softmax(lgt_pos, dim=0, tau=1.2) > (
            5 * (1 / self.centers.shape[0])
        )
        add_neg_mask = F.gumbel_softmax(lgt_neg, dim=0, tau=1.2) < (
            0.02 * (1 / self.centers.shape[0])
        )

        add_mask = add_pos_mask & add_neg_mask
        dup_neurons = add_mask.sum()

        self.addNeurons(torch.where(add_mask)[0])
        added = self.addWeight(torch.where(add_mask)[0])
        organized = organized or added

        del_pos_mask = F.gumbel_softmax(lgt_pos, dim=0, tau=1.2) < (
            0.02 * (1 / self.centers.shape[0])
        )
        del_neg_mask = F.gumbel_softmax(lgt_neg, dim=0, tau=1.2) > (
            5 * (1 / self.centers.shape[0])
        )

        del_mask = del_pos_mask & del_neg_mask
        del_neurons = del_mask.sum()

        self.delNeurons(torch.where(del_mask)[0])
        deled = self.delWeight(torch.where(del_mask)[0])

        organized = organized or deled
        return organized, dup_neurons, del_neurons

    def getNeuronNum(self):
        return torch.tensor(self.centers.shape[0])

    def forward(self, x):
        h = self.kernel_fun(x)
        return h

    def mapping(self, h):
        h = torch.mm(h, self.outweight) + self.bias
        return h
