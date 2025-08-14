"""
Author: Galazxhy galazxhy@163.com
Date: 2025-08-09 11:32:29
LastEditors: Galazxhy galazxhy@163.com
LastEditTime: 2025-08-09 11:32:29
FilePath: /SORBF/model/layer.py
Description:

Copyright (c) 2025 by Astroyd, All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RBFLayer(nn.Module):
    def __init__(self, num_centers, dim_centers):
        super(RBFLayer, self).__init__()
        """
        :param centers: shape=[center_num,data_dim]
        :param n_out:
        """
        self.centers = nn.Parameter(torch.randn((num_centers, dim_centers)))
        self.width = nn.Parameter(torch.randn(num_centers))

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
        distance_g = ((centers - input_g).pow(2)).sum(
            2, keepdim=False
        )  # (batch_size, num_centers)
        result_g = torch.exp((-1 / (2 * widths.pow(2))).mul(distance_g))
        # result_g = widths.pow(2) / distance_g
        # (batch_size, num_centers)

        return result_g

    def forward(self, x):
        hidden = self.kernel_fun(x)
        return hidden

    def addNeurons(self, indices):
        best_centers = nn.Parameter(
            torch.index_select(self.centers.clone().detach(), 0, indices)
            + torch.rand((len(indices), 1)).cuda()
        )
        best_width = nn.Parameter(
            torch.index_select(self.width.clone().detach(), 0, indices)
        )
        if best_centers.shape[0] != 0:
            self.centers = nn.Parameter(
                torch.cat(
                    [self.centers.detach(), best_centers],
                    dim=0,
                )
            )
            self.width = nn.Parameter(
                torch.cat([self.width.detach(), best_width], dim=0)
            )

    def delNeurons(self, indices):
        if indices.shape[0] < self.centers.shape[0]:
            self.centers = nn.Parameter(
                torch.index_select(self.centers.detach(), 0, indices)
            )
            self.width = nn.Parameter(
                torch.index_select(self.width.detach(), 0, indices)
            )

    def getNeuronNum(self):
        return torch.tensor(self.centers.shape[0])
