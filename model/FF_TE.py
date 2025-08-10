"""
Author: Galazxhy galazxhy@163.com
Date: 2025-08-09 15:08:14
LastEditors: Galazxhy galazxhy@163.com
LastEditTime: 2025-08-09 15:08:15
FilePath: /SORBF/model/FF_TE.py
Description:

Copyright (c) 2025 by Astroyd, All Rights Reserved.
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
import random


class FF_TE(torch.utils.data.Dataset):
    def __init__(self, partition, num_classes=4):
        df = pd.read_csv("./data/TE/TEP.csv", header=None)
        df.iloc[df.iloc[:, 9] >= 0.612558, 9] = 3
        df.iloc[(df.iloc[:, 9] < 0.612558) & (df.iloc[:, 9] >= 0.580210), 9] = 2
        df.iloc[(df.iloc[:, 9] < 0.580210) & (df.iloc[:, 9] >= 0.555355), 9] = 1
        df.iloc[(df.iloc[:, 9] < 0.555355), 9] = 0

        scalar = StandardScaler()
        df.iloc[:, :-1] = scalar.fit_transform(df.iloc[:, :-1])
        npdata = df.values.astype(np.float32)
        TE_dataset = torch.utils.data.TensorDataset(
            torch.tensor(npdata[:, :-1]), torch.tensor(npdata[:, -1]).to(torch.int64)
        )
        idx = [i for i in range(len(TE_dataset))]
        random.shuffle(idx)
        if partition == "train":
            self.TE = torch.utils.data.Subset(TE_dataset, idx[:1200])
        elif partition == "val":
            self.TE = torch.utils.data.Subset(TE_dataset, idx[1200:1600])
        elif partition == "test":
            self.TE = torch.utils.data.Subset(TE_dataset, idx[1600:])

        self.num_classes = num_classes
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes

    def __getitem__(self, index):
        pos_sample, neg_sample, neutral_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_sample": pos_sample,
            "neg_sample": neg_sample,
            "natrual_sample": neutral_sample,
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.TE)

    def _get_pos_sample(self, sample, class_label):
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label), num_classes=self.num_classes
        )
        pos_sample = sample.clone()
        pos_sample = torch.cat(
            [one_hot_label.unsqueeze(0), pos_sample.unsqueeze(0)],
            dim=1,
        )
        # pos_sample[:, 0, : self.num_classes] = one_hot_label
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        # Create randomly sampled one-hot label.
        classes = list(range(self.num_classes))
        classes.remove(class_label)  # Remove true label from possible choices.
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label), num_classes=self.num_classes
        )
        neg_sample = sample.clone()
        # neg_sample = torch.cat(
        #     [one_hot_label.unsqueeze(0), neg_sample.reshape(neg_sample.shape[0], -1)],
        #     dim=1,
        # )
        neg_sample = torch.cat(
            [one_hot_label.unsqueeze(0), neg_sample.unsqueeze(0)],
            dim=1,
        )
        # neg_sample[:, 0, : self.num_classes] = one_hot_label
        return neg_sample

    def _get_neutral_sample(self, z):
        # z = torch.cat(
        #     [self.uniform_label.unsqueeze(0), z.reshape(z.shape[0], -1)], dim=1
        # )
        z = torch.cat([self.uniform_label.unsqueeze(0), z.unsqueeze(0)], dim=1)
        # z[:, 0, : self.num_classes] = self.uniform_label
        return z

    def _generate_sample(self, index):
        # Get MNIST sample.
        sample, class_label = self.TE[index]
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        return pos_sample, neg_sample, neutral_sample, class_label
