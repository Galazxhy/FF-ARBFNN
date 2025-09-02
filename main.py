"""
Author: Galazxhy galazxhy@163.com
Date: 2025-08-09 15:06:59
LastEditors: Galazxhy galazxhy@163.com
LastEditTime: 2025-08-09 15:06:59
FilePath: /SORBF/main.py
Description:

Copyright (c) 2025 by Astroyd, All Rights Reserved.
"""

import time
from collections import defaultdict

import os
import torch
from config import config
import random
import numpy as np
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from model import FF_RBF
from model import utils
from torchviz import make_dot


if not os.path.exists("./Log"):
    os.mkdir("./Log")
i = 0
while True:
    if not os.path.exists("./Log/exp_" + str(i)):
        config.path = "./Log/exp_" + str(i)
        writer = SummaryWriter("./Log/exp_" + str(i))
        break
    i = i + 1


def train(model, optimizer):
    start_time = time.time()
    train_loader = utils.get_data("train")
    num_steps_per_epoch = len(train_loader)

    for epoch in range(config.epoch):
        train_results = defaultdict(float)
        optimizer[0] = utils.update_learning_rate(optimizer[0], epoch, config.lr)

        for inputs, labels in train_loader:
            inputs, labels = utils.preprocess_inputs(inputs, labels)

            optimizer[0].zero_grad()

            scalar_outputs = model(inputs, labels)
            # g = make_dot(scalar_outputs["Loss"])
            # g.render(filename="Full")
            scalar_outputs["Loss"].backward()

            optimizer[0].step()

            if model.organized == True:
                _, optimizer = utils.get_optimizer(model)

            train_results = utils.log_results(
                train_results, scalar_outputs, num_steps_per_epoch
            )

        utils.print_results(
            "train", time.time() - start_time, train_results, writer, epoch
        )

        start_time = time.time()

        if epoch % config.val_idx == 0 and config.val_idx != -1:
            validate_or_test(model, "val", epoch=epoch)

        # for epoch in range(config.downstream_epoch):
        train_results = defaultdict(float)

        for inputs, labels in train_loader:
            inputs, labels = utils.preprocess_inputs(inputs, labels)

            optimizer[1].zero_grad()

            scalar_outputs = model.forward_downstream_classification_model(
                inputs, labels
            )
            # g = make_dot(scalar_outputs["Loss"])
            # g.render(filename="Full")
            scalar_outputs["classification_loss"].backward()

            optimizer[1].step()

            train_results = utils.log_results(
                train_results, scalar_outputs, num_steps_per_epoch
            )

        utils.print_results(
            "train", time.time() - start_time, train_results, writer, epoch
        )

        start_time = time.time()

        if epoch % config.val_idx == 0 and config.val_idx != -1:
            validate_or_test(model, "val", epoch=epoch)

    return model


def validate_or_test(model, partition, epoch=None):
    test_time = time.time()
    test_results = defaultdict(float)

    data_loader = utils.get_data(partition)
    num_steps_per_epoch = len(data_loader)

    model.eval()
    print(partition)
    outAll, yAll = None, None
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = utils.preprocess_inputs(inputs, labels)

            scalar_outputs = model.forward_downstream_classification_model(
                inputs, labels
            )

            outAll, yAll = utils.ts_append(
                outAll, scalar_outputs["output"]
            ), utils.ts_append(yAll, labels["class_labels"])

            test_results = utils.log_results(
                test_results, scalar_outputs, num_steps_per_epoch
            )
        test_acc, test_f1_mac, test_f1_mic, test_auc = utils.valid_no_model(
            outAll.argmax(1).cpu().numpy(), yAll.cpu().numpy()
        )
        print(
            "test_acc:",
            test_acc,
            "test_f1_mac:",
            test_f1_mac,
            "test_f1_mic:",
            test_f1_mic,
            "test_auc:",
            test_auc,
        )
    utils.print_results(
        partition, time.time() - test_time, test_results, writer, epoch=epoch
    )

    model.train()


def run():
    # np.random.seed(config.seed)
    # torch.manual_seed(config.seed)
    # random.seed(config.seed)

    model = FF_RBF.FF_RBF(out_features=config.out_dim)
    model, optimizer = utils.get_optimizer(model)
    model = train(model, optimizer)
    validate_or_test(model, "val")
    validate_or_test(model, "test")
    torch.save(model, config.path + "/output_model.pth")


if __name__ == "__main__":
    run()
