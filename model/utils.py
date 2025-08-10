"""
Author: Galazxhy galazxhy@163.com
Date: 2025-08-09 14:54:38
LastEditors: Galazxhy galazxhy@163.com
LastEditTime: 2025-08-09 14:54:38
FilePath: /SORBF/model/utils.py
Description:

Copyright (c) 2025 by Astroyd, All Rights Reserved.
"""

from datetime import timedelta

import torch

from model.FF_TE import FF_TE
from config import config


def get_accuracy(output, target):
    """Computes the accuracy."""
    with torch.no_grad():
        prediction = torch.argmax(output, dim=1)
        return (prediction == target).sum() / config.batch_size


def print_results(partition, iteration_time, scalar_outputs, epoch=None):
    if epoch is not None:
        print(f"Epoch {epoch} \t", end="")

    print(
        f"{partition} \t \t" f"Time: {timedelta(seconds=iteration_time)} \t",
        end="",
    )
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            print(f"{key}: {value:.4f} \t", end="")
    print()


def log_results(result_dict, scalar_outputs, num_steps):
    for key, value in scalar_outputs.items():
        if isinstance(value, float):
            result_dict[key] += value / num_steps
        else:
            result_dict[key] += value.item() / num_steps
    return result_dict


def dict_to_cuda(dict):
    for key, value in dict.items():
        dict[key] = value.cuda(non_blocking=True)
    return dict


def preprocess_inputs(inputs, labels):
    if "cuda" in config.device:
        inputs = dict_to_cuda(inputs)
        labels = dict_to_cuda(labels)
    return inputs, labels


def get_linear_cooldown_lr(epoch, lr):
    if epoch > (config.epoch // 2):
        return lr * 2 * (1 + config.epoch - epoch) / config.epoch
    else:
        return lr


def update_learning_rate(optimizer, epoch):
    optimizer.param_groups[0]["lr"] = get_linear_cooldown_lr(epoch, config.lr)
    optimizer.param_groups[1]["lr"] = get_linear_cooldown_lr(epoch, config.lr)
    return optimizer


def get_data(partition):
    # dataset = ff_mnist.FF_MNIST(opt, partition)
    dataset = FF_TE(partition)

    # Improve reproducibility in dataloader.
    g = torch.Generator()

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        drop_last=True,
        shuffle=True,
        generator=g,
        num_workers=4,
        persistent_workers=True,
    )


def get_optimizer(model):
    if "cuda" in config.device:
        model = model.cuda()
    # print(model, "\n")

    # Create optimizer with different hyper-parameters for the main model
    # and the downstream classification model.
    main_model_params = [
        p
        for p in model.parameters()
        if all(p is not x for x in [model.classification_weight])
    ]
    optimizer = torch.optim.SGD(
        [
            {
                "params": main_model_params,
                "lr": config.lr,
                "weight_decay": config.wd,
                "momentum": config.momentum,
            },
            {
                "params": [model.classification_weight],
                "lr": config.downstream_lr,
                "weight_decay": config.downstream_wd,
                "momentum": config.momentum,
            },
        ]
    )
    return model, optimizer
