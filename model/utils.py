"""
Author: Galazxhy galazxhy@163.com
Date: 2025-08-27 18:13:31
LastEditors: Galazxhy galazxhy@163.com
LastEditTime: 2025-08-27 18:13:32
FilePath: /SORBF/model/utils.py
Description:

Copyright (c) 2025 by Astroyd, All Rights Reserved.
"""

from datetime import timedelta

import torch
from model.TE import FF_TE, FF_MNIST
from config import config

import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score


def get_accuracy(output, target):
    """Computes the accuracy."""
    with torch.no_grad():
        prediction = torch.argmax(output, dim=1)
        return (prediction == target).sum() / config.batch_size


def get_linear_cooldown_lr(epoch, lr):
    if epoch > (config.epoch // 2):
        return lr * 2 * (1 + config.epoch - epoch) / config.epoch
    else:
        return lr


def update_learning_rate(optimizer, epoch, lr):
    optimizer.param_groups[0]["lr"] = get_linear_cooldown_lr(epoch, lr)
    return optimizer


def print_results(partition, iteration_time, scalar_outputs, writer, epoch=None):
    if epoch is not None:
        print(f"Epoch {epoch} \t", end="")

    print(
        f"{partition} \t \t" f"Time: {timedelta(seconds=iteration_time)} \t",
        end="",
    )
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            (
                print(f"{key}: {value:.4f} \t", end="")
                if key != "output"
                else print("output")
            )
    print()

    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            (
                writer.add_scalar(partition + " " + key, value, epoch)
                if key != "output"
                else None
            )
        with open(config.path + "/output_log.txt", mode="a") as f:
            f.write("Epoch:" + str(epoch))
            f.write("\n")
            for key, value in scalar_outputs.items():
                if key != "output":
                    f.write(str(key) + ":" + str(value))
                    f.write("\n")
            f.write("\n")


def log_results(result_dict, scalar_outputs, num_steps):
    for key, value in scalar_outputs.items():
        if "num_neurons_layer" in key:
            result_dict[key] = value
        elif "dup_neurons" in key:
            result_dict[key] += value.item()
        elif "del_neurons" in key:
            result_dict[key] += value.item()
        elif "output" in key:
            result_dict[key] = value
        else:
            if isinstance(value, float):
                result_dict[key] += value / num_steps
            else:
                result_dict[key] += value.item() / num_steps
    return result_dict


def dict_to_cuda(dict):
    for key, value in dict.items():
        dict[key] = value.to(torch.device(config.device))
    return dict


def preprocess_inputs(inputs, labels):
    if "cuda" in config.device:
        inputs = dict_to_cuda(inputs)
        labels = dict_to_cuda(labels)
    return inputs, labels


def get_data(partition):
    dataset = globals()["FF_" + config.dataset](partition)

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
        model = model.to(config.device)
    # print(model, "\n")

    # Create optimizer with different hyper-parameters for the main model
    # and the downstream classification model.
    main_model_params = [
        p
        for p in model.parameters()
        if all(p is not x for x in model.fc_out.parameters())
    ]
    optimizer = [
        torch.optim.AdamW(
            [
                {
                    "params": main_model_params,
                    "lr": config.lr,
                    "weight_decay": config.wd,
                    "momentum": config.momentum,
                }
            ]
        ),
        torch.optim.AdamW(
            [
                {
                    "params": model.fc_out.parameters(),
                    "lr": config.downstream_lr,
                    "weight_decay": config.downstream_wd,
                    "momentum": config.momentum,
                }
            ]
        ),
    ]

    return model, optimizer


def ts_append(a, b):
    """List like 'Append' tool for tensor datatype
    ---
    Parameters:
        a, b: append a with b
    """
    if a is None:
        return b
    else:
        return torch.cat([a, b], dim=0)


def get_indices(output, y):
    with torch.no_grad():
        acc = sum(output == y) / (output.shape[0])
        f1_mac = f1_score(output, y, average="macro")
        f1_mic = cohen_kappa_score(output, y)
        auc_s = roc_auc_score(
            output, F.one_hot(torch.tensor(y)).numpy(), multi_class="ovo"
        )
        return (acc, f1_mac, f1_mic, auc_s)


def valid_no_model(output, y):
    acc = sum(output == y) / (output.shape[0])
    f1_mac = f1_score(output, y, average="macro")
    f1_mic = cohen_kappa_score(output, y)
    auc_s = roc_auc_score(output, F.one_hot(torch.tensor(y)).numpy(), multi_class="ovo")
    return (acc, f1_mac, f1_mic, auc_s)
