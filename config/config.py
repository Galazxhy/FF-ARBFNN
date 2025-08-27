"""
Author: Galazxhy galazxhy@163.com
Date: 2025-08-26 12:54:49
LastEditors: Galazxhy galazxhy@163.com
LastEditTime: 2025-08-26 12:54:50
FilePath: /SORBF/config/config.py
Description:

Copyright (c) 2025 by Astroyd, All Rights Reserved.
"""

device = "cuda:1"
# seed = [42, 15, 263, 745, 32, 85, 52, 63, 37, 73]
seed = 42
epoch = 500
batch_size = 128
init_num_centers = [784, 500]
theta = 65
max_neurons = 3000

out_dim = 10

lr = 5e-3
wd = 1e-5
momentum = 0.9

downstream_lr = 5e-4
downstream_wd = 1e-5

val_idx = -1
