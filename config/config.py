"""
Author: Galazxhy galazxhy@163.com
Date: 2025-08-09 12:22:29
LastEditors: Galazxhy galazxhy@163.com
LastEditTime: 2025-08-09 12:22:30
FilePath: /SORBF/config/config.py
Description

Copyright (c) 2025 by Astroyd, All Rights Reserved.
"""

device = "cuda"
# seed = [42, 15, 263, 745, 32, 85, 52, 63, 37, 73]
seed = 42
epoch = 2000
batch_size = 16


init_num_centers = [13, 2000]

out_dim = 4

lr = 1e-5
wd = 1e-4
momentum = 0.9

downstream_lr = 1e-2
downstream_wd = 1e-3

val_idx = -1
