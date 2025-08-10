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

epoch = 1200
batch_size = 16


init_num_centers = [13, 1200]

out_dim = 4

lr = 1e-5
wd = 3e-3
momentum = 0.95

downstream_lr = 6e-3
downstream_wd = 3e-4

val_idx = 50
