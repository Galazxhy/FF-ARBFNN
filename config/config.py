"""
Author: Galazxhy galazxhy@163.com
Date: 2025-08-26 12:54:49
LastEditors: Galazxhy galazxhy@163.com
LastEditTime: 2025-08-26 12:54:50
FilePath: /SORBF/config/config.py
Description:

Copyright (c) 2025 by Astroyd, All Rights Reserved.
"""

device = "cuda:0"
# seed = [42, 15, 263, 745, 32, 85, 52, 63, 37, 73]
seed = 42
epoch = 3000
batch_size = 210
init_num_centers = [17, 800]
theta = 200
out_dim = 4

lr = 1e-3
wd = 1e-5
momentum = 0.9

downstream_lr = 1e-3
downstream_wd = 1e-5

val_idx = -1

path = None
