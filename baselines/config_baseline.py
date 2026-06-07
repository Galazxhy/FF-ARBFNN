"""Shared config for baseline models."""
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dataset = "TE"  # TE / MNIST / ZINC
seed = 42

# Training
epoch = 1000
batch_size = 210
val_idx = -1  # validate every N epochs, -1 = only at end

# Architecture (FNN-based for most baselines)
hidden_dims = [800, 800]  # hidden layer sizes
input_dim = None  # set automatically from dataset
num_classes = None  # set automatically from dataset

# Optimizer
lr = 1e-3
wd = 1e-5
momentum = 0.9
downstream_lr = 1e-3
downstream_wd = 1e-5

# FF-specific
theta = 200  # goodness threshold (for FF and SymBa)

# CaFo-specific
cafo_alpha = 0.5  # weight for cascaded loss

# InfoPro-specific
infopro_beta = 0.1  # information propagation weight

# PCX-specific
pcx_inference_steps = 10  # PC inference iterations
pcx_learning_rate_pc = 1e-4  # weight update rate during inference

# DTP-specific
dtp_feedback_lr = 1e-3  # feedback network learning rate

path = None
