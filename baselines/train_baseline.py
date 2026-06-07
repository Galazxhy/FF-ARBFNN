"""Unified training script for all local learning baselines.
Usage:
    python -m baselines.train_baseline --model ff_symba --dataset TE
    python -m baselines.train_baseline --model cafo --dataset MNIST
    python -m baselines.train_baseline --model infopro --dataset TE
    python -m baselines.train_baseline --model pcx --dataset TE
    python -m baselines.train_baseline --model dtp --dataset TE
"""
import os
import sys
import time
import argparse
from collections import defaultdict

import torch
import random
import numpy as np

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import utils
from model.TE import FF_TE, FF_MNIST
from baselines.ff_symba import FFModel
from baselines.cafo import CaFo
from baselines.infopro import InfoPro
from baselines.pcx import PCXModel
from baselines.dtp import DTPModel

# Datasets config
DATASET_CONFIG = {
    "TE": {"input_dim": 13, "num_classes": 4, "batch_size": 210},
    "MNIST": {"input_dim": 28 * 28, "num_classes": 10, "batch_size": 210},
}

MODEL_FACTORY = {
    "ff_original": lambda **kw: FFModel(**kw, loss_type="original"),
    "ff_symba": lambda **kw: FFModel(**kw, loss_type="symba"),
    "cafo": CaFo,
    "infopro": InfoPro,
    "pcx": PCXModel,
    "dtp": DTPModel,
}


def get_config(args):
    """Build config dict from args."""
    ds = DATASET_CONFIG[args.dataset]
    return {
        "dataset": args.dataset,
        "seed": args.seed,
        "epoch": args.epoch,
        "batch_size": args.batch_size or ds["batch_size"],
        "input_dim": ds["input_dim"],
        "num_classes": ds["num_classes"],
        "hidden_dims": [int(x) for x in args.hidden_dims.split(",")],
        "lr": args.lr,
        "wd": args.wd,
        "momentum": args.momentum,
        "downstream_lr": args.downstream_lr,
        "downstream_wd": args.downstream_wd,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "out_dim": ds["num_classes"],
        "path": None,
    }


def setup_logging():
    if not os.path.exists("./Log"):
        os.mkdir("./Log")
    i = 0
    while True:
        log_dir = f"./Log/baseline_{i}"
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
            return log_dir
        i += 1


def train_ff_like(model, optimizer, cfg, train_loader, val_loader, test_loader, log_dir):
    """Training for FF-family models (FF_SymBa, CaFo, InfoPro, PCX, DTP).
    Each has its own forward() for local training, then uses
    forward_downstream() for task head training.
    """
    num_steps = len(train_loader)

    for epoch in range(cfg["epoch"]):
        train_results = defaultdict(float)
        t0 = time.time()

        # Phase 1: Local learning
        for inputs, labels in train_loader:
            inputs, labels = utils.preprocess_inputs(inputs, labels)
            optimizer[0].zero_grad()

            scalar_outputs = model(inputs, labels)
            scalar_outputs["Loss"].backward()
            optimizer[0].step()

            train_results = utils.log_results(
                train_results, scalar_outputs, num_steps
            )

        # Phase 2: Task head training (same epoch loop as original FF-ARBFNN)
        for inputs, labels in train_loader:
            inputs, labels = utils.preprocess_inputs(inputs, labels)
            optimizer[1].zero_grad()

            scalar_outputs = model.forward_downstream(inputs, labels)
            scalar_outputs["classification_loss"].backward()
            optimizer[1].step()

            train_results = utils.log_results(
                train_results, scalar_outputs, num_steps
            )

        # Log per epoch
        print(f"Epoch {epoch} \t Time: {time.time() - t0:.2f}s \t", end="")
        for key, value in train_results.items():
            if "loss" in key.lower() or "accuracy" in key.lower():
                print(f"{key}: {value:.4f} \t", end="")
        print()

        # Validation
        if epoch % 10 == 0 or epoch == cfg["epoch"] - 1:
            validate_or_test(model, "val", val_loader, cfg, epoch)
            model.train()

    return model


def validate_or_test(model, partition, data_loader, cfg, epoch=None):
    """Validate or test the model."""
    t0 = time.time()
    model.eval()

    outAll, yAll = None, None
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = utils.preprocess_inputs(inputs, labels)
            scalar_outputs = model.forward_downstream(inputs, labels)

            outAll = utils.ts_append(outAll, scalar_outputs["output"])
            yAll = utils.ts_append(yAll, labels["class_labels"])

    test_acc, test_f1_mac, test_kappa, test_auc = utils.valid_no_model(
        outAll.argmax(1).cpu().numpy(), yAll.cpu().numpy()
    )
    print(
        f"  [{partition}] acc: {test_acc:.4f} | f1_mac: {test_f1_mac:.4f} | "
        f"kappa: {test_kappa:.4f} | auc: {test_auc:.4f} | time: {time.time() - t0:.2f}s"
    )
    model.train()
    return test_acc, test_f1_mac, test_kappa, test_auc


def run(args):
    cfg = get_config(args)
    log_dir = setup_logging()
    print(f"Logging to: {log_dir}")
    print(f"Config: {cfg}")

    # Seed
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])

    # Create model
    ModelClass = MODEL_FACTORY[args.model]
    model = ModelClass(
        input_dim=cfg["input_dim"],
        num_classes=cfg["num_classes"],
        hidden_dims=cfg["hidden_dims"],
    )
    model = model.to(cfg["device"])
    print(f"Model: {args.model}, Params: {sum(p.numel() for p in model.parameters())}")

    # Create data loaders using existing FF_TE / FF_MNIST
    import config.config as orig_config
    orig_config.dataset = cfg["dataset"]
    orig_config.batch_size = cfg["batch_size"]
    orig_config.device = cfg["device"]

    # Override get_data to use num_workers=0 (Windows compatibility)
    from model.TE import FF_TE, FF_MNIST
    _DS_CLASSES = {"TE": FF_TE, "MNIST": FF_MNIST}

    def _get_data_safe(partition):
        dataset = _DS_CLASSES[orig_config.dataset](partition)
        g = torch.Generator()
        return torch.utils.data.DataLoader(
            dataset, batch_size=orig_config.batch_size,
            drop_last=True, shuffle=True, generator=g,
            num_workers=0,  # Windows fix
        )
    train_loader = _get_data_safe("train")
    val_loader = _get_data_safe("val")
    test_loader = _get_data_safe("test")

    # Optimizers
    main_params = [
        p for p in model.parameters()
        if all(p is not x for x in model.fc_out.parameters())
    ]
    optimizer = [
        torch.optim.AdamW(main_params, lr=cfg["lr"], weight_decay=cfg["wd"]),
        torch.optim.AdamW(
            model.fc_out.parameters(),
            lr=cfg["downstream_lr"],
            weight_decay=cfg["downstream_wd"],
        ),
    ]

    # Train
    model = train_ff_like(
        model, optimizer, cfg, train_loader, val_loader, test_loader, log_dir
    )

    # Final test
    print("\n=== Final Test ===")
    acc, f1, kappa, auc = validate_or_test(model, "test", test_loader, cfg)

    # Save
    torch.save(
        {"model": model.state_dict(), "results": (acc, f1, kappa, auc)},
        os.path.join(log_dir, "model.pth"),
    )
    return acc


def main():
    parser = argparse.ArgumentParser(description="Train local learning baselines")
    parser.add_argument("--model", type=str, required=True,
                        choices=["ff_original", "ff_symba", "cafo", "infopro", "pcx", "dtp"],
                        help="Baseline model to train")
    parser.add_argument("--dataset", type=str, default="TE",
                        choices=["TE", "MNIST"],
                        help="Dataset")
    parser.add_argument("--epoch", type=int, default=200,
                        help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size")
    parser.add_argument("--hidden_dims", type=str, default="800,800",
                        help="Hidden layer dims, comma-separated")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum")
    parser.add_argument("--downstream_lr", type=float, default=1e-3,
                        help="Downstream learning rate")
    parser.add_argument("--downstream_wd", type=float, default=1e-5,
                        help="Downstream weight decay")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
