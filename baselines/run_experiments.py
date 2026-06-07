"""Grid search + final training for all baselines on TE, MNIST, Zinc.
Usage:
    python -m baselines.run_experiments --dataset TE --grid_search
    python -m baselines.run_experiments --dataset TE --final
    python -m baselines.run_experiments --all  # all datasets, grid + final
"""
import os
import sys
import time
import json
import itertools
from collections import defaultdict

import torch
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.TE import FF_TE, FF_MNIST, FF_ZINC
from baselines.ff_symba import FFModel
from baselines.cafo import CaFo
from baselines.infopro import InfoPro
from baselines.pcx import PCXModel
from baselines.dtp import DTPModel

# ============================================================
# Config
# ============================================================
DATASETS = {
    # input_dim = raw_features + num_classes (one-hot label concatenated)
    # TE: 9 features + 4 one-hot = 13
    # MNIST: 784 pixels + 10 one-hot (in first row) = still 784 (in-place embedding)
    # ZINC: 13 features + 4 one-hot = 17
    "TE": {"class": FF_TE, "input_dim": 13, "num_classes": 4, "batch_size": 210},
    "MNIST": {"class": FF_MNIST, "input_dim": 784, "num_classes": 10, "batch_size": 210},
    "ZINC": {"class": FF_ZINC, "input_dim": 17, "num_classes": 4, "batch_size": 210},
}

# Grid search space
GRID = {
    "hidden_dims": [[400, 200], [800, 400], [800, 800]],
    "lr": [1e-3, 5e-4, 1e-4],
}
FF_GRID_EXTRA = {"theta": [50, 200, 500]}  # for ff_original only

GRID_EPOCH = 50
FINAL_EPOCH = 200
RESULTS_DIR = "./results"


def setup(config_dict):
    """Apply config to global state."""
    import config.config as orig_config
    for k, v in config_dict.items():
        setattr(orig_config, k, v)


def get_dataloaders(ds_name, batch_size):
    ds_info = DATASETS[ds_name]
    setup({"dataset": ds_name, "batch_size": batch_size, "device": "cuda:0" if torch.cuda.is_available() else "cpu"})
    import config.config as orig_config

    def _loader(partition):
        dataset = ds_info["class"](partition)
        g = torch.Generator()
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, generator=g, num_workers=0,
        )
    return _loader("train"), _loader("val"), _loader("test")


def create_model(model_name, input_dim, num_classes, hidden_dims, **kwargs):
    if model_name in ("ff_original", "ff_symba"):
        loss_type = "original" if model_name == "ff_original" else "symba"
        return FFModel(input_dim, num_classes, hidden_dims, loss_type=loss_type, **kwargs)
    elif model_name == "cafo":
        return CaFo(input_dim, num_classes, hidden_dims)
    elif model_name == "infopro":
        return InfoPro(input_dim, num_classes, hidden_dims)
    elif model_name == "pcx":
        return PCXModel(input_dim, num_classes, hidden_dims)
    elif model_name == "dtp":
        return DTPModel(input_dim, num_classes, hidden_dims)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def validate(model, loader, device):
    model.eval()
    outAll, yAll = None, None
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            labels = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in labels.items()}
            out = model.forward_downstream(inputs, labels)
            outAll = torch.cat([outAll, out["output"]], dim=0) if outAll is not None else out["output"]
            yAll = torch.cat([yAll, labels["class_labels"]], dim=0) if yAll is not None else labels["class_labels"]
    model.train()
    preds = outAll.argmax(1).cpu().numpy()
    ytrue = yAll.cpu().numpy()
    from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score
    acc = (preds == ytrue).mean()
    f1 = f1_score(ytrue, preds, average="macro")
    kappa = cohen_kappa_score(ytrue, preds)
    try:
        probs = torch.softmax(outAll, dim=1).cpu().numpy()
        auc = roc_auc_score(ytrue, probs, multi_class="ovo")
    except:
        auc = 0.5
    return acc, f1, kappa, auc


def preprocess(inputs, labels, device):
    result_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            result_inputs[k] = v.to(device)
        else:
            result_inputs[k] = v
    result_labels = {}
    for k, v in labels.items():
        if isinstance(v, torch.Tensor):
            result_labels[k] = v.to(device)
        else:
            result_labels[k] = v
    return result_inputs, result_labels


def train_one_epoch(model, optimizers, train_loader, device):
    """One epoch: local learning + task head training."""
    for inputs, labels in train_loader:
        inputs, labels = preprocess(inputs, labels, device)
        # Phase 1: local learning
        optimizers[0].zero_grad()
        out = model(inputs, labels)
        out["Loss"].backward()
        optimizers[0].step()
        # Phase 2: task head
        optimizers[1].zero_grad()
        out2 = model.forward_downstream(inputs, labels)
        out2["classification_loss"].backward()
        optimizers[1].step()


def train_model(model, train_loader, val_loader, test_loader, device, epochs, lr=1e-3, wd=1e-5, d_lr=1e-3, d_wd=1e-5):
    main_params = [p for p in model.parameters() if all(p is not x for x in model.fc_out.parameters())]
    optimizers = [
        torch.optim.AdamW(main_params, lr=lr, weight_decay=wd),
        torch.optim.AdamW(model.fc_out.parameters(), lr=d_lr, weight_decay=d_wd),
    ]
    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        train_one_epoch(model, optimizers, train_loader, device)
        if epoch % 10 == 0 or epoch == epochs - 1:
            val_acc, f1, kappa, auc = validate(model, val_loader, device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc, test_f1, test_kappa, test_auc = validate(model, test_loader, device)
    return {"acc": test_acc, "f1": test_f1, "kappa": test_kappa, "auc": test_auc, "best_val_acc": best_val_acc}


def grid_search(ds_name, model_name):
    """Run grid search for one model on one dataset."""
    ds = DATASETS[ds_name]
    train_loader, val_loader, test_loader = get_dataloaders(ds_name, ds["batch_size"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build configs
    configs = []
    for hd in GRID["hidden_dims"]:
        for lr in GRID["lr"]:
            cfg = {"hidden_dims": hd, "lr": lr}
            if model_name == "ff_original":
                for theta in FF_GRID_EXTRA["theta"]:
                    configs.append({**cfg, "theta": theta})
            else:
                configs.append(cfg)

    print(f"\n{'='*60}")
    print(f"Grid Search: {model_name} on {ds_name} ({len(configs)} configs)")
    print(f"{'='*60}")

    best_cfg, best_acc = None, 0.0
    for i, cfg in enumerate(configs):
        torch.manual_seed(42); np.random.seed(42); random.seed(42)
        model = create_model(model_name, ds["input_dim"], ds["num_classes"], cfg["hidden_dims"],
                             theta=cfg.get("theta", 200)).to(device)
        results = train_model(model, train_loader, val_loader, test_loader, device,
                              GRID_EPOCH, lr=cfg["lr"], d_lr=cfg["lr"])
        print(f"  [{i+1}/{len(configs)}] {cfg} -> test acc={results['acc']:.4f}, f1={results['f1']:.4f}")
        if results["best_val_acc"] > best_acc:
            best_acc = results["best_val_acc"]
            best_cfg = cfg

    print(f"  BEST: {best_cfg} (val_acc={best_acc:.4f})")
    return best_cfg


def final_training(ds_name, model_name, best_cfg, n_repeats=5):
    """Final training with best config, repeated n_repeats times."""
    ds = DATASETS[ds_name]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    all_results = []
    print(f"\n  Final training: {model_name} on {ds_name} ({n_repeats} repeats, {FINAL_EPOCH} epochs)")

    for seed in range(n_repeats):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        train_loader, val_loader, test_loader = get_dataloaders(ds_name, ds["batch_size"])
        model = create_model(model_name, ds["input_dim"], ds["num_classes"], best_cfg["hidden_dims"],
                             theta=best_cfg.get("theta", 200)).to(device)
        results = train_model(model, train_loader, val_loader, test_loader, device,
                              FINAL_EPOCH, lr=best_cfg["lr"], d_lr=best_cfg["lr"])
        all_results.append(results)
        print(f"    seed={seed}: acc={results['acc']:.4f}, f1={results['f1']:.4f}, "
              f"kappa={results['kappa']:.4f}, auc={results['auc']:.4f}")

    # Aggregate
    accs = [r["acc"] for r in all_results]
    f1s = [r["f1"] for r in all_results]
    kappas = [r["kappa"] for r in all_results]
    aucs = [r["auc"] for r in all_results]

    summary = {
        "model": model_name, "dataset": ds_name,
        "best_cfg": best_cfg, "n_repeats": n_repeats, "epochs": FINAL_EPOCH,
        "acc_mean": np.mean(accs), "acc_std": np.std(accs),
        "f1_mean": np.mean(f1s), "f1_std": np.std(f1s),
        "kappa_mean": np.mean(kappas), "kappa_std": np.std(kappas),
        "auc_mean": np.mean(aucs), "auc_std": np.std(aucs),
    }
    return summary


def run_all(grid_search_only=False):
    models = ["ff_original", "ff_symba", "cafo", "infopro", "pcx", "dtp"]
    datasets = ["TE", "MNIST", "ZINC"]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_summaries = []

    for ds_name in datasets:
        for model_name in models:
            # Grid search
            best_cfg = grid_search(ds_name, model_name)

            if grid_search_only:
                all_summaries.append({"model": model_name, "dataset": ds_name, "best_cfg": best_cfg})
                continue

            # Final training
            summary = final_training(ds_name, model_name, best_cfg)
            all_summaries.append(summary)

            # Save incrementally
            with open(os.path.join(RESULTS_DIR, f"{ds_name}_{model_name}.json"), "w") as f:
                json.dump(summary, f, indent=2, default=str)

    # Save all
    with open(os.path.join(RESULTS_DIR, "all_results.json"), "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)

    # Print summary table
    print(f"\n\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    for ds_name in datasets:
        print(f"\n--- {ds_name} ---")
        print(f"{'Model':<15} {'Acc':>8} {'F1':>8} {'Kappa':>8} {'AUC':>8}")
        print("-" * 50)
        for s in all_summaries:
            if s["dataset"] == ds_name:
                if "acc_mean" in s:
                    print(f"{s['model']:<15} {s['acc_mean']*100:>7.2f}±{s['acc_std']*100:.2f} "
                          f"{s['f1_mean']*100:>7.2f} {s['kappa_mean']*100:>7.2f} {s['auc_mean']*100:>7.2f}")
                else:
                    print(f"{s['model']:<15} (grid search only)")
    return all_summaries


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, choices=["TE", "MNIST", "ZINC"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--grid_search", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    models = [args.model] if args.model else ["ff_original", "ff_symba", "cafo", "infopro", "pcx", "dtp"]
    datasets = [args.dataset] if args.dataset else ["TE", "MNIST", "ZINC"]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.all:
        run_all(grid_search_only=False)
    else:
        for ds_name in datasets:
            for model_name in models:
                if args.grid_search:
                    best_cfg = grid_search(ds_name, model_name)
                    json.dump({"model": model_name, "dataset": ds_name, "best_cfg": best_cfg},
                              open(os.path.join(RESULTS_DIR, f"{ds_name}_{model_name}_cfg.json"), "w"),
                              indent=2, default=str)
                if args.final:
                    cfg_file = os.path.join(RESULTS_DIR, f"{ds_name}_{model_name}_cfg.json")
                    if os.path.exists(cfg_file):
                        best_cfg = json.load(open(cfg_file))["best_cfg"]
                    else:
                        best_cfg = {"hidden_dims": [800, 400], "lr": 1e-3}
                    summary = final_training(ds_name, model_name, best_cfg)
                    json.dump(summary, open(os.path.join(RESULTS_DIR, f"{ds_name}_{model_name}.json"), "w"),
                              indent=2, default=str)
