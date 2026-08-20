"""
train.py — Training Loop (3 Experiments)
=========================================
Usage:
    python train.py --experiment 1   # Baseline: mean pool, frozen encoder
    python train.py --experiment 2   # Attention pool + partial unfreeze
    python train.py --experiment 3   # Exp2 + hard-negative oversampling data

Each experiment logs to checkpoints/exp{N}/ and saves:
    - best model weights (exp{N}_best.pt)
    - training curves (stats/plots/exp{N}_curves.png)
    - experiment results JSON (stats/exp{N}_results.json)
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm

from model import build_model

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
SPLITS_DIR = BASE_DIR / "splits"
CKPT_DIR = BASE_DIR / "checkpoints"
STATS_DIR = BASE_DIR / "stats"
PLOTS_DIR = STATS_DIR / "plots"
for d in [CKPT_DIR, STATS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────────

class TurnDataset(Dataset):
    """Loads pre-extracted features from splits/."""

    def __init__(self, split_name: str):
        split_dir = SPLITS_DIR / split_name
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split '{split_name}' not found at {split_dir}. "
                "Run: python data_prep.py --mode build_splits"
            )
        self.features = np.load(str(split_dir / "features.npy"), mmap_mode="r")
        self.meta = pd.read_parquet(str(split_dir / "metadata.parquet"))
        assert len(self.features) == len(self.meta), "Feature/metadata length mismatch"

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx].copy()).float()  # (80, 3000)
        y = torch.tensor(self.meta.iloc[idx]["label"], dtype=torch.float32)
        return x, y


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING UTILS
# ──────────────────────────────────────────────────────────────────────────────

def compute_pos_weight(dataset: TurnDataset, device: torch.device) -> torch.Tensor:
    labels = dataset.meta["label"].values
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(1.0).to(device)
    pw = n_neg / n_pos
    print(f"  pos_weight = {pw:.3f}  (neg={n_neg:,}, pos={n_pos:,})")
    return torch.tensor(pw, dtype=torch.float32).to(device)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    all_labels, all_probs, all_losses = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)
            probs = torch.sigmoid(logits)
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_losses.append(loss.item())

    labels = np.array(all_labels)
    probs = np.array(all_probs)
    preds = (probs >= 0.5).astype(int)

    auroc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    f1 = f1_score(labels, preds, zero_division=0)
    acc = accuracy_score(labels, preds)

    return {
        "loss":  float(np.mean(all_losses)),
        "auroc": float(auroc),
        "f1":    float(f1),
        "acc":   float(acc),
    }


def save_curves(train_losses, val_metrics_list, exp_id: int):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Experiment {exp_id} — Training Curves", fontsize=13)
    epochs = list(range(1, len(train_losses) + 1))

    axes[0].plot(epochs, train_losses, "b-o", label="train loss")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")

    axes[1].plot(epochs, [m["auroc"] for m in val_metrics_list], "g-o")
    axes[1].set_title("Val AUROC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)

    axes[2].plot(epochs, [m["f1"] for m in val_metrics_list], "r-o")
    axes[2].set_title("Val F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    path = PLOTS_DIR / f"exp{exp_id}_curves.png"
    plt.savefig(str(path), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Curves saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def train(
    experiment: int,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    patience: int = 3,
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"\n{'='*65}")
    print(f"EXPERIMENT {experiment} — Training on {device}")
    print(f"{'='*65}")
    print(f"  epochs={epochs}, batch_size={batch_size}, lr={lr}, device={device}")
    print(f"  AMP (mixed-precision): {use_amp}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_split = "train_hn" if experiment == 3 else "train"
    print(f"  Train split: '{train_split}'")

    train_ds = TurnDataset(train_split)
    val_ds   = TurnDataset("val")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(experiment).to(device)
    param_info = model.count_params()
    print(f"\n  Model params: {param_info}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    pos_weight = compute_pos_weight(train_ds, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ── Training loop ─────────────────────────────────────────────────────────
    best_auroc = 0.0
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_metrics_list = []
    ckpt_path = CKPT_DIR / f"exp{experiment}_best.pt"

    print(f"\n  Starting training for up to {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_start = time.time()

        for step, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)):
            x, y = x.to(device), y.to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(x).squeeze(-1)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x).squeeze(-1)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, criterion, device)
        train_losses.append(avg_loss)
        val_metrics_list.append(val_metrics)

        elapsed = time.time() - step_start
        print(
            f"  Epoch {epoch:2d}/{epochs} | loss={avg_loss:.4f} "
            f"| val_auroc={val_metrics['auroc']:.4f} "
            f"| val_f1={val_metrics['f1']:.4f} "
            f"| val_acc={val_metrics['acc']:.4f} "
            f"| {elapsed:.0f}s"
        )

        # Early stopping on AUROC
        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "experiment": experiment,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "param_info": param_info,
            }, str(ckpt_path))
            print(f"  ✓ New best AUROC={best_auroc:.4f} → saved checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    # ── Save results ──────────────────────────────────────────────────────────
    save_curves(train_losses, val_metrics_list, experiment)

    results = {
        "experiment": experiment,
        "best_epoch": best_epoch,
        "best_val_auroc": best_auroc,
        "final_val_metrics": val_metrics_list[-1],
        "best_val_metrics": val_metrics_list[best_epoch - 1],
        "param_info": param_info,
        "config": {
            "train_split": train_split,
            "epochs_run": epoch,
            "batch_size": batch_size,
            "lr": lr,
        },
    }
    results_path = STATS_DIR / f"exp{experiment}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Results saved to {results_path}")
    print(f"  Best val AUROC: {best_auroc:.4f} (epoch {best_epoch})")
    print(f"  Checkpoint: {ckpt_path}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--patience",   type=int, default=3)
    args = parser.parse_args()

    train(
        experiment=args.experiment,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
