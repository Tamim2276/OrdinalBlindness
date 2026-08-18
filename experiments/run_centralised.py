"""
run_centralised.py — 30-epoch centralised baseline for OrdinalFed

Trains EfficientNet-B3 on the full DDR training set (all data pooled,
no federation). This is the performance ceiling — the best any FL method
can hope to approach.

Hard gate: validation QWK must reach ≥ 0.83 before proceeding to Day 3.
"""

import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Allow imports from project root regardless of where script is run from
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import DDRDataset
from src.model   import get_model, get_device, get_autocast_context, save_checkpoint, count_parameters
from src.metrics import compute_qwk, compute_accuracy, per_grade_metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Run one full training epoch.

    Returns:
        avg_loss : Mean loss across all batches in the epoch
    """
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="  Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        # Mixed precision: float16 on CUDA, bfloat16 on XPU, disabled on CPU
        with get_autocast_context(device):
            outputs = model(images)
            loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Run full validation pass.

    Returns:
        avg_loss  : Mean loss on validation set
        all_preds : Flat list of predicted class indices
        all_labels: Flat list of ground truth labels
    """
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            loss           = criterion(outputs, labels)

            running_loss += loss.item()

            # torch.max returns (values, indices) — we want the predicted class index
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    return avg_loss, all_preds, all_labels


def main():
    print("=" * 60)
    print("  OrdinalFed — Centralised Baseline (30 Epochs)")
    print("=" * 60)

    #Device
    device = get_device()

    # Paths — all anchored to this script's location, not cwd 
    base_dir    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir    = os.path.join(base_dir, 'data', 'DDR', 'DR_grading')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Datasets
    print("\n[Data] Loading DDR dataset...")
    train_dataset = DDRDataset(root_dir=data_dir, split="train")
    val_dataset   = DDRDataset(root_dir=data_dir, split="valid")
    print(f"[Data] Train: {len(train_dataset):,} images")
    print(f"[Data] Val  : {len(val_dataset):,} images")

    #DataLoaders
    # num_workers > 0 prevents CPU from bottlenecking GPU/XPU data feeding
    # pin_memory=True speeds up host -> device tensor transfers
    # pin_memory only helps on CUDA/XPU — harmless on CPU
    num_workers  = min(4, os.cpu_count())
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    #Model
    print("\n[Model] Building EfficientNet-B3...")
    model = get_model(num_classes=5, pretrained=True).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"[Model] Total params    : {total_params:,}")
    print(f"[Model] Trainable params: {trainable_params:,}")

    #Loss — class-weighted cross entropy 
    # DDR grade distribution is heavily imbalanced:
    #   G0: ~46%  G1: ~2.5%  G2: ~34%  G3: ~1.3%  G4: ~9%
    # We invert counts roughly to force the model to attend to rare grades.
    # Without this, the model learns to predict G0/G2 only and gets ~80% accuracy
    # while being useless for the clinically critical G3/G4 grades.
    class_weights = torch.tensor(
        [1.0, 5.0, 1.5, 10.0, 4.0],
        dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    #Optimiser + Scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Cosine Annealing: LR decays smoothly from 1e-4 -> 0 over T_max epochs.
    # Prevents over-shooting minima in later epochs without needing manual LR drops.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    #Training Loop
    epochs   = 30
    best_qwk = -1.0
    history  = []   # accumulates per-epoch metrics for CSV export

    print(f"\n[Train] Starting {epochs}-epoch training run...\n")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}  |  LR: {scheduler.get_last_lr()[0]:.2e}")

        #Train
        avg_train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Step scheduler AFTER the optimiser — CosineAnnealing is epoch-level
        scheduler.step()

        #Validate
        avg_val_loss, all_preds, all_labels = validate(
            model, val_loader, criterion, device
        )

        #Metrics
        acc    = compute_accuracy(all_labels, all_preds)
        qwk    = compute_qwk(all_labels, all_preds)
        report = per_grade_metrics(all_labels, all_preds)

        print(f"  Train Loss : {avg_train_loss:.4f}")
        print(f"  Val Loss   : {avg_val_loss:.4f}")
        print(f"  Val Acc    : {acc:.4f}")
        print(f"  Val QWK    : {qwk:.4f}  {'⭐ (best so far)' if qwk > best_qwk else ''}")

        #Accumulate history for CSV
        history.append({
            "epoch":      epoch + 1,
            "train_loss": round(avg_train_loss, 6),
            "val_loss":   round(avg_val_loss, 6),
            "val_acc":    round(acc, 6),
            "val_qwk":    round(qwk, 6),
        })

        #Save best checkpoint
        if qwk > best_qwk:
            best_qwk = qwk

            # save_checkpoint stores weights + metadata in one .pth file
            # load_checkpoint can recover both with one call in future experiments
            save_checkpoint(
                model,
                path=os.path.join(results_dir, "best_centralised.pth"),
                extra={"epoch": epoch + 1, "best_qwk": round(best_qwk, 6)}
            )

            # Print detailed per-grade breakdown for every new best
            print(f"\n  ── Per-grade breakdown (new best @ epoch {epoch + 1}) ──")
            for i in range(5):
                # FIX: correct key format is "Grade 0" not "0"
                grade_stats = report.get(
                    f"Grade {i}",
                    {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0}
                )
                print(
                    f"    Grade {i}: "
                    f"P={grade_stats['precision']:.3f}  "
                    f"R={grade_stats['recall']:.3f}  "
                    f"F1={grade_stats['f1-score']:.3f}  "
                    f"(n={grade_stats['support']})"
                )
            print()

        #Early warning if training is going badly
        if epoch == 9 and best_qwk < 0.60:
            print(
                "\n  ⚠️  WARNING: QWK < 0.60 after 10 epochs. "
                "Consider lr=3e-4, heavier augmentation, or EfficientNet-B0.\n"
            )

    #Save training history CSV
    csv_path = os.path.join(results_dir, "centralised_history.csv")
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ["epoch", "train_loss", "val_loss", "val_acc", "val_qwk"]
        writer     = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    #Final gate check
    print("=" * 60)
    print(f"  Training Complete")
    print(f"  Best Validation QWK : {best_qwk:.4f}")
    print(f"  History CSV         : {csv_path}")
    print(f"  Best checkpoint     : {os.path.join(results_dir, 'best_centralised.pth')}")
    print("=" * 60)

    if best_qwk >= 0.83:
        print("\n✅ GATE PASSED — QWK ≥ 0.83. Safe to proceed to Day 3 (FedAvg).")
    elif best_qwk >= 0.80:
        print(
            "\n⚠️  GATE PARTIAL — QWK ≥ 0.80 but below 0.83 target. "
            "Proceed with caution. Try heavier augmentation or lr=3e-4."
        )
    else:
        print(
            "\n🚨 GATE FAILED — QWK < 0.80. DO NOT proceed to Day 3. "
            "Debug: check class weights, augmentation, and learning rate."
        )


if __name__ == "__main__":
    main()