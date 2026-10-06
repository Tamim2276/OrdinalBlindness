"""
run_centralised.py — 30-epoch centralised baseline for OrdinalFed

Trains EfficientNet-B3 on the full DDR training set (all data pooled,
no federation). This is the performance ceiling — the best any FL method
can hope to approach.

Hard gate: validation QWK must reach ≥ 0.83 before proceeding to Day 3.

Load shedding: the full training state is saved after every epoch.
Running the same command again resumes from the last finished epoch.
When training ends, the best checkpoint is also scored on the held-out
DDR test split (minus server set V) → results/centralised_test.json.

Usage:
    python experiments/run_centralised.py            # seed 42
"""

import os
import io
import sys
import csv
import argparse

# Keep compiled GPU kernels for every batch shape (see run_fl.py). Before torch.
os.environ.setdefault("ONEDNN_PRIMITIVE_CACHE_CAPACITY", "16384")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Allow imports from project root regardless of where script is run from
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset    import DDRDataset
from src.model      import get_model, get_device, get_autocast_context, save_checkpoint, count_parameters
from src.metrics    import compute_qwk, compute_accuracy, per_grade_metrics
from src.evaluation import load_test_set, make_eval_loader, predict, summarise
from src.run_utils  import (
    set_seed, get_rng_state, set_rng_state, keep_awake, use_utf8_output,
    atomic_torch_save, atomic_write_text, atomic_write_json,
    load_state_with_fallback, remove_state_files,
)


HISTORY_FIELDS = ["epoch", "train_loss", "val_loss", "val_acc", "val_qwk"]


def write_history_csv(history, csv_path):
    buf    = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=HISTORY_FIELDS)
    writer.writeheader()
    writer.writerows(history)
    atomic_write_text(buf.getvalue(), csv_path)


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
    use_utf8_output()
    parser = argparse.ArgumentParser(description="Centralised baseline (resumable)")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    print("=" * 60)
    print(f"  OrdinalFed — Centralised Baseline ({args.epochs} Epochs, seed {args.seed})")
    print("=" * 60)

    # Paths — all anchored to this script's location, not cwd
    base_dir    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir    = os.path.join(base_dir, 'data', 'DDR', 'DR_grading')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    state_path = os.path.join(results_dir, "centralised_state.pt")
    done_path  = os.path.join(results_dir, "centralised_done.json")
    best_path  = os.path.join(results_dir, "best_centralised.pth")
    csv_path   = os.path.join(results_dir, "centralised_history.csv")

    server_val_path = os.path.join(base_dir, 'data', 'partitions', 'server_val.json')

    # Never overwrite a finished baseline by accident — every FL run warm-starts from it
    if os.path.exists(done_path):
        print(f"\nCentralised baseline already complete ({best_path}).")
        print(f"To retrain from scratch, delete {done_path} first.")
        return

    # V must exist before training: the test score at the end excludes its 50 images
    if not os.path.exists(server_val_path):
        raise SystemExit(f"🚨 Missing {server_val_path}\n   Create it first with: python src/server_val.py")

    keep_awake()
    set_seed(args.seed)

    state, state_file = load_state_with_fallback(state_path)

    #Device
    device = get_device()

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
    # ImageNet weights are only needed (and downloaded) for a fresh start —
    # a resumed run overwrites them with the saved state, so no internet is needed
    model = get_model(num_classes=5, pretrained=(state is None)).to(device)

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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    #Training Loop
    epochs      = args.epochs
    best_qwk    = -1.0
    best_epoch  = 0
    best_state  = None   # CPU copy of the best weights, kept in the resume state
    history     = []     # accumulates per-epoch metrics for CSV export
    start_epoch = 0

    #Resume after a power cut
    if state is not None:
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best_qwk, best_epoch = state["best_qwk"], state["best_epoch"]
        best_state, history  = state["best_state"], state["history"]
        start_epoch          = state["epochs_done"]
        set_rng_state(state["rng"])
        # Re-write the best checkpoint from the state so the two always agree
        if best_state is not None:
            atomic_torch_save(
                {"model_state_dict": best_state, "epoch": best_epoch, "best_qwk": round(best_qwk, 6)},
                best_path,
            )
        print(f"\n[Resume] Resuming from epoch {start_epoch + 1} "
              f"(loaded {os.path.basename(state_file)})")

    print(f"\n[Train] Training epochs {start_epoch + 1}–{epochs}...\n")

    for epoch in range(start_epoch, epochs):
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
            best_qwk   = qwk
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            # save_checkpoint stores weights + metadata in one .pth file
            # load_checkpoint can recover both with one call in future experiments
            save_checkpoint(
                model,
                path=best_path,
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

        #Save history + resume state after every epoch (crash-safe)
        write_history_csv(history, csv_path)
        atomic_torch_save(
            {
                "epochs_done"     : epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer"       : optimizer.state_dict(),
                "scheduler"       : scheduler.state_dict(),
                "best_qwk"        : best_qwk,
                "best_epoch"      : best_epoch,
                "best_state"      : best_state,
                "history"         : history,
                "rng"             : get_rng_state(),
            },
            state_path,
            keep_previous = True,
        )

    #Score the best checkpoint on the held-out test split (minus server set V)
    print("\n[Test] Scoring the best checkpoint on the DDR test split (minus V)...")
    model.load_state_dict(best_state)
    test_loader = make_eval_loader(load_test_set(data_dir, server_val_path), device)
    test_labels, test_preds, _ = predict(model, test_loader, device)
    test_m = summarise(test_labels, test_preds)
    atomic_write_json(
        {
            "seed"        : args.seed,
            "best_epoch"  : best_epoch,
            "best_val_qwk": best_qwk,
            "test_qwk"    : test_m["qwk"],
            "test_acc"    : test_m["acc"],
            "test_cm"     : test_m["cm"],
        },
        os.path.join(results_dir, "centralised_test.json"),
    )
    print(f"  Test QWK {test_m['qwk']:.4f} | acc {test_m['acc']:.4f} "
          f"({len(test_loader.dataset)} images)")

    atomic_write_json({"best_epoch": best_epoch, "best_val_qwk": best_qwk}, done_path)
    remove_state_files(state_path)

    #Final gate check
    print("=" * 60)
    print(f"  Training Complete")
    print(f"  Best Validation QWK : {best_qwk:.4f}")
    print(f"  History CSV         : {csv_path}")
    print(f"  Best checkpoint     : {best_path}")
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