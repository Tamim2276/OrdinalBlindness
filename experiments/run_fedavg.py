"""
run_fedavg.py — Task 3.2: FedAvg baseline across all 5 partitions

Warm-starts from the centralised checkpoint (QWK=0.8494) and runs
30 FL rounds per partition. Run one partition at a time overnight:

    python experiments/run_fedavg.py --partition iid
    python experiments/run_fedavg.py --partition dirichlet_0.1
    python experiments/run_fedavg.py --partition all

Expected gates:
    iid / dirichlet_1.0  → QWK ≥ 0.80  (FL works under mild heterogeneity)
    dirichlet_0.1        → QWK < 0.80  (FL FAILS — confirms paper premise)
"""

import os
import sys
import csv
import json
import argparse
import torch
import numpy as np
from collections import OrderedDict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import DDRDataset
from src.model   import get_model, get_device, load_checkpoint
from src.client  import DRClient, run_fedavg_round


# ── Config ────────────────────────────────────────────────────────────────────
# 30 rounds is sufficient with warm-start from QWK=0.8494.
# The model is already well-trained — FL only needs to adapt it.
# Time estimate: ~40 min/round × 30 rounds = ~20 hours per partition.
NUM_ROUNDS   = 10
NUM_CLIENTS  = 5
LOCAL_EPOCHS = 5   # matches DRClient constant — shown here for clarity


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_partition(path):
    """Load partition JSON and cast string keys back to int."""
    with open(path, 'r') as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def build_clients(train_partition, val_indices, data_dir, device):
    """
    Instantiate all 5 DRClient objects with their local data slices.
    All clients share the same val set for fair per-round evaluation.
    """
    return [
        DRClient(
            client_id     = i,
            train_indices = train_partition[i],
            val_indices   = val_indices,
            data_dir      = data_dir,
            device        = device
        )
        for i in range(NUM_CLIENTS)
    ]


def weights_to_model(global_weights, device):
    """
    Load a list of NumPy weight arrays into a fresh model instance.
    Used when saving the best checkpoint each round.
    """
    model      = get_model(num_classes=5, pretrained=False).to(device)
    state_dict = OrderedDict({
        k: torch.tensor(v, dtype=torch.float32)
        for k, v in zip(model.state_dict().keys(), global_weights)
    })
    model.load_state_dict(state_dict)
    return model


# ── Core experiment ───────────────────────────────────────────────────────────

def run_experiment(partition_name, train_partition, val_indices,
                   data_dir, results_dir, device, init_weights):
    """
    Run full 30-round FedAvg on one partition and save results to CSV.

    Args:
        partition_name  : e.g. "dirichlet_0.1" — used for filenames + logging
        train_partition : Dict mapping client_id -> list of train indices
        val_indices     : Shared val indices for all clients
        data_dir        : Path to data/DDR/DR_grading/
        results_dir     : Path to results/ folder
        device          : torch.device
        init_weights    : Starting weights (from centralised checkpoint)

    Returns:
        best_qwk : Best validation QWK achieved across all rounds
    """
    print(f"\n{'='*60}")
    print(f"  FedAvg | Partition: {partition_name}")
    print(f"  Rounds: {NUM_ROUNDS} | Clients: {NUM_CLIENTS} | "
          f"Local epochs: {LOCAL_EPOCHS}")
    print(f"{'='*60}")

    # ── Build clients ─────────────────────────────────────────────────────────
    print(f"\n[Setup] Building {NUM_CLIENTS} clients for '{partition_name}'...")
    clients = build_clients(train_partition, val_indices, data_dir, device)

    # Print per-client sample counts so skew is visible
    for client in clients:
        print(f"  Client {client.client_id}: "
              f"{len(client.train_dataset)} train samples")

    # ── Warm-start from centralised checkpoint ────────────────────────────────
    # Each partition gets a fresh independent copy of the init weights.
    # .copy() prevents one experiment from mutating another's starting point.
    global_weights = [w.copy() for w in init_weights]

    # ── CSV setup ─────────────────────────────────────────────────────────────
    csv_path   = os.path.join(results_dir, f"fedavg_{partition_name}.csv")
    fieldnames = [
        "round",
        "avg_train_loss", "avg_val_loss",
        "avg_qwk",        "avg_accuracy",
        "client_0_qwk",   "client_1_qwk",
        "client_2_qwk",   "client_3_qwk",
        "client_4_qwk",
    ]

    best_qwk   = -1.0
    best_round = -1
    history    = []

    # ── FL rounds ─────────────────────────────────────────────────────────────
    print(f"\n[Train] Starting {NUM_ROUNDS} rounds...\n")

    for round_num in range(1, NUM_ROUNDS + 1):

        print(f"── Round {round_num}/{NUM_ROUNDS}  [{partition_name}] "
              + "─" * 20)

        global_weights, metrics = run_fedavg_round(clients, global_weights)

        # Pad per-client QWK list to always have exactly 5 entries
        per_client_qwk = metrics["per_client_qwk"]
        while len(per_client_qwk) < NUM_CLIENTS:
            per_client_qwk.append(0.0)

        is_best = metrics["avg_qwk"] > best_qwk

        # ── Console output ────────────────────────────────────────────────
        print(f"\n  Avg Train Loss : {metrics['avg_train_loss']:.4f}")
        print(f"  Avg Val Loss   : {metrics['avg_val_loss']:.4f}")
        print(f"  Avg Val QWK    : {metrics['avg_qwk']:.4f}"
              + ("  ⭐ new best" if is_best else ""))
        print(f"  Avg Accuracy   : {metrics['avg_accuracy']:.4f}")
        print(f"  Per-client QWK : {per_client_qwk}\n")

        # ── Save best checkpoint ──────────────────────────────────────────
        if is_best:
            best_qwk   = metrics["avg_qwk"]
            best_round = round_num

            best_model      = weights_to_model(global_weights, device)
            best_model_path = os.path.join(
                results_dir, f"fedavg_{partition_name}_best.pth"
            )
            torch.save(
                {
                    "model_state_dict" : best_model.state_dict(),
                    "round"            : round_num,
                    "best_qwk"         : round(best_qwk, 6),
                    "partition"        : partition_name,
                },
                best_model_path
            )
            del best_model   # free VRAM immediately after saving

        # ── Accumulate history ────────────────────────────────────────────
        history.append({
            "round"          : round_num,
            "avg_train_loss" : round(metrics["avg_train_loss"], 6),
            "avg_val_loss"   : round(metrics["avg_val_loss"],   6),
            "avg_qwk"        : round(metrics["avg_qwk"],        6),
            "avg_accuracy"   : round(metrics["avg_accuracy"],   6),
            "client_0_qwk"   : round(per_client_qwk[0],        6),
            "client_1_qwk"   : round(per_client_qwk[1],        6),
            "client_2_qwk"   : round(per_client_qwk[2],        6),
            "client_3_qwk"   : round(per_client_qwk[3],        6),
            "client_4_qwk"   : round(per_client_qwk[4],        6),
        })

        # ── Write CSV after every round ───────────────────────────────────
        # Writing incrementally means results are never lost if the run
        # is interrupted (Ctrl+C, power cut, etc.)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)

    # ── Partition complete ────────────────────────────────────────────────────
    print(f"[Done] '{partition_name}' complete.")
    print(f"  Best QWK : {best_qwk:.4f} at round {best_round}")
    print(f"  CSV      : {csv_path}")
    print(f"  Model    : {os.path.join(results_dir, f'fedavg_{partition_name}_best.pth')}")

    return best_qwk


# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    # ── Argument parser — run one partition or all ────────────────────────────
    parser = argparse.ArgumentParser(
        description="FedAvg baseline for OrdinalFed"
    )
    parser.add_argument(
        "--partition",
        type    = str,
        default = "all",
        choices = [
            "all",
            "iid",
            "dirichlet_1.0",
            "dirichlet_0.5",
            "dirichlet_0.1",
            "real_site",
        ],
        help=(
            "Which partition to run. "
            "Default: all (runs all 5 sequentially). "
            "Recommended: run one per night, e.g. --partition iid"
        )
    )
    args = parser.parse_args()

    partitions = (
        ["iid", "dirichlet_1.0", "dirichlet_0.5", "dirichlet_0.1", "real_site"]
        if args.partition == "all"
        else [args.partition]
    )

    print("=" * 60)
    print("  OrdinalFed — Task 3.2: FedAvg Baseline")
    print(f"  {NUM_ROUNDS} rounds × {NUM_CLIENTS} clients × {LOCAL_EPOCHS} local epochs")
    print(f"  Partition(s): {args.partition}")
    print("=" * 60)

    device = get_device()

    # ── Paths ─────────────────────────────────────────────────────────────────
    base_dir       = os.path.abspath(
                         os.path.join(os.path.dirname(__file__), '..')
                     )
    data_dir       = os.path.join(base_dir, 'data', 'DDR', 'DR_grading')
    partitions_dir = os.path.join(base_dir, 'data', 'partitions')
    results_dir    = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # ── Load centralised checkpoint as warm-start ─────────────────────────────
    # Starting from QWK=0.8494 means:
    #   - Clients fine-tune from a strong, already-converged model
    #   - 30 rounds is enough to see the Non-IID degradation clearly
    #   - Results are directly comparable to the centralised ceiling
    ckpt_path = os.path.join(results_dir, "best_centralised.pth")
    assert os.path.exists(ckpt_path), (
        f"\n🚨 Centralised checkpoint not found at:\n   {ckpt_path}\n"
        f"   Run experiments/run_centralised.py first."
    )

    print(f"\n[Init] Loading centralised checkpoint...")
    init_model, meta = load_checkpoint(ckpt_path, num_classes=5, device=device)
    print(f"[Init] Warm-start: epoch={meta.get('epoch','?')} | "
          f"QWK={meta.get('best_qwk','?')}")

    # Extract weights once — each partition gets its own .copy()
    init_weights = [
        val.cpu().numpy()
        for val in init_model.state_dict().values()
    ]
    del init_model   # free memory — weights are all we need

    # ── Shared val set ────────────────────────────────────────────────────────
    # All clients across all experiments use the same val set.
    # This ensures QWK numbers are directly comparable across partitions.
    full_val    = DDRDataset(root_dir=data_dir, split="valid")
    val_indices = list(range(len(full_val)))
    print(f"[Data] Val set: {len(val_indices)} images "
          f"(shared across all clients and partitions)\n")

    # ── Run selected partition(s) ─────────────────────────────────────────────
    summary = {}   # partition_name → best_qwk

    for partition_name in partitions:
        partition_path = os.path.join(
            partitions_dir, f"{partition_name}.json"
        )
        assert os.path.exists(partition_path), (
            f"\n🚨 Partition file not found:\n   {partition_path}\n"
            f"   Run src/partition.py first."
        )
        train_partition = load_partition(partition_path)

        best_qwk = run_experiment(
            partition_name  = partition_name,
            train_partition = train_partition,
            val_indices     = val_indices,
            data_dir        = data_dir,
            results_dir     = results_dir,
            device          = device,
            init_weights    = init_weights,
        )
        summary[partition_name] = best_qwk

    # ── Final summary + gate checks ───────────────────────────────────────────
    # Only print summary if more than one partition was run, or if all done
    centralised_qwk = float(meta.get("best_qwk", 0.8494))

    print("\n" + "=" * 60)
    print("  FedAvg Results Summary")
    print("=" * 60)
    print(f"  Centralised ceiling : {centralised_qwk:.4f}\n")

    for name, qwk in summary.items():
        gap    = qwk - centralised_qwk
        status = "✅" if qwk >= 0.80 else "🚨"
        print(f"  {status} {name:<20} QWK={qwk:.4f}  (gap={gap:+.4f})")

    print()

    # Gate 1 — only check if IID was run
    if "iid" in summary:
        if summary["iid"] >= 0.80:
            print("✅ GATE 1 PASSED — IID QWK ≥ 0.80 "
                  "(FL infrastructure works)")
        else:
            print("🚨 GATE 1 FAILED — IID QWK < 0.80 "
                  "(FL infrastructure broken — debug before continuing)")

    # Gate 2 — only check if dirichlet_0.1 was run
    if "dirichlet_0.1" in summary:
        if summary["dirichlet_0.1"] < 0.80:
            print("✅ GATE 2 PASSED — α=0.1 QWK < 0.80 "
                  "(FedAvg fails Non-IID — paper premise confirmed ✓)")
        else:
            print("⚠️  GATE 2 NOTE — α=0.1 QWK ≥ 0.80 "
                  "(FedAvg did not fail as expected)")
            print("    The Non-IID degradation may need more rounds "
                  "or stronger α to appear clearly")

    print(f"\n  All results saved to: {results_dir}")
    print("  Next step: run_fedprox.py (Task 3.3)")


if __name__ == "__main__":
    main()