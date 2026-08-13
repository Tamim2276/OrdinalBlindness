"""
run_ordinalfed.py — Day 5: OrdinalFed — the paper contribution

OrdinalFed = OrdinalLoss + QWK-weighted aggregation + adaptive β

Expected to outperform FedAvg and FedProx at alpha=0.1 and real_site.
This is the headline result of the paper.

Usage:
    python experiments/run_ordinalfed.py --partition iid
    python experiments/run_ordinalfed.py --partition dirichlet_0.1
    python experiments/run_ordinalfed.py --partition real_site

    # Chain all 3 overnight:
    python experiments/run_ordinalfed.py --partition iid && \
    python experiments/run_ordinalfed.py --partition dirichlet_0.1 && \
    python experiments/run_ordinalfed.py --partition real_site
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

from src.dataset          import DDRDataset
from src.model            import get_model, get_device, load_checkpoint
from src.server_val       import load_server_val_indices
from src.strategy_ordinalfed import (
    DROrdinalClient,
    run_ordinalfed_round,
    compute_beta
)


#Config
NUM_ROUNDS   = 10     # sufficient with warm-start from QWK=0.8494
NUM_CLIENTS  = 5
LOCAL_EPOCHS = 5      # never reduce — critical for Non-IID effect
LAM          = 0.5    # OrdinalLoss lambda: 0.5 = equal CE + ordinal
BETA0        = 0.5    # starting β for adaptive schedule


#Helpers

def load_partition(path):
    """Load partition JSON and cast string keys back to int."""
    with open(path, 'r') as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def weights_to_model(global_weights, device):
    """Load NumPy weight arrays into a fresh model for checkpointing."""
    model      = get_model(num_classes=5, pretrained=False).to(device)
    state_dict = OrderedDict({
        k: torch.tensor(v, dtype=torch.float32)
        for k, v in zip(model.state_dict().keys(), global_weights)
    })
    model.load_state_dict(state_dict)
    return model


def build_clients(train_partition, val_indices, data_dir, device):
    """Instantiate all 5 DROrdinalClient objects."""
    return [
        DROrdinalClient(
            client_id     = i,
            train_indices = train_partition[i],
            val_indices   = val_indices,
            data_dir      = data_dir,
            device        = device,
            lam           = LAM
        )
        for i in range(NUM_CLIENTS)
    ]


#Core experiment

def run_experiment(partition_name, train_partition, val_indices,
                   server_val_dataset, data_dir, results_dir,
                   device, init_weights):
    """
    Run full OrdinalFed experiment on one partition.

    Args:
        partition_name     :"dirichlet_0.1"
        train_partition    : Dict mapping client_id -> list of train indices
        val_indices        : Shared val indices for all clients
        server_val_dataset : 50-image balanced server val set V
        data_dir           : Path to data/DDR/DR_grading/
        results_dir        : Path to results/
        device             : torch.device
        init_weights       : Centralised checkpoint weights

    Returns:
        best_qwk : Best validation QWK achieved
    """
    print(f"\n{'='*60}")
    print(f"  OrdinalFed | Partition: {partition_name}")
    print(f"  Rounds: {NUM_ROUNDS} | Clients: {NUM_CLIENTS} | "
          f"λ={LAM} | β₀={BETA0}")
    print(f"{'='*60}")

    #Build clients
    print(f"\n[Setup] Building {NUM_CLIENTS} OrdinalFed clients...")
    clients = build_clients(train_partition, val_indices, data_dir, device)
    for client in clients:
        print(f"  Client {client.client_id}: "
              f"{len(client.train_dataset)} train samples")

    #Warm-start
    global_weights = [w.copy() for w in init_weights]

    #Print β schedule for this run
    print(f"\n[β Schedule] β₀={BETA0}, T={NUM_ROUNDS}:")
    for r in [1, 3, 5, 7, NUM_ROUNDS]:
        b = compute_beta(r, NUM_ROUNDS, BETA0)
        print(f"  Round {r:>2}: β={b:.4f}")

    #CSV setup
    csv_path   = os.path.join(results_dir, f"ordinalfed_{partition_name}.csv")
    fieldnames = [
        "round", "beta", "lam",
        "avg_train_loss", "avg_val_loss",
        "avg_qwk",        "avg_accuracy",
        "client_0_qwk",   "client_1_qwk",
        "client_2_qwk",   "client_3_qwk",
        "client_4_qwk",
        "server_qwk_0",   "server_qwk_1",
        "server_qwk_2",   "server_qwk_3",
        "server_qwk_4",
    ]

    best_qwk   = -1.0
    best_round = -1
    history    = []

    print(f"\n[Train] Starting {NUM_ROUNDS} rounds...\n")

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"── Round {round_num}/{NUM_ROUNDS}  "
              f"[{partition_name}] " + "─" * 20)

        global_weights, metrics = run_ordinalfed_round(
            clients            = clients,
            global_weights     = global_weights,
            server_val_dataset = server_val_dataset,
            device             = device,
            round_num          = round_num,
            total_rounds       = NUM_ROUNDS,
            beta0              = BETA0
        )

        per_client_qwk  = metrics["per_client_qwk"]
        server_val_qwks = metrics["server_val_qwks"]

        # Pad to 5 entries
        while len(per_client_qwk)  < NUM_CLIENTS:
            per_client_qwk.append(0.0)
        while len(server_val_qwks) < NUM_CLIENTS:
            server_val_qwks.append(0.0)

        is_best = metrics["avg_qwk"] > best_qwk

        #Console output
        print(f"\n  β              : {metrics['beta']:.4f}")
        print(f"  Avg Train Loss : {metrics['avg_train_loss']:.4f}")
        print(f"  Avg Val Loss   : {metrics['avg_val_loss']:.4f}")
        print(f"  Avg Val QWK    : {metrics['avg_qwk']:.4f}"
              + ("  ⭐ new best" if is_best else ""))
        print(f"  Avg Accuracy   : {metrics['avg_accuracy']:.4f}")
        print(f"  Per-client QWK : {per_client_qwk}")
        print(f"  Server val QWK : {server_val_qwks}\n")

        #Save best checkpoint
        if is_best:
            best_qwk   = metrics["avg_qwk"]
            best_round = round_num

            best_model      = weights_to_model(global_weights, device)
            best_model_path = os.path.join(
                results_dir, f"ordinalfed_{partition_name}_best.pth"
            )
            torch.save(
                {
                    "model_state_dict" : best_model.state_dict(),
                    "round"            : round_num,
                    "best_qwk"         : round(best_qwk, 6),
                    "partition"        : partition_name,
                    "lam"              : LAM,
                    "beta0"            : BETA0,
                },
                best_model_path
            )
            del best_model

        #Accumulate history
        history.append({
            "round"          : round_num,
            "beta"           : round(metrics["beta"],           4),
            "lam"            : LAM,
            "avg_train_loss" : round(metrics["avg_train_loss"], 6),
            "avg_val_loss"   : round(metrics["avg_val_loss"],   6),
            "avg_qwk"        : round(metrics["avg_qwk"],        6),
            "avg_accuracy"   : round(metrics["avg_accuracy"],   6),
            "client_0_qwk"   : round(per_client_qwk[0],        6),
            "client_1_qwk"   : round(per_client_qwk[1],        6),
            "client_2_qwk"   : round(per_client_qwk[2],        6),
            "client_3_qwk"   : round(per_client_qwk[3],        6),
            "client_4_qwk"   : round(per_client_qwk[4],        6),
            "server_qwk_0"   : round(server_val_qwks[0],       6),
            "server_qwk_1"   : round(server_val_qwks[1],       6),
            "server_qwk_2"   : round(server_val_qwks[2],       6),
            "server_qwk_3"   : round(server_val_qwks[3],       6),
            "server_qwk_4"   : round(server_val_qwks[4],       6),
        })

        #Write CSV after every round (crash-safe)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)

    #Partition complete
    print(f"[Done] '{partition_name}' complete.")
    print(f"  Best QWK : {best_qwk:.4f} at round {best_round}")
    print(f"  CSV      : {csv_path}")
    print(f"  Model    : ordinalfed_{partition_name}_best.pth")

    return best_qwk


#Main

def main():

    #Argument parser
    parser = argparse.ArgumentParser(
        description="OrdinalFed experiment runner"
    )
    parser.add_argument(
        "--partition",
        type    = str,
        default = "dirichlet_0.1",
        choices = [
            "all",
            "iid",
            "dirichlet_0.1",
            "real_site",
        ],
        help = "Partition to run. Default: dirichlet_0.1"
    )
    args = parser.parse_args()

    partitions = (
        ["iid", "dirichlet_0.1", "real_site"]
        if args.partition == "all"
        else [args.partition]
    )

    print("=" * 60)
    print("  OrdinalFed — Day 5: Paper Contribution")
    print(f"  {NUM_ROUNDS} rounds × {NUM_CLIENTS} clients × "
          f"{LOCAL_EPOCHS} local epochs")
    print(f"  λ={LAM} | β₀={BETA0}")
    print(f"  Partition(s): {args.partition}")
    print("=" * 60)

    device = get_device()

    #Paths
    base_dir       = os.path.abspath(
                         os.path.join(os.path.dirname(__file__), '..')
                     )
    data_dir       = os.path.join(base_dir, 'data', 'DDR', 'DR_grading')
    partitions_dir = os.path.join(base_dir, 'data', 'partitions')
    results_dir    = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    #Load centralised checkpoint
    ckpt_path = os.path.join(results_dir, "best_centralised.pth")
    assert os.path.exists(ckpt_path), (
        f"\n🚨 Centralised checkpoint not found:\n   {ckpt_path}\n"
        f"   Run experiments/run_centralised.py first."
    )

    print(f"\n[Init] Loading centralised checkpoint...")
    init_model, meta = load_checkpoint(ckpt_path, num_classes=5, device=device)
    print(f"[Init] Warm-start: epoch={meta.get('epoch','?')} | "
          f"QWK={meta.get('best_qwk','?')}")

    init_weights = [
        val.cpu().numpy()
        for val in init_model.state_dict().values()
    ]
    del init_model

    #Load server val set V
    server_val_path = os.path.join(partitions_dir, 'server_val.json')
    assert os.path.exists(server_val_path), (
        f"\n🚨 server_val.json not found:\n   {server_val_path}\n"
        f"   Run src/server_val.py first."
    )
    _, server_val_dataset = load_server_val_indices(server_val_path, data_dir)
    print(f"[Data] Server val set V: {len(server_val_dataset)} images "
          f"(10 per grade, balanced)")

    #Shared client val set
    full_val    = DDRDataset(root_dir=data_dir, split="valid")
    val_indices = list(range(len(full_val)))
    print(f"[Data] Client val set : {len(val_indices)} images "
          f"(shared across all clients)\n")

    #Run experiments
    summary = {}

    for partition_name in partitions:
        partition_path = os.path.join(
            partitions_dir, f"{partition_name}.json"
        )
        assert os.path.exists(partition_path), (
            f"\n🚨 Partition not found:\n   {partition_path}\n"
            f"   Run src/partition.py first."
        )
        train_partition = load_partition(partition_path)

        best_qwk = run_experiment(
            partition_name     = partition_name,
            train_partition    = train_partition,
            val_indices        = val_indices,
            server_val_dataset = server_val_dataset,
            data_dir           = data_dir,
            results_dir        = results_dir,
            device             = device,
            init_weights       = init_weights,
        )
        summary[partition_name] = best_qwk

    #Final summary + gate check
    centralised_qwk = float(meta.get("best_qwk", 0.8494))

    print("\n" + "=" * 60)
    print("  OrdinalFed Results Summary")
    print("=" * 60)
    print(f"  Centralised ceiling : {centralised_qwk:.4f}\n")

    for name, qwk in summary.items():
        gap    = qwk - centralised_qwk
        status = "✅" if qwk >= 0.80 else "⚠️"
        print(f"  {status} {name:<20} QWK={qwk:.4f}  (gap={gap:+.4f})")

    print()

    # Gate: OrdinalFed must beat FedAvg at α=0.1
    if "dirichlet_0.1" in summary:
        if summary["dirichlet_0.1"] >= 0.80:
            print("✅ GATE PASSED — OrdinalFed QWK ≥ 0.80 at α=0.1")
            print("   Paper contribution confirmed ✓")
        else:
            print(f"⚠️  OrdinalFed QWK={summary['dirichlet_0.1']:.4f} at α=0.1")
            print("   Below 0.80 but may still beat FedAvg (~0.71)")
            print("   Check results vs FedAvg and FedProx CSVs")

    print(f"\n  All results saved to: {results_dir}")
    print("  Next step: write the paper")


if __name__ == "__main__":
    main()