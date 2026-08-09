"""
run_fedprox.py — Task 3.3: FedProx baseline across partitions

FedProx adds a proximal term to local training:
    loss = CrossEntropy(w) + (μ/2) * ||w - w_global||²

This prevents clients from drifting too far from the global model
under Non-IID conditions, improving over plain FedAvg at alpha=0.1.

Usage:
    # Single partition with specific μ:
    python experiments/run_fedprox.py --partition dirichlet_0.1 --mu 0.01

    # μ sweep on the hardest partition (pick best μ for paper):
    python experiments/run_fedprox.py --partition dirichlet_0.1 --sweep

    # Chain overnight runs:
    python experiments/run_fedprox.py --partition iid           --mu 0.01 &&
    python experiments/run_fedprox.py --partition dirichlet_0.5 --mu 0.01 &&
    python experiments/run_fedprox.py --partition dirichlet_0.1 --mu 0.01

Expected results vs FedAvg:
    iid          : similar to FedAvg (~0.84)
    dirichlet_0.5: slight improvement over FedAvg
    dirichlet_0.1: meaningful improvement — but OrdinalFed should beat this
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

from src.dataset        import DDRDataset
from src.model          import get_model, get_device, load_checkpoint
from src.client_fedprox import DRFedProxClient, run_fedprox_round


#Config
# 10 rounds is sufficient with warm-start from QWK=0.8494.
# Pattern (IID vs Non-IID gap) is clearly visible by round 5-8.
# Time: ~20 min/round × 10 rounds × 3 partitions = ~10 hours overnight.
NUM_ROUNDS   = 10
NUM_CLIENTS  = 5
LOCAL_EPOCHS = 5

# μ values to sweep — run on dirichlet_0.1 to find best μ for paper
MU_SWEEP = [0.001, 0.01, 0.1]


#Helpers

def load_partition(path):
    """Load partition JSON and cast string keys back to int."""
    with open(path, 'r') as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def build_clients(train_partition, val_indices, data_dir, device, mu):
    """
    Instantiate all 5 DRFedProxClient objects.
    mu is set at construction time but can be overridden per-round via config.
    """
    return [
        DRFedProxClient(
            client_id     = i,
            train_indices = train_partition[i],
            val_indices   = val_indices,
            data_dir      = data_dir,
            device        = device,
            mu            = mu
        )
        for i in range(NUM_CLIENTS)
    ]


def weights_to_model(global_weights, device):
    """Load NumPy weight arrays into a fresh model for checkpointing."""
    model      = get_model(num_classes=5, pretrained=False).to(device)
    state_dict = OrderedDict({
        k: torch.tensor(v, dtype=torch.float32)
        for k, v in zip(model.state_dict().keys(), global_weights)
    })
    model.load_state_dict(state_dict)
    return model


#Core experiment

def run_experiment(partition_name, train_partition, val_indices,
                   data_dir, results_dir, device, init_weights, mu):
    """
    Run full FedProx experiment on one partition with one μ value.

    Args:
        partition_name  :"dirichlet_0.1"
        train_partition : Dict mapping client_id -> list of train indices
        val_indices     : Shared val indices for all clients
        data_dir        : Path to data/DDR/DR_grading/
        results_dir     : Path to results/
        device          : torch.device
        init_weights    : Centralised checkpoint weights (list of NumPy arrays)
        mu              : Proximal term coefficient

    Returns:
        best_qwk : Best validation QWK achieved across all rounds
    """
    run_name = f"fedprox_{partition_name}_mu{mu}"

    print(f"\n{'='*60}")
    print(f"  FedProx | Partition: {partition_name} | μ={mu}")
    print(f"  Rounds: {NUM_ROUNDS} | Clients: {NUM_CLIENTS} | "
          f"Local epochs: {LOCAL_EPOCHS}")
    print(f"{'='*60}")

    #Build clients
    print(f"\n[Setup] Building {NUM_CLIENTS} FedProx clients (μ={mu})...")
    clients = build_clients(
        train_partition, val_indices, data_dir, device, mu
    )
    for client in clients:
        print(f"  Client {client.client_id}: "
              f"{len(client.train_dataset)} train samples")

    #Warm-start
    global_weights = [w.copy() for w in init_weights]

    #CSV setup
    csv_path   = os.path.join(results_dir, f"{run_name}.csv")
    fieldnames = [
        "round", "mu",
        "avg_train_loss", "avg_val_loss",
        "avg_qwk",        "avg_accuracy",
        "client_0_qwk",   "client_1_qwk",
        "client_2_qwk",   "client_3_qwk",
        "client_4_qwk",
    ]

    best_qwk   = -1.0
    best_round = -1
    history    = []

    print(f"\n[Train] Starting {NUM_ROUNDS} rounds (μ={mu})...\n")

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"── Round {round_num}/{NUM_ROUNDS}  "
              f"[{partition_name} | μ={mu}] " + "─" * 15)

        global_weights, metrics = run_fedprox_round(
            clients, global_weights, mu=mu
        )

        per_client_qwk = metrics["per_client_qwk"]
        while len(per_client_qwk) < NUM_CLIENTS:
            per_client_qwk.append(0.0)

        is_best = metrics["avg_qwk"] > best_qwk

        #Console output
        print(f"\n  μ              : {mu}")
        print(f"  Avg Train Loss : {metrics['avg_train_loss']:.4f}")
        print(f"  Avg Val Loss   : {metrics['avg_val_loss']:.4f}")
        print(f"  Avg Val QWK    : {metrics['avg_qwk']:.4f}"
              + ("  ⭐ new best" if is_best else ""))
        print(f"  Avg Accuracy   : {metrics['avg_accuracy']:.4f}")
        print(f"  Per-client QWK : {per_client_qwk}\n")

        #Save best checkpoint
        if is_best:
            best_qwk   = metrics["avg_qwk"]
            best_round = round_num

            best_model      = weights_to_model(global_weights, device)
            best_model_path = os.path.join(
                results_dir, f"{run_name}_best.pth"
            )
            torch.save(
                {
                    "model_state_dict" : best_model.state_dict(),
                    "round"            : round_num,
                    "best_qwk"         : round(best_qwk, 6),
                    "partition"        : partition_name,
                    "mu"               : mu,
                },
                best_model_path
            )
            del best_model

        #Accumulate history
        history.append({
            "round"          : round_num,
            "mu"             : mu,
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

        #Write CSV after every round (crash-safe)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)

    #Partition complete
    print(f"[Done] '{partition_name}' μ={mu} complete.")
    print(f"  Best QWK : {best_qwk:.4f} at round {best_round}")
    print(f"  CSV      : {csv_path}")

    return best_qwk


#Main
def main():

    #Argument parser
    parser = argparse.ArgumentParser(
        description="FedProx baseline for OrdinalFed"
    )
    parser.add_argument(
        "--partition",
        type    = str,
        default = "dirichlet_0.1",
        choices = [
            "all",
            "iid",
            "dirichlet_1.0",
            "dirichlet_0.5",
            "dirichlet_0.1",
            "real_site",
        ],
        help = "Which partition to run. Default: dirichlet_0.1"
    )
    parser.add_argument(
        "--mu",
        type    = float,
        default = 0.01,
        help    = "Proximal term coefficient. Default: 0.01"
    )
    parser.add_argument(
        "--sweep",
        action  = "store_true",
        help    = (
            "Sweep μ ∈ {0.001, 0.01, 0.1} on the chosen partition. "
            "Use this to find the best μ for the paper. "
            "Overrides --mu."
        )
    )
    args = parser.parse_args()

    #Determine what to run
    partitions = (
        ["iid", "dirichlet_1.0", "dirichlet_0.5", "dirichlet_0.1", "real_site"]
        if args.partition == "all"
        else [args.partition]
    )

    mu_values = MU_SWEEP if args.sweep else [args.mu]

    print("=" * 60)
    print("  OrdinalFed — Task 3.3: FedProx Baseline")
    print(f"  {NUM_ROUNDS} rounds × {NUM_CLIENTS} clients × "
          f"{LOCAL_EPOCHS} local epochs")
    print(f"  Partition(s) : {args.partition}")
    print(f"  μ value(s)   : {mu_values}")
    if args.sweep:
        print(f"  Mode         : μ sweep — finding best μ for paper")
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
        f"\n🚨 Centralised checkpoint not found at:\n   {ckpt_path}\n"
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

    #Shared val set
    full_val    = DDRDataset(root_dir=data_dir, split="valid")
    val_indices = list(range(len(full_val)))
    print(f"[Data] Val set: {len(val_indices)} images "
          f"(shared across all clients)\n")

    #Run experiments
    # summary[partition][mu] = best_qwk
    summary = {p: {} for p in partitions}

    for partition_name in partitions:
        partition_path = os.path.join(
            partitions_dir, f"{partition_name}.json"
        )
        assert os.path.exists(partition_path), (
            f"\n🚨 Partition not found:\n   {partition_path}\n"
            f"   Run src/partition.py first."
        )
        train_partition = load_partition(partition_path)

        for mu in mu_values:
            best_qwk = run_experiment(
                partition_name  = partition_name,
                train_partition = train_partition,
                val_indices     = val_indices,
                data_dir        = data_dir,
                results_dir     = results_dir,
                device          = device,
                init_weights    = init_weights,
                mu              = mu,
            )
            summary[partition_name][mu] = best_qwk

    #Final summary
    centralised_qwk = float(meta.get("best_qwk", 0.8494))

    print("\n" + "=" * 60)
    print("  FedProx Results Summary")
    print("=" * 60)
    print(f"  Centralised ceiling : {centralised_qwk:.4f}\n")

    for partition_name, mu_results in summary.items():
        print(f"  Partition: {partition_name}")
        for mu, qwk in mu_results.items():
            gap    = qwk - centralised_qwk
            status = "✅" if qwk >= 0.80 else "🚨"
            print(f"    {status} μ={mu:<6} QWK={qwk:.4f}  (gap={gap:+.4f})")
        print()

    #μ sweep recommendation
    if args.sweep and "dirichlet_0.1" in summary:
        best_mu  = max(
            summary["dirichlet_0.1"],
            key=summary["dirichlet_0.1"].get
        )
        best_qwk = summary["dirichlet_0.1"][best_mu]
        print(f"  Best μ on dirichlet_0.1: μ={best_mu} → QWK={best_qwk:.4f}")
        print(f"  Use this μ for all partition runs:")
        print(f"    python experiments/run_fedprox.py "
              f"--partition iid --mu {best_mu}")
        print(f"    python experiments/run_fedprox.py "
              f"--partition dirichlet_0.5 --mu {best_mu}")
        print(f"    python experiments/run_fedprox.py "
              f"--partition dirichlet_0.1 --mu {best_mu}")

    #Gate check
    if "dirichlet_0.1" in summary:
        best_fedprox_qwk = max(summary["dirichlet_0.1"].values())

        print()
        if best_fedprox_qwk > 0.72:
            print("✅ FedProx improves over FedAvg at alpha=0.1 "
                  f"(QWK={best_fedprox_qwk:.4f} vs FedAvg ~0.72)")
            print("   OrdinalFed should improve further — proceed to Day 5")
        else:
            print("⚠️  FedProx did not clearly improve over FedAvg at alpha=0.1")
            print("   Check μ values — try --sweep for a wider search")

    print(f"\n  All results saved to: {results_dir}")
    print("  Next step: experiments/run_ordinalfed.py")


if __name__ == "__main__":
    main()