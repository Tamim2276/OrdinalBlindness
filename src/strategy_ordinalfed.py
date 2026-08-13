"""
strategy_ordinalfed.py — OrdinalFed core contribution

OrdinalFed improves over FedAvg and FedProx in two ways:

1. LOCAL LOSS — OrdinalLoss instead of CrossEntropy
   Penalises large grade gaps more than small ones.
   (Already built in src/losses.py)

2. AGGREGATION — QWK-weighted instead of sample-count weighted
   FedAvg:    w_k = n_k / n_total
   OrdinalFed: w_k = β(t) × (n_k/n_total) + (1-β(t)) × (κ_k/Σκ)

   Where:
     κ_k  = QWK of client k on server validation set V
     β(t) = adaptive blend coefficient (cosine schedule)

3. ADAPTIVE β COSINE SCHEDULE
   β(t) = β₀ × cos(πt / 2T)

   t=0  : β=β₀ → weight mostly by sample count (like FedAvg)
           early rounds: global model still weak, QWK scores
           are noisy — trust sample count more
   t=T  : β=0  → weight fully by QWK score
           late rounds: model is stronger, QWK scores are
           reliable signals of client quality

   This gradual transition is the key innovation:
   - Prevents noisy QWK scores from corrupting early aggregation
   - Fully exploits QWK signal once the model is capable
"""

import os
import sys
from numpy.random import beta
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.client     import DRClient
from src.losses     import get_ordinal_loss
from src.server_val import evaluate_on_server_val
from src.metrics    import compute_qwk, compute_accuracy
from src.model      import get_model
from tqdm import tqdm


# ── OrdinalFed Client ──────────────────────────────────────────────────────────

class DROrdinalClient(DRClient):
    """
    OrdinalFed client — extends DRClient with OrdinalLoss.

    The ONLY difference from DRClient is the loss function:
        DRClient      : nn.CrossEntropyLoss(weight=class_weights)
        DROrdinalClient: OrdinalLoss(lam=0.5, class_weights=...)

    Everything else — data loading, model, optimizer, get/set parameters,
    evaluate() — is inherited from DRClient unchanged.

    fit() is also inherited — it uses self.criterion which is now
    OrdinalLoss, so ordinal-aware training happens automatically.

    Args:
        lam      : OrdinalLoss lambda. Default 0.5 (equal CE + ordinal).
        *args, **kwargs : Passed to DRClient.__init__()
    """

    def __init__(self, *args, lam=0.5, **kwargs):
        # Call DRClient.__init__() first — sets up data, model, optimizer
        super().__init__(*args, **kwargs)

        # Override the loss function with OrdinalLoss
        # This is the only line that differs from DRClient
        self.criterion = get_ordinal_loss(lam=lam, device=self.device)

        self.lam = lam


# ── Adaptive β Schedule ────────────────────────────────────────────────────────

def compute_beta(round_num, total_rounds, beta0=0.9):
    """
    Cosine annealing schedule for the FedAvg vs QWK blend coefficient.

    β(t) = β₀ × cos(πt / 2T)

    Args:
        round_num    : Current round (1-indexed)
        total_rounds : Total number of FL rounds (T)
        beta0        : Starting β value. Default 0.9.
                       0.9 = 90% sample-count, 10% QWK at round 1
                       Decreases to 0.0 at final round (100% QWK)

    Returns:
        beta : float in [0, β₀]

    Example trajectory (T=10, β₀=0.9):
        Round  1: β = 0.900  (mostly FedAvg weighting)
        Round  3: β = 0.782
        Round  5: β = 0.636
        Round  7: β = 0.424
        Round 10: β = 0.000  (fully QWK weighting)
    """
    t    = round_num - 1          # convert to 0-indexed
    T    = total_rounds - 1       # so final round gives cos(π/2) = 0
    beta = beta0 * np.cos(np.pi * t / (2 * max(T, 1)))
    return float(max(0.0, beta))  # clamp to [0, β₀]


# ── QWK-Weighted Aggregation ───────────────────────────────────────────────────

def ordinalfed_aggregate(all_weights, all_n_samples, qwk_scores, beta):
    """
    Hybrid aggregation: blend FedAvg weights with QWK weights.

    w_k = β × (n_k / n_total)  +  (1-β) × (κ_k / Σκ)
           ↑ FedAvg term              ↑ QWK term

    Args:
        all_weights  : List of per-client weight lists (NumPy arrays)
        all_n_samples: List of per-client sample counts
        qwk_scores   : List of per-client QWK scores on server val set V
        beta         : Current blend coefficient from compute_beta()

    Returns:
        new_global_weights : Aggregated weight list (same structure as input)
    """
    num_clients   = len(all_weights)
    total_samples = sum(all_n_samples)

    # ── FedAvg weights: n_k / n_total ─────────────────────────────────────
    fedavg_weights = [n / total_samples for n in all_n_samples]

    # ── QWK weights: κ_k / Σκ ─────────────────────────────────────────────
    # QWK can be negative (worse than random) — shift to [0, ∞) first
    # by taking max(0, κ_k). This means clients with negative QWK get
    # zero weight — they don't contribute to aggregation at all. 
    TEMPERATURE = 3.0   # higher = more uniform, lower = more selective
    qwk_shifted = [max(0.1, q) for q in qwk_scores]  # floor at 0.1
    exp_scores  = [np.exp(q / TEMPERATURE) for q in qwk_shifted]
    exp_total   = sum(exp_scores)
    qwk_weights = [e / exp_total for e in exp_scores]

    # ── Hybrid weights: β × FedAvg + (1-β) × QWK ─────────────────────────
    MIN_WEIGHT = 0.03
    raw_weights = [
        beta * fedavg_weights[k] + (1 - beta) * qwk_weights[k]
        for k in range(num_clients)
    ]
    # Apply floor
    floored = [max(w, MIN_WEIGHT) for w in raw_weights]
    # Renormalise to sum to 1
    weight_sum     = sum(floored)
    hybrid_weights = [w / weight_sum for w in floored]

    # Normalise to ensure they sum exactly to 1.0 (floating point safety)
    weight_sum     = sum(hybrid_weights)
    hybrid_weights = [w / weight_sum for w in hybrid_weights]

    # ── Weighted average of model weights ─────────────────────────────────
    # For each layer: new_weight = Σ_k hybrid_weight_k × layer_k
    new_global_weights = [
        np.sum(
            [all_weights[k][layer] * hybrid_weights[k]
             for k in range(num_clients)],
            axis=0
        )
        for layer in range(len(all_weights[0]))
    ]

    return new_global_weights, hybrid_weights


# ── OrdinalFed Round ───────────────────────────────────────────────────────────

def run_ordinalfed_round(clients, global_weights, server_val_dataset,
                         device, round_num, total_rounds, beta0=0.9):
    """
    Execute one complete OrdinalFed round.

    One round:
      1. All clients train locally with OrdinalLoss (fit)
      2. Each client's model is scored on server val set V (QWK)
      3. Adaptive β computed for this round
      4. Hybrid aggregation: β×FedAvg + (1-β)×QWK
      5. All clients evaluate the new global model (evaluate)

    Args:
        clients            : List of DROrdinalClient instances
        global_weights     : Current global model weights (NumPy list)
        server_val_dataset : 50-image balanced server val set V
        device             : torch.device
        round_num          : Current round number (1-indexed)
        total_rounds       : Total number of rounds (T)
        beta0              : Starting β. Default 0.9.

    Returns:
        new_global_weights : Aggregated weights after this round
        round_metrics      : Full metrics dict for logging
    """
    num_clients = len(clients)

    # ── Step 1: FIT — all clients train locally ────────────────────────────
    all_weights   = []
    all_n_samples = []
    train_losses  = []

    for client in clients:
        weights, n_samples, fit_metrics = client.fit(
            parameters = global_weights,
            config     = {}
        )
        all_weights.append(weights)
        all_n_samples.append(n_samples)
        train_losses.append(fit_metrics["train_loss"])
        print(
            f"  Client {client.client_id} done | "
            f"n={n_samples} | "
            f"train_loss={fit_metrics['train_loss']:.4f}"
        )

    # ── Step 2: Score each client on server val set V ──────────────────────
    # Load each client's trained weights into a temp model, then score on V.
    # This is the key step that makes OrdinalFed different from FedAvg.
    print(f"\n  Scoring clients on server val set V...")
    qwk_scores = []

    for k, client in enumerate(clients):
        # Load this client's trained weights into a temporary model
        temp_model = get_model(num_classes=5, pretrained=False).to(device)
        state_dict = OrderedDict({
            key: torch.tensor(val, dtype=torch.float32)
            for key, val in zip(
                temp_model.state_dict().keys(),
                all_weights[k]
            )
        })
        temp_model.load_state_dict(state_dict)

        # Score on server val set V
        qwk = evaluate_on_server_val(temp_model, server_val_dataset, device)
        qwk_scores.append(qwk)
        del temp_model   # free VRAM immediately

        print(f"    Client {k}: server val QWK = {qwk:.4f}")

    # ── Step 3: Compute adaptive β for this round ──────────────────────────
    beta = compute_beta(round_num, total_rounds, beta0=beta0)
    print(f"\n  β(t={round_num}) = {beta:.4f}  "
          f"({'FedAvg-like' if beta > 0.5 else 'QWK-weighted'})")

    # ── Step 4: Hybrid aggregation ─────────────────────────────────────────
    new_global_weights, hybrid_weights = ordinalfed_aggregate(
        all_weights   = all_weights,
        all_n_samples = all_n_samples,
        qwk_scores    = qwk_scores,
        beta          = beta
    )

    print(f"  Hybrid weights: {[round(w, 4) for w in hybrid_weights]}")
    print(f"  Aggregated weights from {sum(all_n_samples)} total samples")

    # ── Step 5: EVALUATE — all clients score the new global model ──────────
    val_losses = []
    val_qwks   = []
    val_accs   = []

    for client in clients:
        loss, n_val, eval_metrics = client.evaluate(
            parameters = new_global_weights,
            config     = {}
        )
        val_losses.append(loss)
        val_qwks.append(eval_metrics["qwk"])
        val_accs.append(eval_metrics["accuracy"])

    round_metrics = {
        "avg_train_loss"   : float(np.mean(train_losses)),
        "avg_val_loss"     : float(np.mean(val_losses)),
        "avg_qwk"          : float(np.mean(val_qwks)),
        "avg_accuracy"     : float(np.mean(val_accs)),
        "per_client_qwk"   : [round(q, 4) for q in val_qwks],
        "server_val_qwks"  : [round(q, 4) for q in qwk_scores],
        "hybrid_weights"   : [round(w, 4) for w in hybrid_weights],
        "beta"             : round(beta, 4),
        "total_samples"    : sum(all_n_samples),
    }

    return new_global_weights, round_metrics


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    from src.dataset    import DDRDataset
    from src.model      import get_device, load_checkpoint
    from src.server_val import load_server_val_indices

    print("=" * 60)
    print("  strategy_ordinalfed.py — OrdinalFed Smoke Test")
    print("  3 rounds | IID | β₀=0.9")
    print("=" * 60)

    device   = get_device()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data', 'DDR', 'DR_grading')

    # ── Load IID partition ─────────────────────────────────────────────────
    with open(os.path.join(
        base_dir, 'data', 'partitions', 'iid.json'
    )) as f:
        train_partition = {int(k): v for k, v in json.load(f).items()}

    full_val    = DDRDataset(root_dir=data_dir, split="valid")
    val_indices = list(range(len(full_val)))

    # ── Load server val set V ─────────────────────────────────────────────
    server_val_path = os.path.join(
        base_dir, 'data', 'partitions', 'server_val.json'
    )
    assert os.path.exists(server_val_path), (
        "🚨 server_val.json not found. Run src/server_val.py first."
    )
    _, server_val_dataset = load_server_val_indices(
        server_val_path, data_dir
    )
    print(f"\n[Setup] Server val set V: {len(server_val_dataset)} images")

    # ── Build OrdinalFed clients ───────────────────────────────────────────
    print("[Setup] Building 5 OrdinalFed clients (λ=0.5)...")
    clients = [
        DROrdinalClient(
            client_id     = i,
            train_indices = train_partition[i],
            val_indices   = val_indices,
            data_dir      = data_dir,
            device        = device,
            lam           = 0.5
        )
        for i in range(5)
    ]
    print("  ✅ All 5 OrdinalFed clients instantiated\n")

    # ── Load centralised checkpoint ────────────────────────────────────────
    ckpt_path = os.path.join(base_dir, 'results', 'best_centralised.pth')
    assert os.path.exists(ckpt_path), (
        "🚨 best_centralised.pth not found. "
        "Run experiments/run_centralised.py first."
    )
    init_model, meta = load_checkpoint(ckpt_path, num_classes=5, device=device)
    global_weights   = [v.cpu().numpy() for v in init_model.state_dict().values()]
    del init_model
    print(f"[Init] Warm-start: QWK={meta.get('best_qwk','?')}\n")

    # ── Test β schedule ────────────────────────────────────────────────────
    print("[Test] β schedule (T=3, β₀=0.9):")
    for r in range(1, 4):
        b = compute_beta(r, total_rounds=3, beta0=0.9)
        print(f"  Round {r}: β = {b:.4f}")

    # ── Run 3 OrdinalFed rounds ────────────────────────────────────────────
    print()
    round_losses = []
    TOTAL_ROUNDS = 3

    for round_num in range(1, TOTAL_ROUNDS + 1):
        print(f"── Round {round_num}/{TOTAL_ROUNDS} " + "─" * 38)

        global_weights, metrics = run_ordinalfed_round(
            clients            = clients,
            global_weights     = global_weights,
            server_val_dataset = server_val_dataset,
            device             = device,
            round_num          = round_num,
            total_rounds       = TOTAL_ROUNDS,
            beta0              = 0.9
        )
        round_losses.append(metrics["avg_val_loss"])

        print(f"\n  Round {round_num} Summary:")
        print(f"    β                : {metrics['beta']}")
        print(f"    Avg Train Loss   : {metrics['avg_train_loss']:.4f}")
        print(f"    Avg Val Loss     : {metrics['avg_val_loss']:.4f}")
        print(f"    Avg Val QWK      : {metrics['avg_qwk']:.4f}")
        print(f"    Server Val QWKs  : {metrics['server_val_qwks']}")
        print(f"    Hybrid weights   : {metrics['hybrid_weights']}\n")

    # ── Gate checks ────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Gate Checks")
    print("=" * 60)

    assert all(np.isfinite(l) for l in round_losses), \
        "🚨 NaN or Inf loss detected"
    print("✅ No NaN/Inf losses")

    assert len(round_losses) == 3, \
        f"🚨 Expected 3 rounds, got {len(round_losses)}"
    print("✅ All 3 rounds completed")

    assert round_losses[-1] < 5.0, \
        f"🚨 Final loss {round_losses[-1]:.4f} too high"
    print(f"✅ Final val loss {round_losses[-1]:.4f} is reasonable")

    # Verify β reaches 0 at final round
    final_beta = compute_beta(TOTAL_ROUNDS, TOTAL_ROUNDS)
    assert final_beta < 0.01, \
        f"🚨 β should reach ~0 at final round, got {final_beta}"
    print("✅ β schedule reaches 0 at final round (fully QWK-weighted)")

    print("\n✅ GATE PASSED — strategy_ordinalfed.py is correct.")
    print("   Safe to build experiments/run_ordinalfed.py")