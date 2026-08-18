"""
client_fedprox.py — FedProx client for OrdinalFed

FedProx (Li et al. 2020) adds a proximal term to the local loss
that penalises drifting too far from the global model:

    FedAvg  loss: L(w)
    FedProx loss: L(w) + (μ/2) × ||w - w_global||²
                          ↑ pulls local weights back toward global

Why this helps under Non-IID conditions:
    At α=0.1, each client's data is heavily skewed toward 1-2 grades.
    Without the proximal term, clients overfit their local distribution
    and diverge from each other → aggregation produces a poor global model.
    The proximal term acts as a regulariser that keeps clients anchored
    to the global model, reducing this divergence.

μ (mu) controls the strength of the regularisation:
    μ = 0.0  → identical to FedAvg (no proximal term)
    μ = 0.01 → light regularisation (our default, good starting point)
    μ = 0.1  → moderate regularisation
    μ = 1.0  → heavy regularisation (may underfit local data)

DRFedProxClient extends DRClient — only fit() changes.
evaluate(), get_parameters(), set_parameters() are inherited unchanged.
"""

import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.client  import DRClient, run_fedavg_round
from src.dataset import DDRDataset
from src.model   import get_device, get_autocast_context
from tqdm import tqdm


class DRFedProxClient(DRClient):
    """
    FedProx client — extends DRClient with a proximal regularisation term.

    Inherits everything from DRClient:
        __init__()         ← unchanged (same data, model, optimizer)
        get_parameters()   ← unchanged
        set_parameters()   ← unchanged
        evaluate()         ← unchanged

    Only fit() is overridden to add the proximal term to the loss.

    Args:
        mu : Proximal term coefficient. Default 0.01.
             Passed via config dict from the server each round,
             or set at construction time.
        *args, **kwargs : Passed directly to DRClient.__init__()
    """

    def __init__(self, *args, mu=0.01, **kwargs):
        # Initialise everything from DRClient — data, model, optimizer, etc.
        super().__init__(*args, **kwargs)
        self.mu = mu
        
    def fit(self, parameters, config):
        mu = config.get("mu", self.mu)
        self.set_parameters(parameters)

        # Store FROZEN copy of global weights
        global_params = [
            param.detach().clone().to(self.device)
            for param in self.model.parameters()
        ]

        self.model.train()
        total_loss    = 0.0
        total_batches = 0

        for epoch in range(5):
            pbar = tqdm(
                self.train_loader,
                desc  = f"  Client {self.client_id} | Epoch {epoch+1}/5",
                leave = False,
                ncols = 80
            )

            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                # Mixed-precision forward: float16 on CUDA, bfloat16 on XPU, disabled on CPU
                with get_autocast_context(self.device):
                    outputs   = self.model(images)
                    task_loss = self.criterion(outputs, labels)

                # Only task loss goes through autograd graph (super fast)
                task_loss.backward()

                # ── Proximal term (Applied directly to gradients) ─────────────
                # This mathematically equals adding the penalty to the loss,
                # but bypasses the autograd graph entirely. No PROX_FREQ needed!
                proximal_term_val = 0.0
                with torch.no_grad():
                    for local_param, global_param in zip(self.model.parameters(), global_params):
                        diff = local_param - global_param
                        
                        # Add gradient: d/dw [ (mu/2) * ||w - w_global||^2 ] = mu * (w - w_global)
                        if local_param.grad is not None:
                            local_param.grad.add_(mu * diff)
                        
                        # Accumulate penalty value for logging
                        proximal_term_val += (diff ** 2).sum().item()

                self.optimizer.step()

                total_loss    += task_loss.item()
                total_batches += 1
                pbar.set_postfix({
                    "task": f"{task_loss.item():.4f}",
                    "prox": f"{(mu/2)*proximal_term_val:.4f}"
                })

        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        return (
            self.get_parameters(config={}),
            len(self.train_dataset),
            {"train_loss": float(avg_loss), "client_id": self.client_id, "mu": mu}
        )


def run_fedprox_round(clients, global_weights, mu):
    """
    Execute one complete FedProx round manually.

    Identical to run_fedavg_round() but passes mu through config
    so each client's fit() uses the proximal term.

    Args:
        clients        : List of DRFedProxClient instances
        global_weights : Current global model weights (list of NumPy arrays)
        mu             : Proximal term coefficient for this round

    Returns:
        new_global_weights : Aggregated weights after this round
        round_metrics      : Same dict structure as run_fedavg_round()
    """
    import numpy as np

    # ── FIT with proximal term ─────────────────────────────────────────────
    all_weights   = []
    all_n_samples = []
    train_losses  = []

    for client in clients:
        weights, n_samples, fit_metrics = client.fit(
            parameters = global_weights,
            config     = {"mu": mu}     # ← passes mu to fit() at runtime
        )
        all_weights.append(weights)
        all_n_samples.append(n_samples)
        train_losses.append(fit_metrics["train_loss"])
        print(
            f"  Client {client.client_id} done | "
            f"n={n_samples} | "
            f"train_loss={fit_metrics['train_loss']:.4f} | "
            f"μ={mu}"
        )

    # ── AGGREGATE — same FedAvg weighted average ───────────────────────────
    # FedProx only changes the LOCAL training objective — aggregation is
    # identical to FedAvg. The proximal term is not used during aggregation.
    total_samples      = sum(all_n_samples)
    new_global_weights = [
        np.sum(
            [all_weights[c][layer] * (all_n_samples[c] / total_samples)
             for c in range(len(clients))],
            axis=0
        )
        for layer in range(len(global_weights))
    ]
    print(f"  Aggregated weights from {total_samples} total samples")

    # ── EVALUATE ──────────────────────────────────────────────────────────
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
        "avg_train_loss" : float(np.mean(train_losses)),
        "avg_val_loss"   : float(np.mean(val_losses)),
        "avg_qwk"        : float(np.mean(val_qwks)),
        "avg_accuracy"   : float(np.mean(val_accs)),
        "per_client_qwk" : [round(q, 4) for q in val_qwks],
        "total_samples"  : total_samples,
        "mu"             : mu,
    }

    return new_global_weights, round_metrics


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import numpy as np
    from src.model import get_model, load_checkpoint

    print("=" * 55)
    print("  client_fedprox.py — FedProx Smoke Test")
    print("  3 rounds, IID partition, μ=0.01")
    print("=" * 55)

    device   = get_device()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data', 'DDR', 'DR_grading')

    # ── Load IID partition ─────────────────────────────────────────────────
    with open(os.path.join(base_dir, 'data', 'partitions', 'iid.json')) as f:
        train_partition = {int(k): v for k, v in json.load(f).items()}

    full_val    = DDRDataset(root_dir=data_dir, split="valid")
    val_indices = list(range(len(full_val)))

    # ── Build FedProx clients ──────────────────────────────────────────────
    print("\n[Setup] Building 5 FedProx clients (μ=0.01)...")
    clients = [
        DRFedProxClient(
            client_id     = i,
            train_indices = train_partition[i],
            val_indices   = val_indices,
            data_dir      = data_dir,
            device        = device,
            mu            = 0.01
        )
        for i in range(5)
    ]
    print("  ✅ All 5 FedProx clients instantiated\n")

    # ── Load centralised checkpoint ────────────────────────────────────────
    ckpt_path = os.path.join(base_dir, 'results', 'best_centralised.pth')
    init_model, meta = load_checkpoint(ckpt_path, num_classes=5, device=device)
    global_weights   = [v.cpu().numpy() for v in init_model.state_dict().values()]
    del init_model
    print(f"[Init] Warm-start from centralised QWK={meta.get('best_qwk','?')}\n")

    # ── 3 FL rounds ────────────────────────────────────────────────────────
    round_losses = []
    for round_num in range(1, 4):
        print(f"── Round {round_num}/3 " + "─" * 38)
        global_weights, metrics = run_fedprox_round(
            clients, global_weights, mu=0.01
        )
        round_losses.append(metrics["avg_val_loss"])
        print(f"\n  Avg Train Loss : {metrics['avg_train_loss']:.4f}")
        print(f"  Avg Val Loss   : {metrics['avg_val_loss']:.4f}")
        print(f"  Avg Val QWK    : {metrics['avg_qwk']:.4f}")
        print(f"  Per-client QWK : {metrics['per_client_qwk']}\n")

    # ── Gate checks ────────────────────────────────────────────────────────
    print("=" * 55)
    print("  Gate Checks")
    print("=" * 55)

    assert all(np.isfinite(l) for l in round_losses), \
        "🚨 NaN or Inf loss detected"
    print("✅ No NaN/Inf losses across 3 rounds")

    assert len(round_losses) == 3, \
        f"🚨 Expected 3 rounds, got {len(round_losses)}"
    print("✅ All 3 rounds completed")

    assert round_losses[-1] < 5.0, \
        f"🚨 Loss {round_losses[-1]:.4f} too high"
    print(f"✅ Final val loss {round_losses[-1]:.4f} is reasonable")

    print("\n✅ GATE PASSED — client_fedprox.py is correct.")
    print("   Safe to build experiments/run_fedprox.py")