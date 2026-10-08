"""
run_fl.py — One resumable runner for every FL method and ablation

Used for all revision experiments (the original run_fedavg.py,
run_fedprox.py and run_ordinalfed.py are kept for reference). It adds what
the reviewers asked for:

  - --seed           : repeated runs (Python, NumPy, PyTorch, XPU seeded)
  - test-set metrics : every round the global model is scored on the DDR
                       test split minus server set V (2,464 images), with
                       confusion matrices for per-grade analysis
  - ablations        : abl_ordloss, abl_qwkagg, abl_uniform
  - AccWt baseline   : accuracy-weighted aggregation

and what load shedding needs:

  - the full state is saved after every round (crash-safe write + backup)
  - running the same command again resumes from the last finished round

The global model is evaluated once per round. The old scripts made all five
clients evaluate the same global model on the same validation set.

Usage:
    python experiments/run_fl.py --method ordinalfed --partition dirichlet_0.1 --seed 1
    python experiments/run_fl.py --method fedavg --partition real_site --seed 1 --rounds 2 --tag smoke

Outputs in results/runs/<method>_<partition>_s<seed>[_<tag>]/ :
    config.json                  settings of this run
    history.csv / history.json   per-round metrics (rewritten after every round)
    log.txt                      console summary of every round
    state.pt (+ state.pt.prev)   resume state, deleted when the run finishes
    best.pth                     global model at the best validation-QWK round
    final.pth                    global model after the last round
    done.json                    written last — marks the run as complete
"""

import os
import io
import sys
import csv
import time
import argparse
from collections import OrderedDict

# Each round uses many batch shapes (every client's last batch, evaluation,
# server set V). Their compiled GPU kernels overflow oneDNN's default cache
# of 1024, so kernels were recompiled every round: ~1 min extra per round on
# an Arc B580. Must be set before torch is imported.
os.environ.setdefault("ONEDNN_PRIMITIVE_CACHE_CAPACITY", "16384")

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset             import DDRDataset
from src.model               import get_model, get_device, load_checkpoint
from src.partition           import load_partition
from src.client              import DRClient, CLASS_WEIGHTS, LOCAL_EPOCHS, BATCH_SIZE
from src.client_fedprox      import DRFedProxClient
from src.losses              import get_ordinal_loss
from src.strategy_ordinalfed import DROrdinalClient, compute_beta, ordinalfed_aggregate
from src.strategy_accwt      import accwt_aggregate
from src.server_val          import load_server_val_indices, score_on_server_val
from src.evaluation          import load_test_set, make_eval_loader, predict, summarise
from src.run_utils           import (
    set_seed, get_rng_state, set_rng_state, keep_awake, use_utf8_output,
    atomic_torch_save, atomic_write_text, atomic_write_json,
    load_state_with_fallback, remove_state_files,
)


#Config
NUM_CLIENTS    = 5
DEFAULT_ROUNDS = 10     # T = 10 for every partition, as in the paper
BETA0          = 0.5    # β₀ of the cosine schedule (Eq. 4)
TAU            = 3.0    # softmax temperature τ of the quality weights (Eq. 5)
LAM            = 0.5    # OrdinalLoss λ (Eq. 1)
MU             = 0.01   # FedProx proximal coefficient

# client      : local training rule — "avg" (DRClient / DROrdinalClient) or "prox" (FedProx)
# lam         : OrdinalLoss λ — 0 means class-weighted cross entropy, as in FedAvg
# aggregation : "samples"         n_k / Σn                         (FedAvg)
#               "uniform"         1 / K
#               "server_qwk"      QWK on V, softmax + β(t) blend    (OrdinalFed, Eqs. 4–7)
#               "server_accuracy" accuracy on V, same scheme        (AccWt)
METHODS = {
    # Baselines
    "fedavg"      : dict(client="avg",  lam=0.0, aggregation="samples"),
    "fedprox"     : dict(client="prox", lam=0.0, aggregation="samples"),
    "accwt"       : dict(client="avg",  lam=0.0, aggregation="server_accuracy"),
    # Paper method
    "ordinalfed"  : dict(client="avg",  lam=LAM, aggregation="server_qwk"),
    # Ablations (Reviewer #2)
    "abl_ordloss" : dict(client="avg",  lam=LAM, aggregation="samples"),     # OrdinalLoss only
    "abl_qwkagg"  : dict(client="avg",  lam=0.0, aggregation="server_qwk"),  # QWK weighting only
    "abl_uniform" : dict(client="avg",  lam=0.0, aggregation="uniform"),     # equal weights
    # Optional extension (queue Tier 4): OrdinalFed plus FedProx's proximal term, same μ
    "ordinalfed_prox": dict(client="prox", lam=LAM, aggregation="server_qwk"),
    # Sensitivity (queue Tier 5), one setting changed at a time, α=0.1 only.
    # Grid fixed on Day 2 BEFORE any of these ran; a variant may be reported as
    # better only if it wins on mean VALIDATION QWK, and all three are reported.
    "sens_lam01"  : dict(client="avg",  lam=0.1, aggregation="server_qwk"),             # weaker OrdinalLoss
    "sens_beta00" : dict(client="avg",  lam=LAM, aggregation="server_qwk", beta0=0.0),  # quality weights from round 1
    "sens_tau05"  : dict(client="avg",  lam=LAM, aggregation="server_qwk", tau=0.5),    # sharper quality weights
}

PARTITIONS = ["iid", "dirichlet_1.0", "dirichlet_0.5", "dirichlet_0.1", "real_site"]

CSV_FIELDS = (
    ["round", "beta", "avg_train_loss", "val_loss", "val_qwk", "val_acc", "test_qwk", "test_acc"]
    + [f"weight_{k}" for k in range(NUM_CLIENTS)]
    + [f"server_score_{k}" for k in range(NUM_CLIENTS)]
    + ["server_score_time_s", "round_time_s"]
)


#Helpers

def run_name_for(method, partition, seed, tag=None):
    name = f"{method}_{partition}_s{seed}"
    return f"{name}_{tag}" if tag else name


def load_weights(model, weights):
    """Copy a list of NumPy arrays (state_dict order) into a model."""
    state_dict = OrderedDict({
        k: torch.tensor(v, dtype=torch.float32)
        for k, v in zip(model.state_dict().keys(), weights)
    })
    model.load_state_dict(state_dict, strict=True)


def weighted_average(all_weights, client_weights):
    """For each layer: Σ_k client_weights[k] × layer_k."""
    return [
        np.sum(
            [all_weights[k][layer] * client_weights[k] for k in range(len(all_weights))],
            axis=0
        )
        for layer in range(len(all_weights[0]))
    ]


def build_clients(spec, train_partition, data_dir, device):
    """
    Instantiate the 5 clients for this method. The global model is evaluated
    once per round by the runner, so clients get no local validation set.
    """
    clients = []
    for i in range(NUM_CLIENTS):
        kwargs = dict(
            client_id     = i,
            train_indices = train_partition[i],
            val_indices   = [],
            data_dir      = data_dir,
            device        = device,
        )
        if spec["client"] == "prox":
            client = DRFedProxClient(mu=MU, **kwargs)
            if spec["lam"] > 0:   # OrdinalFed + Prox: OrdinalLoss as the task loss
                client.criterion = get_ordinal_loss(lam=spec["lam"], device=device)
            clients.append(client)
        elif spec["lam"] > 0:
            clients.append(DROrdinalClient(lam=spec["lam"], **kwargs))
        else:
            clients.append(DRClient(**kwargs))
    return clients


def save_model_file(model, weights, path, extra):
    """Save weights in the same format as best_centralised.pth (load_checkpoint reads it)."""
    load_weights(model, weights)
    atomic_torch_save({"model_state_dict": model.state_dict(), **extra}, path)


def write_history(history, run_dir):
    atomic_write_json(history, os.path.join(run_dir, "history.json"))

    buf    = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for r in history:
        row = {key: r.get(key) for key in CSV_FIELDS if key in r}
        for k in range(NUM_CLIENTS):
            row[f"weight_{k}"] = r["agg_weights"][k]
            if r["server_scores"] is not None:
                row[f"server_score_{k}"] = r["server_scores"][k]
        writer.writerow(row)
    atomic_write_text(buf.getvalue(), os.path.join(run_dir, "history.csv"))


#Main

def parse_args():
    parser = argparse.ArgumentParser(description="Resumable FL runner — all methods and ablations")
    parser.add_argument("--method",    required=True, choices=sorted(METHODS))
    parser.add_argument("--partition", required=True, choices=PARTITIONS)
    parser.add_argument("--seed",      required=True, type=int)
    parser.add_argument("--rounds",    type=int, default=DEFAULT_ROUNDS,
                        help=f"Number of FL rounds T. Default: {DEFAULT_ROUNDS}")
    parser.add_argument("--tag",       default=None,
                        help="Suffix for test runs, e.g. 'smoke'. Tagged runs are left out of the tables.")
    return parser.parse_args()


def main():
    use_utf8_output()
    args = parse_args()
    spec = METHODS[args.method]

    #Paths
    base_dir        = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir        = os.path.join(base_dir, 'data', 'DDR', 'DR_grading')
    partitions_dir  = os.path.join(base_dir, 'data', 'partitions')
    results_dir     = os.path.join(base_dir, 'results')
    ckpt_path       = os.path.join(results_dir, 'best_centralised.pth')
    server_val_path = os.path.join(partitions_dir, 'server_val.json')
    partition_path  = os.path.join(partitions_dir, f'{args.partition}.json')

    run_name   = run_name_for(args.method, args.partition, args.seed, args.tag)
    run_dir    = os.path.join(results_dir, 'runs', run_name)
    state_path = os.path.join(run_dir, 'state.pt')
    done_path  = os.path.join(run_dir, 'done.json')

    if os.path.exists(done_path):
        print(f"[{run_name}] already complete — nothing to do.")
        return

    for path, how in [
        (ckpt_path,       "python experiments/run_centralised.py"),
        (server_val_path, "python src/server_val.py"),
        (partition_path,  "python src/partition.py"),
    ]:
        if not os.path.exists(path):
            raise SystemExit(f"🚨 Missing {path}\n   Create it with: {how}")

    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, 'log.txt')

    def log(message=""):
        print(message)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(message + "\n")

    keep_awake()

    config = {
        "method"      : args.method,
        "partition"   : args.partition,
        "seed"        : args.seed,
        "rounds"      : args.rounds,
        "client"      : spec["client"],
        "lam"         : spec["lam"],
        "aggregation" : spec["aggregation"],
        "beta0"       : spec.get("beta0", BETA0),
        "tau"         : spec.get("tau", TAU),
        "mu"          : MU,
        "num_clients" : NUM_CLIENTS,
        "local_epochs": LOCAL_EPOCHS,
        "batch_size"  : BATCH_SIZE,
    }

    state, state_file = load_state_with_fallback(state_path)
    if state is not None and state["config"] != config:
        raise SystemExit(
            f"🚨 {run_dir} holds a run with different settings.\n"
            f"   Delete that folder to start this run from scratch."
        )

    log("=" * 60)
    log(f"  {run_name}  ({'resuming' if state else 'new run'})")
    log(f"  client={spec['client']} | λ={spec['lam']} | aggregation={spec['aggregation']} | "
        f"T={args.rounds} | seed={args.seed}")
    log("=" * 60)

    device = get_device()
    set_seed(args.seed)

    #Data
    train_partition = load_partition(partition_path)
    val_loader      = make_eval_loader(DDRDataset(root_dir=data_dir, split="valid"), device)
    test_loader     = make_eval_loader(load_test_set(data_dir, server_val_path), device)

    server_val_dataset = None
    if spec["aggregation"].startswith("server_"):
        _, server_val_dataset = load_server_val_indices(server_val_path, data_dir)

    if val_loader.dataset.cache_path is None:
        log("⚠️  Image cache not found — each round will be ~10× slower. "
            "Build it once with: python src/cache_images.py")

    #Clients + one model for evaluation / server scoring
    clients        = build_clients(spec, train_partition, data_dir, device)
    eval_model     = get_model(num_classes=5, pretrained=False).to(device)
    eval_criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))

    for client in clients:
        log(f"  Client {client.client_id}: {len(client.train_dataset)} train samples")
    log(f"  Val: {len(val_loader.dataset)} | Test (minus V): {len(test_loader.dataset)}")

    #Fresh start or resume
    if state is None:
        init_model, meta = load_checkpoint(ckpt_path, num_classes=5, device=device)
        global_weights   = [v.cpu().numpy() for v in init_model.state_dict().values()]
        del init_model
        history      = []
        best         = {"round": 0, "val_qwk": -1.0, "test_qwk": None}
        best_weights = None
        start_round  = 1
        config_out   = dict(config, warm_start=meta)
        atomic_write_json(config_out, os.path.join(run_dir, 'config.json'))
    else:
        global_weights = [t.numpy() for t in state["global_weights"]]
        best_weights   = ([t.numpy() for t in state["best_weights"]]
                          if state["best_weights"] is not None else None)
        history = state["history"]
        best    = state["best"]
        for client, optimizer_state in zip(clients, state["optimizer_states"]):
            client.optimizer.load_state_dict(optimizer_state)
        set_rng_state(state["rng"])
        start_round = state["round"] + 1
        log(f"[Resume] Resuming from round {start_round} "
            f"(loaded {os.path.basename(state_file)})")

    #FL rounds
    for round_num in range(start_round, args.rounds + 1):
        round_start = time.time()
        log(f"\n── Round {round_num}/{args.rounds}  [{run_name}] " + "─" * 10)

        # 1. Local training
        all_weights, all_n_samples, train_losses = [], [], []
        fit_config = {"mu": MU} if spec["client"] == "prox" else {}
        for client in clients:
            weights, n_samples, fit_metrics = client.fit(parameters=global_weights, config=fit_config)
            all_weights.append(weights)
            all_n_samples.append(n_samples)
            train_losses.append(fit_metrics["train_loss"])
            log(f"  Client {client.client_id} done | n={n_samples} | "
                f"train_loss={fit_metrics['train_loss']:.4f}")

        # 2. Server-side scoring on V (OrdinalFed, AccWt, QWK-weighting ablation)
        server_scores, score_time = None, None
        if server_val_dataset is not None:
            metric     = "qwk" if spec["aggregation"] == "server_qwk" else "accuracy"
            score_from = time.perf_counter()
            server_scores = []
            for weights in all_weights:
                load_weights(eval_model, weights)
                server_scores.append(
                    score_on_server_val(eval_model, server_val_dataset, device, metric)
                )
            score_time = time.perf_counter() - score_from
            log(f"  Server {metric} on V: {[round(float(s), 4) for s in server_scores]} "
                f"({score_time:.2f}s for {NUM_CLIENTS} clients)")

        # 3. Aggregation
        beta = None
        if spec["aggregation"] == "samples":
            total       = sum(all_n_samples)
            agg_weights = [n / total for n in all_n_samples]
            global_weights = weighted_average(all_weights, agg_weights)
        elif spec["aggregation"] == "uniform":
            agg_weights    = [1.0 / NUM_CLIENTS] * NUM_CLIENTS
            global_weights = weighted_average(all_weights, agg_weights)
        else:
            beta = compute_beta(round_num, args.rounds, beta0=spec.get("beta0", BETA0))
            if spec["aggregation"] == "server_qwk":
                global_weights, agg_weights = ordinalfed_aggregate(
                    all_weights, all_n_samples, server_scores, beta, temperature=spec.get("tau", TAU))
            else:
                global_weights, agg_weights = accwt_aggregate(all_weights, all_n_samples, server_scores, beta)
        log(f"  Aggregation weights: {[round(float(w), 4) for w in agg_weights]}"
            + (f"  (β={beta:.4f})" if beta is not None else ""))
        del all_weights

        # 4. Evaluate the new global model once — validation and held-out test
        load_weights(eval_model, global_weights)
        val_labels,  val_preds,  val_loss = predict(eval_model, val_loader, device, eval_criterion)
        test_labels, test_preds, _        = predict(eval_model, test_loader, device)
        val_m  = summarise(val_labels,  val_preds)
        test_m = summarise(test_labels, test_preds)

        is_best = val_m["qwk"] > best["val_qwk"]
        if is_best:
            best         = {"round": round_num, "val_qwk": val_m["qwk"], "test_qwk": test_m["qwk"]}
            best_weights = [w.copy() for w in global_weights]

        history.append({
            "round"              : round_num,
            "beta"               : beta,
            "avg_train_loss"     : float(np.mean(train_losses)),
            "train_losses"       : [float(l) for l in train_losses],
            "client_n"           : all_n_samples,
            "agg_weights"        : [float(w) for w in agg_weights],
            "server_scores"      : server_scores,
            "server_score_time_s": score_time,
            "val_loss"           : float(val_loss),
            "val_qwk"            : val_m["qwk"],
            "val_acc"            : val_m["acc"],
            "val_cm"             : val_m["cm"],
            "test_qwk"           : test_m["qwk"],
            "test_acc"           : test_m["acc"],
            "test_cm"            : test_m["cm"],
            "round_time_s"       : time.time() - round_start,
        })

        # 5. Save — history first, then the resume state (both crash-safe)
        write_history(history, run_dir)
        atomic_torch_save(
            {
                "config"          : config,
                "round"           : round_num,
                "global_weights"  : [torch.from_numpy(np.ascontiguousarray(w)) for w in global_weights],
                "best_weights"    : ([torch.from_numpy(np.ascontiguousarray(w)) for w in best_weights]
                                     if best_weights is not None else None),
                "best"            : best,
                "history"         : history,
                "optimizer_states": [client.optimizer.state_dict() for client in clients],
                "rng"             : get_rng_state(),
            },
            state_path,
            keep_previous = True,
        )

        log(f"  Val  QWK {val_m['qwk']:.4f} | acc {val_m['acc']:.4f} | loss {val_loss:.4f}"
            + ("  ⭐ new best" if is_best else ""))
        log(f"  Test QWK {test_m['qwk']:.4f} | acc {test_m['acc']:.4f}")
        log(f"  Round time {(time.time() - round_start) / 60:.1f} min")

        # Give cached GPU blocks back between rounds (does not change results). The Day 2
        # evening crashes coincided with GPU memory pressure from other applications.
        if device.type == "xpu":
            log(f"  GPU memory peak {torch.xpu.max_memory_reserved() / 2**30:.1f} GB reserved")
            torch.xpu.empty_cache()
            torch.xpu.reset_peak_memory_stats()

    #Run complete
    last = history[-1]
    common = {"partition": args.partition, "method": args.method, "seed": args.seed}
    save_model_file(eval_model, global_weights, os.path.join(run_dir, 'final.pth'),
                    dict(common, round=last["round"], val_qwk=last["val_qwk"]))
    save_model_file(eval_model, best_weights, os.path.join(run_dir, 'best.pth'),
                    dict(common, round=best["round"], best_qwk=best["val_qwk"]))

    atomic_write_json(
        {
            "run_name"            : run_name,
            "config"              : config,
            "best_round"          : best["round"],
            "best_val_qwk"        : best["val_qwk"],
            "test_qwk_at_best"    : best["test_qwk"],
            "final_val_qwk"       : last["val_qwk"],
            "final_test_qwk"      : last["test_qwk"],
            "total_round_time_min": sum(r["round_time_s"] for r in history) / 60,
        },
        done_path,
    )
    remove_state_files(state_path)

    log(f"\n[Done] {run_name}")
    log(f"  Best val QWK {best['val_qwk']:.4f} at round {best['round']} "
        f"→ test QWK {best['test_qwk']:.4f}")
    log(f"  Results: {run_dir}")


if __name__ == "__main__":
    main()
