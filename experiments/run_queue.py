"""
run_queue.py — Runs every revision experiment in priority order, resumably

After a power cut, run this one command again: finished runs are skipped,
the interrupted run resumes from its last finished round, and the rest
follow in order.

    python experiments/run_queue.py              # run / resume everything
    python experiments/run_queue.py --status     # show progress only
    python experiments/run_queue.py --max-tier 1 # only Tier 1

Order — so whatever has finished always forms complete tables:
    Tier 1 + Tier 2 for seeds 1–2  →  seed 3 of Tiers 1–2  →  Tier 3

    Tier 1 (must)  : FedAvg, FedProx, OrdinalFed × {α=0.1, real-site}
    Tier 2 (should): ablations + AccWt × {α=0.1, real-site}; IID seed 1
    Tier 3 (nice)  : IID seeds 2–3

Each run is a separate `python experiments/run_fl.py ...` process, so a
crash in one run cannot take the queue down with it.
"""

import os
import sys
import json
import argparse
import subprocess
from dataclasses import dataclass

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.run_utils import keep_awake, use_utf8_output


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RUNS_DIR = os.path.join(BASE_DIR, 'results', 'runs')
RUN_FL   = os.path.join(BASE_DIR, 'experiments', 'run_fl.py')
ROUNDS   = 10   # must match run_fl.py DEFAULT_ROUNDS

# ── Edit these to trim the queue (REVISION_PLAN.md, "Compute budget") ─────────
MAIN_SEEDS          = (1, 2, 3)   # Tier 1
ABLATION_SEEDS      = (1, 2, 3)   # Tier 2 ablations + AccWt
ABLATION_PARTITIONS = ("dirichlet_0.1", "real_site")
IID_SEEDS           = (1, 2, 3)   # seed 1 is Tier 2, the others Tier 3

MAIN_METHODS     = ("fedavg", "fedprox", "ordinalfed")
MAIN_PARTITIONS  = ("dirichlet_0.1", "real_site")
TIER2_METHODS    = ("abl_ordloss", "abl_qwkagg", "abl_uniform", "accwt")
MAX_CONSECUTIVE_FAILURES = 3


@dataclass(frozen=True)
class Job:
    tier     : int
    method   : str
    partition: str
    seed     : int

    @property
    def name(self):
        return f"{self.method}_{self.partition}_s{self.seed}"


def tier1(seed):
    if seed not in MAIN_SEEDS:
        return []
    return [Job(1, m, p, seed) for p in MAIN_PARTITIONS for m in MAIN_METHODS]


def tier2(seed):
    jobs = []
    if seed in ABLATION_SEEDS:
        jobs += [Job(2, m, p, seed) for p in ABLATION_PARTITIONS for m in TIER2_METHODS]
    if seed == 1 and 1 in IID_SEEDS:
        jobs += [Job(2, m, "iid", 1) for m in MAIN_METHODS]
    return jobs


def tier3():
    return [Job(3, m, "iid", s) for s in IID_SEEDS if s != 1 for m in MAIN_METHODS]


def build_queue(max_tier=3):
    seeds = sorted(set(MAIN_SEEDS) | set(ABLATION_SEEDS))
    jobs  = []
    for seed in seeds[:2]:
        jobs += tier1(seed) + tier2(seed)
    for seed in seeds[2:]:
        jobs += tier1(seed) + tier2(seed)
    jobs += tier3()
    return [job for job in jobs if job.tier <= max_tier]


def job_progress(job):
    """Return (status, rounds_done, info) for a job, read from its run folder."""
    run_dir = os.path.join(RUNS_DIR, job.name)
    done    = os.path.join(run_dir, 'done.json')
    hist    = os.path.join(run_dir, 'history.json')
    if os.path.exists(done):
        with open(done, 'r') as f:
            return "done", ROUNDS, json.load(f)
    if os.path.exists(hist):
        with open(hist, 'r') as f:
            return "partial", len(json.load(f)), None
    return "pending", 0, None


def median_round_minutes():
    """Median round time over every history written so far (None if no data)."""
    times = []
    if os.path.isdir(RUNS_DIR):
        for name in os.listdir(RUNS_DIR):
            hist = os.path.join(RUNS_DIR, name, 'history.json')
            if os.path.exists(hist):
                with open(hist, 'r') as f:
                    times += [r["round_time_s"] for r in json.load(f) if r.get("round_time_s")]
    return float(np.median(times)) / 60 if times else None


def print_status(queue):
    print(f"\n{'#':>3}  {'Tier':<4} {'Run':<36} {'Status':<8} {'Rounds':<7} "
          f"{'Best val':<9} {'Test@best':<9}")
    rounds_left = 0
    n_done      = 0
    for i, job in enumerate(queue, 1):
        status, rounds_done, info = job_progress(job)
        rounds_left += ROUNDS - rounds_done
        n_done      += status == "done"
        best_val  = f"{info['best_val_qwk']:.4f}"     if info else ""
        test_best = f"{info['test_qwk_at_best']:.4f}" if info else ""
        print(f"{i:>3}  {job.tier:<4} {job.name:<36} {status:<8} "
              f"{rounds_done:>2}/{ROUNDS:<4} {best_val:<9} {test_best:<9}")

    minutes = median_round_minutes()
    print(f"\n  Runs done   : {n_done}/{len(queue)}")
    print(f"  Rounds left : {rounds_left}")
    if minutes:
        print(f"  Round time  : {minutes:.1f} min (median so far)")
        print(f"  GPU time left: ~{rounds_left * minutes / 60:.1f} h (plus power cuts)")


def preflight():
    """Check the one-time setup steps before burning GPU time."""
    checks = [
        (os.path.join(BASE_DIR, 'results', 'best_centralised.pth'), "python experiments/run_centralised.py"),
        (os.path.join(BASE_DIR, 'data', 'partitions', 'server_val.json'), "python src/server_val.py"),
        (os.path.join(BASE_DIR, 'data', 'partitions', 'real_site.json'), "python src/partition.py"),
    ]
    missing = [(path, how) for path, how in checks if not os.path.exists(path)]
    for path, how in missing:
        print(f"🚨 Missing {path}\n   Create it with: {how}")
    cache = os.path.join(BASE_DIR, 'data', 'DDR', 'DR_grading', 'cache_224', 'train.npy')
    if not os.path.exists(cache):
        print("⚠️  Image cache not found — rounds will be ~10× slower. "
              "Build it with: python src/cache_images.py")
    return not missing


def main():
    use_utf8_output()
    parser = argparse.ArgumentParser(description="Run all revision experiments in priority order")
    parser.add_argument("--status",   action="store_true", help="Show progress and exit")
    parser.add_argument("--max-tier", type=int, default=3, choices=[1, 2, 3])
    args = parser.parse_args()

    queue = build_queue(args.max_tier)

    if args.status:
        print_status(queue)
        return

    if not preflight():
        sys.exit(1)

    keep_awake()
    print_status(queue)

    consecutive_failures = 0
    failed = []
    try:
        for i, job in enumerate(queue, 1):
            status, rounds_done, _ = job_progress(job)
            if status == "done":
                continue

            print(f"\n{'#' * 60}\n  [Queue {i}/{len(queue)}] Tier {job.tier}: {job.name}"
                  f"{f'  (resuming after round {rounds_done})' if rounds_done else ''}\n{'#' * 60}")

            cmd = [sys.executable, RUN_FL,
                   "--method", job.method, "--partition", job.partition, "--seed", str(job.seed)]
            for attempt in (1, 2):
                return_code = subprocess.call(cmd, cwd=BASE_DIR)
                if return_code == 0:
                    break
                print(f"\n[Queue] {job.name} failed (exit code {return_code}), attempt {attempt}/2")

            if return_code == 0:
                consecutive_failures = 0
                continue

            consecutive_failures += 1
            failed.append(job.name)
            os.makedirs(RUNS_DIR, exist_ok=True)
            with open(os.path.join(RUNS_DIR, 'queue_failures.txt'), 'a') as f:
                f.write(f"{job.name}: exit code {return_code}\n")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"\n🚨 {consecutive_failures} runs failed in a row — stopping the queue.")
                print("   Read the error above, fix it, then run this command again.")
                sys.exit(1)
    except KeyboardInterrupt:
        print("\n[Queue] Stopped. Run the same command again to resume.")
        sys.exit(130)

    print_status(queue)
    if failed:
        print(f"\n⚠️  Queue finished, but {len(failed)} run(s) failed: {', '.join(failed)}")
        print("   Read their errors above (or results/runs/<run>/log.txt), fix, and run this command again.")
        sys.exit(1)
    print("\n✅ Queue complete.")
    print("\n  Next: python experiments/make_tables.py")


if __name__ == "__main__":
    main()
