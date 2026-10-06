"""
make_tables.py — Mean ± std LaTeX tables from results/runs/ (Reviewer #2)

Reads every finished run (done.json), skips tagged test runs (e.g. *_smoke),
groups the seeds of each (method, partition) and writes:

  results/tables/table_main.tex       Test QWK at the best-validation round
  results/tables/table_stability.tex  Validation-QWK stability under α=0.1
  results/tables/table_ablation.tex   OrdinalFed components (ablation)
  results/tables/table_pergrade.tex   Per-grade recall and F1 on the test split
  results/tables/summary.csv          every number above, for checking

Model selection uses validation QWK only. The test split (DDR test minus
server set V) never influences training, aggregation or the chosen round,
so the test numbers are an unbiased estimate.

The tables need \\usepackage{booktabs}.

Usage:
    python experiments/make_tables.py             # finished runs only
    python experiments/make_tables.py --partial   # preview: include unfinished runs too
"""

import os
import io
import csv
import sys
import json
import argparse

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation import per_grade_from_cm
from src.run_utils  import use_utf8_output


BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RUNS_DIR   = os.path.join(BASE_DIR, 'results', 'runs')
TABLES_DIR = os.path.join(BASE_DIR, 'results', 'tables')
DECIMALS   = 3

METHOD_LABELS = {
    "fedavg"      : "FedAvg",
    "fedprox"     : r"FedProx ($\mu=0.01$)",
    "scaffold"    : "SCAFFOLD",
    "accwt"       : "AccWt",
    "ordinalfed"  : r"\textbf{OrdinalFed (Ours)}",
    "abl_ordloss" : "OrdinalLoss + sample-count weights",
    "abl_qwkagg"  : "CE + QWK weights",
    "abl_uniform" : "CE + uniform weights",
}
PARTITION_LABELS = {
    "iid"          : "IID",
    "dirichlet_0.1": r"Dirichlet ($\alpha=0.1$)",
    "real_site"    : "Real-Site",
}
MAIN_ROWS     = ["fedavg", "fedprox", "scaffold", "accwt", "ordinalfed"]
ABLATION_ROWS = ["abl_ordloss", "abl_qwkagg", "abl_uniform", "ordinalfed"]
PERGRADE_ROWS = ["fedavg", "fedprox", "ordinalfed"]
GRADES        = range(5)


#Loading

def load_runs(include_partial):
    """Return {(method, partition): [per-seed metrics]} from results/runs/."""
    groups = {}
    if not os.path.isdir(RUNS_DIR):
        return groups
    for name in sorted(os.listdir(RUNS_DIR)):
        run_dir = os.path.join(RUNS_DIR, name)
        config  = os.path.join(run_dir, 'config.json')
        history = os.path.join(run_dir, 'history.json')
        if not (os.path.exists(config) and os.path.exists(history)):
            continue
        finished = os.path.exists(os.path.join(run_dir, 'done.json'))
        if not finished and not include_partial:
            continue
        with open(config, 'r') as f:
            cfg = json.load(f)
        if name != f"{cfg['method']}_{cfg['partition']}_s{cfg['seed']}":
            continue   # tagged test run (e.g. _smoke)
        with open(history, 'r') as f:
            hist = json.load(f)
        if hist:
            metrics = run_metrics(hist)
            metrics.update(seed=cfg["seed"], finished=finished)
            groups.setdefault((cfg["method"], cfg["partition"]), []).append(metrics)
    return groups


def run_metrics(history):
    """Per-run numbers. The reported round is chosen by validation QWK only."""
    val    = np.array([r["val_qwk"] for r in history])
    best   = history[int(np.argmax(val))]
    grades = per_grade_from_cm(best["test_cm"])
    return {
        "rounds"        : len(history),
        "best_round"    : best["round"],
        "test_qwk"      : best["test_qwk"],
        "test_acc"      : best["test_acc"],
        "final_test_qwk": history[-1]["test_qwk"],
        "val_peak"      : float(val.max()),
        "val_worst"     : float(val.min()),
        "val_mean"      : float(val.mean()),
        "val_range"     : float(val.max() - val.min()),
        "recall"        : grades["recall"].tolist(),
        "f1"            : grades["f1"].tolist(),
    }


#Formatting

def mean_std(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None, None
    std = float(values.std(ddof=1)) if values.size > 1 else None
    return float(values.mean()), std


def cell(values, decimals=DECIMALS):
    mean, std = mean_std(values)
    if mean is None:
        return None, "--"
    text = f"{mean:.{decimals}f}"
    if std is not None:
        text += rf" $\pm$ {std:.{decimals}f}"
    return mean, text


def bold_best(column, higher_is_better=True):
    """Bold the best mean in a column of (mean, text) cells."""
    means = [m for m, _ in column if m is not None]
    if len(means) < 2:
        return [t for _, t in column]
    target = max(means) if higher_is_better else min(means)
    return [rf"\textbf{{{t}}}" if m == target else t for m, t in column]


def latex_table(caption, label, header, rows, wide=False, note=None):
    env  = "table*" if wide else "table"
    cols = "l" + "c" * (len(header) - 1)
    out  = io.StringIO()
    out.write(f"\\begin{{{env}}}[!t]\n\\centering\n")
    out.write(f"\\caption{{{caption}}}\n\\label{{{label}}}\n")
    out.write(f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n")
    out.write(" & ".join(header) + r" \\" + "\n\\midrule\n")
    for row in rows:
        if row == "midrule":
            out.write("\\midrule\n")
        else:
            out.write(" & ".join(row) + r" \\" + "\n")
    out.write("\\bottomrule\n\\end{tabular}\n")
    if note:
        out.write(f"\\\\[2pt]{{\\footnotesize {note}}}\n")
    out.write(f"\\end{{{env}}}\n")
    return out.getvalue()


def seeds_note(groups, keys):
    counts = sorted({len(groups.get(k, [])) for k in keys if groups.get(k)})
    if not counts:
        return None
    n = counts[0] if len(counts) == 1 else f"{counts[0]}--{counts[-1]}"
    return f"Mean $\\pm$ std over {n} seeds."


#Tables

def table_main(groups):
    partitions = ["iid", "dirichlet_0.1", "real_site"]
    methods    = [m for m in MAIN_ROWS if any((m, p) in groups for p in partitions)]

    columns = []
    for p in partitions:
        columns.append(bold_best([cell([r["test_qwk"] for r in groups.get((m, p), [])])
                                  for m in methods]))

    rows = []
    central = os.path.join(BASE_DIR, 'results', 'centralised_test.json')
    if os.path.exists(central):
        with open(central, 'r') as f:
            c = json.load(f)["test_qwk"]
        rows += [["Centralised (ceiling)"] + [f"{c:.{DECIMALS}f}"] * len(partitions), "midrule"]
    for i, m in enumerate(methods):
        rows.append([METHOD_LABELS[m]] + [col[i] for col in columns])

    keys = [(m, p) for m in methods for p in partitions]
    return latex_table(
        caption = "Test QWK (DDR test split, excluding server set $\\mathcal{V}$) at the "
                  "round with the best validation QWK. Bold: best FL method per column.",
        label   = "tab:main",
        header  = ["Method"] + [PARTITION_LABELS[p] for p in partitions],
        rows    = rows,
        note    = seeds_note(groups, keys),
    )


def table_stability(groups, partition="dirichlet_0.1"):
    methods = [m for m in MAIN_ROWS if (m, partition) in groups]
    specs   = [("val_peak", True), ("val_worst", True), ("val_mean", True), ("val_range", False)]
    columns = [bold_best([cell([r[key] for r in groups[(m, partition)]]) for m in methods], higher)
               for key, higher in specs]
    rows    = [[METHOD_LABELS[m]] + [col[i] for col in columns] for i, m in enumerate(methods)]
    return latex_table(
        caption = "Stability of validation QWK across rounds under extreme Dirichlet skew "
                  "($\\alpha=0.1$). Bold: best per column ($\\uparrow$ Peak/Worst/Mean; "
                  "$\\downarrow$ Range).",
        label   = "tab:stability",
        header  = ["Method", "Peak", "Worst", "Mean", "Range"],
        rows    = rows,
        note    = seeds_note(groups, [(m, partition) for m in methods]),
    )


def table_ablation(groups):
    methods = [m for m in ABLATION_ROWS
               if (m, "dirichlet_0.1") in groups or (m, "real_site") in groups]
    columns = [
        bold_best([cell([r["test_qwk"]  for r in groups.get((m, "dirichlet_0.1"), [])]) for m in methods]),
        bold_best([cell([r["val_worst"] for r in groups.get((m, "dirichlet_0.1"), [])]) for m in methods]),
        bold_best([cell([r["test_qwk"]  for r in groups.get((m, "real_site"), [])])     for m in methods]),
    ]
    rows = [[METHOD_LABELS[m]] + [col[i] for col in columns] for i, m in enumerate(methods)]
    keys = [(m, p) for m in methods for p in ("dirichlet_0.1", "real_site")]
    return latex_table(
        caption = "Ablation of OrdinalFed's two components. CE: class-weighted cross-entropy.",
        label   = "tab:ablation",
        header  = ["Configuration", r"Test QWK ($\alpha=0.1$)",
                   r"Worst val QWK ($\alpha=0.1$)", "Test QWK (Real-Site)"],
        rows    = rows,
        note    = seeds_note(groups, keys),
    )


def table_pergrade(groups):
    rows = []
    for p in ("dirichlet_0.1", "real_site"):
        methods = [m for m in PERGRADE_ROWS if (m, p) in groups]
        if not methods:
            continue
        if rows:
            rows.append("midrule")
        for m in methods:
            runs   = groups[(m, p)]
            recall = [f"{np.mean([r['recall'][g] for r in runs]):.2f}" for g in GRADES]
            f1     = [f"{np.mean([r['f1'][g]     for r in runs]):.2f}" for g in GRADES]
            rows.append([f"{PARTITION_LABELS[p]} -- {METHOD_LABELS[m]}"] + recall + f1)
    header = (["Partition -- Method"]
              + [f"R{g}" for g in GRADES] + [f"F1$_{g}$" for g in GRADES])
    return latex_table(
        caption = "Per-grade recall (R) and F1 on the DDR test split (excluding $\\mathcal{V}$), "
                  "mean over seeds, at the best-validation round.",
        label   = "tab:pergrade",
        header  = header,
        rows    = rows,
        wide    = True,
    )


def write_summary_csv(groups, path):
    fields = ["method", "partition", "n_seeds", "seeds", "finished_all"]
    keys   = ["test_qwk", "test_acc", "final_test_qwk", "val_peak", "val_worst",
              "val_mean", "val_range", "best_round"]
    for k in keys:
        fields += [f"{k}_mean", f"{k}_std"]
    fields += [f"recall_g{g}" for g in GRADES] + [f"f1_g{g}" for g in GRADES]

    out    = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    for (method, partition), runs in sorted(groups.items()):
        row = {
            "method"      : method,
            "partition"   : partition,
            "n_seeds"     : len(runs),
            "seeds"       : " ".join(str(r["seed"]) for r in sorted(runs, key=lambda r: r["seed"])),
            "finished_all": all(r["finished"] for r in runs),
        }
        for k in keys:
            row[f"{k}_mean"], row[f"{k}_std"] = mean_std([r[k] for r in runs])
        for g in GRADES:
            row[f"recall_g{g}"] = float(np.mean([r["recall"][g] for r in runs]))
            row[f"f1_g{g}"]     = float(np.mean([r["f1"][g]     for r in runs]))
        writer.writerow(row)
    with open(path, 'w', newline='') as f:
        f.write(out.getvalue())


def print_console_summary(groups):
    print(f"\n{'Method':<14} {'Partition':<15} {'Seeds':<6} {'Test QWK':<17} "
          f"{'Val worst':<17} {'Best rnd':<8}")
    for (method, partition), runs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        tq_m, tq_s = mean_std([r["test_qwk"]  for r in runs])
        vw_m, vw_s = mean_std([r["val_worst"] for r in runs])
        fmt = lambda m, s: f"{m:.4f}" + (f" ± {s:.4f}" if s is not None else "")
        flag = "" if all(r["finished"] for r in runs) else "  (unfinished runs included)"
        print(f"{method:<14} {partition:<15} {len(runs):<6} {fmt(tq_m, tq_s):<17} "
              f"{fmt(vw_m, vw_s):<17} "
              f"{np.mean([r['best_round'] for r in runs]):<8.1f}{flag}")


def main():
    use_utf8_output()
    parser = argparse.ArgumentParser(description="Build paper tables from results/runs/")
    parser.add_argument("--partial", action="store_true",
                        help="Include unfinished runs (preview only — not for the paper)")
    args = parser.parse_args()

    groups = load_runs(args.partial)
    if not groups:
        print("No runs found in results/runs/ yet.")
        return

    os.makedirs(TABLES_DIR, exist_ok=True)
    outputs = {
        "table_main.tex"     : table_main(groups),
        "table_stability.tex": table_stability(groups),
        "table_ablation.tex" : table_ablation(groups),
        "table_pergrade.tex" : table_pergrade(groups),
    }
    for name, text in outputs.items():
        with open(os.path.join(TABLES_DIR, name), 'w', encoding='utf-8') as f:
            f.write(text)
    write_summary_csv(groups, os.path.join(TABLES_DIR, 'summary.csv'))

    print_console_summary(groups)
    print(f"\nTables written to {TABLES_DIR}")
    if args.partial:
        print("⚠️  Preview includes unfinished runs — re-run without --partial for the paper.")


if __name__ == "__main__":
    main()
