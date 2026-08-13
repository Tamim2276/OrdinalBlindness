"""
generate_figures.py — Generate all paper figures from experiment CSVs

Produces:
  1. fig1_qwk_curves_dirichlet.pdf  — QWK vs rounds for α=0.1
  2. fig2_qwk_curves_realsite.pdf   — QWK vs rounds for real_site
  3. fig3_qwk_curves_iid.pdf        — QWK vs rounds for IID
  4. fig4_distributions.pdf         — Client grade distributions for all 3 partitions
  5. fig4_qwk_combined.pdf          — 3-panel QWK comparison

Usage:
    python experiments/generate_figures.py

Requires: matplotlib, pandas, numpy, json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — works without display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR    = os.path.join(BASE_DIR, 'results')
PARTITIONS_DIR = os.path.join(BASE_DIR, 'data', 'partitions')
FIGURES_DIR    = os.path.join(BASE_DIR, 'results', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
COLORS = {
    'fedavg'    : '#E74C3C',   # red
    'fedprox'   : '#F39C12',   # orange
    'ordinalfed': '#2ECC71',   # green
    'central'   : '#95A5A6',   # grey (dashed ceiling)
}
LABELS = {
    'fedavg'    : 'FedAvg',
    'fedprox'   : 'FedProx (μ=0.01)',
    'ordinalfed': 'OrdinalFed (ours)',
}
MARKERS = {
    'fedavg'    : 'o',
    'fedprox'   : 's',
    'ordinalfed': 'D',
}
CENTRALISED_QWK = 0.8494

GRADE_COLORS = ['#3498DB','#2ECC71','#F39C12','#E74C3C','#9B59B6']
GRADE_LABELS = ['Grade 0','Grade 1','Grade 2','Grade 3','Grade 4']


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_qwk_series(csv_path):
    """Load per-round QWK from a results CSV. Returns list of QWK values."""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if 'avg_qwk' not in df.columns:
        return None
    return df['avg_qwk'].tolist()


def load_partition_distributions(partition_name, data_dir):
    """
    Load per-client grade distributions from partition JSON + DDR train labels.
    Returns dict: client_id -> array of grade counts [G0..G4]
    """
    # Load labels
    txt_path = os.path.join(data_dir, 'train.txt')
    labels = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2 and int(parts[1]) != 5:
                labels.append(int(parts[1]))
    labels = np.array(labels)

    # Load partition
    partition_path = os.path.join(PARTITIONS_DIR, f'{partition_name}.json')
    if not os.path.exists(partition_path):
        return None
    with open(partition_path, 'r') as f:
        raw = json.load(f)
    partition = {int(k): v for k, v in raw.items()}

    # Compute per-client grade counts
    distributions = {}
    for client_id, indices in partition.items():
        client_labels = labels[indices]
        counts = np.zeros(5, dtype=int)
        for grade in range(5):
            counts[grade] = int(np.sum(client_labels == grade))
        distributions[client_id] = counts

    return distributions


# ── Figure 1-3: QWK vs Rounds ─────────────────────────────────────────────────

def plot_qwk_curves(partition_name, title, output_path, default_lower=0.45):
    """
    Plot QWK vs FL round for FedAvg, FedProx, OrdinalFed on one partition.
    Dynamically scales the Y-axis if a model crashes below the default_lower limit.
    """
    # Build CSV paths
    csv_paths = {
        'fedavg'    : os.path.join(RESULTS_DIR, f'fedavg_{partition_name}.csv'),
        'fedprox'   : os.path.join(RESULTS_DIR, f'fedprox_{partition_name}_mu0.01.csv'),
        'ordinalfed': os.path.join(RESULTS_DIR, f'ordinalfed_{partition_name}.csv'),
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))

    all_series = {}
    min_qwk = 0.80  # Baseline tracking minimum
    
    # 1. Load data and find the minimum value across all methods
    for method, csv_path in csv_paths.items():
        qwk_series = load_qwk_series(csv_path)
        if qwk_series is not None:
            all_series[method] = qwk_series
            min_qwk = min(min_qwk, min(qwk_series))
        else:
            print(f"  [Skip] {method} on {partition_name} — CSV not found: {csv_path}")

    if not all_series:
        print(f"  [Skip] No data available for {partition_name} yet")
        plt.close()
        return False
        
    # 2. Dynamic Y-Axis lower bound
    # Expand chart downwards if min_qwk is lower than the default
    lower_ylim = min(default_lower, min_qwk - 0.05)
    lower_ylim = max(-0.1, lower_ylim) # Hard floor at -0.1 to avoid massive blank spaces

    # 3. Plot the data
    for method, qwk_series in all_series.items():
        rounds = list(range(1, len(qwk_series) + 1))
        ax.plot(
            rounds, qwk_series,
            color     = COLORS[method],
            marker    = MARKERS[method],
            linewidth = 2,
            markersize= 6,
            label     = LABELS[method],
            zorder    = 3
        )

    # Centralised ceiling
    ax.axhline(
        y         = CENTRALISED_QWK,
        color     = COLORS['central'],
        linestyle = '--',
        linewidth = 1.5,
        label     = f'Centralised (ceiling = {CENTRALISED_QWK})',
        zorder    = 2
    )

    # 0.80 gate line
    ax.axhline(
        y         = 0.80,
        color     = '#BDC3C7',
        linestyle = ':',
        linewidth = 1.0,
        label     = 'Clinical threshold (QWK=0.80)',
        zorder    = 1
    )

    ax.set_xlabel('FL Round', fontsize=12)
    ax.set_ylabel('Validation QWK', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlim(0.5, 10.5)
    
    # Apply dynamic Y-limit
    ax.set_ylim(lower_ylim, 0.90)
    
    ax.set_xticks(range(1, 11))
    
    # Move legend to the left for dirichlet_0.1 so it doesn't block the dip
    if partition_name == 'dirichlet_0.1':
        ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    else:
        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
        
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    return True


def plot_all_qwk_curves_combined(output_path):
    """
    3-panel figure: IID | α=0.1 | real_site side by side.
    This is the main figure for the paper.
    """
    partitions = [
        ('iid',           'IID',              0.75),
        ('dirichlet_0.1', 'Dirichlet α=0.1',  0.45),
        ('real_site',     'Real-site',        0.75),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    for ax, (partition_name, title, default_lower) in zip(axes, partitions):
        csv_paths = {
            'fedavg'    : os.path.join(RESULTS_DIR, f'fedavg_{partition_name}.csv'),
            'fedprox'   : os.path.join(RESULTS_DIR, f'fedprox_{partition_name}_mu0.01.csv'),
            'ordinalfed': os.path.join(RESULTS_DIR, f'ordinalfed_{partition_name}.csv'),
        }

        all_series = {}
        min_qwk = 0.80
        
        # Discover min_qwk for dynamic limits
        for method, csv_path in csv_paths.items():
            qwk_series = load_qwk_series(csv_path)
            if qwk_series is not None:
                all_series[method] = qwk_series
                min_qwk = min(min_qwk, min(qwk_series))

        lower_ylim = min(default_lower, min_qwk - 0.05)
        lower_ylim = max(-0.1, lower_ylim)

        for method, qwk_series in all_series.items():
            rounds = list(range(1, len(qwk_series) + 1))
            ax.plot(
                rounds, qwk_series,
                color     = COLORS[method],
                marker    = MARKERS[method],
                linewidth = 2,
                markersize= 5,
                label     = LABELS[method],
                zorder    = 3
            )

        ax.axhline(
            y=CENTRALISED_QWK, color=COLORS['central'],
            linestyle='--', linewidth=1.5,
            label=f'Centralised ({CENTRALISED_QWK})', zorder=2
        )
        ax.axhline(
            y=0.80, color='#BDC3C7',
            linestyle=':', linewidth=1.0,
            label='Threshold (0.80)', zorder=1
        )

        ax.set_xlabel('FL Round', fontsize=11)
        ax.set_ylabel('Validation QWK' if ax == axes[0] else '', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(0.5, 10.5)
        
        # Apply dynamic limit
        ax.set_ylim(lower_ylim, 0.90)
        
        ax.set_xticks(range(1, 11))
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if ax == axes[2]:
            ax.legend(loc='lower right', fontsize=8, framealpha=0.9)

    plt.suptitle(
        'OrdinalFed vs Baselines: Validation QWK Across FL Rounds',
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ── Figure 4: Data Distributions ──────────────────────────────────────────────

def plot_distributions(output_path):
    """
    3×5 stacked bar chart showing per-client grade distributions
    for IID, dirichlet_0.1, and real_site.
    """
    data_dir = os.path.join(BASE_DIR, 'data', 'DDR', 'DR_grading')

    partitions_to_plot = [
        ('iid',           'IID Partition'),
        ('dirichlet_0.1', 'Dirichlet α=0.1'),
        ('real_site',     'Real-site Partition'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (partition_name, title) in zip(axes, partitions_to_plot):
        dists = load_partition_distributions(partition_name, data_dir)
        if dists is None:
            ax.set_title(f'{title}\n(no data)', fontsize=11)
            continue

        num_clients = len(dists)
        client_ids  = sorted(dists.keys())
        x           = np.arange(num_clients)
        bar_width   = 0.6

        # Convert to percentages
        pct = np.zeros((num_clients, 5))
        totals = []
        for i, cid in enumerate(client_ids):
            counts = dists[cid]
            total  = counts.sum()
            totals.append(total)
            pct[i] = counts / total * 100 if total > 0 else 0

        # Stacked bar
        bottom = np.zeros(num_clients)
        for grade in range(5):
            ax.bar(
                x, pct[:, grade],
                bottom    = bottom,
                width     = bar_width,
                color     = GRADE_COLORS[grade],
                label     = GRADE_LABELS[grade],
                edgecolor = 'white',
                linewidth = 0.5
            )
            bottom += pct[:, grade]

        # Add total sample count above each bar
        for i, total in enumerate(totals):
            ax.text(
                i, 102, f'n={total}',
                ha='center', va='bottom',
                fontsize=8, color='#2C3E50', fontweight='bold'
            )

        ax.set_xlabel('Client', fontsize=11)
        ax.set_ylabel('Grade Distribution (%)' if ax == axes[0] else '', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{i}' for i in client_ids])
        ax.set_ylim(0, 115)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        if ax == axes[2]:
            ax.legend(
                loc='upper right', fontsize=8,
                framealpha=0.9, ncol=1
            )

    plt.suptitle(
        'Per-Client Grade Distributions Under Three Federated Partitions',
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Generating Paper Figures")
    print("=" * 60)

    # ── Figure 1: QWK curves — dirichlet_0.1 (main result) ────────────────
    print("\n[1] QWK curves — Dirichlet α=0.1")
    plot_qwk_curves(
        partition_name = 'dirichlet_0.1',
        title          = 'Validation QWK vs FL Round (Dirichlet α=0.1)',
        output_path    = os.path.join(FIGURES_DIR, 'fig1_qwk_dirichlet01.pdf')
    )

    # ── Figure 2: QWK curves — real_site ──────────────────────────────────
    print("\n[2] QWK curves — Real-site")
    plot_qwk_curves(
        partition_name = 'real_site',
        title          = 'Validation QWK vs FL Round (Real-site)',
        output_path    = os.path.join(FIGURES_DIR, 'fig2_qwk_realsite.pdf')
    )

    # ── Figure 3: QWK curves — IID ────────────────────────────────────────
    print("\n[3] QWK curves — IID")
    plot_qwk_curves(
        partition_name = 'iid',
        title          = 'Validation QWK vs FL Round (IID)',
        output_path    = os.path.join(FIGURES_DIR, 'fig3_qwk_iid.pdf')
    )

    # ── Figure 4: Combined 3-panel QWK (main paper figure) ────────────────
    print("\n[4] Combined QWK curves (all 3 partitions)")
    plot_all_qwk_curves_combined(
        output_path = os.path.join(FIGURES_DIR, 'fig4_qwk_combined.pdf')
    )

    # ── Figure 5: Data distributions ──────────────────────────────────────
    print("\n[5] Data distributions")
    plot_distributions(
        output_path = os.path.join(FIGURES_DIR, 'fig5_distributions.pdf')
    )

    print("\n" + "=" * 60)
    print(f"  All figures saved to: {FIGURES_DIR}")
    print("=" * 60)
    print("\nFiles generated:")
    for f in sorted(os.listdir(FIGURES_DIR)):
        path = os.path.join(FIGURES_DIR, f)
        size = os.path.getsize(path) // 1024
        print(f"  {f}  ({size} KB)")


if __name__ == "__main__":
    main()