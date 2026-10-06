"""
generate_figures.py — Paper figures, drawn at the printed IEEE page width

Reviewer #3 found the text in Figs. 1–2 too small: the old figures were drawn
15 in wide and shrunk to fit the page, so 9 pt text printed at ~4 pt. Both
figures are now drawn at the printed width of a two-column IEEE figure
(7.14 in), so the 8 pt text here is 8 pt on paper.

  Fig. 1  fig5_distributions.pdf  per-client grade mix of the three partitions
  Fig. 2  fig4_qwk_combined.pdf   validation QWK per round, mean ± std over seeds

The file names match the \\includegraphics names in the paper source. Each
figure is written to figures/ and copied into overleaf/ordinalfed_multifile/
so the local paper source stays in sync; upload the two PDFs to Overleaf.

Colour: DR grade is ordinal, so Fig. 1 uses one blue ramp from light
(Grade 0) to dark (Grade 4), which also survives grayscale printing. Fig. 2
gives each method a fixed colour plus its own marker shape, so identity
never depends on colour alone. Both palettes pass the colour-vision checks.

Usage:
    python experiments/generate_figures.py            # both figures
    python experiments/generate_figures.py --fig 1    # only Fig. 1 (needs data/partitions)
    python experiments/generate_figures.py --partial  # Fig. 2 preview incl. unfinished runs
"""

import os
import sys
import json
import shutil
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — works without display
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.run_utils import use_utf8_output


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR       = os.path.join(BASE_DIR, 'data', 'DDR', 'DR_grading')
PARTITIONS_DIR = os.path.join(BASE_DIR, 'data', 'partitions')
RUNS_DIR       = os.path.join(BASE_DIR, 'results', 'runs')
FIGURES_DIR    = os.path.join(BASE_DIR, 'figures')
OVERLEAF_DIR   = os.path.join(BASE_DIR, 'overleaf', 'ordinalfed_multifile')

# ── Page geometry & type ──────────────────────────────────────────────────────
PAGE_WIDTH_IN = 7.14   # IEEE two-column \textwidth (figure*)
FONT_PT       = 8      # minimum text size on paper

INK       = '#0b0b0b'  # primary text
INK_2     = '#52514e'  # secondary text
MUTED     = '#898781'  # axis tick labels / reference line
GRID      = '#e1e0d9'  # hairline gridlines
BASELINE  = '#c3c2b7'  # axis spines

plt.rcParams.update({
    'font.family'      : 'sans-serif',
    'font.sans-serif'  : ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size'        : FONT_PT,
    'axes.titlesize'   : FONT_PT + 0.5,
    'axes.labelsize'   : FONT_PT,
    'xtick.labelsize'  : FONT_PT,
    'ytick.labelsize'  : FONT_PT,
    'legend.fontsize'  : FONT_PT,
    'axes.edgecolor'   : BASELINE,
    'axes.labelcolor'  : INK_2,
    'xtick.color'      : MUTED,
    'ytick.color'      : MUTED,
    'xtick.labelcolor' : INK_2,
    'ytick.labelcolor' : INK_2,
    'axes.linewidth'   : 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size' : 2.5,
    'ytick.major.size' : 2.5,
    'pdf.fonttype'     : 42,   # embed TrueType, not Type 3 (IEEE PDF eXpress)
    'ps.fonttype'      : 42,
})

# Fig. 1 — ordinal ramp, Grade 0 (lightest) → Grade 4 (darkest)
GRADE_COLORS = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']
GRADE_LABELS = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']

# Fig. 2 — colour follows the method in every figure; marker = second channel
METHOD_STYLE = {
    'ordinalfed': dict(color='#2a78d6', marker='o', label='OrdinalFed (ours)', z=4),
    'fedavg'    : dict(color='#eb6834', marker='s', label='FedAvg',            z=3),
    'fedprox'   : dict(color='#1baf7a', marker='D', label=r'FedProx ($\mu$ = 0.01)', z=3),
}
METHOD_ORDER = ['fedavg', 'fedprox', 'ordinalfed']

PANELS = [
    ('iid',           'IID'),
    ('dirichlet_0.1', r'Dirichlet ($\alpha$ = 0.1)'),
    ('real_site',     'Real-site'),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color=GRID, linewidth=0.5, linestyle='-')
    ax.set_axisbelow(True)


def save(fig, filename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    if os.path.isdir(OVERLEAF_DIR):
        shutil.copy(path, os.path.join(OVERLEAF_DIR, filename))
        print(f"  Saved: {path}  (+ copied to overleaf/ordinalfed_multifile/)")
    else:
        print(f"  Saved: {path}")


def load_train_labels():
    labels = []
    with open(os.path.join(DATA_DIR, 'train.txt'), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2 and int(parts[1]) != 5:
                labels.append(int(parts[1]))
    return np.array(labels)


# ── Fig. 1: per-client grade distributions ────────────────────────────────────

def plot_distributions(filename='fig5_distributions.pdf'):
    labels = load_train_labels()
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH_IN, 2.1), sharey=True)

    for ax, (partition, title) in zip(axes, PANELS):
        path = os.path.join(PARTITIONS_DIR, f'{partition}.json')
        if not os.path.exists(path):
            raise SystemExit(f"🚨 Missing {path} — run: python src/partition.py")
        with open(path, 'r') as f:
            partition_map = {int(k): v for k, v in json.load(f).items()}

        clients = sorted(partition_map)
        counts  = np.array([[np.sum(labels[partition_map[c]] == g) for g in range(5)]
                            for c in clients])
        totals  = counts.sum(axis=1)
        shares  = counts / np.maximum(totals[:, None], 1) * 100
        x       = np.arange(len(clients))

        bottom = np.zeros(len(clients))
        for g in range(5):
            present = shares[:, g] > 0   # an empty segment would still draw its edge
            # white edges = the surface gap between stacked segments
            ax.bar(x[present], shares[present, g], bottom=bottom[present], width=0.66,
                   color=GRADE_COLORS[g], edgecolor='white', linewidth=0.8, zorder=2)
            bottom += shares[:, g]

        style_axes(ax)
        ax.set_title(title, color=INK, pad=4)
        ax.set_xticks(x)
        # client id on the first line, number of images it holds on the second
        ax.set_xticklabels([f'C{c}\n{t:,}' for c, t in zip(clients, totals)])
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(axis='x', length=0)

    axes[0].set_ylabel('Share of client images (%)')
    axes[1].set_xlabel('Client and number of images held')

    handles = [Patch(facecolor=c, edgecolor='white', label=l)
               for c, l in zip(GRADE_COLORS, GRADE_LABELS)]
    fig.legend(handles=handles, loc='upper center', ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.08), handlelength=1.2, columnspacing=1.6)
    fig.tight_layout(w_pad=1.2)
    save(fig, filename)


# ── Fig. 2: validation QWK per round ──────────────────────────────────────────

def load_val_curves(include_partial):
    """Return {(method, partition): array (n_seeds, rounds)} from results/runs/."""
    curves = {}
    if not os.path.isdir(RUNS_DIR):
        return curves
    for name in sorted(os.listdir(RUNS_DIR)):
        run_dir = os.path.join(RUNS_DIR, name)
        cfg_path, hist_path = (os.path.join(run_dir, 'config.json'),
                               os.path.join(run_dir, 'history.json'))
        if not (os.path.exists(cfg_path) and os.path.exists(hist_path)):
            continue
        if not include_partial and not os.path.exists(os.path.join(run_dir, 'done.json')):
            continue
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
        if name != f"{cfg['method']}_{cfg['partition']}_s{cfg['seed']}":
            continue   # tagged test run (e.g. _smoke)
        with open(hist_path, 'r') as f:
            qwk = [r['val_qwk'] for r in json.load(f)]
        if qwk:
            curves.setdefault((cfg['method'], cfg['partition']), []).append(qwk)

    out = {}
    for key, runs in curves.items():
        length   = min(len(r) for r in runs)   # partial runs: common prefix only
        out[key] = np.array([r[:length] for r in runs])
    return out


def centralised_val_qwk():
    path = os.path.join(BASE_DIR, 'results', 'centralised_done.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)['best_val_qwk']


def plot_qwk_curves(include_partial, filename='fig4_qwk_combined.pdf'):
    curves = load_val_curves(include_partial)
    if not any((m, p) in curves for m in METHOD_ORDER for p, _ in PANELS):
        print("  [Skip] Fig. 2 — no finished runs in results/runs/ yet "
              "(use --partial for a preview)")
        return False

    ceiling = centralised_val_qwk()
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH_IN, 2.2))
    seed_counts = set()

    for ax, (partition, title) in zip(axes, PANELS):
        lows, highs = [], []
        for method in METHOD_ORDER:
            if (method, partition) not in curves:
                continue
            runs   = curves[(method, partition)]
            style  = METHOD_STYLE[method]
            rounds = np.arange(1, runs.shape[1] + 1)
            mean   = runs.mean(axis=0)
            std    = runs.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)
            seed_counts.add(len(runs))

            ax.fill_between(rounds, mean - std, mean + std, color=style['color'],
                            alpha=0.15, linewidth=0, zorder=style['z'] - 1)
            ax.plot(rounds, mean, color=style['color'], marker=style['marker'],
                    markersize=3.5, markeredgecolor='white', markeredgewidth=0.5,
                    linewidth=1.4, zorder=style['z'])
            lows.append((mean - std).min())
            highs.append((mean + std).max())

        if ceiling is not None:
            ax.axhline(ceiling, color=MUTED, linewidth=0.8, linestyle=(0, (4, 2)), zorder=1)
            highs.append(ceiling)

        if lows:
            pad = 0.015
            ax.set_ylim(min(lows) - pad, max(highs) + pad)
        style_axes(ax)
        ax.set_title(title, color=INK, pad=4)
        ax.set_xticks([2, 4, 6, 8, 10])
        ax.set_xlim(0.6, 10.4)
        ax.set_xlabel('FL round')

    axes[0].set_ylabel('Validation QWK')

    handles = [Line2D([], [], color=METHOD_STYLE[m]['color'], marker=METHOD_STYLE[m]['marker'],
                      markersize=4, linewidth=1.4, label=METHOD_STYLE[m]['label'])
               for m in METHOD_ORDER]
    if ceiling is not None:
        handles.append(Line2D([], [], color=MUTED, linewidth=0.8, linestyle=(0, (4, 2)),
                              label=f'Centralised ({ceiling:.3f})'))
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.09), handlelength=2.2, columnspacing=1.6)
    fig.tight_layout(w_pad=1.4)
    save(fig, filename)

    n = sorted(seed_counts)
    print(f"  Fig. 2: mean ± std over {n[0] if len(n) == 1 else f'{n[0]}–{n[-1]}'} seeds"
          + ("  (PREVIEW — includes unfinished runs)" if include_partial else ""))
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    use_utf8_output()
    parser = argparse.ArgumentParser(description="Generate the paper figures")
    parser.add_argument("--fig", type=int, choices=[1, 2], help="Only this figure")
    parser.add_argument("--partial", action="store_true",
                        help="Fig. 2 preview: include unfinished runs")
    args = parser.parse_args()

    print("=" * 60)
    print("  Generating Paper Figures")
    print("=" * 60)

    if args.fig in (None, 1):
        print("\n[Fig. 1] Per-client grade distributions")
        plot_distributions()
    if args.fig in (None, 2):
        print("\n[Fig. 2] Validation QWK per round")
        plot_qwk_curves(args.partial)

    print("\nUpload the updated PDFs from overleaf/ordinalfed_multifile/ to Overleaf.")


if __name__ == "__main__":
    main()
