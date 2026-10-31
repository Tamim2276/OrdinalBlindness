# Setup guide

This is the "future me" guide: how to rebuild the environment from nothing, get the data back,
rerun anything from the paper, and pick up where we left off. It assumes Windows with Git Bash,
which is what we used, but every step has a Linux/macOS equivalent.

---

## 1. Where things stand (October 2026)

- **The paper is finished and presented.** It was accepted at IEEE WIECON-ECE 2026 (Paper 518) and
  presented in person on 30–31 October 2026. The camera-ready PDF is in `paper/`, and its LaTeX
  source is in `overleaf/ordinalfed_camera_ready/`.
- **All experiments are done:** 66 federated runs (5 methods + 3 ablations + 3 sensitivity settings,
  3 seeds each), plus the centralised baseline and the APTOS 2019 external test.
- **What's in git:** the metrics, logs and per-round history of every run (`results/runs/*/`), the
  paper's tables (`results/tables/`) and the exact data splits (`data/partitions/`). You can rebuild
  every number in the paper without training anything (section 7).
- **What's NOT in git:** the dataset images (`data/DDR`, `data/APTOS`, about 14 GB) and the model
  checkpoints (`*.pth`, about 5.4 GB in total). They are still on the original PC under `results/`.
  The most important one, `best_centralised.pth`, is also in the private archive (section 9).
- **Git history:** commit `8313e35` ("all done") is the full state just before the October 2026
  clean-up, if you ever need an old draft or file. Personal documents were removed from the entire
  history; the private archive has a backup of the original history
  (`backups/OrdinalBlindness_full_history_before_purge.bundle`).

---

## 2. What you need

| | We used | Minimum |
|---|---|---|
| OS | Windows 11 + Git Bash | Windows 10/11, Linux or macOS |
| Python | 3.10.11 | 3.10 |
| GPU | Intel Arc B580 (12 GB), PyTorch XPU build | Any CUDA GPU with 8 GB+, or CPU (very slow) |
| Disk | about 25 GB free | 20 GB |
| Accounts | Hugging Face (free "Read" token) | same |

The code picks the device by itself, in the order CUDA, then Intel XPU, then CPU (`src/model.py`).

---

## 3. Create the environment

```bash
git clone https://github.com/Tamim2276/OrdinalBlindness.git
cd OrdinalBlindness

python -m venv ordinalfed_env
source ordinalfed_env/Scripts/activate      # Git Bash.  cmd: ordinalfed_env\Scripts\activate
                                            # Linux/macOS: source ordinalfed_env/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch for your hardware first.

```bash
# Intel Arc (what we used)
pip install torch==2.13.0+xpu torchvision==0.28.0+xpu torchaudio==2.11.0+xpu --index-url https://download.pytorch.org/whl/xpu

# NVIDIA (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only (enough for tables and figures, far too slow for training)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then install everything else.

```bash
# Intel Arc on Windows: the exact environment we used
pip install -r requirements.txt

# Anything else (CUDA, CPU, Linux, macOS): requirements.txt pins Intel-only packages, so use
pip install timm flwr scikit-learn pandas numpy matplotlib seaborn tqdm pillow huggingface_hub pyarrow opacus grad-cam
```

Check that the GPU is found:

```bash
python -c "from src.model import get_device; print(get_device())"
```

### Environment variables (put these in `~/.bashrc`)

```bash
export PYTHONUTF8=1                        # avoids UnicodeEncodeError on Windows (the scripts print ✓, ±, α)
export ONEDNN_PRIMITIVE_CACHE_CAPACITY=16384   # Intel XPU only: stops the first rounds from crawling
export HF_TOKEN=hf_...                     # your own token, for downloads. Never commit it.
```

---

## 4. Get the data back

### DDR (the training dataset, about 14.2 GB)

```bash
python download_ddr.py
```

The script downloads only `DR_grading/` into `data/DDR/`. It skips files you already have and
waits by itself through rate limits and internet cuts, so you can leave it running. You should end
up with 6,260 training, 2,503 validation and 3,759 test images (after the ungradable ones are
removed).

Next, cache the resized images once. This makes each round about 3 minutes instead of 35:

```bash
python src/cache_images.py      # writes data/DDR/DR_grading/cache_224/*.npy
```

### The data splits: don't regenerate them

`data/partitions/` is tracked in git. It holds the exact client splits (`iid`, `dirichlet_0.1`,
`real_site`, plus `dirichlet_0.5` and `dirichlet_1.0`) and `server_val.json`, the 50-image server
set. Every result in the paper used these files. `src/partition.py` and `src/server_val.py` can make
new ones, but only run them if you deliberately want different splits.

### APTOS 2019 (the external test set)

```bash
python experiments/eval_external.py --download
```

This downloads the public 224 × 224 copy (`bumbledeep/aptos`, 3,662 images), caches it in
`data/APTOS/`, and scores the centralised model plus every run's `best.pth`. Without the
checkpoints, only the download part is useful.

---

## 5. The centralised model (needed before any FL run)

Every FL run starts from the centralised model, `results/best_centralised.pth`. You have two choices.

- **Restore it (quickest):** copy `models/best_centralised.pth` from the private archive into
  `results/`.
- **Retrain it:** 30 epochs with seed 42, a few hours on the Arc B580.

  ```bash
  python experiments/run_centralised.py
  ```

  Our run reached its best validation QWK of 0.834 at epoch 23, with a test QWK of 0.682
  (`results/centralised_test.json`). A new run should land close to that. At the end the script prints
  whether it passed our quality gate (validation QWK of 0.83 or more). Don't start the FL runs if it
  fails.

---

## 6. Run the federated experiments

### Everything, in priority order

```bash
python experiments/run_queue.py              # run or resume all 66 runs
python experiments/run_queue.py --status     # just show progress
python experiments/run_queue.py --max-tier 1 # only the main comparison
bash resume_queue.sh                         # same as the first line, after a power cut
```

Each run is its own process and saves its state after every round, so a power cut costs at most
the round that was running. Just start the queue again. Runs that keep failing are written to
`results/runs/queue_failures*.txt`.

### One run on its own

```bash
python experiments/run_fl.py --method ordinalfed --partition dirichlet_0.1 --seed 1
python experiments/run_fl.py --method fedavg --partition real_site --seed 1 --rounds 2 --tag smoke   # quick test
```

| `--method` | What it is |
|---|---|
| `fedavg` | Standard FedAvg (votes by image count) |
| `fedprox` | FedAvg + proximal term, μ = 0.01 |
| `accwt` | Our server weighting, but scored by accuracy |
| `ordinalfed` | OrdinalLoss (λ = 0.5) + QWK-weighted aggregation (τ = 3, β₀ = 0.5) |
| `ordinalfed_prox` | OrdinalFed + FedProx's proximal term |
| `abl_ordloss` / `abl_qwkagg` / `abl_uniform` | Ablations: loss only / QWK votes only / equal votes |
| `sens_lam01` / `sens_beta00` / `sens_tau05` | Sensitivity: λ = 0.1 / β₀ = 0 / τ = 0.5 |

`--partition` is one of `iid`, `dirichlet_0.1`, `dirichlet_0.5`, `dirichlet_1.0`, `real_site`.

Each run writes `results/runs/<method>_<partition>_s<seed>/` with `config.json`, `history.json`/`.csv`
(validation and test metrics per round), `log.txt`, `done.json` and the `best.pth`/`final.pth`
checkpoints. The checkpoints are not tracked.

### Adding a new method

1. Add one line to the `METHODS` dictionary in `experiments/run_fl.py`. It sets the client type
   (`avg` or `prox`), λ, the aggregation (`samples`, `uniform`, `server_qwk`, `server_accuracy`) and
   optionally `beta0` and `tau`.
2. If it needs a new aggregation rule, add it next to `src/strategy_ordinalfed.py`.
3. Add it to a tier in `experiments/run_queue.py`, so the queue runs it for every seed.

---

## 7. Rebuild the tables and figures

```bash
python experiments/make_tables.py            # results/tables/*.tex and summary.csv (all paper tables)
python experiments/make_tables.py --partial  # also include unfinished runs (preview)
python experiments/generate_figures.py       # figures/fig5_distributions.* and figures/fig4_qwk_combined.*
```

`make_tables.py` only needs the tracked `results/` files, so it works without the data or
checkpoints. We checked: it reproduces the paper's tables exactly. Fig. 2 needs only `results/`;
Fig. 1 also reads `data/partitions/` and the DDR labels.

---

## 8. Build the paper

The final source is `overleaf/ordinalfed_camera_ready/`: `main.tex`, a `sections/` folder and the
two figure PDFs.

- **Overleaf:** zip the contents of that folder (with `main.tex` at the top level) and upload it as a
  new project.
- **Locally (MiKTeX or TeX Live):**

  ```bash
  cd overleaf/ordinalfed_camera_ready
  pdflatex main.tex && pdflatex main.tex
  ```

The camera-ready already has the IEEE header, the copyright line and the author block, and it
passed IEEE PDF eXpress. The response letter source is in `overleaf/response/`.

---

## 9. The private archive

Personal and conference-only files are kept **out of this public repo**, in the separate folder
`OrdinalBlindness-private` (next to this repo on the original PC). Make it a private GitHub repo.
It contains:

- `personal/`: student ID, payment invoice, signed IEEE copyright form, PDF eXpress confirmation
- `conference/`: CMT camera-ready summaries and the reviews screenshot
- `paper_versions/`: the original submission, the revised (anonymous) version and the camera-ready,
  each with its LaTeX source
- `presentation/`: the final slides, speech, study guide and formula notes
- `models/best_centralised.pth`: the warm-start model (saves retraining)
- `backups/` and `notes/`: older results backups and notes

**Never copy anything from `personal/` or `conference/` into this repo.** The `.gitignore` blocks
the obvious file names as a safety net, but don't rely on it.

---

## 10. Troubleshooting (things that actually happened)

| Problem | Fix |
|---|---|
| `UnicodeEncodeError` when printing | `export PYTHONUTF8=1` |
| First rounds on Intel XPU are extremely slow | `export ONEDNN_PRIMITIVE_CACHE_CAPACITY=16384` |
| Random "access violation" or XPU out-of-memory crashes | Another program was using the GPU. Never run two GPU jobs at once; the queue retries failed runs automatically. |
| Power cut in the middle of a run | Run `bash resume_queue.sh`; every run resumes from its last finished round. |
| Pressing Ctrl+C in Git Bash killed the queue | In Git Bash, Ctrl+C stops the program. Copy text by selecting it instead. |
| Hugging Face download stops with HTTP 429 | Set `HF_TOKEN`. The script waits and retries by itself anyway. |
| Validation QWK much higher than test QWK | Expected: the DDR test split has a different camera mix (0.834 vs 0.682 for the centralised model). Report test results. |

---

## 11. Picking the work up again

A good order:

1. Read [LIMITATIONS_AND_FUTURE_WORK.md](LIMITATIONS_AND_FUTURE_WORK.md) and pick one direction.
2. Rebuild the environment (section 3) and check that `make_tables.py` reproduces
   `results/tables/` exactly. Then you know the setup is right.
3. Get the data back (section 4) and restore `best_centralised.pth` (section 5).
4. Do a two-round smoke test (`--rounds 2 --tag smoke`), then add your method (section 6).
5. Keep using three seeds and the test split for the final numbers, as in the paper.
