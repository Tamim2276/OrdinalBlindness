# OrdinalFed — Federated Learning for Diabetic Retinopathy Grading

> **OrdinalFed** combines a distance-aware **OrdinalLoss** with adaptive **QWK-weighted aggregation** to improve stability and diagnostic quality of federated DR grading under Non-IID data distributions.

---

## Table of Contents
1. [Hardware & OS Requirements](#hardware--os-requirements)
2. [Step 1 — Clone the Repository](#step-1--clone-the-repository)
3. [Step 2 — Install Python & Create Virtual Environment](#step-2--install-python--create-virtual-environment)
4. [Step 3 — Install PyTorch (GPU-specific)](#step-3--install-pytorch-gpu-specific)
5. [Step 4 — Install Remaining Dependencies](#step-4--install-remaining-dependencies)
6. [Step 5 — Download the DDR Dataset](#step-5--download-the-ddr-dataset)
7. [Step 6 — Create Partitions (one-time)](#step-6--create-partitions-one-time)
8. [Step 7 — Train the Centralised Baseline](#step-7--train-the-centralised-baseline)
9. [Step 8 — Run Federated Experiments](#step-8--run-federated-experiments)
10. [Step 9 — Generate Figures](#step-9--generate-figures)
11. [Speed Tips — Completing Rounds Faster](#speed-tips--completing-rounds-faster)
12. [Project File Structure](#project-file-structure)
13. [Reproducing Exact Partitions from Git](#reproducing-exact-partitions-from-git)

---

## Hardware & OS Requirements

| | Minimum | Recommended |
|---|---|---|
| **GPU** | None (CPU only — very slow) | NVIDIA GPU with 8GB+ VRAM (CUDA) |
| **RAM** | 16 GB | 32 GB |
| **Storage** | 15 GB free | 25 GB free |
| **OS** | Windows 10/11, Linux, macOS | Windows 10/11 or Ubuntu 22.04 |
| **Python** | 3.10 | 3.10 |

> **Intel Arc users:** The code also supports Intel XPU (Arc B580 etc.) automatically. No extra steps needed.

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/OrdinalBlindness.git
cd OrdinalBlindness
```

---

## Step 2 — Install Python & Create Virtual Environment

The codebase requires **Python 3.10** exactly.

### Windows
```bash
# Check your python version first
python --version

# Create and activate the environment
python -m venv ordinalfed_env
ordinalfed_env\Scripts\activate
```

### Linux / macOS
```bash
python3.10 -m venv ordinalfed_env
source ordinalfed_env/bin/activate
```

---

## Step 3 — Install PyTorch (GPU-specific)

This is the **most important step** — install the correct PyTorch build for your hardware.

### Option A — NVIDIA GPU (CUDA 12.1)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Option B — NVIDIA GPU (CUDA 11.8)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Option C — Intel Arc GPU (XPU)
```bash
pip install torch==2.13.0+xpu torchvision==0.28.0+xpu torchaudio==2.11.0+xpu \
    --index-url https://download.pytorch.org/whl/xpu
```

### Option D — CPU only (slow — not recommended for full runs)
```bash
pip install torch torchvision torchaudio
```

> **Verify GPU is detected:**
> ```bash
> python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
> ```

---

## Step 4 — Install Remaining Dependencies

```bash
pip install timm flwr scikit-learn pandas matplotlib seaborn tqdm pillow
```

Or install everything at once (note: `requirements.txt` contains XPU-specific packages — install PyTorch manually first, then):

```bash
pip install timm flwr scikit-learn pandas matplotlib seaborn tqdm pillow numpy
```

---

## Step 5 — Download the DDR Dataset

The DDR dataset must be placed at:

```
OrdinalBlindness/
└── data/
    └── DDR/
        └── DR_grading/
            ├── train/          ← training images
            ├── valid/          ← validation images
            ├── test/           ← test images
            ├── train.txt       ← label file (image_name label)
            ├── valid.txt
            └── test.txt
```

**Download:**
1. Visit [DDR Dataset on Kaggle](https://www.kaggle.com/datasets/mariaherrerot/ddrdataset) or the [original source](https://github.com/nkicsl/DDR-dataset)
2. Download and extract into `data/DDR/DR_grading/`

**Verify the dataset structure:**
```bash
python -c "
from src.dataset import DDRDataset
d = DDRDataset('data/DDR/DR_grading', split='train')
print(f'Train: {len(d)} images')
d = DDRDataset('data/DDR/DR_grading', split='valid')
print(f'Valid: {len(d)} images')
d = DDRDataset('data/DDR/DR_grading', split='test')
print(f'Test : {len(d)} images')
"
```

Expected output:
```
Train: 6260 images
Valid: 2503 images
Test : 2514 images
```

---

## Step 6 — Create Partitions (one-time)

This creates the 5 client data splits and the server validation set used across all experiments. **Run once — never re-run unless you want to start over.**

```bash
# Create all 5 partition JSON files (iid, dirichlet_0.1, dirichlet_0.5, dirichlet_1.0, real_site)
python src/partition.py

# Create the 50-image balanced server validation set for OrdinalFed
python src/server_val.py
```

This creates:
```
data/partitions/
├── iid.json
├── dirichlet_0.1.json
├── dirichlet_0.5.json
├── dirichlet_1.0.json
├── real_site.json
└── server_val.json
```

> ⚠️ **Reproducibility:** The partition files use `np.random.seed(42)`. As long as the DDR dataset is identical and you run `partition.py` without changes, the partitions will always be the same. They are also committed to Git — so pulling the repo is enough to get the same partitions without re-running this step.

---

## Step 7 — Train the Centralised Baseline

All FL experiments **warm-start** from this checkpoint. Run this first.

```bash
python experiments/run_centralised.py
```

- Trains EfficientNet-B3 for up to 30 epochs with early stopping
- Saves the best model to `results/best_centralised.pth`
- Expected QWK: **~0.849**
- Expected time: 2–6 hours depending on GPU

---

## Step 8 — Run Federated Experiments

Run **one command per session** (overnight). All experiments warm-start from `results/best_centralised.pth`.

### FedAvg Baseline

```bash
# IID partition (10 rounds)
python experiments/run_fedavg.py --partition iid

# Dirichlet Non-IID α=0.1 (30 rounds — captures full collapse arc)
python experiments/run_fedavg.py --partition dirichlet_0.1

# Real-site clinical partition (10 rounds)
python experiments/run_fedavg.py --partition real_site
```

### FedProx Baseline

```bash
# IID partition (10 rounds)
python experiments/run_fedprox.py --partition iid --mu 0.01

# Dirichlet Non-IID α=0.1 (30 rounds)
python experiments/run_fedprox.py --partition dirichlet_0.1 --mu 0.01

# Real-site partition (10 rounds)
python experiments/run_fedprox.py --partition real_site --mu 0.01
```

### OrdinalFed (Paper Contribution)

```bash
# IID partition (10 rounds)
python experiments/run_ordinalfed.py --partition iid

# Dirichlet Non-IID α=0.1 (30 rounds)
python experiments/run_ordinalfed.py --partition dirichlet_0.1

# Real-site partition (10 rounds)
python experiments/run_ordinalfed.py --partition real_site
```

### Chain all runs in one overnight session (Linux/macOS)

```bash
python experiments/run_fedavg.py --partition iid && \
python experiments/run_fedavg.py --partition dirichlet_0.1 && \
python experiments/run_fedavg.py --partition real_site && \
python experiments/run_fedprox.py --partition iid --mu 0.01 && \
python experiments/run_fedprox.py --partition dirichlet_0.1 --mu 0.01 && \
python experiments/run_fedprox.py --partition real_site --mu 0.01 && \
python experiments/run_ordinalfed.py --partition iid && \
python experiments/run_ordinalfed.py --partition dirichlet_0.1 && \
python experiments/run_ordinalfed.py --partition real_site
```

### Chain on Windows (PowerShell)

```powershell
python experiments/run_fedavg.py --partition iid; `
python experiments/run_fedavg.py --partition dirichlet_0.1; `
python experiments/run_fedavg.py --partition real_site; `
python experiments/run_fedprox.py --partition iid --mu 0.01; `
python experiments/run_fedprox.py --partition dirichlet_0.1 --mu 0.01; `
python experiments/run_fedprox.py --partition real_site --mu 0.01; `
python experiments/run_ordinalfed.py --partition iid; `
python experiments/run_ordinalfed.py --partition dirichlet_0.1; `
python experiments/run_ordinalfed.py --partition real_site
```

Results are saved incrementally (crash-safe) to `results/*.csv`.

---

## Step 9 — Generate Figures

After all experiments complete:

```bash
python experiments/generate_figures.py
```

Generates the following figures in `figures/`:
- `fig1_qwk_dirichlet01.pdf`
- `fig2_qwk_realsite.pdf`
- `fig3_qwk_iid.pdf`
- `fig4_qwk_combined.pdf`
- `fig5_distributions.pdf`

---

## Speed Tips — Completing Rounds Faster

These changes improve throughput **without affecting model quality**:

### 1. Use `pin_memory=True` (automatic on CUDA)
Already enabled automatically on CUDA by the code. No action needed.

### 2. Increase `num_workers` for DataLoader (CUDA only)
On CUDA, you can safely use multiple workers. Edit `src/client.py`:
```python
num_workers = 4  # or 8, depending on your CPU cores
```
> ⚠️ **Do NOT do this on Windows + XPU or CPU** — it causes memory explosion.

### 3. Increase batch size if VRAM allows
Edit `BATCH_SIZE` in `src/client.py`. Doubling the batch size roughly halves the number of gradient steps per epoch:
```python
BATCH_SIZE = 128   # default is 64 — double if GPU has ≥ 16GB VRAM
```

### 4. Compile the model with `torch.compile` (PyTorch 2.0+, CUDA only)
In `experiments/run_fedavg.py` (and the other experiment files), after loading the checkpoint, add:
```python
# After: init_model, meta = load_checkpoint(...)
import torch._dynamo
torch._dynamo.config.suppress_errors = True
```
Then wrap the model in each client:
```python
self.model = torch.compile(self.model)   # in client.py __init__
```
> First run will be slow (compilation). All subsequent rounds are 20–30% faster.

### 5. Use AMP GradScaler on CUDA for float16
Mixed precision (`float16`) is already enabled on CUDA via `get_autocast_context()`. For maximum throughput, you can additionally wrap with `GradScaler` (prevents float16 underflow). This is an advanced optimisation — only add if you see NaN losses at high batch sizes.

---

## Project File Structure

```
OrdinalBlindness/
│
├── data/
│   ├── DDR/DR_grading/          ← DDR dataset (you download this)
│   └── partitions/              ← Auto-generated by src/partition.py
│       ├── iid.json
│       ├── dirichlet_0.1.json
│       ├── dirichlet_0.5.json
│       ├── dirichlet_1.0.json
│       ├── real_site.json
│       └── server_val.json
│
├── src/
│   ├── model.py                 ← EfficientNet-B3 + CUDA/XPU/CPU device selection
│   ├── dataset.py               ← DDRDataset / APTOSDataset loaders
│   ├── partition.py             ← IID / Dirichlet / Real-site partitioning (seed=42)
│   ├── client.py                ← FedAvg FL client (DRClient)
│   ├── client_fedprox.py        ← FedProx FL client (DRFedProxClient)
│   ├── strategy_ordinalfed.py   ← OrdinalFed client + QWK-weighted aggregation
│   ├── server_val.py            ← 50-image balanced server validation set V
│   ├── losses.py                ← OrdinalLoss definition
│   └── metrics.py               ← QWK, accuracy
│
├── experiments/
│   ├── run_centralised.py       ← Step 1: train centralised baseline
│   ├── run_fedavg.py            ← Step 2: FedAvg baseline
│   ├── run_fedprox.py           ← Step 3: FedProx baseline
│   ├── run_ordinalfed.py        ← Step 4: OrdinalFed (paper contribution)
│   └── generate_figures.py      ← Generate all paper figures
│
├── results/                     ← Auto-created. Checkpoints + CSVs saved here.
├── figures/                     ← Auto-created. PDFs saved here.
├── paper/                       ← LaTeX paper source
└── requirements.txt
```

---

## Reproducing Exact Partitions from Git

The partition JSON files are **committed to the repository**. When you clone or pull, you get the exact same partitions that were used to produce the paper results — no need to re-run `partition.py`.

If you need to regenerate them from scratch (e.g., on a different dataset version), the seed is hardcoded:

```python
# src/partition.py line ~177
np.random.seed(42)
```

This guarantees that `dirichlet_0.1.json` produced on any machine will have:
- Client 0: 32 images (100% Grade 0)
- Client 4: 2,087 images (98% Grade 2)

These are the exact splits reported in the paper.
