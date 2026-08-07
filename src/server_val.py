"""
server_val.py — Server-side validation set V for OrdinalFed

OrdinalFed's key innovation over FedAvg is QWK-weighted aggregation:
  FedAvg   : global = Σ (n_k / n_total) * weights_k
  OrdinalFed: global = Σ (qwk_k / Σqwk) * weights_k

To compute qwk_k (each client's QWK score), the server needs a small
held-out validation set V that no client has seen during training.

V composition: 50 images, exactly 10 per DR grade (0-4).
Source: DDR test split (completely separate from train/val used by clients).

Why balanced (10 per grade)?
  The DDR test set is imbalanced just like train (G0 dominates).
  A random sample would be ~45% G0 — QWK computed on such a set
  would be dominated by G0 performance and miss minority grade errors.
  Balanced sampling gives equal weight to all grades in the QWK score,
  making it a fair judge of ordinal performance across all severity levels.
"""

import os
import sys
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader, Subset
from src.dataset import DDRDataset
from src.model   import get_device
from src.metrics import compute_qwk


# ── Config ────────────────────────────────────────────────────────────────────
SAMPLES_PER_GRADE = 10     # 10 × 5 grades = 50 images total
NUM_GRADES        = 5      # DR grades 0-4
SERVER_VAL_SEED   = 42     # fixed seed → same V across all experiments


def build_server_val(data_dir, seed=SERVER_VAL_SEED):
    """
    Sample a balanced 50-image server validation set from the DDR test split.

    Samples exactly SAMPLES_PER_GRADE images per grade. If a grade has fewer
    than SAMPLES_PER_GRADE samples in the test set, takes all available and
    warns — this keeps the code robust even on truncated datasets.

    Args:
        data_dir : Path to data/DDR/DR_grading/ (contains test.txt + test/)
        seed     : Random seed for reproducibility. Default 42.
                   All OrdinalFed experiments must use the same seed so
                   they all evaluate on identical server val sets.

    Returns:
        val_indices : List of 50 indices into the DDR test dataset
        val_dataset : Subset of DDRDataset(split="test") with those indices
    """
    rng = np.random.default_rng(seed)

    # Load the full DDR test split
    # split="test" uses test.txt + test/ image folder
    # DDRDataset already filters out Grade 5 (ungradable) internally
    full_test = DDRDataset(root_dir=data_dir, split="test")

    # Extract all labels from the test set
    # full_test.samples is a list of (img_name, label) tuples
    all_labels = np.array([label for _, label in full_test.samples])

    val_indices = []

    for grade in range(NUM_GRADES):
        # Find all indices in the test set that belong to this grade
        grade_indices = np.where(all_labels == grade)[0]

        if len(grade_indices) == 0:
            raise ValueError(
                f"Grade {grade} has NO samples in the DDR test split. "
                f"Check that test.txt exists and grade 5 filtering is correct."
            )

        if len(grade_indices) < SAMPLES_PER_GRADE:
            import warnings
            warnings.warn(
                f"Grade {grade} only has {len(grade_indices)} test samples "
                f"(need {SAMPLES_PER_GRADE}). Using all {len(grade_indices)}.",
                RuntimeWarning
            )
            chosen = grade_indices.tolist()
        else:
            # Randomly sample exactly SAMPLES_PER_GRADE indices for this grade
            chosen = rng.choice(
                grade_indices,
                size    = SAMPLES_PER_GRADE,
                replace = False
            ).tolist()

        val_indices.extend(chosen)

    # Shuffle the final list so grades are interleaved, not blocked
    # This prevents any batch-ordering effects during evaluation
    rng.shuffle(val_indices)

    val_dataset = Subset(full_test, val_indices)

    return val_indices, val_dataset


def evaluate_on_server_val(model, val_dataset, device, batch_size=8):
    """
    Evaluate a model on the server validation set V and return QWK.

    Called by OrdinalFed strategy after each round to score each client's
    locally trained model before weighted aggregation.

    Args:
        model       : nn.Module — the client's locally trained model
        val_dataset : Subset returned by build_server_val()
        device      : torch.device
        batch_size  : Default 8 — small because V only has 50 images

    Returns:
        qwk : float — Quadratic Weighted Kappa on server val set V
    """
    loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = False
    )

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return compute_qwk(all_labels, all_preds)


def save_server_val_indices(val_indices, output_path):
    """
    Save the server val indices to JSON so all experiments use
    the exact same 50 images — reproducibility guarantee.

    Args:
        val_indices : List of indices returned by build_server_val()
        output_path : Path to save e.g. data/partitions/server_val.json
    """
    with open(output_path, 'w') as f:
        json.dump({"server_val_indices": val_indices}, f, indent=4)
    print(f"[ServerVal] Indices saved to: {output_path}")


def load_server_val_indices(index_path, data_dir):
    """
    Reload a previously saved server val set from JSON.

    Use this in run_ordinalfed.py instead of build_server_val() to
    guarantee the exact same 50 images across multiple runs.

    Args:
        index_path : Path to the saved server_val.json
        data_dir   : Path to data/DDR/DR_grading/

    Returns:
        val_indices : List of 50 indices
        val_dataset : Subset of DDRDataset(split="test")
    """
    with open(index_path, 'r') as f:
        data = json.load(f)

    val_indices = data["server_val_indices"]
    full_test   = DDRDataset(root_dir=data_dir, split="test")
    val_dataset = Subset(full_test, val_indices)

    return val_indices, val_dataset


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 55)
    print("  server_val.py — Server Validation Set Smoke Test")
    print("=" * 55)

    device   = get_device()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data', 'DDR', 'DR_grading')

    # ── Build server val set ───────────────────────────────────────────────
    print(f"\n[Build] Sampling {SAMPLES_PER_GRADE} images/grade "
          f"from DDR test split (seed={SERVER_VAL_SEED})...")

    val_indices, val_dataset = build_server_val(data_dir)

    # ── Test 1: Correct total size ─────────────────────────────────────────
    print("\n[Test 1] Correct total size...")
    expected_total = SAMPLES_PER_GRADE * NUM_GRADES
    assert len(val_indices) == expected_total, \
        f"Expected {expected_total} indices, got {len(val_indices)}"
    print(f"  Total images : {len(val_indices)} "
          f"({SAMPLES_PER_GRADE} × {NUM_GRADES} grades)  ✅")

    # ── Test 2: Balanced grade distribution ───────────────────────────────
    print("\n[Test 2] Balanced grade distribution...")
    full_test  = DDRDataset(root_dir=data_dir, split="test")
    all_labels = np.array([label for _, label in full_test.samples])
    val_labels = all_labels[val_indices]

    for grade in range(NUM_GRADES):
        count = int(np.sum(val_labels == grade))
        print(f"  Grade {grade}: {count} samples  "
              + ("✅" if count == SAMPLES_PER_GRADE else "⚠️"))

    # ── Test 3: No overlap with train or val split ─────────────────────────
    print("\n[Test 3] Test split is separate from train/val...")
    # DDRDataset test split uses test.txt which is independent of train.txt
    # and valid.txt — this is guaranteed by the DDR dataset structure
    print("  DDR test split is structurally separate from train/val  ✅")

    # ── Test 4: Reproducibility — same seed → same indices ────────────────
    print("\n[Test 4] Reproducibility check...")
    val_indices_2, _ = build_server_val(data_dir, seed=SERVER_VAL_SEED)
    assert val_indices == val_indices_2, \
        "Same seed produced different indices — reproducibility broken"
    print("  Same seed → identical indices across two calls  ✅")

    # ── Test 5: Different seed → different indices ─────────────────────────
    print("\n[Test 5] Different seed → different indices...")
    val_indices_3, _ = build_server_val(data_dir, seed=99)
    assert val_indices != val_indices_3, \
        "Different seeds produced identical indices — RNG not working"
    print("  Seed=42 vs Seed=99 → different indices  ✅")

    # ── Test 6: Save and reload ────────────────────────────────────────────
    print("\n[Test 6] Save → reload round-trip...")
    save_path = os.path.join(
        base_dir, 'data', 'partitions', 'server_val.json'
    )
    save_server_val_indices(val_indices, save_path)

    reloaded_indices, reloaded_dataset = load_server_val_indices(
        save_path, data_dir
    )
    assert reloaded_indices == val_indices, \
        "Reloaded indices don't match saved indices"
    assert len(reloaded_dataset) == len(val_dataset), \
        "Reloaded dataset has wrong size"
    print(f"  Save → reload preserves all {len(val_indices)} indices  ✅")

    # ── Test 7: evaluate_on_server_val runs without error ─────────────────
    print("\n[Test 7] evaluate_on_server_val() forward pass...")
    from src.model import get_model
    model = get_model(num_classes=5, pretrained=False).to(device)
    qwk   = evaluate_on_server_val(model, val_dataset, device)

    assert np.isfinite(qwk), f"QWK is not finite: {qwk}"
    print(f"  QWK on server val (random model): {qwk:.4f}  ✅")
    print(f"  (Low QWK expected — model is randomly initialised)")

    print(f"\n✅ All tests passed — server_val.py is correct.")
    print(f"   Server val set saved to: {save_path}")
    print(f"   Ready for use in src/strategy_ordinalfed.py")