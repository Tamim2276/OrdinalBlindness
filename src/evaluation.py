"""
evaluation.py — Global-model evaluation for the revision experiments

Reviewer #2 asked for independent validation data and class-wise analysis.
Every round, the global model is now scored on:
  - the DDR validation split (2,503 images) — used for model selection
  - the DDR test split minus server set V (2,464 images) — never used for
    training, aggregation or selection, so it is an unbiased estimate
and the full confusion matrix is stored, from which per-grade recall,
precision and F1 are derived later (experiments/make_tables.py).
"""

import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix

from src.dataset import DDRDataset
from src.model   import get_autocast_context
from src.metrics import compute_qwk, compute_accuracy

NUM_GRADES = 5


def load_test_set(data_dir, server_val_path):
    """DDR test split without the 50 images of server set V."""
    with open(server_val_path, 'r') as f:
        server_val = set(json.load(f)["server_val_indices"])
    full_test = DDRDataset(root_dir=data_dir, split="test")
    keep      = [i for i in range(len(full_test)) if i not in server_val]
    return Subset(full_test, keep)


def make_eval_loader(dataset, device, batch_size=64):
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = (device.type == "cuda"),
    )


def predict(model, loader, device, criterion=None):
    """
    Run inference over a loader.

    Returns:
        labels, preds : NumPy int arrays
        avg_loss      : mean batch loss under `criterion`, or None
    """
    model.eval()
    all_labels, all_preds = [], []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with get_autocast_context(device):
                outputs = model(images)
                if criterion is not None:
                    running_loss += criterion(outputs, labels).item()
            all_preds.append(outputs.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / len(loader) if criterion is not None and len(loader) else None
    return np.concatenate(all_labels), np.concatenate(all_preds), avg_loss


def summarise(labels, preds):
    """QWK, accuracy and the 5×5 confusion matrix (rows = true grade)."""
    return {
        "qwk": float(compute_qwk(labels, preds)),
        "acc": float(compute_accuracy(labels, preds)),
        "cm" : confusion_matrix(labels, preds, labels=list(range(NUM_GRADES))).tolist(),
    }


def per_grade_from_cm(cm):
    """Per-grade precision, recall and F1 from a confusion matrix (rows = true)."""
    cm        = np.asarray(cm, dtype=float)
    tp        = np.diag(cm)
    support   = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall    = np.divide(tp, support,   out=np.zeros_like(tp), where=support > 0)
    denom     = precision + recall
    f1        = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {"precision": precision, "recall": recall, "f1": f1, "support": support}
