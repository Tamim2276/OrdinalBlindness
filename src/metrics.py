import numpy as np
from sklearn.metrics import cohen_kappa_score, classification_report, accuracy_score
import warnings


def compute_qwk(y_true, y_pred):
    """
    Computes Quadratic Weighted Kappa (QWK).
    Primary metric for OrdinalFed — penalizes severe misdiagnoses heavily.

    Score interpretation:
        1.0  = Perfect agreement
        0.0  = No better than random
        <0.0 = Worse than random
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # FIX 1: Guard against empty or mismatched inputs
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples, "
            f"y_pred has {len(y_pred)} samples."
        )

    # FIX 2: QWK is undefined if only one unique class exists in y_true
    # (sklearn raises a division error in this case)
    if len(np.unique(y_true)) < 2:
        warnings.warn(
            "QWK is undefined when y_true contains only one class. Returning 0.0.",
            RuntimeWarning
        )
        return 0.0

    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def compute_accuracy(y_true, y_pred):
    """
    Computes standard accuracy.
    Tracked for comparison against older papers, but QWK is the primary metric
    since class imbalance makes accuracy alone misleading for DR grading.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples, "
            f"y_pred has {len(y_pred)} samples."
        )

    return accuracy_score(y_true, y_pred)


def per_grade_metrics(y_true, y_pred):
    """
    Returns a per-grade breakdown of precision, recall, F1, and support.
    Useful for detecting if the model ignores minority classes (Grade 1, Grade 3).

    Returns a dict keyed by grade name, plus macro/weighted averages.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples, "
            f"y_pred has {len(y_pred)} samples."
        )

    report = classification_report(
        y_true, y_pred,
        labels=[0, 1, 2, 3, 4],
        target_names=["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"],
        output_dict=True,
        zero_division=0
    )

    # FIX 3: Warn if any grade has very low support (common in federated clients)
    # Low support grades will have unreliable F1 scores — important to flag
    for grade in ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"]:
        support = report[grade]["support"]
        if support < 10:
            warnings.warn(
                f"{grade} has very low support ({support} samples). "
                f"F1/precision/recall for this grade may be unreliable.",
                RuntimeWarning
            )

    return report


def summarize_metrics(y_true, y_pred):
    """
    FIX 4: Convenience function that computes and prints all metrics in one call.
    Useful at the end of each federated round for a full evaluation snapshot.
    """
    qwk = compute_qwk(y_true, y_pred)
    acc = compute_accuracy(y_true, y_pred)
    report = per_grade_metrics(y_true, y_pred)

    print(f"  QWK      : {qwk:.4f}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Per-Grade F1:")
    for grade in ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"]:
        f1      = report[grade]['f1-score']
        support = report[grade]['support']
        print(f"    {grade}: F1={f1:.3f}  (n={support})")

    return {"qwk": qwk, "accuracy": acc, "per_grade": report}   