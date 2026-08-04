"""
losses.py — Loss functions including ordinal penalty loss

The ordinal penalty loss augments standard cross-entropy with a
quadratic severity penalty:
  L = L_CE + λ * Σ p_c * (c - y)²

This forces local models to penalise severe misclassifications
(Grade 0→4 costs 16×) more than minor ones (Grade 0→1 costs 1×).

TODO (Day 4): Implement ordinal_loss
"""

import torch
import torch.nn.functional as F


def ordinal_loss(logits, targets, lam=0.5):
    """
    Ordinal-aware loss = CrossEntropy + λ * quadratic ordinal penalty.

    Args:
        logits: model output (B, 5)
        targets: ground truth grades (B,) in {0,1,2,3,4}
        lam: penalty weight (default 0.5)

    Returns:
        Combined loss scalar
    """
    # Standard cross-entropy
    ce = F.cross_entropy(logits, targets)

    # Ordinal penalty: expected squared distance
    probs = F.softmax(logits, dim=1)  # (B, 5)
    grades = torch.arange(5, device=logits.device).float()  # [0,1,2,3,4]
    # (B, 5) * (B, 5) → (B,) → scalar
    penalty = (probs * (grades[None, :] - targets[:, None].float()) ** 2).sum(dim=1).mean()

    return ce + lam * penalty
