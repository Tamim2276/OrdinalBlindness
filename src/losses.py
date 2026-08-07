"""
losses.py — Ordinal loss function for OrdinalFed

Standard CrossEntropyLoss treats all misclassifications equally:
    Predicting G0 when true is G4 → same penalty as predicting G3 when true is G4

This is clinically wrong for DR grading. Missing severe DR (G4) by 4 grades
is far more dangerous than missing it by 1 grade. A patient sent home when
they need urgent treatment is a catastrophic error.

OrdinalLoss fixes this by adding a distance-squared penalty on top of
cross entropy:

    L = (1 - λ) × CrossEntropy(logits, targets)
      +      λ  × OrdinalPenalty(probs, targets)

Where OrdinalPenalty = Σ_i  (predicted_grade_i - true_grade_i)²  × p(grade_i)

The λ (lam) parameter controls the trade-off:
    lam = 0.0  → pure CrossEntropy (same as FedAvg baseline)
    lam = 0.5  → equal blend (default for OrdinalFed)
    lam = 1.0  → pure ordinal penalty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalLoss(nn.Module):
    """
    Hybrid loss = (1-λ) × CrossEntropy  +  λ × OrdinalPenalty

    OrdinalPenalty computes the expected squared distance between the
    predicted grade distribution and the true grade for each sample,
    then averages across the batch.

    Args:
        num_classes   : Number of DR grades (5 for grades 0-4)
        lam           : Weight on ordinal penalty term. Default 0.5.
                        lam=0.0 → pure CE, lam=1.0 → pure ordinal.
        class_weights : Optional tensor of per-class loss weights.
                        Same weights as used in centralised baseline
                        to handle class imbalance.
    """

    def __init__(self, num_classes=5, lam=0.5, class_weights=None):
        super().__init__()

        self.num_classes = num_classes
        self.lam         = lam

        # CrossEntropy with optional class weights for imbalance handling
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

        # Grade index vector [0, 1, 2, 3, 4] — used in penalty computation
        # Registered as a buffer so it moves to the correct device with .to(device)
        # without being treated as a learnable parameter
        self.register_buffer(
            "grade_indices",
            torch.arange(num_classes, dtype=torch.float32)
        )

    def forward(self, logits, targets):
        """
        Compute hybrid ordinal loss.

        Args:
            logits  : Raw model outputs, shape (batch_size, num_classes)
                      NOT softmaxed — CrossEntropy applies softmax internally
            targets : Ground truth grade indices, shape (batch_size,)
                      dtype=torch.long, values in {0, 1, 2, 3, 4}

        Returns:
            loss : Scalar tensor — weighted combination of CE + ordinal penalty

        Example:
            logits  = tensor([[2.1, 0.3, 0.1, 0.0, 0.0],   ← model thinks G0
                               [0.1, 0.2, 0.1, 1.9, 0.5]])  ← model thinks G3
            targets = tensor([4, 1])                          ← true: G4, G1

            Sample 1: predicting G0 when true=G4 → penalty = (0-4)²=16  ← HIGH
            Sample 2: predicting G3 when true=G1 → penalty = (3-1)²=4   ← medium
        """
        # ── Cross entropy term ────────────────────────────────────────────────
        ce_loss = self.ce(logits, targets)

        # ── Ordinal penalty term ──────────────────────────────────────────────
        # Step 1: Convert logits → probability distribution over grades
        # probs shape: (batch_size, num_classes)
        # Each row sums to 1.0: probs[i] = [P(G0|x_i), P(G1|x_i), ..., P(G4|x_i)]
        probs = F.softmax(logits, dim=1)

        # Step 2: Compute expected predicted grade for each sample
        # grade_indices = [0, 1, 2, 3, 4]
        # expected_grade[i] = Σ_k  k × P(grade=k | x_i)
        # Shape: (batch_size,)
        # Example: probs[i]=[0.7,0.2,0.1,0,0] → expected = 0×0.7+1×0.2+2×0.1 = 0.4
        expected_grade = (probs * self.grade_indices).sum(dim=1)

        # Step 3: Compute squared distance between expected and true grade
        # targets cast to float for subtraction
        # penalty[i] = (expected_grade[i] - true_grade[i])²
        # Shape: (batch_size,)
        true_grade = targets.float()
        penalty    = (expected_grade - true_grade) ** 2

        # Step 4: Average penalty across the batch
        ordinal_loss = penalty.mean()

        # ── Combine ───────────────────────────────────────────────────────────
        # lam=0.5: equal contribution from both terms
        # The CE term keeps gradients stable and drives class separation
        # The ordinal term adds distance awareness on top
        loss = (1.0 - self.lam) * ce_loss + self.lam * ordinal_loss

        return loss


def get_ordinal_loss(lam=0.5, device=None):
    """
    Convenience factory — returns a ready-to-use OrdinalLoss with the
    same class weights as the centralised baseline.

    Args:
        lam    : Ordinal penalty weight. Default 0.5.
        device : torch.device. If None, uses CPU.

    Returns:
        OrdinalLoss instance moved to the correct device.

    Usage:
        criterion = get_ordinal_loss(lam=0.5, device=device)
        loss = criterion(logits, labels)
    """
    # Same class weights used throughout all experiments for consistency
    # DDR imbalance: G0:~46%  G1:~2.5%  G2:~34%  G3:~1.3%  G4:~9%
    class_weights = torch.tensor(
        [1.0, 5.0, 1.5, 10.0, 4.0],
        dtype=torch.float32
    )

    criterion = OrdinalLoss(
        num_classes   = 5,
        lam           = lam,
        class_weights = class_weights
    )

    if device is not None:
        criterion = criterion.to(device)

    return criterion


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from src.model import get_device

    print("=" * 55)
    print("  losses.py — OrdinalLoss Smoke Test")
    print("=" * 55)

    device = get_device()

    # ── Test 1: Basic forward pass ─────────────────────────────────────────
    print("\n[Test 1] Basic forward pass...")

    criterion = get_ordinal_loss(lam=0.5, device=device)
    logits    = torch.randn(8, 5, device=device, requires_grad=True)   # batch of 8, 5 classes
    targets   = torch.randint(0, 5, (8,), device=device)

    loss = criterion(logits, targets)
    assert loss.item() > 0,          "Loss must be positive"
    assert torch.isfinite(loss),     "Loss must be finite"
    assert loss.requires_grad,       "Loss must support backprop"
    print(f"  Loss value   : {loss.item():.4f}  ✅")
    print(f"  Requires grad: {loss.requires_grad}  ✅")

    # ── Test 2: Backprop works ─────────────────────────────────────────────
    print("\n[Test 2] Backprop through ordinal loss...")

    loss.backward()
    print(f"  Backward pass completed without error  ✅")

    # ── Test 3: lam=0.0 should equal plain CrossEntropy ───────────────────
    print("\n[Test 3] lam=0.0 should match plain CrossEntropy...")

    logits  = torch.randn(16, 5, device=device)
    targets = torch.randint(0, 5, (16,), device=device)

    class_weights = torch.tensor([1.0, 5.0, 1.5, 10.0, 4.0]).to(device)
    ce_only       = nn.CrossEntropyLoss(weight=class_weights)(logits, targets)
    ordinal_lam0  = get_ordinal_loss(lam=0.0, device=device)(logits, targets)

    assert torch.allclose(ce_only, ordinal_lam0, atol=1e-5), \
        f"lam=0.0 should equal CE: CE={ce_only:.4f}, Ordinal={ordinal_lam0:.4f}"
    print(f"  CE loss      : {ce_only.item():.4f}")
    print(f"  Ordinal lam=0: {ordinal_lam0.item():.4f}  ✅ identical")

    # ── Test 4: Ordinal penalty is distance-aware ──────────────────────────
    print("\n[Test 4] Ordinal penalty punishes large grade gaps more...")

    # Create two scenarios with identical confidence but different grade gaps:
    #   Scenario A: predicts G1 confidently, true=G0  (gap=1, small error)
    #   Scenario B: predicts G4 confidently, true=G0  (gap=4, severe error)
    lam1_criterion = get_ordinal_loss(lam=1.0, device=device)

    # Scenario A: high confidence on G1, true=G0
    logits_small_gap    = torch.tensor([[0.0, 10.0, 0.0, 0.0, 0.0]],
                                        device=device)
    targets_small_gap   = torch.tensor([0], device=device)

    # Scenario B: high confidence on G4, true=G0
    logits_large_gap    = torch.tensor([[0.0, 0.0, 0.0, 0.0, 10.0]],
                                        device=device)
    targets_large_gap   = torch.tensor([0], device=device)

    loss_small = lam1_criterion(logits_small_gap, targets_small_gap)
    loss_large = lam1_criterion(logits_large_gap, targets_large_gap)

    assert loss_large > loss_small, \
        f"Large gap should cost more: small={loss_small:.4f}, large={loss_large:.4f}"

    print(f"  Gap=1 (G1 pred, G0 true): loss = {loss_small.item():.4f}")
    print(f"  Gap=4 (G4 pred, G0 true): loss = {loss_large.item():.4f}  "
          f"({loss_large.item()/loss_small.item():.1f}× higher)  ✅")

    # ── Test 5: Lambda sweep ───────────────────────────────────────────────
    print("\n[Test 5] Lambda sweep (lam=0.0 → 1.0)...")

    logits  = torch.randn(16, 5, device=device)
    targets = torch.randint(0, 5, (16,), device=device)

    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        l = get_ordinal_loss(lam=lam, device=device)(logits, targets)
        print(f"  lam={lam:.2f} → loss={l.item():.4f}")

    print("\n✅ All tests passed — losses.py is correct.")
    print("   OrdinalLoss is ready for use in run_ordinalfed.py")