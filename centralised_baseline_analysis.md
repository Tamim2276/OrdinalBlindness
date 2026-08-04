# Centralised Baseline — Results Analysis

> **Script:** `experiments/run_centralised.py`
> **Model:** EfficientNet-B3 | **Dataset:** DDR | **Epochs:** 30

---

## Key Numbers

| Metric | Value | Target | Status |
|---|---|---|---|
| Best Val QWK | **0.8494** | ≥ 0.83 | ✅ Passed |
| Best Val Acc | **80.7%** | — | ✅ Healthy |
| Best Epoch | **21 / 30** | — | ✅ Not last epoch |
| Train Loss (final) | **0.011** | — | ⚠️ See Overfitting |

---

## What's Going Well

### QWK Trajectory — Clean and Healthy

```
Epoch  1:  0.7229
Epoch  4:  0.8102  ← crossed 0.80 early
Epoch 10:  0.8362
Epoch 13:  0.8416
Epoch 21:  0.8494  ← peak (best checkpoint saved)
```

Steady improvement with no catastrophic drops — the cosine LR schedule is
working exactly as intended.

### Per-Grade Performance at Best Epoch (Epoch 21)

| Grade | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Grade 0 | 0.885 | 0.957 | 0.919 | 1253 |
| Grade 1 | 0.172 | 0.079 | 0.109 | 126 |
| Grade 2 | 0.810 | 0.728 | 0.767 | 895 |
| Grade 3 | 0.210 | 0.468 | 0.289 | 47 |
| Grade 4 | 0.761 | 0.753 | 0.757 | 182 |

**Grade 0** and **Grade 2** are well-learned (the two dominant classes). Grade 4
is also solid. These three grades account for ~88% of the validation set.

---

## ⚠️ One Real Concern — Overfitting

This is the most important pattern to notice:

```
Epoch 11:  Train=0.117   Val=0.965   ← gap starts opening
Epoch 13:  Train=0.068   Val=0.884
Epoch 16:  Train=0.052   Val=0.984
Epoch 21:  Train=0.023   Val=0.981
Epoch 30:  Train=0.011   Val=1.061   ← train near zero, val still ~1.0
```

Train loss collapsed to near zero (0.011) while val loss stayed around 1.0+ —
that is a **10× gap**, which is classic overfitting. The model has memorised
the training set.

The reason QWK stayed reasonable despite this is that QWK is more robust to
overconfident predictions than cross-entropy loss is.

### Why It's Acceptable Here

This is **expected and acceptable** for the centralised baseline — it sets the
ceiling. But it carries an important implication for Day 3:

> **Local FL client training must use fewer epochs (E=5 as planned) to avoid
> the same collapse per round.**

If clients overfit locally before aggregation, the global model diverges faster
under Non-IID conditions — exactly the problem OrdinalFed is designed to fix.

---

## ⚠️ One Clinical Concern — Grade 1 is Weak

```
Grade 1:  P=0.172   R=0.079   F1=0.109   (n=126)
```

Recall of **7.9%** means the model is missing **92% of Grade 1 cases** —
catching only ~10 out of 126 validation samples. Grade 1 (mild NPDR) is the
hardest class clinically, and with only 126 validation samples it is severely
underrepresented even with a class weight of 5.0.

### Why This Happens

- Grade 1 is **genuinely ambiguous** — even expert graders disagree on mild NPDR
- The DDR dataset has very few Grade 1 samples (~2.5% of total)
- Grade 1 images look visually similar to Grade 0 at low severity

### What to Note in the Paper

This is a known DDR dataset limitation. It can be framed as motivation for
OrdinalFed: in a federated setting with skewed clients, Grade 1 detection
degrades even further because most clients see almost none of these cases.
OrdinalFed's QWK-weighted aggregation should help by up-weighting clients that
happen to have better Grade 1 coverage.

---

## What This Means for the Paper

```
Centralised Baseline:  QWK = 0.8494
```

This becomes the **ceiling row** in the main results table. Every FL method is
compared against this number.

### Expected FL Performance Ranges

| Condition | Expected QWK | Notes |
|---|---|---|
| FL IID | ~0.83 – 0.84 | Within ~1–2% of centralised |
| FL α=1.0 | ~0.80 – 0.83 | Mildly degraded |
| FL α=0.5 | ~0.76 – 0.80 | Noticeable drop |
| FedAvg α=0.1 | ~0.70 – 0.76 | Should fail gate (< 0.80) |
| OrdinalFed α=0.1 | > 0.80 | Paper's headline claim |

---

## Verdict

```
✅ Gate passed — safe to proceed to Day 3 (FedAvg)
✅ Ceiling established: QWK = 0.8494
✅ Cosine LR schedule working as intended
✅ Overfitting present but expected for centralised baseline
⚠️ Grade 1 recall (7.9%) — note in paper, monitor across all FL experiments
⚠️ Keep FL local epochs at E=5 to avoid same overfitting pattern per round
```
