"""
strategy_accwt.py — Accuracy-weighted FL aggregation (AccWt baseline)

Baseline that uses validation accuracy (instead of dataset size or QWK)
as aggregation weights. Isolates ordinal awareness from general
performance weighting.

AccWt is OrdinalFed's aggregation with one change: each client is scored
on server set V by plain accuracy instead of QWK. Clients train with
class-weighted cross entropy (no OrdinalLoss), like the server-side
validation-scoring methods in the literature (e.g. Bejenar et al., 2023).

Comparisons it enables:
    AccWt      vs  FedAvg      → does quality weighting help at all?
    AccWt      vs  abl_qwkagg  → does an ordinal score (QWK) beat accuracy?
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategy_ordinalfed import ordinalfed_aggregate


def accwt_aggregate(all_weights, all_n_samples, acc_scores, beta):
    """Same softmax / β(t) / floor scheme as OrdinalFed, applied to accuracy on V."""
    return ordinalfed_aggregate(all_weights, all_n_samples, acc_scores, beta)
