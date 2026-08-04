"""
strategy_ordinalfed.py — OrdinalFed FL aggregation strategy

The novel contribution:
  1. Evaluate each client on server validation set V using QWK
  2. Clip QWK scores: κ_k = max(κ_k, 0.01)
  3. Normalise: w_k = κ_k / Σκ_j
  4. Adaptive β cosine schedule: β(t) * size_weight + (1-β(t)) * qwk_weight
  5. Weighted average of client parameters

TODO (Day 5): Implement OrdinalFedStrategy
"""

# TODO: Day 5 - Tasks 5.1, 5.2, 5.3
