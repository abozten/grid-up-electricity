"""Pure mathematical evaluation metrics for RMSLE optimization."""
from __future__ import annotations

import numpy as np


def rmsle(log_preds: np.ndarray, log_targets: np.ndarray) -> float:
    """Calculate Root Mean Squared Logarithmic Error (evaluated directly on log-space arrays)."""
    p = np.asarray(log_preds, dtype=float)
    y = np.asarray(log_targets, dtype=float)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def test_weighted_rmsle(
    log_preds: np.ndarray,
    log_targets: np.ndarray,
    seen_mask: np.ndarray,
    unseen_share: float = 0.2216,
) -> float:
    """Compute RMSLE re-weighted to match test set's known unseen/seen transformer proportion."""
    p = np.asarray(log_preds, dtype=float)
    y = np.asarray(log_targets, dtype=float)
    s = seen_mask.astype(bool)
    u = ~s

    if s.sum() == 0 or u.sum() == 0:
        return rmsle(p, y)

    seen_mse = np.mean((p[s] - y[s]) ** 2)
    unseen_mse = np.mean((p[u] - y[u]) ** 2)
    weighted_mse = (1.0 - unseen_share) * seen_mse + unseen_share * unseen_mse
    return float(np.sqrt(weighted_mse))
