"""Causal temporal validation harness with paired tests and bootstrap confidence intervals."""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pandas as pd

from src.evaluation.metrics import rmsle, test_weighted_rmsle
from src.config import Config


class CausalValidator:
    """Rigorous causal evaluation harness preventing look-ahead bias and seed noise illusions."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

    @staticmethod
    def paired_delta(
        preds_a: Sequence[np.ndarray],
        preds_b: Sequence[np.ndarray],
        targets: np.ndarray,
        seen_mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Compute paired seed difference where shared seed variance cancels out."""
        assert len(preds_a) == len(preds_b), "Paired comparison requires matching seed counts."
        
        eval_fn = (
            (lambda p: test_weighted_rmsle(p, targets, seen_mask))
            if seen_mask is not None
            else (lambda p: rmsle(p, targets))
        )
        deltas = np.array([eval_fn(b) - eval_fn(a) for a, b in zip(preds_a, preds_b)])
        se = float(deltas.std(ddof=1) / np.sqrt(len(deltas))) if len(deltas) > 1 else float("nan")

        return {
            "mean_delta": float(deltas.mean()),
            "standard_error": se,
            "per_seed_deltas": deltas.tolist(),
            "is_significant": bool(len(deltas) > 1 and abs(deltas.mean()) > 2.0 * se),
        }

    @staticmethod
    def bootstrap_ci(
        pred_a: np.ndarray,
        pred_b: np.ndarray,
        targets: np.ndarray,
        transformers: Sequence[str] | pd.Series,
        seen_mask: np.ndarray | None = None,
        n_resamples: int = 400,
        seed: int = 42,
        unseen_share: float = 0.2216,
    ) -> dict[str, Any]:
        """Compute 95% bootstrap confidence interval on difference by resampling transformers."""
        rng = np.random.default_rng(seed)
        t_series = pd.Series(transformers)
        group_indices = {k: v.values for k, v in t_series.groupby(t_series).groups.items()}
        unique_keys = np.array(list(group_indices.keys()))

        eval_fn = (
            (lambda p, idx: test_weighted_rmsle(p[idx], targets[idx], seen_mask[idx], unseen_share))
            if seen_mask is not None
            else (lambda p, idx: rmsle(p[idx], targets[idx]))
        )

        diffs = []
        for _ in range(n_resamples):
            sampled_keys = rng.choice(unique_keys, size=len(unique_keys), replace=True)
            row_mask = np.concatenate([group_indices[k] for k in sampled_keys])
            diffs.append(eval_fn(pred_b, row_mask) - eval_fn(pred_a, row_mask))

        lo, hi = np.percentile(diffs, [2.5, 97.5])
        return {
            "mean": float(np.mean(diffs)),
            "ci_lower": float(lo),
            "ci_upper": float(hi),
            "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        }
