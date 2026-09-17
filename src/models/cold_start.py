"""Bayesian hierarchical shrinkage prior with Winsorization and Isotonic Regression for cold-start."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.config import ModelConfig


class ColdStartPrior:
    """Computes and applies Winsorized + Isotonic capacity shrinkage prior on unseen transformers."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.alpha: float = getattr(self.config, "cold_start_alpha", 0.8)
        self.k: float = getattr(self.config, "cold_start_k", 5.0)
        self.lo: float = 0.05
        self.hi: float = 0.95
        self.cells: dict[tuple[str, float], float] = {}
        self.bolge_lookup: dict[str, float] = {}
        self.global_mean: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> ColdStartPrior:
        """Fit empirical Bayes prior: winsorize cells at 5/95, shrink to bolge, then monotone in guc."""
        h = train_df[["bolge", "guc", "log_t"]].copy()
        
        # 1. Winsorize cell statistics at 5/95 within (bolge, guc) to remove dead-meter drag
        h["log_t"] = h.groupby(["bolge", "guc"], observed=True)["log_t"].transform(
            lambda s: s.clip(s.quantile(self.lo), s.quantile(self.hi)) if len(s) > 1 else s
        )
        
        g = h.groupby(["bolge", "guc"], observed=True)["log_t"].agg(["mean", "size"])
        im = h.groupby("bolge", observed=True)["log_t"].mean()
        self.global_mean = float(h["log_t"].mean())

        bolge_indices = g.index.get_level_values(0)
        im_vals = im.reindex(bolge_indices).values

        # 2. Empirical Bayes shrinkage toward bolge parent mean
        g["s"] = (g["mean"] * g["size"] + im_vals * self.k) / (g["size"] + self.k)

        df = g["s"].rename("v").reset_index()
        df["n"] = g["size"].values

        # 3. Isotonic regression in capacity (guc) within each bolge weighted by cell count
        self.cells = {}
        for b, s in df.groupby("bolge", observed=True):
            s = s.sort_values("guc")
            if len(s) < 3:
                for _, r in s.iterrows():
                    self.cells[(str(b), float(r["guc"]))] = float(r["v"])
                continue
            ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
            yv = ir.fit_transform(s["guc"].values, s["v"].values, sample_weight=s["n"].values)
            for gu, v in zip(s["guc"].values, yv):
                self.cells[(str(b), float(gu))] = float(v)

        self.bolge_lookup = {str(k): float(v) for k, v in im.items()}
        return self

    def get_prior_array(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorized lookup of isotonic winsorized prior log-consumption for given bolge and guc."""
        bolges = df["bolge"].astype(str).values
        gucs = df["guc"].values
        return np.array([
            self.cells.get((b, float(g)), self.bolge_lookup.get(b, self.global_mean))
            for b, g in zip(bolges, gucs)
        ])

    def adjust(self, log_predictions: np.ndarray, test_df: pd.DataFrame, seen_mask: np.ndarray) -> np.ndarray:
        """Blend raw model predictions with hierarchical isotonic prior in log-space on unseen transformers."""
        adjusted = log_predictions.copy()
        unseen = ~seen_mask.astype(bool)
        if unseen.sum() > 0:
            prior = self.get_prior_array(test_df)
            adjusted[unseen] = (1.0 - self.alpha) * log_predictions[unseen] + self.alpha * prior[unseen]
        return adjusted
