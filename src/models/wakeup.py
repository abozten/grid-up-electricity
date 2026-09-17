"""Seasonal wake-up model for dormant transformers.

Measured on the Apr-Jul split that mirrors the real test window: transformers that
are dormant at the feature cutoff are 5.0% of rows but carry 22.6% of squared log
error, and the damage is seasonal -- RMSLE 0.78 in April, 1.07 in May, 1.99 in June,
4.12 in July, as irrigation load switches back on. A recency baseline predicts zero
for all of them and is simply wrong by July.

The fix uses the structure of the loss. RMSLE is squared error in log space, so for a
transformer that is active with probability p at log-level L and zero otherwise, the
error-minimising constant is not L but p*L. This module estimates p and L per
(ilce, month) -- and per transformer where a prior-year season exists -- and emits
p*L for dormant transformers.

Estimation is strictly causal: priors come only from data before the cutoff handed to
fit(). For the real Apr-Jul 2026 test that means the Apr-Jul 2025 season, which is a
directly comparable seasonal analog.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WakeParams:
    """Shrinkage settings for the hierarchical wake priors."""
    dormant_window: int = 28   # days before cutoff that must be all-zero to count as dormant
    min_ilce_obs: int = 30     # observations below which an ilce-month falls back to province
    self_weight_k: float = 20.0  # pseudo-count controlling trust in a transformer's own season
    max_level: float = 14.0    # log1p clip, guards against a single corrupt reading
    min_wake_p: float = 0.15   # below this cohort wake rate the model stays out of the way


class SeasonalWakeModel:
    """Predicts p(active) * level for transformers dormant at the cutoff.

    fit(hist)   -- learn (ilce, month) and per-transformer seasonal wake priors
    predict(df) -- return a log-space prediction for rows whose transformer is dormant
    """

    def __init__(self, params: WakeParams | None = None) -> None:
        self.params = params or WakeParams()
        self.dormant: set[str] = set()
        self.ilce_prior: pd.DataFrame = pd.DataFrame()
        self.il_prior: pd.DataFrame = pd.DataFrame()
        self.self_prior: pd.DataFrame = pd.DataFrame()
        self.global_prior: float = 0.0

    @staticmethod
    def _pl(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        """Active rate p, active log-level L, and their product, grouped by keys."""
        g = df.groupby(keys, observed=True)
        out = pd.DataFrame({
            "p": g["tuketim"].apply(lambda s: float((s > 0).mean())),
            "n": g.size().astype(float),
        })
        act = df[df["tuketim"] > 0]
        if len(act):
            out = out.join(act.groupby(keys, observed=True)["tuketim"]
                           .apply(lambda s: float(np.log1p(s).mean())).rename("L"))
        else:
            out["L"] = 0.0
        out["L"] = out["L"].fillna(0.0)
        out["pL"] = out["p"] * out["L"]
        return out

    @staticmethod
    def _dormant_at(h: pd.DataFrame, when: pd.Timestamp, window: int) -> set[str]:
        """Transformers whose last `window` days before `when` are all zero."""
        w = h[(h["tarih"] < when) & (h["tarih"] >= when - pd.Timedelta(days=window))]
        if w.empty:
            return set()
        mx = w.groupby("tanim")["tuketim"].max()
        return set(mx[mx == 0.0].index.astype(str))

    def fit(self, hist: pd.DataFrame, cutoff: pd.Timestamp | str,
            lookback_days: int = 365) -> SeasonalWakeModel:
        """Learn wake priors from pre-cutoff history only.

        The prior must describe the population the model is applied to -- transformers
        that were dormant and then may wake -- not the transformer fleet at large.
        Estimating it on all rows was the original error: it predicts a busy ilce-month
        level for meters that in fact stay at zero. So we step back to a reference
        cutoff one season earlier, take who was dormant *there*, and measure what that
        cohort actually did over the following months.
        """
        p = self.params
        cutoff = pd.Timestamp(cutoff)
        h = hist[hist["tarih"] < cutoff].copy()
        if "month" not in h.columns:
            h["month"] = h["tarih"].dt.month

        self.dormant = self._dormant_at(h, cutoff, p.dormant_window)

        ref = cutoff - pd.Timedelta(days=lookback_days)
        cohort = self._dormant_at(h, ref, p.dormant_window)
        obs = h[(h["tarih"] >= ref) & h["tanim"].astype(str).isin(cohort)]
        if obs.empty:
            # No usable seasonal analog: fall back to the observed behaviour of
            # currently-dormant meters, which at worst reproduces "stays asleep".
            obs = h[h["tanim"].astype(str).isin(self.dormant)]

        self.ilce_prior = self._pl(obs, ["ilce", "month"]) if len(obs) else pd.DataFrame()
        self.il_prior = self._pl(obs, ["il", "month"]) if len(obs) else pd.DataFrame()
        self.self_prior = self._pl(obs, ["tanim", "month"]) if len(obs) else pd.DataFrame()
        self.global_prior = float(self._pl(obs.assign(_k=0), ["_k"])["pL"].iloc[0]) if len(obs) else 0.0
        return self

    def _lookup(self, df: pd.DataFrame) -> np.ndarray:
        """Hierarchical p*L: transformer's own season, shrunk toward ilce then province."""
        p = self.params
        month = df["month"].values if "month" in df.columns else df["tarih"].dt.month.values
        idx_i = pd.MultiIndex.from_arrays([df["ilce"].astype(str).values, month])
        idx_l = pd.MultiIndex.from_arrays([df["il"].astype(str).values, month])
        idx_s = pd.MultiIndex.from_arrays([df["tanim"].astype(str).values, month])

        def take(prior: pd.DataFrame, idx: pd.MultiIndex, col: str) -> np.ndarray:
            if prior.empty or col not in prior.columns:
                return np.full(len(df), np.nan)
            return prior[col].reindex(idx).to_numpy(dtype=float)

        il = take(self.il_prior, idx_l, "pL")
        ilce = take(self.ilce_prior, idx_i, "pL")
        ilce_n = take(self.ilce_prior, idx_i, "n")
        base = np.where(np.isfinite(ilce) & (ilce_n >= p.min_ilce_obs), ilce, il)
        base = np.where(np.isfinite(base), base, self.global_prior)

        # Blend in the transformer's own same-month history when it has one
        own = take(self.self_prior, idx_s, "pL")
        own_n = take(self.self_prior, idx_s, "n")
        w = np.where(np.isfinite(own), own_n / (own_n + p.self_weight_k), 0.0)
        own = np.where(np.isfinite(own), own, 0.0)
        est = np.clip(w * own + (1.0 - w) * base, 0.0, p.max_level)

        # Gate: only speak where the reference cohort actually woke in this month.
        # Validated on the winter split, where dormant meters stay dormant and an
        # ungated prior costs 0.17% RMSLE for no reason.
        wake_p = take(self.ilce_prior, idx_i, "p")
        wake_p = np.where(np.isfinite(wake_p), wake_p, take(self.il_prior, idx_l, "p"))
        return np.where(np.isfinite(wake_p) & (wake_p >= p.min_wake_p), est, np.nan)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Log-space wake prediction per row. NaN where the model does not apply."""
        out = np.full(len(df), np.nan)
        if not self.dormant:
            return out
        mask = df["tanim"].astype(str).isin(self.dormant).values
        if mask.any():
            out[mask] = self._lookup(df.loc[mask])
        return out

    def apply(self, log_predictions: np.ndarray, df: pd.DataFrame,
              weight: float = 1.0) -> np.ndarray:
        """Overwrite predictions for dormant transformers with the wake estimate.

        weight < 1 blends toward the incoming model prediction, which is the safer
        setting when the base model already has some dormancy signal of its own.
        """
        wake = self.predict(df)
        out = np.asarray(log_predictions, dtype=float).copy()
        m = np.isfinite(wake)
        out[m] = weight * wake[m] + (1.0 - weight) * out[m]
        return out
