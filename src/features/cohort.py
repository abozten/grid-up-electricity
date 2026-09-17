"""Cohort features: give a cold-start row a weather response and a seasonal shape.

WHY THIS AND NOT MORE PRIOR TUNING. At the test mix (22.16% unseen) the unseen segment carries
65% of the MSE while holding 22% of the rows. The post-hoc prior that v18/v21/v22 kept reworking
spans only 0.012 total from alpha=0 to alpha=1 -- smaller than the 0.015 seed spread, which is why
every variant since v18 landed inside noise. The ceiling is not the blend, it is how little a
cold-start row knows about itself.

WHAT IT KNOWS TODAY. `BASE_FEATS` gives an unseen row weather, calendar, and a handful of scalar
group means (`ilce_mean`, `guc_ilce_mean`, `ilce_lf`). Those scalars are CONSTANT per group, so
they can shift its level but cannot tell it how its own kind of transformer RESPONDS -- to heat, to
weekends, to the month of the year. The weather columns are present and, for an unseen row,
effectively unusable: nothing connects them to that transformer's sensitivity.

WHAT THIS ADDS. For each (ilce, guc-bucket) cohort, statistics estimated ONLY from history before
the cutoff: the OLS slope of log_t on cooling and heating degree days, the weekend/weekday
amplitude, the load factor, the dispersion, and the cohort's month-of-year deviation. A cold-start
transformer inherits its cohort's behaviour instead of only its cohort's average.

CAUSALITY. Every statistic is computed from `hist` (strictly `tarih < cut`) and merged onto
validation/test rows by key. No target-period aggregation, no forward lag of any kind.

SEASONAL NOTE. `coh_moy_dev` needs the target month to appear in history. At the winter fold
(cut 2025-12-01) December is absent and Jan-Mar come from a single year, so that column is at its
WEAKEST there. At the shipped cutoff (2026-04-01, history from 2025-01-01) all four target months
Apr-Jul are present in history. The fold therefore understates this feature rather than flattering
it -- the safe direction for a go/no-go.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_BUCKETS = 8
COHORT_FEATS = ["coh_cdd_slope", "coh_cddlog_slope", "coh_hdd_slope", "coh_dow_amp", "coh_lf", "coh_cv",
                "coh_mean", "coh_n", "coh_moy_dev", "coh_wknd_lf"]


def add_bucket(df: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df["guc_b"] = np.digitize(np.log1p(df.guc.values), edges).astype(np.int16)
    return df


def bucket_edges(hist: pd.DataFrame) -> np.ndarray:
    q = np.linspace(0, 1, N_BUCKETS + 1)[1:-1]
    return np.unique(np.quantile(np.log1p(hist.guc.values), q))


def _slope(g: pd.DataFrame, x: str, y: str = "log_t") -> pd.Series:
    """Vectorised OLS slope of y on x per group; NaN where x has no variance."""
    xm = g.groupby("_k", observed=True)[x].transform("mean")
    ym = g.groupby("_k", observed=True)[y].transform("mean")
    dx, dy = g[x] - xm, g[y] - ym
    num = (dx * dy).groupby(g["_k"], observed=True).sum()
    den = (dx * dx).groupby(g["_k"], observed=True).sum()
    return num / den.replace(0, np.nan)


def build_cohort(hist: pd.DataFrame, wx: pd.DataFrame, edges: np.ndarray,
                 keys: tuple[str, ...] = ("ilce", "guc_b")) -> pd.DataFrame:
    """Cohort statistics from history alone. `wx` must carry ilce, tarih, cdd22, hdd18."""
    h = add_bucket(hist, edges)
    # pipeline4 already merges weather onto TR, so cdd22/hdd18 are usually present. Merging again
    # would suffix them to cdd22_x/_y and silently strip the slopes of their regressor.
    need = [c for c in ("cdd22", "hdd18", "cdd_log") if c not in h.columns and c in wx.columns]
    if need:
        h = h.merge(wx[["ilce", "tarih"] + need], on=["ilce", "tarih"], how="left")
    h["_k"] = list(zip(*[h[k].astype(str) for k in keys]))
    h["lf"] = h.tuketim / (h.guc * 24.0).replace(0, np.nan)
    h["is_wknd"] = h.tarih.dt.dayofweek.isin([5, 6])
    h["moy"] = h.tarih.dt.month

    g = h.groupby("_k", observed=True)
    out = pd.DataFrame({
        "coh_mean": g.log_t.mean(),
        "coh_cv": g.log_t.std(),
        "coh_n": g.size(),
        "coh_lf": g.lf.mean(),
        "coh_cdd_slope": _slope(h, "cdd22"),
        # the cold-start-reachable copy of the saturating heat response
        "coh_cddlog_slope": _slope(h, "cdd_log") if "cdd_log" in h.columns else np.nan,
        "coh_hdd_slope": _slope(h, "hdd18"),
    })
    wk = h.groupby(["_k", "is_wknd"], observed=True).log_t.mean().unstack()
    out["coh_dow_amp"] = (wk.get(True) - wk.get(False)) if wk.shape[1] == 2 else np.nan
    wl = h.groupby(["_k", "is_wknd"], observed=True).lf.mean().unstack()
    out["coh_wknd_lf"] = wl.get(True) if True in wl.columns else np.nan

    # month-of-year deviation, cohort mean removed so it carries shape and not level
    moy = h.groupby(["_k", "moy"], observed=True).log_t.mean()
    out_moy = (moy - out.coh_mean.reindex(moy.index.get_level_values(0)).values).rename("coh_moy_dev")
    return out, out_moy.reset_index()


def attach(X: pd.DataFrame, cohort: pd.DataFrame, moy: pd.DataFrame, edges: np.ndarray,
           keys: tuple[str, ...] = ("ilce", "guc_b")) -> pd.DataFrame:
    X = add_bucket(X, edges)
    X["_k"] = list(zip(*[X[k].astype(str) for k in keys]))
    X = X.merge(cohort, left_on="_k", right_index=True, how="left")
    X["moy"] = X.tarih.dt.month
    X = X.merge(moy, on=["_k", "moy"], how="left")
    # a cohort unseen in history falls back to the global mean of each statistic, not to zero:
    # zero is a meaningful value for the slopes and would assert "no weather response".
    for c in COHORT_FEATS:
        if c in X.columns:
            X[c] = X[c].astype(float).fillna(X[c].astype(float).mean())
    return X.drop(columns=["_k", "moy"])
