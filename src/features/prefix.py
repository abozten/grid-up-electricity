"""Transformer-ID prefix features: the one cold-start signal measured to beat district.

WHERE THIS CAME FROM. The zero-mass audit found 55 unseen transformers that are 100% zero for a
whole window, carrying 94.9% of all unseen zero rows. The hurdle model's classifier scores AUC
0.9800 on seen rows and **0.5333 on unseen** -- it is blind exactly where those feeders live,
because every feature it has (tan_zero, tan_r7_zero, ...) requires history.

Ranking "is this unseen transformer dead", causal signals only:

    tanim prefix, 3 chars, shrunk dead-rate   AUC 0.6421   <- best available
    ilce x guc cell dead-rate                      0.5933
    ilce dead-rate                                 0.5912
    capacity                                       0.4685

`tanim` is an 8- or 9-digit code and the leading digits group transformers in a way that predicts
dead-ness better than geography does -- plausibly a substation or feeder identifier. The model has
never been given it: `tanim` itself is an identifier, never a feature, and correctly so, but its
PREFIX is a legitimate grouping key.

CAUSALITY. Every statistic is computed from `hist` (strictly before the cutoff) over OTHER
transformers. An unseen transformer contributes no rows to its own prefix group, so there is no
self-reference to leak -- the same argument that made leave-one-out an exact no-op for the v18
cold-start prior.

RISK, stated up front. Adding nine cohort features aimed at cold start FAILED earlier today
(+0.0070, and unseen moved +0.0002): a cold-start row already reaches most group information
through `ilce` and `guc`. What makes prefix different is that it is measurably NOT redundant with
those -- it outranks both on the only task that matters here. That is a reason to test it, not a
reason to expect it to work.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PREFIX_FEATS = ["pfx3_dead", "pfx3_mean", "pfx3_n", "pfx5_dead", "pfx5_mean", "pfx5_n"]
K_DEAD, K_MEAN = 5.0, 10.0


def build_prefix(hist: pd.DataFrame) -> dict:
    """Per-prefix dead-rate and level, shrunk toward the global value, from history alone."""
    per = hist.groupby("tanim", observed=True).agg(
        dead=("tuketim", lambda s: float((s == 0).mean() == 1)), lvl=("log_t", "mean"))
    gd, gl = per.dead.mean(), per.lvl.mean()
    out = {}
    for L in (3, 5):
        g = per.assign(k=per.index.str[:L]).groupby("k").agg(
            d=("dead", "mean"), l=("lvl", "mean"), n=("dead", "size"))
        out[L] = pd.DataFrame({
            f"pfx{L}_dead": (g.d * g.n + gd * K_DEAD) / (g.n + K_DEAD),
            f"pfx{L}_mean": (g.l * g.n + gl * K_MEAN) / (g.n + K_MEAN),
            f"pfx{L}_n": np.log1p(g.n),
        })
    out["global"] = (gd, gl)
    return out


def attach(X: pd.DataFrame, pfx: dict) -> pd.DataFrame:
    gd, gl = pfx["global"]
    X = X.copy()
    for L in (3, 5):
        key = X.tanim.str[:L]
        m = pfx[L].reindex(key.values)
        for c in pfx[L].columns:
            X[c] = m[c].values
        # an unseen prefix falls back to the global rate, not to zero: zero would assert
        # "no transformer in this group has ever been dead", which is the opposite of unknown
        X[f"pfx{L}_dead"] = X[f"pfx{L}_dead"].fillna(gd)
        X[f"pfx{L}_mean"] = X[f"pfx{L}_mean"].fillna(gl)
        X[f"pfx{L}_n"] = X[f"pfx{L}_n"].fillna(0.0)
    return X
