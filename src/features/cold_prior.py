"""The cold-start prior, with capacity monotonicity. Drop-in replacement for v18's.

WHAT CHANGES. v18 shrinks each (bolge, guc) cell toward its bolge mean with K=5. A thin cell
therefore collapses onto a regional average and DISCARDS capacity -- the single most informative
thing a never-metered transformer carries. This adds one step: within each bolge, force the shrunk
cell means to be non-decreasing in guc (isotonic regression, weighted by cell count). A larger
transformer serves more load; cell means violate that only where a cell is thin, so isotonic
borrows strength along the capacity axis instead of only from the regional parent.

WHY IT WAS MISSED. `exp_priors.py` ranked this variant on the winter fold's NATURAL cold-start mix
and measured -0.0040 -- real but unremarkable, and it was bundled into v22 alongside winsorisation
and alpha=0.8. The fold mix is not the test mix:

    capacity band   fold cold-start rows   test cold-start rows   weight
    0-250                   21.9%                  16.0%           0.73
    250-400                 19.3%                  14.0%           0.73
    400-630                 28.7%                  17.5%           0.61
    630+                    30.2%                  52.6%           1.74

Test cold start is dominated by LARGE transformers (median 630 kVA against the fold's 400), which
is exactly where cells are thinnest and where what you shrink toward stops being cosmetic. Weighted
to the test mix the gain roughly doubles.

MEASURED, unseen rows only, test-composition weighted, cached predictions so no training variance:

    fold     alpha=0.6            alpha=0.7
    winter   -0.0060 unseen       -0.0077 unseen   (test_comp -0.0024 / -0.0030)
    autumn   -0.0059 unseen       -0.0076 unseen   (test_comp -0.0020 / -0.0026)

Two-direction PASS at every alpha tested from 0.5 to 0.8.

ALPHA. Keep 0.6 to change one thing at a time -- it already passes both folds. The two folds
disagree on the optimum (winter prefers 0.7-0.8, autumn 0.6), so 0.7 is the compromise and sits
within 0.001 of each fold's own best. Do not chase alpha=0.8: its larger DELTA comes from the
baseline degrading faster, not from the variant improving, and autumn's monotone score is worse
there in absolute terms.

CAPPED BY DEFAULT. Isotonic pooling produces a few violent corrections -- METROPOL guc=17900 moves
from 0.189 to 8.067 log because that capacity class is largely DEAD, and pooling averages it with
the active 10900 class rather than repairing a thin cell. Those cells are not where the gain comes
from: on autumn they are 0.01% of rows and contribute -0.3% of the effect, while the 630+ band as a
whole contributes -0.0218. Capping |correction| at 1.0 log unit reproduces the uncapped score
exactly on both folds (-0.0024 / -0.0020) and removes the tail entirely, so it is the default.

DO NOT ALSO WINSORISE. v22 bundled winsor(5,95) with this. Winsorisation helps on the natural fold
mix (-0.0008) and REVERSES on the test-matched one (+0.0010 winter, +0.0057 autumn) -- a sign flip
that the original ablation's weighting hid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K_DEFAULT = 5.0
ALPHA_DEFAULT = 0.6
CAP_DEFAULT = 1.0   # max |isotonic correction|, log units; see the docstring


def _shrunk_cells(hist: pd.DataFrame, K: float):
    g = hist.groupby(["bolge", "guc"], observed=True).log_t.agg(["mean", "size"])
    up = hist.groupby("bolge", observed=True).log_t.mean()
    gm = float(hist.log_t.mean())
    s = (g["mean"] * g["size"] + up.reindex(g.index.get_level_values(0)).values * K) / (g["size"] + K)
    return s, g["size"], up, gm


def build(hist: pd.DataFrame, K: float = K_DEFAULT, monotone: bool = True,
          cap: float = CAP_DEFAULT) -> dict:
    """Prior lookup tables from history alone. `monotone=False` reproduces v18 exactly."""
    s, size, up, gm = _shrunk_cells(hist, K)
    cells = {(str(a), b): float(v) for (a, b), v in s.items()}

    if monotone:
        from sklearn.isotonic import IsotonicRegression
        df = s.rename("v").reset_index()
        df["n"] = size.values
        for b, sub in df.groupby("bolge", observed=True):
            sub = sub.sort_values("guc")
            if len(sub) < 3:
                continue          # two points are always monotone in one direction or the other
            ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
            fitted = ir.fit_transform(sub.guc.values, sub.v.values, sample_weight=sub.n.values)
            for gu, v, base in zip(sub.guc.values, fitted, sub.v.values):
                cells[(str(b), gu)] = float(base + np.clip(v - base, -cap, cap))

    return {"cells": cells, "bolge": {str(k): float(v) for k, v in up.items()}, "global": gm}


def lookup(prior: dict, rows: pd.DataFrame) -> np.ndarray:
    """Per-row prior in LOG space. Falls back cell -> bolge -> global."""
    c, b, gm = prior["cells"], prior["bolge"], prior["global"]
    return np.array([c.get(k, b.get(k[0], gm))
                     for k in zip(rows.bolge.astype(str), rows.guc)])


def apply(pred: np.ndarray, rows: pd.DataFrame, seen: np.ndarray, prior: dict,
          alpha: float = ALPHA_DEFAULT) -> np.ndarray:
    """Blend in LOG space on UNSEEN rows only. Seen rows are returned untouched.

    Log space is not incidental: RMSLE is RMSE on log1p, so the loss-optimal point predictor is the
    conditional mean in log space. The same blend on raw kWh optimises the wrong functional and
    drifts upward through Jensen.
    """
    seen = np.asarray(seen).astype(bool)
    out = np.asarray(pred, dtype=float).copy()
    u = ~seen
    if u.any():
        out[u] = (1 - alpha) * out[u] + alpha * lookup(prior, rows[u])
    return out
