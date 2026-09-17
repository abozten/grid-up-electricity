"""Per-transformer Bayram response: the feast-day deviation each meter actually shows.

WHAT IS MISSING TODAY. `CalendarFeatureExtractor` gives the model three binary flags
(`is_ramazan_bayram`, `is_kurban_bayram`, `is_arefe`). A flag can only carry ONE fleet-average
shift. The EDA (reports/EDA.md section 3) measures the true shift ranging from -89.9% (standby)
and -56.3% (commercial) to +15.7% (tourism) -- so the single flag is not merely weak, it pushes
tourism and agro meters the wrong way.

WHY IT IS A REAL TRAIT AND NOT DRIFT. Measured on train.csv, per-meter feast delta = median(feast
days) - median(surrounding +/-21 days, feast excluded):

    REAL feast -> REAL feast          spearman   PLACEBO -> PLACEBO        spearman
      ram25 -> kur25                    +0.553     p_mar25 -> p_jun25        +0.005
      ram25 -> ram26                    +0.707     p_mar25 -> p_mar26        +0.067
      kur25 -> ram26                    +0.581     p_jun25 -> p_mar26        +0.167

    predicting the kur25 delta:  from ram25 (real feast) +0.553 | from a matched non-feast
    window of the same length and season  -0.191

Placebo windows were matched on length, weekday span and season. They carry no cross-window
signal, and the placebo window ANTI-predicts the real feast. Per-meter spread is also ~2x the
placebo spread (sd 0.42 vs 0.22-0.35). The response is a persistent property of the meter.

TEST-WINDOW FOOTPRINT, stated plainly. The test span is 2026-04-01..2026-07-31 and contains
exactly one feast, Kurban 2026 (May 27-30) -- 4 of 122 days, ~3.3% of rows. This is a targeted
repair of a segment the current flag handles backwards, NOT a broad gain. Judge it on feast-day
rows; pooled it will be small and could sit inside seed noise.

REACHES COLD START. The estimate falls back own-meter -> (ilce, guc_b) cohort -> fleet, and 78.9%
of cold-start transformers land in a cohort with >=5 measured meters. Cohort deltas are themselves
stable across feasts (spearman +0.745), so the fallback carries signal rather than diluting to zero.

CAUSALITY. Everything is computed from `hist`, which callers pass strictly pre-cutoff. At the
shipped cutoff (2026-04-01) history holds Ramazan 2025, Kurban 2025 and Ramazan 2026 -- three
feasts -- and the applied feast, Kurban 2026, is entirely in the future. A cold-start meter
contributes no rows to its own cohort, so there is no self-reference.

AREFE IS NOT MEASURED HERE. The eve behaves differently from the feast itself (Kurban 2025: eve
-0.02 vs feast -0.06 on the fleet mean) and `is_arefe` already exists. Measuring on feast days and
applying on feast days keeps the definition consistent; widening it is untested.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.calendar import CalendarFeatureExtractor
from src.features.cohort import bucket_edges

BAYRAM_FEATS = ["bay_dev"]

WIN = 21        # +/- days around a feast that define the meter's own baseline
MIN_OBS = 2     # a meter needs >=2 feast days in a block before that block counts
K_OWN = 4.0     # shrink a meter's own delta toward its cohort at ~4 feast-days of evidence
K_COH = 5.0     # shrink a thin cohort toward the fleet delta

FEAST_DAYS = pd.DatetimeIndex(sorted(pd.to_datetime(
    list(CalendarFeatureExtractor.RAMAZAN_DATES | CalendarFeatureExtractor.KURBAN_DATES))))


def _blocks(days: pd.DatetimeIndex) -> list[pd.DatetimeIndex]:
    """Split feast dates into contiguous runs, so each feast gets its own local baseline."""
    gap = np.diff(days.values).astype("timedelta64[D]").astype(int) > 1
    grp = np.r_[0, np.cumsum(gap)]
    return [days[grp == g] for g in np.unique(grp)]


def _empty(edges: np.ndarray) -> dict:
    return {"per": pd.Series(dtype=float), "n": pd.Series(dtype=float),
            "coh": pd.Series(dtype=float), "edges": edges, "g": 0.0}


def build_bayram(hist: pd.DataFrame, win: int = WIN) -> dict:
    """Per-meter and per-cohort feast deviation in log space, from history alone.

    The baseline is the meter's own median over the surrounding window with the feast removed, so
    a meter's level cancels out and only its feast RESPONSE survives. Each feast block is measured
    against its own local baseline, which is what keeps the seasonal trend from leaking in.
    """
    edges = bucket_edges(hist)
    blocks = [b for b in _blocks(FEAST_DAYS) if hist["tarih"].isin(b).any()]
    if not blocks:
        return _empty(edges)     # e.g. the 2025-02-01 fold cut: history has no feast yet

    devs, cnts = [], []
    for b in blocks:
        lo, hi = b[0] - pd.Timedelta(win, "D"), b[-1] + pd.Timedelta(win, "D")
        w = hist[(hist["tarih"] >= lo) & (hist["tarih"] <= hi)]
        isf = w["tarih"].isin(b)
        n = w[isf].groupby("tanim").size()
        feast = w[isf].groupby("tanim")["log_t"].median()
        base = w[~isf].groupby("tanim")["log_t"].median()
        devs.append((feast - base).where(n >= MIN_OBS))
        cnts.append(n.where(n >= MIN_OBS))

    per = pd.concat(devs, axis=1).mean(axis=1).dropna()
    nobs = pd.concat(cnts, axis=1).sum(axis=1).reindex(per.index).fillna(0.0)
    if per.empty:
        return _empty(edges)
    g = float(per.mean())

    meta = hist.groupby("tanim").agg(ilce=("ilce", "first"), guc=("guc", "first"))
    meta["guc_b"] = np.digitize(np.log1p(meta["guc"].values.astype(float)), edges).astype(np.int16)
    j = meta.join(per.rename("d"), how="inner")
    c = j.groupby([j["ilce"].astype(str), "guc_b"])["d"].agg(["mean", "size"])
    coh = (c["mean"] * c["size"] + g * K_COH) / (c["size"] + K_COH)

    return {"per": per, "n": nobs, "coh": coh, "edges": edges, "g": g}


def attach(X: pd.DataFrame, B: dict) -> pd.DataFrame:
    """Add `bay_dev`: the row's expected feast deviation, and exactly 0.0 off feast days.

    Zeroing off-feast is the point of the feature. Handed to the model as a standalone trait it
    would be visible on all 122 test days and a 3-observation estimate would get used as a general
    meter descriptor; pre-applying the interaction confines it to where it was measured.
    """
    X = X.copy()
    on_feast = X["tarih"].isin(FEAST_DAYS).values
    if B["per"].empty or not on_feast.any():
        X["bay_dev"] = 0.0
        return X

    # same bucketing rule as src/features/cohort.add_bucket, inlined so no guc_b column is clobbered
    gb = np.digitize(np.log1p(X["guc"].values.astype(float)), B["edges"]).astype(np.int16)
    idx = pd.MultiIndex.from_arrays([X["ilce"].astype(str), pd.Index(gb)])
    coh = B["coh"].reindex(idx).values.astype(float)
    # an unseen cohort falls back to the fleet delta, not to zero: zero asserts "this meter does
    # not react to the feast", which is the opposite of unknown
    coh = np.where(np.isfinite(coh), coh, B["g"])

    own = X["tanim"].map(B["per"]).values.astype(float)
    n = X["tanim"].map(B["n"]).fillna(0.0).values.astype(float)
    own = np.where(np.isfinite(own), own, coh)
    dev = (n * own + K_OWN * coh) / (n + K_OWN)

    X["bay_dev"] = np.where(on_feast, dev, 0.0)
    return X


if __name__ == "__main__":
    # Two districts with opposite, planted feast responses: A drops 0.60, B rises 0.30.
    # Recovering both signs is what catches the flag-only failure this feature exists to fix.
    rng = np.random.default_rng(0)
    dates = pd.date_range("2025-01-01", "2026-03-31", freq="D")
    rows = []
    for i in range(60):
        ilce, shift = ("A", -0.60) if i < 30 else ("B", +0.30)
        lvl = 7.0 + 0.01 * i
        y = lvl + 0.02 * rng.standard_normal(len(dates))
        y = y + np.where(dates.isin(FEAST_DAYS), shift, 0.0)
        rows.append(pd.DataFrame({"tanim": f"t{i}", "tarih": dates, "ilce": ilce,
                                  "guc": 400.0 + 50 * (i % 4), "log_t": y}))
    hist = pd.concat(rows, ignore_index=True)
    hist["tuketim"] = np.expm1(hist.log_t)

    B = build_bayram(hist)
    a = B["per"][[f"t{i}" for i in range(30)]]
    b = B["per"][[f"t{i}" for i in range(30, 60)]]
    assert abs(a.mean() + 0.60) < 0.05, a.mean()      # sign and size both recovered
    assert abs(b.mean() - 0.30) < 0.05, b.mean()
    assert a.max() < 0 < b.min(), "the two districts must not overlap"

    # apply: feast rows carry the deviation, every other row is exactly zero
    X = pd.DataFrame({"tanim": ["t0", "t0", "t35", "t35"],
                      "tarih": pd.to_datetime(["2026-05-28", "2026-05-15",
                                               "2026-05-28", "2026-05-15"]),
                      "ilce": ["A", "A", "B", "B"], "guc": [400.0, 400.0, 450.0, 450.0]})
    out = attach(X, B)
    assert (out.bay_dev[[1, 3]] == 0.0).all(), "off-feast rows must be exactly zero"
    assert out.bay_dev[0] < -0.4 and out.bay_dev[2] > 0.2, out.bay_dev.tolist()

    # cold start: a meter absent from history inherits its cohort, not zero and not the fleet mean
    C = pd.DataFrame({"tanim": ["unseen1"], "tarih": pd.to_datetime(["2026-05-28"]),
                      "ilce": ["A"], "guc": [400.0]})
    cold = attach(C, B).bay_dev.iloc[0]
    assert cold < -0.4, f"cold row in district A should inherit A's drop, got {cold}"
    assert abs(cold - B["g"]) > 0.1, "cold row collapsed to the fleet mean; cohort lookup missed"

    # causality: history with no feast in it yields a neutral, non-crashing feature
    pre = hist[hist.tarih < "2025-03-01"]
    Z = build_bayram(pre)
    assert Z["per"].empty and (attach(X, Z).bay_dev == 0.0).all()

    # a shrunk own-estimate always sits between the raw own delta and its cohort
    own, coh = float(B["per"]["t0"]), float(B["coh"][("A", np.digitize(np.log1p(400.0), B["edges"]))])
    assert min(own, coh) - 1e-9 <= out.bay_dev[0] <= max(own, coh) + 1e-9

    print(f"ok: bay_dev | A {a.mean():+.3f}  B {b.mean():+.3f}  fleet {B['g']:+.3f} "
          f"| {len(B['coh'])} cohorts")
