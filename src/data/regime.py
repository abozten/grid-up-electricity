"""Causal regime detection: segment each transformer's history at its structural breaks.

Motivation (measured, see reports/REGIME.md): transformers whose level shifts between
the feature window and the forecast horizon are ~9.5% of rows but carry ~58% of total
squared log error. Lifetime aggregates (tan_mean, sdly_364, tan_moy_mean, the summer
profile) average across both sides of such a break, so they point at a level the
transformer has already left. Shifts persist -- the optimal carry-forward weight on a
detected shift is k~=1.0 -- so pre-break history is not noise to be averaged away, it is
history denominated in the wrong units.

This module finds the breaks, labels what kind of break each one is, and re-expresses
pre-break history in the units of the current regime so downstream aggregates are
comparable to the horizon being forecast.

Everything here is strictly causal: fit() may only ever be handed pre-cutoff history,
and every returned quantity is a function of that history alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# CT/VT amplification rates a misconfigured meter multiplies or divides by.
STANDARD_RATES = np.array([
    2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 25, 30, 40, 50, 60, 75, 80,
    100, 120, 125, 150, 160, 200, 240, 250, 300, 400, 500, 600, 800, 1000,
], dtype=float)

BREAK_WAKE = "wake"    # dormant -> active
BREAK_SLEEP = "sleep"  # active -> dormant
BREAK_SCALE = "scale"  # abrupt level jump matching a CT rate, daily shape preserved
BREAK_STEP = "step"    # persistent level change that is not a metering artifact
BREAK_NONE = "none"

BREAK_CODES = {BREAK_NONE: 0, BREAK_STEP: 1, BREAK_SCALE: 2, BREAK_WAKE: 3, BREAK_SLEEP: 4}


@dataclass(frozen=True)
class RegimeParams:
    """Detection thresholds. Defaults tuned on the 2025-12 backtest split."""
    min_seg: int = 21          # shortest admissible segment (days)
    probe: int = 28            # half-width of the median comparison window
    tau: float = 0.35          # min |log level change| to call a break (~1.42x)
    max_breaks: int = 4        # cap on binary-segmentation depth
    dormant_frac: float = 0.80 # zero-fraction above which a segment counts as dormant
    rate_tol: float = 0.18     # relative tolerance when matching a CT rate
    abrupt_days: int = 3       # a scale artifact switches over within this many days
    shape_corr: float = 0.55   # min day-of-week profile correlation for a scale artifact
    min_obs: int = 60          # transformers shorter than this are left unsegmented


def _window_medians(y: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Medians of the w-length windows ending at / starting at each index.

    Returns (left, right) where left[i] = median(y[i-w:i]) and right[i] = median(y[i:i+w]),
    NaN where the window does not fit. One vectorised sort over a strided view replaces
    the per-candidate np.median calls, which is where the original scan spent its time.
    """
    n = len(y)
    left = np.full(n + 1, np.nan)
    right = np.full(n + 1, np.nan)
    if n >= w:
        win = np.lib.stride_tricks.sliding_window_view(y, w)  # (n-w+1, w)
        med = np.median(win, axis=1)
        right[: n - w + 1] = med
        left[w:] = med
    return left, right


def _segment_one(y: np.ndarray, p: RegimeParams) -> list[int]:
    """Binary segmentation on a log-space series. Returns interior break indices."""
    left, right = _window_medians(y, p.probe)
    diff = np.abs(right - left)  # candidate split strength at each index
    breaks: list[int] = []

    def recurse(lo: int, hi: int, depth: int) -> None:
        if depth >= p.max_breaks or (hi - lo) < 2 * p.min_seg:
            return
        # A candidate needs min_seg on both sides and a full probe window inside [lo, hi).
        a = max(lo + p.min_seg, lo + p.probe)
        b = min(hi - p.min_seg, hi - p.probe)
        if b < a:
            return
        seg = diff[a:b + 1]
        if not np.isfinite(seg).any():
            return
        k = int(np.nanargmax(seg))
        best_i, best_d = a + k, float(seg[k])
        if not np.isfinite(best_d) or best_d < p.tau:
            return
        breaks.append(best_i)
        recurse(lo, best_i, depth + 1)
        recurse(best_i, hi, depth + 1)

    recurse(0, len(y), 0)
    return sorted(breaks)


def _dow_profile(vals: np.ndarray, dows: np.ndarray) -> np.ndarray:
    """Mean log load per weekday, centred. NaN where a weekday is unobserved."""
    prof = np.full(7, np.nan)
    for d in range(7):
        m = dows == d
        if m.any():
            prof[d] = vals[m].mean()
    return prof - np.nanmean(prof)


def _classify(pre_v: np.ndarray, post_v: np.ndarray, pre_d: np.ndarray, post_d: np.ndarray,
              gap_days: int, cap: float, p: RegimeParams) -> tuple[str, float]:
    """Label a single break and return (kind, level ratio in linear space)."""
    pre_zero = float((pre_v <= 0).mean())
    post_zero = float((post_v <= 0).mean())
    if post_zero >= p.dormant_frac and pre_zero < p.dormant_frac:
        return BREAK_SLEEP, 0.0
    if pre_zero >= p.dormant_frac and post_zero < p.dormant_frac:
        return BREAK_WAKE, np.inf

    pre_pos, post_pos = pre_v[pre_v > 0], post_v[post_v > 0]
    if len(pre_pos) < 5 or len(post_pos) < 5:
        return BREAK_NONE, 1.0
    lo_med, hi_med = float(np.median(pre_pos)), float(np.median(post_pos))
    ratio = hi_med / lo_med

    # A metering artifact is abrupt, lands on a standard CT rate, keeps the daily
    # shape intact, and usually pushes one side past the transformer's nameplate.
    r = max(ratio, 1.0 / ratio)
    rel = np.abs(STANDARD_RATES - r) / STANDARD_RATES
    on_rate = bool(rel.min() < p.rate_tol) and r >= 2.0
    if on_rate and gap_days <= p.abrupt_days:
        a = _dow_profile(np.log1p(pre_pos), pre_d[pre_v > 0])
        b = _dow_profile(np.log1p(post_pos), post_d[post_v > 0])
        ok = ~(np.isnan(a) | np.isnan(b))
        shape_ok = ok.sum() >= 4 and np.corrcoef(a[ok], b[ok])[0, 1] >= p.shape_corr
        impossible = cap > 0 and (hi_med > cap or lo_med > cap)
        if shape_ok or impossible:
            return BREAK_SCALE, ratio
    return BREAK_STEP, ratio


class RegimeDetector:
    """Segments transformer histories at structural breaks and normalises their units.

    Usage is always: fit on pre-cutoff history, then read `segments`/`state`, or call
    `normalize` to obtain that same history restated in current-regime units.
    """

    def __init__(self, params: RegimeParams | None = None) -> None:
        self.params = params or RegimeParams()
        self.segments: pd.DataFrame = pd.DataFrame()
        self.state: pd.DataFrame = pd.DataFrame()

    def fit(self, hist: pd.DataFrame) -> RegimeDetector:
        """Detect breaks in `hist` (must be strictly pre-cutoff)."""
        p = self.params
        need = {"tanim", "tarih", "tuketim", "guc"}
        missing = need - set(hist.columns)
        if missing:
            raise ValueError(f"regime detection needs columns {sorted(missing)}")

        h = hist[["tanim", "tarih", "tuketim", "guc"]].sort_values(["tanim", "tarih"])
        cutoff = h["tarih"].max()
        seg_rows, state_rows = [], []

        for tanim, d in h.groupby("tanim", sort=False):
            v = d["tuketim"].to_numpy(dtype=float)
            dates = d["tarih"].to_numpy()
            dows = d["tarih"].dt.dayofweek.to_numpy()
            cap = float(d["guc"].iloc[0]) * 24.0
            n = len(v)

            idx = _segment_one(np.log1p(v), p) if n >= p.min_obs else []
            bounds = [0, *idx, n]
            segs = []
            for s, e in zip(bounds[:-1], bounds[1:]):
                pos = v[s:e][v[s:e] > 0]
                segs.append(dict(
                    start=dates[s], end=dates[e - 1], n=e - s,
                    level=float(np.median(pos)) if len(pos) else 0.0,
                    zero_frac=float((v[s:e] <= 0).mean()),
                ))

            # Label each boundary, then accumulate the factor that carries every
            # segment forward into the units of the final (current) segment.
            kinds, ratios = [], []
            for k, i in enumerate(idx):
                gap = int((dates[i] - dates[i - 1]) / np.timedelta64(1, "D")) if i > 0 else 1
                kind, ratio = _classify(v[bounds[k]:i], v[i:bounds[k + 2]],
                                        dows[bounds[k]:i], dows[i:bounds[k + 2]], gap, cap, p)
                kinds.append(kind)
                ratios.append(ratio)

            factor = 1.0
            factors = [1.0] * len(segs)
            for k in range(len(segs) - 2, -1, -1):
                # Only metering artifacts are a change of units; a genuine step is a
                # change of load and must not be scaled away.
                if kinds[k] == BREAK_SCALE and np.isfinite(ratios[k]) and ratios[k] > 0:
                    factor *= ratios[k]
                factors[k] = factor

            for k, sg in enumerate(segs):
                seg_rows.append(dict(tanim=tanim, seg=k, **sg, to_current=factors[k],
                                     break_after=kinds[k] if k < len(kinds) else BREAK_NONE,
                                     ratio_after=ratios[k] if k < len(ratios) else 1.0))

            cur = segs[-1]
            last_kind = kinds[-1] if kinds else BREAK_NONE
            state_rows.append(dict(
                tanim=tanim,
                regime_level=np.log1p(cur["level"]),
                regime_days=int((cutoff - pd.Timestamp(cur["start"])).days) + 1,
                regime_zero_frac=cur["zero_frac"],
                regime_n_breaks=len(idx),
                regime_last_break=last_kind,
                regime_last_ratio=float(np.clip(ratios[-1], 0.0, 1e6)) if ratios else 1.0,
                regime_scale_to_current=float(factors[0]),
                regime_is_dormant=int(cur["zero_frac"] >= p.dormant_frac),
                regime_cap_violation=int(cap > 0 and cur["level"] > cap),
            ))

        self.segments = pd.DataFrame(seg_rows)
        self.state = pd.DataFrame(state_rows).set_index("tanim")
        return self

    def normalize(self, hist: pd.DataFrame) -> pd.DataFrame:
        """Restate `hist` in current-regime units, adding `tuketim_adj` and `log_t`.

        Only `scale` breaks are unwound -- those are unit changes. Genuine load steps
        are left alone; the model is told about them through the state features instead.
        """
        if self.segments.empty:
            out = hist.copy()
            out["tuketim_adj"] = out["tuketim"]
            return out

        out = hist.copy()
        # Segments are contiguous per transformer, so the segment covering a row is the
        # last one starting at or before it -- an as-of join rather than a scan.
        seg = (self.segments[["tanim", "start", "to_current"]]
               .sort_values(["start", "tanim"], kind="mergesort"))
        seg["start"] = pd.to_datetime(seg["start"])
        left = out[["tanim", "tarih"]].reset_index().sort_values(["tarih", "tanim"], kind="mergesort")
        joined = pd.merge_asof(left, seg, left_on="tarih", right_on="start",
                               by="tanim", direction="backward")
        factor = (joined.set_index("index")["to_current"]
                  .reindex(out.index).fillna(1.0).to_numpy(dtype=float))
        adj = out["tuketim"].to_numpy(dtype=float) * factor
        # `tuketim` itself is overwritten, not just log_t: HistoryFeatureExtractor
        # derives load factors straight from `tuketim`, and leaving that raw would give
        # scale-corrected transformers a load factor inconsistent with every other
        # feature -- exactly the meters this is supposed to fix. Raw values are kept
        # under `tuketim_raw` for auditing.
        out["tuketim_raw"] = out["tuketim"]
        out["tuketim_adj"] = adj
        out["tuketim"] = adj
        out["log_t"] = np.log1p(adj)
        out["is_pos"] = (adj > 0).astype(int)
        return out

    def attach_state(self, df: pd.DataFrame) -> pd.DataFrame:
        """Join the per-transformer regime state onto a feature frame."""
        if self.state.empty:
            return df
        out = df.merge(self.state, left_on="tanim", right_index=True, how="left")
        # Encoded as an integer rather than a category: CatBoost only declares
        # il/bolge/ilce as categorical, and an undeclared string column errors out.
        out["regime_last_break"] = (out["regime_last_break"].fillna(BREAK_NONE)
                                    .map(BREAK_CODES).fillna(0).astype(int))
        defaults = {
            "regime_days": 0, "regime_n_breaks": 0, "regime_last_ratio": 1.0,
            "regime_scale_to_current": 1.0, "regime_is_dormant": 0,
            "regime_cap_violation": 0, "regime_zero_frac": 0.0,
        }
        for c, v in defaults.items():
            if c in out.columns:
                out[c] = out[c].fillna(v)
        return out


REGIME_FEATURES = [
    "regime_level", "regime_days", "regime_zero_frac", "regime_n_breaks",
    "regime_last_break", "regime_last_ratio", "regime_scale_to_current",
    "regime_is_dormant", "regime_cap_violation",
]
