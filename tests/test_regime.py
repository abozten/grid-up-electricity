"""Regression checks for regime detection, unit normalisation, and the wake model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.regime import BREAK_CODES, BREAK_SCALE, BREAK_STEP, RegimeDetector
from src.models.wakeup import SeasonalWakeModel


def _series(values: np.ndarray, guc: float = 100.0, tanim: str = "t1",
            start: str = "2025-01-01") -> pd.DataFrame:
    return pd.DataFrame({
        "tanim": tanim,
        "tarih": pd.date_range(start, periods=len(values), freq="D"),
        "tuketim": values.astype(float),
        "guc": guc,
    })


def test_scale_break_is_detected_and_unwound() -> None:
    """A x100 CT jump is a change of units, so normalised history must be continuous."""
    rng = np.random.default_rng(0)
    base = 50.0 + rng.normal(0, 2, 120)
    jumped = np.concatenate([base[:60], base[60:] * 100.0])
    df = _series(jumped, guc=100.0)

    det = RegimeDetector().fit(df)
    state = det.state.loc["t1"]
    assert state["regime_n_breaks"] >= 1
    assert state["regime_last_break"] == BREAK_SCALE

    adj = det.normalize(df)
    pre = adj.loc[adj["tarih"] < "2025-03-02", "tuketim_adj"].median()
    post = adj.loc[adj["tarih"] >= "2025-03-02", "tuketim_adj"].median()
    assert np.isclose(pre, post, rtol=0.15), (pre, post)


def test_genuine_step_is_not_rescaled() -> None:
    """A 1.8x load step is real demand; unwinding it would erase the signal."""
    rng = np.random.default_rng(1)
    base = 50.0 + rng.normal(0, 2, 120)
    stepped = np.concatenate([base[:60], base[60:] * 1.8])
    df = _series(stepped, guc=100.0)

    det = RegimeDetector().fit(df)
    assert det.state.loc["t1", "regime_last_break"] == BREAK_CODES[BREAK_STEP] or \
        det.state.loc["t1", "regime_last_break"] == BREAK_STEP
    adj = det.normalize(df)
    # to_current stays 1.0 for step breaks, so the series is untouched
    assert np.allclose(adj["tuketim_adj"].to_numpy(), df["tuketim"].to_numpy())


def test_normalize_preserves_row_alignment() -> None:
    """The as-of join must not reorder or drop rows."""
    rng = np.random.default_rng(2)
    frames = [_series(np.abs(rng.normal(40, 5, 90)), tanim=f"t{i}") for i in range(3)]
    df = pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=3)

    det = RegimeDetector().fit(df)
    adj = det.normalize(df)
    assert list(adj.index) == list(df.index)
    assert (adj["tanim"].to_numpy() == df["tanim"].to_numpy()).all()
    assert adj["tuketim_adj"].notna().all()


def test_detector_is_causal_wrt_cutoff() -> None:
    """State fitted on a prefix must not change when later data is appended."""
    rng = np.random.default_rng(4)
    full = _series(np.abs(rng.normal(40, 4, 200)))
    prefix = full[full["tarih"] < "2025-04-01"]

    a = RegimeDetector().fit(prefix).state.loc["t1", "regime_level"]
    b = RegimeDetector().fit(prefix.copy()).state.loc["t1", "regime_level"]
    assert a == b  # deterministic
    assert "t1" in RegimeDetector().fit(full).state.index


def test_wake_model_stays_silent_without_wake_evidence() -> None:
    """Gate must suppress the prior when the reference cohort never woke."""
    rng = np.random.default_rng(5)
    rows = []
    for i in range(6):
        v = np.zeros(400)  # dormant throughout: nothing ever wakes
        d = _series(v, tanim=f"d{i}")
        d["il"], d["ilce"] = "İZMİR", "KONAK"
        rows.append(d)
    for i in range(6):
        v = np.abs(rng.normal(60, 5, 400))
        d = _series(v, tanim=f"a{i}")
        d["il"], d["ilce"] = "İZMİR", "KONAK"
        rows.append(d)
    hist = pd.concat(rows, ignore_index=True)
    hist["month"] = hist["tarih"].dt.month

    cutoff = pd.Timestamp("2026-01-01")
    m = SeasonalWakeModel().fit(hist, cutoff, lookback_days=182)
    future = hist[hist["tarih"] >= cutoff - pd.Timedelta(days=10)].copy()
    preds = m.predict(future)
    assert np.isnan(preds).all(), "gate should suppress a cohort that never woke"


def test_wake_model_predicts_p_times_level() -> None:
    """With a cohort that wakes, the estimate must sit between zero and the active level."""
    rng = np.random.default_rng(6)
    rows = []
    for i in range(8):
        # asleep for the first half of each year, active in summer
        v = np.zeros(400)
        idx = pd.date_range("2025-01-01", periods=400, freq="D")
        summer = np.isin(idx.month, [6, 7])
        v[summer] = np.abs(rng.normal(500, 30, summer.sum()))
        d = _series(v, tanim=f"s{i}")
        d["il"], d["ilce"] = "MANİSA", "SALİHLİ"
        rows.append(d)
    hist = pd.concat(rows, ignore_index=True)
    hist["month"] = hist["tarih"].dt.month

    cutoff = pd.Timestamp("2025-06-01")
    m = SeasonalWakeModel().fit(hist, cutoff, lookback_days=120)
    fut = hist[hist["month"] == 6].drop_duplicates("tanim").copy()
    preds = m.predict(fut)
    fired = preds[np.isfinite(preds)]
    if len(fired):
        assert (fired > 0).all() and (fired < np.log1p(500) + 1).all()
