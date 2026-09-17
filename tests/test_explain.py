"""Regression checks for the error-explainability tool (src/evaluation/explain.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.explain import (
    add_error_columns, bias_curve, build_report, compare, segment_report, worst_cases,
)


def _make_df(n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ilce = rng.choice(["A", "B"], size=n)
    y = rng.normal(2.0, 0.5, size=n)
    err = np.where(ilce == "B", 0.5, 0.01) * rng.normal(size=n)  # B is the bad segment
    pred = y + err
    feat = rng.normal(size=n)
    return pd.DataFrame({"id": np.arange(n), "y": y, "pred": pred, "ilce": ilce, "feat": feat})


def test_segment_report_flags_the_disproportionate_segment() -> None:
    df = add_error_columns(_make_df(), "y", "pred")
    rep = segment_report(df, "ilce", min_n=5)
    # B has the larger injected error, so its share_of_error must dominate its share_of_rows.
    assert rep.loc["B", "lift"] > rep.loc["A", "lift"]
    assert rep.loc["B", "share_of_error"] > rep.loc["B", "share_of_rows"]


def test_bias_curve_detects_monotonic_miscalibration() -> None:
    n = 500
    feat = np.linspace(-2, 2, n)
    y = np.zeros(n)
    pred = 0.1 * feat  # bias grows monotonically with feat
    df = add_error_columns(pd.DataFrame({"y": y, "pred": pred, "feat": feat}), "y", "pred")
    curve = bias_curve(df, "feat", bins=8, min_n=5)
    assert curve.attrs["monotonic_bias"] is True


def test_bias_curve_no_trend_when_error_is_noise() -> None:
    rng = np.random.default_rng(1)
    n = 500
    feat = rng.normal(size=n)
    y = np.zeros(n)
    pred = rng.normal(scale=0.01, size=n)  # unrelated to feat
    df = add_error_columns(pd.DataFrame({"y": y, "pred": pred, "feat": feat}), "y", "pred")
    curve = bias_curve(df, "feat", bins=8, min_n=5)
    assert curve.attrs["monotonic_bias"] is False


def test_worst_cases_returns_largest_abs_err_first() -> None:
    df = add_error_columns(_make_df(n=50), "y", "pred")
    top = worst_cases(df, id_cols=["id"], k=5)
    assert len(top) == 5
    assert list(top["abs_err"]) == sorted(top["abs_err"], reverse=True)


def test_compare_isolates_the_regressed_segment() -> None:
    base = _make_df(seed=0)
    cand = base.copy()
    # Candidate makes segment B strictly worse, leaves A untouched.
    cand.loc[cand["ilce"] == "B", "pred"] += 1.0
    diff = compare(base, cand, id_col="id", y_col="y", pred_col="pred", group_cols=["ilce"],
                    min_n=5, noise_floor=0.01)
    assert "B" in diff.regressed.index
    assert "B" not in diff.improved.index
    assert diff.delta_rmsle > 0


def test_build_report_end_to_end_smoke() -> None:
    df = _make_df()
    report = build_report(df, y_col="y", pred_col="pred", group_cols=["ilce"],
                           feature_cols=["feat"], id_cols=["id"], name="smoke", min_n=5)
    assert report["overall"]["n"] == len(df)
    assert "ilce" in report["segments"]
    assert "feat" in report["bias_curves"]
    assert len(report["worst_cases"]) == 20


if __name__ == "__main__":
    test_segment_report_flags_the_disproportionate_segment()
    test_bias_curve_detects_monotonic_miscalibration()
    test_bias_curve_no_trend_when_error_is_noise()
    test_worst_cases_returns_largest_abs_err_first()
    test_compare_isolates_the_regressed_segment()
    test_build_report_end_to_end_smoke()
    print("explain tests passed")
