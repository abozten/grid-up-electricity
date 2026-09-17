"""Regime shift, level shift, and network maneuver feature extraction."""
from __future__ import annotations

import numpy as np
import pandas as pd


class RegimeFeatureExtractor:
    """Extracts features capturing structural breaks, load transfers, and capacity shifts."""

    @classmethod
    def extract_regime_features(cls, hist: pd.DataFrame, window_days: int = 60) -> pd.DataFrame:
        """Extract level shift ratio, persistence, and volatility features causally from history.

        hist: strictly pre-cutoff historical training data.
        """
        if hist.empty or "tanim" not in hist.columns or "tarih" not in hist.columns:
            return pd.DataFrame()

        hist = hist.copy()
        if not pd.api.types.is_datetime64_any_dtype(hist["tarih"]):
            hist["tarih"] = pd.to_datetime(hist["tarih"])

        max_date = hist["tarih"].max()
        recent_cut = max_date - pd.Timedelta(days=window_days)

        recent_mask = hist["tarih"] >= recent_cut
        recent_data = hist[recent_mask]
        older_data = hist[~recent_mask]

        out = pd.DataFrame(index=hist["tanim"].unique())
        out.index.name = "tanim"

        # 1. Recent vs Older Median Level Shift Ratio
        recent_med = recent_data[recent_data["tuketim"] > 0].groupby("tanim")["tuketim"].median()
        older_med = older_data[older_data["tuketim"] > 0].groupby("tanim")["tuketim"].median().replace(0, 1.0)
        lifetime_med = hist[hist["tuketim"] > 0].groupby("tanim")["tuketim"].median().replace(0, 1.0)

        # Level shift ratio: recent vs older (default 1.0 if not enough older history)
        level_ratio = (recent_med / older_med).fillna(recent_med / lifetime_med).fillna(1.0)
        out["level_shift_ratio_60d"] = level_ratio.clip(0.1, 10.0)

        # 2. Log-space Level Difference (directly matches RMSLE loss metric)
        recent_mean_log = np.log1p(recent_data.groupby("tanim")["tuketim"].mean().fillna(0.0))
        lifetime_mean_log = np.log1p(hist.groupby("tanim")["tuketim"].mean().replace(0, 1.0))
        out["recent_vs_lifetime_log_diff"] = (recent_mean_log - lifetime_mean_log).fillna(0.0).clip(-5.0, 5.0)

        # 3. Recent Regime Volatility (Coefficient of Variation: sigma / mu)
        recent_std = recent_data.groupby("tanim")["tuketim"].std().fillna(0.0)
        recent_mean = recent_data.groupby("tanim")["tuketim"].mean().replace(0, 1.0)
        out["recent_load_cv"] = (recent_std / recent_mean).fillna(0.0).clip(0.0, 5.0)

        # 4. Scaled Summer Baseline (Projecting new level onto historical summer profile)
        summer_mask = hist["tarih"].dt.month.isin([6, 7, 8])
        summer_data = hist[summer_mask]
        summer_mean = summer_data.groupby("tanim")["tuketim"].mean()

        # Scaled summer: old summer mean * level shift ratio
        out["scaled_summer_baseline"] = np.log1p((summer_mean * out["level_shift_ratio_60d"]).fillna(recent_mean))

        # 5. Consecutive Days in Elevated Regime
        # How many days the transformer stayed above 1.5x of its lifetime median
        hist_ordered = hist.sort_values(["tanim", "tarih"])
        hist_ordered["base_med"] = hist_ordered["tanim"].map(lifetime_med).fillna(1.0)
        hist_ordered["is_elevated"] = (hist_ordered["tuketim"] > (1.5 * hist_ordered["base_med"])).astype(int)

        # Count consecutive trailing days elevated
        def trailing_streak(series: pd.Series) -> int:
            arr = series.values
            count = 0
            for val in reversed(arr):
                if val == 1:
                    count += 1
                else:
                    break
            return count

        elev_streaks = hist_ordered.groupby("tanim")["is_elevated"].apply(trailing_streak)
        out["regime_persistence_days"] = elev_streaks.fillna(0).clip(0, 120)

        return out
