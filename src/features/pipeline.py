"""Feature engineering pipeline orchestrator ensuring leak-free snapshot extraction."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.config import Config, FeatureConfig
from src.data.regime import RegimeDetector
from src.features.calendar import CalendarFeatureExtractor
from src.features.history import HistoryFeatureExtractor
from src.features.district import DistrictFeatureExtractor
from src.features.presence import PresenceFeatureExtractor


class FeaturePipeline:
    """Orchestrates leak-free feature matrix construction for training snapshots and test set."""

    def __init__(self, config: Config | None = None, use_regime: bool | None = None,
                 use_presence: bool | None = None) -> None:
        self.config = config or Config()
        self.feature_cfg = self.config.features
        # Opt-in so the regime-aware and baseline runs are directly comparable.
        self.use_regime = (os.environ.get("GRIDUP_REGIME", "0") == "1"
                           if use_regime is None else use_regime)
        # The detector does two separable things: restate history into current-regime
        # units, and expose the regime state as columns. A per-fold decomposition showed
        # the restatement costs LightGBM +0.0070 pooled while the columns win back exactly
        # that, so the two must be togglable independently to find the better combination.
        self.normalize_history = os.environ.get("GRIDUP_REGIME_NORM", "1") == "1"
        # Presence features (src/features/presence.py) read only the test file's own row
        # calendar, never a consumption value.
        self.use_presence = (os.environ.get("GRIDUP_PRESENCE", "0") == "1"
                             if use_presence is None else use_presence)

    def _prepare_history(self, hist: pd.DataFrame) -> tuple[pd.DataFrame, RegimeDetector | None]:
        """Restate history in current-regime units before any aggregate is taken.

        `hist` is always strictly pre-cutoff, so fitting the detector here stays causal.
        Only metering-scale breaks are unwound; genuine load steps are left in place and
        surfaced to the model through the regime_* state columns instead.
        """
        if not self.use_regime or hist.empty:
            return hist, None
        det = RegimeDetector().fit(hist)
        if not self.normalize_history:
            return hist, det  # state columns only; history left in raw units
        return det.normalize(hist), det

    def enrich_calendar_and_weather(self, df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Merge calendar features and district-level weather features."""
        df = CalendarFeatureExtractor.transform(df)
        df["ilce_str"] = df["ilce"].astype(str)
        wx = weather_df.copy()
        wx["ilce_str"] = wx["ilce"].astype(str)
        if "ilce" in wx.columns:
            wx = wx.drop(columns=["ilce"])
        df = df.merge(wx, on=["ilce_str", "tarih"], how="left")
        return df

    def attach_features(
        self,
        df: pd.DataFrame,
        hist: pd.DataFrame,
        aggs: dict[str, pd.DataFrame],
        cutoff_date: pd.Timestamp | str,
        detector: RegimeDetector | None = None,
    ) -> pd.DataFrame:
        """Attach historical aggregates and district dynamics."""
        df = df.copy()
        cutoff_ts = pd.Timestamp(cutoff_date)
        df["horizon"] = (df["tarih"] - cutoff_ts).dt.days

        # Row-calendar shape inside the block being predicted. Opt-in so the presence run
        # and the baseline are directly comparable, exactly as the regime group is gated.
        if self.use_presence:
            df = PresenceFeatureExtractor.transform(df)

        # Attach transformer and hierarchical tables
        df = df.merge(aggs["tan"], left_on="tanim", right_index=True, how="left")
        df = df.merge(aggs["tm"], on=["tanim", "month"], how="left")
        df = df.merge(aggs["tdw"], on=["tanim", "dow"], how="left")
        df = df.merge(aggs["ilce"], left_on="ilce", right_index=True, how="left")
        df = df.merge(aggs["ilce_m"], on=["ilce", "month"], how="left")
        df = df.merge(aggs["gucm"], left_on="guc", right_index=True, how="left")
        df = df.merge(aggs["gucmed"], left_on="guc", right_index=True, how="left")
        df = df.merge(aggs["gi"], on=["ilce", "guc"], how="left")
        df = df.merge(aggs["gi_n"], on=["ilce", "guc"], how="left")

        # Same Day Last Year (lag 364)
        s = aggs["sdly"]
        lag364 = s.assign(tarih=s["tarih"] + pd.Timedelta(days=364)).rename(columns={"log_t": "sdly_364"})
        df = df.merge(lag364, on=["tanim", "tarih"], how="left")

        # Fallbacks for missing levels
        df["tan_moy_mean"] = df["tan_moy_mean"].fillna(df["tan_mean"])
        df["tan_dow_mean"] = df["tan_dow_mean"].fillna(df["tan_mean"])
        for c in ["tan_r7_mean", "tan_r28_mean", "tan_r91_mean"]:
            if c in df.columns:
                df[c] = df[c].fillna(df["tan_mean"])
        df["sdly_364"] = df["sdly_364"].fillna(df["tan_moy_mean"])

        df["seen"] = df["tan_mean"].notna().astype(int)

        for c in ["tan_mean", "tan_median", "tan_r7_mean", "tan_r28_mean", "tan_moy_mean", "tan_dow_mean", "sdly_364"]:
            if c in df.columns:
                df[c] = df[c].fillna(df["guc_ilce_mean"]).fillna(df["ilce_mean"])

        # Load factors & physics anchors
        df = df.merge(aggs["ilce_lf"], left_on="ilce", right_index=True, how="left")
        df = df.merge(aggs["ilce_lf_m"], on=["ilce", "month"], how="left")
        df["log_guc"] = np.log1p(df["guc"])
        cap = df["guc"].clip(lower=1.0) * 24.0
        df["log_cap"] = np.log1p(cap)
        df["exp_log_ilce"] = np.log1p(df["ilce_lf_moy"].fillna(df["ilce_lf"]) * cap)
        df["exp_log_tan"] = np.log1p(df["tan_lf"] * cap)

        # District dynamics & peer groups
        dist_dyn = DistrictFeatureExtractor.extract_district_dynamics(hist)
        df = df.merge(dist_dyn, on="ilce_str", how="left")

        df["guc_bucket"] = pd.cut(df["guc"], DistrictFeatureExtractor.GUC_BINS, labels=False)
        peer_bench = DistrictFeatureExtractor.extract_peer_benchmarks(hist)
        df = df.merge(peer_bench, on=["ilce_str", "guc_bucket"], how="left")

        # Row-level derived interactions
        if "tan_slope90" in df.columns:
            df["tan_drift_h"] = df["tan_slope90"].fillna(0.0) * df["horizon"]
        else:
            df["tan_drift_h"] = np.nan

        if "peer_mean" in df.columns and "tan_mean" in df.columns:
            df["tan_vs_peer"] = df["tan_mean"] - df["peer_mean"]
        else:
            df["tan_vs_peer"] = np.nan

        if detector is not None:
            df = detector.attach_state(df)

        for c in self.feature_cfg.categorical_cols:
            if c in df.columns:
                df[c] = df[c].astype("category")

        return df

    def build_snapshot_block(
        self,
        full_train: pd.DataFrame,
        feat_cut: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Construct a leakage-free snapshot training block."""
        hist = full_train[full_train["tarih"] < feat_cut]
        target_rows = full_train[(full_train["tarih"] >= start_date) & (full_train["tarih"] < end_date)].copy()
        hist, det = self._prepare_history(hist)
        aggs = HistoryFeatureExtractor.extract_aggregates(hist)
        return self.attach_features(target_rows, hist, aggs, feat_cut, det)

    def build_test_block(
        self,
        full_train: pd.DataFrame,
        test_df: pd.DataFrame,
        test_cut: str = "2026-04-01",
    ) -> pd.DataFrame:
        """Construct the test feature matrix using all training history as pre-cutoff information."""
        hist, det = self._prepare_history(full_train)
        aggs = HistoryFeatureExtractor.extract_aggregates(hist)
        return self.attach_features(test_df.copy(), hist, aggs, test_cut, det)

    def get_feature_names(self, df: pd.DataFrame) -> list[str]:
        """Return list of valid features present in the dataframe excluding target and metadata."""
        exclude = {"id", "tarih", "tanim", "lokasyon", "tuketim", "log_t", "is_pos", "ilce_str", "guc_bucket"}
        return [c for c in df.columns if c not in exclude]
