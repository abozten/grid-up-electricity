"""District-level heat sensitivity, regional trends, and peer comparisons."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.history import compute_group_slope


class DistrictFeatureExtractor:
    """Computes district-level aggregated dynamics and peer group benchmarks."""

    GUC_BINS = [0, 100, 250, 400, 630, 1000, 1600, 1e9]

    @classmethod
    def extract_district_dynamics(cls, hist: pd.DataFrame) -> pd.DataFrame:
        """Extract district-level heat slopes and recent dynamics (transfers to cold-start)."""
        districts = sorted(hist["ilce"].astype(str).unique())
        out = pd.DataFrame(index=pd.Index(districts, name="ilce_str"))
        h = hist.copy()
        h["ilce_str"] = h["ilce"].astype(str)

        base = h.groupby("ilce_str")["log_t"].mean()

        if "cdd22" in h.columns:
            out["ilce_cdd_slope"] = compute_group_slope(h, "cdd22", "log_t", "ilce_str")
            out["ilce_cddlog_slope"] = (compute_group_slope(h, "cdd_log", "log_t", "ilce_str")
                                        if "cdd_log" in h.columns else np.nan)
            summer = h[h["cdd22"] > 0]
            if not summer.empty:
                out["ilce_summer_mean"] = summer.groupby("ilce_str")["log_t"].mean()
                out["ilce_summer_delta"] = out["ilce_summer_mean"] - base
            else:
                out["ilce_summer_mean"] = np.nan
                out["ilce_summer_delta"] = np.nan
        else:
            out["ilce_cdd_slope"] = np.nan
            out["ilce_cddlog_slope"] = np.nan
            out["ilce_summer_mean"] = np.nan
            out["ilce_summer_delta"] = np.nan

        if "temperature_2m_mean" in h.columns:
            hot = h[h["temperature_2m_mean"] > 25.0]
            if not hot.empty:
                out["ilce_hot_delta"] = hot.groupby("ilce_str")["log_t"].mean() - base
            else:
                out["ilce_hot_delta"] = np.nan
        else:
            out["ilce_hot_delta"] = np.nan

        last_date = h["tarih"].max()
        for w, nm in [(28, "ilce_r28_mean"), (91, "ilce_r91_mean")]:
            out[nm] = h[h["tarih"] > last_date - pd.Timedelta(days=w)].groupby("ilce_str")["log_t"].mean()
        out["ilce_trend"] = out["ilce_r28_mean"] - out["ilce_r91_mean"]

        r28 = h[h["tarih"] > last_date - pd.Timedelta(days=28)]
        if not r28.empty and "is_pos" in r28.columns:
            out["ilce_r28_zero"] = 1.0 - r28.groupby("ilce_str")["is_pos"].mean()
        else:
            out["ilce_r28_zero"] = np.nan

        return out.reset_index()

    @classmethod
    def extract_peer_benchmarks(cls, hist: pd.DataFrame) -> pd.DataFrame:
        """Extract benchmark consumption levels of comparable capacity classes within district."""
        last_date = hist["tarih"].max()
        rec = hist[hist["tarih"] > last_date - pd.Timedelta(days=91)].copy()
        if rec.empty:
            return pd.DataFrame(columns=["ilce_str", "guc_bucket", "peer_mean", "peer_n"])
        rec["ilce_str"] = rec["ilce"].astype(str)
        rec["guc_bucket"] = pd.cut(rec["guc"], cls.GUC_BINS, labels=False)
        peers = (
            rec.groupby(["ilce_str", "guc_bucket"])["log_t"]
            .agg(peer_mean="mean", peer_n="size")
            .reset_index()
        )
        return peers
