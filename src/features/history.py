"""Historical consumption aggregations and transformer profile extractors."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_group_slope(df: pd.DataFrame, xcol: str, ycol: str, key: str) -> pd.Series:
    """Compute linear regression slope y on x grouped by key in a vectorized manner."""
    d = df[[key, xcol, ycol]].dropna().copy()
    if d.empty:
        return pd.Series(dtype=float)
    d["_xy"] = d[xcol] * d[ycol]
    d["_xx"] = d[xcol] ** 2
    grp = d.groupby(key)
    n = grp.size()
    sx = grp[xcol].sum()
    sy = grp[ycol].sum()
    cov = grp["_xy"].sum() / n - (sx / n) * (sy / n)
    var = grp["_xx"].sum() / n - (sx / n) ** 2
    slope = cov / var.replace(0.0, np.nan)
    return slope


class HistoryFeatureExtractor:
    """Computes leak-free historical aggregates strictly from prior observations."""

    @classmethod
    def extract_aggregates(cls, hist: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Extract transformer and hierarchical profile tables from pre-cutoff historical data."""
        g = hist.groupby("tanim")["log_t"]
        tan = pd.DataFrame({
            "tan_mean": g.mean(),
            "tan_median": g.median(),
            "tan_std": g.std(),
            "tan_min": g.min(),
            "tan_max": g.max(),
            "tan_zero": hist.groupby("tanim")["is_pos"].apply(lambda s: 1.0 - s.mean()),
            "tan_cnt": g.size().astype(float),
            "tan_p10": g.quantile(0.10),
            "tan_p25": g.quantile(0.25),
            "tan_p75": g.quantile(0.75),
            "tan_p90": g.quantile(0.90),
        })
        tan["tan_iqr"] = tan["tan_p75"] - tan["tan_p25"]

        # Recency rolling metrics
        last_date = hist["tarih"].max()
        for window in (7, 28, 91):
            recent = hist[hist["tarih"] > last_date - pd.Timedelta(days=window)]
            rg = recent.groupby("tanim")["log_t"]
            tan = tan.join(rg.mean().rename(f"tan_r{window}_mean"))
            tan = tan.join((1.0 - recent.groupby("tanim")["is_pos"].mean()).rename(f"tan_r{window}_zero"))
            if window > 7:
                tan = tan.join(rg.std().rename(f"tan_r{window}_std"))

        tan["tan_trend"] = tan["tan_r28_mean"] - tan["tan_r91_mean"]
        tan["tan_zero_trend"] = tan["tan_r28_zero"] - tan["tan_zero"]

        # 90-day trajectory slope per transformer
        rec90 = hist[hist["tarih"] > last_date - pd.Timedelta(days=90)].copy()
        if not rec90.empty:
            rec90["_t"] = (rec90["tarih"] - (last_date - pd.Timedelta(days=90))).dt.days.astype(float)
            slope90 = compute_group_slope(rec90, "_t", "log_t", "tanim")
            tan["tan_slope90"] = slope90
        else:
            tan["tan_slope90"] = np.nan

        # Heat response per transformer if CDD / weather is available
        if "cdd22" in hist.columns:
            tan["tan_cdd_slope"] = compute_group_slope(hist, "cdd22", "log_t", "tanim")
            # Same slope against the saturating regressor. Added ALONGSIDE the linear one rather
            # than replacing it, so the two can be gated head to head; the measured evidence says
            # replace (reports/EDA_FEATURES.md), but tan_cdd_slope is a shipped v33 feature.
            tan["tan_cddlog_slope"] = (compute_group_slope(hist, "cdd_log", "log_t", "tanim")
                                       if "cdd_log" in hist.columns else np.nan)
            summer_mask = hist["cdd22"] > 0
            if summer_mask.any():
                summer_mean = hist[summer_mask].groupby("tanim")["log_t"].mean()
                tan["tan_summer_mean"] = summer_mean
                tan["tan_summer_delta"] = summer_mean - tan["tan_mean"]
            else:
                tan["tan_summer_mean"] = np.nan
                tan["tan_summer_delta"] = np.nan
        else:
            tan["tan_cdd_slope"] = np.nan
            tan["tan_cddlog_slope"] = np.nan
            tan["tan_summer_mean"] = np.nan
            tan["tan_summer_delta"] = np.nan

        if "temperature_2m_mean" in hist.columns:
            hot_mask = hist["temperature_2m_mean"] > 25.0
            if hot_mask.any():
                tan["tan_hot_delta"] = hist[hot_mask].groupby("tanim")["log_t"].mean() - tan["tan_mean"]
            else:
                tan["tan_hot_delta"] = np.nan
        else:
            tan["tan_hot_delta"] = np.nan

        # Month and Day-of-week profiles
        tm = hist.groupby(["tanim", "month"])["log_t"].mean().rename("tan_moy_mean").reset_index()
        tdw = hist.groupby(["tanim", "dow"])["log_t"].mean().rename("tan_dow_mean").reset_index()

        # Regional & capacity fallbacks
        ilce = hist.groupby("ilce")["log_t"].mean().rename("ilce_mean")
        ilce_m = hist.groupby(["ilce", "month"])["log_t"].mean().rename("ilce_moy_mean").reset_index()
        gucm = hist.groupby("guc")["log_t"].mean().rename("guc_mean")
        gucmed = hist.groupby("guc")["log_t"].median().rename("guc_median")
        gi = hist.groupby(["ilce", "guc"])["log_t"].mean().rename("guc_ilce_mean").reset_index()
        gi_n = hist.groupby(["ilce", "guc"])["log_t"].size().rename("guc_ilce_n").reset_index()

        # Load factor proxies
        hist_lf = hist.assign(lf=hist["tuketim"] / (hist["guc"].clip(lower=1.0) * 24.0))
        tan = tan.join(hist_lf.groupby("tanim")["lf"].mean().rename("tan_lf"))
        ilce_lf = hist_lf.groupby("ilce")["lf"].mean().rename("ilce_lf")
        ilce_lf_m = hist_lf.groupby(["ilce", "month"])["lf"].mean().rename("ilce_lf_moy").reset_index()

        # Same Day Last Year table
        sdly = hist[["tanim", "tarih", "log_t"]].copy()

        return {
            "tan": tan,
            "tm": tm,
            "tdw": tdw,
            "ilce": ilce,
            "ilce_m": ilce_m,
            "gucm": gucm,
            "gucmed": gucmed,
            "gi": gi,
            "gi_n": gi_n,
            "sdly": sdly,
            "ilce_lf": ilce_lf,
            "ilce_lf_m": ilce_lf_m,
        }
