"""Seasonal agricultural irrigation, water deficit, and delayed activation feature extraction."""
from __future__ import annotations

import numpy as np
import pandas as pd


class AgroFeatureExtractor:
    """Extracts features for seasonal agricultural irrigation and water balance."""

    HEAVY_AGRO_DISTRICTS = {
        "SALİHLİ", "ALAŞEHİR", "SARIGÖL", "GÖRDES", "ÖDEMİŞ",
        "BEYDAĞ", "KIRKAĞAÇ", "KİRAZ", "AKHİSAR", "KULA", "DEMİRCİ"
    }

    @classmethod
    def extract_agro_profile(cls, hist: pd.DataFrame) -> pd.DataFrame:
        """Extract transformer-level seasonal concentration and irrigation profile from history."""
        if hist.empty or "tanim" not in hist.columns or "tarih" not in hist.columns:
            return pd.DataFrame()

        hist = hist.copy()
        if not pd.api.types.is_datetime64_any_dtype(hist["tarih"]):
            hist["tarih"] = pd.to_datetime(hist["tarih"])

        out = pd.DataFrame(index=hist["tanim"].unique())
        out.index.name = "tanim"

        # 1. Summer observation concentration ratio
        hist["month"] = hist["tarih"].dt.month
        summer_rows = hist[hist["month"].isin([6, 7, 8])].groupby("tanim").size()
        total_rows = hist.groupby("tanim").size()

        summer_ratio = (summer_rows / total_rows).fillna(0.0)
        out["summer_concentration_ratio"] = summer_ratio.clip(0.0, 1.0)
        out["is_summer_exclusive"] = (out["summer_concentration_ratio"] > 0.80).astype(float)

        # 2. Irrigation Ramp Factor (Active days in season)
        # Average consumption ratio between late summer (July/Aug) and early season (May)
        may_mask = hist["month"] == 5
        jul_mask = hist["month"].isin([7, 8])
        may_mean = hist[may_mask].groupby("tanim")["tuketim"].mean().replace(0, 1.0)
        jul_mean = hist[jul_mask].groupby("tanim")["tuketim"].mean()

        out["agro_summer_expansion_ratio"] = (jul_mean / may_mean).fillna(1.0).clip(0.5, 5.0)

        return out

    @classmethod
    def enrich_agro_weather(cls, df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Enrich dataframe with cumulative water deficit and agro-district interactions."""
        df = df.copy()
        if "tarih" not in df.columns:
            return df

        if not pd.api.types.is_datetime64_any_dtype(df["tarih"]):
            df["tarih"] = pd.to_datetime(df["tarih"])

        # Heavy Agro District Flag
        if "ilce" in df.columns:
            df["is_heavy_agro_ilce"] = df["ilce"].astype(str).str.upper().isin(cls.HEAVY_AGRO_DISTRICTS).astype(float)
        else:
            df["is_heavy_agro_ilce"] = 0.0

        # Compute cumulative water deficit if weather data is present
        if "et0_fao_evapotranspiration" in df.columns and "rain_sum" in df.columns:
            # Daily deficit: ET0 - rain (clipped at 0)
            daily_def = (df["et0_fao_evapotranspiration"] - df["rain_sum"]).clip(lower=0.0)
            
            # Cumulative from May 1 of each year
            yr = df["tarih"].dt.year
            in_season = (df["tarih"].dt.month >= 5) & (df["tarih"].dt.month <= 8)
            df["_temp_def"] = daily_def * in_season.astype(float)
            
            grp_col = df["ilce_str"] if "ilce_str" in df.columns else (df["ilce"] if "ilce" in df.columns else None)
            if grp_col is not None:
                df["cum_water_deficit"] = df.groupby([grp_col, yr])["_temp_def"].cumsum().values
            else:
                df["cum_water_deficit"] = df.groupby(yr)["_temp_def"].cumsum().values
            df.drop(columns=["_temp_def"], inplace=True)
            df["agro_district_x_et0"] = df["is_heavy_agro_ilce"] * df["et0_fao_evapotranspiration"]
        else:
            df["cum_water_deficit"] = 0.0
            df["agro_district_x_et0"] = 0.0

        # Delayed start flag (May/June activation)
        df["is_delayed_start"] = (df["tarih"].dt.month >= 5).astype(float) * df["is_heavy_agro_ilce"]

        return df
