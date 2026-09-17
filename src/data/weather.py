"""Weather data ingestion and derived index computation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PathConfig, FeatureConfig


class WeatherLoader:
    """Loads and computes derived meteorological and bioclimatic indicators."""

    def __init__(self, paths: PathConfig | None = None, features: FeatureConfig | None = None) -> None:
        self.paths = paths or PathConfig()
        self.features = features or FeatureConfig()

    def load_weather(self) -> pd.DataFrame:
        """Load per-district weather and compute derived heating/cooling/drought metrics."""
        wx = pd.read_csv(self.paths.weather_path, parse_dates=["tarih"])
        wx = wx.drop(columns=[c for c in ("uv_index_max", "uv_index_clear_sky_max") if c in wx.columns])

        # Degree days and astronomical variables
        wx["hdd18"] = (18.0 - wx["temperature_2m_mean"]).clip(lower=0.0)
        wx["cdd22"] = (wx["temperature_2m_mean"] - 22.0).clip(lower=0.0)
        # Cooling response SATURATES in log space. Fleet lift vs the 18-22C base, measured on
        # train.csv: +0.06 at 22-24C, +0.25 at 24-26C, +0.57 at 26-28C, +0.75 at 28-30C, then only
        # +0.86 at 30-32C and +0.90 above 32C -- the increments shrink from 0.31 to 0.05. A LINEAR
        # cdd22 used as a regressor therefore keeps extrapolating a rise that stops happening.
        # log1p is the concave form with nothing to tune, and it beat a tuned exponential out of
        # range. See reports/EDA_FEATURES.md for the holdout numbers.
        wx["cdd_log"] = np.log1p(wx["cdd22"])
        wx["sunshine_hours"] = wx["sunshine_duration"] / 3600.0
        wx["daylight_hours"] = wx["daylight_duration"] / 3600.0
        wx["sunshine_ratio"] = wx["sunshine_duration"] / wx["daylight_duration"].replace(0, np.nan)
        wx["apparent_delta"] = wx["apparent_temperature_mean"] - wx["temperature_2m_mean"]
        wx["temp_range"] = wx["temperature_2m_max"] - wx["temperature_2m_min"]

        # Sort for temporal rolling computations
        wx = wx.sort_values(["ilce", "tarih"]).reset_index(drop=True)
        grp = wx.groupby("ilce")

        wx["tmean_r7x"] = grp["temperature_2m_mean"].transform(lambda s: s.rolling(7, min_periods=1).mean())
        wx["cdd_r7"] = grp["cdd22"].transform(lambda s: s.rolling(7, min_periods=1).mean())
        wx["hdd_r7"] = grp["hdd18"].transform(lambda s: s.rolling(7, min_periods=1).mean())
        wx["rad_r7"] = grp["shortwave_radiation_sum"].transform(lambda s: s.rolling(7, min_periods=1).mean())

        # Heat accumulation & heatwaves
        wx["cdd_acc"] = grp["cdd22"].transform(lambda s: s.rolling(14, min_periods=1).sum())
        wx["heat_wave_3d"] = grp["temperature_2m_max"].transform(
            lambda s: (s >= 33.0).rolling(3, min_periods=1).sum()
        )
        wx["hot_days_7d"] = grp["temperature_2m_max"].transform(
            lambda s: (s >= 30.0).rolling(7, min_periods=1).sum()
        )

        # Water deficit and drought indicators
        deficit = (wx["et0_fao_evapotranspiration"] - wx["precipitation_sum"]).clip(lower=0.0)
        wx["_deficit"] = deficit
        def_grp = wx.groupby("ilce")["_deficit"]
        wx["water_deficit_7d"] = def_grp.transform(lambda s: s.rolling(7, min_periods=1).sum())
        wx["water_deficit_14d"] = def_grp.transform(lambda s: s.rolling(14, min_periods=1).sum())
        wx["water_deficit_30d"] = def_grp.transform(lambda s: s.rolling(30, min_periods=1).sum())
        wx.drop(columns=["_deficit"], inplace=True)

        wx["et0_cum_30d"] = grp["et0_fao_evapotranspiration"].transform(lambda s: s.rolling(30, min_periods=1).sum())
        wx["rain_cum_30d"] = grp["rain_sum"].transform(lambda s: s.rolling(30, min_periods=1).sum())

        # Days since last rain and dry spells
        has_rain = wx["precipitation_sum"] > 0.5
        rain_event_id = has_rain.groupby(wx["ilce"]).cumsum()
        wx["days_since_rain"] = (~has_rain).astype(int).groupby([wx["ilce"], rain_event_id]).cumsum()
        wx["dry_spell_len"] = (~has_rain).astype(int).groupby([wx["ilce"], rain_event_id]).cumsum()

        # Soil moisture anomaly proxy & vapor pressure deficit proxy
        wx["soil_moist_anom"] = wx["rain_cum_30d"] - wx["et0_cum_30d"]
        vpd_raw = (
            0.61078 * np.exp(17.27 * wx["temperature_2m_mean"] / (wx["temperature_2m_mean"] + 237.3))
            * (1.0 - wx["relative_humidity_2m_mean"] / 100.0)
        )
        wx["vpd_r7"] = vpd_raw.groupby(wx["ilce"]).transform(lambda s: s.rolling(7, min_periods=1).mean())

        feature_cols = ["ilce", "tarih"] + list(self.features.weather_raw) + list(self.features.weather_derived)
        available_cols = [c for c in feature_cols if c in wx.columns]
        return wx[available_cols]
