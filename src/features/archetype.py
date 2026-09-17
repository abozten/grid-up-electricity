"""Transformer consumption archetype and load signature classification."""
from __future__ import annotations

import numpy as np
import pandas as pd


class ArchetypeFeatureExtractor:
    """Classifies transformers into behavioral archetypes (industrial, agro, residential, coastal)."""

    @classmethod
    def extract_archetypes(cls, hist: pd.DataFrame) -> pd.DataFrame:
        """Compute behavioral load ratios and inferred categorical archetype causally from history."""
        if hist.empty or "tanim" not in hist.columns or "tarih" not in hist.columns:
            return pd.DataFrame()

        hist = hist.copy()
        if not pd.api.types.is_datetime64_any_dtype(hist["tarih"]):
            hist["tarih"] = pd.to_datetime(hist["tarih"])

        hist["dow"] = hist["tarih"].dt.dayofweek  # 5=Sat, 6=Sun
        hist["month"] = hist["tarih"].dt.month

        out = pd.DataFrame(index=hist["tanim"].unique())
        out.index.name = "tanim"

        # 1. Sunday to Weekday Ratio (Industrial vs Residential vs Tourism)
        sunday_data = hist[hist["dow"] == 6]
        weekday_data = hist[hist["dow"] < 5]

        sunday_mean = sunday_data.groupby("tanim")["tuketim"].mean()
        weekday_mean = weekday_data.groupby("tanim")["tuketim"].mean().replace(0, 1.0)
        out["sunday_to_weekday_ratio"] = (sunday_mean / weekday_mean).fillna(1.0).clip(0.1, 3.0)

        # 2. Summer to Winter Ratio (Seasonal cooling/irrigation/tourism vs Flat)
        summer_data = hist[hist["month"].isin([6, 7, 8])]
        winter_data = hist[hist["month"].isin([1, 2, 12])]

        summer_mean = summer_data.groupby("tanim")["tuketim"].mean()
        winter_mean = winter_data.groupby("tanim")["tuketim"].mean().replace(0, 1.0)
        out["summer_to_winter_ratio"] = (summer_mean / winter_mean).fillna(1.0).clip(0.1, 10.0)

        # 3. Load Volatility (CV = sigma / mu)
        trafo_std = hist.groupby("tanim")["tuketim"].std().fillna(0.0)
        trafo_mean = hist.groupby("tanim")["tuketim"].mean().replace(0, 1.0)
        out["load_volatility_cv"] = (trafo_std / trafo_mean).fillna(0.0).clip(0.0, 5.0)

        # 4. Inferred Archetype Integer Encoding
        # 1: Industrial 6-day (Sunday drop < 0.60)
        # 2: Agro / Irrigation (Summer/Winter ratio > 3.0)
        # 3: Coastal / Tourism (Sunday ratio > 1.15 & Summer ratio > 1.8)
        # 4: Commercial 5-day (Weekend ratio < 0.40)
        # 5: 24/7 Infrastructure (CV < 0.20)
        # 6: Residential / Mixed (Default)
        archetypes = pd.Series(6, index=out.index)  # Default: 6

        is_infra = out["load_volatility_cv"] < 0.20
        is_agro = out["summer_to_winter_ratio"] > 3.0
        is_ind = out["sunday_to_weekday_ratio"] < 0.60
        is_tourism = (out["sunday_to_weekday_ratio"] > 1.15) & (out["summer_to_winter_ratio"] > 1.8)
        is_comm = out["sunday_to_weekday_ratio"] < 0.40

        archetypes[is_infra] = 5
        archetypes[is_agro] = 2
        archetypes[is_ind] = 1
        archetypes[is_comm] = 4
        archetypes[is_tourism] = 3

        out["archetype_code"] = archetypes.astype(float)
        return out
