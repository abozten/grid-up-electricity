"""Postprocessing pipeline: dormant wake-up floors, expm1 conversion, and clipping."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import PathConfig, ModelConfig


class PostProcessor:
    """Handles domain postprocessing: dormant wake-up floor, exponentiation, and validation."""

    def __init__(
        self,
        paths: PathConfig | None = None,
        config: ModelConfig | None = None,
    ) -> None:
        self.paths = paths or PathConfig()
        self.config = config or ModelConfig()

    @staticmethod
    def identify_dormant_transformers(train_df: pd.DataFrame, cutoff: str = "2026-04-01") -> set[str]:
        """Identify transformers that had zero consumption during the last 28 days before cutoff."""
        last_dt = pd.Timestamp(cutoff)
        w28 = train_df[(train_df["tarih"] > last_dt - pd.Timedelta(days=28)) & (train_df["tarih"] <= last_dt)]
        grp = w28.groupby("tanim")
        all_zero = grp["tuketim"].max() == 0.0
        return set(all_zero[all_zero].index)

    @staticmethod
    def identify_prior_active(train_df: pd.DataFrame, dormant_set: set[str], cutoff: str = "2026-04-01") -> dict[str, bool]:
        """Check if dormant transformers were active earlier in history (indicating seasonal wake-up)."""
        last_dt = pd.Timestamp(cutoff)
        earlier = train_df[(train_df["tarih"] <= last_dt - pd.Timedelta(days=28)) & (train_df["tanim"].isin(dormant_set))]
        act = earlier.groupby("tanim")["tuketim"].max() > 0.0
        return {str(k): bool(v) for k, v in act.items()}

    def apply_wakeup_floor(
        self,
        log_predictions: np.ndarray,
        test_df: pd.DataFrame,
        train_df: pd.DataFrame,
        unseen_mask: np.ndarray,
    ) -> np.ndarray:
        """Apply seasonal wake-up floor for dormant seen meters."""
        if not self.paths.wakeup_floor_path.is_file():
            return log_predictions

        with open(self.paths.wakeup_floor_path, "r") as f:
            floor_data = json.load(f).get("shipped_floor_log", {})

        dormant_set = self.identify_dormant_transformers(train_df, self.config.test_cutoff)
        prior_active = self.identify_prior_active(train_df, dormant_set, self.config.test_cutoff)

        is_dormant = test_df["tanim"].isin(dormant_set).values
        # Never apply floor to cold-start rows (cold-start prior handles those)
        is_dormant_seen = is_dormant & (~unseen_mask.astype(bool))

        months = test_df["tarih"].dt.month.values
        tanims = test_df["tanim"].values

        keys = [f"{m}_{int(bool(prior_active.get(t, False)))}" for m, t in zip(months, tanims)]
        floors = np.array([floor_data.get(k, -np.inf) for k in keys])

        adjusted = log_predictions.copy()
        needs_floor = is_dormant_seen & (floors > adjusted)
        adjusted[needs_floor] = floors[needs_floor]
        return adjusted

    def to_submission_df(self, test_df: pd.DataFrame, log_predictions: np.ndarray) -> pd.DataFrame:
        """Convert log-predictions to final submission dataframe with non-negative kWh."""
        final_kwh = np.expm1(log_predictions).clip(0.0)
        sub = pd.DataFrame({
            "id": test_df["id"].values,
            "tuketim": final_kwh,
        })
        assert len(sub) == len(test_df), "Submission row count mismatch."
        assert sub["tuketim"].notna().all(), "Submission contains null values."
        assert (sub["tuketim"] >= 0.0).all(), "Submission contains negative values."
        return sub
