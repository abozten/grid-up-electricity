"""Data loading, corruption declipping, and location hierarchy parsing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PathConfig


class DataLoader:
    """Loads and sanitizes raw consumption and location datasets."""

    IMPOSSIBLE_MULTIPLIER: float = 3.0  # Multiples of (guc * 24h) considered physical meter corruption

    def __init__(self, paths: PathConfig | None = None) -> None:
        self.paths = paths or PathConfig()

    @staticmethod
    def parse_loc(df: pd.DataFrame) -> pd.DataFrame:
        """Parse hierarchical location strings into structured columns.

        Handles both 2-level (Manisa: IL > ILCE) and 3-level (Izmir: IL > BOLGE > ILCE) formats.
        Guarantees that `ilce` is always the leaf administrative district.
        """
        df = df.copy()
        parts = df["lokasyon"].str.split(">", expand=True)
        for col in parts.columns:
            parts[col] = parts[col].str.strip()
        num_levels = parts.notna().sum(axis=1)

        df["il"] = parts[0]
        # Leaf part is the last present level
        leaf = parts[1].where(num_levels == 2, parts[2] if 2 in parts.columns else parts[1])
        df["ilce"] = leaf.fillna(parts[1]).fillna(parts[0])
        df["bolge"] = np.where(num_levels >= 3, parts[1], parts[0])
        return df

    STANDARD_AMPLIFICATION_RATES = np.array([
        5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80, 100, 120, 125, 150, 160,
        200, 240, 250, 300, 315, 345, 350, 360, 400, 480, 500, 600, 800, 1000,
        1200, 1500, 2000, 2400, 3000, 4000, 5000, 10000
    ], dtype=float)

    @classmethod
    def declip(cls, train_df: pd.DataFrame, multiplier: float = IMPOSSIBLE_MULTIPLIER) -> pd.DataFrame:
        """Repair measurement artifacts and un-divided amplification rates in the training set.

        Detects jump artifacts where meters recorded values without dividing by the
        instrumentation amplification rate (e.g. Current Transformer CT or VT multipliers).
        Divides such readings by the matched amplification rate, or falls back to the
        causal prior clean median for un-rateable physical ceiling violations.
        """
        if "tarih" not in train_df.columns:
            raise ValueError("Causal declipping requires a 'tarih' column.")

        df = train_df.copy()
        ordered = df.sort_values(["tanim", "tarih"], kind="mergesort")
        
        cap = ordered["guc"] * 24.0
        is_extreme = ordered["tuketim"] > (multiplier * cap)
        
        # Calculate expanding prior clean median per transformer (strictly causal)
        clean_vals = ordered["tuketim"].where(~is_extreme)
        prior_medians = clean_vals.groupby(
            ordered["tanim"], sort=False
        ).transform(lambda values: values.expanding().median().shift()).fillna(0.0)

        # Detect jumps: either physically impossible (> multiplier * cap) OR jump vs prior median
        tuketim_vals = ordered["tuketim"].values.copy()
        prior_med_vals = prior_medians.values
        cap_vals = cap.values

        # Identify jump candidates
        ratio_to_prior = np.where(prior_med_vals > 0, tuketim_vals / np.maximum(prior_med_vals, 1e-3), 0.0)
        jump_mask = (tuketim_vals > (multiplier * cap_vals)) | ((ratio_to_prior > 5.0) & (tuketim_vals > 100.0))

        if jump_mask.any():
            for idx in np.where(jump_mask)[0]:
                v = tuketim_vals[idx]
                p_med = prior_med_vals[idx]
                c_cap = cap_vals[idx]
                
                if p_med > 0:
                    r = v / p_med
                    # Find closest standard amplification rate
                    rel_diffs = np.abs(cls.STANDARD_AMPLIFICATION_RATES - r) / cls.STANDARD_AMPLIFICATION_RATES
                    best_idx = np.argmin(rel_diffs)
                    best_rate = cls.STANDARD_AMPLIFICATION_RATES[best_idx]
                    
                    if rel_diffs[best_idx] < 0.40 and (v / best_rate) <= max(c_cap * 1.5, p_med * 4.0):
                        tuketim_vals[idx] = v / best_rate
                    else:
                        tuketim_vals[idx] = p_med
                else:
                    tuketim_vals[idx] = 0.0

            ordered["tuketim"] = tuketim_vals
            df.loc[ordered.index, "tuketim"] = ordered["tuketim"]

        return df

    def load_train_test(
        self, clean: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load, clean, and parse train and test datasets."""
        train = pd.read_csv(self.paths.train_path, parse_dates=["tarih"])
        test = pd.read_csv(self.paths.test_path, parse_dates=["tarih"])

        if clean:
            train = self.declip(train)

        train = self.parse_loc(train)
        test = self.parse_loc(test)

        train["log_t"] = np.log1p(train["tuketim"])
        train["is_pos"] = (train["tuketim"] > 0).astype(int)

        return train, test
