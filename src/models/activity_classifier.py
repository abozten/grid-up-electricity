"""Two-stage activity and outage probability classifier using LightGBM."""
from __future__ import annotations

from typing import Sequence
import lightgbm as lgb
import numpy as np
import pandas as pd


class ActivityClassifier:
    """Predicts daily probability of active consumption vs outage/dormancy (P(y > 0))."""

    FEATURE_SUBSET = [
        "tan_zero", "tan_r7_zero", "tan_r28_zero", "tan_r91_zero", "tan_zero_trend",
        "guc", "month", "dow", "doy", "is_weekend", "is_bayram", "is_holiday",
        "is_summer_exclusive", "is_delayed_start", "cum_water_deficit",
        "sunday_to_weekday_ratio", "level_shift_ratio_60d", "regime_persistence_days",
        "cdd22", "hdd18", "temperature_2m_mean", "seen", "horizon"
    ]

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.model: lgb.Booster | None = None
        self.active_features: list[str] = []

    def fit(self, train_df: pd.DataFrame) -> ActivityClassifier:
        """Fit binary classifier predicting is_pos = (tuketim > 0)."""
        target = (train_df["tuketim"] > 0.0).astype(int).values
        self.active_features = [c for c in self.FEATURE_SUBSET if c in train_df.columns]

        X = train_df[self.active_features].copy()
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = X[col].astype("category")

        dtrain = lgb.Dataset(X, label=target, free_raw_data=False)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbose": -1,
            "seed": self.seed,
            "n_jobs": 4,
        }

        self.model = lgb.train(params, dtrain, num_boost_round=400)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict continuous probability of active consumption P(y > 0)."""
        if self.model is None:
            raise RuntimeError("Model must be fitted before predict_proba.")

        X = df[self.active_features].copy()
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = X[col].astype("category")

        probs = self.model.predict(X)
        return np.clip(probs, 0.001, 0.999)
