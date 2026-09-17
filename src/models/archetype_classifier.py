"""Transformer operational archetype multiclass classifier using LightGBM."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd


class ArchetypeClassifier:
    """Multiclass model predicting 6-dimensional operational archetype probability distributions."""

    ARCHETYPE_NAMES = [
        "prob_industrial_6d",
        "prob_agro_irrigation",
        "prob_coastal_tourism",
        "prob_commercial_5d",
        "prob_infra_247",
        "prob_residential",
    ]

    FEATURE_SUBSET = [
        "sunday_to_weekday_ratio", "summer_to_winter_ratio", "load_volatility_cv",
        "tan_mean", "tan_median", "tan_std", "tan_min", "tan_max", "tan_zero",
        "tan_dow_mean", "tan_moy_mean", "tan_lf", "guc", "log_guc",
        "summer_concentration_ratio", "is_summer_exclusive", "recent_load_cv"
    ]

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.model: lgb.Booster | None = None
        self.active_features: list[str] = []

    def fit(self, train_df: pd.DataFrame) -> ArchetypeClassifier:
        """Fit multiclass classifier on inferred transformer archetypes."""
        if "archetype_code" not in train_df.columns:
            raise KeyError("archetype_code must be present in training dataframe.")

        # Classes are 1 to 6 -> convert to 0 to 5 for LightGBM
        target = (train_df["archetype_code"].fillna(6.0).astype(int) - 1).clip(0, 5).values
        self.active_features = [c for c in self.FEATURE_SUBSET if c in train_df.columns]

        X = train_df[self.active_features].copy()
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = X[col].astype("category")

        dtrain = lgb.Dataset(X, label=target, free_raw_data=False)
        params = {
            "objective": "multiclass",
            "num_class": 6,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "verbose": -1,
            "seed": self.seed,
            "n_jobs": 4,
        }

        self.model = lgb.train(params, dtrain, num_boost_round=300)
        return self

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Output 6-column DataFrame containing calibrated archetype probabilities."""
        if self.model is None:
            raise RuntimeError("Model must be fitted before predict_proba.")

        X = df[self.active_features].copy()
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = X[col].astype("category")

        probs = self.model.predict(X)
        return pd.DataFrame(probs, columns=self.ARCHETYPE_NAMES, index=df.index)
