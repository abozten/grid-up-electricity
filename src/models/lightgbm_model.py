"""LightGBM Quantile Regressor forecaster with leak-free memory management."""
from __future__ import annotations

import gc
from typing import Sequence
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.base import BaseForecaster
from src.config import ModelConfig


class LightGBMForecaster(BaseForecaster):
    """LightGBM quantile regression wrapper ensuring deterministic execution and memory release."""

    def __init__(
        self,
        feature_names: Sequence[str],
        config: ModelConfig | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(feature_names, seed)
        self.config = config or ModelConfig()
        self.booster: lgb.Booster | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        num_boost_round: int | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> LightGBMForecaster:
        """Fit LightGBM Booster on provided features and target."""
        rounds = num_boost_round or self.config.lgb_rounds
        params = {
            "objective": "quantile",
            "alpha": self.config.quantile_alpha,
            "metric": "rmse",
            "learning_rate": self.config.learning_rate,
            "num_leaves": self.config.lgb_num_leaves,
            "min_data_in_leaf": self.config.lgb_min_data_in_leaf,
            "feature_fraction": self.config.lgb_feature_fraction,
            "bagging_fraction": self.config.lgb_bagging_fraction,
            "bagging_freq": 1,
            "verbosity": -1,
            "seed": self.seed,
            "bagging_seed": self.seed,
            "feature_fraction_seed": self.seed,
        }

        features = [f for f in self.feature_names if f in X.columns]
        dataset = lgb.Dataset(X[features], label=y, weight=sample_weight, free_raw_data=True)
        self.booster = lgb.train(params, dataset, num_boost_round=rounds)
        
        # Free dataset buffer immediately
        self.booster.free_dataset()
        del dataset
        gc.collect()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict log-space target values clipped at 0."""
        if self.booster is None:
            raise RuntimeError("Model must be fitted before predict() is called.")
        features = [f for f in self.feature_names if f in X.columns]
        preds = self.booster.predict(X[features])
        return np.clip(preds, 0.0, None)
