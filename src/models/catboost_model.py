"""CatBoost Quantile Regressor forecaster with categorical encoding."""
from __future__ import annotations

import gc
import os
from typing import Sequence
import catboost as cb
import numpy as np
import pandas as pd

from src.models.base import BaseForecaster
from src.config import ModelConfig


def _gpu_enabled() -> bool:
    """GPU is opt-in via GRIDUP_GPU=1 so CPU stays the reproducible default."""
    return os.environ.get("GRIDUP_GPU", "0") == "1"


class CatBoostForecaster(BaseForecaster):
    """CatBoost quantile regression wrapper handling categorical location hierarchies."""

    CATEGORICAL_COLS = ["il", "bolge", "ilce"]

    def __init__(
        self,
        feature_names: Sequence[str],
        config: ModelConfig | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(feature_names, seed)
        self.config = config or ModelConfig()
        self.model: cb.CatBoostRegressor | None = None

    def _prepare_data(self, X: pd.DataFrame) -> pd.DataFrame:
        features = [f for f in self.feature_names if f in X.columns]
        d = X[features].copy()
        for c in self.CATEGORICAL_COLS:
            if c in d.columns:
                d[c] = d[c].astype(object).fillna("NA").astype(str)
        return d

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        iterations: int | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> CatBoostForecaster:
        """Fit CatBoostRegressor on formatted feature matrix."""
        iters = iterations or self.config.cat_iters
        features = [f for f in self.feature_names if f in X.columns]
        cat_feats = [c for c in self.CATEGORICAL_COLS if c in features]

        # GPU changes the histogram border selection, so scores shift slightly versus
        # CPU. Keep it opt-in and never mix the two inside one comparison.
        device_kwargs = {"task_type": "GPU", "devices": "0"} if _gpu_enabled() else {}

        self.model = cb.CatBoostRegressor(
            iterations=iters,
            learning_rate=self.config.learning_rate,
            depth=self.config.cat_depth,
            l2_leaf_reg=self.config.cat_l2_leaf_reg,
            loss_function=f"Quantile:alpha={self.config.quantile_alpha}",
            random_seed=self.seed,
            verbose=0,
            cat_features=cat_feats,
            **device_kwargs,
        )

        d = self._prepare_data(X)
        self.model.fit(d, y, sample_weight=sample_weight)
        del d
        gc.collect()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict log-space target values clipped at 0."""
        if self.model is None:
            raise RuntimeError("Model must be fitted before predict() is called.")
        d = self._prepare_data(X)
        preds = self.model.predict(d)
        del d
        gc.collect()
        return np.clip(preds, 0.0, None)
