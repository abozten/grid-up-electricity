"""Ensemble blender and multi-seed log-space weight optimizer."""
from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.models.lightgbm_model import LightGBMForecaster
from src.models.catboost_model import CatBoostForecaster
from src.config import ModelConfig
from src.evaluation.metrics import rmsle


class EnsembleBlender:
    """Manages multi-seed training and log-space Nelder-Mead blending for GBDT ensembles."""

    def __init__(
        self,
        feature_names: Sequence[str],
        config: ModelConfig | None = None,
    ) -> None:
        self.feature_names = list(feature_names)
        self.config = config or ModelConfig()
        self.lgb_models: list[LightGBMForecaster] = []
        self.cat_models: list[CatBoostForecaster] = []
        self.weights: np.ndarray = np.array([0.468, 0.532])

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        lgb_seeds: Sequence[int] | None = None,
        cat_seeds: Sequence[int] | None = None,
    ) -> EnsembleBlender:
        """Train multi-seed LightGBM and CatBoost models and optimize blend weights if validation set is given."""
        lgb_s = lgb_seeds or self.config.lgb_seeds
        cat_s = cat_seeds or self.config.cat_seeds

        self.lgb_models = []
        for sd in lgb_s:
            m = LightGBMForecaster(self.feature_names, self.config, seed=sd)
            m.fit(train_df, train_df["log_t"].values)
            self.lgb_models.append(m)

        self.cat_models = []
        for sd in cat_s:
            m = CatBoostForecaster(self.feature_names, self.config, seed=sd)
            m.fit(train_df, train_df["log_t"].values)
            self.cat_models.append(m)

        if val_df is not None:
            self.optimize_weights(val_df)

        return self

    def optimize_weights(self, val_df: pd.DataFrame, x0: tuple[float, float] = (0.44, 0.56)) -> np.ndarray:
        """Optimize blend weights on validation set using Nelder-Mead in log-space."""
        if not self.lgb_models or not self.cat_models:
            raise RuntimeError("Fit LightGBM and CatBoost models before optimizing weights.")
        if val_df.empty:
            raise ValueError("Cannot optimize blend weights on an empty validation frame.")

        y_val = val_df["log_t"].values
        lgb_preds = np.mean([m.predict(val_df) for m in self.lgb_models], axis=0)
        cat_preds = np.mean([m.predict(val_df) for m in self.cat_models], axis=0)

        M = np.vstack([lgb_preds, cat_preds])

        def loss_fn(w: np.ndarray) -> float:
            norm_w = np.abs(w) / np.abs(w).sum()
            blend = norm_w @ M
            return rmsle(blend, y_val)

        res = minimize(loss_fn, x0=list(x0), method="Nelder-Mead", options={"maxiter": 1500})
        if not res.success or not np.isfinite(res.fun):
            raise RuntimeError(f"Blend-weight optimization failed: {res.message}")
        w = np.abs(res.x)
        self.weights = w / w.sum()
        return self.weights

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate blended log-space prediction across all fitted seeds."""
        lgb_preds = np.mean([m.predict(X) for m in self.lgb_models], axis=0)
        cat_preds = np.mean([m.predict(X) for m in self.cat_models], axis=0)
        M = np.vstack([lgb_preds, cat_preds])
        return self.weights @ M
