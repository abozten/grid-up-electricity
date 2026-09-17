"""Base forecaster abstraction conforming to Liskov Substitution Principle."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """Abstract base class for all single gradient boosted trees and estimators."""

    def __init__(self, feature_names: Sequence[str], seed: int = 42) -> None:
        self.feature_names = list(feature_names)
        self.seed = seed

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series, **kwargs) -> BaseForecaster:
        """Fit model on feature matrix X and target y in log-space."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict log-space target on feature matrix X."""
        raise NotImplementedError
