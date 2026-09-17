"""MLflow experiment tracking wrapper adhering to Dependency Inversion."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator


class MLflowTracker:
    """Safely encapsulates MLflow experiment management and artifact logging."""

    def __init__(self, experiment_name: str = "grid-up-electricity", enabled: bool = True) -> None:
        self.experiment_name = experiment_name
        self.enabled = enabled

    @contextmanager
    def start_run(self, run_name: str | None = None) -> Generator[MLflowTracker, None, None]:
        """Context manager for an MLflow run."""
        if not self.enabled:
            yield self
            return

        try:
            import mlflow
            mlflow.set_experiment(self.experiment_name)
            with mlflow.start_run(run_name=run_name):
                yield self
        except ImportError:
            yield self

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        if not self.enabled:
            return
        try:
            import mlflow
            formatted = {
                k: v if isinstance(v, (str, int, float, bool)) else json.dumps(v, default=str)
                for k, v in params.items()
            }
            mlflow.log_params(formatted)
        except Exception:
            pass

    def log_metrics(self, metrics: dict[str, float | int]) -> None:
        """Log numeric metrics to MLflow."""
        if not self.enabled:
            return
        try:
            import mlflow
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        except Exception:
            pass

    def log_artifacts(self, *paths: str | Path) -> None:
        """Log output artifact files to MLflow."""
        if not self.enabled:
            return
        try:
            import mlflow
            for p in paths:
                path_obj = Path(p)
                if path_obj.is_file():
                    mlflow.log_artifact(str(path_obj))
        except Exception:
            pass
