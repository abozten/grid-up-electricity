"""Small dependency-free regression checks for causal preprocessing and blending."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loader import DataLoader
from src.models.ensemble import EnsembleBlender


def test_declip_uses_only_prior_clean_values() -> None:
    frame = pd.DataFrame(
        {
            "tanim": ["t1", "t1", "t1", "t2"],
            "tarih": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-01"]),
            "guc": [10.0, 10.0, 10.0, 10.0],
            "tuketim": [10.0, 1000.0, 30.0, 1000.0],
        }
    )

    cleaned = DataLoader.declip(frame)

    # The future clean value 30 must not influence the Jan 2 replacement.
    assert cleaned.loc[1, "tuketim"] == 10.0
    # A first-ever corrupted reading has no causal prior.
    assert cleaned.loc[3, "tuketim"] == 0.0
    assert frame.loc[1, "tuketim"] == 1000.0


def test_blend_weights_require_fitted_models() -> None:
    blender = EnsembleBlender([])
    validation = pd.DataFrame({"log_t": [0.0]})

    try:
        blender.optimize_weights(validation)
    except RuntimeError as exc:
        assert "before optimizing" in str(exc)
    else:
        raise AssertionError("Unfitted blend optimization must fail explicitly")


if __name__ == "__main__":
    test_declip_uses_only_prior_clean_values()
    test_blend_weights_require_fitted_models()
    print("regression checks passed")
