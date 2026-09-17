"""Backward feature reduction on the winter fold.

Two independent lines of evidence point at feature reduction rather than addition:

  1. This repo's own v23/v25 ablation -- all 17 columns added by the v23/v24 commit were
     net harmful, and v23 regressed the LB by +0.0199.
  2. The ASHRAE GEPIII winners (same RMSLE metric, same meter-reading problem shape) took
     1st with 28 heavily curated features; the cross-solution analysis names manual
     pre-processing and curation, not feature count, as the differentiator.

The current pipeline carries ~125 columns. This ranks them by LightGBM gain on the
training pool and scores arms that drop the weakest tail, holding everything else fixed.

Follows the ablate_v23.py protocol: blocks are built ONCE and reused, arms differ only in
which columns the models may see, so deltas are attributable. Scored on the winter fold,
which is the authority under this repo's fold-trust rule -- it trains on 718k rows against
summer's 123k, and test trains deeper still (see reports/V20.md).

Single seed per arm: ranking arms needs far less precision than scoring one, and the paired
structure cancels most seed variance. Re-run the winner multi-seed before acting on it.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src.config import Config
from src.data.loader import DataLoader
from src.data.weather import WeatherLoader
from src.evaluation.metrics import rmsle, test_weighted_rmsle
from src.features.pipeline import FeaturePipeline
from src.models.ensemble import EnsembleBlender

DROP_COUNTS = [0, 10, 20, 30, 45, 60, 80]
OUT_PATH = "outputs/reduce_features.json"


def rank_features(train_pool: pd.DataFrame, feats: list[str], seed: int = 42) -> list[str]:
    """Rank features weakest-first by LightGBM gain on the training pool."""
    import lightgbm as lgb

    X = train_pool[feats].copy()
    for c in X.select_dtypes(include=["category", "object"]).columns:
        X[c] = X[c].astype("category")
    booster = lgb.train(
        {"objective": "quantile", "alpha": 0.55, "learning_rate": 0.05, "num_leaves": 95,
         "min_data_in_leaf": 80, "feature_fraction": 0.8, "bagging_fraction": 0.8,
         "bagging_freq": 1, "verbosity": -1, "seed": seed},
        lgb.Dataset(X, label=train_pool["log_t"].values), num_boost_round=400,
    )
    gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
    gain = gain.reindex(feats).fillna(0.0)
    return list(gain.sort_values().index)  # weakest first


def main(seed: int = 42) -> dict:
    t0 = time.time()
    cfg = Config()
    loader = DataLoader(cfg.paths)
    pipe = FeaturePipeline(cfg)

    print(">>> Building causal blocks once (reused by every arm)...", flush=True)
    train_raw, _ = loader.load_train_test(clean=True)
    weather = WeatherLoader(cfg.paths, cfg.features).load_weather()
    enriched = pipe.enrich_calendar_and_weather(train_raw, weather)
    blocks = [pipe.build_snapshot_block(enriched, fc, s, e) for fc, s, e in cfg.models.train_blocks]
    for i in range(1, len(blocks)):
        assert blocks[i - 1]["tarih"].max() < blocks[i]["tarih"].min(), f"Leakage: block {i-1}/{i}"
    print("    Leakage audit: PASSED", flush=True)

    # Winter fold: train on blocks 0-2, score block 3. The deepest fold available.
    train_pool = pd.concat(blocks[:3], ignore_index=True)
    val_block = blocks[3]
    feats = pipe.get_feature_names(val_block)
    print(f"    train {len(train_pool):,} rows | val {len(val_block):,} rows | {len(feats)} features",
          flush=True)

    print(">>> Ranking features by gain...", flush=True)
    ordered = rank_features(train_pool, feats, seed)
    print(f"    weakest 15: {ordered[:15]}", flush=True)

    y_val = val_block["log_t"].values
    seen = val_block["seen"].values == 1
    results: dict[str, dict] = {}

    for k in DROP_COUNTS:
        keep = [f for f in feats if f not in set(ordered[:k])]
        blender = EnsembleBlender(keep, cfg.models)
        blender.fit(train_pool, lgb_seeds=(seed,), cat_seeds=(seed,))
        pred = blender.predict(val_block)
        arm = {
            "n_features": len(keep),
            "dropped": k,
            "blend_rmsle": rmsle(pred, y_val),
            "test_weighted": test_weighted_rmsle(pred, y_val, seen, cfg.models.test_unseen_share),
            "seen_rmsle": rmsle(pred[seen], y_val[seen]),
            "unseen_rmsle": rmsle(pred[~seen], y_val[~seen]) if (~seen).any() else None,
        }
        results[f"drop_{k}"] = arm
        print(f"    drop {k:3d} -> {len(keep):3d} feats | blend {arm['blend_rmsle']:.4f} "
              f"| tw {arm['test_weighted']:.4f} | seen {arm['seen_rmsle']:.4f}", flush=True)
        del blender

    base = results["drop_0"]["test_weighted"]
    for name, arm in results.items():
        arm["delta_vs_full"] = arm["test_weighted"] - base

    payload = {
        "seed": seed,
        "fold": "winter 2025-12 -> 2026-04 (blocks 0-2 train, block 3 val)",
        "runtime_min": round((time.time() - t0) / 60, 1),
        "feature_order_weakest_first": ordered,
        "arms": results,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f">>> wrote {OUT_PATH} in {payload['runtime_min']} min", flush=True)
    return payload


if __name__ == "__main__":
    main()
