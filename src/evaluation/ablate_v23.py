"""Leave-one-group-out ablation of the v23/v24 feature groups against a v22-shaped baseline.

v22 scored 1.03921 on the public LB; v23 ("more feats") scored 1.05914 -- a 0.0199 regression,
8x the gain v22 itself bought. Those features landed as three extractors in one commit, so this
isolates which group carries the damage by holding EVERYTHING else fixed and varying only which
columns the model is allowed to see.

Blocks are built ONCE and reused across arms: the ablation is column exclusion at fit time, not a
pipeline rebuild. Arms therefore differ only in their feature set -- same rows, same seeds, same
splits -- so the deltas are attributable.

ponytail: single seed per arm. Ranking arms needs far less precision than scoring one, and the
paired structure (identical rows/splits) cancels most seed variance. Re-run the winner multi-seed
before acting on it.
"""
from __future__ import annotations

import json
import time
import numpy as np
import pandas as pd

from src.config import Config
from src.data.loader import DataLoader
from src.data.weather import WeatherLoader
from src.features.pipeline import FeaturePipeline
from src.models.ensemble import EnsembleBlender
from src.evaluation.metrics import rmsle, test_weighted_rmsle

# Column groups introduced by the v23/v24 feature commit, by extractor.
REGIME_COLS = [
    "level_shift_ratio_60d", "recent_vs_lifetime_log_diff", "recent_load_cv",
    "scaled_summer_baseline", "regime_persistence_days",
]
AGRO_COLS = [
    "summer_concentration_ratio", "is_summer_exclusive", "agro_summer_expansion_ratio",
    "is_heavy_agro_ilce", "cum_water_deficit", "agro_district_x_et0", "is_delayed_start",
]
ARCHETYPE_COLS = [
    "sunday_to_weekday_ratio", "summer_to_winter_ratio", "load_volatility_cv",
    "archetype_code", "archetype_x_cdd",
]
ALL_NEW = REGIME_COLS + AGRO_COLS + ARCHETYPE_COLS

ARMS = {
    "full_v24":        [],              # everything on -- current shipped state
    "drop_regime":     REGIME_COLS,
    "drop_agro":       AGRO_COLS,
    "drop_archetype":  ARCHETYPE_COLS,
    "v22_baseline":    ALL_NEW,         # all three groups off -- v22-shaped feature set
}


def run_ablation(seed: int = 42) -> dict:
    t0 = time.time()
    config = Config()
    loader = DataLoader(config.paths)
    pipeline = FeaturePipeline(config)

    print(">>> Loading data and building causal blocks (once, reused by all arms)...")
    train_raw, _ = loader.load_train_test(clean=True)
    weather_df = WeatherLoader(config.paths, config.features).load_weather()
    train_enriched = pipeline.enrich_calendar_and_weather(train_raw, weather_df)
    blocks = [
        pipeline.build_snapshot_block(train_enriched, fc, s, e)
        for fc, s, e in config.models.train_blocks
    ]
    for i in range(1, len(blocks)):
        assert blocks[i - 1]["tarih"].max() < blocks[i]["tarih"].min(), f"Leakage: block {i-1}/{i}"
    print("    Leakage audit: PASSED")

    base_feats = pipeline.get_feature_names(blocks[-1])
    missing = [c for c in ALL_NEW if c not in base_feats]
    if missing:
        print(f"    WARNING: ablation columns absent from pipeline: {missing}")

    fold_names = ["Summer-Autumn 2025", "Winter 2025-2026"]
    results: dict[str, dict] = {}

    for arm, dropped in ARMS.items():
        feats = [f for f in base_feats if f not in set(dropped)]
        print(f"\n>>> ARM {arm}: {len(feats)} features ({len(base_feats) - len(feats)} dropped)")
        per_fold, ys, preds, seens = [], [], [], []

        for idx, val_idx in enumerate(range(2, len(blocks))):
            val_block = blocks[val_idx]
            train_pool = pd.concat(blocks[:val_idx], ignore_index=True)

            blender = EnsembleBlender(feats, config.models)
            blender.fit(train_pool, lgb_seeds=[seed], cat_seeds=[seed])
            log_pred = blender.predict(val_block)

            y = val_block["log_t"].values
            seen = val_block["seen"].values == 1
            score = rmsle(log_pred, y)
            tw = test_weighted_rmsle(log_pred, y, seen, config.models.test_unseen_share)
            unseen_s = rmsle(log_pred[~seen], y[~seen]) if (~seen).sum() else float("nan")
            seen_s = rmsle(log_pred[seen], y[seen]) if seen.sum() else float("nan")

            print(f"    {fold_names[idx]:22s} rmsle {score:.4f} | tw {tw:.4f} | seen {seen_s:.4f} | unseen {unseen_s:.4f}")
            per_fold.append({"fold": fold_names[idx], "rmsle": score, "tw_rmsle": tw,
                             "seen_rmsle": seen_s, "unseen_rmsle": unseen_s})
            ys.append(y); preds.append(log_pred); seens.append(seen)

        y_all = np.concatenate(ys); p_all = np.concatenate(preds); s_all = np.concatenate(seens)
        pooled = rmsle(p_all, y_all)
        pooled_tw = test_weighted_rmsle(p_all, y_all, s_all, config.models.test_unseen_share)
        pooled_unseen = rmsle(p_all[~s_all], y_all[~s_all])
        print(f"    POOLED rmsle {pooled:.4f} | tw {pooled_tw:.4f} | unseen {pooled_unseen:.4f}")
        results[arm] = {"n_features": len(feats), "dropped": dropped, "folds": per_fold,
                        "pooled_rmsle": float(pooled), "pooled_tw_rmsle": float(pooled_tw),
                        "pooled_unseen_rmsle": float(pooled_unseen)}

    base = results["v22_baseline"]["pooled_tw_rmsle"]
    print("\n" + "=" * 72)
    print("ABLATION VERDICT  (delta vs v22-shaped baseline; negative = the group HELPS)")
    print("=" * 72)
    print(f"{'arm':16s} {'pooled_tw':>10s} {'delta':>9s} {'unseen':>9s}")
    for arm, r in results.items():
        print(f"{arm:16s} {r['pooled_tw_rmsle']:10.4f} {r['pooled_tw_rmsle']-base:+9.4f} {r['pooled_unseen_rmsle']:9.4f}")
    print(f"\nElapsed: {time.time()-t0:.1f}s   (single seed={seed}; re-run winner multi-seed)")

    out = {"seed": seed, "baseline_arm": "v22_baseline", "arms": results}
    with open("outputs/ablate_v23.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    run_ablation()
