"""Does the presence group earn its place in the main model? Two arms, everything else fixed.

The presence columns (src/features/presence.py) describe the shape of a transformer's row
calendar inside the window being predicted -- coverage, distance from each window edge,
interior gaps. test.csv states that calendar up front, so it is observable at prediction
time for seen and unseen transformers alike.

The cold-start study established the columns predict the per-transformer LEVEL
(reports/ORACLE_LEVEL.md, R2 0.164 -> 0.437 on a transformer holdout). That is not the
same claim as "they help the shipped model", which already sees each seen transformer's
own history. This script tests that second claim the way this project tests every other
feature group: leave-one-group-out on the causal blocks, reported per fold, and held to
the two-direction rule -- a group that helps summer and hurts winter is noise, not signal,
whatever the pooled number says.

Blocks are built ONCE with presence on; the baseline arm excludes the columns at fit time.
Arms therefore differ only in their feature set -- same rows, same splits, same seed.

Single seed, like ablate_v23: ranking two arms needs less precision than scoring one, and
the paired structure cancels most seed variance. Re-run multi-seed before shipping.

Run: python src/evaluation/ablate_presence.py
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
from src.features.presence import PRESENCE_COLS
from src.models.ensemble import EnsembleBlender
from src.evaluation.metrics import rmsle, test_weighted_rmsle

ARMS = {
    "baseline":     PRESENCE_COLS,   # presence columns withheld -- current shipped shape
    "with_presence": [],             # presence columns visible
}


def run_ablation(seed: int = 42) -> dict:
    t0 = time.time()
    config = Config()
    loader = DataLoader(config.paths)
    pipeline = FeaturePipeline(config, use_presence=True)

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
    missing = [c for c in PRESENCE_COLS if c not in base_feats]
    assert not missing, f"presence columns absent from pipeline: {missing}"

    # What the columns look like on the blocks, before any model sees them.
    for i, blk in enumerate(blocks):
        early = blk.groupby("tanim", observed=True)["pres_ends_early"].first()
        print(f"    block {i} [{blk.tarih.min().date()} -> {blk.tarih.max().date()}]: "
              f"{int(early.sum())}/{len(early)} transformers end early, "
              f"{blk['pres_ends_early'].mean():.2%} of rows")

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
            early = val_block["pres_ends_early"].values == 1
            early_s = rmsle(log_pred[early], y[early]) if early.sum() else float("nan")

            print(f"    {fold_names[idx]:22s} rmsle {score:.4f} | tw {tw:.4f} | seen {seen_s:.4f} "
                  f"| unseen {unseen_s:.4f} | ends_early {early_s:.4f} ({early.mean():.2%} rows)")
            per_fold.append({"fold": fold_names[idx], "rmsle": score, "tw_rmsle": tw,
                             "seen_rmsle": seen_s, "unseen_rmsle": unseen_s,
                             "ends_early_rmsle": early_s, "ends_early_row_share": float(early.mean())})
            ys.append(y); preds.append(log_pred); seens.append(seen)

        y_all = np.concatenate(ys); p_all = np.concatenate(preds); s_all = np.concatenate(seens)
        pooled = rmsle(p_all, y_all)
        pooled_tw = test_weighted_rmsle(p_all, y_all, s_all, config.models.test_unseen_share)
        pooled_unseen = rmsle(p_all[~s_all], y_all[~s_all])
        print(f"    POOLED rmsle {pooled:.4f} | tw {pooled_tw:.4f} | unseen {pooled_unseen:.4f}")
        results[arm] = {"n_features": len(feats), "dropped": dropped, "folds": per_fold,
                        "pooled_rmsle": float(pooled), "pooled_tw_rmsle": float(pooled_tw),
                        "pooled_unseen_rmsle": float(pooled_unseen)}

    base = results["baseline"]
    arm = results["with_presence"]
    print("\n" + "=" * 78)
    print("VERDICT  (negative delta = presence HELPS)")
    print("=" * 78)
    print(f"{'metric':24s} {'baseline':>10s} {'presence':>10s} {'delta':>9s}")
    print(f"{'pooled tw':24s} {base['pooled_tw_rmsle']:10.4f} {arm['pooled_tw_rmsle']:10.4f} "
          f"{arm['pooled_tw_rmsle'] - base['pooled_tw_rmsle']:+9.4f}")
    print(f"{'pooled unseen':24s} {base['pooled_unseen_rmsle']:10.4f} {arm['pooled_unseen_rmsle']:10.4f} "
          f"{arm['pooled_unseen_rmsle'] - base['pooled_unseen_rmsle']:+9.4f}")
    directions = []
    for i, name in enumerate(fold_names):
        d = arm["folds"][i]["tw_rmsle"] - base["folds"][i]["tw_rmsle"]
        de = arm["folds"][i]["ends_early_rmsle"] - base["folds"][i]["ends_early_rmsle"]
        directions.append(d)
        print(f"{name:24s} {base['folds'][i]['tw_rmsle']:10.4f} {arm['folds'][i]['tw_rmsle']:10.4f} "
              f"{d:+9.4f}   (ends_early rows {de:+.4f})")
    two_way = all(d < 0 for d in directions)
    print(f"\nTWO-DIRECTION TEST: {'PASS -- helps both folds' if two_way else 'FAIL -- reverses on a fold'}")
    print(f"Elapsed: {time.time() - t0:.1f}s   (single seed={seed})")

    out = {"seed": seed, "two_direction_pass": bool(two_way), "arms": results}
    with open(f"outputs/ablate_presence_s{seed}.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    import sys
    run_ablation(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 42)
