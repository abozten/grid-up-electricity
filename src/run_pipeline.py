"""Unified CLI entrypoint for causal validation and submission generation."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import Config
from src.data.loader import DataLoader
from src.data.weather import WeatherLoader
from src.features.pipeline import FeaturePipeline
from src.models.ensemble import EnsembleBlender
from src.models.cold_start import ColdStartPrior
from src.models.postprocess import PostProcessor
from src.evaluation.metrics import rmsle, test_weighted_rmsle
from src.tracking.tracker import MLflowTracker


def run_cross_validation(config: Config, tracker: MLflowTracker) -> dict[str, float]:
    """Execute causal walk-forward cross validation on historical blocks."""
    print(">>> Loading datasets and weather...")
    loader = DataLoader(config.paths)
    wx_loader = WeatherLoader(config.paths, config.features)
    pipeline = FeaturePipeline(config)

    train_raw, _ = loader.load_train_test(clean=True)
    weather_df = wx_loader.load_weather()
    train_enriched = pipeline.enrich_calendar_and_weather(train_raw, weather_df)

    print(">>> Building causal snapshot blocks...")
    blocks = []
    for feat_cut, start, end in config.models.train_blocks:
        blk = pipeline.build_snapshot_block(train_enriched, feat_cut, start, end)
        blocks.append(blk)
        print(f"    Block [{start} -> {end}]: {len(blk):,} rows")

    feature_names = pipeline.get_feature_names(blocks[-1])
    print(f">>> Features active: {len(feature_names)}")

    # Each fold uses an earlier block for weight selection and a later block
    # for scoring. The scoring block is never used to fit models or weights.
    fold_results = []
    pooled_targets = []
    pooled_seen = []
    pooled_lgb_preds = []
    pooled_cat_preds = []
    pooled_blend_preds = []

    for val_idx in range(2, len(blocks)):
        weight_block = blocks[val_idx - 1]
        weight_train_pool = pd.concat(blocks[: val_idx - 1], ignore_index=True)
        train_pool = pd.concat(blocks[:val_idx], ignore_index=True)
        val_block = blocks[val_idx]

        print(f">>> Fitting causal weight-selection ensemble for fold {val_idx}...")
        weight_blender = EnsembleBlender(feature_names, config.models)
        weight_blender.fit(
            weight_train_pool,
            lgb_seeds=config.models.eval_seeds,
            cat_seeds=config.models.eval_seeds,
        )
        selected_weights = weight_blender.optimize_weights(weight_block)
        del weight_blender, weight_train_pool
        gc.collect()

        print(f">>> Fitting scoring ensemble for fold {val_idx}...")
        blender = EnsembleBlender(feature_names, config.models)
        blender.fit(train_pool, lgb_seeds=config.models.eval_seeds, cat_seeds=config.models.eval_seeds)
        blender.weights = selected_weights

        y_val = val_block["log_t"].values
        seen_val = val_block["seen"].values == 1
        lgb_preds = np.mean([m.predict(val_block) for m in blender.lgb_models], axis=0)
        cat_preds = np.mean([m.predict(val_block) for m in blender.cat_models], axis=0)
        blend_preds = blender.predict(val_block)

        fold = {
            "fold": float(val_idx),
            "lgb_rmsle": rmsle(lgb_preds, y_val),
            "cat_rmsle": rmsle(cat_preds, y_val),
            "blend_rmsle": rmsle(blend_preds, y_val),
            "test_weighted_rmsle": test_weighted_rmsle(
                blend_preds, y_val, seen_val, config.models.test_unseen_share
            ),
            "lgb_weight": float(blender.weights[0]),
            "cat_weight": float(blender.weights[1]),
        }
        fold_results.append(fold)
        pooled_targets.append(y_val)
        pooled_seen.append(seen_val)
        pooled_lgb_preds.append(lgb_preds)
        pooled_cat_preds.append(cat_preds)
        pooled_blend_preds.append(blend_preds)
        print(
            f"    Fold {val_idx}: blend={fold['blend_rmsle']:.4f}, "
            f"test-weighted={fold['test_weighted_rmsle']:.4f}"
        )

        del blender, train_pool, val_block
        gc.collect()

    y_val = np.concatenate(pooled_targets)
    seen_val = np.concatenate(pooled_seen)
    lgb_preds = np.concatenate(pooled_lgb_preds)
    cat_preds = np.concatenate(pooled_cat_preds)
    blend_preds = np.concatenate(pooled_blend_preds)

    metrics = {
        "lgb_rmsle": rmsle(lgb_preds, y_val),
        "cat_rmsle": rmsle(cat_preds, y_val),
        "blend_rmsle": rmsle(blend_preds, y_val),
        "test_weighted_rmsle": test_weighted_rmsle(blend_preds, y_val, seen_val, config.models.test_unseen_share),
        "lgb_weight": float(np.mean([fold["lgb_weight"] for fold in fold_results])),
        "cat_weight": float(np.mean([fold["cat_weight"] for fold in fold_results])),
    }

    print("\n" + "=" * 50)
    print("CAUSAL WALK-FORWARD OOF RESULTS")
    print("=" * 50)
    print(f"  LightGBM RMSLE:       {metrics['lgb_rmsle']:.4f}")
    print(f"  CatBoost RMSLE:       {metrics['cat_rmsle']:.4f}")
    print(f"  Ensemble Blend RMSLE: {metrics['blend_rmsle']:.4f}")
    print(f"  Test-Weighted RMSLE:  {metrics['test_weighted_rmsle']:.4f}")
    print(f"  Weights (LGB / CAT):  {metrics['lgb_weight']:.3f} / {metrics['cat_weight']:.3f}")
    print("=" * 50 + "\n")

    tracker.log_metrics(metrics)
    return metrics


from src.models.activity_classifier import ActivityClassifier


def run_submission(config: Config, tracker: MLflowTracker, output_file: Path | None = None) -> Path:
    """Train on all causal snapshot blocks, predict test, apply classifiers, cold-start & wake-up floor, and export."""
    out_path = output_file or config.paths.submission_path
    print(">>> Loading datasets and weather for submission generation...")
    loader = DataLoader(config.paths)
    wx_loader = WeatherLoader(config.paths, config.features)
    pipeline = FeaturePipeline(config)

    train_raw, test_raw = loader.load_train_test(clean=True)
    weather_df = wx_loader.load_weather()

    train_enriched = pipeline.enrich_calendar_and_weather(train_raw, weather_df)
    test_enriched = pipeline.enrich_calendar_and_weather(test_raw, weather_df)

    print(">>> Building all causal snapshot blocks...")
    blocks = [
        pipeline.build_snapshot_block(train_enriched, feat_cut, start, end)
        for feat_cut, start, end in config.models.train_blocks
    ]
    full_train_pool = pd.concat(blocks, ignore_index=True)

    print(">>> Building test feature block...")
    test_block = pipeline.build_test_block(train_enriched, test_enriched, config.models.test_cutoff)

    # 1. Fit Activity / Outage / Dormancy Binary Classifier
    print(">>> Training Stage 1 Activity/Dormancy Classifier (P(Active > 0))...")
    act_clf = ActivityClassifier(seed=42)
    act_clf.fit(full_train_pool)
    p_active_test = act_clf.predict_proba(test_block)
    print(f"    Mean predicted test activity probability: {p_active_test.mean():.2%}")

    feature_names = pipeline.get_feature_names(test_block)
    print(f"    Train rows: {len(full_train_pool):,} | Test rows: {len(test_block):,} | Features: {len(feature_names)}")

    print(">>> Selecting blend weights from a prior causal block...")
    weight_blender = EnsembleBlender(feature_names, config.models)
    weight_train_pool = pd.concat(blocks[:-2], ignore_index=True)
    # (the archetype stage was removed in the v25 revert; its columns are no longer
    # features, so nothing needs scoring onto the weight blocks here)
    weight_blender.fit(
        weight_train_pool,
        lgb_seeds=config.models.eval_seeds,
        cat_seeds=config.models.eval_seeds,
    )
    selected_weights = weight_blender.optimize_weights(blocks[-2])
    del weight_blender, weight_train_pool
    gc.collect()

    print(">>> Training full multi-seed regression ensemble...")
    blender = EnsembleBlender(feature_names, config.models)
    blender.fit(full_train_pool)
    blender.weights = selected_weights
    print(f"    Blend weights (LGB / CAT): {blender.weights[0]:.3f} / {blender.weights[1]:.3f}")

    print(">>> Generating raw ensemble predictions...")
    log_raw_preds = blender.predict(test_block)

    # 3. Two-Stage Hurdle Fusion: modulate seen predictions by activity probability
    seen_mask = test_block["seen"].values == 1
    raw_kwh = np.expm1(log_raw_preds)
    # Apply soft probabilistic dampening on low-activity meters
    hurdle_kwh = raw_kwh.copy()
    hurdle_kwh[seen_mask] = raw_kwh[seen_mask] * np.where(p_active_test[seen_mask] < 0.90, p_active_test[seen_mask]**0.5, 1.0)
    log_hurdle_preds = np.log1p(hurdle_kwh)

    print(">>> Fitting and applying Winsorized + Isotonic Cold-Start Prior (alpha=0.8)...")
    cold_prior = ColdStartPrior(config.models)
    cold_prior.fit(train_enriched)
    log_cold_preds = cold_prior.adjust(log_hurdle_preds, test_block, seen_mask)
    unseen_count = int((~seen_mask).sum())
    print(f"    Cold-start adjusted rows: {unseen_count:,} ({unseen_count / len(test_block):.2%})")

    print(">>> Applying Seasonal Wake-Up floor for dormant transformers...")
    postprocessor = PostProcessor(config.paths, config.models)
    log_final_preds = postprocessor.apply_wakeup_floor(log_cold_preds, test_block, train_enriched, ~seen_mask)

    print(f">>> Writing final submission CSV to {out_path}...")
    sub_df = postprocessor.to_submission_df(test_raw, log_final_preds)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(out_path, index=False)

    stats = sub_df["tuketim"].describe().to_dict()
    print("    Prediction statistics (kWh):")
    print(f"      Mean:   {stats.get('mean', 0.0):.2f}")
    print(f"      50%:    {stats.get('50%', 0.0):.2f}")
    print(f"      Max:    {stats.get('max', 0.0):.2f}")
    print(f"      Zeros:  {(sub_df['tuketim'] == 0).sum():,}")

    tracker.log_metrics({
        "pred_mean_kwh": float(stats.get("mean", 0.0)),
        "pred_median_kwh": float(stats.get("50%", 0.0)),
        "pred_max_kwh": float(stats.get("max", 0.0)),
        "unseen_rows": float(unseen_count),
    })
    tracker.log_artifacts(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid Up Electricity Demand Forecasting Pipeline")
    parser.add_argument("--mode", choices=["cv", "submit"], default="cv", help="Execution mode: cv or submit")
    parser.add_argument("--experiment-name", default="grid-up-electricity", help="MLflow experiment name")
    parser.add_argument("--run-name", default=None, help="MLflow run name")
    parser.add_argument("--output", type=Path, default=None, help="Output submission path")
    args = parser.parse_args()

    config = Config(experiment_name=args.experiment_name)
    tracker = MLflowTracker(experiment_name=args.experiment_name, enabled=True)

    with tracker.start_run(run_name=args.run_name or f"run_{args.mode}"):
        tracker.log_params({
            "mode": args.mode,
            "quantile_alpha": config.models.quantile_alpha,
            "lgb_rounds": config.models.lgb_rounds,
            "cat_iters": config.models.cat_iters,
            "cold_start_alpha": config.models.cold_start_alpha,
            "cold_start_k": config.models.cold_start_k,
        })

        if args.mode == "cv":
            run_cross_validation(config, tracker)
        elif args.mode == "submit":
            run_submission(config, tracker, args.output)


if __name__ == "__main__":
    main()
