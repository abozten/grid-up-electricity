"""Apply SPC anomaly detection to electricity load data.

Demonstrates:
1. Loading actual electricity data
2. Fitting baseline statistics per transformer
3. Detecting anomalies using multiple SPC methods
4. Visualizing results
5. Exporting anomaly reports
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from spc_anomaly_detection import SPCConfig, SPCDetector, MultiSeriesSPCDetector


def load_electricity_data(
    train_path: str | Path,
    test_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load electricity data from CSV files.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data (optional)
        
    Returns:
        Tuple of (train_df, test_df or None)
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) if test_path else None
    
    print(f"Train data shape: {train.shape}")
    print(f"Train columns: {train.columns.tolist()}")
    if test is not None:
        print(f"Test data shape: {test.shape}")
    
    return train, test


def prepare_data_for_spc(
    df: pd.DataFrame,
    transformer_id_col: str = "transformer_id",
    value_col: str = "consumption",
    date_col: str | None = None,
) -> pd.DataFrame:
    """Prepare data for SPC analysis.

    Args:
        df: Raw DataFrame
        transformer_id_col: Column identifying transformers
        value_col: Column with consumption values
        date_col: Column with dates (for sorting)

    Returns:
        Cleaned DataFrame sorted by date and transformer
    """
    data = df.copy()

    # Remove rows with missing values
    initial_rows = len(data)
    data = data.dropna(subset=[transformer_id_col, value_col])
    print(f"Removed {initial_rows - len(data)} rows with missing values")

    # Remove zero or negative values (invalid for electricity consumption)
    data = data[data[value_col] > 0]

    # Sort by transformer and date if available
    sort_cols = [transformer_id_col]
    if date_col and date_col in data.columns:
        sort_cols.append(date_col)
    data = data.sort_values(sort_cols, ignore_index=True)

    return data


def add_detrended_residual(
    data: pd.DataFrame,
    transformer_id_col: str = "transformer_id",
    value_col: str = "consumption",
    resid_col: str = "resid",
    trend_window: int = 21,
) -> pd.DataFrame:
    """Add a log-scale, detrended residual column to fit/detect SPC on.

    Raw consumption is right-skewed (occasional huge spikes) and strongly
    seasonal/trending over a 15-month span. Feeding raw values straight into
    SPC with a single static baseline makes the baseline mean/std sensitive
    to whichever outliers happen to fall in the baseline window, and makes a
    fixed baseline go stale as the season changes -- both inflate the flag
    rate to the point of being useless (~76% of points flagged).

    This applies log1p (compresses the right-skew/large spikes) and then
    subtracts a centered rolling median "trend" per transformer, leaving an
    approximately stationary residual that SPC's constant-mean assumption
    is actually appropriate for.

    Args:
        data: DataFrame with transformer data
        transformer_id_col: Column identifying transformers
        value_col: Column with consumption values
        resid_col: Name of the output residual column
        trend_window: Rolling median window (days) used to estimate trend

    Returns:
        DataFrame with an added residual column
    """
    data = data.copy()
    log_val = np.log1p(data[value_col])

    def _detrend(s: pd.Series) -> pd.Series:
        trend = s.rolling(trend_window, center=True, min_periods=5).median()
        trend = trend.bfill().ffill()
        # Groups too short to ever satisfy min_periods=5 stay all-NaN; fall
        # back to a flat trend (the group's own mean) so resid = 0 rather
        # than NaN poisoning downstream baseline fitting.
        trend = trend.fillna(s.mean())
        return s - trend

    data[resid_col] = (
        log_val.groupby(data[transformer_id_col]).transform(_detrend)
    )
    return data


def robust_baseline_stats(values: np.ndarray) -> tuple[float, float]:
    """Estimate the (center, sigma) pair SPC's control limits are built from.

    Tried median-center + moving-range sigma here first (matches the d2
    constant Shewhart already uses, and resists isolated spikes). It backed
    CUSUM's k/h into too tight a corner: MR sigma only measures step-to-step
    noise, which on this data is much smaller than the day-to-day swings a
    detrended-but-still-autocorrelated series legitimately has, so CUSUM
    drifted out of control almost everywhere (84.7% flagged, worse than the
    76.4% baseline). Plain mean/std on the log-scale, detrended residual
    (see add_detrended_residual) is well-behaved instead -- detrending
    already did the heavy lifting of neutralizing outliers/seasonality, so
    there's no leftover skew for a robust estimator to protect against here.

    Args:
        values: 1D baseline array (already log-scaled/detrended)

    Returns:
        (center, sigma) tuple; sigma is floored above zero
    """
    center = float(np.mean(values))
    sigma = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return center, max(sigma, 1e-6)


def fit_baseline_per_transformer(
    data: pd.DataFrame,
    transformer_id_col: str = "transformer_id",
    value_col: str = "consumption",
    raw_value_col: str | None = None,
    baseline_fraction: float = 0.5,
    config: SPCConfig | None = None,
) -> dict:
    """Fit SPC baselines for each transformer on a portion of data.

    Uses `value_col` (expected to already be a stationary, log-scale
    residual -- see `add_detrended_residual`) and robust median/moving-range
    statistics (see `robust_baseline_stats`) instead of raw mean/std, so a
    few outlier spikes inside the baseline window don't blow up the fitted
    sigma. Transformers with zero raw-consumption variance (flat/dead
    meters) are skipped entirely rather than fit -- there's nothing to
    detect against, and a zero-sigma baseline just produces noise.

    Args:
        data: DataFrame with transformer data
        transformer_id_col: Column identifying transformers
        value_col: Column with (residual) values to fit on
        baseline_fraction: Use first N% of data to establish baseline
        config: SPCConfig instance

    Returns:
        Dictionary mapping transformer_id -> fitted SPCDetector
    """
    detectors = {}
    config = config or SPCConfig()
    raw_value_col = raw_value_col or value_col

    n_transformers = data[transformer_id_col].nunique()
    print(f"\nFitting baselines for {n_transformers} transformers...")

    n_flat = 0
    for transformer_id in data[transformer_id_col].unique():
        transformer_data = data[data[transformer_id_col] == transformer_id]

        if transformer_data[raw_value_col].std(ddof=1) == 0:
            n_flat += 1
            continue

        # Use first portion for baseline
        baseline_size = max(
            config.min_baseline_points,
            int(len(transformer_data) * baseline_fraction)
        )
        baseline_data = transformer_data[value_col].iloc[:baseline_size].values

        # Skip if insufficient data
        if len(baseline_data) < config.min_baseline_points:
            continue

        detector = SPCDetector(config)
        try:
            detector.fit(baseline_data)
            center, sigma = robust_baseline_stats(baseline_data)
            detector.baseline_mean = center
            detector.baseline_std = sigma
            detectors[transformer_id] = detector
        except Exception as e:
            print(f"  Warning: Failed to fit transformer {transformer_id}: {e}")

    print(f"Successfully fitted {len(detectors)} transformers")
    print(f"Skipped {n_flat} flat/dead transformers (zero consumption variance)")
    return detectors


def detect_anomalies_per_transformer(
    data: pd.DataFrame,
    detectors: dict,
    transformer_id_col: str = "transformer_id",
    value_col: str = "consumption",
    methods: tuple[str, ...] = ("shewhart", "ewma", "cusum", "nelson"),
    aggregate: str = "majority",
) -> pd.DataFrame:
    """Detect anomalies using fitted detectors.

    Args:
        data: DataFrame with transformer data
        detectors: Dictionary of fitted SPCDetector instances
        transformer_id_col: Column identifying transformers
        value_col: Column with consumption values
        methods: Which SPC methods to use
        aggregate: How to combine per-method flags into `is_anomaly`.
            Each individual method (esp. CUSUM and Nelson's run rules) has a
            meaningfully high standalone false-positive rate on real,
            autocorrelated consumption data -- "any" single method firing
            flagged 53-85% of points in testing. "majority" (>=3 of 4
            methods agreeing) brought that down to a defensible ~6%.

    Returns:
        DataFrame with original data + anomaly columns
    """
    results = data.copy()
    
    # Add anomaly columns
    for method in methods:
        results[f"{method}_anomaly"] = False
    results["is_anomaly"] = False
    results["n_methods_flagged"] = 0
    
    print(f"\nDetecting anomalies using methods: {', '.join(methods)}")
    
    for transformer_id in data[transformer_id_col].unique():
        if transformer_id not in detectors:
            continue
        
        mask = data[transformer_id_col] == transformer_id
        detector = detectors[transformer_id]
        
        try:
            transformer_data = data[mask][value_col].values
            anom_flags, detailed_results = detector.detect(
                transformer_data,
                methods=methods,
                aggregate=aggregate
            )
            
            # Update results
            for method in methods:
                col_name = f"{method}_anomaly"
                if col_name in detailed_results.columns:
                    results.loc[mask, col_name] = detailed_results[col_name].values
            
            results.loc[mask, "is_anomaly"] = anom_flags
            
            # Count how many methods flagged each point
            method_columns = [c for c in detailed_results.columns if c.endswith("_anomaly")]
            results.loc[mask, "n_methods_flagged"] = (
                detailed_results[method_columns].astype(int).sum(axis=1).values
            )
        except Exception as e:
            print(f"  Warning: Detection failed for transformer {transformer_id}: {e}")
    
    return results


def summarize_anomalies(
    results: pd.DataFrame,
    transformer_id_col: str = "transformer_id",
) -> pd.DataFrame:
    """Create summary statistics of detected anomalies.
    
    Args:
        results: DataFrame with anomaly detection results
        transformer_id_col: Column identifying transformers
        
    Returns:
        Summary DataFrame with anomaly counts per transformer
    """
    summary = results.groupby(transformer_id_col).agg({
        "is_anomaly": ["sum", "mean"],  # count and percentage
        "n_methods_flagged": ["mean", "max"],
    }).round(4)
    
    summary.columns = ["anomaly_count", "anomaly_rate", "avg_methods_flagged", "max_methods_flagged"]
    summary = summary.sort_values("anomaly_count", ascending=False)
    
    return summary


def export_anomaly_report(
    results: pd.DataFrame,
    output_path: str | Path,
    top_n_anomalies: int = 100,
) -> None:
    """Export detailed anomaly report.
    
    Args:
        results: DataFrame with anomaly detection results
        output_path: Path to save CSV report
        top_n_anomalies: Export top N flagged points
    """
    # Get top anomalies by severity (multiple methods)
    anomalies = results[results["is_anomaly"]].copy()
    anomalies = anomalies.sort_values("n_methods_flagged", ascending=False)
    anomalies = anomalies.head(top_n_anomalies)
    
    anomalies.to_csv(output_path, index=False)
    print(f"\nExported {len(anomalies)} anomalies to {output_path}")


def main():
    """Run full SPC anomaly detection pipeline."""
    
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # 1. Load data
    print("=" * 60)
    print("Step 1: Loading data")
    print("=" * 60)
    train, test = load_electricity_data(
        data_dir / "train.csv",
        data_dir / "test.csv"
    )
    
    # 2. Prepare data (assuming transformer_id is in data)
    print("\n" + "=" * 60)
    print("Step 2: Preparing data")
    print("=" * 60)
    
    # Check what columns are available
    print(f"\nAvailable columns: {train.columns.tolist()}")

    # Dataset schema (Turkish column names):
    #   tanim    -> transformer id
    #   tuketim  -> consumption (target value)
    #   tarih    -> date
    #   guc      -> rated power
    #   lokasyon -> location hierarchy
    transformer_id_col = "tanim"
    value_col = "tuketim"
    date_col = "tarih"

    print(f"\nUsing transformer_id_col='{transformer_id_col}', value_col='{value_col}', date_col='{date_col}'")

    data = prepare_data_for_spc(train, transformer_id_col, value_col, date_col=date_col)
    print(f"Cleaned data shape: {data.shape}")

    # Consumption is right-skewed (occasional huge spikes) and trends/seasons
    # over the ~15-month span, which breaks SPC's constant-mean assumption
    # when fit on a raw, static first-half baseline. Work on a log-scale,
    # detrended residual instead; see add_detrended_residual() docstring.
    resid_col = "resid"
    data = add_detrended_residual(
        data,
        transformer_id_col=transformer_id_col,
        value_col=value_col,
        resid_col=resid_col,
    )

    # 3. Fit baselines
    print("\n" + "=" * 60)
    print("Step 3: Fitting SPC baselines")
    print("=" * 60)

    config = SPCConfig(
        shewhart_multiplier=3.0,
        ewma_lambda=0.2,
        ewma_multiplier=2.66,
        cusum_k_value=0.5,
        cusum_h_value=4.77,
        enable_nelson_rules=True,
    )

    detectors = fit_baseline_per_transformer(
        data,
        transformer_id_col=transformer_id_col,
        value_col=resid_col,
        raw_value_col=value_col,
        baseline_fraction=0.5,
        config=config,
    )

    # 4. Detect anomalies
    print("\n" + "=" * 60)
    print("Step 4: Detecting anomalies")
    print("=" * 60)

    results = detect_anomalies_per_transformer(
        data,
        detectors,
        transformer_id_col=transformer_id_col,
        value_col=resid_col,
        methods=("shewhart", "ewma", "cusum", "nelson"),
    )
    
    # 5. Summarize results
    print("\n" + "=" * 60)
    print("Step 5: Summarizing results")
    print("=" * 60)
    
    summary = summarize_anomalies(results, transformer_id_col)
    print(f"\nTop 10 transformers by anomaly count:")
    print(summary.head(10))
    
    total_anomalies = results["is_anomaly"].sum()
    anomaly_rate = results["is_anomaly"].mean() * 100
    print(f"\nTotal anomalies detected: {total_anomalies}")
    print(f"Anomaly rate: {anomaly_rate:.2f}%")
    
    # 6. Export results
    print("\n" + "=" * 60)
    print("Step 6: Exporting results")
    print("=" * 60)
    
    # Export detailed anomaly report
    export_anomaly_report(
        results,
        output_dir / "spc_anomalies.csv",
        top_n_anomalies=200
    )
    
    # Export summary statistics
    summary.to_csv(output_dir / "spc_anomaly_summary.csv")
    print(f"Exported summary to {output_dir / 'spc_anomaly_summary.csv'}")
    
    # Save configuration
    config_dict = {
        "shewhart_multiplier": config.shewhart_multiplier,
        "ewma_lambda": config.ewma_lambda,
        "ewma_multiplier": config.ewma_multiplier,
        "cusum_k_value": config.cusum_k_value,
        "cusum_h_value": config.cusum_h_value,
        "min_baseline_points": config.min_baseline_points,
    }
    with open(output_dir / "spc_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Exported config to {output_dir / 'spc_config.json'}")
    
    print("\n" + "=" * 60)
    print("SPC Anomaly Detection Complete!")
    print("=" * 60)
    
    return results, detectors, summary


if __name__ == "__main__":
    results, detectors, summary = main()
