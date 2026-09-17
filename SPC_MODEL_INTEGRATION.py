"""
Integration Guide: Using SPC Anomalies with Your Forecasting Models

Shows how to use detected anomalies to:
1. Filter training data
2. Create anomaly features
3. Segment data by anomaly type
4. Improve model robustness
"""

import numpy as np
import pandas as pd
from pathlib import Path
from spc_anomaly_detection import SPCDetector, SPCConfig


def filter_anomalies_for_training(
    train_data: pd.DataFrame,
    detector: SPCDetector,
    transformer_id_col: str,
    value_col: str,
    strategy: str = "remove",
) -> pd.DataFrame:
    """
    Filter anomalies from training data to improve model quality.
    
    Args:
        train_data: Training DataFrame
        detector: Fitted SPCDetector
        transformer_id_col: Column with transformer IDs
        value_col: Column with values
        strategy: "remove" (delete anomalies), "clip" (cap at limits), "flag" (add flag column)
        
    Returns:
        Filtered/modified training data
    """
    data = train_data.copy()
    
    if strategy == "remove":
        # Completely remove anomalous rows
        print(f"Before filtering: {len(data)} rows")
        
        anom_flags, _ = detector.detect(data[value_col].values)
        data = data[~anom_flags]
        
        print(f"After removing anomalies: {len(data)} rows")
        print(f"Removed {anom_flags.sum()} anomalous points ({anom_flags.mean()*100:.2f}%)")
        
    elif strategy == "clip":
        # Clip anomalous values to control limits
        vals, center, ucl, lcl = detector.shewhart_chart(data[value_col].values)
        data[value_col] = np.clip(data[value_col], lcl[0], ucl[0])
        print(f"Clipped anomalous values to [{lcl[0]:.2f}, {ucl[0]:.2f}]")
        
    elif strategy == "flag":
        # Keep data but flag anomalies for the model
        anom_flags, _ = detector.detect(data[value_col].values)
        data['is_anomaly'] = anom_flags
        print(f"Added anomaly flag column ({anom_flags.sum()} anomalies)")
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return data


def create_anomaly_features(
    data: pd.DataFrame,
    detector: SPCDetector,
    value_col: str,
    window: int = 7,
) -> pd.DataFrame:
    """
    Create features based on anomaly detection for use in models.
    
    Features:
    - is_anomaly: Binary flag
    - anomaly_score: How many methods flagged this point
    - distance_from_center: How far from mean (in sigma units)
    - z_score: Standardized value
    - days_since_anomaly: Days since last detected anomaly
    
    Args:
        data: DataFrame with values
        detector: Fitted SPCDetector
        value_col: Column with values
        window: Window for moving statistics
        
    Returns:
        DataFrame with added feature columns
    """
    data = data.copy()
    values = data[value_col].values
    
    # 1. Anomaly detection results
    anom_flags, detailed_results = detector.detect(values)
    
    # 2. Anomaly score (how many methods agree)
    if 'is_anomaly' in detailed_results.columns:
        anomaly_methods = [col for col in detailed_results.columns 
                          if 'anomaly' in col and col != 'is_anomaly']
        data['anomaly_score'] = detailed_results[anomaly_methods].astype(int).sum(axis=1)
    else:
        data['anomaly_score'] = anom_flags.astype(int)
    
    # 3. Distance from center (in sigma units)
    if detector.baseline_std > 0:
        data['z_score'] = (values - detector.baseline_mean) / detector.baseline_std
        data['distance_from_center'] = np.abs(data['z_score'])
    
    # 4. Is anomaly flag
    data['is_anomaly'] = anom_flags
    
    # 5. Days since last anomaly
    anom_indices = np.where(anom_flags)[0]
    data['days_since_anomaly'] = np.inf
    for i, idx in enumerate(anom_indices):
        if i < len(anom_indices):
            next_idx = anom_indices[i + 1] if i + 1 < len(anom_indices) else len(data)
            data.loc[idx:next_idx, 'days_since_anomaly'] = \
                np.minimum(np.arange(next_idx - idx), data.loc[idx:next_idx, 'days_since_anomaly'].values)
    
    # 6. Anomaly density (rolling count)
    data['anomaly_density'] = data['is_anomaly'].rolling(window).sum()
    
    # 7. EWMA-based trend (for feature importance)
    ewma_vals, _, _, _ = detector.ewma_chart(values)
    data['ewma_trend'] = ewma_vals
    data['deviation_from_ewma'] = values - ewma_vals
    
    return data


def segment_by_anomaly_type(
    data: pd.DataFrame,
    detector: SPCDetector,
    value_col: str,
) -> dict:
    """
    Segment data by type of anomaly detected.
    
    Returns:
        Dictionary with keys: normal, spikes, shifts, trends, mixed
    """
    values = data[value_col].values
    
    # Get anomaly flags from each method
    _, detailed = detector.detect(values)
    
    segments = {
        'normal': data[~detailed['is_anomaly']].copy(),
        'shewhart_only': data[detailed.get('shewhart_anomaly', False) & 
                             ~detailed.get('ewma_anomaly', False) &
                             ~detailed.get('cusum_anomaly', False)].copy(),
        'ewma_only': data[detailed.get('ewma_anomaly', False) & 
                         ~detailed.get('shewhart_anomaly', False)].copy(),
        'cusum_only': data[detailed.get('cusum_anomaly', False) & 
                          ~detailed.get('shewhart_anomaly', False)].copy(),
        'multiple_methods': data[(detailed.get('shewhart_anomaly', False).astype(int) +
                                  detailed.get('ewma_anomaly', False).astype(int) +
                                  detailed.get('cusum_anomaly', False).astype(int)) >= 2].copy(),
    }
    
    print("Data segments by anomaly type:")
    for name, segment in segments.items():
        pct = len(segment) / len(data) * 100
        print(f"  {name:20s}: {len(segment):6d} rows ({pct:5.2f}%)")
    
    return segments


def robust_train_test_split(
    data: pd.DataFrame,
    detector: SPCDetector,
    value_col: str,
    test_size: float = 0.2,
    stratify_by_anomaly: bool = True,
):
    """
    Split data for training/testing while accounting for anomalies.
    
    This ensures both train and test sets have similar anomaly distributions.
    
    Args:
        data: Full dataset
        detector: Fitted detector
        value_col: Column with values
        test_size: Fraction for test set
        stratify_by_anomaly: If True, balance anomalies between train/test
        
    Returns:
        Tuple of (train_data, test_data, split_info)
    """
    from sklearn.model_selection import train_test_split
    
    values = data[value_col].values
    anom_flags, _ = detector.detect(values)
    
    if stratify_by_anomaly:
        train_data, test_data = train_test_split(
            data,
            test_size=test_size,
            stratify=anom_flags,  # Ensure similar anomaly % in train/test
            random_state=42
        )
    else:
        train_data, test_data = train_test_split(
            data,
            test_size=test_size,
            random_state=42
        )
    
    # Get statistics
    train_anom_rate = anom_flags[:len(train_data)].mean() * 100
    test_anom_rate = anom_flags[len(train_data):].mean() * 100
    
    split_info = {
        'train_size': len(train_data),
        'test_size': len(test_data),
        'train_anomaly_rate': train_anom_rate,
        'test_anomaly_rate': test_anom_rate,
    }
    
    print("Train/Test Split with Anomaly Stratification:")
    print(f"  Train: {split_info['train_size']:6d} rows ({split_info['train_anomaly_rate']:.2f}% anomalies)")
    print(f"  Test:  {split_info['test_size']:6d} rows ({split_info['test_anomaly_rate']:.2f}% anomalies)")
    
    return train_data, test_data, split_info


def evaluate_model_on_anomalies(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_anomaly: np.ndarray,
    metric_func=None,
) -> dict:
    """
    Evaluate model performance separately on normal vs anomalous data.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        is_anomaly: Boolean array of anomalies
        metric_func: Function to compute metric (default: RMSLE)
        
    Returns:
        Dictionary with per-segment metrics
    """
    from sklearn.metrics import mean_squared_log_error
    
    if metric_func is None:
        metric_func = lambda y, p: np.sqrt(mean_squared_log_error(y, p))
    
    results = {
        'overall': metric_func(y_true, y_pred),
        'normal': metric_func(y_true[~is_anomaly], y_pred[~is_anomaly]),
        'anomalies': metric_func(y_true[is_anomaly], y_pred[is_anomaly]),
    }
    
    print("Model Performance by Data Type:")
    print(f"  Overall:   {results['overall']:.4f}")
    print(f"  Normal:    {results['normal']:.4f}")
    print(f"  Anomalies: {results['anomalies']:.4f}")
    print(f"  Ratio (Anomaly/Normal): {results['anomalies']/results['normal']:.2f}x worse")
    
    return results


def integration_example():
    """
    Complete example showing how to integrate SPC with forecasting models.
    """
    from pathlib import Path
    
    project_root = Path(__file__).parent
    data_path = project_root / "data" / "train.csv"
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return
    
    # Load data
    print("=" * 70)
    print("STEP 1: Load and Prepare Data")
    print("=" * 70)
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
    
    # TODO: Adjust column names
    ID_COL = "transformer_id"  # Change as needed
    VALUE_COL = "consumption"  # Change as needed
    
    if ID_COL not in df.columns or VALUE_COL not in df.columns:
        print(f"ERROR: Columns {ID_COL} or {VALUE_COL} not found")
        print(f"Available: {df.columns.tolist()}")
        return
    
    # Fit SPC detector
    print("\n" + "=" * 70)
    print("STEP 2: Fit SPC Detector")
    print("=" * 70)
    
    config = SPCConfig()
    detector = SPCDetector(config)
    
    # Use first transformer as example
    first_id = df[ID_COL].iloc[0]
    transformer_data = df[df[ID_COL] == first_id]
    
    baseline_size = min(500, len(transformer_data) // 2)  # Use first 500 or 50%
    detector.fit(transformer_data[VALUE_COL].iloc[:baseline_size].values)
    
    print(f"Fitted on {baseline_size} baseline points")
    print(f"Baseline: mean={detector.baseline_mean:.2f}, std={detector.baseline_std:.2f}")
    
    # Create anomaly features
    print("\n" + "=" * 70)
    print("STEP 3: Create Anomaly Features")
    print("=" * 70)
    
    enriched_data = create_anomaly_features(
        transformer_data.copy(),
        detector,
        value_col=VALUE_COL
    )
    
    print(f"Added features: {enriched_data.columns.tolist()[-7:]}")
    print(f"Anomalies detected: {enriched_data['is_anomaly'].sum()}")
    
    # Filter for training
    print("\n" + "=" * 70)
    print("STEP 4: Prepare Training Data")
    print("=" * 70)
    
    # Option A: Remove anomalies (cleaner data)
    train_clean = filter_anomalies_for_training(
        enriched_data,
        detector,
        ID_COL,
        VALUE_COL,
        strategy="remove"
    )
    
    # Option B: Keep all data with anomaly flag
    train_with_flags = enriched_data.copy()
    
    # Segment by anomaly type
    print("\n" + "=" * 70)
    print("STEP 5: Analyze by Anomaly Type")
    print("=" * 70)
    
    segments = segment_by_anomaly_type(enriched_data, detector, VALUE_COL)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Next Steps")
    print("=" * 70)
    print("""
1. Use train_clean for a baseline model
2. Compare with train_with_flags model (using anomaly features)
3. Check model.feature_importance_ for anomaly_score contribution
4. For production: retrain detector on recent data regularly
5. Monitor anomaly_density for data quality issues
    """)


if __name__ == "__main__":
    integration_example()
