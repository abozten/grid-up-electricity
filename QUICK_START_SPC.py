"""
QUICK START GUIDE: SPC Anomaly Detection

Choose one of the following approaches based on your needs:
"""

# ============================================================================
# APPROACH 1: Run the Full Pipeline (Recommended for First-Time Use)
# ============================================================================
"""
Usage: python src/run_spc_detection.py

This script:
✓ Loads your electricity data
✓ Fits baseline statistics per transformer
✓ Applies all 4 SPC methods (Shewhart, EWMA, CUSUM, Nelson)
✓ Generates 3 output files with results
✓ Prints summary statistics

Output files:
  - outputs/spc_anomalies.csv (top 200 anomalies)
  - outputs/spc_anomaly_summary.csv (per-transformer stats)
  - outputs/spc_config.json (configuration used)
"""


# ============================================================================
# APPROACH 2: Interactive Notebook (Best for Exploration)
# ============================================================================
"""
Usage: jupyter notebook SPC_Anomaly_Detection_Guide.ipynb

This notebook provides:
✓ Step-by-step SPC method explanations
✓ Synthetic examples for each method
✓ Visualization of all control charts
✓ Real data integration examples
✓ Interactive parameter tuning
✓ Export functions

Best for understanding how each method works
"""


# ============================================================================
# APPROACH 3: Custom Python Script (Most Flexible)
# ============================================================================

import numpy as np
import pandas as pd
from pathlib import Path
from spc_anomaly_detection import SPCDetector, SPCConfig

# Define paths
project_root = Path(__file__).parent.parent
data_path = project_root / "data" / "train.csv"

# STEP 1: Load your data
print("Loading data...")
df = pd.read_csv(data_path)

# TODO: Adjust these column names to match your actual data
ID_COLUMN = "transformer_id"  # Column identifying different time series
VALUE_COLUMN = "consumption"   # Column with values to analyze

# STEP 2: Basic preprocessing
print("Preparing data...")
data = df.copy()
data = data.dropna(subset=[ID_COLUMN, VALUE_COLUMN])  # Remove NaN
data = data[data[VALUE_COLUMN] > 0]  # Remove zero/negative values
print(f"Data shape: {data.shape}")

# STEP 3: Configure SPC detector
config = SPCConfig(
    shewhart_multiplier=3.0,      # 3-sigma limits (standard)
    ewma_lambda=0.2,              # Smoothing parameter
    ewma_multiplier=2.66,         # EWMA control limit width
    cusum_k_value=0.5,            # CUSUM reference value
    cusum_h_value=4.77,           # CUSUM decision interval
)

# STEP 4: Process each transformer
print("Detecting anomalies...")
results_list = []

for transformer_id in data[ID_COLUMN].unique():
    transformer_data = data[data[ID_COLUMN] == transformer_id]
    values = transformer_data[VALUE_COLUMN].values
    
    if len(values) < 30:  # Skip if too few points
        continue
    
    # Fit baseline on first 50% of data (assumed clean)
    baseline_size = max(30, len(values) // 2)
    
    detector = SPCDetector(config)
    detector.fit(values[:baseline_size])
    
    # Detect anomalies on full dataset
    anom_flags, detailed_results = detector.detect(
        values,
        methods=("shewhart", "ewma", "cusum", "nelson"),
        aggregate="any"  # Flag if ANY method detects
    )
    
    # Store results with original indices
    transformer_results = transformer_data.copy()
    transformer_results['is_anomaly'] = anom_flags
    results_list.append(transformer_results)

# Combine all results
all_results = pd.concat(results_list, ignore_index=True)

# STEP 5: Generate insights
print("\n" + "=" * 60)
print("ANOMALY DETECTION SUMMARY")
print("=" * 60)

total_anomalies = all_results['is_anomaly'].sum()
total_points = len(all_results)
anomaly_rate = total_anomalies / total_points * 100

print(f"\nTotal anomalies detected: {total_anomalies:,}")
print(f"Total data points: {total_points:,}")
print(f"Anomaly rate: {anomaly_rate:.2f}%")

# Per-transformer statistics
by_transformer = all_results.groupby(ID_COLUMN).agg({
    'is_anomaly': ['sum', 'mean']
}).round(4)
by_transformer.columns = ['anomaly_count', 'anomaly_rate']
by_transformer = by_transformer.sort_values('anomaly_count', ascending=False)

print(f"\nTop 10 transformers by anomaly count:")
print(by_transformer.head(10))

# STEP 6: Export results
output_path = project_root / "outputs" / "spc_anomalies_custom.csv"
all_results[all_results['is_anomaly']].to_csv(output_path, index=False)
print(f"\nExported anomalies to: {output_path}")


# ============================================================================
# APPROACH 4: Real-Time Monitoring (Stream Processing)
# ============================================================================
"""
For continuous/real-time anomaly detection:

import pandas as pd
from spc_anomaly_detection import SPCDetector, SPCConfig

# Fit once on historical data
detector = SPCDetector()
detector.fit(historical_clean_data)

# For each new data point
while True:
    new_value = get_next_measurement()
    
    # Detect in real-time
    is_anomaly, _ = detector.detect(np.array([new_value]))
    
    if is_anomaly[0]:
        alert(f"Anomaly detected: {new_value}")
"""


# ============================================================================
# APPROACH 5: Compare Methods Side-by-Side
# ============================================================================
"""
To understand which method works best for your data:

import matplotlib.pyplot as plt

detector = SPCDetector(config)
detector.fit(baseline_data)

# Get results from each method
anom_s, _ = detector.detect(data, methods=("shewhart",))
anom_e, _ = detector.detect(data, methods=("ewma",))
anom_c, _ = detector.detect(data, methods=("cusum",))
anom_n, _ = detector.detect(data, methods=("nelson",))

# Visualize
fig, axes = plt.subplots(5, 1, figsize=(14, 10))

for ax, (anom, name) in zip(axes, 
    [(anom_s, "Shewhart"), (anom_e, "EWMA"), 
     (anom_c, "CUSUM"), (anom_n, "Nelson")]):
    ax.plot(data, 'b-', alpha=0.5, label='Data')
    ax.scatter(np.where(anom)[0], data[anom], color='red', s=50, label='Anomaly')
    ax.set_title(name)
    ax.legend()

plt.tight_layout()
plt.show()
"""


# ============================================================================
# CONFIGURATION TUNING GUIDE
# ============================================================================

CONFIG_EXAMPLES = {
    "aggressive": {
        "description": "Catches all potential anomalies (higher false positive rate)",
        "config": SPCConfig(
            shewhart_multiplier=2.5,  # Tighter limits
            ewma_lambda=0.1,          # More smoothing = sensitive to small shifts
            ewma_multiplier=2.0,
        ),
    },
    "balanced": {
        "description": "Default - good for most cases",
        "config": SPCConfig(
            shewhart_multiplier=3.0,
            ewma_lambda=0.2,
            ewma_multiplier=2.66,
        ),
    },
    "conservative": {
        "description": "Only catches obvious anomalies (lower false positive rate)",
        "config": SPCConfig(
            shewhart_multiplier=3.5,  # Wider limits
            ewma_lambda=0.3,          # Less smoothing
            ewma_multiplier=3.5,
        ),
    },
}

# To use: detector = SPCDetector(CONFIG_EXAMPLES["aggressive"]["config"])


# ============================================================================
# INTERPRETATION QUICK REFERENCE
# ============================================================================

INTERPRETATION = """
When analyzing results, look for:

1. ANOMALY TYPE
   - Isolated spike: Sudden equipment malfunction or meter error
   - Step shift: Change in consumption pattern, new load added
   - Gradual trend: Seasonal effect or efficiency change
   - Oscillation: Control system hunting or unstable operation

2. CONFIRM WITH CONTEXT
   - Check date/time of anomaly for maintenance or events
   - Correlate with other transformers in same area
   - Review seasonal patterns (summer vs winter)
   - Check for known scheduled maintenance

3. SIGNAL vs NOISE
   - If n_methods_flagged = 4: Very likely real anomaly
   - If n_methods_flagged = 3: Probably real anomaly
   - If n_methods_flagged = 2: Likely real anomaly
   - If n_methods_flagged = 1: May be false positive

4. ACTION ITEMS
   - n_methods >= 3: Investigate further
   - Repeated patterns: Systematic issue
   - Single occurrence: Likely transient event
"""

print(INTERPRETATION)
