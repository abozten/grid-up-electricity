# Statistical Process Control (SPC) Anomaly Detection

## Overview

This module implements Statistical Process Control techniques for detecting anomalies in time-series data, particularly electricity load data. SPC is a collection of methods originally developed for manufacturing quality control but highly applicable to detecting anomalies in any time-series process.

## Implemented Methods

### 1. Shewhart Control Charts (I-Chart)

**Principle**: Individual points are plotted against fixed control limits derived from baseline statistics.

**Formula**:
- Center Line (CL) = μ (baseline mean)
- Upper Control Limit (UCL) = μ + k·σ (typically k=3)
- Lower Control Limit (LCL) = μ - k·σ

**Use Case**: Quick detection of large, sudden shifts or extreme values.

**Pros**:
- Simple and intuitive
- Detects large deviations immediately
- Widely understood and accepted

**Cons**:
- Insensitive to small systematic shifts
- Each point is treated independently

**Example**: A transformer suddenly consuming 50% more than normal = immediate flag

### 2. EWMA (Exponentially Weighted Moving Average)

**Principle**: Gives exponentially decreasing weights to past observations. Recent values have more influence.

**Formula**:
```
EWMA_t = λ · X_t + (1-λ) · EWMA_{t-1}

where λ ∈ (0, 1] is the smoothing parameter
```

- λ = 0.2 (default) means 20% weight to current observation, 80% to historical EWMA
- λ closer to 0: More smoothing, better for small shifts
- λ closer to 1: Less smoothing, faster response to changes

**Control Limits**: Expand over time initially, then stabilize:
```
UCL_t = CL ± L · σ · √(λ/(2-λ) · [1 - (1-λ)^(2t)])
```

**Use Case**: Detecting small, sustained shifts in the mean.

**Pros**:
- Excellent for detecting small shifts over time
- Adapts smoothly to changes
- Accounts for autocorrelation in data

**Cons**:
- Slower response to very large shocks
- Requires tuning of λ parameter

**Example**: Transformer consumption gradually increases by 5% week-over-week = EWMA flags it

### 3. CUSUM (Cumulative Sum Control Chart)

**Principle**: Accumulates standardized deviations from the mean. When cumulative sum exceeds a decision interval, an out-of-control condition is signaled.

**Formula**:
```
C⁺_t = max(0, C⁺_{t-1} + (Z_t - k))  [Upper CUSUM]
C⁻_t = min(0, C⁻_{t-1} + (Z_t + k))  [Lower CUSUM]

where:
  Z_t = (X_t - μ) / σ  [standardized value]
  k = reference value (typically 0.5)
  Decision: Flag if |C±| > h  (typically h ≈ 4.77)
```

**Use Case**: Most sensitive method for detecting small-to-moderate sustained shifts.

**Pros**:
- Most powerful for detecting small/medium shifts
- Accumulates evidence over time
- Optimal for certain shift sizes

**Cons**:
- Slower response to very large single deviations
- Less intuitive than Shewhart

**Example**: Many points slightly above normal (not flagged individually) = CUSUM accumulates and flags pattern

### 4. Nelson Rules

**Principle**: Pattern-based heuristics that flag various out-of-control conditions.

**Implemented Rules**:
1. One point > 3σ from center
2. 9 consecutive points on same side of center
3. 6 consecutive points steadily increasing/decreasing  
4. 2 out of 3 consecutive points > 2σ on same side
5. 4 out of 5 consecutive points > 1σ on same side

**Use Case**: Multi-faceted detection combining statistical and pattern-based approaches.

**Pros**:
- Catches patterns individual point-based methods miss
- Sensitive to trends and patterns
- No additional parameters

**Cons**:
- Can be overly sensitive
- May flag normal process variations

**Example**: Regular upward trend detected even if no single point is extreme

## Configuration Parameters

```python
SPCConfig(
    # Shewhart control limit width (standard deviations)
    shewhart_multiplier: float = 3.0,  
    
    # EWMA control limit width (standard deviations)
    ewma_multiplier: float = 2.66,
    
    # EWMA smoothing parameter (0 < lambda <= 1)
    ewma_lambda: float = 0.2,  
    # 0.05-0.10: Heavy smoothing, detects small shifts
    # 0.20-0.30: Moderate smoothing (default range)
    # 0.40+: Light smoothing, responds quickly to large shifts
    
    # CUSUM reference value (sigma units)
    cusum_k_value: float = 0.5,  
    
    # CUSUM decision interval (sigma units)
    cusum_h_value: float = 4.77,
    
    # Minimum data points for baseline statistics
    min_baseline_points: int = 30,
)
```

## Usage Examples

### Basic Usage

```python
from spc_anomaly_detection import SPCDetector, SPCConfig
import numpy as np

# Create sample data
data = np.array([100, 101, 99, 102, 103, ...])

# Configure detector
config = SPCConfig(shewhart_multiplier=3.0, ewma_lambda=0.2)
detector = SPCDetector(config)

# Fit baseline from clean data (first 100 points assumed anomaly-free)
detector.fit(data[:100])

# Detect anomalies
anomaly_flags, results = detector.detect(
    data,
    methods=("shewhart", "ewma", "cusum", "nelson"),
    aggregate="any"  # Flag if ANY method detects
)

# View results
print(f"Anomalies found: {anomaly_flags.sum()}")
print(results[results['is_anomaly']])
```

### Multi-Series Processing

```python
from spc_anomaly_detection import MultiSeriesSPCDetector
import pandas as pd

# Data with multiple transformers
df = pd.DataFrame({
    'transformer_id': [1, 1, 1, 2, 2, 2, ...],
    'consumption': [100, 101, 99, 200, 198, 202, ...],
    'timestamp': [...]
})

# Fit baseline for each transformer
detector = MultiSeriesSPCDetector()
detector.fit(df, id_column='transformer_id')

# Detect anomalies for each transformer
results = detector.detect(
    df,
    id_column='transformer_id',
    value_column='consumption',
    methods=("shewhart", "ewma", "cusum")
)

# Get summary
summary = results.groupby('transformer_id')['is_anomaly'].agg(['sum', 'mean'])
```

### Visualization

```python
from spc_anomaly_detection import visualize_spc_results
import matplotlib.pyplot as plt

# Plot all control charts
visualize_spc_results(data, detector, title="Electricity Load Analysis")
plt.show()

# Manual plotting
vals, center, ucl, lcl = detector.shewhart_chart(data)
plt.plot(data, label='Data')
plt.axhline(center, linestyle='--', label='Center')
plt.fill_between(range(len(data)), lcl, ucl, alpha=0.2, label='Control Limits')
plt.show()
```

## Running the Full Pipeline

### 1. Using the Script

```bash
cd src/
python run_spc_detection.py
```

This will:
- Load your electricity data
- Fit baselines per transformer
- Detect anomalies
- Export results to `outputs/`:
  - `spc_anomalies.csv` - Top 200 flagged points
  - `spc_anomaly_summary.csv` - Statistics per transformer
  - `spc_config.json` - Configuration used

### 2. Using the Jupyter Notebook

```bash
jupyter notebook SPC_Anomaly_Detection_Guide.ipynb
```

The notebook includes:
- Synthetic data demonstrations
- Step-by-step method explanation
- Real data processing examples
- Visualization tools

## Output Files

### `spc_anomalies.csv`
Detailed report of detected anomalies with columns:
- Original data columns
- `shewhart_anomaly`: Flagged by Shewhart chart
- `ewma_anomaly`: Flagged by EWMA
- `cusum_anomaly`: Flagged by CUSUM
- `nelson_anomaly`: Flagged by Nelson rules
- `n_methods_flagged`: How many methods detected this point

Sort by `n_methods_flagged` to find most robust anomalies.

### `spc_anomaly_summary.csv`
Per-transformer statistics:
- `anomaly_count`: Total anomalies detected
- `anomaly_rate`: Percentage of points flagged
- `avg_methods_flagged`: Average number of methods detecting each anomaly
- `max_methods_flagged`: Maximum (perfect consensus = 4)

## Tuning Recommendations

### For Your Electricity Data

**Recommended configuration for baseline tuning:**

```python
config = SPCConfig(
    shewhart_multiplier=3.0,      # Standard 3-sigma
    ewma_lambda=0.15,             # Sensitive to small shifts
    cusum_k_value=0.5,            # Standard reference value
    cusum_h_value=4.77,           # Standard decision interval
)
```

### Sensitivity Adjustments

**More Sensitive** (more false positives, catch more subtle anomalies):
```python
config = SPCConfig(
    shewhart_multiplier=2.5,      # Tighter Shewhart limits
    ewma_lambda=0.10,             # More smoothing
    ewma_multiplier=2.0,          # Wider EWMA limits
)
```

**Less Sensitive** (fewer false positives, catch only major anomalies):
```python
config = SPCConfig(
    shewhart_multiplier=3.5,      # Wider Shewhart limits
    ewma_lambda=0.30,             # Less smoothing
    ewma_multiplier=3.5,          # Tighter EWMA limits
)
```

### Method Selection

**Detect sudden spikes only**:
```python
methods = ("shewhart", "nelson")
aggregate = "all"  # Only flag if multiple methods agree
```

**Sensitive to gradual shifts**:
```python
methods = ("ewma", "cusum", "nelson")
aggregate = "any"
```

**Balanced approach**:
```python
methods = ("shewhart", "ewma", "cusum", "nelson")
aggregate = "majority"  # Flag if 2+ methods agree
```

## Interpretation Guide

### Understanding Results

**High confidence anomalies**:
- `n_methods_flagged` = 4: All methods agree - likely a real anomaly
- `n_methods_flagged` = 3: Strong consensus
- `n_methods_flagged` = 2: Multiple methods agree

**Low confidence anomalies**:
- `n_methods_flagged` = 1: Only one method detected - verify manually

### Common Patterns

| Pattern | Methods Flagged | Interpretation |
|---------|-----------------|-----------------|
| Spike | Shewhart, Nelson | Sudden, isolated event |
| Step shift | EWMA, CUSUM, Nelson | Sustained change in level |
| Trend | Nelson, CUSUM | Gradual systematic change |
| Oscillation | Nelson | Unstable, alternating pattern |

## Troubleshooting

### Too Many False Positives
- Increase `shewhart_multiplier` (e.g., 3.5 instead of 3.0)
- Increase `ewma_lambda` (e.g., 0.3 instead of 0.2)
- Exclude Nelson rules: `methods = ("shewhart", "ewma", "cusum")`
- Use `aggregate = "majority"` or `"all"`

### Missing Real Anomalies
- Decrease `shewhart_multiplier` (e.g., 2.5 instead of 3.0)
- Decrease `ewma_lambda` (e.g., 0.1 instead of 0.2)
- Include Nelson rules: `methods = ("shewhart", "ewma", "cusum", "nelson")`
- Use `aggregate = "any"`

### Baseline Not Established
- Ensure `baseline_data` has no anomalies
- Use at least 30 clean data points (configurable)
- Check baseline statistics make sense for your data

## References

1. Montgomery, D. C. (2009). *Statistical Quality Control* (6th ed.). Wiley.
2. NIST/SEMATECH e-Handbook of Statistical Methods. https://www.itl.nist.gov/div898/handbook/
3. Wheeler, D. J., & Chambers, D. S. (1992). *Understanding Statistical Process Control*. SPC Press.
4. Knottenbelt, W. J., et al. (2017). "Demand Forecasting in the Smart Grid". https://arxiv.org/abs/1710.02154

## License

This module is provided as-is for analysis of the electricity grid data project.

## Contact & Support

For questions about SPC methods or results interpretation, refer to:
- The included Jupyter notebook for step-by-step walkthroughs
- NIST handbook for theoretical details
- Montgomery's textbook for practical implementation guidance
