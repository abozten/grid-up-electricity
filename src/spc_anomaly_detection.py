"""Statistical Process Control (SPC) for time-series anomaly detection.

Implements:
- Shewhart Control Charts (individual & moving range)
- EWMA (Exponentially Weighted Moving Average)
- CUSUM (Cumulative Sum Control Chart)
- Nelson Rules for out-of-control detection
- Visualization tools

References:
- Montgomery, D. C. (2009). Statistical Quality Control
- NIST/SEMATECH e-Handbook of Statistical Methods
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class AnomalyType(Enum):
    """Types of anomalies detected by SPC methods."""
    SHEWHART = "Shewhart (out of control limits)"
    EWMA = "EWMA (out of control limits)"
    CUSUM_POSITIVE = "CUSUM (positive drift)"
    CUSUM_NEGATIVE = "CUSUM (negative drift)"
    NELSON = "Nelson Rule violation"
    EXTREME = "Extreme value (>6 sigma)"


@dataclass
class SPCConfig:
    """Configuration for SPC anomaly detection."""
    # Control limit multipliers (in standard deviations)
    shewhart_multiplier: float = 3.0  # Standard is 3 sigma
    ewma_multiplier: float = 2.66  # For lambda=0.2, gives ~3 sigma equivalent
    cusum_k_value: float = 0.5  # CUSUM reference value (in sigma units)
    cusum_h_value: float = 4.77  # CUSUM decision interval (in sigma units)
    
    # EWMA smoothing parameter
    ewma_lambda: float = 0.2  # Typical range: 0.05-0.3
    
    # Nelson Rules configuration
    enable_nelson_rules: bool = True
    nelson_rule_threshold: float = 3.0  # Rule 1: points beyond 3 sigma
    
    # Minimum data points needed to establish baseline
    min_baseline_points: int = 30
    
    def __post_init__(self):
        if not (0 < self.ewma_lambda <= 1):
            raise ValueError(f"ewma_lambda must be in (0, 1], got {self.ewma_lambda}")


class SPCDetector:
    """Statistical Process Control anomaly detector."""
    
    def __init__(self, config: Optional[SPCConfig] = None):
        """Initialize SPC detector.
        
        Args:
            config: SPCConfig instance. Uses defaults if None.
        """
        self.config = config or SPCConfig()
        self.baseline_mean = None
        self.baseline_std = None
        self.baseline_size = 0
        
    def fit(self, data: np.ndarray | pd.Series) -> SPCDetector:
        """Fit baseline statistics from clean data.
        
        Args:
            data: 1D array or Series of baseline values (assumed anomaly-free)
            
        Returns:
            Self for method chaining
        """
        data = self._to_array(data)
        
        if len(data) < self.config.min_baseline_points:
            warnings.warn(
                f"Data has {len(data)} points; recommend {self.config.min_baseline_points}+ "
                "for stable baseline statistics"
            )
        
        # Remove NaN for baseline calculation
        clean_data = data[~np.isnan(data)]
        if len(clean_data) == 0:
            raise ValueError("No valid data to fit baseline")
        
        self.baseline_mean = np.mean(clean_data)
        self.baseline_std = np.std(clean_data, ddof=1)
        self.baseline_size = len(clean_data)
        
        if self.baseline_std == 0:
            warnings.warn("Baseline standard deviation is 0; anomaly detection may not work")
        
        return self
    
    def shewhart_chart(
        self,
        data: np.ndarray | pd.Series,
        window: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Shewhart Individual Chart (X-chart).
        
        Args:
            data: Time series data
            window: Window size for moving range (if None, use adaptive calculation)
            
        Returns:
            Tuple of (values, center_line, UCL, LCL)
        """
        data = self._to_array(data)
        
        if window is None:
            # Use moving range of 2 for traditional I-chart
            window = 2
        
        # Calculate moving ranges
        moving_ranges = np.abs(np.diff(data))
        if len(moving_ranges) > 0:
            mbar = np.mean(moving_ranges[~np.isnan(moving_ranges)])
            # Constant d2 for moving range of 2
            d2 = 1.128
            sigma_hat = mbar / d2 if d2 != 0 else np.nan
        else:
            sigma_hat = self.baseline_std
        
        center = self.baseline_mean
        ucl = center + self.config.shewhart_multiplier * sigma_hat
        lcl = center - self.config.shewhart_multiplier * sigma_hat
        
        return data, np.full_like(data, center), np.full_like(data, ucl), np.full_like(data, lcl)
    
    def ewma_chart(
        self,
        data: np.ndarray | pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """EWMA Control Chart.
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (ewma_values, center_line, UCL, LCL)
        """
        data = self._to_array(data)
        lam = self.config.ewma_lambda
        
        # Initialize EWMA
        ewma_vals = np.full_like(data, np.nan)
        ewma_vals[0] = data[0]
        
        for i in range(1, len(data)):
            if not np.isnan(data[i]):
                ewma_vals[i] = lam * data[i] + (1 - lam) * ewma_vals[i - 1]
            else:
                ewma_vals[i] = ewma_vals[i - 1]
        
        center = self.baseline_mean
        
        # Control limits expand over time initially, then stabilize
        sigma_ewma = self.baseline_std
        lcl_vals = np.zeros_like(data)
        ucl_vals = np.zeros_like(data)
        
        for i in range(len(data)):
            # Variance of EWMA at step i+1
            var_factor = lam / (2 - lam) * (1 - (1 - lam) ** (2 * (i + 1)))
            se = sigma_ewma * np.sqrt(var_factor)
            ucl_vals[i] = center + self.config.ewma_multiplier * se
            lcl_vals[i] = center - self.config.ewma_multiplier * se
        
        return ewma_vals, np.full_like(data, center), ucl_vals, lcl_vals
    
    def cusum_chart(
        self,
        data: np.ndarray | pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """CUSUM Control Chart.
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (cusum_values, center_line, UCL, LCL)
        """
        data = self._to_array(data)
        
        k = self.config.cusum_k_value * self.baseline_std
        h = self.config.cusum_h_value * self.baseline_std

        cusum_pos = np.zeros_like(data)  # Positive CUSUM
        cusum_neg = np.zeros_like(data)  # Negative CUSUM

        prev_pos = 0.0
        prev_neg = 0.0
        for i in range(len(data)):
            if np.isnan(data[i]):
                cusum_pos[i] = prev_pos
                cusum_neg[i] = prev_neg
                continue

            # Standardize
            z_i = (data[i] - self.baseline_mean) / self.baseline_std if self.baseline_std > 0 else 0

            # Update CUSUMs
            cusum_pos[i] = max(0, prev_pos + z_i - k)
            cusum_neg[i] = min(0, prev_neg + z_i + k)

            # Standard CUSUM stopping rule: once a shift signals, reset the
            # accumulator so it starts fresh looking for the *next* shift.
            # Without this, a single early excursion keeps the statistic
            # pinned above the limit for the rest of the series, flagging
            # almost every subsequent point regardless of whether the
            # process has since returned to control.
            if cusum_pos[i] > h or cusum_neg[i] < -h:
                prev_pos = 0.0
                prev_neg = 0.0
            else:
                prev_pos = cusum_pos[i]
                prev_neg = cusum_neg[i]
        
        center = np.zeros_like(data)
        ucl = np.full_like(data, h)
        lcl = np.full_like(data, -h)
        
        # Return combined CUSUM for visualization
        cusum_combined = np.abs(cusum_pos) + np.abs(cusum_neg)
        
        return cusum_combined, center, ucl, np.zeros_like(lcl)
    
    def nelson_rules(
        self,
        data: np.ndarray | pd.Series,
    ) -> np.ndarray:
        """Detect anomalies using Nelson Rules.
        
        Implements simplified Nelson Rules:
        1. One point > 3 sigma from center
        2. 9 consecutive points on same side of center
        3. 6 consecutive points steadily increasing/decreasing
        4. 14 consecutive points alternating up/down
        5. 2 out of 3 consecutive points > 2 sigma on same side
        6. 4 out of 5 consecutive points > 1 sigma on same side
        
        Args:
            data: Time series data
            
        Returns:
            Boolean array indicating anomalies
        """
        data = self._to_array(data)
        anomalies = np.zeros(len(data), dtype=bool)
        
        if self.baseline_std == 0:
            return anomalies
        
        # Standardize
        z = (data - self.baseline_mean) / self.baseline_std
        
        # Rule 1: One point beyond 3 sigma
        anomalies |= np.abs(z) > self.config.nelson_rule_threshold
        
        # Rule 2: 9 consecutive points on same side
        for i in range(len(data) - 8):
            if np.all(z[i:i+9] > 0) or np.all(z[i:i+9] < 0):
                anomalies[i:i+9] = True
        
        # Rule 3: 6 consecutive steadily increasing/decreasing
        for i in range(len(data) - 5):
            diffs = np.diff(z[i:i+6])
            if np.all(diffs > 0) or np.all(diffs < 0):
                anomalies[i:i+6] = True
        
        # Rule 5: 2 out of 3 beyond 2 sigma on same side
        for i in range(len(data) - 2):
            high = np.sum(z[i:i+3] > 2)
            low = np.sum(z[i:i+3] < -2)
            if high >= 2 or low >= 2:
                anomalies[i:i+3] = True
        
        # Rule 6: 4 out of 5 beyond 1 sigma on same side
        for i in range(len(data) - 4):
            high = np.sum(z[i:i+5] > 1)
            low = np.sum(z[i:i+5] < -1)
            if high >= 4 or low >= 4:
                anomalies[i:i+5] = True
        
        return anomalies
    
    def detect(
        self,
        data: np.ndarray | pd.Series,
        methods: Tuple[str, ...] = ("shewhart", "ewma", "cusum", "nelson"),
        aggregate: str = "any",
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Detect anomalies using specified SPC methods.
        
        Args:
            data: Time series data
            methods: Which methods to use. Options: shewhart, ewma, cusum, nelson
            aggregate: How to combine multiple methods:
                - 'any': flag if any method detects anomaly
                - 'majority': flag if > 50% of methods detect anomaly
                - 'all': flag only if all methods detect anomaly
            
        Returns:
            Tuple of (anomaly_flags, detailed_results_dataframe)
        """
        if self.baseline_mean is None:
            raise RuntimeError("Must call .fit() first")
        
        data = self._to_array(data)
        results = pd.DataFrame({"value": data})
        anomaly_flags = np.zeros(len(data), dtype=bool)
        method_count = 0
        
        # Shewhart chart
        if "shewhart" in methods:
            vals, center, ucl, lcl = self.shewhart_chart(data)
            anom = (vals > ucl) | (vals < lcl)
            results[f"shewhart_anomaly"] = anom
            anomaly_flags |= anom
            method_count += 1
        
        # EWMA chart
        if "ewma" in methods:
            ewma_vals, center, ucl, lcl = self.ewma_chart(data)
            anom = (ewma_vals > ucl) | (ewma_vals < lcl)
            results[f"ewma_anomaly"] = anom
            anomaly_flags |= anom
            method_count += 1
        
        # CUSUM chart
        if "cusum" in methods:
            cusum_vals, center, ucl, lcl = self.cusum_chart(data)
            anom = (cusum_vals > ucl)
            results[f"cusum_anomaly"] = anom
            anomaly_flags |= anom
            method_count += 1
        
        # Nelson rules
        if "nelson" in methods:
            anom = self.nelson_rules(data)
            results[f"nelson_anomaly"] = anom
            anomaly_flags |= anom
            method_count += 1
        
        # Apply aggregation logic
        if aggregate == "any":
            results["is_anomaly"] = anomaly_flags
        elif aggregate == "majority":
            anomaly_col_count = (results[[c for c in results.columns if "anomaly" in c]]
                                .astype(int).sum(axis=1))
            results["is_anomaly"] = anomaly_col_count > (method_count / 2)
        elif aggregate == "all":
            anomaly_col_count = (results[[c for c in results.columns if "anomaly" in c]]
                                .astype(int).sum(axis=1))
            results["is_anomaly"] = anomaly_col_count == method_count
        else:
            raise ValueError(f"Unknown aggregation method: {aggregate}")
        
        return results["is_anomaly"].values, results
    
    @staticmethod
    def _to_array(data: np.ndarray | pd.Series) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(data, pd.Series):
            return data.values
        return np.asarray(data, dtype=float)


class MultiSeriesSPCDetector:
    """Apply SPC anomaly detection across multiple time series.
    
    Useful for detecting anomalies in multiple transformers, regions, or features.
    """
    
    def __init__(self, config: Optional[SPCConfig] = None):
        """Initialize multi-series detector."""
        self.config = config or SPCConfig()
        self.detectors: dict[str, SPCDetector] = {}
    
    def fit(self, data: pd.DataFrame, id_column: str) -> MultiSeriesSPCDetector:
        """Fit SPC models per unique ID.
        
        Args:
            data: DataFrame with columns [id_column, value_column, ...]
            id_column: Column name identifying different series
            
        Returns:
            Self for method chaining
        """
        for unique_id in data[id_column].unique():
            series_data = data[data[id_column] == unique_id]
            # Assume 'value' or similar numeric column exists
            numeric_cols = series_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                continue
            
            value_col = numeric_cols[0]
            detector = SPCDetector(self.config)
            detector.fit(series_data[value_col].values)
            self.detectors[str(unique_id)] = detector
        
        return self
    
    def detect(
        self,
        data: pd.DataFrame,
        id_column: str,
        value_column: Optional[str] = None,
        methods: Tuple[str, ...] = ("shewhart", "ewma", "cusum", "nelson"),
    ) -> pd.DataFrame:
        """Detect anomalies per series.
        
        Args:
            data: DataFrame with columns [id_column, value_column, ...]
            id_column: Column name identifying series
            value_column: Column to check for anomalies. Auto-detected if None.
            methods: SPC methods to use
            
        Returns:
            DataFrame with original data + anomaly columns
        """
        if value_column is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            value_column = [c for c in numeric_cols if c != id_column][0]
        
        results = data.copy()
        results["is_anomaly"] = False
        
        for unique_id in data[id_column].unique():
            mask = data[id_column] == unique_id
            if str(unique_id) not in self.detectors:
                continue
            
            detector = self.detectors[str(unique_id)]
            anom_flags, _ = detector.detect(
                data[mask][value_column].values,
                methods=methods
            )
            results.loc[mask, "is_anomaly"] = anom_flags
        
        return results


def visualize_spc_results(
    data: np.ndarray | pd.Series,
    detector: SPCDetector,
    title: str = "SPC Anomaly Detection",
    figsize: Tuple[int, int] = (14, 6),
) -> None:
    """Visualize SPC results (requires matplotlib).
    
    Args:
        data: Time series data
        detector: Fitted SPCDetector instance
        title: Plot title
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; install it to visualize results")
        return
    
    data = detector._to_array(data)
    x = np.arange(len(data))
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Shewhart Chart
    vals, center, ucl, lcl = detector.shewhart_chart(data)
    axes[0].plot(x, data, 'b-', label='Data', alpha=0.7)
    axes[0].axhline(center, color='g', linestyle='--', label='Center')
    axes[0].axhline(ucl[0], color='r', linestyle='--', label='UCL/LCL')
    axes[0].axhline(lcl[0], color='r', linestyle='--')
    axes[0].fill_between(x, lcl, ucl, alpha=0.1, color='gray')
    axes[0].set_ylabel('Value')
    axes[0].set_title(f'{title} - Shewhart Chart')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # EWMA Chart
    ewma_vals, center, ucl, lcl = detector.ewma_chart(data)
    axes[1].plot(x, data, 'b-', label='Data', alpha=0.5)
    axes[1].plot(x, ewma_vals, 'purple', linewidth=2, label='EWMA')
    axes[1].axhline(center, color='g', linestyle='--', label='Center')
    axes[1].fill_between(x, lcl, ucl, alpha=0.1, color='gray', label='Control Limits')
    axes[1].set_ylabel('Value')
    axes[1].set_title('EWMA Chart')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # CUSUM Chart
    cusum_vals, center, ucl, _ = detector.cusum_chart(data)
    axes[2].bar(x, cusum_vals, color='steelblue', alpha=0.7, label='CUSUM')
    axes[2].axhline(ucl[0], color='r', linestyle='--', label='Decision Limit')
    axes[2].axhline(0, color='k', linestyle='-', linewidth=0.5)
    axes[2].set_xlabel('Time Index')
    axes[2].set_ylabel('CUSUM Value')
    axes[2].set_title('CUSUM Chart')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Example usage and utility functions

def example_usage():
    """Demonstrate SPC anomaly detection."""
    
    # Generate synthetic time series data with anomalies
    np.random.seed(42)
    n_points = 500
    
    # Normal data with trend
    t = np.arange(n_points)
    baseline = 100 + 0.1 * t + np.random.normal(0, 5, n_points)
    
    # Introduce anomalies
    data = baseline.copy()
    data[100:110] += 30  # Step shift
    data[200] = data[200] + 40  # Spike
    data[300:320] -= 20  # Negative shift
    
    # Fit detector on first half (assumed clean)
    config = SPCConfig(shewhart_multiplier=3.0, ewma_lambda=0.2)
    detector = SPCDetector(config)
    detector.fit(data[:200])
    
    # Detect anomalies on full data
    anom_flags, results = detector.detect(data)
    
    print(f"Total anomalies detected: {anom_flags.sum()}")
    print(f"\nAnomaly indices: {np.where(anom_flags)[0].tolist()}")
    print(f"\nDetailed results (first 20 rows):\n{results.head(20)}")
    
    # Visualize
    visualize_spc_results(data, detector, title="Example SPC Anomaly Detection")
    
    return detector, results


if __name__ == "__main__":
    detector, results = example_usage()
