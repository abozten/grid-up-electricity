# Grid Up 2026 Electricity Demand Forecasting

[![Competition Result](https://img.shields.io/badge/Competition-2nd%20Place%20Solution-gold.svg)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/ML-LightGBM%20%7C%20CatBoost-brightgreen.svg)]()
[![Backend](https://img.shields.io/badge/Web-FastAPI%20%7C%20SQLite-teal.svg)]()
[![Architecture](https://img.shields.io/badge/Architecture-SOLID%20%26%20Causal-orange.svg)]()
[![Metric: RMSLE](https://img.shields.io/badge/Metric-RMSLE-purple.svg)]()

This repository contains the **2nd Place Solution** for the **Grid Up 2026 Datathon** (Adm-Gdz Elektrik Dağıtım A.Ş.), presenting a high-performance, modular, and leakage-free hierarchical time-series machine learning framework engineered for fleet-wide electrical distribution demand forecasting.

The system forecasts daily active electricity consumption ($\text{kWh}$) across thousands of medium-to-low voltage distribution transformers, featuring an end-to-end pipeline from physical meter data sanitization to Bayesian hierarchical cold-start shrinkage and interactive web exploration.

---

## Key Highlights & Engineering Innovations

### 1. SOLID Architecture & Zero Leakage
- **Single Responsibility & Modular Design**: Decoupled loaders, feature extractors, forecasters, post-processors, and experiment tracking.
- **Strict Causal Walk-Forward Validation**: Purely out-of-fold temporal cross-validation with sliding horizons simulating the real test deployment window, ensuring zero look-ahead contamination.

### 2. Quantile Loss & Heavy-Tail Alignment
- **Log-Space Quantile Regression**: Target variable optimized under $\log(1 + y)$ with an asymmetric pinball loss (`Quantile 0.55`), explicitly mitigating the geometric mean under-prediction bias induced by the log-transformation on heavy-tailed power distributions.
- **Structural Tree Diversity**: Ensembling calibrated LightGBM and CatBoost models across multiple seeds, tree depths, and feature partitions.

### 3. Bayesian Hierarchical Cold-Start Priors
- **The Cold-Start Challenge**: Nearly 30% of target assets have zero historical observations in the training window.
- **Multi-Level Shrinkage**: Unseen transformers are accurately predicted using hierarchical Bayesian shrinkage combining capacity rating ($\text{kVA}$), geographic hierarchy (`Province > Region > District`), external weather indicators (Open-Meteo), and nearest-neighbor load archetype clustering.

### 4. Physical Domain Sanitization & Post-Processing
- **CT/VT Amplification Declipping**: Causal physical heuristic to detect and repair telemetry jump anomalies where meters recorded uncalibrated Current Transformer (CT) or Voltage Transformer (VT) multiplier steps ($5\times, 10\times, 50\times, \dots, 10,000\times$).
- **Wake-Up Load Floor**: Empirical floor threshold preventing seasonal dormant transformers from collapsing to infinitesimal floating-point values when re-energizing.
- **Non-Negative Clamping**: Invariant preservation ensuring physical $\text{kWh} \ge 0$.

### 5. Statistical Process Control (SPC) Anomaly Detection
- Built-in industrial quality control engine featuring **Shewhart control charts**, **EWMA (Exponentially Weighted Moving Average)**, **CUSUM (Cumulative Sum)**, and **Nelson Rules** to detect equipment failures, phase drops, meter anomalies, and sudden structural shifts.

### 6. Interactive Web Explorer
- Modern **FastAPI** backend with responsive **Chart.js** frontend for filtering transformers, visualizing historical consumption against multi-model forecasts, and inspecting load archetype classifications.

---

## Repository Layout

```text
├── src/
│   ├── config.py                 # Central typed configuration (paths, seeds, hyperparams)
│   ├── data/
│   │   ├── loader.py             # Data loading, physical declipping, location parser
│   │   ├── weather.py            # Open-Meteo integration (temperature, CDD/HDD, solar radiation)
│   │   ├── load_forecast.py      # EPİAŞ load forecast plan integration
│   │   └── contract.py           # Provenance assertions & data validation contract
│   ├── features/
│   │   ├── calendar.py           # Temporal cyclic (sin/cos), official & religious holiday flags
│   │   ├── history.py            # Leak-free rolling stats, SDLY lags, load factor
│   │   ├── district.py           # Regional weather response & peer load aggregates
│   │   └── pipeline.py           # Leak-free causal feature assembly engine
│   ├── models/
│   │   ├── base.py               # BaseForecaster abstract base class
│   │   ├── lightgbm_model.py     # Quantile(0.55) LightGBM forecaster
│   │   ├── catboost_model.py     # Quantile(0.55) CatBoost forecaster
│   │   ├── ensemble.py           # Multi-seed Nelder-Mead log-space blender
│   │   ├── cold_start.py         # Bayesian hierarchical shrinkage forecaster
│   │   └── postprocess.py        # Wake-up floors, expm1 inversion, non-negative bounds
│   ├── evaluation/
│   │   ├── metrics.py            # Mathematical RMSLE & weighted error functions
│   │   └── validator.py          # Causal walk-forward cross-validation engine
│   ├── tracking/
│   │   └── tracker.py            # MLflow experiment lifecycle tracker
│   ├── web/
│   │   ├── app.py                # FastAPI dashboard server
│   │   └── static/               # HTML/CSS/JS explorer interface
│   └── spc_anomaly_detection.py  # Statistical Process Control anomaly detector
├── scripts/
│   ├── generate_sample_data.py   # Synthetic demo dataset generator
│   ├── build_data_contract.py    # Schema verification & contract validator
│   ├── classify_load_types.py    # Transformer archetype classifier
│   └── run_folds_parallel.py     # Parallel cross-validation execution
├── tests/                        # Pytest unit & regression tests
├── outputs/                      # Output directory for generated forecasts & evaluation metrics
├── requirements.txt              # Project dependencies
├── run_web.sh                    # One-line web explorer launcher
└── LICENSE                       # MIT License
```

---

## Quick Start

### 1. Environment Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/abozten/grid-up-electricity.git
cd grid-up-electricity
pip install -r requirements.txt
```

*(Optional: If fetching live EPİAŞ grid data, copy `.env.example` to `.env.epias` and provide your credentials).*

```bash
cp .env.example .env.epias
```

### 2. Generate Demo Dataset

To immediately run and test the complete pipeline without proprietary data:

```bash
python scripts/generate_sample_data.py
```
This synthesizes realistic seen and cold-start transformers with calendar seasonality, weather dependencies, and standard schemas in `data/`.

### 3. Run Causal Cross-Validation

Execute strict leak-free temporal walk-forward validation:

```bash
python src/run_pipeline.py --mode cv --experiment-name grid-up-demo
```

### 4. Generate Production Forecasts

Fit the ensemble model on historical observations and generate predictions for the test horizon:

```bash
python src/run_pipeline.py --mode submit --output outputs/submission.csv
```

### 5. Launch Interactive Web Explorer

Start the local interactive visualizer:

```bash
chmod +x run_web.sh
./run_web.sh
```
Open **`http://127.0.0.1:8000`** in your browser to explore transformer time series, filter by district, and inspect model predictions.

---

## Evaluation Metric

Model performance is evaluated using **Root Mean Squared Logarithmic Error (RMSLE)**:

$$\text{RMSLE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left(\log(1 + \hat{y}_i) - \log(1 + y_i)\right)^2}$$

- **Target Optimization**: The models are trained on $z = \log(1 + y)$.
- **Prediction Inversion**: Forecasts are mapped back to $\text{kWh}$ via $\hat{y} = \max(0, \exp(\hat{z}) - 1)$.
- **Quantile Tuning**: Because $\mathbb{E}[\exp(X)] \ge \exp(\mathbb{E}[X])$ by Jensen's inequality, an exact median or mean in log-space exhibits bias in original units; tuning the quantile parameter ($\alpha \in [0.52, 0.58]$) optimizes test-set RMSLE under log-space asymmetry.

---

## Statistical Process Control (SPC)

The repository provides a standalone, production-ready SPC anomaly detector (`src/spc_anomaly_detection.py`) designed for power distribution reliability monitoring:

```python
from spc_anomaly_detection import SPCDetector, SPCConfig

# Initialize detector
config = SPCConfig(shewhart_k=3.0, ewma_lambda=0.2, cusum_k=0.5, cusum_h=4.0)
detector = SPCDetector(config)

# Fit normal operating baseline
detector.fit_baseline(train_series)

# Detect anomalies across test stream
anomalies = detector.detect_anomalies(test_series)
print(f"Detected {anomalies['any_anomaly'].sum()} anomalous intervals")
```
See [`SPC_DOCUMENTATION.md`](SPC_DOCUMENTATION.md) and [`SPC_Anomaly_Detection_Guide.ipynb`](SPC_Anomaly_Detection_Guide.ipynb) for detailed tutorials.

---

## Data Privacy & Compliance

To strictly adhere to competition regulations and protect proprietary grid infrastructure data:
- **Zero Confidential Data**: All proprietary grid telemetry, raw consumption logs, transformer databases, competition test labels, and submission notes detailing the private dataset have been completely stripped from this repository.
- **Clean Model Outputs**: All generated submission CSVs, experimental output caches, and internal tracking ledgers have been removed.
- **Reproducible Pipeline**: Users can generate synthetic demonstration data via `python scripts/generate_sample_data.py` or plug in any standard electrical meter or demand dataset conforming to the schema documented in [`data/README.md`](data/README.md).

---

## License

This project is licensed under the [MIT License](LICENSE).
