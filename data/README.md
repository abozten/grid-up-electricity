# Dataset Directory (`data/`)

In compliance with datathon terms and data protection regulations, the proprietary distribution grid readings and confidential telemetry are **not included** in this public repository.

You can run the entire pipeline with either:
1. **Synthetic Demo Data**: Automatically generated via `python scripts/generate_sample_data.py`.
2. **Your Own Smart Meter / Grid Data**: Formatted according to the specification below.

---

## Expected Data Schema

### 1. `data/train.csv`
Historical daily electricity consumption readings for training:

| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `tanim` | string | Unique transformer / meter asset identifier | `TR-100234` |
| `guc` | float / int | Nameplate rated capacity (kVA) | `400` |
| `tarih` | string (YYYY-MM-DD) | Date of measurement | `2025-06-15` |
| `tuketim` | float | Daily active electricity consumption (kWh) | `1420.50` |
| `lokasyon` | string | Hierarchical location (`PROVINCE > [REGION >] DISTRICT`) | `İZMİR > METROPOL > KONAK` |

### 2. `data/test.csv`
Evaluation set containing the target horizons to forecast:

| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `tanim` | string | Unique transformer / meter asset identifier | `TR-100234` |
| `guc` | float / int | Nameplate rated capacity (kVA) | `400` |
| `tarih` | string (YYYY-MM-DD) | Target forecast date | `2026-05-01` |
| `lokasyon` | string | Hierarchical location | `İZMİR > METROPOL > KONAK` |

*(Note: `test.csv` may contain transformers not present in `train.csv` — these are handled by the Bayesian cold-start module).*

### 3. `data/ilce_coords.json` (Included)
Geo-coordinates (latitude and longitude) for administrative districts in the İzmir and Manisa distribution territory, utilized by `src/data/fetch_weather.py` to retrieve historical and forecast weather variables from Open-Meteo.

---

## Quick Generation of Sample Data

To populate `data/train.csv` and `data/test.csv` with synthetic data for testing:

```bash
python scripts/generate_sample_data.py
```

