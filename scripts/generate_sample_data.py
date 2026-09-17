"""Generate realistic synthetic transformer consumption and weather data for demonstration.

Creates:
    - data/train.csv
    - data/test.csv
    - data/sample_submission.csv
    - data/weather_ilce.csv
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def generate_synthetic_dataset(
    n_seen: int = 50,
    n_cold: int = 20,
    train_days: int = 180,
    test_days: int = 60,
    seed: int = 42,
) -> None:
    np.random.seed(seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    coords_file = DATA_DIR / "ilce_coords.json"
    if coords_file.exists():
        with open(coords_file, "r", encoding="utf-8") as f:
            districts = list(json.load(f).keys())
    else:
        districts = ["KONAK", "BORNOVA", "BUCA", "ÇEŞME", "ALİAĞA", "SALİHLİ", "AKHİSAR"]

    power_ratings = [50, 100, 160, 250, 400, 630, 800, 1000, 1250, 1600]

    # Generate Seen Transformers
    seen_ids = [f"TR-SEEN-{i+1:04d}" for i in range(n_seen)]
    cold_ids = [f"TR-COLD-{i+1:04d}" for i in range(n_cold)]

    transformer_meta = {}
    for tid in seen_ids + cold_ids:
        pwr = int(np.random.choice(power_ratings))
        dst = np.random.choice(districts)
        bolge = "METROPOL" if dst in ["KONAK", "BORNOVA", "BUCA"] else "GÜNEY BÖLGE"
        il = "İZMİR" if dst in ["KONAK", "BORNOVA", "BUCA", "ÇEŞME", "ALİAĞA"] else "MANİSA"
        loc = f"{il} > {bolge} > {dst}"
        base_load_factor = np.random.uniform(0.15, 0.45)
        transformer_meta[tid] = {
            "guc": pwr,
            "lokasyon": loc,
            "ilce": dst,
            "base_kwh": pwr * 24.0 * base_load_factor,
        }

    # Date ranges
    start_train = pd.Timestamp("2025-01-01")
    train_dates = pd.date_range(start_train, periods=train_days, freq="D")
    test_dates = pd.date_range(train_dates[-1] + pd.Timedelta(days=1), periods=test_days, freq="D")

    # Generate Train Data
    train_rows = []
    for dt in train_dates:
        dow = dt.dayofweek
        weekend_mult = 0.85 if dow in [5, 6] else 1.0
        # Seasonal wave
        day_of_year = dt.dayofyear
        season_mult = 1.0 + 0.2 * np.sin(2 * np.pi * day_of_year / 365.25)

        for tid in seen_ids:
            meta = transformer_meta[tid]
            # Baseline with noise
            noise = np.random.lognormal(mean=0.0, sigma=0.15)
            # Occasional drop/spike
            if np.random.random() < 0.02:
                val = 0.0  # standby / outage
            else:
                val = meta["base_kwh"] * weekend_mult * season_mult * noise

            train_rows.append({
                "tanim": tid,
                "guc": meta["guc"],
                "tarih": dt.strftime("%Y-%m-%d"),
                "tuketim": round(float(val), 2),
                "lokasyon": meta["lokasyon"],
            })

    train_df = pd.DataFrame(train_rows)
    train_path = DATA_DIR / "train.csv"
    train_df.to_csv(train_path, index=False)
    print(f"Generated {train_path} ({len(train_df)} rows)")

    # Generate Test Data (Seen + Cold Start)
    test_rows = []
    for dt in test_dates:
        for tid in seen_ids + cold_ids:
            meta = transformer_meta[tid]
            test_rows.append({
                "tanim": tid,
                "guc": meta["guc"],
                "tarih": dt.strftime("%Y-%m-%d"),
                "lokasyon": meta["lokasyon"],
            })

    test_df = pd.DataFrame(test_rows)
    test_path = DATA_DIR / "test.csv"
    test_df.to_csv(test_path, index=False)
    print(f"Generated {test_path} ({len(test_df)} rows)")

    # Sample Submission
    sub_df = test_df[["tanim", "tarih"]].copy()
    sub_df["tuketim"] = 0.0
    sub_path = DATA_DIR / "sample_submission.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"Generated {sub_path} ({len(sub_df)} rows)")

    # Generate synthetic weather data for the active districts
    all_dates = list(train_dates) + list(test_dates)
    weather_rows = []
    used_districts = {transformer_meta[t]["ilce"] for t in transformer_meta}
    for dt in all_dates:
        temp_base = 15.0 + 10.0 * np.sin(2 * np.pi * (dt.dayofyear - 105) / 365.25)
        for dst in used_districts:
            temp = temp_base + np.random.normal(0, 2.0)
            weather_rows.append({
                "tarih": dt.strftime("%Y-%m-%d"),
                "ilce": dst,
                "temperature_2m_mean": round(temp, 1),
                "temperature_2m_min": round(temp - 4.0, 1),
                "temperature_2m_max": round(temp + 5.0, 1),
                "apparent_temperature_mean": round(temp - 0.5, 1),
                "relative_humidity_2m_mean": round(float(np.clip(np.random.normal(65, 10), 30, 95)), 1),
                "precipitation_sum": round(float(np.random.exponential(0.5) if np.random.random() < 0.2 else 0.0), 1),
                "shortwave_radiation_sum": round(float(np.random.uniform(5.0, 25.0)), 2),
                "wind_speed_10m_mean": round(float(np.random.uniform(3.0, 15.0)), 1),
            })

    wx_df = pd.DataFrame(weather_rows)
    wx_path = DATA_DIR / "weather_ilce.csv"
    wx_df.to_csv(wx_path, index=False)
    print(f"Generated {wx_path} ({len(wx_df)} rows)")


if __name__ == "__main__":
    generate_synthetic_dataset()
    print("Synthetic sample data generation complete.")

