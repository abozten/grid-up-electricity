"""Configuration module for pipeline paths, feature extraction, and model hyper-parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class PathConfig:
    """Project file paths."""
    root_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = root_dir / "data"
    train_path: Path = data_dir / "train.csv"
    test_path: Path = data_dir / "test.csv"
    weather_path: Path = data_dir / "weather_ilce.csv"
    ilce_coords_path: Path = data_dir / "ilce_coords.json"
    outputs_dir: Path = root_dir / "outputs"
    cache_dir: Path = outputs_dir / "cache"
    wakeup_floor_path: Path = outputs_dir / "wakeup_floor.json"
    submission_path: Path = outputs_dir / "submission.csv"


@dataclass(frozen=True)
class FeatureConfig:
    """Categorical and numerical feature specifications."""
    categorical_cols: Sequence[str] = ("il", "bolge", "ilce")
    
    # Calendar features
    calendar_cols: Sequence[str] = (
        "month", "dow", "doy", "is_weekend",
        "doy_sin", "doy_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
        "is_holiday", "is_ramazan_bayram", "is_kurban_bayram", "is_arefe",
        "is_school_holiday", "days_into_summer_break"
    )
    
    # Raw & derived weather features
    weather_raw: Sequence[str] = (
        "temperature_2m_mean", "temperature_2m_min", "temperature_2m_max",
        "apparent_temperature_mean", "relative_humidity_2m_mean", "precipitation_sum",
        "rain_sum", "cloud_cover_mean", "wind_speed_10m_mean", "wind_speed_10m_max",
        "wind_gusts_10m_max", "surface_pressure_mean", "shortwave_radiation_sum",
        "sunshine_duration", "daylight_duration", "et0_fao_evapotranspiration"
    )
    weather_derived: Sequence[str] = (
        "hdd18", "cdd22", "cdd_log", "sunshine_hours", "daylight_hours", "sunshine_ratio",
        "apparent_delta", "temp_range", "tmean_r7x", "cdd_r7", "hdd_r7", "rad_r7",
        "cdd_acc", "heat_wave_3d", "hot_days_7d", "water_deficit_7d", "water_deficit_14d",
        "water_deficit_30d", "et0_cum_30d", "rain_cum_30d", "days_since_rain",
        "dry_spell_len", "soil_moist_anom", "vpd_r7"
    )
    
    # Historical aggregates per transformer
    history_cols: Sequence[str] = (
        "tan_mean", "tan_median", "tan_std", "tan_min", "tan_max", "tan_zero", "tan_cnt",
        "tan_r7_mean", "tan_r7_zero", "tan_r28_mean", "tan_r28_std", "tan_r28_zero",
        "tan_r91_mean", "tan_r91_std", "tan_r91_zero", "tan_trend", "tan_zero_trend",
        "tan_moy_mean", "tan_dow_mean", "sdly_364", "tan_lf", "exp_log_tan",
        "tan_p10", "tan_p25", "tan_p75", "tan_p90", "tan_iqr", "tan_slope90", "tan_drift_h",
        "tan_cdd_slope", "tan_cddlog_slope", "tan_summer_mean", "tan_summer_delta", "tan_hot_delta"
    )
    
    # Regime shift & structural maneuver features
    regime_cols: Sequence[str] = (
        "level_shift_ratio_60d", "recent_vs_lifetime_log_diff", "recent_load_cv",
        "scaled_summer_baseline", "regime_persistence_days"
    )

    # Seasonal agricultural & irrigation features
    agro_cols: Sequence[str] = (
        "is_summer_exclusive", "summer_concentration_ratio", "agro_summer_expansion_ratio",
        "is_heavy_agro_ilce", "cum_water_deficit", "agro_district_x_et0", "is_delayed_start"
    )

    # Transformer archetype & behavioral features
    archetype_cols: Sequence[str] = (
        "sunday_to_weekday_ratio", "summer_to_winter_ratio", "load_volatility_cv",
        "archetype_code", "archetype_x_cdd"
    )


@dataclass(frozen=True)
class ModelConfig:
    """Model training hyperparameters and ensemble configuration."""
    quantile_alpha: float = 0.55
    lgb_rounds: int = 1000
    cat_iters: int = 800
    learning_rate: float = 0.05
    lgb_num_leaves: int = 95
    lgb_min_data_in_leaf: int = 80
    lgb_feature_fraction: float = 0.8
    lgb_bagging_fraction: float = 0.8
    cat_depth: int = 7
    cat_l2_leaf_reg: float = 3.0
    
    lgb_seeds: Sequence[int] = (42, 7, 2024)
    cat_seeds: Sequence[int] = (42, 7)
    eval_seeds: Sequence[int] = (42, 7)
    
    # Cold start and wake-up settings
    cold_start_alpha: float = 0.8
    cold_start_k: float = 5.0
    wakeup_shrink: float = 0.5
    test_unseen_share: float = 0.2216
    
    # Causal block cuts for temporal CV and multi-block training
    train_blocks: Sequence[tuple[str, str, str]] = (
        ("2025-02-01", "2025-02-01", "2025-04-01"),
        ("2025-04-01", "2025-04-01", "2025-08-01"),
        ("2025-08-01", "2025-08-01", "2025-12-01"),
        ("2025-12-01", "2025-12-01", "2026-04-01"),
    )
    test_cutoff: str = "2026-04-01"


@dataclass(frozen=True)
class Config:
    """Unified application configuration."""
    paths: PathConfig = field(default_factory=PathConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    experiment_name: str = "grid-up-electricity"
