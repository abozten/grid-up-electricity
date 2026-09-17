"""Calendar, cyclical, holiday, and school break feature extraction."""
from __future__ import annotations

import numpy as np
import pandas as pd


class CalendarFeatureExtractor:
    """Extracts calendar-based, cyclical, and official/religious holiday indicators."""

    # Fixed calendar dates for Turkish official holidays
    FIXED_HOLIDAYS = {
        (1, 1),    # New Year
        (4, 23),   # National Sovereignty & Children's Day
        (5, 1),    # Labour Day
        (5, 19),   # Commemoration of Atatürk, Youth & Sports Day
        (7, 15),   # Democracy & National Unity Day
        (8, 30),   # Victory Day
        (10, 29),  # Republic Day
    }

    # Specific lunar holiday periods for 2025 and 2026
    RAMAZAN_DATES = {
        # 2025
        "2025-03-30", "2025-03-31", "2025-04-01",
        # 2026
        "2026-03-20", "2026-03-21", "2026-03-22",
    }
    KURBAN_DATES = {
        # 2025
        "2025-06-06", "2025-06-07", "2025-06-08", "2025-06-09",
        # 2026
        "2026-05-27", "2026-05-28", "2026-05-29", "2026-05-30",
    }
    AREFE_DATES = {
        "2025-03-29", "2025-06-05",
        "2026-03-19", "2026-05-26",
    }

    @classmethod
    def transform(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar, cyclical, holiday, and school break features to dataframe."""
        df = df.copy()
        dates = df["tarih"]

        df["month"] = dates.dt.month
        df["dow"] = dates.dt.dayofweek
        df["doy"] = dates.dt.dayofyear
        df["is_weekend"] = (df["dow"] >= 5).astype(int)

        # Cyclical representations
        df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
        df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)

        # Holiday flags
        month_day_tuples = list(zip(df["month"], dates.dt.day))
        is_fixed = np.array([md in cls.FIXED_HOLIDAYS for md in month_day_tuples], dtype=int)
        
        date_strs = dates.dt.strftime("%Y-%m-%d")
        is_ramazan = date_strs.isin(cls.RAMAZAN_DATES).astype(int).values
        is_kurban = date_strs.isin(cls.KURBAN_DATES).astype(int).values
        is_arefe = date_strs.isin(cls.AREFE_DATES).astype(int).values

        df["is_holiday"] = (is_fixed | is_ramazan | is_kurban).astype(int)
        df["is_ramazan_bayram"] = is_ramazan
        df["is_kurban_bayram"] = is_kurban
        df["is_arefe"] = is_arefe

        # School summer vacation (~June 15 to September 15)
        summer_start, summer_end = 166, 258
        is_school = ((df["doy"] >= summer_start) & (df["doy"] <= summer_end)).astype(int)
        df["is_school_holiday"] = is_school
        df["days_into_summer_break"] = np.where(is_school == 1, df["doy"] - summer_start, 0)

        return df
