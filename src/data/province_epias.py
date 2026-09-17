"""EPIAS province-level monthly consumption (Yuzdesel Tuketim Bilgileri) loader.

Source files are one-row-per-month CSV exports from EPIAS Seffaflik for IZMIR and
MANISA, semicolon separated with Turkish number formatting ("1.222.817,98").
They give MWh by customer segment plus the province share of national consumption,
which is the only external ground truth we have at the aggregate level that the
transformer panel (train/test) must sum up to.
"""
from __future__ import annotations

import glob
import os

import polars as pl

SEGMENTS = {
    "Aydınlatma (MWh)": "aydinlatma_mwh",
    "Mesken (MWh)": "mesken_mwh",
    "Sanayi (MWh)": "sanayi_mwh",
    "Tarımsal Sulama (MWh)": "tarimsal_sulama_mwh",
    "Ticarethane (MWh)": "ticarethane_mwh",
    "Diğer (MWh)": "diger_mwh",
    "Genel Toplam (MWh)": "toplam_mwh",
    "İl Bazında Tüketim Oranı (%)": "il_pay_pct",
}


def _tr_num(col: str) -> pl.Expr:
    """Turkish locale number -> float ('1.222.817,98' -> 1222817.98)."""
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.replace_all(r"\.", "", literal=False)
        .str.replace(",", ".", literal=True)
        .replace({"": None, "nan": None})
        .cast(pl.Float64)
    )


def load_province_monthly(data_dir: str = "data") -> pl.DataFrame:
    """Read every izmir/ and manisa/ monthly export into one tidy frame.

    Duplicate downloads ("... (1).csv") collapse to a single row per (il, ay).
    """
    frames = []
    for il, sub in (("İZMİR", "izmir"), ("MANİSA", "manisa")):
        pattern = os.path.join(data_dir, sub, "Yuzdesel_Tuketim_Bilgileri-*.csv")
        for path in sorted(glob.glob(pattern)):
            df = pl.read_csv(path, separator=";", encoding="utf8-lossy", infer_schema=False)
            df = df.rename({c: c.strip() for c in df.columns})
            df = df.with_columns(
                pl.lit(il).alias("il"),
                pl.lit(os.path.basename(path)).alias("source_file"),
            )
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No province exports found under {data_dir}/izmir|manisa")

    raw = pl.concat(frames, how="diagonal_relaxed")
    out = raw.select(
        pl.col("il"),
        pl.col("Dönem").str.strip_chars().str.strptime(pl.Date, "%m.%Y").alias("ay"),
        *[
            (_tr_num(src) if src in raw.columns else pl.lit(None, dtype=pl.Float64)).alias(dst)
            for src, dst in SEGMENTS.items()
        ],
    )

    out = out.sort(["il", "ay"]).unique(subset=["il", "ay"], keep="last").sort(["il", "ay"])
    out = out.with_columns(
        pl.col("ay").dt.year().alias("yil"),
        pl.col("ay").dt.month().alias("ay_no"),
        # Days in month: monthly MWh -> comparable daily rate
        pl.col("ay").dt.month_end().dt.day().alias("gun_sayisi"),
    )
    out = out.with_columns(
        (pl.col("toplam_mwh") / pl.col("gun_sayisi")).alias("gunluk_ort_mwh")
    )
    return out


def build_parquet(data_dir: str = "data", out_path: str = "data/province_monthly.parquet") -> pl.DataFrame:
    df = load_province_monthly(data_dir)
    df.write_parquet(out_path)
    return df


if __name__ == "__main__":
    df = build_parquet()
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(df)
