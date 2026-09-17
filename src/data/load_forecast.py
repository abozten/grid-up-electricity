"""EPİAŞ Yük Tahmin Planı (day-ahead national load forecast) -> daily features.

WHY THIS ONE IS CLEAN. Unlike EPİAŞ realtime generation/consumption, the load forecast plan is
PUBLISHED AHEAD of the day it describes. There is no lag(24) question to answer: the value for
2026-07-15 was public before 2026-07-15 began. Nothing here is a backward lag of our target and
nothing is a forward leak.

WHAT IT CAN AND CANNOT DO. It is a single national series -- one value per date, shared by every
transformer in İzmir and Manisa. It therefore cannot separate two transformers on the same day; it
can only shape the common daily profile. That makes it a candidate for the cold-start segment
(date-varying signals reach unseen rows, which static district scalars cannot) and near-useless for
ranking transformers against each other.

Source files are EPİAŞ exports: semicolon-separated, Turkish numerals (dot thousands, comma
decimal), one row per hour.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

YTP_FEATS = ["ytp_mean", "ytp_max", "ytp_min", "ytp_range", "ytp_peak_hour",
             "ytp_evening_ratio", "ytp_night_ratio", "ytp_dev_7", "ytp_dev_28", "ytp_yoy"]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(".", "", regex=False)
                         .str.replace(",", ".", regex=False), errors="coerce")


def load_hourly(paths) -> pd.DataFrame:
    fr = []
    for p in paths:
        d = pd.read_csv(p, sep=";", encoding="utf-8-sig")
        d.columns = [c.strip() for c in d.columns]
        val = [c for c in d.columns if "Yük" in c or "MWh" in c][0]
        d["tarih"] = pd.to_datetime(d["Tarih"], format="%d.%m.%Y")
        d["hour"] = d["Saat"].str.slice(0, 2).astype(int)
        d["ytp"] = _num(d[val])
        fr.append(d[["tarih", "hour", "ytp"]])
    out = pd.concat(fr, ignore_index=True).drop_duplicates(["tarih", "hour"])
    return out.sort_values(["tarih", "hour"]).reset_index(drop=True)


def daily_features(h: pd.DataFrame) -> pd.DataFrame:
    g = h.groupby("tarih").ytp
    d = pd.DataFrame({"ytp_mean": g.mean(), "ytp_max": g.max(), "ytp_min": g.min()})
    d["ytp_range"] = d.ytp_max - d.ytp_min
    d["ytp_peak_hour"] = h.loc[h.groupby("tarih").ytp.idxmax()].set_index("tarih").hour
    tot = g.sum()
    for name, hrs in (("ytp_evening_ratio", range(18, 23)), ("ytp_night_ratio", range(0, 6))):
        d[name] = h[h.hour.isin(hrs)].groupby("tarih").ytp.sum() / tot

    d = d.sort_index()
    # deviation from own trailing level: strictly backward-looking, so it stays causal even though
    # the series itself is known ahead of time
    for w in (7, 28):
        d[f"ytp_dev_{w}"] = d.ytp_mean / d.ytp_mean.rolling(w, min_periods=1).mean().shift(1) - 1
    d["ytp_yoy"] = d.ytp_mean / d.ytp_mean.shift(364) - 1
    return d.reset_index()


def build(paths=None, root: Path | None = None) -> pd.DataFrame:
    root = root or Path(__file__).resolve().parents[2]
    if paths is None:
        # Historically these sat at the repo root. `data/` is where they belong (and what
        # .gitignore covers), so look in both rather than silently finding nothing.
        seen, paths = set(), []
        for pat in ("Yuk_Tahmin_Plani-*.csv", "data/**/Yuk_Tahmin_Plani-*.csv"):
            for p in sorted(root.glob(pat)):
                if p.name not in seen:
                    seen.add(p.name)
                    paths.append(p)
    if not paths:
        raise FileNotFoundError(
            f"no Yuk_Tahmin_Plani-*.csv found under {root} or {root / 'data'}")
    return daily_features(load_hourly(paths))


if __name__ == "__main__":
    d = build()
    print(d.shape, d.tarih.min().date(), "->", d.tarih.max().date())
    print(d.isna().mean()[lambda s: s > 0].to_string() or "no NaNs")
    print(d.head(3).to_string(index=False))
    print(d.tail(3).to_string(index=False))
