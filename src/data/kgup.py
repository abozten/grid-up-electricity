"""EPİAŞ Kesinleşmiş Günlük Üretim Planı (day-ahead final generation schedule) -> daily features.

Like the load forecast plan, KGÜP is PUBLISHED AHEAD of the day it describes -- it is a schedule,
not a measurement -- so it carries no lag(24) obligation and is not a backward lag of our target.

WHY IT EARNS ITS PLACE ALONGSIDE ytp. Measured against the GDZ regional daily total (İzmir+Manisa,
455 overlapping days): KGÜP total correlates 0.731, the load forecast 0.622. The two are 0.931
correlated with each other -- close, not interchangeable. The fuel breakdown then adds dimensions
neither total has: hydro 0.283, wind 0.302, geothermal -0.134 against GDZ load.

SHARES, NOT LEVELS. Absolute national MWh partly encodes nationwide installed capacity, which has
nothing to do with İzmir. Fuel SHARES and deviations from a trailing mean are scale-free, so what
transfers to a two-province slice is the shape of the day, not its magnitude.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_forecast import _num

FUELS = ["Doğalgaz", "Rüzgar", "Linyit", "İthal Kömür", "Jeotermal", "Barajlı", "Akarsu", "Gunes"]
KGUP_FEATS = ["kgup_total", "kgup_range", "kgup_peak_hour", "kgup_dev_7", "kgup_dev_28",
              "kgup_sh_gas", "kgup_sh_wind", "kgup_sh_solar", "kgup_sh_hydro",
              "kgup_sh_geo", "kgup_sh_coal", "kgup_sh_renew"]


def load_hourly(paths) -> pd.DataFrame:
    fr = []
    for p in paths:
        d = pd.read_csv(p, sep=";", encoding="utf-8-sig")
        d.columns = [c.strip() for c in d.columns]
        d["tarih"] = pd.to_datetime(d["Tarih"].str.slice(0, 10), format="%d.%m.%Y")
        d["hour"] = d["Saat"].str.slice(0, 2).astype(int)
        for c in ["Toplam(MWh)"] + FUELS:
            if c in d.columns:
                d[c] = _num(d[c])
        fr.append(d[["tarih", "hour", "Toplam(MWh)"] + [f for f in FUELS if f in d.columns]])
    out = pd.concat(fr, ignore_index=True).drop_duplicates(["tarih", "hour"])
    return out.sort_values(["tarih", "hour"]).reset_index(drop=True)


def daily_features(h: pd.DataFrame) -> pd.DataFrame:
    tot = h.groupby("tarih")["Toplam(MWh)"]
    d = pd.DataFrame({"kgup_total": tot.mean()})
    d["kgup_range"] = tot.max() - tot.min()
    d["kgup_peak_hour"] = h.loc[h.groupby("tarih")["Toplam(MWh)"].idxmax()].set_index("tarih").hour

    day = h.groupby("tarih")[FUELS].mean()
    den = d.kgup_total.replace(0, np.nan)
    for key, cols in (("gas", ["Doğalgaz"]), ("wind", ["Rüzgar"]), ("solar", ["Gunes"]),
                      ("hydro", ["Barajlı", "Akarsu"]), ("geo", ["Jeotermal"]),
                      ("coal", ["Linyit", "İthal Kömür"]),
                      ("renew", ["Rüzgar", "Gunes", "Barajlı", "Akarsu", "Jeotermal"])):
        d[f"kgup_sh_{key}"] = day[[c for c in cols if c in day.columns]].sum(axis=1) / den

    d = d.sort_index()
    for w in (7, 28):
        d[f"kgup_dev_{w}"] = d.kgup_total / d.kgup_total.rolling(w, min_periods=1).mean().shift(1) - 1
    return d.reset_index()


def build(paths=None, root: Path | None = None) -> pd.DataFrame:
    root = root or Path(__file__).resolve().parents[2]
    if paths is None:
        # Mirror load_forecast.build: files live under data/kgup/, look there and at the
        # historical root/kgup/ location rather than silently finding nothing.
        seen, paths = set(), []
        for pat in ("kgup/*.csv", "data/kgup/*.csv", "data/**/Kesinlesmis_Gunluk*.csv"):
            for p in sorted(root.glob(pat)):
                if p not in seen:
                    seen.add(p)
                    paths.append(p)
    if not paths:
        raise FileNotFoundError("no kgup/*.csv found")
    return daily_features(load_hourly(paths))


if __name__ == "__main__":
    d = build()
    print(d.shape, d.tarih.min().date(), "->", d.tarih.max().date())
    print(d.isna().mean()[lambda s: s > 0].to_string() or "no NaNs")
    print(d.tail(3).to_string(index=False))
