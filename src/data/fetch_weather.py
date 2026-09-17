"""Regenerate data/ilce_coords.json and data/weather_ilce.csv from Open-Meteo.

The competition download ships only train/test/sample_submission, but the v18 pipeline
(`_archive/legacy/pipeline4.load_wx`) imports `data/weather_ilce.csv` at module scope --
without it v18 cannot even be imported, let alone reproduced. This rebuilds it.

LEAKAGE POSITION. Daily weather is fetched at the SAME timestamp as the target row, including
across the 2026-04-01..07-31 test window. That follows the datathon convention the organisers
state: short-horizon weather forecasts are accurate enough that same-timestamp weather stands in
for a forecast, and no free historical forecast archive exists. No lagged-negative (-1,-2,...,-n)
target-derived column is created anywhere here.
"""
from __future__ import annotations

import json, sys, time, unicodedata
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
COORDS = DATA / "ilce_coords.json"
OUT = DATA / "weather_ilce.csv"

# lead-in before 2025-01-01 so the 7-day rollings are warm on the first training day
START, END = "2024-11-01", "2026-07-31"

DAILY = ["temperature_2m_mean", "temperature_2m_min", "temperature_2m_max",
         "apparent_temperature_mean", "relative_humidity_2m_mean", "precipitation_sum",
         "rain_sum", "cloud_cover_mean", "wind_speed_10m_mean", "wind_speed_10m_max",
         "wind_gusts_10m_max", "surface_pressure_mean", "shortwave_radiation_sum",
         "sunshine_duration", "daylight_duration", "et0_fao_evapotranspiration"]

# Districts whose bare name geocodes to the wrong province or a village of the same name.
# Resolved by hand against the İzmir / Manisa district centres.
MANUAL = {
    "KÖPRÜBAŞI": (38.7422, 28.6297),   # Manisa, not Trabzon Köprübaşı
    "GÖLMARMARA": (38.7167, 27.9000),
    "BEYDAĞ": (38.0847, 28.2114),
    "KINIK": (39.0872, 27.3833),
    "SELENDİ": (38.7419, 28.8628),
    "ŞEHZADELER": (38.6191, 27.4289),
    "YUNUSEMRE": (38.6300, 27.3800),
    "BAYINDIR": (38.2211, 27.6489),
    "BAYRAKLI": (38.4622, 27.1667),
    "KARŞIYAKA": (38.4553, 27.1103),
    "NARLIDERE": (38.3897, 27.0122),
    "TORBALI": (38.1553, 27.3603),
    "KIRKAĞAÇ": (39.1050, 27.6708),
    "SARIGÖL": (38.2333, 28.6944),
    "SARUHANLI": (38.7361, 27.5678),
}


def deaccent(s: str) -> str:
    s = s.replace("İ", "I").replace("ı", "i").replace("Ş", "S").replace("ş", "s")
    s = s.replace("Ğ", "G").replace("ğ", "g").replace("Ç", "C").replace("ç", "c")
    s = s.replace("Ö", "O").replace("ö", "o").replace("Ü", "U").replace("ü", "u")
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def leaf_ilce(lok: pd.Series) -> pd.Series:
    p = lok.str.split(">", expand=True)
    for c in p.columns:
        p[c] = p[c].str.strip()
    n = p.notna().sum(axis=1)
    return p[2].where(n >= 3, p[1]).fillna(p[1]).fillna(p[0])


def districts() -> pd.DataFrame:
    fr = []
    for f in ("train.csv", "test.csv"):
        d = pd.read_csv(DATA / f, usecols=["lokasyon"])
        d["ilce"] = leaf_ilce(d.lokasyon)
        d["il"] = d.lokasyon.str.split(">").str[0].str.strip()
        fr.append(d[["il", "ilce"]])
    return pd.concat(fr).drop_duplicates().sort_values("ilce").reset_index(drop=True)


def geocode(name: str, il: str) -> tuple[float, float] | None:
    if name in MANUAL:
        return MANUAL[name]
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"name": deaccent(name), "count": 20, "language": "tr",
                             "countryCode": "TR"}, timeout=30)
    for hit in r.json().get("results", []):
        if deaccent(hit.get("admin1", "")).upper().startswith(deaccent(il).upper()[:5]):
            return round(hit["latitude"], 4), round(hit["longitude"], 4)
    return None


CACHE = DATA / "_wx_cache"


def fetch_batch(items):
    """One archive call for several districts. Open-Meteo accepts comma-joined coordinate lists
    and returns a JSON array in the same order; batching is what keeps us under the rate limit."""
    lats = ",".join(str(c["lat"]) for _, c in items)
    lons = ",".join(str(c["lon"]) for _, c in items)
    for attempt in range(6):
        r = requests.get("https://archive-api.open-meteo.com/v1/archive",
                         params={"latitude": lats, "longitude": lons, "start_date": START,
                                 "end_date": END, "daily": ",".join(DAILY),
                                 "timezone": "Europe/Istanbul"}, timeout=180)
        if r.status_code == 200:
            js = r.json()
            if isinstance(js, dict):
                js = [js]
            out = []
            for (name, _), block in zip(items, js):
                w = pd.DataFrame(block["daily"]).rename(columns={"time": "tarih"})
                w.insert(0, "ilce", name)
                out.append(w)
            return out
        wait = 20 * (attempt + 1)
        print(f"    {r.status_code}, retrying in {wait}s", flush=True)
        time.sleep(wait)
    r.raise_for_status()


if __name__ == "__main__":
    dd = districts()
    print(f"{len(dd)} districts", flush=True)

    coords = json.loads(COORDS.read_text()) if COORDS.exists() else {}
    for _, row in dd.iterrows():
        if row.ilce in coords:
            continue
        hit = geocode(row.ilce, row.il)
        if hit is None:
            print(f"  UNRESOLVED {row.il} > {row.ilce}", flush=True)
            continue
        coords[row.ilce] = {"il": row.il, "lat": hit[0], "lon": hit[1]}
        time.sleep(0.4)
    COORDS.write_text(json.dumps(coords, ensure_ascii=False, indent=1))
    missing = sorted(set(dd.ilce) - set(coords))
    print(f"geocoded {len(coords)}/{len(dd)}" + (f"  MISSING {missing}" if missing else ""), flush=True)
    if missing:
        sys.exit("refusing to build a weather file with holes")

    CACHE.mkdir(exist_ok=True)
    frames, pending = [], []
    for name, c in sorted(coords.items()):
        f = CACHE / f"{deaccent(name)}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
        else:
            pending.append((name, c))
    print(f"cached {len(frames)}, fetching {len(pending)}", flush=True)

    B = 6
    for i in range(0, len(pending), B):
        chunk = pending[i:i + B]
        for w in fetch_batch(chunk):
            w.to_parquet(CACHE / f"{deaccent(w.ilce.iloc[0])}.parquet", index=False)
            frames.append(w)
        print(f"  [{min(i+B,len(pending)):2d}/{len(pending)}] " +
              ", ".join(n for n, _ in chunk), flush=True)
        time.sleep(12)

    wx = pd.concat(frames, ignore_index=True)
    wx["tarih"] = pd.to_datetime(wx.tarih)
    wx = wx.sort_values(["ilce", "tarih"]).reset_index(drop=True)
    na = wx[DAILY].isna().mean()
    print("\nNaN share by column:\n" + na[na > 0].to_string() if (na > 0).any() else "\nno NaNs")
    wx.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  {wx.shape}  {wx.tarih.min().date()}..{wx.tarih.max().date()}", flush=True)
