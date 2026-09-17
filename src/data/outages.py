"""EPİAŞ planlı/plansız kesinti (outage) -> per (ilçe, gün) transformer features.

Source is GDZ_EDAŞ -- the same distribution company behind train/test -- so these
outages are the real cause behind consumption dips in the panel.

Auth is CAS: credentials -> TGT -> service ticket. Credentials come from the
environment, never from source; .env.epias (gitignored) holds them:

    set -a; . ./.env.epias; set +a
    python -m src.data.outages fetch      # 2 provinces x 2 kinds x ~577 days, resumable
    python -m src.data.outages features   # -> outage_daily table

API notes learned the hard way (the published docs disagree):
  * paths end in `-info`, not the `-data` the doc index implies
  * `period` is ONE day, not a range -- hence one call per day
  * province ids are plate code x10 (İZMİR 350, MANİSA 450), resolved at runtime
  * omitting `page` returns the whole day; `total` is null unless a page truncates
  * planned outages are genuinely rare in İzmir/Manisa; unplanned carry the signal
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

BASE = "https://seffaflik.epias.com.tr/electricity-service/v1/consumption/data"
# Verified working host. EPİAŞ has announced that from 2025-12-01 10:00 ticket
# issuance moves to cas.epias.com.tr -- that host rejects this account today, so
# switch via EPIAS_CAS rather than migrating blind.
CAS = os.environ.get("EPIAS_CAS", "https://giris.epias.com.tr/cas/v1/tickets")
DB = "data/transformers.db"

# EPİAŞ provinceId is an internal id (the swagger example is 160), NOT the plate
# code -- so it is resolved by name at runtime rather than guessed.
PROVINCE_NAMES = ("İZMİR", "MANİSA")
KINDS = ("planned", "unplanned")


def _post(url: str, data: bytes, headers: dict[str, str]) -> bytes:
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def get_tgt(username: str, password: str) -> str:
    """Credentials -> TGT, read from the Location header (body carries it too)."""
    body = urllib.parse.urlencode({"username": username, "password": password}).encode()
    req = urllib.request.Request(CAS, data=body, method="POST", headers={
        "Content-Type": "application/x-www-form-urlencoded", "Accept": "text/plain"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode().strip()


def _service_ticket(tgt: str) -> str:
    """TGT -> short-lived service ticket. One per request; they are single-use."""
    body = b"service=https://seffaflik.epias.com.tr"
    hdr = {"Content-Type": "application/x-www-form-urlencoded"}
    return _post(f"{CAS}/{tgt}", body, hdr).decode().strip()


def province_ids(tgt: str) -> dict[str, int]:
    """Resolve İZMİR/MANİSA to EPİAŞ province ids via /v1/main/province-list.

    GET, and the response is a bare JSON array. Ids are the plate code x10
    (İZMİR 350, MANİSA 450) -- resolved rather than hardcoded so a change shows
    up as an error here instead of as silently empty pulls.
    """
    req = urllib.request.Request(
        "https://seffaflik.epias.com.tr/electricity-service/v1/main/province-list",
        headers={"TGT": tgt, "ST": _service_ticket(tgt)},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        items = json.loads(r.read())
    by_name = {_tr_upper(i.get("name", "")): i.get("id") for i in items}
    out = {n: by_name.get(n) for n in PROVINCE_NAMES}
    missing = [n for n, v in out.items() if v is None]
    if missing:
        raise SystemExit(f"province id not found for {missing}; got {sorted(by_name)[:10]}")
    return out


_ST_CACHE: list[str] = []


def fetch_day(tgt: str, kind: str, province_id: int, day: str) -> list[dict]:
    """One (kind, province, day). Omitting `page` returns every row for the day,
    so there is no paging loop -- `total` is only populated when a page truncates.

    CAS service tickets are single-use by spec, but this API accepts one
    repeatedly, so it is cached -- issuing a fresh ST per call doubles the wall
    clock of a 2300-call pull. Any auth failure just re-issues and retries once.
    """
    payload = json.dumps({"period": f"{day}T00:00:00+03:00", "provinceId": province_id}).encode()
    for attempt in (0, 1):
        if not _ST_CACHE:
            _ST_CACHE.append(_service_ticket(tgt))
        try:
            raw = _post(f"{BASE}/{kind}-power-outage-info", payload,
                        {"Content-Type": "application/json", "TGT": tgt, "ST": _ST_CACHE[0]})
            return json.loads(raw).get("items") or []
        except urllib.error.HTTPError as e:
            if attempt or e.code not in (401, 403):
                raise
            _ST_CACHE.clear()  # stale ticket -- re-issue and try once more
    return []


def fetch(start: str = "2025-01-01", end: str = "2026-07-31") -> None:
    tgt = os.environ.get("EPIAS_TGT")
    if not tgt:
        user, pw = os.environ.get("EPIAS_USERNAME"), os.environ.get("EPIAS_PASSWORD")
        if not (user and pw):
            sys.exit("set EPIAS_TGT, or EPIAS_USERNAME+EPIAS_PASSWORD (see .env.epias)")
        tgt = get_tgt(user, pw)
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS outage_raw (
        kind TEXT, il TEXT, tarih TEXT, district TEXT, start_time TEXT, end_time TEXT,
        reason TEXT, neighbourhoods TEXT, subscribers INTEGER, hourly_load REAL,
        outage_id INTEGER, PRIMARY KEY (kind, outage_id))""")
    # A day with zero outages must be remembered too, or every re-run refetches it.
    con.execute("CREATE TABLE IF NOT EXISTS outage_fetched (kind TEXT, il TEXT, tarih TEXT, n INTEGER, PRIMARY KEY (kind, il, tarih))")
    done = {r[:3] for r in con.execute("SELECT kind, il, tarih FROM outage_fetched")}
    provinces = province_ids(tgt)
    print("province ids:", provinces, flush=True)

    for day in pd.date_range(start, end).strftime("%Y-%m-%d"):
        for il, pid in provinces.items():
            for kind in KINDS:
                if (kind, il, day) in done:
                    continue
                try:
                    items = fetch_day(tgt, kind, pid, day)
                except Exception as e:  # one bad day must not lose the whole pull
                    print(f"!! {kind} {il} {day}: {e}", flush=True)
                    time.sleep(5)
                    continue
                con.executemany(
                    "INSERT OR REPLACE INTO outage_raw VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    [(kind, il, day, r.get("district"), r.get("startTime"), r.get("endTime"),
                      r.get("reason"), r.get("effectedNeighbourhoods"), r.get("effectedSubscribers"),
                      r.get("hourlyLoadAvg"), r.get("id")) for r in items],
                )
                con.execute("INSERT OR REPLACE INTO outage_fetched VALUES (?,?,?,?)", (kind, il, day, len(items)))
                con.commit()
                print(f"{kind} {il} {day}: {len(items)}", flush=True)
                time.sleep(0.1)
    con.close()


# ---------------------------------------------------------------- features
def _tr_upper(x: str) -> str:
    """Turkish-aware upper. Both 'İzmir' and 'IZMIR' must land on 'İZMİR', so the
    dotless/dotted I distinction is normalised away (I -> İ) rather than preserved."""
    return x.replace("ı", "i").replace("I", "i").replace("i", "İ").upper().strip()


def _norm(s: pd.Series) -> pd.Series:
    """EPİAŞ district text vs competition ilçe text: upper, strip, drop punctuation."""
    return (s.astype(str).map(_tr_upper)
            .str.replace(r"[^\wÇĞİÖŞÜ ]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip())


def build_features() -> pd.DataFrame:
    """(il, ilce, tarih) -> outage minutes / subscribers / load, planned vs unplanned.

    Duration is clipped to the calendar day so a 3-day outage does not credit
    4320 minutes to its start date.
    """
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM outage_raw", con)
    if df.empty:
        con.close()
        raise SystemExit("outage_raw empty -- run `fetch` first")

    df["ilce"] = _norm(df["district"])
    day = pd.to_datetime(df["tarih"]).dt.tz_localize("Etc/GMT-3")
    st = pd.to_datetime(df["start_time"], format="ISO8601", utc=True).dt.tz_convert("Etc/GMT-3")
    en = pd.to_datetime(df["end_time"], format="ISO8601", utc=True).dt.tz_convert("Etc/GMT-3")
    st = st.clip(lower=day)
    nxt = day + pd.offsets.Day(1)  # offsets, not Timedelta: avoids a numpy unit warning
    en = en.fillna(nxt).clip(upper=nxt)
    df["minutes"] = ((en - st).dt.total_seconds() / 60).clip(lower=0)
    df["subscribers"] = pd.to_numeric(df["subscribers"], errors="coerce").fillna(0)
    df["hourly_load"] = pd.to_numeric(df["hourly_load"], errors="coerce").fillna(0)
    # kWh actually not served: average hourly draw x outage hours.
    df["kwh_lost"] = df["hourly_load"] * df["minutes"] / 60

    # subscriber-weighted exposure: the size of the hole, not just its length
    df["sub_minutes"] = df["subscribers"] * df["minutes"]
    g = (df.groupby(["il", "ilce", "tarih", "kind"])
           .agg(n=("outage_id", "count"), minutes=("minutes", "sum"),
                subscribers=("subscribers", "sum"), kwh_lost=("kwh_lost", "sum"),
                sub_minutes=("sub_minutes", "sum"))
           .reset_index())

    wide = g.pivot_table(index=["il", "ilce", "tarih"], columns="kind",
                         values=["n", "minutes", "subscribers", "kwh_lost", "sub_minutes"],
                         fill_value=0)
    wide.columns = [f"outage_{k}_{c}" for c, k in wide.columns]
    wide = wide.reset_index()
    wide.to_sql("outage_daily", con, if_exists="replace", index=False)
    con.close()
    print("outage_daily:", wide.shape)
    return wide


def attach(panel: pd.DataFrame, db: str = DB) -> pd.DataFrame:
    """Left-join outage_daily onto a train/test panel via lokasyon -> (il, ilce).

    Left join with zero-fill: a district-day absent from the outage table had no
    reported outage, which is a real zero, not a missing value.
    """
    con = sqlite3.connect(db)
    oc = pd.read_sql("SELECT * FROM outage_daily", con)
    con.close()
    # lokasyon is "İL>BÖLGE>İLÇE" for İzmir but "İL>İLÇE" for Manisa, so the ilçe is
    # the LAST segment, never a fixed index -- indexing [2] silently drops all Manisa.
    seg = panel["lokasyon"].str.split(">")
    panel = panel.copy()
    panel["il"] = _norm(seg.str[0])
    panel["ilce"] = _norm(seg.str[-1])
    oc["il"] = _norm(oc["il"])
    out = panel.merge(oc, on=["il", "ilce", "tarih"], how="left")
    cols = [c for c in oc.columns if c.startswith("outage_")]
    out[cols] = out[cols].fillna(0)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "features"
    if cmd == "fetch":
        fetch(*sys.argv[2:])
    else:
        build_features()
