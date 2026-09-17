"""Outage feature checks: day-clipping and the (il, ilce, tarih) join.

No network -- synthetic rows in a temp DB. What breaks if the logic breaks:
a multi-day outage over-crediting minutes, or the lokasyon->ilce join missing.
"""
import sqlite3
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import outages  # noqa: E402


def _seed(db):
    con = sqlite3.connect(db)
    con.execute("""CREATE TABLE outage_raw (kind TEXT, il TEXT, tarih TEXT, district TEXT,
        start_time TEXT, end_time TEXT, reason TEXT, neighbourhoods TEXT,
        subscribers INTEGER, hourly_load REAL, outage_id INTEGER)""")
    rows = [
        # 90 minutes, entirely inside the day
        ("planned", "İZMİR", "2025-06-01", "KARABAĞLAR",
         "2025-06-01T10:00:00+03:00", "2025-06-01T11:30:00+03:00", "bakım", "X", 100, 60.0, 1),
        # starts inside, ends 2 days later -> must clip to 14h (10:00 -> midnight)
        ("unplanned", "İZMİR", "2025-06-01", "KARABAĞLAR",
         "2025-06-01T10:00:00+03:00", "2025-06-03T10:00:00+03:00", "arıza", "Y", 50, 30.0, 2),
    ]
    con.executemany("INSERT INTO outage_raw VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()


def test_clip_and_join(tmp_path):
    db = str(tmp_path / "t.db")
    _seed(db)
    outages.DB = db
    w = outages.build_features()

    r = w.iloc[0]
    assert r["outage_planned_minutes"] == 90, r["outage_planned_minutes"]
    # clipped to the day boundary, not 2880+
    assert r["outage_unplanned_minutes"] == 14 * 60, r["outage_unplanned_minutes"]
    assert r["outage_planned_kwh_lost"] == 90 / 60 * 60.0
    assert r["outage_unplanned_sub_minutes"] == 50 * 14 * 60

    panel = pd.DataFrame({
        "tanim": [1, 2, 3],
        "tarih": ["2025-06-01", "2025-06-02", "2025-06-01"],
        # İzmir is İL>BÖLGE>İLÇE, Manisa is İL>İLÇE -- both must join
        "lokasyon": ["İZMİR>METROPOL>KARABAĞLAR", "İZMİR>METROPOL>KARABAĞLAR",
                     "MANİSA>SALİHLİ"],
    })
    j = outages.attach(panel, db=db)
    assert j.loc[0, "outage_planned_minutes"] == 90
    # a district-day with no outage is a real zero, not NaN
    assert j.loc[1, "outage_planned_minutes"] == 0
    assert j.loc[2, "ilce"] == "SALİHLİ", j.loc[2, "ilce"]
    assert len(j) == len(panel)


def test_tr_upper():
    # dotted/dotless I must collapse, or EPİAŞ spelling never matches ours
    assert outages._tr_upper("İzmir") == outages._tr_upper("IZMIR") == "İZMİR"
    assert outages._tr_upper("manisa") == "MANİSA"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        test_clip_and_join(Path(d))
    test_tr_upper()
    print("ok")
