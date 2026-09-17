"""
FastAPI Backend Application for Grid-Up Electricity Explorer.
Provides high-performance SQLite API endpoints for filtering transformers,
retrieving historical actuals, and comparing multi-version submission forecasts.
"""

import os
import sqlite3
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

def format_day_label(date_str: str) -> str:
    """Format YYYY-MM-DD to DD-MM (dow) e.g. 2026-08-24 -> 24-08 (mon)."""
    try:
        dt = datetime.strptime(str(date_str), "%Y-%m-%d")
        return dt.strftime("%d-%m (") + dt.strftime("%a").lower() + ")"
    except Exception:
        return str(date_str)

app = FastAPI(title="Grid-Up Transformer Explorer API")

DB_PATH = "data/transformers.db"

def get_db():
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Database not initialized. Please run build_db.py first.")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Static files directory
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Grid-Up Transformer API is running. Static files located in src/web/static."}

@app.get("/api/filters")
def get_filters():
    conn = get_db()
    cur = conn.cursor()
    
    # Unique provinces (İl)
    cur.execute("SELECT DISTINCT il FROM transformers WHERE il != '' ORDER BY il")
    iller = [row["il"] for row in cur.fetchall()]
    
    # Districts (İlçe) grouped by İl
    cur.execute("SELECT DISTINCT il, ilce FROM transformers WHERE ilce != '' ORDER BY il, ilce")
    ilceler_by_il = {}
    all_ilceler = set()
    for row in cur.fetchall():
        il = row["il"]
        ilce = row["ilce"]
        if il not in ilceler_by_il:
            ilceler_by_il[il] = []
        ilceler_by_il[il].append(ilce)
        all_ilceler.add(ilce)
        
    # Regions (Bölge)
    cur.execute("SELECT DISTINCT bolge FROM transformers WHERE bolge != '' ORDER BY bolge")
    bolgeler = [row["bolge"] for row in cur.fetchall()]
    
    # Submission versions
    cur.execute("SELECT version, filepath, description, is_active FROM submission_versions")
    subs = [dict(row) for row in cur.fetchall()]
    
    # Power ratings summary
    cur.execute("SELECT MIN(guc) as min_guc, MAX(guc) as max_guc FROM transformers")
    guc_row = cur.fetchone()
    
    # Load types
    cur.execute("SELECT DISTINCT load_type_tr FROM transformers WHERE load_type_tr IS NOT NULL AND load_type_tr != '' ORDER BY load_type_tr")
    load_types_tr = [row["load_type_tr"] for row in cur.fetchall()]
    
    conn.close()
    return {
        "iller": iller,
        "ilceler_by_il": ilceler_by_il,
        "all_ilceler": sorted(list(all_ilceler)),
        "bolgeler": bolgeler,
        "statuses": ["All", "Seen", "Cold Start", "Train Only"],
        "load_types": ["All"] + load_types_tr,
        "submissions": subs,
        "guc_range": {"min": guc_row["min_guc"] or 0, "max": guc_row["max_guc"] or 2500}
    }

@app.get("/api/transformers")
def list_transformers(
    il: Optional[str] = None,
    ilce: Optional[str] = None,
    bolge: Optional[str] = None,
    status: Optional[str] = None,
    load_type: Optional[str] = None,
    guc_min: Optional[int] = None,
    guc_max: Optional[int] = None,
    q: Optional[str] = None,
    sort_by: str = "train_mean",
    sort_order: str = "desc",
    limit: int = 50,
    offset: int = 0
):
    try:
        limit_val = int(limit)
    except Exception:
        limit_val = 50
    try:
        offset_val = int(offset)
    except Exception:
        offset_val = 0
        
    conn = get_db()
    cur = conn.cursor()
    
    conditions = []
    params = []
    
    if il and il != "All":
        conditions.append("il = ?")
        params.append(il)
    if ilce and ilce != "All":
        conditions.append("ilce = ?")
        params.append(ilce)
    if bolge and bolge != "All":
        conditions.append("bolge = ?")
        params.append(bolge)
    if status and status != "All":
        conditions.append("status = ?")
        params.append(status)
    if load_type and load_type != "All":
        conditions.append("(load_type = ? OR load_type_tr = ?)")
        params.extend([load_type, load_type])
    if guc_min is not None:
        conditions.append("guc >= ?")
        params.append(int(guc_min))
    if guc_max is not None:
        conditions.append("guc <= ?")
        params.append(int(guc_max))
    if q and str(q).strip():
        search = f"%{str(q).strip()}%"
        conditions.append("(tanim LIKE ? OR lokasyon LIKE ? OR load_type_tr LIKE ?)")
        params.extend([search, search, search])
        
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    
    # Count total matching
    count_query = f"SELECT COUNT(*) as cnt FROM transformers {where_clause}"
    cur.execute(count_query, params)
    total_count = cur.fetchone()["cnt"]
    
    # Allowed sort columns
    allowed_sorts = {
        "tanim": "tanim",
        "guc": "guc",
        "train_mean": "train_mean",
        "pred_mean": "pred_mean",
        "yoy_pct": "yoy_pct",
        "train_max": "train_max",
        "ilce": "ilce",
        "confidence": "confidence"
    }
    col = allowed_sorts.get(sort_by, "train_mean")
    order = "ASC" if str(sort_order).lower() == "asc" else "DESC"
    
    # Nulls last in sorting
    query = f"""
        SELECT tanim, guc, lokasyon, il, bolge, ilce, status,
               load_type, load_type_tr, confidence, load_explanation,
               train_days, train_mean, train_std, train_min, train_max, train_sum,
               pred_mean, pred_max, pred_sum, yoy_pct
        FROM transformers
        {where_clause}
        ORDER BY {col} IS NULL, {col} {order}
        LIMIT ? OFFSET ?
    """
    cur.execute(query, params + [limit_val, offset_val])
    items = [dict(row) for row in cur.fetchall()]
    
    conn.close()
    return {
        "total_count": total_count,
        "limit": limit_val,
        "offset": offset_val,
        "items": items
    }

@app.get("/api/transformer/{tanim}")
def get_transformer_details(
    tanim: str,
    submissions: str = "v22,v18,v14"
):
    conn = get_db()
    cur = conn.cursor()
    
    # 1. Fetch Transformer Metadata
    cur.execute("SELECT * FROM transformers WHERE tanim = ?", (tanim,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Transformer '{tanim}' not found")
    trafo_meta = dict(row)
    
    # 2. Fetch District Weather
    trafo_ilce = trafo_meta.get("ilce", "")
    cur.execute(
        "SELECT tarih, temp_mean, temp_min, temp_max, apparent_temp, humidity, precipitation, day_label "
        "FROM weather WHERE ilce = ? ORDER BY tarih ASC",
        (trafo_ilce,)
    )
    wx_rows = cur.fetchall()
    wx_dict = {r["tarih"]: dict(r) for r in wx_rows}
    weather_series = [dict(r) for r in wx_rows]

    # 3. Fetch Historical Time Series
    cur.execute(
        "SELECT tarih, tuketim FROM historical WHERE tanim = ? ORDER BY tarih ASC",
        (tanim,)
    )
    hist_rows = cur.fetchall()
    historical_series = []
    for r in hist_rows:
        dt = r["tarih"]
        wx_info = wx_dict.get(dt, {})
        historical_series.append({
            "tarih": dt,
            "day_label": wx_info.get("day_label", format_day_label(dt)),
            "tuketim": round(r["tuketim"], 2),
            "temp": wx_info.get("temp_mean")
        })
    
    # 4. Fetch Submission Time Series
    sub_versions = [s.strip() for s in submissions.split(",") if s.strip()]
    
    # Check valid columns
    cur.execute("PRAGMA table_info(submissions)")
    valid_cols = set(col["name"] for col in cur.fetchall())
    selected_cols = [v for v in sub_versions if v in valid_cols]
    if not selected_cols and "v22" in valid_cols:
        selected_cols = ["v22"]
        
    cols_sql = ", ".join([f'"{c}"' for c in selected_cols])
    cur.execute(
        f"SELECT tarih, {cols_sql} FROM submissions WHERE tanim = ? ORDER BY tarih ASC",
        (tanim,)
    )
    sub_rows = cur.fetchall()
    
    predictions_series = {ver: [] for ver in selected_cols}
    for r in sub_rows:
        dt = r["tarih"]
        wx_info = wx_dict.get(dt, {})
        for ver in selected_cols:
            val = r[ver]
            predictions_series[ver].append({
                "tarih": dt,
                "day_label": wx_info.get("day_label", format_day_label(dt)),
                "tuketim": round(val, 2) if val is not None else None,
                "temp": wx_info.get("temp_mean")
            })
            
    # 5. Compute Year-over-Year (YoY) Aligned Comparison
    # Historical Summer: 2025-04-01 to 2025-07-31
    # Submission Summer: 2026-04-01 to 2026-07-31
    hist_dict = {r["tarih"]: r["tuketim"] for r in hist_rows}
    yoy_comparison = []
    
    for r in sub_rows:
        pred_dt = r["tarih"] # e.g. "2026-04-15"
        hist_dt = pred_dt.replace("2026-", "2025-") # e.g. "2025-04-15"
        h_val = hist_dict.get(hist_dt, None)
        
        hist_wx = wx_dict.get(hist_dt, {})
        pred_wx = wx_dict.get(pred_dt, {})
        
        entry = {
            "day_label": pred_wx.get("day_label", format_day_label(pred_dt)), # "15-04 (wed)"
            "hist_date": hist_dt,
            "hist_day_label": hist_wx.get("day_label", format_day_label(hist_dt)), # "15-04 (tue)"
            "hist_val": round(h_val, 2) if h_val is not None else None,
            "hist_temp": hist_wx.get("temp_mean"),
            "pred_date": pred_dt,
            "pred_temp": pred_wx.get("temp_mean")
        }
        for ver in selected_cols:
            val = r[ver]
            entry[ver] = round(val, 2) if val is not None else None
        yoy_comparison.append(entry)
        
    # 6. Monthly Aggregations
    monthly_data = {}
    for r in hist_rows:
        m = r["tarih"][:7] # "YYYY-MM"
        if m not in monthly_data:
            monthly_data[m] = {"hist": []}
        monthly_data[m]["hist"].append(r["tuketim"])
        
    for r in sub_rows:
        m = r["tarih"][:7] # "2026-04", etc.
        if m not in monthly_data:
            monthly_data[m] = {"hist": []}
        for ver in selected_cols:
            if ver not in monthly_data[m]:
                monthly_data[m][ver] = []
            if r[ver] is not None:
                monthly_data[m][ver].append(r[ver])
                
    monthly_temps = {}
    for r in wx_rows:
        m = r["tarih"][:7]
        if m not in monthly_temps:
            monthly_temps[m] = []
        if r["temp_mean"] is not None:
            monthly_temps[m].append(r["temp_mean"])

    monthly_summary = []
    all_months = sorted(set(list(monthly_data.keys()) + list(monthly_temps.keys())))
    for m in all_months:
        m_info = {"month": m}
        if m in monthly_data and monthly_data[m]["hist"]:
            m_info["hist_avg"] = round(sum(monthly_data[m]["hist"]) / len(monthly_data[m]["hist"]), 2)
            m_info["hist_sum"] = round(sum(monthly_data[m]["hist"]), 2)
        else:
            m_info["hist_avg"] = None
            m_info["hist_sum"] = None
            
        for ver in selected_cols:
            vals = monthly_data.get(m, {}).get(ver, [])
            if vals:
                m_info[f"{ver}_avg"] = round(sum(vals) / len(vals), 2)
                m_info[f"{ver}_sum"] = round(sum(vals), 2)
            else:
                m_info[f"{ver}_avg"] = None
                m_info[f"{ver}_sum"] = None
                
        if m in monthly_temps and monthly_temps[m]:
            m_info["temp_avg"] = round(sum(monthly_temps[m]) / len(monthly_temps[m]), 1)
        else:
            m_info["temp_avg"] = None

        monthly_summary.append(m_info)
        
    # 7. Next & Prev Transformer IDs for navigation
    cur.execute("SELECT tanim FROM transformers WHERE ilce = ? ORDER BY train_mean DESC", (trafo_meta["ilce"],))
    sibling_ids = [r["tanim"] for r in cur.fetchall()]
    curr_idx = sibling_ids.index(tanim) if tanim in sibling_ids else -1
    prev_id = sibling_ids[curr_idx - 1] if curr_idx > 0 else (sibling_ids[-1] if sibling_ids else None)
    next_id = sibling_ids[curr_idx + 1] if 0 <= curr_idx < len(sibling_ids) - 1 else (sibling_ids[0] if sibling_ids else None)
    
    conn.close()
    
    return {
        "transformer": trafo_meta,
        "nav": {"prev_id": prev_id, "next_id": next_id, "district_count": len(sibling_ids)},
        "historical": historical_series,
        "predictions": predictions_series,
        "weather": weather_series,
        "selected_versions": selected_cols,
        "yoy": yoy_comparison,
        "monthly": monthly_summary
    }

@app.get("/api/district")
def get_district_aggregate(
    il: str,
    ilce: str,
    submissions: str = "v22,v18,v14"
):
    conn = get_db()
    cur = conn.cursor()
    
    # 1. District info
    cur.execute(
        "SELECT COUNT(*) as trafo_count, "
        "SUM(CASE WHEN status = 'Seen' THEN 1 ELSE 0 END) as seen_trafo_count, "
        "SUM(CASE WHEN status = 'Cold Start' THEN 1 ELSE 0 END) as cold_trafo_count, "
        "SUM(guc) as total_guc, SUM(train_mean) as total_train_mean "
        "FROM transformers WHERE il = ? AND ilce = ?",
        (il, ilce)
    )
    dist_info = dict(cur.fetchone())
    dist_info["il"] = il
    dist_info["ilce"] = ilce
    
    # 2. District Weather
    cur.execute(
        "SELECT tarih, temp_mean, temp_min, temp_max, apparent_temp, humidity, precipitation, day_label "
        "FROM weather WHERE ilce = ? ORDER BY tarih ASC",
        (ilce,)
    )
    dist_wx_rows = cur.fetchall()
    dist_wx_dict = {r["tarih"]: dict(r) for r in dist_wx_rows}
    dist_weather_series = [dict(r) for r in dist_wx_rows]

    # 3. Historical district daily totals
    cur.execute(
        "SELECT tarih, total_tuketim, avg_tuketim, trafo_count, seen_total_tuketim, seen_avg_tuketim, seen_trafo_count "
        "FROM district_history "
        "WHERE il = ? AND ilce = ? ORDER BY tarih ASC",
        (il, ilce)
    )
    raw_hist_rows = cur.fetchall()
    hist_rows = []
    for r in raw_hist_rows:
        dt = r["tarih"]
        wx = dist_wx_dict.get(dt, {})
        hist_rows.append({
            "tarih": dt,
            "day_label": wx.get("day_label", format_day_label(dt)),
            "total_tuketim": r["total_tuketim"],
            "avg_tuketim": r["avg_tuketim"],
            "trafo_count": r["trafo_count"],
            "seen_total_tuketim": r["seen_total_tuketim"] if "seen_total_tuketim" in r.keys() else r["total_tuketim"],
            "seen_avg_tuketim": r["seen_avg_tuketim"] if "seen_avg_tuketim" in r.keys() else r["avg_tuketim"],
            "seen_trafo_count": r["seen_trafo_count"] if "seen_trafo_count" in r.keys() else r["trafo_count"],
            "temp": wx.get("temp_mean")
        })
    
    # 4. Submission district daily totals
    sub_versions = [s.strip() for s in submissions.split(",") if s.strip()]
    cur.execute("PRAGMA table_info(district_submissions)")
    valid_cols = set(col["name"] for col in cur.fetchall())
    selected_cols = [v for v in sub_versions if v in valid_cols]
    if not selected_cols and "v22" in valid_cols:
        selected_cols = ["v22"]
        
    req_cols = ["tarih", "trafo_count", "seen_trafo_count"]
    for ver in selected_cols:
        for suffix in ["", "_avg", "_seen", "_seen_avg"]:
            col_name = f"{ver}{suffix}"
            if col_name in valid_cols:
                req_cols.append(f'"{col_name}"')
    cols_sql = ", ".join(req_cols)

    cur.execute(
        f"SELECT {cols_sql} FROM district_submissions "
        f"WHERE il = ? AND ilce = ? ORDER BY tarih ASC",
        (il, ilce)
    )
    sub_rows = cur.fetchall()
    
    predictions = {ver: [] for ver in selected_cols}
    for r in sub_rows:
        dt = r["tarih"]
        wx = dist_wx_dict.get(dt, {})
        t_count = r["trafo_count"] if "trafo_count" in r.keys() else 0
        s_count = r["seen_trafo_count"] if "seen_trafo_count" in r.keys() else 0
        
        for ver in selected_cols:
            val_tot = r[ver] if ver in r.keys() else None
            val_avg = r[f"{ver}_avg"] if f"{ver}_avg" in r.keys() else (round(val_tot / t_count, 2) if val_tot and t_count > 0 else None)
            val_seen_tot = r[f"{ver}_seen"] if f"{ver}_seen" in r.keys() else None
            val_seen_avg = r[f"{ver}_seen_avg"] if f"{ver}_seen_avg" in r.keys() else (round(val_seen_tot / s_count, 2) if val_seen_tot and s_count > 0 else None)
            
            predictions[ver].append({
                "tarih": dt,
                "day_label": wx.get("day_label", format_day_label(dt)),
                "total_tuketim": round(val_tot, 2) if val_tot is not None else None,
                "avg_tuketim": round(val_avg, 2) if val_avg is not None else None,
                "seen_total_tuketim": round(val_seen_tot, 2) if val_seen_tot is not None else None,
                "seen_avg_tuketim": round(val_seen_avg, 2) if val_seen_avg is not None else None,
                "trafo_count": t_count,
                "seen_trafo_count": s_count,
                "temp": wx.get("temp_mean")
            })
            
    conn.close()
    return {
        "district_info": dist_info,
        "historical": hist_rows,
        "predictions": predictions,
        "weather": dist_weather_series,
        "selected_versions": selected_cols
    }

@app.get("/api/stats")
def get_global_stats():
    conn = get_db()
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) as total FROM transformers")
    total_trafos = cur.fetchone()["total"]
    
    cur.execute("SELECT COUNT(*) as seen FROM transformers WHERE status = 'Seen'")
    seen_count = cur.fetchone()["seen"]
    
    cur.execute("SELECT COUNT(*) as cold FROM transformers WHERE status = 'Cold Start'")
    cold_count = cur.fetchone()["cold"]
    
    cur.execute("SELECT COUNT(*) as train_only FROM transformers WHERE status = 'Train Only'")
    train_only_count = cur.fetchone()["train_only"]
    
    cur.execute("SELECT COUNT(DISTINCT ilce) as dist_count FROM transformers")
    district_count = cur.fetchone()["dist_count"]
    
    cur.execute("SELECT MIN(tarih) as min_t, MAX(tarih) as max_t FROM historical")
    hist_range = cur.fetchone()
    
    cur.execute("SELECT MIN(tarih) as min_t, MAX(tarih) as max_t FROM submissions")
    sub_range = cur.fetchone()
    
    conn.close()
    return {
        "total_transformers": total_trafos,
        "seen_transformers": seen_count,
        "cold_start_transformers": cold_count,
        "train_only_transformers": train_only_count,
        "total_districts": district_count,
        "historical_date_range": f"{hist_range['min_t']} to {hist_range['max_t']}",
        "submission_date_range": f"{sub_range['min_t']} to {sub_range['max_t']}"
    }
