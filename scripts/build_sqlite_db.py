"""
Comprehensive ETL script to write ALL repository data to an indexed SQLite database.

Ingests:
1. Competition Data:
   - data/train.csv (1.22M rows historical consumption with declipping & parsed location)
   - data/test.csv (714K rows test horizon metadata)
   - data/sample_submission.csv
   - data/ilce_coords.json (District coordinates)
   - Transformer master metadata & precomputed stats

2. Weather & Environmental Data:
   - data/weather_ilce.csv (27K daily weather records per district with all meteorological features)

3. EPİAŞ External Grid Data:
   - data/epias-data/Gercek_Zamanli_Tuketim-*.csv (2015-2026 hourly national consumption)
   - data/epias-data/Gercek_Zamanli_Uretim-*.csv (2014-2026 hourly national generation by fuel type)
   - data/kgup/Kesinlesmis_Gunluk_Uretim_Plani_*.csv (Hourly day-ahead generation plans)
   - data/yuk-tahmin-plani/Yuk_Tahmin_Plani-*.csv (Hourly day-ahead load forecasts)
   - data/not-used-izmir-manisa-epias/*/*.csv (İzmir & Manisa monthly sectoral consumption)

4. Model Forecasts & Submissions:
   - outputs/submission_*.csv & outputs/v34_*.csv (v33, v24, v23, v34 variants, etc.)
   - submission_versions metadata table

5. Precomputed District Aggregates:
   - district_history (Historical daily totals, averages, transformer counts: all & seen)
   - district_submissions (Forecast daily totals, averages, transformer counts: all & seen)

6. Experiment & Benchmark Artifacts:
   - outputs/*.json (All 133 CV evaluation, benchmark, ablation JSON files)

7. Catalog & High-Speed Indexes:
   - table_catalog (Dynamic table directory with row counts and source files)
   - Complete B-tree indexing on all primary and filtering columns
"""

from __future__ import annotations

import glob
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

# Add repository root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import DataLoader


def parse_lokasyon(df: pd.DataFrame) -> pd.DataFrame:
    """Parse 'lokasyon' column into il, bolge, ilce hierarchy."""
    def parse_one(val: Any) -> Tuple[str, str, str]:
        parts = str(val).split('>')
        if len(parts) >= 3:
            return parts[0].strip(), parts[1].strip(), parts[2].strip()
        elif len(parts) == 2:
            return parts[0].strip(), parts[0].strip(), parts[1].strip()
        elif len(parts) == 1:
            return parts[0].strip(), '', parts[0].strip()
        return '', '', ''

    parsed = df['lokasyon'].apply(parse_one)
    return pd.DataFrame(parsed.tolist(), columns=['il', 'bolge', 'ilce'], index=df.index)


def clean_tr_num(series: pd.Series) -> pd.Series:
    """Convert Turkish locale numeric string ('1.234,56') to float (1234.56)."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
        .str.strip(),
        errors='coerce'
    )


def parse_tr_date(series: pd.Series) -> pd.Series:
    """Standardize Turkish dates ('DD.MM.YYYY' or 'DD.MM.YYYY HH:MM') to 'YYYY-MM-DD'."""
    cleaned = series.astype(str).str.split(' ').str[0].str.strip()
    return pd.to_datetime(cleaned, format='%d.%m.%Y', errors='coerce').dt.strftime('%Y-%m-%d')


def build_all_sqlite_database(db_path: str = "data/transformers.db") -> None:
    t_start = time.time()
    db_file = ROOT / db_path
    print("=" * 75)
    print(f"BUILDING COMPREHENSIVE SQLITE DATABASE: {db_file}")
    print("=" * 75)

    if db_file.exists():
        os.remove(db_file)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("PRAGMA synchronous = OFF;")
    cur.fetchall()
    cur.execute("PRAGMA journal_mode = MEMORY;")
    cur.fetchall()
    cur.close()

    table_meta: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # 1. Ingest Competition Train Data & declip measurement artifacts
    # -------------------------------------------------------------------------
    print("\n[1/12] Ingesting 'train.csv' and declipping measurement artifacts...")
    train_path = ROOT / "data/train.csv"
    train = pd.read_csv(train_path, dtype={'tanim': str})
    train = DataLoader.declip(train)
    train['tarih'] = train['tarih'].astype(str)
    train_geo = parse_lokasyon(train)
    train['il'] = train_geo['il']
    train['bolge'] = train_geo['bolge']
    train['ilce'] = train_geo['ilce']

    train_slim = train[['tanim', 'tarih', 'tuketim', 'guc', 'lokasyon', 'il', 'bolge', 'ilce']].copy()
    train_slim.to_sql('historical', conn, if_exists='replace', index=False)
    table_meta.append({
        'table_name': 'historical',
        'row_count': len(train_slim),
        'column_count': len(train_slim.columns),
        'description': 'Declipped historical daily electricity consumption (2025-01-01 to 2026-03-31)',
        'source_files': 'data/train.csv'
    })
    print(f"  -> Wrote 'historical' ({len(train_slim):,} rows)")

    # -------------------------------------------------------------------------
    # 2. Ingest Test Data & Sample Submission
    # -------------------------------------------------------------------------
    print("\n[2/12] Ingesting 'test.csv' and 'sample_submission.csv'...")
    test_path = ROOT / "data/test.csv"
    test = pd.read_csv(test_path, dtype={'tanim': str})
    test['tarih'] = test['tarih'].astype(str)
    test_geo = parse_lokasyon(test)
    test['il'] = test_geo['il']
    test['bolge'] = test_geo['bolge']
    test['ilce'] = test_geo['ilce']

    test_slim = test[['tanim', 'tarih', 'guc', 'lokasyon', 'il', 'bolge', 'ilce']].copy()
    test_slim.to_sql('test_metadata', conn, if_exists='replace', index=False)
    table_meta.append({
        'table_name': 'test_metadata',
        'row_count': len(test_slim),
        'column_count': len(test_slim.columns),
        'description': 'Test set metadata and calendar horizon (2026-04-01 to 2026-07-31)',
        'source_files': 'data/test.csv'
    })
    print(f"  -> Wrote 'test_metadata' ({len(test_slim):,} rows)")

    sample_sub_path = ROOT / "data/sample_submission.csv"
    if sample_sub_path.exists():
        sample_sub = pd.read_csv(sample_sub_path)
        if 'tarih' not in sample_sub.columns and 'id' in sample_sub.columns:
            parts = sample_sub['id'].astype(str).str.split('_', expand=True)
            sample_sub['tanim'] = parts[0]
            sample_sub['tarih'] = parts[1]
        if 'tarih' in sample_sub.columns:
            sample_sub['tarih'] = sample_sub['tarih'].astype(str)
        if 'tanim' in sample_sub.columns:
            sample_sub['tanim'] = sample_sub['tanim'].astype(str)
        sample_sub.to_sql('sample_submission', conn, if_exists='replace', index=False)
        table_meta.append({
            'table_name': 'sample_submission',
            'row_count': len(sample_sub),
            'column_count': len(sample_sub.columns),
            'description': 'Official sample submission template',
            'source_files': 'data/sample_submission.csv'
        })
        print(f"  -> Wrote 'sample_submission' ({len(sample_sub):,} rows)")

    # Ingest transformer load classification if present
    load_cls_path = ROOT / "outputs/transformer_load_classification.csv"
    if load_cls_path.exists():
        df_load_cls = pd.read_csv(load_cls_path, dtype={'tanim': str})
        df_load_cls.to_sql('transformer_load_classification', conn, if_exists='replace', index=False)
        table_meta.append({
            'table_name': 'transformer_load_classification',
            'row_count': len(df_load_cls),
            'column_count': len(df_load_cls.columns),
            'description': 'Transformer archetype and load behavior classification (residential, agro, industrial, solar, inactive)',
            'source_files': 'outputs/transformer_load_classification.csv'
        })
        print(f"  -> Wrote 'transformer_load_classification' ({len(df_load_cls):,} rows)")

    # -------------------------------------------------------------------------
    # 3. Ingest Submissions & Models
    # -------------------------------------------------------------------------
    print("\n[3/12] Ingesting submission forecasts from 'outputs/'...")
    submission_configs = [
        ("v33", "outputs/submission_v33.csv", "v33: Causal LGBM + Isotonic Cohort Prior (Active PB, LB 1.03376)"),
        ("v24", "outputs/submission_v24.csv", "v24: Classifiers + Two-Stage Hurdle"),
        ("v23", "outputs/submission_v23.csv", "v23: Regime + Agro + Archetype (LB 1.03608)"),
        ("v34_m1_c4_bucket", "outputs/v34_m1_c4_bucket.csv", "v34: M1 Base + C4 Bucket Candidate"),
        ("v34_m1_c4_hard", "outputs/v34_m1_c4_hard.csv", "v34: M1 Base + C4 Hard Mask Candidate"),
        ("v34_m1_c4_r2", "outputs/v34_m1_c4_r2.csv", "v34: M1 Base + C4 R2 Candidate"),
    ]
    # Check any extra CSVs in outputs
    for extra_csv in sorted(ROOT.glob("outputs/*.csv")):
        if extra_csv.name == "transformer_load_classification.csv":
            continue
        rel_p = extra_csv.relative_to(ROOT).as_posix()
        if not any(rel_p == cfg[1] for cfg in submission_configs):
            vkey = extra_csv.stem.replace("submission_", "")
            submission_configs.append((vkey, rel_p, f"Forecast output: {extra_csv.name}"))

    available_subs = []
    sub_data = {}
    test_wide = test[['tanim', 'tarih']].copy()
    sub_ver_rows = []

    for ver_key, rel_path, desc in submission_configs:
        full_p = ROOT / rel_path
        if full_p.exists():
            df_sub = pd.read_csv(full_p)
            if 'tuketim' in df_sub.columns and len(df_sub) == len(test):
                print(f"  Loading {ver_key}: {rel_path}")
                vals = df_sub['tuketim'].values
                sub_data[ver_key] = vals
                test_wide[ver_key] = vals
                available_subs.append((ver_key, rel_path, desc))
                sub_ver_rows.append({
                    'version': ver_key,
                    'filepath': rel_path,
                    'description': desc,
                    'is_active': 1 if ver_key == 'v33' else 0,
                    'row_count': len(vals),
                    'mean_pred': round(float(np.mean(vals)), 4),
                    'max_pred': round(float(np.max(vals)), 4),
                    'sum_pred': round(float(np.sum(vals)), 2)
                })

    test_wide.to_sql('submissions', conn, if_exists='replace', index=False)
    table_meta.append({
        'table_name': 'submissions',
        'row_count': len(test_wide),
        'column_count': len(test_wide.columns),
        'description': 'Test horizon forecasts with columns for each submission version',
        'source_files': ', '.join(s[1] for s in available_subs)
    })

    df_sub_vers = pd.DataFrame(sub_ver_rows)
    df_sub_vers.to_sql('submission_versions', conn, if_exists='replace', index=False)
    table_meta.append({
        'table_name': 'submission_versions',
        'row_count': len(df_sub_vers),
        'column_count': len(df_sub_vers.columns),
        'description': 'Metadata directory of all available submission forecast versions',
        'source_files': 'outputs/*.csv'
    })
    print(f"  -> Wrote 'submissions' ({len(test_wide):,} rows) & 'submission_versions' ({len(df_sub_vers)} rows)")

    # -------------------------------------------------------------------------
    # 4. Ingest Transformer Master Metadata Table
    # -------------------------------------------------------------------------
    print("\n[4/12] Building 'transformers' master table...")
    train_trafos = train[['tanim', 'guc', 'lokasyon', 'il', 'bolge', 'ilce']].drop_duplicates(subset=['tanim'])
    test_trafos = test[['tanim', 'guc', 'lokasyon', 'il', 'bolge', 'ilce']].drop_duplicates(subset=['tanim'])
    train_ids = set(train_trafos['tanim'])
    test_ids = set(test_trafos['tanim'])
    all_ids = sorted(list(train_ids.union(test_ids)))

    meta_dict = {}
    for _, row in test_trafos.iterrows():
        meta_dict[str(row['tanim'])] = {
            'tanim': str(row['tanim']),
            'guc': int(row['guc']) if pd.notnull(row['guc']) else 0,
            'lokasyon': str(row['lokasyon']),
            'il': str(row['il']),
            'bolge': str(row['bolge']),
            'ilce': str(row['ilce']),
            'status': 'Seen' if row['tanim'] in train_ids else 'Cold Start'
        }
    for _, row in train_trafos.iterrows():
        tid = str(row['tanim'])
        if tid not in meta_dict:
            meta_dict[tid] = {
                'tanim': tid,
                'guc': int(row['guc']) if pd.notnull(row['guc']) else 0,
                'lokasyon': str(row['lokasyon']),
                'il': str(row['il']),
                'bolge': str(row['bolge']),
                'ilce': str(row['ilce']),
                'status': 'Train Only'
            }

    hist_stats = train.groupby('tanim')['tuketim'].agg(
        train_days='count',
        train_mean='mean',
        train_std='std',
        train_min='min',
        train_max='max',
        train_sum='sum'
    ).reset_index()
    hist_stats_dict = hist_stats.set_index('tanim').to_dict(orient='index')

    main_ver = 'v33' if 'v33' in test_wide.columns else available_subs[0][0]
    sub_stats = test_wide.groupby('tanim')[main_ver].agg(
        pred_mean='mean',
        pred_max='max',
        pred_sum='sum'
    ).reset_index()
    sub_stats_dict = sub_stats.set_index('tanim').to_dict(orient='index')

    trafo_rows = []
    for tid in all_ids:
        tid_str = str(tid)
        info = meta_dict[tid_str]
        h_stat = hist_stats_dict.get(tid_str, {})
        s_stat = sub_stats_dict.get(tid_str, {})

        t_mean = h_stat.get('train_mean', None)
        p_mean = s_stat.get('pred_mean', None)
        yoy_pct = None
        if t_mean is not None and p_mean is not None and t_mean > 0:
            yoy_pct = round(((p_mean - t_mean) / t_mean) * 100, 2)

        trafo_rows.append({
            'tanim': tid_str,
            'guc': info['guc'],
            'lokasyon': info['lokasyon'],
            'il': info['il'],
            'bolge': info['bolge'],
            'ilce': info['ilce'],
            'status': info['status'],
            'train_days': h_stat.get('train_days', 0),
            'train_mean': round(float(t_mean), 2) if t_mean is not None else None,
            'train_std': round(float(h_stat.get('train_std', 0)), 2) if pd.notnull(h_stat.get('train_std')) else None,
            'train_min': round(float(h_stat.get('train_min', 0)), 2) if pd.notnull(h_stat.get('train_min')) else None,
            'train_max': round(float(h_stat.get('train_max', 0)), 2) if pd.notnull(h_stat.get('train_max')) else None,
            'train_sum': round(float(h_stat.get('train_sum', 0)), 2) if pd.notnull(h_stat.get('train_sum')) else None,
            'pred_mean': round(float(p_mean), 2) if p_mean is not None else None,
            'pred_max': round(float(s_stat.get('pred_max', 0)), 2) if pd.notnull(s_stat.get('pred_max')) else None,
            'pred_sum': round(float(s_stat.get('pred_sum', 0)), 2) if pd.notnull(s_stat.get('pred_sum')) else None,
            'yoy_pct': yoy_pct
        })

    df_trafos = pd.DataFrame(trafo_rows)
    df_trafos.to_sql('transformers', conn, if_exists='replace', index=False)
    table_meta.append({
        'table_name': 'transformers',
        'row_count': len(df_trafos),
        'column_count': len(df_trafos.columns),
        'description': 'Master transformer directory with ratings, location, seen/cold status, and historical/forecast summary statistics',
        'source_files': 'data/train.csv, data/test.csv, outputs/*.csv'
    })
    print(f"  -> Wrote 'transformers' ({len(df_trafos):,} rows)")

    # -------------------------------------------------------------------------
    # 5. Ingest Weather Data & Coordinates
    # -------------------------------------------------------------------------
    print("\n[5/12] Ingesting 'weather_ilce.csv' & 'ilce_coords.json'...")
    wx_path = ROOT / "data/weather_ilce.csv"
    if wx_path.exists():
        wx = pd.read_csv(wx_path)
        wx['tarih'] = wx['tarih'].astype(str)
        dt_series = pd.to_datetime(wx['tarih'])
        wx['day_label'] = dt_series.dt.strftime('%d-%m (') + dt_series.dt.strftime('%a').str.lower() + ')'
        wx['temp_mean'] = wx['temperature_2m_mean'].round(2)
        wx['temp_min'] = wx['temperature_2m_min'].round(2)
        wx['temp_max'] = wx['temperature_2m_max'].round(2)
        wx['apparent_temp'] = wx['apparent_temperature_mean'].round(2)
        wx['humidity'] = wx['relative_humidity_2m_mean'].round(1)
        wx['precipitation'] = wx['precipitation_sum'].round(2)

        wx.to_sql('weather', conn, if_exists='replace', index=False)
        table_meta.append({
            'table_name': 'weather',
            'row_count': len(wx),
            'column_count': len(wx.columns),
            'description': 'District daily meteorological records: temperatures, humidity, solar radiation, wind, and precipitation',
            'source_files': 'data/weather_ilce.csv'
        })
        print(f"  -> Wrote 'weather' ({len(wx):,} rows)")

    coords_path = ROOT / "data/ilce_coords.json"
    if coords_path.exists():
        with open(coords_path) as fp:
            coords_dict = json.load(fp)
        coords_rows = [{'ilce': k, 'latitude': v[0], 'longitude': v[1]} for k, v in coords_dict.items()]
        df_coords = pd.DataFrame(coords_rows)
        df_coords.to_sql('ilce_coordinates', conn, if_exists='replace', index=False)
        table_meta.append({
            'table_name': 'ilce_coordinates',
            'row_count': len(df_coords),
            'column_count': len(df_coords.columns),
            'description': 'Geographic latitude and longitude coordinates for each district',
            'source_files': 'data/ilce_coords.json'
        })
        print(f"  -> Wrote 'ilce_coordinates' ({len(df_coords):,} rows)")

    # -------------------------------------------------------------------------
    # 6. Precompute District Daily Aggregates
    # -------------------------------------------------------------------------
    print("\n[6/12] Precomputing district daily aggregates (all & seen-only)...")
    seen_ids = set(train_ids).intersection(set(test_ids))
    train['is_seen'] = train['tanim'].isin(seen_ids)
    test['is_seen'] = test['tanim'].isin(seen_ids)

    # Historical district daily (all)
    train_dist_all = train.groupby(['il', 'ilce', 'tarih'])['tuketim'].agg(
        total_tuketim='sum',
        avg_tuketim='mean',
        trafo_count='count'
    ).reset_index()

    # Historical district daily (seen only)
    train_dist_seen = train[train['is_seen']].groupby(['il', 'ilce', 'tarih'])['tuketim'].agg(
        seen_total_tuketim='sum',
        seen_avg_tuketim='mean',
        seen_trafo_count='count'
    ).reset_index()

    train_dist = train_dist_all.merge(train_dist_seen, on=['il', 'ilce', 'tarih'], how='left')
    train_dist['seen_trafo_count'] = train_dist['seen_trafo_count'].fillna(0).astype(int)
    train_dist['seen_total_tuketim'] = train_dist['seen_total_tuketim'].fillna(0.0).round(2)
    train_dist['seen_avg_tuketim'] = train_dist['seen_avg_tuketim'].fillna(0.0).round(2)
    train_dist['total_tuketim'] = train_dist['total_tuketim'].round(2)
    train_dist['avg_tuketim'] = train_dist['avg_tuketim'].round(2)
    train_dist.to_sql('district_history', conn, if_exists='replace', index=False)
    table_meta.append({
        'table_name': 'district_history',
        'row_count': len(train_dist),
        'column_count': len(train_dist.columns),
        'description': 'Historical daily aggregated district consumption (total & seen-only)',
        'source_files': 'Derived from historical'
    })

    # Test district daily for submission versions
    test_merged = test.merge(test_wide, on=['tanim', 'tarih'])
    sub_cols = [v[0] for v in available_subs]

    agg_all = {col: 'sum' for col in sub_cols}
    agg_all['tanim'] = 'count'
    test_dist_all = test_merged.groupby(['il', 'ilce', 'tarih']).agg(agg_all).reset_index().rename(columns={'tanim': 'trafo_count'})

    agg_seen = {col: 'sum' for col in sub_cols}
    agg_seen['tanim'] = 'count'
    test_dist_seen = test_merged[test_merged['is_seen']].groupby(['il', 'ilce', 'tarih']).agg(agg_seen).reset_index().rename(columns={'tanim': 'seen_trafo_count'})
    for col in sub_cols:
        test_dist_seen = test_dist_seen.rename(columns={col: f'{col}_seen'})

    test_dist = test_dist_all.merge(test_dist_seen, on=['il', 'ilce', 'tarih'], how='left')
    test_dist['seen_trafo_count'] = test_dist['seen_trafo_count'].fillna(0).astype(int)
    for col in sub_cols:
        test_dist[col] = test_dist[col].round(2)
        test_dist[f'{col}_seen'] = test_dist[f'{col}_seen'].fillna(0.0).round(2)
        test_dist[f'{col}_avg'] = (test_dist[col] / test_dist['trafo_count']).round(2)
        test_dist[f'{col}_seen_avg'] = np.where(
            test_dist['seen_trafo_count'] > 0,
            (test_dist[f'{col}_seen'] / test_dist['seen_trafo_count']).round(2),
            0.0
        )
    test_dist.to_sql('district_submissions', conn, if_exists='replace', index=False)
    table_meta.append({
        'table_name': 'district_submissions',
        'row_count': len(test_dist),
        'column_count': len(test_dist.columns),
        'description': 'Forecast daily aggregated district consumption across model versions (total & seen-only)',
        'source_files': 'Derived from submissions & test'
    })
    print(f"  -> Wrote 'district_history' ({len(train_dist):,} rows) & 'district_submissions' ({len(test_dist):,} rows)")

    # -------------------------------------------------------------------------
    # 7. Ingest EPİAŞ Realtime Hourly Consumption (Gercek_Zamanli_Tuketim)
    # -------------------------------------------------------------------------
    print("\n[7/12] Ingesting EPİAŞ Real-Time Hourly Consumption (data/epias-data/Gercek_Zamanli_Tuketim-*.csv)...")
    tuketim_files = sorted(ROOT.glob("data/epias-data/Gercek_Zamanli_Tuketim-*.csv"))
    t_dfs = []
    for f in tuketim_files:
        df = pd.read_csv(f, sep=None, engine='python')
        df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]
        t_df = pd.DataFrame({
            'tarih': parse_tr_date(df['Tarih']),
            'saat': df['Saat'].astype(str).str.strip(),
            'tuketim_mwh': clean_tr_num(df['Tüketim Miktarı(MWh)'])
        })
        t_dfs.append(t_df)
    if t_dfs:
        all_t = pd.concat(t_dfs).drop_duplicates(subset=['tarih', 'saat']).sort_values(['tarih', 'saat'])
        all_t.to_sql('epias_tuketim_hourly', conn, if_exists='replace', index=False)
        table_meta.append({
            'table_name': 'epias_tuketim_hourly',
            'row_count': len(all_t),
            'column_count': len(all_t.columns),
            'description': 'EPİAŞ national real-time hourly electricity consumption in MWh (2015 to 2026)',
            'source_files': f'data/epias-data/Gercek_Zamanli_Tuketim-*.csv ({len(tuketim_files)} files)'
        })
        print(f"  -> Wrote 'epias_tuketim_hourly' ({len(all_t):,} rows, {all_t['tarih'].min()} to {all_t['tarih'].max()})")

    # -------------------------------------------------------------------------
    # 8. Ingest EPİAŞ Realtime Hourly Generation by Fuel (Gercek_Zamanli_Uretim)
    # -------------------------------------------------------------------------
    print("\n[8/12] Ingesting EPİAŞ Real-Time Hourly Generation (data/epias-data/Gercek_Zamanli_Uretim-*.csv)...")
    uretim_files = sorted(ROOT.glob("data/epias-data/Gercek_Zamanli_Uretim-*.csv"))
    u_dfs = []
    uretim_col_map = {
        'Tarih': 'tarih', 'Saat': 'saat', 'Toplam': 'toplam_mwh', 'Doğal Gaz': 'dogalgaz_mwh',
        'Barajlı': 'barajli_mwh', 'Linyit': 'linyit_mwh', 'Akarsu': 'akarsu_mwh',
        'İthal Kömür': 'ithal_komur_mwh', 'Rüzgar': 'ruzgar_mwh', 'Güneş': 'gunes_mwh',
        'Fuel Oil': 'fuel_oil_mwh', 'Jeotermal': 'jeotermal_mwh', 'Asfaltit Kömür': 'asfaltit_komur_mwh',
        'Taş Kömür': 'tas_komur_mwh', 'Biyokütle': 'biyokutle_mwh', 'Nafta': 'nafta_mwh',
        'LNG': 'lng_mwh', 'Uluslararası': 'uluslararasi_mwh', 'Atık Isı': 'atik_isi_mwh'
    }
    for f in uretim_files:
        df = pd.read_csv(f, sep=None, engine='python')
        df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]
        renamed = {c: uretim_col_map[c] for c in df.columns if c in uretim_col_map}
        sub = df[list(renamed.keys())].rename(columns=renamed).copy()
        sub['tarih'] = parse_tr_date(sub['tarih'])
        sub['saat'] = sub['saat'].astype(str).str.strip()
        for c in sub.columns:
            if c not in ['tarih', 'saat']:
                sub[c] = clean_tr_num(sub[c])
        u_dfs.append(sub)
    if u_dfs:
        all_u = pd.concat(u_dfs).drop_duplicates(subset=['tarih', 'saat']).sort_values(['tarih', 'saat'])
        all_u.to_sql('epias_uretim_hourly', conn, if_exists='replace', index=False)
        table_meta.append({
            'table_name': 'epias_uretim_hourly',
            'row_count': len(all_u),
            'column_count': len(all_u.columns),
            'description': 'EPİAŞ national real-time hourly electricity generation breakdown by fuel type (2014 to 2026)',
            'source_files': f'data/epias-data/Gercek_Zamanli_Uretim-*.csv ({len(uretim_files)} files)'
        })
        print(f"  -> Wrote 'epias_uretim_hourly' ({len(all_u):,} rows, {all_u['tarih'].min()} to {all_u['tarih'].max()})")

    # -------------------------------------------------------------------------
    # 9. Ingest EPİAŞ KGUP Generation Plan (data/kgup/)
    # -------------------------------------------------------------------------
    print("\n[9/12] Ingesting EPİAŞ KGUP Day-Ahead Generation Plan (data/kgup/)...")
    kgup_files = sorted(ROOT.glob("data/kgup/*.csv"))
    k_dfs = []
    kgup_col_map = {
        'Tarih': 'tarih', 'Saat': 'saat', 'Toplam(MWh)': 'toplam_mwh', 'Doğalgaz': 'dogalgaz_mwh',
        'Rüzgar': 'ruzgar_mwh', 'Linyit': 'linyit_mwh', 'Taş Kömür': 'tas_komur_mwh',
        'İthal Kömür': 'ithal_komur_mwh', 'Fueloil': 'fueloil_mwh', 'Jeotermal': 'jeotermal_mwh',
        'Barajlı': 'barajli_mwh', 'Nafta': 'nafta_mwh', 'Biyokütle': 'biyokutle_mwh',
        'Akarsu': 'akarsu_mwh', 'Gunes': 'gunes_mwh', 'Diğer': 'diger_mwh'
    }
    for f in kgup_files:
        df = pd.read_csv(f, sep=None, engine='python')
        df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]
        renamed = {c: kgup_col_map[c] for c in df.columns if c in kgup_col_map}
        sub = df[list(renamed.keys())].rename(columns=renamed).copy()
        sub['tarih'] = parse_tr_date(sub['tarih'])
        sub['saat'] = sub['saat'].astype(str).str.strip()
        for c in sub.columns:
            if c not in ['tarih', 'saat']:
                sub[c] = clean_tr_num(sub[c])
        k_dfs.append(sub)
    if k_dfs:
        all_k = pd.concat(k_dfs).drop_duplicates(subset=['tarih', 'saat']).sort_values(['tarih', 'saat'])
        all_k.to_sql('epias_kgup_hourly', conn, if_exists='replace', index=False)
        table_meta.append({
            'table_name': 'epias_kgup_hourly',
            'row_count': len(all_k),
            'column_count': len(all_k.columns),
            'description': 'EPİAŞ day-ahead final generation plan (KGUP) hourly by fuel type (2025 to 2026)',
            'source_files': f'data/kgup/*.csv ({len(kgup_files)} files)'
        })
        print(f"  -> Wrote 'epias_kgup_hourly' ({len(all_k):,} rows, {all_k['tarih'].min()} to {all_k['tarih'].max()})")

    # -------------------------------------------------------------------------
    # 10. Ingest EPİAŞ Yük Tahmin Planı (data/yuk-tahmin-plani/) & Province Sectoral
    # -------------------------------------------------------------------------
    print("\n[10/12] Ingesting EPİAŞ Yük Tahmin Planı & İzmir/Manisa Sectoral Consumption...")
    ytp_files = sorted(ROOT.glob("data/yuk-tahmin-plani/*.csv"))
    y_dfs = []
    for f in ytp_files:
        df = pd.read_csv(f, sep=None, engine='python')
        df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]
        val_col = [c for c in df.columns if 'Yük Tahmin' in c or 'MWh' in c][0]
        y_df = pd.DataFrame({
            'tarih': parse_tr_date(df['Tarih']),
            'saat': df['Saat'].astype(str).str.strip(),
            'ytp_mwh': clean_tr_num(df[val_col])
        })
        y_dfs.append(y_df)
    if y_dfs:
        all_y = pd.concat(y_dfs).drop_duplicates(subset=['tarih', 'saat']).sort_values(['tarih', 'saat'])
        all_y.to_sql('epias_yuk_tahmin_plani', conn, if_exists='replace', index=False)
        table_meta.append({
            'table_name': 'epias_yuk_tahmin_plani',
            'row_count': len(all_y),
            'column_count': len(all_y.columns),
            'description': 'EPİAŞ day-ahead national electricity load forecast plan (YTP) in MWh (2025 to 2026)',
            'source_files': f'data/yuk-tahmin-plani/*.csv ({len(ytp_files)} files)'
        })
        print(f"  -> Wrote 'epias_yuk_tahmin_plani' ({len(all_y):,} rows)")

    prov_files = sorted(ROOT.glob("data/not-used-izmir-manisa-epias/*/*.csv"))
    p_dfs = []
    prov_col_map = {
        'Dönem': 'donem', 'Şehir': 'sehir', 'Aydınlatma (MWh)': 'aydinlatma_mwh',
        'Mesken (MWh)': 'mesken_mwh', 'Sanayi (MWh)': 'sanayi_mwh',
        'Tarımsal Sulama (MWh)': 'tarimsal_sulama_mwh', 'Ticarethane (MWh)': 'ticarethane_mwh',
        'Diğer (MWh)': 'diger_mwh', 'Genel Toplam (MWh)': 'genel_toplam_mwh',
        'İl Bazında Tüketim Oranı (%)': 'il_pay_pct'
    }
    def norm_donem(val: Any) -> str:
        s = str(val).strip()
        parts = s.split('.')
        if len(parts) == 2:
            m, y = int(parts[0]), int(parts[1])
            return f'{y:04d}-{m:02d}'
        elif len(parts) == 3:
            d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
            return f'{y:04d}-{m:02d}'
        return s

    for f in prov_files:
        df = pd.read_csv(f, sep=None, engine='python')
        df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]
        renamed = {c: prov_col_map[c] for c in df.columns if c in prov_col_map}
        sub = df[list(renamed.keys())].rename(columns=renamed).copy()
        sub['donem'] = sub['donem'].apply(norm_donem)
        sub['sehir'] = sub['sehir'].astype(str).str.strip().str.upper()
        for c in sub.columns:
            if c not in ['donem', 'sehir']:
                sub[c] = clean_tr_num(sub[c])
        p_dfs.append(sub)
    if p_dfs:
        all_p = pd.concat(p_dfs).drop_duplicates(subset=['sehir', 'donem']).sort_values(['sehir', 'donem'])
        all_p.to_sql('epias_province_consumption', conn, if_exists='replace', index=False)
        table_meta.append({
            'table_name': 'epias_province_consumption',
            'row_count': len(all_p),
            'column_count': len(all_p.columns),
            'description': 'EPİAŞ monthly sectoral electricity consumption for İzmir and Manisa (residential, industrial, agricultural, lighting)',
            'source_files': f'data/not-used-izmir-manisa-epias/*/*.csv ({len(prov_files)} files)'
        })
        print(f"  -> Wrote 'epias_province_consumption' ({len(all_p):,} rows)")

    # -------------------------------------------------------------------------
    # 11. Ingest Experiment Results & Benchmarks (outputs/*.json)
    # -------------------------------------------------------------------------
    print("\n[11/12] Ingesting experiment result JSON files (outputs/*.json)...")
    json_files = sorted(ROOT.glob("outputs/*.json"))
    exp_rows = []
    for jf in json_files:
        try:
            stat = jf.stat()
            with open(jf, 'r', encoding='utf-8') as fp:
                content = json.load(fp)
            
            metric_key = None
            metric_val = None
            if isinstance(content, dict):
                for k in ['smape', 'cv_mean', 'mean_smape', 'val_smape', 'score', 'verdict', 'best_score', 'lb']:
                    if k in content:
                        metric_key = k
                        metric_val = str(content[k])
                        break

            exp_rows.append({
                'experiment_name': jf.stem,
                'filename': jf.name,
                'file_size_bytes': stat.st_size,
                'created_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime)),
                'headline_metric': metric_key,
                'headline_value': metric_val,
                'content_json': json.dumps(content, ensure_ascii=False)
            })
        except Exception as e:
            print(f"  [Warning] Failed to ingest {jf.name}: {e}")

    if exp_rows:
        df_exp = pd.DataFrame(exp_rows)
        df_exp.to_sql('experiment_reports', conn, if_exists='replace', index=False)
        table_meta.append({
            'table_name': 'experiment_reports',
            'row_count': len(df_exp),
            'column_count': len(df_exp.columns),
            'description': 'Parsed experiment result metrics, benchmark evaluations, and CV logs',
            'source_files': f'outputs/*.json ({len(json_files)} files)'
        })
        print(f"  -> Wrote 'experiment_reports' ({len(df_exp):,} experiments)")

    # Table Catalog
    df_catalog = pd.DataFrame(table_meta)
    df_catalog.to_sql('table_catalog', conn, if_exists='replace', index=False)

    print("\n[12/12] Creating high-performance SQLite indexes...")
    index_statements = [
        # transformers
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_trafos_tanim ON transformers (tanim);",
        "CREATE INDEX IF NOT EXISTS idx_trafos_il_ilce ON transformers (il, ilce);",
        "CREATE INDEX IF NOT EXISTS idx_trafos_status ON transformers (status);",
        "CREATE INDEX IF NOT EXISTS idx_trafos_guc ON transformers (guc);",
        # historical
        "CREATE INDEX IF NOT EXISTS idx_hist_tanim_tarih ON historical (tanim, tarih);",
        "CREATE INDEX IF NOT EXISTS idx_hist_tarih ON historical (tarih);",
        "CREATE INDEX IF NOT EXISTS idx_hist_il_ilce ON historical (il, ilce);",
        # test_metadata
        "CREATE INDEX IF NOT EXISTS idx_test_tanim_tarih ON test_metadata (tanim, tarih);",
        "CREATE INDEX IF NOT EXISTS idx_test_tarih ON test_metadata (tarih);",
        # sample_submission
        "CREATE INDEX IF NOT EXISTS idx_sample_sub_tanim_tarih ON sample_submission (tanim, tarih);",
        # transformer_load_classification
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_load_cls_tanim ON transformer_load_classification (tanim);",
        "CREATE INDEX IF NOT EXISTS idx_load_cls_type ON transformer_load_classification (load_type);",
        # submissions
        "CREATE INDEX IF NOT EXISTS idx_sub_tanim_tarih ON submissions (tanim, tarih);",
        "CREATE INDEX IF NOT EXISTS idx_sub_tarih ON submissions (tarih);",
        # weather
        "CREATE INDEX IF NOT EXISTS idx_weather_ilce_tarih ON weather (ilce, tarih);",
        "CREATE INDEX IF NOT EXISTS idx_weather_tarih ON weather (tarih);",
        # coordinates
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_coords_ilce ON ilce_coordinates (ilce);",
        # district history & submissions
        "CREATE INDEX IF NOT EXISTS idx_dist_h_il_ilce ON district_history (il, ilce, tarih);",
        "CREATE INDEX IF NOT EXISTS idx_dist_h_tarih ON district_history (tarih);",
        "CREATE INDEX IF NOT EXISTS idx_dist_s_il_ilce ON district_submissions (il, ilce, tarih);",
        "CREATE INDEX IF NOT EXISTS idx_dist_s_tarih ON district_submissions (tarih);",
        # EPIAS Tuketim
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_epias_tuk_tarih_saat ON epias_tuketim_hourly (tarih, saat);",
        "CREATE INDEX IF NOT EXISTS idx_epias_tuk_tarih ON epias_tuketim_hourly (tarih);",
        # EPIAS Uretim
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_epias_ure_tarih_saat ON epias_uretim_hourly (tarih, saat);",
        "CREATE INDEX IF NOT EXISTS idx_epias_ure_tarih ON epias_uretim_hourly (tarih);",
        # KGUP
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_epias_kgup_tarih_saat ON epias_kgup_hourly (tarih, saat);",
        "CREATE INDEX IF NOT EXISTS idx_epias_kgup_tarih ON epias_kgup_hourly (tarih);",
        # YTP
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_epias_ytp_tarih_saat ON epias_yuk_tahmin_plani (tarih, saat);",
        "CREATE INDEX IF NOT EXISTS idx_epias_ytp_tarih ON epias_yuk_tahmin_plani (tarih);",
        # Province Sectoral
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_epias_prov_sehir_donem ON epias_province_consumption (sehir, donem);",
        "CREATE INDEX IF NOT EXISTS idx_epias_prov_donem ON epias_province_consumption (donem);",
        # Experiments
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_exp_name ON experiment_reports (experiment_name);",
        # Catalog
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_catalog_table ON table_catalog (table_name);"
    ]

    cur = conn.cursor()
    for stmt in index_statements:
        try:
            cur.execute(stmt)
        except Exception as e:
            print(f"  [Index Warning] {e} on `{stmt}`")

    cur.close()
    conn.commit()
    conn.close()

    # Automatically run transformer load type classification
    try:
        from scripts.classify_load_types import run_classification
        print("\n" + "=" * 75)
        print("RUNNING AUTOMATIC LOAD TYPE CLASSIFICATION...")
        print("=" * 75)
        run_classification()
    except Exception as e:
        print(f"  [Load Classification Warning] Could not run classification: {e}")

    elapsed = time.time() - t_start
    db_size_mb = os.path.getsize(str(db_file)) / (1024 * 1024)
    print("\n" + "=" * 75)
    print(f"DATABASE BUILD COMPLETE IN {elapsed:.2f}s!")
    print(f"Total SQLite Database Size: {db_size_mb:.2f} MB")
    print(f"Total Tables Ingested: {len(table_meta) + 1}")
    print("=" * 75)


if __name__ == "__main__":
    db_out = sys.argv[1] if len(sys.argv) > 1 else "data/transformers.db"
    build_all_sqlite_database(db_out)
