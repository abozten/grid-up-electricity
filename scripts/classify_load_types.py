"""
Grid-Up Electricity: Comprehensive Transformer Load Type Classifier
===================================================================
Classifies each of the 7,368 transformers into standard electrical distribution load archetypes:
1. Agricultural Irrigation (Tarımsal Sulama)
2. Tourism & Seasonal Residential (Turizm & Yazlık Konut)
3. Industrial & Manufacturing (Sanayi & İmalat)
4. Commercial & Services (Ticarethane & Hizmet)
5. Urban Residential (Kentsel Mesken)
6. Continuous Base Load & Infrastructure (Sürekli Baz Yük & Altyapı)
7. Inactive / Standby (Pasif & Boş / Yedek)

Outputs:
- outputs/transformer_load_classification.csv (Full feature metrics + load type + confidence + explanation)
- outputs/transformer_load_types.json (Quick lookup mapping)
- SQLite DB: transformers.db (updates transformers table and adds transformer_load_types table)
"""

import os
import sys
import json
import sqlite3
import time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = str(ROOT / "data/transformers.db")
OUTPUT_CSV = str(ROOT / "outputs/transformer_load_classification.csv")
OUTPUT_JSON = str(ROOT / "outputs/transformer_load_types.json")

# Regional / Geographic Archetype Dictionaries
AGRO_ILCELER = {
    'ÖDEMİŞ', 'TİRE', 'BAYINDIR', 'KİRAZ', 'BERGAMA', 'MENEMEN', 'SALİHLİ',
    'AKHİSAR', 'ALAŞEHİR', 'GÖLMARMARA', 'SARUHANLI', 'SARIGÖL', 'KULA',
    'DEMİRCİ', 'GÖRDES', 'KINIK', 'BEYDAĞ', 'AHMETLİ', 'KÖPRÜBAŞI', 'SELENDİ', 'KIRKAĞAÇ'
}

TOURISM_ILCELER = {
    'ÇEŞME', 'URLA', 'KARABURUN', 'FOÇA', 'DİKİLİ', 'SEFERİHİSAR', 'SELÇUK', 'GÜZELBAHÇE'
}

IND_ILCELER = {
    'ALİAĞA', 'ÇİĞLİ', 'TORBALI', 'KEMALPAŞA', 'TURGUTLU', 'SOMA'
}

METRO_ILCELER = {
    'KONAK', 'BORNOVA', 'KARŞIYAKA', 'BUCA', 'KARABAĞLAR', 'BAYRAKLI',
    'BALÇOVA', 'NARLIDERE', 'GAZİEMİR', 'YUNUSEMRE', 'ŞEHZADELER'
}

def extract_numeric_id(tanim_str):
    """Extract first continuous digit sequence from transformer tanim string."""
    digits = ''.join(c if c.isdigit() else ' ' for c in str(tanim_str)).split()
    if digits:
        try:
            return int(digits[0])
        except ValueError:
            return None
    return None

def run_classification():
    t0 = time.time()
    print("=" * 70)
    print("⚡ Starting Transformer Load Type Classification Engine")
    print("=" * 70)

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    
    # 1. Load tables
    print("📦 Loading data from SQLite database...")
    transformers = pd.read_sql("SELECT * FROM transformers", conn)
    hist = pd.read_sql("SELECT * FROM historical", conn)
    hist['tarih'] = pd.to_datetime(hist['tarih'])
    
    weather = pd.read_sql("SELECT ilce, tarih, temp_mean, precipitation FROM weather", conn)
    weather['tarih'] = pd.to_datetime(weather['tarih'])
    
    subs = pd.read_sql("SELECT tanim, tarih, v33 FROM submissions", conn)
    subs['tarih'] = pd.to_datetime(subs['tarih'])
    
    print(f"   Transformers: {len(transformers):,} | Hist rows: {len(hist):,} | Subs rows: {len(subs):,}")

    # 2. Merge historical with weather & calendar
    print("📊 Computing multi-season & calendar metrics for historical transformers...")
    cols_to_add = [c for c in ['guc', 'il', 'bolge', 'ilce'] if c not in hist.columns]
    if cols_to_add:
        hist = hist.merge(transformers[['tanim'] + cols_to_add], on='tanim', how='left')
    hist = hist.merge(weather, on=['ilce', 'tarih'], how='left')

    hist['month'] = hist['tarih'].dt.month
    hist['dow'] = hist['tarih'].dt.dayofweek
    hist['is_weekend'] = hist['dow'].isin([5, 6]).astype(int)
    hist['is_sunday'] = (hist['dow'] == 6).astype(int)

    # Religious bayram dates in 2025 (Ramazan & Kurban Bayramı)
    bayram_dates_2025 = pd.to_datetime([
        '2025-03-30', '2025-03-31', '2025-04-01',
        '2025-06-06', '2025-06-07', '2025-06-08', '2025-06-09'
    ])
    hist['is_bayram'] = hist['tarih'].isin(bayram_dates_2025).astype(int)

    # Fast aggregations
    agg_base = hist.groupby('tanim').agg(
        days=('tuketim', 'count'),
        mean_t=('tuketim', 'mean'),
        std_t=('tuketim', 'std'),
        median_t=('tuketim', 'median'),
        max_t=('tuketim', 'max'),
        min_t=('tuketim', 'min'),
        p90_t=('tuketim', lambda x: np.percentile(x, 90)),
        zero_ratio=('tuketim', lambda x: (x == 0).mean()),
        near_zero_ratio=('tuketim', lambda x: (x < 1.0).mean())
    )

    summer_mean = hist[hist['month'].isin([6, 7, 8])].groupby('tanim')['tuketim'].mean().rename('summer_mean')
    winter_mean = hist[hist['month'].isin([1, 2, 12])].groupby('tanim')['tuketim'].mean().rename('winter_mean')
    spring_mean = hist[hist['month'].isin([3, 4, 5])].groupby('tanim')['tuketim'].mean().rename('spring_mean')
    autumn_mean = hist[hist['month'].isin([9, 10, 11])].groupby('tanim')['tuketim'].mean().rename('autumn_mean')

    wd_mean = hist[hist['dow'] < 5].groupby('tanim')['tuketim'].mean().rename('wd_mean')
    we_mean = hist[hist['is_weekend'] == 1].groupby('tanim')['tuketim'].mean().rename('we_mean')
    sun_mean = hist[hist['is_sunday'] == 1].groupby('tanim')['tuketim'].mean().rename('sun_mean')
    bayram_mean = hist[hist['is_bayram'] == 1].groupby('tanim')['tuketim'].mean().rename('bayram_mean')

    # Temperature correlation per transformer
    temp_corrs = hist.groupby('tanim').apply(
        lambda g: g['tuketim'].corr(g['temp_mean']) if len(g) > 30 and g['tuketim'].std() > 0 and g['temp_mean'].std() > 0 else 0.0,
        include_groups=False
    ).rename('temp_corr')

    seen_feats = transformers.set_index('tanim').join(
        [agg_base, summer_mean, winter_mean, spring_mean, autumn_mean, wd_mean, we_mean, sun_mean, bayram_mean, temp_corrs],
        how='left'
    )

    # Derived ratios
    seen_feats['load_factor'] = seen_feats['mean_t'] / (seen_feats['guc'] * 24.0)
    seen_feats['p90_load_factor'] = seen_feats['p90_t'] / (seen_feats['guc'] * 24.0)
    seen_feats['cv'] = seen_feats['std_t'] / (seen_feats['mean_t'] + 1e-4)

    seen_feats['sw_ratio'] = np.where(
        seen_feats['winter_mean'].isna() | (seen_feats['winter_mean'] < 0.1),
        np.where(seen_feats['summer_mean'] > 5.0, 50.0, 1.0),
        seen_feats['summer_mean'] / (seen_feats['winter_mean'] + 1e-4)
    )

    seen_feats['we_wd_ratio'] = np.where(seen_feats['wd_mean'] > 0.1, seen_feats['we_mean'] / seen_feats['wd_mean'], 1.0)
    seen_feats['sun_wd_ratio'] = np.where(seen_feats['wd_mean'] > 0.1, seen_feats['sun_mean'] / seen_feats['wd_mean'], 1.0)
    seen_feats['bayram_ratio'] = np.where(seen_feats['wd_mean'] > 0.1, seen_feats['bayram_mean'] / seen_feats['wd_mean'], 1.0)

    # 3. Classify Seen & Train-Only Transformers
    print("🏷️ Classifying historical / seen transformers (5,344 units)...")
    
    seen_results = {}
    for tanim, row in seen_feats.iterrows():
        if pd.isna(row['days']) or row['days'] == 0:
            continue
            
        mean_v = float(row['mean_t']) if pd.notna(row['mean_t']) else 0.0
        zr = float(row['zero_ratio']) if pd.notna(row['zero_ratio']) else 0.0
        nzr = float(row['near_zero_ratio']) if pd.notna(row['near_zero_ratio']) else 0.0
        lf = float(row['load_factor']) if pd.notna(row['load_factor']) else 0.0
        cv = float(row['cv']) if pd.notna(row['cv']) else 0.0
        sw = float(row['sw_ratio']) if pd.notna(row['sw_ratio']) else 1.0
        we_wd = float(row['we_wd_ratio']) if pd.notna(row['we_wd_ratio']) else 1.0
        sun_wd = float(row['sun_wd_ratio']) if pd.notna(row['sun_wd_ratio']) else 1.0
        bayram_r = float(row['bayram_ratio']) if pd.notna(row['bayram_ratio']) else 1.0
        t_corr = float(row['temp_corr']) if pd.notna(row['temp_corr']) else 0.0
        guc_v = int(row['guc']) if pd.notna(row['guc']) else 400
        ilce = str(row['ilce']).upper()
        
        is_agro_district = ilce in AGRO_ILCELER
        is_tour_district = ilce in TOURISM_ILCELER
        is_ind_district = ilce in IND_ILCELER
        is_metro_district = ilce in METRO_ILCELER

        # Rule 1: Inactive / Standby
        if zr >= 0.75 or (mean_v < 1.0 and nzr >= 0.85):
            seen_results[tanim] = {
                'load_type': 'Inactive / Standby',
                'load_type_tr': 'Pasif & Boş / Yedek',
                'confidence': 0.96 if zr >= 0.90 else 0.88,
                'explanation': f"Sürekli sıfır veya sıfıra yakın tüketim (Sıfır gün oranı: %{zr*100:.1f}, Günlük ortalama: {mean_v:.2f} kWh). Boş, kapalı veya yedek trafo."
            }
            continue

        # Rule 2: Agricultural Irrigation (Tarımsal Sulama)
        if (sw >= 2.2 and is_agro_district) or (sw >= 2.8 and not is_tour_district) or (sw >= 1.75 and is_agro_district and t_corr > 0.40):
            conf = 0.93 if (sw >= 2.5 and is_agro_district) else 0.86
            seen_results[tanim] = {
                'load_type': 'Agricultural Irrigation',
                'load_type_tr': 'Tarımsal Sulama',
                'confidence': conf,
                'explanation': f"Yüksek yaz/kış tüketim oranı (S/W: {sw:.2f}), sıcaklık korelasyonu (r: {t_corr:.2f}) ve {ilce} tarımsal havza karakteristiği."
            }
            continue

        # Rule 3: Tourism & Seasonal Residential (Turizm ve Yazlık Konut)
        if is_tour_district and (sw >= 1.6 or (sw >= 1.35 and we_wd >= 1.02)):
            conf = 0.91 if sw >= 1.8 else 0.83
            seen_results[tanim] = {
                'load_type': 'Tourism & Seasonal Residential',
                'load_type_tr': 'Turizm & Yazlık Konut',
                'confidence': conf,
                'explanation': f"{ilce} sahil/tatil bölgesinde belirgin yaz tüketim artışı (S/W: {sw:.2f}), hafta sonu canlılığı (HaftaSonu/İçi: {we_wd:.2f})."
            }
            continue

        # Rule 4: Continuous Base Load & Public Infrastructure / Lighting
        if cv < 0.20 and 0.95 <= we_wd <= 1.05 and 0.95 <= sun_wd <= 1.05:
            seen_results[tanim] = {
                'load_type': 'Continuous Base Load & Infrastructure',
                'load_type_tr': 'Sürekli Baz Yük & Altyapı',
                'confidence': 0.89,
                'explanation': f"Düşük dalgalanma (CV: {cv:.2f}), 7/24 kesintisiz sabit profil (HaftaSonu/İçi: {we_wd:.2f}). Kamu/altyapı/baz yük karakteri."
            }
            continue
            
        if sw < 0.70 and 0.95 <= we_wd <= 1.05 and t_corr < -0.30:
            seen_results[tanim] = {
                'load_type': 'Continuous Base Load & Infrastructure',
                'load_type_tr': 'Aydınlatma & Şebeke Altyapısı',
                'confidence': 0.87,
                'explanation': f"Kış aylarında artan ters mevsimsellik (S/W: {sw:.2f}) ve gece aydınlatma / kamu altyapı profili."
            }
            continue

        # Rule 5: Industrial & Manufacturing (Sanayi & İmalat)
        if (guc_v >= 400 and lf >= 0.30 and 0.75 <= sw <= 1.45 and (sun_wd <= 0.88 or bayram_r <= 0.80)) or \
           (is_ind_district and guc_v >= 400 and lf >= 0.25 and sun_wd <= 0.85) or \
           (guc_v >= 630 and mean_v >= 4000 and sun_wd <= 0.85):
            conf = 0.92 if (lf >= 0.40 and sun_wd <= 0.80) else 0.85
            seen_results[tanim] = {
                'load_type': 'Industrial & Manufacturing',
                'load_type_tr': 'Sanayi & İmalat',
                'confidence': conf,
                'explanation': f"Yüksek güç ({guc_v} kVA) ve yük faktörü (%{lf*100:.1f}), Pazar/Bayram üretim tatili düşüşü (Pazar/İçi: {sun_wd:.2f}, Bayram/İçi: {bayram_r:.2f})."
            }
            continue

        # Rule 6: Commercial & Services (Ticarethane & Hizmet)
        if (sun_wd <= 0.82 or we_wd <= 0.86) and (is_metro_district or guc_v >= 250):
            conf = 0.87 if sun_wd <= 0.75 else 0.81
            seen_results[tanim] = {
                'load_type': 'Commercial & Services',
                'load_type_tr': 'Ticarethane & Hizmet',
                'confidence': conf,
                'explanation': f"Hafta içi yoğunluğu ve belirgin Pazar günü düşüşü (Pazar/İçi: {sun_wd:.2f}), yaz soğutma klima yükü (S/W: {sw:.2f})."
            }
            continue

        # Rule 7: Urban Residential (Kentsel Mesken)
        seen_results[tanim] = {
            'load_type': 'Urban Residential',
            'load_type_tr': 'Kentsel Mesken',
            'confidence': 0.86,
            'explanation': f"Yıl boyu dengeli kış ısıtma ve yaz soğutma dengesi (S/W: {sw:.2f}), düzenli konut hafta sonu profili (HaftaSonu/İçi: {we_wd:.2f}, Yük Faktörü: %{lf*100:.1f})."
        }

    # 4. District Empirical Archetype Distribution (for cold-start inference)
    print("🗺️ Building district load distribution priors from seen fleet...")
    seen_df_res = pd.DataFrame.from_dict(seen_results, orient='index')
    seen_df_res = seen_df_res.join(transformers.set_index('tanim')[['ilce', 'guc', 'status']], how='inner')
    
    district_priors = {}
    for ilce_name, grp in seen_df_res.groupby('ilce'):
        type_counts = grp['load_type'].value_counts(normalize=True).to_dict()
        dominant_type = grp['load_type'].mode()[0]
        district_priors[ilce_name] = {
            'distribution': type_counts,
            'dominant': dominant_type,
            'dominant_tr': grp['load_type_tr'].mode()[0],
            'count': len(grp)
        }

    # 5. Process Cold Start Transformers (2,024 units)
    print("❄️ Classifying cold-start transformers (2,024 units)...")
    subs['month'] = subs['tarih'].dt.month
    subs['dow'] = subs['tarih'].dt.dayofweek
    subs['is_weekend'] = subs['dow'].isin([5, 6]).astype(int)

    cold_t = transformers[transformers['status'] == 'Cold Start'].copy()
    cold_tanims = set(cold_t['tanim'].unique())
    cold_subs = subs[subs['tanim'].isin(cold_tanims)]

    cold_agg = cold_subs.groupby('tanim').agg(
        sub_mean=('v33', 'mean'),
        sub_std=('v33', 'std'),
        sub_max=('v33', 'max'),
        sub_min=('v33', 'min'),
        sub_zero_ratio=('v33', lambda x: (x == 0).mean())
    )

    cold_summer = cold_subs[cold_subs['month'].isin([6, 7])].groupby('tanim')['v33'].mean().rename('sub_summer')
    cold_spring = cold_subs[cold_subs['month'].isin([4, 5])].groupby('tanim')['v33'].mean().rename('sub_spring')
    cold_we = cold_subs[cold_subs['is_weekend'] == 1].groupby('tanim')['v33'].mean().rename('sub_we')
    cold_wd = cold_subs[cold_subs['is_weekend'] == 0].groupby('tanim')['v33'].mean().rename('sub_wd')

    cold_feats = cold_t.set_index('tanim').join([cold_agg, cold_summer, cold_spring, cold_we, cold_wd], how='left')
    cold_feats['pred_lf'] = cold_feats['sub_mean'] / (cold_feats['guc'] * 24.0)
    cold_feats['pred_cv'] = cold_feats['sub_std'] / (cold_feats['sub_mean'] + 1e-4)
    cold_feats['pred_ramp'] = cold_feats['sub_summer'] / (cold_feats['sub_spring'] + 1e-4)
    cold_feats['pred_we_wd'] = np.where(cold_feats['sub_wd'] > 0.1, cold_feats['sub_we'] / cold_feats['sub_wd'], 1.0)

    # Build numeric ID index for nearest neighbor lookup
    seen_with_id = seen_df_res.copy()
    seen_with_id['num_id'] = [extract_numeric_id(t) for t in seen_with_id.index]
    seen_with_id = seen_with_id.dropna(subset=['num_id'])
    seen_with_id['num_id'] = seen_with_id['num_id'].astype(int)
    seen_with_id_sorted = seen_with_id.sort_values('num_id')

    cold_results = {}
    for tanim, row in cold_feats.iterrows():
        guc_v = int(row['guc']) if pd.notna(row['guc']) else 400
        ilce = str(row['ilce']).upper()
        il = str(row['il']).upper()
        p_mean = float(row['sub_mean']) if pd.notna(row['sub_mean']) else 0.0
        p_lf = float(row['pred_lf']) if pd.notna(row['pred_lf']) else 0.0
        p_ramp = float(row['pred_ramp']) if pd.notna(row['pred_ramp']) else 1.0
        p_we_wd = float(row['pred_we_wd']) if pd.notna(row['pred_we_wd']) else 1.0
        p_zr = float(row['sub_zero_ratio']) if pd.notna(row['sub_zero_ratio']) else 0.0
        
        is_agro_district = ilce in AGRO_ILCELER
        is_tour_district = ilce in TOURISM_ILCELER
        is_ind_district = ilce in IND_ILCELER
        is_metro_district = ilce in METRO_ILCELER

        # Check nearest numerical ID neighbours in same district
        num_id = extract_numeric_id(tanim)
        neighbor_type = None
        neighbor_type_tr = None
        if num_id is not None:
            id_diff = np.abs(seen_with_id_sorted['num_id'] - num_id)
            close_mask = (id_diff <= 200) & (seen_with_id_sorted['ilce'] == row['ilce'])
            if close_mask.sum() >= 1:
                closest = seen_with_id_sorted[close_mask].iloc[id_diff[close_mask].argmin()]
                neighbor_type = closest['load_type']
                neighbor_type_tr = closest['load_type_tr']

        # Rule 1: Inactive / Standby in forecast
        if p_mean < 5.0 or p_zr > 0.80:
            cold_results[tanim] = {
                'load_type': 'Inactive / Standby',
                'load_type_tr': 'Pasif & Boş / Yedek',
                'confidence': 0.85,
                'explanation': f"Geçmiş verisi yok (Cold Start). Tahmin seviyesi sıfıra yakın (Günlük ort: {p_mean:.1f} kWh). Muhtemel yedek/kapalı trafo."
            }
            continue

        # Rule 2: Agricultural Irrigation in Cold Start
        if (is_agro_district and p_ramp >= 1.12) or (p_ramp >= 1.20 and not is_tour_district) or (neighbor_type == 'Agricultural Irrigation' and is_agro_district):
            conf = 0.82 if is_agro_district else 0.75
            cold_results[tanim] = {
                'load_type': 'Agricultural Irrigation',
                'load_type_tr': 'Tarımsal Sulama',
                'confidence': conf,
                'explanation': f"Cold Start trafo. {ilce} tarımsal havzasında yaz sulama artış tahmini (Yaz/İlkbahar: {p_ramp:.2f}) ve bölgesel/komşu trafo profili."
            }
            continue

        # Rule 3: Tourism & Seasonal Residential in Cold Start
        if is_tour_district and (p_ramp >= 1.08 or p_we_wd >= 1.0 or neighbor_type == 'Tourism & Seasonal Residential'):
            conf = 0.84
            cold_results[tanim] = {
                'load_type': 'Tourism & Seasonal Residential',
                'load_type_tr': 'Turizm & Yazlık Konut',
                'confidence': conf,
                'explanation': f"Cold Start trafo. {ilce} sahil/turizm bölgesinde yaz mevsimsel yük artış modeli ve turizm bölgesi karakteristiği."
            }
            continue

        # Rule 4: Industrial & Manufacturing in Cold Start
        if (is_ind_district and guc_v >= 400 and p_lf >= 0.08) or (guc_v >= 630 and neighbor_type == 'Industrial & Manufacturing'):
            conf = 0.81
            cold_results[tanim] = {
                'load_type': 'Industrial & Manufacturing',
                'load_type_tr': 'Sanayi & İmalat',
                'confidence': conf,
                'explanation': f"Cold Start trafo. Yüksek trafo kapasitesi ({guc_v} kVA), {ilce} sanayi/OSB lokasyonu ve endüstriyel komşu trafo yapısı."
            }
            continue

        # Rule 5: Commercial & Services in Cold Start
        if (is_metro_district and guc_v >= 250 and p_we_wd < 0.99) or (neighbor_type == 'Commercial & Services'):
            conf = 0.78
            cold_results[tanim] = {
                'load_type': 'Commercial & Services',
                'load_type_tr': 'Ticarethane & Hizmet',
                'confidence': conf,
                'explanation': f"Cold Start trafo. {ilce} metropol/ticaret merkezinde hafta içi çalışma düzeni ve ticari yük profili."
            }
            continue

        # Rule 6: Continuous Base Load & Infrastructure in Cold Start
        if p_ramp < 1.03 and abs(p_we_wd - 1.0) < 0.01 and (neighbor_type == 'Continuous Base Load & Infrastructure'):
            conf = 0.77
            cold_results[tanim] = {
                'load_type': 'Continuous Base Load & Infrastructure',
                'load_type_tr': 'Sürekli Baz Yük & Altyapı',
                'confidence': conf,
                'explanation': f"Cold Start trafo. Mevsimsel ve haftalık değişimi sıfıra yakın kesintisiz baz yük / kamu altyapı karakteristiği."
            }
            continue

        # Rule 7: Urban Residential default for cold start
        # Use district dominant prior if available
        d_prior = district_priors.get(ilce, {})
        dom_tr = d_prior.get('dominant_tr', 'Kentsel Mesken')
        dom_en = d_prior.get('dominant', 'Urban Residential')
        
        conf = 0.80 if dom_en == 'Urban Residential' else 0.72
        cold_results[tanim] = {
            'load_type': dom_en,
            'load_type_tr': dom_tr,
            'confidence': conf,
            'explanation': f"Cold Start trafo. {ilce} ilçesi baskın dağılımı ({dom_tr}) ve {guc_v} kVA trafo ölçeği baz alınarak sınıflandırıldı."
        }

    # 6. Combine all results
    print("📑 Consolidating classifications for all 7,368 transformers...")
    all_results = {**seen_results, **cold_results}
    
    # Build complete consolidated DataFrame
    records = []
    for idx, row in transformers.iterrows():
        tanim = row['tanim']
        cls_info = all_results.get(tanim, {
            'load_type': 'Urban Residential',
            'load_type_tr': 'Kentsel Mesken',
            'confidence': 0.70,
            'explanation': 'Varsayılan kentsel dağıtım trafosu sınıflandırması.'
        })
        
        # Get metrics if seen
        s_row = seen_feats.loc[tanim] if tanim in seen_feats.index else None
        
        records.append({
            'tanim': tanim,
            'guc': row['guc'],
            'lokasyon': row['lokasyon'],
            'il': row['il'],
            'bolge': row['bolge'],
            'ilce': row['ilce'],
            'status': row['status'],
            'load_type': cls_info['load_type'],
            'load_type_tr': cls_info['load_type_tr'],
            'confidence': round(cls_info['confidence'], 2),
            'explanation': cls_info['explanation'],
            'train_days': int(row['train_days']) if pd.notna(row['train_days']) else 0,
            'train_mean': round(float(row['train_mean']), 2) if pd.notna(row['train_mean']) else 0.0,
            'pred_mean': round(float(row['pred_mean']), 2) if pd.notna(row['pred_mean']) else 0.0,
            'load_factor': round(float(s_row['load_factor']), 4) if (s_row is not None and pd.notna(s_row['load_factor'])) else None,
            'sw_ratio': round(float(s_row['sw_ratio']), 2) if (s_row is not None and pd.notna(s_row['sw_ratio'])) else None,
            'we_wd_ratio': round(float(s_row['we_wd_ratio']), 3) if (s_row is not None and pd.notna(s_row['we_wd_ratio'])) else None,
            'sun_wd_ratio': round(float(s_row['sun_wd_ratio']), 3) if (s_row is not None and pd.notna(s_row['sun_wd_ratio'])) else None,
            'zero_ratio': round(float(s_row['zero_ratio']), 3) if (s_row is not None and pd.notna(s_row['zero_ratio'])) else None,
            'temp_corr': round(float(s_row['temp_corr']), 3) if (s_row is not None and pd.notna(s_row['temp_corr'])) else None,
        })

    full_df = pd.DataFrame(records)
    
    # 7. Save outputs
    print(f"💾 Saving CSV output to {OUTPUT_CSV}...")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    full_df.to_csv(OUTPUT_CSV, index=False)

    print(f"💾 Saving JSON lookup to {OUTPUT_JSON}...")
    json_lookup = {
        r['tanim']: {
            'load_type': r['load_type'],
            'load_type_tr': r['load_type_tr'],
            'confidence': r['confidence'],
            'explanation': r['explanation']
        }
        for r in records
    }
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(json_lookup, f, ensure_ascii=False, indent=2)

    # 8. Update SQLite Database
    print("💾 Updating SQLite database schema & tables...")
    cur = conn.cursor()
    
    # Create or replace transformer_load_types table
    cur.execute("DROP TABLE IF EXISTS transformer_load_types")
    cur.execute("""
        CREATE TABLE transformer_load_types (
            tanim TEXT PRIMARY KEY,
            guc INTEGER,
            il TEXT,
            bolge TEXT,
            ilce TEXT,
            status TEXT,
            load_type TEXT,
            load_type_tr TEXT,
            confidence REAL,
            explanation TEXT,
            train_days INTEGER,
            train_mean REAL,
            pred_mean REAL,
            load_factor REAL,
            sw_ratio REAL,
            we_wd_ratio REAL,
            zero_ratio REAL,
            temp_corr REAL
        )
    """)
    
    insert_data = [
        (
            r['tanim'], r['guc'], r['il'], r['bolge'], r['ilce'], r['status'],
            r['load_type'], r['load_type_tr'], r['confidence'], r['explanation'],
            r['train_days'], r['train_mean'], r['pred_mean'], r['load_factor'],
            r['sw_ratio'], r['we_wd_ratio'], r['zero_ratio'], r['temp_corr']
        )
        for r in records
    ]
    cur.executemany("""
        INSERT INTO transformer_load_types VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, insert_data)
    
    # Update transformers table with new columns if not present
    cur.execute("PRAGMA table_info(transformers)")
    existing_cols = [c[1] for c in cur.fetchall()]
    
    for col_name, col_type in [
        ('load_type', 'TEXT'),
        ('load_type_tr', 'TEXT'),
        ('confidence', 'REAL'),
        ('load_explanation', 'TEXT')
    ]:
        if col_name not in existing_cols:
            cur.execute(f"ALTER TABLE transformers ADD COLUMN {col_name} {col_type}")
            
    # Update values in transformers table
    cur.executemany("""
        UPDATE transformers
        SET load_type = ?, load_type_tr = ?, confidence = ?, load_explanation = ?
        WHERE tanim = ?
    """, [(r['load_type'], r['load_type_tr'], r['confidence'], r['explanation'], r['tanim']) for r in records])

    conn.commit()
    conn.close()

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"✅ Classification finished successfully in {elapsed:.2f} seconds!")
    print("=" * 70)
    print("\n📊 Fleet Load Type Distribution (Total: 7,368 transformers):")
    print(full_df['load_type_tr'].value_counts())
    print("\n📊 Status Breakdown by Load Type:")
    print(pd.crosstab(full_df['load_type_tr'], full_df['status']))
    print("\n" + "=" * 70)

if __name__ == '__main__':
    run_classification()
