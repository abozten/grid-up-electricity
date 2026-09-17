"""Row-wise features from the v22 spec that pipeline4 does not build.

Everything here is a pure function of (tarih, ilce, daily weather) -- no history aggregation,
so there is no leakage surface and no cutoff to respect. That is why it can be applied straight
to the cached training blocks in outputs/cache/ instead of rebuilding them.

NOT IMPLEMENTED, because the data does not exist in this competition:
  bildirimli_sum / sehir      -- no announced-outage table, no separate city column (il covers it)
  wind_dir_10m, prob_precip   -- absent from the Open-Meteo archive pull
  hourly std / most _range    -- weather is DAILY aggregates; only max-min ranges are recoverable
  sample_weight               -- a training knob, not a feature; see exp_extra_feats.py

Splines are fitted on the full integer domain (doy 1-366 etc.), not on the data, so train and
validation get byte-identical basis functions.
"""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer

COORDS = json.load(open('data/ilce_coords.json'))

CAL_FEATS = ['year', 'quarter', 'dayofmonth', 'season', 'is_month_start', 'is_month_end',
             'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end']
SPLINE_FEATS = [f'{b}_sp_{i}' for b in ('doy', 'dom', 'dowk') for i in range(3)]
GEO_FEATS = ['lat', 'lon']
WX2_FEATS = ['dew_point', 'wind_chill', 'heat_index_C', 'gust_excess',
             'sicaklik_nem_orani', 'sicaklik_ruzgar_etkilesimi', 'solar_temp_interaction',
             'cloud_humidity_interaction', 'ters_etkilesim_yagis_gunes']
EXTRA_FEATS = CAL_FEATS + SPLINE_FEATS + GEO_FEATS + WX2_FEATS

# degree=2, n_knots=2 -> 3 basis columns each, matching the *_sp_0/1/2 naming in the spec
_SP = {}
for _name, _dom in (('doy', np.arange(1, 367)), ('dom', np.arange(1, 32)), ('dowk', np.arange(7))):
    _SP[_name] = SplineTransformer(degree=2, n_knots=2).fit(_dom.reshape(-1, 1))


def add_extra_feats(df):
    df = df.copy()
    d = df.tarih

    df['year'] = d.dt.year
    df['quarter'] = d.dt.quarter
    df['dayofmonth'] = d.dt.day
    df['season'] = (d.dt.month % 12) // 3          # 0=winter 1=spring 2=summer 3=autumn
    df['is_month_start'] = d.dt.is_month_start.astype(int)
    df['is_month_end'] = d.dt.is_month_end.astype(int)
    df['is_quarter_start'] = d.dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = d.dt.is_quarter_end.astype(int)
    df['is_year_start'] = d.dt.is_year_start.astype(int)
    df['is_year_end'] = d.dt.is_year_end.astype(int)

    for name, src in (('doy', d.dt.dayofyear), ('dom', d.dt.day), ('dowk', d.dt.dayofweek)):
        B = _SP[name].transform(src.values.reshape(-1, 1))
        for i in range(B.shape[1]):
            df[f'{name}_sp_{i}'] = B[:, i]

    ll = df.ilce.astype(str).map(COORDS)
    df['lat'] = [p[0] if isinstance(p, list) else np.nan for p in ll]
    df['lon'] = [p[1] if isinstance(p, list) else np.nan for p in ll]

    t = df.temperature_2m_mean
    rh = df.relative_humidity_2m_mean
    w = df.wind_speed_10m_mean
    rad = df.shortwave_radiation_sum
    cc = df.cloud_cover_mean

    # Magnus approximation -- the standard one, valid over the -40..50C range this data lives in
    a, b = 17.27, 237.7
    g = (a * t) / (b + t) + np.log(rh.clip(lower=1) / 100)
    df['dew_point'] = (b * g) / (a - g)

    # JAG/TI wind chill, in km/h as the formula requires (Open-Meteo gives m/s)
    wk = (w * 3.6).clip(lower=0)
    df['wind_chill'] = 13.12 + 0.6215*t - 11.37*wk**0.16 + 0.3965*t*wk**0.16

    # Rothfusz heat index, Celsius-native coefficients
    df['heat_index_C'] = (-8.784695 + 1.61139411*t + 2.338549*rh - 0.14611605*t*rh
                          - 0.012308094*t**2 - 0.016424828*rh**2 + 0.002211732*t**2*rh
                          + 0.00072546*t*rh**2 - 0.000003582*t**2*rh**2)

    df['gust_excess'] = df.wind_gusts_10m_max - w
    df['sicaklik_nem_orani'] = t * rh / 100
    df['sicaklik_ruzgar_etkilesimi'] = t * w
    df['solar_temp_interaction'] = rad * t
    df['cloud_humidity_interaction'] = cc * rh
    df['ters_etkilesim_yagis_gunes'] = -rad * df.precipitation_sum
    return df


if __name__ == '__main__':
    n = 400
    df = pd.DataFrame({
        'tarih': pd.date_range('2025-01-01', periods=n, freq='D'),
        'ilce': ['BORNOVA'] * n,
        'temperature_2m_mean': np.linspace(-5, 40, n),
        'relative_humidity_2m_mean': np.linspace(10, 100, n),
        'wind_speed_10m_mean': np.linspace(0, 15, n),
        'wind_gusts_10m_max': np.linspace(0, 30, n),
        'cloud_cover_mean': np.linspace(0, 100, n),
        'shortwave_radiation_sum': np.linspace(0, 30, n),
        'precipitation_sum': np.linspace(0, 50, n),
    })
    X = add_extra_feats(df)

    assert set(EXTRA_FEATS) <= set(X.columns), set(EXTRA_FEATS) - set(X.columns)
    assert X[EXTRA_FEATS].notna().all().all(), X[EXTRA_FEATS].isna().sum().loc[lambda s: s > 0]
    assert (X.lat == 38.479).all() and (X.lon == 27.24).all()
    # dew point never exceeds air temperature -- the physical invariant that catches a sign slip
    assert (X.dew_point <= X.temperature_2m_mean + 1e-6).all()
    # splines partition unity: each basis row sums to 1
    for b in ('doy', 'dom', 'dowk'):
        s = X[[f'{b}_sp_{i}' for i in range(3)]].sum(axis=1)
        assert np.allclose(s, 1.0), (b, s.min(), s.max())
    # seasons: Jan->0, Apr->1, Jul->2, Oct->3
    assert list(X.set_index('tarih').loc[['2025-01-15', '2025-04-15', '2025-07-15', '2025-10-15'],
                                         'season']) == [0, 1, 2, 3]
    # transform is deterministic and row-order independent
    Y = add_extra_feats(df.iloc[::-1]).iloc[::-1]
    assert np.allclose(X[WX2_FEATS + SPLINE_FEATS], Y[WX2_FEATS + SPLINE_FEATS])
    print(f"ok: {len(EXTRA_FEATS)} extra features")
