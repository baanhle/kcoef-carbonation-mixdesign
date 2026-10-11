"""Shared data loading, labels, series definition, augmentations, metrics."""
import json, warnings
import numpy as np, pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')
import os as _os
_PROJ = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project folder (tools/..)
ROOT = _os.path.join(_PROJ, 'data')  # data folder (xls, split json, tuned HP json)
OUT = _PROJ
FULL = ['w/c', 'w/b', 'FA', 'CO2', 'RH', 'T', 'sqrt_t', 't_cure']
NONTIME = ['w/c', 'w/b', 'FA', 'CO2', 'RH', 'T', 't_cure']
SEEDS = list(range(10))
HP = json.load(open(ROOT + r'\task1b_tuned_blackbox.json'))


def load():
    df = pd.read_excel(ROOT + r'\concrete carbonation data.xls')
    ref = df['References'].ffill()
    ren = {'w/c (%)': 'w/c', 'w/b (%)': 'w/b', 'FA (kg/m³)': 'FA', 'CO₂ (%)': 'CO2', 'RH (%)': 'RH',
           'T (℃)': 'T', 'Carbonation time (√d)': 'sqrt_t', 'Curing time (d)': 't_cure'}
    tcol = [c for c in df.columns if 'Carbonation depth' in c][0]
    d = df.rename(columns=ren).copy()
    d['Target'] = d[tcol]
    d[FULL] = d[FULL].fillna(d[FULL].median())  # features only, as in reference scripts
    d['ref_group'] = ref
    n_all = len(d)
    n_nan = int(d['Target'].isna().sum())
    test_idx = set(json.load(open(ROOT + r'\study_grouped_split.json'))['test_row_indices'])
    n_test_nan = int(d.loc[d.index.isin(test_idx), 'Target'].isna().sum())
    d['is_test'] = d.index.isin(test_idx)
    d = d[d['Target'].notna()].copy()
    d['TRH'] = d['T'] * d['RH']
    low = d['TRH'] <= 1837.5
    d['domain'] = np.select([low & (d['w/c'] <= .75), low & (d['w/c'] > .75),
                             (~low) & (d['sqrt_t'] <= 9.875)], ['D1', 'D2', 'D3'], 'D4')
    d['regime'] = np.where(d['CO2'] <= 0.1, 'natural', 'accelerated')
    # series key: study + all non-time features
    key = d['ref_group'].astype(str)
    for c in NONTIME:
        key = key + '|' + d[c].round(6).astype(str)
    d['series'] = pd.factorize(key)[0]
    info = dict(n_rows_total=n_all, n_target_nan_dropped=n_nan, n_target_nan_in_test_idx=n_test_nan,
                n_rows_used=len(d), n_studies=int(d['ref_group'].nunique()))
    return d, info


def fit_n(g):
    """OLS slope of log X vs log t(days) on a series; None if <3 distinct times / positive depth."""
    g = g[(g['Target'] > 0) & (g['sqrt_t'] > 0)]
    s = g['sqrt_t'].round(6)
    if s.nunique() < 3:
        return None
    x = np.log(g['sqrt_t'].values ** 2)
    y = np.log(g['Target'].values)
    return float(np.polyfit(x, y, 1)[0])


def make_model(name, seed):
    p = dict(HP[name]['best_params'])
    p['random_state'] = seed
    if name == 'RandomForest':
        return RandomForestRegressor(**p)
    if name == 'XGBoost':
        return XGBRegressor(n_jobs=8, **p)
    p['thread_count'] = 8; p['allow_writing_files'] = False
    return CatBoostRegressor(**p)


# ---------------- augmentation ----------------
LB = {'w/c': 0.05, 'w/b': 0.05, 'FA': 0, 'CO2': 1e-4, 'RH': 1, 'T': -10, 'sqrt_t': 0.1, 't_cure': 0}
UB = {'RH': 100}


def a1_jitter(tr, rng, r=1.0, frac=0.05):
    std = tr[FULL].std()
    b = tr.sample(n=int(round(r * len(tr))), replace=True, random_state=int(rng.integers(1 << 30))).copy()
    for c in FULL:
        b[c] = (b[c] + rng.normal(0, frac * std[c], len(b))).clip(LB[c], UB.get(c, np.inf))
    return b


def a2_mixup(tr, rng, r=1.0):
    n = int(round(r * len(tr)))
    i, j = rng.integers(0, len(tr), n), rng.integers(0, len(tr), n)
    lam = rng.beta(0.4, 0.4, n)[:, None]
    X = lam * tr[FULL].values[i] + (1 - lam) * tr[FULL].values[j]
    y = lam[:, 0] * tr['Target'].values[i] + (1 - lam[:, 0]) * tr['Target'].values[j]
    b = pd.DataFrame(X, columns=FULL)
    b['Target'] = y
    return b


def series_tables(tr):
    """per-series (smin,smax,n_fit) and fallback per (study, regime) ranges, from rows in tr only."""
    ser = {}
    for sid, g in tr.groupby('series'):
        ser[sid] = (g['sqrt_t'].min(), g['sqrt_t'].max(), g['sqrt_t'].round(6).nunique(), fit_n(g))
    fb = tr.groupby(['ref_group', 'regime'])['sqrt_t'].agg(['min', 'max', lambda x: x.round(6).nunique()])
    return ser, fb


def a3_time(tr, rng, ratio=1.0, use_fit_n=False, base=None, tabs=None, clip_n=(0.2, 1.0)):
    """Physics-guided time augmentation; interpolation only. base = rows eligible as seeds (default tr).
    Returns (aug_df, stats dict)."""
    base = tr if base is None else base
    ser, fb = tabs if tabs is not None else series_tables(tr)
    n_aug = int(round(ratio * len(base)))
    idx = rng.integers(0, len(base), n_aug)
    rows, skipped, via_series, via_fb = [], 0, 0, 0
    B = base.reset_index(drop=True)
    for k in idx:
        r = B.iloc[k]
        lo = hi = None
        n = 0.5
        smin, smax, nd, nf = ser[r['series']]
        if nd >= 2:
            lo, hi = smin, smax
            via_series += 1
            if use_fit_n and nf is not None:
                n = float(np.clip(nf, *clip_n))
        else:
            key = (r['ref_group'], r['regime'])
            if key in fb.index and fb.loc[key].iloc[2] >= 2:
                lo, hi = fb.loc[key].iloc[0], fb.loc[key].iloc[1]
                via_fb += 1
        if lo is None or hi <= lo:
            skipped += 1
            continue
        s2 = rng.uniform(lo, hi)
        y2 = r['Target'] * (s2 / r['sqrt_t']) ** (2 * n) * np.exp(rng.normal(0, 0.05))
        d = r[FULL + ['Target']].copy()
        d['sqrt_t'], d['Target'] = s2, y2
        rows.append(d)
    out = pd.DataFrame(rows).astype(float) if rows else pd.DataFrame(columns=FULL + ['Target'])
    return out, dict(requested=n_aug, produced=len(out), skipped=skipped, via_series=via_series, via_fallback=via_fb)


def augment(variant, tr, seed, ratio=1.0):
    rng = np.random.default_rng(1000 + seed)
    if variant == 'A0':
        return tr[FULL + ['Target']], {}
    if variant == 'A1':
        a = a1_jitter(tr, rng, ratio)
        st = {}
    elif variant == 'A2':
        a, st = a2_mixup(tr, rng, ratio), {}
    elif variant in ('A3', 'A3n'):
        a, st = a3_time(tr, rng, ratio, use_fit_n=(variant == 'A3n'))
    elif variant == 'A3T':
        base = tr[tr['domain'].isin(['D3', 'D4']) | (tr['regime'] == 'natural')]
        a, st = a3_time(tr, rng, ratio, base=base)
        st['n_base'] = len(base)
    else:
        raise ValueError(variant)
    return pd.concat([tr[FULL + ['Target']], a[FULL + ['Target']]], ignore_index=True), st


# ---------------- metrics ----------------
def met(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if len(y) == 0:
        return None
    res = y - p
    ss = ((y - y.mean()) ** 2).sum()
    return dict(n=len(y), rmse=float(np.sqrt((res ** 2).mean())), mae=float(np.abs(res).mean()),
                r2=float(1 - (res ** 2).sum() / ss) if ss > 0 and len(y) > 1 else None)


def strat_met(te, p):
    out = {'overall': met(te['Target'], p)}
    for lab in ['D1', 'D2', 'D3', 'D4']:
        m = (te['domain'] == lab).values
        out[lab] = met(te['Target'].values[m], p[m])
    for lab in ['natural', 'accelerated']:
        m = (te['regime'] == lab).values
        out[lab] = met(te['Target'].values[m], p[m])
    return out


def calib_split(train, frac=0.20):
    """greedy smallest-study-first hold-out of ~frac of train rows (as in reference scripts)."""
    counts = train['ref_group'].value_counts().sort_values(ascending=True)
    cum, studies = 0, []
    for s, n in counts.items():
        studies.append(s)
        cum += n
        if cum / len(train) >= frac:
            break
    m = train['ref_group'].isin(studies)
    return train[~m], train[m], studies


def savej(obj, name):
    with open(OUT + '\\results\\' + name, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
