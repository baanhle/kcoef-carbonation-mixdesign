"""Model wrappers: MX (direct X), MX-mono, Mk (k = X/sqrt_t), Mkn (k_n = X/t^n). XGBoost (fixed task1b HPs) or RF."""
import math
from common import *

MONO = {'MXm1': {'sqrt_t': 1}, 'MXm3': {'sqrt_t': 1, 'w/b': 1, 'w/c': 1, 'CO2': 1, 'T': 1, 't_cure': -1}}  # RH and FA unconstrained
MONO_K = {'w/b': 1, 'w/c': 1, 'CO2': 1, 'T': 1, 't_cure': -1}  # for Mkm (k-model, 7 non-time inputs)
KINDS = ['MX', 'MXm1', 'MXm3', 'Mk', 'Mkn_g', 'Mkn_r', 'Mkc', 'Mkcn_g', 'Mkm']
KKINDS = [k for k in KINDS if k.startswith('Mk')]
LABEL = {'MX': 'MX', 'MXm1': 'MX-mono-t', 'MXm3': 'MX-mono-full', 'Mkm': 'Mk-mono', 'Mk': 'Mk', 'Mkn_g': 'Mkn', 'Mkn_r': 'Mkn per regime', 'Mkc': 'Mk-c', 'Mkcn_g': 'Mkn-c'}  # names as in the manuscript


def qhat(s, a):
    s = np.sort(np.asarray(s)); n = len(s); k = math.ceil((n + 1) * (1 - a))
    return np.inf if k > n else float(s[k - 1])


def base_model(backend, seed, feats, mono=None):
    p = dict(HP[backend]['best_params']); p['random_state'] = seed
    if backend == 'XGBoost':
        if mono: p['monotone_constraints'] = tuple(mono.get(c, 0) for c in feats)
        return XGBRegressor(n_jobs=8, **p)
    return RandomForestRegressor(**p)


def fit_n_within(trn):
    """within-series OLS slope of log X on log t(days) (series fixed effects), global and per regime."""
    out = {}
    for key, sub in [('global', trn), ('natural', trn[trn.regime == 'natural']), ('accelerated', trn[trn.regime == 'accelerated'])]:
        sub = sub[sub.Target > 0]
        nd = sub.groupby('series').sqrt_t.transform(lambda x: x.round(6).nunique())
        sub = sub[nd >= 2]
        lt, lx = np.log(sub.sqrt_t ** 2), np.log(sub.Target)
        ltd = lt - lt.groupby(sub.series).transform('mean'); lxd = lx - lx.groupby(sub.series).transform('mean')
        out[key] = float((ltd * lxd).sum() / (ltd ** 2).sum()) if len(sub) else 0.5
    return out


class KM:
    def __init__(self, kind, backend='XGBoost', seed=0, cfg=None):
        self.kind, self.backend, self.seed = kind, backend, seed
        self.cfg = cfg or dict(log=False, w=False)

    def n_rows(self, df):
        if self.kind in ('Mk', 'Mkc', 'Mkm'): return np.full(len(df), 0.5)
        if self.kind in ('Mkn_g', 'Mkcn_g'): return np.full(len(df), self.n['global'])
        return np.where(df['regime'].values == 'natural', self.n['natural'], self.n['accelerated'])

    def expo(self, df):  # E = t^n = sqrt_t^(2n)
        E = df['sqrt_t'].values ** (2 * self.n_rows(df))
        if self.kind.startswith('Mkc'): E = E * np.sqrt(df['CO2'].values / 20.0)  # Fick-type sqrt(CO2) scaling (extension)
        return E

    def fit(self, trn):
        if self.kind.startswith('MX'):
            self.feats = FULL
            self.m = base_model(self.backend, self.seed, FULL, MONO.get(self.kind) if self.backend == 'XGBoost' else None)
            self.m.fit(trn[FULL].values, trn.Target.values)
            return self
        self.feats = NONTIME
        self.n = fit_n_within(trn)
        E = self.expo(trn); y = trn.Target.values / E
        if self.cfg['log']: y = np.log(y)
        self.m = base_model(self.backend, self.seed, NONTIME, MONO_K if (self.kind == 'Mkm' and self.backend == 'XGBoost') else None)
        self.m.fit(trn[NONTIME].values, y, sample_weight=(E ** 2) if self.cfg['w'] else None)
        return self

    def raw_k(self, df): return self.m.predict(df[NONTIME].values)

    def predict_k(self, df):
        r = self.raw_k(df); return np.exp(r) if self.cfg['log'] else r

    def predict(self, df):
        if self.kind.startswith('MX'): return self.m.predict(df[FULL].values)
        return self.predict_k(df) * self.expo(df)


def agg(lst):
    out = {}
    for lab in lst[0]:
        if lst[0][lab] is None: continue
        out[lab] = {k: dict(mean=float(np.mean([x[lab][k] for x in lst])), std=float(np.std([x[lab][k] for x in lst], ddof=1)) if len(lst) > 1 else 0.0)
                    for k in ['rmse', 'mae', 'r2']}
        out[lab]['n'] = lst[0][lab]['n']
    return out


def load_cfg():
    return json.load(open(OUT + r'\results\p1_model_selection.json'))['chosen_cfg']
