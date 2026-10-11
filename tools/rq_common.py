"""Shared pieces for the RQ analyses (_rq outputs): series table, n-model, Mkn_x (k given n-hat(x)), temporal-split helpers."""
from models import *
rnd = lambda x: np.round(x, 6)
N_LO, N_HI = 0.2, 1.0  # clip of n-hat used inside t^n-hat (documented in the results)


def series_table(df):
    """one row per series with >=3 distinct times and positive depths: fitted n (OLS slope of log X on log t(days)) + the 7 non-time inputs"""
    rows = []
    for sid, g in df.groupby('series'):
        n = fit_n(g)
        if n is None: continue
        r = g.iloc[0]
        rows.append(dict(series=sid, n=n, study=str(r.ref_group), regime=r.regime, nrows=len(g), ntimes=int(g.sqrt_t.round(6).nunique()), **{c: float(r[c]) for c in NONTIME}))
    return pd.DataFrame(rows)


def n_model(seed=0):
    return XGBRegressor(max_depth=2, n_estimators=150, learning_rate=0.05, min_child_weight=3, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=seed, n_jobs=4)


class KMX(KM):
    """Mkn_x: X = k(x) * t^n-hat(x). Two-stage: (1) shallow XGB n-hat(x) trained on series-level n of the training rows only,
    (2) k fitted on X / t^n-hat(x) (n-hat from the stage-1 model, in-sample) with the same k-model as Mkn_g."""
    def __init__(self, backend='XGBoost', seed=0, cfg=None):
        super().__init__('Mkn_x', backend, seed, cfg)

    def n_rows(self, df):
        return np.clip(self.nm.predict(df[NONTIME].values), N_LO, N_HI)

    def fit(self, trn, n_src=None):
        S = series_table(trn if n_src is None else n_src)  # n-model source rows (default: the training rows)
        self.n = fit_n_within(trn); self.n_series = len(S)
        self.nm = n_model(self.seed).fit(S[NONTIME].values, np.clip(S.n.values, 0.1, 1.2))
        self.feats = NONTIME
        E = self.expo(trn); y = trn.Target.values / E
        if self.cfg['log']: y = np.log(y)
        self.m = base_model(self.backend, self.seed, NONTIME, None); self.m.fit(trn[NONTIME].values, y, sample_weight=(E ** 2) if self.cfg['w'] else None)
        return self


def split(d, variant):
    """same as p1b_extrapolation.split: 'last' = hold out last 1-2 times of series with >=4 times; 'x2' = test times >= 2x last train time"""
    test, excl, ratios, nser = [], [], {}, 0
    for sid, g in d.groupby('series'):
        s = np.sort(rnd(g.sqrt_t.unique()))
        if len(s) < 4: continue
        if variant == 'last':
            kh = int(np.clip((s > np.quantile(s, .6)).sum(), 1, 2)); held = s[-kh:]; tmax_tr = s[-kh - 1] ** 2
            tm = rnd(g.sqrt_t).isin(held); test += list(g.index[tm]); ratios.update({i: g.sqrt_t[i] ** 2 / tmax_tr for i in g.index[tm]}); nser += 1
        else:
            chosen = None
            for m in range(len(s) - 1, 1, -1):
                tm_ = s[m - 1] ** 2
                if (s[m:] ** 2 >= 2 * tm_ - 1e-9).any(): chosen = m; break
            if chosen is None: continue
            tm_ = s[chosen - 1] ** 2; tmask = rnd(g.sqrt_t) ** 2 >= 2 * tm_ - 1e-6
            emask = (rnd(g.sqrt_t) > s[chosen - 1] + 1e-9) & ~tmask
            test += list(g.index[tmask]); excl += list(g.index[emask]); nser += 1; ratios.update({i: g.sqrt_t[i] ** 2 / tm_ for i in g.index[tmask]})
    te = d.loc[sorted(test)]; trn = d.drop(index=sorted(set(test) | set(excl)))
    return trn, te, dict(n_series=nser, n_test=len(te), n_train=len(trn))


def m3(y, p):
    r = met(y, p); r['bias'] = float(np.mean(p - y)); return r
