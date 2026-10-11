"""RQ1a: what controls the time exponent n of each series? Study-grouped CV of a shallow XGB n(x), interpretable OLS with study-clustered SE, SHAP + PDP."""
from rq_common import *
import statsmodels.api as sm
import shap
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
d, info = load(); d = d.reset_index(drop=True)
S = series_table(d)
ref = pd.read_csv(OUT + r'\..\03_aug_conformal_archived\results\step1_series_n.csv')
S['is_test'] = S.series.map(d.groupby('series').is_test.first())
y = S.n.values; X = S[NONTIME].values; grp = S.study.values; ustud = np.unique(grp)
print('series', len(S), 'studies', S.study.nunique(), 'matches step1_series_n.csv rows:', len(ref), flush=True)

# ---------------- study-grouped CV (5 folds, 10 random study->fold assignments) ----------------
def oof(make, rep):
    rng = np.random.default_rng(rep); perm = rng.permutation(ustud); fold = {s: i % 5 for i, s in enumerate(perm)}; f = np.array([fold[s] for s in grp]); p = np.zeros(len(y))
    for k in range(5):
        tr, te = f != k, f == k; p[te] = make(tr, te)
    return p


def r2(y, p): return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


Xl = np.column_stack([S['w/c'], S['FA'] / 100, np.log10(S['CO2']), S['RH'] / 10, S['T'] / 10, S['t_cure'] / 10])
models = {
    'XGB_shallow': lambda tr, te: n_model(0).fit(X[tr], np.clip(y[tr], 0.1, 1.2)).predict(X[te]),
    'OLS_linear': lambda tr, te: sm.OLS(y[tr], sm.add_constant(Xl[tr])).fit().predict(sm.add_constant(Xl[te], has_constant='add')),
    'train_mean': lambda tr, te: np.full(te.sum(), y[tr].mean()),
    'regime_mean': lambda tr, te: np.where(S.regime.values[te] == 'natural', y[tr & (S.regime.values == 'natural')].mean() if (tr & (S.regime.values == 'natural')).any() else y[tr].mean(), y[tr & (S.regime.values == 'accelerated')].mean()),
}
cv = {}
for nm, mk in models.items():
    R = []
    for rep in range(10):
        p = oof(mk, rep); R.append(dict(r2=r2(y, p), mae=float(np.mean(np.abs(y - p))), rmse=float(np.sqrt(np.mean((y - p) ** 2)))))
    cv[nm] = {m: dict(mean=float(np.mean([r[m] for r in R])), std=float(np.std([r[m] for r in R], ddof=1))) for m in ('r2', 'mae', 'rmse')}
    print(nm, {m: round(v['mean'], 3) for m, v in cv[nm].items()}, flush=True)
# fixed split (train studies -> test studies)
trm, tem = ~S.is_test.values, S.is_test.values
fixed = {nm: dict(r2=r2(y[tem], mk(trm, tem)), mae=float(np.mean(np.abs(y[tem] - mk(trm, tem)))), n_test_series=int(tem.sum())) for nm, mk in models.items()}

# ---------------- interpretable OLS, study-clustered SE ----------------
names = ['w/c (per 0.1)', 'FA (per 100 kg/m3)', 'log10 CO2 (per decade)', 'RH (per 10 %)', 'T (per 10 C)', 'curing (per 10 d)']
Xo = Xl.copy(); Xo[:, 0] = S['w/c'] / 0.1 if S['w/c'].max() < 5 else S['w/c'] / 10  # w/c is a fraction in the data
wcmax = float(S['w/c'].max())
res = sm.OLS(y, sm.add_constant(Xo)).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(grp)[0]}, use_t=True)
ci = res.conf_int()
ols = {nm: dict(coef=float(res.params[i + 1]), se=float(res.bse[i + 1]), ci95=[float(ci[i + 1][0]), float(ci[i + 1][1])], p=float(res.pvalues[i + 1])) for i, nm in enumerate(names)}
ols['const'] = dict(coef=float(res.params[0]), se=float(res.bse[0]))
uni = {}
for i, nm in enumerate(names):
    r_ = sm.OLS(y, sm.add_constant(Xo[:, i])).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(grp)[0]}, use_t=True); c_ = r_.conf_int()
    uni[nm] = dict(coef=float(r_.params[1]), ci95=[float(c_[1][0]), float(c_[1][1])], p=float(r_.pvalues[1]))
# robust variant: drop series with n outside [0.1, 1.2]
keep = (y >= 0.1) & (y <= 1.2); rr = sm.OLS(y[keep], sm.add_constant(Xo[keep])).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(grp[keep])[0]}, use_t=True)
ols_trim = {nm: dict(coef=float(rr.params[i + 1]), p=float(rr.pvalues[i + 1])) for i, nm in enumerate(names)}

# ---------------- SHAP + PDP on the XGB n-model fitted on all 278 series ----------------
nm_ = n_model(0).fit(X, np.clip(y, 0.1, 1.2)); sv = shap.TreeExplainer(nm_).shap_values(X); ma = np.abs(sv).mean(0)
low = y < 0.45
shap_low = {c: float(sv[low, i].mean()) for i, c in enumerate(NONTIME)}; shap_corr = {c: float(np.corrcoef(S[c].values, sv[:, i])[0, 1]) if np.std(S[c]) > 0 else 0.0 for i, c in enumerate(NONTIME)}
pdp = {}
fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.3)); rng = np.random.default_rng(0); ice = rng.choice(len(S), 40, replace=False)
for ax, c in zip(axs.ravel(), ['RH', 'CO2', 'w/c', 'FA', 't_cure', 'T']):
    j = NONTIME.index(c); gv = np.logspace(np.log10(max(S[c].min(), 0.03)), np.log10(S[c].max()), 25) if c == 'CO2' else np.linspace(S[c].quantile(.02), S[c].quantile(.98), 25)
    M = np.zeros((len(gv), len(S)))
    for i, v in enumerate(gv): Z = X.copy(); Z[:, j] = v; M[i] = nm_.predict(Z)
    for k in ice: ax.plot(gv, M[:, k], color='gray', alpha=.2, lw=.5)
    p_ = M.mean(1); ax.plot(gv, p_, 'r', lw=1.6); ax.axhline(0.5, color='k', ls=':', lw=.7)
    ax.scatter(S[c], np.full(len(S), 0.05), s=3, c='k', alpha=.25, marker='|'); ax.set_xlabel({'CO2': 'CO$_2$ (%)', 'RH': 'RH (%)', 'w/c': 'w/c', 'FA': 'FA (kg/m$^3$)', 't_cure': r'$t_{\mathrm{cure}}$ (d)', 'T': 'T (°C)'}[c]); ax.set_ylabel('$\\hat n$'); ax.set_ylim(0, 0.8)
    if c == 'CO2': ax.set_xscale('log')
    pdp[c] = dict(grid=gv.tolist(), pdp=p_.tolist(), end_minus_start=float(p_[-1] - p_[0]))
fig.tight_layout(); fig.savefig(OUT + r'\plots_export\Fig_n_drivers.png', dpi=300); plt.close(fig)

nat_ = S[S.regime == 'natural']; acc_ = S[S.regime == 'accelerated']
savej(dict(note='Series-level n = OLS slope of log X vs log t(days), series with >=3 distinct times and positive depths (%d series, %d studies; same definition as 03_aug_conformal_archived/results/step1_series_n.csv, which has %d rows). Inputs = 7 non-time features. XGB n-model: depth 2, 150 trees, lr 0.05, min_child_weight 3, lambda 5, n clipped to [0.1,1.2] in training. CV = 5-fold study-grouped, repeated 10x with random study-to-fold assignment, pooled out-of-fold R2/MAE/RMSE. OLS = standardised units listed, 95 percent CI and p from study-clustered covariance (t-dist, %d clusters). w/b omitted from OLS (collinear with w/c and FA).' % (len(S), S.study.nunique(), len(ref), S.study.nunique()),
           n_summary=dict(mean=float(y.mean()), median=float(np.median(y)), sd=float(y.std()), frac_between_0p35_0p65=float(((y > .35) & (y < .65)).mean()), frac_below_0p45=float((y < .45).mean()),
                          natural=dict(n_series=len(nat_), mean=float(nat_.n.mean()), sd=float(nat_.n.std())), accelerated=dict(n_series=len(acc_), mean=float(acc_.n.mean()), sd=float(acc_.n.std()))),
           cv_study_grouped=cv, fixed_split_train_to_test_studies=fixed, ols_clustered=ols, ols_univariate_clustered=uni, ols_trimmed_n_0p1_1p2=ols_trim, w_c_max=wcmax,
           shap=dict(mean_abs=dict(zip(NONTIME, map(float, ma))), rank=[NONTIME[i] for i in np.argsort(-ma)], mean_shap_for_series_with_n_below_0p45=shap_low, corr_feature_vs_shap=shap_corr), pdp=pdp), 'rq1_n_drivers_rq.json')
print('OLS', {k: (round(v['coef'], 3), [round(x, 3) for x in v['ci95']]) for k, v in ols.items() if 'ci95' in v})
print('SHAP rank', [NONTIME[i] for i in np.argsort(-ma)], shap_low); print('fixed', fixed)
