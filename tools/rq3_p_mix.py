"""RQ3 extension: is the accelerated->natural conversion exponent p mix-dependent?  Row-level implied p on the 119 natural rows (8 studies)."""
from rq_common import *
import statsmodels.api as sm
from scipy.optimize import minimize_scalar, minimize
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
cfg = load_cfg(); d, info = load(); d = d.reset_index(drop=True)
acc, nat = d[d.regime == 'accelerated'].reset_index(drop=True), d[d.regime == 'natural'].reset_index(drop=True)
ref = nat.copy(); ref['CO2'] = 20.0
y, lr, g = nat.Target.values, np.log(nat.CO2.values / 20.0), nat.ref_group.astype(str).values; studies = np.unique(g)
B = []
for s in range(10):
    km = KM('Mkn_g', 'XGBoost', s, cfg).fit(acc); B.append(km.predict_k(ref) * km.expo(nat))
base = np.mean(B, 0)  # X predicted at CO2 = 20 % for each natural row (accelerated-only model, global n)
imp = np.log(y / base) / lr  # implied p per row
nat = nat.assign(p_imp=imp, scm=1 - nat['w/b'] / nat['w/c'], FA100=nat['FA'] / 100, wb10=nat['w/b'] / .1, RH10=nat['RH'] / 10, tc10=nat['t_cure'] / 10, T10=nat['T'] / 10)
cl = pd.factorize(g)[0]
print('implied p: mean %.3f median %.3f sd %.3f; per-study median' % (imp.mean(), np.median(imp), imp.std()), flush=True)
per_study = {s: dict(n=int((g == s).sum()), median_p=float(np.median(imp[g == s])), mean_p=float(imp[g == s].mean()), FA=float(nat.FA[g == s].mean()), wb=float(nat['w/b'][g == s].mean()), RH=float(nat.RH[g == s].mean()), t_cure=float(nat.t_cure[g == s].mean())) for s in studies}


def ols(cols):
    r = sm.OLS(nat.p_imp.values, sm.add_constant(nat[cols].values)).fit(cov_type='cluster', cov_kwds={'groups': cl}, use_t=True); ci = r.conf_int()
    return {c: dict(coef=float(r.params[i + 1]), se=float(r.bse[i + 1]), ci95=[float(ci[i + 1][0]), float(ci[i + 1][1])], p=float(r.pvalues[i + 1])) for i, c in enumerate(cols)} | dict(const=float(r.params[0]), r2=float(r.rsquared), n_clusters=int(len(studies)))


uni = {c: ols([c]) for c in ['FA100', 'scm', 'wb10', 'w/c', 'RH10', 'tc10', 'T10']}
multi = ols(['FA100', 'wb10', 'RH10', 'tc10'])
print('FA univariate', uni['FA100']['FA100'], flush=True)


def fitp(idx, feat=None):
    """LS fit in X space of X = base * r^(p0 + p1*feat); feat None -> constant p"""
    if feat is None:
        f = lambda p: np.sum((base[idx] * np.exp(p[0] * lr[idx]) - y[idx]) ** 2); r = minimize_scalar(lambda p: f([p]), bounds=(-.5, 1.5), method='bounded'); return np.array([r.x, 0.0])
    f = lambda p: np.sum((base[idx] * np.exp((p[0] + p[1] * feat[idx]) * lr[idx]) - y[idx]) ** 2)
    return minimize(f, [0.3, 0.0], method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-9, maxiter=2000)).x


def predict(par, idx, feat):
    p = par[0] + (par[1] * feat[idx] if feat is not None else 0); return base[idx] * np.exp(p * lr[idx])


allidx = np.arange(len(y)); rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
out = {}
for fname, feat in [('const', None), ('FA100', nat.FA100.values), ('scm', nat.scm.values), ('wb10', nat.wb10.values)]:
    par = fitp(allidx, feat); pred = np.zeros(len(y)); fold = {}
    for s in studies:
        tr, te = np.where(g != s)[0], np.where(g == s)[0]; pr = fitp(tr, feat); pred[te] = predict(pr, te, feat); fold[s] = rm(y[te], pred[te])
    rng = np.random.default_rng(0); bs = []
    for _ in range(1000):
        idx = np.concatenate([np.where(g == s)[0] for s in rng.choice(studies, len(studies))]); bs.append(fitp(idx, feat))
    bs = np.array(bs)
    out[fname] = dict(p0=float(par[0]), p1=float(par[1]), in_sample_rmse=rm(y, predict(par, allidx, feat)), loso_rmse=rm(y, pred), loso_rmse_per_study=fold,
                      p0_ci95_study_bootstrap=[float(np.quantile(bs[:, 0], .025)), float(np.quantile(bs[:, 0], .975))], p1_ci95_study_bootstrap=[float(np.quantile(bs[:, 1], .025)), float(np.quantile(bs[:, 1], .975))] if feat is not None else None)
    print(fname, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[fname].items() if k in ('p0', 'p1', 'in_sample_rmse', 'loso_rmse')}, out[fname]['p1_ci95_study_bootstrap'], flush=True)
for f in ('FA100', 'scm', 'wb10'):
    out[f]['studies_better_than_const_loso'] = int(sum(out[f]['loso_rmse_per_study'][s] < out['const']['loso_rmse_per_study'][s] for s in studies))

# figure
plt.rcParams.update({'font.size': 7}); fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.7), sharey=True); cm = plt.cm.tab10(np.arange(len(studies)))
for ax, (c, lab) in zip(axs, [('FA', 'FA (kg/m$^3$)'), ('w/b', 'w/b'), ('RH', 'RH (%)')]):
    for i, s in enumerate(studies): m = g == s; ax.scatter(nat[c][m], nat.p_imp[m], s=8, color=cm[i], label=s.strip('[]'))
    ax.axhline(out['const']['p0'], color='k', ls='--', lw=.8); ax.set_xlabel(lab)
xs = np.linspace(0, nat.FA.max(), 50); axs[0].plot(xs, out['FA100']['p0'] + out['FA100']['p1'] * xs / 100, 'r-', lw=1, label='$p_0+p_1$FA')
axs[0].set_ylabel('row-level implied $p$'); axs[0].set_ylim(np.quantile(imp, .01) - .05, np.quantile(imp, .99) + .05); axs[2].legend(fontsize=4.5, frameon=False, ncol=2, title='study', title_fontsize=5)
fig.suptitle('Natural rows (119, 8 studies): implied conversion exponent; dashed = constant $\\hat p$ (study bootstrap CIs in rq3_p_mix_rq.json)', fontsize=6); fig.tight_layout(); fig.savefig(OUT + r'\plots_export\Fig_p_vs_mix.png', dpi=300); plt.close(fig)
savej(dict(note='Implied p per natural row = ln(X_obs / X_k(CO2=20 %)) / ln(CO2/20), X_k from the accelerated-only Mkn_g 10-seed ensemble (global n) evaluated at each row\'s own mix/T/RH; ln(CO2/20) is about -6, so noise in X_obs/X_k is divided by ~6 but row-level p is still heavy-tailed. Regressions: OLS of implied p on the listed covariates with study-clustered covariance (t-dist, 8 clusters; very low power). scm = 1 - (w/b)/(w/c) = non-clinker share of binder (includes slag etc., not FA only). Parametric models p = p0 + p1*feature fitted by least squares in X space; LOSO = leave-one-natural-study-out RMSE (pooled over rows). Only 2 of 8 natural studies contain FA (>0); 1 study (id [69]) has t_cure 10.7 d, so FA, curing and study effects are confounded.',
           n_rows=len(nat), n_studies=len(studies), implied_p=dict(mean=float(imp.mean()), median=float(np.median(imp)), sd=float(imp.std()), q05=float(np.quantile(imp, .05)), q95=float(np.quantile(imp, .95))), per_study=per_study,
           ols_univariate_clustered=uni, ols_multivariate_clustered=multi, parametric_models=out), 'rq3_p_mix_rq.json')
print('done')
