"""RQ1 extras: (1) n variance decomposition (MixedLM, random intercept per study); (2) Mkn_x on the 20 repeated study-grouped splits of p1a (same GroupShuffleSplit seeds, cfg)."""
from rq_common import *
import statsmodels.formula.api as smf, warnings
from sklearn.model_selection import GroupShuffleSplit
warnings.filterwarnings('ignore')
d, info = load(); d = d.reset_index(drop=True); S = series_table(d)
S['lco2'] = np.log10(S.CO2); S['wc'] = S['w/c']; S['FA100'] = S.FA / 100; S['RH10'] = S.RH / 10; S['T10'] = S['T'] / 10; S['tc10'] = S.t_cure / 10
def icc(f):
    r = smf.mixedlm(f, S, groups=S.study).fit(reml=True); vu, ve = float(r.cov_re.iloc[0, 0]), float(r.scale)
    X = r.model.exog; vf = float(np.var(X @ r.fe_params.values)); tot = vf + vu + ve
    return dict(var_study=vu, var_resid=ve, icc=vu / (vu + ve), var_fixed=vf, marginal_R2=vf / tot, conditional_R2=(vf + vu) / tot, converged=bool(r.converged)), r
null, _ = icc('n ~ 1'); fx, rf = icc('n ~ lco2 + wc + FA100 + RH10 + T10 + tc10')
fe = {k: dict(coef=float(rf.fe_params[k]), p=float(rf.pvalues[k])) for k in rf.fe_params.index}
st = S.groupby('study').n.agg(['count', 'mean']).reset_index()
savej(dict(note='MixedLM (REML) per-series n, random intercept per study. ICC = var_study/(var_study+var_resid). Fixed effects: log10 CO2, w/c, FA/100, RH/10, T/10, t_cure/10. marginal R2 = var(fixed part)/(var_fixed+var_study+var_resid) (Nakagawa). Note: studies with one series contribute to resid only; fixed-effect variables are largely study-constant so ICC after adjustment is not a clean test.', n_series=len(S), n_studies=int(S.study.nunique()), series_per_study=dict(min=int(st['count'].min()), median=float(st['count'].median()), max=int(st['count'].max())), null=null, with_fixed=fx, fixed_effects=fe, icc_change=fx['icc'] - null['icc']), 'rq1_n_variance_rq.json')
print('null', null, '\nfixed', fx)
cfg = load_cfg(); g = d.ref_group.astype(str).values; R = json.load(open(OUT + '/results/p1_repeated_split.json')); rows = []
for seed in range(20):
    tri, tei = next(GroupShuffleSplit(1, test_size=.2, random_state=seed).split(d, groups=g)); a, b = d.iloc[tri], d.iloc[tei]
    r = m3(b.Target.values, KMX('XGBoost', seed, cfg).fit(a).predict(b))['rmse']; rows.append(r)
rx = np.array(rows); ref = {k: np.array([x['rmse'] for x in R['per_seed'][k]]) for k in ('MX', 'Mk', 'Mkn_g')}
out = dict(Mkn_x=dict(rmse_mean=float(rx.mean()), rmse_std=float(rx.std(ddof=1)), rmse_median=float(np.median(rx)), per_split=rx.tolist()))
for k, v in ref.items():
    out[k] = dict(rmse_mean=float(v.mean()), rmse_std=float(v.std(ddof=1)), rmse_median=float(np.median(v)))
    out['Mkn_x'][f'beats_{k}'] = int((rx < v).sum()); out['Mkn_x'][f'd_vs_{k}_mean'] = float((rx - v).mean())
savej(dict(note='Mkn_x (XGB, same cfg as p1a repeated splits) on GroupShuffleSplit(test 20% studies, random_state=seed) x 20 seeds, all 2163 rows; baselines read from p1_repeated_split.json (same splits, same seeds).', results=out), 'rq1_mknx_repeated_rq.json')
print({k: {m: (round(x, 3) if isinstance(x, float) else x) for m, x in v.items() if m != 'per_split'} for k, v in out.items()})
