"""P1 (a) fixed 473-row test, model selection on internal study-grouped validation, (c) repeated study-grouped 80/20 splits."""
from models import *
from sklearn.model_selection import GroupShuffleSplit
d, info = load()
tr, te = d[~d.is_test].reset_index(drop=True), d[d.is_test].reset_index(drop=True)
pt, cal, cal_studies = calib_split(tr); pt, cal = pt.reset_index(drop=True), cal.reset_index(drop=True)
rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))
# ---- selection (train only; validation = ~20% studies of train) ----
sel = {}
for lg in [False, True]:
    for w in [False, True]:
        r = [rm(cal.Target.values, KM('Mk', 'XGBoost', s, dict(log=lg, w=w)).fit(pt).predict(cal)) for s in range(3)]
        sel[f'log={lg},w={w}'] = float(np.mean(r))
best_key = min(sel, key=sel.get)
cfg = dict(log='log=True' in best_key, w='w=True' in best_key)
kinds_val = {k: float(np.mean([rm(cal.Target.values, KM(k, 'XGBoost', s, cfg).fit(pt).predict(cal)) for s in range(3)])) for k in KINDS}
CAND = ['Mk', 'Mkn_g', 'Mkn_r', 'Mkm']  # pre-specified candidates; sqrt(CO2)-normalised extensions (Mkc, Mkcn_g) are reported but excluded from selection
best_k = min(CAND, key=lambda k: kinds_val[k])
nfit = fit_n_within(tr)
savej(dict(note='Mk target k=X/sqrt_t; configs {raw k, log k} x {unweighted, weights sqrt_t^2 (=X-space loss)}; selected on X-space RMSE of study-grouped validation inside train (fit on proper-train, validate on calibration studies), 3 seeds. Chosen cfg is reused for Mkn.',
           cfg_val_rmse_Mk=sel, chosen_cfg=cfg, kinds_val_rmse_chosen_cfg=kinds_val, best_k_model_by_val=best_k,
           n_fitted_on_train=nfit, n_fit_method='within-series OLS of log X on log t(days) (series fixed effects), series with >=2 distinct times',
           mono_constraints=MONO, mono_note='MX-mono(t): +1 on sqrt_t only. MX-mono(full)=MXm3: +w/b, +w/c, +CO2, +T, -t_cure, +sqrt_t; RH and FA unconstrained. Mk-mono (Mkm): same on the 7 non-time inputs (no sqrt_t, k-space). Mk (non-mono) shown as Mk. RF has no monotone option (not run).',
           n_val=len(cal), n_proper_train=len(pt)), 'p1_model_selection.json')
print(sel, cfg, kinds_val, best_k, nfit, flush=True)

# ---- (a) fixed test ----
out = {}
for backend, seeds, kinds in [('XGBoost', range(10), KINDS), ('RandomForest', range(5), ['MX', 'Mk', 'Mkn_g', 'Mkn_r', 'Mkc', 'Mkcn_g'])]:
    for k in kinds:
        L = [strat_met(te, KM(k, backend, s, cfg).fit(tr).predict(te)) for s in seeds]
        out[f'{backend}|{k}'] = agg(L)
        print(backend, k, out[f'{backend}|{k}']['overall']['rmse'], flush=True)
savej(dict(note='Fixed study-grouped test (473 real rows), train 1690 rows; mean/std over model seeds (XGB 10, RF 5); fixed task1b HPs.', cfg=cfg, n_fitted_on_train=nfit, results=out), 'p1_fixed_test.json')

# ---- (c) repeated study-grouped 80/20, 20 seeds ----
Xd = d.reset_index(drop=True); g = Xd.ref_group.astype(str).values
R = {k: [] for k in KINDS}
for seed in range(20):
    tri, tei = next(GroupShuffleSplit(1, test_size=.2, random_state=seed).split(Xd, groups=g))
    a, b = Xd.iloc[tri], Xd.iloc[tei]
    for k in KINDS:
        R[k].append(met(b.Target.values, KM(k, 'XGBoost', seed, cfg).fit(a).predict(b)))
summ = {k: {m: dict(mean=float(np.mean([x[m] for x in v])), std=float(np.std([x[m] for x in v], ddof=1)), median=float(np.median([x[m] for x in v]))) for m in ['rmse', 'mae', 'r2']} for k, v in R.items()}
paired = {k: dict(d_rmse_mean=float(np.mean([a['rmse'] - b['rmse'] for a, b in zip(R[k], R['MX'])])), d_rmse_std=float(np.std([a['rmse'] - b['rmse'] for a, b in zip(R[k], R['MX'])], ddof=1)),
                  seeds_better=int(sum(a['rmse'] < b['rmse'] for a, b in zip(R[k], R['MX'])))) for k in KINDS if k != 'MX'}
savej(dict(note='GroupShuffleSplit(test 20% of studies) x 20 seeds on all 2163 rows, XGBoost; paired = per-split RMSE difference vs MX (negative = better).', summary=summ, paired_vs_MX=paired, per_seed=R), 'p1_repeated_split.json')
print(summ, paired)
