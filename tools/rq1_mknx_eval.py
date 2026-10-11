"""RQ1b: Mkn_x (X = k(x) t^n-hat(x), two-stage) vs MX, Mk, Mkn_g, Mkn_r on: fixed test, 'last' and 'x2' extrapolation, combined study+temporal hold-out.
cfg (log-k, weighting) is taken from p1_extrapolation.json / p1_model_selection.json (already selected inside each training part) - no re-selection here."""
from rq_common import *
from sklearn.model_selection import GroupShuffleSplit
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = OUT + '\\results\\'
d, info = load(); d = d.reset_index(drop=True)
E1 = json.load(open(R + 'p1_extrapolation.json')); EC = json.load(open(R + 'p1_extrapolation_combined.json')); cfg_fixed = load_cfg()
KS = ['MX', 'Mk', 'Mkn_g', 'Mkn_r', 'Mkn_x']
mk = lambda k, s, cfg: KMX('XGBoost', s, cfg) if k == 'Mkn_x' else KM(k, 'XGBoost', s, cfg)
rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))


def run_models(trn, te, cfg, seeds=range(10)):
    out, P, nh = {}, {}, []
    for k in KS:
        L, preds = [], []
        for s in seeds:
            km = mk(k, s, cfg).fit(trn); p = km.predict(te); preds.append(p); L.append(m3(te.Target.values, p))
            if k == 'Mkn_x' and s == 0: nh = km.n_rows(te)
        P[k] = np.mean(preds, 0)
        out[k] = {m: dict(mean=float(np.mean([x[m] for x in L])), std=float(np.std([x[m] for x in L], ddof=1))) for m in ('rmse', 'mae', 'bias', 'r2')}
        nat = te.regime.values == 'natural'
        out[k]['natural_rmse'] = float(np.mean([rm(te.Target.values[nat], p[nat]) for p in preds])) if nat.any() else None
        out[k]['accelerated_rmse'] = float(np.mean([rm(te.Target.values[~nat], p[~nat]) for p in preds]))
        out[k]['rmse_of_seed_mean_prediction'] = rm(te.Target.values, P[k])
    return out, P, nh


def boot(te, P, cluster, B=1000, seed=0):
    rng = np.random.default_rng(seed); cl = te[cluster].astype(str).values; u = np.unique(cl); y = te.Target.values; res = {}
    idxs = [np.concatenate([np.where(cl == c)[0] for c in rng.choice(u, len(u))]) for _ in range(B)]
    for a in KS:
        for b in KS:
            if a == b or b not in ('MX', 'Mk', 'Mkn_g') or a != 'Mkn_x': continue
            dd = np.array([rm(y[i], P[a][i]) - rm(y[i], P[b][i]) for i in idxs])
            res[f'{a}_minus_{b}'] = dict(d_rmse=float(rm(y, P[a]) - rm(y, P[b])), ci95=[float(np.quantile(dd, .025)), float(np.quantile(dd, .975))], n_clusters=int(len(u)))
    return res


out = {}
# ---- fixed test ----
trn, te = d[~d.is_test].reset_index(drop=True), d[d.is_test].reset_index(drop=True)
res, P, nh = run_models(trn, te, cfg_fixed)
dom = {l: {k: rm(te.Target.values[(te.domain == l).values], P[k][(te.domain == l).values]) for k in KS} for l in ['D1', 'D2', 'D3', 'D4']}
out['fixed_test'] = dict(n_test=len(te), cfg=cfg_fixed, results=res, domain_rmse_seed_mean_pred=dom, bootstrap_study_cluster=boot(te, P, 'ref_group'), n_hat_test_rows=dict(mean=float(nh.mean()), min=float(nh.min()), max=float(nh.max()), natural_mean=float(nh[te.regime.values == 'natural'].mean())))
print('fixed', {k: round(v['rmse']['mean'], 3) for k, v in res.items()}, out['fixed_test']['bootstrap_study_cluster'], flush=True)
# ---- temporal ----
for variant in ['last', 'x2']:
    trn, te, inf = split(d, variant); cfg = E1['split_info'][variant]['cfg_selected_inside_train_only']
    res, P, nh = run_models(trn, te, cfg)
    out[variant] = dict(split=inf, cfg=cfg, results=res, bootstrap_series_cluster=boot(te, P, 'series'), n_hat_test_rows_mean=float(nh.mean()))
    print(variant, {k: round(v['rmse']['mean'], 3) for k, v in res.items()}, out[variant]['bootstrap_series_cluster'], flush=True)
# ---- combined ----
trn_l, te_l, _ = split(d, 'last'); H = set(te_l.index); gl = d.ref_group.astype(str).values; rows = []
for sd in range(10):
    _, tst = next(GroupShuffleSplit(1, test_size=.2, random_state=sd).split(d, groups=gl)); inT = d.ref_group.astype(str).isin(set(gl[tst])).values
    te_c, tr_c = d[inT & d.index.isin(H)], d[~inT & ~d.index.isin(H)]; cfg_c = EC['per_split'][sd]['cfg']
    row = dict(seed=sd, n_test=len(te_c), rmse={}, mae={})
    for k in KS:
        p = mk(k, sd, cfg_c).fit(tr_c).predict(te_c); row['rmse'][k] = rm(te_c.Target.values, p); row['mae'][k] = float(np.mean(np.abs(te_c.Target.values - p)))
    assert row['n_test'] == EC['per_split'][sd]['n_test']
    rows.append(row); print('combined', sd, {k: round(v, 2) for k, v in row['rmse'].items()}, flush=True)
out['combined'] = dict(per_split=rows, summary={k: dict(rmse_mean=float(np.mean([r['rmse'][k] for r in rows])), rmse_median=float(np.median([r['rmse'][k] for r in rows])), rmse_std=float(np.std([r['rmse'][k] for r in rows], ddof=1)),
                      d_vs_MX_mean=float(np.mean([r['rmse'][k] - r['rmse']['MX'] for r in rows])), splits_better_than_MX=int(sum(r['rmse'][k] < r['rmse']['MX'] for r in rows)),
                      d_vs_Mkn_g_median=float(np.median([r['rmse'][k] - r['rmse']['Mkn_g'] for r in rows])), splits_better_than_Mkn_g=int(sum(r['rmse'][k] < r['rmse']['Mkn_g'] for r in rows))) for k in KS})
savej(dict(note='Mkn_x vs baselines, XGB 10 seeds, fixed HPs. Mkn_x: shallow XGB n-hat(x) (depth 2) trained on series-level n of the TRAINING rows only, clipped to [%.1f,%.1f] inside t^n; k then fitted on X/t^n-hat with the same k-model/cfg as Mkn_g. cfg from p1_extrapolation*.json (selected inside the training parts). Bootstrap: cluster resampling of test studies (fixed test) or test series (temporal) of RMSE(Mkn_x)-RMSE(other) using the 10-seed mean predictions; negative = Mkn_x better. Combined = 10 GroupShuffleSplit seeds (study-grouped AND temporal hold-out).' % (N_LO, N_HI), results=out), 'rq1_mknx_eval_rq.json')

# ---- figure: bars for the existing 9 kinds (p1_extrapolation.json) + Mkn_x ----
plt.rcParams.update({'font.size': 7})
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.8), sharey=False)
for ax, (key, ttl) in zip(axs, [('last', '(a) "last" hold-out'), ('x2', '(b) "×2" hold-out'), ('fixed', '(c) fixed study-grouped test')]):
    labs, vals, errs = [], [], []
    for k in KINDS:
        r = E1['results'][f'{key}|XGBoost|{k}']['rmse'] if key != 'fixed' else json.load(open(R + 'p1_fixed_test.json'))['results'][f'XGBoost|{k}']['overall']['rmse']
        labs.append(LABEL[k]); vals.append(r['mean']); errs.append(r['std'])
    r = out[key]['results']['Mkn_x']['rmse'] if key != 'fixed' else out['fixed_test']['results']['Mkn_x']['rmse']
    labs.append('Mkn$_x$'); vals.append(r['mean']); errs.append(r['std'])
    cols = ['tab:gray'] * 3 + ['tab:blue', 'tab:green', 'tab:red', 'tab:cyan', 'tab:purple', 'tab:pink', 'k']
    ax.bar(range(len(vals)), vals, yerr=errs, color=cols[:len(vals)], capsize=1.5); ax.set_xticks(range(len(vals))); ax.set_xticklabels(labs, rotation=50, ha='right', fontsize=5.5); ax.set_title(ttl, fontsize=6)
axs[0].set_ylabel('RMSE (mm)'); fig.tight_layout(); fig.savefig(OUT + r'\plots_export\Fig_extrapolation_rmse_rq.png', dpi=300); plt.close(fig)
print('done')
