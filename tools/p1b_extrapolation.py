"""P1 (b) temporal extrapolation within series (>=4 distinct times). Two variants: 'last' (hold out last 1-2 times above series 60th pct time)
and 'x2' (test times >= 2x last training time)."""
from models import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
cfg0 = load_cfg()
d, info = load(); d = d.reset_index(drop=True)
rnd = lambda x: np.round(x, 6)


def split(d, variant):
    test, excl, ratios, nser = [], [], {}, 0
    for sid, g in d.groupby('series'):
        s = np.sort(rnd(g.sqrt_t.unique()))
        if len(s) < 4: continue
        if variant == 'last':
            kh = int(np.clip((s > np.quantile(s, .6)).sum(), 1, 2))
            held = s[-kh:]; tmax_tr = s[-kh - 1] ** 2
            tm = rnd(g.sqrt_t).isin(held)
            test += list(g.index[tm]); ratios.update({i: g.sqrt_t[i] ** 2 / tmax_tr for i in g.index[tm]}); nser += 1
        else:
            chosen = None
            for m in range(len(s) - 1, 1, -1):  # m = number of training times (>=2)
                tm_ = s[m - 1] ** 2
                if (s[m:] ** 2 >= 2 * tm_ - 1e-9).any(): chosen = m; break
            if chosen is None: continue
            tm_ = s[chosen - 1] ** 2
            tmask = rnd(g.sqrt_t) ** 2 >= 2 * tm_ - 1e-6
            emask = (rnd(g.sqrt_t) > s[chosen - 1] + 1e-9) & ~tmask
            test += list(g.index[tmask]); excl += list(g.index[emask]); nser += 1
            ratios.update({i: g.sqrt_t[i] ** 2 / tm_ for i in g.index[tmask]})
    te = d.loc[sorted(test)]; trn = d.drop(index=sorted(set(test) | set(excl)))
    r = np.array([ratios[i] for i in te.index])
    return trn, te, dict(n_series=nser, n_test=len(te), n_excluded_from_train=len(excl), n_train=len(trn), n_test_natural=int((te.regime == 'natural').sum()),
                         ratio_median=float(np.median(r)), ratio_q25=float(np.quantile(r, .25)), ratio_q75=float(np.quantile(r, .75)), ratio_max=float(r.max()), ratio_min=float(r.min()),
                         test_studies=int(te.ref_group.nunique()), te_idx=None), r



def cfg_select(trn):
    """cfg for Mk-kinds chosen by X-RMSE on a study-grouped validation carved from trn only (held-out points are not in trn)."""
    pt_, cal_, _ = calib_split(trn); pt_, cal_ = pt_.reset_index(drop=True), cal_.reset_index(drop=True)
    sc = {}
    for lg in [False, True]:
        for w in [False, True]:
            sc[(lg, w)] = np.mean([np.sqrt(np.mean((cal_.Target.values - KM('Mk', 'XGBoost', s_, dict(log=lg, w=w)).fit(pt_).predict(cal_)) ** 2)) for s_ in range(3)])
    b = min(sc, key=sc.get); return dict(log=b[0], w=b[1])


def conf_cov(trn, te, cfg, kinds=('MX', 'Mk', 'Mkn_g', 'Mkm'), a=0.10):
    """split conformal with calibration = ~20% studies of trn (held-out points excluded); coverage/width on extrapolated test points; mean of 3 seeds"""
    pt_, cal_, _ = calib_split(trn); pt_, cal_ = pt_.reset_index(drop=True), cal_.reset_index(drop=True); out = {}
    for k in kinds:
        cv, wd = [], []
        for s_ in range(3):
            km = KM(k, 'XGBoost', s_, cfg).fit(pt_)
            if k == 'MX':
                q = qhat(np.abs(cal_.Target.values - km.predict(cal_)), a); p = km.predict(te); lo, hi = np.maximum(p - q, 0), p + q
            else:
                yr = cal_.Target.values / km.expo(cal_); yr = np.log(yr) if cfg['log'] else yr
                q = qhat(np.abs(yr - km.raw_k(cal_)), a); pr = km.raw_k(te); E = km.expo(te)
                lo, hi = ((np.exp(pr - q), np.exp(pr + q)) if cfg['log'] else (np.maximum(pr - q, 0), pr + q)); lo, hi = lo * E, hi * E
            cv.append(float(((te.Target.values >= lo) & (te.Target.values <= hi)).mean())); wd.append(float((hi - lo).mean()))
        out[k] = dict(coverage=float(np.mean(cv)), mean_width=float(np.mean(wd)))
    return out

def m3(y, p):
    r = met(y, p); r['bias'] = float(np.mean(p - y)); r['mape'] = float(np.mean(np.abs(p - y) / np.maximum(y, 1e-6)) * 100); return r


res, infos, store = {}, {}, {}
for variant in ['last', 'x2']:
    trn, te, inf, ratios = split(d, variant); inf.pop('te_idx'); infos[variant] = inf; print(variant, inf, flush=True)
    cfg = cfg_select(trn); inf['cfg_selected_inside_train_only'] = cfg; inf['conformal_90_on_extrapolated_pts'] = conf_cov(trn, te, cfg)
    for backend, seeds, kinds in [('XGBoost', range(10), KINDS), ('RandomForest', range(5), ['MX', 'Mk', 'Mkn_g', 'Mkn_r', 'Mkc', 'Mkcn_g'])]:
        for k in kinds:
            L, Lr = [], []
            for s in seeds:
                km = KM(k, backend, s, cfg).fit(trn); p = km.predict(te)
                L.append(m3(te.Target.values, p))
                nm = te.regime.values == 'natural'
                Lr.append(dict(natural=m3(te.Target.values[nm], p[nm]) if nm.any() else None, accelerated=m3(te.Target.values[~nm], p[~nm])))
                if backend == 'XGBoost' and s == 0: store[(variant, k)] = km
            f = lambda lst, key: dict(mean=float(np.mean([x[key] for x in lst])), std=float(np.std([x[key] for x in lst], ddof=1)) if len(lst) > 1 else 0.0)
            res[f'{variant}|{backend}|{k}'] = dict({m: f(L, m) for m in ['rmse', 'mae', 'bias', 'mape', 'r2']},
                                                    natural={m: f([x['natural'] for x in Lr], m) for m in ['rmse', 'mae', 'bias']} if inf['n_test_natural'] else None,
                                                    accelerated={m: f([x['accelerated'] for x in Lr], m) for m in ['rmse', 'mae', 'bias']})
            print(variant, backend, k, round(res[f'{variant}|{backend}|{k}']['rmse']['mean'], 3), flush=True)
    # paired vs MX (XGB) per seed handled via differences of means over seeds is enough; also bootstrap over series-clusters
    tr_te = (trn, te, ratios)
    store[(variant, 'data')] = tr_te
# ---- combined: study-grouped AND temporal hold-out ----
from sklearn.model_selection import GroupShuffleSplit
trn_l, te_l, _, _ = split(d, 'last'); H = set(te_l.index)
comb, gl = [], d.ref_group.astype(str).values
for sd in range(10):
    _, tst = next(GroupShuffleSplit(1, test_size=.2, random_state=sd).split(d, groups=gl)); tst_studies = set(gl[tst])
    inT = d.ref_group.astype(str).isin(tst_studies).values
    te_c = d[inT & d.index.isin(H)]; tr_c = d[~inT & ~d.index.isin(H)]
    cfg_c = cfg_select(tr_c); rowc = dict(seed=sd, n_test=len(te_c), n_test_studies=int(te_c.ref_group.nunique()), n_train=len(tr_c), cfg=cfg_c, rmse={}, mae={})
    for k in KINDS:
        p = KM(k, 'XGBoost', sd, cfg_c).fit(tr_c).predict(te_c); mm = met(te_c.Target.values, p); rowc['rmse'][k] = mm['rmse']; rowc['mae'][k] = mm['mae']
    rowc['conformal_90'] = conf_cov(tr_c, te_c, cfg_c); comb.append(rowc); print('combined', sd, rowc['n_test'], {k: round(v, 2) for k, v in rowc['rmse'].items()}, flush=True)
csum = {k: dict(rmse_mean=float(np.mean([r['rmse'][k] for r in comb])), rmse_std=float(np.std([r['rmse'][k] for r in comb], ddof=1)), rmse_median=float(np.median([r['rmse'][k] for r in comb])),
                mae_mean=float(np.mean([r['mae'][k] for r in comb])), d_vs_MX_mean=float(np.mean([r['rmse'][k] - r['rmse']['MX'] for r in comb])), d_vs_MX_std=float(np.std([r['rmse'][k] - r['rmse']['MX'] for r in comb], ddof=1)),
                splits_better_than_MX=int(sum(r['rmse'][k] < r['rmse']['MX'] for r in comb))) for k in KINDS}
csum_cov = {k: dict(coverage_mean=float(np.mean([r['conformal_90'][k]['coverage'] for r in comb])), coverage_std=float(np.std([r['conformal_90'][k]['coverage'] for r in comb], ddof=1)),
                    width_mean=float(np.mean([r['conformal_90'][k]['mean_width'] for r in comb]))) for k in ('MX', 'Mk', 'Mkn_g', 'Mkm')}
savej(dict(note='COMBINED variant: 20% of studies held out (GroupShuffleSplit, 10 seeds) AND the last 1-2 times of every eligible series removed from training; test = late points of held-out-study series. cfg and conformal calibration use training rows only (no held-out points).',
           per_split=comb, summary=csum, conformal_summary=csum_cov), 'p1_extrapolation_combined.json')

savej(dict(note='Within-series temporal extrapolation on all 2163 rows (series with >=4 distinct times). Train = everything not in test (variant x2 also drops rows between last train time and 2x). Fixed HPs, XGB 10 seeds / RF 5 seeds.',
           cfg_note='cfg for Mk-kinds re-selected inside each variant on a study-grouped validation carved from the TRAINING part only (held-out time points excluded from model fitting, validation and conformal calibration); see split_info.*.cfg_selected_inside_train_only', split_info=infos, results=res), 'p1_extrapolation.json')

# ---- paired cluster-bootstrap difference vs MX, XGB seed 0 ----
rng = np.random.default_rng(0); boot = {}
for variant in ['last', 'x2']:
    trn, te, ratios = store[(variant, 'data')]
    pr = {k: store[(variant, k)].predict(te) for k in KINDS}
    ser = te.series.values; us = np.unique(ser)
    for k in KINDS:
        if k == 'MX': continue
        diffs = []
        for _ in range(500):
            sel = rng.choice(us, len(us)); idx = np.concatenate([np.where(ser == u)[0] for u in sel])
            y = te.Target.values[idx]
            diffs.append(np.sqrt(np.mean((y - pr[k][idx]) ** 2)) - np.sqrt(np.mean((y - pr['MX'][idx]) ** 2)))
        boot[f'{variant}|{k}'] = dict(d_rmse_vs_MX_mean=float(np.mean(diffs)), ci95=[float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975))])
savej(dict(note='series-cluster bootstrap (500) of RMSE(model)-RMSE(MX), XGB seed 0; negative = better than MX', boot=boot), 'p1_extrapolation_bootstrap.json')

# ---- figures ----
plt.rcParams.update({'font.size': 7})
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
cols = ['tab:gray', 'tab:olive', 'tab:brown', 'tab:blue', 'tab:green', 'tab:red', 'tab:cyan', 'tab:purple', 'tab:pink']
for ax, variant in zip(axs, ['last', 'x2']):
    for i, k in enumerate(KINDS):
        r = res[f'{variant}|XGBoost|{k}']['rmse']; ax.bar(i, r['mean'], yerr=r['std'], color=cols[i], capsize=2)
    ax.set_xticks(range(len(KINDS))); ax.set_xticklabels([LABEL[k] for k in KINDS], rotation=40, ha='right', fontsize=5.5)
    ax.set_title(f"{'hold out last 1-2 times' if variant == 'last' else 'test times >= 2x last train time'}  (n={infos[variant]['n_test']}, median ratio {infos[variant]['ratio_median']:.1f})", fontsize=6)
axs[0].set_ylabel('RMSE, extrapolated pts (mm)'); fig.tight_layout()
fig.savefig(OUT + r'\plots_export\Fig_extrapolation_rmse.png', dpi=300); plt.close(fig)

trn, te, _ = store[('last', 'data')]
cand = []
for sid, g in d.groupby('series'):
    if g.sqrt_t.round(6).nunique() >= 5 and sid in set(te.series): cand.append((g.regime.iloc[0], g.sqrt_t.round(6).nunique(), sid))
acc = sorted([c for c in cand if c[0] == 'accelerated'], key=lambda c: -c[1])
natl = sorted([c for c in cand if c[0] == 'natural'], key=lambda c: -c[1])
# pick spread-out accelerated series (different studies) + natural
picked, seen = [], set()
for c in natl[:2] + acc:
    st = d[d.series == c[2]].ref_group.iloc[0]
    if st in seen: continue
    seen.add(st); picked.append(c)
    if len(picked) == 6: break
fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.4))
for ax, (rg, nd, sid) in zip(axs.ravel(), picked):
    g = d[d.series == sid]; ist = g.index.isin(te.index)
    sg = np.linspace(0.05, 2 * g.sqrt_t.max(), 120)
    G = pd.concat([g.iloc[[0]]] * len(sg), ignore_index=True); G['sqrt_t'] = sg
    for k, c, ls in [('MX', 'tab:gray', '-'), ('MXm3', 'tab:brown', '--'), ('Mk', 'tab:blue', '-'), ('Mkn_r', 'tab:red', ':'), ('Mkc', 'tab:cyan', '-.')]:
        ax.plot(sg ** 2, store[('last', k)].predict(G), color=c, ls=ls, lw=1, label=LABEL[k])
    ax.scatter(g.sqrt_t[~ist] ** 2, g.Target[~ist], c='k', s=12, zorder=3, label='train')
    ax.scatter(g.sqrt_t[ist] ** 2, g.Target[ist], c='m', marker='s', s=16, zorder=3, label='held out')
    ax.set_title(f"{rg}, CO$_2$={g['CO2'].iloc[0]:g}%, w/b={g['w/b'].iloc[0]:.2f}", fontsize=6); ax.set_xlabel('t (days)'); ax.set_ylabel('depth (mm)')
axs[0, 0].legend(fontsize=5, frameon=False); fig.tight_layout()
fig.savefig(OUT + r'\plots_export\Fig_extrapolation_examples.png', dpi=300); plt.close(fig)
print('done')
