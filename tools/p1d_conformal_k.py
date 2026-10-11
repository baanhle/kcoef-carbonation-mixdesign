"""P1 (f) split conformal in k-space (calibration on held-out studies), mapped to X intervals widening as t^n. Saves artifact for Part 2."""
from models import *
import pickle
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
cfg = load_cfg(); sel = json.load(open(OUT + r'\results\p1_model_selection.json')); best = sel['best_k_model_by_val']
d, info = load()
tr, te = d[~d.is_test].reset_index(drop=True), d[d.is_test].reset_index(drop=True)
pt, cal, cs = calib_split(tr); pt, cal = pt.reset_index(drop=True), cal.reset_index(drop=True)
ALPHAS = [0.10, 0.05]


def stratify(te, lo, hi):
    y = te['Target'].values; cov = (y >= lo) & (y <= hi); w = hi - lo; out = {}
    masks = {'overall': np.ones(len(te), bool), 'natural': (te.regime == 'natural').values, 'accelerated': (te.regime == 'accelerated').values}
    for l in ['D1', 'D2', 'D3', 'D4']: masks[l] = (te.domain == l).values
    for k, m in masks.items(): out[k] = dict(n=int(m.sum()), coverage=float(cov[m].mean()), mean_width=float(w[m].mean()), median_width=float(np.median(w[m])))
    return out


def raw_target(km, df):
    y = df.Target.values / km.expo(df); return np.log(y) if km.cfg['log'] else y


def k_interval(km, praw, q):
    if km.cfg['log']: return np.exp(praw - q), np.exp(praw + q)
    return np.maximum(praw - q, 0), praw + q


def conf_k(models, kind, a):
    """models: list of fitted KM (ensemble if >1: raw outputs averaged). returns stratified metrics for plain + Mondrian."""
    km = models[0]
    pc = np.mean([m.raw_k(cal) for m in models], 0); pt_ = np.mean([m.raw_k(te) for m in models], 0)
    sc = np.abs(raw_target(km, cal) - pc); out = {}
    q = qhat(sc, a); lo, hi = k_interval(km, pt_, q); E = km.expo(te)
    out['split_k'] = dict(stratify(te, lo * E, hi * E), q_raw=q, mean_k_width=float((hi - lo).mean()))
    qs = {}
    for r in ['natural', 'accelerated']:
        m = (cal.regime == r).values; qr = qhat(sc[m], a) if m.sum() else np.inf; qs[r] = q if not np.isfinite(qr) else qr
    qv = np.array([qs[r] for r in te.regime.values]); lo2, hi2 = k_interval(km, pt_, qv)
    out['mondrian_k'] = dict(stratify(te, lo2 * E, hi2 * E), q_natural=qs['natural'], q_accelerated=qs['accelerated'])
    return out, (pc, pt_, q, qs, lo, hi)


def agg2(xs):
    if isinstance(xs[0], dict): return {k: agg2([x[k] for x in xs]) for k in xs[0]}
    a = np.array(xs, float); return dict(mean=float(a.mean()), std=float(a.std(ddof=1))) if len(a) > 1 else float(a[0])


res = {}
art = None
for kind in KKINDS:
    ms = [KM(kind, 'XGBoost', s, cfg).fit(pt) for s in range(10)]
    for a in ALPHAS:
        per = [conf_k([m], kind, a)[0] for m in ms]
        res[f'{kind}|per_seed|a{a}'] = agg2(per)
        e, extra = conf_k(ms, kind, a); res[f'{kind}|ensemble10|a{a}'] = e
        if kind == best and a == 0.10: art_extra = extra
    if kind == best: best_models = ms
# MX baseline (X-space abs residual, split)
mx = [KM('MX', 'XGBoost', s, cfg).fit(pt) for s in range(10)]
pcx, ptx = np.mean([m.predict(cal) for m in mx], 0), np.mean([m.predict(te) for m in mx], 0)
for a in ALPHAS:
    q = qhat(np.abs(cal.Target.values - pcx), a)
    res[f'MX|ensemble10|a{a}'] = dict(split_X=dict(stratify(te, np.maximum(ptx - q, 0), ptx + q), q=q))
pc, pte, q10, qs10, lo10, hi10 = art_extra
E = best_models[0].expo(te)
# artifact for part 2 (alpha 0.10 and 0.05)
art = dict(kind=best, cfg=cfg, models=best_models, n=best_models[0].n,
           q={str(a): qhat(np.abs(raw_target(best_models[0], cal) - np.mean([m.raw_k(cal) for m in best_models], 0)), a) for a in ALPHAS},
           q_mond={str(a): {r: (lambda sc, m: qhat(sc[m], a))(np.abs(raw_target(best_models[0], cal) - np.mean([m.raw_k(cal) for m in best_models], 0)), (cal.regime == r).values) for r in ['natural', 'accelerated']} for a in ALPHAS},
           n_cal=len(cal), n_cal_natural=int((cal.regime == 'natural').sum()), n_cal_studies=len(cs))
pickle.dump(art, open(OUT + r'\results\kmodel_artifact.pkl', 'wb'))
savej(dict(note='Split conformal in raw k-space (|k-khat|, or |log k - log khat| if log cfg), calibration = ~20% of train STUDIES (greedy smallest-first, 371 real rows), proper-train = rest; X interval = k-interval x t^n (width grows as t^n), coverage in X equals coverage in k. per_seed = mean/std over 10 seeds; ensemble10 = mean raw prediction of 10 seeds. MX baseline = X-space absolute-residual split conformal. Lower bound clipped at 0 in X for MX.',
           best_k_model=best, cfg=cfg, n_fitted=best_models[0].n, split=dict(n_proper_train=len(pt), n_cal=len(cal), n_cal_studies=len(cs), cal_regime=cal.regime.value_counts().to_dict()),
           q_artifact=art['q'], q_mond_artifact=art['q_mond'], results=res), 'p1_conformal_k.json')
print(json.dumps({k: v for k, v in res.items() if 'ensemble10' in k and 'a0.1' in k and 'MX' in k or (k.startswith(best) and 'ensemble10' in k and 'a0.1' in k)}, default=float)[:3000])

# ---- figure: parity in k-space and X-space with intervals ----
plt.rcParams.update({'font.size': 7})
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.3))
yk = te.Target.values / E; ptk = np.exp(pte) if cfg['log'] else pte
for r, c in [('accelerated', 'tab:red'), ('natural', 'tab:blue')]:
    m = (te.regime == r).values
    axs[0].errorbar(yk[m], ptk[m], yerr=[ptk[m] - lo10[m], hi10[m] - ptk[m]], fmt='o', ms=2, lw=.3, color=c, alpha=.45, label=r)
    axs[1].errorbar(te.Target.values[m], (ptk * E)[m], yerr=[(ptk * E - lo10 * E)[m], (hi10 * E - ptk * E)[m]], fmt='o', ms=2, lw=.3, color=c, alpha=.45, label=r)
for ax, nm, tag in zip(axs, ['$k$ (mm/d$^n$)', '$X$ (mm)'], ['(a)', '(b)']):
    ax.set_xlabel('Observed ' + nm); ax.set_ylabel('Predicted ' + nm); ax.set_title(tag, fontsize=7, loc='left')
axs[0].set_xscale('log'); axs[0].set_yscale('log'); axs[0].set_ylim(1e-2, 1e2)
lim = [min(yk.min(), ptk.min()), max(yk.max(), ptk.max())]; axs[0].plot(lim, lim, 'k--', lw=.7)
mx_ = te.Target.values.max(); axs[1].plot([0, mx_], [0, mx_], 'k--', lw=.7); axs[1].set_ylim(-5, 120)
s = res[f'{best}|ensemble10|a0.1']['split_k']['overall']
axs[1].text(0.03, 0.97, f"90% interval, coverage {s['coverage']:.2f}\nmean width {s['mean_width']:.1f} mm", transform=axs[1].transAxes, va='top', fontsize=6)
axs[0].legend(frameon=False, fontsize=6, loc='lower right'); fig.tight_layout(); fig.savefig(OUT + r'\plots_export\Fig_parity_k_conformal.png', dpi=300); plt.close(fig)
