"""Part 2: NSGA-II mix design (durability vs embodied CO2), conformal upper-bound depth, data-support constraint.
Variants: main (B fixed 350, w/b<=0.55), wb050 (B fixed 350, w/b<=0.50), Bfree (B in [300,450], w/b<=0.55; appendix).
Rerun cheaply after editing tools/config_ef.json:  python p2_nsga.py"""
import pickle, sys, itertools
from models import *
from scipy.spatial import cKDTree
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = OUT + '\\results\\'
EF = json.load(open(OUT + r'\tools\config_ef.json'))
base_ef = {k: EF[k] for k in ('EF_c', 'EF_fa', 'EF_w')}
art = pickle.load(open(R + 'kmodel_artifact.pkl', 'rb'))
models, KM0 = art['models'], art['models'][0]
d, info = load()
tr = d[~d.is_test].reset_index(drop=True)
Q = art['q']['0.1']; QM = art['q_mond']['0.1']
LOGK = art['cfg']['log']
B_FIX = 350.0
print('k-model:', art['kind'], 'cfg', art['cfg'], 'n', art['n'], 'q', Q, flush=True)

# ---------------- condition A support check ----------------
def count(df, co2=(15, 25), rh=(60, 75)):
    m = df.CO2.between(*co2) & df.RH.between(*rh)
    return dict(rows=int(m.sum()), studies=int(df[m].ref_group.nunique()), T_values=df[m]['T'].round(0).value_counts().head(5).to_dict())
sup = dict(all_data=count(d), train=count(tr))
m_ = d.CO2.between(15, 25) & d.RH.between(60, 75) & d['T'].between(18, 22)
sup['all_data_T18_22'] = dict(rows=int(m_.sum()), studies=int(d[m_].ref_group.nunique()))
mt = tr.CO2.between(15, 25) & tr.RH.between(60, 75) & tr['T'].between(18, 22)
sup['train_T18_22'] = dict(rows=int(mt.sum()), studies=int(tr[mt].ref_group.nunique()), sqrt_t_3_to_8=int((mt & tr.sqrt_t.between(3, 8)).sum()))
good = sup['train_T18_22']['rows'] >= 100 and sup['train_T18_22']['studies'] >= 5
COND = dict(CO2=20.0, T=20.0, RH=70.0, sqrt_t=float(np.sqrt(28.0)), t_cure=28.0)
sup['decision'] = 'condition A (CO2 20 %, T 20 C, RH 70 %, 28 d) kept: well supported (>=100 train rows, >=5 studies at T 18-22 C)' if good else 'POOR support'
if not good: sys.exit('support poor; add fallback condition')
savej(sup, 'p2_support_check.json')

# ---------------- kNN support in standardized 7-D non-time inputs ----------------
U = tr.drop_duplicates(NONTIME)[NONTIME].reset_index(drop=True)
mu_, sd_ = U.mean().values, U.std().values
Z = (U.values - mu_) / sd_; tree = cKDTree(Z)
dd, _ = tree.query(Z, k=6); thr = float(np.quantile(dd[:, 5], .95))  # 6th = 5th neighbour excluding self
zmin, zmax = Z.min(0), Z.max(0)
support_info = dict(n_unique_train_mixes=len(U), knn_k=5, knn_thr_p95=thr, knn_median_within_train=float(np.median(dd[:, 5])),
                    note='reference set = unique non-time input vectors of the 1690 train rows (series replicates collapsed)')


def to_df(B, wb, f, cond, regime):
    B, wb, f = [np.atleast_1d(np.asarray(x, float)) for x in (B, wb, f)]
    df = pd.DataFrame({'w/c': wb / (1 - f), 'w/b': wb, 'FA': f * B, 'CO2': cond['CO2'], 'RH': cond['RH'], 'T': cond['T'], 'sqrt_t': cond['sqrt_t'], 't_cure': cond['t_cure']})
    df['regime'] = regime; return df


def support(df):
    z = (df[NONTIME].values - mu_) / sd_
    return np.maximum(zmin - z, 0).sum(1) + np.maximum(z - zmax, 0).sum(1), tree.query(z, k=5)[0][:, 4]


def predict(df, regime=None):
    raw = np.mean([m.raw_k(df) for m in models], 0); E = KM0.expo(df); reg = regime or df['regime'].iloc[0]
    tk = (lambda r: np.exp(r)) if LOGK else (lambda r: r)
    return dict(k=tk(raw), k_up=tk(raw + Q), k_up_mond=tk(raw + QM[reg]), E=E)


def co2_of(B, wb, f, ef):
    B, wb, f = [np.asarray(x, float) for x in (B, wb, f)]
    return (1 - f) * B * ef['EF_c'] + f * B * ef['EF_fa'] + wb * B * ef['EF_w']


def X_up_A(B, wb, f):
    df = to_df(B, wb, f, COND, 'accelerated'); p = predict(df); return p['k_up'] * p['E'], p, df


# ---------------- B sensitivity of k-hat ----------------
rows = []
for f_, wb_ in itertools.product([0, .1, .2, .3, .4], [.35, .45, .55, .65]):
    ks = [predict(to_df(B_, wb_, f_, COND, 'accelerated'))['k'][0] for B_ in (300, 350, 400, 450)]
    rows.append(dict(f=f_, w_b=wb_, k_hat_B300_450=[float(x) for x in ks], rel_range=float((max(ks) - min(ks)) / np.mean(ks))))
rr = np.array([r['rel_range'] for r in rows]); rr_pos = np.array([r['rel_range'] for r in rows if r['f'] > 0])
Bsens = dict(note='B is NOT a feature of the dataset; k-hat depends on B only through FA = f*B (kg/m3) (w/c = (w/b)/(1-f) does not involve B). At f=0 B has no effect. Table: k-hat at B = 300/350/400/450 at fixed w/b and f, condition A.',
             rows=rows, median_rel_range_all=float(np.median(rr)), max_rel_range=float(rr.max()), median_rel_range_f_gt0=float(np.median(rr_pos)), max_rel_range_f_gt0=float(rr_pos.max()),
             decision='B fixed at 350 kg/m3 in the main result: dataset has no binder content, so any B-effect is an uncontrolled FA(kg/m3) proxy rather than physics. B-free variant (Bfree) reported as appendix.')
savej(Bsens, 'p2_B_sensitivity.json'); print('B sens', Bsens['median_rel_range_f_gt0'], Bsens['max_rel_range_f_gt0'], flush=True)

VARIANTS = {'main': dict(fixB=B_FIX, wb_hi=.55), 'wb050': dict(fixB=B_FIX, wb_hi=.50), 'Bfree': dict(fixB=None, wb_hi=.55)}


def decode(X, var):
    if var['fixB']: return np.full(len(X), var['fixB']), X[:, 0], X[:, 1]
    return X[:, 0], X[:, 1], X[:, 2]


class MixProblem(Problem):
    def __init__(self, ef, var):
        self.ef, self.var = ef, var
        xl, xu = ([.35, 0.], [var['wb_hi'], .40]) if var['fixB'] else ([300., .35, 0.], [450., var['wb_hi'], .40])
        super().__init__(n_var=len(xl), n_obj=2, n_ieq_constr=1, xl=np.array(xl), xu=np.array(xu))

    def _evaluate(self, X, out, *a, **k):
        B, wb, f = decode(X, self.var); f1, p, df = X_up_A(B, wb, f); ex, kd = support(df)
        out['F'] = np.column_stack([f1, co2_of(B, wb, f, self.ef)]); out['G'] = (ex + np.maximum(kd - thr, 0))[:, None]


def run(ef, var, seed, hist=False):
    return minimize(MixProblem(ef, var), NSGA2(pop_size=100), ('n_gen', 200), seed=seed, verbose=False, save_history=hist)


def merged(results, var):
    X = np.vstack([r.X for r in results]); F = np.vstack([r.F for r in results])
    X, idx = np.unique(X, axis=0, return_index=True); F = F[idx]
    nd = NonDominatedSorting().do(F, only_non_dominated_front=True); X, F = X[nd], F[nd]; o = np.argsort(F[:, 1])
    X, F = X[o], F[o]; B, wb, f = decode(X, var); return np.column_stack([B, wb, f]), F


def knee_idx(F):
    Fn = (F - F.min(0)) / (F.max(0) - F.min(0) + 1e-12); a_, b_ = Fn[np.argmin(Fn[:, 0])], Fn[np.argmin(Fn[:, 1])]
    v = (b_ - a_) / (np.linalg.norm(b_ - a_) + 1e-12); return int(np.argmax(np.abs((Fn - a_)[:, 0] * v[1] - (Fn - a_)[:, 1] * v[0])))


def describe(B, wb, f, ef, label=''):
    f1, p, df = X_up_A(B, wb, f); ex, kd = support(df)
    return dict(label=label, B=float(B), w_b=float(wb), f=float(f), FA=float(f * B), cement=float((1 - f) * B), water=float(wb * B), w_c=float(wb / (1 - f)),
                X_pred_28d_mm=float(p['k'][0] * p['E'][0]), X_up_28d_mm=float(f1[0]), X_up_mond_28d_mm=float(p['k_up_mond'][0] * p['E'][0]), k_hat=float(p['k'][0]), k_up=float(p['k_up'][0]),
                embodied_CO2_kg_m3=float(co2_of(B, wb, f, ef)), range_exceed=float(ex[0]), knn_dist=float(kd[0]), supported=bool(ex[0] <= 0 and kd[0] <= thr))


# ---------------- Condition B machinery ----------------
CB = dict(CO2=0.04, T=20.0, RH=65.0, t_cure=28.0); T50 = 50 * 365.25; condB = dict(CB, sqrt_t=float(np.sqrt(T50)))
PS = [0.4, 0.5, 0.6]
mkc = art['kind'].startswith('Mkc')


def cond_B(B, wb, f):
    dfN = to_df(B, wb, f, condB, 'natural'); pN = predict(dfN); exN, kdN = support(dfN)
    raw = np.mean([m.raw_k(dfN) for m in models], 0); kupM = np.exp(raw + QM['natural']) if LOGK else raw + QM['natural']
    _, pA, _ = X_up_A(B, wb, f)
    o = dict(X_pred_50y_direct=pN['k'] * pN['E'], X_up_50y_direct=pN['k_up'] * pN['E'], X_up_50y_direct_mondnat=kupM * pN['E'], supported_B=(exN <= 0) & (kdN <= thr), knn_B=kdN)
    for p_ in PS:
        fac = 1.0 if mkc else (0.04 / 20.0) ** p_
        o[f'X_pred_50y_sqrtCO2_p{p_}'] = pA['k'] * fac * pN['E']; o[f'X_up_50y_sqrtCO2_p{p_}'] = pA['k_up'] * fac * pN['E']
    return o


# ---------------- main loop over variants ----------------
out_var, figdata = {}, {}
for vname, var in VARIANTS.items():
    rng = np.random.default_rng(0)
    S = rng.uniform([.35, 0], [var['wb_hi'], .40], size=(50000, 2)); BS = np.full(50000, var['fixB'] or 375.0)
    if not var['fixB']: BS = rng.uniform(300, 450, 50000)
    dfS = to_df(BS, S[:, 0], S[:, 1], COND, 'accelerated'); exS, kdS = support(dfS)
    rej = dict(fraction_rejected=float(((exS > 0) | (kdS > thr)).mean()), fraction_rejected_range=float((exS > 0).mean()), fraction_rejected_knn=float((kdS > thr).mean(), ), n_samples=50000)
    runs = [run(base_ef, var, s, hist=True) for s in range(3)]
    Pd, Ff = merged(runs, var)
    ref = np.array([Ff[:, 0].max() * 1.1, Ff[:, 1].max() * 1.1])
    hvf = lambda F: float(HV(ref_point=ref)(F)) if len(F) else 0.0
    hv = {f'seed{s}': dict(final=hvf(r.F), by_gen={str(g): hvf(r.history[g - 1].pop.get('F')[NonDominatedSorting().do(r.history[g - 1].pop.get('F'), only_non_dominated_front=True)]) for g in (25, 50, 100, 200)}) for s, r in enumerate(runs)}
    ik = knee_idx(Ff); qi = sorted(set([0, int(.25 * (len(Ff) - 1)), ik, int(.75 * (len(Ff) - 1)), len(Ff) - 1])); nm_ = {0: 'min-CO2 end', ik: 'KNEE', len(Ff) - 1: 'min-depth end'}
    reps = [describe(*Pd[i], base_ef, nm_.get(i, 'Q25' if i == int(.25 * (len(Ff) - 1)) else 'Q75')) for i in qi]
    front_desc = [describe(*p, base_ef) for p in Pd]
    conv = [describe(350, .50, f, base_ef, f'conventional B=350, w/b=0.50, f={f:.0%}') for f in (0, .2, .3)]
    for c in conv:
        ok = Ff[:, 0] <= c['X_up_28d_mm']; c['CO2_saving_pct_at_same_or_lower_depth'] = float(100 * (1 - Ff[ok, 1].min() / c['embodied_CO2_kg_m3'])) if ok.any() else None
        ok2 = Ff[:, 1] <= c['embodied_CO2_kg_m3']; c['depth_saving_pct_at_same_or_lower_CO2'] = float(100 * (1 - Ff[ok2, 0].min() / c['X_up_28d_mm'])) if ok2.any() else None
    ranges = dict(f_range=[float(Pd[:, 2].min()), float(Pd[:, 2].max())], wb_range=[float(Pd[:, 1].min()), float(Pd[:, 1].max())], B_range=[float(Pd[:, 0].min()), float(Pd[:, 0].max())],
                  frac_front_at_wb_upper_bound=float((Pd[:, 1] > var['wb_hi'] - .005).mean()), frac_front_at_f_upper_bound=float((Pd[:, 2] > .395).mean()), n_front_supported=int(sum(x['supported'] for x in front_desc)))
    out_var[vname] = dict(var=var, rejected=rej, hypervolume=hv, ref_point=ref.tolist(), front_size=len(Ff), knee=front_desc[ik], representatives=reps, conventional=conv, front_ranges=ranges, front=front_desc)
    figdata[vname] = (Pd, Ff, ik, conv)
    print(vname, 'front', len(Ff), rej, 'knee', {k: round(v, 3) if isinstance(v, float) else v for k, v in front_desc[ik].items()}, flush=True)

# ---------------- functional-unit analysis (main variant domain, B=350) ----------------
gw, gf = np.meshgrid(np.linspace(.35, .55, 41), np.linspace(0, .40, 41)); gw, gf = gw.ravel(), gf.ravel(); gB = np.full(len(gw), B_FIX)
fA, _, dfg = X_up_A(gB, gw, gf); exA, kdA = support(dfg); supA = (exA <= 0) & (kdA <= thr); cB = cond_B(gB, gw, gf); co2g = co2_of(gB, gw, gf, base_ef)
methods = ['X_up_50y_direct', 'X_up_50y_direct_mondnat'] + [f'X_up_50y_sqrtCO2_p{p}' for p in PS]
fu = {'note': 'Functional unit: cover target c (mm); mix must satisfy X_up(50 y, natural cond. B) <= c AND be inside condition-A data support (w/b<=0.55, B=350). CO2 = min embodied CO2 among compliant mixes (grid 41x41 over w/b,f). EXTRAPOLATION.',
      'n_grid': len(gw), 'frac_supported_A': float(supA.mean()), 'frac_supported_B_natural_cond': float(cB['supported_B'].mean()), 'by_method': {}}
conv_all = [dict(label=f'conventional f={f:.0%}', **{m: float(cond_B(350, .5, f)[m][0]) for m in methods}, CO2=float(co2_of(350, .5, f, base_ef))) for f in (0, .2, .3)]
fu['conventional_B'] = conv_all
for m in methods:
    fu['by_method'][m] = {}
    for cov in (30, 40, 50):
        ok = supA & (cB[m] <= cov)
        if ok.any():
            i = np.where(ok)[0][np.argmin(co2g[ok])]
            fu['by_method'][m][str(cov)] = dict(n_compliant=int(ok.sum()), min_CO2=float(co2g[i]), at=dict(w_b=float(gw[i]), f=float(gf[i]), X_up_50y=float(cB[m][i])),
                                               conventional_compliant=[dict(label=c['label'], CO2=c['CO2']) for c in conv_all if c[m] <= cov])
        else:
            fu['by_method'][m][str(cov)] = dict(n_compliant=0, min_X_up_50y_in_domain=float(cB[m][supA].min()), note='no supported mix meets this cover')
    fu['by_method'][m]['min_cover_needed_any_supported_mix'] = float(cB[m][supA].min())
    fu['by_method'][m]['max_cover_needed'] = float(cB[m][supA].max())
savej(fu, 'p2_functional_unit.json')
for m in methods: print('FU', m, {k: (v.get('min_CO2'), v.get('n_compliant')) for k, v in fu['by_method'][m].items() if isinstance(v, dict) and 'n_compliant' in v}, 'min cover', round(fu['by_method'][m]['min_cover_needed_any_supported_mix'], 1), flush=True)

# ---------------- condition B for main-variant front & reps ----------------
Pd, Ff, ik, conv = figdata['main']
cBf = cond_B(Pd[:, 0], Pd[:, 1], Pd[:, 2]); PB = pd.DataFrame({'B': Pd[:, 0], 'w_b': Pd[:, 1], 'f': Pd[:, 2], 'X_up_28d_acc': Ff[:, 0], 'CO2': Ff[:, 1], **{k: v for k, v in cBf.items()}})
PB.to_csv(R + 'p2_condition_B_front.csv', index=False)
qi_main = [r['label'] for r in out_var['main']['representatives']]
repB = []
for r in out_var['main']['representatives'] + out_var['main']['conventional']:
    c = cond_B(r['B'], r['w_b'], r['f']); repB.append(dict(label=r['label'], w_b=r['w_b'], f=r['f'], CO2=r['embodied_CO2_kg_m3'], X_up_28d=r['X_up_28d_mm'], **{k: float(v[0]) for k, v in c.items()},
                                                          meets_cover={str(cv): bool(c['X_up_50y_direct'][0] <= cv) for cv in (30, 40, 50)}))
condB_summary = dict(condition=CB, t_years=50, extrapolation_note='EXTRAPOLATION: natural data = 2 train studies (T 20-21.7 C, RH 50-75 %, CO2 0.03-0.05 %, <= ~5 y). 50 y is ~10x the longest natural exposure; k-model trained mostly on accelerated tests.',
                     front_summary={m: dict(min=float(PB[m].min()), median=float(PB[m].median()), max=float(PB[m].max())) for m in methods + ['X_pred_50y_direct']},
                     frac_front_supported_natural_cond=float(PB.supported_B.mean()), representatives_and_conventional=repB)
savej(condB_summary, 'p2_condition_B.json')
print(json.dumps(condB_summary['front_summary'], default=float)[:1500], flush=True)

# ---------------- EF sensitivity (main variant, seed 0) ----------------
sens = {}
S_ = EF['sensitivity']
for efc, efa, efw in itertools.product(S_['EF_c'], S_['EF_fa'], S_['EF_w']):
    ef = dict(EF_c=efc, EF_fa=efa, EF_w=efw); r = run(ef, VARIANTS['main'], 0); Pn, Fn_ = merged([r], VARIANTS['main']); kn = knee_idx(Fn_)
    sens[f'EFc={efc},EFfa={efa},EFw={efw}'] = dict(EF=ef, F=Fn_.tolist(), knee=dict(w_b=float(Pn[kn, 1]), f=float(Pn[kn, 2]), X_up=float(Fn_[kn, 0]), CO2=float(Fn_[kn, 1])), f_range=[float(Pn[:, 2].min()), float(Pn[:, 2].max())],
                                                  co2_min=float(Fn_[:, 1].min()), co2_max=float(Fn_[:, 1].max()), conv_f0_CO2=float(co2_of(350, .5, 0, ef)), conv_f30_CO2=float(co2_of(350, .5, .3, ef)),
                                                  saving_f0_vs_front_min_pct=float(100 * (1 - Fn_[:, 1].min() / co2_of(350, .5, 0, ef))))
savej(dict(note='EF sensitivity: NSGA-II seed 0, main variant (B=350, w/b<=0.55); factorial EF_c x EF_fa x EF_w', sens=sens), 'p2_ef_sensitivity.json')

savej(dict(note='Condition A accelerated: CO2 20 %, T 20 C, RH 70 %, 28 d (sqrt_t=sqrt(28)), t_cure 28 d. f1 = conformal UPPER bound of 28-d depth = k_up * t^n (split conformal in k-space, alpha=0.10, global q, calibration on held-out studies). f2 = embodied CO2 (kg/m3) = cement*EF_c + FA*EF_fa + water*EF_w (A1-A3, binder and water only; aggregates/admixtures excluded => PARTIAL totals). Durability-class assumption: w/b<=0.55 (main) or <=0.50; NO strength model, strength not constrained. NSGA-II pop 100, 200 gen, 3 seeds, merged non-dominated front.',
           k_model=dict(kind=art['kind'], cfg=art['cfg'], n=art['n'], q_global_a010=Q, q_mond_a010=QM, n_cal=art['n_cal'], n_cal_natural=art['n_cal_natural']),
           EF=EF, support_check_condition_A=sup, support_constraint=support_info, variants=out_var), 'p2_pareto.json')

# ---------------- figures ----------------
plt.rcParams.update({'font.size': 7})
Pd, Ff, ik, conv = figdata['main']
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
for ax, (col, lab, cm) in zip(axs, [(Pd[:, 2], 'FA share f', 'viridis'), (Pd[:, 1], 'w/b (range 0.35-0.55 allowed)', 'plasma')]):
    sc = ax.scatter(Ff[:, 1], Ff[:, 0], c=col, cmap=cm, s=10, **({'vmin': .35, 'vmax': .55} if 'w/b' in lab else {}))
    ax.scatter(Ff[ik, 1], Ff[ik, 0], marker='*', s=90, c='k', label='knee')
    ax.scatter([c['embodied_CO2_kg_m3'] for c in conv], [c['X_up_28d_mm'] for c in conv], marker='D', facecolors='none', edgecolors='r', s=30, label='conventional (w/b 0.5; f=0, .2, .3)')
    if 'wb050' in figdata: ax.plot(figdata['wb050'][1][:, 1], figdata['wb050'][1][:, 0], 'b--', lw=.7, label='w/b <= 0.50 front')
    ax.set_xlabel('embodied CO$_2$ (kg/m$^3$), partial (binder+water)'); ax.set_ylabel('upper-bound 28-d accel. depth (mm)'); plt.colorbar(sc, ax=ax, label=lab)
axs[0].legend(fontsize=5, frameon=False); fig.tight_layout(); fig.savefig(OUT + r'\plots_export\Fig_pareto.png', dpi=300); plt.close(fig)

fig, ax = plt.subplots(figsize=(3.4, 2.8))
ax.plot(Ff[:, 1], Ff[:, 0], 'k-', lw=1.8, label='default')
cols = plt.cm.tab10(np.arange(len(sens)))
for (kk, v_), c in zip(sens.items(), cols):
    F_ = np.array(v_['F']); ax.plot(F_[:, 1], F_[:, 0], color=c, lw=.8, label=kk.replace('EFc', 'c').replace('EFfa', 'fa').replace('EFw', 'w').replace('=', ''))
ax.set_xlabel('embodied CO$_2$ (kg/m$^3$)'); ax.set_ylabel('upper-bound 28-d depth (mm)'); ax.legend(fontsize=3.8, frameon=False); fig.tight_layout()
fig.savefig(OUT + r'\plots_export\Fig_pareto_EF_sensitivity.png', dpi=300); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
axs[0].scatter(PB.CO2, PB.X_up_50y_direct, s=8, c='tab:blue', label='k-model direct, X$_{up}$')
for p_, c, mk in zip(PS, ['tab:orange', 'tab:red', 'tab:green'], ['x', '+', '1']): axs[0].scatter(PB.CO2, PB[f'X_up_50y_sqrtCO2_p{p_}'], s=10, c=c, marker=mk, label=f'k$_{{acc}}$(CO$_2$ ratio)$^{{{p_}}}$, X$_{{up}}$')
axs[0].scatter(PB.CO2, PB.X_pred_50y_direct, s=6, c='tab:cyan', label='direct, point')
axs[1].scatter(PB.X_up_28d_acc, PB.X_up_50y_direct, s=8, c=PB.f, cmap='viridis')
for ax in axs:
    for cv in (30, 40, 50): ax.axhline(cv, color='gray', ls=':', lw=.7)
    ax.set_yscale('log')
axs[0].set_xlabel('embodied CO$_2$ (kg/m$^3$)'); axs[0].set_ylabel('cover needed: X$_{up}$(50 y) (mm)'); axs[0].legend(fontsize=4.5, frameon=False)
axs[1].set_xlabel('28-d accelerated X$_{up}$ (mm)'); axs[1].set_ylabel('X$_{up}$(50 y), direct (mm); colour = f')
fig.suptitle('Condition B (20 C, 65 % RH, 0.04 % CO$_2$): EXTRAPOLATION (50 y vs max ~5 y observed)', fontsize=6.5); fig.tight_layout()
fig.savefig(OUT + r'\plots_export\Fig_cover_requirement.png', dpi=300); plt.close(fig)

rng = np.random.default_rng(1); Sg = rng.uniform([.35, 0], [.55, .40], size=(6000, 2)); dS = to_df(np.full(6000, B_FIX), Sg[:, 0], Sg[:, 1], COND, 'accelerated'); eS, kS = support(dS); okm = (eS <= 0) & (kS <= thr)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
axs[0].scatter(Sg[:, 0], Sg[:, 1] * B_FIX, c=np.where(okm, 'tab:green', 'tab:red'), s=2, alpha=.5)
near = tr[tr.CO2.between(15, 25) & tr.RH.between(60, 75)]
axs[0].scatter(near['w/b'], near['FA'], c='k', s=4, label='train rows, CO$_2$ 15-25 %, RH 60-75 %')
axs[0].set_xlabel('w/b'); axs[0].set_ylabel('FA = f x 350 (kg/m$^3$)'); axs[0].legend(fontsize=5, frameon=False, loc='upper right')
axs[0].set_title('design samples (B=350): green supported, red rejected; rejected %.2f' % out_var['main']['rejected']['fraction_rejected'], fontsize=5.5)
axs[1].hist(dd[:, 5], bins=40, color='gray', alpha=.6, label='within-train 5-NN distance'); axs[1].hist(kS, bins=40, color='tab:blue', alpha=.5, label='design candidates (cond. A)')
axs[1].axvline(thr, color='r', ls='--', label='95th pct threshold'); axs[1].set_xlabel('5-NN distance (standardized 7-D)'); axs[1].set_ylabel('count'); axs[1].legend(fontsize=5, frameon=False)
fig.tight_layout(); fig.savefig(OUT + r'\plots_export\Fig_support_map.png', dpi=300); plt.close(fig)
print('done')
