"""Part 2 v2: fixed water W=175, B=W/(w/b), EN206-type k-value w_eq constraint, CO2 depends on w/b. Condition B with calibrated p-hat (p1e_calibrated_p.json). Writes *_v2 files only."""
import pickle, itertools
from models import *
from scipy.spatial import cKDTree
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = OUT + '\\results\\'
EF = json.load(open(OUT + r'\tools\config_ef.json')); base_ef = {k: EF[k] for k in ('EF_c', 'EF_fa', 'EF_w')}
art = pickle.load(open(R + 'kmodel_artifact.pkl', 'rb')); models, KM0 = art['models'], art['models'][0]
cal = json.load(open(R + 'p1e_calibrated_p.json'))['results']['Mkn_g']
P_HAT, P_LO, P_HI = cal['p_hat'], cal['ci95_cluster_bootstrap'][0], cal['ci95_cluster_bootstrap'][1]
d, info = load(); tr = d[~d.is_test].reset_index(drop=True)
Q = art['q']['0.1']; LOGK = art['cfg']['log']
W = 175.0; B_LO, B_HI, C_MIN = 280.0, 500.0, 260.0; WB_LO, WB_HI, F_HI = .35, .65, .40
LIMITS = {'weq055': .55, 'weq050': .50, 'weq060': .60}
COND = dict(CO2=20.0, T=20.0, RH=70.0, sqrt_t=float(np.sqrt(28.0)), t_cure=28.0)
CB = dict(CO2=0.04, T=20.0, RH=65.0, t_cure=28.0); condB = dict(CB, sqrt_t=float(np.sqrt(50 * 365.25)))
U = tr.drop_duplicates(NONTIME)[NONTIME].reset_index(drop=True); mu_, sd_ = U.mean().values, U.std().values
Z = (U.values - mu_) / sd_; tree = cKDTree(Z); thr = float(np.quantile(tree.query(Z, k=6)[0][:, 5], .95)); zmin, zmax = Z.min(0), Z.max(0)


def mix(wb, f):
    wb, f = np.atleast_1d(np.asarray(wb, float)), np.atleast_1d(np.asarray(f, float)); B = W / wb; C = (1 - f) * B; FA = f * B
    weq = W / (C + 0.4 * np.minimum(FA, 0.33 * C)); return dict(B=B, C=C, FA=FA, wc=W / C, weq=weq, wb=wb, f=f)


def to_df(m, cond, regime):
    df = pd.DataFrame({'w/c': m['wc'], 'w/b': m['wb'], 'FA': m['FA'], 'CO2': cond['CO2'], 'RH': cond['RH'], 'T': cond['T'], 'sqrt_t': cond['sqrt_t'], 't_cure': cond['t_cure']}); df['regime'] = regime; return df


def support(df):
    z = (df[NONTIME].values - mu_) / sd_; return np.maximum(zmin - z, 0).sum(1) + np.maximum(z - zmax, 0).sum(1), tree.query(z, k=5)[0][:, 4]


def kpred(df):
    raw = np.mean([m.raw_k(df) for m in models], 0); tk = np.exp if LOGK else (lambda x: x); return tk(raw), tk(raw + Q), KM0.expo(df)


def co2(m, ef): return m['C'] * ef['EF_c'] + m['FA'] * ef['EF_fa'] + W * ef['EF_w']


def xup28(m): k, ku, E = kpred(to_df(m, COND, 'accelerated')); return ku * E, k * E


def constraints(m, lim, df):  # <=0 feasible
    ex, kd = support(df)
    return np.column_stack([m['weq'] - lim, C_MIN - m['C'], B_LO - m['B'], m['B'] - B_HI, ex + np.maximum(kd - thr, 0)])


CNAMES = ['w_eq', 'C_min', 'B_min', 'B_max', 'data_support']


class Prob(Problem):
    def __init__(self, ef, lim):
        self.ef, self.lim = ef, lim; super().__init__(n_var=2, n_obj=2, n_ieq_constr=5, xl=np.array([WB_LO, 0.]), xu=np.array([WB_HI, F_HI]))

    def _evaluate(self, X, out, *a, **k):
        m = mix(X[:, 0], X[:, 1]); out['F'] = np.column_stack([xup28(m)[0], co2(m, self.ef)]); out['G'] = constraints(m, self.lim, to_df(m, COND, 'accelerated'))


def run(ef, lim, seed): return minimize(Prob(ef, lim), NSGA2(pop_size=100), ('n_gen', 200), seed=seed, verbose=False)


def merged(rs):
    X = np.vstack([r.X for r in rs]); F = np.vstack([r.F for r in rs]); X, i = np.unique(X, axis=0, return_index=True); F = F[i]
    nd = NonDominatedSorting().do(F, only_non_dominated_front=True); X, F = X[nd], F[nd]; o = np.argsort(F[:, 1]); return X[o], F[o]


def knee(F):
    Fn = (F - F.min(0)) / (F.max(0) - F.min(0) + 1e-12); a, b = Fn[np.argmin(Fn[:, 0])], Fn[np.argmin(Fn[:, 1])]; v = (b - a) / (np.linalg.norm(b - a) + 1e-12)
    return int(np.argmax(np.abs((Fn - a)[:, 0] * v[1] - (Fn - a)[:, 1] * v[0])))


def condB_X(m, p):  # upper bound at 50 y natural, calibrated conversion
    k, ku, E = kpred(to_df(m, COND, 'accelerated')); En = KM0.expo(to_df(m, condB, 'natural')); fac = (0.04 / 20.0) ** p; return ku * fac * En, k * fac * En


def describe(wb, f, lim, ef, label=''):
    m = mix(wb, f); xu, xp = xup28(m); G = constraints(m, lim, to_df(m, COND, 'accelerated'))[0]
    tol = np.array([.002, 1., 1., 1., 1e-6]); act = [n for n, g, t in zip(CNAMES[:4], G[:4], tol) if abs(g) <= t]  # data_support: G=0 means satisfied, not active
    o = dict(label=label, w_b=float(wb), f=float(f), B=float(m['B'][0]), C=float(m['C'][0]), FA=float(m['FA'][0]), w_c=float(m['wc'][0]), w_eq=float(m['weq'][0]), X_up_28d_mm=float(xu[0]), X_pred_28d_mm=float(xp[0]),
             CO2=float(co2(m, ef)[0]), feasible=bool((G <= 1e-9).all()), active_constraints=act, violated=[n for n, g in zip(CNAMES, G) if g > 1e-9])
    for tag, p in (('p_hat', P_HAT), ('p_lo', P_LO), ('p_hi', P_HI)): o['X_up_50y_' + tag] = float(condB_X(m, p)[0][0])
    return o


if __name__ == '__main__':
    out, fig_ = {}, {}
    for name, lim in LIMITS.items():
        runs = [run(base_ef, lim, s) for s in range(3)]; X, F = merged(runs); ik = knee(F); n = len(F)
        qi = sorted(set([0, int(.25 * (n - 1)), ik, int(.75 * (n - 1)), n - 1])); lab = {0: 'min-CO2 end', ik: 'KNEE', n - 1: 'min-depth end', int(.25 * (n - 1)): 'Q25', int(.75 * (n - 1)): 'Q75'}
        front = [describe(x[0], x[1], lim, base_ef) for x in X]
        reps = [dict(front[i], label=lab.get(i, 'Q')) for i in qi]
        conv = [describe(.50, f, lim, base_ef, f'conventional W=175, w/b=0.50, f={f:.0%}') for f in (0, .2, .3)]
        for c in conv:
            ok = F[:, 0] <= c['X_up_28d_mm']; c['CO2_saving_pct_at_equal_X_up28'] = float(100 * (1 - F[ok, 1].min() / c['CO2'])) if ok.any() else None
            ok2 = F[:, 1] <= c['CO2']; c['X_up28_reduction_pct_at_equal_CO2'] = float(100 * (1 - F[ok2, 0].min() / c['X_up_28d_mm'])) if ok2.any() else None
        cnt = {nme: float(np.mean([nme in p_['active_constraints'] for p_ in front])) for nme in CNAMES[:4]}
        out[name] = dict(w_eq_limit=lim, front_size=n, knee=front[ik], representatives=reps, conventional=conv, frac_front_with_constraint_active=cnt,
                         front_ranges=dict(w_b=[float(X[:, 0].min()), float(X[:, 0].max())], f=[float(X[:, 1].min()), float(X[:, 1].max())], B=[min(p_['B'] for p_ in front), max(p_['B'] for p_ in front)], CO2=[float(F[:, 1].min()), float(F[:, 1].max())], X_up28=[float(F[:, 0].min()), float(F[:, 0].max())]), front=front)
        fig_[name] = (X, F, ik, conv)
        print(name, 'n', n, 'knee', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in front[ik].items()}, 'active', cnt, flush=True)
        for c in conv: print('  conv', c['label'], round(c['X_up_28d_mm'], 1), round(c['CO2'], 1), 'weq', round(c['w_eq'], 3), 'C', round(c['C'], 0), 'viol', c['violated'], 'sav', c['CO2_saving_pct_at_equal_X_up28'], c['X_up28_reduction_pct_at_equal_CO2'], flush=True)
    # EF sensitivity (main case, seed 0)
    sens = {}; S_ = EF['sensitivity']
    for c_, a_, w_ in itertools.product(S_['EF_c'], S_['EF_fa'], S_['EF_w']):
        ef = dict(EF_c=c_, EF_fa=a_, EF_w=w_); X, F = merged([run(ef, .55, 0)]); k = knee(F); cv = [describe(.5, f, .55, ef) for f in (0, .3)]
        okc = F[:, 0] <= cv[0]['X_up_28d_mm']
        sens[f'c{c_},fa{a_},w{w_}'] = dict(EF=ef, F=F.tolist(), knee=dict(w_b=float(X[k, 0]), f=float(X[k, 1]), X_up=float(F[k, 0]), CO2=float(F[k, 1])), CO2_range=[float(F[:, 1].min()), float(F[:, 1].max())], conv_f0_CO2=cv[0]['CO2'], conv_f30_CO2=cv[1]['CO2'],
                                           saving_vs_conv_f0_at_equal_X_up28_pct=float(100 * (1 - F[okc, 1].min() / cv[0]['CO2'])) if okc.any() else None)
    savej(dict(note='EF factorial sensitivity, case w_eq<=0.55 (assumed), NSGA-II seed 0, W=175.', sens=sens), 'p2_ef_sensitivity_v2.json')
    print('EF knees', {k: (round(v['knee']['w_b'], 3), round(v['knee']['f'], 3)) for k, v in sens.items()}, flush=True)
    # functional unit (grid, each case) with p_hat band
    gw, gf = np.meshgrid(np.linspace(WB_LO, WB_HI, 61), np.linspace(0, F_HI, 41)); gw, gf = gw.ravel(), gf.ravel(); m = mix(gw, gf); dfA = to_df(m, COND, 'accelerated'); fu = {}
    co = co2(m, base_ef)
    for name, lim in LIMITS.items():
        feas = (constraints(m, lim, dfA) <= 1e-9).all(1); fu[name] = {'n_feasible': int(feas.sum())}
        for tag, p in (('p_hat', P_HAT), ('p_lo(CI low -> larger depth)', P_LO), ('p_hi(CI high -> smaller depth)', P_HI)):
            x50 = condB_X(m, p)[0]; fu[name][tag] = dict(p=p, min_cover_needed_feasible=float(x50[feas].min()), max_cover_needed_feasible=float(x50[feas].max()))
            for cv in (30, 40, 50):
                ok = feas & (x50 <= cv); i = np.where(ok)[0][np.argmin(co[ok])] if ok.any() else None
                fu[name][tag][str(cv)] = dict(n=int(ok.sum()), **({} if i is None else dict(min_CO2=float(co[i]), w_b=float(gw[i]), f=float(gf[i]))))
        fu[name]['conventional'] = [dict(label=c['label'], X_up_50y_p_hat=c['X_up_50y_p_hat'], X_up_50y_band=[c['X_up_50y_p_hi'], c['X_up_50y_p_lo']], CO2=c['CO2'], violated=c['violated']) for c in out[name]['conventional']]
        print('FU', name, {t: {k: (v.get('min_CO2') if isinstance(v, dict) else v) for k, v in fu[name][t].items() if k in ('30', '40', '50', 'min_cover_needed_feasible')} for t in ('p_hat',)}, flush=True)
    savej(dict(note='Condition B (20 C, 65 %% RH, 0.04 %% CO2, 50 y) EXTRAPOLATION. X_up(50y) = k_up(cond A, conformal alpha=0.10) * (0.04/20)^p * t^n, p = p-hat %.3f (cluster-bootstrap 95%% CI %.3f-%.3f, 8 natural studies; p1e_calibrated_p.json). Band reflects uncertainty in p ONLY (not conformal/model/time-extrapolation uncertainty; 50 y is ~10x the longest natural exposure). Functional unit: min CO2 over a 61x41 (w/b,f) grid satisfying all constraints and X_up(50y)<=cover.' % (P_HAT, P_LO, P_HI), by_case=fu), 'p2_functional_unit_v2.json')
    savej(dict(note='Mix design v2. W=175 kg/m3 (assumed workability). B=W/(w/b), C=(1-f)B, FA=fB, w/c=W/C. Decision vars w/b in [0.35,0.65], f in [0,0.40]. B in [280,500]. w_eq=W/(C+0.4*min(FA,0.33C)) (EN 206 k-value concept). ASSUMED limits, to be verified against EN 206: w_eq<=0.55 (main, "XC3/XC4-type limit, assumed"), 0.50, 0.60; C>=260 kg/m3 (assumed). Strength not modelled/constrained. f1 = conformal UPPER bound (alpha 0.10) of 28-d depth, condition A (CO2 20 %%, 20 C, 70 %% RH). f2 = embodied CO2 kg/m3 = C*EF_c+FA*EF_fa+W*EF_w (A1-A3, binder+water; aggregates/admixtures excluded => PARTIAL). NSGA-II pop 100, 200 gen, 3 seeds, merged non-dominated front. Data-support constraint (range + 5-NN <= p95 = %.3f) also applied.' % thr,
               EF=EF, p_hat=dict(p_hat=P_HAT, ci95=[P_LO, P_HI]), cases=out), 'p2_pareto_v2.json')
    # ---------------- figures ----------------
    plt.rcParams.update({'font.size': 7}); cols = {'weq050': 'tab:green', 'weq055': 'k', 'weq060': 'tab:orange'}
    X, F, ik, conv = fig_['weq055']
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for nme, (Xn, Fn, ikn, cn) in fig_.items(): axs[0].plot(Fn[:, 1], Fn[:, 0], '-o', ms=2, lw=.8, color=cols[nme], label=f'w$_{{eq}}\\leq${LIMITS[nme]:.2f}')
    for ci, c in enumerate(conv): axs[0].scatter(c['CO2'], c['X_up_28d_mm'], marker='D', facecolors='none', edgecolors='r', s=25, label='conventional, w/b 0.50, $f$ = 0, 0.2, 0.3' if ci == 0 else None)
    axs[0].scatter(F[ik, 1], F[ik, 0], marker='*', s=90, c='k', zorder=5, label='knee, w$_{eq}\\leq$0.55'); axs[0].legend(fontsize=5, frameon=False)
    sc = axs[1].scatter(F[:, 1], F[:, 0], c=X[:, 0], cmap='plasma', s=12); axs[1].scatter(F[ik, 1], F[ik, 0], marker='*', s=90, c='k'); plt.colorbar(sc, ax=axs[1], label='w/b')
    for ax, tag in zip(axs, ['(a)', '(b) w$_{eq}\\leq$0.55']): ax.set_xlabel('Partial embodied CO$_2$ (kg/m$^3$)'); ax.set_ylabel('$X_{\\mathrm{up},28}$ (mm)'); ax.set_title(tag, fontsize=7, loc='left')
    fig.tight_layout(); fig.savefig(OUT + r'\plots_export\Fig_pareto_v2.png', dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.4, 2.8)); ax.plot(F[:, 1], F[:, 0], 'k-', lw=1.8, label='default')
    for (kk, v_), c in zip(sens.items(), plt.cm.tab10(np.arange(len(sens)))): F_ = np.array(v_['F']); ax.plot(F_[:, 1], F_[:, 0], color=c, lw=.8, label=kk)
    ax.set_xlabel('embodied CO$_2$ (kg/m$^3$)'); ax.set_ylabel('upper-bound 28-d depth (mm)'); ax.legend(fontsize=3.8, frameon=False); fig.tight_layout(); fig.savefig(OUT + r'\plots_export\Fig_pareto_EF_sensitivity_v2.png', dpi=300); plt.close(fig)
    fr = out['weq055']['front']; fig, ax = plt.subplots(figsize=(3.4, 2.9)); cx = np.array([p_['CO2'] for p_ in fr])
    ax.fill_between(cx, [p_['X_up_50y_p_hi'] for p_ in fr], [p_['X_up_50y_p_lo'] for p_ in fr], color='tab:blue', alpha=.3, label='band: $\\hat p$ 95% CI'); ax.plot(cx, [p_['X_up_50y_p_hat'] for p_ in fr], 'b-', lw=1, label='$\\hat p$ = %.2f' % P_HAT)
    for cv in (30, 40, 50): ax.axhline(cv, color='gray', ls=':', lw=.7)
    ax.set_yscale('log'); ax.set_xlabel('embodied CO$_2$ (kg/m$^3$)'); ax.set_ylabel('cover needed X$_{up}$(50 y) (mm)'); ax.set_title('Condition B: EXTRAPOLATION, w$_{eq}\\leq$0.55 front', fontsize=6); ax.legend(fontsize=5, frameon=False); fig.tight_layout()
    fig.savefig(OUT + r'\plots_export\Fig_cover_requirement_v2.png', dpi=300); plt.close(fig); print('done')
