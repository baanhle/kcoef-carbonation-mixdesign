"""RQ4: carbon intensity per year of carbonation service life, CI = embodied CO2 (kg/m3) / t_SL (yr).
t_SL = (c / k_nat)^(1/n) / 365.25, k_nat = k_hat(condition A, accelerated) * (0.04/20)^p  (only CO2 and time are converted; the k-model stays at condition A).
Natural: 20 C, 65 % RH, 0.04 % CO2, curing 28 d. EXTRAPOLATION: t_SL beyond ~5 y natural exposure is outside the data (longest natural series ~ 5 y)."""
import sys; sys.path.insert(0, r'd:\Research\TienThe\04_kcoef_mixdesign\tools')
from p2_v2 import *
from rq_common import KMX
from scipy.optimize import minimize_scalar
NG = float(art['n']['global']); cfgA = art['cfg']; YR = 365.25
dfN = lambda m: to_df(m, dict(CB, sqrt_t=1.0), 'natural')


# ---------- Mkn_x ensemble (n-hat(x) model); conformal q from the same calibration studies as the artifact ----------
pt, cal, cs = calib_split(tr); pt, cal = pt.reset_index(drop=True), cal.reset_index(drop=True)
KX = [KMX('XGBoost', s, cfgA).fit(pt) for s in range(10)]
rawt = lambda km, df: np.log(df.Target.values / km.expo(df)) if km.cfg['log'] else df.Target.values / km.expo(df)
QX = float(qhat(np.abs(rawt(KX[0], cal) - np.mean([m.raw_k(cal) for m in KX], 0)), 0.10))
# p-hat for Mkn_x: accelerated-only Mkn_x (n-model sees accelerated series only), natural rows, k at CO2=20, n-hat at the row's own condition
acc, nat = tr[tr.regime == 'accelerated'].reset_index(drop=True), tr[tr.regime == 'natural'].reset_index(drop=True)
ref = nat.copy(); ref['CO2'] = 20.0; bb = []
for s in range(10):
    km = KMX('XGBoost', s, cfgA).fit(acc); bb.append(km.predict_k(ref) * km.expo(nat))
bb = np.mean(bb, 0); yv = nat.Target.values; rr = nat.CO2.values / 20.0
P_X = float(minimize_scalar(lambda p: np.sum((bb * rr ** p - yv) ** 2), bounds=(-.5, 1.5), method='bounded', options=dict(xatol=1e-5)).x)
print('QX', QX, 'q global', Q, 'P_X', P_X, 'P_HAT', P_HAT, flush=True)

# ---------- grid ----------
NW, NF = 61, 41
gw, gf = np.meshgrid(np.linspace(WB_LO, WB_HI, NW), np.linspace(0, F_HI, NF)); sh = gw.shape; gw, gf = gw.ravel(), gf.ravel()
m = mix(gw, gf); dfA = to_df(m, COND, 'accelerated'); feas = (constraints(m, .55, dfA) <= 1e-9).all(1); CO = co2(m, base_ef)
kg, kug, _ = kpred(dfA); ng = np.full(len(gw), NG)
kx = np.mean([mm.raw_k(dfA) for mm in KX], 0); kx, kxu = np.exp(kx), np.exp(np.mean([mm.raw_k(dfA) for mm in KX], 0) + QX)
nx = np.mean([mm.n_rows(dfN(m)) for mm in KX], 0)
print('global n', NG, 'n-hat natural grid min/mean/max', nx.min(), nx.mean(), nx.max(), 'feasible', int(feas.sum()), flush=True)


def tsl(k, n, p, c): return (c / (k * (0.04 / 20.0) ** p)) ** (1.0 / n) / YR


def variant(nv, bound, c, p, mm_=None, kk=None):
    k = {('global', 'point'): kg, ('global', 'upper'): kug, ('xhat', 'point'): kx, ('xhat', 'upper'): kxu}[(nv, bound)]; n = ng if nv == 'global' else nx
    return tsl(k, n, p, c)


def analyse(T):
    """T = t_SL on grid (yr). returns summary; CI map on grid"""
    CI = CO / T; C2 = np.where(feas, CI, np.nan).reshape(sh); ratio = C2 / C2[0:1, :]  # row 0 = f=0
    ok = np.isfinite(C2).any(0); fs, fstar_ci = np.full(NW, np.nan), np.full(NW, np.nan)
    for j in range(NW):
        if np.isfinite(C2[:, j]).any(): i = np.nanargmin(C2[:, j]); fs[j] = gf.reshape(sh)[i, j]; fstar_ci[j] = C2[i, j]
    i = np.nanargmin(np.where(feas, CI, np.nan)); Tf = T[feas]
    return dict(CI=CI, fs=fs, fstar_ci=fstar_ci, ratio=ratio, i=i, Tf=Tf)


def fixed_slice(nv, bound, c, p, we):
    F_ = analyse_fixed[we]; dfa, mm_ = F_['dfa'], F_['mm']
    if nv == 'global': k = kpred(dfa)[0 if bound == 'point' else 1]; n = np.full(len(dfa), NG)
    else:
        raw = np.mean([x.raw_k(dfa) for x in KX], 0); k = np.exp(raw + (QX if bound == 'upper' else 0)); n = np.mean([x.n_rows(dfN(mm_)) for x in KX], 0)
    T = tsl(k, n, p, c); CI = np.where(F_['ok'], F_['co2'] / T, np.nan); return T, CI


analyse_fixed = {}
for we in (0.45, 0.50, 0.55):
    f = np.linspace(0, F_HI, 81); wb = we * (1 - f + 0.4 * np.minimum(f, 0.33 * (1 - f))); mm_ = mix(wb, f); dfa = to_df(mm_, COND, 'accelerated')
    ok_ = (constraints(mm_, we + 1e-6, dfa) <= 1e-9).all(1) & (wb >= WB_LO - 1e-9) & (wb <= WB_HI + 1e-9)
    analyse_fixed[we] = dict(ok=ok_, f=f, wb=wb, co2=co2(mm_, base_ef), dfa=dfa, mm=mm_)

# ---------- all cases ----------
PS = {'global': dict(p_lo=P_LO, p_hat=P_HAT, p_hi=P_HI), 'xhat': dict(p_lo=P_LO, p_hat=P_HAT, p_hi=P_HI, p_x=P_X)}
cases, store = {}, {}
for nv in ('global', 'xhat'):
    for bound in ('point', 'upper'):
        for c in (30, 40, 50):
            for pn, p in PS[nv].items():
                key = f'n={nv}|{bound}|c={c}|{pn}'; T = variant(nv, bound, c, p); A = analyse(T); i = A['i']; C2 = np.where(feas, A['CI'], np.nan).reshape(sh)
                fx = {}
                for we in (0.45, 0.50, 0.55):
                    Tx, CIx = fixed_slice(nv, bound, c, p, we); F_ = analyse_fixed[we]; j = np.nanargmin(CIx) if np.isfinite(CIx).any() else None
                    last = np.where(F_['ok'])[0].max() if F_['ok'].any() else None
                    fx[str(we)] = dict(f_range_ok=[float(F_['f'][F_['ok']].min()), float(F_['f'][F_['ok']].max())] if F_['ok'].any() else None, f_star=None if j is None else float(F_['f'][j]), CI_f0=float(CIx[0]) if np.isfinite(CIx[0]) else None,
                                       CI_min=None if j is None else float(CIx[j]), CI_fmax_ok=None if last is None else float(CIx[last]), ratio_min_over_f0=None if j is None or not np.isfinite(CIx[0]) else float(CIx[j] / CIx[0]),
                                       ratio_fmax_over_f0=None if last is None or not np.isfinite(CIx[0]) else float(CIx[last] / CIx[0]), tSL_f0=float(Tx[0]), tSL_fmax_ok=None if last is None else float(Tx[last]), CO2_f0=float(F_['co2'][0]), CO2_fmax_ok=None if last is None else float(F_['co2'][last]))
                be = np.isfinite(A['ratio']) & (A['ratio'] > 1.0 + 1e-9); fin = np.isfinite(A['ratio']) & (gf.reshape(sh) > 0)
                cases[key] = dict(n_variant=nv, bound=bound, cover_mm=c, p_name=pn, p=p, tSL_feasible=dict(min=float(A['Tf'].min()), median=float(np.median(A['Tf'])), max=float(A['Tf'].max()), frac_gt_50y=float(np.mean(A['Tf'] > 50)), frac_gt_100y=float(np.mean(A['Tf'] > 100)), frac_gt_5y=float(np.mean(A['Tf'] > 5))),
                                  global_opt=dict(w_b=float(gw[i]), f=float(gf[i]), CI=float(A['CI'][i]), t_SL_yr=float(T[i]), CO2=float(CO[i])),
                                  f_star_by_wb=dict(wb=np.linspace(WB_LO, WB_HI, NW).tolist(), f_star=[None if not np.isfinite(v) else float(v) for v in A['fs']]),
                                  f_star_summary=dict(n_wb_with_feasible=int(np.isfinite(A['fs']).sum()), frac_wb_f_star_gt0=float(np.mean(A['fs'][np.isfinite(A['fs'])] > 1e-9)), mean_f_star=float(np.nanmean(A['fs'])), frac_wb_f_star_at_F_HI=float(np.mean(A['fs'][np.isfinite(A['fs'])] >= F_HI - 1e-9))),
                                  break_even=dict(frac_feasible_cells_with_f_gt0_where_CI_exceeds_f0=float(be[fin].mean()) if fin.any() else None, n_cells=int(fin.sum())), fixed_weq=fx)
                store[key] = dict(CI=A['CI'], ratio=A['ratio'], T=T)
        print(nv, bound, 'done', flush=True)
MAIN = 'n=global|point|c=40|p_hat'; print('MAIN', {k: cases[MAIN][k] for k in ('global_opt', 'f_star_summary', 'break_even', 'tSL_feasible')}, flush=True)

# ---------- Pareto (p2_v2) and conventional mixes ----------
P2 = json.load(open(R + 'p2_pareto_v2.json'))['cases']['weq055']; ev = {}


def eval_mixes(items):
    wb_, f_ = np.array([x['w_b'] for x in items]), np.array([x['f'] for x in items]); mm_ = mix(wb_, f_); dfa = to_df(mm_, COND, 'accelerated'); co = co2(mm_, base_ef); o = {}
    for nv in ('global', 'xhat'):
        for bound in ('point', 'upper'):
            for c in (30, 40, 50):
                for pn, p in PS[nv].items():
                    if nv == 'global': k = kpred(dfa)[0 if bound == 'point' else 1]; n = np.full(len(dfa), NG)
                    else: k = np.exp(np.mean([x.raw_k(dfa) for x in KX], 0) + (QX if bound == 'upper' else 0)); n = np.mean([x.n_rows(dfN(mm_)) for x in KX], 0)
                    T = tsl(k, n, p, c); o[f'n={nv}|{bound}|c={c}|{pn}'] = dict(t_SL=T.tolist(), CI=(co / T).tolist())
    return o, co


fr = P2['front']; ev_front, co_front = eval_mixes(fr)
conv_items = [dict(w_b=.5, f=f_) for f_ in (0., .2, .3)]; ev_conv, co_conv = eval_mixes(conv_items)
pareto_sum = {}
for key, v in ev_front.items():
    CIv, Tv = np.array(v['CI']), np.array(v['t_SL']); j = int(np.argmin(CIv))
    pareto_sum[key] = dict(CI_min=float(CIv[j]), w_b=fr[j]['w_b'], f=fr[j]['f'], CO2=fr[j]['CO2'], t_SL=float(Tv[j]), CI_median=float(np.median(CIv)), frac_gt_50y=float(np.mean(Tv > 50)), frac_gt_100y=float(np.mean(Tv > 100)),
                           CI_at_p2_knee=float(CIv[[i for i, x in enumerate(fr) if abs(x['w_b'] - P2['knee']['w_b']) < 1e-9 and abs(x['f'] - P2['knee']['f']) < 1e-9][0]]))
conv_sum = {key: [dict(w_b=.5, f=cf['f'], t_SL=t, CI=c_, CO2=float(co_conv[i]), w_eq=float(mix(.5, cf['f'])['weq'][0])) for i, (cf, t, c_) in enumerate(zip(conv_items, v['t_SL'], v['CI']))] for key, v in ev_conv.items() if key.startswith('n=global|point|c=40') or key.startswith('n=global|upper|c=40')}

# ---------- bi-objective NSGA-II: min CO2, max t_SL (global n, point, c=40, p_hat) ----------
class Prob2(Problem):
    def __init__(self): super().__init__(n_var=2, n_obj=2, n_ieq_constr=5, xl=np.array([WB_LO, 0.]), xu=np.array([WB_HI, F_HI]))

    def _evaluate(self, X, out, *a, **k):
        mm_ = mix(X[:, 0], X[:, 1]); dfa = to_df(mm_, COND, 'accelerated'); k_ = kpred(dfa)[0]; T = tsl(k_, np.full(len(dfa), NG), P_HAT, 40.0)
        out['F'] = np.column_stack([co2(mm_, base_ef), -np.log10(T)]); out['G'] = constraints(mm_, .55, dfa)


runs = [minimize(Prob2(), NSGA2(pop_size=100), ('n_gen', 200), seed=s, verbose=False) for s in range(3)]
Xp, Fp = merged(runs); o = np.argsort(Fp[:, 0]); Xp, Fp = Xp[o], Fp[o]; ik = knee(Fp)
mk_ = mix(Xp[:, 0], Xp[:, 1]); dfp = to_df(mk_, COND, 'accelerated'); kp = kpred(dfp)
front2 = []
for i in range(len(Xp)):
    r = dict(w_b=float(Xp[i, 0]), f=float(Xp[i, 1]), B=float(mk_['B'][i]), w_eq=float(mk_['weq'][i]), CO2=float(Fp[i, 0]), t_SL_yr=float(10 ** -Fp[i, 1]), CI=float(Fp[i, 0] / 10 ** -Fp[i, 1]))
    for pn, p in PS['global'].items(): r['t_SL_' + pn] = float(tsl(kp[0][i:i + 1], np.array([NG]), p, 40.0)[0])
    r['t_SL_upper_p_hat'] = float(tsl(kp[1][i:i + 1], np.array([NG]), P_HAT, 40.0)[0]); front2.append(r)
kn2 = front2[ik]; kn2['X_up28'] = float(xup28(mix(kn2['w_b'], kn2['f']))[0][0])
p2k = P2['knee']; p2k_ev = {n: ev_front[n]['t_SL'][[i for i, x in enumerate(fr) if abs(x['w_b'] - p2k['w_b']) < 1e-9 and abs(x['f'] - p2k['f']) < 1e-9][0]] for n in ('n=global|point|c=40|p_hat', 'n=global|upper|c=40|p_hat')}
print('knee2', kn2, 'front', len(front2), flush=True)
print('p2 knee', {k: p2k[k] for k in ('w_b', 'f', 'CO2', 'X_up_28d_mm')}, 'tSL', p2k_ev, flush=True)

savej(dict(note='CI = embodied CO2 (kg/m3, partial A1-A3 binder+water, p2_v2 EF) / t_SL (yr). t_SL = (c / (k_hat(cond A) (0.04/20)^p))^(1/n)/365.25, n = global %.3f or n-hat(x) from an XGB Mkn_x ensemble evaluated at the natural condition (20 C, 65 %% RH, 0.04 %% CO2) with the SAME mix; k stays at condition A (CO2 20 %%, 70 %% RH, 20 C, 28 d). Only CO2 and time are converted. point = ensemble k, upper = conformal alpha 0.10 k-bound (global: artifact q=%.3f; Mkn_x: q=%.3f from the same calibration studies). p in {p_lo, p_hat, p_hi} = Mkn_g calibration (%.3f CI %.3f-%.3f); p_x = %.3f recalibrated for Mkn_x (accelerated-only fit; its n-model sees accelerated series only, caveat). EXTRAPOLATION: longest natural series ~5 y; t_SL beyond that is not supported by data and is extremely sensitive to 1/n and p. Grid 61x41 over w/b 0.35-0.65, f 0-0.40, feasible = p2_v2 weq055 constraint set (W=175, w_eq<=0.55, C>=260, B 280-500, data support). break_even: fraction of feasible f>0 cells where CI exceeds CI(f=0) at the same w/b. fixed_weq: slices w/b = w_eq (1-f+0.4 min(f,0.33(1-f))), f 0-0.4, other constraints applied.' % (NG, Q, QX, P_HAT, P_LO, P_HI, P_X),
           p=dict(p_hat=P_HAT, p_lo=P_LO, p_hi=P_HI, p_x=P_X), q=dict(global_=Q, xhat=QX), n_hat_grid_natural=dict(min=float(nx.min()), mean=float(nx.mean()), max=float(nx.max()), global_n=float(NG)), n_feasible=int(feas.sum()),
           cases=cases, pareto_p2v2_weq055=pareto_sum, p2v2_knee=dict(w_b=p2k['w_b'], f=p2k['f'], CO2=p2k['CO2'], X_up_28d_mm=p2k['X_up_28d_mm'], t_SL_point_c40_phat=p2k_ev['n=global|point|c=40|p_hat'], t_SL_upper_c40_phat=p2k_ev['n=global|upper|c=40|p_hat']),
           conventional=conv_sum, nsga_ci=dict(note='min CO2 and max t_SL (global n, point, c=40 mm, p_hat), pop 100, 200 gen, 3 seeds, p2_v2 constraints, knee on log10 t_SL', front_size=len(front2), knee=kn2, front=front2)), 'rq4_ci_rq.json')

# ---------- figures ----------
plt.rcParams.update({'font.size': 7}); WBv = np.linspace(WB_LO, WB_HI, NW); Fv = np.linspace(0, F_HI, NF)
Mx = np.where(feas, store[MAIN]['CI'], np.nan).reshape(sh)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
im = axs[0].pcolormesh(WBv, Fv, np.log10(Mx), cmap='viridis', shading='auto'); plt.colorbar(im, ax=axs[0], label='log$_{10}$ CI (kg CO$_2$/m$^3$/yr)')
rt = store[MAIN]['ratio']; rt = np.where(feas.reshape(sh), rt, np.nan)
if np.nanmin(rt) < 1 < np.nanmax(rt): axs[0].contour(WBv, Fv, rt, levels=[1.0], colors='w', linewidths=1.2, linestyles='--')
g_ = cases[MAIN]['global_opt']; axs[0].plot(g_['w_b'], g_['f'], 'r*', ms=9, label='CI-optimal'); axs[0].plot(p2k['w_b'], p2k['f'], 'wo', mfc='none', ms=6, label='Figure 7 knee'); axs[0].legend(fontsize=5, frameon=True, facecolor='0.7', edgecolor='none', loc='upper right')
axs[0].set_title('(a)', fontsize=7, loc='left')
for lab, key, col in [('global $n$, any $p$, point or upper', MAIN, 'k'), (r'$\hat n(x)$, $\hat p_x$ = %.2f' % P_X, 'n=xhat|point|c=40|p_x', 'tab:green')]:
    r_ = np.where(feas.reshape(sh), store[key]['ratio'], np.nan); lo_, hi_ = np.nanmin(r_), np.nanmax(r_)
    if lo_ < 1 < hi_: axs[1].contour(WBv, Fv, r_, levels=[1.0], colors=col, linewidths=1.1); axs[1].plot([], [], color=col, label=lab)
axs[1].contour(WBv, Fv, np.where(True, mix(gw, gf)['weq'].reshape(sh), 0), levels=[.55], colors='gray', linestyles=':', linewidths=.8); axs[1].plot([], [], 'gray', ls=':', label='w$_{eq}$ = 0.55')
axs[1].legend(fontsize=4.8, frameon=False, loc='upper right'); axs[1].set_title('(b) break-even, CI($f$)/CI(0) = 1', fontsize=7, loc='left')
for ax in axs: ax.set_xlabel('w/b'); ax.set_ylabel('Fly ash share of binder $f$')
fig.tight_layout(); fig.savefig(OUT + r'\plots_export\Fig_CI_map.png', dpi=300); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9)); cm_ = plt.cm.viridis(np.linspace(0, .9, 5))
for c_, wbv in zip(cm_, (.40, .45, .50, .55, .60)):
    j = int(np.argmin(np.abs(WBv - wbv))); ok = np.isfinite(Mx[:, j])
    if not ok.any(): continue
    lo_ = np.where(feas, store['n=global|point|c=40|p_hi']['CI'], np.nan).reshape(sh)[:, j]; hi_ = np.where(feas, store['n=global|point|c=40|p_lo']['CI'], np.nan).reshape(sh)[:, j]
    axs[0].fill_between(Fv[ok], lo_[ok], hi_[ok], color=c_, alpha=.2); axs[0].plot(Fv[ok], Mx[ok, j], color=c_, lw=1.1, label='w/b %.2f' % WBv[j])
axs[0].set_yscale('log'); axs[0].set_title('(a) fixed w/b', fontsize=7, loc='left'); axs[0].legend(fontsize=5, frameon=False)
for we, col in zip((0.45, 0.50, 0.55), ('tab:green', 'tab:blue', 'k')):
    F_ = analyse_fixed[we]; ok = F_['ok']
    for pn, ls in (('p_hat', '-'),): _, ci = fixed_slice('global', 'point', 40, P_HAT, we); axs[1].plot(F_['f'][ok], ci[ok], color=col, lw=1.2, label='w$_{eq}$ = %.2f' % we)
    _, cl = fixed_slice('global', 'point', 40, P_LO, we); _, ch = fixed_slice('global', 'point', 40, P_HI, we); axs[1].fill_between(F_['f'][ok], ch[ok], cl[ok], color=col, alpha=.2)
    _, cu = fixed_slice('global', 'upper', 40, P_HAT, we); axs[1].plot(F_['f'][ok], cu[ok], color=col, lw=.8, ls='--')
    _, cx = fixed_slice('xhat', 'point', 40, P_X, we); axs[1].plot(F_['f'][ok], cx[ok], color=col, lw=.8, ls=':')
axs[1].plot([], [], 'gray', ls='--', label='upper bound'); axs[1].plot([], [], 'gray', ls=':', label='$\\hat n(x)$, $\\hat p_x$'); axs[1].set_yscale('log'); axs[1].legend(fontsize=4.8, frameon=False); axs[1].set_title('(b) fixed w$_{eq}$', fontsize=7, loc='left')
for ax in axs: ax.set_xlabel('Fly ash share of binder $f$'); ax.set_ylabel('CI (kg CO$_2$/m$^3$/yr)')
fig.tight_layout(); fig.savefig(OUT + r'\plots_export\Fig_CI_vs_f.png', dpi=300); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.6, 3.3)); fm = feas
ax.scatter(CO[fm], store[MAIN]['T'][fm], s=1.5, c='lightgray', label='feasible grid')
cx_ = np.array([x['CO2'] for x in front2]); ax.fill_between(cx_, [x['t_SL_p_lo'] for x in front2], [x['t_SL_p_hi'] for x in front2], color='tab:blue', alpha=.25, label=r'$\hat p$ 95% CI')
ax.plot(cx_, [x['t_SL_yr'] for x in front2], 'b-', lw=1.3, label='Pareto front, max $t_{SL}$ and min CO$_2$'); ax.plot(cx_, [x['t_SL_upper_p_hat'] for x in front2], 'b--', lw=.8, label='same front, upper-bound $k$')
ax.plot(kn2['CO2'], kn2['t_SL_yr'], 'b*', ms=10, label='knee'); ax.plot(p2k['CO2'], p2k_ev['n=global|point|c=40|p_hat'], 'r*', ms=10, label='Figure 7 knee')
ax.scatter([x['CO2'] for x in fr], ev_front['n=global|point|c=40|p_hat']['t_SL'], s=6, c='tab:orange', zorder=3, label='Figure 7 Pareto mixes')
ax.scatter(co_conv, ev_conv['n=global|point|c=40|p_hat']['t_SL'], marker='D', facecolors='none', edgecolors='k', s=22, zorder=4, label='conventional, w/b 0.50, $f$ = 0, 0.2, 0.3')
for yv_, lb in ((5, '5 y (longest natural data)'), (50, '50 y'), (100, '100 y')): ax.axhline(yv_, color='gray', ls=':', lw=.7); ax.text(222, yv_, lb, fontsize=4.5, va='bottom', ha='left')
ax.set_yscale('log'); ax.set_xlabel('Partial embodied CO$_2$ (kg/m$^3$)'); ax.set_ylabel('Service life $t_{SL}$ (yr)'); ax.legend(fontsize=4.5, frameon=False, loc='lower right')
fig.tight_layout(); fig.savefig(OUT + r'\plots_export\Fig_pareto_CO2_vs_life.png', dpi=300); plt.close(fig)
print('done')
