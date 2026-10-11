"""P1 (d) physical consistency: non-monotone curve fraction, PDP/ICE sign checks; (e) SHAP for Mk."""
from models import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
cfg = load_cfg()
d, info = load()
tr, te = d[~d.is_test].reset_index(drop=True), d[d.is_test].reset_index(drop=True)

# ---- monotonicity of X(t) over a sqrt_t grid for each unique test mix ----
mixes = te.drop_duplicates('series').reset_index(drop=True)
grid = np.linspace(1, 30, 30)
G = pd.concat([mixes] * len(grid), ignore_index=True)
G['sqrt_t'] = np.repeat(grid, len(mixes)); nm = len(mixes)
mono = {}
for k in KINDS:
    fr, frm, worst = [], [], []
    for s in range(10):
        km = KM(k, 'XGBoost', s, cfg).fit(tr)
        P = km.predict(G).reshape(len(grid), nm)  # [t, mix]
        dd = np.diff(P, axis=0)
        fr.append(float((dd.min(0) < -1e-6).mean())); frm.append(float((dd.min(0) < -0.5).mean())); worst.append(float(dd.min()))
    mono[k] = dict(frac_nonmonotone=dict(mean=float(np.mean(fr)), std=float(np.std(fr, ddof=1))),
                   frac_decrease_gt_0p5mm=dict(mean=float(np.mean(frm)), std=float(np.std(frm, ddof=1))), worst_single_step_decrease_mm=float(np.min(worst)))
    print(k, mono[k], flush=True)

# ---- PDP / ICE of k-hat (Mk model, seed 0, trained on train) ----
km = KM('Mk', 'XGBoost', 0, cfg).fit(tr)
bg = tr.drop_duplicates(NONTIME).reset_index(drop=True)
rng = np.random.default_rng(0)
ice_idx = rng.choice(len(bg), 40, replace=False)
q = lambda c, a, b: np.linspace(bg[c].quantile(a), bg[c].quantile(b), 25)
panels = {'w/b': ('w/b', q('w/b', .02, .98), False), 'FA (w/c from w/b, B=400)': ('FA', np.linspace(0, 250, 25), True),
          'FA (w/c fixed)': ('FA', np.linspace(0, 250, 25), False), 'CO2': ('CO2', np.logspace(np.log10(.03), 2, 30), False),
          'RH': ('RH', np.linspace(40, 90, 26), False), 'T': ('T', q('T', .02, .98), False)}
pdp, checks = {}, {}
fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.4))
for ax, (name, (col, gv, adj)) in zip(axs.ravel(), panels.items()):
    M = np.zeros((len(gv), len(bg)))
    for i, v in enumerate(gv):
        B = bg.copy(); B[col] = v
        if adj: B['w/c'] = B['w/b'] / (1 - v / 400.0)
        M[i] = km.predict_k(B)
    p = M.mean(1); pdp[name] = dict(grid=gv.tolist(), pdp_k=p.tolist())
    for j in ice_idx: ax.plot(gv, M[:, j], color='gray', alpha=.25, lw=.5)
    ax.plot(gv, p, 'r', lw=1.6); ax.set_yscale('log'); ax.set_xlabel(name); ax.set_ylabel('$\\hat k$ (mm/$\\sqrt{d}$)')
    if col == 'CO2': ax.set_xscale('log')
    inc = (np.diff(M, axis=0) > 0).mean()
    lo, hi = int(.1 * len(gv)), int(.9 * len(gv))
    checks[name] = dict(pdp_end_over_start=float(p[-1] / p[0]), pdp_p90_over_p10=float(p[hi] / p[lo]), frac_ice_steps_increasing=float(inc),
                        monotone_increasing_pdp=bool((np.diff(p) >= -1e-9).all()), argmax_grid=float(gv[int(p.argmax())]))
rh = np.array(pdp['RH']['grid']); pr = np.array(pdp['RH']['pdp_k'])
checks['RH']['pdp_at_40_60_90'] = [float(pr[np.argmin(abs(rh - v))]) for v in (40, 60, 90)]
checks['RH']['peak_in_50_70'] = bool(50 <= checks['RH']['argmax_grid'] <= 70)
fig.tight_layout(); fig.savefig(OUT + r'\plots_export\Fig_monotonicity_pdp.png', dpi=300); plt.close(fig)
exp = {'w/b': 'increase', 'FA (w/c from w/b, B=400)': 'increase', 'CO2': 'increase', 'RH': 'peak 50-70%', 'T': 'increase (not required)'}
savej(dict(note='Monotonicity: unique test mixes (n=%d) x sqrt_t grid 1..30 (30 pts), XGB seeds 0-9. PDP/ICE of k-hat from Mk (seed 0, train 1690 rows), background = unique non-time train rows (n=%d). FA panel with w/c recomputed assumes binder 400 kg/m3.' % (nm, len(bg)),
           monotonicity=mono, pdp_checks=checks, expected_sign=exp, pdp=pdp), 'p1_physical_consistency.json')


# ---- sign-constraint violations of unconstrained models (ICE over unique test mixes) ----
cons = {'w/b': 1, 'w/c': 1, 'CO2': 1, 'T': 1, 't_cure': -1, 'sqrt_t': 1}
Ubg = tr.drop_duplicates(NONTIME)
viol = {}
for kind in ['MX', 'Mk', 'MXm3', 'Mkm']:
    viol[kind] = {}
    for c, sg in cons.items():
        if kind in ('Mk', 'Mkm') and c == 'sqrt_t': continue
        gv = np.logspace(np.log10(max(Ubg[c].quantile(.05), 0.03)), np.log10(Ubg[c].quantile(.95)), 15) if c == 'CO2' else np.linspace(Ubg[c].quantile(.05) if c != 'sqrt_t' else 1, Ubg[c].quantile(.95) if c != 'sqrt_t' else 30, 15)
        fr, frm = [], []
        for s_ in range(3):
            kmv = KM(kind, 'XGBoost', s_, cfg).fit(tr)
            M_ = np.zeros((len(gv), nm))
            for i, v in enumerate(gv):
                B_ = mixes.copy(); B_[c] = v
                if kind.startswith('MX') and c != 'sqrt_t': B_['sqrt_t'] = 9.0
                M_[i] = kmv.predict(B_) if kind.startswith('MX') else kmv.predict_k(B_)
            dlt = np.diff(M_, axis=0) * sg; rel = dlt / np.maximum(np.abs(M_[:-1]), 1e-9)
            fr.append(float((dlt.min(0) < -1e-9).mean())); frm.append(float((rel.min(0) < -0.05).mean()))
        viol[kind][c] = dict(frac_mixes_any_violation=float(np.mean(fr)), frac_mixes_drop_gt5pct=float(np.mean(frm)), sign_required=sg)
    print('viol', kind, {c: round(v['frac_mixes_any_violation'], 3) for c, v in viol[kind].items()}, flush=True)
savej(dict(note='Sign constraints: +w/b, +w/c, +CO2, +T, -t_cure, +sqrt_t (RH, FA unconstrained). ICE over %d unique test mixes, 15-pt grids (5-95 pct of train), XGB 3 seeds; MX evaluated at sqrt_t=9 for non-time variables. Fractions = mixes whose single-variable curve violates the sign anywhere / drops >5%% relative.' % nm,
           violations=viol), 'p1_constraint_violations.json')

# ---- SHAP for Mk ----
import shap
X = tr[NONTIME]
try:
    sv = shap.TreeExplainer(km.m).shap_values(X.values); how = 'shap.TreeExplainer'
except Exception as e:
    import xgboost as xgb
    sv = km.m.get_booster().predict(xgb.DMatrix(X.values, feature_names=None), pred_contribs=True)[:, :-1]; how = 'xgboost pred_contribs (TreeSHAP) fallback: ' + str(e)[:80]
ma = np.abs(sv).mean(0)
savej(dict(method=how, space='log k' if cfg['log'] else 'k', n_rows=len(X), mean_abs_shap=dict(zip(NONTIME, map(float, ma))),
           rank=[NONTIME[i] for i in np.argsort(-ma)]), 'p1_shap.json')
plt.rcParams.update({'font.size': 6}); plt.figure(); shap.summary_plot(sv, X, feature_names=NONTIME, show=False, plot_size=None)
plt.gcf().set_size_inches(3.4, 2.8); plt.xlabel('SHAP value (%s)' % ('log k' if cfg['log'] else 'k'), fontsize=6); plt.title('SHAP, Mk', fontsize=7)
for a_ in plt.gcf().axes: a_.tick_params(labelsize=6); a_.yaxis.label.set_size(6)
plt.tight_layout(); plt.savefig(OUT + r'\plots_export\Fig_shap_k.png', dpi=300); plt.close('all')
print(checks, ma)
