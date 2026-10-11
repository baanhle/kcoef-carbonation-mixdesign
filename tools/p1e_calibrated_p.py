"""Calibrate the accelerated->natural conversion exponent p: X_nat = X_k(CO2=20 %, row's mix/T/RH) * (CO2/20)^p, least squares on natural rows.
Accelerated-only models (trained on accelerated rows only; natural rows never seen). Ensemble of 10 XGB seeds."""
from models import *
from scipy.optimize import minimize_scalar
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
cfg = load_cfg(); d, info = load(); d = d.reset_index(drop=True)
acc, nat = d[d.regime == 'accelerated'].reset_index(drop=True), d[d.regime == 'natural'].reset_index(drop=True)
ref = nat.copy(); ref['CO2'] = 20.0
y, r, g = nat.Target.values, nat.CO2.values / 20.0, nat.ref_group.values
studies = np.unique(g); print('natural rows', len(nat), 'studies', len(studies), {s: int((g == s).sum()) for s in studies}, flush=True)


def fitp(base, y, r):
    f = lambda p: np.sum((base * r ** p - y) ** 2)
    return float(minimize_scalar(f, bounds=(-0.5, 1.5), method='bounded', options=dict(xatol=1e-5)).x)


rm = lambda b, y, r, p: (float(np.sqrt(np.mean((b * r ** p - y) ** 2))), float(np.mean(b * r ** p - y)), float(np.mean(np.abs(b * r ** p - y))))
res = {}; keep = {}
rng = np.random.default_rng(0)
for kind in ['Mk', 'Mkn_g']:
    base = np.mean([KM(kind, 'XGBoost', s, cfg).fit(acc).predict_k(ref) * KM(kind, 'XGBoost', s, cfg).fit(acc).expo(nat) for s in range(10)], 0) if False else None
    B = []
    for s in range(10):
        km = KM(kind, 'XGBoost', s, cfg).fit(acc); B.append(km.predict_k(ref) * km.expo(nat))
    base = np.mean(B, 0)
    ph = fitp(base, y, r)
    boot = [fitp(base[i], y[i], r[i]) for i in (np.concatenate([np.where(g == s)[0] for s in rng.choice(studies, len(studies))]) for _ in range(2000))]
    rowb = [fitp(base[i], y[i], r[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(2000))]
    loso = {str(s): fitp(base[g != s], y[g != s], r[g != s]) for s in studies}
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    res[kind] = dict(p_hat=ph, ci95_cluster_bootstrap=ci(boot), n_distinct_cluster_boot=len(set(np.round(boot, 4))), ci95_row_bootstrap=ci(rowb), loso_p=loso, loso_range=[min(loso.values()), max(loso.values())],
                     at_p_hat=dict(zip(['rmse', 'bias', 'mae'], rm(base, y, r, ph))), at_p_0p5=dict(zip(['rmse', 'bias', 'mae'], rm(base, y, r, .5))),
                     at_ci_lo=dict(zip(['rmse', 'bias', 'mae'], rm(base, y, r, ci(boot)[0]))), at_ci_hi=dict(zip(['rmse', 'bias', 'mae'], rm(base, y, r, ci(boot)[1]))))
    keep[kind] = (base, ph, boot)
    print(kind, json.dumps(res[kind], default=float)[:900], flush=True)
savej(dict(note='p fitted by least squares on natural rows (n=%d, %d studies: %s). Accelerated-only models, 10-seed ensemble; X_k evaluated at CO2=20 %% for each natural row, scaled by (CO2/20)^p. Cluster bootstrap resamples natural STUDIES (2000 reps); with so few studies the cluster CI is coarse and a row bootstrap (2000 reps, rows treated as independent, over-optimistic) is also given. LOSO = p refitted leaving one natural study out. NOTE natural CO2 is ~constant (0.03-0.05 %%), so p is essentially a single level-calibration constant (p and a multiplicative scale are confounded) - valid only for ~0.04 %% CO2, not a validated CO2 exponent.' % (len(nat), len(studies), list(studies)),
           natural_mean_depth=float(y.mean()), natural_co2_range=[float(nat.CO2.min()), float(nat.CO2.max())], results=res), 'p1e_calibrated_p.json')
# figure
kind = 'Mkn_g'; base, ph, boot = keep[kind]
plt.rcParams.update({'font.size': 7}); fig, ax = plt.subplots(figsize=(3.4, 3.2))
ax.scatter(y, base * r ** .5, s=6, c='tab:red', label='$p$ = 0.5, RMSE %.1f mm' % res[kind]['at_p_0p5']['rmse'])
ax.scatter(y, base * r ** ph, s=6, c='tab:blue', label=r'$\hat p$ = %.2f, RMSE %.1f mm' % (ph, res[kind]['at_p_hat']['rmse']))
m = max(y.max(), (base * r ** ph).max()) * 1.05; ax.plot([0, m], [0, m], 'k--', lw=.7); ax.set_xlabel('Observed natural depth (mm)'); ax.set_ylabel('Predicted natural depth (mm)'); ax.legend(fontsize=5.5, frameon=False, loc='upper left')
ia = fig.add_axes([0.6, 0.5, 0.33, 0.2]); ia.hist(boot, bins=20, color='gray'); ia.axvline(ph, color='b'); ia.set_xlabel(r'$\hat p$, study bootstrap', fontsize=5); ia.tick_params(labelsize=5); ia.set_yticks([])
fig.tight_layout(); fig.savefig(OUT + r'\plots_export\Fig_p_calibration.png', dpi=300); plt.close(fig)
