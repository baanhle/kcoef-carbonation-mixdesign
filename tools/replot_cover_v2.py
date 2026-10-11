"""Re-plot Fig_cover_requirement_v2 from results/p2_pareto_v2.json (adds cover lines labels, ylim)."""
from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
J = json.load(open(OUT + r'\results\p2_pareto_v2.json')); fr = J['cases']['weq055']['front']; ph = J['p_hat']['p_hat']; cx = [p['CO2'] for p in fr]
plt.rcParams.update({'font.size': 7}); fig, ax = plt.subplots(figsize=(3.4, 2.9))
ax.fill_between(cx, [p['X_up_50y_p_hi'] for p in fr], [p['X_up_50y_p_lo'] for p in fr], color='tab:blue', alpha=.3, label='band: $\hat p$ 95% CI'); ax.plot(cx, [p['X_up_50y_p_hat'] for p in fr], 'b-', lw=1, label='$\hat p$ = %.2f' % ph)
for cv in (30, 40, 50): ax.axhline(cv, color='gray', ls=':', lw=.7); ax.text(cx[0], cv * 1.03, f'{cv} mm cover', fontsize=5, color='gray')
ax.set_yscale('log'); ax.set_ylim(25, 450); ax.set_yticks([30, 50, 100, 200, 400]); ax.set_yticklabels(['30', '50', '100', '200', '400'])
ax.set_xlabel('embodied CO$_2$ (kg/m$^3$)'); ax.set_ylabel('cover needed X$_{up}$(50 y) (mm)'); ax.set_title('Condition B: EXTRAPOLATION, w$_{eq}\leq$0.55 front', fontsize=6); ax.legend(fontsize=5, frameon=False, loc='upper right'); fig.tight_layout()
fig.savefig(OUT + r'\plots_export\Fig_cover_requirement_v2.png', dpi=300)
