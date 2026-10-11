"""Fig1_workflow.png (17 cm wide) and GA.png (13 x 5 cm). Numbers are read from results/*.json (no new values created)."""
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
O = r'd:\Research\TienThe\04_kcoef_mixdesign'; R = O + '/results/'
J4 = json.load(open(R + 'rq4_ci_rq.json')); J1 = json.load(open(R + 'rq1_n_variance_rq.json')); C = J4['cases']['n=global|point|c=40|p_hat']
go, kn, cv = C['global_opt'], J4['p2v2_knee'], J4['conventional']['n=global|point|c=40|p_hat']
p = J4['p']; icc = J1['null']['icc']
CM = 1 / 2.54


def box(ax, x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.0,rounding_size=0.12', fc=fc, ec='k', lw=.7))
    ax.text(x + w / 2, y + h - .17, title, ha='center', va='top', fontsize=6.8, fontweight='bold')
    ax.text(x + w / 2, y + h - .55, body, ha='center', va='top', fontsize=6.3, linespacing=1.35)


def arrow(ax, a, b, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=8, lw=.9, color='k', connectionstyle=f'arc3,rad={rad}'))


# ---------------- Fig1 ----------------
fig = plt.figure(figsize=(17 * CM, 6.9 * CM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 17); ax.set_ylim(0, 6.9); ax.axis('off')
w, h = 5.0, 2.35; xs = [0.25, 6.0, 11.75]; y1, y2 = 4.25, 0.7
box(ax, xs[0], y1, w, h, 'Database', '2,163 rows, 45 studies\n(accelerated + natural carbonation)\n7 inputs: w/c, w/b, FA, CO$_2$, RH, T,\ncuring; target: depth X (mm)', '#e8eef7')
box(ax, xs[1], y1, w, h, 'Validation', 'study-grouped hold-out (20 splits)\n+ temporal extrapolation within series\n(last 1-2 times; t $\\geq$ 2x last train)\n+ combined study + temporal', '#e8eef7')
box(ax, xs[2], y1, w, h, 'Time exponent', 'series-level n (278 series)\nMixedLM: ICC = %.2f; n unpredictable\nfrom mix (grouped-CV R$^2$ $\\approx$ 0)\n$\\Rightarrow$ global n $\\approx$ 0.45' % icc, '#fdf0d5')
box(ax, xs[0], y2, w, h, 'Learning formulations', 'MX (learn X) vs Mk / Mkn (learn k)\nMk extrapolates better in time\nconformal interval in k-space\n(width grows as t$^n$)', '#fdf0d5')
box(ax, xs[1], y2, w, h, 'Accelerated-to-natural conversion', 'accelerated $\\rightarrow$ natural\n$k_{nat}=k_{acc}(0.04/20)^{p}$\n$\\hat p$ = %.2f (95%% CI %.2f-%.2f)\n119 natural rows, 8 studies' % (p['p_hat'], p['p_lo'], p['p_hi']), '#fdf0d5')
box(ax, xs[2], y2, w, h, 'Mix design and carbon intensity', 'NSGA-II: CO$_2$ vs X$_{up28}$; CO$_2$ vs $t_{SL}$\ncarbon intensity CI = CO$_2$/$t_{SL}$\n(kg CO$_2$/m$^3$/yr), W = 175 kg/m$^3$\nw$_{eq}\\leq$0.55, data-support constraint', '#d9ead3')
for i in (0, 1): arrow(ax, (xs[i] + w, y1 + h / 2), (xs[i + 1], y1 + h / 2))
for i in (0, 1): arrow(ax, (xs[i] + w, y2 + h / 2), (xs[i + 1], y2 + h / 2))
xe, ym = xs[2] + w / 2, 3.95
ax.plot([xe, xe, xs[0] + w / 2], [y1, ym, ym], 'k', lw=.9); arrow(ax, (xs[0] + w / 2, ym + .001), (xs[0] + w / 2, y2 + h))
ax.text(8.5, ym - .13, 'global n and conformal bound feed the service-life calculation', ha='center', va='top', fontsize=5.5, style='italic')
ax.text(8.5, 0.3, 't$_{SL}$ beyond ~5 y of natural exposure is extrapolation (longest natural series ~5 y)', ha='center', fontsize=5.5, style='italic', color='#8b0000')
fig.savefig(O + r'\plots_export\Fig1_workflow.png', dpi=300); plt.close(fig)

# ---------------- GA ----------------
fig = plt.figure(figsize=(13 * CM, 5 * CM)); ax = fig.add_axes([0.1, 0.24, 0.40, 0.5])
lab = ['lowest CI', 'knee', 'conv.', 'conv.']; lab = [l + chr(10) + t for l, t in zip(lab, ['%.2f, %.2f' % (go['w_b'], go['f']), '%.2f, %.2f' % (kn['w_b'], kn['f']), '0.50, 0.00', '0.50, 0.30*'])]
ci = [go['CI'], C and J4['pareto_p2v2_weq055']['n=global|point|c=40|p_hat']['CI_at_p2_knee'], cv[0]['CI'], cv[2]['CI']]; co = [go['CO2'], kn['CO2'], cv[0]['CO2'], cv[2]['CO2']]
cl = ['#2e7d32', '#66a61e', '#9e9e9e', '#d95f02']; b = ax.bar(range(4), ci, color=cl, width=.7)
b[3].set_hatch('//'); b[3].set_edgecolor('w')  # w_eq-infeasible mix, kept for contrast
for i, (c_, o_) in enumerate(zip(ci, co)): ax.text(i, c_ + .3, '%.1f' % c_, ha='center', fontsize=6, fontweight='bold'); ax.text(i, .3, '%.0f' % o_, ha='center', fontsize=5, color='w')
ax.set_xticks(range(4)); ax.set_xticklabels(lab, fontsize=5); ax.set_xlabel('mix (w/b, f)', fontsize=5.5, labelpad=1); ax.set_ylabel('CI (kg CO$_2$/m$^3$/yr)', fontsize=6); ax.tick_params(axis='y', labelsize=5.5); ax.set_ylim(0, 17)
ax.set_title('per service year (white = kg CO$_2$/m$^3$)', fontsize=5.8)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.text(0.5, 0.93, 'Per service year, lowering w/b beats adding fly ash', ha='center', fontsize=8.5, fontweight='bold')
t = ('Fly ash cuts CO$_2$ per m$^3$ (knee: %.0f vs %.0f kg/m$^3$),\nbut the shorter carbonation life offsets it:\n'
     '- at fixed w/b, CI rises with f in %.0f%% of feasible mixes\n- CI-optimal mix is a low-w/b, low-f concrete\n- $\\hat p$ = %.2f, n = 0.45; $t_{SL}$ is extrapolated\n  (point estimate, 40 mm cover)') % (kn['CO2'], cv[0]['CO2'], 100 * C['break_even']['frac_feasible_cells_with_f_gt0_where_CI_exceeds_f0'], p['p_hat'])
fig.text(0.54, 0.7, t, ha='left', va='top', fontsize=5.3, linespacing=1.5)
fig.text(0.1, 0.01, '*exceeds w$_{eq}$ 0.55 (EN 206 k-value); shown for contrast only', fontsize=4.5, color='#555')
fig.text(0.54, 0.06, '2,163 rows, 45 studies | XGBoost k-model | NSGA-II', fontsize=4.8, color='#555')
fig.savefig(O + r'\plots_export\GA.png', dpi=300); plt.close(fig)
print(ci, co)
