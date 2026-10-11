"""FigS1_support_map_v2: applicability domain of the v2 design space (W=175, B=W/(w/b)). Figure only; no results file is written."""
import sys; sys.path.insert(0, r'd:\Research\TienThe\04_kcoef_mixdesign\tools')
from p2_v2 import *
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
P2 = json.load(open(R + 'p2_pareto_v2.json'))['cases']['weq055']
Wb = np.linspace(WB_LO, WB_HI, 121); Fv = np.linspace(0, F_HI, 81); gw, gf = np.meshgrid(Wb, Fv); sh = gw.shape; m = mix(gw.ravel(), gf.ravel())
df = to_df(m, COND, 'accelerated'); ex, kd = support(df); G = constraints(m, .55, df)
supp_ok = (ex <= 1e-9) & (kd <= thr)
cat = np.zeros(len(gw.ravel()), int)  # 0 feasible, 1 w_eq>0.55, 2 B/C bound, 3 outside data support
cat[~supp_ok] = 3; cat[(G[:, 1:4] > 1e-9).any(1)] = 2; cat[G[:, 0] > 1e-9] = 1; cat = cat.reshape(sh)
cols = ['#4daf4a', '#bdbdbd', '#fdb863', '#984ea3']
plt.rcParams.update({'font.size': 7}); fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
axs[0].pcolormesh(Wb, Fv, cat, cmap=ListedColormap(cols), vmin=-.5, vmax=3.5, shading='auto')
axs[0].contour(Wb, Fv, m['weq'].reshape(sh), levels=[.55], colors='k', linewidths=1, linestyles='--')
im = axs[1].pcolormesh(Wb, Fv, kd.reshape(sh), cmap='magma_r', shading='auto'); plt.colorbar(im, ax=axs[1], label='5-NN distance to training mixes (z-units)')
axs[1].contour(Wb, Fv, m['weq'].reshape(sh), levels=[.55], colors='k', linewidths=1, linestyles='--')
fr = P2['front']; kn = P2['knee']
for ax in axs:
    ax.scatter([x['w_b'] for x in fr], [x['f'] for x in fr], s=7, c='w', edgecolors='k', linewidths=.5, zorder=3)
    ax.scatter(kn['w_b'], kn['f'], marker='*', s=90, c='r', edgecolors='k', linewidths=.5, zorder=4)
    ax.scatter([.5] * 3, [0, .2, .3], marker='D', s=18, facecolors='none', edgecolors='b', zorder=4); ax.set_xlabel('w/b'); ax.set_ylabel('FA share of binder f')
h = [Patch(color=cols[0], label='feasible (all constraints + supported)'), Patch(color=cols[1], label='w$_{eq}$ > 0.55')] + ([Patch(color=cols[2], label='B or C bound violated')] if (cat == 2).any() else []) + ([Patch(color=cols[3], label='outside data support')] if (cat == 3).any() else []) + [
     plt.Line2D([], [], color='k', ls='--', label='w$_{eq}$ = 0.55'), plt.Line2D([], [], marker='o', mfc='w', mec='k', ls='', ms=3, label='Pareto points'), plt.Line2D([], [], marker='*', color='r', ls='', ms=7, label='knee'), plt.Line2D([], [], marker='D', mfc='none', mec='b', ls='', ms=4, label='conventional (w/b 0.5)')]
fig.legend(handles=h, fontsize=5, frameon=False, ncol=4, loc='lower center'); axs[0].set_title('a) constraint status', fontsize=6.5); axs[1].set_title('b) kNN support: max %.2f < p95 threshold %.2f everywhere' % (kd.max(), thr), fontsize=6.5)
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(OUT + r'\plots_export\FigS1_support_map_v2.png', dpi=300); plt.close(fig)
print('cells', {int(k): int((cat == k).sum()) for k in range(4)}, 'front in feasible', int(np.mean([x['feasible'] for x in fr]) * len(fr)), len(fr))
