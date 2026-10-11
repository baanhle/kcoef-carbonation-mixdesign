"""Validate the sqrt(CO2) accelerated->natural conversion: k-model trained on ACCELERATED rows only predicts depths of the natural-exposure rows.
k_nat = k_acc(CO2=20) * (CO2_row/20)^p ; X = k_nat * t^n."""
from models import *
cfg = load_cfg()
d, info = load(); d = d.reset_index(drop=True)
acc, nat = d[d.regime == 'accelerated'].reset_index(drop=True), d[d.regime == 'natural'].reset_index(drop=True)
ref = nat.copy(); ref['CO2'] = 20.0
res = {}
for kind in ['Mk', 'Mkn_g', 'Mkm']:
    for pw in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]:
        L = []
        for s in range(10):
            km = KM(kind, 'XGBoost', s, cfg).fit(acc)
            p = km.predict_k(ref) * (nat.CO2.values / 20.0) ** pw * km.expo(nat)
            e = p - nat.Target.values; L.append((float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e))), float(np.mean(e)), float(np.median(p / nat.Target.values))))
        a = np.array(L); res[f'{kind}|p={pw}'] = dict(rmse=dict(mean=float(a[:, 0].mean()), std=float(a[:, 0].std(ddof=1))), mae=float(a[:, 1].mean()), bias=float(a[:, 2].mean()), median_ratio_pred_over_obs=float(a[:, 3].mean()))
        print(kind, pw, res[f'{kind}|p={pw}']['rmse'], flush=True)
    res[f'{kind}|best_p_grid'] = min([float(k.split('=')[1]) for k in res if k.startswith(kind + '|p=')], key=lambda q: res[f'{kind}|p={q}']['rmse']['mean'])
# reference: natural-trained MX on natural test rows is in p1_fixed_test.json (different split); here all 119 natural rows are unseen by construction.
savej(dict(note='Models trained on all accelerated rows (n=%d), evaluated on all natural rows (n=%d, never seen). k evaluated with CO2 set to 20 %% at each natural row\'s own mix/T/RH, scaled by (CO2/20)^p. p grid incl. 0.4/0.5/0.6; best p on grid is diagnostic only.' % (len(acc), len(nat)),
           natural_mean_depth=float(nat.Target.mean()), natural_std=float(nat.Target.std()), results=res), 'p1_sqrtco2_validation.json')
