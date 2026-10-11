# kcoef_mixdesign tools (run with conda env microgen, in this order)
- common.py: data loading, split, helpers (copy of aug_conformal/tools/common.py, OUT -> kcoef_mixdesign)
- models.py: KM class (MX, MX-mono, Mk, Mkn, Mkc, Mkm variants), monotone constraints, within-series exponent fit
- p1a_fixed_test.py: config/model selection (study-grouped validation inside train), fixed test, 20 repeated study-grouped splits
- p1b_extrapolation.py: temporal extrapolation ('last' and 'x2'), combined study+temporal hold-out, cluster bootstrap, figures
- p1c_physics_shap.py: monotonicity, PDP/ICE, sign-constraint violations, SHAP for Mk
- p1d_conformal_k.py: k-space split/Mondrian conformal, writes results/kmodel_artifact.pkl, parity figure
- p1e_sqrtco2.py: validation of the sqrt(CO2) conversion on natural-exposure rows (p = 0..0.6)
- p2_nsga.py: Part 2 (support check, B-sensitivity, NSGA-II, functional unit, condition B, EF sensitivity, figures)
- config_ef.json: emission factors (Wang 2019 base; AusLCI sensitivity)
- run_p1.sh, run_rest.sh: chains
- p1e_calibrated_p.py: calibrate conversion exponent p on natural rows (bootstrap, LOSO), Fig_p_calibration
- p2_v2.py: Part 2 v2 (W=175, w_eq limits, NSGA-II, EF sensitivity, functional unit, condition B with p-hat), writes *_v2
- replot_cover_v2.py: re-plots Fig_cover_requirement_v2 from p2_pareto_v2.json

## RQ analyses (_rq outputs)
- rq_common.py: series table, n-model, Mkn_x, temporal splits
- rq1_n_drivers.py, rq1_mknx_eval.py, rq3_p_mix.py, rq4_ci.py -> results/*_rq.json, plots_export/Fig_n_drivers.png, Fig_extrapolation_rmse_rq.png, Fig_p_vs_mix.png, Fig_CI_map.png, Fig_CI_vs_f.png, Fig_pareto_CO2_vs_life.png

## Other
- table9_strength.py: indicative 28-day strength (Jiang fly-ash Abrams law, 150 mm cube equivalent)
- figS1_support_v2.py, fig_workflow_ga.py, p1b_fig_rq.py: support check and figures
