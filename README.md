# k-model carbonation and fly-ash mix design

Code, configuration and result files for the manuscript on learning the carbonation coefficient k (X = k t^n) of fly-ash concrete and designing mixes by carbon per year of carbonation service life.

## Data

The carbonation database is not redistributed here. Download `concrete carbonation data.xls` from Mendeley Data, https://doi.org/10.17632/n9jxmcyzhw.1, and place it in `data/`. The study-grouped split (`study_grouped_split.json`) and the tuned hyperparameters (`task1b_tuned_blackbox.json`) are included.

## Environment

Python 3.12 with the packages in `requirements.txt` (versions used for the paper). Scripts were run on Windows 11; some paths are built with Windows separators.

## Run order

1. `tools/run_p1.sh`: model selection, fixed and repeated study-grouped tests, temporal extrapolation with cluster bootstrap, physical consistency and SHAP, k-space conformal intervals, and calibration of the accelerated-to-natural exponent p. Writes `results/kmodel_artifact.pkl`.
2. `tools/run_rest.sh`: NSGA-II mix design with w_eq limits, carbon intensity per service year, exponent drivers, applicability-domain check and indicative strength.

`tools/README.md` lists every script and its outputs. `results/` holds the JSON results reported in the paper and `figures/` the generated figures.
