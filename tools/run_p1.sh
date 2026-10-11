#!/bin/bash
# Part 1: learning formulations, extrapolation, physics checks, conformal k
cd "$(dirname "$0")"; mkdir -p ../logs
for s in p1a_fixed_test p1b_extrapolation p1c_physics_shap p1d_conformal_k p1e_sqrtco2 p1e_calibrated_p; do python $s.py > ../logs/$s.log 2>&1; done
