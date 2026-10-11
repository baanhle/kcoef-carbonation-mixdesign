#!/bin/bash
# Part 2 and RQ analyses (needs results/kmodel_artifact.pkl from run_p1.sh)
cd "$(dirname "$0")"; mkdir -p ../logs
for s in p2_v2 replot_cover_v2 rq1_n_drivers rq1_mknx_eval rq3_p_mix rq4_ci figS1_support_v2 table9_strength; do python $s.py > ../logs/$s.log 2>&1; done
