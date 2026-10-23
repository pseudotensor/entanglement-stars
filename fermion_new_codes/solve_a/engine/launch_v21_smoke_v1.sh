#!/bin/bash
# The smoke test of engine_collapse_v21 (fields v15)'s massless-shell member on box GPU 0 (r_s 16, m 0, g 1/4, 512 sites, ell 4, tau 4, T/rs 1.5): the
# initial data's passes, the packets' momentum signs, the first records. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p V21T
echo "v21 smoke start $(date -u +%H:%M:%S)" >> V21T/status.txt
nice -n 10 ${QI2_ROOT}/python code/engine_collapse_v21.py --m 0 --g 0.25 --rs 16 --nsites 512 --ell 4 --tau 4 --T-over-rs 1.5 --kappa-sea 10 --reference R2 --site-fields averaged --massless-shell --out V21T --gpu 0 --tag t21_ms_rs16 > V21T/log_t21_ms_rs16.log 2>&1
echo "v21 smoke exit $? $(date -u +%H:%M:%S)" >> V21T/status.txt
