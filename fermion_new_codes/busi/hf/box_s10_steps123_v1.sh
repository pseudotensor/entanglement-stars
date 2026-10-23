#!/bin/bash
# RUN_ORDER_S10 steps 1 to 3 in order on the box (builder): the cluster self-test, the cluster benchmark, the star self-test.
cd ${QI2_ROOT}/hf
PY=${QI2_ROOT}/python; CODE=${QI2_ROOT}/hf
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
nice -n 10 $PY $CODE/hf_cluster13.py --selftest --threads 8 > cluster13_selftest.log 2>&1; echo "step1 exit $?" >> s10_status.txt
nice -n 10 $PY $CODE/hf_cluster13.py --threads 8 --out cluster13_results.json > cluster13.log 2>&1; echo "step2 exit $?" >> s10_status.txt
nice -n 10 $PY $CODE/hf_star_variance.py --selftest --threads 8 > star_var_selftest.log 2>&1; echo "step3 exit $?" >> s10_status.txt
