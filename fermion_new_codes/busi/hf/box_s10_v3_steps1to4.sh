#!/bin/bash
# RUN_ORDER_S10_v3 steps 1 to 4 on the box CPU (builder), in order, 16 threads at most.
cd ${QI2_ROOT}/hf
OUT=${QI2_ROOT}/hf; PY=${QI2_ROOT}/python
STAR=${QI2_ROOT}/star_m0.1_g0.25_ell8.json; ENG=${QI2_ROOT}/code; MF=${QI2_ROOT}/MF
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
st() { echo "$1 $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $OUT/s10_v3_status.txt; }
nice -n 10 $PY hf_cluster13_v3.py --selftest --threads 16 > $OUT/cluster13_v3_selftest.log 2>&1; st "step1 exit $?"
nice -n 10 $PY hf_cluster13_v3.py --threads 16 --out $OUT/cluster13_v3_results.json > $OUT/cluster13_v3.log 2>&1; st "step2 exit $?"
nice -n 10 $PY hf_star_variance_v3.py --selftest > $OUT/star_var_v3_selftest.log 2>&1; st "step3 exit $?"
nice -n 10 $PY hf_star_variance_v3.py --star $STAR --member all --certificates-only --engine-dir $ENG --mf-dir $MF --threads 16 --out $OUT/star_var_v3_cert > $OUT/star_var_v3_cert.log 2>&1; st "step4 exit $?"
st "all done"
