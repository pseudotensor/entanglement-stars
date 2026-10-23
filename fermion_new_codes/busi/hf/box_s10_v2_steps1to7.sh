#!/bin/bash
# RUN_ORDER_S10_v2 steps 1 to 7 on the box CPU (builder), in order, 16 threads at most.
cd ${QI2_ROOT}/hf
OUT=${QI2_ROOT}/hf; PY=${QI2_ROOT}/python
STAR=${QI2_ROOT}/star_m0.1_g0.25_ell8.json; ENG=${QI2_ROOT}/code; MF=${QI2_ROOT}/MF
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
st() { echo "$1 $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $OUT/s10_v2_status.txt; }
nice -n 10 $PY hf_cluster13_v2.py --selftest --threads 16 > $OUT/cluster13_v2_selftest.log 2>&1; st "step1 exit $?"
nice -n 10 $PY hf_cluster13_v2.py --threads 16 --out $OUT/cluster13_v2_results.json > $OUT/cluster13_v2.log 2>&1; st "step2 exit $?"
nice -n 10 $PY hf_cluster13_v2.py --selftest --hf-rule aufbau --threads 16 > $OUT/cluster13_v2_selftest_aufbau_rule.log 2>&1; st "step3 exit $?"
nice -n 10 $PY gap06_addon_symmetry_refsector_v2.py --gap06-dir ${QI2_ROOT}/hf --gap06-json $OUT/gap06_full_v1.json --refsector-exact --threads 16 --out $OUT/gap06_addon_v2.json > $OUT/gap06_addon_v2.log 2>&1; st "step4 exit $?"
nice -n 10 $PY hf_star_variance_v2.py --selftest > $OUT/star_var_v2_selftest.log 2>&1; st "step5 exit $?"
nice -n 10 $PY hf_star_variance_v2.py --star $STAR --member all --skip-vertex --engine-dir $ENG --mf-dir $MF --threads 16 --out $OUT/star_var_v2_df > $OUT/star_var_v2_df.log 2>&1; st "step6 exit $?"
nice -n 10 $PY hf_star_variance_v2.py --star $STAR --member 1 --response-mode both --engine-dir $ENG --mf-dir $MF --threads 16 --out $OUT/star_var_v2_e_check > $OUT/star_var_v2_e_check_m1.log 2>&1; st "step7 exit $?"
st "all done"
