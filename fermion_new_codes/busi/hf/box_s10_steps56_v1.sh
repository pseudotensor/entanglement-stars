#!/bin/bash
# RUN_ORDER_S10 steps 5 and 6 (the e0 pass on one member, member 1) on the box (builder).
cd ${QI2_ROOT}/hf
PY=${QI2_ROOT}/python; CODE=${QI2_ROOT}/hf
FAM=${QI2_ROOT}/star_m0.1_g0.25_ell8.json; ENG=${QI2_ROOT}/code
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
nice -n 10 $PY $CODE/hf_star_variance.py --star $FAM --member all --engine-dir $ENG --threads 8 --skip-vertex --out star_var_df > star_var_df.log 2>&1; echo "step5 exit $?" >> s10_status.txt
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
nice -n 10 $PY $CODE/hf_star_variance.py --star $FAM --member 1 --engine-dir $ENG --threads 16 --no-response --out star_var_e0 > star_var_e0.log 2>&1; echo "step6a exit $?" >> s10_status.txt
