#!/bin/bash
# run_gate_v16f_rows_v1.sh (busi, analyst; ReviewA_s4fail_v1 sec. 2): the bitwise row gate of v16f against v16d, box CPU, nice 10,
# four processes x 2 threads (8 in all), QI2_FIELDS_* unset.
set -u
B=${QI2_ROOT}/fold16f; PY=${QI2_ROOT}/python; CODE=${QI2_ROOT}/code16d_flat
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
unset QI2_FIELDS_ROUTE QI2_FIELDS_WARM H2_PW_ONE H2_LOCAL_SPLINE PYTHONPATH
cd $B
G="$PY -I code/gate_v16f_bitwise_v1.py rows --code $CODE --v16d kit/engine_fields_v16d.py --v16f kit/engine_fields_v16f.py"
(nice -n 10 $G --fields ${QI2_ROOT}/collapse_lad_ISO_128_s4_nkqh2_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq_fields.npz --last 30 --tag iso128 --out gate/rows_iso128.json > logs/rows_iso128.log 2>&1; echo "iso128 rc $?" >> logs/rows_done.txt) &
(nice -n 10 $G --fields ${QI2_ROOT}/collapse_RUNG2c40b_srf_fields_T0012.00.npz --last 20 --tag rung2 --out gate/rows_rung2.json > logs/rows_rung2.log 2>&1; echo "rung2 rc $?" >> logs/rows_done.txt) &
(nice -n 10 $G --fields ${QI2_ROOT}/collapse_lad_F256_s8_fbt05_m0p2_rs256_n4096_R04_k15_tau16_ell8_prep4_A_R2_avg_newton_v36nq_rd13mfb_fields.npz --tag f256s8 --out gate/rows_f256s8.json > logs/rows_f256s8.log 2>&1; echo "f256s8 rc $?" >> logs/rows_done.txt) &
(nice -n 10 $G --fields ${QI2_ROOT}/collapse_lad_F256_s4_fbt05_m0p2_rs256_n4096_R04_k15_tau16_ell4_prep4_A_R2_avg_newton_v36nq_rd13mfb_fields.npz --tag f256s4 --out gate/rows_f256s4.json > logs/rows_f256s4.log 2>&1; echo "f256s4 rc $?" >> logs/rows_done.txt) &
wait
echo "all done" >> logs/rows_done.txt
