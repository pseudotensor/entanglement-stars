#!/bin/bash
B=${QI2_ROOT}/C5_v1d
SL=${QI2_ROOT}/orbitals_lad_W5_k10_tau8_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36d_T0000.00.npz
cd $B
mv run_s8_columns.out run_s8_columns_slow_attempt_killed.out
rm -rf $B/s8_columns
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
nohup nice -n 10 ${QI2_ROOT}/python code/dust_comparator_v1d.py --slice $SL --sigma 8 --modes columns --T-end 12.0 --budget-seconds 30000 --out $B/s8_columns > $B/run_s8_columns.out 2>&1 < /dev/null &
