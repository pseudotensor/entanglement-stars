#!/bin/bash
# Builder diagnostic of the stalled R4 (2,1) continuation (timing1): assembled Jacobian against a brute-force one.
cd ${QI2_ROOT}/s2b_row || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p debug1
nice -n 10 ${QI2_ROOT}/python code/strand2b_debug_jac.py --case timing1/case_R4_b21_m0.1_gM0.2_gB1_L1_k1.json --out debug1 > debug1/debug.log 2>&1
echo "exit $?" >> debug1/debug.log
