#!/bin/bash
# Builder diagnostic: the C8 analytic Jacobian against central differences of F (strand2b_c8_debug_jac_v1.py, library v4); one thread.
cd ${QI2_ROOT}/s2b_row || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p c8jac1
nice -n 10 ${QI2_ROOT}/python code/strand2b_c8_debug_jac_v1.py --out c8jac1 > c8jac1/check.log 2>&1
echo "exit $?" >> c8jac1/check.log
