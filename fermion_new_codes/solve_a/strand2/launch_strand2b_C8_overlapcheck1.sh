#!/bin/bash
# Builder check of the C8 library v2 overlap (strand2b_c8_overlap_check_v1.py) against version 1's determinant route; one thread.
cd ${QI2_ROOT}/s2b_row || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p c8ovl1
nice -n 10 ${QI2_ROOT}/python code/strand2b_c8_overlap_check_v1.py --out c8ovl1 > c8ovl1/check.log 2>&1
echo "exit $?" >> c8ovl1/check.log
