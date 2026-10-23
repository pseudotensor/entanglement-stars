#!/bin/bash
# Strand 0 builder diagnostic of control C5e (U = 0.001, 0.01, 0.1 on B0 and B3_e0.2).
cd ${QI2_ROOT}/s0_source_allocation || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
nice -n 10 ${QI2_ROOT}/python code/strand0_c5e_diag.py --out full_v2 > full_v2/c5e_diag.log 2>&1
echo "c5e_diag exit $?" >> full_v2/status.txt
