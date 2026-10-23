#!/bin/bash
# Builder timing of the C8 2b route with library v5 (strand2b_row_C8_v5.py: FD Jacobian of F, noise-floor Newton): block (2,0), m = 0.1,
# (0.2, 1), Lambda 1, k = 1; solve and lock only.  One thread.  (Timing 4, library v4, lost the branch at lambda 0.052: its Newton asked
# for 1e-12 below the residual's noise floor at every path step.)
cd ${QI2_ROOT}/s2b_row || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${QI2_ROOT}/python
mkdir -p c8timing5
/usr/bin/time -v nice -n 10 $PY code/strand2b_row_C8_v5.py --out c8timing5 --block 2,0 --m 0.1 --gM 0.2 --gB 1 --Lam 1 --k 1 > c8timing5/case_2,0.log 2>&1
echo "exit $?" >> c8timing5/case_2,0.log
