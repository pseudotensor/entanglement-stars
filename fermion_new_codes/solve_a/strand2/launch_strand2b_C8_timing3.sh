#!/bin/bash
# Builder timing of the C8 2b route with library v3 (strand2b_row_C8_v3.py: the common-sea Schur overlap, the projector rank fixed at
# the flat start): block (2,0), m = 0.1, (0.2, 1), Lambda 1, k = 1; solve and lock only.  One thread.  (Timing 2's (2,0) case, on
# library v1, was stopped after 18 min: superseded by the v3 route.)
cd ${QI2_ROOT}/s2b_row || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${QI2_ROOT}/python
mkdir -p c8timing3
/usr/bin/time -v nice -n 10 $PY code/strand2b_row_C8_v3.py --out c8timing3 --block 2,0 --m 0.1 --gM 0.2 --gB 1 --Lam 1 --k 1 > c8timing3/case_2,0.log 2>&1
echo "exit $?" >> c8timing3/case_2,0.log
