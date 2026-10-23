#!/bin/bash
# Builder timing of the C8 2b route with library v4 (strand2b_row_C8_v4.py: the Kronecker build, the Schur overlap, the projector rank
# fixed at the flat start): block (2,0), m = 0.1, (0.2, 1), Lambda 1, k = 1; solve and lock only.  One thread.  (Timing 3, library v3,
# was stopped: its build was the cost.)
cd ${QI2_ROOT}/s2b_row || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${QI2_ROOT}/python
mkdir -p c8timing4
/usr/bin/time -v nice -n 10 $PY code/strand2b_row_C8_v4.py --out c8timing4 --block 2,0 --m 0.1 --gM 0.2 --gB 1 --Lam 1 --k 1 > c8timing4/case_2,0.log 2>&1
echo "exit $?" >> c8timing4/case_2,0.log
