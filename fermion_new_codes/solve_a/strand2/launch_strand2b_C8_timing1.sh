#!/bin/bash
# Builder timing of the C8 2b route (strand2b_row_C8.py with strand2b_c8_lib.py): block (1,0) then (2,0), m = 0.1, (0.2, 1), Lambda 1,
# k = 1; solve and lock only (no enclosure or uniqueness).  One thread.
cd ${QI2_ROOT}/s2b_row || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${QI2_ROOT}/python
mkdir -p c8timing1
for blk in 1,0 2,0; do
  /usr/bin/time -v nice -n 10 $PY code/strand2b_row_C8.py --out c8timing1 --block $blk --m 0.1 --gM 0.2 --gB 1 --Lam 1 --k 1 > c8timing1/case_$blk.log 2>&1
  echo "exit $?" >> c8timing1/case_$blk.log
done
