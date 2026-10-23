#!/bin/bash
# Builder timing for 2b driver v8 (library v6): R4 (2,1), first without enclosure and uniqueness, then the same case with both
# (to measure the full per-case cost).
cd ${QI2_ROOT}/s2b_row || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${QI2_ROOT}/python
mkdir -p timing7a timing7b
/usr/bin/time -v nice -n 10 $PY code/strand2b_row_R4_v8.py --out timing7a --block 2,1 --m 0.1 --gM 0.2 --gB 1 --Lam 1 --k 1 --no-enclosure --no-unique > timing7a/case.log 2>&1
echo "exit $?" >> timing7a/case.log
/usr/bin/time -v nice -n 10 $PY code/strand2b_row_R4_v8.py --out timing7b --block 2,1 --m 0.1 --gM 0.2 --gB 1 --Lam 1 --k 1 > timing7b/case.log 2>&1
echo "exit $?" >> timing7b/case.log
