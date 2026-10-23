#!/bin/bash
# Builder timing case for 2b library v6 (path tolerance 1e-9, final 1e-11): R4 (2,1), no enclosure or uniqueness.
cd ${QI2_ROOT}/s2b_row || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${QI2_ROOT}/python
mkdir -p timing5
/usr/bin/time -v nice -n 10 $PY code/strand2b_row_R4_v6.py --out timing5 --block 2,1 --m 0.1 --gM 0.2 --gB 1 --Lam 1 --k 1 --no-enclosure --no-unique > timing5/case.log 2>&1
echo "exit $?" >> timing5/case.log
