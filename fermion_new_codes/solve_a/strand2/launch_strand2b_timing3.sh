#!/bin/bash
# Builder checks for 2b library v4: equivalence with v3, then the R4 (2,1) timing case (no enclosure or uniqueness).
cd ${QI2_ROOT}/s2b_row || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${QI2_ROOT}/python
mkdir -p timing3
nice -n 10 $PY code/strand2b_v4_check.py > timing3/v4_check.log 2>&1
/usr/bin/time -v nice -n 10 $PY code/strand2b_row_R4_v4.py --out timing3 --block 2,1 --m 0.1 --gM 0.2 --gB 1 --Lam 1 --k 1 --no-enclosure --no-unique > timing3/case.log 2>&1
echo "exit $?" >> timing3/case.log
