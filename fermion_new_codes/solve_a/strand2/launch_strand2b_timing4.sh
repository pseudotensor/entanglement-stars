#!/bin/bash
# Builder timing case for 2b library v5 (Newton best iterate and noise-floor stop): R4 (2,1), no enclosure or uniqueness.
cd ${QI2_ROOT}/s2b_row || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${QI2_ROOT}/python
mkdir -p timing4
/usr/bin/time -v nice -n 10 $PY code/strand2b_row_R4_v5.py --out timing4 --block 2,1 --m 0.1 --gM 0.2 --gB 1 --Lam 1 --k 1 --no-enclosure --no-unique > timing4/case.log 2>&1
echo "exit $?" >> timing4/case.log
