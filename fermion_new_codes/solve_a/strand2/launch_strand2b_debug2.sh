#!/bin/bash
# Builder diagnostic 2: the same R4 (2,1) case with library v2 (fixed projector rank): Jacobian comparison, then the timing case.
cd ${QI2_ROOT}/s2b_row || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${QI2_ROOT}/python
mkdir -p debug2 timing2
nice -n 10 $PY code/strand2b_debug_jac_v2.py --case timing1/case_R4_b21_m0.1_gM0.2_gB1_L1_k1.json --out debug2 > debug2/debug.log 2>&1
echo "exit $?" >> debug2/debug.log
/usr/bin/time -v nice -n 10 $PY code/strand2b_row_R4_v2.py --out timing2 --block 2,1 --m 0.1 --gM 0.2 --gB 1 --Lam 1 --k 1 --no-enclosure --no-unique > timing2/case.log 2>&1
echo "exit $?" >> timing2/case.log
