#!/bin/bash
# Strand 2d sec. 7(2), stage 1 on C8 (strand2d_prep31_v1.py): the free (3,1)' term of J^prep, the fixed-vacuum cost and the norm-defect
# check at (g_M, g_B) = (0, 0) and (0.2, 1), Lambda 1, m 0.1; CPU, 2 threads.
set -u
D=${QI2_ROOT}/s2d_jet
cd $D || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p prep31_C8
for g in "0 0" "0.2 1"; do
  set -- $g
  nice -n 10 ${QI2_ROOT}/python code/strand2d_prep31_v1.py --out prep31_C8 --lattice C8 --gM $1 --gB $2 --device cpu >> prep31_C8/run.log 2>&1
  echo "C8 ($1, $2) exit $?" >> prep31_C8/status.txt
done
echo DONE >> prep31_C8/status.txt
