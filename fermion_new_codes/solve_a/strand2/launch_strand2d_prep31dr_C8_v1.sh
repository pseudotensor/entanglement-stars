#!/bin/bash
# Strand 2d sec. 7(2), stage 2 on C8 (strand2d_prep31_dressed_v1.py): the (3,1)' term and the fixed-vacuum cost with the completion through
# the q-block route, at (0, 0) (convention check: equals stage 1) and (0.2, 1), Lambda 1, m 0.1; CPU, 2 threads (the lane's budget).
set -u
D=${QI2_ROOT}/s2d_jet
cd $D || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p prep31dr_C8
for g in "0 0" "0.2 1"; do
  set -- $g
  /usr/bin/time -v nice -n 10 ${QI2_ROOT}/python code/strand2d_prep31_dressed_v1.py --out prep31dr_C8 --gM $1 --gB $2 >> prep31dr_C8/run.log 2>&1
  echo "C8 ($1, $2) exit $?" >> prep31dr_C8/status.txt
done
echo DONE >> prep31dr_C8/status.txt
