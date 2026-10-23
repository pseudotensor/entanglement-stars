#!/bin/bash
# Task 4 sec. 1 (SPEC_task4_injection.md; task4_static_v2.py): ingoing packets at omega r_h = 0.5, 1, 2, 4 on the static chart's locked background
# L (E16, h = 1) frozen at N_s = 0.1 and 0.01, and the flat chart (A_0 = 0); j = 1/2 (both kappa signs) and j = 3/2 (kappa = -2); m = 0, and
# m = 0.05 at N_s = 0.01 as the massive control.  One CPU thread, sequential, gated (at most 15).  Label: lattice at rest, v = 0, locked
# background L.  Builder analyst.
F=${QI2_ROOT}/flow
cd $F || exit 1
mkdir -p task4
export STRAND7_CODE=${QI2_ROOT}/code STRANDS_CODE=${QI2_ROOT}/code OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run() { local tag=$1; shift; until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 15 ]; do sleep 30; done
  date -u "+task4 $tag start %H:%M:%S" >> task4/status.txt
  nice -n 10 ${QI2_ROOT}/python code/task4_static_v2.py "$@" --out-json task4/$tag.json > task4/$tag.log 2>&1
  echo "task4 $tag exit $? $(date -u +%H:%M:%S)" >> task4/status.txt; }
for Ns in 0.1 0.01 1; do
  run Ns${Ns}_j0.5_m1 --E0 16 --Ns $Ns --j 0.5 --sign -1
  run Ns${Ns}_j0.5_p1 --E0 16 --Ns $Ns --j 0.5 --sign 1
  run Ns${Ns}_j1.5_m1 --E0 16 --Ns $Ns --j 1.5 --sign -1
done
run Ns0.01_j0.5_m1_mass0.05 --E0 16 --Ns 0.01 --j 0.5 --sign -1 --mass 0.05
echo DONE >> task4/status.txt
