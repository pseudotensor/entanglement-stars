#!/bin/bash
# Task 3, flow_chart_v7.py (the lead's flow_chart version 5 of SPEC_task3_prescribed_flow.md sec. 11.4; file v7 because v5 and v6 are task
# 5's files): the validation against flow_chart_v4.py before any new run.  The tests mode (each fix against v4), the symbol (A1), the ten
# A2 configurations of v4's records and D4 re-run with --keep-rejected (every v4 packet evolved, the geometry verdict recorded beside it),
# then compare_v4 on every pair (reproduced quantities with their largest differences; changed ones with old and new values).
# One CPU thread, nice 10, sequential, gated (below 15 busy lane threads).  Label: prescribed flow (simpler version); v declared.
# Builder analyst.
F=${QI2_ROOT}/flow
cd $F || exit 1
PY="nice -n 10 ${QI2_ROOT}/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
O=task3v7
mkdir -p $O
gate() { until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -lt 15 ]; do sleep 60; done; }
run() { local tag=$1; shift; gate; echo "$tag start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $O/status.txt
  $PY code/flow_chart_v7.py "$@" --output $O/$tag.json > $O/$tag.log 2>&1
  echo "$tag exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $O/status.txt; }
run tests_v7 tests
run val_A1 symbol
for m in 0 0.1; do for kj in -1 1; do
  run val_A2_m${m}_kj${kj} radial --launch A2 --rs 64 --Rm 16 --nsites 512 --mass $m --kappa-j $kj --keep-rejected; done; done
for m in 0 0.1; do run val_A2r256_m${m} radial --launch A2 --rs 256 --Rm 64 --nsites 1024 --mass $m --kappa-j -1 --keep-rejected; done
for m in 0 0.1; do run val_A2r1024_m${m} radial --launch A2 --rs 1024 --Rm 256 --nsites 2048 --mass $m --kappa-j -1 --keep-rejected; done
for m in 0 0.1; do run val_A2r1024rm8_m${m} radial --launch A2 --rs 1024 --Rm 128 --nsites 2048 --mass $m --kappa-j -1 --keep-rejected; done
run val_D4_m0 radial --launch D4 --rs 32 --Rm 8 --nsites 256 --mass 0 --kappa-j -1 --tsample 2 --keep-rejected
P="task3v4/A1.json:$O/val_A1.json"
for t in A2_m0_kj-1 A2_m0_kj1 A2_m0.1_kj-1 A2_m0.1_kj1 A2r256_m0 A2r256_m0.1 A2r1024_m0 A2r1024_m0.1 A2r1024rm8_m0 A2r1024rm8_m0.1 D4_m0; do
  P="$P task3v4/$t.json:$O/val_$t.json"; done
run validation_v7_vs_v4 compare_v4 --pairs $P
echo "VALIDATION DONE $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $O/status.txt
