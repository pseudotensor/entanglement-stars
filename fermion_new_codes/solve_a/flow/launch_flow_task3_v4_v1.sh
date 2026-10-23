#!/bin/bash
# Task 3 on flow_chart_v4.py (the Hermitian completion and the radial spinors corrected; every radial record of versions 2 and 3 superseded):
# the checks, A2 at r_s = 64 (512 sites), 256 (1024) and 1024 (2048) for m = 0, 0.1 (kappa_j = -1; at r_s = 64 also +1), A3, D1, D2, D3, D4.
# One CPU thread, sequential, gated (at most 15).  Label: prescribed flow (simpler version); PG radial; v declared.  Builder analyst.
F=${QI2_ROOT}/flow
cd $F || exit 1
PY="nice -n 10 ${QI2_ROOT}/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p task3v4
gate() { until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 15 ]; do sleep 30; done; }
run() { local tag=$1; shift; gate; date -u "+task3v4 $tag start %H:%M:%S" >> task3v4/status.txt
  $PY code/flow_chart_v4.py "$@" --output task3v4/$tag.json > task3v4/$tag.log 2>&1; echo "task3v4 $tag exit $? $(date -u +%H:%M:%S)" >> task3v4/status.txt; }
run checks_v4 checks
run A1 symbol
for m in 0 0.1; do for kj in -1 1; do run A2_m${m}_kj${kj} radial --launch A2 --rs 64 --Rm 16 --nsites 512 --mass $m --kappa-j $kj; done; done
for m in 0 0.1; do run A2r256_m${m} radial --launch A2 --rs 256 --Rm 64 --nsites 1024 --mass $m --kappa-j -1; done
for m in 0 0.1; do run A2r1024_m${m} radial --launch A2 --rs 1024 --Rm 256 --nsites 2048 --mass $m --kappa-j -1; done
for m in 0 0.1; do run A3_m${m} radial --launch A3 --rs 64 --Rm 16 --nsites 512 --mass $m --kappa-j -1; done
for m in 0.5 0.1; do run D1_m${m} radial --launch D1 --rs 32 --Rm 8 --nsites 256 --mass $m --kappa-j -1 --tsample 2; run D2_m${m} radial --launch D2 --rs 32 --Rm 8 --nsites 256 --mass $m --kappa-j -1 --tsample 2; done
for m in 0 0.1; do run D3_m${m} radial --launch D3 --rs 32 --Rm 8 --nsites 1024 --mass $m --kappa-j -1; done
run D4_m0 radial --launch D4 --rs 32 --Rm 8 --nsites 256 --mass 0 --kappa-j -1 --tsample 2
echo DONE >> task3v4/status.txt
