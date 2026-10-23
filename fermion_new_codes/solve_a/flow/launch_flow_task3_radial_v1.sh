#!/bin/bash
# Task 3 (SPEC_task3_prescribed_flow.md secs. 4 and 7): A1 (the symbol), A2 and A3 (r_s = 64, R_m = 16, 512 sites), D1 and D2 (r_s = 32,
# R_m = 8, 256 sites; m = 0.5, 0.1), D3 (1024 sites; m = 0, 0.1), D4 (m = 0) on flow_chart_v2.py's radial channel, one CPU thread, sequential,
# each start gated on the lane's busy processes (at most 15).  Label: prescribed flow (simpler version); PG radial; v declared.  Builder analyst.
F=${QI2_ROOT}/flow
cd $F || exit 1
PY="nice -n 10 ${QI2_ROOT}/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p task3
gate() { until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 15 ]; do sleep 30; done; }
run() {  # run TAG args...
  local tag=$1; shift
  gate
  date -u "+task3 $tag start %H:%M:%S" >> task3/status.txt
  $PY code/flow_chart_v2.py "$@" --output task3/$tag.json > task3/$tag.log 2>&1
  echo "task3 $tag exit $? $(date -u +%H:%M:%S)" >> task3/status.txt
}
run A1 symbol
for m in 0 0.1; do for kj in -1 1; do run A2_m${m}_kj${kj} radial --launch A2 --rs 64 --Rm 16 --nsites 512 --mass $m --kappa-j $kj; done; done
for m in 0 0.1; do run A3_m${m} radial --launch A3 --rs 64 --Rm 16 --nsites 512 --mass $m --kappa-j -1; done
for m in 0.5 0.1; do run D1_m${m} radial --launch D1 --rs 32 --Rm 8 --nsites 256 --mass $m --kappa-j -1 --tsample 2; run D2_m${m} radial --launch D2 --rs 32 --Rm 8 --nsites 256 --mass $m --kappa-j -1 --tsample 2; done
for m in 0 0.1; do run D3_m${m} radial --launch D3 --rs 32 --Rm 8 --nsites 1024 --mass $m --kappa-j -1; done
run D4_m0 radial --launch D4 --rs 32 --Rm 8 --nsites 256 --mass 0 --kappa-j -1 --tsample 2
echo DONE >> task3/status.txt
