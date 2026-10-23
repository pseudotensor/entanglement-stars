#!/bin/bash
# Task 3 on flow_chart_v3.py: D3 (m = 0 and 0.1; the momentum-resolved norms inside r_s), D4 (the same diagnostics), A2 at r_s = 1024 (R_m = 256,
# 2048 sites; the 3 pi/16 carrier's R_turn = 54 at r_s = 256 sits on the ball) for m = 0, 0.1 (kappa_j = -1), and A2 at r_s = 256 again with the
# budget recorded.  One CPU thread, sequential, gated (at most 15).  Label: prescribed flow (simpler version); PG radial; v declared.  Builder analyst.
F=${QI2_ROOT}/flow
cd $F || exit 1
PY="nice -n 10 ${QI2_ROOT}/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
gate() { until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 15 ]; do sleep 30; done; }
run() { local tag=$1; shift; gate; date -u "+task3 $tag start %H:%M:%S" >> task3/status.txt
  $PY code/flow_chart_v3.py "$@" --output task3/$tag.json > task3/$tag.log 2>&1; echo "task3 $tag exit $? $(date -u +%H:%M:%S)" >> task3/status.txt; }
for m in 0 0.1; do run v3_D3_m${m} radial --launch D3 --rs 32 --Rm 8 --nsites 1024 --mass $m --kappa-j -1; done
run v3_D4_m0 radial --launch D4 --rs 32 --Rm 8 --nsites 256 --mass 0 --kappa-j -1 --tsample 2
for m in 0 0.1; do run v3_A2r256_m${m} radial --launch A2 --rs 256 --Rm 64 --nsites 1024 --mass $m --kappa-j -1; done
for m in 0 0.1; do run v3_A2r1024_m${m} radial --launch A2 --rs 1024 --Rm 256 --nsites 2048 --mass $m --kappa-j -1; done
echo "DONE v3" >> task3/status.txt
