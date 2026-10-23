#!/bin/bash
# Task 3 A2 at r_s = 256 (R_m = 64, 1024 sites; flow_chart_v2.py unchanged): at the spec's r_s = 64 the outgoing carriers' packets (sigma_x 16
# to 32) overlap the centre at their launch radii (the 4 sigma margin fails), so the outgoing scan runs where R_turn >> sigma_x (BUILDER,
# labelled); the ingoing k0 = -3 pi/8 set stays on the r_s = 64 records.  One CPU thread, sequential, gated (at most 15).  Builder analyst.
F=${QI2_ROOT}/flow
cd $F || exit 1
PY="nice -n 10 ${QI2_ROOT}/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
gate() { until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 15 ]; do sleep 30; done; }
for m in 0 0.1; do for kj in -1 1; do
  tag=A2big_m${m}_kj${kj}; gate
  date -u "+task3 $tag start %H:%M:%S" >> task3/status.txt
  $PY code/flow_chart_v2.py radial --launch A2 --rs 256 --Rm 64 --nsites 1024 --mass $m --kappa-j $kj --output task3/$tag.json > task3/$tag.log 2>&1
  echo "task3 $tag exit $? $(date -u +%H:%M:%S)" >> task3/status.txt
done; done
echo "DONE A2big" >> task3/status.txt
