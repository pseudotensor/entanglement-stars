#!/bin/bash
# Task 3 A2 at r_s = 1024 with the ball radius R_m = r_s/8 = 128 (flow_chart_v4.py unchanged): at R_m = r_s/4 the 3 pi/16 carrier's R_turn
# (0.21 r_s) lies inside the ball, where the profile is linear and the PG turning law does not apply (BUILDER, labelled; the packets never
# reach R_m).  One CPU thread, gated (at most 15).  Builder analyst.
F=${QI2_ROOT}/flow
cd $F || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for m in 0 0.1; do
  until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 15 ]; do sleep 30; done
  date -u "+task3v4 A2r1024rm8_m$m start %H:%M:%S" >> task3v4/status.txt
  nice -n 10 ${QI2_ROOT}/python code/flow_chart_v4.py radial --launch A2 --rs 1024 --Rm 128 --nsites 2048 --mass $m --kappa-j -1 --output task3v4/A2r1024rm8_m$m.json > task3v4/A2r1024rm8_m$m.log 2>&1
  echo "task3v4 A2r1024rm8_m$m exit $? $(date -u +%H:%M:%S)" >> task3v4/status.txt
done
