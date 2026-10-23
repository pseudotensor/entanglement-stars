#!/bin/bash
# Task 3 part (c) (SPEC sec. 6; C6, C7): the cut tables at the forming surface on flow_chart_v4.py's cut mode (planar, y = 0), the record's
# forming-surface parameters (RESULT_T128.md Item 2: delta 0.4, width delta/kappa, ramp 40, dt 0.5, nx 128) at kappa a = 1/64 (nk 12, then 24),
# then 1/32 and 1/16 (nk 12, 24); holds 0, 12, 24, 36 (units 1/kappa); both rectangles at aM = 0.3, 0.1, 0.03 and the massless sea.
# 10 workers, started only after strand S's sphere (S3v3) has finished (version 2 of the launcher: version 1's gate raced with S3's at 04:50 UTC).  Label: prescribed flow (simpler version); planar tanh; v declared.
# Builder analyst.
F=${QI2_ROOT}/flow
cd $F || exit 1
mkdir -p task3C
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for spec in 64:12 64:24 32:12 32:24 16:12 16:24; do
  ik=${spec%%:*}; nk=${spec##*:}
  until grep -q DONE ${QI2_ROOT}/status.txt && [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 6 ]; do sleep 30; done
  date -u "+task3C k$ik nk$nk start %H:%M:%S" >> task3C/status.txt
  nice -n 10 ${QI2_ROOT}/python code/flow_chart_v4.py cut --inv-kappa $ik --delta 0.4 --nsites 128 --ramp-time 40 --dt 0.5 \
    --nk $nk --aM 0.3 0.1 0.03 --hold-over-kappa 0 12 24 36 --workers 10 --output task3C/cut_k${ik}_nk${nk}.json > task3C/cut_k${ik}_nk${nk}.log 2>&1
  echo "task3C k$ik nk$nk exit $? $(date -u +%H:%M:%S)" >> task3C/status.txt
done
echo DONE >> task3C/status.txt
