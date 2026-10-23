#!/bin/bash
# Strand 6 (i) revision 3 (JUDGMENT_strand6_validation.md sec. 12's hand-back, item 4): strand6i_kernel_sign_v2.py (the heads and tails at the
# panel edges, the grouped entries by the P form, the direct F_0, the line kernel's heads, the bisection, the even-r scan), 2 threads, started when
# the lane runs at most 14 busy processes.  Builder analyst.
D=${QI2_ROOT}/s6_val
cd $D || exit 1
mkdir -p s6i_run3
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > s6i_run3/status.txt
until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 14 ]; do sleep 30; done
date -u '+start %Y-%m-%d %H:%M:%S UTC' >> s6i_run3/status.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand6i_kernel_sign_v2.py \
  --md s6i_run3/RESULT_s6i_tables_v2.md --out-json s6i_run3/s6i_quadrature_v2.json > s6i_run3/run.log 2>&1
echo "quadrature v2 exit $? $(date -u +%H:%M:%S)" >> s6i_run3/status.txt
echo DONE >> s6i_run3/status.txt
