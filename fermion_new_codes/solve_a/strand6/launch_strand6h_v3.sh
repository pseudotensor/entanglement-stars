#!/bin/bash
# Version 3: strand6h_vertex_v3.py (part E's pi valley at the laboratory momenta; JUDGMENT_strand6_validation.md sec. 13, item 1).
# Strand 6 (h) revision 2 (JUDGMENT_strand6_validation.md sec. 12's hand-back, items 1 and 2): strand6h_vertex_v2.py (H5: the two-law
# backscattering ratio with the computed comparator; H1: the transposed difference; H6: strand 2a's part E), 2 threads, started when the lane runs
# at most 14 busy processes.  Builder analyst.
D=${QI2_ROOT}/s6_val
cd $D || exit 1
mkdir -p s6h_run3
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > s6h_run3/status.txt
until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 14 ]; do sleep 30; done
date -u '+start %Y-%m-%d %H:%M:%S UTC' >> s6h_run3/status.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand6h_vertex_v3.py \
  --partE-dir ${QI2_ROOT}/rtchecks1 --md s6h_run3/RESULT_s6h_tables_v3.md --out-json s6h_run3/s6h_v3.json > s6h_run3/run.log 2>&1
echo "vertex v3 exit $? $(date -u +%H:%M:%S)" >> s6h_run3/status.txt
echo DONE >> s6h_run3/status.txt
