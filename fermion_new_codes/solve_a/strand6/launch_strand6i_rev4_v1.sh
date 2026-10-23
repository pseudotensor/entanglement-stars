#!/bin/bash
# Strand 6 (i) revision 4 (JUDGMENT_strand6_validation.md sec. 13, the hand-back's item 2 (a), (b)): strand6i_rev4_v1.py, one CPU thread
# (the GPUs carry item 7's runs), minutes; started when the lane runs at most 15 busy processes.  Builder analyst.
D=${QI2_ROOT}/s6_val
cd $D || exit 1
mkdir -p s6i_run4
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > s6i_run4/status.txt
until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 15 ]; do sleep 20; done
date -u '+start %Y-%m-%d %H:%M:%S UTC' >> s6i_run4/status.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand6i_rev4_v1.py \
  --rev3-json s6i_run3/s6i_quadrature_v2.json --L 64 --md s6i_run4/RESULT_s6i_rev4_tables.md --out-json s6i_run4/s6i_rev4.json > s6i_run4/run.log 2>&1
echo "rev4 exit $? $(date -u +%H:%M:%S)" >> s6i_run4/status.txt
echo DONE >> s6i_run4/status.txt
