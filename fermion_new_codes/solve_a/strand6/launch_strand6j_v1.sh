#!/bin/bash
# Strand 6 (j) on ring4 (strand6j_constraint_var_v1.py over every pass2v13 case record: the spec's cases and the rest of the batch), one thread,
# started when the lane runs at most 15 busy processes (lane_threads_v1.sh).  Builder analyst.
D=${QI2_ROOT}/j
cd $D || exit 1
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > run1/status.txt
until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 15 ]; do sleep 30; done
date -u '+start %H:%M:%S UTC' >> run1/status.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nice -n 10 ${QI2_ROOT}/python code/strand6/strand6j_constraint_var_v1.py --cases "cases/case_R4_*.json" --out-json run1/j_ring4.json --md run1/RESULT_j_ring4_tables_v1.md > run1/run.log 2>&1
echo "exit $? $(date -u +%H:%M:%S)" >> run1/status.txt
echo DONE >> run1/status.txt
