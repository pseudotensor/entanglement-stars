#!/bin/bash
# The 2b sec. 13 verification (strand2b_c8_jacgap_v1.py on the c8timing5 record), queued behind the first small-job chain (c7 DONE) in
# the same two-thread slot.
set -u
B=${QI2_ROOT}/solve_a
until grep -q DONE $B/s2d_jet/c7/status.txt 2>/dev/null; do sleep 30; done
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd $B/s2b_row && mkdir -p jacgap1 && nice -n 10 ${QI2_ROOT}/python code/strand2b_c8_jacgap_v1.py --out jacgap1 --record $(ls c8timing5/case_*.json | head -1) > jacgap1/jacgap.log 2>&1
echo "exit $?" >> jacgap1/jacgap.log
