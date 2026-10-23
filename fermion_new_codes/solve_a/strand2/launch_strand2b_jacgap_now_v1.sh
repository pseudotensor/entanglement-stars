#!/bin/bash
# The 2b jacobian-gap check (strand2b_c8_jacgap_v1.py) now, in two spare threads; replaces the waiter launch_small2_after_c7_v1.sh, which
# would have started it only after the small chain's C7 (behind the 2c kernel).  Same command and output (s2b_row/jacgap1).
set -u
B=${QI2_ROOT}/solve_a
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd $B/s2b_row && mkdir -p jacgap1 && date -u '+start %Y-%m-%d %H:%M:%S UTC (launch_strand2b_jacgap_now_v1.sh)' > jacgap1/status.txt && nice -n 10 ${QI2_ROOT}/python code/strand2b_c8_jacgap_v1.py --out jacgap1 --record $(ls c8timing5/case_*.json | head -1) > jacgap1/jacgap.log 2>&1
echo "exit $?" >> jacgap1/jacgap.log
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> jacgap1/status.txt
