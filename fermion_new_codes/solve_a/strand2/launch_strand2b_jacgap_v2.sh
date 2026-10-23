#!/bin/bash
# The 2b jacobian-gap check, version 2 (strand2b_c8_jacgap_v2.py: the specified pair term with the norm-defect companion and the undressed
# variants beside it), same record, output s2b_row/jacgap2.  Two threads.
set -u
B=${QI2_ROOT}/solve_a
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd $B/s2b_row && mkdir -p jacgap2 && date -u '+start %Y-%m-%d %H:%M:%S UTC' > jacgap2/status.txt && nice -n 10 ${QI2_ROOT}/python code/strand2b_c8_jacgap_v2.py --out jacgap2 --record $(ls c8timing5/case_*.json | head -1) > jacgap2/jacgap.log 2>&1
echo "exit $?" >> jacgap2/jacgap.log
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> jacgap2/status.txt
