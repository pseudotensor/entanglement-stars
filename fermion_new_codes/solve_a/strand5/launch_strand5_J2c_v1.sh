#!/bin/bash
# Strand 5 J2c (spec sec. 10(f)): the first-variation form against the 2c second-order kernel (control, L = 64, 128), then the
# eta-summed physical-row symbol at L = 1024, 2048, n = 1..4, both runs (massless light with twist 0.371; m_light 1e-4 with twist 0),
# 8 single-thread workers (CPU).  Output s5_star/J2c_run1.
set -u
D=${QI2_ROOT}/s5_star
cd $D || exit 1
O=J2c_run1
mkdir -p $O
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
nice -n 10 ${QI2_ROOT}/python code/strand5_J2c_v1.py --out $O --control > $O/control.log 2>&1
echo "control exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
nice -n 10 ${QI2_ROOT}/python code/strand5_J2c_v1.py --out $O --workers 8 > $O/run.log 2>&1
echo "main exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
echo DONE >> $O/status.txt
