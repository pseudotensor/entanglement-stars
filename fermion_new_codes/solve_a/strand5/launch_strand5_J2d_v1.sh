#!/bin/bash
# Strand 5 J2d (spec sec. 10(i)): the heavy packets' L = 2048 symbols (n = 1..4) and the light packet's q = 0 constant at L = 2048, 4096,
# 8192, both runs; 16 single-thread workers (CPU).  Output s5_star/J2d_run1.
D=${QI2_ROOT}/s5_star
cd $D || exit 1
O=J2d_run1
mkdir -p $O
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
nice -n 10 ${QI2_ROOT}/python code/strand5_J2d_v1.py --out $O --workers 16 > $O/run.log 2>&1
echo "exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
echo DONE >> $O/status.txt
