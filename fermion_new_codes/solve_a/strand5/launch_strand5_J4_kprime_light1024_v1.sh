#!/bin/bash
# Strand 5 J4 stage 2, P14 eq. 7 at (0.01, 0.1): the light packet's k(m) at m = 0.01 (1 -+ 1e-3) at L = 1024 (the L = 256 ring's wrap enters
# P14's small k_a'(1) = -4 k_a(0) + 2 sum eta m k'(m) at the size of the result), the J2b routine (set C, twist 0), 8 single-thread workers.
# Output s5_star/J4kprime_run2.
D=${QI2_ROOT}/s5_star
cd $D || exit 1
O=J4kprime_run2
mkdir -p $O
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand5_kernel_packets_v1.py --out $O --set C --L 1024 --workers 8 --masses 0.0099900000000000006,0.01001 > $O/run.log 2>&1
echo "exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
echo DONE >> $O/status.txt
