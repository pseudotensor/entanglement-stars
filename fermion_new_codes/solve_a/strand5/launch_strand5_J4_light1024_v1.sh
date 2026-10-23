#!/bin/bash
# Strand 5 sec. 10(k): the light packet m a = 0.01 (twist 0) at L = 1024 by the J2b routine (set C), 8 single-thread workers, for the
# (0.01, 0.1) kernel rebuilt with the light packet's L = 1024 line kernel.  Output s5_star/J4kernels_run2.
D=${QI2_ROOT}/s5_star
cd $D || exit 1
mkdir -p J4kernels_run2
date -u '+start %Y-%m-%d %H:%M:%S UTC' > J4kernels_run2/status.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand5_kernel_packets_v1.py --out J4kernels_run2 --set C --L 1024 --workers 8 --masses 0.01 > J4kernels_run2/run.log 2>&1
echo "exit $? $(date -u +%H:%M:%S)" >> J4kernels_run2/status.txt
echo DONE >> J4kernels_run2/status.txt
