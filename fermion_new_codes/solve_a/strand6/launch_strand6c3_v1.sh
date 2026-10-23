#!/bin/bash
# Strand 6 (c3): strand6c3_collar_v1.py on the open 3^3 cube (the gate's realization and seed, lane's parameters), CPU, 4 threads.
# Output s6_validation/c3run1.
set -u
D=${QI2_ROOT}/s6_validation
cd $D || exit 1
mkdir -p c3run1
date -u '+start %Y-%m-%d %H:%M:%S UTC' > c3run1/status.txt
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 10 ${QI2_ROOT}/python code/strand6c3_collar_v1.py --out c3run1 > c3run1/run.log 2>&1
echo "exit $?" >> c3run1/status.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> c3run1/status.txt
echo DONE >> c3run1/status.txt
