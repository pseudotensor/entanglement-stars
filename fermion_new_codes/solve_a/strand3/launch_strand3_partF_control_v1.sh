#!/bin/bash
# Strand 3 part F, controls: kappa = infinity against packet_terms; dense W_0 at L = 8, 12 (CPU, 16 threads).
cd ${QI2_ROOT}/s3_entropy || exit 1
mkdir -p partF/control
echo "start $(date -u '+%Y-%m-%d %H:%M:%S') UTC" > partF/control/status.txt
PARTF_THREADS=16 nice -n 10 ${QI2_ROOT}/python code/strand3_partF_v1.py --out partF/control --control > partF/control/run.log 2>&1
echo "control exit $? $(date -u '+%H:%M:%S')" >> partF/control/status.txt
