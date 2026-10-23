#!/bin/bash
# Strand 3 part F, main run (after the controls): the three settings, kappa in {0.003, 0.0023, 0.002, inf}, both placements, every packet,
# n = 1..64 at L = 128; 12 single-thread workers.  Output s3_entropy/partF/run1.
cd ${QI2_ROOT}/s3_entropy || exit 1
mkdir -p partF/run1
echo "start $(date -u '+%Y-%m-%d %H:%M:%S') UTC" > partF/run1/status.txt
nice -n 10 ${QI2_ROOT}/python code/strand3_partF_v1.py --out partF/run1 --L 128 --workers 12 > partF/run1/run.log 2>&1
echo "main exit $? $(date -u '+%H:%M:%S')" >> partF/run1/status.txt
