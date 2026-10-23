#!/bin/bash
# Strand 3 part F, the 40-digit control on the 8 x 3 x 3 box (six cases, one single-thread process each).
cd ${QI2_ROOT}/s3_entropy || exit 1
mkdir -p partF/mpcheck
echo "start $(date -u '+%Y-%m-%d %H:%M:%S') UTC" > partF/mpcheck/status.txt
nice -n 10 ${QI2_ROOT}/python code/strand3_partF_mpcheck_v1.py --out partF/mpcheck --workers 6 > partF/mpcheck/run.log 2>&1
echo "mpcheck exit $? $(date -u '+%H:%M:%S')" >> partF/mpcheck/status.txt
