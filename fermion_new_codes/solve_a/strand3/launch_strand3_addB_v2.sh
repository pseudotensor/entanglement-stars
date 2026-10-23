#!/bin/bash
# Version 2 (builder analyst): as version 1, but also waits for the lane's slot D2 queue (s2a_gate/rtchecks1 DONE), so its 12 workers
# never run beside slots A, C and D at once (16 threads).
# Strand 3 addendum part B (strand3_addAB_v1.py --part B: the 8-valley family M' = q + kappa'^-1 sum sin^2 k_i), queued behind parts C and E
# (CE DONE), 12 single-thread workers.
set -u
D=${QI2_ROOT}/s3_entropy
cd $D || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
date -u '+queued B (v2: also after slot D2) %Y-%m-%d %H:%M:%S UTC' >> addendum1/status.txt
until grep -q "CE DONE" addendum1/status.txt 2>/dev/null && grep -q DONE ${QI2_ROOT}/status.txt 2>/dev/null; do sleep 60; done
date -u '+start B %Y-%m-%d %H:%M:%S UTC' >> addendum1/status.txt
nice -n 10 ${QI2_ROOT}/python code/strand3_addAB_v1.py --out addendum1 --part B --workers 12 > addendum1/addB.log 2>&1
echo "B exit $?" >> addendum1/status.txt
date -u '+end B %Y-%m-%d %H:%M:%S UTC' >> addendum1/status.txt
echo "B DONE" >> addendum1/status.txt
