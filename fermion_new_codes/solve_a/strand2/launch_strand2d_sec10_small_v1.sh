#!/bin/bash
# Strand 2d sec. 10, the small computations (strand2d_sec10_small_v1.py): shells, heavy, lowest31 (one thread each, seconds), then
# c8dressed31 (two threads, the dense 8960-dimensional (3,1) block on C8 at both couplings).  Output s2d_jet/sec10.
set -u
B=${QI2_ROOT}/s2d_jet
cd $B || exit 1
mkdir -p sec10
date -u '+start %Y-%m-%d %H:%M:%S UTC' > sec10/status.txt
for P in shells heavy lowest31; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand2d_sec10_small_v1.py --out sec10 --part $P --jetdir B4run1 > sec10/$P.log 2>&1
  echo "$P exit $?" >> sec10/status.txt
done
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 /usr/bin/time -v ${QI2_ROOT}/python code/strand2d_sec10_small_v1.py --out sec10 --part c8dressed31 > sec10/c8dressed31.log 2>&1
echo "c8dressed31 exit $?" >> sec10/status.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> sec10/status.txt
echo DONE >> sec10/status.txt
