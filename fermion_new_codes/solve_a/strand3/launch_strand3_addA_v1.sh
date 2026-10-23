#!/bin/bash
# Strand 3 addendum part A (strand3_addAB_v1.py --part A): the Wilson-type family's eta_100 at kappa in {0.01, 0.005, 0.003, 0.002}, q = 0,
# with the nested transverse quadrature, L in {64, 128, 256}, nfft in {16 L, 32 L}; CPU, 12 single-thread workers.
set -u
D=${QI2_ROOT}/s3_entropy
cd $D || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p addendum1
date -u '+start A %Y-%m-%d %H:%M:%S UTC' >> addendum1/status.txt
nice -n 10 ${QI2_ROOT}/python code/strand3_addAB_v1.py --out addendum1 --part A --workers 12 > addendum1/addA.log 2>&1
echo "A exit $?" >> addendum1/status.txt
date -u '+end A %Y-%m-%d %H:%M:%S UTC' >> addendum1/status.txt
