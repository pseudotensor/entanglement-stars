#!/bin/bash
# Strand 3 addendum part D (strand3_addCD_v1.py D) and the part C job file (C-masses); one thread, seconds.
set -u
D=${QI2_ROOT}/s3_entropy
cd $D || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p addendum1
nice -n 10 ${QI2_ROOT}/python code/strand3_addCD_v1.py D --out addendum1 > addendum1/addD.log 2>&1; echo "D exit $?" >> addendum1/status.txt
nice -n 10 ${QI2_ROOT}/python code/strand3_addCD_v1.py C-masses --out addendum1/C --masses run1/masses.json > addendum1/addC_masses.log 2>&1; echo "C-masses exit $?" >> addendum1/status.txt
