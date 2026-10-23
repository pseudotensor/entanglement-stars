#!/bin/bash
# Strand 1: run the leaf-by-leaf comparison of the rerun against the stored results (box, run directory).
cd ${QI2_ROOT}/s1_baseline || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nice -n 10 ${QI2_ROOT}/python compare_s1_baseline.py > compare_s1.txt 2> compare_s1.err
echo "compare exit $?" >> compare_s1.txt
