#!/bin/bash
# Strand 5 J2e (spec sec. 10(l)): strand5_J2e_v1.py, the Schwinger-representation integrals by two quadrature routes, the per-record
# masses, the small-m fit and the heavy direct sums (16 single-thread workers).  Output s5_star/J2e_run1.
set -u
D=${QI2_ROOT}/s5_star
cd $D || exit 1
mkdir -p J2e_run1
date -u '+start %Y-%m-%d %H:%M:%S UTC' > J2e_run1/status.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nice -n 10 ${QI2_ROOT}/python code/strand5_J2e_v1.py --out J2e_run1 --records J2b_run1,J4kernels_run1,J4kernels_run2 \
  --workers 16 > J2e_run1/run.log 2>&1
echo "exit $? $(date -u +%H:%M:%S)" >> J2e_run1/status.txt
echo DONE >> J2e_run1/status.txt
