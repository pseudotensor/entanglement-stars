#!/bin/bash
# Strand 5 J3 (the lattice EOS and the matter rows), run 2 with version 3: strand5_eos_v3.py, m b in {1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1}, theta_F = k_F b (axis) in
# 20 values up to 1.55, quadrature N2 = 2000 (coarse 1000), N3 = 64 (coarse 48); CPU, 2 threads.  Output s5_star/J3run2.
set -u
D=${QI2_ROOT}/s5_star
cd $D || exit 1
mkdir -p J3run2
date -u '+start %Y-%m-%d %H:%M:%S UTC' > J3run2/status.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand5_eos_v3.py --out J3run2 --N2 2000 --N3 64 > J3run2/run.log 2>&1
echo "exit $?" >> J3run2/status.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> J3run2/status.txt
echo DONE >> J3run2/status.txt
