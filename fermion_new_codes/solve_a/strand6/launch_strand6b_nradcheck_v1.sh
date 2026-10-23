#!/bin/bash
# Strand 6 (b): the radial form's convergence in its node count at the pilot (a Omega = 1/250, L = 1024, nfft 16 L, one block): n_rad = 48 and
# 192 beside the production runs' 96, orders four and six; CPU (12 single-thread workers; the GPUs carry the spacing trend).  Output
# omega/nradcheck.
D=${QI2_ROOT}/s6_validation
cd $D || exit 1
O=omega/nradcheck
mkdir -p $O
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
for nr in 48 192; do
  for o in 4 6; do
    mkdir -p $O/nrad$nr
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand6b_omega_v4.py --out $O/nrad$nr --order $o --aOmega 0.004 --L 1024 --mult 16 --workers 12 --nrad $nr --oneblock >> $O/run.log 2>&1
    echo "nrad $nr order $o exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
  done
done
echo DONE >> $O/status.txt
