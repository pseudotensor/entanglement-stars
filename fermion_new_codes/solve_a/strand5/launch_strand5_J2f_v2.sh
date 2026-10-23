#!/bin/bash
# Strand 5 J2f (spec sec. 10(n)), version 2 (C_lat route 2 at 80 digits, route 1 with the exact asymptotic tail from u = 1e3): strand5_J2f_v2.py (C_lat by two routes; P14's expansion at the physical and window-edge masses beside
# the integrals; J2's equal control rectangles against their moments; the J4run2 kernel families' k_op against the integral).
# Output s5_star/J2f_run2.
set -u
D=${QI2_ROOT}/s5_star
cd $D || exit 1
mkdir -p J2f_run2
date -u '+start %Y-%m-%d %H:%M:%S UTC' > J2f_run2/status.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nice -n 10 ${QI2_ROOT}/python code/strand5_J2f_v2.py --out J2f_run2 --j2 ${QI2_ROOT}/kernel_v3.json \
  --families J4run2/lstar_m0.1_aH1_kernel_g0_0.json,J4run2/lstar_m0.1_aH0.5_kernel_g0_0.json,J4run2/lstar_m0.03_aH0.3_kernel_g0_0.json,J4run2/lstar_m0.01_aH0.1_kernel_g0_0.json \
  > J2f_run2/run.log 2>&1
echo "exit $? $(date -u +%H:%M:%S)" >> J2f_run2/status.txt
echo DONE >> J2f_run2/status.txt
