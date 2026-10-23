#!/bin/bash
# Strand 6 (b): strand6b_omega_v2.py (folded zone and radial form), 12 single-thread workers, sequential settings:
# the pilot a Omega = 1/250 at L = 1024 and 2048 (nfft 16 L and 32 L), orders 4 and 6; the spacing trend a Omega = 1/125, 1/500 at L = 1024
# and 2048 (nfft 16 L); then L = 4096 (nfft 16 L) at the pilot.  Output s6_validation/omega/run1.
set -u
D=${QI2_ROOT}/s6_validation
cd $D || exit 1
O=omega/run1
mkdir -p $O
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
run() {  # order aOmega L mult
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand6b_omega_v2.py --out $O --order $1 --aOmega $2 --L $3 --mult $4 --workers 12 >> $O/run.log 2>&1
  echo "order $1 aOmega $2 L $3 mult $4 exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
}
for L in 1024 2048; do
  for mult in 16 32; do
    for o in 4 6; do run $o 0.004 $L $mult; done
  done
done
for aO in 0.008 0.002; do
  for L in 1024 2048; do
    for o in 4 6; do run $o $aO $L 16; done
  done
done
for o in 4 6; do run $o 0.004 4096 16; done
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
echo DONE >> $O/status.txt
