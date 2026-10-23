#!/bin/bash
# Strand 6 (b) on GPU 1 (free at launch; one compute process): strand6b_omega_v4.py (torch), sequential settings:
# 1-2 the pilot a Omega = 1/250 at L = 1024 (nfft 16 L) with the spec's folded quadrature and both blocks, orders 4 and 6;
# 3 L = 1024 nfft 32 L radial, both blocks; 4 L = 2048 nfft 16 L with the spec quadrature, one block; 5 L = 2048 nfft 32 L radial;
# 6 L = 4096 nfft 16 L radial; 7 the spacing trend a Omega = 1/125, 1/500 at L = 1024 (spec, both blocks) and 2048 (radial).
# Output s6_validation/omega/run2.
set -u
D=${QI2_ROOT}/s6_validation
cd $D || exit 1
O=omega/run2
mkdir -p $O
date -u '+start %Y-%m-%d %H:%M:%S UTC (GPU 1)' > $O/status.txt
run() {  # order aOmega L mult extra...
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand6b_omega_v4.py --out $O --order $1 --aOmega $2 --L $3 --mult $4 --gpu 1 "${@:5}" >> $O/run.log 2>&1
  echo "order $1 aOmega $2 L $3 mult $4 ${*:5} exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
}
for o in 4 6; do run $o 0.004 1024 16 --spec; done
for o in 4 6; do run $o 0.004 1024 32; done
for o in 4 6; do run $o 0.004 2048 16 --spec --oneblock; done
for o in 4 6; do run $o 0.004 2048 32 --oneblock; done
for o in 4 6; do run $o 0.004 4096 16 --oneblock; done
for aO in 0.008 0.002; do
  for o in 4 6; do run $o $aO 1024 16 --spec; run $o $aO 2048 16 --oneblock; done
done
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
echo DONE >> $O/status.txt
