#!/bin/bash
# Strand 5 J4 sec. 10(o): strand5_J4_symgauss_v1.py on the two kernel families of J4run4 (packet-built, light packet at L = 1024):
# the stencil symmetrized at the build, each member re-solved from its stored solution, the Gauss check before and after; one
# process, 2 threads, families in sequence.  Output s5_star/J4sym2.
set -u
D=${QI2_ROOT}/s5_star
cd $D || exit 1
O=J4sym2
mkdir -p $O
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
K2C=${QI2_ROOT}/kernel_L256.json
KJ4=$D/J2b_run1/kernel_J4_L256.json
K=$D/J4kernels_run1
P1024="1:$D/J4kernels_run2/packets_C_m0.01_L1024.json,-1:$K/packets_C_m0.10049876_L256.json,-1:$K/packets_C_m0.10049876_L256.json,1:$K/packets_C_m0.14177447_L256.json"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for f in J4run4/lstar_*_kernel_*_pk.json; do
  case $f in *_profiles*) continue;; esac
  opt="--packets $P1024"
  nice -n 10 ${QI2_ROOT}/python code/strand5_J4_symgauss_v1.py --family $f $opt --out $O > $O/log_$(basename $(dirname $f))_$(basename $f .json).txt 2>&1
  echo "$f exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
done
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
echo DONE >> $O/status.txt
