#!/bin/bash
# Strand 5 J4 sec. 10(m)(3): strand5_J4_exterior_v1.py on every stage-1 family of J4run2 (the 2c table kernel for m a = 0.1, J2b's
# kernel_J4_L256.json for 0.03 and 0.01) and J4run3 (packet-built, light at L = 512), the three most compact members each; one process,
# 2 threads, families in sequence.  Output s5_star/J4ext1.  (J4run4's families follow when that run is done, by the same script.)
set -u
D=${QI2_ROOT}/s5_star
cd $D || exit 1
O=J4ext1
mkdir -p $O
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
K2C=${QI2_ROOT}/kernel_L256.json
KJ4=$D/J2b_run1/kernel_J4_L256.json
K=$D/J4kernels_run1
P512="1:$K/packets_C_m0.01_L512.json,-1:$K/packets_C_m0.10049876_L256.json,-1:$K/packets_C_m0.10049876_L256.json,1:$K/packets_C_m0.14177447_L256.json"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for f in J4run2/lstar_*.json J4run3/lstar_*_pk.json; do
  case $f in *_profiles*) continue;; esac
  case $f in J4run3/*) opt="--packets $P512";; *m0.1_aH*) opt="--kernel $K2C";; *) opt="--kernel $KJ4";; esac
  nice -n 10 ${QI2_ROOT}/python code/strand5_J4_exterior_v1.py --family $f $opt --out $O > $O/log_$(basename $(dirname $f))_$(basename $f .json).txt 2>&1
  echo "$f exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
done
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
echo DONE >> $O/status.txt
