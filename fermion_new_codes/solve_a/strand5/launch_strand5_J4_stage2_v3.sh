#!/bin/bash
# Strand 5 J4 stage 2 (spec sec. 10(h), (j)), batch 3 with strand5_J4_stage2_v2.py: the 24 primary families with the Hessian on the selected
# members (J4run3's four packet-built (0.01, 0.1) families; J4run2's (0.1, 1), (0.1, 0.5), (0.03, 0.3) families and its (0.01, 0.1)
# local_klat pair) and J4run2's four (0.01, 0.1) kernel/local_ka0 families (the L = 256 table kernel, T6's comparison) with the lock
# pieces only; the slowest first; 12 families at a time, 2 threads each.  Output s5_star/J4s2run3.
set -u
D=${QI2_ROOT}/s5_star
cd $D || exit 1
O=J4s2run3
mkdir -p $O
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
K2C=${QI2_ROOT}/kernel_L256.json
KJ4=$D/J2b_run1/kernel_J4_L256.json
KP=$D/J4kernels_run1
P="1:$KP/packets_C_m0.01_L512.json,-1:$KP/packets_C_m0.10049876_L256.json,-1:$KP/packets_C_m0.10049876_L256.json,1:$KP/packets_C_m0.14177447_L256.json"
LIST=$O/list.txt
: > $LIST
for f in J4run3/lstar_*_pk.json; do echo "$f --packets $P selected" >> $LIST; done
for f in J4run2/lstar_m0.01_aH0.1_local_klat_*.json; do case $f in *_profiles*) continue;; esac; echo "$f --kernel $KJ4 selected" >> $LIST; done
for f in J4run2/lstar_m0.03_aH0.3_*.json; do case $f in *_profiles*) continue;; esac; echo "$f --kernel $KJ4 selected" >> $LIST; done
for f in J4run2/lstar_m0.1_aH*.json; do case $f in *_profiles*) continue;; esac; echo "$f --kernel $K2C selected" >> $LIST; done
for f in J4run2/lstar_m0.01_aH0.1_kernel_*.json J4run2/lstar_m0.01_aH0.1_local_ka0_*.json; do case $f in *_profiles*) continue;; esac; echo "$f --kernel $KJ4 none" >> $LIST; done
run() {  # family, kernel option, its value, hessian mode
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand5_J4_stage2_v2.py --family $1 $2 $3 --out $O --hessian $4 > $O/log_$(basename $1 .json).txt 2>&1
  echo "$(basename $1) hessian $4 exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
}
export -f run
export O
cat $LIST | xargs -P 12 -L 1 bash -c 'run $0 $1 $2 $3'
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
echo DONE >> $O/status.txt
