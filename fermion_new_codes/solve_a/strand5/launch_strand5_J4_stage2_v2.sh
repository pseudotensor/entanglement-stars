#!/bin/bash
# Strand 5 J4 stage 2 (spec sec. 10(h)), version 2 (pointwise-bounded Hessian steps): strand5_J4_stage2_v2.py on every stage-1 family (J4run2's 24, then J4run3's 4 packet-built
# (0.01, 0.1) families once J4run3 is DONE), the Hessian on the selected members; 8 families at a time, 2 threads each.  Output s5_star/J4s2run2.
set -u
D=${QI2_ROOT}/s5_star
cd $D || exit 1
O=J4s2run2
mkdir -p $O
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
K2C=${QI2_ROOT}/kernel_L256.json
KJ4=$D/J2b_run1/kernel_J4_L256.json
KP=$D/J4kernels_run1
P="1:$KP/packets_C_m0.01_L512.json,-1:$KP/packets_C_m0.10049876_L256.json,-1:$KP/packets_C_m0.10049876_L256.json,1:$KP/packets_C_m0.14177447_L256.json"
LIST=$O/list.txt
: > $LIST
for f in J4run2/lstar_*.json; do
  case $f in *_profiles*) continue;; esac
  case $f in *m0.1_aH*) k="--kernel $K2C";; *) k="--kernel $KJ4";; esac
  echo "$f $k" >> $LIST
done
run() {  # family, kernel option pair
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand5_J4_stage2_v2.py --family $1 $2 $3 --out $O --hessian selected > $O/log_$(basename $1 .json).txt 2>&1
  echo "$(basename $1) exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
}
export -f run
export O
cat $LIST | xargs -P 8 -L 1 bash -c 'run $0 $1 $2'
until grep -q DONE J4run3/status.txt; do sleep 60; done
for f in J4run3/lstar_*_pk.json; do echo "$f"; done | xargs -P 4 -I{} bash -c "run {} --packets $P"
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
echo DONE >> $O/status.txt
