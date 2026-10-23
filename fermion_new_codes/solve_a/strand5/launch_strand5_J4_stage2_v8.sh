#!/bin/bash
# Strand 5 J4 stage 2 (spec sec. 10(m)(5), (6)), batch 7 with strand5_J4_stage2_v8.py: the Hessian in basis (c) (density-weighted modes on all of
# r < r_s) and basis (a) (version 7's, the comparison column) at EVERY member (--hessian all), kappa_max and the EOS series' truncation per
# member, on batch 6's 20 J4run2 families, 8 at a time, 2 threads each (output J4s2run7); then, once batch 6's launcher is DONE (it waits for
# J4run4 and builds the (0.01, 0.1) local_ka0 families at the rebuilt kernel's moment), the four J4run4 families likewise (output J4s2run7b).
set -u
D=${QI2_ROOT}/s5_star
cd $D || exit 1
O=J4s2run7
mkdir -p $O J4s2run7b
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
KP=$D/J4kernels_run1
P1024="1:$D/J4kernels_run2/packets_C_m0.01_L1024.json,-1:$KP/packets_C_m0.10049876_L256.json,-1:$KP/packets_C_m0.10049876_L256.json,1:$KP/packets_C_m0.14177447_L256.json"
LIST=$O/list.txt
sed 's/ selected$/ all/' J4s2run6/list.txt > $LIST
run() {  # family, kernel option, its value, hessian mode
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand5_J4_stage2_v8.py --family $1 $2 $3 --out $O --hessian $4 --basis c,a > $O/log_$(basename $(dirname $1))_$(basename $1 .json).txt 2>&1
  echo "$(basename $(dirname $1))/$(basename $1) hessian $4 exit $? $(date -u +%H:%M:%S)" >> J4s2run7/status.txt
}
export -f run
export O
cat $LIST | xargs -P 8 -L 1 bash -c 'run $0 $1 $2 $3'
echo "J4run2 families done $(date -u +%H:%M:%S)" >> J4s2run7/status.txt
until grep -q DONE J4s2run6/status.txt; do sleep 60; done
O=J4s2run7b
export O
for f in J4run4/lstar_*_pk.json; do echo "$f --packets $P1024 all"; done | xargs -P 4 -L 1 bash -c 'run $0 $1 $2 $3'
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> J4s2run7/status.txt
echo DONE >> J4s2run7/status.txt
