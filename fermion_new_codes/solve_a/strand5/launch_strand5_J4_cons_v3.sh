#!/bin/bash
# Strand 5 J4 stage 2, the conservative-transport cross-check (spec sec. 10(j)(5)), for batch 5: strand5_J4_stage2_cons_v1.py on every family
# of J4s2run5's list as soon as that family's stage-2 run has exited (dilute, maximum, most compact members), 4 at a time, 2 threads each;
# J4run4's families (stage 2 in J4s2run5b) after them.  Output s5_star/J4consrun3 (J4run4's in J4consrun3b).
set -u
D=${QI2_ROOT}/s5_star
cd $D || exit 1
mkdir -p J4consrun3 J4consrun3b
date -u '+start %Y-%m-%d %H:%M:%S UTC' > J4consrun3/status.txt
run() {  # family, kernel option, its value, stage-2 folder, output folder
  b=$(basename $1 .json); d=$(basename $(dirname $1))
  until grep -q "^$d/$b.json hessian .* exit" $4/status.txt 2>/dev/null; do sleep 60; done
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand5_J4_stage2_cons_v1.py --family $1 $2 $3 --stage2 $4/stage2_$b.json --out $5 > $5/log_${d}_$b.txt 2>&1
  echo "$d/$b.json exit $? $(date -u +%H:%M:%S)" >> J4consrun3/status.txt
}
export -f run
awk '{print $1, $2, $3, "J4s2run5", "J4consrun3"}' J4s2run5/list.txt | xargs -P 4 -L 1 bash -c 'run $0 $1 $2 $3 $4'
KP=$D/J4kernels_run1
P1024="1:$D/J4kernels_run2/packets_C_m0.01_L1024.json,-1:$KP/packets_C_m0.10049876_L256.json,-1:$KP/packets_C_m0.10049876_L256.json,1:$KP/packets_C_m0.14177447_L256.json"
until grep -q DONE J4s2run5/status.txt; do sleep 60; done
for f in J4run4/lstar_*_pk.json; do echo "$f --packets $P1024 J4s2run5b J4consrun3b"; done | xargs -P 4 -L 1 bash -c 'run $0 $1 $2 $3 $4'
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> J4consrun3/status.txt
echo DONE >> J4consrun3/status.txt
