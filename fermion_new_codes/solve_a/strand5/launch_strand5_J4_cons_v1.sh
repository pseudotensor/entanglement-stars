#!/bin/bash
# Strand 5 J4 stage 2, the conservative-transport cross-check (spec sec. 10(j)(5)): strand5_J4_stage2_cons_v1.py on every family of
# J4s2run3's list as soon as that family's stage-2 run has exited (dilute, maximum, most compact members), 4 at a time, 2 threads each.
# Output s5_star/J4consrun1.
set -u
D=${QI2_ROOT}/s5_star
cd $D || exit 1
O=J4consrun1
mkdir -p $O
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
run() {  # family, kernel option, its value
  b=$(basename $1 .json)
  until grep -q "^$b.json hessian .* exit" J4s2run3/status.txt; do sleep 60; done
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand5_J4_stage2_cons_v1.py --family $1 $2 $3 --stage2 J4s2run3/stage2_$b.json --out $O > $O/log_$b.txt 2>&1
  echo "$b.json exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
}
export -f run
export O
awk '{print $1, $2, $3}' J4s2run3/list.txt | xargs -P 4 -L 1 bash -c 'run $0 $1 $2'
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
echo DONE >> $O/status.txt
