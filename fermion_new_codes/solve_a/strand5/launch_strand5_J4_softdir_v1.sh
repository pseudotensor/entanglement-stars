#!/bin/bash
# Strand 5 sec. 10(r)(a): the soft full-node directions at the five S3b members of every family (strand5_J4_softdir_v1.py): the 20 J4run2
# families (stage 2 J4s2run9) and the 4 J4run4 families (J4s2run9b), 4 at a time, 2 threads each; builder analyst.
set -u
D=${QI2_ROOT}/s5_star
cd $D || exit 1
O=J4softdir1
mkdir -p $O
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
KP=$D/J4kernels_run1
P1024="1:$D/J4kernels_run2/packets_C_m0.01_L1024.json,-1:$KP/packets_C_m0.10049876_L256.json,-1:$KP/packets_C_m0.10049876_L256.json,1:$KP/packets_C_m0.14177447_L256.json"
{ while read -r f opt val; do echo "$f $opt $val J4s2run9/stage2_$(basename $f)"; done < J4s2run9/list.txt
  for f in J4run4/lstar_*_pk.json; do echo "$f --packets $P1024 J4s2run9b/stage2_$(basename $f)"; done; } > $O/list.txt
sd() {  # family option value stage2
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand5_J4_softdir_v1.py --family $1 $2 $3 --stage2 $4 --out $O > $O/log_$(basename $(dirname $1))_$(basename $1 .json).txt 2>&1
  echo "$(basename $(dirname $1))/$(basename $1) exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
}
export -f sd
export O
cat $O/list.txt | xargs -P 4 -L 1 bash -c 'sd $0 $1 $2 $3'
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
echo DONE >> $O/status.txt
