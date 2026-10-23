#!/bin/bash
# Strand 5 J4 stage 2 with strand5_J4_stage2_v9.py (spec sec. 10(m)(5'): the exact second variation; basis (c) at every member, the exact
# basis (a) beside it, the full matter-node inertia and the gradient checks at five members per family).  Waits until strand 2b item 6's
# 12 R4 workers are done (thread budget), then the 20 J4run2 families 4 at a time, 2 threads each (J4s2run9).  Then, once J4run4 (the
# (0.01, 0.1) rebuild with the light packet at L = 1024) is DONE: its two local_ka0 families at the rebuilt moment (stage 1, as batch 6's
# launcher would have built them), v9 on its four families (J4s2run9b), and the conservative cross-check on v9's lock pieces (J4consrun9b).
set -u
D=${QI2_ROOT}/s5_star
cd $D || exit 1
O=J4s2run9
mkdir -p $O J4s2run9b J4consrun9b
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
until grep -q "R4 cases done" ${QI2_ROOT}/status.txt 2>/dev/null; do sleep 30; done
date -u '+start %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
KP=$D/J4kernels_run1
P1024="1:$D/J4kernels_run2/packets_C_m0.01_L1024.json,-1:$KP/packets_C_m0.10049876_L256.json,-1:$KP/packets_C_m0.10049876_L256.json,1:$KP/packets_C_m0.14177447_L256.json"
sed 's/ selected$//' J4s2run6/list.txt > $O/list.txt
run9() {  # family, option, value
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand5_J4_stage2_v9.py --family $1 $2 $3 --out $O > $O/log_$(basename $(dirname $1))_$(basename $1 .json).txt 2>&1
  echo "$(basename $(dirname $1))/$(basename $1) exit $? $(date -u +%H:%M:%S)" >> J4s2run9/status.txt
}
export -f run9
export O
cat $O/list.txt | xargs -P 4 -L 1 bash -c 'run9 $0 $1 $2'
echo "J4run2 families done $(date -u +%H:%M:%S)" >> J4s2run9/status.txt
until grep -q DONE J4run4/status.txt; do sleep 60; done
RC=$(grep '^R_c' J4run4/status.txt | awk '{print $2}')
for g in 0,0 0.2,1; do
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand5_lattice_star_v3.py --out J4run4 --packets $P1024 --mlight 0.01 --aH 0.1 --Rc $RC --operator local_ka0 --couplings $g --zmin 0.05 --zmax 3 --nmembers 20 >> J4run4/run_local_ka0_g$g.log 2>&1
  echo "m 0.01 aH 0.1 local_ka0 g $g exit $? $(date -u +%H:%M:%S)" >> J4s2run9/status.txt
done
O=J4s2run9b
export O
for f in J4run4/lstar_*_pk.json; do echo "$f --packets $P1024"; done | xargs -P 4 -L 1 bash -c 'run9 $0 $1 $2'
for f in J4run4/lstar_*_pk.json; do
  OMP_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand5_J4_stage2_cons_v2.py --family $f --packets $P1024 --stage2 J4s2run9b/stage2_$(basename $f) --out J4consrun9b > J4consrun9b/log_$(basename $f .json).txt 2>&1
  echo "cons $(basename $f) exit $? $(date -u +%H:%M:%S)" >> J4s2run9/status.txt
done
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> J4s2run9/status.txt
echo DONE >> J4s2run9/status.txt
