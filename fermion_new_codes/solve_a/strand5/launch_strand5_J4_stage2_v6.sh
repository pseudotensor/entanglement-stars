#!/bin/bash
# Strand 5 J4 stage 2 (spec sec. 10(h), (j), (k)), batch 6 with strand5_J4_stage2_v7.py (two step pairs; the single-mode positivity cap): the 20
# primary families of J4run2 with the Hessian on the selected members, 12 at a time; batch 5's eight lock-only comparison families copied in
# (their lock pieces do not depend on the Hessian's version); then, once J4run4 (sec. 10(k)) is DONE, the (0.01, 0.1) local_ka0 stage 1 at the
# rebuilt kernel's moment (both couplings, into J4run4) and the four J4run4 families' stage 2 with the Hessian (into J4s2run6b: their file
# names repeat J4run3's), then their conservative cross-check (strand5_J4_stage2_cons_v2.py, into J4consrun5b).  2 threads per family.
set -u
D=${QI2_ROOT}/s5_star
cd $D || exit 1
O=J4s2run6
mkdir -p $O J4s2run6b J4consrun5b
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
K2C=${QI2_ROOT}/kernel_L256.json
KJ4=$D/J2b_run1/kernel_J4_L256.json
KP=$D/J4kernels_run1
P1024="1:$D/J4kernels_run2/packets_C_m0.01_L1024.json,-1:$KP/packets_C_m0.10049876_L256.json,-1:$KP/packets_C_m0.10049876_L256.json,1:$KP/packets_C_m0.14177447_L256.json"
for f in J4run2/lstar_m0.01_aH0.1_kernel_g0_0 J4run2/lstar_m0.01_aH0.1_kernel_g0.2_1 J4run2/lstar_m0.01_aH0.1_local_ka0_g0_0 J4run2/lstar_m0.01_aH0.1_local_ka0_g0.2_1; do
  cp J4s2run5/stage2_$(basename $f).json $O/; done
for f in J4run3/lstar_*_pk.json; do cp J4s2run5/stage2_$(basename $f) $O/stage2_J4run3_$(basename $f) 2>/dev/null || true; done
LIST=$O/list.txt
: > $LIST
for f in J4run2/lstar_m0.03_aH0.3_*.json J4run2/lstar_m0.01_aH0.1_local_klat_*.json J4run2/lstar_m0.1_aH*.json; do case $f in *_profiles*) continue;; esac
  case $f in *m0.1_aH*) k="--kernel $K2C";; *) k="--kernel $KJ4";; esac; echo "$f $k selected" >> $LIST; done
run() {  # family, kernel option, its value, hessian mode
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand5_J4_stage2_v7.py --family $1 $2 $3 --out $O --hessian $4 > $O/log_$(basename $(dirname $1))_$(basename $1 .json).txt 2>&1
  echo "$(basename $(dirname $1))/$(basename $1) hessian $4 exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
}
export -f run
export O
cat $LIST | xargs -P 12 -L 1 bash -c 'run $0 $1 $2 $3'
until grep -q DONE J4run4/status.txt; do sleep 60; done
RC=$(grep '^R_c' J4run4/status.txt | awk '{print $2}')
for g in 0,0 0.2,1; do
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand5_lattice_star_v3.py --out J4run4 --packets $P1024 --mlight 0.01 --aH 0.1 --Rc $RC --operator local_ka0 --couplings $g --zmin 0.05 --zmax 3 --nmembers 20 >> J4run4/run_local_ka0_g$g.log 2>&1
  echo "m 0.01 aH 0.1 local_ka0 g $g exit $? $(date -u +%H:%M:%S)" >> J4run4/status.txt
done
O=J4s2run6b
export O
for f in J4run4/lstar_*_pk.json; do echo "$f --packets $P1024 selected"; done | xargs -P 4 -L 1 bash -c 'run $0 $1 $2 $3'
for f in J4run4/lstar_*_pk.json; do
  OMP_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand5_J4_stage2_cons_v2.py --family $f --packets $P1024 --stage2 J4s2run6b/stage2_$(basename $f) --out J4consrun5b > J4consrun5b/log_$(basename $f .json).txt 2>&1
  echo "cons $(basename $f) exit $? $(date -u +%H:%M:%S)" >> J4s2run6/status.txt
done
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> J4s2run6/status.txt
echo DONE >> J4s2run6/status.txt
