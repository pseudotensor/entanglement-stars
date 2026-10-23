#!/bin/bash
# selftest_v36w2_box.sh (busi SYM-W1, inbox #370): the v36v self-test's prim fixture (vii resume, --dg-assign prim, --dg-sub-max 0, 512 sites) under
# engine_collapse_v36w2 named engine_collapse_v36p.py (the harness's name, as selftest_v36v_box.sh did), three runs at 8 threads in parallel on the box CPU:
#   def: the defaults (to be bitwise v36v's stored prim_vii output); log: --hbar-prop-log on (eigh); cheb: --hbar-prop chebyshev.
set -e -o pipefail
PYB=${QI2_ROOT}/python
SRC=${QI2_ROOT}/engine_collapse_v36w2.py
ST=${QI2_ROOT}/selftest
FIX="--m 0.2 --g 0.25 --rs 16 --R0-over-rs 4 --reading A --nsites 512 --kappa-sea 4 --reference R2 --site-fields averaged --ell 4 --prep-ell 4 --tau 2 --mix plain --T-over-rs 0.375 --slices-c128 0.125"
test ! -e $ST || { echo "refusing: $ST exists"; exit 1; }
mkdir -p $ST
rsync -a --exclude '__pycache__' ${QI2_ROOT}/code $ST/code/
cp -p $SRC $ST/code/engine_collapse_v36p.py; cp -p $SRC $ST/code/
md5sum $ST/code/engine_collapse_v36p.py | tee $ST/md5.txt
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
prim_run() {
  local out=$1; shift
  cp -r ${QI2_ROOT}/vii $out; rm -f $out/collapse_fz*
  local t0=$(date +%s.%N); local rc=0
  ( cd $ST/code && nice -n 10 timeout 3000 $PYB $ST/code/engine_collapse_v36p.py $FIX --v36-fixes --field-kernel-cache --out $out --tag fz --slice-dir $out/states --step dg --dg-chart frozen --dg-mix plain --resume-slice $out/states/orbitals_fz_T0000.12.npz --resume-run $out/snap/fz --dg-assign prim "$@" > $out/prim.log 2>&1 < /dev/null ) || rc=$?
  echo "prim $* -> rc $rc, $(awk -v a=$t0 -v b=$(date +%s.%N) 'BEGIN{printf "%.1f", b - a}') s ($out)" | tee -a $ST/chain.log
}
prim_run $ST/def --dg-sub-max 0 &
prim_run $ST/log --dg-sub-max 0 --hbar-prop-log on &
prim_run $ST/cheb --dg-sub-max 0 --hbar-prop chebyshev &
wait
echo "done $(date -u +%T)" >> $ST/chain.log
