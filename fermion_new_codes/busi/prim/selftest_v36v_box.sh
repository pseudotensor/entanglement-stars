#!/bin/bash
# v36v dg (busi/notes/W2_pass12625_ruling_reviewa_v1.md sec. 3: --dg-stall, --dg-sub-counts) self-test on the box CPU, selftest_v36u_box.sh's pattern.
# Phase 1, options off, OMP 16 threads (fixv4c's fixture thread count), serial: the prim fixture (vii resume, --dg-assign prim, sub-max 0) and the
#   harness's frozen-chart runs fz (plain, level 0) and fs (anderson, level 1, two sub-steps), v36v under the harness's engine names, compared bitwise
#   with fixv4c's stored outputs in ${QI2_ROOT}/selftest_v36p_fixv4c_cpu by cmp_v36v_busi.py.
# Phase 2, options on, four runs in parallel at 4 threads each (16 in all): (c1) the prim fixture + --dg-stall 1 (ruling (c)); (c2) the anderson fs
#   run + --dg-stall 1 (the Anderson purge and mixer.beta exercised); (d) a fresh prim run with --dg-stall 8 --dg-sub-counts 1,3,2 --dg-sub-start 1
#   --dg-sub-max 2 (ruling (d); fresh because the vii slice carries dg_level_v36k 0, which a resume reads in place of --dg-sub-start); (e) the prim
#   fixture with --dg-cap 2 --dg-sub-counts 1,2,3 --dg-sub-start 0 --dg-sub-max 2 (ruling (e): the counts' control flow to the STOP).
set -e -o pipefail
PYB=${QI2_ROOT}/python
ENG=${QI2_ROOT}/engine
REF=${QI2_ROOT}/selftest_v36p_fixv4c_cpu
ST=${QI2_ROOT}/selftest_v36v_cpu
FIX="--m 0.2 --g 0.25 --rs 16 --R0-over-rs 4 --reading A --nsites 512 --kappa-sea 4 --reference R2 --site-fields averaged --ell 4 --prep-ell 4 --tau 2 --mix plain --T-over-rs 0.375 --slices-c128 0.125"
test ! -e $ST || { echo "refusing: $ST exists"; exit 1; }
mkdir -p $ST
rsync -a --exclude '__pycache__' $REF/code/ $ST/code/
for n in engine_collapse_v36k.py engine_collapse_v36l.py engine_collapse_v36p.py; do cp -p $ENG/engine_collapse_v36v.py $ST/code/$n; done
cp -p $ENG/engine_collapse_v36p_fixv4c.py $ENG/engine_collapse_v36v.py $ENG/build_engine_collapse_v36v.py $ST/code/
md5sum $ST/code/engine_collapse_v36k.py $ST/code/engine_collapse_v36l.py $ST/code/engine_collapse_v36p.py $ST/code/engine_collapse_v36v.py $ST/code/engine_collapse_v36p_fixv4c.py $ST/code/assign_primitive_v1d.py | tee $ST/md5.txt
thr() { export OMP_NUM_THREADS=$1 MKL_NUM_THREADS=$1 OPENBLAS_NUM_THREADS=$1; }
prim_run() {  # $1 out dir, rest: extra args (selftest_v36u_box.sh's prim_run, engine v36v as engine_collapse_v36p.py)
  local out=$1; shift
  cp -r ${QI2_ROOT}/vii $out; rm -f $out/collapse_fz*
  local t0=$(date +%s.%N); local rc=0
  ( cd $ST/code && nice -n 10 $PYB $ST/code/engine_collapse_v36p.py $FIX --v36-fixes --field-kernel-cache --out $out --tag fz --slice-dir $out/states --step dg --dg-chart frozen --dg-mix plain --resume-slice $out/states/orbitals_fz_T0000.12.npz --resume-run $out/snap/fz --dg-assign prim "$@" > $out/prim.log 2>&1 < /dev/null ) || rc=$?
  echo "prim $* -> rc $rc, $(awk -v a=$t0 -v b=$(date +%s.%N) 'BEGIN{printf "%.1f", b - a}') s ($out)" | tee -a $ST/chain.log
}
fx_run() {  # $1 out dir, $2 tag, rest: extra args (the harness's run_engine command line; engine v36v as engine_collapse_v36k.py)
  local out=$1 tag=$2; shift 2
  mkdir -p $out
  local t0=$(date +%s.%N); local rc=0
  ( cd $ST/code && nice -n 10 $PYB $ST/code/engine_collapse_v36k.py $FIX --v36-fixes --field-kernel-cache --out $out --tag $tag --slice-dir $out/states "$@" > $out/$tag.log 2>&1 < /dev/null ) || rc=$?
  echo "fx $* -> rc $rc, $(awk -v a=$t0 -v b=$(date +%s.%N) 'BEGIN{printf "%.1f", b - a}') s ($out)" | tee -a $ST/chain.log
}
(
  set +e
  echo "== phase 1 (options off, 16 threads): $(date -u +%T)" >> $ST/chain.log
  thr 16
  prim_run $ST/prim_vii --dg-sub-max 0
  fx_run $ST/off/fs fs --step dg --dg-chart frozen --dg-mix anderson --dg-sub-max 1 --dg-sub-start 1
  fx_run $ST/off/fz fz --step dg --dg-chart frozen --dg-mix plain --dg-sub-max 0
  echo "== phase 2 (options on, 4 x 4 threads): $(date -u +%T)" >> $ST/chain.log
  thr 4
  prim_run $ST/on_c1 --dg-sub-max 0 --dg-stall 1 &
  fx_run $ST/on_c2/fs fs --step dg --dg-chart frozen --dg-mix anderson --dg-sub-max 1 --dg-sub-start 1 --dg-stall 1 &
  fx_run $ST/on_d/fz fz --step dg --dg-chart frozen --dg-mix plain --dg-assign prim --dg-stall 8 --dg-sub-counts 1,3,2 --dg-sub-start 1 --dg-sub-max 2 &
  prim_run $ST/on_e --dg-cap 2 --dg-sub-counts 1,2,3 --dg-sub-start 0 --dg-sub-max 2 &
  wait
  echo "chain done $(date -u +%T)" >> $ST/chain.log
) > $ST/chain.out 2>&1 < /dev/null &
echo "chain subshell pid $!" | tee $ST/chain.pid
