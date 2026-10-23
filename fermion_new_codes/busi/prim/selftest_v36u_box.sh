#!/bin/bash
# v36u perf (W2-P1 eigh cache, W2-P2 node workers) self-test on the box CPU: the v36l harness fixture at the default flags (fixv4c's harness and
# arguments, ${QI2_ROOT}/selftest_v36p_fixv4c_cpu) and the prim fixture, compared bitwise with fixv4c's stored outputs; then the options on
# (--dg-eigh-cache-gib 1 --dg-node-workers 3) on the frozen-chart runs and the prim fixture, against fixv4c; then a prim fixture with two sub-steps
# per record (fixv4c A, v36u options on B) for the bitwise check and the timing. OMP/MKL/OPENBLAS 16 threads as fixv4c's fixture.
set -e -o pipefail
PYB=${QI2_ROOT}/python
W2=${QI2_ROOT}/w2
ENG=$W2/engine
REF=${QI2_ROOT}/selftest_v36p_fixv4c_cpu
ST=${QI2_ROOT}/selftest_v36u_cpu
FIX="--m 0.2 --g 0.25 --rs 16 --R0-over-rs 4 --reading A --nsites 512 --kappa-sea 4 --reference R2 --site-fields averaged --ell 4 --prep-ell 4 --tau 2 --mix plain --T-over-rs 0.375 --slices-c128 0.125"
ON="--dg-eigh-cache-gib 1 --dg-node-workers 3"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
test ! -e $ST || { echo "refusing: $ST exists"; exit 1; }
mkdir -p $ST
rsync -a --exclude '__pycache__' $REF/code/ $ST/code/
for n in engine_collapse_v36k.py engine_collapse_v36l.py engine_collapse_v36p.py; do cp -p $ENG/engine_collapse_v36u.py $ST/code/$n; done
cp -p $ENG/engine_collapse_v36p_fixv4c.py $ENG/engine_collapse_v36u.py $ENG/engine_pairsolve_v2.py $ENG/build_engine_collapse_v36u.py $ST/code/
md5sum $ST/code/engine_collapse_v36k.py $ST/code/engine_collapse_v36l.py $ST/code/engine_collapse_v36p.py $ST/code/engine_collapse_v36u.py $ST/code/engine_collapse_v36p_fixv4c.py $ST/code/engine_pairsolve_v2.py $ST/code/engine_collapse_v36k_selftest.py $ST/code/assign_primitive_v1d.py | tee $ST/md5.txt
vii_copy() { cp -r ${QI2_ROOT}/vii $1; rm -f $1/collapse_fz*; }
prim_run() {  # $1 engine file, $2 out dir, rest: extra args
  local eng=$1 out=$2; shift 2
  vii_copy $out
  local t0=$(date +%s.%N)
  ( cd $ST/code && nice -n 10 $PYB $ST/code/$eng $FIX --v36-fixes --field-kernel-cache --out $out --tag fz --slice-dir $out/states --step dg --dg-chart frozen --dg-mix plain --resume-slice $out/states/orbitals_fz_T0000.12.npz --resume-run $out/snap/fz --dg-assign prim "$@" > $out/prim.log 2>&1 < /dev/null ); local rc=$?
  echo "$eng $* -> rc $rc, $(awk -v a=$t0 -v b=$(date +%s.%N) 'BEGIN{printf "%.1f", b - a}') s ($out)" | tee -a $ST/chain.log
}
fx_run() {  # $1 out dir, $2 tag, rest: extra args (the harness's run_engine command line)
  local out=$1 tag=$2; shift 2
  mkdir -p $out
  local t0=$(date +%s.%N)
  ( cd $ST/code && nice -n 10 $PYB $ST/code/engine_collapse_v36k.py $FIX --v36-fixes --field-kernel-cache --out $out --tag $tag --slice-dir $out/states "$@" > $out/$tag.log 2>&1 < /dev/null ); local rc=$?
  echo "engine_collapse_v36k.py(=v36u) $* -> rc $rc, $(awk -v a=$t0 -v b=$(date +%s.%N) 'BEGIN{printf "%.1f", b - a}') s ($out)" | tee -a $ST/chain.log
}
echo "== the harness at the default flags (background)"
( cd $ST && nice -n 10 $PYB $ST/code/engine_collapse_v36k_selftest.py --fixture --code $ST/code --out $ST/fx --threads 16 "--fixture-args=$FIX --v36-fixes --field-kernel-cache" > $ST/selftest_fixture_cpu.log 2>&1 < /dev/null; echo "harness rc=$?" >> $ST/selftest_fixture_cpu.log ) &
echo "harness subshell pid $!" | tee $ST/harness.pid
echo "== the chain (background): prim default, prim on, fz on, fs on, prim two sub-steps fixv4c / v36u on"
(
  set +e
  prim_run engine_collapse_v36p.py $ST/prim_vii --dg-sub-max 0
  prim_run engine_collapse_v36p.py $ST/prim_vii_on --dg-sub-max 0 $ON
  fx_run $ST/on/fz fz --step dg --dg-chart frozen --dg-mix plain --dg-sub-max 0 $ON
  fx_run $ST/on/fs fs --step dg --dg-chart frozen --dg-mix anderson --dg-sub-max 1 --dg-sub-start 1 $ON
  prim_run engine_collapse_v36p_fixv4c.py $ST/prim_fs_A --dg-sub-max 1 --dg-sub-start 1
  prim_run engine_collapse_v36p.py $ST/prim_fs_B --dg-sub-max 1 --dg-sub-start 1 $ON
  echo "chain done" >> $ST/chain.log
) > $ST/chain.out 2>&1 < /dev/null &
echo "chain subshell pid $!" | tee $ST/chain.pid
echo "== launched"
