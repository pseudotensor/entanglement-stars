#!/bin/bash
# v36w perf (W2-P3: --streams 1 made safe) self-test on the box CPU, the v36u harness and arguments (selftest_v36u_box.sh), run SERIALLY (the box is
# loaded: one process at a time, 16 threads, nice 10): the helper unit test; the prim fixture at the defaults and the --streams 1 runs (alone, and with
# --dg-eigh-cache-gib 1 --dg-node-workers 3) against fixv4c (prim_fs_A of the v36u stage, fx/fz of fixv4c's stage); then the v36l harness at the
# default flags against fixv4c's stored fixture; then cmp_v36w_busi.py. On the CPU eigh_all's threaded branch is never entered (its guard needs a CUDA
# device), so these runs show the defaults and the --streams 1 flag bitwise; the warm-up and the re-raise under real threads need the H100 A/B.
set -e -o pipefail
PYB=${QI2_ROOT}/python
W2=${QI2_ROOT}/w2
ENG=$W2/engine
REF=${QI2_ROOT}/selftest_v36p_fixv4c_cpu
ST=${QI2_ROOT}/selftest_v36w_cpu
FIX="--m 0.2 --g 0.25 --rs 16 --R0-over-rs 4 --reading A --nsites 512 --kappa-sea 4 --reference R2 --site-fields averaged --ell 4 --prep-ell 4 --tau 2 --mix plain --T-over-rs 0.375 --slices-c128 0.125"
ON="--dg-eigh-cache-gib 1 --dg-node-workers 3"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
test ! -e $ST || { echo "refusing: $ST exists"; exit 1; }
mkdir -p $ST
rsync -a --exclude '__pycache__' $REF/code/ $ST/code/
for n in engine_collapse_v36k.py engine_collapse_v36l.py engine_collapse_v36p.py; do cp -p $ENG/engine_collapse_v36w.py $ST/code/$n; done
cp -p $ENG/engine_collapse_v36p_fixv4c.py $ENG/engine_collapse_v36u.py $ENG/engine_collapse_v36w.py $ENG/engine_pairsolve_v2.py $ENG/build_engine_collapse_v36w.py $ENG/test_v36w_helpers.py $ENG/cmp_v36w_busi.py $ST/code/
md5sum $ST/code/engine_collapse_v36k.py $ST/code/engine_collapse_v36l.py $ST/code/engine_collapse_v36p.py $ST/code/engine_collapse_v36w.py $ST/code/engine_collapse_v36u.py $ST/code/engine_collapse_v36p_fixv4c.py $ST/code/engine_pairsolve_v2.py $ST/code/engine_collapse_v36k_selftest.py $ST/code/assign_primitive_v1d.py | tee $ST/md5.txt
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
  echo "engine_collapse_v36k.py(=v36w) $* -> rc $rc, $(awk -v a=$t0 -v b=$(date +%s.%N) 'BEGIN{printf "%.1f", b - a}') s ($out)" | tee -a $ST/chain.log
}
echo "== the serial chain (background): unit test, prim streams-1 runs, fz on, prim default, harness, compare"
(
  set +e
  ( cd $ST/code && nice -n 10 $PYB $ST/code/test_v36w_helpers.py $ST/code > $ST/test_helpers.log 2>&1 < /dev/null ); echo "helpers rc $?: $(tail -1 $ST/test_helpers.log)" | tee -a $ST/chain.log
  prim_run engine_collapse_v36p.py $ST/prim_fs_B --dg-sub-max 1 --dg-sub-start 1 --streams 1 $ON
  prim_run engine_collapse_v36p.py $ST/prim_fs_S1 --dg-sub-max 1 --dg-sub-start 1 --streams 1
  fx_run $ST/on/fz fz --step dg --dg-chart frozen --dg-mix plain --dg-sub-max 0 --streams 1 $ON
  prim_run engine_collapse_v36p.py $ST/prim_vii --dg-sub-max 0
  ( cd $ST && nice -n 10 $PYB $ST/code/engine_collapse_v36k_selftest.py --fixture --code $ST/code --out $ST/fx --threads 16 "--fixture-args=$FIX --v36-fixes --field-kernel-cache" > $ST/selftest_fixture_cpu.log 2>&1 < /dev/null; echo "harness rc=$?" >> $ST/selftest_fixture_cpu.log )
  echo "harness done" >> $ST/chain.log
  ( cd $ST && nice -n 10 $PYB $ST/code/cmp_v36w_busi.py all > $ST/cmp.log 2>&1 < /dev/null ); echo "cmp: $(tail -1 $ST/cmp.log)" >> $ST/chain.log
  echo "chain done" >> $ST/chain.log
) > $ST/chain.out 2>&1 < /dev/null &
echo "chain subshell pid $!" | tee $ST/chain.pid
echo "== launched"
