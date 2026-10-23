#!/bin/bash
# v36v dg self-test, phase 3 (after selftest_v36v_box.sh, same stage ${QI2_ROOT}/selftest_v36v_cpu): the stall rule EXERCISED. In phase 2 the
# fixture's node map contracts about 100x per iterate (residual 7e-5, 1e-6, 1e-8, 2e-10), so --dg-stall never fired (0 events: the residual always
# fell below 0.7 of its best). Here the damping is lowered so that the residual falls by about 0.75 per iterate and the rule must fire, on one record
# (the prim fixture's resume step 0.125 -> 0.25; --T-over-rs 0.25 overrides the fixture's 0.375), two runs in parallel at 8 threads each:
#   (c3) plain, --dg-beta 0.25 --dg-stall 2 --dg-cap 80: stall events every 2 iterates, the trust region halved to its floor (the plain path's beta_eff/clip_eff);
#   (c4) anderson, --dg-beta 0.25 --dg-stall 1 --dg-cap 80: the Anderson history purged and mixer.beta set at every event.
# Gate (ruling (c)): every sub-step converges, E/E0 at 0.25 within 1e-10 of the options-off prim_vii record, no Traceback.
set -e -o pipefail
PYB=${QI2_ROOT}/python
ST=${QI2_ROOT}/selftest_v36v_cpu
FIX="--m 0.2 --g 0.25 --rs 16 --R0-over-rs 4 --reading A --nsites 512 --kappa-sea 4 --reference R2 --site-fields averaged --ell 4 --prep-ell 4 --tau 2 --mix plain --T-over-rs 0.375 --slices-c128 0.125"
test -d $ST/code && test ! -e $ST/on_c3 && test ! -e $ST/on_c4 || { echo "refusing: $ST/code missing or on_c3/on_c4 exist"; exit 1; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
prim_run() {  # $1 out dir, rest: extra args (selftest_v36v_box.sh's prim_run)
  local out=$1; shift
  cp -r ${QI2_ROOT}/vii $out; rm -f $out/collapse_fz*
  local t0=$(date +%s.%N); local rc=0
  ( cd $ST/code && nice -n 10 $PYB $ST/code/engine_collapse_v36p.py $FIX --v36-fixes --field-kernel-cache --out $out --tag fz --slice-dir $out/states --step dg --dg-chart frozen --dg-mix plain --resume-slice $out/states/orbitals_fz_T0000.12.npz --resume-run $out/snap/fz --dg-assign prim "$@" > $out/prim.log 2>&1 < /dev/null ) || rc=$?
  echo "prim $* -> rc $rc, $(awk -v a=$t0 -v b=$(date +%s.%N) 'BEGIN{printf "%.1f", b - a}') s ($out)" | tee -a $ST/chain.log
}
(
  set +e
  echo "== phase 3 (the stall rule exercised, 2 x 8 threads): $(date -u +%T)" >> $ST/chain.log
  prim_run $ST/on_c3 --dg-sub-max 0 --T-over-rs 0.25 --dg-beta 0.25 --dg-stall 2 --dg-cap 80 &
  prim_run $ST/on_c4 --dg-sub-max 0 --T-over-rs 0.25 --dg-mix anderson --dg-beta 0.25 --dg-stall 1 --dg-cap 80 &
  wait
  echo "phase 3 done $(date -u +%T)" >> $ST/chain.log
) > $ST/chain3.out 2>&1 < /dev/null &
echo "phase 3 subshell pid $!" | tee $ST/chain3.pid
