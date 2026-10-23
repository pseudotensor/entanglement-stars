#!/bin/bash
# [rebased copy for ReviewA_busi_open_q12_v1 sec. 1] run_g0pp9c2.sh with the late-step-fixed engines (v9c1b/v9c2b/v9c3b, assign_primitive_v1e), the self-test v1y
# (x and the y toy) and the work dir on ${QI2_ROOT}/busi_bulk; every step as in run_g0pp9c2.sh.
# G0'' of engine_twofield_v9c2 (PERF C-A1, C-A3, C-A4, C-A5; owner order #299) on the box CPU, nice 10, 8 threads, no card touched:
# the build over v9c1 (md5 84938595) with the reverse byte for byte; v9c_selftest's layer, import and fixture checks on v9c2 (the "equals the given
# engine byte for byte" line is expected to FAIL: v9c2 is not the v9c builder's output; x is v9c1's known CPU FAIL, ready_jobs_v3 11.0); the v9b
# self-test's fixture with v9c2 in v9b's place; the stand-in fixture (1024 sites, 4 records) end to end, v9c1 against v9c2 at the defaults and with
# every option on, in three forms: sector lagged lz with a slice at 0.0625 (the gates' form), its resume from that slice to 0.1875 (each arm from
# its own straight run's files: the writer's files are read back), and record implicit with a Simpson step (the eig cache under continuation, halfp).
ST=${QI2_ROOT}/v9c2b
cd $ST || exit 2
export CUDA_VISIBLE_DEVICES="" BUSI_ENGINE_DIR=${QI2_ROOT}/compC_engine_v36 BUSI_PAGE_DIR=${QI2_ROOT}/page
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
PY=${QI2_ROOT}/python
P="nice -n 10 $PY"
m() { md5sum < $1 | cut -c1-32; }
ON="--perf-eig-cache-gib 4 --perf-dens-memo on --perf-uhx-cache-gib 2 --perf-writer on"
echo "# G0'' v9c2 on the box CPU (stage $ST) $(date -u +%FT%TZ)"
echo "== build $(date -u +%T)"
if [ ! -f engine_twofield_v9c2b.py ]; then $P build_engine_twofield_v9c2b.py engine_twofield_v9c1b.py engine_twofield_v9c2b.py; echo "rc $?"; else echo "engine_twofield_v9c2b.py exists (not rebuilt)"; $P build_engine_twofield_v9c2b.py engine_twofield_v9c1b.py --check; fi
rm -f ${QI2_ROOT}/v9c2b_restored.py
$P build_engine_twofield_v9c2b.py engine_twofield_v9c2b.py ${QI2_ROOT}/v9c2b_restored.py --reverse --against engine_twofield_v9c1b.py; echo "rc $?"
$P -m py_compile engine_twofield_v9c2b.py && echo "py_compile ok"
echo "# md5 engine_twofield_v9c1b.py $(m engine_twofield_v9c1b.py); engine_twofield_v9c2b.py $(m engine_twofield_v9c2b.py); build_engine_twofield_v9c2b.py $(m build_engine_twofield_v9c2b.py); cmp_v9c2_runs.py $(m cmp_v9c2_runs.py); run_g0pp9c2b.sh $(m run_g0pp9c2b.sh); lines marked [v9c2 perf] $(grep -c 'v9c2 perf\]' engine_twofield_v9c2b.py)"
echo "== G0'' selftest on v9c2 (CPU): the layer, the import, the fixture (u skipped here: it runs the builder's v9c1 text; v9c2's v9b-suite run is below) $(date -u +%T)"
$P engine_twofield_v9c_selftest_v1y.py --v9b engine_twofield_v9b.py --v9c engine_twofield_v9c2b.py --fixture --skip-suite 2>&1 | grep -v UserWarning | grep -v "csr = lambda" | grep -v "^T/rs" | grep -v "^  v9c prim"; echo "rc ${PIPESTATUS[0]}"
echo "== the v9b self-test's fixture with v9c2 in v9b's place (as v9c_selftest's u does for v9c1) $(date -u +%T)"
W=${QI2_ROOT}/g0pp9c2b; mkdir -p $W
SU=$W/suite_code; rm -rf $SU; mkdir -p $SU
cp engine_twofield_v9c2b.py $SU/engine_twofield_v9b.py; cp engine_twofield_v9b_selftest.py engine_twofield_v8_selftest.py engine_twofield_v8.py assign_primitive_v1d.py assign_primitive_v1e.py $SU/
( cd $SU && $P engine_twofield_v9b_selftest.py fixture 2>&1 | grep -E "^(PASS|FAIL|SKIP|selftest)" ); echo "rc ${PIPESTATUS[0]}"
rm -rf $SU
E2E="--m 0.2 --g 0.25 --rs 32.0 --R0-over-rs 4 --reading A --nsites 1024 --prep-ell 4 --ell 4 --kappa-sea 10 --tau 2.0 --T-over-rs 0.1875 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 0 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --nk-jvp exact --delta-rtol-every 10 --commutator-every 1 --streams 0 --c-LZ 1.0 --window-max 4 --light-kappas 1,2 --light-guard-kappas 3 --light-coupling on --light-transport banded --light-d-min 1e-7 --s11-collars 8,16,24 --s11-stop 1 --field-kernel-cache --v36-fixes --slices-c64 '' --slices-light-c128 ''"
run() { local log=$1; local eng=$2; shift 2; local t0=$(date +%s); eval $P $eng $E2E "$@" > $log 2>&1; local rc=$?; echo "rc $rc in $(( $(date +%s) - t0 )) s ($log)"; return $rc; }
SLZ="--reference-solve lagged --cut-transport sector --anchor-passage lz --m1-simpson-steps ''"
RI="--cut-transport record --m1-simpson-steps 1"
for arm in ON REF D; do
  case $arm in REF) eng=engine_twofield_v9c1b.py; opt="";; D) eng=engine_twofield_v9c2b.py; opt="";; ON) eng=engine_twofield_v9c2b.py; opt="$ON";; esac
  d=$W/$arm; rm -rf $d; mkdir -p $d/sLz/states $d/res/states $d/ri
  echo "== $arm ($eng $opt): sector lagged lz with a slice at 0.0625 $(date -u +%T)"
  run $d/sLz/run.log $eng $SLZ --slices-c128 0.0625 --slice-dir $d/sLz/states --out $d/sLz --tag e2e_v9cL $opt
  if [ $arm = D ]; then continue; fi   # the defaults arm: the production form only (its wrappers run v9c1's own expressions)
  SL=$(ls $d/sLz/states/orbitals_e2e_v9cL_T*.npz | head -1)
  echo "== $arm: its resume from $(basename $SL) to 0.1875 $(date -u +%T)"
  run $d/res/run.log $eng $SLZ --slices-c128 "''" --slice-dir $d/res/states --out $d/res --tag e2e_v9cL --resume-slice $SL --resume-run $d/sLz/e2e_v9cL --segment-end-T-over-rs 0.1875 $opt
  echo "== $arm: record implicit with a Simpson step at step 1 $(date -u +%T)"
  run $d/ri/run.log $eng $RI --slices-c128 "''" --out $d/ri --tag e2e_v9ri $opt
  ls $d/sLz $d/res | grep -c v9c2tmp | sed 's/^/  leftover .v9c2tmp files: /'
done
echo "== compares (v9c1 = A) $(date -u +%T)"
$PY cmp_v9c2_runs.py $W/REF/sLz/e2e_v9cL $W/D/sLz/e2e_v9cL --args-equal --b-log $W/D/sLz/run.log
for form in sLz/e2e_v9cL res/e2e_v9cL ri/e2e_v9ri; do
  $PY cmp_v9c2_runs.py $W/REF/$form $W/ON/$form --b-log $W/ON/${form%%/*}/run.log
done
echo "-- the resume against the straight run after 0.0625 (G0'' finding 8's strip list), per arm"
for arm in REF ON; do
$PY - $W/$arm/sLz/e2e_v9cL $W/$arm/res/e2e_v9cL $arm <<'EOF'
import sys
sys.path.insert(0, "${QI2_ROOT}/v9c2b")
import engine_twofield_v8_selftest as S8
import engine_twofield_v9b_selftest as S9
common, diffs = S9.compare_runs_stripped(S8, sys.argv[1], sys.argv[2], {"engine_v34", "label"}, after=0.0625)
kept = [d for d in diffs if not str(d[0]).startswith("s11/r_nodes")]
print("%s %s: straight against resume over %d common records after 0.0625: %d differences%s" % ("PASS" if (common and not kept) else "FAIL", sys.argv[3], len(common), len(kept), (" %s" % kept[:4]) if kept else ""))
EOF
done
du -sh $W
echo "== done $(date -u +%T)"
