#!/bin/bash
# G0'' of engine_twofield_v9c3 (PERF C-A7; owner order #299) on the box CPU, nice 10, 8 threads per run, no card touched:
# the build over v9c2 (md5 6f4a0e28) with the reverse byte for byte; v9c_selftest's layer, import and fixture checks on v9c3 (the "equals the given
# engine byte for byte" line is expected to FAIL: v9c3 is not the v9c builder's output); the v9b self-test's fixture with v9c3 in v9b's place; the
# stand-in fixture (1024 sites, 4 records) end to end, v9c2 at the defaults (REF) against v9c3 at the defaults (D), with --perf-solve-memo alone (M)
# and with the memo and every v9c2 option (ALL), in v9c2's three forms: sector lagged lz with a slice at 0.0625, its resume from that slice to
# 0.1875 (each arm from its own straight run's files), and record implicit with a Simpson step. The four arms run side by side (4 x 8 threads).
# Afterwards the npz files and slices are pruned (the box disk): the logs and jsons stay.
ST=${QI2_ROOT}/v9c3
cd $ST || exit 2
export CUDA_VISIBLE_DEVICES="" BUSI_ENGINE_DIR=${QI2_ROOT}/compC_engine_v36 BUSI_PAGE_DIR=${QI2_ROOT}/page
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
PY=${QI2_ROOT}/python
P="nice -n 10 $PY"
m() { md5sum < $1 | cut -c1-32; }
ALLV9C2="--perf-eig-cache-gib 4 --perf-dens-memo on --perf-uhx-cache-gib 2 --perf-writer on"
echo "# G0'' v9c3 on the box CPU (stage $ST) $(date -u +%FT%TZ)"
echo "== build $(date -u +%T)"
if [ ! -f engine_twofield_v9c3.py ]; then $P build_engine_twofield_v9c3.py engine_twofield_v9c2.py engine_twofield_v9c3.py; echo "rc $?"; else echo "engine_twofield_v9c3.py exists (not rebuilt)"; $P build_engine_twofield_v9c3.py engine_twofield_v9c2.py --check; fi
rm -f $ST/v9c3_restored.py
$P build_engine_twofield_v9c3.py engine_twofield_v9c3.py $ST/v9c3_restored.py --reverse --against engine_twofield_v9c2.py; echo "rc $?"
rm -f $ST/v9c3_restored.py
$P -m py_compile engine_twofield_v9c3.py && echo "py_compile ok"
echo "# md5 engine_twofield_v9c2.py $(m engine_twofield_v9c2.py); engine_twofield_v9c3.py $(m engine_twofield_v9c3.py); build_engine_twofield_v9c3.py $(m build_engine_twofield_v9c3.py); cmp_v9c3_runs.py $(m cmp_v9c3_runs.py); run_g0pp9c3.sh $(m run_g0pp9c3.sh); lines marked [v9c3 perf] $(grep -c 'v9c3 perf\]' engine_twofield_v9c3.py)"
echo "== G0'' selftest on v9c3 (CPU): the layer, the import, the fixture $(date -u +%T)"
$P engine_twofield_v9c_selftest.py --v9b engine_twofield_v9b.py --v9c engine_twofield_v9c3.py --fixture --skip-suite 2>&1 | grep -v UserWarning | grep -v "csr = lambda" | grep -v "^T/rs" | grep -v "^  v9c prim"; echo "rc ${PIPESTATUS[0]}"
echo "== the v9b self-test's fixture with v9c3 in v9b's place $(date -u +%T)"
W=$ST/g0runs; mkdir -p $W
SU=$W/suite_code; rm -rf $SU; mkdir -p $SU
cp engine_twofield_v9c3.py $SU/engine_twofield_v9b.py; cp engine_twofield_v9b_selftest.py engine_twofield_v8_selftest.py engine_twofield_v8.py assign_primitive_v1d.py $SU/
( cd $SU && $P engine_twofield_v9b_selftest.py fixture 2>&1 | grep -E "^(PASS|FAIL|SKIP|selftest)" ); echo "rc ${PIPESTATUS[0]}"
rm -rf $SU
E2E="--m 0.2 --g 0.25 --rs 32.0 --R0-over-rs 4 --reading A --nsites 1024 --prep-ell 4 --ell 4 --kappa-sea 10 --tau 2.0 --T-over-rs 0.1875 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 0 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --nk-jvp exact --delta-rtol-every 10 --commutator-every 1 --streams 0 --c-LZ 1.0 --window-max 4 --light-kappas 1,2 --light-guard-kappas 3 --light-coupling on --light-transport banded --light-d-min 1e-7 --s11-collars 8,16,24 --s11-stop 1 --field-kernel-cache --v36-fixes --slices-c64 '' --slices-light-c128 ''"
run() { local log=$1; local eng=$2; shift 2; local t0=$(date +%s); eval $P $eng $E2E "$@" > $log 2>&1; local rc=$?; echo "rc $rc in $(( $(date +%s) - t0 )) s ($log)"; return $rc; }
SLZ="--reference-solve lagged --cut-transport sector --anchor-passage lz --m1-simpson-steps ''"
RI="--cut-transport record --m1-simpson-steps 1"
arm_run() {
  local arm=$1 eng opt
  case $arm in REF) eng=engine_twofield_v9c2.py; opt="";; D) eng=engine_twofield_v9c3.py; opt="";; M) eng=engine_twofield_v9c3.py; opt="--perf-solve-memo on";; ALL) eng=engine_twofield_v9c3.py; opt="--perf-solve-memo on $ALLV9C2";; esac
  local d=$W/$arm; rm -rf $d; mkdir -p $d/sLz/states $d/res/states $d/ri
  echo "== $arm ($eng $opt): sector lagged lz with a slice at 0.0625 $(date -u +%T)"
  run $d/sLz/run.log $eng $SLZ --slices-c128 0.0625 --slice-dir $d/sLz/states --out $d/sLz --tag e2e_v9cL $opt
  local SL=$(ls $d/sLz/states/orbitals_e2e_v9cL_T*.npz | head -1)
  echo "== $arm: its resume from $(basename $SL) to 0.1875 $(date -u +%T)"
  run $d/res/run.log $eng $SLZ --slices-c128 "''" --slice-dir $d/res/states --out $d/res --tag e2e_v9cL --resume-slice $SL --resume-run $d/sLz/e2e_v9cL --segment-end-T-over-rs 0.1875 $opt
  echo "== $arm: record implicit with a Simpson step at step 1 $(date -u +%T)"
  run $d/ri/run.log $eng $RI --slices-c128 "''" --out $d/ri --tag e2e_v9ri $opt
  echo "  $arm leftover .v9c2tmp files: $(ls $d/sLz $d/res | grep -c v9c2tmp)"
}
for arm in REF D M ALL; do arm_run $arm > $W/arm_$arm.log 2>&1 & done
wait
for arm in REF D M ALL; do cat $W/arm_$arm.log; done
echo "== compares (v9c2 at the defaults = A) $(date -u +%T)"
for form in sLz/e2e_v9cL res/e2e_v9cL ri/e2e_v9ri; do
  $PY cmp_v9c3_runs.py $W/REF/$form $W/D/$form --args-equal --b-log $W/D/${form%%/*}/run.log
  for arm in M ALL; do $PY cmp_v9c3_runs.py $W/REF/$form $W/$arm/$form --b-log $W/$arm/${form%%/*}/run.log; done
done
echo "-- the resume against the straight run after 0.0625 (G0'' finding 8's strip list), per arm"
for arm in REF D M ALL; do
$PY - $W/$arm/sLz/e2e_v9cL $W/$arm/res/e2e_v9cL $arm <<'EOF'
import sys
sys.path.insert(0, "${QI2_ROOT}/v9c3")
import engine_twofield_v8_selftest as S8
import engine_twofield_v9b_selftest as S9
common, diffs = S9.compare_runs_stripped(S8, sys.argv[1], sys.argv[2], {"engine_v34", "label"}, after=0.0625)
kept = [d for d in diffs if not str(d[0]).startswith("s11/r_nodes")]
print("%s %s: straight against resume over %d common records after 0.0625: %d differences%s" % ("PASS" if (common and not kept) else "FAIL", sys.argv[3], len(common), len(kept), (" %s" % kept[:4]) if kept else ""))
EOF
done
echo "-- the resumed runs compared with the straight-run strip list (label and engine_v34 carry the run paths), M and ALL against REF"
for arm in D M ALL; do
$PY - $W/REF/res/e2e_v9cL $W/$arm/res/e2e_v9cL $arm <<'EOF'
import sys
sys.path.insert(0, "${QI2_ROOT}/v9c3")
import engine_twofield_v8_selftest as S8
import engine_twofield_v9b_selftest as S9
common, diffs = S9.compare_runs_stripped(S8, sys.argv[1], sys.argv[2], {"engine_v34", "label"})
print("%s %s res against REF res, label and engine_v34 stripped: %d common records, %d differences%s" % ("PASS" if (common and not diffs) else "FAIL", sys.argv[3], len(common), len(diffs), (" %s" % diffs[:4]) if diffs else ""))
EOF
done
echo "-- record seconds per arm and form (indicative only: 4 arms side by side)"
for arm in REF D M ALL; do for form in sLz/collapse_e2e_v9cL res/collapse_e2e_v9cL ri/collapse_e2e_v9ri; do
$PY - $W/$arm/$form.json $arm $form <<'EOF'
import json, sys
J = json.load(open(sys.argv[1])); R = [r for r in J["records"] if r.get("E") is not None]
s = [r.get("seconds", 0) for r in R]; t = [(r.get("timing") or {}) for r in R]
print("%s %s: record seconds %s sum %.1f; solve_s sum %.1f; solve_n %s" % (sys.argv[2], sys.argv[3], [round(x, 1) for x in s], sum(s), sum(x.get("solve_s", 0) for x in t), [x.get("solve_n") for x in t]))
EOF
done; done
echo "-- prune (the box disk): slices and npz removed, logs and jsons kept"
find $W -name "*.npz" -delete; find $W -type d -name states -empty -delete
du -sh $W
echo "== done $(date -u +%T)"
