#!/bin/bash
# G0'' of engine_twofield_v9c1b (ReviewA_busi_open_q12_v1 sec. 1: v9c1 + the late-step fix: assign_primitive_v1e, entries carry the samples) on the
# box CPU only (no box GPU was free at the build: all three A6000s carried other lanes' processes), nice 10, 8 threads per run.
# Stage compC/v9c1b (a copy of compC/v9c1's top-level files + build_engine_twofield_v9c1b.py staged as build_engine_twofield_v9c.py, v1e, the
# self-test v1y with the y group and toy_late_step_v1.py). Work and outputs on ${QI2_ROOT}/g0pp9c1b.
# Steps: the builds (v1e over v1d, v9c1b over v9b) with their reverses; the self-test v1y (layer with x and y, build, import, fixture, the v9b suite
# in v9b's place) on the CPU; the stand-in fixture (1024 sites, 4 records) v9c1 (REF) against v9c1b (B), side by side on the CPU, in four forms:
# sector lagged floor (bitwise: every record and npz row), sector lagged lz with a slice at 0.0625 (bitwise but for the passage blocks' prim
# VERSION and the entries' samples: compared with the sector keys stripped), its resume from that slice to 0.1875, and record implicit with a
# Simpson step (bitwise); the two expected refusals on B; the toy standalone on v9c1b's built engine.
ST=${QI2_ROOT}/v9c1b
W=${QI2_ROOT}/g0pp9c1b
cd $ST || exit 2
mkdir -p $W
export CUDA_VISIBLE_DEVICES="" BUSI_ENGINE_DIR=${QI2_ROOT}/compC_engine_v36 BUSI_PAGE_DIR=${QI2_ROOT}/page
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
PY=${QI2_ROOT}/python
P="nice -n 10 $PY"
m() { md5sum < $1 | cut -c1-32; }
echo "# G0'' v9c1b on the box CPU (stage $ST, work $W) $(date -u +%FT%TZ)"
echo "== builds $(date -u +%T)"
$P build_assign_primitive_v1e.py assign_primitive_v1d.py --check; echo "rc $?"
rm -f $W/v1e_restored.py; $P build_assign_primitive_v1e.py assign_primitive_v1e.py $W/v1e_restored.py --reverse --against assign_primitive_v1d.py; echo "rc $?"; rm -f $W/v1e_restored.py
$P build_engine_twofield_v9c.py engine_twofield_v9b.py --check | cut -c1-400; echo "rc ${PIPESTATUS[0]}"
rm -f $W/v9c1b_restored.py; $P build_engine_twofield_v9c.py engine_twofield_v9c1b.py $W/v9c1b_restored.py --reverse --against engine_twofield_v9b.py | cut -c1-300; echo "rc ${PIPESTATUS[0]}"; rm -f $W/v9c1b_restored.py
echo "-- v9c1 -> v9c1b: changed lines $(diff engine_twofield_v9c1.py engine_twofield_v9c1b.py | grep -c '^>'), unmarked $(diff engine_twofield_v9c1.py engine_twofield_v9c1b.py | grep '^>' | grep -vc 'v9c1b x')"
echo "# md5 engine_twofield_v9b.py $(m engine_twofield_v9b.py); engine_twofield_v9c1.py $(m engine_twofield_v9c1.py); engine_twofield_v9c1b.py $(m engine_twofield_v9c1b.py); build_engine_twofield_v9c.py (= v9c1b's builder) $(m build_engine_twofield_v9c.py); build_engine_twofield_v9c1b.py $(m build_engine_twofield_v9c1b.py); assign_primitive_v1d.py $(m assign_primitive_v1d.py); assign_primitive_v1e.py $(m assign_primitive_v1e.py); build_assign_primitive_v1e.py $(m build_assign_primitive_v1e.py); engine_twofield_v9c_selftest_v1y.py $(m engine_twofield_v9c_selftest_v1y.py); toy_late_step_v1.py $(m toy_late_step_v1.py); run_g0pp9c1b.sh $(m run_g0pp9c1b.sh)"
echo "== selftest v1y CPU: the layer (x and y), the build, the marks, the import, the fixture, the v9b suite in v9b's place $(date -u +%T)"
$P engine_twofield_v9c_selftest_v1y.py --v9b engine_twofield_v9b.py --v9c engine_twofield_v9c1b.py --fixture --v9b-selftest engine_twofield_v9b_selftest.py 2>&1 | grep -v UserWarning | grep -v "csr = lambda" | grep -v "^T/rs" | grep -v "^  v9c prim"; echo "rc ${PIPESTATUS[0]}"
echo "== the toy standalone on the built engine_twofield_v9c1b.py $(date -u +%T)"
$P toy_late_step_v1.py --engine engine_twofield_v9c1b.py 2>&1 | grep -E "^(PASS|FAIL|INFO|toy_late|y on)" | cut -c1-260; echo "rc ${PIPESTATUS[0]}"
E2E="--m 0.2 --g 0.25 --rs 32.0 --R0-over-rs 4 --reading A --nsites 1024 --prep-ell 4 --ell 4 --kappa-sea 10 --tau 2.0 --T-over-rs 0.1875 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 0 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --nk-jvp exact --delta-rtol-every 10 --commutator-every 1 --streams 0 --c-LZ 1.0 --window-max 4 --light-kappas 1,2 --light-guard-kappas 3 --light-coupling on --light-transport banded --light-d-min 1e-7 --s11-collars 8,16,24 --s11-stop 1 --field-kernel-cache --v36-fixes --slices-c64 '' --slices-light-c128 ''"
run() { local log=$1; local eng=$2; shift 2; local t0=$(date +%s); eval $P $eng $E2E "$@" > $log 2>&1; local rc=$?; echo "rc $rc in $(( $(date +%s) - t0 )) s ($log)"; return $rc; }
SLZ="--reference-solve lagged --cut-transport sector --anchor-passage lz --m1-simpson-steps ''"
SFL="--reference-solve lagged --cut-transport sector --m1-simpson-steps ''"
RI="--cut-transport record --m1-simpson-steps 1"
arm_run() {
  local arm=$1 eng
  case $arm in REF) eng=engine_twofield_v9c1.py;; B) eng=engine_twofield_v9c1b.py;; esac
  local d=$W/$arm; rm -rf $d; mkdir -p $d/sLz/states $d/res/states $d/ri $d/sfl
  echo "== $arm ($eng): sector lagged floor $(date -u +%T)"
  run $d/sfl/run.log $eng $SFL --slices-c128 "''" --out $d/sfl --tag e2e_v9sL
  echo "== $arm: sector lagged lz with a slice at 0.0625 $(date -u +%T)"
  run $d/sLz/run.log $eng $SLZ --slices-c128 0.0625 --slice-dir $d/sLz/states --out $d/sLz --tag e2e_v9cL
  local SL=$(ls $d/sLz/states/orbitals_e2e_v9cL_T*.npz | head -1)
  echo "== $arm: its resume from $(basename $SL) to 0.1875 $(date -u +%T)"
  run $d/res/run.log $eng $SLZ --slices-c128 "''" --slice-dir $d/res/states --out $d/res --tag e2e_v9cL --resume-slice $SL --resume-run $d/sLz/e2e_v9cL --segment-end-T-over-rs 0.1875
  echo "== $arm: record implicit with a Simpson step at step 1 $(date -u +%T)"
  run $d/ri/run.log $eng $RI --slices-c128 "''" --out $d/ri --tag e2e_v9ri
}
for arm in REF B; do arm_run $arm > $W/arm_$arm.log 2>&1 & done
wait
for arm in REF B; do cat $W/arm_$arm.log; done
echo "== compares (v9c1 = A, v9c1b = B) $(date -u +%T)"
$PY cmp_v9c2_runs.py $W/REF/sfl/e2e_v9sL $W/B/sfl/e2e_v9sL --args-equal 2>&1 | head -6
$PY cmp_v9c2_runs.py $W/REF/ri/e2e_v9ri $W/B/ri/e2e_v9ri --args-equal 2>&1 | head -6
for form in sLz/e2e_v9cL res/e2e_v9cL; do
  $PY engine_twofield_v9b_selftest.py compare-runs $W/REF/$form $W/B/$form --strip-sector | head -4
  $PY cmp_v9c2_runs.py $W/REF/$form $W/B/$form 2>&1 | head -6
done
$PY engine_twofield_v9b_selftest.py compare-runs $W/B/sfl/e2e_v9sL $W/B/sLz/e2e_v9cL --strip-sector | head -4
$PY - $W/B/sLz/collapse_e2e_v9cL.json <<'EOF'
import json, sys, collections
J = json.load(open(sys.argv[1])); a = J["args"]; recs = [r for r in J["records"] if r.get("E") is not None]
print("B lz run: prim_module %r anchor_passage_prim %r; records %d" % (a.get("prim_module"), a.get("anchor_passage_prim"), len(recs)))
t = collections.Counter()
for r in recs:
    for sp, chans in (("light", (r.get("light_v36") or {}).get("channels") or {}), ("heavy", r.get("continuation") or {})):
        for k, c in chans.items():
            P = (c.get("sector") or {}).get("passage") or {}
            t[(sp, P.get("prim"), "quiet" if P.get("quiet") else ("ran" if P.get("ran") else ("skipped" if P.get("skipped") else "none")))] += 1
print("passage blocks (species, prim, state):", dict(t))
EOF
echo "-- the resume against the straight run after 0.0625 (finding 8's strip list), per arm"
for arm in REF B; do
$PY - $W/$arm/sLz/e2e_v9cL $W/$arm/res/e2e_v9cL $arm <<'EOF'
import sys
sys.path.insert(0, "${QI2_ROOT}/v9c1b")
import engine_twofield_v8_selftest as S8
import engine_twofield_v9b_selftest as S9
common, diffs = S9.compare_runs_stripped(S8, sys.argv[1], sys.argv[2], {"engine_v34", "label"}, after=0.0625)
kept = [d for d in diffs if not str(d[0]).startswith("s11/r_nodes")]
print("%s %s: straight against resume over %d common records after 0.0625: %d differences%s" % ("PASS" if (common and not kept) else "FAIL", sys.argv[3], len(common), len(kept), (" %s" % kept[:4]) if kept else ""))
EOF
done
G=$W/B/refusals; rm -rf $G; mkdir -p $G/refused/states
SL=$(ls $W/B/sLz/states/orbitals_e2e_v9cL_T*.npz | head -1)
echo "== B: resume at an anchor-passage change (floor from the lz slice; expected refused) $(date -u +%T)"
run $G/refused/run.log engine_twofield_v9c1b.py --reference-solve lagged --cut-transport sector --anchor-passage floor --m1-simpson-steps "''" --slices-c128 "''" --slice-dir $G/refused/states --out $G/refused --tag e2e_v9cL --resume-slice $SL --resume-run $W/B/sLz/e2e_v9cL --segment-end-T-over-rs 0.1875
tail -1 $G/refused/run.log | cut -c1-300
echo "== B: lz with --cut-transport record (expected refused) $(date -u +%T)"
run $G/record_lz.log engine_twofield_v9c1b.py --cut-transport record --anchor-passage lz --m1-simpson-steps "''" --slices-c128 "''" --slice-dir $G --out $G --tag e2e_v9c_rec
tail -1 $G/record_lz.log | cut -c1-300
find $W -name "*.npz" -path "*states*" -delete
du -sh $W
echo "== done $(date -u +%T)"
