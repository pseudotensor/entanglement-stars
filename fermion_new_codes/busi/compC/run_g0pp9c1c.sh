#!/bin/bash
# G0'' of engine_twofield_v9c1c (REVIEWA_C64_toy_readout_v1 sec. 1 and 4: v9c1b with the default primitive assign_primitive_v1f, the import refusal naming it)
# on the box CPU, nice 10, 8 threads per run. Stage compC/v9c1c (a copy of compC/v9c1b's top-level files + build_engine_twofield_v9c1c.py staged as
# build_engine_twofield_v9c.py, v1f and its builder, the self-test v1z with the corrected x lines and the y lines y1 to y8 of toy_late_step_v2.py,
# cmp_v9c2_runs.py, cmp_v9c1c_fixture.py, count_anchor_three_sample_v1.py). Work and outputs on ${QI2_ROOT}/g0pp9c1c.
# Steps: the builds (v1f over v1e, v9c1c over v9b, v9c1c to v9c1b, v1z over v1y) with their reverses; the self-test v1z (layer with x and y, build,
# import, fixture, the v9b suite in v9b's place) on the CPU; the toy standalone; the stand-in fixture (1024 sites, 4 records) of v9c1c (C) in v9c1b's
# four forms compared with v9c1's runs of the v9c1b G0'' (REF, g0pp9c1b/REF: the same engine, args and box; v9c1b's own compare showed them bitwise
# reproducible), bitwise except the prim string; the two expected refusals on C; the section 1 reporting count on C's runs.
ST=${QI2_ROOT}/v9c1c
W=${QI2_ROOT}/g0pp9c1c
REF=${QI2_ROOT}/REF
cd $ST || exit 2
mkdir -p $W
export CUDA_VISIBLE_DEVICES="" BUSI_ENGINE_DIR=${QI2_ROOT}/compC_engine_v36 BUSI_PAGE_DIR=${QI2_ROOT}/page
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
PY=${QI2_ROOT}/python
P="nice -n 10 $PY"
m() { md5sum < $1 | cut -c1-32; }
echo "# G0'' v9c1c on the box CPU (stage $ST, work $W) $(date -u +%FT%TZ)"
echo "== builds $(date -u +%T)"
$P build_assign_primitive_v1f.py assign_primitive_v1e.py --check; echo "rc $?"
rm -f $W/v1f_restored.py; $P build_assign_primitive_v1f.py assign_primitive_v1f.py $W/v1f_restored.py --reverse --against assign_primitive_v1e.py; echo "rc $?"; rm -f $W/v1f_restored.py
$P build_engine_twofield_v9c.py engine_twofield_v9b.py --check | cut -c1-400; echo "rc ${PIPESTATUS[0]}"
rm -f $W/v9c1c_restored.py; $P build_engine_twofield_v9c.py engine_twofield_v9c1c.py $W/v9c1c_restored.py --reverse --against engine_twofield_v9b.py | cut -c1-300; echo "rc ${PIPESTATUS[0]}"; rm -f $W/v9c1c_restored.py
$P build_engine_twofield_v9c1c.py engine_twofield_v9c1c.py --to-v9c1b engine_twofield_v9c1b.py; echo "rc $?"
echo "-- v9c1b -> v9c1c: changed lines $(diff engine_twofield_v9c1b.py engine_twofield_v9c1c.py | grep -c '^>'), unmarked $(diff engine_twofield_v9c1b.py engine_twofield_v9c1c.py | grep '^>' | grep -vc 'v9c1c x')"
$P build_engine_twofield_v9c_selftest_v1z.py engine_twofield_v9c_selftest_v1y.py --check; echo "rc $?"
rm -f $W/v1z_restored.py; $P build_engine_twofield_v9c_selftest_v1z.py engine_twofield_v9c_selftest_v1z.py $W/v1z_restored.py --reverse --against engine_twofield_v9c_selftest_v1y.py; echo "rc $?"; rm -f $W/v1z_restored.py
echo "# md5 engine_twofield_v9b.py $(m engine_twofield_v9b.py); engine_twofield_v9c1.py $(m engine_twofield_v9c1.py); engine_twofield_v9c1b.py $(m engine_twofield_v9c1b.py); engine_twofield_v9c1c.py $(m engine_twofield_v9c1c.py); build_engine_twofield_v9c.py (= v9c1c's builder) $(m build_engine_twofield_v9c.py); build_engine_twofield_v9c1c.py $(m build_engine_twofield_v9c1c.py); assign_primitive_v1e.py $(m assign_primitive_v1e.py); assign_primitive_v1f.py $(m assign_primitive_v1f.py); build_assign_primitive_v1f.py $(m build_assign_primitive_v1f.py); engine_twofield_v9c_selftest_v1z.py $(m engine_twofield_v9c_selftest_v1z.py); build_engine_twofield_v9c_selftest_v1z.py $(m build_engine_twofield_v9c_selftest_v1z.py); toy_late_step_v1.py $(m toy_late_step_v1.py); toy_late_step_v2.py $(m toy_late_step_v2.py); cmp_v9c2_runs.py $(m cmp_v9c2_runs.py); cmp_v9c1c_fixture.py $(m cmp_v9c1c_fixture.py); count_anchor_three_sample_v1.py $(m count_anchor_three_sample_v1.py); run_g0pp9c1c.sh $(m run_g0pp9c1c.sh)"
echo "== selftest v1z CPU: the layer (x and y), the build, the marks, the import, the fixture, the v9b suite in v9b's place $(date -u +%T)"
$P engine_twofield_v9c_selftest_v1z.py --v9b engine_twofield_v9b.py --v9c engine_twofield_v9c1c.py --fixture --v9b-selftest engine_twofield_v9b_selftest.py 2>&1 | grep -v UserWarning | grep -v "csr = lambda" | grep -v "^T/rs" | grep -v "^  v9c prim"; echo "rc ${PIPESTATUS[0]}"
echo "== the toy standalone on the built engine_twofield_v9c1c.py $(date -u +%T)"
$P toy_late_step_v2.py --engine engine_twofield_v9c1c.py 2>&1 | grep -E "^(PASS|FAIL|toy_late|y on)" | cut -c1-260; echo "rc ${PIPESTATUS[0]}"
E2E="--m 0.2 --g 0.25 --rs 32.0 --R0-over-rs 4 --reading A --nsites 1024 --prep-ell 4 --ell 4 --kappa-sea 10 --tau 2.0 --T-over-rs 0.1875 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 0 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --nk-jvp exact --delta-rtol-every 10 --commutator-every 1 --streams 0 --c-LZ 1.0 --window-max 4 --light-kappas 1,2 --light-guard-kappas 3 --light-coupling on --light-transport banded --light-d-min 1e-7 --s11-collars 8,16,24 --s11-stop 1 --field-kernel-cache --v36-fixes --slices-c64 '' --slices-light-c128 ''"
run() { local log=$1; local eng=$2; shift 2; local t0=$(date +%s); eval $P $eng $E2E "$@" > $log 2>&1; local rc=$?; echo "rc $rc in $(( $(date +%s) - t0 )) s ($log)"; return $rc; }
SLZ="--reference-solve lagged --cut-transport sector --anchor-passage lz --m1-simpson-steps ''"
SFL="--reference-solve lagged --cut-transport sector --m1-simpson-steps ''"
RI="--cut-transport record --m1-simpson-steps 1"
d=$W/C; eng=engine_twofield_v9c1c.py; rm -rf $d; mkdir -p $d/sLz/states $d/res/states $d/ri $d/sfl
( echo "== C ($eng): sector lagged lz with a slice at 0.0625 $(date -u +%T)"
  run $d/sLz/run.log $eng $SLZ --slices-c128 0.0625 --slice-dir $d/sLz/states --out $d/sLz --tag e2e_v9cL
  SL=$(ls $d/sLz/states/orbitals_e2e_v9cL_T*.npz | head -1)
  echo "== C: its resume from $(basename $SL) to 0.1875 $(date -u +%T)"
  run $d/res/run.log $eng $SLZ --slices-c128 "''" --slice-dir $d/res/states --out $d/res --tag e2e_v9cL --resume-slice $SL --resume-run $d/sLz/e2e_v9cL --segment-end-T-over-rs 0.1875 ) > $W/arm_C1.log 2>&1 &
( echo "== C ($eng): sector lagged floor $(date -u +%T)"
  run $d/sfl/run.log $eng $SFL --slices-c128 "''" --out $d/sfl --tag e2e_v9sL
  echo "== C: record implicit with a Simpson step at step 1 $(date -u +%T)"
  run $d/ri/run.log $eng $RI --slices-c128 "''" --out $d/ri --tag e2e_v9ri ) > $W/arm_C2.log 2>&1 &
wait
cat $W/arm_C1.log $W/arm_C2.log
echo "== compares (v9c1 = REF from $REF, v9c1c = C) $(date -u +%T)"
for form in sfl/e2e_v9sL ri/e2e_v9ri sLz/e2e_v9cL res/e2e_v9cL; do
  $PY cmp_v9c2_runs.py $REF/$form $W/C/$form 2>&1 | cut -c1-400 | head -3
  $PY cmp_v9c1c_fixture.py $REF/$form $W/C/$form 2>&1 | cut -c1-600 | head -3
done
for form in sLz/e2e_v9cL res/e2e_v9cL; do
  $PY engine_twofield_v9b_selftest.py compare-runs $REF/$form $W/C/$form --strip-sector | head -4
done
$PY engine_twofield_v9b_selftest.py compare-runs $W/C/sfl/e2e_v9sL $W/C/sLz/e2e_v9cL --strip-sector | head -4
$PY - $W/C/sLz/collapse_e2e_v9cL.json <<'PYEOF'
import json, sys, collections
J = json.load(open(sys.argv[1])); a = J["args"]; recs = [r for r in J["records"] if r.get("E") is not None]
print("C lz run: prim_module %r anchor_passage_prim %r; records %d" % (a.get("prim_module"), a.get("anchor_passage_prim"), len(recs)))
t = collections.Counter()
for r in recs:
    for sp, chans in (("light", (r.get("light_v36") or {}).get("channels") or {}), ("heavy", r.get("continuation") or {})):
        for k, c in chans.items():
            P = (c.get("sector") or {}).get("passage") or {}
            t[(sp, P.get("prim"), "quiet" if P.get("quiet") else ("ran" if P.get("ran") else ("skipped" if P.get("skipped") else "none")))] += 1
print("passage blocks (species, prim, state):", dict(t))
PYEOF
echo "-- the resume against the straight run after 0.0625 (finding 8's strip list)"
$PY - $W/C/sLz/e2e_v9cL $W/C/res/e2e_v9cL C <<'PYEOF'
import sys
sys.path.insert(0, "${QI2_ROOT}/v9c1c")
import engine_twofield_v8_selftest as S8
import engine_twofield_v9b_selftest as S9
common, diffs = S9.compare_runs_stripped(S8, sys.argv[1], sys.argv[2], {"engine_v34", "label"}, after=0.0625)
kept = [d for d in diffs if not str(d[0]).startswith("s11/r_nodes")]
print("%s %s: straight against resume over %d common records after 0.0625: %d differences%s" % ("PASS" if (common and not kept) else "FAIL", sys.argv[3], len(common), len(kept), (" %s" % kept[:4]) if kept else ""))
PYEOF
echo "-- the section 1 reporting count (decided events with exactly 3 samples, one at s_a; bound: one before s_p) on C's runs"
$PY count_anchor_three_sample_v1.py $W/C/sLz/collapse_e2e_v9cL.json $W/C/res/collapse_e2e_v9cL.json $W/C/sfl/collapse_e2e_v9sL.json $W/C/ri/collapse_e2e_v9ri.json
G=$W/C/refusals; rm -rf $G; mkdir -p $G/refused/states
SL=$(ls $W/C/sLz/states/orbitals_e2e_v9cL_T*.npz | head -1)
echo "== C: resume at an anchor-passage change (floor from the lz slice; expected refused) $(date -u +%T)"
run $G/refused/run.log engine_twofield_v9c1c.py --reference-solve lagged --cut-transport sector --anchor-passage floor --m1-simpson-steps "''" --slices-c128 "''" --slice-dir $G/refused/states --out $G/refused --tag e2e_v9cL --resume-slice $SL --resume-run $W/C/sLz/e2e_v9cL --segment-end-T-over-rs 0.1875
tail -1 $G/refused/run.log | cut -c1-300
echo "== C: lz with --cut-transport record (expected refused) $(date -u +%T)"
run $G/record_lz.log engine_twofield_v9c1c.py --cut-transport record --anchor-passage lz --m1-simpson-steps "''" --slices-c128 "''" --slice-dir $G --out $G --tag e2e_v9c_rec
tail -1 $G/record_lz.log | cut -c1-300
find $W -name "*.npz" -path "*states*" -delete
du -sh $W
echo "== done $(date -u +%T)"
