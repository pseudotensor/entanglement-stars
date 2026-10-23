#!/bin/bash
# G0'' (C64_v9c_ruling_v1 2(c); RUN_C64_fix_v5b section 1's form) on the box: the A6000 index 2 and the CPU. The build of engine_twofield_v9c.py over
# engine_twofield_v9b.py (never overwritten), py_compile, the reverse byte for byte; the v9c self-test on the CPU (the layer, the build, the marks, the
# real routes on the manufactured paths, the v9b suite with v9c in v9b's place) and on the A6000 (the routes and the suite); the stand-in fixture end to
# end with v9c at the defaults (floor), compared bitwise against the stored v9b e2e runs (${QI2_ROOT}/v9b2_e2e); an lz run on the stand-in with a resume on lz
# and a resume at an anchor-passage change (refused). Nothing of this touches a card beyond the box's A6000.
cd ${QI2_ROOT}/compC
export CUDA_VISIBLE_DEVICES=2 BUSI_ENGINE_DIR=${QI2_ROOT}/compC_engine_v36 BUSI_PAGE_DIR=${QI2_ROOT}/page
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${QI2_ROOT}/python
P="nice -n 10 $PY"
echo "# G0'' v9c (build_engine_twofield_v9c.py rewritten against the built v9b; C64_v9c_ruling_v1 2(c)) $(date -u +%FT%TZ)"
echo "== build $(date -u +%T)"
$P build_engine_twofield_v9c.py engine_twofield_v9b.py engine_twofield_v9c.py; r0=$?; echo "rc $r0"
$P -m py_compile engine_twofield_v9c.py && echo "py_compile ok"
$P build_engine_twofield_v9c.py engine_twofield_v9c.py ${QI2_ROOT}/v9c_restored.py --reverse --against engine_twofield_v9b.py; echo "rc $?"
echo "# md5 engine_twofield_v9b.py $(md5sum < engine_twofield_v9b.py | cut -c1-32); engine_twofield_v9c.py $(md5sum < engine_twofield_v9c.py | cut -c1-32); build_engine_twofield_v9c.py $(md5sum < build_engine_twofield_v9c.py | cut -c1-32); engine_twofield_v9c_selftest.py $(md5sum < engine_twofield_v9c_selftest.py | cut -c1-32); assign_primitive_v1d.py $(md5sum < assign_primitive_v1d.py | cut -c1-32); engine_twofield_v8.py $(md5sum < engine_twofield_v8.py | cut -c1-32)"
echo "== selftest CPU: the layer, the build, the marks, the routes on the manufactured paths, the v9b suite in v9b's place $(date -u +%T)"
$P engine_twofield_v9c_selftest.py --v9b engine_twofield_v9b.py --v9c engine_twofield_v9c.py --fixture --v9b-selftest engine_twofield_v9b_selftest.py 2>&1 | grep -v UserWarning | grep -v "csr = lambda" | grep -v "^T/rs"; r1=${PIPESTATUS[0]}; echo "rc $r1"
echo "== selftest A6000: the routes on the manufactured paths and the v9b suite in v9b's place $(date -u +%T)"
$P engine_twofield_v9c_selftest.py --skip-layer --v9b engine_twofield_v9b.py --v9c engine_twofield_v9c.py --fixture --gpu 0 --v9b-selftest engine_twofield_v9b_selftest.py 2>&1 | grep -v UserWarning | grep -v "csr = lambda" | grep -v "^T/rs"; r2=${PIPESTATUS[0]}; echo "rc $r2"
if [ $r0 -ne 0 ] || [ $r1 -ne 0 ] || [ $r2 -ne 0 ]; then echo "== the self-test is not clean: endtoend and the lz run not run $(date -u +%T)"; exit 0; fi
W=${QI2_ROOT}/v9c_e2e; rm -rf $W
echo "== endtoend with v9c at the defaults (A6000, 1024 sites, 4 records) $(date -u +%T)"
$P engine_twofield_v9b_selftest.py endtoend --v8 engine_twofield_v8.py --v9 engine_twofield_v9c.py --gpu 0 --nsites 1024 --rs 32 --records 4 --work $W; r3=$?; echo "rc $r3"
echo "== the stand-in fixture bitwise against v9b at the defaults (the stored v9b runs of ${QI2_ROOT}/v9b2_e2e, every key) $(date -u +%T)"
for tag in e2e_v9ri e2e_v9rL e2e_v9sL; do $PY engine_twofield_v9b_selftest.py compare-runs ${QI2_ROOT}/$tag $W/$tag | head -4; done
$PY - $W <<'EOF'
import json, sys, os
W = sys.argv[1]
for tag in ("e2e_v9ri", "e2e_v9rL", "e2e_v9sL"):
    J = json.load(open(os.path.join(W, "collapse_%s.json" % tag)))
    a = J["args"]; recs = [r for r in J["records"] if r.get("E") is not None]
    print("%s: script %s; args anchor_passage %r prim_module %r lz_c %r prim_levels %r; records %d; a passage block in any sector entry: %s" % (
        tag, J.get("script"), a.get("anchor_passage"), a.get("prim_module"), a.get("lz_c"), a.get("prim_levels"), len(recs),
        any("passage" in (c.get("sector") or {}) for r in recs for c in list(((r.get("light_v36") or {}).get("channels") or {}).values()) + list((r.get("continuation") or {}).values()))))
EOF
# the lz run on the stand-in (sector lagged, --anchor-passage lz) with a slice at 0.0625, then the resume on lz and the refused resume at a change
G=$W/lz; rm -rf $G; mkdir -p $G/lz_st $G/lz_res/states $G/refused/states
E2E="--m 0.2 --g 0.25 --rs 32.0 --R0-over-rs 4 --reading A --nsites 1024 --prep-ell 4 --ell 4 --kappa-sea 10 --tau 2.0 --T-over-rs 0.1875 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 0 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --nk-jvp exact --delta-rtol-every 10 --commutator-every 1 --streams 0 --c-LZ 1.0 --window-max 4 --light-kappas 1,2 --light-guard-kappas 3 --light-coupling on --light-transport banded --light-d-min 1e-7 --s11-collars 8,16,24 --s11-stop 1 --field-kernel-cache --v36-fixes --slices-c64 '' --slices-light-c128 '' --gpu 0 --reference-solve lagged"
run() { local log=$1; shift; local t0=$(date +%s); eval $P engine_twofield_v9c.py $E2E --m1-simpson-steps "''" "$@" > $log 2>&1; local rc=$?; echo "rc $rc in $(( $(date +%s) - t0 )) s ($log)"; return $rc; }
echo "== lz straight: --cut-transport sector --anchor-passage lz with a slice at 0.0625 $(date -u +%T)"
run $G/lz_st/run.log --cut-transport sector --anchor-passage lz --slices-c128 0.0625 --slice-dir $G/lz_st --out $G/lz_st --tag e2e_v9cL
grep -m1 "v9c prim: --anchor-passage lz" $G/lz_st/run.log | cut -c1-300
grep "v9c prim light\|v9c prim heavy" $G/lz_st/run.log | head -5 | cut -c1-300
SL=$(ls $G/lz_st/orbitals_e2e_v9cL_T*.npz | head -1); echo "slice $SL"
$PY - $G/lz_st/collapse_e2e_v9cL.json <<'EOF'
import json, sys, importlib.util, os, collections
spec = importlib.util.spec_from_file_location("engine_twofield_v9c", "${QI2_ROOT}/engine_twofield_v9c.py")
V9 = importlib.util.module_from_spec(spec); sys.modules["engine_twofield_v9c"] = V9; spec.loader.exec_module(V9)
J = json.load(open(sys.argv[1])); a = J["args"]; recs = [r for r in J["records"] if r.get("E") is not None]
print("lz run: args anchor_passage %r prim_module %r lz_c %r lz_k %r prim_levels %r prim_w_mix %r lz_skip_gap %r; records %d" % (a.get("anchor_passage"), a.get("prim_module"), a.get("lz_c"), a.get("lz_k"), a.get("prim_levels"), a.get("prim_w_mix"), a.get("lz_skip_gap"), len(recs)))
tally = collections.Counter(); holds = []
for r in recs:
    t = r["T_over_rs"]
    for sp, chans in (("light", (r.get("light_v36") or {}).get("channels") or {}), ("heavy", r.get("continuation") or {})):
        for k, c in chans.items():
            S = c.get("sector") or {}; P = S.get("passage") or {}
            state = "none" if not P else ("fallback" if P.get("fallback") else ("skipped" if P.get("skipped") else ("quiet" if P.get("quiet") else ("ran" if P.get("ran") else "?"))))
            tally[(sp, S.get("route"), state, P.get("prim"))] += 1
            hr = V9.v9c_hold_reasons(S)
            if hr: holds.append((t, sp, k, [h[0][:80] for h in hr]))
print("per (species, route, passage state, primitive version):"); [print("  ", k, v) for k, v in sorted(tally.items(), key=str)]
print("v9c_hold_reasons on every channel of every record:", holds if holds else "none")
EOF
echo "== lz against the floor sector run (the sector keys stripped) $(date -u +%T)"
$PY engine_twofield_v9b_selftest.py compare-runs $W/e2e_v9sL $G/lz_st/e2e_v9cL --strip-sector | head -4
echo "== resume on lz from that slice to 0.1875 $(date -u +%T)"
run $G/lz_res/run.log --cut-transport sector --anchor-passage lz --slices-c128 "''" --slice-dir $G/lz_res/states --out $G/lz_res --tag e2e_v9cL --resume-slice $SL --resume-run $G/lz_st/e2e_v9cL --segment-end-T-over-rs 0.1875
$PY engine_twofield_v9b_selftest.py compare-runs $G/lz_st/e2e_v9cL $G/lz_res/e2e_v9cL --after 0.0625 | head -8
echo "== resume at an anchor-passage change (floor from the lz run's slice; expected refused) $(date -u +%T)"
run $G/refused/run.log --cut-transport sector --anchor-passage floor --slices-c128 "''" --slice-dir $G/refused/states --out $G/refused --tag e2e_v9cL --resume-slice $SL --resume-run $G/lz_st/e2e_v9cL --segment-end-T-over-rs 0.1875
tail -2 $G/refused/run.log | cut -c1-400
echo "== lz with --cut-transport record (expected refused at the start) $(date -u +%T)"
run $G/record_lz.log --cut-transport record --anchor-passage lz --slices-c128 "''" --slice-dir $G --out $G --tag e2e_v9c_rec
tail -1 $G/record_lz.log | cut -c1-300
echo "== done $(date -u +%T)"
