#!/bin/bash
# G0'' rerun after C64_v9c_review_v1 (CLEAR WITH EDITS; E1-E5 on the builder's block and the self-test, L1-L21 on the launcher builder) on the box:
# the A6000 index 2 and the CPU. The same steps as run_g0pp9c.sh (C64_v9c_ruling_v1 2(c)) in the stage directory compC/v9c1 (a copy of compC's
# top-level files with the edited build_engine_twofield_v9c.py and engine_twofield_v9c_selftest.py), building engine_twofield_v9c1.py over
# engine_twofield_v9b.py (the current engine_twofield_v9c.py, ebb2c654, untouched); the self-test now carries x (finding 9) and F11/F11b (findings 2, 9;
# F11b alone may FAIL without holding the gates); the lz resume compared with the resume keys stripped (finding 8: engine_v34 and label on the records,
# s11/r_nodes on the npz, a declared strip list); the lz json's head (E1), the passage blocks' spectrum (E3) and extension (E2), n_spectra (finding 12);
# the heavy route's path reads against V9cEndPath (finding 2); v12c built with bash -n. Nothing of this touches a card beyond the box's A6000.
ST=${QI2_ROOT}/v9c1
cd $ST || exit 2
export CUDA_VISIBLE_DEVICES=2 BUSI_ENGINE_DIR=${QI2_ROOT}/compC_engine_v36 BUSI_PAGE_DIR=${QI2_ROOT}/page
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${QI2_ROOT}/python
P="nice -n 10 $PY"
ENG=engine_twofield_v9c1.py
echo "# G0'' v9c1 (C64_v9c_review_v1 edits E1-E5 applied; the stage $ST) $(date -u +%FT%TZ)"
echo "== build $(date -u +%T)"
$P build_engine_twofield_v9c.py engine_twofield_v9b.py $ENG; r0=$?; echo "rc $r0"
$P -m py_compile $ENG && echo "py_compile ok"
$P build_engine_twofield_v9c.py $ENG ${QI2_ROOT}/v9c1_restored.py --reverse --against engine_twofield_v9b.py; echo "rc $?"
m() { md5sum < $1 | cut -c1-32; }
echo "# md5 engine_twofield_v9b.py $(m engine_twofield_v9b.py); $ENG $(m $ENG); build_engine_twofield_v9c.py $(m build_engine_twofield_v9c.py); engine_twofield_v9c_selftest.py $(m engine_twofield_v9c_selftest.py); assign_primitive_v1d.py $(m assign_primitive_v1d.py); engine_twofield_v8.py $(m engine_twofield_v8.py); build_pod_launch_C_v12c.py $(m build_pod_launch_C_v12c.py); the untouched compC/engine_twofield_v9c.py $(m ../engine_twofield_v9c.py)"
echo "== finding 2: the heavy route's reads of its path (def heavy_sector_walk_v9 to the next def) against V9cEndPath's set {ch, eigh_n, eigvals_n, trial, fields, h_host, eig, eigvals}"
awk '/^def heavy_sector_walk_v9\(/{f=1; print NR": "$0; next} f&&/^def /{exit} f{print NR": "$0}' engine_twofield_v9b.py | grep 'path\.' | cut -c1-220
READS=$(awk '/^def heavy_sector_walk_v9\(/{f=1; next} f&&/^def /{exit} f' engine_twofield_v9b.py | grep -o '\bpath\.[A-Za-z_0-9]*' | sort -u | sed 's/path\.//' | tr '\n' ' ')
OUTSIDE=""; for r in $READS; do case $r in ch|eigh_n|eigvals_n|trial|fields|h_host|eig|eigvals) ;; *) OUTSIDE="$OUTSIDE $r";; esac; done
echo "heavy route reads: $READS"; if [ -z "$OUTSIDE" ]; then echo "PASS finding 2: every heavy path read is in V9cEndPath's set"; else echo "FAIL finding 2: reads outside V9cEndPath's set:$OUTSIDE (to be added to V9cEndPath before G4)"; fi
echo "== selftest CPU: the layer (with x), the build, the marks, the routes on the manufactured paths (with F11, F11b), the v9b suite in v9b's place $(date -u +%T)"
$P engine_twofield_v9c_selftest.py --v9b engine_twofield_v9b.py --v9c $ENG --fixture --v9b-selftest engine_twofield_v9b_selftest.py 2>&1 | grep -v UserWarning | grep -v "csr = lambda" | grep -v "^T/rs"; r1=${PIPESTATUS[0]}; echo "rc $r1"
echo "== selftest A6000: the routes on the manufactured paths (with F11, F11b) and the v9b suite in v9b's place $(date -u +%T)"
$P engine_twofield_v9c_selftest.py --skip-layer --v9b engine_twofield_v9b.py --v9c $ENG --fixture --gpu 0 --v9b-selftest engine_twofield_v9b_selftest.py 2>&1 | grep -v UserWarning | grep -v "csr = lambda" | grep -v "^T/rs"; r2=${PIPESTATUS[0]}; echo "rc $r2"
if [ $r0 -ne 0 ]; then echo "== the build is not clean: nothing further $(date -u +%T)"; exit 0; fi
if [ $r1 -ne 0 ] || [ $r2 -ne 0 ]; then echo "== NOTE: a self-test is not clean (F11b alone may FAIL without holding the gates, finding 9); the end-to-end runs continue for the record $(date -u +%T)"; fi
W=${QI2_ROOT}/v9c1_e2e; rm -rf $W
echo "== endtoend with v9c1 at the defaults (A6000, 1024 sites, 4 records) $(date -u +%T)"
$P engine_twofield_v9b_selftest.py endtoend --v8 engine_twofield_v8.py --v9 $ENG --gpu 0 --nsites 1024 --rs 32 --records 4 --work $W; r3=$?; echo "rc $r3"
echo "== the stand-in fixture bitwise against v9b at the defaults (the stored v9b runs of ${QI2_ROOT}/v9b2_e2e, every key) $(date -u +%T)"
for tag in e2e_v9ri e2e_v9rL e2e_v9sL; do $PY engine_twofield_v9b_selftest.py compare-runs ${QI2_ROOT}/$tag $W/$tag | head -4; done
$PY - $W <<'EOF'
import json, sys, os
W = sys.argv[1]
for tag in ("e2e_v9ri", "e2e_v9rL", "e2e_v9sL"):
    J = json.load(open(os.path.join(W, "collapse_%s.json" % tag)))
    a = J["args"]; recs = [r for r in J["records"] if r.get("E") is not None]
    print("%s: script %s; args anchor_passage %r prim_module %r lz_c %r prim_levels %r; anchor_passage_engine in args %s (floor sets nothing); records %d; a passage block in any sector entry: %s" % (
        tag, J.get("script"), a.get("anchor_passage"), a.get("prim_module"), a.get("lz_c"), a.get("prim_levels"), "anchor_passage_engine" in a, len(recs),
        any("passage" in (c.get("sector") or {}) for r in recs for c in list(((r.get("light_v36") or {}).get("channels") or {}).values()) + list((r.get("continuation") or {}).values()))))
EOF
G=$W/lz; rm -rf $G; mkdir -p $G/lz_st $G/lz_res/states $G/refused/states
E2E="--m 0.2 --g 0.25 --rs 32.0 --R0-over-rs 4 --reading A --nsites 1024 --prep-ell 4 --ell 4 --kappa-sea 10 --tau 2.0 --T-over-rs 0.1875 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 0 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --nk-jvp exact --delta-rtol-every 10 --commutator-every 1 --streams 0 --c-LZ 1.0 --window-max 4 --light-kappas 1,2 --light-guard-kappas 3 --light-coupling on --light-transport banded --light-d-min 1e-7 --s11-collars 8,16,24 --s11-stop 1 --field-kernel-cache --v36-fixes --slices-c64 '' --slices-light-c128 '' --gpu 0 --reference-solve lagged"
run() { local log=$1; shift; local t0=$(date +%s); eval $P $ENG $E2E --m1-simpson-steps "''" "$@" > $log 2>&1; local rc=$?; echo "rc $rc in $(( $(date +%s) - t0 )) s ($log)"; return $rc; }
echo "== lz straight: --cut-transport sector --anchor-passage lz with a slice at 0.0625 $(date -u +%T)"
run $G/lz_st/run.log --cut-transport sector --anchor-passage lz --slices-c128 0.0625 --slice-dir $G/lz_st --out $G/lz_st --tag e2e_v9cL
grep -m1 "v9c prim: --anchor-passage lz" $G/lz_st/run.log | cut -c1-300
grep "v9c prim light\|v9c prim heavy" $G/lz_st/run.log | head -5 | cut -c1-300
SL=$(ls $G/lz_st/orbitals_e2e_v9cL_T*.npz | head -1); echo "slice $SL"
$PY - $G/lz_st/collapse_e2e_v9cL.json $ST/$ENG <<'EOF'
import json, sys, importlib.util, os, collections
spec = importlib.util.spec_from_file_location("engine_twofield_v9c", sys.argv[2])
V9 = importlib.util.module_from_spec(spec); sys.modules["engine_twofield_v9c"] = V9; spec.loader.exec_module(V9)
J = json.load(open(sys.argv[1])); a = J["args"]; recs = [r for r in J["records"] if r.get("E") is not None]
print("lz run: script %s; args anchor_passage %r anchor_passage_engine %r anchor_passage_prim %r prim_module %r lz_c %r lz_k %r prim_levels %r prim_w_mix %r lz_skip_gap %r; records %d" % (J.get("script"), a.get("anchor_passage"), a.get("anchor_passage_engine"), a.get("anchor_passage_prim"), a.get("prim_module"), a.get("lz_c"), a.get("lz_k"), a.get("prim_levels"), a.get("prim_w_mix"), a.get("lz_skip_gap"), len(recs)))
e1 = a.get("anchor_passage_engine") == "engine_twofield_v9c" and str(a.get("anchor_passage_prim") or "").startswith("assign_primitive")
print("%s E1: the lz json's args name the engine and the primitive (%r, %r)" % ("PASS" if e1 else "FAIL", a.get("anchor_passage_engine"), a.get("anchor_passage_prim")))
tally = collections.Counter(); holds = []; nspec = collections.Counter(); ext = collections.Counter(); spect = collections.Counter(); blocks = 0; spec_ok = True
for r in recs:
    t = r["T_over_rs"]
    for sp, chans in (("light", (r.get("light_v36") or {}).get("channels") or {}), ("heavy", r.get("continuation") or {})):
        for k, c in chans.items():
            S = c.get("sector") or {}; P = S.get("passage") or {}
            state = "none" if not P else ("fallback" if P.get("fallback") else ("skipped" if P.get("skipped") else ("quiet" if P.get("quiet") else ("ran" if P.get("ran") else "?"))))
            tally[(sp, S.get("route"), state, P.get("prim"))] += 1
            if P and not P.get("fallback"):
                nspec[(sp, state, P.get("n_spectra"))] += 1
                if P.get("ran") or P.get("quiet") or P.get("skipped"):
                    blocks += 1; ext[(sp, repr(P.get("extension", "(key absent)")))] += 1; spect[(sp, P.get("spectrum", "(key absent)")[:60])] += 1
                    if "spectrum" not in P: spec_ok = False
            hr = V9.v9c_hold_reasons(S)
            if hr: holds.append((t, sp, k, [h[0][:80] for h in hr]))
print("per (species, route, passage state, primitive version):"); [print("  ", k, v) for k, v in sorted(tally.items(), key=str)]
print("finding 12, n_spectra per channel-record (species, state, n_spectra): count:"); [print("  ", k, v) for k, v in sorted(nspec.items(), key=str)]
print("E2 extension per passage block (species, extension): count:", dict(ext))
print("E3 spectrum per passage block (species, spectrum[:60]): count:", dict(spect))
print("%s E2/E3: every passage block that ran, was quiet or skipped carries spectrum (%d blocks)" % ("PASS" if (spec_ok and blocks > 0) else ("FAIL" if blocks > 0 else "INFO no block reached the decision"), blocks))
print("v9c_hold_reasons on every channel of every record:", holds if holds else "none")
EOF
echo "== lz against the floor sector run (the sector keys stripped) $(date -u +%T)"
$PY engine_twofield_v9b_selftest.py compare-runs $W/e2e_v9sL $G/lz_st/e2e_v9cL --strip-sector | head -4
echo "== resume on lz from that slice to 0.1875 $(date -u +%T)"
run $G/lz_res/run.log --cut-transport sector --anchor-passage lz --slices-c128 "''" --slice-dir $G/lz_res/states --out $G/lz_res --tag e2e_v9cL --resume-slice $SL --resume-run $G/lz_st/e2e_v9cL --segment-end-T-over-rs 0.1875
echo "-- finding 8: the straight lz run against the resume after 0.0625 with the resume's own marks stripped (declared strip list: records engine_v34, label; npz s11/r_nodes)"
$PY - $G/lz_st/e2e_v9cL $G/lz_res/e2e_v9cL <<'EOF'
import sys
sys.path.insert(0, "${QI2_ROOT}/v9c1")
import engine_twofield_v8_selftest as S8
import engine_twofield_v9b_selftest as S9
RESUME_KEYS = {"engine_v34", "label"}          # the records' resume marks (the restart sentence of the label, the engine_v34 restart record)
RESUME_NPZ = ("s11/r_nodes",)                  # the s11 nodes: the resume's own run carries its records only
common, diffs = S9.compare_runs_stripped(S8, sys.argv[1], sys.argv[2], RESUME_KEYS, after=0.0625)
kept = [d for d in diffs if not any(str(d[0]).startswith(k) for k in RESUME_NPZ)]
dropped = [d for d in diffs if d not in kept]
big = 0.0
for d in kept:
    try: big = max(big, float(d[2]))
    except Exception: pass
print("%s compare-runs (resume keys stripped: records %s, npz %s) %s against %s over %d common records after T/rs 0.0625: %d differences%s; stripped npz differences %s" % (
    "PASS" if (not kept and len(common) > 0) else "FAIL", sorted(RESUME_KEYS), list(RESUME_NPZ), sys.argv[1], sys.argv[2], len(common), len(kept), (" %s" % kept[:5]) if kept else "", dropped))
EOF
echo "== resume at an anchor-passage change (floor from the lz run's slice; expected refused) $(date -u +%T)"
run $G/refused/run.log --cut-transport sector --anchor-passage floor --slices-c128 "''" --slice-dir $G/refused/states --out $G/refused --tag e2e_v9cL --resume-slice $SL --resume-run $G/lz_st/e2e_v9cL --segment-end-T-over-rs 0.1875
tail -1 $G/refused/run.log | cut -c1-400
echo "== lz with --cut-transport record (expected refused at the start) $(date -u +%T)"
run $G/record_lz.log --cut-transport record --anchor-passage lz --slices-c128 "''" --slice-dir $G --out $G --tag e2e_v9c_rec
tail -1 $G/record_lz.log | cut -c1-300
echo "== v12c: build_pod_launch_C_v12c.py over pod_launch_C_v11.sh, bash -n $(date -u +%T)"
$PY build_pod_launch_C_v12c.py --v11 pod_launch_C_v11.sh --out pod_launch_C_v12c.sh; echo "rc $?"
bash -n pod_launch_C_v12c.sh && echo "PASS bash -n pod_launch_C_v12c.sh"
echo "# md5 pod_launch_C_v12c.sh $(m pod_launch_C_v12c.sh); lines $(wc -l < pod_launch_C_v12c.sh); engine_twofield_v9c.py named $(grep -c 'engine_twofield_v9c.py' pod_launch_C_v12c.sh) times, engine_twofield_v9b.py on launch lines $(grep -c 'python \$B/code/engine_twofield_v9b.py' pod_launch_C_v12c.sh), --anchor-passage $(grep -c -- '--anchor-passage' pod_launch_C_v12c.sh), prim_check calls $(grep -c '^ *prim_check   # v12c' pod_launch_C_v12c.sh)"
echo "== done $(date -u +%T)"
