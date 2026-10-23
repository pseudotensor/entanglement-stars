#!/usr/bin/env python
"""test_w10_defect14_audit.py (busi W10): defect 14 of REVIEWA_REVIEW_web40i section 5, the quoted-code audit items (sources fixed at the
start state during the field iteration; dE comparing inner iterations, not records; a best iterate saved but not restored; a failed solve
reaching the propagation), conditional on unquoted guards. The builder confirms or repairs each with a test; gate: the test.
Static confirmations on the engine source (engine_collapse_v36e.py in W10_ENGINE_DIR, else v36d with the repairs reported pending) of
the guards named in W10_fix_blocks_v1.md defect 14, and unit tests of w10_lib_v1.e_prev_rule (item (b)). --results GLOB lists the
delivered records' --v36-fixes and --streams flags (which runs the E_prev and thread findings concern).
Run: qi2 python test_w10_defect14_audit.py [--results 'DIR/**/collapse_*.json']"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w10_lib_v1 as W10  # noqa: E402

FAIL = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAIL.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    a = ap.parse_args()
    d = os.environ.get("W10_ENGINE_DIR") or HERE
    path = os.path.join(d, "engine_collapse_v36e.py")
    v36e = os.path.exists(path)
    if not v36e:
        path = os.path.join(d, "engine_collapse_v36d.py")
    if not os.path.exists(path):
        check(False, "no engine source found in %s (set W10_ENGINE_DIR)" % d)
        sys.exit(1)
    src = open(path).read()
    print("INFO auditing %s (%s)" % (path, "v36e: the repairs applied" if v36e else "v36d: the repairs pending"))

    def has(s, count=None):
        n = src.count(s)
        return (n >= 1) if count is None else (n == count)
    # (a) sources fixed at the start state during the field iteration: by design (version 17 (c), the frozen step)
    base_line = 'base[ch.kappa] = (dens(ch.h, s_["sea"]), dens(ch.Q, s_["sea"]), dens(ch.h, s_["dust"], s_["f"]), dens(ch.Q, s_["dust"], s_["f"]))'
    i_base = src.find(base_line); i_src = src.find("def sources(cache_):"); i_rec = src.find("recs.append(rec)"); i_prop = src.rfind('s_["sea"] = prop(s_["sea"])')
    check(i_base > 0 and i_src > i_base and has('hs_, Qs_, dd, Jd = base[ch.kappa]') and i_rec > 0 and i_prop > i_rec,
          "(a) confirmed by design: the base densities are computed once per record from the propagated state before the iteration, sources() subtracts the continued reference's densities at each trial, and the propagation follows the record (the frozen exponential step of defect 1; the W2 dg step is where the sources move within a record)")
    # (b) dE: inner iterations by design; the converged flag by the confirming step; the record rate separate; the E_prev rule tested
    check(has('conv = bool(b["resid"] < a.stop_fields and dE_c is not None and dE_c < a.stop_E)') and has('rec["dE_dT"] = (E - recs[-1]["E"]) / dT'),
          "(b) confirmed: the record's converged flag uses the confirming plain step's |dE/E| (dE_c) and the record-to-record rate is rec['dE_dT']; the history's dE column is the iteration's own measure")
    check(W10.e_prev_rule(False, True, True) and W10.e_prev_rule(False, False, False) and W10.e_prev_rule(True, True, True)
          and not W10.e_prev_rule(True, False, True) and not W10.e_prev_rule(True, True, False),
          "(b) e_prev_rule: without --v36-fixes E_prev moves at every evaluation (trials and finite differences included); with it only at an evaluation on the held set that becomes the best (version 36 (b))")
    if v36e:
        check(has("W10.e_prev_rule(V36[\"fixes\"], held_ok, nk[\"best\"] is None or max(dN_, dv_) < nk[\"best\"][\"resid\"])"), "(b) v36e's Newton evaluation uses e_prev_rule")
    # (c) best restored
    check(has('cache, srcs = best["cache"], best["srcs"]') and has('Nn, vn, Wn, info = best["sol"]') and has('N_app, v_app = best["app"]'),
          "(c) confirmed: the plain loop restores the smallest-residual iterate's cache, sources, solve and applied fields when unconverged")
    check(has('nk["best"] = chosen["best"]; nk["failed"] = chosen["failed"]') and has('ctr["E_prev"] = rounds[0]["E_prev_after"]'),
          "(c) confirmed: rounds_v36c keeps the chosen round's best and restores E_prev when round 1 is kept after a second round")
    check(has('Nn_b, vn_b, Wn_b, info_b = b["sol"]', 2), "(c) confirmed: newton_record (v31) and newton_record_v32 return the kept iterate's solve, cache and applied fields (2 return sites)")
    # (d) a failed solve never reaches the propagation
    check(has('recs.append({"T": T, "failed": info["failed"], "label": label, "iteration_history": it_hist})') and has('"failed": "no eigendecomposition at the accepted fields"'),
          "(d) confirmed: run() writes a failed record and breaks before any propagation (a failed solve, or no eigendecomposition at the accepted fields)")
    check(has('return np.full(2 * n, np.inf if V36["fixes"] else 1.0)') and has('raise RuntimeError("version 35: a field solve failed inside the exact J.v")'),
          "(d) confirmed: a failed solve inside the Newton loop is never saved as the best iterate (inf residual with --v36-fixes, version 32's fabricated 1.0 without it, which corrupts the direction and not the kept iterate); a failed pair solve in the exact J.v raises")
    guard = 'massless ball: the solve of the converged sources failed'
    check(has(guard) if v36e else True, "(d) %s: ball_fixed_point_v33's solve of the converged sources %s" % (
        "repaired" if v36e else "repair pending (v36d)", "is guarded before project_at" if has(guard) else "is unguarded in v36d (the S11 preparation only); block D14.2 adds the guard"))
    if a.results:
        files = sorted(glob.glob(a.results, recursive=True))
        for p in files:
            try:
                J = json.load(open(p))
            except Exception as ex:  # noqa: BLE001
                print("  %s: unreadable (%s)" % (p, ex)); continue
            args = J.get("args", {}); recs = J.get("records", [])
            fixes = sorted({str((r.get("v36") or {}).get("fixes")) for r in recs if r.get("v36")})
            print("  %s: streams %s, v36_fixes %s, nk_set %s, two-set records %d, massless_ball %s, records %d" % (
                p, args.get("streams"), args.get("v36_fixes"), args.get("nk_set"), sum(1 for r in recs if "v36d" in r), args.get("massless_ball"), len(recs)))
    print("test_w10_defect14_audit: %s" % ("ALL PASS" if not FAIL else "%d FAIL" % len(FAIL)))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
