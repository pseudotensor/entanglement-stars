#!/usr/bin/env python
"""stall_anatomy_v1.py (busi; ReviewA_v16e_v1.md sec. 2 (i)): for given (json, T/r_s) pairs, the record's Newton anatomy from the stored
json: per Newton step the held set, residual before, every trial (step, residual, set, crosses, flips), the accepted step, the plain-direction
fallback's trials and its JVP check (|G|_2, Newton slope, FD slope, exact slope, FD vs exact); the record's iteration_history (dE, dN, dv per
evaluation), v36c evaluation_sets (evaluation, round, set id, residual, gap), the record's field-route counts (fields_rd2), timing counts and
the gauge (negative centre, r_*). Usage: stall_anatomy_v1.py OUT.json JSON:T [JSON:T ...]"""
import json
import sys

out = {}
for spec in sys.argv[2:]:
    js, t = spec.rsplit(":", 1); t = float(t)
    d = json.load(open(js))
    r = [x for x in d["records"] if x.get("T_over_rs") is not None and abs(x["T_over_rs"] - t) < 1e-9][0]
    sv = r["solve_v32"]["solves"]
    print("=" * 100); print("%s  T/r_s %.4f  converged %s  iterations %s  eighs %s" % (js, t, r.get("converged"), r.get("iterations"), r.get("eigendecompositions")))
    print("  fields_rd2:", json.dumps(r.get("fields_rd2", {}).get("since_previous_record")))
    tm = r.get("timing") or {}
    print("  timing counts: solve_n %s eigh_n %s lagged %s fast %s midpoint %s fallbacks %s" % (tm.get("solve_n"), tm.get("eigh_n"), tm.get("lagged"), tm.get("fast"), tm.get("midpoint"), tm.get("fallbacks")))
    g = r.get("gauge") or {}
    print("  gauge: %s; zeros %s" % ((g.get("label") or "")[:60], g.get("zeros")))
    print("  jvp_v35: %s" % json.dumps(r.get("jvp_v35")))
    rec = {"converged": r.get("converged"), "steps": [], "history": r.get("iteration_history"), "fields_rd2": r.get("fields_rd2"), "gauge_zeros": g.get("zeros")}
    for s in sv:
        print("  solve %s: %s; evaluations %s kept %s (evaluation %s) confirming %s" % (s.get("name"), s.get("status"), s.get("evaluations"), s.get("kept_residual"), s.get("kept_evaluation"), s.get("confirming_residual")))
        for ls in s.get("line_search") or []:
            tr = ls.get("trials") or []
            print("    round %s step %s held %s r_before %.4e accepted %s; trials %s" % (ls.get("round"), ls.get("newton_step"), ls.get("held_set"), ls.get("residual_before"),
                  ls.get("accepted_step"), " ".join("[%g %.4e s%s%s%s]" % (q["step"], q["residual"], q["set"], " X" if q.get("crosses") else "", (" f%d" % q["flips_vs_held"]) if q.get("flips_vs_held") else "") for q in tr)))
            pf = ls.get("plain_fallback")
            if pf:
                print("      plain omega %s accepted %s; trials %s" % (pf.get("omega"), pf.get("accepted_step"), " ".join("[%g %.4e s%s]" % (q["step"], q["residual"], q["set"]) for q in pf.get("trials") or [])))
                print("      jvp_check %s" % json.dumps(pf.get("jvp_check")))
            rec["steps"].append(ls)
        es = (s.get("v36c") or {}).get("evaluation_sets")
        if es:
            print("    evaluation_sets (evaluation, round, set id, residual, gap):")
            for e in es:
                print("      %s" % e)
            rec["evaluation_sets"] = es
    out["%s:%s" % (js, t)] = rec
json.dump(out, open(sys.argv[1], "w"), indent=1)
