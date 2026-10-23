#!/usr/bin/env python
"""dprime_hist_v1.py (busi, analyst, 21 Oct): ReviewA_rd1entry_v1 sec. 3, the mechanism record shown with the D' read (no card time):
the stored per-iteration residual histories and step/trial counts of the arms at one record, and the evaluation at which each pair of
arms first differs. Reads only the stored jsons.
  dprime_hist_v1.py --T 5.0625 NAME=path/to/collapse.json [NAME=...]   (pairs reported in the given order: 1-2, 2-3, ...; plus 1-3)
Per arm: iterations, kept residual and evaluation, confirming residual, and for each Newton step of each solve: residual_before, the
number of line-search trials, the accepted step (None = no trial accepted), the plain fallback's trials and accepted step, and the
iteration_history row of each evaluation ([|dE/E|, max|dN|, max|dv|]).
Per pair: the first evaluation index at which the iteration_history rows are not bitwise equal, and the relative difference of
max|dN| at each evaluation.
"""
import json, sys, argparse

ap = argparse.ArgumentParser(); ap.add_argument("--T", type=float, required=True); ap.add_argument("arms", nargs="+")
a = ap.parse_args()
R = {}
for s in a.arms:
    nm, p = s.split("=", 1)
    d = json.load(open(p))
    r = [x for x in d["records"] if abs(x.get("T_over_rs", -1) - a.T) < 1e-9]
    if not r:
        print("%s: no record at %.5f" % (nm, a.T)); continue
    R[nm] = r[-1]
names = list(R)
for nm in names:
    r = R[nm]; s = r["solve_v32"]
    print("== %s at T/rs %.5f: iterations %s, converged %s, kept_from %s, kept_residual %.6e (evaluation %s), confirming %.6e, evaluations %s, meets_stop_rule %s" % (
        nm, a.T, r.get("iterations"), r.get("converged"), s.get("kept_from"), s["kept_residual"], s.get("kept_evaluation"), s["confirming_residual"], s.get("evaluations"), s.get("meets_stop_rule")))
    for q in s["solves"]:
        print("   solve %s: status %s; iteration_history_slice %s" % (q["name"], q["status"], q.get("iteration_history_slice")))
        for ls in q.get("line_search", []):
            pf = ls.get("plain_fallback")
            pfs = ""
            if pf:
                pfs = "; plain fallback: %d trials, accepted %s, residual %s" % (len(pf.get("trials", [])), pf.get("accepted_step"),
                      ("%.6e" % pf["trials"][-1]["residual"]) if pf.get("trials") else None)
            tr = ls.get("trials", [])
            print("     step %2d: residual_before %.6e; %d trials (last residual %.6e), accepted %s%s" % (
                ls["newton_step"], ls["residual_before"], len(tr), tr[-1]["residual"] if tr else float("nan"), ls.get("accepted_step"), pfs))
    print("   iteration_history ([|dE/E|, max|dN|, max|dv|] per evaluation):")
    for i, h in enumerate(r["iteration_history"]):
        print("     %2d  %s" % (i, "  ".join("None" if x is None else "%.6e" % x for x in h)))
pairs = [(names[i], names[i + 1]) for i in range(len(names) - 1)] + ([(names[0], names[-1])] if len(names) > 2 else [])
for A, B in pairs:
    ha, hb = R[A]["iteration_history"], R[B]["iteration_history"]
    first = None
    for i in range(min(len(ha), len(hb))):
        if ha[i] != hb[i]:
            first = i; break
    rel = []
    for i in range(min(len(ha), len(hb))):
        x, y = ha[i][1], hb[i][1]
        rel.append(abs(x - y) / max(abs(x), abs(y), 1e-300))
    print("== %s against %s: %d and %d evaluations; first evaluation not bitwise equal: %s" % (A, B, len(ha), len(hb), first))
    print("   rel. difference of max|dN| per evaluation: %s" % " ".join("%.1e" % x for x in rel))
