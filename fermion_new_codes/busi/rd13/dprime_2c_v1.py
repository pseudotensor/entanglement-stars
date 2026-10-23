#!/usr/bin/env python
"""dprime_2c_v1.py (busi, analyst, 21 Oct): ReviewA_rd1entry_v1 sec. 3 (alpha), erratum v3: the RD-1 entry replay's 8 records read by
ruling v2 2(c) with the flip evidence taken from every stored parent-route arm at the record and the perturbed parent arm D' (so the first
flip is 5.0625: the box arm D took 7 against 16 there, and D' at the shared count 16 stops on another iterate, kept residual 8.4e-8
against D's 3.2e-7). Reads the stored tolcmp_v2 outputs of the entry read (X = B, D, P); no lines move; tolcmp_v2's own functions
(recvals inputs, three_way, the C lines and the kept lines) are imported unchanged.
  dprime_2c_v1.py ENTRY_V2_DIR TOLCMP_V2_PY --first 5.06250 [--percall TEXT]
"""
import json, os, sys, argparse, importlib.util
ap = argparse.ArgumentParser(); ap.add_argument("d"); ap.add_argument("tol"); ap.add_argument("--first", default="5.06250"); ap.add_argument("--percall", default="")
a = ap.parse_args()
sp = importlib.util.spec_from_file_location("tolcmp_v2", a.tol); t = importlib.util.module_from_spec(sp); sp.loader.exec_module(t)
O = {k: json.load(open(os.path.join(a.d, "v2_%s.json" % k))) for k in ("X_vs_D", "X_vs_P", "D_vs_P")}
def vals(o):
    ts = [float(x) for x in o["per_record"]]
    R = t.per_record_rows(o, ts); V = {}
    for tk, p in o["per_record"].items():
        r = R.get(tk, {})
        V[tk] = {"E_over_E0": p.get("E_over_E0", 0.0), "E_rel": p.get("E_rel", 0.0), "roots": p.get("roots", 0.0), "depth": p.get("depth", 0.0),
                 "R_sh": p.get("R_sh", 0.0), "named": p.get("named", 0.0), "fields": r.get("fields", 0.0), "rows": r.get("rows", 0.0),
                 "spectra_rel": r.get("spectra_rel", 0.0), "nocc": r.get("nocc", 0.0), "decisions": p["n_decisions"]}
    return V
VXD, VXP, VDP = vals(O["X_vs_D"]), vals(O["X_vs_P"]), vals(O["D_vs_P"])
fails = []; worstC = {}; worst2 = {}
print("2(c) read, flip from %s on (erratum v3 evidence); kept lines X-D at every record; (1) X-P and D-P within C, 0 decisions; (2) X-P <= 2 D-P" % a.first)
for tk in sorted(VXD):
    v = VXD[tk]; post = tk >= a.first; line = []
    for c, lim in t.LINES_A2.items():
        ok = v[c] <= lim; line.append("%s %.2e%s" % (c, v[c], "" if ok else " FAIL")); ok or fails.append((tk, c))
    for c in ("R_sh", "decisions"):
        ok = v[c] == 0; line.append("%s %s%s" % (c, v[c], "" if ok else " FAIL")); ok or fails.append((tk, c))
    if post:
        for arm, VV in (("X", VXP), ("D", VDP)):
            for c, lim in t.C_LINES.items():
                ok = VV[tk][c] <= lim; worstC[(arm, c)] = max(worstC.get((arm, c), 0.0), VV[tk][c] / lim); ok or fails.append((tk, arm, c))
            ok = VV[tk]["decisions"] == 0; ok or fails.append((tk, arm, "decisions"))
        for c in t.FIELD_CLASSES:
            ok, r = t.three_way(VXP[tk][c], VDP[tk][c], None); worst2[c] = max(worst2.get(c, 0.0), r); ok or fails.append((tk, "(2)", c, r))
            line.append("(2) %s %.2f%s" % (c, r, "" if ok else " FAIL"))
        line.append("(1) X-P fields %.2e D-P %.2e" % (VXP[tk]["fields"], VDP[tk]["fields"]))
    print("  %s %s  %s" % (tk, "post-flip" if post else "pre-flip ", "  ".join(line)))
print("(1) largest fraction of C: " + ", ".join("%s-P %s %.3f (%.0fx inside)" % (k[0], k[1], x, 1 / x if x else float("inf")) for k, x in sorted(worstC.items())))
print("(2) largest X-P/D-P per class: " + ", ".join("%s %.2f" % kv for kv in worst2.items()))
print("(3) per-call: %s" % (a.percall or "OPEN"))
print("2(c) READ: %s" % ("PASS" if not fails and a.percall else ("FAIL %s" % fails if fails else "INCONCLUSIVE (3) open")))
