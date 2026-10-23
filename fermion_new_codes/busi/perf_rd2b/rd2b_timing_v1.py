#!/usr/bin/env python
"""rd2b_timing_v1.py (busi, analyst; PERF RD-2b self-test (c), reported part): per record of the replay arms, the record wall, the solve
count and seconds, the eigh seconds, and (for a collocation arm) the route counters of fields_rd2; the card factor = I's record wall over
C's. Also the I arm against a reference I arm (RD-2's on v16d, the same card): the records but walls (json), and the fields, spectra and
handoff npz byte for byte.
usage: rd2b_timing_v1.py --arm I=DIR:TAG --arm C=DIR:TAG [--ref-I DIR:TAG] [--rs 64] --out JSON"""
import argparse, json, os, sys
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--arm", action="append", required=True); ap.add_argument("--ref-I", default=None)
ap.add_argument("--rs", type=float, default=64.0); ap.add_argument("--out", required=True); ap.add_argument("--selftest", default=None)
a = ap.parse_args()
arms = {}
for s in a.arm:
    n, rest = s.split("=", 1); d, t = rest.split(":", 1)
    arms[n] = (d, t, json.load(open(os.path.join(d, "collapse_%s.json" % t))))
rep = {"records": {}}
I = arms["I"][2]["records"]; C = arms["C"][2]["records"]
ti = {r["T_over_rs"]: r for r in I}; tc = {r["T_over_rs"]: r for r in C}
tot = {"I": 0.0, "C": 0.0}
print("T/rs | I: record s, solves n, solve s | C: record s, solves n, solve s | C route: collocation, negative_centre, fallbacks, ivp by tolerance | factor per record, per solve")
for t in sorted(set(ti) & set(tc)):
    ri, rc = ti[t], tc[t]
    if t <= 2.0:
        continue
    fi = ri["timing"]; fc = rc["timing"]; c = (rc.get("fields_rd2") or {}).get("since_previous_record", {})
    row = {"I_s": ri["seconds"], "I_solve_n": fi["solve_n"], "I_solve_s": fi["solve_s"], "I_eigh_s": fi.get("eigh_s"),
           "C_s": rc["seconds"], "C_solve_n": fc["solve_n"], "C_solve_s": fc["solve_s"], "C_eigh_s": fc.get("eigh_s"),
           "collocation": c.get("collocation"), "negative_centre": c.get("negative_centre"), "fallback": c.get("fallback"), "ivp_by_tolerance": c.get("ivp_by_tolerance"),
           "factor_record": ri["seconds"] / rc["seconds"], "factor_solve": (fi["solve_s"] / fi["solve_n"]) / (fc["solve_s"] / fc["solve_n"])}
    tot["I"] += ri["seconds"]; tot["C"] += rc["seconds"]
    rep["records"][t] = row
    print("%.3f | %.1f, %d, %.1f | %.1f, %d, %.1f | %s, %s, %s, %s | %.2fx, %.1fx" % (t, row["I_s"], row["I_solve_n"], row["I_solve_s"], row["C_s"], row["C_solve_n"], row["C_solve_s"],
          row["collocation"], row["negative_centre"], row["fallback"], row["ivp_by_tolerance"], row["factor_record"], row["factor_solve"]))
rep["segment"] = {"I_s": tot["I"], "C_s": tot["C"], "factor": tot["I"] / tot["C"]}
print("segment: I %.1f s, C %.1f s, factor %.2fx" % (tot["I"], tot["C"], tot["I"] / tot["C"]))
if a.ref_I:
    d0, t0 = a.ref_I.split(":", 1)
    sys.path.insert(0, os.path.dirname(a.selftest)) if a.selftest else None
    import importlib.util
    spec = importlib.util.spec_from_file_location("st36z", a.selftest); ST = importlib.util.module_from_spec(spec); spec.loader.exec_module(ST)
    J0 = json.load(open(os.path.join(d0, "collapse_%s.json" % t0)))
    JI = arms["I"][2]
    def P(x):
        if isinstance(x, dict):
            return {k: P(v) for k, v in x.items()}
        if isinstance(x, list):
            return [P(v) for v in x]
        if isinstance(x, str) and "/workspace" in x:
            return "<path>"
        return x

    def S(rs):
        return [json.dumps(P(ST.strip(r)), sort_keys=True, default=float) for r in rs]
    nI = S(JI["records"]); n0 = S(J0["records"])
    same = [x == y for x, y in zip(n0, nI)] + ([False] if len(n0) != len(nI) else [])
    diffs = []
    for f in ("fields", "spectra", "handoff"):
        b = ST.same_npz(os.path.join(arms["I"][0], "collapse_%s_%s.npz" % (arms["I"][1], f)), os.path.join(d0, "collapse_%s_%s.npz" % (t0, f)))
        diffs.append("%s %s" % (f, b or "byte for byte"))
    rep["I_vs_ref_I"] = {"records_equal_but_walls": same, "npz": diffs}
    print("I (v16e default) against RD-2's I (v16d default), same card: records equal but walls and paths %s; %s" % (same, "; ".join(diffs)))
json.dump(rep, open(a.out, "w"), indent=1)
