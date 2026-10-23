#!/usr/bin/env python
"""Strand 7 S7a part 2 report (SPEC_strand7_collapse.md sec. 10.2 (b), (e), (f); builder analyst).  Reads the S7a JSONs and writes markdown:
  (b) C3's ladder at E_0 = 16, self: dt = 0.01, 0.005, 0.0025 r_s (--ladder base,half,quarter): at the throat crossing r_sh = A (the head's
      record, A interpolated in T from the records) T, A/A(0) - 1, B/A, N(r_sh), c_+^(r); the differences between rungs and the observed order
      p = log2(|X(dt) - X(dt/2)| / |X(dt/2) - X(dt/4)|); the discrete energy balance over each run: the largest |sum nu H/sum nu H(0) - 1| and the
      largest |E_bdy/E_bdy(0) - 1| (= |A/A(0) - 1|) before the crossing and over the run;
  (e) the exterior moved (--rmax pair base:moved): A, B, B/A and r_sh at the common outputs every 5 r_s (their largest differences before and
      after the crossing) and the two crossing records;
  (f) record checks: per run the outputs whose outer loop did not converge, the largest lagged-source residual; C1's static-check records
      (resolve_B_rel and resolve_A_rel at z_c = 0.3 and the maximum, --static); C2's energy error of the run's own stepper against DOP853
      (--frozen-pair dop853:verlet: the largest |sum nu H rel|)."""
import argparse
import json
import math

import numpy as np


def f(x, d=5):
    if x is None:
        return "-"
    x = float(x)
    if not math.isfinite(x):
        return str(x)
    if x == 0:
        return "0"
    return ("%." + str(d) + "g") % x if 1e-3 <= abs(x) < 1e5 else "%.3e" % x


def load(p):
    with open(p) as fh:
        return json.load(fh)


def crossing(R):
    tc = R["run"].get("throat_crossing")
    if not tc:
        return None
    recs = R["run"]["records"]
    T = np.array([q["T_over_rs"] for q in recs])
    A = np.array([q["A_rel"] for q in recs])
    return {"T": tc["T_over_rs"], "A_rel": float(np.interp(tc["T_over_rs"], T, A)), "B_over_A": tc["B_over_A"], "N": tc["N_sh"], "c_plus": tc["c_plus_r"]}


def energy(R):
    recs = R["run"]["records"]
    H0 = recs[0]["sum_nu_H"]
    tc = crossing(R)
    Tc = tc["T"] if tc else np.inf
    before = [q for q in recs if q["T_over_rs"] <= Tc]
    return {"H_rel_max": max(abs(q["sum_nu_H"] / H0 - 1) for q in recs), "H_rel_max_before": max(abs(q["sum_nu_H"] / H0 - 1) for q in before),
            "A_rel_max": max(abs(q["A_rel"]) for q in recs), "A_rel_max_before": max(abs(q["A_rel"]) for q in before)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True, help="three S7a JSONs: dt, dt/2, dt/4")
    ap.add_argument("--rmax", default="", help="base:moved (R_max 64 and 128 r_s)")
    ap.add_argument("--runs", default="", help="comma list of S7a JSONs for the record checks (f)")
    ap.add_argument("--static", default="", help="comma list of static-check JSONs (revision 1)")
    ap.add_argument("--frozen-pair", default="", help="dop853:verlet frozen_schw JSONs (revision 1)")
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    L = ["# Strand 7 S7a part 2: the ladder, the moved exterior, the record checks (strand7_part2_report_v1.py)", ""]
    js = {}
    lad = [load(p) for p in a.ladder.split(",")]
    cr = [crossing(R) for R in lad]
    en = [energy(R) for R in lad]
    rows = []
    orders = {}
    for k in ("T", "A_rel", "B_over_A", "N", "c_plus"):
        v = [c[k] if c else None for c in cr]
        d1 = abs(v[0] - v[1]) if None not in v[:2] else None
        d2 = abs(v[1] - v[2]) if None not in v[1:] else None
        p = math.log2(d1 / d2) if (d1 and d2) else None
        orders[k] = {"values": v, "d_dt_dt2": d1, "d_dt2_dt4": d2, "order": p}
        rows.append("| %s | %s | %s | %s | %s | %s | %s |" % (k, f(v[0], 8), f(v[1], 8), f(v[2], 8), f(d1, 3), f(d2, 3), f(p, 3)))
    js["ladder"] = {"runs": a.ladder.split(","), "crossings": cr, "orders": orders, "energy": en}
    L += ["## (b) C3's ladder at E_0 = 16 (self): the throat crossing at dt = %s r_s" % ", ".join(f(R["meta"]["dt_over_rs"], 3) for R in lad), "",
          "| quantity | dt | dt/2 | dt/4 | difference dt to dt/2 | dt/2 to dt/4 | observed order |", "|---|---|---|---|---|---|---|"] + rows + ["",
          "| dt/r_s | max |sum nu H rel| before the crossing | over the run | max |A/A(0) - 1| before the crossing | over the run |", "|---|---|---|---|---|"]
    for R, e in zip(lad, en):
        L.append("| %s | %s | %s | %s | %s |" % (f(R["meta"]["dt_over_rs"], 3), f(e["H_rel_max_before"], 3), f(e["H_rel_max"], 3), f(e["A_rel_max_before"], 3),
                                               f(e["A_rel_max"], 3)))
    L.append("")
    if a.rmax:
        b, m = (load(p) for p in a.rmax.split(":"))
        tb = {round(q["T_over_rs"], 6): q for q in b["run"]["records"]}
        tm = {round(q["T_over_rs"], 6): q for q in m["run"]["records"]}
        common = sorted(t for t in tb if t in tm)
        cb, cm = crossing(b), crossing(m)
        Tc = cb["T"] if cb else np.inf
        diffs = {}
        for k in ("A", "B", "B_over_A", "r_sh"):
            pre = [abs(tm[t][k] / tb[t][k] - 1) for t in common if t <= Tc]
            post = [abs(tm[t][k] / tb[t][k] - 1) for t in common if t > Tc]
            diffs[k] = {"before": max(pre) if pre else None, "after": max(post) if post else None}
        js["rmax"] = {"diffs": diffs, "crossing_base": cb, "crossing_moved": cm, "R_max": [b["meta"]["Rmax_over_rs"], m["meta"]["Rmax_over_rs"]]}
        L += ["## (e) The exterior moved: R_max = %s r_s against %s r_s (largest relative differences at the %d common outputs)" % (
            f(m["meta"]["Rmax_over_rs"], 4), f(b["meta"]["Rmax_over_rs"], 4), len(common)), "",
              "| quantity | before the crossing | after |", "|---|---|---|"] + [
            "| %s | %s | %s |" % (k, f(v["before"], 3), f(v["after"], 3)) for k, v in diffs.items()] + ["",
              "- The crossing records: base %s; moved %s." % (json.dumps(cb, default=float), json.dumps(cm, default=float)), ""]
    if a.runs:
        rows = []
        js["records"] = {}
        for p in a.runs.split(","):
            R = load(p)
            recs = R["run"]["records"]
            nc = sum(1 for q in recs if q.get("outer_converged") is False)
            mr = max(q["residual"] for q in recs if q.get("residual") is not None)
            js["records"][R["meta"]["tag"]] = {"outputs": len(recs), "not_converged": nc, "max_residual": mr}
            rows.append("| %s | %d | %d | %s |" % (R["meta"]["tag"], len(recs), nc, f(mr, 3)))
        L += ["## (f) Record checks: the outer loop's convergence flag per output and the largest lagged-source residual", "",
              "| run | outputs | outputs not converged | largest residual |", "|---|---|---|---|"] + rows + [""]
    if a.static:
        for p in a.static.split(","):
            S = load(p)["static_check"]
            L.append("- C1 static check %s: resolve_A_rel %s, resolve_B_rel %s, resolve outer/newton/residual %s." % (
                p.split("/")[-1], f(S["resolve_A_rel"], 3), f(S["resolve_B_rel"], 3), json.dumps(S["resolve_info"], default=float)))
        L.append("")
    if a.frozen_pair:
        d, v = (load(p) for p in a.frozen_pair.split(":"))
        md = max(abs(q["sum_nu_H_rel"]) for q in d["run"]["records"] if q.get("sum_nu_H_rel") is not None)
        mv = max(abs(q["sum_nu_H_rel"]) for q in v["run"]["records"] if q.get("sum_nu_H_rel") is not None)
        L += ["- C2 energy: the largest |sum nu H rel| of frozen_schw by DOP853 %s and by the run's own stepper %s (the spec's level 1e-8)." % (f(md, 3), f(mv, 3)), ""]
        js["C2"] = {"dop853": md, "stepper": mv}
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump(js, open(a.out_json, "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
