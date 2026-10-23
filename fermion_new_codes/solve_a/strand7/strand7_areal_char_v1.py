#!/usr/bin/env python
"""Strand 7 S7a part 2 (d): both areal characteristics from the stored fields of a self-consistent run (SPEC_strand7_collapse.md sec. 10.2(d);
builder analyst).  From <run>_fields.npz (u = psi - 1, v = chi - 1 on the nodes at every output) and the run's JSON (the median radius r_sh and
the charges per output): c_pm^(R)(r, T) = 2 r psi psi_dot + (psi^2 + 2 r psi psi') (pm N psi^-2), N = chi/psi, with psi' from the cubic spline
in r (zero slope at r = 0, not-a-knot; the S7a code's interpolation) and psi_dot by the central difference in T between the neighbouring outputs
(one-sided at the ends); per output: c_+^(R) and c_-^(R) at r_sh and at the coordinate throat r = A(T), their signs, and on the nodes (r >= h)
the sign changes of c_+^(R) (the radii where it crosses zero, by linear interpolation between nodes) with the node count where c_+^(R) <= 0.
GAP2_transport sec. 4.2's diagnostic with the inward branch.  Writes --md (the table at every r_s and every output of the last r_s; every
output with a sign change) and --out-json."""
import argparse
import json

import numpy as np
from scipy.interpolate import CubicSpline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    z = np.load(a.fields)
    R = json.load(open(a.json))
    rs = R["meta"]["r_s"]
    h = R["meta"]["h"]
    recs = {round(q["T"], 6): q for q in R["run"]["records"]}
    r, T, U, V = z["r"], z["T"], z["u"], z["v"]
    nT = len(T)
    rows = []
    for k in range(nT):
        q = recs.get(round(float(T[k]), 6))
        if q is None:
            continue
        su = CubicSpline(r, U[k], bc_type=((1, 0.0), "not-a-knot"))
        sv = CubicSpline(r, V[k], bc_type=((1, 0.0), "not-a-knot"))
        k0, k1 = max(k - 1, 0), min(k + 1, nT - 1)
        dT = T[k1] - T[k0]
        sdot = CubicSpline(r, (U[k1] - U[k0]) / dT, bc_type=((1, 0.0), "not-a-knot"))

        def cpm(x):
            ps = 1 + su(x)
            ch = 1 + sv(x)
            N = ch / ps
            base = 2 * x * ps * sdot(x)
            g = (ps * ps + 2 * x * ps * su(x, 1)) * N / ps ** 2
            return base + g, base - g
        A = q["A"]
        cp_sh, cm_sh = (float(c) for c in cpm(np.array([q["r_sh"]])))
        cp_A, cm_A = (float(c) for c in cpm(np.array([A])))
        nodes = r[r >= h]
        cpn, cmn = cpm(nodes)
        sg = np.sign(cpn)
        idx = np.nonzero(sg[:-1] * sg[1:] < 0)[0]
        zeros = [float(nodes[i] - cpn[i] * (nodes[i + 1] - nodes[i]) / (cpn[i + 1] - cpn[i])) for i in idx]
        rows.append({"T_over_rs": float(T[k] / rs), "A": A, "r_sh": q["r_sh"], "r_sh_over_A": q["r_sh"] / A, "c_plus_R_sh": cp_sh, "c_minus_R_sh": cm_sh,
                     "c_plus_R_A": cp_A, "c_minus_R_A": cm_A, "nonpositive_nodes": int(np.sum(cpn <= 0)), "zeros_r": zeros,
                     "zeros_r_over_A": [x / A for x in zeros], "min_c_plus_nodes": float(cpn.min()), "r_at_min": float(nodes[int(np.argmin(cpn))]),
                     "max_c_minus_nodes": float(cmn.max()), "monitor_c_plus_R": q.get("c_plus_R")})
    json.dump({"meta": {"fields": a.fields, "json": a.json, "r_s": rs, "h": h, "script": "strand7_areal_char_v1.py"}, "rows": rows},
              open(a.out_json, "w"), indent=1, default=float)

    def f(x, d=4):
        if x is None:
            return "-"
        x = float(x)
        return ("%." + str(d) + "g") % x if (x == 0 or 1e-3 <= abs(x) < 1e5) else "%.3e" % x
    Tl = rows[-1]["T_over_rs"]
    sel = [q for q in rows if abs(q["T_over_rs"] - round(q["T_over_rs"])) < 1e-6 or q["T_over_rs"] >= Tl - 1 - 1e-9 or q["zeros_r"]]
    any_zero = [q for q in rows if q["zeros_r"]]
    L = ["# Strand 7 S7a part 2 (d): the areal characteristics from the stored fields (strand7_areal_char_v1.py; %s)" % a.fields, "",
         "- Outputs: %d; outputs with a sign change of c_+^(R) on the nodes: %d%s; outputs with c_+^(R) <= 0 on some node: %d." % (
             len(rows), len(any_zero), (" (first at T = %s r_s, last at %s r_s)" % (f(any_zero[0]["T_over_rs"], 5), f(any_zero[-1]["T_over_rs"], 5)) if any_zero else ""),
             sum(1 for q in rows if q["nonpositive_nodes"] > 0)),
         "- Largest |c_+^(R)(r_sh) - the run's own monitor c_plus_R| over the outputs: %s (the monitor uses the backward difference in T)." % f(
             max(abs(q["c_plus_R_sh"] - q["monitor_c_plus_R"]) for q in rows if q["monitor_c_plus_R"] is not None), 3), "",
         "| T/r_s | r_sh/A | c_+^(R)(r_sh) | c_-^(R)(r_sh) | c_+^(R)(A) | c_-^(R)(A) | min c_+^(R) on nodes (r/A) | max c_-^(R) | nodes with c_+^(R) <= 0 | zeros of c_+^(R) (r/A) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for q in sel:
        L.append("| %s | %s | %s | %s | %s | %s | %s (%s) | %s | %d | %s |" % (
            f(q["T_over_rs"], 5), f(q["r_sh_over_A"]), f(q["c_plus_R_sh"]), f(q["c_minus_R_sh"]), f(q["c_plus_R_A"]), f(q["c_minus_R_A"]),
            f(q["min_c_plus_nodes"]), f(q["r_at_min"] / q["A"], 3), f(q["max_c_minus_nodes"]), q["nonpositive_nodes"],
            ", ".join(f(x) for x in q["zeros_r_over_A"]) or "-"))
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:4]))


if __name__ == "__main__":
    main()
