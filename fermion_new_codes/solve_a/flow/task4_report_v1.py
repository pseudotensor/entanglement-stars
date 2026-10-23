#!/usr/bin/env python
"""Task 4 report (SPEC_task4_injection.md sec. 1, read with the lead's comparator of SPEC_task3_prescribed_flow.md sec. 11.6; builder analyst):
the tables of task4_static_v4.py's records (the static chart, the lattice at rest, the locked background L frozen at the shell lapse N_s).
Per configuration (N_s, j, kappa sign, mass) and packet (omega r_h): the reflected fraction (the largest norm beyond r_ref after its minimum)
and its end value, the largest norm inside the shell, the measured delay against the three WKB integrals (the lead's light-speed 2 int psi^2/N
dr, the sine law's, the staggered law's with the measured band top), the turning radius (the sine law's and the staggered law's N = h omega
psi^2/2) against the innermost radius reached; per configuration the fit delay = a + b log(1/omega) over the packets with a measured delay,
b against 16 A_0 (= 4 r_s with r_s = 4 A_0), and the same fit of each WKB column.  Args: --jsons (comma list) --json --md."""
import argparse
import json
import math

import numpy as np


def f(v, d=4):
    if v is None:
        return "-"
    v = float(v)
    if not math.isfinite(v):
        return "-"
    return "%.*g" % (d, v) if (v == 0 or 1e-3 <= abs(v) < 1e6) else "%.3e" % v


def fit(om, y):
    ok = [(math.log(1 / o), v) for o, v in zip(om, y) if v is not None and o > 0]
    if len(ok) < 2:
        return None
    x = np.array([p[0] for p in ok])
    yy = np.array([p[1] for p in ok])
    A = np.vstack([x, np.ones_like(x)]).T
    c, *_ = np.linalg.lstsq(A, yy, rcond=None)
    return {"slope": float(c[0]), "intercept": float(c[1]), "n": len(ok), "rms": float(np.sqrt(np.mean((yy - A @ c) ** 2)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsons", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    runs = []
    for p in a.jsons.split(","):
        d = json.load(open(p))
        d["_path"] = p
        runs.append(d)
    runs.sort(key=lambda d: (-d["meta"]["Ns"], d["meta"]["j"], d["meta"]["sign"], d["meta"].get("mass", 0.0)))
    L = ["# Task 4: packet injection on the static chart (task4_report_v1.py over task4_static_v4.py's records; locked background L frozen at "
         "N_s; the lattice at rest)", "",
         "| N_s | j | kappa sign | m | omega r_h | reflected (first return) | at the end | max inside the shell | delay | WKB light-speed | WKB sine "
         "law | WKB staggered | r_t sine law | r_t staggered | innermost (1 percent edge, peak) |", "|" + "---|" * 15]
    out = {"configs": []}
    for d in runs:
        m = d["meta"]
        for q in d["packets"]:
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s, %s |" % (
                f(m["Ns"]), f(m["j"]), "%+d" % m["sign"], f(m.get("mass", 0.0)), f(q["omega_rh"]), f(q["reflected"]), f(q["reflected_end"]),
                f(q["inside_shell_max"]), f(q["delay"], 5), f(q["wkb_delay"], 5), f(q["wkb_lattice"], 5), f(q["wkb_staggered"], 5), f(q["r_t_pred"]),
                f(q["r_t_staggered"]), f(q["innermost_1pct"]), f(q["innermost_peak"])))
        om = [q["omega"] for q in d["packets"]]
        fits = {k: fit(om, [q[k] for q in d["packets"]]) for k in ("delay", "wkb_delay", "wkb_lattice", "wkb_staggered")}
        out["configs"].append({"Ns": m["Ns"], "j": m["j"], "sign": m["sign"], "mass": m.get("mass", 0.0), "A0": m["A0"], "r_h": m["r_h"],
                               "eps_max": m.get("eps_max"), "fits": fits, "slope_test_16A0": 16 * m["A0"], "path": d["_path"]})
    L += ["", "| N_s | j | kappa sign | m | A_0 | band top eps_max | delay slope b (n) | 16 A_0 | WKB light-speed slope | WKB sine slope | WKB staggered "
          "slope | delay intercept | staggered intercept |", "|" + "---|" * 13]
    for c in out["configs"]:
        F = c["fits"]
        g = lambda k, w: F[k][w] if F[k] else None
        L.append("| %s | %s | %+d | %s | %s | %s | %s (%s) | %s | %s | %s | %s | %s | %s |" % (
            f(c["Ns"]), f(c["j"]), c["sign"], f(c["mass"]), f(c["A0"]), f(c["eps_max"]), f(g("delay", "slope"), 5), F["delay"]["n"] if F["delay"] else 0,
            f(c["slope_test_16A0"], 5), f(g("wkb_delay", "slope"), 5), f(g("wkb_lattice", "slope"), 5), f(g("wkb_staggered", "slope"), 5),
            f(g("delay", "intercept"), 5), f(g("wkb_staggered", "intercept"), 5)))
    L += ["", "- Every number above is read from the records by the script; a dash is a packet whose outgoing crossing did not occur in its run "
          "(the record's t_out is null) or a quantity the record does not define."]
    json.dump(out, open(a.json, "w"), indent=1, default=float)
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
