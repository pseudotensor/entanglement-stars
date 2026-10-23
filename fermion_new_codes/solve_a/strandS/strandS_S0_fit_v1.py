#!/usr/bin/env python
"""Strand S, job S0 (SPEC_strandS_graded_entropy.md sec. 5 C5; builder analyst): the stored-table fit.  From strand 3's A.1 records (the
eta.converge JSONs planar_1_0_0_d32_nk24_q<q>.json, the (32,24) ladder: depth 32, nk 24, final configuration (4L, 2nk, 2F)) fit
    [eta_100(q) - eta_100(0)]/q^2 = a_1 log q^2 + b_1 + c_1 q^2 log q^2 + d_1 q^2
by least squares on 0 < q <= --q-max (weights 1), and report a_1 (the lead's expectation 0.106 +- 0.005, i.e. 1/(3 pi) = 0.106103), b_1, c_1,
d_1, the largest residual, the same fit on the (16,12) ladder, and the fit's dependence on the lower end (q >= 0.02, 0.05) because the
uniform transverse grid (spacing 2 pi/(2 nk)) cannot resolve q^2 log q^2 below its node offset.  Args: --planar (the folder), --q-max,
--out-json, --md."""
import argparse
import glob
import json
import math
import os
import re

import numpy as np


def load(folder, tag):
    rows = []
    for p in glob.glob(os.path.join(folder, "planar_1_0_0_%s_q*.json" % tag)):
        d = json.load(open(p))
        rows.append((float(d["mass"]), float(d["eta"]), float(d["estimated_error"])))
    rows.sort()
    return np.array(rows)


def fit(rows, qmin, qmax):
    e0 = rows[rows[:, 0] == 0][0, 1]
    sel = (rows[:, 0] > 0) & (rows[:, 0] >= qmin) & (rows[:, 0] <= qmax)
    q = rows[sel, 0]
    y = (rows[sel, 1] - e0) / q ** 2
    lq = np.log(q * q)
    X = np.stack([lq, np.ones_like(q), q * q * lq, q * q], 1)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = X @ c - y
    return {"q_range": [qmin, qmax], "points": int(sel.sum()), "eta0": float(e0), "a_1": float(c[0]), "b_1": float(c[1]), "c_1": float(c[2]),
            "d_1": float(c[3]), "max_resid": float(np.max(np.abs(r))), "max_resid_times_q2": float(np.max(np.abs(r) * q * q))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planar", required=True)
    ap.add_argument("--q-max", type=float, default=0.1)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    out = {"target_a1": 1 / (3 * math.pi), "fits": {}}
    for tag in ("d32_nk24", "d16_nk12"):
        rows = load(a.planar, tag)
        out["fits"][tag] = [fit(rows, qmin, a.q_max) for qmin in (0.0, 0.02, 0.05)]
        out["fits"][tag + "_qmax0.5"] = [fit(rows, 0.0, 0.5)]
        out["n_records_" + tag] = int(len(rows))
        sel = [r for r in rows if abs(r[0] - 0.1) < 0.003 or abs(r[0] - 0.5) < 0.01]
        out["drops_" + tag] = [{"q": float(r[0]), "eta_minus_eta0": float(r[1] - rows[rows[:, 0] == 0][0, 1]), "estimated_error": float(r[2])} for r in sel]
    json.dump(out, open(a.out_json, "w"), indent=1)

    def f(v, d=6):
        return "%.*g" % (d, v)
    L = ["# Strand S job S0: the stored-table fit of strand 3's A.1 ladders (strandS_S0_fit_v1.py; SPEC_strandS_graded_entropy.md sec. 5 C5)", "",
         "Fit: [eta_100(q) - eta_100(0)]/q^2 = a_1 log q^2 + b_1 + c_1 q^2 log q^2 + d_1 q^2; the lead's expectation a_1 = 1/(3 pi) = %s." % f(1 / (3 * math.pi)), "",
         "| ladder | q range | points | a_1 | b_1 | c_1 | d_1 | largest residual (of the quotient) | largest residual times q^2 |", "|---|---|---|---|---|---|---|---|---|"]
    for tag, fl in out["fits"].items():
        for r in fl:
            L.append("| %s | [%s, %s] | %d | %s | %s | %s | %s | %s | %s |" % (tag, f(r["q_range"][0], 3), f(r["q_range"][1], 3), r["points"], f(r["a_1"]),
                                                                       f(r["b_1"]), f(r["c_1"]), f(r["d_1"]), "%.3e" % r["max_resid"], "%.3e" % r["max_resid_times_q2"]))
    for tag in ("d32_nk24", "d16_nk12"):
        L.append("")
        L.append("- %s (%d records): the drops eta_100(q) - eta_100(0): %s." % (tag, out["n_records_" + tag], "; ".join(
            "q = %s: %s (estimated error %s)" % (f(d["q"], 5), f(d["eta_minus_eta0"], 6), "%.2e" % d["estimated_error"]) for d in out["drops_" + tag])))
    L.append("")
    L.append("- The uniform transverse grid of eta.value (nk points per direction at the final configuration's 2nk) places no point closer than "
             "pi/(2nk) to a node in each direction, so below that scale the grid sum is analytic in q^2 and the fitted a_1 depends on the lower end of the range.")
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
