#!/usr/bin/env python
"""Strand 3: the mass rectangle fixed by the one match, the scan rows, the required coefficient and the job lists
(spec solve/specs/SPEC_strand3_entropy.md secs. 1.4, 3; builder analyst).

h_1/m_P^2 = (pi/6) / [(1 + lambda) log(1 + lambda) - lambda log lambda]; m_2 = sqrt(h_1), m_3 = sqrt(lambda h_1),
m_4 = sqrt((1 + lambda) h_1), m_1 = 0 (light packet massless on the lattice); q_s = (a/l_P)(m_s/m_P);
eta_req = (a/l_P)^2/4.  Control C3: sum_s eta_s m_s^2 = 0 and J = h_1 [(1 + lambda) log(1 + lambda) - lambda log lambda]
from the computed masses, eta_req(1.909).  Writes <out>/masses.json (rows, Q set, job lists).
"""
import argparse
import json
import os

import numpy as np

A_OVER_LP = (1.909, 1.627, 1.0, 0.5, 0.1627, 0.1, 0.05, 0.0213, 0.02, 0.01627, 0.0151, 0.00594, 0.00427)
LAMBDAS = (1.0, 4.0, 16.0)
EXTRA_Q = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.4, 2.0)
ETA_SIGN = (+1, -1, -1, +1)


def masses(lam):
    g = (1 + lam) * np.log(1 + lam) - lam * np.log(lam)
    h1 = (np.pi / 6) / g
    return h1, np.array([0.0, np.sqrt(h1), np.sqrt(lam * h1), np.sqrt((1 + lam) * h1)]), g


def qkey(q):
    return "%.12g" % q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows, qset, c3 = [], {}, []
    for lam in LAMBDAS:
        h1, m, g = masses(lam)
        m2 = m ** 2
        srule = float(np.dot(ETA_SIGN, m2))
        # J from the masses with m_1 -> 0 (the m_1^2 log m_1^2 term vanishes)
        J = float(sum(e * mm * np.log(mm) for e, mm in zip(ETA_SIGN, m2) if mm > 0))
        c3.append({"lambda": lam, "h1_over_mP2": float(h1), "sum_eta_m2": srule, "J": J, "J_formula": float(h1 * g),
                   "J_minus_formula": J - float(h1 * g), "mP2_from_J_minus_1": float(6 * J / np.pi - 1.0),
                   "m_over_mP": [float(v) for v in m]})
        for a in A_OVER_LP:
            q = [float(a * v) for v in m]
            rows.append({"a_over_lP": a, "lambda": lam, "q": q, "eta_req": a * a / 4.0,
                         "allowed_beta": a <= 0.0213, "allowed_LLR": a <= 0.00594})
            for v in q:
                qset[qkey(v)] = v
    for v in EXTRA_Q:
        qset[qkey(v)] = v
    Q = sorted(qset.values())
    # job lists (normal, depth, nk, q)
    planar = []
    for q in Q:
        planar.append(((1, 0, 0), 16, 12, q))
        planar.append(((1, 0, 0), 32, 24, q))
        if q <= 0.02:
            planar.append(((1, 0, 0), 64, 12, q))
    for q in (0.0, 0.02, 0.1, 0.5, 1.0):
        planar.append(((1, 1, 0), 16, 12, q))
        planar.append(((1, 1, 1), 16, 12, q))
    sphere_q = {qkey(v): v for v in (0.0, 0.1, 1.0)}
    for r in rows:
        if r["lambda"] == 1.0 and r["a_over_lP"] in (0.02, 0.00594):
            for v in r["q"]:
                sphere_q[qkey(v)] = v
    out = {"rows": rows, "Q": Q, "C3": c3, "eta_req_1.909": 1.909 ** 2 / 4, "paper_4piZhat0": 0.9106,
           "planar_jobs": [[list(p), d, nk, q] for (p, d, nk, q) in planar],
           "sphere_masses": sorted(sphere_q.values()),
           "ball_jobs": [[q, [0.0, 0.0, 0.0]] for q in (0.0, 0.1, 1.0)] + [[0.0, [0.5, 0.5, 0.5]]]}
    with open(os.path.join(args.out, "masses.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("rows %d, Q %d masses, planar jobs %d (depth 64: %d), sphere masses %d" % (
        len(rows), len(Q), len(planar), sum(1 for j in planar if j[1] == 64), len(out["sphere_masses"])))
    for c in c3:
        print(c)


if __name__ == "__main__":
    main()
