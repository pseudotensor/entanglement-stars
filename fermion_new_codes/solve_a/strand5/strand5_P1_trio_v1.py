#!/usr/bin/env python
"""Strand 5 P1 addendum (builder analyst): the trio's own extrema against the crossing of omega_1.  From the new members' family record
(strand5_J4_members_v1.py: z_c = z_max - 0.002, z_max, z_max + 0.002) the parabola vertices of M(z_c) and N_B(z_c) through the three members,
and from P1's stage-2 record of the same members the crossing of omega_1 by the quadratic through the three (as strand5_P1_cert_v1.py); prints
the vertices, the crossing and their differences, and the first law's check along the trio (dM/dN_B by the centred differences against
mu_inf at the middle member).  Numbers emitted, not retyped."""
import argparse
import json

import numpy as np


def vertex(z, y):
    c = np.polyfit(z, y, 2)
    return float(-c[1] / (2 * c[0])), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family-new", required=True)
    ap.add_argument("--s2-new", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    mem = sorted(json.load(open(a.family_new))["members"], key=lambda q: q["z_c"])
    trio = [q for q in mem if q["z_c"] < 1.0]
    z = np.array([q["z_c"] for q in trio])
    M = np.array([q["M"] for q in trio])
    NB = np.array([q["N_B"] for q in trio])
    zM, cM = vertex(z, M)
    zN, cN = vertex(z, NB)
    om = {round(d["z_c"], 9): float(d["full_node"]["lowest3"][0]) for d in json.load(open(a.s2_new))["members"]}
    w = np.array([om.get(round(q, 9), np.nan) for q in z])
    cw = np.polyfit(z, w, 2)
    roots = [float(x.real) for x in np.roots(cw) if abs(x.imag) < 1e-12 and z[0] - 0.01 < x.real < z[-1] + 0.01]
    zx = min(roots, key=lambda x: abs(x - z[1])) if roots else None
    dMdN = float((M[2] - M[0]) / (NB[2] - NB[0]))
    mu = trio[1].get("mu_inf")
    out = ["# Strand 5 P1 addendum: the trio's extrema and the crossing (strand5_P1_trio_v1.py)", "",
           "- trio z_c: %s" % ", ".join("%.6f" % x for x in z),
           "- M: %s; the parabola's vertex z_c = %.8f" % (", ".join("%.10e" % x for x in M), zM),
           "- N_B: %s; the parabola's vertex z_c = %.8f" % (", ".join("%.10e" % x for x in NB), zN),
           "- omega_1 (stage 2, full node): %s; the quadratic's root nearest the middle member z_x = %s" % (", ".join("%.6e" % x for x in w), "%.8f" % zx if zx is not None else "-"),
           "- z_x - vertex(M) = %s; z_x - vertex(N_B) = %s; vertex(M) - vertex(N_B) = %.3e" % (
               "%.3e" % (zx - zM) if zx is not None else "-", "%.3e" % (zx - zN) if zx is not None else "-", zM - zN),
           "- the first law along the trio: (M_+ - M_-)/(N_B+ - N_B-) = %.10e against the middle member's mu_inf = %s (ratio - 1 = %s)" % (
               dMdN, "%.10e" % mu if mu is not None else "-", "%.3e" % (dMdN / mu - 1) if mu else "-"), ""]
    open(a.md, "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
