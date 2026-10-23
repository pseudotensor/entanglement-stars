#!/usr/bin/env python
"""test_fields_v7_v1.py (builder analyst): engine_fields_v7 against engine_fields_v6 on the stored slices of a collapse fields npz (the
re-solve from the shell totals, both versions, the same N0): the largest differences of N, v, W at the sites, the iteration counts and v7's
converged_at_floor entry. Args: --fields --g --collar-policy."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_fields_v6 as F6  # noqa: E402
import engine_fields_v7 as F7  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fields", required=True)
ap.add_argument("--g", type=float, default=0.25)
ap.add_argument("--collar-policy", default="flag")
a = ap.parse_args()
Z = np.load(a.fields, allow_pickle=True)
R = np.asarray(Z["R"], float); ell = int(Z["ell"])
for i in range(len(Z["T"])):
    eps = np.asarray(Z["eps_tot"][i], float); J = np.asarray(Z["J_tot"][i], float); N0 = np.asarray(Z["N"][i], float)
    out = []
    for F in (F6, F7):
        N, v, W, info = F.solve_constraints(eps, J, a.g, R, ell, N0=N0, collar_policy=a.collar_policy)
        out.append((N, v, W, info))
    (N6, v6, W6, i6), (N7, v7, W7, i7) = out
    if i6.get("failed") or i7.get("failed"):
        print("slice %d: v6 failed %s; v7 failed %s; v7 floor %s" % (i, i6.get("failed"), i7.get("failed"), i7.get("converged_at_floor")))
        continue
    print("slice %d T %.1f: max |dN| %.2e |dv| %.2e |dW| %.2e; iterations v6 %s v7 %s; v7 converged_at_floor %s; E v6 %.12f v7 %.12f" % (
        i, float(Z["T"][i]), np.max(np.abs(N6 - N7)), np.max(np.abs(v6 - v7)), np.max(np.abs(W6 - W7)), i6["it"], i7["it"], i7.get("converged_at_floor"),
        F6.charge(i6, a.g), F7.charge(i7, a.g)))
