#!/usr/bin/env python
"""engine_fields_v10_test_v1.py (builder analyst): engine_fields_v10's two new functions on stored slices of a collapse fields npz.
Per chosen record: (1) solve_constraints (version 9's Picard sweeps, the floor rule) and solve_constraints_shoot (the coupled shooting
solve) from the stored sources with the stored lapse as the start: their seconds, sweeps or integrations, the largest differences of N
and of W/max|W| on the fine grid, the charges E; (2) site_fields_kernel_average on the shooting pair against point sampling (the dense
functions at the sites): max |N_avg - N_point|, max |v_avg - v_point|, and the kernel weights' sums (min, max over the sites).
Args: --fields --records --md."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_fields_v10 as FLD  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True)
    ap.add_argument("--records", type=int, nargs="+", default=[0, -1])
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    Z = np.load(a.fields, allow_pickle=True)
    R = np.asarray(Z["R"], float); g = float(Z["g"]); ell = float(Z["ell"]); T = np.asarray(Z["T"], float)
    L = ["## engine_fields_v10 on stored slices (engine_fields_v10_test_v1.py over %s)" % os.path.basename(a.fields), "",
         "| record | T | Picard: s, sweeps | shooting: s, integrations, last F | max dN (fine grid) | max dW/max W | E Picard | E shooting | E rel diff | kernel avg: max dN, max dv | weights min, max |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in a.records:
        eps = np.asarray(Z["eps_tot"][k], float); J = np.asarray(Z["J_tot"][k], float); N0 = np.asarray(Z["N"][k], float)
        t0 = time.time()
        N1, v1, W1, i1 = FLD.solve_constraints(eps, J, g, R, ell, N0=N0, collar_policy="flag", support_rel=1e-14, on_maxit="return")
        s1 = time.time() - t0
        t0 = time.time()
        N2, v2, W2, i2 = FLD.solve_constraints_shoot(eps, J, g, R, ell, N0=N0, collar_policy="flag", support_rel=1e-14, on_maxit="return")
        s2 = time.time() - t0
        if i1.get("failed") or i2.get("failed"):
            L.append("| %d | %.2f | %s | %s |" % (k, T[k], i1.get("failed"), i2.get("failed")))
            continue
        rf = i1["rfine"]
        dN = float(np.max(np.abs(i1["N_dense"](rf) - i2["N_dense"](rf))))
        Wa = i1["W_field"](rf); dW = float(np.max(np.abs(Wa - i2["W_field"](rf)))) / max(float(np.max(np.abs(Wa))), 1e-300)
        E1 = FLD.charge(i1, g); E2 = FLD.charge(i2, g)
        Na, va, norms = FLD.site_fields_kernel_average(i2, R, ell)
        L.append("| %d | %.2f | %.1f, %d | %.1f, %d, %.1e | %.2e | %.2e | %.12f | %.12f | %.1e | %.2e, %.2e | %.12f, %.12f |" % (
            k, T[k], s1, i1["it"], s2, i2["it"], i2["shoot"]["last_F"], dN, dW, E1, E2, abs(E1 - E2) / abs(E1),
            float(np.max(np.abs(Na - N2))), float(np.max(np.abs(va - v2))), float(norms.min()), float(norms.max())))
        print(L[-1], flush=True)
    L += ["", "- every number above is printed by the script; the Picard solve is version 9's rule with on_maxit = return."]
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
