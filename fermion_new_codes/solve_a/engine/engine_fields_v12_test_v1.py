#!/usr/bin/env python
"""engine_fields_v12_test_v1.py (builder analyst; engine_fields_v11_test_v1.py on engine_fields_v12; engine 18.5 and 18.1): on stored slices of a collapse fields npz, solve_constraints_ivp
against solve_constraints (Picard, version 9's rule) and solve_constraints_shoot: seconds, integrations, max dN and max dW/max W on the fine
grid, the charges; the original-form residuals of the IVP pair; the covariance check (the IVP's lnN and U against the Picard pair's ln N and
W/N^2 rescaled by N(r_collar)); padded_collar_cost on the slice; and once, the kernel identities: K_0 preserving constants and K_1 preserving
v = c r (site_fields_kernel_average on N = 1 and v = c r), each to 1e-12. Args: --fields --records --md."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_fields_v12 as FLD  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True)
    ap.add_argument("--records", type=int, nargs="+", default=[0, -1])
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    Z = np.load(a.fields, allow_pickle=True)
    R = np.asarray(Z["R"], float); g = float(Z["g"]); ell = float(Z["ell"]); T = np.asarray(Z["T"], float)
    L = ["## engine_fields_v12 on stored slices (engine_fields_v12_test_v1.py over %s)" % os.path.basename(a.fields), "",
         "| record | T | Picard s, sweeps | shooting s, integrations | IVP s (fevals) | IVP-Picard max dN | IVP-Picard max dW/W | IVP-shoot max dN | E Picard | E IVP | E rel | IVP residuals W, 1/N | padded collar: dE/E, ln N_b(a), passes |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in a.records:
        eps = np.asarray(Z["eps_tot"][k], float); J = np.asarray(Z["J_tot"][k], float); N0 = np.asarray(Z["N"][k], float)
        t0 = time.time(); N1, v1, W1, i1 = FLD.solve_constraints(eps, J, g, R, ell, N0=N0, collar_policy="flag", support_rel=1e-14, on_maxit="return"); s1 = time.time() - t0
        t0 = time.time(); N2, v2, W2, i2 = FLD.solve_constraints_shoot(eps, J, g, R, ell, N0=N0, collar_policy="flag", support_rel=1e-14, on_maxit="return"); s2 = time.time() - t0
        t0 = time.time(); N3, v3, W3, i3 = FLD.solve_constraints_ivp(eps, J, g, R, ell, collar_policy="flag", support_rel=1e-14); s3 = time.time() - t0
        if i1.get("failed") or i3.get("failed"):
            L.append("| %d | %.2f | %s | %s | %s |" % (k, T[k], i1.get("failed"), i2.get("failed"), i3.get("failed")))
            continue
        rf = i1["rfine"]
        dN13 = float(np.max(np.abs(i1["N_dense"](rf) - i3["N_dense"](rf))))
        Wa = i1["W_field"](rf); dW13 = float(np.max(np.abs(Wa - i3["W_field"](rf)))) / max(float(np.max(np.abs(Wa))), 1e-300)
        dN23 = float(np.max(np.abs(i2["N_dense"](rf) - i3["N_dense"](rf)))) if not i2.get("failed") else float("nan")
        E1, E3 = FLD.charge(i1, g), FLD.charge(i3, g)
        pc = FLD.padded_collar_cost(eps, J, g, R, ell, i3)
        L.append("| %d | %.2f | %.1f, %d | %.1f, %s | %.2f (%d) | %.2e | %.2e | %.2e | %.12f | %.12f | %.1e | %.1e, %.1e | %.2e, %.2e, %s |" % (
            k, T[k], s1, i1["it"], s2, i2.get("it"), s3, i3["ivp"]["nfev"], dN13, dW13, dN23, E1, E3, abs(E1 - E3) / abs(E1),
            i3.get("residual_W") or 0.0, i3.get("residual_invN") or 0.0, pc["delta_E_rel"], pc["lnN_b_at_a"], pc["passes"]))
        print(L[-1], flush=True)
        last = i3
    # the kernel identities, once, on the last slice's grid
    class Fake:
        pass
    infoN = {"N_dense": lambda r: np.ones_like(np.atleast_1d(np.asarray(r, float))), "v_dense": lambda r: 0.37 * np.atleast_1d(np.asarray(r, float))}
    Na, va, norms = FLD.site_fields_kernel_average(infoN, R, ell)
    L += ["", "Kernel identities (site_fields_kernel_average on N = 1 and v = 0.37 r over the %d sites): max |N_avg - 1| = %.2e; max |v_avg - 0.37 R_n| = %.2e (relative %.2e); the K_0 weights' sums %.15f to %.15f." % (
        len(R), float(np.max(np.abs(Na - 1.0))), float(np.max(np.abs(va - 0.37 * R))), float(np.max(np.abs(va - 0.37 * R) / (0.37 * R))), float(norms.min()), float(norms.max())),
          "", "- every number above is printed by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[-3:]))


if __name__ == "__main__":
    main()
