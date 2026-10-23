#!/usr/bin/env python
"""engine_fields_v12_covariance_v1.py (builder analyst; spec 18.7: "the single-IVP solver of 18.5 as solve_constraints_ivp with the
covariance check N -> cN"): on stored slices of a collapse fields npz, the outward initial-value problem of engine_fields_v12 integrated
from the series start with ln N(r_0) shifted by delta (c = e^delta: the pair N -> cN, W -> c^2 W at fixed sources, U = W/N^2 unchanged), then
normalized at the collar as the solver does; printed: the largest difference of the normalized lapse and of U from the unshifted
integration on the fine grid, and solve_constraints_ivp's own lapse against the unshifted integration (the same equations, the same start).
Args: --fields --records --deltas --md."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_fields_v12 as FLD  # noqa: E402


def integrate(src, g, r0, r_collar, U0, lnN0):
    rho_, J_ = src.rho, src.J
    flags = {"qmax": 0.0}

    def rhs(r, y):
        U = y[0]
        J = float(J_(r)); rho = float(rho_(r))
        if U > 0.0:
            q = -np.sqrt(U / r)
            flags["qmax"] = max(flags["qmax"], -q)
            if -q < 1e-12 * max(flags["qmax"], 1e-300):
                return [8.0 * np.pi * g * r * r * rho, 0.0]
            return [8.0 * np.pi * g * r * r * (rho + q * J), -4.0 * np.pi * g * r * J / q]
        return [8.0 * np.pi * g * r * r * rho, 0.0]
    return solve_ivp(rhs, (r0, r_collar), [U0, lnN0], method="DOP853", dense_output=True, rtol=1e-12, atol=1e-16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True)
    ap.add_argument("--records", type=int, nargs="+", default=[0, -1])
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.3, -0.7])
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    Z = np.load(a.fields, allow_pickle=True)
    R = np.asarray(Z["R"], float); g = float(Z["g"]); ell = float(Z["ell"]); T = np.asarray(Z["T"], float)
    L = ["## The covariance N -> cN of engine_fields_v12's initial-value solve (engine_fields_v12_covariance_v1.py over %s)" % os.path.basename(a.fields), "",
         "| record | T | delta | max |N_shifted - N| (normalized) | max |U_shifted - U|/max U | solve_constraints_ivp's N against the unshifted integration |", "|---|---|---|---|---|---|"]
    for k in a.records:
        eps = np.asarray(Z["eps_tot"][k], float); J = np.asarray(Z["J_tot"][k], float)
        Ns, vs, Ws, info = FLD.solve_constraints_ivp(eps, J, g, R, ell, collar_policy="flag", support_rel=1e-14)
        src = info["src"]; iv = info["ivp"]; r_collar = info["r_collar"]
        rf = np.linspace(iv["r0"], r_collar, 4001)
        base = integrate(src, g, iv["r0"], r_collar, iv["U0"], iv["lnN0"])
        Nb = np.exp(base.sol(rf)[1] - base.y[1, -1]); Ub = base.sol(rf)[0]
        dsolver = float(np.max(np.abs(info["N_dense"](rf) - Nb)))
        for d in a.deltas:
            sh = integrate(src, g, iv["r0"], r_collar, iv["U0"], iv["lnN0"] + d)
            Nsh = np.exp(sh.sol(rf)[1] - sh.y[1, -1]); Ush = sh.sol(rf)[0]
            L.append("| %d | %.2f | %+.2f | %.2e | %.2e | %.2e |" % (k, T[k], d, float(np.max(np.abs(Nsh - Nb))), float(np.max(np.abs(Ush - Ub))) / float(np.max(np.abs(Ub))), dsolver))
            print(L[-1], flush=True)
    L += ["", "- every number above is printed by the script; the right-hand sides do not contain ln N, so the shift is exact up to the integrator's rounding."]
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
