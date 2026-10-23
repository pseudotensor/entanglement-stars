#!/usr/bin/env python
"""engine_collar_check_v1.py (builder analyst; red team R24c defect 1: the collar's "flag" policy imposes the vacuum exterior at the chain's
last node without a certified tail bound). Per stored slice of a collapse fields npz (the shell totals eps_tot, J_tot): the flag solve as the
run did it (engine_fields_v7, collar policy flag, the sites as stored) against the padded solve (the shell totals continued by zeros for
--pad-ell ell beyond the chain's end, collar policy fail: the collar then sits R_last + 8 ell out with its full margin; the matter chain itself
is not extended, so this measures the smoothing tail and the exterior's imposition, not the polarized chain's own convergence). Printed: E
and the relative change, W at the flag collar and at 2 r_s, N at the flag collar, the trapping zeros of both solves, and on the padded solve's
dense functions the signed and absolute weighted tail integrals beyond the flag collar, 4 pi int r^2 (rho_ell + q J_ell) dr and
4 pi g int r J_ell/q dr with q = v/N (R24c's charge and lapse-normalization forms). Args: --fields --g --slices --pad-ell --md."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_fields_v7 as F7  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--slices", type=int, nargs="*", default=None)
    ap.add_argument("--pad-ell", type=float, default=16.0)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    Z = np.load(a.fields, allow_pickle=True)
    R = np.asarray(Z["R"], float); ell = int(Z["ell"])
    idx = a.slices if a.slices else list(range(len(Z["T"])))
    npad = int(np.ceil(a.pad_ell * ell))
    Rp = np.concatenate([R, R[-1] + np.arange(1, npad + 1, dtype=float)])
    L = ["## The collar's flag against a padded exterior (engine_collar_check_v1.py on %s; ell %d; padding %d sites = %g ell of zero sources)" % (
        os.path.basename(a.fields), ell, npad, a.pad_ell), "",
         "| slice | T | E flag | E padded | relative change | W(collar) flag, padded | N(collar) flag, padded | trapping zeros flag; padded | tail: 4 pi int r^2 (rho + q J) (signed; absolute) | tail: 4 pi g int r J/q (signed; absolute) | collar flag; padded |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i in idx:
        eps = np.asarray(Z["eps_tot"][i], float); J = np.asarray(Z["J_tot"][i], float); N0 = np.asarray(Z["N"][i], float)
        Nf, vf, Wf, inf_f = F7.solve_constraints(eps, J, a.g, R, ell, N0=N0, collar_policy="flag")
        epsp = np.concatenate([eps, np.zeros(npad)]); Jp = np.concatenate([J, np.zeros(npad)]); N0p = np.concatenate([N0, np.ones(npad)])
        Np, vp, Wp, inf_p = F7.solve_constraints(epsp, Jp, a.g, Rp, ell, N0=N0p, collar_policy="fail")
        if inf_f.get("failed") or inf_p.get("failed"):
            L.append("| %d | %.1f | failed: flag %s; padded %s |" % (i, float(Z["T"][i]), inf_f.get("failed"), inf_p.get("failed")) + " - |" * 8)
            continue
        Ef = F7.charge(inf_f, a.g); Ep = F7.charge(inf_p, a.g)
        rc = float(inf_f["r_collar"])
        Wc_f = float(inf_f["W_field"](rc)[0]); Wc_p = float(inf_p["W_field"](rc)[0])
        Nc_f = float(inf_f["N_dense"](rc)[0]); Nc_p = float(inf_p["N_dense"](rc)[0])
        tz_f = F7.trapping_radius(Nf, vf, R, inf_f)["zeros"]; tz_p = F7.trapping_radius(Np, vp, Rp, inf_p)["zeros"]
        rr = np.linspace(rc, float(inf_p["r_collar"]), 4001)
        rho = inf_p["rho_ell"](rr); Jl = inf_p["J_ell"](rr)
        Nd = inf_p["N_dense"](rr); vd = inf_p["v_dense"](rr)
        q = np.where(np.abs(Nd) > 0, vd / Nd, 0.0)
        f1 = 4 * np.pi * rr ** 2 * (rho + q * Jl)
        with np.errstate(divide="ignore", invalid="ignore"):
            f2 = np.where(np.abs(q) > 1e-300, 4 * np.pi * a.g * rr * Jl / q, 0.0)
        tr = lambda f: float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(rr)))
        L.append("| %d | %.1f | %.12f | %.12f | %.3e | %.10f, %.10f | %.12f, %.12f | %s; %s | %.3e; %.3e | %.3e; %.3e | %.1f; %.1f |" % (
            i, float(Z["T"][i]), Ef, Ep, (Ep - Ef) / Ef, Wc_f, Wc_p, Nc_f, Nc_p, ["%.4f" % z for z in tz_f], ["%.4f" % z for z in tz_p],
            tr(f1), tr(np.abs(f1)), tr(f2), tr(np.abs(f2)), rc, float(inf_p["r_collar"])))
        print(L[-1], flush=True)
    L += ["", "- every number above is printed by the script; the padding extends the smoothing's support, not the matter chain (R24c: extending the geometric grid",
          "  recovers known smoothing tails; it does not establish convergence of the polarized chain itself)."]
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
