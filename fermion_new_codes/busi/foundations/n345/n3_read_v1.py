#!/usr/bin/env python
"""n3_read_v1.py (busi, foundations N3): reads the n3_background_v1 outputs (json + npz) and prints the derived table.

(1) The margin. C(u) = C_0 + P(u) - P(u_0) + X, X = R[s_g(u) - s_g(u_0)], C_0 = (1 - 2 eps) P(u_0) + eps I. Writing
    I - 2 P(u_0) = (I - 2 P(u)) + 2 (P(u) - P(u_0)): C(u) = (1 - 2 eps) P(u) + eps I + 2 eps dP + X, whose spectrum lies in
    [eps - 2 eps ||dP|| - ||X||, 1 - eps + 2 eps ||dP|| + ||X||] (Weyl). Strictly inside (0, 1) iff ||X|| + 2 eps ||dP|| < eps, so the
    first-order admissible bump amplitude is eta* = eps/(||X_1|| + 2 eps ||dP_1||) (X_1, dP_1 per unit amplitude). The data script's
    'eta_star_first_order' is the cruder eps/(||X_1|| + ||dP_1||); both are printed.
(2) How well the retained tests determine the averaged sources: the averaged energy density rho_ell on the fine profile against its
    cubic-spline interpolant through the test radii (spacing ell and ell/2) inside the parent.
Box only (CPU, seconds)."""
import glob
import json
import sys

import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, "<internal>/code16d")
import engine_fields_v4 as F4  # noqa: E402  (kernels: the field module's K_0, K_1)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "<internal>/n345"
    rows = []
    for jf in sorted(glob.glob(d + "/n3_*.json")):
        R = json.load(open(jf)); a = R["args"]; eps = a["eps"]
        Z = np.load(jf[:-5] + ".npz")
        rpar = a["n_parent"] if R["parent"] == "chain" else a["Rb"]
        Rs, es = (Z["R"], Z["eps_n"]) if R["parent"] == "chain" else (Z["R_uniq"], Z["eps_r"])
        r = np.linspace(0.0, rpar, 4001)
        rho = F4.kernels(r, Rs, a["ell"])[0] @ es                 # rho_ell exactly (the kernel sums)
        rho_at = lambda x: F4.kernels(x, Rs, a["ell"])[0] @ es
        res = {"file": jf.split("/")[-1], "parent": R["parent"], "eps": eps, "g": a["g"], "ell": a["ell"]}
        for tag, dr in (("spacing_ell", float(a["ell"])), ("spacing_ell_over_2", a["ell"] / 2.0)):
            nodes = np.arange(0.0, rpar + 1e-9, dr)
            cs = CubicSpline(nodes, rho_at(nodes), bc_type=((1, 0.0), "not-a-knot"))
            m = r <= rpar
            res["interp_rel_err_" + tag] = float(np.max(np.abs(cs(r[m]) - rho[m])) / np.max(np.abs(rho[m])))
            g_ = R["gram"][tag]
            res["gram_" + tag] = {"n": g_["n_tests"], "eig_min": g_["eig_min"], "cond": g_["cond"]}
            for kind in ("N", "v"):
                mm = R["margin"][tag][kind]
                X1 = mm["norm_R_ds_per_unit_bump"]; P1 = mm["norm_dP_per_unit_bump_first_order"]
                res["eta_star_%s_%s" % (kind, tag)] = {"sharp": eps / (X1 + 2 * eps * P1), "crude": mm["eta_star_first_order"],
                                                       "X1": X1, "dP1": P1, "moment_check_rel": mm["moment_check_rel"]}
        rows.append(res)
    print(json.dumps(rows, indent=1))
    with open(d + "/n3_read_v1.json", "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
