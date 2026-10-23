#!/usr/bin/env python
"""s6_jet_sensitivity_v1.py (busi, reviewb5 item 3/9): how much the PM1 background's correction Z = R[s_g(u_0) - L(eps(I - 2P))] moves when
the jets of u_0 are agent A's analytic-spline jets (pm1_common_v1.RadialBG on the shared cache) instead of B's finite-difference jets.
dZ = R[s_g^A(u_0) - s_g^B(u_0)] = sum_mu A_mu (Gamma^-1 ds)_mu; ||dZ|| by ARPACK (floating). d = 2, 3, 4. Box, minutes.
Output: <out>/s6_jet_sensitivity_v1.json."""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "8"
import json  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse.linalg as spl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "n345")); sys.path.insert(0, "<internal>/code16d")
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import engine_ops_v1 as OPS  # noqa: E402
import n3_background_v1 as N3  # noqa: E402
import items34_lib as L  # noqa: E402
import pm1_common_v1 as PM  # noqa: E402

OUT = "<internal>/items34"


def main():
    P = N3.BallParent(OPS, 8, 0.2)
    C = np.load("<internal>/pm1_background_v1.npz")
    bg = PM.RadialBG(C["r"], C["N"], C["v"], C["rho"], C["J"])
    res = {}
    for d in (4, 3, 2):
        Z = np.load(OUT + "/s2_items34_d%d.npz" % d)
        Y = Z["Y"]
        N, dN, dv, ddv = bg.jets3d(Y)
        rho, Jv = PM.laws(N, dN, dv, ddv, 0.0)
        sA = np.stack([rho, Jv[:, 0], Jv[:, 1], Jv[:, 2]], axis=1).ravel()
        ds = sA - Z["sg0"]
        Gm = Z["Gram"]
        y = np.linalg.solve(Gm, ds)
        Wt = L.test_weights(Y, P.X, 2.0)
        V = L.vertex_list(P, Wt)
        rows, cols, Vp = L.pattern_matrix(V, P.dim)
        dZ = L.op_from_coef(rows, cols, Vp, y.astype(np.complex128), P.dim)
        v0 = np.random.default_rng(3).standard_normal(P.dim) + 0j
        lv = spl.eigsh(dZ, k=1, which="LM", tol=1e-8, v0=v0, return_eigenvectors=False)
        R = json.load(open(OUT + "/s2_items34_d%d.json" % d))
        res["d%d" % d] = {"ds_energy_max_abs": float(np.abs(ds[0::4]).max()), "ds_momentum_max_abs": float(np.abs(ds.reshape(-1, 4)[:, 1:]).max()),
                          "norm_dZ_float": float(np.abs(lv).max()), "norm_dZ_rowsum_upper": L.inf_norm(dZ),
                          "margin_C0_lower_B": R["background"]["margin_C0_lower"],
                          "margin_C0_lower_with_A_jets_float": R["background"]["margin_C0_lower"] - float(np.abs(lv).max())}
        print(d, res["d%d" % d], flush=True)
    with open(OUT + "/s6_jet_sensitivity_v1.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
