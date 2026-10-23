#!/usr/bin/env python
"""s5_background_spectrum_v1.py (busi, reviewb5 item 3): the evaluated spectrum edge of the PM1 background C_0 itself (eta = 0) at each d,
against the Weyl bound eps - ||Z|| of s2. C_0 = (1 - 2 eps) P(u_0) + eps I + Z, Z = sum_mu A_mu yZ_mu (yZ stored by s2).
Floating (dense eigvalsh, backward error ~ n u ||C_0|| ~ 1e-12, negligible against the 1e-5 margins). Box, ~3 min per d.
Output: <out>/s5_background_spectrum_v1.json."""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "8"
import json  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "n345")); sys.path.insert(0, "<internal>/code16d")
import torch  # noqa: E402
torch.set_num_threads(8)
import engine_ops_v1 as OPS  # noqa: E402
import n3_background_v1 as N3  # noqa: E402
import items34_lib as L  # noqa: E402

OUT = "<internal>/items34"
EPS, ELL = 1e-4, 2.0


def main():
    P = N3.BallParent(OPS, 8, 0.2)
    E1 = np.load(OUT + "/scratch/eigh_hu0.npz")
    e0, U0 = E1["e"], E1["U"]
    Uo = torch.from_numpy(np.ascontiguousarray(U0[:, e0 < 0]))
    P0 = (Uo @ Uo.conj().T).numpy(); del Uo, U0
    res = {}
    for d in (4, 3, 2):
        f = OUT + "/s2_items34_d%d" % d
        if not os.path.exists(f + ".npz"):
            continue
        Z = np.load(f + ".npz"); R = json.load(open(f + ".json"))
        Wt = L.test_weights(Z["Y"], P.X, ELL)
        V = L.vertex_list(P, Wt)
        rows, cols, Vp = L.pattern_matrix(V, P.dim)
        Zop = L.op_from_coef(rows, cols, Vp, Z["yZ"].astype(np.complex128), P.dim).toarray()
        C0 = (1 - 2 * EPS) * P0 + EPS * np.eye(P.dim) + Zop
        ev = torch.linalg.eigvalsh(torch.from_numpy(C0)).numpy()
        res["d%d" % d] = {"lambda_min_C0": float(ev[0]), "one_minus_lambda_max_C0": float(1 - ev[-1]),
                          "margin_C0_eval": float(min(ev[0], 1 - ev[-1])), "weyl_lower_s2": R["background"]["margin_C0_lower"]}
        print(d, res["d%d" % d], flush=True)
        del C0, Zop
    with open(OUT + "/s5_background_spectrum_v1.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
