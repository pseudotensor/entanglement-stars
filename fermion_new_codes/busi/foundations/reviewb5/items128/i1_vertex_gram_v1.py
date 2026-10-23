#!/usr/bin/env python
"""i1_vertex_gram_v1.py (busi, foundations reviewb5 item 1 with item 9; agent A). DATA script, box CPU, 1 thread, minutes.

The Hilbert-Schmidt Gram matrices Gamma_T = [Tr(X_mu X_nu)] of the T_d test vertices and Gamma_U = [Tr(V_j V_k)] of the U_d bump
vertices (PM1). They give the operator-norm form of the chi-variation bound of i1_assemble_v1:
    ||chi(u) - chi(u_0)||_2 <= 2 tau/(pi (d - tau)^2) * 3 (1 + 2 tau/(d - tau))^2 * sqrt(lambda_max(Gamma_T) lambda_max(Gamma_U)),
because for unit x, y the combined vertices X_x = sum x_mu X_mu, Y_y = sum y_j V_j have ||X_x||_F^2 = x^T Gamma_T x (and the
off-diagonal parts are bounded by the full ones). Exact formula for vertices {D_f, O}/2, {D_g, O'}/2 (D diagonal, O, O' Hermitian):
    Tr = (1/4) [2 sum_ij f_i Re(O_ij O'_ji) g_j + sum_i f_i g_i (O O' + O' O)_ii],
so Gamma = (1/4) [2 F^T M F' + F^T diag(c) F'] with M = Re(O o O'^T) sparse. lambda_max is enclosed by verify_v1 (Ostrowski +
Gershgorin) with the products' rounding bound. Output: <out>.json."""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import hashlib  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import pm1_common_v1 as PM  # noqa: E402
import verify_v1 as VF  # noqa: E402


def main():
    out = "<internal>/i1_vertex_gram_v1"
    if os.path.exists(out + ".json"):
        sys.exit("refusing to overwrite")
    bg = PM.load_background("<internal>/pm1_background_v1.npz")
    P = bg["P"]
    Ops = [P.h] + list(P.Q)
    ell, w = 2.0, 4.0
    res = {"version": "i1_vertex_gram_v1", "script_md5": hashlib.md5(open(os.path.abspath(__file__), "rb").read()).hexdigest()}
    # pair tables M[o][o'] = Re(O o O'^T), c[o][o'] = diag(O O' + O' O)
    M = {}; cdiag = {}
    for a in range(4):
        for b in range(4):
            M[a, b] = (Ops[a].multiply(Ops[b].T)).real.tocsr()
            cdiag[a, b] = np.real((Ops[a] @ Ops[b] + Ops[b] @ Ops[a]).diagonal())
    for d in (4, 3, 2):
        Y = PM.test_points(d)
        G = np.stack([np.repeat(PM.gauss_weights(y, P.X, ell), 4) for y in Y], axis=1)               # (n, nY) test weights
        Phi = np.stack([np.repeat(PM.bump_backavg(np.sqrt(((P.X - z[None, :]) ** 2).sum(axis=1)), w, ell), 4) for z in Y], axis=1)
        ent = {}
        for name, W in (("T", G), ("U", Phi)):
            nY = W.shape[1]
            Gam = np.zeros((4 * nY, 4 * nY))
            for a in range(4):
                for b in range(4):
                    blk = 0.25 * (2.0 * (W.T @ (M[a, b] @ W)) + W.T @ (cdiag[a, b][:, None] * W))
                    Gam[a::4, b::4] = blk
            Gam = 0.5 * (Gam + Gam.T)
            lo, hi, info = VF.sym_extreme_eigs(Gam)
            # rounding of Gam itself: each entry is a sum of <= 30 n products of O(1) terms; bound by gamma * |W|^T|M||W| etc.
            gerr = 0.0
            for a in range(4):
                for b in range(4):
                    gerr = max(gerr, float(np.max(0.25 * VF.gamma(40 * W.shape[0]) * (2 * np.abs(W).T @ (abs(M[a, b]) @ np.abs(W)) + np.abs(W).T @ (np.abs(cdiag[a, b])[:, None] * np.abs(W))))))
            ent[name] = {"n": 4 * nY, "lambda_max_upper": hi + 4 * nY * gerr, "lambda_min_lower": lo - 4 * nY * gerr, "trace": float(np.trace(Gam)),
                         "sum_F2": float(np.trace(Gam)), "float_lambda_max": info["float_eigs_max"], "entry_rounding_bound": gerr}
        res["d%d" % d] = ent
        print(d, json.dumps(ent), flush=True)
    json.dump(res, open(out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
