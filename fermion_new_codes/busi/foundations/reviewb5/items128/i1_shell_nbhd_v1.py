#!/usr/bin/env python
"""i1_shell_nbhd_v1.py (busi, foundations reviewb5 item 1; agent A). DATA script, box CPU, 1 thread (beside the chi run), ~1 h.

A FLOATING-POINT check of how sigma_min(I + K) moves over a neighbourhood of the shell-space branch point (the actual N3 loop), by
recomputing K at displaced source points instead of bounding the variation: for s = s_0 + ds, u(s) = the engine's rows (solve_rows),
P(u(s)) = the sea of h[u(s)], chi_shell(u(s)) (the same vertices and formula as i1_chi_v1), J_solve(s) by central differences (the
h/4 step of i1_solve_shell_v1), K(s) = chi J, sigma_min(I + K(s)) in the same Euclidean shell norm. Displacements: ds = alpha s_0
(the sources rescaled) and ds = alpha |s_0|_max xi (xi a fixed Gaussian vector, seed 7), alpha in --alphas.
Output: <out>/i1_shell_nbhd_v1.json."""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import pm1_common_v1 as PM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", default="<internal>/pm1_background_v1.npz")
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.0, 1e-2, -1e-2, 3e-2])
    ap.add_argument("--out", default="<internal>/i1_shell_nbhd_v1")
    a = ap.parse_args()
    if os.path.exists(a.out + ".json"):
        sys.exit("refusing to overwrite")
    t0 = time.time()
    OPS, FLD, N3 = PM._imports()
    bg = PM.load_background(a.bg)
    P = bg["P"]
    Rb, m, g, eps, ell = bg["args"]; ell = int(ell)
    R = bg["Rsrc"]; e0 = bg["eps_r"]; j0 = bg["J_r"]; nr = len(R)
    pad_to = float(int(Rb) + 12 * ell)

    def fields(e_, j_):
        Rf, info, Na, va = N3.solve_rows(FLD, R, e_, j_, g, ell, pad_to)
        return Na, va, info

    def Jsolve(e_, j_):
        h = 0.25e-5 * float(np.max(np.abs(e0)))
        Jm = np.zeros((2 * nr - 1, 2 * nr - 1))
        for k in range(2 * nr - 1):
            ep, jp = e_.copy(), j_.copy(); em, jm = e_.copy(), j_.copy()
            if k < nr:
                ep[k] += h; em[k] -= h
            else:
                jp[k - nr + 1] += h; jm[k - nr + 1] -= h
            Na, va, _ = fields(ep, jp); Nb, vb, _ = fields(em, jm)
            Jm[:, k] = (np.concatenate([Na, va[1:]]) - np.concatenate([Nb, vb[1:]])) / (2 * h)
        return Jm

    # shell vertices (as i1_chi_v1.vertex_terms_shell)
    terms = []
    for Ri in range(nr):
        f = np.repeat((P.inv == Ri).astype(float), 4); terms.append([(0, f, np.nonzero(f)[0])])
    for Ri in range(1, nr):
        terms.append([(1 + i, np.repeat((P.inv == Ri) * P.xhat[:, i], 4), np.nonzero(np.repeat((P.inv == Ri) * P.xhat[:, i], 4))[0]) for i in range(3)])
    Ops = [P.h] + list(P.Q)

    def chi_shell(N_site, v_vec):
        H = P.hm_sparse(N_site, v_vec)
        e, U = torch.linalg.eigh(torch.from_numpy(H.toarray()))
        e = e.numpy(); U = U.numpy(); o = e < 0
        Uo = U[:, o]; Ue = U[:, ~o]
        w = 1.0 / (e[o][:, None] - e[~o][None, :])
        OUo = [np.asarray(Op @ Uo) for Op in Ops]; OUe = [np.asarray(Op @ Ue) for Op in Ops]
        n = len(terms)
        chi = torch.zeros((n, n), dtype=torch.float64)
        ne = Ue.shape[1]
        for c0 in range(0, ne, 512):
            cs = slice(c0, min(c0 + 512, ne))
            Zs = []
            for tl in terms:
                Z = 0
                for (oi, f, rows) in tl:
                    Z = Z + 0.5 * (Uo[rows].conj().T @ (f[rows][:, None] * OUe[oi][rows, cs]) + OUo[oi][rows].conj().T @ (f[rows][:, None] * Ue[rows, cs]))
                Zs.append(Z)
            Zt = torch.from_numpy(np.stack(Zs)).reshape(n, -1)
            Zw = (torch.from_numpy(np.stack(Zs)) * torch.from_numpy(w[:, cs])[None]).reshape(n, -1)
            chi += 2.0 * torch.real(Zt @ Zw.conj().T)
            del Zs, Zt, Zw
        chi = chi.numpy()
        return chi, float(np.min(np.abs(e))), int(o.sum())

    rng = np.random.default_rng(7)
    xi_e = rng.standard_normal(nr); xi_j = rng.standard_normal(nr); xi_j[0] = 0.0
    scale = float(np.max(np.abs(e0)))
    res = {"version": "i1_shell_nbhd_v1", "script_md5": hashlib.md5(open(os.path.abspath(__file__), "rb").read()).hexdigest(), "rows": []}
    cases = [("scale", al, e0 * (1 + al), j0 * (1 + al)) for al in a.alphas] + [("random", al, e0 + al * scale * xi_e, j0 + al * scale * xi_j) for al in a.alphas[1:2]]
    for kind, al, e_, j_ in cases:
        if kind == "random" and al == 0.0:
            continue
        Na, va, info = fields(e_, j_)
        N_site = Na[P.inv]; v_vec = va[P.inv][:, None] * P.xhat
        dN = float(np.max(np.abs(N_site - bg["N_site"]))); dv = float(np.max(np.abs(v_vec - bg["v_vec"])))
        chi, gap, nocc = chi_shell(N_site, v_vec)
        J = Jsolve(e_, j_)
        K = chi @ J
        s = np.linalg.svd(np.eye(K.shape[0]) + K, compute_uv=False)
        ev = np.linalg.eigvals(K)
        H1 = P.hm_sparse(N_site, v_vec); H0 = P.hm_sparse(bg["N_site"], bg["v_vec"])
        row = {"kind": kind, "alpha": al, "max_dN_site": dN, "max_dv_site": dv, "dh_inf_norm": float(np.asarray(abs(H1 - H0).sum(axis=1)).max()),
               "E": float(info["E_collar"]), "gap": gap, "n_occ": nocc, "sigma_min_I_plus_K": float(s[-1]), "K_norm2": float(np.linalg.norm(K, 2)),
               "K_spectral_radius": float(np.max(np.abs(ev))), "t": time.time() - t0}
        res["rows"].append(row)
        print(json.dumps(row), flush=True)
    json.dump(res, open(a.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
