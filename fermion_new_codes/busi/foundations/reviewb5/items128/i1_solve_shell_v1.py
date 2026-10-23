#!/usr/bin/env python
"""i1_solve_shell_v1.py (busi, foundations reviewb5 item 1; agent A). DATA script, box CPU, 1 thread, seconds to minutes.

The field-solve Jacobian on the N3 shell space, J_solve = d(N_R, v_R)/d(eps_R, J_R): the shell totals at the 55 distinct radii of the
ball (54 for the radial momentum), through the engine's rows (n3_background_v1.solve_rows = engine_fields_v16.solve_constraints_ivp on
the default ivp route, then site_fields_kernel_average), to the back-averaged site fields at the same radii. Two routes:
  (FD)  central differences of the engine solve at two step sizes (h and h/4 of max|eps_R|), the difference of the two the estimate of
        the FD and solver-noise error;
  (LV)  the linearized rows on the dense background, a Volterra pair solved by fixed-point iteration on a grid of step 1/256 with
        cumulative Simpson quadrature:
           dW(r) = int_0^r 8 pi g s^2 (2 N rho dN + N^2 drho) ds,   dv = dW/(2 r v),
           d(1/N)(r) = -int_r^{r_c} 4 pi g s (dJ/v - J dv/v^2) ds,   dN = -N^2 d(1/N),   dN = 0, dW = dW(r_c) beyond r_c,
        then back-averaged with the engine's kernels (the adjoint of the source averaging).
Outputs: <out>/i1_solve_shell_v1.npz (J_fd_h, J_fd_h4, J_lv) and .json."""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy.integrate import cumulative_simpson  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import pm1_common_v1 as PM  # noqa: E402


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main():
    bgp = sys.argv[1] if len(sys.argv) > 1 else "<internal>/pm1_background_v1.npz"
    outd = "<internal>/items128"
    out = os.path.join(outd, "i1_solve_shell_v1")
    if os.path.exists(out + ".json"):
        sys.exit("refusing to overwrite")
    t0 = time.time()
    OPS, FLD, N3 = PM._imports()
    bg = PM.load_background(bgp, with_parent=False)
    Rb, m, g, eps, ell = bg["args"]; ell = int(ell); Rb = int(Rb)
    R = bg["Rsrc"]; e0 = bg["eps_r"].copy(); j0 = bg["J_r"].copy()
    pad_to = float(Rb + 12 * ell)
    nr = len(R)

    def fields(e_, j_):
        Rf, info, Na, va = N3.solve_rows(FLD, R, e_, j_, g, ell, pad_to)
        return np.concatenate([Na, va[1:]]), info

    f0, info0 = fields(e0, j0)
    res = {"version": "i1_solve_shell_v1", "script_md5": md5(os.path.abspath(__file__)), "bg_md5": md5(bgp), "nr": nr,
           "E0": float(info0["E_collar"]), "r_collar": float(info0["r_collar"]),
           "fields_vs_cache": float(max(np.max(np.abs(f0[:nr] - bg["Na"])), np.max(np.abs(f0[nr:] - bg["va"][1:]))))}
    scale = float(np.max(np.abs(e0)))
    Js = {}
    for tag, h in (("h", 1e-5 * scale), ("h4", 0.25e-5 * scale)):
        Jm = np.zeros((2 * nr - 1, 2 * nr - 1))
        for k in range(2 * nr - 1):
            ep, jp = e0.copy(), j0.copy(); em, jm = e0.copy(), j0.copy()
            if k < nr:
                ep[k] += h; em[k] -= h
            else:
                jp[k - nr + 1] += h; jm[k - nr + 1] -= h
            fp, _ = fields(ep, jp); fm, _ = fields(em, jm)
            Jm[:, k] = (fp - fm) / (2 * h)
        Js[tag] = Jm
        print("FD %s done (%.0f s)" % (tag, time.time() - t0), flush=True)
    # ------------------------------------------------------------ the linearized Volterra route
    info = info0
    rc = float(info["r_collar"])
    hgrid = 1.0 / 256.0
    r = np.arange(0.0, 48.0 + 1e-9, hgrid)
    inside = r <= rc + 1e-12
    N = np.asarray(info["N_dense"](r), float); v = np.asarray(info["v_dense"](r), float)
    K0r, K1r = FLD.kernels(r, R, ell)
    rho = K0r @ e0; J = K1r @ j0
    # back-averaging weights: int 4 pi r^2 f(r) K_l(r, R) dr by Simpson on the same grid
    wS = np.ones(len(r)); wS[1:-1:2] = 4.0; wS[2:-1:2] = 2.0
    if len(r) % 2 == 0:
        raise RuntimeError("Simpson needs an odd grid")
    wS *= hgrid / 3.0
    BA0 = (K0r * (4 * np.pi * r ** 2 * wS)[:, None]).T          # (nr, n_grid)
    BA1 = (K1r * (4 * np.pi * r ** 2 * wS)[:, None]).T
    rs = np.where(r > 0, r, 1.0)

    def lin_solve(drho, dJ, iters=60):
        dN = np.zeros_like(r); dv = np.zeros_like(r)
        for it in range(iters):
            fW = 8 * np.pi * g * r ** 2 * (2 * N * rho * dN + N ** 2 * drho)
            fW = np.where(inside, fW, 0.0)
            dW = cumulative_simpson(fW, x=r, initial=0.0)
            dv_new = np.where(r > 0, dW / (2 * rs * np.where(v != 0, v, -1.0)), 0.0)
            fL = np.where(inside & (r > 0), 4 * np.pi * g * r * (dJ / np.where(v != 0, v, -1.0) - J * dv_new / np.where(v != 0, v, -1.0) ** 2), 0.0)
            cum = cumulative_simpson(fL, x=r, initial=0.0)
            ic = int(np.argmin(np.abs(r - rc)))
            dinvN = -(cum[ic] - cum)
            dinvN = np.where(inside, dinvN, 0.0)
            dN_new = -N ** 2 * dinvN
            ch = max(np.max(np.abs(dN_new - dN)), np.max(np.abs(dv_new - dv)))
            dN, dv = dN_new, dv_new
            if ch < 1e-15 * max(1.0, np.max(np.abs(dN)), np.max(np.abs(dv))):
                break
        return dN, dv, it

    Jlv = np.zeros((2 * nr - 1, 2 * nr - 1))
    its = []
    for k in range(2 * nr - 1):
        if k < nr:
            drho = K0r[:, k]; dJ = np.zeros_like(r)
        else:
            drho = np.zeros_like(r); dJ = K1r[:, k - nr + 1]
        dN, dv, it = lin_solve(drho, dJ)
        its.append(it)
        Jlv[:, k] = np.concatenate([BA0 @ dN, (BA1 @ dv)[1:]])
    print("LV done (%.0f s)" % (time.time() - t0), flush=True)
    nrm = lambda A: float(np.linalg.norm(A, 2))
    res.update({"J_fd_h_norm2": nrm(Js["h"]), "fd_h_vs_h4_rel": nrm(Js["h"] - Js["h4"]) / nrm(Js["h4"]),
                "lv_vs_fd_h4_rel": nrm(Jlv - Js["h4"]) / nrm(Js["h4"]), "lv_iters_max": int(max(its)),
                "blocks_norm2_fd_h4": {"dN_deps": nrm(Js["h4"][:nr, :nr]), "dN_dJ": nrm(Js["h4"][:nr, nr:]), "dv_deps": nrm(Js["h4"][nr:, :nr]),
                                       "dv_dJ": nrm(Js["h4"][nr:, nr:])},
                "blocks_lv_vs_fd_rel": {"dN_deps": nrm(Jlv[:nr, :nr] - Js["h4"][:nr, :nr]) / max(nrm(Js["h4"][:nr, :nr]), 1e-300),
                                        "dN_dJ": nrm(Jlv[:nr, nr:] - Js["h4"][:nr, nr:]) / nrm(Js["h4"][:nr, nr:]),
                                        "dv_deps": nrm(Jlv[nr:, :nr] - Js["h4"][nr:, :nr]) / nrm(Js["h4"][nr:, :nr]),
                                        "dv_dJ": nrm(Jlv[nr:, nr:] - Js["h4"][nr:, nr:]) / max(nrm(Js["h4"][nr:, nr:]), 1e-300)},
                "elapsed_s": time.time() - t0})
    np.savez(out + ".npz", J_fd_h=Js["h"], J_fd_h4=Js["h4"], J_lv=Jlv, R=R)
    json.dump(res, open(out + ".json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
