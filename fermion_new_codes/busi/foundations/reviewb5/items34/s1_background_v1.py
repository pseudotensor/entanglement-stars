#!/usr/bin/env python
"""s1_background_v1.py (busi, reviewb5 items 3/4, PM1). DATA, box CPU (8 threads, torch eigh).

Stage 1: the PM1 background fields u_0 = N3's radial fixed point on the ball (the same loop and args as n3_background_v1 run_ball:
R_b 8, m 0.2, g 0.25, eps 1e-4, ell 2, pad_to 32, tol 1e-11), the full eigendecomposition of h[u_0] (scratch, deleted at the end of
stage 2), the eps-mixture's site sources, and the background jets at the PM1 test points of T_d for d = 2, 3, 4 (two FD steps).
Outputs: <out>/s1_background_v1.json, s1_background_v1.npz (small); <scratch>/eigh_hu0.npz (1.1 GB, scratch)."""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "8"
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N345 = os.path.join(HERE, "..", "..", "n345")
CODE = "<internal>/code16d"
sys.path.insert(0, HERE); sys.path.insert(0, N345); sys.path.insert(0, CODE)
import torch  # noqa: E402
torch.set_num_threads(8)
import engine_ops_v1 as OPS  # noqa: E402
import engine_fields_v16 as FLD  # noqa: E402
import n3_background_v1 as N3  # noqa: E402
import items34_lib as L  # noqa: E402

OUT = "<internal>/items34"
SCR = "<internal>/scratch"
M_, G_, EPS, ELL, RB, PAD, TOL = 0.2, 0.25, 1e-4, 2, 8, 32.0, 1e-11


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(SCR, exist_ok=True)
    t0 = time.time()
    P = N3.BallParent(OPS, RB, M_)
    Rsrc = P.runiq
    N_site = np.ones(P.nsite); v_vec = np.zeros((P.nsite, 3))
    e, U = P.spectra(N_site, v_vec)
    hist = []
    for it in range(8):
        rho, J = P.site_sources(U[:, e < 0], -2.0 * EPS)
        eps_r, J_r = P.radial(rho, J)
        Rf, info, Na, va = N3.solve_rows(FLD, Rsrc, eps_r, J_r, G_, ELL, PAD)
        N_new = Na[P.inv]; v_new = va[P.inv][:, None] * P.xhat
        dN = float(np.max(np.abs(N_new - N_site))); dv = float(np.max(np.abs(v_new - v_vec)))
        hist.append({"it": it, "dN": dN, "dv": dv, "E": float(info["E_collar"]), "t": time.time() - t0})
        print(hist[-1], flush=True)
        N_site, v_vec = N_new, v_new
        del U
        e, U = P.spectra(N_site, v_vec)
        if dN < TOL and dv < TOL:
            break
    Uneg = U[:, e < 0]
    rho, J = P.site_sources(Uneg, -2.0 * EPS)          # the eps-mixture's sources at u_0: Tr(eps (I - 2P) e_x) = -2 eps Tr(P e_x)
    eps_r, J_r = P.radial(rho, J)
    Rf, info, Na, va = N3.solve_rows(FLD, Rsrc, eps_r, J_r, G_, ELL, PAD)
    fpr = {"dN": float(np.max(np.abs(Na[P.inv] - N_site))), "dv": float(np.max(np.abs(va[P.inv][:, None] * P.xhat - v_vec)))}
    np.savez(os.path.join(SCR, "eigh_hu0.npz"), e=e, U=U)
    jets = {}
    for d in (2, 3, 4):
        Y = L.test_points(d, RB)
        J5 = L.background_jets(Y, info, h=0.05); J10 = L.background_jets(Y, info, h=0.1)
        r = np.sqrt((Y ** 2).sum(1))
        jets[d] = {"Y": Y, "J5": J5, "J10": J10, "rho_sph": np.asarray(info["rho_ell"](r), float), "J_sph": np.asarray(info["J_ell"](r), float)}
    rr = np.linspace(0.0, 32.0, 6401)
    fs = N3.field_summary(info, G_, float(RB), ELL)
    res = {"version": "s1_background_v1", "iterations": hist, "fixed_point_residual": fpr, "fields": fs,
           "nsite": int(P.nsite), "dim": int(P.dim), "n_neg": int((e < 0).sum()), "min_abs_eig_float": float(np.abs(e).min()),
           "E_collar": float(info["E_collar"]), "r_collar": float(info["r_collar"]),
           "jet_fd_step_diff_max": {str(d): float(np.max(np.abs(jets[d]["J5"] - jets[d]["J10"]))) for d in jets},
           "provenance": {"script_md5": md5(__file__), "lib_md5": md5(L.__file__), "n3_md5": md5(N3.__file__), "engine_ops_v1": md5(OPS.__file__),
                          "engine_fields_v16": md5(FLD.__file__), "torch": torch.__version__, "numpy": np.__version__}, "elapsed_s": time.time() - t0}
    sv = {"N_site": N_site, "v_vec": v_vec, "rho_site": rho, "J_site": J, "e": e, "r_prof": rr, "N_prof": info["N_dense"](rr),
          "v_prof": info["v_dense"](rr), "rho_prof": info["rho_ell"](rr), "X": P.X, "r_site": P.r}
    for d in jets:
        for k, v in jets[d].items():
            sv["d%d_%s" % (d, k)] = v
    np.savez(os.path.join(OUT, "s1_background_v1.npz"), **sv)
    with open(os.path.join(OUT, "s1_background_v1.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
