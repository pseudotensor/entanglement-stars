#!/usr/bin/env python
"""i2_projected_v1.py (busi, foundations reviewb5 item 2 with item 9; agent A). DATA script, box CPU, <= 8 threads, minutes.

The linearized projected constraint map of PM1 (common/item2_projected_solve.txt) at the N3 background u_0, and a bound on its inverse.
 (0) consistency: the collocated 3D laws at u_0 against the engine's smoothed sources at the test points (rho_ell(|y|), J_ell(|y|) yhat).
 (A) 3D, the PM1 solve: A_d = [D s_g[u_0](e_a psi_j)(y_mu)] for d in {4, 3, 2}, b in {0, +1, -1}; sigma_min(A_d) enclosed (verify_v1:
     inverse-residual bound, rigorous for the float matrix) and the float matrix's own error estimated by the background-table resolution
     (splines through every 2nd point of the 1/128 table); the smallest singular vector's make-up (N / v, radial / tangential).
     The Killing directions: D s_g[u_0](c) and D s_g[u_0](omega x x) at the test points for b = 0 and b = 1 (exactly 0 for b = 0).
 (B) spherical sector: (B1) the continuum linearized rows' inverse (sources -> fields; dN(r_c) = 0, dW(0) = 0) as an operator on
     L^2(4 pi r^2 dr) over [0, r_c] and on sup norms; (B2) the Galerkin band map on the Dirichlet modes j_0(k_n r) (scalars) and
     j_1(k_n r) rhat (radial vectors), k_n = n pi / r_c <= k_c, k_c = pi/d, orthonormal in L^2(4 pi r^2 dr); sigma_min enclosed.
Outputs: <out>/i2_projected_v1.npz (A_d matrices, singular values) and .json."""
import os
import sys


def _threads():
    t = "8"
    if "--threads" in sys.argv:
        t = sys.argv[sys.argv.index("--threads") + 1]
    t = str(max(1, min(int(t), 8)))
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = t
    return int(t)


NTH = _threads()
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy.special import spherical_jn  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import pm1_common_v1 as PM  # noqa: E402
import verify_v1 as VF  # noqa: E402

G = 0.25


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", default="<internal>/pm1_background_v1.npz")
    ap.add_argument("--d", type=int, nargs="*", default=[4, 3, 2])
    ap.add_argument("--w", type=float, default=4.0)
    ap.add_argument("--dvar", type=int, nargs="*", default=[4, 3])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--iv", type=int, default=1, help="mpmath interval cross-check for d >= 3")
    ap.add_argument("--out", default="<internal>/i2_projected_v1")
    a = ap.parse_args()
    if os.path.exists(a.out + ".json"):
        sys.exit("refusing to overwrite")
    t0 = time.time()
    log = lambda s: print("[%6.0f s] %s" % (time.time() - t0, s), flush=True)
    bg = PM.load_background(a.bg, with_parent=False)
    r = bg["r"]
    sp1 = PM.RadialBG(r, bg["N"], bg["v"], bg["rho_direct"], bg["J_direct"])
    sp2 = PM.RadialBG(r[::2], bg["N"][::2], bg["v"][::2], bg["rho_direct"][::2], bg["J_direct"][::2])
    res = {"version": "i2_projected_v1", "script_md5": md5(os.path.abspath(__file__)), "pm1_common_md5": md5(PM.__file__),
           "verify_md5": md5(VF.__file__), "bg_md5": md5(a.bg), "args": vars(a)}
    save = {}
    rc = float(bg["r_collar"])
    # ---------------------------------------------------------------- (0) consistency of the 3D laws with the engine's rows
    cons = {}
    for d in a.d:
        Y = PM.test_points(d)
        N0, dN0, dv0, ddv0 = sp1.jets3d(Y)
        rho_g, J_g = PM.laws(N0, dN0, dv0, ddv0, 0.0)
        ry = np.sqrt((Y ** 2).sum(axis=1))
        rho_e = sp1.rho(ry); Jr_e = sp1.J(ry)
        yh = np.where(ry[:, None] > 0, Y / np.where(ry[:, None] > 0, ry[:, None], 1.0), 0.0)
        J_e = Jr_e[:, None] * yh
        cons["d%d" % d] = {"rho_rel_maxdiff": float(np.max(np.abs(rho_g - rho_e)) / np.max(np.abs(rho_e))),
                           "J_maxdiff": float(np.max(np.abs(J_g - J_e))), "J_scale": float(np.max(np.abs(J_e))),
                           "J_g_maxabs": float(np.max(np.abs(J_g))), "rho_scale": float(np.max(np.abs(rho_e)))}
    res["laws_vs_engine_rows"] = cons
    log("laws vs rows: %s" % json.dumps(cons))
    # ---------------------------------------------------------------- (A) 3D PM1 solve
    res["A3d"] = {}
    for d in a.d:
        Y = PM.test_points(d)
        for b in (0.0, 1.0, -1.0):
            A = PM.linearized_A(sp1, Y, Y, a.w, b)
            tag = "d%d_b%+d" % (d, int(b))
            ent = {"n": int(A.shape[0])}
            s = np.linalg.svd(A, compute_uv=False)
            ent["float_sigma"] = {"min": float(s[-1]), "max": float(s[0]), "cond": float(s[0] / s[-1]), "smallest5": s[-5:].tolist()}
            if b == 0.0:
                A2 = PM.linearized_A(sp2, Y, Y, a.w, b)
                dA = float(np.linalg.norm(A - A2))
                ent["table_resolution_dA_F"] = dA
                enc = VF.sigma_min_enclosure(A)
                ent["enclosure_float_matrix"] = enc
                ent["sigma_min_lower_incl_table_estimate"] = enc["lo"] - dA
                ent["inv_norm_upper"] = 1.0 / (enc["lo"] - dA) if enc["lo"] > dA else None
                if a.iv and A.shape[0] <= 340:
                    ent["iv_check"] = VF.iv_sigma_min_check(A)
                    log("iv check %s: %s" % (tag, ent["iv_check"]))
                U_, s_, Vt = np.linalg.svd(A)
                vmin = Vt[-1].reshape(-1, 4)
                ent["smallest_vector_N_share"] = float((vmin[:, 0] ** 2).sum())
                # radial vs tangential share of the v part, at the bump centres
                rz = np.sqrt((Y ** 2).sum(axis=1)); yh = np.where(rz[:, None] > 0, Y / np.where(rz[:, None] > 0, rz[:, None], 1.0), 0.0)
                vr = (vmin[:, 1:] * yh).sum(axis=1)
                ent["smallest_vector_v_radial_share"] = float((vr ** 2).sum() / max((vmin[:, 1:] ** 2).sum(), 1e-300))
                umin = U_[:, -1].reshape(-1, 4)
                ent["smallest_left_vector_rho_share"] = float((umin[:, 0] ** 2).sum())
                save["A_" + tag] = A
                save["sv_" + tag] = s
                # blocks: the N columns against the momentum rows, the v columns against the energy rows
                ent["block_norms"] = {"rhoN": float(np.linalg.norm(A[0::4, 0::4], 2)), "JN": float(np.linalg.norm(A[np.r_[1:A.shape[0]:4, 2:A.shape[0]:4, 3:A.shape[0]:4]][:, 0::4], 2)),
                                      "rhov": float(np.linalg.norm(A[0::4][:, np.sort(np.r_[1:A.shape[1]:4, 2:A.shape[1]:4, 3:A.shape[1]:4])], 2))}
            res["A3d"][tag] = ent
            log("A3d %s: %s" % (tag, json.dumps({k: ent[k] for k in ent if k in ("n", "float_sigma", "enclosure_float_matrix", "table_resolution_dA_F")})))
    # the variation of A_d over the neighbourhood |c|_2 <= eta: first order L_A = sqrt(sum_k ||D^2 s_g[psi_k, .]||_2^2) by central
    # differences of A at u_0 +- h psi_k (h = 1e-4), and the second-order remainder of A along the worst direction at eta = 1e-3
    res["A_variation"] = {}
    for d in [dd for dd in a.d if dd in a.dvar]:
        Y = PM.test_points(d)
        J0 = sp1.jets3d(Y)
        A0 = PM.linearized_A(sp1, Y, Y, a.w, 0.0, jets=J0)
        hh = 1e-4
        nrm2 = []
        Dk_list = []
        for j in range(len(Y)):
            for aa in range(4):
                bj = PM.bump_jets(Y, Y[j], a.w, aa)
                if not np.any(np.abs(bj[0]) + np.abs(bj[2]).sum(axis=(1, 2)) > 0):
                    nrm2.append(0.0); continue
                Jp = tuple(J0[q] + hh * bj[q] for q in range(4)); Jm = tuple(J0[q] - hh * bj[q] for q in range(4))
                Dk = (PM.linearized_A(sp1, Y, Y, a.w, 0.0, jets=Jp) - PM.linearized_A(sp1, Y, Y, a.w, 0.0, jets=Jm)) / (2 * hh)
                nrm2.append(float(np.linalg.norm(Dk, 2)))
        nrm2 = np.array(nrm2)
        LA = float(np.sqrt((nrm2 ** 2).sum()))
        kmax = int(np.argmax(nrm2))
        bj = PM.bump_jets(Y, Y[kmax // 4], a.w, kmax % 4)
        rem = {}
        for eta in (1e-3, 1e-2):
            Jp = tuple(J0[q] + eta * bj[q] for q in range(4))
            Ae = PM.linearized_A(sp1, Y, Y, a.w, 0.0, jets=Jp)
            Jh = tuple(J0[q] + 0.5 * eta * bj[q] for q in range(4))
            rem["eta_%g" % eta] = {"norm_dA": float(np.linalg.norm(Ae - A0, 2)), "first_order_pred": float(eta * nrm2[kmax]),
                                   "half_step_ratio": float(np.linalg.norm(Ae - A0, 2) / max(np.linalg.norm(PM.linearized_A(sp1, Y, Y, a.w, 0.0, jets=Jh) - A0, 2), 1e-300))}
        res["A_variation"]["d%d" % d] = {"L_A_first_order": LA, "max_single_direction": float(nrm2.max()), "worst_direction": [int(kmax // 4), int(kmax % 4)], "remainder_check": rem}
        save["DA_norms_d%d" % d] = nrm2
        log("A variation d %d: %s" % (d, json.dumps(res["A_variation"]["d%d" % d])))
    # Killing directions (full fields, not in U): c = (0.3, -0.2, 0.5), omega = (0.1, -0.2, 0.3); b = 0 and b = 1
    Y = PM.test_points(2)
    N0, dN0, dv0, ddv0 = sp1.jets3d(Y)
    kil = {}
    for b in (0.0, 1.0):
        for name in ("translation", "rotation"):
            hcs = 1e-30
            dv = dv0.astype(complex); ddv = ddv0.astype(complex)
            if name == "rotation":
                om = np.array([0.1, -0.2, 0.3])
                # v = omega x y: d_i v_j = eps_jki omega_k ; constant gradient
                Eps = np.zeros((3, 3, 3)); Eps[0, 1, 2] = Eps[1, 2, 0] = Eps[2, 0, 1] = 1; Eps[0, 2, 1] = Eps[2, 1, 0] = Eps[1, 0, 2] = -1
                Gm = np.einsum("jki,k->ij", Eps, om)
                dv = dv + 1j * hcs * Gm[None]
            rho, J = PM.laws(N0.astype(complex), dN0.astype(complex), dv, ddv, b)
            kil["b%d_%s" % (int(b), name)] = {"drho_maxabs": float(np.max(np.abs(rho.imag / hcs))), "dJ_maxabs": float(np.max(np.abs(J.imag / hcs)))}
    res["killing"] = kil
    log("killing: %s" % json.dumps(kil))
    # ---------------------------------------------------------------- (B1) continuum linearized rows: the inverse on L^2 and sup
    hgrid = 1.0 / 32.0
    rg = np.arange(0.0, rc + 1e-9, hgrid)
    if len(rg) % 2 == 0:
        rg = np.arange(0.0, rc + hgrid / 2 + 1e-9, hgrid)
    n = len(rg)
    N = sp1.N(rg); v = sp1.v(rg); rho = sp1.rho(rg); J = sp1.J(rg)
    # cumulative trapezoid matrix C: (C f)(r_i) = int_0^{r_i} f ; tail T: int_{r_i}^{r_c} f
    C = np.zeros((n, n))
    for i in range(1, n):
        C[i, :i + 1] = hgrid
        C[i, 0] -= hgrid / 2; C[i, i] -= hgrid / 2
    T = C[-1][None, :] - C
    rs = np.where(rg > 0, rg, 1.0); vs = np.where(rg > 0, v, -1.0)
    # unknowns x = (dN, dv); sources y = (drho, dJ); equations:
    #   dv = diag(1/(2 r v)) C diag(8 pi g r^2) (2 N rho dN + N^2 drho)
    #   dN = N^2 T diag(4 pi g r) (dJ/v - J dv / v^2)          [dN = -N^2 d(1/N), d(1/N) = -T(...)]
    Dv = np.diag(np.where(rg > 0, 1.0 / (2 * rs * vs), 0.0)) @ C @ np.diag(8 * np.pi * G * rg ** 2)
    DN = np.diag(N ** 2) @ T @ np.diag(np.where(rg > 0, 4 * np.pi * G * rg, 0.0))
    # x = M_xx x + M_xy y
    Mxx = np.zeros((2 * n, 2 * n)); Mxy = np.zeros((2 * n, 2 * n))
    Mxx[n:, :n] = Dv @ np.diag(2 * N * rho)
    Mxy[n:, :n] = Dv @ np.diag(N ** 2)
    Mxx[:n, n:] = DN @ np.diag(np.where(rg > 0, -J / vs ** 2, 0.0))
    Mxy[:n, n:] = DN @ np.diag(np.where(rg > 0, 1.0 / vs, 0.0))
    Minv = np.linalg.solve(np.eye(2 * n) - Mxx, Mxy)      # sources -> fields
    wq = np.full(n, hgrid); wq[0] = wq[-1] = hgrid / 2
    wL = np.concatenate([4 * np.pi * rg ** 2 * wq] * 2)
    sw = np.sqrt(wL); isw = np.where(sw > 0, 1.0 / np.where(sw > 0, sw, 1.0), 0.0)
    ML2 = sw[:, None] * Minv * isw[None, :]
    keep = sw > 0
    sL2 = np.linalg.svd(ML2[np.ix_(keep, keep)], compute_uv=False)
    blocks = {"dN_from_drho": Minv[:n, :n], "dN_from_dJ": Minv[:n, n:], "dv_from_drho": Minv[n:, :n], "dv_from_dJ": Minv[n:, n:]}
    kk = np.where(keep[:n])[0]
    bn = {"dN_from_drho": ML2[np.ix_(kk, kk)], "dN_from_dJ": ML2[np.ix_(kk, n + kk)], "dv_from_drho": ML2[np.ix_(n + kk, kk)], "dv_from_dJ": ML2[np.ix_(n + kk, n + kk)]}
    res["continuum_inverse"] = {"grid_h": hgrid, "r_c": rc, "L2_norm_float": float(sL2[0]),
                                "L2_block_norms": {k_: float(np.linalg.norm(B_, 2)) for k_, B_ in bn.items()},
                                "sup_norm_float": float(np.max(np.abs(Minv).sum(axis=1))),
                                "sup_block_norms": {"dN_from_drho": float(np.abs(Minv[:n, :n]).sum(axis=1).max()), "dN_from_dJ": float(np.abs(Minv[:n, n:]).sum(axis=1).max()),
                                                    "dv_from_drho": float(np.abs(Minv[n:, :n]).sum(axis=1).max()), "dv_from_dJ": float(np.abs(Minv[n:, n:]).sum(axis=1).max())},
                                "note": "L2 on L^2(4 pi r^2 dr) over [0, r_c] (trapezoid weights); sup = max row sum of the discretized operator (weights inside C, T)"}
    # resolution check of the L^2 norm at h/2
    log("continuum inverse: %s" % json.dumps({k: res["continuum_inverse"][k] for k in ("L2_norm_float", "L2_block_norms", "sup_norm_float")}))
    # ---------------------------------------------------------------- (B2) Dirichlet band Galerkin
    from numpy.polynomial.legendre import leggauss
    xg, wg = leggauss(400)
    # composite Gauss on [0, r_c] in 24 panels
    edges = np.linspace(0, rc, 25)
    rq = np.concatenate([0.5 * (edges[i + 1] - edges[i]) * (xg + 1) + edges[i] for i in range(24)])
    wqq = np.concatenate([0.5 * (edges[i + 1] - edges[i]) * wg for i in range(24)])
    meas = 4 * np.pi * rq ** 2 * wqq
    Nq = sp1.N(rq); N1q = sp1.N(rq, 1); vq = sp1.v(rq); v1q = sp1.v(rq, 1); rhoq = sp1.rho(rq); Jq = sp1.J(rq)
    band = {}
    for d in a.d:
        kc = np.pi / d
        nb = int(np.floor(kc * rc / np.pi + 1e-9))
        ks = np.arange(1, nb + 1) * np.pi / rc
        j0 = np.stack([spherical_jn(0, k * rq) for k in ks], axis=1)
        j1 = np.stack([spherical_jn(1, k * rq) for k in ks], axis=1)
        dj0 = np.stack([-k * spherical_jn(1, k * rq) for k in ks], axis=1)
        dj1 = np.stack([k * spherical_jn(1, k * rq, derivative=True) for k in ks], axis=1)
        def orth(F):
            Gm = F.T @ (meas[:, None] * F)
            L = np.linalg.cholesky(Gm)
            return np.linalg.inv(L).T, Gm     # F @ Linv^T orthonormal
        TN, GN = orth(j0); Tv, Gv = orth(j1)
        phiN = j0 @ TN; dphiN = dj0 @ TN; phiv = j1 @ Tv; dphiv = dj1 @ Tv
        # forward linearized laws on the basis
        # drho = (2 r v dv)'/(8 pi g r^2 N^2) - 2 rho dN / N ; dJ = -(N' dv + v dN')/(4 pi g r N^2) - 2 J dN / N
        dr_from_v = (2 * vq[:, None] * phiv + 2 * rq[:, None] * v1q[:, None] * phiv + 2 * rq[:, None] * vq[:, None] * dphiv) / (8 * np.pi * G * rq[:, None] ** 2 * Nq[:, None] ** 2)
        dr_from_N = -2 * rhoq[:, None] * phiN / Nq[:, None]
        dJ_from_v = -(N1q[:, None] * phiv) / (4 * np.pi * G * rq[:, None] * Nq[:, None] ** 2)
        dJ_from_N = -(vq[:, None] * dphiN) / (4 * np.pi * G * rq[:, None] * Nq[:, None] ** 2) - 2 * Jq[:, None] * phiN / Nq[:, None]
        # project: energy rows on the scalar band (phiN family), momentum rows on the vector band (phiv family)
        Ab = np.block([[phiN.T @ (meas[:, None] * dr_from_N), phiN.T @ (meas[:, None] * dr_from_v)],
                       [phiv.T @ (meas[:, None] * dJ_from_N), phiv.T @ (meas[:, None] * dJ_from_v)]])
        s = np.linalg.svd(Ab, compute_uv=False)
        enc = VF.sigma_min_enclosure(Ab)
        ivc = VF.iv_sigma_min_check(Ab) if a.iv else None
        band["d%d" % d] = {"k_c": kc, "n_modes_each": nb, "dim": 2 * nb, "float_sigma_min": float(s[-1]), "float_sigma_max": float(s[0]),
                           "cond": float(s[0] / s[-1]), "enclosure": enc, "iv_check": ivc,
                           "orthonormality_defect": float(max(np.abs(phiN.T @ (meas[:, None] * phiN) - np.eye(nb)).max(), np.abs(phiv.T @ (meas[:, None] * phiv) - np.eye(nb)).max())),
                           "block_sigma_min": {"rho_v": float(np.linalg.svd(Ab[:nb, nb:], compute_uv=False)[-1]), "J_N": float(np.linalg.svd(Ab[nb:, :nb], compute_uv=False)[-1])},
                           "H0_k1_over_4pig": float(0.03183 * ks[0] / (4 * np.pi * G))}
        save["Aband_d%d" % d] = Ab
    res["band_galerkin"] = band
    log("band galerkin: %s" % json.dumps({k: {kk: band[k][kk] for kk in ("dim", "float_sigma_min", "float_sigma_max", "cond")} for k in band}))
    res["elapsed_s"] = time.time() - t0
    np.savez(a.out + ".npz", **save)
    json.dump(res, open(a.out + ".json", "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
