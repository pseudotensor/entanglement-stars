#!/usr/bin/env python
"""s2_items34_v1.py (busi, reviewb5 items 3, 4 with 9; PM1 = common/item2_projected_solve.txt). DATA, box CPU, 8 threads.

Per test spacing d (T_d, PM1):
 (G)  Gram of the full vector test family (energy + 3 momentum components per point): enclosure of every eigenvalue
      (Kahan/Parlett residual theorem with the entrywise rounding radius of Gamma).
 (B)  the PM1 background: C_0 = (1 - 2 eps) P(u_0) + eps I + Z, Z = R[res], res = s_g(u_0) - L(eps (I - 2 P(u_0))) on T_d, so that
      L(C_0 - P(u_0)) = s_g(u_0) on T_d exactly (the projected laws at u_0). ||Z|| enclosed (row-sum bound, and dense eigvalsh).
 (V)  full vector-field variations: every basis direction of U (4 per test point): the first-order realization X_j = R[A_d e_j],
      A_d = D s_g[u_0] on U (PM1's linearized projected map); moment check; S = sum_j X_j^2 and the rigorous bound
      sup_{|c|<=1} ||sum c_j X_j|| <= ||S||^(1/2); its clock-only and flow-only parts; the computed worst direction (lower bound).
      Singular values of A_d (floating; item 2 encloses them) and the near-Killing directions (PM1 note).
 (M)  item 4: on the neighbourhood B_eta = {u_0 + sum c_j phi_j : |c|_2 <= eta},
      min(lambda_min C(u), 1 - lambda_max C(u)) >= eps - ||Z|| - 2 eps e/(2 g0 - e) - eta ||S||^(1/2) - ||R[Rem]||,
      e = eta ||sum_j h[phi_j]^2||^(1/2) (Davis-Kahan sin-theta with the enclosed gap g0 of h[u_0]),
      Rem the nonlinear remainder of the laws, enclosed by interval Hessians (mpmath iv) on the jet box. eta_max: largest eta with
      the bound > 0. Also for the clock-only and flow-only subspaces, and for b = +-1.
 (F)  item 4 evaluated: exact C(u) (eigh of h[u], exact nonlinear laws) at finite amplitude along the worst directions and random
      ones, lambda_min and 1 - lambda_max by dense eigvalsh (floating), against the bound.
Outputs (small): <out>/s2_items34_d<d>.json and .npz. Usage: s2_items34_v1.py --d 4 [--finite N]."""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "8"
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N345 = os.path.join(HERE, "..", "..", "n345")
CODE = "<internal>/code16d"
sys.path.insert(0, HERE); sys.path.insert(0, N345); sys.path.insert(0, CODE)
import torch  # noqa: E402
torch.set_num_threads(8)
import engine_ops_v1 as OPS  # noqa: E402
import engine_fields_v4 as F4  # noqa: E402
import n3_background_v1 as N3  # noqa: E402
import items34_lib as L  # noqa: E402

OUT = "<internal>/items34"
SCR = OUT + "/scratch"
M_, G_, EPS, ELL, RB = 0.2, 0.25, 1e-4, 2.0, 8
W_BUMP = 2 * ELL


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def T(x):
    return torch.from_numpy(np.ascontiguousarray(x))


def dense_eigvalsh(A):
    return torch.linalg.eigvalsh(T(A)).numpy()


def sparse_top(A, k=1):
    """largest |eigenvalue| and vector of a sparse Hermitian matrix (floating, ARPACK)."""
    v0 = np.random.default_rng(3).standard_normal(A.shape[0]) + 0j
    try:
        vals, vecs = spl.eigsh(A, k=k, which="LM", tol=1e-10, maxiter=20000, v0=v0)
    except Exception:
        vals, vecs = np.linalg.eigh(A.toarray())
    i = int(np.argmax(np.abs(vals)))
    return float(vals[i]), vecs[:, i]


def gram_enclosure(Gm, rad):
    lam, X = np.linalg.eigh(Gm)
    radF = float(np.linalg.norm(rad))
    enc = L.eig_enclosure_dense(Gm, lam, X, A_rad_F=radF)
    r = enc["r"]
    return lam, X, {"lam_min_float": float(lam[0]), "lam_max_float": float(lam[-1]), "radius": r,
                    "lam_min_lower": float(lam[0] - r), "lam_max_upper": float(lam[-1] + r),
                    "cond_float": float(lam[-1] / lam[0]) if lam[0] > 0 else None,
                    "certified_positive": bool(lam[0] - r > 0), "enclosure_parts": enc}


def solve_err_bound(Gm, Ysol, Rhs, lam_lo):
    """columnwise |Gamma^-1 Rhs - Ysol|_2 <= |Rhs - Gamma Ysol|_2 / lam_lo (+ rounding of the residual)."""
    resid = Rhs - Gm @ Ysol
    rnd = L.gam(Gm.shape[0] + 2) * (np.abs(Gm) @ np.abs(Ysol) + np.abs(Rhs))
    col = np.sqrt(((np.abs(resid) + rnd) ** 2).sum(0))
    return col / lam_lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--finite", type=int, default=0, help="number of finite-amplitude directions (0: none)")
    ap.add_argument("--amps", default="0.5,1,2,4,8")
    ap.add_argument("--worst-iters", type=int, default=25)
    ap.add_argument("--dirs", default="", help="comma list of finite-amplitude directions (worst_all,worst_N,worst_v,random); empty = all")
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--precise", action="store_true", help="Gram in long double (structured), certified by Cholesky; long-double solves")
    a = ap.parse_args()
    t0 = time.time()
    d = a.d
    tag = "d%d" % d
    out = os.path.join(OUT, "s2_items34_" + tag + a.tag_suffix)
    S1 = np.load(os.path.join(OUT, "s1_background_v1.npz"))
    E1 = np.load(os.path.join(SCR, "eigh_hu0.npz"))
    e0, U0 = E1["e"], E1["U"]
    P = N3.BallParent(OPS, RB, M_)
    n = P.dim
    assert np.allclose(P.X, S1["X"])
    N_site, v_vec = S1["N_site"], S1["v_vec"]
    H0 = P.hm_sparse(N_site, v_vec)
    res = {"version": "s2_items34_v1", "d": d, "PM1": "common/item2_projected_solve.txt v1", "eps": EPS, "m": M_, "G": G_, "ell": ELL,
           "w_bump": W_BUMP, "provenance": {"script_md5": md5(__file__), "lib_md5": md5(L.__file__), "s1_npz_md5": md5(os.path.join(OUT, "s1_background_v1.npz"))}}

    # ------------------------------------------------------------------ gap of h[u_0], enclosed
    enc_h = L.eig_enclosure_sparseH(H0, e0, U0, torch)
    g0_float = float(np.abs(e0).min())
    g0_lo = g0_float - enc_h["r"]
    dH = (H0 - P.h).tocsr()
    ndH_up = L.inf_norm(dH)
    ndH_float = float(np.max(np.abs(dense_eigvalsh(dH.toarray()))))
    res["gap"] = {"g0_float": g0_float, "radius": enc_h["r"], "g0_lower": g0_lo, "n_neg": int((e0 < 0).sum()), "parts": enc_h,
                  "h0_gap_exact": M_, "norm_h_u0_minus_h0_rowsum_upper": ndH_up, "norm_h_u0_minus_h0_float": ndH_float,
                  "path_gapped_rigorous": bool(ndH_up < M_ + g0_lo),
                  "path_note": "gap(h_0 + s dH) >= max(m - s||dH||, g0 - (1 - s)||dH||) >= (m + g0 - ||dH||)/2 > 0 for all s in [0,1]; |h_0| >= m exactly (h_0^2 = m^2 + D^2)",
                  "path_min_gap_lower": float((M_ + g0_lo - ndH_up) / 2)}
    print("gap", res["gap"], flush=True)

    # ------------------------------------------------------------------ tests, vertices, Gram
    Y = S1[tag + "_Y"]; npt = len(Y); nT = 4 * npt
    Wt = L.test_weights(Y, P.X, ELL)
    V = L.vertex_list(P, Wt)
    rows, cols, Vp = L.pattern_matrix(V, n)
    Gm, Grad = L.gram(Vp)
    lam, Xg, genc = gram_enclosure(Gm, Grad)
    PG = None
    if a.precise:
        PG = L.PreciseGram(P, Wt)
        cert = PG.certify()
        Gp = PG.dense_double()
        genc = dict(genc)
        genc["precise_blocks"] = cert
        genc["precise_vs_double_gram_max_abs"] = float(np.abs(Gp - Gm).max())
        genc["lam_min_lower_double_route"] = genc["lam_min_lower"]
        genc["lam_min_lower"] = float(PG.lam_lo)
        genc["certified_positive"] = bool(PG.lam_lo > 0)
        Gm = Gp
    blk = Gm.reshape(npt, 4, npt, 4)
    res["gram"] = dict(genc, n_points=int(npt), n_tests=int(nT), lam_min_energy_block=float(np.linalg.eigvalsh(blk[:, 0, :, 0])[0]),
                       lam_min_momentum_block=float(np.linalg.eigvalsh(blk[:, 1:, :, 1:].reshape(3 * npt, 3 * npt))[0]),
                       max_energy_momentum_coupling_over_max_diag=float(np.abs(blk[:, :1, :, 1:]).max() / np.abs(np.diag(Gm)).max()),
                       n_eig_below_1em12_of_max=int((lam < 1e-12 * lam[-1]).sum()))
    print("gram", res["gram"], flush=True)
    lam_lo = genc["lam_min_lower"]
    certified = lam_lo > 0
    lam_use = lam_lo if certified else max(lam[0], 1e-300)
    SA = sp.csr_matrix((n, n), dtype=np.complex128)
    for A in V:
        SA = SA + A @ A
    nSA = L.inf_norm(SA)
    maxA = max(L.inf_norm(A) for A in V)

    def gsolve(X):
        """Gamma^-1 X and, per column, a rigorous bound of the OPERATOR error ||sum_mu A_mu (Y - Gamma^-1 X)_mu||_op of the exact vertices:
        |exact residual|_2 / sqrt(lambda_lo) (HS bound) plus the stored-vertex rounding 2u sum_mu |y_mu| max_mu ||A_mu||."""
        if PG is not None:
            Yv, rn = PG.solve(X)
        else:
            Yv = np.linalg.solve(Gm, X)
            X2 = X if np.ndim(X) == 2 else X[:, None]; Y2 = Yv if np.ndim(Yv) == 2 else Yv[:, None]
            r = X2 - Gm @ Y2
            rnd = L.gam(Gm.shape[0] + 2) * (np.abs(Gm) @ np.abs(Y2) + np.abs(X2)) + Grad @ np.abs(Y2)
            rn = np.sqrt(((np.abs(r) + rnd) ** 2).sum(0)) * (1 + 1e-12)
            rn = rn if np.ndim(X) == 2 else float(rn[0])
        vert = 2 * 2.0 ** -53 * np.abs(Yv).sum(0) * maxA
        return Yv, rn / np.sqrt(lam_use) + vert

    # ------------------------------------------------------------------ the background
    rho_s, J_s = S1["rho_site"], S1["J_site"]
    mom = np.zeros((npt, 4)); mom[:, 0] = Wt @ rho_s
    for i in range(3):
        mom[:, 1 + i] = Wt @ J_s[:, i]
    mom = mom.ravel()
    jet0 = S1[tag + "_J5"]
    laws = {b: L.Laws(b, G_) for b in (0.0, 1.0, -1.0)}
    sg0 = laws[0.0].values(jet0).ravel()
    sg0_10 = laws[0.0].values(S1[tag + "_J10"]).ravel()
    resid = sg0 - mom
    yZ, yZ_err = gsolve(resid)
    yZ_err = float(yZ_err)
    Z = L.op_from_coef(rows, cols, Vp, yZ.astype(np.complex128), n)
    nZ_row = L.inf_norm(Z)
    Zd_ = Z.toarray()
    lz, Xz = torch.linalg.eigh(T(Zd_)); lz = lz.numpy(); Xz = Xz.numpy()
    encZ = L.eig_enclosure_sparseH(Z, lz, Xz, torch)
    del Xz, Zd_
    nZ_float = float(np.max(np.abs(lz)))
    nZ_eig_up = nZ_float + encZ["r"] + yZ_err
    nZ_row_up = nZ_row + yZ_err
    # the projector rounding: P_exact vs U_o U_o^* (sin-theta with the residual), moved into the residual absorbed by R
    nZ_up = min(nZ_eig_up, nZ_row_up)
    rho_mean = float(np.mean(sg0[0::4][np.sqrt((Y ** 2).sum(1)) <= RB - 3]))
    res["background"] = {
        "construction": "C_0 = (1 - 2 eps) P(u_0) + eps I + R[s_g(u_0) - L(eps (I - 2P(u_0)))], so L(C_0 - P(u_0)) = s_g(u_0) on T_d",
        "rho_g_at_tests_min": float(sg0[0::4].min()), "rho_g_bulk_mean": rho_mean,
        "rho_g_vs_rows_sph_max_rel": float(np.max(np.abs(sg0[0::4] - S1[tag + "_rho_sph"])) / np.abs(S1[tag + "_rho_sph"]).max()),
        "jet_step_change_max_abs_law": float(np.max(np.abs(sg0 - sg0_10))),
        "res_energy_max_abs": float(np.abs(resid[0::4]).max()), "res_energy_max_rel": float(np.abs(resid[0::4]).max() / np.abs(sg0[0::4]).max()),
        "res_momentum_max_abs": float(np.abs(resid.reshape(npt, 4)[:, 1:]).max()),
        "res_norm2": float(np.linalg.norm(resid)),
        "norm_Z_upper": nZ_up, "norm_Z_rowsum_upper": nZ_row_up, "norm_Z_eig_enclosure_upper": nZ_eig_up, "Z_eig_radius": encZ["r"], "norm_Z_float": nZ_float, "solve_err_Z_op": yZ_err,
        "margin_C0_lower": float(EPS - nZ_up), "margin_C0_float": float(EPS - nZ_float),
        "strictly_mixed_certified": bool(certified and EPS - nZ_up > 0),
        "note": "spectrum of (1-2eps)P + eps I is {eps, 1-eps} exactly; Weyl: min(lambda_min C_0, 1 - lambda_max C_0) >= eps - ||Z||"}
    print("background", res["background"], flush=True)

    # ------------------------------------------------------------------ full vector-field variations
    Zc = Y.copy()
    Jb = L.basis_jets(Y, Zc, W_BUMP)                                  # (npt, nU, NJ)
    nU = Jb.shape[1]
    Fsite = L.basis_site_fields(P.X, Zc, W_BUMP, ELL, F4.kernels)    # (npt, nsite)
    _dd = np.linspace(0.0, 14.0, 57)
    res["F_quadrature_check"] = float(np.max(np.abs(L.psi_backavg(_dd, W_BUMP, ELL, F4.kernels, nq=96) - L.psi_backavg(_dd, W_BUMP, ELL, F4.kernels, nq=192))) /
                                      np.max(np.abs(L.psi_backavg(_dd, W_BUMP, ELL, F4.kernels, nq=192))))
    out_b = {}
    for b in (0.0, 1.0, -1.0):
        gr = laws[b].grads(jet0)                                      # (npt, 4, NJ)
        Ad = np.einsum("qtk,qjk->qtj", gr, Jb).reshape(nT, nU)
        sv = np.linalg.svd(Ad, compute_uv=False)
        Ysol, Yerr = gsolve(Ad)
        Vab = abs(Vp)
        S_all = sp.csr_matrix((n, n), dtype=np.complex128); S_N = S_all.copy(); S_v = S_all.copy()
        A_all = sp.csr_matrix((n, n)); A_N = A_all.copy(); A_v = A_all.copy()
        mom_err = 0.0
        Xn = np.zeros(nU)
        rnd_F2 = np.zeros(nU)
        for j in range(nU):
            Xj = L.op_from_coef(rows, cols, Vp, Ysol[:, j].astype(np.complex128), n)
            X2 = Xj @ Xj
            Xa = abs(Xj); Xa2 = Xa @ Xa
            S_all = S_all + X2; A_all = A_all + Xa2
            if j % 4 == 0:
                S_N = S_N + X2; A_N = A_N + Xa2
            else:
                S_v = S_v + X2; A_v = A_v + Xa2
            Xn[j] = L.inf_norm(Xj)
            rnd_F2[j] = (L.gam(nT + 2) * np.linalg.norm(Vab @ np.abs(Ysol[:, j]))) ** 2
            if j % 37 == 0:
                mj = np.array([np.real((Vp[:, mu].conj().T @ (Vp @ Ysol[:, j]))[0]) for mu in range(0, nT, max(1, nT // 64))])
                tgt = Ad[::max(1, nT // 64), j]
                mom_err = max(mom_err, float(np.max(np.abs(mj - tgt)) / max(np.abs(Ad[:, j]).max(), 1e-300)))
        kX = int(np.diff(L.op_from_coef(rows, cols, Vp, np.ones(nT, complex), n).indptr).max())

        jN = np.arange(0, nU, 4); jv = np.array([j for j in range(nU) if j % 4])

        def sq(Sm, Am, js):
            # ||S(X^)||_inf <= ||fl S||_inf + gamma(kX nU + 2) || sum |X^_j|^2 ||_inf ; + the X^ rounding and the solve (operator) errors
            # of the subspace's columns (Cauchy-Schwarz: ||sum c_j E_j|| <= |c| (sum_j ||E_j||^2)^(1/2))
            sN = L.inf_norm(Sm) + L.gam(kX * nU + 2) * L.inf_norm(Am)
            return float(np.sqrt(sN)) + float(np.sqrt(rnd_F2[js].sum())) + float(np.linalg.norm(Yerr[js]))
        lin_all, lin_N, lin_v = sq(S_all, A_all, np.arange(nU)), sq(S_N, A_N, jN), sq(S_v, A_v, jv)
        lin_parts = {"sqrt_S_all": float(np.sqrt(L.inf_norm(S_all))), "rounding_all": float(np.sqrt(rnd_F2.sum())), "solve_all": float(np.linalg.norm(Yerr))}
        # computed worst direction (lower bound): alternate c ~ (<psi, X_j psi>)_j, psi = top eigvec of sum c_j X_j
        rng = np.random.default_rng(7)
        best = {}
        for sub, idx in (("all", np.arange(nU)), ("N", np.arange(0, nU, 4)), ("v", np.array([j for j in range(nU) if j % 4]))):
            c = rng.standard_normal(len(idx)); c /= np.linalg.norm(c)
            val = 0.0
            for it in range(a.worst_iters if b == 0.0 else 6):
                cf = np.zeros(nU); cf[idx] = c
                Xc = L.op_from_coef(rows, cols, Vp, (Ysol @ cf).astype(np.complex128), n)
                lv, psi = sparse_top(Xc)
                val = abs(lv)
                # <psi, X_j psi> = sum_mu Y_mu,j <psi, A_mu psi>
                apsi = np.array([np.real(np.vdot(psi, A @ psi)) for A in V])
                g = (Ysol[:, idx].T @ apsi)
                if not np.linalg.norm(g) > 0:
                    break
                cnew = g / np.linalg.norm(g)
                if np.linalg.norm(cnew - c) < 1e-9:
                    c = cnew; break
                c = cnew
            cf = np.zeros(nU); cf[idx] = c
            best[sub] = {"norm_lower": val, "c": cf}
        # h-basis
        if b == 0.0:
            Sh = sp.csr_matrix((n, n), dtype=np.complex128); Sh_N = Sh.copy(); Sh_v = Sh.copy()
            Ah = sp.csr_matrix((n, n)); Ah_N = Ah.copy(); Ah_v = Ah.copy()
            for j in range(nU):
                hj = L.h_of_basis(P, Fsite, j)
                h2 = hj @ hj; ha = abs(hj); ha2 = ha @ ha
                Sh = Sh + h2; Ah = Ah + ha2
                if j % 4 == 0:
                    Sh_N = Sh_N + h2; Ah_N = Ah_N + ha2
                else:
                    Sh_v = Sh_v + h2; Ah_v = Ah_v + ha2
            # F (the back-averaged bumps) by 96-node Gauss-Legendre: its quadrature error is below 1e-13 relative (checked at nq 192,
            # see 'F_quadrature_check'), carried as the factor (1 + 1e-10)
            lh = {k: float(np.sqrt(L.inf_norm(Sm) + L.gam(40 * nU) * L.inf_norm(Am))) * (1 + 1e-10)
                  for k, Sm, Am in (("all", Sh, Ah), ("N", Sh_N, Ah_N), ("v", Sh_v, Ah_v))}
        # killing-like near-null directions of A_d: smallest singular vectors and their flow content
        out_b[b] = {"Ad": Ad, "Ysol": Ysol, "sv": sv, "lin": {"all": lin_all, "N": lin_N, "v": lin_v}, "lh": lh, "best": best,
                    "Xn_max": float(Xn.max()), "Xn_N_max": float(Xn[0::4].max()), "Xn_v_max": float(np.delete(Xn, np.arange(0, nU, 4)).max()),
                    "moment_check_rel": mom_err, "Yerr_F": float(np.linalg.norm(Yerr)), "lin_parts": lin_parts}
        print("b", b, {k: out_b[b][k] for k in ("lin", "lh", "Xn_max", "moment_check_rel")}, {s: best[s]["norm_lower"] for s in best},
              "sv min/max", sv[-1], sv[0], flush=True)

    # ------------------------------------------------------------------ item 4: the rigorous neighbourhood
    # ||R[e_mu]|| upper bounds (row sums of sum_nu A_nu (Gamma^-1)_nu,mu, plus the solve error): ||R[x]|| <= sum_mu |x_mu| rho_mu
    Ginv, Gerr = gsolve(np.eye(nT))
    rho_mu = np.array([L.inf_norm(L.op_from_coef(rows, cols, Vp, Ginv[:, mu].astype(np.complex128), n)) for mu in range(nT)]) + Gerr
    res["R_unit_norms"] = {"max": float(rho_mu.max()), "min": float(rho_mu.min()), "energy_max": float(rho_mu[0::4].max()),
                           "momentum_max": float(np.delete(rho_mu, np.arange(0, nT, 4)).max())}
    beta_unit = np.sqrt((Jb ** 2).sum(1))                             # (npt, NJ): |row of the jet map|_2 per test point
    beta_sub = {"all": beta_unit, "N": np.sqrt((Jb[:, 0::4, :] ** 2).sum(1)), "v": np.sqrt((np.delete(Jb, np.arange(0, nU, 4), axis=1) ** 2).sum(1))}

    def rem_bound(b, sub, eta):
        bt = eta * beta_sub[sub]
        Rm = np.zeros((npt, 4))
        for q in range(npt):
            for t_ in range(4):
                Rm[q, t_] = laws[b].remainder_bound(jet0[q], bt[q], t_)
        return Rm

    def margin(b, sub, eta, Rref, eta_ref):
        lin = out_b[b]["lin"][sub]; lh = out_b[b]["lh"][sub]
        e_ = eta * lh
        if e_ >= g0_lo:
            return -1.0, {}
        dk = e_ / (2 * g0_lo - e_)
        Rm = Rref * (eta / eta_ref) ** 2
        nRrem = min(float(np.linalg.norm(Rm)) / np.sqrt(lam_use), float(np.sum(Rm.ravel() * rho_mu))) * (1 + 1e-12)
        parts = {"Z": nZ_up, "dP": 2 * EPS * dk, "lin": eta * lin, "rem": nRrem}
        return EPS - sum(parts.values()), parts

    nb = {}
    for b in (0.0, 1.0, -1.0):
        for sub in ("all", "N", "v"):
            lin = out_b[b]["lin"][sub]
            eta_lin = max((EPS - nZ_up), 0.0) / lin if lin > 0 else np.inf
            if not (eta_lin > 0) or not certified:
                nb["%g_%s" % (b, sub)] = {"eta_max_lower": 0.0, "reason": "no margin at the background or Gram not certified"}
                continue
            eta_ref = 1.5 * eta_lin
            Rref = rem_bound(b, sub, eta_ref)
            lo, hi = 0.0, eta_ref
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if margin(b, sub, mid, Rref, eta_ref)[0] > 0:
                    lo = mid
                else:
                    hi = mid
            mval, parts = margin(b, sub, lo, Rref, eta_ref)
            nb["%g_%s" % (b, sub)] = {"eta_max_lower": lo, "eta_linear_only": eta_lin, "parts_at_eta_max": parts, "margin_at_eta_max": mval,
                                      "rem_ref_max": float(Rref.max()), "eta_ref": eta_ref,
                                      "field_peak_at_tests_per_component_upper": float(lo * np.sqrt((Jb[:, 0::4, 0] ** 2).sum(1)).max())}
            print("nbhd", b, sub, nb["%g_%s" % (b, sub)], flush=True)
    res["neighbourhood"] = nb

    # ------------------------------------------------------------------ item 4 evaluated at finite amplitude (b = 0)
    fin = []
    if a.finite and certified:
        Uo0 = T(U0[:, e0 < 0])
        P0 = (Uo0 @ Uo0.conj().T).numpy()
        del Uo0
        dirs = [("worst_all", out_b[0.0]["best"]["all"]["c"]), ("worst_N", out_b[0.0]["best"]["N"]["c"]), ("worst_v", out_b[0.0]["best"]["v"]["c"])]
        rng = np.random.default_rng(11)
        for k in range(max(0, a.finite - 3)):
            c = rng.standard_normal(nU); dirs.append(("random%d" % k, c / np.linalg.norm(c)))
        eta_sub = {"worst_all": nb["0_all"]["eta_max_lower"], "worst_N": nb["0_N"]["eta_max_lower"], "worst_v": nb["0_v"]["eta_max_lower"]}
        dirsel = a.dirs.split(",") if a.dirs else None
        Zd = Z.toarray()
        for name, c in dirs:
            if dirsel and name not in dirsel and not (name.startswith("random") and "random" in dirsel):
                continue
            eta_all = eta_sub.get(name, nb["0_all"]["eta_max_lower"])
            sub_name = {"worst_N": "N", "worst_v": "v"}.get(name, "all")
            for amp in ([float(x) for x in a.amps.split(",")] if name in ("worst_all", "worst_N") else [1.0, 4.0]):
                eta = amp * max(eta_all, 1e-300)
                cc = eta * c
                Hu = (H0 + L.h_of_field(P, Fsite, cc)).toarray()
                eu, Uu = torch.linalg.eigh(T(Hu))
                eu = eu.numpy()
                Uo = Uu[:, torch.from_numpy(eu < 0)]
                Pu = (Uo @ Uo.conj().T).numpy(); del Uo, Uu
                jet = jet0 + np.einsum("qjk,j->qk", Jb, cc)
                ds = laws[0.0].values(jet).ravel() - sg0
                yds, _ = gsolve(ds)
                Rds = L.op_from_coef(rows, cols, Vp, yds.astype(np.complex128), n).toarray()
                Yop = 2 * EPS * (Pu - P0) + Zd + Rds
                Cu = (1 - 2 * EPS) * Pu + EPS * np.eye(n) + Yop
                ev = dense_eigvalsh(Cu)
                nY = float(np.max(np.abs(dense_eigvalsh(Yop))))
                nRds = float(np.max(np.abs(dense_eigvalsh(Rds)))) if name == "worst_all" else None
                ndP = float(np.max(np.abs(dense_eigvalsh(Pu - P0)))) if name == "worst_all" else None
                lin_ds = (out_b[0.0]["Ad"] @ cc)
                bnd, parts = margin(0.0, sub_name, eta, np.zeros((npt, 4)), 1.0) if eta * out_b[0.0]["lh"][sub_name] < g0_lo else (None, {})
                row = {"dir": name, "amp_over_eta_max": amp, "eta": eta, "lambda_min": float(ev[0]), "one_minus_lambda_max": float(1 - ev[-1]),
                       "margin_eval": float(min(ev[0], 1 - ev[-1])), "norm_Y": nY, "norm_R_ds": nRds, "norm_dP": ndP,
                       "gap_hu": float(np.abs(eu).min()), "n_neg_hu": int((eu < 0).sum()),
                       "subspace": sub_name, "nonlinear_ds_rel": float(np.linalg.norm(ds - lin_ds) / max(np.linalg.norm(lin_ds), 1e-300)),
                       "bound_without_remainder": bnd}
                fin.append(row)
                print("finite", row, flush=True)
                del Hu, Pu, Rds, Yop, Cu
    res["finite_amplitude"] = fin

    # ------------------------------------------------------------------ store
    b0 = out_b[0.0]
    res["variations"] = {b: {"sv_Ad_min": float(o["sv"][-1]), "sv_Ad_max": float(o["sv"][0]), "sv_Ad_smallest8": o["sv"][-8:].tolist(),
                             "lin_bound": o["lin"], "lh_bound": o["lh"], "worst_lower": {s: o["best"][s]["norm_lower"] for s in o["best"]},
                             "Xn_max_rowsum": o["Xn_max"], "Xn_N_max": o["Xn_N_max"], "Xn_v_max": o["Xn_v_max"],
                             "moment_check_rel": o["moment_check_rel"], "Yerr_F": o["Yerr_F"], "lin_parts": o["lin_parts"]} for b, o in out_b.items()}
    res["norm_sum_A2_rowsum"] = nSA
    res["elapsed_s"] = time.time() - t0
    np.savez(out + ".npz", Y=Y, Gram=Gm, Gram_rad=Grad, gram_eigs=lam, moments=mom, sg0=sg0, resid=resid, yZ=yZ, jet0=jet0,
             Ad_b0=b0["Ad"], sv_b0=b0["sv"], c_worst_all=b0["best"]["all"]["c"], c_worst_N=b0["best"]["N"]["c"], c_worst_v=b0["best"]["v"]["c"])
    with open(out + ".json", "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
