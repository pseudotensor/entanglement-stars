#!/usr/bin/env python
"""i1_chi_v1.py (busi, foundations reviewb5 item 1 with item 9; agent A). DATA script, box CPU (torch, <= 8 threads), no card.

The reference's static response chi on the N3 ball background (PM1, common/item2_projected_solve.txt):
    chi_XY = Tr(X P'_h[Y]) = 2 Re sum_{a occ, b empty} <a|X|b> <b|Y|a> / (e_a - e_b),   P = 1_(-inf,0)(h[u_0]),
for X in a readout set and Y in a field-direction set of vertex operators {diag(f), O}/2, O in {h_0, Q_x, Q_y, Q_z}:
  shell set   : E_R = {1_shell R, h_0}/2, Jr_R = sum_i {1_shell R xhat_i, Q_i}/2 (the shell totals N3's rows consume; the coupled
                generator's derivative in the shell site fields N_R, v_R is the same set), R the 55 distinct radii (54 for Jr).
  T_d / U_d   : B's tests {G_ell(y_mu - x), h_0 or Q_i}/2 and the back-averaged bumps {Phi(|x - z_j|), h_0 or Q_a}/2 (PM1), d in --d.
  identity    : X = hbar = h[u_0] (exact Tr(hbar P'[Y]) = 0) as an extra readout row in every set.
Item 9 parts computed here: the eigen-decomposition's residual and orthogonality defect (rigorous upper bounds with the a-priori
rounding bound), the backward error delta_bwd (h' = U_hat Lam U_hat^+ exactly diagonal with ||h - h'|| <= delta_bwd), the gap
enclosure, ||h[u_0] - h_0|| and ||h_0||, ||Q_i|| upper bounds (||A||_2 <= ||A^4||_inf^(1/4), exact sparse products with rounding),
||V||_F and || |V| ||_inf per vertex, the per-entry rounding/backward bound on chi, and ||P'[V]||_2 per shell vertex (power iteration,
a lower estimate) against Heinz's bound ||V_oe||/Delta.
Outputs: <out>/i1_chi_v1_<tag>.npz (chi blocks, norms, eigenvalues) and .json (scalars). Never overwrites."""
import os
import sys


def _threads():
    t = "8"
    if "--threads" in sys.argv:
        t = sys.argv[sys.argv.index("--threads") + 1]
    t = str(max(1, min(int(t), 8)))
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = t
    return int(t)


NTH = _threads()
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(NTH)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import pm1_common_v1 as PM  # noqa: E402

U_ROUND = 2.0 ** -53


def gam(n):
    return n * U_ROUND / (1 - n * U_ROUND)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def norm_upper_sparse(A, k=2):
    """||A||_2 <= ||A^(2^k)||_inf^(1/2^k) for Hermitian A (Gelfand; || ||_2 <= || ||_inf for Hermitian), with rounding inflation."""
    B = A.copy().tocsr()
    nnz_row = int(np.diff(B.indptr).max())
    for _ in range(k):
        B = (B @ B).tocsr()
    rs = float(np.asarray(abs(B).sum(axis=1)).ravel().max())
    p = 2 ** k
    absinf = float(np.asarray(abs(A).sum(axis=1)).ravel().max())
    # computed B = A^p + E, |E| <= p gamma_(p nnz) |A|^p elementwise, so ||A^p||_inf <= rs + p gamma || |A| ||_inf^p
    rs_up = (rs + p * gam(4 * p * max(nnz_row, 2) * B.shape[0] ** 0) * absinf ** p) * (1 + 1e-12)
    return float(rs_up ** (1.0 / p)) * (1 + 1e-12)


def vertex_terms_shell(P):
    """the 109 shell vertices as term lists [(O, f, rows)], f per row (length 4 nsite), rows the nonzero rows."""
    terms = []; labels = []
    nr = len(P.runiq)
    for R in range(nr):
        f = np.repeat((P.inv == R).astype(float), 4)
        rows = np.nonzero(f)[0]
        terms.append([(0, f, rows)]); labels.append(("E", float(P.runiq[R])))
    for R in range(1, nr):
        tl = []
        for i in range(3):
            f = np.repeat((P.inv == R) * P.xhat[:, i], 4)
            rows = np.nonzero(f)[0]
            tl.append((1 + i, f, rows))
        terms.append(tl); labels.append(("Jr", float(P.runiq[R])))
    return terms, labels


def vertex_terms_TU(P, d, ell, w):
    Y = PM.test_points(d, P.Rb)
    T = []; labT = []
    for mu, y in enumerate(Y):
        gw = np.repeat(PM.gauss_weights(y, P.X, ell), 4)
        T.append([(0, gw, None)]); labT.append(("rho", mu))
        for i in range(3):
            T.append([(1 + i, gw, None)]); labT.append(("J%d" % i, mu))
    Uset = []; labU = []
    for j, z in enumerate(Y):
        dist = np.sqrt(((P.X - z[None, :]) ** 2).sum(axis=1))
        phi = np.repeat(PM.bump_backavg(dist, w, ell), 4)
        Uset.append([(0, phi, None)]); labU.append(("N", j))
        for a in range(3):
            Uset.append([(1 + a, phi, None)]); labU.append(("v%d" % a, j))
    return Y, T, labT, Uset, labU


def vertex_sparse(P, tl):
    Ops = [P.h] + list(P.Q)
    V = None
    for (o, f, rows) in tl:
        D = sp.diags(f)
        t = 0.5 * (D @ Ops[o] + Ops[o] @ D)
        V = t if V is None else V + t
    return V.tocsr()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, nargs="*", default=[4, 3])
    ap.add_argument("--shell", type=int, default=1)
    ap.add_argument("--bg", default="<internal>/pm1_background_v1.npz")
    ap.add_argument("--nb", type=int, default=128)
    ap.add_argument("--w", type=float, default=4.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", default="<internal>/items128")
    a = ap.parse_args()
    out = os.path.join(a.outdir, "i1_chi_v1_%s" % a.tag)
    if os.path.exists(out + ".json"):
        sys.exit("refusing to overwrite " + out)
    t0 = time.time()
    log = lambda s: print("[%7.0f s] %s" % (time.time() - t0, s), flush=True)
    res = {"version": "i1_chi_v1", "args": vars(a), "script_md5": md5(os.path.abspath(__file__)),
           "pm1_common_md5": md5(PM.__file__), "torch": torch.__version__, "threads": NTH}
    # ---------------------------------------------------------------- background (built once, cached for both agents)
    if os.path.exists(a.bg):
        bg = PM.load_background(a.bg)
        P = bg["P"]
        N_site, v_vec = bg["N_site"], bg["v_vec"]
        log("background loaded from cache %s (md5 %s)" % (a.bg, md5(a.bg)))
        H = P.hm_sparse(N_site, v_vec)
        Ht = torch.from_numpy(H.toarray())
        e, Uc = torch.linalg.eigh(Ht); del Ht
        e = e.numpy(); Uc = Uc.numpy()
        res["eig_vs_cache_max_abs"] = float(np.max(np.abs(e - bg["e"])))
    else:
        bgd = PM.build_background(log=log)
        os.makedirs(os.path.dirname(a.bg), exist_ok=True)
        PM.save_background(bgd, a.bg)
        log("background built and cached: %s md5 %s" % (a.bg, md5(a.bg)))
        P = bgd["P"]; N_site, v_vec = bgd["N_site"], bgd["v_vec"]; e, Uc = bgd["e"], bgd["U"]
        H = P.hm_sparse(N_site, v_vec)
        res["background_hist"] = bgd["hist"]; res["fixed_point_residual"] = bgd["fixed_point_residual"]
    res["bg_md5"] = md5(a.bg)
    n = P.dim
    o = e < 0
    n_o = int(o.sum())
    res["dim"] = n; res["n_occ"] = n_o
    log("dim %d, occupied %d, min|e| %.12f" % (n, n_o, float(np.min(np.abs(e)))))
    # ---------------------------------------------------------------- item 9: eigen residual, orthogonality, backward error
    Ut = torch.from_numpy(Uc)
    HU = torch.from_numpy(np.asarray(H @ Uc))
    Rm = HU - Ut * torch.from_numpy(e)[None, :]
    resF = float(torch.linalg.norm(Rm)); del Rm, HU
    nnzH = int(np.diff(H.indptr).max())
    absH = abs(H)
    # rounding of the residual itself: |fl(HU) - HU| <= gamma |H||U|, |fl(U Lam) - U Lam| <= u |U||Lam|; Frobenius of |H||U| <= || |H| ||_2 ||U||_F
    absH_norm = float(np.asarray(absH.sum(axis=1)).max())
    resF_up = (resF + gam(2 * nnzH + 2) * absH_norm * np.sqrt(n) + U_ROUND * float(np.max(np.abs(e))) * np.sqrt(n)) * (1 + 1e-10)
    S = Ut.conj().T @ Ut
    S.diagonal().sub_(1.0)
    defF = float(torch.linalg.norm(S)); del S
    eta = (defF + 2 * gam(2 * n + 2) * n) * (1 + 1e-10)        # | |U|^+ |U| |_F <= ||U||_F^2 = n, complex factor 2
    lam_max = float(np.max(np.abs(e)))
    dbwd = resF_up / np.sqrt(1 - eta) + 2 * np.sqrt(1 + eta) * lam_max * (1 / np.sqrt(1 - eta) - 1)
    gap_lo = float(np.min(np.abs(e))) - dbwd
    De = float(e[~o].min() - e[o].max())
    res["eig_cert"] = {"residual_F_float": resF, "residual_F_upper": resF_up, "orth_defect_F_float": defF, "eta_upper": eta,
                       "delta_bwd_upper": dbwd, "min_abs_eig_float": float(np.min(np.abs(e))), "gap_halfwidth_lower": gap_lo,
                       "full_gap_float": De, "full_gap_lower": De - 2 * dbwd, "n_occ_certified": bool(gap_lo > 0),
                       "e_occ_max": float(e[o].max()), "e_emp_min": float(e[~o].min()), "e_min": float(e.min()), "e_max": float(e.max())}
    log("eig cert: resF %.3e (up %.3e), defect %.3e (eta up %.3e), delta_bwd %.3e, gap half-width >= %.10f" % (resF, resF_up, defF, eta, dbwd, gap_lo))
    # norms of h_0, Q_i, h[u_0] - h_0 (rigorous upper bounds)
    nrm = {"h0": norm_upper_sparse(P.h), "Q": [norm_upper_sparse(q) for q in P.Q], "h_u0_minus_h0": norm_upper_sparse((H - P.h).tocsr()),
           "h_u0": norm_upper_sparse(H)}
    nrm["weyl_gap_lower_from_m"] = 0.2 - nrm["h_u0_minus_h0"]
    res["norm_upper"] = nrm
    log("norm upper bounds: %s" % json.dumps(nrm))
    # ---------------------------------------------------------------- precompute O U_o, O U_e
    Uo = torch.from_numpy(np.ascontiguousarray(Uc[:, o])); Ue = torch.from_numpy(np.ascontiguousarray(Uc[:, ~o]))
    eo = torch.from_numpy(e[o]); ee = torch.from_numpy(e[~o])
    Ops = [P.h] + list(P.Q)
    OUo = [torch.from_numpy(np.asarray(Op @ Uc[:, o])) for Op in Ops]
    OUe = [torch.from_numpy(np.asarray(Op @ Uc[:, ~o])) for Op in Ops]
    del Uc
    n_e = n - n_o
    log("precomputed O U blocks")

    def zblock(tl, cols):
        """Z = U_o^+ V U_e[:, cols] for the vertex term list tl."""
        Z = None
        for (oi, f, rows) in tl:
            ft = torch.from_numpy(f)
            if rows is not None:
                r = torch.from_numpy(rows)
                A1 = Uo[r].conj().T @ (ft[r][:, None] * OUe[oi][r][:, cols])
                A2 = OUo[oi][r].conj().T @ (ft[r][:, None] * Ue[r][:, cols])
            else:
                A1 = Uo.conj().T @ (ft[:, None] * OUe[oi][:, cols])
                A2 = OUo[oi].conj().T @ (ft[:, None] * Ue[:, cols])
            z = 0.5 * (A1 + A2)
            Z = z if Z is None else Z + z
        return Z

    def chi_sets(Xset, Yset, nb):
        """chi[X, Y] = 2 Re sum Z_X conj(Z_Y) w over the blocks of empty states; also sum |Z_X|^2 |w| norms."""
        nX, nY = len(Xset), len(Yset)
        chi = torch.zeros((nX, nY), dtype=torch.float64)
        zf2X = torch.zeros(nX, dtype=torch.float64); zf2Y = torch.zeros(nY, dtype=torch.float64)
        for c0 in range(0, n_e, nb):
            cols = torch.arange(c0, min(c0 + nb, n_e))
            wblk = 1.0 / (eo[:, None] - ee[cols][None, :])
            ZX = torch.stack([zblock(tl, cols) for tl in Xset])
            ZY = torch.stack([zblock(tl, cols) * wblk for tl in Yset])
            chi += 2.0 * torch.real(ZX.reshape(nX, -1) @ ZY.reshape(nY, -1).conj().T)
            zf2X += (ZX.abs() ** 2).sum(dim=(1, 2)); zf2Y += (ZY.abs() ** 2).sum(dim=(1, 2))
            del ZX, ZY
        return chi.numpy(), zf2X.numpy(), zf2Y.numpy()

    def vnorms(tl_list):
        F = []; Ainf = []
        for tl in tl_list:
            V = vertex_sparse(P, tl)
            F.append(float(np.sqrt((abs(V).power(2)).sum())))
            Ainf.append(float(np.asarray(abs(V).sum(axis=1)).max()))
        return np.array(F), np.array(Ainf)

    hbar_terms = None  # hbar as a vertex list: N_site e + v . p
    fN = np.repeat(N_site, 4)
    hbar_terms = [(0, fN, None)] + [(1 + i, np.repeat(v_vec[:, i], 4), None) for i in range(3)]
    cert = res["eig_cert"]
    d_half = cert["gap_halfwidth_lower"]; Delta_lo = cert["full_gap_lower"]
    eta_p = eta / (1 - eta)

    def chi_error_bound(FX, FY, AX, AY):
        """per-entry bound |chi_true - chi_float| (backward error + non-orthogonality + a-priori rounding)."""
        bw = (2.0 / (np.pi * (d_half - dbwd) ** 2)) * dbwd
        orth = (4.0 / Delta_lo) * eta_p * (2 + eta_p) * (1 + eta)
        # rounding of Z: |dZ|_F <= gamma_n || |U_o| ||_F || |V| ||_2 || |U_e| ||_F <= gamma_n sqrt(n_o n_e) ||V||_inf(abs); x2 for the two terms
        gz = 4 * gam(2 * n + 4)    # x2 for the two terms, x2 for |D||O| + |O||D| against |V| when f changes sign
        dZX = gz * np.sqrt(n_o * n_e) * AX; dZY = gz * np.sqrt(n_o * n_e) * AY
        rnd = (2.0 / Delta_lo) * (np.outer(dZX, FY) + np.outer(FX, dZY) + np.outer(dZX, dZY))
        rsum = gam(2 * n_o * n_e + 4) * (2.0 / Delta_lo) * np.outer(FX + dZX, FY + dZY)
        return np.outer(FX, FY) * (bw + orth) + rnd + rsum

    save = {}
    # ---------------------------------------------------------------- shell set
    if a.shell:
        terms, labels = vertex_terms_shell(P)
        Xset = terms + [hbar_terms]
        chi, zx, zy = chi_sets(Xset, terms, a.nb)
        FX, AX = vnorms(Xset); FY, AY = vnorms(terms)
        err = chi_error_bound(FX, FY, AX, AY)
        save["shell_chi"] = chi[:-1]; save["shell_hbar_row"] = chi[-1]; save["shell_err"] = err[:-1]; save["shell_hbar_err"] = err[-1]
        save["shell_F"] = FY; save["shell_Ainf"] = AY; save["shell_R"] = np.array([l[1] for l in labels]); save["shell_kind"] = np.array([l[0] for l in labels])
        # the u^T chi = 0 identity (N_R, v_R the shell site fields)
        nr = len(P.runiq)
        NR = np.array([N_site[P.inv == R][0] for R in range(nr)])
        vR = np.array([float((v_vec[P.inv == R] * P.xhat[P.inv == R]).sum(axis=1)[0]) for R in range(1, nr)])
        uvec = np.concatenate([NR, vR])
        ident = uvec @ chi[:-1]
        save["shell_u"] = uvec
        res["shell"] = {"n": len(terms), "chi_norm2_float": float(np.linalg.norm(chi[:-1], 2)), "chi_maxabs": float(np.abs(chi[:-1]).max()),
                        "err_bound_max": float(err[:-1].max()), "err_bound_rel_to_maxabs": float(err[:-1].max() / np.abs(chi[:-1]).max()),
                        "identity_uTchi_maxabs": float(np.abs(ident).max()), "identity_hbar_row_maxabs": float(np.abs(chi[-1]).max()),
                        "identity_scale": float(np.abs(uvec[:, None] * chi[:-1]).sum(axis=0).max()),
                        "chi_symmetry_defect": float(np.abs(chi[:-1] - chi[:-1].T).max())}
        log("shell chi done: %s" % json.dumps(res["shell"]))
        # P' operator norms per shell vertex (power iteration: a lower estimate of ||P'[V]||_2 = ||X_oe||_2) vs Heinz ||V||/Delta
        pn = []
        allc = torch.arange(n_e)
        for tl in terms:
            Xoe = zblock(tl, allc) / (eo[:, None] - ee[None, :])
            x = torch.ones(n_e, dtype=torch.complex128) / np.sqrt(n_e)
            s_ = 0.0
            for _ in range(60):
                y = Xoe.conj().T @ (Xoe @ x)
                s_ = float(torch.linalg.norm(y)) ** 0.5
                x = y / torch.linalg.norm(y)
            pn.append(s_)
            del Xoe
        Vn = np.array([norm_upper_sparse(vertex_sparse(P, tl), k=1) for tl in terms])
        save["shell_Pprime_norm_est"] = np.array(pn); save["shell_V_norm_upper"] = Vn
        res["shell"]["Pprime_over_heinz_max"] = float(np.max(np.array(pn) / (Vn / Delta_lo)))
        log("shell P' norms done (max ratio to Heinz %.3f)" % res["shell"]["Pprime_over_heinz_max"])
    # ---------------------------------------------------------------- T_d / U_d sets
    for d in a.d:
        Y, T, labT, Uset, labU = vertex_terms_TU(P, d, 2.0, a.w)
        nb = a.nb if d > 2 else max(32, a.nb // 2)
        Xset = T + [hbar_terms]
        log("d %d: %d tests, %d field directions" % (d, len(T), len(Uset)))
        chi, zx, zy = chi_sets(Xset, Uset, nb)
        FX, AX = vnorms(Xset); FY, AY = vnorms(Uset)
        err = chi_error_bound(FX, FY, AX, AY)
        save["d%d_chi" % d] = chi[:-1]; save["d%d_hbar_row" % d] = chi[-1]; save["d%d_err" % d] = err[:-1]; save["d%d_hbar_err" % d] = err[-1]
        save["d%d_Y" % d] = Y; save["d%d_FT" % d] = FX[:-1]; save["d%d_FU" % d] = FY; save["d%d_AU" % d] = AY
        Vn = np.array([norm_upper_sparse(vertex_sparse(P, tl), k=1) for tl in Uset])
        save["d%d_U_norm_upper" % d] = Vn
        res["d%d" % d] = {"nT": len(T), "nU": len(Uset), "chi_norm2_float": float(np.linalg.norm(chi[:-1], 2)), "chi_maxabs": float(np.abs(chi[:-1]).max()),
                          "err_bound_max": float(err[:-1].max()), "identity_hbar_row_maxabs": float(np.abs(chi[-1]).max()),
                          "identity_hbar_err_bound_max": float(err[-1].max())}
        log("d %d chi done: %s" % (d, json.dumps(res["d%d" % d])))
        np.savez(out + "_partial.npz", **save)
    res["elapsed_s"] = time.time() - t0
    np.savez(out + ".npz", e=e, **save)
    with open(out + ".json", "w") as f:
        json.dump(res, f, indent=1)
    if os.path.exists(out + "_partial.npz"):
        os.remove(out + "_partial.npz")
    log("done")


if __name__ == "__main__":
    main()
