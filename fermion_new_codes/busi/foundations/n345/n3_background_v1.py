#!/usr/bin/env python
"""n3_background_v1.py (busi, foundations item N3; ReviewA_foundations_v1.md sec. 3.2 N3, sec. 1 row 24; paper App. app:rows).

DATA script (box CPU, torch for the eigendecompositions, no card). Exhibits the candidate admitted strictly mixed background
C_0 = (1 - 2 eps) P + eps I on a finite spherical parent with no hopping across its boundary, and checks the admission conditions
App. app:rows states: strict mixedness, the sources (positive energy density, momentum density), the rows solved for N and v
(regular centre, N = 1 at infinity, inward branch), a differentiable (gapped) reference branch from the preparation to the
background's fields, a nonsingular Gram matrix of the retained tests spanning the averaged sources, and the margin eps against a
field change (the size of the covariance change C(u) - C_0 that realizes a clock or flow bump, first order).

Two parents, both from the engine's own operators (engine_ops_v1): --parent chain (the spherical sector the paper computes: the radial
chains kappa = 1..K with weights 16 kappa, chain_parts/clocked/site_density) and --parent ball (the declared three-dimensional lattice:
lattice3d_parts restricted to the sites |x - c| <= R_b, so no hopping crosses the ball's surface). The rows are solved by the
engine's field module (engine_fields_v16.solve_constraints_ivp, ivp route) on the spherically averaged sources; the fields enter h
back-averaged with the same kernel (engine_fields_v16.site_fields_kernel_average, K_0 for N, K_1 for v).

Reference. P(u) is the negative spectral projector of h[u] (gapped, so it is the continued sea along the path from the preparation
u = (1, 0), which the path check certifies). Variant B (the admitted background, iterated to its fixed point): C_0 = (1 - 2 eps) P(u_0)
+ eps I, so Delta = C_0 - P(u_0) = eps sgn(h[u_0]). Variant A (reviewer A's formula with P_0 = the sea of h_0, read at the same fields as a
diagnostic, not iterated): Delta_A = eps sgn(h_0) + P_0 - P(u_0); the extra term is the reference's own response to the background's
strain. At the start of the iteration (u = (1, 0)) the two coincide.

Outputs (small): <out>/n3_<parent>_<tag>.json (every number) and .npz (radial profiles). Never edits its inputs."""
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
VERSION = "n3_background_v1"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def eigh_t(H):
    """torch eigh of a dense Hermitian matrix (complex128), numpy back."""
    Ht = torch.from_numpy(np.ascontiguousarray(H, dtype=np.complex128))
    e, U = torch.linalg.eigh(Ht)
    return e.numpy(), U.numpy()


def eigvalsh_t(H):
    return torch.linalg.eigvalsh(torch.from_numpy(np.ascontiguousarray(H, dtype=np.complex128))).numpy()


def opnorm_sparse(A, iters=200):
    """largest |eigenvalue| of a Hermitian sparse matrix by power iteration on A^2 (A Hermitian), torch-free (sparse matvecs only)."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(A.shape[0]) + 1j * rng.standard_normal(A.shape[0])
    x /= np.linalg.norm(x)
    lam = 0.0
    for _ in range(iters):
        y = A @ (A @ x)
        ny = np.linalg.norm(y)
        if ny == 0:
            return 0.0
        lam_new = np.sqrt(ny)
        x = y / ny
        if abs(lam_new - lam) < 1e-12 * max(lam_new, 1e-300):
            lam = lam_new
            break
        lam = lam_new
    return float(lam)


def sigma_max(B, iters=300):
    """largest singular value of a dense matrix by power iteration (torch matmuls)."""
    Bt = torch.from_numpy(np.ascontiguousarray(B))
    g = torch.Generator().manual_seed(2)
    x = torch.randn(Bt.shape[1], dtype=torch.float64, generator=g).to(Bt.dtype)
    x /= torch.linalg.norm(x)
    s = 0.0
    for _ in range(iters):
        y = Bt.conj().T @ (Bt @ x)
        ny = float(torch.linalg.norm(y))
        if ny == 0:
            return 0.0
        s_new = ny ** 0.5
        x = y / ny
        if abs(s_new - s) < 1e-10 * max(s_new, 1e-300):
            s = s_new
            break
        s = s_new
    return float(s)


# ============================================================================ the two parents
class ChainParent:
    """the spherical sector: channels kappa = 1..K, each a radial chain of n_p sites (R_n = n + 1), open at its last site."""

    def __init__(self, OPS, n_p, m, K):
        self.OPS, self.n_p, self.m, self.K = OPS, n_p, m, K
        self.kappas = list(range(1, K + 1))
        self.w = {k: 16.0 * k for k in self.kappas}
        self.parts = {}
        for k in self.kappas:
            h, Q, R = OPS.chain_parts(n_p, m, float(k))
            self.parts[k] = (h, Q)
        self.R = R.copy()                      # site radii (shell radii)
        self.ncomp = 2
        self.dim = sum(2 * n_p for _ in self.kappas)

    def hm(self, k, N, v):
        h, Q = self.parts[k]
        return self.OPS.clocked(h, Q, N, v)

    def spectra(self, N, v):
        out = {}
        for k in self.kappas:
            e, U = eigh_t(self.hm(k, N, v))
            out[k] = (e, U)
        return out

    def sources(self, Ulist, coef):
        """shell totals sum_kappa w_kappa coef * site_density(A, U) for A = h_chain (energy) and Q_chain (momentum)."""
        eps = np.zeros(self.n_p); J = np.zeros(self.n_p)
        for k in self.kappas:
            h, Q = self.parts[k]
            U = Ulist[k]
            eps += self.w[k] * coef * self.OPS.site_density(h, U, None, 2)
            J += self.w[k] * coef * self.OPS.site_density(Q, U, None, 2)
        return eps, J


class BallParent:
    """the three-dimensional lattice restricted to the sites within R_b of the centre site (no hopping across the surface)."""

    def __init__(self, OPS, Rb, m):
        self.OPS, self.Rb, self.m = OPS, Rb, m
        L = 2 * Rb + 1
        h, Qs = OPS.lattice3d_parts(L, m, periodic=False)
        g = np.arange(L) - Rb
        X = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3).astype(float)   # site x = (i*L + j)*L + k
        r = np.sqrt((X ** 2).sum(axis=1))
        keep = np.where(r <= Rb + 1e-9)[0]
        comp = (4 * keep[:, None] + np.arange(4)[None, :]).ravel()
        self.h = h.tocsr()[comp][:, comp].tocsr()
        self.Q = [q.tocsr()[comp][:, comp].tocsr() for q in Qs]
        self.X = X[keep]; self.r = r[keep]
        with np.errstate(invalid="ignore", divide="ignore"):
            self.xhat = np.where(self.r[:, None] > 0, self.X / np.where(self.r[:, None] > 0, self.r[:, None], 1.0), 0.0)
        self.nsite = len(keep); self.dim = 4 * self.nsite; self.ncomp = 4
        self.runiq, self.inv = np.unique(np.round(self.r, 10), return_inverse=True)

    def hm_sparse(self, N_site, v_site_vec):
        Nd = sp.diags(np.repeat(N_site, 4))
        H = 0.5 * (Nd @ self.h + self.h @ Nd)
        for i in range(3):
            Vd = sp.diags(np.repeat(v_site_vec[:, i], 4))
            H = H + 0.5 * (Vd @ self.Q[i] + self.Q[i] @ Vd)
        return H.tocsr()

    def spectra(self, N_site, v_site_vec):
        e, U = eigh_t(self.hm_sparse(N_site, v_site_vec).toarray())
        return e, U

    def site_sources(self, U, coef):
        """per-site energy and momentum (vector) densities coef * Tr(U U^dag p_x)."""
        rho = coef * self.OPS.site_density(self.h, U, None, 4)
        J = np.stack([coef * self.OPS.site_density(self.Q[i], U, None, 4) for i in range(3)], axis=1)
        return rho, J

    def radial(self, rho, J):
        """totals per distinct radius: energy and the radial momentum x_hat . J."""
        Jr = (J * self.xhat).sum(axis=1)
        eps = np.bincount(self.inv, weights=rho, minlength=len(self.runiq))
        Jt = np.bincount(self.inv, weights=Jr, minlength=len(self.runiq))
        return eps, Jt


# ============================================================================ fields from the rows
def solve_rows(FLD, R_src, eps, J, g, ell, pad_to):
    """the rows on the spherically averaged sources; the sources at the radii R_src, zero (vacuum, no parent) out to pad_to."""
    R_src = np.asarray(R_src, float)
    pad = np.arange(np.floor(R_src[-1]) + 1.0, pad_to + 1.0)
    Rf = np.concatenate([R_src, pad])
    ef = np.concatenate([eps, np.zeros(len(pad))]); Jf = np.concatenate([J, np.zeros(len(pad))])
    Ns, vs, Wn, info = FLD.solve_constraints_ivp(ef, Jf, g, Rf, ell, collar_policy="fail", support_rel=1e-14)
    if info.get("failed"):
        raise RuntimeError("rows failed: %s" % info["failed"])
    Na, va, norms = FLD.site_fields_kernel_average(info, Rf, ell)
    return Rf, info, Na[:len(R_src)], va[:len(R_src)]


def field_summary(info, g, r_parent, ell):
    rr = np.linspace(0.0, float(info["r_collar"]), 4001)
    N = info["N_dense"](rr); v = info["v_dense"](rr); rho = info["rho_ell"](rr); Jl = info["J_ell"](rr)
    H0 = float(np.sqrt(8.0 * np.pi * g * float(info["rho_ell"](0.0)) / 3.0))
    inner = (rr > 0) & (rr <= max(r_parent - 3.0 * ell, ell))
    fr = -H0 * rr
    return {"E_collar": float(info["E_collar"]), "r_collar": float(info["r_collar"]), "N_centre": float(N[0]),
            "max_abs_N_minus_1": float(np.max(np.abs(N - 1.0))), "min_N_minus_abs_v": float(np.min(N - np.abs(v))),
            "max_abs_v": float(np.max(np.abs(v))), "max_2gM_over_r": float(np.max(info["W_dense"](rr[1:]) / rr[1:])),
            "rho_ell_centre": float(rho[0]), "rho_ell_min_inside_parent": float(rho[rr <= r_parent].min()),
            "J_ell_max_abs": float(np.max(np.abs(Jl))), "J_ell_over_rho_ell_max": float(np.max(np.abs(Jl[rr <= r_parent])) / max(float(np.max(np.abs(rho))), 1e-300)),
            "H0_friedmann": H0, "v_minus_friedmann_rel_inner": (float(np.max(np.abs(v[inner] - fr[inner])) / max(float(np.max(np.abs(fr[inner]))), 1e-300)) if inner.any() else None),
            "regime": info.get("regime", {}).get("label"), "gauge": info.get("gauge", {}).get("label"),
            "residual_W": info.get("residual_W"), "residual_invN": info.get("residual_invN"), "collar_margin": info.get("collar_margin"),
            "centre_flag": info.get("centre_flag")}


def law_bump_sources(info, g, r_mu, kind, r1, w):
    """first-order change of the sources the spherical rows assign, on the test radii r_mu, for a unit bump
    b(r) = exp(-(r - r1)^2/(2 w^2)) in the clock (kind 'N', flow fixed) or in the flow (kind 'v', clock fixed).
    rho = (r v^2)'/(8 pi g r^2 N^2), J = -v N'/(4 pi g r N^2) (Eqs. dr-radial)."""
    r = np.asarray(r_mu, float)
    b = lambda x: np.exp(-(np.asarray(x, float) - r1) ** 2 / (2 * w * w))
    db = lambda x: -(np.asarray(x, float) - r1) / (w * w) * b(x)
    hh = 1e-4
    Nf, vf = info["N_dense"], info["v_dense"]
    N = Nf(r); v = vf(r)
    dN = (Nf(r + hh) - Nf(np.abs(r - hh))) / (2 * hh)
    rho = info["rho_ell"](r)
    rs = np.where(r > 0, r, 1.0)
    if kind == "N":
        drho = -2.0 * rho * b(r) / N
        dJ = np.where(r > 0, -(v / (4 * np.pi * g * rs)) * (db(r) / N ** 2 - 2.0 * dN * b(r) / N ** 3), 0.0)
    else:
        F = lambda x: 2.0 * x * vf(x) * b(x)                 # delta (r v^2) = 2 r v delta v
        dF = (F(r + hh) - F(r - hh)) / (2 * hh)           # used only at r > 0 (r_mu >= ell/2 > hh); r = 0 is set to 0 below
        drho = np.where(r > 0, dF / (8 * np.pi * g * rs ** 2 * N ** 2), 0.0)
        dJ = np.where(r > 0, -b(r) * dN / (4 * np.pi * g * rs * N ** 2), 0.0)
    return drho, dJ, b, db


# ============================================================================ the chain parent
def run_chain(a, OPS, FLD, out):
    P = ChainParent(OPS, a.n_parent, a.m, a.K)
    t0 = time.time()
    log = []
    N_app = np.ones(P.n_p); v_app = np.zeros(P.n_p)
    hist = []
    spec0 = P.spectra(N_app, v_app)
    U0neg = {k: spec0[k][1][:, spec0[k][0] < 0] for k in P.kappas}
    gap0 = min(float(np.min(np.abs(spec0[k][0]))) for k in P.kappas)
    spec = spec0
    for it in range(a.maxit):
        Uneg = {k: spec[k][1][:, spec[k][0] < 0] for k in P.kappas}
        eps, J = P.sources(Uneg, -2.0 * a.eps)          # Delta = eps (I - 2P): Tr(Delta e_n) = -2 eps Tr(P e_n), e_n traceless
        Rf, info, N_new, v_new = solve_rows(FLD, P.R, eps, J, a.g, a.ell, a.pad_to)
        dN = float(np.max(np.abs(N_new - N_app))); dv = float(np.max(np.abs(v_new - v_app)))
        hist.append({"it": it, "dN_applied": dN, "dv_applied": dv, "E": float(info["E_collar"]), "t": time.time() - t0})
        print("chain it %d: dN %.3e dv %.3e E %.10e (%.0f s)" % (it, dN, dv, info["E_collar"], time.time() - t0), flush=True)
        N_app, v_app = N_new, v_new
        spec = P.spectra(N_app, v_app)
        if dN < a.tol and dv < a.tol:
            break
    # final sources at the fixed point (the reference at the returned fields)
    Uneg = {k: spec[k][1][:, spec[k][0] < 0] for k in P.kappas}
    eps, J = P.sources(Uneg, -2.0 * a.eps)
    Rf, info, N_chk, v_chk = solve_rows(FLD, P.R, eps, J, a.g, a.ell, a.pad_to)
    fixed_point_residual = {"dN_applied": float(np.max(np.abs(N_chk - N_app))), "dv_applied": float(np.max(np.abs(v_chk - v_app)))}
    # variant A diagnostic: the reference's own response P_0 - P(u_0)
    eA0, JA0 = P.sources(U0neg, 1.0)
    eA1, JA1 = P.sources(Uneg, 1.0)
    vp_eps, vp_J = eA0 - eA1, JA0 - JA1
    # positivity, strict mixedness
    C0_eigs = {"min": a.eps, "max": 1.0 - a.eps, "note": "C_0 = (1 - 2 eps) P + eps I with P a projector: eigenvalues eps and 1 - eps exactly"}
    pos = {"min_eps_n": float(eps.min()), "min_eps_n_over_bound": float(np.min(eps / (a.eps * np.array([sum(P.w[k] * 2 * a.m for k in P.kappas)] * P.n_p)))),
           "bound": "eps_n >= eps * sum_kappa 16 kappa * 2 m (|h_m| >= m N_min on every chain: h^2 = m^2 + (rest)^2; at u = (1,0) exact)"}
    # the path from the preparation (1, 0) to the background: gap of h along u_s = (1 + s (N - 1), s v)
    path = []
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        Ns = 1.0 + s * (N_app - 1.0); vs = s * v_app
        gmin = min(float(np.min(np.abs(eigvalsh_t(P.hm(k, Ns, vs))))) for k in P.kappas)
        path.append({"s": s, "min_abs_eig": gmin})
    # retained tests and the Gram matrix
    kern = FLD.kernels
    gram = {}
    margins = {}
    for tag, dr in (("spacing_ell", float(a.ell)), ("spacing_ell_over_2", a.ell / 2.0)):
        r_mu = np.arange(0.0, P.R[-1] + 1e-9, dr)
        K0, _ = kern(r_mu, P.R, a.ell); _, K1 = kern(r_mu[1:], P.R, a.ell)   # no momentum test at r = 0 (K_1(0, R) = 0: J_ell odd)
        M = len(r_mu); MJ = M - 1; MT = M + MJ
        # per channel vertices: energy tests 1/2{D, h}, momentum tests 1/2{D, Q}, D = diag(phi_mu(R_n)) on both components
        Gam = np.zeros((MT, MT))
        verts = {}
        for k in P.kappas:
            h, Q = P.parts[k]
            Vs = []
            for wts, A in ((K0, h), (K1, Q)):
                for mu in range(len(wts)):
                    d = np.repeat(wts[mu], 2)
                    Vs.append(0.5 * (d[:, None] * A + A * d[None, :]))
            Vs = np.array(Vs)                                  # (MT, n2, n2)
            Vf = Vs.reshape(MT, -1)
            Gk = np.real(Vf.conj() @ Vf.T)                      # Tr(A_mu A_nu) = sum conj(A_mu)_ab (A_nu)_ab (Hermitian)
            Gam += P.w[k] * Gk
            verts[k] = Vs
        ev = np.linalg.eigvalsh(Gam)
        EE = Gam[:M, :M]; JJ = Gam[M:, M:]; EJ = Gam[:M, M:]
        gram[tag] = {"n_tests": int(MT), "n_energy_tests": int(M), "n_momentum_tests": int(MJ), "r_mu_max": float(r_mu[-1]), "eig_min": float(ev.min()), "eig_max": float(ev.max()),
                     "cond": float(ev.max() / ev.min()) if ev.min() > 0 else None,
                     "eig_min_EE": float(np.linalg.eigvalsh(EE).min()), "eig_min_JJ": float(np.linalg.eigvalsh(JJ).min()),
                     "max_abs_EJ_over_max_diag": float(np.max(np.abs(EJ)) / np.max(np.abs(np.diag(Gam))))}
        # margin: unit clock bump and unit flow bump at r1, width w
        r1 = a.bump_r1 if a.bump_r1 else 0.5 * P.R[-1]; wb = a.bump_w if a.bump_w else 2.0 * a.ell
        mres = {}
        for kind in ("N", "v"):
            drho, dJ, b, db = law_bump_sources(info, a.g, r_mu, kind, r1, wb)
            ds = np.concatenate([drho, dJ[1:]])
            c = np.linalg.solve(Gam, ds)
            # check: the moments of X reproduce ds
            Xn = 0.0; mom_err = 0.0
            # the applied field change (back-averaged)
            bd = {"N_dense": (lambda x: b(np.abs(x))) if kind == "N" else (lambda x: np.zeros_like(np.atleast_1d(x), float)),
                  "v_dense": (lambda x: b(np.abs(x))) if kind == "v" else (lambda x: np.zeros_like(np.atleast_1d(x), float))}
            dNa, dva, _ = FLD.site_fields_kernel_average(bd, P.R, a.ell)
            dPn = 0.0
            moms = np.zeros(MT)
            for k in P.kappas:
                h, Q = P.parts[k]
                X = np.tensordot(c, verts[k], axes=(0, 0))
                Xn = max(Xn, float(np.max(np.abs(eigvalsh_t(X)))))
                moms += P.w[k] * np.real(np.einsum("mab,ba->m", verts[k], X))
                dh = OPS.clocked(h, Q, dNa, np.zeros(P.n_p)) if kind == "N" else OPS.clocked(h, Q, np.zeros(P.n_p), dva)
                e_k, U_k = spec[k]
                o = e_k < 0
                B = (U_k[:, o].conj().T @ dh @ U_k[:, ~o]) / (e_k[o][:, None] - e_k[~o][None, :])
                dPn = max(dPn, sigma_max(B))
            mom_err = float(np.max(np.abs(moms - ds)) / max(float(np.max(np.abs(ds))), 1e-300))
            eta_star = a.eps / (Xn + dPn) if (Xn + dPn) > 0 else None
            mres[kind] = {"norm_R_ds_per_unit_bump": Xn, "norm_dP_per_unit_bump_first_order": dPn, "moment_check_rel": mom_err,
                          "eta_star_first_order": eta_star, "r1": float(r1), "w": float(wb)}
        margins[tag] = mres
    prof_r = np.linspace(0.0, float(info["r_collar"]), 801)
    res = {"version": VERSION, "parent": "chain", "args": vars(a), "iterations": hist, "fixed_point_residual": fixed_point_residual,
           "gap_at_preparation": gap0, "path_gap": path, "C0_eigenvalues": C0_eigs, "positivity": pos,
           "sources": {"eps_n_first": eps[:4].tolist(), "eps_n_last": eps[-4:].tolist(), "eps_total": float(eps.sum()),
                       "J_n_max_abs": float(np.max(np.abs(J))), "J_over_eps_max": float(np.max(np.abs(J)) / np.max(np.abs(eps))),
                       "eps_n_R2_profile_ratio_last_over_mid": float((eps[-8] / P.R[-8] ** 2) / (eps[P.n_p // 2] / P.R[P.n_p // 2] ** 2))},
           "variantA_reference_response": {"eps_vp_max_abs": float(np.max(np.abs(vp_eps))), "eps_vp_total": float(vp_eps.sum()),
                                            "J_vp_max_abs": float(np.max(np.abs(vp_J))), "ratio_eps_vp_to_eps_max": float(np.max(np.abs(vp_eps)) / np.max(np.abs(eps)))},
           "fields": field_summary(info, a.g, float(P.R[-1]), a.ell), "gram": gram, "margin": margins, "elapsed_s": time.time() - t0}
    np.savez(out + ".npz", r=prof_r, N=info["N_dense"](prof_r), v=info["v_dense"](prof_r), rho_ell=info["rho_ell"](prof_r), J_ell=info["J_ell"](prof_r),
             R=P.R, eps_n=eps, J_n=J, eps_vp=vp_eps, J_vp=vp_J, N_applied=N_app, v_applied=v_app)
    return res


# ============================================================================ the ball parent
def run_ball(a, OPS, FLD, out):
    P = BallParent(OPS, a.Rb, a.m)
    t0 = time.time()
    print("ball R_b %d: %d sites, dim %d" % (a.Rb, P.nsite, P.dim), flush=True)
    Rsrc = P.runiq
    N_site = np.ones(P.nsite); v_vec = np.zeros((P.nsite, 3))
    hist = []
    e, U = P.spectra(N_site, v_vec)
    gap0 = float(np.min(np.abs(e)))
    U0neg = U[:, e < 0].copy()
    rho0, J0 = P.site_sources(U0neg, -2.0 * a.eps)
    # uniformity of the literal candidate at the preparation (u = (1, 0)): per-site energy density by radius
    bulk = P.r <= P.Rb - a.layer
    unif0 = {"rho_bulk_mean": float(rho0[bulk].mean()), "rho_bulk_rel_spread": float((rho0[bulk].max() - rho0[bulk].min()) / rho0[bulk].mean()),
             "rho_min": float(rho0.min()), "rho_max": float(rho0.max()), "rho_min_over_4m_eps": float(rho0.min() / (4 * a.m * a.eps)),
             "J_max_abs": float(np.abs(J0).max()), "bulk_radius": float(P.Rb - a.layer)}
    eneg = e[e < 0]
    for it in range(a.maxit):
        Uneg = U[:, e < 0]
        rho, J = P.site_sources(Uneg, -2.0 * a.eps)
        eps_r, J_r = P.radial(rho, J)
        Rf, info, Na, va = solve_rows(FLD, Rsrc, eps_r, J_r, a.g, a.ell, a.pad_to)
        N_new = Na[P.inv]; v_new = va[P.inv][:, None] * P.xhat
        dN = float(np.max(np.abs(N_new - N_site))); dv = float(np.max(np.abs(v_new - v_vec)))
        hist.append({"it": it, "dN_applied": dN, "dv_applied": dv, "E": float(info["E_collar"]), "t": time.time() - t0})
        print("ball it %d: dN %.3e dv %.3e E %.10e (%.0f s)" % (it, dN, dv, info["E_collar"], time.time() - t0), flush=True)
        N_site, v_vec = N_new, v_new
        del U
        e, U = P.spectra(N_site, v_vec)
        if dN < a.tol and dv < a.tol:
            break
    Uneg = U[:, e < 0]
    rho, J = P.site_sources(Uneg, -2.0 * a.eps)
    eps_r, J_r = P.radial(rho, J)
    Rf, info, Na, va = solve_rows(FLD, Rsrc, eps_r, J_r, a.g, a.ell, a.pad_to)
    fpr = {"dN_applied": float(np.max(np.abs(Na[P.inv] - N_site))), "dv_applied": float(np.max(np.abs(va[P.inv][:, None] * P.xhat - v_vec)))}
    gap1 = float(np.min(np.abs(e)))
    # path: Weyl bound with ||h[u_0] - h_0|| against the preparation's gap (|h_0| >= m exactly: h_0^2 = m^2 + (alpha.D)^2)
    dH = P.hm_sparse(N_site, v_vec) - P.h
    ndH = opnorm_sparse(dH)
    path = {"gap_h0": gap0, "gap_h_u0": gap1, "norm_h_u0_minus_h0": ndH, "weyl_path_gapped": bool(ndH < gap0),
            "note": "every h_s = h_0 + s (h[u_0] - h_0) has |eig| >= gap_h0 - s ||h[u_0] - h_0|| > 0 when the flag is true"}
    # variant A diagnostic
    rA0, JA0 = P.site_sources(U0neg, 1.0); rA1, JA1 = P.site_sources(Uneg, 1.0)
    vp_r, vp_J = rA0 - rA1, JA0 - JA1
    unif = {"rho_bulk_mean": float(rho[bulk].mean()), "rho_bulk_rel_spread": float((rho[bulk].max() - rho[bulk].min()) / rho[bulk].mean()),
            "rho_min": float(rho.min()), "rho_max": float(rho.max()), "J_max_abs": float(np.abs(J).max()),
            "J_r_max_abs": float(np.abs((J * P.xhat).sum(axis=1)).max())}
    # Gram of the radial tests and the margin
    kern = FLD.kernels
    gram = {}; margins = {}
    for tag, dr in (("spacing_ell", float(a.ell)), ("spacing_ell_over_2", a.ell / 2.0)):
        r_mu = np.arange(0.0, P.Rb + 1e-9, dr)
        M = len(r_mu); MJ = M - 1; MT = M + MJ
        K0, _ = kern(r_mu, P.r, a.ell); _, K1 = kern(r_mu[1:], P.r, a.ell)   # (M, nsite), (M - 1, nsite): no momentum test at r = 0
        verts = []
        for mu in range(M):
            D = sp.diags(np.repeat(K0[mu], 4))
            verts.append((0.5 * (D @ P.h + P.h @ D)).tocsr())
        for mu in range(MJ):
            A = None
            for i in range(3):
                D = sp.diags(np.repeat(K1[mu] * P.xhat[:, i], 4))
                t = 0.5 * (D @ P.Q[i] + P.Q[i] @ D)
                A = t if A is None else A + t
            verts.append(A.tocsr())
        Gam = np.zeros((MT, MT))
        for i in range(MT):
            for j in range(i, MT):
                val = float(np.real(verts[i].multiply(verts[j].T).sum()))
                Gam[i, j] = Gam[j, i] = val
        ev = np.linalg.eigvalsh(Gam)
        gram[tag] = {"n_tests": int(MT), "n_energy_tests": int(M), "n_momentum_tests": int(MJ), "eig_min": float(ev.min()), "eig_max": float(ev.max()), "cond": float(ev.max() / ev.min()) if ev.min() > 0 else None,
                     "eig_min_EE": float(np.linalg.eigvalsh(Gam[:M, :M]).min()), "eig_min_JJ": float(np.linalg.eigvalsh(Gam[M:, M:]).min()),
                     "max_abs_EJ_over_max_diag": float(np.max(np.abs(Gam[:M, M:])) / np.max(np.abs(np.diag(Gam))))}
        r1 = a.bump_r1 if a.bump_r1 else 0.5 * P.Rb; wb = a.bump_w if a.bump_w else float(a.ell)
        mres = {}
        o = e < 0
        Uo = torch.from_numpy(np.ascontiguousarray(U[:, o])); Ue = torch.from_numpy(np.ascontiguousarray(U[:, ~o]))
        den = torch.from_numpy(e[o][:, None] - e[~o][None, :])
        for kind in ("N", "v"):
            drho, dJ, b, db = law_bump_sources(info, a.g, r_mu, kind, r1, wb)
            ds = np.concatenate([drho, dJ[1:]])
            c = np.linalg.solve(Gam, ds)
            X = None
            for mu in range(MT):
                X = c[mu] * verts[mu] if X is None else X + c[mu] * verts[mu]
            X = X.tocsr()
            moms = np.array([float(np.real(verts[i].multiply(X.T).sum())) for i in range(MT)])
            Xn = opnorm_sparse(X)
            bd = {"N_dense": (lambda x: b(np.abs(x))) if kind == "N" else (lambda x: np.zeros_like(np.atleast_1d(x), float)),
                  "v_dense": (lambda x: b(np.abs(x))) if kind == "v" else (lambda x: np.zeros_like(np.atleast_1d(x), float))}
            dNa, dva, _ = FLD.site_fields_kernel_average(bd, Rsrc, a.ell)
            if kind == "N":
                Nd = sp.diags(np.repeat(dNa[P.inv], 4)); dh = (0.5 * (Nd @ P.h + P.h @ Nd)).tocsr()
            else:
                dv3 = dva[P.inv][:, None] * P.xhat; dh = None
                for i in range(3):
                    Vd = sp.diags(np.repeat(dv3[:, i], 4)); t = 0.5 * (Vd @ P.Q[i] + P.Q[i] @ Vd)
                    dh = t if dh is None else dh + t
                dh = dh.tocsr()
            dhUe = torch.from_numpy(np.ascontiguousarray(dh @ U[:, ~o]))
            B = (Uo.conj().T @ dhUe) / den
            dPn = sigma_max(B.numpy())
            del dhUe, B
            mres[kind] = {"norm_R_ds_per_unit_bump": Xn, "norm_dP_per_unit_bump_first_order": dPn,
                          "moment_check_rel": float(np.max(np.abs(moms - ds)) / max(float(np.max(np.abs(ds))), 1e-300)),
                          "eta_star_first_order": a.eps / (Xn + dPn) if (Xn + dPn) > 0 else None, "r1": float(r1), "w": float(wb)}
        del Uo, Ue, den
        margins[tag] = mres
    prof_r = np.linspace(0.0, float(info["r_collar"]), 801)
    res = {"version": VERSION, "parent": "ball", "args": vars(a), "nsite": int(P.nsite), "dim": int(P.dim), "iterations": hist,
           "fixed_point_residual": fpr, "path": path, "C0_eigenvalues": {"min": a.eps, "max": 1.0 - a.eps, "note": "exact by construction"},
           "literal_candidate_at_preparation": unif0, "sources_at_fixed_point": unif,
           "variantA_reference_response": {"rho_vp_max_abs": float(np.abs(vp_r).max()), "rho_vp_total": float(vp_r.sum()), "J_vp_max_abs": float(np.abs(vp_J).max()),
                                            "ratio_rho_vp_to_rho_max": float(np.abs(vp_r).max() / np.abs(rho).max())},
           "fields": field_summary(info, a.g, float(P.Rb), a.ell), "gram": gram, "margin": margins, "elapsed_s": time.time() - t0}
    np.savez(out + ".npz", r=prof_r, N=info["N_dense"](prof_r), v=info["v_dense"](prof_r), rho_ell=info["rho_ell"](prof_r), J_ell=info["J_ell"](prof_r),
             r_site=P.r, rho_site=rho, rho_site_prep=rho0, Jr_site=(J * P.xhat).sum(axis=1), rho_vp=vp_r, R_uniq=Rsrc, eps_r=eps_r, J_r=J_r)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", choices=["chain", "ball"], required=True)
    ap.add_argument("--code", default="<internal>/code16d")
    ap.add_argument("--m", type=float, default=0.2); ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--eps", type=float, default=1e-4); ap.add_argument("--ell", type=int, default=4)
    ap.add_argument("--n-parent", type=int, default=128); ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--Rb", type=int, default=8); ap.add_argument("--layer", type=float, default=3.0)
    ap.add_argument("--pad-to", type=float, default=None)
    ap.add_argument("--maxit", type=int, default=8); ap.add_argument("--tol", type=float, default=1e-11)
    ap.add_argument("--bump-r1", type=float, default=None); ap.add_argument("--bump-w", type=float, default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", required=True); ap.add_argument("--outdir", default="<internal>/n345")
    a = ap.parse_args()
    sys.path.insert(0, a.code)
    import engine_ops_v1 as OPS
    import engine_fields_v16 as FLD
    rmax = a.n_parent if a.parent == "chain" else a.Rb
    if a.pad_to is None:
        a.pad_to = float(rmax + 12 * a.ell)
    out = os.path.join(a.outdir, "n3_%s_%s" % (a.parent, a.tag))
    if os.path.exists(out + ".json"):
        sys.exit("refusing to overwrite " + out + ".json")
    prov = {"script": os.path.abspath(__file__), "script_md5": md5(os.path.abspath(__file__)),
            "engine_ops_v1": md5(OPS.__file__), "engine_fields_v16": md5(FLD.__file__),
            "engine_fields_v12": md5(os.path.join(a.code, "engine_fields_v12.py")), "engine_fields_v4": md5(os.path.join(a.code, "engine_fields_v4.py")),
            "torch": torch.__version__, "numpy": np.__version__, "threads": NTH, "fields_route": os.environ.get("QI2_FIELDS_ROUTE", "ivp (default)")}
    print(json.dumps(prov), flush=True)
    res = run_chain(a, OPS, FLD, out) if a.parent == "chain" else run_ball(a, OPS, FLD, out)
    res["provenance"] = prov
    with open(out + ".json", "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps({k: res[k] for k in res if k not in ("iterations",)}, indent=1, default=str)[:20000], flush=True)


if __name__ == "__main__":
    main()
