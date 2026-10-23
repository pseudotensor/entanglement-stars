#!/usr/bin/env python
"""pm1_common_v1.py (busi, foundations reviewb5; agent A, shared with agent B). The projected model PM1 of item2_projected_solve.txt.

Contents (box only; imports the N3 code and the engine read-only):
  background(cache, rebuild)  N3's ball fixed point (n3_background_v1.BallParent + solve_rows, same arguments as the N3 run), cached
                              as an npz of the site fields, shell sources, eigenvalues and dense radial profiles (step 1/128 to r = 48).
                              Returns the parent, the site fields, the eigen-decomposition at u_0 (when computed) and radial splines.
  RadialBG                    parity-extended quintic splines of N (even), v (odd), rho_ell (even), J_ell (odd) and their derivatives;
                              jets3d(y) returns the 3D jets (N, dN, dv, ddv) of the spherical background at points y.
  test_points(d, Rb)          the T_d points y = d n, |y| <= R_b.
  bump(z, w)                  psi = (1 - |z|^2/w^2)_+^5 with gradient and Hessian (analytic).
  bump_backavg(Rq, w, ell)    Phi(rho) = (G_ell * psi)(rho), the back-averaged bump as a radial function, Gauss-Legendre exact to 1e-15.
  laws(jets, b)               rho_g, J_g of the laws on 3D jets (complex-safe, for the complex-step linearization).
  linearized_A(...)           the collocated linearized projected constraint map A_d (rows: tests (mu, c in rho, Jx, Jy, Jz); columns:
                              bumps (j, a in N, vx, vy, vz)).
Conventions: dv[..., i, j] = d_i v_j, ddv[..., k, i, j] = d_k d_i v_j; (div M)_i = sum_j d_j M_ji; Om_ij = (d_i v_j - d_j v_i)/(2N).
Lattice units a = 1."""
import os
import sys

import numpy as np

N3_DIR = "<internal>/n345"
CODE16D = "<internal>/code16d"
G_NEWTON = 0.25


def _imports():
    for p in (CODE16D, N3_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    import engine_ops_v1 as OPS
    import engine_fields_v16 as FLD
    import n3_background_v1 as N3
    return OPS, FLD, N3


# ============================================================================ background
def build_background(Rb=8, m=0.2, g=0.25, eps=1e-4, ell=2, maxit=8, tol=1e-11, log=print):
    """N3's ball fixed point, iterated exactly as n3_background_v1.run_ball (variant B). Returns dict (with e, U at u_0)."""
    OPS, FLD, N3 = _imports()
    P = N3.BallParent(OPS, Rb, m)
    pad_to = float(Rb + 12 * ell)
    Rsrc = P.runiq
    N_site = np.ones(P.nsite); v_vec = np.zeros((P.nsite, 3))
    e, U = P.spectra(N_site, v_vec)
    hist = []
    for it in range(maxit):
        Uneg = U[:, e < 0]
        rho, J = P.site_sources(Uneg, -2.0 * eps)
        eps_r, J_r = P.radial(rho, J)
        Rf, info, Na, va = N3.solve_rows(FLD, Rsrc, eps_r, J_r, g, ell, pad_to)
        N_new = Na[P.inv]; v_new = va[P.inv][:, None] * P.xhat
        dN = float(np.max(np.abs(N_new - N_site))); dv = float(np.max(np.abs(v_new - v_vec)))
        hist.append([it, dN, dv, float(info["E_collar"])])
        log("bg it %d dN %.3e dv %.3e E %.12e" % (it, dN, dv, info["E_collar"]))
        N_site, v_vec = N_new, v_new
        del U
        e, U = P.spectra(N_site, v_vec)
        if dN < tol and dv < tol:
            break
    Uneg = U[:, e < 0]
    rho, J = P.site_sources(Uneg, -2.0 * eps)
    eps_r, J_r = P.radial(rho, J)
    Rf, info, Na, va = N3.solve_rows(FLD, Rsrc, eps_r, J_r, g, ell, pad_to)
    fpr = [float(np.max(np.abs(Na[P.inv] - N_site))), float(np.max(np.abs(va[P.inv][:, None] * P.xhat - v_vec)))]
    rr = np.arange(0.0, 48.0 + 1e-9, 1.0 / 128.0)
    prof = {"r": rr, "N": np.asarray(info["N_dense"](rr), float), "v": np.asarray(info["v_dense"](rr), float),
            "W": np.asarray(info["W_dense"](rr), float), "rho": np.asarray(info["rho_ell"](np.minimum(rr, info["rfine"][-1])), float),
            "J": np.asarray(info["J_ell"](np.minimum(rr, info["rfine"][-1])), float)}
    # beyond the source spline's range the smoothed sources are the kernel tails (below 1e-14 relative): set to the kernel sum directly
    K0, K1 = FLD.kernels(rr, Rsrc, ell)
    prof["rho_direct"] = K0 @ eps_r; prof["J_direct"] = K1 @ J_r
    return {"P": P, "N_site": N_site, "v_vec": v_vec, "Na": Na, "va": va, "e": e, "U": U, "eps_r": eps_r, "J_r": J_r, "Rsrc": Rsrc,
            "info": info, "prof": prof, "hist": hist, "fixed_point_residual": fpr, "E": float(info["E_collar"]),
            "r_collar": float(info["r_collar"]), "args": dict(Rb=Rb, m=m, g=g, eps=eps, ell=ell, pad_to=pad_to)}


def save_background(bg, path):
    p = bg["prof"]
    np.savez(path, N_site=bg["N_site"], v_vec=bg["v_vec"], Na=bg["Na"], va=bg["va"], e=bg["e"], eps_r=bg["eps_r"], J_r=bg["J_r"],
             Rsrc=bg["Rsrc"], r=p["r"], N=p["N"], v=p["v"], W=p["W"], rho=p["rho"], J=p["J"], rho_direct=p["rho_direct"],
             J_direct=p["J_direct"], E=bg["E"], r_collar=bg["r_collar"], hist=np.array(bg["hist"]),
             fixed_point_residual=np.array(bg["fixed_point_residual"]), args=np.array([bg["args"][k] for k in ("Rb", "m", "g", "eps", "ell")]))


def load_background(path, with_parent=True):
    Z = np.load(path)
    bg = {k: Z[k] for k in Z.files}
    if with_parent:
        OPS, FLD, N3 = _imports()
        Rb, m = int(Z["args"][0]), float(Z["args"][1])
        bg["P"] = N3.BallParent(OPS, Rb, m)
    return bg


# ============================================================================ radial background splines and 3D jets
class RadialBG:
    def __init__(self, r, N, v, rho, J):
        from scipy.interpolate import make_interp_spline
        rr = np.concatenate([-r[:0:-1], r])
        ext = lambda f, par: np.concatenate([par * f[:0:-1], f])
        self.N = make_interp_spline(rr, ext(N, 1.0), k=5)
        self.v = make_interp_spline(rr, ext(v, -1.0), k=5)
        self.rho = make_interp_spline(rr, ext(rho, 1.0), k=5)
        self.J = make_interp_spline(rr, ext(J, -1.0), k=5)
        self.rmax = float(r[-1])

    def jets3d(self, y):
        """y (n, 3) -> N (n,), dN (n, 3), dv (n, 3, 3) [d_i v_j], ddv (n, 3, 3, 3) [d_k d_i v_j] of N(r), v(r) x_hat."""
        y = np.atleast_2d(np.asarray(y, float))
        r = np.sqrt((y ** 2).sum(axis=1))
        n = len(r)
        N = self.N(r); N1 = self.N(r, 1)
        v = self.v(r); v1 = self.v(r, 1); v2 = self.v(r, 2); v3 = self.v(r, 3)
        I = np.eye(3)
        dN = np.zeros((n, 3)); dv = np.zeros((n, 3, 3)); ddv = np.zeros((n, 3, 3, 3))
        for q in range(n):
            if r[q] < 1e-3:
                # series about the centre: v = v1(0) r xhat + v3(0)/6 r^2 r_vec ... ; N = N(0) + N2(0) r^2/2
                c1 = v1[q]; c3 = self.v(0.0, 3) / 6.0          # v(r) = c1 r + c3 r^3
                yq = y[q]
                dN[q] = self.N(0.0, 2) * yq                       # grad of N2 r^2/2
                # v_j = c1 y_j + c3 |y|^2 y_j
                dv[q] = c1 * I + c3 * (2.0 * np.outer(yq, yq) + (yq @ yq) * I)
                ddv[q] = c3 * 2.0 * (np.einsum("ki,j->kij", I, yq) + np.einsum("kj,i->kij", I, yq) + np.einsum("ij,k->kij", I, yq))
                continue
            xh = y[q] / r[q]
            dN[q] = N1[q] * xh
            a = v1[q] - v[q] / r[q]; bb = v[q] / r[q]
            da = v2[q] - v1[q] / r[q] + v[q] / r[q] ** 2; db = v1[q] / r[q] - v[q] / r[q] ** 2
            dv[q] = a * np.outer(xh, xh) + bb * I
            dx = (I - np.outer(xh, xh)) / r[q]                      # d_k xhat_i
            ddv[q] = (da * np.einsum("k,i,j->kij", xh, xh, xh) + a * (np.einsum("ki,j->kij", dx, xh) + np.einsum("kj,i->kij", dx, xh))
                      + db * np.einsum("k,ij->kij", xh, I))
        return N, dN, dv, ddv


# ============================================================================ the laws on jets
def laws(N, dN, dv, ddv, b=0.0, G=G_NEWTON):
    """rho_g, J_g (vector) on jets; complex-safe. Shapes: N (n,), dN (n,3), dv (n,3,3), ddv (n,3,3,3)."""
    S = 0.5 * (dv + np.swapaxes(dv, -1, -2)); A = 0.5 * (dv - np.swapaxes(dv, -1, -2))
    trS = np.einsum("nii->n", S)
    rho = (trS ** 2 - np.einsum("nij,nij->n", S, S) - b * np.einsum("nij,nij->n", A, A)) / (16.0 * np.pi * G * N ** 2)
    # d_j S_ji = (d_j d_j v_i + d_j d_i v_j)/2 ; d_i trS = d_i d_k v_k ; d_j A_ji = (d_j d_j v_i - d_j d_i v_j)/2
    lap = np.einsum("njji->ni", ddv)            # sum_j d_j d_j v_i
    gdiv = np.einsum("nijj->ni", ddv)           # d_i sum_j d_j v_j   (ddv[k,i,j] = d_k d_i v_j: set k = i, i = j, j = j)
    divS = 0.5 * (lap + gdiv)
    divA = 0.5 * (lap - gdiv)
    dtr = gdiv
    divK = divS / N[:, None] - np.einsum("nji,nj->ni", S, dN) / N[:, None] ** 2
    dtrK = dtr / N[:, None] - trS[:, None] * dN / N[:, None] ** 2
    divOm = divA / N[:, None] - np.einsum("nji,nj->ni", A, dN) / N[:, None] ** 2
    J = -(divK - dtrK + b * divOm) / (8.0 * np.pi * G)
    return rho, J


def test_points(d, Rb=8.0):
    m = int(np.floor(Rb / d))
    g = np.arange(-m, m + 1) * float(d)
    Y = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    return Y[(Y ** 2).sum(axis=1) <= Rb ** 2 + 1e-9]


def bump(z, w):
    """psi, grad psi (n,3), hess psi (n,3,3) of (1 - |z|^2/w^2)_+^5."""
    z = np.atleast_2d(z)
    s = (z ** 2).sum(axis=1) / w ** 2
    t = np.clip(1.0 - s, 0.0, None)
    psi = t ** 5
    gr = (-10.0 / w ** 2) * (t ** 4)[:, None] * z
    H = (80.0 / w ** 4) * (t ** 3)[:, None, None] * np.einsum("ni,nj->nij", z, z) - (10.0 / w ** 2) * (t ** 4)[:, None, None] * np.eye(3)[None]
    return psi, gr, H


def bump_backavg(Rq, w, ell, nq=96):
    """Phi(rho) = int G_ell(y - x) psi(y) d^3y at |x| = rho (the bump centred at 0), via the shell kernel K_0 (Gauss-Legendre on [0, w])."""
    OPS, FLD, N3 = _imports()
    xg, wg = np.polynomial.legendre.leggauss(nq)
    s = 0.5 * w * (xg + 1.0); ws = 0.5 * w * wg
    psi = np.clip(1.0 - s ** 2 / w ** 2, 0.0, None) ** 5
    K0, _ = FLD.kernels(np.asarray(Rq, float), s, ell)
    return K0 @ (4.0 * np.pi * s ** 2 * psi * ws)


def gauss_weights(y, X, ell):
    """G_ell(y - x) for sites X (n, 3): the point Gaussian of per-axis width ell (B's test weight)."""
    d2 = ((X - np.asarray(y, float)[None, :]) ** 2).sum(axis=1)
    return np.exp(-d2 / (2.0 * ell ** 2)) / ((2.0 * np.pi * ell ** 2) ** 1.5)


def bump_jets(Y, z, w, a):
    """the jets (N, dN, dv, ddv) of the field e_a psi(. - z) at the points Y (a = 0: N; a = 1, 2, 3: v_x, v_y, v_z)."""
    psi, gr, H = bump(Y - np.asarray(z, float)[None, :], w)
    n = len(Y)
    N = np.zeros(n); dN = np.zeros((n, 3)); dv = np.zeros((n, 3, 3)); ddv = np.zeros((n, 3, 3, 3))
    if a == 0:
        N = psi; dN = gr
    else:
        dv[:, :, a - 1] = gr; ddv[:, :, :, a - 1] = H
    return N, dN, dv, ddv


def linearized_A(bgsp, Y, Zc, w, b=0.0, hcs=1e-30, jets=None):
    """A[(mu, c), (j, a)] = D s_g[u_0](e_a psi(. - z_j))(y_mu), c in (rho, Jx, Jy, Jz), a in (N, vx, vy, vz). Complex step (exact
    to rounding: the laws are rational in the jets). Rows ordered mu-major (4 per test point), columns j-major (4 per bump)."""
    N0, dN0, dv0, ddv0 = bgsp.jets3d(Y) if jets is None else jets
    nT, nU = len(Y), len(Zc)
    A = np.zeros((4 * nT, 4 * nU))
    for j in range(nU):
        psi, gr, H = bump(Y - Zc[j][None, :], w)
        if not np.any(psi > 0):
            continue
        for a in range(4):
            N = N0.astype(complex); dN = dN0.astype(complex); dv = dv0.astype(complex); ddv = ddv0.astype(complex)
            if a == 0:
                N = N + 1j * hcs * psi; dN = dN + 1j * hcs * gr
            else:
                comp = a - 1
                dv = dv.copy(); dv[:, :, comp] += 1j * hcs * gr               # d_i (psi e_comp)_j = d_i psi delta_j,comp
                ddv = ddv.copy(); ddv[:, :, :, comp] += 1j * hcs * H        # d_k d_i psi delta_j,comp
            rho, J = laws(N, dN, dv, ddv, b)
            col = 4 * j + a
            A[0::4, col] = rho.imag / hcs
            A[1::4, col] = J[:, 0].imag / hcs; A[2::4, col] = J[:, 1].imag / hcs; A[3::4, col] = J[:, 2].imag / hcs
    return A
