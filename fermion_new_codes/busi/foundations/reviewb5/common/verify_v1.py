#!/usr/bin/env python
"""verify_v1.py (busi, foundations reviewb5; agent A). Rigorous a-posteriori enclosures for small and medium REAL float matrices, with
the classical floating-point bound |fl(AB) - AB| <= gamma_n |A| |B| (Higham, Thm 3.5; gamma_n = n u/(1 - n u), u = 2^-53), and an
mpmath interval cross-check for small matrices. The input matrix is treated as exact (the error of how it was produced is a separate
item and is stated separately in the deliveries).

  gamma(n)
  matmul_err(A, B)            -> (C, E): C = fl(A B), E >= |A B - C| elementwise (E itself inflated for its own rounding)
  sym_extreme_eigs(G)         -> (lo, hi): lo <= lambda_min(G), hi >= lambda_max(G) (G symmetric float), by Ostrowski + Gershgorin
  sigma_min_enclosure(M)      -> (lo, hi): lo <= sigma_min(M) <= hi, by an approximate inverse X: rho = ||I - X M|| < 1 gives
                                 ||M^-1|| <= ||X||/(1 - rho); hi = ||M v||/||v|| for the float right singular vector (rounding-bounded)
  norm2_upper(M)              -> an upper bound on ||M||_2 (sqrt of sym_extreme_eigs(M^T M)'s hi, with the product's error)
  iv_sigma_min_check(M)       -> mpmath interval version of the inverse-residual bound (small n only)"""
import numpy as np

U_ROUND = 2.0 ** -53
SAFE = 1.0 + 1e-10          # inflation of every computed upper bound for the rounding in forming the bound itself


def gamma(n):
    nu = n * U_ROUND
    if nu >= 0.5:
        raise ValueError("n too large for gamma_n")
    return nu / (1.0 - nu)


def matmul_err(A, B):
    A = np.asarray(A, float); B = np.asarray(B, float)
    C = A @ B
    n = A.shape[1]
    E = gamma(n + 2) * (np.abs(A) @ np.abs(B)) * SAFE
    return C, E


def fro_upper(E):
    return float(np.sqrt(np.sum(np.asarray(E, float) ** 2))) * SAFE


def sym_extreme_eigs(G):
    """bounds on the extreme eigenvalues of the exact symmetric matrix G (given in float)."""
    G = np.asarray(G, float)
    G = 0.5 * (G + G.T)  # exact for a symmetric input; for a nearly symmetric one the symmetric part is the object
    n = G.shape[0]
    w, V = np.linalg.eigh(G)
    GV, E1 = matmul_err(G, V)
    D, E2 = matmul_err(V.T, GV)
    # |V^T G V - D| <= E2 + |V^T| E1
    ED = E2 + gamma(n + 2) * 0 + (np.abs(V.T) @ E1) * SAFE
    S, ES = matmul_err(V.T, V)
    eta = fro_upper(np.abs(S - np.eye(n)) + ES)
    if eta >= 1.0:
        raise RuntimeError("eigenvector matrix too far from orthogonal")
    off = np.abs(D).copy(); np.fill_diagonal(off, 0.0)
    rad = (off.sum(axis=1) + ED.sum(axis=1)) * SAFE
    dlo = np.diag(D) - rad; dhi = np.diag(D) + rad
    Lo, Hi = float(dlo.min()), float(dhi.max())
    # Ostrowski: lambda_k(V^T G V) = theta_k lambda_k(G), theta_k in [1 - eta, 1 + eta]
    lo = Lo / (1.0 + eta) if Lo > 0 else Lo / (1.0 - eta)
    hi = Hi / (1.0 - eta) if Hi > 0 else Hi / (1.0 + eta)
    return lo * (1 - 1e-12 * np.sign(lo)), hi * (1 + 1e-12 * np.sign(hi)), {"eta": eta, "max_radius": float(rad.max()), "float_eigs_min": float(w.min()), "float_eigs_max": float(w.max())}


def norm2_upper(M):
    M = np.asarray(M, float)
    G, E = matmul_err(M.T, M)
    _, hi, _ = sym_extreme_eigs(G)
    return float(np.sqrt(max(hi, 0.0) + fro_upper(E))) * SAFE


def sigma_min_enclosure(M):
    M = np.asarray(M, float)
    n = M.shape[0]
    assert M.shape[1] == n
    X = np.linalg.inv(M)
    XM, E = matmul_err(X, M)
    rho = fro_upper(np.abs(np.eye(n) - XM) + E)
    out = {"rho_residual": rho}
    if rho >= 1.0:
        out["lo"] = 0.0
    else:
        nX = norm2_upper(X)
        out["lo"] = (1.0 - rho) / nX / SAFE
        out["normX_upper"] = nX
    # upper bound: ||M v|| / ||v|| for the float right singular vector of sigma_min (any v gives an upper bound on sigma_min)
    _, s, Vt = np.linalg.svd(M)
    v = Vt[-1]
    Mv, Ev = matmul_err(M, v[:, None])
    nMv = float(np.linalg.norm(np.abs(Mv[:, 0]) + Ev[:, 0])) * SAFE
    nv = float(np.linalg.norm(v)) / SAFE
    out["hi"] = nMv / nv
    out["float_sigma_min"] = float(s[-1]); out["float_sigma_max"] = float(s[0])
    return out


def iv_sigma_min_check(M, X=None, prec=80):
    """mpmath interval arithmetic: rho = ||I - X M||_F (interval upper), ||X||_F (interval upper), lo = (1 - rho)/||X||_F.
    Uses the Frobenius norm of X (looser than the 2-norm, but every operation is an interval operation). Small n only."""
    import mpmath
    from mpmath import iv
    iv.prec = prec
    M = np.asarray(M, float); n = M.shape[0]
    if X is None:
        X = np.linalg.inv(M)
    Mi = iv.matrix([[iv.mpf(float(M[i, j])) for j in range(n)] for i in range(n)])
    Xi = iv.matrix([[iv.mpf(float(X[i, j])) for j in range(n)] for i in range(n)])
    R = Xi * Mi
    s = iv.mpf(0)
    for i in range(n):
        for j in range(n):
            d = (iv.mpf(1) if i == j else iv.mpf(0)) - R[i, j]
            s += d * d
    rho = iv.sqrt(s)
    sx = iv.mpf(0)
    for i in range(n):
        for j in range(n):
            sx += Xi[i, j] * Xi[i, j]
    nX = iv.sqrt(sx)
    rho_hi = float(mpmath.mpf(rho.b)); nX_hi = float(mpmath.mpf(nX.b))
    lo = (1.0 - rho_hi) / nX_hi if rho_hi < 1 else 0.0
    return {"rho_hi": rho_hi, "normXF_hi": nX_hi, "lo_frobenius": lo * (1 - 1e-15)}
