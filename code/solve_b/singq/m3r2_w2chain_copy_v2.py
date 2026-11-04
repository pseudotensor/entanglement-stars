# m3r2_w2chain_copy_v2.py -- VERBATIM copy of vref_r1's <internal>/w2chain_v2.py (md5 a8bd80a8bc0a58a8f4db7f77a7070e33) for
# route 2's amendment-33 Stage 3 (full-flow W_2 with clock dependence). Unchanged below this header.
"""w2chain_v2: chain-channel second-gradient vacuum polarization W_2 of the prepared band, EXACT in the flow amplitude at this gradient order
(ReviewB_w2fulla_v1 sec. 1, eqs. (2)-(4); paper 2927a70d9c9 eq:dm-reference), standalone module (route2 route; lead #353).

DEFINITION. Channel kappa (k), chain sites r_n (a = 1), d = (sin p, k/r, m), eps = |d|, n = d/eps, q = sin(2p)/2, q' = cos 2p,
n_xi = (I - n n^T) d_xi / eps, d_p = (cos p, 0, 0), d_r = (0, -k/r^2, 0). Per site
  A_n = int dp/2pi q^2 |n_p|^2/(8 eps),  B_n = int dp/2pi q q' n_p.n_r/(8 eps),  C_n = int dp/2pi q'^2 |n_r|^2/(8 eps),
  F_n = A_n (Dv)_n^2 - 2 B_n v_n (Dv)_n + C_n v_n^2,   W_2[N, v] = - sum_n F_n / N_n     (per copy of the channel; weight 16 k by the caller),
with D the discrete radial derivative (centred, one-sided at the two ends) acting on the site field v. DISCRETIZED FIRST: the field derivatives
below are the exact gradients of this discrete functional (adjoint of D), not discretized continuum formulas:
  dW_2/dN_n = F_n / N_n^2   (the clock source t_N),
  dW_2/dv_n = -(2 C_n v_n - 2 B_n (Dv)_n)/N_n - [D^T ((2 A_n (Dv)_n - 2 B_n v_n)/N_n)]_n.
Preparation subtraction (eq:dm-reference): r = D W[u] uses W_2[u] - DW_2[u_star](u); its field derivatives are dW_2[u] - dW_2[u_star] (sources_vs_prep).
Degree one: W_2[lambda N, lambda v] = lambda W_2[N, v] (exact, discrete). The A coefficient equals w2chain_v1's c_aa = (1/2) kappa_stiff(M); B, C carry
the mass-vector (k/r)' terms (reviewer B: "do not omit v q' n_r").

SCOPE. Bulk formula; endpoint terms not certified (reviewer B). The expansion parameter eta_M = k/(k^2 + (m r)^2) is reported by eta_M(); reviewer B's
explicit-exclusion diagnostic: exclude eta_M > 0.05 (r < 50a for every k at ma = 0.2); the chain check (VACLAW_CHAIN_CHECK_route2_v1) failed in the core.
Low-k massless channels are excluded (eta_M = 1/k at all r).

API
  coeffs(n_sites, kappa, m, Np=4096, r=None) -> dict(A, B, C, r, D)        (site coefficient arrays and the difference matrix)
  W2(N, v, C)                -> float
  grad_v(N, v, C), grad_N(N, v, C) -> arrays (exact discrete gradients)
  sources_vs_prep(N, v, N_star, v_star, C) -> (dW/dN[u] - dW/dN[u_star], dW/dv[u] - dW/dv[u_star])
  eta_M(r, kappa, m)
"""
import numpy as np
import scipy.sparse as sp


def _Dmat(n):
    D = sp.lil_matrix((n, n))
    for i in range(1, n - 1):
        D[i, i + 1] = 0.5; D[i, i - 1] = -0.5
    D[0, 1] = 1.0; D[0, 0] = -1.0
    D[n - 1, n - 1] = 1.0; D[n - 1, n - 2] = -1.0
    return D.tocsr()


def coeffs(n_sites, kappa, m, Np=4096, r=None, chunk=512):
    r = np.arange(1, n_sites + 1, dtype=float) if r is None else np.asarray(r, float)
    p = 2 * np.pi * np.arange(Np) / Np
    sp_, cp = np.sin(p), np.cos(p)
    q = sp_ * cp; qp = np.cos(2 * p)
    A = np.zeros(len(r)); B = np.zeros(len(r)); Cc = np.zeros(len(r))
    for a in range(0, len(r), chunk):
        rr = r[a:a + chunk][:, None]
        d = np.stack([np.broadcast_to(sp_, (len(rr), Np)), np.broadcast_to(kappa / rr, (len(rr), Np)), np.full((len(rr), Np), float(m))])
        eps = np.sqrt(np.sum(d ** 2, 0)); n = d / eps
        dp = np.stack([np.broadcast_to(cp, eps.shape), np.zeros(eps.shape), np.zeros(eps.shape)])
        dr = np.stack([np.zeros(eps.shape), np.broadcast_to(-kappa / rr ** 2, eps.shape), np.zeros(eps.shape)])
        npv = (dp - n * np.sum(n * dp, 0)) / eps
        nrv = (dr - n * np.sum(n * dr, 0)) / eps
        w = 1.0 / (8 * eps * Np)
        A[a:a + len(rr)] = np.sum(q ** 2 * np.sum(npv ** 2, 0) * w, 1)
        B[a:a + len(rr)] = np.sum(q * qp * np.sum(npv * nrv, 0) * w, 1)
        Cc[a:a + len(rr)] = np.sum(qp ** 2 * np.sum(nrv ** 2, 0) * w, 1)
    return {"A": A, "B": B, "C": Cc, "r": r, "D": _Dmat(len(r)), "kappa": kappa, "m": m, "Np": Np}


def _F(v, C):
    dv = C["D"] @ v
    return C["A"] * dv ** 2 - 2 * C["B"] * v * dv + C["C"] * v ** 2, dv


def W2(N, v, C):
    F, _ = _F(np.asarray(v, float), C)
    return float(-np.sum(F / np.asarray(N, float)))


def grad_N(N, v, C):
    F, _ = _F(np.asarray(v, float), C)
    return F / np.asarray(N, float) ** 2


def grad_v(N, v, C):
    v = np.asarray(v, float); N = np.asarray(N, float)
    F, dv = _F(v, C)
    explicit = -(2 * C["C"] * v - 2 * C["B"] * dv) / N
    via_D = -(C["D"].T @ ((2 * C["A"] * dv - 2 * C["B"] * v) / N))
    return explicit + via_D


def sources_vs_prep(N, v, N_star, v_star, C):
    return grad_N(N, v, C) - grad_N(N_star, v_star, C), grad_v(N, v, C) - grad_v(N_star, v_star, C)


def eta_M(r, kappa, m):
    r = np.asarray(r, float)
    return kappa / (kappa ** 2 + (m * r) ** 2)
