#!/usr/bin/env python
"""page_ops_v1.py (lead reviewer A, GAP03 integration; builder analyst runs it). The radial channel operators of the declared model,
re-derived from engine_ops_v1.py and engine_vacflow_v1.py with identical conventions (compared term by term in page_selftest_v1 when the
engine is importable), the declared exterior-plus-ball flow and its back-averaging with the engine's l = 1 kernel (engine_fields_v2.kernels),
banded application of the clocked operator on the device, the Chebyshev exponential and the commutator-free fourth-order step for a
time-dependent flow, dense eigendecompositions, Gaussian entropies and per-half orthonormalization. Lattice units a = 1; N = 1 everywhere
(the vacuum exterior and the marginally bound ball both carry unit clock); torch complex128 on cuda or cpu.

Conventions (engine_ops_v1 docstring, spec 2.1): channel kappa = 1, 2, ...; 2 components per site; R_n = n + 1 (half offset: n + 1/2);
on-site m sigma_z + (kappa/R_n) sigma_y; sine bond -(i/2) sigma_x from site n to n + 1 (symbol sin k sigma_x); two-step current
Q = -(i/4) 1 from site n to n + 2 (symbol sin(2k)/2, unit slope at every valley); h[N, v] = (1/2){N, h_0} + (1/2){v, Q}: the bond (x, y)
carries (N_x + N_y)/2 h_xy and (v_x + v_y)/2 Q_xy; the weight of every chain mode is 16 kappa (8 cubic valleys times 2j + 1 = 2 kappa;
the chain's own two valleys are the continuum channels +kappa and -kappa). Flattened index 2 n + a (site n, component a)."""
import math

import numpy as np
import scipy.sparse as sp
from scipy.special import jv

import torch

CD = torch.complex128
RD = torch.float64
SQ3 = math.sqrt(3.0)
CF4_C = (0.5 - SQ3 / 6.0, 0.5 + SQ3 / 6.0)                 # the Gauss nodes of the commutator-free fourth-order step
CF4_ALPHA = ((3.0 + 2.0 * SQ3) / 12.0, (3.0 - 2.0 * SQ3) / 12.0)   # alpha_1 = 0.5387 (the earlier node in the first factor), alpha_2 = -0.0387


# ----------------------------------------------------------------------------------------------------------------------------- chain
def chain_sparse(R, m, kappa):
    """h_0 (unit clock, no flow) and the two-step Q as host CSR matrices (2n x 2n), engine_vacflow_v1.chain_sparse's entries, vectorized."""
    R = np.asarray(R, float)
    n = len(R)
    even = 2 * np.arange(n)
    rows = np.concatenate([even, even + 1, even, even + 1])
    cols = np.concatenate([even, even + 1, even + 1, even])
    vals = np.concatenate([np.full(n, m, complex), np.full(n, -m, complex), -1j * kappa / R, 1j * kappa / R])
    b = 2 * np.arange(n - 1)
    # -(i/2) sigma_x from site i to i + 1: (b, b + 3), (b + 1, b + 2) and the conjugates
    rows = np.concatenate([rows, b, b + 3, b + 1, b + 2])
    cols = np.concatenate([cols, b + 3, b, b + 2, b + 1])
    vals = np.concatenate([vals, np.full(n - 1, -0.5j), np.full(n - 1, 0.5j), np.full(n - 1, -0.5j), np.full(n - 1, 0.5j)])
    h0 = sp.csr_matrix((vals, (rows, cols)), shape=(2 * n, 2 * n))
    t = np.arange(2 * n - 4)
    Q = sp.csr_matrix((np.concatenate([np.full(2 * n - 4, -0.25j), np.full(2 * n - 4, 0.25j)]),
                       (np.concatenate([t, t + 4]), np.concatenate([t + 4, t]))), shape=(2 * n, 2 * n))
    return h0, Q


def clocked_csr(h0, Q, N, v):
    """h[N, v] = (1/2){N, h_0} + (1/2){v, Q} as a host CSR (engine_collapse_v19.clocked_host_sparse)."""
    Nd = sp.diags(np.repeat(np.asarray(N, float), 2))
    Vd = sp.diags(np.repeat(np.asarray(v, float), 2))
    return (0.5 * (Nd @ h0 + h0 @ Nd) + 0.5 * (Vd @ Q + Q @ Vd)).tocsr()


def velocity_csr(h, R):
    """i[h, X] with X the radius on both components: (i h_ab (R_b - R_a)); its symbol is the group velocity d omega/dk."""
    co = h.tocoo()
    r2 = np.repeat(np.asarray(R, float), 2)
    return sp.csr_matrix((1j * co.data * (r2[co.col] - r2[co.row]), (co.row, co.col)), shape=co.shape)


def current_out_csr(h, R, R_c):
    """J_out = (i/2)[h^2, Pi_{R > R_c}], the outward Killing-energy current through the cut at R_c (d/dt Tr[C H_out] = Tr[C J_out] with
    H_out = (1/2){Pi_out, h}); h has range 2 sites so h^2 has range 4 and J_out is local at the cut."""
    pi = np.repeat((np.asarray(R, float) > R_c).astype(float), 2)
    h2 = (h @ h).tocoo()
    data = 0.5j * h2.data * (pi[h2.col] - pi[h2.row])
    keep = np.abs(data) > 0
    return sp.csr_matrix((data[keep], (h2.row[keep], h2.col[keep])), shape=h2.shape)


def gershgorin(A):
    """an upper bound on the spectral radius of the Hermitian CSR A: the largest row sum of |A|."""
    return float(np.max(np.asarray(abs(A).sum(axis=1)).ravel()))


class Banded:
    """a Hermitian banded host CSR applied on the device through its nonzero diagonals: Y_i = sum_d A_{i, i+d} X_{i+d}."""

    def __init__(self, A, device, dmax=16):
        A = A.tocsr()
        self.n2 = A.shape[0]
        self.diags = []
        for d in range(-dmax, dmax + 1):
            vals = A.diagonal(d)
            if np.any(vals != 0):
                self.diags.append((d, torch.as_tensor(np.asarray(vals, complex), dtype=CD, device=device)))
        self.bound = gershgorin(A)

    def apply(self, X):
        Y = torch.zeros_like(X)
        n2 = self.n2
        for d, D in self.diags:
            if d >= 0:
                Y[: n2 - d] += D[:, None] * X[d:]
            else:
                Y[-d:] += D[:, None] * X[: n2 + d]
        return Y


class ChainBanded:
    """h[1, v] = h_0 + (1/2){v, Q} applied on the device for a flow v given at call time: h_0's diagonals are fixed, the flow adds the
    +-4 diagonals -(i/4)(v_{n} + v_{n+2})/2 between site n and n + 2 (both components)."""

    def __init__(self, h0, Q, R, device):
        self.device = device
        self.n2 = h0.shape[0]
        self.nsites = len(R)
        self.h0 = Banded(h0, device)
        t = np.arange(self.n2 - 4)
        self.s_lo = torch.as_tensor(t // 2, dtype=torch.int64, device=device)
        self.s_hi = torch.as_tensor(t // 2 + 2, dtype=torch.int64, device=device)
        self.q4 = torch.as_tensor(np.asarray(Q.diagonal(4), complex), dtype=CD, device=device)
        self.qm4 = torch.as_tensor(np.asarray(Q.diagonal(-4), complex), dtype=CD, device=device)

    def to_dev(self, v):
        return torch.as_tensor(np.asarray(v, float), dtype=RD, device=self.device)

    def apply(self, X, v):
        """h[1, v] X for a device flow vector v (nsites,) float64."""
        Y = self.h0.apply(X)
        vavg = (0.5 * (v[self.s_lo] + v[self.s_hi])).to(CD)
        n2 = self.n2
        Y[: n2 - 4] += (self.q4 * vavg)[:, None] * X[4:]
        Y[4:] += (self.qm4 * vavg)[:, None] * X[: n2 - 4]
        return Y

    def bound(self, v):
        return self.h0.bound + 0.5 * float(v.abs().max())


# ---------------------------------------------------------------------------------------------------------------------- the flow
def flow_dense(rs, Rb):
    """the declared exterior-plus-ball flow: v = -sqrt(rs/r) for r >= Rb, -sqrt(rs/Rb^3) r inside (continuous at Rb; regular centre)."""
    def v(r):
        r = np.atleast_1d(np.asarray(r, float))
        out = -np.sqrt(rs / np.maximum(r, 1e-300))
        inside = r < Rb
        out[inside] = -np.sqrt(rs / Rb ** 3) * r[inside]
        return out
    return v


def kernels(r, R, ell):
    """K_0(r, R), K_1(r, R) (engine_fields_v2.kernels, verbatim structure): the density at r per unit volume from a unit total on the shell
    of radius R for a scalar (l = 0) and a radial vector (l = 1) density, K_l = (2 pi ell^2)^(-3/2) exp(-(r^2 + R^2)/(2 ell^2)) i_l(r R/ell^2),
    i_0 = sinh z/z, i_1 = cosh z/z - sinh z/z^2, in the exponentially scaled form; the series below z = 1e-2. r: (n_r,), R: (n_R,) -> (n_r, n_R)."""
    r = np.asarray(r, float)[:, None]
    R = np.asarray(R, float)[None, :]
    Em = np.exp(-(r - R) ** 2 / (2.0 * ell ** 2))
    Ep = np.exp(-(r + R) ** 2 / (2.0 * ell ** 2))
    EA = np.exp(-(r ** 2 + R ** 2) / (2.0 * ell ** 2))
    z = r * R / ell ** 2
    small = z < 1e-2
    zs = np.where(small, 1.0, z)
    i0 = np.where(small, EA * (1.0 + z ** 2 / 6.0), (Em - Ep) / (2.0 * zs))
    i1 = np.where(small, EA * (z / 3.0) * (1.0 + z ** 2 / 10.0), ((zs - 1.0) * Em + (zs + 1.0) * Ep) / (2.0 * zs ** 2))
    pref = 1.0 / ((2.0 * np.pi) ** 1.5 * ell ** 3)
    return pref * i0, pref * i1


class FlowAverager:
    """v_n = int 4 pi r^2 v(r) K_1(r, R_n) dr on a fine grid of spacing ell/16 (engine_fields_v12.site_fields_kernel_average's quadrature),
    the adjoint of the source averaging; mode 'point' returns v(R_n)."""

    def __init__(self, R, ell, mode="averaged", half_width=8.0, h_frac=1.0 / 16.0):
        self.R = np.asarray(R, float)
        self.mode = mode
        self.ell = float(ell)
        if mode == "point":
            return
        h = self.ell * h_frac
        rr = np.arange(0.0, self.R[-1] + half_width * self.ell + 1e-9, h)
        wtr = np.full(len(rr), h); wtr[0] = 0.5 * h
        rows, cols, v1, v0 = [], [], [], []
        for n, Rn in enumerate(self.R):
            lo = max(0, int(np.floor((Rn - half_width * self.ell) / h)))
            hi = min(len(rr), int(np.ceil((Rn + half_width * self.ell) / h)) + 1)
            seg = rr[lo:hi]
            K0, K1 = kernels(seg, np.array([Rn]), self.ell)
            w = 4.0 * np.pi * seg * seg * wtr[lo:hi]
            rows.append(np.arange(lo, hi)); cols.append(np.full(hi - lo, n)); v1.append(w * K1[:, 0]); v0.append(w * K0[:, 0])
        rows = np.concatenate(rows); cols = np.concatenate(cols)
        self.rr = rr
        self.A1 = sp.csr_matrix((np.concatenate(v1), (cols, rows)), shape=(len(self.R), len(rr)))
        self.A0 = sp.csr_matrix((np.concatenate(v0), (cols, rows)), shape=(len(self.R), len(rr)))
        self.norm0 = np.asarray(self.A0.sum(axis=1)).ravel()

    def __call__(self, vfun):
        if self.mode == "point":
            return vfun(self.R)
        return self.A1 @ vfun(self.rr)


def trapping_zeros(v_sites, R):
    """the outer and inner zeros of 1 - |v| on the site fields (N = 1), linearly interpolated; None where absent."""
    f = 1.0 - np.abs(np.asarray(v_sites, float))
    R = np.asarray(R, float)
    r_t = r_in = None
    up = np.where((f[:-1] <= 0) & (f[1:] > 0))[0]
    if len(up):
        i = up[-1]
        r_t = float(R[i] + (R[i + 1] - R[i]) * (-f[i]) / (f[i + 1] - f[i]))
    dn = np.where((f[:-1] > 0) & (f[1:] <= 0))[0]
    if len(dn):
        j = dn[0]
        r_in = float(R[j] + (R[j + 1] - R[j]) * f[j] / (f[j] - f[j + 1]))
    return r_t, r_in


def dust_radius(t, Rb0, rs):
    """the marginally bound ball's edge, N = 1 (eq. star-dust-control): R_b(t)^(3/2) = R_b0^(3/2) - (3/2) sqrt(r_s) t."""
    x = Rb0 ** 1.5 - 1.5 * math.sqrt(rs) * t
    return max(x, 0.0) ** (2.0 / 3.0)


def dust_time(Rb0, Rbf, rs):
    return (Rb0 ** 1.5 - Rbf ** 1.5) / (1.5 * math.sqrt(rs))


def ramp_c1(s):
    """the C^1 ramp s^2 (3 - 2 s) on [0, 1]."""
    s = min(max(float(s), 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


# ------------------------------------------------------------------------------------------------------------------- propagation
def cheb_expm(B, X, tau, v, tol=1e-15):
    """exp(-i tau h[1, v]) X by the Chebyshev series on [-w, w], w the Gershgorin bound of h[1, v]; tau of either sign
    (J_k(-z) = (-1)^k J_k(z) is handled by jv itself)."""
    if tau == 0.0:
        return X.clone()
    w = B.bound(v)
    z = w * tau
    K = int(abs(z) + 12)
    while abs(jv(K, abs(z))) > tol or K < abs(z) + 4:
        K += 4
    k = np.arange(K + 1)
    ck = jv(k, z) * ((-1j) ** k) * np.where(k == 0, 1.0, 2.0)
    T0 = X
    T1 = B.apply(X, v) / w
    Y = complex(ck[0]) * T0 + complex(ck[1]) * T1
    for kk in range(2, K + 1):
        T2 = 2.0 * B.apply(T1, v) / w - T0
        Y = Y + complex(ck[kk]) * T2
        T0, T1 = T1, T2
    return Y


def cf4_propagate(B, X, vfun_t, t0, t1, dt, backward=False):
    """U(t1, t0) X (or its adjoint for backward=True) for h(t) = h[1, v(t)], v linear in the Hamiltonian: the commutator-free fourth-order
    step U(t + D, t) = exp(-i (D/2) h[v_B]) exp(-i (D/2) h[v_A]) with v_A = 2(alpha_1 v(t_1) + alpha_2 v(t_2)), v_B = 2(alpha_2 v(t_1) +
    alpha_1 v(t_2)), t_{1,2} = t + (1/2 -+ sqrt 3/6) D (Blanes and Moan; the lane's flow_page_v2 and engine_vacflow_v1 scheme); the right
    factor acts first and weights the earlier node. vfun_t(t) returns the site flow (nsites,) as a numpy array."""
    nst = max(1, int(math.ceil((t1 - t0) / dt - 1e-12)))
    D = (t1 - t0) / nst
    a1, a2 = CF4_ALPHA
    steps = range(nst) if not backward else reversed(range(nst))
    for i in steps:
        t = t0 + i * D
        v1 = B.to_dev(vfun_t(t + CF4_C[0] * D)); v2 = B.to_dev(vfun_t(t + CF4_C[1] * D))
        vA = 2.0 * (a1 * v1 + a2 * v2); vB = 2.0 * (a2 * v1 + a1 * v2)
        if not backward:
            X = cheb_expm(B, X, 0.5 * D, vA)
            X = cheb_expm(B, X, 0.5 * D, vB)
        else:
            X = cheb_expm(B, X, -0.5 * D, vB)
            X = cheb_expm(B, X, -0.5 * D, vA)
    return X


# -------------------------------------------------------------------------------------------------------------------- dense algebra
def dense_from_csr(A, device):
    co = A.tocoo()
    M = torch.zeros(co.shape, dtype=CD, device=device)
    M.index_put_((torch.as_tensor(co.row, dtype=torch.int64, device=device), torch.as_tensor(co.col, dtype=torch.int64, device=device)),
                 torch.as_tensor(np.asarray(co.data, complex), dtype=CD, device=device), accumulate=True)
    return M


def eigh_csr(A, device):
    """the dense eigendecomposition of the Hermitian CSR A on the device: (e ascending float64, V complex128)."""
    M = dense_from_csr(A, device)                 # exactly Hermitian from the Hermitian CSR; eigh reads the lower triangle (no extra copy)
    e, V = torch.linalg.eigh(M)
    del M
    return e.to(RD), V


def binary_entropy_t(x):
    x = x.clamp(0.0, 1.0)
    return -(torch.special.xlogy(x, x) + torch.special.xlogy(1.0 - x, 1.0 - x))


def entropy_cov(M):
    """tr s(M) of a Hermitian covariance tensor (eigenvalues clipped to [0, 1]); returns (S, largest excursion outside [0, 1])."""
    if M.shape[0] == 0:
        return 0.0, 0.0
    ev = torch.linalg.eigvalsh(0.5 * (M + M.mH))
    exc = max(0.0, float(-ev.min()), float(ev.max()) - 1.0)
    return float(binary_entropy_t(ev).sum()), exc


def sqrt_m_one_minus_m(M):
    """(M (I - M))^(1/2) for a Hermitian covariance tensor, through its eigendecomposition."""
    ev, U = torch.linalg.eigh(0.5 * (M + M.mH))
    evc = ev.clamp(0.0, 1.0)
    s = torch.sqrt((evc * (1.0 - evc)).clamp_min(0.0)).to(CD)
    return (U * s[None, :]) @ U.mH


def binary_entropy_np(x):
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    q = x[(x > 1e-300) & (x < 1.0 - 1e-16)]
    return float(np.sum(-q * np.log(q) - (1.0 - q) * np.log1p(-q)))


def orthonormalize_halves(cols, halves, eps_abs=1e-12, eps_rel=1e-14):
    """cols (n2 x k) with each column supported on one spectral half (halves: (k,) of +-1): an orthonormal basis per half by the Gram
    eigendecomposition (directions with Gram eigenvalue below max(eps_abs, eps_rel g_max) dropped). Returns (c_new, halves_new, T) with
    c_new = cols @ T (T is k x k')."""
    outs, hs, Tblocks = [], [], []
    k = cols.shape[1]
    for hsign in (1, -1):
        idx = (halves == hsign).nonzero().ravel()
        if idx.numel() == 0:
            continue
        X = cols[:, idx]
        G = X.mH @ X
        g, u = torch.linalg.eigh(0.5 * (G + G.mH))
        keep = g > max(eps_abs, eps_rel * float(g.max()))
        if not bool(keep.any()):
            continue
        Tk = u[:, keep] / torch.sqrt(g[keep]).to(CD)[None, :]
        outs.append(X @ Tk)
        hs.append(torch.full((int(keep.sum()),), hsign, dtype=torch.int64, device=cols.device))
        Tblocks.append((idx, Tk))
    if not outs:
        return cols[:, :0], halves[:0], torch.zeros((k, 0), dtype=CD, device=cols.device)
    kp = sum(t.shape[1] for _, t in Tblocks)
    T = torch.zeros((k, kp), dtype=CD, device=cols.device)
    off = 0
    for idx, Tk in Tblocks:
        T[idx, off: off + Tk.shape[1]] = Tk
        off += Tk.shape[1]
    return torch.cat(outs, dim=1), torch.cat(hs), T


def split_by_half(c, negmask):
    """columns c (n2 x k) in the eigenbasis -> (2k columns, half labels): the rows with e < 0 zeroed for the positive copy and the rows
    with e > 0 zeroed for the negative copy."""
    cpos = c.clone(); cpos[negmask] = 0.0
    cneg = c - cpos
    k = c.shape[1]
    halves = torch.cat([torch.ones(k, dtype=torch.int64, device=c.device), -torch.ones(k, dtype=torch.int64, device=c.device)])
    return torch.cat([cpos, cneg], dim=1), halves


def chart_retarded_time(t, rs, r_det):
    """the retarded chart time of a detection event at (t, r_det) on a stage of charge r_s = 2GE (eq. example-chart with u = T - r_*):
    u = t - 2 sqrt(r_s r) - r_s ln|(sqrt r - sqrt r_s)/(sqrt r + sqrt r_s)| - r - r_s ln(r/r_s - 1)."""
    sr, ss = math.sqrt(r_det), math.sqrt(rs)
    return t - 2.0 * sr * ss - rs * math.log(abs((sr - ss) / (sr + ss))) - r_det - rs * math.log(r_det / rs - 1.0)
