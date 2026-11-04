#!/usr/bin/env python
"""page_ops_v6.py. Changes against page_ops_v5.py (builder analyst, 2026-10-16; SPEC_final4b_engine.md 37.1 "Ruled" (2), inbox #31 S1's
repairs (1) and (2); the ideas of P01_theorems.md sec. 2 and SHARP_ENTROPY_THEOREM.md re-derived, not used as sources):
(S1 repair 1) hull_in_coords takes mode "exact" (the default): for every half whose block of the kept eigenvectors is nonzero it keeps every
    left singular vector the thin SVD returns (at most min(r_h, k) per half, at most twice the kept count; no eps_orth threshold and no
    division by a singular value), so the kept eigenvectors lie in the hull exactly (L U_keep = U_keep) and the compression's reset is
    supported on the dropped eigenspace; mode "v5" is v5's threshold s^2 > max(eps_abs, eps_rel s_max^2), bitwise. An optional stats dict
    records, per call, the columns kept, the columns v5's rule would have dropped and the smallest kept singular value.
(S1 repair 2) new pair_ledger_sharp(sines): the dropped-pair ledger sum_j F(s_j) with F(s) <= min{C* arcsin s, 2 h(min(s, 1/2))},
    C* = 1.325486838698363 (the maximum of 2 sqrt(p(1-p)) |ln((1-p)/p)|, at x tanh x = 1 with x = ln((1-p)/p)/2), over the individual
    dropped sines (never h(s^2), never the count times the largest); pair_ledger_v5(sines) is v5's sum_j h(s_j^2), printed beside.
    new rank_sensitive_bound(q, delta) = q h(min(delta/q, 1/2)) (0 for q = 0): the entropy modulus for two covariance contractions whose
    difference has rank at most q and trace norm at most delta, printed for the compression's measured reset residual.
Nothing else changes; every v5 function keeps its arithmetic.

page_ops_v5.py. Changes against page_ops_v4.py (builder analyst; SPEC_final4b_engine.md 33.1 "Ruled" (6) and 33.6 BUILDER item 1):
(33.1 (6)) new device_memory and reset_device_peak: the device's allocated memory and its peak (torch.cuda.memory_allocated and
torch.cuda.max_memory_allocated, GB) and the reset of the peak, so that page_core_v5 records the lower-memory transition's phases and
page_recursion_v5 prints the device peak per transition; on a CPU device both report None. Nothing else changes.

page_ops_v4.py. Changes against page_ops_v3.py (builder analyst; SPEC_final4b_engine.md 31.4 and 31.6 BUILDER item 1; the GAP03 review
review_GAP03_page_curve.tex): (R6) cheb_expm stops the Chebyshev series by the tail sum 2 sum_{j>K} |J_j(z)| below tol (v3 checked one
Bessel coefficient) and accumulates the tail bounds of every call in CHEB (cheb_stats() and cheb_delta() print the accumulated bound, the
call count and the largest order; the bound is the operator-norm error of the series on the Gershgorin interval, per unit column);
(R6) new FlowAverager.quadrature_check: the averaged flow at the declared quadrature (ell/16, 8 ell) against ell/32 and 10 ell, and the
l = 0 norms in the bulk; (R5, D8's dimension factor) orthonormalize_halves records in an optional stats dict the dropped Gram directions
(their count, the largest dropped Gram eigenvalue and the input dimension) so the caller can enter them in the truncation bound; new
ramp_rate_max (the largest d sigma/dt of the C^1 ramp over a sub-interval of its parameter, for 31.4 (1)'s per-sub-step ||h_dot||);
new binary_entropy_scalar. Nothing else changes.

page_ops_v3.py. Changes against page_ops_v2.py (builder analyst, BRIEF_page_v3_builder.md): new number_current_csr, the outward number
current i[h, Pi_{R > R_c}] through a cut, for the branch-resolved transmission (SPEC_final4b_engine.md 30.1 (5)). Nothing else changes.

page_ops_v2.py. Changes against page_ops_v1.py (builder analyst, BRIEF_page_debug_v1.md): orthonormalize_halves evaluates each half's
Gram eigendecomposition through the QR and SVD of that half's columns restricted to that half's rows (Gram eigenvalues g = s^2, the same
drop rule g <= max(eps_abs, eps_rel g_max)) and returns the left singular vectors, which are orthonormal to machine precision and exactly
zero on the other half's rows (new argument negmask, the rows with e < 0); v1's X W diag(g)^(-1/2) from the explicit X^dagger X lost
orthonormality as eps g_max/g (6.7e-3 on the T4 initial memory, then repeated collection), which failed T4d, T4g (Gram defect 69.8 on
the H100, 80.9 on the box CPU), T4e on the CPU and corrupted T4f and T4i; an unrestricted QR keeps the halves orthonormal but leaks
rounding onto the other half's rows, which grows block by block (measured), hence the row restriction. New helpers for page_core_v2:
project_out (classical Gram-Schmidt twice against an orthonormal set) and hull_in_coords (the split hull of spec 4.2 (g) computed in the
memory's own coordinates, so that it lies in the memory's span exactly). Nothing else changes.

page_ops_v1.py (lead reviewer A, GAP03 integration; builder analyst runs it). The radial channel operators of the declared model,
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


def number_current_csr(h, R, R_c):
    """v3 (30.1 (5)): the outward number current through the cut at R_c, i[h, Pi_{R > R_c}] (d/dt Tr[C Pi_out] = Tr[C i[h, Pi_out]]): the
    entries i h_ab (pi_b - pi_a), nonzero only on the bonds that cross the cut (h has range 2 sites, so 4 sites about the cut)."""
    co = h.tocoo()
    pi = np.repeat((np.asarray(R, float) > R_c).astype(float), 2)
    data = 1j * co.data * (pi[co.col] - pi[co.row])
    keep = np.abs(data) > 0
    return sp.csr_matrix((data[keep], (co.row[keep], co.col[keep])), shape=co.shape)


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

    def quadrature_check(self, vfun, bulk_min_over_ell=8.0):
        """v4 (R6): the averaged flow at this quadrature (spacing ell h_frac, half-width 8 ell) against spacing ell/32 and half-width 10 ell
        (max |dv| over the sites), and the l = 0 norms' largest deviation from 1 over the sites beyond bulk_min_over_ell ell (the centre's
        parity images make the norm differ from 1 there by construction)."""
        if self.mode == "point":
            return {"mode": "point", "max_abs_dv": 0.0, "max_abs_norm0_minus_1_bulk": 0.0}
        fine = FlowAverager(self.R, self.ell, self.mode, half_width=10.0, h_frac=1.0 / 32.0)
        v0 = self(vfun); v1 = fine(vfun)
        bulk = self.R > bulk_min_over_ell * self.ell
        return {"mode": self.mode, "max_abs_dv": float(np.max(np.abs(v0 - v1))), "max_abs_v": float(np.max(np.abs(v0))),
                "max_abs_norm0_minus_1_bulk": float(np.max(np.abs(self.norm0[bulk] - 1.0))) if bulk.any() else None,
                "spacing_over_ell": [1.0 / 16.0, 1.0 / 32.0], "half_width_over_ell": [8.0, 10.0], "gaussian_tail_beyond_8_ell": math.exp(-32.0)}


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


def ramp_c1_inverse(sig):
    """v4: the u in [0, 1] with ramp_c1(u) = sig (bisection; the ramp is monotone)."""
    sig = min(max(float(sig), 0.0), 1.0)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ramp_c1(mid) < sig:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ramp_rate_max(sig_a, sig_b):
    """v4 (31.4 (1)): the largest d sigma/du = 6 u (1 - u) of the C^1 ramp while its parameter sigma runs over [sig_a, sig_b] (1.5 when
    the interval contains sigma = 1/2, else at the end nearer 1/2); ||h_dot|| over that sub-step is ||dH|| times this over tau."""
    ua, ub = ramp_c1_inverse(sig_a), ramp_c1_inverse(sig_b)
    if ua <= 0.5 <= ub:
        return 1.5
    u = ub if ub < 0.5 else ua
    return 6.0 * u * (1.0 - u)


def binary_entropy_scalar(x):
    """v4: s(x) = -x ln x - (1 - x) ln(1 - x) for one x, clipped to [0, 1/2] (the D8 lemma's per-dimension term)."""
    x = min(max(float(x), 0.0), 0.5)
    if x <= 0.0:
        return 0.0
    return float(-x * math.log(x) - (1.0 - x) * math.log1p(-x))


# ------------------------------------------------------------------------------------------------------------------- propagation
CHEB = {"calls": 0, "tail_bound_acc": 0.0, "K_max": 0, "tail_max": 0.0}


def cheb_stats():
    """v4 (R6): a copy of the accumulated Chebyshev record."""
    return dict(CHEB)


def cheb_delta(before):
    """v4 (R6): the Chebyshev record accumulated since `before` (a cheb_stats() copy): calls, the summed tail bound (the operator-norm
    error bound of the propagation per unit column, summed over the exponentials applied in sequence), the largest order."""
    return {"calls": CHEB["calls"] - before["calls"], "tail_bound_acc": CHEB["tail_bound_acc"] - before["tail_bound_acc"],
            "K_max": CHEB["K_max"], "tail_max": CHEB["tail_max"]}


def cheb_order(z, tol):
    """v4 (R6): the smallest K >= |z| + 4 with 2 sum_{j>K} |J_j(|z|)| <= tol, and that tail; the tail is summed from the computed
    coefficients up to |z| + 120 (beyond, |J_j| < (|z|/2)^j/j! is below 1e-300 for the steps used here)."""
    az = abs(float(z))
    jmax = int(az + 120)
    J = np.abs(jv(np.arange(jmax + 1), az))
    tail = 2.0 * np.concatenate([np.cumsum(J[::-1])[::-1][1:], [0.0]])          # tail[K] = 2 sum_{j>K} |J_j|
    K = int(math.ceil(az + 4))
    while K < jmax and tail[K] > tol:
        K += 1
    return K, float(tail[min(K, jmax)])


def cheb_expm(B, X, tau, v, tol=1e-15):
    """exp(-i tau h[1, v]) X by the Chebyshev series on [-w, w], w the Gershgorin bound of h[1, v]; tau of either sign
    (J_k(-z) = (-1)^k J_k(z) is handled by jv itself). v4 (R6): the order K by the tail sum 2 sum_{j>K} |J_j(z)| <= tol (cheb_order), the
    tail accumulated in CHEB."""
    if tau == 0.0:
        return X.clone()
    w = B.bound(v)
    z = w * tau
    K, tail = cheb_order(z, tol)
    CHEB["calls"] += 1; CHEB["tail_bound_acc"] += tail; CHEB["K_max"] = max(CHEB["K_max"], K); CHEB["tail_max"] = max(CHEB["tail_max"], tail)
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


def device_memory(device):
    """v5 (33.1 (6)): {"allocated_GB", "peak_GB"} of the device (torch.cuda.memory_allocated and torch.cuda.max_memory_allocated since
    the last reset_device_peak, in units of 1e9 bytes); None on a CPU device."""
    if device is None or getattr(device, "type", str(device)) != "cuda" or not torch.cuda.is_available():
        return None
    return {"allocated_GB": torch.cuda.memory_allocated(device) / 1e9, "peak_GB": torch.cuda.max_memory_allocated(device) / 1e9}


def reset_device_peak(device):
    """v5 (33.1 (6)): torch.cuda.reset_peak_memory_stats on a cuda device (the peak then restarts from the allocated memory); returns
    the peak before the reset (GB) or None on a CPU device."""
    m = device_memory(device)
    if m is None:
        return None
    torch.cuda.reset_peak_memory_stats(device)
    return m["peak_GB"]


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


def orthonormalize_halves(cols, halves, eps_abs=1e-12, eps_rel=1e-14, negmask=None, stats=None):
    """cols (n2 x k) with each column supported on one spectral half (halves: (k,) of +-1; negmask: (n2,) bool, the eigenbasis rows with
    e < 0): an orthonormal basis per half by the Gram eigendecomposition G = X^dagger X = W diag(g) W^dagger (directions with Gram
    eigenvalue below max(eps_abs, eps_rel g_max) dropped). v2: G is not formed; the half's columns restricted to the half's rows are
    factored X_h = Q R, R = U_R diag(s) W^dagger (so g = s^2 and W are G's eigenpairs), and the basis is Q U_R on the kept directions,
    zero on the other half's rows. Without negmask the rows are those where the half's columns are nonzero (exact for split columns).
    Returns (c_new, halves_new, T) with c_new = cols @ T up to rounding (T = W diag(s)^(-1) on the kept directions, k x k').
    v4 (R5): with stats (a dict) the dropped Gram directions are recorded: stats["n_dropped"] += their count, stats["g_dropped_max"] the
    largest dropped Gram eigenvalue, stats["dim"] the largest input dimension of a call with a drop (D8's dimension factor). In v4 the
    halves are those of the stage's reference mask (negmask = the occupied rows of the reference, which after a transition is the
    continued mask of 31.4 (1), not e < 0)."""
    outs, hs, Tblocks = [], [], []
    k = cols.shape[1]
    for hsign in (1, -1):
        idx = (halves == hsign).nonzero().ravel()
        if idx.numel() == 0:
            continue
        X = cols[:, idx]
        if negmask is not None:
            rows = (negmask if hsign < 0 else ~negmask).nonzero().ravel()
        else:
            rows = (X != 0).any(dim=1).nonzero().ravel()
        if rows.numel() == 0:
            continue
        Qh, Rh = torch.linalg.qr(X[rows], mode="reduced")
        Ur, s, Wh = torch.linalg.svd(Rh, full_matrices=False)
        g = s * s
        keep = g > max(eps_abs, eps_rel * float(g.max()))
        if stats is not None:
            nd = int((~keep).sum()) + max(0, int(idx.numel()) - int(g.numel()))
            if nd:
                stats["n_dropped"] = stats.get("n_dropped", 0) + nd
                gd = float(g[~keep].max()) if bool((~keep).any()) else 0.0
                stats["g_dropped_max"] = max(stats.get("g_dropped_max", 0.0), gd)
                stats["dim"] = max(stats.get("dim", 0), int(k))
        if not bool(keep.any()):
            continue
        out = torch.zeros((cols.shape[0], int(keep.sum())), dtype=CD, device=cols.device)
        out[rows] = Qh @ Ur[:, keep]
        Tk = Wh.mH[:, keep] / s[keep].to(CD)[None, :]
        outs.append(out)
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


def project_out(X, G):
    """v2: X minus its components on span(G) (G with orthonormal columns) by classical Gram-Schmidt twice."""
    if G is None or G.shape[1] == 0 or X.shape[1] == 0:
        return X
    X = X - G @ (G.mH @ X)
    return X - G @ (G.mH @ X)


def hull_in_coords(c, half, U, eps_abs=1e-12, eps_rel=1e-14, mode="exact", stats=None):
    """v2 (spec 4.2 (g), the split hull): for c (n2 x r) orthonormal with each column in one spectral half (half: (r,) of +-1) and
    U (r x k), the per-half orthonormal basis of the split components of the columns c U. The half-h candidates are c_h U_h (c_h the
    half-h columns of c, U_h the matching rows of U), whose Gram matrix is U_h^dagger U_h, so the column space of U_h from its SVD
    (Gram eigenvalues s^2, v1's drop rule) mapped by c_h is the same basis as v1's per-half orthonormalization of [c U_pos, c U_neg]; it
    lies in span(c) exactly instead of inheriting the full-space factorization's rounding. Returns (c_hull, half_hull).
    v6 (S1 repair 1): mode "exact" keeps every returned left singular vector of each nonzero half block (no threshold: the kept
    eigenvectors' halves are inside the hull exactly, P01 sec. 2's repair); mode "v5" applies v5's rule g > max(eps_abs, eps_rel g_max).
    stats (a dict, optional): "kept" += the columns kept, "v5_would_drop" += the columns v5's rule drops, "s_min_kept" the smallest kept
    singular value."""
    outs, hs = [], []
    for hsign in (1, -1):
        cols = (half == hsign).nonzero().ravel()
        if cols.numel() == 0:
            continue
        Uo, so, _ = torch.linalg.svd(U[cols, :], full_matrices=False)
        if so.numel() == 0:
            continue
        g = so * so
        keep_v5 = g > max(eps_abs, eps_rel * float(g.max()))
        if mode == "v5":
            keep = keep_v5
        else:
            if not bool((so > 0).any()):                                     # an exactly zero half block: nothing of the kept vectors there
                continue
            keep = torch.ones_like(keep_v5)
        if stats is not None:
            stats["kept"] = stats.get("kept", 0) + int(keep.sum())
            stats["v5_would_drop"] = stats.get("v5_would_drop", 0) + int((~keep_v5).sum())
            if bool(keep.any()):
                stats["s_min_kept"] = min(stats.get("s_min_kept", math.inf), float(so[keep].min()))
        if not bool(keep.any()):
            continue
        outs.append(c[:, cols] @ Uo[:, keep])
        hs.append(torch.full((int(keep.sum()),), hsign, dtype=torch.int64, device=c.device))
    if not outs:
        return c[:, :0], half[:0]
    return torch.cat(outs, dim=1), torch.cat(hs)


C_STAR = 1.325486838698363                                                   # v6 (S1 repair 2): max_p 2 sqrt(p(1-p)) |ln((1-p)/p)|


def pair_ledger_sharp(sines):
    """v6 (S1 repair 2): sum_j F(s_j) with F(s) <= min{C* arcsin s, 2 h(min(s, 1/2))} over the individual dropped sines (nats)."""
    s = np.clip(np.asarray(sines, float).ravel(), 0.0, 1.0)
    if s.size == 0:
        return 0.0
    a = C_STAR * np.arcsin(s)
    x = np.minimum(s, 0.5)
    hx = np.where(x > 0, -x * np.log(np.where(x > 0, x, 1.0)) - (1.0 - x) * np.log1p(-x), 0.0)
    return float(np.sum(np.minimum(a, 2.0 * hx)))


def pair_ledger_v5(sines):
    """v6: v5's dropped-pair ledger sum_j h(s_j^2) (printed beside the sharp ledger; not a bound for arbitrary cuts, SHARP_ENTROPY sec. 5)."""
    s = np.asarray(sines, float).ravel()
    return binary_entropy_np(s * s) if s.size else 0.0


def rank_sensitive_bound(q, delta):
    """v6: q h(min(delta/q, 1/2)) for a covariance change of rank at most q and trace norm at most delta (0 for q = 0)."""
    q = int(q)
    if q <= 0 or delta <= 0.0:
        return 0.0
    return q * binary_entropy_scalar(min(float(delta) / q, 0.5))


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
