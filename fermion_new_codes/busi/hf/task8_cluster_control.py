#!/usr/bin/env python3
"""
task8_cluster_control.py

Exact small-cluster comparison of the gravitational Hartree, Hartree-Fock and exact levels of the quadratic
reduced boundary Hamiltonian (strand_E_spectral_task8_control.tex, T8.1-T8.12): the specified N = 18 sector of
the open 5x1x1 parent (ma = 1, ell = 4a, R = 8a; 190 dimensions), its 4-hole sector (4845), the 2- and 4-hole
sectors of the open 7x1x1 parent (378, 20475) and the averaging-scale control ell/a = 2, 4, 8.

Units a = 1, ma = 1. Coupling G/a^2 (written G); T8's g = sqrt(G). complex128 throughout.
Conventions (manuscript Sec. II, RQ1):
  h0 = m beta + alpha_1 p_1, p_1 = (T - T^T)/(2i), Q_1 = (T^2 - (T^T)^2)/(4i), (T u)_n = u_{n+1} (open chain)
  e_x = {Pi_x, h0}/2, p_x = {Pi_x, Q_1}/2; vertices A_mu = (e_x)_x ++ (p_x)_x (2L of them)
  h[D] = sum_mu Phi_mu A_mu, Phi = (N_x, v_x) back-averaged with the l = 0 / l = 1 kernels (exact adjoint)
  M' = 4 pi r^2 (rho + w J), (log N)' = -4 pi G r J / w, w = -sqrt(2 G M / r), M(0) = 0, N(R) = 1
  C_{AB} = <c_B^dag c_A>, <c^dag A c> = Tr(C A); R(h) = continued occupied sheet (no refilling)
Computational symmetry (checked numerically): S = I_L x (I_2 x sigma_x) and Sigma = (site reflection) x beta
commute with every vertex; the one-particle space splits into 4 blocks (S, Sigma) = (+,+), (-,+), (+,-), (-,-)
and every fixed-number sector into blocks of conserved block numbers.
"""
import os, sys
if '--threads' in sys.argv:
    _t = sys.argv[sys.argv.index('--threads') + 1]
    for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(_v, _t)
import argparse, json, math, time
from itertools import combinations, combinations_with_replacement
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

FOURPI = 4.0 * math.pi
SQRT3H = math.sqrt(3.0) / 2.0


def log(*a):
    print(*a, flush=True)


def herm(X): return 0.5 * (X + X.conj().T)
def fnorm(X): return float(np.linalg.norm(X))
def opnorm(X): return float(np.linalg.norm(X, 2))
def trr(X): return float(np.trace(X).real)


def simpson_weights(n, h):
    W = np.ones(n + 1); W[1:-1:2] = 4.0; W[2:-1:2] = 2.0
    return W * h / 3.0


def cumsimpson(f, h):
    """Cumulative integral on a uniform grid with an even number of intervals."""
    f = np.asarray(f); n = len(f) - 1
    panels = h / 3.0 * (f[0:-1:2] + 4.0 * f[1::2] + f[2::2])
    I = np.zeros(n + 1, dtype=panels.dtype)
    I[2::2] = np.cumsum(panels)
    I[1::2] = I[0:-1:2] + h / 12.0 * (5.0 * f[0:-1:2] + 8.0 * f[1::2] - f[2::2])
    return I


def popcount64(x):
    x = np.asarray(x, dtype=np.uint64).copy()
    m1 = np.uint64(0x5555555555555555); m2 = np.uint64(0x3333333333333333)
    m4 = np.uint64(0x0F0F0F0F0F0F0F0F); h01 = np.uint64(0x0101010101010101)
    x = x - ((x >> np.uint64(1)) & m1)
    x = (x & m2) + ((x >> np.uint64(2)) & m2)
    x = (x + (x >> np.uint64(4))) & m4
    return ((x * h01) >> np.uint64(56)).astype(np.int64)


def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray): return jsonable(o.tolist())
    if isinstance(o, (np.bool_, bool)): return bool(o)
    if isinstance(o, (np.integer, int)): return int(o)
    if isinstance(o, complex): return [o.real, o.imag]
    if isinstance(o, (np.floating, float)):
        f = float(o); return f if math.isfinite(f) else str(f)
    return o


# ======================================================================= parent
class Parent:
    def __init__(self, L, m=1.0):
        assert L % 2 == 1
        self.L = L; self.d = 4 * L; self.m = m; self.nv = 2 * L
        self.xs = np.arange(L) - (L - 1) // 2
        T = np.zeros((L, L))
        for n in range(L - 1): T[n, n + 1] = 1.0
        self.p1 = (T - T.T) / (2j)
        self.Q1 = (T @ T - T.T @ T.T) / (4j)
        sx = np.array([[0, 1], [1, 0]], complex); sz = np.array([[1, 0], [0, -1]], complex); I2 = np.eye(2)
        self.beta = np.kron(sz, I2); self.alpha1 = np.kron(sx, sx)
        I4 = np.eye(4); IL = np.eye(L); Id = np.eye(self.d)
        self.h0 = np.kron(self.p1, self.alpha1) + m * np.kron(IL, self.beta)
        self.Q = np.kron(self.Q1, I4)
        self.Pi_sp = [np.diag(IL[n]) for n in range(L)]
        self.Pi = [np.kron(self.Pi_sp[n], I4) for n in range(L)]
        self.e = [herm(Pi @ self.h0) for Pi in self.Pi]   # (1/2){Pi_x, h0}
        self.pc = [herm(Pi @ self.Q) for Pi in self.Pi]   # (1/2){Pi_x, Q}
        self.A = self.e + self.pc
        eps, U = np.linalg.eigh(self.h0)
        self.eps0, self.U0 = eps, U
        self.N0 = int(np.sum(eps < 0)); self.Dplus = self.d - self.N0
        self.E_sea0 = float(np.sum(eps[eps < 0]))
        self.P0 = U[:, :self.N0] @ U[:, :self.N0].conj().T
        self.F0 = Id - self.P0
        self.A_e = [U.conj().T @ A @ U for A in self.A]
        # computational symmetry blocks
        self.S_op = np.kron(IL, np.kron(I2, sx))
        Pref = np.zeros((L, L)); Pref[np.arange(L), L - 1 - np.arange(L)] = 1.0
        self.Sig_op = np.kron(Pref, self.beta)
        self.sym_comm = max(max(fnorm(self.S_op @ A - A @ self.S_op), fnorm(self.Sig_op @ A - A @ self.Sig_op))
                            for A in self.A + [self.h0])
        self.blocks = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        self.Ub = []
        for s, g in self.blocks:
            Pb = herm(0.25 * (Id + s * self.S_op) @ (Id + g * self.Sig_op))
            w, Uw = np.linalg.eigh(Pb); self.Ub.append(np.ascontiguousarray(Uw[:, w > 0.5]))
        self.dims = [u.shape[1] for u in self.Ub]
        self.U_blk = np.concatenate(self.Ub, axis=1)
        self.block_of_mode = np.concatenate([np.full(dm, b) for b, dm in enumerate(self.dims)])
        self.Pb = [u @ u.conj().T for u in self.Ub]
        self.A_blk = [self.to_blk(A) for A in self.A]
        self.offblock = max(self.offblock_norm(X) for X in self.A_blk + [self.to_blk(self.h0)])
        self.eps0_blk = [np.linalg.eigh(u.conj().T @ self.h0 @ u)[0] for u in self.Ub]

    def to_blk(self, X): return self.U_blk.conj().T @ X @ self.U_blk
    def from_blk(self, X): return self.U_blk @ X @ self.U_blk.conj().T

    def offblock_norm(self, Xb):
        mask = self.block_of_mode[:, None] != self.block_of_mode[None, :]
        return fnorm(Xb * mask)

    def src(self, D):
        return np.array([np.trace(D @ A).real for A in self.A])

    def h_of(self, Phi):
        return sum(Phi[k] * self.A[k] for k in range(self.nv))

    def block_counts(self, C): return [trr(Pb @ C) for Pb in self.Pb]

    def E_free_pattern(self, pat):
        return float(sum(np.sum(self.eps0_blk[b][:n]) for b, n in enumerate(pat))) - self.E_sea0


def natural_patterns(parent, N):
    """Block occupations of the free lowest-N state; for an open shell the S-doublet-filled patterns."""
    eps = parent.eps0; tol = 1e-9; ef = eps[N - 1]
    if eps[N] - ef > tol:
        U = parent.U0[:, :N]; C = U @ U.conj().T
        return [tuple(int(round(x)) for x in parent.block_counts(C))], 'closed'
    below = int(np.sum(eps < ef - tol)); shell = np.where(np.abs(eps - ef) < tol)[0]; k = N - below
    Ub = parent.U0[:, :below]; base = [int(round(x)) for x in parent.block_counts(Ub @ Ub.conj().T)]
    Us = parent.U0[:, shell]; mult = [int(round(x)) for x in parent.block_counts(Us @ Us.conj().T)]
    pats = set()
    for comb in combinations_with_replacement(range(4), k):
        add = [sum(1 for c in comb if c == b) for b in range(4)]
        if all(add[b] <= mult[b] for b in range(4)):
            pats.add(tuple(base[b] + add[b] for b in range(4)))
    doublet = [p for p in sorted(pats) if p[0] == p[1] and p[2] == p[3]]
    return (doublet if doublet else sorted(pats)), 'open'


def free_state(parent, pat):
    C = np.zeros((parent.d, parent.d), complex)
    for b, u in enumerate(parent.Ub):
        nb = pat[b]
        if nb > 0:
            eb, Vb = np.linalg.eigh(u.conj().T @ parent.h0 @ u); occ = u @ Vb[:, :nb]; C += occ @ occ.conj().T
    return C


def natural_start(parent, C_Q, pat):
    C = np.zeros_like(C_Q)
    for b, u in enumerate(parent.Ub):
        nb = pat[b]
        if nb > 0:
            w, W = np.linalg.eigh(herm(u.conj().T @ C_Q @ u)); occ = u @ W[:, -nb:]; C += occ @ occ.conj().T
    return C


# ======================================================================= kernel
class KernelTables:
    """Angular-averaged l = 0 and l = 1 kernels kbar_0(r;x), kbar_1(r;x) of k_{ell,a} = K_ell * B_a on the data
    grid (step dr) and the fine grid (step dr/2). The l = 0 tables are normalized exactly under the quadrature
    that uses them (fine-grid Simpson = RK4 at G = 0 for the mass; data-grid Simpson for the back-averaging)."""

    def __init__(self, parent, ell, R, dr=0.01, nq=24):
        self.ell = float(ell); self.R = float(R); L = parent.L; xs = parent.xs
        self.Rs = float(np.max(np.abs(xs)) + self.ell + SQRT3H)
        if not self.Rs < self.R:
            raise ValueError('collar violated: support radius %.4f >= R = %.4f' % (self.Rs, self.R))
        ns = int(round(self.R / dr)); ns += ns % 2
        self.ns = ns; self.dr = self.R / ns
        self.r = np.linspace(0.0, self.R, ns + 1); self.rf = np.linspace(0.0, self.R, 2 * ns + 1)
        self.W = simpson_weights(ns, self.dr); self.Wf = simpson_weights(2 * ns, self.dr / 2.0)
        self.kappa = 9009.0 / (1024.0 * math.pi * self.ell ** 3)
        self.nq = nq
        KT0f, KT1f = self._tables(xs, nq)
        KT0 = np.ascontiguousarray(KT0f[::2]); KT1 = np.ascontiguousarray(KT1f[::2])
        w2 = self.W * FOURPI * self.r ** 2; w2f = self.Wf * FOURPI * self.rf ** 2
        self.norm_err_raw = float(np.max(np.abs(w2 @ KT0 - 1.0)))
        self.norm_err_raw_fine = float(np.max(np.abs(w2f @ KT0f - 1.0)))
        self.KT0 = KT0 / (w2 @ KT0)[None, :]; self.KT0f = KT0f / (w2f @ KT0f)[None, :]
        self.KT1 = KT1; self.KT1f = KT1f
        self.norm_err = float(np.max(np.abs(w2 @ self.KT0 - 1.0)))
        self.mom1_err = float(np.max(np.abs((w2 * self.r) @ self.KT1 - xs)))
        self.mom2_err = float(np.max(np.abs((w2 * self.r ** 2) @ self.KT0 - (xs ** 2 + self.ell ** 2 / 5.0 + 0.25))))
        self.nx = np.stack([cumsimpson(FOURPI * self.r ** 2 * self.KT0[:, x], self.dr) for x in range(L)], axis=1)

    def _t(self, s): return np.clip(1.0 - (s / self.ell) ** 2, 0.0, None)
    def _K(self, s): return self.kappa * self._t(s) ** 5
    def _F1(self, s): return -self.kappa * self.ell ** 2 / 12.0 * self._t(s) ** 6

    def _F3(self, s):
        t = self._t(s); return self.kappa * self.ell ** 4 / 2.0 * (-t ** 6 / 6.0 + t ** 7 / 7.0)

    def _tables(self, xs, nq):
        t, wt = np.polynomial.legendre.leggauss(nq); t = 0.5 * t; wt = 0.5 * wt
        Z1, Z2, Z3 = np.meshgrid(t, t, t, indexing='ij')
        Wq = (wt[:, None, None] * wt[None, :, None] * wt[None, None, :]).ravel()
        Z1 = Z1.ravel(); Z2 = Z2.ravel(); Z3 = Z3.ravel()
        nf = len(self.rf); L = len(xs)
        KT0 = np.zeros((nf, L)); KT1 = np.zeros((nf, L)); cache = {}
        for ix, x in enumerate(xs):
            ax = abs(int(x))
            if ax not in cache:
                c = np.sqrt((ax + Z1) ** 2 + Z2 ** 2 + Z3 ** 2); cos1 = (ax + Z1) / c
                k0 = np.zeros(nf); k1 = np.zeros(nf)
                k0[0] = float(np.sum(Wq * self._K(c)))
                chunk = 128
                for a0 in range(1, nf, chunk):
                    rr = self.rf[a0:a0 + chunk][:, None]; cc = c[None, :]
                    sp_ = rr + cc; sm_ = np.abs(rr - cc)
                    dF1 = self._F1(sp_) - self._F1(sm_); dF3 = self._F3(sp_) - self._F3(sm_)
                    F0 = dF1 / (2.0 * rr * cc)
                    G1 = ((rr ** 2 + cc ** 2) * dF1 - dF3) / (4.0 * rr ** 2 * cc ** 2)
                    k0[a0:a0 + chunk] = F0 @ Wq
                    k1[a0:a0 + chunk] = (G1 * cos1[None, :]) @ Wq
                if ax == 0: k1[:] = 0.0
                cache[ax] = (k0, k1)
            k0, k1 = cache[ax]
            KT0[:, ix] = k0; KT1[:, ix] = k1 * np.sign(x) if x != 0 else 0.0
        return KT0, KT1


def kernel_convergence(parent, ell, R, dr, nq1, nq2):
    k1 = KernelTables(parent, ell, R, dr, nq1); k2 = KernelTables(parent, ell, R, dr, nq2)
    w2 = FOURPI * k1.r ** 2
    return dict(dKT0=float(np.max(np.abs(w2[:, None] * (k1.KT0 - k2.KT0)))),
                dKT1=float(np.max(np.abs(w2[:, None] * (k1.KT1 - k2.KT1)))),
                norm_err_raw=k2.norm_err_raw, mom1_err=k2.mom1_err, mom2_err=k2.mom2_err)


# ======================================================================= radial rows
class NegativeMass(Exception):
    pass


def radial_solve(e, j, G, kt):
    """RQ1 on the data grid. e, j: (L,) site energies Tr(D e_x) and currents Tr(D p_x), real or complex
    (complex-step differentiation). Returns profiles, back-averaged fields Phi = (N_x, v_x) and E = M(R)."""
    rf = kt.rf; ns = kt.ns
    rhof = kt.KT0f @ e; Jf = kt.KT1f @ j
    dt = np.result_type(e, j, np.float64)
    M = np.zeros(ns + 1, dtype=dt)

    def fval(idx, MM):
        if idx == 0: return 0.0 * MM
        rr = rf[idx]
        w = -np.sqrt(2.0 * G * MM / rr) if G > 0.0 else 0.0
        return FOURPI * rr * rr * (rhof[idx] + w * Jf[idx])

    for k in range(ns):
        i0, i1, i2 = 2 * k, 2 * k + 1, 2 * k + 2; h = rf[i2] - rf[i0]; m0 = M[k]
        k1 = fval(i0, m0); k2 = fval(i1, m0 + 0.5 * h * k1)
        k3 = fval(i1, m0 + 0.5 * h * k2); k4 = fval(i2, m0 + h * k3)
        M[k + 1] = m0 + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    r = kt.r
    if np.min(M.real[1:]) <= 0.0:
        bad = r[1:][M.real[1:] <= 0.0][0]
        raise NegativeMass('M(r) <= 0 at r = %.4f (positive-mass branch fails)' % bad)
    rho = rhof[::2]; J = Jf[::2]
    w = np.zeros(ns + 1, dtype=dt); q = np.zeros(ns + 1, dtype=dt)
    if G > 0.0:
        w[1:] = -np.sqrt(2.0 * G * M[1:] / r[1:])
        q[1:] = -FOURPI * G * r[1:] * J[1:] / w[1:]
    Q = cumsimpson(q, kt.dr); logN = Q - Q[-1]
    N = np.exp(logN); v = N * w
    wgt = kt.W * FOURPI * r * r
    Nx = (wgt * N) @ kt.KT0; vx = (wgt * v) @ kt.KT1
    return dict(M=M, w=w, N=N, v=v, logN=logN, q=q, rho=rho, J=J, Nx=Nx, vx=vx,
                E=M[-1], Phi=np.concatenate([Nx, vx]))


def continued_projector(h, Pprev, N0):
    eps, U = np.linalg.eigh(h)
    ov = np.real(np.einsum('ia,ij,ja->a', U.conj(), Pprev, U))
    occ = np.sort(np.argsort(-ov)[:N0]); unocc = np.setdiff1d(np.arange(len(eps)), occ)
    R = U[:, occ] @ U[:, occ].conj().T
    gap = float(eps[unocc].min() - eps[occ].max())
    lowest = bool(np.array_equal(occ, np.arange(N0)))
    return R, eps, U, occ, gap, lowest


def riesz_check(h, eps, occ, npts=128):
    """Riesz projector on a circle around the occupied sheet (meaningful when the sheet is the lowest set)."""
    emin = eps[occ].min(); emax = eps[occ].max()
    unocc = np.setdiff1d(np.arange(len(eps)), occ); gap = eps[unocc].min() - emax
    c0 = 0.5 * (emin + emax); rad = 0.5 * (emax - emin) + 0.5 * gap
    th = 2 * np.pi * (np.arange(npts) + 0.5) / npts
    R = np.zeros_like(h); I = np.eye(len(eps))
    for t in th:
        z = c0 + rad * np.exp(1j * t)
        R += (rad * np.exp(1j * t) / npts) * np.linalg.inv(z * I - h)
    U = np.linalg.eigh(h)[1]; Re = U[:, occ] @ U[:, occ].conj().T
    return fnorm(R - Re)


class Background:
    pass


def solve_source(parent, kt, G, C, Pprev, ref, tol=1e-13, maxit=500):
    """SB4: D = C - R(h[D]) (moving reference) or D = C - P0 (fixed)."""
    L = parent.L; bg = Background(); bg.ref = ref; bg.G = G
    if ref == 'fixed' or G == 0.0:
        D = C - parent.P0; s = parent.src(D); sol = radial_solve(s[:L], s[L:], G, kt)
        h = parent.h_of(sol['Phi'].real)
        Rd, eps, U, occ, gap, lowest = continued_projector(h, parent.P0, parent.N0)
        R = parent.P0 if ref == 'fixed' else Rd
        D = C - R
        bg.inner_iterations = 0; bg.inner_residual = 0.0
    else:
        R = Pprev; D = C - R; err = np.inf
        for it in range(maxit):
            s = parent.src(D); sol = radial_solve(s[:L], s[L:], G, kt); h = parent.h_of(sol['Phi'].real)
            Rn, eps, U, occ, gap, lowest = continued_projector(h, R, parent.N0)
            Dn = C - Rn; err = fnorm(Dn - D); D, R = Dn, Rn
            if err < tol: break
        s = parent.src(D); sol = radial_solve(s[:L], s[L:], G, kt); h = parent.h_of(sol['Phi'].real)
        R, eps, U, occ, gap, lowest = continued_projector(h, R, parent.N0)
        bg.inner_iterations = it + 1; bg.inner_residual = float(err)
    bg.D, bg.R, bg.h, bg.sol = D, R, h, sol
    bg.s = parent.src(D); bg.Phi = sol['Phi'].real; bg.E = float(sol['E'].real)
    bg.eps, bg.U, bg.occ, bg.gap, bg.sheet_is_lowest = eps, U, occ, gap, lowest
    r = kt.r; M = sol['M'].real
    bg.mass_margin = float(np.min(M[1:] / r[1:] ** 3)); bg.rho_center = float(sol['rho'][0].real)
    bg.compactness = 2.0 * G * bg.E / kt.Rs; bg.N_center = float(sol['N'][0].real)
    bg.wmax = float(np.max(np.abs(sol['w'].real))); bg.min_abs_eps = float(np.min(np.abs(eps)))
    bg.dP0_op = opnorm(R - parent.P0); bg.dP0_fro = fnorm(R - parent.P0)
    bg.Em = trr(h @ D)
    return bg


# ======================================================================= Hessian, response, V
def hessian_B(parent, kt, G, sbar, eta=1e-30):
    """Complex-step dPhi/ds (Hessian of M(R) in source coordinates) and dM(r_j)/ds (T8.7's b(r))."""
    L = parent.L; n2 = 2 * L
    B = np.zeros((n2, n2)); bM = np.zeros((kt.ns + 1, n2))
    for nu in range(n2):
        s = sbar.astype(complex); s[nu] += 1j * eta
        sol = radial_solve(s[:L], s[L:], G, kt)
        B[:, nu] = sol['Phi'].imag / eta; bM[:, nu] = sol['M'].imag / eta
    return B, bM


def hessian_fd(parent, kt, G, sbar, eta=1e-5):
    L = parent.L; n2 = 2 * L; B = np.zeros((n2, n2))
    for nu in range(n2):
        sp_ = sbar.copy(); sp_[nu] += eta; sm_ = sbar.copy(); sm_[nu] -= eta
        Pp = radial_solve(sp_[:L], sp_[L:], G, kt)['Phi'].real
        Pm = radial_solve(sm_[:L], sm_[L:], G, kt)['Phi'].real
        B[:, nu] = (Pp - Pm) / (2.0 * eta)
    return B


def hessian_lin(parent, kt, G, sol):
    """Linearized radial equations (RQ4 at n = 1) in a data-grid Simpson discretization."""
    L = parent.L; n2 = 2 * L; r = kt.r
    w = sol['w'].real; N = sol['N'].real; J = sol['J'].real
    wgt = kt.W * FOURPI * r * r
    B = np.zeros((n2, n2))
    for nu in range(n2):
        de = np.zeros(L); dj = np.zeros(L)
        if nu < L: de[nu] = 1.0
        else: dj[nu - L] = 1.0
        drho = kt.KT0 @ de; dJ = kt.KT1 @ dj
        Z = cumsimpson(FOURPI * r * r * N * (drho + w * dJ), kt.dr); dM = Z / N
        dw = np.zeros_like(r); dq = np.zeros_like(r)
        if G > 0.0:
            dw[1:] = G * dM[1:] / (r[1:] * w[1:])
            dq[1:] = -FOURPI * G * r[1:] * (dJ[1:] / w[1:] - J[1:] * dw[1:] / w[1:] ** 2)
        dQ = cumsimpson(dq, kt.dr); dlogN = dQ - dQ[-1]
        dN = N * dlogN; dv = dN * w + N * dw
        B[:L, nu] = (wgt * dN) @ kt.KT0; B[L:, nu] = (wgt * dv) @ kt.KT1
    return B


def response_PA(parent, bg):
    """P^A_{mu nu} = Tr(A_mu P(A_nu)), P the spectral derivative of the continued projector of hbar."""
    U = bg.U; eps = bg.eps; d = parent.d
    n = np.zeros(d); n[bg.occ] = 1.0
    num = n[:, None] - n[None, :]; den = eps[:, None] - eps[None, :]
    Wm = np.zeros((d, d)); mask = num != 0; Wm[mask] = num[mask] / den[mask]
    At = [U.conj().T @ A @ U for A in parent.A]
    nv = parent.nv; PA = np.zeros((nv, nv))
    for mu in range(nv):
        for nu in range(nv):
            PA[mu, nu] = float(np.sum(Wm * At[nu] * At[mu].T).real)
    return 0.5 * (PA + PA.T), Wm, At


def build_V(B, PA):
    n = B.shape[0]; I = np.eye(n); Jm = I + PA @ B; Jinv = np.linalg.inv(Jm)
    Vraw = B @ Jinv; V = 0.5 * (Vraw + Vraw.T); ev = np.linalg.eigvalsh(V)
    info = dict(qstar=opnorm(PA @ B), Lnorm=opnorm(Jinv),
                smin=float(np.linalg.svd(Jm, compute_uv=False).min()),
                asym=fnorm(Vraw - Vraw.T) / max(fnorm(Vraw), 1e-300), Vnorm=opnorm(V),
                V_eig_min=float(ev[0]), V_eig_max=float(ev[-1]))
    return V, info


def hessian_and_V(parent, kt, G, bg):
    B, bM = hessian_B(parent, kt, G, bg.s); B = 0.5 * (B + B.T)
    PA = response_PA(parent, bg)[0]; V, vinfo = build_V(B, PA)
    return B, bM, PA, V, vinfo


# ======================================================================= quadratic Hamiltonian
class Quad:
    """E^[2] about the background bg (RQ7 at p = 2 / SB12) in vertex form, Wick-ordered against bg.R."""

    def __init__(self, parent, bg, V):
        self.par = parent; self.A = parent.A; self.V = V; self.nv = parent.nv
        self.P = bg.R; self.D = bg.D; self.h = bg.h; self.E0 = bg.E
        self.I = np.eye(parent.d)
        self.VA = [sum(V[mu, nu] * parent.A[nu] for nu in range(self.nv)) for mu in range(self.nv)]
        self.s = parent.src(self.D)
        self.B21 = self.h - self.K(self.D)                                             # linear contact l_sp
        self.B20 = self.E0 - trr(self.h @ self.D) + 0.5 * float(self.s @ V @ self.s)   # scalar e_sc
        self.p = parent.src(self.P); self.G = self.Xc(self.I)
        self.Y = self.B21 - self.K(self.P) - 0.5 * self.G + self.Xc(self.P)             # raw one-body
        self.Es = self.B20 - trr(self.P @ self.B21) + 0.5 * (float(self.p @ V @ self.p) - trr(self.P @ self.Xc(self.P)))

    def K(self, X):
        c = self.V @ self.par.src(X); return sum(c[k] * self.A[k] for k in range(self.nv))

    def Xc(self, X):
        return sum(self.A[k] @ X @ self.VA[k] for k in range(self.nv))

    def E_poly(self, C):
        Dl = C - self.P; s = self.par.src(Dl)
        return self.B20 + trr(self.B21 @ Dl) + 0.5 * float(s @ self.V @ s)

    def E_pair(self, C):
        s = self.par.src(C - self.P); return 0.5 * float(s @ self.V @ s)

    def E_exch(self, C):
        Dl = C - self.P
        return -0.5 * sum(trr(Dl @ self.A[k] @ Dl @ self.VA[k]) for k in range(self.nv))

    def E_HF(self, C): return self.E_poly(C) + self.E_exch(C)
    def h_H(self, C): return self.B21 + self.K(C - self.P)

    def h_HF(self, C):
        Dl = C - self.P; return self.B21 + self.K(Dl) - self.Xc(Dl)

    def E_raw_gauss(self, C):
        sC = self.par.src(C)
        ex = sum(trr(self.A[k] @ (self.I - C) @ self.VA[k] @ C) for k in range(self.nv))
        return self.Es + trr(self.Y @ C) + 0.5 * float(sC @ self.V @ sC) + 0.5 * ex


# ======================================================================= self-consistent fields
def scf_projector(hfun, N, C0, tol=1e-12, maxit=600, alpha=0.5):
    """Plain aufbau SCF (lowest-N projector), damped. Used as a cross-check only."""
    C = C0.copy(); hist = []; res = np.inf
    for it in range(maxit):
        h = hfun(C); eps, U = np.linalg.eigh(h)
        Cn = U[:, :N] @ U[:, :N].conj().T
        res = fnorm(Cn - C); hist.append(res)
        if res < tol:
            C = Cn; break
        if it > 60 and res > 0.9 * min(hist[-60:]): alpha = max(alpha * 0.5, 0.02)
        C = (1.0 - alpha) * C + alpha * Cn
    h = hfun(C); eps, U = np.linalg.eigh(h)
    gap = float(eps[N] - eps[N - 1]) if N < len(eps) else float('nan')
    return C, h, eps, dict(iterations=it + 1, residual=float(res), gap=gap,
                           comm=fnorm(h @ C - C @ h), converged=bool(res < tol))


def scf_blocked(parent, hfun, pat, C0, tol=1e-12, maxit=600, alpha=0.7):
    """Symmetry-blocked SCF: lowest-n_b projector of h in every block (S, Sigma), prescribed n_b = pat[b]."""
    C = C0.copy(); hist = []; res = np.inf; gmin = np.inf
    for it in range(maxit):
        h = hfun(C); Cn = np.zeros_like(C); gmin = np.inf
        for b, u in enumerate(parent.Ub):
            nb = pat[b]
            eb, Vb = np.linalg.eigh(herm(u.conj().T @ h @ u))
            if nb > 0:
                occ = u @ Vb[:, :nb]; Cn += occ @ occ.conj().T
            if 0 < nb < len(eb): gmin = min(gmin, float(eb[nb] - eb[nb - 1]))
        res = fnorm(Cn - C); hist.append(res)
        if res < tol:
            C = Cn; break
        if it > 40 and res > 0.9 * min(hist[-40:]): alpha = max(alpha * 0.5, 0.05)
        C = (1.0 - alpha) * C + alpha * Cn
    h = hfun(C)
    eps, U = np.linalg.eigh(h); occw = np.real(np.einsum('ia,ij,ja->a', U.conj(), C, U)); occ = occw > 0.5
    aufbau_gap = float(eps[~occ].min() - eps[occ].max()) if occ.any() and (~occ).any() else float('nan')
    return C, h, dict(iterations=it + 1, residual=float(res), converged=bool(res < tol), block_gap=float(gmin),
                      aufbau_gap=aufbau_gap, comm=fnorm(h @ C - C @ h))


def hartree_blocked(parent, kt, G, pat, C0, ref):
    st = {'P': parent.P0, 'bg': None}

    def hfun(C):
        bg = solve_source(parent, kt, G, C, st['P'], ref); st['P'] = bg.R; st['bg'] = bg; return bg.h

    C, h, info = scf_blocked(parent, hfun, pat, C0)
    return C, st['bg'], info


def hartree_full(parent, kt, G, N, C0, ref):
    st = {'P': parent.P0, 'bg': None}

    def hfun(C):
        bg = solve_source(parent, kt, G, C, st['P'], ref); st['P'] = bg.R; st['bg'] = bg; return bg.h

    C, h, eps, info = scf_projector(hfun, N, C0)
    return C, st['bg'], info


# ======================================================================= fixed-number sectors
class Sector:
    """Bit-string basis of the N-particle sector of d modes with Jordan-Wigner signs; optional grouping of the
    basis by conserved block numbers (block_of_mode gives the block of each mode)."""

    def __init__(self, d, N, block_of_mode=None, nblk=4):
        self.d, self.N = d, N
        codes = [sum(1 << i for i in comb) for comb in combinations(range(d), N)]
        self.basis = np.array(sorted(codes), dtype=np.uint64); self.dim = len(self.basis)
        b = self.basis; self.pairs = {}
        for i in range(d):
            for j in range(d):
                if i == j:
                    idx = np.nonzero(((b >> np.uint64(i)) & np.uint64(1)) == 1)[0]
                    self.pairs[(i, i)] = (idx, idx, np.ones(len(idx)))
                else:
                    occ_j = ((b >> np.uint64(j)) & np.uint64(1)) == 1
                    emp_i = ((b >> np.uint64(i)) & np.uint64(1)) == 0
                    sel = np.nonzero(occ_j & emp_i)[0]; bs = b[sel]
                    maskj = np.uint64((1 << j) - 1); maski = np.uint64((1 << i) - 1)
                    s1 = 1 - 2 * (popcount64(bs & maskj) % 2)
                    b1 = bs & ~np.uint64(1 << j)
                    s2 = 1 - 2 * (popcount64(b1 & maski) % 2)
                    b2 = b1 | np.uint64(1 << i)
                    rows = np.searchsorted(b, b2)
                    self.pairs[(i, j)] = (rows, sel, (s1 * s2).astype(float))
        self.groups = None
        if block_of_mode is not None:
            masks = [np.uint64(sum(1 << int(i) for i in np.where(block_of_mode == bl)[0])) for bl in range(nblk)]
            counts = np.stack([popcount64(self.basis & mk) for mk in masks], axis=1)
            self.groups = {}
            for row in np.unique(counts, axis=0):
                self.groups[tuple(int(x) for x in row)] = np.where(np.all(counts == row, axis=1))[0]

    def one_body(self, A, dense=False):
        rows = []; cols = []; data = []
        for (i, j), (r, c, s) in self.pairs.items():
            a = A[i, j]
            if abs(a) > 1e-15 and len(r):
                rows.append(r); cols.append(c); data.append(a * s)
        if rows:
            M = sp.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(self.dim, self.dim), dtype=complex).tocsr()
        else:
            M = sp.csr_matrix((self.dim, self.dim), dtype=complex)
        return M.toarray() if dense else M

    def expect_C(self, psi):
        """C_{AB} = <c_B^dag c_A>: C[j,i] = <c_i^dag c_j>."""
        C = np.zeros((self.d, self.d), complex)
        for (i, j), (r, c, s) in self.pairs.items():
            C[j, i] = np.sum(np.conj(psi[r]) * s * psi[c])
        return herm(C)

    def indices_of(self, codes):
        idx = np.searchsorted(self.basis, np.asarray(codes, dtype=np.uint64))
        assert np.all(self.basis[idx] == np.asarray(codes, dtype=np.uint64))
        return idx


def build_sector_H(sector, Es, Y, A_list, V, dense):
    lam, Uv = np.linalg.eigh(0.5 * (V + V.T)); dim = sector.dim
    lmax = max(float(np.max(np.abs(lam))), 1e-300)
    if dense:
        H = Es * np.eye(dim, dtype=complex) + sector.one_body(Y, dense=True)
        for k in range(len(lam)):
            if abs(lam[k]) < 1e-15 * lmax: continue
            Ak = sum(Uv[mu, k] * A_list[mu] for mu in range(len(A_list)))
            Om = sector.one_body(Ak, dense=True); H += (0.5 * lam[k]) * (Om @ Om)
        return herm(H)
    H = Es * sp.identity(dim, dtype=complex, format='csr') + sector.one_body(Y)
    for k in range(len(lam)):
        if abs(lam[k]) < 1e-15 * lmax: continue
        Ak = sum(Uv[mu, k] * A_list[mu] for mu in range(len(A_list)))
        Om = sector.one_body(Ak); H = (H + (0.5 * lam[k]) * (Om @ Om)).tocsr()
    return H


def exact_by_blocks(H, groups, dense):
    """Lowest 2 eigenvalues of H in every conserved block (dense eigvalsh per block)."""
    out = {}
    for pat, idx in groups.items():
        Hs = H[np.ix_(idx, idx)] if dense else H[idx][:, idx].toarray()
        w = np.linalg.eigvalsh(herm(Hs))
        out[pat] = (float(w[0]), float(w[1]) if len(w) > 1 else None, int(len(idx)))
    return out


def eigsolve(H, dense, k=6, gpu_matvec=False):
    if dense:
        w, Vv = np.linalg.eigh(H); return w, Vv[:, :k]
    if gpu_matvec:
        try:
            import torch
            if torch.cuda.is_available():
                dev = torch.device('cuda'); Hc = H.tocsr()
                Ht = torch.sparse_csr_tensor(torch.from_numpy(Hc.indptr.astype(np.int64)),
                                             torch.from_numpy(Hc.indices.astype(np.int64)),
                                             torch.from_numpy(Hc.data.astype(np.complex128)),
                                             size=Hc.shape, dtype=torch.complex128).to(dev)

                def mv(x):
                    xt = torch.from_numpy(np.ascontiguousarray(x.astype(np.complex128))).to(dev)
                    return (Ht @ xt.reshape(-1, 1)).reshape(-1).cpu().numpy()

                op = spla.LinearOperator(H.shape, matvec=mv, dtype=np.complex128)
                w, Vv = spla.eigsh(op, k=k, which='SA', tol=1e-12, maxiter=50000)
                o = np.argsort(w); return w[o], Vv[:, o]
        except Exception as ex:
            log('  GPU matvec unavailable (%s); CPU Lanczos' % ex)
    w, Vv = spla.eigsh(H, k=k, which='SA', tol=1e-12, maxiter=50000)
    o = np.argsort(w); return w[o], Vv[:, o]


def gpu_dense_ground(H_csr):
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available(): return None
    dim = H_csr.shape[0]; need = 3 * dim * dim * 16
    free, total = torch.cuda.mem_get_info()
    if free < need: return None
    dev = torch.device('cuda'); Ht = torch.zeros((dim, dim), dtype=torch.complex128, device=dev)
    step = 2048
    for i0 in range(0, dim, step):
        blk = H_csr[i0:i0 + step].toarray(); Ht[i0:i0 + blk.shape[0]] = torch.from_numpy(blk).to(dev)
    w, Vv = torch.linalg.eigh(Ht)
    return float(w[0].item()), Vv[:, 0].cpu().numpy()


def two_hole_T8(quad, parent, V):
    """T8.12 in the site basis: H^(2h) = E_I - (h_I^T x I + I x h_I^T)|_wedge + V_wedge^T, pair coefficient 1."""
    d = parent.d; I = np.eye(d)
    E_I = quad.E_HF(I); h_I = quad.h_HF(I)
    H2 = -(np.kron(h_I.T, I) + np.kron(I, h_I.T))
    for mu in range(parent.nv):
        for nu in range(parent.nv):
            if V[mu, nu] != 0.0: H2 += V[mu, nu] * np.kron(parent.A[mu].T, parent.A[nu].T)
    pairs = list(combinations(range(d), 2)); Wm = np.zeros((d * d, len(pairs)))
    for k, (i, j) in enumerate(pairs):
        Wm[i * d + j, k] = 1.0 / math.sqrt(2.0); Wm[j * d + i, k] = -1.0 / math.sqrt(2.0)
    return herm(E_I * np.eye(len(pairs)) + Wm.T @ H2 @ Wm)


def det_overlap(sector, Uocc, psi):
    """|<Slater(Uocc)|psi>|^2 with Uocc (d x N) in the basis in which the sector is built."""
    d, N = sector.d, sector.N
    bits = ((sector.basis[:, None] >> np.arange(d, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(bool)
    modes = np.argsort(~bits, axis=1, kind='stable')[:, :N]
    amps = np.linalg.det(Uocc[modes, :])
    return float(abs(np.vdot(amps, psi)) ** 2), float(np.sum(np.abs(amps) ** 2))


def orbitals_of_projector(C, N):
    w, U = np.linalg.eigh(C); return U[:, -N:]


def var_gauss(C, A):
    return trr(C @ A @ (np.eye(len(C)) - C) @ A)


def var_exact(sector, A, psi):
    O = sector.one_body(A); Op = O @ psi; mean = float(np.vdot(psi, Op).real)
    return float(np.vdot(Op, Op).real - mean ** 2), mean


# ======================================================================= one case
def run_case(parent, kt, G, holes, ref, sectors, args):
    L = parent.L; d = parent.d; N = d - holes; n = N - parent.N0
    tag = 'L%d_ell%g_N%d_%s_G%.6g' % (L, kt.ell, N, ref, G)
    log('\n=== case %s (dim %d) ===' % (tag, math.comb(d, holes)))
    out = dict(tag=tag, parent=L, ell=kt.ell, R=kt.R, Rs=kt.Rs, N=N, n=n, holes=holes, ref=ref, G=G)
    pats_nat, shell = natural_patterns(parent, N)
    out['shell'] = shell; out['natural_patterns'] = [list(p) for p in pats_nat]
    # ---- Hartree (full nonlinear SB10), one block-symmetric solution per natural pattern
    hart = {}
    for pat in pats_nat:
        Cp, bgp, infop = hartree_blocked(parent, kt, G, pat, free_state(parent, pat), ref)
        hart[pat] = (Cp, bgp, infop)
        log('  Hartree pattern %s: E_H=%.12f  E_free=%.12f  it %d res %.1e  block gap %.3e  aufbau gap %.3e'
            % (pat, bgp.E, parent.E_free_pattern(pat), infop['iterations'], infop['residual'], infop['block_gap'], infop['aufbau_gap']))
    pat_H = min(hart, key=lambda p: hart[p][1].E)
    C_H, bgH, infoH = hart[pat_H]
    E_free = parent.E_free_pattern(pat_H); dE = bgH.E - E_free
    out['pattern_H'] = list(pat_H); out['E_free'] = E_free; out['E_H'] = bgH.E; out['E_m_H'] = bgH.Em; out['dE_grav'] = dE
    out['hartree'] = infoH
    out['hartree_patterns'] = {str(list(p)): dict(E=hart[p][1].E, E_free=parent.E_free_pattern(p), **hart[p][2]) for p in hart}
    if not args.no_aufbau:
        Ca, bga, infoa = hartree_full(parent, kt, G, N, parent.U0[:, :N] @ parent.U0[:, :N].conj().T, ref)
        out['hartree_aufbau'] = dict(E=bga.E, block_counts=parent.block_counts(Ca), dC_to_H=fnorm(Ca - C_H), **infoa)
        log('  Hartree aufbau cross-check: E=%.12f  converged=%s  block counts %s  ||C_a-C_H||=%.2e'
            % (bga.E, infoa['converged'], np.round(parent.block_counts(Ca), 3), fnorm(Ca - C_H)))
    out['margins'] = dict(mass_margin=bgH.mass_margin, rho_center=bgH.rho_center, sheet_gap=bgH.gap,
                          min_abs_eps=bgH.min_abs_eps, sheet_is_lowest=bgH.sheet_is_lowest,
                          compactness=bgH.compactness, N_center=bgH.N_center, wmax=bgH.wmax,
                          inner_iterations=bgH.inner_iterations, inner_residual=bgH.inner_residual,
                          riesz_vs_eigh=riesz_check(bgH.h, bgH.eps, bgH.occ))
    out['dP0'] = dict(op=bgH.dP0_op, fro=bgH.dP0_fro)
    log('  Hartree ground: pattern %s  E_H=%.12f  E_free=%.12f  dE_grav=%.3e  compactness=%.3e  N(0)=%.6f  |w|max=%.4f'
        % (pat_H, bgH.E, E_free, dE, bgH.compactness, bgH.N_center, bgH.wmax))
    log('  margins: mass %.4e  sheet gap %.4f  min|eps| %.4f  ||R-P0||_op %.3e  inner it %d res %.1e  riesz %.1e'
        % (bgH.mass_margin, bgH.gap, bgH.min_abs_eps, bgH.dP0_op, bgH.inner_iterations, bgH.inner_residual,
           out['margins']['riesz_vs_eigh']))
    # ---- Hessian and response
    B, bM = hessian_B(parent, kt, G, bgH.s)
    Bl = hessian_lin(parent, kt, G, bgH.sol); Bfd = hessian_fd(parent, kt, G, bgH.s)
    nB = max(fnorm(B), 1e-300)
    out['hessian_checks'] = dict(asym=fnorm(B - B.T) / nB, cs_vs_lin=fnorm(B - Bl) / nB, cs_vs_fd=fnorm(B - Bfd) / nB,
                                 variational=fnorm(bM[-1] - bgH.Phi) / fnorm(bgH.Phi), Bnorm=opnorm(B))
    B = 0.5 * (B + B.T)
    PA = response_PA(parent, bgH)[0] if ref == 'moving' else np.zeros_like(B)
    V, vinfo = build_V(B, PA); out['V_info'] = vinfo
    Atil = bM @ np.linalg.inv(np.eye(2 * L) + B @ PA).T
    out['A_R_is_hbar'] = fnorm(parent.h_of(Atil[-1]) - bgH.h) / fnorm(bgH.h)
    log('  Hessian: asym %.1e  cs-lin %.1e  cs-fd %.1e  variational %.1e  q*=%.3e  ||L||=%.4f  A_R=hbar %.1e  eig(V) [%.3e, %.3e]'
        % (out['hessian_checks']['asym'], out['hessian_checks']['cs_vs_lin'], out['hessian_checks']['cs_vs_fd'],
           out['hessian_checks']['variational'], vinfo['qstar'], vinfo['Lnorm'], out['A_R_is_hbar'], vinfo['V_eig_min'], vinfo['V_eig_max']))
    quad = Quad(parent, bgH, V)
    # ---- Wick identities at this background
    rng = np.random.default_rng(1)
    Wt = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)); Ct = herm(Wt) / d + C_H
    ymin = []
    for _ in range(5):
        u = (np.eye(d) - quad.P) @ (rng.normal(size=d) + 1j * rng.normal(size=d)); u /= np.linalg.norm(u)
        Cu = quad.P + np.outer(u, u.conj()); ymin.append(quad.E_exch(Cu) / quad.E_pair(Cu))
    X = herm(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))); eta = 1e-5
    grad_fd = (quad.E_HF(Ct + eta * X) - quad.E_HF(Ct - eta * X)) / (2 * eta)
    out['wick_checks'] = dict(raw_vs_HF=abs(quad.E_raw_gauss(Ct) - quad.E_HF(Ct)) / max(abs(quad.E_HF(Ct)), 1e-300),
                              y_minus_one=float(np.max(np.abs(np.array(ymin) + 1.0))),
                              pair_at_P=abs(quad.E_pair(quad.P)) + abs(quad.E_exch(quad.P)),
                              first_variation_at_P=fnorm(quad.h_HF(quad.P) - quad.B21),
                              hHF_gradient=abs(grad_fd - trr(quad.h_HF(Ct) @ X)) / max(abs(grad_fd), 1e-300),
                              E_poly_at_H_minus_E_H=quad.E_poly(C_H) - bgH.E)
    out['scalars'] = dict(B20=quad.B20, Es=quad.Es, TrPB21=trr(quad.P @ quad.B21))
    E_pairH = quad.E_pair(C_H); E_exchH = quad.E_exch(C_H)
    out['pair_H'] = dict(E_pair_H=E_pairH, E_exch_H=E_exchH, fock_over_pair=(E_exchH / E_pairH if E_pairH != 0 else None),
                         inv_Nf=1.0 / n, V_eig_min=vinfo['V_eig_min'], V_eig_max=vinfo['V_eig_max'])
    log('  pair at C_H: E_pair=%.4e  E_exch=%.4e  |E_exch/E_pair|=%s  1/N_f=%.4f  (kernel indefinite: %s)'
        % (E_pairH, E_exchH, 'n/a' if E_pairH == 0 else '%.3e' % abs(E_exchH / E_pairH), 1.0 / n, vinfo['V_eig_min'] < 0 < vinfo['V_eig_max']))
    # ---- exact: block-resolved sector in the symmetry-adapted basis, plus the full-sector route
    sector = sectors[(d, N)]; dense = sector.dim <= args.dense_max
    t0 = time.time()
    H = build_sector_H(sector, quad.Es, parent.to_blk(quad.Y), parent.A_blk, V, dense)
    blk = exact_by_blocks(H, sector.groups, dense)
    pat_Q = min(blk, key=lambda p: blk[p][0]); E_Q = blk[pat_Q][0]
    idx = sector.groups[pat_Q]
    Hs = H[np.ix_(idx, idx)] if dense else H[idx][:, idx].toarray()
    wq, Vq = np.linalg.eigh(herm(Hs)); psi = np.zeros(sector.dim, complex); psi[idx] = Vq[:, 0]
    low = sorted(((v[0], p) for p, v in blk.items()))[:6]
    out['exact_blocks_lowest'] = [dict(E=e, pattern=list(p), dim=blk[p][2]) for e, p in low]
    out['pattern_Q'] = list(pat_Q); out['E_Q'] = E_Q; out['E_Q_block_second'] = blk[pat_Q][1]
    out['E_Q_degenerate_partner'] = bool(len(low) > 1 and abs(low[1][0] - low[0][0]) < 1e-9)
    wfull, _ = eigsolve(H, dense, k=min(6, sector.dim - 1), gpu_matvec=args.gpu_matvec)
    out['E_Q_fullsector'] = float(wfull[0]); out['blocks_vs_full'] = abs(E_Q - wfull[0]) / max(abs(E_Q), 1e-300)
    resid = fnorm(H @ psi - E_Q * psi); out['exact_residual'] = resid
    out['exact_build_seconds'] = time.time() - t0
    log('  exact: dim %d  %s  E_Q=%.12f (pattern %s, block dim %d)  full-sector %.12f  resid %.1e  (%.1f s)'
        % (sector.dim, 'dense' if dense else 'Lanczos', E_Q, pat_Q, blk[pat_Q][2], wfull[0], resid, time.time() - t0))
    log('  lowest block energies: ' + '  '.join('%s %.10f' % (p, e) for e, p in low))
    if not dense and args.gpu_dense:
        g = gpu_dense_ground(H)
        out['gpu_dense_E_Q'] = None if g is None else g[0]
        if g is not None: log('  GPU dense cross-check: E_Q=%.12f  diff %.2e' % (g[0], g[0] - E_Q))
    if holes == 2:
        HA = two_hole_T8(quad, parent, V); wA = np.linalg.eigvalsh(HA)
        wB = np.linalg.eigvalsh(H) if dense else None
        spread = max(float(wA.max() - wA.min()), 1e-300)
        out['routeA_vs_B'] = float(np.max(np.abs(wA - wB)) / spread) if wB is not None else None
        out['routeA_E_Q'] = float(wA[0])
        log('  route A (T8.12, site basis) vs route B (adapted basis): max relative spectrum difference %.2e' % out['routeA_vs_B'])
    C_Q = parent.from_blk(sector.expect_C(psi))
    occQ, UQ = np.linalg.eigh(C_Q)
    out['block_counts_Q'] = parent.block_counts(C_Q)
    # ---- Hartree-Fock (quadratic polynomial), blocked, in the natural patterns and in the exact ground pattern
    hf = {}
    for pat in list(dict.fromkeys(list(pats_nat) + [pat_Q])):
        start = hart[pat][0] if pat in hart else natural_start(parent, C_Q, pat)
        Cp, hp, infop = scf_blocked(parent, quad.h_HF, pat, start)
        hf[pat] = (quad.E_HF(Cp), Cp, infop)
        if pat == pat_Q and pat in hart:
            Cq, hq, infoq = scf_blocked(parent, quad.h_HF, pat, natural_start(parent, C_Q, pat))
            if quad.E_HF(Cq) < hf[pat][0] - 1e-12: hf[pat] = (quad.E_HF(Cq), Cq, infoq)
    pat_F = min(hf, key=lambda p: hf[p][0]); E_HF, C_F, infoF = hf[pat_F]
    out['HF_patterns'] = {str(list(p)): dict(E=hf[p][0], **hf[p][2]) for p in hf}; out['pattern_F'] = list(pat_F)
    if pat_Q not in hart:
        Cx, bgx, infox = hartree_blocked(parent, kt, G, pat_Q, free_state(parent, pat_Q), ref)
        out['hartree_in_pattern_Q'] = dict(E=bgx.E, E_free=parent.E_free_pattern(pat_Q), **infox)
        log('  NOTE: exact ground pattern %s differs from the Hartree ground pattern %s; Hartree there: E=%.12f' % (pat_Q, pat_H, bgx.E))
    if not args.no_aufbau:
        Cu, hu, epsu, infou = scf_projector(quad.h_HF, N, C_H); out['HF_aufbau'] = dict(E=quad.E_HF(Cu), **infou)
    E_HF_at_H = quad.E_HF(C_H); E_F = quad.E_exch(C_F)
    out.update(E_HF=E_HF, E_HF_at_CH=E_HF_at_H, E_F=E_F, E_pair_HF=quad.E_pair(C_F), E_Q_minus_B20=E_Q - quad.B20, E_Q_minus_Es=E_Q - quad.Es)
    out['within_pattern_H'] = dict(E_Q=blk[pat_H][0], E_HF=hf[pat_H][0], HF_over_H=hf[pat_H][0] / bgH.E, Q_over_HF=blk[pat_H][0] / hf[pat_H][0])
    out['variational'] = dict(Q_le_HF=bool(E_Q <= E_HF + 1e-10), Q_le_HF_at_CH=bool(E_Q <= E_HF_at_H + 1e-10), gap_Q_HF=E_HF - E_Q)
    E_HF_at_Q = quad.E_HF(C_Q); out['E_HF_at_CQ'] = E_HF_at_Q; out['E_conn_at_CQ'] = E_Q - E_HF_at_Q
    out['natural_occupations_Q'] = [float(x) for x in occQ[::-1]]
    out['fractional_natural_occupations_Q'] = int(np.sum((occQ > 0.05) & (occQ < 0.95)))
    ovl, nrm = det_overlap(sector, parent.U_blk.conj().T @ orbitals_of_projector(C_F, N), psi)
    out['overlap_HF_exact'] = ovl; out['det_norm_check'] = nrm
    # ---- Gauss readouts, source-Taylor errors, reference re-solve
    bgF = solve_source(parent, kt, G, C_F, bgH.R, ref); bgQ = solve_source(parent, kt, G, C_Q, bgH.R, ref)
    out['E_gauss_HF'] = bgF.E; out['E_gauss_Q'] = bgQ.E
    out['source_taylor'] = dict(rh_F=fnorm(bgF.h - quad.h_H(C_F)), rh_Q=fnorm(bgQ.h - quad.h_H(C_Q)),
                                dE_F=bgF.E - quad.E_poly(C_F), dE_Q=bgQ.E - quad.E_poly(C_Q),
                                dC_F=fnorm(C_F - C_H), dC_Q=fnorm(C_Q - C_H))
    out['reference_resolve'] = dict(dR_F=fnorm(bgF.R - bgH.R), dR_Q=fnorm(bgQ.R - bgH.R), dR_bar_P0=bgH.dP0_fro)
    # ---- ratios
    out['ratios'] = dict(HF_over_H=E_HF / bgH.E, Q_over_HF=E_Q / E_HF, Q_over_H=E_Q / bgH.E,
                         collar_HF_over_H=bgF.E / bgH.E, collar_Q_over_HF=bgQ.E / bgF.E,
                         y_HF=(E_HF - bgH.E) / dE if dE != 0 else None, y_F=E_F / dE if dE != 0 else None,
                         y_corr=(E_Q - E_HF) / dE if dE != 0 else None, inv_Nf=1.0 / n)
    log('  E_HF=%.12f (pattern %s)  E_Q=%.12f  E_F=%.3e  E_conn(C_Q)=%.3e  B20=%.6f  Es=%.6f  E_Q-B20=%.6f'
        % (E_HF, pat_F, E_Q, E_F, out['E_conn_at_CQ'], quad.B20, quad.Es, E_Q - quad.B20))
    log('  ratios: HF/H=%.12f  Q/HF=%.12f  collar HF/H=%.12f  Q/HF=%.12f  y_HF=%s y_F=%s y_corr=%s  1/N_f=%.4f  variational %s'
        % (out['ratios']['HF_over_H'], out['ratios']['Q_over_HF'], out['ratios']['collar_HF_over_H'],
           out['ratios']['collar_Q_over_HF'], out['ratios']['y_HF'], out['ratios']['y_F'], out['ratios']['y_corr'], 1.0 / n,
           out['variational']))
    log('  exact state: |<HF|Q>|^2=%.6f  fractional natural occupations=%d  HF block gap=%.3e  ||R(C_Q)-Rbar||=%.2e  ||R(C_F)-Rbar||=%.2e'
        % (ovl, out['fractional_natural_occupations_Q'], infoF['block_gap'], out['reference_resolve']['dR_Q'], out['reference_resolve']['dR_F']))
    # ---- variances of the linearized mass observable
    rows = []
    for rr in [1.0, 2.0, 3.0, kt.Rs, kt.R]:
        j = min(int(round(rr / kt.dr)), kt.ns); Ar = parent.h_of(Atil[j]); Ar_b = parent.to_blk(Ar)
        Nr_mat = sum(kt.nx[j, x] * parent.Pi[x] for x in range(L)); Nr_b = parent.to_blk(Nr_mat)
        Mbar = float(bgH.sol['M'][j].real)
        sea = var_gauss(bgH.R, Ar); nsea = var_gauss(bgH.R, Nr_mat)
        row = dict(r=float(kt.r[j]), M_H=Mbar, sea=sea, num_sea=nsea, states={})
        for name, C, bgc in [('H', C_H, bgH), ('HF', C_F, bgF), ('Q', C_Q, bgQ)]:
            m = Mbar + trr((C - C_H) @ Ar); vg = var_gauss(C, Ar); Nr = trr((C - bgH.R) @ Nr_mat)
            nv_ = var_gauss(C, Nr_mat)
            st = dict(mean_linear=m, mean_resolved=float(bgc.sol['M'][j].real), var_gauss=vg, exc=vg - sea, N_r=Nr,
                      num_var=nv_, num_exc=nv_ - nsea)
            if name == 'Q':
                vx, _ = var_exact(sector, Ar_b, psi); nvx, _ = var_exact(sector, Nr_b, psi)
                st['var_total'] = vx; st['conn'] = vx - vg; st['exc_total'] = vx - sea
                st['num_var_total'] = nvx; st['num_exc_total'] = nvx - nsea
            else:
                st['var_total'] = vg; st['conn'] = 0.0; st['exc_total'] = vg - sea
                st['num_var_total'] = nv_; st['num_exc_total'] = nv_ - nsea
            exc = st['exc_total']
            st['rel_total'] = math.sqrt(max(st['var_total'], 0.0)) / m if m > 0 else None
            st['rel_exc'] = math.copysign(math.sqrt(abs(exc)), exc) / m if m > 0 else None
            st['bound_Nr'] = Nr ** -0.5 if Nr > 0 else None
            st['num_exc_over_Nr'] = st['num_exc_total'] / Nr if Nr > 0 else None
            row['states'][name] = st
        rows.append(row)
        if rr in (3.0, kt.Rs):
            sQ = row['states']['Q']; sH = row['states']['H']; sF = row['states']['HF']
            log('  r=%.2f M=%.5f N_r=%.4f  Var: H %.3e  HF %.3e  Q %.3e (conn %.2e)  sea %.3e  rel_exc(Q) %s  N_r^-1/2 %s  numexc/N_r(H,Q) %s %s'
                % (row['r'], Mbar, sQ['N_r'], sH['var_total'], sF['var_total'], sQ['var_total'], sQ['conn'], sea,
                   'n/a' if sQ['rel_exc'] is None else '%.3e' % sQ['rel_exc'], 'n/a' if sQ['bound_Nr'] is None else '%.3e' % sQ['bound_Nr'],
                   'n/a' if sH['num_exc_over_Nr'] is None else '%.3f' % sH['num_exc_over_Nr'],
                   'n/a' if sQ['num_exc_over_Nr'] is None else '%.3f' % sQ['num_exc_over_Nr']))
    out['variances'] = rows
    sel = [0, int(round(1.0 / kt.dr)), int(round(2.0 / kt.dr)), int(round(3.0 / kt.dr)), int(round(kt.Rs / kt.dr)), kt.ns]
    out['profile_H'] = dict(r=[float(kt.r[j]) for j in sel], M=[float(bgH.sol['M'][j].real) for j in sel],
                            N=[float(bgH.sol['N'][j].real) for j in sel], v=[float(bgH.sol['v'][j].real) for j in sel])
    return out


# ======================================================================= T8 canonical block
def V1_direct(parent, kt, nu):
    L = parent.L; Gam = nu * parent.F0; e = parent.src(Gam)[:L]
    M0 = kt.nx @ e; r = kt.r
    b = np.zeros_like(r); b[1:] = -np.sqrt(2.0 * M0[1:] / r[1:])
    fac = np.zeros_like(r); fac[1:] = kt.W[1:] * FOURPI * r[1:] ** 2 / (r[1:] * b[1:])
    V1 = np.zeros((2 * L, 2 * L))
    V1[L:, :L] = (kt.KT1 * fac[:, None]).T @ kt.nx
    V1[:L, L:] = V1[L:, :L].T
    return V1


def rk4_step(C, dt, rhs):
    k1 = rhs(C); k2 = rhs(C + 0.5 * dt * k1); k3 = rhs(C + 0.5 * dt * k2); k4 = rhs(C + dt * k3)
    return herm(C + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4))


def canonical_block(parent, kt, G, sector_e, args, do_evolve):
    L = parent.L; d = parent.d; N0 = parent.N0; Dp = parent.Dplus; N = sector_e.N; n = N - N0
    nu = n / Dp; tau = nu * (1 - nu) / (Dp - 1)
    log('\n=== T8 canonical block: L=%d ell=%g G=%.6g  nu=%.3f tau=%.5f ===' % (L, kt.ell, G, nu, tau))
    out = dict(G=G, nu=nu, tau=tau)
    Cnu = parent.P0 + nu * parent.F0
    bg = solve_source(parent, kt, G, Cnu, parent.P0, 'moving')
    B, bM, PA, V, vinfo = hessian_and_V(parent, kt, G, bg)
    out['margins'] = dict(E=bg.E, mass_margin=bg.mass_margin, rho_center=bg.rho_center, sheet_gap=bg.gap,
                          sheet_is_lowest=bg.sheet_is_lowest, qstar=vinfo['qstar'], smin=vinfo['smin'], Lnorm=vinfo['Lnorm'],
                          compactness=bg.compactness, dP0_op=bg.dP0_op, inner_residual=bg.inner_residual,
                          riesz_vs_eigh=riesz_check(bg.h, bg.eps, bg.occ))
    log('  SB4 at C_nu: E=%.12f  mass margin %.4e  sheet gap %.4f  q*=%.3e  ||R-P0||=%.3e' % (bg.E, bg.mass_margin, bg.gap, vinfo['qstar'], bg.dP0_op))
    quad = Quad(parent, bg, V)
    U0 = parent.U0; toe = lambda X: U0.conj().T @ X @ U0
    Hd = build_sector_H(sector_e, quad.Es, toe(quad.Y), parent.A_e, V, dense=True)
    sea_mask = sum(1 << k for k in range(N0)); pos = list(range(N0, d))
    codes = [sea_mask + sum(1 << k for k in comb) for comb in combinations(pos, n)]
    idx = sector_e.indices_of(codes)
    rho_diag = np.zeros(sector_e.dim); rho_diag[idx] = 1.0 / len(idx)
    # T8.4
    F = parent.F0; E_F = quad.E_exch(Cnu); sF = parent.src(F)
    E_can = -0.5 * tau * (float(sF @ V @ sF) - sum(trr(F @ parent.A[k] @ F @ quad.VA[k]) for k in range(parent.nv)))
    E_rho = float(np.sum(rho_diag * np.diag(Hd).real))
    out['T8_4'] = dict(E_rho=E_rho, E_Cnu=bg.E, E_F=E_F, E_can=E_can, residual=E_rho - (bg.E + E_F + E_can),
                       E_F_plus_E_can_over_G=(E_F + E_can) / G)
    log('  T8.4: Tr(rho H)=%.12f  E(Cnu)+E_F+E_can=%.12f  residual %.2e   E_F=%.3e E_can=%.3e' % (E_rho, bg.E + E_F + E_can, out['T8_4']['residual'], E_F, E_can))
    # T8.6
    rho = np.diag(rho_diag).astype(complex); comm = Hd @ rho - rho @ Hd
    Cdot_e = np.zeros((d, d), complex)
    for (i, j), (r, c, s) in sector_e.pairs.items():
        Cdot_e[j, i] = -1j * np.sum(comm[c, r] * s)
    Cdot_Q = U0 @ Cdot_e @ U0.conj().T
    hHF = quad.h_HF(Cnu); Cdot_F = -1j * (hHF @ Cnu - Cnu @ hHF); Cdot_H = -1j * (bg.h @ Cnu - Cnu @ bg.h)
    BF = quad.K(F) - quad.Xc(F); Ycan = 1j * tau * (BF @ F - F @ BF)
    XD = quad.Xc(bg.D); YF = 1j * (XD @ Cnu - Cnu @ XD)
    V1d = V1_direct(parent, kt, nu)
    K1F = sum((V1d @ parent.src(F))[k] * parent.A[k] for k in range(parent.nv))
    X1F = sum(V1d[mu, nv_] * parent.A[mu] @ F @ parent.A[nv_] for mu in range(parent.nv) for nv_ in range(parent.nv))
    Blead = K1F - Dp * X1F; Ylead = 1j * math.sqrt(G) * tau * (Blead @ F - F @ Blead)
    out['T8_6'] = dict(Ycan_norm=fnorm(Ycan), Ycan_resid=fnorm(Cdot_Q - Cdot_F - Ycan) / max(fnorm(Ycan), 1e-300),
                       YF_norm=fnorm(YF), YF_resid=fnorm(Cdot_F - Cdot_H - YF) / max(fnorm(YF), 1e-300),
                       CdotH_norm=fnorm(Cdot_H), lead_resid=fnorm(Ycan + YF - Ylead) / max(fnorm(Ylead), 1e-300))
    log('  T8.6: Y_can resid %.2e (|Y_can|=%.3e)  Y_F resid %.2e (|Y_F|=%.3e)  leading-form resid %.2e' % (
        out['T8_6']['Ycan_resid'], fnorm(Ycan), out['T8_6']['YF_resid'], fnorm(YF), out['T8_6']['lead_resid']))
    # T8.8
    Atil = bM @ np.linalg.inv(np.eye(2 * L) + B @ PA).T
    t88 = []
    for rr in [kt.R, kt.Rs, 3.0, 2.0]:
        j = min(int(round(rr / kt.dr)), kt.ns); Ar = parent.h_of(Atil[j])
        O = sector_e.one_body(toe(Ar), dense=True)
        mean = float(np.sum(rho_diag * np.diag(O).real)); second = float(np.sum(rho_diag * np.sum(np.abs(O) ** 2, axis=0)))
        P = parent.P0; Bm = F @ Ar @ F
        formula = (1 - nu) * fnorm(P @ Ar @ F) ** 2 + Dp * tau * fnorm(Bm - (trr(Bm) / Dp) * F) ** 2
        t88.append(dict(r=float(kt.r[j]), var_direct=second - mean ** 2, formula=formula,
                        resid=(second - mean ** 2 - formula) / max(formula, 1e-300),
                        A_R_is_hbar=(fnorm(Ar - bg.h) / fnorm(bg.h)) if j == kt.ns else None))
    out['T8_8'] = t88
    log('  T8.8: ' + '  '.join('r=%.2f var=%.4e resid %.1e' % (x['r'], x['var_direct'], x['resid']) for x in t88))
    # T8.2 leading coefficient
    bg4 = solve_source(parent, kt, G / 4.0, Cnu, parent.P0, 'moving')
    V4 = hessian_and_V(parent, kt, G / 4.0, bg4)[3]
    g = math.sqrt(G); rich = (4.0 * V4 - V) / g
    out['T8_2'] = dict(V1_direct_norm=opnorm(V1d), V_over_g_resid=fnorm(V / g - V1d) / fnorm(V1d),
                       richardson_resid=fnorm(rich - V1d) / fnorm(V1d))
    log('  T8.2: ||V/g - V1||/||V1|| = %.3e   Richardson (4V(G/4)-V(G))/g: %.3e  (O(g) and O(g^2) remainders, no threshold)' % (out['T8_2']['V_over_g_resid'], out['T8_2']['richardson_resid']))
    # T8 Sec. 7 witnesses
    if L == 5:
        vx = bg.sol['vx'].real / g; ix = int(np.where(parent.xs == 2)[0][0]); B1 = float(vx[ix])
        V1sp = sum(vx[x] * herm(parent.Pi_sp[x] @ parent.Q1) for x in range(L))
        lam, Uf = np.linalg.eigh(parent.p1); f = Uf[:, np.argmin(np.abs(lam))]
        ic = (L - 1) // 2
        lhs = (V1sp @ f)[ic]; rhs = f[ic] * B1 / (4j)
        E0sp = Uf @ np.diag(np.sqrt(parent.m ** 2 + lam ** 2)) @ Uf.conj().T
        chi = np.zeros(4); chi[0] = 1.0
        V1 = np.kron(V1sp, np.eye(4)); psi = np.kron(f, chi)
        lhs2 = fnorm(parent.P0 @ V1 @ psi) ** 2
        rhs2 = 0.5 * float(np.vdot(V1sp @ f, (np.eye(L) - parent.m * np.linalg.inv(E0sp)) @ (V1sp @ f)).real)
        out['T8_7_witnesses'] = dict(B1=B1, V1f_center=[lhs.real, lhs.imag], formula=[rhs.real, rhs.imag],
                                     spectral_contact=lhs2, spectral_formula=rhs2)
        log('  T8 Sec.7: B1=%.4e (<0 expected)  (V1 f)(0)=%s vs %s   ||P V1 (f x chi+)||^2=%.4e vs %.4e' % (B1, lhs, rhs, lhs2, rhs2))
    # exact ground energy of E_2(C_nu), HF and quadratic Hartree about C_nu
    wd = np.linalg.eigvalsh(Hd)
    C_F, _, _, infoF = scf_projector(quad.h_HF, N, Cnu); C_H2, _, _, infoH2 = scf_projector(quad.h_H, N, Cnu)
    out['about_Cnu'] = dict(E_Q=float(wd[0]), E_Q_second=float(wd[1]), E_HF=quad.E_HF(C_F), E_H2=quad.E_poly(C_H2),
                            B20=quad.B20, Es=quad.Es, HF_gap=infoF['gap'], H2_gap=infoH2['gap'],
                            HF_converged=infoF['converged'], H2_converged=infoH2['converged'])
    log('  about C_nu: E_Q=%.10f  E_HF=%.10f  E_H2=%.10f  B20=%.6f' % (wd[0], quad.E_HF(C_F), quad.E_poly(C_H2), quad.B20))
    # short-time evolution
    if do_evolve:
        lam, Qm = np.linalg.eigh(Hd); psi0 = Qm.conj().T[:, idx]
        state = {'P': bg.R}

        def C_exact(t):
            psis = Qm @ (np.exp(-1j * lam * t)[:, None] * psi0)
            C = sum(sector_e.expect_C(psis[:, k]) for k in range(psis.shape[1])) / psis.shape[1]
            return U0 @ C @ U0.conj().T

        def rhs_F(C):
            h = quad.h_HF(C); return -1j * (h @ C - C @ h)

        def rhs_H(C):
            b = solve_source(parent, kt, G, C, state['P'], 'moving'); state['P'] = b.R
            return -1j * (b.h @ C - C @ b.h)

        tmax, nsteps = 0.5, 100; times = np.linspace(0.0, tmax, nsteps + 1)
        stF = Cnu.copy(); stH = Cnu.copy(); recs = []
        E_HF0 = quad.E_HF(Cnu); E_H0 = bg.E
        for it in range(1, nsteps + 1):
            dt = times[it] - times[it - 1]
            stF = rk4_step(stF, dt, rhs_F); stH = rk4_step(stH, dt, rhs_H)
            if it in (2, 5, 10, 20, 50, 100):
                t = times[it]; CQ = C_exact(t)
                bH = solve_source(parent, kt, G, stH, state['P'], 'moving'); bQ = solve_source(parent, kt, G, CQ, state['P'], 'moving')
                recs.append(dict(t=t, dQF=fnorm(CQ - stF), dQH=fnorm(CQ - stH),
                                 jetQF_over_t2=fnorm(CQ - stF - t * Ycan) / t ** 2,
                                 jetQH_over_t2=fnorm(CQ - stH - t * (Ycan + YF)) / t ** 2,
                                 dC_Q=fnorm(CQ - Cnu), dC_F=fnorm(stF - Cnu), dC_H=fnorm(stH - Cnu),
                                 E_HF_drift=quad.E_HF(stF) - E_HF0, E_H_gauss_drift=bH.E - E_H0,
                                 E_gauss_Q=bQ.E, E_gauss_Q_drift=bQ.E - E_H0))
                log('   t=%.3f |CQ-CF|=%.2e |CQ-CH|=%.2e jets/t^2: %.3e %.3e  E_HF drift %.1e  E_H(Gauss) drift %.1e  E(C_Q) drift %.2e'
                    % (t, recs[-1]['dQF'], recs[-1]['dQH'], recs[-1]['jetQF_over_t2'], recs[-1]['jetQH_over_t2'],
                       recs[-1]['E_HF_drift'], recs[-1]['E_H_gauss_drift'], recs[-1]['E_gauss_Q_drift']))
        out['evolution'] = recs
    return out


# ======================================================================= scans and checks
def coupling_scan(parent, kt, N, Glist):
    rows = []; pats, shell = natural_patterns(parent, N)
    for G in Glist:
        try:
            best = None
            for pat in pats:
                C, bg, info = hartree_blocked(parent, kt, G, pat, free_state(parent, pat), ref='moving')
                if best is None or bg.E < best[1].E: best = (C, bg, info, pat)
            C, bg, info, pat = best
            B, bM, PA, V, vinfo = hessian_and_V(parent, kt, G, bg)
            row = dict(G=G, ok=True, E=bg.E, pattern=list(pat), compactness=bg.compactness, mass_margin=bg.mass_margin,
                       sheet_gap=bg.gap, min_abs_eps=bg.min_abs_eps, sheet_lowest=bg.sheet_is_lowest, qstar=vinfo['qstar'],
                       smin=vinfo['smin'], N_center=bg.N_center, wmax=bg.wmax, inner_res=bg.inner_residual,
                       scf_res=info['residual'], block_gap=info['block_gap'], aufbau_gap=info['aufbau_gap'], dP0_op=bg.dP0_op)
            row['admissible'] = bool(row['mass_margin'] > 0 and row['sheet_gap'] > 0 and row['sheet_lowest'] and
                                     row['qstar'] < 1 and row['inner_res'] < 1e-10 and row['scf_res'] < 1e-9)
        except Exception as ex:
            row = dict(G=G, ok=False, admissible=False, error=str(ex))
        rows.append(row)
        log('  scan G=%.6g: %s' % (G, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}))
    return rows


def g0_check(parent, kt, holes, sectors, args):
    d = parent.d; N = d - holes; pats, shell = natural_patterns(parent, N); pat = pats[0]
    C_H, bg, info = hartree_blocked(parent, kt, 0.0, pat, free_state(parent, pat), 'moving')
    B, bM = hessian_B(parent, kt, 0.0, bg.s); V = np.zeros_like(B); quad = Quad(parent, bg, V)
    C_F, _, _ = scf_blocked(parent, quad.h_HF, pat, C_H); E_HF = quad.E_HF(C_F)
    sector = sectors[(d, N)]
    H = build_sector_H(sector, quad.Es, parent.to_blk(quad.Y), parent.A_blk, V, dense=True)
    blk = exact_by_blocks(H, sector.groups, True); E_Q = min(v[0] for v in blk.values())
    E_free = parent.E_free_pattern(pat)
    return dict(E_H=bg.E, E_HF=E_HF, E_Q=E_Q, E_free=E_free, pattern=list(pat),
                max_dev=max(abs(bg.E - E_free), abs(E_HF - E_free), abs(E_Q - E_free)))


def refinement_check(parent, ell, R, dr, nq, G, N, ref):
    """Discretization error of the Hartree background and of the discrete variational identity at dr and dr/2."""
    pats, shell = natural_patterns(parent, N); pat = pats[0]; res = {}
    for step in (dr, dr / 2.0):
        kt = KernelTables(parent, ell, R, step, nq)
        C, bg, info = hartree_blocked(parent, kt, G, pat, free_state(parent, pat), ref)
        B, bM = hessian_B(parent, kt, G, bg.s)
        res[step] = dict(E=bg.E, Phi=bg.Phi, B=B, var=fnorm(bM[-1] - bg.Phi) / fnorm(bg.Phi), asym=fnorm(B - B.T) / fnorm(B))
    a, b = res[dr], res[dr / 2.0]
    return dict(dE=abs(a['E'] - b['E']), dPhi=fnorm(a['Phi'] - b['Phi']), dB_rel=fnorm(a['B'] - b['B']) / max(fnorm(b['B']), 1e-300),
                variational_dr=a['var'], variational_dr2=b['var'], asym_dr=a['asym'], asym_dr2=b['asym'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--couplings', default='0.0009765625,0.00390625,0.015625', help='G/a^2 values (DECLARED defaults 2^-10,2^-8,2^-6)')
    ap.add_argument('--nq', type=int, default=24)
    ap.add_argument('--dr', type=float, default=0.01)
    ap.add_argument('--dense-max', type=int, default=5000)
    ap.add_argument('--out', default='task8_results.json')
    ap.add_argument('--no-evolve', action='store_true')
    ap.add_argument('--no-aufbau', action='store_true', help='skip the plain aufbau SCF cross-checks')
    ap.add_argument('--skip-large', action='store_true', help='skip the 20475-dimensional sector')
    ap.add_argument('--fixed-large', action='store_true', help='also run the fixed reference on the 20475-dimensional sector')
    ap.add_argument('--ell-full', action='store_true', help='run the 20475-dimensional sector at ell = 2a and 8a too')
    ap.add_argument('--gpu-dense', action='store_true', help='dense torch eigh on GPU as a cross-check when memory allows')
    ap.add_argument('--gpu-matvec', action='store_true', help='Lanczos with the sparse matvec on the GPU (torch)')
    ap.add_argument('--no-scan', action='store_true')
    ap.add_argument('--threads', type=int, default=None, help='thread cap (applied through the BLAS environment variables)')
    args = ap.parse_args()
    couplings = [float(x) for x in args.couplings.split(',')]
    results = dict(settings=dict(couplings=couplings, nq=args.nq, dr=args.dr, ma=1.0), cases=[], T8=[], checks={})
    sectors = {}

    def get_sector(parent, N):
        key = (parent.d, N)
        if key not in sectors:
            t0 = time.time(); sectors[key] = Sector(parent.d, N, parent.block_of_mode)
            log('  sector d=%d N=%d dim=%d blocks=%d (%.1f s)' % (parent.d, N, sectors[key].dim, len(sectors[key].groups), time.time() - t0))
        return sectors[key]

    # ---------------- 5x1x1 parent, ell = 4a, R = 8a (T8.12)
    P5 = Parent(5); kt5 = KernelTables(P5, 4.0, 8.0, args.dr, args.nq)
    log('5x1x1: d=%d N0=%d D+=%d  h0 spectrum %s' % (P5.d, P5.N0, P5.Dplus, np.round(P5.eps0, 4)))
    log('symmetry: max commutator with vertices %.1e  off-block norm %.1e  block dims %s  sea block counts %s'
        % (P5.sym_comm, P5.offblock, P5.dims, np.round(P5.block_counts(P5.P0), 6)))
    log('kernel: raw norm err %.2e (data) %.2e (fine) -> normalized %.1e;  moment1 err %.2e  moment2 err %.2e  Rs=%.4f < R=%.1f'
        % (kt5.norm_err_raw, kt5.norm_err_raw_fine, kt5.norm_err, kt5.mom1_err, kt5.mom2_err, kt5.Rs, kt5.R))
    results['checks']['symmetry_5'] = dict(comm=P5.sym_comm, offblock=P5.offblock, dims=P5.dims, sea_counts=P5.block_counts(P5.P0))
    results['checks']['kernel_5'] = dict(norm_err_raw=kt5.norm_err_raw, norm_err_raw_fine=kt5.norm_err_raw_fine, norm_err=kt5.norm_err,
                                         mom1_err=kt5.mom1_err, mom2_err=kt5.mom2_err,
                                         convergence=kernel_convergence(P5, 4.0, 8.0, args.dr, args.nq, args.nq + 8))
    log('kernel convergence (nq %d vs %d): %s' % (args.nq, args.nq + 8, results['checks']['kernel_5']['convergence']))
    for hol in (2, 4): get_sector(P5, P5.d - hol)
    results['checks']['G0_N18'] = g0_check(P5, kt5, 2, sectors, args)
    log('G=0 check (N=18): %s' % results['checks']['G0_N18'])
    results['checks']['refinement_5'] = refinement_check(P5, 4.0, 8.0, args.dr, args.nq, couplings[0], 18, 'moving')
    log('grid refinement (dr vs dr/2) at G=%.3g: %s' % (couplings[0], results['checks']['refinement_5']))
    if not args.no_scan:
        log('\n--- coupling scan (5x1x1, N=18, moving reference) ---')
        glist = sorted(set([2.0 ** -k for k in range(12, -1, -1)] + [0.25]))
        results['scan_5'] = coupling_scan(P5, kt5, 18, glist)
        adm = [r['G'] for r in results['scan_5'] if r.get('admissible')]
        results['scan_5_max_admissible'] = max(adm) if adm else None
        log('largest admissible G/a^2 in the scan: %s' % results['scan_5_max_admissible'])
    for G in couplings:
        for ref in ('moving', 'fixed'):
            results['cases'].append(run_case(P5, kt5, G, 2, ref, sectors, args))
            results['cases'].append(run_case(P5, kt5, G, 4, ref, sectors, args))
    sec190e = Sector(P5.d, 18)
    for i, G in enumerate(couplings):
        results['T8'].append(canonical_block(P5, kt5, G, sec190e, args, do_evolve=(not args.no_evolve and i == 0)))

    # ---------------- 7x1x1 parent: size control and ell control
    P7 = Parent(7)
    log('\n7x1x1: symmetry comm %.1e off-block %.1e dims %s' % (P7.sym_comm, P7.offblock, P7.dims))
    results['checks']['symmetry_7'] = dict(comm=P7.sym_comm, offblock=P7.offblock, dims=P7.dims)
    get_sector(P7, P7.d - 2)
    if not args.skip_large: get_sector(P7, P7.d - 4)
    for ell in (4.0, 2.0, 8.0):
        R = 8.0 if ell <= 4.0 else 16.0
        kt7 = KernelTables(P7, ell, R, args.dr, args.nq)
        results['checks']['kernel_7_ell%g' % ell] = dict(norm_err_raw=kt7.norm_err_raw, mom1_err=kt7.mom1_err, mom2_err=kt7.mom2_err)
        log('\n7x1x1 ell=%g: kernel raw norm err %.2e  mom1 %.2e  mom2 %.2e  Rs=%.3f R=%.1f' % (ell, kt7.norm_err_raw, kt7.mom1_err, kt7.mom2_err, kt7.Rs, kt7.R))
        for G in couplings:
            for ref in ('moving', 'fixed'):
                results['cases'].append(run_case(P7, kt7, G, 2, ref, sectors, args))
            if not args.skip_large and (ell == 4.0 or args.ell_full):
                results['cases'].append(run_case(P7, kt7, G, 4, 'moving', sectors, args))
                if args.fixed_large: results['cases'].append(run_case(P7, kt7, G, 4, 'fixed', sectors, args))

    # ---------------- summary
    log('\n================ SUMMARY (energies in 1/a; sector energies include the scalar B20) ================')
    log('%-32s %8s %-12s %-12s %13s %13s %13s %13s %11s %11s %8s %8s %6s' % ('case', 'G', 'pat_H', 'pat_Q', 'E_free', 'E_H', 'E_HF', 'E_Q', 'HF/H-1', 'Q/HF-1', 'y_F', 'y_corr', '1/N_f'))
    for c in results['cases']:
        r = c['ratios']
        log('%-32s %8.3g %-12s %-12s %13.8f %13.8f %13.8f %13.8f %11.3e %11.3e %8.3f %8.3f %6.3f' % (
            c['tag'], c['G'], str(c['pattern_H']), str(c['pattern_Q']), c['E_free'], c['E_H'], c['E_HF'], c['E_Q'],
            r['HF_over_H'] - 1, r['Q_over_HF'] - 1,
            r['y_F'] if r['y_F'] is not None else float('nan'), r['y_corr'] if r['y_corr'] is not None else float('nan'), r['inv_Nf']))
    log('\n%-32s %12s %12s %12s %10s %10s %10s %10s %10s' % ('case', 'collarHF/H-1', 'collarQ/HF-1', '|<HF|Q>|^2', '||R-P0||', 'q*', 'rh(C_Q)', 'dR(C_Q)', 'Q<=HF'))
    for c in results['cases']:
        log('%-32s %12.3e %12.3e %12.6f %10.2e %10.2e %10.2e %10.2e %10s' % (c['tag'], c['ratios']['collar_HF_over_H'] - 1, c['ratios']['collar_Q_over_HF'] - 1,
                                                                             c['overlap_HF_exact'], c['dP0']['op'], c['V_info']['qstar'],
                                                                             c['source_taylor']['rh_Q'], c['reference_resolve']['dR_Q'], c['variational']['Q_le_HF']))
    log('\nScalar term (moving minus fixed reference, same parent/ell/N/G):')
    pairs = {}
    for c in results['cases']:
        pairs.setdefault((c['parent'], c['ell'], c['N'], c['G']), {})[c['ref']] = c
    for key, pr in sorted(pairs.items()):
        if 'moving' in pr and 'fixed' in pr:
            m, f = pr['moving'], pr['fixed']
            log('  L%d ell%g N%d G%.3g: B20 %.4e  Es %.4e  E_Q %.4e  E_H %.4e  E_HF %.4e  (E_Q-B20: mov %.6f fix %.6f)' % (
                key[0], key[1], key[2], key[3], m['scalars']['B20'] - f['scalars']['B20'], m['scalars']['Es'] - f['scalars']['Es'],
                m['E_Q'] - f['E_Q'], m['E_H'] - f['E_H'], m['E_HF'] - f['E_HF'], m['E_Q_minus_B20'], f['E_Q_minus_B20']))
    log('\nSlater-bound comparison at r = R_s (excess part of the mass variance; number-variance excess of the Slater states as the domain control):')
    for c in results['cases']:
        row = [x for x in c['variances'] if abs(x['r'] - c['Rs']) < 0.011][0]
        q = row['states']['Q']; hh = row['states']['H']; ff = row['states']['HF']
        verdict = 'n/a' if (q['rel_exc'] is None or q['bound_Nr'] is None) else ('at/below' if abs(q['rel_exc']) <= q['bound_Nr'] else 'ABOVE')
        fp = c['pair_H']['fock_over_pair']
        fock = 'n/a' if fp is None else ('at/below' if abs(fp) <= 1.0 / c['n'] else 'ABOVE')
        fmt = lambda v: 'n/a' if v is None else '%.3e' % v
        log('  %-32s rel_exc(Q)=%s  N_r^-1/2=%s -> %s ;  |E_exch/E_pair|(C_H)=%s vs 1/N_f=%.3f -> %s (kernel indefinite: %s) ;  numexc/N_r: H %s HF %s Q %s' % (
            c['tag'], fmt(q['rel_exc']), fmt(q['bound_Nr']), verdict, 'n/a' if fp is None else '%.3e' % abs(fp), 1.0 / c['n'], fock,
            c['pair_H']['V_eig_min'] < 0 < c['pair_H']['V_eig_max'], fmt(hh['num_exc_over_Nr']), fmt(ff['num_exc_over_Nr']), fmt(q['num_exc_over_Nr'])))
    # acceptance
    acc = {}
    acc['symmetry'] = P5.sym_comm < 1e-12 and P5.offblock < 1e-12 and P7.sym_comm < 1e-12 and P7.offblock < 1e-12
    acc['kernel_raw_norm'] = kt5.norm_err_raw < 1e-7 and kt5.norm_err < 1e-14
    acc['hessian_routes'] = all(c['hessian_checks']['cs_vs_lin'] < 1e-6 and c['hessian_checks']['cs_vs_fd'] < 1e-6 and c['hessian_checks']['asym'] < 1e-7 for c in results['cases'])
    acc['variational_identity'] = all(c['hessian_checks']['variational'] < 1e-7 and c['A_R_is_hbar'] < 1e-7 for c in results['cases'])
    acc['wick_identities'] = all(c['wick_checks']['raw_vs_HF'] < 1e-12 and c['wick_checks']['y_minus_one'] < 1e-12 and
                                 c['wick_checks']['hHF_gradient'] < 1e-7 and c['wick_checks']['first_variation_at_P'] < 1e-12 for c in results['cases'])
    acc['routes_A_B'] = all((c.get('routeA_vs_B') is None) or c['routeA_vs_B'] < 1e-10 for c in results['cases'])
    acc['blocks_vs_full'] = all(c['blocks_vs_full'] < 1e-10 for c in results['cases'])
    acc['variational_ordering'] = all(c['variational']['Q_le_HF'] and c['variational']['Q_le_HF_at_CH'] for c in results['cases'])
    acc['G0'] = results['checks']['G0_N18']['max_dev'] < 1e-12
    acc['T8_4'] = all(abs(t['T8_4']['residual']) < 1e-10 for t in results['T8'])
    acc['T8_6'] = all(t['T8_6']['Ycan_resid'] < 1e-8 and t['T8_6']['YF_resid'] < 1e-8 for t in results['T8'])
    acc['T8_8'] = all(all(abs(x['resid']) < 1e-10 for x in t['T8_8']) for t in results['T8'])
    acc['chart'] = all(c['margins']['mass_margin'] > 0 and c['margins']['sheet_gap'] > 0 and c['margins']['sheet_is_lowest'] and
                       c['V_info']['qstar'] < 1 and c['hartree']['converged'] for c in results['cases'])
    results['acceptance'] = acc
    log('\nACCEPTANCE: ' + '  '.join('%s=%s' % (k, 'PASS' if v else 'FAIL') for k, v in acc.items()))
    with open(args.out, 'w') as fh:
        json.dump(jsonable(results), fh, indent=1)
    log('written %s' % args.out)


if __name__ == '__main__':
    main()
