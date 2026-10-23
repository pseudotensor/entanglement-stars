#!/usr/bin/env python3
"""
hf_cluster13.py  (busi lane, task 4 = S10; first draft reviewer A, run by the builder on the box CPU)

The 13-site axial-cluster benchmark of the gravitational Hartree, Hartree-Fock and exact levels with the scalar
monopole kernel and the fixed free reference (S10's targets; the P03 web return's section 12.2 is re-derived here,
never used as a source), the 7-site control (blind to the radial current), the second-order coefficients and the
connected reference energy, with a self-test mode.

Units a = 1. Conventions (DECLARED):
  sites {0, +-e_i, +-2e_i} (13 sites, 52 modes; --cluster cross7: {0, +-e_i}, 7 sites, 28 modes)
  h0 = m beta + sum_i alpha_i p_i, p_i = (T_i - T_i^T)/(2i), (T_i u)(x) = u(x + e_i)   (sine hopping)
  beta = sigma_z x I_2, alpha_i = sigma_x x sigma_i
  shells r = |x| in {0, 1, 2}; Pi_r the shell projectors; e_r = {Pi_r, h0}/2 the shell energy vertices
  reference P0 = 1_{h0 < 0} (fixed, free; N0 = 26)
  pair kernel V = -G K, K = [[2.4, 1, 1/2], [1, 1, 1/2], [1/2, 1/2, 1/2]]: the monopole kernel 1/max(r, s) with the
  centre a uniform ball of radius 1/2 (self-kernel 6/(5R) = 2.4) and thin shells at 1 and 2
  E^ = dG(h0) - Tr(P0 h0) + (1/2) sum_rs V_rs W_{P0}(e_r, e_s), W_P the fully Wick-ordered pair (re-derived):
     W_P(A,B) = dG(A)dG(B) - Tr(PA)dG(B) - Tr(PB)dG(A) - dG(A(1-P)B) + dG(BPA) + Tr(PA)Tr(PB) - Tr(PBPA)
     <W_P(A,B)>_C = Tr(DA)Tr(DB) - Tr(DADB), D = C - P, for Gaussian C
  => E^ = E_s + dG(Y) + (1/2) sum_rs V_rs dG(e_r)dG(e_s),
     Y = h0 - sum_rs V_rs Tr(P e_r) e_s - (1/2) sum_rs V_rs [e_r(1-P)e_s - e_s P e_r],
     E_s = -Tr(P h0) + (1/2) sum_rs V_rs [Tr(P e_r)Tr(P e_s) - Tr(P e_s P e_r)]
  Hartree: E_H[C] = Tr(h0 D) + (1/2) q^T V q, q_r = Tr(D e_r), C the self-consistent lowest-(N0+Nf) projector of
           h_H = h0 + sum_rs V_rs q_s e_r (aufbau; the sea polarizes; --frozen-sea restricts D to the positive band)
  HF:      E_HF = E_H - (1/2) sum_rs V_rs Tr(D e_r D e_s), h_HF = h_H - sum_rs V_rs e_r D e_s
  exact:   the (N0 + Nf)-particle sector of E^ in the multiplicity-symmetric block.
Block structure (DERIVED; verified numerically at construction): every one-body operator entering E^ (h0, e_r, P0)
has the form (+)_s [X_{b,s} x I_2] (+)_b [X_{d,b} x I_10]: b sector = Krylov space of h0 from a centre spinor of
beta = s (centre, symmetric shell 1, symmetric shell 2: 3 levels, 2 copies, h = [[sm, sqrt6/2, 0], [sqrt6/2, -sm,
1/2], [0, 1/2, sm]]); d sector = shell-1 states orthogonal to the b sector with beta = b and their shell-2 partners
(2 levels, 10 copies, h = [[bm, 1/2], [1/2, -bm]]). E^ commutes with U(2) x U(2) x U(10) x U(10) and acts on the
component containing the reference and the closed-shell additions: S_lambda(C^3) x S_lambda(C^3) x Sym^10(C^2) x
Sym^10(C^2), 6 x 6 x 11 x 11 = 4356 (Nf = 0, 2), 8 x 6 x 11 x 11 = 5808 (Nf = 1, one copy-doublet component). The d
factors are the spin-5 quasi-spin multiplets (E_10 = J_+, E_00 = 10 - k, E_11 = k); the b factors are built in the
explicit 6-mode Fock space and projected on the copy-singlet (or copy-doublet) component. A norm bound certifies
that the global ground state of the sector lies in the block (2 G ||H_1|| < the free gap).
Outputs: JSON with every energy, the S10 targets and the deviations; PASS/FAIL lines in --selftest.
"""
import os
import sys


def _set_threads():
    if '--threads' in sys.argv:
        t = sys.argv[sys.argv.index('--threads') + 1]
        for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[v] = str(t)


_set_threads()
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from itertools import combinations  # noqa: E402

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402

SX = np.array([[0.0, 1.0], [1.0, 0.0]], complex)
SY = np.array([[0.0, -1j], [1j, 0.0]], complex)
SZ = np.array([[1.0, 0.0], [0.0, -1.0]], complex)
I2 = np.eye(2)
BETA = np.kron(SZ, I2)
ALPHA = [np.kron(SX, s) for s in (SX, SY, SZ)]
K_P03 = np.array([[2.4, 1.0, 0.5], [1.0, 1.0, 0.5], [0.5, 0.5, 0.5]])
K_7 = np.array([[2.4, 1.0], [1.0, 1.0]])
# S10's targets (binding) and P03's intermediate numbers (printed for diagnosis only)
S10 = dict(B_HF_over_B_H=0.5000102342, B_exact_sub_over_B_H=0.5000377984, E_Nf1_sub=0.1999999998282)
P03 = dict(E_H=0.3999894769740143, E_HF=0.3999947383793123, E_exact_raw=0.3999947379699671, E_ref_raw=-1.19286625e-10,
           E_exact_sub=0.3999947380892537, c2_ref=-0.00200087509614, c2_Nf2=-0.00923221355854,
           c2_Nf2_singles=-0.00236659565393, c2_Nf2_doubles=-0.00686561790462, c2_Nf1=-0.00488265354221,
           c1_Nf2=-0.02155102040816, free_gap=0.33851648071345, Vbound_over_G=389.729)


def log(*a):
    print(*a, flush=True)


def herm(X):
    return 0.5 * (X + X.conj().T)


def fnorm(X):
    return float(np.linalg.norm(X))


def trr(X):
    return float(np.trace(X).real)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, complex):
        return [o.real, o.imag]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else str(f)
    return o


# ============================================================================ cluster and one-body operators
class Cluster:
    def __init__(self, kind, m):
        kmax = {'star13': 2, 'cross7': 1}[kind]
        sites = [(0, 0, 0)]
        for i in range(3):
            for k in range(1, kmax + 1):
                for sgn in (1, -1):
                    v = [0, 0, 0]
                    v[i] = sgn * k
                    sites.append(tuple(v))
        self.kind, self.m, self.sites = kind, float(m), sites
        self.ns = len(sites)
        self.d = 4 * self.ns
        idx = {s: n for n, s in enumerate(sites)}
        self.T = []
        for i in range(3):
            T = np.zeros((self.ns, self.ns))
            for s in sites:
                t = list(s)
                t[i] += 1
                t = tuple(t)
                if t in idx:
                    T[idx[s], idx[t]] = 1.0
            self.T.append(T)
        self.p = [(T - T.T) / (2j) for T in self.T]
        self.Q = [(T @ T - T.T @ T.T) / (4j) for T in self.T]
        I4 = np.eye(4)
        Ins = np.eye(self.ns)
        self.h0 = m * np.kron(Ins, BETA) + sum(np.kron(self.p[i], ALPHA[i]) for i in range(3))
        self.rad = np.array([int(round(math.sqrt(sum(c * c for c in s)))) for s in sites])
        self.shells = sorted(set(self.rad.tolist()))
        self.nsh = len(self.shells)
        self.Pi = [np.kron(np.diag((self.rad == r).astype(float)), I4) for r in self.shells]
        self.e = [herm(P @ self.h0) for P in self.Pi]
        self.Jrad = []  # shell-summed radial two-step current sum_{x in shell} xhat . P_x, P_{x,i} = {Pi_x, Q_i}/2
        for r in self.shells:
            J = np.zeros((self.d, self.d), complex)
            for n, s in enumerate(sites):
                if self.rad[n] == r and r > 0:
                    Px = np.kron(np.diag(Ins[n]), I4)
                    for i in range(3):
                        if s[i] != 0:
                            J += (s[i] / r) * herm(Px @ np.kron(self.Q[i], I4))
            self.Jrad.append(J)
        eps, U = np.linalg.eigh(self.h0)
        self.eps0, self.U0 = eps, U
        self.N0 = int(np.sum(eps < 0))
        self.P0 = U[:, :self.N0] @ U[:, :self.N0].conj().T
        self.E_sea = float(np.sum(eps[eps < 0]))

    def analytic_spectrum(self):
        m = self.m
        if self.kind == 'star13':
            eb = math.sqrt(m * m + 1.75)
            ed = math.sqrt(m * m + 0.25)
            return sorted([-eb] * 4 + [-ed] * 20 + [-m] * 2 + [m] * 2 + [ed] * 20 + [eb] * 4)
        eb = math.sqrt(m * m + 1.5)
        return sorted([-eb] * 4 + [-m] * 10 + [m] * 10 + [eb] * 4)


def gauss_pair(cl, V, C):
    """(1/2) sum_rs V_rs [Tr(De_r)Tr(De_s) - Tr(De_rDe_s)] (direct plus exchange), and its two parts."""
    D = C - cl.P0
    q = np.array([trr(D @ A) for A in cl.e])
    direct = 0.5 * float(q @ V @ q)
    exch = -0.5 * sum(V[r, s] * trr(D @ cl.e[r] @ D @ cl.e[s]) for r in range(cl.nsh) for s in range(cl.nsh))
    return direct, exch, q


def wick_onebody(cl, V):
    """E^ = E_s + dG(Y) + (1/2) sum_rs V_rs dG(e_r) dG(e_s): the scalar and one-body coefficients of the fully
    Wick-ordered pair about P0 (docstring)."""
    P, e, I = cl.P0, cl.e, np.eye(cl.d)
    tP = np.array([trr(P @ A) for A in e])
    Y = cl.h0.astype(complex).copy()
    Es = -trr(P @ cl.h0)
    for r in range(cl.nsh):
        for s in range(cl.nsh):
            if V[r, s] == 0.0:
                continue
            Y = Y - V[r, s] * tP[r] * e[s] - 0.5 * V[r, s] * (e[r] @ (I - P) @ e[s] - e[s] @ P @ e[r])
            Es += 0.5 * V[r, s] * (tP[r] * tP[s] - trr(P @ e[s] @ P @ e[r]))
    return herm(Y), float(Es)


def E_raw_gauss(cl, V, Y, Es, C):
    """<E^>_C for Gaussian C from the raw form: Es + Tr(CY) + (1/2) sum V [Tr(Ce_r)Tr(Ce_s) + Tr(Ce_r(1-C)e_s)]."""
    I = np.eye(cl.d)
    val = Es + trr(C @ Y)
    for r in range(cl.nsh):
        for s in range(cl.nsh):
            if V[r, s] != 0.0:
                val += 0.5 * V[r, s] * (trr(C @ cl.e[r]) * trr(C @ cl.e[s]) + trr(C @ cl.e[r] @ (I - C) @ cl.e[s]))
    return float(val)


# ============================================================================ self-consistent fields
def select_occupied(eps, U, N, Cprev, tol=1e-9):
    """Lowest-N orbitals; inside a degenerate shell cut by the Fermi level, the combination of maximal weight in
    the previous state (maximum overlap)."""
    d = len(eps)
    if N == 0:
        return U[:, :0]
    if N >= d or eps[N] - eps[N - 1] > tol:
        return U[:, :N]
    ef = eps[N - 1]
    i0 = int(np.searchsorted(eps, ef - tol))
    i1 = int(np.searchsorted(eps, ef + tol))
    need = N - i0
    Ug = U[:, i0:i1]
    w, R = np.linalg.eigh(herm(Ug.conj().T @ Cprev @ Ug))
    sel = Ug @ R[:, ::-1][:, :need]
    return np.concatenate([U[:, :i0], sel], axis=1)


def scf(cl, V, N, mode, frozen_sea=False, tol=1e-13, maxit=3000, damp=0.5):
    """Hartree ('H') or Hartree-Fock ('HF') self-consistent lowest-N projector; with frozen_sea the matter part
    D = C - P0 is restricted to the positive band of h0 (labelled alternative)."""
    e, nsh, P0 = cl.e, cl.nsh, cl.P0
    F0 = np.eye(cl.d) - P0
    Nf = N - cl.N0
    C = cl.U0[:, :N] @ cl.U0[:, :N].conj().T
    res = np.inf
    it = 0
    for it in range(maxit):
        D = C - P0
        q = np.array([trr(D @ A) for A in e])
        u = V @ q
        h = cl.h0 + sum(u[r] * e[r] for r in range(nsh))
        if mode == 'HF':
            h = h - sum(V[r, s] * (e[r] @ D @ e[s]) for r in range(nsh) for s in range(nsh))
        h = herm(h)
        if frozen_sea:
            hp = herm(F0 @ h @ F0)
            eps, U = np.linalg.eigh(hp)
            # positive-band eigenvectors: those with eigenvalue of hp not in the (near-)null space of the sea
            pos = np.where(np.abs(np.einsum('ia,ij,ja->a', U.conj(), F0, U)).real > 0.5)[0]
            Up = U[:, pos]
            ep = eps[pos]
            Uo = select_occupied(ep, Up, Nf, D)
            Cn = P0 + Uo @ Uo.conj().T
        else:
            eps, U = np.linalg.eigh(h)
            Uo = select_occupied(eps, U, N, C)
            Cn = Uo @ Uo.conj().T
        res = fnorm(Cn - C)
        C = Cn if res < 1e-7 else (1.0 - damp) * C + damp * Cn
        if res < tol:
            break
    D = C - P0
    direct, exch, q = gauss_pair(cl, V, C)
    E0 = trr(cl.h0 @ D)
    out = dict(mode=mode, frozen_sea=frozen_sea, N=N, Nf=Nf, E0=E0, EH=direct, EF=exch, q=q.tolist(),
               E=E0 + direct + (exch if mode == 'HF' else 0.0), E_HF_functional=E0 + direct + exch,
               iterations=it + 1, residual=res, idempotency=fnorm(C @ C - C), converged=bool(res < tol))
    return C, out


# ============================================================================ block structure
def lanczos(h, v0, maxdim, tol=1e-9):
    vecs = [v0 / np.linalg.norm(v0)]
    al, be = [], []
    w = h @ vecs[0]
    a = float(np.vdot(vecs[0], w).real)
    al.append(a)
    w = w - a * vecs[0]
    for _ in range(1, maxdim):
        b = float(np.linalg.norm(w))
        if b < tol:
            break
        v = w / b
        be.append(b)
        vecs.append(v)
        w = h @ v
        a = float(np.vdot(v, w).real)
        al.append(a)
        w = w - a * v - b * vecs[-2]
    return np.stack(vecs, axis=1), np.array(al), np.array(be)


class Blocks:
    """Types ('b', s): the Krylov spaces of h0 from the centre spinors of beta = s; ('d', b): the Krylov spaces from
    the shell-1 states orthogonal to the b sector with beta = b. Every type is rotated to the h0 eigenbasis of its
    levels (ascending); the copies of a type carry identical matrices for every block-scalar operator."""

    def __init__(self, cl):
        d, ns, h0 = cl.d, cl.ns, cl.h0
        self.types = []
        bvecs = []
        for s in (1, -1):
            copies = []
            al = be = None
            for k in range(4):
                if int(round(BETA[k, k].real)) != s:
                    continue
                v0 = np.zeros(d, complex)
                v0[k] = 1.0  # centre site is site 0
                V, al, be = lanczos(h0, v0, cl.nsh)
                copies.append(V)
            self.types.append(dict(name='b%+d' % s, copies=copies, lanczos=(al, be)))
            bvecs += copies
        Ub = np.concatenate(bvecs, axis=1)
        shell1 = [n for n in range(ns) if cl.rad[n] == 1]
        for b in (1, -1):
            cols = [4 * n + k for n in shell1 for k in range(4) if int(round(BETA[k, k].real)) == b]
            Bm = np.eye(d, dtype=complex)[:, cols]
            Bm = Bm - Ub @ (Ub.conj().T @ Bm)
            u, sv, _ = np.linalg.svd(Bm, full_matrices=False)
            phis = u[:, sv > 0.5]
            copies = [lanczos(h0, phis[:, c], cl.nsh)[0] for c in range(phis.shape[1])]
            _, al, be = lanczos(h0, phis[:, 0], cl.nsh)
            self.types.append(dict(name='d%+d' % b, copies=copies, lanczos=(al, be)))
        for t in self.types:
            W1 = t['copies'][0]
            X = herm(W1.conj().T @ h0 @ W1)
            eps, R = np.linalg.eigh(X)
            R = R.astype(complex)
            for c in range(R.shape[1]):
                i = int(np.argmax(np.abs(R[:, c])))
                R[:, c] *= np.conj(R[i, c]) / abs(R[i, c])
            t['copies'] = [V @ R for V in t['copies']]
            t['n'], t['mult'], t['eps'] = len(eps), len(t['copies']), eps
        Wall = np.concatenate([V for t in self.types for V in t['copies']], axis=1)
        self.Wall = Wall
        self.ortho_err = fnorm(Wall.conj().T @ Wall - np.eye(Wall.shape[1]))
        self.complete = Wall.shape[1] == d

    def reduce(self, X):
        """Per-type level matrices (real) and the structure error: deviation between copies plus the rebuild error."""
        mats, err = [], 0.0
        for t in self.types:
            X1 = t['copies'][0].conj().T @ X @ t['copies'][0]
            for V in t['copies'][1:]:
                err = max(err, fnorm(V.conj().T @ X @ V - X1))
            mats.append(X1)
        rebuilt = np.zeros(X.shape, complex)
        for t, X1 in zip(self.types, mats):
            for V in t['copies']:
                rebuilt += V @ X1 @ V.conj().T
        err = max(err, fnorm(rebuilt - X))
        imag = max(float(np.max(np.abs(X1.imag))) for X1 in mats)
        return [X1.real.copy() for X1 in mats], err, imag


# ============================================================================ Fock-space components
class FermiSpace:
    """N-particle space of nmodes modes in the bit basis; c_i^dag c_j as dense real matrices."""

    def __init__(self, nmodes, N):
        self.nmodes, self.N = nmodes, N
        self.basis = [sum(1 << i for i in c) for c in combinations(range(nmodes), N)]
        self.index = {b: k for k, b in enumerate(self.basis)}
        self.dim = len(self.basis)

    @staticmethod
    def _par(b, i):
        return -1.0 if bin(b & ((1 << i) - 1)).count('1') % 2 else 1.0

    def hop(self, i, j):
        M = np.zeros((self.dim, self.dim))
        for k, b in enumerate(self.basis):
            if not (b >> j) & 1:
                continue
            s = self._par(b, j)
            b1 = b & ~(1 << j)
            if (b1 >> i) & 1:
                continue
            s *= self._par(b1, i)
            M[self.index[b1 | (1 << i)], k] += s
        return M

    def dG(self, A):
        return sum(A[i, j] * self.hop(i, j) for i in range(self.nmodes) for j in range(self.nmodes) if A[i, j] != 0)

    def det_vector(self, occ):
        v = np.zeros(self.dim)
        v[self.index[sum(1 << i for i in occ)]] = 1.0
        return v


def explicit_component(n, mult, N, kind, mz=None):
    """gl(n) generators E_ij = sum_c c^dag_{ic} c_{jc} (mode = mult*i + c) on the U(2) component of the N-particle
    space of n x 2 modes: 'singlet' (every level empty or full in both copies) or 'doublet' with K_z = mz."""
    assert mult == 2
    fs = FermiSpace(n * mult, N)
    E = [[sum(fs.hop(mult * i + c, mult * j + c) for c in range(mult)) for j in range(n)] for i in range(n)]
    K00 = sum(fs.hop(mult * l, mult * l) for l in range(n))
    K11 = sum(fs.hop(mult * l + 1, mult * l + 1) for l in range(n))
    Kp = sum(fs.hop(mult * l, mult * l + 1) for l in range(n))
    Kz = 0.5 * (K00 - K11)
    Cas = Kp.T @ Kp + Kz @ Kz + Kz
    target = 0.0 if kind == 'singlet' else 0.75
    w, R = np.linalg.eigh(0.5 * (Cas + Cas.T))
    W = R[:, np.abs(w - target) < 1e-8]
    if kind == 'doublet':
        wz, Rz = np.linalg.eigh(0.5 * (W.T @ Kz @ W + (W.T @ Kz @ W).T))
        W = W @ Rz[:, np.abs(wz - mz) < 1e-8]
    Ec = [[W.T @ E[i][j] @ W for j in range(n)] for i in range(n)]
    return Ec, W, fs


def quasispin_rep(mult):
    """gl(2) generators on the symmetric (spin mult/2) multiplet |k> (k particles in the upper level), E_10 = J_+."""
    k = np.arange(mult + 1)
    E00 = np.diag((mult - k).astype(float))
    E11 = np.diag(k.astype(float))
    E10 = np.zeros((mult + 1, mult + 1))
    for kk in range(mult):
        E10[kk + 1, kk] = math.sqrt((kk + 1) * (mult - kk))
    return [[E00, E10.T.copy()], [E10, E11]]


class BlockFock:
    """The multiplicity-symmetric component of the (N0 + Nf)-particle sector: a tensor product over the types."""

    def __init__(self, blocks, Nf):
        pos = [(float(t['eps'][l]), ti, l) for ti, t in enumerate(blocks.types) for l in range(t['n']) if t['eps'][l] > 0]
        self.emin, self.ti_star, l_star = min(pos)
        self.Nf = Nf
        self.reps = []
        for ti, t in enumerate(blocks.types):
            n, mult, eps = t['n'], t['mult'], t['eps']
            nsea = int(np.sum(eps < 0))
            if t['name'].startswith('b'):
                extra = Nf if ti == self.ti_star else 0
                if ti == self.ti_star:
                    assert l_star == nsea, 'the lowest positive level must be the first empty level of its type'
                    assert Nf <= mult
                N = mult * nsea + extra
                k, rem = divmod(N, mult)
                kind = 'singlet' if rem == 0 else 'doublet'
                E, W, fs = explicit_component(n, mult, N, kind, 0.5 if rem else None)
                free_modes = [mult * l + c for l in range(k) for c in range(mult)] + ([mult * k] if rem else [])
                free = W.T @ fs.det_vector(free_modes)
            else:
                assert nsea == 1 and n == 2, 'd types must be two-level'
                E = quasispin_rep(mult)
                free = np.eye(mult + 1)[0].copy()
                N = mult
            Hk = sum(eps[l] * E[l][l] for l in range(n))
            w, R = np.linalg.eigh(0.5 * (Hk + Hk.T))
            E = [[R.T @ E[i][j] @ R for j in range(n)] for i in range(n)]
            free = R.T @ free
            weights = np.array([[E[l][l][a, a] for l in range(n)] for a in range(len(w))])
            i0 = int(np.argmax(np.abs(free)))
            rank = np.rint(np.sum(np.abs(weights - weights[i0][None, :]), axis=1) / 2.0).astype(int)
            self.reps.append(dict(name=t['name'], E=E, dim=len(w), free=free, n=n, mult=mult, N=N, H0=w,
                                  weights=weights, rank=rank, free_is_basis=float(abs(free[i0])), i0=i0))
        self.dims = [r['dim'] for r in self.reps]
        self.dim = int(np.prod(self.dims))

    def embed(self, k, A):
        left = int(np.prod(self.dims[:k])) if k > 0 else 1
        right = int(np.prod(self.dims[k + 1:])) if k + 1 < len(self.dims) else 1
        return np.kron(np.kron(np.eye(left), A), np.eye(right))

    def dG_type(self, k, X):
        r = self.reps[k]
        return sum(X[i, j] * r['E'][i][j] for i in range(r['n']) for j in range(r['n']) if X[i, j] != 0.0)

    def dG(self, Xtypes):
        return sum(self.embed(k, self.dG_type(k, X)) for k, X in enumerate(Xtypes))

    def kron_sum(self, vecs):
        out = np.zeros(1)
        for v in vecs:
            out = (out[:, None] + np.asarray(v)[None, :]).ravel()
        return out

    def H0_diag(self):
        return self.kron_sum([r['H0'] for r in self.reps])

    def rank_total(self):
        return self.kron_sum([r['rank'] for r in self.reps]).astype(int)

    def free_vector(self):
        v = self.reps[0]['free']
        for r in self.reps[1:]:
            v = np.kron(v, r['free'])
        return v

    def onebody_density(self, psi):
        """Per-type level density matrices <E_ij>/mult of a block vector."""
        T = psi.reshape(self.dims)
        out = []
        for k, r in enumerate(self.reps):
            pk = np.moveaxis(T, k, 0).reshape(self.dims[k], -1)
            M = pk.conj() @ pk.T
            rho = np.array([[np.sum(r['E'][i][j] * M) for j in range(r['n'])] for i in range(r['n'])]) / r['mult']
            out.append(rho)
        return out

    def closed_shell_vector(self, blocks, C):
        """The Slater determinant of a block-scalar projector C (closed shells in every type), via Gamma(u) =
        exp(dGamma(log u)) applied to the free vector type by type; None if C is not block-scalar."""
        vecs = []
        for t, r in zip(blocks.types, self.reps):
            W1 = t['copies'][0]
            Ct = W1.conj().T @ C @ W1
            for V in t['copies'][1:]:
                if fnorm(V.conj().T @ C @ V - Ct) > 1e-8:
                    return None
            kf = r['N'] // r['mult']
            if kf * r['mult'] != r['N']:
                return None
            w, R = np.linalg.eigh(herm(Ct))
            u = np.concatenate([R[:, ::-1][:, :kf], R[:, ::-1][:, kf:]], axis=1).astype(complex)
            for c in range(u.shape[1]):
                if abs(u[c, c]) > 1e-12:
                    u[:, c] *= np.conj(u[c, c]) / abs(u[c, c])
            X = sla.logm(u)
            Gm = sum(X[i, j] * r['E'][i][j] for i in range(r['n']) for j in range(r['n']))
            vecs.append(sla.expm(Gm) @ r['free'].astype(complex))
        v = vecs[0]
        for x in vecs[1:]:
            v = np.kron(v, x)
        return v


def build_H(bf, e_types, Y_types, Es, V):
    """E_s + dG(Y) + (1/2) sum_rs V_rs dG(e_r) dG(e_s) on the block (real symmetric)."""
    dGe = [bf.dG(Xe) for Xe in e_types]
    H = Es * np.eye(bf.dim) + bf.dG(Y_types)
    for r in range(len(dGe)):
        for s in range(len(dGe)):
            if V[r, s] != 0.0:
                H += 0.5 * V[r, s] * (dGe[r] @ dGe[s])
    return 0.5 * (H + H.T)


def op_norm_dG(X):
    w = np.linalg.eigvalsh(herm(X))
    return float(max(np.sum(w[w > 0]), -np.sum(w[w < 0])))


def trace_norm(X):
    return float(np.sum(np.linalg.svd(X, compute_uv=False)))


def wick_bound(A, B, P):
    """Operator-norm bound of W_P(A, B) on the full Fock space, term by term."""
    I = np.eye(len(A))
    tA, tB = trr(P @ A), trr(P @ B)
    return (op_norm_dG(A) * op_norm_dG(B) + abs(tA) * op_norm_dG(B) + abs(tB) * op_norm_dG(A)
            + trace_norm(A @ (I - P) @ B) + trace_norm(B @ P @ A) + abs(tA * tB) + abs(trr(P @ B @ P @ A)))


# ============================================================================ sectors
def run_sector(cl, blocks, Nf, G, K, types_cache, frozen_alt=True):
    t0 = time.time()
    N = cl.N0 + Nf
    V = -G * K
    out = dict(Nf=Nf, N=N, G=G)
    # one-body levels
    CH, oH = scf(cl, V, N, 'H')
    CF, oF = scf(cl, V, N, 'HF')
    out['Hartree'], out['HF'] = oH, oF
    if frozen_alt:
        _, oHf = scf(cl, V, N, 'H', frozen_sea=True)
        _, oFf = scf(cl, V, N, 'HF', frozen_sea=True)
        out['Hartree_frozen_sea'], out['HF_frozen_sea'] = oHf, oFf
    log('  Nf=%d: Hartree E=%.16f (E0 %.16f, EH %.6e, EF %.6e, it %d, res %.1e)  HF E=%.16f (it %d, res %.1e)'
        % (Nf, oH['E'], oH['E0'], oH['EH'], oH['EF'], oH['iterations'], oH['residual'], oF['E'], oF['iterations'], oF['residual']))
    # block Hamiltonian H = H0 + G H1 (affine in G)
    bf = BlockFock(blocks, Nf)
    out['block_dims'] = bf.dims
    out['block_dim'] = bf.dim
    out['free_is_basis'] = [r['free_is_basis'] for r in bf.reps]
    e_types, h0_types, P_types = types_cache
    Y0, Es0 = wick_onebody(cl, np.zeros_like(K))
    Y1, Es1 = wick_onebody(cl, -K)  # at G = 1
    H0 = build_H(bf, e_types, blocks.reduce(Y0)[0], Es0, np.zeros_like(K))
    H1 = build_H(bf, e_types, blocks.reduce(Y1)[0], Es1, -K) - H0
    H0d = bf.H0_diag()
    out['H0_offdiag'] = fnorm(H0 - np.diag(H0d)) / max(fnorm(H0), 1e-300)
    E_free = Nf * bf.emin
    out['E_free'] = E_free
    vfree = bf.free_vector()
    i0 = int(np.argmax(np.abs(vfree)))
    out['free_vector_weight'] = float(abs(vfree[i0]))
    out['E_free_from_H0'] = float(H0d[i0])
    # expectation checks at the free determinant
    Cfree = cl.U0[:, :N] @ cl.U0[:, :N].conj().T
    dir0, exc0, _ = gauss_pair(cl, V, Cfree)
    out['free_expectation_block'] = float(vfree @ (H0 + G * H1) @ vfree)
    out['free_expectation_gauss'] = E_free + dir0 + exc0
    # perturbation theory
    rank = bf.rank_total()
    dE = H0d - H0d[i0]
    amp = H1[:, i0].copy()
    amp[i0] = 0.0
    mask = np.abs(dE) > 1e-10
    terms = np.zeros_like(amp)
    terms[mask] = -np.abs(amp[mask]) ** 2 / dE[mask]
    out['c1'] = float(H1[i0, i0])
    out['c2'] = float(np.sum(terms))
    out['c2_singles'] = float(np.sum(terms[rank == 1]))
    out['c2_doubles'] = float(np.sum(terms[rank == 2]))
    out['c2_higher'] = float(np.sum(terms[rank > 2]))
    out['degenerate_free_coupling'] = float(np.max(np.abs(amp[~mask]))) if np.any(~mask & (np.arange(bf.dim) != i0)) else 0.0
    # exact at G and G/2
    exact = {}
    for g in (G, 0.5 * G):
        H = H0 + g * H1
        w, X = np.linalg.eigh(H)
        psi = X[:, 0]
        exact[g] = dict(E=float(w[0]), E2=float(w[1]), residual=fnorm(H @ psi - w[0] * psi), psi=psi)
    out['E_exact'] = exact[G]['E']
    out['E_exact_second'] = exact[G]['E2']
    out['exact_residual'] = exact[G]['residual']
    out['E_exact_half_G'] = exact[0.5 * G]['E']
    c2G = (exact[G]['E'] - E_free - out['c1'] * G) / G ** 2
    c2h = (exact[0.5 * G]['E'] - E_free - out['c1'] * 0.5 * G) / (0.5 * G) ** 2
    out['c2_richardson'] = 2.0 * c2h - c2G
    out['c2_from_E_at_G'] = c2G
    psi = exact[G]['psi']
    rhos = bf.onebody_density(psi)
    out['natural_occupations'] = [sorted(np.linalg.eigvalsh(herm(r)).tolist(), reverse=True) for r in rhos]
    # HF and Hartree determinants in the block (closed shells only)
    for lab, C, oo in (('HF', CF, oF), ('Hartree', CH, oH)):
        v = bf.closed_shell_vector(blocks, C)
        if v is None:
            out['%s_in_block' % lab] = None
            continue
        val = float(np.vdot(v, (H0 + G * H1) @ v).real)
        out['%s_in_block' % lab] = dict(norm=float(np.linalg.norm(v)), expectation=val, one_body=oo['E_HF_functional'],
                                        diff=val - oo['E_HF_functional'], overlap_exact=float(abs(np.vdot(v, psi)) ** 2))
    # certificate
    bound = sum(abs(K[r, s]) / 2.0 * wick_bound(cl.e[r], cl.e[s], cl.P0) for r in range(cl.nsh) for s in range(cl.nsh))
    out['H1_norm_bound'] = bound
    out['free_gap_sector'] = float(min(cl.eps0[cl.eps0 > bf.emin + 1e-9].min() - bf.emin,
                                       bf.emin - cl.eps0[cl.eps0 < 0].max())) if Nf > 0 else float(cl.eps0[cl.eps0 > 0].min() - cl.eps0[cl.eps0 < 0].max())
    out['certificate'] = bool(2.0 * G * bound < out['free_gap_sector'])
    out['seconds'] = time.time() - t0
    log('  Nf=%d: block %s dim %d  E_exact=%.16f (second %.12f, resid %.1e)  c1=%.14f  c2=%.14f (s %.14f d %.14f h %.1e)  '
        'Richardson c2=%.12f  bound 2G||H1||=%.3e < gap %.5f: %s  (%.0f s)'
        % (Nf, bf.dims, bf.dim, out['E_exact'], out['E_exact_second'], out['exact_residual'], out['c1'], out['c2'],
           out['c2_singles'], out['c2_doubles'], out['c2_higher'], out['c2_richardson'], 2.0 * G * bound, out['free_gap_sector'],
           out['certificate'], out['seconds']))
    for lab in ('HF', 'Hartree'):
        x = out.get('%s_in_block' % lab)
        if x:
            log('  Nf=%d: %s determinant in block: <Phi|H|Phi>=%.16f  one-body %.16f  diff %.1e  |<Phi|Q>|^2=%.8f'
                % (Nf, lab, x['expectation'], x['one_body'], x['diff'], x['overlap_exact']))
    return out


def run_main(args):
    cl = Cluster('star13', args.ma)
    blocks = Blocks(cl)
    K = K_P03
    log('13-site axial star: d=%d N0=%d  spectrum vs analytic max dev %.1e  blocks ortho %.1e complete %s'
        % (cl.d, cl.N0, float(np.max(np.abs(cl.eps0 - np.array(cl.analytic_spectrum())))), blocks.ortho_err, blocks.complete))
    for t in blocks.types:
        log('  type %s: n=%d mult=%d levels %s  Lanczos alpha %s beta %s' % (t['name'], t['n'], t['mult'], np.round(t['eps'], 6),
                                                                        np.round(t['lanczos'][0], 10), np.round(t['lanczos'][1], 10)))
    e_types = [blocks.reduce(A)[0] for A in cl.e]
    h0_types, err_h0, _ = blocks.reduce(cl.h0)
    P_types, err_P, _ = blocks.reduce(cl.P0)
    err_e = max(blocks.reduce(A)[1] for A in cl.e)
    log('  block-scalar structure errors: h0 %.1e  e_r %.1e  P0 %.1e' % (err_h0, err_e, err_P))
    log('  shell radial-current norms: %s' % [fnorm(J) for J in cl.Jrad])
    results = dict(settings=dict(ma=args.ma, G=args.G, cluster='star13', K=K.tolist()), types=[dict(name=t['name'], n=t['n'], mult=t['mult'],
                   eps=t['eps'].tolist(), lanczos_alpha=t['lanczos'][0].tolist(), lanczos_beta=t['lanczos'][1].tolist()) for t in blocks.types],
                   structure_errors=dict(h0=err_h0, e=err_e, P0=err_P, ortho=blocks.ortho_err),
                   radial_current_norms=[fnorm(J) for J in cl.Jrad], sectors={})
    cache = (e_types, h0_types, P_types)
    Nfs = [int(x) for x in args.Nf.split(',')]
    for Nf in Nfs:
        results['sectors'][Nf] = run_sector(cl, blocks, Nf, args.G, K, cache, frozen_alt=not args.no_frozen)
    sec = results['sectors']
    summary = {}
    if 0 in sec and 2 in sec:
        E_ref = sec[0]['E_exact']
        B_H = 2 * sec[2]['Hartree']['E0'] * 0 + (sec[2]['E_free'] - sec[2]['Hartree']['E'])
        B_HF = sec[2]['E_free'] - sec[2]['HF']['E']
        E_sub = sec[2]['E_exact'] - E_ref
        B_Q = sec[2]['E_free'] - E_sub
        summary.update(E_ref_raw=E_ref, E_H=sec[2]['Hartree']['E'], E_HF=sec[2]['HF']['E'], E_exact_raw=sec[2]['E_exact'],
                       E_exact_sub=E_sub, B_H=B_H, B_HF=B_HF, B_exact_sub=B_Q, B_HF_over_B_H=B_HF / B_H,
                       B_exact_sub_over_B_H=B_Q / B_H, B_exact_raw_over_B_H=(sec[2]['E_free'] - sec[2]['E_exact']) / B_H,
                       c2_ref=sec[0]['c2'], c2_Nf2=sec[2]['c2'], c2_Nf2_minus_ref=sec[2]['c2'] - sec[0]['c2'],
                       c2_ref_richardson=sec[0]['c2_richardson'], c2_Nf2_richardson=sec[2]['c2_richardson'])
        if 1 in sec:
            summary.update(E_Nf1_raw=sec[1]['E_exact'], E_Nf1_sub=sec[1]['E_exact'] - E_ref, c2_Nf1=sec[1]['c2'],
                           E_Nf1_HF=sec[1]['HF']['E'], E_Nf1_H=sec[1]['Hartree']['E'])
        summary['targets_S10'] = {k: dict(target=v, ours=summary.get(k), rel_dev=(summary.get(k) - v) / v if summary.get(k) is not None else None)
                                  for k, v in S10.items()}
        summary['P03_intermediate'] = {k: dict(P03=v, ours=summary.get(k)) for k, v in P03.items() if k in summary}
        log('\nSUMMARY (G/a^2 = %.10g, ma = %g):' % (args.G, args.ma))
        for k in ('E_ref_raw', 'E_H', 'E_HF', 'E_exact_raw', 'E_exact_sub', 'B_H', 'B_HF', 'B_exact_sub', 'B_HF_over_B_H',
                  'B_exact_sub_over_B_H', 'B_exact_raw_over_B_H', 'E_Nf1_raw', 'E_Nf1_sub', 'E_Nf1_HF', 'c2_ref', 'c2_Nf2',
                  'c2_Nf2_minus_ref', 'c2_Nf1'):
            if k in summary:
                log('  %-24s %.16g' % (k, summary[k]))
        for k, v in summary['targets_S10'].items():
            log('  TARGET %-22s S10 %.13g  ours %.13g  rel dev %s' % (k, v['target'], v['ours'] if v['ours'] is not None else float('nan'),
                                                                    'n/a' if v['rel_dev'] is None else '%.2e' % v['rel_dev']))
        acc = all(v['rel_dev'] is not None and abs(v['rel_dev']) < 1e-9 for v in summary['targets_S10'].values())
        summary['targets_pass'] = acc
        log('  S10 targets within 1e-9 relative: %s' % ('PASS' if acc else 'FAIL'))
    results['summary'] = summary
    with open(args.out, 'w') as fh:
        json.dump(jsonable(results), fh, indent=1)
    log('written %s' % args.out)


# ============================================================================ self-tests
def selftest(args):
    fails = 0

    def report(name, val, tol, ok=None):
        nonlocal fails
        ok = (val < tol) if ok is None else ok
        fails += 0 if ok else 1
        log('%s  %-58s %.3e (tol %.0e)' % ('PASS' if ok else 'FAIL', name, val, tol))

    m = args.ma
    cl = Cluster('star13', m)
    report('A1 spectrum of h0 equals the analytic set', float(np.max(np.abs(cl.eps0 - np.array(cl.analytic_spectrum())))), 1e-12)
    bl = Blocks(cl)
    report('A2 block basis orthonormal and complete', bl.ortho_err + (0.0 if bl.complete else 1.0), 1e-12)
    lanz = 0.0
    for t in bl.types:
        al, be = t['lanczos']
        s = 1 if t['name'].endswith('+1') else -1
        if t['name'].startswith('b'):
            lanz = max(lanz, fnorm(al - np.array([s * m, -s * m, s * m])), fnorm(be - np.array([math.sqrt(6) / 2, 0.5])))
        else:
            lanz = max(lanz, fnorm(al - np.array([s * m, -s * m])), fnorm(be - np.array([0.5])))
    report('A2 Lanczos coefficients (sm, -sm, sm; sqrt6/2, 1/2; bm, -bm; 1/2)', lanz, 1e-12)
    errs = [bl.reduce(X)[1] for X in [cl.h0, cl.P0] + cl.e]
    report('A2 block-scalar structure of h0, P0, e_r', max(errs), 1e-12)
    report('A2 imaginary parts of the reduced matrices', max(bl.reduce(X)[2] for X in [cl.h0, cl.P0] + cl.e), 1e-12)
    jr = [fnorm(J) for J in cl.Jrad]
    report('A2 radial current vanishes on shells 0,1 of the star', max(jr[:2]), 1e-14)
    report('A2 radial current nonzero on shell 2 of the star', jr[2], 1e-3, ok=jr[2] > 1e-3)
    c7 = Cluster('cross7', m)
    report('A2 7-site cluster blind to the radial current (all shells)', max(fnorm(J) for J in c7.Jrad), 1e-14)
    report('A2 7-site radial energy vertices commute with h0', max(fnorm(A @ c7.h0 - c7.h0 @ A) for A in c7.e), 1e-12)
    report('A2 13-site radial energy vertices do not commute with h0', fnorm(cl.e[0] @ cl.h0 - cl.h0 @ cl.e[0]), 1e-3,
           ok=fnorm(cl.e[0] @ cl.h0 - cl.h0 @ cl.e[0]) > 1e-3)
    # A3 components
    for Nf in (0, 2, 1):
        bf = BlockFock(bl, Nf)
        comm = 0.0
        for r in bf.reps:
            n = r['n']
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        for l in range(n):
                            lhs = r['E'][i][j] @ r['E'][k][l] - r['E'][k][l] @ r['E'][i][j]
                            rhs = (r['E'][i][l] if j == k else 0.0) - (r['E'][k][j] if l == i else 0.0)
                            comm = max(comm, fnorm(lhs - rhs))
        report('A3 gl(n) commutation relations on the components (Nf=%d, dims %s)' % (Nf, bf.dims), comm, 1e-12)
        report('A3 free state is a basis vector of the component (Nf=%d)' % Nf, 1.0 - min(r['free_is_basis'] for r in bf.reps), 1e-12)
        expected = {0: 4356, 2: 4356, 1: 5808}[Nf]
        report('A3 block dimension %d (Nf=%d)' % (expected, Nf), abs(bf.dim - expected), 0.5)
    # quasi-spin against the explicit construction at multiplicity 3
    mult = 3
    fs = FermiSpace(2 * mult, mult)
    E = [[sum(fs.hop(mult * i + c, mult * j + c) for c in range(mult)) for j in range(2)] for i in range(2)]
    v = fs.det_vector(list(range(mult)))
    basis = [v]
    for _ in range(mult):
        v = E[1][0] @ v
        v = v / np.linalg.norm(v)
        basis.append(v)
    Wq = np.stack(basis, axis=1)
    Eq = quasispin_rep(mult)
    report('A3 quasi-spin generators vs explicit Fock construction (mult 3)', max(fnorm(Wq.T @ E[i][j] @ Wq - Eq[i][j]) for i in range(2) for j in range(2)), 1e-12)
    # Wick identity on an explicit 6-mode space
    rng = np.random.default_rng(3)
    fs6 = FermiSpace(6, 3)
    A = herm(rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)))
    B = herm(rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)))
    P = np.diag([1.0, 1.0, 0, 0, 0, 0]).astype(complex)
    I6 = np.eye(6)
    dA, dB = fs6.dG(A), fs6.dG(B)
    W = (dA @ dB - trr(P @ A) * dB - trr(P @ B) * dA - fs6.dG(A @ (I6 - P) @ B) + fs6.dG(B @ P @ A)
         + (trr(P @ A) * trr(P @ B) - trr(P @ B @ P @ A)) * np.eye(fs6.dim))
    phi = fs6.det_vector([0, 2, 4])
    C = np.diag([1.0, 0, 1.0, 0, 1.0, 0]).astype(complex)
    D = C - P
    report('A3 Wick expectation <W_P(A,B)> = Tr(DA)Tr(DB) - Tr(DADB)', abs(np.vdot(phi, W @ phi) - (trr(D @ A) * trr(D @ B) - trr(D @ A @ D @ B))), 1e-12)
    # A4 raw one-body form equals the HF functional on random Gaussians
    G = args.G
    V = -G * K_P03
    Y, Es = wick_onebody(cl, V)
    Xr = herm(rng.normal(size=(cl.d, cl.d)) + 1j * rng.normal(size=(cl.d, cl.d)))
    w, U = np.linalg.eigh(Xr)
    Cr = U[:, :cl.N0 + 2] @ U[:, :cl.N0 + 2].conj().T
    dir_, exc_, _ = gauss_pair(cl, V, Cr)
    raw = E_raw_gauss(cl, V, Y, Es, Cr)
    hf = trr(cl.h0 @ (Cr - cl.P0)) + dir_ + exc_
    report('A4 raw form equals HF functional on a random projector', abs(raw - hf) / max(abs(hf), 1e-300), 1e-12)
    e_types = [bl.reduce(X)[0] for X in cl.e]
    cache = (e_types, bl.reduce(cl.h0)[0], bl.reduce(cl.P0)[0])
    sec2 = run_sector(cl, bl, 2, G, K_P03, cache, frozen_alt=False)
    sec0 = run_sector(cl, bl, 0, G, K_P03, cache, frozen_alt=False)
    report('A4 <free|H|free> equals its Gaussian value (Nf=2)', abs(sec2['free_expectation_block'] - sec2['free_expectation_gauss']), 1e-13)
    report('A4 <ref|H|ref> = 0 (Nf=0)', abs(sec0['free_expectation_block']), 1e-13)
    report('A4 H0 diagonal in the tensor basis', max(sec2['H0_offdiag'], sec0['H0_offdiag']), 1e-12)
    report('A4 c1 = -m^2 q^T K q = -0.021551020408 (Nf=2)', abs(sec2['c1'] + 26.4 * m * m / 49.0), 1e-12)
    for lab in ('HF', 'Hartree'):
        x = sec2['%s_in_block' % lab]
        report('A5 %s determinant in the block equals the one-body functional (Nf=2)' % lab, abs(x['diff']) if x else 1.0, 1e-12)
    report('A6 SCF residuals (H, HF; Nf=2)', max(sec2['Hartree']['residual'], sec2['HF']['residual']), 1e-12)
    report('A6 idempotency (H, HF; Nf=2)', max(sec2['Hartree']['idempotency'], sec2['HF']['idempotency']), 1e-12)
    report('A6 E_exact <= E_HF (Nf=2)', sec2['E_exact'] - sec2['HF']['E'], 1e-14, ok=sec2['E_exact'] <= sec2['HF']['E'] + 1e-14)
    report('A6 exact eigen-residual (Nf=2)', sec2['exact_residual'], 1e-12)
    report('A7 gap certificate 2G||H1|| < free gap (Nf=2)', 2 * G * sec2['H1_norm_bound'] / sec2['free_gap_sector'], 1.0)
    report('A8 Richardson c2 vs perturbative c2 (Nf=2)', abs(sec2['c2_richardson'] - sec2['c2']) / abs(sec2['c2']), 1e-5)
    report('A8 Richardson c2 vs perturbative c2 (Nf=0)', abs(sec0['c2_richardson'] - sec0['c2']) / abs(sec0['c2']), 1e-5)
    # A9 seven-site analytic control at G = 1/16
    V7 = -(1.0 / 16.0) * K_7
    vals = {}
    for Nf, tgt in ((1, (0.19875, 0.2)), (2, (0.395, 0.3975))):
        _, oH = scf(c7, V7, c7.N0 + Nf, 'H')
        _, oF = scf(c7, V7, c7.N0 + Nf, 'HF')
        vals[Nf] = (oH['E'], oF['E'])
        report('A9 7-site G=1/16 Nf=%d Hartree %.6g, HF %.6g (analytic)' % (Nf, tgt[0], tgt[1]), max(abs(oH['E'] - tgt[0]), abs(oF['E'] - tgt[1])), 1e-10)
    log('SELFTEST %s (%d failures)' % ('PASS' if fails == 0 else 'FAIL', fails))
    return 0 if fails == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ma', type=float, default=0.2)
    ap.add_argument('--G', type=float, default=1.0 / 4096)
    ap.add_argument('--Nf', default='0,2,1')
    ap.add_argument('--out', default='cluster13_results.json')
    ap.add_argument('--no-frozen', action='store_true', help='skip the frozen-sea alternative')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--threads', type=int, default=None)
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest(args))
    run_main(args)


if __name__ == '__main__':
    main()
