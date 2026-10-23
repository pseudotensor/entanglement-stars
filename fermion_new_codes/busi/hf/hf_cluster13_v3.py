#!/usr/bin/env python3
"""
hf_cluster13_v3.py  (busi lane, task 4 = S10; version 3; versions 1 and 2 are kept as hf_cluster13.py and
hf_cluster13_v2.py)

The 13-site axial-cluster benchmark of the gravitational Hartree, Hartree-Fock and exact levels with the scalar
monopole kernel and the fixed free reference (S10's targets; the P03 web return's section 12.2 and P18's DERIVATION.md
are idea files, re-derived here, never sources), the 7-site control (blind to the radial current), the second- and
third-order coefficients, the connected reference energy and the interaction cumulant, with a self-test mode.

Units a = 1. Conventions (DECLARED; as in version 2):
  sites {0, +-e_i, +-2e_i} (13 sites, 52 modes; --cluster cross7: {0, +-e_i}, 7 sites, 28 modes)
  h0 = m beta + sum_i alpha_i p_i, p_i = (T_i - T_i^T)/(2i), (T_i u)(x) = u(x + e_i)   (sine hopping)
  beta = sigma_z x I_2, alpha_i = sigma_x x sigma_i
  shells r = |x| in {0, 1, 2}; Pi_r the shell projectors; e_r = {Pi_r, h0}/2 the shell energy vertices
  reference P0 = 1_{h0 < 0} (fixed, free; N0 = 26)
  pair kernel V = -G K, K = [[2.4, 1, 1/2], [1, 1, 1/2], [1/2, 1/2, 1/2]]
  E^ = dG(h0) - Tr(P0 h0) + (1/2) sum_rs V_rs W_{P0}(e_r, e_s), W_P the fully Wick-ordered pair:
     W_P(A,B) = dG(A)dG(B) - Tr(PA)dG(B) - Tr(PB)dG(A) - dG(A(1-P)B) + dG(BPA) + Tr(PA)Tr(PB) - Tr(PBPA)
     <W_P(A,B)>_C = Tr(DA)Tr(DB) - Tr(DADB), D = C - P, for Gaussian C
  => E^ = E_s + dG(Y) + (1/2) sum_rs V_rs dG(e_r)dG(e_s),
     Y = h0 - sum_rs V_rs Tr(P e_r) e_s - (1/2) sum_rs V_rs [e_r(1-P)e_s - e_s P e_r],
     E_s = -Tr(P h0) + (1/2) sum_rs V_rs [Tr(P e_r)Tr(P e_s) - Tr(P e_s P e_r)]
  Hartree: E_H[C] = Tr(h0 D) + (1/2) q^T V q, q_r = Tr(D e_r), C the self-consistent lowest-(N0+Nf) projector of
           h_H = h0 + sum_rs V_rs q_s e_r (aufbau; the sea polarizes; --frozen-sea restricts D to the positive band)
  HF:      E_HF = E_H - (1/2) sum_rs V_rs Tr(D e_r D e_s), h_HF = h_H - sum_rs V_rs e_r D e_s; the HF projector is
           the stationary rank-N projector reached by the maximum-overlap rule from the Hartree state.
  exact:   the (N0 + Nf)-particle sector of E^ in the multiplicity-symmetric block (dimension 4356 / 5808).
  bindings: B_H,total = Nf m - E_H, B_HF = Nf m - E_HF, B_exact,sub = Nf m - [E_Q(Nf) - E_Q(0)] (the S10 targets'
           denominator is B_H,total; the optimized and frozen pair bindings are printed beside it).

Changes against version 2 (judged in busi/notes/HF_judgment_v3.md):
  1. Conditioned bindings. E_H - Nf m and E_HF - Nf m are evaluated without the cancellation of the sea energy
     (Tr(h0 C) ~ -16.1 against Tr(h0 P0) = -16.52; their roundoff of about 1e-14 is 1e-9 of the bindings and set the
     floor of the ratio targets). In the h0 eigenbasis
        Tr(h0 (C - P0)) - Nf m = sum_{a: eps_a > 0} (eps_a - m) <a|C|a> + sum_{i: eps_i < 0} (m - eps_i) <i|1 - C|i>,
     with <i|1 - C|i> from the unoccupied orbitals of C (small squares). The direct route is printed beside it and
     their difference is the measured roundoff of the direct route.
  2. Partitioned shifts. E_exact - E_free = G c1 + G^2 f(G) from the Loewdin partition, free of the construction
     roundoff of the block diagonal (H0[free,free] - Nf m, about 1e-14, printed as the residue; version 2's raw exact
     energies carried it).
  3. The interaction cumulant E_Q - E_HF(C_Q): the one-body covariance C_Q of the exact block state (per type and copy;
     the explicit b components carry per-copy hop operators so that the Nf = 1 doublet component is treated exactly)
     and the Gaussian energy of E^ at C_Q; printed with the ground shift E_Q - E_free and their ratio (2 at second
     order) and, for Nf > 0, with E_Q - E_HF.
  4. The three denominators (total optimized Hartree binding; optimized pair binding -EH at the Hartree state; frozen
     pair binding at the free state) and the ratios on each.
  5. The ratio acceptance A10 at RATIO_TOL = 3e-9, the propagated floor of the direct route (version 2's 1e-9 lay
     below it); the conditioned-route ratios are printed with the route difference; the energies are compared with
     P03's intermediate values (information).
  6. The level-rank split of c2 (keys c2_singles, c2_doubles) is labelled: it equals the excitation-rank split for the
     closed shells Nf = 0 and 2 only (at Nf = 1 the level pattern (1,1,1) of the b+ type contains a double excitation
     of the free determinant).
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

VERSION = 'hf_cluster13_v3'
RATIO_TOL = 3e-9      # propagated double-precision floor of the ratio targets (HF_judgment_v3.md, section 1.1)
RATIO_TOL_V2 = 1e-9   # the version-2 tolerance, printed for the record
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
    """(1/2) sum_rs V_rs [Tr(De_r)Tr(De_s) - Tr(De_rDe_s)] (direct plus exchange), and its two parts; D = C - P0,
    valid for any Gaussian covariance C (Wick's theorem does not need idempotency)."""
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


def kinetic_excess(cl, C):
    """Tr(h0 (C - P0)) - Nf m without the cancellation of the sea energy (version 3, change 1). Because [h0, P0] = 0,
    Tr(h0 (C - P0)) = Tr(h0 (1-P0) C (1-P0)) - Tr(h0 P0 (1-C) P0) and Nf = Tr(C - P0) splits the same way, so
      Tr(h0 (C - P0)) - Nf m = sum_{a: eps_a > 0} (eps_a - m) <a|C|a> + sum_{i: eps_i < 0} (m - eps_i) <i|1 - C|i>,
    both parts small and positive for the states of this control. For an idempotent C the weights of its eigenvectors
    are taken as exact 0/1 (the unoccupied orbitals give <i|1-C|i> as small squares); for a general 0 <= C <= 1 the
    eigenvalues w_k are used. Returns (excess, particle part, hole part, idempotency defect max |w(1-w)|)."""
    m = cl.m
    w, Uc = np.linalg.eigh(herm(C))
    idem = float(np.max(np.abs(w * (1.0 - w)))) if len(w) else 0.0
    wocc = (w > 0.5).astype(float) if idem < 1e-12 else np.clip(w, 0.0, 1.0)
    A2 = np.abs(cl.U0.conj().T @ Uc) ** 2           # |<l|u_k>|^2 over the h0 eigenstates l and the eigenvectors of C
    pos, neg = cl.eps0 > 0, cl.eps0 < 0
    part = float(np.sum((cl.eps0[pos] - m)[:, None] * A2[pos] * wocc[None, :]))
    hole = float(np.sum((m - cl.eps0[neg])[:, None] * A2[neg] * (1.0 - wocc)[None, :]))
    return part + hole, part, hole, idem


# ============================================================================ self-consistent fields
def select_occupied(eps, U, N, Cprev, rule='aufbau', tol=1e-9):
    """Occupied orbitals from the eigenpairs (eps ascending, U) of the generator.
    'aufbau': the lowest N; inside a degenerate cluster cut by the Fermi level the combinations of maximal weight in
    the previous state. 'mom' (maximum overlap; the HF rule): every degenerate cluster is first rotated to diagonalize
    the previous state, then the N orbitals of largest overlap with it are taken (ties by energy). Returns the orbitals
    and the aufbau gap min(eps empty) - max(eps occupied) (negative: the state is not the aufbau state)."""
    d = len(eps)
    if N <= 0:
        return U[:, :0].astype(complex), dict(aufbau_gap=float(eps[0]) if d else float('nan'), is_aufbau=True,
                                               homo=None, lumo=float(eps[0]) if d else None)
    Ur = U.astype(complex).copy()
    ov = np.zeros(d)
    i = 0
    while i < d:
        j = i + 1
        while j < d and eps[j] - eps[j - 1] < tol:
            j += 1
        Ug = Ur[:, i:j]
        if j - i > 1:
            w, R = np.linalg.eigh(herm(Ug.conj().T @ Cprev @ Ug))
            Ur[:, i:j] = Ug @ R[:, ::-1]
            ov[i:j] = np.clip(w[::-1], 0.0, 1.0)
        else:
            ov[i] = float(np.clip(np.real(np.vdot(Ug[:, 0], Cprev @ Ug[:, 0])), 0.0, 1.0))
        i = j
    if rule == 'aufbau':
        sel = np.arange(N)
    elif rule == 'mom':
        sel = np.sort(np.lexsort((eps, -ov))[:N])
    else:
        raise ValueError('unknown occupation rule %s' % rule)
    unsel = np.setdiff1d(np.arange(d), sel)
    homo = float(np.max(eps[sel]))
    lumo = float(np.min(eps[unsel])) if len(unsel) else float('nan')
    return Ur[:, sel], dict(aufbau_gap=lumo - homo, is_aufbau=bool(np.array_equal(sel, np.arange(N))), homo=homo, lumo=lumo)


def generator(cl, V, C, mode):
    """h_H[C] = h0 + sum_rs V_rs q_s e_r (q_r = Tr(D e_r), D = C - P0); h_HF[C] = h_H[C] - sum_rs V_rs e_r D e_s."""
    D = C - cl.P0
    q = np.array([trr(D @ A) for A in cl.e])
    u = V @ q
    h = cl.h0 + sum(u[r] * cl.e[r] for r in range(cl.nsh))
    if mode == 'HF':
        h = h - sum(V[r, s] * (cl.e[r] @ D @ cl.e[s]) for r in range(cl.nsh) for s in range(cl.nsh))
    return herm(h), q


def scf(cl, V, N, mode, frozen_sea=False, rule=None, tol=1e-13, maxit=3000, damp=0.5):
    """Hartree ('H') or Hartree-Fock ('HF') self-consistent rank-N projector; rule 'aufbau' (default for H) or 'mom'
    (default for HF, see select_occupied); with frozen_sea the matter part D = C - P0 is restricted to the positive
    band of h0 (labelled alternative). Returns C and a record with the convergence, idempotency, stationarity
    ||[h[C], C]||_F, the aufbau gap of the final state, and (version 3) the energy relative to the free energy by the
    direct and by the conditioned route."""
    rule = rule or ('mom' if mode == 'HF' else 'aufbau')
    P0 = cl.P0
    F0 = np.eye(cl.d) - P0
    Nf = N - cl.N0
    C = cl.U0[:, :N] @ cl.U0[:, :N].conj().T
    res = np.inf
    it = 0
    sel = dict(aufbau_gap=float('nan'), is_aufbau=True, homo=None, lumo=None)
    for it in range(maxit):
        h, _ = generator(cl, V, C, mode)
        if frozen_sea:
            hp = herm(F0 @ h @ F0)
            eps, U = np.linalg.eigh(hp)
            pos = np.where(np.abs(np.einsum('ia,ij,ja->a', U.conj(), F0, U)).real > 0.5)[0]
            Uo, sel = select_occupied(eps[pos], U[:, pos], Nf, C - P0, rule)
            Cn = P0 + Uo @ Uo.conj().T
        else:
            eps, U = np.linalg.eigh(h)
            Uo, sel = select_occupied(eps, U, N, C, rule)
            Cn = Uo @ Uo.conj().T
        res = fnorm(Cn - C)
        C = Cn if res < 1e-7 else (1.0 - damp) * C + damp * Cn
        if res < tol:
            break
    h, _ = generator(cl, V, C, mode)
    hs = herm(F0 @ h @ F0) if frozen_sea else h
    D = C - P0
    direct, exch, q = gauss_pair(cl, V, C)
    E0 = trr(cl.h0 @ D)
    kex, kpart, khole, _ = kinetic_excess(cl, C)
    xch = exch if mode == 'HF' else 0.0
    E_dir = E0 + direct + xch
    E_cond = kex + direct + xch
    out = dict(mode=mode, rule=rule, frozen_sea=frozen_sea, N=N, Nf=Nf, E0=E0, EH=direct, EF=exch, q=q.tolist(),
               E=E_dir, E_HF_functional=E0 + direct + exch,
               E_minus_free_direct=E_dir - Nf * cl.m, E_minus_free_conditioned=E_cond,
               roundoff_direct_route=(E_dir - Nf * cl.m) - E_cond,
               E_HF_functional_minus_free_conditioned=kex + direct + exch,
               kinetic_excess=kex, kinetic_particle_part=kpart, kinetic_hole_part=khole,
               iterations=it + 1, residual=res, idempotency=fnorm(C @ C - C), converged=bool(res < tol),
               stationarity=fnorm(hs @ C - C @ hs), aufbau_gap=sel['aufbau_gap'], is_aufbau=sel['is_aufbau'],
               homo=sel['homo'], lumo=sel['lumo'])
    return C, out


def orbital_hessian(cl, V, C, N, mode, nfd=3, eps_fd=1e-3, seed=11):
    """Second variation of E_H (mode 'H') or E_HF (mode 'HF') on the Grassmannian of rank-N projectors at C, in the
    particle-hole chart C(z) = e^K C e^-K, K = vir z occ^+ - occ z^+ vir^+ (z: n_vir x n_occ complex):
    E(C(z)) = E(C) + <grad, z> + (1/2) zeta^T M zeta + O(z^3), zeta = (Re z, Im z),
    M = [[A_R + 2 B_R, -(A_I + 2 B_I)], [(.)^T, A_R - 2 B_R]] with the stability matrices (derived in HF_judgment_v2.md)
      A_{ai,bj} = 2[d_ij F_ab - d_ab F_ji] + 2 sum_rs V_rs [<a|e_r|i><j|e_s|b> - <a|e_r|b><j|e_s|i>]   (exchange part HF only)
      B_{ai,bj} = sum_rs V_rs [<i|e_r|a><j|e_s|b> - <i|e_r|b><j|e_s|a>]                                 (second part HF only)
    F = h_H[C] or h_HF[C]. The minimal eigenvalue of M decides the local minimum (0 for flat directions); a central
    second difference of the energy along random directions guards the assembly."""
    w, Uc = np.linalg.eigh(C)
    occ = Uc[:, w > 0.5]
    vir = Uc[:, w < 0.5]
    no, nv = occ.shape[1], vir.shape[1]
    F, _ = generator(cl, V, C, mode)
    Fvv = vir.conj().T @ F @ vir
    Foo = occ.conj().T @ F @ occ
    Eov = [occ.conj().T @ A @ vir for A in cl.e]
    Evv = [vir.conj().T @ A @ vir for A in cl.e]
    Eoo = [occ.conj().T @ A @ occ for A in cl.e]
    Inv, Ino = np.eye(nv), np.eye(no)
    A = 2.0 * (np.einsum('ab,ij->aibj', Fvv, Ino) - np.einsum('ab,ji->aibj', Inv, Foo))
    B = np.zeros((nv, no, nv, no), complex)
    for r in range(cl.nsh):
        for s in range(cl.nsh):
            if V[r, s] == 0.0:
                continue
            Evo_r = Eov[r].conj().T
            A = A + 2.0 * V[r, s] * np.einsum('ai,jb->aibj', Evo_r, Eov[s])
            B = B + V[r, s] * np.einsum('ia,jb->aibj', Eov[r], Eov[s])
            if mode == 'HF':
                A = A - 2.0 * V[r, s] * np.einsum('ab,ji->aibj', Evv[r], Eoo[s])
                B = B - V[r, s] * np.einsum('ib,ja->aibj', Eov[r], Eov[s])
    n_p = nv * no
    A = A.reshape(n_p, n_p)
    B = B.reshape(n_p, n_p)
    X = -(A.imag + 2.0 * B.imag)
    M = np.block([[A.real + 2.0 * B.real, X], [X.T, A.real - 2.0 * B.real]])
    M = 0.5 * (M + M.T)
    ev = np.linalg.eigvalsh(M)
    scale = max(float(ev[-1]), 1e-300)

    def energy(Cx):
        dr_, ex_, _ = gauss_pair(cl, V, Cx)
        return trr(cl.h0 @ (Cx - cl.P0)) + dr_ + (ex_ if mode == 'HF' else 0.0)

    rng = np.random.default_rng(seed)
    E0 = energy(C)
    fd_err = 0.0
    for _ in range(nfd):
        z = rng.normal(size=(nv, no)) + 1j * rng.normal(size=(nv, no))
        z /= np.linalg.norm(z)
        zeta = np.concatenate([z.real.ravel(), z.imag.ravel()])
        quad = float(zeta @ M @ zeta)
        Kop = vir @ z @ occ.conj().T
        Kop = Kop - Kop.conj().T
        Up = sla.expm(eps_fd * Kop)
        Um = sla.expm(-eps_fd * Kop)
        fd = (energy(Up @ C @ Up.conj().T) + energy(Um @ C @ Um.conj().T) - 2.0 * E0) / eps_fd ** 2
        fd_err = max(fd_err, abs(fd - quad) / max(abs(quad), 1e-300))
    return dict(min_eig=float(ev[0]), max_eig=float(ev[-1]), min_eig_rel=float(ev[0]) / scale, dim=2 * n_p,
                n_zero_modes=int(np.sum(np.abs(ev) < 1e-9 * scale)), fd_check=fd_err,
                A_hermiticity=fnorm(A - A.conj().T) / max(fnorm(A), 1e-300), B_symmetry=fnorm(B - B.T) / max(fnorm(B), 1e-300))


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
    """Types ('b', s): the Krylov spaces of h0 from the centre spinors of beta = s (the first Krylov vector is the
    spin-orbit combination sum_i (delta_{-e_i} - delta_{e_i}) x alpha_i chi on shell 1, the second the symmetric shell-2
    combination with chi); ('d', b): the Krylov spaces from the shell-1 states orthogonal to the b sector with beta = b.
    Every type is rotated to the h0 eigenbasis of its levels (ascending); the copies of a type carry identical matrices
    for every block-scalar operator."""

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
    """The multiplicity-symmetric component of the (N0 + Nf)-particle sector: a tensor product over the types.
    Version 3 keeps, for the explicit b components, the per-copy hop operators c^+_{ic} c_{jc} projected to the
    component (for the one-body covariance of the exact state)."""

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
                hop_copy = [[[W.T @ fs.hop(mult * i + c, mult * j + c) @ W for j in range(n)] for i in range(n)] for c in range(mult)]
                free_modes = [mult * l + c for l in range(k) for c in range(mult)] + ([mult * k] if rem else [])
                free = W.T @ fs.det_vector(free_modes)
            else:
                assert nsea == 1 and n == 2, 'd types must be two-level'
                E = quasispin_rep(mult)
                hop_copy = None
                free = np.eye(mult + 1)[0].copy()
                N = mult
            Hk = sum(eps[l] * E[l][l] for l in range(n))
            w, R = np.linalg.eigh(0.5 * (Hk + Hk.T))
            E = [[R.T @ E[i][j] @ R for j in range(n)] for i in range(n)]
            if hop_copy is not None:
                hop_copy = [[[R.T @ X @ R for X in row] for row in hc] for hc in hop_copy]
            free = R.T @ free
            weights = np.array([[E[l][l][a, a] for l in range(n)] for a in range(len(w))])
            i0 = int(np.argmax(np.abs(free)))
            rank = np.rint(np.sum(np.abs(weights - weights[i0][None, :]), axis=1) / 2.0).astype(int)
            self.reps.append(dict(name=t['name'], E=E, dim=len(w), free=free, n=n, mult=mult, N=N, H0=w,
                                  weights=weights, rank=rank, free_is_basis=float(abs(free[i0])), i0=i0, hop_copy=hop_copy))
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
        """Kronecker sum of the per-type one-body energies (the sea included; the scalar E_s is not part of it)."""
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

    def onebody_covariance(self, blocks, psi):
        """One-body covariance C_Q of a block vector on the d modes (C_qp = <c^+_p c_q>, so that <dG(A)> = Tr(A C_Q)).
        Per type the reduced density matrix M of the factor; for the explicit b components the per-copy level densities
        rho_c[i,j] = <c^+_{ic} c_{jc}> (the Nf = 1 doublet component is not copy-symmetric), for the symmetric d
        multiplets rho_c = <E_ij>/mult exactly. The copy assignment between the Fock construction and Blocks is a
        convention under which the Gaussian energy is invariant (E^ commutes with the copy U(2)). Returns C_Q and the
        largest deviation between copies of the b types (zero for the singlet components to roundoff)."""
        T = psi.reshape(self.dims)
        d = blocks.Wall.shape[0]
        C = np.zeros((d, d), complex)
        asym = 0.0
        for k, (t, r) in enumerate(zip(blocks.types, self.reps)):
            pk = np.moveaxis(T, k, 0).reshape(self.dims[k], -1)
            M = pk.conj() @ pk.T
            n = r['n']
            rho_avg = np.array([[np.sum(r['E'][i][j] * M) for j in range(n)] for i in range(n)]) / r['mult']
            for c, Vc in enumerate(t['copies']):
                if r['hop_copy'] is not None:
                    rho = np.array([[np.sum(r['hop_copy'][c][i][j] * M) for j in range(n)] for i in range(n)])
                    asym = max(asym, fnorm(rho - rho_avg))
                else:
                    rho = rho_avg
                C += Vc @ rho.T @ Vc.conj().T
        return herm(C), float(asym)

    def closed_shell_vector(self, blocks, C):
        """The Slater determinant of a block-scalar projector C (closed shells in every type), via Gamma(u) =
        exp(dGamma(log u)) applied to the free vector type by type; None if C is not block-scalar or not closed."""
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


def partitioned_energy(H0, H1, i0, g, E_start, niter=8, tol=1e-18):
    """Loewdin partitioning of H(g) = H0 + g H1 about the free basis state i0: the lowest eigenvalue obeys
    E = H_00 + g^2 f(E), f(E) = u^T (E - H_QQ)^{-1} u with u = H1[Q, i0], Q the complement of i0 (exact Schur-complement
    condition). H_QQ - E is positive definite for the ground state (interlacing), so f is evaluated by a Cholesky
    solve; E - H_00 is obtained directly, without cancellation against the free energy. Returns E, f = (E - H_00)/g^2,
    the last change of the fixed-point iteration and the iteration count."""
    dim = H0.shape[0]
    idx = np.delete(np.arange(dim), i0)
    HQQ = H0[np.ix_(idx, idx)] + g * H1[np.ix_(idx, idx)]
    u0 = H1[idx, i0]
    H00 = float(H0[i0, i0] + g * H1[i0, i0])
    E = float(E_start)
    f = 0.0
    dlast = np.inf
    it = 0
    for it in range(niter):
        A = HQQ.copy()
        A[np.diag_indices_from(A)] -= E
        try:
            cf = sla.cho_factor(A, lower=True, check_finite=False)
            x = sla.cho_solve(cf, u0, check_finite=False)
        except np.linalg.LinAlgError:
            x = sla.solve(A, u0, assume_a='sym', check_finite=False)
        f = -float(u0 @ x)
        E_new = H00 + g * g * f
        dlast = abs(E_new - E)
        E = E_new
        if dlast < tol:
            break
    return E, f, dlast, it + 1


# ============================================================================ sectors
def run_sector(cl, blocks, Nf, G, K, types_cache, frozen_alt=True, hf_rule=None):
    t0 = time.time()
    N = cl.N0 + Nf
    V = -G * K
    out = dict(Nf=Nf, N=N, G=G, closed_shell_free_state=bool(Nf != 1),
               c2_split_label='level-rank split (by the level-occupation pattern of the block basis); equals the excitation-rank '
                              'split for closed shells only (Nf = 0, 2), not at Nf = 1 (HF_judgment_v3.md, section 1.3)')
    # one-body levels
    CH, oH = scf(cl, V, N, 'H')
    CF, oF = scf(cl, V, N, 'HF', rule=hf_rule)
    oH['stability'] = orbital_hessian(cl, V, CH, N, 'H')
    oF['stability'] = orbital_hessian(cl, V, CF, N, 'HF')
    out['Hartree'], out['HF'] = oH, oF
    if frozen_alt:
        _, oHf = scf(cl, V, N, 'H', frozen_sea=True)
        _, oFf = scf(cl, V, N, 'HF', frozen_sea=True, rule=hf_rule)
        out['Hartree_frozen_sea'], out['HF_frozen_sea'] = oHf, oFf
    log('  Nf=%d: Hartree E=%.16f (E0 %.16f, EH %.10e, EF %.10e, it %d, res %.1e, converged %s, stationarity %.1e, Hessian min eig %.2e rel, FD %.1e)'
        % (Nf, oH['E'], oH['E0'], oH['EH'], oH['EF'], oH['iterations'], oH['residual'], oH['converged'], oH['stationarity'],
           oH['stability']['min_eig_rel'], oH['stability']['fd_check']))
    log('  Nf=%d: HF      E=%.16f (rule %s, it %d, res %.1e, converged %s, idempotency %.1e, stationarity %.1e, aufbau gap %+.3e, is_aufbau %s, '
        'Hessian min eig %.2e rel (%d zero modes), FD %.1e)'
        % (Nf, oF['E'], oF['rule'], oF['iterations'], oF['residual'], oF['converged'], oF['idempotency'], oF['stationarity'],
           oF['aufbau_gap'], oF['is_aufbau'], oF['stability']['min_eig_rel'], oF['stability']['n_zero_modes'], oF['stability']['fd_check']))
    log('  Nf=%d: conditioned route: E_H - Nf m = %.16e (direct %.16e, roundoff of the direct route %+.1e; kinetic excess %.6e = particle %.6e + hole %.6e); '
        'E_HF - Nf m = %.16e (direct %.16e, roundoff %+.1e)'
        % (Nf, oH['E_minus_free_conditioned'], oH['E_minus_free_direct'], oH['roundoff_direct_route'], oH['kinetic_excess'],
           oH['kinetic_particle_part'], oH['kinetic_hole_part'], oF['E_minus_free_conditioned'], oF['E_minus_free_direct'], oF['roundoff_direct_route']))
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
    H0diag = np.diag(H0).copy()
    out['E_s_free'] = Es0
    out['H0_offdiag'] = fnorm(H0 - np.diag(H0diag)) / max(fnorm(H0), 1e-300)
    out['H0_diag_vs_H0d_plus_Es'] = float(np.max(np.abs(H0diag - (H0d + Es0))))
    out['H0_offdiag_v1_formula'] = fnorm(H0 - np.diag(H0d)) / max(fnorm(H0), 1e-300)
    E_free = Nf * bf.emin
    out['E_free'] = E_free
    vfree = bf.free_vector()
    i0 = int(np.argmax(np.abs(vfree)))
    out['free_vector_weight'] = float(abs(vfree[i0]))
    out['E_free_from_H0'] = float(H0diag[i0])
    out['H0_free_residue'] = float(H0diag[i0]) - E_free   # the construction roundoff of the block diagonal (exact arithmetic: 0)
    # expectation checks at the free determinant
    Cfree = cl.U0[:, :N] @ cl.U0[:, :N].conj().T
    dir0, exc0, _ = gauss_pair(cl, V, Cfree)
    out['free_direct'] = dir0
    out['free_exchange'] = exc0
    out['free_expectation_block'] = float(vfree @ (H0 + G * H1) @ vfree)
    out['free_expectation_gauss'] = E_free + dir0 + exc0
    # perturbation theory: second order by level rank, third order
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
    w3 = np.zeros_like(amp)
    w3[mask] = amp[mask] / dE[mask]
    out['c3'] = float(w3 @ (H1 @ w3)) - out['c1'] * float(w3 @ w3)
    others = ~mask & (np.arange(bf.dim) != i0)
    out['degenerate_free_coupling'] = float(np.max(np.abs(amp[others]))) if np.any(others) else 0.0
    # exact at G and G/2 (eigh)
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
    out['c2_richardson_eigh'] = 2.0 * c2h - c2G
    out['c2_from_E_at_G_eigh'] = c2G
    # exact by Loewdin partitioning at G, G/2, G/4 (conditioned route to E - H_00)
    part = {}
    for g in (G, 0.5 * G, 0.25 * G):
        E_start = float(H0diag[i0]) + g * out['c1'] + g * g * out['c2']
        E_p, f_p, dlast, nit = partitioned_energy(H0, H1, i0, g, E_start)
        part[g] = dict(E=E_p, f=f_p, last_change=dlast, iterations=nit)
    fG, fh, fq = part[G]['f'], part[0.5 * G]['f'], part[0.25 * G]['f']
    out['partition'] = dict(E_G=part[G]['E'], E_half=part[0.5 * G]['E'], E_quarter=part[0.25 * G]['E'],
                            f_G=fG, f_half=fh, f_quarter=fq,
                            eigh_minus_partition_G=exact[G]['E'] - part[G]['E'],
                            eigh_minus_partition_half=exact[0.5 * G]['E'] - part[0.5 * G]['E'],
                            c2_R2=2.0 * fh - fG, c2_R3=(8.0 * fq - 6.0 * fh + fG) / 3.0,
                            c3_fd=2.0 * (fh - fq) / (0.25 * G) - (fG - fh) / (0.5 * G),
                            c4_est_G=(fG - out['c2'] - out['c3'] * G) / G ** 2,
                            c4_est_half=(fh - out['c2'] - out['c3'] * 0.5 * G) / (0.5 * G) ** 2,
                            fixed_point_last_change=max(p['last_change'] for p in part.values()),
                            fixed_point_iterations=max(p['iterations'] for p in part.values()))
    dEmax = max(abs(out['partition']['eigh_minus_partition_G']), abs(out['partition']['eigh_minus_partition_half']))
    out['c2_richardson_eigh_roundoff_floor_rel'] = 8.0 * dEmax / G ** 2 / max(abs(out['c2']), 1e-300)
    # the partitioned shift E_exact - E_free = G c1 + G^2 f(G), free of the construction residue of the block diagonal
    out['E_shift_partition'] = G * out['c1'] + G * G * fG
    out['E_exact_minus_free_eigh'] = out['E_exact'] - E_free
    out['shift_check'] = out['E_exact_minus_free_eigh'] - out['E_shift_partition'] - out['H0_free_residue']   # = eigh - partition
    psi = exact[G]['psi']
    rhos = bf.onebody_density(psi)
    out['natural_occupations'] = [sorted(np.linalg.eigvalsh(herm(r)).tolist(), reverse=True) for r in rhos]
    # the interaction cumulant: the exact state's one-body covariance and the Gaussian energy of E^ at it
    CQ, copy_asym = bf.onebody_covariance(blocks, psi)
    Yv, Esv = wick_onebody(cl, V)
    E_gauss_CQ = E_raw_gauss(cl, V, Yv, Esv, CQ)
    dirQ, excQ, qQ = gauss_pair(cl, V, CQ)
    kexQ, _, _, idemQ = kinetic_excess(cl, CQ)
    E_hf_form_CQ = trr(cl.h0 @ (CQ - cl.P0)) + dirQ + excQ
    E_hf_form_CQ_cond = E_free + kexQ + dirQ + excQ        # the conditioned kinetic route at a non-idempotent C (floor about 1e-15)
    wCQ = np.linalg.eigvalsh(CQ)
    out['CQ'] = dict(trace=trr(CQ), N=N, eig_min=float(wCQ[0]), eig_max=float(wCQ[-1]), nonidempotency=trr(CQ @ (np.eye(cl.d) - CQ)),
                     copy_asymmetry=copy_asym, q=qQ.tolist(), E_HF_of_CQ=E_gauss_CQ, E_HF_of_CQ_functional_form=E_hf_form_CQ,
                     E_HF_of_CQ_conditioned=E_hf_form_CQ_cond, form_diff=E_gauss_CQ - E_hf_form_CQ)
    E_cq = E_hf_form_CQ_cond
    out['ground_shift_eigh'] = out['E_exact_minus_free_eigh']
    out['ground_shift_partition'] = out['E_shift_partition']
    out['cumulant_eigh'] = out['E_exact'] - E_gauss_CQ
    out['cumulant_partition'] = (E_free + out['E_shift_partition']) - E_cq
    out['ratio_cumulant_over_shift_partition'] = (out['cumulant_partition'] / out['ground_shift_partition']
                                                  if abs(out['ground_shift_partition']) > 1e-300 else float('nan'))
    out['E_exact_minus_E_HF_eigh'] = out['E_exact'] - oF['E']
    out['E_exact_minus_E_HF_partition'] = out['E_shift_partition'] - oF['E_minus_free_conditioned']
    out['ratio_cumulant_over_corr_partition'] = (out['cumulant_partition'] / out['E_exact_minus_E_HF_partition']
                                                 if abs(out['E_exact_minus_E_HF_partition']) > 1e-300 else float('nan'))
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
    log('  Nf=%d: block %s dim %d  E_exact=%.16f (second %.12f, resid %.1e)  c1=%.14f  c2=%.14f (level-rank 1 %.14f, 2 %.14f, higher %.1e%s)  c3=%.10f  '
        'bound 2G||H1||=%.3e < gap %.5f: %s  (%.0f s)'
        % (Nf, bf.dims, bf.dim, out['E_exact'], out['E_exact_second'], out['exact_residual'], out['c1'], out['c2'],
           out['c2_singles'], out['c2_doubles'], out['c2_higher'], '' if Nf != 1 else '; not the excitation-rank split at Nf = 1',
           out['c3'], 2.0 * G * bound, out['free_gap_sector'], out['certificate'], out['seconds']))
    p = out['partition']
    log('  Nf=%d: partitioned energies E(G) %.16f (eigh - partition %.1e at G, %.1e at G/2; fixed point %d it, last change %.1e)  '
        'c2 three-point %.14f (rel dev %.1e)  two-point %.14f  c3 finite-difference %.10f (rel dev %.1e)  c4 est %.4f / %.4f  '
        '[v1 eigh Richardson %.12f, roundoff floor %.1e rel]'
        % (Nf, p['E_G'], p['eigh_minus_partition_G'], p['eigh_minus_partition_half'], p['fixed_point_iterations'], p['fixed_point_last_change'],
           p['c2_R3'], abs(p['c2_R3'] - out['c2']) / max(abs(out['c2']), 1e-300), p['c2_R2'], p['c3_fd'],
           abs(p['c3_fd'] - out['c3']) / max(abs(out['c3']), 1e-300), p['c4_est_G'], p['c4_est_half'],
           out['c2_richardson_eigh'], out['c2_richardson_eigh_roundoff_floor_rel']))
    log('  Nf=%d: shift E_exact - E_free: eigh %.16e, partition G c1 + G^2 f = %.16e, block-diagonal residue H0[free,free] - Nf m = %+.3e, '
        'eigh - partition - residue = %+.1e'
        % (Nf, out['E_exact_minus_free_eigh'], out['E_shift_partition'], out['H0_free_residue'], out['shift_check']))
    cq = out['CQ']
    log('  Nf=%d: cumulant: Tr C_Q = %.12f (N %d), eig [%+.1e, 1%+.1e], Tr C_Q(1-C_Q) %.3e, copy asymmetry %.1e; E_HF(C_Q) - E_free = %.10e '
        '(raw/functional forms differ by %.1e); ground shift %.10e (partition); cumulant E_Q - E_HF(C_Q) = %.10e (partition; eigh %.10e); '
        'cumulant/shift = %.6f; E_Q - E_HF = %.6e, cumulant/(E_Q - E_HF) = %.6f'
        % (Nf, cq['trace'], cq['N'], cq['eig_min'], cq['eig_max'] - 1.0, cq['nonidempotency'], cq['copy_asymmetry'],
           cq['E_HF_of_CQ_conditioned'] - E_free, cq['form_diff'], out['ground_shift_partition'], out['cumulant_partition'], out['cumulant_eigh'],
           out['ratio_cumulant_over_shift_partition'], out['E_exact_minus_E_HF_partition'], out['ratio_cumulant_over_corr_partition']))
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
    log('%s: 13-site axial star: d=%d N0=%d  spectrum vs analytic max dev %.1e  blocks ortho %.1e complete %s'
        % (VERSION, cl.d, cl.N0, float(np.max(np.abs(cl.eps0 - np.array(cl.analytic_spectrum())))), blocks.ortho_err, blocks.complete))
    for t in blocks.types:
        log('  type %s: n=%d mult=%d levels %s  Lanczos alpha %s beta %s' % (t['name'], t['n'], t['mult'], np.round(t['eps'], 6),
                                                                        np.round(t['lanczos'][0], 10), np.round(t['lanczos'][1], 10)))
    e_types = [blocks.reduce(A)[0] for A in cl.e]
    h0_types, err_h0, _ = blocks.reduce(cl.h0)
    P_types, err_P, _ = blocks.reduce(cl.P0)
    err_e = max(blocks.reduce(A)[1] for A in cl.e)
    log('  block-scalar structure errors: h0 %.1e  e_r %.1e  P0 %.1e' % (err_h0, err_e, err_P))
    log('  shell radial-current norms: %s' % [fnorm(J) for J in cl.Jrad])
    results = dict(version=VERSION, settings=dict(ma=args.ma, G=args.G, cluster='star13', K=K.tolist(), hf_rule=args.hf_rule,
                                                  ratio_tol=RATIO_TOL, ratio_tol_v2=RATIO_TOL_V2),
                   types=[dict(name=t['name'], n=t['n'], mult=t['mult'], eps=t['eps'].tolist(), lanczos_alpha=t['lanczos'][0].tolist(),
                               lanczos_beta=t['lanczos'][1].tolist()) for t in blocks.types],
                   structure_errors=dict(h0=err_h0, e=err_e, P0=err_P, ortho=blocks.ortho_err),
                   radial_current_norms=[fnorm(J) for J in cl.Jrad], sectors={})
    cache = (e_types, h0_types, P_types)
    Nfs = [int(x) for x in args.Nf.split(',')]
    for Nf in Nfs:
        results['sectors'][Nf] = run_sector(cl, blocks, Nf, args.G, K, cache, frozen_alt=not args.no_frozen, hf_rule=args.hf_rule)
    sec = results['sectors']
    summary = {}
    if 0 in sec and 2 in sec:
        G, m = args.G, args.ma
        s0, s2 = sec[0], sec[2]
        E_ref = s0['E_exact']
        E_free2 = s2['E_free']
        B_H = E_free2 - s2['Hartree']['E']
        B_HF = E_free2 - s2['HF']['E']
        E_sub = s2['E_exact'] - E_ref
        B_Q = E_free2 - E_sub
        B_H_c = -s2['Hartree']['E_minus_free_conditioned']
        B_HF_c = -s2['HF']['E_minus_free_conditioned']
        B_Q_p = -(s2['E_shift_partition'] - s0['E_shift_partition'])
        B_pair_frozen = -s2['free_direct']
        B_pair_opt = -s2['Hartree']['EH']
        summary.update(
            E_ref_raw=E_ref, E_ref_shift_partition=s0['E_shift_partition'], H0_free_residue_ref=s0['H0_free_residue'],
            H0_free_residue_Nf2=s2['H0_free_residue'],
            E_H=s2['Hartree']['E'], E_HF=s2['HF']['E'], E_exact_raw=s2['E_exact'], E_exact_sub=E_sub,
            E_exact_sub_partition=E_free2 + s2['E_shift_partition'] - s0['E_shift_partition'],
            B_H=B_H, B_HF=B_HF, B_exact_sub=B_Q,
            B_H_conditioned=B_H_c, B_HF_conditioned=B_HF_c, B_exact_sub_partition=B_Q_p,
            B_H_pair_frozen=B_pair_frozen, B_H_pair_optimized=B_pair_opt,
            B_HF_over_B_H=B_HF / B_H, B_exact_sub_over_B_H=B_Q / B_H, B_exact_raw_over_B_H=(E_free2 - s2['E_exact']) / B_H,
            B_HF_over_B_H_conditioned=B_HF_c / B_H_c, B_exact_sub_over_B_H_partition=B_Q_p / B_H_c,
            route_difference_B_HF_over_B_H=B_HF / B_H - B_HF_c / B_H_c,
            route_difference_B_exact_sub_over_B_H=B_Q / B_H - B_Q_p / B_H_c,
            B_HF_over_B_H_pair_frozen=B_HF_c / B_pair_frozen, B_exact_sub_over_B_H_pair_frozen=B_Q_p / B_pair_frozen,
            B_HF_over_B_H_pair_optimized=B_HF_c / B_pair_opt, B_exact_sub_over_B_H_pair_optimized=B_Q_p / B_pair_opt,
            c2_ref=s0['c2'], c2_Nf2=s2['c2'], c2_Nf2_minus_ref=s2['c2'] - s0['c2'],
            c3_ref=s0['c3'], c3_Nf2=s2['c3'],
            c2_ref_R3_partition=s0['partition']['c2_R3'], c2_Nf2_R3_partition=s2['partition']['c2_R3'],
            c2_ref_richardson_eigh=s0['c2_richardson_eigh'], c2_Nf2_richardson_eigh=s2['c2_richardson_eigh'],
            eigh_minus_partition_max=max(abs(sec[n]['partition']['eigh_minus_partition_G']) for n in sec),
            HF_Nf2_converged=s2['HF']['converged'], HF_Nf2_stationarity=s2['HF']['stationarity'],
            HF_Nf2_hessian_min_eig_rel=s2['HF']['stability']['min_eig_rel'],
            B_HF_minus_half_B_H=B_HF_c - 0.5 * B_H_c,
            decomposition_B_HF_minus_half_B_H=-s2['c2_singles'] * G ** 2 - 0.5 * s2['Hartree']['kinetic_excess'],
            E_exact_sub_minus_E_HF=E_sub - s2['HF']['E'],
            E_exact_sub_minus_E_HF_partition=(s2['E_shift_partition'] - s0['E_shift_partition']) - s2['HF']['E_minus_free_conditioned'],
            decomposition_E_exact_sub_minus_E_HF=(s2['c2_doubles'] - s0['c2']) * G ** 2,
            E_ref_second_order=s0['c2'] * G ** 2,
            ground_shift_ref=s0['ground_shift_partition'], E_HF_of_CQ_ref_minus_free=s0['CQ']['E_HF_of_CQ_conditioned'] - s0['E_free'],
            cumulant_ref=s0['cumulant_partition'], ratio_cumulant_over_shift_ref=s0['ratio_cumulant_over_shift_partition'],
            cumulant_Nf2=s2['cumulant_partition'], E_exact_minus_E_HF_Nf2=s2['E_exact_minus_E_HF_partition'],
            ratio_cumulant_over_corr_Nf2=s2['ratio_cumulant_over_corr_partition'])
        if 1 in sec:
            s1 = sec[1]
            summary.update(E_Nf1_raw=s1['E_exact'], E_Nf1_sub=s1['E_exact'] - E_ref,
                           E_Nf1_sub_partition=s1['E_free'] + s1['E_shift_partition'] - s0['E_shift_partition'],
                           c2_Nf1=s1['c2'], c3_Nf1=s1['c3'],
                           E_Nf1_HF=s1['HF']['E'], E_Nf1_HF_minus_free_conditioned=s1['HF']['E_minus_free_conditioned'],
                           E_Nf1_HF_converged=s1['HF']['converged'],
                           E_Nf1_HF_stationarity=s1['HF']['stationarity'], E_Nf1_HF_is_aufbau=s1['HF']['is_aufbau'],
                           E_Nf1_HF_aufbau_gap=s1['HF']['aufbau_gap'], E_Nf1_H=s1['Hartree']['E'],
                           decomposition_E_Nf1_sub_minus_free=(s1['c2'] - s0['c2']) * G ** 2,
                           cumulant_Nf1=s1['cumulant_partition'], E_exact_minus_E_HF_Nf1=s1['E_exact_minus_E_HF_partition'],
                           ratio_cumulant_over_corr_Nf1=s1['ratio_cumulant_over_corr_partition'])
        ours_direct = dict(B_HF_over_B_H=summary['B_HF_over_B_H'], B_exact_sub_over_B_H=summary['B_exact_sub_over_B_H'],
                           E_Nf1_sub=summary.get('E_Nf1_sub'))
        ours_cond = dict(B_HF_over_B_H=summary['B_HF_over_B_H_conditioned'], B_exact_sub_over_B_H=summary['B_exact_sub_over_B_H_partition'],
                         E_Nf1_sub=summary.get('E_Nf1_sub_partition'))
        summary['targets_S10'] = {}
        for k, v in S10.items():
            od, oc = ours_direct.get(k), ours_cond.get(k)
            summary['targets_S10'][k] = dict(target=v, ours_direct=od, rel_dev_direct=None if od is None else (od - v) / v,
                                             ours_conditioned=oc, rel_dev_conditioned=None if oc is None else (oc - v) / v)
        summary['P03_intermediate'] = {k: dict(P03=v, ours=summary.get(k), diff=None if summary.get(k) is None else summary.get(k) - v)
                                       for k, v in P03.items() if k in summary}
        log('\nSUMMARY (%s, G/a^2 = %.10g, ma = %g):' % (VERSION, args.G, args.ma))
        for k in ('E_ref_raw', 'E_ref_shift_partition', 'H0_free_residue_ref', 'H0_free_residue_Nf2', 'E_H', 'E_HF', 'E_exact_raw',
                  'E_exact_sub', 'E_exact_sub_partition', 'B_H', 'B_H_conditioned', 'B_HF', 'B_HF_conditioned', 'B_exact_sub',
                  'B_exact_sub_partition', 'B_H_pair_frozen', 'B_H_pair_optimized',
                  'B_HF_over_B_H', 'B_HF_over_B_H_conditioned', 'route_difference_B_HF_over_B_H',
                  'B_exact_sub_over_B_H', 'B_exact_sub_over_B_H_partition', 'route_difference_B_exact_sub_over_B_H', 'B_exact_raw_over_B_H',
                  'B_HF_over_B_H_pair_frozen', 'B_exact_sub_over_B_H_pair_frozen', 'B_HF_over_B_H_pair_optimized', 'B_exact_sub_over_B_H_pair_optimized',
                  'E_Nf1_raw', 'E_Nf1_sub', 'E_Nf1_sub_partition', 'E_Nf1_HF', 'E_Nf1_HF_minus_free_conditioned', 'E_Nf1_H',
                  'c2_ref', 'c2_Nf2', 'c2_Nf2_minus_ref', 'c2_Nf1', 'c3_ref', 'c3_Nf2', 'c3_Nf1', 'c2_ref_R3_partition', 'c2_Nf2_R3_partition',
                  'eigh_minus_partition_max', 'B_HF_minus_half_B_H', 'decomposition_B_HF_minus_half_B_H',
                  'E_exact_sub_minus_E_HF', 'E_exact_sub_minus_E_HF_partition', 'decomposition_E_exact_sub_minus_E_HF',
                  'E_ref_second_order', 'decomposition_E_Nf1_sub_minus_free',
                  'ground_shift_ref', 'E_HF_of_CQ_ref_minus_free', 'cumulant_ref', 'ratio_cumulant_over_shift_ref',
                  'cumulant_Nf2', 'E_exact_minus_E_HF_Nf2', 'ratio_cumulant_over_corr_Nf2',
                  'cumulant_Nf1', 'E_exact_minus_E_HF_Nf1', 'ratio_cumulant_over_corr_Nf1'):
            if k in summary:
                log('  %-44s %.16g' % (k, summary[k]))
        log('  denominators: total optimized Hartree binding %.13e (the targets\'); optimized pair binding %.13e; frozen pair binding %.13e'
            % (B_H_c, B_pair_opt, B_pair_frozen))
        if 'E_Nf1_HF' in summary:
            log('  Nf=1 HF: converged %s, stationarity %.1e, is_aufbau %s, aufbau gap %+.3e'
                % (summary['E_Nf1_HF_converged'], summary['E_Nf1_HF_stationarity'], summary['E_Nf1_HF_is_aufbau'], summary['E_Nf1_HF_aufbau_gap']))
        for k, v in summary['P03_intermediate'].items():
            log('  P03 intermediate %-16s P03 %.16g  ours %.16g  diff %+.1e' % (k, v['P03'], v['ours'], v['diff']))
        for k, v in summary['targets_S10'].items():
            log('  TARGET %-22s S10 %.13g  direct %.13g (rel dev %s)  conditioned %.13g (rel dev %s)'
                % (k, v['target'], v['ours_direct'] if v['ours_direct'] is not None else float('nan'),
                   'n/a' if v['rel_dev_direct'] is None else '%.2e' % v['rel_dev_direct'],
                   v['ours_conditioned'] if v['ours_conditioned'] is not None else float('nan'),
                   'n/a' if v['rel_dev_conditioned'] is None else '%.2e' % v['rel_dev_conditioned']))
        acc = all(v['rel_dev_direct'] is not None and abs(v['rel_dev_direct']) < RATIO_TOL for v in summary['targets_S10'].values())
        acc_v2 = all(v['rel_dev_direct'] is not None and abs(v['rel_dev_direct']) < RATIO_TOL_V2 for v in summary['targets_S10'].values())
        summary['targets_pass'] = acc
        summary['targets_pass_v2_tolerance'] = acc_v2
        log('  S10 targets within %.0e relative (the propagated floor of the ratio form): %s   [within the version-2 tolerance %.0e: %s; information]'
            % (RATIO_TOL, 'PASS' if acc else 'FAIL', RATIO_TOL_V2, 'PASS' if acc_v2 else 'FAIL'))
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
        log('%s  %-78s %.3e (tol %.0e)' % ('PASS' if ok else 'FAIL', name, val, tol))

    log('%s self-test (HF rule %s)' % (VERSION, args.hf_rule))
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
        # the per-copy hop operators sum to the gl(n) generators
        hsum = 0.0
        for r in bf.reps:
            if r['hop_copy'] is not None:
                for i in range(r['n']):
                    for j in range(r['n']):
                        hsum = max(hsum, fnorm(sum(r['hop_copy'][c][i][j] for c in range(r['mult'])) - r['E'][i][j]))
        report('A3 per-copy hop operators sum to the gl(n) generators (Nf=%d)' % Nf, hsum, 1e-12)
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
    # A4 raw one-body form equals the HF functional on random Gaussians; the conditioned kinetic route
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
    kx, _, _, _ = kinetic_excess(cl, Cr)
    report('A4 conditioned kinetic excess equals Tr(h0 D) - Nf m on a random projector', abs(kx - (trr(cl.h0 @ (Cr - cl.P0)) - 2 * m)), 5e-14)
    Cg = U @ np.diag(rng.uniform(0.0, 1.0, cl.d)) @ U.conj().T
    kxg, _, _, idg = kinetic_excess(cl, Cg)
    report('A4 conditioned kinetic excess on a random Gaussian covariance (idempotency defect %.2f)' % idg,
           abs(kxg - (trr(cl.h0 @ (Cg - cl.P0)) - (trr(Cg) - cl.N0) * m)), 1e-12)
    rawg = E_raw_gauss(cl, V, Y, Es, Cg)
    dg_, eg_, _ = gauss_pair(cl, V, Cg)
    report('A4 raw form equals HF functional on a random Gaussian covariance', abs(rawg - (trr(cl.h0 @ (Cg - cl.P0)) + dg_ + eg_)) / max(abs(rawg), 1e-300), 1e-12)
    e_types = [bl.reduce(X)[0] for X in cl.e]
    cache = (e_types, bl.reduce(cl.h0)[0], bl.reduce(cl.P0)[0])
    secs = {}
    for Nf in (2, 0, 1):
        secs[Nf] = run_sector(cl, bl, Nf, G, K_P03, cache, frozen_alt=False, hf_rule=args.hf_rule)
    sec2, sec0 = secs[2], secs[0]
    report('A4 <free|H|free> equals its Gaussian value (Nf=2)', abs(sec2['free_expectation_block'] - sec2['free_expectation_gauss']), 1e-13)
    report('A4 <ref|H|ref> = 0 (Nf=0)', abs(sec0['free_expectation_block']), 1e-13)
    report('A4 H0 diagonal in the tensor basis (off-diagonal norm; Nf=0,1,2)', max(s['H0_offdiag'] for s in secs.values()), 1e-12)
    report('A4 diag(H0) = H0_diag + E_s (Nf=0,1,2)', max(s['H0_diag_vs_H0d_plus_Es'] for s in secs.values()), 1e-12)
    report('A4 H0[free,free] = E_free = Nf m (Nf=0,1,2; the block-diagonal residue)', max(abs(s['H0_free_residue']) for s in secs.values()), 1e-12)
    report('A4 conditioned kinetic excess equals Tr(h0 D) - Nf m on the SCF states (Nf=0,1,2)',
           max(abs(s[lab]['roundoff_direct_route']) for s in secs.values() for lab in ('Hartree', 'HF')), 5e-14)
    log('      (v1 formula ||H0 - diag(H0_diag)||/||H0|| = E_s/rms: %s)' % ', '.join('%.5f' % s['H0_offdiag_v1_formula'] for s in secs.values()))
    report('A4 c1 = -m^2 q^T K q = -0.021551020408 (Nf=2)', abs(sec2['c1'] + 26.4 * m * m / 49.0), 1e-12)
    report('A4 frozen pair binding at the free state = -2 c1 G (Nf=2)', abs(-sec2['free_direct'] + 2.0 * sec2['c1'] * G), 1e-18)
    for lab in ('HF', 'Hartree'):
        x = sec2['%s_in_block' % lab]
        report('A5 %s determinant in the block equals the one-body functional (Nf=2)' % lab, abs(x['diff']) if x else 1.0, 1e-12)
    for Nf, s in secs.items():
        for lab in ('Hartree', 'HF'):
            o = s[lab]
            report('A6 %s converged (Nf=%d, rule %s, %d it)' % (lab, Nf, o['rule'], o['iterations']), 0.0 if o['converged'] else 1.0, 0.5)
            report('A6 %s SCF residual (Nf=%d)' % (lab, Nf), o['residual'], 1e-12)
            report('A6 %s idempotency (Nf=%d)' % (lab, Nf), o['idempotency'], 1e-13)
            report('A6 %s stationarity ||[h[C], C]|| (Nf=%d)' % (lab, Nf), o['stationarity'], 1e-11)
            st = o['stability']
            report('A6 %s orbital Hessian has no negative eigenvalue (Nf=%d; min/max %.2e, %d zero modes)' % (lab, Nf, st['min_eig_rel'], st['n_zero_modes']),
                   -st['min_eig_rel'], 1e-10)
            report('A6 %s orbital Hessian vs finite-difference curvature (Nf=%d)' % (lab, Nf), st['fd_check'], 1e-5)
        report('A6 E_exact <= E_HF (Nf=%d)' % Nf, s['E_exact'] - s['HF']['E'], 1e-14, ok=s['E_exact'] <= s['HF']['E'] + 1e-14)
        report('A6 exact eigen-residual (Nf=%d)' % Nf, s['exact_residual'], 1e-12)
        if Nf == 1:
            log('      Nf=1 HF: aufbau gap %+.3e (negative: open shell, occupied orbital above an empty degenerate partner), is_aufbau %s'
                % (s['HF']['aufbau_gap'], s['HF']['is_aufbau']))
    report('A7 gap certificate 2G||H1|| < free gap (Nf=2)', 2 * G * sec2['H1_norm_bound'] / sec2['free_gap_sector'], 1.0)
    for Nf, s in secs.items():
        p = s['partition']
        report('A8 eigh energy vs Loewdin-partitioned energy at G and G/2 (Nf=%d)' % Nf,
               max(abs(p['eigh_minus_partition_G']), abs(p['eigh_minus_partition_half'])), 5e-13)
        report('A8 three-point Richardson c2 (partitioned f at G, G/2, G/4) vs perturbative c2 (Nf=%d)' % Nf,
               abs(p['c2_R3'] - s['c2']) / max(abs(s['c2']), 1e-300), 1e-7)
        if abs(s['c3']) > 1e-6:
            report('A8 finite-difference c3 vs third-order perturbative c3 (Nf=%d)' % Nf, abs(p['c3_fd'] - s['c3']) / abs(s['c3']), 1e-3)
        else:
            report('A8 finite-difference c3 vs third-order perturbative c3, absolute (Nf=%d)' % Nf, abs(p['c3_fd'] - s['c3']), 1e-9)
        log('      Nf=%d: v1 eigh-Richardson c2 %.12f vs perturbative %.12f (rel dev %.1e; propagated roundoff floor %.1e; not a test)'
            % (Nf, s['c2_richardson_eigh'], s['c2'], abs(s['c2_richardson_eigh'] - s['c2']) / abs(s['c2']), s['c2_richardson_eigh_roundoff_floor_rel']))
    # A9 seven-site analytic control at G = 1/16
    V7 = -(1.0 / 16.0) * K_7
    for Nf, tgt in ((1, (0.19875, 0.2)), (2, (0.395, 0.3975))):
        _, oH = scf(c7, V7, c7.N0 + Nf, 'H')
        _, oF = scf(c7, V7, c7.N0 + Nf, 'HF', rule=args.hf_rule)
        report('A9 7-site G=1/16 Nf=%d Hartree %.6g, HF %.6g (analytic)' % (Nf, tgt[0], tgt[1]), max(abs(oH['E'] - tgt[0]), abs(oF['E'] - tgt[1])), 1e-10)
        report('A9 7-site Nf=%d HF converged, idempotent and stationary' % Nf, max(0.0 if oF['converged'] else 1.0, oF['idempotency'], oF['stationarity']), 1e-11)
        report('A9 7-site Nf=%d conditioned route equals the direct route' % Nf, max(abs(oH['roundoff_direct_route']), abs(oF['roundoff_direct_route'])), 5e-14)
        log('      7-site Nf=%d: Hartree E %.12f (it %d)  HF E %.12f (rule %s, it %d, res %.1e, aufbau gap %+.5f, is_aufbau %s)'
            % (Nf, oH['E'], oH['iterations'], oF['E'], oF['rule'], oF['iterations'], oF['residual'], oF['aufbau_gap'], oF['is_aufbau']))
    # A11 the one-body covariance of the exact state and the cumulant
    for Nf, s in secs.items():
        cq = s['CQ']
        report('A11 Tr C_Q = N (Nf=%d)' % Nf, abs(cq['trace'] - cq['N']), 1e-10)
        report('A11 0 <= C_Q <= 1 (Nf=%d)' % Nf, max(-cq['eig_min'], cq['eig_max'] - 1.0), 1e-10)
        report('A11 raw Gaussian energy equals the HF functional at C_Q (Nf=%d)' % Nf, abs(cq['form_diff']), 1e-13)
        if Nf != 1:
            report('A11 C_Q copy-symmetric in the singlet component (Nf=%d)' % Nf, cq['copy_asymmetry'], 1e-12)
        else:
            log('      Nf=1: C_Q copy asymmetry %.3e (the doublet component is not copy-symmetric; information)' % cq['copy_asymmetry'])
    report('A11 cumulant / ground shift of the reference = 2 + O(G) (Nf=0; printed %.6f)' % secs[0]['ratio_cumulant_over_shift_partition'],
           abs(secs[0]['ratio_cumulant_over_shift_partition'] - 2.0), 1e-2)
    # A12 the partitioned shift against eigh
    report('A12 E_exact(eigh) - E_free - [G c1 + G^2 f] - residue = eigh - partition (Nf=0,1,2)', max(abs(s['shift_check']) for s in secs.values()), 5e-13)
    log('SELFTEST %s (%d failures)' % ('PASS' if fails == 0 else 'FAIL', fails))
    return 0 if fails == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ma', type=float, default=0.2)
    ap.add_argument('--G', type=float, default=1.0 / 4096)
    ap.add_argument('--Nf', default='0,2,1')
    ap.add_argument('--out', default='cluster13_v3_results.json')
    ap.add_argument('--no-frozen', action='store_true', help='skip the frozen-sea alternative')
    ap.add_argument('--hf-rule', default='mom', choices=['mom', 'aufbau'],
                    help='occupation rule of the HF iteration: mom (maximum overlap, stationary projector; default) or aufbau (version 1)')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--threads', type=int, default=None)
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest(args))
    run_main(args)


if __name__ == '__main__':
    main()
