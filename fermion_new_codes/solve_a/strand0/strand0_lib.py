"""Strand 0 (test T2) library: the displayed law H_* on a ring (light packet only) and its maps to the native action.

Spec: solve/specs/SPEC_strand0_source_allocation.md (lead, 2026-10-14); derivations in solve/lead/LEAD.tex sec. 3.
Builder code (analyst); no physics choices beyond the spec.

Conventions.  Ring of n sites, scalar sine stencil D = (T_+ - T_-)/(2i).  Modes j = 2x + sigma (sigma = 0: chiral block u,
sigma = 1: block d), the layout of the frozen native code (mode(x, a) = x * ncomp + a).  "Block layout" orders the 2n
one-body indices as (u sites 0..n-1, then d sites 0..n-1); PERM maps block layout to mode layout.  Fixed-N sectors and
the Jordan-Wigner signs are those of native_core.Sector and native_sources.Fock (frozen copies in ./frozen).

Law (spec sec. 1): A = m psi^-1 + i psi^-3 D, E0 = (A^dag A)^(1/2), E1 = (A A^dag)^(1/2), U = A E0^-1,
B0 = E0^(1/2) (N psi) E0^(1/2), B1 = E1^(1/2) (N psi) E1^(1/2); h_* = [[0, B0 U^dag], [U B0, 0]] (block layout).
Quasiparticle basis |r> = eigenvectors of B0 at N = 1 (held fixed for every N); v_r^+- = (|r>, +-U|r>)/sqrt2;
d_r = (v_r^+)^dag c, e_r = (v_r^-)^dag c, f_r = e_r^dag.  Quasiparticle index i: i < n is d_i, i >= n is f_{i-n}.
"""
import os
import sys

import numpy as np
import scipy.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "frozen"))
import native_core as nc  # noqa: E402
import native_sources as ns  # noqa: E402

SX, SY, SZ, I2 = nc.SX, nc.SY, nc.SZ, nc.I2


def herm(A):
    return 0.5 * (A + A.conj().T)


def ring_D(L):
    """Scalar sine stencil on a periodic ring of L sites: D[x, x+1] = -i/2, D[x, x-1] = +i/2."""
    D = np.zeros((L, L), dtype=complex)
    for x in range(L):
        D[x, (x + 1) % L] += -0.5j
        D[x, (x - 1) % L] += 0.5j
    return D


def perm(n):
    """PERM[mode(x, s), block(x, s)] = 1 with mode(x, s) = 2x + s and block(x, s) = n s + x."""
    P = np.zeros((2 * n, 2 * n))
    for x in range(n):
        for s in range(2):
            P[2 * x + s, n * s + x] = 1.0
    return P


def polar(m, psi, D):
    """Polar transport objects of spec sec. 1.2 (eigh-based positive roots)."""
    psi = np.asarray(psi, dtype=float)
    A = m * np.diag(1.0 / psi) + 1j * np.diag(psi ** -3) @ D
    w0, V0 = np.linalg.eigh(herm(A.conj().T @ A))
    w1, V1 = np.linalg.eigh(herm(A @ A.conj().T))
    if w0.min() <= 0:
        raise ValueError("A^dag A is not positive definite: min eig %.3e" % w0.min())
    E0 = (V0 * np.sqrt(w0)) @ V0.conj().T
    E0h = (V0 * w0 ** 0.25) @ V0.conj().T
    E0inv = (V0 / np.sqrt(w0)) @ V0.conj().T
    E1 = (V1 * np.sqrt(np.clip(w1, 0, None))) @ V1.conj().T
    E1h = (V1 * np.clip(w1, 0, None) ** 0.25) @ V1.conj().T
    U = A @ E0inv
    return {"A": A, "E0": E0, "E1": E1, "E0h": E0h, "E1h": E1h, "U": U, "minAA": float(w0.min())}


class Law:
    """One-body objects of the law at background (psi, N); quasiparticle basis fixed at N = 1."""

    def __init__(self, m, psi, N=None, D=None):
        self.m = float(m)
        self.psi = np.asarray(psi, dtype=float)
        self.n = n = len(self.psi)
        self.N = np.ones(n) if N is None else np.asarray(N, dtype=float)
        self.D = ring_D(n) if D is None else D
        self.__dict__.update(polar(self.m, self.psi, self.D))
        self.P = perm(n)
        self.B0_unit = herm(self.E0h @ np.diag(self.psi) @ self.E0h)
        self.B0 = herm(self.E0h @ np.diag(self.N * self.psi) @ self.E0h)
        self.B1 = herm(self.E1h @ np.diag(self.N * self.psi) @ self.E1h)
        self.eps, self.R = np.linalg.eigh(self.B0_unit)
        UR = self.U @ self.R
        self.vp = self.P @ np.vstack([self.R, UR]) / np.sqrt(2.0)
        self.vm = self.P @ np.vstack([self.R, -UR]) / np.sqrt(2.0)
        self.Vq = np.hstack([self.vp, self.vm])

    def rbasis(self, M):
        return self.R.conj().T @ M @ self.R

    def h_star(self, B0=None):
        B0 = self.B0 if B0 is None else B0
        Z = np.zeros((self.n, self.n), dtype=complex)
        Mb = np.block([[Z, B0 @ self.U.conj().T], [self.U @ B0, Z]])
        return self.P @ Mb @ self.P.T

    def X(self, x):
        Px = np.zeros((self.n, self.n))
        Px[x, x] = 1.0
        return herm(self.psi[x] * self.E0h @ Px @ self.E0h)

    def dh_dN(self, x):
        return self.h_star(B0=self.X(x))

    def G(self):
        """Sea density matrix sum_r v_r^- v_r^-dag (mode layout); <c^dag M c>_Omega = Tr(M G)."""
        return self.vm @ self.vm.conj().T

    def site_modes(self, x):
        Pm = np.zeros((2 * self.n, 2 * self.n))
        Pm[2 * x, 2 * x] = Pm[2 * x + 1, 2 * x + 1] = 1.0
        return Pm

    def midpoint(self, x, h=None):
        """(1/2){P_x (x) 1_2, h} (mode layout); h defaults to h_* at the background N."""
        h = self.h_star() if h is None else h
        Pm = self.site_modes(x)
        return 0.5 * (Pm @ h + h @ Pm)


class QFock:
    """Quasiparticle operators of a Law as maps between the fixed-N sectors of its 2n modes."""

    def __init__(self, law, fock):
        self.law, self.fock = law, fock
        self.n, self.nm = law.n, 2 * law.n
        self._a = {}

    def dN(self, i):
        return -1 if i < self.n else +1

    def a(self, i, N):
        """a_i acting on sector N (to N + dN(i)), or None if the target sector does not exist."""
        key = (i, N)
        if key in self._a:
            return self._a[key]
        n, nm = self.n, self.nm
        M = None
        if i < n and 1 <= N <= nm:
            v = self.law.vp[:, i]
            M = np.zeros((self.fock.sector(N - 1).d, self.fock.sector(N).d), dtype=complex)
            for j in range(nm):
                if v[j] != 0:
                    M += np.conj(v[j]) * self.fock.ann(N, j)
        elif i >= n and 0 <= N <= nm - 1:
            v = self.law.vm[:, i - n]
            M = np.zeros((self.fock.sector(N + 1).d, self.fock.sector(N).d), dtype=complex)
            for j in range(nm):
                if v[j] != 0:
                    M += v[j] * self.fock.cdag(N + 1, j)
        self._a[key] = M
        return M

    def adag(self, i, N):
        """a_i^dag acting on sector N (to N - dN(i)): the adjoint of a_i acting on N - dN(i)."""
        M = self.a(i, N - self.dN(i))
        return None if M is None else M.conj().T

    def vacuum(self):
        """|Omega> = f_0 f_1 ... f_{n-1} |0> (the Slater determinant of the v_r^-), in sector N = n."""
        vec = np.ones(1, dtype=complex)
        N = 0
        for r in reversed(range(self.n)):
            vec = self.a(self.n + r, N) @ vec
            N += 1
        return vec / np.linalg.norm(vec), N

    def nexc(self, N):
        d = self.fock.sector(N).d
        out = np.zeros((d, d), dtype=complex)
        for i in range(self.nm):
            M = self.a(i, N)
            if M is not None:
                out += M.conj().T @ M
        return herm(out)

    def bare_ops(self, N):
        return [self.a(r, N) for r in range(self.n)], [self.a(self.n + r, N) for r in range(self.n)]

    def onebody(self, Br, N, ops=None):
        """sum_rr' Br[r,r'] F_dr^dag F_dr' + sum_rr' Br[r',r] F_fr^dag F_fr' on sector N (bare ops if ops is None)."""
        Dm, Fm = self.bare_ops(N) if ops is None else ops
        d = self.fock.sector(N).d
        out = np.zeros((d, d), dtype=complex)
        for r in range(self.n):
            for rp in range(self.n):
                if Dm[r] is not None and Dm[rp] is not None and Br[r, rp] != 0:
                    out += Br[r, rp] * (Dm[r].conj().T @ Dm[rp])
                if Fm[r] is not None and Fm[rp] is not None and Br[rp, r] != 0:
                    out += Br[rp, r] * (Fm[r].conj().T @ Fm[rp])
        return out

    def chain(self, ops, N):
        """Matrix of the product ops[0] ops[1] ... ops[-1] (each ('a'|'ad', i)) applied to sector N; None if empty."""
        M, cur = None, N
        for kind, i in reversed(ops):
            A = self.a(i, cur) if kind == "a" else self.adag(i, cur)
            if A is None:
                return None, None
            cur = cur + self.dN(i) if kind == "a" else cur - self.dN(i)
            M = A if M is None else A @ M
        return M, cur


def site_pm(n, x, sign):
    """Pi_+- = (1 +- sigma_1)/2 on the two modes of site x (mode layout)."""
    P = np.zeros((2 * n, 2 * n), dtype=complex)
    P[2 * x:2 * x + 2, 2 * x:2 * x + 2] = 0.5 * (I2 + sign * SX)
    return P


def seed_V(law, fock, N, U):
    """V[psi] = U sum_x psi_x^-6 n_{x,+} n_{x,-} on sector N (regulator 1, no bond seed: spec sec. 1.5)."""
    sec = fock.sector(N)
    out = np.zeros((sec.d, sec.d), dtype=complex)
    for x in range(law.n):
        npl = sec.bilinear(site_pm(law.n, x, +1))
        nmi = sec.bilinear(site_pm(law.n, x, -1))
        out += law.psi[x] ** -6 * (npl @ nmi)
    return U * out


def nexc_projectors(nexc, tol=1e-8):
    w, V = np.linalg.eigh(nexc)
    k = np.rint(w).astype(int)
    if np.max(np.abs(w - k)) > tol:
        raise ValueError("n_exc eigenvalues not integer: %.3e" % np.max(np.abs(w - k)))
    return {int(kk): V[:, k == kk] for kk in sorted(set(k))}


class Completion:
    """Quasiparticle normal order of the seed, the p = q >= 2 projection V_*, S_* and the dressing (spec sec. 1.5)."""

    def __init__(self, law, fock, U, sectors=range(2, 8), dress=(4, 5, 6)):
        self.law, self.fock, self.U = law, fock, U
        self.qf = qf = QFock(law, fock)
        n, nm = law.n, 2 * law.n
        self.n, self.nm = n, nm
        self.eps_qp = np.concatenate([law.eps, law.eps])
        self.Omega, self.N0 = qf.vacuum()
        self.sectors = list(sectors)
        allN = sorted(set(self.sectors) | {self.N0 - 1, self.N0, self.N0 + 1})
        self.V = {N: seed_V(law, fock, N, U) for N in allN}
        Om = self.Omega
        self.v0 = complex(Om.conj() @ self.V[self.N0] @ Om)
        # one-excitation states and v1
        self.one = {}
        for i in range(nm):
            A = qf.adag(i, self.N0)
            self.one[i] = (A @ Om, self.N0 - qf.dN(i))
        self.v1 = np.zeros((nm, nm), dtype=complex)
        for i in range(nm):
            for j in range(nm):
                (vi, Ni), (vj, Nj) = self.one[i], self.one[j]
                if Ni == Nj:
                    self.v1[i, j] = vi.conj() @ self.V[Ni] @ vj - (self.v0 if i == j else 0.0)
        # :V:, n_exc projectors, V_*
        self.nex, self.proj, self.Vno, self.Vstar = {}, {}, {}, {}
        for N in self.sectors:
            d = fock.sector(N).d
            dG = np.zeros((d, d), dtype=complex)
            for i in range(nm):
                for j in range(nm):
                    if abs(self.v1[i, j]) > 0 and qf.dN(i) == qf.dN(j):
                        M, Nout = qf.chain([("ad", i), ("a", j)], N)
                        if M is not None:
                            dG += self.v1[i, j] * M
            self.Vno[N] = self.V[N] - dG - self.v0 * np.eye(d)
            self.nex[N] = qf.nexc(N)
            self.proj[N] = nexc_projectors(self.nex[N])
            Vs = np.zeros((d, d), dtype=complex)
            for k, Pk in self.proj[N].items():
                Vs += Pk @ (Pk.conj().T @ self.Vno[N] @ Pk) @ Pk.conj().T
            self.Vstar[N] = Vs
        # two-excitation states and v_IJ
        self.pairs = [(i1, i2) for i1 in range(nm) for i2 in range(i1 + 1, nm)]
        self.two = {}
        for (i1, i2) in self.pairs:
            M, Nout = qf.chain([("ad", i1), ("ad", i2)], self.N0)
            self.two[(i1, i2)] = (M @ Om, Nout)
        self.vIJ = {}
        for I in self.pairs:
            for J in self.pairs:
                (vI, NI), (vJ, NJ) = self.two[I], self.two[J]
                if NI == NJ and NI in self.Vstar:
                    self.vIJ[(I, J)] = complex(vI.conj() @ self.Vstar[NI] @ vJ)
        # S_* on every needed sector
        needS = sorted(set(self.sectors) | {N + dd for N in dress for dd in (-1, 0, 1)})
        self.S = {}
        for N in needS:
            d = fock.sector(N).d
            S = np.zeros((d, d), dtype=complex)
            for (I, J), v in self.vIJ.items():
                if v == 0:
                    continue
                den = self.eps_qp[I[0]] + self.eps_qp[I[1]] + self.eps_qp[J[0]] + self.eps_qp[J[1]]
                M, Nout = qf.chain([("ad", I[0]), ("ad", I[1]), ("a", J[1]), ("a", J[0])], N)
                if M is not None:
                    S += (v / den) * M
            self.S[N] = S
        self.expS = {N: sla.expm(S) for N, S in self.S.items()}
        self.expmS = {N: sla.expm(-S) for N, S in self.S.items()}
        self.dressed = {N: self._dress(N) for N in dress}

    def monomial_sum(self, N, coef=None):
        """sum_IJ c_IJ a_I^dag a_J on sector N (c = v_IJ by default): control C5d."""
        d = self.fock.sector(N).d
        out = np.zeros((d, d), dtype=complex)
        for (I, J), v in self.vIJ.items():
            c = v if coef is None else coef(I, J, v)
            if c == 0:
                continue
            M, Nout = self.qf.chain([("ad", I[0]), ("ad", I[1]), ("a", J[1]), ("a", J[0])], N)
            if M is not None:
                out += c * M
        return out

    def _dress(self, N):
        qf, n = self.qf, self.n
        Dm, Fm = [], []
        for r in range(n):
            A = qf.a(r, N)
            Dm.append(None if A is None else self.expmS[N - 1] @ A @ self.expS[N])
            A = qf.a(n + r, N)
            Fm.append(None if A is None else self.expmS[N + 1] @ A @ self.expS[N])
        return Dm, Fm

    def H_exc(self, N, B0=None):
        B0 = self.law.B0 if B0 is None else B0
        return self.qf.onebody(self.law.rbasis(B0), N, self.dressed[N])

    def rhoP(self, x, N, dressed=True):
        return self.qf.onebody(self.law.rbasis(self.law.X(x)), N, self.dressed[N] if dressed else None)

    def H_free(self, N, B0=None):
        B0 = self.law.B0 if B0 is None else B0
        return self.qf.onebody(self.law.rbasis(B0), N, None)

    def contact1(self, x, N):
        """First-order leg-weighted contact source, LEAD.tex eq. (contact-1), on sector N."""
        n, nm, qf = self.n, self.nm, self.qf
        Xr = self.law.rbasis(self.law.X(x))
        Xq = np.zeros((nm, nm), dtype=complex)
        Xq[:n, :n] = Xr
        Xq[n:, n:] = Xr.T
        d = self.fock.sector(N).d
        out = np.zeros((d, d), dtype=complex)
        for (I, J), v in self.vIJ.items():
            if v == 0:
                continue
            i1, i2 = I
            j1, j2 = J
            c = v / (self.eps_qp[i1] + self.eps_qp[i2] + self.eps_qp[j1] + self.eps_qp[j2])
            for r in range(nm):
                terms = []
                if Xq[r, i1] != 0:
                    terms.append((Xq[r, i1], [("ad", r), ("ad", i2), ("a", j2), ("a", j1)]))
                if Xq[r, i2] != 0:
                    terms.append((Xq[r, i2], [("ad", i1), ("ad", r), ("a", j2), ("a", j1)]))
                if Xq[j1, r] != 0:
                    terms.append((Xq[j1, r], [("ad", i1), ("ad", i2), ("a", j2), ("a", r)]))
                if Xq[j2, r] != 0:
                    terms.append((Xq[j2, r], [("ad", i1), ("ad", i2), ("a", r), ("a", j1)]))
                for w, ops in terms:
                    M, Nout = qf.chain(ops, N)
                    if M is not None:
                        out += c * w * M
        return out


def normal_ordered_bilinear(sec, M, Omega_full, N0, fock):
    """:c^dag M c: against Omega: the sector bilinear minus its Omega expectation (computed in sector N0)."""
    c = complex(Omega_full.conj() @ fock.sector(N0).bilinear(M) @ Omega_full)
    return sec.bilinear(M) - c * np.eye(sec.d), c


# ------------------------------------------------------------------ native side and the spinor map
def W_candidates():
    W = 0.5 * (I2 - 1j * (SX + SY + SZ))
    return {"W": W, "Wdag": W.conj().T}


def gamma_of(Wfull, sec):
    """Induced sector unitary Gamma(W)_{IJ} = det(Wfull[I, J]) (rows I, columns J occupied tuples)."""
    d = sec.d
    G = np.zeros((d, d), dtype=complex)
    for a, I in enumerate(sec.basis):
        for b, J in enumerate(sec.basis):
            sub = Wfull[np.ix_(I, J)]
            if np.any(sub != 0):
                G[a, b] = np.linalg.det(sub)
    return G


def native_h(lat, m, s):
    """h_nat(s) = m beta_nat + s^-1 D_nat s^-1 (mode layout, ncomp = 2)."""
    Sinv = np.kron(np.diag(1.0 / np.asarray(s, dtype=float)), I2)
    return m * lat["beta"] + Sinv @ lat["D"] @ Sinv


def native_midpoint(lat, m, s, x):
    h = native_h(lat, m, s)
    nmodes = h.shape[0]
    Pm = np.zeros((nmodes, nmodes))
    Pm[2 * x, 2 * x] = Pm[2 * x + 1, 2 * x + 1] = 1.0
    return 0.5 * (Pm @ h + h @ Pm)


def lowest_state_dm(H, tol=1e-9, P=None):
    """Block-averaged density matrix of the lowest level of H (optionally compressed to the columns P)."""
    if P is not None:
        Hb = herm(P.conj().T @ H @ P)
    else:
        Hb = herm(H)
    w, V = np.linalg.eigh(Hb)
    sel = np.where(w - w[0] < tol)[0]
    Vl = V[:, sel]
    if P is not None:
        Vl = P @ Vl
    rho = Vl @ Vl.conj().T / len(sel)
    return float(w[0]), len(sel), rho


def eig_range(M):
    w = np.linalg.eigvalsh(herm(M))
    return float(w[0]), float(w[-1])


def rel_norms(Delta, Ref):
    fro = np.linalg.norm(Delta) / max(np.linalg.norm(Ref), 1e-300)
    spec = np.linalg.norm(Delta, 2) / max(np.linalg.norm(Ref, 2), 1e-300)
    return float(fro), float(spec)
