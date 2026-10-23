"""Strand 2a library (spec solve/specs/SPEC_strand2a_gate.md; derivations solve/lead/LEAD.tex sec. 5; builder analyst).

Lattices L2 (2-site open chain, 2-component Weyl), R4 (ring4, 1-component, as strand 0), C8 (open 2 x 2 x 2 cube,
2-component Weyl).  Weyl-space index w = site * ncomp + a (site-major, component-minor).  Mode index (the one-body
space of h_*) j = 2 w + chiral (chiral fastest: 0 = u, the first block of h_* = [[0, B U^dag], [U B, 0]], 1 = d).
Packets s = 1..4 with masses m_s and orientations eta_s; the light packet s = 1 carries the interaction.  Quasiparticle
index i: i < nW is d_i (particle), i >= nW is f_{i - nW} (hole); q-language: q_i = d_i, q_{nW + r} = f_r = e_r^dag.
"""
import itertools
import os
import sys

import numpy as np
import scipy.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "frozen"))
import native_core as nc  # noqa: E402

SX, SY, SZ, I2 = nc.SX, nc.SY, nc.SZ, nc.I2
SIG = (SX, SY, SZ)


def herm(A):
    return 0.5 * (A + A.conj().T)


# ------------------------------------------------------------------ lattices
def lattice(name):
    """Return dict: nsite, ncomp, DW (nW x nW), positions (nsite x 3), dist (nsite x nsite), dirs (current directions)."""
    if name == "L2":
        D2 = np.array([[0, -0.5j], [0.5j, 0]])
        DW = np.kron(D2, SX)
        pos = np.array([[0.0, 0, 0], [1.0, 0, 0]])
        dist = np.abs(pos[:, None, 0] - pos[None, :, 0])
        return {"name": name, "nsite": 2, "ncomp": 2, "DW": DW, "positions": pos, "dist": dist, "dirs": [0],
                "coord": pos.astype(int)}
    if name == "R4":
        D = np.zeros((4, 4), dtype=complex)
        for x in range(4):
            D[x, (x + 1) % 4] += -0.5j
            D[x, (x - 1) % 4] += 0.5j
        d = np.array([[min(abs(x - y), 4 - abs(x - y)) for y in range(4)] for x in range(4)], dtype=float)
        pos = np.array([[x, 0, 0] for x in range(4)], dtype=float)
        return {"name": name, "nsite": 4, "ncomp": 1, "DW": D, "positions": pos, "dist": d, "dirs": [0],
                "coord": pos.astype(int)}
    if name == "C8":
        lat = nc.lattice("cube")
        pos = np.array(lat["positions"], dtype=float)
        dist = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
        return {"name": name, "nsite": 8, "ncomp": 2, "DW": lat["D"].copy(), "positions": pos, "dist": dist,
                "dirs": [0, 1, 2], "coord": pos.astype(int)}
    raise ValueError(name)


def backgrounds(lat):
    """(psi, N) per site: flat, the nonuniform one of spec sec. 2, and the seed-7 random one."""
    n = lat["nsite"]
    c = lat["coord"]
    if lat["name"] == "L2":
        nonu = (np.array([1.2, 0.8]), np.array([1.3, 0.7]))
    elif lat["name"] == "R4":
        x = c[:, 0]
        nonu = (1 + 0.2 * np.cos(2 * np.pi * x / 4), 1 + 0.3 * (-1.0) ** x)
    else:
        nonu = (1 + 0.2 * (-1.0) ** c[:, 0], 1 + 0.3 * (-1.0) ** c[:, 1])
    rng = np.random.default_rng(7)
    psi_r = rng.uniform(0.7, 1.3, n)
    N_r = rng.uniform(0.7, 1.3, n)
    return {"flat": (np.ones(n), np.ones(n)), "nonuniform": nonu, "random7": (psi_r, N_r)}


def perm(nW):
    P = np.zeros((2 * nW, 2 * nW))
    for w in range(nW):
        for s in range(2):
            P[2 * w + s, nW * s + w] = 1.0
    return P


# ------------------------------------------------------------------ one packet
class Packet:
    """Polar transport of one packet on the Weyl space; eta = +1 or -1 (the whole packet swapped)."""

    def __init__(self, lat, m, eta, psi, N):
        self.lat, self.m, self.eta = lat, float(m), int(eta)
        nc_ = lat["ncomp"]
        self.nW = nW = lat["nsite"] * nc_
        self.psi_s, self.N_s = np.asarray(psi, float), np.asarray(N, float)
        self.Psi = np.kron(np.diag(self.psi_s), np.eye(nc_))
        self.Psiinv = np.kron(np.diag(1.0 / self.psi_s), np.eye(nc_))
        self.L = np.kron(np.diag(self.N_s * self.psi_s), np.eye(nc_))
        self.L1 = np.kron(np.diag(self.psi_s), np.eye(nc_))
        A = self.m * self.Psiinv + 1j * self.Psiinv @ self.Psiinv @ self.Psiinv @ lat["DW"]
        self.A = A
        w0, V0 = np.linalg.eigh(herm(A.conj().T @ A))
        w1, V1 = np.linalg.eigh(herm(A @ A.conj().T))
        self.minAA = float(w0.min())
        if w0.min() <= 0:
            raise ValueError("A^dag A not positive definite (min %.3e)" % w0.min())
        self.E0 = (V0 * np.sqrt(w0)) @ V0.conj().T
        self.E0h = (V0 * w0 ** 0.25) @ V0.conj().T
        self.E1 = (V1 * np.sqrt(np.clip(w1, 0, None))) @ V1.conj().T
        self.E1h = (V1 * np.clip(w1, 0, None) ** 0.25) @ V1.conj().T
        self.U = A @ ((V0 / np.sqrt(w0)) @ V0.conj().T)
        self.P = perm(nW)

    def B(self, j, L=None):
        L = self.L if L is None else L
        Eh = self.E0h if j == 0 else self.E1h
        return herm(Eh @ L @ Eh)

    def jferm(self):
        return 0 if self.eta > 0 else 1

    def h(self, L=None):
        """Fermion one-body Hamiltonian (mode layout): eta = +: [[0, B0 U^dag],[U B0, 0]]; eta = -: [[0, B1 U],[U^dag B1, 0]]."""
        Z = np.zeros((self.nW, self.nW), dtype=complex)
        if self.eta > 0:
            B = self.B(0, L)
            Mb = np.block([[Z, B @ self.U.conj().T], [self.U @ B, Z]])
        else:
            B = self.B(1, L)
            Mb = np.block([[Z, B @ self.U], [self.U.conj().T @ B, Z]])
        return self.P @ Mb @ self.P.T

    def vacuum_energy(self, L=None):
        """eta Tr L (E1 - E0) = sea (-Tr B_j) + bosons (+Tr B_{1-j})."""
        j = self.jferm()
        return -np.trace(self.B(j, L)).real + np.trace(self.B(1 - j, L)).real

    def pieces(self, L=None):
        j = self.jferm()
        return {"sea": float(-np.trace(self.B(j, L)).real), "bosons": float(np.trace(self.B(1 - j, L)).real),
                "total": float(self.vacuum_energy(L)), "eta_TrL_E1_minus_E0": float(self.eta * np.trace((self.L if L is None else L) @ (self.E1 - self.E0)).real)}


def masses(m, H):
    return [m, np.sqrt(m * m + H * H), np.sqrt(m * m + H * H), np.sqrt(m * m + 2 * H * H)], [+1, -1, -1, +1]


def W0(lat, m, H, psi, N):
    ms, etas = masses(m, H)
    pk = [Packet(lat, mm, ee, psi, N) for mm, ee in zip(ms, etas)]
    return float(sum(p.vacuum_energy() for p in pk)), pk


def Jsum(m, H):
    ms, etas = masses(m, H)
    m2 = np.array(ms) ** 2
    return float(sum(e * x * np.log(x) for e, x in zip(etas, m2))), float(np.dot(etas, m2))


# ------------------------------------------------------------------ light packet quasiparticles
class Light:
    """The light packet (eta = +): one-body objects and the quasiparticle basis at N = 1 for the given psi."""

    def __init__(self, lat, m, psi, N):
        self.pk = Packet(lat, m, +1, psi, N)
        self.lat, self.nW = lat, self.pk.nW
        B0u = self.pk.B(0, self.pk.L1)
        self.eps, self.R = np.linalg.eigh(B0u)
        UR = self.pk.U @ self.R
        self.vp = self.pk.P @ np.vstack([self.R, UR]) / np.sqrt(2.0)
        self.vm = self.pk.P @ np.vstack([self.R, -UR]) / np.sqrt(2.0)
        self.Q = np.hstack([self.vp, self.vm])  # c = Q g, g = (d, e)
        self.eps_q = np.concatenate([self.eps, self.eps])

    def rbasis(self, M):
        return self.R.conj().T @ M @ self.R


# ------------------------------------------------------------------ the seed (spec sec. 3)
def regulator(lat, psi, Lam):
    nW = lat["nsite"] * lat["ncomp"]
    if not np.isfinite(Lam):
        return np.eye(2 * nW)
    Pinv = np.kron(np.diag(1.0 / np.asarray(psi, float)), np.eye(lat["ncomp"]))
    K = herm(Pinv @ lat["DW"] @ Pinv)
    R = sla.expm(-(K @ K) / (2 * Lam * Lam))
    return np.kron(herm(R), I2)


def site_op(lat, x, comp_mat, chiral_mat):
    Px = np.zeros((lat["nsite"], lat["nsite"]))
    Px[x, x] = 1.0
    return np.kron(np.kron(Px, comp_mat), chiral_mat)


def seed_bilinears(lat, psi, gM, gB, Lam):
    """List of (weight, M1, M2): V = sum weight (c^dag M1 c)(c^dag M2 c) (operator products, one-body parts included)."""
    R = regulator(lat, psi, Lam)
    psi = np.asarray(psi, float)
    ncp = lat["ncomp"]
    terms = []
    for x in range(lat["nsite"]):
        Mx = R @ site_op(lat, x, np.eye(ncp), SX) @ R
        terms.append((-gM * psi[x] ** -6, Mx, Mx))
    if gB != 0:
        if not np.isfinite(Lam):
            raise ValueError("V_B needs finite Lambda")
        ell = 1.0 / Lam
        for i in lat["dirs"]:
            sig = SIG[i] if ncp == 2 else np.eye(1)
            J = [R @ site_op(lat, x, sig, SY) @ R for x in range(lat["nsite"])]
            for x in range(lat["nsite"]):
                for y in range(lat["nsite"]):
                    d = lat["dist"][x, y] * (psi[x] ** 2 + psi[y] ** 2) / 2.0
                    w = -(gB / 2.0) * ell ** -3 * np.exp(-d * d / (ell * ell))
                    terms.append((w, J[x], J[y]))
    return terms


def seed_tensor(terms, n):
    """T_{abcd} with V = sum T_{abcd} c_a^dag c_b^dag c_d c_c + one-body: (c^dag M1 c)(c^dag M2 c) -> T_{abcd} += w M1_ac M2_bd."""
    T = np.zeros((n, n, n, n), dtype=complex)
    for w, M1, M2 in terms:
        T += w * np.einsum("ac,bd->abcd", M1, M2)
    return T


def transform(T, Q):
    """T'_{ijkl} = sum conj(Q_ai) conj(Q_bj) Q_ck Q_dl T_abcd: V = sum T'_{ijkl} g_i^dag g_j^dag g_l g_k with c = Q g."""
    Qc = Q.conj()
    T1 = np.einsum("abcd,ai->ibcd", T, Qc, optimize=True)
    T1 = np.einsum("ibcd,bj->ijcd", T1, Qc, optimize=True)
    T1 = np.einsum("ijcd,ck->ijkd", T1, Q, optimize=True)
    return np.einsum("ijkd,dl->ijkl", T1, Q, optimize=True)


def two_two(Tp, nW):
    """The (2,2) part in q-language: W_A (antisymmetric in (c1, c2) and (a1, a2)) with V_* = sum W_A q_c1^dag q_c2^dag q_a2 q_a1;
    v_IJ = 4 W_A[i1, i2, j1, j2].  Position types of g_i^dag g_j^dag g_l g_k: particle index -> (C, C, A, A),
    hole index -> (A, A, C, C) (e = f^dag)."""
    n = 2 * nW
    W = np.zeros((n, n, n, n), dtype=complex)
    rng = {0: slice(0, nW), 1: slice(nW, n)}
    for ti, tj, tk, tl in itertools.product((0, 1), repeat=4):
        types = ["C" if ti == 0 else "A", "C" if tj == 0 else "A", "A" if tl == 0 else "C", "A" if tk == 0 else "C"]
        if types.count("C") != 2:
            continue
        labels = ["i", "j", "l", "k"]
        cpos = [p for p in range(4) if types[p] == "C"]
        apos = [p for p in range(4) if types[p] == "A"]
        permn = cpos + apos
        inv = sum(1 for x in range(4) for y in range(x + 1, 4) if permn[x] > permn[y])
        sign = -1.0 if inv % 2 else 1.0
        c1, c2 = labels[cpos[0]], labels[cpos[1]]
        a1p, a2p = labels[apos[0]], labels[apos[1]]
        # W[c1, c2, a1 = a2p, a2 = a1p] += sign T'[i, j, k, l]
        sub = Tp[rng[ti], rng[tj], rng[tk], rng[tl]]
        order = "".join([c1, c2, a2p, a1p])
        blk = np.einsum("ijkl->" + order, sub)
        tmap = {"i": ti, "j": tj, "k": tk, "l": tl}
        W[rng[tmap[c1]], rng[tmap[c2]], rng[tmap[a2p]], rng[tmap[a1p]]] += sign * blk
    WA = (W - W.transpose(1, 0, 2, 3) - W.transpose(0, 1, 3, 2) + W.transpose(1, 0, 3, 2)) / 4.0
    return WA


def classes(nW):
    n = 2 * nW
    pairs = [(i1, i2) for i1 in range(n) for i2 in range(i1 + 1, n)]
    cls = lambda I: "pp" if I[1] < nW else ("hh" if I[0] >= nW else "ph")
    return pairs, cls


def vIJ_from_WA(WA, nW):
    pairs, cls = classes(nW)
    v = {}
    for I in pairs:
        for J in pairs:
            if cls(I) == cls(J):
                val = 4.0 * WA[I[0], I[1], J[0], J[1]]
                if val != 0:
                    v[(I, J)] = complex(val)
    return v


# ------------------------------------------------------------------ quasiparticle blocks (q-language Fock space)
class QBlocks:
    """States with n_p particles and n_h holes over Omega: sorted q-index tuples; q_j annihilators between blocks with the
    Jordan-Wigner sign (-1)^(number of occupied q-modes below j)."""

    def __init__(self, nW):
        self.nW = nW
        self.basis, self.index = {}, {}
        self._ann = {}

    def block(self, npp, nh):
        key = (npp, nh)
        if key not in self.basis:
            P = list(itertools.combinations(range(self.nW), npp))
            H = list(itertools.combinations(range(self.nW, 2 * self.nW), nh))
            b = [p + h for p in P for h in H]
            self.basis[key] = b
            self.index[key] = {t: i for i, t in enumerate(b)}
        return self.basis[key]

    def dim(self, npp, nh):
        return len(self.block(npp, nh))

    def ann(self, j, blk):
        """q_j from block blk to the block with one fewer of j's type: dense (d_out x d_in) matrix, or None."""
        key = (j, blk)
        if key in self._ann:
            return self._ann[key]
        npp, nh = blk
        out = (npp - 1, nh) if j < self.nW else (npp, nh - 1)
        if out[0] < 0 or out[1] < 0:
            self._ann[key] = None
            return None
        bi = self.block(*blk)
        self.block(*out)
        io = self.index[out]
        M = np.zeros((len(io), len(bi)))
        for col, t in enumerate(bi):
            if j in t:
                rest = tuple(o for o in t if o != j)
                M[io[rest], col] = (-1) ** sum(1 for o in t if o < j)
        self._ann[key] = (M, out)
        return self._ann[key]

    def op(self, kind, j, blk):
        """('a', j) or ('ad', j) acting on blk: (matrix, out_block) or (None, None)."""
        if kind == "a":
            r = self.ann(j, blk)
            return (None, None) if r is None else r
        npp, nh = blk
        src = (npp + 1, nh) if j < self.nW else (npp, nh + 1)
        if src[0] > self.nW or src[1] > self.nW:
            return None, None
        r = self.ann(j, src)
        if r is None:
            return None, None
        return r[0].T, src

    def chain(self, ops, blk):
        M, cur = None, blk
        for kind, j in reversed(ops):
            A, nxt = self.op(kind, j, cur)
            if A is None:
                return None, None
            M = A if M is None else A @ M
            cur = nxt
        return M, cur

    def monomial_op(self, coef, blk):
        """sum_IJ coef[(I, J)] q_I^dag q_J on block blk by direct action on basis states (I = (i1 < i2), q_I^dag = q_i1^dag q_i2^dag,
        q_J = q_j2 q_j1)."""
        b = self.block(*blk)
        idx = self.index[blk]
        d = len(b)
        M = np.zeros((d, d), dtype=complex)
        byJ = {}
        for (I, J), c in coef.items():
            byJ.setdefault(J, []).append((I, c))
        for col, t in enumerate(b):
            ts = set(t)
            for J, lst in byJ.items():
                if J[0] not in ts or J[1] not in ts:
                    continue
                # q_j2 q_j1 |t>: remove j1 then j2
                s1 = (-1) ** sum(1 for o in t if o < J[0])
                t1 = tuple(o for o in t if o != J[0])
                s2 = (-1) ** sum(1 for o in t1 if o < J[1])
                t2 = tuple(o for o in t1 if o != J[1])
                for I, c in lst:
                    if I[0] in t2 or I[1] in t2:
                        continue
                    # q_i1^dag q_i2^dag |t2>: add i2 then i1
                    s3 = (-1) ** sum(1 for o in t2 if o < I[1])
                    t3 = tuple(sorted(t2 + (I[1],)))
                    s4 = (-1) ** sum(1 for o in t3 if o < I[0])
                    t4 = tuple(sorted(t3 + (I[0],)))
                    M[idx[t4], col] += c * s1 * s2 * s3 * s4
        return M


BLOCKS_ALL = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (2, 1), (1, 2))


class Dressed:
    """S_*, e^{+-S_*} and the dressed excitation Hamiltonian on q-blocks, from v_IJ and the light packet's one-body data."""

    def __init__(self, light, v, blocks=BLOCKS_ALL):
        self.light, self.v = light, v
        self.nW = light.nW
        self.qb = QBlocks(self.nW)
        eq = light.eps_q
        self.coefS = {(I, J): c / (eq[I[0]] + eq[I[1]] + eq[J[0]] + eq[J[1]]) for (I, J), c in v.items()}
        self.S, self.eS, self.emS, self.Vstar = {}, {}, {}, {}
        for blk in blocks:
            self.qb.block(*blk)
            self.S[blk] = self.qb.monomial_op(self.coefS, blk)
            self.Vstar[blk] = self.qb.monomial_op(v, blk)
            self.eS[blk] = sla.expm(self.S[blk])
            self.emS[blk] = sla.expm(-self.S[blk])

    def H(self, blk, Br, dressed=True):
        """sum_rr' Br[r,r'] F_dr^dag F_dr' + sum_rr' Br[r',r] F_fr^dag F_fr' on blk (Br in the |r> basis)."""
        nW = self.nW
        d = self.qb.dim(*blk)
        out = np.zeros((d, d), dtype=complex)
        npp, nh = blk
        for part, rs, sub in (("p", range(nW), (npp - 1, nh)), ("h", range(nW, 2 * nW), (npp, nh - 1))):
            if sub[0] < 0 or sub[1] < 0:
                continue
            ops = []
            for j in rs:
                r = self.qb.ann(j, blk)
                ops.append(None if r is None else r[0])
            if all(o is None for o in ops):
                continue
            G = self.emS[sub] @ self.emS[sub] if dressed else None
            n = nW
            for a in range(n):
                if ops[a] is None:
                    continue
                Y = np.zeros_like(ops[a], dtype=complex)
                for b in range(n):
                    if ops[b] is None:
                        continue
                    w = Br[a, b] if part == "p" else Br[b, a]
                    if w != 0:
                        Y += w * ops[b]
                if dressed:
                    out += ops[a].T @ (G @ Y)
                else:
                    out += ops[a].T @ Y
        if dressed:
            out = self.eS[blk] @ out @ self.eS[blk]
        return out
