"""Native light-fermion sources and their promotions (integ69_nativedefs2, as briefed by the lead 2026-10-12 03:00).

Scalar action (item 2):  e_i(s) = M_i + (1/2) sum_{b in i} T_b/(s_i s_j) + s_i^-6 V_i,  M_i = m c_i^dag beta c_i,
V_i = U n_i1 n_i2, T_b = B_b + B_b^dag, B_b = c_i^dag D_ij c_j (degrees 0, -2, -6).
Subtraction (item 3): CAR coefficients normal-ordered against the frozen s = 1 Dirac sea of h_1 = m beta + D,
contractions G_ab = <c_a^dag c_b> frozen before any response.  'wick': full Wick ordering of every coefficient
(the quartic loses its bilinear contractions too); 'energy': only the sea expectation of each coefficient is
removed.  The two differ only in the contact.  Promotions (item 4), A_i = S_i^-1, C_i = S_i^-3, X o Y = {X, Y}/2:

  J_W6  Jordan:   M_i + S_i^-2 o T_i + S_i^-6 o V_i,      T_i = (1/2) sum_{b in i} T_b
  J_CVC Jordan:   M_i + S_i^-2 o T_i + C_i V_i C_i
  END   endpoint: M_i + (1/2) sum_b (A_i B_b A_j + A_j B_b^dag A_i) + C_i V_i C_i
        (right placement q = c A, F = c_2 c_1 C gives the same operator; checked literally by right_placement_e)
  FI    field-interposed: M_i + (1/2) sum_b (c_i^dag A_i D_ij A_j c_j + h.c. - <B_ij> A_i A_j - <B_ji> A_j A_i)
        + C_i V_i C_i, rulers inside the fields acting on N - 1 particles (sector chain)
  SJ    coefficient spectral/Jordan: M_i + (1/2) sum_b (A_i o A_j) o T_b + S_i^-6 o V_i
  PIN   commuting-ruler pinching sum_a Pi_a e_i(lambda_a) Pi_a (scalar bridge per basis state; see pinning_sources)

Counterterms take the placement of the coefficient they subtract (A_i (B_ij - <B_ij>) A_j, C (V - ct) C).
"""
import itertools

import numpy as np

import native_core as nc
from native_block import BExpr

PROMOTIONS = ("J_W6", "J_CVC", "END", "FI", "SJ", "PIN")
SUBTRACTIONS = ("wick", "energy")


# ------------------------------------------------------------------ Fock operators
class Fock:
    """Fixed-number sectors of nmodes modes with c_j : N -> N - 1 as rectangular matrices."""

    def __init__(self, nmodes):
        self.nmodes = nmodes
        self.sec = {}
        self._ann = {}

    def sector(self, N):
        if N not in self.sec:
            self.sec[N] = nc.Sector(self.nmodes, N)
        return self.sec[N]

    def ann(self, N, j):
        """c_j from sector N to sector N - 1 (d_{N-1} x d_N), Jordan-Wigner sign (-1)^(#occupied modes < j)."""
        key = (N, j)
        if key not in self._ann:
            hi, lo = self.sector(N), self.sector(N - 1)
            M = np.zeros((lo.d, hi.d))
            for col, occ in enumerate(hi.basis):
                if j in occ:
                    rest = tuple(o for o in occ if o != j)
                    M[lo.index[rest], col] = (-1) ** sum(1 for o in occ if o < j)
            self._ann[key] = M
        return self._ann[key]

    def cdag(self, N, i):
        """c_i^dag from sector N - 1 to sector N."""
        return self.ann(N, i).T


def bonds_of(lat):
    """Unordered nearest-neighbour pairs (i < j) of the lattice graph (independent of the twist)."""
    D0 = nc.lattice(lat["name"], (0.7, 0.4, 0.0) if lat["name"] != "cube" else (0, 0, 0), lat["ncomp"])["D"]
    nc_ = lat["ncomp"]
    out = []
    for i in range(lat["n"]):
        for j in range(i + 1, lat["n"]):
            if np.abs(D0[i * nc_:(i + 1) * nc_, j * nc_:(j + 1) * nc_]).max() > 0:
                out.append((i, j))
    return out


def sea_contractions(lat, m, nmodes):
    """G_ab = <sea| c_a^dag c_b |sea> for the negative-energy states of h_1 = m beta + D (zero on extra modes).
    For a lattice without beta (the cube) the reference is the empty state, G = 0."""
    G = np.zeros((nmodes, nmodes), dtype=complex)
    if lat["beta"] is None:
        return G, 0
    h1 = m * lat["beta"] + lat["D"]
    w, V = np.linalg.eigh(h1)
    if np.min(np.abs(w)) < 1e-12:
        raise ValueError("zero mode in h_1")
    Vs = V[:, w < 0]
    nm = h1.shape[0]
    G[:nm, :nm] = np.conj(Vs) @ Vs.T
    return G, int(Vs.shape[1])


class Coefficients:
    """Subtracted CAR coefficients of one sector: Mbar_x, Bbar_ij, Vbar_x (and the sea values <B_ij>)."""

    def __init__(self, lat, fock, N, m, U, G, subtraction, extra=None):
        self.N = N
        sec = fock.sector(N)
        d = sec.d
        I = np.eye(d, dtype=complex)
        ncp = lat["ncomp"]
        mode = lambda x, a: x * ncp + a
        hopbar = lambda a, b: sec.hop(a, b) - G[a, b] * I
        self.M, self.V = [], []
        for x in range(lat["n"]):
            if lat["beta"] is not None:
                beta = lat["beta"][x * ncp:(x + 1) * ncp, x * ncp:(x + 1) * ncp]
            else:
                beta = np.eye(ncp)
            Mx = np.zeros((d, d), dtype=complex)
            for a in range(ncp):
                for b in range(ncp):
                    if beta[a, b] != 0:
                        Mx += m * beta[a, b] * hopbar(mode(x, a), mode(x, b))
            if extra is not None and x in extra:
                Mx = Mx + extra[x](sec)
            self.M.append(Mx)
            if ncp >= 2:
                a, b = mode(x, 0), mode(x, 1)
                nab = sec.number(a) @ sec.number(b)
                pair = G[a, a] * G[b, b] - G[a, b] * G[b, a]
                if subtraction == "wick":
                    Vx = (nab - G[a, a] * hopbar(b, b) - G[b, b] * hopbar(a, a) + G[a, b] * hopbar(b, a)
                          + G[b, a] * hopbar(a, b) - pair * I)
                elif subtraction == "energy":
                    Vx = nab - pair * I
                else:
                    raise ValueError(subtraction)
                self.V.append(U * Vx)
            else:
                self.V.append(np.zeros((d, d), dtype=complex))
        self.B, self.Bsea = {}, {}
        D = lat["D"]
        for (i, j) in bonds_of(lat):
            Bij = np.zeros((d, d), dtype=complex)
            sea = 0.0
            for a in range(ncp):
                for b in range(ncp):
                    Dab = D[mode(i, a), mode(j, b)]
                    if Dab != 0:
                        Bij += Dab * hopbar(mode(i, a), mode(j, b))
                        sea += Dab * G[mode(i, a), mode(j, b)]
            self.B[(i, j)] = Bij
            self.Bsea[(i, j)] = complex(sea)
        self.d = d

    def site_bonds(self, x):
        return [b for b in self.B if x in b]

    def T_half(self, x):
        """T_x = (1/2) sum_{b in x} (B_b + B_b^dag)."""
        out = np.zeros((self.d, self.d), dtype=complex)
        for b in self.site_bonds(x):
            out += 0.5 * (self.B[b] + self.B[b].conj().T)
        return out

    def scalar_source(self, x, s):
        """e_x(s) at scalar rulers s (vector): the common S -> s limit of every promotion."""
        out = self.M[x].copy()
        for (i, j) in self.site_bonds(x):
            Tb = self.B[(i, j)] + self.B[(i, j)].conj().T
            out += 0.5 * Tb / (s[i] * s[j])
        out += s[x] ** -6 * self.V[x]
        return out


def single_sector_exprs(promotion, co, nsite):
    """BExpr per site for one sector, blocks y = x."""
    ex = []
    for x in range(nsite):
        e = BExpr(co.d).add(1.0, ('M', co.M[x]))
        if promotion in ("J_W6", "J_CVC"):
            Tx = co.T_half(x)
            e.add(0.5, ('S', x, -2), ('M', Tx)).add(0.5, ('M', Tx), ('S', x, -2))
        elif promotion == "END":
            for (i, j) in co.site_bonds(x):
                Bij = co.B[(i, j)]
                e.add(0.5, ('S', i, -1), ('M', Bij), ('S', j, -1))
                e.add(0.5, ('S', j, -1), ('M', Bij.conj().T), ('S', i, -1))
        elif promotion == "SJ":
            for (i, j) in co.site_bonds(x):
                Tb = co.B[(i, j)] + co.B[(i, j)].conj().T
                for f1, f2 in (((i, -1), (j, -1)), ((j, -1), (i, -1))):
                    e.add(0.125, ('S',) + f1, ('S',) + f2, ('M', Tb))
                    e.add(0.125, ('M', Tb), ('S',) + f1, ('S',) + f2)
        else:
            raise ValueError(promotion)
        if promotion in ("J_W6", "SJ"):
            e.add(0.5, ('S', x, -6), ('M', co.V[x])).add(0.5, ('M', co.V[x]), ('S', x, -6))
        else:
            e.add(1.0, ('S', x, -3), ('M', co.V[x]), ('S', x, -3))
        ex.append(e)
    return ex


def fi_chain_exprs(lat, fock, Ntop, cos):
    """Field-interposed sources on the sector chain N = 0..Ntop; blocks y = (x, N) in order N-major.
    cos[N] = Coefficients of sector N.  Returns exprs, blocks, dims."""
    nsite, ncp = lat["n"], lat["ncomp"]
    blocks = [(x, N) for N in range(Ntop + 1) for x in range(nsite)]
    yi = {b: i for i, b in enumerate(blocks)}
    dims = {N: cos[N].d for N in range(Ntop + 1)}
    D = lat["D"]
    mode = lambda x, a: x * ncp + a
    exprs = []
    for (x, N) in blocks:
        co = cos[N]
        e = BExpr(co.d).add(1.0, ('M', co.M[x]))
        for (i, j) in co.site_bonds(x):
            if N >= 1:
                for a in range(ncp):
                    for b in range(ncp):
                        Dab = D[mode(i, a), mode(j, b)]
                        if Dab == 0:
                            continue
                        # c_i^dag A_i A_j c_j (rulers of sector N - 1) and its conjugate c_j^dag A_j A_i c_i
                        e.add(0.5 * Dab, ('M', fock.cdag(N, mode(i, a))), ('S', yi[(i, N - 1)], -1),
                              ('S', yi[(j, N - 1)], -1), ('M', fock.ann(N, mode(j, b))))
                        e.add(0.5 * np.conj(Dab), ('M', fock.cdag(N, mode(j, b))), ('S', yi[(j, N - 1)], -1),
                              ('S', yi[(i, N - 1)], -1), ('M', fock.ann(N, mode(i, a))))
            g = co.Bsea[(i, j)]
            e.add(-0.5 * g, ('S', yi[(i, N)], -1), ('S', yi[(j, N)], -1))
            e.add(-0.5 * np.conj(g), ('S', yi[(j, N)], -1), ('S', yi[(i, N)], -1))
        e.add(1.0, ('S', yi[(x, N)], -3), ('M', co.V[x]), ('S', yi[(x, N)], -3))
        exprs.append(e)
    return exprs, blocks, dims


def right_placement_e(lat, fock, N, co, G, U, subtraction, Svals):
    """Literal right placement on sector N at given rulers: q_ia = c_ia A_i, F_i = c_i2 c_i1 C_i,
    H_b^R = q_i^dag D_ij q_j + h.c. with counterterms in the same placement; returns e_x for comparison with END."""
    nsite, ncp = lat["n"], lat["ncomp"]
    mode = lambda x, a: x * ncp + a
    w = [np.linalg.eigh(s) for s in Svals]
    pw = lambda x, p: (w[x][1] * w[x][0] ** p) @ w[x][1].conj().T
    D = lat["D"]
    d = co.d
    I = np.eye(d, dtype=complex)
    q = {(x, a): fock.ann(N, mode(x, a)) @ pw(x, -1) for x in range(nsite) for a in range(ncp)}
    out = []
    for x in range(nsite):
        e = co.M[x].copy()
        for (i, j) in co.site_bonds(x):
            Hb = np.zeros((d, d), dtype=complex)
            for a in range(ncp):
                for b in range(ncp):
                    Dab = D[mode(i, a), mode(j, b)]
                    if Dab != 0:
                        Hb += Dab * (q[(i, a)].conj().T @ q[(j, b)] - G[mode(i, a), mode(j, b)] * pw(i, -1) @ pw(j, -1))
            e += 0.5 * (Hb + Hb.conj().T)
        a, b = mode(x, 0), mode(x, 1)
        F = fock.ann(N - 1, a) @ fock.ann(N, b) @ pw(x, -3) if N >= 2 else None  # c_a c_b C (= c_i2 c_i1 C up to sign)
        FF = F.conj().T @ F if F is not None else np.zeros((d, d), dtype=complex)
        C = pw(x, -3)
        Gab = lambda u, v: G[u, v]
        if subtraction == "wick":
            ct = (-Gab(a, a) * (fock.sector(N).hop(b, b) - Gab(b, b) * I) - Gab(b, b) * (fock.sector(N).hop(a, a) - Gab(a, a) * I)
                  + Gab(a, b) * (fock.sector(N).hop(b, a) - Gab(b, a) * I) + Gab(b, a) * (fock.sector(N).hop(a, b) - Gab(a, b) * I)
                  - (Gab(a, a) * Gab(b, b) - Gab(a, b) * Gab(b, a)) * I)
        else:
            ct = -(Gab(a, a) * Gab(b, b) - Gab(a, b) * Gab(b, a)) * I
        e += U * (FF + C @ ct @ C)
        out.append(e)
    return out


# ------------------------------------------------------------------ pinching
def resolving_operator(lat, sec):
    """Generic translation-invariant one-body operator dGamma(o) used to pick a basis inside degenerate blocks."""
    n, ncp = lat["n"], lat["ncomp"]
    if lat["name"] == "ring4":
        shifts = [(lambda x: (x + 1) % 4)]
        wts = [(1.0, 0.37)]
    elif lat["name"] == "plaq":
        shifts = [(lambda x: (x ^ 1)), (lambda x: (x ^ 2))]
        wts = [(1.0, 0.37), (0.61, 0.23)]
    else:
        raise ValueError(lat["name"])
    o = np.zeros((n * ncp, n * ncp), dtype=complex)
    for f, (c1, c2) in zip(shifts, wts):
        t = np.zeros((n, n))
        for x in range(n):
            t[f(x), x] = 1.0
        t = np.kron(t, np.eye(ncp))
        o += c1 * 0.5 * (t + t.T) + c2 * 0.5j * (t - t.T)
    return sec.bilinear(nc.herm(o))


def pinning_basis(Hmat, Ores, tol=1e-9):
    """Eigenbasis of Hmat, degenerate blocks resolved by diagonalizing Ores inside each block."""
    w, V, blocks = nc.energy_blocks(Hmat, tol)
    cols = []
    for bl in blocks:
        Vb = V[:, bl]
        if len(bl) > 1:
            o = Vb.conj().T @ Ores @ Vb
            _, u = np.linalg.eigh(nc.herm(o))
            Vb = Vb @ u
        cols.append(Vb)
    return np.concatenate(cols, axis=1)


# ------------------------------------------------------------------ 2 x 2 analytic control
def two_by_two_control(s0=1.1, z=0.2, eps=1e-3):
    S = np.diag([s0 + z, s0 - z])
    A = np.linalg.inv(S)
    T = eps * nc.SX.real
    jor = 0.5 * (A @ A @ T + T @ A @ A)
    sand = A @ T @ A
    lhs = jor - sand
    rhs = 0.5 * (A @ (A @ T - T @ A) - (A @ T - T @ A) @ A)
    lam = np.diag(S)
    # pinching in the common eigenbasis (S diagonal): Pi_a e(lambda_a) Pi_a has no off-diagonal entries
    pin = np.diag([(np.diag([1 / l ** 2] * 2) @ T)[a, a] for a, l in enumerate(lam)])
    return {"identity_err": float(np.abs(lhs - rhs).max()), "ratio": float(sand[0, 1] / jor[0, 1]),
            "ratio_pred": (s0 ** 2 - z ** 2) / (s0 ** 2 + z ** 2), "pin_offdiag": float(abs(pin[0, 1]))}
