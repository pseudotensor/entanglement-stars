#!/usr/bin/env python3
"""
j4lib.py -- shared routines for J4, the genuine 3-dimensional ball cut (busi lane, task 1).
Specification: busi/specs/J4_spec_v1.md.  Python 3, numpy, scipy only.

Setting: the flat massless half-filled ground state of the paper's 4-component sine operator
h_0 = sum_i alpha_i p_i, p_i = (T_i - T_i^dagger)/(2i), on an even cubic box of side L with
antiperiodic boundary conditions on all 3 axes (lattice units a = 1), and the entanglement
entropy (nats) of the site-centred digital ball A = {x in Z^3 : |x|^2 <= n2max},
n2max = ceil(R_eff^2) - 1, i.e. |x| < R_eff.

Exact reductions implemented here (derivations in the spec, checks in j4_tests.py):
  (1) kernel structure: C(x,y) = delta/2 - (i/2) sum_i G_i(x-y) alpha_i with G_i real, odd and
      supported on displacements whose i-th component is odd and the other 2 even;
  (2) on-site transform Omega_x = alpha_1^{x_1} alpha_2^{x_2} alpha_3^{x_3}: the 4 components
      become 4 identical copies of the scalar staggered correlation C_st = (1 - iY)/2,
      Y(x,y) = sum_i eta_i(x) G_i(x-y), eta_i(x) = (-1)^{x_1+...+x_{i-1}}; S = 4 S_scalar;
  (3) the gauge-signed cubic group (V_g f)(x) = phi_g(x) f(g^{-1} x), an ordinary representation
      of O_h (48 elements) commuting with C_st; irrep blocks with multiplicities d_lambda;
  (4) even/odd sublattice structure of Y: S_scalar = sum_lambda d_lambda [sum_j 2 s((1+s_j)/2)
      + |m_e - m_o| ln 2], s_j the singular values of the even-odd block K_lambda;
  (5) assembly of K_lambda directly from the kernel by a 48-term group sum per pair of orbits.
"""
import hashlib
import os
import itertools
import time

import numpy as np
from scipy import linalg, special

LN2 = float(np.log(2.0))
C_LOG_CONTINUUM = -44.0 / 45.0
IRREP_NAMES = ['A1+', 'A2+', 'E+', 'T1+', 'T2+', 'A1-', 'A2-', 'E-', 'T1-', 'T2-']
IRREP_DIM = {'A1': 1, 'A2': 1, 'E': 2, 'T1': 3, 'T2': 3}


def irrep_dim(name):
    return IRREP_DIM[name[:-1]]


# ----------------------------------------------------------------------------------------------
# Clifford matrices.  Any representation gives the same entropies; this is the Dirac one.
# ----------------------------------------------------------------------------------------------
def dirac_matrices():
    """alpha_i = tau_x (x) sigma_i, beta = tau_z (x) 1: {alpha_i, alpha_j} = 2 delta, {alpha_i, beta} = 0."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    alpha = [np.kron(sx, s) for s in (sx, sy, sz)]
    beta = np.kron(sz, np.eye(2, dtype=complex))
    return alpha, beta


def omega_matrix(parity, alpha):
    """Omega_x = alpha_1^{x_1} alpha_2^{x_2} alpha_3^{x_3} (left to right); depends on x mod 2 only."""
    m = np.eye(4, dtype=complex)
    for i in range(3):
        if int(parity[i]) % 2:
            m = m @ alpha[i]
    return m


# ----------------------------------------------------------------------------------------------
# The cubic group O_h as 48 signed permutation matrices
# ----------------------------------------------------------------------------------------------
def cubic_group():
    """int64 array (48, 3, 3); element 0 is the identity.  (g x)_i = g[i, perm[i]] x_{perm[i]}."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            g = np.zeros((3, 3), dtype=np.int64)
            for i in range(3):
                g[i, perm[i]] = signs[i]
            out.append(g)
    G = np.array(out)
    if not np.array_equal(G[0], np.eye(3, dtype=np.int64)):
        raise ArithmeticError('element 0 is not the identity')
    return G


def group_tables(G):
    """mult[a, b] = index of G[a] @ G[b]; inv[a] = index of G[a]^{-1} = G[a]^T.  Raises if not closed."""
    key = {np.ascontiguousarray(G[i]).tobytes(): i for i in range(len(G))}
    if len(key) != len(G):
        raise ArithmeticError('group elements are not distinct')
    n = len(G)
    mult = np.zeros((n, n), dtype=np.int64)
    for a in range(n):
        for b in range(n):
            mult[a, b] = key[np.ascontiguousarray(G[a] @ G[b]).tobytes()]
    inv = np.array([key[np.ascontiguousarray(G[a].T).tobytes()] for a in range(n)], dtype=np.int64)
    return mult, inv


def parity_class(x):
    """Code c = (x_1 mod 2) + 2 (x_2 mod 2) + 4 (x_3 mod 2) for an int array x of shape (..., 3)."""
    x = np.asarray(x)
    return (x[..., 0] % 2) + 2 * (x[..., 1] % 2) + 4 * (x[..., 2] % 2)


def class_bits(c):
    c = int(c)
    return np.array([(c >> 0) & 1, (c >> 1) & 1, (c >> 2) & 1], dtype=np.int64)


def scalar_phase_table(G):
    """phi[g, c] = sign with Omega_x^dagger rho_g(Omega_{g^{-1} x}) = phi_g(x) I_4 for x in parity class c,
    rho_g(alpha_i) = sum_j g_{ji} alpha_j.  int64 array (48, 8).  Raises if any matrix is not +-I_4."""
    alpha, _ = dirac_matrices()
    phi = np.zeros((len(G), 8), dtype=np.int64)
    eye = np.eye(4)
    for gi, g in enumerate(G):
        rho_alpha = [sum(int(g[j, i]) * alpha[j] for j in range(3)) for i in range(3)]
        for c in range(8):
            p = class_bits(c)
            y = (p @ g) % 2                       # parity of g^{-1} x = g^T x
            m = np.eye(4, dtype=complex)
            for i in range(3):
                if y[i]:
                    m = m @ rho_alpha[i]
            m = omega_matrix(p, alpha).conj().T @ m
            ph = float(np.trace(m).real) / 4.0
            if abs(abs(ph) - 1.0) > 1e-12 or np.abs(m - ph * eye).max() > 1e-12:
                raise ArithmeticError('scalar phase is not a sign for g=%d, class=%d' % (gi, c))
            phi[gi, c] = int(round(ph))
    return phi


def scalar_phase_closed_form(G):
    """The closed form of the spec (section 2.3), for the test against the Clifford computation."""
    phi = np.zeros((len(G), 8), dtype=np.int64)
    for gi, g in enumerate(G):
        jrow = np.argmax(np.abs(g), axis=0)        # jrow[i] = row of the nonzero entry in column i
        s = np.array([g[jrow[i], i] for i in range(3)])
        for c in range(8):
            p = class_bits(c)
            val = 1
            for i in range(3):
                if p[jrow[i]] % 2:
                    val *= int(s[i])
            for i in range(3):
                for i2 in range(i + 1, 3):
                    if jrow[i] > jrow[i2] and p[jrow[i]] % 2 and p[jrow[i2]] % 2:
                        val = -val
            phi[gi, c] = val
    return phi


def permutation_sign(perm):
    inv = 0
    for i in range(3):
        for j in range(i + 1, 3):
            if perm[i] > perm[j]:
                inv += 1
    return -1.0 if inv % 2 else 1.0


def irrep_matrices(G):
    """Real orthogonal irreps of O_h: name -> array (48, d, d).  g = det(g) r with r proper;
    a2(g) = sign of the coordinate permutation; A1: 1, A2: a2, E: B^T |g| B on the traceless
    diagonal quadratic forms, T1: r, T2: a2 r; the '-' irreps multiply by -1 on improper g."""
    B = np.array([[1 / np.sqrt(2.0), 1 / np.sqrt(6.0)],
                  [-1 / np.sqrt(2.0), 1 / np.sqrt(6.0)],
                  [0.0, -2 / np.sqrt(6.0)]])
    out = {name: [] for name in IRREP_NAMES}
    for g in G:
        det = int(round(np.linalg.det(g)))
        r = (det * g).astype(float)
        perm = np.argmax(np.abs(g), axis=1)
        a2 = permutation_sign(perm)
        base = {'A1': np.ones((1, 1)), 'A2': np.array([[a2]]), 'E': B.T @ np.abs(r) @ B,
                'T1': r, 'T2': a2 * r}
        for name in IRREP_NAMES:
            s = -1.0 if (name[-1] == '-' and det == -1) else 1.0
            out[name].append(s * base[name[:-1]])
    return {k: np.array(v) for k, v in out.items()}


def irrep_characters(D):
    return {name: np.einsum('kii->k', M) for name, M in D.items()}


# ----------------------------------------------------------------------------------------------
# The ball
# ----------------------------------------------------------------------------------------------
def ball_n2max(R_eff):
    """|x|^2 < R_eff^2  <=>  |x|^2 <= ceil(R_eff^2) - 1 (exact for integer, half-integer and sixteenth radii)."""
    return int(np.ceil(float(R_eff) ** 2)) - 1


def ball_sites(R_eff=None, n2max=None):
    """Sites x in Z^3 with |x|^2 <= n2max, centre at the origin; int64 (n, 3) in lexicographic order."""
    if n2max is None:
        n2max = ball_n2max(R_eff)
    if n2max < 0:
        return np.zeros((0, 3), dtype=np.int64)
    M = int(np.floor(np.sqrt(n2max)))
    ax = np.arange(-M, M + 1, dtype=np.int64)
    X1, X2, X3 = np.meshgrid(ax, ax, ax, indexing='ij')
    pts = np.stack([X1.ravel(), X2.ravel(), X3.ravel()], axis=1)
    n2 = np.sum(pts * pts, axis=1)
    return np.ascontiguousarray(pts[n2 <= n2max])


def sites_hash(sites):
    return hashlib.sha256(np.ascontiguousarray(sites, dtype=np.int64).tobytes()).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------
# Entropy functions
# ----------------------------------------------------------------------------------------------
def binary_entropy(x):
    """s(x) = -x ln x - (1-x) ln(1-x), elementwise on [0, 1] (tiny excursions clipped)."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return -special.xlogy(x, x) - special.xlog1py(1.0 - x, -x)


def pair_entropy_from_t(t):
    """2 s((1+s)/2) written in t = 1 - s in [0, 1]:  -(2 - t) log1p(-t/2) - t ln(t/2)."""
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    out = -(2.0 - t) * np.log1p(-0.5 * t)
    pos = t > 0
    out[pos] -= t[pos] * np.log(0.5 * t[pos])
    return out


def pair_entropy_from_singular(s):
    return pair_entropy_from_t(1.0 - np.asarray(s, dtype=float))


def pair_entropy_from_gram(ev):
    """ev = s^2 (eigenvalues of K K^T); t = 1 - s = (1 - ev)/(1 + sqrt ev)."""
    ev = np.clip(np.asarray(ev, dtype=float), 0.0, 1.0)
    return pair_entropy_from_t((1.0 - ev) / (1.0 + np.sqrt(ev)))


def sublattice_entropy(singular_values, m_e, m_o):
    """S_scalar = sum_j 2 s((1+s_j)/2) + |m_e - m_o| ln 2 for one scalar staggered copy."""
    return float(np.sum(pair_entropy_from_singular(singular_values)) + abs(int(m_e) - int(m_o)) * LN2)


# ----------------------------------------------------------------------------------------------
# The massless antiperiodic kernel
# ----------------------------------------------------------------------------------------------
class MasslessKernel:
    """Massless antiperiodic box of even side L.  G_1(n) = Im (1/L^3) sum_k (sin k_1/eps(k)) e^{ik.n},
    k_i = 2 pi (j_i + 1/2)/L, tabulated on n in [0, L)^3; other n by antiperiodicity, other components
    by G_2(r) = G_1(r_2, r_1, r_3), G_3(r) = G_1(r_3, r_2, r_1).  C(x,y) = delta/2 - (i/2) sum_i G_i(x-y) alpha_i."""

    def __init__(self, L):
        L = int(L)
        if L % 2 or L < 4:
            raise ValueError('L must be even and at least 4')
        self.L = L
        k = 2.0 * np.pi * (np.arange(L) + 0.5) / L
        s = np.sin(k)
        eps2 = s[:, None, None] ** 2 + s[None, :, None] ** 2 + s[None, None, :] ** 2
        np.sqrt(eps2, out=eps2)
        f1 = s[:, None, None] / eps2
        del eps2
        F1 = np.fft.ifftn(f1)                      # (1/L^3) sum_j f(k_j) e^{2 pi i j.n/L}
        del f1
        ph = np.exp(1j * np.pi * np.arange(L) / L)  # e^{i k.n} = e^{2 pi i j.n/L} e^{i pi sum n/L}
        F1 *= ph[:, None, None]
        F1 *= ph[None, :, None]
        F1 *= ph[None, None, :]
        self.real_part_max = float(np.abs(F1.real).max())
        self.G1 = np.ascontiguousarray(F1.imag)
        del F1

    def g1(self, r):
        """G_1 at integer displacements r (..., 3) of any size, with the antiperiodic sign."""
        r = np.asarray(r, dtype=np.int64)
        wraps = np.sum(r // self.L, axis=-1)
        sign = 1.0 - 2.0 * (wraps % 2)
        idx = r % self.L
        return self.G1[idx[..., 0], idx[..., 1], idx[..., 2]] * sign

    def g_component(self, i, r):
        """G_i(r) through the relabelling of components (i = 0, 1, 2)."""
        r = np.asarray(r, dtype=np.int64)
        if i == 0:
            return self.g1(r)
        if i == 1:
            return self.g1(r[..., [1, 0, 2]])
        return self.g1(r[..., [2, 1, 0]])

    def staggered_table(self, M):
        """Gtab(r) = G_{i(r)}(r) for r in [-M, M]^3 with exactly 1 odd component, else 0.
        Returns (table (2M+1,)^3, max |discarded value|); the discarded values vanish by the
        sublattice structure of the massless kernel (section 2.1 of the spec)."""
        T = 2 * M + 1
        ax = np.arange(-M, M + 1, dtype=np.int64)
        tab = np.zeros((T, T, T))
        junk = 0.0
        R2, R3 = np.meshgrid(ax, ax, indexing='ij')
        for a1, r1 in enumerate(ax):
            r = np.stack([np.full_like(R2, r1), R2, R3], axis=-1)      # (T, T, 3)
            odd = r % 2
            nodd = odd.sum(axis=-1)
            slab = np.zeros((T, T))
            for i in range(3):
                gi = self.g_component(i, r)
                mask = (nodd == 1) & (odd[..., i] == 1)
                slab[mask] = gi[mask]
                if np.any(~mask):
                    junk = max(junk, float(np.abs(gi[~mask]).max()))
            tab[a1] = slab
        return tab, junk


def stag_sign_table():
    """eta[c, i] = eta_{i+1}(x) = (-1)^{x_1 + ... + x_i} for x in parity class c (i = 0, 1, 2); eta[:, 3] = 0."""
    eta = np.zeros((8, 4))
    for c in range(8):
        p = class_bits(c)
        eta[c, 0] = 1.0
        eta[c, 1] = (-1.0) ** int(p[0])
        eta[c, 2] = (-1.0) ** int(p[0] + p[1])
    return eta


def signed_tables(gtab, M, classes):
    """Ytab_c(r) = eta_{i(r)}(c) Gtab(r) for each parity class c in `classes`: dict c -> flat (T^3,) array."""
    T = 2 * M + 1
    ax = np.arange(-M, M + 1, dtype=np.int64) % 2
    o1 = ax[:, None, None]
    o2 = ax[None, :, None]
    o3 = ax[None, None, :]
    nodd = o1 + o2 + o3
    iodd = np.where(nodd == 1, o2 + 2 * o3, 3)        # 0, 1, 2 = index of the odd component; 3 otherwise
    eta = stag_sign_table()
    out = {}
    for c in classes:
        out[int(c)] = np.ascontiguousarray((gtab * eta[int(c)][iodd]).ravel())
    return out


# ----------------------------------------------------------------------------------------------
# Orbits, multiplicities and the block assembly
# ----------------------------------------------------------------------------------------------
class BallSymmetry:
    """Site-centred ball, its O_h orbits under the gauge-signed action, and the irrep bookkeeping."""

    def __init__(self, sites, G, phi, D):
        self.sites = np.ascontiguousarray(sites, dtype=np.int64)
        self.n = int(len(self.sites))
        self.G, self.phi, self.D = G, phi, D
        par = self.sites.sum(axis=1) % 2
        self.n_e = int(np.sum(par == 0))
        self.n_o = int(np.sum(par == 1))
        self.M = 2 * int(np.abs(self.sites).max()) if self.n else 0
        reps = np.unique(np.sort(np.abs(self.sites), axis=1)[:, ::-1], axis=0)   # a >= b >= c >= 0
        self.reps = np.ascontiguousarray(reps, dtype=np.int64)
        self.imgs = np.einsum('kij,rj->rki', G, self.reps)                        # (N, 48, 3): g_k y
        self.sg = phi[np.arange(len(G))[None, :], parity_class(self.imgs)]       # phi_k(k y)
        self.stab = np.all(self.imgs == self.reps[:, None, :], axis=2)            # k y == y
        self.stab_size = self.stab.sum(axis=1)
        if np.any(len(G) % self.stab_size):
            raise ArithmeticError('stabilizer order does not divide 48')
        if int(np.sum(len(G) // self.stab_size)) != self.n:
            raise ArithmeticError('orbit sizes do not sum to the site count')
        self.rep_par = self.reps.sum(axis=1) % 2
        self.rep_class = parity_class(self.reps)
        even = np.where(self.rep_par == 0)[0]
        odd = np.where(self.rep_par == 1)[0]
        # even representatives grouped by parity class (stable): the rows of each block are then
        # contiguous per class, which the assembly needs
        self.even = even[np.argsort(self.rep_class[even], kind='stable')]
        self.odd = odd
        self.chi_V = self._fixed_point_characters()
        self._irrep_cache = {}

    def _fixed_point_characters(self):
        """chi[g, p] = sum over sites x of parity p fixed by g of phi_g(x)  (the trace of V_g on the sublattice)."""
        cls = parity_class(self.sites)
        chi = np.zeros((len(self.G), 2))
        for gi, g in enumerate(self.G):
            fixed = np.all(self.sites @ g.T == self.sites, axis=1)
            counts = np.bincount(cls[fixed], minlength=8)
            for c in range(8):
                p = int(np.sum(class_bits(c)) % 2)
                chi[gi, p] += counts[c] * self.phi[gi, c]
        return chi

    def character_multiplicities(self, name):
        """(m_even, m_odd) of irrep `name` by characters; raises if not integral."""
        chi_l = np.einsum('kii->k', self.D[name])
        m = chi_l @ self.chi_V / len(self.G)
        mi = np.rint(m).astype(np.int64)
        if np.abs(m - mi).max() > 1e-8:
            raise ArithmeticError('non-integral multiplicity for ' + name)
        return int(mi[0]), int(mi[1])

    def irrep_data(self, name):
        """Per-representative factors for irrep `name` (row 0 of the irrep is used).
        Pi_y = sum_{h in Stab(y)} phi_h(y) D(h) has eigenvalues 0 and |Stab(y)|; W_y = its eigenvectors
        of eigenvalue |Stab(y)| (d x m_y); fac[y] = W_y / sqrt|Stab(y)| padded to d x d with zero columns.
        The symmetry-adapted block is fac[x]^T M(x,y) fac[y]."""
        if name in self._irrep_cache:
            return self._irrep_cache[name]
        Dm = self.D[name]
        d = Dm.shape[1]
        Pi = np.einsum('rk,kij->rij', (self.stab * self.sg).astype(float), Dm)
        w, V = np.linalg.eigh(Pi)
        H = self.stab_size.astype(float)[:, None]
        dist = np.minimum(np.abs(w), np.abs(w - H))
        if dist.max() > 1e-9:
            raise ArithmeticError('stabilizer projector spectrum is not {0, |H|} for ' + name)
        keep = w > 0.5 * H
        m_rep = keep.sum(axis=1).astype(np.int64)
        fac = np.zeros((len(self.reps), d, d))
        inv_sqrt = 1.0 / np.sqrt(self.stab_size.astype(float))
        for r in range(len(self.reps)):
            mr = int(m_rep[r])
            if mr:
                fac[r, :, :mr] = V[r][:, keep[r]] * inv_sqrt[r]
        m_e = int(m_rep[self.even].sum())
        m_o = int(m_rep[self.odd].sum())
        ce, co = self.character_multiplicities(name)
        if (m_e, m_o) != (ce, co):
            raise ArithmeticError('projector multiplicities %s disagree with characters %s for %s'
                                  % ((m_e, m_o), (ce, co), name))
        data = dict(name=name, d=d, m_rep=m_rep, fac=fac, m_e=m_e, m_o=m_o)
        self._irrep_cache[name] = data
        return data

    def assemble_blocks(self, names, ytabs, chunk_elems=1.6e7, log=None):
        """Assemble the even x odd blocks K_lambda for the irreps in `names` in one pass over the
        orbit-representative pairs.  ytabs: dict parity class -> flat signed kernel table (signed_tables).
        Returns dict name -> float64 array (m_e, m_o).
            K[(x,j),(y,j')] = [W_x^T M(x,y) W_y]_{jj'} / sqrt(|H_x| |H_y|),
            M(x,y) = sum_k D(k) phi_k(k y) Y(x, k y),  Y(x,z) = eta_{i(x-z)}(x) G_{i(x-z)}(x - z)."""
        nG = len(self.G)
        T = 2 * self.M + 1
        const = self.M * (T * T + T + 1)

        def code(v):
            return v[..., 0] * (T * T) + v[..., 1] * T + v[..., 2]

        odd, even = self.odd, self.even
        N_o = len(odd)
        zcode = code(self.imgs[odd])                          # (N_o, 48)
        sgo = self.sg[odd].astype(float)                      # (N_o, 48)
        data = {nm: self.irrep_data(nm) for nm in names}
        K = {nm: np.zeros((data[nm]['m_e'], data[nm]['m_o'])) for nm in names}
        Dflat = {nm: np.ascontiguousarray(self.D[nm].reshape(nG, -1)) for nm in names}
        colsel, rfac, row_off = {}, {}, {}
        for nm in names:
            d = data[nm]['d']
            mr_o = data[nm]['m_rep'][odd]
            colsel[nm] = np.concatenate([np.arange(b * d, b * d + mr_o[b]) for b in range(N_o)]
                                        ).astype(np.int64) if N_o else np.zeros(0, dtype=np.int64)
            rfac[nm] = np.ascontiguousarray(data[nm]['fac'][odd])          # (N_o, d, d)
            row_off[nm] = np.concatenate([[0], np.cumsum(data[nm]['m_rep'][even])]).astype(np.int64)
        A = max(1, int(chunk_elems // max(1, N_o * nG)))
        t0 = time.time()
        done = 0
        for c in np.unique(self.rep_class[even]):
            ytab = ytabs[int(c)]
            sel = np.where(self.rep_class[even] == c)[0]      # contiguous positions within `even`
            for s0 in range(0, len(sel), A):
                pos = sel[s0:s0 + A]
                xcode = code(self.reps[even[pos]])                        # (A,)
                idx = xcode[:, None, None] - zcode[None, :, :] + const    # displacement x - k y
                vals = ytab[idx]
                vals *= sgo[None, :, :]
                vflat = vals.reshape(-1, nG)
                for nm in names:
                    d = data[nm]['d']
                    Mb = (vflat @ Dflat[nm]).reshape(len(pos), N_o, d, d)
                    lfacT = np.ascontiguousarray(np.transpose(data[nm]['fac'][even[pos]], (0, 2, 1)))
                    Mb = np.matmul(lfacT[:, None, :, :], Mb)
                    Mb = np.matmul(Mb, rfac[nm][None, :, :, :])
                    blk = np.transpose(Mb, (0, 2, 1, 3)).reshape(len(pos) * d, N_o * d)[:, colsel[nm]]
                    mr_e = data[nm]['m_rep'][even[pos]]
                    rowsel = np.concatenate([np.arange(a * d, a * d + mr_e[a]) for a in range(len(pos))]
                                            ).astype(np.int64)
                    r0 = int(row_off[nm][pos[0]])
                    r1 = int(row_off[nm][pos[-1] + 1])
                    K[nm][r0:r1, :] = blk[rowsel, :]
                done += len(pos)
                if log is not None and (done % (50 * A) < len(pos) or done == len(even)):
                    log('  assembly %s: %d/%d even orbits, %.0f s' % ('+'.join(names), done, len(even), time.time() - t0))
        return K


def plan_irrep_groups(sym, names, method, mem_limit_bytes):
    """Greedy first-fit grouping of irreps: sum of K bytes + largest Gram bytes per group <= limit."""
    info = []
    for nm in names:
        dat = sym.irrep_data(nm)
        kb = 8 * dat['m_e'] * dat['m_o']
        gb = 8 * min(dat['m_e'], dat['m_o']) ** 2 if method == 'gram' else 0
        info.append((kb, gb, nm))
    info.sort(reverse=True)
    groups = []
    for kb, gb, nm in info:
        placed = False
        for grp in groups:
            tot_k = sum(x[0] for x in grp) + kb
            mx_g = max([x[1] for x in grp] + [gb])
            if tot_k + mx_g <= mem_limit_bytes:
                grp.append((kb, gb, nm))
                placed = True
                break
        if not placed:
            groups.append([(kb, gb, nm)])
    return [[x[2] for x in grp] for grp in groups]


def block_pair_entropy(K, method='gram'):
    """(sum_j 2 s((1+s_j)/2), singular values) for the real block K (m_e x m_o).  K may be destroyed.
    'gram': eigenvalues of the smaller Gram matrix (dsyrk, no copy, then dsyevd in place);
    'svd' : LAPACK gesdd singular values of K."""
    me, mo = K.shape
    if me == 0 or mo == 0:
        return 0.0, np.zeros(0)
    if method == 'svd':
        s = linalg.svdvals(K.T, overwrite_a=True, check_finite=False)   # K.T is Fortran-ordered: no copy
        return float(np.sum(pair_entropy_from_singular(s))), np.asarray(s)
    if method == 'gram':
        from scipy.linalg import blas
        KT = K.T
        if me <= mo:
            Gm = blas.dsyrk(1.0, KT, trans=1, lower=0)    # (K^T)^T K^T = K K^T, me x me, upper triangle
        else:
            Gm = blas.dsyrk(1.0, KT, trans=0, lower=0)    # K^T K, mo x mo
        ev = linalg.eigvalsh(Gm, lower=False, overwrite_a=True, driver='evd', check_finite=False)
        del Gm
        ev = np.asarray(ev, dtype=float)
        ev_min, ev_max = float(ev.min()), float(ev.max())
        if ev_min < -1e-10 or ev_max > 1.0 + 1e-10:
            raise ArithmeticError('Gram spectrum outside [0, 1]: %g, %g' % (ev_min, ev_max))
        ev = np.clip(ev, 0.0, 1.0)
        return float(np.sum(pair_entropy_from_gram(ev))), np.sqrt(ev)
    if method == 'cupy':
        import cupy as cp                                    # optional GPU path; identical arithmetic
        Kd = cp.asarray(K)
        Gm = Kd @ Kd.T if me <= mo else Kd.T @ Kd
        del Kd
        ev = cp.asnumpy(cp.linalg.eigvalsh(Gm))
        del Gm
        ev = np.clip(np.asarray(ev, dtype=float), 0.0, 1.0)
        return float(np.sum(pair_entropy_from_gram(ev))), np.sqrt(ev)
    if method == 'torch':
        # GPU path on the lane's H100 (builder's addition, the cupy path's arithmetic with torch, FP64):
        # the Gram matrix of the smaller side and its eigenvalues by cuSOLVER syevd.
        # One process at a time on the GPU (the R = 62 Gram solve peaks near 52 GB of the 80 GB): an exclusive
        # file lock around the GPU section; the other workers keep assembling their blocks on the CPU meanwhile.
        import fcntl
        import torch
        with open(os.environ.get('J4_GPU_LOCK', '<internal>/j4_gpu.lock'), 'w') as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            Kd = torch.from_numpy(np.ascontiguousarray(K)).to('cuda')
            Gm = Kd @ Kd.T if me <= mo else Kd.T @ Kd
            del Kd
            ev = torch.linalg.eigvalsh(Gm).cpu().numpy()
            del Gm
            torch.cuda.empty_cache()
            fcntl.flock(lockf, fcntl.LOCK_UN)
        ev = np.asarray(ev, dtype=float)
        ev_min, ev_max = float(ev.min()), float(ev.max())
        if ev_min < -1e-10 or ev_max > 1.0 + 1e-10:
            raise ArithmeticError('Gram spectrum outside [0, 1]: %g, %g' % (ev_min, ev_max))
        ev = np.clip(ev, 0.0, 1.0)
        return float(np.sum(pair_entropy_from_gram(ev))), np.sqrt(ev)
    raise ValueError('unknown method ' + str(method))


# ----------------------------------------------------------------------------------------------
# The reduced computation
# ----------------------------------------------------------------------------------------------
def reduced_ball_entropy(R_eff=None, n2max=None, L=None, method='gram', mem_limit_gb=16.0,
                         chunk_elems=1.6e7, names=None, keep_values=False, log=None):
    """The symmetry-reduced, sublattice-SVD entropy of the ball |x| < R_eff (or |x|^2 <= n2max) in the
    antiperiodic massless box of side L.  Returns a record dict (and, with keep_values, also the
    singular values per irrep, the BallSymmetry and the MasslessKernel for the tests)."""
    def say(msg):
        if log is not None:
            log(msg)
    timing = {}
    t0 = time.time()
    G = cubic_group()
    phi = scalar_phase_table(G)
    D = irrep_matrices(G)
    if n2max is None:
        n2max = ball_n2max(R_eff)
    if R_eff is None:
        R_eff = float(np.sqrt(n2max + 0.5))
    sites = ball_sites(n2max=n2max)
    if 2.0 * R_eff >= L:
        raise ValueError('the ball must fit in the box: need L > 2 R_eff')
    sym = BallSymmetry(sites, G, phi, D)
    if sym.M >= L:
        raise ValueError('displacement range exceeds the box')
    timing['symmetry_s'] = time.time() - t0
    say('sites %d (even %d, odd %d), orbits %d, M %d, L %d' % (sym.n, sym.n_e, sym.n_o, len(sym.reps), sym.M, L))
    t1 = time.time()
    ker = MasslessKernel(L)
    gtab, junk = ker.staggered_table(sym.M)
    classes = [int(c) for c in np.unique(sym.rep_class[sym.even])]
    ytabs = signed_tables(gtab, sym.M, classes)
    del gtab
    timing['kernel_s'] = time.time() - t1
    say('kernel: real part max %.2e, structure junk %.2e, %.1f s' % (ker.real_part_max, junk, timing['kernel_s']))
    if junk > 1e-13:
        raise ArithmeticError('kernel sublattice structure violated: %g' % junk)
    names = list(IRREP_NAMES) if names is None else list(names)
    irr = {nm: sym.irrep_data(nm) for nm in names}
    groups = plan_irrep_groups(sym, names, method, mem_limit_gb * 1e9)
    say('irrep groups: %s' % groups)
    results, values = {}, {}
    t2 = time.time()
    for grp in groups:
        ta = time.time()
        K = sym.assemble_blocks(grp, ytabs, chunk_elems=chunk_elems, log=log)
        t_asm = time.time() - ta
        for nm in grp:
            dat = irr[nm]
            tb = time.time()
            e_pair, svals = block_pair_entropy(K.pop(nm), method)
            contrib = dat['d'] * (e_pair + abs(dat['m_e'] - dat['m_o']) * LN2)
            results[nm] = dict(irrep=nm, d=dat['d'], m_e=dat['m_e'], m_o=dat['m_o'],
                               pair_entropy=e_pair, scalar_contribution=contrib,
                               field_contribution=4.0 * contrib, n_singular=int(len(svals)),
                               assembly_group_s=t_asm, eigen_s=time.time() - tb)
            if keep_values:
                values[nm] = svals
            say('  %-4s d=%d block %d x %d  4*d*(pair + imbalance ln2) = %.9f  (eigen %.0f s)'
                % (nm, dat['d'], dat['m_e'], dat['m_o'], 4.0 * contrib, results[nm]['eigen_s']))
    timing['blocks_s'] = time.time() - t2
    timing['total_s'] = time.time() - t0
    S_scalar = float(sum(r['scalar_contribution'] for r in results.values()))
    S = 4.0 * S_scalar
    full = set(names) == set(IRREP_NAMES)
    dim_ok = (sum(r['d'] * r['m_e'] for r in results.values()) == sym.n_e and
              sum(r['d'] * r['m_o'] for r in results.values()) == sym.n_o) if full else None
    largest = max(((r['m_e'], r['m_o']) for r in results.values()), key=lambda t: max(t), default=(0, 0))
    rec = dict(R_eff=float(R_eff), n2max=int(n2max), L=int(L), n=sym.n, n_e=sym.n_e, n_o=sym.n_o,
               sites_hash=sites_hash(sites), M_disp=int(sym.M), orbits=int(len(sym.reps)),
               kernel_real_part_max=ker.real_part_max, kernel_structure_junk=float(junk),
               irreps=[results[nm] for nm in names], irrep_groups=groups, method=method,
               S_scalar=S_scalar, S=float(S), S_over_4piR2=float(S / (4.0 * np.pi * R_eff ** 2)),
               largest_block=[int(largest[0]), int(largest[1])], dimension_check=dim_ok, timing=timing)
    if full and not dim_ok:
        raise ArithmeticError('sum of d_lambda m_lambda is not the sublattice site count')
    if keep_values:
        return rec, values, sym, ker
    return rec


# ----------------------------------------------------------------------------------------------
# Dense scalar routines (tests and --check-dense)
# ----------------------------------------------------------------------------------------------
def dense_scalar_Y(sites, kernel):
    """Real antisymmetric Y with C_st = (I - iY)/2 from the unstructured G_i tables: Y(x,z) = sum_i eta_i(x) G_i(x-z)."""
    sites = np.asarray(sites, dtype=np.int64)
    n = len(sites)
    r = sites[:, None, :] - sites[None, :, :]
    eta = stag_sign_table()
    cls = parity_class(sites)
    Y = np.zeros((n, n))
    for i in range(3):
        Y += eta[cls, i][:, None] * kernel.g_component(i, r)
    return Y


def dense_scalar_entropy(sites, kernel):
    """Dense n x n eigen-entropy of C_st and the sublattice formula without symmetry; returns a dict."""
    Y = dense_scalar_Y(sites, kernel)
    n = len(Y)
    par = np.asarray(sites).sum(axis=1) % 2
    e_idx, o_idx = np.where(par == 0)[0], np.where(par == 1)[0]
    C = 0.5 * (np.eye(n) - 1j * Y)
    ev = np.linalg.eigvalsh(C)
    S_eig = float(np.sum(binary_entropy(ev)))
    K = Y[np.ix_(e_idx, o_idx)]
    s = linalg.svdvals(K)
    S_svd = sublattice_entropy(s, len(e_idx), len(o_idx))
    same_parity = max(float(np.abs(Y[np.ix_(e_idx, e_idx)]).max()) if len(e_idx) else 0.0,
                      float(np.abs(Y[np.ix_(o_idx, o_idx)]).max()) if len(o_idx) else 0.0)
    return dict(S_scalar_eigen=S_eig, S_scalar_sublattice=S_svd, antisym=float(np.abs(Y + Y.T).max()),
                same_parity_block_max=same_parity, ev_min=float(ev.min()), ev_max=float(ev.max()),
                Y=Y, e_idx=e_idx, o_idx=o_idx)


def dense_V(g_index, sym, signed=True):
    """Dense V_g on the ball: V[i, j] = phi_g(x_i) [x_j = g^{-1} x_i]  (signed=False: plain permutation action)."""
    sites, G, phi = sym.sites, sym.G, sym.phi
    lookup = {tuple(x): i for i, x in enumerate(sites.tolist())}
    g = G[g_index]
    V = np.zeros((len(sites), len(sites)))
    cls = parity_class(sites)
    pre = sites @ g                                       # rows: g^T x = g^{-1} x
    for i in range(len(sites)):
        j = lookup[tuple(pre[i].tolist())]
        V[i, j] = phi[g_index, cls[i]] if signed else 1.0
    return V


def orbit_projector_basis(sym, name, parity):
    """Dense orthonormal basis (n_p x m) of the row-0 isotypic subspace of irrep `name` on the sublattice of
    parity `parity`, built orbit by orbit from P = (d/48) sum_g D_00(g) V_g (the P36-style construction)."""
    sites = sym.sites
    par = sites.sum(axis=1) % 2
    sub = np.where(par == parity)[0]
    lookup = {tuple(x): i for i, x in enumerate(sites[sub].tolist())}
    D = sym.D[name]
    d = D.shape[1]
    cls = parity_class(sites[sub])
    canon = np.sort(np.abs(sites[sub]), axis=1)[:, ::-1]
    cols = []
    for rep in np.unique(canon, axis=0):
        orb = np.where(np.all(canon == rep, axis=1))[0]
        olook = {int(o): t for t, o in enumerate(orb)}
        P = np.zeros((len(orb), len(orb)))
        for gi, g in enumerate(sym.G):
            coef = d / len(sym.G) * D[gi, 0, 0]
            for t, o in enumerate(orb):
                x = sites[sub[o]]
                y = tuple((x @ g).tolist())
                P[t, olook[lookup[y]]] += coef * sym.phi[gi, cls[o]]
        w, U = np.linalg.eigh(0.5 * (P + P.T))
        if np.max(np.minimum(np.abs(w), np.abs(w - 1.0))) > 1e-10:
            raise ArithmeticError('orbit projector is not idempotent')
        for j in np.where(w > 0.5)[0]:
            v = np.zeros(len(sub))
            v[orb] = U[:, j]
            cols.append(v)
    return np.array(cols).T if cols else np.zeros((len(sub), 0))
