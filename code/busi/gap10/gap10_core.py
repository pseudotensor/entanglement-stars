"""gap10_core.py -- the retarded cross-surface response on the 3-dimensional sine/current lattice (GAP10, repaired).

The exact one-particle anticommutator {c_y(t), c_x^dagger(0)} = (exp(-i h t))_{yx} with
    h = (1/2){N, h_0} + (1/2) sum_i {v^i, Q_i},
    h_0 = m beta + sum_i alpha_i (T_i - T_i^dagger)/(2i),   Q_i = (T_i^2 - T_i^dagger^2)/(4i),
    (T_i u)_x = u_{x+e_i},   a = 1, hbar = c = 1,
on an open Cartesian parent (every link crossing the boundary is omitted). Real site fields give endpoint means:
    (h u)_x = m N_x beta u_x + sum_i alpha_i [ n^+_{i,x} u_{x+e_i} - n^-_{i,x} u_{x-e_i} ]/(2i)
              + sum_i [ w^+_{i,x} u_{x+2e_i} - w^-_{i,x} u_{x-2e_i} ]/(4i),
    n^+_{i,x} = (N_x + N_{x+e_i})/2,  n^-_{i,x} = (N_x + N_{x-e_i})/2,
    w^+_{i,x} = (v^i_x + v^i_{x+2e_i})/2,  w^-_{i,x} = (v^i_x + v^i_{x-2e_i})/2.
N = 1 everywhere in the analytic control (the clock drops out); a loaded record may carry N < 1 inside its matter.
State layout: complex128 (B, 4, L, L, L) = (batch, spinor, x, y, z); the site (x, y, z) sits at index (x + c, y + c, z + c).

Repairs against csr_core.py of GAP10 (review R2 to R5): rigorous Chebyshev certificate integrals (trapezoid plus a
second-derivative remainder with Bessel caps), uniform truncation tails for Chebyshev and Taylor, the tight Taylor
boundary integral, allocated (cell-averaged) axis fields for every root, slope, crossing and ray, cell-only sampling
as the ell -> 0 end, the zero-mode split in band_weights, N weights and ghost weights, region Gram matrices for the
transmissivity, the normal-form delay, Lowdin orthonormalization of receiver sets.
"""
import math
import numpy as np
import scipy.integrate as spint
import scipy.interpolate as spi
import scipy.optimize as spo
import scipy.special as sps
import torch

CD = torch.complex128
FD = torch.float64
_trapz = getattr(np, "trapezoid", None) or np.trapz
J1_MAX = 0.5819          # sup_x |J_1(x)| = 0.58187; a valid cap for every |J_m|, m >= 1 (|J_0| <= 1)
SQRT3 = math.sqrt(3.0)

# ------------------------------------------------------------------ Dirac matrices
SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)
SIGMA = (SX, SY, SZ)
_Z2 = np.zeros((2, 2), dtype=np.complex128)
ALPHA = [np.block([[_Z2, S], [S, _Z2]]) for S in SIGMA]
SIGMA4 = [np.block([[S, _Z2], [_Z2, S]]) for S in SIGMA]
BETA = np.diag([1.0, 1.0, -1.0, -1.0]).astype(np.complex128)
BETA_DIAG = np.array([1.0, 1.0, -1.0, -1.0])


def _nz_table(A):
    tab = []
    for row in range(4):
        cols = np.flatnonzero(np.abs(A[row]) > 0)
        assert cols.size == 1
        tab.append((row, int(cols[0]), complex(A[row, cols[0]])))
    return tab


ALPHA_NZ = [_nz_table(A) for A in ALPHA]  # alpha_i[row, col] = val, exactly 1 nonzero per row


# ------------------------------------------------------------------ directions, sites, symmetry
def direction_vector(d):
    return np.array([1.0, 0.0, 0.0]) if d == '100' else np.ones(3) / SQRT3


def site_on_direction(r, d):
    """The lattice site nearest to radius r on the direction ('100' or '111'); returns (pos, actual radius)."""
    if d == '100':
        pos = np.array([int(round(r)), 0, 0], dtype=int)
    else:
        n = int(round(r / SQRT3))
        pos = np.array([n, n, n], dtype=int)
    return pos, float(np.linalg.norm(pos))


def reflection_spinor(d):
    """Spinor matrix of the lattice reflection fixing the axis and flipping helicity: [100]: z -> -z, W = beta Sigma_z;
    [111]: x <-> y, W = beta (Sigma_x - Sigma_y)/sqrt2. W alpha_i W^-1 = alpha_i - 2 n_i (n.alpha) for the plane normal n."""
    if d == '100':
        return BETA @ SIGMA4[2]
    return BETA @ ((SIGMA4[0] - SIGMA4[1]) / math.sqrt(2.0))


# ------------------------------------------------------------------ kernel and radial profiles
def kernel_K(s, ell):
    """K_ell(s) = 9009/(1024 pi ell^3) (1 - s^2/ell^2)_+^5, unit integral in 3 dimensions, sigma_K^2 = ell^2/15 per axis."""
    Cn = 9009.0 / (1024.0 * math.pi * ell ** 3)
    x = np.clip(1.0 - (np.asarray(s, dtype=np.float64) / ell) ** 2, 0.0, None)
    return Cn * x ** 5


def radial_projection(fn, ell, r_max, l, dr=0.125, drp=1.0 / 64.0):
    """l = 0: (K_ell * f)(r) for a scalar radial f; l = 1: the radial magnitude of K_ell * (f(r) rhat).
    l = 0: (2 pi / r) int dr' r' f(r') I_0,  I_0 = int_{|r-r'|}^{min(r+r', ell)} s K_ell(s) ds             (degree 11 in s);
    l = 1: (pi / r^2) int dr' f(r') I_1,     I_1 = int_{|r-r'|}^{min(r+r', ell)} s (r^2 + r'^2 - s^2) K_ell(s) ds (degree 13).
    8-point Gauss-Legendre is exact for both inner integrals; the outer integral is the trapezoid at step drp."""
    rg = np.arange(0.0, r_max + dr, dr)
    out = np.zeros_like(rg)
    xg, wg = np.polynomial.legendre.leggauss(8)
    for j, r in enumerate(rg):
        if r <= 0.0:
            if l == 0:
                s = np.linspace(0.0, ell, 4001)
                out[j] = 4.0 * math.pi * _trapz(s * s * kernel_K(s, ell) * fn(s), s)
            continue
        lo, hi = max(0.0, r - ell), r + ell
        rp = np.linspace(lo, hi, int(round((hi - lo) / drp)) + 1)
        s_lo = np.abs(r - rp)
        s_hi = np.minimum(r + rp, ell)
        half = 0.5 * np.clip(s_hi - s_lo, 0.0, None)
        mid = 0.5 * (s_hi + s_lo)
        s = mid[:, None] + half[:, None] * xg[None, :]
        if l == 0:
            I = half * ((s * kernel_K(s, ell)) @ wg)
            out[j] = (2.0 * math.pi / r) * _trapz(rp * fn(rp) * I, rp)
        else:
            I = half * ((s * (r * r + rp[:, None] ** 2 - s * s) * kernel_K(s, ell)) @ wg)
            out[j] = (math.pi / (r * r)) * _trapz(fn(rp) * I, rp)
    return rg, out


class RadialProfile:
    """|v|(r) = u(r) >= 0 and N(r) before averaging. 'ball': the analytic control (vacuum exterior sqrt(rs/r), N = 1;
    homogeneous marginally bound interior sqrt(rs/Rb) r/Rb for r < Rb). 'file': a radial table (r, N, v or u) of a solved
    record, interpolated linearly, continued regularly below its first radius and by the vacuum exterior sqrt(rs_file/r)
    beyond its last one, rs_file = r v^2 at the last point."""

    def __init__(self, rs, Rb, kind='ball', label='', N_is_one=True, table=None):
        self.rs, self.Rb, self.kind, self.label, self.N_is_one, self.table = float(rs), float(Rb), kind, label, N_is_one, table
        self.rs_file = float(rs) if table is None else float(table[0][-1] * table[2][-1] ** 2)

    @staticmethod
    def ball(rs, Rb):
        return RadialProfile(rs, Rb, 'ball', f'ball rs={rs:g} Rb={Rb:g}', True, None)

    @staticmethod
    def from_file(path, rs, Rb):
        if path.endswith('.npz'):
            z = np.load(path)
            r = np.asarray(z['r'], dtype=float)
            N = np.asarray(z['N'], dtype=float) if 'N' in z.files else np.ones_like(r)
            u = -np.asarray(z['v'], dtype=float) if 'v' in z.files else np.asarray(z['u'], dtype=float)
        else:
            tab = np.genfromtxt(path, names=True)
            names = tab.dtype.names
            r = np.asarray(tab['r'], dtype=float)
            N = np.asarray(tab['N'], dtype=float) if 'N' in names else np.ones_like(r)
            u = -np.asarray(tab['v'], dtype=float) if 'v' in names else np.asarray(tab['u'], dtype=float)
        o = np.argsort(r)
        r, N, u = r[o], N[o], np.abs(u[o])
        return RadialProfile(rs, Rb, 'file', path, bool(np.all(np.abs(N - 1.0) < 1e-12)), (r, N, u))

    def u(self, r):
        r = np.asarray(r, dtype=np.float64)
        if self.kind == 'ball':
            return np.where(r < self.Rb, math.sqrt(self.rs / self.Rb) * r / self.Rb, np.sqrt(self.rs / np.maximum(r, 1e-300)))
        rt, Nt, ut = self.table
        inner = ut[0] * r / max(rt[0], 1e-300)
        outer = np.sqrt(self.rs_file / np.maximum(r, 1e-300))
        return np.where(r < rt[0], inner, np.where(r > rt[-1], outer, np.interp(r, rt, ut)))

    def N(self, r):
        r = np.asarray(r, dtype=np.float64)
        if self.kind == 'ball':
            return np.ones_like(r)
        rt, Nt, ut = self.table
        return np.where(r < rt[0], Nt[0], np.where(r > rt[-1], Nt[-1], np.interp(r, rt, Nt)))


def projected_profiles(prof, ell, sampling, r_max, dr=0.125, drp=1.0 / 64.0):
    """The radial profiles entering the allocation: for 'averaged' the K_ell projections (l = 1 of u, l = 0 of N);
    for 'cell' and 'point' the continuum profiles themselves. Returns (uf, Nf or None, comparator dict)."""
    comp = dict(sampling=sampling, ell=float(ell))
    if sampling == 'averaged':
        rg, V = radial_projection(prof.u, ell, r_max, 1, dr, drp)
        uf = lambda r: np.interp(np.asarray(r, dtype=float), rg, V)
        comp.update(rg=rg, V=V)
        if prof.N_is_one:
            Nf = None
        else:
            rgN, Nl = radial_projection(prof.N, ell, r_max, 0, dr, drp)
            Nf = lambda r: np.interp(np.asarray(r, dtype=float), rgN, Nl)
            comp.update(Nl=Nl)
    else:
        uf = prof.u
        Nf = None if prof.N_is_one else prof.N
    return uf, Nf, comp


def _allocate_at(points, uf, Nf, sampling, ngauss=3):
    """v (n, 3) and N (n,) at the continuum points (n, 3). 'point': the value at the point; 'cell' or 'averaged': the cell
    average over point + [-1/2, 1/2]^3 (ngauss-point Gauss per axis) of the radial profiles uf (vector magnitude,
    inward: v = -uf(r) rhat) and Nf (scalar; None = 1)."""
    P = np.asarray(points, dtype=np.float64)
    v = np.zeros_like(P)
    Nout = None if Nf is None else np.zeros(P.shape[0])
    if sampling == 'point':
        nodes = [((0.0, 0.0, 0.0), 1.0)]
    else:
        xg, wg = np.polynomial.legendre.leggauss(ngauss)
        xg, wg = 0.5 * xg, 0.5 * wg
        nodes = [((xg[a], xg[b], xg[c]), wg[a] * wg[b] * wg[c]) for a in range(ngauss) for b in range(ngauss) for c in range(ngauss)]
    for (dx, dy, dz), w in nodes:
        Q = P + np.array([dx, dy, dz])
        r = np.sqrt((Q * Q).sum(1))
        rr = np.where(r > 0.0, r, 1.0)
        f = -np.where(r > 0.0, uf(rr) / rr, 0.0)
        v += w * f[:, None] * Q
        if Nout is not None:
            Nout += w * Nf(r)
    return v, Nout


def build_site_fields(L, c, ell, prof, sampling, mode='radial', uniform_v=(0.0, 0.0, 0.0), ngauss=3, chunk=2000000):
    """Site flow v^i_x (3 arrays (L, L, L), negative = inward) and N_x ((L, L, L) or None for N = 1) on the grid of L^3 sites
    centred at index c, plus the comparator dict of the radial profiles used. mode 'flat' and 'uniform' are controls."""
    if mode == 'flat':
        return [np.zeros((L, L, L)) for _ in range(3)], None, dict(sampling='flat')
    if mode == 'uniform':
        return [np.full((L, L, L), float(uniform_v[i])) for i in range(3)], None, dict(sampling='uniform')
    r_max = SQRT3 * (c + 1.0) + ell + 2.0
    uf, Nf, comp = projected_profiles(prof, ell, sampling, r_max)
    idx = (np.arange(L) - c).astype(np.float64)
    X, Y, Z = np.meshgrid(idx, idx, idx, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    del X, Y, Z
    v = np.zeros((pts.shape[0], 3))
    Narr = None if Nf is None else np.zeros(pts.shape[0])
    for s in range(0, pts.shape[0], chunk):
        vv, nn = _allocate_at(pts[s:s + chunk], uf, Nf, sampling, ngauss)
        v[s:s + chunk] = vv
        if Narr is not None:
            Narr[s:s + chunk] = nn
    vlist = [np.ascontiguousarray(v[:, i].reshape(L, L, L)) for i in range(3)]
    N3 = None if Narr is None else np.ascontiguousarray(Narr.reshape(L, L, L))
    return vlist, N3, comp


def axis_allocated(prof, ell, sampling, direction, nmax, ngauss=3):
    """The allocated field at the axis sites n e (e = (1,0,0) or (1,1,1)), n = 0 .. nmax, by the same quadrature as
    build_site_fields: returns (r_nodes, u_nodes = -v.ehat, N_nodes or None, comparator dict)."""
    e = np.array([1.0, 0.0, 0.0]) if direction == '100' else np.ones(3)
    pts = np.arange(nmax + 1)[:, None] * e[None, :]
    r_nodes = np.linalg.norm(pts, axis=1)
    r_max = float(r_nodes[-1]) + ell + 3.0
    uf, Nf, comp = projected_profiles(prof, ell, sampling, r_max)
    v, N = _allocate_at(pts, uf, Nf, sampling, ngauss)
    ehat = e / np.linalg.norm(e)
    return r_nodes, -(v @ ehat), N, comp


class AxisField:
    """The allocated field along a lattice axis: nodes r_n with u_n = |v| and N_n, natural cubic-spline interpolants.
    c(r) = N(r) - u(r) is the low-cone trapping function whose zeros are the crossings."""

    def __init__(self, r_nodes, u_nodes, N_nodes=None, label=''):
        self.r = np.asarray(r_nodes, dtype=float)
        self.u_nodes = np.asarray(u_nodes, dtype=float)
        self.N_nodes = None if N_nodes is None else np.asarray(N_nodes, dtype=float)
        self._u = spi.CubicSpline(self.r, self.u_nodes, bc_type='natural')
        self._N = None if self.N_nodes is None else spi.CubicSpline(self.r, self.N_nodes, bc_type='natural')
        self.label = label

    def u(self, r):
        r = np.clip(np.asarray(r, dtype=float), self.r[0], self.r[-1])
        return self._u(r)

    def N(self, r):
        r = np.asarray(r, dtype=float)
        if self._N is None:
            return np.ones_like(r)
        return self._N(np.clip(r, self.r[0], self.r[-1]))

    def c(self, r):
        return self.N(r) - self.u(r)


# ------------------------------------------------------------------ weights
def bond_weights_np(v):
    """w^+_i[x] = (v^i_x + v^i_{x+2e_i})/2 (0 on the last 2 planes); w^-_i[x] = (v^i_x + v^i_{x-2e_i})/2 (0 on the first 2)."""
    L = v[0].shape[0]
    wp, wm = [], []
    for i in range(3):
        a = v[i]
        p = np.zeros_like(a)
        q = np.zeros_like(a)
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[i] = slice(0, L - 2)
        hi[i] = slice(2, L)
        lo, hi = tuple(lo), tuple(hi)
        p[lo] = 0.5 * (a[lo] + a[hi])
        q[hi] = 0.5 * (a[hi] + a[lo])
        wp.append(p)
        wm.append(q)
    return wp, wm


def link_weights_np(N):
    """n^+_i[x] = (N_x + N_{x+e_i})/2 (0 on the last plane); n^-_i[x] = (N_x + N_{x-e_i})/2 (0 on the first plane)."""
    L = N.shape[0]
    lp, lm = [], []
    for i in range(3):
        p = np.zeros_like(N)
        q = np.zeros_like(N)
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[i] = slice(0, L - 1)
        hi[i] = slice(1, L)
        lo, hi = tuple(lo), tuple(hi)
        p[lo] = 0.5 * (N[lo] + N[hi])
        q[hi] = 0.5 * (N[hi] + N[lo])
        lp.append(p)
        lm.append(q)
    return lp, lm


def ghost_weights_np(v_full, L, N_full=None):
    """From the fields on the (L+4)^3 halo grid (interior = halo index 2 .. L+1), the weights of the omitted links as (L, L)
    planes per axis, in the axis coordinate x = halo index - 2: near_lo = (v_{-1} + v_{+1})/2, far_lo = (v_{-2} + v_0)/2,
    near_hi = (v_L + v_{L-2})/2, far_hi = (v_{L+1} + v_{L-1})/2; nN_lo = (N_{-1} + N_0)/2, nN_hi = (N_L + N_{L-1})/2."""
    out = []
    for i in range(3):
        a = v_full[i]

        def plane(arr, k, i=i):
            sl = [slice(2, L + 2)] * 3
            sl[i] = k
            return arr[tuple(sl)]

        g = dict(near_lo=0.5 * (plane(a, 1) + plane(a, 3)), far_lo=0.5 * (plane(a, 0) + plane(a, 2)),
                 near_hi=0.5 * (plane(a, L + 2) + plane(a, L)), far_hi=0.5 * (plane(a, L + 3) + plane(a, L + 1)))
        if N_full is None:
            g['nN_lo'] = np.ones((L, L))
            g['nN_hi'] = np.ones((L, L))
        else:
            g['nN_lo'] = 0.5 * (plane(N_full, 1) + plane(N_full, 2))
            g['nN_hi'] = 0.5 * (plane(N_full, L + 2) + plane(N_full, L + 1))
        out.append(g)
    return out


def spectral_bound(m, wp, wm, N=None):
    """||h|| <= N_max sqrt(m^2 + 3) + max_x sum_i (|w^+_{i,x}| + |w^-_{i,x}|)/4  (compression + Schur test)."""
    rowsum = 0.25 * sum(np.abs(wp[i]) + np.abs(wm[i]) for i in range(3))
    Nmax = 1.0 if N is None else float(np.max(N))
    return Nmax * math.sqrt(m * m + 3.0) + float(rowsum.max())


# ------------------------------------------------------------------ the operator on the device
def shift(psi, dim, s):
    """out[..., x, ...] = psi[..., x + s, ...] along tensor dim `dim`; zero where x + s lies outside."""
    out = torch.roll(psi, shifts=-s, dims=dim)
    L = psi.shape[dim]
    if s > 0:
        out.narrow(dim, L - s, s).zero_()
    else:
        out.narrow(dim, 0, -s).zero_()
    return out


def _axpy_supported():
    try:
        x = torch.zeros(2, dtype=CD)
        x.add_(torch.ones(2, dtype=CD), alpha=0.5 + 0.25j)
        return bool(torch.allclose(x, torch.full((2,), 0.5 + 0.25j, dtype=CD)))
    except Exception:
        return False


AXPY_OK = _axpy_supported()


def axpy_(acc, x, cst):
    """acc += cst * x for a complex scalar cst."""
    if AXPY_OK:
        acc.add_(x, alpha=complex(cst))
    else:
        acc.add_(x * complex(cst))


def vec_norm(psi):
    """||psi_b|| per batch element."""
    return torch.linalg.vector_norm(psi.reshape(psi.shape[0], -1), dim=1)


class LatticeH:
    """h on the open parent. apply(psi) returns h psi for psi of shape (B, 4, L, L, L);
    boundary_residual(psi) returns ||(1 - Pi) h Pi psi|| per batch element (the omitted-link amplitude)."""

    def __init__(self, m, wp_np, wm_np, device, ghost_np=None, N_np=None, lp_np=None, lm_np=None):
        self.m = float(m)
        self.device = device
        self.L = int(wp_np[0].shape[0])
        self.mbeta = torch.tensor(self.m * BETA_DIAG, dtype=CD, device=device).view(1, 4, 1, 1, 1)
        self.wp = [torch.tensor(w * (-0.25j), dtype=CD, device=device).view(1, 1, *w.shape) for w in wp_np]
        self.wm = [torch.tensor(w * (-0.25j), dtype=CD, device=device).view(1, 1, *w.shape) for w in wm_np]
        self.N = None if N_np is None else torch.tensor(N_np, dtype=FD, device=device).view(1, 1, *N_np.shape)
        self.lp = None if lp_np is None else [torch.tensor(w, dtype=FD, device=device).view(1, 1, *w.shape) for w in lp_np]
        self.lm = None if lm_np is None else [torch.tensor(w, dtype=FD, device=device).view(1, 1, *w.shape) for w in lm_np]
        self.ghost = None
        if ghost_np is not None:
            self.ghost = [{k: torch.tensor(g[k], dtype=FD, device=device) for k in g} for g in ghost_np]
        self.napply = 0

    def apply(self, psi):
        out = psi * self.mbeta
        if self.N is not None:
            out.mul_(self.N)
        for i in range(3):
            dim = i + 2
            D = shift(psi, dim, 1)
            if self.lp is not None:
                D.mul_(self.lp[i])
            T = shift(psi, dim, -1)
            if self.lm is not None:
                T.mul_(self.lm[i])
            D.sub_(T)
            del T
            for (row, col, val) in ALPHA_NZ[i]:
                axpy_(out[:, row], D[:, col], val * (-0.5j))
            del D
            S = shift(psi, dim, 2)
            S.mul_(self.wp[i])
            T = shift(psi, dim, -2)
            T.mul_(self.wm[i])
            S.sub_(T)
            del T
            out.add_(S)
            del S
        self.napply += 1
        return out

    @staticmethod
    def _alpha_face(F, i):
        rows = [None] * 4
        for (row, col, val) in ALPHA_NZ[i]:
            rows[row] = F[:, col] * val
        return torch.stack(rows, 1)

    def boundary_residual(self, psi):
        B = psi.shape[0]
        total = torch.zeros(B, dtype=FD, device=psi.device)
        if self.ghost is None:
            return total
        for i in range(3):
            dim = i + 2
            L = psi.shape[dim]
            g = self.ghost[i]
            f0 = psi.narrow(dim, 0, 1).squeeze(dim)
            f1 = psi.narrow(dim, 1, 1).squeeze(dim)
            fL1 = psi.narrow(dim, L - 1, 1).squeeze(dim)
            fL2 = psi.narrow(dim, L - 2, 1).squeeze(dim)
            vec = self._alpha_face(f0, i) * ((-0.5j) * g['nN_lo']) + f1 * ((-0.25j) * g['near_lo'])
            total += (vec.real ** 2 + vec.imag ** 2).sum((1, 2, 3))
            vec = f0 * ((-0.25j) * g['far_lo'])
            total += (vec.real ** 2 + vec.imag ** 2).sum((1, 2, 3))
            vec = self._alpha_face(fL1, i) * ((0.5j) * g['nN_hi']) + fL2 * ((0.25j) * g['near_hi'])
            total += (vec.real ** 2 + vec.imag ** 2).sum((1, 2, 3))
            vec = fL1 * ((0.25j) * g['far_hi'])
            total += (vec.real ** 2 + vec.imag ** 2).sum((1, 2, 3))
        return torch.sqrt(total)

    def energy(self, psi):
        """<psi|h|psi> and ||h psi||^2 per batch element (1 application)."""
        hp = self.apply(psi)
        e = (psi.conj() * hp).sum((1, 2, 3, 4)).real
        e2 = (hp.real ** 2 + hp.imag ** 2).sum((1, 2, 3, 4))
        return e, e2


# ------------------------------------------------------------------ propagators
def cheb_tail_bound(n, z):
    """2 sum_{k>n} |J_k(z)| <= 2 (z/2)^{n+1}/((n+1)! (1 - z/(2(n+2)))) for z < 2(n+2) (|J_k(z)| <= (z/2)^k/k!, AS 9.1.62):
    a uniform bound on the Chebyshev truncation error sup_x |exp(-i z x) - P_n(x)| over [-1, 1], increasing in z."""
    if z <= 0.0:
        return 0.0
    if z >= 2.0 * (n + 2):
        return float('inf')
    lg = (n + 1) * math.log(z / 2.0) - math.lgamma(n + 2.0)
    return 2.0 * math.exp(lg) / (1.0 - z / (2.0 * (n + 2)))


def cheb_order(dE, tau, tol=1e-17):
    """The least order n with the rigorous tail bound below tol at z = dE tau (plus 1)."""
    z = dE * tau
    n = max(2, int(z))
    while cheb_tail_bound(n, z) >= tol:
        n += 1
    return n + 1


def cheb_readout(dE, n, times):
    """C[j, k] = (2 - delta_k0)(-i)^k J_k(dE t_j): the readout matrix at the given times within a sweep."""
    kk = np.arange(n + 1)
    ph = np.array([1.0, -1.0j, -1.0, 1.0j])[kk % 4]
    fac = np.where(kk == 0, 1.0, 2.0)
    Jm = sps.jv(kk[None, :], dE * np.asarray(times, dtype=float)[:, None])
    return (fac * ph)[None, :] * Jm


def cheb_coefficients(dE, tau, dt_read, tol=1e-17, nsub=512):
    """Chebyshev order n, the readout times tr within the sweep, the readout matrix Cm[j, k], the rigorous upper bounds
    Ik[j, k] >= int_0^{t_j} |c_k(s)| ds (composite trapezoid at step dt_read/nsub plus the remainder
    sum_intervals ds^3 sup|g''|/8 with g = J_k(dE s), sup|g''| <= (dE^2/4)(M_|k-2| + 2 M_k + M_{k+2}),
    M_m(s) = min(cap_m, (dE s/2)^m/m!), cap_0 = 1, cap_m = J1_MAX), and the uniform truncation tail over the sweep."""
    n = cheb_order(dE, tau, tol)
    nread = int(round(tau / dt_read))
    assert abs(nread * dt_read - tau) < 1e-9, "tau must be a multiple of dt_read"
    tr = dt_read * np.arange(1, nread + 1)
    Cm = cheb_readout(dE, n, tr)
    kk = np.arange(n + 1)
    fac = np.where(kk == 0, 1.0, 2.0)
    nfine = nread * nsub
    ds = tau / nfine
    tf = ds * np.arange(nfine + 1)
    Jf = np.abs(sps.jv(kk[None, :], dE * tf[:, None]))
    trap = np.zeros_like(Jf)
    trap[1:] = np.cumsum(0.5 * (Jf[1:] + Jf[:-1]) * ds, axis=0)
    lsb = np.log(0.5 * dE * tf[1:])

    def cap(mm):
        mm = np.asarray(mm, dtype=float)
        cp = np.where(mm == 0, 1.0, J1_MAX)
        lg = mm[None, :] * lsb[:, None] - sps.gammaln(mm[None, :] + 1.0)
        return np.minimum(cp[None, :], np.exp(np.minimum(lg, 0.0)))

    Msum = cap(np.abs(kk - 2)) + 2.0 * cap(kk) + cap(kk + 2)
    rem = np.zeros_like(Jf)
    rem[1:] = np.cumsum((ds ** 3 / 8.0) * (dE ** 2 / 4.0) * Msum, axis=0)
    Ik = (fac[None, :] * (trap + rem))[nsub::nsub]
    assert Ik.shape[0] == nread
    return n, tr, Cm.astype(np.complex128), Ik, cheb_tail_bound(n, dE * tau)


def cheb_sweep(H, psi, dE, c_end, n, project, residual=True):
    """psi(t0 + tau) = sum_k c_end[k] T_k(h/dE) psi(t0); returns the new state, P[k] = project(T_k psi),
    R[k] = ||B_Lambda T_k psi|| per batch element (None if residual is False) and Nr[k] = ||T_k psi|| (stability monitor)."""
    P = [project(psi)]
    R = [H.boundary_residual(psi)] if residual else None
    Nr = [vec_norm(psi)]
    phi_prev = psi
    phi = H.apply(psi)
    phi.mul_(1.0 / dE)
    acc = psi * complex(c_end[0])
    axpy_(acc, phi, c_end[1])
    P.append(project(phi))
    if residual:
        R.append(H.boundary_residual(phi))
    Nr.append(vec_norm(phi))
    for k in range(2, n + 1):
        new = H.apply(phi)
        new.mul_(2.0 / dE)
        new.sub_(phi_prev)
        axpy_(acc, new, c_end[k])
        P.append(project(new))
        if residual:
            R.append(H.boundary_residual(new))
        Nr.append(vec_norm(new))
        phi_prev, phi = phi, new
        del new
    del phi_prev, phi
    return acc, torch.stack(P, 0), (torch.stack(R, 0) if residual else None), torch.stack(Nr, 0)


def taylor_tail_bound(J, x):
    """sum_{j>J} x^j/j! <= x^{J+1}/((J+1)! (1 - x/(J+2))) for x < J + 2."""
    if x <= 0.0:
        return 0.0
    if x >= J + 2:
        return float('inf')
    return math.exp((J + 1) * math.log(x) - math.lgamma(J + 2.0)) / (1.0 - x / (J + 2.0))


def taylor_order(dE, dt, tol=1e-17):
    x = dE * dt
    J = max(2, int(x))
    while taylor_tail_bound(J, x) >= tol:
        J += 1
    return J


def taylor_step(H, psi, dt, J, project=None, residual=True):
    """psi(t + dt) by the Taylor polynomial sum_{j<=J} (-i dt h)^j psi/j!. Returns the new state, the boundary integral
    int_0^dt ||B_Lambda psi~(t + s)|| ds <= sum_j dt/(j+1) ||B_Lambda term_j|| of the in-step polynomial trajectory, and the
    projections of the terms (J+1, B, nrec) so that A(t + s) = sum_j (s/dt)^j P_j (None if project is None)."""
    acc = psi.clone()
    term = psi
    rsum = (H.boundary_residual(psi) * dt) if residual else None
    P = [project(psi)] if project is not None else None
    for j in range(1, J + 1):
        new = H.apply(term)
        new.mul_(-1j * dt / j)
        acc.add_(new)
        term = new
        if residual:
            rsum = rsum + H.boundary_residual(term) * (dt / (j + 1))
        if P is not None:
            P.append(project(term))
    return acc, rsum, (torch.stack(P, 0) if P is not None else None)


def free_propagate_fft(psi0, t, m, vconst):
    """Exact exp(-i h t) psi0 on the PERIODIC box for N = 1 and constant v (control only)."""
    L = psi0.shape[2]
    dev = psi0.device
    p = 2.0 * math.pi * torch.fft.fftfreq(L, d=1.0, dtype=FD, device=dev)
    Px, Py, Pz = torch.meshgrid(p, p, p, indexing='ij')
    s = [torch.sin(Px), torch.sin(Py), torch.sin(Pz)]
    eps = torch.sqrt(m * m + s[0] ** 2 + s[1] ** 2 + s[2] ** 2)
    flow = 0.5 * (vconst[0] * torch.sin(2 * Px) + vconst[1] * torch.sin(2 * Py) + vconst[2] * torch.sin(2 * Pz))
    phase = torch.exp(-1j * t * flow)
    cos_t = torch.cos(eps * t)
    pos = eps > 0
    sinc = torch.where(pos, torch.sin(eps * t) / torch.where(pos, eps, torch.ones_like(eps)), torch.full_like(eps, float(t)))
    ph = torch.fft.fftn(psi0, dim=(2, 3, 4))
    out = ph * cos_t[None]
    for i in range(3):
        for (row, col, val) in ALPHA_NZ[i]:
            out[:, row] += (-1j * val) * (sinc * s[i])[None] * ph[:, col]
    for row in range(4):
        out[:, row] += (-1j * m * BETA_DIAG[row]) * sinc[None] * ph[:, row]
    out *= phase[None, None]
    return torch.fft.ifftn(out, dim=(2, 3, 4))


# ------------------------------------------------------------------ receivers and diagnostics
class Receivers:
    """Site receivers (4 components each) and packet groups (packets sharing 1 site index set), with the overlap matrix
    S_ab = <g_a, g_b> of every group for the Lowdin orthonormalization of the receiver set."""

    def __init__(self, L, c, device):
        self.L, self.c, self.device = L, c, device
        self.site_pos, self.site_meta, self.groups = [], [], []

    def add_site(self, pos, role, r):
        self.site_pos.append(np.asarray(pos, dtype=int))
        self.site_meta.append(dict(kind='site', role=role, r=float(r), pos=[int(p) for p in pos]))

    def add_packet_group(self, idx, values_list, metas):
        G = np.stack([np.conj(v.reshape(-1)) for v in values_list], 1)
        S = G.T @ np.conj(G)
        self.groups.append(dict(idx=torch.tensor(np.asarray(idx, dtype=np.int64), device=self.device),
                                G=torch.tensor(G, dtype=CD, device=self.device), S=S, metas=list(metas)))

    def finalize(self):
        L, c = self.L, self.c
        flat = [int((p[0] + c) * L * L + (p[1] + c) * L + (p[2] + c)) for p in self.site_pos]
        self.site_flat = torch.tensor(flat, dtype=torch.int64, device=self.device)
        self.meta = []
        for sm in self.site_meta:
            for comp in range(4):
                d = dict(sm)
                d['comp'] = comp
                d['label'] = f"site|{sm['role']}|r={sm['r']:.3f}|c={comp}"
                d['group'] = -1
                self.meta.append(d)
        self.group_cols = []
        for gi, g in enumerate(self.groups):
            c0 = len(self.meta)
            for mm in g['metas']:
                mm['group'] = gi
            self.meta.extend(g['metas'])
            self.group_cols.append((c0, len(self.meta)))
        self.labels = [d['label'] for d in self.meta]
        self.nrec = len(self.labels)

    def project(self, phi):
        B = phi.shape[0]
        flat = phi.reshape(B, 4, -1)
        outs = []
        if len(self.site_pos):
            outs.append(flat[:, :, self.site_flat].permute(0, 2, 1).reshape(B, -1))
        for g in self.groups:
            outs.append(flat.index_select(2, g['idx']).reshape(B, -1) @ g['G'])
        return torch.cat(outs, 1)


class Geometry:
    """Radial shells, escaped-share masks, region masks for the transmissivity Gram matrices, boundary-layer mask, planes."""

    def __init__(self, L, c, rs, r_h, device, region_fracs=(1.5, 2.0)):
        idx = torch.arange(L, dtype=FD, device=device) - c
        X, Y, Z = torch.meshgrid(idx, idx, idx, indexing='ij')
        r = torch.sqrt(X * X + Y * Y + Z * Z)
        self.shell = torch.floor(r + 0.5).to(torch.int64).reshape(-1)
        self.nshell = int(self.shell.max().item()) + 1
        self.share_radii = [('r_h', float(r_h)), ('1.25rs', 1.25 * rs), ('1.5rs', 1.5 * rs), ('2rs', 2.0 * rs), ('3rs', 3.0 * rs)]
        self.masks = [(r > rad) for (_, rad) in self.share_radii]
        self.region_names = ['r_h'] + [f'{f:g}rs' for f in region_fracs]
        self.region_masks = [r > float(r_h)] + [r > f * rs for f in region_fracs]
        bl = torch.zeros((L, L, L), dtype=torch.bool, device=device)
        for dd in range(3):
            bl.narrow(dd, 0, 2).fill_(True)
            bl.narrow(dd, L - 2, 2).fill_(True)
        self.bmask = bl
        self.c = c
        self.ar = torch.arange(L, device=device)
        del X, Y, Z, r

    def measure(self, psi):
        dens = (psi.real ** 2 + psi.imag ** 2).sum(1)
        B = dens.shape[0]
        norms = dens.sum((1, 2, 3))
        flat = dens.reshape(B, -1)
        hist = torch.stack([torch.bincount(self.shell, weights=flat[b], minlength=self.nshell) for b in range(B)])
        shares = torch.stack([dens[:, msk].sum(1) for msk in self.masks], 1)
        bnd = dens[:, self.bmask].sum(1)
        pz = dens[:, :, :, self.c]
        pd = dens[:, :, self.ar, self.ar]
        return (norms.cpu().numpy(), hist.cpu().numpy(), shares.cpu().numpy(), bnd.cpu().numpy(),
                pz.float().cpu().numpy(), pd.float().cpu().numpy())

    def gram(self, psi):
        """G[k, a, b] = <P_D psi_a, P_D psi_b> for every region k: the Gram matrix whose largest eigenvalue over a source
        family is s_max^2(P_D U B_X), the retarded cross-surface transmissivity."""
        B = psi.shape[0]
        out = np.zeros((len(self.region_masks), B, B), dtype=np.complex128)
        for a in range(B):
            for b in range(a, B):
                prod = (psi[a].conj() * psi[b]).sum(0)
                for k, msk in enumerate(self.region_masks):
                    val = complex(prod[msk].sum().item())
                    out[k, a, b] = val
                    out[k, b, a] = np.conj(val)
                del prod
        return out


# ------------------------------------------------------------------ dispersion, spinors, packets
def disp_omega(k, u, n, m, N=1.0):
    """The axial positive branch H_+(k) = N eps(k n) - u sum_i n_i sin(2 k n_i)/2 (flow magnitude u >= 0, inward)."""
    k = np.atleast_1d(np.asarray(k, dtype=float))
    n = np.asarray(n, dtype=float)
    z = k[:, None] * n[None, :]
    eps = np.sqrt(m * m + np.sum(np.sin(z) ** 2, 1))
    return N * eps - u * np.sum(n[None, :] * np.sin(2.0 * z), 1) / 2.0


def disp_vg(k, u, n, m, N=1.0):
    k = np.atleast_1d(np.asarray(k, dtype=float))
    n = np.asarray(n, dtype=float)
    z = k[:, None] * n[None, :]
    eps = np.sqrt(m * m + np.sum(np.sin(z) ** 2, 1))
    return N * np.sum(np.sin(z) * np.cos(z) * n[None, :], 1) / eps - u * np.sum((n * n)[None, :] * np.cos(2.0 * z), 1)


def outgoing_root(omega, u, n, m, N=1.0, check_unique=False):
    """The outgoing positive-branch root k in (0, k_max) of H_+(k; u, N) = omega along n (m < omega), k_max = (pi/2)/max|n_i|.
    For u > N the massless root lies above arccos(N/u)/max|n_i| (a lower endpoint, review R4). With check_unique the
    number of sign changes of H_+ - omega on (0, k_max) is returned beside the root."""
    n = np.asarray(n, dtype=float)
    kmax = (math.pi / 2.0) / float(np.max(np.abs(n)))

    def f(k):
        return float(disp_omega(k, u, n, m, N)[0] - omega)

    f0, f1 = f(1e-9), f(kmax)
    if not (f0 < 0.0 < f1):
        raise ValueError(f"no propagating outgoing root: omega={omega}, u={u:.4f}, N={N:.4f}, m={m}, f(0)={f0:.3e}, f(kmax)={f1:.3e}")
    k = spo.brentq(f, 1e-9, kmax, xtol=1e-14, rtol=1e-14)
    if check_unique:
        kg = np.linspace(1e-9, kmax, 4001)
        g = disp_omega(kg, u, n, m, N) - omega
        return k, int(np.sum(np.sign(g[:-1]) != np.sign(g[1:])))
    return k


def helicity_spinors(k, n, m):
    """Positive-energy eigenvectors of h_0(p = k n) with helicity Sigma.n = +1 and -1 (in that order)."""
    n = np.asarray(n, dtype=float)
    p = k * n
    d = np.sin(p)
    eps = math.sqrt(m * m + float(d @ d))
    sn = sum(n[i] * SIGMA[i] for i in range(3))
    evals, evecs = np.linalg.eigh(sn)
    chis = [evecs[:, 1], evecs[:, 0]]
    sd = sum(d[i] * SIGMA[i] for i in range(3))
    out = []
    for chi in chis:
        top = math.sqrt((eps + m) / (2.0 * eps)) * chi
        bottom = (sd @ chi) / math.sqrt(2.0 * eps * (eps + m))
        uvec = np.concatenate([top, bottom]).astype(np.complex128)
        out.append(uvec / np.linalg.norm(uvec))
    return out


def clock_offsets(r, rs):
    """F, Fu with T = t - F(r) (eq. example-chart) and u = t - Fu(r), Fu = F + r + rs ln(r/rs - 1); r > rs. Fu' = 1/(1 - S_0)."""
    r = np.asarray(r, dtype=np.float64)
    F = 2.0 * np.sqrt(rs * r) + rs * np.log((np.sqrt(r) - math.sqrt(rs)) / (np.sqrt(r) + math.sqrt(rs)))
    return F, F + r + rs * np.log(r / rs - 1.0)


def chart_T(t, r, rs):
    return float(t - clock_offsets(r, rs)[0])


def chart_u(t, r, rs):
    return float(t - clock_offsets(r, rs)[1])


def packet_mode(L, c, pos, nhat, k, spinor, sigma, cut, rs, restriction='raw', common_half=None):
    """Packet f_x = Z exp(-|x - pos|^2/(4 sigma^2)) exp(i k nhat.(x - pos)) chi on the ball |x - pos| <= cut sigma, clipped to
    the box (and to |x| <= common_half if given: common vectors for parent comparisons), restricted to r < rs ('in'),
    r > rs ('out') or not ('raw'), then normalized. Returns flat indices (n,), values (4, n), coords (n, 3), probabilities, meta."""
    pos = np.asarray(pos, dtype=int)
    nhat = np.asarray(nhat, dtype=float)
    R = cut * sigma
    hw = int(math.ceil(R))
    q = np.arange(-hw, hw + 1)
    X, Y, Z = np.meshgrid(q, q, q, indexing='ij')
    delta = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    d2 = (delta ** 2).sum(1)
    ball = d2 <= R * R
    delta, d2 = delta[ball], d2[ball]
    scalar = np.exp(-d2 / (4.0 * sigma * sigma)) * np.exp(1j * (delta @ (k * nhat)))
    full_norm2 = float(np.sum(np.abs(scalar) ** 2))
    coords = delta + pos
    hcl = c if common_half is None else min(c, int(common_half))
    inbox = np.all(np.abs(coords) <= hcl, 1)
    r = np.linalg.norm(coords, axis=1)
    keep = inbox.copy()
    if restriction == 'in':
        keep &= r < rs
    elif restriction == 'out':
        keep &= r > rs
    elif restriction != 'raw':
        raise ValueError(restriction)
    retained = float(np.sum(np.abs(scalar[keep]) ** 2) / full_norm2)
    coords, scalar, r = coords[keep], scalar[keep], r[keep]
    if scalar.size == 0:
        raise ValueError("empty packet support")
    scalar = scalar / math.sqrt(float(np.sum(np.abs(scalar) ** 2)))
    values = np.asarray(spinor, dtype=np.complex128)[:, None] * scalar[None, :]
    prob = np.abs(scalar) ** 2
    flat = (((coords[:, 0] + c) * L + (coords[:, 1] + c)) * L + (coords[:, 2] + c)).astype(np.int64)
    mean = prob @ coords
    meta = dict(retained=retained, clipped=bool(np.any(~inbox)), n_sites=int(scalar.size),
                exterior_probability=float(prob[r > rs].sum()), interior_probability=float(prob[r < rs].sum()),
                mean_r=float(prob @ r), mean_pos=[float(x) for x in mean],
                position_std=[float(x) for x in np.sqrt(prob @ (coords - mean) ** 2)])
    if np.all(r > rs):
        F, Fu = clock_offsets(r, rs)
        mT, mu = float(prob @ F), float(prob @ Fu)
        meta.update(T_offset_mean=-mT, u_offset_mean=-mu,
                    T_rms=float(math.sqrt(max(prob @ (F - mT) ** 2, 0.0))),
                    u_rms=float(math.sqrt(max(prob @ (Fu - mu) ** 2, 0.0))))
    return flat, values, coords, prob, meta


def band_weights(values, coords, pos, m, v_vec, nhat, hw, N=1.0):
    """Fourier transform of the packet on its bounding cube: weights on the upper band of h_0, on positive frequency of the
    local symbol omega_pm = v.q(p) pm N eps(p), on outgoing positive frequency, at folded momentum above pi/4, and the weight
    of the massless Fourier zero mode (eps = 0, no band assignment; review R5), reported separately."""
    n = 2 * hw + 1
    cube = np.zeros((4, n, n, n), dtype=np.complex128)
    rel = (np.asarray(coords, dtype=int) - np.asarray(pos, dtype=int)) + hw
    cube[:, rel[:, 0], rel[:, 1], rel[:, 2]] = values
    phi = np.fft.fftn(cube, axes=(1, 2, 3), norm='ortho')
    p = 2.0 * math.pi * np.fft.fftfreq(n)
    P = list(np.meshgrid(p, p, p, indexing='ij'))
    d = [np.sin(x) for x in P]
    q = [0.5 * np.sin(2.0 * x) for x in P]
    eps = np.sqrt(m * m + d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
    zero = eps == 0
    safe = np.where(zero, 1.0, eps)
    h0phi = m * BETA_DIAG[:, None, None, None] * phi
    for i in range(3):
        for (row, col, val) in ALPHA_NZ[i]:
            h0phi[row] += val * d[i] * phi[col]
    plus = 0.5 * (phi + np.where(zero, 0.0, h0phi / safe))
    minus = phi - plus
    wp = (np.abs(plus) ** 2).sum(0)
    wm = (np.abs(minus) ** 2).sum(0)
    wz = float((np.abs(phi) ** 2).sum(0)[zero].sum())
    wp[zero] = 0.0
    wm[zero] = 0.0
    drag = sum(v_vec[i] * q[i] for i in range(3))
    om_p, om_m = drag + N * eps, drag - N * eps
    vg_p = sum(nhat[i] * (N * d[i] * np.cos(P[i]) / safe + v_vec[i] * np.cos(2 * P[i])) for i in range(3))
    vg_m = sum(nhat[i] * (-N * d[i] * np.cos(P[i]) / safe + v_vec[i] * np.cos(2 * P[i])) for i in range(3))
    folded = np.maximum.reduce([np.minimum(np.abs(x), math.pi - np.abs(x)) for x in P])
    return dict(norm=float(wp.sum() + wm.sum() + wz), upper_band=float(wp.sum()), zero_mode=wz,
                positive_frequency=float(wp[om_p > 0].sum() + wm[om_m > 0].sum()),
                outgoing_positive_frequency=float(wp[(om_p > 0) & (vg_p > 0)].sum() + wm[(om_m > 0) & (vg_m > 0)].sum()),
                outgoing_any=float(wp[vg_p > 0].sum() + wm[vg_m > 0].sum()),
                lattice_scale=float((wp + wm)[folded > math.pi / 4].sum()))


def carrier_shift(k, nhat, du):
    """Review eq. G10.6: omega_run = omega_label + (u_run - u_prep)(x) . q(p_prep) with v = -u nhat: the predicted shift
    of a fixed-preparation carrier is -du sum_i n_i sin(2 k n_i)/2."""
    nhat = np.asarray(nhat, dtype=float)
    return float(-du * np.sum(nhat * np.sin(2.0 * k * nhat)) / 2.0)


# ------------------------------------------------------------------ rays, crossings, the normal form
def ray_time(r0, r1, omega, n, m, u_fn, N_fn=None, crossings=None):
    """Characteristic flight time on the outgoing positive branch between radii r0 and r1 at frequency omega along n, for the
    flow magnitude u_fn(r) and lapse N_fn(r) (1 if None): int dr / c_g(r; omega), adaptive quadrature with the crossings as
    break points (the integrand peaks there at small omega)."""

    def inv_cg(r):
        u = float(np.asarray(u_fn(r), dtype=float).reshape(-1)[0])
        N = 1.0 if N_fn is None else float(np.asarray(N_fn(r), dtype=float).reshape(-1)[0])
        k = outgoing_root(omega, u, n, m, N)
        return 1.0 / float(disp_vg(k, u, n, m, N)[0])

    lo, hi = min(r0, r1), max(r0, r1)
    pts = [x for x in (crossings or []) if lo < x < hi] or None
    val, err = spint.quad(inv_cg, lo, hi, points=pts, limit=500, epsabs=1e-8, epsrel=1e-8)
    return float(val)


def crossings_between(c_fn, r_a, r_b, nscan=20001, h=0.05):
    """All radii in [min, max] with c(r) = N - |v| = 0, each with its inverse |slope| (the log coefficient of that crossing)."""
    lo, hi = min(r_a, r_b), max(r_a, r_b)
    if hi <= lo:
        return []
    rg = np.linspace(lo, hi, nscan)
    g = np.asarray(c_fn(rg), dtype=float).reshape(-1)
    idx = np.flatnonzero(np.sign(g[:-1]) != np.sign(g[1:]))

    def fs(x):
        return float(np.asarray(c_fn(x), dtype=float).reshape(-1)[0])

    out = []
    for i in idx:
        try:
            rr = spo.brentq(fs, rg[i], rg[i + 1])
        except ValueError:
            continue
        slope = (fs(rr + h) - fs(rr - h)) / (2.0 * h)
        if slope == 0.0:
            continue
        out.append((float(rr), float(1.0 / abs(slope))))
    return out


def nf_momentum(x, omega, gamma, b=0.5):
    """The positive root p of omega = gamma x p + b p^3 (the normal form near a simple crossing, x = r - r_h)."""
    roots = np.roots([b, 0.0, gamma * x, -omega])
    real = roots[np.abs(roots.imag) < 1e-9 * np.maximum(1.0, np.abs(roots))].real
    pos = real[real > 0]
    if pos.size == 0:
        raise ValueError("no positive normal-form root")
    return float(pos.max())


def nf_delay(omega, d, Lrec, gamma, b=0.5):
    """t_nf = gamma^-1 ln(p(-d)/p(L)): the normal-form flight time from depth d below the crossing to distance L above it
    (dp/dt = -gamma p). Coefficient of ln(1/omega): 1/gamma for d >> delta_omega, 2/(3 gamma) for d = 0."""
    return math.log(nf_momentum(-d, omega, gamma, b) / nf_momentum(Lrec, omega, gamma, b)) / gamma


def nf_delta(omega, gamma, b=0.5):
    """delta_omega = b^(1/3) omega^(2/3)/gamma: the crossover depth of the normal form."""
    return b ** (1.0 / 3.0) * omega ** (2.0 / 3.0) / gamma


# ------------------------------------------------------------------ linear algebra for the transmissivity
def smax2_from_gram(G):
    """lambda_max and its eigenvector of the Hermitian Gram matrix G_ab = <P_D U f_a, P_D U f_b>: s_max^2(P_D U B_X)."""
    Gh = 0.5 * (G + G.conj().T)
    w, V = np.linalg.eigh(Gh)
    return float(max(w[-1], 0.0)), V[:, -1]


def lowdin_inverse_sqrt(S, tol=1e-10):
    """S^{-1/2} on the range of the Hermitian overlap matrix S (eigenvalues below tol * max dropped); returns (W, rank)."""
    Sh = 0.5 * (S + S.conj().T)
    w, V = np.linalg.eigh(Sh)
    keep = w > tol * max(float(w.max()), 1e-300)
    W = (V[:, keep] / np.sqrt(w[keep])[None, :]) @ V[:, keep].conj().T
    return W, int(keep.sum())
