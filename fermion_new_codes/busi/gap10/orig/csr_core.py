"""csr_core.py -- retarded cross-surface response on the 3-dimensional sine/current lattice.

The exact one-particle anticommutator {c_y(t), c_x^dagger(0)} = (exp(-i h t))_{yx} with
    h = h_0 + (1/2) sum_i {v^i, Q_i}                 (N = 1),
    h_0 = m beta + sum_i alpha_i (T_i - T_i^dagger)/(2i),   Q_i = (T_i^2 - T_i^dagger^2)/(4i),
    (T_i u)_x = u_{x+e_i},   a = 1,  hbar = c = 1,
on an open Cartesian parent (every link crossing the boundary is omitted). Real site fields give
endpoint means on the 2-link hops:
    (h u)_x = m beta u_x + sum_i alpha_i (u_{x+e_i} - u_{x-e_i})/(2i)
              + sum_i [ w^+_{i,x} u_{x+2e_i} - w^-_{i,x} u_{x-2e_i} ]/(4i),
    w^+_{i,x} = (v^i_x + v^i_{x+2e_i})/2,   w^-_{i,x} = (v^i_x + v^i_{x-2e_i})/2.
State layout: complex128 tensor (B, 4, L, L, L) = (batch, spinor, x, y, z); the site (x, y, z) sits at
index (x + c, y + c, z + c), c the centre index.
"""
import math
import numpy as np
import scipy.special as sps
import scipy.optimize as spo
import torch

CD = torch.complex128
FD = torch.float64
_trapz = getattr(np, "trapezoid", None) or np.trapz

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


# ------------------------------------------------------------------ kernel and radial profile
def kernel_K(s, ell):
    """K_ell(s) = 9009/(1024 pi ell^3) (1 - s^2/ell^2)_+^5, unit integral in 3 dimensions."""
    Cn = 9009.0 / (1024.0 * math.pi * ell ** 3)
    x = np.clip(1.0 - (np.asarray(s, dtype=np.float64) / ell) ** 2, 0.0, None)
    return Cn * x ** 5


def u_continuum(r, rs, Rb):
    """|v|(r): sqrt(rs/r) for r >= Rb; sqrt(rs/Rb) r/Rb for r < Rb (homogeneous regular interior)."""
    r = np.asarray(r, dtype=np.float64)
    scalar = (r.ndim == 0)
    r = np.atleast_1d(r)
    out = np.where(r < Rb, math.sqrt(rs / Rb) * r / Rb, np.sqrt(rs / np.maximum(r, 1e-300)))
    return out[0] if scalar else out


def l1_projected_profile(u_fn, ell, r_max, dr=0.125, drp=1.0 / 64.0):
    """V(r) = l = 1 projection of K_ell * (u(r) rhat): the smoothed radial magnitude.
    V(r) = (pi / r^2) int dr' u(r') I(r, r'),
    I(r, r') = int_{|r - r'|}^{min(r + r', ell)} s (r^2 + r'^2 - s^2) K_ell(s) ds
    (a degree-13 polynomial in s: 8-point Gauss-Legendre is exact)."""
    rg = np.arange(0.0, r_max + dr, dr)
    V = np.zeros_like(rg)
    xg, wg = np.polynomial.legendre.leggauss(8)
    for j, r in enumerate(rg):
        if r <= 0.0:
            continue
        lo, hi = max(0.0, r - ell), r + ell
        rp = np.linspace(lo, hi, int(round((hi - lo) / drp)) + 1)
        s_lo = np.abs(r - rp)
        s_hi = np.minimum(r + rp, ell)
        half = 0.5 * np.clip(s_hi - s_lo, 0.0, None)
        mid = 0.5 * (s_hi + s_lo)
        s = mid[:, None] + half[:, None] * xg[None, :]
        f = s * (r * r + rp[:, None] ** 2 - s * s) * kernel_K(s, ell)
        I = half * (f @ wg)
        V[j] = (math.pi / (r * r)) * _trapz(u_fn(rp) * I, rp)
    return rg, V


def build_site_flow(L, c, ell, u_fn, averaged, mode='radial', uniform_v=(0.0, 0.0, 0.0), ngauss=3):
    """Site flow v^i_x (negative = inward): 3 float64 arrays (L, L, L); plus the profile (rg, V) or None.
    averaged: v_x = cell average over x + [-1/2, 1/2]^3 (ngauss-point Gauss per axis) of (K_ell * v)(y);
    point sampling: v_x = -u(|x|) xhat."""
    if mode == 'flat':
        return [np.zeros((L, L, L)) for _ in range(3)], None
    if mode == 'uniform':
        return [np.full((L, L, L), float(uniform_v[i])) for i in range(3)], None
    idx = (np.arange(L) - c).astype(np.float64)
    X, Y, Z = np.meshgrid(idx, idx, idx, indexing='ij')
    if averaged:
        r_max = math.sqrt(3.0) * (c + 1.0) + ell + 2.0
        rg, V = l1_projected_profile(u_fn, ell, r_max)
        xg, wg = np.polynomial.legendre.leggauss(ngauss)
        xg, wg = 0.5 * xg, 0.5 * wg
        v = [np.zeros((L, L, L)) for _ in range(3)]
        for ia in range(ngauss):
            for ib in range(ngauss):
                for ic in range(ngauss):
                    xx, yy, zz = X + xg[ia], Y + xg[ib], Z + xg[ic]
                    r = np.sqrt(xx * xx + yy * yy + zz * zz)
                    rr = np.where(r > 0.0, r, 1.0)
                    f = -np.where(r > 0.0, np.interp(r, rg, V) / rr, 0.0)
                    wgt = wg[ia] * wg[ib] * wg[ic]
                    v[0] += wgt * f * xx
                    v[1] += wgt * f * yy
                    v[2] += wgt * f * zz
        return v, (rg, V)
    r = np.sqrt(X * X + Y * Y + Z * Z)
    rr = np.where(r > 0.0, r, 1.0)
    f = np.where(r > 0.0, -u_fn(rr) / rr, 0.0)
    return [f * X, f * Y, f * Z], None


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


def ghost_weights_np(v_full, L):
    """From the field on the (L+4)^3 halo grid (interior = halo index 2 .. L+1), the bond weights of the
    omitted links as (L, L) planes per axis, in the axis coordinate x = halo index - 2:
    near_lo = (v_{-1} + v_{+1})/2, far_lo = (v_{-2} + v_0)/2, near_hi = (v_L + v_{L-2})/2, far_hi = (v_{L+1} + v_{L-1})/2."""
    out = []
    for i in range(3):
        a = v_full[i]

        def plane(k, i=i, a=a):
            sl = [slice(2, L + 2)] * 3
            sl[i] = k
            return a[tuple(sl)]

        out.append(dict(near_lo=0.5 * (plane(1) + plane(3)), far_lo=0.5 * (plane(0) + plane(2)),
                        near_hi=0.5 * (plane(L + 2) + plane(L)), far_hi=0.5 * (plane(L + 3) + plane(L + 1))))
    return out


def spectral_bound(m, wp, wm):
    """||h|| <= sqrt(m^2 + 3) + max_x sum_i (|w^+_{i,x}| + |w^-_{i,x}|)/4  (compression + Schur test)."""
    rowsum = 0.25 * sum(np.abs(wp[i]) + np.abs(wm[i]) for i in range(3))
    return math.sqrt(m * m + 3.0) + float(rowsum.max())


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


class LatticeH:
    """h on the open parent, N = 1. apply(psi) returns h psi for psi of shape (B, 4, L, L, L).
    boundary_residual(psi) returns ||(1 - Pi) h Pi psi|| per batch element (the omitted-link amplitude)."""

    def __init__(self, m, wp_np, wm_np, device, ghost_np=None):
        self.m = float(m)
        self.device = device
        self.L = int(wp_np[0].shape[0])
        self.mbeta = torch.tensor(self.m * BETA_DIAG, dtype=CD, device=device).view(1, 4, 1, 1, 1)
        self.wp = [torch.tensor(w * (-0.25j), dtype=CD, device=device).view(1, 1, *w.shape) for w in wp_np]
        self.wm = [torch.tensor(w * (-0.25j), dtype=CD, device=device).view(1, 1, *w.shape) for w in wm_np]
        self.ghost = None
        if ghost_np is not None:
            self.ghost = [{k: torch.tensor(g[k], dtype=FD, device=device) for k in g} for g in ghost_np]
        self.napply = 0

    def apply(self, psi):
        out = psi * self.mbeta
        for i in range(3):
            dim = i + 2
            D = shift(psi, dim, 1)
            D.sub_(shift(psi, dim, -1))
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
            vec = self._alpha_face(f0, i) * (-0.5j) + f1 * ((-0.25j) * g['near_lo'])
            total += (vec.real ** 2 + vec.imag ** 2).sum((1, 2, 3))
            vec = f0 * ((-0.25j) * g['far_lo'])
            total += (vec.real ** 2 + vec.imag ** 2).sum((1, 2, 3))
            vec = self._alpha_face(fL1, i) * (0.5j) + fL2 * ((0.25j) * g['near_hi'])
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
def cheb_coefficients(dE, tau, dt_read, tol=1e-17, nsub=8):
    """Chebyshev order n, the readout times tr (within the sweep), the readout matrix
    C[j, k] = (2 - delta_k0)(-i)^k J_k(dE t_j) and the certificate integrals Ik[j, k] = 1.01 int_0^{t_j} |c_k|."""
    x = dE * tau
    K = int(x + 120 + 20.0 * x ** (1.0 / 3.0))
    J = sps.jv(np.arange(K + 1), x)
    n = int(np.flatnonzero(np.abs(J) > tol)[-1]) + 3
    nread = int(round(tau / dt_read))
    assert abs(nread * dt_read - tau) < 1e-9, "tau must be a multiple of dt_read"
    tr = dt_read * np.arange(1, nread + 1)
    kk = np.arange(n + 1)
    ph = np.array([1.0, -1.0j, -1.0, 1.0j])[kk % 4]
    fac = np.where(kk == 0, 1.0, 2.0)
    Jm = sps.jv(kk[None, :], dE * tr[:, None])
    Cm = (fac * ph)[None, :] * Jm
    tf = np.linspace(0.0, tau, nread * nsub + 1)
    Jf = np.abs(sps.jv(kk[None, :], dE * tf[:, None])) * fac[None, :]
    cum = np.zeros_like(Jf)
    cum[1:] = np.cumsum(0.5 * (Jf[1:] + Jf[:-1]) * (tf[1] - tf[0]), axis=0)
    Ik = 1.01 * cum[nsub::nsub]
    return n, tr, Cm.astype(np.complex128), Ik


def cheb_sweep(H, psi, dE, c_end, n, project, residual=True):
    """psi(t0 + tau) = sum_k c_end[k] T_k(h/dE) psi(t0); returns the new state, P[k] = project(T_k psi) and
    R[k] = ||B_Lambda T_k psi|| per batch element (None if residual is False)."""
    P = [project(psi)]
    R = [H.boundary_residual(psi)] if residual else None
    phi_prev = psi
    phi = H.apply(psi)
    phi.mul_(1.0 / dE)
    acc = psi * complex(c_end[0])
    axpy_(acc, phi, c_end[1])
    P.append(project(phi))
    if residual:
        R.append(H.boundary_residual(phi))
    for k in range(2, n + 1):
        new = H.apply(phi)
        new.mul_(2.0 / dE)
        new.sub_(phi_prev)
        axpy_(acc, new, c_end[k])
        P.append(project(new))
        if residual:
            R.append(H.boundary_residual(new))
        phi_prev, phi = phi, new
        del new
    del phi_prev, phi
    return acc, torch.stack(P, 0), (torch.stack(R, 0) if residual else None)


def taylor_terms(dE, dt, tol=1e-17):
    x = dE * dt
    J, term = 0, 1.0
    while not (term < tol and J > x):
        J += 1
        term *= x / J
    return J


def taylor_step(H, psi, dt, J, residual=True):
    """psi(t + dt) by the Taylor polynomial; returns the new state and the certificate increment
    dt * sum_j ||B_Lambda (-i dt h)^j psi / j!|| >= int_0^dt ||B_Lambda psi(t + s)|| ds."""
    acc = psi.clone()
    term = psi
    rsum = H.boundary_residual(psi) if residual else None
    for j in range(1, J + 1):
        new = H.apply(term)
        new.mul_(-1j * dt / j)
        acc.add_(new)
        term = new
        if residual:
            rsum = rsum + H.boundary_residual(term)
    return acc, (dt * rsum if residual else None)


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
    """Site receivers (4 components each) and packet groups (packets sharing 1 site index set)."""

    def __init__(self, L, c, device):
        self.L, self.c, self.device = L, c, device
        self.site_pos, self.site_meta, self.groups = [], [], []

    def add_site(self, pos, role, r):
        self.site_pos.append(np.asarray(pos, dtype=int))
        self.site_meta.append(dict(kind='site', role=role, r=float(r), pos=[int(p) for p in pos]))

    def add_packet_group(self, idx, values_list, metas):
        G = np.stack([np.conj(v.reshape(-1)) for v in values_list], 1)
        self.groups.append(dict(idx=torch.tensor(np.asarray(idx, dtype=np.int64), device=self.device),
                                G=torch.tensor(G, dtype=CD, device=self.device), metas=list(metas)))

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
                self.meta.append(d)
        for g in self.groups:
            self.meta.extend(g['metas'])
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
    """Radial shells, escaped-share masks, boundary-layer mask and the 2 planes."""

    def __init__(self, L, c, rs, r_h, device):
        idx = torch.arange(L, dtype=FD, device=device) - c
        X, Y, Z = torch.meshgrid(idx, idx, idx, indexing='ij')
        r = torch.sqrt(X * X + Y * Y + Z * Z)
        self.shell = torch.floor(r + 0.5).to(torch.int64).reshape(-1)
        self.nshell = int(self.shell.max().item()) + 1
        self.share_radii = [('r_h', float(r_h)), ('1.25rs', 1.25 * rs), ('1.5rs', 1.5 * rs), ('2rs', 2.0 * rs), ('3rs', 3.0 * rs)]
        self.masks = [(r > rad) for (_, rad) in self.share_radii]
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


# ------------------------------------------------------------------ dispersion, spinors, packets
def disp_omega(k, u, n, m):
    k = np.atleast_1d(np.asarray(k, dtype=float))
    z = k[:, None] * n[None, :]
    eps = np.sqrt(m * m + np.sum(np.sin(z) ** 2, 1))
    return eps - u * np.sum(n[None, :] * np.sin(2.0 * z), 1) / 2.0


def disp_vg(k, u, n, m):
    k = np.atleast_1d(np.asarray(k, dtype=float))
    z = k[:, None] * n[None, :]
    eps = np.sqrt(m * m + np.sum(np.sin(z) ** 2, 1))
    return np.sum(np.sin(z) * np.cos(z) * n[None, :], 1) / eps - u * np.sum((n * n)[None, :] * np.cos(2.0 * z), 1)


def outgoing_root(omega, u, n, m):
    """The unique outgoing positive-branch root k in (0, k_max) of H_+(k; u) = omega along n (m < omega)."""
    kmax = (math.pi / 2.0) / float(np.max(np.abs(n)))

    def f(k):
        return float(disp_omega(k, u, n, m)[0] - omega)

    f0, f1 = f(1e-9), f(kmax)
    if not (f0 < 0.0 < f1):
        raise ValueError(f"no propagating outgoing root: omega={omega}, u={u:.4f}, m={m}, f(0)={f0:.3e}, f(kmax)={f1:.3e}")
    return spo.brentq(f, 1e-9, kmax, xtol=1e-14, rtol=1e-14)


def helicity_spinors(k, n, m):
    """Positive-energy eigenvectors of h_0(p = k n) with helicity Sigma.n = +1 and -1 (in that order)."""
    n = np.asarray(n, dtype=float)
    p = k * n
    d = np.sin(p)
    eps = math.sqrt(m * m + float(d @ d))
    sn = sum(n[i] * SIGMA[i] for i in range(3))
    evals, evecs = np.linalg.eigh(sn)
    chis = [evecs[:, 1], evecs[:, 0]]  # ascending eigenvalues: index 1 is +1, index 0 is -1
    sd = sum(d[i] * SIGMA[i] for i in range(3))
    out = []
    for chi in chis:
        top = math.sqrt((eps + m) / (2.0 * eps)) * chi
        bottom = (sd @ chi) / math.sqrt(2.0 * eps * (eps + m))
        u = np.concatenate([top, bottom]).astype(np.complex128)
        out.append(u / np.linalg.norm(u))
    return out


def clock_offsets(r, rs):
    """F, Fu with T = t - F(r) (eq. example-chart) and u = t - Fu(r), Fu = F + r + rs ln(r/rs - 1); r > rs."""
    r = np.asarray(r, dtype=np.float64)
    F = 2.0 * np.sqrt(rs * r) + rs * np.log((np.sqrt(r) - math.sqrt(rs)) / (np.sqrt(r) + math.sqrt(rs)))
    return F, F + r + rs * np.log(r / rs - 1.0)


def chart_T(t, r, rs):
    return float(t - clock_offsets(r, rs)[0])


def chart_u(t, r, rs):
    return float(t - clock_offsets(r, rs)[1])


def packet_mode(L, c, pos, nhat, k, spinor, sigma, cut, rs, restriction='raw'):
    """Packet f_x = Z exp(-|x - pos|^2/(4 sigma^2)) exp(i k nhat.(x - pos)) chi on the ball |x - pos| <= cut sigma,
    clipped to the box and restricted to r < rs ('in'), r > rs ('out') or not ('raw'), then normalized.
    Returns flat site indices (n,), values (4, n), coords (n, 3), probabilities (n,) and metadata."""
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
    inbox = np.all(np.abs(coords) <= c, 1)
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


def band_weights(values, coords, pos, m, v_vec, nhat, hw):
    """Fourier transform of the packet on its bounding cube: weights on the upper band of h_0, on positive
    frequency of the local symbol omega_pm = v.q(p) pm eps(p), on outgoing positive frequency, and at folded
    momentum above pi/4."""
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
    safe = np.where(eps > 0, eps, 1.0)
    h0phi = m * BETA_DIAG[:, None, None, None] * phi
    for i in range(3):
        for (row, col, val) in ALPHA_NZ[i]:
            h0phi[row] += val * d[i] * phi[col]
    plus = 0.5 * (phi + np.where(eps > 0, h0phi / safe, 0.0))
    minus = phi - plus
    wp = (np.abs(plus) ** 2).sum(0)
    wm = (np.abs(minus) ** 2).sum(0)
    drag = sum(v_vec[i] * q[i] for i in range(3))
    om_p, om_m = drag + eps, drag - eps
    vg_p = sum(nhat[i] * (d[i] * np.cos(P[i]) / safe + v_vec[i] * np.cos(2 * P[i])) for i in range(3))
    vg_m = sum(nhat[i] * (-d[i] * np.cos(P[i]) / safe + v_vec[i] * np.cos(2 * P[i])) for i in range(3))
    folded = np.maximum.reduce([np.minimum(np.abs(x), math.pi - np.abs(x)) for x in P])
    return dict(norm=float(wp.sum() + wm.sum()), upper_band=float(wp.sum()),
                positive_frequency=float(wp[om_p > 0].sum() + wm[om_m > 0].sum()),
                outgoing_positive_frequency=float(wp[(om_p > 0) & (vg_p > 0)].sum() + wm[(om_m > 0) & (vg_m > 0)].sum()),
                outgoing_any=float(wp[vg_p > 0].sum() + wm[vg_m > 0].sum()),
                lattice_scale=float((wp + wm)[folded > math.pi / 4].sum()))


# ------------------------------------------------------------------ rays
def ray_time(r0, r1, omega, n, m, u_fn, dr=0.05):
    """Characteristic flight time on the outgoing positive branch from r0 to r1 at frequency omega along n,
    for the radial flow magnitude u_fn(r): int dr / c_g(r; omega)."""
    npts = max(int(math.ceil(abs(r1 - r0) / dr)) + 1, 3)
    rg = np.linspace(r0, r1, npts)
    inv = np.empty_like(rg)
    for j, r in enumerate(rg):
        u = float(u_fn(np.array([r]))[0])
        k = outgoing_root(omega, u, n, m)
        inv[j] = 1.0 / float(disp_vg(k, u, n, m)[0])
    return float(_trapz(inv, rg))


def crossings_between(u_fn, r_a, r_b, nscan=20001, h=0.25):
    """All radii in [min, max] with u = 1, each with its inverse |slope| (the log coefficient of that crossing)."""
    lo, hi = min(r_a, r_b), max(r_a, r_b)
    if hi <= lo:
        return []
    rg = np.linspace(lo, hi, nscan)
    g = np.asarray(u_fn(rg), dtype=float) - 1.0
    idx = np.flatnonzero(np.sign(g[:-1]) != np.sign(g[1:]))
    out = []
    for i in idx:
        try:
            rr = spo.brentq(lambda x: float(u_fn(np.array([x]))[0]) - 1.0, rg[i], rg[i + 1])
        except ValueError:
            continue
        slope = (float(u_fn(np.array([rr + h]))[0]) - float(u_fn(np.array([rr - h]))[0])) / (2.0 * h)
        if slope == 0.0:
            continue
        out.append((float(rr), float(1.0 / abs(slope))))
    return out
