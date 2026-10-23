#!/usr/bin/env python3
"""route2_common.py (busi lane, ROUTE2 v1; spec REVIEWA_write_ROUTE2_v1.md, sections S0 to S4).

The second route's shared code: its own radial-chain builder and zero-temperature filling, the family's Gaussian
kernel (re-typed from its definition) and the compact kernel, the radial rows by RK4, the cumulative windows, the
fixed-reference Hessian in closed form through enclosure products, the sea's spectral response applied as an
operator, and the Slater variances by the column and block formulas. Nothing here imports route 1's code
(engine_massfluct_v3, hf_star_variance_v3, hf_cluster13_v3); the self-test compares against engine_ops_v1 and
engine_fields_v2 only to verify the definitions (K-S0, the kernel test).

Conventions (S0). Chain (channel kappa, weight 16 kappa): two components per site, R_n = n + 1, on-site
m SZ + (kappa/R_n) SY, sine bond -(i/2) SX, two-step bond -(i/4) 1; clocked h_m = (1/2){N, h} + (1/2){v, Q}.
Vertex of a coefficient vector x = (x^e, x^p) (energy block first): A_x = (1/2){D_e, h} + (1/2){D_p, Q}.
Lattice units; G = g = G/a^2.
"""
import os
import json
import math
import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh

FAMILY = '<internal>/star_m0.1_g0.25_ell8.json'
ENGINE_DIR = '<internal>/code'
MF_DIR = '<internal>/MF'
CLOSED_MEMBERS = (1, 3, 5, 7, 9, 11, 13, 15)
FOURPI = 4.0 * math.pi
T0 = time.time()


def log(*a):
    print('[%7.1fs]' % (time.time() - T0), *a, flush=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, complex):
        return [o.real, o.imag]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return str(o)
    return o


def dump_json(path, obj):
    with open(path, 'w') as fh:
        json.dump(jsonable(obj), fh, indent=1)


def fnorm2(X):
    X = np.asarray(X)
    return float(np.vdot(X, X).real)


# ============================================================================ the family
def load_member(path, k):
    D = json.load(open(path))
    M = D['members'][k]
    mem = dict(path=path, member=k, m=float(M['m']), g=float(M['g']), nsites=int(M['nsites']),
               half=bool(D['args'].get('half_offset', 0)), ell=float(M.get('ell', D['args'].get('ell'))),
               N=np.array(M['N'], float), v=np.array(M['v'], float), E=float(M['E']), N_B=float(M['N_B']),
               eps_tot=np.array(M['eps_tot'], float), J_tot=np.array(M['J_tot'], float),
               E_top=M.get('E_top'), channels=M.get('channels'), kappa_max_occupied=M.get('kappa_max_occupied'),
               levels=M.get('levels'), R_surf=M.get('R_surf'), R_999=M.get('R_999'), r_collar=M.get('r_collar'),
               N_min=M.get('N_min'), failed=bool(M.get('failed', False)), passes=M.get('passes'),
               residual_H_max=M.get('residual_H_max'), residual_M_max=M.get('residual_M_max'))
    mem['R'] = np.arange(mem['nsites'], dtype=float) + (0.5 if mem['half'] else 1.0)
    return mem, D


# ============================================================================ the chain (route 2's builder)
def chain_ops(nsites, m, kappa, half=False):
    """sparse (csr) h (unit clock, no flow) and Q of channel kappa; R the site radii."""
    R = np.arange(nsites, dtype=float) + (0.5 if half else 1.0)
    n = np.arange(nsites)
    a, b = 2 * n, 2 * n + 1
    rows = [a, b, a, b]
    cols = [a, b, b, a]
    vals = [np.full(nsites, m, complex), np.full(nsites, -m, complex), -1j * kappa / R + 0j, 1j * kappa / R + 0j]
    nb = np.arange(nsites - 1)
    # bond (n, n+1): -(i/2) SX : (2n, 2n+3), (2n+1, 2n+2); and the conjugate transposes
    rows += [2 * nb, 2 * nb + 1, 2 * nb + 3, 2 * nb + 2]
    cols += [2 * nb + 3, 2 * nb + 2, 2 * nb, 2 * nb + 1]
    vals += [np.full(nsites - 1, -0.5j), np.full(nsites - 1, -0.5j), np.full(nsites - 1, 0.5j), np.full(nsites - 1, 0.5j)]
    h = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * nsites, 2 * nsites))
    nq = np.arange(nsites - 2)
    rq = [2 * nq, 2 * nq + 1, 2 * nq + 4, 2 * nq + 5]
    cq = [2 * nq + 4, 2 * nq + 5, 2 * nq, 2 * nq + 1]
    vq = [np.full(nsites - 2, -0.25j), np.full(nsites - 2, -0.25j), np.full(nsites - 2, 0.25j), np.full(nsites - 2, 0.25j)]
    Q = sp.csr_matrix((np.concatenate(vq), (np.concatenate(rq), np.concatenate(cq))), shape=(2 * nsites, 2 * nsites))
    return h, Q, R


def vertex(h, Q, xe, xp):
    """A = (1/2){D_e, h} + (1/2){D_p, Q} (sparse), xe, xp per site."""
    De = sp.diags(np.repeat(np.asarray(xe, float), 2))
    Dp = sp.diags(np.repeat(np.asarray(xp, float), 2))
    return (0.5 * (De @ h + h @ De) + 0.5 * (Dp @ Q + Q @ Dp)).tocsr()


def clocked(h, Q, N, v):
    return vertex(h, Q, N, v)


def spectrum(hm_dense):
    e, U = eigh(hm_dense, driver='evd', check_finite=False, overwrite_a=True)
    return e, np.asfortranarray(U)


# ============================================================================ the filling (route 2's own)
CACHE_DIR = os.environ.get('ROUTE2_CACHE', '<internal>/cache')


def channel_eig(N, v, m, nsites, kappa, half=False, use_cache=True):
    """eigh of the clocked chain of channel kappa; cached on disk under CACHE_DIR keyed by an md5 of (N, v, m, kappa, half)."""
    import hashlib
    h, Q, R = chain_ops(nsites, m, float(kappa), half)
    key = hashlib.md5(np.asarray(N, float).tobytes() + np.asarray(v, float).tobytes() + np.array([m, kappa, float(half)]).tobytes()).hexdigest()
    path = os.path.join(CACHE_DIR, 'ch_%s_k%d.npz' % (key, kappa))
    if use_cache and os.path.exists(path):
        Z = np.load(path)
        return h, Q, Z['e'], np.asfortranarray(Z['U'])
    e, U = spectrum(clocked(h, Q, N, v).toarray())
    if use_cache:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            tmp = path + '.tmp.npz'
            np.savez(tmp, e=e, U=U)
            os.replace(tmp, path)
        except OSError:
            pass
    return h, Q, e, U


def build_channels(N, v, m, nsites, NB, half=False, extra=0, keep=True, logf=log):
    """channels kappa = 1, 2, ... until two successive channels have their lowest positive mode above the top energy
    (the energy where the 16 kappa-weighted positive levels reach N_B); zero-temperature filling to N_B, a degenerate
    top multiplet (within 1e-9) filled fractionally. extra: sea-only channels appended (must receive no filling)."""
    chans = []
    above = 0
    kappa = 0
    while True:
        kappa += 1
        h, Q, e, U = channel_eig(N, v, m, nsites, kappa, half)
        chans.append(dict(kappa=kappa, w=16.0 * kappa, h=h, Q=Q, e=e, U=U if keep else None, extra=False))
        lv = np.concatenate([c['e'][c['e'] > 0] for c in chans])
        wv = np.concatenate([np.full(int(np.sum(c['e'] > 0)), c['w']) for c in chans])
        o = np.argsort(lv)
        cw = np.cumsum(wv[o])
        E_top_est = float(lv[o][int(np.searchsorted(cw, NB - 1e-9))]) if cw[-1] >= NB - 1e-9 else np.inf
        above = above + 1 if e[e > 0].min() > E_top_est else 0
        logf('  channel kappa=%d: lowest positive %.10f, top-energy estimate %.10f, successive above %d' % (kappa, e[e > 0].min(), E_top_est, above))
        if above >= 2:
            break
        if kappa > 400:
            raise RuntimeError('filling did not terminate')
    nrec = len(chans)
    for kk in range(nrec + 1, nrec + 1 + int(extra)):
        h, Q, e, U = channel_eig(N, v, m, nsites, kk, half)
        chans.append(dict(kappa=kk, w=16.0 * kk, h=h, Q=Q, e=e, U=U if keep else None, extra=True))
    lev = sorted([(float(c['e'][j]), c['w'], ic, j) for ic, c in enumerate(chans) for j in np.nonzero(c['e'] > 0)[0]])
    for c in chans:
        c['f'] = np.where(c['e'] < 0, 1.0, 0.0)
    left = NB
    i = 0
    E_top = None
    closed = True
    while i < len(lev) and left > 1e-12:
        j = i
        while j < len(lev) and abs(lev[j][0] - lev[i][0]) < 1e-9:
            j += 1
        wsum = sum(x[1] for x in lev[i:j])
        frac = min(1.0, left / wsum)
        if frac < 1.0 - 1e-12:
            closed = False
        for x in lev[i:j]:
            chans[x[2]]['f'][x[3]] = frac
        left -= frac * wsum
        E_top = lev[i][0]
        i = j
    NBf = sum(c['w'] * float(np.sum(c['f'][c['e'] > 0])) for c in chans)
    for c in chans:
        c['nneg'] = int(np.sum(c['e'] < 0))
        c['mat'] = np.nonzero((c['e'] > 0) & (c['f'] > 0))[0]
        c['n_mat'] = len(c['mat'])
    extra_fill = sum(c['w'] * float(np.sum(c['f'][c['e'] > 0])) for c in chans if c['extra'])
    info = dict(channels=nrec, channels_total=len(chans), E_top=E_top, N_B_filled=NBf, closed=closed, extra_fill=extra_fill,
                kappa_max_occupied=max([c['kappa'] for c in chans if c['n_mat'] > 0] or [0]),
                levels={str(c['kappa']): [float(c['e'][j]) for j in c['mat']] for c in chans if c['n_mat'] > 0},
                sea_is_half=all(c['nneg'] == nsites for c in chans))
    return chans, info


# ============================================================================ kernels
def gauss_kernels(r, Rs, ell):
    """the family's kernel (engine_fields_v2.kernels' definition, re-typed): K_l(r, R) = (2 pi ell^2)^(-3/2)
    exp(-(r^2 + R^2)/(2 ell^2)) i_l(r R/ell^2), i_0 = sinh z/z, i_1 = cosh z/z - sinh z/z^2."""
    r = np.asarray(r, float)[:, None]
    Rs = np.asarray(Rs, float)[None, :]
    z = r * Rs / ell ** 2
    em = np.exp(-(r - Rs) ** 2 / (2 * ell ** 2))
    ep = np.exp(-(r + Rs) ** 2 / (2 * ell ** 2))
    ea = np.exp(-(r ** 2 + Rs ** 2) / (2 * ell ** 2))
    small = z < 1e-2
    zs = np.where(small, 1.0, z)
    i0 = np.where(small, ea * (1 + z ** 2 / 6), (em - ep) / (2 * zs))                  # the definition's series cut (z < 1e-2)
    i1 = np.where(small, ea * (z / 3) * (1 + z ** 2 / 10), ((zs - 1) * em + (zs + 1) * ep) / (2 * zs ** 2))
    pref = 1.0 / ((2 * math.pi) ** 1.5 * ell ** 3)
    return pref * i0, pref * i1


def compact_kernels(r, Rs, ell):
    """l = 0 and l = 1 shell projections of k(s) = kappa (1 - s^2/ell^2)^5_+, kappa = 9009/(1024 pi ell^3) (GAP06's
    compact kernel, the declared definition): K_0 = [F1(r+R) - F1(|r-R|)]/(2 r R), K_1 = [(r^2+R^2) dF1 - dF3]/(4 r^2 R^2),
    F1' = s k(s), F3' = s^3 k(s)."""
    kap = 9009.0 / (1024.0 * math.pi * ell ** 3)
    t = lambda s: np.clip(1.0 - (s / ell) ** 2, 0.0, None)
    F1 = lambda s: -kap * ell ** 2 / 12.0 * t(s) ** 6
    F3 = lambda s: kap * ell ** 4 / 2.0 * (-t(s) ** 6 / 6.0 + t(s) ** 7 / 7.0)
    r = np.asarray(r, float)
    c = np.asarray(Rs, float)[None, :]
    K0 = np.zeros((len(r), c.shape[1]))
    K1 = np.zeros_like(K0)
    nz = r > 0
    rr = r[nz][:, None]
    dF1 = F1(rr + c) - F1(np.abs(rr - c))
    dF3 = F3(rr + c) - F3(np.abs(rr - c))
    K0[nz] = dF1 / (2 * rr * c)
    K1[nz] = ((rr ** 2 + c ** 2) * dF1 - dF3) / (4 * rr ** 2 * c ** 2)
    if (~nz).any():
        K0[~nz] = kap * t(c) ** 5
    return K0, K1


def simpson_weights(n, h):
    W = np.ones(n + 1)
    W[1:-1:2] = 4.0
    W[2:-1:2] = 2.0
    return W * h / 3.0


def cumsimpson(f, h):
    """cumulative Simpson along axis 0 (even number of intervals; the odd points by the 3-point partial panel)."""
    f = np.asarray(f)
    out = np.zeros(f.shape, dtype=np.result_type(f, np.float64))
    pan = h / 3.0 * (f[0:-1:2] + 4.0 * f[1::2] + f[2::2])
    out[2::2] = np.cumsum(pan, axis=0)
    out[1::2] = out[0:-1:2] + h / 12.0 * (5.0 * f[0:-1:2] + 8.0 * f[1::2] - f[2::2])
    return out


class Grid:
    """the fine grid r = 0 .. R_out at spacing dr (even number of intervals), the kernel tables K_0 (normalized column
    by column under the grid's Simpson weights) and K_1, and the half-step source evaluation for RK4."""

    def __init__(self, Rs, ell, R_out, dr, kind='gauss', chunk=2000):
        n = int(round(R_out / dr))
        n += n % 2
        self.n, self.dr, self.R_out = n, R_out / n, R_out
        self.r = np.linspace(0.0, R_out, n + 1)
        self.W = simpson_weights(n, self.dr)
        self.Rs, self.ell, self.kind = np.asarray(Rs, float), float(ell), kind
        self.kfun = gauss_kernels if kind == 'gauss' else compact_kernels
        K0, K1 = self.kfun(self.r, self.Rs, self.ell)
        w2 = self.W * FOURPI * self.r ** 2
        self.norm_raw = w2 @ K0
        self.K0 = K0 / self.norm_raw[None, :]
        self.K1 = K1
        self.norm_err = float(np.max(np.abs(w2 @ self.K0 - 1.0)))
        self.norm_raw_err = float(np.max(np.abs(self.norm_raw - 1.0)))
        self.mom1_err = float(np.max(np.abs((w2 * self.r) @ self.K1 - self.Rs) / self.Rs))
        self.rh = self.r[:-1] + 0.5 * self.dr

    def half_tables(self):
        if not hasattr(self, 'K0h'):
            K0h = np.zeros((self.n, len(self.Rs)))
            K1h = np.zeros((self.n, len(self.Rs)))
            for i0 in range(0, self.n, 2000):
                sl = slice(i0, min(i0 + 2000, self.n))
                a0, a1 = self.kfun(self.rh[sl], self.Rs, self.ell)
                K0h[sl] = a0 / self.norm_raw[None, :]
                K1h[sl] = a1
            self.K0h, self.K1h = K0h, K1h
        return self.K0h, self.K1h

    def sources(self, e, j):
        """rho, J on the grid and at the half points (the half points by the kernel formula with the same column
        normalization)."""
        K0h, K1h = self.half_tables()
        return self.K0 @ e, self.K1 @ j, K0h @ e, K1h @ j


def rows_from_profiles(r, h, rho, J, rho_h, J_h, G):
    """M' = 4 pi r^2 (rho + w J), w = -sqrt(2 G M/r) by RK4 at step h on the grid r (rho_h, J_h at the half points);
    (ln N)' = -4 pi G r J/w by cumulative Simpson; N(r[-1]) = 1; v = N w."""
    n = len(r) - 1
    M = np.zeros(n + 1)

    def f(rr, MM, rh, Jh):
        if rr <= 0.0:
            return 0.0
        w = -math.sqrt(2.0 * G * max(MM, 0.0) / rr)
        return FOURPI * rr * rr * (rh + w * Jh)

    for k in range(n):
        r0, m0 = r[k], M[k]
        k1 = f(r0, m0, rho[k], J[k])
        k2 = f(r0 + 0.5 * h, m0 + 0.5 * h * k1, rho_h[k], J_h[k])
        k3 = f(r0 + 0.5 * h, m0 + 0.5 * h * k2, rho_h[k], J_h[k])
        k4 = f(r0 + h, m0 + h * k3, rho[k + 1], J[k + 1])
        M[k + 1] = m0 + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    if np.min(M[1:]) <= 0:
        raise RuntimeError('M(r) <= 0 on the grid')
    w = np.zeros_like(M)
    w[1:] = -np.sqrt(2.0 * G * M[1:] / r[1:])
    q = np.zeros_like(M)
    q[1:] = -FOURPI * G * r[1:] * J[1:] / w[1:]
    lnN = cumsimpson(q, h)
    lnN -= lnN[-1]
    N = np.exp(lnN)
    return dict(M=M, w=w, N=N, v=N * w, rho=rho, J=J, E=M[-1])


def rows_rk4(grid, e, j, G):
    """the radial rows from shell sources e, j through the grid's kernel tables."""
    rho, J, rho_h, J_h = grid.sources(e, j)
    return rows_from_profiles(grid.r, grid.dr, rho, J, rho_h, J_h, G)


def gradient_E(grid, sol):
    """dE/ds = the windows at R_out: (int 4 pi r^2 N K_0, int 4 pi r^2 v K_1)/N(R_out)."""
    wr = grid.W * FOURPI * grid.r ** 2
    return np.concatenate([(wr * sol['N']) @ grid.K0, (wr * sol['v']) @ grid.K1]) / sol['N'][-1]


def windows(grid, sol):
    """cumulative windows a_n(s) = int_0^s 4 pi r^2 N K_0(r, R_n) dr and c_n(s) = int_0^s 4 pi r^2 v K_1 dr."""
    pref = FOURPI * grid.r ** 2
    a = cumsimpson((pref * sol['N'])[:, None] * grid.K0, grid.dr)
    c = cumsimpson((pref * sol['v'])[:, None] * grid.K1, grid.dr)
    return a, c


def window_at(a, c, sol, i):
    """b(R) = (f, g) at the grid index i: f = a(R)/N(R), g = c(R)/N(R)."""
    return np.concatenate([a[i], c[i]]) / sol['N'][i]


def hessian_closed(grid, sol, a, c, G):
    """the fixed-reference Hessian of E in the shell sources, by the closed-form blocks (REVIEWA API's, verified in
    check C) through enclosure products: B^ee = G^2 a^T diag(j3 W) a; B^ep = G a^T diag(W z 4 pi s^2) K_1 +
    G^2 a^T diag(j3 W) c; B^pp = G [c^T diag(W z 4 pi s^2) K_1 + transpose] + G^2 c^T diag(j3 W) c;
    j3 = 4 pi J/(N |w|^3), z = 1/(s w)."""
    r, W = grid.r, grid.W
    N, w, J = sol['N'], sol['w'], sol['J']
    j3 = np.zeros_like(r)
    z = np.zeros_like(r)
    j3[1:] = FOURPI * J[1:] / (N[1:] * np.abs(w[1:]) ** 3)
    z[1:] = 1.0 / (r[1:] * w[1:])
    dj = (j3 * W)[:, None]
    dz = (W * z * FOURPI * r ** 2)[:, None]
    Bee = G * G * (a.T @ (dj * a))
    Bep = G * (a.T @ (dz * grid.K1)) + G * G * (a.T @ (dj * c))
    T = c.T @ (dz * grid.K1)
    Bpp = G * (T + T.T) + G * G * (c.T @ (dj * c))
    B = np.block([[Bee, Bep], [Bep.T, Bpp]])
    return B


# ============================================================================ variances and the response
def channel_split(ch):
    e, U = ch['e'], ch['U']
    no = ch['nneg']
    return U[:, :no], U[:, no:], e[:no], e[no:]


def band_diag_of_op_pi(op, Z, Uu):
    """(op Pi)_ii for Pi = Y + Y^dag, Y = Z Uu^dag, op banded (sparse): sum_d op[i, i+d] (Y[i+d, i] + conj(Y[i, i+d]));
    only the band of Y is formed (row dot products), never Y itself."""
    n = Z.shape[0]
    out = np.zeros(n, complex)
    opc = op.tocsr()
    offs = sorted(set((opc.nonzero()[1] - opc.nonzero()[0]).tolist()))
    Ucj = Uu.conj()
    for d in offs:
        dg = opc.diagonal(d)             # op[i, i+d], i = 0 .. n-1-d (d >= 0) or i = -d .. n-1 (d < 0), length n - |d|
        if d >= 0:
            i = np.arange(n - d)
        else:
            i = np.arange(-d, n)
        # Y[i, i+d] = sum_l Z[i, l] conj(Uu[i+d, l]);  Y[i+d, i] = sum_l Z[i+d, l] conj(Uu[i, l])
        Yid = np.sum(Z[i] * Ucj[i + d], axis=1)
        Ydi = np.sum(Z[i + d] * Ucj[i], axis=1)
        out[i] += dg * (Ydi + np.conj(Yid))
    return out


def response_apply(chans, x, ns):
    """y = P^A x: y_mu = sum_kappa 16 kappa Tr(A_mu Pi_kappa(A_x)), Pi(A_x) = U_o M U_u^dag + h.c.,
    M = (U_o^dag A_x U_u)/(e_o - e_u); read off as the site energy and current densities of Pi(A_x),
    Tr(hcal_n Pi) = Re sum_{i in n} (h Pi)_ii (the band of Pi only)."""
    y = np.zeros(2 * ns)
    for ch in chans:
        Uo, Uu, eo, eu = channel_split(ch)
        A = vertex(ch['h'], ch['Q'], x[:ns], x[ns:])
        Wm = Uo.conj().T @ (A @ Uu)
        Mm = Wm / (eo[:, None] - eu[None, :])
        Z = Uo @ Mm
        for op, sl in ((ch['h'], slice(0, ns)), (ch['Q'], slice(ns, 2 * ns))):
            d = np.real(band_diag_of_op_pi(op, Z, Uu))
            y[sl] += ch['w'] * d.reshape(-1, 2).sum(axis=1)
    return y


def response_apply_dense(ch, x, ns):
    """the same for one channel with Pi formed densely (the check of the band route)."""
    Uo, Uu, eo, eu = channel_split(ch)
    A = vertex(ch['h'], ch['Q'], x[:ns], x[ns:])
    Mm = (Uo.conj().T @ (A @ Uu)) / (eo[:, None] - eu[None, :])
    Pi = Uo @ Mm @ Uu.conj().T
    Pi = Pi + Pi.conj().T
    y = np.zeros(2 * ns)
    for op, sl in ((ch['h'], slice(0, ns)), (ch['Q'], slice(ns, 2 * ns))):
        d = np.real(np.diag((op @ Pi)))
        y[sl] += ch['w'] * d.reshape(-1, 2).sum(axis=1)
    return y


def response_matrix_element(ch, xa, xb, ns):
    """x_a . P^A x_b by the spec's formula sum_ou 2 Re[(A_a)_ou (A_b)_uo]/(e_o - e_u) (one channel, weighted)."""
    Uo, Uu, eo, eu = channel_split(ch)
    Aa = Uo.conj().T @ (vertex(ch['h'], ch['Q'], xa[:ns], xa[ns:]) @ Uu)
    Ab = Uu.conj().T @ (vertex(ch['h'], ch['Q'], xb[:ns], xb[ns:]) @ Uo)
    return ch['w'] * float(np.sum(2 * np.real(Aa * Ab.T) / (eo[:, None] - eu[None, :])))


def channel_variances(ch, x, ns, eigen_check=False):
    """per copy (no weight): Var_P by the block formula ||U_u^dag A U_o||^2; Var_C by the column formula
    ||A Phi||^2 - ||Phi^dag A Phi||^2 over the occupied columns Phi = [U_o, U_m] (integer occupations), evaluated in its
    reduced form ||U_u^dag A U_o||^2 + ||A U_m||^2 - 2 ||U_m^dag A U_o||^2 - ||U_m^dag A U_m||^2 (the identity
    ||A U_o||^2 - ||U_o^dag A U_o||^2 = ||U_u^dag A U_o||^2 removes the large cancelling pair). eigen_check: also the
    unreduced column formula and the eigenbasis formula sum f (1 - f') |A_kk'|^2."""
    Uo, Uu, eo, eu = channel_split(ch)
    f = ch['f']
    assert np.all(f[ch['mat']] > 1 - 1e-12), 'the column formula needs integer occupations (closed multiplets)'
    A = vertex(ch['h'], ch['Q'], x[:ns], x[ns:])
    Xo = A @ Uo
    Y = Uu.conj().T @ Xo
    VarP = fnorm2(Y)
    if len(ch['mat']):
        Um = ch['U'][:, ch['mat']]
        Xm = A @ Um
        Gmo = Um.conj().T @ Xo
        Gmm = Um.conj().T @ Xm
        VarC = VarP + fnorm2(Xm) - 2 * fnorm2(Gmo) - fnorm2(Gmm)
    else:
        VarC = VarP
    out = dict(VarC=VarC, VarP=VarP)
    if eigen_check:
        Goo = Uo.conj().T @ Xo
        if len(ch['mat']):
            out['VarC_column_unreduced'] = fnorm2(Xo) + fnorm2(Xm) - fnorm2(Goo) - 2 * fnorm2(Gmo) - fnorm2(Gmm)
        else:
            out['VarC_column_unreduced'] = fnorm2(Xo) - fnorm2(Goo)
        At = ch['U'].conj().T @ (A @ ch['U'])
        A2 = np.abs(At) ** 2
        out['VarC_eigen'] = float(f @ A2 @ (1 - f))
        fs = np.where(ch['e'] < 0, 1.0, 0.0)
        out['VarP_eigen'] = float(fs @ A2 @ (1 - fs))
    return out


def variances(chans, x, ns, eigen_check_first=False):
    per = []
    VC = VP = 0.0
    for i, ch in enumerate(chans):
        r = channel_variances(ch, x, ns, eigen_check=(eigen_check_first and i == 0))
        r['kappa'] = ch['kappa']
        r['w'] = ch['w']
        per.append(r)
        VC += ch['w'] * r['VarC']
        VP += ch['w'] * r['VarP']
    return VC, VP, per


def matter_count_inside(chans, ic):
    """N_r: the matter's weighted count inside site ic (inclusive)."""
    tot = 0.0
    for ch in chans:
        if ch['n_mat'] == 0:
            continue
        Um = ch['U'][:2 * (ic + 1), ch['mat']]
        tot += ch['w'] * float(np.sum(ch['f'][ch['mat']] * np.sum(np.abs(Um) ** 2, axis=0)))
    return tot


def site_densities(ch, cols):
    """matter's site energies and currents of the columns cols (unit occupation): Re u^dag P_n h u per site."""
    U = ch['U'][:, cols]
    de = np.real(np.sum(U.conj() * (ch['h'] @ U), axis=1)).reshape(-1, 2).sum(axis=1)
    dp = np.real(np.sum(U.conj() * (ch['Q'] @ U), axis=1)).reshape(-1, 2).sum(axis=1)
    return de, dp
