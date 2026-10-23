#!/usr/bin/env python3
"""
j4m_lib_v1.py -- the massive reduction of the genuine ball cut and its kernel tables (busi lane, task 4;
specification busi/specs/J4_massive_spec_v1.md, items (a) and (b)).  Imports j4lib and j4lib_v2 from
busi/code/j4 unchanged.  Python 3, numpy, scipy.

Model.  h_mu(k) = mu beta + sum_i alpha_i sin k_i (alpha_i = tau_x (x) sigma_i, beta = tau_z (x) 1, as j4_dense.py),
occupied projector P_-(k) = (1 - h_mu/eps_mu)/2, eps_mu = sqrt(mu^2 + sum_i sin^2 k_i), on the even antiperiodic
box L (k_i = 2 pi (j_i + 1/2)/L).  Then
    C(x,y) = delta/2 - (i/2) sum_i G_i(x-y) alpha_i - (mu/2) Gm(x-y) beta,
    G_i(n) = Im (1/L^3) sum_k (sin k_i/eps_mu) e^{ik.n}    (j4lib's kinetic kernel with eps_mu; one odd component),
    Gm(n)  = Re (1/L^3) sum_k e^{ik.n}/eps_mu             (the mass kernel; all components even).
Reduction (spec section 1.2).  With Omega_x = alpha_1^{x_1} alpha_2^{x_2} alpha_3^{x_3} the field is 2 copies of
C_+ = (1 - iY - mu E Gm)/2 and 2 of C_- = E (1 - C_+) E, so S_full(A) = 4 Tr s(C_{+,A}); with the odd-site phase C_+
is real symmetric:
    C(x,z) = delta_{xz}/2 + (eps_x/2) [ sum_i eta_i(x) G_i(x-z) - mu Gm(x-z) ],   eps_x = (-1)^{x_1+x_2+x_3}.
The gauge-signed O_h of j4lib commutes with it; the symmetry-adapted blocks are the full (even + odd) blocks per
irrep, m_lambda = m_e + m_o, B_lambda = (1 + K_lambda^full)/2, and S = 4 sum_lambda d_lambda sum_j s(c_j).
"""
import os
import sys
import tempfile
import time

import numpy as np
from scipy import linalg

HERE = os.path.dirname(os.path.abspath(__file__))
J4_DIR = os.path.normpath(os.path.join(HERE, '..', 'j4'))
for _p in (HERE, J4_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import j4lib as jl          # noqa: E402
import j4lib_v2 as jl2      # noqa: E402

LN2 = jl.LN2
IRREP_NAMES = list(jl.IRREP_NAMES)


def mu_tag(mu):
    """Canonical text of a mass for file names and case ids: 0 -> '0', 0.2 -> '0.2'."""
    s = ('%.6f' % float(mu)).rstrip('0').rstrip('.')
    return '0' if s in ('', '-0') else s


def class_eps(c):
    """eps_x = (-1)^{x_1+x_2+x_3} for the parity class c."""
    return -1.0 if int(jl.class_bits(c).sum()) % 2 else 1.0


# ----------------------------------------------------------------------------------------------
# the massive antiperiodic box kernel
# ----------------------------------------------------------------------------------------------
class MassiveBoxKernel(jl.MasslessKernel):
    """Antiperiodic box of even side L at mass mu: G1 (kinetic, j4lib's convention with eps_mu) and Gm (mass kernel).
    g1, g_component and staggered_table are inherited from j4lib unchanged; gm and mass_table are new."""

    def __init__(self, L, mu):
        L = int(L)
        mu = float(mu)
        if L % 2 or L < 4:
            raise ValueError('L must be even and at least 4')
        self.L, self.mu = L, mu
        k = 2.0 * np.pi * (np.arange(L) + 0.5) / L
        s = np.sin(k)
        eps = s[:, None, None] ** 2 + s[None, :, None] ** 2 + s[None, None, :] ** 2
        eps += mu * mu
        np.sqrt(eps, out=eps)
        ph = np.exp(1j * np.pi * np.arange(L) / L)
        F = np.fft.ifftn(s[:, None, None] / eps)
        F *= ph[:, None, None]
        F *= ph[None, :, None]
        F *= ph[None, None, :]
        self.real_part_max = float(np.abs(F.real).max())
        self.G1 = np.ascontiguousarray(F.imag)
        del F
        F = np.fft.ifftn(1.0 / eps)
        del eps
        F *= ph[:, None, None]
        F *= ph[None, :, None]
        F *= ph[None, None, :]
        self.imag_part_max = float(np.abs(F.imag).max())
        self.Gm = np.ascontiguousarray(F.real)
        del F

    def gm(self, r):
        """Gm at integer displacements r (..., 3) of any size, with the antiperiodic sign."""
        r = np.asarray(r, dtype=np.int64)
        wraps = np.sum(r // self.L, axis=-1)
        sign = 1.0 - 2.0 * (wraps % 2)
        idx = r % self.L
        return self.Gm[idx[..., 0], idx[..., 1], idx[..., 2]] * sign

    def mass_table(self, M):
        """Mtab(r) = Gm(r) on the all-even displacements of [-M, M]^3, zero elsewhere; (table, max |discarded|)."""
        M = int(M)
        T = 2 * M + 1
        ax = np.arange(-M, M + 1, dtype=np.int64)
        R = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), axis=-1)
        vals = self.gm(R)
        alleven = np.all(R % 2 == 0, axis=-1)
        junk = float(np.abs(vals[~alleven]).max()) if np.any(~alleven) else 0.0
        tab = np.where(alleven, vals, 0.0)
        return np.ascontiguousarray(tab.reshape(T, T, T)), junk

    def tables(self, M):
        """(gtab, mtab, diag): the one-odd kinetic table (j4lib's staggered_table) and the all-even mass table."""
        gtab, junk_g = self.staggered_table(M)
        mtab, junk_m = self.mass_table(M)
        diag = dict(L=self.L, mu=self.mu, M=int(M), junk=float(max(junk_g, junk_m)), junk_kinetic=float(junk_g),
                    junk_mass=float(junk_m), real_part_max=self.real_part_max, imag_part_max=self.imag_part_max)
        return gtab, mtab, diag


class TableKernel:
    """Kernel lookups from tables on [-M, M]^3 (dense checks and the infinite-lattice route): same interface as the box kernel."""

    def __init__(self, gtab, mtab, M):
        self.gtab, self.mtab, self.M = np.asarray(gtab), np.asarray(mtab), int(M)
        if self.gtab.shape != (2 * self.M + 1,) * 3 or self.mtab.shape != self.gtab.shape:
            raise ValueError('table shapes do not match M')

    def _idx(self, r):
        r = np.asarray(r, dtype=np.int64)
        if r.size and int(np.abs(r).max()) > self.M:
            raise ValueError('displacement outside the table range')
        return r + self.M

    def g_component(self, i, r):
        r = np.asarray(r, dtype=np.int64)
        idx = self._idx(r)
        odd = r % 2
        mask = (odd.sum(axis=-1) == 1) & (odd[..., i] == 1)
        return np.where(mask, self.gtab[idx[..., 0], idx[..., 1], idx[..., 2]], 0.0)

    def gm(self, r):
        idx = self._idx(r)
        return self.mtab[idx[..., 0], idx[..., 1], idx[..., 2]]


# ----------------------------------------------------------------------------------------------
# tables per parity class and the full-block assembly
# ----------------------------------------------------------------------------------------------
def signed_q_tables(gtab, mtab, M, mu, classes, drop_mass=False):
    """q_c(r) = eps_c [eta_{i(r)}(c) G_{i(r)}(r) - mu Gm(r)] as flat tables over [-M, M]^3, one per parity class.
    drop_mass=True reproduces the mixed-covariance shortcut (for the tests only)."""
    ytabs = jl.signed_tables(gtab, M, classes)
    mflat = np.ascontiguousarray(mtab).ravel()
    out = {}
    for c in classes:
        e = class_eps(c)
        q = e * ytabs[int(c)]
        if float(mu) != 0.0 and not drop_mass:
            q = q - (e * float(mu)) * mflat
        out[int(c)] = np.ascontiguousarray(q)
    return out


def rep_order(sym):
    """Orbit representatives ordered even (class-sorted, j4lib's order) then odd (class-sorted): every parity class contiguous."""
    odd = sym.odd
    odd_sorted = odd[np.argsort(sym.rep_class[odd], kind='stable')]
    return np.concatenate([sym.even, odd_sorted]).astype(np.int64)


def assemble_full_blocks(sym, names, qtabs, chunk_elems=1.6e7, log=None):
    """The full symmetry-adapted blocks K_lambda^full (m_lambda x m_lambda, m_lambda = m_e + m_o) of the real symmetric
    scalar matrix Q(x, z) = q_{class(x)}(x - z) for the irreps in `names`, by j4lib's construction extended to both
    sublattices:  K[(x,j),(y,j')] = [W_x^T M(x,y) W_y]_{jj'} / sqrt(|H_x| |H_y|),  M(x,y) = sum_k D(k) phi_k(k y) Q(x, k y).
    Returns (dict name -> array, the representative order)."""
    nG = len(sym.G)
    T = 2 * sym.M + 1
    const = sym.M * (T * T + T + 1)

    def code(v):
        return v[..., 0] * (T * T) + v[..., 1] * T + v[..., 2]

    order = rep_order(sym)
    N = len(order)
    zcode = code(sym.imgs[order])                      # (N, 48)
    sgo = sym.sg[order].astype(float)                  # (N, 48)
    data = {nm: sym.irrep_data(nm) for nm in names}
    K, Dflat, colsel, cfac, off = {}, {}, {}, {}, {}
    for nm in names:
        d = data[nm]['d']
        mr = data[nm]['m_rep'][order]
        mtot = int(mr.sum())
        K[nm] = np.zeros((mtot, mtot))
        Dflat[nm] = np.ascontiguousarray(sym.D[nm].reshape(nG, -1))
        colsel[nm] = (np.concatenate([np.arange(b * d, b * d + mr[b]) for b in range(N)]).astype(np.int64)
                      if N else np.zeros(0, dtype=np.int64))
        cfac[nm] = np.ascontiguousarray(data[nm]['fac'][order])
        off[nm] = np.concatenate([[0], np.cumsum(mr)]).astype(np.int64)
    A = max(1, int(chunk_elems // max(1, N * nG)))
    cls = sym.rep_class[order]
    t0 = time.time()
    done = 0
    for c in np.unique(cls):
        qtab = qtabs[int(c)]
        sel = np.where(cls == c)[0]
        if len(sel) > 1 and np.any(np.diff(sel) != 1):
            raise ArithmeticError('representatives of one parity class are not contiguous')
        for s0 in range(0, len(sel), A):
            pos = sel[s0:s0 + A]
            xcode = code(sym.reps[order[pos]])
            idx = xcode[:, None, None] - zcode[None, :, :] + const
            vals = qtab[idx]
            vals *= sgo[None, :, :]
            vflat = vals.reshape(-1, nG)
            for nm in names:
                d = data[nm]['d']
                Mb = (vflat @ Dflat[nm]).reshape(len(pos), N, d, d)
                lfacT = np.ascontiguousarray(np.transpose(data[nm]['fac'][order[pos]], (0, 2, 1)))
                Mb = np.matmul(lfacT[:, None, :, :], Mb)
                Mb = np.matmul(Mb, cfac[nm][None, :, :, :])
                blk = np.transpose(Mb, (0, 2, 1, 3)).reshape(len(pos) * d, N * d)[:, colsel[nm]]
                mr_rows = data[nm]['m_rep'][order[pos]]
                rowsel = np.concatenate([np.arange(a * d, a * d + mr_rows[a]) for a in range(len(pos))]).astype(np.int64)
                r0 = int(off[nm][pos[0]])
                r1 = int(off[nm][pos[-1] + 1])
                K[nm][r0:r1, :] = blk[rowsel, :]
            done += len(pos)
            if log is not None and (done % (50 * A) < len(pos) or done == N):
                log('  full assembly %s: %d/%d orbits, %.0f s' % ('+'.join(names), done, N, time.time() - t0))
    return K, order


def plan_full_groups(sym, names, mem_limit_bytes):
    """Greedy grouping of irreps by the memory of their full blocks (block + the eigensolver's workspace)."""
    info = sorted(((8 * (sym.irrep_data(nm)['m_e'] + sym.irrep_data(nm)['m_o']) ** 2, nm) for nm in names), reverse=True)
    groups = []
    for kb, nm in info:
        for grp in groups:
            if sum(x[0] for x in grp) + kb + 2 * max([x[0] for x in grp] + [kb]) <= mem_limit_bytes:
                grp.append((kb, nm))
                break
        else:
            groups.append([(kb, nm)])
    return [[x[1] for x in grp] for grp in groups]


def full_block_entropy(K, method='eigh', asym_tol=1e-10):
    """(sum_j s(c_j), eigenvalues, info) of the covariance block B = (1 + (K + K^T)/2)/2.  K is destroyed."""
    m = int(K.shape[0])
    if m == 0:
        return 0.0, np.zeros(0), dict(asym=0.0, ev_min=None, ev_max=None)
    asym = float(np.abs(K - K.T).max())
    if asym > asym_tol:
        raise ArithmeticError('full block is not symmetric: %g' % asym)
    B = K + K.T
    del K
    B *= 0.25
    B.flat[::m + 1] += 0.5
    if method == 'eigh':
        ev = linalg.eigvalsh(B, overwrite_a=True, check_finite=False, driver='evd')
    elif method == 'torch':
        import fcntl
        import torch
        with open(os.environ.get('J4_GPU_LOCK', '<internal>/j4_gpu.lock'), 'w') as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            Bd = torch.from_numpy(np.ascontiguousarray(B)).to('cuda')
            ev = torch.linalg.eigvalsh(Bd).cpu().numpy()
            del Bd
            torch.cuda.empty_cache()
            fcntl.flock(lockf, fcntl.LOCK_UN)
    else:
        raise ValueError('unknown method ' + str(method))
    del B
    ev = np.asarray(ev, dtype=float)
    ev_min, ev_max = float(ev.min()), float(ev.max())
    if ev_min < -1e-10 or ev_max > 1.0 + 1e-10:
        raise ArithmeticError('block spectrum outside [0, 1]: %g, %g' % (ev_min, ev_max))
    ev = np.clip(ev, 0.0, 1.0)
    return float(np.sum(jl.binary_entropy(ev))), ev, dict(asym=asym, ev_min=ev_min, ev_max=ev_max)


def massive_entropy_from_tables(sym, gtab, mtab, mu, method='eigh', names=None, chunk_elems=1.6e7, mem_limit_gb=16.0,
                                drop_mass=False, keep_values=True, log=None):
    """S = 4 sum_lambda d_lambda sum_j s(c_j) of the ball `sym` from the kinetic and mass tables (box or infinite).
    Returns (record dict, dict irrep -> eigenvalues)."""
    def say(msg):
        if log is not None:
            log(msg)
    names = list(IRREP_NAMES) if names is None else list(names)
    t0 = time.time()
    shape = (2 * sym.M + 1,) * 3
    if np.asarray(gtab).shape != shape or np.asarray(mtab).shape != shape:
        raise ValueError('table shape %s does not match the displacement range M = %d' % (np.asarray(gtab).shape, sym.M))
    classes = [int(c) for c in np.unique(sym.rep_class)]
    qtabs = signed_q_tables(gtab, mtab, sym.M, mu, classes, drop_mass=drop_mass)
    groups = plan_full_groups(sym, names, mem_limit_gb * 1e9)
    results, values = {}, {}
    for grp in groups:
        ta = time.time()
        K, _order = assemble_full_blocks(sym, grp, qtabs, chunk_elems=chunk_elems, log=log)
        t_asm = time.time() - ta
        for nm in grp:
            dat = sym.irrep_data(nm)
            tb = time.time()
            S_l, ev, info = full_block_entropy(K.pop(nm), method=method)
            contrib = dat['d'] * S_l
            results[nm] = dict(irrep=nm, d=int(dat['d']), m=int(dat['m_e'] + dat['m_o']), m_e=int(dat['m_e']),
                               m_o=int(dat['m_o']), block_entropy=float(S_l), scalar_contribution=float(contrib),
                               field_contribution=float(4.0 * contrib), n_eig=int(len(ev)), ev_min=info['ev_min'],
                               ev_max=info['ev_max'], block_asymmetry=info['asym'], assembly_group_s=t_asm,
                               eigen_s=time.time() - tb)
            if keep_values:
                values[nm] = ev
            say('  %-4s d=%d block %d  4*d*S_block = %.9f  (eigen %.1f s)' % (nm, dat['d'], results[nm]['m'],
                                                                              4.0 * contrib, results[nm]['eigen_s']))
    S_scalar = float(sum(r['scalar_contribution'] for r in results.values()))
    full = set(names) == set(IRREP_NAMES)
    dim_ok = (sum(r['d'] * r['m'] for r in results.values()) == sym.n) if full else None
    if full and not dim_ok:
        raise ArithmeticError('sum of d_lambda m_lambda is not the site count')
    out = dict(S=4.0 * S_scalar, S_scalar=S_scalar, mu=float(mu), method=method, irreps=[results[nm] for nm in names],
               irrep_groups=groups, dimension_check=dim_ok, drop_mass=bool(drop_mass), seconds=time.time() - t0,
               largest_block=int(max(r['m'] for r in results.values())))
    return out, values


def massless_entropy_from_gtab(sym, gtab, method='gram', names=None, chunk_elems=1.6e7, mem_limit_gb=16.0,
                               keep_values=True, log=None):
    """j4lib's sublattice-SVD entropy of the ball `sym` from any one-odd kinetic table (box or infinite)."""
    names = list(IRREP_NAMES) if names is None else list(names)
    t0 = time.time()
    if np.asarray(gtab).shape != (2 * sym.M + 1,) * 3:
        raise ValueError('table shape does not match the displacement range')
    classes = [int(c) for c in np.unique(sym.rep_class[sym.even])]
    ytabs = jl.signed_tables(gtab, sym.M, classes)
    groups = jl.plan_irrep_groups(sym, names, method, mem_limit_gb * 1e9)
    results, values = {}, {}
    for grp in groups:
        ta = time.time()
        K = sym.assemble_blocks(grp, ytabs, chunk_elems=chunk_elems, log=log)
        t_asm = time.time() - ta
        for nm in grp:
            dat = sym.irrep_data(nm)
            tb = time.time()
            e_pair, sv = jl.block_pair_entropy(K.pop(nm), method)
            contrib = dat['d'] * (e_pair + abs(dat['m_e'] - dat['m_o']) * LN2)
            results[nm] = dict(irrep=nm, d=int(dat['d']), m_e=int(dat['m_e']), m_o=int(dat['m_o']), pair_entropy=float(e_pair),
                               scalar_contribution=float(contrib), field_contribution=float(4.0 * contrib),
                               n_singular=int(len(sv)), assembly_group_s=t_asm, eigen_s=time.time() - tb)
            if keep_values:
                values[nm] = np.asarray(sv, dtype=float)
    S_scalar = float(sum(r['scalar_contribution'] for r in results.values()))
    full = set(names) == set(IRREP_NAMES)
    dim_ok = (sum(r['d'] * r['m_e'] for r in results.values()) == sym.n_e and
              sum(r['d'] * r['m_o'] for r in results.values()) == sym.n_o) if full else None
    if full and not dim_ok:
        raise ArithmeticError('sum of d_lambda m_lambda is not the sublattice site count')
    out = dict(S=4.0 * S_scalar, S_scalar=S_scalar, mu=0.0, method=method, irreps=[results[nm] for nm in names],
               irrep_groups=groups, dimension_check=dim_ok, seconds=time.time() - t0)
    return out, values


def setup_symmetry(n2max):
    """(meta, sites, BallSymmetry) of the canonical digital ball {|x|^2 <= n2max} (meta from j4lib_v2.ball_radius_data)."""
    G = jl.cubic_group()
    phi = jl.scalar_phase_table(G)
    D = jl.irrep_matrices(G)
    meta, sites = jl2.ball_radius_data(n2max)
    sym = jl.BallSymmetry(sites, G, phi, D)
    return meta, sites, sym


# ----------------------------------------------------------------------------------------------
# dense routines for the tests
# ----------------------------------------------------------------------------------------------
def dense_scalar_covariance(sites, ker, mu, drop_mass=False):
    """The real symmetric scalar covariance C(x,z) = delta/2 + (eps_x/2)[sum_i eta_i(x) G_i(x-z) - mu Gm(x-z)] on `sites`
    from any kernel object with g_component(i, r) and gm(r)."""
    sites = np.asarray(sites, dtype=np.int64)
    n = len(sites)
    r = sites[:, None, :] - sites[None, :, :]
    eta = jl.stag_sign_table()
    cls = jl.parity_class(sites)
    Q = np.zeros((n, n))
    for i in range(3):
        Q += eta[cls, i][:, None] * ker.g_component(i, r)
    if float(mu) != 0.0 and not drop_mass:
        Q -= float(mu) * ker.gm(r)
    eps = 1.0 - 2.0 * (sites.sum(axis=1) % 2)
    C = 0.5 * eps[:, None] * Q
    C[np.arange(n), np.arange(n)] += 0.5
    return C


def dense_scalar_entropy(sites, ker, mu, drop_mass=False):
    """4 Tr s(C) of the dense scalar covariance; returns (S, eigenvalues, symmetry defect)."""
    C = dense_scalar_covariance(sites, ker, mu, drop_mass=drop_mass)
    sym_def = float(np.abs(C - C.T).max())
    ev = np.linalg.eigvalsh(0.5 * (C + C.T))
    return 4.0 * float(np.sum(jl.binary_entropy(ev))), ev, sym_def


# ----------------------------------------------------------------------------------------------
# the box-kernel cache, keyed by (L, mu, M_cache)
# ----------------------------------------------------------------------------------------------
def cache_M(L, M_need, M_default=64):
    return max(int(M_need), min(int(L) // 2 - 1, int(M_default)))


def box_cache_path(cache_dir, L, mu, Mc):
    return os.path.join(cache_dir, 'j4m_box_L%d_mu%s_M%d.npz' % (int(L), mu_tag(mu), int(Mc)))


def slice_table(tab, Mc, M):
    s = slice(Mc - M, Mc + M + 1)
    return np.ascontiguousarray(tab[s, s, s])


def box_tables_cached(L, mu, M_need, cache_dir=None, log=None):
    """(gtab, mtab, diag, source) on [-M_need, M_need]^3 for the box (L, mu); the file holds the tables on [-Mc, Mc]^3 with
    Mc = cache_M(L, M_need) and is keyed by (L, mu, Mc), all verified on reading."""
    L, mu, M_need = int(L), float(mu), int(M_need)
    Mc = cache_M(L, M_need)
    if Mc >= L:
        raise ValueError('displacement range exceeds the box')
    path = box_cache_path(cache_dir, L, mu, Mc) if cache_dir else None
    if path and os.path.exists(path):
        try:
            with np.load(path) as z:
                ok = (int(z['L']) == L and float(z['mu']) == mu and int(z['M']) == Mc and
                      z['gtab'].shape == (2 * Mc + 1,) * 3 and z['mtab'].shape == (2 * Mc + 1,) * 3)
                if ok:
                    diag = dict(L=L, mu=mu, M=Mc, junk=float(z['junk']), real_part_max=float(z['real_part_max']),
                                imag_part_max=float(z['imag_part_max']))
                    return slice_table(z['gtab'], Mc, M_need), slice_table(z['mtab'], Mc, M_need), diag, 'cache'
            if log:
                log('box kernel cache %s rejected (key mismatch); recomputing' % path)
        except Exception as e:
            if log:
                log('box kernel cache %s unreadable (%s); recomputing' % (path, e))
    t0 = time.time()
    ker = MassiveBoxKernel(L, mu)
    gfull, mfull, diag = ker.tables(Mc)
    del ker
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix='.npz')
        os.close(fd)
        np.savez(tmp, gtab=gfull, mtab=mfull, L=L, mu=mu, M=Mc, junk=diag['junk'],
                 real_part_max=diag['real_part_max'], imag_part_max=diag['imag_part_max'])
        os.replace(tmp, path)
    if log:
        log('box kernel L %d mu %s M %d computed in %.1f s%s' % (L, mu_tag(mu), Mc, time.time() - t0, ' and cached' if path else ''))
    return slice_table(gfull, Mc, M_need), slice_table(mfull, Mc, M_need), diag, 'fft'
