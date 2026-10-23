#!/usr/bin/env python3
"""
j4m_kernel_v1.py -- the box-free flat-sea kernel (busi lane, task 4; spec section 4): P_inf = (1/2)(I - h_mu K),
K = |h_mu|^{-1} = (mu^2 + sum_i sin^2 k_i)^{-1/2} on the infinite lattice, by the Schwinger-Bessel representation

    G_mu(2s) = pi^{-1/2} int_0^inf t^{-1/2} e^{-mu^2 t} prod_i ive(|s_i|, t/2) dt,   ive(n, x) = e^{-x} I_n(x),
    G_mu(r) = 0 unless every component of r is even;   G_i(r) = (G_mu(r - e_i) - G_mu(r + e_i))/2.

Quadrature (spec 4.2): t = u^2, Gauss-Legendre with `nodes_per_panel` nodes on the octave panels [0,1],[1,2],...,
[2^(o-1), 2^o], and the analytic tail beyond T = 4^o from the Hankel asymptotic series,
    tail = pi^{-2} sum_{K <= kmax} C_K T^{-1-K} E_{2+K}(mu^2 T),
C_K the convolution of b_k(n) = (-1)^k prod_{j<=k} (4n^2 - (2j-1)^2)/(k! 4^k) over the three components,
E_n the generalized exponential integral (E_n(0) = 1/(n-1)).  The alternate quadrature (twice the nodes, one more
octave, kmax + 2) is computed with every cached table and the maximum difference recorded.
"""
import os
import sys
import tempfile
import time

import numpy as np
from scipy import special

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import j4m_lib_v1 as jm   # noqa: E402


class BesselKernel:
    def __init__(self, mu, nodes_per_panel=48, octaves=11, kmax_tail=6):
        self.mu = float(mu)
        self.nodes = int(nodes_per_panel)
        self.octaves = int(octaves)
        self.kmax = int(kmax_tail)
        x, w = np.polynomial.legendre.leggauss(self.nodes)
        edges = [0.0] + [2.0 ** j for j in range(self.octaves + 1)]
        us, ws = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            us.append(0.5 * (b - a) * x + 0.5 * (a + b))
            ws.append(0.5 * (b - a) * w)
        self.u = np.concatenate(us)
        self.wu = np.concatenate(ws)
        self.T = float(edges[-1]) ** 2
        self._grid = {}

    def gm_grid(self, smax):
        """G_mu(2s) for all s in [0, smax]^3 (array indexed by |s_1|, |s_2|, |s_3|)."""
        smax = int(smax)
        if smax in self._grid:
            return self._grid[smax]
        n = np.arange(smax + 1)
        xq = 0.5 * self.u ** 2
        iv = special.ive(n[:, None], xq[None, :])                             # (smax+1, Q)
        wq = self.wu * (2.0 / np.sqrt(np.pi)) * np.exp(-self.mu ** 2 * self.u ** 2)
        ivw = iv * wq[None, :]
        pab = iv[:, None, :] * iv[None, :, :]                                 # (smax+1, smax+1, Q)
        direct = np.tensordot(pab, ivw, axes=([2], [1]))                      # (a, b, c)
        bk = np.zeros((self.kmax + 1, smax + 1))
        bk[0] = 1.0
        for k in range(1, self.kmax + 1):
            bk[k] = -bk[k - 1] * (4.0 * n * n - (2 * k - 1) ** 2) / (4.0 * k)
        En = np.array([special.expn(2 + K, self.mu ** 2 * self.T) for K in range(self.kmax + 1)])
        tail = np.zeros_like(direct)
        for K in range(self.kmax + 1):
            cK = np.zeros_like(direct)
            for k1 in range(K + 1):
                for k2 in range(K + 1 - k1):
                    k3 = K - k1 - k2
                    cK += bk[k1][:, None, None] * bk[k2][None, :, None] * bk[k3][None, None, :]
            tail += cK * (self.T ** (-1.0 - K) * En[K] / np.pi ** 2)
        g = direct + tail
        self._grid[smax] = g
        return g

    def gm(self, r):
        """G_mu at integer displacements r (..., 3): zero unless all components are even."""
        r = np.asarray(r, dtype=np.int64)
        s = np.abs(r) // 2
        g = self.gm_grid(int(s.max()) + 1 if s.size else 1)
        val = g[s[..., 0], s[..., 1], s[..., 2]]
        return np.where(np.all(r % 2 == 0, axis=-1), val, 0.0)

    def tables(self, M):
        """(gtab, mtab) on [-M, M]^3: the one-odd kinetic table G_i(r) and the all-even mass table G_mu(r)."""
        M = int(M)
        T = 2 * M + 1
        smax = (M + 2) // 2 + 1
        g = self.gm_grid(smax)
        ax = np.arange(-M, M + 1, dtype=np.int64)
        R = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), axis=-1)
        odd = R % 2
        nodd = odd.sum(axis=-1)

        def lookup(V):
            s = np.abs(V) // 2
            return g[s[..., 0], s[..., 1], s[..., 2]]

        mtab = np.zeros((T, T, T))
        alleven = nodd == 0
        mtab[alleven] = lookup(R[alleven])
        gtab = np.zeros((T, T, T))
        for i in range(3):
            mask = (nodd == 1) & (odd[..., i] == 1)
            e = np.zeros(3, dtype=np.int64)
            e[i] = 1
            Rm = R[mask]
            gtab[mask] = 0.5 * (lookup(Rm - e) - lookup(Rm + e))
        return gtab, mtab


# ----------------------------------------------------------------------------------------------
# the cache, keyed by (mu, M_cache, nodes, octaves); the alternate quadrature inside the same file
# ----------------------------------------------------------------------------------------------
def inf_cache_path(cache_dir, mu, Mc, nodes, octaves):
    return os.path.join(cache_dir, 'j4m_inf_mu%s_M%d_q%d_o%d.npz' % (jm.mu_tag(mu), int(Mc), int(nodes), int(octaves)))


def inf_tables_cached(mu, M_need, cache_dir=None, nodes=48, octaves=11, kmax=6, log=None):
    """(gtab, mtab, (gtab_alt, mtab_alt), info, source) on [-M_need, M_need]^3 for the infinite lattice at mass mu."""
    mu, M_need = float(mu), int(M_need)
    Mc = max(M_need, 64)
    path = inf_cache_path(cache_dir, mu, Mc, nodes, octaves) if cache_dir else None
    shape = (2 * Mc + 1,) * 3
    if path and os.path.exists(path):
        try:
            with np.load(path) as z:
                ok = (float(z['mu']) == mu and int(z['M']) == Mc and int(z['nodes']) == nodes and int(z['octaves']) == octaves
                      and z['gtab'].shape == shape and z['mtab'].shape == shape and z['gtab_alt'].shape == shape)
                if ok:
                    info = dict(mu=mu, M=Mc, nodes=nodes, octaves=octaves, kmax=int(z['kmax']), T=float(z['T']),
                                alt_nodes=int(z['alt_nodes']), alt_octaves=int(z['alt_octaves']), alt_kmax=int(z['alt_kmax']),
                                alt_max_diff=float(z['alt_max_diff']))
                    alt = (jm.slice_table(z['gtab_alt'], Mc, M_need), jm.slice_table(z['mtab_alt'], Mc, M_need))
                    return jm.slice_table(z['gtab'], Mc, M_need), jm.slice_table(z['mtab'], Mc, M_need), alt, info, 'cache'
            if log:
                log('infinite kernel cache %s rejected (key mismatch); recomputing' % path)
        except Exception as e:
            if log:
                log('infinite kernel cache %s unreadable (%s); recomputing' % (path, e))
    t0 = time.time()
    ker = BesselKernel(mu, nodes, octaves, kmax)
    gtab, mtab = ker.tables(Mc)
    kera = BesselKernel(mu, 2 * nodes, octaves + 1, kmax + 2)
    gtab_a, mtab_a = kera.tables(Mc)
    alt_diff = float(max(np.abs(gtab - gtab_a).max(), np.abs(mtab - mtab_a).max()))
    info = dict(mu=mu, M=Mc, nodes=nodes, octaves=octaves, kmax=kmax, T=ker.T, alt_nodes=2 * nodes, alt_octaves=octaves + 1,
                alt_kmax=kmax + 2, alt_max_diff=alt_diff)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix='.npz')
        os.close(fd)
        np.savez(tmp, gtab=gtab, mtab=mtab, gtab_alt=gtab_a, mtab_alt=mtab_a, mu=mu, M=Mc, nodes=nodes, octaves=octaves,
                 kmax=kmax, T=ker.T, alt_nodes=2 * nodes, alt_octaves=octaves + 1, alt_kmax=kmax + 2, alt_max_diff=alt_diff)
        os.replace(tmp, path)
    if log:
        log('infinite kernel mu %s M %d computed in %.1f s (alternate quadrature differs by %.1e)%s'
            % (jm.mu_tag(mu), Mc, time.time() - t0, alt_diff, ' and cached' if path else ''))
    alt = (jm.slice_table(gtab_a, Mc, M_need), jm.slice_table(mtab_a, Mc, M_need))
    return jm.slice_table(gtab, Mc, M_need), jm.slice_table(mtab, Mc, M_need), alt, info, 'bessel'


if __name__ == '__main__':
    for mu in (0.0, 0.2):
        ker = BesselKernel(mu)
        pts = np.array([(0, 0, 0), (2, 0, 0), (2, 2, 0), (2, 2, 2), (4, 0, 0), (12, 4, 0)])
        print('mu %s: %s' % (jm.mu_tag(mu), ', '.join('G%s = %.16f' % (tuple(p), v) for p, v in zip(pts, ker.gm(pts)))))
