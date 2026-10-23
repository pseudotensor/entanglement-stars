#!/usr/bin/env python3
"""
j4prefix_v3b.py -- exact-prefix assembly of the digital balls of one radius window, diagnostic version (busi lane,
task 1; lead's task B4; busi/notes/J4_v2_B4_v1.md section 4; busi/notes/J4_v2_B4_judgment_v1.md section 2).

The hypothesis: every basis vector of the cubic-irrep reduction is supported on one O_h orbit and the basis of the
largest ball B_N of a window lists its orbits in shell order, so that the block K_lambda(B_n) of a smaller ball is the
leading m_e(n) x m_o(n) submatrix of K_lambda(B_N).  The version-3 test refuted it on the pod: equal shapes, entries
differing by order 1, sorted row norms differing by 0.87.  Either j4lib's orbit order is not the shell order (the
leading rows then belong partly to orbits outside B_n) or its basis vectors are not orbit-local.  This version adds
the diagnostics that separate the two: reps_order() reads whether the orbit representatives of the largest ball are in
non-decreasing |x|^2; PrefixWindow(shell_sort=True) builds BallSymmetry from the sites sorted by |x|^2 (the remedy if
j4lib numbers orbits by first occurrence); compare_blocks() reports which case the entries show.  NOT FOR PRODUCTION
in version 2 (the version-2 driver is used; this route saves CPU-hours, not wall time).
Reused from version 1 unchanged: BallSymmetry, assemble_blocks, signed_tables, block_pair_entropy, sites_hash, and the
version-2 kernel cache.
"""
import time

import numpy as np

import j4lib as jl
import j4lib_v2 as jl2
import j4window_v2 as jw


def _say(log, msg):
    if log is not None:
        log(msg)


def shell_sorted(sites):
    """The sites ordered by |x|^2, then lexicographically."""
    s = np.asarray(sites)
    if s.ndim != 2 or s.shape[1] != 3:
        raise ValueError('sites must be an (n, 3) array; got shape %s' % (s.shape,))
    key = np.lexsort((s[:, 2], s[:, 1], s[:, 0], (s * s).sum(axis=1)))
    return np.ascontiguousarray(s[key])


class PrefixWindow:
    """The 10 blocks of the largest ball of the window (R_c, W) in the box L, and the entropies of its balls from their
    leading submatrices (the hypothesis under test)."""

    def __init__(self, Rc, W, L, cache_dir=None, chunk_elems=1.6e7, log=None, names=None, shell_sort=False):
        self.Rc, self.W, self.L = float(Rc), float(W), int(L)
        self.log = log
        self.names = list(names or jl.IRREP_NAMES)
        self.chunk_elems = chunk_elems
        self.shell_sort = bool(shell_sort)
        self.balls = [int(n) for n, _, _ in jw.balls_in_window(Rc, W)]
        self.N = max(self.balls)
        self.G = jl.cubic_group()
        self.phi = jl.scalar_phase_table(self.G)
        self.D = jl.irrep_matrices(self.G)
        metaN, sitesN = jl2.ball_radius_data(self.N)
        if 2.0 * metaN['R_hi'] >= self.L:
            raise ValueError('the largest ball must fit in the box: need L > 2 R_hi')
        if self.shell_sort:
            sitesN = shell_sorted(sitesN)
        self.sitesN = sitesN
        t0 = time.time()
        self.symN = jl.BallSymmetry(sitesN, self.G, self.phi, self.D)
        if self.symN.M >= self.L:
            raise ValueError('displacement range exceeds the box')
        _say(log, 'prefix window R %.1f W %.1f L %d%s: %d balls, largest %d with %d sites (even %d, odd %d), %d orbits, M %d (%.1f s)'
             % (self.Rc, self.W, self.L, ' (sites shell-sorted)' if self.shell_sort else '', len(self.balls), self.N, self.symN.n, self.symN.n_e, self.symN.n_o,
                len(self.symN.reps), self.symN.M, time.time() - t0))
        gtab, junk, rpm, source = jl2.kernel_table_cached(self.L, self.symN.M, cache_dir=cache_dir, log=log)
        if junk > 1e-13:
            raise ArithmeticError('kernel sublattice structure violated: %g' % junk)
        even = np.asarray(self.symN.even)
        rep_class = np.asarray(self.symN.rep_class)
        classes = [int(c) for c in np.unique(rep_class[even])]
        self.ytabs = jl.signed_tables(gtab, self.symN.M, classes)
        del gtab
        self.kernel = dict(source=source, junk=float(junk), real_part_max=float(rpm))
        self.blocks = {}
        self.sizes = {}
        self.timing = dict(symmetry_s=time.time() - t0, assembly_s={})

    def reps_order(self):
        """(monotone, norms): whether |rep|^2 is non-decreasing along the orbit representatives of the largest ball (the
        shell-order hypothesis read directly), or None if the representation of symN.reps is not recognised."""
        try:
            reps = np.asarray(self.symN.reps)
            sites = np.asarray(self.sitesN)
            if reps.ndim == 2 and reps.shape[1] == 3:
                norms = (reps * reps).sum(axis=1)
            elif reps.ndim == 1 and np.issubdtype(reps.dtype, np.integer) and sites.ndim == 2 and reps.max() < len(sites):
                norms = (sites[reps] * sites[reps]).sum(axis=1)
            else:
                return None
        except Exception:
            return None
        return bool(np.all(np.diff(norms) >= 0)), np.asarray(norms)

    def assemble(self, names=None):
        """Assemble the blocks of the largest ball, one irrep at a time (memory: one block plus the assembly temporaries)."""
        for nm in list(names or self.names):
            if nm in self.blocks:
                continue
            t0 = time.time()
            K = self.symN.assemble_blocks([nm], self.ytabs, chunk_elems=self.chunk_elems, log=None)
            self.blocks[nm] = np.ascontiguousarray(K[nm])
            self.timing['assembly_s'][nm] = time.time() - t0
            _say(self.log, '  assembled %-4s block %d x %d (%.1f GB) in %.0f s' % (nm, self.blocks[nm].shape[0], self.blocks[nm].shape[1], self.blocks[nm].nbytes / 1e9, time.time() - t0))

    def ball_sizes(self, n):
        """Multiplicities per irrep and the sublattice counts of the ball n (the version-1 symmetry object of that ball)."""
        n = int(n)
        if n in self.sizes:
            return self.sizes[n]
        meta, sites = jl2.ball_radius_data(n)
        if meta['n2_ball'] != n:
            raise ValueError('%d is not a canonical ball label' % n)
        sym = jl.BallSymmetry(sites, self.G, self.phi, self.D)
        irr = {nm: sym.irrep_data(nm) for nm in self.names}
        info = dict(meta=meta, n=int(sym.n), n_e=int(sym.n_e), n_o=int(sym.n_o), M=int(sym.M), orbits=int(len(sym.reps)), sites_hash=jl.sites_hash(sites),
                    dims={nm: (int(irr[nm]['m_e']), int(irr[nm]['m_o']), int(irr[nm]['d'])) for nm in self.names})
        self.sizes[n] = info
        return info

    def sub_block(self, nm, n):
        m_e, m_o, _ = self.ball_sizes(n)['dims'][nm]
        K = self.blocks[nm]
        if m_e > K.shape[0] or m_o > K.shape[1]:
            raise ValueError('ball %d irrep %s: %d x %d exceeds the largest block %s' % (n, nm, m_e, m_o, K.shape))
        return np.ascontiguousarray(K[:m_e, :m_o])

    def entropy(self, n, method='gram', keep_values=False):
        """The record of the ball n (version-2 fields, assembly 'prefix')."""
        n = int(n)
        info = self.ball_sizes(n)
        meta = info['meta']
        results, values = {}, {}
        t0 = time.time()
        for nm in self.names:
            m_e, m_o, d = info['dims'][nm]
            tb = time.time()
            e_pair, svals = jl.block_pair_entropy(self.sub_block(nm, n), method)
            contrib = d * (e_pair + abs(m_e - m_o) * jl.LN2)
            results[nm] = dict(irrep=nm, d=d, m_e=m_e, m_o=m_o, pair_entropy=float(e_pair), scalar_contribution=float(contrib),
                               field_contribution=float(4.0 * contrib), n_singular=int(len(svals)), assembly_group_s=0.0, eigen_s=time.time() - tb)
            if keep_values:
                values[nm] = svals
        S_scalar = float(sum(r['scalar_contribution'] for r in results.values()))
        S = 4.0 * S_scalar
        dim_ok = (sum(r['d'] * r['m_e'] for r in results.values()) == info['n_e'] and sum(r['d'] * r['m_o'] for r in results.values()) == info['n_o'])
        if not dim_ok:
            raise ArithmeticError('ball %d: sum of d_lambda m_lambda is not the sublattice site count' % n)
        largest = max(((r['m_e'], r['m_o']) for r in results.values()), key=lambda t: max(t))
        rec = dict(version=3, assembly='prefix', prefix_ball=int(self.N), shell_sort=self.shell_sort, n2max=int(meta['n2_ball']), n2max_requested=n,
                   R_lo=meta['R_lo'], R_hi=meta['R_hi'], R_mid=meta['R_mid'], R_eff=meta['R_mid'], R_vol=meta['R_vol'], window_centre=self.Rc,
                   L=self.L, L_over_Rmid=self.L / meta['R_mid'], n=info['n'], n_e=info['n_e'], n_o=info['n_o'], sites_hash=info['sites_hash'],
                   M_disp=info['M'], M_table=int(self.symN.M), orbits=info['orbits'], kernel_source=self.kernel['source'],
                   kernel_real_part_max=self.kernel['real_part_max'], kernel_structure_junk=self.kernel['junk'],
                   irreps=[results[nm] for nm in self.names], irrep_groups=[[nm] for nm in self.names], method=method,
                   S_scalar=S_scalar, S=float(S), S_over_4piRmid2=float(S / (4.0 * np.pi * meta['R_mid'] ** 2)),
                   S_over_4piRvol2=float(S / (4.0 * np.pi * meta['R_vol'] ** 2)), largest_block=[int(largest[0]), int(largest[1])],
                   dimension_check=bool(dim_ok), timing=dict(total_s=time.time() - t0, assembly_shared_s=sum(self.timing['assembly_s'].values())))
        return (rec, values) if keep_values else rec


def direct_blocks(n, L, cache_dir=None, names=None, chunk_elems=1.6e7, log=None):
    """The version-1/2 direct assembly of the blocks of the ball n in the box L (for the exact test)."""
    G = jl.cubic_group()
    phi = jl.scalar_phase_table(G)
    D = jl.irrep_matrices(G)
    meta, sites = jl2.ball_radius_data(n)
    sym = jl.BallSymmetry(sites, G, phi, D)
    gtab, junk, rpm, src = jl2.kernel_table_cached(L, sym.M, cache_dir=cache_dir, log=log)
    even = np.asarray(sym.even)
    classes = [int(c) for c in np.unique(np.asarray(sym.rep_class)[even])]
    ytabs = jl.signed_tables(gtab, sym.M, classes)
    out = {}
    for nm in list(names or jl.IRREP_NAMES):
        out[nm] = np.ascontiguousarray(sym.assemble_blocks([nm], ytabs, chunk_elems=chunk_elems, log=None)[nm])
    return out, sym


def compare_blocks(pw, n, cache_dir=None, tol=1e-12):
    """Entrywise comparison of the leading submatrices of the window's largest ball with the direct blocks of ball n.
    Reasons: 'permutation' (same rows and columns in another order: the ordering within the selected orbits), 'other
    orbits or other vectors' (the leading submatrix selects orbits outside the ball, or the basis vectors are not
    orbit-local), with the singular-value comparison as the permutation-invariant check."""
    direct, sym = direct_blocks(n, pw.L, cache_dir=cache_dir, names=pw.names)
    report = {}
    for nm in pw.names:
        Kd = direct[nm]
        m_e, m_o, _ = pw.ball_sizes(n)['dims'][nm]
        shape_ok = (Kd.shape == (m_e, m_o))
        if not shape_ok:
            report[nm] = dict(ok=False, reason='shape %s against the version-1 object %s' % (Kd.shape, (m_e, m_o)))
            continue
        Ks = pw.sub_block(nm, n)
        diff = float(np.max(np.abs(Ks - Kd))) if Kd.size else 0.0
        entry = dict(ok=diff <= tol, max_diff=diff, shape=list(Kd.shape))
        if diff > tol and Kd.size:
            rn = float(np.max(np.abs(np.sort(np.linalg.norm(Ks, axis=1)) - np.sort(np.linalg.norm(Kd, axis=1)))))
            cn = float(np.max(np.abs(np.sort(np.linalg.norm(Ks, axis=0)) - np.sort(np.linalg.norm(Kd, axis=0)))))
            try:
                sv = float(np.max(np.abs(np.linalg.svd(Ks, compute_uv=False) - np.linalg.svd(Kd, compute_uv=False))))
            except Exception:
                sv = float('nan')
            entry.update(sorted_row_norm_diff=rn, sorted_col_norm_diff=cn, singular_value_diff=sv,
                         reason=('permutation: the same rows and columns in another order (the entropy is unchanged; the within-orbit ordering differs)' if max(rn, cn) <= 1e-9
                                 else 'other orbits or other vectors: the leading submatrix of the largest ball is not the block of this ball (j4lib orbit order not the shell order, or basis vectors not orbit-local); see reps_order()'))
        report[nm] = entry
    return report
