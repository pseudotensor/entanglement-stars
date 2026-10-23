#!/usr/bin/env python3
"""
j4lib_v2.py -- version-2 additions to j4lib.py (busi lane, task 1; spec busi/specs/J4_spec_v2.md).
The single-ball arithmetic is j4lib's, unchanged and validated by j4_tests.py (the builder's torch route
included); this module adds
  (1) a disk cache of the staggered kernel table per (L, M), so that the ~200 balls of one radius window
      share one FFT of the 8R box (kernel_table_cached);
  (2) reduced_ball_entropy_v2: the version-1 computation driven by the canonical ball label, with the
      cache and the version-2 record fields (ball label, radius interval, volume radius, window centre).
"""
import os
import tempfile
import time

import numpy as np

import j4lib as jl
from j4lib import *          # noqa: F401,F403  re-export the version-1 routines
import j4window_v2 as jw


def kernel_cache_path(cache_dir, L, M):
    return os.path.join(cache_dir, 'gtab_L%d_M%d.npz' % (int(L), int(M)))


def kernel_table_cached(L, M, cache_dir=None, log=None):
    """(gtab, junk, real_part_max, source): gtab = MasslessKernel(L).staggered_table(M)[0] on [-M, M]^3,
    read from cache_dir if present (source 'cache'), else computed and saved atomically (source 'fft')."""
    L, M = int(L), int(M)
    path = kernel_cache_path(cache_dir, L, M) if cache_dir else None
    if path and os.path.exists(path):
        try:
            with np.load(path) as z:
                gtab = np.ascontiguousarray(z['gtab'])
                junk, rpm = float(z['junk']), float(z['real_part_max'])
                ok = (int(z['L']) == L and int(z['M']) == M and gtab.shape == (2 * M + 1,) * 3)
            if ok:
                return gtab, junk, rpm, 'cache'
            if log:
                log('kernel cache %s has the wrong shape; recomputing' % path)
        except Exception as e:                      # unreadable file: recompute
            if log:
                log('kernel cache %s unreadable (%s); recomputing' % (path, e))
    t0 = time.time()
    ker = jl.MasslessKernel(L)
    gtab, junk = ker.staggered_table(M)
    rpm = ker.real_part_max
    del ker
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix='.npz')
        os.close(fd)
        np.savez(tmp, gtab=gtab, junk=junk, real_part_max=rpm, L=L, M=M)
        os.replace(tmp, path)
    if log:
        log('kernel table L %d M %d computed in %.1f s%s' % (L, M, time.time() - t0, ' and cached' if path else ''))
    return gtab, junk, rpm, 'fft'


def ball_radius_data(n2max):
    """Canonical label, radius interval, site count and volume radius of the ball {|x|^2 <= n2max}; returns (meta, sites)."""
    n_c = jw.canonical_ball(n2max)
    R_lo, R_hi = jw.ball_interval(n_c)
    sites = jl.ball_sites(n2max=n_c)
    n_sites = int(len(sites))
    meta = dict(n2max_requested=int(n2max), n2_ball=int(n_c), R_lo=R_lo, R_hi=R_hi, R_mid=0.5 * (R_lo + R_hi),
                n=n_sites, R_vol=float((3.0 * n_sites / (4.0 * np.pi)) ** (1.0 / 3.0)))
    return meta, sites


def reduced_ball_entropy_v2(n2max, L, method='gram', mem_limit_gb=16.0, chunk_elems=1.6e7, cache_dir=None,
                            window_centre=None, keep_values=False, log=None):
    """The symmetry-reduced, sublattice-SVD entropy of the digital ball {|x|^2 <= n2max} (canonicalized) in the
    antiperiodic massless box of side L; the same arithmetic as j4lib.reduced_ball_entropy, the kernel table
    from the cache.  Returns the record dict (and the singular values per irrep and the BallSymmetry with keep_values)."""
    def say(msg):
        if log is not None:
            log(msg)
    timing = {}
    t0 = time.time()
    L = int(L)
    G = jl.cubic_group()
    phi = jl.scalar_phase_table(G)
    D = jl.irrep_matrices(G)
    meta, sites = ball_radius_data(n2max)
    if 2.0 * meta['R_hi'] >= L:
        raise ValueError('the ball must fit in the box: need L > 2 R_hi')
    sym = jl.BallSymmetry(sites, G, phi, D)
    if sym.M >= L:
        raise ValueError('displacement range exceeds the box')
    timing['symmetry_s'] = time.time() - t0
    say('ball %d (requested %d): sites %d (even %d, odd %d), R in (%.5f, %.5f], R_vol %.5f, orbits %d, M %d, L %d'
        % (meta['n2_ball'], meta['n2max_requested'], sym.n, sym.n_e, sym.n_o, meta['R_lo'], meta['R_hi'],
           meta['R_vol'], len(sym.reps), sym.M, L))
    t1 = time.time()
    gtab, junk, rpm, source = kernel_table_cached(L, sym.M, cache_dir=cache_dir, log=log)
    classes = [int(c) for c in np.unique(sym.rep_class[sym.even])]
    ytabs = jl.signed_tables(gtab, sym.M, classes)
    del gtab
    timing['kernel_s'] = time.time() - t1
    say('kernel (%s): real part max %.2e, structure junk %.2e, %.1f s' % (source, rpm, junk, timing['kernel_s']))
    if junk > 1e-13:
        raise ArithmeticError('kernel sublattice structure violated: %g' % junk)
    names = list(jl.IRREP_NAMES)
    irr = {nm: sym.irrep_data(nm) for nm in names}
    groups = jl.plan_irrep_groups(sym, names, method, mem_limit_gb * 1e9)
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
            e_pair, svals = jl.block_pair_entropy(K.pop(nm), method)
            contrib = dat['d'] * (e_pair + abs(dat['m_e'] - dat['m_o']) * jl.LN2)
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
    dim_ok = (sum(r['d'] * r['m_e'] for r in results.values()) == sym.n_e and
              sum(r['d'] * r['m_o'] for r in results.values()) == sym.n_o)
    if not dim_ok:
        raise ArithmeticError('sum of d_lambda m_lambda is not the sublattice site count')
    largest = max(((r['m_e'], r['m_o']) for r in results.values()), key=lambda t: max(t))
    rec = dict(version=2, n2max=meta['n2_ball'], n2max_requested=meta['n2max_requested'],
               R_lo=meta['R_lo'], R_hi=meta['R_hi'], R_mid=meta['R_mid'], R_eff=meta['R_mid'], R_vol=meta['R_vol'],
               window_centre=(None if window_centre is None else float(window_centre)),
               L=L, L_over_Rmid=L / meta['R_mid'], n=sym.n, n_e=sym.n_e, n_o=sym.n_o,
               sites_hash=jl.sites_hash(sites), M_disp=int(sym.M), orbits=int(len(sym.reps)),
               kernel_source=source, kernel_real_part_max=rpm, kernel_structure_junk=float(junk),
               irreps=[results[nm] for nm in names], irrep_groups=groups, method=method,
               S_scalar=S_scalar, S=float(S),
               S_over_4piRmid2=float(S / (4.0 * np.pi * meta['R_mid'] ** 2)),
               S_over_4piRvol2=float(S / (4.0 * np.pi * meta['R_vol'] ** 2)),
               largest_block=[int(largest[0]), int(largest[1])], dimension_check=bool(dim_ok), timing=timing)
    if keep_values:
        return rec, values, sym
    return rec
