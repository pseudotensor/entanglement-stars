#!/usr/bin/env python3
"""
j4_reduced_v2s.py -- one digital ball {|x|^2 <= n} (canonicalized) in the antiperiodic massless box L,
version-2 command line (spec busi/specs/J4_spec_v2.md).
Version 2s (COMMS rule 10, the owner: every producer saves the raw arrays any later readout needs): identical to
j4_reduced_v2.py, and with --out it also writes <out stem>_svals.npz beside the JSON record: the singular values of
each irrep block K (the per-irrep pair spectra the entropy is built from) with the irrep multiplicity d and the
sublattice dimensions m_e, m_o, so that any other function of the spectrum is a reading of saved data, never a rerun;
the entropy recomputed from the saved arrays is checked against the record before the files are written.

Usage:
  j4_reduced_v2.py --n2max 3844 --L 496 [--window-centre 62] [--method gram|svd|torch|cupy]
                   [--kernel-cache DIR] [--mem-limit-gb 16] [--check-dense4] [--out file.json] [--threads 1]
  j4_reduced_v2.py --Reff 62.0625 --L 496 ...            (the strict ball |x| < Reff)
  j4_reduced_v2.py --kernel-only --L 496 --M 124 --kernel-cache DIR     (pre-warm the kernel cache)
Prints one summary line and writes the JSON record (ball label and interval, site counts and hash,
multiplicities, per-irrep contributions, S, timings, peak memory).
"""
import argparse
import json
import os
import resource
import sys
import time


def _early_threads():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--threads', type=int, default=1)
    a, _ = p.parse_known_args()
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[v] = str(a.threads)


_early_threads()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np            # noqa: E402
import scipy                  # noqa: E402
import j4lib_v2 as jl2        # noqa: E402
import j4window_v2 as jw      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description='reduced ball-cut entropy of the flat massless sea, version 2')
    ap.add_argument('--n2max', type=int, default=None, help='the ball |x|^2 <= n2max (canonicalized)')
    ap.add_argument('--Reff', type=float, default=None, help='alternative: the strict ball |x| < Reff')
    ap.add_argument('--L', type=int, required=True)
    ap.add_argument('--window-centre', type=float, default=None, help='the primary radius of the window (recorded)')
    ap.add_argument('--method', choices=['gram', 'svd', 'cupy', 'torch'], default='gram')
    ap.add_argument('--mem-limit-gb', type=float, default=16.0)
    ap.add_argument('--chunk-elems', type=float, default=1.6e7)
    ap.add_argument('--kernel-cache', default=None)
    ap.add_argument('--kernel-only', action='store_true', help='compute and cache the kernel table for --L and --M, then exit')
    ap.add_argument('--M', type=int, default=None, help='displacement range of the table (kernel-only): 2 floor(sqrt n)')
    ap.add_argument('--check-dense4', action='store_true', help='dense 4-component control from j4_dense.py (n <= 2500)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()
    log = None if args.quiet else (lambda s: print(s, flush=True))

    if args.kernel_only:
        if args.M is None or not args.kernel_cache:
            ap.error('--kernel-only needs --M and --kernel-cache')
        t0 = time.time()
        gtab, junk, rpm, src = jl2.kernel_table_cached(args.L, args.M, args.kernel_cache, log)
        print('kernel-only: L %d M %d source %s junk %.2e real-part %.2e table %s  [%.1f s]'
              % (args.L, args.M, src, junk, rpm, gtab.shape, time.time() - t0), flush=True)
        return

    if args.n2max is None and args.Reff is None:
        ap.error('give --n2max or --Reff (or --kernel-only)')
    n2max = args.n2max if args.n2max is not None else jw.ball_at_radius(args.Reff)
    t0 = time.time()
    rec, svals, _sym = jl2.reduced_ball_entropy_v2(n2max, args.L, method=args.method, mem_limit_gb=args.mem_limit_gb,
                                                   chunk_elems=args.chunk_elems, cache_dir=args.kernel_cache,
                                                   window_centre=args.window_centre, keep_values=True, log=log)
    del _sym
    # rule 10: the entropy rebuilt from the spectra that will be saved must equal the record's
    import j4lib as jl
    S_re = 0.0
    for r in rec['irreps']:
        sv = np.asarray(svals[r['irrep']], dtype=float)
        S_re += r['d'] * (float(np.sum(jl.pair_entropy_from_singular(sv))) + abs(r['m_e'] - r['m_o']) * jl.LN2)
    S_re *= 4.0
    rec['svals_check_S_minus_record'] = float(S_re - rec['S'])
    if abs(S_re - rec['S']) > 1e-9 * max(1.0, abs(rec['S'])):
        raise ArithmeticError('entropy from the saved spectra differs from the record: %r against %r' % (S_re, rec['S']))
    rec.update(threads=args.threads, numpy=np.__version__, scipy=scipy.__version__,
               peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
    checks = {}
    if args.check_dense4:
        if rec['n'] > 2500:
            checks['dense_4component'] = 'skipped: n > 2500'
        else:
            import j4_dense as jd
            d4 = jd.dense_entropy(n2max=rec['n2max'], L=args.L, m=0.0)
            checks['dense_4component'] = dict(S=d4['S'], reduced_minus_dense4=rec['S'] - d4['S'],
                                              hermiticity=d4.get('hermiticity'), trace_minus_2n=d4.get('trace_minus_2n'))
    rec['checks'] = checks
    rec['wall_s'] = time.time() - t0
    print('reduced v2: ball %d  R in (%.5f, %.5f]  R_vol %.5f  L %d  sites %d (e %d, o %d)  largest block %s  '
          'S = %.12f  S/(4 pi R_vol^2) = %.8f  [%s, kernel %s, %.0f s, peak %.2f GB]'
          % (rec['n2max'], rec['R_lo'], rec['R_hi'], rec['R_vol'], rec['L'], rec['n'], rec['n_e'], rec['n_o'],
             rec['largest_block'], rec['S'], rec['S_over_4piRvol2'], rec['method'], rec['kernel_source'],
             rec['wall_s'], rec['peak_rss_gb']), flush=True)
    for key, val in checks.items():
        print('  check %s: %s' % (key, json.dumps(val) if isinstance(val, dict) else val), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        tmp = args.out + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(rec, fh, indent=1)
        os.replace(tmp, args.out)
        stem = args.out[:-5] if args.out.endswith('.json') else args.out
        npz = stem + '_svals.npz'
        arrays = {('svals_' + r['irrep']): np.asarray(svals[r['irrep']], dtype=float) for r in rec['irreps']}
        arrays['irrep_names'] = np.array([r['irrep'] for r in rec['irreps']])
        arrays['irrep_d'] = np.array([r['d'] for r in rec['irreps']], dtype=int)
        arrays['irrep_m_e'] = np.array([r['m_e'] for r in rec['irreps']], dtype=int)
        arrays['irrep_m_o'] = np.array([r['m_o'] for r in rec['irreps']], dtype=int)
        arrays['S'] = np.array(rec['S']); arrays['n2max'] = np.array(rec['n2max']); arrays['L'] = np.array(rec['L'])
        tmpz = stem + '_svals.tmp.npz'
        np.savez_compressed(tmpz, **arrays)
        os.replace(tmpz, npz)


if __name__ == '__main__':
    main()
