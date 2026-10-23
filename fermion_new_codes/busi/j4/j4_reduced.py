#!/usr/bin/env python3
"""
j4_reduced.py -- the scalar staggered, O_h-reduced, sublattice-SVD entropy of the site-centred ball
|x| < R_eff in the antiperiodic massless box of side L (spec busi/specs/J4_spec_v1.md).

Usage: j4_reduced.py --R 12 --offset-sixteenths -7 --L 48 [--method gram|svd|cupy] [--mem-limit-gb 16]
                     [--check-dense] [--check-dense4] [--out file.json] [--threads 1] [--quiet]
Prints one summary line and writes the JSON record (inputs, site counts and hash, multiplicities,
per-irrep contributions, S_scalar, S, timings, peak memory).
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
import j4lib as jl            # noqa: E402


def main():
    ap = argparse.ArgumentParser(description='reduced ball-cut entropy of the flat massless sea')
    ap.add_argument('--R', type=float, default=None, help='nominal radius; R_eff = R + offset/16')
    ap.add_argument('--offset-sixteenths', type=int, default=0)
    ap.add_argument('--Reff', type=float, default=None)
    ap.add_argument('--n2max', type=int, default=None, help='use the ball |x|^2 <= n2max directly')
    ap.add_argument('--L', type=int, required=True)
    ap.add_argument('--method', choices=['gram', 'svd', 'cupy', 'torch'], default='gram')
    ap.add_argument('--mem-limit-gb', type=float, default=16.0, help='block memory per assembly pass')
    ap.add_argument('--chunk-elems', type=float, default=1.6e7)
    ap.add_argument('--check-dense', action='store_true', help='dense scalar eigen-entropy and sublattice formula (n <= 8000)')
    ap.add_argument('--check-dense4', action='store_true', help='dense 4-component control from j4_dense.py (n <= 2500)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    R_eff = args.Reff if args.Reff is not None else (None if args.R is None else args.R + args.offset_sixteenths / 16.0)
    if R_eff is None and args.n2max is None:
        ap.error('give --R (with --offset-sixteenths), --Reff or --n2max')
    log = None if args.quiet else (lambda s: print(s, flush=True))
    t0 = time.time()
    rec = jl.reduced_ball_entropy(R_eff=R_eff, n2max=args.n2max, L=args.L, method=args.method,
                                  mem_limit_gb=args.mem_limit_gb, chunk_elems=args.chunk_elems, log=log)
    rec.update(R_nominal=args.R, offset_sixteenths=args.offset_sixteenths, threads=args.threads,
               numpy=np.__version__, scipy=scipy.__version__,
               peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
    checks = {}
    if args.check_dense:
        if rec['n'] > 8000:
            checks['dense_scalar'] = 'skipped: n > 8000'
        else:
            sites = jl.ball_sites(n2max=rec['n2max'])
            ker = jl.MasslessKernel(args.L)
            d = jl.dense_scalar_entropy(sites, ker)
            checks['dense_scalar'] = dict(S_scalar_eigen=d['S_scalar_eigen'], S_scalar_sublattice=d['S_scalar_sublattice'],
                                          four_times_eigen=4.0 * d['S_scalar_eigen'],
                                          reduced_minus_4eigen=rec['S'] - 4.0 * d['S_scalar_eigen'],
                                          sublattice_minus_eigen=d['S_scalar_sublattice'] - d['S_scalar_eigen'],
                                          antisym=d['antisym'], same_parity_block_max=d['same_parity_block_max'])
    if args.check_dense4:
        if rec['n'] > 2500:
            checks['dense_4component'] = 'skipped: n > 2500'
        else:
            import j4_dense as jd
            d4 = jd.dense_entropy(n2max=rec['n2max'], L=args.L, m=0.0)
            d4['R_eff'] = rec['R_eff']
            checks['dense_4component'] = dict(S=d4['S'], reduced_minus_dense4=rec['S'] - d4['S'],
                                              hermiticity=d4['hermiticity'], trace_minus_2n=d4['trace_minus_2n'])
    rec['checks'] = checks
    rec['wall_s'] = time.time() - t0
    print('reduced: R_eff %.6f (n2max %d) L %d sites %d (e %d, o %d) largest block %s  S = %.12f  S/(4 pi R^2) = %.8f  '
          '[%s, %.0f s, peak %.2f GB]' % (rec['R_eff'], rec['n2max'], rec['L'], rec['n'], rec['n_e'], rec['n_o'],
                                          rec['largest_block'], rec['S'], rec['S_over_4piR2'], rec['method'],
                                          rec['wall_s'], rec['peak_rss_gb']), flush=True)
    for key, val in checks.items():
        print('  check %s: %s' % (key, json.dumps(val) if isinstance(val, dict) else val), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        tmp = args.out + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(rec, fh, indent=1)
        os.replace(tmp, args.out)


if __name__ == '__main__':
    main()
