#!/usr/bin/env python3
"""
j4_prefix_production_v3.py -- production of one radius window by the exact-prefix assembly (busi lane, task 1;
lead's task B4; busi/notes/J4_v2_B4_v1.md section 4).  Optional: use only after j4_prefix_tests_v3.py passed.

One process per window: the 10 blocks of the largest ball are assembled once (CPU, about 20 minutes at R 62, 46 GB
of host memory), then every ball of the window, in the order of the dyadic sampling levels, gets its entropy from the
leading submatrices (the eigenproblems on the GPU with --method torch, under the GPU lock as in version 2).  Records
have the version-2 fields plus assembly = 'prefix' and are written to --outdir with the version-2 file names
(R%d_n%d_L%d.json), so the fit and the version-2 driver see them; cases present in --outdir or --v1-dirs are skipped.

Usage:
  j4_prefix_production_v3.py --R 62 [--W 2] [--L 496] --outdir prod_v2 --v1-dirs prod --cache-dir prod_v2/kcache
                             --method torch [--redo-v1] [--max-level 5] [--dry-run]
"""
import argparse
import json
import os
import resource
import sys
import time


def _early_threads():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--threads', type=int, default=4)
    a, _ = p.parse_known_args()
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[v] = str(a.threads)


_early_threads()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import j4window_v2 as jw                     # noqa: E402
from j4production_scan import scan_done      # noqa: E402,F401


def main():
    ap = argparse.ArgumentParser(description='J4 version-3 prefix production of one window')
    ap.add_argument('--R', type=int, required=True)
    ap.add_argument('--W', type=float, default=None, help='window width (default 3 for R <= wide-upto, 2 above)')
    ap.add_argument('--L', type=int, default=None, help='box side (default 8 R)')
    ap.add_argument('--outdir', default='prod_v2')
    ap.add_argument('--v1-dirs', nargs='*', default=[])
    ap.add_argument('--cache-dir', default=None, help='kernel-table cache (default outdir/kcache)')
    ap.add_argument('--method', default='torch', choices=['gram', 'svd', 'cupy', 'torch'])
    ap.add_argument('--wide-upto', type=int, default=40)
    ap.add_argument('--box-ladder-upto', type=int, default=48)
    ap.add_argument('--redo-v1', action='store_true', help='recompute the version-1 balls of R > box-ladder-upto at this box')
    ap.add_argument('--max-level', type=int, default=None)
    ap.add_argument('--chunk-elems', type=float, default=1.6e7)
    ap.add_argument('--gpu-lock', default=os.environ.get('J4_GPU_LOCK', '<internal>/j4_gpu.lock'))
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    os.environ['J4_GPU_LOCK'] = args.gpu_lock
    import j4prefix_v3 as jp  # noqa: E402  (after the thread and lock environment is set)

    R = args.R
    W = args.W if args.W is not None else (3.0 if R <= args.wide_upto else 2.0)
    L = args.L if args.L is not None else 8 * R
    outdir = os.path.abspath(args.outdir)
    cache_dir = os.path.abspath(args.cache_dir) if args.cache_dir else os.path.join(outdir, 'kcache')
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    log = lambda s: print(s, flush=True)  # noqa: E731

    levels = jw.ball_levels(R, W, 3, 7)
    v1 = set(jw.v1_balls(R))
    done = scan_done(list(args.v1_dirs) + [outdir])
    plan = []
    for n, lev in sorted(levels.items(), key=lambda t: (t[1], t[0])):
        if args.max_level is not None and lev > args.max_level:
            continue
        if n in v1 and R > args.box_ladder_upto and not args.redo_v1:
            continue
        if (n, L) in done:
            continue
        plan.append((n, lev))
    log('prefix production R %d W %.1f L %d: %d balls in the window, %d pending (%d done)' % (R, W, L, len(levels), len(plan), len(levels) - len(plan)))
    if args.dry_run or not plan:
        for n, lev in plan:
            log('  pending R%d_n%d_L%d level %d' % (R, n, L, lev))
        return

    t_start = time.time()
    pw = jp.PrefixWindow(R, W, L, cache_dir=cache_dir, chunk_elems=args.chunk_elems, log=log)
    pw.assemble()
    log('all blocks assembled in %.0f s; host peak %.1f GB' % (time.time() - t_start, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))
    tsv = open(os.path.join(outdir, 'production_log.tsv'), 'a')
    if tsv.tell() == 0:
        tsv.write('case\tR\tn\tL\tlevel\tkind\tstatus\twall_s\tpeak_rss_gb\tS\tn_sites\tlargest_block\tmethod\tkernel\n')
    for i, (n, lev) in enumerate(plan):
        cid = 'R%d_n%d_L%d' % (R, n, L)
        t0 = time.time()
        try:
            rec = pw.entropy(n, method=args.method)
            rec.update(threads=args.threads, wall_s=time.time() - t0, peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, level=int(lev))
            out = os.path.join(outdir, cid + '.json')
            tmp = out + '.tmp'
            with open(tmp, 'w') as fh:
                json.dump(rec, fh, indent=1)
            os.replace(tmp, out)
            status, S = 'OK', rec['S']
            tsv.write('%s\t%d\t%d\t%d\t%d\t%s\t%s\t%.0f\t%.2f\t%.9f\t%d\t%s\t%s\t%s\n'
                      % (cid, R, n, L, lev, 'prefix', status, rec['wall_s'], rec['peak_rss_gb'], S, rec['n'], rec['largest_block'], args.method, rec['kernel_source']))
        except Exception as e:
            status, S = 'FAIL(%s)' % type(e).__name__, float('nan')
            tsv.write('%s\t%d\t%d\t%d\t%d\tprefix\t%s\t%.0f\tnan\tnan\t-1\tNone\t%s\tNone\n' % (cid, R, n, L, lev, status, time.time() - t0, args.method))
            log('  %s: %s' % (cid, e))
        tsv.flush()
        log('[%7.0f s] %-20s %s  level %d  wall %.0f s  S %.9f  (%d of %d)' % (time.time() - t_start, cid, status, lev, time.time() - t0, S, i + 1, len(plan)))
    tsv.close()
    log('window finished (%.0f s)' % (time.time() - t_start))


if __name__ == '__main__':
    main()
