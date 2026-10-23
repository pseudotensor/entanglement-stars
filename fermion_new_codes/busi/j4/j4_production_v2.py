#!/usr/bin/env python3
"""
j4_production_v2.py -- resumable driver for J4 version 2 (spec busi/specs/J4_spec_v2.md, section 6).

The plan for each primary radius R:
  * every distinct digital ball whose radius interval meets the window (R - W/2, R + W/2), W = 3 for
    R <= --wide-upto (default 40) and W = 2 above, at the box L = 8 R, in the order of the dyadic sampling
    levels (level 3 = spacing 1/8, which includes the 8 version-1 balls; then 4, 5, 6, 7, the rest);
  * for R <= --box-ladder-upto (default 48): the 8 version-1 balls also at 4 R and 6 R (the finite-box ladder;
    most of these exist from version 1 and are skipped);
  * the version-1 balls of R > --box-ladder-upto keep their 4 R value and are not recomputed at 8 R unless --redo-v1.
Cases already present (same canonical ball and box) in --v1-dirs or --outdir are skipped.  Before the
workers start, the kernel table of every (L, M) in the plan is computed once into --cache-dir
(j4_reduced_v2.py --kernel-only), so that the workers share one FFT per box.  Each case runs
j4_reduced_v2.py in its own single-thread process; wall time and peak memory go to outdir/production_log.tsv.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import j4window_v2 as jw  # noqa: E402


def scan_done(dirs):
    """(canonical ball, L) -> path of an existing record with an entropy."""
    done = {}
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith('.json'):
                continue
            p = os.path.join(d, f)
            try:
                with open(p) as fh:
                    r = json.load(fh)
            except Exception:
                continue
            if all(k in r for k in ('S', 'n2max', 'L')) and r['S'] == r['S']:
                done[(jw.canonical_ball(int(r['n2max'])), int(r['L']))] = p
    return done


def estimate(n, L, method, mem_limit_gb, prewarmed=True):
    """Rough peak memory (bytes) and time (hours) of a case (the version-1 model, kernel from the cache)."""
    Reff = math.sqrt(n)
    nsites = 4.18879 * Reff ** 3 + 6.2832 * Reff ** 2
    mmax = 0.033 * nsites
    Kmax = 8.0 * mmax ** 2
    gram = Kmax if method in ('gram', 'cupy', 'torch') else 0.0
    total = 8.0 * nsites ** 2 / 192.0
    blocks = max(Kmax + gram, min(mem_limit_gb * 1e9, total + gram))
    fft = 0.0 if prewarmed else 2.5 * 16.0 * L ** 3
    mem = max(fft, blocks) + 2.0e9
    base = {'gram': 8.0, 'svd': 24.0, 'torch': 0.45, 'cupy': 0.45}[method]
    hours = base * (Reff / 62.0) ** 9 + 0.01
    return mem, hours


def build_plan(args):
    cases, seen = [], set()

    def add(n, L, R, level, kind):
        key = (n, L)
        if key in seen:
            return
        seen.add(key)
        method = args.method if args.method != 'auto' else ('torch' if R >= args.torch_from else 'gram')
        mem, hours = estimate(n, L, method, args.mem_limit_gb, prewarmed=not args.no_prewarm)
        cases.append(dict(id='R%d_n%d_L%d' % (R, n, L), n=int(n), L=int(L), R=int(R), level=int(level), kind=kind,
                          M=2 * int(math.floor(math.sqrt(n))), method=method, mem=mem, hours=hours))

    for R in args.radii:
        W = 3.0 if R <= args.wide_upto else 2.0
        L8 = 8 * R
        levels = jw.ball_levels(R, W, 3, 7)
        v1 = set(jw.v1_balls(R))
        for n, lev in sorted(levels.items(), key=lambda t: (t[1], t[0])):
            if args.max_level is not None and lev > args.max_level:
                continue
            if n in v1 and R > args.box_ladder_upto and not args.redo_v1:
                continue
            add(n, L8, R, lev, 'window')
        if R <= args.box_ladder_upto:
            for n in sorted(v1):
                for L in (4 * R, 6 * R, 8 * R):
                    add(n, L, R, 0, 'ladder')
    return cases


def main():
    ap = argparse.ArgumentParser(description='J4 version-2 production driver')
    ap.add_argument('--radii', type=int, nargs='+', required=True)
    ap.add_argument('--outdir', default='prod_v2')
    ap.add_argument('--v1-dirs', nargs='*', default=[], help='directories with version-1 (or other) records to skip')
    ap.add_argument('--cache-dir', default=None, help='kernel-table cache (default outdir/kcache)')
    ap.add_argument('--wide-upto', type=int, default=40, help='W = 3 for R <= this, W = 2 above')
    ap.add_argument('--box-ladder-upto', type=int, default=48)
    ap.add_argument('--max-level', type=int, default=None, help='only balls of dyadic level <= this (3..7; default all)')
    ap.add_argument('--redo-v1', action='store_true', help='recompute the version-1 balls of R > box-ladder-upto at 8R')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--threads-per-worker', type=int, default=1)
    ap.add_argument('--mem-budget-gb', type=float, default=230.0)
    ap.add_argument('--mem-limit-gb', type=float, default=16.0, help='block memory per assembly pass (passed on)')
    ap.add_argument('--method', default='auto', choices=['auto', 'gram', 'svd', 'cupy', 'torch'])
    ap.add_argument('--torch-from', type=int, default=40, help='with --method auto: torch for R >= this')
    ap.add_argument('--gpu-lock', default=os.environ.get('J4_GPU_LOCK', '<internal>/j4_gpu.lock'))
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--nice', type=int, default=0)
    ap.add_argument('--poll', type=float, default=10.0)
    ap.add_argument('--no-prewarm', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    code_dir = os.path.dirname(os.path.abspath(__file__))
    reduced = os.path.join(code_dir, 'j4_reduced_v2.py')
    outdir = os.path.abspath(args.outdir)
    logdir = os.path.join(outdir, 'logs')
    cache_dir = os.path.abspath(args.cache_dir) if args.cache_dir else os.path.join(outdir, 'kcache')
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    cases = build_plan(args)
    done = scan_done(list(args.v1_dirs) + [outdir])
    for c in cases:
        c['out'] = os.path.join(outdir, c['id'] + '.json')
        c['done'] = (c['n'], c['L']) in done
    pending = [c for c in cases if not c['done']]
    pending.sort(key=lambda c: (c['R'], c['level'], c['n'], c['L']))
    print('plan: %d cases, %d pending, %.1f estimated hours of single-case time pending, largest estimated memory %.1f GB'
          % (len(cases), len(pending), sum(c['hours'] for c in pending), max([c['mem'] for c in pending] + [0.0]) / 1e9), flush=True)
    by_R = {}
    for c in cases:
        by_R.setdefault(c['R'], [0, 0])
        by_R[c['R']][0] += 1
        by_R[c['R']][1] += 0 if c['done'] else 1
    for R in sorted(by_R):
        print('  R %2d: %3d cases, %3d pending' % (R, by_R[R][0], by_R[R][1]), flush=True)
    if args.dry_run:
        for c in pending:
            print('  %-20s level %d %-6s %-5s est %.2f h  %.1f GB' % (c['id'], c['level'], c['kind'], c['method'], c['hours'], c['mem'] / 1e9))
        return

    if not args.no_prewarm:
        pairs = sorted({(c['L'], c['M']) for c in pending})
        for L, M in pairs:
            from j4lib_v2 import kernel_cache_path
            if os.path.exists(kernel_cache_path(cache_dir, L, M)):
                continue
            cmd = [args.python, reduced, '--kernel-only', '--L', str(L), '--M', str(M), '--kernel-cache', cache_dir,
                   '--threads', str(max(1, args.threads_per_worker))]
            print('prewarm: ' + ' '.join(cmd), flush=True)
            subprocess.run(cmd, check=True)

    log = open(os.path.join(outdir, 'production_log.tsv'), 'a')
    if log.tell() == 0:
        log.write('case\tR\tn\tL\tlevel\tkind\tstatus\twall_s\tpeak_rss_gb\tS\tn_sites\tlargest_block\tmethod\tkernel\n')
        log.flush()
    running = {}
    env = dict(os.environ)
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[v] = str(args.threads_per_worker)
    env['J4_GPU_LOCK'] = args.gpu_lock
    preexec = (lambda: os.nice(args.nice)) if args.nice else None
    t_start = time.time()
    while pending or running:
        for cid in list(running):
            proc, t0, c, lf = running[cid]
            if proc.poll() is None:
                continue
            lf.close()
            wall = time.time() - t0
            rec = None
            if os.path.exists(c['out']):
                try:
                    with open(c['out']) as fh:
                        rec = json.load(fh)
                except Exception:
                    rec = None
            if rec is not None and 'S' in rec and rec['S'] == rec['S']:
                status, S, ns, blk, pk, ker = 'OK', rec['S'], rec['n'], rec.get('largest_block'), rec.get('peak_rss_gb', float('nan')), rec.get('kernel_source')
            else:
                status, S, ns, blk, pk, ker = 'FAIL(rc=%s)' % proc.returncode, float('nan'), -1, None, float('nan'), None
            log.write('%s\t%d\t%d\t%d\t%d\t%s\t%s\t%.0f\t%.2f\t%.9f\t%d\t%s\t%s\t%s\n'
                      % (cid, c['R'], c['n'], c['L'], c['level'], c['kind'], status, wall, pk, S, ns, blk, c['method'], ker))
            log.flush()
            print('[%7.0f s] %-20s %s  wall %.0f s  peak %.2f GB  S %.9f' % (time.time() - t_start, cid, status, wall, pk, S), flush=True)
            del running[cid]
        used = sum(c['mem'] for (_, _, c, _) in running.values())
        i = 0
        while i < len(pending) and len(running) < args.workers:
            c = pending[i]
            if used + c['mem'] <= args.mem_budget_gb * 1e9 or not running:
                if used + c['mem'] > args.mem_budget_gb * 1e9:
                    print('warning: %s alone exceeds the memory budget (est %.1f GB); running it alone' % (c['id'], c['mem'] / 1e9), flush=True)
                cmd = [args.python, reduced, '--n2max', str(c['n']), '--L', str(c['L']), '--window-centre', str(c['R']),
                       '--method', c['method'], '--mem-limit-gb', str(args.mem_limit_gb), '--kernel-cache', cache_dir,
                       '--threads', str(args.threads_per_worker), '--out', c['out']]
                lf = open(os.path.join(logdir, c['id'] + '.log'), 'a')
                lf.write('# ' + ' '.join(cmd) + '\n')
                lf.flush()
                proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, preexec_fn=preexec)
                running[c['id']] = (proc, time.time(), c, lf)
                used += c['mem']
                print('[%7.0f s] launched %-20s level %d (%s, est %.2f h, %.1f GB; running %d, pending %d)'
                      % (time.time() - t_start, c['id'], c['level'], c['method'], c['hours'], c['mem'] / 1e9, len(running), len(pending) - 1), flush=True)
                pending.pop(i)
            else:
                i += 1
        time.sleep(args.poll)
    log.close()
    print('all cases finished (%.0f s)' % (time.time() - t_start), flush=True)


if __name__ == '__main__':
    main()
