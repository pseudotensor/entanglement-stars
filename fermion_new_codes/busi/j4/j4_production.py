#!/usr/bin/env python3
"""
j4_production.py -- resumable, memory-aware driver for the J4 ladder.  Pure Python (no numpy).
Each case (R, offset k, L) runs j4_reduced.py in its own single-thread process (OMP/OPENBLAS/MKL threads
set from --threads-per-worker), writes outdir/R{R}_o{k}_L{L}.json, and is skipped if that JSON exists
with an entropy.  Wall time and peak memory per case are logged to outdir/production_log.tsv.

Default plan rule (spec section 0 and 5): L = 4R for every radius, 6R in addition for R <= 40, 8R in
addition for R <= 24; offsets k in {-7,-5,-3,-1,1,3,5,7} (R_eff = R + k/16).
Usage examples:
  j4_production.py --radii 12 16 20 24 --workers 8 --outdir prod
  j4_production.py --radii 48 56 62 --workers 8 --mem-budget-gb 300 --outdir prod
  j4_production.py --radii 29 --offsets 1 --boxes 4 --method svd --outdir checks   (a cross-check case)
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

DEFAULT_OFFSETS = [-7, -5, -3, -1, 1, 3, 5, 7]


def default_boxes(R):
    mults = [4]
    if R <= 40:
        mults.append(6)
    if R <= 24:
        mults.append(8)
    return mults


def case_id(R, k, L):
    return 'R%d_o%+d_L%d' % (R, k, L)


def estimate(R, k, L, method, mem_limit_gb):
    """Rough peak memory (bytes) and time (hours, 1 thread) of a case: R^9 scaling from the spec's estimates."""
    Reff = R + k / 16.0
    n = 4.18879 * Reff ** 3 + 6.2832 * Reff ** 2
    mmax = 0.033 * n
    Kmax = 8.0 * mmax ** 2
    gram = Kmax if method in ('gram', 'cupy', 'torch') else 0.0
    total = 8.0 * n * n / 192.0
    blocks = max(Kmax + gram, min(mem_limit_gb * 1e9, total + gram))
    fft = 2.5 * 16.0 * L ** 3
    mem = max(fft, blocks) + 1.5e9
    hours = (8.0 if method == 'gram' else (24.0 if method == 'svd' else 0.1)) * (Reff / 62.0) ** 9 + 0.01
    return mem, hours


def is_done(path):
    if not os.path.exists(path):
        return False
    try:
        with open(path) as fh:
            d = json.load(fh)
        return 'S' in d and d['S'] == d['S']
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description='J4 production driver')
    ap.add_argument('--radii', type=int, nargs='+', required=True)
    ap.add_argument('--offsets', type=int, nargs='+', default=DEFAULT_OFFSETS, help='in sixteenths')
    ap.add_argument('--boxes', type=int, nargs='+', default=None, help='box multipliers for all radii (default: the spec rule)')
    ap.add_argument('--outdir', default='prod')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--threads-per-worker', type=int, default=1)
    ap.add_argument('--mem-budget-gb', type=float, default=250.0, help='sum of estimated peak memories of running cases')
    ap.add_argument('--mem-limit-gb', type=float, default=16.0, help='passed to j4_reduced.py (block memory per pass)')
    ap.add_argument('--method', default='gram', choices=['gram', 'svd', 'cupy', 'torch'])
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--nice', type=int, default=0, help='extra niceness for workers (the driver itself is run under nice)')
    ap.add_argument('--order', choices=['largest-first', 'smallest-first'], default='largest-first')
    ap.add_argument('--poll', type=float, default=10.0)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    code_dir = os.path.dirname(os.path.abspath(__file__))
    reduced = os.path.join(code_dir, 'j4_reduced.py')
    outdir = os.path.abspath(args.outdir)
    logdir = os.path.join(outdir, 'logs')
    os.makedirs(logdir, exist_ok=True)

    cases = []
    for R in args.radii:
        for mult in (args.boxes if args.boxes else default_boxes(R)):
            L = mult * R
            if L % 2:
                L += 1
            for k in args.offsets:
                mem, hours = estimate(R, k, L, args.method, args.mem_limit_gb)
                cid = case_id(R, k, L)
                cases.append(dict(id=cid, R=R, k=k, L=L, mem=mem, hours=hours, out=os.path.join(outdir, cid + '.json')))
    cases.sort(key=lambda c: c['hours'], reverse=(args.order == 'largest-first'))
    pending = [c for c in cases if not is_done(c['out'])]
    print('plan: %d cases, %d pending, %.1f estimated core-hours pending, largest estimated memory %.1f GB'
          % (len(cases), len(pending), sum(c['hours'] for c in pending), max([c['mem'] for c in pending] + [0]) / 1e9), flush=True)
    if args.dry_run:
        for c in cases:
            print('  %-16s R_eff %.4f L %4d  est %.2f h  %.1f GB  %s' % (c['id'], c['R'] + c['k'] / 16.0, c['L'], c['hours'],
                                                                          c['mem'] / 1e9, 'done' if is_done(c['out']) else 'pending'))
        return

    log = open(os.path.join(outdir, 'production_log.tsv'), 'a')
    if log.tell() == 0:
        log.write('case\tR\tk\tL\tstatus\twall_s\tpeak_rss_gb\tS\tn\tlargest_block\tmethod\n')
        log.flush()
    running = {}
    env = dict(os.environ)
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[v] = str(args.threads_per_worker)
    preexec = (lambda: os.nice(args.nice)) if args.nice else None
    t_start = time.time()
    while pending or running:
        for cid in list(running):
            proc, t0, c, lf = running[cid]
            if proc.poll() is None:
                continue
            lf.close()
            wall = time.time() - t0
            if is_done(c['out']):
                with open(c['out']) as fh:
                    rec = json.load(fh)
                status = 'OK'
                S, n, blk, pk = rec['S'], rec['n'], rec.get('largest_block'), rec.get('peak_rss_gb', float('nan'))
            else:
                status, S, n, blk, pk = 'FAIL(rc=%s)' % proc.returncode, float('nan'), -1, None, float('nan')
            log.write('%s\t%d\t%d\t%d\t%s\t%.0f\t%.2f\t%.9f\t%d\t%s\t%s\n' % (cid, c['R'], c['k'], c['L'], status, wall, pk, S, n, blk, args.method))
            log.flush()
            print('[%6.0f s] %-16s %s  wall %.0f s  peak %.2f GB  S %.9f' % (time.time() - t_start, cid, status, wall, pk, S), flush=True)
            del running[cid]
        used = sum(c['mem'] for (_, _, c, _) in running.values())
        i = 0
        while i < len(pending) and len(running) < args.workers:
            c = pending[i]
            if used + c['mem'] <= args.mem_budget_gb * 1e9 or not running:
                if used + c['mem'] > args.mem_budget_gb * 1e9:
                    print('warning: %s alone exceeds the memory budget (est %.1f GB); running it alone' % (c['id'], c['mem'] / 1e9), flush=True)
                cmd = [args.python, reduced, '--R', str(c['R']), '--offset-sixteenths', str(c['k']), '--L', str(c['L']),
                       '--method', args.method, '--mem-limit-gb', str(args.mem_limit_gb),
                       '--threads', str(args.threads_per_worker), '--out', c['out']]
                lf = open(os.path.join(logdir, c['id'] + '.log'), 'a')
                lf.write('# ' + ' '.join(cmd) + '\n')
                lf.flush()
                proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, preexec_fn=preexec)
                running[c['id']] = (proc, time.time(), c, lf)
                used += c['mem']
                print('[%6.0f s] launched %-16s (est %.2f h, %.1f GB; running %d, pending %d)'
                      % (time.time() - t_start, c['id'], c['hours'], c['mem'] / 1e9, len(running), len(pending) - 1), flush=True)
                pending.pop(i)
            else:
                i += 1
        time.sleep(args.poll)
    log.close()
    print('all cases finished (%.0f s)' % (time.time() - t_start), flush=True)


if __name__ == '__main__':
    main()
