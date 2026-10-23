#!/usr/bin/env python3
"""
j4m_driver_v1.py -- resumable driver of the massive J4 rerun (busi lane, task 4; spec sections 7 and 8).

Plan (build_plan): for every primary radius R the 8 version-1 balls (R + k/16, k odd in -7..7) at both masses in the
production box (192 for R <= 24, 224 above); the massless 3-box ladders 4R, 6R, 8R at R 12, 16, 20, 24 (the A_4
reading); the massive box ladders 48, 72, 96 at R 12 and 96, 128 at R 16 (the exponential check); the five P22
comparison balls n = R^2 at R 12, 16, 20, 24 (box 192) and 28 (224) at both masses.  Cases whose record (same
canonical ball, box and mass) exists are skipped.  The kernel tables of every (box, mass) and of the infinite
lattice are computed once before the workers start.  Each case runs j4m_case_v1.py in its own process (nice, at most
--threads threads); after each case the record pair is copied with --copy-cmd (e.g. 'scp -q {files} box:/dir/').
"""
import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
J4_DIR = os.path.normpath(os.path.join(HERE, '..', 'j4'))
for _p in (HERE, J4_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import j4window_v2 as jw          # noqa: E402
import j4lib_v2 as jl2            # noqa: E402
import j4m_lib_v1 as jm           # noqa: E402
import j4m_kernel_v1 as jk        # noqa: E402
from j4m_case_v1 import case_id   # noqa: E402

RADII_DEFAULT = [12, 14, 16, 18, 20, 22, 24, 26, 28]
MASSES_DEFAULT = [0.0, 0.2]
LADDER_RADII = [12, 16, 20, 24]
LADDER_FACTORS = (4, 6, 8)
MBOX = {12: (48, 72, 96), 16: (96, 128)}
P22_RADII = [12, 16, 20, 24, 28]


def production_box(R):
    return 192 if R <= 24 else 224


def estimate_seconds(n, mu):
    """Rough single-case time (two routes) for the log: ~ n^2 assembly plus the eigenproblems."""
    base = 3.0 + 1.6e-8 * float(n) ** 2
    return base * (1.0 if mu != 0.0 else 0.35)


def build_plan(radii, masses, ladder=True, mbox=True, p22=True):
    cases, seen = [], set()

    def add(n, L, mu, R, k, kind):
        n = jw.canonical_ball(int(n))
        key = (n, int(L), float(mu))
        if key in seen:
            return
        seen.add(key)
        cases.append(dict(id=case_id(R, n, L, mu), n=n, L=int(L), mu=float(mu), R=R, k=k, kind=kind,
                          M=2 * int(math.floor(math.sqrt(n))), est_s=estimate_seconds(4.19 * n ** 1.5, mu)))

    for R in radii:
        L = production_box(R)
        for k in jw.V1_OFFSETS:
            n = jw.ball_at_radius(R + k / 16.0)
            for mu in masses:
                add(n, L, mu, R, k, 'prod')
    if ladder:
        for R in LADDER_RADII:
            if R not in radii:
                continue
            for k in jw.V1_OFFSETS:
                n = jw.ball_at_radius(R + k / 16.0)
                for f in LADDER_FACTORS:
                    add(n, f * R, 0.0, R, k, 'ladder')
    if mbox:
        for R, Ls in MBOX.items():
            if R not in radii:
                continue
            for k in jw.V1_OFFSETS:
                n = jw.ball_at_radius(R + k / 16.0)
                for L in Ls:
                    for mu in masses:
                        if mu != 0.0:
                            add(n, L, mu, R, k, 'mbox')
    if p22:
        for R in P22_RADII:
            if R not in radii:
                continue
            for mu in masses:
                add(R * R, production_box(R), mu, R, None, 'p22')
    return cases


def scan_done(dirs):
    """(canonical ball, L, mu) -> path of an existing complete record."""
    done = {}
    for d in dirs if isinstance(dirs, (list, tuple)) else [dirs]:
        if not d or not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith('.json'):
                continue
            p = os.path.join(d, f)
            try:
                with open(p) as fh:
                    r = json.load(fh)
                S = r['routes']['box']['S']
                if S == S:
                    done[(jw.canonical_ball(int(r['n2max'])), int(r['L']), float(r['mu']))] = p
            except Exception:
                continue
    return done


def prewarm(cases, cache_dir, log):
    box_M = {}
    for c in cases:
        key = (c['L'], c['mu'])
        box_M[key] = max(box_M.get(key, 0), c['M'])
    for (L, mu), M in sorted(box_M.items()):
        jm.box_tables_cached(L, mu, M, cache_dir, log=log)
    for L, M in sorted({(c['L'], c['M']) for c in cases if c['mu'] == 0.0}):
        jl2.kernel_table_cached(L, M, os.path.join(cache_dir, 'v2_massless'), log=log)
    inf_M = {}
    for c in cases:
        inf_M[c['mu']] = max(inf_M.get(c['mu'], 0), c['M'])
    for mu, M in sorted(inf_M.items()):
        jk.inf_tables_cached(mu, M, cache_dir, log=log)


def copy_files(cmd_template, files):
    cmd = cmd_template.replace('{files}', ' '.join(shlex.quote(f) for f in files))
    try:
        return subprocess.run(cmd, shell=True).returncode == 0
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description='J4 massive rerun: resumable driver')
    ap.add_argument('--radii', type=int, nargs='+', default=RADII_DEFAULT)
    ap.add_argument('--masses', type=float, nargs='+', default=MASSES_DEFAULT)
    ap.add_argument('--outdir', default='j4m_records')
    ap.add_argument('--cache-dir', default=None, help='kernel caches (default outdir/../cache)')
    ap.add_argument('--no-ladder', action='store_true')
    ap.add_argument('--no-mbox', action='store_true')
    ap.add_argument('--no-p22', action='store_true')
    ap.add_argument('--no-inf', action='store_true')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--threads', type=int, default=4, help='threads per case (at most 16)')
    ap.add_argument('--method', choices=['eigh', 'torch'], default='eigh')
    ap.add_argument('--nice', type=int, default=10)
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--copy-cmd', default=None, help="shell template with {files}, e.g. 'scp -q {files} box:<internal>/records'")
    ap.add_argument('--poll', type=float, default=5.0)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.threads > 16:
        ap.error('at most 16 threads per job')
    outdir = os.path.abspath(args.outdir)
    cache_dir = os.path.abspath(args.cache_dir) if args.cache_dir else os.path.join(os.path.dirname(outdir), 'cache')
    logdir = os.path.join(outdir, 'logs')
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    log = lambda s: print(s, flush=True)   # noqa: E731

    cases = build_plan(args.radii, args.masses, ladder=not args.no_ladder, mbox=not args.no_mbox, p22=not args.no_p22)
    done = scan_done(outdir)
    pending = [c for c in cases if (c['n'], c['L'], c['mu']) not in done]
    pending.sort(key=lambda c: (c['n'], c['mu'], c['L']))
    log('plan: %d cases, %d done, %d pending, about %.1f h of single-worker time pending'
        % (len(cases), len(cases) - len(pending), len(pending), sum(c['est_s'] for c in pending) / 3600.0))
    by = {}
    for c in cases:
        by.setdefault(c['kind'], [0, 0])
        by[c['kind']][0] += 1
        by[c['kind']][1] += 0 if (c['n'], c['L'], c['mu']) in done else 1
    for kind in sorted(by):
        log('  %-7s %4d cases, %4d pending' % (kind, by[kind][0], by[kind][1]))
    if args.dry_run:
        for c in pending:
            log('  %-26s %-7s est %6.0f s' % (c['id'], c['kind'], c['est_s']))
        return
    if pending:
        prewarm(pending, cache_dir, log)

    tsv = os.path.join(outdir, 'production_log.tsv')
    logf = open(tsv, 'a')
    if logf.tell() == 0:
        logf.write('case\tR\tn\tL\tmu\tkind\tstatus\twall_s\tS_box\tS_inf\tbox_minus_inf\tcopied\n')
        logf.flush()
    failures = open(os.path.join(outdir, 'copy_failures.txt'), 'a')
    env = dict(os.environ)
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[v] = str(args.threads)
    preexec = (lambda: os.nice(args.nice)) if args.nice else None
    running = {}
    t_start = time.time()
    code = os.path.join(HERE, 'j4m_case_v1.py')
    while pending or running:
        for cid in list(running):
            proc, t0, c, lf = running[cid]
            if proc.poll() is None:
                continue
            lf.close()
            wall = time.time() - t0
            jpath = os.path.join(outdir, c['id'] + '.json')
            zpath = os.path.join(outdir, c['id'] + '_eig.npz')
            rec = None
            if os.path.exists(jpath):
                try:
                    with open(jpath) as fh:
                        rec = json.load(fh)
                except Exception:
                    rec = None
            if rec is not None and 'routes' in rec and 'box' in rec['routes']:
                status = 'OK'
                S_box = rec['routes']['box']['S']
                S_inf = rec['routes'].get('inf', {}).get('S', float('nan'))
                bmi = rec.get('checks', {}).get('box_minus_inf', float('nan'))
                copied = ''
                if args.copy_cmd:
                    ok = copy_files(args.copy_cmd, [jpath, zpath])
                    copied = 'yes' if ok else 'FAILED'
                    if not ok:
                        failures.write('%s\n%s\n' % (jpath, zpath))
                        failures.flush()
            else:
                status, S_box, S_inf, bmi, copied = 'FAIL(rc=%s)' % proc.returncode, float('nan'), float('nan'), float('nan'), ''
            logf.write('%s\t%s\t%d\t%d\t%s\t%s\t%s\t%.0f\t%.12f\t%.12f\t%.3e\t%s\n'
                       % (cid, c['R'], c['n'], c['L'], jm.mu_tag(c['mu']), c['kind'], status, wall, S_box, S_inf, bmi, copied))
            logf.flush()
            log('[%7.0f s] %-26s %s  wall %.0f s  S_box %.9f  box-inf %.2e  %s' % (time.time() - t_start, cid, status, wall, S_box, bmi, copied))
            del running[cid]
        while pending and len(running) < args.workers:
            c = pending.pop(0)
            cmd = [args.python, code, '--n2max', str(c['n']), '--L', str(c['L']), '--mu', repr(c['mu']), '--kind', c['kind'],
                   '--outdir', outdir, '--cache-dir', cache_dir, '--threads', str(args.threads), '--method', args.method]
            if c['R'] is not None:
                cmd += ['--window-centre', str(c['R'])]
            if c['k'] is not None:
                cmd += ['--offset', str(c['k'])]
            if args.no_inf:
                cmd.append('--no-inf')
            lf = open(os.path.join(logdir, c['id'] + '.log'), 'a')
            lf.write('# ' + ' '.join(cmd) + '\n')
            lf.flush()
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, preexec_fn=preexec)
            running[c['id']] = (proc, time.time(), c, lf)
            log('[%7.0f s] launched %-26s %-7s (est %.0f s; running %d, pending %d)'
                % (time.time() - t_start, c['id'], c['kind'], c['est_s'], len(running), len(pending)))
        time.sleep(args.poll)
    logf.close()
    failures.close()
    log('all cases finished (%.0f s)' % (time.time() - t_start))


if __name__ == '__main__':
    main()
