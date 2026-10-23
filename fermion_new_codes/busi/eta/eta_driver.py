#!/usr/bin/env python3
"""eta_driver.py -- run eta_direction.py over the cubic orbits with max component <= pmax (or an explicit
list), one single-thread worker process per case, largest estimated cost first; resumable (existing JSON
files are skipped); --dry-run prints the cost model; after the sweep the tables are combined.
"""
import os
import sys
import argparse
import math
import time
import subprocess


def _early_threads():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--threads', type=int, default=1)
    a, _ = pre.parse_known_args()
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[v] = str(a.threads)
    return a.threads


THREADS = _early_threads()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eta_common as ec  # noqa: E402
from eta_direction import case_name  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_orbits(items):
    out = []
    for it in items:
        q = tuple(int(v) for v in it.split(','))
        if len(q) != 3 or not ec.is_primitive(q):
            raise SystemExit(f"bad orbit {it}: need a primitive a,b,c")
        out.append(q)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--threads', type=int, default=1, help='BLAS threads per worker (1 recommended)')
    ap.add_argument('--pmax', type=int, default=10)
    ap.add_argument('--orbits', nargs='*', default=None, help='explicit list a,b,c ... (overrides --pmax)')
    ap.add_argument('--min-pnorm', type=float, default=0.0)
    ap.add_argument('--max-pnorm', type=float, default=1e9)
    ap.add_argument('--masses', nargs='+', type=float, default=[0.0])
    ap.add_argument('--levels', nargs='+', type=int, default=[0, 1])
    ap.add_argument('--h0', type=float, default=0.1)
    ap.add_argument('--cL', type=float, default=10.0)
    ap.add_argument('--Lmin', type=int, default=24)
    ap.add_argument('--outdir', default='dirs')
    ap.add_argument('--logdir', default='logs')
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--nice', type=int, default=10)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--no-combine', action='store_true')
    ap.add_argument('--table', default='eta_table.tsv')
    ap.add_argument('--btable', default='b_table.tsv')
    args = ap.parse_args()

    orbits = parse_orbits(args.orbits) if args.orbits else ec.direction_orbits(args.pmax)
    orbits = [q for q in orbits if args.min_pnorm <= math.sqrt(sum(v * v for v in q)) <= args.max_pnorm]
    cases = []
    for q in orbits:
        for m in args.masses:
            for lv in args.levels:
                est = ec.estimate_case(q, m, lv, args.h0, args.cL, args.Lmin, args.device)
                path = os.path.join(args.outdir, case_name(q, m, lv))
                cases.append(dict(p=q, mass=m, level=lv, est=est, path=path, done=os.path.exists(path)))
    cases.sort(key=lambda c: -c['est']['seconds'])
    tot = sum(c['est']['seconds'] for c in cases)
    todo = [c for c in cases if not c['done']]
    tot_todo = sum(c['est']['seconds'] for c in todo)
    print(f"{len(orbits)} orbits, {len(cases)} cases ({len(todo)} to do); model cost total {tot/3600:.1f} core-hours, "
          f"to do {tot_todo/3600:.1f} core-hours, about {tot_todo/3600/max(args.workers,1):.1f} wall-hours on {args.workers} workers "
          f"(T_EIG_1000 = {ec.T_EIG_1000} s; calibrate from the first finished cases)")
    print(f"largest case: p={cases[0]['p']} m={cases[0]['mass']:g} level={cases[0]['level']} n_max={cases[0]['est']['n_max']} "
          f"points={cases[0]['est']['n_points']} est {cases[0]['est']['seconds']/60:.1f} min mem {cases[0]['est']['mem_gb']:.1f} GB")
    if args.dry_run:
        for c in cases[:25]:
            e = c['est']
            print(f"  p={c['p']} m={c['mass']:g} L={c['level']} N={e['N1']}x{e['N2']} pts={e['n_points']} n_max={e['n_max']} "
                  f"est={e['seconds']/60:.1f} min mem={e['mem_gb']:.1f} GB {'done' if c['done'] else ''}")
        return

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)
    pending = list(todo)
    running = []
    t0 = time.time()
    finished = 0
    failed = []

    def preexec():
        try:
            os.nice(args.nice)
        except OSError:
            pass

    while pending or running:
        while pending and len(running) < args.workers:
            c = pending.pop(0)
            q = c['p']
            cmd = [sys.executable, os.path.join(HERE, 'eta_direction.py'), '--threads', str(args.threads),
                   '--p', str(q[0]), str(q[1]), str(q[2]), '--masses', f"{c['mass']:g}", '--levels', str(c['level']),
                   '--h0', str(args.h0), '--cL', str(args.cL), '--Lmin', str(args.Lmin), '--outdir', args.outdir,
                   '--device', args.device]
            logpath = os.path.join(args.logdir, case_name(q, c['mass'], c['level']).replace('.json', '.log'))
            logf = open(logpath, 'a')
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, preexec_fn=preexec)
            running.append((proc, c, logf, time.time()))
            print(f"[{ec.now()}] start p={q} m={c['mass']:g} L={c['level']} est {c['est']['seconds']/60:.1f} min "
                  f"({len(pending)} pending, {len(running)} running)", flush=True)
        still = []
        for proc, c, logf, ts in running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, c, logf, ts))
                continue
            logf.close()
            finished += 1
            ok = rc == 0 and os.path.exists(c['path'])
            if not ok:
                failed.append(c)
            print(f"[{ec.now()}] {'done' if ok else 'FAILED'} p={c['p']} m={c['mass']:g} L={c['level']} in {(time.time()-ts)/60:.1f} min "
                  f"(est {c['est']['seconds']/60:.1f}); {finished}/{len(todo)} finished, elapsed {(time.time()-t0)/3600:.2f} h", flush=True)
        running = still
        time.sleep(2.0)
    print(f"sweep finished: {finished} cases, {len(failed)} failed, {(time.time()-t0)/3600:.2f} h wall")
    for c in failed:
        print(f"  failed: p={c['p']} m={c['mass']:g} level={c['level']} (see {args.logdir})")
    if not args.no_combine:
        subprocess.call([sys.executable, os.path.join(HERE, 'eta_direction.py'), '--threads', '1', '--combine',
                         '--outdir', args.outdir, '--table', args.table, '--btable', args.btable])


if __name__ == '__main__':
    main()
