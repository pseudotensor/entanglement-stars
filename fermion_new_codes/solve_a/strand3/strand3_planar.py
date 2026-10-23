#!/usr/bin/env python
"""Strand 3 A.1/A.2 (and the sphere's planar inputs): run the paper's eta.converge, unmodified, for a job list, one
job per pool worker (single-threaded BLAS), one JSON per run (spec sec. 3; builder analyst).

Jobs come from masses.json ("planar_jobs": [normal, depth, nk, q]); existing outputs are skipped (restartable).
Output: <out>/planar/planar_<n1>_<n2>_<n3>_d<depth>_nk<nk>_q<q>.json = converge's dict plus mass, depth, nk, seconds.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import socket
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "orient"))
import eta as E  # noqa: E402  (the paper's module, unmodified)


def fname(out, p, depth, nk, q):
    return os.path.join(out, "planar", "planar_%d_%d_%d_d%d_nk%d_q%.12g.json" % (tuple(p) + (depth, nk, q)))


def cost(job):
    p, depth, nk, q = job
    L = max(8, 2 * int(-(-depth * sum(v * v for v in p) ** 0.5 // 2)))
    return (4 * L) ** 3 * (2 * nk) ** 2


def run(args):
    job, out = args
    p, depth, nk, q = job
    f = fname(out, p, depth, nk, q)
    if os.path.exists(f):
        return f, 0.0, "skipped"
    t0 = time.time()
    rec = E.converge(tuple(p), depth=float(depth), nk=int(nk), mass=float(q), workers=1, gpu=-1)
    rec.update(mass=float(q), depth=depth, nk=nk, seconds=time.time() - t0, host=socket.gethostname())
    tmp = f + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=2)
    os.replace(tmp, f)
    return f, time.time() - t0, "done"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", default="", help="comma list of depth values to run (default all)")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N jobs after sorting (smoke tests)")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "planar"), exist_ok=True)
    jobs = [tuple(j) for j in json.load(open(args.masses))["planar_jobs"]]
    if args.only:
        keep = {int(v) for v in args.only.split(",")}
        jobs = [j for j in jobs if j[1] in keep]
    jobs.sort(key=lambda j: -cost(j))
    if args.limit:
        jobs = jobs[-args.limit:]
    t0 = time.time()
    print("planar: %d jobs, %d workers, host %s" % (len(jobs), args.workers, socket.gethostname()), flush=True)
    with Pool(args.workers) as pool:
        for i, (f, sec, st) in enumerate(pool.imap_unordered(run, [(j, args.out) for j in jobs], chunksize=1), 1):
            print("  %d/%d %s %s %.0f s (total %.0f s)" % (i, len(jobs), os.path.basename(f), st, sec, time.time() - t0),
                  flush=True)
    print("planar: done in %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
