#!/usr/bin/env python
"""Strand 3 planar runs on one GPU (the paper's eta.converge, unmodified, with gpu = g and workers = 1, as its code
requires): the depth-64 jobs of masses.json, split round-robin over --ngpu GPUs; same output files as strand3_planar.py
(existing outputs skipped).  Builder analyst."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "orient"))
import eta as E  # noqa: E402
from strand3_planar import fname, cost  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--ngpu", type=int, default=3)
    ap.add_argument("--slot", type=int, default=-1, help="share index (defaults to --gpu)")
    ap.add_argument("--depth", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "planar"), exist_ok=True)
    jobs = [tuple(j) for j in json.load(open(args.masses))["planar_jobs"] if j[1] == args.depth]
    jobs.sort(key=lambda j: (-cost(j), j[3]))
    slot = args.gpu if args.slot < 0 else args.slot
    mine = jobs[slot::args.ngpu]
    t0 = time.time()
    print("gpu %d: %d jobs of depth %d, host %s" % (args.gpu, len(mine), args.depth, socket.gethostname()), flush=True)
    for i, (p, depth, nk, q) in enumerate(mine, 1):
        f = fname(args.out, p, depth, nk, q)
        if os.path.exists(f):
            print("  %d/%d %s skipped" % (i, len(mine), os.path.basename(f)), flush=True)
            continue
        t1 = time.time()
        rec = E.converge(tuple(p), depth=float(depth), nk=int(nk), mass=float(q), workers=1, gpu=args.gpu)
        rec.update(mass=float(q), depth=depth, nk=nk, seconds=time.time() - t1, host=socket.gethostname(), gpu=args.gpu)
        with open(f + ".tmp", "w") as fh:
            json.dump(rec, fh, indent=2)
        os.replace(f + ".tmp", f)
        print("  %d/%d %s done %.0f s (total %.0f s)" % (i, len(mine), os.path.basename(f), time.time() - t1, time.time() - t0),
              flush=True)
    print("gpu %d: done in %.0f s" % (args.gpu, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
