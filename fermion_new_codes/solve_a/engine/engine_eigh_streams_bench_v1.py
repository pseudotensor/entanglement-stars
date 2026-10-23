#!/usr/bin/env python
"""engine_eigh_streams_bench_v1.py (builder analyst; the goal of 19:57 UTC, economy (c): "run the 10 channels' diagonalizations on separate
CUDA streams"): the wall time of K channel eigendecompositions (complex128, cuSOLVER) run one after another on the default stream against
the same K run from K Python threads, each on its own torch.cuda.Stream (torch releases the GIL inside the solver), for K = 2, 5 and 10
channels of a stored collapse slice (the channels kappa = 1..K of engine_collapse_v5.Chan at the record's N, v). Prints the per-eigh
seconds, the speed-up, and the largest eigenvalue difference between the two routes. Args: --fields --record --kmax --m --gpu --reps --md."""
import argparse
import os
import sys
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument("--fields", required=True)
ap.add_argument("--record", type=int, default=-1)
ap.add_argument("--kmax", type=int, default=10)
ap.add_argument("--m", type=float, default=0.2)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--md", required=True)
a = ap.parse_args()
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_collapse_v5 as C5  # noqa: E402

C5.backend(a.gpu)
dev = C5.BK["dev"]
Z = np.load(a.fields, allow_pickle=True)
R = np.asarray(Z["R"], float); N = np.asarray(Z["N"][a.record], float); v = np.asarray(Z["v"][a.record], float)
mats = [C5.Chan(k, a.m, len(R), False).clocked_dense(N, v) for k in range(1, a.kmax + 1)]
torch.linalg.eigh(mats[0]); torch.cuda.synchronize(dev)
rows = []
for K in (2, 5, a.kmax):
    sub = mats[:K]
    ts_seq, ts_par = [], []
    for _ in range(a.reps):
        torch.cuda.synchronize(dev); t0 = time.time()
        e_seq = [torch.linalg.eigh(h)[0] for h in sub]
        torch.cuda.synchronize(dev); ts_seq.append(time.time() - t0)
        out = [None] * K
        streams = [torch.cuda.Stream(device=dev) for _ in range(K)]

        def work(i):
            with torch.cuda.stream(streams[i]):
                out[i] = torch.linalg.eigh(sub[i])[0]
            streams[i].synchronize()
        torch.cuda.synchronize(dev); t0 = time.time()
        th = [threading.Thread(target=work, args=(i,)) for i in range(K)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        torch.cuda.synchronize(dev); ts_par.append(time.time() - t0)
    de = max(float((x - y).abs().max()) for x, y in zip(e_seq, out))
    rows.append((K, min(ts_seq), min(ts_par), de))
    print("K %d: sequential %.2f s (%.2f s per eigh), threads and streams %.2f s (%.2f s per eigh), speed-up %.2f, max de %.1e" % (
        K, min(ts_seq), min(ts_seq) / K, min(ts_par), min(ts_par) / K, min(ts_seq) / min(ts_par), de), flush=True)
L = ["## The channel eigendecompositions on separate CUDA streams (engine_eigh_streams_bench_v1.py; %s, record %d, n = %d)" % (
    os.path.basename(a.fields), a.record, 2 * len(R)), "", "| channels | sequential s | per eigh | threads and streams s | per eigh | speed-up | max eigenvalue difference |",
     "|---|---|---|---|---|---|---|"]
for K, s1, s2, de in rows:
    L.append("| %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.1e |" % (K, s1, s1 / K, s2, s2 / K, s1 / s2, de))
L += ["", "- every number above is printed by the script (the minimum over %d repetitions)." % a.reps]
open(a.md, "w").write("\n".join(L) + "\n")
