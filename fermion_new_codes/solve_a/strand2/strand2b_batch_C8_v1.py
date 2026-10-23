#!/usr/bin/env python
"""Strand 2b C8 batch driver (builder analyst; a copy of the R4 batch driver v8 pointed at strand2b_row_C8_v2.solve_case, the
sec. 10 route): the cases of SPEC_strand2b_row sec. 1 on the cube, one case per pool worker (single-threaded), one JSON per case,
existing outputs skipped.  Blocks (1,0) and (2,0) (--blocks selects); m in {0.1, 0.2, 0.4}; (g_M, g_B, Lambda) in {(0,0,1),
(0.05,0,1), (0.2,1,1), (0.05,0,inf)}; k in {1, 10, 100, 1e3, 1e4}."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import socket
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand2b_row_C8_v2 as RC  # noqa: E402

COUPLINGS = [(0.0, 0.0, "1"), (0.05, 0.0, "1"), (0.2, 1.0, "1"), (0.05, 0.0, "inf")]
KS = (1.0, 10.0, 100.0, 1e3, 1e4)
MS = (0.1, 0.2, 0.4)


def tag(c):
    blk, m, gM, gB, L, k = c
    return "C8_b%s_m%g_gM%g_gB%g_L%s_k%g" % (blk.replace(",", ""), m, gM, gB, L, k)


_G = {}


def init():
    import strand2a_lib as S2
    import native_core as nc
    _G["lat"] = S2.lattice("C8")
    _G["K"], _ = nc.K_matrix(_G["lat"]["positions"].tolist())


def work(args):
    c, out, enc, uni = args
    f = os.path.join(out, "case_%s.json" % tag(c))
    if os.path.exists(f):
        return tag(c), 0.0, "skipped"
    blk, m, gM, gB, L, k = c
    Lam = np.inf if L == "inf" else float(L)
    t0 = time.time()
    try:
        rec = RC.solve_case(_G["lat"], _G["K"], tuple(int(v) for v in blk.split(",")), m, gM, gB, Lam, k, enc, uni)
    except Exception as e:  # report, do not hide
        rec = {"error": repr(e), "case": list(c)}
    rec["host"] = socket.gethostname()
    with open(f + ".tmp", "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    os.replace(f + ".tmp", f)
    return tag(c), time.time() - t0, "error" if "error" in rec else "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--blocks", default="1,0;2,0")
    ap.add_argument("--ms", default="0.1,0.2,0.4")
    ap.add_argument("--ks", default="1,10,100,1000,10000")
    ap.add_argument("--couplings", default="all")
    ap.add_argument("--no-enclosure", action="store_true")
    ap.add_argument("--no-unique", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    blocks = a.blocks.split(";")
    ms = [float(v) for v in a.ms.split(",")]
    ks = [float(v) for v in a.ks.split(",")]
    coups = COUPLINGS if a.couplings == "all" else [COUPLINGS[int(i)] for i in a.couplings.split(",")]
    cases = [(b, m, gM, gB, L, k) for b in blocks for m in ms for (gM, gB, L) in coups for k in ks]
    t0 = time.time()
    print("strand2b C8 batch: %d cases, %d workers, host %s" % (len(cases), a.workers, socket.gethostname()), flush=True)
    with Pool(a.workers, initializer=init) as pool:
        for i, (t, sec, st) in enumerate(pool.imap_unordered(work, [(c, a.out, not a.no_enclosure, not a.no_unique) for c in cases],
                                                            chunksize=1), 1):
            print("  %d/%d %s %s %.0f s (total %.0f s)" % (i, len(cases), t, st, sec, time.time() - t0), flush=True)
    print("strand2b C8 batch done in %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
