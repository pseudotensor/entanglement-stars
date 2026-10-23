#!/usr/bin/env python
"""Strand 2b R4 batch driver (builder analyst): all cases of SPEC_strand2b_row sec. 1 on R4 through strand2b_row_R4.solve_case,
one case per pool worker (single-threaded), one JSON per case, existing outputs skipped.  Cases: blocks (1,0), (2,0), (2,1), (1,1),
the q = 0 control (0,0) and (0,1) (a builder addition for the (2,1) hole-contribution comparison of sec. 4); m in {0.1, 0.2, 0.4};
(g_M, g_B, Lambda) in {(0,0,-), (0.05,0,1), (0.2,1,1), (0.05,0,inf)}; k in {1, 10, 100, 1e3, 1e4}."""
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
import strand2b_row_R4 as RR  # noqa: E402

COUPLINGS = [(0.0, 0.0, "1"), (0.05, 0.0, "1"), (0.2, 1.0, "1"), (0.05, 0.0, "inf")]
KS = (1.0, 10.0, 100.0, 1e3, 1e4)
MS = (0.1, 0.2, 0.4)


def tag(c):
    blk, m, gM, gB, L, k = c
    return "R4_b%s_m%g_gM%g_gB%g_L%s_k%g" % (blk.replace(",", ""), m, gM, gB, L, k)


_G = {}


def init():
    import strand2a_lib as S2
    import strand2b_lib as B
    import native_core as nc
    _G["lat"] = S2.lattice("R4")
    _G["K"], _ = nc.K_matrix(nc.lattice("ring4")["positions"])
    _G["fops"] = B.FockOps(8)


def work(args):
    c, out, enc, uni = args
    f = os.path.join(out, "case_%s.json" % tag(c))
    if os.path.exists(f):
        return tag(c), 0.0, "skipped"
    blk, m, gM, gB, L, k = c
    Lam = np.inf if L == "inf" else float(L)
    t0 = time.time()
    try:
        rec = RR.solve_case(_G["lat"], _G["K"], RR.BLOCKS[blk], m, gM, gB, Lam, k, _G["fops"], enc, uni)
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
    ap.add_argument("--blocks", default="1,0;2,0;2,1;1,1;0,0;0,1")
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
    print("strand2b R4 batch: %d cases, %d workers, host %s" % (len(cases), a.workers, socket.gethostname()), flush=True)
    with Pool(a.workers, initializer=init) as pool:
        for i, (t, sec, st) in enumerate(pool.imap_unordered(work, [(c, a.out, not a.no_enclosure, not a.no_unique) for c in cases],
                                                            chunksize=1), 1):
            print("  %d/%d %s %s %.0f s (total %.0f s)" % (i, len(cases), t, st, sec, time.time() - t0), flush=True)
    print("strand2b R4 batch done in %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
