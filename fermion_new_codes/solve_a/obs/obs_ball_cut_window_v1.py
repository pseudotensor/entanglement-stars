#!/usr/bin/env python3
"""obs_ball_cut_window_v1.py (builder analyst; S6 = J4 registry diagnostic): the exact unit-window registry average of the ball entropy.

The eight registry offsets R + (2i + 1)/16, i = -4 .. 3, are the midpoint rule for the average of S(R') over R' in [R - 1/2, R + 1/2].
S(R') is a staircase: the ball {|x| <= R'} changes only when R'^2 passes a populated shell k = |x|^2.  This script enumerates every
distinct ball in the window (the ball at R - 1/2 and one per shell entering inside it), computes each entropy by obs_ball_cut_v1.py's
irrep-block route (imported unmodified as a library), and prints the exact window average S_win = sum_segments S w (w the segment
lengths in R'), the eight-point midpoint average read off the same staircase (equal to obs_ball_cut_v1.py's registry mean at the same box),
and the exact window means of the fit functions (mean R'^2 = R^2 + 1/12, mean R' = R, mean ln R').  CPU; one box L = c R.
Args: --R ... --box-ratio --solver --out-jsonl --md."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "16")
import argparse
import json
import math
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import obs_ball_cut_v1 as B  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, nargs="+", required=True)
    ap.add_argument("--box-ratio", type=float, default=6.0)
    ap.add_argument("--solver", choices=["svd", "gram"], default="svd")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    cub = B.Cubic()
    md = ["## S6 (J4) exact unit-window registry averages (obs_ball_cut_window_v1.py, box %gR, solver %s)" % (a.box_ratio, a.solver), "",
          "| R | L | balls in window | S_win (exact) | S_mid8 (8 midpoints) | S_mid8 - S_win | S_win / 4 pi (R^2 + 1/12) | S_mid8 / 4 pi (R^2 + 0.08203) |",
          "|---|---|---|---|---|---|---|---|"]
    out = open(a.out_jsonl, "a")
    for R in a.R:
        lo, hi = R - 0.5, R + 0.5
        L = int(round(a.box_ratio * R))
        L += L % 2
        rmax = 2 * int(math.ceil(hi)) + 2
        t0 = time.time()
        tb = B.kernel_tables(L, rmax)
        sgt = B.signed_tables(tb["gtab"], tb["itab"])
        X = B.ball_sites(hi)
        kall = np.unique((X * X).sum(1))
        kin = kall[(kall > lo * lo) & (kall <= hi * hi)]
        K0 = int(kall[kall <= lo * lo].max())
        Ks = [K0] + [int(k) for k in kin]
        edges = [lo] + [math.sqrt(k) for k in kin] + [hi]
        w = np.diff(edges)
        segs = []
        for K, wk in zip(Ks, w):
            t1 = time.time()
            bs = B.ball_structure(math.sqrt(K + 0.5), cub)
            blocks = B.assemble_blocks(bs, cub, sgt, rmax)
            rows, S, frob_b = B.block_entropies(blocks, cub, a.solver, "cpu")
            del blocks
            segs.append(dict(K=K, n=bs["n"], S=S, w=float(wk), frob_blocks=frob_b, t=time.time() - t1))
            B.log("window R %g L %d: shell K %d n %d S %.10f weight %.6f (%.1f s)" % (R, L, K, bs["n"], S, wk, time.time() - t1))
        Sw = float(sum(s["S"] * s["w"] for s in segs))
        mids = [R + (2 * i + 1) / 16.0 for i in range(-4, 4)]
        Smid = []
        for x in mids:
            K = int(kall[kall <= x * x].max())
            Smid.append(next(s["S"] for s in segs if s["K"] == K))
        Sm8 = float(np.mean(Smid))
        mlog = ((hi * math.log(hi) - hi) - (lo * math.log(lo) - lo))
        rec = dict(script="obs_ball_cut_window_v1.py", R=R, L=L, box_ratio=a.box_ratio, solver=a.solver, segments=segs, S_win=Sw, S_mid8=Sm8,
                   mean_R2=R * R + 1.0 / 12, mean_R=R, mean_lnR=mlog, weight_sum=float(w.sum()), t=time.time() - t0)
        out.write(json.dumps(rec) + "\n")
        out.flush()
        md.append("| %g | %d | %d | %.10f | %.10f | %.3e | %.6f | %.6f |" % (R, L, len(segs), Sw, Sm8, Sm8 - Sw, Sw / (4 * math.pi * (R * R + 1.0 / 12)),
                                                                        Sm8 / (4 * math.pi * (R * R + 168.0 / 2048))))
    out.close()
    with open(a.md, "a") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
