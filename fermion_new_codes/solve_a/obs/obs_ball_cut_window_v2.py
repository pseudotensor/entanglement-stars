#!/usr/bin/env python3
"""obs_ball_cut_window_v2.py (builder analyst; S6 = J4): version 2 of obs_ball_cut_window_v1.py for a device (the H100): the same exact
unit-window average (every distinct ball in [R - 1/2, R + 1/2], one box L = cR), each ball's ten irrep blocks assembled and solved on
--device by obs_ball_cut_v2.py's torch routines (one block at a time; --solver gram or svd), the kernel tables, orbit data and entropy formula
of obs_ball_cut_v1.py (both imported unmodified).  Resumable: balls already in --out-jsonl's segment file are skipped.
Output: one JSONL line per ball (segment file <out-jsonl>.segments) and one line per radius in --out-jsonl (the format of window_v1).
Args: --R ... --box-ratio --solver --device --max-elems --out-jsonl --md."""
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
import obs_ball_cut_v2 as B2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, nargs="+", required=True)
    ap.add_argument("--box-ratio", type=float, default=6.0)
    ap.add_argument("--solver", choices=["svd", "gram"], default="gram")
    ap.add_argument("--device", required=True)
    ap.add_argument("--max-elems", type=float, default=5e7)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    torch = B2.torch_mod()
    dev = a.device
    cub = B.Cubic()
    segf = a.out_jsonl + ".segments"
    done = {}
    if os.path.exists(segf):
        for line in open(segf):
            if line.strip():
                s = json.loads(line)
                done[(float(s["R"]), int(s["L"]), int(s["K"]))] = s
    md = ["## S6 (J4) exact unit-window registry averages (obs_ball_cut_window_v2.py on %s, box %gR, solver %s)" % (dev, a.box_ratio, a.solver), "",
          "| R | L | balls in window | S_win (exact) | S_mid8 (8 midpoints) | S_mid8 - S_win | S_win / 4 pi (R^2 + 1/12) |", "|---|---|---|---|---|---|---|"]
    for R in a.R:
        lo, hi = R - 0.5, R + 0.5
        L = int(round(a.box_ratio * R))
        L += L % 2
        rmax = 2 * int(math.ceil(hi)) + 2
        t0 = time.time()
        tb = B.kernel_tables(L, rmax)
        sgt_dev = [torch.as_tensor(s.ravel(), dtype=torch.float64, device=dev) for s in B.signed_tables(tb["gtab"], tb["itab"])]
        X = B.ball_sites(hi)
        kall = np.unique((X * X).sum(1))
        kin = kall[(kall > lo * lo) & (kall <= hi * hi)]
        K0 = int(kall[kall <= lo * lo].max())
        Ks = [K0] + [int(k) for k in kin]
        edges = [lo] + [math.sqrt(k) for k in kin] + [hi]
        w = np.diff(edges)
        segs = []
        for K, wk in zip(Ks, w):
            key = (float(R), int(L), int(K))
            if key in done:
                segs.append(done[key])
                continue
            t1 = time.time()
            bs = B.ball_structure(math.sqrt(K + 0.5), cub)
            db = B2.DevBall(bs, cub, rmax, dev, torch)
            Ssum, frob_b, peak = 0.0, 0.0, 0
            for nm in B.IRREPS:
                if str(dev).startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(dev)
                Kb = B2.assemble_block_torch(bs, cub, nm, db, sgt_dev, dev, torch, a.max_elems)
                frob_b += cub.dim[nm] * float((Kb * Kb).sum().item())
                s, route = B2.solve_block_torch(Kb, a.solver, torch)
                me, mo = Kb.shape
                del Kb
                if str(dev).startswith("cuda"):
                    peak = max(peak, int(torch.cuda.max_memory_allocated(dev)))
                    torch.cuda.empty_cache()
                Ssum += cub.dim[nm] * (B.S_from_svals(s) + abs(me - mo) * B.LN2)
            fd, _, _ = B.frob_direct(bs, tb["gtab"], rmax)
            seg = dict(R=R, L=L, K=K, n=bs["n"], S=4.0 * Ssum, w=float(wk), frob_blocks=frob_b, frob_rel=(frob_b - fd) / fd,
                       t=time.time() - t1, device=str(dev), solver=a.solver, gpu_peak_bytes=peak)
            with open(segf, "a") as fh:
                fh.write(json.dumps(seg) + "\n")
            segs.append(seg)
            del db
            B.log("window R %g L %d: shell K %d n %d S %.10f weight %.6f frob rel %.1e (%.1f s, device peak %.2f GB)" % (
                R, L, K, bs["n"], seg["S"], wk, seg["frob_rel"], seg["t"], peak / 1e9))
        Sw = float(sum(s["S"] * s["w"] for s in segs))
        Smid = []
        for x in [R + (2 * i + 1) / 16.0 for i in range(-4, 4)]:
            K = int(kall[kall <= x * x].max())
            Smid.append(next(s["S"] for s in segs if s["K"] == K))
        Sm8 = float(np.mean(Smid))
        mlog = ((hi * math.log(hi) - hi) - (lo * math.log(lo) - lo))
        rec = dict(script="obs_ball_cut_window_v2.py", R=R, L=L, box_ratio=a.box_ratio, solver=a.solver, device=str(dev), segments=segs, S_win=Sw,
                   S_mid8=Sm8, mean_R2=R * R + 1.0 / 12, mean_R=R, mean_lnR=mlog, weight_sum=float(w.sum()), t=time.time() - t0)
        with open(a.out_jsonl, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        md.append("| %g | %d | %d | %.10f | %.10f | %.3e | %.6f |" % (R, L, len(segs), Sw, Sm8, Sm8 - Sw, Sw / (4 * math.pi * (R * R + 1.0 / 12))))
        del sgt_dev
    with open(a.md, "a") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
