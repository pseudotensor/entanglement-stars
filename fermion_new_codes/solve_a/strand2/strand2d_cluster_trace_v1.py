#!/usr/bin/env python
"""Strand 2d sec. 11 (spec solve/specs/SPEC_strand2d_jet.md sec. 11; builder analyst): the exact cluster-trace clock susceptibility of the
dressed B4 (2,0) block at (g_M, g_B) = (0.2, 1), m = 0.1, Lambda = 1.  The block H[N] = e^S H0pp(sum_x N_x X_x) e^S is exactly affine in N
(strand2d_lib_v1.PairBlock.H, as in strand2d_jnn_v1.py), so with the full eigendecomposition H[1] = sum_r E_r |r><r| (recomputed here; the
sec. 10 run did not store it) and H_N(q) = PairBlock.H(b0, c_q), c_q = cos(q x_1):
  J_NN^cl  = -(2/120) sum_{a < 120} sum_{r >= 120} |<r|H_N|a>|^2/(E_r - E_a)   (the lowest 120 levels, the cluster),
  J_NN^(8) = -(2/8)   sum_{a < 8}   sum_{r >= 8}   |<r|H_N|a>|^2/(E_r - E_a),
  J_NN^(0) = -2 sum_{r >= 1} |<r|H_N|0>|^2/(E_r - E_0)  (the ground level, sec. 10's -237.5886 at pi and -0.16498 at pi/2, as the check),
for q = pi/2 and pi; the ten largest terms of the ground-level sum with their level indices r and E_r - E_0; the first-order within-cluster
matrix's trace/120 (the cluster average's first derivative) and the largest |<a|H_N|b>| inside the cluster.  Writes <out>/cluster_trace.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import json
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand2d_lib_v1 as L2D  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gM", type=float, default=0.2)
    ap.add_argument("--gB", type=float, default=1.0)
    ap.add_argument("--m", type=float, default=0.1)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    t0 = time.time()
    lat = L2D.lattice("B4")
    n = lat["nsite"]
    one = np.ones(n)
    pb = L2D.PairBlock(lat, a.m, a.gM, a.gB, 1.0, a.device)
    b0 = pb.build(one)
    H1 = pb.H(b0, one)
    w, V = torch.linalg.eigh(H1)
    E = w.cpu().numpy()
    print("eigh done, dim %d (%.0f s)" % (len(E), time.time() - t0), flush=True)
    x1 = lat["coord"][:, 0].astype(float)
    res = {"meta": {"gM": a.gM, "gB": a.gB, "m": a.m, "dim": int(len(E)), "E_lowest": [float(x) for x in E[:10]],
                    "cluster_width_E119_minus_E0": float(E[119] - E[0]), "gap_E120_minus_E119": float(E[120] - E[119]),
                    "E1_minus_E0": float(E[1] - E[0])}, "rows": []}
    for q in (np.pi / 2, np.pi):
        c = np.cos(q * x1)
        HN = pb.H(b0, c)
        lin = float(torch.max(torch.abs(pb.H(b0, one + c) - H1 - HN)).cpu())
        A = (V.conj().T @ (HN @ V[:, :120])).cpu().numpy()     # <r|H_N|a>, r all, a < 120
        P2 = np.abs(A) ** 2
        row = {"q": q, "affinity_check": lin}
        for nc in (120, 8):
            num = P2[nc:, :nc]
            den = E[nc:, None] - E[None, :nc]
            row["J_NN_trace_lowest_%d" % nc] = float(-2.0 / nc * np.sum(num / den))
            row["first_derivative_trace_lowest_%d" % nc] = float(np.real(np.trace(A[:nc, :nc])) / nc)
        t = P2[1:, 0] / (E[1:] - E[0])
        row["J_NN_ground"] = float(-2 * np.sum(t))
        row["J_NN_ground_within_cluster_r_1_to_119"] = float(-2 * np.sum(t[:119]))
        row["J_NN_ground_outside_cluster"] = float(-2 * np.sum(t[119:]))
        top = np.argsort(t)[::-1][:10]
        row["ground_top10"] = [{"r": int(i + 1), "E_r_minus_E0": float(E[i + 1] - E[0]), "term": float(t[i]), "abs_amp": float(np.sqrt(P2[i + 1, 0]))}
                               for i in top]
        inner = np.abs(A[:120, :120])
        np.fill_diagonal(inner, 0.0)
        row["max_offdiag_within_cluster"] = float(inner.max())
        res["rows"].append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "ground_top10"}), "(%.0f s)" % (time.time() - t0), flush=True)
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "cluster_trace.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
