#!/usr/bin/env python
"""Strand 2d sec. 10 item 3 (builder analyst): J_NN by the exact sum over states at the dressed B4 flat point.  The (2,0) block's
H[N] = e^S H0pp(sum_x N_x X_x) e^S is exactly linear in N (strand2d_lib_v1.PairBlock.H), so for N = 1 + lam c_q the lowest level's
second derivative is J_NN(q) = -2 sum_{r>0} |<r|H_N(q)|0>|^2/(E_r - E_0), with H_N(q) = PairBlock.H(b0, c_q) (the same affine map at the
profile c_q = cos(q x_1)) and the full eigendecomposition of H[1] (8128-dimensional at (0.2, 1)); also the first derivative <0|H_N(q)|0>,
the contributions of the lowest 120 levels (the cluster) and of the rest, and the check H[1 + c] - H[1] - H_N(c) = 0.  q in {0, pi/2, pi}.
Writes <out>/jnn_B4_gM<gM>_gB<gB>.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

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
    t0 = time.time()
    lat = L2D.lattice("B4")
    n = lat["nsite"]
    one = np.ones(n)
    pb = L2D.PairBlock(lat, a.m, a.gM, a.gB, 1.0, a.device)
    b0 = pb.build(one)
    H1 = pb.H(b0, one)
    w, V = torch.linalg.eigh(H1)
    E = w.cpu().numpy()
    print("eigh done (%.0f s)" % (time.time() - t0), flush=True)
    x1 = lat["coord"][:, 0].astype(float)
    res = {"meta": {"gM": a.gM, "gB": a.gB, "m": a.m, "device": a.device, "dim": int(len(E)), "E_lowest": [float(x) for x in E[:6]]}, "rows": []}
    v0 = V[:, 0]
    for q in (0.0, np.pi / 2, np.pi):
        c = one if q == 0 else np.cos(q * x1)
        HN = pb.H(b0, c)
        lin = float(torch.max(torch.abs(pb.H(b0, one + c) - H1 - HN)).cpu())
        amp = (V.conj().T @ (HN @ v0)).cpu().numpy()        # <r|H_N|0>
        d1 = float(np.real(amp[0]))
        den = E[1:] - E[0]
        terms = np.abs(amp[1:]) ** 2 / den
        J = -2 * float(np.sum(terms))
        J_cluster = -2 * float(np.sum(terms[:119]))
        res["rows"].append({"q": q, "dE_dlamN": d1, "J_NN": J, "J_NN_cluster_levels_1_to_119": J_cluster, "J_NN_rest": J - J_cluster,
                            "largest_terms": [float(x) for x in np.sort(terms)[::-1][:5]], "linearity_check": lin})
        print(json.dumps(res["rows"][-1]), "(%.0f s)" % (time.time() - t0), flush=True)
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "jnn_B4_gM%g_gB%g.json" % (a.gM, a.gB)), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
