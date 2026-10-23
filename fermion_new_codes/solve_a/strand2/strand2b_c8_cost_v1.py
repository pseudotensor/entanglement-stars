#!/usr/bin/env python
"""Strand 2b C8 cost probe (builder analyst): wall time of the pieces the enclosure and uniqueness stages repeat on the cube,
per block: one BlockModel.build (the block of rho^P_x at one psi), one residual F, one full Jacobian (jacobian_c8), at the flat
point z = (1, 1) and m 0.1, (g_M, g_B) = (0.2, 1), Lambda 1, k 1.  Timing only; writes <out>/c8_cost.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand2a_lib as S2  # noqa: E402
import strand2b_c8_lib as C  # noqa: E402
import native_core as nc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", default="1,0;2,0")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lat = S2.lattice("C8")
    K, _ = nc.K_matrix(lat["positions"].tolist())
    res = []
    for bs in a.blocks.split(";"):
        blk = tuple(int(v) for v in bs.split(","))
        t0 = time.time()
        bm = C.BlockModel(lat, 0.1, 0.2, 1.0, 1.0, blk)
        sys_ = C.SystemC8(bm, K, 1.0)
        t_init = time.time() - t0
        n = sys_.n
        z = np.ones(2 * n)
        t0 = time.time()
        bm.build(np.ones(n))
        t_build = time.time() - t0
        t0 = time.time()
        sys_.F(z)
        t_F = time.time() - t0
        t0 = time.time()
        C.jacobian_c8(sys_, z)
        t_J = time.time() - t0
        rec = {"block": bs, "seconds_init": t_init, "seconds_build": t_build, "seconds_F": t_F, "seconds_jacobian": t_J}
        res.append(rec)
        print(json.dumps(rec), flush=True)
    with open(os.path.join(a.out, "c8_cost.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
