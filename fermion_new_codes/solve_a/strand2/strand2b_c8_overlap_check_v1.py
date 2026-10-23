#!/usr/bin/env python
"""Strand 2b C8 library v2 check (builder analyst): the common-sea Schur-complement overlap against version 1's determinant
route, on blocks (1,0) and (2,0), at a nonuniform psi (default_rng(5), psi = 1 + 0.05 u, u uniform on [-1, 1]) and psi' = psi +
h e_y for h in {1e-4, 1e-2} and y in {0, 7}, couplings (0.2, 1, Lambda 1) and m 0.1; plus the timing of one Jacobian of each
route's library.  Writes <out>/overlap_check.json."""
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
import strand2b_c8_lib_v2 as C2  # noqa: E402
import native_core as nc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lat = S2.lattice("C8")
    K, _ = nc.K_matrix(lat["positions"].tolist())
    rng = np.random.default_rng(5)
    psi = 1 + 0.05 * rng.uniform(-1, 1, 8)
    res = []
    for blk in ((1, 0), (2, 0)):
        bm = C2.BlockModel(lat, 0.1, 0.2, 1.0, 1.0, blk)
        bo = bm.build(psi)
        for h in (1e-4, 1e-2):
            for y in (0, 7):
                ps = psi.copy()
                ps[y] += h
                bn = bm.build(ps)
                t0 = time.time()
                Od = bm.overlap_det(bn["orb"], bo["orb"])
                td = time.time() - t0
                t0 = time.time()
                Of = bm.overlap(bn, bo)
                tf = time.time() - t0
                rec = {"block": "%d,%d" % blk, "h": h, "y": y, "max_abs_det": float(np.max(np.abs(Od))),
                       "max_abs_diff": float(np.max(np.abs(Of - Od))), "rel": float(np.max(np.abs(Of - Od)) / np.max(np.abs(Od))),
                       "seconds_det": td, "seconds_schur": tf}
                res.append(rec)
                print(json.dumps(rec), flush=True)
        sysv = C2.SystemC8(bm, K, 1.0)
        z = np.concatenate([psi, np.ones(8)])
        t0 = time.time()
        C2.jacobian_c8(sysv, z)
        rec = {"block": "%d,%d" % blk, "seconds_jacobian_v2": time.time() - t0}
        res.append(rec)
        print(json.dumps(rec), flush=True)
    with open(os.path.join(a.out, "overlap_check.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
