#!/usr/bin/env python
"""Strand 2b C8 diagnostic (builder analyst): the analytic Jacobian of the reduced map (strand2b_c8_lib_v4.jacobian_c8) against
central differences of F (the state re-solved at each z, h = 1e-6), block (2,0), m 0.1, (0.2, 1, Lambda 1), k 1, at three points:
the flat start with lambda = 0.05, a perturbed point (psi, N = 1 + 0.01 u, default_rng(2)) with lambda = 0.05 and with lambda = 1.
Reports per block of J (C rows / adjoint rows x psi / N columns) the largest absolute difference and the largest entry.
Also block (1,0) at the perturbed point, lambda = 1.  Writes <out>/jac_check.json."""
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
import strand2b_c8_lib_v4 as C  # noqa: E402
import native_core as nc  # noqa: E402


def check(blk, z, lam, K, lat, h=1e-6):
    bm = C.BlockModel(lat, 0.1, 0.2, 1.0, 1.0, blk)
    sys_ = C.SystemC8(bm, K, 1.0)
    n = sys_.n
    sys_.d0_fixed = sys_.state(np.ones(n), np.ones(n))["d0_at_tolerance"]
    t0 = time.time()
    Ja, st = C.jacobian_c8(sys_, z, lam)
    ta = time.time() - t0
    t0 = time.time()
    Jf = np.zeros_like(Ja)
    for j in range(2 * n):
        e = np.zeros(2 * n)
        e[j] = h
        Jf[:, j] = (sys_.F(z + e, lam)[0] - sys_.F(z - e, lam)[0]) / (2 * h)
    tf = time.time() - t0
    out = {"block": "%d,%d" % blk, "lam": lam, "d0": st["d0"], "gap": st["gap"], "seconds_analytic": ta, "seconds_fd": tf}
    for rn, rs in (("C_rows", slice(0, n)), ("adjoint_rows", slice(n, 2 * n))):
        for cn, cs in (("psi_cols", slice(0, n)), ("N_cols", slice(n, 2 * n))):
            d = Ja[rs, cs] - Jf[rs, cs]
            out["%s_%s" % (rn, cn)] = {"max_abs_diff": float(np.max(np.abs(d))), "max_abs_fd": float(np.max(np.abs(Jf[rs, cs])))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lat = S2.lattice("C8")
    K, _ = nc.K_matrix(lat["positions"].tolist())
    rng = np.random.default_rng(2)
    zflat = np.ones(16)
    zpert = 1 + 0.01 * rng.uniform(-1, 1, 16)
    res = []
    for blk, z, lam in (((2, 0), zflat, 0.05), ((2, 0), zpert, 0.05), ((2, 0), zpert, 1.0), ((1, 0), zpert, 1.0)):
        r = check(blk, z, lam, K, lat)
        res.append(r)
        print(json.dumps(r), flush=True)
    with open(os.path.join(a.out, "jac_check.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
