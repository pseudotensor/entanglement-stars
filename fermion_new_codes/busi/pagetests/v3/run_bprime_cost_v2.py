#!/usr/bin/env python
"""run_bprime_cost_v2.py (busi; B' cost inputs per ReviewB_pc13 sec. 1.8, throughput only). A separate script beside run_bprime_cost_v1.py,
whose window eigensolves at s = 16 F2 and above run long; this one measures only the operator side so the card-hours can be written now:
per size s = L/a in --sizes: the box (H = 2.5 L), amplitudes and bytes per column, the cube apply time per column at batch 1 and --batch on
CPU (torch, --threads), the Chebyshev order of one step dt (Cube.bound x dt), hence the CPU seconds per column per unit time a.
Fields: F2 (the frozen dust ball; cheap to build; the operator cost does not depend on the family). Box CPU only."""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pt3_core as K3  # noqa: E402
import pt3b_cube as CU  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="16,32,64")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--batch-max-size", type=float, default=32)
    ap.add_argument("--dt", type=float, default=4.0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    res = {"script": "run_bprime_cost_v2.py", "threads": a.threads, "dt": a.dt, "sizes": {}, "own_md5": K3.own_md5s()}
    import page_ops_v6 as OPS
    for s in [float(x) for x in a.sizes.split(",")]:
        L = s; H = int(round(2.5 * L))
        t0 = time.time()
        N, v, r, _ = CU.cube_fields(CU.f2_profile(), L, H, 0.0, 1.0 / 8)
        cube = CU.Cube(N, v, CU.cap_reviewa(r, L))
        K, tail = OPS.cheb_order(cube.bound * a.dt, 1e-15)
        row = {"H": H, "amplitudes": 4 * (2 * H + 1) ** 3, "bytes_per_column": 16 * 4 * (2 * H + 1) ** 3, "fields_seconds": time.time() - t0,
               "cheb_bound": cube.bound, "cheb_order_per_step": int(K) + 1}
        for b in ([1, a.batch] if s <= a.batch_max_size else [1]):
            X = torch.randn((b, 4) + N.shape, dtype=CU.CD)
            cube.apply(X)
            t1 = time.time(); nrep = 3
            for _ in range(nrep):
                cube.apply(X)
            row["apply_seconds_batch%d" % b] = (time.time() - t1) / nrep
            row["apply_seconds_per_column_batch%d" % b] = row["apply_seconds_batch%d" % b] / b
            del X
        best = min(v_ for k_, v_ in row.items() if k_.startswith("apply_seconds_per_column"))
        row["cpu_seconds_per_column_per_a"] = best * row["cheb_order_per_step"] / a.dt
        K3.log("s %g: H %d, %.3f GB/column, Chebyshev %d per dt %g, apply/column %.3f s, CPU s per column per a %.3f" % (
            s, H, row["bytes_per_column"] / 1e9, row["cheb_order_per_step"], a.dt, best, row["cpu_seconds_per_column_per_a"]))
        res["sizes"][str(int(s))] = row
        K3.dump_json(res, os.path.join(a.out, "bprime_cost_v2.json"))
        del cube, N, v, r


if __name__ == "__main__":
    main()
