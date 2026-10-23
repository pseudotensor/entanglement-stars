#!/usr/bin/env python
"""bench_cube_apply_v1.py (busi; the B' short H100 benchmark, part 1: ReviewB_bpchk_v1 sec. 3 "measure before allocating": the card model and
memory, dtype complex128, the production Cube.apply and one Chebyshev-Strang step at batch sizes --batches on the prep's fields, with
synchronized timing (3 repeats after a warm-up) and the peak allocated memory per batch). Writes <out>/bench_apply.json."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pt3_core  # noqa: E402,F401  (puts code_v6/page_ops_v6 on the path for Cube.expm)
import pt3b_cube as CU  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batches", default="1,8,32,120")
    ap.add_argument("--dt", type=float, default=4.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    Z = np.load(a.prep)
    N = Z["N"]; v = [Z["vx"], Z["vy"], Z["vz"]]; eta = Z["eta"]
    cube = CU.Cube(N, v, eta, a.device)
    res = {"device": str(dev), "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
           "total_mem_GB": torch.cuda.get_device_properties(dev).total_memory / 1e9 if dev.type == "cuda" else None,
           "torch": torch.__version__, "box_L": int(N.shape[0]), "amplitudes_per_column": int(4 * N.size), "dtype": "complex128",
           "cheb_bound": cube.bound, "dt": a.dt, "rows": []}
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    for b in [int(x) for x in a.batches.split(",")]:
        try:
            if dev.type == "cuda":
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
            X = torch.randn((b, 4) + N.shape, dtype=CU.CD, device=dev)
            cube.apply(X); sync()
            t0 = time.time()
            for _ in range(3):
                Y = cube.apply(X)
            sync(); ta = (time.time() - t0) / 3
            del Y
            cube.step_cap(X, a.dt); sync()
            t0 = time.time()
            for _ in range(3):
                Y = cube.step_cap(X, a.dt)
            sync(); ts = (time.time() - t0) / 3
            row = {"batch": b, "apply_s": ta, "apply_ns_per_amplitude": ta / (b * 4 * N.size) * 1e9, "step_s": ts,
                   "step_s_per_column_per_a": ts / (b * a.dt),
                   "peak_mem_GB": torch.cuda.max_memory_allocated(dev) / 1e9 if dev.type == "cuda" else None}
            del X, Y
        except RuntimeError as ex:
            row = {"batch": b, "error": repr(ex)[:300]}
        res["rows"].append(row)
        print(json.dumps(row), flush=True)
        json.dump(res, open(os.path.join(a.out, "bench_apply.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
