#!/usr/bin/env python
"""bench_b1_m1_v1.py (busi lane, B1 review MUST-FIX M1): the two unmeasured costs of the light channel on the target card.
(a) page_interior_v6s.propagate_lean on BandedLean: one record step of a random complex128 state n x n/2 (columns normalized) under the + sector
    h_+ of B1's fields, chunk --chunk, tol --tol, for every tau of --taus: seconds per step, the Chebyshev order K, and the flow's two full applies
    (engine_b1_light_v1.row_energy) per record; the device peak.
(b) page_interior_v6s._dense_init_plus: the T_0 vacuum Phi_+ by torch eigh on the device (and on the host with --host-init) for n in --init-n:
    seconds and the device peak (M6).
The fields (labelled synthetic, the B1 point's scales): N = 1, v = -sqrt(2 g M(R)/R) with M(R) = E0 (1 - exp(-(R/R0)^3)), R0 = 4 r_s, E0 = --E0,
g = --g (B1: E0 16.9, g 0.25, r_s 8.44); the bound w and so K are the production's to the percent (the exterior's v dominates nowhere).
Usage: python bench_b1_m1_v1.py --n 16896,33792 --taus 1.05,4.2 --init-n 16896,33792 [--device cuda:0] [--host-init 0]"""
import argparse
import json
import math
import os
import sys
import time

for _p in (os.environ.get("B1_PATHS") or "<internal>/code_v6").split(":"):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np        # noqa: E402
import torch              # noqa: E402
import page_ops_v6 as OPS            # noqa: E402
import page_w1_sector_v1d as W1S     # noqa: E402
import page_interior_v6s as IR       # noqa: E402
import engine_b1_light_v1 as B1L     # noqa: E402


def hp_of(n, E0, g, rs, kappa):
    R = np.arange(n, dtype=float) + 1.0
    M = E0 * (1.0 - np.exp(-(R / (4.0 * rs)) ** 3))
    N = np.ones(n); v = -np.sqrt(2.0 * g * M / R)
    h0, Q = OPS.chain_sparse(R, 0.0, float(kappa))
    return W1S.sector_plus_csr(OPS.clocked_csr(h0, Q, N, v)).tocsr()


def peak(dev):
    return torch.cuda.max_memory_allocated(dev) / 1e9 if dev.type == "cuda" else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="16896,33792")
    ap.add_argument("--taus", default="1.05,4.2")
    ap.add_argument("--init-n", default="16896")
    ap.add_argument("--host-init", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--tol", type=float, default=1e-14)
    ap.add_argument("--E0", type=float, default=16.9)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--rs", type=float, default=8.44)
    ap.add_argument("--kappa", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    rows = []
    for n in [int(x) for x in a.n.split(",") if x.strip()]:
        hp = hp_of(n, a.E0, a.g, a.rs, a.kappa)
        B = IR.BandedLean(hp, dev)
        m = n // 2
        torch.manual_seed(0)
        if dev.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
        X = torch.randn(n, m, dtype=torch.float64, device=dev).to(torch.complex128)
        X /= X.abs().pow(2).sum(dim=0).sqrt()[None, :]
        for tau in [float(x) for x in a.taus.split(",") if x.strip()]:
            B1L.sync(dev); t0 = time.time()
            X, K, tail = IR.propagate_lean(B, X, tau, a.tol, a.chunk)
            B1L.sync(dev); tp = time.time() - t0
            t0 = time.time()
            r0 = B1L.row_energy(B, X); r1 = B1L.row_energy(B, X)
            B1L.sync(dev); tf = time.time() - t0
            row = {"item": "propagate_lean", "n": n, "n_plus": m, "state_GB": n * m * 16 / 1e9, "tau": tau, "K": int(K), "tail": float(tail), "w": B.bound,
                   "seconds_step": tp, "seconds_flow_two_applies": tf, "seconds_per_record_per_channel": tp + tf, "device_peak_GB": peak(dev), "device": str(dev)}
            rows.append(row); print(json.dumps(row), flush=True)
        del X, B
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    for n in [int(x) for x in a.init_n.split(",") if x.strip()]:
        hp = hp_of(n, a.E0, a.g, a.rs, a.kappa)
        if dev.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
        try:
            Phi, info = IR._dense_init_plus(hp, dev, dev, a.threads, B1L.log)
            row = {"item": "init_device", "n": n, "seconds": info["seconds"], "n_neg_plus": info["n_neg_plus"], "device_peak_GB": peak(dev), "device": str(dev)}
            del Phi
        except Exception as ex:                                               # noqa: BLE001
            row = {"item": "init_device", "n": n, "failed": "%s: %s" % (type(ex).__name__, str(ex)[:300]), "device_peak_GB": peak(dev)}
        rows.append(row); print(json.dumps(row), flush=True)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        if a.host_init:
            Phi, info = IR._dense_init_plus(hp, torch.device("cpu"), torch.device("cpu"), a.threads, B1L.log)
            row = {"item": "init_host", "n": n, "seconds": info["seconds"], "n_neg_plus": info["n_neg_plus"]}
            del Phi
            rows.append(row); print(json.dumps(row), flush=True)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
