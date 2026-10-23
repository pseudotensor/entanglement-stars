#!/usr/bin/env python
"""Strand 7 S7b cost benchmark (builder analyst; for the lead's sizing of SPEC_strand7_collapse.md's S7b on the box's GPUs): float64 timings
of torch.linalg.eigh on random symmetric matrices of the sizes S7b needs at the primary setting (n = 2000 radial nodes per channel at
R_max = 125 r_s, h = 1: the full chiral h is 4n x 4n; with m = 0 it splits exactly into two 2n x 2n sectors, and E_0 = (A'^T A')^(1/2) into
two n x n blocks) and of the propagation matmul (2n x 2n real times 2n x n complex per sector); then the cost model per channel: every
refresh (K steps) two sector eigendecompositions, two E_0 blocks and the propagation, i.e. 2560/K refreshes to T = 40 r_s at delta_t = 0.25 h
(r_s = 16 h), with the K steps of one refresh collapsed into one exact exponential of the held h (V e^(-i Lambda K delta_t) V^T)."""
import argparse
import time

import torch


def tm(f, reps):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        f()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.manual_seed(0)
    n = a.n
    out = {}
    for size, reps in ((n, 2), (2 * n, 2), (4 * n, 1)):
        X = torch.randn(size, size, dtype=torch.float64, device=dev)
        S = X + X.T
        torch.linalg.eigh(S)
        out["eigh_%d" % size] = tm(lambda: torch.linalg.eigh(S), reps)
        print("eigh %d: %.2f s" % (size, out["eigh_%d" % size]), flush=True)
    V = torch.randn(2 * n, 2 * n, dtype=torch.float64, device=dev)
    P = torch.randn(2 * n, n, dtype=torch.complex128, device=dev)
    Vc = V.to(torch.complex128)
    out["prop"] = tm(lambda: Vc @ (Vc.T @ P), 3)
    print("propagation (2n x 2n by 2n x n, complex, two products): %.3f s" % out["prop"], flush=True)
    for K in (1, 10, 100):
        nref = 2560 // K
        per = 2 * out["eigh_%d" % (2 * n)] + 2 * out["eigh_%d" % n] + 2 * out["prop"]
        print("K = %d: %d refreshes per channel to T = 40 r_s; %.1f s per refresh; %.2f h per channel; %.2f h for three channels" % (
            K, nref, per, nref * per / 3600, 3 * nref * per / 3600), flush=True)
    print("done")


if __name__ == "__main__":
    main()
