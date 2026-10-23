#!/usr/bin/env python
"""eigh_backend_test_v1.py (busi perf audit, #359): torch.linalg.eigh on the device under the cusolver and the magma backends at the
production matrix sizes (2n = 4096 for 2048 sites, 8192 for 4096 sites), complex128 Hermitian with the clocked chain's structure
(D + iA, D real diagonal, A real antisymmetric banded) and a dense random Hermitian for reference: wall per call and the backends'
eigenvalue agreement (the tolerance a backend change would need).  Box only, nice 10, one free GPU."""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "<internal>/engine")
sys.path.insert(0, "<internal>/code_v6")
import engine_vacflow_v1 as VF          # noqa: E402
import page_ops_v6 as PO6               # noqa: E402


def dense_dev(A, dev):
    co = A.tocoo()
    M = torch.zeros(co.shape, dtype=torch.complex128, device=dev)
    M.index_put_((torch.as_tensor(co.row, dtype=torch.int64, device=dev), torch.as_tensor(co.col, dtype=torch.int64, device=dev)),
                 torch.as_tensor(np.asarray(co.data, complex), dtype=torch.complex128, device=dev), accumulate=True)
    return M


def timed(M, reps):
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.time()
        e, U = torch.linalg.eigh(M)
        torch.cuda.synchronize(); ts.append(time.time() - t0)
    return e, U, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True)
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--kappa", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--sizes", default="2048,4096")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    dev = torch.device("cuda:%d" % a.gpu)
    torch.set_num_threads(8)
    z = np.load(a.fields, allow_pickle=True)
    N = np.asarray(z["N"][-1], float); v = np.asarray(z["v"][-1], float)
    print("device", torch.cuda.get_device_name(dev), "torch", torch.__version__, flush=True)
    for nsites in [int(x) for x in a.sizes.split(",")]:
        R = np.arange(nsites, dtype=float) + 1.0
        Nn = np.interp(R, np.arange(len(N)) + 1.0, N) if nsites != len(N) else N
        vn = np.interp(R, np.arange(len(v)) + 1.0, v) if nsites != len(v) else v
        h0, Q = VF.chain_sparse(R, a.m, float(a.kappa))
        M = dense_dev(PO6.clocked_csr(h0, Q, Nn, vn), dev)
        res = {}
        for backend in ("cusolver", "magma"):
            try:
                torch.backends.cuda.preferred_linalg_library(backend)
                e, U, ts = timed(M, a.reps)
                r = (M @ U - U * e[None, :].to(U.dtype)).abs().max().item()
                res[backend] = (e.cpu().numpy(), ts, r)
                print("n %d (2n %d) %-8s eigh: %s s; residual ||M U - U e|| %.2e" % (nsites, 2 * nsites, backend, " ".join("%.3f" % t for t in ts), r), flush=True)
            except Exception as ex:
                print("n %d %s FAILED: %s" % (nsites, backend, str(ex)[:200]), flush=True)
        if "cusolver" in res and "magma" in res:
            de = np.max(np.abs(res["cusolver"][0] - res["magma"][0]))
            print("n %d: max |e_cusolver - e_magma| = %.3e (relative to max|e| %.3e: %.2e); median wall cusolver %.3f magma %.3f -> magma/cusolver %.2f" % (
                nsites, de, np.max(np.abs(res["cusolver"][0])), de / np.max(np.abs(res["cusolver"][0])), np.median(res["cusolver"][1]), np.median(res["magma"][1]),
                np.median(res["magma"][1]) / np.median(res["cusolver"][1])), flush=True)
        torch.backends.cuda.preferred_linalg_library("default")
        del M
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
