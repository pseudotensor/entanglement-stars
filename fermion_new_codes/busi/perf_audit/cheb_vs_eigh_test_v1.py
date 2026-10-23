#!/usr/bin/env python
"""cheb_vs_eigh_test_v1.py (busi perf audit, #359): the dg step's propagation exp(-i tau_s H-bar) X (engine_collapse_v36p dgk_substep_v36k:
10 eigendecompositions per iterate, one per channel, used only for this propagation) against the Chebyshev series on the banded H-bar that
page_history_v6 uses (page_ops_v6.cheb_order at tol 1e-16; the Banded diagonal apply): the difference of the propagated blocks, the
unitarity defects, the Chebyshev order K and the walls on the box GPU.  A tolerance-class candidate (W2 review sec. 3.3), quantified here.
Box only, nice 10, one free GPU."""
import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "<internal>/engine")
sys.path.insert(0, "<internal>/code_v6")
import engine_vacflow_v1 as VF          # noqa: E402
import page_ops_v6 as PO6               # noqa: E402
from scipy.special import jv            # noqa: E402


def dense_dev(A, dev):
    co = A.tocoo()
    M = torch.zeros(co.shape, dtype=torch.complex128, device=dev)
    M.index_put_((torch.as_tensor(co.row, dtype=torch.int64, device=dev), torch.as_tensor(co.col, dtype=torch.int64, device=dev)),
                 torch.as_tensor(np.asarray(co.data, complex), dtype=torch.complex128, device=dev), accumulate=True)
    return M


def cheb_apply(Bh, X, tau, tol):
    """page_history_v6o.cheb_apply verbatim (exp(-i tau h) X on [-w, w])."""
    w = Bh.bound
    z = w * tau
    K, tail = PO6.cheb_order(z, tol)
    k = np.arange(K + 1)
    ck = jv(k, z) * ((-1j) ** k) * np.where(k == 0, 1.0, 2.0)
    T0 = X
    T1 = Bh.apply(X) / w
    Y = complex(ck[0]) * T0 + complex(ck[1]) * T1
    for kk in range(2, K + 1):
        T2 = 2.0 * Bh.apply(T1) / w - T0
        Y = Y + complex(ck[kk]) * T2
        T0, T1 = T1, T2
    return Y, K, tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True)
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--kappas", default="1,5,10")
    ap.add_argument("--taus", default="8,4,2")
    ap.add_argument("--dust", type=int, default=200)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-16)
    a = ap.parse_args()
    dev = torch.device("cuda:%d" % a.gpu)
    torch.set_num_threads(8)
    z = np.load(a.fields, allow_pickle=True)
    N = np.asarray(z["N"][-1], float); v = np.asarray(z["v"][-1], float); N0 = np.asarray(z["N"][-2], float); v0 = np.asarray(z["v"][-2], float)
    n = len(N); R = np.arange(n, dtype=float) + 1.0
    print("device %s; nsites %d; m %g; tol %g" % (torch.cuda.get_device_name(dev), n, a.m, a.tol), flush=True)
    for kappa in [int(x) for x in a.kappas.split(",")]:
        h0, Q = VF.chain_sparse(R, a.m, float(kappa))
        hprev = PO6.clocked_csr(h0, Q, N0, v0); hbar = PO6.clocked_csr(h0, Q, N, v)
        # the state: the previous record's sea (negative half) plus random dust columns, as the dg step propagates them
        e0, U0 = torch.linalg.eigh(dense_dev(hprev, dev))
        sea = U0[:, e0 < 0]
        g = torch.Generator(device="cpu").manual_seed(7)
        dust = torch.randn((2 * n, a.dust), dtype=torch.complex128, generator=g).to(dev)
        dust, _ = torch.linalg.qr(dust)
        X = torch.cat([sea, dust], dim=1)
        M = dense_dev(hbar, dev)
        Bh = PO6.Banded(hbar, dev)
        for tau in [float(x) for x in a.taus.split(",")]:
            # route 1: the engine's (eigh + two GEMMs)
            torch.cuda.synchronize(); t0 = time.time()
            e, U = torch.linalg.eigh(M)
            ph = torch.exp(-1j * e.to(torch.complex128) * tau)
            Y1 = U @ (ph[:, None] * (U.conj().T @ X))
            torch.cuda.synchronize(); t1 = time.time() - t0
            # route 2: Chebyshev on the banded operator (the Page engine's propagation)
            torch.cuda.synchronize(); t0 = time.time()
            Y2, K, tail = cheb_apply(Bh, X, tau, a.tol)
            torch.cuda.synchronize(); t2 = time.time() - t0
            d = (Y1 - Y2).abs().max().item()
            dn = torch.linalg.norm(Y1 - Y2).item() / torch.linalg.norm(Y1).item()
            u1 = (Y1.conj().T @ Y1 - torch.eye(X.shape[1], dtype=X.dtype, device=dev)).abs().max().item()
            u2 = (Y2.conj().T @ Y2 - torch.eye(X.shape[1], dtype=X.dtype, device=dev)).abs().max().item()
            # the energy of the propagated sea under hbar is conserved by both routes: a scalar check
            E1 = (Y1[:, :sea.shape[1]].conj() * (M @ Y1[:, :sea.shape[1]])).sum().real.item(); E2 = (Y2[:, :sea.shape[1]].conj() * (M @ Y2[:, :sea.shape[1]])).sum().real.item()
            print("kappa %2d tau %4.1f: eigh route %.3f s (eigh %d), Chebyshev K %d tail %.1e %.3f s (x%.1f); max|Y1-Y2| %.2e rel %.2e; unitarity defect eigh %.1e cheb %.1e; sea energy diff %.2e (of %.6e)" % (
                kappa, tau, t1, 2 * n, K, tail, t2, t1 / t2, d, dn, u1, u2, E1 - E2, E1), flush=True)
        del M, Bh, U0, sea, dust, X
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
