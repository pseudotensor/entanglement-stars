#!/usr/bin/env python
"""j6_timing_cheb_v1.py (builder analyst): the cost of J6's formation ramp by a Chebyshev propagation of the occupied block only (the banded
chain Hamiltonian as a sparse matrix; exp(-i h dt) X by a Chebyshev series in the rescaled h, terms until the Bessel coefficient falls below
1e-14), against flow_chart_v4's dense step (eigh plus two products); the accuracy check: one channel's block after --steps steps by both
routes (the dense route's product of exact midpoint exponentials). One thread. Args: --mc --nsites --steps --dt."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.special import jv

sys.path.insert(0, "<internal>/code")
import flow_chart_v4 as F  # noqa: E402


def cheb_step(Hs, X, dt, c, r):
    """exp(-i H dt) X with H = c + r Ht, ||Ht|| <= 1: e^{-i c dt} sum_k (2 - delta_k0) (-i)^k J_k(r dt) T_k(Ht) X."""
    z = r * dt
    K = int(z + 10)
    while abs(jv(K, z)) > 1e-14:
        K += 1
    Ht = lambda Y: (Hs @ Y - c * Y) / r
    T0 = X; T1 = Ht(X)
    out = jv(0, z) * T0 + 2 * (-1j) * jv(1, z) * T1
    for k in range(2, K + 1):
        T2 = 2 * Ht(T1) - T0
        out = out + 2 * ((-1j) ** k) * jv(k, z) * T2
        T0, T1 = T1, T2
    return np.exp(-1j * c * dt) * out, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=float, default=0.3)
    ap.add_argument("--nsites", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--ramp-time", type=float, default=2560.0)
    a = ap.parse_args()
    N = a.nsites
    mz, yy, x = F.planar_fields(N, a.mc)
    vfinal = F.tanh_profile(x, 0.4, 0.4 * 64)
    V0, occ0 = F.flat_sea(F.chain_hamiltonian(mz, yy, np.zeros(N)))
    X = V0.astype(complex)
    Xd = X.copy()
    tc = td = 0.0
    Ks = []
    for i in range(a.steps):
        s = (i + 0.5) * a.dt / a.ramp_time
        Hd = F.chain_hamiltonian(mz, yy, np.sin(np.pi * s / 2) ** 2 * vfinal)
        Hs = sp.csr_matrix(Hd)
        t = time.time()
        bound = float(abs(Hs).sum(axis=1).max())
        X, K = cheb_step(Hs, X, a.dt, 0.0, bound)
        tc += time.time() - t; Ks.append(K)
        t = time.time()
        e, Q = F.spectrum(Hd)
        Xd = (Q * np.exp(-1j * a.dt * e)) @ (Q.conj().T @ Xd)
        td += time.time() - t
    err = float(np.max(np.abs(X - Xd)))
    nnz = Hs.nnz
    print("mc %g, %d sites, block %d columns, nnz %d: Chebyshev %.3f s per step (terms %s), dense %.3f s per step; max |difference| after %d steps %.2e; "
          "a full ramp of %d steps: Chebyshev %.2f h, dense %.2f h (one thread)" % (a.mc, N, X.shape[1], nnz, tc / a.steps, Ks, td / a.steps, a.steps, err,
                                                                                     int(np.ceil(a.ramp_time / a.dt)), tc / a.steps * a.ramp_time / a.dt / 3600,
                                                                                     td / a.steps * a.ramp_time / a.dt / 3600))


if __name__ == "__main__":
    main()
