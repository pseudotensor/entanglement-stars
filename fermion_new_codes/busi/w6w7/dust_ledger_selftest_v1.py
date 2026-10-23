#!/usr/bin/env python
"""dust_ledger_selftest_v1.py -- W7: the arithmetic of dust_ledger_postpass_v1.py on manufactured arrays (engine-free; seconds).

S1 the ledger: v2b's arrays (beyond_Rt, moving_cut_step, norm) are built from manufactured per-site weights at 4 records with moving roots;
   the post-pass's T_j must equal the direct fixed-cut difference and p_(j+1) - p_j + T_j - S_j must equal minus the norm drift, to roundoff.
S2 the certificate: a packet split at a root, a random unitary, the far weight p_late: the lower bound [sqrt p - sqrt eps]_+^2 never exceeds the
   true inside-origin far content ||Pi U psi_in||^2 and the upper bound never falls below it, over 300 trials.
S3 the bins: the 5 half-open bins of a diagonal operator; the exterior split sums to the block's weight.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import dust_ledger_postpass_v1 as P  # noqa: E402


def main():
    rng = np.random.default_rng(7)
    fails = 0
    # S1
    n, nc, nr = 60, 3, 4
    R = 0.5 + np.arange(n)
    f = np.array([1.0, 0.5, 0.25])
    Wall = [rng.random((n, nc)) for _ in range(nr)]
    Wall[2] = Wall[2] * (1 + 1e-9)
    roots = [20.3, 22.7, 21.1, 25.0]
    T = np.array([15.0, 15.125, 15.25, 15.375])
    rs = 64.0
    Z = {"k_T": T, "k_f": f, "k_norm": np.stack([w.sum(axis=0) for w in Wall]), "k_Rt_dense": np.array(roots),
         "k_beyond_Rt_dense": np.stack([Wall[j][R > roots[j]].sum(axis=0) for j in range(nr)]),
         "k_moving_cut_step_dense": np.array([np.nan] + [float(np.dot(f, Wall[j][R > roots[j]].sum(axis=0) - Wall[j][R > roots[j - 1]].sum(axis=0))) for j in range(1, nr)]),
         "k_current_outer_dense": np.zeros(nr)}
    L = P.ledger_arrays(Z, "k", "dense", rs)
    ok = True
    for j in range(1, nr):
        T_direct = float(np.dot(f, Wall[j][R > roots[j - 1]].sum(axis=0) - Wall[j - 1][R > roots[j - 1]].sum(axis=0)))
        S_direct = -float(np.dot(f, Wall[j][R > roots[j]].sum(axis=0) - Wall[j][R > roots[j - 1]].sum(axis=0)))
        drift = -float(np.dot(f, Wall[j].sum(axis=0) - Wall[j - 1].sum(axis=0)))
        ok &= abs(L["T_j"][j] - T_direct) < 1e-12 and abs(L["S_j"][j] - S_direct) < 1e-12 and abs(L["resid"][j] - drift) < 1e-12 and abs(L["dt"][j] - 8.0) < 1e-12
    print("%s S1 ledger: T_j equals the fixed-cut difference, S_j the sweep, the residual the norm drift (max |resid| %.2e, the manufactured drift at record 2 1e-9 x norm)" % ("PASS" if ok else "FAIL", np.nanmax(np.abs(L["resid"]))))
    fails += 0 if ok else 1
    # S2
    ok = True
    d = 40
    worst = 0.0
    for _ in range(300):
        psi = rng.normal(size=d) + 1j * rng.normal(size=d)
        psi /= np.linalg.norm(psi)
        inside = np.arange(d) < rng.integers(5, 30)
        psi_in = psi * inside
        psi_out = psi * (~inside)
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))
        far = np.arange(d) >= rng.integers(20, 38)
        p_late = float(np.sum(np.abs((Q @ psi)[far]) ** 2))
        eps_early = float(np.sum(np.abs(psi_out) ** 2))
        true_in = float(np.sum(np.abs((Q @ psi_in)[far]) ** 2))
        lower = max(np.sqrt(p_late) - np.sqrt(eps_early), 0.0) ** 2
        upper = min((np.sqrt(p_late) + np.sqrt(eps_early)) ** 2, 1.0 - eps_early, p_late)
        ok &= lower <= true_in + 1e-12 and true_in <= upper + 1e-12
        worst = max(worst, lower - true_in, true_in - upper)
    print("%s S2 certificate: lower <= inside-origin far content <= upper over 300 random packets and unitaries (worst excess %.1e)" % ("PASS" if ok else "FAIL", worst))
    fails += 0 if ok else 1
    # S3
    m, b = 0.2, 1.0198
    e = np.array([-2.0, -b, -0.5, -m, 0.0, m, 0.5, b, 1.5])
    idx = P.bin_index(e, m, b)
    ok = list(idx) == [0, 0, 1, 1, 2, 2, 3, 3, 4]
    U = np.eye(e.size, dtype=complex)
    S = rng.normal(size=(e.size, 2)) + 1j * rng.normal(size=(e.size, 2))
    w = P.spectral_weights(U, S)
    ok &= np.allclose(w.sum(axis=0), np.sum(np.abs(S) ** 2, axis=0))
    print("%s S3 bins: the half-open bins (lo, hi] of horizon3_bins_v1 reproduced on the edges; the spectral weights sum to the block's weight" % ("PASS" if ok else "FAIL"))
    fails += 0 if ok else 1
    print("%d checks, %d failed" % (3, fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
