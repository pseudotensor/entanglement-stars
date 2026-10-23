#!/usr/bin/env python
"""dust_ledger_selftest_v1b.py -- W7: the arithmetic of dust_ledger_postpass_v1b.py on manufactured arrays (engine-free; seconds).  v1b.

Version 1b (W6W7_selftest_fix_v1b.md). S1: version 1 manufactured 4 independent uniform random weight matrices, so the weighted norm changed by order 4
between its records (the 'max |resid| 4.05e+00' of its FAIL line), and asserted the residual equal to MINUS the weighted norm change; the post-pass's
residual is p_(j+1) - p_j + T_j - S_j = +f . (norm_(j+1) - norm_j) identically, the norm the propagation gains, which is right. Now the records are
site permutations of one weight matrix (norm-preserving to the summation roundoff) with the factor (1 + 1e-9) at record 2, and the expectations are:
T_j the direct fixed-cut difference and S_j the direct sweep (1e-12), the residual the weighted norm change with its sign (1e-12), zero at the step
without a manufactured drift (1e-12) and +-1e-9 times the weighted norm at the two steps touching record 2 (1e-12). S2: version 1 capped the certificate's
upper bound by the far weight p_late (in the post-pass and here); the inside-origin far content exceeds the total far content whenever the outside-origin
part interferes destructively, so that cap was false (the 'worst excess 4.4e-02' was true_in - p_late); the corrected upper bound is
min((sqrt p_late + sqrt eps_early)^2, 1 - eps_early); the lower bound, the edit's certificate, is rigorous and is checked as before; an exact 2-site
witness with true_in = 0.4 > p_late = 0.1 is added. S3 unchanged.

S1 the ledger: v2b's arrays (beyond_Rt, moving_cut_step, norm) built from manufactured per-site weights at 4 records with moving roots.
S2 the certificate: a packet split at a root, a random unitary, the far weight p_late: lower <= ||Pi U psi_in||^2 <= upper over 300 trials, the witness.
S3 the bins: the 5 half-open bins of a diagonal operator; the exterior split sums to the block's weight.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import dust_ledger_postpass_v1b as P  # noqa: E402


def main():
    rng = np.random.default_rng(7)
    fails = 0
    # S1: norm-preserving records (site permutations of one weight matrix), a manufactured norm factor at record 2
    n, nc, nr = 60, 3, 4
    R = 0.5 + np.arange(n)
    f = np.array([1.0, 0.5, 0.25])
    W0 = rng.random((n, nc))
    perms = [np.arange(n)] + [rng.permutation(n) for _ in range(nr - 1)]
    Wall = [W0[p] for p in perms]
    drift_factor = 1e-9
    Wall[2] = Wall[2] * (1.0 + drift_factor)
    roots = [20.3, 22.7, 21.1, 25.0]
    T = np.array([15.0, 15.125, 15.25, 15.375])
    rs = 64.0
    Z = {"k_T": T, "k_f": f, "k_norm": np.stack([w.sum(axis=0) for w in Wall]), "k_Rt_dense": np.array(roots),
         "k_beyond_Rt_dense": np.stack([Wall[j][R > roots[j]].sum(axis=0) for j in range(nr)]),
         "k_moving_cut_step_dense": np.array([np.nan] + [float(np.dot(f, Wall[j][R > roots[j]].sum(axis=0) - Wall[j][R > roots[j - 1]].sum(axis=0))) for j in range(1, nr)]),
         "k_current_outer_dense": np.zeros(nr)}
    L = P.ledger_arrays(Z, "k", "dense", rs)
    ok = True
    worst_T = worst_S = worst_D = 0.0
    for j in range(1, nr):
        T_direct = float(np.dot(f, Wall[j][R > roots[j - 1]].sum(axis=0) - Wall[j - 1][R > roots[j - 1]].sum(axis=0)))
        S_direct = -float(np.dot(f, Wall[j][R > roots[j]].sum(axis=0) - Wall[j][R > roots[j - 1]].sum(axis=0)))
        norm_change = float(np.dot(f, Wall[j].sum(axis=0) - Wall[j - 1].sum(axis=0)))
        worst_T = max(worst_T, abs(L["T_j"][j] - T_direct))
        worst_S = max(worst_S, abs(L["S_j"][j] - S_direct))
        worst_D = max(worst_D, abs(L["resid"][j] - norm_change))
        ok &= abs(L["dt"][j] - 8.0) < 1e-12
    ok &= worst_T < 1e-12 and worst_S < 1e-12 and worst_D < 1e-12
    wn1 = float(np.dot(f, Wall[1].sum(axis=0)))
    resid_undrifted = float(L["resid"][1])
    resid_gain = float(L["resid"][2])
    resid_loss = float(L["resid"][3])
    ok &= abs(resid_undrifted) < 1e-12 and abs(resid_gain - drift_factor * wn1) < 1e-12 and abs(resid_loss + drift_factor * wn1) < 1e-12
    print("%s S1 ledger: T_j equals the fixed-cut difference (max |diff| %.1e), S_j the sweep (%.1e), the residual the weighted norm change with its sign (%.1e); "
          "the residual at the undrifted step %.1e (0 expected), at the drifted step %+.3e against the manufactured +1e-9 x weighted norm %+.3e, at the following step %+.3e (minus that); dt %s" % (
              "PASS" if ok else "FAIL", worst_T, worst_S, worst_D, resid_undrifted, resid_gain, drift_factor * wn1, resid_loss, P.fmt(L["dt"][1])))
    fails += 0 if ok else 1
    # S2: the certificate's bounds against the exact inside-origin far content
    ok_lower = ok_upper = True
    d = 40
    worst_lower = -np.inf
    worst_upper = -np.inf
    n_destructive = 0
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
        upper = min((np.sqrt(p_late) + np.sqrt(eps_early)) ** 2, 1.0 - eps_early)
        ok_lower &= lower <= true_in + 1e-12
        ok_upper &= true_in <= upper + 1e-12
        worst_lower = max(worst_lower, lower - true_in)
        worst_upper = max(worst_upper, true_in - upper)
        if true_in > p_late + 1e-12:
            n_destructive += 1
    # the witness: site 0 inside the launch root, site 1 the far cut; the rotation by 3 pi/4 puts the inside amplitude into the far site with the sign
    # opposite to the outside amplitude's: the inside-origin far content 0.4 exceeds the total far content 0.1
    e0 = 0.2
    psi_in_w = np.array([np.sqrt(1.0 - e0), 0.0])
    psi_out_w = np.array([0.0, np.sqrt(e0)])
    th = 3.0 * np.pi / 4.0
    Uw = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    far_w = np.array([False, True])
    true_in_w = float(np.sum(np.abs((Uw @ psi_in_w)[far_w]) ** 2))
    p_late_w = float(np.sum(np.abs((Uw @ (psi_in_w + psi_out_w))[far_w]) ** 2))
    lower_w = max(np.sqrt(p_late_w) - np.sqrt(e0), 0.0) ** 2
    upper_w = min((np.sqrt(p_late_w) + np.sqrt(e0)) ** 2, 1.0 - e0)
    ok_w = abs(true_in_w - 0.4) < 1e-12 and abs(p_late_w - 0.1) < 1e-12 and lower_w <= true_in_w + 1e-12 and true_in_w <= upper_w + 1e-12 and true_in_w > p_late_w
    ok = ok_lower and ok_upper and ok_w
    print("%s S2 certificate: lower <= inside-origin far content <= upper over 300 random packets and unitaries (worst lower - exact %.1e, worst exact - upper %.1e, both at most 0; "
          "in %d trials the exact content exceeds the far weight p_late, where version 1's cap failed); witness: true_in %.4f, p_late %.4f, lower %.4f, upper %.4f (%s)" % (
              "PASS" if ok else "FAIL", worst_lower, worst_upper, n_destructive, true_in_w, p_late_w, lower_w, upper_w, ok_w))
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
