#!/usr/bin/env python
"""Strand 2c addendum, part A (spec solve/specs/SPEC_strand2c_addendum.md; builder analyst): the vacuum kernel in momentum space.

W_0 = sum_s eta_s Tr L (E_s1 - E_s0), E_s0 = (A_s^dag A_s)^(1/2), E_s1 = (A_s A_s^dag)^(1/2), A_s = m_s Psi^-1 + i Psi^-3 D (2c's
packets; D = sum_j sigma_j sin k_j on the periodic box with twist theta_j, k_j = (2 pi n_j + theta_j)/L), L = N Psi.  For
psi = 1 + eps c, c = cos(q x_1), N = 1 + eps_N c: A = A0 + eps A1 + eps^2 A2 with A0 = m + i D, A1 = -c (m + 3 i D) (c acts after D),
A2 = c^2 (m + 6 i D).  X = A^dag A = X0 + eps X' + eps^2 X''/2 with X0(k) = (m^2 + s_k^2) 1, s_k^2 = sum_j sin^2 k_j, and, with
c_kk' = (1/2)(delta_{k', k+q} + delta_{k', k-q}) (combined when 2q = 0 mod 2 pi),
  X'_kk'  = c_kk' [-2 m^2 - 6 d_k.d_k' + i (4 m (d_k - d_k') - 6 d_k x d_k').sigma]                 (A^dag A)
  X~'_kk' = c_kk' [-2 m^2 - 3 (s_k^2 + s_k'^2) + 2 i m (d_k - d_k').sigma]                           (A A^dag)
  X''_kk  = 2 [(sum_k' c_kk'^2)(m^2 + 9 s_k^2) + (c^2)_kk (2 m^2 + 12 s_k^2)]                       (A^dag A)
  X~''_kk = 2 [sum_k' c_kk'^2 (m^2 + 9 s_k'^2) + (c^2)_kk (2 m^2 + 12 s_k^2)]                        (A A^dag)
(the eps^2 terms of A0^dag A2 + A2^dag A0 reduce on the diagonal to (c^2)_kk (2 m^2 + 12 s_k^2) since the m D cross terms cancel).
Second order by the Daleckii-Krein formula with F = sqrt: for T(eps) = Tr(Psi F(X)),
  d^2 T/d eps^2 = 2 sum_kk' F1(x_k, x_k') tr[c_k'k X'_kk'] + sum_k F'(x_k) tr X''_kk + sum_kk' F'1(x_k, x_k') ||X'_kk'||_F^2,
F1(a, b) = 1/(sqrt a + sqrt b), F'(a) = 1/(2 sqrt a), F'1(a, b) = -1/(2 sqrt a sqrt b (sqrt a + sqrt b)); the same for U with X~.
W^(2) = (1/2) sum_s eta_s (U_s'' - T_s''); the mixed eps eps_N term W^(11) = sum_s eta_s sum_kk' F1 tr[c_k'k (X~' - X')_kk'].
k_a(q) = -2 W^(2)/(q_lat^2 L^3), mixed = -W^(11)/(q_lat^2 L^3), q_lat^2 = 2 (1 - cos q), k_lat = 3 J/pi^2, J = sum_s eta_s m_s^2 log m_s^2.
Everything depends on (k_1, s_perp^2 = sin^2 k_2 + sin^2 k_3) only (the cross products reduce to |d_perp|^2), so the sum runs over k_1
and the distinct s_perp^2 values of the (k_2, k_3) grid with their counts.
Control (--control): the dense real-space W_0 of 2c (strand2c_mismatch_v2's packet routine) at eps in {2e-2, 1e-2, 5e-3}, g(eps) =
W_0/eps^2 extrapolated twice (R1(e) = (4 g(e/2) - g(e))/3, R2 = (16 R1(e/2) - R1(e))/15, removing the eps^2 and eps^4 terms; the
dense W_0 carries a rounding floor near 1e-13, so smaller eps would not help), against W^(2), at L in {8, 12}, H in {1, 0.5, 0.25},
q = 2 pi/L, twist 0, m = 0.1; also the 2c run's value at eps = 1e-3 (k_a of RESULT_strand2c T1) for reference.
Writes <out>/kernel_<tag>.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def grid(L, theta):
    k = (2 * np.pi * np.arange(L) + theta) / L
    return k


def perp_classes(L, th2, th3):
    s2 = np.sin(grid(L, th2)) ** 2
    s3 = np.sin(grid(L, th3)) ** 2
    sp = (s2[:, None] + s3[None, :]).ravel()
    key = np.round(sp, 13)
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    vals = np.bincount(inv, weights=sp) / cnt
    return vals, cnt.astype(float)


def packet_terms(m, L, n_q, theta, sp, cnt):
    """(T'', U'', W11 contribution) for one packet of mass m, shift n_q along axis 1."""
    k1 = grid(L, theta[0])
    s1 = np.sin(k1)
    shifts = [n_q % L, (-n_q) % L]
    if shifts[0] == shifts[1]:
        nb = [(shifts[0], 1.0)]
    else:
        nb = [(shifts[0], 0.5), (shifts[1], 0.5)]
    sum_c2 = sum(w * w for _, w in nb)                     # sum_k' c_kk'^2
    c2diag = 1.0 if (2 * n_q) % L == 0 else 0.5            # (c^2)_kk
    S1 = s1[:, None]                                       # (L, 1)
    SP = sp[None, :]                                       # (1, P)
    W = cnt[None, :]
    x = m * m + S1 ** 2 + SP                               # x_k
    sq = np.sqrt(x)
    Fp = 0.5 / sq
    T2 = 0.0
    U2 = 0.0
    W11 = 0.0
    # diagonal X'' terms (trace over the 2 spinor components: factor 2)
    Xpp = 2 * (sum_c2 * (m * m + 9 * (S1 ** 2 + SP)) + c2diag * (2 * m * m + 12 * (S1 ** 2 + SP)))
    T2 += float(np.sum(W * Fp * 2 * Xpp))
    Utmp = np.zeros_like(x)
    for sh, w in nb:
        s1p = np.roll(s1, -sh)[:, None]                    # sin k1' with k' = k + shift
        Utmp = Utmp + w * w * (m * m + 9 * (s1p ** 2 + SP))
    Xtpp = 2 * (Utmp + c2diag * (2 * m * m + 12 * (S1 ** 2 + SP)))
    U2 += float(np.sum(W * Fp * 2 * Xtpp))
    for sh, w in nb:
        s1p = np.roll(s1, -sh)[:, None]
        xp = m * m + s1p ** 2 + SP
        sqp = np.sqrt(xp)
        F1 = 1.0 / (sq + sqp)
        Fp1 = -1.0 / (2 * sq * sqp * (sq + sqp))
        dd = S1 * s1p + SP                                  # d_k . d_k'
        ds = S1 - s1p                                       # (d_k - d_k') along axis 1 only
        # X' = w [a0 + i a.sigma]: a0 = -2 m^2 - 6 dd; a = 4 m (d_k - d_k') - 6 d_k x d_k', |a|^2 = 16 m^2 ds^2 + 36 ds^2 SP
        a0 = -2 * m * m - 6 * dd
        a2 = 16 * m * m * ds ** 2 + 36 * ds ** 2 * SP
        trX = 2 * w * a0
        frob = 2 * w * w * (a0 ** 2 + a2)
        T2 += float(np.sum(W * (2 * F1 * w * trX + Fp1 * frob)))
        # X~' = w [b0 + i b.sigma]: b0 = -2 m^2 - 3 (s_k^2 + s_k'^2), b = 2 m (d_k - d_k')
        b0 = -2 * m * m - 3 * (S1 ** 2 + s1p ** 2 + 2 * SP)
        b2 = 4 * m * m * ds ** 2
        trXt = 2 * w * b0
        frobt = 2 * w * w * (b0 ** 2 + b2)
        U2 += float(np.sum(W * (2 * F1 * w * trXt + Fp1 * frobt)))
        W11 += float(np.sum(W * F1 * w * (trXt - trX)))
    return T2, U2, W11


def kernel(L, masses, etas, q_n, theta=(0.0, 0.0, 0.0)):
    sp, cnt = perp_classes(L, theta[1], theta[2])
    T, U, M = 0.0, 0.0, 0.0
    for m, e in zip(masses, etas):
        t2, u2, w11 = packet_terms(m, L, q_n, theta, sp, cnt)
        T += e * t2
        U += e * u2
        M += e * w11
    W2 = 0.5 * (U - T)
    q = 2 * np.pi * q_n / L
    ql2 = 2 * (1 - np.cos(q))
    return {"W2": W2, "W11": M, "k_a": -2 * W2 / (ql2 * L ** 3), "mixed": -M / (ql2 * L ** 3), "q": q, "q_lat2": ql2}


def masses_eq(m, H):
    return [m, np.sqrt(m * m + H * H), np.sqrt(m * m + H * H), np.sqrt(m * m + 2 * H * H)], [1, -1, -1, 1]


def Jsum(ms, etas):
    return float(sum(e * (mm * mm) * np.log(mm * mm) for mm, e in zip(ms, etas) if mm > 0))


def control(out):
    import strand2c_mismatch_v2 as M2
    rows = []
    for L in (8, 12):
        D = M2.weyl(L)
        X = M2.coords(L)
        q = 2 * np.pi / L
        for H in (1.0, 0.5, 0.25):
            ms, etas = masses_eq(0.1, H)
            kr = kernel(L, ms, etas, 1)
            g = {}
            for eps in (2e-2, 1e-2, 5e-3, 1e-3):
                psi = 1 + eps * np.cos(q * X[:, 0])
                wx, W0, tr, _ = M2.w_field(D, psi, np.ones(L ** 3), H, -1, {}, "c%g" % eps)
                g[eps] = W0 / eps ** 2
            R1a = (4 * g[1e-2] - g[2e-2]) / 3
            R1b = (4 * g[5e-3] - g[1e-2]) / 3
            R2 = (16 * R1b - R1a) / 15
            rows.append({"L": L, "H": H, "W2_kernel": kr["W2"], "W2_dense_eps1e-3": g[1e-3], "W2_dense_R1": R1b, "W2_dense_R2": R2,
                         "rel_diff_R2": abs(kr["W2"] - R2) / abs(R2), "rel_diff_R1": abs(kr["W2"] - R1b) / abs(R1b),
                         "rel_diff_eps1e-3": abs(kr["W2"] - g[1e-3]) / abs(g[1e-3]), "k_a_kernel": kr["k_a"],
                         "mixed_over_k_a": kr["mixed"] / kr["k_a"]})
            print(json.dumps(rows[-1]), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--L", type=int, default=256)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    res = {"meta": {"host": socket.gethostname(), "L": a.L, "start_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}}
    if a.control:
        res["control"] = control(a.out)
    L = a.L
    rows = []
    for aH in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.013):
        ms, etas = masses_eq(0.1, aH)
        kl = 3 * Jsum(ms, etas) / np.pi ** 2
        for n in range(1, L // 2 + 1):
            r = kernel(L, ms, etas, n)
            rows.append({"set": "equal rectangle", "aH": aH, "n": n, "q": r["q"], "k_a": r["k_a"], "mixed": r["mixed"], "k_lat": kl,
                         "k_a_over_k_lat": r["k_a"] / kl, "mixed_over_k_a": r["mixed"] / r["k_a"]})
        print("aH %g done (%.0f s)" % (aH, time.time() - t0), flush=True)
    res["equal"] = rows
    # the physical row: strand 3's lambda = 1 rectangle at a/l_P = 0.0213, light packet massless, twist to avoid the nodes
    qs = [0.0, 0.013090347820517666, 0.013090347820517666, 0.01851254742395717]
    th = (0.371, 0.371, 0.371)
    kl = 3 * Jsum(qs, [1, -1, -1, 1]) / np.pi ** 2
    rows = []
    for n in range(1, L // 2 + 1):
        r = kernel(L, qs, [1, -1, -1, 1], n, th)
        rows.append({"set": "physical a/l_P 0.0213", "n": n, "q": r["q"], "k_a": r["k_a"], "mixed": r["mixed"], "k_lat": kl,
                     "k_a_over_k_lat": r["k_a"] / kl, "mixed_over_k_a": r["mixed"] / r["k_a"]})
    res["physical"] = {"masses": qs, "twist": th, "rows": rows}
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "kernel_L%d.json" % L), "w") as fh:
        json.dump(res, fh, indent=1)
    print("kernel L %d done in %.0f s" % (L, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
