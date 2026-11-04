#!/usr/bin/env python3
"""
firstlaw_selftest_v4bw6.py -- the self-test of firstlaw_records_v4b.py (busi W6); engine-free; seconds.  v4b W6

Version 4b (W6W7_selftest_fix_v1b.md). The version 4 self-test's S1 asserted |quadrature - closed form| of w_n(R) below 1e-9 for a 4th-order Simpson rule
at h = ell/16, whose truncation error at a cut inside the kernel's support is h^4 |g'''|/24 (the even-count end formula) plus (h^4/180) g''' (the composite
part), about 1e-6 for the n = 0 kernel at R = R_0 + 1.7: the observed 1.4e-6 was the rule's error, not the closed form's (the closed form is exact), and
no 4th-order rule meets 1e-9 at that spacing. The code's rule is now 6th order (Boole's rule on 4-cell panels, the end cells by the degree-6 local
polynomial), because the representation defect it serves must read the engine's source representation (1e-8 relative) and not the quadrature.  S1 now
(a) checks the kernel's normalization at roundoff, (b) certifies the closed form independently of any quadrature by its derivative dw_n/dR = 4 pi R^2 K_n(R)
(Richardson centred differences, 1e-10) and its limits w_n(0) = 0, w_n(inf) = 1, and (c) checks the quadrature against the closed form at h = ell/16 against
the a-priori bound (2/945) h^6 x 2 max|g^(5)| + 4 (12.3/5040) h^8 max|g^(7)| with the derivatives measured on the sampled kernel, the convergence ratio
between ell/16 and ell/32 in [40, 110] (6th order 64; 5th 32, 7th 128) and the error at ell/64 below 1e-11.  S2's tolerance is tightened from 1e-8 to 1e-9
relative (version 4 passed it at 3.2e-9, Simpson's error; the 6th-order rule gives about 1e-11).  S3 to S6 are version 4's.

S1 kernel: as above.
S2 representation: M_rep(r) = sum_n Q_n w_n(r) - C_u(r) against the quadrature of 4 pi s^2 (rho_ell + u J_ell) with rho_ell, J_ell the kernel sums on the
   dense grid, on manufactured sources and a manufactured flow (1e-9 relative): the identity M = sum Q w - C_u.
S3 closure: the four columns sum to sum_n w_n dQ_n - Delta C_u to roundoff (1e-13 relative) on two manufactured records with a reference split, a
   changing flow and a moving cut; the sum equals Delta M_rep exactly.
S4 signs and units (energies in the units of eps_tot, lattice units): (i) a transfer of delta from a site far inside the cut to a site far outside gives
   col1 = -delta, col2 = 0, Delta M = -delta; (ii) the same from a site ell/2 inside the cut gives col1 = -delta and col2 = delta (1 - w_in) > 0, the sum
   -delta w_in; (iii) a reference change +delta at a site far inside gives col4 = -delta with col1 = col2 = 0; (iv) one site's momentum J at R_n with the
   flow changing from a constant to a linear u_1 (s - R_n) gives col3 = u_1 J ell^2/R_n (the kernel's first moment, 1e-6), positive for u_1 J > 0; (v) a
   constant flow gives col3 = 0 to roundoff; (vi) scaling every energy and momentum by lambda scales every column by lambda.
S5 full clock: the residual closes to roundoff on a manufactured consistent (N, u, rho, J, p_r) pair whose s is linear in time (1e-12 against the terms);
   with rho + p_r > 0, N' = 0 and s_t = 0 the residual is -4 pi G r (rho + p_r) < 0 exactly.
S6 gate: ell/R: 4/36.3 = 0.110 (outside the working value), 2/46.3 inside, 2/7 outside.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import firstlaw_records_v4b as V4  # noqa: E402

FOUR_PI = 4.0 * np.pi
G = 0.25
ELL = 2.0


def manufactured_grid(n_sites=320, fine=16):
    R = 0.5 + np.arange(n_sites, dtype=float)
    h = ELL / fine
    r = np.arange(int(round((R[-1] + 12 * ELL) / h)) + 1, dtype=float) * h
    return R, r, h


def test_radii(Rn):
    return [3.3, 10.0, Rn, Rn + 0.3, Rn + 0.4, Rn + 1.7, Rn + 2.9, Rn + 15.0]


def main():
    fails = 0
    R, r, h = manufactured_grid()
    K = V4.kernel_matrix(ELL, R, r)
    sites = (0, 1, 9, 99, 250)
    # S1 (a): the normalization over the full range (exact up to roundoff: the integrand vanishes at both ends)
    worst_norm = 0.0
    for n in sites:
        g = FOUR_PI * r * r * K[n]
        worst_norm = max(worst_norm, abs(V4.integral_0_to(r, g, float(r[-1])) - 1.0))
    # S1 (b): the closed form's derivative against the kernel, independent of any quadrature
    worst_der = 0.0
    delta = 1e-3 * ELL
    for n in sites:
        Rn = float(R[n])
        for Rc in test_radii(Rn):
            if 2 * delta < Rc <= r[-1] - 2 * delta:
                def D(dd):
                    return (V4.cumulative_weight(ELL, Rn, Rc + dd)[0] - V4.cumulative_weight(ELL, Rn, Rc - dd)[0]) / (2.0 * dd)
                der = (4.0 * D(0.5 * delta) - D(delta)) / 3.0
                g_exact = FOUR_PI * Rc * Rc * float(V4.shell_kernel(ELL, Rn, np.array([Rc]))[0])
                worst_der = max(worst_der, abs(der - g_exact))
    # S1 (c): the limits
    Rsel = R[list(sites)]
    w0 = float(np.max(np.abs(V4.cumulative_weight(ELL, Rsel, 0.0))))
    winf = float(np.max(np.abs(V4.cumulative_weight(ELL, Rsel, float(R[-1]) + 30 * ELL) - 1.0)))
    # S1 (d): the quadrature against the closed form at three spacings, the a-priori bound from the measured derivatives
    errs = {}
    f5max = 0.0
    f7max = 0.0
    for fine in (16, 32, 64):
        hf = ELL / fine
        rf = np.arange(int(round((R[-1] + 12 * ELL) / hf)) + 1, dtype=float) * hf
        worst = 0.0
        for n in sites:
            Rn = float(R[n])
            g = FOUR_PI * rf * rf * V4.shell_kernel(ELL, Rn, rf)
            if fine == 64:
                d = g.copy()
                for _ in range(5):
                    d = np.gradient(d, hf, edge_order=2)
                f5max = max(f5max, float(np.max(np.abs(d[20:-20]))))
                for _ in range(2):
                    d = np.gradient(d, hf, edge_order=2)
                f7max = max(f7max, float(np.max(np.abs(d[30:-30]))))
            for Rc in test_radii(Rn):
                if Rc <= rf[-1]:
                    worst = max(worst, abs(V4.integral_0_to(rf, g, Rc) - float(V4.cumulative_weight(ELL, Rn, Rc)[0])))
        errs[fine] = worst
    h16 = ELL / 16
    bound16 = (2.0 / 945.0) * h16 ** 6 * 2.0 * f5max + 4.0 * (12.3 / 5040.0) * h16 ** 8 * f7max
    ratio = errs[16] / errs[32] if errs[32] > 0 else float('inf')
    ok = worst_norm < 1e-12 and worst_der < 1e-10 and w0 < 1e-15 and winf < 1e-14 and errs[16] <= bound16 and 40.0 <= ratio <= 110.0 and errs[64] < 1e-11
    print('%s S1 kernel: norm max |int - 1| %.1e (1e-12); the closed form dw/dR against 4 pi R^2 K_n(R) max |diff| %.1e (1e-10); w(0) %.1e, w(inf) - 1 %.1e; '
          'quadrature against the closed form: max |err| at ell/16 %.2e against the a-priori bound %.2e (max |g^(5)| %.3g, max |g^(7)| %.3g), at ell/32 %.2e (ratio %.1f; 6th order 64, band 40 to 110), at ell/64 %.1e (1e-11)' % (
              'PASS' if ok else 'FAIL', worst_norm, worst_der, w0, winf, errs[16], bound16, f5max, f7max, errs[32], ratio, errs[64]))
    fails += 0 if ok else 1
    # S2: manufactured sources and flow
    rng = np.random.default_rng(3)
    eps = np.exp(-((R - 40.0) / 12.0) ** 2) * (1.0 + 0.1 * rng.standard_normal(R.size))
    J = 0.3 * eps * (1.0 + 0.2 * np.sin(R / 5.0))
    u = -0.8 * np.tanh(r / 30.0) * (1.0 + 0.05 * np.sin(r / 7.0))
    u_sites = np.interp(R, r, u)
    rho_l = eps @ K
    J_l = J @ K
    gM = FOUR_PI * r * r * (rho_l + u * J_l)
    gCu, _ = V4.commutator_integrand(r, u, J, u_sites, K)
    Q = eps + u_sites * J
    worst = 0.0
    for Rc in (25.3, 35.0, 44.7, 60.0):
        Mq = V4.integral_0_to(r, gM, Rc)
        Mrep = float(V4.cumulative_weight(ELL, R, Rc) @ Q) - V4.integral_0_to(r, gCu, Rc)
        worst = max(worst, abs(Mq - Mrep) / abs(Mq))
    ok = worst < 1e-9
    print('%s S2 representation: max |M_quad - M_rep|/M %.1e over 4 radii (tolerance 1e-9)' % ('PASS' if ok else 'FAIL', worst))
    fails += 0 if ok else 1
    # S3: closure on two records
    eps_b = eps * (1.0 + 0.03 * rng.standard_normal(R.size))
    J_b = J * (1.0 + 0.05 * rng.standard_normal(R.size))
    ref_a = 0.1 * np.exp(-((R - 30.0) / 20.0) ** 2)
    ref_b = ref_a * 1.02
    u_b = u * 1.01
    u_sites_b = np.interp(R, r, u_b)
    Rc = 38.6
    w = V4.cumulative_weight(ELL, R, Rc)
    chi = (R <= Rc).astype(float)
    Q_a_state, Q_b_state = eps + ref_a + u_sites * J, eps_b + ref_b + u_sites_b * J_b
    q_a_ref, q_b_ref = ref_a, ref_b
    Cu_a = V4.integral_0_to(r, V4.commutator_integrand(r, u, J, u_sites, K)[0], Rc)
    Cu_b = V4.integral_0_to(r, V4.commutator_integrand(r, u_b, J_b, u_sites_b, K)[0], Rc)
    cols = V4.four_columns(w, chi, Q_b_state - Q_a_state, q_b_ref - q_a_ref, Cu_b - Cu_a)
    dM_rep = (float(w @ (Q_b_state - q_b_ref)) - Cu_b) - (float(w @ (Q_a_state - q_a_ref)) - Cu_a)
    ok = abs(cols['closure']) <= 1e-13 * max(1.0, abs(cols['rep'])) and abs(cols['sum4'] - dM_rep) <= 1e-12 * max(1.0, abs(dM_rep))
    print('%s S3 closure: sum4 - rep %.1e, sum4 - Delta M_rep %.1e (sum4 %.6g)' % ('PASS' if ok else 'FAIL', cols['closure'], cols['sum4'] - dM_rep, cols['sum4']))
    fails += 0 if ok else 1
    # S4 signs
    delta_q = 0.37
    z = np.zeros(R.size)
    dQ = z.copy()
    dQ[10] = -delta_q
    dQ[300] = delta_q
    c = V4.four_columns(w, chi, dQ, z, 0.0)
    ok_i = abs(c['col1_flux_old_cut'] + delta_q) < 1e-14 and abs(c['col2_cut_assignment']) < 1e-12 and abs(c['sum4'] + delta_q) < 1e-12
    n_in = int(np.argmin(np.abs(R - (Rc - ELL / 2))))
    dQ = z.copy()
    dQ[n_in] = -delta_q
    dQ[300] = delta_q
    c = V4.four_columns(w, chi, dQ, z, 0.0)
    w_in = w[n_in]
    ok_ii = abs(c['col1_flux_old_cut'] + delta_q) < 1e-14 and abs(c['col2_cut_assignment'] - delta_q * (1 - w_in)) < 1e-12 and c['col2_cut_assignment'] > 0 and abs(c['sum4'] + delta_q * w_in) < 1e-12
    dref = z.copy()
    dref[10] = delta_q
    c = V4.four_columns(w, chi, z, dref, 0.0)
    ok_iii = abs(c['col4_continued_reference'] + delta_q) < 1e-14 and c['col1_flux_old_cut'] == 0.0 and c['col2_cut_assignment'] == 0.0
    n0 = int(np.argmin(np.abs(R - 60.0)))
    Jone = z.copy()
    Jone[n0] = 1.3
    u1 = 0.01
    ua = -0.5 * np.ones_like(r)
    ub = -0.5 + u1 * (r - R[n0])
    Rfar = R[n0] + 20 * ELL
    Cu_a4 = V4.integral_0_to(r, V4.commutator_integrand(r, ua, Jone, np.interp(R, r, ua), K)[0], Rfar)
    Cu_b4 = V4.integral_0_to(r, V4.commutator_integrand(r, ub, Jone, np.interp(R, r, ub), K)[0], Rfar)
    c = V4.four_columns(V4.cumulative_weight(ELL, R, Rfar), (R <= Rfar).astype(float), z, z, Cu_b4 - Cu_a4)
    expect = u1 * Jone[n0] * ELL * ELL / R[n0]
    ok_iv = abs(c['col3_averaging_commutator'] - expect) < 1e-6 * abs(expect) and c['col3_averaging_commutator'] > 0
    ok_v = abs(Cu_a4) < 1e-13
    lam = 3.7
    c1 = V4.four_columns(w, chi, Q_b_state - Q_a_state, q_b_ref - q_a_ref, Cu_b - Cu_a)
    c2 = V4.four_columns(w, chi, lam * (Q_b_state - Q_a_state), lam * (q_b_ref - q_a_ref), lam * (Cu_b - Cu_a))
    ok_vi = all(abs(c2[k] - lam * c1[k]) < 1e-12 * max(1.0, abs(lam * c1[k])) for k in ('col1_flux_old_cut', 'col2_cut_assignment', 'col3_averaging_commutator', 'col4_continued_reference'))
    ok = ok_i and ok_ii and ok_iii and ok_iv and ok_v and ok_vi
    print('%s S4 signs and units: (i) transfer across the cut col1 = -delta %s; (ii) from ell/2 inside, col2 = delta (1 - w_in) = %.6g > 0 %s; (iii) reference +delta gives col4 = -delta %s; '
          '(iv) linear flow on one site: col3 %.6g against u_1 J ell^2/R_n %.6g %s; (v) constant flow col3 = 0 (%.1e) %s; (vi) linear in lambda %s' % (
              'PASS' if ok else 'FAIL', ok_i, delta_q * (1 - w_in), ok_ii, ok_iii, c['col3_averaging_commutator'], expect, ok_iv, Cu_a4, ok_v, ok_vi))
    fails += 0 if ok else 1
    # S5 full clock
    Rs = 40.0
    dt = 8.0
    N = lambda x: 1.0 - 0.3 * np.exp(-((x - 40.0) / 15.0) ** 2)  # noqa: E731
    Np = lambda x: 0.3 * np.exp(-((x - 40.0) / 15.0) ** 2) * 2.0 * (x - 40.0) / 15.0 ** 2  # noqa: E731
    u0 = lambda x: -0.8 / (1.0 + (x / 50.0) ** 2)  # noqa: E731
    u1f = lambda x: 0.002 * np.sin(x / 9.0)  # noqa: E731
    ua_v, ub_v = u0(Rs) - 0.5 * dt * u1f(Rs), u0(Rs) + 0.5 * dt * u1f(Rs)
    Nv, Npv = N(Rs), Np(Rs)
    Ja, Jb = -ua_v * Npv / (FOUR_PI * G * Rs * Nv), -ub_v * Npv / (FOUR_PI * G * Rs * Nv)
    rho_plus_p = ((1.0 + u0(Rs) ** 2) * Npv / Nv - u1f(Rs) / Nv) / (FOUR_PI * G * Rs)
    rho_v, p_v = 0.6 * rho_plus_p, 0.4 * rho_plus_p
    lnNp_a, lnNp_b = -FOUR_PI * G * Rs * Ja / ua_v, -FOUR_PI * G * Rs * Jb / ub_v
    fc = V4.full_clock_values(Rs, dt, G, -ua_v, -ub_v, Nv, Nv, lnNp_a, lnNp_b, rho_v, rho_v, p_v, p_v)
    scale = max(abs(fc['term_st_over_N_mid']), abs(fc['term_clock_mid']), abs(fc['term_matter_mid']))
    ok_a = abs(fc['residual_mid']) < 1e-12 * scale
    fc2 = V4.full_clock_values(Rs, dt, G, 1.0, 1.0, 0.7, 0.7, 0.0, 0.0, 1e-4, 1e-4, 2e-5, 2e-5)
    ok_b = abs(fc2['residual_mid'] + FOUR_PI * G * Rs * 1.2e-4) < 1e-18 and fc2['residual_mid'] < 0
    ok = ok_a and ok_b
    print('%s S5 full clock: residual on the consistent manufactured pair %.1e against the terms %.3g; at rest with N\' = 0 the residual is -4 pi G r (rho + p_r) = %.6g < 0 %s' % (
        'PASS' if ok else 'FAIL', fc['residual_mid'], scale, fc2['residual_mid'], ok_b))
    fails += 0 if ok else 1
    # S6 gate
    g1, g2, g3 = V4.gate_value(4.0, 36.3), V4.gate_value(2.0, 46.3), V4.gate_value(2.0, 7.0)
    ok = abs(g1['gate'] - 0.1102) < 1e-3 and g1['inside'] is False and g2['inside'] is True and g3['inside'] is False
    print('%s S6 gate: 4/36.3 = %.4f (inside %s), 2/46.3 = %.4f (inside %s), 2/7 = %.3f (inside %s)' % ('PASS' if ok else 'FAIL', g1['gate'], g1['inside'], g2['gate'], g2['inside'], g3['gate'], g3['inside']))
    fails += 0 if ok else 1
    print('6 checks, %d failed' % fails)
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
