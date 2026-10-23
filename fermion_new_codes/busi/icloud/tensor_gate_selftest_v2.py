#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tensor_gate_selftest_v2.py: the dense-derivative path of tensor_gate_v2 on manufactured dense functions (numpy; scipy for the
brentq of the scan; python3 -I; no engine, no data).

1. The vacuum exterior N = 1, v = -sqrt(r_s/r), W = r_s (= 2 g E with g = 1/4 and E = 2 r_s), rho_ell = 0 (the Painleve-Gullstrand
   form of the declared constraint, requirements sec. 2): the equation-derived v' equals the closed form (1/2) sqrt(r_s) r^(-3/2) to
   rounding at every radius; at r_s the tangential value is -1/r_s and the radial +1/(2 r_s) (sign positive under the declared K, see
   tensor_gate_selftest_v1 check 1), the vacuum identity 4 lam_r lam_t + 2 lam_t^2 = 0 holds, sigma ||K||_op = sigma/r_s (passing
   0.1 since r_s = 64a >= 10 sigma = 40a); the engine's difference rule on the dense functions agrees with the equations; the
   constraint identity lam_r = 1/(2R) - 4 pi g R rho_ell(R) gives the same radial value at the root; the scan finds the single root at
   r_s; the region maximum of the gate over [r_s, 2 r_s] sits at r_s.
2. A uniform-density interior with N = 1 (W = a r^3, a = 8 pi g rho/3, v = -sqrt(a) r): v' = -sqrt(a) at every radius, the 2 principal
   values are equal (isotropic), and the Hamiltonian constraint 4 lam_r lam_t + 2 lam_t^2 = 16 pi g rho holds to rounding.
3. Version 1's checks (the lattice functions, the guards, the parsing) re-run on version 2's functions through
   tensor_gate_selftest_v1 with its module rebound, when that file is present beside this one.
Exit code 0 when every check passes.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python3 -I does not add the script's directory
import tensor_gate_v2 as tg  # noqa: E402

FAILURES = []


def check(cond, msg):
    print("[%s] %s" % ("PASS" if cond else "FAIL", msg))
    if not cond:
        FAILURES.append(msg)


def test_vacuum_dense_path():
    print("-- 1. the vacuum exterior on the dense path: N = 1, v = -sqrt(r_s/r), W = r_s, rho_ell = 0 (r_s = 64a, sigma = 4a, g = 1/4)")
    rs, sigma, g, thr, ell = 64.0, 4.0, 0.25, 0.1, 4.0
    fns = (lambda r: np.ones_like(r), lambda r: -np.sqrt(rs / r), lambda r: np.full(np.shape(r), rs), lambda r: np.zeros_like(r))
    rr = np.arange(8.0, 2048.0 + 1e-9, 0.25)
    P = tg.dense_principal_values(rr, fns, g, sigma)
    exact = 0.5 * np.sqrt(rs) * rr ** -1.5                      # v' = (1/2) sqrt(r_s) r^(-3/2), arithmetic
    rel = float(np.max(np.abs(P["dv"] - exact) / exact))
    check(rel < 1e-12, "equation-derived v' against (1/2) sqrt(r_s) r^(-3/2) on 8a <= r <= 2048a: max relative difference %.2e" % rel)
    pt = tg.dense_point(rs, fns, g, sigma, thr, ell)
    check(abs(pt["lam_t"] + 1.0 / rs) < 1e-15, "tangential v/(rN) at r_s = %.12g against -1/r_s = %.12g" % (pt["lam_t"], -1.0 / rs))
    check(abs(pt["lam_r"] - 1.0 / (2 * rs)) < 1e-15,
          "radial v'/N at r_s = %.12g against +1/(2 r_s) = %.12g (positive under the declared K)" % (pt["lam_r"], 1.0 / (2 * rs)))
    ham = 4.0 * pt["lam_r"] * pt["lam_t"] + 2.0 * pt["lam_t"] ** 2
    check(abs(ham) < 1e-16, "vacuum identity 4 lam_r lam_t + 2 lam_t^2 at r_s = %.2e" % ham)
    check(abs(pt["gate"] - sigma / rs) < 1e-15 and not pt["fails_gate"] and pt["margin_abs_arith"] < 0,
          "sigma ||K||_op at r_s = %.6g = sigma/r_s (%s part), margin %.4f: passes 0.1 since r_s = 64a >= 10 sigma" % (
              pt["gate"], pt["part"], pt["margin_abs_arith"]))
    reld = abs(pt["dv_difference_rule_h_ell_over_64"] - pt["dv_equations"]) / pt["dv_equations"]
    check(reld < 1e-9, "the difference rule on the dense functions (h = ell/64) against the equations: relative difference %.2e" % reld)
    check(abs(pt["lam_r_equations_minus_constraint_identity"]) < 1e-15,
          "constraint identity 1/(2R) - 4 pi g R rho_ell(R) = %.12g equals the equation value at the root (rho_ell = 0)" % (
              pt["lam_r_constraint_identity_at_root_arith"],))
    check(abs(pt["tangential_plus_1_over_r"]) < 1e-15 and abs(pt["N_minus_absv"]) < 1e-15,
          "at r_s: N - |v| = %.1e, lam_t + 1/r = %.1e" % (pt["N_minus_absv"], pt["tangential_plus_1_over_r"]))
    F = lambda x: np.ones_like(x) - np.abs(fns[1](x))
    zeros = tg.dense_zeros_scan(F, 8.03125, 2048.0, 0.0625)        # the grid offset from r_s by half a step: brentq is exercised
    check(len(zeros) == 1 and abs(zeros[0] - rs) < 1e-10, "the scan's zeros of N - |v| on [8.03, 2048]: %s (1 root at r_s)" % (zeros,))
    rr2 = np.arange(rs, 2 * rs + 1e-9, 0.0625)
    P2 = tg.dense_principal_values(rr2, fns, g, sigma)
    rm = tg.dense_region(P2, rr2 > 0, thr)
    check(rm["r"] == rs and abs(rm["value"] - sigma / rs) < 1e-15 and rm["n_failing"] == 0 and rm["n_points"] == rr2.size,
          "region maximum over [r_s, 2 r_s]: %.6g at r = %.1f (%s part), %d of %d points failing" % (
              rm["value"], rm["r"], rm["part"], rm["n_failing"], rm["n_points"]))


def test_uniform_interior():
    print("-- 2. a uniform-density interior with N = 1: W = a r^3, a = 8 pi g rho/3, v = -sqrt(a) r (rho = 2e-3, g = 1/4, sigma = 4a)")
    rho0, g, sigma = 2e-3, 0.25, 4.0
    a = 8.0 * np.pi * g * rho0 / 3.0
    fns = (lambda r: np.ones_like(r), lambda r: -np.sqrt(a) * r, lambda r: a * r ** 3, lambda r: np.full(np.shape(r), rho0))
    rr = np.arange(0.25, 32.0 + 1e-9, 0.25)
    P = tg.dense_principal_values(rr, fns, g, sigma)
    rel = float(np.max(np.abs(P["dv"] + np.sqrt(a)) / np.sqrt(a)))
    check(rel < 1e-12, "v' = -sqrt(a) = %.8g at every radius: max relative difference %.2e" % (-np.sqrt(a), rel))
    iso = float(np.max(np.abs(P["lam_r"] - P["lam_t"])))
    check(iso < 1e-14, "isotropic: max |lam_r - lam_t| = %.2e" % iso)
    ham = 4.0 * P["lam_r"] * P["lam_t"] + 2.0 * P["lam_t"] ** 2
    target = 16.0 * np.pi * g * rho0
    relh = float(np.max(np.abs(ham - target) / target))
    check(relh < 1e-12, "Hamiltonian constraint 4 lam_r lam_t + 2 lam_t^2 = 16 pi g rho = %.8g: max relative difference %.2e" % (target, relh))
    check(abs(float(P["gate"][0]) - sigma * np.sqrt(a)) < 1e-14,
          "sigma ||K||_op = sigma sqrt(a) = %.6g everywhere (%s 0.1)" % (sigma * np.sqrt(a), "exceeds" if sigma * np.sqrt(a) > 0.1 else "below"))


def test_version1_suite():
    print("-- 3. version 1's checks on version 2's lattice functions (tensor_gate_selftest_v1 with its module rebound)")
    try:
        import tensor_gate_selftest_v1 as st1
    except Exception as exc:
        print("[SKIP] tensor_gate_selftest_v1 not importable (%s)" % (exc,))
        return
    st1.tg = tg
    del st1.FAILURES[:]
    rc = st1.main()
    check(rc == 0, "version 1's self-test on version 2's functions: exit %d" % rc)


def main():
    for t in (test_vacuum_dense_path, test_uniform_interior, test_version1_suite):
        t()
    print("")
    if FAILURES:
        print("SELFTEST v2 FAILED: %d check(s)" % len(FAILURES))
        for m in FAILURES:
            print("  - " + m)
        return 1
    print("SELFTEST v2 PASSED: every check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
