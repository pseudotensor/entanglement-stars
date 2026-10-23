#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tensor_gate_selftest_v1.py: manufactured-profile checks of tensor_gate_v1's functions (numpy only; python3 -I).

1. The vacuum exterior N = 1, v = -sqrt(r_s/r) (the Painleve-Gullstrand form of the declared constraint,
   requirements sec. 2): the tangential principal value at r_s is -1/r_s exactly; the radial one is
   v'(r_s) = +1/(2 r_s): factor 1/2, sign positive under the declared K_ij = (d_i v_j + d_j v_i)/(2N)
   (the brief's -1/(2 r_s) is this value under K^ADM = -K, the record label's convention, whose tangential
   value would then be +1/r_s; the operator norm is the same either way); the vacuum identity
   K^2 - K_ij K^ij = 4 lam_r lam_t + 2 lam_t^2 = 0 to the finite-difference accuracy; the single root of
   N - |v| at r_s with sigma ||K||_op = sigma/r_s there.
2. A manufactured dense trapped annulus (2 roots of N - |v|): the roots found against their closed forms,
   the tangential value -1/R at each root, and the bound sigma ||K||_op >= sigma/R at each root.
3. The derivative rules: exact on an affine flow at every site including the ends; their errors on the
   vacuum profile ordered as the stencils' orders say.
4. The guards: a regular centre (R = 0) and sites with N <= 0.
5. The clock parsing and the unit detection of the stored T.
Exit code 0 when every check passes.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python3 -I does not add the script's directory
import tensor_gate_v1 as tg  # noqa: E402

FAILURES = []


def check(cond, msg):
    print("[%s] %s" % ("PASS" if cond else "FAIL", msg))
    if not cond:
        FAILURES.append(msg)


def test_vacuum_exterior():
    print("-- 1. the vacuum exterior N = 1, v = -sqrt(r_s/r) (manufactured; r_s = 64a the history's nominal, sigma = 4a)")
    rs, sigma = 64.0, 4.0
    R = np.arange(1.0, 2049.0)                     # 2048 sites at spacing a = 1
    N = np.ones_like(R)
    v = -np.sqrt(rs / R)
    j = int(np.flatnonzero(R == rs)[0])
    dv = {name: tg.fd_derivative(v, R, rule, s) for name, rule, s in
          (("centred", "centred", 1), ("forward", "forward", 1), ("backward", "backward", 1), ("stride_sigma", "centred", 4))}
    exact_dv = 0.5 * np.sqrt(rs) * R ** -1.5       # v' = (1/2) sqrt(r_s) r^(-3/2), arithmetic
    lam_r, lam_t, ok = tg.principal_values(N, v, R, dv["centred"])
    check(abs(lam_t[j] + 1.0 / rs) < 1e-15,
          "tangential v/(rN) at r_s = %.12g against -1/r_s = %.12g (no derivative enters)" % (lam_t[j], -1.0 / rs))
    check(abs(exact_dv[j] - 1.0 / (2 * rs)) < 1e-15,
          "closed form v'(r_s) = (1/2) sqrt(r_s) r_s^(-3/2) = 1/(2 r_s) = %.12g: the factor is 1/2" % exact_dv[j])
    check(lam_r[j] > 0,
          "sign of the radial value at r_s under the declared K is positive (%.6g = +1/(2 r_s)); the brief's -1/(2 r_s) "
          "is the value under K^ADM = -K, whose tangential value would be +1/r_s" % lam_r[j])
    rel = abs(lam_r[j] - exact_dv[j]) / exact_dv[j]
    pred = (5.0 / 8.0) / rs ** 2                   # (a^2/6)|v'''|/v' = (5/8)/r^2, arithmetic
    check(rel < 5e-4, "centred v'/N at r_s = %.9g against +1/(2 r_s) = %.9g: relative error %.2e (stencil estimate %.2e)" % (
        lam_r[j], exact_dv[j], rel, pred))
    errs = {k: abs(dv[k][j] - exact_dv[j]) / exact_dv[j] for k in dv}
    check(errs["forward"] < 2e-2 and errs["backward"] < 2e-2 and errs["stride_sigma"] < 5e-3,
          "rule errors at r_s: forward %.2e, backward %.2e, sigma-stride %.2e (first-order estimate (3/4)/r_s = %.2e; "
          "sigma-stride estimate (5/8)(sigma/r_s)^2 = %.2e)" % (errs["forward"], errs["backward"], errs["stride_sigma"],
                                                              0.75 / rs, (5.0 / 8.0) * (sigma / rs) ** 2))
    check(errs["centred"] < errs["forward"] and errs["centred"] < errs["backward"] and errs["centred"] < errs["stride_sigma"],
          "the centred lattice-spacing rule is the most accurate of the 4 on this profile")
    ham = 4.0 * lam_r * lam_t + 2.0 * lam_t ** 2   # K^2 - K_ij K^ij for 3 principal values lam_r, lam_t, lam_t
    sel = R >= rs
    worst = float(np.max(np.abs(ham[sel]) / lam_t[sel] ** 2))
    check(worst < 1e-3, "vacuum identity K^2 - K_ij K^ij = 0 on r >= r_s: max |4 lam_r lam_t + 2 lam_t^2| / lam_t^2 = %.2e" % worst)
    gate = tg.gate_profile(lam_r, lam_t, sigma)
    check(abs(gate[j] - sigma / rs) < 1e-12,
          "sigma ||K||_op at r_s = %.6g = sigma/r_s (tangential dominates, |v'| = |v|/(2r) there); passes 0.1 since "
          "r_s = 64a >= 10 sigma = 40a" % gate[j])
    roots = tg.interpolated_roots(N - np.abs(v), R)
    check(len(roots) == 1 and abs(roots[0][0] - rs) < 1e-12 and roots[0][1] == 1,
          "roots of N - |v|: %s (1 root at r_s, direction +1 = outer)" % (roots,))
    o = tg.interpolated_roots(N + v, R)
    check(len(o) == 1 and abs(o[0][0] - rs) < 1e-12, "roots of N + v agree with N - |v| where v < 0")


def test_manufactured_dense_root():
    print("-- 2. a manufactured dense trapped annulus (2 roots of N - |v|)")
    sigma = 4.0
    R = np.arange(1.0, 2049.0)
    r0, w, c, A = 25.0, 12.0, 0.6, 0.7
    N = 1.0 - 0.4 * np.exp(-(R / 40.0) ** 2)                 # a smooth lapse below 1 inside
    g = c + A * np.exp(-((R - r0) / w) ** 2)                  # g = 1 at the 2 roots, > 1 between
    v = -N * g                                                # N + v = N (1 - g)
    half = w * np.sqrt(np.log(A / (1.0 - c)))                 # arithmetic: the roots are r0 -+ half
    R_in_exact, R_out_exact = r0 - half, r0 + half
    dv = tg.fd_derivative(v, R, "centred", 1)
    lam_r, lam_t, ok = tg.principal_values(N, v, R, dv)
    gate = tg.gate_profile(lam_r, lam_t, sigma)
    F = N - np.abs(v)
    roots = tg.interpolated_roots(F, R)
    check(len(roots) == 2, "2 roots found: %s" % (roots,))
    if len(roots) != 2:
        return
    (r_in, d_in), (r_out, d_out) = roots
    check(d_in == -1 and d_out == 1, "directions: inner falling (-1), outer rising (+1)")
    check(abs(r_in - R_in_exact) < 1e-2 and abs(r_out - R_out_exact) < 1e-2,
          "roots %.5f, %.5f against the closed forms %.5f, %.5f (linear interpolation on a = 1)" % (
              r_in, r_out, R_in_exact, R_out_exact))
    mid = (R > r_in) & (R < r_out)
    check(bool(np.all(F[mid] < 0)) and bool(np.all(F[~mid & (R > 0)] > 0)),
          "N - |v| < 0 between the roots and > 0 outside (a trapped annulus)")
    for r, kind in ((r_in, "inner"), (r_out, "outer")):
        lt = tg.interp(r, R, lam_t)
        gi = tg.interp(r, R, gate)
        check(abs(lt + 1.0 / r) < 1e-3 / r,
              "%s root %.4f: tangential %.7g against -1/R = %.7g (residual %.1e)" % (kind, r, lt, -1.0 / r, lt + 1.0 / r))
        check(gi >= sigma / r * (1.0 - 1e-3),
              "%s root %.4f: sigma ||K||_op = %.5f >= sigma/R = %.5f (radial part %.5f)" % (
                  kind, r, gi, sigma / r, sigma * abs(tg.interp(r, R, lam_r))))
    check(r_out < 10 * sigma and sigma / r_out > 0.1,
          "outer root %.3f < 10 sigma = %.1f, so sigma/R = %.4f > 0.1: the criterion fails at the root by the tangential bound" % (
              r_out, 10 * sigma, sigma / r_out))
    rm = tg.region_max(gate, lam_r, lam_t, N, R, (R > 0) & (R <= r_out), 0.1)
    inside = (R > 0) & (R <= r_out)
    jmax = int(np.flatnonzero(inside)[np.argmax(gate[inside])])
    check(rm["value"] == float(gate[jmax]) and rm["r"] == float(R[jmax]) and rm["n_undefined"] == 0,
          "region_max over 0 < r <= R_out: %.5f at r = %.1f (%s part, %d of %d sites failing)" % (
              rm["value"], rm["r"], rm["part"], rm["n_failing"], rm["n_sites"]))


def test_derivative_rules():
    print("-- 3. the derivative rules on an affine flow (exact at every site, ends included)")
    R = np.arange(1.0, 101.0)
    v = -0.01 * R + 0.002
    for name, rule, s in (("centred", "centred", 1), ("forward", "forward", 1), ("backward", "backward", 1),
                          ("stride_sigma", "centred", 4)):
        d = tg.fd_derivative(v, R, rule, s)
        check(bool(np.all(np.isfinite(d))) and float(np.max(np.abs(d + 0.01))) < 1e-13,
              "%s rule exact on v = -0.01 r + 0.002: max |v' + 0.01| = %.1e" % (name, float(np.max(np.abs(d + 0.01)))))


def test_guards():
    print("-- 4. the guards: a regular centre and N <= 0")
    R = np.arange(0.0, 50.0)
    N = np.ones_like(R)
    v = -0.01 * R
    lam_r, lam_t, ok = tg.principal_values(N, v, R, tg.fd_derivative(v, R, "centred", 1))
    check(abs(lam_t[0] + 0.01) < 1e-13 and abs(lam_t[5] + 0.01) < 1e-13,
          "regular centre: v/(rN) at R = 0 replaced by v'/N = %.4f; at R = 5 it is %.4f" % (lam_t[0], lam_t[5]))
    N2 = N.copy()
    N2[10], N2[11] = 0.0, -0.1
    lam_r, lam_t, ok = tg.principal_values(N2, v, R, tg.fd_derivative(v, R, "centred", 1))
    gate = tg.gate_profile(lam_r, lam_t, 4.0)
    rm = tg.region_max(gate, lam_r, lam_t, N2, R, R > 0, 0.1)
    check((not ok[10]) and (not ok[11]) and np.isnan(gate[10]) and np.isnan(gate[11]) and rm["n_undefined"] == 2
          and rm["value"] is not None,
          "N <= 0 at 2 sites: tensor undefined there, excluded from the maximum and counted (%d)" % rm["n_undefined"])


def test_parsing():
    print("-- 5. clock parsing and the stored-T unit detection")
    cl = tg.parse_clocks("12.125,30,40:46:0.125")
    check(len(cl) == 51 and cl[0] == 12.125 and cl[1] == 30.0 and cl[2] == 40.0 and cl[-1] == 46.0,
          "'12.125,30,40:46:0.125' -> %d clocks, first %s, last %s" % (len(cl), cl[0], cl[-1]))
    T = 8.0 * np.arange(776)                                  # 0 to 6200 in steps of 8 = 0.125 r_s at r_s = 64
    tors, note = tg.clocks_in_rs(T, 64.0)
    check(abs(tors[97] - 12.125) < 1e-12 and abs(tors[-1] - 96.875) < 1e-12,
          "T = 776 at index 97 reads T/r_s = %.3f; last %.3f (%s)" % (tors[97], tors[-1], note))


def main():
    for t in (test_vacuum_exterior, test_manufactured_dense_root, test_derivative_rules, test_guards, test_parsing):
        t()
    print("")
    if FAILURES:
        print("SELFTEST FAILED: %d check(s)" % len(FAILURES))
        for m in FAILURES:
            print("  - " + m)
        return 1
    print("SELFTEST PASSED: every check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
