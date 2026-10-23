#!/usr/bin/env python
"""test_w10_defect4_cheb_tail.py (busi W10): defect 4 of REVIEWA_REVIEW_web40i section 5. page_ops_v6.cheb_order sums the Bessel tail
to |z| + 120 and appends a zero: at z = 2000 it returns (K 2120, tail 0.0) while the true tail 2 sum_{j > 2120} |J_j(2000)| is 7.1e-14.
Gate: the bound reproduces 7.1e-14 at z = 2000 and stays below tolerance at the z used.
(1) v6's cheb_order (a verbatim copy here, and page_ops_v6 itself when W10_PAGE_DIR holds it) returns (2120, 0.0) at z = 2000;
(2) w10_lib_v1.cheb_tail(2000, 2120) returns the true tail, about 7.1e-14, with the contour bound of its remainder below 1e-6 of it;
(3) cheb_order(2000, 1e-15) now returns an order above 2120 with a positive tail at or below tol;
(4) the rigorous contour bound at K = 2120 is above the true tail (the Debye prefactor, a factor of order 50) and below 1e-10;
(5) at the z used (0.3 to 200): (K, tail) bitwise v6's and the remainder bound beyond |z| + 120 below 1e-18 (far below tol 1e-15);
    at z = 300 to 1500 the order K is v6's and the tail agrees to 1e-9 (the remainder bound enters the last digits);
(6) page_ops_v7 (when W10_PAGE_DIR holds it) equals the lib bitwise; (7) mpmath, when installed, cross-checks scipy's J_2121(2000);
(8) --page-json FILE: the largest Chebyshev order K_max in a delivered Page json (the "chebyshev" entries) printed and checked below 1500.
Run: qi2 python test_w10_defect4_cheb_tail.py [--page-json FILE]"""
import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.special import jv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w10_lib_v1 as W10  # noqa: E402

FAIL = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAIL.append(msg)


def cheb_order_v6(z, tol):
    """page_ops_v6.cheb_order verbatim."""
    az = abs(float(z))
    jmax = int(az + 120)
    J = np.abs(jv(np.arange(jmax + 1), az))
    tail = 2.0 * np.concatenate([np.cumsum(J[::-1])[::-1][1:], [0.0]])
    K = int(math.ceil(az + 4))
    while K < jmax and tail[K] > tol:
        K += 1
    return K, float(tail[min(K, jmax)])


def find_kmax(obj, acc):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "K_max" and isinstance(v, (int, float)):
                acc.append(float(v))
            else:
                find_kmax(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            find_kmax(v, acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page-json", default=None)
    a = ap.parse_args()
    d = os.environ.get("W10_PAGE_DIR")
    if d and d not in sys.path:
        sys.path.insert(0, d)
    tol = 1e-15
    K6, t6 = cheb_order_v6(2000.0, tol)
    check(K6 == 2120 and t6 == 0.0, "defect 4 reproduced: v6 cheb_order(2000, 1e-15) = (K %d, tail %.3e): the appended zero reported as the tail" % (K6, t6))
    try:
        import page_ops_v6 as P6
        K6b, t6b = P6.cheb_order(2000.0, tol)
        check((K6b, t6b) == (K6, t6), "page_ops_v6.cheb_order itself returns (%d, %.3e) at z = 2000" % (K6b, t6b))
    except ImportError:
        print("INFO page_ops_v6 not importable (set W10_PAGE_DIR); the verbatim copy stands for it")
    tail, rem, jmax = W10.cheb_tail(2000.0, 2120)
    check(6.3e-14 <= tail <= 8.0e-14 and rem <= 1e-6 * tail,
          "gate (the bound reproduces 7.1e-14 at z = 2000): cheb_tail(2000, 2120) = %.4e (computed to j_max %d, the remainder bound beyond it %.1e)" % (tail, jmax, rem))
    K7, t7 = W10.cheb_order(2000.0, tol)
    t7chk = W10.cheb_tail(2000.0, K7)[0]
    check(K7 > 2120 and 0.0 < t7 <= tol and abs(t7chk - t7) <= 1e-6 * t7,
          "the fixed cheb_order(2000, 1e-15) = (K %d, tail %.3e): the order grows past |z| + 120 until the tail is at or below tol (v6 stopped at 2120 with a false 0)" % (K7, t7))
    bound = W10.cheb_tail_bound(2000.0, 2120)
    check(bound >= tail and bound < 1e-10, "the rigorous contour bound at K = 2120 is %.3e, above the true tail by the factor %.1f (the Debye prefactor sqrt(2 pi j tanh alpha)) and below 1e-10" % (bound, bound / tail))
    exact_z = [0.3, 1.0, 2.0, 5.0, 10.0, 20.0, 48.0, 100.0, 200.0]
    ok = True; worst = 0.0
    for z in exact_z:
        a6 = cheb_order_v6(z, tol); a7 = W10.cheb_order(z, tol)
        rb = W10.cheb_tail_bound(z, int(z + 120))
        worst = max(worst, rb)
        if a6 != a7 or rb > 1e-18:
            ok = False
            print("  z %g: v6 (%d, %.17e) v7 (%d, %.17e) remainder bound %.1e" % (z, a6[0], a6[1], a7[0], a7[1], rb))
    check(ok, "gate (stays below tolerance at the z used): for z in %s (K, tail) are v6's bitwise and the remainder bound beyond |z| + 120 is at most %.1e (tol 1e-15; the runs' z is of order 10 to 100)" % (exact_z, worst))
    ok2 = True
    for z in (300.0, 400.0, 700.0, 1000.0, 1500.0):
        a6 = cheb_order_v6(z, tol); a7 = W10.cheb_order(z, tol)
        rel = abs(a7[1] - a6[1]) / max(a6[1], 1e-300)
        print("  z %g: K v6 %d v7 %d; tail v6 %.6e v7 %.6e (relative change %.1e); remainder bound beyond |z| + 120: %.1e" % (z, a6[0], a7[0], a6[1], a7[1], rel, W10.cheb_tail_bound(z, int(z + 120))))
        if a6[0] != a7[0] or rel > 1e-9:
            ok2 = False
    check(ok2, "for z 300 to 1500 the order K is v6's and the tail changes by less than 1e-9 relative (the remainder bound enters the last digits)")
    try:
        import page_ops_v7 as P7
        ok3 = all(P7.cheb_order(z, tol) == W10.cheb_order(z, tol) for z in exact_z + [2000.0]) and P7.cheb_tail(2000.0, 2120) == W10.cheb_tail(2000.0, 2120)
        check(ok3, "page_ops_v7.cheb_order and cheb_tail equal w10_lib_v1's bitwise on the grid and at z = 2000")
    except ImportError:
        print("INFO page_ops_v7 not importable yet (run w10_apply_fixes_v1.py with W10_PAGE_DIR set): the lib stands for its block")
    try:
        import mpmath as mp
        mp.mp.dps = 30
        ref = float(mp.besselj(2121, 2000))
        sv = float(jv(2121, 2000.0))
        check(abs(sv - ref) <= 1e-8 * abs(ref), "mpmath cross-check: J_2121(2000) = %.6e (scipy %.6e, relative difference %.1e)" % (ref, sv, abs(sv - ref) / abs(ref)))
    except ImportError:
        print("INFO mpmath not installed: scipy's jv stands alone (its uniform asymptotics at order 2121 are accurate to about 1e-14)")
    if a.page_json:
        acc = []
        find_kmax(json.load(open(a.page_json)), acc)
        if acc:
            km = max(acc)
            check(km < 1500, "the delivered Page json %s: the largest Chebyshev order K_max %d is below 1500, so v6's appended zero never changed an order (no running number changes)" % (a.page_json, int(km)))
        else:
            print("INFO no K_max entry found in %s" % a.page_json)
    print("test_w10_defect4_cheb_tail: %s" % ("ALL PASS" if not FAIL else "%d FAIL" % len(FAIL)))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
