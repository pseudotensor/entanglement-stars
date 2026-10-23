#!/usr/bin/env python
"""test_w10_defect4_cheb_tail_v2.py (busi W10): defect 4 of REVIEWA_REVIEW_web40i section 5. page_ops_v6.cheb_order sums the Bessel tail
to |z| + 120 and appends a zero: at z = 2000 it returns (K 2120, tail 0.0) while the true tail 2 sum_{j > 2120} |J_j(2000)| is 7.1e-14.
Gate: the bound reproduces 7.1e-14 at z = 2000 and stays below tolerance at the z used.
Version 2 (W10_reading_v1.md section 3): item (5)'s second assertion of v1, "for z 300 to 1500 the order K is v6's and the tail changes by
less than 1e-9 relative", was a prediction of mine that the box disproved (the fix adds the remainder bound itself to the tail, 4.4e-2 of
it at z = 1000; at z = 1500 v6's K was one short, its true tail 1.4e-15 above tol). It is replaced by what the fix does: through z = 1000
the summation limit |z| + 120 is not extended (the remainder bound is below 0.1 tol), K is v6's at the tested z and the tail is v6's plus
that bound to rounding; at z = 1500 K is v6's + 1, the tail is at or below tol and v6's true tail at its K exceeds tol. The --page-json
check reads K_max against K(z = 200) (everything bitwise below it) and 1000 (the tested range), not 1500. The per-z lines stay.
(1) v6's cheb_order (a verbatim copy here, and page_ops_v6 itself when W10_PAGE_DIR holds it) returns (2120, 0.0) at z = 2000;
(2) w10_lib_v1.cheb_tail(2000, 2120) returns the true tail, about 7.1e-14, with the contour bound of its remainder below 1e-6 of it;
(3) cheb_order(2000, 1e-15) now returns an order above 2120 with a positive tail at or below tol;
(4) the rigorous contour bound at K = 2120 is above the true tail (the Debye prefactor, a factor of order 50) and below 1e-10;
(5) at the z used (0.3 to 200): (K, tail) bitwise v6's and the remainder bound beyond |z| + 120 below 1e-18 (far below tol 1e-15);
    z 300 to 1000 and z 1500 as above;
(6) page_ops_v7 (when W10_PAGE_DIR holds it) equals the lib bitwise; (7) mpmath, when installed, cross-checks scipy's J_2121(2000);
(8) --page-json FILE: the largest Chebyshev order K_max in a delivered Page json (the "chebyshev" entries) against K(z = 200) and 1000.
Run: qi2 python test_w10_defect4_cheb_tail_v2.py [--page-json FILE]"""
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
    # (1) the defect reproduced
    K6, t6 = cheb_order_v6(2000.0, tol)
    check(K6 == 2120 and t6 == 0.0, "defect 4 reproduced: v6 cheb_order(2000, 1e-15) = (K %d, tail %.3e): the appended zero reported as the tail" % (K6, t6))
    try:
        import page_ops_v6 as P6
        K6b, t6b = P6.cheb_order(2000.0, tol)
        check((K6b, t6b) == (K6, t6), "page_ops_v6.cheb_order itself returns (%d, %.3e) at z = 2000" % (K6b, t6b))
    except ImportError:
        print("INFO page_ops_v6 not importable (set W10_PAGE_DIR); the verbatim copy stands for it")
    # (2) the gate at z = 2000
    tail, rem, jmax = W10.cheb_tail(2000.0, 2120)
    check(6.3e-14 <= tail <= 8.0e-14 and rem <= 1e-6 * tail,
          "gate (the bound reproduces 7.1e-14 at z = 2000): cheb_tail(2000, 2120) = %.4e (computed to j_max %d, the remainder bound beyond it %.1e)" % (tail, jmax, rem))
    # (3) the fixed order
    K7, t7 = W10.cheb_order(2000.0, tol)
    t7chk = W10.cheb_tail(2000.0, K7)[0]
    check(K7 > 2120 and 0.0 < t7 <= tol and abs(t7chk - t7) <= 1e-6 * t7,
          "the fixed cheb_order(2000, 1e-15) = (K %d, tail %.3e): the order grows past |z| + 120 until the tail is at or below tol (v6 stopped at 2120 with a false 0)" % (K7, t7))
    # (4) the rigorous bound
    bound = W10.cheb_tail_bound(2000.0, 2120)
    check(bound >= tail and bound < 1e-10, "the rigorous contour bound at K = 2120 is %.3e, above the true tail by the factor %.1f (the Debye prefactor sqrt(2 pi j tanh alpha)) and below 1e-10" % (bound, bound / tail))
    # (5a) the z used: bitwise
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
    # (5b) z 300 to 1000: what the fix does there (the v1 assertion was a wrong prediction)
    ok2 = True; rb_max = 0.0
    for z in (300.0, 400.0, 700.0, 1000.0):
        a6 = cheb_order_v6(z, tol); a7 = W10.cheb_order(z, tol)
        rb = W10.cheb_tail_bound(z, int(z + 120))
        rb_max = max(rb_max, rb)
        rel = abs(a7[1] - a6[1]) / max(a6[1], 1e-300)
        print("  z %g: K v6 %d v7 %d; tail v6 %.6e v7 %.6e (relative change %.1e); remainder bound beyond |z| + 120: %.1e" % (z, a6[0], a7[0], a6[1], a7[1], rel, rb))
        if not (a7[0] == a6[0] and rb <= 0.1 * tol and a7[1] <= tol and abs((a7[1] - a6[1]) - rb) <= 1e-9 * a7[1]):
            ok2 = False
    check(ok2, "z 300 to 1000: K is v6's at every tested z, the summation limit |z| + 120 is not extended (the remainder bound is below 0.1 tol through z = 1000: at most %.1e) and the tail is v6's plus that bound to rounding (v1 claimed a 1e-9 agreement; it is 4.4e-2 at 1000)" % rb_max)
    # (5c) z 1500: v6's K was one short
    z = 1500.0
    a6 = cheb_order_v6(z, tol); a7 = W10.cheb_order(z, tol)
    true6 = W10.cheb_tail(z, a6[0])[0]
    rb = W10.cheb_tail_bound(z, int(z + 120))
    print("  z %g: K v6 %d v7 %d; tail v6 %.6e v7 %.6e; v6's true tail at its K %.4e (cheb_tail); remainder bound beyond |z| + 120: %.1e" % (z, a6[0], a7[0], a6[1], a7[1], true6, rb))
    check(a7[0] == a6[0] + 1 and a7[1] <= tol and true6 > tol and rb > 0.1 * tol,
          "z 1500: K v7 %d = v6's %d + 1, the tail %.3e at or below tol, and v6's true tail at its K, %.3e, exceeded tol (v6 reported %.3e: the single term 2|J_%d| with the remainder beyond its limit appended as 0); the limit is extended because the remainder bound %.1e exceeds 0.1 tol" % (
              a7[0], a6[0], a7[1], true6, a6[1], a6[0] + 1, rb))
    # (6) the page module equals the lib
    try:
        import page_ops_v7 as P7
        ok3 = all(P7.cheb_order(z, tol) == W10.cheb_order(z, tol) for z in exact_z + [1500.0, 2000.0]) and P7.cheb_tail(2000.0, 2120) == W10.cheb_tail(2000.0, 2120)
        check(ok3, "page_ops_v7.cheb_order and cheb_tail equal w10_lib_v1's bitwise on the grid, at z = 1500 and at z = 2000")
    except ImportError:
        print("INFO page_ops_v7 not importable yet (run w10_apply_fixes_v2.py with W10_PAGE_DIR set): the lib stands for its block")
    # (7) mpmath
    try:
        import mpmath as mp
        mp.mp.dps = 30
        ref = float(mp.besselj(2121, 2000))
        sv = float(jv(2121, 2000.0))
        check(abs(sv - ref) <= 1e-8 * abs(ref), "mpmath cross-check: J_2121(2000) = %.6e (scipy %.6e, relative difference %.1e)" % (ref, sv, abs(sv - ref) / abs(ref)))
    except ImportError:
        print("INFO mpmath not installed: scipy's jv stands alone (its uniform asymptotics at order 2121 are accurate to about 1e-14)")
    # (8) a delivered Page json
    K200 = cheb_order_v6(200.0, tol)[0]
    print("INFO K(z = 200) = %d: below it every (K, tail) of the propagation is v6's bitwise; below 1000 the order is v6's at the tested z and only the printed tail bound grows" % K200)
    if a.page_json:
        acc = []
        find_kmax(json.load(open(a.page_json)), acc)
        if acc:
            km = int(max(acc))
            if km <= K200:
                check(True, "the delivered Page json %s: the largest Chebyshev order K_max %d is at or below K(z = 200) = %d: every propagated number and every printed tail bound of the record is v6's bitwise" % (a.page_json, km, K200))
            elif km < 1000:
                check(True, "the delivered Page json %s: K_max %d is above K(z = 200) = %d and below 1000: the order is v6's at the tested z (it can differ by one only where v6's tail at its K lay within the remainder bound, at most 3.8e-17, below tol), so no propagated number is expected to change; the printed tail bounds grow by the remainder bound" % (a.page_json, km, K200))
            else:
                check(False, "the delivered Page json %s: K_max %d is at or above 1000, beyond the tested range: the order may differ from v6's by one there (recompute with page_ops_v7 before quoting)" % (a.page_json, km))
        else:
            print("INFO no K_max entry found in %s" % a.page_json)
    print("test_w10_defect4_cheb_tail_v2: %s" % ("ALL PASS" if not FAIL else "%d FAIL" % len(FAIL)))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
