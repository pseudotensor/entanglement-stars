#!/usr/bin/env python
"""test_w10_defect4_cheb_tail_v3.py (busi W10): defect 4 of REVIEWA_REVIEW_web40i section 5. page_ops_v6.cheb_order sums the Bessel tail
to |z| + 120 and appends a zero: at z = 2000 it returns (K 2120, tail 0.0) while the true tail 2 sum_{j > 2120} |J_j(2000)| is 7.1e-14.
Gate: the bound reproduces 7.1e-14 at z = 2000 and stays below tolerance at the z used.
v3 (the W10 close, after readings A to E): the --page-json reader. v1 and v2 looked for the key "K_max" and reported "no K_max entry"
on the delivered Page jsons, whose entries carry the order as "cheb_K_max" (the box read 37 to 58 by hand, with accumulated tail
bounds at most 3.95e-12); v3 accepts K_max, cheb_K_max, Kmax, cheb_Kmax, lists of them, and a "K" or "order" under any key containing
"cheb", reads the tail bounds beside (keys containing "tail_bound", or "tail" under a "cheb" key), reports the smallest and the largest
order against K(z = 200) (at or below it every (K, tail) of the propagation is v6's bitwise) and 1000 (below it the order is v6's at
every tested z and only the printed tail bound grows), and fails when it finds no order at all (a silent reader is what v1 and v2
were); --page-json is repeatable. The assertions are v2's (the box's first round corrected them; the box's values in parentheses):
(1) v6's cheb_order (a verbatim copy here, and page_ops_v6 itself when W10_PAGE_DIR holds it) returns (2120, 0.0) at z = 2000;
(2) w10_lib_v1.cheb_tail(2000, 2120) returns the true tail (7.1477e-14, summed to j_max 3300) with the remainder bound below 1e-6 of it;
(3) cheb_order(2000, 1e-15) returns an order above 2120 with a positive tail at or below tol ((2132, 9.555e-16));
(4) the rigorous contour bound at K = 2120 is above the true tail (4.834e-12, the factor 67.6: the Debye prefactor) and below 1e-10;
(5) at the z used (0.3 to 200): (K, tail) bitwise v6's and the remainder bound beyond |z| + 120 below 1e-18 (at most 1.0e-37);
    at z = 300, 400, 700, 1000: K is v6's (371, 478, 793, 1105), the summation limit is not extended (the remainder bound below 0.1 tol)
    and the tail is v6's plus that bound; at z = 1500: K = v6's + 1 (1620 for 1619), the tail at or below tol, v6's true tail at its K
    (1.393e-15) above tol;
(6) page_ops_v7 (when W10_PAGE_DIR holds it) equals the lib bitwise on the grid, at 1500 and at 2000;
(7) mpmath, when installed, cross-checks scipy's J_2121(2000) (1.060801e-14);
(8) --page-json FILE (repeatable): the reader above.
Run: qi2 python test_w10_defect4_cheb_tail_v3.py [--page-json FILE ...]"""
import argparse
import json
import math
import os
import re
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


ORDER_KEY = re.compile(r"(cheb_)?k_?max|k_?max_?cheb|cheb_k")      # lower-cased keys: K_max, cheb_K_max, Kmax, cheb_Kmax, K_max_cheb, cheb_K


def numbers_of(v):
    """the numbers in v: a number, or the numbers of a list/tuple (bools excluded); [] otherwise."""
    if isinstance(v, bool):
        return []
    if isinstance(v, (int, float)):
        return [float(v)]
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return []


def find_orders(obj, acc, keys, tails, under_cheb=False):
    """v3: the Chebyshev orders of a delivered Page json: every value under a key matching ORDER_KEY at any depth, a "K" or "order" under
    any key containing "cheb", and beside them the tail bounds (keys containing "tail_bound", or "tail" under a "cheb" key)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if ORDER_KEY.fullmatch(lk) or (under_cheb and lk in ("k", "order")):
                nums = numbers_of(v)
                if nums:
                    acc.extend(nums); keys.add(str(k))
                    continue
            if "tail_bound" in lk or (under_cheb and lk == "tail"):
                tails.extend(numbers_of(v))
            find_orders(v, acc, keys, tails, under_cheb or ("cheb" in lk))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            find_orders(v, acc, keys, tails, under_cheb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page-json", action="append", default=None, help="a delivered Page json (repeatable): its Chebyshev orders read against K(z = 200) and 1000")
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
    # v2: z 300 to 1000: K unchanged, the limit not extended (the bound below 0.1 tol), the tail v6's plus the bound (the box: 371, 478, 793, 1105)
    ok2 = True; worst2 = 0.0
    for z in (300.0, 400.0, 700.0, 1000.0):
        a6 = cheb_order_v6(z, tol); a7 = W10.cheb_order(z, tol)
        rb = W10.cheb_tail_bound(z, int(z + 120))
        worst2 = max(worst2, rb)
        rel = abs(a7[1] - a6[1]) / max(a6[1], 1e-300)
        print("  z %g: K v6 %d v7 %d; tail v6 %.6e v7 %.6e (relative change %.1e); remainder bound beyond |z| + 120: %.1e" % (z, a6[0], a7[0], a6[1], a7[1], rel, rb))
        if a6[0] != a7[0] or rb > 0.1 * tol or abs(a7[1] - (a6[1] + rb)) > 1e-12 * max(a7[1], 1e-300):
            ok2 = False
    check(ok2, "z 300 to 1000: K is v6's at every tested z, the summation limit |z| + 120 is not extended (the remainder bound is below 0.1 tol through z = 1000: at most %.1e) and the tail is v6's plus that bound to rounding" % worst2)
    # v2: z 1500: K one above v6's, whose true tail at its K exceeded tol (the box: 1619 -> 1620, v6's true tail 1.3931e-15)
    z = 1500.0
    a6 = cheb_order_v6(z, tol); a7 = W10.cheb_order(z, tol)
    true6 = W10.cheb_tail(z, a6[0])[0]
    rb = W10.cheb_tail_bound(z, int(z + 120))
    print("  z 1500: K v6 %d v7 %d; tail v6 %.6e v7 %.6e; v6's true tail at its K %.4e (cheb_tail); remainder bound beyond |z| + 120: %.1e" % (a6[0], a7[0], a6[1], a7[1], true6, rb))
    check(a7[0] == a6[0] + 1 and 0.0 < a7[1] <= tol and true6 > tol and rb > 0.1 * tol,
          "z 1500: K v7 %d = v6's %d + 1, the tail %.3e at or below tol, and v6's true tail at its K, %.3e, exceeded tol (v6 reported %.3e: the single term 2|J_%d| with a zero appended for the rest); the limit is extended because the remainder bound %.1e exceeds 0.1 tol" % (
              a7[0], a6[0], a7[1], true6, a6[1], a6[0] + 1, rb))
    try:
        import page_ops_v7 as P7
        grid = exact_z + [1500.0, 2000.0]
        ok3 = all(P7.cheb_order(zz, tol) == W10.cheb_order(zz, tol) for zz in grid) and P7.cheb_tail(2000.0, 2120) == W10.cheb_tail(2000.0, 2120)
        check(ok3, "page_ops_v7.cheb_order and cheb_tail equal w10_lib_v1's bitwise on the grid, at z = 1500 and at z = 2000")
    except ImportError:
        print("INFO page_ops_v7 not importable (run the apply with W10_PAGE_DIR set): the lib stands for its block")
    try:
        import mpmath as mp
        mp.mp.dps = 30
        ref = float(mp.besselj(2121, 2000))
        sv = float(jv(2121, 2000.0))
        check(abs(sv - ref) <= 1e-8 * abs(ref), "mpmath cross-check: J_2121(2000) = %.6e (scipy %.6e, relative difference %.1e)" % (ref, sv, abs(sv - ref) / abs(ref)))
    except ImportError:
        print("INFO mpmath not installed: scipy's jv stands alone (its uniform asymptotics at order 2121 are accurate to about 1e-14)")
    K200 = W10.cheb_order(200.0, tol)[0]
    print("INFO K(z = 200) = %d: at or below it every (K, tail) of the propagation is v6's bitwise; below 1000 the order is v6's at the tested z and only the printed tail bound grows" % K200)
    for pj in (a.page_json or []):
        acc = []; keys = set(); tails = []
        try:
            J = json.load(open(pj))
        except Exception as ex:  # noqa: BLE001
            check(False, "the Page json %s is unreadable: %s" % (pj, ex))
            continue
        find_orders(J, acc, keys, tails)
        if not acc:
            check(False, "the delivered Page json %s: no Chebyshev order found (keys tried: K_max, cheb_K_max, Kmax, cheb_Kmax, a K or order under a cheb entry): the reader is wrong, not the json; report the json's chebyshev entry" % pj)
            continue
        km = max(acc); kmin = min(acc)
        if km <= K200:
            reading = "at or below K(z = 200) = %d: every printed tail bound is bitwise v6's too" % K200
        else:
            reading = "above K(z = 200) = %d: the printed tail bounds grow by the remainder bound (the propagated numbers do not)" % K200
        tb = (" accumulated tail bound at most %.3e (the per-step bounds summed over the steps; each step's bound is at or below tol);" % max(tails)) if tails else ""
        check(km < 1000, "the delivered Page json %s: Chebyshev orders %d to %d over %d entries (keys %s);%s the largest is below 1000, so v6's appended zero never changed an order (no propagated number changes); %s" % (
            pj, int(kmin), int(km), len(acc), sorted(keys), tb, reading))
    print("test_w10_defect4_cheb_tail_v3: %s" % ("ALL PASS" if not FAIL else "%d FAIL" % len(FAIL)))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
