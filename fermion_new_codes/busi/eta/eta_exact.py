#!/usr/bin/env python3
"""eta_exact.py -- the exact [100] and [111] planar coefficients of the massless sine-Dirac sea and the
exact axial finite-mass coefficient b_100, each by 2 independent numerical evaluations.

Route 1 ([100]): eta_100 = int_0^2 rho(s) s_4(sqrt(s + y)) ds (1-dimensional, mpmath tanh-sinh).
Route 2 ([100]): the 2-dimensional transverse integral of s_4(sqrt(sin^2 x + sin^2 y)) over the pi-zone in
                 polar coordinates about the node (no rho, no 1-dimensional reduction).
Route 1 ([111]): eta_111 = (1/sqrt 3) int_0^3 p_3(z) F(z) dz (1-dimensional, mpmath tanh-sinh).
Route 2 ([111]): the 2-dimensional transverse integral of F(z(phi, psi)) in polar coordinates (no p_3).
b_100:           int_0^2 rho(s) [f'(s) + 1/(3s)] ds - (1 + log 8)/(3 pi) with the analytic derivative.
All elliptic integrals through the arithmetic-geometric mean with explicit complements.
Requires mpmath (box). --double runs scipy-quad versions of the 1-dimensional routes (about 11 digits).
"""
import os
import sys
import argparse
import json
import math
import time


def _early_threads():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--threads', type=int, default=1)
    a, _ = pre.parse_known_args()
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[v] = str(a.threads)
    return a.threads


THREADS = _early_threads()
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eta_common as ec  # noqa: E402

try:
    import mpmath
    from mpmath import mp
    HAVE_MP = True
except ImportError:
    HAVE_MP = False


def need_mp():
    if not HAVE_MP:
        raise SystemExit("eta_exact.py: mpmath is required for the exact routes at 12 or more digits "
                         "(pip install mpmath, or run this script on the box); --double gives about 11 digits.")


# ---------------------------------------------------------------------------------------------- mp pieces
def mpK_compl(c):
    """K(1 - c) in the parameter convention through the AGM."""
    return mp.pi / (2 * mp.agm(1, mp.sqrt(c)))


def rho100_mp(s):
    return 2 / mp.pi ** 2 * mpK_compl((1 - s) ** 2)


def s4_mp(M):
    """Closed form of the [100] chain entropy at transverse gap M > 0."""
    s = M * M
    t = 1 / (1 + s)
    q = s / (1 + s)
    Kt = mpK_compl(q)
    Kq = mpK_compl(t)
    return (mp.log(16 * (1 + s) ** 2 / s) + (4 / mp.pi) * Kt * Kq * (1 - s) / (1 + s)) / 6


def s4_ladder_mp(M, cut=60):
    s = M * M
    t = 1 / (1 + s)
    q = s / (1 + s)
    eps = mp.pi * mp.agm(1, mp.sqrt(q)) / mp.agm(1, mp.sqrt(t))
    jmax = int(mp.ceil(cut / eps))
    tot = mp.mpf(0)
    for j in range(jmax + 1):
        x = (2 * j + 1) * eps
        tot += mp.log(1 + mp.exp(-x)) + x / (1 + mp.exp(x))
    return 4 * tot


def fprime100_mp(s):
    """Analytic derivative of f(s) = s_4(sqrt s)."""
    t = 1 / (1 + s)
    q = s / (1 + s)
    Kt = mpK_compl(q)
    Kq = mpK_compl(t)
    Et = mp.ellipe(t)
    Eq = mp.ellipe(q)
    kappa = Kt * Kq
    kappap = (Kt * Eq - Et * Kq + (q - t) * kappa) / (2 * s)
    sigma = (1 - s) / (1 + s)
    return (2 / (1 + s) - 1 / s + (4 / mp.pi) * (kappap * sigma - 2 * kappa / (1 + s) ** 2)) / 6


def eta100_1d(y=0.0, dps=25):
    need_mp()
    mp.dps = dps
    yy = mp.mpf(y)
    f = lambda s: rho100_mp(s) * s4_mp(mp.sqrt(s + yy))
    return mp.quad(f, [0, 1, 2])


def b100_exact(dps=30):
    need_mp()
    mp.dps = dps
    g = lambda s: rho100_mp(s) * (fprime100_mp(s) + 1 / (3 * s))
    I = mp.quad(g, [0, 1, 2])
    return I - (1 + mp.log(8)) / (3 * mp.pi)


def p3_mp(z):
    """Density of the length of 3 unit planar steps."""
    if z <= 0:
        return mp.mpf(0)
    if z < 1:
        A = (1 + z) ** 3 * (3 - z)
        comp = (1 - z) ** 3 * (z + 3) / A
        return 4 * z / (mp.pi ** 2 * mp.sqrt(A)) * mpK_compl(comp)
    if z >= 3:
        return mp.mpf(0)
    comp = (z - 1) ** 3 * (z + 3) / (16 * z)
    return mp.sqrt(z) / mp.pi ** 2 * mpK_compl(comp)


def p3_integral_mp(z):
    """Check representation: (2z/pi^2) int dt / sqrt(t(4-t)((z+1)^2-t)(t-(z-1)^2)) between the middle roots."""
    lo = (z - 1) ** 2
    hi = min(mp.mpf(4), (z + 1) ** 2)
    # builder's repair: the product is positive inside (lo, hi); at nodes within roundoff of an endpoint it can
    # come out slightly negative and mp.sqrt then returns a complex number (T07 failed on float(mpc)).
    f = lambda t: 1 / mp.sqrt(abs(t * (4 - t) * ((z + 1) ** 2 - t) * (t - (z - 1) ** 2)))
    return 2 * z / mp.pi ** 2 * mp.quad(f, [lo, hi])


def F111_mp(z, cut=60):
    if z <= 0:
        return 2 * mp.log(2)
    u = mp.sqrt((3 + z) / 2)
    v = mp.sqrt((3 - z) / 2)
    k = (u - v) / (u + v)
    k2c = 4 * u * v / (u + v) ** 2
    eps = mp.pi * mp.agm(1, mp.sqrt(k2c)) / mp.agm(1, k)
    jmax = int(mp.ceil(cut / eps))
    tot = mp.mpf(0)
    for j in range(1, jmax + 1):
        x = j * eps
        tot += mp.log(1 + mp.exp(-x)) + x / (1 + mp.exp(x))
    return 2 * mp.log(2) + 4 * tot


def eta111_1d(dps=25):
    need_mp()
    mp.dps = dps
    f = lambda z: p3_mp(z) * F111_mp(z)
    return mp.quad(f, [0, 1, 3]) / mp.sqrt(3)


def z111_mp(x, y):
    w = 3 + 2 * mp.cos(2 * x) + 2 * mp.cos(2 * y) + 2 * mp.cos(2 * x - 2 * y)
    return mp.sqrt(w) if w > 0 else mp.mpf(0)


def torus_node_integral_mp(f, sectors, mults, dps=20, verbose=False):
    """(1/pi^2) int over [-pi/2, pi/2]^2 of f(x, y), f pi-periodic with a log singularity at the origin only:
    polar coordinates about the origin, the square cut into the 8 angular sectors [k pi/4, (k+1) pi/4]."""
    mp.dps = dps
    half = mp.pi / 2
    total = mp.mpf(0)
    err_total = mp.mpf(0)
    for k, mult in zip(sectors, mults):
        ta = k * mp.pi / 4
        tb = (k + 1) * mp.pi / 4
        t0 = time.time()

        def outer(t):
            ct = mp.cos(t)
            st = mp.sin(t)
            R = half / max(abs(ct), abs(st))
            return mp.quad(lambda r: r * f(r * ct, r * st), [0, R])

        val, err = mp.quad(outer, [ta, tb], error=True)
        total += mult * val
        err_total += mult * err
        if verbose:
            print(f"  sector {k}: value {mp.nstr(val, 20)} error estimate {mp.nstr(err, 3)} "
                  f"multiplicity {mult} ({time.time() - t0:.0f} s)", flush=True)
    return total / mp.pi ** 2, err_total / mp.pi ** 2


def eta100_2d(dps=20, all_sectors=False, verbose=False):
    need_mp()
    f = lambda x, y: s4_mp(mp.sqrt(mp.sin(x) ** 2 + mp.sin(y) ** 2))
    sectors, mults = (list(range(8)), [1] * 8) if all_sectors else ([0], [8])
    return torus_node_integral_mp(f, sectors, mults, dps=dps, verbose=verbose)


def F111_deficit_mp(d, cut=60):
    """F(z) with the deficit d = 9 - z^2 given directly (builder's repair for the polar route: near the torus
    origin z = sqrt(9 - d) rounds to 3 at working precision, v = 0 and eps divided by zero (T06 failed);
    v^2 = (3 - z)/2 = d/(2(3 + z)) has no cancellation). d = 0 is the node itself, a set of measure zero."""
    if d <= 0:
        return mp.mpf(0)
    z = mp.sqrt(9 - d)
    if z <= 0:
        return 2 * mp.log(2)
    u = mp.sqrt((3 + z) / 2)
    v = mp.sqrt(d / (2 * (3 + z)))
    k = (u - v) / (u + v)
    k2c = 4 * u * v / (u + v) ** 2
    eps = mp.pi * mp.agm(1, mp.sqrt(k2c)) / mp.agm(1, k)
    jmax = int(mp.ceil(cut / eps))
    tot = mp.mpf(0)
    for j in range(1, jmax + 1):
        x = j * eps
        tot += mp.log(1 + mp.exp(-x)) + x / (1 + mp.exp(x))
    return 2 * mp.log(2) + 4 * tot


def eta111_2d(dps=20, all_sectors=False, verbose=False):
    need_mp()
    # 9 - w = 4 (sin^2 x + sin^2 y + sin^2(x - y)) exactly, with w the z^2 of z111_mp
    f = lambda x, y: F111_deficit_mp(4 * (mp.sin(x) ** 2 + mp.sin(y) ** 2 + mp.sin(x - y) ** 2))
    sectors, mults = (list(range(8)), [1] * 8) if all_sectors else ([0, 2], [4, 4])
    val, err = torus_node_integral_mp(f, sectors, mults, dps=dps, verbose=verbose)
    return val / mp.sqrt(3), err / mp.sqrt(3)


# ---------------------------------------------------------------------------------------------- double fallbacks
def eta100_1d_double(y=0.0):
    from scipy.integrate import quad
    f = lambda s: float(ec.rho100(s)) * ec.s4_axial(math.sqrt(s + y))
    v1, e1 = quad(f, 0.0, 1.0, limit=400, epsabs=1e-13, epsrel=1e-13)
    v2, e2 = quad(f, 1.0, 2.0, limit=400, epsabs=1e-13, epsrel=1e-13)
    return v1 + v2, e1 + e2


def p3_double(z):
    if z <= 0.0 or z >= 3.0:
        return 0.0
    if z < 1.0:
        A = (1 + z) ** 3 * (3 - z)
        comp = (1 - z) ** 3 * (z + 3) / A
        return 4 * z / (math.pi ** 2 * math.sqrt(A)) * float(ec.ellipK_compl(comp))
    comp = (z - 1) ** 3 * (z + 3) / (16 * z)
    return math.sqrt(z) / math.pi ** 2 * float(ec.ellipK_compl(comp))


def eta111_1d_double():
    from scipy.integrate import quad
    f = lambda z: p3_double(z) * ec.F111(z)
    v1, e1 = quad(f, 0.0, 1.0, limit=400, epsabs=1e-13, epsrel=1e-13)
    v2, e2 = quad(f, 1.0, 3.0, limit=400, epsabs=1e-13, epsrel=1e-13)
    return (v1 + v2) / math.sqrt(3.0), (e1 + e2) / math.sqrt(3.0)


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--which', nargs='+', default=['all'], help='subset of: 100 111 b100 mass all')
    ap.add_argument('--y', nargs='*', type=float, default=[0.01, 0.0225, 0.04, 0.09],
                    help='values of y = (ma)^2 for the exact eta_100(y)')
    ap.add_argument('--dps', type=int, default=25, help='mpmath digits for the 1-dimensional routes')
    ap.add_argument('--dps2d', type=int, default=20, help='mpmath digits for the 2-dimensional routes')
    ap.add_argument('--all-sectors', action='store_true', help='do not use the sector symmetries (check)')
    ap.add_argument('--double', action='store_true', help='scipy-quad versions of the 1-dimensional routes only')
    ap.add_argument('--out', default='eta_exact.json')
    args = ap.parse_args()
    which = set(args.which)
    if 'all' in which:
        which = {'100', '111', 'b100', 'mass'}
    out = dict(conventions='nats; one 4-component sine-Dirac field; one face; per unit area a^2; a = 1',
               threads=THREADS, dps=args.dps, dps2d=args.dps2d, double=args.double, results={})
    R = out['results']

    if args.double:
        if '100' in which:
            t0 = time.time()
            v, e = eta100_1d_double()
            R['eta100_1d_double'] = dict(value=v, quad_error=e, seconds=time.time() - t0,
                                         diff_target=v - ec.TARGET_ETA100)
            print(f"eta100 1d (scipy quad) = {v!r}  quad error {e:.1e}  diff from target {v - ec.TARGET_ETA100:.2e}")
        if '111' in which:
            t0 = time.time()
            v, e = eta111_1d_double()
            R['eta111_1d_double'] = dict(value=v, quad_error=e, seconds=time.time() - t0,
                                         diff_target=v - ec.TARGET_ETA111)
            print(f"eta111 1d (scipy quad) = {v!r}  quad error {e:.1e}  diff from target {v - ec.TARGET_ETA111:.2e}")
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=1)
        return

    need_mp()
    if '100' in which:
        t0 = time.time()
        v1 = eta100_1d(0.0, dps=args.dps)
        s1 = time.time() - t0
        print(f"eta100 route 1 (1d, {args.dps} digits): {mp.nstr(v1, 20)}   ({s1:.1f} s)", flush=True)
        t0 = time.time()
        v2, e2 = eta100_2d(dps=args.dps2d, all_sectors=args.all_sectors, verbose=True)
        s2 = time.time() - t0
        print(f"eta100 route 2 (2d polar, {args.dps2d} digits): {mp.nstr(v2, 20)}  error estimate {mp.nstr(e2, 3)}  ({s2:.0f} s)")
        print(f"  routes differ by {mp.nstr(v2 - v1, 3)}; route 1 minus target {mp.nstr(v1 - ec.TARGET_ETA100, 3)}")
        R['eta100'] = dict(route1_1d=str(v1), route2_2d=str(v2), route2_error_estimate=str(e2),
                           route_difference=float(v2 - v1), diff_target_route1=float(v1 - ec.TARGET_ETA100),
                           seconds_route1=s1, seconds_route2=s2, target=ec.TARGET_ETA100)
    if '111' in which:
        t0 = time.time()
        mp.dps = args.dps
        norm = mp.quad(p3_mp, [0, 1, 3])
        v1 = eta111_1d(dps=args.dps)
        s1 = time.time() - t0
        print(f"p3 normalization: {mp.nstr(norm, 20)};  F(0) = {mp.nstr(F111_mp(mp.mpf(0)), 20)} vs 2 log 2 = {mp.nstr(2*mp.log(2), 20)}")
        print(f"eta111 route 1 (1d, {args.dps} digits): {mp.nstr(v1, 20)}   ({s1:.1f} s)", flush=True)
        t0 = time.time()
        v2, e2 = eta111_2d(dps=args.dps2d, all_sectors=args.all_sectors, verbose=True)
        s2 = time.time() - t0
        print(f"eta111 route 2 (2d polar, {args.dps2d} digits): {mp.nstr(v2, 20)}  error estimate {mp.nstr(e2, 3)}  ({s2:.0f} s)")
        print(f"  routes differ by {mp.nstr(v2 - v1, 3)}; route 1 minus target {mp.nstr(v1 - ec.TARGET_ETA111, 3)}")
        R['eta111'] = dict(route1_1d=str(v1), route2_2d=str(v2), route2_error_estimate=str(e2),
                           route_difference=float(v2 - v1), diff_target_route1=float(v1 - ec.TARGET_ETA111),
                           p3_normalization=str(norm), seconds_route1=s1, seconds_route2=s2, target=ec.TARGET_ETA111)
    if 'b100' in which:
        t0 = time.time()
        b = b100_exact(dps=max(args.dps, 30))
        sb = time.time() - t0
        print(f"b100 exact (analytic derivative, {max(args.dps,30)} digits): {mp.nstr(b, 20)}  diff target {mp.nstr(b - ec.TARGET_B100, 3)}  ({sb:.1f} s)")
        mp.dps = args.dps
        e0 = eta100_1d(0.0, dps=args.dps)
        fd = []
        for y in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
            ey = eta100_1d(y, dps=args.dps)
            est = (ey - e0) / y - mp.log(y) / (3 * mp.pi)
            fd.append(dict(y=y, eta=str(ey), b_effective=float(est)))
            print(f"  finite difference at y = {y:g}: b_effective = {mp.nstr(est, 12)} (approaches b100 like y log y)")
        R['b100'] = dict(value=str(b), diff_target=float(b - ec.TARGET_B100), seconds=sb,
                         finite_difference_sequence=fd, target=ec.TARGET_B100)
    if 'mass' in which and args.y:
        rows = []
        e0 = eta100_1d(0.0, dps=args.dps)
        for y in args.y:
            t0 = time.time()
            ey = eta100_1d(y, dps=args.dps)
            r = (ey - e0 - y * mp.log(y) / (3 * mp.pi)) / y
            rows.append(dict(y=y, ma=math.sqrt(y), eta100=str(ey), eta100_float=float(ey), r=float(r),
                             seconds=time.time() - t0))
            print(f"eta100(y = {y:g}, ma = {math.sqrt(y):.4f}) = {mp.nstr(ey, 18)}   r(y) = {mp.nstr(r, 10)}")
        R['eta100_massive'] = dict(eta100_0=str(e0), rows=rows)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
