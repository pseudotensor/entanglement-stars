#!/usr/bin/env python3
"""eta_tests.py -- acceptance tests of the planar-coefficient code (PASS or FAIL per test, with numbers and
tolerances). Default: the pre-sweep tests T01 to T20 of ETA_spec_v1.md section 10 (they must all pass before
the sweep). --post-sweep --table eta_table.tsv: the checks on the finished table.
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
from scipy.integrate import quad  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eta_common as ec  # noqa: E402
import eta_direction as ed  # noqa: E402
import eta_bounds as eb  # noqa: E402

try:
    import eta_exact as ex
    HAVE_MP = ex.HAVE_MP
except Exception:
    HAVE_MP = False

RESULTS = []


def record(name, passed, detail):
    RESULTS.append(dict(name=name, passed=bool(passed), detail=detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)


# ---------------------------------------------------------------------------------------------- exact formulas
def t01_s4_ladder_vs_closed():
    worst = 0.0
    for M in (0.01, 0.1, 0.5, 1.0, 2.0, 5.0):
        worst = max(worst, abs(ec.s4_axial(M, ladder=True) - ec.s4_axial(M)))
    record('T01 s4 ladder vs closed form', worst < 1e-12, f"max |diff| = {worst:.2e} (tol 1e-12)")


def t02_s4_asymptotics():
    slope = -(ec.s4_axial(2e-4) - ec.s4_axial(1e-4)) / math.log(2.0)
    e1 = abs(ec.s4_axial(100.0) * 4e4 / (1 + math.log(16e4)) - 1.0)
    e2 = abs(ec.s4_axial(200.0) * 16e4 / (1 + math.log(64e4)) - 1.0)
    ok = abs(slope - 2.0 / 3.0) < 1e-6 and e1 <= 5e-4 and e2 < e1
    record('T02 s4 asymptotics', ok, f"small-M slope {slope:.9f} vs 2/3 (tol 1e-6); large-M errors {e1:.2e} (tol 5e-4), {e2:.2e} (decreasing)")


def t03_rho_normalization_and_moment():
    f = lambda s: float(ec.rho100(s))
    n = quad(f, 0.0, 1.0, limit=400, epsabs=1e-12)[0] + quad(f, 1.0, 2.0, limit=400, epsabs=1e-12)[0]
    g = lambda s: float(ec.rho100(s)) * math.exp(-s)
    m1 = quad(g, 0.0, 1.0, limit=400, epsabs=1e-12)[0] + quad(g, 1.0, 2.0, limit=400, epsabs=1e-12)[0]
    N = 128
    k = math.pi * (np.arange(N) + 0.5) / N
    KY, KZ = np.meshgrid(k, k, indexing='ij')
    m2 = float(np.mean(np.exp(-(np.sin(KY) ** 2 + np.sin(KZ) ** 2))))
    ok = abs(n - 1.0) < 1e-9 and abs(m1 - m2) < 1e-10
    record('T03 rho normalization and smooth moment', ok, f"int rho = {n:.12f} (tol 1e-9); moment 1d {m1:.13f} vs torus {m2:.13f} (tol 1e-10)")


def t04_resolvent_identity():
    worst = 0.0
    for y in (0.1, 1.0):
        f = lambda s: float(ec.rho100(s)) / (s + y)
        lhs = quad(f, 0.0, 1.0, limit=400, epsabs=1e-12)[0] + quad(f, 1.0, 2.0, limit=400, epsabs=1e-12)[0]
        comp = 1.0 - 1.0 / (1.0 + y) ** 2
        rhs = 2.0 * float(ec.ellipK_compl(comp)) / (math.pi * (1.0 + y))
        worst = max(worst, abs(lhs - rhs))
    record('T04 square-lattice resolvent identity', worst < 1e-10, f"max |diff| = {worst:.2e} (tol 1e-10)")


def t05_eta100_exact(args):
    if not HAVE_MP:
        record('T05 eta100 exact (2 routes)', False, 'mpmath missing: run on the box or install mpmath')
        return
    t0 = time.time()
    v1 = float(ex.eta100_1d(0.0, dps=25))
    v2, e2 = ex.eta100_2d(dps=args.dps2d)
    v2 = float(v2)
    ok = abs(v1 - v2) < 1e-12 and abs(v1 - ec.TARGET_ETA100) < 1e-12
    record('T05 eta100 exact (2 routes)', ok, f"1d {v1!r}, 2d {v2!r}, |1d-2d| = {abs(v1-v2):.1e}, |1d-target| = {abs(v1-ec.TARGET_ETA100):.1e} (tol 1e-12) [{time.time()-t0:.0f} s]")


def t06_eta111_exact(args):
    if not HAVE_MP:
        record('T06 eta111 exact (2 routes)', False, 'mpmath missing')
        return
    t0 = time.time()
    v1 = float(ex.eta111_1d(dps=25))
    v2, e2 = ex.eta111_2d(dps=args.dps2d)
    v2 = float(v2)
    ok = abs(v1 - v2) < 1e-12 and abs(v1 - ec.TARGET_ETA111) < 5e-13
    record('T06 eta111 exact (2 routes)', ok, f"1d {v1!r}, 2d {v2!r}, |1d-2d| = {abs(v1-v2):.1e}, |1d-target| = {abs(v1-ec.TARGET_ETA111):.1e} (tol 5e-13) [{time.time()-t0:.0f} s]")


def t07_p3():
    if not HAVE_MP:
        record('T07 p3 density', False, 'mpmath missing')
        return
    mp = ex.mp
    mp.dps = 25
    norm = float(mp.quad(ex.p3_mp, [0, 1, 3]))
    worst = 0.0
    for z in (0.3, 0.9, 1.1, 2.5):
        worst = max(worst, abs(float(ex.p3_mp(mp.mpf(z)) - ex.p3_integral_mp(mp.mpf(z)))))
    F0 = float(ex.F111_mp(mp.mpf(0)))
    ok = abs(norm - 1.0) < 1e-12 and worst < 1e-8 and abs(F0 - 2 * math.log(2)) < 1e-15
    record('T07 p3 normalization and quartic form; F(0)', ok, f"int p3 = {norm:.14f} (tol 1e-12); max |closed - quartic| = {worst:.1e} (tol 1e-8); F(0) - 2log2 = {F0 - 2*math.log(2):.1e}")


# ---------------------------------------------------------------------------------------------- chain solver
def t08_F111_vs_chain(args):
    p = (1, 1, 1)
    r, s, G = ec.plane_basis(p)
    pn = math.sqrt(3.0)
    rng = np.random.default_rng(7)
    worst = 0.0
    details = []
    pts = [(1.0, 1.2), (0.4, 2.6), (1.9, 0.7), (math.pi / 3 + 0.05, 2 * math.pi / 3 - 0.05)]
    for phi, psi in pts:
        c = ec.chain_phases(r, s, phi, psi)
        z = ec.z111_from_phases(c)
        D = ec.chain_gap(p, r, s, phi, psi, 0.0)
        L = args.Lmin + int(math.ceil(14.0 * pn / D))
        S = ec.slab_entropy(p, r, s, phi, psi, 0.0, L)
        Fz = ec.F111(z)
        worst = max(worst, abs(S - Fz))
        details.append(f"z={z:.3f} L={L} |S-F|={abs(S-Fz):.1e}")
    record('T08 F(z) vs direct [111] slab', worst < 1e-9, f"max |diff| = {worst:.1e} (tol 1e-9); " + '; '.join(details))


def t09_s4_vs_chain(args):
    p = (1, 0, 0)
    r, s, G = ec.plane_basis(p)
    worst = 0.0
    details = []
    for m, ky, kz in ((0.0, 0.37, 0.82), (0.2, 0.11, 0.29), (0.7, 1.1, 0.4)):
        M = math.sqrt(m * m + math.sin(ky) ** 2 + math.sin(kz) ** 2)
        D = ec.chain_gap(p, r, s, ky, kz, m)
        L = args.Lmin + int(math.ceil(14.0 / D))
        S = ec.slab_entropy(p, r, s, ky, kz, m, L)
        worst = max(worst, abs(S - ec.s4_axial(M)))
        details.append(f"(m,ky,kz)=({m},{ky},{kz}) M={M:.4f} gap={D:.4f} L={L} |S-s4|={abs(S-ec.s4_axial(M)):.1e}")
    # 4-component massless slab against the Weyl slab for a generic chain
    q = (3, 2, 1)
    r2, s2, G2 = ec.plane_basis(q)
    phi, psi = 0.9, 2.1
    D = ec.chain_gap(q, r2, s2, phi, psi, 0.0)
    L = args.Lmin + int(math.ceil(12.0 * math.sqrt(14.0) / D))
    Sw = ec.slab_entropy(q, r2, s2, phi, psi, 0.0, L)
    S4 = ec.slab_entropy(q, r2, s2, phi, psi, 0.0, L, force4=True)
    ok = worst < 1e-9 and abs(Sw - S4) < 1e-10
    record('T09 s4 vs direct [100] slab; Weyl vs 4-component', ok, f"max |diff| = {worst:.1e} (tol 1e-9); |Weyl - 4comp| = {abs(Sw-S4):.1e} (tol 1e-10); " + '; '.join(details))


def t10_deficit_scan(args):
    p = (1, 0, 0)
    r, s, G = ec.plane_basis(p)
    M = 0.05
    ky = math.asin(M)
    exact = ec.s4_axial(M)
    defs = {}
    for c in (4, 6, 8, 10, 12, 16):
        L = int(math.ceil(c / M))
        S = ec.slab_entropy(p, r, s, ky, 0.0, 0.0, L)
        defs[c] = S - exact
    ok = abs(defs[10]) < 1e-7 and abs(defs[12]) < abs(defs[10]) < abs(defs[8])
    record('T10 slab deficit scan at M = 0.05', ok, 'deficits S_slab(L = c/M) - s4: ' + ', '.join(f"c={c}: {d:+.2e}" for c, d in defs.items()) + ' (c=10 below 1e-7 and decreasing)')


def t11_integer_geometry():
    rng = np.random.default_rng(911)
    count = 0
    bad = 0
    tested = []
    while count < 3000:
        q = tuple(int(v) for v in rng.integers(-50, 51, 3))
        if q == (0, 0, 0) or not ec.is_primitive(q):
            continue
        tested.append(q)
        count += 1
    tested += ec.direction_orbits(10) + [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1), (-1, 0, 0), (0, -1, 0)]
    for q in tested:
        r, s, G = ec.plane_basis(q)
        g11, g12, g22 = ec.gram_int(q, r, s)
        pp = q[0] ** 2 + q[1] ** 2 + q[2] ** 2
        if ec.det3(q, r, s) != 1 or g11 * g22 - g12 * g12 != pp or not (g11 <= g22 and 2 * abs(g12) <= g11):
            bad += 1
    record('T11 unimodular completion and reduction', bad == 0, f"{len(tested)} normals, {bad} failures (det = 1, det(G)|p|^2 = 1, reduced)")


def t12_symmetries(args):
    p = (3, 2, 1)
    r, s, G = ec.plane_basis(p)
    pn = math.sqrt(14.0)
    worst = 0.0
    for m in (0.0, 0.3):
        for phi, psi in ((0.7, 1.9), (2.3, 0.4)):
            D = ec.chain_gap(p, r, s, phi, psi, m)
            L = args.Lmin + int(math.ceil(12.0 * pn / D))
            S0 = ec.slab_entropy(p, r, s, phi, psi, m, L)
            for a, b in ((phi + math.pi, psi), (phi, psi + math.pi), (-phi, -psi)):
                worst = max(worst, abs(ec.slab_entropy(p, r, s, a, b, m, L) - S0))
    record('T12 pi-periodicity and inversion of the chain entropy', worst < 1e-10, f"p = (3,2,1), m = 0 and 0.3: max |diff| = {worst:.1e} (tol 1e-10)")


def t13_reference_integral():
    worst = 0.0
    details = []
    for p, y in (((1, 0, 0), 0.0), ((3, 2, 1), 0.0), ((3, 2, 1), 0.5)):
        r, s, G = ec.plane_basis(p)
        R, Rerr = ec.reference_log_integral(G, y)
        vals = {}
        for N in (512, 1024, 2048):
            k = math.pi * (np.arange(N) + 0.5) / N
            PH, PS = np.meshgrid(k, k, indexing='ij')
            vals[N] = float(np.mean(np.log(ec.node_model(G, y, PH, PS))))
        ra = (4 * vals[1024] - vals[512]) / 3.0
        rb = (4 * vals[2048] - vals[1024]) / 3.0
        worst = max(worst, abs(rb - R))
        details.append(f"p={p} y={y}: closed {R:.11f} brute {rb:.11f} (|rb-ra| {abs(rb-ra):.1e}) quad err {Rerr:.1e}")
    record('T13 reference integral vs brute force', worst < 1e-8, f"max |diff| = {worst:.1e} (tol 1e-8); " + '; '.join(details))


def _direction_levels(p, mass, args, levels=(0, 1)):
    recs = {}
    for lv in levels:
        recs[lv] = ed.run_level(p, mass, lv, h0=args.h0, cL=args.cL, Lmin=args.Lmin, device=args.device, verbose=False)
    return recs


def t14_direction_100(args):
    t0 = time.time()
    recs = _direction_levels((1, 0, 0), 0.0, args)
    c = ed.combine_levels(recs)
    e0, e1 = recs[0]['eta'] - ec.TARGET_ETA100, recs[1]['eta'] - ec.TARGET_ETA100
    r0, r1 = recs[0]['eta_raw_unsubtracted'] - ec.TARGET_ETA100, recs[1]['eta_raw_unsubtracted'] - ec.TARGET_ETA100
    ratio = e0 / e1 if e1 != 0 else float('inf')
    covers = abs(c['value'] - ec.TARGET_ETA100) <= c['hw']
    ok = abs(e1) <= 5e-6 and abs(e0) <= 1e-4 and covers and not (abs(e1) > 1e-7 and ratio < 6.0)
    record('T14 directional solver on [100]', ok,
           f"eta0-exact {e0:+.2e}, eta1-exact {e1:+.2e} (tol 5e-6), ratio {ratio:.1f} (expected 8 to 16; fail below 6); "
           f"unsubtracted errors {r0:+.2e}, {r1:+.2e}; value {c['value']:.10f} +/- {c['hw']:.1e} covers exact: {covers}; "
           f"chain_err {c['chain_err']:.1e}; N {recs[1]['N1']}x{recs[1]['N2']} n_max {recs[1]['n_max']} [{time.time()-t0:.0f} s]")


def t15_direction_111_110(args):
    t0 = time.time()
    recs = _direction_levels((1, 1, 1), 0.0, args)
    c = ed.combine_levels(recs)
    e0, e1 = recs[0]['eta'] - ec.TARGET_ETA111, recs[1]['eta'] - ec.TARGET_ETA111
    ratio = e0 / e1 if e1 != 0 else float('inf')
    covers = abs(c['value'] - ec.TARGET_ETA111) <= c['hw']
    ok = abs(e1) <= 5e-6 and abs(e0) <= 1e-4 and covers and not (abs(e1) > 1e-7 and ratio < 6.0)
    record('T15a directional solver on [111]', ok,
           f"eta0-exact {e0:+.2e}, eta1-exact {e1:+.2e} (tol 5e-6), ratio {ratio:.1f}; value {c['value']:.10f} +/- {c['hw']:.1e} covers exact: {covers}; "
           f"N {recs[1]['N1']}x{recs[1]['N2']} n_max {recs[1]['n_max']} [{time.time()-t0:.0f} s]")
    t0 = time.time()
    recs = _direction_levels((1, 1, 0), 0.0, args)
    c = ed.combine_levels(recs)
    j7 = ec.SOLVE_LANE_J7[(1, 1, 0)]
    d = c['value'] - j7
    ok = abs(d) <= 3e-6 + c['hw']
    record('T15b directional solver on [110] vs solve lane', ok,
           f"value {c['value']:.10f} +/- {c['hw']:.1e}; J7 {j7:.8f} diff {d:+.2e} (tol 3e-6 + hw); P23 (information) {ec.P23_INFO[(1,1,0)]:.10f} diff {c['value']-ec.P23_INFO[(1,1,0)]:+.2e} [{time.time()-t0:.0f} s]")


def t16_massive_100(args):
    if not HAVE_MP:
        record('T16 massive solver on [100]', False, 'mpmath missing (exact eta100(y) needed)')
        return
    t0 = time.time()
    ok_all = True
    details = []
    for m in (0.1, 0.3):
        exact = float(ex.eta100_1d(m * m, dps=20))
        recs = _direction_levels((1, 0, 0), m, args)
        c = ed.combine_levels(recs)
        e1 = recs[1]['eta'] - exact
        covers = abs(c['value'] - exact) <= c['hw']
        ok = abs(e1) <= 5e-6 and covers
        ok_all = ok_all and ok
        details.append(f"ma={m}: exact {exact:.10f}, eta1-exact {e1:+.2e}, value {c['value']:.10f} +/- {c['hw']:.1e} covers {covers}, n_max {recs[1]['n_max']}")
    record('T16 massive solver on [100]', ok_all, '; '.join(details) + f" (tol 5e-6) [{time.time()-t0:.0f} s]")


def t17_b_fit_exact():
    if not HAVE_MP:
        record('T17 b fit on exact [100] data', False, 'mpmath missing')
        return
    e0 = float(ex.eta100_1d(0.0, dps=20))
    masses = [0.1, 0.15, 0.2, 0.3]
    ys = [m * m for m in masses]
    etas = [float(ex.eta100_1d(y, dps=20)) for y in ys]
    fit = ed.fit_b(ys, etas, [1e-12] * 4, e0, 1e-12)
    d = fit['b'] - ec.TARGET_B100
    record('T17 b fit on exact [100] data', abs(d) < 5e-3, f"b_fit {fit['b']:.6f} vs b100 {ec.TARGET_B100:.10f}: diff {d:+.2e} (tol 5e-3; this is the fit's truncation); resid {fit['resid_max']:.1e}")


def t18_bounds_known_bodies(args):
    details = []
    # cube: 6 face normals, eta = 1/2 -> P = [-1/2, 1/2]^3, mean support 3/4
    U = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    V, hull = eb.polytope_from_upper(U, np.full(6, 0.5))
    up, E = eb.mean_support_polytope(V, hull)
    lb = eb.lower_bound_mesh(V, E, U, np.full(6, 0.5), 64, K=6)
    ok1 = abs(up - 0.75) < 1e-12 and lb['eta_lower'] <= 0.75 + 1e-12 and lb['eta_upper_interp'] >= up - 1e-12
    details.append(f"cube: upper {up:.14f} (exact 0.75), cap lower {lb['eta_lower']:.6f} <= 0.75, interp upper {lb['eta_upper_interp']:.6f} >= exact")
    # octahedron: 8 facet normals (+-1,+-1,+-1)/sqrt3 with eta = 1/sqrt3 -> P = {|x|+|y|+|z| <= 1}
    from itertools import product
    U8 = np.array([sg for sg in product((-1.0, 1.0), repeat=3)]) / math.sqrt(3.0)
    V, hull = eb.polytope_from_upper(U8, np.full(8, 1.0 / math.sqrt(3.0)))
    up8, E8 = eb.mean_support_polytope(V, hull)
    z, wz = np.polynomial.legendre.leggauss(1200)
    phi = 2 * math.pi * (np.arange(2400) + 0.5) / 2400
    Z, PH = np.meshgrid(z, phi, indexing='ij')
    n = np.stack([np.sqrt(1 - Z * Z) * np.cos(PH), np.sqrt(1 - Z * Z) * np.sin(PH), Z], axis=-1)
    hval = np.max(np.abs(n), axis=-1)
    direct = float(np.sum(wz[:, None] / 2.0 * hval / 2400.0))
    ok2 = abs(up8 - direct) < 1e-5
    details.append(f"octahedron: edge formula {up8:.8f} vs direct quadrature of max|n_i| {direct:.8f} (tol 1e-5)")
    # ball of radius 1 at the sweep's directions: upper >= 1 >= lower; the gap is the geometric width of the set
    ok3 = True
    for pm in (4, 6):
        rows = [dict(p=q, pnorm=math.sqrt(sum(v * v for v in q)), eta=1.0, hw=0.0) for q in ec.direction_orbits(pm)]
        res = eb.run_bounds(rows, hw_scale=1.0, mesh_list=(64, 128), K=32, verbose=False)
        ok3 = ok3 and res['eta_upper'] >= 1.0 - 1e-12 and res['eta_lower'] <= 1.0 + 1e-12
        details.append(f"ball pmax {pm} ({res['n_normals']} normals): upper-1 = {res['eta_upper']-1:.2e}, 1-lower = {1-res['eta_lower']:.2e} (geometric width)")
    record('T18 bound arithmetic on known bodies', ok1 and ok2 and ok3, '; '.join(details))


def t19_vector_areas():
    N = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=float)
    vec = eb.triangle_vector_areas(N)[0]
    ok1 = np.max(np.abs(vec - math.pi / 4)) < 1e-12
    Vm, T = eb.fundamental_mesh(8)
    vsum = eb.triangle_vector_areas(Vm[T]).sum(axis=0)
    A, B, C = np.array([1, 0, 0.]), np.array([1, 1, 0.]) / math.sqrt(2), np.array([1, 1, 1.]) / math.sqrt(3)
    vfund = eb.triangle_vector_areas(np.array([[A, B, C]]))[0]
    ok2 = np.max(np.abs(vsum - vfund)) < 1e-12
    # linear function reproduced exactly: g(n) = n.e via a one-point cap
    e = np.array([0.3, -0.5, 0.8])
    Nt = Vm[T]
    W = eb.triangle_weights(Nt, eb.triangle_vector_areas(Nt))
    lin = float(np.sum(W * (Nt @ e)))
    ok3 = abs(lin - float(vfund @ e)) < 1e-12
    # concave test: g(n) = min over cube vertices of n.v = -(|n1|+|n2|+|n3|)/2, sphere average -3/4
    cube = np.array([v for v in np.array(np.meshgrid([-.5, .5], [-.5, .5], [-.5, .5])).reshape(3, -1).T])
    vals = {}
    for M in (64, 128):
        Vm2, T2 = eb.fundamental_mesh(M)
        N2 = Vm2[T2]
        W2 = eb.triangle_weights(N2, eb.triangle_vector_areas(N2))
        g = (N2 @ cube.T).min(axis=2)
        vals[M] = float((12.0 / math.pi) * np.sum(W2 * g))
    # builder's repair: g is exactly linear inside the fundamental triangle, so both meshes reproduce -3/4 to
    # roundoff and the earlier strict decrease failed on roundoff; non-increase within 1e-12 is the test.
    ok4 = vals[64] <= -0.75 + 1e-12 and vals[128] <= -0.75 + 1e-12 and abs(vals[128] + 0.75) <= abs(vals[64] + 0.75) + 1e-12 and abs(vals[128] + 0.75) < 1e-3
    record('T19 vector areas, linear reproduction, concave interpolant', ok1 and ok2 and ok3 and ok4,
           f"octant {vec} (pi/4 = {math.pi/4:.12f}); mesh sum vs fundamental {np.max(np.abs(vsum-vfund)):.1e}; linear {abs(lin-float(vfund@e)):.1e}; "
           f"concave cube: M=64 {vals[64]:.8f}, M=128 {vals[128]:.8f} <= -0.75 and converging")


def t20_three_direction_artifact():
    e100 = ec.TARGET_ETA100
    e110 = ec.P23_INFO[(1, 1, 0)]
    e111 = ec.TARGET_ETA111
    c0 = (10 * e100 + 16 * e110 + 9 * e111) / 35.0
    # the weights: zero mean of H4 = p4 - 3/5 and H6 = p6 - 15 p4/11 + 30/77 at the 3 directions
    H4 = np.array([2 / 5, -1 / 10, -4 / 15])
    H6 = np.array([2 / 77, -13 / 308, 32 / 693])
    w = np.array([10, 16, 9]) / 35.0
    ok = abs(w.sum() - 1) < 1e-15 and abs(w @ H4) < 1e-15 and abs(w @ H6) < 1e-15
    record('T20 three-direction weights and artifact', ok, f"weights (10,16,9)/35 annihilate H4, H6: residuals {w@H4:.1e}, {w@H6:.1e}; value {c0:.5f} (an artifact, not the sphere average)")


# ---------------------------------------------------------------------------------------------- post-sweep
def post_sweep(args):
    rows = eb.read_table(args.table, mass=0.0)
    byp = {rw['p']: rw for rw in rows}
    for key, target in (((1, 0, 0), ec.TARGET_ETA100), ((1, 1, 1), ec.TARGET_ETA111)):
        if key in byp:
            rw = byp[key]
            d = rw['eta'] - target
            record(f'P1 exact coverage {key}', abs(d) <= rw['hw'] and abs(d) <= 5e-6, f"eta {rw['eta']:.10f} +/- {rw['hw']:.1e}, exact {target:.12f}, diff {d:+.2e}")
    n_ok = 0
    worst = 0.0
    for key, val in ec.SOLVE_LANE_J7.items():
        if key in byp:
            d = byp[key]['eta'] - val
            worst = max(worst, abs(d))
            if abs(d) <= 3e-6 + byp[key]['hw']:
                n_ok += 1
    record('P2 solve-lane 22 values', n_ok == sum(1 for k in ec.SOLVE_LANE_J7 if k in byp), f"{n_ok} of {sum(1 for k in ec.SOLVE_LANE_J7 if k in byp)} within 3e-6 + hw; worst |diff| {worst:.2e}")
    hws = np.array([rw['hw'] for rw in rows])
    record('P3 half-width distribution', hws.max() <= 1e-5, f"{len(rows)} directions: max hw {hws.max():.2e} (tol 1e-5), median {np.median(hws):.2e}")
    # level-2 ratios from the JSON files if present
    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.table)), 'dirs')
    cases = ed.load_records(outdir) if os.path.isdir(outdir) else {}
    rats = []
    for key, recs in cases.items():
        if key[1] == 0.0 and len(recs) >= 3:
            c = ed.combine_levels(recs)
            if c and c.get('ratio') is not None:
                rats.append((key[0], c['ratio']))
    if rats:
        ok = all(6.0 <= r <= 40.0 for _, r in rats)
        record('P4 level-2 error ratios', ok, '; '.join(f"{p}: {r:.1f}" for p, r in rats) + ' (expected 8 to 16; accept 6 to 40)')
    else:
        record('P4 level-2 error ratios', False, f"no 3-level cases in {outdir}")
    for sc in (1.0, 2.0, 10.0):
        res = eb.run_bounds(rows, hw_scale=sc, mesh_list=(64, 128), K=32, verbose=False)
        ok = res['support_violations'] == 0 and all(m['eta_upper_interp'] >= res['eta_upper'] - 1e-10 for m in res['meshes'])
        record(f'P5 bounds consistency, hw x {sc:g}', ok, f"{res['eta_lower']:.8f} < eta_sphere < {res['eta_upper']:.8f}; support violations {res['support_violations']}; interp upper >= exact at every mesh: {ok}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--h0', type=float, default=0.1)
    ap.add_argument('--cL', type=float, default=10.0)
    ap.add_argument('--Lmin', type=int, default=24)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--dps2d', type=int, default=20)
    ap.add_argument('--skip', nargs='*', default=[], help='test numbers to skip, e.g. 05 06')
    ap.add_argument('--post-sweep', action='store_true')
    ap.add_argument('--table', default='eta_table.tsv')
    ap.add_argument('--outdir', default=None, help='JSON directory for the level-2 ratio check (default: dirs beside the table)')
    ap.add_argument('--out', default='eta_tests.json')
    args = ap.parse_args()
    t0 = time.time()
    if args.post_sweep:
        post_sweep(args)
    else:
        tests = [('01', t01_s4_ladder_vs_closed), ('02', t02_s4_asymptotics), ('03', t03_rho_normalization_and_moment),
                 ('04', t04_resolvent_identity), ('05', lambda: t05_eta100_exact(args)), ('06', lambda: t06_eta111_exact(args)),
                 ('07', t07_p3), ('08', lambda: t08_F111_vs_chain(args)), ('09', lambda: t09_s4_vs_chain(args)),
                 ('10', lambda: t10_deficit_scan(args)), ('11', t11_integer_geometry), ('12', lambda: t12_symmetries(args)),
                 ('13', t13_reference_integral), ('14', lambda: t14_direction_100(args)), ('15', lambda: t15_direction_111_110(args)),
                 ('16', lambda: t16_massive_100(args)), ('17', t17_b_fit_exact), ('18', lambda: t18_bounds_known_bodies(args)),
                 ('19', t19_vector_areas), ('20', t20_three_direction_artifact)]
        for num, fn in tests:
            if num in args.skip:
                print(f"[SKIP] T{num}")
                continue
            try:
                fn()
            except Exception as e:  # a crash is a failure, not a silent skip
                record(f'T{num}', False, f"exception: {type(e).__name__}: {e}")
    npass = sum(1 for r in RESULTS if r['passed'])
    print(f"\n{npass} PASS, {len(RESULTS)-npass} FAIL in {time.time()-t0:.0f} s (threads {THREADS}, h0 {args.h0}, cL {args.cL}, Lmin {args.Lmin})")
    with open(args.out, 'w') as f:
        json.dump(dict(results=RESULTS, h0=args.h0, cL=args.cL, Lmin=args.Lmin, threads=THREADS, seconds=time.time() - t0), f, indent=1)
    sys.exit(0 if npass == len(RESULTS) else 1)


if __name__ == '__main__':
    main()
