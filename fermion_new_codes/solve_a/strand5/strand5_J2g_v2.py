#!/usr/bin/env python
"""Strand 5 J2g (spec solve/specs/SPEC_strand5_kernel_star.md sec. 10(q); builder analyst): the physical inventory's ruler derivative and
P14 eq. 7's piece on J1's continuum profiles.
With J2e's weight w(t) = 3 pi^-1/2 t^-1/2 A(t) F(t)^2 (strand5_J2e_v2.AF2) and the rectangle (0, m_1, m_1, m_2), m_2^2 = 2 m_1^2,
orientations (+, -, -, +), h = m_1^2:
  K(h) = int w (1 - e^-ht)^2 dt  (= k_a(0; phys) at P = 1),   D(h) = int w [-4 (1 - e^-ht)^2 + 8 h t (e^-ht - e^-2ht)] dt  (= -4 K + 4 h K'),
  k_a(P) = P^-4 K(h P^4),   k_a'(P) = P^-5 D(h P^4)   (the masses m_s P^2 under the ruler P; k_a'(1) = D(h)).
(1) K and D at P = 1 on J2e's two routes: route 1, scipy quad in u = t^(1/2) with ive on [0, 1e7^(1/2)] split at 1, 10, 100, 1000, plus the
    analytic tail beyond T = 1e7 from A F^2 = (pi t)^(-3/2) [1 + 1/(4t) + 13/(32 t^2)], term by term for each c t^j e^(-a t) in the weight
    (j = 0: E_2(aT)/T + E_3(aT)/(4T^2) + 13 E_4(aT)/(32 T^3); j = 1: E_1(aT) + E_2(aT)/(4T) + 13 E_3(aT)/(32 T^2)), the differences written with
    expm1; route 2, mpmath tanh-sinh at 30 digits in x = log t over (-inf, 30] with mpmath besseli and the weight at 30 digits, the tail
    beyond e^30 from its t -> infinity value.  Two assignments: m_1 = mu = 0.0130903478 (J2e) and J2f's window edge (m_1 mu 0.0055/0.0213).
    Reported: k_a'(1)/k_a(0), k_a'(1)/k_lat (k_lat = 6 h log 2/pi^2) against the red team's expansion estimates 7.33e-4 and 7.12e-5.
(2) k_a(P) and k_a'(P) for P from 1 to P_max = max(1.5, 1.02 psi_c) at 51 points (route 1), route 2 at P = 1, 1.25, 1.5; the cubic spline of
    k_a'(P)/k_a(1) used in (3), checked against route 1 at the midpoints.
(3) P14 eq. 7's vacuum piece  -(1/(4 pi k)) int psi k_a'(psi) grad psi . grad chi d^3x  on J1's continuum profiles (strand5_star_v4.shoot,
    the reference ODE solution with dense output; G = m = 1, k = 1/(2 pi), g = 16), with the kernel's relative ruler derivative
    k_a'(psi)/k_a(1) of (2) in place of k_a'/k: the maximum member (z_c of J1's reference maximum) and the member with GM/R_s = 2.1e-6
    (R_s areal; z_c by Brent's method on the reference).  Interior by composite Gauss-Legendre in the shooting variable on the dense output
    (int 4 pi r^2 psi f(psi) psi' chi' dr = beta^-1 int 4 pi s^2 Y_0 f(Y_0/alpha) Y_1 Y_3 ds), 200 and 400 panels of 16 nodes; the control
    with f = 1 and no psi factor against the ODE's own component IW (times alpha/beta); the exterior analytically, -4 pi B int_0^(A/r_s)
    (1 + u) f(1 + u) du.  The piece over A and the implied gamma - 1 = D_7/(E_bdy - D_7) with D_7 = the piece (G = 1; read as a contribution
    to A - B, as the stage-2 tables compare it) and E_bdy = 2A (P33 [S]); against Cassini's gamma - 1 = (2.1 +- 2.3)e-5 for the solar
    member; "no named bound" for the strong-field member.
(4) The same at the window-edge assignment.
Writes <out>/J2g.json.
(version 2: version 1 stopped at its first JSON write on a numpy bool in the Cassini comparison; the values are cast to Python types.)"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
import time
from multiprocessing import Pool

import mpmath as mp
import numpy as np
from scipy import integrate
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.special import expn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand5_J2e_v2 as J2E  # noqa: E402
import strand5_star_v4 as STAR  # noqa: E402

MU = 0.013090347820517666
EDGE = MU * 0.0055 / 0.0213
C3 = 3.0 / math.sqrt(math.pi)
T1 = 1e7
X2 = 30
RT_ESTIMATE = {"physical": 7.33e-4, "window edge": 7.12e-5}
CASSINI = (2.1e-5, 2.3e-5)
G_SOLAR = 2.1e-6


def terms(kind, h):
    """The weight as sum of c t^j e^(-a t): list of (c, j, a)."""
    if kind == "K":
        return [(1.0, 0, 0.0), (-2.0, 0, h), (1.0, 0, 2 * h)]
    return [(-4.0, 0, 0.0), (8.0, 0, h), (-4.0, 0, 2 * h), (8 * h, 1, h), (-8 * h, 1, 2 * h)]


def weight64(kind, h, t):
    e1 = -math.expm1(-h * t)
    if kind == "K":
        return e1 * e1
    return -4 * e1 * e1 + 8 * h * t * math.exp(-h * t) * e1


def tail1(kind, h):
    tot = []
    for c, j, a in terms(kind, h):
        x = a * T1
        if j == 0:
            tot.append(c * (expn(2, x) / T1 + expn(3, x) / (4 * T1 ** 2) + 13 * expn(4, x) / (32 * T1 ** 3)))
        else:
            tot.append(c * (expn(1, x) + expn(2, x) / (4 * T1) + 13 * expn(3, x) / (32 * T1 ** 2)))
    return C3 * math.pi ** -1.5 * math.fsum(tot)


def route1(args):
    kind, h = args
    f = lambda u: 2 * C3 * J2E.AF2(u * u) * weight64(kind, h, u * u)
    U = math.sqrt(T1)
    edges = [0.0, 1.0, 10.0, 100.0, 1e3, U]
    tot, err = 0.0, 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        v, e = integrate.quad(f, a, b, epsabs=1e-17, epsrel=1e-13, limit=500)
        tot += v
        err += e
    tl = tail1(kind, h)
    return tot + tl, err, tl


def route2(args):
    kind, h = args
    mp.mp.dps = 30
    hm = mp.mpf(h)

    def wt(t):
        e1 = -mp.expm1(-hm * t)
        if kind == "K":
            return e1 * e1
        return -4 * e1 * e1 + 8 * hm * t * mp.e ** (-hm * t) * e1

    def g(x):
        t = mp.e ** x
        y = t / 2
        b0 = mp.besseli(0, y)
        b1 = mp.besseli(1, y)
        return mp.sqrt(t) * mp.e ** (-3 * y) * (b0 + b1) / 2 * b0 * b0 * wt(t)
    v = mp.quad(g, [-mp.inf, -20, -5, 0, 5, 10, 20, X2])
    w_inf = 1 if kind == "K" else -4
    tail = mp.pi ** mp.mpf(-1.5) * w_inf / mp.e ** X2
    return float(mp.mpf(3) / mp.sqrt(mp.pi) * (v + tail)), float(mp.mpf(3) / mp.sqrt(mp.pi) * tail)


def member_profile(zc, g=16.0):
    ref = STAR.shoot(zc, g, keep=True)
    return ref


def eq7(ref, fspl, npan):
    """int 4 pi r^2 psi f(psi) psi' chi' dr over all r (interior by GL on the dense output, exterior analytic), and the f = 1 control."""
    sol = ref["_sol"]
    alpha, beta, R = ref["_scal"]
    x, w = np.polynomial.legendre.leggauss(16)
    edges = np.linspace(sol.t[0], R, npan + 1)
    tot, ctl = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = 0.5 * (b - a) * x + 0.5 * (b + a)
        ws = 0.5 * (b - a) * w
        Y = sol.sol(s)
        P = Y[0] / alpha
        tot.append(float(np.sum(ws * 4 * np.pi * s * s * Y[0] * fspl(P) * Y[1] * Y[3])) / beta)
        ctl.append(float(np.sum(ws * 4 * np.pi * s * s * Y[1] * Y[3])))
    A, B, rs = ref["A"], ref["B"], ref["r_s"]
    ue = A / rs
    xg, wg = np.polynomial.legendre.leggauss(64)
    u = 0.5 * ue * (xg + 1)
    ext = -4 * np.pi * B * float(np.sum(0.5 * ue * wg * (1 + u) * fspl(1 + u)))
    iw = float(sol.y_events[0][0][8])
    return {"interior": math.fsum(tot), "exterior": ext, "total": math.fsum(tot) + ext, "control_IW_quad": math.fsum(ctl), "control_IW_ode": iw,
            "control_rel": math.fsum(ctl) / iw - 1, "psi_c": 1 / alpha, "panels": npan}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--zc-max", type=float, default=0.8338537642148843, help="J1's reference maximum (star_v4.json controls.reference_maximum)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    res = {"meta": {"mu": MU, "edge": EDGE, "T1": T1, "X2": X2, "red_team_estimates_over_k_lat": RT_ESTIMATE, "cassini": CASSINI,
                    "GM_over_R_solar": G_SOLAR, "zc_max": a.zc_max}}
    # the members
    mem = {}
    mem["maximum"] = member_profile(a.zc_max)
    target = 2 * G_SOLAR
    lo, hi = 1e-4, 0.05
    zsol = brentq(lambda z: math.log(STAR.shoot(z, 16.0)["compactness_2GM_over_Rs"] / target), lo, hi, xtol=1e-14, rtol=1e-13)
    mem["solar"] = member_profile(zsol)
    res["members"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in mem.items()}
    for k, v in mem.items():
        print("member %s: z_c %.10g, 2GM/R_s %.6e, GM/R_s %.6e, A %.6e, B %.6e, r_s %.6e, psi_c %.8f (%.0f s)" % (
            k, v["z_c"], v["compactness_2GM_over_Rs"], v["compactness_2GM_over_Rs"] / 2, v["A"], v["B"], v["r_s"], 1 / v["_scal"][0], time.time() - t0), flush=True)
    psi_c = max(1 / v["_scal"][0] for v in mem.values())
    Pmax = max(1.5, 1.02 * psi_c)
    Ps = np.linspace(1.0, Pmax, 51)
    Pmid = 0.5 * (Ps[:-1] + Ps[1:])[::5]
    res["P_grid"] = Ps.tolist()
    with Pool(a.workers) as pool:
        for name, m1 in (("physical", MU), ("window edge", EDGE)):
            h = m1 * m1
            k_lat = 6 * h * math.log(2) / math.pi ** 2
            jobs = [(kind, h * P ** 4) for P in Ps for kind in ("K", "D")] + [(kind, h * P ** 4) for P in Pmid for kind in ("K", "D")]
            r1 = pool.map(route1, jobs, chunksize=1)
            P2 = [1.0, 1.25, 1.5]
            jobs2 = [(kind, h * P ** 4) for P in P2 for kind in ("K", "D")]
            r2 = pool.map(route2, jobs2, chunksize=1)
            n = len(Ps)
            Kv = np.array([r1[2 * i][0] for i in range(n)])
            Dv = np.array([r1[2 * i + 1][0] for i in range(n)])
            ka = Kv / Ps ** 4
            kap = Dv / Ps ** 5
            Km = np.array([r1[2 * n + 2 * i][0] for i in range(len(Pmid))])
            Dm = np.array([r1[2 * n + 2 * i + 1][0] for i in range(len(Pmid))])
            fspl = CubicSpline(Ps, kap / ka[0])
            spl_check = float(np.max(np.abs(fspl(Pmid) - (Dm / Pmid ** 5) / ka[0])))
            sec = {"h": h, "m1": m1, "k_lat": k_lat, "K_route1": float(Kv[0]), "D_route1": float(Dv[0]), "K_quad_error": r1[0][1], "D_quad_error": r1[1][1],
                   "K_tail": r1[0][2], "D_tail": r1[1][2],
                   "route2": [{"P": P, "K": r2[2 * i][0], "D": r2[2 * i + 1][0], "K_tail": r2[2 * i][1], "D_tail": r2[2 * i + 1][1]} for i, P in enumerate(P2)],
                   "ka_prime_1_over_ka_0": float(Dv[0] / Kv[0]), "ka_prime_1_over_k_lat": float(Dv[0] / k_lat), "ka_0_over_k_lat": float(Kv[0] / k_lat),
                   "red_team_estimate_over_k_lat": RT_ESTIMATE[name], "table": [{"P": float(P), "k_a": float(x), "k_a_prime": float(y), "k_a_prime_over_k_a1": float(y / ka[0])}
                                                                               for P, x, y in zip(Ps, ka, kap)],
                   "spline_check_max_abs": spl_check}
            # route 2 against route 1 at the same P (route 1 evaluated there directly)
            direct = pool.map(route1, jobs2, chunksize=1)
            for i, P in enumerate(P2):
                sec["route2"][i]["K_route1_at_P"] = direct[2 * i][0]
                sec["route2"][i]["D_route1_at_P"] = direct[2 * i + 1][0]
                sec["route2"][i]["K_minus_route1"] = sec["route2"][i]["K"] - direct[2 * i][0]
                sec["route2"][i]["D_minus_route1"] = sec["route2"][i]["D"] - direct[2 * i + 1][0]
            print("%s: K %.15e D %.15e k_a'(1)/k_a(0) %.10e k_a'(1)/k_lat %.10e (red team %.3g) route 2 - route 1 at P = 1: %.2e %.2e; spline check %.1e (%.0f s)" % (
                name, Kv[0], Dv[0], Dv[0] / Kv[0], Dv[0] / k_lat, RT_ESTIMATE[name], sec["route2"][0]["K_minus_route1"], sec["route2"][0]["D_minus_route1"],
                spl_check, time.time() - t0), flush=True)
            sec["eq7"] = {}
            for mname, ref in mem.items():
                rr = [eq7(ref, fspl, npan) for npan in (200, 400)]
                e = rr[-1]
                piece = -e["total"] / (4 * np.pi)
                A = ref["A"]
                d7 = piece
                gam = d7 / (2 * A - d7)
                rec = {"I7_200": float(rr[0]["total"]), "I7_400": float(rr[1]["total"]), "I7_interior": float(e["interior"]), "I7_exterior": float(e["exterior"]),
                       "control_rel": float(e["control_rel"]), "piece": float(piece), "piece_over_A": float(piece / A), "gamma_minus_1": float(gam),
                       "beta_minus_1": float(d7 / (2 * (2 * A - d7))), "GM_over_R_s": float(ref["compactness_2GM_over_Rs"] / 2), "z_c": float(ref["z_c"]),
                       "psi_c": float(e["psi_c"])}
                if mname == "solar":
                    rec["against_cassini"] = {"gamma_minus_1": float(gam), "cassini_central": CASSINI[0], "cassini_sigma": CASSINI[1],
                                              "within_one_sigma": bool(abs(gam - CASSINI[0]) <= CASSINI[1]), "ratio_to_sigma": float(gam / CASSINI[1])}
                else:
                    rec["against_named_bound"] = "no named bound"
                sec["eq7"][mname] = rec
                print("  eq. 7 (%s, %s): I %.10e (200 panels %.10e), piece/A %.6e, gamma - 1 %.6e, control %.1e (%.0f s)" % (
                    name, mname, e["total"], rr[0]["total"], piece / A, gam, e["control_rel"], time.time() - t0), flush=True)
            res[name] = sec
            json.dump(res, open(os.path.join(a.out, "J2g.json"), "w"), indent=1)
    res["meta"]["seconds"] = time.time() - t0
    json.dump(res, open(os.path.join(a.out, "J2g.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
