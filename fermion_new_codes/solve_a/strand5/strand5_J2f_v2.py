#!/usr/bin/env python
"""Strand 5 J2f (spec solve/specs/SPEC_strand5_kernel_star.md sec. 10(n); builder analyst), with strand5_J2e_v2's two quadrature routes.
(i) P14's finite-spacing expansion (FRESH1184 files/p14_mass_lock/DERIVATION.md, "Polar two-derivative coefficient"): at a = P = 1,
x_s = m_s^2, Z = sum_s eta_s x_s^2 [C_lat - log x_s] (the massless packet contributing zero), k_a = k + 3 Z/(8 pi^2) + O(a^4 log a) with
k = k_lat = 3 sum_s eta_s x_s log x_s/pi^2; the expansion's excess 3 Z/(8 pi^2 k_lat) beside the integral's k_a/k_lat - 1.  C_lat =
3/2 - gamma_E + 4 int_0^inf [f(u) - 1 - 1/(4(1 + u))] du, f = (pi u)^(3/2) A F^2 (J2e's A F^2), evaluated here by two routes (route 1: scipy
quad with ive on [0, U], U = 1e3 (and 3e3 as a check), plus the tail beyond U from f's asymptotic series f = sum_k f_k u^-k, the f_k exact
rationals from the Bessel series e^-y I_nu(y) (2 pi y)^(1/2) = sum_k (-1)^k a_k(nu) y^-k, a_k(nu) = prod_{j<=k} (4 nu^2 - (2j - 1)^2)/(k! 8^k),
to k = 12: tail = sum_{k>=2} [f_k + (-1)^k/4]/((k - 1) U^(k-1)); route 2: mpmath tanh-sinh in log u at 80 digits with mpmath besseli on
(-inf, 40] plus the leading tail 21/(32 e^40)) beside P14's quoted -0.3562147416; f_1 = 1/4 and f_2 = 13/32 checked against u^2 (f - 1 - 1/(4u))
at u = 1e3, 1e4, 1e5.  The expansion is evaluated with the computed C_lat (route 1).
(version 2: version 1's route 2 at 30 digits failed beyond u of about 1e10, where mpmath's besseli at 30 digits loses the relative accuracy
the cancellation f - 1 - 1/(4(1 + u)) needs (checked on the box: at 80 digits it holds to e^36); version 1's route 1 ran ive to u = 1e7, where
the same cancellation in double precision limited it to about 1e-9.)
(ii) The physical rectangle's masses scaled by 0.0055/0.0213 (the window edge): k_a(0) by both routes, its k_lat, the ratio, P14's excess.
(iii) The equal control rectangles of J2 (light mass 0.1; masses 0.1, (0.01 + aH^2)^(1/2) twice, (0.01 + 2 aH^2)^(1/2); orientations
(+, -, -, +)) for every aH in J2's kernel_v3.json: k_a(0) by both routes against J2's moment k_a0_moment and its fit k_a0; and the J4
families' rectangles (m_l, (m_l^2 + aH^2)^(1/2) twice, (m_l^2 + 2 aH^2)^(1/2)) against the k_op in each stage-1 family file given.
Writes <out>/J2f.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
import time

import mpmath as mp
import numpy as np
from scipy import integrate
from scipy.special import ive

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand5_J2e_v2 as E  # noqa: E402

C_LAT_P14 = -0.3562147416
SCALE = 0.0055 / 0.0213


def f_u(u):
    x = 0.5 * u
    i0, i1 = ive(0, x), ive(1, x)
    return (math.pi * u) ** 1.5 * 0.5 * (i0 + i1) * i0 * i0


def f_series(K=12):
    """Exact rationals f_k, f = sum_k f_k u^-k (k = 0..K), from the Bessel asymptotic series with y = u/2."""
    from fractions import Fraction

    def S(nu):
        out = []
        for k in range(K + 1):
            a = Fraction(1)
            for j in range(1, k + 1):
                a *= Fraction(4 * nu * nu - (2 * j - 1) ** 2, 8 * j)
            out.append((-1) ** k * a * 2 ** k)                  # coefficient of u^-k in e^-y I_nu(y) (pi u)^(1/2), y = u/2
        return out
    s0, s1 = S(0), S(1)
    half = [(x + y) / 2 for x, y in zip(s0, s1)]
    sq = [sum(s0[i] * s0[k - i] for i in range(k + 1)) for k in range(K + 1)]
    return [sum(sq[i] * half[k - i] for i in range(k + 1)) for k in range(K + 1)]


FK = f_series()


def c_lat_route1(U=1e3):
    g = lambda u: f_u(u) - 1.0 - 0.25 / (1.0 + u)
    edges = [0.0, 1.0, 10.0, 100.0, U]
    tot, err = 0.0, 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        v, e = integrate.quad(g, a, b, epsabs=1e-15, epsrel=1e-13, limit=500)
        tot += v
        err += e
    tail = math.fsum(float(FK[k] + (-1) ** k * 0.25) / ((k - 1) * U ** (k - 1)) for k in range(2, len(FK)))
    return 1.5 - float(np.euler_gamma) + 4 * (tot + tail), err, tail


def c_lat_route2():
    mp.mp.dps = 80
    X = 40

    def g(x):
        u = mp.e ** x
        y = u / 2
        b0 = mp.besseli(0, y)
        b1 = mp.besseli(1, y)
        f = (mp.pi * u) ** mp.mpf(1.5) * mp.e ** (-3 * y) * (b0 + b1) / 2 * b0 * b0
        return (f - 1 - mp.mpf(1) / (4 * (1 + u))) * u
    v = mp.quad(g, [-mp.inf, -20, -5, 0, 5, 10, 20, 30, X])
    tail = mp.mpf(21) / (32 * mp.e ** X)
    out = float(mp.mpf(1.5) - mp.euler + 4 * (v + tail)), float(tail)
    mp.mp.dps = 30
    return out


def rect(masses, etas):
    """k_a(0) of a rectangle by both routes (route 1's weight as the eta-sum of exponentials)."""
    m2 = [m * m for m in masses]
    w = lambda t: math.fsum(e * math.exp(-t * x) for x, e in zip(m2, etas))
    r1, e1, tail1 = E.route1(w, m2, etas)
    r2, tail2 = E.route2(m2, etas)
    kl = 3 * sum(e * x * math.log(x) for x, e in zip(m2, etas) if x > 0) / math.pi ** 2
    return {"masses": list(masses), "etas": list(etas), "route1": r1, "route2": r2, "difference": r1 - r2, "route1_quad_error": e1,
            "k_lat": kl, "ratio": r1 / kl}


def p14_excess(masses, etas, clat):
    m2 = [m * m for m in masses]
    Z = sum(e * x * x * (clat - math.log(x)) for x, e in zip(m2, etas) if x > 0)
    kl = 3 * sum(e * x * math.log(x) for x, e in zip(m2, etas) if x > 0) / math.pi ** 2
    return {"Z": Z, "dk": 3 * Z / (8 * math.pi ** 2), "excess": 3 * Z / (8 * math.pi ** 2 * kl)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--j2", required=True, help="J2's kernel_v3.json")
    ap.add_argument("--families", required=True, help="comma-separated stage-1 family files (lstar_*.json) whose meta k_op to compare")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    res = {"meta": {"C_lat_P14": C_LAT_P14, "scale": SCALE}}
    c1, ce1, ct1 = c_lat_route1(1e3)
    c1b, ce1b, ct1b = c_lat_route1(3e3)
    c2, ct2 = c_lat_route2()
    res["C_lat"] = {"route1": c1, "route1_quad_error": ce1, "route1_tail": ct1, "route1_U3e3": c1b, "route1_U3e3_quad_error": ce1b,
                    "route2": c2, "route2_tail": ct2, "difference": c1 - c2, "minus_P14": c1 - C_LAT_P14, "route2_minus_P14": c2 - C_LAT_P14,
                    "f_k": [str(x) for x in FK]}
    res["f_coefficient_check"] = [{"u": u, "u2_times_f_minus_1_minus_1_over_4u": u * u * (f_u(u) - 1 - 0.25 / u), "13/32": 13 / 32} for u in (1e3, 1e4, 1e5)]
    print("C_lat: route 1 %.15f (quad error %.1e, tail %.1e; with U = 3e3 %.15f) route 2 %.15f difference %.2e; minus P14's %.2e (%.0f s)" % (
        c1, ce1, ct1, c1b, c2, c1 - c2, c1 - C_LAT_P14, time.time() - t0), flush=True)
    print("  f_k: " + ", ".join(str(x) for x in FK[:5]), flush=True)
    for r in res["f_coefficient_check"]:
        print("  u %.0e: u^2 (f - 1 - 1/(4u)) = %.8f against 13/32 = %.8f" % (r["u"], r["u2_times_f_minus_1_minus_1_over_4u"], 13 / 32), flush=True)
    # (i) and (ii)
    phys = [0.0, E.M1, E.M1, E.M2]
    et = [1, -1, -1, 1]
    for name, ms in (("physical", phys), ("window edge", [m * SCALE for m in phys])):
        r = rect(ms, et)
        r["P14"] = p14_excess(ms, et, c1)
        r["integral_excess"] = r["ratio"] - 1
        r["integral_minus_P14_excess"] = r["integral_excess"] - r["P14"]["excess"]
        res[name] = r
        print("%s (masses %s): k_a(0) route 1 %.15e route 2 %.15e difference %.2e; k_lat %.13e ratio %.12f; excess %.6e, P14 %.6e, "
              "difference %.2e (%.0f s)" % (name, ", ".join("%.10g" % m for m in ms), r["route1"], r["route2"], r["difference"], r["k_lat"],
                                             r["ratio"], r["integral_excess"], r["P14"]["excess"], r["integral_minus_P14_excess"], time.time() - t0), flush=True)
    # (iii) J2's equal rectangles
    J2 = json.load(open(a.j2))
    res["controls"] = []
    for row in J2["rows"]:
        if not row["set"].startswith("equal rectangle"):
            continue
        H = row["aH"]
        ms = [0.1, math.sqrt(0.01 + H * H), math.sqrt(0.01 + H * H), math.sqrt(0.01 + 2 * H * H)]
        r = rect(ms, et)
        r.update({"set": row["set"], "aH": H, "J2_moment": row["k_a0_moment"], "J2_fit": row["k_a0"], "moment_over_integral": row["k_a0_moment"] / r["route1"],
                  "fit_over_integral": row["k_a0"] / r["route1"], "P14": p14_excess(ms, et, c1)})
        res["controls"].append(r)
        print("control aH %g: integral %.13e (routes differ %.1e) J2 moment %.13e (ratio %.10f) fit %.11e (ratio %.8f) k_lat ratio %.8f (%.0f s)" % (
            H, r["route1"], r["difference"], row["k_a0_moment"], r["moment_over_integral"], row["k_a0"], r["fit_over_integral"], r["ratio"],
            time.time() - t0), flush=True)
    # the J4 families' k_op
    res["families"] = []
    for p in a.families.split(","):
        M = json.load(open(p))["meta"]
        ml, H = M["mlight"], M["aH"]
        ms = [ml, math.sqrt(ml * ml + H * H), math.sqrt(ml * ml + H * H), math.sqrt(ml * ml + 2 * H * H)]
        r = rect(ms, et)
        r.update({"family": p, "operator": M["operator"], "mlight": ml, "aH": H, "k_op": M["k_op"], "k_op_over_integral": M["k_op"] / r["route1"]})
        res["families"].append(r)
        print("family %s (%s): integral %.13e (routes differ %.1e) k_op %.13e, k_op/integral %.10f (%.0f s)" % (
            p, M["operator"], r["route1"], r["difference"], M["k_op"], r["k_op_over_integral"], time.time() - t0), flush=True)
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "J2f.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
