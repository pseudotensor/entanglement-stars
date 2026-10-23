#!/usr/bin/env python
"""Strand 5 J2e (spec solve/specs/SPEC_strand5_kernel_star.md sec. 10(l); builder analyst): the infinite-volume q = 0 coefficients by the
Schwinger representation y^-1/2 = pi^-1/2 int_0^inf t^-1/2 e^(-t y) dt, which factorizes the zone average:
    k(m) = 3 pi^-1/2 int_0^inf t^-1/2 e^(-t m^2) A(t) F(t)^2 dt,   F(t) = <e^(-t sin^2 k)> = e^(-t/2) I_0(t/2),
    A(t) = <cos^2 k e^(-t sin^2 k)> = e^(-t/2) [I_0(t/2) + I_1(t/2)]/2,
    k_a(0; phys) = 3 pi^-1/2 int t^-1/2 A F^2 (1 - e^(-h t))^2 dt,  h = m_1^2 (the lambda = 1 rectangle (0, m_1, m_1, m_2), m_2^2 = 2 m_1^2,
    orientations (+, -, -, +): sum eta e^(-t m^2) = (1 - e^(-h t))^2, the rectangle summed under the integral).
Route 1: scipy quad in u = t^(1/2) (t^-1/2 dt = 2 du), the integrand with scipy.special.ive, on [0, 1e6] split at 1, 10, 100, ..., 1e5
(t to 1e12; the tail beyond, of order 3/(pi^2 U^2) = 3e-13 for the m = 0 integrands, added from the leading asymptotic form), epsabs 1e-15 and epsrel 1e-13 per piece.
Route 2: mpmath (30 digits) tanh-sinh on t = e^x over x in (-inf, 30] with mpmath.besseli (unscaled; the e^(-3t/2) factor applied in
the working precision), the analytic tail beyond t = e^30: an independent quadrature and an independent Bessel implementation; the
rectangle's weight there is the four-term eta-sum of exponentials at 30 digits (route 1 uses the closed form (1 - e^(-ht))^2).  Reported: both routes and their difference for
k(0) = c_0, k(m_1), k(m_2), k_a(0; phys), k_a(0; phys)/k_lat (k_lat = 3 sum_s eta_s m_s^2 log m_s^2/pi^2, the massless packet contributing
zero); k(m) at the J2b table's masses (read from the J2b per-packet records, beside their L = 256 moments); k(m) at 16 small masses
1e-4 to 2e-2 and the fit (k(m) - c_0)/m^2 = alpha log m^2 + beta (alpha against 3/pi^2); the two heavy packets' q = 0 constants by the
direct sum (3/L^3) sum_k cos^2 k_1/E(k) at L = 2048 and 4096 (twist 0.371, strand5_J2c_v1.classes and strand5_J2d_v1.chunk_sum, the chunks
over the pool).  The J2b-table masses are read from every per-packet record (packets_*.json) in the folders given (J2b_run1, J4kernels_run1,
J4kernels_run2), each record's moment beside the integral at its mass.  Writes <out>/J2e.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import math
import sys
import time
from multiprocessing import Pool

import mpmath as mp
import numpy as np
from scipy import integrate
from scipy.special import ive

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand5_J2c_v1 as J2C  # noqa: E402
import strand5_J2d_v1 as J2D  # noqa: E402

M1 = 0.013090347820517666
M2 = 0.01851254742395717
H = M1 * M1
C3 = 3.0 / math.sqrt(math.pi)
X2 = 30  # route 2's upper limit in x = log t (t = 1.07e13), the analytic tail beyond


def AF2(t):
    x = 0.5 * t
    i0, i1 = ive(0, x), ive(1, x)
    return 0.5 * (i0 + i1) * i0 * i0


def route1(weight):
    """2 C3 int_0^U du AF2(u^2) weight(u^2) + tail; weight(t) -> 1 or e^(-t m^2) etc."""
    f = lambda u: 2 * C3 * AF2(u * u) * weight(u * u)
    edges = [0.0, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6]
    tot = 0.0
    errs = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        v, e = integrate.quad(f, a, b, epsabs=1e-15, epsrel=1e-13, limit=500)
        tot += v
        errs += e
    U = edges[-1]
    # tail: AF2(t) ~ (pi t)^(-3/2) at large t; 2 C3 int_U^inf (pi u^2)^(-3/2) w du ~ w(inf) 2 C3 pi^-3/2/(2 U^2)
    w_inf = weight(1e30)
    tail = w_inf * 2 * C3 * math.pi ** -1.5 / (2 * U * U)
    return tot + tail, errs, tail


def route2(mass2_list, etas):
    """mpmath tanh-sinh in x = log t: C3 int t^(1/2) e^(-t) [I_0 + I_1](t/2) I_0(t/2)^2/2 sum eta e^(-t m^2) dx."""
    mp.mp.dps = 30

    def g(x):
        t = mp.e ** x
        y = t / 2
        b0 = mp.besseli(0, y)
        b1 = mp.besseli(1, y)
        w = mp.fsum([e * mp.e ** (-t * m2) for m2, e in zip(mass2_list, etas)])
        return mp.sqrt(t) * mp.e ** (-3 * y) * (b0 + b1) / 2 * b0 * b0 * w
    v = mp.quad(g, [-mp.inf, -20, -5, 0, 5, 10, 20, X2])
    # beyond T = e^X2: A F^2 -> (pi t)^(-3/2), the weight -> its t -> infinity value; C3 int_T^inf t^(-2) pi^(-3/2) w dt
    w_inf = mp.fsum([e for m2, e in zip(mass2_list, etas) if m2 == 0])
    tail = mp.pi ** mp.mpf(-1.5) * w_inf / mp.e ** X2
    return float(mp.mpf(3) / mp.sqrt(mp.pi) * (v + tail)), float(mp.mpf(3) / mp.sqrt(mp.pi) * tail)


def heavy_direct(pool, L, m, theta, workers):
    sp, cnt = J2C.classes(L, theta)
    nch = workers * 8
    bounds = np.linspace(0, sp.size, nch + 1).astype(int)
    args = [(L, theta, m, sp[b0:b1], cnt[b0:b1]) for b0, b1 in zip(bounds[:-1], bounds[1:]) if b1 > b0]
    tot = math.fsum(pool.map(J2D.chunk_sum, args, chunksize=1))
    return {"L": L, "m": m, "theta": theta, "c0_direct": 3.0 * tot / L ** 3, "classes": int(sp.size)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", required=True, help="comma-separated folders of per-packet records (packets_*.json)")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    res = {"meta": {"m1": M1, "m2": M2, "h": H}}
    kl = 3 * sum(e * m * m * math.log(m * m) for m, e in zip([M1, M1, M2], [-1, -1, 1])) / math.pi ** 2
    res["k_lat"] = kl
    rows = {}
    for name, w, ml, et in (("c0", lambda t: 1.0, [0.0], [1]), ("k(m1)", lambda t: math.exp(-t * M1 * M1), [M1 * M1], [1]),
                            ("k(m2)", lambda t: math.exp(-t * M2 * M2), [M2 * M2], [1]),
                            ("k_a(0; phys)", lambda t: (-math.expm1(-H * t)) ** 2, [0.0, H, H, 2 * H], [1, -1, -1, 1])):
        r1, e1, tail = route1(w)
        r2, tail2 = route2(ml, et)
        rows[name] = {"route1": r1, "route1_quad_error": e1, "route1_tail": tail, "route2": r2, "route2_tail": tail2, "difference": r1 - r2}
        print("%s: route 1 %.15e (quad error %.1e, tail %.1e) route 2 %.15e (tail %.1e) difference %.2e (%.0f s)" % (
            name, r1, e1, tail, r2, tail2, r1 - r2, time.time() - t0), flush=True)
    res["integrals"] = rows
    res["k_a_over_k_lat"] = {"route1": rows["k_a(0; phys)"]["route1"] / kl, "route2": rows["k_a(0; phys)"]["route2"] / kl}
    res["rectangle_from_packets"] = rows["c0"]["route1"] - 2 * rows["k(m1)"]["route1"] + rows["k(m2)"]["route1"]
    # the J2b masses
    j2b = []
    cache = {}
    for folder in a.records.split(","):
        for p in sorted(glob.glob(os.path.join(folder, "packets_*.json"))):
            P = json.load(open(p))
            m = float(P["m"])
            if m not in cache:
                cache[m] = route1(lambda t, m=m: math.exp(-t * m * m))[0]
            v = cache[m]
            j2b.append({"record": os.path.join(os.path.basename(folder.rstrip("/")), os.path.basename(p)), "m": m, "set": P["set"], "L": P["L"],
                        "theta": P["theta"][0], "k_integral": v, "k_moment": P["analysis"]["k_moment"], "moment_minus_integral": P["analysis"]["k_moment"] - v})
    res["records"] = sorted(j2b, key=lambda r: (r["m"], r["L"]))
    for r in res["records"]:
        print("record %s: m %.10g L %d moment %.13e integral %.13e moment - integral %.3e" % (r["record"], r["m"], r["L"], r["k_moment"],
                                                                                          r["k_integral"], r["moment_minus_integral"]), flush=True)
    print("records: %d, distinct masses %d (%.0f s)" % (len(j2b), len(cache), time.time() - t0), flush=True)
    # the T3i fit form on the integral at the L = 256 records' masses m <= 0.3 (m > 0)
    ms = sorted({r["m"] for r in j2b if r["L"] == 256 and 0 < r["m"] <= 0.3})
    Xf = np.stack([np.ones(len(ms)), np.array(ms) ** 2 * np.log(np.array(ms) ** 2), np.array(ms) ** 2], 1)
    yf = np.array([cache[m] for m in ms])
    cf, *_ = np.linalg.lstsq(Xf, yf, rcond=None)
    res["T3i_form_on_integral"] = {"masses": ms, "c0": float(cf[0]), "alpha": float(cf[1]), "beta": float(cf[2]),
                                   "max_residual": float(np.max(np.abs(yf - Xf @ cf)))}
    print("T3i form on the integral (%d masses <= 0.3): c0 %.10f alpha %.6f beta %.6f max residual %.1e" % (
        len(ms), cf[0], cf[1], cf[2], res["T3i_form_on_integral"]["max_residual"]), flush=True)
    # small-m fit
    c0 = rows["c0"]["route1"]
    small = []
    for m in np.geomspace(1e-4, 2e-2, 16):
        v, _, _ = route1(lambda t, m=m: math.exp(-t * m * m))
        small.append({"m": float(m), "k": v, "y": (v - c0) / (m * m)})
    X = np.stack([np.log(np.array([s["m"] for s in small]) ** 2), np.ones(len(small))], 1)
    y = np.array([s["y"] for s in small])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res["small_m"] = small
    res["small_m_fit"] = {"alpha": float(coef[0]), "beta": float(coef[1]), "alpha_over_3_pi2": float(coef[0] / (3 / math.pi ** 2)),
                          "residuals": (y - X @ coef).tolist()}
    sm4 = [s for s in small if s["m"] <= 2e-3]
    X4 = np.stack([np.log(np.array([s["m"] for s in sm4]) ** 2), np.ones(len(sm4))], 1)
    c4, *_ = np.linalg.lstsq(X4, np.array([s["y"] for s in sm4]), rcond=None)
    res["small_m_fit_m_le_2e-3"] = {"alpha": float(c4[0]), "beta": float(c4[1]), "alpha_over_3_pi2": float(c4[0] / (3 / math.pi ** 2))}
    print("small-m fit: alpha %.8f (alpha/(3/pi^2) %.8f), beta %.6f; m <= 2e-3: alpha %.8f (%.0f s)" % (
        coef[0], coef[0] / (3 / math.pi ** 2), coef[1], c4[0], time.time() - t0), flush=True)
    # heavy direct sums
    hd = []
    with Pool(a.workers) as pool:
        for L in (2048, 4096):
            for m in (M1, M2):
                hd.append(heavy_direct(pool, L, m, 0.371, a.workers))
    for r in hd:
        r["integral"] = rows["k(m1)" if abs(r["m"] - M1) < 1e-12 else "k(m2)"]["route1"]
        r["direct_minus_integral"] = r["c0_direct"] - r["integral"]
    res["heavy_direct"] = hd
    for r in hd:
        print("heavy direct L %d m %.7f: %.15e (integral %.15e, difference %.2e)" % (r["L"], r["m"], r["c0_direct"], r["integral"], r["direct_minus_integral"]), flush=True)
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "J2e.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
