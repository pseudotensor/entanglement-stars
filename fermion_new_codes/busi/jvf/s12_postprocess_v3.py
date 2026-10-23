#!/usr/bin/env python
"""s12_postprocess_v3.py, version 3 (busi lane task 5 = review task S12, J-VF post-processing; the lead's web40f task B1 added; first
draft by reviewer A at maximum effort; the builder analyst runs it on the box CPU with 1 thread). Reads the solve lane's vacuum-flow
records in place and writes a small JSON and a markdown table set; it reruns nothing. Version 2 is kept beside this file as
s12_postprocess.py; every table and verdict of version 2 is reproduced unchanged, and the B1 tests are added (busi/specs/S12_addendum_B1_v1.md).

Added in version 3 (the addendum's section numbers in brackets):
  T7a  the ramp's filter and the smooth-ramp pair coefficient [1.2, 1.3]: Phi(x) = pi^2 cos(x/2)/(pi^2 - x^2) for the records' ramp
       s = sin^2(pi t/2 t_r) (3 j_1(x/2)/(x/2) for the idea file's smoothstep), I_gen = int x Phi^2 dx = 9.4206 (9 for the smoothstep),
       B_pair/B_L = I_gen/(15 pi^2 a^2 B_L) (a/tau)^2 = 3.2778 (3.1314) with tau = t_r; the clock filter at the hold's end,
       I_clock = I_gen + 2 J(t_h/tau); the 3D valley-continuum predictions per e-fold over the full-zone slope (reported).
  T7b  the channel-local second-order excitation [1.4]: the exact-M channel sums with the record's K and weights of the clock, flow and
       generator parts at theta = 0 (the '_end' totals) and theta = t_h/tau (the '_hold' totals and the running sums), per channel, per
       probe annulus and per fine bin, against the observed NO (and R2 where stored); C1 the ramp law (control), C2 the magnitude at
       the fastest ramp (control), C3 the channel structure kbar (control), the tau -> infinity extrapolations and the end/hold ratios
       (reported).
  T7c  the remainder in the pocket-free channels, where the continued and the instantaneous references coincide (the declared
       reference's only reading at r_s 32): sizes against 1e-5 of the slope per e-fold, with the retained-space ratio; the check that
       the pocket-free NO equals the pocket-free R2 where R2 is stored (control).
  T8a  the static clock annuli against the channel-local law (3K(K+1)/(4 pi)) r_s (R1^-2 - R2^-2) and its exact-M form [2.2, 2.3]
       (controls on the far bins and on the channels kappa >= 2 at [8, 16] r_s).
  T8b  the far ratio 1 : 2/3 : 5/3 (not available: no flow or generator part beyond 2 r_s); the window-Euler reading beta/Q_c =
       flow/clock + 2 of vf10's and vf9's exterior annuli (reported).
  T8c  every remainder also divided by the retained-space flat signal raw - NO of the same bin and channel set [3] (reported).
Modes:
  --selftest                       version 2's checks plus the B1 constants, kernels, limits and a synthetic record with a planted excitation
  --dump-keys FILE [--depth D]     prints a record's structure
  --vf10 F --vf9 F --vf3-32 F --vf3-16 F --vf7 F --out DIR [--kmax 16 --BL 0.0194139 --smax 0.04 --hold-over-rs 2 --ramp sine --n-u 4001 --n-r 12]
  --static-run --lead-approved ... the static run of spec section 5 (gated; not part of B1)
Outputs: DIR/s12_tables.md, DIR/s12_results.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
import time

import numpy as np

try:
    from scipy.special import erfc as _erfc
except Exception:  # scipy is in the qi2 env; math.erfc is the fallback
    _erfc = math.erfc

VERSION = 3
NAN = float("nan")
PROBES = [1.5, 2.0, 4.0, 8.0, 16.0]
BIN_EDGES = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0, 40.0, 48.0, 60.0]
FIT_RANGES = (("far [4, 32] r_s", 4.0, 32.0), ("mid [2, 32] r_s", 2.0, 32.0), ("all [2, 60] r_s", 2.0, 60.0), ("near [1.5, 4] r_s", 1.5, 4.0))
B_L_DEFAULT = 0.0194139
QUANT = ("clock_ref", "flow_ref", "gen_ref", "clock_C", "flow_C", "gen_C", "clock_NO", "flow_NO", "gen_NO")
SUBKEY = {"raw": "E_raw_at", "NO": "E_NO_at", "R2": "E_R2_at"}
OFFSETS = (-2, -1, 0, 1, 2)
RULES = {"residual_sharp_abs": 1e-13, "residual_smooth_over_tail": 100.0, "residual_integrated_over_slope": 1e-5, "identity_rel": 1e-9,
         "static_spread_rel": 1e-6, "arith_residual_over_slope": 1e-5, "alpha_window": (-2.5, -1.5), "alpha_log_sigmas": 3.0, "ratio_tol": 0.10,
         "two_size_tol": 0.05, "rs_linearity_tol": 0.01, "annulus_threshold_over_slope": 1e-5, "record_consistency_rel": 1e-8, "running_vs_totals_abs": 1e-9,
         # B1
         "ramp_ratio_min": 6.0, "ramp_ratio_band": 3.0, "magnitude_band": (0.5, 2.0), "kbar_drop": 2.0, "model_error_fast_ramp": 0.3,
         "static_law_tol": 0.05, "static_law_channel_tol": 0.10, "static_law_inner_sites": 384, "static_law_outer_over_rs": 48.0, "pocket_free_R2_check_rel": 1e-8}
CACHE = {}


# ----------------------------------------------------------------------------------------------------------------------------- utilities
def load(path):
    with open(path) as f:
        return json.load(f)


def nmax(x):
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    return float(np.max(x)) if len(x) else NAN


def trapz(y, x):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    m = np.isfinite(y) & np.isfinite(x)
    y, x = y[m], x[m]
    if len(x) < 2:
        return NAN
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def simpson(y, x):
    """Simpson's rule on a uniform grid (an even count falls back to a trapezoid on the last panel)."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return trapz(y, x)
    if n % 2 == 0:
        return simpson(y[:-1], x[:-1]) + 0.5 * (y[-1] + y[-2]) * (x[-1] - x[-2])
    h = (x[-1] - x[0]) / (n - 1)
    return float(h / 3.0 * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2])))


def fmt(x, f="%.4e"):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(xf):
        return "nan" if np.isnan(xf) else ("inf" if xf > 0 else "-inf")
    return f % xf


def clean(o):
    """json-ready: numpy scalars and arrays to python, tuples to lists."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [clean(v) for v in o.tolist()]
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def table(md, headers, rows):
    md.append("| " + " | ".join(headers) + " |")
    md.append("|" + "---|" * len(headers))
    for r in rows:
        md.append("| " + " | ".join(str(x) for x in r) + " |")
    md.append("")


def verdict(RES, test, value, rule, ok, note="", kind="control"):
    """kind 'control': pass/fail, a failure being a defect (of the test or of the records) to be judged; 'size': met / not met, a reported
    number held against a literal threshold, not a defect when not met; 'reported': no rule, the numbers stated."""
    if kind == "reported":
        word = "reported"
    elif ok is None:
        word = "not available"
    elif kind == "size":
        word = "met" if ok else "not met"
    else:
        word = "pass" if ok else "fail"
    RES.setdefault("verdicts", []).append({"test": test, "kind": kind, "value": value, "rule": rule, "verdict": word, "note": note})


def slope_full_zone(BL, rs):
    """the full-zone flat comparator's clock charge per e-fold, (9 pi/2) B_L r_s (review 1.25: 8.78/a at r_s/a 32 and m a 0)."""
    return 4.5 * math.pi * BL * rs


def describe(obj, depth, prefix="", lines=None, maxkeys=80):
    if lines is None:
        lines = []
    if isinstance(obj, dict):
        keys = list(obj.keys())
        lines.append("%s dict %d keys: %s%s" % (prefix, len(keys), ", ".join(str(k) for k in keys[:maxkeys]), " ..." if len(keys) > maxkeys else ""))
        if depth > 0:
            for k in keys[:maxkeys]:
                describe(obj[k], depth - 1, prefix + "/" + str(k), lines, maxkeys)
    elif isinstance(obj, list):
        lines.append("%s list %d" % (prefix, len(obj)))
        if depth > 0 and obj:
            describe(obj[0], depth - 1, prefix + "[0]", lines, maxkeys)
    else:
        lines.append("%s %s %s" % (prefix, type(obj).__name__, repr(obj)[:70]))
    return lines


# ---------------------------------------------------------------------------------------------------------------------------------- fits
def sinhc(x):
    return math.sinh(x) / x if abs(x) > 1e-12 else 1.0


def fit_powerlaw(lo, hi, E, iters=8):
    """the annulus energies E_j over [lo_j, hi_j] of a line density A R^(alpha - 1): E_j = A (hi^alpha - lo^alpha)/alpha
    = A ln(q) R_geo^alpha sinhc(alpha ln q / 2) with q = hi/lo and R_geo = sqrt(lo hi) (alpha = 0: the logarithm, A ln q). Least squares in
    ln(E/ln q) against ln R_geo with the width factor removed iteratively. Points with E <= 0 are dropped and counted. Errors: the standard
    error from the scatter (n > 2) and the jackknife half-range over single deletions (n > 3); err is the larger."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    E = np.asarray(E, float)
    m = np.isfinite(E) & (E > 0) & (lo > 0) & (hi > lo)
    res = {"n": int(m.sum()), "n_dropped": int(len(E) - m.sum()), "alpha": NAN, "se": NAN, "jack_half_range": NAN, "err": NAN,
           "ratio_per_doubling": NAN, "lnA": NAN}
    if res["n"] < 2:
        return res
    lo, hi, E = lo[m], hi[m], E[m]
    lq = np.log(hi / lo)
    x = 0.5 * np.log(lo * hi)
    y0 = np.log(E / lq)

    def solve(xs, ys, lqs):
        alpha = 0.0
        coef, A, ysc = None, None, None
        for _ in range(iters):
            ysc = ys - np.log(np.array([sinhc(alpha * l / 2.0) for l in lqs]))
            A = np.vstack([np.ones(len(xs)), xs]).T
            coef = np.linalg.lstsq(A, ysc, rcond=None)[0]
            alpha = float(coef[1])
        return coef, A, ysc

    coef, A, yc = solve(x, y0, lq)
    res["lnA"], res["alpha"] = float(coef[0]), float(coef[1])
    n = res["n"]
    if n > 2:
        r = yc - A @ coef
        cov = (float(r @ r) / (n - 2)) * np.linalg.inv(A.T @ A)
        res["se"] = float(math.sqrt(max(cov[1, 1], 0.0)))
    if n > 3:
        al = [solve(np.delete(x, i), np.delete(y0, i), np.delete(lq, i))[0][1] for i in range(n)]
        res["jack_half_range"] = float((max(al) - min(al)) / 2.0)
    vals = [v for v in (res["se"], res["jack_half_range"]) if np.isfinite(v)]
    res["err"] = float(max(vals)) if vals else NAN
    res["ratio_per_doubling"] = float(2.0 ** (-res["alpha"]))
    return res


def fit_even(s, y, orders=(2, 4)):
    """y = sum_k c_k s^k over the even orders, least squares in the scaled variable s/max|s|; the errors from the residual variance."""
    s = np.asarray(s, float)
    y = np.asarray(y, float)
    sc = float(np.max(np.abs(s))) or 1.0
    u = s / sc
    A = np.vstack([u ** k for k in orders]).T
    coef_u = np.linalg.lstsq(A, y, rcond=None)[0]
    r = y - A @ coef_u
    dof = len(y) - len(orders)
    if dof > 0:
        cov = (float(r @ r) / dof) * np.linalg.inv(A.T @ A)
        err_u = np.sqrt(np.maximum(np.diag(cov), 0.0))
    else:
        err_u = np.full(len(orders), NAN)
    scale = np.array([sc ** k for k in orders], float)
    return coef_u / scale, err_u / scale, r


def small_s(s, y, smax):
    """the s^2 coefficient of y(s) from y = c2 s^2 + c4 s^4 on 0 < s <= smax; the error is the larger of the fit's standard error and the
    change of c2 when smax is halved (the s^6 term's effect)."""
    s = np.asarray(s, float)
    y = np.asarray(y, float)
    out = {"c2": NAN, "c4": NAN, "se": NAN, "half_shift": NAN, "err": NAN, "n": 0, "smax": smax}
    m = np.isfinite(y) & (s > 0) & (s <= smax)
    if m.sum() < 3:
        return out
    coef, err, _ = fit_even(s[m], y[m])
    out.update(c2=float(coef[0]), c4=float(coef[1]), se=float(err[0]), n=int(m.sum()))
    m2 = np.isfinite(y) & (s > 0) & (s <= smax / 2.0)
    if m2.sum() >= 3:
        out["half_shift"] = float(abs(fit_even(s[m2], y[m2])[0][0] - coef[0]))
    elif m2.sum() >= 2:
        out["half_shift"] = float(abs(fit_even(s[m2], y[m2], (2,))[0][0] - coef[0]))
    vals = [v for v in (out["se"], out["half_shift"]) if np.isfinite(v)]
    out["err"] = float(max(vals)) if vals else NAN
    return out


def intercept(s, y, npts=3):
    """a + b s on the first npts finite points with s > 0 (the s to 0 extrapolation of a quantity linear in s)."""
    s = np.asarray(s, float)
    y = np.asarray(y, float)
    m = np.isfinite(y) & (s > 0)
    s_, y_ = s[m][:npts], y[m][:npts]
    if len(s_) < 2:
        return NAN, NAN
    A = np.vstack([np.ones(len(s_)), s_]).T
    a, b = np.linalg.lstsq(A, y_, rcond=None)[0]
    return float(a), float(b)


def ratio_err(num, den, enum, eden):
    """num/den with the propagated error sqrt((enum/den)^2 + (num eden/den^2)^2)."""
    if not (np.isfinite(num) and np.isfinite(den)) or den == 0:
        return NAN, NAN
    r = num / den
    e1 = (enum / den) if np.isfinite(enum) else 0.0
    e2 = (num * eden / den ** 2) if np.isfinite(eden) else 0.0
    return float(r), float(math.sqrt(e1 ** 2 + e2 ** 2))


# ------------------------------------------------------------------------------------------------- B1 (a): the switch-on's excitation
def phi_sine(x):
    """Phi(x) = int_0^1 F'(u) cos(x (u - 1/2)) du for the records' ramp F(u) = sin^2(pi u/2), F'(u) = (pi/2) sin(pi u):
    pi^2 cos(x/2)/(pi^2 - x^2) = (pi^2/(2 (pi + x))) sin(z)/z with z = (pi - x)/2; even, Phi(0) = 1, Phi(pi) = pi/4, Phi(2 pi) = 1/3."""
    x = np.abs(np.asarray(x, float))
    z = (math.pi - x) / 2.0
    return (math.pi ** 2 / (2.0 * (math.pi + x))) * np.sinc(z / math.pi)


def phi_smoothstep(x):
    """the same for F(u) = 3 u^2 - 2 u^3 (the idea file's ramp): 3 j_1(x/2)/(x/2), with the small-argument form 1 - (x/2)^2/10."""
    x = np.abs(np.asarray(x, float))
    y = x / 2.0
    small = y < 1e-4
    ys = np.where(small, 1.0, y)
    val = 3.0 * (np.sin(ys) - ys * np.cos(ys)) / ys ** 3
    return np.where(small, 1.0 - y ** 2 / 10.0, val)


PHI = {"sine": phi_sine, "smoothstep": phi_smoothstep}


def ramp_filter(part, x, theta=0.0, ramp="sine"):
    """the filters of the second-order excitation read at t_h = theta tau after the ramp, x = Delta tau (addendum 1.1):
    gen Phi^2 (the pair energy), flow 2 Phi cos(x (1/2 + theta)) (the particle-hole coherences against V), clock = gen - flow; static 1."""
    x = np.asarray(x, float)
    if part == "static":
        return np.ones_like(x)
    ph = PHI[ramp](x)
    if part == "gen":
        return ph * ph
    fl = 2.0 * ph * np.cos(x * (0.5 + theta))
    if part == "flow":
        return fl
    if part == "clock":
        return ph * ph - fl
    raise ValueError(part)


def I_part(part, theta=0.0, ramp="sine", X=4000.0, dx=0.01):
    """I_X = int_0^inf x W_X(x) dx by Simpson on [0, X] plus the analytic tail of the gen part (pi^4/(4 X^2) sine, 144/(4 X^2)
    smoothstep). The flow part at theta = 0 diverges logarithmically (returned as inf); at theta > 0 use the closed form -2 J(theta)."""
    if part in ("flow", "clock") and theta <= 0:
        return float("inf") if part == "clock" else float("-inf")
    x = np.arange(0.0, X + dx / 2, dx)
    W = ramp_filter(part, x, theta, ramp)
    val = simpson(x * W, x)
    if part in ("gen", "clock"):
        amp = math.pi ** 4 if ramp == "sine" else 144.0
        val += amp / (4.0 * X ** 2)
    return float(val)


def I_gen_st(ramp="sine", n=801):
    """I_gen by the identity int_0^inf x |g^(x)|^2 dx = (1/2) intint |g(s) - g(t)|^2/(s - t)^2 ds dt for g = F' on [0, 1] (zero outside):
    the in-in double integral plus the in-out part 2 int_0^1 g^2/s ds."""
    s = np.linspace(0.0, 1.0, n)
    if ramp == "sine":
        g = (math.pi / 2.0) * np.sin(math.pi * s)
        gp = (math.pi ** 2 / 2.0) * np.cos(math.pi * s)
    else:
        g = 6.0 * s * (1.0 - s)
        gp = 6.0 - 12.0 * s
    S, T = np.meshgrid(s, s, indexing="ij")
    G1, G2 = np.meshgrid(g, g, indexing="ij")
    D = S - T
    with np.errstate(divide="ignore", invalid="ignore"):
        Q = np.where(np.abs(D) > 1e-12, (G1 - G2) ** 2 / D ** 2, 0.0)
    Q[np.arange(n), np.arange(n)] = gp ** 2
    h = s[1] - s[0]
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    w *= h / 3.0
    in_in = 0.5 * float(w @ Q @ w)
    with np.errstate(divide="ignore", invalid="ignore"):
        f2 = np.where(s > 0, g ** 2 / np.where(s > 0, s, 1.0), 0.0)
    in_out = 2.0 * simpson(f2, s)
    return float(in_in + in_out), float(in_in), float(in_out)


def J_theta(theta):
    """J(theta) = int_0^1 (pi/2) sin(pi u) (1 + theta - u)^-2 du, theta = t_h/tau > 0: (pi^2/2)[cos(pi theta)(Ci(pi(1+theta)) - Ci(pi theta))
    + sin(pi theta)(Si(pi(1+theta)) - Si(pi theta))]; the flow part's integral is I_flow = -2 J(theta); J ~ (pi^2/2) ln(1/theta) as theta -> 0."""
    if theta <= 0:
        return float("inf")
    from scipy.special import sici
    si1, ci1 = sici(math.pi * (1.0 + theta))
    si0, ci0 = sici(math.pi * theta)
    return float((math.pi ** 2 / 2.0) * (math.cos(math.pi * theta) * (ci1 - ci0) + math.sin(math.pi * theta) * (si1 - si0)))


def J_theta_quad(theta, n=20001):
    u = np.linspace(0.0, 1.0, n)
    return simpson((math.pi / 2.0) * np.sin(math.pi * u) / (1.0 + theta - u) ** 2, u)


def continuum(BL, tau, t_h, ramp="sine"):
    """the 3D valley-continuum prediction per e-fold over the full-zone slope (addendum 1.3): B_X/B_L = I_X/(15 pi^2 a^2 B_L) (a/tau)^2,
    I_gen = int x Phi^2 dx, I_flow = -2 J(t_h/tau), I_clock = I_gen - I_flow (infinite at t_h = 0: the logarithm the hold cuts)."""
    Ig = I_part("gen", 0.0, ramp)
    theta = t_h / tau if tau else NAN
    J = J_theta(theta) if theta > 0 else float("inf")
    If = -2.0 * J
    Ic = Ig - If
    den = 15.0 * math.pi ** 2 * BL
    return {"tau": tau, "theta": theta, "I_gen": Ig, "J": J, "I_flow": If, "I_clock": Ic, "gen_over_slope": Ig / den / tau ** 2,
            "flow_over_slope": If / den / tau ** 2, "clock_over_slope": Ic / den / tau ** 2, "B_gen_over_BL_coefficient": Ig / den, "B_clock_over_BL_coefficient": Ic / den}


def psi_part(y, theta=0.0, part="gen", ramp="sine", n_u=6001, umax=12.0):
    """the M -> 0 channel kernel Psi_X(y) = int_{-inf}^{inf} sinh^2 u cosh^-4 u W_X(y cosh u) du (Psi_static = 2/3), y = 2 tau M."""
    u = np.linspace(0.0, umax, n_u)
    ch = np.cosh(u)
    wgt = np.sinh(u) ** 2 / ch ** 4
    ys = np.atleast_1d(np.asarray(y, float))
    out = np.empty(len(ys))
    for i, yy in enumerate(ys):
        out[i] = 2.0 * simpson(wgt * ramp_filter(part, yy * ch, theta, ramp), u)
    return out if np.ndim(y) else float(out[0])


def chain_kernel(M, tau, theta, parts, n_u=4001, ramp="sine"):
    """the k-integral of the channel-local second-order density per unit weight (addendum 1.4, the idea file's (15) with the filter):
    I_X(M) = int_0^{pi/2} dk M^2 sin^2 k [cos^2 k/2 + cos 2k]^2 W_X(2 tau eps)/eps^5, eps = sqrt(sin^2 k + M^2), M = kappa/r; k = M sinh u,
    Simpson in u. Static (W = 1): 3/4 as M -> 0 ((9/8) Psi_static)."""
    n = n_u if n_u % 2 == 1 else n_u + 1
    umax = math.asinh(0.5 * math.pi / M)
    u = np.linspace(0.0, umax, n)
    k = M * np.sinh(u)
    s2 = np.sin(k) ** 2
    c = np.cos(k)
    eps = np.sqrt(s2 + M * M)
    base = (M * np.cosh(u)) * M * M * s2 * (0.5 * c * c + np.cos(2.0 * k)) ** 2 / eps ** 5
    x = 2.0 * tau * eps
    return {p: simpson(base * ramp_filter(p, x, theta, ramp), u) for p in parts}


_GL = {}


def gauss_legendre(n):
    if n not in _GL:
        _GL[n] = np.polynomial.legendre.leggauss(n)
    return _GL[n]


def chain_annulus(rs, tau, theta, kappas, weights, R1, R2, parts, n_r=12, n_u=4001, ramp="sine", cache=None, shift=0.5):
    """per-channel annulus energies (weights included) of the parts over the sites R in (R1, R2], read as the integral over
    [R1 + shift, R2 + shift] (shift 1/2 for the site convention R in (lo r_s, hi r_s]): E_X,kappa = w int (r_s/r^3)/(4 pi) I_X(kappa/r) dr,
    Gauss-Legendre in ln r. Returns dict part -> array over the channels."""
    nodes, wts = gauss_legendre(n_r)
    a_, b_ = math.log(R1 + shift), math.log(R2 + shift)
    ell = 0.5 * (b_ - a_) * nodes + 0.5 * (b_ + a_)
    r = np.exp(ell)
    jac = 0.5 * (b_ - a_) * wts * r
    out = {p: np.zeros(len(kappas)) for p in parts}
    for i, (kap, w) in enumerate(zip(kappas, weights)):
        for rr, jj in zip(r, jac):
            key = (float(rs), round(float(tau), 9), round(float(theta), 12), int(kap), round(float(rr), 9), ramp, int(n_u))
            ker = cache.get(key) if cache is not None else None
            need = [p for p in parts if ker is None or p not in ker]
            if need:
                k2 = chain_kernel(float(kap) / float(rr), tau, theta, need, n_u, ramp)
                ker = dict(ker or {})
                ker.update(k2)
                if cache is not None:
                    cache[key] = ker
            for p in parts:
                out[p][i] += w * jj * (rs / rr ** 3) / (4.0 * math.pi) * ker[p]
    return out


def law_annulus(rs, weights, R1, R2, shift=0.5):
    """the M -> 0 static law per channel, w (3/(32 pi)) r_s ((R1 + shift)^-2 - (R2 + shift)^-2) (the idea file's (19) summed gives
    (3K(K+1)/(4 pi)) r_s (R1^-2 - R2^-2) for the weights 16 kappa)."""
    return np.array([w * (3.0 / (32.0 * math.pi)) * rs * ((R1 + shift) ** -2 - (R2 + shift) ** -2) for w in weights])


def predict_bins(rs, tau, t_h, kappas, weights, bins, cache, n_r, n_u, ramp, shift=0.5):
    """arrays [n_bins, n_channels]: static, gen, flow, clock at theta = t_h/tau (the hold's end), gen_end, flow_end, clock_end at theta 0."""
    theta = t_h / tau
    keys = ("static", "gen", "flow", "clock", "gen_end", "flow_end", "clock_end")
    res = {k: np.zeros((len(bins), len(kappas))) for k in keys}
    for j, (lo, hi) in enumerate(bins):
        R1, R2 = lo * rs, hi * rs
        ph = chain_annulus(rs, tau, theta, kappas, weights, R1, R2, ("static", "gen", "flow", "clock"), n_r, n_u, ramp, cache, shift)
        pe = chain_annulus(rs, tau, 0.0, kappas, weights, R1, R2, ("gen", "flow", "clock"), n_r, n_u, ramp, cache, shift)
        for k in ("static", "gen", "flow", "clock"):
            res[k][j] = ph[k]
        res["gen_end"][j], res["flow_end"][j], res["clock_end"][j] = pe["gen"], pe["flow"], pe["clock"]
    return res


# ----------------------------------------------------------------------------------------------------------------------- vf9/vf10 records
class VF:
    """an engine_vacflow_v9/v10 record."""

    def __init__(self, D, name, path=""):
        self.D, self.name, self.path = D, name, path
        self.rs = float(D["args"]["rs"])
        self.ch = D["channels"]
        self.nfull = max(len(c["reads"]) for c in self.ch)
        first = next(c for c in self.ch if c["reads"])
        self.cuts = sorted(first["reads"][0]["cuts"].keys(), key=float)
        self.t_r = float(D["args"].get("tr", 32.0)) * self.rs
        self.current = str(D["args"].get("chain_current", "?"))

    def w(self, c):
        return float(c.get("weight", 16.0 * c["kappa"]))

    def complete(self, c):
        return (not c.get("rejected")) and len(c["reads"]) == self.nfull

    def arr(self, c, key):
        return np.array([q.get(key, NAN) for q in c["reads"]], float)

    def cut_arr(self, c, ck, key):
        return np.array([q.get("cuts", {}).get(ck, {}).get(key, NAN) for q in c["reads"]], float)

    def cut_rs(self):
        return min(self.cuts, key=lambda k: abs(float(k) - self.rs))

    def regions(self):
        reg = [("inside %g r_s" % (float(k) / self.rs), "cut", (k,)) for k in self.cuts]
        for lo, hi in zip(self.cuts[:-1], self.cuts[1:]):
            reg.append(("annulus [%g, %g] r_s" % (float(lo) / self.rs, float(hi) / self.rs), "annulus", (lo, hi)))
        return reg

    def smooth_tail(self):
        if "smooth_tail" in self.D:
            return float(self.D["smooth_tail"])
        sw = float(self.D["args"].get("smooth_width", 7.0))
        swin = int(self.D["args"].get("smooth_window", 64))
        return 0.5 * float(_erfc((swin // 2) / sw))


def decompose(V, c):
    """per readout, inside each cut: the generator, flow and clock energies of R0 relative to the flat sea (ref; static), of C relative
    to the flat sea (C; raw) and of C relative to R0 (NO). gen = E_K difference, flow = s times the V-expectation inside the cut read as
    W/sdot where sdot != 0 (nan where sdot = 0, 0 where s = 0), clock = gen - flow."""
    s = V.arr(c, "s")
    sdot = V.arr(c, "sdot")
    ok = sdot != 0
    d = {"s": s, "sdot": sdot, "t": V.arr(c, "t_over_rs") * V.rs, "phase": [q.get("phase", "?") for q in c["reads"]], "ok": ok, "cuts": {}}
    for ck in V.cuts:
        EKC = V.cut_arr(c, ck, "E_K_C")
        EKR0 = V.cut_arr(c, ck, "E_K_R0")
        WC = V.cut_arr(c, ck, "W_C")
        WR0 = V.cut_arr(c, ck, "W_R0")
        W = V.cut_arr(c, ck, "W")
        with np.errstate(divide="ignore", invalid="ignore"):
            TrCV = np.where(ok, WC / sdot, NAN)
            TrR0V = np.where(ok, WR0 / sdot, NAN)
            TrDV = np.where(ok, W / sdot, NAN)
        q = {"TrCV": TrCV, "TrR0V": TrR0V}
        for tag, EA, E0, T in (("ref", EKR0, EKR0[0], TrR0V), ("C", EKC, EKC[0], TrCV)):
            gen = EA - E0
            flow = np.where(s == 0, 0.0, s * T)
            q["gen_" + tag], q["flow_" + tag], q["clock_" + tag] = gen, flow, gen - flow
        gen = EKC - EKR0
        flow = np.where(s == 0, 0.0, s * TrDV)
        q["gen_NO"], q["flow_NO"], q["clock_NO"] = gen, flow, gen - flow
        d["cuts"][ck] = q
    return d


def weighted_sum(V, decs, key, ck, which="all"):
    """the weighted channel sum per readout index of a decomposed quantity; 'all' sums the channels present at the index, 'complete'
    only the complete channels; nan where no channel contributes."""
    tot = np.zeros(V.nfull)
    cnt = np.zeros(V.nfull, int)
    for c, d in zip(V.ch, decs):
        if which == "complete" and not V.complete(c):
            continue
        y = np.asarray(d["cuts"][ck][key], float)
        k = len(y)
        m = np.isfinite(y)
        tot[:k][m] += V.w(c) * y[m]
        cnt[:k][m] += 1
    return np.where(cnt > 0, tot, NAN), cnt


def weighted_region(V, decs, key, keys, which):
    if len(keys) == 1:
        return weighted_sum(V, decs, key, keys[0], which)
    y_hi, cnt = weighted_sum(V, decs, key, keys[1], which)
    y_lo, _ = weighted_sum(V, decs, key, keys[0], which)
    return y_hi - y_lo, cnt


def weighted_top(V, key, which="all"):
    tot = np.zeros(V.nfull)
    cnt = np.zeros(V.nfull, int)
    for c in V.ch:
        if which == "complete" and not V.complete(c):
            continue
        y = V.arr(c, key)
        k = len(y)
        m = np.isfinite(y)
        tot[:k][m] += V.w(c) * y[m]
        cnt[:k][m] += 1
    return np.where(cnt > 0, tot, NAN), cnt


def analyze_vf(V, a, RES, md):
    slope = slope_full_zone(a.BL, V.rs)
    tail = V.smooth_tail()
    out = {"file": V.path, "rs": V.rs, "chain_current": V.current, "n_channels": len(V.ch), "n_readouts_full": V.nfull, "cuts": V.cuts,
           "slope_full_zone_per_efold": slope, "smooth_tail": tail}
    comp = [c["kappa"] for c in V.ch if V.complete(c)]
    out["complete_channels"] = comp
    out["channels_apart"] = [{"kappa": c["kappa"], "readouts": len(c["reads"]), "why": str(c.get("rejected") or "readout count")} for c in V.ch if not V.complete(c)]
    decs = [decompose(V, c) for c in V.ch]
    iref = next((i for i, c in enumerate(V.ch) if V.complete(c)), int(np.argmax([len(c["reads"]) for c in V.ch])))
    s_grid, sdot_grid, phase = decs[iref]["s"], decs[iref]["sdot"], decs[iref]["phase"]
    ok = sdot_grid != 0
    ramp_ok = [i for i in range(V.nfull) if phase[i] == "ramp" and ok[i]]
    i_end = max(ramp_ok) if ramp_ok else V.nfull - 1
    i_hold = V.nfull - 1
    out["readouts"] = {"s_first_12": s_grid[:12].tolist(), "i_end_of_ramp": int(i_end), "s_end": float(s_grid[i_end]), "sdot_end": float(sdot_grid[i_end]),
                       "t_end_over_rs": float(decs[iref]["t"][i_end] / V.rs), "t_hold_end_over_rs": float(decs[iref]["t"][i_hold] / V.rs)}
    md += ["## %s: %s" % (V.name, V.path), "",
           "- r_s %g, chain current %s, %d channels, %d readouts on the full grid; complete channels %s; apart: %s" % (
               V.rs, V.current, len(V.ch), V.nfull, comp, [(x["kappa"], x["readouts"]) for x in out["channels_apart"]] or "none"),
           "- the full-zone flat slope (9 pi/2) B_L r_s = %.4f per e-fold at a^2 B_L = %g; the smoothed cut's tail at the window's edge %.2e" % (slope, a.BL, tail),
           "- the ramp's last readout with sdot != 0: index %d, t/r_s %.2f, s %.8f, sdot %.3e (the flow part at s = 1 is read there by W/sdot)" % (
               i_end, out["readouts"]["t_end_over_rs"], out["readouts"]["s_end"], out["readouts"]["sdot_end"]), ""]

    # (1) the continued-reference residual
    res = {}
    rows = []
    for ck in V.cuts:
        acc = {k: [] for k in ("I_C", "dI_R2", "dI_R0", "IS_C", "dIS_R2", "dIS_R0")}
        integ_R2, integ_R0, have_R2 = 0.0, 0.0, False
        for c in V.ch:
            t = V.arr(c, "t_over_rs") * V.rs
            IC = V.cut_arr(c, ck, "I_C")
            INO = V.cut_arr(c, ck, "I_NO")
            IR2 = V.cut_arr(c, ck, "I_R2")
            ISC = V.cut_arr(c, ck, "IS_C")
            ISNO = V.cut_arr(c, ck, "IS_NO")
            ISR2 = V.cut_arr(c, ck, "IS_R2")
            acc["I_C"].append(IC)
            acc["dI_R0"].append(IC - INO)
            acc["dI_R2"].append(IC - IR2)
            acc["IS_C"].append(ISC)
            acc["dIS_R0"].append(ISC - ISNO)
            acc["dIS_R2"].append(ISC - ISR2)
            if np.isfinite(IR2).any():
                have_R2 = True
                integ_R2 += V.w(c) * trapz(np.abs(IC - IR2), t)
            integ_R0 += V.w(c) * trapz(np.abs(IC - INO), t)
        scale = nmax(np.abs(np.concatenate(acc["I_C"])))

        def stat(key, acc=acc):
            v = np.concatenate(acc[key])
            v = v[np.isfinite(v)]
            return {"max_abs": float(np.max(np.abs(v))) if len(v) else NAN, "rms": float(np.sqrt(np.mean(v ** 2))) if len(v) else NAN, "n": int(len(v))}

        r = {"R_over_rs": float(ck) / V.rs, "max_abs_I_C": scale, "sharp_R2": stat("dI_R2"), "sharp_R0": stat("dI_R0"), "smooth_R2": stat("dIS_R2"),
             "smooth_R0": stat("dIS_R0"), "integrated_abs_R2_weighted": integ_R2 if have_R2 else NAN, "integrated_abs_R0_weighted": integ_R0}
        r["integrated_R2_over_slope_ln2"] = r["integrated_abs_R2_weighted"] / (slope * math.log(2.0))
        r["sharp_R2_rel_info"] = r["sharp_R2"]["max_abs"] / scale if scale else NAN
        r["smooth_R2_rel_info"] = r["smooth_R2"]["max_abs"] / scale if scale else NAN
        r["smooth_R2_over_tail"] = r["smooth_R2"]["max_abs"] / tail if tail else NAN
        r["smooth_R0_over_tail"] = r["smooth_R0"]["max_abs"] / tail if tail else NAN
        res[ck] = r
        rows.append([fmt(r["R_over_rs"], "%g"), fmt(scale), fmt(r["sharp_R2"]["max_abs"]), fmt(r["sharp_R2"]["rms"]), fmt(r["sharp_R0"]["max_abs"]),
                     fmt(r["smooth_R2"]["max_abs"]), fmt(r["smooth_R0"]["max_abs"]), fmt(r["sharp_R2_rel_info"], "%.2e"), fmt(r["smooth_R2_over_tail"], "%.2f"),
                     fmt(r["integrated_R2_over_slope_ln2"], "%.2e")])
    out["residual"] = res
    md += ["### (1) The continued-reference residual: the reference's Killing-energy current through each cut, I_C - I_R2 = (i/2) Tr[R2 [h^2, P_c]] (zero for a projector commuting with h(s); the measured value is the rounding of O(1) summands); R0 likewise; the smoothed cut IS_C - IS_R2 carries the window truncation of the erf step's tail and is read in units of the tail", ""]
    table(md, ["cut R/r_s", "max |I_C| (information)", "sharp R2 max", "sharp R2 rms", "sharp R0 max", "smooth R2 max", "smooth R0 max", "sharp R2 / max|I_C| (information)",
               "smooth R2 / tail", "sum_k w int |I_C - I_R2| dt / (slope ln 2)"], rows)
    sharp_abs = nmax([r["sharp_R2"]["max_abs"] for r in res.values()])
    sharp_rms = nmax([r["sharp_R2"]["rms"] for r in res.values()])
    sharp_rel = nmax([r["sharp_R2_rel_info"] for r in res.values()])
    smooth_tails = nmax([r["smooth_R2_over_tail"] for r in res.values()])
    integ_rel = nmax([r["integrated_R2_over_slope_ln2"] for r in res.values()])
    verdict(RES, "%s T1 continued-reference residual, sharp cut: max |I_C - I_R2| over channels, readouts and cuts (absolute)" % V.name, sharp_abs,
            "< %g (the rounding of O(1) summands)" % RULES["residual_sharp_abs"], (sharp_abs < RULES["residual_sharp_abs"]) if np.isfinite(sharp_abs) else None,
            "rms %s; relative to the record's largest |I_C| %s (information: that scale depends on how far the pocket channels ran)" % (fmt(sharp_rms, "%.1e"), fmt(sharp_rel, "%.1e")))
    verdict(RES, "%s T1 continued-reference residual, smoothed cut: max |IS_C - IS_R2| in units of the step's tail at the window's edge" % V.name, smooth_tails,
            "< %g tails (tail %.2e)" % (RULES["residual_smooth_over_tail"], tail), (smooth_tails < RULES["residual_smooth_over_tail"]) if np.isfinite(smooth_tails) else None,
            "consistent with the window truncation of the erf step; the sharp cut is the exact statement")
    verdict(RES, "%s T1 integrated |I_C - I_R2| (weighted) over slope ln 2" % V.name, integ_rel, "< %g" % RULES["residual_integrated_over_slope"],
            (integ_rel < RULES["residual_integrated_over_slope"]) if np.isfinite(integ_rel) else None, "an upper bound on any energy the reference could have moved across a cut over the run")

    # (2) the field identities at the cut r_s
    ck0 = V.cut_rs()
    dNO, draw, sc = [], [], []
    for c, d in zip(V.ch, decs):
        q = d["cuts"][ck0]
        m = d["ok"]
        EinNO = V.arr(c, "E_in_NO")
        Einraw = V.arr(c, "E_in_raw")
        dNO.append(nmax(np.abs(q["clock_NO"][m] - EinNO[m])))
        draw.append(nmax(np.abs(q["clock_C"][m] - Einraw[m])))
        sc.append(nmax(np.abs(np.concatenate([EinNO, Einraw, V.cut_arr(c, ck0, "E_K_C")]))))
    ident = {"cut": ck0, "max_dev_E_in_NO": nmax(dNO), "max_dev_E_in_raw": nmax(draw), "scale": nmax(sc)}
    ident["rel"] = max(ident["max_dev_E_in_NO"], ident["max_dev_E_in_raw"]) / ident["scale"] if ident["scale"] else NAN
    out["identity"] = ident
    md += ["### (2) The field identities at the cut R = r_s (the reading of E_K, W and E_in): E_in_NO = E_K_C - E_K_R0 - s W/sdot and E_in_raw = E_K_C(t) - E_K_C(0) - s W_C/sdot over the readouts with sdot != 0", "",
           "- max deviation %.3e (NO) and %.3e (raw) against the scale %.3e: relative %.2e" % (ident["max_dev_E_in_NO"], ident["max_dev_E_in_raw"], ident["scale"], ident["rel"]), ""]
    verdict(RES, "%s T2 field identities at r_s (relative deviation)" % V.name, ident["rel"], "< %g" % RULES["identity_rel"], (ident["rel"] < RULES["identity_rel"]) if np.isfinite(ident["rel"]) else None)

    # (3) the decomposition: the small-s coefficients, the ramp's end, the hold's end
    dec = {}
    rows_small, rows_end, rows_hold = [], [], []
    idx_s = np.where((s_grid > 0) & (s_grid <= a.smax))[0]
    for label, kind, keys in V.regions():
        lnr = math.log(float(keys[1]) / float(keys[0])) if kind == "annulus" else NAN
        reg = {"kind": kind, "keys": list(keys), "ln_R_ratio": lnr, "small_s": {}, "end_of_ramp": {}, "hold_end": {}}
        for key in QUANT:
            y_all, cnt = weighted_region(V, decs, key, keys, "all")
            fit = small_s(s_grid, y_all, a.smax)
            fit["channels_summed_at_smax"] = int(cnt[idx_s[-1]]) if len(idx_s) else 0
            reg["small_s"][key] = fit
            y_c, _ = weighted_region(V, decs, key, keys, "complete")
            reg["end_of_ramp"][key] = float(y_c[i_end])
            reg["hold_end"][key] = float(y_c[i_hold])
        c2 = {k: reg["small_s"][k]["c2"] for k in QUANT}
        e2 = {k: reg["small_s"][k]["err"] for k in QUANT}
        B = -c2["flow_ref"]
        own2 = c2["gen_ref"] - c2["flow_ref"] / 2.0
        rf, rfe = ratio_err(c2["flow_ref"], c2["clock_ref"], e2["flow_ref"], e2["clock_ref"])
        rg, rge = ratio_err(c2["gen_ref"], c2["clock_ref"], e2["gen_ref"], e2["clock_ref"])
        der = {"B": B, "B_err": e2["flow_ref"], "own2": own2, "own2_over_B": (own2 / B) if B else NAN, "own_share_2own_over_B": (2 * own2 / B) if B else NAN,
               "local_share_halfB_over_clock2": (0.5 * B / c2["clock_ref"]) if c2["clock_ref"] else NAN,
               "flow_over_clock": rf, "flow_over_clock_err": rfe, "gen_over_clock": rg, "gen_over_clock_err": rge,
               "ratio_predicted_from_own_share": [(-2.0 / (1.0 + 2 * own2 / B)) if (B and (1.0 + 2 * own2 / B) != 0) else NAN,
                                                  ((-1.0 + 2 * own2 / B) / (1.0 + 2 * own2 / B)) if (B and (1.0 + 2 * own2 / B) != 0) else NAN],
               "signs_ok": bool(np.isfinite(c2["clock_ref"]) and c2["clock_ref"] > 0 and c2["flow_ref"] < 0 and c2["gen_ref"] < 0),
               "ratio_within_tol": bool(np.isfinite(rf) and np.isfinite(rg) and abs(rf + 2.0) <= 2.0 * RULES["ratio_tol"] and abs(rg + 1.0) <= RULES["ratio_tol"]),
               "excitation_c2_gen_NO": c2["gen_NO"], "excitation_c4_gen_NO": reg["small_s"]["gen_NO"]["c4"]}
        if kind == "annulus":
            der["slope_per_efold_s2"] = {k: c2[k] / lnr for k in ("clock_ref", "flow_ref", "gen_ref")}
        reg["derived"] = der
        dec[label] = reg
        rows_small.append([label, fmt(c2["clock_ref"]), fmt(c2["flow_ref"]), fmt(c2["gen_ref"]), fmt(e2["clock_ref"], "%.1e"), fmt(B), fmt(own2),
                           fmt(der["own_share_2own_over_B"], "%.3e"), fmt(der["local_share_halfB_over_clock2"], "%.3e"),
                           "%s +- %s" % (fmt(rf, "%.4f"), fmt(rfe, "%.1e")), "%s +- %s" % (fmt(rg, "%.4f"), fmt(rge, "%.1e")), fmt(der["signs_ok"]), fmt(der["ratio_within_tol"])])
        e = reg["end_of_ramp"]
        rows_end.append([label] + [fmt(e[k]) for k in QUANT])
        h = reg["hold_end"]
        rows_hold.append([label, fmt(h["gen_ref"]), fmt(h["gen_C"]), fmt(h["gen_NO"])])
    out["decomposition"] = dec
    md += ["### (3) The clock, flow and generator parts inside the cuts and in the annuli between them: the s^2 coefficients at 0 < s <= %g of the static response of R0 relative to the flat sea (weighted channel sums; the virial prediction clock : flow : generator = 1 : -2 : -1 up to the cut's own term: clock2 = B/2 + own2, flow2 = -B, gen2 = -B/2 + own2, own share = 2 own2/B, local share = (B/2)/clock2; the ratios predicted from the own share are -2/(1 + own share) and (-1 + own share)/(1 + own share))" % a.smax, ""]
    table(md, ["region", "clock_ref c2", "flow_ref c2", "gen_ref c2", "err(clock c2)", "B = -flow c2", "own2 = gen2 - flow2/2", "own share 2 own2/B", "local share (B/2)/clock2",
               "flow/clock (-2)", "gen/clock (-1)", "signs", "within tol"], rows_small)
    md += ["The same parts at the ramp's last readout (s = %.6f; complete channels; the exterior annuli have v^2 = 0.5 to 0.67 there and no quadratic ratio is expected):" % out["readouts"]["s_end"], ""]
    table(md, ["region"] + list(QUANT), rows_end)
    md += ["The generators at the hold's end (t/r_s %.1f; the flow parts are not readable at sdot = 0):" % out["readouts"]["t_hold_end_over_rs"], ""]
    table(md, ["region", "gen_ref (R0 - flat)", "gen_C (C - flat)", "gen_NO (C - R0)"], rows_hold)
    big = "inside %g r_s" % (float(V.cuts[-1]) / V.rs)
    dbig = dec[big]["derived"]
    verdict(RES, "%s T4 signs of the s^2 response (clock > 0, flow < 0, generator < 0) in %s" % (V.name, big), dbig["signs_ok"], "all three", dbig["signs_ok"])
    verdict(RES, "%s T4 ratios flow/clock, gen/clock in %s" % (V.name, big), [dbig["flow_over_clock"], dbig["gen_over_clock"]],
            "within %g of -2 and -1" % RULES["ratio_tol"], dbig["ratio_within_tol"],
            "own share 2 own2/B = %s; the ratios predicted from it %s, %s" % (fmt(dbig["own_share_2own_over_B"], "%.3e"), fmt(dbig["ratio_predicted_from_own_share"][0], "%.4f"), fmt(dbig["ratio_predicted_from_own_share"][1], "%.4f")))
    for label, reg in dec.items():
        dr = reg["derived"]
        if reg["kind"] == "cut" and label != big:
            verdict(RES, "%s T4 ratios flow/clock, gen/clock %s" % (V.name, label), [dr["flow_over_clock"], dr["gen_over_clock"]],
                    "the rule (within %g of -2 and -1) is set for the largest cut; the smaller cuts are reported" % RULES["ratio_tol"], None,
                    "own share 2 own2/B = %s; predicted from it %s, %s; signs as predicted: %s" % (fmt(dr["own_share_2own_over_B"], "%.3e"), fmt(dr["ratio_predicted_from_own_share"][0], "%.4f"),
                                                                                                   fmt(dr["ratio_predicted_from_own_share"][1], "%.4f"), fmt(dr["signs_ok"])), kind="reported")
        elif reg["kind"] == "annulus":
            verdict(RES, "%s T4 the %s: flow/clock, gen/clock" % (V.name, label), [dr["flow_over_clock"], dr["gen_over_clock"]],
                    "no rule: the virial ratio is a statement for a region whose boundary term is small; the own share and the local share say how far this region is from that", None,
                    "own2 = %s, own share 2 own2/B = %s, local share (B/2)/clock2 = %s, B_ann = %s; signs as predicted: %s" % (
                        fmt(dr["own2"]), fmt(dr["own_share_2own_over_B"], "%.3e"), fmt(dr["local_share_halfB_over_clock2"], "%.3e"), fmt(dr["B"]), fmt(dr["signs_ok"])), kind="reported")

    # the window-Euler reading (B1 test (b)): Q_f = -2 Q_c + beta, beta/Q_c = flow/clock + 2 = gen/clock + 1
    rows = []
    we = {}
    for label, reg in dec.items():
        c2 = {k: reg["small_s"][k]["c2"] for k in ("clock_ref", "flow_ref", "gen_ref")}
        e1 = reg["end_of_ramp"]
        bf = (c2["flow_ref"] / c2["clock_ref"] + 2.0) if c2["clock_ref"] else NAN
        bg = (c2["gen_ref"] / c2["clock_ref"] + 1.0) if c2["clock_ref"] else NAN
        b1 = (e1["flow_ref"] / e1["clock_ref"] + 2.0) if e1["clock_ref"] else NAN
        we[label] = {"beta_over_Qc_small_s_flow": bf, "beta_over_Qc_small_s_gen": bg, "beta_over_Qc_s1_flow": b1}
        rows.append([label, fmt(bf, "%.4f"), fmt(bg, "%.4f"), fmt(b1, "%.4f")])
    out["window_euler"] = we
    md += ["The window-Euler reading of the same regions (B1 test (b), addendum 2.1): Q_f = -2 Q_c + beta with beta = [f dL/df'] the region's boundary term, so beta/Q_c = flow/clock + 2 = gen/clock + 1; 0 is the full-zone exterior law (1 : -2 : -1, beta constant in R), 8/3 the channel-local far law (1 : 2/3 : 5/3, beta ~ R^-2); inside the cuts beta is the cut's own term", ""]
    table(md, ["region", "beta/Q_c from flow (s^2 coefficients)", "from gen", "from flow at s = 1"], rows)
    ann_labels = [l for l, r_ in dec.items() if r_["kind"] == "annulus"]
    verdict(RES, "%s T8b the window-Euler boundary term over Q_c in the exterior annuli %s" % (V.name, ", ".join(ann_labels)), [we[l]["beta_over_Qc_small_s_flow"] for l in ann_labels],
            "reported: 0 is the full-zone law, 8/3 the channel-local far law; the annuli to 2 r_s are in neither regime", None,
            "from the generator %s; at s = 1 %s" % (", ".join(fmt(we[l]["beta_over_Qc_small_s_gen"], "%.4f") for l in ann_labels), ", ".join(fmt(we[l]["beta_over_Qc_s1_flow"], "%.4f") for l in ann_labels)), kind="reported")

    # per channel at the largest cut
    rows = []
    perch = []
    for c, d in zip(V.ch, decs):
        q = d["cuts"][V.cuts[-1]]
        f = {k: small_s(d["s"], q[k], a.smax)["c2"] for k in ("clock_ref", "flow_ref", "gen_ref")}
        Bk = -f["flow_ref"]
        ownk = f["gen_ref"] - f["flow_ref"] / 2.0
        row = {"kappa": c["kappa"], "B": Bk, "own_share": (2 * ownk / Bk) if Bk else NAN, "flow_over_clock": (f["flow_ref"] / f["clock_ref"]) if f["clock_ref"] else NAN,
               "gen_over_clock": (f["gen_ref"] / f["clock_ref"]) if f["clock_ref"] else NAN}
        perch.append(row)
        rows.append([c["kappa"], fmt(Bk), fmt(row["own_share"], "%.3e"), fmt(row["flow_over_clock"], "%.4f"), fmt(row["gen_over_clock"], "%.4f")])
    out["per_channel_largest_cut"] = perch
    md += ["Per channel in %s (unweighted; the same fit):" % big, ""]
    table(md, ["kappa", "B", "own share", "flow/clock", "gen/clock"], rows)

    # the intercepts: Tr[flat V_c] (expected 0 by time reversal)
    inter = {}
    rows = []
    for ck in V.cuts:
        yC, _ = weighted_sum(V, decs, "TrCV", ck, "all")
        yR, _ = weighted_sum(V, decs, "TrR0V", ck, "all")
        aC, bC = intercept(s_grid, yC)
        aR, bR = intercept(s_grid, yR)
        inter[ck] = {"TrCV_intercept": aC, "TrCV_slope": bC, "TrR0V_intercept": aR, "TrR0V_slope": bR, "intercept_over_B": (aR / abs(bR)) if bR else NAN,
                     "C_intercept_over_B": (aC / abs(bC)) if bC else NAN}
        rows.append([fmt(float(ck) / V.rs, "%g"), fmt(aR), fmt(bR), fmt(aC), fmt(bC), fmt(inter[ck]["intercept_over_B"], "%.2e"), fmt(inter[ck]["C_intercept_over_B"], "%.2e")])
    out["flat_flow_intercepts"] = inter
    md += ["The s to 0 intercepts of Tr[R0(s) V_c] = W_R0/sdot and Tr[C V_c] = W_C/sdot (the first 3 readouts; the intercept is Tr[flat V_c], expected 0 by time reversal; its size is the systematic error of the flow parts relative to the flat sea; the slope is -B_c):", ""]
    table(md, ["cut R/r_s", "R0 intercept", "R0 slope", "C intercept", "C slope", "R0 intercept / |slope|", "C intercept / |slope|"], rows)

    # the adiabaticity at smax and the clock energies in/out of r_s
    EinNO, _ = weighted_top(V, "E_in_NO", "all")
    Einraw, _ = weighted_top(V, "E_in_raw", "all")
    EoutNO, _ = weighted_top(V, "E_out_NO", "all")
    Eoutraw, _ = weighted_top(V, "E_out_raw", "all")
    i_s = int(idx_s[-1]) if len(idx_s) else 0
    adiab = {"i": i_s, "s": float(s_grid[i_s]), "E_in_NO": float(EinNO[i_s]), "E_in_raw": float(Einraw[i_s]), "E_out_NO": float(EoutNO[i_s]), "E_out_raw": float(Eoutraw[i_s])}
    adiab["in_NO_over_raw"] = adiab["E_in_NO"] / adiab["E_in_raw"] if adiab["E_in_raw"] else NAN
    adiab["out_NO_over_raw"] = adiab["E_out_NO"] / adiab["E_out_raw"] if adiab["E_out_raw"] else NAN
    out["adiabaticity_at_smax"] = adiab
    md += ["- adiabaticity at s = %.4f (readout %d): E_in_NO/E_in_raw = %s, E_out_NO/E_out_raw = %s (the excitation's share of the clock response; E_out_NO %s, E_out_raw %s)" % (
        adiab["s"], i_s, fmt(adiab["in_NO_over_raw"], "%.3e"), fmt(adiab["out_NO_over_raw"], "%.3e"), fmt(adiab["E_out_NO"]), fmt(adiab["E_out_raw"])), ""]

    # the exterior at the hold's end
    rows = []
    ext = []
    sums = {"E_out_raw_minus_NO": 0.0, "E_out_R2_minus_NO": 0.0, "E_in_raw_minus_NO": 0.0, "E_in_R2_minus_NO": 0.0, "E_out_NO": 0.0, "E_out_R2": 0.0, "N_p": 0.0, "N_h": 0.0}
    for c in V.ch:
        q = c["reads"][-1]
        pk = c.get("continuation", {}).get("pockets_max", None)
        if pk is None:
            pk = max([(r.get("continuation") or {}).get("n_pocket", 0) or 0 for r in c["reads"]] + [0])
        row = {"kappa": c["kappa"], "t_over_rs": q["t_over_rs"], "complete": V.complete(c), "E_in_NO": q.get("E_in_NO"), "E_in_R2": q.get("E_in_R2"), "E_out_NO": q.get("E_out_NO"),
               "E_out_R2": q.get("E_out_R2"), "E_out_raw": q.get("E_out_raw"), "N_p": q.get("N_p"), "N_h": q.get("N_h"), "pockets_max": pk}
        row["E_out_raw_minus_NO"] = (q.get("E_out_raw", NAN) - q.get("E_out_NO", NAN))
        row["E_out_R2_minus_NO"] = (q.get("E_out_R2", NAN) - q.get("E_out_NO", NAN))
        ext.append(row)
        if V.complete(c):
            w = V.w(c)
            sums["E_out_raw_minus_NO"] += w * row["E_out_raw_minus_NO"]
            sums["E_out_R2_minus_NO"] += w * row["E_out_R2_minus_NO"]
            sums["E_in_raw_minus_NO"] += w * (q.get("E_in_raw", NAN) - q.get("E_in_NO", NAN))
            sums["E_in_R2_minus_NO"] += w * (q.get("E_in_R2", NAN) - q.get("E_in_NO", NAN))
            sums["E_out_NO"] += w * q.get("E_out_NO", NAN)
            sums["E_out_R2"] += w * q.get("E_out_R2", NAN)
            sums["N_p"] += w * q.get("N_p", NAN)
            sums["N_h"] += w * q.get("N_h", NAN)
        rows.append([c["kappa"], fmt(row["t_over_rs"], "%.1f"), fmt(row["complete"]), fmt(row["E_in_NO"]), fmt(row["E_in_R2"]), fmt(row["E_out_NO"]), fmt(row["E_out_R2"]),
                     fmt(row["E_out_raw_minus_NO"]), fmt(row["E_out_R2_minus_NO"]), fmt(row["N_p"], "%.2e"), fmt(row["N_h"], "%.2e"), pk])
    out["exterior_last_readout"] = {"per_channel": ext, "weighted_sums_complete": sums}
    md += ["### (4) The last readout per channel: the clock energies in and out of r_s under NO and R2 (R >= r_s), the exterior static clock response E_out_raw - E_out_NO (R0 against flat at s = 1), the exterior difference of the two references E_out_R2 - E_out_NO, the pair counts", ""]
    table(md, ["kappa", "t/r_s", "complete", "E_in_NO", "E_in_R2", "E_out_NO", "E_out_R2", "E_out raw - NO", "E_out R2 - NO", "N_p", "N_h", "pockets max"], rows)
    md += ["- weighted sums over the complete channels: E_out raw - NO (exterior static clock response at s = 1, %s current) %s; E_out R2 - NO %s; E_in raw - NO %s; E_in R2 - NO %s; E_out_NO %s; E_out_R2 %s; N_p %s; N_h %s" % (
        V.current, fmt(sums["E_out_raw_minus_NO"]), fmt(sums["E_out_R2_minus_NO"]), fmt(sums["E_in_raw_minus_NO"]), fmt(sums["E_in_R2_minus_NO"]), fmt(sums["E_out_NO"]), fmt(sums["E_out_R2"]),
        fmt(sums["N_p"], "%.3e"), fmt(sums["N_h"], "%.3e")), ""]
    RES[V.name] = out
    return out


def compare_vf(Va, Vb, a, RES, md):
    """the reference quantities (E_K_R0, W_R0) and the propagated state's (E_K_C, W_C, E_in_raw, E_in_NO), both independent of the
    continuation convention, over the common readouts of the channels present in both records."""
    kb = {c["kappa"]: c for c in Vb.ch}
    rows = []
    worst = NAN
    out = {"a": Va.name, "b": Vb.name, "per_channel": []}
    for c in Va.ch:
        if c["kappa"] not in kb:
            continue
        d = kb[c["kappa"]]
        n = min(len(c["reads"]), len(d["reads"]))
        ta = np.array([q["t_over_rs"] for q in c["reads"][:n]], float)
        tb = np.array([q["t_over_rs"] for q in d["reads"][:n]], float)
        row = {"kappa": c["kappa"], "n_common": n}
        if n == 0 or np.max(np.abs(ta - tb)) > 1e-9:
            row["note"] = "the readout grids differ"
            out["per_channel"].append(row)
            rows.append([c["kappa"], n, "grids differ", "", "", "", "", ""])
            continue
        vals = []
        for key in ("E_K_R0", "W_R0", "E_K_C", "W_C"):
            dev = []
            for ck in Va.cuts:
                if ck not in Vb.cuts:
                    continue
                x = np.array([q["cuts"][ck].get(key, NAN) for q in c["reads"][:n]], float)
                y = np.array([q["cuts"][ck].get(key, NAN) for q in d["reads"][:n]], float)
                scl = nmax(np.abs(x))
                dev.append(nmax(np.abs(x - y)) / scl if scl else NAN)
            row[key] = nmax(dev)
            vals.append(row[key])
        for key in ("E_in_raw", "E_in_NO"):
            x = np.array([q.get(key, NAN) for q in c["reads"][:n]], float)
            y = np.array([q.get(key, NAN) for q in d["reads"][:n]], float)
            scl = nmax(np.abs(x))
            row[key] = nmax(np.abs(x - y)) / scl if scl else NAN
            vals.append(row[key])
        w_ = nmax(vals)
        worst = w_ if not np.isfinite(worst) else max(worst, w_)
        out["per_channel"].append(row)
        rows.append([c["kappa"], n] + [fmt(row.get(k), "%.2e") for k in ("E_K_R0", "W_R0", "E_K_C", "W_C", "E_in_raw", "E_in_NO")])
    out["worst_rel"] = worst
    md += ["## %s against %s over the common readouts (relative maxima; the continuation convention is the only difference between the records, so all of these should be rounding)" % (Va.name, Vb.name), ""]
    table(md, ["kappa", "n common", "E_K_R0", "W_R0", "E_K_C", "W_C", "E_in_raw", "E_in_NO"], rows)
    verdict(RES, "T6a %s vs %s: worst relative difference of E_K_R0, W_R0, E_K_C, W_C, E_in_raw, E_in_NO" % (Va.name, Vb.name), worst, "< %g" % RULES["record_consistency_rel"],
            (worst < RULES["record_consistency_rel"]) if np.isfinite(worst) else None)
    RES["%s_vs_%s" % (Va.name, Vb.name)] = out
    return out


# ------------------------------------------------------------------------------------------------------------------------ vf3/vf7 records
def find_total_key(tot, sub, which="hold"):
    """the per-site totals key of the part sub ('raw', 'NO', 'R2') at the time which ('hold': the hold's end; 'end': the ramp's end)."""
    sub_l, which_l = sub.lower(), which.lower()

    def toks(k):
        return [tok.lower() for tok in k.replace("-", "_").split("_")]

    exact = [k for k in tot if which_l in toks(k) and sub_l in toks(k)]
    if exact:
        return exact[0]
    loose = [k for k in tot if which_l in k.lower() and sub_l in k.lower()]
    return loose[0] if loose else None


class OLD:
    """an engine_vacflow_v3/v7 record: running channel sums at the probes, per-site totals at the ramp's end and at the hold's end."""

    def __init__(self, D, name, path=""):
        self.D, self.name, self.path = D, name, path
        self.rs = float(D["args"]["rs"])
        self.kdone = int(D.get("kappa_done", 0) or 0)
        self.trs = sorted(D.get("running", {}).keys(), key=lambda k: float(k))
        self.current = str(D["args"].get("chain_current", "not in args (engine v3/v7: the two-step current)"))
        self.probes = list(PROBES)
        self.keys_seen = sorted({k for t in self.trs for x in D["running"][t] for k in x.keys()})
        if self.trs and D["running"][self.trs[0]]:
            n_probe = len(D["running"][self.trs[0]][0].get("E_raw_at", []))
            if n_probe != len(PROBES):
                self.probes = list(D["args"].get("probes", [NAN] * n_probe))[:n_probe]

    def run_last(self, t, kmax):
        xs = [x for x in self.D["running"][t] if x["kappa"] <= kmax]
        return xs[-1] if xs else None

    def cumul(self, t, kmax, sub):
        r = self.run_last(t, kmax)
        key = SUBKEY[sub]
        if r is None or key not in r:
            return None, None
        return np.array(r[key], float), int(r["kappa"])

    def annuli(self, t, kmax, sub):
        E, k = self.cumul(t, kmax, sub)
        return (np.diff(E), k) if E is not None else (None, None)

    def per_channel(self, t, sub):
        key = SUBKEY[sub]
        out = {}
        prev = None
        for x in self.D["running"][t]:
            if key not in x:
                return None
            E = np.array(x[key], float)
            d = E - prev if prev is not None else E.copy()
            out[int(x["kappa"])] = np.diff(d)
            prev = E
        return out

    def total_at(self, t, sub, which="hold"):
        tot = self.D.get("totals", {}).get(t, {})
        key = find_total_key(tot, sub, which) if isinstance(tot, dict) else None
        if key is None:
            return None, sorted(tot.keys()) if isinstance(tot, dict) else []
        return np.array(tot[key], float), key

    def total_hold(self, t, sub):
        return self.total_at(t, sub, "hold")

    def nsites(self):
        for t in self.trs:
            arr, _ = self.total_at(t, "raw", "hold")
            if arr is not None:
                return int(len(arr))
        return int(64 * self.rs)

    def per_channel_info(self):
        info = {}
        for e in self.D.get("per_channel", []) or []:
            try:
                info[int(e["kappa"])] = e
            except Exception:
                pass
        return info

    def weights(self, kappas):
        info = self.per_channel_info()
        return [float(info[k].get("weight", 16.0 * k)) if k in info else 16.0 * k for k in kappas]

    def pocket_flags(self, kappas):
        """the record's own pocket flag per channel: per_channel[k]['pocket'] if present, else the 'pockets' list ([kappa, n])."""
        info = self.per_channel_info()
        flags = {}
        src = "per_channel.pocket"
        for k in kappas:
            if k in info and "pocket" in info[k]:
                v = info[k]["pocket"]
                flags[k] = int(len(v) > 0) if isinstance(v, (list, tuple, dict)) else int(bool(v))
        if not flags:
            src = "pockets list"
            for e in self.D.get("pockets", []) or []:
                try:
                    k = int(e[0])
                    v = e[1]
                    flags[k] = int(v > 0) if isinstance(v, (int, float)) else int(bool(v))
                except Exception:
                    pass
        for k in kappas:
            flags.setdefault(k, 0)
        return flags, src

    def r2_rejected(self, kappas):
        info = self.per_channel_info()
        rej = []
        for k in kappas:
            r2 = (info.get(k) or {}).get("r2")
            if isinstance(r2, dict) and r2.get("rejected"):
                rej.append(k)
        return rej


def cum_below(arr, N):
    """the sum of a per-site array over the indices n < N (N clipped to the array)."""
    N = int(max(0, min(len(arr), N)))
    return float(np.nansum(arr[:N])) if N > 0 else 0.0


def detect_probe_offset(arr, rs, probes, running_annuli, offsets=OFFSETS):
    """the engine's probe convention: the cumulative energy below the probe p is read as the sum over the indices n < round(p r_s) - 1 + d;
    d = 0 is 'R = n + 1 and R < p r_s' (the convention of the vf10 cuts, where 'site' is the count R < cut), d = +1 is 'R <= p r_s'.
    Returns the offset whose probe annuli reproduce the running annuli best and the max abs discrepancy per offset."""
    table_ = {}
    for d in offsets:
        cums = [cum_below(arr, int(round(p * rs)) - 1 + d) for p in probes]
        table_[int(d)] = float(np.max(np.abs(np.diff(cums) - np.asarray(running_annuli, float))))
    best = min(table_, key=lambda d: table_[d])
    return best, table_


def binned(arr, rs, edges, offset=0):
    """the sums of a per-site array over the bins with indices round(lo r_s) - 1 + offset <= n < round(hi r_s) - 1 + offset (offset 0:
    R = n + 1 in [lo r_s, hi r_s); offset +1: R in (lo r_s, hi r_s], the engine's probe convention if detected so); bins reaching past
    the chain are left out."""
    idx = np.arange(len(arr))
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        i_lo = max(0, int(round(lo * rs)) - 1 + offset)
        i_hi = int(round(hi * rs)) - 1 + offset
        if i_hi > len(arr):
            break
        m = (idx >= i_lo) & (idx < i_hi)
        rows.append({"lo": lo, "hi": hi, "R_geo": math.sqrt(lo * hi) * rs, "E": float(np.nansum(arr[m])), "n_sites": int(m.sum()), "ln_ratio": math.log(hi / lo)})
    return rows


def analyze_old(O, a, RES, md):
    kmax = min(a.kmax, O.kdone) if O.kdone else a.kmax
    slope = slope_full_zone(a.BL, O.rs)
    out = {"file": O.path, "rs": O.rs, "kappa_done": O.kdone, "kmax_used": kmax, "ramps_t_r_over_rs": O.trs, "chain_current": O.current, "probes_over_rs": O.probes,
           "running_keys": O.keys_seen, "slope_full_zone_per_efold": slope}
    md += ["## %s: %s" % (O.name, O.path), "", "- r_s %g, channels done %d (used kappa <= %d), ramps t_r/r_s %s, chain current %s; the running entries carry the keys %s" % (
        O.rs, O.kdone, kmax, O.trs, O.current, O.keys_seen), "- the full-zone flat slope (9 pi/2) B_L r_s = %.4f per e-fold at this r_s" % slope, ""]
    if not O.trs:
        md += ["- no running sums in the record", ""]
        RES[O.name] = out
        return out
    lo_p = np.array(O.probes[:-1], float) * O.rs
    hi_p = np.array(O.probes[1:], float) * O.rs
    ann_lab = ["[%g, %g] r_s" % (lo, hi) for lo, hi in zip(O.probes[:-1], O.probes[1:])]
    lnr = np.log(hi_p / lo_p)
    Rgeo = np.sqrt(lo_p * hi_p)
    ann = {}
    rows = []
    for t in O.trs:
        raw, k = O.annuli(t, kmax, "raw")
        no, _ = O.annuli(t, kmax, "NO")
        r2_raw, _ = O.annuli(t, kmax, "R2")
        r2_state = "absent" if r2_raw is None else ("nan" if not np.isfinite(r2_raw).any() else "stored")
        r2 = r2_raw if r2_state == "stored" else None
        st = (raw - no) if (raw is not None and no is not None) else None
        st2 = (raw - r2) if (raw is not None and r2 is not None) else None
        ann[t] = {"raw": raw, "NO": no, "R2": r2, "R2_state": r2_state, "static": st, "static_R2": st2, "kappa_summed": k}
        for j, lab in enumerate(ann_lab):
            rows.append([t, lab, fmt(raw[j]) if raw is not None else "-", fmt(no[j]) if no is not None else "-",
                         fmt(r2[j]) if r2 is not None else ("not in record" if r2_state == "absent" else "nan in record"),
                         fmt(st[j]) if st is not None else "-", fmt(st2[j]) if st2 is not None else "-",
                         fmt(st[j] / lnr[j]) if st is not None else "-", fmt(st[j] / lnr[j] / slope, "%.3e") if st is not None else "-",
                         fmt((kmax / Rgeo[j]) ** 2, "%.3e"), fmt(no[j] / lnr[j] / slope, "%.3e") if no is not None else "-",
                         fmt(r2[j] / lnr[j] / slope, "%.3e") if r2 is not None else "-"])
    out["annuli"] = {t: {k_: (v.tolist() if isinstance(v, np.ndarray) else v) for k_, v in d.items()} for t, d in ann.items()}
    md += ["### (a) The annuli from the running channel sums (kappa <= %d) after the hold, in the engine's own probe convention: raw (against the flat sea), NO (against R0), R2 (if stored and finite), the removed response raw - NO = Tr[(R0 - flat) hcal] and raw - R2 (the sea's clock response at s = 1, no dynamics in it); per e-fold; against the full-zone flat slope; the resolved-channel fraction (kmax/R_geo)^2 for orientation; the NO and R2 energies per e-fold against the slope (the literal threshold 1e-5, a size)" % kmax, ""]
    table(md, ["t_r/r_s", "annulus", "raw", "NO", "R2", "raw - NO", "raw - R2", "(raw - NO) per e-fold", "(raw - NO) per e-fold / slope", "(kmax/R_geo)^2", "NO per e-fold / slope", "R2 per e-fold / slope"], rows)
    for key, label in (("static", "raw - NO"), ("static_R2", "raw - R2")):
        S = np.array([ann[t][key] for t in O.trs if ann[t][key] is not None])
        if len(S) >= 2:
            mean = S.mean(axis=0)
            dev = np.max(np.abs(S - mean), axis=0)
            spread = dev / np.maximum(np.abs(mean), 1e-300)
            arith = dev / lnr / slope
            out["%s_ramp_spread_rel" % key] = spread.tolist()
            out["%s_arith_residual_per_efold_over_slope" % key] = arith.tolist()
            md += ["- the removed response %s: spread across the %d ramps (max |deviation| / |mean|): %s; the same deviation per e-fold over the slope (a reading check of the records, the identity raw - NO = Tr[(R0(1) - flat) hcal], not the annulus-residual control): %s" % (
                label, len(S), ", ".join("%s %.2e" % (lab, sp) for lab, sp in zip(ann_lab, spread)), ", ".join("%s %.1e" % (lab, x) for lab, x in zip(ann_lab, arith))), ""]
            verdict(RES, "%s T3a%s the removed response %s identical across the %d ramps (max relative spread over the annuli)" % (O.name, "" if key == "static" else "-R2", label, len(S)),
                    float(np.max(spread)), "< %g" % RULES["static_spread_rel"], bool(np.max(spread) < RULES["static_spread_rel"]))
            verdict(RES, "%s T3a per e-fold: the ramp-to-ramp deviation of the removed response %s over the full-zone slope (a reading check of the identity, not the annulus-residual control, which is T5a/T7c)" % (O.name, label),
                    float(np.max(arith)), "< %g" % RULES["arith_residual_over_slope"], bool(np.max(arith) < RULES["arith_residual_over_slope"]),
                    "rounding of an identity; the energy left under NO or R2 is the remainder, a size (T5a, T5b, T7c)")
    fits = {}
    rows = []
    for t in O.trs:
        st = ann[t]["static"]
        if st is None:
            continue
        f_far = fit_powerlaw(lo_p[1:], hi_p[1:], st[1:])
        f_all = fit_powerlaw(lo_p, hi_p, st)
        fits[t] = {"far_probe_annuli": f_far, "all_probe_annuli": f_all, "ratios_per_doubling": [float(st[j] / st[j + 1]) if st[j + 1] else NAN for j in range(len(st) - 1)]}
        rows.append([t, fmt(f_far["alpha"], "%.3f"), fmt(f_far["err"], "%.3f"), fmt(f_far["ratio_per_doubling"], "%.2f"), fmt(f_all["alpha"], "%.3f"), fmt(f_all["err"], "%.3f"),
                     ", ".join(fmt(x, "%.3f") for x in fits[t]["ratios_per_doubling"])])
    out["static_fits_probes"] = fits
    md += ["### (b) The falloff of the removed response raw - NO: alpha in E_annulus ~ R^alpha (the channel-limited prediction -2, 4 per doubling, the density r^-5; the logarithm 0); the far fit uses %s" % ", ".join(ann_lab[1:]), ""]
    table(md, ["t_r/r_s", "alpha far", "err", "ratio per doubling", "alpha all 4", "err", "successive ratios"], rows)
    if fits:
        tl = O.trs[-1]
        f = fits[tl]["far_probe_annuli"]
        lo_w, hi_w = RULES["alpha_window"]
        verdict(RES, "%s T3b alpha of the static probe annuli (far, t_r %s r_s)" % (O.name, tl), [f["alpha"], f["err"]], "within [%g, %g]" % (lo_w, hi_w), bool(np.isfinite(f["alpha"]) and lo_w <= f["alpha"] <= hi_w),
                "ratio per doubling %s" % fmt(f["ratio_per_doubling"], "%.2f"))
    for sub in ("NO", "R2"):
        rows = []
        for t in O.trs:
            arr = ann[t][sub]
            if arr is None:
                continue
            rel = arr / lnr / slope
            for j, lab in enumerate(ann_lab):
                if j >= 1:
                    st = ann[t]["static"]
                    verdict(RES, "%s T5a %s energy per e-fold over the full-zone slope, t_r %s r_s, %s" % (O.name, sub, t, lab), float(rel[j]), "< %g (literal threshold)" % RULES["annulus_threshold_over_slope"],
                            bool(abs(rel[j]) < RULES["annulus_threshold_over_slope"]),
                            "%s / (raw - NO) removed in the same annulus (the retained-space normalization) = %s" % (sub, fmt(arr[j] / st[j] if (st is not None and st[j]) else NAN, "%.3e")), kind="size")
            rows.append([t] + [fmt(x, "%.3e") for x in rel])
        if rows:
            md += ["The %s energy per e-fold over the full-zone slope per annulus (the physical state's excess over the %s reference, the switch-on's excitation; for the pocket-free channels NO equals R2):" % (
                sub, "instantaneous" if sub == "NO" else "continued"), ""]
            table(md, ["t_r/r_s"] + ann_lab, rows)
    tl = O.trs[-1]
    pr = O.per_channel(tl, "raw")
    pn = O.per_channel(tl, "NO")
    if pr is not None and pn is not None:
        rows = []
        perch = {}
        for kappa in sorted(pr.keys()):
            if kappa > kmax:
                continue
            st = pr[kappa] - pn[kappa]
            perch[kappa] = st.tolist()
            rows.append([kappa] + [fmt(x) for x in st] + [", ".join(fmt(st[j] / st[j + 1] if st[j + 1] else NAN, "%.2f") for j in range(len(st) - 1))] +
                        [", ".join(fmt(x / (16.0 * kappa), "%.3e") for x in st)])
        out["per_channel_static_longest_ramp"] = perch
        md += ["### (c) Per channel (weights included): the static annuli at t_r = %s r_s, their successive ratios, and the values per unit weight (the channel-limited reading needs them nearly equal at large R)" % tl, ""]
        table(md, ["kappa"] + ann_lab + ["ratios", "per unit weight"], rows)
    tot_out = {}
    for t in O.trs:
        raw_arr, raw_key = O.total_hold(t, "raw")
        no_arr, no_key = O.total_hold(t, "NO")
        r2_arr, r2_key = O.total_hold(t, "R2")
        r2_ok = r2_arr is not None and bool(np.isfinite(r2_arr).any())
        info = {"keys": {"raw": raw_key if raw_arr is not None else None, "NO": no_key if no_arr is not None else None, "R2": r2_key if r2_arr is not None else None},
                "R2_totals_state": "absent" if r2_arr is None else ("stored" if r2_ok else "nan")}
        if raw_arr is None or no_arr is None:
            info["note"] = "per-site totals not found; the totals' keys at this ramp: %s" % (raw_key if raw_arr is None else no_key)
            tot_out[t] = info
            md += ["### (d) The per-site totals at t_r = %s r_s: %s" % (t, info["note"]), ""]
            continue
        n = min(len(raw_arr), len(no_arr))
        st = raw_arr[:n] - no_arr[:n]
        info["n_sites"] = n
        d_best, d_table = 0, {}
        if ann[t]["static"] is not None:
            d_best, d_table = detect_probe_offset(st, O.rs, O.probes, ann[t]["static"])
        info["probe_offset_detected"] = int(d_best)
        info["running_vs_totals_by_offset"] = {str(d): v for d, v in d_table.items()}
        info["running_vs_totals_static_max_abs"] = float(d_table.get(d_best, NAN))
        info["running_vs_totals_static_max_abs_offset0"] = float(d_table.get(0, NAN))
        b_st = binned(st, O.rs, BIN_EDGES, d_best)
        b_no = binned(no_arr[:n], O.rs, BIN_EDGES, d_best)
        b_raw = binned(raw_arr[:n], O.rs, BIN_EDGES, d_best)
        b_r2 = binned(r2_arr[:n], O.rs, BIN_EDGES, d_best) if r2_ok else None
        b_st0 = binned(st, O.rs, BIN_EDGES, 0) if d_best != 0 else b_st
        info["fits"], info["fits_offset0"] = {}, {}
        for lab, rlo, rhi in FIT_RANGES:
            for key, bins_ in (("fits", b_st), ("fits_offset0", b_st0)):
                sel = [b for b in bins_ if b["lo"] >= rlo - 1e-9 and b["hi"] <= rhi + 1e-9]
                info[key][lab] = fit_powerlaw([b["lo"] * O.rs for b in sel], [b["hi"] * O.rs for b in sel], [b["E"] for b in sel]) if len(sel) >= 2 else \
                    {"alpha": NAN, "err": NAN, "n": len(sel), "ratio_per_doubling": NAN, "n_dropped": 0}

        def below_from(bins_):
            rel = [b["E"] / b["ln_ratio"] / slope for b in bins_]
            r_from = None
            for j in range(len(rel) - 1, -1, -1):
                if abs(rel[j]) < RULES["annulus_threshold_over_slope"]:
                    r_from = bins_[j]["lo"]
                else:
                    break
            return rel, r_from

        no_rel, no_from = below_from(b_no)
        if b_r2:
            r2_rel, r2_from = below_from(b_r2)
        else:
            r2_rel, r2_from = [None] * len(b_no), None
        info["NO_per_efold_over_slope_by_bin"] = no_rel
        info["NO_below_threshold_from_R_over_rs"] = no_from
        info["R2_per_efold_over_slope_by_bin"] = r2_rel if b_r2 else None
        info["R2_below_threshold_from_R_over_rs"] = r2_from
        ext = {}
        for d in sorted({0, int(d_best)}):
            i_ext = max(0, int(round(O.rs)) - 1 + d)
            ext[str(d)] = {"static": float(np.nansum(st[i_ext:])), "NO": float(np.nansum(no_arr[:n][i_ext:])), "raw": float(np.nansum(raw_arr[:n][i_ext:])),
                           "R2": float(np.nansum(r2_arr[:n][i_ext:])) if r2_ok else None, "sites_from_R": i_ext + 1}
        info["exterior_by_offset"] = ext
        info["bins"] = [{"lo": b["lo"], "hi": b["hi"], "n_sites": b["n_sites"], "static": b["E"], "NO": bn["E"], "raw": br["E"], "R2": (b_r2[j]["E"] if b_r2 else None),
                         "static_per_efold_over_slope": b["E"] / b["ln_ratio"] / slope, "NO_per_efold_over_slope": no_rel[j], "R2_per_efold_over_slope": (r2_rel[j] if b_r2 else None),
                         "NO_minus_R2": ((bn["E"] - b_r2[j]["E"]) if b_r2 else None), "NO_over_retained": (bn["E"] / b["E"] if b["E"] else NAN),
                         "R2_over_retained": ((b_r2[j]["E"] / b["E"]) if (b_r2 and b["E"]) else None)} for j, (b, bn, br) in enumerate(zip(b_st, b_no, b_raw))]
        tot_out[t] = info
        md += ["### (d) The per-site totals at t_r = %s r_s (keys %s; %d sites): the engine's probe convention detected as offset %+d site(s) relative to 'R = n + 1, R < p r_s' (the running annuli agree with the totals' sums to %s under it; at offset 0 to %s; by offset %s); the bins below use the detected convention; the last two columns are the remainder over the retained-space flat signal raw - NO of the same bin (B1 test (b)'s normalization)" % (
            t, info["keys"], n, d_best, fmt(info["running_vs_totals_static_max_abs"], "%.2e"), fmt(info["running_vs_totals_static_max_abs_offset0"], "%.2e"),
            ", ".join("%+d: %s" % (d, fmt(v, "%.1e")) for d, v in sorted(d_table.items()))), ""]
        rows = [[fmt(b["lo"], "%g"), fmt(b["hi"], "%g"), b["n_sites"], fmt(b["static"]), fmt(b["static_per_efold_over_slope"], "%.3e"), fmt(b["NO"]), fmt(b["NO_per_efold_over_slope"], "%.3e"), fmt(b["raw"]),
                 fmt(b["R2"]) if b["R2"] is not None else info["R2_totals_state"], fmt(b["R2_per_efold_over_slope"], "%.3e") if b["R2"] is not None else "-",
                 fmt(b["NO_minus_R2"]) if b["R2"] is not None else "-", fmt(b["NO_over_retained"], "%.3e"), fmt(b["R2_over_retained"], "%.3e") if b["R2"] is not None else "-"] for b in info["bins"]]
        table(md, ["R_lo/r_s", "R_hi/r_s", "sites", "raw - NO", "(raw - NO) per e-fold / slope", "NO", "NO per e-fold / slope", "raw", "R2", "R2 per e-fold / slope", "NO - R2", "NO / (raw - NO)", "R2 / (raw - NO)"], rows)
        rows = [[lab, f["n"], fmt(f["alpha"], "%.3f"), fmt(f["err"], "%.3f"), fmt(f["ratio_per_doubling"], "%.2f"), f.get("n_dropped", 0),
                 fmt(info["fits_offset0"][lab]["alpha"], "%.3f"), fmt(abs(info["fits_offset0"][lab]["alpha"] - f["alpha"]), "%.3f")] for lab, f in info["fits"].items()]
        table(md, ["fit range", "bins", "alpha (detected convention)", "err", "ratio per doubling", "bins dropped (E <= 0)", "alpha (offset 0)", "|difference|"], rows)
        e_d = ext[str(int(d_best))]
        md += ["- exterior sums of the totals from the site R = %d on (R > r_s in the detected convention; vf10's E_out is R >= r_s): raw - NO %s, NO %s, raw %s, R2 %s; the NO energy per e-fold stays below %g of the slope from R/r_s = %s outward and the R2 energy from %s (none if '-')" % (
            e_d["sites_from_R"], fmt(e_d["static"]), fmt(e_d["NO"]), fmt(e_d["raw"]), fmt(e_d["R2"]) if e_d["R2"] is not None else info["R2_totals_state"], RULES["annulus_threshold_over_slope"], fmt(no_from, "%g"), fmt(r2_from, "%g") if b_r2 else "not stored"), ""]
        verdict(RES, "%s T6c running annuli against the totals' sums under the detected probe convention (t_r %s r_s, max abs)" % (O.name, t), info["running_vs_totals_static_max_abs"],
                "< %g" % RULES["running_vs_totals_abs"], (info["running_vs_totals_static_max_abs"] < RULES["running_vs_totals_abs"]) if np.isfinite(info["running_vs_totals_static_max_abs"]) else None,
                "detected offset %+d site(s) relative to 'R = n + 1, R < p r_s'; at offset 0 the discrepancy is %s; a pass means the totals are the running sums' channel set" % (d_best, fmt(info["running_vs_totals_static_max_abs_offset0"], "%.2e")))
        if t == O.trs[-1]:
            f = info["fits"]["far [4, 32] r_s"]
            f0 = info["fits_offset0"]["far [4, 32] r_s"]
            lo_w, hi_w = RULES["alpha_window"]
            syst_range = nmax([abs(info["fits"][l]["alpha"] - f["alpha"]) for l in ("mid [2, 32] r_s", "all [2, 60] r_s") if np.isfinite(info["fits"][l]["alpha"])])
            syst_conv = abs(f0["alpha"] - f["alpha"]) if (np.isfinite(f0["alpha"]) and np.isfinite(f["alpha"])) else 0.0
            syst = nmax([syst_range, syst_conv])
            comb = math.sqrt((f["err"] if np.isfinite(f["err"]) else 0.0) ** 2 + (syst if np.isfinite(syst) else 0.0) ** 2)
            verdict(RES, "%s T3b alpha of the static response, binned totals far [4, 32] r_s, t_r %s r_s" % (O.name, t), [f["alpha"], f["err"], syst], "within [%g, %g]" % (lo_w, hi_w),
                    bool(np.isfinite(f["alpha"]) and lo_w <= f["alpha"] <= hi_w), "the systematic is the larger of the change over the ranges [2, 32] and [2, 60] r_s (%s) and the change of the edge convention (%s)" % (fmt(syst_range, "%.3f"), fmt(syst_conv, "%.3f")))
            verdict(RES, "%s T3b the logarithm (alpha 0) absent on the chain: alpha + %g combined errors < 0" % (O.name, RULES["alpha_log_sigmas"]), f["alpha"] + RULES["alpha_log_sigmas"] * comb, "< 0",
                    bool(np.isfinite(f["alpha"]) and f["alpha"] + RULES["alpha_log_sigmas"] * comb < 0))
            verdict(RES, "%s T5b NO energy per e-fold below %g of the slope in every bin from R/r_s" % (O.name, RULES["annulus_threshold_over_slope"]), no_from, "a radius exists within the chain (literal threshold)",
                    no_from is not None, "bins to 60 r_s at t_r %s r_s" % t, kind="size")
            if b_r2:
                verdict(RES, "%s T5b R2 energy per e-fold below %g of the slope in every bin from R/r_s" % (O.name, RULES["annulus_threshold_over_slope"]), r2_from, "a radius exists within the chain (literal threshold)",
                        r2_from is not None, "bins to 60 r_s at t_r %s r_s" % t, kind="size")
    out["totals"] = tot_out
    RES[O.name] = out
    return out


# ------------------------------------------------------------------------------------------------- B1 analysis of a vf3/vf7 record
def kbar_of(values, kappas, mask):
    v = np.asarray(values, float)[mask]
    kk = np.asarray(kappas, float)[mask]
    m = np.isfinite(v)
    tot = float(np.sum(v[m])) if m.any() else 0.0
    return float(np.sum(kk[m] * v[m]) / tot) if tot > 0 else NAN


def analyze_b1_old(O, a, RES, md, cache):
    """B1 (a) and (b) on a vf3/vf7 record (busi/specs/S12_addendum_B1_v1.md sections 1.5, 2.3, 3); needs analyze_old's result in RES."""
    name = O.name
    base = RES.get(name) or {}
    if not O.trs or "annuli" not in base:
        md += ["## B1 on %s: no running sums in the record; skipped" % name, ""]
        return None
    rs = O.rs
    kmax = int(base.get("kmax_used", a.kmax))
    slope = slope_full_zone(a.BL, rs)
    hold = a.hold_over_rs * rs
    kappas = list(range(1, kmax + 1))
    weights = O.weights(kappas)
    flags, flag_src = O.pocket_flags(kappas)
    rejected = O.r2_rejected(kappas)
    probes = list(O.probes)
    n_ann = len(probes) - 1
    ann_lab = ["[%g, %g] r_s" % (probes[j], probes[j + 1]) for j in range(n_ann)]
    lnr = np.log(np.array(probes[1:], float) / np.array(probes[:-1], float))
    nsites = O.nsites()
    bins = [(lo, hi) for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]) if int(round(hi * rs)) <= nsites]
    nb = len(bins)
    sites_in = [int(round(hi * rs)) - int(round(lo * rs)) for lo, hi in bins]
    tr_sorted = sorted(O.trs, key=float)

    def bins_of(lo, hi):
        return [j for j, (bl, bh) in enumerate(bins) if bl >= lo - 1e-9 and bh <= hi + 1e-9]

    free_mask = np.array([flags.get(k, 0) == 0 for k in kappas])
    out = {"rs": rs, "kmax": kmax, "hold_over_rs": a.hold_over_rs, "ramp": a.ramp, "kappas": kappas, "weights": weights, "pocket_flags": {str(k): v for k, v in flags.items()},
           "pocket_flag_source": flag_src, "pocket_channels_record": [k for k in kappas if flags.get(k, 0)], "r2_rejected_channels": rejected, "n_u": a.n_u, "n_r": a.n_r, "bins": bins, "ramps": {}}
    md += ["## B1 on %s: the switch-on's excitation (T7) and the channel-truncated far law (T8)" % name, "",
           "- the records' ramp s(t) = sin^2(pi t/2 t_r), Phi(x) = pi^2 cos(x/2)/(pi^2 - x^2), tau = t_r; the remainder is read at t_h = 0 (the '_end' totals) and t_h = %g r_s (the '_hold' totals and the running sums; the hold is a declared input, not in these records' args); K = %d channels, weights %s; pocket flags from %s: pocket channels %s; R2 continuation rejected in channels %s" % (
               a.hold_over_rs, kmax, "16 kappa" if all(abs(w - 16.0 * k) < 1e-9 for k, w in zip(kappas, weights)) else str(weights), flag_src, out["pocket_channels_record"] or "none", rejected or "none"),
           "- the predictions: the channel-local second-order excitation (addendum 1.4: exact M = kappa/r, the K truncation and the discreteness of kappa included; the clock part at the two readout times with the generator and flow parts beside it) and the 3D valley continuum (addendum 1.3); the static law (3K(K+1)/(4 pi)) r_s (R1^-2 - R2^-2) and its exact-M form (addendum 2.2, 2.3); the integrals over a bin run from R_lo + 1/2 to R_hi + 1/2 (the site convention R in (lo r_s, hi r_s]); every remainder per e-fold over the full-zone slope %.4f and over the retained-space flat signal raw - NO of the same bin and channel set" % slope, ""]

    # T7a: the constants and the 3D continuum
    Ig_s, Ig_ss = I_part("gen", 0.0, "sine"), I_part("gen", 0.0, "smoothstep")
    den = 15.0 * math.pi ** 2 * a.BL
    const = {"I_gen_sine": Ig_s, "I_gen_smoothstep": Ig_ss, "B_pair_over_BL_coeff_sine": Ig_s / den, "B_pair_over_BL_coeff_smoothstep": Ig_ss / den,
             "B_pair_over_BL_coeff_smoothstep_closed_form": 3.0 / (5.0 * math.pi ** 2 * a.BL)}
    out["constants"] = const
    cont = {}
    rows = []
    for t in tr_sorted:
        c = continuum(a.BL, float(t) * rs, hold, a.ramp)
        cont[t] = c
        rows.append([t, fmt(c["tau"], "%g"), fmt(c["theta"], "%.4f"), fmt(c["I_gen"], "%.4f"), fmt(c["J"], "%.4f"), fmt(c["I_flow"], "%.4f"), fmt(c["I_clock"], "%.4f"),
                     fmt(c["gen_over_slope"], "%.3e"), fmt(c["flow_over_slope"], "%.3e"), fmt(c["clock_over_slope"], "%.3e")])
    out["continuum"] = cont
    md += ["### T7a the ramp's filter and the 3D valley-continuum prediction per e-fold over the full-zone slope, B_X/B_L = I_X/(15 pi^2 a^2 B_L) (a/tau)^2: I_gen = %.5f for the records' ramp (%.5f for the idea file's smoothstep, closed form 9), so B_pair/B_L = %.4f (a/t_r)^2 (%.4f for the smoothstep; the idea file's 3.1314, closed form %.4f); the clock part at the hold's end carries I_clock = I_gen + 2 J(theta), theta = t_h/tau; at the ramp's end the 3D clock part diverges logarithmically and only the chain's finite K makes it finite (the channel-local prediction below)" % (
        Ig_s, Ig_ss, const["B_pair_over_BL_coeff_sine"], const["B_pair_over_BL_coeff_smoothstep"], const["B_pair_over_BL_coeff_smoothstep_closed_form"]), ""]
    table(md, ["t_r/r_s", "tau/a", "theta = t_h/tau (hold)", "I_gen", "J(theta)", "I_flow", "I_clock", "gen / slope per e-fold", "flow / slope", "clock / slope"], rows)
    verdict(RES, "%s T7a the smooth-ramp pair coefficient in B_pair/B_L = c (a/tau)^2: the records' ramp, the idea file's ramp" % name, [const["B_pair_over_BL_coeff_sine"], const["B_pair_over_BL_coeff_smoothstep"]],
            "derived: 3.2778 (sine), 3.1314 (smoothstep)", None, "I_gen %.5f and %.5f; the prediction is for the generator (pair) part; the records store the clock part, predicted beside it" % (Ig_s, Ig_ss), kind="reported")

    # the law per bin per channel (M -> 0) and the per-ramp loop
    law_bins = np.array([law_annulus(rs, weights, lo * rs, hi * rs) for lo, hi in bins])
    static_pred = None
    ramps = {}
    for t in tr_sorted:
        tau = float(t) * rs
        theta = hold / tau
        pred = predict_bins(rs, tau, hold, kappas, weights, bins, cache, a.n_r, a.n_u, a.ramp)
        if static_pred is None:
            static_pred = pred["static"]
        # observed bins at the hold and at the ramp's end
        tinfo = (base.get("totals") or {}).get(t, {}) if isinstance(base.get("totals"), dict) else {}
        d = int(tinfo.get("probe_offset_detected", 1)) if "probe_offset_detected" in tinfo else 1
        obs = {}
        for which in ("hold", "end"):
            arrs = {}
            for sub in ("raw", "NO", "R2"):
                arr, _ = O.total_at(t, sub, which)
                arrs[sub] = arr if (arr is not None and np.isfinite(np.asarray(arr, float)).any()) else None
            if arrs["raw"] is None or arrs["NO"] is None:
                obs[which] = None
                continue
            n = min(len(arrs["raw"]), len(arrs["NO"]))
            o = {}
            for sub, arr in arrs.items():
                if arr is None:
                    o[sub] = None
                else:
                    bb = {(x["lo"], x["hi"]): x["E"] for x in binned(np.asarray(arr[:n], float), rs, BIN_EDGES, d)}
                    o[sub] = np.array([bb.get(b, NAN) for b in bins])
            o["static"] = o["raw"] - o["NO"]
            obs[which] = o
        # observed per channel at the probes (hold)
        pr, pn, p2 = O.per_channel(t, "raw"), O.per_channel(t, "NO"), O.per_channel(t, "R2")
        ch = None
        if pr is not None and pn is not None and all(k in pr and k in pn for k in kappas):
            ch = {"raw": np.array([pr[k] for k in kappas], float), "NO": np.array([pn[k] for k in kappas], float)}
            ch["static"] = ch["raw"] - ch["NO"]
            if p2 is not None and all(k in p2 for k in kappas):
                r2 = np.array([p2[k] for k in kappas], float)
                ch["R2"] = r2
                ch["R2_finite"] = np.isfinite(r2).all(axis=1)
            else:
                ch["R2"] = None
                ch["R2_finite"] = np.zeros(len(kappas), bool)
            data_pocket = {}
            for i, k in enumerate(kappas):
                if ch["R2_finite"][i]:
                    scale = max(float(np.max(np.abs(ch["NO"][i]))), 1e-300)
                    data_pocket[k] = bool(np.max(np.abs(ch["NO"][i] - ch["R2"][i])) > 1e-9 * scale)
            ch["data_pocket"] = data_pocket
        # predictions at the probe annuli
        pred_ann = {}
        for part, arr in pred.items():
            pa = np.full((len(kappas), n_ann), NAN)
            for j in range(n_ann):
                idx = bins_of(probes[j], probes[j + 1])
                if idx:
                    pa[:, j] = arr[idx].sum(axis=0)
            pred_ann[part] = pa
        law_ann = np.full((len(kappas), n_ann), NAN)
        for j in range(n_ann):
            idx = bins_of(probes[j], probes[j + 1])
            if idx:
                law_ann[:, j] = law_bins[idx].sum(axis=0)
        # channel sums and kbar
        sets = {"all": np.ones(len(kappas), bool), "pocket_free": free_mask}

        def ssum(arr, mask):
            return np.nansum(np.asarray(arr, float)[mask], axis=0) if arr is not None else None

        sums, kb = {}, {}
        for sname, m in sets.items():
            r2m = (m & ch["R2_finite"]) if ch else None
            s_ = {"n_channels": int(m.sum()), "channels": [k for k, mm in zip(kappas, m) if mm],
                  "NO_obs": ssum(ch["NO"], m) if ch else None, "static_obs": ssum(ch["static"], m) if ch else None,
                  "R2_obs": (ssum(ch["R2"], r2m) if (ch and ch["R2"] is not None and r2m.any()) else None),
                  "R2_channels": ([k for k, mm in zip(kappas, r2m) if mm] if ch else []),
                  "NO_obs_on_R2_channels": (ssum(ch["NO"], r2m) if (ch and ch["R2"] is not None and r2m.any()) else None),
                  "pred_clock": ssum(pred_ann["clock"], m), "pred_gen": ssum(pred_ann["gen"], m), "pred_flow": ssum(pred_ann["flow"], m),
                  "pred_clock_end": ssum(pred_ann["clock_end"], m), "pred_flow_end": ssum(pred_ann["flow_end"], m), "pred_static": ssum(pred_ann["static"], m), "law_static": ssum(law_ann, m)}
            sums[sname] = s_
            kb[sname] = {}
            for key, arr in (("NO_obs", ch["NO"] if ch else None), ("static_obs", ch["static"] if ch else None), ("pred_clock", pred_ann["clock"]), ("pred_gen", pred_ann["gen"])):
                kb[sname][key] = [kbar_of(arr[:, j], kappas, m) if arr is not None else NAN for j in range(n_ann)]
        jo = n_ann - 1
        # tables
        if ch is not None:
            rows = []
            for i, k in enumerate(kappas):
                w = weights[i]
                rows.append([k, fmt(w, "%g"), flags.get(k, 0), ("yes" if ch["data_pocket"][k] else "no") if k in ch["data_pocket"] else "-", fmt(ch["NO"][i, jo]), fmt(ch["NO"][i, jo] / w, "%.3e"),
                             fmt(pred_ann["clock"][i, jo]), fmt(pred_ann["gen"][i, jo]), fmt(pred_ann["flow"][i, jo]), fmt(pred_ann["clock_end"][i, jo]),
                             fmt(ch["NO"][i, jo] / pred_ann["clock"][i, jo] if pred_ann["clock"][i, jo] else NAN, "%.3f"), fmt(ch["NO"][i, jo] / pred_ann["gen"][i, jo] if pred_ann["gen"][i, jo] else NAN, "%.3f"),
                             fmt(ch["R2"][i, jo]) if (ch["R2"] is not None and ch["R2_finite"][i]) else "-",
                             fmt(ch["static"][i, jo] / w, "%.4e"), fmt(pred_ann["static"][i, jo] / w, "%.4e"), fmt(law_ann[i, jo] / w, "%.4e"), fmt(ch["static"][i, jo] / pred_ann["static"][i, jo] if pred_ann["static"][i, jo] else NAN, "%.4f")])
            md += ["### T7b/T8a per channel at %s, t_r = %s r_s (weights included where not divided by w): the NO remainder at the hold against the channel-local clock prediction (hold), with the generator, flow and clock-at-the-end predictions; the static annulus per unit weight against the exact-M law and the M -> 0 law (3/(32 pi)) r_s (R1^-2 - R2^-2)" % (ann_lab[jo], t), ""]
            table(md, ["kappa", "w", "pocket (record)", "R2 != R0 (data)", "NO", "NO / w", "pred clock (hold)", "pred gen", "pred flow", "pred clock (end)", "NO / pred clock", "NO / pred gen", "R2", "static / w", "static exact-M / w", "law / w", "static / exact-M"], rows)
            rows = []
            for sname in ("all", "pocket_free"):
                s_ = sums[sname]
                for j in (jo - 1, jo):
                    no_ = s_["NO_obs"][j]
                    st_ = s_["static_obs"][j]
                    rows.append([sname, s_["n_channels"], ann_lab[j], fmt(no_), fmt(no_ / lnr[j] / slope, "%.3e"), fmt(no_ / st_ if st_ else NAN, "%.3e"),
                                 fmt(s_["pred_clock"][j]), fmt(s_["pred_clock"][j] / lnr[j] / slope, "%.3e"), fmt(no_ / s_["pred_clock"][j] if s_["pred_clock"][j] else NAN, "%.3f"),
                                 fmt(s_["pred_gen"][j]), fmt(no_ / s_["pred_gen"][j] if s_["pred_gen"][j] else NAN, "%.3f"), fmt(s_["pred_flow"][j]), fmt(s_["pred_clock_end"][j]),
                                 fmt(s_["R2_obs"][j]) if s_["R2_obs"] is not None else "-", fmt(s_["R2_obs"][j] / lnr[j] / slope, "%.3e") if s_["R2_obs"] is not None else "-",
                                 fmt(kb[sname]["NO_obs"][j], "%.2f"), fmt(kb[sname]["pred_clock"][j], "%.2f"), fmt(kb[sname]["pred_gen"][j], "%.2f"), fmt(kb[sname]["static_obs"][j], "%.2f")])
            md += ["Channel sums at t_r = %s r_s (hold): the NO remainder, per e-fold over the slope and over the retained-space flat signal of the same channels, against the channel-local predictions; the weighted mean channel kbar of the remainder, of the predictions and of the static response" % t, ""]
            table(md, ["channels", "n", "annulus", "NO", "NO / slope per e-fold", "NO / retained", "pred clock", "pred clock / slope", "NO / pred clock", "pred gen", "NO / pred gen", "pred flow", "pred clock (end)", "R2", "R2 / slope", "kbar NO", "kbar pred clock", "kbar pred gen", "kbar static"], rows)
        oh, oe = obs.get("hold"), obs.get("end")
        tot_pred = {k: v.sum(axis=1) for k, v in pred.items()}
        law_tot = law_bins.sum(axis=1)
        static_check = NAN
        if oh is not None and oe is not None:
            static_check = float(nmax(np.abs(oe["static"] - oh["static"])) / max(nmax(np.abs(oh["static"])), 1e-300))
        if oh is not None:
            rows = []
            for j, (lo, hi) in enumerate(bins):
                st = oh["static"][j]
                rows.append([fmt(lo, "%g"), fmt(hi, "%g"), sites_in[j], fmt(st), fmt(law_tot[j]), fmt(tot_pred["static"][j]), fmt(st / law_tot[j] if law_tot[j] else NAN, "%.4f"),
                             fmt(st / tot_pred["static"][j] if tot_pred["static"][j] else NAN, "%.4f")])
            if t == tr_sorted[0]:
                md += ["### T8a the static clock bins (raw - NO at the hold; ramp-independent) against the channel-local law sum_kappa w (3/(32 pi)) r_s (R1^-2 - R2^-2) = (3K(K+1)/(4 pi)) r_s (R1^-2 - R2^-2) for the weights 16 kappa (the density (3K(K+1)/(8 pi^2)) r_s/r^5) and against its exact-M form; the static response from the '_end' totals differs from the '_hold' one by %s relative (both at s = 1)" % fmt(static_check, "%.1e"), ""]
                table(md, ["R_lo/r_s", "R_hi/r_s", "sites", "raw - NO", "law (M -> 0)", "law (exact M)", "obs / law", "obs / exact-M"], rows)
            rows = []
            for j, (lo, hi) in enumerate(bins):
                lq = math.log(hi / lo)
                st = oh["static"][j]
                noh = oh["NO"][j]
                noe = oe["NO"][j] if oe is not None else NAN
                r2h = oh["R2"][j] if oh["R2"] is not None else NAN
                rows.append([fmt(lo, "%g"), fmt(hi, "%g"), fmt(noh), fmt(noh / lq / slope, "%.3e"), fmt(noh / st if st else NAN, "%.3e"), fmt(tot_pred["clock"][j]), fmt(tot_pred["clock"][j] / lq / slope, "%.3e"),
                             fmt(noh / tot_pred["clock"][j] if tot_pred["clock"][j] else NAN, "%.3f"), fmt(tot_pred["gen"][j] / lq / slope, "%.3e"), fmt(tot_pred["flow"][j] / lq / slope, "%.3e"),
                             fmt(noe), fmt(noe / lq / slope, "%.3e"), fmt(tot_pred["clock_end"][j] / lq / slope, "%.3e"), fmt(noe / tot_pred["clock_end"][j] if tot_pred["clock_end"][j] else NAN, "%.3f"),
                             fmt(noe / noh if noh else NAN, "%.3f"), fmt(tot_pred["clock_end"][j] / tot_pred["clock"][j] if tot_pred["clock"][j] else NAN, "%.3f"),
                             fmt(r2h) if np.isfinite(r2h) else "-", fmt(r2h / lq / slope, "%.3e") if np.isfinite(r2h) else "-", fmt(r2h / st if (np.isfinite(r2h) and st) else NAN, "%.3e") if np.isfinite(r2h) else "-"])
            md += ["### T7b per bin, t_r = %s r_s: the NO remainder at the hold's end and at the ramp's end against the channel-local clock prediction at theta = %.4f and theta = 0 (summed over the channels), per e-fold over the slope and over the retained-space flat signal; the generator and flow predictions; the end/hold ratios observed and predicted (the flow part's dephasing); R2 where stored" % (t, theta), ""]
            table(md, ["R_lo/r_s", "R_hi/r_s", "NO hold", "NO hold / slope", "NO hold / retained", "pred clock hold", "pred clock hold / slope", "NO / pred (hold)", "pred gen / slope", "pred flow / slope",
                       "NO end", "NO end / slope", "pred clock end / slope", "NO / pred (end)", "NO end/hold obs", "end/hold pred", "R2 hold", "R2 / slope", "R2 / retained"], rows)
        ramps[t] = {"tau": tau, "theta": theta, "probe_offset": d, "static_end_vs_hold_rel": static_check, "sums": sums, "kbar": kb,
                    "pred_ann": {k: v for k, v in pred_ann.items()}, "law_ann": law_ann, "obs_ch": ({k: v for k, v in ch.items() if k != "data_pocket"} if ch else None),
                    "data_pocket": ({str(k): v for k, v in ch["data_pocket"].items()} if ch else None),
                    "bins_obs": {w_: ({k: v for k, v in o.items()} if o else None) for w_, o in obs.items()}, "bins_pred_total": tot_pred, "law_tot": law_tot}
    out["ramps"] = ramps
    out["static_pred_bins"] = static_pred
    out["law_bins"] = law_bins

    # ------------------------------------------------------------------ verdicts
    jo = n_ann - 1
    lab_o = ann_lab[jo]
    ta = tr_sorted[0]
    tb = tr_sorted[1] if len(tr_sorted) > 1 else None
    tc = tr_sorted[2] if len(tr_sorted) > 2 else None

    def S(t, sname, key, j=jo):
        if t is None:
            return NAN
        v = ramps[t]["sums"][sname].get(key)
        if v is None:
            return NAN
        v = np.asarray(v, float)
        return float(v[j]) if (j < len(v) and np.isfinite(v[j])) else NAN

    # C1 the ramp law
    if tb is not None:
        for sname, key, tag, kind in (("pocket_free", "NO_obs", "NO in the pocket-free channels (R2 = R0 there)", "control"), ("all", "NO_obs", "NO over all channels", "reported"),
                                      ("all", "R2_obs", "R2 over its stored channels", "control")):
            oa, ob = S(ta, sname, key), S(tb, sname, key)
            if not (np.isfinite(oa) and np.isfinite(ob)) or ob == 0:
                continue
            pa_, pb_ = S(ta, sname, "pred_clock"), S(tb, sname, "pred_clock")
            ga, gb = S(ta, sname, "pred_gen"), S(tb, sname, "pred_gen")
            r_obs = oa / ob
            r_pred = pa_ / pb_ if (np.isfinite(pb_) and pb_) else NAN
            r_gen = ga / gb if (np.isfinite(gb) and gb) else NAN
            ok = bool(np.isfinite(r_pred) and r_obs >= RULES["ramp_ratio_min"] and (1.0 / RULES["ramp_ratio_band"]) <= r_obs / r_pred <= RULES["ramp_ratio_band"])
            verdict(RES, "%s T7b-C1 the ramp law at %s: observed %s(t_r %s)/%s(t_r %s) [%s] against the channel-local clock prediction's ratio (hold)" % (name, lab_o, key[:2], ta, key[:2], tb, tag),
                    [r_obs, r_pred, (r_obs / r_pred) if (np.isfinite(r_pred) and r_pred) else NAN], "observed >= %g and observed/predicted within [1/%g, %g]" % (RULES["ramp_ratio_min"], RULES["ramp_ratio_band"], RULES["ramp_ratio_band"]),
                    ok if kind == "control" else None, "the generator (pair) prediction's ratio %s; the pure tau^-2 law %g; a tau-independent remainder would give 1; the model's systematic at the slower ramp is O(1)" % (fmt(r_gen, "%.2f"), (float(tb) / float(ta)) ** 2), kind=kind)
    # C2 the magnitude at the fastest ramp
    for sname, key, kind in (("pocket_free", "NO_obs", "control"), ("all", "NO_obs", "reported"), ("all", "R2_obs", "control")):
        o_ = S(ta, sname, key)
        p_ = S(ta, sname, "pred_clock")
        g_ = S(ta, sname, "pred_gen")
        if not (np.isfinite(o_) and np.isfinite(p_)) or p_ == 0:
            continue
        ratio = o_ / p_
        ok = bool(RULES["magnitude_band"][0] <= ratio <= RULES["magnitude_band"][1])
        verdict(RES, "%s T7b-C2 the magnitude at the fastest ramp (t_r %s r_s), %s: observed %s over the channel-local clock prediction (hold) [%s channels]" % (name, ta, lab_o, key[:2], sname), ratio,
                "within [%g, %g]" % RULES["magnitude_band"], ok if kind == "control" else None,
                "over the generator (pair) prediction %s; over the 3D continuum's clock %s and gen %s; the model's systematic at this ramp about 30 percent (propagation, K truncation)" % (
                    fmt(o_ / g_ if (np.isfinite(g_) and g_) else NAN, "%.3f"), fmt(o_ / lnr[jo] / slope / cont[ta]["clock_over_slope"] if cont[ta]["clock_over_slope"] else NAN, "%.3f"),
                    fmt(o_ / lnr[jo] / slope / cont[ta]["gen_over_slope"] if cont[ta]["gen_over_slope"] else NAN, "%.3f")), kind=kind)
    # C3 the channel structure
    kbs = ramps[ta]["kbar"]["all"]["static_obs"][jo]
    slow = tr_sorted[1:] if len(tr_sorted) > 1 else tr_sorted
    vals = [ramps[t]["kbar"]["all"]["NO_obs"][jo] for t in slow]
    ok = bool(np.isfinite(kbs) and all(np.isfinite(v) and v <= kbs - RULES["kbar_drop"] for v in vals))
    verdict(RES, "%s T7b-C3 the channel structure at %s: the weighted mean channel of the NO remainder for t_r %s r_s, and the static response's" % (name, lab_o, ", ".join(slow)), vals + [kbs],
            "each below the static response's by at least %g" % RULES["kbar_drop"], ok,
            "predicted (channel-local clock, hold): %s; the fastest ramp (t_r %s) observed %s, predicted %s; an excitation sits in the channels the ramp does not keep adiabatic, the static response in all channels nearly equally per unit weight" % (
                ", ".join(fmt(ramps[t]["kbar"]["all"]["pred_clock"][jo], "%.1f") for t in slow), ta, fmt(ramps[ta]["kbar"]["all"]["NO_obs"][jo], "%.1f"), fmt(ramps[ta]["kbar"]["all"]["pred_clock"][jo], "%.1f")))
    # T7c the pocket-free remainder (sizes) and the check against R2
    for t in tr_sorted:
        for j in (jo - 1, jo):
            o_ = S(t, "pocket_free", "NO_obs", j)
            st = S(t, "pocket_free", "static_obs", j)
            if not np.isfinite(o_):
                continue
            rel = o_ / lnr[j] / slope
            verdict(RES, "%s T7c the remainder under the declared reference in the pocket-free channels (NO = R2 there; %d channels) per e-fold over the full-zone slope, t_r %s r_s, %s" % (
                name, ramps[t]["sums"]["pocket_free"]["n_channels"], t, ann_lab[j]), rel, "< %g (literal threshold)" % RULES["annulus_threshold_over_slope"], bool(abs(rel) < RULES["annulus_threshold_over_slope"]),
                    "over the retained-space flat signal of the same channels %s; the pocket channels' NO %s (per e-fold over the slope %s); predicted clock %s" % (
                        fmt(o_ / st if st else NAN, "%.3e"), fmt(S(t, "all", "NO_obs", j) - o_, "%.3e"), fmt((S(t, "all", "NO_obs", j) - o_) / lnr[j] / slope, "%.3e"), fmt(S(t, "pocket_free", "pred_clock", j) / lnr[j] / slope, "%.3e")), kind="size")
    chk = []
    for t in tr_sorted:
        s_ = ramps[t]["sums"]["pocket_free"]
        if s_["R2_obs"] is not None and s_["NO_obs_on_R2_channels"] is not None:
            den_ = np.maximum(np.abs(np.asarray(s_["NO_obs_on_R2_channels"], float)), 1e-300)
            chk.append(nmax(np.abs(np.asarray(s_["R2_obs"], float) - np.asarray(s_["NO_obs_on_R2_channels"], float)) / den_))
    if chk:
        v_ = nmax(chk)
        verdict(RES, "%s T7c-check the pocket-free NO equals the pocket-free R2 where R2 is stored (relative, over ramps and probe annuli)" % name, v_, "< %g" % RULES["pocket_free_R2_check_rel"], bool(v_ < RULES["pocket_free_R2_check_rel"]),
                "the record's pocket flags %s; the data's R2 != R0 channels %s" % (out["pocket_channels_record"], [k for k, v in (ramps[tr_sorted[-1]]["data_pocket"] or {}).items() if v]))
    # the tau -> infinity extrapolations (reported)
    if tb is not None:
        for sname, key in (("pocket_free", "NO_obs"), ("all", "NO_obs"), ("all", "R2_obs")):
            oa, ob = S(ta, sname, key), S(tb, sname, key)
            pa_, pb_ = S(ta, sname, "pred_clock"), S(tb, sname, "pred_clock")
            if not (np.isfinite(oa) and np.isfinite(ob) and np.isfinite(pa_) and np.isfinite(pb_)) or pa_ == pb_:
                continue
            q = (float(tb) / float(ta)) ** 2
            c2 = (q * ob - oa) / (q - 1.0)
            A = (oa - ob) / (pa_ - pb_)
            c_m = ob - A * pb_
            err = RULES["model_error_fast_ramp"] * abs(oa) / (q - 1.0)
            nrm = lnr[jo] * slope
            note = "pure tau^-2 law: c = %s +- %s (per e-fold over the slope %s +- %s); channel-local model NO = c + A P: A = %s, c = %s (%s of the slope per e-fold)" % (
                fmt(c2), fmt(err), fmt(c2 / nrm, "%.2e"), fmt(err / nrm, "%.2e"), fmt(A, "%.3f"), fmt(c_m), fmt(c_m / nrm, "%.2e"))
            if tc is not None:
                o3, p3 = S(tc, sname, key), S(tc, sname, "pred_clock")
                note += "; the t_r %s r_s ramp: observed %s (%s of the slope per e-fold) against c_m + A P = %s and the locally made excitation A P = %s" % (tc, fmt(o3), fmt(o3 / nrm, "%.2e"), fmt(c_m + A * p3), fmt(A * p3))
            verdict(RES, "%s T7b the tau -> infinity extrapolation of the remainder at %s from t_r %s and %s r_s [%s, %s channels]" % (name, lab_o, ta, tb, key[:2], sname), [c2 / nrm, c_m / nrm],
                    "reported: a tau-independent part would be a residual of the subtraction or the reference difference; the error is the model's at the fastest ramp", None, note, kind="reported")
    # the end/hold ratios (reported)
    sel = [j for j, (lo, hi) in enumerate(bins) if int(round(lo * rs)) >= RULES["static_law_inner_sites"] and hi <= RULES["static_law_outer_over_rs"] + 1e-9]
    for t in tr_sorted:
        oh, oe = ramps[t]["bins_obs"].get("hold"), ramps[t]["bins_obs"].get("end")
        tp = ramps[t]["bins_pred_total"]
        if oh is None or oe is None or not sel:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            r_obs = oe["NO"][sel] / oh["NO"][sel]
            r_pred = tp["clock_end"][sel] / tp["clock"][sel]
        verdict(RES, "%s T7b the NO remainder at the ramp's end over its value at the hold's end, far bins (inner edge >= %d sites, outer edge <= %g r_s), t_r %s r_s: median observed, median predicted" % (
            name, RULES["static_law_inner_sites"], RULES["static_law_outer_over_rs"], t), [float(np.nanmedian(r_obs)), float(np.nanmedian(r_pred))],
                "reported: the flow part's dephasing during the hold (clock = gen - flow)", None, "per bin observed %s; predicted %s" % (", ".join(fmt(x, "%.2f") for x in r_obs), ", ".join(fmt(x, "%.2f") for x in r_pred)), kind="reported")
    # T8a the static law (controls)
    oh = ramps[tr_sorted[-1]]["bins_obs"].get("hold")
    if oh is not None and static_pred is not None:
        law_tot = law_bins.sum(axis=1)
        ex_tot = static_pred.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_law = oh["static"] / law_tot
            r_ex = oh["static"] / ex_tot
        out["static_ratio_law_by_bin"] = r_law
        out["static_ratio_exact_by_bin"] = r_ex
        if sel:
            d_law = nmax(np.abs(r_law[sel] - 1.0))
            d_ex = nmax(np.abs(r_ex[sel] - 1.0))
            verdict(RES, "%s T8a the static clock bins against the channel-local law (3K(K+1)/(4 pi)) r_s (R1^-2 - R2^-2), K %d, bins with inner edge >= %d sites and outer edge <= %g r_s: max |obs/law - 1|" % (
                name, kmax, RULES["static_law_inner_sites"], RULES["static_law_outer_over_rs"]), d_law, "< %g" % RULES["static_law_tol"], bool(np.isfinite(d_law) and d_law < RULES["static_law_tol"]),
                    "ratios by bin %s; the inner bins approach the law from below: %s" % (", ".join("[%g, %g] %s" % (bins[j][0], bins[j][1], fmt(r_law[j], "%.3f")) for j in sel),
                                                                                           ", ".join("[%g, %g] %s" % (bins[j][0], bins[j][1], fmt(r_law[j], "%.3f")) for j in range(nb) if j not in sel)))
            verdict(RES, "%s T8a the same against the exact-M channel-local law: max |obs/exact - 1|" % name, d_ex, "< %g" % RULES["static_law_tol"], bool(np.isfinite(d_ex) and d_ex < RULES["static_law_tol"]),
                    "ratios by bin %s" % ", ".join("[%g, %g] %s" % (bins[j][0], bins[j][1], fmt(r_ex[j], "%.3f")) for j in sel))
    chs = ramps[tr_sorted[-1]]["obs_ch"]
    if chs is not None:
        pe = ramps[tr_sorted[-1]]["pred_ann"]["static"][:, jo]
        with np.errstate(divide="ignore", invalid="ignore"):
            r_ch = chs["static"][:, jo] / pe
        m2 = np.array([k >= 2 for k in kappas])
        d_ch = nmax(np.abs(r_ch[m2] - 1.0)) if m2.any() else NAN
        verdict(RES, "%s T8a per channel at %s: the static annulus against the exact-M channel-local law, kappa >= 2, max |ratio - 1|" % (name, lab_o), d_ch, "< %g" % RULES["static_law_channel_tol"],
                bool(np.isfinite(d_ch) and d_ch < RULES["static_law_channel_tol"]), "kappa 1 (reported; the expansion parameter 1/kappa is of order 1 there) %s; all channels %s" % (
                    fmt(r_ch[0], "%.4f"), ", ".join("%d: %s" % (k, fmt(x, "%.3f")) for k, x in zip(kappas, r_ch))))
        verdict(RES, "%s T8a the probe annuli against the M -> 0 law, all channels (the approach from below), t_r %s r_s" % (name, tr_sorted[-1]),
                [S(tr_sorted[-1], "all", "static_obs", j) / S(tr_sorted[-1], "all", "law_static", j) if S(tr_sorted[-1], "all", "law_static", j) else NAN for j in range(n_ann)],
                "reported: %s" % ", ".join(ann_lab), None, "against the exact-M law %s" % ", ".join(fmt(S(tr_sorted[-1], "all", "static_obs", j) / S(tr_sorted[-1], "all", "pred_static", j) if S(tr_sorted[-1], "all", "pred_static", j) else NAN, "%.4f") for j in range(n_ann)), kind="reported")
    RES.setdefault("B1", {})[name] = out
    return out


def two_size(Oa, Ob, a, RES, md):
    """the static clock response at the smaller and the larger r_s at equal R/r_s and t_r/r_s: the ratio larger/smaller against the
    scale-covariant small/large (an energy at fixed R/r_s scales as 1/r_s); and at equal lattice sites: the ratio against r_s_big/r_s_small
    (the response quadratic in the flow amplitude, v^2 = r_s/R at fixed R)."""
    small, big = (Oa, Ob) if Oa.rs < Ob.rs else (Ob, Oa)
    kmax = min(a.kmax, small.kdone or a.kmax, big.kdone or a.kmax)
    common = [t for t in small.trs if t in big.trs]
    labs = ["[%g, %g] r_s" % (lo, hi) for lo, hi in zip(small.probes[:-1], small.probes[1:])]
    labs_big = ["[%g, %g] r_s" % (lo, hi) for lo, hi in zip(big.probes[:-1], big.probes[1:])]
    edges_small = [(int(round(lo * small.rs)), int(round(hi * small.rs))) for lo, hi in zip(small.probes[:-1], small.probes[1:])]
    edges_big = [(int(round(lo * big.rs)), int(round(hi * big.rs))) for lo, hi in zip(big.probes[:-1], big.probes[1:])]
    pairs = [(i, j) for i, e in enumerate(edges_small) for j, f in enumerate(edges_big) if e == f]
    out = {"small_rs": small.rs, "big_rs": big.rs, "kmax": kmax, "scale_covariant": small.rs / big.rs, "rs_ratio": big.rs / small.rs, "ratios": {}, "rs_linearity": {},
           "lattice_annulus_pairs": [{"sites": list(edges_small[i]), "small": labs[i], "big": labs_big[j]} for i, j in pairs]}
    rows, rows_lin = [], []
    for t in common:
        s1, _ = small.annuli(t, kmax, "raw")
        n1, _ = small.annuli(t, kmax, "NO")
        s2, _ = big.annuli(t, kmax, "raw")
        n2, _ = big.annuli(t, kmax, "NO")
        if any(x is None for x in (s1, n1, s2, n2)):
            continue
        st1, st2 = s1 - n1, s2 - n2
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = st2 / st1
        out["ratios"][t] = ratio.tolist()
        rows.append([t] + [fmt(x, "%.4f") for x in ratio] + [fmt(out["scale_covariant"], "%.3f")])
        lin = [float(st2[j] / st1[i]) if st1[i] else NAN for i, j in pairs]
        out["rs_linearity"][t] = lin
        rows_lin.append([t] + [fmt(x, "%.5f") for x in lin] + [fmt(out["rs_ratio"], "%.3f")])
    md += ["## Two sizes: the static clock response at r_s %g over r_s %g at equal R/r_s and t_r/r_s (kappa <= %d) against the scale-covariant %.3f" % (big.rs, small.rs, kmax, out["scale_covariant"]), ""]
    table(md, ["t_r/r_s"] + labs + ["scale-covariant"], rows)
    if pairs:
        md += ["The same at equal lattice sites (the response quadratic in the flow amplitude gives the ratio r_s %g / r_s %g = %.3f exactly): %s" % (
            big.rs, small.rs, out["rs_ratio"], "; ".join("sites [%d, %d] = r_s %g's %s = r_s %g's %s" % (p["sites"][0], p["sites"][1], small.rs, p["small"], big.rs, p["big"]) for p in out["lattice_annulus_pairs"])), ""]
        table(md, ["t_r/r_s"] + ["sites [%d, %d]" % tuple(p["sites"]) for p in out["lattice_annulus_pairs"]] + ["r_s ratio"], rows_lin)
    if out["ratios"]:
        tl = common[-1]
        r = out["ratios"][tl][-1]
        verdict(RES, "T3c two-size ratio of the static response at %s, t_r %s r_s" % (labs[-1], tl), r, "within %g of %.3f" % (RULES["two_size_tol"], out["scale_covariant"]),
                bool(np.isfinite(r) and abs(r - out["scale_covariant"]) < RULES["two_size_tol"]))
        for q, (i, j) in enumerate(pairs):
            x = out["rs_linearity"][tl][q]
            verdict(RES, "T3f linearity in r_s at the lattice annulus [%d, %d] sites (r_s %g %s over r_s %g %s), t_r %s r_s" % (edges_small[i][0], edges_small[i][1], big.rs, labs_big[j], small.rs, labs[i], tl), x,
                    "within %g percent of %.3f" % (100 * RULES["rs_linearity_tol"], out["rs_ratio"]), bool(np.isfinite(x) and abs(x / out["rs_ratio"] - 1.0) < RULES["rs_linearity_tol"]),
                    "the deviation from the ratio is the quartic term's share in the flow amplitude at the larger size: %s" % fmt(x / out["rs_ratio"] - 1.0, "%.1e"))
    RES["two_size"] = out
    return out


# ------------------------------------------------------------------------------------------------------------------- the static run (gated)
def static_run(a):
    """the smallest run supplying the far-zone clock, flow and generator annuli (spec section 5): the instantaneous ground state of
    h(s) = h_0 + s V per channel at a few s, no time evolution. Not part of B1. Starts only with --lead-approved."""
    if not a.lead_approved:
        print("the static run is not approved: pass --lead-approved only after the lead's approval (spec section 5)")
        return 2
    if not a.out:
        print("--out is required")
        return 2
    sys.path.insert(0, a.engine_dir)
    import engine_vacflow_v7 as V7  # noqa: E402
    V7.CC["full"] = (a.chain_current == "full")
    rs = a.rs
    Rfull = np.arange(int(64 * rs), dtype=float) + 1.0
    n = len(Rfull)
    svals = [float(x) for x in a.svals.split(",")]
    acc = {s: {"clock": np.zeros(n), "flow": np.zeros(n), "gen": np.zeros(n)} for s in svals}
    flat_flow = {}
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    log = []
    for kappa in range(1, a.kmax + 1):
        n0 = int(np.searchsorted(Rfull, kappa / a.Ecut))
        Rk = Rfull[n0:]
        h0, Q = V7.chain_sparse(Rk, a.m, kappa)
        V = V7.flow_term(Q, V7.profile(Rk, rs, a.profile))
        h0 = h0.tocsr()
        V = V.tocsr()

        def sea(s):
            e, U = np.linalg.eigh((h0 + s * V).toarray())
            return U[:, e < 0]

        def dens(A, U):
            AU = A @ U
            return np.real(np.einsum("ik,ik->i", U.conj(), AU)).reshape(-1, 2).sum(axis=1)

        Uf = sea(0.0)
        d_h0_flat = dens(h0, Uf)
        flat_flow[kappa] = float(dens(V, Uf).sum())
        w = 16.0 * kappa
        for s in svals:
            U = sea(s)
            clock = dens(h0, U) - d_h0_flat
            flow = s * dens(V, U)
            acc[s]["clock"][n0:] += w * clock
            acc[s]["flow"][n0:] += w * flow
            acc[s]["gen"][n0:] += w * (clock + flow)
        msg = "kappa %d done (%.0f s; n0 %d; Tr[flat V] %.3e)" % (kappa, time.time() - t0, n0, flat_flow[kappa])
        print(msg, flush=True)
        log.append(msg)
    out = {"args": vars(a), "svals": svals, "flat_flow_expectation_per_channel": flat_flow, "seconds": time.time() - t0, "per_s": {}}
    md = ["# S12 static response (the run of spec section 5): r_s %g, m %g, profile %s, current %s, kappa <= %d, E_cut %g" % (rs, a.m, a.profile, a.chain_current, a.kmax, a.Ecut), ""]
    slope = slope_full_zone(a.BL, rs)
    for s in svals:
        bins = {k: binned(acc[s][k], rs, BIN_EDGES) for k in ("clock", "flow", "gen")}
        rows = []
        per = []
        for j in range(len(bins["clock"])):
            bc, bf, bg = bins["clock"][j], bins["flow"][j], bins["gen"][j]
            row = {"lo": bc["lo"], "hi": bc["hi"], "clock": bc["E"], "flow": bf["E"], "gen": bg["E"], "flow_over_clock": bf["E"] / bc["E"] if bc["E"] else NAN,
                   "gen_over_clock": bg["E"] / bc["E"] if bc["E"] else NAN, "boundary_term_gen_minus_half_flow_over_abs_flow": (bg["E"] - bf["E"] / 2) / abs(bf["E"]) if bf["E"] else NAN,
                   "clock_per_efold_over_slope": bc["E"] / bc["ln_ratio"] / slope}
            per.append(row)
            rows.append([fmt(row["lo"], "%g"), fmt(row["hi"], "%g"), fmt(row["clock"]), fmt(row["flow"]), fmt(row["gen"]), fmt(row["flow_over_clock"], "%.4f"), fmt(row["gen_over_clock"], "%.4f"),
                         fmt(row["boundary_term_gen_minus_half_flow_over_abs_flow"], "%.3e"), fmt(row["clock_per_efold_over_slope"], "%.3e")])
        fits = {}
        for lab, rlo, rhi in FIT_RANGES:
            fits[lab] = {}
            for k in ("clock", "flow", "gen"):
                sel = [b for b in bins[k] if b["lo"] >= rlo - 1e-9 and b["hi"] <= rhi + 1e-9]
                fits[lab][k] = fit_powerlaw([b["lo"] * rs for b in sel], [b["hi"] * rs for b in sel], [abs(b["E"]) for b in sel]) if len(sel) >= 2 else {"alpha": NAN, "err": NAN}
        tot = {k: float(acc[s][k].sum()) for k in ("clock", "flow", "gen")}
        out["per_s"][str(s)] = {"bins": per, "fits": fits, "whole_chain": tot, "whole_chain_gen_over_flow": tot["gen"] / tot["flow"] if tot["flow"] else NAN,
                                "density": {k: acc[s][k].tolist() for k in ("clock", "flow", "gen")}}
        md += ["## s = %g: whole chain clock %s, flow %s, generator %s (gen/flow %s; the virial 1/2 in the quadratic regime)" % (s, fmt(tot["clock"]), fmt(tot["flow"]), fmt(tot["gen"]), fmt(out["per_s"][str(s)]["whole_chain_gen_over_flow"], "%.4f")), ""]
        table(md, ["R_lo/r_s", "R_hi/r_s", "clock", "flow", "gen", "flow/clock", "gen/clock", "(gen - flow/2)/|flow|", "clock per e-fold / slope"], rows)
        rows = [[lab, k, fmt(fits[lab][k]["alpha"], "%.3f"), fmt(fits[lab][k].get("err", NAN), "%.3f")] for lab in fits for k in ("clock", "flow", "gen")]
        table(md, ["fit range", "part", "alpha (|E| ~ R^alpha)", "err"], rows)
    out["log"] = log
    json.dump(clean(out), open(os.path.join(a.out, "s12_static_response.json"), "w"), indent=1)
    open(os.path.join(a.out, "s12_static_tables.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


# ---------------------------------------------------------------------------------------------------------------------------- selftest
def synthetic_vf(rs=32.0, tr=32.0, nramp=64, nhold=4):
    """a vf10-like record with planted quadratic responses: Tr[R0 V_c] = -B s - B4 s^3, an own term, a physical excitation ~ s^4."""
    t_r = tr * rs
    every = t_r / nramp
    keys = ["%g" % rs, "%g" % (1.5 * rs), "%g" % (2 * rs)]
    sites = {keys[0]: int(rs) - 1, keys[1]: int(1.5 * rs) - 1, keys[2]: int(2 * rs) - 1}
    plant = {keys[0]: dict(B=50.0, B4=20.0, own=1.5, E0=-500.0, exc=0.02, excf=-0.01),
             keys[1]: dict(B=56.0, B4=22.0, own=0.8, E0=-700.0, exc=0.025, excf=-0.012),
             keys[2]: dict(B=60.0, B4=24.0, own=0.3, E0=-900.0, exc=0.03, excf=-0.015)}
    chans = []
    for kappa in (1, 2):
        f = 1.0 + 0.1 * kappa
        reads = []
        times = [k * every for k in range(nramp + 1)] + [t_r + (j + 1) * 2 * rs / nhold for j in range(nhold)]
        for t in times:
            ramp = t <= t_r + 1e-9
            s = math.sin(math.pi * t / (2 * t_r)) ** 2 if ramp else 1.0
            sdot = (math.pi / (2 * t_r)) * math.sin(math.pi * t / t_r) if ramp else 0.0
            cuts = {}
            for ck, p in plant.items():
                B, B4, own, E0, exc, excf = (f * p[k] for k in ("B", "B4", "own", "E0", "exc", "excf"))
                TrR0V = -B * s - B4 * s ** 3
                gen_ref = -B * s ** 2 / 2 - B4 * s ** 4 / 4 + own * s ** 2
                gen_NO = exc * s ** 4
                TrDV = excf * s ** 3
                EKR0 = E0 + gen_ref
                EKC = EKR0 + gen_NO
                WR0 = sdot * TrR0V
                W = sdot * TrDV
                IC = -1e-3 * s ** 2 * f
                cuts[ck] = {"site": sites[ck], "E_K_C": EKC, "E_K_R0": EKR0, "W": W, "W_C": WR0 + W, "W_R0": WR0, "I_C": IC, "I_NO": IC + 1e-17, "I_R2": IC - 2e-17,
                            "IS_C": 0.99 * IC, "IS_NO": 0.99 * IC + 3e-8, "IS_R2": 0.99 * IC + 3e-8}
            p = plant[keys[0]]
            B, B4, own, exc, excf = (f * p[k] for k in ("B", "B4", "own", "exc", "excf"))
            clock_ref0 = (-B * s ** 2 / 2 - B4 * s ** 4 / 4 + own * s ** 2) - s * (-B * s - B4 * s ** 3)
            E_in_NO = exc * s ** 4 - excf * s ** 4
            reads.append({"s": s, "sdot": sdot, "E_in_NO": E_in_NO, "E_out_NO": 0.1 * s ** 4, "E_in_raw": E_in_NO + clock_ref0, "E_out_raw": 0.1 * s ** 4 + 0.3 * s ** 2,
                          "E_in_R2": E_in_NO, "E_out_R2": 0.1 * s ** 4, "cuts": cuts, "t_over_rs": t / rs, "N_p": 0.0, "N_h": 0.0, "nneg": 100, "nsea": 100,
                          "phase": "ramp" if ramp else "hold", "continuation": {"n_pocket": 0}})
        chans.append({"kappa": kappa, "n0": 0, "nsea": 100, "halvings": 0, "reads": reads, "rejected": None, "seconds": 0.0, "bound": 1.0,
                      "continuation": {"pockets_max": 0}, "weight": 16.0 * kappa, "fluences": {}})
    return {"script": "synthetic", "args": {"rs": rs, "tr": tr, "hold": 2.0, "chain_current": "synthetic", "smooth_width": 6.0, "smooth_window": 48}, "bands": [],
            "kappa_done": 2, "channels": chans, "smooth_tail": 0.5 * float(_erfc(24 / 6.0)), "totals": {}}


def synthetic_old(rs=32.0, kmax=3, inclusive=False):
    """a vf3-like record: a static clock response ~ 0.3 r_s kappa / R^3 with a staggered part, an NO pulse at 5 r_s plus a far tail; the
    probe sums below p r_s in the convention R < p r_s (inclusive False) or R <= p r_s (inclusive True; the engine's)."""
    n = int(64 * rs)
    R = np.arange(n) + 1.0
    alt = (-1.0) ** np.arange(n)
    running, totals = {}, {}
    for t in ("2", "8", "32"):
        tf = float(t)
        cum_raw = np.zeros(len(PROBES))
        cum_no = np.zeros(len(PROBES))
        run = []
        tot_raw = np.zeros(n)
        tot_no = np.zeros(n)
        for kappa in range(1, kmax + 1):
            w = 16.0 * kappa
            stat = np.where(R >= 1.5 * rs, 0.3 * rs * kappa / R ** 3 * (1 + 0.3 * alt), -0.5 / kappa)
            no = (1e-2 / tf) * np.exp(-((R - 5 * rs) / (0.5 * rs)) ** 2) + 1e-7 * rs / R
            tot_raw += w * (stat + no)
            tot_no += w * no
            if inclusive:
                cum_raw += np.array([np.sum((w * (stat + no))[R <= p * rs]) for p in PROBES])
                cum_no += np.array([np.sum((w * no)[R <= p * rs]) for p in PROBES])
            else:
                cum_raw += np.array([np.sum((w * (stat + no))[R < p * rs]) for p in PROBES])
                cum_no += np.array([np.sum((w * no)[R < p * rs]) for p in PROBES])
            run.append({"kappa": kappa, "E_raw_at": cum_raw.tolist(), "E_NO_at": cum_no.tolist()})
        running[t] = run
        totals[t] = {"raw_hold": tot_raw.tolist(), "NO_hold": tot_no.tolist(), "R2_hold": [NAN] * n}
    return {"script": "synthetic", "args": {"rs": rs}, "kappa_done": kmax, "klist": list(range(1, kmax + 1)), "pockets": {}, "running": running, "per_channel": {}, "summary": {},
            "totals": totals, "seconds": 0.0}


def synthetic_old_b1(psi_tab, rs=32.0, kmax=6, A=1.3, pocket=(2, 3), offset=2e-7, end_factor=1.2):
    """a vf3-like record with a planted excitation: the static per-site clock energy is the M -> 0 channel law w (r_s/R^3)/(4 pi) (3/4),
    the NO excitation A times the M -> 0 channel-local generator prediction w (r_s/R^3)/(4 pi) (9/8) Psi_gen(2 tau kappa/R) (tabulated),
    plus a tau-independent offset in the pocket channels (R2 = NO - offset); the '_end' NO is end_factor times the '_hold' one."""
    y_tab, psi_vals = psi_tab
    n = int(64 * rs)
    R = np.arange(n) + 1.0
    running, totals, per_channel, pockets = {}, {}, [], []
    for t in ("2", "8", "32"):
        tau = float(t) * rs
        cum = {s: np.zeros(len(PROBES)) for s in ("raw", "NO", "R2")}
        run = []
        tot = {s: np.zeros(n) for s in ("raw_hold", "NO_hold", "R2_hold", "raw_end", "NO_end", "R2_end")}
        for kappa in range(1, kmax + 1):
            w = 16.0 * kappa
            stat = np.where(R >= 1.5 * rs, w * (rs / R ** 3) / (4 * math.pi) * 0.75, -0.5 / kappa)
            y = 2.0 * tau * kappa / R
            exc = w * A * (rs / R ** 3) / (4 * math.pi) * 1.125 * np.interp(y, y_tab, psi_vals, right=0.0)
            off = (offset * w) if kappa in pocket else 0.0
            no_h, r2_h = exc + off, exc
            no_e, r2_e = end_factor * exc + off, end_factor * exc
            tot["raw_hold"] += stat + no_h
            tot["NO_hold"] += no_h
            tot["R2_hold"] += r2_h
            tot["raw_end"] += stat + no_e
            tot["NO_end"] += no_e
            tot["R2_end"] += r2_e
            for s, arr in (("raw", stat + no_h), ("NO", no_h), ("R2", r2_h)):
                cum[s] += np.array([np.sum(arr[R <= p * rs]) for p in PROBES])
            run.append({"kappa": kappa, "E_raw_at": cum["raw"].tolist(), "E_NO_at": cum["NO"].tolist(), "E_R2_at": cum["R2"].tolist(), "r2": {}})
        running[t] = run
        totals[t] = {k: v.tolist() for k, v in tot.items()}
    for kappa in range(1, kmax + 1):
        per_channel.append({"kappa": kappa, "weight": 16.0 * kappa, "pocket": 1 if kappa in pocket else 0, "n0": 0, "r2": {"halvings": 0, "rejected": False}})
        pockets.append([kappa, 1 if kappa in pocket else 0])
    return {"script": "synthetic", "args": {"rs": rs}, "kappa_done": kmax, "klist": list(range(1, kmax + 1)), "pockets": pockets, "running": running,
            "per_channel": per_channel, "summary": {}, "totals": totals, "seconds": 0.0}


def selftest():
    fails = []
    count = [0]

    def check(name, ok, detail=""):
        count[0] += 1
        print("%s %s %s" % ("ok  " if ok else "FAIL", name, detail), flush=True)
        if not ok:
            fails.append(name)

    # 1 power laws on doubling bins and on the unequal bins of BIN_EDGES (the width factor)
    for alpha in (-2.0, 0.0, -1.0, -3.0):
        for edges in (np.array([2.0, 4.0, 8.0, 16.0, 32.0]) * 32, np.array(BIN_EDGES[4:]) * 32):
            lo, hi = edges[:-1], edges[1:]
            E = np.log(hi / lo) if alpha == 0.0 else (hi ** alpha - lo ** alpha) / alpha
            f = fit_powerlaw(lo, hi, np.abs(E))
            check("power law alpha %g on %d bins" % (alpha, len(lo)), abs(f["alpha"] - alpha) < 1e-6 and abs(f["ratio_per_doubling"] - 2 ** (-alpha)) < 1e-5,
                  "fit %.8f ratio %.5f" % (f["alpha"], f["ratio_per_doubling"]))
    # 2 the s^2 coefficient
    s_grid = np.array([math.sin(math.pi * k / 128) ** 2 for k in range(0, 65)])
    y = 3.0 * s_grid ** 2 - 7.0 * s_grid ** 4 + 2.0 * s_grid ** 6
    f = small_s(s_grid, y, 0.04)
    check("small-s c2", abs(f["c2"] - 3.0) < 1e-3, "c2 %.6f c4 %.4f n %d err %.1e" % (f["c2"], f["c4"], f["n"], f["err"]))
    # 3 the intercept
    a_, b_ = intercept(s_grid, 0.5 + 2.0 * s_grid)
    check("intercept", abs(a_ - 0.5) < 1e-9 and abs(b_ - 2.0) < 1e-6, "a %.9f b %.6f" % (a_, b_))
    # 4 the synthetic vf record through the analysis
    args = argparse.Namespace(kmax=16, BL=B_L_DEFAULT, smax=0.04, hold_over_rs=2.0, ramp="sine", n_u=4001, n_r=12)
    V = VF(synthetic_vf(), "synthetic_vf")
    RES, md = {"verdicts": []}, []
    out = analyze_vf(V, args, RES, md)
    wsum = 16.0 * 1.1 + 32.0 * 1.2
    reg = out["decomposition"]["inside 1 r_s"]
    for k, v in (("flow_ref", -50.0 * wsum), ("gen_ref", (-25.0 + 1.5) * wsum), ("clock_ref", (25.0 + 1.5) * wsum)):
        got = reg["small_s"][k]["c2"]
        check("synthetic vf c2 %s" % k, abs(got - v) < 1e-3 * abs(v), "got %.6f expected %.6f" % (got, v))
    check("synthetic vf own2", abs(reg["derived"]["own2"] - 1.5 * wsum) < 1e-3 * 1.5 * wsum, "%.6f" % reg["derived"]["own2"])
    pred = reg["derived"]["ratio_predicted_from_own_share"]
    check("synthetic vf ratios predicted from the own share", abs(pred[0] - reg["derived"]["flow_over_clock"]) < 1e-6 and abs(pred[1] - reg["derived"]["gen_over_clock"]) < 1e-6,
          "%.6f %.6f against %.6f %.6f" % (pred[0], pred[1], reg["derived"]["flow_over_clock"], reg["derived"]["gen_over_clock"]))
    ann = out["decomposition"]["annulus [1, 1.5] r_s"]
    check("synthetic vf annulus B", abs(ann["derived"]["B"] - 6.0 * wsum) < 1e-3 * 6.0 * wsum, "%.6f" % ann["derived"]["B"])
    check("synthetic vf identities", out["identity"]["rel"] < 1e-12, "%.2e" % out["identity"]["rel"])
    r0 = out["residual"][V.cuts[0]]
    check("synthetic vf residual sharp (absolute)", r0["sharp_R2"]["max_abs"] < 1e-13, "%.2e" % r0["sharp_R2"]["max_abs"])
    check("synthetic vf residual smooth (tails)", 1.0 < r0["smooth_R2_over_tail"] < 100.0, "%.2f tails" % r0["smooth_R2_over_tail"])
    exp_flow1 = -(50.0 + 20.0) * wsum
    check("synthetic vf flow at s = 1 from W/sdot at the ramp's last readout", abs(reg["end_of_ramp"]["flow_ref"] - exp_flow1) < 1e-6 * abs(exp_flow1),
          "got %.6f expected %.6f (sdot %.2e)" % (reg["end_of_ramp"]["flow_ref"], exp_flow1, out["readouts"]["sdot_end"]))
    check("synthetic vf verdicts T1/T2 pass", all(v["verdict"] == "pass" for v in RES["verdicts"] if " T1 " in v["test"] or " T2 " in v["test"]))
    check("synthetic vf annuli reported, not judged", all(v["verdict"] == "reported" for v in RES["verdicts"] if " T4 the annulus" in v["test"]))
    we = out["window_euler"]["annulus [1, 1.5] r_s"]
    check("synthetic vf window-Euler reading beta/Q_c = 2 own2_ann/clock2_ann = -0.6087", abs(we["beta_over_Qc_small_s_flow"] + 1.4 / 2.3) < 1e-4 and abs(we["beta_over_Qc_small_s_gen"] + 1.4 / 2.3) < 1e-4,
          "%.5f %.5f" % (we["beta_over_Qc_small_s_flow"], we["beta_over_Qc_small_s_gen"]))
    # 5 the synthetic old record (the probe convention 'R < p r_s' planted: offset 0 must be detected)
    O = OLD(synthetic_old(32.0), "synthetic_old")
    RES2, md2 = {"verdicts": []}, []
    o = analyze_old(O, args, RES2, md2)
    t32 = o["totals"]["32"]
    far = t32["fits"]["far [4, 32] r_s"]
    check("synthetic old far alpha", abs(far["alpha"] + 2.0) < 0.03, "%.4f err %.4f" % (far["alpha"], far["err"]))
    check("synthetic old ramp spread", max(o["static_ramp_spread_rel"]) < 1e-10, "%.2e" % max(o["static_ramp_spread_rel"]))
    check("synthetic old arithmetic residual", max(o["static_arith_residual_per_efold_over_slope"]) < 1e-10, "%.2e" % max(o["static_arith_residual_per_efold_over_slope"]))
    check("synthetic old probe convention detected (offset 0) and running vs totals", t32["probe_offset_detected"] == 0 and t32["running_vs_totals_static_max_abs"] < 1e-9,
          "offset %d, %.2e" % (t32["probe_offset_detected"], t32["running_vs_totals_static_max_abs"]))
    check("synthetic old probe alpha", abs(o["static_fits_probes"]["32"]["far_probe_annuli"]["alpha"] + 2.0) < 0.03, "%.4f" % o["static_fits_probes"]["32"]["far_probe_annuli"]["alpha"])
    check("synthetic old T5a are sizes", all(v["kind"] == "size" for v in RES2["verdicts"] if " T5a " in v["test"]) and any(" T5a " in v["test"] for v in RES2["verdicts"]))
    O1 = OLD(synthetic_old(32.0, inclusive=True), "synthetic_old_inclusive")
    RES2b, md2b = {"verdicts": []}, []
    o1 = analyze_old(O1, args, RES2b, md2b)
    t1 = o1["totals"]["32"]
    check("synthetic old inclusive convention detected (offset +1); offset 0 shows the mismatch",
          t1["probe_offset_detected"] == 1 and t1["running_vs_totals_static_max_abs"] < 1e-9 and t1["running_vs_totals_static_max_abs_offset0"] > 1e-6,
          "offset %d, %.2e under it, %.2e at offset 0" % (t1["probe_offset_detected"], t1["running_vs_totals_static_max_abs"], t1["running_vs_totals_static_max_abs_offset0"]))
    check("synthetic old inclusive far alpha (bins in the detected convention)", abs(t1["fits"]["far [4, 32] r_s"]["alpha"] + 2.0) < 0.03, "%.4f" % t1["fits"]["far [4, 32] r_s"]["alpha"])
    # 6 two sizes
    O16 = OLD(synthetic_old(16.0), "synthetic_old16")
    RES3, md3 = {"verdicts": []}, []
    ts = two_size(O16, O, args, RES3, md3)
    r = ts["ratios"]["32"]
    check("synthetic two-size ratio at the far annulus", abs(r[-1] - 0.5) < 0.02, "%s" % ["%.4f" % x for x in r])
    lin = ts["rs_linearity"]["32"]
    check("synthetic linearity in r_s at fixed lattice annuli (2 pairs, ratio 2)", len(lin) == 2 and all(abs(x - 2.0) < 1e-8 for x in lin), "%s" % ["%.10f" % x for x in lin])
    # 7 the compare path
    RES4, md4 = {"verdicts": []}, []
    cv = compare_vf(V, VF(synthetic_vf(), "synthetic_vf_b"), args, RES4, md4)
    check("synthetic compare", cv["worst_rel"] == 0.0 or cv["worst_rel"] < 1e-15, "%s" % cv["worst_rel"])

    # ---------------------------------------------------------------- B1 (version 3)
    # 8 the filter's values and the pair integrals
    check("Phi(0), Phi(pi), Phi(2 pi) = 1, pi/4, 1/3", abs(float(phi_sine(0.0)) - 1.0) < 1e-12 and abs(float(phi_sine(math.pi)) - math.pi / 4) < 1e-12 and abs(float(phi_sine(2 * math.pi)) - 1.0 / 3) < 1e-12,
          "%.12f %.12f %.12f" % (phi_sine(0.0), phi_sine(math.pi), phi_sine(2 * math.pi)))
    check("Phi smoothstep(0) = 1 and the small-x form joins", abs(float(phi_smoothstep(0.0)) - 1.0) < 1e-12 and abs(float(phi_smoothstep(2e-4)) - float(phi_smoothstep(2.0001e-4))) < 1e-9)
    Iss = I_part("gen", 0.0, "smoothstep")
    check("I_gen smoothstep = 9 (closed form) by quadrature", abs(Iss - 9.0) < 2e-3, "%.5f" % Iss)
    Is = I_part("gen", 0.0, "sine")
    Ist, in_in, in_out = I_gen_st("sine")
    check("I_gen sine by the x-quadrature and by the (s, t) identity agree, 9.4206", abs(Is - Ist) < 3e-3 and abs(Is - 9.4206) < 3e-3, "%.5f %.5f (in-in %.5f, in-out %.5f)" % (Is, Ist, in_in, in_out))
    Ist9 = I_gen_st("smoothstep")
    check("the (s, t) identity gives 3 + 6 = 9 for the smoothstep", abs(Ist9[0] - 9.0) < 2e-3 and abs(Ist9[1] - 3.0) < 2e-3, "%.5f %.5f %.5f" % Ist9)
    check("B_pair/B_L coefficients 3.2778 (sine), 3.1314 (smoothstep)", abs(Is / (15 * math.pi ** 2 * B_L_DEFAULT) - 3.2778) < 2e-3 and abs(Iss / (15 * math.pi ** 2 * B_L_DEFAULT) - 3.1314) < 2e-3,
          "%.4f %.4f" % (Is / (15 * math.pi ** 2 * B_L_DEFAULT), Iss / (15 * math.pi ** 2 * B_L_DEFAULT)))
    # 9 J(theta): closed form against quadrature and the hand values
    for th, ref in ((1.0, 0.4749), (0.25, 2.4388), (0.0625, 6.785)):
        jc, jq = J_theta(th), J_theta_quad(th)
        check("J(%g) closed form %.4f = quadrature, hand %.4f" % (th, jc, ref), abs(jc - jq) < 1e-6 and abs(jc - ref) < 2e-3, "%.6f %.6f" % (jc, jq))
    # 10 the Abel identity behind I_flow = -2 J: the regularized direct quadrature at theta 1
    x = np.arange(0.0, 4000.0 + 0.005, 0.01)
    If_dir = simpson(x * ramp_filter("flow", x, 1.0, "sine") * np.exp(-0.002 * x), x)
    check("I_flow(theta 1) direct Abel-regularized quadrature = -2 J(1)", abs(If_dir + 2 * J_theta(1.0)) < 0.01 * 2 * J_theta(1.0), "%.5f against %.5f" % (If_dir, -2 * J_theta(1.0)))
    # 11 the channel kernel: the M -> 0 static limit 3/4, the Psi link, the convergence in n_u
    k0 = chain_kernel(1e-3, 1.0, 0.0, ("static",), 4001)["static"]
    check("chain kernel static at M 1e-3 -> 3/4", abs(k0 - 0.75) < 1e-3, "%.6f" % k0)
    kg = chain_kernel(1e-3, 200.0, 0.3, ("gen", "flow", "clock"), 4001)
    pg = psi_part(2 * 200.0 * 1e-3, 0.3, "gen")
    pf = psi_part(2 * 200.0 * 1e-3, 0.3, "flow")
    check("chain kernel gen and flow at small M = (9/8) Psi(2 tau M)", abs(kg["gen"] - 1.125 * pg) < 1e-3 * abs(1.125 * pg) and abs(kg["flow"] - 1.125 * pf) < 2e-3 * max(abs(1.125 * pf), 1e-3),
          "%.6f vs %.6f; %.6f vs %.6f" % (kg["gen"], 1.125 * pg, kg["flow"], 1.125 * pf))
    check("chain kernel clock = gen - flow", abs(kg["clock"] - (kg["gen"] - kg["flow"])) < 1e-12)
    ka, kb_ = chain_kernel(0.05, 1024.0, 1.0 / 16, ("static", "gen", "flow"), 4001), chain_kernel(0.05, 1024.0, 1.0 / 16, ("static", "gen", "flow"), 8001)
    check("chain kernel converged in n_u (4001 against 8001) at M 0.05, tau 1024, theta 1/16", abs(ka["static"] - kb_["static"]) < 1e-4 * ka["static"] and abs(ka["gen"] - kb_["gen"]) < 1e-3 * max(abs(ka["gen"]), 1e-6)
          and abs(ka["flow"] - kb_["flow"]) < 1e-3 * max(abs(ka["flow"]), 1e-6), "%.3e %.3e %.3e against %.3e %.3e %.3e" % (ka["static"], ka["gen"], ka["flow"], kb_["static"], kb_["gen"], kb_["flow"]))
    # 12 the dense-channel limit reproduces the 3D continuum: sum kappa Psi_gen(2 tau kappa/r) -> (r/2 tau)^2 (4/15) I_gen
    r_, tau_, K_ = 1.0e4, 100.0, 1500
    kk = np.arange(1, K_ + 1, dtype=float)
    ps = psi_part(2 * tau_ * kk / r_, 0.0, "gen", n_u=3001)
    dense = float(np.sum(kk * ps))
    cont_ = (r_ / (2 * tau_)) ** 2 * (4.0 / 15.0) * Is
    check("dense-channel sum of the gen kernel = (r/2 tau)^2 (4/15) I_gen (the 2 normalizations agree)", abs(dense - cont_) < 5e-3 * cont_, "%.3f against %.3f" % (dense, cont_))
    # 13 the static law assembly: K = 16 channels at r_s 32, [8, 16] r_s with M forced tiny reproduces (3K(K+1)/(4 pi)) r_s (R1^-2 - R2^-2) to 1e-3, and the exact M within 5 percent
    kappas = list(range(1, 17))
    weights = [16.0 * k for k in kappas]
    cache = {}
    ex = chain_annulus(32.0, 1.0, 0.0, kappas, weights, 256.0, 512.0, ("static",), 12, 4001, "sine", cache, shift=0.0)["static"].sum()
    tiny = chain_annulus(32.0, 1.0, 0.0, [1e-3 * k for k in kappas], weights, 256.0, 512.0, ("static",), 12, 4001, "sine", {}, shift=0.0)["static"].sum()
    law = 3 * 16 * 17 / (4 * math.pi) * 32.0 * (256.0 ** -2 - 512.0 ** -2)
    check("static law assembly: tiny M -> (3K(K+1)/(4 pi)) r_s (R1^-2 - R2^-2) = %.6f; exact M within 5 percent" % law, abs(tiny - law) < 1e-3 * law and abs(ex - law) < 0.05 * law, "tiny %.6f exact %.6f" % (tiny, law) if False else "tiny %.6f exact %.6f law %.6f" % (tiny, ex, law))
    # 14 the synthetic record with the planted excitation through the B1 analysis
    y_tab = np.linspace(0.0, 400.0, 2001)
    psi_tab = (y_tab, psi_part(y_tab, 0.0, "gen", n_u=3001))
    Ob = OLD(synthetic_old_b1(psi_tab), "synthetic_b1")
    RES5, md5 = {"verdicts": []}, []
    analyze_old(Ob, args, RES5, md5)
    b1 = analyze_b1_old(Ob, args, RES5, md5, {})
    jo = len(PROBES) - 2
    check("synthetic b1 pocket channels from the record's flags and from the data", b1["pocket_channels_record"] == [2, 3] and sorted(int(k) for k, v in b1["ramps"]["32"]["data_pocket"].items() if v) == [2, 3],
          "%s %s" % (b1["pocket_channels_record"], b1["ramps"]["32"]["data_pocket"]))
    for t in ("2", "8"):
        s_ = b1["ramps"][t]["sums"]["pocket_free"]
        ratio = float(s_["NO_obs"][jo] / s_["pred_gen"][jo])
        check("synthetic b1 pocket-free NO over the generator prediction at [8, 16] r_s, t_r %s = planted 1.3 (exact M against M -> 0)" % t, abs(ratio - 1.3) < 0.04 * 1.3, "%.4f" % ratio)
    s2, s8 = b1["ramps"]["2"]["sums"]["pocket_free"], b1["ramps"]["8"]["sums"]["pocket_free"]
    r_obs = float(s2["NO_obs"][jo] / s8["NO_obs"][jo])
    r_gen = float(s2["pred_gen"][jo] / s8["pred_gen"][jo])
    check("synthetic b1 ramp ratio of the pocket-free NO = the generator prediction's ratio", abs(r_obs / r_gen - 1.0) < 0.03, "%.4f %.4f" % (r_obs, r_gen))
    c1 = [v for v in RES5["verdicts"] if "T7b-C1" in v["test"] and "pocket-free" in v["test"]]
    check("synthetic b1 C1 verdict present with finite values", len(c1) == 1 and all(np.isfinite(x) for x in c1[0]["value"]), "%s" % (c1[0]["value"] if c1 else None))
    ch8 = b1["ramps"]["8"]["obs_ch"]
    pa8 = b1["ramps"]["8"]["pred_ann"]["gen"]
    offs = [(ch8["NO"][k - 1, jo] - 1.3 * pa8[k - 1, jo]) / (16.0 * k * 256) for k in (2, 3)]
    check("synthetic b1 the pocket channels' offset recovered (2e-7 per site per unit weight)", all(abs(x - 2e-7) < 0.05 * 2e-7 for x in offs), "%s" % ["%.3e" % x for x in offs])
    chk = [v for v in RES5["verdicts"] if "T7c-check" in v["test"]]
    check("synthetic b1 pocket-free NO equals pocket-free R2", len(chk) == 1 and chk[0]["verdict"] == "pass", "%s" % (chk[0]["value"] if chk else None))
    kb_free = b1["ramps"]["8"]["kbar"]["pocket_free"]
    check("synthetic b1 kbar of the pocket-free remainder = the generator prediction's kbar", abs(kb_free["NO_obs"][jo] - kb_free["pred_gen"][jo]) < 0.1, "%.3f %.3f" % (kb_free["NO_obs"][jo], kb_free["pred_gen"][jo]))
    bins = b1["bins"]
    rl, rx = np.asarray(b1["static_ratio_law_by_bin"]), np.asarray(b1["static_ratio_exact_by_bin"])
    selb = [j for j, (lo, hi) in enumerate(bins) if lo >= 4.0]
    check("synthetic b1 static bins against the planted M -> 0 law (2e-3) and the exact-M law (3 percent) from 4 r_s on", nmax(np.abs(rl[selb] - 1.0)) < 2e-3 and nmax(np.abs(rx[selb] - 1.0)) < 0.03,
          "%.2e %.2e" % (nmax(np.abs(rl[selb] - 1.0)), nmax(np.abs(rx[selb] - 1.0))))
    oh, oe = b1["ramps"]["2"]["bins_obs"]["hold"], b1["ramps"]["2"]["bins_obs"]["end"]
    with np.errstate(divide="ignore", invalid="ignore"):
        eh = oe["NO"] / oh["NO"]
    check("synthetic b1 end/hold NO ratio 1.2 and the static identical at the end and the hold", nmax(np.abs(eh - 1.2)) < 1e-9 and b1["ramps"]["2"]["static_end_vs_hold_rel"] < 1e-12,
          "%.3e %.2e" % (nmax(np.abs(eh - 1.2)), b1["ramps"]["2"]["static_end_vs_hold_rel"]))
    print("SELFTEST %s (%d checks%s)" % ("PASS" if not fails else "FAIL", count[0], "" if not fails else "; failed: " + ", ".join(fails)))
    return 0 if not fails else 1


# -------------------------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vf10")
    ap.add_argument("--vf9")
    ap.add_argument("--vf3-32", dest="vf3_32")
    ap.add_argument("--vf3-16", dest="vf3_16")
    ap.add_argument("--vf7")
    ap.add_argument("--out")
    ap.add_argument("--kmax", type=int, default=16)
    ap.add_argument("--BL", type=float, default=B_L_DEFAULT, help="a^2 B_L at m a 0 (review 1.25); the slope (9 pi/2) B_L r_s is formed per record")
    ap.add_argument("--smax", type=float, default=0.04, help="the s range of the quadratic fits (0 < s <= smax)")
    ap.add_argument("--hold-over-rs", dest="hold_over_rs", type=float, default=2.0, help="the hold after the ramp in units of r_s (the records' 2 r_s; not in the vf3/vf7 args)")
    ap.add_argument("--ramp", default="sine", choices=["sine", "smoothstep"], help="the ramp shape of the prediction (the records' is sine: s = sin^2(pi t/2 t_r))")
    ap.add_argument("--n-u", dest="n_u", type=int, default=4001, help="Simpson points of the channel kernel's k-integral")
    ap.add_argument("--n-r", dest="n_r", type=int, default=12, help="Gauss-Legendre nodes in ln r per bin")
    ap.add_argument("--skip-b1", dest="skip_b1", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dump-keys", dest="dump_keys")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--static-run", dest="static_run", action="store_true")
    ap.add_argument("--lead-approved", dest="lead_approved", action="store_true")
    ap.add_argument("--engine-dir", dest="engine_dir", default="<internal>/code")
    ap.add_argument("--rs", type=float, default=32.0)
    ap.add_argument("--Ecut", type=float, default=6.0)
    ap.add_argument("--m", type=float, default=0.0)
    ap.add_argument("--profile", default="exact")
    ap.add_argument("--chain-current", dest="chain_current", default="full")
    ap.add_argument("--svals", default="0.02,0.04,0.5,1.0")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.dump_keys:
        print("\n".join(describe(load(a.dump_keys), a.depth, os.path.basename(a.dump_keys))))
        return
    if a.static_run:
        sys.exit(static_run(a))
    if not a.out:
        ap.error("--out is required")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    RES = {"script": "s12_postprocess_v3.py", "version": VERSION, "inputs": {k: getattr(a, k) for k in ("vf10", "vf9", "vf3_32", "vf3_16", "vf7")}, "args": vars(a), "rules": RULES, "verdicts": []}
    md = ["# S12 J-VF post-processing tables (s12_postprocess_v3.py version %d; the records read in place; a^2 B_L = %g, the full-zone flat slope (9 pi/2) B_L r_s per record; the B1 tests of busi/specs/S12_addendum_B1_v1.md added; the hold %g r_s and the ramp shape '%s' are declared inputs)" % (VERSION, a.BL, a.hold_over_rs, a.ramp), ""]
    vfs = {}
    for name, path in (("vf10", a.vf10), ("vf9", a.vf9)):
        if path:
            vfs[name] = VF(load(path), name, path)
            analyze_vf(vfs[name], a, RES, md)
    if "vf10" in vfs and "vf9" in vfs:
        compare_vf(vfs["vf9"], vfs["vf10"], a, RES, md)
        r10, r9 = RES["vf10"]["residual"], RES["vf9"]["residual"]
        rows = []
        for ck in vfs["vf10"].cuts:
            if ck in r9:
                num = r10[ck]["smooth_R2"]["max_abs"] / r9[ck]["smooth_R2"]["max_abs"] if r9[ck]["smooth_R2"]["max_abs"] else NAN
                rows.append([fmt(float(ck) / vfs["vf10"].rs, "%g"), fmt(r10[ck]["smooth_R2"]["max_abs"]), fmt(r9[ck]["smooth_R2"]["max_abs"]), fmt(num, "%.1f"),
                             fmt(RES["vf10"]["smooth_tail"] / RES["vf9"]["smooth_tail"], "%.1f"), fmt(r10[ck]["sharp_R2"]["max_abs"]), fmt(r9[ck]["sharp_R2"]["max_abs"])])
        RES["smoothed_residual_scaling"] = rows
        md += ["## The smoothed residual against the window tail between the records (consistent with the window truncation): vf10 against vf9 per cut, the tails' ratio beside it; the sharp residuals are the same rounding in both", ""]
        table(md, ["cut R/r_s", "vf10 smooth R2 max", "vf9 smooth R2 max", "ratio", "tails' ratio", "vf10 sharp R2 max", "vf9 sharp R2 max"], rows)
    olds = {}
    for name, path in (("vf3_rs32", a.vf3_32), ("vf3_rs16", a.vf3_16), ("vf7_rs32", a.vf7)):
        if path:
            olds[name] = OLD(load(path), name, path)
            analyze_old(olds[name], a, RES, md)
            if not a.skip_b1:
                t1 = time.time()
                analyze_b1_old(olds[name], a, RES, md, CACHE)
                print("B1 on %s: %.0f s" % (name, time.time() - t1), flush=True)
    if "vf3_rs16" in olds and "vf3_rs32" in olds:
        two_size(olds["vf3_rs16"], olds["vf3_rs32"], a, RES, md)
    verdict(RES, "T8b the far-annulus ratio clock : flow : generator = 1 : 2/3 : 5/3 of the kappa-truncated chain beyond 2 r_s (the window-Euler boundary term of the r^-2 channel-local functional; 1 : -2 : -1 is the full-zone logarithm's law)", None,
            "not testable from the records: no record stores a flow or generator part beyond 2 r_s", None,
            "what exists: the r^-5 density with its channel-local coefficient in the far bins (T8a) and the window-Euler reading of vf10's and vf9's exterior annuli to 2 r_s (between 0 and 8/3)")
    md += ["## Verdicts (the rules of busi/specs/S12_spec_v1.md section 3 as corrected in busi/notes/S12_judgment_v1.md, and of busi/specs/S12_addendum_B1_v1.md; kind: control = pass/fail, a failure a defect to be judged; size = met / not met against a literal threshold; reported = no rule)", ""]
    table(md, ["test", "kind", "value", "rule", "verdict", "note"], [[v["test"], v["kind"], fmt(v["value"]) if not isinstance(v["value"], list) else ", ".join(fmt(x) for x in v["value"]), v["rule"], v["verdict"], v["note"]] for v in RES["verdicts"]])
    kinds = {}
    for v in RES["verdicts"]:
        kinds.setdefault(v["kind"], {}).setdefault(v["verdict"], 0)
        kinds[v["kind"]][v["verdict"]] += 1
    md += ["Summary: " + "; ".join("%s: %s" % (k, ", ".join("%d %s" % (n_, w) for w, n_ in sorted(d.items()))) for k, d in sorted(kinds.items())), ""]
    md += ["Not in any record (spec section 2.4 and the addendum): the flow and generator energies beyond 2 r_s (so neither the far ratio 1 : 2/3 : 5/3 nor the generator part of the far excitation, the quantity the pair coefficient refers to, is testable; the clock part is compared with the flow term predicted), the clock/flow/generator split against the continued reference R2, the continued reference's annuli at r_s 32 outside the pocket-free channels (vf7's R2 sums are nan), the massive case.", ""]
    RES["seconds"] = time.time() - t0
    json.dump(clean(RES), open(os.path.join(a.out, "s12_results.json"), "w"), indent=1)
    open(os.path.join(a.out, "s12_tables.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
