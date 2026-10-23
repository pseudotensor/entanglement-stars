"""One bounded b110 certificate implementing the original pair's E20.

Serves insert I6a and item 7's light-cut ratio.  No old fitted coefficient,
native matrix calculation, axial rerun or b111 rerun is used.  Execute only
on the prescribed box CPU with the isolated python-flint 0.9.0 dependency.
"""

from contextlib import contextmanager
from fractions import Fraction
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time


OUTPUT = Path(__file__).with_name("p110_result.json")
START = time.monotonic()
REPORT = {
    "status": "running",
    "science_specification": "ETA_massive_directions_v1.tex E20.1--8",
    "purpose": "insert I6a; item 7 light-cut ratio",
    "scope": "b110 only; accepted axial and b111 checks remain fixed",
    "resources": {"wall_seconds": 300, "internal_save_deadline_seconds": 295,
                  "address_space_bytes": 805306368, "threads": 1,
                  "nice": 10, "cuda": "disabled"},
    "quadrature": {"bits": 192, "rule": "8x8 Gauss-Legendre",
                   "complex_rectangle_halfwidth_factor": 4,
                   "analytic_error": "(16/3)*4^-16*cell_area*M",
                   "panel_width_limit": "2e-5", "final_width_limit": "2e-4",
                   "evaluation_limit": 1000000, "created_cell_limit": 10000,
                   "real_bound_fallback": "signed E20 real-cell integral bound"},
    "finite_sums": {"frobenius": "n=0..63, value and derivative disks",
                    "poisson": "n=1..64, E20.5 tail starts J=65",
                    "direct": "j=1..63, E20.7 tail starts J=64",
                    "tail_images": "real/imaginary rectangles before prefactors"},
    "cover": {"triangles": ["theta=c*r,phi=c*r*t", "theta=c*r*t,phi=c*r"],
              "c": "pi/2", "t": ["0", "1"],
              "panels_per_triangle": [
                  {"chart": "node", "r": "2^-z", "z": ["2", "20"]},
                  {"chart": "direct", "r": "2^-z", "z": ["1", "2"]},
                  {"chart": "direct", "r": "1-2^-z", "z": ["1", "32"]}],
              "positive_weight": "log(2)*r*2^-z/sqrt(2)"},
    "counts": {"evaluations": 0, "created_cells": 0, "refused_images": 0,
               "refusals_by_reason": {}},
    "guard_samples": {}, "tail_radius_maxima": {}, "panels": [], "results": {},
}
LEAVES = {}
ACTIVE = None
NEXT_ID = 0
PENDING = {}
ATTEMPTED = {}
SIGNALS = {signal.SIGALRM, signal.SIGXCPU}
JOURNAL_PATH = OUTPUT.with_name("p110_attempts.jsonl")
JOURNAL = None
JOURNAL_COUNT = 0
RULE_PATH = Path('<internal>/fixed_mass_sharp_repair_result.json')
RULE_SHA = '0f28c5a62b55b59f6ad3eba202fa04c563b077da4bd0a1a74df96f20a69fcd3e'


@contextmanager
def protected():
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def journal(record):
    global JOURNAL_COUNT
    JOURNAL_COUNT += 1
    record = dict(sequence=JOURNAL_COUNT, **record)
    JOURNAL.write(json.dumps(record, allow_nan=False) + "\n")
    JOURNAL.flush()
    return JOURNAL_COUNT


def complex_interval(value):
    return dict(real=interval(value.real), imaginary=interval(value.imag))


def attempt(context, operation, publish=None):
    # A returned scientific call and all publication finish before handled
    # timeout delivery; each outcome is append-only and separately referenced.
    with protected():
        counts = REPORT['counts']
        if counts['evaluations'] >= 1000000:
            raise ArithmeticError('the agreed integrand evaluation cap was reached')
        counts['evaluations'] += 1
        identity = counts['evaluations']
        try:
            value = operation()
            payload = complex_interval(value) if hasattr(value, 'real') else value
            ref = journal(dict(event='evaluation', attempt=identity, context=context,
                               outcome='returned', raw=payload))
            if publish is not None:
                publish(value, ref)
            return value
        except NonanalyticImage as error:
            counts['refused_images'] += 1
            by_reason = counts['refusals_by_reason']
            by_reason[error.reason] = by_reason.get(error.reason, 0) + 1
            refusal = dict(reason=error.reason, tested_image=error.image,
                           tested_input=error.input_image)
            REPORT['guard_samples'].setdefault(error.reason, dict(context=context, **refusal))
            journal(dict(event='evaluation', attempt=identity, context=context,
                         outcome='refused', **refusal))
            raise
        except Exception as error:
            journal(dict(event='evaluation', attempt=identity, context=context,
                         outcome='error', reason=type(error).__name__ + ': ' + str(error)))
            raise



class NonanalyticImage(Exception):
    def __init__(self, reason, value, input_argument=None):
        self.reason = reason
        self.image = str(value)
        self.input_image = None if input_argument is None else str(input_argument)
        super().__init__(reason)


def deadline(_signal, _frame):
    raise TimeoutError("internal deadline before the hard 300-second cap")


def finite(z, stage="nonfinite_function_image"):
    if not z.is_finite():
        raise NonanalyticImage(stage, z)
    return z

def denominator(z):
    finite(z, "nonfinite_denominator")
    if z.contains(0):
        raise NonanalyticImage("denominator_contains_zero", z)
    return z

def positive_real(z, stage):
    finite(z, "nonfinite_" + stage)
    if not z.real > 0:
        raise NonanalyticImage("nonpositive_real_" + stage, z)
    return z

def positive_sqrt(z):
    positive_real(z, "sqrt_argument")
    return finite(z.sqrt(), "nonfinite_sqrt_value")

def logarithm(z, analytic):
    finite(z, "nonfinite_logarithm_argument")
    if z.imag.contains(0) and not z.real > 0:
        raise NonanalyticImage("logarithm_cut", z)
    return finite(z.log(analytic=analytic), "nonfinite_logarithm_value")

def elliptic_k(z):
    finite(z, "nonfinite_elliptic_parameter")
    if z.imag.contains(0) and not z.real < 1:
        raise NonanalyticImage("elliptic_cut", z)
    value = z.elliptic_k()
    if not value.is_finite():
        raise NonanalyticImage("nonfinite_elliptic_value", value, z)
    return value

def modulus(z, limit, stage, strict=False):
    # Moduli occur only in domain guards and rigorous tail bounds.
    radius = finite(abs(z), "nonfinite_modulus_" + stage).upper()
    passes = radius < limit if strict else radius <= limit
    if not passes:
        raise NonanalyticImage("modulus_guard_" + stage, radius, z)
    return radius

def exact_endpoint(z):
    rational = z.fmpq()
    return int(rational.p), int(rational.q)

def decimal_endpoint(z, lower, places=40):
    numerator, divisor = exact_endpoint(z)
    scale = 10 ** places
    integer = numerator * scale // divisor if lower else -((-numerator * scale) // divisor)
    digits = str(abs(integer)).zfill(places + 1)
    return ("-" if integer < 0 else "") + digits[:-places] + "." + digits[-places:]

def interval(z):
    finite(z)
    lo, hi = z.lower(), z.upper()
    p, q = exact_endpoint(lo)
    r, s = exact_endpoint(hi)
    return {"ball": z.str(65, more=True),
            "lower": decimal_endpoint(lo, True), "upper": decimal_endpoint(hi, False),
            "lower_exact": {"numerator": str(p), "denominator": str(q)},
            "upper_exact": {"numerator": str(r), "denominator": str(s)},
            "width_upper": decimal_endpoint((hi - lo).upper(), False)}

def polynomial_and_derivative(items, z):
    value, derivative = acb(items[-1]), acb(0)
    for coefficient in reversed(items[:-1]):
        derivative = derivative * z + value
        value = value * z + coefficient
    return value, derivative


def rational(x):
    p, q = exact_endpoint(x)
    return Fraction(p, q)


def point(x):
    return arb(x.numerator) / x.denominator


def rational_interval(lo, hi):
    def dec(x, lower):
        scale = 10 ** 40
        n = (x.numerator * scale // x.denominator if lower
             else -((-x.numerator * scale) // x.denominator))
        digits = str(abs(n)).zfill(41)
        return ("-" if n < 0 else "") + digits[:-40] + "." + digits[-40:]
    return {"lower": dec(lo, True), "upper": dec(hi, False),
            "lower_exact": {"numerator": str(lo.numerator), "denominator": str(lo.denominator)},
            "upper_exact": {"numerator": str(hi.numerator), "denominator": str(hi.denominator)},
            "width_upper": dec(hi - lo, False)}


def cell_record(cell):
    record = {"id": cell["id"], "bounds_z_t": [str(x) for x in cell["bounds"]],
              "depths": cell["depths"], "method": cell["method"],
              "real": interval(cell["value"].real),
              "imaginary": interval(cell["value"].imag)}
    for key in ("analytic_M", "quadrature_error", "real_bound_radius"):
        if key in cell:
            record[key] = interval(cell[key])
    if "guard_reason" in cell:
        record["guard_reason"] = cell["guard_reason"]
    for key in ('joint_ref', 'probe_refs', 'completed_node_indices', 'node_refs', 'partial_rule_status'):
        if key in cell:
            record[key] = cell[key]
    if 'weighted_accumulator' in cell:
        record['raw_weighted_accumulator'] = complex_interval(cell['weighted_accumulator'])
    return record


def current_panel_record(accepted=False):
    if ACTIVE is None:
        return None
    # Exact dyadic endpoint sums avoid dependency growth on replacing leaves.
    real_lo = real_hi = imag_lo = imag_hi = Fraction(0)
    for cell in LEAVES.values():
        value = cell["value"]
        real_lo += rational(value.real.lower())
        real_hi += rational(value.real.upper())
        imag_lo += rational(value.imag.lower())
        imag_hi += rational(value.imag.upper())
    return {**ACTIVE, "accepted": accepted, "cover_complete": bool(LEAVES), "leaf_count": len(LEAVES),
            "real": rational_interval(real_lo, real_hi),
            "imaginary": rational_interval(imag_lo, imag_hi),
            "leaves": [cell_record(x) for x in sorted(LEAVES.values(), key=lambda c: c["id"])]}


def save():
    with protected():
        REPORT["elapsed_seconds"] = time.monotonic() - START
        REPORT["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if ACTIVE is not None:
            REPORT["active_panel"] = current_panel_record()
        REPORT['attempted_children_registry'] = ATTEMPTED
        REPORT['unfinished_children'] = [cell_record(c) for c in PENDING.values()]
        REPORT['outcome_journal'] = dict(path=str(JOURNAL_PATH), records=JOURNAL_COUNT)
        temporary = OUTPUT.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(REPORT, indent=2) + "\n")
        temporary.replace(OUTPUT)


def tail_rectangle(radius, label):
    upper = finite(radius, "nonfinite_tail_" + label).upper()
    if not upper >= 0:
        raise NonanalyticImage("negative_tail_" + label, upper)
    previous = TAIL_MAXIMA.get(label)
    if previous is None or upper > previous:
        TAIL_MAXIMA[label] = upper
        REPORT["tail_radius_maxima"][label] = interval(upper)
    return acb(arb(0, upper), arb(0, upper))


def frobenius(x):
    rx = modulus(x, arb(1) / 4, "frobenius_x")
    z, zp = polynomial_and_derivative(COEFFICIENTS, x)
    n, np = polynomial_and_derivative(NUMERATOR_COEFFICIENTS, x)
    ztail = rx ** 64 / (1 - rx)
    dtail = rx ** 63 * (64 / (1 - rx) + rx / (1 - rx) ** 2)
    z += tail_rectangle(ztail, "node_Z")
    zp += tail_rectangle(dtail, "node_Zprime")
    n += tail_rectangle(2 * ztail, "node_N")
    np += tail_rectangle(2 * dtail, "node_Nprime")
    denominator(z)
    return n / z, (np * z - n * zp) / (z * z)


def geometry(z, t, triangle, upper, node):
    d = finite((-z * LOG2).exp(), "nonfinite_d")
    r = 1 - d if upper else d
    theta, phi = ((C * r, C * r * t) if triangle == 0 else (C * r * t, C * r))
    u = SQRT2 * (theta / 2).cos()
    v = SQRT2 * (theta / 2).sin()
    mass = phi.sin()
    y = mass * mass
    u2, q = u * u + y, v * v + y
    U, V = positive_sqrt(u2), positive_sqrt(q)
    product, total = denominator(U * V), denominator(U + V)
    if upper:
        distance = d if triangle == 0 else (1 - t) + d * t
        cosine = (C * distance).sin()
    else:
        cosine = theta.cos()
    if node:
        m = 4 * product / (total * total)
        p = 1 - m
    else:
        k = 2 * cosine / (total * total)
        p, m = k * k, 1 - k * k
    positive_real(p, "p")
    positive_real(m, "m")
    hchi = (1 / denominator(U + u) + 1 / denominator(V + v)) * (U + V + u + v) / (4 * product)
    root_hchi = positive_sqrt(hchi)
    chi = y * hchi
    one_chi = positive_real(1 - chi, "one_minus_chi")
    one_mchi = positive_real(1 - m * chi, "one_minus_mchi")
    rf = finite(acb.elliptic_rf(one_chi, one_mchi, acb(1)), "nonfinite_RF")
    rd = finite(acb.elliptic_rd(one_chi, acb(1), one_mchi), "nonfinite_incomplete_RD")
    km = elliptic_k(m)
    qchi_y = ((u + v) / 2) ** 2 * (u * (v / V) + y / V) / U ** 3
    # All derivatives in E20 are at fixed chain u,v. Preserve p factors.
    phase_numerator = mass * root_hchi * rf
    return {"u": u, "v": v, "y": y, "d": d, "r": r, "q": q, "u2": u2, "U": U, "V": V,
            "product": product, "p": p, "m": m, "chi": chi,
            "one_chi": one_chi, "one_mchi": one_mchi, "rf": rf, "rd": rd,
            "km": km, "qchi_y": qchi_y, "phase_numerator": phase_numerator,
            "cosine": cosine}


def node_qzy(g):
    U, V = g['U'], g['V']
    rf, rd, km, chi, p = g['rf'], g['rd'], g['km'], g['chi'], g['p']
    qm_y = 2 * p * V / U
    rd_p = finite(acb.elliptic_rd(acb(0), acb(1), p), "nonfinite_complete_RD_p")
    qzy = (rf * g["qchi_y"] / (positive_sqrt(g["one_chi"]) * positive_sqrt(g["one_mchi"]))
           + chi ** 2 * rf * rd * qm_y / 3
           - chi * rf ** 2 * qm_y * rd_p / denominator(3 * km)) / denominator(km ** 2)
    return qzy


def node_value(g):
    U, V, q, u2 = g["U"], g["V"], g["q"], g["u2"]
    rf, rd, km, chi, p = g["rf"], g["rd"], g["km"], g["chi"], g["p"]
    qzy = node_qzy(g)
    eta = g["phase_numerator"] / denominator(km)
    x, xy = q / u2, 2 * g["cosine"] / u2 ** 2
    df, dfp = frobenius(x)
    tvalue = (2 / PI) * (LOG4 - logarithm(x, True) / 2 - df)
    if not finite(tvalue).real > arb(1) / 2:
        raise NonanalyticImage("T_real_guard", tvalue)
    exp2d = finite((2 * df).exp(), "nonfinite_exp_2D")
    w = x * exp2d / 16
    rw = modulus(w, arb(1) / 8, "poisson_w")
    mu = finite((2 * arb.pi() * abs(eta.imag).upper()).exp(), "nonfinite_phase_majorant").upper()
    mixed = rw * mu
    if not mixed <= arb(1) / 4:
        raise NonanalyticImage("mixed_phase_nome_guard", mixed, eta)
    w2, power, even_power = w * w, acb(1), w * w
    sum_q, sum_e = acb(0), acb(0)
    for n in range(1, 65):
        inv = 1 / denominator(1 - even_power)
        bn, cn = 2 * power * inv, (1 + even_power) * inv
        qn = 4 * PI * bn * cn - 2 * PI ** 2 * tvalue * n * (bn + 2 * w2 * bn ** 3)
        en = 2 * PI * tvalue * bn * cn - 2 * bn / n
        phase = 2 * PI * n * eta
        sum_q += qn * phase.cos()
        sum_e += n * n * en * phase.sinc()
        power *= w
        even_power *= w2
    j, rem = 65, rw ** 130
    denom = 1 - rem
    c0 = 2 * (1 + rem) / denom ** 2
    c1 = 2 / denom + 16 * rem / denom ** 3
    cb = 2 / denom
    b = 1 - mixed
    l0 = mixed ** (j - 1) / b
    l1 = mixed ** (j - 1) * (j / b + mixed / b ** 2)
    l2 = mixed ** (j - 1) * (j * j / b + 2 * j * mixed / b ** 2 + mixed * (1 + mixed) / b ** 3)
    sum_q += tail_rectangle(mu * (4 * arb.pi() * c0 * l0 + 2 * arb.pi() ** 2 * abs(tvalue).upper() * c1 * l1), "node_Qphase")
    sum_e += tail_rectangle(2 * mu * (arb.pi() * abs(tvalue).upper() * c0 * l2 + cb * l1), "node_Ephase")
    return (1 / (3 * u2) + xy * (-2 * dfp / 3 - exp2d * (1 + 2 * x * dfp) * sum_q / (16 * PI))
            - 2 * PI ** 2 * exp2d * qzy * sum_e / (16 * u2))


def direct_delta2_y(g, kp):
    p, m, product = g['p'], g['m'], g['product']
    rf, rd, chi = g['rf'], g['rd'], g['chi']
    B = chi * rf ** 2
    by = (rf * (g["qchi_y"] / g["q"]) / (positive_sqrt(g["one_chi"]) * positive_sqrt(g["one_mchi"]))
          + chi ** 2 * rf * rd * (2 * p / product) / 3)
    rd_m = finite(acb.elliptic_rd(acb(0), acb(1), m), "nonfinite_complete_RD_m")
    delta2_y = PI ** 2 * (by + 2 * p * B * rd_m / denominator(3 * product * kp)) / kp ** 2
    return delta2_y


def direct_pair(a, power, delta):
    ch, sh, shc = delta.cosh(), delta.sinh(), (acb(0, 1) * delta).sinc()
    dj = denominator(1 + power ** 2 + 2 * power * ch)
    aj = 2 * power * (-a * (2 * power + (1 + power ** 2) * ch)
                     + delta * (1 - power ** 2) * sh) / dj ** 2
    bj = 2 * power * (a * (1 - power ** 2) * shc
                     - (1 + power ** 2) * ch - 2 * power) / dj ** 2
    return dj, aj, bj


def direct_value(g):
    p, m, product = g["p"], g["m"], g["product"]
    rf, rd, chi = g["rf"], g["rd"], g["chi"]
    kp = denominator(elliptic_k(p))
    epsilon = PI * g["km"] / kp
    if not finite(epsilon).real > 2:
        raise NonanalyticImage("epsilon_real_guard", epsilon)
    epsilon_y = PI ** 2 / denominator(2 * m * kp ** 2 * product)
    delta = PI * g["phase_numerator"] / kp
    delta2_y = direct_delta2_y(g, kp)
    ch, sh, shc = delta.cosh(), delta.sinh(), (acb(0, 1) * delta).sinc()
    z = finite((-epsilon).exp(), "nonfinite_direct_nome")
    rz = modulus(z, arb(-2).exp(), "direct_nome", strict=True)
    power, tepsilon = z, acb(0)
    zdelta = -1 / denominator(4 * (delta / 2).cosh() ** 2)
    for j in range(1, 64):
        a = j * epsilon
        dj, aj, bj = direct_pair(a, power, delta)
        tepsilon += j * aj
        zdelta += bj
        power *= z
    j, rem = 64, rz ** 64
    eb = abs(delta).upper()
    E = finite(eb.exp(), "nonfinite_direct_phase_majorant")
    de = 1 - E * rem
    if not de > 0:
        raise NonanalyticImage("direct_tail_denominator_guard", de, delta)
    b = 1 - rz
    s0 = rem / b
    s1 = rem * (j / b + rz / b ** 2)
    s2 = rem * (j * j / b + 2 * j * rz / b ** 2 + rz * (1 + rz) / b ** 3)
    epb = abs(epsilon).upper()
    tepsilon += tail_rectangle(2 * E * (epb * s2 + eb * s1) / de ** 2, "direct_Tepsilon")
    zdelta += tail_rectangle(2 * E * (s0 / de ** 2 + (1 + E * rem) * (epb * s1 + eb * s0) / de ** 3), "direct_Zdelta")
    return 2 * epsilon_y * tepsilon + delta2_y * zdelta + 1 / denominator(3 * g["q"])


def evaluate(z, t, panel, kind, cell, publish=None, node_index=None):
    context = dict(panel=panel['index'], cell=cell['id'], kind=kind,
                   z=complex_interval(z), t=complex_interval(t), node_index=node_index)
    def kernel():
        g = geometry(z, t, panel['triangle'], panel['upper'], panel['node'])
        value = node_value(g) if panel['node'] else direct_value(g)
        return finite(LOG2 * g['r'] * g['d'] * value / SQRT2)
    return attempt(context, kernel, publish)


def real_bound(bounds, panel):
    loz, hiz, lot, hit = map(point, bounds)
    dmax, dmin = (-(loz * arb(2).log())).exp(), (-(hiz * arb(2).log())).exp()
    rmin, rmax = (1 - dmax, 1 - dmin) if panel["upper"] else (dmin, dmax)
    rmin = rmin.lower()
    if not rmin > 0:
        raise ArithmeticError("positive real-cell radial lower bound was lost")
    majorant = 1000 * (2 + arb(2).log() - 2 * rmin.log()) ** 2
    weight = arb(2).log() * rmax.upper() * dmax.upper() / arb(2).sqrt()
    radius = (majorant * weight * (hiz - loz) * (hit - lot)).upper()
    return acb(arb(0, radius), 0), radius


def axis_for_cell(cell, panel):
    bounds = cell["bounds"]
    cz = (bounds[0] + bounds[1]) / 2
    ct = (bounds[2] + bounds[3]) / 2
    hz, ht = (bounds[1] - bounds[0]) / 2, (bounds[3] - bounds[2]) / 2
    probes = []
    for axis in (0, 1):
        z = acb(arb(point(cz), 4 * point(hz)), arb(0, 4 * point(hz))) if axis == 0 else acb(point(cz))
        t = acb(arb(point(ct), 4 * point(ht)), arb(0, 4 * point(ht))) if axis == 1 else acb(point(ct))
        try:
            def publish_probe(value, ref):
                cell.setdefault('probe_refs', []).append(ref)
            value = evaluate(z, t, panel, "single_axis_" + str(axis), cell, publish_probe)
            probes.append((True, float((value.real.rad() + value.imag.rad()).upper())))
        except NonanalyticImage:
            probes.append((False, math.inf))
    if probes[0][0] != probes[1][0]:
        return 1 if probes[0][0] else 0
    if probes[0][0] and probes[0][1] != probes[1][1]:
        return 0 if probes[0][1] > probes[1][1] else 1
    if panel["upper"] and panel["triangle"] == 1:
        d = math.exp(-float(cz) * math.log(2))
        zchange = float(hz) * d * float(ct) * math.log(2)
        tchange = float(ht) * (1 - d)
        preferred = 0 if zchange >= tchange else 1
        # A periodic other-axis split guarantees fair progress on double failures.
        return 1 - preferred if sum(cell["depths"]) % 4 == 3 else preferred
    return sum(cell["depths"]) % 2


def new_cell(bounds, depths, panel, parent=None, selected=False):
    global NEXT_ID
    with protected():
        if REPORT['counts']['created_cells'] >= 10000:
            raise ArithmeticError('the agreed created-cell cap was reached')
        NEXT_ID += 1
        REPORT['counts']['created_cells'] += 1
        fallback, radius = real_bound(bounds, panel)
        cell = dict(id=NEXT_ID, bounds=bounds, depths=depths, value=fallback,
                    method='signed_real_bound', real_bound_radius=radius,
                    completed_node_indices=[], node_refs=[], weighted_accumulator=acb(0),
                    partial_rule_status='not_started')
        ref = journal(dict(event='cell_created', panel=panel['index'], parent=parent,
                           selected=selected, cell=cell_record(cell)))
        ATTEMPTED[str(NEXT_ID)] = dict(panel=panel['index'], parent=parent,
                                     status='created', creation_ref=ref)
        if selected:
            LEAVES[cell['id']] = cell
        else:
            PENDING[cell['id']] = cell
        return cell


def refine_cell(cell, panel):
    bounds = cell['bounds']
    cz, ct = (bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2
    hz, ht = (bounds[1] - bounds[0]) / 2, (bounds[3] - bounds[2]) / 2
    zball = acb(arb(point(cz), 4 * point(hz)), arb(0, 4 * point(hz)))
    tball = acb(arb(point(ct), 4 * point(ht)), arb(0, 4 * point(ht)))
    try:
        def publish_joint(image, ref):
            M = finite(abs(image)).upper()
            error = (arb(16) / 3 * arb(4) ** -16 * 4 * point(hz) * point(ht) * M).upper()
            cell.update(joint_ref=ref, analytic_M=M, quadrature_error=error,
                        partial_rule_status='in_progress')
        evaluate(zball, tball, panel, 'joint_analytic_box', cell, publish_joint)
        for ix, (x, wx) in enumerate(NODES):
            z = acb(point(cz) + point(hz) * x)
            for iy, (y, wy) in enumerate(NODES):
                t = acb(point(ct) + point(ht) * y)
                index = 8 * ix + iy
                def publish_node(value, ref):
                    cell['weighted_accumulator'] += wx * wy * value
                    cell['completed_node_indices'].append(index)
                    cell['node_refs'].append(ref)
                evaluate(z, t, panel, 'gauss_node', cell, publish_node, index)
        with protected():
            error = cell['quadrature_error']
            value = point(hz) * point(ht) * cell['weighted_accumulator'] + acb(arb(0, error), arb(0, error))
            finite(value)
            cell['partial_rule_status'] = 'complete_64_nodes'
            if value.real.upper() - value.real.lower() < 2 * cell['real_bound_radius']:
                cell.update(value=value, method='tensor_gauss')
    except NonanalyticImage as error:
        with protected():
            cell['guard_reason'] = error.reason
            cell['partial_rule_status'] = 'refused'
    with protected():
        ref = journal(dict(event='cell_finished', panel=panel['index'], cell=cell_record(cell)))
        ATTEMPTED[str(cell['id'])].update(status='finished', completion_ref=ref)
    return cell


def width_fraction(cell):
    return rational(cell["value"].real.upper()) - rational(cell["value"].real.lower())


def import_rule():
    raw = RULE_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RULE_SHA:
        raise ValueError('Accepted eight-point rule source changed')
    saved = json.loads(raw)['gauss_nodes_and_weights']
    def imported(record):
        def fraction(end):
            v = record[end + '_exact']
            return Fraction(int(v['numerator']), int(v['denominator']))
        lo, hi = fraction('lower'), fraction('upper')
        value = arb(point((lo + hi) / 2), point((hi - lo) / 2).upper())
        if not value.contains(point(lo)) or not value.contains(point(hi)):
            raise ArithmeticError('Imported outward rule enclosure lost containment')
        return value
    if len(saved) != 8:
        raise ValueError('Expected accepted eight-point rule')
    nodes = [(imported(row['node']), imported(row['weight'])) for row in saved]
    REPORT['gauss_rule_import'] = dict(path=str(RULE_PATH), sha256=RULE_SHA,
        field='gauss_nodes_and_weights', order='preserved', regenerated=False)
    REPORT['gauss_nodes_and_weights'] = [dict(node=interval(x), weight=interval(w)) for x, w in nodes]
    return nodes


def known_answers():
    REPORT['new_derivative_known_answers'] = []
    for node, z in ((True, 3), (False, 1)):
        context = dict(kind='new_derivative_known_answer', triangle=0, t='0',
                       z=str(z), chart='node' if node else 'direct_lower')
        def fixture():
            g = geometry(acb(z), acb(0), 0, False, node)
            u, v = g['u'], g['v']
            a = (u + v) / 2
            kp = elliptic_k(g['p'])
            eta = g['phase_numerator'] / g['km']
            delta = PI * g['phase_numerator'] / kp
            zero = {'chi': g['chi'], 'eta': eta, 'delta': delta}
            differences = {'RF_minus_one': g['rf'] - 1,
                'q_chiY': g['qchi_y'] - a*a/(u*u),
                'q_ZY': node_qzy(g) - a*a/(u*u*g['km']**2),
                'delta2_Y': direct_delta2_y(g, kp) - PI**2*a*a/(u*u*v*v*kp**2),
                'sinc_zero': acb(0).sinc() - 1,
                'sinhc_zero': (acb(0, 1)*acb(0)).sinc() - 1,
                'central': -1/(4*(delta/2).cosh()**2) + acb(1)/4}
            pairs = []
            epsilon = PI * g['km'] / kp
            for j in range(1, 64):
                aj = j * epsilon
                x = (-aj).exp()
                D, A, B = direct_pair(aj, x, delta)
                diffs = dict(D=D-(1+x)**2, A=A+2*aj*x/(1+x)**2,
                    B=B-2*x/(1+x)**2*(aj*(1-x)/(1+x)-1))
                pairs.append(dict(j=j, a=complex_interval(aj), x=complex_interval(x),
                    D=complex_interval(D), A=complex_interval(A), B=complex_interval(B),
                    differences={k:complex_interval(v) for k,v in diffs.items()}))
                differences.update({str(j)+'_'+k:v for k,v in diffs.items()})
            passed = all(value.real.lower()==0 and value.real.upper()==0 and value.imag.lower()==0 and value.imag.upper()==0 for value in zero.values())
            passed = passed and all(value.real.contains(0) and value.imag.contains(0) for value in differences.values())
            return dict(context=context, pass_fixture=bool(passed),
                geometry={k:complex_interval(v) for k,v in g.items()},
                zero_phases={k:complex_interval(v) for k,v in zero.items()},
                differences={k:complex_interval(v) for k,v in differences.items()}, pairs=pairs)
        def publish(value, ref):
            REPORT['new_derivative_known_answers'].append(dict(journal_ref=ref, **value))
        result = attempt(context, fixture, publish)
        if not result['pass_fixture']:
            raise ArithmeticError('Required new derivative known answer failed')


def main():
    global arb, acb, PI, LOG2, LOG4, SQRT2, C, COEFFICIENTS, NUMERATOR_COEFFICIENTS
    global TAIL_MAXIMA, NODES, ACTIVE, LEAVES, JOURNAL
    with OUTPUT.open("x") as stream:
        json.dump(REPORT, stream)
    JOURNAL = JOURNAL_PATH.open("x")
    signal.signal(signal.SIGALRM, deadline)
    signal.signal(signal.SIGXCPU, deadline)
    signal.alarm(295)
    resource.setrlimit(resource.RLIMIT_AS, (805306368, 805306368))
    resource.setrlimit(resource.RLIMIT_CPU, (285, 300))
    import flint
    from flint import arb, acb, ctx
    ctx.prec, ctx.threads = 192, 1
    PI, LOG2, LOG4, SQRT2 = acb.pi(), acb(2).log(), acb(4).log(), acb(2).sqrt()
    C = PI / 2
    TAIL_MAXIMA = {}
    REPORT["implementation"] = {"python": sys.version.split()[0], "python_flint": flint.__version__,
                                "box": "compute-host", "pid": os.getpid(),
                                "nice_actual": os.getpriority(os.PRIO_PROCESS, 0),
                                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    COEFFICIENTS, bs = [arb(1)], [arb(0)]
    for n in range(63):
        COEFFICIENTS.append(COEFFICIENTS[-1] * (2 * n + 1) ** 2 / (4 * (n + 1) ** 2))
        bs.append(bs[-1] + arb(1) / ((2 * n + 1) * (n + 1)))
    NUMERATOR_COEFFICIENTS = [a * b for a, b in zip(COEFFICIENTS, bs)]
    NODES = import_rule()
    enode = 4 * arb(2) ** -40
    bnode = 2 - enode.log()
    onode = 200 * enode * (bnode ** 2 + 2 * bnode + 2)
    oouter = 25000 * arb(2).sqrt() * arb(2) ** -32
    omission = (onode + oouter).upper()
    REPORT["omitted_integral_bounds"] = {"node": interval(onode), "outer": interval(oouter),
                                         "signed_absolute_total": interval(omission)}
    if not omission < arb('9.1e-6'):
        raise ArithmeticError('E20 signed omission exceeds required budget')
    REPORT['omission_budget_check'] = dict(pass_check=True, omission_limit='9.1e-6',
        panels=6, per_panel_full_width='2e-5', signed_omission_multiplier=2,
        total_full_width_budget=interval(6*arb('2e-5')+2*omission), final_full_width_limit='2e-4')
    save()
    known_answers()
    save()
    completed = []
    for triangle in (0, 1):
        for mode, zlo, zhi in ((0, 2, 20), (1, 1, 2), (2, 1, 32)):
            panel = {"index": 3 * triangle + mode, "triangle": triangle,
                     "upper": mode == 2, "node": mode == 0,
                     "initial_bounds_z_t": [str(zlo), str(zhi), "0", "1"]}
            with protected():
                ACTIVE, LEAVES = panel, {}
                initial = new_cell(tuple(map(Fraction, (zlo, zhi, 0, 1))), (0, 0), panel, selected=True)
            save()  # complete signed initial cover precedes ANY integrand call
            refine_cell(initial, panel)
            heap = [(-float(width_fraction(initial)), initial["id"])]
            lo = rational(initial["value"].real.lower())
            hi = rational(initial["value"].real.upper())
            il = rational(initial["value"].imag.lower())
            ih = rational(initial["value"].imag.upper())
            save()
            while hi - lo > Fraction(1, 50000) or not il <= 0 <= ih:
                _, selected_id = heapq.heappop(heap)
                parent = LEAVES[selected_id]
                axis = axis_for_cell(parent, panel)
                a, b, c, d = parent["bounds"]
                depths = tuple(v + (i == axis) for i, v in enumerate(parent["depths"]))
                if axis == 0:
                    mid = (a + b) / 2
                    bounds1, bounds2 = (a, mid, c, d), (mid, b, c, d)
                else:
                    mid = (c + d) / 2
                    bounds1, bounds2 = (a, b, c, mid), (a, b, mid, d)
                # Persistent creation/finish records retain both attempted children;
                # selected parent remains the cover until both refinements finish.
                child1 = new_cell(bounds1, depths, panel, parent=selected_id)
                refine_cell(child1, panel)
                child2 = new_cell(bounds2, depths, panel, parent=selected_id)
                refine_cell(child2, panel)
                with protected():
                    children = (child1, child2)
                    pv = parent['value']
                    lo -= rational(pv.real.lower())
                    hi -= rational(pv.real.upper())
                    il -= rational(pv.imag.lower())
                    ih -= rational(pv.imag.upper())
                    replacement = LEAVES.copy()
                    del replacement[selected_id]
                    for child in children:
                        cv = child['value']
                        lo += rational(cv.real.lower())
                        hi += rational(cv.real.upper())
                        il += rational(cv.imag.lower())
                        ih += rational(cv.imag.upper())
                        replacement[child['id']] = child
                        heapq.heappush(heap, (-float(width_fraction(child)), child['id']))
                    adoption = journal(dict(event='complete_cover_adoption', panel=panel['index'],
                        parent=selected_id, children=[child['id'] for child in children]))
                    LEAVES = replacement
                    ATTEMPTED[str(selected_id)].update(status='replaced', adoption_ref=adoption)
                    for child in children:
                        ATTEMPTED[str(child['id'])].update(status='selected', adoption_ref=adoption)
                        del PENDING[child['id']]
                if REPORT["counts"]["created_cells"] % 32 in (0, 1):
                    save()
            with protected():
                record = current_panel_record(accepted=True)
                REPORT['panels'].append(record)
                completed.append((lo, hi))
                ACTIVE, LEAVES = None, {}
                REPORT.pop('active_panel', None)
            save()
    # Convert the exact summed endpoints outward at the SAME192-bit precision.
    lo = point(sum((p[0] for p in completed), Fraction(0))).lower()
    hi = point(sum((p[1] for p in completed), Fraction(0))).upper()
    shift = -((arb(32) / 9).log() + 1) / (3 * arb.pi())
    final_lo, final_hi = (lo - omission + shift).lower(), (hi + omission + shift).upper()
    if not (final_hi - final_lo <= arb("2e-4")):
        raise ArithmeticError("final b110 interval failed the prescribed width test")
    REPORT["results"]["b110"] = rational_interval(rational(final_lo), rational(final_hi))
    REPORT["status"] = "complete"
    save()


if __name__ == "__main__":
    if OUTPUT.exists() or JOURNAL_PATH.exists():
        raise SystemExit('Refusing to replace existing products or repeat this check')
    try:
        main()
    except Exception as error:
        REPORT['status'] = 'incomplete'
        REPORT['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        # Stop handled alarms and leave CPU save grace inside unchanged hard cap.
        signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
        signal.alarm(0)
        signal.signal(signal.SIGALRM, signal.SIG_IGN)
        signal.signal(signal.SIGXCPU, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
        if JOURNAL is not None:
            JOURNAL.flush()
            JOURNAL.close()
            h = hashlib.sha256()
            with JOURNAL_PATH.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024), b''):
                    h.update(block)
            REPORT['terminal_journal'] = dict(path=str(JOURNAL_PATH), records=JOURNAL_COUNT,
                size=JOURNAL_PATH.stat().st_size, sha256=h.hexdigest())
        save()
    print(json.dumps(dict(status=REPORT['status'], elapsed_seconds=REPORT.get('elapsed_seconds'),
                          completed_panels=len(REPORT['panels']), counts=REPORT['counts'])))
    sys.exit(0 if REPORT['status'] == 'complete' else 1)
