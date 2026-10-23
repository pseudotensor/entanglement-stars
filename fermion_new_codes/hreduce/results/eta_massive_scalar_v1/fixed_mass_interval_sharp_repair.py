"""Necessary fixed-y=1/25 enclosure repair, canonical V2.10, bounded box CPU.

Serves insert I6a and item 7 light-cut ratio. No finite-part calculation,
axial rerun, native matrix or evolved history is evaluated.
"""
from fractions import Fraction
import heapq
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time

OUTPUT = Path(__file__).with_name("fixed_mass_sharp_repair_result.json")
START = time.monotonic()
REPORT = {
    "status": "running", "science_specification": "ETA_massive_directions_v2.tex V2.5, V2.8 and necessary repair V2.10",
    "purpose": "insert I6a; item 7 light-cut ratio", "y": "1/25",
    "scope": "eta110 and eta111 at fixed mass, then support and conditional ratio",
    "resources": {"wall_seconds": 300, "internal_save_deadline_seconds": 295,
                  "address_space_bytes": 805306368, "threads": 1, "nice": 10, "cuda": "disabled"},
    "quadrature": {"bits": 192, "rule110": "8x8 Gauss-Legendre",
                   "complex_rectangle_halfwidth_factor": 4,
                   "analytic_error110": "(16/3)*4^-16*cell_area*M",
                   "cell_sum_width110": "1e-4", "final_direction_width": "2e-4",
                   "piece_width111": "2e-5", "evaluation_limit": 1000000,
                   "created_cell_limit110": 10000, "real_bound_fallback": "nonnegative S<7"},
    "finite_sums": {"central": "2*hF(delta)", "paired_terms": "j=1..31",
                    "tail_start": 32, "complex_tail": "V2.5a before density and Jacobian"},
    "counts": {"evaluations": 0, "created_cells": 0, "refused_images": 0,
               "refusals_by_reason": {}},
    "guard_samples": {}, "tail_radius_maxima": {}, "attempts111": [],
    "panels": [], "results": {}, "pieces111": [], "accepted_subcharts111": [],
    "enclosure_repair": {
        "sqrt": "V2.10a rectangle intersected with midpoint disk and finite raw principal image",
        "phase": "V2.9a C,D,U,V,k; V2.10b chi and principal chi root",
        "Kc": "V2.9 midpoint radius rho/(2a), intersected with finite raw image",
        "RF": "V2.9 midpoint radius (rho_x+rho_y)/(6a^(3/2)), intersected with finite raw image",
        "complex_box110": "V2.10c entire entropy modulus, including all j",
        "point_and_111": "unchanged central and j1..31 terms with J32 tail",
    },
}
LEAVES = {}
ACTIVE = None
NEXT_ID = 0
TAIL_MAXIMUM = None
CHART_TAIL_MAXIMUM = None

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
    center, rho = midpoint_and_radius(z)
    a, b, c = z.real.lower(), z.real.upper(), abs(z.imag).upper()
    point_value = finite(center.sqrt(), "nonfinite_midpoint_sqrt")
    midpoint_image = point_value + rectangle((rho/(2*a.sqrt())).upper())
    lower = a.sqrt().lower()
    upper = (((b*b+c*c).sqrt()+b)/2).sqrt().upper()
    imaginary = (c/(2*((a*a+c*c).sqrt()+a)).sqrt()).upper()
    elementary = acb(lower.union(upper), arb(0, imaginary))
    return intersect_images(elementary, midpoint_image, z.sqrt())


def rectangle(radius):
    radius = finite(radius, "nonfinite_enclosure_radius").upper()
    if not radius >= 0:
        raise ArithmeticError("negative proved enclosure radius")
    return acb(arb(0, radius), arb(0, radius))


def midpoint_and_radius(z):
    return (acb(z.real.mid(), z.imag.mid()),
            (z.real.rad()**2+z.imag.rad()**2).sqrt().upper())


def intersect_images(*images):
    # Each finite image contains the SAME principal analytic branch.
    result = None
    for image in images:
        if not image.is_finite():
            continue
        if result is None:
            result = image
            continue
        try:
            result = acb(result.real.intersection(image.real),
                         result.imag.intersection(image.imag))
        except ValueError as error:
            # A proved-image disagreement is an implementation fault, not
            # an analytic-domain refusal that adaptive subdivision may hide.
            raise ArithmeticError("empty intersection of proved analytic images") from error
    if result is None:
        raise NonanalyticImage("no_finite_proved_image", acb("nan"))
    return finite(result, "nonfinite_intersected_image")

def logarithm(z, analytic):
    finite(z, "nonfinite_logarithm_argument")
    if z.imag.contains(0) and not z.real > 0:
        raise NonanalyticImage("logarithm_cut", z)
    return finite(z.log(analytic=analytic), "nonfinite_logarithm_value")

def elliptic_k_complement(c):
    positive_real(c, "Kc_complement")
    center, rho = midpoint_and_radius(c)
    point_value = finite((1-center).elliptic_k(), "nonfinite_midpoint_K")
    error = (rho/(2*c.real.lower())).upper()
    return intersect_images(point_value + rectangle(error), (1-c).elliptic_k())


def local_rf(x, y):
    positive_real(x, "RF_x")
    positive_real(y, "RF_y")
    a = min(arb(1), x.real.lower(), y.real.lower())
    if not a > 0:
        raise NonanalyticImage("RF_midpoint_domain", a)
    x0, rx = midpoint_and_radius(x)
    y0, ry = midpoint_and_radius(y)
    point_value = finite(acb.elliptic_rf(x0, y0, acb(1)), "nonfinite_midpoint_RF")
    error = ((rx+ry)/(6*a*a.sqrt())).upper()
    return intersect_images(point_value + rectangle(error),
                            acb.elliptic_rf(x, y, acb(1)))

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
    REPORT["elapsed_seconds"] = time.monotonic() - START
    REPORT["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if ACTIVE is not None:
        REPORT["active_panel"] = current_panel_record()
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(REPORT, indent=2) + "\n")
    temporary.replace(OUTPUT)

def width_fraction(cell):
    return rational(cell["value"].real.upper()) - rational(cell["value"].real.lower())


def constants():
    global PI, LOG2, SQRT2, SQRT3
    PI, LOG2 = acb.pi(), acb(2).log()
    SQRT2, SQRT3 = acb(2).sqrt(), acb(3).sqrt()


def tail_rectangle(radius):
    global TAIL_MAXIMUM, CHART_TAIL_MAXIMUM
    radius = finite(radius, "nonfinite_entropy_tail").upper()
    if not radius >= 0:
        raise NonanalyticImage("negative_entropy_tail", radius)
    if TAIL_MAXIMUM is None or radius > TAIL_MAXIMUM:
        TAIL_MAXIMUM = radius
        REPORT["tail_radius_maxima"]["entropy_series"] = interval(radius)
    if "active_piece111" in REPORT and (CHART_TAIL_MAXIMUM is None or radius > CHART_TAIL_MAXIMUM):
        CHART_TAIL_MAXIMUM = radius
        REPORT["active_piece111"]["entropy_tail_maximum"] = interval(radius)
    return acb(arb(0, radius), arb(0, radius))


def hfermi(z, analytic):
    positive_real(z, "fermi_energy")
    ez = finite((-z).exp())
    base = positive_real(1 + ez, "fermi_log_argument")
    return logarithm(base, analytic) + z * ez / denominator(base)


def entropy(C, D, Y, uv, analytic, modulus=False):
    positive_real(Y, "mass_squared")
    U, V = positive_sqrt(C+D), positive_sqrt(C-D)
    denominator(U+V)
    W = denominator(U*V)
    k = D/denominator(C+W)
    p = k*k
    m = 1 - p
    positive_real(p, "p")
    positive_real(m, "m")
    kp = denominator(elliptic_k_complement(m))
    km = elliptic_k_complement(p)
    epsilon = finite(PI * km / kp)
    if not epsilon.real > arb(3)/2:
        raise NonanalyticImage("epsilon_real_guard", epsilon)
    chi = positive_real((1+(Y-uv)/W)/2, "chi")
    one_chi = positive_real(1-chi, "one_minus_chi")
    one_mchi = positive_real(1-m*chi, "one_minus_mchi")
    rf = local_rf(one_chi, one_mchi)
    delta = positive_real(PI*positive_sqrt(chi)*rf/kp, "delta")
    positive_real(epsilon-delta, "epsilon_minus_delta")
    if modulus:
        a, d = epsilon.real.lower(), delta.real.lower()
        A, e = abs(epsilon).upper(), abs(delta).upper()
        E, r = delta.real.upper().exp().upper(), (-a).exp().upper()
        if not (1-E*r > 0 and 1-r > 0):
            raise NonanalyticImage("entropy_modulus_geometric_margin", 1-E*r)
        rd = (-d).exp().upper()
        if not 1-rd > 0:
            raise NonanalyticImage("entropy_modulus_delta_margin", 1-rd)
        return finite((2*rd*(1+e)/(1-rd)
            +4*E*((1+e)*r/(1-r)+A*r/(1-r)**2)/(1-E*r)).upper(),
            "nonfinite_exact_entropy_modulus")
    value = 2*hfermi(delta, analytic)
    for j in range(1, 32):
        value += 2*(hfermi(j*epsilon+delta, analytic) + hfermi(j*epsilon-delta, analytic))
    r = finite(abs((-epsilon).exp())).upper()
    if not (r >= 0 and r < (arb(-3)/2).exp()):
        raise NonanalyticImage("entropy_nome_guard", r)
    e, E = abs(delta).upper(), abs(delta.real).upper().exp()
    rem = r**32
    dtail = 1-E*rem
    if not dtail > 0:
        raise NonanalyticImage("entropy_tail_denominator", dtail)
    s0 = rem/(1-r)
    s1 = rem*(arb(32)/(1-r) + r/(1-r)**2)
    tail = 4*E*((1+e)*s0 + abs(epsilon).upper()*s1)/dtail
    return finite(value + tail_rectangle(tail))


def evaluate(z, t, panel, kind, analytic=True, modulus=False):
    counts = REPORT["counts"]
    if counts["evaluations"] >= 1000000:
        raise ArithmeticError("the agreed full-integrand evaluation cap was reached")
    counts["evaluations"] += 1
    try:
        x = finite((-z*LOG2).exp())
        if panel["direction"] == "110":
            phi = PI/2*t
            Y = acb(1)/25 + phi.sin()**2
            C, D = 1+Y, (PI/2*x).sin()
            uv = (PI/2*x).cos()
            s = entropy(C, D, Y, uv, analytic, modulus=modulus)
            if modulus:
                result = (s*abs(LOG2*x/SQRT2).upper()).upper()
            else:
                result = s*LOG2*x/SQRT2
        else:
            if modulus:
                raise ArithmeticError("the entire-entropy modulus is only for110 Cauchy boxes")
            piece = panel["piece"]
            q = x if piece == 0 else 1-x if piece == 1 else 1+x if piece == 2 else acb(3)/2-x
            u2, v2 = 3-q, q
            u, v = positive_sqrt(u2), positive_sqrt(v2)
            Y = acb(1)/25
            C, D = acb(3)/2+Y, x if piece == 3 else acb(3)/2-q
            s = entropy(C, D, Y, u*v, analytic)
            if piece == 0:
                c = (1-q)**3*(3-q)/denominator(3-2*q)
                positive_real(c, "density_complement")
                rho = 2*positive_sqrt(1-2*q/3)*elliptic_k_complement(c)/PI**2
            elif piece == 1:
                c = x**3*(2+x)/denominator(1+2*x)
                positive_real(c, "density_complement")
                rho = 2*positive_sqrt(1+2*x)*elliptic_k_complement(c)/(SQRT3*PI**2)
            elif piece == 2:
                rad = (1-x)**3*(1+x)
                c = x**3*(2-x)/denominator(rad)
                positive_real(c, "density_complement")
                rho = 2*(1-2*x)*elliptic_k_complement(c)/denominator(SQRT3*PI**2*positive_sqrt(rad))
            else:
                d3 = (1+2*x)**3*(3-2*x)
                c = (1-2*x)**3*(3+2*x)/denominator(d3)
                positive_real(c, "density_complement")
                rho = 16*x*elliptic_k_complement(c)/denominator(SQRT3*PI**2*positive_sqrt(d3))
            result = rho*s*LOG2*x
        return finite(result)
    except NonanalyticImage as error:
        counts["refused_images"] += 1
        reasons = counts["refusals_by_reason"]
        reasons[error.reason] = reasons.get(error.reason, 0)+1
        if error.reason not in REPORT["guard_samples"]:
            REPORT["guard_samples"][error.reason] = {
                "direction": panel["direction"], "piece": panel.get("piece"),
                "subchart": panel.get("subchart"),
                "kind": kind, "analytic": bool(analytic), "z": str(z),
                "t": None if t is None else str(t), "tested_image": error.image,
                "tested_input": error.input_image}
        raise


def acceptable(value, width):
    return (value.is_finite() and value.imag.contains(0)
            and value.real.upper()-value.real.lower() <= arb(width))


def certify111(ctx):
    global CHART_TAIL_MAXIMUM
    pieces = []
    for piece in range(4):
        begin = 1 if piece < 2 else 2
        n = 4*(32-begin)
        local = []
        for chart in range(n):
            left = Fraction(begin)+Fraction(chart, 4)
            right = left+Fraction(1, 4)
            accepted = False
            for precision in (192, 256):
                ctx.prec = precision
                constants()
                CHART_TAIL_MAXIMUM = None
                calls = {"total": 0, "refused_images": 0, "refusals_by_reason": {}}
                REPORT["active_piece111"] = {
                    "piece": piece, "subchart": chart, "subchart_count": n,
                    "bits": precision, "z": [str(left), str(right)], "calls": calls}
                save()
                panel = {"direction": "111", "piece": piece, "subchart": chart}

                def integrand(z, analytic):
                    calls["total"] += 1
                    try:
                        return evaluate(z, None, panel, "acb_integral", analytic)
                    except NonanalyticImage as error:
                        calls["refused_images"] += 1
                        reasons = calls["refusals_by_reason"]
                        reasons[error.reason] = reasons.get(error.reason, 0)+1
                        return acb("nan")

                value = acb.integral(integrand, acb(point(left)), acb(point(right)),
                                     rel_tol=arb("1e-8"), abs_tol=arb("1e-8"),
                                     deg_limit=64, eval_limit=20000,
                                     depth_limit=2*precision, use_heap=True)
                valid = value.is_finite() and value.imag.contains(0)
                if valid:
                    lo, hi = rational(value.real.lower()), rational(value.real.upper())
                    valid = hi-lo <= Fraction(1, 50000*n)
                record = {**REPORT["active_piece111"], "queue_limit": 2*precision,
                          "width_limit_exact": str(Fraction(1, 50000*n)),
                          "accepted": bool(valid)}
                if value.is_finite():
                    record.update(real=interval(value.real), imaginary=interval(value.imag))
                else:
                    record["nonfinite"] = True
                REPORT["attempts111"].append(record)
                if valid:
                    local.append((lo, hi))
                    REPORT["accepted_subcharts111"].append({
                        "piece": piece, "subchart": chart,
                        "attempt_index": len(REPORT["attempts111"])-1})
                    accepted = True
                save()
                if accepted:
                    break
            if not accepted:
                REPORT["failure111"] = {
                    "kind": "local_analytic_or_width_failure", "piece": piece,
                    "subchart": chart, "z": [str(left), str(right)]}
                REPORT["pieces111"].append({
                    "piece": piece, "accepted": False, "partial": True,
                    "accepted_subcharts": len(local), "required_subcharts": n,
                    "partial_real_sum": rational_interval(
                        sum((p[0] for p in local), Fraction(0)),
                        sum((p[1] for p in local), Fraction(0)))})
                REPORT.pop("active_piece111", None)
                save()
                ctx.prec = 192
                constants()
                return False
        lo = sum((p[0] for p in local), Fraction(0))
        hi = sum((p[1] for p in local), Fraction(0))
        if hi-lo > Fraction(1, 50000):
            raise ArithmeticError("exact111 piece sum failed the unchanged width test")
        REPORT["pieces111"].append({
            "piece": piece, "accepted": True, "subchart_count": n,
            "real": rational_interval(lo, hi)})
        pieces.append((lo, hi))
        REPORT.pop("active_piece111", None)
        save()
    ctx.prec = 192
    constants()
    lo = sum((p[0] for p in pieces), Fraction(0))
    hi = sum((p[1] for p in pieces), Fraction(0))+rational(OMIT111)
    if hi-lo > Fraction(1, 5000):
        raise ArithmeticError("final eta111 width test failed")
    REPORT["results"]["eta111_y1_25"] = rational_interval(lo, hi)
    save()
    return True



def real_bound(bounds):
    a, b, c, d = map(point, bounds)
    radius = (7*(b-a)*(d-c)*arb(2).log()*(-a*arb(2).log()).exp()/arb(2).sqrt()).upper()
    return acb(arb(radius/2, radius/2)), radius


def base_cell(bounds, depths):
    global NEXT_ID
    if REPORT["counts"]["created_cells"] >= 10000:
        raise ArithmeticError("the agreed created-cell cap was reached")
    NEXT_ID += 1
    REPORT["counts"]["created_cells"] += 1
    value, radius = real_bound(bounds)
    return {"id": NEXT_ID, "bounds": bounds, "depths": depths,
            "value": value, "method": "nonnegative_real_bound", "real_bound_radius": radius}


def improve_cell(cell, panel):
    a, b, c, d = cell["bounds"]
    cz, ct, hz, ht = (a+b)/2, (c+d)/2, (b-a)/2, (d-c)/2
    zb = acb(arb(point(cz), 4*point(hz)), arb(0, 4*point(hz)))
    tb = acb(arb(point(ct), 4*point(ht)), arb(0, 4*point(ht)))
    try:
        image = evaluate(zb, tb, panel, "joint_analytic_box", modulus=True)
        M = finite(abs(image)).upper()
        error = (arb(16)/3*arb(4)**-16*4*point(hz)*point(ht)*M).upper()
        value = acb(0)
        for x, wx in NODES:
            z = acb(point(cz)+point(hz)*x)
            for y, wy in NODES:
                t = acb(point(ct)+point(ht)*y)
                value += wx*wy*evaluate(z, t, panel, "gauss_node")
        value = point(hz)*point(ht)*value + acb(arb(0, error), arb(0, error))
        if not value.is_finite() or not value.imag.contains(0):
            raise NonanalyticImage("nonreal_cell_integral", value)
        if value.real.upper()-value.real.lower() < cell["real_bound_radius"]:
            cell.update(value=value, method="tensor_gauss", analytic_M=M, quadrature_error=error)
    except NonanalyticImage as error:
        cell["guard_reason"] = error.reason
    return cell


def axis_for_cell(cell, panel):
    # Fairness is geometric, independent of all floating-point priorities.
    depths = cell["depths"]
    if sum(depths) % 4 == 3:
        return 0 if depths[0] <= depths[1] else 1
    a, b, c, d = cell["bounds"]
    cz, ct, hz, ht = (a+b)/2, (c+d)/2, (b-a)/2, (d-c)/2
    probes = []
    for axis in (0, 1):
        z = acb(arb(point(cz), 4*point(hz)), arb(0, 4*point(hz))) if axis == 0 else acb(point(cz))
        t = acb(arb(point(ct), 4*point(ht)), arb(0, 4*point(ht))) if axis == 1 else acb(point(ct))
        try:
            value = evaluate(z, t, panel, "single_axis_"+str(axis))
            probes.append((True, float((value.real.rad()+value.imag.rad()).upper())))
        except NonanalyticImage:
            probes.append((False, math.inf))
    if probes[0][0] != probes[1][0]:
        return 1 if probes[0][0] else 0
    if probes[0][0] and probes[0][1] != probes[1][1]:
        return 0 if probes[0][1] > probes[1][1] else 1
    return 0 if depths[0] <= depths[1] else 1


def certify110():
    global ACTIVE, LEAVES
    panel = {"direction": "110", "initial_bounds_z_t": ["0", "32", "0", "1"]}
    ACTIVE, LEAVES = panel, {}
    initial = base_cell(tuple(map(Fraction, (0, 32, 0, 1))), (0, 0))
    LEAVES[initial["id"]] = initial
    save()
    improve_cell(initial, panel)
    heap = [(-float(width_fraction(initial)), initial["id"])]
    lo, hi = rational(initial["value"].real.lower()), rational(initial["value"].real.upper())
    il, ih = rational(initial["value"].imag.lower()), rational(initial["value"].imag.upper())
    while hi-lo > Fraction(1, 10000) or not il <= 0 <= ih:
        _, selected_id = heapq.heappop(heap)
        parent = LEAVES[selected_id]
        axis = axis_for_cell(parent, panel)
        a, b, c, d = parent["bounds"]
        depths = tuple(v+(i == axis) for i, v in enumerate(parent["depths"]))
        if axis == 0:
            mid = (a+b)/2
            bounds1, bounds2 = (a, mid, c, d), (mid, b, c, d)
        else:
            mid = (c+d)/2
            bounds1, bounds2 = (a, b, c, mid), (a, b, mid, d)
        children = [improve_cell(base_cell(bounds1, depths), panel),
                    improve_cell(base_cell(bounds2, depths), panel)]
        pv = parent["value"]
        nlo, nhi = lo-rational(pv.real.lower()), hi-rational(pv.real.upper())
        nil, nih = il-rational(pv.imag.lower()), ih-rational(pv.imag.upper())
        replacement = LEAVES.copy()
        del replacement[selected_id]
        for child in children:
            cv = child["value"]
            nlo += rational(cv.real.lower())
            nhi += rational(cv.real.upper())
            nil += rational(cv.imag.lower())
            nih += rational(cv.imag.upper())
            replacement[child["id"]] = child
            heapq.heappush(heap, (-float(width_fraction(child)), child["id"]))
        # Until this atomic replacement, a timeout saves the valid parent.
        LEAVES = replacement
        lo, hi, il, ih = nlo, nhi, nil, nih
        if REPORT["counts"]["created_cells"] % 32 in (0, 1):
            save()
    REPORT["panels"].append(current_panel_record(accepted=True))
    hi += rational(OMIT110)
    if hi-lo > Fraction(1, 5000):
        raise ArithmeticError("final eta110 width test failed")
    REPORT["results"]["eta110_y1_25"] = rational_interval(lo, hi)
    ACTIVE, LEAVES = None, {}
    REPORT.pop("active_panel", None)
    save()


def load_endpoints(record):
    return tuple(Fraction(int(record[k]["numerator"]), int(record[k]["denominator"]))
                 for k in ("lower_exact", "upper_exact"))


def support_and_ratio():
    source = Path(__file__).with_name("stable_result.json")
    axial = json.loads(source.read_text())["results"]["eta100_y1_25"]
    alo, ahi = map(point, load_endpoints(axial))
    dlo, dhi = map(point, load_endpoints(REPORT["results"]["eta110_y1_25"]))
    tlo, thi = map(point, load_endpoints(REPORT["results"]["eta111_y1_25"]))
    pi, s2, s3 = arb.pi(), arb(2).sqrt(), arb(3).sqrt()
    theta, phi = (arb(2)/3).sqrt().acos(), (1/s3).acos()
    m1, m2, m3 = 3*s2*theta/pi, 3*s2*(phi-theta)/pi, arb(3)/2-3*s2*phi/pi
    candidates = (m1*alo, (m1+m2)*dlo/s2, s3*tlo/2)
    lower = max(x.lower() for x in candidates)
    upper = ((m1-m2)*ahi+s2*(m2-m3)*dhi+s3*m3*thi).upper()
    if not (lower > 0 and upper >= lower):
        raise ArithmeticError("support interval ordering or positivity failed")
    rlo = (arb("0.91905664456747")/upper).lower()
    rhi = (arb("0.91929747790884")/lower).upper()
    REPORT["inputs"] = {"axial_source": str(source), "eta100_y1_25": axial,
                        "massless_sphere": ["0.91905664456747", "0.91929747790884"],
                        "massless_premise": "previous directional-input error assumptions retained"}
    REPORT["support_weights"] = {"m1": interval(m1), "m2": interval(m2), "m3": interval(m3)}
    REPORT["results"]["eta_sphere_y1_25"] = rational_interval(rational(lower), rational(upper))
    REPORT["results"]["massless_to_massive_ratio"] = rational_interval(rational(rlo), rational(rhi))
    save()


def main():
    global arb, acb, OMIT111, OMIT110, NODES
    with OUTPUT.open("x") as stream:
        json.dump(REPORT, stream)
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(295)
    resource.setrlimit(resource.RLIMIT_AS, (805306368, 805306368))
    resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
    import flint
    from flint import arb, acb, ctx
    ctx.prec, ctx.threads = 192, 1
    constants()
    REPORT["implementation"] = {"python": sys.version.split()[0], "python_flint": flint.__version__,
                                "box": "compute-host", "pid": os.getpid(),
                                "nice_actual": os.getpriority(os.PRIO_PROCESS, 0),
                                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    cut = arb(2)**-32
    OMIT111 = (7*(2*cut+4*cut*(3-cut.log()))).upper()
    OMIT110 = (7*cut/arb(2).sqrt()).upper()
    REPORT["omitted_integral_bounds"] = {
        "cut": "2^-32", "eta111_nonnegative": interval(OMIT111),
        "eta110_nonnegative": interval(OMIT110)}
    save()
    complete111 = certify111(ctx)
    NODES = [arb.legendre_p_root(8, k, weight=True) for k in range(8)]
    for k, (x, w) in enumerate(NODES):
        if not (x.is_finite() and w.is_finite() and x > -1 and x < 1 and w > 0):
            raise ArithmeticError("Gauss node/weight certificate failed")
        if k and not NODES[k-1][0].lower() > x.upper():
            raise ArithmeticError("Gauss root intervals are not disjoint and decreasing")
    if not sum((w for x, w in NODES), arb(0)).contains(2):
        raise ArithmeticError("Gauss weight sum does not contain two")
    REPORT["gauss_nodes_and_weights"] = [{"node": interval(x), "weight": interval(w)} for x, w in NODES]
    save()
    certify110()
    if complete111:
        support_and_ratio()
        REPORT["status"] = "complete"
    else:
        REPORT["status"] = "incomplete"
        REPORT["error"] = "111 local width/analytic failure; independently accepted110 retained"
    save()


if __name__ == "__main__":
    if OUTPUT.exists():
        raise SystemExit("Refusing to replace an existing result or repeat this check")
    try:
        main()
    except Exception as error:
        REPORT["status"] = "incomplete"
        REPORT["error"] = type(error).__name__+": "+str(error)
        save()
    finally:
        signal.alarm(0)
    print(json.dumps({"status": REPORT["status"], "elapsed_seconds": REPORT.get("elapsed_seconds"),
                      "accepted111": sum(bool(x["accepted"]) for x in REPORT["attempts111"]),
                      "counts": REPORT["counts"]}))
    sys.exit(0 if REPORT["status"] == "complete" else 1)
