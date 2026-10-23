#!/usr/bin/env python3
"""Bounded C3.1 scalar calculation, C1--C5 and C2a--b.

One remote CPU process uses directed python-flint 0.9.0 balls.  Numerical
claims require the actual sign, width, derivative and independent controls;
requested Petras goals are not error certificates.  No ground/clock/COM
consumer or production parameter is chosen by this program.
"""

from __future__ import annotations

import argparse
from fractions import Fraction as F
import json
import os
from pathlib import Path
import sys
import time


SPEC = "solve/strand_C3.tex, C3.1 C1--C5 and C2a--b"
REVIEW = "solve/review_C3.tex"
U = 2**22
POOL_NAMES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")


class Unresolved(RuntimeError):
    pass


class EngineeringCap(Unresolved):
    pass


class Conflict(Unresolved):
    pass


def a(value=0):
    if isinstance(value, F):
        return arb(value.numerator) / value.denominator
    return arb(value)


def encode(value):
    if isinstance(value, acb):
        return {"real": encode(value.real), "imag": encode(value.imag)}
    if not value.is_finite():
        return {"status": "nonfinite_unresolved"}
    def endpoint(point):
        mantissa, exponent = point.man_exp()
        return {"mantissa": str(mantissa), "exponent2": int(exponent)}
    return {"lower_dyadic": endpoint(value.lower()), "upper_dyadic": endpoint(value.upper()),
            "display_ball": value.str(24)}


def hull(left, right):
    lo, hi = a(left).lower(), a(right).upper()
    if not lo.is_finite() or not hi.is_finite() or lo > hi:
        raise Unresolved("A required directed interval is nonfinite or unordered")
    return (lo + hi) / 2 + arb(0, ((hi - lo) / 2).upper())


def widen(value, bound):
    return value + arb(0, a(bound).upper())


def positive(value, label):
    if not value.is_finite() or not value.lower() > 0:
        raise Unresolved(label + " lacks a strict positive endpoint")
    return value


class Work:
    def __init__(self, args):
        self.args, self.start = args, time.perf_counter()
        self.evaluations = 0
        self.maximum_rss_kb = 0
        self.phase = "initial controls"
        self.integration_exception = None
        self.integration_depth = 0

    def check(self, increment=0):
        self.evaluations += increment  # A rejecting requested tick remains counted.
        if self.evaluations > self.args.max_evaluations:
            raise EngineeringCap("The declared two-million evaluation cap was reached")
        if time.perf_counter() - self.start > self.args.seconds:
            raise EngineeringCap("The declared 900-second numerical wall cap was reached")
        if os.nice(0) < 10 or len(os.listdir("/proc/self/task")) > 4:
            raise RuntimeError("The own process left its nice/four-thread contract")
        with open("/proc/self/status") as stream:
            for line in stream:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1])
                    self.maximum_rss_kb = max(self.maximum_rss_kb, rss)
                    if rss > self.args.memory_mb * 1024:
                        raise EngineeringCap("The declared 4096-MB resident-memory cap was reached")


class Integrals:
    def __init__(self, work):
        self.work = work

    def real(self, function, left, right, goal, record):
        w = self.work
        record.update({"left": str(left), "right": str(right), "requested_absolute_goal": encode(goal),
                       "requested_relative_goal": "0", "preceding_counter": w.evaluations,
                       "status": "unresolved", "returned_ball": None})
        w.check()
        w.integration_depth += 1
        value, wrapper = None, None
        try:
            def callback(z, analytic):
                if w.integration_exception is not None:
                    return acb("nan")
                try:
                    w.check(1)
                    out = function(z, analytic)
                    return out if out.is_finite() else acb("nan")
                except BaseException as error:
                    if w.integration_exception is None:
                        w.integration_exception = (error, error.__traceback__)
                    return acb("nan")
            try:
                value = acb.integral(callback, acb(a(left)), acb(a(right)), rel_tol=arb(0), abs_tol=goal,
                    eval_limit=max(1, w.args.max_evaluations - w.evaluations), depth_limit=40, use_heap=True)
            except BaseException as error:
                wrapper = error
            if w.integration_exception is not None:
                error, traceback = w.integration_exception
                raise error.with_traceback(traceback)
            if wrapper is not None:
                raise wrapper
            w.check()
            if value is None or not value.is_finite() or not value.imag.contains(0):
                raise Unresolved("Petras did not return a finite real-character enclosure")
            result = value.real
            record.update({"returned_ball": encode(result), "actual_radius_upper": encode(result.rad().upper()),
                           "requested_goal_is_not_an_error_certificate": True, "status": "finite_directed_integral_enclosed"})
            return result
        except BaseException as error:
            record["failure"] = {"class": type(error).__name__, "message": str(error), "phase": w.phase}
            raise
        finally:
            record["counter_after"] = w.evaluations
            w.integration_depth -= 1
            if w.integration_depth == 0:
                w.integration_exception = None


def grouped(y, x, h1, h2):
    e0, e1, e2 = (-x * y).exp(), (-h1 * y).exp(), (-h2 * y).exp()
    aa, bb = -(-h1 * y).expm1(), -(-h2 * y).expm1()
    mixed = h1 * e1 * bb + h2 * e2 * aa
    m = e0 * aa * bb
    first = e0 * (-x * aa * bb + mixed)
    second = e0 * (x**2 * aa * bb - 2 * x * mixed - h1**2 * e1 * bb - h2**2 * e2 * aa
                   + 2 * h1 * h2 * e1 * e2)
    return m, first, second


def heat_g(t):
    z = t / 2
    b0, b1 = z.bessel_i(0, scaled=True), z.bessel_i(1, scaled=True)
    return (b0 + b1) * b0**2 / 2


def exact_endpoint(value):
    mantissa, exponent = value.man_exp()
    return F(int(mantissa)) * F(2)**int(exponent)


def width(value):
    return exact_endpoint(value.upper()) - exact_endpoint(value.lower())


def sign(value):
    return "positive" if value.lower() > 0 else "negative" if value.upper() < 0 else "unresolved"


def overlap(values, label):
    lo = max(exact_endpoint(v.lower()) for v in values)
    hi = min(exact_endpoint(v.upper()) for v in values)
    if lo > hi:
        raise Conflict(label + " has disjoint purported enclosures")
    return {"lower": str(lo), "upper": str(hi), "overlap": True}


def describe(value):
    record = encode(value)
    record["full_width_exact"] = str(width(value))
    return record


def ball_key(value):
    return tuple((str(v.man_exp()[0]), int(v.man_exp()[1]))
                 for v in (value.real.mid(), value.real.rad(), value.imag.mid(), value.imag.rad()))


class Heat:
    def __init__(self, work, result):
        self.work, self.result = work, result
        self.integrals = Integrals(work)
        self.g_cache = {}

    def g(self, t):
        key = (ctx.prec, ball_key(t))
        if key not in self.g_cache:
            if len(self.g_cache) >= 4096:
                self.g_cache.clear()
            self.g_cache[key] = heat_g(t)
        return self.g_cache[key]

    def shape(self, name):
        if name == "R_D":
            x, h1, h2 = a(F(1, 3)), a(F(1, 3)), a(F(1, 3))
        elif name == "R_1_family":
            x = hull(0, F(1, 10**30))
            h1 = h2 = (1 - x) / 2
        elif name == "R_1_massless":
            x, h1, h2 = a(0), a(F(1, 2)), a(F(1, 2))
        elif name == "R_2":
            x, h1, h2 = a(0), a(F(1, 5)), a(F(4, 5))
        elif name == "R_3":
            x, h1, h2 = a(0), a(F(1, 20)), a(F(19, 20))
        else:
            raise ValueError("Undeclared C3 rectangle")
        return x, h1, h2

    def tails(self, mu, parameters):
        x, h1, h2 = parameters
        t0 = arb.pi() / (8 * U**2)
        first, second = a(0), a(0)
        for rho in (x, x + h1, x + h2, a(1)):
            if rho == 0:
                continue
            if rho.lower() <= 0:
                f1 = f2 = a(1)
            else:
                v = (mu**2 * rho * U**2).lower()
                positive(v, "Positive-mass derivative tail argument")
                f1, f2 = (-v).exp(), (1 + v) * (-v).exp()
            first += f1.upper()
            second += f2.upper()
        return t0, arb.pi() * first / (4 * U**2), arb.pi() * second / (2 * U**2)

    def evaluate(self, name, mu, components, digits, bisected, label):
        ctx.dps = digits
        self.work.phase = label
        self.work.check()
        mu = a(mu) if not isinstance(mu, arb) else mu
        positive(mu, "The declared positive mu")
        parameters = self.shape(name)
        panels = [(F(0), F(1))] + [(F(2**j), F(2**(j+1))) for j in range(22)]
        if bisected:
            panels = [(a0, (a0+b0)/2) for a0, b0 in panels] + [((a0+b0)/2, b0) for a0, b0 in panels]
            panels.sort()
        record = {"label": label, "shape": name, "mu": describe(mu), "decimal_digits": digits,
                  "panels_bisected": bisected, "parameters": [encode(v) for v in parameters],
                  "components": {}, "status": "unresolved", "full_parameter_balls_retained": True}
        self.result["heat_calls"].append(record)
        self.result["last_active_call"] = label
        tails = self.tails(mu, parameters)
        output = {}
        for j in components:
            def function(u, analytic):
                y = mu**2 * u**2
                m, d1, d2 = grouped(y, *parameters)
                term = m if j == 0 else y*d1 if j == 1 else y*d1 + y**2*d2
                return 2**(j+1) / arb.pi().sqrt() * term * self.g(u**2)
            component = {"panels": [], "tail_upper": encode(tails[j].upper()),
                         "tail_kind": "positive" if j == 0 else "symmetric", "status": "unresolved"}
            record["components"][str(j)] = component
            total = a(0)
            for left, right in panels:
                self.work.check(1)
                midpoint = function(acb(a((left+right)/2)), False)
                if not midpoint.is_finite():
                    raise Unresolved("The full-parameter tolerance-request midpoint is nonfinite")
                radius = midpoint.real.rad().upper()
                goal = (a(F(1, 10**13 * len(panels))) + 8*a(right-left)*radius).upper()
                row = {"midpoint_full_parameter_radius": encode(radius),
                       "radius_is_only_a_request_heuristic": True}
                component["panels"].append(row)
                total += self.integrals.real(function, left, right, goal, row)
            full = total + hull(0, tails[j].upper()) if j == 0 else widen(total, tails[j])
            component.update({"finite_panels_sum": describe(total), "with_tail": describe(full),
                              "status": "enclosed", "numerical_and_parameter_widths": "joint_full_enclosure"})
            output[j] = full
        record["status"] = "enclosed"
        if 0 in output:
            positive(output[0], "Grouped positive I")
        print("Enclosed " + label + "; evaluations=" + str(self.work.evaluations)
              + "; elapsed=" + format(time.perf_counter()-self.work.start, ".2f") + "s", flush=True)
        return output


def logarithmic_q(values):
    return -4 + 2*values[1]/positive(values[0], "I in q")


def logarithmic_derivative(values):
    denominator = positive(values[0], "I in logarithmic derivative")
    return 2*(values[2]/denominator - (values[1]/denominator)**2)


def positive_light_bound(mu):
    """C2b only; no derivative or massless-to-family numerical substitution."""
    delta = a(F(1, 10**30))
    mu0, mu1 = mu.lower(), mu.upper()
    positive(mu0, "C2b lower mu")
    if not (mu1*(2*delta).sqrt()).upper() <= 1:
        raise Unresolved("C2b's stated mu sqrt(2 delta) domain was not proved")
    bracket = 2/(3*arb.pi().sqrt()) + arb.pi()/4 + arb.pi()/2*(1/(mu0*(2*delta).sqrt())).log()
    bound = delta * mu1**2 * bracket
    return {"delta_range": ["0", "1/1000000000000000000000000000000"],
            "difference_I_massless_minus_I_family": encode(hull(0, bound.upper())),
            "continuous_value_at_delta_zero": "0", "mu": describe(mu),
            "supremum_method": "delta maximum; separate positive mu-square and logarithm upper factors",
            "not_used_for_derivative_transfer": True, "direct_family_balls_are_retained": True}


def initial_controls(heat, result):
    ctx.dps = 80
    heat.work.phase = "zero-width and angular normalization controls"
    controls = result["initial_controls"]
    zero_rows = controls["zero_width_rectangles"] = []
    for zero_leg in (1, 2):
        parameters = (a(F(1, 3)), a(0) if zero_leg == 1 else a(F(1, 3)),
                      a(0) if zero_leg == 2 else a(F(1, 3)))
        row = {"zero_leg": zero_leg, "evaluations": [], "passed": False}
        zero_rows.append(row)
        for y in (acb(0), acb(1), acb(a(F(1, 4)), a(F(1, 3)))):
            heat.work.check(1)
            values = grouped(y, *parameters)
            row["evaluations"].append({"y": encode(y), "M_Mprime_Msecond": [encode(v) for v in values]})
            if any(v != 0 for v in values):
                raise Conflict("A zero-width rectangle did not give exact zero before division")
        row["passed"] = True
    angular_rows = controls["angular_G"] = []
    for t in (F(1, 4), F(1), F(4)):
        row = {"t": str(t), "integrals": [], "passed": False}
        angular_rows.append(row)
        factors = []
        for cosine in (False, True):
            def angular(theta, analytic):
                value = (-a(t)*theta.sin()**2).exp()
                return value*theta.cos()**2 if cosine else value
            item = {"cosine_squared_weight": cosine}
            row["integrals"].append(item)
            factors.append(2/arb.pi()*heat.integrals.real(angular, F(0), arb.pi()/2,
                                                        a(F(1, 10**30)), item))
        angular_value = factors[1]*factors[0]**2
        heat.work.check(1)
        bessel_value = heat_g(acb(a(t))).real
        row.update({"angular_G": describe(angular_value), "scaled_Bessel_G": describe(bessel_value),
                    "overlap": overlap([angular_value, bessel_value], "Angular/Bessel normalization"),
                    "passed": True})
    controls["passed"] = True


def match(values, mu):
    spacing = (6*arb.pi()*positive(values[0], "I in the single Newton match")).sqrt()
    positive(spacing, "a over Planck length")
    return {"a_over_lP": spacing, "M_times_lP": mu/spacing}


def ratio(heat, name, mu, denominator, p, digits, bisected, label):
    p = a(p) if not isinstance(p, arb) else p
    numerator = heat.evaluate(name, mu*p**2, (0,), digits, bisected, label)[0]
    return numerator/(positive(denominator, "Matched I in coefficient ratio")*p**4)


def accepted_build(heat, name, left, right, digits, bisected, row, label):
    ctx.dps = digits
    mu = hull(left, right)
    row.update({"decimal_digits": digits, "panels_bisected": bisected, "status": "unresolved"})
    signs = []
    for endpoint, expected in ((left, "positive"), (right, "negative")):
        values = heat.evaluate(name, a(endpoint), (0, 1), digits, bisected, label + " sign " + str(endpoint))
        q = logarithmic_q(values)
        signs.append({"mu_exact": str(endpoint), "q": describe(q), "sign": sign(q)})
        if sign(q) != expected:
            raise Unresolved("The repeated accepted bracket lacks its required endpoint sign")
    row["endpoint_signs"] = signs
    values = heat.evaluate(name, mu, (0, 1, 2), digits, bisected, label + " root interval")
    q, derivative = logarithmic_q(values), logarithmic_derivative(values)
    row.update({"mu_bracket": describe(mu), "I_I1_I2": [describe(values[j]) for j in range(3)],
                "q": describe(q), "dq_dlogmu": describe(derivative),
                "root_exists_for_every_declared_member": True,
                "uniqueness_scope": "this retained bracket only"})
    if not derivative.upper() < 0:
        raise Unresolved("The whole-bracket logarithmic derivative is not strictly negative")
    matched = match(values, mu)
    for key, value in matched.items():
        row[key] = describe(value)
    if width(matched["a_over_lP"]) > F(1, 10**6):
        raise Unresolved("The matched spacing full width exceeds one millionth")
    endpoint = ratio(heat, name, mu, values[0], F(2), digits, bisected, label + " P=2 endpoint")
    row["kappa_P2"] = describe(endpoint)
    if width(endpoint) > F(1, 10**6):
        raise Unresolved("The P=2 coefficient full width exceeds one millionth")
    row["status"] = "accepted_root_and_endpoint_enclosed"
    return {"mu": mu, "I": values[0], "q": q, "derivative": derivative, "endpoint": endpoint, **matched}


def evaluate_shape(heat, name, row):
    ctx.dps = 80
    row.update({"status": "unresolved", "scan": [], "bisection": [], "accepted_builds": [],
                "point_curve": [], "interval_curve": [], "exhaustive_root_count": False})
    previous = None
    bracket = None
    for exponent in range(-8, 3):
        mu = F(2)**exponent
        values = heat.evaluate(name, a(mu), (0, 1), 80, False, name + " coarse scan " + str(mu))
        q = logarithmic_q(values)
        item = {"mu_exact": str(mu), "q": describe(q), "sign": sign(q)}
        row["scan"].append(item)
        if previous is not None and previous[1] == "positive" and item["sign"] == "negative":
            bracket = (previous[0], mu)
            break
        previous = (mu, item["sign"])
    if bracket is None:
        raise Unresolved("The declared coarse scan found no adjacent certified positive-to-negative pair")
    left, right = bracket
    while right-left >= F(1, 10**8):
        midpoint = (left+right)/2
        values = heat.evaluate(name, a(midpoint), (0, 1), 80, False, name + " bisection " + str(midpoint))
        q = logarithmic_q(values)
        item = {"mu_exact": str(midpoint), "q": describe(q), "sign": sign(q)}
        row["bisection"].append(item)
        if item["sign"] == "positive":
            left = midpoint
        elif item["sign"] == "negative":
            right = midpoint
        else:
            raise Unresolved("A bisection sign is unresolved before the declared bracket width")
    row["bracket_exact"] = [str(left), str(right)]
    row["bracket_width_exact"] = str(right-left)
    builds = []
    for digits, bisected, label in ((80, False, "baseline"), (80, True, "bisected panels"),
                                     (120, False, "higher precision")):
        build_row = {"label": label}
        row["accepted_builds"].append(build_row)
        builds.append(accepted_build(heat, name, left, right, digits, bisected, build_row, name + " " + label))
    row["independent_controls"] = {key: overlap([b[key] for b in builds], name + " " + key)
                for key in ("I", "q", "derivative", "a_over_lP", "M_times_lP", "endpoint")}
    ctx.dps = 80
    baseline = builds[0]
    row["endpoint_decision"] = {"kappa_P2_excludes_one": not baseline["endpoint"].contains(1),
          "twenty_percent_deviation_proved": (baseline["endpoint"].upper() <= a(F(4, 5)).lower()
                                              or baseline["endpoint"].lower() >= a(F(6, 5)).upper()),
          "scope": "necessary two-spatial-derivative vacuum amplitude test, no material existence theorem"}
    if name == "R_1_family":
        row["positive_light_analytical_bound"] = positive_light_bound(baseline["mu"])
        row["endpoint_positive_light_analytical_bound"] = positive_light_bound(4*baseline["mu"])
    for j in range(17):
        p = F(1) + F(j, 16)
        value = a(1) if j == 0 else baseline["endpoint"] if j == 16 else ratio(
                    heat, name, baseline["mu"], baseline["I"], p, 80, False, name + " point P=" + str(p))
        item = {"P_exact": str(p), "kappa": describe(value), "point_width_target_met": width(value) <= F(1, 10**6)}
        row["point_curve"].append(item)
        if name == "R_1_family":
            item["positive_light_analytical_bound"] = positive_light_bound(baseline["mu"]*a(p)**2)
        if not item["point_width_target_met"]:
            raise Unresolved("A point coefficient ratio full width exceeds one millionth")
    for j in range(64):
        lo, hi = F(1) + F(j, 64), F(1) + F(j+1, 64)
        p = hull(lo, hi)
        value = ratio(heat, name, baseline["mu"], baseline["I"], p, 80, False,
                      name + " full P bin " + str(j))
        item = {"P_exact_endpoints": [str(lo), str(hi)], "kappa_full_bin": describe(value),
                "sample_extrema_used": False}
        row["interval_curve"].append(item)
        if name == "R_1_family":
            item["positive_light_analytical_bound"] = positive_light_bound(baseline["mu"]*p**2)
    row["status"] = "shape_root_controls_and_curve_enclosed"


def write_outputs(result, path):
    path.write_text(json.dumps(result, indent=2) + "\n")
    lines = ["C3.1 bounded scalar result: " + result["status"],
             "shape | status | root bracket | a/lP | M*lP | kappa(2) | excludes 1 | >=20%"]
    for row in result["shapes"]:
        first = row.get("accepted_builds", [{}])[0] if row.get("accepted_builds") else {}
        fields = [row["shape"], row["status"], ", ".join(row.get("bracket_exact", [])),
                  first.get("a_over_lP", {}).get("display_ball", "pending"),
                  first.get("M_times_lP", {}).get("display_ball", "pending"),
                  first.get("kappa_P2", {}).get("display_ball", "pending"),
                  str(row.get("endpoint_decision", {}).get("kappa_P2_excludes_one", "pending")),
                  str(row.get("endpoint_decision", {}).get("twenty_percent_deviation_proved", "pending"))]
        lines.append(" | ".join(fields))
        if "failure" in row:
            lines.append("  " + row["failure"]["class"] + ": " + row["failure"]["message"])
    resource = result.get("resources", {})
    lines.append("Requested integrand evaluations: " + str(resource.get("evaluations", "pending")))
    lines.append("No production coupling, material, ground, clock, COM or stellar certificate.")
    path.with_suffix(".txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--max-evaluations", type=int, default=2_000_000)
    parser.add_argument("--memory-mb", type=int, default=4096)
    args = parser.parse_args()
    if not (0 < args.seconds <= 900 and 0 < args.max_evaluations <= 2_000_000 and 0 < args.memory_mb <= 4096):
        parser.error("Caps must be positive and no larger than the declared C3 caps")
    if args.output.suffix != ".json":
        parser.error("The result destination must end in .json")
    if any(os.environ.get(key) != "1" for key in POOL_NAMES):
        parser.error("All six own numerical thread pools must equal 1")
    if os.environ.get("QI2_REMOTE_RUN") != "1" or sys.executable != "<internal>/python":
        parser.error("Numerical execution is restricted to the remote qi2 interpreter")
    if os.nice(0) < 10:
        parser.error("The own process must run at nice 10 or lower scheduling priority")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.output.with_suffix(".txt").exists():
        parser.error("Existing scientific result files must not be overwritten")
    global arb, acb, ctx
    import flint
    if flint.__version__ != "0.9.0":
        raise RuntimeError("The directed backend must be pinned python-flint 0.9.0")
    arb, acb, ctx = flint.arb, flint.acb, flint.ctx
    ctx.dps, ctx.threads = 80, 1
    work = Work(args)
    result = {"status": "unresolved", "specification": SPEC, "existing_single_review": REVIEW,
              "runtime": {"PID": os.getpid(), "interpreter": sys.executable, "nice": os.nice(0),
                          "backend": "python-flint 0.9.0 directed Arb/Acb", "scaled_Bessel": "exp(-z) I_n(z)",
                          "CPU_only": True, "GPU_required": False, "thread_pools": {k: os.environ[k] for k in POOL_NAMES}},
              "caps": {"seconds": args.seconds, "integrand_evaluations": args.max_evaluations,
                       "threads": 4, "memory_MB": args.memory_mb},
              "U_exact": str(U), "initial_controls": {}, "shapes": [], "heat_calls": [],
              "production_parameters_adopted": False, "ground_certified": False, "clock_certified": False,
              "COM_certified": False, "stellar_certified": False, "physical_mass_selected": False}
    heat = Heat(work, result)
    print("C3 admitted CPU process PID=" + str(os.getpid()) + " nice=" + str(os.nice(0)), flush=True)
    try:
        work.check()
        initial_controls(heat, result)
        print("Zero-width and angular controls passed", flush=True)
        for name in ("R_D", "R_1_family", "R_1_massless", "R_2", "R_3"):
            row = {"shape": name}
            result["shapes"].append(row)
            try:
                evaluate_shape(heat, name, row)
            except BaseException as error:
                row["failure"] = {"class": type(error).__name__, "message": str(error), "phase": work.phase}
                raise
            print(name + " completed root, controls, 17 points and 64 full bins", flush=True)
            write_outputs(result, args.output)
        result["status"] = "all_declared_shapes_enclosed"
    except BaseException as error:
        result["failure"] = {"class": type(error).__name__, "message": str(error), "phase": work.phase}
        print(type(error).__name__ + ": " + str(error) + " at " + work.phase, flush=True)
    finally:
        result["resources"] = {"elapsed_seconds": time.perf_counter()-work.start,
              "evaluations": work.evaluations, "rejecting_requested_tick_counted": True,
              "maximum_RSS_kB": work.maximum_rss_kb, "own_threads_at_exit": len(os.listdir("/proc/self/task"))}
        write_outputs(result, args.output)
        print("Final " + result["status"] + "; result=" + str(args.output), flush=True)
    return 0 if result["status"] == "all_declared_shapes_enclosed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

