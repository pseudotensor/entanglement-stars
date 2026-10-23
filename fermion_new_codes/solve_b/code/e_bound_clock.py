#!/usr/bin/env python3
"""Directed certificates for the frozen original-H_* bound-pair control.

Governing inputs: strand_E_clock_spec.tex, review_E_clock_spec.tex, and
strand_E_bound_clock.tex.  No free spectrum is discretized.  The contact
rule, exact lattice measure, fixed trial/readout, and every error category
remain part of the certificate.  Numerical work is remote-only, after
separate actual-code reviews and Ops admission.  No COM/medium extension.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import time

from remote_runtime import remote_execution
import e_bound_clock_ground_input as ground_input


F = Fraction
ALPHA = F(1, 10**6)
T0, ETA, RHO = F(1, 64), F(1, 32), F(5, 2)
CUTOFFS = (16, 24, 32, 48, 64)
ORDERS = (16, 32, 64, 128, 256)
PRECISIONS = (128, 256, 512, 1024, 2048)
MAX_RANK = 2048
SOURCE_PATH = "solve/strand_E_clock_spec.tex"
REVIEW_PATH = "solve/review_E_clock_spec.tex"
GROUND_CODE_REVIEW_PATH = "solve/review_E_ground_code.tex"


class Unresolved(RuntimeError):
    """A certified gate or a predeclared resource budget remains unresolved."""


class EngineeringCap(Unresolved):
    """Must escape integration callbacks, rather than becoming a local NaN."""


class RefineOrder(Unresolved):
    """The declared analytic projection/tail budget dominates this order."""


class AnalyticDomain(Exception):
    """A requested integration cell has not proved all analytic conditions."""


class DirectInputRejected(Unresolved):
    """A fixed direct-input contradiction is retained for review, not bypassed."""


def load_backend():
    """Host-only loading; Ops supplies an isolated pinned wheel, never pip here."""
    global arb, acb, arb_mat, acb_mat, ctx, flint_module
    import flint
    flint_module = flint
    from flint import arb, acb, arb_mat, acb_mat, ctx
    if getattr(flint, "__version__", None) != "0.9.0":
        raise RuntimeError("The isolated pinned python-flint0.9.0 backend is required")
    required = ((arb, "legendre_p_root"), (arb, "mid_rad_10exp"),
                (acb, "integral"), (arb_mat, "inv"), (acb_mat, "eig"))
    for owner, name in required:
        if not hasattr(owner, name):
            raise RuntimeError("Pinned directed backend lacks required API: " + name)
    return "python-flint" + flint.__version__ + "; Arb/Acb directed ball operations"


def a(value=0):
    if isinstance(value, Fraction):
        return arb(value.numerator) / value.denominator
    return arb(value)


def c(value=0, imaginary=0):
    return acb(a(value), a(imaginary))


def upper(value):
    return value.upper()


def lower(value):
    return value.lower()


def abs_upper(value):
    return abs(value).upper()


def positive(value, label):
    if not value.is_finite() or not lower(value) > 0:
        raise Unresolved(label + " does not have a certified positive lower endpoint")
    return value


def nonzero(value, label):
    if not value.is_finite() or value.contains(0):
        raise Unresolved(label + " contains zero or is non-finite")
    return value


def widen(value, error):
    """Add a proved real or complex absolute error without erasing radii."""
    radius = upper(a(error))
    if isinstance(value, acb):
        return value + acb(arb(0, radius), arb(0, radius))
    return value + arb(0, radius)


def interval_hull(left, right):
    lo, hi = lower(a(left)), upper(a(right))
    if lo > hi:
        raise ArithmeticError("Unordered ball hull")
    return a((lo + hi) / 2) + arb(0, upper((hi - lo) / 2))


def dyadic(value):
    """Exact endpoint serialization; displayed decimal radii are never stripped."""
    mantissa, exponent = value.man_exp()
    return {"mantissa": str(mantissa), "exponent2": int(exponent)}


def encoded(value):
    if isinstance(value, acb):
        return {"real": encoded(value.real), "imag": encoded(value.imag)}
    if not value.is_finite():
        return {"status": "non-finite/unresolved"}
    return {"lower_dyadic": dyadic(lower(value)), "upper_dyadic": dyadic(upper(value)),
            "display_ball": value.str(30)}


def matrix_midpoint(matrix):
    """Untrusted arbitrary-precision dyadic approximation, subsequently checked."""
    output = arb_mat(matrix.nrows(), matrix.ncols())
    for i in range(matrix.nrows()):
        for j in range(matrix.ncols()):
            output[i, j] = matrix[i, j].mid()
    return output


def identity(n, complex_values=False):
    output = acb_mat(n, n) if complex_values else arb_mat(n, n)
    for i in range(n):
        output[i, i] = 1
    return output


def norm_bound(matrix):
    """A directed sqrt(norm1*normInf) operator-norm bound."""
    rows = [sum((abs_upper(matrix[i, j]) for j in range(matrix.ncols())), a(0))
            for i in range(matrix.nrows())]
    columns = [sum((abs_upper(matrix[i, j]) for i in range(matrix.nrows())), a(0))
               for j in range(matrix.ncols())]
    rowmax = max((upper(item) for item in rows), default=a(0))
    colmax = max((upper(item) for item in columns), default=a(0))
    return upper((rowmax * colmax).sqrt())


def matrix_enclosure_radius(matrix):
    remainder = arb_mat(matrix.nrows(), matrix.ncols())
    for i in range(matrix.nrows()):
        for j in range(matrix.ncols()):
            remainder[i, j] = arb(0, matrix[i, j].rad())
    return norm_bound(remainder)


class Work:
    def __init__(self, args):
        self.args, self.start = args, time.perf_counter()
        self.evaluations = 0
        self.rules = {}

    def reserve_matrix_entries(self, entries, label):
        self.check()
        # A predeclared conservative engineering estimate, followed by
        # actual VmRSS checks.  This is not an arithmetic or physics bound.
        estimate = entries * (192 + 2 * ((ctx.prec + 7) // 8))
        if estimate > self.args.memory_mb * 1024**2:
            raise EngineeringCap(label + " exceeds the predeclared matrix-memory estimate cap")

    def check(self, increment=0):
        self.evaluations += increment
        if self.evaluations > self.args.max_evaluations:
            raise EngineeringCap("Predeclared integrand/matrix-work evaluation cap reached")
        if time.perf_counter() - self.start > self.args.seconds:
            raise EngineeringCap("Predeclared engineering time cap reached")
        if len(os.listdir("/proc/self/task")) > 4:
            raise RuntimeError("The admitted one-worker job exceeded four threads")
        with open("/proc/self/status") as stream:
            for line in stream:
                if line.startswith("VmRSS:") and int(line.split()[1]) > self.args.memory_mb * 1024:
                    raise EngineeringCap("Predeclared engineering resident-memory cap reached")
        # No foreign-PID query or midrun ownership abort: valid admission stands.

    def rule(self, order):
        key = ctx.prec, order
        if key not in self.rules:
            self.check(order)
            values = [arb.legendre_p_root(order, k, weight=True) for k in range(order)]
            for node, weight in values:
                positive(weight, "Gauss weight")
                if not lower(node) > -1 or not upper(node) < 1:
                    raise Unresolved("Gauss node enclosure leaves(-1,1)")
            if not sum((weight for _, weight in values), a(0)).contains(2):
                raise ArithmeticError("Certified Gauss weights fail total measure")
            self.rules[key] = values
        return self.rules[key]


@dataclass
class Jet:
    """Ordinary scalar Taylor coefficients, with rigorous directed operations."""
    values: list

    def coerce(self, other):
        if isinstance(other, Jet):
            return other
        scalar = other if isinstance(other, (arb, acb)) else a(other)
        return Jet([scalar] + [a(0)] * (len(self.values) - 1))

    def __add__(self, other):
        other = self.coerce(other)
        return Jet([left + right for left, right in zip(self.values, other.values)])

    __radd__ = __add__

    def __neg__(self):
        return Jet([-item for item in self.values])

    def __sub__(self, other):
        return self + -self.coerce(other)

    def __rsub__(self, other):
        return self.coerce(other) - self

    def __mul__(self, other):
        other = self.coerce(other)
        return Jet([sum((self.values[k] * other.values[n - k] for k in range(n + 1)), a(0))
                    for n in range(len(self.values))])

    __rmul__ = __mul__

    def reciprocal(self):
        if isinstance(self.values[0], acb) and (not self.values[0].is_finite() or self.values[0].contains(0)):
            raise AnalyticDomain("Taylor denominator unresolved across analytic cell")
        output = [1 / nonzero(self.values[0], "Jet denominator")]
        for n in range(1, len(self.values)):
            output.append(-sum((self.values[k] * output[n - k] for k in range(1, n + 1)), a(0))
                          / self.values[0])
        return Jet(output)

    def __truediv__(self, other):
        return self * self.coerce(other).reciprocal()

    def __rtruediv__(self, other):
        return self.coerce(other) * self.reciprocal()

    def sqrt(self, analytic=False):
        value = self.values[0]
        root = value.sqrt(analytic=analytic) if isinstance(value, acb) else positive(value, "Real Taylor root").sqrt()
        if not root.is_finite() or root.contains(0):
            raise AnalyticDomain("Taylor root or reciprocal branch unresolved on cell")
        output = [root]
        for n in range(1, len(self.values)):
            output.append((self.values[n] - sum((output[k] * output[n - k]
                                                for k in range(1, n)), a(0))) / (2 * output[0]))
        return Jet(output)

    def exp(self):
        output = [self.values[0].exp()]
        for n in range(1, len(self.values)):
            output.append(sum((k * self.values[k] * output[n - k]
                               for k in range(1, n + 1)), a(0)) / n)
        return Jet(output)


def elementary_jet(radius, reference_t, degree=4, analytic=False):
    u0 = (-2 * a(reference_t)).exp() * radius**2
    u = Jet([u0 * a((-2)**j) / math.factorial(j) for j in range(degree + 1)])
    epsilon = (u + 1).sqrt(analytic=analytic)
    regulator = (-u).exp()
    f = regulator * (1 + 1 / epsilon) / 2
    j = regulator * (1 - 1 / epsilon) / 2
    return epsilon, f, j


def kernel_jet(r, s, reference_t, degree=4):
    er, fr, jr = elementary_jet(r, reference_t, degree)
    es, fs, js = elementary_jet(s, reference_t, degree)
    scale0 = (-3 * a(reference_t)).exp()
    scale = Jet([scale0 * a((-3)**j) / math.factorial(j) for j in range(degree + 1)])
    return scale * (fr * fs + jr * js) / (er + es)


def factorial_double(n):
    output = 1
    for value in range(n, 0, -2):
        output *= value
    return output


def angular_coefficients(order):
    result = []
    for k in range(order + 1):
        total = F(0)
        for k1 in range(k + 1):
            for k2 in range(k - k1 + 1):
                k3 = k - k1 - k2
                term = F(1, factorial_double(2 * k + 1))
                for ki in (k1, k2, k3):
                    term *= F(math.comb(2 * ki, ki) * factorial_double(2 * ki - 1), 4**ki)
                total += term
        if not 0 <= total <= 1:
            raise ArithmeticError("Exact angular coefficient leaves[0,1]")
        result.append(total)
    if result[:3] != [F(1), F(1, 2), F(11, 40)]:
        raise ArithmeticError("Exact angular coefficient normalization failed")
    return result


def lattice_weight(z, coefficients):
    if not abs_upper(z) < 1:
        if isinstance(z, acb):
            raise AnalyticDomain("Lattice-weight series outside analytic cell domain")
        raise Unresolved("Full-sphere lattice-weight series leaves its exact domain")
    value = sum((a(coefficient) * z**k for k, coefficient in enumerate(coefficients)), a(0))
    bound = abs_upper(z)**len(coefficients) / (1 - abs_upper(z))
    return widen(value, upper(bound))


def absorption(radius, coefficients, analytic=False):
    weight = lattice_weight(a(ALPHA)**2 * radius**2, coefficients)
    root = weight.sqrt(analytic=analytic) if isinstance(weight, acb) else positive(weight, "Lattice weight").sqrt()
    if not root.is_finite():
        raise AnalyticDomain("Lattice-weight root branch unresolved")
    return radius / (a(2).sqrt() * arb.pi()) * root


def signed_ldl_inertia(matrix, work):
    """Ball-certified full signed inertia, with no small-direction deletion."""
    n = matrix.nrows()
    if n != matrix.ncols():
        raise ArithmeticError("Inertia matrix is not square")
    current = arb_mat(matrix)
    positive_count, negative_count = 0, 0
    for k in range(n):
        work.check()
        candidates = [j for j in range(k, n) if not current[j, j].contains(0)]
        if not candidates:
            raise Unresolved("All certified LDL diagonal pivots are ambiguous; no direction is discarded")
        selected = max(candidates, key=lambda j: current[j, j].abs_lower())
        if selected != k:
            for j in range(n):
                current[k, j], current[selected, j] = current[selected, j], current[k, j]
            for i in range(n):
                current[i, k], current[i, selected] = current[i, selected], current[i, k]
        pivot = nonzero(current[k, k], "LDL pivot")
        if lower(pivot) > 0:
            positive_count += 1
        elif upper(pivot) < 0:
            negative_count += 1
        else:
            raise Unresolved("LDL pivot sign unresolved")
        for i in range(k + 1, n):
            for j in range(i, n):
                value = current[j, i] - current[i, k] * current[j, k] / pivot
                current[j, i] = current[i, j] = value
    return positive_count, negative_count


def certified_matrix_exp(matrix, work, tolerance):
    """Explicit directed Taylor tail and its propagation through squaring."""
    n = matrix.nrows()
    if n != matrix.ncols():
        raise ArithmeticError("Matrix exponential argument is not square")
    bound, squarings = norm_bound(matrix), 0
    while not bound / a(2)**squarings < a(F(1, 4)):
        squarings += 1
        work.check()
    scaled = matrix / a(2)**squarings
    scaled_norm = norm_bound(scaled)
    term, total = identity(n), identity(n)
    remainder, degree = a(1), 0
    local_target = a(tolerance) * (-2 * bound).exp() / (a(4)**squarings * a(10))
    while not upper(remainder) < lower(local_target):
        degree += 1
        work.check(n * n)
        term = term * scaled / degree
        total += term
        remainder = upper(scaled_norm.exp() * scaled_norm**(degree + 1) / math.factorial(degree + 1))
        if degree > 16 * ctx.prec:
            raise Unresolved("Matrix exponential Taylor cap reached")
    # Keep arithmetic balls and the analytic norm tail separately.  The
    # latter is only spread entrywise at the final return, after its norm
    # propagation; native ball arithmetic covers each multiplication.
    for _ in range(squarings):
        work.check(n * n)
        remainder = upper(2 * norm_bound(total) * remainder + remainder**2)
        total = total * total
    if not upper(remainder) < lower(a(tolerance)):
        raise Unresolved("Propagated matrix exponential Taylor remainder misses its budget")
    result = arb_mat(total)
    for i in range(n):
        for j in range(n):
            result[i, j] = widen(result[i, j], remainder)
    return result, {"Taylor_degree": degree, "squarings": squarings,
                    "propagated_analytic_norm_tail": encoded(remainder),
                    "directed_arithmetic_matrix_radius_bound": encoded(matrix_enclosure_radius(total))}


def exponential_jet(derivatives, work, tolerance):
    """Minus the entire five-block Toeplitz matrix; never exp(+J)."""
    d = derivatives[0].nrows()
    work.reserve_matrix_entries(8 * (5 * d)**2, "Five-block minus-Toeplitz exponential")
    block = arb_mat(5 * d, 5 * d)
    for i in range(5):
        for j in range(i, 5):
            entry = -derivatives[j - i] / math.factorial(j - i)
            for row in range(d):
                for column in range(d):
                    block[i * d + row, j * d + column] = entry[row, column]
    result, error = certified_matrix_exp(block, work, tolerance)
    jets = []
    for j in range(5):
        matrix = arb_mat(d, d)
        for row in range(d):
            for column in range(d):
                matrix[row, column] = math.factorial(j) * result[row, j * d + column]
        jets.append(matrix)
    return jets, error


def transpose(matrix):
    output = acb_mat(matrix.ncols(), matrix.nrows()) if isinstance(matrix, acb_mat) else arb_mat(matrix.ncols(), matrix.nrows())
    for i in range(matrix.nrows()):
        for j in range(matrix.ncols()):
            output[j, i] = matrix[i, j]
    return output


def as_complex_matrix(matrix):
    output = acb_mat(matrix.nrows(), matrix.ncols())
    for i in range(matrix.nrows()):
        for j in range(matrix.ncols()):
            output[i, j] = matrix[i, j]
    return output


def finite_matrix(matrix):
    return all(matrix[i, j].is_finite() for i in range(matrix.nrows()) for j in range(matrix.ncols()))


def verified_inverse(matrix, work, analytic=False):
    """Prove invertibility on the entire input ball by a Neumann residual.

    A midpoint inverse only seeds B0.  ||I-B0 M||<1 certifies every
    represented matrix, hence holomorphic inversion on an integration
    cell.  The final native inverse retains outward ball radii.
    """
    work.check(matrix.nrows()**2)
    complex_values = isinstance(matrix, acb_mat)
    midpoint = acb_mat(matrix.nrows(), matrix.ncols()) if complex_values else arb_mat(matrix.nrows(), matrix.ncols())
    for i in range(matrix.nrows()):
        for j in range(matrix.ncols()):
            midpoint[i, j] = matrix[i, j].mid()
    try:
        seed = midpoint.inv()
    except ZeroDivisionError:
        if analytic:
            raise AnalyticDomain("Midpoint inverse not certified at current precision")
        raise Unresolved("Midpoint inverse not certified at current precision")
    if not finite_matrix(seed):
        if analytic:
            raise AnalyticDomain("Midpoint inverse unavailable")
        raise Unresolved("Midpoint inverse unavailable")
    for i in range(seed.nrows()):
        for j in range(seed.ncols()):
            seed[i, j] = seed[i, j].mid()
    residual = norm_bound(identity(matrix.nrows(), complex_values) - seed * matrix)
    if not residual.is_finite() or not upper(residual) < 1:
        if analytic:
            raise AnalyticDomain("Finite inverse Neumann test failed across analytic cell")
        raise Unresolved("Finite inverse Neumann test failed across input interval")
    try:
        inverse = matrix.inv()
    except ZeroDivisionError:
        if analytic:
            raise AnalyticDomain("Outward inverse not certified across analytic cell")
        raise Unresolved("Outward inverse not certified across input interval")
    if not finite_matrix(inverse):
        if analytic:
            raise AnalyticDomain("Outward inverse is non-finite")
        raise Unresolved("Outward inverse is non-finite")
    return inverse, residual


def certified_matrix_expm1(matrix, work, tolerance):
    """Stable exp(M)-I, including small retained positive directions.

    Taylor starts at degree one.  Each doubling uses D -> D(D+2I),
    never subtracts two rounded unit matrices.  Error propagation is
    2||I+D||e+e^2, with all native arithmetic balls retained.
    """
    n, bound, squarings = matrix.nrows(), norm_bound(matrix), 0
    while not bound / a(2)**squarings < a(F(1, 4)):
        squarings += 1
        work.check()
    scaled = matrix / a(2)**squarings
    scaled_norm = norm_bound(scaled)
    total, term, degree = arb_mat(n, n), identity(n), 0
    # This intentionally conservative local target also covers large
    # positive exponentials.  The propagated final tail is checked again.
    target = a(tolerance) * (-2 * bound).exp() / (10 * a(4)**squarings)
    remainder = a(1)
    while not upper(remainder) < lower(target):
        degree += 1
        work.check(n * n)
        term = term * scaled / degree
        total += term
        remainder = upper(scaled_norm.exp() * scaled_norm**(degree + 1) / math.factorial(degree + 1))
        if degree > 16 * ctx.prec:
            raise Unresolved("Stable expm1 Taylor cap reached")
    for _ in range(squarings):
        work.check(n * n)
        remainder = upper(2 * norm_bound(identity(n) + total) * remainder + remainder**2)
        total = total * (total + 2 * identity(n))
    if not upper(remainder) < lower(a(tolerance)):
        raise Unresolved("Stable expm1 propagated tail misses budget")
    result = arb_mat(total)
    for i in range(n):
        for j in range(n):
            result[i, j] = widen(result[i, j], remainder)
    return result, {"Taylor_degree": degree, "squarings": squarings,
                    "analytic_norm_tail": encoded(remainder),
                    "directed_matrix_radius": encoded(matrix_enclosure_radius(total))}


class Integrals:
    """Rigorous Acb integration; tolerances are goals, returned radii are facts.

    Contract: python-flint0.9.0 acb.integral, official documentation
    https://python-flint.readthedocs.io/en/latest/acb.html#acb.integral
    and #acb.sqrt.  Analytic callbacks enclose the complete cell and
    forward analytic=True through every non-entire elementary function.
    Failed branch/denominator/inverse cells return non-finite, permitting
    the documented integrator to subdivide.  Exhausted global work caps
    escape; no failed integral is treated as an accurate point value.
    """
    def __init__(self, work, tolerance):
        self.work, self.tolerance = work, a(tolerance)
        self.calls, self.radius_sum = 0, a(0)
        self.radius_max = a(0)
        self.fatal_callback_error = None
        self.integration_depth = 0

    def integrate(self, function, left, right, label):
        if self.fatal_callback_error is not None:
            raise self.fatal_callback_error

        def callback(z, analytic):
            if self.fatal_callback_error is not None:
                return acb("nan")
            try:
                self.work.check(1)
                value = function(z, analytic)
                return value if value.is_finite() else acb("nan")
            except AnalyticDomain:
                return acb("nan")
            except BaseException as error:
                if self.fatal_callback_error is None:
                    self.fatal_callback_error = error
                return acb("nan")
        remaining = self.work.args.max_evaluations - self.work.evaluations
        if remaining <= 0:
            raise EngineeringCap("No remaining rigorous-integral evaluation budget")
        self.integration_depth += 1
        try:
            try:
                result = acb.integral(callback, left, right, abs_tol=self.tolerance,
                                      rel_tol=self.tolerance, eval_limit=min(remaining, self.work.args.integral_evaluations),
                                      depth_limit=80, deg_limit=ctx.prec, use_heap=True)
            except BaseException as error:
                if self.fatal_callback_error is not None and error is not self.fatal_callback_error:
                    raise self.fatal_callback_error from error
                raise
            if self.fatal_callback_error is not None:
                raise self.fatal_callback_error
            self.calls += 1
            if not result.is_finite():
                raise AnalyticDomain(label + " did not return a finite rigorous enclosure")
            self.radius_sum += upper(result.rad())
            self.radius_max = max(self.radius_max, upper(result.rad()))
            # No assertion that requested tolerance was reached: downstream
            # gates inspect the actual resulting intervals and propagated widths.
            return result
        finally:
            self.integration_depth -= 1
            if self.integration_depth == 0:
                self.fatal_callback_error = None

    def real(self, function, left, right, label):
        result = self.integrate(function, c(left), c(right), label)
        if not result.imag.contains(0):
            raise ArithmeticError(label + " lost its independently known real character")
        return result.real


def guarded_root(value, analytic):
    root = value.sqrt(analytic=analytic)
    if not root.is_finite() or root.contains(0):
        raise AnalyticDomain("Root/reciprocal branch not proved on integration cell")
    return root


def free_multiplier(r, source_t=0, lapse=1, analytic=False):
    return 2 * c(lapse) * guarded_root(1 + (-2 * c(source_t)).exp() * r**2, analytic)


def trial_value(r, index, coefficients, analytic=False):
    epsilon = guarded_root(1 + r**2, analytic)
    return absorption(r, coefficients, analytic) * (-r**2).exp() * (1 + 1 / epsilon) / 2 * epsilon**index


def legendre_values(r, panel, order):
    mapped = (2 * r - a(panel.left + panel.right)) / a(panel.length)
    values = [r * 0 + 1]
    if order > 1:
        values.append(mapped)
    for n in range(1, order - 1):
        values.append(((2 * n + 1) * mapped * values[-1] - n * values[-2]) / (n + 1))
    return [value * a(F(2 * n + 1, 1) / panel.length).sqrt() for n, value in enumerate(values)]


@dataclass(frozen=True)
class Panel:
    left: Fraction
    right: Fraction

    @property
    def length(self):
        return self.right - self.left


def panels_for(cutoff):
    endpoints = [F(0), F(1, 2), F(1), F(2), F(4), F(8), F(16)]
    endpoints.extend(F(value) for value in CUTOFFS if 16 < value <= cutoff)
    return [Panel(left, right) for left, right in zip(endpoints, endpoints[1:])]


def analytic_panel_bounds(panel):
    center, half = a((panel.left + panel.right) / 2), a(panel.length / 2)
    major = half * (a(RHO) + 1 / a(RHO)) / 2
    minor = half * (a(RHO) - 1 / a(RHO)) / 2
    radial = acb(center + arb(0, upper(major)), arb(0, upper(minor)))
    tc = a(T0 + ETA)
    source = acb(arb(0, upper(tc)), arb(0, upper(a(ETA))))
    u = (-2 * source).exp() * radial**2
    # Independent rectangle geometry avoids spurious dependencies from
    # complex squaring.  Both this bound and the native complex rectangle
    # enclosure are outward bounds, so their stronger lower endpoint is safe.
    xlo, xhi = lower(center - major), upper(center + major)
    xmin2 = a(0) if xlo <= 0 <= xhi else min(lower(xlo**2), lower(xhi**2))
    real_square_lower = lower(xmin2 - upper(minor**2))
    imag_square_upper = upper(2 * max(abs_upper(xlo), abs_upper(xhi)) * upper(minor))
    positive_scale = lower((-2 * tc).exp() * (2 * a(ETA)).cos())
    negative_scale = upper((2 * tc).exp())
    real_term_lower = (positive_scale * real_square_lower if real_square_lower >= 0
                       else negative_scale * real_square_lower)
    explicit_lower = lower(real_term_lower - negative_scale * upper((2 * a(ETA)).sin()) * imag_square_upper)
    u_lower = max(lower(u.real), explicit_lower)
    delta = lower(1 + u_lower)
    positive(delta, "Panel root margin")
    negative_real = max(a(0), -u_lower)
    radius = upper(center + a(RHO) * half)
    if not upper(a(ALPHA)**2 * radius**2) < a(F(1, 2)):
        raise Unresolved("Panel lattice ellipse exceeds its full-sphere branch margin")
    qnorm = upper(radius / (a(2).sqrt() * arb.pi() * (1 - a(ALPHA)**2 * radius**2).sqrt()))
    anorm = upper((1 + 1 / delta.sqrt()) / 2)
    epsilon_norm = upper((1 + (2 * tc).exp() * radius**2).sqrt())
    return {"delta": delta, "v": negative_real, "R": radius,
            "Q": qnorm, "A": anorm, "E": epsilon_norm}


def approximation_bounds(panels, order, quadrature, cutoff):
    bounds = [analytic_panel_bounds(panel) for panel in panels]
    denominator = (1 - 1 / a(RHO))**2
    projection_sum, quadrature_sum = a(0), a(0)
    for i, pi in enumerate(panels):
        for j, pj in enumerate(panels):
            bi, bj = bounds[i], bounds[j]
            kernel_bound = upper(2 * (3 * a(T0 + ETA)).exp() * bi["Q"] * bj["Q"]
                                 * bi["A"] * bj["A"] * (bi["v"] + bj["v"]).exp()
                                 / (bi["delta"].sqrt() + bj["delta"].sqrt()))
            e_projection = upper(8 * kernel_bound * a(RHO)**(-order) / denominator)
            e_quadrature = upper(8 * kernel_bound * a(RHO)**(-(2 * quadrature - order + 1)) / denominator)
            projection_sum += a(pi.length * pj.length) * e_projection**2
            quadrature_sum += a(pi.length * pj.length) * e_quadrature**2
    q = positive((-2 * a(T0 + ETA)).exp() * (2 * a(ETA)).cos(), "Complex cutoff Gaussian rate")
    mass = 1 / (8 * a(ALPHA)**3)
    tail = upper(a(2).sqrt() * (3 * a(T0 + ETA)).exp() * mass * (-q * cutoff**2).exp())
    return bounds, {"tail_projection": upper(tail + projection_sum.sqrt()),
                    "tail": tail, "projection": upper(projection_sum.sqrt()),
                    "quadrature": upper(2 * quadrature_sum.sqrt())}


def ground_reference_bounds(discretization, saved):
    """GE1--GE10, ONLY at source t=0/order0; old window errors stay intact."""
    d = discretization
    rho_ref = F(3)
    saved.update({"source": "solve/strand_E_ground_reference.tex",
                  "independent_source_review": "solve/review_E_ground_reference.tex",
                  "source_parameter_certified": 0, "source_derivative_orders_certified": [0],
                  "rho_reference_exact": "3", "global_source_window_rho_unchanged": str(RHO),
                  "actual_Legendre_order_per_panel": d.order, "actual_Gauss_order_per_panel": d.quadrature,
                  "panel_records": [], "ordered_product_panel_records": [], "status": "unresolved"})
    if d.quadrature < d.order:
        raise Unresolved("Reference quadrature order must retain Q>=N")
    bounds = []
    for index, panel in enumerate(d.panels):
        d.work.check(1)
        center, half = (panel.left + panel.right) / 2, panel.length / 2
        xlo, xhi = center - F(5, 3) * half, center + F(5, 3) * half
        distance = F(0) if xlo <= 0 <= xhi else min(abs(xlo), abs(xhi))
        delta_exact = 1 + distance**2 - F(16, 9) * half**2
        v_exact = max(F(0), F(16, 9) * half**2 - distance**2)
        radius_exact = center + rho_ref * half
        z_exact = ALPHA**2 * radius_exact**2
        first = index == 0
        gates = {"actual_panel_root_margin_positive": delta_exact > 0,
                 "actual_full_weight_root_disk_margin": z_exact < F(1, 2),
                 "actual_radius_within_frozen_ladder_bound80": radius_exact <= 80,
                 "first_panel_exact_geometry": not first or
                    (panel.left == 0 and panel.right == F(1, 2) and delta_exact == F(8, 9)
                     and v_exact == F(1, 9) and radius_exact == 1),
                 "noninitial_panel_c_at_least3h_delta_at_least1_v_zero": first or
                    (center >= 3 * half and delta_exact >= 1 and v_exact == 0)}
        record = {"panel_left_exact": str(panel.left), "panel_right_exact": str(panel.right),
                  "rectangle_x_lower_exact": str(xlo), "rectangle_x_upper_exact": str(xhi),
                  "rectangle_y_absolute_upper_exact": str(F(4, 3) * half),
                  "delta_exact": str(delta_exact), "v_exact": str(v_exact), "R_exact": str(radius_exact),
                  "alpha2_R2_exact": str(z_exact), "exact_geometry_gates": gates}
        saved["panel_records"].append(record)
        if not all(gates.values()):
            raise Unresolved("An actual reference panel lacks its GE branch/geometry certificate")
        delta = positive(a(delta_exact), "Reference principal epsilon margin")
        qbound = upper(a(radius_exact) / (a(2).sqrt() * arb.pi() * positive(1 - a(z_exact), "Reference weight margin").sqrt()))
        abound = upper((1 + 1 / delta.sqrt()) / 2)
        record.update({"q_alpha_absolute_upper": encoded(qbound), "both_spinor_legs_without_Gaussian_upper": encoded(abound),
                       "epsilon_real_part_lower": encoded(lower(delta.sqrt())),
                       "weight_root_disk_exclusion_upper": encoded(upper(a(z_exact) / (1 - a(z_exact))))})
        bounds.append({"delta": delta, "Q": qbound, "A": abound, "v": a(v_exact)})
    denominator = positive((1 - 1 / a(rho_ref))**2, "Reference tensor Chebyshev margin")
    projection_sum, quadrature_sum = a(0), a(0)
    for i, pi in enumerate(d.panels):
        for j, pj in enumerate(d.panels):
            d.work.check(1)
            bi, bj = bounds[i], bounds[j]
            majorant = upper(2 * bi["Q"] * bj["Q"] * bi["A"] * bj["A"] * (bi["v"] + bj["v"]).exp()
                             / positive(bi["delta"].sqrt() + bj["delta"].sqrt(), "Reference product denominator margin"))
            ep = upper(4 * majorant * (a(rho_ref)**(-d.order) + a(rho_ref)**(-d.order)) / denominator)
            nquad = 2 * d.quadrature - d.order + 1
            eq = upper(4 * majorant * (a(rho_ref)**(-nquad) + a(rho_ref)**(-nquad)) / denominator)
            projection_sum += a(pi.length * pj.length) * ep**2
            quadrature_sum += a(pi.length * pj.length) * eq**2
            saved["ordered_product_panel_records"].append({"left_panel": i, "right_panel": j,
                "B_reference_upper": encoded(majorant), "Chebyshev_projection_remainder_upper": encoded(ep),
                "Chebyshev_quadrature_remainder_upper": encoded(eq)})
    tail = d.errors["tail"]  # Original complete complex-window cutoff tail, unchanged.
    errors = {"tail": tail, "projection": upper(projection_sum.sqrt()),
              "tail_projection": upper(tail + projection_sum.sqrt()),
              "quadrature": upper(2 * quadrature_sum.sqrt())}
    saved.update({"interaction_error_bounds": {name: encoded(value) for name, value in errors.items()},
                  "complete_original_full_lattice_cutoff_tail_reused": True,
                  "source_window_error_object_replaced": False, "finite_matrix_and_rule_changed": False,
                  "status": "reference_interaction_error_enclosed"})
    return {"source_parameter_certified": 0, "source_derivative_orders_certified": [0],
            "errors": errors, "record": saved}


class RadialFunction:
    """A continuous, real-on-real, cutoff-supported analytic panel function."""
    def __init__(self, evaluate, label, support_panels=None):
        self.evaluate, self.label = evaluate, label
        self.cutoff_supported = True
        self.support_panels = support_panels

    def __call__(self, panel, r, analytic=False):
        return self.evaluate(panel, r, analytic)

    def scaled(self, factor):
        return RadialFunction(lambda panel, r, analytic: factor * self(panel, r, analytic), self.label + "*scale", self.support_panels)

    def __add__(self, other):
        support = (self.support_panels | other.support_panels
                   if self.support_panels is not None and other.support_panels is not None else None)
        return RadialFunction(lambda panel, r, analytic: self(panel, r, analytic) + other(panel, r, analytic),
                              self.label + "+" + other.label, support)

    def __sub__(self, other):
        return self + other.scaled(-1)


class Discretization:
    def __init__(self, cutoff, order, quadrature, weight_order, work, tolerance):
        self.cutoff, self.order, self.quadrature = cutoff, order, quadrature
        self.panels, self.work = panels_for(cutoff), work
        self.rank = len(self.panels) * order
        if self.rank > MAX_RANK:
            raise Unresolved("Frozen total polynomial-rank cap reached")
        self.coefficients = angular_coefficients(weight_order)
        self.panel_bounds, self.errors = approximation_bounds(self.panels, order, quadrature, cutoff)
        self.integrals = Integrals(work, tolerance)
        self.tolerance = a(tolerance)
        self.node_data = []
        for p, panel in enumerate(self.panels):
            center, half = a((panel.left + panel.right) / 2), a(panel.length / 2)
            for node, weight in work.rule(quadrature):
                r, w = center + half * node, half * weight
                values = legendre_values(r, panel, order)
                self.node_data.append((p, r, w, values, absorption(r, self.coefficients)))
        self.matrix_radii = []

    def basis(self, index):
        target, degree = divmod(index, self.order)
        return RadialFunction(lambda panel, r, analytic: legendre_values(r, self.panels[panel], self.order)[degree]
                              if panel == target else r * 0, "v_" + str(index), {target})

    def polynomial(self, coefficients, label):
        return RadialFunction(lambda panel, r, analytic:
                              sum((c(coefficients[panel * self.order + n]) * value
                                   for n, value in enumerate(legendre_values(r, self.panels[panel], self.order))), c(0)), label)

    def inner(self, left, right):
        # Analytic product of real functions; no conjugation of the complex
        # radial variable is introduced in the integration callback.
        return sum((self.integrals.real(lambda r, analytic, p=p: left(p, r, analytic) * right(p, r, analytic),
                                        panel.left, panel.right, left.label + "/" + right.label)
                    for p, panel in enumerate(self.panels)
                    if (left.support_panels is None or p in left.support_panels)
                    and (right.support_panels is None or p in right.support_panels)), a(0))

    def norm(self, function, certificate=False):
        squared = self.inner(function, function)
        bound = upper(squared)
        if not bound >= 0:
            raise ArithmeticError("A real norm-squared enclosure has negative upper endpoint")
        result = upper(bound.sqrt())
        if certificate:
            return result, {"joint_squared_norm_integral_enclosure": encoded(squared),
                            "joint_directed_norm_upper": encoded(result),
                            "all_scalar_matrix_energy_coefficient_normalization_uncertainties_retained": True}
        return result

    def project(self, function):
        return [self.inner(self.basis(i), function) for i in range(self.rank)]

    def kernel_matrices(self, source_t=0, degree=4):
        nodes = self.node_data
        count = len(nodes)
        self.work.reserve_matrix_entries((degree + 1) * count**2 + 4 * self.rank * count + 8 * self.rank**2,
                                         "Positive interaction cubature")
        kernels = [arb_mat(count, count) for _ in range(degree + 1)]
        for i, (_, ri, _, _, qi) in enumerate(nodes):
            for j in range(i, count):
                self.work.check(1)
                _, rj, _, _, qj = nodes[j]
                jet = kernel_jet(ri, rj, source_t, degree)
                for k, value in enumerate(jet.values):
                    kernels[k][i, j] = kernels[k][j, i] = qi * qj * value * math.factorial(k)
        embedding = arb_mat(self.rank, count)
        for j, (panel, _, weight, values, _) in enumerate(nodes):
            for n, value in enumerate(values):
                embedding[panel * self.order + n, j] = weight * value
        matrices = []
        for kernel in kernels:
            self.work.check(self.rank**2)
            matrix = embedding * kernel * transpose(embedding)
            # Exact input matrix is symmetric.  Intersect, rather than
            # select a midpoint, the two enclosures for each paired entry.
            for i in range(self.rank):
                for j in range(i + 1, self.rank):
                    lo = max(lower(matrix[i, j]), lower(matrix[j, i]))
                    hi = min(upper(matrix[i, j]), upper(matrix[j, i]))
                    matrix[i, j] = matrix[j, i] = interval_hull(lo, hi)
            self.matrix_radii.append(matrix_enclosure_radius(matrix))
            matrices.append(matrix)
        return matrices

    def trial(self, index):
        return RadialFunction(lambda panel, r, analytic: trial_value(r, index, self.coefficients, analytic),
                              "q_alpha*f*epsilon^" + str(index))

    def trial_tail(self, total_power):
        return upper(1 / (8 * a(ALPHA)**3) * (1 + 3 / a(ALPHA)**2)**a(F(total_power, 2))
                     * a(-2 * self.cutoff**2).exp())

    def trial_projection_error(self, index):
        summation = a(0)
        for panel, bound in zip(self.panels, self.panel_bounds):
            norm = bound["Q"] * bound["A"] * bound["v"].exp() * bound["E"]**index
            summation += a(panel.length) * (2 * norm * a(RHO)**(-self.order) / (1 - 1 / a(RHO)))**2
        return upper(summation.sqrt() + self.trial_tail(2 * index).sqrt())

    def multiplication_matrix(self, source_t=0, lapse=1):
        output = arb_mat(self.rank, self.rank)
        for p, panel in enumerate(self.panels):
            for i in range(self.order):
                for j in range(i, self.order):
                    def integrand(r, analytic):
                        values = legendre_values(r, panel, self.order)
                        return values[i] * values[j] * free_multiplier(r, source_t, lapse, analytic)
                    value = self.integrals.real(integrand, panel.left, panel.right, "V*h*V")
                    output[p * self.order + i, p * self.order + j] = value
                    output[p * self.order + j, p * self.order + i] = value
        return output

    def free_gram(self, z, source_t=0, lapse=1, analytic=False):
        output = acb_mat(self.rank, self.rank)
        for p, panel in enumerate(self.panels):
            for i in range(self.order):
                for j in range(i, self.order):
                    def integrand(r, radial_analytic):
                        values = legendre_values(r, panel, self.order)
                        denominator = free_multiplier(r, source_t, lapse, radial_analytic) - z
                        if denominator.contains(0):
                            raise AnalyticDomain("Exact free-resolvent denominator touches zero")
                        return values[i] * values[j] / denominator
                    value = self.integrals.integrate(integrand, c(panel.left), c(panel.right), "exact V*R0*V")
                    output[p * self.order + i, p * self.order + j] = value
                    output[p * self.order + j, p * self.order + i] = value
        return output


def trial_compression(discretization, matrix, diagnostics=None, refinement_inputs=None):
    """Full trial Gram and compression, with every exterior/projection error."""
    d = discretization
    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics["phase"] = "projecting_trial_vectors"
    trials = [d.trial(i) for i in range(2)]
    gram, compressed = arb_mat(2, 2), arb_mat(2, 2)
    radial_gram, tail_gram, raw_compressed = arb_mat(2, 2), arb_mat(2, 2), arb_mat(2, 2)
    projections = [d.project(function) for function in trials]
    for i in range(2):
        for j in range(i, 2):
            radial = d.inner(trials[i], trials[j])
            tail = interval_hull(0, d.trial_tail(i + j))
            value = radial + tail
            gram[i, j] = gram[j, i] = value
            radial_gram[i, j] = radial_gram[j, i] = radial
            tail_gram[i, j] = tail_gram[j, i] = tail
    serialize = lambda item: [[encoded(item[i, j]) for j in range(2)] for i in range(2)]
    diagnostics.update({"Gram": serialize(gram), "Gram_radial": serialize(radial_gram),
                        "Gram_exterior_tail": serialize(tail_gram),
                        "projection_coefficient_numerical_radii": [[encoded(upper(value.rad())) for value in vector]
                                                                   for vector in projections],
                        "matrix_numerical_radius_norm_upper": encoded(matrix_enclosure_radius(matrix)),
                        "operator_tail_projection_error_upper": encoded(d.errors["tail_projection"]),
                        "operator_quadrature_error_upper": encoded(d.errors["quadrature"])})
    norms = [upper(positive(gram[i, i], "Full trial Gram norm").sqrt()) for i in range(2)]
    projection_errors = [d.trial_projection_error(i) for i in range(2)]
    diagnostics["trial_projection_and_exterior_norm_errors"] = [encoded(value) for value in projection_errors]
    diagnostics["trial_exterior_norm_error_bounds"] = [encoded(upper(d.trial_tail(2 * i).sqrt())) for i in range(2)]
    operator_error = d.errors["tail_projection"] + d.errors["quadrature"]
    mqnorm = norm_bound(matrix)
    error_entries = arb_mat(2, 2)
    tail_projection_entries, quadrature_entries, trial_projection_entries = (
        arb_mat(2, 2), arb_mat(2, 2), arb_mat(2, 2))
    for i in range(2):
        for j in range(i, 2):
            value = sum((projections[i][row] * matrix[row, column] * projections[j][column]
                         for row in range(d.rank) for column in range(d.rank)), a(0))
            # The operator estimate alone already applies to exact full
            # trial vectors.  Also retain the explicit conservative trial
            # projection bound requested by the source; no tail is dropped.
            error = upper(operator_error * norms[i] * norms[j]
                          + mqnorm * (norms[i] * projection_errors[j] + norms[j] * projection_errors[i]
                                      + projection_errors[i] * projection_errors[j]))
            compressed[i, j] = compressed[j, i] = widen(value, error)
            raw_compressed[i, j] = raw_compressed[j, i] = value
            error_entries[i, j] = error_entries[j, i] = error
            tail_projection_entries[i, j] = tail_projection_entries[j, i] = upper(
                d.errors["tail_projection"] * norms[i] * norms[j])
            quadrature_entries[i, j] = quadrature_entries[j, i] = upper(
                d.errors["quadrature"] * norms[i] * norms[j])
            trial_projection_entries[i, j] = trial_projection_entries[j, i] = upper(
                mqnorm * (norms[i] * projection_errors[j] + norms[j] * projection_errors[i]
                          + projection_errors[i] * projection_errors[j]))
    detg_raw = gram[0, 0] * gram[1, 1] - gram[0, 1]**2
    detk_raw = raw_compressed[0, 0] * raw_compressed[1, 1] - raw_compressed[0, 1]**2
    detk_full = compressed[0, 0] * compressed[1, 1] - compressed[0, 1]**2
    analytic_error_max = max(upper(error_entries[i, j]) for i in range(2) for j in range(2))
    raw_radius_max = max(upper(raw_compressed[i, j].rad()) for i in range(2) for j in range(2))
    raw_positive = detk_raw.is_finite() and lower(detk_raw) > 0
    full_positive = detk_full.is_finite() and lower(detk_full) > 0
    bound_dominance = lower(analytic_error_max) >= upper(16 * raw_radius_max)
    diagnostics.update({"raw_projected_compression": serialize(raw_compressed),
                        "phase": "compression_and_determinants_enclosed_before_positivity_gates",
                        "compression": serialize(compressed),
                        "compression_absolute_error_bounds": serialize(error_entries),
                        "per_entry_analytic_error_contributions": {
                            "operator_tail_projection": serialize(tail_projection_entries),
                            "operator_quadrature": serialize(quadrature_entries),
                            "trial_projection_and_exterior": serialize(trial_projection_entries)},
                        "raw_compression_numerical_radii": [[encoded(upper(raw_compressed[i, j].rad()))
                                                             for j in range(2)] for i in range(2)],
                        "Gram_determinant_before_gate": encoded(detg_raw),
                        "raw_compression_determinant_before_gate": encoded(detk_raw),
                        "full_compression_determinant_before_gate": encoded(detk_full),
                        "maximum_analytic_entry_error_bound": encoded(analytic_error_max),
                        "maximum_raw_entry_numerical_radius": encoded(raw_radius_max),
                        "order_choice_rule": "Full determinant unresolved, raw determinant positive, analytic bound at least16 times raw numerical radius",
                        "raw_determinant_certified_positive": raw_positive,
                        "full_determinant_certified_positive": full_positive,
                        "analytic_bound_dominates_raw_radius_by16": bound_dominance})
    detg = positive(detg_raw, "Full trial Gram determinant")
    if not full_positive and raw_positive and bound_dominance:
        diagnostics["refinement_choice"] = "advance_existing_projection_and_quadrature_order; no impossibility verdict"
        raise RefineOrder("Trial compression determinant unresolved with declared analytic-bound dominance; advance existing order")
    diagnostics["refinement_choice"] = "retain_existing_precision_refinement_unless_final_gate_passes"
    detk = positive(detk_full, "Trial compression positive determinant")
    trace = (compressed[0, 0] * gram[1, 1] + compressed[1, 1] * gram[0, 0]
             - 2 * compressed[0, 1] * gram[0, 1]) / detg
    determinant = detk / detg
    discriminant = positive(trace**2 - 4 * determinant, "Trial generalized-eigenvalue discriminant")
    larger = positive((trace + discriminant.sqrt()) / 2, "Larger generalized trial eigenvalue")
    mu2 = positive(determinant / larger, "Frozen smaller trial eigenvalue mu2")
    emax = 2 * (1 + 3 / a(ALPHA)**2).sqrt()
    gamma = positive((4 * emax).log() / (2 * mu2), "Exact contact-rule gamma")
    if refinement_inputs is not None:
        # Current completed-call values only; never decode display records.
        # This private object is not part of the serialized compression.
        refinement_inputs.update({"current_compression_completed": True,
                                  "maximum_analytic_entry_error_bound": analytic_error_max,
                                  "maximum_raw_entry_numerical_radius": raw_radius_max,
                                  "raw_determinant_certified_positive": raw_positive,
                                  "full_determinant_certified_positive": full_positive,
                                  "analytic_bound_dominates_raw_radius_by16": bound_dominance})
    return mu2, gamma, diagnostics


def bell_bounds(p, degree=4):
    output = [a(1)]
    for j in range(1, degree + 1):
        output.append(upper(sum((math.comb(j - 1, k - 1) * p[k] * output[j - k]
                                 for k in range(1, j + 1)), a(0))))
    return output


def completion_bounds(discretization, gamma, gamma_midpoint, degree=4, reference_errors=None):
    d = discretization
    if reference_errors is not None and (degree != 0 or reference_errors.get("source_parameter_certified") != 0
                                        or reference_errors.get("source_derivative_orders_certified") != [0]):
        raise ValueError("Reference errors certify only explicit t=0 and derivative order zero")
    selected_errors = d.errors if reference_errors is None else reference_errors["errors"]
    q = positive((-2 * a(T0 + ETA)).exp() * (2 * a(ETA)).cos(), "Complex Gaussian q")
    def integrand(r, analytic):
        qa = absorption(r, d.coefficients, analytic)
        return qa**2 * (-2 * q * r**2).exp()
    low = sum((d.integrals.real(integrand, panel.left, panel.right, "complex T norm Gaussian")
               for panel in d.panels), a(0))
    mass = 1 / (8 * a(ALPHA)**3)
    gaussian_norm = upper(low + mass * (-2 * q * d.cutoff**2).exp())
    tnorm_complex = upper((3 * a(T0 + ETA)).exp() * gaussian_norm)
    window_error = d.errors["tail_projection"] + d.errors["quadrature"]
    tqnorm_complex = upper(tnorm_complex + window_error)
    total_error = selected_errors["tail_projection"] + selected_errors["quadrature"]
    tqnorm_selected = tqnorm_complex if reference_errors is None else upper(tnorm_complex + total_error)
    gamma_error = upper(abs(gamma - gamma_midpoint))
    p = [a(0)]
    for j in range(1, degree + 1):
        p.append(upper(math.factorial(j) * a(ETA)**(-j)
                       * max(upper(gamma) * tnorm_complex, gamma_midpoint * tqnorm_complex)))
    bells = bell_bounds(p, degree)
    free_complex = upper(2 * (1 + 3 / a(ALPHA)**2 * (2 * a(T0 + ETA)).exp()).sqrt())
    free_real = upper(2 * (1 + 3 / a(ALPHA)**2 * (2 * a(T0)).exp()).sqrt())
    h = [free_real] + [upper(math.factorial(j) * a(ETA)**(-j) * free_complex) for j in range(1, degree + 1)]
    categories = {}
    for name in ("tail_projection", "quadrature", "coupling"):
        differences = []
        for j in range(degree + 1):
            derivative_scale = math.factorial(j) * a(ETA)**(-j)
            value = (gamma_error * derivative_scale * tqnorm_selected if name == "coupling"
                     else upper(gamma) * derivative_scale * selected_errors[name])
            differences.append(upper(value))
        exponent_errors = [upper(differences[0] * bells[j]
                                 + sum((math.comb(j, k) * bells[j - k] * differences[k]
                                        for k in range(1, j + 1)), a(0))) for j in range(degree + 1)]
        hj_errors = []
        for j in range(degree + 1):
            value = a(0)
            for i in range(j + 1):
                for l in range(j - i + 1):
                    k = j - i - l
                    multiplier = math.factorial(j) // (math.factorial(i) * math.factorial(l) * math.factorial(k))
                    value += multiplier * h[l] * (exponent_errors[i] * bells[k] + bells[i] * exponent_errors[k])
            hj_errors.append(upper(value))
        categories[name] = hj_errors
    operator_norms = []
    for j in range(degree + 1):
        value = a(0)
        for i in range(j + 1):
            for l in range(j - i + 1):
                k = j - i - l
                multiplier = math.factorial(j) // (math.factorial(i) * math.factorial(l) * math.factorial(k))
                value += multiplier * bells[i] * h[l] * bells[k]
        operator_norms.append(upper(value))
    errors = [upper(sum((categories[name][j] for name in categories), a(0))) for j in range(degree + 1)]
    mixed_norms = [upper((1 + a(T0)) * operator_norms[j]
                         + (j * operator_norms[j - 1] if j else 0)) for j in range(degree + 1)]
    mixed_errors = [upper((1 + a(T0)) * errors[j] + (j * errors[j - 1] if j else 0)) for j in range(degree + 1)]
    output = {"T_complex_norm": tnorm_complex, "TQ_complex_norm": tqnorm_complex,
            "T_errors_total": upper(total_error),
            "gamma_radius": gamma_error, "Bell": bells, "categories": categories,
            "H_errors": errors, "H_norms": operator_norms,
            "mixed_norms": mixed_norms, "mixed_errors": mixed_errors}
    if reference_errors is not None:
        output.update({"TQ_reference_norm": tqnorm_selected, "source_parameter_certified": 0,
                       "source_derivative_orders_certified": [0], "reference_error_object": reference_errors,
                       "interaction_error_certificate_scope": "reference_t0_order0_GE1_GE12"})
    return output


class FiniteModel:
    """Only the interaction is finite rank; h stays exact multiplication."""
    def __init__(self, discretization, matrices, gamma_midpoint, degree=4, source_t=0, lapse=1):
        self.d, self.gamma = discretization, gamma_midpoint
        self.source_t, self.lapse = source_t, lapse
        self.matrices = matrices
        self.exp_metadata = []
        if degree == 4:
            self.exponential_derivatives, error = exponential_jet([gamma_midpoint * matrix for matrix in matrices],
                                                                  self.d.work, self.d.tolerance)
            self.exp_metadata.append(error)
        else:
            expm1, error = certified_matrix_expm1(-gamma_midpoint * matrices[0], self.d.work, self.d.tolerance)
            self.exponential_derivatives = [identity(self.d.rank) + expm1]
            self.z = expm1
            self.exp_metadata.append(error)
        # Stable Z is computed directly even when the Toeplitz block also
        # encloses exp(-gamma M).  No subtraction of rounded unit entries.
        if degree == 4:
            self.z, error = certified_matrix_expm1(-gamma_midpoint * matrices[0], self.d.work, self.d.tolerance)
            self.exp_metadata.append(error)
        self.b = self.d.multiplication_matrix(source_t, lapse)
        n = self.d.rank
        self.coupling_matrix = arb_mat(2 * n, 2 * n)
        zhz = self.z * self.b * self.z
        for i in range(n):
            for j in range(n):
                self.coupling_matrix[i, j] = zhz[i, j]
                self.coupling_matrix[i, n + j] = self.z[i, j]
                self.coupling_matrix[n + i, j] = self.z[i, j]
        self.project_cache = {}

    def projection(self, function):
        key = id(function)
        if key not in self.project_cache:
            # Retain the function itself so Python cannot recycle its id.
            self.project_cache[key] = function, self.d.project(function)
        return self.project_cache[key][1]

    def exponent_action(self, function, derivative):
        projection = self.projection(function)
        matrix = self.z if derivative == 0 else self.exponential_derivatives[derivative]
        coefficients = [sum((matrix[i, j] * projection[j] for j in range(self.d.rank)), a(0))
                        for i in range(self.d.rank)]
        correction = self.d.polynomial(coefficients, "Ejet_" + str(derivative))
        return function + correction if derivative == 0 else correction

    def h_action(self, function, derivative):
        def evaluate(panel, r, analytic):
            epsilon, _, _ = elementary_jet(r, self.source_t, derivative, analytic)
            scalar = 2 * c(self.lapse) * math.factorial(derivative) * epsilon.values[derivative]
            return scalar * function(panel, r, analytic)
        return RadialFunction(evaluate, "hjet_" + str(derivative) + "(" + function.label + ")")

    def apply_derivative(self, function, derivative):
        terms = []
        for i in range(derivative + 1):
            for l in range(derivative - i + 1):
                k = derivative - i - l
                coefficient = math.factorial(derivative) // (math.factorial(i) * math.factorial(l) * math.factorial(k))
                term = self.exponent_action(self.h_action(self.exponent_action(function, k), l), i)
                terms.append(term.scaled(coefficient))
        result = terms[0]
        for term in terms[1:]:
            result = result + term
        return result

    def resolvent_bilinear(self, z, left, right, analytic=False):
        if not left.cutoff_supported or not right.cutoff_supported:
            raise Unresolved("Dunford vectors require an explicitly retained exterior free block")
        d, n = self.d, self.d.rank
        g0 = d.free_gram(z, self.source_t, self.lapse, analytic)
        g = acb_mat(2 * n, 2 * n)
        for i in range(n):
            for j in range(n):
                unit = int(i == j)
                g[i, j] = g0[i, j]
                g[i, n + j] = g[n + i, j] = unit + z * g0[i, j]
                g[n + i, n + j] = self.b[i, j] + z * unit + z**2 * g0[i, j]
        cc = as_complex_matrix(self.coupling_matrix)
        inverse, _ = verified_inverse(identity(2 * n, True) + cc * g, d.work, analytic=analytic)
        def radial(left_function, right_function):
            total = c(0)
            for p, panel in enumerate(d.panels):
                def integrand(r, radial_analytic):
                    denominator = free_multiplier(r, self.source_t, self.lapse, radial_analytic) - z
                    if denominator.contains(0):
                        raise AnalyticDomain("Free bilinear denominator touches zero")
                    return left_function(p, r, radial_analytic) * right_function(p, r, radial_analytic) / denominator
                total += d.integrals.integrate(integrand, c(panel.left), c(panel.right), "continuous R0 bilinear")
            return total
        base = radial(left, right)
        pl, pr = self.projection(left), self.projection(right)
        row, column = acb_mat(1, 2 * n), acb_mat(2 * n, 1)
        for i in range(n):
            li, ri = radial(left, d.basis(i)), radial(d.basis(i), right)
            row[0, i], column[i, 0] = li, ri
            row[0, n + i], column[n + i, 0] = pl[i] + z * li, pr[i] + z * ri
        return base - (row * inverse * cc * column)[0, 0]


def spectral_data(model, tolerance, target_count=2):
    """All positive directions, endpoint inertia counts, and simple root boxes."""
    d, n = model.d, model.d.rank
    pos, neg = signed_ldl_inertia(model.matrices[0], d.work)
    if pos != n or neg:
        raise Unresolved("Not every retained M direction is certified strictly positive")
    exponential_difference, exp_metadata = certified_matrix_expm1(2 * model.gamma * model.matrices[0], d.work, d.tolerance)
    inverse_difference, inverse_residual = verified_inverse(exponential_difference, d.work)
    count_cache, count_records = {}, []

    def pencil(energy):
        g0 = d.free_gram(c(energy))
        real = arb_mat(n, n)
        for i in range(n):
            for j in range(n):
                if not g0[i, j].imag.contains(0):
                    raise ArithmeticError("Subcontinuum free Gram is not real")
                real[i, j] = a(energy) * g0[i, j].real - inverse_difference[i, j]
        return real

    def count(energy):
        if not lower(a(energy)) > 0 or not upper(a(energy)) < 2:
            raise Unresolved("Count endpoint is outside the positive subcontinuum interval")
        key = dyadic(a(energy)) if a(energy).is_exact() else None
        cache_key = json.dumps(key, sort_keys=True) if key else None
        if cache_key is not None and cache_key in count_cache:
            return count_cache[cache_key]
        positive_count, negative_count = signed_ldl_inertia(pencil(energy), d.work)
        if positive_count + negative_count != n:
            raise Unresolved("Count omitted a retained direction")
        record = (a(energy), positive_count)
        for previous_energy, previous_count in count_records:
            if upper(previous_energy) < lower(record[0]) and previous_count > positive_count:
                raise ArithmeticError("Birman--Schwinger certified counts violate monotonicity")
            if upper(record[0]) < lower(previous_energy) and positive_count > previous_count:
                raise ArithmeticError("Birman--Schwinger certified counts violate monotonicity")
        count_records.append(record)
        if cache_key is not None:
            count_cache[cache_key] = positive_count
        return positive_count

    lower_bound = positive(2 * (-2 * model.gamma * norm_bound(model.matrices[0])).exp(), "Strict positive H_Q ground bound")
    low = lower(lower_bound) / 2
    if count(low) != 0:
        raise ArithmeticError("Certified positive lower bound conflicts with endpoint count")
    high = a(F(1, 4))
    while count(high) < target_count:
        high = (high + 2) / 2
        d.work.check()
        if not high < 2:
            raise Unresolved("No required count endpoint before breakup")
    intervals = []
    for target in range(1, target_count + 1):
        lo, hi = low, high
        for _ in range(8 * ctx.prec):
            middle = ((lo + hi) / 2).mid()
            if count(middle) >= target:
                hi = middle
            else:
                lo = middle
            if upper(hi - lo) < lower(a(tolerance)) and count(lo) == target - 1 and count(hi) == target:
                break
        else:
            raise Unresolved("Root isolation exhausted precision, or simple count jump not proved")
        intervals.append(interval_hull(lo, hi))
    if len(intervals) == 2 and not upper(intervals[0]) < lower(intervals[1]):
        raise Unresolved("H_Q simple level intervals overlap")
    # Enlarge the count-two guard to provide a useful certified distance
    # from the first excited state to every higher level and breakup.
    guard = upper(intervals[-1])
    probe = (guard + 2) / 2
    for _ in range(8 * ctx.prec):
        if count(probe) == target_count:
            guard = probe
            break
        probe = (guard + probe) / 2
    if not lower(guard - upper(intervals[-1])) > 0:
        raise Unresolved("Last isolated level separation from higher spectrum unresolved")
    metadata = {"positive_retained_directions": pos, "exp2gammaM_minus_I": exp_metadata,
                "inverse_Neumann_residual_bound": encoded(inverse_residual),
                "endpoint_counts": [{"E": encoded(energy), "count": value} for energy, value in count_records],
                "higher_spectrum_guard_count": target_count,
                "higher_spectrum_count_guard": encoded(guard), "strict_HQ_ground_bound": encoded(lower_bound)}
    return intervals, guard, pencil, metadata


def eigenfunction(model, energy, pencil, name):
    """Enclose the actual simple H_Q eigenvector throughout its root interval.

    Endpoint counts prove existence and multiplicity.  An invertible
    complementary minor over the entire energy interval lets a_p=1 fix
    the exact nullvector; the remaining row then follows at the true root.
    No midpoint eigenvector is accepted as a normalized state.
    """
    d, n = model.d, model.d.rank
    matrix = pencil(energy)
    coefficients = None
    last = "no invertible complementary minor"
    for selected in range(n):
        indices = [i for i in range(n) if i != selected]
        minor, column = arb_mat(n - 1, n - 1), arb_mat(n - 1, 1)
        for i, row in enumerate(indices):
            column[i, 0] = -matrix[row, selected]
            for j, col in enumerate(indices):
                minor[i, j] = matrix[row, col]
        try:
            inverse, _ = verified_inverse(minor, d.work)
        except Unresolved as error:
            if isinstance(error, EngineeringCap):
                raise
            last = str(error)
            continue
        solved = inverse * column
        coefficients = [a(0)] * n
        coefficients[selected] = a(1)
        for i, row in enumerate(indices):
            coefficients[row] = solved[i, 0]
        break
    if coefficients is None:
        raise Unresolved(name + ": " + last)
    va = d.polynomial(coefficients, name + "_Va")
    def y_value(panel, r, analytic):
        denominator = free_multiplier(r, analytic=analytic) - c(energy)
        if denominator.contains(0):
            raise AnalyticDomain("Continuous bound-function denominator touches zero")
        # E>0 is separately certified at the true root.  Cancel the common
        # positive E before normalization: y_bar=y/E=R0(E)V a.  This leaves
        # the normalized state/sign unchanged and needs no relative E accuracy.
        return va(panel, r, analytic) / denominator
    y = RadialFunction(y_value, name + "_y")
    plus, metadata = certified_matrix_expm1(model.gamma * model.matrices[0], d.work, d.tolerance)
    py = d.project(y)
    correction_coefficients = [sum((plus[i, j] * py[j] for j in range(n)), a(0)) for i in range(n)]
    correction = d.polynomial(correction_coefficients, name + "_plus")
    function = y + correction
    norm2 = positive(d.inner(function, function), name + " continuous norm squared")
    normalization = 1 / norm2.sqrt()
    function = function.scaled(normalization)
    overlap = d.inner(d.trial(0), function)
    if upper(overlap) < 0:
        function = function.scaled(-1)
        normalization = -normalization
        overlap = -overlap
    positive(overlap, name + " trial/readout overlap")
    normalized_norm = positive(d.inner(function, function), name + " normalized radial norm enclosure")
    if not normalized_norm.contains(1):
        raise Unresolved(name + " continuous normalization identity not enclosed")
    return function, {"invertible_minor_fixed_component": selected, "continuous_norm_squared_before": encoded(norm2),
                      "positive_trial_overlap": encoded(overlap), "plus_exponential": metadata,
                      "continuous_normalization_enclosure": encoded(normalized_norm),
                      "actual_HQ_eigenvector_family_enclosed": True,
                      "continuous_representation": {
                          "formula": "normalization*(sum(a_i*v_i)/(2*sqrt(1+r^2)-E)+sum(b_i*v_i)) on[0,L], zero outside",
                          "positive_common_E_factor_cancelled_before_normalization": True,
                          "E": encoded(energy), "a": [encoded(value) for value in coefficients],
                          "b": [encoded(value) for value in correction_coefficients],
                          "normalization": encoded(normalization),
                          "panel_endpoints": [[str(panel.left), str(panel.right)] for panel in d.panels],
                          "Legendre_order_per_panel": d.order}}


def expanded_levels(intervals, model_bounds, gamma, matrix):
    error = model_bounds["H_errors"][0]
    tnorm = upper(norm_bound(matrix) + model_bounds["T_errors_total"])
    original_positive = positive(2 * (-2 * upper(gamma) * tnorm).exp(), "Strict original-H ground positivity")
    ground_lo = max(lower(original_positive), lower(intervals[0]) - error)
    ground_hi = min(upper(intervals[0]) + error, a(F(1, 4)))
    if lower(ground_lo) > upper(a(F(1, 4))):
        raise ArithmeticError("Original ground count interval conflicts with fixed binding theorem E_g<=1/4")
    ground = interval_hull(ground_lo, ground_hi)
    excited = widen(intervals[1], error)
    if not lower(excited) > upper(ground) or not upper(excited) < 2:
        raise Unresolved("Full-operator expanded levels/gap/breakup are not separated")
    gap = positive(excited - ground, "Expanded original-H gap")
    return ground, excited, gap, original_positive


def certify_ground(model, ground_vector, energy_hq, ground_interval, next_interval, model_bounds, metadata):
    d = model.d
    residual_function = model.apply_derivative(ground_vector, 0) - ground_vector.scaled(energy_hq)
    numerical_residual, residual_certificate = d.norm(residual_function, certificate=True)
    residual_categories = {name: values[0] for name, values in model_bounds["categories"].items()}
    residual_categories["scalar_quadrature"] = numerical_residual
    residual_categories["matrix_arithmetic"] = numerical_residual
    residual_categories["spectral_energy_enclosure"] = upper(energy_hq) - lower(energy_hq)
    # Each entire directed numerical residual is conservatively allocated
    # once to scalar quadrature and again to matrix arithmetic.  Both are
    # added here, along with the explicit energy-enclosure contribution.
    full_residual = upper(sum(residual_categories.values(), a(0)))
    reference_energy_upper = max(upper(ground_interval), upper(energy_hq))
    separation = positive(lower(next_interval) - reference_energy_upper,
                          "Ground residual-reference distance to every other full-H level")
    ratio = upper(full_residual / separation)
    if not ratio < a(F(1, 2)):
        raise Unresolved("Ground residual/separation angle gate unresolved")
    vector_error = upper(2 * ratio)
    trial_norm = upper((d.inner(d.trial(0), d.trial(0)) + d.trial_tail(0)).sqrt())
    overlap = d.inner(d.trial(0), ground_vector)
    sign_margin = positive(lower(overlap) - trial_norm * vector_error, "Original ground declared sign margin")
    metadata.update({"full_continuous_HQ_residual_upper": encoded(numerical_residual),
                     "original_full_operator_residual_upper": encoded(full_residual),
                     "original_spectral_separation_lower": encoded(separation),
                     "residual_reference_energy_HQ_enclosure": encoded(energy_hq),
                     "separation_subtracts_both_original_interval_and_residual_reference": True,
                     "phase_aligned_vector_error_upper": encoded(vector_error),
                     "original_declared_sign_margin_lower": encoded(sign_margin),
                     "residual_error_allocation": {name: encoded(value) for name, value in residual_categories.items()},
                     "joint_numerical_residual_certificate": residual_certificate,
                     "residual_allocation_explanation": "joint numerical norm enclosure assigned twice conservatively, once to scalar quadrature and once to matrix arithmetic; these are duplicate allocations, not independent component estimates",
                     "normalized_actual_HQ_family": True,
                     "reference_trial_is_q_alpha_f": True})
    return vector_error, metadata


def expanded_ground(interval, guard, model_bounds, gamma, matrix):
    error = model_bounds["H_errors"][0]
    norm_t = upper(norm_bound(matrix) + model_bounds["T_errors_total"])
    original_positive = positive(2 * (-2 * upper(gamma) * norm_t).exp(), "Strict original-H ground positivity")
    ground_lo = max(lower(original_positive), lower(interval) - error)
    if lower(ground_lo) > upper(a(F(1, 4))):
        raise ArithmeticError("Original ground count interval conflicts with fixed binding theorem E_g<=1/4")
    ground = interval_hull(ground_lo, min(upper(interval) + error, a(F(1, 4))))
    other_lower = positive(lower(guard) - error, "Ground-stage lower bound on every other original-H level")
    positive(other_lower - upper(ground), "Ground-stage expanded count separation")
    return ground, interval_hull(other_lower, other_lower), original_positive


def certify_excited_and_orthogonality(model, ground_vector, excited_vector, intervals, full_levels, guard,
                                      model_bounds, metadata):
    d = model.d
    cross = d.inner(ground_vector, excited_vector)
    if not cross.contains(0):
        raise ArithmeticError("Certified distinct H_Q eigenfamilies lost exact orthogonality")
    orthogonalized = excited_vector - ground_vector.scaled(cross)
    normalization = positive(d.inner(orthogonalized, orthogonalized), "Excited Gram--Schmidt norm").sqrt()
    orthogonalized = orthogonalized.scaled(1 / normalization)
    vector_change = d.norm(orthogonalized - excited_vector)
    normalized_cross = d.inner(ground_vector, orthogonalized)
    normalized_norm = d.inner(orthogonalized, orthogonalized)
    if not normalized_cross.contains(0) or not normalized_norm.contains(1):
        raise Unresolved("Continuous mutual orthonormalization identities unresolved")
    error = model_bounds["H_errors"][0]
    higher_lower = lower(guard) - error
    separation = positive(min(lower(full_levels[1]) - upper(full_levels[0]),
                              higher_lower - upper(full_levels[1]), 2 - upper(full_levels[1])),
                          "Excited distance to all other original-H spectrum")
    residual_function = model.apply_derivative(orthogonalized, 0) - orthogonalized.scaled(intervals[1])
    numerical_residual = d.norm(residual_function)
    full_residual = upper(numerical_residual + error)
    ratio = upper(full_residual / separation)
    if not ratio < a(F(1, 2)):
        raise Unresolved("Excited residual/separation angle gate unresolved")
    vector_error = upper(2 * ratio + vector_change)
    trial_norm = upper((d.inner(d.trial(0), d.trial(0)) + d.trial_tail(0)).sqrt())
    overlap = d.inner(d.trial(0), orthogonalized)
    margin = positive(lower(overlap) - trial_norm * vector_error,
                      "Nonzero first excited f-projection and declared original-H sign")
    metadata.update({"continuous_cross_before": encoded(cross), "continuous_cross_after": encoded(normalized_cross),
                     "continuous_norm_after": encoded(normalized_norm), "orthonormalization_vector_change_upper": encoded(vector_change),
                     "full_continuous_HQ_residual_upper": encoded(numerical_residual),
                     "original_full_operator_residual_upper": encoded(full_residual),
                     "distance_to_every_other_original_level_lower": encoded(separation),
                     "phase_aligned_vector_error_upper": encoded(vector_error),
                     "first_excited_f_projection_sign_margin_lower": encoded(margin),
                     "simple_first_excited_projection_certified": True})
    return orthogonalized, vector_error, metadata


def f_time(x, tau):
    # sinc is entire and has its documented removable value at zero.
    argument = c(x) if isinstance(x, arb) else x
    return c(tau) * (-c(0, 1) * argument * c(tau) / 2).exp() * (argument * c(tau) / 2).sinc()


def j_time(x, y, tau, work, tolerance):
    difference = x - y
    if not difference.contains(0):
        return (f_time(y, tau) - f_time(x, tau)) / (c(0, 1) * difference)
    # Entire triangular integral at coincidences.  Its homogeneous term
    # is tau^2 (-i tau)^n sum_{k=0}^n x^k y^(n-k)/(n+2)!.
    # Taylor's integrand remainder gives area*exp(A)*A^(p+1)/(p+1)!.
    maximum = upper(abs_upper(c(tau)) * max(abs_upper(x), abs_upper(y)))
    if maximum > ctx.prec:
        # A very wide analytic cell is not evaluated by a huge, cancelling
        # Taylor series.  Returning non-finite permits the integrator to
        # subdivide; it does not remove the coincidence or its spectrum.
        raise AnalyticDomain("Entire J coincidence cell exceeds declared series work range")
    result, remainder, degree = c(0), a(1), -1
    while degree < 0 or not upper(remainder) < lower(a(tolerance)):
        degree += 1
        work.check(degree + 1)
        polynomial = sum((x**k * y**(degree - k) for k in range(degree + 1)), c(0))
        result += c(tau)**2 * (-c(0, 1) * c(tau))**degree * polynomial / math.factorial(degree + 2)
        remainder = upper(a(tau)**2 / 2 * maximum.exp() * maximum**(degree + 1) / math.factorial(degree + 1))
        if degree > 16 * ctx.prec:
            raise Unresolved("Entire J Taylor certificate exhausted degree cap")
    return widen(result, remainder)


def dunford(model, left, right, entire, tau, source_radius=0):
    """All four CCW segments, full analytic Woodbury guards, exact free h."""
    d = model.d
    if not left.cutoff_supported or not right.cutoff_supported:
        raise Unresolved("Exterior free block not supplied for Dunford vector")
    height = min(a(F(1, 4)), 1 / positive(a(tau), "Dunford duration"))
    if source_radius:
        right_bound = (1 + a(source_radius)) * 2 * (1 + (2 * a(source_radius)).exp() * d.cutoff**2).sqrt()
    else:
        right_bound = 2 * a(1 + d.cutoff**2).sqrt()
    right_endpoint = right_bound + 1
    points = [acb(-1, -height), acb(right_endpoint, -height),
              acb(right_endpoint, height), acb(-1, height)]
    values, radius = [], a(0)
    for i in range(4):
        def integrand(z, analytic):
            # Every nested integral guards its radial roots/denominators
            # and the finite inverse is proved across this whole z ball.
            value = entire(z) * model.resolvent_bilinear(z, left, right, analytic)
            if not value.is_finite():
                raise AnalyticDomain("Complete Dunford integrand non-finite")
            return value
        value = d.integrals.integrate(integrand, points[i], points[(i + 1) % 4], "Dunford segment " + str(i))
        values.append(value)
        radius += upper(value.rad()) / (2 * lower(arb.pi()))
    result = -sum(values, c(0)) / (2 * acb.pi() * c(0, 1))
    return result, {"CCW_segment_enclosures": [encoded(value) for value in values],
                    "summed_segment_error_radius_after_1_over_2pi": encoded(upper(radius)),
                    "contour_height": encoded(height), "right_endpoint": encoded(right_endpoint),
                    "exact_free_multiplier_retained": True, "all_vectors_cutoff_supported": True}


def coefficient_model_errors(bounds, vector_errors, gap, tau):
    astar, bstar = bounds["H_norms"][1], bounds["H_norms"][2]
    eps = upper(sum(vector_errors, a(0)))
    eps_o, eps_xi = eps, upper(eps / a(2).sqrt())
    duration, time_radius = upper(tau), upper(tau.rad())
    errors = {}
    for name, hj in bounds["categories"].items():
        delta_a, delta_b = upper(2 * hj[1]), upper(2 * hj[2])
        commutator = upper(4 * astar * hj[1] * duration**2 + 4 * astar**2 * hj[0] * duration**3)
        errors[name] = [upper(delta_a / lower(gap)),
                        upper(delta_b / (2 * lower(gap)) + 8 / lower(arb.pi()) * commutator)]
    expectation_a, expectation_b = upper(2 * astar * eps), upper(2 * bstar * eps)
    commutator = upper(2 * astar**2 * duration**2 * eps_o
                       + 4 * astar**2 * duration**2 * eps_xi
                       + 4 * astar**2 * duration * time_radius)
    errors["spectral_vectors_gap_time"] = [upper(expectation_a / lower(gap)),
                                          upper(expectation_b / (2 * lower(gap)) + 8 / lower(arb.pi()) * commutator)]
    return errors


def spectral_coefficients(model, vectors, energies_hq, full_gap, tau, bounds, vector_errors):
    d = model.d
    a_images = [model.apply_derivative(vector, 1) for vector in vectors]
    b_images = [model.apply_derivative(vector, 2) for vector in vectors]
    expectation_a = [[d.inner(vectors[i], a_images[j]) for j in range(2)] for i in range(2)]
    expectation_b = [d.inner(vectors[i], b_images[i]) for i in range(2)]
    if not expectation_a[0][1].overlaps(expectation_a[1][0]):
        raise ArithmeticError("Full noncommuting A lost self-adjoint expectation symmetry")
    delta_a, delta_b = expectation_a[1][1] - expectation_a[0][0], expectation_b[1] - expectation_b[0]
    bi, ri, contour_metadata = [], [], []
    for i in range(2):
        bi.append(-c(0, 1) * sum((c(expectation_a[i][j]) * f_time(c(energies_hq[j] - energies_hq[i]), tau)
                                 for j in range(2)), c(0)))
        summation = c(0)
        for j in range(2):
            def entire(z, i=i, j=j):
                return j_time(z - c(energies_hq[i]), c(energies_hq[j] - energies_hq[i]), tau, d.work, d.tolerance)
            value, metadata = dunford(model, a_images[i], a_images[j], entire, tau)
            summation += value
            contour_metadata.append({"i": i, "j": j, **metadata})
        ri.append(summation)
    base_a1 = delta_a / full_gap
    base_a2 = delta_b / (2 * full_gap) + 4 / arb.pi() * (ri[1] - ri[0] + (bi[1]**2 - bi[0]**2) / 2).imag
    errors = coefficient_model_errors(bounds, vector_errors, full_gap, tau)
    # All directed numerical radii already occur in base_a1/base_a2.
    # To give a conservative explicit allocation, assign that whole radius
    # both to matrix arithmetic and to calculus quadrature, and add a
    # second copy once.  No subtraction or undocumented radius split occurs.
    numerical = [upper(base_a1.rad()), upper(base_a2.rad())]
    errors["matrix_arithmetic"] = numerical
    errors["functional_calculus_quadrature"] = numerical
    model_sums = [upper(sum((values[i] for name, values in errors.items()
                            if name not in ("matrix_arithmetic", "functional_calculus_quadrature")), a(0)))
                  for i in range(2)]
    a1 = widen(base_a1, model_sums[0] + numerical[0])
    a2 = widen(base_a2, model_sums[1] + numerical[1])
    metadata = {"A_matrix": [[encoded(value) for value in row] for row in expectation_a],
                "B_diagonal": [encoded(value) for value in expectation_b],
                "b": [encoded(value) for value in bi], "R_full_intermediate_spectrum": [encoded(value) for value in ri],
                "contours": contour_metadata,
                "coefficient_radius_allocation": {name: [encoded(value) for value in values] for name, values in errors.items()},
                "numerical_allocation": "entire directed numerical radius assigned twice conservatively; second copy explicitly added",
                "Omega_times_tau_is_exact_pi_over_4": True}
    return a1, a2, metadata


def coherence(model, vectors, tau, model_error, vector_errors, source_radius):
    preparation = (vectors[0] + vectors[1]).scaled(1 / a(2).sqrt())
    amplitudes, metadata = [], []
    for vector in vectors:
        value, contour = dunford(model, vector, preparation, lambda z: (-c(0, 1) * z * c(tau)).exp(), tau, source_radius)
        amplitudes.append(value)
        metadata.append(contour)
    raw = 2 * amplitudes[0].conjugate() * amplitudes[1]
    eps = upper(sum(vector_errors, a(0)))
    model_radius = upper(2 * eps + 4 * eps / a(2).sqrt() + 4 * upper(tau) * model_error)
    value = widen(raw, model_radius)
    if not lower(abs(value)) >= a(F(1, 2)):
        raise Unresolved("Certified coherence signal lower bound1/2 not reached")
    rotated = value * (c(0, 1) * acb.pi() / 4).exp()
    if not lower(rotated.real) > 0:
        raise Unresolved("Ramsey phase branch through exp(-i*pi/4) unresolved")
    logarithm = rotated.log(analytic=True)
    if not logarithm.is_finite():
        raise Unresolved("Ramsey analytic logarithm branch unresolved")
    nop = 1 - 4 / arb.pi() * logarithm.imag
    return nop, {"coherence": encoded(value), "model_and_vector_coherence_radius": encoded(model_radius),
                 "phase_branch_positive_real_margin": encoded(lower(rotated.real)),
                 "contours": metadata}


def phase_controls(model, vectors, tau, gap, bounds, vector_errors, a1, a2):
    d = model.d
    hats = bounds["mixed_norms"]
    delta = lower(min(a(T0), 1 / (16 * upper(tau) * hats[1])))
    positive(delta, "Predeclared finite-source interval radius")
    tt = upper(tau)
    m1 = upper(4 * tt * hats[1])
    m2 = upper(4 * tt * hats[2] + 8 * tt**2 * hats[1]**2)
    m3 = upper(4 * tt * hats[3] + 24 * tt**2 * hats[1] * hats[2] + 16 * tt**3 * hats[1]**3)
    m4 = upper(4 * tt * hats[4] + 32 * tt**2 * hats[1] * hats[3] + 24 * tt**2 * hats[2]**2
               + 96 * tt**3 * hats[1]**2 * hats[2] + 32 * tt**4 * hats[1]**4)
    l3 = upper(4 / lower(arb.pi()) * (2 * m3 + 12 * m2 * m1 + 16 * m1**3))
    l4 = upper(4 / lower(arb.pi()) * (2 * m4 + 16 * m3 * m1 + 12 * m2**2
                                      + 96 * m2 * m1**2 + 96 * m1**4))
    nop0, baseline = coherence(model, vectors, tau, bounds["H_errors"][0], vector_errors, delta)
    if not nop0.contains(1):
        raise Unresolved("Reference Ramsey identity Nop(0)=1 unresolved")
    output = {"delta": encoded(delta), "L3": encoded(l3), "L4": encoded(l4), "baseline": baseline, "rows": []}
    for divisor in (2, 4, 8):
        h = (lower(delta) / divisor).mid()  # Fixed directed dyadic no larger than allowed h.
        positive(h, "Finite-source dyadic step")
        values, rows = [], []
        for sign in (-1, 1):
            source = sign * h
            matrices = d.kernel_matrices(source, degree=0)
            changed = FiniteModel(d, matrices, model.gamma, degree=0, source_t=source, lapse=1 - source)
            nop, metadata = coherence(changed, vectors, tau, bounds["mixed_errors"][0], vector_errors, delta)
            values.append(nop - (1 - source))
            rows.append({"lambda": encoded(source), **metadata})
        baseline_f = nop0 - 1
        first = widen((values[1] - values[0]) / (2 * h), l3 * h**2 / 6)
        second = widen((values[1] + values[0] - 2 * baseline_f) / (2 * h**2), l4 * h**2 / 24)
        if not first.overlaps(a1) or not second.overlaps(a2 - a1):
            raise Unresolved("Direct coefficient and independent finite-source phase intervals fail overlap")
        # Broad intervals are not an observed agreement control.  Each
        # independently evaluated estimator must meet the coefficient target.
        if not (upper(width(first)) <= lower(a(F(1, 10**4))) and upper(width(second)) <= lower(a(F(2, 10**4)))):
            raise Unresolved("Finite-source phase control intervals exceed coefficient-width targets")
        output["rows"].append({"h": encoded(h), "first_a1": encoded(first), "second_a2_minus_a1": encoded(second),
                               "first_truncation_bound": encoded(upper(l3 * h**2 / 6)),
                               "second_truncation_bound": encoded(upper(l4 * h**2 / 24)), "source_rows": rows})
    # Independently integrate a lapse-only model, with the same vectors/time.
    lapse = 1 - lower(delta) / 2
    lapse_model = FiniteModel(d, model.matrices[:1], model.gamma, degree=0, source_t=0, lapse=lapse)
    nop_lapse, metadata = coherence(lapse_model, vectors, tau, upper(abs(lapse)) * bounds["H_errors"][0], vector_errors, delta)
    discrepancy = nop_lapse - lapse
    if not discrepancy.contains(0) or not upper(width(discrepancy)) <= lower(a(F(1, 10**4))):
        raise Unresolved("Constant-lapse Nop=N direct contour control unresolved")
    output["constant_lapse"] = {"N": encoded(lapse), "Nop": encoded(nop_lapse), "discrepancy": encoded(discrepancy), **metadata}
    return output


def measure_controls(d):
    """Independent Cartesian eight-preimage coordinates, no sphere replacement."""
    rows = []
    for power in (0, 1, 2):
        def outer(x, analytic_x):
            def middle(y, analytic_y):
                def inner(z, analytic_z):
                    analytic = analytic_x or analytic_y or analytic_z
                    radius2 = x**2 + y**2 + z**2
                    epsilon = guarded_root(1 + radius2, analytic)
                    f = (-radius2).exp() * (epsilon + 1) / (2 * epsilon)
                    jacobian = c(1)
                    for variable in (x, y, z):
                        jacobian /= guarded_root(1 - a(ALPHA)**2 * variable**2, analytic)
                    return f**2 * epsilon**power * jacobian
                return d.integrals.integrate(inner, 0, d.cutoff, "Cartesian inner trial Gram")
            return d.integrals.integrate(middle, 0, d.cutoff, "Cartesian middle trial Gram")
        cartesian_complex = 8 * d.integrals.integrate(outer, 0, d.cutoff, "Cartesian outer trial Gram") / (2 * acb.pi())**3
        if not cartesian_complex.imag.contains(0):
            raise ArithmeticError("Cartesian Gram lost real character")
        cartesian = cartesian_complex.real
        radial = d.inner(d.trial(0), d.trial(power))
        tail = d.trial_tail(power)
        arithmetic_radius = upper(cartesian.rad() + radial.rad())
        discrepancy = cartesian - radial
        allowed = interval_hull(-arithmetic_radius, tail + arithmetic_radius)
        if not discrepancy.overlaps(allowed):
            raise Unresolved("Cartesian/radial eight-preimage Gram control fails positive-tail agreement")
        scale = max(a(1), abs_upper(radial))
        if not arithmetic_radius < a(F(1, 10**12)) * scale:
            raise Unresolved("Cartesian normalization control arithmetic radius remains unresolved")
        rows.append({"epsilon_power": power, "Cartesian_cube": encoded(cartesian), "radial_sphere": encoded(radial),
                     "cube_minus_sphere": encoded(discrepancy), "positive_Gaussian_tail_upper": encoded(tail),
                     "directed_integration_radius_sum": encoded(arithmetic_radius)})
    return {"angular_first_three_exact": [str(value) for value in d.coefficients[:3]],
            "Cartesian_factor": "8/(2*pi)^3; octant symmetry after exact physical eight-preimage change of variables",
            "rows": rows}


def kernel_completion_controls(model, bounds):
    d = model.d
    samples = []
    for panel in d.panels:
        r = a((panel.left + panel.right) / 2)
        s = a(panel.left + panel.length / 3)
        for source in (-a(T0), a(0), a(T0)):
            kernel = kernel_jet(r, s, source, degree=0).values[0]
            er, es = (1 + (-2 * source).exp() * r**2).sqrt(), (1 + (-2 * source).exp() * s**2).sqrt()
            fr = (-(-2 * source).exp() * r**2).exp() * (er + 1) / (2 * er)
            fs = (-(-2 * source).exp() * s**2).exp() * (es + 1) / (2 * es)
            jr = (-(-2 * source).exp() * r**2).exp() * (er - 1) / (2 * er)
            js = (-(-2 * source).exp() * s**2).exp() * (es - 1) / (2 * es)
            independently_assembled = 2 * (-3 * source).exp() * (fr * fs + jr * js)
            discrepancy = (2 * er + 2 * es) * kernel - independently_assembled
            if not discrepancy.contains(0) or not abs_upper(discrepancy) < d.tolerance * 100:
                raise Unresolved("Independent pre-measure kernel identity control unresolved")
            samples.append(encoded(discrepancy))
    f = d.trial(0)
    def lower_leg(panel, r, analytic):
        epsilon = guarded_root(1 + r**2, analytic)
        return absorption(r, d.coefficients, analytic) * (-r**2).exp() * (epsilon - 1) / (2 * epsilon)
    jleg = RadialFunction(lower_leg, "q_alpha*j")
    rows = []
    def tq_action(function):
        projection = d.project(function)
        coefficients = [sum((model.matrices[0][i, j] * projection[j] for j in range(d.rank)), a(0)) for i in range(d.rank)]
        return d.polynomial(coefficients, "TQ_action")
    emax = 2 * (1 + 3 / a(ALPHA)**2).sqrt()
    terr = bounds["T_errors_total"]  # This coupling-derivative control is literally at t=0.
    for trial in (d.trial(0), d.trial(1)):
        # Exact coupling derivative of the projected completion at zero:
        # -TQ h - h TQ.  Neither h nor the trial is projected first.
        completed = tq_action(model.h_action(trial, 0)).scaled(-1) + model.h_action(tq_action(trial), 0).scaled(-1)
        overlap_f, overlap_j = d.inner(f, trial), d.inner(jleg, trial)
        bare_cutoff = f.scaled(2 * overlap_f) + jleg.scaled(2 * overlap_j)
        discrepancy_function = completed + bare_cutoff
        exterior_bare = upper(2 * d.trial_tail(0).sqrt() * (abs_upper(overlap_f) + abs_upper(overlap_j)))
        mismatch = upper(d.norm(discrepancy_function) + exterior_bare)
        analytic = upper(2 * emax * terr * d.norm(trial))
        if not mismatch <= analytic:
            raise Unresolved("Full coupling derivative/bare-potential operator-error control unresolved")
        rows.append({"trial": trial.label, "continuous_mismatch_plus_exterior_upper": encoded(mismatch),
                     "full_free_energy_operator_bound": encoded(analytic), "P_not_commuted_with_h": True})
    def trace_integrand(r, analytic):
        epsilon = guarded_root(1 + r**2, analytic)
        upper_leg = (-r**2).exp() * (epsilon + 1) / (2 * epsilon)
        lower_leg_value = (-r**2).exp() * (epsilon - 1) / (2 * epsilon)
        return absorption(r, d.coefficients, analytic)**2 * (upper_leg**2 + lower_leg_value**2) / (2 * epsilon)
    trace_low = sum((d.integrals.real(trace_integrand, panel.left, panel.right, "real T trace") for panel in d.panels), a(0))
    trace_tail = upper(1 / (16 * a(ALPHA)**3) * a(-2 * d.cutoff**2).exp())
    trace_full = upper(trace_low + trace_tail)
    real_tail_check = upper(trace_tail + 2 * (trace_full * trace_tail).sqrt())
    return {"kernel_identity_discrepancy_balls": samples, "coupling_derivative_controls": rows,
            "real_full_trace_upper": encoded(trace_full), "real_crossblock_tail_control_upper": encoded(real_tail_check),
            "real_tail_does_not_replace_complex_derivative_tail": True}


@dataclass(frozen=True)
class Configuration:
    cutoff: int
    order: int
    quadrature: int
    digits: int
    weight_order: int
    energy_digits: int

    def saved(self):
        return {"cutoff": self.cutoff, "Legendre_order_per_panel": self.order,
                "positive_Gauss_order_per_panel": self.quadrature,
                "precision_decimal_digits": self.digits, "weight_series_last_power": self.weight_order,
                "root_absolute_width_goal": "1e-" + str(self.energy_digits),
                "total_rank": len(panels_for(self.cutoff)) * self.order}


def intersect_enclosures(left, right, label):
    if left is None:
        return right
    lo, hi = max(lower(left), lower(right)), min(upper(left), upper(right))
    if lo > hi:
        raise Unresolved(label + " has an empty certified refinement intersection")
    return interval_hull(lo, hi)


def width(value):
    return upper(value) - lower(value)


def ground_preflight_refinement(bounds, compression_inputs):
    """GR1--GR7 engineering order choice after the unchanged H gate fails."""
    threshold = a(F(1, 10**13))
    numeric = {"full_H_error": bounds["H_errors"][0], "threshold": threshold,
               "tail_projection_allocation": bounds["categories"]["tail_projection"][0],
               "quadrature_allocation": bounds["categories"]["quadrature"][0],
               "coupling_allocation": bounds["categories"]["coupling"][0],
               "maximum_analytic_entry_error_bound": compression_inputs.get("maximum_analytic_entry_error_bound"),
               "maximum_raw_entry_numerical_radius": compression_inputs.get("maximum_raw_entry_numerical_radius")}
    valid = {name: value is not None and value.is_finite() and lower(value) >= 0
             for name, value in numeric.items()}
    reasons = [name + (" absent" if numeric[name] is None else " nonfinite or not certified nonnegative")
               for name in numeric if not valid[name]]
    completed = compression_inputs.get("current_compression_completed") is True
    if not completed:
        reasons.append("current compression completion diagnostic absent or false")
    boolean_names = ("raw_determinant_certified_positive", "full_determinant_certified_positive",
                     "analytic_bound_dominates_raw_radius_by16")
    for name in boolean_names:
        if type(compression_inputs.get(name)) is not bool:
            reasons.append(name + " absent or invalid")
    booleans_valid = all(type(compression_inputs.get(name)) is bool for name in boolean_names)
    original_rule = False
    if all(valid[name] for name in ("tail_projection_allocation", "quadrature_allocation", "coupling_allocation", "threshold")):
        other_categories = sum((values[0] for name, values in bounds["categories"].items()
                                if name != "tail_projection"), a(0))
        original_rule = (other_categories.is_finite() and lower(other_categories) >= 0
                         and lower(numeric["tail_projection_allocation"]) >= upper(threshold)
                         and lower(numeric["tail_projection_allocation"]) >= upper(other_categories))
    analytic_positive = valid["maximum_analytic_entry_error_bound"] and lower(numeric["maximum_analytic_entry_error_bound"]) > 0
    coupling_at_threshold = valid["coupling_allocation"] and lower(numeric["coupling_allocation"]) >= upper(threshold)
    new_rule = (all(valid.values()) and booleans_valid and completed
                and compression_inputs["raw_determinant_certified_positive"]
                and compression_inputs["full_determinant_certified_positive"]
                and compression_inputs["analytic_bound_dominates_raw_radius_by16"]
                and analytic_positive and coupling_at_threshold)
    advance = bool(original_rule or new_rule)
    record = {"rule": "failed_H_coupling_compression_v1", "full_H_gate_passed": False,
              **{name: encoded(value) if valid[name] else None for name, value in numeric.items()},
              "current_compression_completed": completed,
              **{name: compression_inputs.get(name) is True for name in boolean_names},
              "analytic_bound_strictly_positive": bool(analytic_positive),
              "coupling_allocation_at_least_threshold": bool(coupling_at_threshold),
              "original_projection_rule": bool(original_rule),
              "new_coupling_compression_rule": bool(new_rule), "advance_existing_order": advance,
              "decision": "advance_existing_NQ" if advance else "retain_existing_precision_fallback",
              "diagnostic_reason": "; ".join(reasons) if reasons else None}
    return advance, record


def direct_input_allocation_diagnostic(bounds, saved):
    """GT4--GT7 diagnostic; the full unchanged preflight remains decisive."""
    threshold = a(F(1, 10**13))
    dp, dq, dc = (bounds["categories"][name][0] for name in ("tail_projection", "quadrature", "coupling"))
    slack = threshold - dp - dq
    cq = bounds.get("TQ_reference_norm", bounds["TQ_complex_norm"])
    coefficient = upper(2 * bounds["H_norms"][0] * cq)
    budget, reason = None, None
    if not slack.is_finite() or not coefficient.is_finite() or not lower(coefficient) > 0:
        reason = "Slack or retained coupling coefficient is nonfinite or lacks a positive margin"
    elif lower(slack) > 0:
        candidate = lower(lower(slack) / coefficient)
        if candidate > 0:
            budget = encoded(candidate)
        else:
            reason = "Directed slack/coefficient division does not certify a positive mismatch budget"
    elif upper(slack) <= 0:
        reason = "Retained projection/quadrature allocations exhaust this gate even at zero mismatch; not a lower bound on actual error"
    else:
        reason = "The retained allocations supply no certified positive slack"
    saved.update({"actual_completion_allocation": {"tail_projection": encoded(dp), "quadrature": encoded(dq),
                                                   "coupling": encoded(dc), "full_H_error": encoded(bounds["H_errors"][0])},
                  "coupling_mismatch_upper": encoded(bounds["gamma_radius"]),
                  "strict_preflight_threshold": encoded(threshold), "slack_interval_GT6": encoded(slack),
                  "coupling_coefficient_upper": encoded(coefficient),
                  "sufficient_mismatch_budget_lower_GT7": budget, "mismatch_budget_unresolved_reason": reason,
                  "diagnostic_replaces_full_preflight": False,
                  "norm_scope": "reference_t0" if "TQ_reference_norm" in bounds else "original_source_window"})


def evaluate_configuration(configuration, work, shared_gamma, saved, stage, controls,
                           direct_input=None, shared_mu2=None):
    """A fully corrected point; ground export precedes any Ramsey work."""
    ctx.dps = configuration.digits
    tolerance = F(1, 10**max(64, configuration.digits // 2))
    saved.update({"configuration": configuration.saved(), "model_evaluation_attempted": True,
                  "model_evaluated": False,
                  "ground_point_certified": False, "ground_preparation_certified": False,
                  "ramsey_certified": False, "full_clock_certified": False,
                  "excited_vector_status": "unexecuted", "source_jets_status": "unexecuted",
                  "Ramsey_functional_calculus_status": "unexecuted"})
    d = Discretization(configuration.cutoff, configuration.order, configuration.quadrature,
                       configuration.weight_order, work, tolerance)
    reference_errors = None
    if stage == "ground":
        saved["ground_reference_error_certificate"] = {}
        reference_errors = ground_reference_bounds(d, saved["ground_reference_error_certificate"])
    # Ground mode computes only the reference M and exponential.  The
    # error formulas are evaluated only at order zero in this mode; no
    # source-jet matrix or Ramsey contour is built before the ground export.
    matrices0 = d.kernel_matrices(degree=0)
    saved["trial_compression"] = {}
    compression_refinement_inputs = {}
    if direct_input is None:
        mu2, gamma, compression_metadata = trial_compression(d, matrices0[0], saved["trial_compression"],
                                                            compression_refinement_inputs)
    else:
        if stage != "ground":
            raise ValueError("Direct compression is only a separately reviewed ground-stage input")
        saved["direct_compression_input"] = {"artifact": str(direct_input.path), "consumer_rechecks": [],
                                             "direct_compression_used_as_input": True,
                                             "direct_compression_is_ground_certificate": False}
        try:
            mu2, gamma = direct_input.recheck(work, saved["direct_compression_input"]["consumer_rechecks"])
        except ground_input.direct.InconsistentEnclosures as error:
            raise DirectInputRejected("Direct mathematical-input contradiction: " + str(error)) from error
        except (ground_input.InputUnresolved, ground_input.direct.Unresolved) as error:
            raise Unresolved("Direct mathematical-input recheck unresolved: " + str(error)) from error
        compression_metadata = {"phase": "current_projected_trial_compression_not_evaluated",
                                "current_compression_completed": False,
                                "maximum_analytic_entry_error_bound": None, "maximum_raw_entry_numerical_radius": None,
                                "raw_determinant_certified_positive": None, "full_determinant_certified_positive": None,
                                "analytic_bound_dominates_raw_radius_by16": None,
                                "reason": "Same-rule coupling supplied by validated direct input; no current projected-trial diagnostics fabricated"}
        compression_refinement_inputs["current_compression_completed"] = False
    if shared_mu2 is not None:
        before = shared_mu2[0]
        saved["same_exact_mu2_refinement_intersection"] = {"preceding": encoded(before) if before is not None else None,
                                                         "current": encoded(mu2), "intersection": None}
        try:
            shared_mu2[0] = intersect_enclosures(before, mu2, "Same exact physical mu2")
        except Unresolved as error:
            if direct_input is not None:
                raise DirectInputRejected(str(error)) from error
            raise
        mu2 = shared_mu2[0]
        saved["same_exact_mu2_refinement_intersection"]["intersection"] = encoded(mu2)
    saved["same_exact_gamma_refinement_intersection"] = {"preceding": encoded(shared_gamma[0]) if shared_gamma[0] is not None else None,
                                                       "current": encoded(gamma), "intersection": None}
    try:
        shared_gamma[0] = intersect_enclosures(shared_gamma[0], gamma, "Same exact physical gamma")
    except Unresolved as error:
        if direct_input is not None:
            raise DirectInputRejected(str(error)) from error
        raise
    gamma = shared_gamma[0]
    saved["same_exact_gamma_refinement_intersection"]["intersection"] = encoded(gamma)
    gamma_midpoint = positive(gamma.mid(), "Deterministic dyadic gamma midpoint")
    if not gamma_midpoint.is_exact():
        raise Unresolved("The coupling midpoint was not retained as an exact dyadic approximation")
    saved.update({"mu2": encoded(mu2), "gamma": encoded(gamma), "gamma_midpoint_exact": encoded(gamma_midpoint),
                  "gamma_midpoint_mismatch_upper": encoded(upper(abs(gamma - gamma_midpoint))),
                  "trial_compression": compression_metadata,
                  "interaction_error_bounds": {name: encoded(value) for name, value in d.errors.items()},
                  "same_exact_coupling_rule_across_all_refinements": True})
    bounds = completion_bounds(d, gamma, gamma_midpoint, degree=0, reference_errors=reference_errors)
    if reference_errors is not None:
        reference_errors["record"].update({"T_original_complex_window_norm_upper": encoded(bounds["T_complex_norm"]),
                                           "TQ_original_complex_window_norm_upper": encoded(bounds["TQ_complex_norm"]),
                                           "TQ_reference_norm_upper_GE11": encoded(bounds["TQ_reference_norm"]),
                                           "complete_reference_H_error_upper_GE12": encoded(bounds["H_errors"][0]),
                                           "same_reference_object_used_for_preflight_spectral_residual_vector_export": True})
    if direct_input is not None:
        direct_input_allocation_diagnostic(bounds, saved["direct_compression_input"])
    saved["ground_operator_error_allocation"] = {name: encoded(values[0]) for name, values in bounds["categories"].items()}
    saved["ground_full_operator_norm_error_upper"] = encoded(bounds["H_errors"][0])
    if not bounds["H_errors"][0] < a(F(1, 10**13)):
        advance, refinement_record = ground_preflight_refinement(bounds, compression_refinement_inputs)
        saved["ground_preflight_refinement"] = refinement_record
        if advance:
            raise RefineOrder("Failed ground H preflight meets the declared existing-order choice; no impossibility verdict")
        raise Unresolved("Analytic operator error needs projection/cutoff refinement for the ground energy target")
    model = FiniteModel(d, matrices0, gamma_midpoint, degree=0)
    saved["ground_negative_exponential_certificates"] = model.exp_metadata
    saved["reference_interaction_matrix_enclosure_radii"] = [encoded(value) for value in d.matrix_radii]
    ground_intervals, ground_guard, ground_pencil, ground_count_metadata = spectral_data(
        model, F(1, 10**configuration.energy_digits), target_count=1)
    saved["ground_counts"] = ground_count_metadata
    eg, other_lower, positive_lower = expanded_ground(ground_intervals[0], ground_guard, bounds, gamma, matrices0[0])
    saved.update({"E_g": encoded(eg), "strict_original_ground_positive_bound": encoded(positive_lower),
                  "fixed_binding_theorem_ground_upper_bound": "1/4",
                  "binding_theorem_checked_after_independent_BS_isolation": True})
    g, g_metadata = eigenfunction(model, ground_intervals[0], ground_pencil, "ground")
    eg_vector, g_metadata = certify_ground(model, g, ground_intervals[0], eg, other_lower, bounds, g_metadata)
    saved["ground_radial_function"] = g_metadata
    if not (upper(width(eg)) <= lower(a(F(1, 10**12))) and upper(eg_vector) <= lower(a(F(1, 10**8)))):
        raise Unresolved("Ground energy width or phase-aligned vector error target not reached")
    saved["ground_point_certified"] = True
    saved["ground_export"] = {
        "radial_chart": "L2(nu_alpha), alpha=1/1000000; implemented chart L2(dr)",
        "radial_function": g_metadata["continuous_representation"],
        "undo_measure_absorption": "u_E(r)=u_E_dr(r)/q_alpha(r), almost everywhere; q_alpha=r*sqrt(W(alpha^2*r^2))/(sqrt(2)*pi)",
        "angular_weight_coefficients_exact": [str(value) for value in d.coefficients],
        "angular_weight_remainder_bound": "abs(z)^(K+1)/(1-abs(z)), K=len(coefficients)-1, z=alpha^2*r^2",
        "full_momentum_embedding": "u_full(k)=u_E(sqrt(sum_i sin(alpha*k_i)^2)/alpha)*chi_s/sqrt(8)",
        "full_momentum_measure": "d^3k/(2*pi)^3; pushforward=8*nu_alpha",
        "full_embedding_is_isometric": True, "full_embedding_phase_aligned_norm_error_upper": encoded(eg_vector),
        "interval_coefficients_enclose_actual_correlated_normalized_HQ_ground": True,
        "arbitrary_componentwise_midpoint_is_not_a_certified_state": True,
        "COM_response_or_triplet_gap_certified": False}
    if reference_errors is not None:
        saved["ground_export"]["reference_only_operator_error_certificate"] = reference_errors["record"]
        saved["ground_export"]["original_full_operator_error_allocation"] = saved["ground_operator_error_allocation"]
    if direct_input is not None:
        saved["ground_export"]["direct_compression_is_ground_certificate"] = False
        saved["ground_export"]["direct_compression_mathematical_input"] = saved["direct_compression_input"]
    raw = {"mu2": mu2, "gamma": gamma, "E_g": eg, "ground_vector_error": eg_vector,
           "_ground_function": g, "_discretization": d}
    if controls:
        saved["measure_controls"] = measure_controls(d)
        saved["kernel_completion_controls"] = kernel_completion_controls(model, bounds)
    saved["scalar_integration_raw_controls"] = {
        "backend_certificate": "python-flint0.9.0 rigorous acb.integral with analytic branch/denominator guards",
        "finite_enclosed_call_count": d.integrals.calls,
        "unpropagated_radius_sum": encoded(upper(d.integrals.radius_sum)),
        "largest_returned_radius": encoded(d.integrals.radius_max),
        "tolerance_goal_only": encoded(d.integrals.tolerance),
        "radius_sum_is_not_a_propagated_residual_bound": True}
    if stage == "ground":
        saved["model_evaluated"] = True
        return raw
    # Everything below is strictly after the full ground preparation point.
    saved["source_jets_status"] = "attempted"
    bounds = completion_bounds(d, gamma, gamma_midpoint, degree=4)
    matrices = d.kernel_matrices(degree=4)
    model = FiniteModel(d, matrices, gamma_midpoint, degree=4)
    intervals, guard, pencil, count_metadata = spectral_data(model, F(1, 10**configuration.energy_digits), target_count=2)
    eg, ee, gap, positive_lower = expanded_levels(intervals, bounds, gamma, matrices[0])
    if not (upper(width(eg)) <= lower(a(F(1, 10**12))) and upper(width(ee)) <= lower(a(F(1, 10**12)))):
        raise Unresolved("Full ground/excited energy-width target unresolved")
    if not width(gap) < a(F(1, 10**6)) * lower(gap):
        raise Unresolved("Full positive-gap relative-width target unresolved")
    tau = positive(arb.pi() / (4 * gap), "External reference duration")
    saved.update({"E_g": encoded(eg), "E_e": encoded(ee), "Omega": encoded(gap), "tau": encoded(tau),
                  "full_two_level_counts": count_metadata, "source_jets_status": "enclosed",
                  "complete_operator_jet_error_allocation": {name: [encoded(value) for value in values]
                                                              for name, values in bounds["categories"].items()},
                  "minus_Toeplitz_matrix_exponentials": model.exp_metadata})
    g, g_metadata = eigenfunction(model, intervals[0], pencil, "ground")
    e, e_metadata = eigenfunction(model, intervals[1], pencil, "first_excited_f_projection")
    eg_vector, g_metadata = certify_ground(model, g, intervals[0], eg, ee, bounds, g_metadata)
    e, ee_vector, e_metadata = certify_excited_and_orthogonality(model, g, e, intervals, [eg, ee], guard, bounds, e_metadata)
    saved.update({"ground_radial_function": g_metadata, "excited_radial_function": e_metadata,
                  "excited_vector_status": "simple_f_projection_sign_and_orthonormalization_enclosed",
                  "Ramsey_functional_calculus_status": "attempted"})
    saved["ground_export"]["radial_function"] = g_metadata["continuous_representation"]
    saved["ground_export"]["full_embedding_phase_aligned_norm_error_upper"] = encoded(eg_vector)
    a1, a2, coefficient_metadata = spectral_coefficients(model, [g, e], intervals, gap, tau, bounds, [eg_vector, ee_vector])
    saved.update({"a1": encoded(a1), "a2": encoded(a2), "a2_minus_a1": encoded(a2 - a1),
                  "direct_spectral_coefficients": coefficient_metadata, "Ramsey_functional_calculus_status": "enclosed"})
    if not (upper(width(a1)) <= lower(a(F(1, 10**4))) and upper(width(a2)) <= lower(a(F(1, 10**4)))):
        raise Unresolved("Fully corrected Ramsey coefficient-width target unresolved; refine gap/time/order/precision")
    if controls:
        saved["independent_finite_source_phase_controls"] = phase_controls(model, [g, e], tau, gap, bounds,
                                                                           [eg_vector, ee_vector], a1, a2)
    raw.update({"E_g": eg, "E_e": ee, "Omega": gap, "tau": tau, "a1": a1, "a2": a2, "a2_minus_a1": a2 - a1,
                "ground_vector_error": eg_vector, "_ground_function": g,
                "_excited_function": e, "_excited_vector_error": ee_vector})
    saved["model_evaluated"] = True
    return raw


def next_configurations(base):
    ni, pi = ORDERS.index(base.order), PRECISIONS.index(base.digits)
    if ni + 1 == len(ORDERS) or pi + 1 == len(PRECISIONS):
        raise Unresolved("Next independent projection/precision controls exceed frozen ladders")
    projected = Configuration(base.cutoff, ORDERS[ni + 1], max(base.quadrature, ORDERS[ni + 1]),
                              base.digits, base.weight_order, base.energy_digits + 2)
    quadrature = Configuration(base.cutoff, base.order, 2 * base.quadrature, base.digits,
                               base.weight_order, base.energy_digits + 2)
    precision = Configuration(base.cutoff, base.order, base.quadrature, PRECISIONS[pi + 1],
                              base.weight_order, base.energy_digits + 2)
    return [("projection", projected), ("quadrature", quadrature), ("precision", precision)]


def refinement_controls(base, baseline, work, shared_gamma, stage, saved, direct_input=None, shared_mu2=None):
    output = []
    for name, configuration in next_configurations(base):
        row = {"axis": name, "status": "attempted"}
        output.append(row)
        saved["next_level_controls"] = output
        candidate = evaluate_configuration(configuration, work, shared_gamma, row, stage, controls=False,
                                           direct_input=direct_input, shared_mu2=shared_mu2)
        for key in baseline:
            if key == "ground_vector_error" or key.startswith("_"):
                continue
            if not baseline[key].overlaps(candidate[key]):
                raise Unresolved("Fully corrected next-" + name + " " + key + " enclosures fail overlap")
        difference = candidate["_discretization"].norm(baseline["_ground_function"] - candidate["_ground_function"])
        allowance = upper(baseline["ground_vector_error"] + candidate["ground_vector_error"])
        if not difference <= allowance:
            raise Unresolved("Phase-aligned ground functions fail next-" + name + " L2 error overlap")
        row["ground_function_L2_difference_upper"] = encoded(difference)
        row["ground_function_combined_phase_aligned_error_upper"] = encoded(allowance)
        if stage == "clock":
            difference = candidate["_discretization"].norm(baseline["_excited_function"] - candidate["_excited_function"])
            allowance = upper(baseline["_excited_vector_error"] + candidate["_excited_vector_error"])
            if not difference <= allowance:
                raise Unresolved("Phase-aligned excited functions fail next-" + name + " L2 error overlap")
            row["excited_function_L2_difference_upper"] = encoded(difference)
        row["status"] = "all_requested_corrected_enclosures_overlap"
    return output


def configurations():
    # Frozen ladders, deterministic engineering progression.  No response
    # selects a mass, coupling, state, source or readout.  Resource caps may
    # interrupt this progression and retain the last certified partial data.
    for cutoff_index, cutoff in enumerate(CUTOFFS):
        for order_index, order in enumerate(ORDERS):
            if len(panels_for(cutoff)) * order > MAX_RANK:
                continue
            for precision_index, digits in enumerate(PRECISIONS):
                weight_order = 8 * 2**min(precision_index, 4)
                energy_digits = min(digits - 32, 14 + 4 * order_index + 8 * precision_index + 4 * cutoff_index)
                yield Configuration(cutoff, order, 2 * order, digits, weight_order, energy_digits)


def new_result(args, backend, admission):
    return {"scope": "original sine-polar contact completion; externally prepared uniform K=0 pair",
            "source": SOURCE_PATH, "independent_source_review": REVIEW_PATH,
            "independent_ground_code_review": GROUND_CODE_REVIEW_PATH, "stage": args.stage,
            "fixed_parameters": {"m_over_mP": "1/1000", "a_over_lP": "1/1000", "alpha": "1/1000000",
                                 "Lambda_over_m": "1", "ell_times_m": "1", "gB": "0",
                                 "sector": "even standard-spinor singlet, half-period character+, K=0, positive-charge two excitations",
                                 "law": "H_t=exp(-gamma*T_t)*(2*epsilon_t)*exp(-gamma*T_t)",
                                 "gamma_rule": "log(4*Emax)/(2*mu2), same exact trial compression at every refinement",
                                 "sea": "W0=0; the unique fixed symmetric Newton-match rectangle is not refitted"},
            "backend": backend, "admission": admission,
            "predeclared_engineering_caps": {"evaluations": args.max_evaluations, "seconds": args.seconds,
                                             "resident_memory_MB": args.memory_mb, "integrator_evaluations_per_call": args.integral_evaluations,
                                             "one_worker": True, "maximum_threads": 4,
                                             "polynomial_rank": MAX_RANK, "last_precision_decimal_digits": 2048,
                                             "last_cutoff": 64},
            "status": "unresolved", "model_evaluated": False,
            "model_evaluation_attempted": False,
            "model_evaluated_semantics": "A numerical configuration point completed; certification requires separate gates and refinement controls",
            "ground_preparation_certified": False, "ramsey_certified": False, "full_clock_certified": False,
            "direct_compression_used_as_input": False, "direct_compression_is_ground_certificate": False,
            "direct_compression_input": None,
            "direct_compression_input_validation": None,
            "COM_response_or_triplet_gap_certified": False,
            "mu2": None, "gamma": None, "E_g": None, "E_e": None, "Omega": None, "tau": None,
            "a1": None, "a2": None, "a2_minus_a1": None,
            "unexecuted_outputs": ["E_e", "Omega", "tau", "a1", "a2", "a2_minus_a1"],
            "unexecuted_controls": ["excited_readout", "source_jets", "Ramsey_contours", "finite_source_phase", "constant_lapse"],
            "attempts": []}


def run(args, backend, admission):
    work, result, shared_gamma = Work(args), new_result(args, backend, admission), [None]
    direct_input, shared_mu2 = None, None
    advance_orders = set()
    try:
        if getattr(args, "direct_compression_input", None) is not None:
            result["direct_compression_input_validation"] = {}
            try:
                direct_input = ground_input.DirectInput(args.direct_compression_input, flint_module, admission, work,
                                                        result["direct_compression_input_validation"])
            except EngineeringCap:
                raise
            except (ground_input.InputUnresolved, ground_input.direct.Unresolved, json.JSONDecodeError) as error:
                raise DirectInputRejected("Direct input rejected: " + str(error)) from error
            shared_gamma[0], shared_mu2 = direct_input.gamma, [direct_input.mu2]
            result["direct_compression_used_as_input"] = True
        for configuration in configurations():
            if (configuration.cutoff, configuration.order) in advance_orders:
                continue
            row = {"status": "attempted"}
            result["attempts"].append(row)
            try:
                baseline = evaluate_configuration(configuration, work, shared_gamma, row, args.stage, controls=True,
                                                  direct_input=direct_input, shared_mu2=shared_mu2)
                refinement_controls(configuration, baseline, work, shared_gamma, args.stage, row,
                                    direct_input=direct_input, shared_mu2=shared_mu2)
            except (EngineeringCap, DirectInputRejected):
                raise
            except (Unresolved, AnalyticDomain) as error:
                if isinstance(error, RefineOrder):
                    advance_orders.add((configuration.cutoff, configuration.order))
                row.update({"status": "unresolved", "unresolved_item": str(error)})
                # Keep compact actual enclosures from the most recent
                # partial ground point, without promoting its controls.
                for key in ("mu2", "gamma", "gamma_midpoint_exact", "gamma_midpoint_mismatch_upper", "E_g",
                            "ground_counts", "ground_radial_function", "ground_export", "ground_operator_error_allocation",
                            "ground_full_operator_norm_error_upper", "ground_reference_error_certificate", "direct_compression_input"):
                    if key in row:
                        result[key] = row[key]
                result["model_evaluated"] = result["model_evaluated"] or row.get("model_evaluated", False)
                result["unresolved_item"] = str(error)
                continue
            row["status"] = "all_requested_stage_controls_passed"
            for key, value in row.items():
                if key not in ("status", "next_level_controls"):
                    result[key] = value
            result["ground_preparation_certified"] = True
            result["ground_export"]["certified_after_measure_kernel_and_next_level_controls"] = True
            result["next_level_controls"] = row["next_level_controls"]
            result["status"] = "certified_ground_preparation" if args.stage == "ground" else "certified_full_clock_control"
            if args.stage == "clock":
                result["ramsey_certified"] = result["full_clock_certified"] = True
                result["unexecuted_outputs"] = result["unexecuted_controls"] = []
            result.pop("unresolved_item", None)
            break
        else:
            result["unresolved_item"] = "Frozen cutoff/order/precision ladder exhausted without every requested stage control"
    except EngineeringCap as error:
        result["unresolved_item"] = str(error)
        if result["attempts"]:
            row = result["attempts"][-1]
            row.update({"status": "unresolved", "unresolved_item": str(error)})
            for key in ("mu2", "gamma", "gamma_midpoint_exact", "gamma_midpoint_mismatch_upper", "E_g",
                        "ground_counts", "ground_radial_function", "ground_export", "ground_operator_error_allocation",
                        "ground_full_operator_norm_error_upper", "ground_reference_error_certificate", "direct_compression_input"):
                if key in row:
                    result[key] = row[key]
            result["model_evaluated"] = result["model_evaluated"] or row.get("model_evaluated", False)
    except DirectInputRejected as error:
        result.update({"status": "unresolved", "unresolved_item": str(error),
                       "direct_input_halted_for_review": True})
        if result["attempts"]:
            result["attempts"][-1].update({"status": "unresolved", "unresolved_item": str(error)})
    except Exception as error:
        result["status"] = "backend_or_arithmetic_error"
        result["unresolved_item"] = type(error).__name__ + ": " + str(error)
        if result["attempts"]:
            result["attempts"][-1].update({"status": "backend_or_arithmetic_error",
                                           "unresolved_item": result["unresolved_item"]})
        result["model_evaluated"] = result["model_evaluated"] or any(
            row.get("model_evaluated", False) for row in result["attempts"])
        result["model_evaluation_attempted"] = any(
            row.get("model_evaluation_attempted", False) for row in result["attempts"])
        result["elapsed_seconds"] = time.perf_counter() - work.start
        result["evaluation_count"] = work.evaluations
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        raise
    result["model_evaluated"] = result["model_evaluated"] or any(
        row.get("model_evaluated", False) for row in result["attempts"])
    result["model_evaluation_attempted"] = any(
        row.get("model_evaluation_attempted", False) for row in result["attempts"])
    result["elapsed_seconds"] = time.perf_counter() - work.start
    result["evaluation_count"] = work.evaluations
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result["status"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("ground", "clock"), default="ground")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-evaluations", type=int, default=2_000_000)
    parser.add_argument("--integral-evaluations", type=int, default=20_000)
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--memory-mb", type=int, default=4096)
    parser.add_argument("--direct-compression-input", type=Path,
                        help="Ground-only completed, author/independently reviewed same-rule direct compression JSON")
    parser.add_argument("--direct-compression-input-audits-released", action="store_true")
    args = parser.parse_args()
    if args.stage == "clock":
        parser.error("The clock branch needs its separate actual-code audits; only the ground-stage boundary is submitted for release")
    if args.direct_compression_input is not None and (
            not args.direct_compression_input_audits_released or not args.direct_compression_input.is_file()):
        parser.error("Direct input requires its actual artifact audits and an explicit existing JSON; full validation still runs")
    if args.max_evaluations <= 0 or args.integral_evaluations <= 0 or args.seconds <= 0 or args.memory_mb <= 0:
        parser.error("Engineering caps must be fixed positive integers before evaluation")
    if not args.output.parent.is_dir() or args.output.exists():
        parser.error("Output must be a new file in an existing admitted result directory")
    pools = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
    if any(os.environ.get(name) != "1" for name in pools):
        parser.error("All six worker/BLAS pools must be predeclared as one")
    backend = load_backend()  # Complete host-only loading before admission.
    if hasattr(ctx, "threads"):
        ctx.threads = 1
    admission = remote_execution(max_threads=4)  # Final free-device/own-PID launch guard.
    status = run(args, backend, admission)
    print(status, flush=True)


if __name__ == "__main__":
    main()
