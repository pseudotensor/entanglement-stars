#!/usr/bin/env python3
"""Current scalar UV-law spacing certificates, point delta=1/1000 only.

The source/error contract is solve/strand_E_uv_spacing.tex and its independent
PASS review.  Runtime, tests, and numerical postprocessing are remote-only.
This module reuses frozen arithmetic helpers, never earlier-law physics or
results.  The continuum K0 is the single Newton match.  A point coefficient
test does not establish a source lock, an operational chart, a stellar or
physical PPN window, or an improved-stencil entropy result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import heapq
import itertools
import json
import os
from pathlib import Path
import time

from mpmath import iv
import numpy as np

import uv_native_certificate as arithmetic
from remote_runtime import remote_execution


F = Fraction
X0 = F(1, 10**6)
MASS_STEP = (1 - X0) / 2
MASS_SQUARES = (X0, X0 + MASS_STEP, X0 + MASS_STEP, F(1))
GRADING = (1, -1, -1, 1)
BI = arithmetic.BinaryInterval
I = arithmetic.interval
SQRT = arithmetic.certified_iv_sqrt
ENDS = arithmetic.endpoints
HULL = arithmetic.hull


def assert_rectangle():
    if not (X0 > 0 and MASS_STEP > 0
            and MASS_SQUARES == (X0, X0 + MASS_STEP, X0 + MASS_STEP,
                                X0 + 2 * MASS_STEP)
            and MASS_SQUARES[-1] == 1):
        raise ValueError("This stable grouping requires the frozen equal-step rectangle")


def interval_abs_upper(value):
    lower, upper = ENDS(value)
    return max(abs(lower), abs(upper))


def midpoint(value):
    """An exact arbitrary-precision dyadic, never a binary64 solution seed."""
    lower, upper = ENDS(value)
    return (lower + upper) / 2


def enlarged(value, error):
    lower, upper = ENDS(value)
    return HULL(lower - error, upper + error)


def finite(value):
    if isinstance(value, BI):
        if (np.any(~np.isfinite(value.lower)) or np.any(~np.isfinite(value.upper))
                or np.any(value.lower > value.upper)):
            raise ArithmeticError("Non-finite or unordered directed tensor interval")
    else:
        ENDS(value)
    return value


def root(value):
    return value.sqrt() if isinstance(value, BI) else SQRT(value)


def stable_groups(y, yi, nu):
    """Exact RH and RG using the positive symmetric mass rectangle.

    S = R[V^-1/2] and R[A^-1 V^-1/2] are rationalized before evaluation.
    All log factors below differentiate positive denominator factors; none
    divides by c or y.  c=0 at a cone is therefore covered directly.
    """
    assert_rectangle()
    scalar = BI.rational if isinstance(y, BI) else I
    b = y ** (2 * nu)
    a = [b + scalar(x) for x in (X0, X0 + MASS_STEP, F(1))]
    v = [item + y for item in a]
    q = [root(item) for item in v]
    q01, q02, q12 = q[0] + q[1], q[0] + q[2], q[1] + q[2]
    s = (2 * scalar(MASS_STEP**2) * (q[0] + q[1] + q[2])
         / (q[0] * q[1] * q[2] * q01 * q02 * q12))
    da = scalar(MASS_STEP) / (a[0] * a[1])
    db = scalar(MASS_STEP) / (q[1] * q[2] * q12)
    ra = 2 * scalar(MASS_STEP**2) / (a[0] * a[1] * a[2])
    terms = (s / a[0], 2 * da * db, ra / q[2])
    e = y + 2 * nu * b
    la = [-4 * nu * b / item for item in a]
    lq = [-e / item for item in v]

    def log_sum(indices):
        return (sum((q[j] * lq[j] for j in indices), scalar(0))
                / sum((q[j] for j in indices), scalar(0)))

    ls = (log_sum((0, 1, 2)) - sum(lq, scalar(0))
          - log_sum((0, 1)) - log_sum((0, 2)) - log_sum((1, 2)))
    lt = (ls - la[0], -la[0] - la[1] - lq[1] - lq[2]
          - log_sum((1, 2)), -sum(la, scalar(0)) - lq[2])
    c = 4 * nu**2 * yi * y ** (4 * nu - 2)
    h = s + c * sum(terms, scalar(0))
    g = (s * (ls - 3)
         + c * sum((term * (factor - (8 * nu - 2) - 3)
                    for term, factor in zip(terms, lt)), scalar(0)))
    return finite(h), finite(g), finite(s), finite(sum(terms, scalar(0)))


def direct_groups(y, yi, nu):
    """Independent arbitrary-precision four-mass point diagnostic only."""
    b = y ** (2 * nu)
    c = 4 * nu**2 * yi * y ** (4 * nu - 2)
    e = y + 2 * nu * b
    hs, gs = I(0), I(0)
    for mass, sign in zip(MASS_SQUARES, GRADING):
        a, v = I(mass) + b, I(mass) + b + y
        inverse_root = 1 / SQRT(v)
        h = (1 + c / a) * inverse_root
        g = (-3 - (8 * nu + 1) * c / a + 4 * nu * b * c / a**2
             + e / v + c * e / (a * v)) * inverse_root
        hs += sign * h
        gs += sign * g
    return hs, gs


def stencil(z, name):
    sine, cosine = iv.sin(z), iv.cos(z)
    if name == "nn":
        return sine, cosine
    if name == "imp":
        return sine + sine**3 / 6, cosine * (1 + sine**2 / 2)
    raise ValueError("Unknown frozen stencil")


def second_mass_bound(r, s, a, v):
    """Reviewed common-variable B_(r,s), with exact rational exponents."""
    def vpow(power):
        return 1 / (v ** int(power) if power.denominator == 1
                    else v ** int(power - F(1, 2)) * SQRT(v))
    result = I(0)
    if r:
        result += I(r * (r + 1)) / a ** (r + 2) * vpow(s)
    if r and s:
        result += I(2 * r * s) / a ** (r + 1) * vpow(s + 1)
    if s:
        result += I(s * (s + 1)) / a**r * vpow(s + 2)
    return result


def grouped_bounds(ymax, yimax, amin, vmin, weight, nu):
    b, e = ymax ** (2 * nu), ymax + 2 * nu * ymax ** (2 * nu)
    c = 4 * nu**2 * yimax * ymax ** (4 * nu - 2)
    b01 = second_mass_bound(0, F(1, 2), amin, vmin)
    b11 = second_mass_bound(1, F(1, 2), amin, vmin)
    h = weight * I(MASS_STEP**2) * (b01 + c * b11)
    g = weight * I(MASS_STEP**2) * (
        3 * b01 + (8 * nu + 1) * c * b11
        + 4 * nu * b * c * second_mass_bound(2, F(1, 2), amin, vmin)
        + e * second_mass_bound(0, F(3, 2), amin, vmin)
        + c * e * second_mass_bound(1, F(3, 2), amin, vmin))
    return ENDS(h)[1], ENDS(g)[1]


@dataclass(frozen=True)
class Box:
    bounds: tuple[tuple[Fraction, Fraction], ...]

    @property
    def volume(self):
        return (self.bounds[0][1] - self.bounds[0][0]) * (
            self.bounds[1][1] - self.bounds[1][0]) * (
            self.bounds[2][1] - self.bounds[2][0])

    def bisect(self, direction):
        lower, upper = self.bounds[direction]
        center = (lower + upper) / 2
        result = []
        for interval in ((lower, center), (center, upper)):
            bounds = list(self.bounds)
            bounds[direction] = interval
            result.append(Box(tuple(bounds)))
        if sum((item.volume for item in result), F(0)) != self.volume:
            raise ArithmeticError("Dyadic box bisection lost domain measure")
        return result


@dataclass
class Bounds:
    real_h: Fraction
    real_g: Fraction
    complex_h: Fraction | None
    complex_g: Fraction | None
    direction: int
    ymin: Fraction
    ymax: Fraction
    yimin: Fraction
    yimax: Fraction


def box_bounds(box, tau, name, nu):
    tlo, thi = ENDS(tau)
    if tlo <= 0:
        raise ArithmeticError("Spacing input must have positive lower endpoint")
    min_parts, max_parts, centers, deltas, js = [], [], [], [], []
    for lower, upper in box.bounds:
        left, right = PI * I(lower) / 2, PI * I(upper) / 2
        fl, _ = stencil(left, name)
        fr, _ = stencil(right, name)
        min_parts.append(max(F(0), ENDS(fl**2 / I(thi)**2)[0]))
        max_parts.append(ENDS(fr**2 / I(tlo)**2)[1])
        center = PI * I((lower + upper) / 2) / 2
        half_width = PI * I(upper - lower) / 4
        radius = 5 * half_width / 4
        imaginary = 3 * half_width / 4
        cosh = (iv.exp(imaginary) + iv.exp(-imaginary)) / 2
        if name == "nn":
            fmax = jmax = lmax = cosh
        else:
            fmax, jmax = cosh + cosh**3 / 6, cosh + cosh**3 / 2
            lmax = 3 * cosh**3 / 2
        fc, dc = stencil(center, name)
        fcdc = interval_abs_upper(fc * dc)
        error = 2 * I(fcdc) * radius + (jmax**2 + fmax * lmax) * radius**2
        centers.append(fc**2 / tau**2)
        deltas.append(ENDS(error / I(tlo)**2)[1])
        js.append(ENDS(jmax**2)[1])
    ymin, ymax = sum(min_parts, F(0)), sum(max_parts, F(0))
    amin, vmin = I(X0) + I(ymin)**(2 * nu), I(X0) + I(ymin) + I(ymin)**(2 * nu)
    real_h, real_g = grouped_bounds(I(ymax), I(max_parts[0]), amin, vmin, I(1), nu)
    total_center = sum(centers, I(0))
    clo, chi = ENDS(total_center)
    d = sum(deltas, F(0))
    # Bounding positive binomial coefficients with the upper center preserves
    # their common variable; no subtraction of unrelated mass intervals occurs.
    delta_b = ENDS((I(chi) + I(d))**(2 * nu) - I(chi)**(2 * nu))[1]
    delta_v = d + delta_b
    gap_a = ENDS(I(X0) + I(clo)**(2 * nu))[0]
    gap_v = ENDS(I(X0) + I(clo) + I(clo)**(2 * nu))[0]
    complex_h = complex_g = None
    if delta_b <= gap_a / 2 and delta_v <= gap_v / 2:
        astar, vstar = I(gap_a - delta_b), I(gap_v - delta_v)
        ystar = I(chi + d)
        yistar = I(ENDS(centers[0])[1] + deltas[0])
        complex_h, complex_g = grouped_bounds(ystar, yistar, astar, vstar, I(js[0]), nu)
    direction = max(range(3), key=lambda j: deltas[j])
    return Bounds(real_h, real_g, complex_h, complex_g, direction,
                  ymin, ymax, min_parts[0], max_parts[0])


@dataclass
class Leaf:
    box: Box
    bounds: Bounds
    order: int
    k_lower: Fraction
    k_upper: Fraction
    d_lower: Fraction
    d_upper: Fraction
    arithmetic_k: Fraction = F(0)
    arithmetic_d: Fraction = F(0)
    gauss_k: Fraction = F(0)
    gauss_d: Fraction = F(0)
    box_k: Fraction = F(0)
    box_d: Fraction = F(0)

    def components(self):
        return (self.k_lower, self.k_upper, self.d_lower, self.d_upper,
                self.arithmetic_k, self.arithmetic_d, self.gauss_k,
                self.gauss_d, self.box_k, self.box_d)


class LimitReached(RuntimeError):
    pass


class Work:
    def __init__(self, args):
        self.args = args
        self.start = time.perf_counter()
        self.evaluations = 0
        self.gauss = {}

    def check(self, increment=0):
        self.evaluations += increment
        if (self.evaluations > self.args.max_evaluations
                or time.perf_counter() - self.start > self.args.seconds):
            raise LimitReached("Declared node/time resource cap reached")
        if len(os.listdir("/proc/self/task")) > self.args.max_threads:
            raise RuntimeError("Remote spacing job exceeds its coordinated thread allowance")

    def rule(self, order):
        if order not in self.gauss:
            self.check()
            nodes, weights, detail = arithmetic.gauss_enclosures(
                order, max(self.args.root_bits, 4 * order))
            self.gauss[order] = nodes, weights, detail
        return self.gauss[order]


def direct_leaf(box, bounds, tau):
    pref = 3 / tau**3
    kh = ENDS(pref * I(box.volume) * I(bounds.real_h))[1]
    dh = ENDS(pref * I(box.volume) * I(bounds.real_g))[1]
    return Leaf(box, bounds, 0, F(0), kh, -dh, dh, box_k=kh / 2, box_d=dh)


def quadrature_leaf(box, bounds, tau, name, nu, order, work, backend):
    if bounds.complex_h is None:
        raise ArithmeticError("Gauss integration requires both reviewed complex gap tests")
    nodes, weights, _ = work.rule(order)
    points, measures = [], []
    for lower, upper in box.bounds:
        mapped, mapped_weights = arithmetic.mapped_gauss(lower, upper, nodes, weights)
        fv, dv = zip(*(stencil(PI * item / 2, name) for item in mapped))
        if backend == "binary":
            points.append(BI.vector([item**2 / tau**2 for item in fv]))
            measures.append(BI.vector(mapped_weights))
            if len(points) == 1:
                directional_weight = BI.vector([item**2 for item in dv])
        else:
            points.append([item**2 / tau**2 for item in fv])
            measures.append(mapped_weights)
            if len(points) == 1:
                directional_weight = [item**2 for item in dv]
    work.check(order**3)
    if backend == "binary":
        yi = points[0][:, None, None]
        y = yi + points[1][None, :, None] + points[2][None, None, :]
        h, g, _, _ = stable_groups(y, yi, nu)
        weight = (measures[0][:, None, None] * measures[1][None, :, None]
                  * measures[2][None, None, :] * directional_weight[:, None, None])
        qk = finite(h * weight).sum().to_iv() * 3 / tau**3
        qd = finite(g * weight).sum().to_iv() * 3 / tau**3
    else:
        qk, qd = I(0), I(0)
        for j in range(order):
            work.check()
            for k in range(order):
                for ell in range(order):
                    y = points[0][j] + points[1][k] + points[2][ell]
                    h, g, _, _ = stable_groups(y, points[0][j], nu)
                    weight = (measures[0][j] * measures[1][k] * measures[2][ell]
                              * directional_weight[j])
                    qk += weight * h
                    qd += weight * g
        qk *= 3 / tau**3
        qd *= 3 / tau**3
    constant = 3 / tau**3 * 8 * I(box.volume) * 3 / (1 << (2 * order))
    ek = ENDS(constant * I(bounds.complex_h))[1]
    ed = ENDS(constant * I(bounds.complex_g))[1]
    klo, khi = ENDS(qk)
    dlo, dhi = ENDS(qd)
    return Leaf(box, bounds, order, klo - ek, khi + ek, dlo - ed, dhi + ed,
                (khi - klo) / 2, (dhi - dlo) / 2, ek, ed)


def result_point(tau, name, nu, totals, leaves, work, reason, requested_k, requested_d):
    klo, khi, dlo, dhi, ak, ad, gk, gd, bk, bd = totals
    positive = klo > 0
    kval, dval = HULL(klo, khi), HULL(dlo, dhi)
    kapp = dval / kval if positive else None
    kradius = (khi - klo) / 2
    kappa_radius = (ENDS(kapp)[1] - ENDS(kapp)[0]) / 2 if positive else None
    meets_k = positive and kradius <= klo / 10**9
    meets_kappa = positive and kappa_radius <= F(1, 10**7)
    return {
        "stencil": name, "nu": nu, "tau": arithmetic.enclosure(tau, 32),
        "scope": "Pointwise delta=1/1000 thermodynamic small-q vacuum coefficient only",
        "K": arithmetic.enclosure(kval, 32), "D": arithmetic.enclosure(dval, 32),
        "kappa": arithmetic.enclosure(kapp, 32) if positive else None,
        "K_positive": positive, "relative_radius_K_at_most_1e-9": meets_k,
        "absolute_radius_kappa_at_most_1e-7": meets_kappa,
        "certificate_status": "certified_point" if meets_k and meets_kappa else "unresolved",
        "termination": reason, "boxes": len(leaves),
        "Gauss_boxes": sum(item.order > 0 for item in leaves.values()),
        "directly_enclosed_real_boxes": sum(item.order == 0 for item in leaves.values()),
        "radii": {
            "arithmetic_K": arithmetic.decimal_endpoint(ak, 40, True),
            "arithmetic_D": arithmetic.decimal_endpoint(ad, 40, True),
            "analytic_Gauss_K": arithmetic.decimal_endpoint(gk, 40, True),
            "analytic_Gauss_D": arithmetic.decimal_endpoint(gd, 40, True),
            "finite_domain_real_box_K": arithmetic.decimal_endpoint(bk, 40, True),
            "finite_domain_real_box_D": arithmetic.decimal_endpoint(bd, 40, True),
            "omitted_infinite_momentum_tail": "0",
            "combined_K": arithmetic.decimal_endpoint(kradius, 40, True),
            "combined_kappa": arithmetic.decimal_endpoint(kappa_radius, 40, True)
                               if positive else None},
        "requested_absolute_work_targets": {
            "K": arithmetic.decimal_endpoint(requested_k, 40, True),
            "D": arithmetic.decimal_endpoint(requested_d, 40, True)},
        "node_evaluations_cumulative": work.evaluations,
        "conditional_gamma1_coefficient_channel": (
            "inside" if positive and interval_abs_upper(kapp) < F(4, 10**5)
            else "outside" if positive and (ENDS(kapp)[0] > F(4, 10**5)
                                             or ENDS(kapp)[1] < -F(4, 10**5))
            else "unresolved"),
        "spacing_interval_certified": False,
    }, kval, dval


def coefficient_point(tau, name, nu, k_reference, work, backend="binary",
                      target_k=None, target_d=None):
    """Adaptive global enclosure; exact endpoint totals retain correlations.

    Real boxes always cover the entire cube.  Replacing a box by children or
    a Gauss enclosure subtracts exact stored rational endpoints, not an
    interval evaluated as an unrelated second copy.  A heap only schedules
    work; all acceptance, arithmetic, and error comparisons are exact.
    """
    target_k = target_k or k_reference / 10**10
    target_d = target_d or k_reference / 10**8
    initial = Box(((F(0), F(1)),) * 3)
    first = direct_leaf(initial, box_bounds(initial, tau, name, nu), tau)
    leaves, heap, totals = {}, [], [F(0)] * 10
    serial = 0

    def add(leaf):
        nonlocal serial
        serial += 1
        leaves[serial] = leaf
        for j, amount in enumerate(leaf.components()):
            totals[j] += amount
        priority = max((leaf.k_upper - leaf.k_lower) / (2 * target_k),
                       (leaf.d_upper - leaf.d_lower) / (2 * target_d))
        heapq.heappush(heap, (-priority, serial))

    def remove(identifier):
        leaf = leaves.pop(identifier)
        for j, amount in enumerate(leaf.components()):
            totals[j] -= amount
        return leaf

    add(first)
    reason = "Requested combined budgets reached"
    while ((totals[1] - totals[0]) / 2 > target_k
           or (totals[3] - totals[2]) / 2 > target_d or totals[0] <= 0):
        try:
            work.check()
            if len(leaves) >= work.args.max_boxes:
                raise LimitReached("Declared dyadic box resource cap reached")
            _, identifier = heapq.heappop(heap)
            leaf = leaves[identifier]
            if leaf.order and leaf.order >= work.args.max_order:
                # Splitting can tighten analytic bounds; it cannot remove a
                # directed arithmetic floor.  Permit a user-selected iv run.
                if ((leaf.arithmetic_k >= leaf.gauss_k and leaf.arithmetic_k > target_k / 16)
                        or (leaf.arithmetic_d >= leaf.gauss_d
                            and leaf.arithmetic_d > target_d / 16)):
                    raise LimitReached("Arithmetic enclosure needs the iv backend or higher precision")
            if leaf.bounds.complex_h is None or leaf.order >= work.args.max_order:
                children = []
                for box in leaf.box.bisect(leaf.bounds.direction):
                    bounds = box_bounds(box, tau, name, nu)
                    children.append(direct_leaf(box, bounds, tau))
                remove(identifier)
                for child in children:
                    add(child)
            else:
                order = 8 if leaf.order == 0 else 2 * leaf.order
                # Orders are the reviewed ladder.  This heuristic changes
                # cost only; the actual bound below controls acceptance.
                while order < work.args.max_order:
                    scale = 3 / tau**3 * 8 * I(leaf.box.volume) * 3 / (1 << (2 * order))
                    if (ENDS(scale * I(leaf.bounds.complex_h))[1] <= target_k / 16
                            and ENDS(scale * I(leaf.bounds.complex_g))[1] <= target_d / 16):
                        break
                    order *= 2
                replacement = quadrature_leaf(leaf.box, leaf.bounds, tau, name, nu,
                                               order, work, backend)
                remove(identifier)
                add(replacement)
        except LimitReached as error:
            reason = str(error)
            break
    result, kval, dval = result_point(tau, name, nu, totals, leaves, work, reason,
                                     target_k, target_d)
    return result, kval, dval, list(leaves.values())


def radial_panel_bound(lower, upper, nu):
    center, half = (lower + upper) / 2, (upper - lower) / 2
    radius = 5 * half / 4
    maximum = I(center + radius)
    db = maximum**(4 * nu) - I(center)**(4 * nu)
    dv = maximum**2 - I(center)**2 + db
    gap_a = I(X0) + I(center)**(4 * nu)
    gap_v = gap_a + I(center)**2
    if ENDS(db)[1] > ENDS(gap_a)[0] / 2 or ENDS(dv)[1] > ENDS(gap_v)[0] / 2:
        return None
    astar = I(ENDS(gap_a)[0] - ENDS(db)[1])
    vstar = I(ENDS(gap_v)[0] - ENDS(dv)[1])
    bound = maximum**2 * I(MASS_STEP**2) * (
        second_mass_bound(0, F(1, 2), astar, vstar)
        + I(F(4 * nu**2, 3)) * maximum**(8 * nu - 2)
        * second_mass_bound(1, F(1, 2), astar, vstar))
    return ENDS(bound * 12 / PI**2)[1]


def continuum_certificate(nu, work):
    """Point-positive radial reference through L=16 and reviewed positive tail."""
    todo, panels = [(F(0), F(16))], []
    while todo:
        work.check()
        lower, upper = todo.pop()
        bound = radial_panel_bound(lower, upper, nu)
        if bound is None:
            middle = (lower + upper) / 2
            todo.extend(((middle, upper), (lower, middle)))
        else:
            panels.append((lower, upper, bound))
        if len(todo) + len(panels) > work.args.max_boxes:
            raise LimitReached("Continuum panel resource cap reached")
    panels.sort()
    if (panels[0][0] != 0 or panels[-1][1] != 16
            or any(a[1] != b[0] for a, b in zip(panels, panels[1:]))):
        raise ArithmeticError("Continuum exact rational panels do not tile [0,16]")
    tail = (12 * I(MASS_STEP**2) / PI**2
            * (I(F(3, 4 * (10 * nu - 3))) * I(16)**(3 - 10 * nu)
               + I(F(5 * nu**2, 6 * nu - 1)) * I(16)**(1 - 6 * nu)))
    tail_upper = ENDS(tail)[1]
    retained = None
    for order in (8, 16, 32, 64, 128):
        if order > work.args.max_order:
            break
        nodes, weights, _ = work.rule(order)
        quadrature, error = I(0), F(0)
        for lower, upper, bound in panels:
            work.check(order)
            points, measures = arithmetic.mapped_gauss(lower, upper, nodes, weights)
            for p, measure in zip(points, measures):
                y = p**2
                h, _, _, _ = stable_groups(y, y / 3, nu)
                quadrature += measure * 12 / PI**2 * y * h
            error += ENDS(8 * I(upper - lower) * I(bound) / (1 << (2 * order)))[1]
        qlo, qhi = ENDS(quadrature)
        coefficient = HULL(qlo - error, qhi + error + tail_upper)
        lower, upper = ENDS(coefficient)
        radius = (upper - lower) / 2
        passed = lower > 0 and radius <= lower / 10**9
        retained = (coefficient, {
            "nu": nu, "mass_scope": "delta=1/1000 only", "L": 16,
            "K0": arithmetic.enclosure(coefficient, 32),
            "certificate_status": "certified" if passed else "unresolved",
            "panels": len(panels), "Gauss_order": order,
            "arithmetic_radius": arithmetic.decimal_endpoint((qhi - qlo) / 2, 40, True),
            "analytic_Gauss_radius": arithmetic.decimal_endpoint(error, 40, True),
            "positive_radial_tail_upper": arithmetic.decimal_endpoint(tail_upper, 40, True),
            "combined_radius": arithmetic.decimal_endpoint(radius, 40, True),
            "sole_match": "Omega^2 K0=1/(2 pi G); no finite-a refit"})
        if passed:
            break
    if retained is None:
        raise LimitReached("No continuum Gauss order was permitted")
    coefficient, result = retained
    if ENDS(coefficient)[0] > 0:
        result["Omega_lP"] = arithmetic.enclosure(1 / SQRT(2 * PI * coefficient), 32)
    return coefficient, result


def derivative_terms(nu, count):
    """Reviewed L recurrence, exact sparse polynomial coefficients.

    Keys are (r,s,power_y,power_yi), coefficient multiplies A^-r V^-s.
    """
    terms = {(0, F(1, 2), 0, 0): F(1),
             (1, F(1, 2), 4 * nu - 2, 1): F(4 * nu**2)}
    for _ in range(count):
        next_terms = {}

        def add(key, coefficient):
            next_terms[key] = next_terms.get(key, F(0)) + coefficient

        for (r, s, p, pi), coefficient in terms.items():
            add((r, s, p, pi), (-3 - 2 * (p + pi)) * coefficient)
            if r:
                add((r + 1, s, p + 2 * nu, pi), 4 * nu * r * coefficient)
            add((r, s + 1, p + 1, pi), 2 * s * coefficient)
            add((r, s + 1, p + 2 * nu, pi), 4 * s * nu * coefficient)
        terms = {key: value for key, value in next_terms.items() if value}
    return terms


def log_derivative_bounds(leaves, tau_cell, name, nu, work):
    results = []
    prefactor = 3 / I(ENDS(tau_cell)[0])**3
    for count in (1, 2, 3):
        total = I(0)
        terms = derivative_terms(nu, count)
        for leaf in leaves:
            work.check()
            bound = box_bounds(leaf.box, tau_cell, name, nu)
            amin = I(X0) + I(bound.ymin)**(2 * nu)
            vmin = amin + I(bound.ymin)
            point = I(0)
            for (r, s, p, pi), coefficient in terms.items():
                point += (I(abs(coefficient)) * I(bound.ymax)**p * I(bound.yimax)**pi
                          * second_mass_bound(r, s, amin, vmin))
            total += (prefactor * I(leaf.box.volume) * I(MASS_STEP**2) * point)
        results.append(ENDS(total)[1])
    return results


def derivative_check(tau, name, nu, kval, dval, leaves, work):
    """Independent central difference, separately certified or unresolved.

    This check never upgrades a point into a spacing interval.  The final
    h uses M3 on an enclosing log cell, not an unbounded numerical estimate.
    """
    lower_k = ENDS(kval)[0]
    if lower_k <= 0:
        return {"status": "unresolved", "reason": "No positive central K enclosure"}
    h = F(1, 1024)
    broad = HULL(ENDS(tau * iv.exp(-I(h)))[0], ENDS(tau * iv.exp(I(h)))[1])
    try:
        m1, m2, m3 = log_derivative_bounds(leaves, broad, name, nu, work)
        while h**2 * m3 / 6 > lower_k / 10**8:
            h /= 2
            if h < F(1, 1 << 256):
                raise LimitReached("Derivative truncation bound needs a smaller permitted log step")
        endpoint_budget = h * lower_k / 10**8
        if work.args.backend == "binary" and endpoint_budget < lower_k / 10**12:
            return {"status": "unresolved", "reason": "Endpoint budget requires iv tensor precision",
                    "h": str(h), "M1": arithmetic.decimal_endpoint(m1, 32, True),
                    "M2": arithmetic.decimal_endpoint(m2, 32, True),
                    "M3": arithmetic.decimal_endpoint(m3, 32, True),
                    "endpoint_absolute_budget": arithmetic.decimal_endpoint(endpoint_budget, 40, True)}
        endpoint_values = []
        for sign in (-1, 1):
            point = tau * iv.exp(sign * I(h))
            _, value, _, _ = coefficient_point(point, name, nu, lower_k, work,
                                               work.args.backend, endpoint_budget, lower_k / 10**8)
            endpoint_values.append(value)
        minus, plus = endpoint_values
        endpoint_error = sum(((ENDS(item)[1] - ENDS(item)[0]) / 2
                              for item in endpoint_values), F(0)) / (2 * h)
        truncation = h**2 * m3 / 6
        central = enlarged((plus - minus) / (2 * I(h)), truncation)
        difference = central - dval
        complete = (endpoint_error <= lower_k / 10**8
                    and truncation <= lower_k / 10**8)
        contains_zero = ENDS(difference)[0] <= 0 <= ENDS(difference)[1]
        return {"status": "certified_check" if complete and contains_zero else "unresolved",
                "h": str(h), "M1": arithmetic.decimal_endpoint(m1, 32, True),
                "M2": arithmetic.decimal_endpoint(m2, 32, True),
                "M3": arithmetic.decimal_endpoint(m3, 32, True),
                "endpoint_divided_radius": arithmetic.decimal_endpoint(endpoint_error, 40, True),
                "truncation_error_upper": arithmetic.decimal_endpoint(truncation, 40, True),
                "central_difference_enclosure": arithmetic.enclosure(central, 32),
                "difference_from_D": arithmetic.enclosure(difference, 32)}
    except LimitReached as error:
        return {"status": "unresolved", "reason": str(error)}


# The finite matrix test uses complete 2x2 Fourier blocks and both source
# modes.  Complex entries are (real interval, imaginary interval) pairs.
def cz(real=0, imaginary=0):
    return I(real), I(imaginary)


def ca(left, right):
    return left[0] + right[0], left[1] + right[1]


def cn(value):
    return -value[0], -value[1]


def cm(left, right):
    return left[0] * right[0] - left[1] * right[1], left[0] * right[1] + left[1] * right[0]


def cs(value, scalar):
    return value[0] * scalar, value[1] * scalar


def cc(value):
    return value[0], -value[1]


def ma(left, right):
    return [[ca(left[i][j], right[i][j]) for j in range(2)] for i in range(2)]


def mn(value):
    return [[cn(value[i][j]) for j in range(2)] for i in range(2)]


def ms(value, scalar):
    return [[cs(value[i][j], scalar) for j in range(2)] for i in range(2)]


def mm(left, right):
    return [[ca(cm(left[i][0], right[0][j]), cm(left[i][1], right[1][j]))
             for j in range(2)] for i in range(2)]


def md(value):
    return [[cc(value[j][i]) for j in range(2)] for i in range(2)]


def diagonal(value):
    return [[(value, I(0)), cz()], [cz(), (value, I(0))]]


def norm2(value):
    return sum((entry[0]**2 + entry[1]**2 for row in value for entry in row), I(0))


def point_matrix(value):
    return [[(I(midpoint(entry[0])), I(midpoint(entry[1]))) for entry in row] for row in value]


def spin_dirac(components):
    dx, dy, dz = components
    return [[(dz, I(0)), (dx, -dy)], [(dx, dy), (-dz, I(0))]]


def i_times(value):
    return [[(-entry[1], entry[0]) for entry in row] for row in value]


def matrix_attempt(name, nu, ruler, mode, work):
    sizes, m = (8, 4, 4), 128
    modes = list(itertools.product(*(range(size) for size in sizes)))
    symbols, deltas, dirs = {}, {}, {}
    for p in modes:
        components = [stencil(2 * PI * I(F(index, size)), name)[0]
                      for index, size in zip(p, sizes)]
        symbols[p] = components
        deltas[p] = sum((item**2 for item in components), I(0))
        dirs[p] = spin_dirac(components)
    phi = SQRT2 / 2
    pval = I(ruler)
    trace = I(0)
    momentum = I(0)
    packet_errors, packet_residuals = [], []
    for mass, sign in zip(MASS_SQUARES, GRADING):
        work.check()
        r, e, t, avec = {}, {}, {}, {}
        a_norm2 = I(0)
        for p in modes:
            t[p] = deltas[p] / pval**4
            r[p] = SQRT(I(mass) + t[p]**(2 * nu))
            e[p] = SQRT(I(mass) + t[p]**(2 * nu) + t[p]) / pval
            avec[p] = ma(diagonal(r[p] / pval),
                         ms(i_times(dirs[p]), 1 / pval**3))
            a_norm2 += norm2(avec[p])
        approximate_f, exact_rhs_f = {}, {}
        for p in modes:
            for orientation in (-1, 1):
                dest = tuple((j + orientation * q) % size
                             for j, q, size in zip(p, mode, sizes))
                tprime = -2 / pval**5 * (deltas[dest] + deltas[p]) * phi
                rhs_f = sum((t[dest]**j * tprime * t[p]**(2 * nu - 1 - j)
                             for j in range(2 * nu)), I(0))
                exact_rhs_f[dest, p] = rhs_f
                approximate_f[dest, p] = I(midpoint(rhs_f / (r[dest] + r[p])))
        f_residual2 = I(0)
        aprime = {}
        for (dest, p), fprime in approximate_f.items():
            residual = (r[dest] + r[p]) * fprime - exact_rhs_f[dest, p]
            f_residual2 += 2 * residual**2
            # Reevaluate the displayed source formula against enclosed exact
            # uniform inputs; construction rounding is included here.
            real = pval**-2 * phi * (-2 * r[p] + r[dest]) + fprime / pval
            aprime[dest, p] = ma(diagonal(real),
                                 ms(i_times(dirs[p]), -3 * phi / pval**4))
        f_residual = SQRT(f_residual2)
        minimum_mass = SQRT(I(mass))
        f_error = f_residual / (2 * minimum_mass)
        a_max = SQRT(a_norm2)  # Frobenius upper bound also bounds operator norm.
        residuals = [I(0), I(0)]
        packet_trace = I(0)
        for dest, p in aprime:
            work.check()
            source = aprime[dest, p]
            source_dagger = md(aprime[p, dest])
            rhs0 = ma(mm(source_dagger, avec[p]), mm(md(avec[dest]), source))
            rhs1 = ma(mm(source, md(avec[p])), mm(avec[dest], source_dagger))
            x0 = point_matrix(ms(rhs0, 1 / (e[dest] + e[p])))
            x1 = point_matrix(ms(rhs1, 1 / (e[dest] + e[p])))
            for j, (value, rhs) in enumerate(((x0, rhs0), (x1, rhs1))):
                residual = ma(ms(value, e[dest] + e[p]), mn(rhs))
                residuals[j] += norm2(residual)
            difference = ma(x1, mn(x0))
            packet_trace += phi * (difference[0][0][0] + difference[1][1][0])
        errors = [(SQRT(item) + 2 * a_max / pval * f_error)
                  / (2 * minimum_mass / pval) for item in residuals]
        trace_error = pval * 16 * sum(errors, I(0)) / m
        trace += sign * pval * packet_trace / m
        packet_errors.append(ENDS(trace_error)[1])
        packet_residuals.append({
            "mass_squared": str(mass), "Fprime_frobenius_residual_upper": arithmetic.enclosure(f_residual)["upper"],
            "X0_frobenius_residual_upper": arithmetic.enclosure(SQRT(residuals[0]))["upper"],
            "X1_frobenius_residual_upper": arithmetic.enclosure(SQRT(residuals[1]))["upper"],
            "source_and_solve_trace_error_upper": arithmetic.decimal_endpoint(ENDS(trace_error)[1], 40, True)})
        packet_momentum = I(0)
        for p in modes:
            dest = tuple((j + q) % size for j, q, size in zip(p, mode, sizes))
            difference2 = sum(((left - right)**2 for left, right in
                               zip(symbols[dest], symbols[p])), I(0))
            numerator = difference2 + pval**4 * (r[dest] - r[p])**2
            packet_momentum += numerator / (pval * (e[dest] + e[p]))
        momentum += sign * (-6 / (m * pval**5)) * packet_momentum
    total_error = sum(packet_errors, F(0))
    enclosed_trace = enlarged(trace, total_error)
    discrepancy = enclosed_trace - momentum
    lower_momentum, upper_momentum = ENDS(momentum)
    min_abs_momentum = (F(0) if lower_momentum <= 0 <= upper_momentum
                        else min(abs(lower_momentum), abs(upper_momentum)))
    tolerance = F(1, 10**8) * max(F(1), min_abs_momentum)
    passed = interval_abs_upper(discrepancy) <= tolerance
    return {
        "stencil": name, "nu": nu, "geometry": [8, 4, 4], "a": "1",
        "P": str(ruler), "N0": "9/10", "mode_indices": list(mode),
        "phi": "sqrt(2) cos(q.x); both Fourier modes retained; <phi^2>=1",
        "decimal_precision": iv.dps,
        "matrix_trace": arithmetic.enclosure(enclosed_trace, 32),
        "momentum_vertex": arithmetic.enclosure(momentum, 32),
        "difference": arithmetic.enclosure(discrepancy, 32),
        "positive_packet_residual_error_sum": arithmetic.decimal_endpoint(total_error, 40, True),
        "momentum_arithmetic_radius": arithmetic.decimal_endpoint((upper_momentum - lower_momentum) / 2, 40, True),
        "complete_tolerance_lower": arithmetic.decimal_endpoint(tolerance, 32, False),
        "packet_residuals": packet_residuals,
        "status": "certified_matrix_agreement" if passed else "unresolved_numerical_conditioning",
        "scope": "Finite source/operator algebra; no low-momentum physical window or full source jet"}


def matrix_certificate(name, nu, work):
    global PI, SQRT2
    original = iv.dps
    original_pi, original_sqrt2 = PI, SQRT2
    result = []
    try:
        for ruler, mode in itertools.product((F(1), F(6, 5)), ((1, 0, 0), (1, 1, 0))):
            retained = None
            for precision in (original, max(original, 120), max(original, 160)):
                iv.dps = precision
                # Regeneration removes the old iv rounding floor.  The frozen
                # exact Machin-series and 320-bit sqrt2 bracket widths remain
                # enclosed; failure at that input floor stays unresolved.
                PI, _, SQRT2, _, _ = arithmetic.constant_enclosures()
                retained = matrix_attempt(name, nu, ruler, mode, work)
                if retained["status"] == "certified_matrix_agreement":
                    break
            result.append(retained)
    finally:
        iv.dps = original
        PI, SQRT2 = original_pi, original_sqrt2
    return result


def grouping_diagnostics(nu):
    original = iv.dps
    cases = ((F(0), F(0)), (F(1, 10**6), F(1, 3 * 10**6)),
             (F(1, 10), F(1, 20)), (F(1), F(1, 3)),
             (F(100), F(20)), (F(10**10), F(10**10, 3)))
    try:
        iv.dps = max(original, 220)
        for y, yi in cases:
            expected_h, expected_g = direct_groups(I(y), I(yi), nu)
            h, g, _, _ = stable_groups(BI.rational(y), BI.rational(yi), nu)
            for binary, reference in ((h.to_iv(), expected_h), (g.to_iv(), expected_g)):
                blo, bhi = ENDS(binary)
                rlo, rhi = ENDS(reference)
                if not (blo <= rlo <= rhi <= bhi):
                    raise ArithmeticError("Stable grouped directed interval failed independent point containment")
    finally:
        iv.dps = original
    return {"symmetric_rectangle_asserted": True, "independent_point_cases": len(cases),
            "includes_cone_and_high_UV_cancellation": True,
            "reference_digits": max(original, 220),
            "reference_role": "Diagnostic only; operation enclosures and analytic budgets remain required"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nu", type=int, choices=(3, 2), default=3)
    parser.add_argument("--stencils", nargs="+", choices=("nn", "imp"), default=("nn", "imp"))
    parser.add_argument("--indices", nargs="+", type=int, default=list(range(13)))
    parser.add_argument("--digits", type=int, default=80)
    parser.add_argument("--root-bits", type=int, default=120)
    parser.add_argument("--max-order", type=int, choices=(8, 16, 32, 64, 128), default=64)
    parser.add_argument("--max-boxes", type=int, default=20000)
    parser.add_argument("--max-evaluations", type=int, default=100000000)
    parser.add_argument("--seconds", type=int, default=10800)
    parser.add_argument("--max-threads", type=int, default=2)
    parser.add_argument("--backend", choices=("binary", "iv"), default="binary")
    parser.add_argument("--derivative-check", action="store_true")
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--diagnostics-only", action="store_true")
    args = parser.parse_args()
    if (args.digits < 60 or args.root_bits < 100 or not 1 <= args.max_threads <= 8
            or any(not 0 <= j <= 12 for j in args.indices)):
        raise ValueError("Require digits>=60, roots>=100bits, coordinated1..8threads and indices0..12")
    # A single process holds the visible CUDA context throughout every test
    # and CPU integration; no child worker or numerical subprocess is used.
    execution = remote_execution(max_threads=args.max_threads)
    iv.dps = args.digits
    global PI, SQRT2
    PI, _, SQRT2, _, _ = arithmetic.constant_enclosures()
    contract = arithmetic.verify_binary64_contract()
    assert_rectangle()
    work = Work(args)
    result = {
        "source_contract": "solve/strand_E_uv_spacing.tex; solve/review_E_uv_spacing.tex independent PASS",
        "scope": ("Current nu=3 UV point diagnostic" if args.nu == 3
                  else "Earlier nu=2 rejected planar-benchmark control; no transfer to nu=3"),
        "parameters": {"Omega": "1", "delta": "1/1000",
                       "mass_squared": [str(x) for x in MASS_SQUARES],
                       "equal_mass_square_step": str(MASS_STEP), "orientations": list(GRADING)},
        "execution": execution,
        "arithmetic": {"frozen_helper": "solve/code/uv_native_certificate.py",
                       "interval_backend": "mpmath.iv with frozen directed BinaryInterval tensor backend",
                       "decimal_digits": args.digits, "directed_binary_contract": contract,
                       "iv_to_binary": "Exact dyadic endpoints then verified outward_float; no midpoint bounds",
                       "grouping_diagnostics": grouping_diagnostics(args.nu)},
        "points": [], "matrix": [],
        "requested_point_grid": {"stencils": list(args.stencils), "indices": args.indices},
        "all_requested_points_attempted": False,
        "all_requested_points_certified": False,
        "matrix_check": {"requested": args.matrix, "status": "unexecuted",
                         "scope": "Separate full source/operator check; not credited by coefficient points",
                         "constant_floor": "Frozen exact Machin brackets and 320-bit sqrt2; retained in residuals"},
        "independent_FD_check_requested": args.derivative_check,
        "limitations": ["delta=1/1000 point only; no uniform smaller-mass claim",
                        "K0 continuum normalization is fixed; finite-spacing drift is an output",
                        "No charge lock, gamma/beta source result, physical PPN or stellar window",
                        "Improved vacuum a^4 behavior does not establish improved entropy or source behavior",
                        "Resource/precision failures are unresolved numerical points"],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result["all_requested_points_attempted"] = (
            len(result["points"]) == len(args.stencils) * len(args.indices))
        result["all_requested_points_certified"] = (
            result["all_requested_points_attempted"]
            and all(point["certificate_status"] == "certified_point" for point in result["points"]))
        if result["matrix"]:
            result["matrix_check"]["status"] = (
                "certified" if len(result["matrix"]) == 4 * len(args.stencils)
                and all(item["status"] == "certified_matrix_agreement" for item in result["matrix"])
                else "unresolved_or_partial")
        result["elapsed_seconds"] = time.perf_counter() - work.start
        result["node_evaluations"] = work.evaluations
        result["arithmetic"]["certified_Gauss_rules"] = [item[2] for item in work.gauss.values()]
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    if args.diagnostics_only:
        result["diagnostics_status"] = "Passed guarded arithmetic and independent grouped-point checks"
        save()
        print(result["diagnostics_status"], flush=True)
        return
    try:
        print(f"nu={args.nu}: certifying the point-positive continuum reference", flush=True)
        k0, continuum = continuum_certificate(args.nu, work)
        result["continuum"] = continuum
        save()
        if continuum["certificate_status"] != "certified":
            result["termination"] = "Continuum target unresolved; sole match not released"
            save()
            return
        k_reference = ENDS(k0)[0]
        for name, j in itertools.product(args.stencils, args.indices):
            work.check()
            tau = I(F(1, 20 * 2**j))
            print(f"nu={args.nu} {name} tau=1/{20 * 2**j}: enclosing full finite domain", flush=True)
            point, kval, dval, leaves = coefficient_point(tau, name, args.nu, k_reference,
                                                        work, args.backend)
            point["index"] = j
            point["normalization_defect_K_over_K0_minus1"] = arithmetic.enclosure(kval / k0 - 1, 32)
            point["a_over_lP_at_P1"] = arithmetic.enclosure(tau * SQRT(2 * PI * k0), 32)
            if args.derivative_check:
                point["independent_log_derivative_check"] = derivative_check(
                    tau, name, args.nu, kval, dval, leaves, work)
            result["points"].append(point)
            save()
            print(f"{name} j={j}: {point['certificate_status']}; {point['termination']}", flush=True)
            if time.perf_counter() - work.start > args.seconds:
                raise LimitReached("Declared total time cap reached")
        if args.matrix:
            for name in args.stencils:
                print(f"nu={args.nu} {name}: independently assembling complete finite source", flush=True)
                result["matrix"].extend(matrix_certificate(name, args.nu, work))
                save()
    except LimitReached as error:
        result["termination"] = str(error)
    except (ArithmeticError, FloatingPointError) as error:
        result["termination"] = "Unresolved arithmetic enclosure: " + str(error)
    save()
    print(f"Saved {args.output}; all numerical results require reviewer B source/result review", flush=True)


if __name__ == "__main__":
    main()
