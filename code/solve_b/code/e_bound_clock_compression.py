#!/usr/bin/env python3
"""Standalone directed enclosure of the SAME frozen exact trial compression.

Governing source: strand_E_clock_compression.tex GC1--GC20 and its
independent review.  This program supplies no ground state, model point,
Ramsey/clock or COM certificate.  Author E and independent B actual-source
audits must precede separately admitted remote execution.  Existing ground
and Ramsey programs are neither imported nor modified by this entry point.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction as F
import json
import math
import os
from pathlib import Path
import time

from remote_runtime import remote_execution


ALPHA, RHO = F(1, 10**6), F(5, 2)
CUTOFFS = (16, 24, 32, 48, 64)
QUADRATURES = (32, 64, 128, 256, 512)
PRECISIONS = (128, 256, 512, 1024, 2048)
SPEC = "solve/strand_E_clock_compression.tex"
REVIEW = "solve/review_E_clock_compression.tex"
POOL_NAMES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
FIXED_PARAMETERS = {
    "m_over_mP": "1/1000", "a_over_lP": "1/1000", "alpha": "1/1000000",
    "Lambda_over_m": "1", "ell_times_m": "1", "gB": "0",
    "sector": "even standard-spinor singlet, half-period character+, K=0, positive-charge two excitations",
    "law": "H_t=exp(-gamma*T_t)*(2*epsilon_t)*exp(-gamma*T_t)",
    "gamma_rule": "log(4*Emax)/(2*mu2), same exact trial compression at every refinement",
    "sea": "W0=0; the unique fixed symmetric Newton-match rectangle is not refitted",
}
TRIAL_IDENTIFIER = "v0=f, v1=epsilon*f in L2(nu_alpha), at reference t=0"
MEASURE_IDENTIFIER = "original sine pushforward nu_alpha; total mass=1/(8*alpha^3); w=r^2*W(alpha^2*r^2)/(2*pi^2) only for alpha*r<1"


class Unresolved(RuntimeError):
    """A stated directed gate or required input/control is not completed."""


class EngineeringCap(Unresolved):
    """The one job's predeclared total resources are exhausted."""


class InconsistentEnclosures(Unresolved):
    """Disjoint claimed enclosures must be retained and sent for review."""


class BallBackend:
    """Pinned Arb real arithmetic, activated only after actual admission."""

    def __init__(self, module):
        self.module, self.arb, self.ctx = module, module.arb, module.ctx
        self.admission = None

    @classmethod
    def load_host(cls):
        import flint
        if getattr(flint, "__version__", None) != "0.9.0":
            raise RuntimeError("The isolated pinned python-flint0.9.0 backend is required")
        for name in ("legendre_p_root", "man_exp", "sqrt", "log", "exp"):
            if not hasattr(flint.arb, name):
                raise RuntimeError("Pinned directed backend lacks the required Arb API: " + name)
        return cls(flint)

    def activate(self, admission):
        required = ("same_process_fresh_device_query_before_allocation",
                    "cuda_context_in_this_single_job_process",
                    "post_allocation_single_owned_compute_pid_verified")
        if any(admission.get(name) is not True for name in required):
            raise RuntimeError("Direct compression lacks complete same-process launch admission")
        if admission.get("single_process_pid") != os.getpid() or os.nice(0) != 10:
            raise RuntimeError("Admission does not identify this nice10 process")
        if os.environ.get("QI2_REMOTE_RUN") != "1":
            raise RuntimeError("Direct compression is authorized only in the selected remote job")
        self.admission = admission
        self.ctx.threads = 1
        self.ctx.dps = PRECISIONS[0]

    def require(self):
        if self.admission is None:
            raise RuntimeError("Scientific Arb arithmetic was requested before admission")

    def precision(self, digits):
        self.require()
        if type(digits) is not int or digits not in PRECISIONS:
            raise Unresolved("Precision is outside the frozen arithmetic ladder")
        self.ctx.dps = digits

    def real(self, value=0):
        self.require()
        if isinstance(value, float):
            raise TypeError("Binary64 values are not scientific compression inputs")
        if isinstance(value, F):
            return self.arb(value.numerator) / value.denominator
        return self.arb(value)

    def positive(self, value, label, gates=None):
        ok = value.is_finite() and value.lower() > 0
        if gates is not None:
            gates[label] = ok
        if not ok:
            raise Unresolved(label + " lacks a strictly positive directed lower endpoint")
        return value

    def absolute_upper(self, value):
        return abs(value).upper()

    def hull(self, lower, upper):
        lo, hi = self.real(lower).lower(), self.real(upper).upper()
        if not lo.is_finite() or not hi.is_finite() or lo > hi:
            raise Unresolved("A directed endpoint hull is nonfinite or unordered")
        center = (lo + hi) / 2
        return center + self.arb(0, ((hi - lo) / 2).upper())

    def symmetric_error(self, value, bound):
        error = self.real(bound)
        if not error.is_finite() or error.lower() < 0:
            raise Unresolved("An absolute error bound is nonfinite or negative")
        return value + self.arb(0, error.upper())

    def nonnegative_part(self, value, label):
        """Intersect with proved positivity; never manufacture a strict margin."""
        if not value.is_finite() or value.upper() < 0:
            raise InconsistentEnclosures(label + " contradicts the exact positive-integrand identity")
        return self.hull(max(self.real(0), value.lower()), value.upper())

    def encode(self, value):
        if value is None:
            return None
        if not value.is_finite():
            return {"status": "non-finite/unresolved"}
        def endpoint(bound):
            mantissa, exponent = bound.man_exp()
            return {"mantissa": str(mantissa), "exponent2": int(exponent)}
        return {"lower_dyadic": endpoint(value.lower()), "upper_dyadic": endpoint(value.upper()),
                "display_ball": value.str(30)}

    def decode(self, data):
        """Exact dyadic endpoint input only; display_ball is never interpreted."""
        self.require()
        if not isinstance(data, dict) or not all(key in data for key in ("lower_dyadic", "upper_dyadic")):
            raise Unresolved("Prior compression quantity lacks exact dyadic endpoints")
        exact = []
        for key in ("lower_dyadic", "upper_dyadic"):
            endpoint = data[key]
            if not isinstance(endpoint, dict) or not isinstance(endpoint.get("mantissa"), str):
                raise Unresolved("Prior endpoint mantissa is not an integer string")
            if type(endpoint.get("exponent2")) is not int:
                raise Unresolved("Prior endpoint exponent is not a genuine integer")
            try:
                mantissa = int(endpoint["mantissa"])
            except ValueError as error:
                raise Unresolved("Prior endpoint mantissa cannot be decoded exactly") from error
            value = self.arb((mantissa, endpoint["exponent2"]))
            if not value.is_exact():
                raise Unresolved("Pinned backend did not preserve an exact dyadic endpoint")
            exact.append(value)
        if exact[0] > exact[1]:
            raise Unresolved("Prior compression endpoints are reversed")
        return exact[0] if data["lower_dyadic"] == data["upper_dyadic"] else self.hull(*exact)


class Work:
    """One total 900s/2M/4096MB job; no numerical budget reset per control."""

    def __init__(self, backend, args):
        backend.require()
        self.b, self.args, self.start = backend, args, time.perf_counter()
        self.evaluations, self.rules, self.coefficients = 0, {}, {}

    def check(self, increment=0):
        if type(increment) is not int or increment < 0:
            raise RuntimeError("A work charge must be a nonnegative integer")
        if increment > self.args.max_evaluations - self.evaluations:
            raise EngineeringCap("The total direct-compression work cap is exhausted")
        self.evaluations += increment
        if time.perf_counter() - self.start > self.args.seconds:
            raise EngineeringCap("The total direct-compression elapsed-time cap is exhausted")
        if os.nice(0) != 10 or len(os.listdir("/proc/self/task")) > 4:
            raise RuntimeError("The admitted nice10/single-worker/four-thread contract was violated")
        with open("/proc/self/status") as stream:
            for line in stream:
                if line.startswith("VmRSS:") and int(line.split()[1]) > self.args.memory_mb * 1024:
                    raise EngineeringCap("The total direct-compression resident-memory cap is exhausted")

    def rule(self, order):
        b = self.b
        if type(order) is not int or order not in QUADRATURES:
            raise Unresolved("Scalar Gauss order is outside the frozen ladder")
        key = b.ctx.prec, order
        if key not in self.rules:
            values = []
            for index in range(order):
                self.check(1)
                node, weight = b.arb.legendre_p_root(order, index, weight=True)
                b.positive(weight, "Certified algebraic Gauss weight")
                if not node.is_finite() or not node.lower() > -1 or not node.upper() < 1:
                    raise Unresolved("A certified algebraic Gauss node leaves (-1,1)")
                values.append((node, weight))
            if not sum((weight for _, weight in values), b.real(0)).contains(2):
                raise Unresolved("The exact positive Gauss rule does not enclose its total weight 2")
            self.rules[key] = values
        return self.rules[key]

    def angular_coefficients(self, order):
        """The exact finite rational coefficient formula in the frozen measure."""
        if type(order) is not int or order not in (8, 16, 32, 64, 128):
            raise Unresolved("Weight-series order leaves the existing 8..128 ladder")
        if order in self.coefficients:
            return self.coefficients[order]
        def odd_double_factorial(n):
            value = 1
            for item in range(n, 0, -2):
                value *= item
            return value
        values = []
        for k in range(order + 1):
            total = F(0)
            for k1 in range(k + 1):
                for k2 in range(k - k1 + 1):
                    self.check(1)
                    k3 = k - k1 - k2
                    term = F(1, odd_double_factorial(2 * k + 1))
                    for ki in (k1, k2, k3):
                        term *= F(math.comb(2 * ki, ki) * odd_double_factorial(2 * ki - 1), 4**ki)
                    total += term
            if not 0 <= total <= 1:
                raise Unresolved("An exact angular coefficient leaves its proved [0,1] range")
            values.append(total)
        if values[:3] != [F(1), F(1, 2), F(11, 40)]:
            raise Unresolved("The exact angular-series normalization control failed")
        self.coefficients[order] = values
        return values


@dataclass(frozen=True)
class Panel:
    left: F
    right: F

    @property
    def length(self):
        return self.right - self.left

    def saved(self):
        return {"left_exact_rational": str(self.left), "right_exact_rational": str(self.right)}


def panels_for(cutoff):
    if type(cutoff) is not int or cutoff not in CUTOFFS:
        raise Unresolved("Cutoff is outside the frozen radial ladder")
    endpoints = [F(0), F(1, 2), F(1), F(2), F(4), F(8), F(16)]
    endpoints.extend(F(value) for value in CUTOFFS if 16 < value <= cutoff)
    return [Panel(left, right) for left, right in zip(endpoints, endpoints[1:])]


@dataclass(frozen=True)
class Configuration:
    cutoff: int
    quadrature: int
    digits: int
    weight_order: int

    def saved(self):
        return {"cutoff": self.cutoff, "positive_scalar_Gauss_order_per_panel": self.quadrature,
                "precision_decimal_digits": self.digits, "weight_series_last_power": self.weight_order,
                "panels": [panel.saved() for panel in panels_for(self.cutoff)],
                "Legendre_projection_or_free_spectrum_discretization": False}


def panel_bounds(backend, panel, saved):
    """GC15--GC18 bounds on the complete rho=5/2 Bernstein ellipse."""
    b = backend
    center, half = (panel.left + panel.right) / 2, panel.length / 2
    major = half * (RHO + 1 / RHO) / 2
    minor = half * (RHO - 1 / RHO) / 2
    xlo, xhi = center - major, center + major
    distance = F(0) if xlo <= 0 <= xhi else min(abs(xlo), abs(xhi))
    delta_exact = 1 + distance**2 - minor**2
    v_exact = max(F(0), minor**2 - distance**2)
    radius_exact = center + RHO * half
    series_argument_exact = ALPHA**2 * radius_exact**2
    saved.update({"panel": panel.saved(), "rho_exact_rational": str(RHO),
                  "rectangle_x_lower_exact": str(xlo), "rectangle_x_upper_exact": str(xhi),
                  "rectangle_y_absolute_upper_exact": str(minor),
                  "distance_from_zero_to_x_interval_exact": str(distance),
                  "delta_exact_rational": str(delta_exact), "v_exact_rational": str(v_exact),
                  "R_exact_rational": str(radius_exact),
                  "alpha2_R2_exact_rational": str(series_argument_exact),
                  "gates": {"exact_root_margin_positive": delta_exact > 0,
                            "exact_full_sphere_series_margin_positive": series_argument_exact < 1}})
    if delta_exact <= 0 or series_argument_exact >= 1:
        raise Unresolved("The full panel ellipse lacks a proved root or lattice-series margin")
    delta = b.positive(b.real(delta_exact), "Directed panel root margin", saved["gates"])
    series_margin = b.positive(b.real(1 - series_argument_exact), "Directed lattice-series margin", saved["gates"])
    root_lower = b.positive(delta.sqrt().lower(), "Principal epsilon real-part lower bound", saved["gates"])
    epsilon_upper = (1 + b.real(radius_exact)**2).sqrt().upper()
    leg_upper = ((1 + 1 / root_lower) / 2).upper()
    weight_upper = (b.real(radius_exact)**2 / (2 * b.arb.pi()**2 * series_margin)).upper()
    gaussian_upper = b.real(2 * v_exact).exp().upper()
    moment_bounds = [(weight_upper * leg_upper**2 * gaussian_upper * epsilon_upper**k).upper()
                     for k in range(3)]
    saved.update({"epsilon_real_part_lower": b.encode(root_lower),
                  "epsilon_absolute_upper": b.encode(epsilon_upper),
                  "both_spinor_legs_without_Gaussian_absolute_upper": b.encode(leg_upper),
                  "exact_weight_density_absolute_upper": b.encode(weight_upper),
                  "Gaussian_square_absolute_upper": b.encode(gaussian_upper),
                  "moment_integrand_absolute_bounds_GC17": [b.encode(value) for value in moment_bounds]})
    return {"root_lower": root_lower, "moments": moment_bounds}


def analytic_errors(backend, work, panels, quadrature, record):
    b = backend
    cells = []
    record["complete_ellipse_branch_and_majorant_records"] = []
    for panel in panels:
        work.check(1)
        saved = {}
        record["complete_ellipse_branch_and_majorant_records"].append(saved)
        cells.append(panel_bounds(b, panel, saved))
    power = b.real(RHO)**(-2 * quadrature)
    denominator = b.positive(1 - 1 / b.real(RHO), "Chebyshev geometric-series margin")
    moments = [b.real(0) for _ in range(3)]
    moment_panels, double_panels = [], []
    for panel, cell in zip(panels, cells):
        local = [(4 * b.real(panel.length) * cell["moments"][k] * power / denominator).upper()
                 for k in range(3)]
        for k in range(3):
            moments[k] += local[k]
        moment_panels.append({"panel": panel.saved(), "absolute_errors_GC19": [b.encode(value) for value in local]})
    diagonals = [b.real(0), b.real(0)]
    for left, (pi, bi) in enumerate(zip(panels, cells)):
        for right, (pj, bj) in enumerate(zip(panels, cells)):
            work.check(1)
            sum_roots = b.positive(bi["root_lower"] + bj["root_lower"], "Product-ellipse denominator lower bound")
            majorants = [(2 * bi["moments"][index] * bj["moments"][index] / sum_roots).upper()
                         for index in range(2)]
            # Both tensor telescoping terms remain: for equal Q the
            # coefficient is eight, not four, times the full GC18 M.
            local = [(4 * b.real(pi.length * pj.length) * majorants[index]
                      * (power + power) / denominator).upper() for index in range(2)]
            for index in range(2):
                diagonals[index] += local[index]
            double_panels.append({"left_panel": left, "right_panel": right,
                                  "integrand_absolute_bounds_GC18": [b.encode(value) for value in majorants],
                                  "absolute_errors_GC20": [b.encode(value) for value in local]})
    output = {"U0": moments[0].upper(), "U1": moments[1].upper(),
              "U2": moments[2].upper(), "V0": moments[0].upper(),
              "K00": diagonals[0].upper(), "K11": diagonals[1].upper()}
    record["analytic_quadrature_error_allocation"] = {
        "moment_panel_bounds": moment_panels, "ordered_product_panel_bounds": double_panels,
        "whole_finite_region_absolute_errors": {name: b.encode(value) for name, value in output.items()},
        "exact_rule_positive_and_full_tensor_error_used": True,
        "requested_tolerance_used_as_certificate": False,
    }
    return output


def moment_tails(backend, cutoff, saved):
    b = backend
    mass = 1 / (8 * b.real(ALPHA)**3)
    epsilon_max = (1 + 3 / b.real(ALPHA)**2).sqrt()
    exponential = b.real(-2 * cutoff**2).exp()
    tails = [(mass * epsilon_max**k * exponential).upper() for k in range(3)]
    saved.update({"full_lattice_total_measure": b.encode(mass),
                  "full_lattice_epsilon_max": b.encode(epsilon_max),
                  "tail_Gaussian_square_upper": b.encode(exponential.upper()),
                  "tau_k_GC11": [b.encode(value) for value in tails],
                  "uses_full_measure_including_restricted_angles_and_van_Hove_region": True,
                  "full_sphere_density_continued_to_outer_region": False})
    return epsilon_max, tails


def generalized_root(backend, values, saved):
    """Directed GC9--GC10 algebra; each gate is saved before it can fail."""
    b = backend
    gates, quantities = {}, {}
    saved["positivity_and_root_gates"] = gates
    saved["generalized_root_intermediate_quantities"] = quantities
    def retain(name, value):
        quantities[name] = b.encode(value)
        return value
    gram = [[values["U0"], values["U1"]], [values["U1"], values["U2"]]]
    compression = [[values["K00"], values["K01"]], [values["K01"], values["K11"]]]
    saved["Gram"] = [[b.encode(value) for value in row] for row in gram]
    saved["compression"] = [[b.encode(value) for value in row] for row in compression]
    for name, value in (("Gram00", gram[0][0]), ("Gram11", gram[1][1]),
                        ("compression00", compression[0][0]), ("compression11", compression[1][1])):
        b.positive(value, name, gates)
    detg = retain("detG", gram[0][0] * gram[1][1] - gram[0][1]**2)
    detk = retain("detK", compression[0][0] * compression[1][1] - compression[0][1]**2)
    b.positive(detg, "Full exact Gram determinant", gates)
    b.positive(detk, "Full exact compression determinant", gates)
    trace = retain("t", (compression[0][0] * gram[1][1] + compression[1][1] * gram[0][0]
                         - 2 * compression[0][1] * gram[0][1]) / detg)
    determinant = retain("d", detk / detg)
    b.positive(trace, "Generalized trace", gates)
    b.positive(determinant, "Generalized determinant", gates)
    discriminant = retain("t_squared_minus_4d", trace**2 - 4 * determinant)
    b.positive(discriminant, "Generalized-eigenvalue discriminant", gates)
    denominator = retain("t_plus_root_discriminant", trace + discriminant.sqrt())
    b.positive(denominator, "Stable smaller-root denominator", gates)
    mu2 = retain("mu2", 2 * determinant / denominator)
    b.positive(mu2, "Same frozen smaller generalized trial eigenvalue", gates)
    emax = retain("Emax", 2 * (1 + 3 / b.real(ALPHA)**2).sqrt())
    gamma = retain("gamma", (4 * emax).log() / (2 * mu2))
    b.positive(gamma, "Same frozen exact contact coupling", gates)
    saved.update({"mu2": b.encode(mu2), "gamma": b.encode(gamma),
                  "generalized_root_formula": "2*d/(t+sqrt(t^2-4*d)), GC10",
                  "same_exact_coupling_rule_across_all_refinements": True})
    return {**values, "G00": gram[0][0], "G01": gram[0][1], "G10": gram[1][0], "G11": gram[1][1],
            "K10": compression[1][0], "mu2": mu2, "gamma": gamma}


def lattice_weight(backend, radius, coefficients):
    """Full W node enclosure, with its positive real-series remainder."""
    b = backend
    z = b.real(ALPHA)**2 * radius**2
    if not z.is_finite() or z.lower() < 0 or not z.upper() < 1:
        raise Unresolved("The real Gauss node lacks the exact full-sphere weight-series domain")
    polynomial = b.real(0)
    for coefficient in reversed(coefficients):
        polynomial = polynomial * z + b.real(coefficient)
    b.positive(polynomial, "Positive finite angular weight series")
    z_upper = z.upper()
    remainder = (z_upper**len(coefficients) / (1 - z_upper)).upper()
    return polynomial, polynomial + b.hull(0, remainder), remainder


def real_node(backend, radius, quadrature_weight, coefficients):
    b = backend
    eps = b.positive(1 + radius**2, "Real-node principal-root argument").sqrt()
    b.positive(eps, "Real-node principal epsilon")
    gaussian = (-radius**2).exp()
    f = gaussian * (eps + 1) / (2 * eps)
    j = b.nonnegative_part(gaussian * (eps - 1) / (2 * eps), "Real lower-spinor leg")
    A, B = f**2, f * j
    polynomial, full_weight, remainder = lattice_weight(b, radius, coefficients)
    density_factor = radius**2 / (2 * b.arb.pi()**2)
    wp, wf = density_factor * polynomial, density_factor * full_weight
    return {"epsilon": eps, "polynomial_A": quadrature_weight * wp * A,
            "polynomial_B": quadrature_weight * wp * B,
            "full_A": quadrature_weight * wf * A, "full_B": quadrature_weight * wf * B,
            "weight_series_remainder_upper": remainder}


def corrected_scalar(backend, polynomial, full_rule, quadrature_error, positive_tail, label):
    b = backend
    low = b.nonnegative_part(b.symmetric_error(full_rule, quadrature_error), label + " finite exact integral")
    final = low + b.hull(0, positive_tail)
    saved = {"finite_region_polynomial_weight_rule": b.encode(polynomial),
             "polynomial_only_arithmetic_radius_upper_diagnostic": b.encode(polynomial.rad().upper()),
             "finite_region_full_W_Gauss_value": b.encode(full_rule),
             "full_rule_joint_arithmetic_and_weight_series_radius_upper": b.encode(full_rule.rad().upper()),
             "full_minus_polynomial_rule_absolute_upper_decorrelation_diagnostic":
                 b.encode(b.absolute_upper(full_rule - polynomial)),
             "analytic_quadrature_absolute_error_upper": b.encode(quadrature_error),
             "finite_exact_integral_after_Gauss_error": b.encode(low),
             "positive_full_outer_region_bound_upper": b.encode(positive_tail),
             "full_exact_integral": b.encode(final),
             "correction_formula": "full_W_rule + symmetric analytic Gauss bound + [0,positive outer tail]",
             "polynomial_and_decorrelation_diagnostics_added_as_extra_errors": False,
             "Legendre_projection_error": "not applicable: direct exact scalar integral",
             "full_W_remainder_already_propagated_in_rule_ball": True}
    return final, low, saved


def evaluate_configuration(configuration, backend, work, record):
    """One whole exact compression enclosure; saved progress survives caps."""
    b = backend
    b.precision(configuration.digits)
    work.check()
    record.update({"configuration": configuration.saved(), "status": "unresolved",
                   "method": "exact_scalar_moments_product_Gauss",
                   "trial_identifier": TRIAL_IDENTIFIER, "measure_identifier": MEASURE_IDENTIFIER,
                   "reference_source_t_exact": "0", "moments": {name: None for name in ("U0", "U1", "U2", "V0")},
                   "diagonal_compression_entries": {"K00": None, "K11": None},
                   "Gram": None, "compression": None, "mu2": None, "gamma": None,
                   "positivity_and_root_gates": {},
                   "full_W_uncertainty_record_semantics": "joint directed arithmetic/node/weight/function and exact-series radius; polynomial-only and subtraction diagnostics are not extra errors",
                   "Cartesian_measure_control_status": "deferred_to_ground_stage",
                   "ground_adoption_allowed": False})
    panels = panels_for(configuration.cutoff)
    errors = analytic_errors(b, work, panels, configuration.quadrature, record)
    coefficients = work.angular_coefficients(configuration.weight_order)
    record["weight_series"] = {"exact_coefficients": [str(value) for value in coefficients],
                              "last_power": configuration.weight_order,
                              "remainder": "z^(K+1)/(1-z) on real Gauss nodes; |z|^(K+1)/(1-|z|) on analytic domain",
                              "first_three_exact_coefficients_control_pass": True}
    record["positive_full_lattice_tails"] = {}
    epsilon_max, tails = moment_tails(b, configuration.cutoff, record["positive_full_lattice_tails"])
    rule = work.rule(configuration.quadrature)
    names = ("U0", "U1", "U2", "V0")
    polynomial_sums = {name: b.real(0) for name in names}
    full_sums = {name: b.real(0) for name in names}
    nodes = []
    record["certified_algebraic_Gauss_rule"] = {
        "backend_API": "arb.legendre_p_root(Q,index,weight=True)",
        "all_real_nodes_inside_panel_and_all_weights_positive": False,
        "root_solver_tolerance_used_as_certificate": False, "panels": []}
    expected_nodes = len(panels) * configuration.quadrature
    moment_nodes_completed = 0
    try:
        for panel_index, panel in enumerate(panels):
            half, center = b.real(panel.length / 2), b.real((panel.left + panel.right) / 2)
            saved_panel = {"panel": panel.saved(), "nodes_and_weights": [], "completed": False}
            record["certified_algebraic_Gauss_rule"]["panels"].append(saved_panel)
            for index, (abscissa, gauss_weight) in enumerate(rule):
                work.check(1)
                radius, weight = center + half * abscissa, half * gauss_weight
                if radius.lower() < b.real(panel.left).lower() or radius.upper() > b.real(panel.right).upper():
                    raise Unresolved("An enclosed Gauss node leaves its exact radial panel")
                b.positive(weight, "Mapped positive algebraic Gauss weight")
                node = real_node(b, radius, weight, coefficients)
                node["panel_index"] = panel_index
                nodes.append(node)
                saved_panel["nodes_and_weights"].append({
                    "algebraic_root_index": index, "r": b.encode(radius), "positive_dr_weight": b.encode(weight),
                    "actual_radius_of_r_upper": b.encode(radius.rad().upper()),
                    "actual_radius_of_dr_weight_upper": b.encode(weight.rad().upper()),
                    "W_series_remainder_upper": b.encode(node["weight_series_remainder_upper"])})
                for k in range(3):
                    polynomial_sums["U" + str(k)] += node["epsilon"]**k * node["polynomial_A"]
                    full_sums["U" + str(k)] += node["epsilon"]**k * node["full_A"]
                polynomial_sums["V0"] += node["polynomial_B"]
                full_sums["V0"] += node["full_B"]
                moment_nodes_completed += 1
            saved_panel["completed"] = True
        record["certified_algebraic_Gauss_rule"]["all_real_nodes_inside_panel_and_all_weights_positive"] = True
    finally:
        record["moment_rule_accumulation_progress"] = {
            "evaluated_nodes": len(nodes), "moment_nodes_completed": moment_nodes_completed,
            "required_nodes": expected_nodes,
            "finite_region_node_sum_completed": moment_nodes_completed == expected_nodes,
            "polynomial_weight_prefix": {name: b.encode(value) for name, value in polynomial_sums.items()},
            "full_W_prefix": {name: b.encode(value) for name, value in full_sums.items()},
            "incomplete_prefix_is_not_a_full_integral_enclosure": moment_nodes_completed != expected_nodes}
    values, low_values = {}, {}
    for name in names:
        k = int(name[1])
        values[name], low_values[name], record["moments"][name] = corrected_scalar(
            b, polynomial_sums[name], full_sums[name], errors[name], tails[k], name)
    record["Gram"] = [[b.encode(values["U0"]), b.encode(values["U1"])],
                      [b.encode(values["U1"]), b.encode(values["U2"])] ]
    # GC12 contains both low/high rectangles AND both high/high squares.
    tau0 = tails[0]
    exterior_product_mass = (2 * low_values["U0"].upper() * tau0 + tau0**2
                             + 2 * low_values["V0"].upper() * tau0 + tau0**2).upper()
    diagonal_tails = [(exterior_product_mass / 2).upper(), (epsilon_max * exterior_product_mass / 2).upper()]
    record["positive_full_lattice_tails"].update({
        "U0_low_exact_integral_upper_for_GC12": b.encode(low_values["U0"].upper()),
        "V0_low_exact_integral_upper_for_GC12": b.encode(low_values["V0"].upper()),
        "both_channel_cross_and_high_high_product_mass_GC12": b.encode(exterior_product_mass),
        "K00_outer_upper_GC13": b.encode(diagonal_tails[0]),
        "K11_outer_upper_GC14": b.encode(diagonal_tails[1]),
        "K01_outer_upper_GC6_GC12_diagnostic_only": b.encode(diagonal_tails[0])})
    values["K01"] = (values["U0"]**2 + values["V0"]**2) / 2
    record["off_diagonal_exact_moment_identity"] = {
        "formula": "K01=K10=(U0^2+V0^2)/2, GC6, using FULL moments",
        "full_exact_K01": b.encode(values["K01"]),
        "direct_double_sum_evaluated": False,
        "independent_numerical_Laplace_control_evaluated": False,
        "GC6_and_GC7_analytical_controls_preserved": True,
        "extra_exterior_error_added_to_full_moment_identity": False}
    poly_diagonals, full_diagonals = [b.real(0), b.real(0)], [b.real(0), b.real(0)]
    processed_pairs, required_pairs = 0, len(nodes) * (len(nodes) + 1) // 2
    try:
        for row, left in enumerate(nodes):
            for column in range(row, len(nodes)):
                work.check(4)  # Two diagonal integrands, polynomial and full-W rules.
                right = nodes[column]
                denominator = b.positive(left["epsilon"] + right["epsilon"], "Real double-integral denominator")
                multiplicity = 1 if row == column else 2
                poly_term = multiplicity * (left["polynomial_A"] * right["polynomial_A"]
                                           + left["polynomial_B"] * right["polynomial_B"]) / denominator
                full_term = multiplicity * (left["full_A"] * right["full_A"]
                                           + left["full_B"] * right["full_B"]) / denominator
                poly_diagonals[0] += poly_term
                full_diagonals[0] += full_term
                poly_diagonals[1] += left["epsilon"] * right["epsilon"] * poly_term
                full_diagonals[1] += left["epsilon"] * right["epsilon"] * full_term
                processed_pairs += 1
    finally:
        record["double_rule_accumulation_progress"] = {
            "processed_symmetric_node_pairs": processed_pairs, "required_symmetric_node_pairs": required_pairs,
            "full_ordered_node_pair_count": len(nodes)**2,
            "off_diagonal_node_pair_multiplicity_exact": 2,
            "finite_square_double_sum_completed": processed_pairs == required_pairs,
            "polynomial_weight_prefix_K00_K11": [b.encode(value) for value in poly_diagonals],
            "full_W_prefix_K00_K11": [b.encode(value) for value in full_diagonals],
            "incomplete_prefix_is_not_a_full_double_integral_enclosure": processed_pairs != required_pairs}
    for index, name in enumerate(("K00", "K11")):
        values[name], _, record["diagonal_compression_entries"][name] = corrected_scalar(
            b, poly_diagonals[index], full_diagonals[index], errors[name], diagonal_tails[index], name)
    root_values = generalized_root(b, values, record)
    record["status"] = "this_configuration_full_integrals_and_root_gates_enclosed"
    return root_values


def decode_matrix(backend, work, data, label, nonnegative_bounds=False):
    if not isinstance(data, list) or len(data) != 2 or any(not isinstance(row, list) or len(row) != 2 for row in data):
        raise Unresolved(label + " is not a complete real 2x2 exact-endpoint matrix")
    output = []
    for row in data:
        decoded = []
        for entry in row:
            work.check(1)
            value = backend.decode(entry)
            if not value.is_finite() or (nonnegative_bounds and value.lower() < 0):
                raise Unresolved(label + " has a nonfinite quantity or a negative absolute-error bound")
            decoded.append(value)
        output.append(decoded)
    return output


def read_prior_ground(path, backend, work, saved):
    """The reviewed retry3 PARTIAL artifact, never an imported ground state."""
    b = backend
    saved.update({"path": str(path), "status": "unresolved", "header_validated": False,
                  "completed_phase_rows": [], "skipped_incomplete_phase_rows": [],
                  "top_level_mu2_gamma_used": False, "ground_certificate_or_vector_imported": False,
                  "initial_declared_prior": "ground retry3; all completed N16/N32/N64 matrices, incomplete N128 skipped"})
    work.check()
    if path.stat().st_size > work.args.memory_mb * 1024**2:
        raise EngineeringCap("Prior-artifact text alone exceeds the job's declared resident-memory allowance")
    data = json.loads(path.read_text())
    work.check()
    if not isinstance(data, dict):
        raise Unresolved("The declared prior ground artifact is not a JSON object")
    expected_header = {"stage": "ground", "source": "solve/strand_E_clock_spec.tex",
                       "independent_source_review": "solve/review_E_clock_spec.tex",
                       "independent_ground_code_review": "solve/review_E_ground_code.tex",
                       "backend": "python-flint0.9.0; Arb/Acb directed ball operations",
                       "scope": "original sine-polar contact completion; externally prepared uniform K=0 pair"}
    for key, expected in expected_header.items():
        if data.get(key) != expected:
            raise Unresolved("Prior artifact does not identify the frozen ground source/units/backend: " + key)
    if data.get("fixed_parameters") != FIXED_PARAMETERS:
        raise Unresolved("Prior compression belongs to a different frozen point/sector/law/coupling rule")
    admission = data.get("admission")
    required = ("same_process_fresh_device_query_before_allocation",
                "cuda_context_in_this_single_job_process", "post_allocation_single_owned_compute_pid_verified")
    if not isinstance(admission, dict) or any(admission.get(key) is not True for key in required):
        raise Unresolved("Prior artifact lacks its directed remote launch-admission records")
    if type(admission.get("single_process_pid")) is not int or admission["single_process_pid"] <= 0:
        raise Unresolved("Prior artifact lacks a valid generating process identifier")
    if admission.get("niceness") != 10 or admission.get("maximum_threads") != 4:
        raise Unresolved("Prior generating process did not use the declared nice10/four-thread scope")
    if not isinstance(admission.get("assigned_gpu_uuid"), str) or not admission["assigned_gpu_uuid"].startswith("GPU-"):
        raise Unresolved("Prior artifact lacks its assigned device identifier")
    attempts = data.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise Unresolved("Prior artifact has no declared attempt rows")
    saved.update({"header_validated": True, "top_level_ground_status": data.get("status"),
                  "source": data["source"], "stage": data["stage"], "fixed_parameters": data["fixed_parameters"],
                  "backend": data["backend"], "admission": admission,
                  "trial_and_measure_binding": "validated frozen source/units/fixed_parameters identify q_alpha*f and q_alpha*epsilon*f in L2(dr), equivalently the GC exact trials in L2(nu_alpha)",
                  "raw_projected_compression_substituted": False})
    selected = []
    complete_phase = "compression_and_determinants_enclosed_before_positivity_gates"
    for index, attempt in enumerate(attempts):
        work.check(1)
        if not isinstance(attempt, dict):
            raise Unresolved("A prior attempt row is not an object")
        configuration = attempt.get("configuration")
        if not isinstance(configuration, dict):
            raise Unresolved("A prior attempt lacks its numerical configuration")
        for name, ladder in (("cutoff", CUTOFFS), ("Legendre_order_per_panel", (16, 32, 64, 128, 256)),
                             ("precision_decimal_digits", PRECISIONS)):
            if type(configuration.get(name)) is not int or configuration[name] not in ladder:
                raise Unresolved("A prior attempt has an invalid frozen numerical configuration: " + name)
        q = configuration.get("positive_Gauss_order_per_panel")
        last_power = configuration.get("weight_series_last_power")
        if type(q) is not int or q < configuration["Legendre_order_per_panel"] or q not in QUADRATURES:
            raise Unresolved("A prior attempt has an undeclared Gauss rule")
        if type(last_power) is not int or last_power not in (8, 16, 32, 64, 128):
            raise Unresolved("A prior attempt has an undeclared angular-series order")
        diagnostic = attempt.get("trial_compression")
        phase = diagnostic.get("phase") if isinstance(diagnostic, dict) else None
        if phase != complete_phase:
            if attempt.get("ground_point_certified") is True or attempt.get("ground_preparation_certified") is True:
                raise Unresolved("A prior row claims ground completion but lacks the full compression phase")
            saved["skipped_incomplete_phase_rows"].append({
                "attempt_index": index, "configuration": configuration, "phase": phase,
                "reason": attempt.get("unresolved_item") or "Full widened Gram/compression stage was not completed",
                "incomplete_row_used_as_matrix_enclosure": False})
            continue
        entry = {"attempt_index": index, "configuration": configuration, "phase": phase,
                 "status": "validating_completed_matrix_stage", "Gram": None, "compression": None,
                 "supplied_full_Gram_exact_endpoint_record": diagnostic.get("Gram"),
                 "supplied_full_compression_exact_endpoint_record": diagnostic.get("compression"),
                 "mu2": None, "gamma": None, "ground_certificate_accepted": False,
                 "raw_determinant_certified_positive": diagnostic.get("raw_determinant_certified_positive"),
                 "full_determinant_certified_positive": diagnostic.get("full_determinant_certified_positive")}
        saved["completed_phase_rows"].append(entry)
        try:
            gram = decode_matrix(b, work, diagnostic.get("Gram"), "Prior full Gram")
            matrix = decode_matrix(b, work, diagnostic.get("compression"), "Prior FULL widened compression")
            allocations = {}
            allocations["compression_absolute_error_bounds"] = decode_matrix(
                b, work, diagnostic.get("compression_absolute_error_bounds"), "Prior compression absolute errors", True)
            contribution_data = diagnostic.get("per_entry_analytic_error_contributions")
            if not isinstance(contribution_data, dict):
                raise Unresolved("Completed prior compression lacks per-entry analytical errors")
            for name in ("operator_tail_projection", "operator_quadrature", "trial_projection_and_exterior"):
                allocations[name] = decode_matrix(b, work, contribution_data.get(name), "Prior " + name, True)
            gram_tail = decode_matrix(b, work, diagnostic.get("Gram_exterior_tail"), "Prior full Gram exterior intervals")
            entry["Gram"] = [[b.encode(value) for value in row] for row in gram]
            entry["compression"] = [[b.encode(value) for value in row] for row in matrix]
            entry["Gram_exterior_tail"] = [[b.encode(value) for value in row] for row in gram_tail]
            entry["analytical_error_records"] = {
                name: [[b.encode(value) for value in row] for row in allocation] for name, allocation in allocations.items()}
            for name in ("operator_tail_projection_error_upper", "operator_quadrature_error_upper"):
                work.check(1)
                bound = b.decode(diagnostic.get(name))
                if bound.lower() < 0:
                    raise Unresolved("A prior operator absolute-error upper bound is negative")
                entry["analytical_error_records"][name] = b.encode(bound)
            quantities = {"G" + str(i) + str(j): gram[i][j] for i in range(2) for j in range(2)}
            quantities.update({"K" + str(i) + str(j): matrix[i][j] for i in range(2) for j in range(2)})
            for name in ("mu2", "gamma"):
                if attempt.get(name) is not None:
                    if attempt.get("same_exact_coupling_rule_across_all_refinements") is not True:
                        raise Unresolved("A prior scalar is not labelled as the same exact trial coupling")
                    work.check(1)
                    scalar = b.positive(b.decode(attempt[name]), "Prior exact-trial " + name)
                    quantities[name] = scalar
                    entry[name] = b.encode(scalar)
            entry["status"] = "full_widened_matrix_stage_accepted_for_comparison_only"
            selected.append((index, quantities))
        except Unresolved as error:
            entry.update({"status": "invalid_completed_record", "unresolved_item": str(error)})
            raise
    if not selected:
        raise Unresolved("The declared prior artifact supplies no usable completed full-compression record")
    completed_orders = {row["configuration"]["Legendre_order_per_panel"] for row in saved["completed_phase_rows"]}
    skipped_orders = {row["configuration"]["Legendre_order_per_panel"] for row in saved["skipped_incomplete_phase_rows"]}
    if not {16, 32, 64} <= completed_orders or 128 not in skipped_orders:
        raise Unresolved("The first declared retry3 input must retain all completed N16/N32/N64 rows and the incomplete N128 row")
    saved.update({"status": "all_declared_completed_prior_rows_validated", "usable_completed_matrix_rows": len(selected),
                  "required_retry3_completed_rows_and_incomplete_row_present": True})
    return selected


def intersect_pair(backend, left, right, label, saved):
    b = backend
    lo, hi = max(left.lower(), right.lower()), min(left.upper(), right.upper())
    overlap = lo <= hi
    saved.update({"quantity": label, "left_full_enclosure": b.encode(left), "right_full_enclosure": b.encode(right),
                  "overlap": overlap, "intersection": b.encode(b.hull(lo, hi)) if overlap else None})
    return b.hull(lo, hi) if overlap else None


def compare_enclosures(backend, left, right, saved, required=None):
    names = sorted(required if required is not None else set(left) & set(right))
    saved.update({"status": "unresolved", "comparisons": [], "all_same_exact_quantities_overlap": False})
    output, failed = {}, []
    for name in names:
        if name not in left or name not in right:
            saved["comparisons"].append({"quantity": name, "overlap": False, "reason": "Required full enclosure is missing"})
            failed.append(name)
            continue
        record = {}
        saved["comparisons"].append(record)
        intersection = intersect_pair(backend, left[name], right[name], name, record)
        if intersection is None:
            failed.append(name)
        else:
            output[name] = intersection
    saved.update({"all_same_exact_quantities_overlap": bool(names) and not failed,
                  "status": "all_full_enclosures_overlap" if names and not failed else "unresolved"})
    if failed or not names:
        raise InconsistentEnclosures("Full corrected enclosures do not overlap: " + ", ".join(failed))
    return output


def configurations():
    # The exact nested order and baseline weight ladder are frozen in the
    # reviewed standalone-builder clarification.  There is no refitting.
    for cutoff in CUTOFFS:
        for quadrature in QUADRATURES:
            for precision_index, digits in enumerate(PRECISIONS):
                yield Configuration(cutoff, quadrature, digits, 8 * 2**precision_index)


def next_configurations(baseline):
    qi, pi = QUADRATURES.index(baseline.quadrature), PRECISIONS.index(baseline.digits)
    next_q = (None if qi + 1 == len(QUADRATURES) else
              Configuration(baseline.cutoff, QUADRATURES[qi + 1], baseline.digits, baseline.weight_order))
    next_precision = (None if pi + 1 == len(PRECISIONS) else
                      Configuration(baseline.cutoff, baseline.quadrature, PRECISIONS[pi + 1], baseline.weight_order))
    return next_q, next_precision


def serialize_reference(backend, values):
    b = backend
    return {"moments": {name: b.encode(values.get(name)) for name in ("U0", "U1", "U2", "V0")},
            "Gram": [[b.encode(values.get("G" + str(i) + str(j))) for j in range(2)] for i in range(2)],
            "compression": [[b.encode(values.get("K" + str(i) + str(j))) for j in range(2)] for i in range(2)],
            "mu2": b.encode(values.get("mu2")), "gamma": b.encode(values.get("gamma"))}


def accumulate_configuration(backend, current, common, prior, record):
    """Keep every successful exact configuration and every selected prior row."""
    b = backend
    if common is None:
        common = dict(current)
        record["all_previous_successful_configuration_intersection"] = {"status": "first_successful_configuration"}
    else:
        comparison = {}
        record["all_previous_successful_configuration_intersection"] = comparison
        common = compare_enclosures(b, common, current, comparison, required=set(common) | set(current))
    record["all_completed_prior_matrix_comparisons"] = []
    for attempt_index, prior_quantities in prior:
        comparison = {"prior_attempt_index": attempt_index}
        record["all_completed_prior_matrix_comparisons"].append(comparison)
        intersections = compare_enclosures(b, common, prior_quantities, comparison, required=set(prior_quantities))
        common.update(intersections)
    # These are literal equalities in GC4 and matrix symmetry.  Enforce
    # common intervals for their duplicate names, retaining every input.
    record["exact_identity_interval_synchronization"] = []
    for names in (("U0", "G00"), ("U1", "G01", "G10"), ("U2", "G11"), ("K01", "K10")):
        intersection = common[names[0]]
        for name in names[1:]:
            comparison = {}
            record["exact_identity_interval_synchronization"].append(comparison)
            intersection = intersect_pair(b, intersection, common[name], "exact equality " + "=".join(names), comparison)
            if intersection is None:
                raise InconsistentEnclosures("Intervals for a literal moment/matrix equality are disjoint")
        for name in names:
            common[name] = intersection
    record["common_exact_reference_after_configuration_and_all_prior_rows"] = serialize_reference(b, common)
    return common


def new_result(args, admission):
    return {"scope": "standalone direct exact-trial compression at the frozen original-H point; no model/ground/clock/COM result",
            "source": SPEC, "independent_source_review": REVIEW, "stage": "direct_trial_compression",
            "fixed_parameters": FIXED_PARAMETERS, "trial_identifier": TRIAL_IDENTIFIER,
            "measure_identifier": MEASURE_IDENTIFIER,
            "backend": "isolated python-flint0.9.0; Arb directed ball arithmetic", "admission": admission,
            "predeclared_engineering_caps": {"seconds": args.seconds, "counted_work_evaluations": args.max_evaluations,
                                             "resident_memory_MB": args.memory_mb, "maximum_threads": 4,
                                             "arithmetic_workers": 1,
                                             "quota_counter_convention": "requested batch tested against remaining quota before increment; quota-rejected batch excluded; admitted proxy charges are not elementary Arb operation counts",
                                             "work_charge_semantics": "one per exact angular summand, Gauss root, scalar node, or majorant cell; four per symmetric pair for the two diagonal/full-and-polynomial integrands"},
            "status": "unresolved", "model_evaluation_attempted": False, "model_evaluated": False,
            "ground_point_certified": False, "ground_preparation_certified": False,
            "ramsey_certified": False, "full_clock_certified": False,
            "COM_response_or_triplet_gap_certified": False, "COM_enclosure_certified": False,
            "physical_clock_load_continuation_certified": False,
            "full_fixed_charge_Fock_ground_certified": False, "open_exterior_flow_certified": False,
            "differentiable_physical_window_certified": False,
            "interacting_native_Ward_error_certified": False,
            "trapping_radius_certified": False, "temperature_certified": False,
            "Cartesian_measure_control_status": "deferred_to_ground_stage", "ground_adoption_allowed": False,
            "original_H_operator_error_or_coupling_mismatch_waived": False,
            "existing_ground_or_Ramsey_source_modified_or_imported": False,
            "mu2": None, "gamma": None, "gamma_midpoint_exact": None, "gamma_midpoint_mismatch_upper": None,
            "E_g": None, "ground_vector": None, "ground_export": None,
            "E_e": None, "Omega": None, "tau": None, "a1": None, "a2": None,
            "direct_trial_compression": {
                "method": "exact_scalar_moments_product_Gauss", "status": "unresolved",
                "trial_identifier": TRIAL_IDENTIFIER, "measure_identifier": MEASURE_IDENTIFIER,
                "prior_projected_artifact": {}, "baseline_attempts": [],
                "tightened_common_reference": None, "all_required_completion_controls_pass": False,
                "Cartesian_measure_control_status": "deferred_to_ground_stage", "ground_adoption_allowed": False},
            "unresolved_item": None}


def run(args, backend, admission):
    backend.activate(admission)
    work, result, common = Work(backend, args), new_result(args, admission), None
    stage = result["direct_trial_compression"]
    active_record = None
    try:
        prior = read_prior_ground(args.prior_ground, backend, work, stage["prior_projected_artifact"])
        for configuration in configurations():
            work.check()
            row = {"status": "unresolved", "configuration": configuration.saved(),
                   "baseline": {}, "next_quadrature_control": None, "next_precision_control": None,
                   "separate_baseline_relative_controls_both_completed": False}
            stage["baseline_attempts"].append(row)
            active_record = row["baseline"]
            try:
                baseline = evaluate_configuration(configuration, backend, work, active_record)
                common = accumulate_configuration(backend, baseline, common, prior, active_record)
                stage["tightened_common_reference"] = serialize_reference(backend, common)
                next_q, next_precision = next_configurations(configuration)
                # Both controls are separately relative to this baseline.
                # Neither is the next baseline, which may change W order.
                if next_q is None or next_precision is None:
                    row["missing_control_rung"] = {"next_quadrature_available": next_q is not None,
                                                   "next_precision_available": next_precision is not None}
                    raise Unresolved("The next scalar-order or next-precision rung is exhausted for this baseline")
                for name, next_configuration in (("next_quadrature_control", next_q),
                                                  ("next_precision_control", next_precision)):
                    control = {"baseline_configuration": configuration.saved(),
                               "only_changed_numerical_choice": "Gauss order" if name == "next_quadrature_control" else "arithmetic precision",
                               "baseline_weight_series_order_retained": True,
                               "configuration_result": {}, "status": "unresolved"}
                    row[name] = control
                    active_record = control["configuration_result"]
                    evaluated = evaluate_configuration(next_configuration, backend, work, active_record)
                    control["baseline_relative_full_enclosure_overlap"] = {}
                    compare_enclosures(backend, baseline, evaluated, control["baseline_relative_full_enclosure_overlap"],
                                       required=set(baseline) | set(evaluated))
                    common = accumulate_configuration(backend, evaluated, common, prior, active_record)
                    stage["tightened_common_reference"] = serialize_reference(backend, common)
                    control["status"] = "separate_baseline_relative_control_pass"
                row["separate_baseline_relative_controls_both_completed"] = True
                # Recompute from the intersected exact matrix box, keeping
                # all old root intervals and repeating every root gate.
                recomputation = {}
                row["directed_root_recomputation_from_intersected_G_K"] = recomputation
                refined = generalized_root(backend, common, recomputation)
                row["intersected_reference_and_recomputed_root_overlap"] = {}
                common = compare_enclosures(backend, common, refined,
                                            row["intersected_reference_and_recomputed_root_overlap"],
                                            required=set(common) | set(refined))
                stage["tightened_common_reference"] = serialize_reference(backend, common)
                midpoint = backend.positive(common["gamma"].mid(), "Deterministic dyadic gamma midpoint")
                if not midpoint.is_exact():
                    raise Unresolved("The directed backend did not retain an exact dyadic coupling midpoint")
                mismatch = backend.absolute_upper(common["gamma"] - midpoint)
                work.check()
                stage.update({"successful_baseline_configuration": configuration.saved(),
                              "same_exact_coupling_rule_across_all_refinements": True,
                              "gamma_midpoint_exact": backend.encode(midpoint),
                              "gamma_midpoint_mismatch_upper": backend.encode(mismatch),
                              "midpoint_is_a_computational_approximation_not_a_refitted_coupling": True,
                              "all_required_completion_controls_pass": True,
                              "status": "compression_enclosed"})
                result.update({"status": "compression_enclosed", "mu2": backend.encode(common["mu2"]),
                               "gamma": backend.encode(common["gamma"]),
                               "gamma_midpoint_exact": backend.encode(midpoint),
                               "gamma_midpoint_mismatch_upper": backend.encode(mismatch), "unresolved_item": None})
                row["status"] = "baseline_and_both_separate_controls_completed"
                break
            except (EngineeringCap, InconsistentEnclosures):
                raise
            except Unresolved as error:
                active_record.update({"status": "unresolved", "unresolved_item": str(error)})
                row.update({"status": "unresolved", "unresolved_item": str(error)})
                result["unresolved_item"] = str(error)
                # Keep exact partial records; proceed only on this printed
                # ladder, without hiding a disjoint-enclosure contradiction.
                continue
        else:
            result["unresolved_item"] = "The frozen cutoff/scalar-order/precision ladder exhausted before all compression controls"
    except (Unresolved, OSError, ValueError) as error:
        result.update({"status": "unresolved", "unresolved_item": type(error).__name__ + ": " + str(error)})
        if active_record is not None:
            active_record.update({"status": "unresolved", "unresolved_item": result["unresolved_item"]})
        if isinstance(error, InconsistentEnclosures):
            result["halted_for_source_or_input_review"] = True
    except Exception as error:
        result.update({"status": "unresolved", "unresolved_item": type(error).__name__ + ": " + str(error),
                       "backend_or_arithmetic_error": True})
        if active_record is not None:
            active_record.update({"status": "unresolved", "unresolved_item": result["unresolved_item"]})
    finally:
        try:
            work.check()
        except Exception as error:
            result.update({"status": "unresolved", "unresolved_item": type(error).__name__ + ": " + str(error)})
        if result["status"] != "compression_enclosed":
            stage.update({"status": "unresolved", "all_required_completion_controls_pass": False})
        if common is not None:
            stage["tightened_common_reference"] = serialize_reference(backend, common)
            result["mu2"], result["gamma"] = backend.encode(common.get("mu2")), backend.encode(common.get("gamma"))
        result["elapsed_engineering_seconds"] = time.perf_counter() - work.start
        result["counted_work_evaluations"] = work.evaluations
    return result


def write_result(path, result):
    path.write_text(json.dumps(result, indent=2) + "\n")
    stage = result["direct_trial_compression"]
    lines = [result["scope"], "status=" + result["status"],
             "counted_work_evaluations=" + str(result["counted_work_evaluations"]),
             "elapsed_engineering_seconds=" + str(result["elapsed_engineering_seconds"]),
             "baseline_attempts=" + str(len(stage["baseline_attempts"])),
             "all_required_completion_controls_pass=" + str(stage["all_required_completion_controls_pass"]),
             "Cartesian_measure_control_status=deferred_to_ground_stage; ground_adoption_allowed=false"]
    for name in ("mu2", "gamma", "gamma_midpoint_mismatch_upper"):
        value = result.get(name)
        lines.append(name + "=" + ("null" if value is None else value.get("display_ball", "exact endpoint record")))
    if result.get("unresolved_item"):
        lines.append("unresolved: " + result["unresolved_item"])
    lines.append("Ground/model/Ramsey/clock/COM/physical-window/full-Fock/flow/trapping/temperature claims: false.")
    path.with_suffix(".txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-ground", type=Path, required=True,
                        help="The reviewed immutable ground retry3 partial JSON; every completed matrix row is required")
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--actual-code-audits-released", action="store_true")
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--max-evaluations", type=int, default=2_000_000)
    parser.add_argument("--memory-mb", type=int, default=4096)
    args = parser.parse_args()
    if not args.actual_code_audits_released:
        parser.error("Author E/independent B actual-code audits and parent runtime admission are separate prerequisites")
    if not 0 < args.seconds <= 900 or not 0 < args.max_evaluations <= 2_000_000 or not 0 < args.memory_mb <= 4096:
        parser.error("The predeclared total caps must be positive and at most 900 seconds/2M work evaluations/4096MB")
    if args.output.suffix != ".json" or args.output.resolve() == args.output.with_suffix(".txt").resolve():
        parser.error("Output must use distinct .json and .txt destinations")
    if not args.output.parent.is_dir() or args.output.exists() or args.output.with_suffix(".txt").exists():
        parser.error("Both outputs must be new files in an existing parent-admitted result directory")
    if not args.prior_ground.is_file():
        parser.error("The declared immutable retry3 prior JSON is required; it cannot be silently omitted")
    if any(os.environ.get(name) != "1" for name in POOL_NAMES):
        parser.error("All six numerical worker pools must be predeclared as one")
    if os.environ.get("QI2_GPU_UUID") != args.gpu_uuid:
        parser.error("Requested GPU UUID must match the parent-selected environment")
    backend = BallBackend.load_host()  # Complete all host imports before the final free query.
    admission = remote_execution(max_threads=4)
    print("Direct trial compression admitted " + json.dumps(admission), flush=True)
    result = run(args, backend, admission)
    write_result(args.output, result)
    print(result["status"], flush=True)
    if result["status"] != "compression_enclosed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
