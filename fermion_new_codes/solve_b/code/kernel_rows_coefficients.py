#!/usr/bin/env python3
"""Bounded original-H_* radial coefficient table from K15--K20, K36--K38.

Run only after author and independent actual-code release and Ops admission.
The model here is the original sqrt(m^2+p^2) resolved-cone coefficient law.
This is not the exact finite-Brillouin-zone law, a production calibration,
a finite-coupling EOS, or an equivalence with the old lattice bond operator.
All runtime, controls and postprocessing belong on the remote qi2 host.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import time

import mpmath as mp
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad

from remote_runtime import remote_execution


F = Fraction
MASS = F(6741313, 10**8)
G_M, G_B = F(1, 5), F(1)
RANGES = (F(1), F(2))
FERMI_RADII = (F(1, 100), F(3, 100), MASS, F(1, 10), F(3, 10))
TARGET = 1e-9
NR_SUBLEADING_TARGET = 1e-5
MAIN_COLUMNS = ("k", "n", "e0", "eM", "eB", "mu0", "muM", "muB", "P0", "PM", "PB")
SOURCE_COLUMNS = ("JN0", "JN1", "Jpsi0", "Jpsi1")
COEFFICIENT_COLUMNS = MAIN_COLUMNS[1:] + SOURCE_COLUMNS
POOLS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
         "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def validate_inputs(args):
    if not (MASS > 0 and G_M == F(1, 5) and G_B == 1
            and RANGES == (F(1), F(2))
            and FERMI_RADII == (F(1, 100), F(3, 100), MASS, F(1, 10), F(3, 10))):
        raise ValueError("The frozen K36--K37 diagnostic dictionary was changed")
    for label, orders in (("radial", args.orders), ("angular", args.angular_orders)):
        if (len(orders) < 3 or any(not 8 <= order <= 256 for order in orders)
                or any(right != 2 * left for left, right in zip(orders, orders[1:]))):
            raise ValueError(f"Require at least three doubling {label} quadrature orders in8..256")
    if args.digits < 50 or args.seconds <= 0:
        raise ValueError("Require >=50decimal digits and a positive remote time cap")
    # All host-only imports and input checks precede final device admission.
    # Ops sets the pools at launch; this script does not reconfigure a shared job.
    for name in POOLS:
        if os.environ.get(name) != "1":
            raise RuntimeError(f"Ops must set {name}=1 before host imports")
    if len(os.listdir("/proc/self/task")) > 4:
        raise RuntimeError("Host imports exceeded the four-thread admission cap")


class ResourceLimit(RuntimeError):
    pass


class Work:
    def __init__(self, seconds):
        self.start = time.perf_counter()
        self.seconds = seconds
        self.rules = {}
        self.angular_evaluations = 0

    def check(self):
        if time.perf_counter() - self.start > self.seconds:
            raise ResourceLimit("Declared remote runtime cap reached; unfinished controls remain unresolved")
        if len(os.listdir("/proc/self/task")) > 4:
            raise RuntimeError("One-worker BLAS1 job exceeded its four-thread cap")

    def rule(self, order):
        if order not in self.rules:
            self.check()
            # These are ordinary numerical Gauss nodes, not certified balls.
            # Successive orders and independent controls determine agreement.
            nodes, weights = leggauss(order)
            if (np.any(~np.isfinite(nodes)) or np.any(~np.isfinite(weights))
                    or np.any(weights <= 0) or np.any(nodes <= -1) or np.any(nodes >= 1)):
                raise ArithmeticError("Invalid ordinary Gauss-Legendre rule")
            self.rules[order] = nodes, weights
        return self.rules[order]


def mp_value(value):
    return (mp.mpf(value.numerator) / value.denominator
            if isinstance(value, Fraction) else mp.mpf(value))


def free_closed(k, digits):
    """K17,K19 in high precision, avoiding small-k pressure cancellation."""
    with mp.workdps(digits):
        radius, mass = mp_value(k), mp_value(MASS)
        epsilon = mp.sqrt(mass**2 + radius**2)
        logarithm = mp.asinh(radius / mass)
        density = 8 * radius**3 / (3 * mp.pi**2)
        energy = (radius * epsilon * (2 * radius**2 + mass**2)
                  - mass**4 * logarithm) / mp.pi**2
        pressure = (radius * epsilon * (2 * radius**2 - 3 * mass**2)
                    + 3 * mass**4 * logarithm) / (3 * mp.pi**2)
        return {"k": float(radius), "n": float(density), "e0": float(energy),
                "mu0": float(epsilon), "P0": float(pressure)}


def radial_rule(k, order, work):
    nodes, weights = work.rule(order)
    points = float(k) * (nodes + 1) / 2
    measures = float(k) * weights / 2
    return points, measures


def angular_continuations(z):
    """Removable K16 limits, using convergent small-z even polynomials.

    A=sum z^(2j)/(2j+1)! and B=sum (2j+2)z^(2j+1)/(2j+3)!.
    The diagnostic has z<=0.18.  Twelve terms cover the whole input range;
    direct-u quadrature checks the closed kernel independently.  No floating
    subtraction z*cosh(z)-sinh(z) is used near the removable singularity.
    """
    if np.any(np.abs(z) > 0.5):
        raise ValueError("The bounded diagnostic small-z continuation domain was exceeded")
    square = z * z
    a, b = np.zeros_like(z), np.zeros_like(z)
    for j in range(11, -1, -1):
        a = a * square + 1.0 / math.factorial(2 * j + 1)
        b = b * square + (2 * j + 2) / math.factorial(2 * j + 3)
    return a, z * b


def epsilon_gap(p, q, mass):
    """Stable exact rationalization of eps_p*eps_q-m^2, including p=q=0."""
    p2, q2 = p * p, q * q
    product = np.sqrt(mass * mass + p2) * np.sqrt(mass * mass + q2)
    gap = (mass * mass * (p2 + q2) + p2 * q2) / (product + mass * mass)
    return product, gap


def closed_kernel(p, q, ell):
    mass = float(MASS)
    product, gap = epsilon_gap(p, q, mass)
    a, b = angular_continuations(ell * ell * p * q / 2)
    value = (np.pi**1.5 * np.exp(-ell * ell * (p * p + q * q) / 4)
             * (3 * gap / product * a - p * q / product * b))
    if np.any(~np.isfinite(value)) or np.any(value < 0):
        raise ArithmeticError("Closed current kernel lost its finite nonnegative value")
    return value


def source_columns(row):
    row["JN0"] = row["e0"]
    row["JN1"] = row["eM"] + row["eB"]
    row["Jpsi0"] = -6 * row["P0"]
    row["Jpsi1"] = -6 * (row["PM"] + row["PB"])
    return row


def radial_row(k, ell, order, digits, work):
    work.check()
    row = free_closed(k, digits)
    radius, length = float(k), float(ell)
    cutoff = 1 / length
    points, weights = radial_rule(radius, order, work)
    epsilon = np.sqrt(float(MASS)**2 + points**2)
    regulated = np.exp(-points**2 / cutoff**2)
    measure = weights * points**2 / (2 * np.pi**2)
    i0 = float(np.sum(measure * regulated))
    im = float(np.sum(measure * regulated * float(MASS) / epsilon))
    kernel = closed_kernel(points[:, None], points[None, :], length)
    energy_b = float(4 * float(G_B) * np.sum(
        measure[:, None] * measure[None, :] * regulated[:, None] * regulated[None, :] * kernel))
    kernel_end = closed_kernel(np.asarray(radius), points, length)
    boundary = float(np.sum(measure * regulated * kernel_end))
    regulator_end = math.exp(-radius**2 / cutoff**2)
    mu_m = -float(G_M) * regulator_end * (3 * im * float(MASS) / row["mu0"] - i0)
    mu_b = float(G_B) / 2 * regulator_end * boundary
    energy_m = -8 * float(G_M) * (3 * im**2 - i0**2)
    row.update({"eM": energy_m, "eB": energy_b, "muM": mu_m, "muB": mu_b,
                "PM": row["n"] * mu_m - energy_m, "PB": row["n"] * mu_b - energy_b})
    if any(not math.isfinite(row[name]) for name in MAIN_COLUMNS):
        raise ArithmeticError("Non-finite radial coefficient")
    source_columns(row)
    # Unsigned term scales expose cancellation near coefficient sign changes.
    # They supplement the strict nonzero-coefficient relative checks; they
    # never replace those checks by a looser passing criterion.
    scales = {name: abs(row[name]) for name in COEFFICIENT_COLUMNS}
    scales["eM"] = 8 * abs(float(G_M)) * (3 * im**2 + i0**2)
    scales["muM"] = abs(float(G_M)) * regulator_end * (3 * abs(im) * float(MASS) / row["mu0"] + abs(i0))
    scales["PM"] = abs(row["n"] * mu_m) + abs(energy_m)
    scales["PB"] = abs(row["n"] * mu_b) + abs(energy_b)
    scales["JN1"] = abs(energy_m) + abs(energy_b)
    scales["Jpsi1"] = 6 * (abs(row["PM"]) + abs(row["PB"]))
    return row, {"I0": i0, "Im": im, "boundary_current_integral": boundary,
                                 "coefficient_scales": scales,
                                 "free_energy_integral": float(8 / np.pi**2 * np.sum(weights * points**2 * epsilon)),
                                 "free_pressure_integral": float(8 / (3 * np.pi**2)
                                                                 * np.sum(weights * points**4 / epsilon)),
                                 "regulator_endpoint": regulator_end}


def relative_disagreement(value, reference):
    scale = max(abs(value), abs(reference))
    return abs(value - reference) / scale if scale else 0.0


def comparison(left, right, names, scales=None):
    detail = {}
    for name in names:
        difference = abs(left[name] - right[name])
        scale = (scales[name] if scales is not None
                 else max(abs(left[name]), abs(right[name])))
        detail[name] = {"absolute_disagreement": difference,
                        "relative_disagreement": relative_disagreement(left[name], right[name]),
                        "unsigned_comparison_scale": scale,
                        "scale_normalized_absolute_disagreement": difference / scale if scale else 0.0,
                        "reference_is_exact_floating_zero": right[name] == 0.0}
    return detail


def largest_relative(details):
    return max((item["relative_disagreement"] for item in details.values()), default=0.0)


def radial_refinements(k, ell, args, work):
    passes = []
    for order in args.orders:
        row, moments = radial_row(k, ell, order, args.digits, work)
        passes.append({"order": order, "row": row, "moments": moments})
    differences = [{"orders": [left["order"], right["order"]],
                    "columns": comparison(left["row"], right["row"], COEFFICIENT_COLUMNS,
                                          right["moments"]["coefficient_scales"])}
                   for left, right in zip(passes, passes[1:])]
    # Require the last two distinct refinements.  A single tiny difference
    # never by itself claims a rigorous quadrature or model-error bound.
    observed = {name: max(detail["columns"][name]["relative_disagreement"]
                          for detail in differences[-2:]) for name in COEFFICIENT_COLUMNS}
    return passes, differences, observed


def direct_angular(p, q, ell, work):
    """Distinct adaptive direct-u K38 route; does not call K16 or its series."""
    work.angular_evaluations += 1
    mass = float(MASS)
    epsilon_p, epsilon_q = math.sqrt(mass**2 + p**2), math.sqrt(mass**2 + q**2)
    product = epsilon_p * epsilon_q
    # Algebraic cancellation removal inside the unreduced trace, without
    # angular reduction, sign clipping, or a low-momentum approximation.
    difference = (mass**2 * (p**2 + q**2) + p**2 * q**2) / (product + mass**2)

    def integrand(u):
        trace = (3 * difference - p * q * u) / product
        value = math.exp(-ell**2 * (p**2 + q**2 - 2 * p * q * u) / 4) * trace
        if not math.isfinite(value) or value < 0:
            raise ArithmeticError("Unreduced angular integrand lost nonnegativity")
        return value

    scale = np.pi**1.5 / 2
    size = (3 * difference + p * q) / product
    answer = quad(integrand, -1, 1, epsabs=max(1e-300, size * 1e-14),
                  epsrel=2e-13, limit=100, full_output=1)
    value, estimated_error = answer[0], answer[1]
    return scale * value, scale * estimated_error, len(answer) == 3


def angular_row(k, ell, order, work):
    points, weights = radial_rule(k, order, work)
    length, cutoff = float(ell), 1 / float(ell)
    regulated = np.exp(-points**2 / cutoff**2)
    measure = weights * points**2 / (2 * np.pi**2)
    energy, energy_error, boundary, boundary_error = 0.0, 0.0, 0.0, 0.0
    all_successful, kernel_absolute, kernel_relative = True, 0.0, 0.0
    for j, p in enumerate(points):
        work.check()
        for i, q in enumerate(points):
            # Both ordered radial legs are retained.  No borrowed multiplicity
            # or additional valley factor accompanies K17's factor4.
            value, error, success = direct_angular(float(p), float(q), length, work)
            weight = float(4 * float(G_B) * measure[j] * measure[i] * regulated[j] * regulated[i])
            energy += weight * value
            energy_error += weight * error
            all_successful &= success
            closed = float(closed_kernel(np.asarray(p), np.asarray(q), length))
            kernel_absolute = max(kernel_absolute, abs(value - closed))
            kernel_relative = max(kernel_relative, relative_disagreement(value, closed))
        value, error, success = direct_angular(float(k), float(p), length, work)
        weight = float(measure[j] * regulated[j])
        boundary += weight * value
        boundary_error += weight * error
        all_successful &= success
    factor = float(G_B) / 2 * math.exp(-float(k)**2 / cutoff**2)
    return {"eB": energy, "muB": factor * boundary,
            "estimated_adaptive_angular_error_eB": energy_error,
            "estimated_adaptive_angular_error_muB": factor * boundary_error,
            "maximum_sample_kernel_absolute_disagreement": kernel_absolute,
            "maximum_sample_kernel_relative_disagreement": kernel_relative,
            "adaptive_angular_integrations_successful": bool(all_successful)}


def angular_controls(k, ell, final, args, work):
    passes = [{"order": order, **angular_row(k, ell, order, work)} for order in args.angular_orders]
    differences = [{"orders": [left["order"], right["order"]],
                    "columns": comparison(left, right, ("eB", "muB"))}
                   for left, right in zip(passes, passes[1:])]
    against_closed = comparison(passes[-1], final, ("eB", "muB"))
    achieved = max(largest_relative(against_closed),
                   max((largest_relative(item["columns"]) for item in differences[-2:]), default=0.0),
                   passes[-1]["maximum_sample_kernel_relative_disagreement"])
    return {"route": "Adaptive direct-u K38 then positive radial Gauss integration; independent of K16",
            "refinements": passes, "successive_differences": differences,
            "comparison_with_closed_angular_row": against_closed,
            "achieved_relative_agreement": achieved,
            "passed_1e-9_agreement": achieved <= TARGET and all(
                item["adaptive_angular_integrations_successful"] for item in passes),
            "error_status": "Adaptive error estimates and observed order differences, not interval bounds"}


def endpoint_controls(k, final, moments):
    radius, mass = float(k), float(MASS)
    derivative_density = 8 * radius**2 / np.pi**2
    # Differentiate both upper limits of K17 before dividing by n'(k).
    derivative_i0 = radius**2 / (2 * np.pi**2) * moments["regulator_endpoint"]
    derivative_im = derivative_i0 * mass / math.sqrt(mass**2 + radius**2)
    derivative_m = -16 * float(G_M) * (3 * moments["Im"] * derivative_im
                                       - moments["I0"] * derivative_i0)
    derivative_b = 8 * float(G_B) * derivative_i0 * moments["boundary_current_integral"]
    derivative_0 = 8 / np.pi**2 * radius**2 * math.sqrt(mass**2 + radius**2)
    measured = {"mu0": derivative_0 / derivative_density,
                "muM": derivative_m / derivative_density,
                "muB": derivative_b / derivative_density}
    detail = comparison(measured, final, ("mu0", "muM", "muB"))
    return {"method": "Explicit upper-endpoint derivatives with two current-energy legs and n'(k)",
            "measured_chemical_potentials": measured, "comparison": detail,
            "achieved_relative_agreement": largest_relative(detail),
            "passed_1e-9_agreement": largest_relative(detail) <= TARGET}


def density_derivative_controls(k, ell, final, args, work):
    radius, density = float(k), final["n"]
    steps = tuple(F(1, 20 * 2**j) for j in range(10))
    passes = []
    for step in steps:
        work.check()
        h_relative = float(step)
        energies = {}
        for offset in (-2, -1, 1, 2):
            # n is varied; k(n) is recomputed from n proportional to k^3.
            perturbed_k = radius * (1 + offset * h_relative)**(1 / 3)
            row, _ = radial_row(perturbed_k, ell, args.orders[-1], args.digits, work)
            energies[offset] = row
        measured = {}
        for energy, chemical in (("e0", "mu0"), ("eM", "muM"), ("eB", "muB")):
            measured[chemical] = (-energies[2][energy] + 8 * energies[1][energy]
                                  - 8 * energies[-1][energy] + energies[-2][energy]) / (
                                      12 * density * h_relative)
        details = comparison(measured, final, ("mu0", "muM", "muB"))
        passes.append({"relative_density_step": str(step), "values": measured,
                       "comparison_with_endpoint": details})
    successive = [{"steps": [left["relative_density_step"], right["relative_density_step"]],
                   "columns": comparison(left["values"], right["values"], ("mu0", "muM", "muB"))}
                  for left, right in zip(passes, passes[1:])]
    # Selection requires adjacent FD convergence and independent endpoint
    # agreement together.  It cannot pass by picking one coincident step.
    candidates = []
    for j in range(2, len(passes)):
        achieved = max(largest_relative(passes[j]["comparison_with_endpoint"]),
                       largest_relative(passes[j - 1]["comparison_with_endpoint"]),
                       largest_relative(successive[j - 1]["columns"]),
                       largest_relative(successive[j - 2]["columns"]))
        candidates.append((achieved, j))
    achieved, selected = min(candidates)
    return {"method": "Five-point density central difference, O(h^4), recomputing all K17 energies",
            "refinements": passes, "successive_differences": successive,
            "selected_converged_step_index": selected,
            "achieved_relative_agreement": achieved,
            "passed_1e-9_agreement": achieved <= TARGET,
            "error_status": "FD convergence and endpoint agreement; no interval truncation enclosure"}


def free_controls(k, final, moments, digits):
    higher = free_closed(k, digits + 30)
    numerical = {"e0": moments["free_energy_integral"], "P0": moments["free_pressure_integral"]}
    details = comparison(numerical, final, ("e0", "P0"))
    precision = comparison(higher, final, ("n", "e0", "mu0", "P0"))
    achieved = max(largest_relative(details), largest_relative(precision))
    return {"method": "Original closed formulas in high precision vs independent positive energy/pressure integrals",
            "positive_integrals": numerical, "comparison": details,
            "closed_formula_precision_digits": [digits, digits + 30],
            "precision_comparison_after_binary64_conversion": precision,
            "achieved_relative_agreement": achieved,
            "passed_1e-9_agreement": achieved <= TARGET}


def nr_controls(ell, args, work):
    passes = []
    for inverse_ratio in (10, 100, 1000, 10000):
        work.check()
        radius = MASS / inverse_ratio
        row, _ = radial_row(radius, ell, args.orders[-1], args.digits, work)
        before, _ = radial_row(radius, ell, args.orders[-2], args.digits, work)
        k, n, mass, cutoff = float(radius), row["n"], float(MASS), 1 / float(ell)
        leading_m = -float(G_M) * n**2 / 16
        next_m = leading_m * (1 - 6 * k**2 / (5 * cutoff**2) - 9 * k**2 / (10 * mass**2))
        leading_b = 9 * np.pi**1.5 * float(G_B) * n**2 * k**2 / (320 * mass**2)
        ratios = {"eM_over_leading_K20": row["eM"] / leading_m,
                  "eB_over_leading_K20": row["eB"] / leading_b,
                  "PM_over_eM": row["PM"] / row["eM"],
                  "PB_over_eB": row["PB"] / row["eB"]}
        deviations = {"eM_leading": abs(ratios["eM_over_leading_K20"] - 1),
                      "eB_leading": abs(ratios["eB_over_leading_K20"] - 1),
                      "PM_over_eM": abs(ratios["PM_over_eM"] - 1),
                      "PB_over_eB": abs(ratios["PB_over_eB"] - 5 / 3)}
        contact_coefficient = (1 - ratios["eM_over_leading_K20"]) / k**2
        expected_contact_coefficient = 6 / (5 * cutoff**2) + 9 / (10 * mass**2)
        before_contact_ratio = before["eM"] / leading_m
        coefficient_refinement = abs(ratios["eM_over_leading_K20"] - before_contact_ratio) / k**2
        # Subtracting a ratio near one magnifies floating errors by1/k^2.
        # This indicator is diagnostic, not a rigorous arithmetic enclosure.
        subtraction_indicator = (8 * np.finfo(float).eps
                                 * max(1.0, abs(ratios["eM_over_leading_K20"])) / k**2)
        relative_resolution = max(coefficient_refinement, subtraction_indicator) / expected_contact_coefficient
        relative_coefficient_deviation = abs(contact_coefficient - expected_contact_coefficient) / expected_contact_coefficient
        resolved_coefficient = (abs(1 - ratios["eM_over_leading_K20"])
                                > 10 * (abs(ratios["eM_over_leading_K20"] - before_contact_ratio)
                                        + 8 * np.finfo(float).eps)
                                and relative_resolution <= NR_SUBLEADING_TARGET / 4)
        passes.append({"k_over_m": str(F(1, inverse_ratio)), "k": str(radius),
                       "row": row, "leading_energy_M": leading_m, "next_order_energy_M": next_m,
                       "leading_energy_B": leading_b, "ratios": ratios, "limit_deviations": deviations,
                       "relative_contact_next_order_remainder": abs(row["eM"] - next_m) / abs(leading_m),
                       "contact_subleading": {
                           "c_M": contact_coefficient,
                           "K20_expected_c_M": expected_contact_coefficient,
                           "finite_k_absolute_coefficient_deviation": abs(contact_coefficient - expected_contact_coefficient),
                           "finite_k_relative_coefficient_deviation": relative_coefficient_deviation,
                           "successive_quadrature_absolute_coefficient_disagreement": coefficient_refinement,
                           "subtraction_roundoff_magnification_indicator": subtraction_indicator,
                           "relative_resolution_indicator": relative_resolution,
                           "numerically_resolved": bool(resolved_coefficient),
                           "resolution_indicator_is_not_an_interval_bound": True}})
    keys = tuple(passes[-1]["limit_deviations"])
    # These are asymptotic limits, not finite-k identities.  State the final
    # deviations and trend separately from the table's 1e-9 numerical checks.
    trend = all(passes[j + 1]["limit_deviations"][name] < passes[j]["limit_deviations"][name]
                for name in keys for j in range(len(passes) - 1))
    small_final = all(passes[-1]["limit_deviations"][name] < 1e-6 for name in keys)
    candidates = []
    for j in range(1, len(passes)):
        left, right = passes[j - 1]["contact_subleading"], passes[j]["contact_subleading"]
        expected = right["K20_expected_c_M"]
        achieved = max(left["finite_k_relative_coefficient_deviation"],
                       right["finite_k_relative_coefficient_deviation"],
                       left["relative_resolution_indicator"], right["relative_resolution_indicator"],
                       abs(left["c_M"] - right["c_M"]) / expected)
        approach = right["finite_k_relative_coefficient_deviation"] < (
            passes[0]["contact_subleading"]["finite_k_relative_coefficient_deviation"] / 10)
        candidates.append({"sequence_indices": [j - 1, j], "achieved_relative_agreement": achieved,
                           "resolved_pair": left["numerically_resolved"] and right["numerically_resolved"],
                           "approach_from_initial_point": bool(approach)})
    admissible = [item for item in candidates if item["resolved_pair"] and item["approach_from_initial_point"]]
    selected = min(admissible, key=lambda item: item["achieved_relative_agreement"]) if admissible else None
    subleading_pass = selected is not None and selected["achieved_relative_agreement"] <= NR_SUBLEADING_TARGET
    return {"scope": "K20 low-density normalization and pressure limits; no replacement of full table coefficients",
            "sequence": passes, "decreasing_limit_deviations": trend,
            "final_absolute_limit_target": 1e-6,
            "contact_subleading_control": {
                "definition": "c_M(k)=(1-eM/eM_lead)/k^2; K20 gives6/(5Lambda^2)+9/(10m^2)",
                "declared_asymptotic_relative_tolerance": NR_SUBLEADING_TARGET,
                "adjacent_sequence_candidates": candidates, "selected_resolved_pair": selected,
                "passed": bool(subleading_pass),
                "status": "resolved_asymptotic_agreement" if subleading_pass else "unresolved_asymptotics_or_roundoff",
                "smallest_point_monotonicity_not_required": True},
            "passed_asymptotic_control": bool(trend and small_final and subleading_pass),
            "caveat": "Finite-k departures are physical asymptotic remainders, not quadrature errors"}


def evaluate_point(k, ell, args, work):
    passes, differences, observed = radial_refinements(k, ell, args, work)
    final = passes[-1]["row"]
    free = free_controls(k, final, passes[-1]["moments"], args.digits)
    endpoint = endpoint_controls(k, final, passes[-1]["moments"])
    angular = angular_controls(k, ell, final, args, work)
    fd = density_derivative_controls(k, ell, final, args, work)
    observed["e0"] = max(observed["e0"], free["comparison"]["e0"]["relative_disagreement"])
    observed["P0"] = max(observed["P0"], free["comparison"]["P0"]["relative_disagreement"])
    observed["eB"] = max(observed["eB"], angular["comparison_with_closed_angular_row"]["eB"]["relative_disagreement"])
    observed["muB"] = max(observed["muB"], angular["comparison_with_closed_angular_row"]["muB"]["relative_disagreement"])
    achieved = max(max(observed.values()), free["achieved_relative_agreement"],
                   endpoint["achieved_relative_agreement"], angular["achieved_relative_agreement"],
                   fd["achieved_relative_agreement"])
    passed = (achieved <= TARGET and free["passed_1e-9_agreement"]
              and endpoint["passed_1e-9_agreement"] and angular["passed_1e-9_agreement"]
              and fd["passed_1e-9_agreement"])
    return {"exact_k": str(k), "exact_ell": str(ell), "exact_Lambda": str(1 / ell),
            "row": final, "radial_refinements": passes, "successive_radial_differences": differences,
            "observed_relative_disagreement_by_reported_column": observed,
            "free_controls": free, "endpoint_controls": endpoint,
            "direct_angular_controls": angular, "density_FD_controls": fd,
            "achieved_maximum_relative_agreement": achieved,
            "numerical_control_status": "passed_observed_agreement" if passed else "unresolved_tolerance",
            "adoption_status": "Saved-output author and independent review required"}


def render_text(result):
    maximum = max((item["achieved_maximum_relative_agreement"]
                   for block in result["range_blocks"] for item in block["points"]), default=None)
    lines = [
        "Original-H_* bounded resolved-cone radial coefficient diagnostic (K15--K20,K36--K38).",
        "Exact inputs: a=t0=1; m=0.06741313; gM=.2; gB=1; ell=1,2; Lambda=1/ell.",
        "k=.01,.03,.06741313,.1,.3; proper n=8k^3/(3pi^2); all rows at N=psi=1.",
        "Named coupling-label dictionary gA/(t0 a^3)=UA/t0; Gaussian is unnormalized.",
        "No old-bond equivalence or imported old saddle/HF mass.",
        "lambda=1 evaluates the first coefficient along the stated direction; finite-g EOS is not certified.",
        "The table uses sqrt(m^2+p^2), the full seed regulator and Gaussian range.",
        "Finite-BZ replacement error, spatial TF remainder, exact-ground o(lambda), and dressing norm window are unevaluated.",
        "Achieved errors below are independent-route/order/FD agreement, not interval enclosures or model-error bounds.",
        f"Requested relative agreement={TARGET:.1e}; achieved maximum={maximum:.9g}." if maximum is not None
        else "No completed numerical row; achieved tolerance unresolved.",
        "Saved-output author and independent review are required before [N] adoption.", ""]
    for block in result["range_blocks"]:
        lines += [f"ell/a={block['exact_ell']} aLambda={block['exact_Lambda']}",
                  "ka na^3 e0a^4 eMa^4 eBa^4 mu0a muMa muBa P0a^4 PMa^4 PBa^4"]
        for item in block["points"]:
            lines.append(" ".join(f"{item['row'][name]:.16e}" for name in MAIN_COLUMNS))
        lines += ["", "Companion radial sources (not finite-b strain):", "ka JN0 JN1 Jpsi0 Jpsi1"]
        for item in block["points"]:
            lines.append(" ".join(f"{item['row'][name]:.16e}" for name in ("k",) + SOURCE_COLUMNS))
        lines += ["", "ka achieved_relative_agreement status"]
        for item in block["points"]:
            lines.append(f"{float(F(item['exact_k'])):.8g} {item['achieved_maximum_relative_agreement']:.9g} "
                         + item["numerical_control_status"])
        if "NR_controls" in block:
            nr = block["NR_controls"]
            lines.append(f"K20 asymptotic trend/control={nr['passed_asymptotic_control']}; final limit deviations="
                         + json.dumps(nr["sequence"][-1]["limit_deviations"], sort_keys=True))
            lines.append("K20 subleading contact coefficient: "
                         + nr["contact_subleading_control"]["status"]
                         + f"; declared asymptotic relative tolerance={NR_SUBLEADING_TARGET:.1e}.")
        else:
            lines.append("K20 asymptotic controls unexecuted or unfinished.")
        lines.append("")
    lines += ["Overall numerical control status: " + result["numerical_control_status"],
              "Unresolved/unfinished work: " + result.get("termination", "none within the requested numerical controls")]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--orders", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--angular-orders", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--digits", type=int, default=70)
    parser.add_argument("--seconds", type=int, default=1800)
    args = parser.parse_args()
    validate_inputs(args)
    # Fresh same-process free-device query, visible context, and sole-own PID
    # guard occur only here, after all host imports and validated inputs.
    execution = remote_execution(max_threads=4)
    np.seterr(over="raise", invalid="raise", divide="raise", under="ignore")
    work = Work(args.seconds)
    result = {
        "source": "solve/strand_A_kernel_rows.tex K15--K20,K36--K38; solve/review_A_kernel_rows.tex",
        "scope": "Bounded original-law resolved-cone first coefficient diagnostic only",
        "parameters": {"a": "1", "t0": "1", "m": str(MASS), "gM": str(G_M), "gB": str(G_B),
                       "ell": [str(x) for x in RANGES], "Lambda": [str(1 / x) for x in RANGES],
                       "k": [str(x) for x in FERMI_RADII], "valleys": 8, "positive_spins_per_cone": 2,
                       "density": "n=8k^3/(3pi^2), proper density",
                       "named_dictionary": "gA/(t0a^3)=UA/t0, a=t0=1; no Gaussian/old-bond equivalence"},
        "execution": {**execution, "workers": 1, "BLAS_pools": {name: os.environ[name] for name in POOLS}},
        "numerics": {"radial_orders": args.orders, "direct_angular_radial_orders": args.angular_orders,
                     "free_closed_decimal_precision": args.digits, "relative_agreement_target": TARGET,
                     "ordinary_Gauss_nodes_not_interval_certified": True,
                     "arithmetic_and_quadrature_agreement_not_interval_bounds": True},
        "range_blocks": [], "all_requested_rows_attempted": False,
        "numerical_control_status": "unfinished", "adoption_status": "Author and independent saved-output review pending",
        "unevaluated_model_errors": ["Exact finite-BZ/filter/alias replacement error",
                                    "Spatial TF/inhomogeneous state-response remainder",
                                    "Exact-ground o(lambda) and finite-coupling EOS",
                                    "Sufficient full-seed dressing norm window",
                                    "Production matter inputs and physical/star window"],
    }

    def save():
        result["elapsed_seconds"] = time.perf_counter() - work.start
        result["direct_angular_integrations"] = work.angular_evaluations
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "kernel_rows_coefficients.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n")
        (args.output_dir / "kernel_rows_coefficients.txt").write_text(render_text(result))

    try:
        for ell in RANGES:
            block = {"exact_ell": str(ell), "exact_Lambda": str(1 / ell), "points": []}
            result["range_blocks"].append(block)
            for k in FERMI_RADII:
                work.check()
                print(f"Original cone ell={ell} k={k}: radial and independent angular/source controls", flush=True)
                point = evaluate_point(k, ell, args, work)
                block["points"].append(point)
                save()
                print(f"ell={ell} k={k}: {point['numerical_control_status']}; "
                      f"achieved={point['achieved_maximum_relative_agreement']:.6g}", flush=True)
            block["NR_controls"] = nr_controls(ell, args, work)
            save()
        result["all_requested_rows_attempted"] = True
        passed = all(point["numerical_control_status"] == "passed_observed_agreement"
                     for block in result["range_blocks"] for point in block["points"])
        passed &= all(block["NR_controls"]["passed_asymptotic_control"] for block in result["range_blocks"])
        result["numerical_control_status"] = "passed_observed_agreement" if passed else "unresolved_tolerance_or_control"
    except ResourceLimit as error:
        result["termination"] = str(error)
    except (ArithmeticError, FloatingPointError) as error:
        result["termination"] = "Unresolved arithmetic/control: " + str(error)
    save()
    print(f"Saved guarded coefficient output under {args.output_dir}; no [N] claim before independent result review", flush=True)


if __name__ == "__main__":
    main()
