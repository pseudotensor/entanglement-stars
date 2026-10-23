#!/usr/bin/env python3
"""TG1--TG21 ground-only proof for the SAME frozen original prepared pair.

This new standalone branch never builds a dense interaction or completed-H
exponential.  Its rectangular Cauchy factor is an action representation of
the full same-quadrature matrix with an explicit tail, not a changed law.
Author E and independent B actual-code audits precede separate parent
admission/execution.  No existing N/Ramsey consumer adapter is provided.
"""

from __future__ import annotations

import argparse
from fractions import Fraction as F
import json
import os
from pathlib import Path
import time

import e_bound_clock as base
import e_bound_clock_ground_input as input_api
import e_bound_clock_cartesian as cartesian_api
from remote_runtime import remote_execution


SPEC = "solve/strand_E_ground_top.tex"
REVIEW = "solve/review_E_ground_top.tex"
REPRESENTATION = "normalized_piecewise_Legendre_dyadic_seed"
BASE_POWERS = 32
POOL_NAMES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")


def exact_fraction(point):
    if not point.is_finite() or not point.is_exact():
        raise base.Unresolved("A required seed or endpoint is not a finite exact dyadic")
    mantissa, exponent = point.man_exp()
    mantissa, exponent = int(mantissa), int(exponent)
    return F(mantissa * 2**exponent) if exponent >= 0 else F(mantissa, 2**(-exponent))


def dyadic_record(value):
    """Exact Fraction known to have a power-of-two denominator."""
    denominator = value.denominator
    if denominator & (denominator - 1):
        raise base.Unresolved("An exact seed datum is not dyadic")
    return {"mantissa": str(value.numerator), "exponent2": -(denominator.bit_length() - 1)}


def exact_ball(value):
    item = dyadic_record(value)
    point = base.arb((int(item["mantissa"]), item["exponent2"]))
    if not point.is_exact():
        raise base.Unresolved("Pinned Arb did not retain the exact dyadic seed datum")
    return point


def endpoint_interval(lower, upper):
    """Avoid a wide Arb hull which could lose the positive TG7 lower bound."""
    lo, hi = exact_fraction(lower), exact_fraction(upper)
    if not 0 < lo < hi:
        raise base.Unresolved("The theorem energy endpoints are not strictly positive and ordered")
    return {"lower_dyadic": dyadic_record(lo), "upper_dyadic": dyadic_record(hi)}, (lo, hi)


class Work(base.Work):
    """Unchanged requested counter, plus honest macro/guard progress."""

    def __init__(self, args):
        super().__init__(args)
        self.operations, self.guard_segments, self.memory_reservations = [], [], []
        self.context = "retained scalar/rule/input work"
        self.live_entries, self.live_precisions, self.next_token = {}, {}, 0

    def check(self, increment=0):
        before = self.evaluations
        admitted, reason = False, None
        try:
            super().check(increment)  # Includes a rejecting requested increment, unchanged.
            admitted = True
        except BaseException as error:
            reason = type(error).__name__ + ": " + str(error)
            raise
        finally:
            if increment:
                segment = {"context": self.context, "requested_units_each": increment,
                           "preceding_counter_first": before, "counter_after_last": self.evaluations,
                           "guard_admitted": admitted, "guard_completed": admitted,
                           "scientific_operation_completion_asserted_by_guard": False,
                           "requests": 1, "rejection_reason": reason}
                previous = self.guard_segments[-1] if self.guard_segments else None
                if (previous is not None and admitted and previous["guard_admitted"]
                        and previous["context"] == segment["context"]
                        and previous["requested_units_each"] == increment
                        and previous["counter_after_last"] == before):
                    previous["requests"] += 1
                    previous["counter_after_last"] = self.evaluations
                else:
                    self.guard_segments.append(segment)

    def operation(self, label, charge, function):
        row = {"operation": label, "proposed_charge": charge, "preceding_counter": self.evaluations,
               "guard_admitted": False, "backend_result_returned": False, "completed_under_caps": False}
        self.operations.append(row)
        print("TG operation requested " + str(charge) + " at " + str(self.evaluations) + ": " + label, flush=True)
        old_context, self.context = self.context, label
        try:
            self.check(charge)
            row["guard_admitted"] = True
            value = function()
            row["backend_result_returned"] = True
            self.check()
            row["completed_under_caps"] = True
            print("TG operation completed at " + str(self.evaluations) + ": " + label, flush=True)
            return value
        except BaseException as error:
            row["failure"] = type(error).__name__ + ": " + str(error)
            raise
        finally:
            row["counter_after"] = self.evaluations
            self.context = old_context

    def reserve(self, entries, label):
        """Conservative existing estimate for ALL tracked simultaneous arrays."""
        token, self.next_token = self.next_token, self.next_token + 1
        total = sum(self.live_entries.values()) + entries
        row = {"token": token, "label": label, "proposed_entries": entries,
               "proposed_simultaneous_entries": total, "binary_precision_at_reservation": int(base.ctx.prec),
               "guard_admitted": False, "currently_reserved": False}
        self.memory_reservations.append(row)
        try:
            precision = self.reserve_matrix_entries(total, label)
            self.live_entries[token] = entries
            self.live_precisions[token] = int(base.ctx.prec)
            row.update({"guard_admitted": True, "currently_reserved": True,
                        "estimate_binary_precision_for_all_live_entries": precision,
                        "all_live_entries_priced_at_least_at_creation_precision": True})
            return token
        except BaseException as error:
            row["failure"] = type(error).__name__ + ": " + str(error)
            raise

    def release(self, token):
        self.live_entries.pop(token, None)
        self.live_precisions.pop(token, None)
        for row in reversed(self.memory_reservations):
            if row["token"] == token:
                row["currently_reserved"] = False
                break

    def reserve_matrix_entries(self, entries, label):
        self.check()  # Retains the actual RSS/wall/thread checks.
        precision = max([int(base.ctx.prec)] + list(self.live_precisions.values()))
        # The unchanged base estimate, conservatively priced at every active
        # reservation's creation precision even after a return to baseline.
        estimate = entries * (192 + 2 * ((precision + 7) // 8))
        if estimate > self.args.memory_mb * 1024**2:
            raise base.EngineeringCap(label + " exceeds the predeclared matrix-memory estimate cap")
        return precision

    def rule(self, order):
        key = base.ctx.prec, order
        if key in self.rules:
            return super().rule(order)
        token = self.reserve(2 * order, "TG retained exact-rule node/weight enclosures")
        try:
            return super().rule(order)
        except BaseException:
            self.release(token)
            raise


def norm_vector(vector):
    squared = sum((base.abs_upper(vector[i, 0])**2 for i in range(vector.nrows())), base.a(0))
    return base.upper(squared.sqrt())


def multiply(work, left, right, charge, label):
    return work.operation(label, charge, lambda: left * right)


def theorem_bounds(direct_input, gamma, work, record):
    b = direct_input.b
    reference = direct_input.data["direct_trial_compression"]["tightened_common_reference"]
    gram = input_api.direct.decode_matrix(b, work, reference["Gram"], "TG imported full Gram")
    kernel = input_api.direct.decode_matrix(b, work, reference["compression"], "TG imported full compression")
    emax = 2 * (1 + 3 / base.a(base.ALPHA)**2).sqrt()
    gates = {"K00_above_1_over62500": kernel[0][0].lower() > base.a(F(1, 62500)).upper(),
             "G00_below4_over625": gram[0][0].upper() < base.a(F(4, 625)).lower(),
             "gamma_above350000": gamma.lower() > 350000,
             "gamma_below400000": gamma.upper() < 400000, "Emax_below4000000": emax.upper() < 4_000_000}
    record.update({"consuming_exact_endpoint_gates": gates, "G00": base.encoded(gram[0][0]),
                   "K00": base.encoded(kernel[0][0]), "gamma": base.encoded(gamma), "Emax": base.encoded(emax)})
    if not all(gates.values()):
        raise base.Unresolved("The full TG interaction/ground theorem endpoint prerequisites are unresolved")
    lo = base.lower(2 * base.a(-2560).exp())
    hi = base.upper(4_000_000 * base.a(-1750).exp())
    energy, endpoints = endpoint_interval(lo, hi)
    other = base.lower(2 * base.a(-560).exp())
    thresholds = [base.a(-value).exp() for value in (2560, 1000, 800)]
    scalar_gates = {"count0_lower_threshold_below_ground_lower": thresholds[0].upper() < lo,
                    "count1_middle_threshold_above_ground_upper": hi < thresholds[1].lower(),
                    "count1_middle_threshold_below_other_spectrum": thresholds[1].upper() < other,
                    "count1_upper_threshold_above_ground_upper": hi < thresholds[2].lower(),
                    "count1_upper_threshold_below_other_spectrum": thresholds[2].upper() < other,
                    "energy_width_below1e_minus700": base.a(endpoints[1] - endpoints[0]).upper() < base.a(F(1, 10**700)).lower(),
                    "unchanged_energy_width_target": base.a(endpoints[1] - endpoints[0]).upper() <= base.a(F(1, 10**12)).lower()}
    separation = base.positive(other - hi, "TG strictly positive intrinsic original-H separation")
    epsilon_h = base.upper(2000 * base.a(-630).exp())
    record.update({"energy_interval_exact_endpoints_TG7": energy,
                   "scalar_theorem_count_and_width_gates": scalar_gates,
                   "full_H_variational_counts": [{"threshold": base.encoded(value), "count": count}
                                                 for value, count in zip(thresholds, (0, 1, 1))],
                   "count_method": "TG full-operator min-max theorem, not numerical BS/LDL",
                   "other_spectrum_lower": base.encoded(other), "intrinsic_H_separation_lower": base.encoded(base.lower(separation)),
                   "H_ground_to_top_interaction_vector_error_upper": base.encoded(epsilon_h),
                   "interaction_top_lower_exact": "1/400", "interaction_complement_upper_exact": "7/10000",
                   "interaction_gap_lower_exact": "9/5000", "intrinsic_f_phase_square_margin_exact": "18/25"})
    if not all(scalar_gates.values()):
        raise base.Unresolved("A directed TG energy/count/width theorem gate is unresolved")
    return {"E_g": energy, "energy_endpoints": endpoints, "Emax": emax, "epsilon_H": epsilon_h,
            "G00": gram[0][0], "separation": base.lower(separation)}


def factor_signature(d):
    return (d.cutoff, d.order, d.quadrature, int(base.ctx.dps), tuple(d.coefficients))


def build_factor(d, powers, work, record, reuse=None):
    """Literal TG16, interleaved f/j columns, and full same-rule TG17 tail."""
    n, m, r = d.rank, len(d.node_data), 2 * powers
    signature = factor_signature(d)
    start_power = 0
    if reuse is not None:
        if reuse["signature"] != signature or reuse["powers"] > powers:
            raise base.Unresolved("A Cauchy prefix was offered from a different rule/basis/precision/weight definition")
        start_power = reuse["powers"]
    added = 2 * (powers - start_power)
    record.update({"powers": powers, "feature_columns": r, "polynomial_rank": n, "node_count": m,
                   "column_order": "column2*ell=f, column2*ell+1=j",
                   "prefix_columns_reused": 2 * start_power, "status": "unresolved",
                   "whole_ball_factors_not_arbitrary_independent_entry_choices": True,
                   "full_rank_M_replaced_by_low_rank_Hamiltonian": False})
    factor_token = work.reserve(n * r, "TG retained full rectangular factor")
    scratch_token = work.reserve(n * m + m * added + n * added + n * r + n + r + m * (d.order + 8),
                                 "TG simultaneous embedding/node-feature/new-factor/node/basis arrays")
    try:
        node_info = []
        if reuse is not None:
            node_info = reuse["node_info"]
        else:
            for p, radius, weight, values, absorption in d.node_data:
                work.check()
                for name, value in (("radius", radius), ("Gauss weight", weight), ("q_alpha", absorption)):
                    base.positive(value, "TG actual positive " + name)
                epsilon = base.positive(1 + radius**2, "TG real epsilon root argument").sqrt()
                f = (-radius**2).exp() * (epsilon + 1) / (2 * epsilon)
                j = (-radius**2).exp() * (epsilon - 1) / (2 * epsilon)
                base.positive(f, "TG positive upper-spinor leg")
                z = (epsilon - 4) / (epsilon + 4)
                node_info.append({"panel": p, "radius": radius, "weight": weight, "basis": values,
                                  "q": absorption, "epsilon": epsilon, "f": f, "j": j, "z": z})
            ordered = sorted(node_info, key=lambda node: exact_fraction(node["radius"].lower()))
            if any(left["radius"].upper() >= right["radius"].lower()
                   or left["epsilon"].upper() >= right["epsilon"].lower()
                   for left, right in zip(ordered, ordered[1:])):
                raise base.Unresolved("Actual positive Gauss radii/epsilon enclosures do not certify distinct nodes")
        rho_z = max((base.abs_upper(node["z"]) for node in node_info), default=base.a(0))
        if not rho_z < 1 or (d.cutoff == 16 and not rho_z < base.a(F(5, 8)).lower()):
            raise base.Unresolved("The actual Cauchy geometric ratio lacks its strict bound")
        if d.quadrature < d.order or n != len(d.panels) * d.order or m != len(d.panels) * d.quadrature:
            raise base.Unresolved("Actual Gauss/basis dimensions do not support the full-row-rank proof")
        record["full_M_positivity_proof"] = {
            "method": "CE1_CE3 full Cauchy/Laplace positive nodal kernel + exact positive-rule full-row-rank embedding",
            "positive_radii_weights_absorption_upper_legs_checked": True,
            "distinct_radii_and_epsilon_checked": True, "Q_at_least_N": True,
            "all_polynomial_directions_retained": True, "symmetric_exact_reference_construction": True,
            "numeric_full_M_LDL_executed": False,
            "Gauss_basis_square_degree": 2 * d.order - 2, "Gauss_exactness_degree": 2 * d.quadrature - 1}
        embedding = base.arb_mat(n, m)
        for column, node in enumerate(node_info):
            work.check()
            for degree, value in enumerate(node["basis"]):
                embedding[node["panel"] * d.order + degree, column] = node["weight"] * value
        def node_features():
            output = base.arb_mat(m, added)
            sqrt8 = base.a(8).sqrt()
            for row, node in enumerate(node_info):
                work.check()
                denominator = node["epsilon"] + 4
                for ell in range(start_power, powers):
                    prefactor = sqrt8 * node["q"] * node["z"]**ell / denominator
                    output[row, 2 * (ell - start_power)] = prefactor * node["f"]
                    output[row, 2 * (ell - start_power) + 1] = prefactor * node["j"]
            return output
        x = work.operation("TG node-feature entries", m * added, node_features)
        new_columns = multiply(work, embedding, x, n * added, "TG rectangular factor formation")
        factor = base.arb_mat(n, r)
        for row in range(n):
            work.check()
            for column in range(2 * start_power):
                factor[row, column] = reuse["F"][row, column]
            for column in range(added):
                factor[row, 2 * start_power + column] = new_columns[row, column]
        if reuse is None:
            def trace_sum():
                value = base.a(0)
                for node in node_info:
                    work.check()
                    value += node["weight"] * node["q"]**2 * (node["f"]**2 + node["j"]**2) / (2 * node["epsilon"])
                return base.upper(value)
            trace = work.operation("TG same-rule trace sum", m, trace_sum)
        else:
            trace = reuse["trace"]
        tail = base.upper(rho_z**(2 * powers) * trace)
        record.update({"rho_z_actual_upper": base.encoded(rho_z), "same_rule_trace_upper": base.encoded(trace),
                       "full_M_Cauchy_remainder_norm_upper_TG17": base.encoded(tail),
                       "factor_arithmetic_radius_norm_upper": base.encoded(base.matrix_enclosure_radius(factor)),
                       "positive_same_rule_remainder_includes_every_omitted_power": True,
                       "whole_rectangular_ball_action_required": True, "status": "same_full_rule_factor_and_remainder_enclosed"})
        return {"F": factor, "tail": tail, "trace": trace, "rho_z": rho_z, "node_info": node_info,
                "powers": powers, "signature": signature, "token": factor_token, "d": d}
    except BaseException:
        work.release(factor_token)
        raise
    finally:
        work.release(scratch_token)


def candidate_seed(factor, work, record):
    """TG19 bounded approximate call is used ONLY as an untrusted seed."""
    matrix = factor["F"]
    n, r = matrix.nrows(), matrix.ncols()
    token = work.reserve(2 * n * r + 12 * r**2 + 8 * n + 6 * r,
                         "TG simultaneous dyadic factor/small Gram/complex eigensolver/seed arrays")
    try:
        f0 = base.matrix_midpoint(matrix)
        gram = multiply(work, base.transpose(f0), f0, r**2, "TG small untrusted seed Gram")
        wrapper = base.as_complex_matrix(gram)
        tolerance = base.arb((1, -(base.ctx.prec - 32)))
        record.update({"role": "untrusted candidate only; no convergence/eigenvalue enclosure inferred",
                       "algorithm": "approx", "maxiter": 8 * r, "tolerance_exact": base.encoded(tolerance),
                       "all_feature_columns_retained": True, "status": "unresolved"})
        values, right = work.operation("TG bounded untrusted eigensolver", r**2,
            lambda: wrapper.eig(right=True, algorithm="approx", maxiter=8 * r, tol=tolerance))
        if len(values) != r or right.nrows() != r or right.ncols() != r:
            raise base.Unresolved("The untrusted seed call returned wrong dimensions")
        if any(not value.is_finite() for value in values) or not base.finite_matrix(right):
            raise base.Unresolved("The untrusted seed call returned nonfinite entries")
        centers = [exact_fraction(value.real.mid()) for value in values]
        selected = max(range(r), key=lambda index: (centers[index], -index))
        column = [(exact_fraction(right[i, selected].real.mid()), exact_fraction(right[i, selected].imag.mid()))
                  for i in range(r)]
        pivot = max(range(r), key=lambda index: (column[index][0]**2 + column[index][1]**2, -index))
        pre, pim = column[pivot]
        rotated_real = base.arb_mat(r, 1)
        for i, (real, imaginary) in enumerate(column):
            rotated_real[i, 0] = exact_ball(real * pre + imaginary * pim)
        lifted = multiply(work, f0, rotated_real, n * r, "TG untrusted seed lift")
        def normalize():
            seed = [exact_fraction(lifted[i, 0].mid()) for i in range(n)]
            squared = sum((value**2 for value in seed), F(0))
            if squared <= 0:
                raise base.Unresolved("The exact lifted dyadic seed is zero")
            norm = base.positive(exact_ball(squared), "TG exact positive seed sum of squares").sqrt()
            coeffs = base.arb_mat(n, 1)
            for i, value in enumerate(seed):
                coeffs[i, 0] = exact_ball(value) / norm
            return seed, squared, norm, coeffs
        seed, squared, normalization, coeffs = work.operation("TG exact seed normalization/recording", n, normalize)
        center = centers[selected]
        if not center > F(7, 10000):
            raise base.Unresolved("The untrusted exact dyadic center does not exceed the full interaction complement bound")
        record.update({"selected_eigenvalue_column": selected, "phase_pivot_component": pivot,
                       "tie_rule": "largest exact dyadic real midpoint/squared modulus, then lowest index",
                       "lambda_hat_exact_dyadic": dyadic_record(center),
                       "unnormalized_seed_exact_dyadics": [dyadic_record(value) for value in seed],
                       "sum_of_seed_squares_exact_dyadic": dyadic_record(squared),
                       "positive_square_root_normalization": base.encoded(normalization),
                       "componentwise_midpoint_is_not_an_accepted_normalized_state": True,
                       "status": "nonzero_exact_correlated_normalized_seed_defined"})
        return {"seed": seed, "squared": squared, "normalization": normalization, "coeffs": coeffs, "lambda": center}
    finally:
        work.release(token)


def normalized_function(d, seed, normalization):
    return base.RadialFunction(lambda panel, radius, analytic:
        sum((base.c(exact_ball(seed[panel * d.order + degree]) / normalization) * value
             for degree, value in enumerate(base.legendre_values(radius, d.panels[panel], d.order))), base.c(0)),
        "TG exact normalized polynomial seed", set(range(len(d.panels))))


def certify_vector(d, factor, candidate, theorem, bounds, work, record):
    f, c = factor["F"], candidate["coeffs"]
    n, r = f.nrows(), f.ncols()
    token = work.reserve(n * r + 8 * n + 4 * r, "TG simultaneous transposed factor/normalized coefficients/residual actions")
    try:
        ftc = multiply(work, base.transpose(f), c, n * r, "TG residual first rectangular action")
        action = multiply(work, f, ftc, n * r, "TG residual second rectangular action")
        residual = action - exact_ball(candidate["lambda"]) * c
        action_error = norm_vector(residual)
        full_error = base.upper(action_error + factor["tail"] + bounds["T_errors_total"])
        denominator = base.positive(base.a(candidate["lambda"] - F(7, 10000)), "TG exact-rational complement denominator")
        s = base.upper(full_error / denominator)
        epsilon = base.upper(2 * s + theorem["epsilon_H"])
        record.update({"whole_feature_and_coefficient_ball_residual_norm_upper": base.encoded(action_error),
                       "full_M_Cauchy_remainder_upper": base.encoded(factor["tail"]),
                       "complete_GE_T_error_upper": base.encoded(bounds["T_errors_total"]),
                       "actual_full_T_residual_upper_TG18": base.encoded(full_error),
                       "complement_denominator_exact_rational": str(candidate["lambda"] - F(7, 10000)),
                       "interaction_sine_angle_upper": base.encoded(s),
                       "interaction_vector_error_upper": base.encoded(base.upper(2 * s)),
                       "completed_H_vector_transfer_error_upper": base.encoded(theorem["epsilon_H"]),
                       "phase_aligned_ground_vector_error_upper_TG13": base.encoded(epsilon),
                       "whole_ball_action_not_midpoint_residual": True})
        if not s < 1 or not epsilon <= base.a(F(1, 10**8)).lower():
            raise base.Unresolved("The actual full-T vector-angle/ground-error target is unresolved")
        vector = normalized_function(d, candidate["seed"], candidate["normalization"])
        overlap = d.inner(d.trial(0), vector)
        flipped = False
        if overlap.upper() < 0:
            candidate["seed"] = [-value for value in candidate["seed"]]
            candidate["coeffs"] = -candidate["coeffs"]
            vector = normalized_function(d, candidate["seed"], candidate["normalization"])
            overlap = d.inner(d.trial(0), vector)
            flipped = True
        margin = base.positive(overlap.lower() - base.upper(theorem["G00"]).sqrt() * epsilon,
                               "TG same frozen f-overlap phase margin")
        ecenter = (theorem["energy_endpoints"][0] + theorem["energy_endpoints"][1]) / 2
        width = theorem["energy_endpoints"][1] - theorem["energy_endpoints"][0]
        derived_residual = base.upper((theorem["Emax"] + abs(exact_ball(ecenter))) * epsilon + base.a(width / 2))
        record.update({"continuous_f_overlap_enclosure": base.encoded(overlap), "exact_seed_sign_flipped": flipped,
                       "same_residual_norm_is_invariant_under_exact_sign_flip": True,
                       "strict_true_ground_phase_margin_lower": base.encoded(base.lower(margin)),
                       "original_H_energy_center_exact_dyadic": dyadic_record(ecenter),
                       "energy_center_is_exact_endpoint_midpoint": True, "energy_center_displacement": "0",
                       "derived_original_H_residual_upper_TG14": base.encoded(derived_residual),
                       "derived_H_residual_used_as_small_residual_over_gap_proof": False,
                       "numerical_completed_H_residual": None, "status": "same_full_H_ground_vector_certified"})
        return vector, epsilon
    finally:
        work.release(token)


def build(configuration, powers, work, direct_input, shared, record):
    """GE preflight precedes features; all TG point gates apply to every build."""
    base.ctx.dps = configuration.digits
    record.update({"configuration": configuration.saved(), "Cauchy_powers": powers,
                   "status": "unresolved", "model_evaluation_attempted": True, "model_evaluated": False,
                   "ground_point_certified": False, "ground_preparation_certified": False,
                   "numeric_BS_counts": None, "numeric_full_M_LDL": None, "finite_HQ_eigenvalue": None,
                   "numeric_completed_H_residual": None, "completed_H_exponentials": None,
                   "consumer_root_rechecks": []})
    mu2, gamma = direct_input.recheck(work, record["consumer_root_rechecks"])
    record["same_exact_mu2_gamma_intersections_across_all_builds"] = {}
    intersections = input_api.direct.compare_enclosures(direct_input.b, shared, {"mu2": mu2, "gamma": gamma},
        record["same_exact_mu2_gamma_intersections_across_all_builds"], {"mu2", "gamma"})
    shared.update(intersections)  # Disjoint input enclosures halt for review; never refine away.
    mu2, gamma = shared["mu2"], shared["gamma"]
    midpoint = base.positive(gamma.mid(), "TG deterministic same-coupling midpoint")
    record.update({"mu2": base.encoded(mu2), "gamma": base.encoded(gamma),
                   "gamma_midpoint_exact": base.encoded(midpoint),
                   "gamma_midpoint_mismatch_upper": base.encoded(base.upper(abs(gamma - midpoint))),
                   "direct_compression_used_as_input": True, "direct_compression_is_ground_certificate": False})
    panels = base.panels_for(configuration.cutoff)
    node_token = work.reserve(len(panels) * configuration.quadrature * (configuration.order + 8),
                              "TG live actual node/basis data before construction")
    coefficient_token, factor = None, None
    try:
        tolerance = F(1, 10**max(64, configuration.digits // 2))
        d = work.operation("TG unchanged radial discretization", 0, lambda: base.Discretization(
            configuration.cutoff, configuration.order, configuration.quadrature, configuration.weight_order, work, tolerance))
        record["GE_reference_error_certificate"] = {}
        reference = base.ground_reference_bounds(d, record["GE_reference_error_certificate"])
        bounds = work.operation("TG unchanged GE completed-H preflight bounds", 0,
            lambda: base.completion_bounds(d, gamma, midpoint, degree=0, reference_errors=reference))
        record["GE_reference_error_certificate"].update({
            "T_complex_norm_upper": base.encoded(bounds["T_complex_norm"]),
            "TQ_complex_norm_upper": base.encoded(bounds["TQ_complex_norm"]),
            "TQ_reference_norm_upper": base.encoded(bounds["TQ_reference_norm"]),
            "full_completed_H_error_upper": base.encoded(bounds["H_errors"][0])})
        record["ground_operator_error_allocation"] = {name: base.encoded(values[0]) for name, values in bounds["categories"].items()}
        record["ground_full_operator_norm_error_upper"] = base.encoded(bounds["H_errors"][0])
        if not bounds["H_errors"][0] < base.a(F(1, 10**13)).lower():
            advance, refinement = base.ground_preflight_refinement(bounds, {"current_compression_completed": False})
            record["ground_preflight_refinement"] = refinement
            if advance:
                raise base.RefineOrder("TG failed unchanged GE preflight advances the existing GR order")
            raise base.Unresolved("TG unchanged GE completed-H preflight remains unresolved")
        record["GE_preflight_passed_before_features"] = True
        record["full_operator_theorem"] = {}
        theorem = theorem_bounds(direct_input, gamma, work, record["full_operator_theorem"])
        record["Cauchy_factor"] = {}
        factor = build_factor(d, powers, work, record["Cauchy_factor"])
        coefficient_token = work.reserve(4 * d.rank, "TG retained exact/normalized polynomial coefficient data")
        record["untrusted_seed"] = {}
        candidate = candidate_seed(factor, work, record["untrusted_seed"])
        record["full_T_vector_certificate"] = {}
        vector, epsilon = certify_vector(d, factor, candidate, theorem, bounds, work, record["full_T_vector_certificate"])
        record["untrusted_seed"]["unnormalized_seed_exact_dyadics"] = [dyadic_record(value) for value in candidate["seed"]]
        record["ground_export"] = {
            "representation": REPRESENTATION, "proof": "TG full-T residual + full-H min-max/vector transfer",
            "certified_after_all_controls": False, "full_H_ground_not_finite_HQ_eigenstate": True,
            "radial_chart": "L2(nu_alpha), alpha=1/1000000; implemented chart L2(dr)",
            "panels": [{"left_exact_rational": str(p.left), "right_exact_rational": str(p.right)} for p in d.panels],
            "Legendre_order_per_panel": d.order, "cutoff": d.cutoff, "zero_extension_beyond_cutoff": True,
            "unnormalized_seed_exact_dyadics": [dyadic_record(value) for value in candidate["seed"]],
            "sum_of_seed_squares_exact_dyadic": dyadic_record(candidate["squared"]),
            "positive_sqrt_normalization": base.encoded(candidate["normalization"]),
            "normalization_is_the_exact_positive_sqrt_of_the_retained_exact_sum": True,
            "normalized_coefficient_balls": [base.encoded(candidate["coeffs"][i, 0]) for i in range(d.rank)],
            "arbitrary_componentwise_midpoints_are_not_this_certified_state": True,
            "E_g": theorem["E_g"], "original_intrinsic_separation_lower": base.encoded(theorem["separation"]),
            "phase_aligned_vector_error_upper": base.encoded(epsilon),
            "derived_original_H_residual": record["full_T_vector_certificate"],
            "full_T_and_reference_error_records": {"factor": record["Cauchy_factor"],
                                                    "GE": record["GE_reference_error_certificate"]},
            "undo_measure_absorption": "v_E=v_dr/q_alpha almost everywhere; q_alpha=r*sqrt(W(alpha^2*r^2))/(sqrt(2)*pi)",
            "full_momentum_embedding": "u_full(k)=v_E(r(k))*chi_s/sqrt(8)",
            "full_momentum_measure": "d^3k/(2*pi)^3; pushforward=8*nu_alpha", "full_embedding_is_isometric": True,
            "old_null_minor_HQ_representation": None, "numerical_completed_H_residual": None,
            "N_consumer_adapter_released": False, "Ramsey_consumer_adapter_released": False}
        record.update({"E_g": theorem["E_g"], "ground_point_certified": True, "model_evaluated": True,
                       "status": "TG_point_counts_energy_vector_phase_and_preflight_pass"})
        return {"d": d, "factor": factor, "function": vector, "epsilon": epsilon, "theorem": theorem,
                "bounds": bounds, "node_token": node_token, "coefficient_token": coefficient_token,
                "configuration": configuration, "record": record}
    except BaseException:
        if factor is not None:
            work.release(factor["token"])
        if coefficient_token is not None:
            work.release(coefficient_token)
        work.release(node_token)
        raise


def release_build(work, item):
    work.release(item["factor"]["token"])
    work.release(item["node_token"])
    work.release(item["coefficient_token"])
    item["factor"].clear()
    item.clear()  # Discard the actual arrays when their simultaneous reservation ends.


def compare_builds(baseline, candidate, record):
    left, right = baseline["theorem"]["energy_endpoints"], candidate["theorem"]["energy_endpoints"]
    overlap = max(left[0], right[0]) <= min(left[1], right[1])
    record["same_full_H_analytical_energy_intervals_overlap"] = overlap
    record["energy_overlap_is_not_independent_numeric_eigenvalue_agreement"] = True
    difference = candidate["d"].norm(baseline["function"] - candidate["function"])
    allowance = base.upper(baseline["epsilon"] + candidate["epsilon"])
    record.update({"continuous_normalized_ground_function_difference_upper": base.encoded(difference),
                   "combined_phase_aligned_vector_errors_upper": base.encoded(allowance),
                   "whole_continuous_integral_balls_retained": True})
    if not overlap or not difference <= allowance:
        raise base.Unresolved("A baseline-relative TG full-energy/continuous-vector control failed")
    record["status"] = "same_full_H_baseline_relative_control_pass"


def h_action(function):
    return base.RadialFunction(lambda panel, radius, analytic:
        base.free_multiplier(radius, source_t=0, analytic=analytic) * function(panel, radius, analytic),
        "TG exact continuous H0*" + function.label, function.support_panels)


def factored_action(d, factor, function, work):
    token = work.reserve(d.rank * factor["F"].ncols() + 6 * d.rank + 3 * factor["F"].ncols(),
                         "TG kernel transposed factor/projected-function/action arrays")
    try:
        projection = d.project(function)
        column = base.arb_mat(d.rank, 1)
        for index, value in enumerate(projection):
            column[index, 0] = value
        n, r = factor["F"].nrows(), factor["F"].ncols()
        inner = multiply(work, base.transpose(factor["F"]), column, n * r, "TG kernel first rectangular action")
        output = multiply(work, factor["F"], inner, n * r, "TG kernel second rectangular action")
        return d.polynomial([output[i, 0] for i in range(n)], "TG factored projected action")
    finally:
        work.release(token)


def native_kernel_controls(baseline, representation_control, work, record):
    d, bounds, emax = baseline["d"], baseline["bounds"], baseline["theorem"]["Emax"]
    base.ctx.dps = baseline["configuration"].digits
    powers = 5 * d.order // 4
    if 4 * powers != 5 * d.order:
        raise base.Unresolved("The declared kernel-only Cauchy order is not integral")
    record["kernel_only_factor"] = {}
    reuse = representation_control["factor"] if representation_control["factor"]["powers"] <= powers else None
    accurate = build_factor(d, powers, work, record["kernel_only_factor"], reuse=reuse)
    try:
        samples = []
        record["kernel_identity_discrepancy_balls"] = samples
        for panel in d.panels:
            radius, other = base.a((panel.left + panel.right) / 2), base.a(panel.left + panel.length / 3)
            for source in (-base.a(base.T0), base.a(0), base.a(base.T0)):
                native = base.kernel_jet(radius, other, source, degree=0).values[0]
                ur, us = (-2 * source).exp() * radius**2, (-2 * source).exp() * other**2
                er, es = (1 + ur).sqrt(), (1 + us).sqrt()
                fr, fs = (-ur).exp() * (er + 1) / (2 * er), (-us).exp() * (es + 1) / (2 * es)
                jr, js = (-ur).exp() * (er - 1) / (2 * er), (-us).exp() * (es - 1) / (2 * es)
                discrepancy = (2 * er + 2 * es) * native - 2 * (-3 * source).exp() * (fr * fs + jr * js)
                if not discrepancy.contains(0) or not base.abs_upper(discrepancy) < d.tolerance * 100:
                    raise base.Unresolved("TG original native pre-measure kernel identity unresolved")
                samples.append(base.encoded(discrepancy))
        f = d.trial(0)
        jleg = base.RadialFunction(lambda panel, radius, analytic:
            base.absorption(radius, d.coefficients, analytic) * (-radius**2).exp()
            * (base.guarded_root(1 + radius**2, analytic) - 1) / (2 * base.guarded_root(1 + radius**2, analytic)),
            "TG q_alpha*j")
        rows = []
        record["coupling_derivative_controls"] = rows
        for trial in (d.trial(0), d.trial(1)):
            polynomial_token = work.reserve(4 * d.rank, "TG simultaneous two returned kernel-action polynomials")
            try:
                completed = factored_action(d, accurate, h_action(trial), work).scaled(-1) + h_action(
                    factored_action(d, accurate, trial, work)).scaled(-1)
                overlap_f, overlap_j = d.inner(f, trial), d.inner(jleg, trial)
                bare = f.scaled(2 * overlap_f) + jleg.scaled(2 * overlap_j)
                exterior = base.upper(2 * d.trial_tail(0).sqrt() * (base.abs_upper(overlap_f) + base.abs_upper(overlap_j)))
                norm_trial = d.norm(trial)
                action_tail = base.upper(2 * emax * accurate["tail"] * norm_trial)
                raw_mismatch = d.norm(completed + bare)
                measured = base.upper(raw_mismatch + exterior + action_tail)
                threshold = base.upper(2 * emax * bounds["T_errors_total"] * norm_trial)
                rows.append({"trial": trial.label, "directed_continuous_factored_discrepancy_upper": base.encoded(raw_mismatch),
                             "original_exterior_allowance_upper": base.encoded(exterior),
                             "separate_complete_action_tail_on_measured_side_TG21": base.encoded(action_tail),
                             "measured_total_upper": base.encoded(measured), "UNCHANGED_original_threshold": base.encoded(threshold)})
                if not measured <= threshold:
                    raise base.Unresolved("TG21 original coupling control fails its unchanged threshold")
            finally:
                completed = bare = None
                work.release(polynomial_token)
        def trace_integrand(radius, analytic):
            epsilon = base.guarded_root(1 + radius**2, analytic)
            upper_leg = (-radius**2).exp() * (epsilon + 1) / (2 * epsilon)
            lower_leg = (-radius**2).exp() * (epsilon - 1) / (2 * epsilon)
            return base.absorption(radius, d.coefficients, analytic)**2 * (upper_leg**2 + lower_leg**2) / (2 * epsilon)
        low = sum((d.integrals.real(trace_integrand, panel.left, panel.right, "TG retained real T trace") for panel in d.panels), base.a(0))
        tail = base.upper(1 / (16 * base.a(base.ALPHA)**3) * base.a(-2 * d.cutoff**2).exp())
        trace = base.upper(low + tail)
        record.update({"kernel_identity_discrepancy_balls": samples, "coupling_derivative_controls": rows,
                       "real_full_trace_upper": base.encoded(trace),
                       "real_crossblock_tail_check_upper": base.encoded(base.upper(tail + 2 * (trace * tail).sqrt())),
                       "real_tail_does_not_replace_GE_full_lattice_tail": True, "status": "all_original_native_kernel_controls_pass"})
    finally:
        work.release(accurate["token"])


def new_result(args, admission):
    return {"scope": "TG same original-H prepared radial reducing fiber; no full-Fock/star/clock/COM/consumer result",
            "source": SPEC, "independent_source_review": REVIEW, "stage": "ground_top_interaction",
            "representation": REPRESENTATION, "fixed_parameters": input_api.direct.FIXED_PARAMETERS,
            "backend": "isolated python-flint0.9.0; directed Arb/Acb; bounded approximate seed is untrusted",
            "admission": admission, "status": "unresolved", "model_evaluation_attempted": False,
            "model_evaluated": False, "ground_point_certified": False, "ground_preparation_certified": False,
            "ramsey_certified": False, "full_clock_certified": False, "COM_enclosure_certified": False,
            "COM_response_or_triplet_gap_certified": False, "N_input_certified": False,
            "N_consumer_adapter_released": False, "Ramsey_consumer_adapter_released": False,
            "full_fixed_charge_Fock_ground_certified": False, "stellar_ground_certified": False,
            "open_exterior_flow_certified": False, "differentiable_physical_window_certified": False,
            "physical_clock_load_continuation_certified": False, "trapping_radius_certified": False,
            "temperature_certified": False, "numerical_BS_counts": None, "numerical_full_M_LDL": None,
            "finite_HQ_energy": None, "old_null_minor_HQ_representation": None,
            "completed_H_exponentials": None, "numerical_completed_H_residual": None,
            "E_g": None, "ground_export": None, "mu2": None, "gamma": None,
            "accepted_direct_input_validation": {}, "baseline_attempts": [], "unresolved_item": None,
            "predeclared_engineering_caps": {"seconds": args.seconds, "requested_work_units": args.max_evaluations,
                "integrator_evaluations_per_call": args.integral_evaluations, "resident_memory_MB": args.memory_mb,
                "maximum_threads": 4, "one_arithmetic_worker": True,
                "requested_counter_includes_rejected_reservations": True,
                "proxy_is_not_elementary_operation_count": True,
                "baseline_Cauchy_powers": BASE_POWERS, "representation_control_Cauchy_powers": 2 * BASE_POWERS,
                "no_predicted_total_cost_or_success": True}}


def run(args, admission):
    work, result = Work(args), new_result(args, admission)
    advance_orders, shared = set(), {"mu2": None, "gamma": None}
    direct_input = None
    active = None
    try:
        direct_input = input_api.DirectInput(args.direct_compression_input, base.flint_module, admission, work,
                                             result["accepted_direct_input_validation"])
        shared.update({"mu2": direct_input.mu2, "gamma": direct_input.gamma})
        for configuration in base.configurations():
            if (configuration.cutoff, configuration.order) in advance_orders:
                continue
            row = {"configuration": configuration.saved(), "status": "unresolved", "baseline": {},
                   "baseline_relative_controls": [], "Cartesian_measure_controls": None,
                   "native_kernel_controls": None}
            result["baseline_attempts"].append(row)
            baseline, representation_control = None, None
            result["model_evaluation_attempted"] = True
            try:
                active = row["baseline"]
                baseline = build(configuration, BASE_POWERS, work, direct_input, shared, active)
                result.update({"model_evaluated": True, "ground_point_certified": True,
                               "E_g": active["E_g"], "ground_export": active["ground_export"]})
                choices = [(name, cfg, BASE_POWERS) for name, cfg in base.next_configurations(configuration)]
                choices.append(("Cauchy_representation", configuration, 2 * BASE_POWERS))
                for name, cfg, powers in choices:
                    control = {"axis": name, "baseline_configuration": configuration.saved(),
                               "configuration_result": {}, "comparison": {}, "status": "unresolved"}
                    row["baseline_relative_controls"].append(control)
                    active = control["configuration_result"]
                    candidate = build(cfg, powers, work, direct_input, shared, active)
                    try:
                        compare_builds(baseline, candidate, control["comparison"])
                        control["status"] = "independent_baseline_relative_build_and_continuous_ground_control_pass"
                        if name == "Cauchy_representation":
                            representation_control = candidate
                            candidate = None
                    finally:
                        if candidate is not None:
                            release_build(work, candidate)
                base.ctx.dps = configuration.digits
                row["Cartesian_measure_controls"] = {}
                work.operation("TG original Cartesian/radial all-preimage control", 0,
                               lambda: cartesian_api.measure_controls(baseline["d"], row["Cartesian_measure_controls"]))
                row["native_kernel_controls"] = {}
                native_kernel_controls(baseline, representation_control, work, row["native_kernel_controls"])
                work.check()
                row["status"] = "baseline_four_separate_controls_Cartesian_and_native_kernel_pass"
                baseline["record"]["ground_export"]["certified_after_all_controls"] = True
                result.update({"status": "certified_ground_preparation_TG", "ground_preparation_certified": True,
                               "ground_point_certified": True, "model_evaluated": True,
                               "ground_export": baseline["record"]["ground_export"],
                               "E_g": baseline["record"]["E_g"], "accepted_control_records": row,
                               "unresolved_item": None})
                break
            except base.EngineeringCap:
                raise
            except input_api.direct.InconsistentEnclosures:
                raise
            except (base.Unresolved, base.AnalyticDomain, input_api.InputUnresolved, input_api.direct.Unresolved) as error:
                if isinstance(error, base.RefineOrder):
                    advance_orders.add((configuration.cutoff, configuration.order))
                row.update({"status": "unresolved", "unresolved_item": type(error).__name__ + ": " + str(error)})
                result["unresolved_item"] = row["unresolved_item"]
                if active is not None:
                    active["unresolved_item"] = row["unresolved_item"]
                continue
            finally:
                if representation_control is not None:
                    release_build(work, representation_control)
                if baseline is not None:
                    release_build(work, baseline)
        else:
            result["unresolved_item"] = "Existing TG cutoff/order/precision ladder exhausted without every required gate/control"
    except (base.EngineeringCap, base.Unresolved, input_api.InputUnresolved, input_api.direct.Unresolved) as error:
        result.update({"status": "unresolved", "ground_preparation_certified": False,
                       "unresolved_item": type(error).__name__ + ": " + str(error)})
        if result["baseline_attempts"]:
            result["baseline_attempts"][-1].update({"status": "unresolved", "unresolved_item": result["unresolved_item"]})
        if isinstance(error, input_api.direct.InconsistentEnclosures):
            result["halted_for_source_input_review"] = True
    except Exception as error:
        result.update({"status": "backend_or_arithmetic_error", "ground_preparation_certified": False,
                       "unresolved_item": type(error).__name__ + ": " + str(error)})
    finally:
        result["requested_work_units"] = work.evaluations
        result["elapsed_engineering_seconds"] = time.perf_counter() - work.start
        result["rectangular_operations_requested_admitted_completed"] = work.operations
        result["guard_progress_run_length_segments"] = work.guard_segments
        result["simultaneous_memory_reservations"] = work.memory_reservations
        if shared["mu2"] is not None:
            result["mu2"], result["gamma"] = base.encoded(shared["mu2"]), base.encoded(shared["gamma"])
    return result


def write_result(path, result):
    path.write_text(json.dumps(result, indent=2) + "\n")
    lines = [result["scope"], "status=" + result["status"],
             "requested_work_units=" + str(result["requested_work_units"]),
             "elapsed_engineering_seconds=" + str(result["elapsed_engineering_seconds"]),
             "baseline_attempts=" + str(len(result["baseline_attempts"])),
             "ground_point_certified=" + str(result["ground_point_certified"]),
             "ground_preparation_certified=" + str(result["ground_preparation_certified"]),
             "representation=" + REPRESENTATION]
    if result["unresolved_item"]:
        lines.append("unresolved: " + result["unresolved_item"])
    lines.append("Numerical BS/LDL/HQ/completed-H residual: unexecuted. N/Ramsey/fullclock/COM/consumer/full-Fock/star/flow claims: false.")
    path.with_suffix(".txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direct-compression-input", type=Path, required=True)
    parser.add_argument("--actual-code-audits-released", action="store_true")
    parser.add_argument("--direct-compression-input-audits-released", action="store_true")
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--max-evaluations", type=int, default=2_000_000)
    parser.add_argument("--integral-evaluations", type=int, default=20_000)
    parser.add_argument("--memory-mb", type=int, default=4096)
    args = parser.parse_args()
    if not args.actual_code_audits_released or not args.direct_compression_input_audits_released:
        parser.error("Separate E/B TG code audits, accepted direct artifact audits and parent runtime release are required")
    if (args.seconds, args.max_evaluations, args.integral_evaluations, args.memory_mb) != (900, 2_000_000, 20_000, 4096):
        parser.error("TG retains the exact declared 900s/2M/20000-per-integral/4096MB caps")
    if not args.direct_compression_input.is_file():
        parser.error("The accepted direct compression JSON is required")
    if args.output.suffix != ".json" or args.output.resolve() == args.output.with_suffix(".txt").resolve():
        parser.error("Distinct new JSON/TXT output destinations are required")
    if not args.output.parent.is_dir() or args.output.exists() or args.output.with_suffix(".txt").exists():
        parser.error("Output must be new files in the parent-admitted results directory")
    if any(os.environ.get(name) != "1" for name in POOL_NAMES) or os.environ.get("QI2_GPU_UUID") != args.gpu_uuid:
        parser.error("All six pools must be one and the GPU UUID must match the launch environment")
    base.load_backend()  # All host imports finish before final same-process admission.
    base.ctx.threads = 1
    admission = remote_execution(max_threads=4)
    print("TG admitted " + json.dumps(admission), flush=True)
    result = run(args, admission)
    write_result(args.output, result)
    print(result["status"], flush=True)
    if not result["ground_preparation_certified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
