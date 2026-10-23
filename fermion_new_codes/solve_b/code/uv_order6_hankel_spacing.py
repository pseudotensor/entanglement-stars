#!/usr/bin/env python3
"""Bounded original-sine spacing enclosure, reviewed equations W1--W4.

The fixed rectangle and exact sixteen beta intervals are declared before
work.  This is a parameter enclosure, not sampled-spacing interpolation.
The first failed completed interval test ends this finite contract as
inconclusive; there is no automatic refinement or change of resolution.

The accepted fixed-pilot source supplies unchanged arithmetic functions
and declared packet/input data.  Its physical fourth-component symbol
and H3 recurrence are not called.  This file implements the auxiliary w
and every explicit beta factor in W1.  Execution is remote qi2 only, after the same-process
free-device query, visible CUDA allocation and sole-owned-PID check.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from fractions import Fraction
from pathlib import Path

# Complete host-only imports with all vendor thread pools limited before
# the launch admission guard.  Numerical work has one Python worker.
for _thread_variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
from mpmath import iv

import remote_runtime
import uv_order6_hankel as baseline

arithmetic = baseline.arithmetic
BI = arithmetic.BinaryInterval
SELECTOR = "r3-sine-m1000-beta0to1e-4"
NORMAL_BLOCKS = 16
FFT_SIZE = 16384
GAUSS_ORDER = 12
DIGITS = 80
ROOT_BITS = 120
MAX_THREADS = 4
PREFIX = "uv_order6_hankel_spacing"
BETA_INTERVALS = tuple((Fraction(index, 160000), Fraction(index + 1, 160000))
                       for index in range(16))
EVALUATION_ORDER = (0, 15, *range(1, 15))
PACKETS = baseline.PACKETS
INPUT_LABELS = ("v0_minus_one", "v1", "v2", "w")


def resource_check(execution):
    count = len(list(Path("/proc/self/task").iterdir()))
    if os.getpriority(os.PRIO_PROCESS, 0) < 10 or count > MAX_THREADS:
        raise RuntimeError(f"Remote numerical work requires nice>=10 and <={MAX_THREADS} total threads")
    execution["maximum_threads_observed"] = max(execution.get("maximum_threads_observed", 0), count)


def save_json(output_dir, suffix, result):
    output_dir.mkdir(parents=True, exist_ok=True)
    name = PREFIX + suffix + ".json"
    destination = output_dir / name
    temporary = output_dir / (name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    if suffix == "":
        (output_dir / (PREFIX + ".txt")).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def auxiliary_symbol(c, u, transverse, mass_squared, beta):
    """W's homogeneous real inputs on the entire assigned beta interval."""
    mass = BI.rational(mass_squared)
    s = c.square() + beta * u.square()
    t = transverse.square() * s + u.square()
    f = mass * (s ** 6) + t ** 6
    e = f + t * (s ** 5)
    root_e = e.sqrt()
    z = t * (s ** 5) / f
    root_one_plus_z = (1 + z).sqrt()
    # This is v0-1 exactly; do not subtract one from rounded v0.
    first = -z / (root_one_plus_z * (1 + root_one_plus_z))
    return (first, transverse * (s ** 3) / root_e,
            u * c * s.square() / root_e, u.square() * s.square() / root_e)


def factored_moment_recurrence(real, imaginary, beta, blocks):
    """W1, with the auxiliary 4x4 outer block and explicit beta weights.

    Its upper-left 3x3 block is B, last column is c, last row is d,
    and last diagonal entry is e.  The row d is not conjugated again.
    Simultaneously exchanging r,s conjugates the block, preserving the
    real-beta summand, so each nonzero diagonal offset has multiplicity2.
    """
    norms = real.square() + imaginary.square()
    v_norms = norms[:, 0] + norms[:, 1] + norms[:, 2]
    weights = np.asarray([min(index, 2 * blocks - index) for index in range(1, 2 * blocks)],
                         dtype=np.float64)
    m1 = (2 * BI(weights) * (v_norms + beta * norms[:, 3])).sum()
    m2 = BI.rational(0)
    diagonal = np.arange(3, dtype=np.int64)
    first = np.asarray([0, 0, 1], dtype=np.int64)
    second = np.asarray([1, 2, 2], dtype=np.int64)
    for difference in range(blocks):
        a_real, a_imaginary = BI(np.zeros((4, 4))), BI(np.zeros((4, 4)))
        for index in range(blocks):
            term_real, term_imaginary = baseline.outer_product(real, imaginary, index, index + difference)
            a_real, a_imaginary = a_real + term_real, a_imaginary + term_imaginary
        pair_weight = 1 if difference == 0 else 2
        for row in range(blocks - difference):
            trace_real = a_real[diagonal, diagonal].sum() + beta * a_real[3, 3]
            trace_imaginary = a_imaginary[diagonal, diagonal].sum() + beta * a_imaginary[3, 3]
            antisymmetric_real = a_real[first, second] - a_real[second, first]
            antisymmetric_imaginary = a_imaginary[first, second] - a_imaginary[second, first]
            mixed_real = a_real[diagonal, 3] - a_real[3, diagonal]
            mixed_imaginary = a_imaginary[diagonal, 3] - a_imaginary[3, diagonal]
            summand = trace_real.square() + trace_imaginary.square()
            summand += (antisymmetric_real.square() + antisymmetric_imaginary.square()).sum()
            summand += beta * (mixed_real.square() + mixed_imaginary.square()).sum()
            m2 += Fraction(pair_weight, 2) * summand
            if row + 1 < blocks - difference:
                column = row + difference
                removed_real, removed_imaginary = baseline.outer_product(real, imaginary, row, column)
                added_real, added_imaginary = baseline.outer_product(real, imaginary, row + blocks, column + blocks)
                a_real = a_real - removed_real + added_real
                a_imaginary = a_imaginary - removed_imaginary + added_imaginary
    result = []
    for value in (m1, m2):
        lower, upper = arithmetic.endpoints(value.to_iv())
        if upper < 0:
            raise ArithmeticError("Factored moment enclosure contradicts positivity")
        result.append(arithmetic.hull(max(Fraction(0), lower), upper))
    return tuple(result)


def node_moments(transverse, mass_squared, beta, c, u, reversal, twiddles, twiddle_error, aliases):
    components = auxiliary_symbol(c, u, BI.from_iv(transverse), mass_squared, beta)
    centers, input_errors = [], []
    for component in components:
        center, radii = baseline.centers_and_radii(component)
        centers.append(center)
        input_errors.append(Fraction.from_float(float(np.max(radii))))
    errors = baseline.fft_error_allowances(input_errors, twiddle_error, FFT_SIZE)
    real, imaginary = baseline.normalized_radix_two(np.stack(centers, axis=1), reversal, twiddles)
    real, imaginary = real[1:2 * NORMAL_BLOCKS], imaginary[1:2 * NORMAL_BLOCKS]
    coefficient_errors = [[alias + error[0] for error in errors] for alias in aliases]
    enclosed_real, enclosed_imaginary = baseline.coefficient_rectangles(real, imaginary, coefficient_errors)
    moments = factored_moment_recurrence(enclosed_real, enclosed_imaginary, beta, NORMAL_BLOCKS)
    # A separate arithmetic diagnostic retains interval beta but replaces
    # Fourier coefficients by their actual dyadic centers.  It is never
    # substituted for the full uncertainty enclosure or used as an error.
    center_moments = factored_moment_recurrence(BI(real), BI(imaginary), beta, NORMAL_BLOCKS)
    coefficients = []
    for row, index in enumerate(range(1, 2 * NORMAL_BLOCKS)):
        coefficients.append({
            "k": index,
            "components": [{"mu": component, "auxiliary_input": INPUT_LABELS[component],
                            "real": baseline.reported(enclosed_real[row, component].to_iv()),
                            "imaginary": baseline.reported(enclosed_imaginary[row, component].to_iv()),
                            "added_complex_modulus_error_upper": baseline.upper_report(coefficient_errors[row][component])}
                           for component in range(4)],
        })
    return moments, {
        "Q": baseline.reported(transverse),
        "Fourier_components": [{"mu": index, "auxiliary_input": INPUT_LABELS[index], **budget}
                               for index, (_, budget) in enumerate(errors)],
        "parameter_width_included_in_input_radii": True,
        "coefficients": coefficients,
        "moment_enclosures_before_radial_quadrature": {"m1": baseline.reported(moments[0]),
                                                       "m2": baseline.reported(moments[1])},
        "moment_arithmetic_at_dyadic_FFT_centers_with_interval_beta": {
            "m1": baseline.reported(center_moments[0]), "m2": baseline.reported(center_moments[1])},
        "moment_arithmetic_contract": "W1 factored auxiliary blocks; outward primitives, sliding updates and pairwise sums; beta encloses the entire assigned interval",
        "moment_center_diagnostic_usage": "Retains beta uncertainty but sets Fourier coefficients to dyadic centers; never substitutes for or modifies the full moment enclosures",
    }


def run_parameter_interval(index, output_dir, execution, context):
    lower_beta, upper_beta = BETA_INTERVALS[index]
    beta = BI.from_iv(arithmetic.hull(lower_beta, upper_beta))
    actual_lower = Fraction.from_float(float(beta.lower))
    actual_upper = Fraction.from_float(float(beta.upper))
    if not actual_lower <= lower_beta <= upper_beta <= actual_upper:
        raise ArithmeticError("Binary beta endpoints fail exact containment of the declared parameter interval")
    result = {
        "selector": SELECTOR, "parameter_index": index,
        "beta_interval_exact": [baseline.exact_string(lower_beta), baseline.exact_string(upper_beta)],
        "beta_input_binary_interval_exact": [baseline.exact_string(actual_lower), baseline.exact_string(actual_upper)],
        "beta_interval_containment_checked": True,
        "scope": "W1--W3 whole-parameter-interval native plane at the fixed mass rectangle",
        "comparison": {"common_Newton_target_outward_upper": baseline.upper_report(context["target_upper"])},
        "packets": [], "arithmetic_complete": False, "strict_interval_excess_pass": False,
        "entropy_normalization_disposition": "Interval arithmetic incomplete; no comparison credited",
    }
    suffix = f"_j{index:02d}"
    text_destination = output_dir / (PREFIX + suffix + ".txt")
    progress_lines = []
    save_json(output_dir, suffix, result)
    interval_start = time.perf_counter()
    total_lower = arithmetic.interval(0)
    for mass_squared, multiplicity, packet_indices in PACKETS:
        packet = {"mass_squared": baseline.exact_string(mass_squared), "packet_indices": packet_indices,
                  "multiplicity": multiplicity, "panels": []}
        result["packets"].append(packet)
        packet_lower = arithmetic.interval(0)
        for panel_index, (lower, upper) in enumerate(context["panels"]):
            resource_check(execution)
            points, measures = arithmetic.mapped_gauss(lower, upper, context["nodes"], context["weights"])
            quadratures = [arithmetic.interval(0), arithmetic.interval(0)]
            node_data = []
            for point, measure in zip(points, measures):
                resource_check(execution)
                moments, node = node_moments(point, mass_squared, beta, context["c"], context["u"],
                                             context["reversal"], context["twiddles"], context["twiddle_error"],
                                             context["aliases"])
                node["radial_Gauss_measure"] = baseline.reported(measure)
                node_data.append(node)
                prefactor = (2 / context["pi"]) * point * measure
                for moment in range(2):
                    quadratures[moment] += prefactor * moments[moment]
            analytic_errors = [baseline.radial_error(lower, upper, NORMAL_BLOCKS, GAUSS_ORDER,
                                                    context["pi"], moment) for moment in (1, 2)]
            moment_integrals = [baseline.positive_integral_enclosure(value, error)
                                for value, error in zip(quadratures, analytic_errors)]
            bound, selected = baseline.panel_tangent_bound(*moment_integrals, context["constants"])
            packet_lower += arithmetic.interval(bound)
            packet["panels"].append({
                "index": panel_index, "Q_interval": [baseline.exact_string(lower), baseline.exact_string(upper)],
                "quadrature_arithmetic_enclosures_before_H10": {
                    "M1": baseline.reported(quadratures[0]), "M2": baseline.reported(quadratures[1])},
                "H10_separate_analytic_Gauss_error_upper": {
                    "M1": baseline.reported(analytic_errors[0])["upper"],
                    "M2": baseline.reported(analytic_errors[1])["upper"]},
                "moment_integrals_after_H10": {"M1": baseline.reported(moment_integrals[0]),
                                               "M2": baseline.reported(moment_integrals[1])},
                **selected, "nodes": node_data,
            })
            progress_line = (f"Spacing interval {index}, packet {packet_indices}, panel {panel_index + 1}/{len(context['panels'])} enclosed; "
                             f"selected t={selected['selected_tangent']}, lower={selected['panel_entropy_lower_bound']}")
            progress_lines.append(progress_line)
            text_destination.write_text("\n".join(progress_lines) + "\n")
            print(progress_line, flush=True)
            packet["partial_lower_sum_enclosure"] = baseline.reported(packet_lower)
            save_json(output_dir, suffix, result)
        packet["single_packet_lower_sum_enclosure"] = baseline.reported(packet_lower)
        packet["multiplicity_weighted_lower_sum_enclosure"] = baseline.reported(multiplicity * packet_lower)
        total_lower += multiplicity * packet_lower
    resource_check(execution)
    lower_endpoint = arithmetic.endpoints(total_lower)[0]
    margin = lower_endpoint - context["target_upper"]
    result["comparison"].update({
        "four_packet_entropy_lower_sum_enclosure": baseline.reported(total_lower),
        "four_packet_entropy_certified_lower": baseline.lower_report(lower_endpoint),
        "lower_excess_margin": baseline.lower_report(margin),
        "margin_exact_rational_lower": baseline.exact_string(margin),
    })
    result["arithmetic_complete"] = True
    result["strict_interval_excess_pass"] = margin > 0
    result["entropy_normalization_disposition"] = (
        "Strict native planar lower-bound excess certified on this entire parameter interval"
        if margin > 0 else "Inconclusive interval lower-bound test; no entropy upper bound or normalization success")
    result["elapsed_seconds"] = time.perf_counter() - interval_start
    save_json(output_dir, suffix, result)
    summary = {key: result[key] for key in (
        "parameter_index", "beta_interval_exact", "beta_interval_containment_checked", "arithmetic_complete",
        "strict_interval_excess_pass", "comparison", "entropy_normalization_disposition", "elapsed_seconds")}
    summary["packet_lower_sums"] = [
        {key: packet[key] for key in ("mass_squared", "packet_indices", "multiplicity",
                                      "single_packet_lower_sum_enclosure", "multiplicity_weighted_lower_sum_enclosure")}
        for packet in result["packets"]]
    summary_text = json.dumps(summary, indent=2, allow_nan=False)
    text_destination.write_text("\n".join(progress_lines) + "\n" + summary_text + "\n")
    print(summary_text, flush=True)
    return summary, lower_endpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", choices=[SELECTOR], default=SELECTOR)
    parser.add_argument("--normal-blocks", type=int, choices=[NORMAL_BLOCKS], default=NORMAL_BLOCKS)
    parser.add_argument("--fft-size", type=int, choices=[FFT_SIZE], default=FFT_SIZE)
    parser.add_argument("--gauss-order", type=int, choices=[GAUSS_ORDER], default=GAUSS_ORDER)
    parser.add_argument("--digits", type=int, choices=[DIGITS], default=DIGITS)
    parser.add_argument("--root-bits", type=int, choices=[ROOT_BITS], default=ROOT_BITS)
    parser.add_argument("--newton-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    if os.environ.get("QI2_GPU_UUID") != args.gpu_uuid:
        raise RuntimeError("CLI GPU UUID must match the explicitly assigned remote device")
    newton, newton_upper = baseline.load_newton(args.newton_result)
    newton_lower = Fraction(newton["K3_positive_family_enclosure"]["lower"])
    if newton_lower <= Fraction(2, 3):
        raise ValueError("The accepted Newton lower bound must certify K3>2/3 for the W spacing superset")
    if (len(BETA_INTERVALS) != 16 or BETA_INTERVALS[0][0] != 0 or BETA_INTERVALS[-1][1] != Fraction(1, 10000)
            or any(first[1] != second[0] for first, second in zip(BETA_INTERVALS, BETA_INTERVALS[1:]))
            or EVALUATION_ORDER != (0, 15, *range(1, 15)) or set(EVALUATION_ORDER) != set(range(16))):
        raise ValueError("The fixed exact parameter partition and bounded evaluation order changed")

    # Host-only imports and input metadata are complete.  Numerical/backend
    # work follows the final in-process free query and immediate allocation,
    # then a matching-UUID sole-owned-PID check, all in this same job.
    execution = remote_runtime.remote_execution(max_threads=MAX_THREADS)
    start = time.perf_counter()
    iv.dps = DIGITS
    worker_contract = arithmetic.verify_binary64_contract()
    resource_check(execution)
    pi, log_two, _, _, _ = arithmetic.constant_enclosures()
    if arithmetic.endpoints(pi)[0] <= 3:
        raise ArithmeticError("The constant enclosure must certify pi>3")
    constants, tangent_data = baseline.tangent_constants(log_two, DIGITS)
    nodes, weights, gauss_data = arithmetic.gauss_enclosures(GAUSS_ORDER, ROOT_BITS)
    gauss_data["actual_root_brackets_and_weights"] = baseline.gauss_bracket_details(nodes, weights)
    gauss_data["seed_usage"] = "Untrusted numerical seeds; exact rational sign brackets exhaust the polynomial degree"
    c, u, twiddles, twiddle_error, root_data = baseline.root_table(FFT_SIZE)
    reversal = baseline.bit_reversal(FFT_SIZE)
    aliases = baseline.alias_allowances(FFT_SIZE, NORMAL_BLOCKS)
    panels = arithmetic.geometric_panels(Fraction(1, 10), Fraction(1), Fraction(7, 6))
    panels += arithmetic.geometric_panels(Fraction(1), Fraction(4), Fraction(7, 6))
    if (panels[0][0] != Fraction(1, 10) or panels[-1][1] != 4
            or any(first[1] != second[0] for first, second in zip(panels, panels[1:]))):
        raise ArithmeticError("Q panels must tile the exact accepted two-region interval")
    target = pi / 2 * arithmetic.interval(newton_upper)
    target_upper = arithmetic.endpoints(target)[1]
    physical_beta_upper = arithmetic.interval(Fraction(1, 2500)) / (2 * pi * arithmetic.interval(newton_lower))
    if arithmetic.endpoints(physical_beta_upper)[1] >= Fraction(1, 10000):
        raise ArithmeticError("The declared computational beta domain does not cover the reference spacing range")
    result = {
        "scope": "W1--W4 bounded original-sine planar spacing test; fixed delta=0.001 rectangle and unchanged continuum Newton match",
        "selector": SELECTOR, "execution": execution,
        "law": {
            "order": 6, "stencil": "original sine", "uniform_P": "1", "Omega": "m4=1",
            "m1_over_Omega": "1/1000",
            "mass_squared_over_Omega_squared": ["1/1000000", "1000001/2000000", "1000001/2000000", "1"],
            "packet_multiplicities": [1, 2, 1],
            "state": "Prescribed uniform polar sea; empty local number-conserving bosons",
            "spatial_algebra": "Ordinary original-site CAR; exact two-site normal cells and one axis-plane cut",
            "auxiliary_fourier_vector": "(v0-1,v1,v2,w); physical fourth component is sqrt(beta)*w",
            "first_input": "-z/[sqrt(1+z)*(1+sqrt(1+z))], z=T*S^5/F; never subtract rounded v0 from one",
            "moment_formula": "W1 exact beta-weighted trace, three-component antisymmetry and c-d; no extra conjugation on d",
            "beta_zero": "Continuous periodic-symbol and finite-compression endpoint only; not a Hardy unitary at alpha=0",
            "transverse_measure": "(2/pi) Q dQ; four transverse sine preimages; positive excess Jacobian discarded",
            "extra_valley_factor": 1, "second_interface_divisor": 1,
            "Newton_match": "One unchanged continuum match; no finite-BZ rematch, fit or new mass input",
        },
        "resolution": {"normal_blocks_L": NORMAL_BLOCKS, "fft_size_N": FFT_SIZE, "Gauss_order": GAUSS_ORDER,
                       "digits": DIGITS, "root_bits": ROOT_BITS, "worker_count": 1,
                       "automatic_refinement": False, "maximum_parameter_calculations": 16},
        "parameter_domain": {
            "beta_interval_exact": ["0", "1/10000"],
            "intervals_exact": [[baseline.exact_string(a), baseline.exact_string(b)] for a, b in BETA_INTERVALS],
            "evaluation_order": list(EVALUATION_ORDER), "first_last_feasibility_gate": [0, 15],
            "stop_rule": "First completed parameter interval with nonpositive strict excess ends as inconclusive; no further intervals or automatic subdivisions",
            "reference_a_over_lP_maximum": "1/50",
            "physical_beta_upper_from_accepted_Newton": baseline.reported(physical_beta_upper),
            "reference_range_covered_by_computational_superset": True,
            "interpolation_used": False,
        },
        "arithmetic": {
            "unchanged_baseline_arithmetic_source": "solve/code/uv_order6_hankel.py",
            "frozen_helper_source": "solve/code/uv_native_certificate.py: arithmetic only",
            "worker_FP_contract": worker_contract,
            "worker_checks": "One numerical worker checks FE_TONEAREST/ties/FTZ/DAZ/gradual underflow before work; nice/thread cap checked per node",
            "roots_and_twiddles": root_data, "Gauss_nodes_weights": gauss_data,
            "pi": baseline.reported(pi), "log2": baseline.reported(log_two),
            "FFT": "Accepted normalized radix-two real-primitive butterfly with /2 at each stage",
            "FFT_error": "H9 separate propagated input, twiddle and butterfly-roundoff allowances, including whole-beta input uncertainty",
            "complex_error_convention": "Absolute complex-modulus errors; sum of real and imaginary radii",
            "moment_arithmetic": "New W1 4x4 auxiliary blocks; every beta product/sliding update/outward pairwise reduction enclosed",
            "outer_aggregation": "Sequential outward mpmath.iv sums; exact endpoint comparisons",
            "serialization": "Outward decimal endpoints; exact parameter, root and beta-input bracket endpoints",
        },
        "auxiliary_Fourier_bounds": {
            "real_input_absolute_bound": "1 for each auxiliary input",
            "w_strip_bound": "8sqrt(40)*10^6<10^12 by W's uniform root bound",
            "H8_sigma": "1/250", "H8_envelope_M": "1000000000000", "rational_base": "3/8",
            "allowances": [{"k": k, "absolute_component_error_upper": baseline.upper_report(error),
                            "absolute_component_error_exact_upper": baseline.exact_string(error)}
                           for k, error in enumerate(aliases, 1)],
        },
        "entropy_tangents": tangent_data, "newton_input": newton,
        "comparison": {"common_target_enclosure_pi_over_2_K3_upper": baseline.reported(target),
                       "common_target_outward_upper": baseline.upper_report(target_upper)},
        "completed_parameter_interval_indices": [], "interval_results": [],
        "arithmetic_complete": False, "completed_coverage": False, "strict_uniform_excess_pass": False,
        "entropy_normalization_disposition": "Bounded contract incomplete; no uniform spacing comparison credited",
        "caveat": "A failed lower-bound test is inconclusive, including failure in the computational superset outside the physical reference range. It proves no entropy upper bound or normalization success. No other mass, stencil, sphere, physical/interacting/H_* state, T9 gravity response or thermodynamic transfer is certified.",
    }
    save_json(args.output_dir, "", result)
    context = {"pi": pi, "constants": constants, "nodes": nodes, "weights": weights, "panels": panels,
               "c": c, "u": u, "reversal": reversal, "twiddles": twiddles, "twiddle_error": twiddle_error,
               "aliases": aliases, "target_upper": target_upper}
    completed_lowers = []
    for index in EVALUATION_ORDER:
        summary, lower_endpoint = run_parameter_interval(index, args.output_dir, execution, context)
        result["interval_results"].append(summary)
        result["completed_parameter_interval_indices"].append(index)
        completed_lowers.append(lower_endpoint)
        save_json(args.output_dir, "", result)
        if not summary["strict_interval_excess_pass"]:
            result["stopped_at_first_nonpositive_excess_parameter_index"] = index
            break
    resource_check(execution)
    result["arithmetic_complete"] = True
    result["arithmetic_complete_means"] = "Every attempted parameter calculation finished all declared enclosures and the finite contract reached its prescribed exit; this is separate from full parameter coverage"
    result["completed_coverage"] = set(result["completed_parameter_interval_indices"]) == set(range(16))
    strict_uniform = result["completed_coverage"] and all(item["strict_interval_excess_pass"]
                                                         for item in result["interval_results"])
    result["strict_uniform_excess_pass"] = strict_uniform
    minimum_lower = min(completed_lowers)
    result["comparison"].update({
        "minimum_certified_lower_over_completed_intervals": baseline.lower_report(minimum_lower),
        "minimum_margin_over_completed_intervals": baseline.lower_report(minimum_lower - target_upper),
        "all_sixteen_intervals_required_for_W4": True,
    })
    result["entropy_normalization_disposition"] = (
        "Strict ordinary native axis-plane lower-bound excess over every positive reference spacing for these fixed masses"
        if strict_uniform else "Inconclusive bounded lower-bound test; no uniform-spacing rejection or normalization success")
    result["elapsed_seconds"] = time.perf_counter() - start
    save_json(args.output_dir, "", result)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
