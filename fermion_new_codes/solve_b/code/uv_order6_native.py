#!/usr/bin/env python3
"""Reviewed order-six T31--35 compact native interval calculation.

Remote qi2 execution only.  Frozen r2 helpers supply arithmetic, not
physics bounds or numbers.  All r3 domains, coefficients and errors are
those of the independently reviewed order-six specification.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from threading import Lock

import numpy as np
from mpmath import iv

import uv_native_certificate as arithmetic
import remote_runtime

BI = arithmetic.BinaryInterval
MAX_THREADS = 8


def kernel_parts(p_sum, effective_q, d_r_factor, d_e_factor, e_q, r_q, denominator, transverse):
    d_r, d_e = p_sum * d_r_factor, p_sum * d_e_factor
    real = (d_r * e_q - r_q * d_e) / denominator
    normal = (e_q - effective_q * d_e) / denominator
    tangential = -(transverse * d_e) / denominator
    return real.square() + normal.square() + tangential.square()


def kernel_binary(p, q, transverse, mass):
    t_p, t_q = p.square() + transverse.square(), q.square() + transverse.square()
    r_p_squared, r_q_squared = mass + t_p ** 6, mass + t_q ** 6
    r_p, r_q = r_p_squared.sqrt(), r_q_squared.sqrt()
    e_p, e_q = (r_p_squared + t_p).sqrt(), (r_q_squared + t_q).sqrt()
    p_five = (t_p ** 3 + t_q ** 3) * (t_p.square() + t_p * t_q + t_q.square())
    return kernel_parts(p + q, q, p_five / (r_p + r_q), (1 + p_five) / (e_p + e_q),
                        e_q, r_q, e_p * e_q, transverse)


def kernel_reference(p, q, transverse, mass):
    t_p, t_q = p ** 2 + transverse ** 2, q ** 2 + transverse ** 2
    r_p = arithmetic.certified_iv_sqrt(mass + t_p ** 6)
    r_q = arithmetic.certified_iv_sqrt(mass + t_q ** 6)
    e_p = arithmetic.certified_iv_sqrt(mass + t_p + t_p ** 6)
    e_q = arithmetic.certified_iv_sqrt(mass + t_q + t_q ** 6)
    p_five = (t_p ** 3 + t_q ** 3) * (t_p ** 2 + t_p * t_q + t_q ** 2)
    d_r, d_e = (p + q) * p_five / (r_p + r_q), (p + q) * (1 + p_five) / (e_p + e_q)
    denominator = e_p * e_q
    components = ((d_r * e_q - r_q * d_e) / denominator,
                  (e_q - q * d_e) / denominator, -transverse * d_e / denominator)
    return sum((component ** 2 for component in components), arithmetic.interval(0))


def verify_worker():
    if len(os.listdir("/proc/self/task")) > MAX_THREADS:
        raise RuntimeError(f"Dense process exceeds {MAX_THREADS} total threads")
    arithmetic.verify_binary64_contract()
    cases = ((Fraction(0), Fraction(0), Fraction(1, 10), Fraction(1, 10 ** 6)),
             (Fraction(1, 10), Fraction(1, 10), Fraction(1, 10), Fraction(1, 10 ** 6)),
             (Fraction(1), Fraction(1), Fraction(1), Fraction(1, 10 ** 6)),
             (Fraction(96), Fraction(96), Fraction(4), Fraction(1)),
             (Fraction(96), Fraction(-96), Fraction(1, 10), Fraction(1, 2)),
             (Fraction(2), Fraction(-1, 10), Fraction(1, 10), Fraction(1, 10 ** 6)))
    for values in cases:
        calculated = kernel_binary(*(BI.rational(value) for value in values)).to_iv()
        reference = kernel_reference(*(arithmetic.interval(value) for value in values))
        left, right = arithmetic.endpoints(calculated)
        ref_left, ref_right = arithmetic.endpoints(reference)
        if left > ref_left or right < ref_right:
            raise ArithmeticError("Order-six interval kernel failed reference containment")
    return {"FE_TONEAREST_ties_to_even_gradual_underflow_FTZ_DAZ_off": True,
            "NumPy_over_invalid_divide_raise_underflow_ignore": True,
            "r3_kernel_reference_cases_enclosed": len(cases)}


def normal_rule(nodes, weights):
    half_center = Fraction(1, 160)
    center_points, center_weights = arithmetic.mapped_gauss(-half_center, half_center, nodes, weights)
    points, measures = center_points[len(nodes) // 2:], center_weights[len(nodes) // 2:]
    panels = arithmetic.geometric_panels(half_center, Fraction(96), Fraction(7, 6))
    for lower, upper in panels:
        panel_points, panel_weights = arithmetic.mapped_gauss(lower, upper, nodes, weights)
        points += panel_points
        measures += panel_weights
    if any(arithmetic.endpoints(point)[0] <= 0 for point in points):
        raise ArithmeticError("Positive normal abscissa enclosure crosses zero")
    return points, measures, 1 + 2 * len(panels)


def packet_tensor(nodes, weights, workers):
    p_nodes, p_weights, normal_panels = normal_rule(nodes, weights)
    q_panels = arithmetic.geometric_panels(Fraction(1, 10), Fraction(1), Fraction(7, 6))
    q_panels += arithmetic.geometric_panels(Fraction(1), Fraction(4), Fraction(7, 6))
    tasks = []
    for lower, upper in q_panels:
        transverse_points, transverse_weights = arithmetic.mapped_gauss(lower, upper, nodes, weights)
        for point, weight in zip(transverse_points, transverse_weights):
            tasks.append((BI.from_iv(point), BI.from_iv(weight), lower >= 1))
    p_vector, w_vector = BI.vector(p_nodes), BI.vector(p_weights)
    row, column = np.triu_indices(len(p_nodes))
    pair_p, pair_q = p_vector[row], p_vector[column]
    p_sum, p_difference = pair_p + pair_q, pair_p - pair_q
    pair_measure = w_vector[row] * w_vector[column] * BI(np.where(row == column, 2.0, 4.0))
    p_squared = p_vector.square()
    prefactor = BI.from_iv(1 / (2 * PI ** 3))
    masses = [Fraction(1, 10 ** 6), Fraction(1000001, 2000000), Fraction(1)]
    worker_checks, worker_lock = [], Lock()

    def initializer():
        check = verify_worker()
        with worker_lock:
            worker_checks.append(check)

    def node_integrals(task):
        if len(os.listdir("/proc/self/task")) > MAX_THREADS:
            raise RuntimeError(f"Dense process exceeds {MAX_THREADS} total threads")
        transverse, measure, high_region = task
        # These common expressions depend on Q and p, not on packet mass.
        # Calculate roots on the normal vector before triangle indexing;
        # this is the same reviewed formula with repeated expressions reused.
        t_vector = p_squared + transverse.square()
        t_two, t_three = t_vector.square(), t_vector ** 3
        t_six = t_three.square()
        t_p, t_q = t_vector[row], t_vector[column]
        p_five = (t_three[row] + t_three[column]) * (t_two[row] + t_p * t_q + t_two[column])
        radial_measure = prefactor * transverse * measure
        integrals = []
        for mass in masses:
            real_squared = BI.rational(mass) + t_six
            real_vector = real_squared.sqrt()
            energy_vector = (real_squared + t_vector).sqrt()
            r_p, r_q = real_vector[row], real_vector[column]
            e_p, e_q = energy_vector[row], energy_vector[column]
            d_r_factor = p_five / (r_p + r_q)
            d_e_factor = (1 + p_five) / (e_p + e_q)
            denominator = e_p * e_q
            combined = kernel_parts(p_sum, pair_q, d_r_factor, d_e_factor, e_q, r_q, denominator, transverse)
            combined = combined + kernel_parts(p_difference, -pair_q, d_r_factor, d_e_factor,
                                               e_q, r_q, denominator, transverse)
            integrals.append((radial_measure * (pair_measure * combined).sum()).to_iv())
        return integrals, high_region

    totals = [arithmetic.interval(0) for _ in masses]
    high_totals = [arithmetic.interval(0) for _ in masses]
    print(f"Order-six compact: {normal_panels} normal panels, {len(p_nodes)} positive nodes, "
          f"{len(tasks)} transverse nodes, {workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=workers, initializer=initializer) as executor:
        for number, (values, high_region) in enumerate(executor.map(node_integrals, tasks), 1):
            for index, value in enumerate(values):
                totals[index] += value
                if high_region:
                    high_totals[index] += value
            if number % len(nodes) == 0 or number == len(tasks):
                print(f"Order-six compact enclosed {number}/{len(tasks)} transverse nodes", flush=True)
    if len(worker_checks) != workers:
        raise ArithmeticError("Not every dense worker validated its floating-point contract")
    return totals, high_totals, {
        "normal_full_panels": normal_panels, "normal_positive_nodes": len(p_nodes),
        "normal_triangle_pairs": len(row), "transverse_panels": len(q_panels),
        "transverse_nodes": len(tasks), "worker_count_checked": len(worker_checks),
        "worker_contract": worker_checks[0], "exact_rational_panel_tiling_checked": True,
        "symmetry": "4 off-diagonal and2 diagonal; simultaneous sign reversal and argument exchange",
    }


def enclose_compact_integral(quadrature, error):
    lower = max(Fraction(0), arithmetic.endpoints(quadrature)[0] - arithmetic.endpoints(error)[1])
    upper = arithmetic.endpoints(quadrature)[1] + arithmetic.endpoints(error)[1]
    return arithmetic.hull(lower, upper)


def native_certificate(nodes, weights, order, workers, newton):
    totals, high_totals, layout = packet_tensor(nodes, weights, workers)
    entropy_base, entropy_extra, entropy_max = 4 * LOG2, (4 * LOG2 - 2) / 2, 6 * LOG2 - 1
    total_sum = totals[0] + 2 * totals[1] + totals[2]
    high_sum = high_totals[0] + 2 * high_totals[1] + high_totals[2]
    cubature = entropy_base * total_sum + entropy_extra * high_sum
    q_min, q_max, normal_limit = arithmetic.interval(Fraction(1, 10)), arithmetic.interval(4), arithmetic.interval(96)
    disk = q_min / 1000
    single_packet_envelope = 1 / (2 * PI ** 3) * (4 * q_max / 3) * (12 / disk) ** 2
    envelope = 4 * entropy_max * single_packet_envelope
    tensor_factor = 12 * (2 * normal_limit) ** 2 / (3 ** (2 * order)) / arithmetic.interval(Fraction(2, 3))
    total_error = tensor_factor * (q_max - q_min) * envelope
    packet_total_error = tensor_factor * (q_max - q_min) * single_packet_envelope
    packet_high_error = tensor_factor * (q_max - 1) * single_packet_envelope
    compact_lower = max(Fraction(0), arithmetic.endpoints(cubature)[0] - arithmetic.endpoints(total_error)[1])
    mass_difference = arithmetic.interval(Fraction(1, 10 ** 6)) + arithmetic.interval(Fraction(1, 4 * 10 ** 12))
    sqrt_ten = arithmetic.certified_iv_sqrt(arithmetic.interval(10))
    stability = 2 / PI * entropy_max * sqrt_ten / (2 * SQRT2) * LOG40 * mass_difference
    root = arithmetic.certified_iv_sqrt(arithmetic.interval(compact_lower))
    root -= arithmetic.certified_iv_sqrt(arithmetic.interval(arithmetic.endpoints(stability)[1]))
    uniform = arithmetic.interval(0) if arithmetic.endpoints(root)[0] <= 0 else root ** 2
    k_lower = Fraction(newton["K3_massless_enclosure"]["lower"])
    k_upper = Fraction(newton["K3_massless_enclosure"]["upper"])
    loss_upper = Fraction(newton["positive_mass_loss_upper_T30"])
    threshold = arithmetic.interval(Fraction(48, 25)) ** 2 / (2 * PI)
    target = PI / 2 * arithmetic.interval(k_upper)
    margin = uniform - arithmetic.interval(arithmetic.endpoints(target)[1])
    packets = []
    for packet, unique in enumerate([0, 1, 1, 2], 1):
        packets.append({"packet": packet,
                        "mass_squared": ["0.000001", "0.5000005", "0.5000005", "1"][packet - 1],
                        "J_total_cubature": arithmetic.enclosure(totals[unique]),
                        "J_high_cubature": arithmetic.enclosure(high_totals[unique]),
                        "J_total_compact_integral_enclosure": arithmetic.enclosure(enclose_compact_integral(totals[unique], packet_total_error)),
                        "J_high_compact_integral_enclosure": arithmetic.enclosure(enclose_compact_integral(high_totals[unique], packet_high_error))})
    return {
        "scope": "Order-six prescribed uniform planar native cut at P=1; preserved symmetric positive mass family",
        "domains": {"Q": ["0.1", "4"], "Q_split": "1", "p_q": ["-96", "96"],
                    "central_normal": ["-0.00625", "0.00625"], "panel_ratio": "7/6"},
        "Gauss_order": order, "Bernstein_ellipse": "3", "layout": layout,
        "packet_integrals": packets,
        "packet_total_analytic_error_upper": arithmetic.enclosure(packet_total_error)["upper"],
        "packet_high_analytic_error_upper": arithmetic.enclosure(packet_high_error)["upper"],
        "entropy_weight": "4ln2 plus(4ln2-2)/2 onQ>=1, using reviewed accretive compression",
        "cubature_arithmetic_enclosure": arithmetic.enclosure(cubature),
        "new_tensor_envelope": arithmetic.enclosure(envelope),
        "analytic_tensor_absolute_error_upper_T34": arithmetic.enclosure(total_error)["upper"],
        "zero_clipped_compact_lower": arithmetic.decimal_endpoint(compact_lower, 24, False),
        "new_mass_stability_D3_upper_T33": arithmetic.enclosure(stability)["upper"],
        "uniform_native_lower_bound": arithmetic.enclosure(uniform),
        "target_upper_from_fresh_K3": arithmetic.enclosure(target)["upper"],
        "uniform_margin": arithmetic.enclosure(margin),
        "uniform_ratio_lower_bound": arithmetic.enclosure(uniform / arithmetic.interval(arithmetic.endpoints(target)[1])),
        "spacing_applicability_pass_T35": k_lower - loss_upper > arithmetic.endpoints(threshold)[1],
        "strict_entropy_excess_pass_T35": arithmetic.endpoints(margin)[0] > 0,
        "status": "Executed arithmetic; independent source/result inspection required",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--newton-result", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--order", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--digits", type=int, default=80)
    parser.add_argument("--root-bits", type=int, default=120)
    args = parser.parse_args()
    if args.order < 24 or args.order % 2 or not 1 <= args.workers <= 4 or args.digits < 60 or args.root_bits < 100:
        raise ValueError("Require even order>=24,1..4workers, interval>=60digits and roots>=100bits")
    saved_newton = json.loads(args.newton_result.read_text())
    if not saved_newton.get("newton_complete"):
        raise ValueError("A completed fresh order-six Newton enclosure is required")
    newton = saved_newton["newton"]
    if newton["delta_interval"] != "0<delta<=0.001" or newton["massless_rectangle"] != ["0", "0.5", "0.5", "1"]:
        raise ValueError("Newton mass family does not match the released entropy specification")
    if os.environ.get("QI2_GPU_UUID") != args.gpu_uuid:
        raise RuntimeError("The assigned CLI GPU UUID must match the remote execution guard")
    # All host-only imports and input validation are complete.  The shared
    # guard queries ownership in this process immediately before allocating
    # its visible CUDA context, then requires this PID to be the sole owner.
    # Interval/backend arithmetic starts only after both checks pass.
    execution = remote_runtime.remote_execution(max_threads=MAX_THREADS)
    iv.dps = args.digits
    global PI, LOG2, SQRT2, LOG40, LOG1000
    PI, LOG2, SQRT2, LOG40, LOG1000 = arithmetic.constant_enclosures()
    start = time.perf_counter()
    verify_worker()
    nodes, weights, gauss = arithmetic.gauss_enclosures(args.order, args.root_bits)
    entropy = native_certificate(nodes, weights, args.order, args.workers, newton)
    result = {"scope": "Reviewed order-six native compact arithmetic, T31--35",
              "execution": execution, "newton_reference": newton,
              "arithmetic": {"helper": "Frozen uv_native_certificate.py arithmetic only",
                             "decimal_precision": args.digits, "Gauss_nodes_weights": gauss,
                             "constants": "Exact rational recipes and verified sqrt endpoints",
                             "dense_reduction": "Outward binary64 pairwise additions",
                             "outer_aggregation": "Sequential outward mpmath.iv additions"},
              "native_entropy": entropy, "arithmetic_complete": True, "complete": True,
              "complete_means": "All declared arithmetic enclosures executed; entropy normalization is not marked solved",
              "entropy_normalization_disposition": (
                  "strict lower-bound excess established in the stated native window"
                  if entropy["spacing_applicability_pass_T35"] and entropy["strict_entropy_excess_pass_T35"]
                  else "inconclusive: this lower-bound calculation has not settled entropy normalization"),
              "elapsed_seconds": time.perf_counter() - start,
              "caveat": "A failed lower-bound rejection test is inconclusive and does not prove entropy below target. Stellar, spherical, thermal and finite-spacing Newton-rematch claims remain unproved."}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.dumps(result, indent=2, allow_nan=False) + "\n"
    (args.output_dir / "uv_order6_native.json").write_text(summary)
    (args.output_dir / "uv_order6_native.txt").write_text(summary)
    print(summary, end="", flush=True)


if __name__ == "__main__":
    main()
