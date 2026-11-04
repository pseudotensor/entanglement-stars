#!/usr/bin/env python3
"""T10 order-six Newton-only interval enclosure from reviewed T28--30.

Runtime, tests and numerical postprocessing run only on the remote qi2
host.  Reuses frozen arithmetic helpers from uv_native_certificate.py,
not its order-four physics integrands, error bounds or numerical results.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from fractions import Fraction
from pathlib import Path

from mpmath import iv

import uv_native_certificate as arithmetic


def radial_integrand(point):
    light = point * (1 + 12 * point ** 10) / arithmetic.certified_iv_sqrt(1 + point ** 10)
    grouped = light
    for mass_squared, multiplicity in ((Fraction(1, 2), -2), (Fraction(1), 1)):
        real_squared = arithmetic.interval(mass_squared) + point ** 12
        massive = point ** 2 * (1 + 12 * point ** 22 / real_squared)
        massive /= arithmetic.certified_iv_sqrt(real_squared + point ** 2)
        grouped += multiplicity * massive
    return 12 / PI ** 2 * grouped


def ellipse_bound(lower: Fraction, upper: Fraction):
    maximum = arithmetic.interval(lower + Fraction(9, 8) * (upper - lower))
    if lower == 0:
        light = maximum * (1 + 12 * maximum ** 10)
        light /= arithmetic.certified_iv_sqrt(1 - maximum ** 10)
        massive = maximum ** 2 * (1 + 12 * maximum ** 22
                                  / (arithmetic.interval(Fraction(1, 2)) - maximum ** 12))
        massive /= arithmetic.certified_iv_sqrt(arithmetic.interval(Fraction(1, 2))
                                                - maximum ** 2 - maximum ** 12)
        return 12 / PI ** 2 * (light + 3 * massive)
    return 4 * 12 / PI ** 2 * SQRT2 * maximum * (1 + 24 * maximum ** 10)


def resource_contract():
    niceness = os.getpriority(os.PRIO_PROCESS, 0)
    threads = len(os.listdir("/proc/self/task"))
    if niceness < 10 or threads > 16:
        raise RuntimeError("Remote numerical job requires nice>=10 and <=16 total threads")
    return {"niceness": niceness, "total_threads_checked": threads, "thread_limit": 16}


def newton_certificate(nodes: list, weights: list, order: int):
    panels = [(Fraction(0), Fraction(1, 10))]
    panels += arithmetic.geometric_panels(Fraction(1, 10), Fraction(4), Fraction(7, 6))
    quadrature, analytic_error = arithmetic.interval(0), arithmetic.interval(0)
    for lower, upper in panels:
        resource_contract()
        points, measures = arithmetic.mapped_gauss(lower, upper, nodes, weights)
        quadrature += sum((w * radial_integrand(p) for p, w in zip(points, measures)), arithmetic.interval(0))
        analytic_error += 8 * arithmetic.interval(upper - lower) * ellipse_bound(lower, upper) / (1 << (2 * order))
    finite_lower = arithmetic.endpoints(quadrature)[0] - arithmetic.endpoints(analytic_error)[1]
    finite_upper = arithmetic.endpoints(quadrature)[1] + arithmetic.endpoints(analytic_error)[1]
    limit = arithmetic.interval(4)
    tail = 12 / PI ** 2 * (arithmetic.interval(Fraction(1, 144)) * limit ** -27
                           + arithmetic.interval(Fraction(45, 68)) * limit ** -17)
    massless = arithmetic.hull(finite_lower, finite_upper + arithmetic.endpoints(tail)[1])
    delta = arithmetic.interval(Fraction(1, 1000))
    mass_loss = 12 / PI ** 2 * delta ** 2 * ((1 + 12 * delta ** 10) / 2
                                           + arithmetic.interval(Fraction(37, 2))
                                           * (LOG1000 + arithmetic.interval(Fraction(1, 5))))
    family_lower = arithmetic.endpoints(massless)[0] - arithmetic.endpoints(mass_loss)[1]
    family = arithmetic.hull(family_lower, arithmetic.endpoints(massless)[1])
    threshold = arithmetic.interval(Fraction(48, 25)) ** 2 / (2 * PI)
    return {
        "scope": "Order-six continuum full-source Newton coefficient K3=kOmega/Omega^2, Omega=m4=1",
        "massless_rectangle": ["0", "0.5", "0.5", "1"],
        "positive_family_mass_squared": ["delta^2", "(1+delta^2)/2", "(1+delta^2)/2", "1"],
        "delta_interval": "0<delta<=0.001",
        "full_source_integrand": "12/pi^2 p^2 sum_eta (1+12p^22/(x+p^12))/sqrt(x+p^2+p^12)",
        "finite_interval": ["0", "4"], "Gauss_order_per_panel": order,
        "panels": len(panels), "Bernstein_ellipse_parameter": "2",
        "panel_tiling": "Exact [0,1/10] and ratio<=7/6 positive rational panels through4",
        "quadrature_arithmetic_enclosure": arithmetic.enclosure(quadrature),
        "analytic_Gauss_absolute_error_upper_T28": arithmetic.enclosure(analytic_error)["upper"],
        "positive_radial_tail_upper_T29": arithmetic.enclosure(tail)["upper"],
        "K3_massless_enclosure": arithmetic.enclosure(massless),
        "positive_mass_loss_upper_T30": arithmetic.enclosure(mass_loss)["upper"],
        "K3_positive_family_enclosure": arithmetic.enclosure(family),
        "native_spacing_applicability_threshold": arithmetic.enclosure(threshold),
        "native_spacing_applicability_pass": family_lower > arithmetic.endpoints(threshold)[1],
        "status": "Executed Newton enclosure; independent source/result review required before adoption",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--order", type=int, default=32)
    parser.add_argument("--digits", type=int, default=80)
    parser.add_argument("--root-bits", type=int, default=120)
    args = parser.parse_args()
    if args.order < 32 or args.digits < 60 or args.root_bits < 100:
        raise ValueError("Use at least Gauss32, interval60digits and root100bits")
    execution = {"single_process_pid": os.getpid()}
    execution.update(arithmetic.hold_cuda_slot(args.gpu_uuid))
    execution.update(resource_contract())
    iv.dps = args.digits
    global PI, LOG2, SQRT2, LOG40, LOG1000
    PI, LOG2, SQRT2, LOG40, LOG1000 = arithmetic.constant_enclosures()
    start = time.perf_counter()
    print("Order-six Newton: certifying exact Gauss brackets and interval weights", flush=True)
    nodes, weights, gauss = arithmetic.gauss_enclosures(args.order, args.root_bits)
    print("Order-six Newton: enclosing finite integral, fresh Gauss error, tail and mass allowance", flush=True)
    newton = newton_certificate(nodes, weights, args.order)
    result = {
        "scope": "Reviewed order-six Newton-only T28--30 numerical execution",
        "execution": execution,
        "arithmetic": {"helper_source": "solve/code/uv_native_certificate.py, frozen arithmetic helpers only",
                       "library": "mpmath.iv", "decimal_precision": args.digits,
                       "sqrt": "Every endpoint verified by exact rational square comparisons",
                       "constants": "Exact rational Machin/atanh bounds and integer-square sqrt2 brackets",
                       "Gauss_nodes_weights": gauss,
                       "aggregation": "Sequential outward mpmath.iv scalar additions",
                       "serialization": "Exact dyadic endpoints rounded outward to decimal strings"},
        "newton": newton,
        "newton_complete": True, "entropy_complete": False,
        "elapsed_seconds": time.perf_counter() - start,
        "caveat": "No order-six entropy, finite-spacing gravity, stellar, spherical-cut or thermodynamic result is implied by this Newton enclosure.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.dumps(result, indent=2, allow_nan=False) + "\n"
    (args.output_dir / "uv_order6_newton.json").write_text(summary)
    (args.output_dir / "uv_order6_newton.txt").write_text(summary)
    print(summary, end="", flush=True)


if __name__ == "__main__":
    main()
