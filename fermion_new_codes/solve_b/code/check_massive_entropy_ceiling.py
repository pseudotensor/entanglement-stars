#!/usr/bin/env python3
"""Two fixed conditional smooth-volume R2C14a checks; run on the qi2 box.

This is a scalar massive Gaussian bound, separate from the massless D5 pilot.
The root brackets/sandwich use ordinary floating point and are not certified
interval enclosures. An independent 50-digit bounded-objective check avoids
double-precision ties close to the maximum.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import resource
import socket
import time

START = time.monotonic()
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import mpmath as mp

mp.mp.dps = 50
COEFF = 16.0 * math.pi / 3.0
COUNT_SLOPE = 10.0


def budget() -> None:
    if time.monotonic() - START > 60:
        raise RuntimeError("60-second wall budget exceeded")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 256 * 1024:
        raise RuntimeError("256-MiB RSS budget exceeded")


def h2(p: float) -> float:
    if p == 0.0 or p == 1.0:
        return 0.0
    return -p * math.log(p) - (1.0 - p) * math.log1p(-p)


def objective(x: float, initial_radius: float) -> float:
    if x == 0.0 or x == initial_radius:
        return 0.0
    modes = COEFF * x ** 3
    count = COUNT_SLOPE * (initial_radius - x)
    return modes * h2(min(count / modes, 0.5))


def derivative(x: float, initial_radius: float) -> tuple[float, float, float]:
    if x == initial_radius:
        return -math.inf, 0.0, math.inf
    y = COUNT_SLOPE * (initial_radius - x) / (COEFF * x ** 3)
    positive = 3.0 * COEFF * x * x * (-math.log1p(-y))
    negative = COUNT_SLOPE * (math.log1p(-y) - math.log(y))
    return positive - negative, positive, negative


def bisect_join(initial_radius: float) -> tuple[float, float, int, str]:
    lower, upper = 0.0, initial_radius
    tolerance = 1e-12 * initial_radius
    reason = "iteration_limit"
    for iteration in range(1, 81):
        budget()
        middle = (lower + upper) / 2.0
        if upper - lower <= tolerance:
            reason = "requested_width"
            break
        if middle == lower or middle == upper:
            reason = "floating_point_stagnation"
            break
        residual = COEFF * middle ** 3 + 2 * COUNT_SLOPE * (middle - initial_radius)
        if residual >= 0:
            upper = middle
        else:
            lower = middle
    return lower, upper, iteration, reason


def bisect_stationary(join_upper: float, initial_radius: float
                      ) -> tuple[float, float, int, str]:
    lower, upper = join_upper, initial_radius
    tolerance = 1e-12 * initial_radius
    reason = "iteration_limit"
    for iteration in range(1, 81):
        budget()
        middle = (lower + upper) / 2.0
        if upper - lower <= tolerance:
            reason = "requested_width"
            break
        if middle == lower or middle == upper:
            reason = "floating_point_stagnation"
            break
        if derivative(middle, initial_radius)[0] > 0:
            lower = middle
        else:
            upper = middle
    return lower, upper, iteration, reason


def bounded_objective_check(join_upper: float, initial_radius: float) -> dict:
    """Independent objective-only golden-section maximization at fixed precision."""
    lower, upper = mp.mpf(join_upper), mp.mpf(initial_radius)
    initial = mp.mpf(initial_radius)
    coefficient = 16 * mp.pi / 3
    ratio = (mp.sqrt(5) - 1) / 2
    tolerance = mp.mpf("1e-10") * initial

    def value(x):
        modes = coefficient * x ** 3
        count = 10 * (initial - x)
        y = min(count / modes, mp.mpf("0.5"))
        return modes * (-y * mp.log(y) - (1 - y) * mp.log1p(-y))

    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    value_left, value_right = value(left), value(right)
    iterations = 0
    while upper - lower > tolerance:
        budget()
        iterations += 1
        if iterations > 80:
            raise RuntimeError("bounded-objective iteration cap exceeded")
        if value_left < value_right:
            lower = left
            left, value_left = right, value_right
            right = lower + ratio * (upper - lower)
            value_right = value(right)
        else:
            upper = right
            right, value_right = left, value_left
            left = upper - ratio * (upper - lower)
            value_left = value(left)
    middle = (lower + upper) / 2
    return {"precision_digits": 50, "iterations": iterations,
            "bracket": [str(lower), str(upper)],
            "bracket_width": str(upper - lower),
            "requested_radius_tolerance": str(tolerance),
            "radius": str(middle), "entropy": str(value(middle))}


def evaluate(initial_radius: float) -> dict:
    join_l, join_u, join_iterations, join_stop = bisect_join(initial_radius)
    # Upper joining bracket lies on the unsaturated side, within the prescribed
    # joining-radius tolerance. This avoids an artificial y>1/2 start by rounding.
    lower, upper, iterations, stop = bisect_stationary(join_u, initial_radius)
    optimum = (lower + upper) / 2
    count = COUNT_SLOPE * (initial_radius - optimum)
    volume = 4.0 * math.pi * optimum ** 3 / 3.0
    modes = 4.0 * volume
    y = count / modes
    residual, term_positive, term_negative = derivative(optimum, initial_radius)
    entropy = objective(optimum, initial_radius)
    sandwich_lower = max(objective(x, initial_radius) for x in (lower, optimum, upper))
    modes_upper = COEFF * upper ** 3
    count_lower_radius = COUNT_SLOPE * (initial_radius - lower)
    sandwich_upper = modes_upper * h2(min(count_lower_radius / modes_upper, 0.5))
    crosscheck = bounded_objective_check(join_u, initial_radius)
    initial_volume = 4.0 * math.pi * initial_radius ** 3 / 3.0
    initial_modes = 4 * initial_volume
    initial_count = COUNT_SLOPE * initial_radius
    return {
        "initial_radius": initial_radius,
        "joining_radius": (join_l + join_u) / 2,
        "joining_bracket": [join_l, join_u],
        "joining_bracket_width": join_u - join_l,
        "joining_iterations": join_iterations, "joining_stop": join_stop,
        "joining_equation_residual": COEFF * ((join_l + join_u) / 2) ** 3
                                     + 2 * COUNT_SLOPE * ((join_l + join_u) / 2 - initial_radius),
        "stationary_bracket": [lower, upper], "stationary_bracket_width": upper - lower,
        "stationary_iterations": iterations, "stationary_stop": stop,
        "stationary_endpoint_signs": [derivative(lower, initial_radius)[0],
                                      derivative(upper, initial_radius)[0]],
        "optimum_radius": optimum, "occupation_fraction": y,
        "excitation_count": count, "volume_sites_smooth": volume, "retained_modes": modes,
        "max_entropy_nats": entropy, "endpoint_entropies": [0.0, 0.0],
        "joining_entropy_nats": objective((join_l + join_u) / 2, initial_radius),
        "normalized_stationary_residual": residual / (term_positive + term_negative),
        "observed_sandwich": [sandwich_lower, sandwich_upper],
        "observed_sandwich_width": sandwich_upper - sandwich_lower,
        "bounded_objective_crosscheck": crosscheck,
        "crosscheck_entropy_difference": float(mp.mpf(crosscheck["entropy"]) - entropy),
        "crosscheck_radius_difference": float(mp.mpf(crosscheck["radius"]) - optimum),
        "initial_volume_comparison": {
            "volume_sites_smooth": initial_volume, "retained_modes": initial_modes,
            "excitation_count": initial_count,
            "entropy_nats": initial_modes * h2(min(initial_count / initial_modes, 0.5))},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = {"control": "R2C14a conditional massive smooth-volume ceiling",
              "producer": Path(__file__).name, "host": socket.gethostname(),
              "python": platform.python_version(), "mpmath": mp.__version__,
              "nice": os.getpriority(os.PRIO_PROCESS, 0),
              "threads": {key: os.environ[key] for key in
                          ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
              "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
              "g": 0.25, "mu": 0.2, "count_slope": COUNT_SLOPE,
              "root_relative_width": 1e-12, "root_max_iterations": 80,
              "crosscheck_relative_radius_tolerance": 1e-10,
              "wall_limit_seconds": 60, "rss_limit_kib": 256 * 1024,
              "scope": "Conditional R2C14 Gaussian/purifier/closed-energy premises; "
                       "smooth volumes only. Observed floating-point sandwich, no "
                       "outward-rounded certificate. Not a massless Page run or state count.",
              "complete": False, "rows": []}
    try:
        for initial_radius in (32.0, 64.0):
            result["rows"].append(evaluate(initial_radius))
        result["complete"] = True
    except Exception as error:
        result["error"] = type(error).__name__ + ": " + str(error)
    result["elapsed_seconds"] = time.monotonic() - START
    result["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({key: result[key] for key in
                      ("complete", "elapsed_seconds", "max_rss_kib")}), flush=True)


if __name__ == "__main__":
    main()
