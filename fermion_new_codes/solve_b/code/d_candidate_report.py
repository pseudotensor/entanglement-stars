"""Report provisional Newton geometry from a saved original-A D center.

This performs no matrix assembly, eigensolve, response solve, or box proof.
Run only on the authorized remote host under the same CUDA admission guard.
"""

from __future__ import annotations

import argparse
from decimal import localcontext, ROUND_FLOOR
import json
import os
import sys
from pathlib import Path

import numpy as np
import cupy

import d_verified as verified
from d_enclosure import (newton_inequalities, rectangle_in_D0, point_bounds,
                         lower_product)
from d_verified import (Ball, add_up, mul_up, squared_norm, sub_down,
                        sqrt_down, sqrt_up)
from remote_runtime import remote_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-radius", type=float, default=1e-12)
    parser.add_argument("--radius-factor", type=float, default=4.0)
    args = parser.parse_args()
    if (os.environ.get("QI2_REMOTE_RUN") != "1"
            or Path(sys.executable).resolve()
            != Path("<internal>/python").resolve()):
        raise SystemExit("Use the authorized remote qi2 interpreter.")
    if args.minimum_radius <= 0 or args.radius_factor <= 1:
        raise SystemExit("Use positive radii and radius factor greater than one.")

    # Host imports precede the final same-process fresh query/allocation.
    remote_execution(max_threads=8)
    verified.configure("cupy", certified=True, digits=80)
    backend = verified.verify_backend()
    result = json.loads((args.output_dir / "d_enclosure.json").read_text())
    if (result.get("status") != "candidate only; enclosure not executed"
            or result.get("certificate_passed") is not False):
        raise RuntimeError("Report only a saved candidate-only result.")

    def scalar_ball(entry):
        return Ball(entry["center"], entry["radius"])

    center = {
        "R": [scalar_ball(entry) for entry in result["center_residual"]],
        "J": [[scalar_ball(entry) for entry in row]
              for row in result["center_J"]],
    }
    jacobian_center = np.array([[entry["center"] for entry in row]
                                for row in result["center_J"]])
    inverse = np.linalg.inv(jacobian_center)
    preliminary = newton_inequalities(
        center, np.ones(2), inverse, np.zeros((2, 2, 2)))
    widths = np.array([max(args.minimum_radius,
                           mul_up(args.radius_factor, value))
                       for value in preliminary["beta"]])
    candidate = np.array([result["candidate"]["s"],
                          result["candidate"]["n"]])
    required_keys = {"state_center", "state_radius", "pair_ordering",
                     "ruler_profile", "lapse_profile", "candidate",
                     "eigenvalue_interval", "excitation_source_profile"}
    with np.load(args.output_dir / "d_candidate.npz", allow_pickle=False) as archive:
        state = np.array(archive["state_center"], copy=True)
        state_radius = float(archive["state_radius"])
        ruler = np.ones(27)
        lapse = np.ones(27)
        ruler[13], lapse[13] = candidate
        expected_pairs = np.array([(i, j) for i in range(54)
                                   for j in range(i + 1, 54)], dtype=int)
        energy = result["candidate"]["energy"]
        archive_checks = {
            "exact_schema": set(archive.files) == required_keys,
            "full_state_shape": state.shape == (1431,),
            "complex128_state": state.dtype == np.dtype(np.complex128),
            "finite_state": bool(np.all(np.isfinite(state))),
            "full_integer_pair_ordering": (
                archive["pair_ordering"].dtype.kind in "iu"
                and np.array_equal(archive["pair_ordering"], expected_pairs)),
            "candidate_coordinates": np.array_equal(archive["candidate"], candidate),
            "all_ruler_fields": np.array_equal(archive["ruler_profile"], ruler),
            "all_lapse_fields": np.array_equal(archive["lapse_profile"], lapse),
            "excitation_energy_interval": np.array_equal(
                archive["eigenvalue_interval"], [energy["lower"], energy["upper"]]),
            "all_source_centers": np.array_equal(
                archive["excitation_source_profile"],
                result["source_profile"]["excitation_sources"]),
            "state_radius_includes_eigenvector_error": (
                np.isfinite(state_radius)
                and state_radius >= result["eigenpair_validation"]["vector_error"]),
        }
    if not all(archive_checks.values()):
        raise RuntimeError(f"Saved archive correspondence failed: {archive_checks}")
    point_square = squared_norm(Ball(state[:, None]))
    state_square = squared_norm(Ball(state[:, None], state_radius))
    lower_square, upper_square = state_square.bounds()
    archive_checks["unit_squared_norm_in_state_norm_interval"] = (
        lower_square <= 1 <= upper_square)
    point_lower, point_upper = point_square.bounds()
    norm_lower = max(0.0, sqrt_down(max(0.0, point_lower)))
    norm_upper = sqrt_up(point_upper)
    unit_distance_upper = max(0.0, add_up(1, -norm_lower),
                              add_up(norm_upper, -1))
    archive_checks["state_ball_contains_a_unit_vector"] = (
        norm_lower > 0 and unit_distance_upper <= state_radius)
    if not (archive_checks["unit_squared_norm_in_state_norm_interval"]
            and archive_checks["state_ball_contains_a_unit_vector"]):
        raise RuntimeError("Saved state radius does not cover unit normalization.")
    correspondence = {
        "scope": ("Exact archive fields and directed normalization checks. "
                  "Ground-state certification is inherited from the audited writer; "
                  "no Hamiltonian/eigenpair recomputation is performed."),
        "checks": archive_checks,
        "passed": all(archive_checks.values()),
        "archive_state_radius": state_radius,
        "JSON_eigenvector_error": result["eigenpair_validation"]["vector_error"],
        "additional_normalization_radius_lower": sub_down(
            state_radius, result["eigenpair_validation"]["vector_error"]),
        "point_norm_interval": [norm_lower, norm_upper],
        "unit_normalization_distance_upper": unit_distance_upper,
        "point_squared_norm": point_bounds(point_square),
        "state_ball_squared_norm": point_bounds(state_square),
    }
    source = result["center_source"]
    amplitude_lower = max(0.0, sqrt_down(max(0.0, source["lower"])))
    amplitude_upper = sqrt_up(source["upper"])
    with localcontext() as context:
        context.prec = verified.DECIMAL_DIGITS
        context.rounding = ROUND_FLOOR
        s_threshold = verified.float_down(
            verified.decimal(amplitude_lower) / verified.decimal(widths[0]))
    clock_argument = mul_up(75, widths[1])
    clock_lower = Ball(-clock_argument).scalar_exp().bounds()[0]
    clock_upper = verified.exp_up(clock_argument)
    clock_inside = rectangle_in_D0(candidate, (0, widths[1]))
    source_geometry = {
        "scope": ("Released P8/P9 feasibility diagnostic. The source transport "
                  "interval fixes s=s0; the full two-variable source box is not certified."),
        "center_amplitude_lower": amplitude_lower,
        "center_amplitude_upper": amplitude_upper,
        "L_s_strict_pass_threshold_lower": s_threshold,
        "required_s_condition": "A whole-box P7 bound must satisfy L_s < this threshold.",
        "whole_D0_clock_coefficient": 75,
        "clock_multiplier_bounds": [clock_lower, clock_upper],
        "clock_interval_contained_in_D0": clock_inside,
        "source_interval_at_fixed_s0": (
            [lower_product(max(0.0, source["lower"]), clock_lower),
             mul_up(source["upper"], clock_upper)] if clock_inside else None),
        "P5_P7_whole_box_norms_computed": False,
        "full_source_box_passed": False,
    }
    geometry = {
        "scope": ("Original-A preliminary geometry from the certified center residual/J. "
                  "Whole-box derivative and source inequalities are not executed."),
        "candidate": candidate.tolist(),
        "Y": inverse.tolist(),
        "Y_operator_norm_upper": Ball(inverse).norm(),
        "Y_infinity_norm_upper": max(add_up(*(abs(value) for value in row))
                                     for row in inverse),
        "beta": preliminary["beta"],
        "center_defect": preliminary["center_defect"],
        "radius_factor": args.radius_factor,
        "minimum_radius": args.minimum_radius,
        "proposed_radii": widths.tolist(),
        "proposed_rectangle_contained_in_D0": rectangle_in_D0(candidate, widths),
        "postprocess_arithmetic_prerequisite_passed": backend["passed"],
        "archive_correspondence": correspondence,
        "sharpened_source_geometry": source_geometry,
        "arithmetic_assumptions": result["arithmetic_hypotheses"],
        "FTZ_clarification": (
            "NVRTC --ftz applies to single precision. The explicit double kernels "
            "preserve subnormal inputs/results; the absolute allowance covers "
            "result underflow. Elementary-operation assumptions remain required."),
        "FTZ_sources": [
            "https://docs.nvidia.com/cuda/archive/12.8.0/nvrtc/index.html",
            "https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#floating-point-instructions"],
        "root_enclosure_passed": False,
        "stellar_type_passed": False,
        "operator_claim": False,
    }
    (args.output_dir / "d_candidate_geometry.json").write_text(
        json.dumps(geometry, indent=2, allow_nan=False) + "\n")
    lines = [geometry["scope"],
             f"preliminary Newton beta: {geometry['beta']}",
             f"proposed radii: {geometry['proposed_radii']}",
             f"preconditioner operator norm upper: {geometry['Y_operator_norm_upper']}",
             f"archive correspondence passed: {correspondence['passed']}",
             f"archive state radius: {state_radius}",
             f"P9 strict L_s threshold lower: {s_threshold}",
             f"fixed-s0 clock source interval: {source_geometry['source_interval_at_fixed_s0']}"]
    (args.output_dir / "d_candidate_geometry.txt").write_text(
        "\n".join(lines) + "\n")
    print(json.dumps(geometry, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
