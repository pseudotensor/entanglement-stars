#!/usr/bin/env python3
"""Reviewed strand-A periodic4^3 finite-q trace gate, not a continuum match.

Uses one parent sine operator and all four polar packets. There is no extra
valley multiplier. Run only with the remote qi2 interpreter.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from source_allocation import adjoint, frobenius, hermitian, positive_sqrt, real_trace


PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex)
MASSES_SQUARED = np.array([0.25, 0.75, 1.0, 1.5])
GRADING = np.array([1, -1, -1, 1])


def sine_lattice(shape: tuple[int, int, int], spacing: float, periodic: bool) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.array(list(itertools.product(*(range(n) for n in shape))), dtype=int)
    indices = {tuple(point): index for index, point in enumerate(coordinates)}
    size = len(coordinates)
    dirac = np.zeros((2 * size, 2 * size), dtype=complex)
    for axis in range(3):
        shift = np.zeros((size, size), dtype=complex)
        for row, point in enumerate(coordinates):
            neighbor = point.copy()
            neighbor[axis] += 1
            if periodic:
                neighbor[axis] %= shape[axis]
            if neighbor[axis] < shape[axis]:
                shift[row, indices[tuple(neighbor)]] = 1
        dirac += np.kron((shift - adjoint(shift)) / (2j * spacing), PAULI[axis])
    return coordinates * spacing, hermitian(dirac)


def polar_packet(mass: float, ruler: np.ndarray, dirac: np.ndarray) -> dict:
    ruler_spin = np.repeat(ruler, 2)
    matrix_a = mass * np.diag(ruler_spin ** -1) + 1j * np.diag(ruler_spin ** -3) @ dirac
    energy0 = positive_sqrt(adjoint(matrix_a) @ matrix_a)
    energy1 = positive_sqrt(matrix_a @ adjoint(matrix_a))
    polar_u = matrix_a @ np.linalg.inv(energy0)
    return {"A": matrix_a, "E0": energy0, "E1": energy1, "U": polar_u,
            "rootE0": positive_sqrt(energy0), "rootE1": positive_sqrt(energy1)}


def trace_source(ruler: np.ndarray, dirac: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    size = len(ruler)
    grouped = np.zeros_like(dirac)
    algebra = []
    for mass_squared, sign in zip(MASSES_SQUARED, GRADING):
        packet = polar_packet(float(np.sqrt(mass_squared)), ruler, dirac)
        delta = packet["E1"] - packet["E0"]
        grouped += sign * delta
        identity = np.eye(2 * size)
        algebra.append({"mass_squared": float(mass_squared), "orientation": int(sign),
                        "unitarity_error_frobenius": frobenius(adjoint(packet["U"]) @ packet["U"] - identity),
                        "polar_factor_error_frobenius": frobenius(packet["A"] - packet["U"] @ packet["E0"]),
                        "polar_conjugacy_error_frobenius": frobenius(packet["E1"] - packet["U"] @ packet["E0"] @ adjoint(packet["U"]))})
    site_trace = np.array([real_trace(grouped[2 * site:2 * site + 2, 2 * site:2 * site + 2]) for site in range(size)])
    return site_trace, algebra


def momentum_vertex(ruler: float, shape: tuple[int, int, int], spacing: float, q: np.ndarray) -> float:
    momenta = np.array(list(itertools.product(*(2 * np.pi * np.arange(n) / (n * spacing) for n in shape))))
    sine = np.sin(spacing * momenta) / spacing
    sine_prime = np.sin(spacing * (momenta + q)) / spacing
    difference_squared = np.sum((sine_prime - sine) ** 2, axis=1)
    summand = np.zeros(len(momenta))
    for mass_squared, sign in zip(MASSES_SQUARED, GRADING):
        omega = np.sqrt(mass_squared + ruler ** -4 * np.sum(sine ** 2, axis=1))
        omega_prime = np.sqrt(mass_squared + ruler ** -4 * np.sum(sine_prime ** 2, axis=1))
        summand += sign * difference_squared / (omega + omega_prime)
    volume = len(momenta) * spacing ** 3
    return float(-6 * ruler ** -6 * np.sum(summand) / volume)


def mixed_matrix_derivative(ruler: float, step: float, cosine: np.ndarray, dirac: np.ndarray, volume: float) -> tuple[float, float]:
    source_plus, algebra_plus = trace_source(ruler + step * cosine, dirac)
    source_minus, algebra_minus = trace_source(ruler - step * cosine, dirac)
    clock_plus, clock_minus = 1 + step * cosine, 1 - step * cosine
    wpp = float(clock_plus @ source_plus)
    wpm = float(clock_minus @ source_plus)
    wmp = float(clock_plus @ source_minus)
    wmm = float(clock_minus @ source_minus)
    derivative = (wpp - wpm - wmp + wmm) / (4 * step ** 2 * volume)
    algebra_error = max(row["polar_conjugacy_error_frobenius"] for row in algebra_plus + algebra_minus)
    return derivative, algebra_error


def evaluate() -> dict:
    shape, spacing = (4, 4, 4), 1.0
    coordinates, dirac = sine_lattice(shape, spacing, periodic=True)
    nsite = len(coordinates)
    volume = nsite * spacing ** 3
    q = np.array([np.pi / 2, 0.0, 0.0])
    cosine = np.cos(coordinates @ q)
    rows = []
    drift_rows = []
    for ruler in (1.0, 1.2):
        prediction = momentum_vertex(ruler, shape, spacing, q) / 2
        for step in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
            measured, algebra_error = mixed_matrix_derivative(ruler, step, cosine, dirac, volume)
            rows.append({"P": ruler, "epsilon": step, "zeta": step,
                         "matrix_mixed_derivative_per_volume": measured,
                         "independent_A26_momentum_prediction_over_two": prediction,
                         "difference": measured - prediction,
                         "relative_difference": (measured - prediction) / prediction,
                         "max_packet_polar_conjugacy_error_frobenius": algebra_error})
        # Independent finite-volume ruler drift: matrix perturbation and momentum prediction.
        perturbation_step, uniform_step = 1e-4, 1e-3
        matrix_plus, _ = mixed_matrix_derivative(ruler + uniform_step, perturbation_step, cosine, dirac, volume)
        matrix_minus, _ = mixed_matrix_derivative(ruler - uniform_step, perturbation_step, cosine, dirac, volume)
        measured_drift = (matrix_plus - matrix_minus) / (2 * uniform_step)
        predicted_drift = (momentum_vertex(ruler + uniform_step, shape, spacing, q)
                           - momentum_vertex(ruler - uniform_step, shape, spacing, q)) / (4 * uniform_step)
        drift_rows.append({"P": ruler, "uniform_P_step": uniform_step, "mixed_perturbation_step": perturbation_step,
                           "matrix_dP_mixed_derivative_per_volume": measured_drift,
                           "independent_momentum_dP_prediction_over_two": predicted_drift,
                           "difference": measured_drift - predicted_drift,
                           "relative_difference": (measured_drift - predicted_drift) / predicted_drift})
    prescribed_ruler = 1.0 + 0.03 * cosine
    clock_allocation = 1.0 + 0.025 * cosine
    lapse = clock_allocation / prescribed_ruler
    source, algebra = trace_source(prescribed_ruler, dirac)
    vacuum_source = prescribed_ruler * source
    w0 = float(clock_allocation @ source)
    clock_rows = []
    for step in (1e-3, 1e-4, 1e-5):
        derivative_errors = []
        hessian_errors = []
        for site in range(nsite):
            plus, minus = lapse.copy(), lapse.copy()
            plus[site] += step
            minus[site] -= step
            w_plus = float((plus * prescribed_ruler) @ source)
            w_minus = float((minus * prescribed_ruler) @ source)
            derivative_errors.append(abs((w_plus - w_minus) / (2 * step) - vacuum_source[site]))
            hessian_errors.append(abs((w_plus - 2 * w0 + w_minus) / step ** 2))
        clock_rows.append({"step": step, "max_clock_derivative_absolute_error": max(derivative_errors),
                           "max_clamped_clock_hessian_absolute": max(hessian_errors)})
    result = {
        "scope": "Reviewed A free periodic4^3 trace gate at diagnostic masses and a=1. Finite-q/finite-volume control only; no Newton match, resolved continuum coefficient, seed protection, star, or isolated boundary claim.",
        "parameters": {"shape": list(shape), "a": spacing, "mass_squared": MASSES_SQUARED.tolist(),
                       "orientations": GRADING.tolist(), "q": q.tolist(), "site_count": nsite,
                       "one_Weyl_dimension": 2 * nsite, "Dirac_dimension_per_packet": 4 * nsite,
                       "independent_sine_operators": 1, "extra_valley_factor": 1,
                       "cosine_mean_square": float(np.mean(cosine ** 2))},
        "mixed_trace_rows": rows, "finite_volume_ruler_drift_rows": drift_rows,
        "clock_checks": {"W0": w0, "clock_affinity_error_absolute": abs(w0 - float(lapse @ vacuum_source)),
                         "constant_L_W0": float(np.sum(source)), "site_clock_derivative": vacuum_source.tolist(),
                         "finite_differences": clock_rows, "packet_algebra": algebra},
    }
    last_rows = [row for row in rows if row["epsilon"] == 1e-4]
    result["validation_flags"] = {
        "finite_q_trace_relative_error_below_1e_minus_5": bool(max(abs(row["relative_difference"]) for row in last_rows) < 1e-5),
        "finite_volume_ruler_drift_relative_error_below_1e_minus_4": bool(max(abs(row["relative_difference"]) for row in drift_rows) < 1e-4),
        "constant_L_cancellation_absolute_below_1e_minus_10": bool(abs(float(np.sum(source))) < 1e-10),
        "clock_derivative_error_below_1e_minus_8": bool(max(row["max_clock_derivative_absolute_error"] for row in clock_rows) < 1e-8),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "lattice_trace.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = [result["scope"], "", "P step matrix mixed derivative /V independent momentum prediction /2 relative error:"]
    for row in result["mixed_trace_rows"]:
        lines.append(f"{row['P']:.6g} {row['epsilon']:.6g} {row['matrix_mixed_derivative_per_volume']:.12g} "
                     f"{row['independent_A26_momentum_prediction_over_two']:.12g} {row['relative_difference']:.9g}")
    lines += ["", "Finite-volume ruler drift (P, matrix, momentum, relative error):"]
    for row in result["finite_volume_ruler_drift_rows"]:
        lines.append(f"{row['P']:.6g} {row['matrix_dP_mixed_derivative_per_volume']:.12g} "
                     f"{row['independent_momentum_dP_prediction_over_two']:.12g} {row['relative_difference']:.9g}")
    lines += ["", f"constant-L W0: {result['clock_checks']['constant_L_W0']:.12g}", "", "Validation flags:"]
    lines.extend(f"{name}: {value}" for name, value in result["validation_flags"].items())
    summary = "\n".join(lines) + "\n"
    (args.output_dir / "lattice_trace.txt").write_text(summary)
    print(summary, end="")
    if not all(result["validation_flags"].values()):
        raise SystemExit("A finite-q free trace numerical flag failed.")


if __name__ == "__main__":
    main()
