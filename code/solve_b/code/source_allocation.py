#!/usr/bin/env python3
"""reviewer B-specified A0 validation of one free polar packet on prescribed fields.

Run only in the remote qi2 environment. This ring probe does not implement the
interaction, four-packet inventory, an internal ruler, a star, or the T1 gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SIGMA1 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)


def adjoint(matrix: np.ndarray) -> np.ndarray:
    return matrix.conj().T


def hermitian(matrix: np.ndarray) -> np.ndarray:
    return (matrix + adjoint(matrix)) / 2


def positive_sqrt(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, vectors = np.linalg.eigh(hermitian(matrix))
    if eigenvalues[0] <= 0:
        raise ValueError(f"Strictly positive square-root input required: {eigenvalues[0]}")
    return hermitian((vectors * np.sqrt(eigenvalues)) @ adjoint(vectors))


def frobenius(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(matrix, ord="fro"))


def real_trace(matrix: np.ndarray) -> float:
    trace = np.trace(matrix)
    if abs(trace.imag) > 1e-10 * max(1.0, abs(trace.real)):
        raise ValueError(f"Expected real trace: {trace}")
    return float(trace.real)


def ring_probe() -> dict:
    nsite, spacing, mass = 8, 1.0, 0.7
    ruler = np.array([1.00, 1.07, 0.96, 1.02, 1.04, 0.99, 1.03, 0.97])
    lapse = np.array([1.0, 0.98, 1.02, 1.01, 0.99, 1.03, 1.0, 0.97])
    shift_plus = np.zeros((nsite, nsite), dtype=complex)
    for site in range(nsite):
        shift_plus[site, (site + 1) % nsite] = 1
    derivative = -1j * (shift_plus - adjoint(shift_plus)) / (2 * spacing)
    dirac = np.kron(derivative, SIGMA1)
    ruler_spin = np.repeat(ruler, 2)
    matrix_a = mass * np.diag(ruler_spin ** -1) + 1j * np.diag(ruler_spin ** -3) @ dirac
    energy0 = positive_sqrt(adjoint(matrix_a) @ matrix_a)
    energy1 = positive_sqrt(matrix_a @ adjoint(matrix_a))
    root_energy0 = positive_sqrt(energy0)
    polar_u = matrix_a @ np.linalg.inv(energy0)
    size = energy0.shape[0]
    identity = np.eye(size, dtype=complex)
    transform = np.block([[identity, identity], [polar_u, -polar_u]]) / np.sqrt(2)
    tplus, tminus = transform[:, :size], transform[:, size:]
    zeros = np.zeros_like(energy0)

    def b0(clock: np.ndarray) -> np.ndarray:
        return hermitian(root_energy0 @ np.diag(np.repeat(clock * ruler, 2)) @ root_energy0)

    def one_body(clock: np.ndarray) -> np.ndarray:
        allocation = b0(clock)
        return np.block([[zeros, allocation @ adjoint(polar_u)], [polar_u @ allocation, zeros]])

    def sea_projector(clock: np.ndarray) -> tuple[np.ndarray, dict]:
        values, vectors = np.linalg.eigh(hermitian(one_body(clock)))
        occupied = vectors[:, values < 0]
        if occupied.shape[1] != size:
            raise ValueError("Unexpected sea rank")
        singular = np.linalg.svd(adjoint(tminus) @ occupied, compute_uv=False)
        return occupied @ adjoint(occupied), {
            "rank": int(occupied.shape[1]),
            "spectral_gap": float(np.min(np.abs(values))),
            "min_eigenspace_overlap_singular_value": float(singular[-1]),
            "max_eigenspace_overlap_deviation": float(np.max(np.abs(singular - 1))),
        }

    base_b0 = b0(lapse)
    base_h = one_body(lapse)
    diagonal_target = np.block([[base_b0, zeros], [zeros, -base_b0]])
    reference_projector = tminus @ adjoint(tminus)
    base_projector, base_sea = sea_projector(lapse)
    unit_projector, unit_sea = sea_projector(np.ones(nsite))
    eigen_a = np.linalg.svd(matrix_a, compute_uv=False)
    delta_energy = energy1 - energy0
    vacuum_source = np.array([
        ruler[site] * real_trace(delta_energy[2 * site:2 * site + 2, 2 * site:2 * site + 2])
        for site in range(nsite)
    ])

    def vacuum(clock: np.ndarray) -> float:
        return real_trace(np.diag(np.repeat(clock * ruler, 2)) @ delta_energy)

    analytic_sources = []
    per_site = []
    h_unit = one_body(np.ones(nsite))
    for site in range(nsite):
        projector = np.zeros_like(identity)
        projector[2 * site:2 * site + 2, 2 * site:2 * site + 2] = np.eye(2)
        polar_source = hermitian(ruler[site] * root_energy0 @ projector @ root_energy0)
        hole_source = polar_source.T
        analytic_sources.append(polar_source)
        dirac_projector = np.block([[projector, zeros], [zeros, projector]])
        midpoint_derivative = (dirac_projector @ h_unit + h_unit @ dirac_projector) / 2
        midpoint_source = hermitian(adjoint(tplus) @ midpoint_derivative @ tplus)
        difference = polar_source - midpoint_source
        per_site.append({
            "site": site,
            "polar_source_frobenius": frobenius(polar_source),
            "midpoint_source_frobenius": frobenius(midpoint_source),
            "difference_frobenius": frobenius(difference),
            "difference_relative_to_polar": frobenius(difference) / frobenius(polar_source),
            "polar_min_eigenvalue": float(np.linalg.eigvalsh(polar_source)[0]),
            "hole_min_eigenvalue": float(np.linalg.eigvalsh(hermitian(hole_source))[0]),
            "midpoint_min_eigenvalue": float(np.linalg.eigvalsh(midpoint_source)[0]),
        })

    finite_differences = []
    for step in (1e-3, 1e-4, 1e-5):
        particles, holes, projected, vacuum_errors, projector_errors = [], [], [], [], []
        for site, source in enumerate(analytic_sources):
            plus, minus = lapse.copy(), lapse.copy()
            plus[site] += step
            minus[site] -= step
            fd_b = (b0(plus) - b0(minus)) / (2 * step)
            fd_h = (one_body(plus) - one_body(minus)) / (2 * step)
            fd_h_chart = adjoint(transform) @ fd_h @ transform
            full_excitation_source = np.block([[source, zeros], [zeros, -source]])
            particles.append(frobenius(fd_b - source))
            holes.append(frobenius(fd_b.T - source.T))
            projected.append(frobenius(fd_h_chart - full_excitation_source))
            fd_w = (vacuum(plus) - vacuum(minus)) / (2 * step)
            vacuum_errors.append(abs(fd_w - vacuum_source[site]))
            perturbed_projector, _ = sea_projector(plus)
            projector_errors.append(frobenius(perturbed_projector - reference_projector))
        finite_differences.append({
            "step": step,
            "max_particle_source_error_frobenius": max(particles),
            "max_hole_transpose_error_frobenius": max(holes),
            "max_projected_dirac_source_error_frobenius": max(projected),
            "max_vacuum_source_error_absolute": max(vacuum_errors),
            "max_perturbed_sea_projector_error_frobenius": max(projector_errors),
        })

    result = {
        "parameters": {"nsite": nsite, "spin_dimension": 2, "a": spacing, "m": mass,
                       "psi": ruler.tolist(), "N": lapse.tolist(), "orientation": "+"},
        "algebra": {
            "min_singular_value_A": float(eigen_a[-1]),
            "unitarity_error_frobenius": frobenius(adjoint(polar_u) @ polar_u - identity),
            "E0_square_error_frobenius": frobenius(energy0 @ energy0 - adjoint(matrix_a) @ matrix_a),
            "E1_square_error_frobenius": frobenius(energy1 @ energy1 - matrix_a @ adjoint(matrix_a)),
            "E1_polar_conjugacy_error_frobenius": frobenius(energy1 - polar_u @ energy0 @ adjoint(polar_u)),
            "block_diagonalization_error_frobenius": frobenius(adjoint(transform) @ base_h @ transform - diagonal_target),
            "base_sea_projector_error_frobenius": frobenius(base_projector - reference_projector),
            "lapse_change_sea_projector_error_frobenius": frobenius(base_projector - unit_projector),
            "base_sea": base_sea,
            "unit_lapse_sea": unit_sea,
        },
        "vacuum": {
            "W0": vacuum(lapse),
            "clock_derivative": vacuum_source.tolist(),
            "clock_affinity_error_absolute": abs(vacuum(lapse) - float(lapse @ vacuum_source)),
            "constant_L_W0": vacuum(1 / ruler),
            "inverse_psi_weighted_clock_derivative_sum": float(np.sum(vacuum_source / ruler)),
        },
        "finite_differences": finite_differences,
        "midpoint_comparison": {
            "scope": "Two different nonuniform-clock laws in the same polar positive-excitation chart.",
            "site_rows": per_site,
            "uniform_clock_polar_source_sum_error_frobenius": frobenius(sum(analytic_sources) - b0(np.ones(nsite))),
        },
    }
    return result


def uniform_vertices() -> dict:
    mass, ruler, spacing = 0.7, 1.2, 0.1
    identity = np.eye(2, dtype=complex)
    scalar_target = 1 / (4 * mass * ruler ** 4)
    square_target = -1 / (4 * ruler ** 4)
    rows = []
    for wavevector in (0.04, 0.02, 0.01):
        sine = float(np.sin(spacing * wavevector) / spacing)
        omega0 = mass
        omegaq = float(np.sqrt(mass ** 2 + ruler ** -4 * sine ** 2))
        u0 = identity
        uq = (mass * identity + 1j * ruler ** -2 * sine * SIGMA1) / omegaq
        vertex_polar = np.sqrt(omega0 * omegaq) * identity
        vertex_midpoint = ((omega0 + omegaq) / 4) * (identity + adjoint(u0) @ uq)
        difference = vertex_polar - vertex_midpoint
        square_difference = adjoint(vertex_midpoint) @ vertex_midpoint - adjoint(vertex_polar) @ vertex_polar
        half_trace = real_trace(difference) / 2 / sine ** 2
        half_square_trace = real_trace(square_difference) / 2 / sine ** 2
        rows.append({
            "q": wavevector, "d_a_q": sine, "omega_q": omegaq,
            "raw_spin_trace_vertex_difference_over_d_squared": real_trace(difference) / sine ** 2,
            "half_spin_trace_vertex_difference_over_d_squared": half_trace,
            "half_spin_trace_target": scalar_target,
            "half_spin_trace_error_absolute": abs(half_trace - scalar_target),
            "half_spin_trace_singular_square_difference_over_d_squared": half_square_trace,
            "singular_square_scalar_target": square_target,
            "singular_square_target_error_absolute": abs(half_square_trace - square_target),
            "singular_square_isotropy_error_frobenius": frobenius(square_difference / sine ** 2 - half_square_trace * identity),
        })
    return {"parameters": {"p": 0.0, "m": mass, "psi": ruler, "a": spacing}, "rows": rows}


def validation_flags(ring: dict, vertices: dict) -> dict:
    algebra = ring["algebra"]
    errors = [value for name, value in algebra.items() if name.endswith("error_frobenius")]
    finite_errors = [row["max_projected_dirac_source_error_frobenius"] for row in ring["finite_differences"]]
    half_trace_errors = [row["half_spin_trace_error_absolute"] for row in vertices["rows"]]
    square_errors = [row["singular_square_target_error_absolute"] for row in vertices["rows"]]
    return {
        "algebra_errors_below_1e_minus_10": max(errors) < 1e-10,
        "clock_fd_errors_below_1e_minus_8": max(finite_errors) < 1e-8,
        "constant_L_W0_below_1e_minus_10": abs(ring["vacuum"]["constant_L_W0"]) < 1e-10,
        "polar_source_nonnegative_to_1e_minus_10": min(row["polar_min_eigenvalue"] for row in ring["midpoint_comparison"]["site_rows"]) >= -1e-10,
        "scalar_vertex_error_decreases_under_halving_q": all(a > b for a, b in zip(half_trace_errors, half_trace_errors[1:])),
        "singular_square_error_decreases_under_halving_q": all(a > b for a, b in zip(square_errors, square_errors[1:])),
    }


def text_summary(result: dict) -> str:
    ring, vertices = result["ring"], result["uniform_vertices"]
    lines = [result["scope"], "", "Ring8 algebra:"]
    for key, value in ring["algebra"].items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.12g}")
    lines += ["", f"W0: {ring['vacuum']['W0']:.12g}",
              f"constant_L_W0: {ring['vacuum']['constant_L_W0']:.12g}",
              f"weighted_clock_source_sum: {ring['vacuum']['inverse_psi_weighted_clock_derivative_sum']:.12g}",
              "", "Clock finite differences (step, particle, hole, projected Dirac, vacuum):"]
    for row in ring["finite_differences"]:
        lines.append(" ".join(f"{row[key]:.9g}" for key in (
            "step", "max_particle_source_error_frobenius", "max_hole_transpose_error_frobenius",
            "max_projected_dirac_source_error_frobenius", "max_vacuum_source_error_absolute")))
    lines += ["", "Same-chart midpoint comparison (site, norm difference, relative norm difference):"]
    for row in ring["midpoint_comparison"]["site_rows"]:
        lines.append(f"{row['site']} {row['difference_frobenius']:.9g} {row['difference_relative_to_polar']:.9g}")
    lines += ["", "Uniform vertices (q, half trace/d², target, singular-square half trace/d², target):"]
    for row in vertices["rows"]:
        lines.append(" ".join(f"{row[key]:.12g}" for key in (
            "q", "half_spin_trace_vertex_difference_over_d_squared", "half_spin_trace_target",
            "half_spin_trace_singular_square_difference_over_d_squared", "singular_square_scalar_target")))
    lines += ["", "Numerical validation flags:"]
    lines.extend(f"{key}: {value}" for key, value in result["validation_flags"].items())
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ring = ring_probe()
    vertices = uniform_vertices()
    result = {
        "scope": "A0 single free positive-orientation polar packet on prescribed commuting fields; no interacting completion, full T1 inventory, internal ruler, or stellar claim.",
        "ring": ring,
        "uniform_vertices": vertices,
        "validation_flags": validation_flags(ring, vertices),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "source_allocation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    summary = text_summary(result)
    (args.output_dir / "source_allocation.txt").write_text(summary)
    print(summary, end="")
    if not all(result["validation_flags"].values()):
        raise SystemExit("Numerical validation flag failed; inspect the saved result.")


if __name__ == "__main__":
    main()
