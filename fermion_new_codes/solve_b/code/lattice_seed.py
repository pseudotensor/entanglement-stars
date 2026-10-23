#!/usr/bin/env python3
"""Reviewed A actual open2^3 seed gate on exact excitation sectors n=0,1,2.

Builds the regulator-before-Walsh filtered physical seed, its full sea-Wick
(2,2) projection in all charge blocks, the positive denominator, and every
contact in the exact two-excitation congruence. No larger Fock sector,
stellar state, nontrivial cell-distance derivative, or isolated charge is
claimed. Run only in remote qi2.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, expm

from lattice_trace import MASSES_SQUARED, GRADING, PAULI, polar_packet, sine_lattice
from polar_law import polar_packet as named_polar_packet
from source_allocation import adjoint, frobenius, hermitian, real_trace


TAUX = np.array([[0, 1], [1, 0]], dtype=complex)
TAUY = np.array([[0, -1j], [1j, 0]], dtype=complex)
BETA = np.kron(TAUX, np.eye(2))
ALPHA = [np.kron(TAUY, matrix) for matrix in PAULI]


def pairs_for(nmode: int) -> np.ndarray:
    return np.array(list(itertools.combinations(range(nmode), 2)), dtype=int)


def second_quantized_pair(matrix: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    i, j = pairs[:, 0, None], pairs[:, 1, None]
    k, ell = pairs[None, :, 0], pairs[None, :, 1]
    return (matrix[i, k] * (j == ell) - matrix[i, ell] * (j == k)
            - matrix[j, k] * (i == ell) + matrix[j, ell] * (i == k))


def pair_product(matrix_m: np.ndarray, matrix_n: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    # Independent external-leg formula specified in the reviewed C experiment.
    i, j = pairs[:, 0, None], pairs[:, 1, None]
    k, ell = pairs[None, :, 0], pairs[None, :, 1]
    return (matrix_m[i, k] * matrix_n[j, ell] - matrix_m[i, ell] * matrix_n[j, k]
            - matrix_m[j, k] * matrix_n[i, ell] + matrix_m[j, ell] * matrix_n[i, k])


def bilinear_data(particle: np.ndarray, hole: np.ndarray, gamma: np.ndarray, pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    particle_number = adjoint(particle) @ gamma @ particle
    hole_number = -(adjoint(hole) @ gamma @ hole).T
    number_matrix = block_diag(particle_number, hole_number)
    pair_coefficients = adjoint(particle) @ gamma @ hole
    nparticle = particle.shape[1]
    vector = np.zeros(len(pairs), dtype=complex)
    for row, (left, right) in enumerate(pairs):
        if left < nparticle <= right:
            vector[row] = pair_coefficients[left, right - nparticle]
    return number_matrix, vector


def projected_bilinear_product(number_m: np.ndarray, vector_m: np.ndarray,
                               number_n: np.ndarray, vector_n: np.ndarray,
                               pairs: np.ndarray) -> np.ndarray:
    # Exact full-Wick (2,2) extraction independently confirmed by the A author.
    qm = second_quantized_pair(number_m, pairs)
    qn = second_quantized_pair(number_n, pairs)
    return (qm @ qn - second_quantized_pair(number_m @ number_n, pairs)
            + np.outer(vector_m, vector_n.conj()) + np.outer(vector_n, vector_m.conj()))


def assemble(ruler: np.ndarray, lapse: np.ndarray, coordinates: np.ndarray, dirac: np.ndarray,
             uv_r: int | None = None, omega: float = 1.0) -> dict:
    nsite, size = len(ruler), dirac.shape[0]
    def packet_for(mass):
        if uv_r is None:
            return polar_packet(mass, ruler, dirac)
        return named_polar_packet(ruler, dirac, mass, uv_r=uv_r, omega=omega)
    light = packet_for(0.5)
    identity = np.eye(size, dtype=complex)
    wplus = np.vstack([identity, light["U"]]) / np.sqrt(2)
    wminus = np.vstack([identity, -light["U"]]) / np.sqrt(2)
    root = light["rootE0"]
    b = hermitian(root @ np.diag(np.repeat(lapse * ruler, 2)) @ root)
    b_unit = hermitian(root @ np.diag(np.repeat(ruler, 2)) @ root)
    excitation = block_diag(b, b.T)
    excitation_unit = block_diag(b_unit, b_unit.T)
    nmode = 2 * size
    pairs = pairs_for(nmode)
    h0_pair = hermitian(second_quantized_pair(excitation_unit, pairs))
    h_pair = hermitian(second_quantized_pair(excitation, pairs))
    inverse_ruler = np.diag(np.repeat(ruler ** -1, 2))
    kernel = hermitian(inverse_ruler @ dirac @ inverse_ruler)
    regulator = hermitian(expm(-(kernel @ kernel) / (2 * 0.5 ** 2)))
    regulator_native = block_diag(regulator, regulator)
    psi_c = float(np.sqrt(np.mean(ruler ** 2)))
    contact_weight = -0.05 * psi_c ** -6
    bond_weight = -0.02 / 2 * 2.0 ** -3
    seed = np.zeros_like(h0_pair)
    raw_seed = np.zeros_like(seed)
    pp_pairs = pairs_for(size)
    pp_indices = np.flatnonzero(pairs[:, 1] < size)
    independent_pp_seed = np.zeros((len(pp_pairs), len(pp_pairs)), dtype=complex)
    filter_car_errors = []
    filter_ranks = []
    for valley in itertools.product((0, 1), repeat=3):
        native_filter = np.zeros((4, 2 * size), dtype=complex)
        for site, point in enumerate(coordinates.astype(int)):
            sign = (-1) ** int(np.dot(valley, point)) / 8
            native_filter[0:2, 2 * site:2 * site + 2] = sign * np.eye(2)
            native_filter[2:4, size + 2 * site:size + 2 * site + 2] = sign * np.eye(2)
        filter_car_errors.append(frobenius(native_filter @ adjoint(native_filter) - np.eye(4) / 8))
        filtered = native_filter @ regulator_native
        filter_ranks.append(int(np.linalg.matrix_rank(filtered)))
        particle, hole = filtered @ wplus, filtered @ wminus
        for gamma, weight in [(BETA, contact_weight)] + [(matrix, bond_weight) for matrix in ALPHA]:
            number_matrix, vector = bilinear_data(particle, hole, gamma, pairs)
            quartic = projected_bilinear_product(number_matrix, vector, number_matrix, vector, pairs)
            seed += weight * quartic
            raw_seed += weight * (second_quantized_pair(number_matrix, pairs) @ second_quantized_pair(number_matrix, pairs))
            particle_bilinear = adjoint(particle) @ gamma @ particle
            independent_pp_seed += weight * pair_product(particle_bilinear, particle_bilinear, pp_pairs)
    seed_hermiticity_error = frobenius(seed - adjoint(seed))
    seed = hermitian(seed)
    free_values, free_vectors = np.linalg.eigh(h0_pair)
    seed_energy_chart = adjoint(free_vectors) @ seed @ free_vectors
    s2 = hermitian(free_vectors @ (seed_energy_chart / (free_values[:, None] + free_values[None, :])) @ adjoint(free_vectors))
    s_values, s_vectors = np.linalg.eigh(s2)
    dressing = hermitian((s_vectors * np.exp(s_values)) @ adjoint(s_vectors))
    dressed_excitation = hermitian(dressing @ h_pair @ dressing)
    vacuum_operator = np.zeros_like(dirac)
    packets = []
    for mass_squared, sign in zip(MASSES_SQUARED, GRADING):
        packet = light if mass_squared == 0.25 else packet_for(float(np.sqrt(mass_squared)))
        vacuum_operator += sign * (packet["E1"] - packet["E0"])
        packets.append(packet)
    t_source = np.array([real_trace(vacuum_operator[2 * site:2 * site + 2, 2 * site:2 * site + 2]) for site in range(nsite)])
    vacuum_source = ruler * t_source
    w0 = float(lapse @ vacuum_source)
    clock_sources, excitation_clock_sources = [], []
    for site in range(nsite):
        projector = np.zeros_like(identity)
        projector[2 * site:2 * site + 2, 2 * site:2 * site + 2] = np.eye(2)
        particle_source = hermitian(ruler[site] * root @ projector @ root)
        one_source = block_diag(particle_source, particle_source.T)
        dressed_source = hermitian(dressing @ second_quantized_pair(one_source, pairs) @ dressing)
        excitation_clock_sources.append(dressed_source)
        clock_sources.append(dressed_source + vacuum_source[site] * np.eye(len(pairs)))
    return {"ruler": ruler, "lapse": lapse, "light": light, "packets": packets,
            "wplus": wplus, "wminus": wminus, "B": b, "B_unit": b_unit, "excitation": excitation,
            "pairs": pairs, "H0_pair": h0_pair, "H_pair": h_pair, "V2": seed, "S2": s2,
            "S_eigenvalues": s_values, "S_vectors": s_vectors, "expS2": dressing,
            "H2_exc": dressed_excitation, "W0": w0, "vacuum_source": vacuum_source,
            "clock_sources": clock_sources, "excitation_clock_sources": excitation_clock_sources,
            "seed_hermiticity_error": seed_hermiticity_error,
            "pp_coefficient_independent_error": frobenius(seed[np.ix_(pp_indices, pp_indices)] - independent_pp_seed),
            "raw_number_product_difference_frobenius": frobenius(seed - raw_seed),
            "filter_car_error_max": max(filter_car_errors), "filter_ranks": filter_ranks}


def gate(ruler: np.ndarray, lapse: np.ndarray, coordinates: np.ndarray, dirac: np.ndarray, clamped: bool,
         uv_r: int | None = None, omega: float = 1.0) -> dict:
    data = assemble(ruler, lapse, coordinates, dirac, uv_r=uv_r, omega=omega)
    pairs, s2, seed = data["pairs"], data["S2"], data["V2"]
    size, dimension = dirac.shape[0], len(pairs)
    charge = np.where(pairs < size, 1, -1).sum(axis=1)
    commutator_charge = (charge[:, None] - charge[None, :]) * s2
    sylvester_error = frobenius(s2 @ data["H0_pair"] + data["H0_pair"] @ s2 - seed)
    annihilators = np.zeros((2 * size, 2 * size, dimension), dtype=complex)
    for column, (first, second) in enumerate(pairs):
        annihilators[first, second, column] = 1
        annihilators[second, first, column] = -1
    dressed_annihilators = annihilators @ data["expS2"]
    from_forms = np.einsum("jri,jk,krb->ib", dressed_annihilators.conj(), data["excitation"], dressed_annihilators, optimize=True)
    source_fd_rows = []
    for step in (1e-3, 1e-4, 1e-5):
        errors = []
        for site in range(len(ruler)):
            plus, minus = lapse.copy(), lapse.copy()
            plus[site] += step
            minus[site] -= step
            root = data["light"]["rootE0"]
            bplus = hermitian(root @ np.diag(np.repeat(plus * ruler, 2)) @ root)
            bminus = hermitian(root @ np.diag(np.repeat(minus * ruler, 2)) @ root)
            hplus = data["expS2"] @ second_quantized_pair(block_diag(bplus, bplus.T), pairs) @ data["expS2"]
            hminus = data["expS2"] @ second_quantized_pair(block_diag(bminus, bminus.T), pairs) @ data["expS2"]
            # Vacuum term included independently from its affine site derivative.
            hplus += float(plus @ data["vacuum_source"]) * np.eye(dimension)
            hminus += float(minus @ data["vacuum_source"]) * np.eye(dimension)
            errors.append(frobenius((hplus - hminus) / (2 * step) - data["clock_sources"][site]))
        source_fd_rows.append({"step": step, "max_total_clock_source_error_frobenius": max(errors)})
    coupling_rows = []
    for step in (0.1, 0.01, 0.001):
        eplus = hermitian((data["S_vectors"] * np.exp(step * data["S_eigenvalues"])) @ adjoint(data["S_vectors"]))
        eminus = hermitian((data["S_vectors"] * np.exp(-step * data["S_eigenvalues"])) @ adjoint(data["S_vectors"]))
        derivative = (eplus @ data["H0_pair"] @ eplus - eminus @ data["H0_pair"] @ eminus) / (2 * step)
        coupling_rows.append({"step": step, "projected_seed_derivative_error_frobenius": frobenius(derivative - seed),
                              "relative_to_seed": frobenius(derivative - seed) / frobenius(seed)})
    boundary = list(range(4, 8))
    interior = list(range(4))
    full_energy = data["W0"] * np.eye(dimension) + data["H2_exc"]
    gamma = sum(lapse[site] * data["clock_sources"][site] for site in boundary)
    a19_error = frobenius(full_energy - sum(lapse[site] * data["clock_sources"][site] for site in interior) - gamma)
    a20_error = None
    if clamped:
        rho_sum = sum(data["excitation_clock_sources"][site] / ruler[site] for site in range(8))
        a20_error = frobenius(gamma - rho_sum + sum(data["clock_sources"][site] / ruler[site] for site in interior))
    pi = np.zeros_like(dirac)
    for site in interior:
        pi[2 * site:2 * site + 2, 2 * site:2 * site + 2] = np.eye(2)
    allocation = np.diag(np.repeat(lapse * ruler, 2))
    crossing_lhs, crossing_rhs = 0j, 0j
    for sign, packet in zip(GRADING, data["packets"]):
        u = packet["U"]
        z = packet["E0"] @ adjoint(u)
        crossing_lhs += sign * np.trace(pi @ allocation @ (packet["E1"] - packet["E0"]))
        crossing_rhs += sign * (np.trace(pi @ (allocation @ u - u @ allocation) @ z)
                               + np.trace((pi @ u - u @ pi) @ allocation @ z))
    time = 0.37
    h_native = np.block([[np.zeros_like(data["B"]), data["B"] @ adjoint(data["light"]["U"])],
                         [data["light"]["U"] @ data["B"], np.zeros_like(data["B"])]])
    pplus = data["wplus"] @ adjoint(data["wplus"])
    pminus = data["wminus"] @ adjoint(data["wminus"])
    native_evolution = expm(-1j * h_native * time)
    protected_greater = data["wplus"] @ expm(-1j * data["B"] * time) @ adjoint(data["wplus"])
    protected_lesser = data["wminus"] @ expm(1j * data["B"].T * time).T @ adjoint(data["wminus"])
    h_values = np.linalg.eigvalsh(data["H2_exc"])
    source_minimum = min(float(np.linalg.eigvalsh(source)[0]) for source in data["excitation_clock_sources"])
    result = {
        "profile": "separately rebuilt collar-clamped" if clamped else "nonuniform prescribed diagnostic",
        "psi": ruler.tolist(), "N": lapse.tolist(), "clamped_collar": clamped,
        "dimensions": {"n0": 1, "n1": 32, "n2": dimension, "charge_plus2": int(np.sum(charge == 2)),
                       "charge_zero": int(np.sum(charge == 0)), "charge_minus2": int(np.sum(charge == -2))},
        "seed": {"V2_frobenius": frobenius(seed), "S2_frobenius": frobenius(s2),
                 "S2_min_eigenvalue": float(data["S_eigenvalues"][0]), "S2_max_eigenvalue": float(data["S_eigenvalues"][-1]),
                 "V2_hermiticity_error_frobenius": data["seed_hermiticity_error"],
                 "S2_hermiticity_error_frobenius": frobenius(s2 - adjoint(s2)),
                 "charge_commutator_error_frobenius": frobenius(commutator_charge),
                 "unit_lapse_sylvester_error_frobenius": sylvester_error,
                 "independent_pp_four_leg_coefficient_error_frobenius": data["pp_coefficient_independent_error"],
                 "full_Wick_vs_raw_number_product_difference_frobenius": data["raw_number_product_difference_frobenius"],
                 "unregulated_Walsh_CAR_1_over_8_error_frobenius": data["filter_car_error_max"],
                 "filtered_native_field_ranks": data["filter_ranks"]},
        "completion": {"exact_congruence_vs_annihilator_form_error_frobenius": frobenius(data["H2_exc"] - from_forms),
                       "minimum_two_excitation_energy_above_W0": float(h_values[0]),
                       "first_two_excitation_energies_above_W0": h_values[:12].tolist(),
                       "interaction_energy_matrix_difference_frobenius": frobenius(data["H2_exc"] - data["H_pair"]),
                       "S0_norm": 0.0, "S1_norm": 0.0,
                       "native_greater_two_point_error_frobenius": frobenius(protected_greater - native_evolution @ pplus),
                       "native_lesser_two_point_error_frobenius": frobenius(protected_lesser - native_evolution @ pminus),
                       "two_point_time": time, "minimum_excitation_clock_source_eigenvalue": source_minimum},
        "source_finite_differences": source_fd_rows, "coupling_finite_differences": coupling_rows,
        "vacuum": {"W0": data["W0"], "trace_X_sum": float(np.sum(data["vacuum_source"] / ruler))},
        "boundary": {"interior": interior, "collar": boundary, "geometric_three_dimensional_interior": False,
                     "A19_affine_boundary_identity_error_frobenius": a19_error,
                     "A20_clamped_boundary_identity_error_frobenius": a20_error,
                     "A21_vacuum_crossing_identity_absolute_error": float(abs(crossing_lhs - crossing_rhs))},
    }
    result["validation_flags"] = {
        "nonzero_actual_seed": bool(frobenius(seed) > 1e-10),
        "independent_pp_coefficients_below_1e_minus_10": bool(data["pp_coefficient_independent_error"] < 1e-10),
        "sylvester_error_below_1e_minus_10": bool(sylvester_error < 1e-10),
        "exact_annihilator_congruence_below_1e_minus_10": bool(frobenius(data["H2_exc"] - from_forms) < 1e-10),
        "positive_n2_excitation_energy": bool(h_values[0] > 0),
        "source_positive_to_1e_minus_10": bool(source_minimum >= -1e-10),
        "clock_FD_error_below_1e_minus_6": bool(max(row["max_total_clock_source_error_frobenius"] for row in source_fd_rows) < 1e-6),
        "boundary_identities_below_1e_minus_10": bool(a19_error < 1e-10 and (a20_error is None or a20_error < 1e-10)),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--uv-r", type=int, choices=(2, 3), default=None)
    parser.add_argument("--omega", type=float, default=1.0)
    args = parser.parse_args()
    coordinates, dirac = sine_lattice((2, 2, 2), 1.0, periodic=False)
    xi = 2 * coordinates - 1
    ruler = 1 + 0.04 * (xi[:, 0] + xi[:, 1] / 2 + xi[:, 2] / 4)
    lapse = 1 + 0.03 * xi[:, 2]
    profiles = []
    print("Building actual open2^3 all-charge n2 projected seed.", flush=True)
    profiles.append(gate(ruler, lapse, coordinates, dirac, clamped=False, uv_r=args.uv_r, omega=args.omega))
    print("First profile complete; rebuilding the collar-clamped profile.", flush=True)
    clamped_ruler, clamped_lapse = ruler.copy(), lapse.copy()
    clamped_ruler[4:] = 1
    clamped_lapse[4:] = 1
    profiles.append(gate(clamped_ruler, clamped_lapse, coordinates, dirac, clamped=True, uv_r=args.uv_r, omega=args.omega))
    result = {"scope": "Actual reviewed open2^3 seed on exact n0,n1,n2 blocks with all pp/ph/hh charges. Full 2^32 light Fock space and higher sectors are not built. One complete cell does not test a nontrivial ruler-distance derivative; arbitrary algebraic partition is not a geometric interior or isolated exterior.",
              "free_family": "original_sine_polar" if args.uv_r is None else f"uv_r{args.uv_r}_polar_candidate",
              "parameters": {"a": 1.0, "mass_squared": MASSES_SQUARED.tolist(), "Lambda": 0.5, "ell": 2.0,
                             "g_M": 0.05, "g_B": 0.02, "complete_cells": 1, "one_sine_operator": True,
                             "uv_r": args.uv_r, "Omega": None if args.uv_r is None else args.omega,
                             "regulator_filter_order": "Walsh M_nu after R_psi", "Walsh_weight": "1/8, no sqrt8", "bond_distance": 0.0},
              "profiles": profiles}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "lattice_seed.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = [result["scope"]]
    for profile in profiles:
        lines += ["", profile["profile"], f"dimensions: {profile['dimensions']}",
                  f"V2 norm: {profile['seed']['V2_frobenius']:.12g}; S2 norm: {profile['seed']['S2_frobenius']:.12g}",
                  f"independent pp coefficient error: {profile['seed']['independent_pp_four_leg_coefficient_error_frobenius']:.9g}",
                  f"Sylvester error: {profile['seed']['unit_lapse_sylvester_error_frobenius']:.9g}",
                  f"annihilator vs congruence error: {profile['completion']['exact_congruence_vs_annihilator_form_error_frobenius']:.9g}",
                  f"minimum n2 excitation energy: {profile['completion']['minimum_two_excitation_energy_above_W0']:.12g}",
                  f"minimum source eigenvalue: {profile['completion']['minimum_excitation_clock_source_eigenvalue']:.9g}",
                  f"boundary errors: {profile['boundary']}"]
        lines.extend(f"{name}: {value}" for name, value in profile["validation_flags"].items())
    summary = "\n".join(lines) + "\n"
    (args.output_dir / "lattice_seed.txt").write_text(summary)
    print(summary, end="", flush=True)
    if not all(all(profile["validation_flags"].values()) for profile in profiles):
        raise SystemExit("An actual seeded n<=2 numerical validation flag failed.")


if __name__ == "__main__":
    main()
