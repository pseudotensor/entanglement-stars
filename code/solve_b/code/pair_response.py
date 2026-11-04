"""Reviewed C finite prescribed-background two-excitation response builder.

The default family is the reviewed original sine polar law. An explicitly
named uv_r=2 or uv_r=3 selects the independently specified T10 ingredient; its
sea, packet roots, and unit-lapse denominators are rebuilt. This module
does not adopt that candidate as a physical production law. All execution
and postprocessing must run in remote qi2.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from lattice_seed import ALPHA, BETA, pairs_for, pair_product, second_quantized_pair
from lattice_trace import GRADING, MASSES_SQUARED, sine_lattice
from matrix_jet import Jet, sylvester_jet
from polar_law import diagonal_field, polar_jet
from remote_runtime import remote_execution
from source_allocation import adjoint, frobenius, hermitian


SOURCE_NAMES = ("N0", "N1", "s0", "s1")


def pair_product_jet(matrix_m: Jet, matrix_n: Jet, pairs: np.ndarray) -> Jet:
    i, j = pairs[:, 0, None], pairs[:, 1, None]
    k, ell = pairs[None, :, 0], pairs[None, :, 1]
    def take(matrix, row, column):
        return matrix.map(lambda x: x[row, column])
    return (take(matrix_m, i, k) * take(matrix_n, j, ell)
            - take(matrix_m, i, ell) * take(matrix_n, j, k)
            - take(matrix_m, j, k) * take(matrix_n, i, ell)
            + take(matrix_m, j, ell) * take(matrix_n, i, k))


def interpolation_weights(point, coordinates, shape, spacing):
    native = np.asarray(point) / spacing
    lower = np.minimum(np.floor(native).astype(int), np.array(shape) - 2)
    fraction = native - lower
    indices = {tuple(np.rint(x / spacing).astype(int)): k for k, x in enumerate(coordinates)}
    weights = np.zeros(len(coordinates))
    for corner in itertools.product((0, 1), repeat=3):
        weights[indices[tuple(lower + corner)]] += np.prod([
            fraction[d] if corner[d] else 1 - fraction[d] for d in range(3)])
    return weights


def line_weights(first, second, coordinates, shape, spacing):
    distance = float(np.linalg.norm(second - first))
    if distance == 0:
        return interpolation_weights(first, coordinates, shape, spacing), distance
    breaks = {0.0, 1.0}
    for axis in range(3):
        change = second[axis] - first[axis]
        if change:
            for plane in range(shape[axis]):
                t = (plane * spacing - first[axis]) / change
                if 0 < t < 1:
                    breaks.add(float(t))
    weights = np.zeros(len(coordinates))
    # Multilinear interpolation is cubic along each segment. Two Gauss
    # nodes integrate its weights exactly, before any ruler is assigned.
    node = 1 / np.sqrt(3)
    partition = sorted(breaks)
    for left, right in zip(partition[:-1], partition[1:]):
        for sign in (-1, 1):
            t = (left + right) / 2 + sign * node * (right - left) / 2
            point = first + t * (second - first)
            weights += (right - left) / 2 * interpolation_weights(point, coordinates, shape, spacing)
    return weights, distance


def cell_geometry(coordinates, shape, spacing):
    origins = list(itertools.product(*(range(n - 1) for n in shape)))
    centres = [(np.array(origin) + 0.5) * spacing for origin in origins]
    centre_weights = [interpolation_weights(x, coordinates, shape, spacing) for x in centres]
    lines = [[line_weights(x, y, coordinates, shape, spacing) for y in centres] for x in centres]
    return origins, centres, centre_weights, lines


def source_fields(ruler, lapse, modes, analytic=True):
    ruler_first = np.zeros((4, len(ruler)))
    lapse_first = np.zeros((4, len(lapse)))
    if analytic:
        lapse_first[:2] = modes
        ruler_first[2:] = modes
    return Jet(ruler, ruler_first), Jet(lapse, lapse_first)


def assemble_pair(ruler, lapse, coordinates, dirac, shape=(3, 2, 2), spacing=1.0,
                  modes=None, analytic=False, uv_r=None, omega=1.0,
                  cutoff=0.5, length=2.0, couplings=(0.05, 0.02), coupling_scale=1.0,
                  check_seed=False, mutation=None):
    """Return prescribed scalar-source response in the moving polar +2 chart.

    modes is (phi0,phi1), with source order N0,N1,s0,s1. This API
    never computes the fixed-native-CAR clamp or omitted higher sectors.
    """
    if min(ruler) <= 0 or min(lapse) <= 0 or min(cutoff, length, omega) <= 0:
        raise ValueError("Prescribed inputs left the positive domain.")
    if modes is None:
        modes = np.zeros((2, len(ruler)))
    psi, clock = source_fields(np.asarray(ruler), np.asarray(lapse), np.asarray(modes), analytic)
    particles = len(dirac)
    pairs = pairs_for(particles)
    light = polar_jet(psi, dirac, float(np.sqrt(MASSES_SQUARED[0])), uv_r, omega)
    wplus = light["U"].map(lambda x: np.vstack([np.zeros_like(x), x]))
    wplus += np.vstack([np.eye(particles), np.zeros((particles, particles))])
    wplus /= np.sqrt(2)
    if mutation == "frozen_Wplus_derivative":
        wplus = wplus.constant()
    root = light["rootE0"]
    b = (root @ diagonal_field(clock * psi) @ root).hermitian()
    b_unit = (root @ diagonal_field(psi) @ root).hermitian()
    hpair = b.map(lambda x: second_quantized_pair(x, pairs)).hermitian()
    hunit = b_unit.map(lambda x: second_quantized_pair(x, pairs)).hermitian()
    inverse = diagonal_field(psi ** -1)
    kernel = (inverse @ dirac @ inverse).hermitian()
    regulator = (-(kernel @ kernel) / (2 * cutoff ** 2)).exponential().hermitian()
    if mutation == "frozen_regulator_derivative":
        regulator = regulator.constant()
    native_regulator = regulator.map(lambda x: np.kron(np.eye(2), x))
    origins, centres, centre_weights, lines = cell_geometry(coordinates, shape, spacing)
    psi_squared = psi ** 2
    contact = [(psi_squared.map(lambda x, weights=w: np.dot(weights, x))) ** -3 for w in centre_weights]
    bond = [[(-(psi_squared.map(lambda x, weights=lines[r][s][0]: np.dot(weights, x))
                * lines[r][s][1]) ** 2 / length ** 2).exponential() / length ** 3
             for s in range(len(origins))] for r in range(len(origins))]
    seed = Jet(np.zeros((len(pairs), len(pairs))))
    independent_seed = np.zeros_like(seed.value)
    indices = {tuple(np.rint(x / spacing).astype(int)): k for k, x in enumerate(coordinates)}
    g_m, g_b = [x * coupling_scale for x in couplings]
    for valley in itertools.product((0, 1), repeat=3):
        vertices = []
        for origin in origins:
            native_filter = np.zeros((4, 2 * particles), complex)
            for corner in itertools.product((0, 1), repeat=3):
                site = indices[tuple(np.array(origin) + corner)]
                sign = (-1) ** int(np.dot(valley, corner)) / 8
                native_filter[:2, 2 * site:2 * site + 2] = sign * np.eye(2)
                native_filter[2:, particles + 2 * site:particles + 2 * site + 2] = sign * np.eye(2)
            field = native_filter @ native_regulator @ wplus
            vertices.append([(field.dagger() @ gamma @ field).hermitian() for gamma in [BETA] + ALPHA])
        for r in range(len(origins)):
            matrix = vertices[r][0]
            weight = -g_m * spacing ** -3 * contact[r]
            seed += weight * pair_product_jet(matrix, matrix, pairs)
            if check_seed:
                qm = second_quantized_pair(matrix.value, pairs)
                independent_seed += weight.value * (qm @ qm - second_quantized_pair(matrix.value @ matrix.value, pairs))
            for s in range(len(origins)):
                for direction in range(1, 4):
                    m, n = vertices[r][direction], vertices[s][direction]
                    weight = -g_b / 2 * bond[r][s]
                    seed += weight * pair_product_jet(m, n, pairs)
                    if check_seed:
                        independent_seed += weight.value * (second_quantized_pair(m.value, pairs)
                            @ second_quantized_pair(n.value, pairs) - second_quantized_pair(m.value @ n.value, pairs))
    seed = seed.hermitian()
    s2 = sylvester_jet(hunit, seed)
    dressing = s2.exponential().hermitian()
    excitation = (dressing @ hpair @ dressing).hermitian()
    grouped = Jet(np.zeros_like(dirac))
    packet_algebra = []
    for mass_squared, sign in zip(MASSES_SQUARED, GRADING):
        packet = light if mass_squared == MASSES_SQUARED[0] else polar_jet(psi, dirac, float(np.sqrt(mass_squared)), uv_r, omega)
        grouped += sign * (packet["E1"] - packet["E0"])
        packet_algebra.append({"mass_squared": float(mass_squared),
            "unitary_error": frobenius(adjoint(packet["U"].value) @ packet["U"].value - np.eye(particles)),
            "polar_conjugacy_error": frobenius(packet["E1"].value - packet["U"].value @ packet["E0"].value @ adjoint(packet["U"].value))})
    site_trace = grouped.map(lambda x: np.trace(x.reshape(len(ruler), 2, len(ruler), 2), axis1=1, axis2=3).diagonal())
    vacuum = (clock * psi * site_trace).map(np.sum)
    total = excitation + vacuum * np.eye(len(pairs))
    d_squared = dirac @ dirac
    delta_scalar = np.trace(d_squared.reshape(len(ruler), 2, len(ruler), 2), axis1=1, axis2=3) / 2
    return {"H": total, "H_exc": excitation, "W0": vacuum, "B": b,
            "H_unit": hunit, "V2": seed, "S2": s2, "expS2": dressing,
            "pairs": pairs, "packet_algebra": packet_algebra,
            "independent_seed_error": None if not check_seed else frobenius(seed.value - independent_seed),
            "geometry": {"centres": np.array(centres).tolist(),
                         "centre_weights": [x.tolist() for x in centre_weights],
                         "line_weights": [[w.tolist() for w, _ in row] for row in lines],
                         "line_coordinate_lengths": [[d for _, d in row] for row in lines]},
            "scalar_spin_delta_error": frobenius(d_squared - np.kron(delta_scalar, np.eye(2)))}


def diagnostic_profiles(coordinates, spacing=1.0):
    native = np.rint(coordinates / spacing).astype(int)
    phi0 = np.ones(len(coordinates))
    phi1 = np.sqrt(2) * np.cos(np.pi / 3 * (native[:, 0] + 0.5))
    modes = np.array([phi0, phi1])
    w2, w3 = (-1.0) ** native[:, 1], (-1.0) ** native[:, 2]
    ruler = 1 + 0.07 * phi1 + 0.04 * w2 + 0.02 * w3 + 0.01 * phi1 * w2 + 0.005 * w2 * w3
    lapse = 1 + 0.03 * w2
    return modes, [("U", np.ones(len(coordinates)), np.ones(len(coordinates))), ("R", ruler, lapse)]


def real_expectation(vector, matrix):
    return float(np.real(np.vdot(vector, matrix @ vector)))


def profile_gate(label, ruler, lapse, coordinates, dirac, modes, options, output_dir=None):
    print(f"Building {label}: intrinsic source jet on dimension276.", flush=True)
    base = assemble_pair(ruler, lapse, coordinates, dirac, modes=modes,
                         analytic=True, check_seed=True, **options)
    h = base["H"]
    energies, vectors = np.linalg.eigh(h.value)
    residuals = np.linalg.norm(h.value @ vectors - vectors * energies, axis=0)
    scale = float(max(1.0, np.linalg.norm(h.value, 2)))
    tolerance = max(100 * float(np.max(residuals)), 100 * np.finfo(float).eps * scale)
    cluster_size = int(np.sum(energies - energies[0] <= tolerance))
    gap = float(energies[cluster_size] - energies[0]) if cluster_size < len(energies) else None
    simple = cluster_size == 1
    u0 = vectors[:, 0]
    cluster = vectors[:, :cluster_size]
    projected_vertices = [hermitian(adjoint(cluster) @ vertex @ cluster) for vertex in h.first]
    directional_slopes = [np.linalg.eigvalsh(vertex).tolist() for vertex in projected_vertices]
    if output_dir is not None:
        np.savez_compressed(output_dir / f"pair_response_{label}.npz",
                            H=h.value, H_first=h.first, H_second=h.second,
                            H_exc=base["H_exc"].value, V2=base["V2"].value, S2=base["S2"].value,
                            W0=base["W0"].value, W0_first=base["W0"].first,
                            W0_second=base["W0"].second, eigenvalues=energies, eigenvectors=vectors,
                            lowest_cluster=cluster, cluster_projected_vertices=np.array(projected_vertices))
    result = {"label": label, "psi": ruler.tolist(), "N": lapse.tolist(),
              "dimension": len(energies), "lowest_energies": energies[:12].tolist(),
              "maximum_eigenpair_residual": float(max(residuals)), "cluster_tolerance": tolerance,
              "lowest_cluster_size": cluster_size, "gap_above_cluster": gap,
              "branch_status": "simple at base; neighborhood not enclosed" if simple else "degenerate cluster; scalar Hessian not established",
              "directional_cluster_slopes": directional_slopes,
              "directional_right_slopes": [float(min(slopes)) for slopes in directional_slopes],
              "directional_left_slopes": [float(max(slopes)) for slopes in directional_slopes],
              "cluster_projected_vertices": [{"real": vertex.real.tolist(), "imaginary": vertex.imag.tolist()} for vertex in projected_vertices],
              "directional_slope_spreads": [float(max(slopes) - min(slopes)) for slopes in directional_slopes],
              "independent_seed_error": base["independent_seed_error"],
              "unit_lapse_sylvester_error": frobenius(base["H_unit"].value @ base["S2"].value + base["S2"].value @ base["H_unit"].value - base["V2"].value),
              "V2_frobenius": frobenius(base["V2"].value), "S2_frobenius": frobenius(base["S2"].value),
              "packet_algebra": base["packet_algebra"], "geometry": base["geometry"],
              "scalar_spin_delta_error": base["scalar_spin_delta_error"],
              "bare_NN_matrix_second_norm": float(max(frobenius(h.second[a, b]) for a in range(2) for b in range(2))),
              "W0": float(base["W0"].value.real),
              "W0_first": base["W0"].first.real.tolist(), "W0_second": base["W0"].second.real.tolist()}
    if simple:
        clamped = np.array([[real_expectation(u0, h.second[a, b]) for b in range(4)] for a in range(4)])
        transition = np.array([adjoint(vectors) @ vertex @ u0 for vertex in h.first])
        susceptibility = np.array([[2 * np.real(np.vdot(transition[a, 1:] / (energies[1:] - energies[0]), transition[b, 1:]))
                                    for b in range(4)] for a in range(4)])
        relaxed = clamped - susceptibility
        result.update({"analytic_clamped_Hessian": clamped.tolist(), "complete_pair_resolvent_susceptibility": susceptibility.tolist(),
                       "analytic_ground_Hessian": relaxed.tolist(),
                       "excitation_clamped_Hessian": (clamped - base["W0"].second.real).tolist(),
                       "ground_NN_maximum_eigenvalue": float(np.linalg.eigvalsh(relaxed[:2, :2])[-1])})
        if label == "U":
            result["uniform_U_source_checks"] = {
                "uniform_clock_vertex_minus_H_error": frobenius(h.first[0] - h.value),
                "ground_N0_N1_curvature": float(relaxed[0, 1]),
                "uniform_clock_resolvent_row_norm": float(np.linalg.norm(susceptibility[0])),
                "vacuum_first_jet_norm": float(np.linalg.norm(base["W0"].first)),
                "vacuum_static_minus_two_mixed": float(np.real(base["W0"].second[3, 3] - 2 * base["W0"].second[1, 3]))}
    cache = {}
    def rebuilt(theta):
        key = tuple(theta)
        if key not in cache:
            varied_lapse = lapse + theta[:2] @ modes
            varied_ruler = ruler + theta[2:] @ modes
            built = assemble_pair(varied_ruler, varied_lapse, coordinates, dirac, modes=modes, **options)
            value = built["H"].value
            row = {"matrix": value, "clamp": real_expectation(u0, value)}
            if simple:
                eigs = np.linalg.eigvalsh(value)
                row.update({"energy": float(eigs[0]), "gap": float(eigs[1] - eigs[0])})
            cache[key] = row
        return cache[key]
    finite_rows = []
    zero = np.zeros(4)
    centre = {"matrix": h.value, "clamp": float(energies[0]), "energy": float(energies[0])}
    for step in (1e-3, 5e-4, 2.5e-4):
        print(f"Full rebuild {label}, central step={step:g}.", flush=True)
        first_errors, second_errors, neighbourhood_gaps = [], [], []
        fd_clamp = np.zeros((4, 4))
        fd_ground = np.zeros((4, 4))
        for a in range(4):
            direction = np.eye(4)[a] * step
            plus, minus = rebuilt(direction), rebuilt(-direction)
            plus2, minus2 = rebuilt(2 * direction), rebuilt(-2 * direction)
            first_errors.append(frobenius((plus["matrix"] - minus["matrix"]) / (2 * step) - h.first[a]))
            matrix_second = (-plus2["matrix"] + 16 * plus["matrix"] - 30 * h.value + 16 * minus["matrix"] - minus2["matrix"]) / (12 * step ** 2)
            second_errors.append(frobenius(matrix_second - h.second[a, a]))
            fd_clamp[a, a] = (-plus2["clamp"] + 16 * plus["clamp"] - 30 * centre["clamp"] + 16 * minus["clamp"] - minus2["clamp"]) / (12 * step ** 2)
            if simple:
                neighbourhood_gaps += [row["gap"] for row in (plus, minus, plus2, minus2)]
                fd_ground[a, a] = (-plus2["energy"] + 16 * plus["energy"] - 30 * centre["energy"] + 16 * minus["energy"] - minus2["energy"]) / (12 * step ** 2)
            for b in range(a):
                second_direction = np.eye(4)[b] * step
                rows = [rebuilt(x * direction + y * second_direction) for x, y in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
                signs = (1, -1, -1, 1)
                matrix_second = sum(sign * row["matrix"] for sign, row in zip(signs, rows)) / (4 * step ** 2)
                second_errors.append(frobenius(matrix_second - h.second[a, b]))
                fd_clamp[a, b] = fd_clamp[b, a] = sum(sign * row["clamp"] for sign, row in zip(signs, rows)) / (4 * step ** 2)
                if simple:
                    neighbourhood_gaps += [row["gap"] for row in rows]
                    fd_ground[a, b] = fd_ground[b, a] = sum(sign * row["energy"] for sign, row in zip(signs, rows)) / (4 * step ** 2)
        row = {"step": step, "max_first_matrix_error": max(first_errors), "max_second_matrix_error": max(second_errors),
               "max_second_energy_scaled_error": max(second_errors) / scale}
        if simple:
            row.update({"minimum_neighbourhood_gap": min(neighbourhood_gaps), "FD_clamped_Hessian": fd_clamp.tolist(),
                        "FD_ground_Hessian": fd_ground.tolist(), "clamped_Hessian_error_frobenius": frobenius(fd_clamp - clamped),
                        "ground_Hessian_error_frobenius": frobenius(fd_ground - relaxed)})
        finite_rows.append(row)
    result["finite_difference_checks"] = finite_rows
    if simple:
        result["minimum_sampled_neighbourhood_gap"] = min(row["minimum_neighbourhood_gap"] for row in finite_rows)
        result["branch_status"] = ("simple at base; sampled gaps positive, neighborhood not enclosed"
                                   if result["minimum_sampled_neighbourhood_gap"] > tolerance
                                   else "simple at base only; sampled gaps do not resolve the branch")
    for mutation in ("frozen_Wplus_derivative", "frozen_regulator_derivative"):
        changed = assemble_pair(ruler, lapse, coordinates, dirac, modes=modes, analytic=True, mutation=mutation, **options)
        result[mutation] = {"ruler_first_matrix_change": [frobenius(changed["H"].first[a] - h.first[a]) for a in (2, 3)],
                            "ruler_second_matrix_change": [frobenius(changed["H"].second[a, b] - h.second[a, b]) for a, b in ((2, 2), (2, 3), (3, 3))],
                            "scope": "deliberately incorrect derivative mutation only"}
    coupling_checks = []
    unit = assemble_pair(ruler, np.ones_like(lapse), coordinates, dirac, modes=modes, **options)
    for step in (0.1, 0.01):
        plus = assemble_pair(ruler, np.ones_like(lapse), coordinates, dirac, modes=modes, coupling_scale=step, **options)
        minus = assemble_pair(ruler, np.ones_like(lapse), coordinates, dirac, modes=modes, coupling_scale=-step, **options)
        error = frobenius((plus["H"].value - minus["H"].value) / (2 * step) - unit["V2"].value)
        coupling_checks.append({"step": step, "first_seed_derivative_error": error,
                                "vacuum_change": float(abs(plus["W0"].value - minus["W0"].value))})
    result["unit_lapse_first_coupling_checks"] = coupling_checks
    scaled = assemble_pair(ruler, 1.1 * lapse, coordinates, dirac, modes=modes, **options)
    result["uniform_full_lapse_scaling_matrix_error"] = frobenius(scaled["H"].value - 1.1 * h.value)
    cutoff_options = dict(options, cutoff=0.6, length=1 / 0.6)
    regulator_changed = assemble_pair(ruler, lapse, coordinates, dirac, modes=modes, analytic=True, **cutoff_options)
    result["seed_regulator_vacuum_invariance"] = {
        "changed_Lambda": 0.6,
        "changed_ell": 1 / 0.6,
        "W0_error": float(abs(regulator_changed["W0"].value - base["W0"].value)),
        "W0_first_error": float(np.linalg.norm(regulator_changed["W0"].first - base["W0"].first)),
        "W0_second_error": float(np.linalg.norm(regulator_changed["W0"].second - base["W0"].second))}
    result["validation_scope"] = "finite prescribed-background two-excitation response; numerical estimates, not a stellar, native-clamped, low-q or higher-sector certificate"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--uv-r", type=int, choices=(2, 3), default=None)
    parser.add_argument("--omega", type=float, default=1.0)
    args = parser.parse_args()
    execution = remote_execution(max_threads=8)
    print("Remote execution: " + json.dumps(execution), flush=True)
    coordinates, dirac = sine_lattice((3, 2, 2), 1.0, periodic=False)
    modes, profiles = diagnostic_profiles(coordinates)
    options = {"uv_r": args.uv_r, "omega": args.omega}
    result = {"scope": "finite prescribed-background two-excitation response",
              "execution": execution,
              "free_family": "original_sine_polar" if args.uv_r is None else f"uv_r{args.uv_r}_polar_candidate",
              "parameters": {"shape": [3, 2, 2], "a": 1.0, "mass_squared": MASSES_SQUARED.tolist(),
                             "Lambda": 0.5, "ell": 2.0, "g_M": 0.05, "g_B": 0.02,
                             "uv_r": args.uv_r, "Omega": None if args.uv_r is None else args.omega},
              "source_names": list(SOURCE_NAMES), "source_modes": modes.tolist(), "profiles": []}
    # The independently specified odd-block Schur control: A2=gamma=kappa=1.
    result["odd_block_Schur_control"] = [
        {"q": q, "A2": 1.0, "gamma": 1.0, "kappa": 1.0,
         "complete_Z": float(np.real((q ** 2 - (1j * q) * (-1j * q)) / q ** 2)),
         "incorrect_all_blocks_even_Z": 1.0}
        for q in (0.04, 0.02, 0.01)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, ruler, lapse in profiles:
        result["profiles"].append(profile_gate(label, ruler, lapse, coordinates, dirac, modes, options, args.output_dir))
        (args.output_dir / "pair_response.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = [result["scope"], f"free family: {result['free_family']}"]
    for profile in result["profiles"]:
        lines += [f"{profile['label']}: {profile['branch_status']}; cluster={profile['lowest_cluster_size']}, gap={profile['gap_above_cluster']}",
                  f"  independent seed={profile['independent_seed_error']:.3g}; Sylvester={profile['unit_lapse_sylvester_error']:.3g}"]
        for row in profile["finite_difference_checks"]:
            lines += [f"  h={row['step']:.3g}: first matrix={row['max_first_matrix_error']:.3g}, second matrix={row['max_second_matrix_error']:.3g}"]
    summary = "\n".join(lines) + "\n"
    (args.output_dir / "pair_response.txt").write_text(summary)
    print(summary, end="", flush=True)


if __name__ == "__main__":
    main()
