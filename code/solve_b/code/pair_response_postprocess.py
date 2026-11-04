"""Saved-data C checks prescribed by the reviewer B author; no new field runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from remote_runtime import remote_execution


def error_summary(observed, reference, names):
    difference = np.asarray(observed) - np.asarray(reference)
    location = np.unravel_index(np.argmax(np.abs(difference)), difference.shape)
    norm = float(np.linalg.norm(reference))
    return {"maximum_entry_error": float(np.max(np.abs(difference))),
            "location_indices": list(map(int, location)),
            "location_sources": [names[i] for i in location],
            "frobenius_error": float(np.linalg.norm(difference)),
            "relative_frobenius_error": None if norm == 0 else float(np.linalg.norm(difference) / norm)}


def examine(profile, path, names):
    data = np.load(path)
    h, first, second = data["H"], data["H_first"], data["H_second"]
    energies, vectors = data["eigenvalues"], data["eigenvectors"]
    state = vectors[:, 0]
    vacuum_second = data["W0_second"].real
    total_ground = np.asarray(profile["analytic_ground_Hessian"])
    clamped = np.array([[np.vdot(state, second[a, b] @ state).real for b in range(4)] for a in range(4)])
    transitions = np.array([vectors.conj().T @ vertex @ state for vertex in first])
    gaps = energies[1:] - energies[0]
    spectral = np.array([[2 * np.vdot(transitions[a, 1:] / gaps, transitions[b, 1:]).real
                          for b in range(4)] for a in range(4)])
    identity = np.eye(len(state), dtype=complex)
    projector = np.outer(state, state.conj())
    complement = identity - projector
    reduced = complement @ (h - energies[0] * identity) @ complement + projector
    rhs = np.column_stack([complement @ vertex @ state for vertex in first])
    response = complement @ np.linalg.solve(reduced, rhs)
    direct = 2 * (rhs.conj().T @ response).real
    output = {"label": profile["label"], "free_family": "original_sine_polar",
              "source_names": names,
              "E_total": float(energies[0]), "W0": float(data["W0"].real),
              "E_excitation_ground": float(energies[0] - data["W0"].real),
              "H_matter_ground": (total_ground - vacuum_second).tolist(),
              "saved_matrix_checks": {
                  "H_hermiticity_frobenius": float(np.linalg.norm(h - h.conj().T)),
                  "first_hermiticity_frobenius": [float(np.linalg.norm(x - x.conj().T)) for x in first],
                  "second_hermiticity_max_frobenius": float(max(np.linalg.norm(x - x.conj().T) for row in second for x in row)),
                  "second_source_symmetry_max_frobenius": float(max(np.linalg.norm(second[a, b] - second[b, a]) for a in range(4) for b in range(4))),
                  "eigenvector_orthonormality_frobenius": float(np.linalg.norm(vectors.conj().T @ vectors - identity)),
                  "maximum_saved_eigenpair_residual": float(np.max(np.linalg.norm(h @ vectors - vectors * energies, axis=0))),
                  "rebuilt_clamped_Hessian_error": error_summary(clamped, profile["analytic_clamped_Hessian"], names),
                  "rebuilt_spectral_susceptibility_error": error_summary(spectral, profile["complete_pair_resolvent_susceptibility"], names),
                  "rebuilt_ground_Hessian_error": error_summary(clamped - spectral, total_ground, names),
                  "direct_projected_linear_solve_susceptibility_error": error_summary(direct, spectral, names),
                  "direct_projected_linear_solve_residual_frobenius": float(np.linalg.norm((h - energies[0] * identity) @ response - rhs)),
                  "direct_response_ground_orthogonality_norm": float(np.linalg.norm(state.conj() @ response))},
              "ground_FD_errors": [],
              "scope": "saved-matrix numerical consistency and existing finite differences; no interval, neighborhood, native or thermodynamic certificate"}
    finite_rows = profile["finite_difference_checks"]
    for row in finite_rows:
        output["ground_FD_errors"].append({"step": row["step"],
                                           **error_summary(row["FD_ground_Hessian"], total_ground, names)})
    if len(finite_rows) >= 2:
        larger, smaller = finite_rows[-2:]
        if larger["step"] == 2 * smaller["step"]:
            estimate = (4 * np.array(smaller["FD_ground_Hessian"]) - np.array(larger["FD_ground_Hessian"])) / 3
            output["existing_ladder_Richardson_estimate"] = {
                "steps": [larger["step"], smaller["step"]], "Hessian": estimate.tolist(),
                "error_against_analytic": error_summary(estimate, total_ground, names),
                "scope": "heuristic cancellation of the leading h^2 error, using existing data only"}
    if profile["label"] == "U":
        spectra = []
        for source in (1, 3):
            elements = np.abs(transitions[source, 1:]) ** 2
            weights = 2 * elements / gaps
            total = float(np.sum(weights))
            row = {"source": names[source], "total_positive_susceptibility_weight": total,
                   "total_matrix_element_squared": float(np.sum(elements)),
                   "base_minimum_gap": float(np.min(gaps)), "cumulative_gap_cuts": [],
                   "smallest_gap_at_relative_weight_thresholds": []}
            for cutoff in (1e-5, 1e-3):
                chosen = gaps < cutoff
                weight = float(np.sum(weights[chosen]))
                row["cumulative_gap_cuts"].append({"gap_less_than": cutoff,
                                                    "states": int(np.sum(chosen)),
                                                    "matrix_element_squared_sum": float(np.sum(elements[chosen])),
                                                    "susceptibility_weight_sum": weight,
                                                    "fraction_of_total_susceptibility": None if total == 0 else weight / total})
            for relative in (1e-12, 1e-10, 1e-8):
                threshold = relative * total
                chosen = weights > threshold
                row["smallest_gap_at_relative_weight_thresholds"].append({
                    "relative_to_total_susceptibility": relative,
                    "absolute_weight_threshold": threshold,
                    "smallest_gap": None if not np.any(chosen) else float(np.min(gaps[chosen]))})
            spectra.append(row)
        output["uniform_spectral_weights"] = spectra
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    execution = remote_execution(max_threads=8)
    saved = json.loads((args.input_dir / "pair_response.json").read_text())
    result = {"scope": "reviewer B-prescribed postprocessing of saved finite C matrices; no new source rebuilds",
              "execution": execution, "profiles": [], "complete": False}
    for profile in saved["profiles"]:
        if profile["lowest_cluster_size"] != 1:
            raise RuntimeError("Scalar postprocessing requires the saved simple branch")
        result["profiles"].append(examine(profile, args.input_dir / f"pair_response_{profile['label']}.npz", saved["source_names"]))
    result["complete"] = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pair_response_postprocess.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = [result["scope"]]
    for profile in result["profiles"]:
        lines.append(f"{profile['label']}: Eexc={profile['E_excitation_ground']:.12g}")
        lines.append("Hmatter,ground=" + json.dumps(profile["H_matter_ground"]))
        for row in profile["ground_FD_errors"]:
            lines.append(f"  h={row['step']:g} max-entry={row['maximum_entry_error']:.6g} at{row['location_sources']}, relativeFrobenius={row['relative_frobenius_error']:.6g}")
        lines.append("  projected direct-resolvent discrepancy=" + str(profile["saved_matrix_checks"]["direct_projected_linear_solve_susceptibility_error"]["frobenius_error"]))
        if "uniform_spectral_weights" in profile:
            lines.append("  spectral weights=" + json.dumps(profile["uniform_spectral_weights"]))
    summary = "\n".join(lines) + "\n"
    (args.output_dir / "pair_response_postprocess.txt").write_text(summary)
    print(summary, end="", flush=True)


if __name__ == "__main__":
    main()
