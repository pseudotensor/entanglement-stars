#!/usr/bin/env python3
"""Original-A finite collar work at an actual retained D load-one candidate.

Implements the reviewed B1--B23 in strand_B_diagnostic.tex.  This is an
ordinary point evaluator and rebuilt finite-difference study, not an
interval assembly or a root-box/isolated-exterior certificate.  It does
not solve fields or minimize a state.  Missing, incompatible, or invalid
retained candidate inputs are refused; no synthetic candidate is supplied.

All execution, validation and postprocessing are remote qi2 only.  The
new source leaves the D, shared polar/seed/jet, and entropy sources intact.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from fractions import Fraction
from pathlib import Path

# One numerical worker.  Set the dense-library limits before host imports;
# remote_runtime's final launch check follows those imports and input reads.
for _thread_variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
from scipy.linalg import expm, expm_frechet

from lattice_seed import ALPHA, BETA, pair_product, pairs_for, second_quantized_pair
from lattice_trace import GRADING, MASSES_SQUARED, sine_lattice
from matrix_jet import Jet, positive_sylvester_solver
from pair_response import cell_geometry
from polar_law import polar_jet
from remote_runtime import remote_execution


PARAMETERS = {
    "law": "original_A_sine_polar", "shape": [3, 3, 3],
    "a": "1", "mass_squared": ["1/4", "3/4", "1", "3/2"],
    "eta": [1, -1, -1, 1], "Lambda": "1/2", "ell": "2",
    "g_M": "1/80", "g_B": "1/200", "source_load": "1",
    "complete_cells": 8, "ordered_cell_pairs": 64,
    "particle_modes": 54, "pair_dimension": 1431,
    "interior": [1, 1, 1], "collar_s_n": "1",
    "domain": {"s": ["999/1000", "1001/1000"], "n": ["999/1000", "1001/1000"]},
}
ARCHIVE_FIELDS = {
    "state_center", "state_radius", "pair_ordering", "ruler_profile",
    "lapse_profile", "candidate", "eigenvalue_interval", "excitation_source_profile",
}
PROBE_NAMES = ("dilation", "collar_ruler", "face_ruler", "face_clock", "central_ruler", "central_clock")
# An engineering stop, declared before inputs: conspicuous point rebuild
# disagreement only.  This is never an assembly bound or a certificate.
REBUILD_ROUNDOFF_SCALE_MULTIPLIER = 1 << 20


def adjoint(matrix):
    return np.asarray(matrix).conj().T


def norm(value):
    """Ordinary Euclidean/Frobenius estimate, never an outward norm bound."""
    return float(np.linalg.norm(np.asarray(value).reshape(-1)))


def finite(value, name):
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains a non-finite value")


def clean_hermitian(raw, label, diagnostics):
    finite(raw, label)
    diagnostics[label + "_raw_Hermitian_defect_Frobenius"] = norm(raw - adjoint(raw))
    return (raw + adjoint(raw)) / 2


def clean_antihermitian(raw, label, diagnostics):
    finite(raw, label)
    diagnostics[label + "_raw_antiHermitian_defect_Frobenius"] = norm(raw + adjoint(raw))
    return (raw - adjoint(raw)) / 2


def real_value(value, label, diagnostics):
    value = complex(value)
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise ArithmeticError(f"{label} is non-finite")
    diagnostics[label + "_imaginary_defect"] = abs(value.imag)
    return float(value.real)


def resource_check(runtime):
    threads = len(list(Path("/proc/self/task").iterdir()))
    if threads > 16 or os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError("Remote B work requires nice>=10 and <=16 total threads")
    runtime["maximum_threads_observed"] = max(runtime.get("maximum_threads_observed", 0), threads)


def require_scalar(value, name, nonnegative=False, positive=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"Retained {name} must be a finite numerical scalar")
    if nonnegative and value < 0 or positive and value <= 0:
        raise ValueError(f"Retained {name} has the wrong sign")
    return float(value)


def require_interval(item, name):
    if not isinstance(item, dict) or not {"center", "lower", "upper", "radius"} <= item.keys():
        raise ValueError(f"Retained {name} lacks its complete scalar interval")
    center = require_scalar(item["center"], name + ".center")
    lower = require_scalar(item["lower"], name + ".lower")
    upper = require_scalar(item["upper"], name + ".upper")
    require_scalar(item["radius"], name + ".radius", nonnegative=True)
    if not lower <= center <= upper:
        raise ValueError(f"Retained {name} has unordered/incompatible endpoints")
    return lower, upper


def load_candidate_metadata(result_path, archive_path):
    """The existing D JSON/NPZ adapter; no invented alternate candidate."""
    if not result_path.is_file() or not archive_path.is_file():
        raise ValueError("An actual retained D result and d_candidate.npz are required; candidate is absent")
    data = json.loads(result_path.read_text())
    parameters = data.get("parameters")
    if not isinstance(parameters, dict) or any(parameters.get(key) != value for key, value in PARAMETERS.items()):
        raise ValueError("Retained law/couplings/domain/geometry do not equal reviewed original-A B1/D E1")
    if (parameters.get("uv_r") is not None or data.get("uv_r") is not None
            or data.get("free_family", "original_A_sine_polar") != "original_A_sine_polar"):
        raise ValueError("This evaluator accepts uv_r=None original law only")
    if data.get("candidate_state_archive") != "d_candidate.npz" or archive_path.name != "d_candidate.npz":
        raise ValueError("The retained D archive contract is d_candidate.npz")
    if type(data.get("certificate_passed")) is not bool or not isinstance(data.get("status"), str):
        raise ValueError("Retain D's actual root-certificate boolean and status")
    for gate in ("candidate_gate", "reusable_seed_gate", "backend_verification"):
        if not isinstance(data.get(gate), dict) or data[gate].get("passed") is not True:
            raise ValueError(f"Retained D {gate} has not actually passed")
    seed_gate = data["reusable_seed_gate"]
    if seed_gate.get("shape") != [2, 2, 2] or seed_gate.get("pp_dimension") != 120:
        raise ValueError("The actual complete sea-Wick/pp seed gate is required")
    gap = data.get("gap_budget", {})
    gap_lower = require_scalar(gap.get("ground_gap_lower"), "full_sector_gap_lower", positive=True)
    require_scalar(gap.get("second_level_lower"), "second_level_lower", positive=True)
    if gap_lower <= 2 / 25:
        raise ValueError("Retained full-sector gap does not satisfy the reviewed original-law domain gate")
    validation = data.get("eigenpair_validation", {})
    require_scalar(validation.get("eigenpair_residual"), "eigenpair_residual", nonnegative=True)
    require_scalar(validation.get("vector_error"), "phase_adjusted_vector_error", nonnegative=True)
    require_scalar(validation.get("second_level_separation"), "second_level_separation", positive=True)
    candidate = data.get("candidate", {})
    require_scalar(candidate.get("s"), "s", positive=True)
    require_scalar(candidate.get("n"), "n", positive=True)
    require_interval(candidate.get("energy"), "ground_energy")
    residuals = data.get("center_residual")
    if not isinstance(residuals, list) or len(residuals) != 2:
        raise ValueError("Both actual D central residuals R1,R2 are required")
    for index, item in enumerate(residuals):
        require_interval(item, f"R{index + 1}")
    return data


def load_candidate_arrays(archive_path, metadata, coordinates, pairs):
    with np.load(archive_path, allow_pickle=False) as archive:
        if not ARCHIVE_FIELDS <= set(archive.files):
            raise ValueError("Retained D archive is missing actual state/profile/basis/error arrays")
        arrays = {name: np.array(archive[name], copy=True) for name in ARCHIVE_FIELDS}
    expected_shapes = {
        "state_center": (1431,), "state_radius": (), "pair_ordering": (1431, 2),
        "ruler_profile": (27,), "lapse_profile": (27,), "candidate": (2,),
        "eigenvalue_interval": (2,), "excitation_source_profile": (27,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape or arrays[name].dtype.kind not in "fciu":
            raise ValueError(f"Retained {name} shape/dtype is incompatible with D's archive")
        finite(arrays[name], name)
    if arrays["pair_ordering"].dtype.kind not in "iu" or not np.array_equal(arrays["pair_ordering"], pairs):
        raise ValueError("Pair ordering must equal the entire lexicographic wedge^2 C54 basis")
    if arrays["state_center"].dtype != np.dtype(np.complex128):
        raise ValueError("Retain D's actual binary64 complex center without dtype conversion")
    for name in ARCHIVE_FIELDS - {"state_center", "pair_ordering"}:
        if arrays[name].dtype != np.dtype(np.float64):
            raise ValueError(f"Retain D's actual real binary64 {name} without dtype conversion")
    ruler = np.asarray(arrays["ruler_profile"].real, dtype=np.float64)
    lapse = np.asarray(arrays["lapse_profile"].real, dtype=np.float64)
    s, n = np.asarray(arrays["candidate"].real, dtype=np.float64)
    if not (Fraction(999, 1000) <= Fraction.from_float(float(s)) <= Fraction(1001, 1000)
            and Fraction(999, 1000) <= Fraction.from_float(float(n)) <= Fraction(1001, 1000)):
        raise ValueError("Actual retained s,n lie outside D0")
    if float(s) != metadata["candidate"]["s"] or float(n) != metadata["candidate"]["n"]:
        raise ValueError("D JSON and unrounded NPZ field coordinates disagree")
    centre = np.flatnonzero(np.all(coordinates == [1, 1, 1], axis=1))
    if len(centre) != 1:
        raise ValueError("Parent lattice has no unique prescribed interior centre")
    centre = int(centre[0])
    expected_ruler, expected_lapse = np.ones(27), np.ones(27)
    expected_ruler[centre], expected_lapse[centre] = s, n
    if not np.array_equal(ruler, expected_ruler) or not np.array_equal(lapse, expected_lapse):
        raise ValueError("Candidate full profiles do not have the actual clamped 26-site collar")
    original_radius = float(arrays["state_radius"].real)
    if original_radius < 0 or original_radius < metadata["eigenpair_validation"]["vector_error"]:
        raise ValueError("Retained Euclidean state ball is incompatible with the eigenvector error")
    expected_energy = require_interval(metadata["candidate"]["energy"], "ground_energy")
    if not np.array_equal(arrays["eigenvalue_interval"].real, expected_energy):
        raise ValueError("D JSON and NPZ eigenvalue intervals disagree")
    original = np.asarray(arrays["state_center"], dtype=np.complex128)
    original_norm = norm(original)
    if not original_norm > 0:
        raise ValueError("Actual retained pair vector must be nonzero")
    vector = original / original_norm
    return ruler, lapse, vector, arrays, {
        "original_vector_norm_estimate": original_norm,
        "normalization_displacement_estimate": norm(vector - original),
        "original_D_Euclidean_state_radius": original_radius,
        "normalization_propagated_Euclidean_radius_estimate": original_radius + abs(1 - original_norm),
        "radius_meaning": "D Ball operator-norm radius on a column vector, hence Euclidean; phase-adjusted eigenvector error included",
        "normalization_error_certified": False,
        "ground_status": "Ground status from D input certificate; B point assembly unvalidated",
        "normalization_contract": "Normalize the finite retained center once; all contractions still divide by its one common rho=v^dag v",
        "centre_index": centre,
    }


class ProfileOperator:
    """Full profile values plus arbitrary independent B14--B20 directions."""

    def __init__(self, ruler, lapse, coordinates, dirac, pairs, runtime):
        self.ruler, self.lapse = np.asarray(ruler, float), np.asarray(lapse, float)
        self.coordinates, self.dirac, self.pairs, self.runtime = coordinates, dirac, pairs, runtime
        if self.ruler.shape != (27,) or self.lapse.shape != (27,):
            raise ValueError("Full 27-site field profiles are required")
        finite(self.ruler, "ruler")
        finite(self.lapse, "lapse")
        if np.any(self.ruler <= 0) or np.any(self.lapse <= 0):
            raise ValueError("A rebuilt profile left the positive-field domain")
        self.diagnostics = {"arithmetic": "Ordinary binary64/SciPy point arithmetic; all norm/error scales are estimates"}
        self.spin_ruler = np.diag(np.repeat(self.ruler, 2))
        self.l_matrix = np.diag(np.repeat(self.ruler * self.lapse, 2))
        self.packets, self.packet_solvers = [], []
        self.grouped = np.zeros((54, 54), complex)
        self.packet_vacuum = []
        for index, (mass_squared, orientation) in enumerate(zip(MASSES_SQUARED, GRADING)):
            packet = {name: item.value for name, item in
                      polar_jet(Jet(self.ruler, dimension=0), dirac, float(np.sqrt(mass_squared)), uv_r=None).items()}
            self.packets.append(packet)
            self.packet_solvers.append({name: positive_sylvester_solver(packet[name]) for name in ("E0", "E1", "rootE0")})
            difference = packet["E1"] - packet["E0"]
            self.grouped += int(orientation) * difference
            self.packet_vacuum.append(complex(np.trace(self.l_matrix @ difference)))
            for name, product in (("E0", adjoint(packet["A"]) @ packet["A"]),
                                  ("E1", packet["A"] @ adjoint(packet["A"])),
                                  ("rootE0", packet["E0"])):
                self.diagnostics[f"packet{index + 1}_{name}_root_equation_residual_Frobenius"] = norm(packet[name] @ packet[name] - product)
                self.diagnostics[f"packet{index + 1}_{name}_minimum_eigenvalue_estimate"] = float(np.linalg.eigvalsh(packet[name])[0])
            self.diagnostics[f"packet{index + 1}_unitary_defect_Frobenius"] = norm(adjoint(packet["U"]) @ packet["U"] - np.eye(54))
        self.W = real_value(np.trace(self.l_matrix @ self.grouped), "W_grouped", self.diagnostics)
        self.diagnostics["vacuum_packet_trace_cancellation_scale"] = float(sum(abs(x) for x in self.packet_vacuum))
        self.C, self.U = self.packets[0]["rootE0"], self.packets[0]["U"]
        self.b_unit = clean_hermitian(self.C @ self.spin_ruler @ self.C, "b_unit", self.diagnostics)
        self.b_lapse = clean_hermitian(self.C @ self.l_matrix @ self.C, "b_lapse", self.diagnostics)
        self.H0 = clean_hermitian(second_quantized_pair(self.b_unit, pairs), "H0", self.diagnostics)
        self.Hf = clean_hermitian(second_quantized_pair(self.b_lapse, pairs), "Hf", self.diagnostics)
        self.h0_solver = positive_sylvester_solver(self.H0)
        onebody_values = np.linalg.eigvalsh(self.b_unit)
        self.diagnostics["H0_minimum_eigenvalue_estimate"] = float(onebody_values[0] + onebody_values[1])
        self.diagnostics["Sylvester_minimum_denominator_estimate"] = float(2 * (onebody_values[0] + onebody_values[1]))
        self.inverse_ruler = np.diag(np.repeat(self.ruler ** -1, 2))
        self.K = clean_hermitian(self.inverse_ruler @ dirac @ self.inverse_ruler, "K", self.diagnostics)
        self.Y = -(self.K @ self.K) / (2 * 0.5 ** 2)
        self.R = clean_hermitian(expm(self.Y), "R", self.diagnostics)
        self.R_native = np.kron(np.eye(2), self.R)
        self.Wplus = np.vstack([np.eye(54), self.U]) / np.sqrt(2)
        self.Wminus = np.vstack([np.eye(54), -self.U]) / np.sqrt(2)
        self.origins, self.centres, self.centre_weights, self.lines = cell_geometry(coordinates, (3, 3, 3), 1.0)
        if len(self.origins) != 8 or sum(len(row) for row in self.lines) != 64:
            raise ValueError("Full eight-cell/64-ordered-pair geometry is required")
        squared = self.ruler ** 2
        self.Z = np.asarray([np.dot(weight, squared) for weight in self.centre_weights])
        self.distance = np.asarray([[distance * np.dot(weight, squared) for weight, distance in row] for row in self.lines])
        self.G = np.exp(-self.distance ** 2 / 4) / 8
        self.filters, self.fields, self.vertices = self.build_vertices()
        self.V2 = clean_hermitian(self.build_seed(self.vertices, self.Z, self.G), "V2", self.diagnostics)
        self.S2 = clean_hermitian(self.h0_solver(self.V2), "S2", self.diagnostics)
        self.M = clean_hermitian(expm(self.S2), "M", self.diagnostics)
        self.H = clean_hermitian(self.M @ self.Hf @ self.M, "H_exc", self.diagnostics)
        self.diagnostics["S2_unit_lapse_equation_residual_Frobenius"] = norm(self.H0 @ self.S2 + self.S2 @ self.H0 - self.V2)
        self.diagnostics["M_commutator_with_S2_residual_Frobenius"] = norm(self.S2 @ self.M - self.M @ self.S2)
        self.diagnostics["matrix_norm_scales_Frobenius"] = {name: norm(getattr(self, name)) for name in ("H0", "Hf", "V2", "S2", "M", "H", "R", "K")}
        resource_check(runtime)

    def build_vertices(self):
        indices = {tuple(point.astype(int)): index for index, point in enumerate(self.coordinates)}
        all_filters, all_fields, all_vertices = [], [], []
        for valley_index, valley in enumerate(itertools.product((0, 1), repeat=3)):
            filters, fields, vertices = [], [], []
            for cell_index, origin in enumerate(self.origins):
                matrix = np.zeros((4, 108), complex)
                for corner in itertools.product((0, 1), repeat=3):
                    site = indices[tuple(np.asarray(origin) + corner)]
                    sign = (-1) ** int(np.dot(valley, corner)) / 8
                    matrix[:2, 2 * site:2 * site + 2] = sign * np.eye(2)
                    matrix[2:, 54 + 2 * site:54 + 2 * site + 2] = sign * np.eye(2)
                field = matrix @ self.R_native @ self.Wplus
                filters.append(matrix)
                fields.append(field)
                vertices.append([clean_hermitian(adjoint(field) @ gamma @ field,
                                 f"vertex_v{valley_index}_c{cell_index}_g{gamma_index}", self.diagnostics)
                                 for gamma_index, gamma in enumerate([BETA] + ALPHA)])
            all_filters.append(filters)
            all_fields.append(fields)
            all_vertices.append(vertices)
        return all_filters, all_fields, all_vertices

    def build_seed(self, vertices, z, bond):
        """B17; bilinear collection retains every ordered cell pair."""
        seed = np.zeros((1431, 1431), complex)
        for cells in vertices:
            for row, cell in enumerate(cells):
                seed += (-1 / 80) * z[row] ** -3 * pair_product(cell[0], cell[0], self.pairs)
                for direction in range(1, 4):
                    weighted = np.zeros((54, 54), complex)
                    for column, other in enumerate(cells):
                        weighted += (-1 / 400) * bond[row, column] * other[direction]
                    seed += pair_product(cell[direction], weighted, self.pairs)
        return seed

    def seed_direction(self, vertex_dot, z_dot, bond_dot):
        seed_dot = np.zeros((1431, 1431), complex)
        for cells, derivative_cells in zip(self.vertices, vertex_dot):
            for row, (cell, dcell) in enumerate(zip(cells, derivative_cells)):
                contact = (-1 / 80) * self.Z[row] ** -3
                contact_dot = (3 / 80) * self.Z[row] ** -4 * z_dot[row]
                seed_dot += contact_dot * pair_product(cell[0], cell[0], self.pairs)
                seed_dot += contact * (pair_product(dcell[0], cell[0], self.pairs)
                                       + pair_product(cell[0], dcell[0], self.pairs))
                for direction in range(1, 4):
                    weighted, weighted_dot = np.zeros((54, 54), complex), np.zeros((54, 54), complex)
                    for column, (other, dother) in enumerate(zip(cells, derivative_cells)):
                        weighted += (-1 / 400) * self.G[row, column] * other[direction]
                        weighted_dot += (-1 / 400) * (bond_dot[row, column] * other[direction]
                                                     + self.G[row, column] * dother[direction])
                    seed_dot += pair_product(dcell[direction], weighted, self.pairs)
                    seed_dot += pair_product(cell[direction], weighted_dot, self.pairs)
        return seed_dot

    def direction(self, ruler_direction, lapse_direction, label):
        """B14--B20 at the same base; no central-profile trace shortcut."""
        ppsi, pn = np.asarray(ruler_direction, float), np.asarray(lapse_direction, float)
        if ppsi.shape != (27,) or pn.shape != (27,):
            raise ValueError("Each source direction must specify all 27 site values")
        finite(ppsi, "ruler direction")
        finite(pn, "clock direction")
        diagnostics = {"label": label, "assembly_error_certified": False}
        spin_dot = np.diag(np.repeat(ppsi, 2))
        l_dot = np.diag(np.repeat(self.lapse * ppsi + self.ruler * pn, 2))
        packet_derivatives, grouped_dot, packet_work = [], np.zeros((54, 54), complex), []
        for index, (packet, solvers, mass_squared, orientation) in enumerate(zip(self.packets, self.packet_solvers, MASSES_SQUARED, GRADING)):
            a_dot = (-float(np.sqrt(mass_squared)) * np.diag(np.repeat(self.ruler ** -2 * ppsi, 2))
                     - 3j * np.diag(np.repeat(self.ruler ** -4 * ppsi, 2)) @ self.dirac)
            rhs0 = adjoint(a_dot) @ packet["A"] + adjoint(packet["A"]) @ a_dot
            rhs1 = a_dot @ adjoint(packet["A"]) + packet["A"] @ adjoint(a_dot)
            e0_dot = clean_hermitian(solvers["E0"](rhs0), f"packet{index + 1}_E0_dot", diagnostics)
            e1_dot = clean_hermitian(solvers["E1"](rhs1), f"packet{index + 1}_E1_dot", diagnostics)
            inverse0 = np.linalg.inv(packet["E0"])
            u_dot = a_dot @ inverse0 - packet["U"] @ e0_dot @ inverse0
            c_dot = clean_hermitian(solvers["rootE0"](e0_dot), f"packet{index + 1}_C_dot", diagnostics)
            packet_derivatives.append({"A": a_dot, "E0": e0_dot, "E1": e1_dot, "U": u_dot, "C": c_dot})
            difference_dot = e1_dot - e0_dot
            grouped_dot += int(orientation) * difference_dot
            packet_work.append(complex(np.trace(l_dot @ (packet["E1"] - packet["E0"]) + self.l_matrix @ difference_dot)))
            diagnostics[f"packet{index + 1}_E0_dot_equation_residual_Frobenius"] = norm(packet["E0"] @ e0_dot + e0_dot @ packet["E0"] - rhs0)
            diagnostics[f"packet{index + 1}_E1_dot_equation_residual_Frobenius"] = norm(packet["E1"] @ e1_dot + e1_dot @ packet["E1"] - rhs1)
            diagnostics[f"packet{index + 1}_C_dot_equation_residual_Frobenius"] = norm(packet["rootE0"] @ c_dot + c_dot @ packet["rootE0"] - e0_dot)
        w_dot = real_value(np.trace(l_dot @ self.grouped + self.l_matrix @ grouped_dot), "w_grouped", diagnostics)
        diagnostics["packet_work_before_grouping"] = [{"real": value.real, "imaginary": value.imag} for value in packet_work]
        diagnostics["vacuum_derivative_cancellation_scale"] = float(sum(abs(value) for value in packet_work))
        light_dot = packet_derivatives[0]
        c_dot, u_dot = light_dot["C"], light_dot["U"]
        h0_dot = clean_hermitian(second_quantized_pair(c_dot @ self.spin_ruler @ self.C
                        + self.C @ spin_dot @ self.C + self.C @ self.spin_ruler @ c_dot, self.pairs), "H0_dot", diagnostics)
        hf_dot = clean_hermitian(second_quantized_pair(c_dot @ self.l_matrix @ self.C
                        + self.C @ l_dot @ self.C + self.C @ self.l_matrix @ c_dot, self.pairs), "Hf_dot", diagnostics)
        wplus_dot = np.vstack([np.zeros((54, 54)), u_dot]) / np.sqrt(2)
        wminus_dot = np.vstack([np.zeros((54, 54)), -u_dot]) / np.sqrt(2)
        diagnostics["Wplus_tangent_unitarity_defect_Frobenius"] = norm(adjoint(wplus_dot) @ self.Wplus + adjoint(self.Wplus) @ wplus_dot)
        diagnostics["Wminus_tangent_unitarity_defect_Frobenius"] = norm(adjoint(wminus_dot) @ self.Wminus + adjoint(self.Wminus) @ wminus_dot)
        k_dot = -self.inverse_ruler @ spin_dot @ self.K - self.K @ spin_dot @ self.inverse_ruler
        y_dot = -(self.K @ k_dot + k_dot @ self.K) / (2 * 0.5 ** 2)
        r_dot = clean_hermitian(expm_frechet(self.Y, y_dot, compute_expm=False, check_finite=True), "R_dot", diagnostics)
        r_native_dot = np.kron(np.eye(2), r_dot)
        field_source = 2 * self.ruler * ppsi
        z_dot = np.asarray([np.dot(weight, field_source) for weight in self.centre_weights])
        distance_dot = np.asarray([[distance * np.dot(weight, field_source) for weight, distance in row] for row in self.lines])
        bond_dot = -self.distance * distance_dot * self.G / 2
        vertex_dot = []
        for valley_index, (filters, fields) in enumerate(zip(self.filters, self.fields)):
            cells = []
            for cell_index, (matrix, field) in enumerate(zip(filters, fields)):
                field_dot = matrix @ r_native_dot @ self.Wplus + matrix @ self.R_native @ wplus_dot
                cells.append([clean_hermitian(adjoint(field_dot) @ gamma @ field + adjoint(field) @ gamma @ field_dot,
                              f"vertex_dot_v{valley_index}_c{cell_index}_g{gamma_index}", diagnostics)
                              for gamma_index, gamma in enumerate([BETA] + ALPHA)])
            vertex_dot.append(cells)
        seed_dot = clean_hermitian(self.seed_direction(vertex_dot, z_dot, bond_dot), "V2_dot", diagnostics)
        denominator_rhs = seed_dot - h0_dot @ self.S2 - self.S2 @ h0_dot
        s2_dot = clean_hermitian(self.h0_solver(denominator_rhs), "S2_dot", diagnostics)
        m_dot = clean_hermitian(expm_frechet(self.S2, s2_dot, compute_expm=False, check_finite=True), "M_dot", diagnostics)
        h_dot = clean_hermitian(m_dot @ self.Hf @ self.M + self.M @ hf_dot @ self.M
                               + self.M @ self.Hf @ m_dot, "H_exc_dot", diagnostics)
        diagnostics["S2_dot_unit_lapse_equation_residual_Frobenius"] = norm(self.H0 @ s2_dot + s2_dot @ self.H0 - denominator_rhs)
        diagnostics["M_dot_Frechet_commutator_residual_Frobenius"] = norm(self.S2 @ m_dot - m_dot @ self.S2 - (self.M @ s2_dot - s2_dot @ self.M))
        diagnostics["R_dot_Frechet_commutator_residual_Frobenius"] = norm(self.Y @ r_dot - r_dot @ self.Y - (self.R @ y_dot - y_dot @ self.R))
        diagnostics["source_norm_scales_Frobenius"] = {"h_dot": norm(h_dot), "w_dot_absolute": abs(w_dot),
                                "V2_dot": norm(seed_dot), "H0_dot": norm(h0_dot), "Hf_dot": norm(hf_dot),
                                "S2_dot": norm(s2_dot), "M_dot": norm(m_dot), "R_dot": norm(r_dot), "U_dot": norm(u_dot)}
        if not np.any(ppsi):
            diagnostics["pure_lapse_affinity_zero_derivative_estimates"] = {
                "V2_dot": norm(seed_dot), "H0_dot": norm(h0_dot), "S2_dot": norm(s2_dot), "M_dot": norm(m_dot), "R_dot": norm(r_dot),
            }
        resource_check(self.runtime)
        return {"w": w_dot, "h": h_dot, "U_dot": u_dot, "Wplus_dot": wplus_dot, "Wminus_dot": wminus_dot,
                "H0_dot": h0_dot, "V2_dot": seed_dot, "S2_dot": s2_dot, "M_dot": m_dot,
                "geometry_dot": {"Z": z_dot, "distance": distance_dot, "G": bond_dot},
                "packet_derivatives": packet_derivatives, "diagnostics": diagnostics}


def dilation_geometry(coordinates, ruler, lapse):
    """B4/B6 and five independently declared full-site source probes."""
    indices = {tuple(point.astype(int)): index for index, point in enumerate(coordinates)}
    centre = indices[(1, 1, 1)]
    central = np.zeros(27)
    central[centre] = 1
    faces = np.asarray([int(np.sum(np.abs(point - [1, 1, 1])) == 1) for point in coordinates], float)
    collar = 1 - central
    v0 = np.zeros((27, 27), float)
    for axis in range(3):
        forward = np.zeros((27, 27), float)
        for row, point in enumerate(coordinates.astype(int)):
            neighbor = point.copy()
            neighbor[axis] += 1
            if tuple(neighbor) in indices:
                forward[row, indices[tuple(neighbor)]] = 1
        derivative = (forward - forward.T) / 2
        v0 += np.diag(coordinates[:, axis] - 1) @ derivative
    v = v0 - np.diag(v0 @ np.ones(27))
    asp = np.kron((v - v.T) / 2, np.eye(2))
    s, n = float(ruler[centre]), float(lapse[centre])
    zero = np.zeros(27)
    profiles = {
        "dilation": (ruler / 2 - (s - 1) * faces / 2, -(n - 1) * faces / 2),
        "collar_ruler": (collar, zero.copy()),
        "face_ruler": (faces, zero.copy()),
        "face_clock": (zero.copy(), faces),
        "central_ruler": (central, zero.copy()),
        "central_clock": (zero.copy(), central),
    }
    diagnostics = {
        "V_constant_annihilation_defect": norm(v @ np.ones(27)),
        "V_central_indicator_plus_half_faces_defect": norm(v @ central + faces / 2),
        "dilation_ruler_B5_vs_B6_defect": norm(v @ ruler + ruler / 2 - profiles["dilation"][0]),
        "dilation_lapse_B5_vs_B6_defect": norm(v @ lapse - profiles["dilation"][1]),
        "Asp_antiHermitian_defect": norm(asp + adjoint(asp)),
        "six_face_sites": coordinates[faces != 0].astype(int).tolist(),
        "collar_sites": coordinates[collar != 0].astype(int).tolist(),
    }
    if int(np.sum(faces)) != 6 or int(np.sum(collar)) != 26:
        raise ValueError("The reviewed six-face/26-site collar allocation changed")
    return v, asp, profiles, faces, collar, centre, diagnostics


def geometry_checks(model, derivatives, s):
    """B21--B22 checks only; never used to build a displaced profile."""
    ih = np.asarray([1 / 8, 3 / 16, 7 / 24, 15 / 32])
    fh = np.asarray([3 / 8, 7 / 16, 11 / 24, 11 / 32])
    h = np.asarray([[sum(a != b for a, b in zip(first, second)) for second in model.origins] for first in model.origins])
    z = (7 + s * s) / 8
    expected_distance = np.sqrt(h) * (1 + (s * s - 1) * ih[h])
    expected_bond = np.exp(-expected_distance ** 2 / 4) / 8
    expected_z_dot = {
        "dilation": z - 3 * (s - 1) / 8, "collar_ruler": 7 / 4,
        "face_ruler": 3 / 4, "central_ruler": s / 4, "face_clock": 0, "central_clock": 0,
    }
    distance_factors = {
        "dilation": 1 + (s * s - 1) * ih[h] - (s - 1) * fh[h],
        "collar_ruler": 2 * (1 - ih[h]), "face_ruler": 2 * fh[h],
        "central_ruler": 2 * s * ih[h], "face_clock": np.zeros_like(h), "central_clock": np.zeros_like(h),
    }
    rows = {}
    for label, derivative in derivatives.items():
        geometry = derivative["geometry_dot"]
        predicted_dot = np.sqrt(h) * distance_factors[label]
        rows[label] = {
            "Z_dot_B22_absolute_error_norm": norm(geometry["Z"] - expected_z_dot[label]),
            "distance_dot_B22_absolute_error_norm": norm(geometry["distance"] - predicted_dot),
            "G_dot_B20_absolute_error_norm": norm(geometry["G"] + expected_distance * predicted_dot * expected_bond / 2),
        }
    return {"usage": "Independent point arithmetic geometry checks; central shortcuts never supply actual profile geometry",
            "Z_B21_absolute_error_norm": norm(model.Z - z),
            "distance_B21_absolute_error_norm": norm(model.distance - expected_distance),
            "G_B21_absolute_error_norm": norm(model.G - expected_bond), "probes": rows}


def normalized_mean(vector, matrix, rho, label, diagnostics):
    return real_value(np.vdot(vector, matrix @ vector) / rho, label, diagnostics)


def state_observables(model, derivatives, vector, j2):
    """Independent definitions B9--B13; no residual reconstruction."""
    diagnostics = {}
    rho = real_value(np.vdot(vector, vector), "rho", diagnostics)
    if rho <= 0:
        raise ValueError("Pair vector normalization denominator must be positive")
    rayleigh = normalized_mean(vector, model.H, rho, "Rayleigh", diagnostics)
    residual = model.H @ vector - rayleigh * vector
    j_vector = j2 @ vector
    commutator_action = j2 @ (model.H @ vector) - model.H @ j_vector
    kappa = real_value(np.vdot(vector, commutator_action) / rho, "kappa_commutator", diagnostics)
    kappa_residual = float(-2 * np.vdot(j_vector, residual).real / rho)
    bound = float(2 * norm(j_vector) * norm(residual) / rho)
    probes = {}
    for label, derivative in derivatives.items():
        matter = normalized_mean(vector, derivative["h"], rho, label + "_matter_work", diagnostics)
        probes[label] = {"vacuum_work": derivative["w"], "matter_work": matter,
                         "F_fixed_amplitudes": derivative["w"] + matter}
    centre = int(np.flatnonzero(np.all(model.coordinates == [1, 1, 1], axis=1))[0])
    s, n = float(model.ruler[centre]), float(model.lapse[centre])
    tb, tf, cf = (probes[name]["F_fixed_amplitudes"] for name in ("collar_ruler", "face_ruler", "face_clock"))
    r1, r2 = (probes[name]["F_fixed_amplitudes"] for name in ("central_clock", "central_ruler"))
    d_vac = probes["dilation"]["vacuum_work"]
    dilation_matter = probes["dilation"]["matter_work"]
    d_mat = dilation_matter - kappa
    d_bdy = (s - 1) * tf / 2 + (n - 1) * cf / 2
    r_stat = -s * r2 / 2 + kappa
    collar_work = tb / 2
    rhs = math.fsum((d_vac, d_mat, d_bdy, r_stat))
    return {
        "rho_common_denominator": rho, "Rayleigh_excitation_energy": rayleigh,
        "Rayleigh_residual_norm_estimate": norm(residual) / math.sqrt(rho),
        "J2_vector_norm_estimate": norm(j_vector) / math.sqrt(rho),
        "kappa_commutator_action": kappa, "kappa_eigenpair_residual_contraction": kappa_residual,
        "kappa_two_forms_difference": kappa - kappa_residual,
        "kappa_B9_absolute_bound_estimate": bound,
        "kappa_bound_arithmetic_certified": False, "probe_components": probes,
        "T_B": tb, "T_F": tf, "C_F": cf, "R1": r1, "R2": r2,
        "finite_identity": {
            "Q_a": collar_work, "D_vac": d_vac, "dilation_matter_fixed_amplitudes": dilation_matter,
            "kappa": kappa, "D_mat": d_mat, "D_bdy": d_bdy, "R_stat": r_stat,
            "independently_defined_RHS_sum": rhs, "Delta_B": collar_work - rhs,
            "identity_certified": False,
            "left_side_meaning": "Prescribed-ruler collar work in the natural transported sea chart",
        },
        "imaginary_defects": diagnostics,
    }, residual


def input_point_consistency(model, observable, metadata, normalization):
    """Reported certificate versus point rebuild; no new certification."""
    delta = normalization["normalization_propagated_Euclidean_radius_estimate"]
    lower, upper = require_interval(metadata["candidate"]["energy"], "ground_energy")
    matrix_scale = max(1.0, norm(model.H))
    # Author/reviewer clarification: the Rayleigh shift for a normalized
    # delta-close eigenvector is <=2K delta^2 in exact assembly.  Here both
    # K and the normalization propagation are estimates, so this widening
    # is not certified and does not strengthen D's original interval.
    state_rayleigh_shift = 2 * matrix_scale * delta ** 2
    estimated_interval = (lower - state_rayleigh_shift, upper + state_rayleigh_shift)
    rayleigh = observable["Rayleigh_excitation_energy"]
    distance = max(estimated_interval[0] - rayleigh, rayleigh - estimated_interval[1], 0.0)
    unit_roundoff = float(np.finfo(np.float64).eps / 2)
    engineering_tolerance = REBUILD_ROUNDOFF_SCALE_MULTIPLIER * unit_roundoff * matrix_scale
    original_residual = metadata["eigenpair_validation"]["eigenpair_residual"]
    residual_disagreement = max(0.0, observable["Rayleigh_residual_norm_estimate"] - original_residual)
    conspicuous_disagreement = distance > engineering_tolerance or residual_disagreement > engineering_tolerance
    return {
        "ground_status": "Ground status from D input certificate; B point assembly unvalidated",
        "original_D_ground_interval": [lower, upper],
        "estimated_same_vector_Rayleigh_interval": list(estimated_interval),
        "estimated_state_error_Rayleigh_shift_2K_delta_squared": state_rayleigh_shift,
        "Rayleigh_distance_to_estimated_interval": distance,
        "B_point_residual_excess_over_D_residual_bound": residual_disagreement,
        "engineering_stop_tolerance": engineering_tolerance,
        "engineering_tolerance_recipe": "2^20*u*max(1,||H_exc||F); predeclared, not fitted, not an assembly bound",
        "point_rebuild_consistent_under_engineering_stop": not conspicuous_disagreement,
        "comparison_certified": False,
        "central_residual_comparisons": {
            "R1": {"B_point": observable["R1"], "D_retained": metadata["center_residual"][0]},
            "R2": {"B_point": observable["R2"], "D_retained": metadata["center_residual"][1]},
        },
    }


def wedge_matrix(unitary, pairs):
    """Group lift wedge^2(U); not the additive Lie-algebra lift dGamma2."""
    i, j = pairs[:, 0, None], pairs[:, 1, None]
    k, ell = pairs[None, :, 0], pairs[None, :, 1]
    return unitary[i, k] * unitary[j, ell] - unitary[i, ell] * unitary[j, k]


def energy_of(model, vector):
    rho = float(np.vdot(vector, vector).real)
    if rho <= 0:
        raise ValueError("Finite-path pair state must be nonzero")
    return model.W + float(np.vdot(vector, model.H @ vector).real / rho)


def symmetric_profile_study(model, derivatives, profiles, vector, steps):
    """All displaced matrices fully rebuilt; fixed natural-chart amplitudes."""
    unit_roundoff = float(np.finfo(np.float64).eps / 2)
    rho = float(np.vdot(vector, vector).real)
    study = {}
    for label in PROBE_NAMES:
        ruler_dot, lapse_dot = profiles[label]
        analytic = derivatives[label]
        rows = []
        for rational_step in steps:
            step = float(rational_step)
            plus = ProfileOperator(model.ruler + step * ruler_dot, model.lapse + step * lapse_dot,
                                   model.coordinates, model.dirac, model.pairs, model.runtime)
            minus = ProfileOperator(model.ruler - step * ruler_dot, model.lapse - step * lapse_dot,
                                    model.coordinates, model.dirac, model.pairs, model.runtime)
            w_fd = (plus.W - minus.W) / (2 * step)
            h_fd = (plus.H - minus.H) / (2 * step)
            matter_fd = float(np.vdot(vector, h_fd @ vector).real / rho)
            analytic_matter = float(np.vdot(vector, analytic["h"] @ vector).real / rho)
            rows.append({
                "step": str(rational_step), "w_symmetric_difference": w_fd,
                "matter_symmetric_difference": matter_fd, "F_symmetric_difference": w_fd + matter_fd,
                "w_absolute_derivative_error": abs(w_fd - analytic["w"]),
                "h_absolute_derivative_error_Frobenius": norm(h_fd - analytic["h"]),
                "matter_absolute_derivative_error": abs(matter_fd - analytic_matter),
                "heuristic_W_subtraction_scale": unit_roundoff * (abs(plus.W) + abs(minus.W)) / (2 * step),
                "heuristic_H_subtraction_scale_Frobenius": unit_roundoff * (norm(plus.H) + norm(minus.H)) / (2 * step),
                "plus_W_cancellation_scale": plus.diagnostics["vacuum_packet_trace_cancellation_scale"],
                "minus_W_cancellation_scale": minus.diagnostics["vacuum_packet_trace_cancellation_scale"],
                "plus_base_equation_and_symmetry_diagnostics": plus.diagnostics,
                "minus_base_equation_and_symmetry_diagnostics": minus.diagnostics,
                "arithmetic_error_scales_certified": False,
            })
            del plus, minus, h_fd
            resource_check(model.runtime)
        for previous, current in zip(rows, rows[1:]):
            previous_error = previous["h_absolute_derivative_error_Frobenius"]
            current["h_error_ratio_to_previous_step"] = (
                current["h_absolute_derivative_error_Frobenius"] / previous_error if previous_error > 0 else None
            )
        study[label] = rows
        print(f"B rebuilt symmetric study complete for {label} at {len(steps)} steps", flush=True)
    return {
        "fixed_amplitude_chart": "Same natural Wplus chart and same pair amplitudes; no state re-minimization",
        "error_interpretation": "Absolute norm discrepancies and heuristic subtraction scales; convergence/plateaus are not certificates",
        "operator_assembly_absolute_error_bound": None, "assembly_error_certified": False,
        "probes": study,
        "collar_work_independent_step_measurements": [
            {"step": row["step"], "Q_a_half_collar_c_derivative": row["F_symmetric_difference"] / 2}
            for row in study["collar_ruler"]
        ],
    }


def offset_and_orbital_studies(model, vector, g, v, faces, collar, steps, observables):
    """Independent B12 offset work and actual B5/B23 orbital path."""
    centre = int(np.flatnonzero(np.all(model.coordinates == [1, 1, 1], axis=1))[0])
    s, n = model.ruler[centre], model.lapse[centre]
    sigma_dot, tau_dot = -(s - 1) * faces / 2, -(s + n - 2) * faces / 2
    offset_rows, orbital_rows = [], []
    for rational_step in steps:
        step = float(rational_step)
        offset_energy, orbital_energy, orbital_norms = [], [], []
        for sign in (1, -1):
            t = sign * step
            ruler, lapse = model.ruler.copy(), model.lapse.copy()
            mask = collar != 0
            ruler[mask] = 1 + t * sigma_dot[mask]
            lapse[mask] = (1 + t * tau_dot[mask]) / (1 + t * sigma_dot[mask])
            displaced = ProfileOperator(ruler, lapse, model.coordinates, model.dirac, model.pairs, model.runtime)
            offset_energy.append(energy_of(displaced, vector))
            del displaced
            ruler_path = np.exp(t / 2) * (expm(t * v) @ model.ruler)
            lapse_path = expm(t * v) @ model.lapse
            state_path = wedge_matrix(expm(t * g), model.pairs) @ vector
            displaced = ProfileOperator(ruler_path, lapse_path, model.coordinates, model.dirac, model.pairs, model.runtime)
            orbital_energy.append(energy_of(displaced, state_path))
            orbital_norms.append(norm(state_path))
            del displaced
            resource_check(model.runtime)
        boundary_difference = -(offset_energy[0] - offset_energy[1]) / (2 * step)
        orbital_difference = (orbital_energy[0] - orbital_energy[1]) / (2 * step)
        finite_identity = observables["finite_identity"]
        orbital_target = finite_identity["D_vac"] + finite_identity["D_mat"]
        offset_rows.append({"step": str(rational_step), "negative_offset_path_derivative": boundary_difference,
                            "independent_D_bdy": finite_identity["D_bdy"],
                            "absolute_error": abs(boundary_difference - finite_identity["D_bdy"]),
                            "heuristic_subtraction_scale": float(np.finfo(np.float64).eps / 2) * sum(map(abs, offset_energy)) / (2 * step)})
        orbital_rows.append({"step": str(rational_step), "actual_B5_B23_path_derivative": orbital_difference,
                             "independent_D_vac_plus_D_mat": orbital_target,
                             "absolute_error": abs(orbital_difference - orbital_target),
                             "displaced_pair_norm_estimates": orbital_norms,
                             "heuristic_subtraction_scale": float(np.finfo(np.float64).eps / 2) * sum(map(abs, orbital_energy)) / (2 * step)})
        print(f"B independent offset/orbital paths complete at step {rational_step}", flush=True)
    return {"offset_boundary_work": offset_rows, "orbital_sign_B23": orbital_rows,
            "orbital_group_lift": "wedge^2(exp(tg)); ordered-pair determinants, not dGamma2(exp(tg))",
            "certified": False}


def save_result(directory, result):
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "b_dilation_diagnostic.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(directory / "b_dilation_diagnostic.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-result", type=Path, required=True)
    parser.add_argument("--candidate-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--steps", nargs="+", default=["1/1024", "1/2048", "1/4096"])
    parser.add_argument("--companion", action="store_true", help="Also evaluate the predeclared nonstationary companion from B's specification")
    args = parser.parse_args()
    steps = [Fraction(item) for item in args.steps]
    if len(steps) < 3 or any(step <= 0 for step in steps) or any(a <= b for a, b in zip(steps, steps[1:])):
        raise ValueError("Use at least three positive strictly decreasing derivative step sizes")
    metadata = load_candidate_metadata(args.candidate_result, args.candidate_archive)
    if os.environ.get("QI2_GPU_UUID", args.gpu_uuid) != args.gpu_uuid:
        raise RuntimeError("CLI GPU UUID differs from the freshly selected device")
    os.environ["QI2_GPU_UUID"] = args.gpu_uuid
    # Host-only imports and JSON/existence checks complete.  The final
    # fresh query -> this-process allocation -> sole-PID admission is now
    # immediately before numerical input validation and operator work.
    runtime = remote_execution(max_threads=16)
    started = time.perf_counter()
    np.seterr(over="raise", invalid="raise", divide="raise", under="ignore")
    if not np.array_equal(MASSES_SQUARED, [0.25, 0.75, 1.0, 1.5]) or not np.array_equal(GRADING, [1, -1, -1, 1]):
        raise ValueError("Shared original-law inventory is incompatible with the reviewed B1 law")
    coordinates, dirac = sine_lattice((3, 3, 3), 1, periodic=False)
    pairs = pairs_for(54)
    if dirac.shape != (54, 54) or pairs.shape != (1431, 2):
        raise ValueError("The full original-A finite parent or sector dimension changed")
    ruler, lapse, vector, retained_arrays, normalization = load_candidate_arrays(
        args.candidate_archive, metadata, coordinates, pairs)
    v, asp, profiles, faces, collar, centre, dilation_data = dilation_geometry(coordinates, ruler, lapse)
    model = ProfileOperator(ruler, lapse, coordinates, dirac, pairs, runtime)
    derivatives = {}
    for name in PROBE_NAMES:
        derivatives[name] = model.direction(*profiles[name], name)
        print(f"B independent full-profile first derivative complete: {name}", flush=True)
    connection_diagnostics = {}
    g_raw = (asp + adjoint(model.U) @ asp @ model.U - adjoint(model.U) @ derivatives["dilation"]["U_dot"]) / 2
    g = clean_antihermitian(g_raw, "g", connection_diagnostics)
    j2 = clean_antihermitian(second_quantized_pair(g, pairs), "J2", connection_diagnostics)
    observables, residual = state_observables(model, derivatives, vector, j2)
    consistency = input_point_consistency(model, observables, metadata, normalization)
    result = {
        "scope": "Original-A open3^3 finite transported-chart collar/dilation diagnostic at an actual retained D load-one pair vector",
        "parameters": PARAMETERS, "uv_r": None, "runtime": runtime,
        "input_candidate_result": str(args.candidate_result), "input_candidate_archive": str(args.candidate_archive),
        "D_input_certificate": metadata, "D_input_normalization": normalization,
        "D_root_certificate_status_preserved": metadata["status"],
        "D_root_certificate_passed_preserved": metadata["certificate_passed"],
        "state_space": "Entire wedge^2 C54, charge+2/excitation number2, dimension1431; ph/hh are outside this exact reducing sector",
        "reference": "Both Wplus/Wminus tangents retained; full pp seed external legs use dotWplus and no additional sea contraction",
        "dilation_geometry": dilation_data,
        "profiles": {name: {"p_psi": psi.tolist(), "p_N": clock.tolist()} for name, (psi, clock) in profiles.items()},
        "base_operator_diagnostics": model.diagnostics,
        "base_W_grouped": model.W,
        "base_W_central_shortcut_check_only": {
            "central_clamped_value": float((lapse[centre] * ruler[centre] - 1) * np.trace(model.grouped[2 * centre:2 * centre + 2, 2 * centre:2 * centre + 2]).real),
            "usage": "Comparison at the central clamped base only; never used for any collar, face, dilation or displaced-profile derivative",
        },
        "full_profile_derivative_diagnostics": {name: item["diagnostics"] for name, item in derivatives.items()},
        "geometry_B21_B22_checks": geometry_checks(model, derivatives, ruler[centre]),
        "native_chart_connection_diagnostics": connection_diagnostics,
        "ground_vector_observables": observables, "candidate_point_consistency": consistency,
        "arithmetic": {"kind": "Ordinary point arithmetic", "binary64_unit_roundoff": float(np.finfo(np.float64).eps / 2),
                       "symmetry_contract": "Raw defects recorded before each explicit Hermitian/skew projection",
                       "state_contractions": "All means, Rayleigh energy, kappa and residual bound divide by one rho=v^dag v",
                       "assembly_error_certified": False, "operator_assembly_absolute_error_bound": None},
        "evaluation_complete": False, "derivative_step_study_complete": False,
        "assembly_error_certified": False, "uniform_root_box_certified": False, "finite_identity_certified": False,
        "disposition": "Runtime partial: independent point observables formed; rebuilt derivative checks pending",
        "caveat": "Ordinary B operator assembly and collar derivatives are unvalidated. D's eigenpair/root statuses are inputs, not new B certificates. No isolated charge, current-r3 transfer, full-neighborhood, physical ruler, stellar or continuum-dilation conclusion is supplied.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_values = {
        "original_D_state_center": retained_arrays["state_center"], "original_D_state_radius": retained_arrays["state_radius"],
        "pair_ordering": pairs, "ruler_profile": ruler, "lapse_profile": lapse, "normalized_pair_vector": vector,
        "H_exc_rebuilt": model.H, "W_grouped_rebuilt": np.array(model.W), "Rayleigh_residual_rebuilt": residual,
        "H0_unit_lapse": model.H0, "Hf_actual_lapse": model.Hf, "V2_complete_pp": model.V2,
        "S2_unit_lapse_denominator": model.S2, "expS2": model.M,
        "g_native_chart": g, "J2": j2, "V_scalar_site_dilation": v, "Asp": asp,
        "Z_all_cells": model.Z, "distance_all_ordered_cell_pairs": model.distance, "G_all_ordered_cell_pairs": model.G,
    }
    for name, derivative in derivatives.items():
        archive_values[name + "_h_exc_first"] = derivative["h"]
        archive_values[name + "_w_first"] = np.array(derivative["w"])
        archive_values[name + "_U_first"] = derivative["U_dot"]
        archive_values[name + "_Wplus_first"] = derivative["Wplus_dot"]
        archive_values[name + "_Wminus_first"] = derivative["Wminus_dot"]
    np.savez_compressed(args.output_dir / "b_dilation_diagnostic.npz", **archive_values)
    result["operator_and_state_archive"] = "b_dilation_diagnostic.npz"
    if not consistency["point_rebuild_consistent_under_engineering_stop"]:
        result["disposition"] = "Point rebuild inconsistent/unvalidated under the predeclared engineering stop; derivative study not executed"
        save_result(args.output_dir, result)
        raise RuntimeError(result["disposition"])
    save_result(args.output_dir, result)
    result["rebuilt_symmetric_derivative_step_study"] = symmetric_profile_study(model, derivatives, profiles, vector, steps)
    save_result(args.output_dir, result)
    result["independent_offset_and_orbital_paths"] = offset_and_orbital_studies(model, vector, g, v, faces, collar, steps, observables)
    if args.companion:
        rho = float(np.vdot(vector, vector).real)
        tangent = j2 @ vector
        transverse = tangent - vector * (np.vdot(vector, tangent) / rho)
        transverse_norm = norm(transverse)
        if transverse_norm == 0:
            result["predeclared_companion"] = {"available": False, "reason": "The prescribed projected J2 tangent vanishes"}
        else:
            companion = (vector / math.sqrt(rho) + 0.001 * transverse / transverse_norm) / math.sqrt(1 + 0.001 ** 2)
            companion_observables, _ = state_observables(model, derivatives, companion, j2)
            result["predeclared_companion"] = {
                "available": True, "zeta": "1/1000", "state": "Same fields and operators; predeclared nonstationary companion, not D's certified ground vector",
                "observables": companion_observables,
                "orbital_and_offset_paths": offset_and_orbital_studies(model, companion, g, v, faces, collar, steps, companion_observables),
            }
            np.savez_compressed(args.output_dir / "b_dilation_companion.npz", state=companion, pair_ordering=pairs)
    resource_check(runtime)
    result["evaluation_complete"] = True
    result["derivative_step_study_complete"] = True
    result["disposition"] = "Completed ordinary finite point identity and rebuilt derivative/path diagnostics; numerical values and all new operator assembly remain uncertified"
    result["elapsed_seconds"] = time.perf_counter() - started
    save_result(args.output_dir, result)
    print(json.dumps({"evaluation_complete": True, "finite_identity": observables["finite_identity"],
                      "assembly_error_certified": False, "uniform_root_box_certified": False,
                      "output": str(args.output_dir / "b_dilation_diagnostic.json")}, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
