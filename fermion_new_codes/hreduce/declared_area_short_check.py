"""Bounded remote check implementing the reviewed reviewer B area-law SPEC.

The integral and finite-chain/count controls are specified in
declared_area_law.tex.  No tilted or spherical three-dimensional cut is run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
import warnings

import numpy as np
import scipy
from scipy.integrate import IntegrationWarning, quad
from scipy.linalg import eigh, eigvalsh
from scipy.special import ellipk, ellipkm1, xlogy


def chain_formula(mass: float) -> float:
    mass2 = mass*mass
    x = 1.0/(1.0+mass2)
    y = mass2/(1.0+mass2)
    kx = ellipkm1(y) if y < 0.5 else ellipk(x)
    ky = ellipkm1(x) if x < 0.5 else ellipk(y)
    logx = -math.log1p(mass2)
    logy = 2.0*math.log(mass)-math.log1p(mass2)
    return float((math.log(4.0)-(logx+logy)/2.0
                  +(x-y)*2.0*kx*ky/math.pi)/3.0)


def density(w: float) -> float:
    # 1-w(2-w)=(1-w)^2, avoiding loss of the small complement.
    return float(2.0*ellipkm1((w-1.0)**2)/math.pi**2)


def integrate_segments(function, tolerance: float) -> dict:
    values = []
    errors = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IntegrationWarning)
        for left, right in ((0.0, 1.0), (1.0, 2.0)):
            value, error = quad(function, left, right, epsabs=tolerance,
                                epsrel=tolerance, limit=300)
            values.append(value)
            errors.append(error)
    return {"value": sum(values), "reported_error": sum(errors),
            "segment_values": values, "segment_errors": errors,
            "tolerance": tolerance,
            "warnings": [str(item.message) for item in caught]}


def entropy(covariance: np.ndarray) -> dict:
    eigenvalues = eigvalsh((covariance+covariance.conj().T)/2.0,
                          check_finite=True)
    if eigenvalues.min() < -1e-11 or eigenvalues.max() > 1.0+1e-11:
        raise ValueError("Covariance spectrum exceeds numerical roundoff")
    clipped = np.clip(eigenvalues, 0.0, 1.0)
    endpoint_mask = (clipped < 1e-14) | (clipped > 1.0-1e-14)
    omitted_entropy = float(np.sum(-xlogy(clipped[endpoint_mask],
                                         clipped[endpoint_mask])
                                   -xlogy(1.0-clipped[endpoint_mask],
                                          1.0-clipped[endpoint_mask])))
    clipped[clipped < 1e-14] = 0.0
    clipped[clipped > 1.0-1e-14] = 1.0
    value = float(np.sum(-xlogy(clipped, clipped)
                         -xlogy(1.0-clipped, 1.0-clipped)))
    return {"entropy": value, "minimum_eigenvalue": float(eigenvalues.min()),
            "maximum_eigenvalue": float(eigenvalues.max()),
            "endpoint_omitted_entropy": omitted_entropy}


def chain_covariance(onsite: np.ndarray, forward: np.ndarray, length: int,
                     *, periodic: bool = False) -> dict:
    components = onsite.shape[0]
    hamiltonian = np.kron(np.eye(length), onsite)
    shift = np.diag(np.ones(length-1), 1)
    if periodic:
        shift[-1, 0] = 1.0
    hamiltonian = (hamiltonian + np.kron(shift, forward)
                   + np.kron(shift.T, forward.conj().T))
    if np.max(np.abs(hamiltonian-hamiltonian.conj().T)) > 1e-13:
        raise ValueError("Non-Hermitian implementation")
    energies, vectors = eigh(hamiltonian, check_finite=True)
    occupied = energies < 0.0
    expected_occupied = components*length//2
    if int(np.count_nonzero(occupied)) != expected_occupied:
        raise ValueError("Unexpected negative-mode count")
    negative_vectors = vectors[:components*(length//2), occupied]
    result = entropy(negative_vectors @ negative_vectors.conj().T)
    result.update({"sites": length, "components": components,
                   "periodic": periodic,
                   "occupied_modes": expected_occupied,
                   "gap_to_zero": float(np.min(np.abs(energies)))})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    checks = []

    def check(name: str, value: float, tolerance: float):
        checks.append({"name": name, "error": float(value),
                       "tolerance": tolerance,
                       "passed": bool(np.isfinite(value) and value <= tolerance)})

    eta_results = []
    for mass in (0.0, 0.1, 0.3, 1.0, 2.0):
        refinements = [integrate_segments(
            lambda w: density(w)*chain_formula(math.sqrt(mass*mass+w)), tol)
            for tol in (1e-7, 1e-9, 1e-11)]
        difference = abs(refinements[-1]["value"]-refinements[-2]["value"])
        check(f"eta_refinement_mu_{mass}", difference, 1e-9)
        eta_results.append({"mu": mass, "refinements": refinements})
    check("historical_massless_coefficient",
          abs(eta_results[0]["refinements"][-1]["value"]-0.78658), 1e-5)
    density_result = integrate_segments(density, 1e-11)
    check("transverse_density_normalization", abs(density_result["value"]-1.0), 1e-9)

    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.diag([1.0, -1.0]).astype(complex)
    identity2 = np.eye(2, dtype=complex)
    alpha = [np.kron(sx, pauli) for pauli in (sx, sy, sz)]
    beta = np.kron(sz, identity2)
    finite_results = []
    triples = ((0.0, 0.37, 0.82), (0.2, 0.11, 0.29), (0.7, 1.1, 0.4))
    for index, (mass, ky, kz) in enumerate(triples):
        effective_mass = math.sqrt(mass*mass+math.sin(ky)**2+math.sin(kz)**2)
        onsite = mass*beta + math.sin(ky)*alpha[1] + math.sin(kz)*alpha[2]
        rows = [chain_covariance(onsite, -0.5j*alpha[0], length)
                for length in (64, 128)]
        exact = chain_formula(effective_mass)
        check(f"finite_chain_length_shift_{index}",
              abs(rows[1]["entropy"]-rows[0]["entropy"]), 1e-8)
        for row in rows:
            check(f"finite_chain_formula_{index}_L{row['sites']}",
                  abs(row["entropy"]-exact), 1e-8)
        result = {"mu": mass, "ky": ky, "kz": kz,
                  "effective_mass": effective_mass,
                  "formula": exact, "open_chains": rows}
        if index == 0:
            two_component = chain_covariance(effective_mass*sz, -0.5j*sx, 128)
            periodic = chain_covariance(onsite, -0.5j*alpha[0], 128, periodic=True)
            check("two_component_multiplicity",
                  abs(2.0*two_component["entropy"]-rows[1]["entropy"]), 1e-8)
            check("periodic_two_boundary_count",
                  abs(periodic["entropy"]/2.0-rows[1]["entropy"]), 1e-8)
            result.update({"two_component_open": two_component,
                           "four_component_periodic": periodic})
        finite_results.append(result)

    product = chain_covariance(beta, np.zeros((4, 4), dtype=complex), 64)
    check("zero_hopping_product", abs(product["entropy"]), 1e-10)
    small_mass = 1e-4
    cc_slope = -(chain_formula(2.0*small_mass)-chain_formula(small_mass))/math.log(2.0)
    check("small_mass_cc_slope", abs(cc_slope-2.0/3.0), 1e-6)
    heavy = [{"M": mass, "ratio": chain_formula(mass)*4.0*mass*mass
              /(1.0+math.log(16.0*mass*mass))} for mass in (100.0, 200.0)]
    heavy_errors = [abs(row["ratio"]-1.0) for row in heavy]
    check("large_mass_schmidt_count", heavy_errors[0], 5e-4)
    checks.append({"name": "large_mass_refinement", "passed": bool(heavy_errors[1] < heavy_errors[0]),
                   "errors": heavy_errors})
    passed = all(row["passed"] for row in checks)
    output = {"check": "reviewed reviewer B bounded area-law SPEC",
              "numpy_version": np.__version__, "scipy_version": scipy.__version__,
              "nice": os.getpriority(os.PRIO_PROCESS, 0),
              "elapsed_seconds": time.monotonic()-started,
              "units": "a=1, natural-log entropy, full four-component one-boundary convention",
              "eta_100": eta_results, "density_normalization": density_result,
              "finite_chain_controls": finite_results, "product_control": product,
              "cc_slope": cc_slope, "heavy_mass_controls": heavy,
              "checks": checks, "all_passed": passed,
              "scope": "No tilted or spherical three-dimensional cuts; no coupled evolution"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "checks": len(checks),
                      "all_passed": passed, "elapsed_seconds": output["elapsed_seconds"],
                      "nice": output["nice"],
                      "eta_100_mu0": eta_results[0]["refinements"][-1]["value"],
                      "failed_checks": [row["name"] for row in checks if not row["passed"]]}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
