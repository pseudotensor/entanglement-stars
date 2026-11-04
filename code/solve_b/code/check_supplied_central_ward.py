#!/usr/bin/env python3
"""Required I6 Ward integral at three supplied central chemical potentials.

Run on the remote qi2 host. This does not find a stellar branch or peak.
The independent formulas and prescribed 16/32/64 controls are recorded in
solve/STATUS.md. Only scalar quadratures are saved; no arrays or fit.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import socket
import time

import numpy as np


SUPPLIED_INPUTS = (
    (0.2, 0.8660804616110328, 0.6414711076754969),
    (0.1, 0.8754123046950786, 0.6670336822405672),
    (0.05, 0.8774485038885971, 0.6726082670547097),
)


def evaluate(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    u = (nodes + 1.0) / 2.0
    phi = (math.pi / 4.0) * (nodes + 1.0)
    polar_radius = np.sqrt(1.0 - u * u)
    nx = polar_radius[:, None] * np.cos(phi)[None, :]
    ny = polar_radius[:, None] * np.sin(phi)[None, :]
    nz = np.broadcast_to(u[:, None], nx.shape)
    squared_directions = np.stack((nx * nx, ny * ny, nz * nz), axis=-1)
    fourth_sum = np.sum(squared_directions ** 2, axis=-1)
    sphere_weights = math.pi * weights[:, None] * weights[None, :]
    prefactor = 16.0 / (2.0 * math.pi) ** 3
    rows = []
    for y, supplied_lambda_over_m, supplied_N_center in SUPPLIED_INPUTS:
        sigma = supplied_lambda_over_m / supplied_N_center
        z = math.sqrt(sigma * sigma - 1.0)
        guard = y * z
        if not guard < 1.0:
            raise ValueError("Specified sine-coordinate guard y*z<1 failed")
        radial_nodes = (z / 2.0) * (nodes + 1.0)
        radial_weights = (z / 2.0) * weights
        rho_integral = pressure_integral = defect_integral = 0.0
        for t, weight in zip(radial_nodes, radial_weights):
            energy = math.sqrt(1.0 + t * t)
            jacobian = 1.0 / np.sqrt(np.prod(
                1.0 - y * y * t * t * squared_directions, axis=-1
            ))
            mean_jacobian = float(np.sum(sphere_weights * jacobian))
            mean_pressure = float(np.sum(
                sphere_weights * (1.0 - y * y * t * t * fourth_sum) * jacobian
            ))
            rho_integral += float(weight) * t * t * energy * mean_jacobian
            pressure_integral += float(weight) * t ** 4 / energy * mean_pressure
            defect_integral += float(weight) * t ** 4 * energy * mean_jacobian
        edge_jacobian = 1.0 / np.sqrt(np.prod(
            1.0 - y * y * z * z * squared_directions, axis=-1
        ))
        edge_integral = float(np.sum(
            sphere_weights * (1.0 - y * y * z * z * fourth_sum) * edge_jacobian
        ))
        rho = prefactor * rho_integral
        pressure = (prefactor / 3.0) * pressure_integral
        defect = (2.0 * y * y * prefactor / 3.0) * defect_integral
        chi = (prefactor * sigma * z ** 3 / 3.0) * edge_integral
        residual = rho + pressure - chi - defect
        rows.append({
            "order": order,
            "ma": y,
            "supplied_lambda_over_m": supplied_lambda_over_m,
            "supplied_N_center": supplied_N_center,
            "sigma": sigma,
            "z": z,
            "y_times_z": guard,
            "guard_yz_less_than_one": True,
            "rhohat": rho,
            "phat": pressure,
            "Dhat": defect,
            "chihat": chi,
            "Dhat_over_phat": defect / pressure,
            "ward_identity_residual": residual,
            "ward_identity_residual_over_rhohat": residual / rho,
            "sphere_weight_sum": float(np.sum(sphere_weights)),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    rows = []
    previous = {}
    for order in (16, 32, 64):
        for row in evaluate(order):
            y = row["ma"]
            if y in previous:
                row["observed_absolute_change_from_previous_order"] = {
                    key: abs(row[key] - previous[y][key])
                    for key in ("rhohat", "phat", "Dhat", "chihat", "Dhat_over_phat")
                }
            previous[y] = row
            rows.append(row)
    result = {
        "control": "WEB40E I6 / paper insert 8: Ward integral at supplied central chemical potentials",
        "formula_provenance": "Independent reviewer B I6 sine-coordinate specification recorded in solve/STATUS.md",
        "scalar_input_provenance": (
            "<internal>/84_20261015_web40e"
            "FRESH1342_20261015-2209_summarize-model-programme/files/p12_static/results_summary.json"
        ),
        "input_scope": "Supplied central inputs only; neither branch nor first turning point reproduced",
        "method": "Streamed tensor Gauss-Legendre on radial interval and octant sphere; orders 16,32,64; no fit",
        "normalization": "A=16/(2*pi)^3, full sphere via eight octants",
        "error_scope": "Observed refinement and Ward residual are controls, not certified interval errors",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "host": socket.gethostname(),
        "numpy_version": np.__version__,
        "thread_settings": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
        )},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "maximum_resident_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Saved {len(rows)} prescribed Ward records to {args.output}; "
          f"wall={result['elapsed_seconds']:.3f}s, max_RSS={result['maximum_resident_kib']} KiB")


if __name__ == "__main__":
    main()
