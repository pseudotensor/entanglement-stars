#!/usr/bin/env python3
"""Required I2 full-band quadrature; run only on the remote qi2 compute host.

The formulas and six prescribed evaluations are reviewer B's independent I2
specification, recorded in solve/STATUS.md before implementation. This is
a streamed CPU check, with no field evolution, fitting, or stored arrays.
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


def evaluate(order, mass_parameters):
    abscissae, legendre_weights = np.polynomial.legendre.leggauss(order)
    theta = (math.pi / 4.0) * (abscissae + 1.0)
    weights = legendre_weights / 2.0
    sine_squared = np.sin(theta) ** 2
    cosine_squared = np.cos(theta) ** 2
    current_squared = sine_squared * cosine_squared
    yz_sine = sine_squared[:, None] + sine_squared[None, :]
    yz_current = current_squared[:, None] + current_squared[None, :]
    yz_weights = weights[:, None] * weights[None, :]
    rows = []
    for mu in mass_parameters:
        energy_mean = pressure_mean = cosine_energy_mean = 0.0
        for index, weight in enumerate(weights):
            energy = np.sqrt(mu * mu + sine_squared[index] + yz_sine)
            mean_slab = float(np.sum(yz_weights * energy))
            energy_mean += float(weight) * mean_slab
            pressure_mean += float(weight) * float(
                np.sum(yz_weights * (current_squared[index] + yz_current) / energy)
            )
            cosine_energy_mean += float(weight) * math.cos(2.0 * theta[index]) * mean_slab
        rho = 2.0 * energy_mean
        pressure = (2.0 / 3.0) * pressure_mean
        cosine_energy = 2.0 * cosine_energy_mean
        ratio = pressure / rho
        rows.append({
            "order": order,
            "ma": mu,
            "rho_a4": rho,
            "p_Q_a4": pressure,
            "p_Q_over_rho": ratio,
            "twice_mean_cos2theta1_e": cosine_energy,
            "cosine_identity_residual_C_plus_P": cosine_energy + pressure,
            "homogeneous_residual_over_H2": 1.5 * (1.0 + ratio),
            "aH_at_g_0_25": math.sqrt((8.0 * math.pi / 3.0) * 0.25 * rho),
            "aH_conditional_eta_0_919": math.sqrt((2.0 * math.pi / 3.0) * rho / 0.919),
            "normalized_weight_sum": float(np.sum(weights) ** 3),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    rows = []
    previous = {}
    for order in (24, 48, 96):
        for row in evaluate(order, (0.0, 0.2)):
            mu = row["ma"]
            if mu in previous:
                row["observed_absolute_change_from_previous_order"] = {
                    key: abs(row[key] - previous[mu][key])
                    for key in ("rho_a4", "p_Q_a4", "twice_mean_cos2theta1_e")
                }
            previous[mu] = row
            rows.append(row)
    result = {
        "control": "WEB40E I2 / paper insert 12: native full-band benchmark",
        "formula_provenance": "Independent reviewer B I2 specification recorded in solve/STATUS.md",
        "domain": "Normal-band full-filling benchmark with the reference premises stated in the author source",
        "measure": "(2/pi)^3 integral over [0,pi/2]^3, equivalent to the normalized full Brillouin zone",
        "method": "Tensor Gauss-Legendre, first-axis slabs streamed; orders 24,48,96; no fit",
        "spin_factor": 2,
        "valley_counting": "Already included by full-zone normalization; no extra factor of eight",
        "error_scope": "Refinement differences and identity residuals are observed controls, not certified error radii",
        "eta_scope": "eta=0.919 is supplied only as an illustrative conditional entropy-match input",
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
    print(f"Saved {len(rows)} prescribed quadrature records to {args.output}; "
          f"wall={result['elapsed_seconds']:.3f}s, max_RSS={result['maximum_resident_kib']} KiB")


if __name__ == "__main__":
    main()
