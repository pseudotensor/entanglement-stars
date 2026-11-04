#!/usr/bin/env python3
"""T10 restricted radial transverse lower-bound estimate in the small-a window.

Uses the complete normal sine FFT (both normal cones), four transverse sine
preimages, all four packet contributions, and no additional valley factor.
The exact transverse weight is >=1 and omitted for the weaker bound. Runtime
and all numerical post-processing must be remote qi2.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss


def quadrature(order: int, qmax: float) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = leggauss(order)
    boundaries = [0.0, 1.0]
    boundaries += [value for value in (2.0, 4.0) if value < qmax]
    boundaries += [qmax]
    points, measures = [], []
    for lower, upper in zip(boundaries, boundaries[1:]):
        points.extend(((upper + lower) / 2 + (upper - lower) / 2 * nodes).tolist())
        measures.extend(((upper - lower) / 2 * weights).tolist())
    return np.array(points), np.array(measures)


def packet_variance(spacing: float, mass_squared: float, normal_grid: int, q: np.ndarray) -> dict:
    normal_q = -np.pi + 2 * np.pi * (np.arange(normal_grid) + 0.5) / normal_grid
    dx = np.sin(normal_q)[:, None] / spacing
    transverse = q[None, :]
    t_squared = dx ** 2 + transverse ** 2
    real_part = np.sqrt(mass_squared + t_squared ** 4)
    energy = np.sqrt(mass_squared + t_squared + t_squared ** 4)
    bloch = np.stack([real_part / energy - 1.0,
                      np.broadcast_to(dx, energy.shape) / energy,
                      np.broadcast_to(transverse, energy.shape) / energy], axis=-1)
    coefficients = np.fft.ifft(bloch, axis=0)
    distances = np.arange(1, normal_grid // 2)
    norm_squared = np.sum(np.abs(coefficients[distances]) ** 2, axis=-1)
    weighted = distances[:, None] * norm_squared
    variance = np.sum(weighted, axis=0)
    tail = np.sum(weighted[distances >= normal_grid // 4], axis=0)
    return {"variance_at_Q": variance, "tail_at_Q": tail}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["0.01:16384", "0.005:32768", "0.005:65536"])
    parser.add_argument("--orders", type=int, nargs="+", default=[24, 48])
    parser.add_argument("--light-masses", type=float, nargs="+", default=[0.0, 0.001])
    parser.add_argument("--qmax", type=float, default=4.0)
    args = parser.parse_args()
    cases = [(float(value.split(":")[0]), int(value.split(":")[1])) for value in args.cases]
    if args.qmax <= 1 or any(a * args.qmax >= 1 or nx % 2 or nx <= 0 for a, nx in cases):
        raise ValueError("Use even Nx and 1 < Qmax < 1/a")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gamma = float((4 * np.log(2) - 2) * (np.sqrt(2) - 1))
    result = {
        "scope": "Restricted Q<=Qmax planar entropy lower-bound estimate for the UV candidate at small finite spacing, based on full normal sine FFT. Quadrature and normal aliasing errors remain uncertified.",
        "normalization": {
            "Omega": 1.0, "normal_Fourier": "canonical normal-site 1/Nx full-sine FFT on half-step grid",
            "normal_cones": "both x cones already included",
            "transverse_measure_factor": "4/(2pi) Q dQ after dividing native eta by a^2",
            "transverse_exact_weight": "w(aQ)>=1; omitted to weaken the bound",
            "Q_above_Qmax": "omitted nonnegative contribution",
            "additional_valley_factor": 1,
            "four_packet_rule": "positive multiplicities (1,2,1), rectangle (delta^2,(1+delta^2)/2,(1+delta^2)/2,1)",
            "gamma": gamma,
            "entropy_lower_bound": "4ln2 V_total + gamma V_Q>=1, with restricted radial measure",
            "normal_Nyquist_endpoint": "omitted; distances1..Nx/2-1",
        },
        "Qmax": args.qmax, "rows": [], "complete": False,
        "caveat": "These are finite FFT and Gauss estimates, not interval-enclosed integrals or a certified full cut entropy. No thermodynamic or stellar-state result is inferred.",
    }
    print(result["scope"], flush=True)
    for order in args.orders:
        q, weights = quadrature(order, args.qmax)
        measure = 4 / (2 * np.pi) * q * weights
        high_region = q >= 1
        for spacing, normal_grid in cases:
            for delta in args.light_masses:
                start = time.perf_counter()
                packets = []
                print(f"start a={spacing} Nx={normal_grid} order={order} delta={delta}", flush=True)
                for mass_squared, multiplicity in zip([delta ** 2, (1 + delta ** 2) / 2, 1.0], [1, 2, 1]):
                    data = packet_variance(spacing, mass_squared, normal_grid, q)
                    total = float(measure @ data["variance_at_Q"])
                    regional = float(measure[high_region] @ data["variance_at_Q"][high_region])
                    tail = float(measure @ data["tail_at_Q"])
                    packets.append({"mass_squared": mass_squared, "multiplicity": multiplicity,
                                    "restricted_variance_per_physical_area": total,
                                    "restricted_Q_ge_one_variance_per_physical_area": regional,
                                    "normal_last_quarter_variance_per_physical_area": tail,
                                    "variance_at_Q": data["variance_at_Q"].tolist()})
                total = sum(row["multiplicity"] * row["restricted_variance_per_physical_area"] for row in packets)
                regional = sum(row["multiplicity"] * row["restricted_Q_ge_one_variance_per_physical_area"] for row in packets)
                row = {"a_Omega": spacing, "Nx": normal_grid, "normal_physical_period": spacing * normal_grid,
                       "radial_Gauss_order_per_interval": order, "m_light_over_Omega": delta,
                       "Q_nodes": q.tolist(), "Q_weights": weights.tolist(), "packets": packets,
                       "four_packet_restricted_variance_per_physical_area": total,
                       "four_packet_restricted_high_region_variance_per_physical_area": regional,
                       "four_packet_sigma_4ln2_variance_estimate": float(4 * np.log(2) * total),
                       "regional_gamma_correction": gamma * regional,
                       "four_packet_sigma_enhanced_lower_bound_estimate": float(4 * np.log(2) * total + gamma * regional),
                       "elapsed_seconds": time.perf_counter() - start}
                result["rows"].append(row)
                (args.output_dir / "uv_restricted_variance.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
                print(f"done sigma4ln2={row['four_packet_sigma_4ln2_variance_estimate']:.12g} regionV={regional:.12g} enhanced={row['four_packet_sigma_enhanced_lower_bound_estimate']:.12g} seconds={row['elapsed_seconds']:.3g}", flush=True)
    result["complete"] = True
    (args.output_dir / "uv_restricted_variance.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = [result["scope"], "", "aOmega Nx order delta 4ln2variance regionalV gammaCorrection enhancedSigma:"]
    for row in result["rows"]:
        lines.append(f"{row['a_Omega']} {row['Nx']} {row['radial_Gauss_order_per_interval']} {row['m_light_over_Omega']} "
                     f"{row['four_packet_sigma_4ln2_variance_estimate']:.12g} "
                     f"{row['four_packet_restricted_high_region_variance_per_physical_area']:.12g} "
                     f"{row['regional_gamma_correction']:.12g} {row['four_packet_sigma_enhanced_lower_bound_estimate']:.12g}")
    lines += ["", result["caveat"]]
    summary = "\n".join(lines) + "\n"
    (args.output_dir / "uv_restricted_variance.txt").write_text(summary)
    print(summary, end="", flush=True)


if __name__ == "__main__":
    main()
