#!/usr/bin/env python3
"""Evaluate the T10 reviewer B-specified sine-projector lower-bound integral.

All grids cover one full Brillouin zone. There is no separate valley factor.
Quadrature values are numerical estimates; the analytic inequality supplied
by the T10 author is reported separately, without a quadrature certificate.
Run only on the remote qi2 interpreter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def evaluate(grid_sizes: list[int], masses: list[float]) -> dict:
    rows = []
    log_two = float(np.log(2))
    previous = {}
    for ngrid in grid_sizes:
        # A half-step avoids the zero-mass nodes of the sine symbol.
        momenta = -np.pi + 2 * np.pi * (np.arange(ngrid) + 0.5) / ngrid
        sine_squared = np.sin(momenta) ** 2
        x_squared = sine_squared[:, None, None]
        symbol_squared = x_squared + sine_squared[None, :, None] + sine_squared[None, None, :]
        for mass in masses:
            integral = float(np.mean(x_squared / np.sqrt(mass ** 2 + symbol_squared)))
            numerical_bound_estimate = 4 * log_two * integral ** 2
            analytic_bound = log_two / (mass ** 2 + 3)
            rows.append({
                "N_BZ": ngrid,
                "mu": mass,
                "I_quadrature": integral,
                "I_change_from_previous_grid": None if mass not in previous else integral - previous[mass],
                "single_packet_4ln2_I_squared_estimate": numerical_bound_estimate,
                "single_packet_analytic_lower_bound_ln2_over_mu_squared_plus_3": analytic_bound,
                "four_equal_mass_packets_analytic_lower_bound": 4 * analytic_bound,
                "four_equal_mass_packets_4ln2_I_squared_estimate": 4 * numerical_bound_estimate,
            })
            previous[mass] = integral
        del symbol_squared
    return {
        "scope": "T10 ordinary planar cut of the specified four-component free sine projectors; the full Brillouin zone already contains all valleys. This file evaluates only the lower-bound integral, not the Toeplitz entropy or thermodynamic route.",
        "normalization": {
            "I": "BZ average sin(q_x)^2 / sqrt(mu^2 + sin(q_x)^2 + sin(q_y)^2 + sin(q_z)^2)",
            "measure": "d^3q/(2pi)^3 over [-pi,pi)^3",
            "momentum_grid": "uniform midpoint grid, half-step shifted in all three directions",
            "separate_valley_factor": 1,
            "logarithms": "natural",
            "single_packet_eta_inequality_from_reviewer B": "eta_s >= 4 ln(2) I(mu_s)^2 >= ln(2)/(mu_s^2+3)",
            "four_packet_rule": "sum four positive packet contributions; equal-mass columns are diagnostics, not a matched rectangle",
        },
        "grid_sizes": grid_sizes,
        "masses": masses,
        "rows": rows,
        "uniform_resolved_window_comparison": {
            "condition_from_reviewer B": "all four mu_s <= 1, and a^2/(4G) <= 1e-4",
            "four_packet_analytic_lower_bound_ln2": log_two,
            "comparison_area_coefficient_upper": 1e-4,
            "analytic_lower_bound_to_comparison_upper_ratio": log_two / 1e-4,
        },
        "caveat": "The integral values and 4 ln(2) I^2 columns have grid differences but no certified quadrature error; the closed analytic bound is stated as the T10 author's inequality, not inferred from floating-point integration.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grids", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--masses", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.04])
    args = parser.parse_args()
    if any(n <= 0 or n % 2 for n in args.grids) or any(mu < 0 for mu in args.masses):
        raise ValueError("Use positive even grid sizes and nonnegative dimensionless masses")
    result = evaluate(args.grids, args.masses)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "entropy_lower_bound.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = [result["scope"], "", "N_BZ mu I numerical 4ln2 I² (single packet) analytic ln2/(mu²+3) grid change:"]
    for row in result["rows"]:
        change = row["I_change_from_previous_grid"]
        change_text = "n/a" if change is None else f"{change:.9g}"
        lines.append(f"{row['N_BZ']} {row['mu']:.6g} {row['I_quadrature']:.12g} "
                     f"{row['single_packet_4ln2_I_squared_estimate']:.12g} "
                     f"{row['single_packet_analytic_lower_bound_ln2_over_mu_squared_plus_3']:.12g} {change_text}")
    comparison = result["uniform_resolved_window_comparison"]
    lines += ["", f"Four-packet analytic window lower bound: {comparison['four_packet_analytic_lower_bound_ln2']:.12g}",
              f"Comparison coefficient upper: {comparison['comparison_area_coefficient_upper']:.12g}",
              f"Ratio: {comparison['analytic_lower_bound_to_comparison_upper_ratio']:.12g}", "", result["caveat"]]
    summary = "\n".join(lines) + "\n"
    (args.output_dir / "entropy_lower_bound.txt").write_text(summary)
    print(summary, end="")


if __name__ == "__main__":
    main()
