#!/usr/bin/env python3
"""T10 finite-a planar cross-cut variance for the reviewed UV candidate.

Uses the complete parent Brillouin zone, one sine operator, and four packet
contributions. This computes an entropy lower-bound estimate, not the full
entropy or a certified continuum limit. All runtime is remote qi2.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def packet_variance(spacing: float, mass_squared: float, normal_grid: int, transverse_grid: int) -> dict:
    normal_q = -np.pi + 2 * np.pi * (np.arange(normal_grid) + 0.5) / normal_grid
    transverse_q = -np.pi + 2 * np.pi * (np.arange(transverse_grid) + 0.5) / transverse_grid
    dx = np.sin(normal_q)[:, None] / spacing
    dz = np.sin(transverse_q)[None, :] / spacing
    distances = np.arange(1, normal_grid // 2)
    weighted_sum = 0.0
    regional_weighted_sum = 0.0
    weighted_tail_sum = 0.0
    max_bloch_norm_error = 0.0
    for qy in transverse_q:
        dy = float(np.sin(qy) / spacing)
        t_squared = dx ** 2 + dy ** 2 + dz ** 2
        real_part = np.sqrt(mass_squared + t_squared ** 4)
        energy = np.sqrt(mass_squared + t_squared + t_squared ** 4)
        bloch = np.stack([
            real_part / energy,
            np.broadcast_to(dx, energy.shape) / energy,
            np.full_like(energy, dy) / energy,
            np.broadcast_to(dz, energy.shape) / energy,
        ], axis=-1)
        max_bloch_norm_error = max(max_bloch_norm_error, float(np.max(np.abs(np.sum(bloch ** 2, axis=-1) - 1))))
        # Removing this exact constant affects r=0 only; no offsite contribution
        # is subtracted from the variance or entropy.
        bloch[:, :, 0] -= 1.0
        coefficients = np.fft.ifft(bloch, axis=0)
        # Tr Gamma_alpha Gamma_beta = 4 delta_alpha,beta; P has factor 1/2.
        norm_squared = np.sum(np.abs(coefficients[distances]) ** 2, axis=-1)
        weighted = distances[:, None] * norm_squared
        weighted_sum += float(np.sum(weighted))
        transverse_region = dy ** 2 + dz[0] ** 2 >= 1.0
        regional_weighted_sum += float(np.sum(weighted[:, transverse_region]))
        weighted_tail_sum += float(np.sum(weighted[distances >= normal_grid // 4]))
    variance = weighted_sum / transverse_grid ** 2
    regional_variance = regional_weighted_sum / transverse_grid ** 2
    tail = weighted_tail_sum / transverse_grid ** 2
    return {
        "mass_squared": mass_squared, "mass": float(np.sqrt(mass_squared)),
        "cross_cut_variance_per_transverse_site": variance,
        "high_transverse_region_variance_per_transverse_site": regional_variance,
        "single_packet_entropy_lower_bound_eta_estimate": float(4 * np.log(2) * variance),
        "last_quarter_distance_variance": tail,
        "last_quarter_distance_fraction": 0.0 if variance == 0 else tail / variance,
        "max_bloch_unit_norm_error": max_bloch_norm_error,
    }


def save_result(output_dir: Path, result: dict) -> None:
    (output_dir / "uv_cut_variance.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spacings", type=float, nargs="+", default=[0.2, 0.1])
    parser.add_argument("--light-masses", type=float, nargs="+", default=[0.0, 0.001])
    parser.add_argument("--resolutions", nargs="+", default=["256:64", "512:128"])
    args = parser.parse_args()
    resolutions = [tuple(map(int, value.split(":"))) for value in args.resolutions]
    if any(nx <= 0 or nperp <= 0 or nx % 2 or nperp % 2 for nx, nperp in resolutions):
        raise ValueError("Use positive even Nx:Nperp grids")
    if any(a <= 0 for a in args.spacings) or any(delta < 0 or delta >= 1 for delta in args.light_masses):
        raise ValueError("Use positive spacings and 0 <= m_light < 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "scope": "Finite-a UV candidate planar variance on one full parent sine Brillouin zone. The 4ln2 variance columns are numerical entropy lower-bound estimates, not full cut entropy or certified continuum coefficients.",
        "normalization": {
            "Omega": 1.0, "projector": "P=(I-Gamma0 R-Gamma_i d_i)/2 after dividing each Bloch component by epsilon",
            "R": "sqrt(m_s^2+(sum_i sin(q_i)^2/a^2)^4)",
            "epsilon": "sqrt(m_s^2+sum_i sin(q_i)^2/a^2+(sum_i sin(q_i)^2/a^2)^4)",
            "spatial_algebra": "original site CAR, four Dirac components per packet",
            "extra_valley_factor": 1, "four_packet_rule": "multiplicities (1,2,1), all contributions positive",
            "normal_Fourier": "1/Nx sum exp(i q_x r), half-step-shifted full [-pi,pi) grid; offsite norm phase cancels",
            "transverse_measure": "uniform midpoint dq_y dq_z/(2pi)^2 on full parent BZ",
            "variance_per_plane": "sum_{r=1}^{Nx/2-1} r average_{qy,qz} ||C_r(qy,qz)||F^2",
            "entropy_inequality": "eta >= 4 ln(2) variance per transverse site",
            "enhanced_entropy_inequality_from_reviewer B": "sigma >= 4 ln(2) V_total/a^2 + gamma V_region/a^2, gamma=(4ln(2)-2)(sqrt(2)-1)",
            "high_transverse_region": "sqrt(sin(q_y)^2+sin(q_z)^2)/a >= Omega=1",
            "physical_area_coefficient": "sigma_lower_estimate=eta_lower_estimate/(a Omega)^2",
        },
        "rows": [],
        "complete": False,
        "caveat": "Normal-distance truncation, periodic aliasing, transverse quadrature and finite-a-to-continuum errors are not interval certified. No cut/thermodynamic equality or stellar-state entropy claim.",
    }
    print(result["scope"], flush=True)
    for normal_grid, transverse_grid in resolutions:
        for spacing in args.spacings:
            for delta in args.light_masses:
                start = time.perf_counter()
                masses_squared = [delta ** 2, (1 + delta ** 2) / 2, 1.0]
                multiplicities = [1, 2, 1]
                packets = []
                print(f"start Nx={normal_grid} Nperp={transverse_grid} a={spacing} delta={delta}", flush=True)
                for mass_squared, multiplicity in zip(masses_squared, multiplicities):
                    packet = packet_variance(spacing, mass_squared, normal_grid, transverse_grid)
                    packet["multiplicity"] = multiplicity
                    packets.append(packet)
                variance = sum(row["multiplicity"] * row["cross_cut_variance_per_transverse_site"] for row in packets)
                regional_variance = sum(row["multiplicity"] * row["high_transverse_region_variance_per_transverse_site"] for row in packets)
                entropy_bound = float(4 * np.log(2) * variance)
                gamma = float((4 * np.log(2) - 2) * (np.sqrt(2) - 1))
                row = {
                    "Nx": normal_grid, "Nperp": transverse_grid, "a_Omega": spacing, "m_light_over_Omega": delta,
                    "packet_mass_squared": [delta ** 2, (1 + delta ** 2) / 2, (1 + delta ** 2) / 2, 1.0],
                    "normal_physical_period": normal_grid * spacing,
                    "transverse_momentum_spacing_over_a_Omega": float(2 * np.pi / transverse_grid / spacing),
                    "packets": packets, "four_packet_cross_cut_variance_per_transverse_site": variance,
                    "four_packet_eta_entropy_lower_bound_estimate": entropy_bound,
                    "four_packet_sigma_entropy_lower_bound_estimate": entropy_bound / spacing ** 2,
                    "four_packet_high_transverse_region_variance_per_transverse_site": regional_variance,
                    "four_packet_high_transverse_region_variance_per_physical_area": regional_variance / spacing ** 2,
                    "enhancement_gamma": gamma,
                    "regional_gamma_entropy_correction_per_physical_area": gamma * regional_variance / spacing ** 2,
                    "four_packet_sigma_enhanced_entropy_lower_bound_estimate": (entropy_bound + gamma * regional_variance) / spacing ** 2,
                    "elapsed_seconds": time.perf_counter() - start,
                }
                result["rows"].append(row)
                save_result(args.output_dir, result)
                print(f"done a={spacing} delta={delta} sigma_variance_bound={row['four_packet_sigma_entropy_lower_bound_estimate']:.12g} V_region/area={row['four_packet_high_transverse_region_variance_per_physical_area']:.12g} enhanced_bound={row['four_packet_sigma_enhanced_entropy_lower_bound_estimate']:.12g} seconds={row['elapsed_seconds']:.3g}", flush=True)
    result["complete"] = True
    save_result(args.output_dir, result)
    lines = [result["scope"], "", "Nx Nperp aOmega delta variance/site sigma lower-bound estimate V_region/area gamma correction enhanced sigma estimate:"]
    for row in result["rows"]:
        lines.append(f"{row['Nx']} {row['Nperp']} {row['a_Omega']} {row['m_light_over_Omega']} "
                     f"{row['four_packet_cross_cut_variance_per_transverse_site']:.12g} "
                     f"{row['four_packet_sigma_entropy_lower_bound_estimate']:.12g} "
                     f"{row['four_packet_high_transverse_region_variance_per_physical_area']:.12g} "
                     f"{row['regional_gamma_entropy_correction_per_physical_area']:.12g} "
                     f"{row['four_packet_sigma_enhanced_entropy_lower_bound_estimate']:.12g}")
    lines += ["", result["caveat"]]
    summary = "\n".join(lines) + "\n"
    (args.output_dir / "uv_cut_variance.txt").write_text(summary)
    print(summary, end="", flush=True)


if __name__ == "__main__":
    main()
