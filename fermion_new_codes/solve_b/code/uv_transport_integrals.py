#!/usr/bin/env python3
"""Integrals for the T10 author's proposed full UV transport source vertex.

The proposed A_Omega and coefficient formulas await independent review.
Report base and real-vertex terms separately; do not infer a Newton match,
entropy normalization, or a finite-spacing observation window.
All computation must run with the remote qi2 interpreter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.integrate import quad


def integrate_at_precision(digits: int, light_mass: str = "0") -> dict:
    mp.mp.dps = digits
    x0 = mp.mpf(light_mass) ** 2
    midpoint = (1 + x0) / 2
    mass_squared = [x0, midpoint, midpoint, mp.mpf(1)]
    grading = [1, -1, -1, 1]
    normalization = 1 / (2 * np.pi ** 2)
    rows = []
    for moment in (0, 2, 4):
        for term in ("base", "real_vertex_R_prime_squared"):
            def integrand(p: float) -> float:
                if p == 0:
                    return 0.0
                radial = mp.mpf(p)
                power8 = radial ** 8
                energy_squared_common = radial ** 2 + power8
                terms = []
                for sign, x in zip(grading, mass_squared):
                    energy = mp.sqrt(x + energy_squared_common)
                    weight = mp.mpf(1) if term == "base" else 16 * radial ** 14 / (x + power8)
                    terms.append(sign * weight / energy)
                grouped = mp.fsum(terms)
                return float(radial ** (2 + moment) * grouped) * normalization

            segments = []
            for lower, upper in ((0.0, 1.0), (1.0, 10.0), (10.0, np.inf)):
                value, estimated_error = quad(integrand, lower, upper, epsabs=1e-12, epsrel=1e-11, limit=300)
                segments.append({"lower": lower, "upper": "infinity" if np.isinf(upper) else upper,
                                 "value": float(value), "quad_estimated_absolute_error": float(estimated_error)})
            rows.append({"n": moment, "term": term, "integral": sum(row["value"] for row in segments),
                         "quad_estimated_absolute_error": sum(row["quad_estimated_absolute_error"] for row in segments),
                         "segments": segments})
    values = {(row["n"], row["term"]): row["integral"] for row in rows}
    base = {n: values[n, "base"] for n in (0, 2, 4)}
    vertex = {n: values[n, "real_vertex_R_prime_squared"] for n in (0, 2, 4)}
    coefficient_base = 24 * base[0]
    coefficient_vertex = 8 * vertex[0]
    coefficient = coefficient_base + coefficient_vertex
    c2_base, c2_vertex = 4 * base[2], -24 * vertex[2] / 30
    c4_base, c4_vertex = 24 * 13 * base[4] / 120, -24 * vertex[4] / 120
    return {
        "mpmath_decimal_digits": digits, "rows": rows,
        "candidate_coefficient_PENDING_review": {"base": coefficient_base, "real_vertex": coefficient_vertex, "total": coefficient},
        "candidate_C2_PENDING_review": {"base": c2_base, "real_vertex": c2_vertex, "total": c2_base + c2_vertex},
        "candidate_C4_PENDING_review": {"base": c4_base, "real_vertex": c4_vertex, "total": c4_base + c4_vertex},
        "candidate_signed_relative_dilation_a_squared_psi_four_coefficient_PENDING_review": 2 * (c2_base + c2_vertex) / coefficient,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--light-mass", type=str, default="0")
    args = parser.parse_args()
    light_mass = float(args.light_mass)
    if not 0 <= light_mass < 1:
        raise ValueError("Preserved Omega=m4=1 rectangle requires 0 <= m_light < 1")
    passes = [integrate_at_precision(digits, args.light_mass) for digits in (40, 70)]
    result = {
        "scope": "T10 proposed full UV polar-transport source-vertex pilot; formulas and realization pending independent review. No Newton or entropy normalization established.",
        "parameters": {"Omega": 1.0, "m_light": light_mass,
                       "mass_squared": [light_mass ** 2, (1 + light_mass ** 2) / 2, (1 + light_mass ** 2) / 2, 1.0],
                       "orientations": [1, -1, -1, 1]},
        "proposed_dictionary_from_reviewer B": "A_Omega=psi^-3[psi F_s(T_psi) psi+iD], T_psi=psi^-2 Delta_a psi^-2, F_s=sqrt(m_s^2+T_psi^4/Omega^6)",
        "radial_definitions": {"measure": "p^2 dp/(2pi^2)", "base": "sum eta_s / sqrt(m_s^2+p^2+p^8)",
                               "real_vertex": "sum eta_s R_s_prime^2 / sqrt(m_s^2+p^2+p^8)",
                               "R_prime_squared": "16 p^14/(m_s^2+p^8), Omega=1",
                               "candidate_k": "24 (I0 + J0/3)", "candidate_C2": "24 (I2/6-J2/30)",
                               "candidate_C4": "24 (13 I4-J4)/120"},
        "precision_passes": passes,
        "precision_differences_absolute": {
            f"{before['n']}:{before['term']}": abs(after["integral"] - before["integral"])
            for before, after in zip(passes[0]["rows"], passes[1]["rows"])
        },
        "caveat": "Grouped integrands evaluated at 40/70-digit precision before binary64 adaptive quadrature. Error estimates are not interval certificates. Inhomogeneous matrix rebuilding and entropy calculation are not part of this radial test.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "uv_transport_integrals.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    final = passes[-1]
    lines = [result["scope"], "", "n term integral estimated quadrature absolute error:"]
    for row in final["rows"]:
        lines.append(f"{row['n']} {row['term']} {row['integral']:.15g} {row['quad_estimated_absolute_error']:.9g}")
    for name in ("candidate_coefficient_PENDING_review", "candidate_C2_PENDING_review", "candidate_C4_PENDING_review"):
        values = final[name]
        lines.append(f"{name}: base={values['base']:.15g}, real_vertex={values['real_vertex']:.15g}, total={values['total']:.15g}")
    lines.append(f"Candidate signed relative dilation coefficient: {final['candidate_signed_relative_dilation_a_squared_psi_four_coefficient_PENDING_review']:.15g}")
    lines += ["", result["caveat"]]
    summary = "\n".join(lines) + "\n"
    (args.output_dir / "uv_transport_integrals.txt").write_text(summary)
    print(summary, end="")


if __name__ == "__main__":
    main()
