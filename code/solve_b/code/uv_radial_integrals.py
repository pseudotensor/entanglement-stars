#!/usr/bin/env python3
"""Uniform-only radial integrals for the T10 author's separate UV pilot.

These integrals do not determine the inhomogeneous polar trace coefficient.
No Newton normalization, entropy normalization, or regulator-independent
gravitational claim is made. Run only with the remote qi2 interpreter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.integrate import quad


def integrate_at_precision(digits: int) -> dict:
    mp.mp.dps = digits
    # Exact mass squares preserve the rectangle cancellation at every precision.
    mass_squared = [mp.mpf(0), mp.mpf("0.5"), mp.mpf("0.5"), mp.mpf(1)]
    grading = [1, -1, -1, 1]
    normalization = 1 / (2 * np.pi ** 2)
    rows = []
    for moment in (0, 2, 4):
        def integrand(p: float) -> float:
            if p == 0:
                return 0.0
            radial = mp.mpf(p)
            common = radial ** 2 + radial ** 8
            grouped = mp.fsum(sign / mp.sqrt(common + x) for sign, x in zip(grading, mass_squared))
            return float(radial ** (2 + moment) * grouped) * normalization

        segments = []
        for lower, upper in ((0.0, 1.0), (1.0, 10.0), (10.0, np.inf)):
            value, estimated_error = quad(integrand, lower, upper, epsabs=1e-12, epsrel=1e-11, limit=300)
            segments.append({"lower": lower, "upper": "infinity" if np.isinf(upper) else upper,
                             "value": float(value), "quad_estimated_absolute_error": float(estimated_error)})
        rows.append({"n": moment, "I_n": sum(row["value"] for row in segments),
                     "quad_estimated_absolute_error": sum(row["quad_estimated_absolute_error"] for row in segments),
                     "segments": segments})
    values = {row["n"]: row["I_n"] for row in rows}
    return {"mpmath_decimal_digits": digits, "rows": rows,
            "uniform_proxy_24_I0_NOT_matched_trace_coefficient": 24 * values[0],
            "I2_over_I0": values[2] / values[0], "I4_over_I0": values[4] / values[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    passes = [integrate_at_precision(digits) for digits in (40, 70)]
    result = {
        "scope": "Separate UV-suppressing uniform-dispersion pilot; not the current strand-A sine law and not a derivation of its inhomogeneous polar response or matched gravitational coefficient.",
        "parameters": {"Omega": 1.0, "mass_squared": [0.0, 0.5, 0.5, 1.0],
                       "orientations": [1, -1, -1, 1],
                       "uniform_energy": "sqrt(m_s^2+p^2+p^8), Omega=1"},
        "definition": "I_n=(1/(2pi^2)) integral_0^infinity p^(2+n) sum_s eta_s/sqrt(m_s^2+p^2+p^8) dp",
        "cancellation_method": "Exact rectangle mass squares and 40/70-digit mpmath evaluation of the grouped sum before binary64 adaptive quadrature.",
        "precision_passes": passes,
        "precision_differences_absolute": {
            str(before["n"]): abs(after["I_n"] - before["I_n"])
            for before, after in zip(passes[0]["rows"], passes[1]["rows"])
        },
        "caveat": "Adaptive quadrature error estimates are not interval certificates. The full A_Omega dictionary and ruler derivatives are not implemented; additional source terms may alter the trace coefficient. No Newton scale or entropy comparison is inferred.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "uv_radial_integrals.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = [result["scope"], "", "digits n I_n estimated quadrature absolute error:"]
    for precision_pass in passes:
        for row in precision_pass["rows"]:
            lines.append(f"{precision_pass['mpmath_decimal_digits']} {row['n']} {row['I_n']:.15g} {row['quad_estimated_absolute_error']:.9g}")
        lines.append(f"Uniform-only proxy 24 I0 (NOT matched trace coefficient): {precision_pass['uniform_proxy_24_I0_NOT_matched_trace_coefficient']:.15g}")
        lines.append(f"I2/I0: {precision_pass['I2_over_I0']:.15g}; I4/I0: {precision_pass['I4_over_I0']:.15g}")
    lines += ["", result["caveat"]]
    summary = "\n".join(lines) + "\n"
    (args.output_dir / "uv_radial_integrals.txt").write_text(summary)
    print(summary, end="")


if __name__ == "__main__":
    main()
