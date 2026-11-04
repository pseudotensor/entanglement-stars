"""Execute the reviewer B-authored final3_vacuum_surface_code_spec.tex remotely."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import warnings

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    if os.environ.get(variable) != "1":
        raise RuntimeError(f"{variable} must be 1 before numerical imports")
if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("CUDA must be disabled")
if os.getpriority(os.PRIO_PROCESS, 0) != 10:
    raise RuntimeError("Run with nice 10")

import numpy as np
import scipy
from scipy.integrate import IntegrationWarning, quad
from scipy.special import i0e, i1e


DESTINATION = Path(__file__).resolve().parent
EPSABS = 1e-12
EPSREL = 1e-10
ETA = (1, -1, -1, 1)
RECTANGLE = (1, 2, 2, 3)


def heat_integrands(t: float, mu: float) -> tuple[float, float]:
    if t == 0.0:
        return 0.0, 0.0
    s = mu * mu * t
    exponential = math.exp(-s)
    difference = -math.expm1(-s)
    weight = exponential * difference * difference
    bessel0 = i0e(t / 2.0)
    heat_weight = 0.5 * (bessel0 + i1e(t / 2.0)) * bessel0 * bessel0
    value = heat_weight * weight / math.sqrt(math.pi * t)
    ratio = 1.0 if s == 0.0 else s * exponential / difference
    multiplier = -4.0 + 4.0 * (-s + 2.0 * ratio)
    return value, value * multiplier


def primary(mu: float) -> dict:
    results = []
    for component in (0, 1):
        with warnings.catch_warnings():
            warnings.simplefilter("error", IntegrationWarning)
            result = quad(
                lambda t: heat_integrands(t, mu)[component],
                0.0, np.inf, epsabs=EPSABS, epsrel=EPSREL,
                limit=400, full_output=1,
            )
        if len(result) != 3:
            raise RuntimeError(f"Heat quadrature failed for mu={mu}: {result[3]}")
        value, error, information = result
        results.append((float(value), float(error), int(information["neval"])))
    integral, integral_error, integral_evaluations = results[0]
    numerator, numerator_error, numerator_evaluations = results[1]
    if integral <= 0:
        raise RuntimeError(f"Nonpositive I at mu={mu}")
    defect = numerator / integral
    return {
        "mu": float(mu), "I": integral, "q_numerator": numerator, "q": defect,
        "I_error_estimate": integral_error,
        "q_numerator_error_estimate": numerator_error,
        "q_error_estimate": (numerator_error + abs(defect) * integral_error) / integral,
        "I_evaluations": integral_evaluations,
        "q_numerator_evaluations": numerator_evaluations,
    }


def brillouin(mu: float, order: int) -> dict:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    k = np.asarray((nodes + 1.0) * (math.pi / 4.0), dtype=np.longdouble)
    normalized_weights = np.asarray(weights / 2.0, dtype=np.longdouble)
    sine2 = np.sin(k) ** 2
    cosine2 = np.cos(k) ** 2
    spectrum = sine2[:, None, None] + sine2[None, :, None] + sine2[None, None, :]
    grouped_integral = np.zeros_like(spectrum)
    grouped_derivative = np.zeros_like(spectrum)
    mu2 = np.longdouble(mu) ** 2
    for sign, mass_square in zip(ETA, RECTANGLE):
        denominator = spectrum + mu2 * mass_square
        root = np.sqrt(denominator)
        grouped_integral += sign / root
        grouped_derivative += sign * mass_square / (denominator * root)
    product_weights = (
        normalized_weights[:, None, None]
        * normalized_weights[None, :, None]
        * normalized_weights[None, None, :]
        * cosine2[:, None, None]
    )
    integral = np.sum(product_weights * grouped_integral, dtype=np.longdouble)
    derivative = -mu2 * np.sum(product_weights * grouped_derivative, dtype=np.longdouble)
    if integral <= 0:
        raise RuntimeError(f"Nonpositive BZ I at order={order}")
    defect = -4 + 2 * derivative / integral
    return {"order_per_axis": order, "I": float(integral), "mu_Iprime": float(derivative), "q": float(defect)}


def execute() -> dict:
    started = time.monotonic()
    report = {
        "scope": "One (1,2,2,3) rectangle; thermodynamic zero-momentum sine coefficient; sole finite-a match; no source or exterior solution.",
        "status": "running", "rectangle": list(RECTANGLE), "eta": list(ETA),
        "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                    "host": platform.node(), "nice": os.getpriority(os.PRIO_PROCESS, 0),
                    "threads": 1, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]},
        "settings": {"heat_epsabs": EPSABS, "heat_epsrel": EPSREL, "heat_limit": 400,
                     "scan_mu": [0.05, 0.5], "scan_points": 17,
                     "root_bracket_relative_width_target": 1e-11, "bz_orders": [32, 48, 64]},
        "error_scope": "Adaptive-quadrature estimates and convergence controls; no interval certificate.",
    }
    try:
        scan = [primary(float(mu)) for mu in np.geomspace(0.05, 0.5, 17)]
        report["scan"] = scan
        pairs = [(left, right) for left, right in zip(scan[:-1], scan[1:]) if left["q"] * right["q"] < 0]
        if not pairs:
            raise RuntimeError("No sign-change bracket in specified mu scan")
        left, right = pairs[0]
        report["initial_bracket"] = [left, right]
        iterations = 0
        while (right["mu"] - left["mu"]) / ((right["mu"] + left["mu"]) / 2) >= 1e-11:
            middle = primary((right["mu"] + left["mu"]) / 2)
            if left["q"] * middle["q"] <= 0:
                right = middle
            else:
                left = middle
            iterations += 1
            if iterations > 100:
                raise RuntimeError("Safeguarded bisection failed to converge")
        root = primary((left["mu"] + right["mu"]) / 2)
        mu = root["mu"]
        spacing = math.sqrt(6 * math.pi * root["I"])
        root.update({"mu_squared": mu * mu, "a_over_lP": spacing,
                     "M_times_lP": mu / spacing, "maximum_am_s": math.sqrt(3) * mu,
                     "bracket_relative_width": (right["mu"] - left["mu"]) / mu,
                     "bisection_iterations": iterations})
        report["final_bracket"] = [left, right]
        report["root"] = root
        rows = [brillouin(mu, order) for order in (32, 48, 64)]
        report["brillouin_convergence"] = rows
        convergence = {
            "last_two_I_relative_difference": abs(rows[-1]["I"] - rows[-2]["I"]) / abs(rows[-1]["I"]),
            "last_two_q_absolute_difference": abs(rows[-1]["q"] - rows[-2]["q"]),
            "heat_BZ_I_relative_difference": abs(rows[-1]["I"] - root["I"]) / abs(root["I"]),
            "heat_BZ_q_absolute_difference": abs(rows[-1]["q"] - root["q"]),
        }
        report["agreement"] = convergence
        at_two = primary(4 * mu)
        report["coefficient_at_P2"] = {"heat_integral_at_4mu": at_two,
                                     "k_a_P2_over_P1": at_two["I"] / (16 * root["I"])}
        checks = {
            "root_residual": abs(root["q"]) < 1e-9,
            "final_bracket_width": root["bracket_relative_width"] < 1e-8,
            "final_bracket_sign_change": left["q"] * right["q"] <= 0,
            "last_two_I": convergence["last_two_I_relative_difference"] < 1e-8,
            "last_two_q": convergence["last_two_q_absolute_difference"] < 1e-7,
            "heat_BZ_I": convergence["heat_BZ_I_relative_difference"] < 1e-8,
            "heat_BZ_q": convergence["heat_BZ_q_absolute_difference"] < 1e-7,
            "external_spacing_comparison": 0.1497 < spacing < 0.1500,
        }
        report["checks"] = checks
        report["status"] = "passed" if all(checks.values()) else "failed"
    except Exception:
        report["status"] = "failed"
        report["failure"] = traceback.format_exc()
    report["elapsed_seconds"] = time.monotonic() - started
    return report


if __name__ == "__main__":
    data = execute()
    (DESTINATION / "results.json").write_text(json.dumps(data, indent=2) + "\n")
    lines = ["# Finite-spacing rectangle-zero controls", "", data["scope"], "", data["error_scope"], "", "Status: " + data["status"] + ".", ""]
    for name in ("root", "final_bracket", "brillouin_convergence", "agreement", "coefficient_at_P2", "checks", "failure"):
        if name in data:
            lines.extend([name + ":", "", "```json", json.dumps(data[name], indent=2), "```", ""])
    (DESTINATION / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": data["status"], "root": data.get("root"), "agreement": data.get("agreement"), "checks": data.get("checks"), "elapsed_seconds": data["elapsed_seconds"]}), flush=True)
    raise SystemExit(0 if data["status"] == "passed" else 1)
