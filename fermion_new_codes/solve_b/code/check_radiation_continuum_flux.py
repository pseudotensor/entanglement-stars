#!/usr/bin/env python3
"""The two fixed R2C7--R2C9 continuum controls; run on the qi2 box only.

Stream two-component Jost solves and retain scalar records. This is neither a
Page evolution nor a certified infinite-frequency/angular-tail calculation.
The prescribed ten-minute cap returns partial output rather than restarting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import socket
import time

START = time.monotonic()
WALL_LIMIT = 600.0
RSS_LIMIT_KIB = 256 * 1024
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import scipy
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp


class BudgetExhausted(Exception):
    pass


def check_budget() -> None:
    if time.monotonic() - START >= WALL_LIMIT:
        raise BudgetExhausted("ten-minute wall budget exhausted")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >= RSS_LIMIT_KIB:
        raise BudgetExhausted("256 MiB RSS budget exhausted")


def atomic_json(path: Path, result: dict) -> None:
    result["elapsed_seconds"] = time.monotonic() - START
    result["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def inner_endpoint(k: int, x: float) -> tuple[complex, complex, float]:
    """R2C8 in the sigma2 eigenbasis (u_minus, u_plus)."""
    minus = np.zeros(9, dtype=complex)
    plus = np.zeros(9, dtype=complex)
    minus[0] = 1.0

    def at(values: np.ndarray, n: int) -> complex:
        return values[n] if n >= 0 else 0j

    for n in range(1, 9):
        minus[n] = (2 * (n - 2) * at(minus, n - 2)
                    - (n - 4) * at(minus, n - 4)
                    - 2 * k * at(plus, n - 1)
                    + 2 * k * at(plus, n - 3)) / n
        plus[n] = (2 * (n - 2) * at(plus, n - 2)
                   - (n - 4) * at(plus, n - 4)
                   - 2 * k * at(minus, n - 1)
                   + 2 * k * at(minus, n - 3)) / (n - 4j * x)
    z = 1e-4
    dm = np.polynomial.polynomial.polyval(z, minus)
    ep = np.polynomial.polynomial.polyval(z, plus)
    flux = float(abs(ep) ** 2 - abs(dm) ** 2)
    if not flux < 0:
        raise ValueError("inner endpoint has wrong signed flux")
    norm = math.sqrt(-flux)
    y = 1.0 / (1.0 - z * z)
    xi = math.log(z * z / (1.0 - z * z))
    s = y + xi
    return dm / norm, ep * np.exp(-2j * x * s) / norm, flux


def outer_column(k: int, x: float, sign: int, order: int,
                 y: float) -> tuple[complex, complex, float]:
    """R2C9, with polynomial sums before the free oscillatory phase."""
    d = np.zeros(order + 1, dtype=complex)
    e = np.zeros(order + 1, dtype=complex)
    bs = np.ones(order + 1)
    for j in range(1, order + 1):
        bs[j] = bs[j - 1] * (j - 1.5) / j
    d[0] = 1.0
    for n in range(1, order + 1):
        e_previous = e[n - 2] if n >= 2 else 0j
        e[n] = ((n - 1) * e[n - 1] - (n - 2) * e_previous
                - k * np.dot(bs[:n], d[n - 1::-1])) / (2j * sign * x)
        d[n] = ((n - 1) * d[n - 1]
                + k * np.dot(bs[:n + 1], e[n::-1])) / n
    ds = np.polynomial.polynomial.polyval(1.0 / y, d)
    es = np.polynomial.polynomial.polyval(1.0 / y, e)
    raw_flux = float(sign * (abs(ds) ** 2 - abs(es) ** 2))
    if not sign * raw_flux > 0:
        raise ValueError("outer endpoint has wrong signed flux")
    norm = math.sqrt(sign * raw_flux)
    return ds / norm, es / norm, raw_flux


def transmission(k: int, x: float, setting: dict) -> dict:
    check_budget()
    z = 1e-4
    xi0 = math.log(z * z / (1.0 - z * z))
    outer_y = setting["outer_multiplier"] * max(80.0, 40.0 * k * k / x)
    xi1 = math.log(outer_y - 1.0)
    a0, b0, inner_flux = inner_endpoint(k, x)
    calls = 0

    def rhs(xi: float, state: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls % 128 == 1:
            check_budget()
        t = math.exp(xi)
        phase = 2.0 * x * (1.0 + t + xi)
        oscillation = complex(math.cos(phase), math.sin(phase))
        coupling = -k * math.sqrt(t / (1.0 + t))
        # phi=u_- exp(-ixs) a + u_+ exp(ixs) b exactly.
        return np.array((coupling * oscillation * state[1],
                         coupling * oscillation.conjugate() * state[0]))

    integration = solve_ivp(rhs, (xi0, xi1), np.array((a0, b0)),
                            method="DOP853", rtol=setting["rtol"],
                            atol=setting["atol"], t_eval=[xi1])
    if not integration.success:
        raise ValueError("DOP853 failure: " + integration.message)
    final = integration.y[:, -1]
    dm, em, flux_m = outer_column(k, x, -1, setting["outer_order"], outer_y)
    dp, ep, flux_p = outer_column(k, x, 1, setting["outer_order"], outer_y)
    phase = 2.0 * x * (outer_y + xi1)
    oscillation = complex(math.cos(phase), math.sin(phase))
    columns = np.array(((dm, ep * oscillation),
                        (em * oscillation.conjugate(), dp)))
    ain, aout = np.linalg.solve(columns, final)
    gamma = float(1.0 / abs(ain) ** 2)
    reflection = float(abs(aout / ain) ** 2)
    matching = float(np.linalg.norm(columns @ np.array((ain, aout)) - final)
                     / np.linalg.norm(final))
    return {"k": k, "x": x, "outer_y": outer_y,
            "gamma": gamma, "reflection": reflection,
            "flux_defect": gamma + reflection - 1.0,
            "gamma_range_defect": max(0.0, -gamma, gamma - 1.0),
            "inner_raw_flux": inner_flux, "outer_raw_flux_minus": flux_m,
            "outer_raw_flux_plus": flux_p, "relative_matching_residual": matching,
            "ode_evaluations": int(integration.nfev)}


def entropy(p: float) -> float:
    if not 0.0 <= p <= 1.0:
        raise ValueError("unclipped outgoing occupation outside [0,1]")
    if p == 0.0 or p == 1.0:
        return 0.0
    return -p * math.log(p) - (1.0 - p) * math.log1p(-p)


def aggregate(rows: list[dict], cap: int) -> dict:
    selected = [row for row in rows if row["k"] <= cap]
    luminosity = number = entropic = 0.0
    for row in selected:
        factor = 8.0 / math.pi * row["weight_dx"] * 4 * row["k"]
        occupation = row["gamma"] / (math.exp(4.0 * math.pi * row["x"]) + 1.0)
        luminosity += factor * row["x"] * occupation
        number += factor * occupation
        entropic += factor * entropy(occupation)
    return {"k_cap": cap, "completed_scalar_rows": len(selected),
            "Lcal": luminosity, "Ncal": number, "Scal": entropic,
            "b": entropic / (4 * math.pi * luminosity) if luminosity else None,
            "max_abs_flux_defect": max((abs(r["flux_defect"]) for r in selected),
                                       default=None),
            "max_gamma_range_defect": max((r["gamma_range_defect"] for r in selected),
                                          default=None)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__)
    result = {
        "control": "WEB40E I4 R2C7--R2C9 fixed continuum flux readbacks",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "host": socket.gethostname(), "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "threads": {name: os.environ[name] for name in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "wall_limit_seconds": WALL_LIMIT, "rss_limit_kib": RSS_LIMIT_KIB,
        "rs": 1, "x_range": [1e-3, 3.0], "inner_z": 1e-4, "inner_order": 8,
        "complete": False, "reason": "running", "readbacks": [],
        "scope": "Observed finite endpoint/quadrature controls only; no certified "
                 "frequency or angular tails, native evolution, Page run or state count.",
    }
    atomic_json(args.output, result)
    settings = (
        {"name": "first", "nx": 64, "outer_order": 10,
         "outer_multiplier": 1, "rtol": 1e-8, "atol": 1e-10},
        {"name": "refined", "nx": 96, "outer_order": 12,
         "outer_multiplier": 2, "rtol": 2e-9, "atol": 2e-11},
    )
    try:
        for setting in settings:
            node, weight = leggauss(setting["nx"])
            lo, hi = math.log(1e-3), math.log(3.0)
            logx = (hi + lo) / 2 + (hi - lo) / 2 * node
            xvalues = np.exp(logx)
            dxweights = weight * (hi - lo) / 2 * xvalues
            readback = {"settings": setting, "complete": False,
                        "expected_scalar_rows": 8 * setting["nx"], "rows": []}
            result["readbacks"].append(readback)
            for k in range(1, 9):
                for j, (x, dxweight) in enumerate(zip(xvalues, dxweights)):
                    row = transmission(k, float(x), setting)
                    row.update({"quadrature_index": j, "weight_dx": float(dxweight)})
                    readback["rows"].append(row)
                    readback["k_le_6"] = aggregate(readback["rows"], 6)
                    readback["k_le_8"] = aggregate(readback["rows"], 8)
                    atomic_json(args.output, result)
                print(json.dumps({"readback": setting["name"], "completed_k": k,
                                  "elapsed_seconds": time.monotonic() - START}), flush=True)
            readback["complete"] = True
            atomic_json(args.output, result)
        result["complete"] = True
        result["reason"] = "both prescribed finite readbacks completed"
    except BudgetExhausted as error:
        result["reason"] = str(error)
    except Exception as error:
        result["reason"] = type(error).__name__ + ": " + str(error)
        result["failed"] = True
    finally:
        atomic_json(args.output, result)
        print(json.dumps({"complete": result["complete"], "reason": result["reason"],
                          "elapsed_seconds": result["elapsed_seconds"],
                          "max_rss_kib": result["max_rss_kib"]}), flush=True)


if __name__ == "__main__":
    main()
