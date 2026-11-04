#!/usr/bin/env python3
"""Translate reviewed X1 W1--W24; CPU qi2 box only, no history solver.

The scalar/reweight stage never solves a greybody ODE. New IR/support rows
belong to a separate bounded stage after inspection by the original pair.
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
import signal
import socket
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import mpmath as mp

START = time.monotonic()
CPU_START = time.process_time()
WALL_LIMIT = 60
RSS_LIMIT_KIB = 512 * 1024
ROOT_EVALUATIONS = 0
CRITERIA = ("0.95", "0.80", "0.50")
SIZES = (64, 128, 326)
CONVENTIONS = ("gap", "lapse")


def budget(*_args) -> None:
    if time.monotonic() - START >= WALL_LIMIT:
        raise TimeoutError("scalar/reweight wall limit reached")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >= RSS_LIMIT_KIB:
        raise MemoryError("scalar/reweight peak RSS limit reached")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    obj["resources"] = {
        "elapsed_wall_seconds": time.monotonic() - START,
        "cpu_seconds": time.process_time() - CPU_START,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "root_function_evaluations": ROOT_EVALUATIONS,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def F(y):
    return 2 * y**3 / 3 + y**2 + 2 * y + 2 * mp.log(y - 1)


def P(y):
    return (y - 1) * (y + 1)**3 / (y**3 * (y + 2))


def A(y):
    return 6 / (y**4 * (y + 1) * (y + 2))


def T(y):
    return -2 * y**3 / 3 - 2 * y - mp.log((y - 1) / (y + 1))


def bisection(function, low, high, increasing: bool, digits: int):
    """Interior evaluation only; logarithmic endpoint can be divergent."""
    global ROOT_EVALUATIONS
    count = math.ceil((digits + 4) * math.log2(10))
    for _ in range(count):
        budget()
        middle = (low + high) / 2
        value = function(middle)
        ROOT_EVALUATIONS += 1
        if (value < 0) == increasing:
            low = middle
        else:
            high = middle
    return (low + high) / 2, low, high


def encoded(value, digits: int):
    if isinstance(value, dict):
        return {k: encoded(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encoded(v, digits) for v in value]
    if isinstance(value, mp.mpf):
        return mp.nstr(value, digits + 12)
    return value


def scalar_stage(digits: int) -> dict:
    with mp.workdps(digits + 18):
        zero, one = mp.mpf(0), mp.mpf(1)
        freqs = [mp.mpf("0.001"), 1 / (4 * mp.pi)]
        freqs += [mp.mpf(x) for x in ("0.1", "0.2", "0.25", "0.5", "1", "2")]
        result = {"decimal_digits": digits, "onsets": [], "segments": [],
                  "thresholds": [], "period_maxima": [], "integral_checks": []}
        for criterion in CRITERIA:
            c = mp.mpf(criterion)
            fn = lambda y: y**4 * (y + 1) * (y + 2) - 6 / c
            yc, low, high = bisection(fn, one, mp.mpf(2), True, digits)
            fc, dc, zc = yc**2 - 1, 1 - 1 / yc**2, yc - 1
            constant = F(yc) - mp.mpf(11) / 3 + 2 * mp.log(2)
            rational_brackets = {"0.95": ("1.01066", "1.01067"),
                                 "0.80": ("1.0471", "1.0472"),
                                 "0.50": ("1.1530", "1.1531")}
            rlo, rhi = map(mp.mpf, rational_brackets[criterion])
            logderivative = 1 / (yc - 1) + 3 / (yc + 1) - 3 / yc - 1 / (yc + 2)
            result["onsets"].append({
                "criterion": criterion, "y": yc, "z": zc, "gap": fc,
                "lapse": dc, "constant": constant, "two_log_gap": 2 * mp.log(fc),
                "u_from_y2_over_rs": F(2) - F(yc), "bracket": [low, high],
                "bracket_width": high - low, "residual": fn(yc),
                "A_log_derivative": -4 / yc - 1 / (yc + 1) - 1 / (yc + 2),
                "endpoint_residuals": [fn(low), fn(high)],
                "rational_bracket": rational_brackets[criterion],
                "rational_endpoint_residuals": [fn(rlo), fn(rhi)],
                "A_log_derivative_difference": (yc - 1) / yc**3 * logderivative - A(yc),
            })
            for convention in CONVENTIONS:
                fraction = fc if convention == "gap" else dc
                for size in SIZES:
                    qmax = mp.mpf(size) / 2 * fraction**mp.mpf("1.5")
                    for index, q in enumerate(freqs + [qmax]):
                        b = (2 * q / size)**(mp.mpf(2) / 3)
                        outside = convention == "lapse" and b >= 1
                        yd = None if outside else (mp.sqrt(1 + b) if convention == "gap"
                                                   else 1 / mp.sqrt(1 - b))
                        empty = outside or q >= qmax
                        duration = zero if empty else F(yc) - F(yd)
                        npeel = zero if empty else mp.log(P(yc) / P(yd))
                        delta_t = zero if empty else T(yd) - T(yc)
                        lead = 4 * mp.log(size / (2 * q)) / 3
                        row = {"criterion": criterion, "convention": convention,
                               "size": size, "frequency_index": index, "q": q,
                               "is_support_endpoint": index == len(freqs), "b": b,
                               "y_endpoint": yd, "q_max": qmax,
                               "outside_convention": outside, "empty": empty,
                               "D": duration, "N_H": duration / 2, "N_peel": npeel,
                               "leading_duration": lead, "constant_approximation": lead + constant,
                               "exact_minus_constant_approximation": duration - lead - constant,
                               "inverse_temperature_units": duration / (4 * mp.pi),
                               "oscillation_cycles": q * duration / (2 * mp.pi),
                               "zero_constant_N_H": lead / 2, "delta_T_over_rs": delta_t,
                               "clock_difference": duration - 2 * delta_t,
                               "peeling_lower_defect": min(zero, npeel - c * duration / 2),
                               "peeling_upper_defect": max(zero, npeel - duration / 2)}
                        if yd is not None:
                            s = 1 / yd
                            p0 = (q / size) / (1 - s)
                            linear = (1 - s) * p0
                            exact = mp.sin(p0) - s * mp.sin(2 * p0) / 2
                            cubic = (2 * s / 3 - mp.mpf(1) / 6) * p0**3
                            remainder = (1 + 16 * s) * abs(p0)**5 / 120
                            row.update({"A_endpoint": A(yd), "symbol_p0": p0,
                                        "p0_positive_axial_halfbranch": 0 < p0 < mp.pi / 2,
                                        "exact_symbol_minus_linear": exact - linear,
                                        "exact_symbol_correction_ratio": (exact - linear) / linear,
                                        "cubic_correction_ratio": cubic / linear,
                                        "fifth_order_allowance": remainder,
                                        "fifth_order_residual_defect": max(zero, abs(exact - linear - cubic) - remainder),
                                        "maximum_adiabatic_diagnostic": None if empty else
                                        (yc - 1) * (6 * yc**2 + 15 * yc + 8) / 6})
                        result["segments"].append(row)
                    equation = lambda y: F(yc) - F(y) - mp.mpf(2) / 3 * y**(
                        2 if convention == "gap" else 4) * (y + 1)
                    # The maximizer's y does not depend on size: solve once.
                    if size == SIZES[0]:
                        ym, lo, hi = bisection(equation, one, yc, False, digits)
                    fm = ym**2 - 1 if convention == "gap" else 1 - 1 / ym**2
                    qm = mp.mpf(size) / 2 * fm**mp.mpf("1.5")
                    dm = F(yc) - F(ym)
                    result["period_maxima"].append({"criterion": criterion,
                        "convention": convention, "size": size, "y": ym,
                        "bracket": [lo, hi], "bracket_width": hi - lo,
                        "endpoint_residuals": [equation(lo), equation(hi)],
                        "residual": equation(ym),
                        "q": qm, "D": dm, "qD": qm * dm,
                        "cycles": qm * dm / (2 * mp.pi)})
                # One direct finite-endpoint quadrature per criterion/convention.
                qtest = min(mp.mpf("0.1"), mp.mpf(64) / 4 * fraction**mp.mpf("1.5"))
                bb = (2 * qtest / 64)**(mp.mpf(2) / 3)
                yd = mp.sqrt(1 + bb) if convention == "gap" else 1 / mp.sqrt(1 - bb)
                direct_u = mp.quad(lambda y: 2 * y**3 / (y - 1), [yd, yc])
                direct_U = mp.quad(lambda y: -2 * (y + 1)**3 / (y + 2), [yc, yd])
                V = lambda y: -2 * y**3 / 3 + y**2 - 2 * y + 2 * mp.log(y + 1)
                regular_clock = -2 * mp.quad(lambda y: y**3 / (y + 1), [yd, yc])
                result["integral_checks"].append({"criterion": criterion,
                    "convention": convention, "q": qtest,
                    "u_difference_residual": direct_u - (F(yc) - F(yd)),
                    "U_difference_residual": direct_U - (V(yd + 1) - V(yc + 1)),
                    "clock_identity_residual": regular_clock - (direct_u - 2 * (T(yd) - T(yc)))})
            if criterion == "0.95":
                for definition in ("nominal_H", "integrated_peeling"):
                    equation = (lambda y: F(yc) - F(y) - 2) if definition == "nominal_H" else (
                        lambda y: mp.log(P(yc) / P(y)) - 1)
                    ys, lo, hi = bisection(equation, one, yc, False, digits)
                    result["thresholds"].append({"criterion": criterion, "q": 1,
                        "efold_definition": definition, "y": ys,
                        "bracket": [lo, hi], "bracket_width": hi - lo,
                        "endpoint_residuals": [equation(lo), equation(hi)],
                        "residual": equation(ys),
                        "size_gap": 2 / (ys**2 - 1)**mp.mpf("1.5"),
                        "size_lapse": 2 / (1 - 1 / ys**2)**mp.mpf("1.5"),
                        "leading_log_size": 2 * mp.exp(mp.mpf("1.5")) / fc**mp.mpf("1.5")})
        return encoded(result, digits)


def duration_float(q: float, size: float, onset: dict, convention: str) -> float:
    b = (2 * q / size)**(2 / 3)
    if convention == "lapse" and b >= 1:
        return 0.0
    fd = b if convention == "gap" else b / (1 - b)
    fc = float(onset["gap"])
    if fd >= fc:
        return 0.0
    def stable_extra(f):
        z = f / (math.sqrt(1 + f) + 1)
        return 6 * z + 3 * z*z + 2 * z*z*z / 3 - 2 * math.log1p(z / 2)
    return 2 * math.log(fc / fd) + stable_extra(fc) - stable_extra(fd)


def h2(p: float) -> float:
    if not 0 <= p <= 1:
        raise ValueError("unclipped occupation out of [0,1]")
    return 0.0 if p in (0, 1) else -p * math.log(p) - (1 - p) * math.log1p(-p)


def reweight(data: dict, onsets: list[dict]) -> dict:
    result = {"scope": "Inherited finite k<=8, q in [0.001,3]; no new ODEs.",
              "readbacks": [], "comparisons": [], "frequency_tail_allowances": []}
    for readback_index, rb in enumerate(data["readbacks"]):
        if not rb["complete"] or len(rb["rows"]) != rb["expected_scalar_rows"]:
            raise ValueError("inherited readback incomplete")
        sums = {(o["criterion"], cv, k): [0.0, 0.0, 0.0]
                for o in onsets for cv in CONVENTIONS for k in range(1, 9)}
        rows = []
        for index, old in enumerate(rb["rows"]):
            budget()
            k, q, gamma = old["k"], old["x"], old["gamma"]
            if not 0 <= gamma <= 1:
                raise ValueError("inherited gamma outside [0,1]; not clipped")
            f = 1 / (1 + math.exp(4 * math.pi * q))
            occupation = gamma * f
            weight = 32 * k / math.pi * old["weight_dx"]
            new = {"origin_readback": readback_index, "origin_row": index,
                   "k": k, "q": q, "gamma": gamma, "weight_dx": old["weight_dx"],
                   "occupation": occupation, "duration_integrands": []}
            for onset in onsets:
                for cv in CONVENTIONS:
                    d = duration_float(q, 64, onset, cv)
                    vals = [weight * d * h2(occupation), weight * d * occupation,
                            weight * d * q * occupation]
                    total = sums[(onset["criterion"], cv, k)]
                    for j in range(3):
                        total[j] += vals[j]
                    new["duration_integrands"].append({"criterion": onset["criterion"],
                        "convention": cv, "D": d, "weighted_S_N_rsE": vals})
            rows.append(new)
        aggregates = []
        for onset in onsets:
            for cv in CONVENTIONS:
                channel = [sums[(onset["criterion"], cv, k)] for k in range(1, 9)]
                aggregates.append({"criterion": onset["criterion"], "convention": cv,
                    "per_channel_S_N_rsE": channel,
                    "cumulative": [{"k_cap": cap, "S_N_rsE": [math.fsum(
                        channel[k][j] for k in range(cap)) for j in range(3)]}
                        for cap in (1, 2, 4, 6, 8)]})
        result["readbacks"].append({"settings": rb["settings"], "rows": rows,
            "aggregates": aggregates, "inherited_diagnostics": {
                "max_abs_flux_defect": max(abs(x["flux_defect"]) for x in rb["rows"]),
                "max_matching_residual": max(x["relative_matching_residual"] for x in rb["rows"])}})
    for onset in onsets:
        cst = float(onset["constant"])
        aa = 4 / 3 * math.log(32) + cst
        gk, beta = 16 * 8 * 9 / math.pi, 4 * math.pi
        for cv in CONVENTIONS:
            def ir_bound(q0):
                b0 = q0 * (aa + 4 / 3 * (1 - math.log(q0)))
                return [gk * math.log(2) * b0, gk * b0 / 2, gk * q0 * b0 / 2]
            dq = duration_float(3, 64, onset, cv)
            uv = [gk * dq * math.exp(-3 * beta) * (3 + 2 / beta),
                  gk * dq * math.exp(-3 * beta) / beta,
                  gk * dq * math.exp(-3 * beta) * (3 / beta + beta**-2)]
            a, b = [next(x for x in rb["aggregates"] if x["criterion"] == onset["criterion"]
                         and x["convention"] == cv)["cumulative"][-1]["S_N_rsE"]
                    for rb in result["readbacks"]]
            delta = [b[j] - a[j] for j in range(3)]
            result["comparisons"].append({"criterion": onset["criterion"], "convention": cv,
                "refined_minus_first_S_N_rsE": delta,
                "support_refinement_requested": abs(delta[0]) > max(1e-4, 0.005 * abs(b[0])),
                "empirical_indicator_only": True})
            result["frequency_tail_allowances"].append({"criterion": onset["criterion"],
                "convention": cv, "inherited_omitted_IR_S_N_rsE_bound": ir_bound(1e-3),
                "target_omitted_IR_S_N_rsE_bound": ir_bound(1e-8), "UV_above_3_S_N_rsE_bound": uv})
    result["unresolved"] = ["Actual transmission errors are not certified by requested solver tolerance.",
        "Support-kink quadrature and changed endpoint/tolerance are confounded in inherited comparison.",
        "IR interval [1e-8,1e-3] requires its separately prescribed bounded computation.",
        "All angular channels k>8 remain unbounded.",
        "Stationary-rate duration weighting does not define actual finite-time quantum entropy."]
    return result


def precision_comparison(first: dict, refined: dict) -> dict:
    """Retain explicit changes of all high-precision numeric leaf fields."""
    changes = []
    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in a.keys() & b.keys():
                walk(a[key], b[key], path + "/" + str(key))
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                raise ValueError("precision-stage row counts differ")
            for i, (aa, bb) in enumerate(zip(a, b)):
                walk(aa, bb, path + "/" + str(i))
        elif isinstance(a, str) and isinstance(b, str):
            try:
                aa, bb = mp.mpf(a), mp.mpf(b)
            except ValueError:
                return
            error = abs(bb - aa)
            changes.append({"path": path, "absolute_change": mp.nstr(error, 90),
                "relative_change": None if bb == 0 else mp.nstr(error / abs(bb), 90)})
    with mp.workdps(105):
        walk(first, refined, "")
    return {"scope": "Observed 50/80-digit changes, not rigorous transcendental enclosures.",
            "numeric_leaf_changes": changes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inherited", type=Path, required=True)
    parser.add_argument("--inherited-sha256", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refuse to overwrite saved numerical output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGALRM, budget)
    signal.alarm(WALL_LIMIT)
    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024**2, 768 * 1024**2))
    result = {"task": "WEB40I X1 scalar ray-map and inherited Gamma reweight",
        "source_sha256": digest(Path(__file__)), "host": socket.gethostname(),
        "python": platform.python_version(), "mpmath": mp.__version__,
        "nice": os.getpriority(os.PRIO_PROCESS, 0), "threads": 1,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "wall_limit_seconds": WALL_LIMIT, "rss_limit_kib": RSS_LIMIT_KIB,
        "address_space_limit_bytes": 768 * 1024**2,
        "complete": False, "new_ode_solves": 0,
        "scope": "Exact-map scalar comparator; inherited finite-domain stationary-rate readout only."}
    try:
        actual = digest(args.inherited)
        if actual != args.inherited_sha256:
            raise ValueError("inherited input does not match prescribed SHA256")
        result["inherited"] = {"path": str(args.inherited), "sha256": actual}
        data = json.loads(args.inherited.read_text())
        result["scalar_50"] = scalar_stage(50)
        atomic_json(args.output, result)
        result["scalar_80"] = scalar_stage(80)
        result["precision_comparison"] = precision_comparison(result["scalar_50"], result["scalar_80"])
        atomic_json(args.output, result)
        result["content"] = reweight(data, result["scalar_80"]["onsets"])
        result["complete"] = True
    except Exception as error:
        result["error"] = type(error).__name__ + ": " + str(error)
    finally:
        atomic_json(args.output, result)
        signal.alarm(0)
        print(json.dumps({"complete": result["complete"], "error": result.get("error"),
                          "new_ode_solves": 0, "resources": result["resources"]}))


if __name__ == "__main__":
    main()
