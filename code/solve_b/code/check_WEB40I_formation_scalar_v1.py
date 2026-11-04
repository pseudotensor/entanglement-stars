#!/usr/bin/env python3
"""X4/Y3/Y1: the combined point/sine and bound-rest scalar producer.

Implements F2, F4 and F7--F9 of WEB40I_formation_comparator_v1.tex.
One supplied metadata case, at the two prescribed precisions only.
The canonical production T0 count/current export and its lineage are not
supplied here. No profile, field solve, trajectory ODE, multistream history
or root-formation value is manufactured. The full X4 task stays incomplete.
The ideal bound-rest map implements the original pair's closed B8--B24
contract in its own module. The combined code is unrun and needs actual
original-pair code/output assessment. It does not supply the production T0.

Scientific execution is restricted to the box's qi2 interpreter. Review this
source before running it; local syntax checking must not execute the module.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time


SPEC_PATH = "<internal>/WEB40I_formation_comparator_v1.tex"
SPEC_SHA256 = "989906d849eabbc95d4be437320bd24708e67003bf3ac8b271557af2db047f8a"
BOX_PYTHON = Path("<internal>/python")
BULK_ROOT = Path("<internal>/solve_b_bulk")
MEMORY_BYTES = 4 * 1024**3
SCALAR_CAP_SECONDS = 28
OUTER_CAP_SECONDS = 30
WHOLE_RECIPE_CAP_SECONDS = 1200
PRECISIONS = (50, 80)
METADATA = (
    (2, "<internal>/v36d_from0"
     "collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_plain_v36d_from0.json",
     "248af7288b6c15e6591f279847708fab5e2130ebdf2b09e81eb22e4cb3a82e79"),
    (4, "<internal>/v36d_from0"
     "collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0.json",
     "c733341fa3c75e3e61dc757d57359dfa408ebc9b287a84d30422ce909d9fadc6"),
    (8, "<internal>/v36d_from0"
     "collapse_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0.json",
     "246e51d95a91c9459ca69619a243bcc55a5e3224245803221333c476ee71c9b9"),
)
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


class ScalarBudgetExpired(RuntimeError):
    pass


def expire(_signum, _frame):
    raise ScalarBudgetExpired("28-second graceful deadline for the combined scalar stage reached")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_save(path, data):
    temp = path.with_name(path.name + ".partial")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, default=json_default, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def inspect_metadata(width, name, expected):
    """Read named small JSONs in place; never import a foreign producer."""
    path = Path(name)
    item = {"width_over_a": width, "path": name, "expected_sha256": expected}
    try:
        if path.stat().st_size > 5 * 1024**2:
            raise ValueError("named metadata JSON exceeds the small-input ceiling")
        raw = path.read_bytes()
        item.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        if item["sha256"] != expected:
            item["status"] = "pinned_metadata_identity_mismatch"
            return item, None
        data = json.loads(raw, parse_float=Decimal)
        args, head, first = data["args"], data["head"], data["records"][0]
        arg_keys = ("m", "g", "rs", "R0_over_rs", "ell", "prep_ell", "tau",
                    "reference", "mix", "reading", "kappa_sea", "nsites",
                    "v36_fixes", "slices_c128", "resume_slice")
        head_keys = ("E0", "N_B", "R0", "rs_rest_label", "prep_ell", "evolution_ell")
        item.update(script_label=data.get("script"),
                    args={k: args.get(k) for k in arg_keys},
                    head={k: head.get(k) for k in head_keys},
                    first_record={k: first.get(k) for k in ("T", "E", "slice_saved")})
        required = {"m": Decimal("0.2"), "g": Decimal("0.25"),
                    "rs": Decimal("64"), "R0_over_rs": Decimal("4"),
                    "ell": width, "prep_ell": 4}
        if any(args[k] != value for k, value in required.items()):
            raise ValueError("metadata arguments differ from the prescribed case")
        if any(head[k] != value for k, value in
               {"E0": Decimal("118.13786225596981"), "N_B": 640,
                "R0": 256, "rs_rest_label": 64, "prep_ell": 4,
                "evolution_ell": width}.items()):
            raise ValueError("metadata head differs from the prescribed case")
        if first["T"] != 0:
            raise ValueError("the first record is not T=0")
        item["status"] = "pinned_metadata_verified_only"
        case = {"a": "1", "ma": str(args["m"]), "G_over_a2": str(args["g"]),
                "r0_over_a": str(args["rs"]), "R0_over_a": str(head["R0"]),
                "E0_times_a": str(head["E0"]), "N_B": str(head["N_B"])}
        return item, case
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        item.update(status="metadata_unavailable_or_invalid",
                    error_type=type(error).__name__, error=str(error))
        return item, None


def bisect(mp, function, lo, hi, digits, observe=None):
    flo, fhi = function(lo), function(hi)
    if observe is not None:
        observe({"lower": lo, "upper": hi, "f_lower": flo, "f_upper": fhi,
                 "iterations": 0, "function_evaluations": 2})
    if flo * fhi >= 0:
        raise ValueError("missing numerical sign-changing root bracket")
    tolerance = mp.power(10, -(digits - 10))
    for iteration in range(1, 4097):
        mid = (lo + hi) / 2
        fm = function(mid)
        if observe is not None:
            observe({"lower": lo, "upper": hi, "f_lower": flo, "f_upper": fhi,
                     "midpoint": mid, "residual": fm, "iterations": iteration,
                     "function_evaluations": iteration+2})
        if hi - lo <= tolerance and abs(fm) <= tolerance:
            return mid, {"lower": lo, "upper": hi, "f_lower": flo,
                         "f_upper": fhi, "residual": fm,
                         "width": hi - lo, "iterations": iteration,
                         "function_evaluations": iteration + 2,
                         "stopping_tolerance": tolerance}
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    raise ArithmeticError("bisection iteration limit")


def evaluate_precision(mp, digits, case, checkpoint):
    partial = {"status": "started", "decimal_working_precision": digits,
               "case": case, "active_operation": "parameters",
               "completed_quantities": {}, "completed_quadratures": {},
               "root_function_evaluations": {}, "quadrature_function_evaluations": {}}
    try:
        return _evaluate_precision(mp, digits, case, partial, checkpoint)
    except BaseException:
        partial["status"] = "interrupted_or_failed"
        checkpoint(partial)
        raise


def _evaluate_precision(mp, digits, case, partial, checkpoint):
    with mp.workdps(digits):
        fmt = lambda number: mp.nstr(number, n=digits, strip_zeros=False)
        r0, radius = mp.mpf(case["r0_over_a"]), mp.mpf(case["R0_over_a"])
        charge, coupling, mu = (mp.mpf(case[k]) for k in
                               ("E0_times_a", "G_over_a2", "ma"))
        rg = 2 * coupling * charge
        c, x0 = rg / radius, radius / rg
        e, b0 = mp.sqrt(1 - c), mp.sqrt(c)
        amplitude = mp.sqrt(radius**3 / rg)
        theta_h = mp.acos(mp.sqrt(c))
        quad_calls = {}
        partial["quadrature_function_evaluations"] = quad_calls
        def quadrature(name, function, interval):
            partial["active_operation"] = name
            partial["active_quadrature"] = {"name": name, "interval": [fmt(x) for x in interval]}
            quad_calls[name] = 0
            def counted(point):
                quad_calls[name] += 1
                partial["active_quadrature"]["last_abscissa"] = fmt(point)
                value = function(point)
                partial["active_quadrature"]["last_integrand_value"] = fmt(value)
                return value
            value, error = mp.quad(counted, interval, error=True)
            partial["completed_quadratures"][name] = {"integral": fmt(value),
                                                      "error_estimate": fmt(error),
                                                      "function_evaluations": quad_calls[name]}
            checkpoint(partial)
            return value, error

        def root(name, function, lo, hi):
            partial["active_operation"] = name
            partial["active_root"] = {"name": name, "lower": fmt(lo), "upper": fmt(hi)}
            partial["root_function_evaluations"][name] = 0
            def counted(z):
                partial["root_function_evaluations"][name] += 1
                partial["active_root"]["last_abscissa"] = fmt(z)
                value = function(z)
                partial["active_root"]["last_value"] = fmt(value)
                return value
            def observed(values):
                partial["active_root"].update({key: value if isinstance(value, int) else fmt(value)
                                               for key, value in values.items()})
            value, bracket = bisect(mp, counted, lo, hi, digits, observed)
            partial["completed_quantities"][name] = fmt(value)
            checkpoint(partial)
            return value, bracket
        continuum_integrand = lambda u: mp.cos(u)**2 * (1 + e * mp.sin(u)) / (e + mp.sin(u))
        root_quad, root_error = quadrature("continuum_crossing", continuum_integrand, [0, theta_h])
        continuum_cross = 2 * amplitude * root_quad
        continuum_proper_cross = amplitude * (theta_h + mp.sin(theta_h) * mp.cos(theta_h))
        partial["completed_quantities"].update(continuum_crossing_t_over_a=fmt(continuum_cross),
                                                continuum_crossing_proper_over_a=fmt(continuum_proper_cross))
        checkpoint(partial)
        center_quad, center_error = quadrature("continuum_center", continuum_integrand, [0, mp.pi/2])
        partial["active_operation"] = "formal_point_center_closed_form"
        continuum_center = rg * (mp.pi/2 * (x0 + 2) * mp.sqrt(x0 - 1)
                                - 2 * mp.sqrt(x0)
                                - mp.log((mp.sqrt(x0) - 1)/(mp.sqrt(x0) + 1)))
        continuum_proper_center = rg * mp.pi/2 * x0**mp.mpf("1.5")
        partial["completed_quantities"].update(formal_point_center_t_over_a=fmt(continuum_center),
                                                formal_point_center_proper_over_a=fmt(continuum_proper_center))

        energy = lambda z: mp.sqrt(mu**2 + mp.sin(z)**2)
        current = lambda z: mp.sin(2*z) / 2
        energy_prime = lambda z: mp.sin(z) * mp.cos(z) / energy(z)
        current_prime = lambda z: mp.cos(2*z)
        z0, release_bracket = root("sine_release_root", lambda z: energy_prime(z) - b0*current_prime(z),
                                  mp.mpf(0), mp.pi/4)
        ea = energy(z0) - b0*current(z0)
        partial["completed_quantities"]["sine_release_energy"] = fmt(ea)
        zh, crossing_bracket = root("sine_crossing_root", lambda z: energy(z) - current(z) - ea,
                                    mp.mpf(0), z0)
        sine_integrand = lambda z: current(z)**2 / (energy(z) - ea)**3
        sine_quad, sine_error = quadrature("sine_crossing", sine_integrand, [zh, z0])
        sine_cross = 2 * rg * sine_quad
        partial["completed_quantities"]["sine_crossing_t_over_a"] = fmt(sine_cross)

        diagnostics = []
        partial["completed_F6_F8_F9_diagnostics"] = diagnostics
        partial["active_operation"] = "fixed_F6_F8_F9_diagnostics"
        # Fixed three-point Gauss abscissas inside this one integration interval.
        fractions = ((1-mp.sqrt(mp.mpf(3)/5))/2, mp.mpf("0.5"),
                     (1+mp.sqrt(mp.mpf(3)/5))/2)
        for fraction in fractions:
            z = zh + fraction * (z0-zh)
            q, s = current(z), energy(z)
            r = rg*q**2/(s-ea)**2
            b = (s-ea)/q
            r_prime = 2*r*(current_prime(z)/q - energy_prime(z)/(s-ea))
            dt_dz = -2*rg*q**2/(s-ea)**3
            dz_dt_native = -b*q/(2*r)
            dr_dt_native = energy_prime(z)-b*current_prime(z)
            diagnostics.append({k: fmt(v) for k, v in {
                "fraction": fraction, "z": z, "r_over_a": r,
                "integrand_F9": sine_integrand(z), "b": b,
                "dr_dt_native": dr_dt_native, "dr_dt_from_F8": r_prime/dt_dz,
                "dr_dt_residual": r_prime/dt_dz-dr_dt_native,
                "dz_dt_native": dz_dt_native, "dz_dt_from_F9": 1/dt_dz,
                "dz_dt_residual": 1/dt_dz-dz_dt_native,
                "energy_residual": s-b*q-ea,
                "radius_relation_residual": b-mp.sqrt(rg/r)}.items()})

        clocks = {"continuum_crossing_t": continuum_cross,
                  "continuum_crossing_proper": continuum_proper_cross,
                  "formal_point_center_t": continuum_center,
                  "formal_point_center_proper": continuum_proper_center,
                  "sine_crossing_t": sine_cross}
        out = {
            "decimal_working_precision": digits,
            "parameters": {k: fmt(v) for k, v in
                           {"rg_over_a": rg, "r0_over_a": r0,
                            "R0_over_a": radius, "c": c, "x0": x0,
                            "e_continuum": e, "b0": b0, "mu": mu}.items()},
            "release_and_crossing": {"z0": fmt(z0), "ea": fmt(ea), "zH": fmt(zh)},
            "release_bracket": {k: (v if isinstance(v, int) else fmt(v))
                                for k, v in release_bracket.items()},
            "crossing_bracket": {k: (v if isinstance(v, int) else fmt(v))
                                 for k, v in crossing_bracket.items()},
            "clocks": {k: {"over_a": fmt(v), "over_r0": fmt(v/r0),
                           "over_rg": fmt(v/rg)} for k, v in clocks.items()},
            "root_radius": {"over_a": fmt(rg), "over_r0": fmt(rg/r0), "over_rg": "1"},
            "quadrature": {"method": "mpmath tanh-sinh",
                           "function_evaluations": quad_calls,
                           "continuum_crossing_reported_error_over_a": fmt(2*amplitude*root_error),
                           "continuum_center_reported_error_over_a": fmt(2*amplitude*center_error),
                           "continuum_center_F2_minus_F4_over_a": fmt(2*amplitude*center_quad-continuum_center),
                           "sine_crossing_reported_error_over_a": fmt(2*rg*sine_error),
                           "error_estimates_certified": False},
            "F6_F8_F9_algebraic_diagnostics": diagnostics,
            "same_case_comparisons": {"sine_minus_continuum_t_over_r0": fmt((sine_cross-continuum_cross)/r0),
                                      "continuum_t_over_r0_minus_13_22": fmt(continuum_cross/r0-mp.mpf("13.22")),
                                      "sine_t_over_r0_minus_13_17": fmt(sine_cross/r0-mp.mpf("13.17")),
                                      "formal_center_t_over_r0_minus_13_42": fmt(continuum_center/r0-mp.mpf("13.42")),
                                      "root_over_r0_minus_0_93": fmt(rg/r0-mp.mpf("0.93"))},
            "x0_equals_4_closed_form_constants_only": {
                "formal_point_center_t_over_rg": fmt(3*mp.pi*mp.sqrt(3)-4+mp.log(3)),
                "formal_point_center_proper_over_rg": fmt(4*mp.pi),
                "new_parameter_run": False},
        }
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--marginal-json", type=Path, required=True)
    args = parser.parse_args()
    if Path(sys.executable).resolve() != BOX_PYTHON.resolve():
        parser.error("scientific execution requires the box's qi2 interpreter")
    out_dir = args.output_dir.resolve()
    if not out_dir.is_relative_to(BULK_ROOT.resolve()):
        parser.error("output must remain under <internal>/solve_b_bulk")
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "scalar.json"
    if result_path.exists():
        parser.error("existing scalar result is not overwritten or recomputed")
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, affinity[:1])
    for name in THREAD_ENV:
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.setpriority(os.PRIO_PROCESS, 0, 10)
    try:
        started = float(os.environ["QI2_X4_OUTER_STARTED_MONOTONIC"])
    except (KeyError, ValueError):
        parser.error("the reviewed outer launcher must supply its monotonic start")
    if not math.isfinite(started) or not 0 <= time.monotonic()-started < SCALAR_CAP_SECONDS:
        parser.error("outer scalar deadline is absent or expired")
    usage0 = resource.getrusage(resource.RUSAGE_SELF)
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, SCALAR_CAP_SECONDS-(time.monotonic()-started))
    result = {
        "schema": "WEB40I_X4_scalar_v1", "status": "started",
        "case_kind": "supplied_v36d_metadata_point_exterior",
        "purpose": "Y3 point/sine comparator plus Y1 bound-rest ray map; production-profile X4 remains incomplete",
        "specification": {"path": SPEC_PATH, "sha256": SPEC_SHA256,
                          "equations": ["F2", "F4", "F7", "F8", "F9"],
                          "contract": "section 8 scalar/preflight stage only"},
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "production_lineage_verified": False,
        "root_formation_computed": False,
        "profile_stage": {"status": "input_incomplete", "executed": False,
                          "reason": "No source-bound canonical production T0 count/current export and lineage were supplied to this scalar-only producer; no profile was manufactured.",
                          "global_absence_claim": False},
        "full_X4_task_complete": False,
        "extended_bound_ball_ray_map_implemented": True,
        "numerical_error_certificate": False,
        "rounded_FW7_baseline_reused_not_recomputed": {
            "rg_over_a": "59.05", "crossing_t_over_r0": "13.2248621796281",
            "crossing_proper_over_r0": "12.4186017984530",
            "source": SPEC_PATH + ", F2--F3 prior FW7 readback"},
        "runtime": {"python": sys.version, "platform": platform.platform(),
                    "affinity": sorted(os.sched_getaffinity(0)),
                    "nice": os.getpriority(os.PRIO_PROCESS, 0),
                    "thread_environment": {k: os.environ[k] for k in THREAD_ENV},
                    "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
                    "rlimit_as_bytes": list(resource.getrlimit(resource.RLIMIT_AS)),
                    "scalar_cap_seconds": SCALAR_CAP_SECONDS,
                    "hard_outer_cap_seconds": OUTER_CAP_SECONDS,
                    "whole_recipe_cap_seconds": WHOLE_RECIPE_CAP_SECONDS,
                    "automatic_retry_or_continuation": False},
        "metadata_preflight": [], "precision_results": [], "bound_precision_results": [],
    }
    exit_status = 0
    try:
        atomic_save(result_path, result)
        selected = None
        for width, name, expected in METADATA:
            entry, case = inspect_metadata(width, name, expected)
            result["metadata_preflight"].append(entry)
            if width == 4:
                selected = case
        result["case"] = selected
        result["point_scalar_status"] = "ready" if selected is not None else "metadata_input_incomplete"
        module_path = Path(__file__).with_name("WEB40I_bound_ball_scalar_v1.py")
        module_digest = sha256(module_path)
        if module_digest != os.environ.get("QI2_X4_BOUND_MODULE_SHA256"):
            raise ValueError("bound module differs from the reviewed identity")
        import WEB40I_bound_ball_scalar_v1 as bound_ball
        if Path(bound_ball.__file__).resolve() != module_path.resolve():
            raise ValueError("unexpected bound-module import location")
        result["bound_module"] = {"path": str(module_path.resolve()), "sha256": module_digest,
                                  "specification_path": bound_ball.SPEC_PATH,
                                  "specification_sha256": bound_ball.SPEC_SHA256}
        result["inherited_marginal_comparison"] = bound_ball.inherited_comparison(args.marginal_json)
        atomic_save(result_path, result)
        import mpmath as mp
        result["runtime"]["mpmath_version"] = mp.__version__
        def bound_checkpoint(partial):
            result["bound_active_partial"] = partial
            atomic_save(result_path, result)
        def point_checkpoint(partial):
            result["point_active_partial"] = partial
            atomic_save(result_path, result)
        for digits in PRECISIONS:
            if selected is not None:
                result["precision_results"].append(evaluate_precision(mp, digits, selected, point_checkpoint))
                result.pop("point_active_partial", None)
                atomic_save(result_path, result)
            result["bound_precision_results"].append(bound_ball.evaluate_precision(mp, digits, bound_checkpoint))
            result.pop("bound_active_partial", None)
            atomic_save(result_path, result)
        if selected is not None:
            low, high = result["precision_results"]
            with mp.workdps(80):
                result["precision_comparison"] = {
                    key: {unit: mp.nstr(abs(mp.mpf(high["clocks"][key][unit])
                                               - mp.mpf(low["clocks"][key][unit])), 80)
                          for unit in ("over_a", "over_r0", "over_rg")}
                    for key in high["clocks"]}
                result["release_crossing_precision_comparison"] = {
                    key: mp.nstr(abs(mp.mpf(high["release_and_crossing"][key])
                                    - mp.mpf(low["release_and_crossing"][key])), 80)
                    for key in high["release_and_crossing"]}
            result["point_scalar_status"] = "complete"
        result["bound_precision_comparison"] = bound_ball.compare_precisions(mp, *result["bound_precision_results"])
        if not result["bound_precision_comparison"]["passed"]:
            raise ArithmeticError("bound cross-precision consistency threshold failed")
        result["status"] = ("scalar_complete_profile_input_incomplete" if selected is not None
                            else "bound_scalar_complete_point_metadata_input_incomplete_profile_input_incomplete")
    except BaseException as error:
        result.update(status="scalar_partial_or_failed_profile_input_incomplete",
                      error_type=type(error).__name__, error=str(error))
        exit_status = 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        elapsed = time.monotonic()-started
        result["runtime"].update(main_stage_wall_seconds_before_final_write=elapsed,
                                 user_cpu_seconds=usage.ru_utime-usage0.ru_utime,
                                 system_cpu_seconds=usage.ru_stime-usage0.ru_stime,
                                 peak_rss_kib=usage.ru_maxrss,
                                 outer_execution_receipt_required=True,
                                 aggregate_budget_debit_source="execution.json full 30-second scalar allowance; producer startup/finalization observed separately")
        atomic_save(result_path, result)
    print(json.dumps({"status": result["status"], "result": str(result_path),
                      "full_X4_task_complete": False}))
    return exit_status


if __name__ == "__main__":
    sys.exit(main())
