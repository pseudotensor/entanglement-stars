#!/usr/bin/env python3
"""Evaluate only the fixed R3A11 scalar table; run on the lane's box CPU."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import sys
import time


L_HAT = "0.005237315421862865"
N_HAT = "0.015541819792029855"
B_HAT = "1.6391110175949173"


def evaluate(num, sqrt, log, log1p, tan, cos, pi):
    """Translate the author-specified displayed formulas, with no equation solve."""
    one = num(1)
    two = num(2)
    three = num(3)
    g = num(1) / num(4)
    x = one / (num(4) * pi)
    d = num("0.1")
    epsilon0 = num("0.1")
    lhat, nhat, bhat = map(num, (L_HAT, N_HAT, B_HAT))
    z0 = pi / num(4)
    splus = one / sqrt(two)
    cin_over_rs = two * (log(two * tan(z0 / two)) + cos(z0) - one)
    cout_over_rs = two * (
        log1p(-splus)
        + one / (two * splus**2)
        + one / splus
        - log(splus)
        - num(3) / two
    )

    def primitive(e):
        return (
            two / three * (one - e) ** (-three / two)
            + two * (one - e) ** (-one / two)
            + log(e)
            - two * log1p(sqrt(one - e))
        )

    def tortoise(e):
        return one / (one - e) + log(e) - log1p(-e)

    inputs = (
        ("run_rs32", one / num(32)),
        ("run_rs64", one / num(64)),
        ("solar_rounded", num("5.47e-39") / sqrt(g)),
    )
    rows = []
    for label, chi in inputs:
        nu = chi * x
        ud = (nu**2 / (two * d)) ** (one / three)
        epsilon = two * ud - ud**2
        alpha = sqrt(epsilon)
        s = sqrt(one - epsilon)
        cone_b = num(27) / num(4) * epsilon * (one - epsilon) ** 2
        cone_c = sqrt(one - cone_b)
        one_minus_c = cone_b / (one + cone_c)
        held_fraction = one_minus_c / two
        infall_fraction = one_minus_c * ud / (two * (one + s * cone_c))
        gap = epsilon / (one - epsilon)
        clock_log = log(epsilon0 / epsilon)
        delta_t = primitive(epsilon0) - primitive(epsilon)
        clock_bound = (
            num(5) / two
            * (one - epsilon0) ** (-num(7) / two)
            * (epsilon0 - epsilon)
        )
        delta_retarded = delta_t - (tortoise(epsilon) - tortoise(epsilon0))
        asymptotic_clock = (
            two / three * log(one / nu)
            + log(epsilon0)
            + one / three * log(d)
            - two / three * log(two)
        )
        branch_b = (two * nu) ** (one / three)
        wavelength_leading = two ** (-one / three) * chi ** (two / three) * x ** (-one / three)
        wavelength_corrected = wavelength_leading * (one - branch_b**2 / num(12))
        speed_leading = num(3) / two * branch_b**2
        speed_corrected = speed_leading * (one - branch_b**2 / num(4))
        flight_h = two * log(one / nu) + cin_over_rs + cout_over_rs
        ha_shift = num(207) / num(80) * chi**2
        axial_temperature = chi**2 * (num(5) / num(32) + x**2)
        initial_rs_rate = two * g * lhat * chi**2
        energy_time = one / initial_rs_rate
        lifetime = energy_time / three
        page_radius = sqrt(bhat / (bhat + one))
        page_time_fraction = one - (bhat / (bhat + one)) ** (three / two)
        values = {
            "chi": chi,
            "nu": nu,
            "u_d": ud,
            "epsilon_d": epsilon,
            "alpha_d": alpha,
            "continuum_momentum_at_endpoint": nu / ud,
            "held_frequency_ratio": alpha,
            "infall_radial_frequency_ratio": ud,
            "held_redshift_factor": one / alpha,
            "infall_radial_redshift_factor": one / ud,
            "held_escape_number_fraction": held_fraction,
            "infall_escape_number_fraction": infall_fraction,
            "fixed_sphere_outward_number_fraction": ud / two,
            "held_leading_power_ratio": num(27) / num(16) * epsilon**2,
            "infall_leading_power_ratio": num(27) / num(256) * epsilon**4,
            "coordinate_gap_over_rs": gap,
            "coordinate_gap_over_a": gap / chi,
            "surface_clock_efolds": clock_log / two,
            "clock_log_epsilon_ratio": clock_log,
            "delta_T_over_rs": delta_t,
            "delta_T_minus_log_epsilon_ratio": delta_t - clock_log,
            "R3A4_clock_upper_bound": clock_bound,
            "R3A16_displayed_leading_clock": asymptotic_clock,
            "delta_T_minus_R3A16_displayed_leading_clock": delta_t - asymptotic_clock,
            "delta_retarded_over_rs": delta_retarded,
            "delta_retarded_minus_twice_log_epsilon_ratio": delta_retarded - two * clock_log,
            "surface_reduced_wavelength_leading_over_rs": wavelength_leading,
            "surface_reduced_wavelength_first_corrected_over_rs": wavelength_corrected,
            "surface_group_speed_leading": speed_leading,
            "surface_group_speed_first_corrected": speed_corrected,
            "C_in_over_rs": cin_over_rs,
            "C_out_over_rs": cout_over_rs,
            "symbol_flight_H_nu": flight_h,
            "compact_HA_leading_shift_ell4": ha_shift,
            "compact_HA_shift_over_u_d": ha_shift / ud,
            "formal_axial_temperature_shift": axial_temperature,
            "compact_HA_shift_over_formal_axial_temperature_shift": ha_shift / axial_temperature,
            "conditional_energy_time_over_rs": energy_time,
            "conditional_zero_radius_lifetime_over_rs": lifetime,
            "conditional_kappa_adiabaticity": num(4) * g * lhat * chi**2,
            "conditional_rs_over_energy_time": initial_rs_rate,
            "conditional_40rs_over_energy_time": num(40) / energy_time,
            "initial_rate_freezing_indicator": initial_rs_rate * flight_h / ud,
            "conditional_Page_crossing_radius_fraction_c1": page_radius,
            "conditional_Page_crossing_lifetime_fraction_c1": page_time_fraction,
            "conditional_Page_crossing_time_over_rs_c1": page_time_fraction * lifetime,
            "separate_am_point2_massive_suppression_exponent": num(4) * pi * num("0.2") / chi,
        }
        if label == "solar_rounded":
            rs_m = num(2950)
            rs_seconds = rs_m / num(299792458)
            year = num(31557600)
            values.update({
                "rounded_solar_rs_m": rs_m,
                "rounded_solar_rs_seconds": rs_seconds,
                "rounded_solar_coordinate_gap_m": rs_m * gap,
                "rounded_solar_approach_seconds": rs_seconds * delta_t,
                "rounded_solar_retarded_seconds": rs_seconds * delta_retarded,
                "rounded_solar_symbol_flight_comparison_seconds": rs_seconds * flight_h,
                "conditional_rounded_solar_energy_time_seconds": rs_seconds * energy_time,
                "conditional_rounded_solar_energy_time_Julian_years": rs_seconds * energy_time / year,
                "conditional_rounded_solar_lifetime_seconds": rs_seconds * lifetime,
                "conditional_rounded_solar_lifetime_Julian_years": rs_seconds * lifetime / year,
                "conditional_rounded_solar_Page_crossing_seconds_c1": rs_seconds * page_time_fraction * lifetime,
                "conditional_rounded_solar_Page_crossing_Julian_years_c1": rs_seconds * page_time_fraction * lifetime / year,
            })
        checks = {
            "u_d_lt_one_quarter": bool(ud < one / num(4)),
            "nu_over_u_d_lt_one_half": bool(nu / ud < one / two),
            "epsilon_d_lt_epsilon0": bool(epsilon < epsilon0),
            "escape_radius_inside_photon_sphere": bool(epsilon < one / num(3)),
            "clock_excess_nonnegative": bool(delta_t - clock_log >= 0),
            "clock_excess_le_displayed_bound": bool(delta_t - clock_log <= clock_bound),
        }
        rows.append({"label": label, "values": values, "domain_checks": checks})
    return {
        "g": g, "x": x, "mass": num(0), "d": d, "epsilon0": epsilon0,
        "L_hat": lhat, "N_hat": nhat, "b_hat": bhat, "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))

    def expired(_signum, _frame):
        raise TimeoutError("The fixed scalar check exceeded 60 seconds")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(60)
    result = {
        "control": "R3A11 fixed surface-lapse arithmetic, three prescribed inputs only",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "specification": "solve/strand_R_surface_lapse_asymptotics.tex paragraph R3A11",
        "complete": False,
        "scope": [
            "Formula evaluation only; no evolution, scattering, quadrature or physical root solve.",
            "Dispersion endpoint is a validity endpoint, not a physical minimum lapse.",
            "Escape fractions are the specified continuum emitter readouts.",
            "Surface wavelength and speed corrections are asymptotic evaluations.",
            "H_nu is a matched characteristic comparison without an evaluated remainder bound.",
            "Compact-kernel HA audit is not transfer of any Gaussian-engine history.",
            "Flux hats are existing uncertified finite-domain control inputs.",
            "Lifetimes, adiabaticity and Page crossing require the stated population/charge/capacity premises.",
            "initial_rate_freezing_indicator uses the initial rate and is not a uniform whole-flight bound.",
            "Binary64 versus 50-digit agreement is observed arithmetic, not a rigorous enclosure.",
            "Solar SI values use the explicitly prescribed rounded inputs.",
        ],
    }
    try:
        import mpmath as mp
        mp.mp.dps = 50
        low = evaluate(float, math.sqrt, math.log, math.log1p, math.tan, math.cos, math.pi)
        high = evaluate(mp.mpf, mp.sqrt, mp.log, mp.log1p, mp.tan, mp.cos, mp.pi)

        def encode_mp(value):
            if isinstance(value, dict):
                return {k: encode_mp(v) for k, v in value.items()}
            if isinstance(value, list):
                return [encode_mp(v) for v in value]
            if isinstance(value, (str, bool)):
                return value
            return mp.nstr(value, 52)

        differences = []
        for low_row, high_row in zip(low["rows"], high["rows"]):
            delta = {}
            for key, high_value in high_row["values"].items():
                signed = mp.mpf(low_row["values"][key]) - high_value
                delta[key] = {
                    "binary64_minus_50digit": mp.nstr(signed, 52),
                    "absolute_difference": mp.nstr(abs(signed), 52),
                    "relative_difference": mp.nstr(abs(signed / high_value), 52) if high_value else None,
                }
            differences.append({"label": low_row["label"], "values": delta})
        result.update({
            "complete": True,
            "binary64": low,
            "mpmath_50digit": encode_mp(high),
            "differences": differences,
            "mpmath_version": mp.__version__,
        })
    except Exception as exc:
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        signal.alarm(0)
        result["execution"] = {
            "host": socket.gethostname(),
            "python": sys.version.split()[0],
            "pid": os.getpid(),
            "nice": os.getpriority(os.PRIO_PROCESS, 0),
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "wall_seconds": time.perf_counter() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "wall_limit_seconds": 60,
            "address_space_limit_bytes": 256 * 1024**2,
            "cpu_limit_seconds": 60,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": result["complete"], "output": str(args.output), "execution": result["execution"]}))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
