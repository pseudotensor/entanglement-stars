"""Owner inbox-3 fixed scalar readback, specified by the original reviewer B pair.

Execute on compute-host with qi2 Python, nice 10 and one CPU thread.
No state evolution, geometry solve, integral, fit or certified remainder.
"""

import json
import os
import resource
import sys
import time


resource.setrlimit(resource.RLIMIT_CPU, (25, 30))
resource.setrlimit(resource.RLIMIT_AS, (128 * 1024**2, 128 * 1024**2))
started = time.perf_counter()

import mpmath as mp


def dec(value):
    return mp.mpf(str(value))


def saved(value):
    return mp.nstr(value, mp.mp.dps)


def calculate(digits):
    mp.mp.dps = digits
    tolerance = dec("1e-40")
    g, C = dec("0.25"), dec("0.005237315422")
    b_control, N_control = dec("1.639111017595"), dec("0.015541819792")
    c, h = dec("299792458"), dec("6.62607015e-34")
    hbar, k_B = h / (2 * mp.pi), dec("1.380649e-23")
    G, eV_J = dec("6.67430e-11"), dec("1.602176634e-19")
    GM_nominal, year_s = dec("1.3271244e20"), dec("31557600")
    lp = mp.sqrt(hbar * G / c**3)
    identities = {}

    def identity(name, left, right):
        scale = max(abs(left), abs(right))
        residual = abs(left - right) / scale if scale else mp.mpf(0)
        identities[name] = saved(residual)

    def red_crossovers(chi):
        rows = []
        for label, x in (("x=1/(4pi)", 1 / (4 * mp.pi)), ("x=1", dec(1))):
            rows.append({
                "label": label,
                "x_omega_rs": saved(x),
                "y_cross_equals_chi_x_to_2_over_3": saved((chi * x) ** (mp.mpf(2) / 3)),
                "alpha_cross_equals_chi_x_to_1_over_3": saved((chi * x) ** (mp.mpf(1) / 3)),
                "z_H_equals_2_chi_x_to_1_over_3": saved((2 * chi * x) ** (mp.mpf(1) / 3)),
            })
        return rows

    def scale_case(chi, label):
        n = 1 / chi
        tE = n**2 / (2 * g * C)
        tev = n**2 / (6 * g * C)
        epsilon = 2 * g * C * chi**2
        delta = (2 * chi) ** (mp.mpf(2) / 3)
        alpha = (2 * chi) ** (mp.mpf(1) / 3)
        wavelength = (chi**2 / 2) ** (mp.mpf(1) / 3)
        T_log = -mp.log(delta)
        exponent = (4 * mp.pi / 5) / chi
        identity(label + ":delta_d_cubed", delta**3, 4 * chi**2)
        identity(label + ":alpha_d_squared", alpha**2, delta)
        identity(label + ":lambda_h_cubed", wavelength**3, chi**2 / 2)
        identity(label + ":tE_equals_3tev", tE, 3 * tev)
        identity(label + ":epsilon_times_tE", epsilon * tE, mp.mpf(1))
        identity(label + ":kappa_adiabaticity", 4 * g * C * chi**2, 2 * epsilon)
        identity(label + ":a_TH_nat_units", (1 / n) * n / (4 * mp.pi * n), chi / (4 * mp.pi))
        return {
            "label": label,
            "rs_over_a": saved(n),
            "g": saved(g),
            "chi": saved(chi),
            "ma_at_m_equals_T_H": saved(chi / (4 * mp.pi)),
            "axial_q1_massless_formal_temperature_term": saved(mp.mpf(37) / 32 * chi**2),
            "m_over_T_H_at_ma_0_2": saved(exponent),
            "energy_timescale_E_over_L_over_rs": saved(tE),
            "conditional_lifetime_over_rs": saved(tev),
            "conditional_lifetime_over_40rs": saved(tev / 40),
            "conditional_time_to_E0_over_e_over_rs": saved(tev * (1 - mp.exp(-3))),
            "epsilon_E": saved(epsilon),
            "epsilon_kappa": saved(2 * epsilon),
            "author_q1_defined_comparison": {
                "delta_d": saved(delta),
                "alpha_d": saved(alpha),
                "lambda_h_over_rs": saved(wavelength),
                "DeltaR_d_over_rs": saved(delta / (1 - delta)),
                "T_d_log_over_rs": saved(T_log),
                "u_d_log_over_rs": saved(2 * T_log),
                "held_redshift_comparison_not_maximum": saved(1 / alpha),
                "infall_redshift_comparison_not_maximum": saved(2 / delta),
            },
            "red_defined_crossover_convention": red_crossovers(chi),
        }

    runs = []
    raw_lifetimes = {}
    for n_int in (32, 64):
        chi = 1 / dec(n_int)
        row = scale_case(chi, "run_" + str(n_int))
        row["exp_minus_m_over_T_H_at_ma_0_2"] = saved(mp.exp(-(4 * mp.pi / 5) / chi))
        raw_lifetimes[n_int] = dec(n_int)**2 / (6 * g * C)
        runs.append(row)
    identity("run64_to_run32_lifetime_ratio", raw_lifetimes[64] / raw_lifetimes[32], dec(4))

    rs = 2 * GM_nominal / c**2
    a = lp / mp.sqrt(g)
    chi = a / rs
    TH_K = hbar * c / (4 * mp.pi * k_B * rs)
    L_W = C * hbar * c**2 / rs**2
    tE_s = rs**3 / (2 * C * lp**2 * c)
    tev_s = rs**3 / (6 * C * lp**2 * c)
    E_J = (GM_nominal / G) * c**2
    solar = scale_case(chi, "nominal_solar_g_0_25")
    q1 = solar["author_q1_defined_comparison"]
    solar.update({
        "rs_m": saved(rs),
        "rs_light_crossing_s": saved(rs / c),
        "a_m": saved(a),
        "T_H_K": saved(TH_K),
        "k_B_T_H_eV": saved(k_B * TH_K / eV_J),
        "E_quantum_q1_J": saved(hbar * c / rs),
        "DeltaR_d_m": saved(rs * dec(q1["DeltaR_d_over_rs"])),
        "lambda_h_m": saved(rs * dec(q1["lambda_h_over_rs"])),
        "rho_d_leading_m": saved(2 * rs * dec(q1["alpha_d"])),
        "T_d_log_seconds": saved((rs / c) * dec(q1["T_d_log_over_rs"])),
        "u_d_log_seconds": saved((rs / c) * dec(q1["u_d_log_over_rs"])),
        "conditional_luminosity_W": saved(L_W),
        "energy_timescale_E_over_L_s": saved(tE_s),
        "conditional_lifetime_s": saved(tev_s),
        "conditional_lifetime_Julian_years": saved(tev_s / year_s),
    })
    identity("solar_L_times_tE_equals_E", L_W * tE_s, E_J)
    identity("solar_tE_equals_3tev", tE_s, 3 * tev_s)
    identity("solar_SI_adiabaticity", (rs / c) / (E_J / L_W), 2 * C * lp**2 / rs**2)
    identity("solar_g_chi_conversion", 2 * g * C * chi**2, 2 * C * lp**2 / rs**2)
    identity("solar_natural_to_SI_tE", tE_s, (rs / c) / (2 * g * C * chi**2))

    averaging = []
    for ell_int in (4, 8, 16):
        ell = dec(ell_int)
        averaging.append({
            "rs_over_a": 64,
            "ell_over_a": ell_int,
            "formal_gaussian_relative_shift": saved(-mp.mpf(9) / 4 * (ell / 64) ** 2),
            "formal_compact_cell_relative_shift": saved(-mp.mpf(9) / 4 * (ell**2 / 15 + mp.mpf(1) / 12) / 64**2),
        })

    capacity_crossings = []
    for label, b in (("finite_domain_b_input", b_control), ("b=1_capacity_example", dec(1))):
        fraction_squared = b / (1 + b)
        capacity_crossings.append({
            "label": label,
            "b": saved(b),
            "conditional_crossing_mass_over_initial_mass": saved(mp.sqrt(fraction_squared)),
            "conditional_crossing_time_over_lifetime": saved(1 - fraction_squared ** (mp.mpf(3) / 2)),
        })

    return {
        "conditional_control_inputs": {
            "C_L": saved(C), "b": saved(b_control), "C_N": saved(N_control),
            "qualification": "Existing finite-domain continuum readbacks; omitted tails are not certified here",
        },
        "constants_SI": {
            "c_m_per_s": saved(c), "h_J_s": saved(h), "hbar_J_s": saved(hbar),
            "k_B_J_per_K": saved(k_B), "G_m3_per_kg_s2": saved(G),
            "eV_J": saved(eV_J), "GM_sun_nominal_m3_per_s2": saved(GM_nominal),
            "lp_m_computed_consistently": saved(lp),
        },
        "run_cases": runs,
        "nominal_solar_g_0_25": solar,
        "common_40rs_conditional_mean_number": saved(40 * N_control),
        "formal_averaging_conventions": averaging,
        "conditional_capacity_crossings_not_a_Page_measurement": capacity_crossings,
        "identity_relative_residuals": identities,
        "all_identity_checks_passed": all(dec(r) <= tolerance for r in identities.values()),
    }


low = calculate(50)
high = calculate(80)
mp.mp.dps = 80


def numeric_leaves(value, path=""):
    if isinstance(value, bool):
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "identity_relative_residuals":
                continue
            yield from numeric_leaves(child, path + "/" + str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from numeric_leaves(child, path + "/" + str(index))
    else:
        try:
            yield path, dec(value)
        except (ValueError, TypeError):
            pass


low_values, high_values = dict(numeric_leaves(low)), dict(numeric_leaves(high))
assert low_values.keys() == high_values.keys()
differences = {}
for path, value in high_values.items():
    scale = max(abs(value), abs(low_values[path]))
    differences[path] = abs(value - low_values[path]) / scale if scale else mp.mpf(0)
worst_path = max(differences, key=differences.get)
failures = [path for path, residual in differences.items() if residual > dec("1e-40")]

result = {
    "purpose": "Owner inbox 3; paper item 6 surface-lapse expansion and run-to-solar arithmetic",
    "scope": "Conditional scalar formulas only; no physical history, floor, measured lifetime or Page curve",
    "precision": "80-digit outputs with a 50/80-digit arithmetic comparison; not physical input accuracy or interval enclosures",
    "constant_sources": {
        "CODATA_2022": "https://physics.nist.gov/cuu/pdf/wall_2022.pdf",
        "CODATA_2022_full": "https://physics.nist.gov/cuu/pdf/JPCRD2022CODATA.pdf",
        "IAU_2015_B3": "https://www.iau.org/common/Uploaded%20files/IAUGA2015-Resolution-B3-recommended-nominal-conversion.pdf",
        "solar_convention": "Nominal GM conversion factor, not the measured instantaneous solar mass",
        "G_standard_uncertainty_SI": "1.5e-15",
        "Julian_year_seconds_convention": "31557600",
    },
    **high,
    "precision_comparison": {
        "working_digits": [50, 80], "relative_tolerance": "1e-40",
        "compared_field_count": len(differences), "worst_field": worst_path,
        "max_relative_difference": saved(differences[worst_path]),
        "failures": failures,
        "all_identity_checks_passed_at_50_digits": low["all_identity_checks_passed"],
    },
    "all_checks_passed": not failures and low["all_identity_checks_passed"] and high["all_identity_checks_passed"],
    "execution": {
        "python": sys.executable, "mpmath": mp.__version__,
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_limit_seconds": 30, "address_space_limit_MiB": 128,
        "elapsed_seconds": time.perf_counter() - started,
        "max_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    },
}
print(json.dumps(result, indent=2, allow_nan=False))
if not result["all_checks_passed"]:
    raise SystemExit(1)
