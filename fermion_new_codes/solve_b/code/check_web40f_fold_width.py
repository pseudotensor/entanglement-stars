#!/usr/bin/env python3
"""One box-only evaluation of the reviewed WEB40F I2 scalar formulas."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))

    def expired(_signum, _frame):
        raise TimeoutError("The fixed scalar check exceeded 60 seconds")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(60)
    result = {
        "control": "WEB40F I2: reviewed dust clocks and Gaussian comparison maxima",
        "specification": [
            "solve/strand_B_web40f_fold_width.tex FW7/FW10/FW12-FW13",
            "solve/review_B_web40f_fold_width.tex section 7, I2R17-I2R18",
        ],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "complete": False,
        "scope": [
            "Fixed scalar quadratures only; no evolution or physical-parameter sweep.",
            "Binary64/50-digit agreement and quadrature estimates are not rigorous enclosures.",
            "The exact theorem uses the defined suprema, not an adopted external decimal.",
            "The 64a nominal clock differs from the release charge radius 59.05a.",
        ],
    }
    try:
        import mpmath as mp
        import scipy
        from scipy import integrate, optimize, special

        tol = 2e-12
        root_tol = 2e-13
        cutoff = 20.0
        pref = math.sqrt(2 / math.pi)
        phi_c = float(2**(-0.75) * special.gamma(1.75) / special.gamma(2.5))

        def phi(y):
            return float(phi_c * y * special.hyp1f1(0.75, 2.5, -y*y/2))

        def phi_prime(y):
            return float(phi_c * (special.hyp1f1(0.75, 2.5, -y*y/2)
                                  - 0.3*y*y*special.hyp1f1(1.75, 3.5, -y*y/2)))

        def scaled_i1(x, y):
            z = x*y
            if z < 1:
                term = z/3
                total = term
                for k in range(40):
                    term *= (k+2)/(k+1) * z*z/((2*k+5)*(2*k+4))
                    total += term
                    if abs(term) <= 1e-17 * abs(total):
                        break
                return math.exp(-(x*x+y*y)/2)*total
            return ((z-1)*math.exp(-(x-y)**2/2)
                    + (z+1)*math.exp(-(x+y)**2/2))/(2*z*z)

        def psi(y):
            def integrand(x):
                return (pref*x**1.5*math.sqrt(special.gammainc(1.5, x*x/2))
                        * scaled_i1(x, y))
            return integrate.quad(integrand, 0, cutoff, points=[1/y, y],
                                  epsabs=tol, epsrel=tol, limit=160)

        def tail(y):
            return pref*math.sqrt(cutoff)/(2*y*(cutoff-y))*math.exp(-(cutoff-y)**2/2)

        radius0, radius_g = 256.0, 59.05
        c = radius_g/radius0
        e = math.sqrt(1-c)
        phase_h = math.acos(math.sqrt(c))
        scale = math.sqrt(radius0**3/radius_g)
        phase, phase_err = integrate.quad(
            lambda t: math.cos(t)**2*(1+e*math.sin(t))/(e+math.sin(t)),
            0, phase_h, epsabs=tol, epsrel=tol)
        proper = scale*(phase_h+math.sqrt(c*(1-c)))
        chart = 2*scale*phase
        phi_y = optimize.brentq(phi_prime, 2, 4, xtol=root_tol)
        grid = []
        for j in range(501):
            y = 1.5+j/200
            value, error = psi(y)
            grid.append([y, value, error, tail(y)])
        maximum_j = max(range(len(grid)), key=lambda j: grid[j][1])
        if not 0 < maximum_j < 500:
            raise ValueError("The maximizing grid cell is not interior")
        lo, hi = grid[maximum_j-1][0], grid[maximum_j+1][0]
        peak = optimize.minimize_scalar(lambda y: -psi(y)[0], bounds=(lo, hi),
                                       method="bounded", options={"xatol": 2e-11})
        if not peak.success:
            raise RuntimeError("Maximizing-cell refinement failed")
        peak_y, peak_value = float(peak.x), float(-peak.fun)
        derivative_bound = math.sqrt(2)*(pref/3)**(1/6)
        interpolation = derivative_bound/(8*200**2)
        interior_upper = max(v+err+tr for _, v, err, tr in grid)+interpolation
        exterior = [phi(1.5), phi(4.0)]
        upper_estimate = max(interior_upper, *exterior)**2
        low = {
            "a": 1.0, "R0_over_a": radius0, "rg_over_a": radius_g,
            "nominal_radius_over_a": 64.0,
            "proper_time_over_a": proper, "chart_time_over_a": chart,
            "proper_time_over_nominal64": proper/64,
            "chart_time_over_nominal64": chart/64,
            "proper_time_over_rg": proper/radius_g, "chart_time_over_rg": chart/radius_g,
            "phase_quad_error_estimate": phase_err,
            "chart_quad_error_estimate": 2*scale*phase_err,
            "gamma_release": 1/e, "gamma_horizon": (1+e*e)/(2*e),
            "phi_derivative_bracket": [phi_prime(2), phi_prime(4)],
            "phi_maximizer": phi_y, "c1": phi(phi_y)**2,
            "phi_exterior_values": exterior,
            "grid_columns": ["y", "psi", "quad_error_estimate", "analytic_tail_allowance"],
            "grid": grid, "maximizing_grid_index": maximum_j,
            "psi_refined_y": peak_y, "psi_refined_value": peak_value,
            "c2_observed": peak_value**2,
            "psi_second_derivative_bound": derivative_bound,
            "amplitude_interpolation_allowance": interpolation,
            "c2_global_upper_estimate": upper_estimate,
            "global_estimate_below_0_2282": upper_estimate < 0.2282,
            "global_estimate_is_rigorous_enclosure": False,
        }
        result["binary64"] = low

        mp.mp.dps = 50
        m = mp.mpf
        c_mp = m("59.05")/256
        e_mp = mp.sqrt(1-c_mp)
        angle_mp = mp.acos(mp.sqrt(c_mp))
        scale_mp = mp.sqrt(m(256)**3/m("59.05"))
        proper_mp = scale_mp*(angle_mp+mp.sqrt(c_mp*(1-c_mp)))
        chart_mp = 2*scale_mp*mp.quad(
            lambda t: mp.cos(t)**2*(1+e_mp*mp.sin(t))/(e_mp+mp.sin(t)), [0, angle_mp])
        coefficient = mp.power(2, -m(3)/4)/mp.gamma(m(3)/4)

        def phi_mp(y, derivative=False):
            return coefficient*mp.quad(
                lambda s: 4*s*s*mp.power(1-s**4, m(3)/4)*mp.exp(-y*y*s**4/2)
                * (1-y*y*s**4 if derivative else y), [0, 1])

        phi_y_mp = mp.findroot(lambda y: phi_mp(y, True), (2, 4))

        def psi_mp(y):
            def fun(x):
                z = x*y
                p3 = mp.gammainc(m(3)/2, 0, x*x/2, regularized=True)
                if z < 1:
                    term = z/3
                    i1 = term
                    for k in range(80):
                        term *= m(k+2)/m(k+1)*z*z/((2*k+5)*(2*k+4))
                        i1 += term
                        if abs(term) <= mp.eps*abs(i1)/100:
                            break
                    kernel = mp.exp(-(x*x+y*y)/2)*i1
                else:
                    kernel = ((z-1)*mp.exp(-(x-y)**2/2)+(z+1)*mp.exp(-(x+y)**2/2))/(2*z*z)
                return mp.sqrt(2/mp.pi)*x**m("1.5")*mp.sqrt(p3)*kernel
            breaks = sorted({m(0), 1/y, m(1), y, m(5), m(10), m(20)})
            return mp.quad(fun, breaks)

        repeats = []
        for y in [grid[maximum_j-1][0], grid[maximum_j][0], grid[maximum_j+1][0], peak_y]:
            high = psi_mp(m(y))
            value, err = psi(y)
            repeats.append({"y_binary64_exact": mp.nstr(m(y), 52),
                            "psi_50digit": mp.nstr(high, 52),
                            "binary64_minus_50digit": mp.nstr(m(value)-high, 52),
                            "binary64_quad_error_estimate": err})
        result.update({
            "complete": True, "binary64": low,
            "mpmath_50digit": {
                "proper_time_over_a": mp.nstr(proper_mp, 52),
                "chart_time_over_a": mp.nstr(chart_mp, 52),
                "proper_binary64_minus_50digit": mp.nstr(m(proper)-proper_mp, 52),
                "chart_binary64_minus_50digit": mp.nstr(m(chart)-chart_mp, 52),
                "phi_maximizer": mp.nstr(phi_y_mp, 52),
                "c1": mp.nstr(phi_mp(phi_y_mp)**2, 52),
                "phi_maximizer_binary64_minus_50digit": mp.nstr(m(phi_y)-phi_y_mp, 52),
                "c1_binary64_minus_50digit": mp.nstr(m(low["c1"])-phi_mp(phi_y_mp)**2, 52),
                "phi_exterior_values": [mp.nstr(phi_mp(m(y)), 52) for y in [1.5, 4]],
                "phi_exterior_binary64_minus_50digit": [
                    mp.nstr(m(v)-phi_mp(m(y)), 52) for y, v in zip([1.5, 4], exterior)],
                "psi_fixed_repeats": repeats,
            },
            "settings": {"quad_epsabs": tol, "quad_epsrel": tol, "phi_root_xtol": root_tol,
                         "x_cut": cutoff, "y_min": 1.5, "y_max": 4, "grid_count": 501,
                         "grid_spacing": 0.005, "mp_dps": 50},
            "checks": {"phi_root_bracket_signs": phi_prime(2) > 0 > phi_prime(4),
                       "phi_root_inside_bracket": 2 < phi_y < 4,
                       "mp_phi_root_inside_bracket": bool(2 < phi_y_mp < 4),
                       "finite_scalar_values": all(math.isfinite(v) for v in
                           [proper, chart, phi_y, peak_y, peak_value, upper_estimate]),
                       "chart_exceeds_proper": chart > proper},
            "versions": {"scipy": scipy.__version__, "mpmath": mp.__version__},
        })
    except Exception as exc:
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        signal.alarm(0)
        result["execution"] = {
            "host": socket.gethostname(), "python": sys.version.split()[0],
            "pid": os.getpid(), "nice": os.getpriority(os.PRIO_PROCESS, 0),
            "threads": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]},
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "wall_seconds": time.perf_counter()-started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "wall_limit_seconds": 60, "cpu_limit_seconds": 60,
            "address_space_limit_bytes": 1024**3,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"complete": result["complete"], "output": str(args.output),
                      "failure": result.get("failure"), "execution": result["execution"]}))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
