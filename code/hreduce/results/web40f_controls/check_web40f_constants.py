#!/usr/bin/env python3
"""One bounded scalar control for WEB40F inserts 6 and 12.

The scientific specifications are EM5/MC5 of the original mechanics author
and EMR7/MCR1--5 of the original mechanics RED.  Arb certifies Gaussian
envelopes; 50/80-digit ordinary arithmetic checks only the named scalar
identities.  No profile, history, physical-parameter sweep or native capture
calculation.  Run only on the box in qi2, nice 10, CPU1, CUDA disabled.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import flint
from flint import arb, acb, ctx
import mpmath as mp


def exact_fraction(x):
    mantissa, exponent = x.man_exp()
    return Fraction(int(mantissa)) * Fraction(2) ** int(exponent)


def real_ball(q):
    return arb(q.numerator) / arb(q.denominator)


def decimal_endpoint(x, rounding):
    q = exact_fraction(x)
    with localcontext() as context:
        context.prec = 75
        context.rounding = rounding
        return str(Decimal(q.numerator) / Decimal(q.denominator))


def ball_record(x):
    if not x.is_finite():
        return {"finite": False}
    lo, hi = x.lower(), x.upper()
    return {
        "finite": True, "ball": x.str(65),
        "lower": decimal_endpoint(lo, ROUND_FLOOR),
        "upper": decimal_endpoint(hi, ROUND_CEILING),
        "lower_dyadic": [str(lo.man_exp()[0]), int(lo.man_exp()[1])],
        "upper_dyadic": [str(hi.man_exp()[0]), int(hi.man_exp()[1])],
        "radius": x.rad().str(20),
    }


def fraction_record(q):
    return [str(q.numerator), str(q.denominator)]


def one_sided_allowance(upper):
    upper = upper.upper()
    result = arb(upper / 2, upper / 2)
    assert result.contains(0) and result.contains(upper)
    return result


def atomic_report(path, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def limit_signal(_number, _frame):
    raise TimeoutError("single bounded scalar control resource limit")


def amplitude_constant():
    return (arb(2).sqrt().sqrt() * (arb(7) / 4).gamma()
            / (3 * (arb(3) / 2).gamma()))


def phi(y, aa):
    return aa * y * (-y * y / 2).hypgeom_1f1(arb(3) / 4, arb(5) / 2)


def phi_prime(y, aa):
    zz = -y * y / 2
    return aa * (zz.hypgeom_1f1(arb(3) / 4, arb(5) / 2)
                 - arb(3) / 10 * y * y
                 * zz.hypgeom_1f1(arb(7) / 4, arb(7) / 2))


def beta_integrand(y, calls):
    yy = acb(y)

    def evaluate(z, analytic):
        calls[0] += 1
        dd = z ** 4 + (1 - z) ** 4
        # This sufficient positive-real check proves the chosen square root
        # analytic on every accepted integration ball. Otherwise subdivide.
        if not dd.is_finite() or not dd.real > 0:
            return acb("nan")
        root = dd.sqrt(analytic=analytic)
        if not root.is_finite():
            return acb("nan")
        return (4 * z ** 2 * (1 - z) ** 6
                * (-(yy ** 2 / 2) * z ** 4 / dd).exp()
                / (dd ** 2 * root))
    return evaluate


def point_envelope(report):
    aa = amplitude_constant()
    assert aa < arb(42) / 100
    left, right = Fraction(2), Fraction(3)
    derivative_left = phi_prime(real_ball(left), aa)
    derivative_right = phi_prime(real_ball(right), aa)
    assert derivative_left > 0 and derivative_right < 0
    iterations = 0
    while right - left > Fraction(1, 2 ** 120):
        middle = (left + right) / 2
        derivative = phi_prime(real_ball(middle), aa)
        if derivative > 0:
            left = middle
        elif derivative < 0:
            right = middle
        else:
            raise ArithmeticError("Phi derivative sign is unresolved")
        iterations += 1
    midpoint = real_ball((left + right) / 2)
    bracket = arb(midpoint, real_ball((right - left) / 2))
    assert bracket.contains(real_ball(left)) and bracket.contains(real_ball(right))
    c1 = phi(bracket, aa) ** 2
    assert c1.is_finite() and c1 > 0
    beta = ((arb(3) / 4).gamma() * (arb(7) / 4).gamma()
            / (arb(5) / 2).gamma())
    calls = [0]
    integral = acb.integral(
        beta_integrand(midpoint, calls), acb(0), acb(1), rel_tol=0,
        abs_tol=arb(2) ** -130, deg_limit=64, eval_limit=20000,
        depth_limit=30)
    if not integral.is_finite() or not integral.imag.contains(0):
        raise ArithmeticError("independent Phi beta integral is unresolved")
    direct = phi(midpoint, aa)
    independent = aa * midpoint * integral.real / beta
    discrepancy = direct - independent
    assert direct.overlaps(independent)
    assert abs(discrepancy) < arb("1e-30")
    report["point_envelope"] = {
        "status": "certified", "A": ball_record(aa), "A_lt_0.42": True,
        "derivative_at_2": ball_record(derivative_left),
        "derivative_at_3": ball_record(derivative_right),
        "critical_bracket": ball_record(bracket),
        "critical_bracket_exact_endpoints": [fraction_record(left), fraction_record(right)],
        "root_bisections": iterations, "maximum_C1": ball_record(c1),
        "midpoint_hypergeometric": ball_record(direct),
        "midpoint_beta_integral": ball_record(independent),
        "route_difference": ball_record(discrepancy),
        "beta_callback_evaluations": calls[0],
        "global_basis": "EM.1/EMR5 unique Phi maximum, certified signs and bracket",
    }


def psi_integrand(y, calls):
    yy = acb(y)
    sqrt_two = acb(2).sqrt()
    prefactor = (acb(2) / acb.pi()).sqrt()

    def evaluate(z, analytic):
        calls[0] += 1
        if z.contains(0):
            return acb("nan")
        pp = (z / sqrt_two).erf() - prefactor * z * (-z * z / 2).exp()
        az = pp / z ** 3
        # a(z)=sqrt(2/pi) int_0^1 t^2 exp(-z^2 t^2/2)dt.
        # A certified positive real image is sufficient for the principal
        # root to be analytic; failure returns nan so acb subdivides.
        if not az.is_finite() or not az.real > 0:
            return acb("nan")
        root = az.sqrt(analytic=analytic)
        if not root.is_finite():
            return acb("nan")
        gg = z * root
        ww = z * yy
        scaled_numerator = ((ww - 1) * (-(z - yy) ** 2 / 2).exp()
                            + (ww + 1) * (-(z + yy) ** 2 / 2).exp())
        # x^2 in the positive radial integrand cancels the denominator
        # 2*x^2*y^2 of the scaled i1 kernel. y>=1, x>=2^-12.
        return prefactor * gg * scaled_numerator / (2 * yy ** 2)
    return evaluate


def double_envelope(report, save):
    delta = Fraction(1, 2 ** 12)
    small = real_ball(delta) ** 5 / 3
    large = (-(arb(121) / 2)).exp() * (1 + arb(5) / 11) / (2 * arb.pi()).sqrt()
    assert large < arb("1e-25")
    allowance = one_sided_allowance(small + large)
    section = {
        "status": "incomplete", "node_cap": 512,
        "quadrature_interval": [fraction_record(delta), ["16", "1"]],
        "small_positive_cap": ball_record(small),
        "large_positive_cap": ball_record(large),
        "combined_one_sided_allowance": ball_record(allowance),
        "curvature_absolute_bound": ball_record(arb(2).sqrt()),
        "node_width_cap": "1e-10", "squared_global_width_cap": "2e-6",
        "global_basis": "Psi<=Phi<=min(A*y,y^-1/2); A<.42, Psi(3)>.45; |Psi''|<=sqrt(2)",
        "nodes": [],
    }
    report["double_envelope"] = section
    nodes = {}

    def evaluate_node(q):
        if q in nodes:
            return
        if len(nodes) >= 512:
            raise ArithmeticError("512 distinct Psi nodes reached before enclosure target")
        attempts = []
        began = time.monotonic()
        for bits in (192, 256):
            ctx.prec = bits
            calls = [0]
            value = acb.integral(
                psi_integrand(real_ball(q), calls), acb(real_ball(delta)), acb(16),
                rel_tol=0, abs_tol=arb(2) ** -45, deg_limit=64,
                eval_limit=20000, depth_limit=30)
            attempts.append({"bits": bits, "callback_evaluations": calls[0],
                             "integral": ball_record(value.real),
                             "imaginary_contains_zero": bool(value.imag.contains(0))})
            if (value.is_finite() and value.imag.contains(0)
                    and value.real.upper() - value.real.lower() <= arb("1e-10")):
                break
        else:
            section["failed_node"] = {"y": fraction_record(q), "attempts": attempts}
            save()
            raise ArithmeticError("Psi node lacks the prescribed real integral enclosure")
        result = value.real + allowance
        nodes[q] = result
        section["nodes"].append({
            "y": fraction_record(q), "attempts": attempts,
            "with_positive_caps": ball_record(result),
            "elapsed_seconds": time.monotonic() - began})
        save()

    for j in range(17):
        evaluate_node(Fraction(1) + Fraction(j, 4))
    assert nodes[Fraction(3)] > arb(45) / 100
    assert 1 / arb(5).sqrt() < arb(45) / 100
    section["Psi_at_3_gt_0.45"] = True
    section["outside_1_5_excluded"] = True

    while True:
        points = sorted(nodes)
        best_q = max(points, key=lambda q: exact_fraction(nodes[q].lower()))
        lower = nodes[best_q].lower()
        intervals = []
        for left, right in zip(points, points[1:]):
            endpoint = max(nodes[left].upper(), nodes[right].upper())
            upper = (endpoint + arb(2).sqrt() * real_ball(right - left) ** 2 / 8).upper()
            intervals.append((upper, left, right))
        upper, left, right = max(intervals, key=lambda row: exact_fraction(row[0]))
        gap = upper ** 2 - lower ** 2
        section.update({
            "attained_lower_at_y": fraction_record(best_q),
            "amplitude_lower": ball_record(lower), "amplitude_upper": ball_record(upper),
            "C2_lower": ball_record(lower ** 2), "C2_upper": ball_record(upper ** 2),
            "C2_enclosure_width": ball_record(gap), "distinct_nodes": len(nodes),
            "active_max_interval": [fraction_record(left), fraction_record(right)],
        })
        save()
        if gap <= arb("2e-6"):
            section["status"] = "certified"
            break
        assert upper > lower
        evaluate_node((left + right) / 2)
    ctx.prec = 192


def ordinary_arithmetic(report):
    outputs = {}
    residual_maxima = {}
    for digits in (50, 80):
        with mp.workdps(digits):
            mu = mp.mpf(64) / 5
            aa = 2 ** (mp.mpf(1) / 4) * mp.gamma(mp.mpf(7) / 4) / (3 * mp.gamma(mp.mpf(3) / 2))
            values = {"A": aa}
            residuals = []
            for om_int in (13, 25, 50):
                om = mp.mpf(om_int)
                qq = (mu / om) ** 2
                xx = 1 + 2 / (1 + mp.sqrt(9 - 8 * qq))
                bb = mp.sqrt(xx ** 3 / (2 - xx))
                cm = mp.mpf(4) / 27 * xx ** 3 / (2 - xx)
                ll = om * mp.sqrt(1 - qq) * bb
                eta = mp.sqrt((3 - xx) / xx)
                curvature = 2 * ll * (xx - 1) * mp.sqrt(3 - xx) / xx ** (mp.mpf(7) / 2)
                derivative = 2 * ll * (1 - 1 / xx) / xx ** 2
                ss = (om / 64) ** 2 - (mp.mpf(1) / 5) ** 2
                assert 0 < ss < 1
                row = {
                    "q": qq, "x_c": xx, "b_c_over_rs": bb, "C_m": cm,
                    "L_c": ll, "eta_c": eta, "full_10_90_relative_width": eta * mp.log(9) / (mp.pi * ll),
                    "a_omega": om / 64, "v_infinity": mp.sqrt(1 - qq),
                    "axial_native_group_speed": mp.sqrt(1 - qq) * mp.sqrt(1 - ss),
                    "D_c": curvature, "dL_Vmax_over_Dc": derivative / curvature,
                }
                prefix = "Omega_" + str(om_int) + "."
                values.update({prefix + name: val for name, val in row.items()})
                equations = [
                    (2 * (1 - qq) * xx ** 2 + (4 * qq - 3) * xx - 2 * qq, 1),
                    (mu ** 2 * xx ** 2 - ll ** 2 * (2 * xx - 3), mu ** 2 * xx ** 2),
                    (om ** 2 * xx ** 3 - 2 * ll ** 2 * (xx - 1) ** 2, om ** 2 * xx ** 3),
                    ((om ** 2 - mu ** 2) * xx ** 3 - ll ** 2 * (2 - xx),
                     (om ** 2 - mu ** 2) * xx ** 3),
                    (derivative / curvature - 1 / eta, 1 / eta),
                ]
                for index, (residual, scale) in enumerate(equations):
                    norm = abs(residual) / max(mp.mpf(1), abs(scale))
                    assert norm < mp.mpf("1e-40")
                    residuals.append(norm)
                    values[prefix + "normalized_residual_" + str(index)] = norm
            # q=0 is checked algebraically at its exact rational radius.
            xx = Fraction(3, 2)
            assert xx ** 3 / (2 - xx) == Fraction(27, 4)
            assert Fraction(4, 27) * xx ** 3 / (2 - xx) == 1
            assert (3 - xx) / xx == 1
            outputs[str(digits)] = {key: mp.nstr(val, digits) for key, val in values.items()}
            residual_maxima[str(digits)] = mp.nstr(max(residuals), digits)
    # Near-zero residuals are compared absolutely; other values relatively
    # to max(1,|value|), with a fixed arithmetic tolerance 1e-40.
    with mp.workdps(90):
        differences = {key: abs(mp.mpf(outputs["50"][key]) - mp.mpf(outputs["80"][key]))
                       / max(mp.mpf(1), abs(mp.mpf(outputs["80"][key])))
                       for key in outputs["50"]}
        worst = max(differences, key=differences.get)
        assert differences[worst] < mp.mpf("1e-40")
        report["ordinary_scalar_check"] = {
            "status": "passed", "digits": [50, 80], "values": outputs,
            "normalized_residual_maxima": residual_maxima,
            "maximum_precision_difference": mp.nstr(differences[worst], 80),
            "maximum_difference_key": worst, "arithmetic_tolerance": "1e-40",
            "q_0_exact_identities": True,
            "q_to_1": "x_c->2, b_c~2*r_s/sqrt(1-q), L_c->2*m*r_s, eta_c->1/sqrt(2); analytic only",
            "scope": "ordinary scalar arithmetic; no physical, WKB, native or capture error certificate",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    os.nice(max(0, 10 - os.getpriority(os.PRIO_PROCESS, 0)))
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {available[0]})
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 ** 2, 512 * 1024 ** 2))
    resource.setrlimit(resource.RLIMIT_CPU, (175, 180))
    signal.signal(signal.SIGALRM, limit_signal)
    signal.signal(signal.SIGXCPU, limit_signal)
    signal.setitimer(signal.ITIMER_REAL, 175)
    ctx.prec = 192
    ctx.threads = 1
    began = time.monotonic()
    report = {
        "status": "incomplete", "purpose": "WEB40F inserts6/12 exact Gaussian thresholds and three massive-orbit scalar controls",
        "specifications": ["web40f_enclosed_mass_fold.tex EM5", "redteam_web40f_enclosed_mass_fold.tex EMR7",
                           "web40f_massive_critical_disk.tex MC1--5", "redteam_web40f_massive_critical_disk.tex MCR1--5"],
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_flint": flint.__version__, "mpmath": mp.__version__,
        "pid": os.getpid(), "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "threads": 1,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "memory_cap_bytes": 512 * 1024 ** 2, "hard_wall_seconds": 180,
        "graceful_wall_seconds": 175, "ordinary_scalar_check": {"status": "not_started"},
    }

    def save():
        report["elapsed_seconds"] = time.monotonic() - began
        atomic_report(args.out, report)

    save()
    try:
        # Finish the independent finite algebra first, preserving it even if
        # the bounded interval integral is incomplete.
        ordinary_arithmetic(report)
        save()
        point_envelope(report)
        save()
        double_envelope(report, save)
        report["status"] = "completed"
    except (ArithmeticError, AssertionError, MemoryError, TimeoutError, ValueError) as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report["cpu_seconds"] = usage.ru_utime + usage.ru_stime
        report["peak_rss_kib"] = usage.ru_maxrss
        save()
        print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"],
                          "peak_rss_kib": report["peak_rss_kib"], "failure": report.get("failure")}))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
