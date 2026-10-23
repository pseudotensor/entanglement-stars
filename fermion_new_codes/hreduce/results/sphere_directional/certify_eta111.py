#!/usr/bin/env python3
"""Certify the exact native [111] integral specified in H1.8.

Scientific specification: web40e_h1_sphere_bounds.tex H1.8; independent
prefix: redteam_web40e_h1_sphere_bounds.tex C2. No directional sweep.
Run on the box in qi2 with isolated python-flint on PYTHONPATH, nice10,
one thread. Partial panel records are numerical results, never acceptance.
"""
import argparse
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

import flint
from flint import arb, acb, ctx


def exact_fraction(x):
    mantissa, exponent = x.man_exp()
    return Fraction(int(mantissa)) * Fraction(2) ** int(exponent)


def decimal_endpoint(x, rounding):
    q = exact_fraction(x)
    with localcontext() as context:
        context.prec = 70
        context.rounding = rounding
        return str(Decimal(q.numerator) / Decimal(q.denominator))


def ball_record(x):
    if not x.is_finite():
        return {"finite": False}
    lo, hi = x.lower(), x.upper()
    lm, le = lo.man_exp()
    hm, he = hi.man_exp()
    return {
        "finite": True, "ball": x.str(65),
        "lower": decimal_endpoint(lo, ROUND_FLOOR),
        "upper": decimal_endpoint(hi, ROUND_CEILING),
        "lower_dyadic": [str(lm), int(le)],
        "upper_dyadic": [str(hm), int(he)],
        "radius": x.rad().str(20),
    }


def ball_of_fraction(x):
    return arb(x.numerator) / arb(x.denominator)


def k_complement(c, analytic):
    # Explicit complement avoids losing c when it is much smaller than one.
    # Principal sqrt has positive real part away from its negative-real cut;
    # AGM(1,sqrt(c)) then defines the intended analytic elliptic branch.
    root = c.sqrt(analytic=analytic)
    if not root.is_finite() or not root.real > 0:
        return acb("nan")
    return acb.pi() / (2 * acb(1).agm(root))


def epsilon(z, analytic):
    r2 = (3 - z) / (3 + z)
    return 2 * acb.pi() * k_complement(1 - r2, analytic) / k_complement(r2, analytic)


def density(z, side, analytic):
    pi2 = acb.pi() ** 2
    if side == 0:
        aa = (1 + z) ** 3 * (3 - z)
        cc = (1 - z) ** 3 * (z + 3) / aa
        return 4 * z * k_complement(cc, analytic) / (pi2 * aa.sqrt(analytic=analytic))
    cc = (z - 1) ** 3 * (z + 3) / (16 * z)
    return z.sqrt(analytic=analytic) * k_complement(cc, analytic) / pi2


def ladder_bound(eps, count):
    qq = (-eps).exp()
    return 4 * qq ** (count + 1) * (
        (1 + (count + 1) * eps) / (1 - qq)
        + eps * qq / (1 - qq) ** 2
    )


def choose_count(right, tau):
    ee = epsilon(acb(ball_of_fraction(right)), False)
    if not ee.is_finite() or not ee.real > 0 or not ee.imag.contains(0):
        raise ArithmeticError("uncertified real endpoint epsilon")
    count = 0
    while not ladder_bound(ee.real, count) < tau:
        count += 1
        if count > 10000:
            raise ArithmeticError("ladder count exceeds specified implementation bound")
    return count, ee.real


def integrand(side, count, calls):
    def evaluate(z, analytic):
        calls[0] += 1
        ee = epsilon(z, analytic)
        if not ee.is_finite() or not ee.real > 0:
            return acb("nan")
        qq = (-ee).exp()
        power = acb(1)
        ff = 2 * acb(2).log(analytic=analytic)
        for j in range(1, count + 1):
            power *= qq
            base = 1 + power
            if not base.real > 0:
                return acb("nan")
            ff += 4 * (base.log(analytic=analytic) + j * ee * power / base)
        return density(z, side, analytic) * ff / acb(3).sqrt(analytic=analytic)
    return evaluate


def panels():
    dyadics = [Fraction(1, 2 ** k) for k in range(1, 65)]
    left = sorted(set(dyadics + [1 - q for q in dyadics]))
    right = sorted(set([1 + q for q in dyadics] + [3 - q for q in dyadics] + [Fraction(2)]))
    return [(side, a, b) for side, points in enumerate((left, right))
            for a, b in zip(points, points[1:])]


def endpoint_budgets():
    dd = arb(2) ** -64
    pp = arb.pi()
    b0 = 4 * dd ** 2 / (3 * pp * (1 - 16 * dd / 3).sqrt())
    b1 = 8 * dd / (pp * arb(3).sqrt()) * (2 + arb(3).log() + arb(3) / 2 * (1 - dd.log()))
    c0 = 2 * arb(2).log() + pp / 3 * (11 * arb(6).sqrt() / 3).log() + 2 * pp / (9 * arb(3).sqrt())
    b3 = dd / (pp * arb(2).sqrt()) * (c0 + pp / 6 * (1 - dd.log()))
    tail = arb(2) ** -100 / arb(3).sqrt()
    return {"zero_cap": b0, "one_cap": b1, "three_cap": b3, "ladder_tail": tail}


def atomic_report(path, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def limit_signal(_number, _frame):
    raise TimeoutError("bounded certificate prefix resource limit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, default=600)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGALRM, limit_signal)
    signal.signal(signal.SIGXCPU, limit_signal)
    resource.setrlimit(resource.RLIMIT_AS, (1024 ** 3, 1024 ** 3))
    resource.setrlimit(resource.RLIMIT_CPU, (args.wall_seconds - 5, args.wall_seconds))
    signal.setitimer(signal.ITIMER_REAL, args.wall_seconds - 5)
    ctx.prec = 192
    ctx.threads = 1
    started = time.monotonic()
    report = {
        "status": "incomplete", "scientific_specification": "web40e_h1_sphere_bounds.tex H1.8",
        "independent_prefix": "redteam_web40e_h1_sphere_bounds.tex C2",
        "conventions": "one native four-component field; one planar face; nats; a=1; massless",
        "python_flint": flint.__version__, "pid": os.getpid(), "threads": 1,
        "initial_bits": 192, "memory_cap_bytes": 1024 ** 3,
        "wall_cap_seconds": args.wall_seconds, "cutoffs": "all 2^-64",
        "panels": [], "expected_panels": len(panels()),
    }
    atomic_report(args.out, report)
    regular_sum = arb(0)
    try:
        for index, (side, left, right) in enumerate(panels()):
            one_start = time.monotonic()
            for precision in (192, 256, 320):
                ctx.prec = precision
                tau = arb(2) ** -100
                count, endpoint_epsilon = choose_count(right, tau)
                calls = [0]
                value = acb.integral(
                    integrand(side, count, calls), acb(ball_of_fraction(left)),
                    acb(ball_of_fraction(right)), rel_tol=0, abs_tol=arb(2) ** -90,
                    deg_limit=64, eval_limit=20000, depth_limit=30,
                )
                if value.is_finite():
                    break
            report["last_panel_attempt"] = {
                "index": index, "side": side, "left": str(left), "right": str(right),
                "bits": precision, "ladder_terms": count,
                "callback_evaluations": calls[0], "integral": ball_record(value.real),
            }
            if not value.is_finite() or not value.imag.contains(0):
                raise ArithmeticError("regular-panel integral is indeterminate")
            regular_sum += value.real
            report["panels"].append({
                "index": index, "side": side, "left": str(left), "right": str(right),
                "bits": precision, "ladder_terms": count,
                "epsilon_at_right": ball_record(endpoint_epsilon),
                "integral": ball_record(value.real), "callback_evaluations": calls[0],
                "elapsed_seconds": time.monotonic() - one_start,
            })
            report["partial_regular_sum"] = ball_record(regular_sum)
            report["elapsed_seconds"] = time.monotonic() - started
            atomic_report(args.out, report)
            if (index + 1) % 16 == 0:
                print(json.dumps({"panels_done": index + 1, "expected": report["expected_panels"],
                                  "seconds": report["elapsed_seconds"]}), flush=True)
        if not regular_sum.rad() < arb(2) ** -80:
            raise ArithmeticError("total regular quadrature radius exceeds 2^-80")
        budgets = endpoint_budgets()
        upper = sum(budgets.values(), arb(0)).upper()
        positive_error = arb(upper / 2, upper / 2)
        assert positive_error.contains(0) and positive_error.contains(upper)
        complete = regular_sum + positive_error
        report.update({"status": "certified", "eta111": ball_record(complete),
                       "regular_integral": ball_record(regular_sum),
                       "positive_error_budgets": {k: ball_record(v) for k, v in budgets.items()},
                       "combined_positive_error_upper": ball_record(upper)})
    except (ArithmeticError, MemoryError, TimeoutError, ValueError) as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        report["elapsed_seconds"] = time.monotonic() - started
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report["cpu_seconds"] = usage.ru_utime + usage.ru_stime
        report["peak_rss_KiB"] = usage.ru_maxrss
        atomic_report(args.out, report)
    print(json.dumps({"status": report["status"], "panels_done": len(report["panels"]),
                      "expected": report["expected_panels"], "seconds": report["elapsed_seconds"],
                      "failure": report.get("failure"), "eta111": report.get("eta111")}), flush=True)
    return 0 if report["status"] == "certified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
