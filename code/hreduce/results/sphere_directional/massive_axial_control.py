#!/usr/bin/env python3
"""Bounded exact-axial numerical prefix for the required H1 massive correction.

Implements the already derived eta100 integral and the H1.12 positive-mass
linear functional. These are ordinary mpmath evaluations, not interval
certificates or bounds on other directions' mass/angular remainders.
Scientific runs belong on the authorized box, in qi2, nice10, one CPU.
"""
import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import resource
import signal
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import mpmath as mp


def number(q):
    q = Fraction(q)
    return mp.mpf(q.numerator) / q.denominator


def elliptic_complement(c):
    """K(1-c) using the complementary parameter, without subtraction."""
    return mp.pi / (2 * mp.agm(1, mp.sqrt(c)))


def axial_chain(s):
    """Full-four, one-face entropy f(s), with s the squared effective gap."""
    t = 1 / (1 + s)
    q = s / (1 + s)
    return (mp.log(16) + 2 * mp.log1p(s) - mp.log(s)
            + (4 / mp.pi) * elliptic_complement(q)
            * elliptic_complement(t) * (1-s) / (1+s)) / 6


def eta100(y, split):
    """Integrate s=1-t and s=1+t; the integrable log is at t=0."""
    yy = number(y)

    def integrand(t):
        if not t:
            # A single endpoint has zero measure; avoid an explicit infinity.
            return mp.mpf(0)
        rho = 1 / (mp.pi * mp.agm(1, t))
        return rho * (axial_chain(1-t+yy) + axial_chain(1+t+yy))

    points = [mp.mpf(0), mp.mpf(1)]
    if split == "halves":
        points.insert(1, mp.mpf("0.5"))
    return mp.quad(integrand, points, method="tanh-sinh")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--squared-masses", nargs="+", required=True)
    ap.add_argument("--filter-y", nargs="*", default=[])
    ap.add_argument("--digits", nargs="+", type=int, default=[40, 60])
    ap.add_argument("--b100-reference", required=True)
    ap.add_argument("--wall-seconds", type=int, default=600)
    ap.add_argument("--memory-mib", type=int, default=1024)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    masses = [Fraction(x) for x in args.squared_masses]
    filters = [Fraction(x) for x in args.filter_y]
    if len(set(masses)) != len(masses) or any(x <= 0 for x in masses):
        ap.error("need distinct strictly positive squared masses")
    if min(args.digits) < 30 or args.wall_seconds <= 0 or args.memory_mib < 128:
        ap.error("invalid numerical/resource controls")
    for y in filters:
        if y <= 0 or not {2*y, y, y/2, y/4}.issubset(set(masses)):
            ap.error("each positive filter needs its four supplied squared masses")

    os.nice(max(0, 10-os.getpriority(os.PRIO_PROCESS, 0)))
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    memory = args.memory_mib * 1024**2
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_CPU, (args.wall_seconds, args.wall_seconds+1))

    def timeout(signum, frame):
        raise TimeoutError("axial control wall cap reached")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.wall_seconds)
    start = time.monotonic()
    result = {
        "status": "incomplete",
        "purpose": "required H1 b_sphere exact-axial mass-estimator control",
        "scientific_source": "existing H1.4 exact axial integral and H1.12 positive-mass functional",
        "units": "a=1; y=(ma)^2; nats; full four-component field; one physical face",
        "scope": "ordinary numerical control only; no interval certificate, nonaxial value, angular or mass-tail bound",
        "pid": os.getpid(), "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "wall_cap_seconds": args.wall_seconds, "memory_cap_bytes": memory,
        "mpmath_version": mp.__version__,
        "squared_masses": [str(x) for x in masses],
        "b100_reference": args.b100_reference,
        "evaluations": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result.update(elapsed_seconds=time.monotonic()-start,
                      cpu_seconds=usage.ru_utime+usage.ru_stime,
                      peak_rss_KiB=usage.ru_maxrss)
        tmp = args.output.with_suffix(args.output.suffix+".tmp")
        tmp.write_text(json.dumps(result, indent=2)+"\n")
        tmp.replace(args.output)

    save()
    try:
        for digits in args.digits:
            with mp.workdps(digits):
                render = lambda x: mp.nstr(x, digits+8)
                for split in ("whole", "halves"):
                    values = {}
                    row = {"digits": digits, "integration_partition": split,
                           "eta100": [], "positive_mass_filter": []}
                    result["evaluations"].append(row)
                    for y in masses:
                        value = eta100(y, split)
                        if not mp.isfinite(value):
                            raise ArithmeticError("nonfinite exact-axial integral")
                        values[y] = value
                        row["eta100"].append({"y": str(y), "ma": render(mp.sqrt(number(y))),
                                              "eta": render(value)})
                        save()
                    for y in filters:
                        yy = number(y)
                        weighted = (values[2*y] - 9*values[y]
                                    + 24*values[y/2] - 16*values[y/4]) / yy
                        estimate = weighted - (mp.log(yy)-2*mp.log(2))/(3*mp.pi)
                        row["positive_mass_filter"].append({
                            "y": str(y), "E100": render(estimate),
                            "bias_from_b100_reference": render(estimate-mp.mpf(args.b100_reference)),
                            "weights_eta_2y_y_yhalf_yquarter": [str(Fraction(a, 1)/y) for a in (1, -9, 24, -16)],
                            "numerical_error_bound": None,
                            "mass_remainder_bound": None,
                        })
                    save()
        signal.alarm(0)
        result["status"] = "completed_numerical_control"
        save()
    except Exception as exc:
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        save()
        raise
    print(json.dumps({k: result[k] for k in ("status", "elapsed_seconds", "cpu_seconds", "peak_rss_KiB")}, indent=2))


if __name__ == "__main__":
    main()
