"""Required H1 one-dimensional elliptic check, from the original reviewer B pair's plan.

Run only on compute-host with qi2 Python, nice 10 and one CPU thread.
Precision agreement is numerical evidence, not interval quadrature certification.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import mpmath as mp


def evaluate(digits, extra_split=False):
    mp.mp.dps = digits
    delta = mp.mpf('1e-16')
    ladder_tolerance = mp.mpf('1e-35')
    series_terms = []
    calls = 0

    def p3(z):
        if z == 0 or z == 1:
            return mp.mpf(0)  # The isolated logarithmic singularity is integrable.
        if z == 3:
            return mp.sqrt(3) / (2 * mp.pi)
        area = (1 + z) ** 3 * (3 - z)
        if z < 1:
            complement = (1 - z) ** 3 * (z + 3) / area
            return 4 * z * mp.elliprf(0, complement, 1) / (mp.pi**2 * mp.sqrt(area))
        complement = (z - 1) ** 3 * (z + 3) / (16 * z)
        return mp.sqrt(z) * mp.elliprf(0, complement, 1) / mp.pi**2

    def entropy(z):
        nonlocal calls
        calls += 1
        if z == 0:
            return 2 * mp.log(2)
        ratio = (3 - z) / (3 + z)
        epsilon = 2 * mp.pi * mp.elliprf(0, 1 - ratio, 1) / mp.elliprf(0, ratio, 1)
        q = mp.exp(-epsilon)
        value = 2 * mp.log(2)
        power = q
        j = 0
        while True:
            j += 1
            value += 4 * (mp.log1p(power) + j * epsilon * power / (1 + power))
            following = power * q
            tail = 4 * following * (
                (1 + (j + 1) * epsilon) / (1 - q)
                + epsilon * q / (1 - q)**2
            )
            if tail <= ladder_tolerance:
                break
            power = following
            if j > 2000:
                raise RuntimeError('Unexpected ladder cost outside the specified endpoint cutoff')
        series_terms.append(j)
        return value

    panels = [mp.mpf(0), mp.mpf(1), 3 - delta]
    if extra_split:
        panels = list(map(mp.mpf, ['0', '.5', '1', '1.5', '2.5'])) + [3 - delta]

    def integrand(z):
        if z == 0 or z == 1:
            return mp.mpf(0)
        return p3(z) * entropy(z) / mp.sqrt(3)

    started = time.perf_counter()
    value, quadrature_estimate = mp.quad(integrand, panels, error=True, maxdegree=7)
    normalization, normalization_estimate = mp.quad(p3, [0, 1, 3], error=True, maxdegree=7)
    c0 = 2 * mp.log(2) + mp.pi / 3 * mp.log(11 * mp.sqrt(6) / 3) + 2 * mp.pi / (9 * mp.sqrt(3))
    endpoint_author = delta / (mp.pi * mp.sqrt(2)) * (c0 + mp.pi / 6 * (1 - mp.log(delta)))
    cf = 2 * mp.log(2) + mp.mpf(8) / 3 + mp.mpf(2) / 3 * mp.log(3 / mp.sqrt(5))
    endpoint_red = (2 + mp.log(2) / 2) * delta / mp.pi**2 * (cf + (1 + mp.log(1 / delta)) / 3)
    def b_integrand(s):
        if s == 0 or s == 1:
            return mp.mpf(0)
        # Extra precision controls cancellation in the exact analytic derivative.
        with mp.extradps(digits + 30):
            t = 1 / (1 + s)
            q = s / (1 + s)
            kt = mp.elliprf(0, q, 1)
            kq = mp.elliprf(0, t, 1)
            et = mp.ellipe(t)
            g = (1 - 2 / mp.pi * kq * (q * kt + (2 * t - 1) * et)) / (3 * s)
            density = 2 * mp.elliprf(0, (1 - s)**2, 1) / mp.pi**2
            return +(density * g)
    b_panels = [0, 1, 2] if not extra_split else [0, mp.mpf('.5'), 1, mp.mpf('1.5'), 2]
    b_integral, b_estimate = mp.quad(b_integrand, b_panels, error=True, maxdegree=7)
    b100 = b_integral - (1 + mp.log(8)) / (3 * mp.pi)
    text = lambda x: mp.nstr(x, min(digits, 65))
    return {
        'digits': digits, 'extra_split': extra_split, 'eta111_truncated_integral': text(value),
        'endpoint_delta': text(delta), 'analytic_endpoint_bound_author': text(endpoint_author),
        'analytic_endpoint_bound_red': text(endpoint_red),
        'analytic_integrated_ladder_bound': text(ladder_tolerance / mp.sqrt(3)),
        'quadrature_estimate_not_certified': text(quadrature_estimate),
        'p3_normalization': text(normalization),
        'p3_normalization_estimate_not_certified': text(normalization_estimate),
        'b100': text(b100), 'b100_quadrature_estimate_not_certified': text(b_estimate),
        'F_at_zero': text(entropy(mp.mpf(0))),
        'entropy_calls': calls, 'maximum_series_terms': max(series_terms),
        'elapsed_seconds': time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (600, 610))
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    start = time.perf_counter()
    records = []
    for digits, extra in [(70, False), (100, False), (100, True)]:
        records.append(evaluate(digits, extra))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'status': 'running', 'records': records}, indent=2) + '\n')
    mp.mp.dps = 100
    values = [mp.mpf(r['eta111_truncated_integral']) for r in records]
    b_values = [mp.mpf(r['b100']) for r in records]
    result = {
        'status': 'complete', 'serves': 'WEB40E H1, duty2c/insert20 exact [111] integral',
        'records': records,
        'maximum_precision_or_partition_change': mp.nstr(max(abs(x - values[-1]) for x in values), 25),
        'b100_maximum_precision_or_partition_change': mp.nstr(max(abs(x - b_values[-1]) for x in b_values), 25),
        'difference_from_supplied_decimal': mp.nstr(values[-1] - mp.mpf('.92338180595796'), 25),
        'difference_from_previous_lane_directional_value': mp.nstr(values[-1] - mp.mpf('.9233818782026300'), 25),
        'quad_status': 'Numerical precision/partition evidence only; panel interval quadrature is not implemented.',
        'scope': 'No sphere enclosure, directional error certificate, massive fit or genuine ball-cut calculation.',
        'elapsed_seconds': time.perf_counter() - start,
        'cpu_seconds': resource.getrusage(resource.RUSAGE_SELF).ru_utime,
        'maximum_resident_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: val for key, val in result.items() if key != 'records'}, indent=2))


if __name__ == '__main__':
    main()
