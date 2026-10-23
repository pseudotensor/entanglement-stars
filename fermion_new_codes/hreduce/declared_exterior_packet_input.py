#!/usr/bin/env python3
"""Bounded interval evaluation of the reviewed MP8 preparation inputs.

Scientific specification: declared_exterior.tex, BOUNDED NECESSARY
INPUT-COMPONENT SPECIFICATION; independent review: MP26--MP27 in
redteam_declared_exterior.tex. This evaluates no propagated observable.
Run only on the box, with the original pair's implementation clearance.
"""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time


START = time.monotonic()
MEMORY_BYTES = 512 * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
resource.setrlimit(resource.RLIMIT_CPU, (179, 180))


def cap_exceeded(signum, frame):
    raise TimeoutError("Reviewed 180-second resource boundary reached")


signal.signal(signal.SIGALRM, cap_exceeded)
signal.signal(signal.SIGXCPU, cap_exceeded)
signal.alarm(179)

import mpmath  # noqa: E402
from mpmath import iv  # noqa: E402

iv.dps = 100
N = 40
ZERO = iv.mpf(0)
ONE = iv.mpf(1)
EPSILON = ONE / (1 << 24)
ENLARGEMENT = iv.mpf(["-2e-24", "2e-24"])
WIDTH_LIMIT = iv.mpf("1e-25")
DYADIC_ZERO = ZERO._mpi_


def interval(x):
    return x if hasattr(x, "_mpi_") else iv.mpf(x)


class Jet:
    """Value and first derivatives in the two equatorial carrier entries."""

    __slots__ = ("v", "dx", "dy")

    def __init__(self, v=0, dx=0, dy=0):
        self.v, self.dx, self.dy = map(interval, (v, dx, dy))

    def __add__(self, b):
        if isinstance(b, Jet):
            return Jet(self.v + b.v, self.dx + b.dx, self.dy + b.dy)
        return Jet(self.v + interval(b), self.dx, self.dy)

    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.v, -self.dx, -self.dy)

    def __sub__(self, b):
        return self + (-b)

    def __rsub__(self, b):
        return -self + b

    def __mul__(self, b):
        if isinstance(b, Jet):
            return Jet(self.v * b.v,
                       self.dx * b.v + self.v * b.dx,
                       self.dy * b.v + self.v * b.dy)
        b = interval(b)
        return Jet(self.v * b, self.dx * b, self.dy * b)

    __rmul__ = __mul__

    def __truediv__(self, b):
        if isinstance(b, Jet):
            r = ONE / b.v
            v = self.v * r
            return Jet(v, (self.dx - v * b.dx) * r,
                       (self.dy - v * b.dy) * r)
        b = interval(b)
        return Jet(self.v / b, self.dx / b, self.dy / b)

    def __rtruediv__(self, b):
        return Jet(b) / self

    def sqrt(self):
        v = iv.sqrt(self.v)
        return Jet(v, self.dx / (2 * v), self.dy / (2 * v))

    def power(self, n):
        if n < 0:
            return 1 / self.power(-n)
        a, b = JONE, self
        while n:
            if n & 1:
                a = a * b
            b = b * b
            n //= 2
        return a

    def components(self):
        return self.v, self.dx, self.dy

    def is_zero(self):
        return all(x._mpi_ == DYADIC_ZERO for x in self.components())


JZERO, JONE = Jet(), Jet(1)


def mmul(a, b):
    return [[sum((a[i][k] * b[k][j] for k in range(len(b))), 0)
             for j in range(len(b[0]))] for i in range(len(a))]


def transpose(a):
    return list(map(list, zip(*a)))


def dot(a, b):
    return sum((y * x if isinstance(y, Jet) and not isinstance(x, Jet) else x * y
                for x, y in zip(a, b)), 0)


def contains_zero(x):
    return bool(x.a <= ZERO and x.b >= ZERO)


def require(condition, description):
    if not condition:
        raise ArithmeticError(description)


def inverse3(a):
    cof = []
    for i in range(3):
        row = []
        ii = [k for k in range(3) if k != i]
        for j in range(3):
            jj = [k for k in range(3) if k != j]
            row.append(((-1) ** (i + j)) *
                       (a[ii[0]][jj[0]] * a[ii[1]][jj[1]] -
                        a[ii[0]][jj[1]] * a[ii[1]][jj[0]]))
        cof.append(row)
    det = dot(a[0], cof[0])
    require(bool(det.a > ZERO), "Covariance determinant is not positive")
    return [[cof[j][i] / det for j in range(3)] for i in range(3)], det


def fixed_shape():
    r2, r3 = iv.sqrt(2), iv.sqrt(3)
    a = [[-iv.mpf(3) / 2, 3 * r2 / 2], [3 * r2 / 2, iv.mpf(3)]]
    b = [[iv.sqrt(ONE * 2 / 3) / 2, -ONE / r3],
         [ONE / r3, -iv.sqrt(ONE * 2 / 3)]]
    c = [[ONE / 9, -r2 / 9], [-r2 / 9, ONE * 2 / 9]]
    bt = transpose(b)
    kh = [bt[i] + c[i] for i in range(2)]
    kh += [[-x for x in a[i]] + [-x for x in b[i]] for i in range(2)]
    kh2 = mmul(kh, kh)
    ep, em = iv.exp(iv.pi), iv.exp(-iv.pi)
    ch, sh = (ep + em) / 2, (ep - em) / 2
    shoriz = [[(ONE if i == j else ZERO) +
               3 * (ch - 1) * kh2[i][j] - r3 * sh * kh[i][j]
               for j in range(4)] for i in range(4)]
    s = [[ZERO for _ in range(6)] for _ in range(6)]
    indices = [0, 1, 3, 4]
    for i, ii in enumerate(indices):
        for j, jj in enumerate(indices):
            s[ii][jj] = shoriz[i][j]
    s[2][2] = s[5][5] = -ONE
    g = mmul(s, transpose(s))
    v = [row[3:] for row in g[3:]]
    cqk = [row[3:] for row in g[:3]]
    vinv, det = inverse3(v)
    phase = mmul(cqk, vinv)
    sigma = [[EPSILON * x / 2 for x in row] for row in v]
    residuals = {name: [[matrix[i][j] - matrix[j][i] for j in range(3)]
                        for i in range(3)]
                 for name, matrix in (("V", v), ("B_phase", phase))}
    require(all(contains_zero(x) for a in residuals.values()
                for row in a for x in row), "Shape symmetry not enclosed")
    return s, v, phase, sigma, det, residuals


def series_mul(a, b):
    result = [JZERO] * N
    for i, x in enumerate(a):
        if x.is_zero():
            continue
        for j in range(N - i):
            y = b[j]
            if not y.is_zero():
                result[i + j] = result[i + j] + x * y
    return result


def series_add(a, b):
    return [x + y for x, y in zip(a, b)]


def series_scale(a, scalar):
    return [x * scalar for x in a]


def raw_moments(k, sigma):
    kappa = dot(k, k).sqrt()
    kap2 = kappa * kappa
    tmat = [[Jet(2 * x) / kap2 for x in row] for row in sigma]
    ell = [2 * x / kap2 for x in k]
    powers = [[[JONE if i == j else JZERO for j in range(3)]
               for i in range(3)]]
    for _ in range(1, N):
        powers.append(mmul(powers[-1], tmat))
    cs = [mmul(p, sigma) for p in powers]
    means = [[k[i]] + [dot(cs[n - 1][i], ell) for n in range(1, N)]
             for i in range(3)]
    covars = [[[cs[n][i][j] for n in range(N)] for j in range(3)]
              for i in range(3)]
    logf = [JZERO] * N
    for n in range(1, N):
        logf[n] = sum((powers[n][i][i] for i in range(3)), JZERO) / (2 * n)
        if n >= 2:
            logf[n] = logf[n] + dot(ell, [dot(row, ell) for row in cs[n - 2]]) / 2
    f = [JONE] + [JZERO] * (N - 1)
    for n in range(1, N):
        f[n] = sum((logf[l] * f[n - l] * l for l in range(1, n + 1)), JZERO) / n
    indices = sorted((alpha for alpha in itertools.product(range(6), repeat=3)
                      if sum(alpha) <= 5), key=lambda x: (sum(x), x))
    require(len(indices) == 56, "Unexpected raw-moment count")
    w = {(0, 0, 0): [JONE] + [JZERO] * (N - 1)}
    moments = {}
    for alpha in indices:
        if any(alpha):
            i = next(j for j in range(3) if alpha[j])
            beta = list(alpha)
            beta[i] -= 1
            value = series_mul(means[i], w[tuple(beta)])
            for j in range(3):
                if beta[j]:
                    gamma = beta.copy()
                    gamma[j] -= 1
                    value = series_add(value, series_scale(
                        series_mul(covars[i][j], w[tuple(gamma)]), beta[j]))
            w[alpha] = value
        moments[alpha] = series_mul(f, w[alpha])
    checks = {"F_0_minus_one": f[0] - JONE,
              "zeroth_moment_minus_one": moments[(0, 0, 0)][0] - JONE}
    for i in range(3):
        alpha = [0, 0, 0]
        alpha[i] += 1
        checks[f"first_{i}_residual"] = moments[tuple(alpha)][0] - k[i]
        for j in range(i, 3):
            beta = alpha.copy()
            beta[j] += 1
            checks[f"second_{i}_{j}_residual"] = (
                moments[tuple(beta)][0] - (k[i] * k[j] + sigma[i][j]))
    require(all(contains_zero(c) for x in checks.values() for c in x.components()),
            "Gaussian zero/first/second moment identity not enclosed")
    return kappa, moments, checks


def floor_scaled_endpoint(raw, places, upper=False):
    """Exact dyadic -> outward fixed-decimal conversion, using integers only."""
    sign, mantissa, exponent, bitcount = raw
    require(bitcount >= 0, "Cannot serialize a nonfinite endpoint")
    numerator = (-mantissa if sign else mantissa) * 10 ** places
    if exponent >= 0:
        return numerator << exponent
    denominator = 1 << (-exponent)
    return -((-numerator) // denominator) if upper else numerator // denominator


def decimal_scaled(n, places):
    sign = "-" if n < 0 else ""
    digits = str(abs(n)).rjust(places + 1, "0")
    return sign + digits[:-places] + "." + digits[-places:]


def record_interval(x):
    a, b = x._mpi_
    return {"lower": decimal_scaled(floor_scaled_endpoint(a, 110), 110),
            "upper": decimal_scaled(floor_scaled_endpoint(b, 110, upper=True), 110)}


def record_jet(x):
    return dict(zip(("value", "carrier_dx", "carrier_dy"),
                    (record_interval(c) for c in x.components())))


def record_matrix(a):
    return [[record_interval(x) for x in row] for row in a]


def evaluate(carrier_x=None, carrier_y=None):
    s, v, phase, sigma, det, symmetry = fixed_shape()
    reference = [iv.sqrt(6), iv.sqrt(3), ZERO]
    if carrier_x is None and carrier_y is None:
        values = reference
    else:
        require(carrier_x is not None and carrier_y is not None,
                "Supply both equatorial carrier entries")
        values = [iv.mpf(carrier_x), iv.mpf(carrier_y), ZERO]
    distance = iv.sqrt(sum(((a - b) ** 2 for a, b in zip(values, reference)), ZERO))
    require(bool(distance.b <= ONE / 4), "Carrier is outside the reviewed neighborhood")
    k = [Jet(values[0], 1, 0), Jet(values[1], 0, 1), Jet(0)]
    kappa, moments, checks = raw_moments(k, sigma)
    normal = [x / kappa for x in k]
    cache = {}

    def expectation(alpha, inverse_power):
        key = tuple(alpha), inverse_power
        if key not in cache:
            series = moments[tuple(alpha)]
            total, coefficient = JZERO, ONE
            for l in range(N):
                if l:
                    # binom(-j/2,l)*l! is the exact falling product.
                    coefficient = coefficient * (-iv.mpf(inverse_power) / 2 - (l - 1))
                total = total + series[l] * coefficient
            cache[key] = total / kappa.power(inverse_power)
        return cache[key]

    def unit(i, j=None):
        a = [0, 0, 0]
        a[i] += 1
        if j is not None:
            a[j] += 1
        return a

    ep1 = [expectation(unit(i), 1) for i in range(3)]
    z = (JONE + dot(normal, ep1)) / 2
    ezs = []
    for i in range(3):
        # E[z_i]=0 exactly for the uncut centered polynomial comparison.
        ezs.append(sum((normal[j] * (expectation(unit(i, j), 1) - k[i] * ep1[j])
                        for j in range(3)), JZERO) / 2)
    ezp2 = [expectation(unit(i), 2) - k[i] * expectation((0, 0, 0), 2)
            for i in range(3)]
    cross = [normal[1] * ezp2[2] - normal[2] * ezp2[1],
             normal[2] * ezp2[0] - normal[0] * ezp2[2],
             normal[0] * ezp2[1] - normal[1] * ezp2[0]]
    mu = [(dot(phase[i], ezs) - cross[i] * EPSILON / 4) / z for i in range(3)]
    positive, negative = [1, 3, 0], [3, 1, 0]
    numerator = expectation(positive, 3) - expectation(negative, 3)
    for j in range(3):
        ap, an = positive.copy(), negative.copy()
        ap[j] += 1
        an[j] += 1
        numerator = numerator + normal[j] * (expectation(ap, 4) - expectation(an, 4))
    mz = numerator / (24 * z)
    outputs = {"Z": z, "normalization_factor": 1 / z.sqrt(), "m_native_z": mz,
               "native_centroid_spacing_derivative_z": mz * EPSILON}
    for i in range(3):
        outputs[f"mu_{i}"] = mu[i]
        outputs[f"q_c_{i}"] = (1 if i == 0 else 0) - mu[i]
    widths = [c.delta for x in outputs.values() for c in x.components()]
    require(all(bool(w.b <= WIDTH_LIMIT) for w in widths),
            "A finite-polynomial output interval exceeds 1e-25 width")
    max_width = max(w.b for w in widths)
    enlarged = {name: Jet(*(c + ENLARGEMENT for c in x.components()))
                for name, x in outputs.items()}
    muv = [enlarged[f"mu_{i}"] for i in range(3)]
    munorm = iv.sqrt(sum((x.v ** 2 for x in muv), ZERO))
    derivative_norm = iv.sqrt(sum((x.dx ** 2 + x.dy ** 2 for x in muv), ZERO))
    require(bool(enlarged["Z"].v.a > iv.mpf(193) / 200), "Z lower bound failed")
    require(bool(munorm.b < ONE / 20), "Continuum-centering bound failed")
    require(bool(derivative_norm.b < iv.mpf(17) / 50), "Centering-derivative bound failed")
    require(bool(abs(enlarged["m_native_z"].v).b <= iv.mpf(17) / 48),
            "Native-centering bound failed")
    require(all(bool(c.delta.b <= iv.mpf("1e-20"))
                for x in enlarged.values() for c in x.components()),
            "Enlarged output does not meet the final absolute target")
    return {
        "status": "passed",
        "scope": "Necessary exact-preparation inputs; no propagation or return coefficient",
        "parameters": {"epsilon": record_interval(EPSILON),
                       "carrier": [record_interval(x) for x in values],
                       "carrier_distance": record_interval(distance),
                       "S": record_matrix(s), "V": record_matrix(v),
                       "B_phase": record_matrix(phase), "Sigma": record_matrix(sigma),
                       "V_determinant": record_interval(det)},
        "finite_polynomial_outputs": {name: record_jet(x) for name, x in outputs.items()},
        "enlarged_outputs": {name: record_jet(x) for name, x in enlarged.items()},
        "checks": {"maximum_raw_interval_width": record_interval(max_width),
                   "shape_symmetry_residuals": {name: record_matrix(a)
                                                for name, a in symmetry.items()},
                   "gaussian_moment_residuals": {name: record_jet(x)
                                                 for name, x in checks.items()},
                   "mu_norm_enclosure": record_interval(munorm),
                   "mu_first_derivative_frobenius_enclosure": record_interval(derivative_norm)},
        "analytic_budget": {"inverse_power_terms": N, "raw_moment_count": 56,
                            "truncation_absolute_bound": "6e-25",
                            "cutoff_absolute_bound": "1e-42",
                            "endpoint_enlargement": "2e-24",
                            "finite_polynomial_width_limit": "1e-25",
                            "final_absolute_target": "1e-20",
                            "source": "Author bounded specification and RED MP26--MP27"},
        "exact_zero_native_components": ["x", "y"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--carrier-x")
    parser.add_argument("--carrier-y")
    args = parser.parse_args()
    try:
        data = evaluate(args.carrier_x, args.carrier_y)
    except Exception as error:
        data = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
    data["execution"] = {
        "runtime_seconds": time.monotonic() - START,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "memory_limit_bytes": MEMORY_BYTES, "hard_wall_limit_seconds": 180,
        "cpu_limit_seconds": 180, "iv_decimal_digits": iv.dps,
        "endpoint_decimal_places": 110, "mpmath_version": mpmath.__version__,
        "python_version": sys.version, "hostname": os.uname().nodename,
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "thread_limits": {key: os.environ.get(key) for key in
                          ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": data["status"], "output": str(args.output),
                      "runtime_seconds": data["execution"]["runtime_seconds"],
                      "peak_rss_kib": data["execution"]["peak_rss_kib"]}))
    return 0 if data["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
