#!/usr/bin/env python3
"""Bounded first two-spin amplitude of the specified packet recurrence.

Scientific method: declared_exterior.tex, PROPOSED BOUNDED FIRST MATRIX-
AMPLITUDE COMPONENT; independent reduction/review: RED MP40--41.
This computes one finite recurrence coefficient, not a propagated packet,
residual, return root, or finite-frequency orbit coefficient.
Run on the box only after the original pair reads this actual producer.
"""

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import resource
import signal
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
RZERO = iv.mpf(0)
DYADIC_ZERO = RZERO._mpi_
WIDTH_LIMIT = iv.mpf("1e-25")
CHECKS = []
MAX_WIDTH = RZERO


def require(condition, description):
    if not condition:
        raise ArithmeticError(description)


class CI:
    """Complex pairs of directed real intervals; no floating arithmetic."""

    __slots__ = ("r", "i")

    def __init__(self, real=0, imag=0):
        self.r = real if hasattr(real, "_mpi_") else iv.mpf(real)
        self.i = imag if hasattr(imag, "_mpi_") else iv.mpf(imag)

    @staticmethod
    def coerce(value):
        return value if isinstance(value, CI) else CI(value)

    def __add__(self, other):
        other = CI.coerce(other)
        return CI(self.r + other.r, self.i + other.i)

    __radd__ = __add__

    def __neg__(self):
        return CI(-self.r, -self.i)

    def __sub__(self, other):
        return self + -CI.coerce(other)

    def __rsub__(self, other):
        return CI.coerce(other) + -self

    def __mul__(self, other):
        other = CI.coerce(other)
        return CI(self.r * other.r - self.i * other.i,
                  self.r * other.i + self.i * other.r)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = CI.coerce(other)
        denominator = other.r * other.r + other.i * other.i
        require(bool(denominator.a > 0), "Complex denominator reaches zero")
        return CI((self.r * other.r + self.i * other.i) / denominator,
                  (self.i * other.r - self.r * other.i) / denominator)

    def __rtruediv__(self, other):
        return CI.coerce(other) / self

    def conjugate(self):
        return CI(self.r, -self.i)

    def exact_zero(self):
        return self.r._mpi_ == DYADIC_ZERO and self.i._mpi_ == DYADIC_ZERO


ZERO = CI()
ONE = CI(1)
I = CI(0, 1)
SQRT2 = CI(iv.sqrt(2))
SQRT3 = CI(iv.sqrt(3))
SQRT6 = CI(iv.sqrt(6))
PI = CI(iv.pi)
FZERO = (0, 0)
FREQUENCIES = sorted((m, n) for m in range(-3, 4) for n in range(-3, 4)
                     if abs(m) + abs(n) <= 3)
MODES = sorted(nu for nu in itertools.product(range(4), repeat=3)
               if sum(nu) in (1, 3))


def check_zero(value, description):
    value = CI.coerce(value)
    require(bool(value.r.a <= 0 <= value.r.b)
            and bool(value.i.a <= 0 <= value.i.b), description)
    CHECKS.append((description, value))


def fq_clean(values):
    return {key: value for key, value in values.items() if not value.exact_zero()}


def fq_constant(value):
    value = CI.coerce(value)
    return {} if value.exact_zero() else {FZERO: value}


def fq_add(left, right):
    answer = dict(left)
    for key, value in right.items():
        answer[key] = answer.get(key, ZERO) + value
    return fq_clean(answer)


def fq_scale(values, scalar):
    scalar = CI.coerce(scalar)
    return fq_clean({key: value * scalar for key, value in values.items()})


def fq_mul(left, right):
    answer = {}
    for (m, n), x in left.items():
        for (p, q), y in right.items():
            key = (m + p, n + q)
            answer[key] = answer.get(key, ZERO) + x * y
    return fq_clean(answer)


def endpoint_exp(frequency, endpoint):
    """endpoint=-1,0,+1 denotes u=-pi,0,+pi; integer phases are exact."""
    m, n = frequency
    if endpoint == 0:
        return ONE
    return CI((-1 if n % 2 else 1) * iv.exp(endpoint * m * iv.pi))


def fq_evaluate(values, endpoint):
    return sum((coefficient * endpoint_exp(key, endpoint)
                for key, coefficient in values.items()), ZERO)


def p_add(left, right):
    answer = dict(left)
    for nu, coefficient in right.items():
        answer[nu] = fq_add(answer.get(nu, {}), coefficient)
    return {nu: c for nu, c in answer.items() if c}


def p_scale(poly, coefficient):
    return {nu: fq_mul(c, coefficient) for nu, c in poly.items()}


def p_mul(left, right):
    answer = {}
    for nu, x in left.items():
        for mu, y in right.items():
            key = tuple(a + b for a, b in zip(nu, mu))
            answer[key] = fq_add(answer.get(key, {}), fq_mul(x, y))
    return {nu: c for nu, c in answer.items() if c}


def p_const(value, dimension):
    return {(0,) * dimension: fq_constant(value)}


def p_var(axis, dimension):
    return {tuple(int(j == axis) for j in range(dimension)): fq_constant(1)}


def p_sum(polys):
    answer = {}
    for poly in polys:
        answer = p_add(answer, poly)
    return answer


def ci_matmul(left, right):
    return [[sum((left[i][k] * right[k][j] for k in range(len(right))), ZERO)
             for j in range(len(right[0]))] for i in range(len(left))]


def native_inputs():
    """Literal A,B,C,K and exact five-frequency S from the cleared spec."""
    flow = SQRT2 / SQRT3
    a = [[-ONE * 3 / 2, SQRT2 * 3 / 2, ZERO],
         [SQRT2 * 3 / 2, ONE * 3, ZERO], [ZERO, ZERO, ONE * 3]]
    b = [[flow / 2, -ONE / SQRT3, ZERO],
         [ONE / SQRT3, -flow, ZERO], [ZERO, ZERO, -flow]]
    c = [[ONE / 9, -SQRT2 / 9, ZERO],
         [-SQRT2 / 9, ONE * 2 / 9, ZERO], [ZERO, ZERO, ONE / 3]]
    k = [[ZERO for _ in range(6)] for _ in range(6)]
    for i in range(3):
        for j in range(3):
            k[i][j], k[i][j + 3] = b[j][i], c[i][j]
            k[i + 3][j], k[i + 3][j + 3] = -a[i][j], -b[i][j]
    horizontal, vertical = [0, 1, 3, 4], [2, 5]
    kh = [[k[i][j] for j in horizontal] for i in horizontal]
    ph = [[value * 3 for value in row] for row in ci_matmul(kh, kh)]
    s = [[{} for _ in range(6)] for _ in range(6)]
    for i, row in enumerate(horizontal):
        for j, col in enumerate(horizontal):
            identity = ONE if i == j else ZERO
            s[row][col] = fq_clean({FZERO: identity - ph[i][j],
                                   (1, 0): (ph[i][j] + SQRT3 * kh[i][j]) / 2,
                                   (-1, 0): (ph[i][j] - SQRT3 * kh[i][j]) / 2})
    for i, row in enumerate(vertical):
        for j, col in enumerate(vertical):
            identity = ONE if i == j else ZERO
            s[row][col] = fq_clean({(0, 1): (identity - I * SQRT3 * k[row][col]) / 2,
                                   (0, -1): (identity + I * SQRT3 * k[row][col]) / 2})
    # Coefficientwise checks do not sample a time grid.
    for i in range(6):
        for j in range(6):
            check_zero(fq_evaluate(s[i][j], 0) - int(i == j), f"S(0)-I[{i},{j}]")
            symplectic = {}
            for axis in range(3):
                symplectic = fq_add(symplectic, fq_mul(s[axis][i], s[axis + 3][j]))
                symplectic = fq_add(symplectic,
                                    fq_scale(fq_mul(s[axis + 3][i], s[axis][j]), -1))
            target = int(j == i + 3) - int(i == j + 3)
            symplectic = fq_add(symplectic, fq_constant(-target))
            for frequency, value in symplectic.items():
                check_zero(value, f"S^TJS-J[{i},{j},{frequency}]")
    return a, b, c, k, s


def cubic_symbol():
    """The spec's commuting six-variable h3, without an inferred term."""
    x = [p_var(j, 6) for j in range(3)]
    z = [p_var(j + 3, 6) for j in range(3)]
    scale = lambda poly, value: p_scale(poly, fq_constant(value))
    x2 = p_sum(p_mul(y, y) for y in x)
    z2 = p_sum(p_mul(y, y) for y in z)
    t = p_add(scale(z[0], SQRT2 / SQRT3), scale(z[1], ONE / SQRT3))
    f1 = scale(x[0], -ONE * 3 / 2)
    f2 = p_add(scale(x2, -ONE * 3 / 4),
               scale(p_mul(x[0], x[0]), ONE * 21 / 8))
    f3 = p_add(scale(p_mul(x[0], x2), ONE * 21 / 8),
               scale(p_mul(x[0], p_mul(x[0], x[0])), -ONE * 77 / 16))
    l1 = p_sum([z[0], scale(x[0], SQRT6), scale(x[1], SQRT3)])
    l2 = p_sum(p_mul(x[j], z[j]) for j in range(3))
    kinetic = scale(p_mul(t, p_add(z2, scale(p_mul(t, t), -1))), -ONE / 18)
    flow = scale(p_sum([scale(f3, SQRT6), p_mul(l1, f2), p_mul(l2, f1)]),
                 -SQRT2 / SQRT3)
    return p_add(kinetic, flow)


def creation_polynomials(s):
    creation = []
    for axis in range(6):
        poly = {}
        for j in range(3):
            coefficient = fq_scale(fq_add(s[axis][j], fq_scale(s[axis][j + 3], I)),
                                   ONE / SQRT2)
            poly = p_add(poly, p_scale(p_var(j, 3), coefficient))
        creation.append(poly)
    sigma = [[{} for _ in range(6)] for _ in range(6)]
    for i in range(6):
        for j in range(6):
            total = {}
            for axis in range(6):
                # Ordinary transpose S S^T, not coefficient conjugation.
                total = fq_add(total, fq_mul(s[i][axis], s[j][axis]))
            sigma[i][j] = fq_scale(total, ONE / 2)
    return creation, sigma


def vacuum_sources(s):
    creation, sigma = creation_polynomials(s)
    scalar = {}
    for nu, coefficient in cubic_symbol().items():
        axes = [axis for axis, count in enumerate(nu) for _ in range(count)]
        require(len(axes) == 3, "h3 is not homogeneous cubic")
        a, b, c = axes
        # Weyl/Wick degree-three term plus its three single contractions.
        term = p_mul(creation[a], p_mul(creation[b], creation[c]))
        term = p_add(term, p_scale(creation[c], sigma[a][b]))
        term = p_add(term, p_scale(creation[b], sigma[a][c]))
        term = p_add(term, p_scale(creation[a], sigma[b][c]))
        scalar = p_add(scalar, p_scale(term, coefficient))
    # Literal componentwise h1_linear in the final cleared specification.
    spin_rows = [[ZERO, ZERO, -SQRT3 / 6, ZERO, ZERO, ZERO],
                 [ZERO, ZERO, SQRT6 / 6, ZERO, ZERO, -ONE / 6],
                 [-SQRT3 / 4, -SQRT6 / 12, ZERO, -SQRT2 / 36, ONE / 18, ZERO]]
    spin = [p_sum(p_scale(creation[j], fq_constant(row[j])) for j in range(6))
            for row in spin_rows]
    require(set(scalar).issubset(set(MODES)), "Unexpected scalar creation degree")
    for poly in [scalar] + spin:
        require(all(key in FREQUENCIES for values in poly.values() for key in values),
                "Source has a frequency outside the finite specified set")
    return creation, scalar, spin


def pauli_vector(vector, chi):
    x, y, z = vector
    return [z * chi[0] + (x - I * y) * chi[1],
            (x + I * y) * chi[0] - z * chi[1]]


def spin_source_mode(spin, nu, chi):
    x, y, z = (poly.get(nu, {}) for poly in spin)
    return [fq_add(fq_scale(z, chi[0]),
                   fq_scale(fq_add(x, fq_scale(y, -I)), chi[1])),
            fq_add(fq_scale(fq_add(x, fq_scale(y, I)), chi[0]),
                   fq_scale(z, -chi[1]))]


def amplitude(source, initial):
    constant = initial
    coefficients = {}
    linear = ZERO
    for frequency, value in source.items():
        if frequency == FZERO:
            linear = -I * SQRT3 * value
            constant = constant + linear * PI
        else:
            coefficient = -I * SQRT3 * value / CI(*frequency)
            coefficients[frequency] = coefficient
            constant = constant - coefficient * endpoint_exp(frequency, -1)
    coefficients[FZERO] = coefficients.get(FZERO, ZERO) + constant
    return {0: fq_clean(coefficients), 1: fq_constant(linear)}


def fpoly_evaluate(poly, endpoint):
    u = PI * endpoint
    answer = ZERO
    power = ONE
    for j in range(max(poly, default=0) + 1):
        answer = answer + power * fq_evaluate(poly.get(j, {}), endpoint)
        power = power * u
    return answer


def check_amplitude(source, initial, poly, label):
    check_zero(fpoly_evaluate(poly, -1) - initial, label + ":initial")
    derivative = {}
    for frequency, coefficient in poly[0].items():
        derivative[frequency] = CI(*frequency) * coefficient
    derivative = fq_add(derivative, poly[1])
    residual = fq_add(fq_scale(derivative, I / SQRT3), fq_scale(source, -1))
    for frequency, value in residual.items():
        check_zero(value, label + ":differential:" + str(frequency))


J_CACHE = {}


def j_integral(frequency, power):
    key = (frequency, power)
    if key in J_CACHE:
        return J_CACHE[key]
    require(power in (0, 1, 2), "Unexpected exponential-integral power")
    if frequency == FZERO:
        value = [PI * 2, ZERO, PI * PI * PI * 2 / 3][power]
    else:
        z = CI(*frequency)
        plus, minus = endpoint_exp(frequency, 1), endpoint_exp(frequency, -1)
        difference, total = plus - minus, plus + minus
        if power == 0:
            value = difference / z
        elif power == 1:
            value = (PI * z * total - difference) / (z * z)
        else:
            value = (PI * PI * z * z * difference - PI * z * total * 2
                     + difference * 2) / (z * z * z)
    J_CACHE[key] = value
    return value


def square_integral(poly):
    total = ZERO
    for j, left in poly.items():
        for k, right in poly.items():
            for (m, n), x in left.items():
                for (p, q), y in right.items():
                    total = total + x.conjugate() * y * j_integral((m + p, q - n), j + k)
    return total * SQRT3


def norm_squared(values):
    return sum((value.conjugate() * value for value in values), ZERO)


def midpoint_checks(scalar, spin):
    expected_scalar = {
        (1, 0, 0): -ONE * 45 / (SQRT2 * 16) + I * 73 / (SQRT3 * 144),
        (0, 1, 0): ONE * 9 / 16 - I / (SQRT6 * 18),
        (0, 0, 1): ZERO,
        (3, 0, 0): SQRT2 * 5 / 32 - I * 77 / (SQRT3 * 432)}
    for nu, expected in expected_scalar.items():
        check_zero(fq_evaluate(scalar.get(nu, {}), 0) - expected,
                   "MP34:unnormalized_creation:" + str(nu))
    expected_spin = [
        [ZERO, ZERO, -SQRT6 / 8 - I / 36],
        [ZERO, ZERO, -SQRT3 / 12 + I / (SQRT2 * 18)],
        [-SQRT6 / 12, SQRT3 / 6 - I / (SQRT2 * 6), ZERO]]
    for mode in range(3):
        nu = tuple(int(j == mode) for j in range(3))
        for axis in range(3):
            check_zero(fq_evaluate(spin[axis].get(nu, {}), 0) - expected_spin[mode][axis],
                       f"MP37:spin_creation[{mode},{axis}]")


def floor_scaled_endpoint(raw, places, upper=False):
    """Audited exact dyadic-to-outward-decimal serializer from input run."""
    sign, mantissa, exponent, bitcount = raw
    require(bitcount >= 0, "Cannot serialize a nonfinite endpoint")
    numerator = (-mantissa if sign else mantissa) * 10 ** places
    if exponent >= 0:
        return numerator << exponent
    denominator = 1 << (-exponent)
    return -((-numerator) // denominator) if upper else numerator // denominator


def decimal_scaled(number, places=110):
    sign = "-" if number < 0 else ""
    digits = str(abs(number)).rjust(places + 1, "0")
    return sign + digits[:-places] + "." + digits[-places:]


def record_interval(value):
    global MAX_WIDTH
    width = (value.b - value.a).b
    require(bool(width <= WIDTH_LIMIT), "Reported interval exceeds raw width limit")
    if bool(width > MAX_WIDTH):
        MAX_WIDTH = width
    lower, upper = value._mpi_
    return {"lower": decimal_scaled(floor_scaled_endpoint(lower, 110)),
            "upper": decimal_scaled(floor_scaled_endpoint(upper, 110, upper=True))}


def record_ci(value):
    return {"real": record_interval(value.r), "imag": record_interval(value.i)}


def record_fq(values):
    require(set(values).issubset(set(FREQUENCIES)), "Unlisted output frequency")
    return [{"frequency": list(frequency),
             "value": record_ci(values.get(frequency, ZERO)),
             "exact_zero": values.get(frequency, ZERO).exact_zero()}
            for frequency in FREQUENCIES]


def record_fpoly(poly):
    require(set(poly[1]).issubset({FZERO}), "Nonconstant coefficient of u")
    return {"exponential": record_fq(poly[0]),
            "linear_u": record_ci(poly[1].get(FZERO, ZERO))}


def carrier_result(eta, creation, scalar, spin):
    chi = [ONE / SQRT2, (SQRT2 + I) * eta / (SQRT2 * SQRT3)]
    check_zero(norm_squared(chi) - 1, f"carrier{eta}:normalization")
    n = [SQRT2 / SQRT3, ONE / SQRT3, ZERO]
    polarization = pauli_vector(n, chi)
    for j in range(2):
        check_zero(polarization[j] - chi[j] * eta, f"carrier{eta}:eigenvector{j}")
    longitudinal = p_sum(p_scale(creation[j + 3], fq_constant(n[j])) for j in range(3))
    transverse = [p_add(creation[j + 3], p_scale(longitudinal, fq_constant(-n[j])))
                  for j in range(3)]
    initial_spin = [p_scale(poly, fq_constant(ONE * eta / 6)) for poly in transverse]
    integrals = {key: ZERO for key in ["source_norm2", "source_A_quarter",
                                      "amplitude_norm2", "amplitude_A_quarter"]}
    initial_weighted, midpoint_spin_norm = ZERO, ZERO
    midpoint_cross = ZERO
    records = []
    for nu in MODES:
        factorial = math.prod(math.factorial(j) for j in nu)
        normalization = CI(iv.sqrt(factorial))
        weighted = 4 ** sum(nu)
        scalar_mode = fq_scale(scalar.get(nu, {}), normalization)
        spin_mode = [fq_scale(x, normalization) for x in spin_source_mode(spin, nu, chi)]
        source = [fq_add(fq_scale(scalar_mode, chi[j]), spin_mode[j]) for j in range(2)]
        input_mode = spin_source_mode(initial_spin, nu, chi)
        initial = [fq_evaluate(fq_scale(x, normalization), -1) for x in input_mode]
        amplitudes = [amplitude(source[j], initial[j]) for j in range(2)]
        for j in range(2):
            check_amplitude(source[j], initial[j], amplitudes[j], f"eta{eta}:{nu}:spin{j}")
            source_integral = square_integral({0: source[j]})
            amplitude_integral = square_integral(amplitudes[j])
            integrals["source_norm2"] = integrals["source_norm2"] + source_integral
            integrals["source_A_quarter"] = integrals["source_A_quarter"] + source_integral * weighted
            integrals["amplitude_norm2"] = integrals["amplitude_norm2"] + amplitude_integral
            integrals["amplitude_A_quarter"] = integrals["amplitude_A_quarter"] + amplitude_integral * weighted
        initial_weighted = initial_weighted + norm_squared(initial) * weighted
        midpoint_spin = [fq_evaluate(x, 0) for x in spin_mode]
        midpoint_scalar = fq_evaluate(scalar_mode, 0)
        midpoint_spin_norm = midpoint_spin_norm + norm_squared(midpoint_spin)
        midpoint_cross = midpoint_cross + sum(
            ((midpoint_scalar * chi[j]).conjugate() * midpoint_spin[j] for j in range(2)), ZERO)
        records.append({"nu": list(nu), "hermite_factorial": factorial,
                        "source": [record_fq(x) for x in source],
                        "initial_amplitude": [record_ci(x) for x in initial],
                        "amplitude": [record_fpoly(x) for x in amplitudes],
                        "amplitude_endpoints": {
                            str(endpoint): [record_ci(fpoly_evaluate(x, endpoint)) for x in amplitudes]
                            for endpoint in [-1, 0, 1]}})
    check_zero(midpoint_spin_norm - ONE * 221 / 864, f"eta{eta}:MP37_spin_norm")
    check_zero(CI(midpoint_cross.r), f"eta{eta}:midpoint_scalar_spin_real_overlap")
    for key, value in integrals.items():
        check_zero(CI(0, value.i), f"eta{eta}:{key}:imaginary")
        require(bool(value.r.a >= 0), f"eta{eta}:{key}:nonnegative enclosure failed")
    total_time = PI * SQRT3 * 2
    uniform_bound = iv.sqrt(initial_weighted.r) + iv.sqrt(total_time.r * integrals["source_A_quarter"].r)
    return {"eta": eta, "carrier_spinor": [record_ci(x) for x in chi],
            "modes": records, "integrals": {key: record_ci(value) for key, value in integrals.items()},
            "initial_A_quarter": record_ci(initial_weighted),
            "uniform_first_coefficient_A_quarter_sqrt_bound": record_interval(uniform_bound),
            "midpoint_spin_source_norm2": record_ci(midpoint_spin_norm)}


def evaluate():
    a, b, c, k, s = native_inputs()
    creation, scalar, spin = vacuum_sources(s)
    midpoint_checks(scalar, spin)
    carriers = [carrier_result(eta, creation, scalar, spin) for eta in [1, -1]]
    require(len(MODES) == 13 and len(FREQUENCIES) == 25, "Finite array counts changed")
    matrix = lambda values: [[record_ci(value) for value in row] for row in values]
    result = {
        "status": "passed",
        "scientific_scope": "First finite two-spin recurrence coefficient only; no propagated error, return root, or MP8 coefficient.",
        "parameters": {
            "epsilon_exact": "2^-24", "iv_decimal_digits": iv.dps,
            "u_interval": ["-pi", "pi"], "tau_relation": "tau=sqrt(3)*(u+pi)",
            "endpoint_labels": {"-1": "u=-pi", "0": "u=0", "1": "u=pi"},
            "weighted_norm_rho_exact": "1/4", "carrier_mixture_weights": ["1/2", "1/2"],
            "hermite_indices": [list(nu) for nu in MODES],
            "integer_frequencies": [list(key) for key in FREQUENCIES],
            "normalization": "|nu>=(a_dagger)^nu|0>/sqrt(nu!)",
            "input_normalizer_scope": "Exact accepted Z retained in full preparation; its next-order effect is outside this first coefficient.",
            "analytic_cutoff_L2_allowance": "1e-35",
            "analytic_cutoff_scope": "Includes first-source propagation and initial-a1 omitted-cutoff discrepancy only; no weighted norm, graph, launch, spacing or higher-order remainder bound.",
            "norm_scope": "All exponential arrays, norms and weighted norms are the finite polynomial coefficient, not the exact cut state's propagated amplitude tail.",
            "next_source_scope": "Full matrix V4 and both bands/native response/root remain required.",
            "raw_width_limit": "1e-25", "outward_decimal_places": 110},
        "literal_matrices": {"A": matrix(a), "B": matrix(b), "C": matrix(c), "K": matrix(k)},
        "S_exact_frequency_representation": [[record_fq(value) for value in row] for row in s],
        "carriers": carriers,
        "checks": [{"description": description, "residual": record_ci(value)}
                   for description, value in CHECKS]}
    result["maximum_reported_raw_width"] = record_interval(MAX_WIDTH)
    result["checked_identity_count"] = len(CHECKS)
    result["cached_exact_integral_count"] = len(J_CACHE)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.getpriority(os.PRIO_PROCESS, 0) == 10, "Expected nice10 box execution")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "GPU visibility must be disabled")
    result = evaluate()
    result["producer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["runtime_seconds"] = time.monotonic() - START
    result["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["cpu_limit_seconds"] = 180
    result["memory_limit_bytes"] = MEMORY_BYTES
    result["nice"] = os.getpriority(os.PRIO_PROCESS, 0)
    result["mpmath_version"] = mpmath.__version__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    require(temporary.stat().st_size < 20 * 1024 * 1024, "Result exceeds lean-file cap")
    temporary.replace(args.output)
    signal.alarm(0)
    print(json.dumps({key: result[key] for key in ["status", "runtime_seconds", "peak_rss_kib",
                                                  "producer_sha256", "checked_identity_count"]}))


if __name__ == "__main__":
    main()
