#!/usr/bin/env python3
"""Validated T10 arithmetic for the declared order-four polar law.

Run only on the remote qi2 host.  The root-finder's NumPy seeds are not
trusted: exact rational sign brackets certify every Gauss node.  Newton
arithmetic uses mpmath intervals; dense tensor arithmetic uses directed
binary64 intervals with verified square-root brackets.  The
analytical quadrature and tail bounds are those in the reviewed T10 source.
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from threading import Lock
from uuid import UUID

from mpmath import iv
import numpy as np
from numpy.polynomial.legendre import leggauss


def hold_cuda_slot(expected_uuid: str):
    """One small allocation in this job, so its assigned GPU is visible.

    Uses NVIDIA's documented device/context/memory Driver API.  UUID matching
    works whether the driver enumerates the visible subset or all devices.
    No GPU kernel or extra process is launched.
    """
    driver = ctypes.CDLL("libcuda.so.1")
    signatures = {
        "cuInit": [ctypes.c_uint],
        "cuDeviceGetCount": [ctypes.POINTER(ctypes.c_int)],
        "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
        "cuDeviceGetUuid": [ctypes.c_void_p, ctypes.c_int],
        "cuCtxCreate_v2": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int],
        "cuMemAlloc_v2": [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t],
        "cuMemFree_v2": [ctypes.c_ulonglong],
        "cuCtxDestroy_v2": [ctypes.c_void_p],
    }
    for name, arguments in signatures.items():
        function = getattr(driver, name)
        function.argtypes, function.restype = arguments, ctypes.c_int

    def call(name, *arguments):
        error = getattr(driver, name)(*arguments)
        if error:
            raise RuntimeError(f"CUDA Driver API {name} failed with code {error}")

    call("cuInit", 0)
    count = ctypes.c_int()
    call("cuDeviceGetCount", ctypes.byref(count))
    selected = None
    for ordinal in range(count.value):
        device = ctypes.c_int()
        identifier = (ctypes.c_ubyte * 16)()
        call("cuDeviceGet", ctypes.byref(device), ordinal)
        call("cuDeviceGetUuid", ctypes.byref(identifier), device.value)
        if "GPU-" + str(UUID(bytes=bytes(identifier))) == expected_uuid:
            selected = device.value
            break
    if selected is None:
        raise RuntimeError("The assigned GPU UUID is not available to this process")
    context, memory = ctypes.c_void_p(), ctypes.c_ulonglong()
    call("cuCtxCreate_v2", ctypes.byref(context), 0, selected)
    allocation_bytes = 1 << 20
    call("cuMemAlloc_v2", ctypes.byref(memory), allocation_bytes)

    def release():
        call("cuMemFree_v2", memory)
        call("cuCtxDestroy_v2", context)

    atexit.register(release)
    return {"assigned_gpu_uuid": expected_uuid, "cuda_allocation_bytes": allocation_bytes,
            "cuda_context_in_this_single_job_process": True}


def interval(value: Fraction | int | str):
    if isinstance(value, Fraction):
        return iv.mpf(value.numerator) / value.denominator
    return iv.mpf(value)


def hull(lower: Fraction, upper: Fraction):
    return iv.mpf([interval(lower).a, interval(upper).b])


def endpoints(value) -> tuple[Fraction, Fraction]:
    """Read the exact dyadic endpoints; never round via an ordinary float."""
    result = []
    for sign, mantissa, exponent, bits in value._mpi_:
        if bits < 0:
            raise ArithmeticError("Non-finite interval endpoint")
        item = Fraction((-1 if sign else 1) * mantissa)
        item = item * (1 << exponent) if exponent >= 0 else item / (1 << -exponent)
        result.append(item)
    return tuple(result)


def decimal_endpoint(value: Fraction, digits: int, upper: bool) -> str:
    scale = 10 ** digits
    quotient, remainder = divmod(value.numerator * scale, value.denominator)
    if upper and remainder:
        quotient += 1
    sign = "-" if quotient < 0 else ""
    whole, part = divmod(abs(quotient), scale)
    return f"{sign}{whole}.{part:0{digits}d}"


def enclosure(value, digits: int = 24) -> dict[str, str]:
    lower, upper = endpoints(value)
    return {"lower": decimal_endpoint(lower, digits, False),
            "upper": decimal_endpoint(upper, digits, True)}


def certified_iv_sqrt(value):
    lower, upper = endpoints(value)
    if not 0 <= lower <= upper:
        raise ArithmeticError("Interval sqrt requires finite ordered nonnegative input")
    result = iv.sqrt(value)
    root_lower, root_upper = endpoints(result)
    if not (0 <= root_lower <= root_upper and root_lower ** 2 <= lower and root_upper ** 2 >= upper):
        raise ArithmeticError("mpmath sqrt endpoints fail exact rational square inequalities")
    return result


def constant_enclosures():
    def arctangent(argument: Fraction, count: int):
        total = sum(((-1) ** index * argument ** (2 * index + 1) / (2 * index + 1)
                     for index in range(count)), Fraction(0))
        next_total = total + (-1) ** count * argument ** (2 * count + 1) / (2 * count + 1)
        return hull(min(total, next_total), max(total, next_total))

    def twice_atanh(argument: Fraction, count: int):
        total = 2 * sum((argument ** (2 * index + 1) / (2 * index + 1)
                         for index in range(count)), Fraction(0))
        tail = 2 * argument ** (2 * count + 1) / ((2 * count + 1) * (1 - argument ** 2))
        return hull(total, total + tail)

    pi = 16 * arctangent(Fraction(1, 5), 100) - 4 * arctangent(Fraction(1, 239), 40)
    log_two = twice_atanh(Fraction(1, 3), 120)
    log_one_and_quarter = twice_atanh(Fraction(1, 9), 70)
    root_bits = 320
    root_scale = 1 << root_bits
    integer_root = math.isqrt(2 * root_scale ** 2)
    root_lower = Fraction(integer_root, root_scale)
    root_upper = Fraction(integer_root + 1, root_scale)
    if not root_lower ** 2 < 2 < root_upper ** 2:
        raise ArithmeticError("Exact sqrt2 constant bracket failed")
    sqrt_two = hull(root_lower, root_upper)
    return pi, log_two, sqrt_two, 5 * log_two + log_one_and_quarter, 9 * log_two + 3 * log_one_and_quarter


def outward_float(value: Fraction, upper: bool) -> float:
    rounded = float(value)
    represented = Fraction.from_float(rounded)
    if (upper and represented < value) or (not upper and represented > value):
        rounded = math.nextafter(rounded, math.inf if upper else -math.inf)
    if (upper and Fraction.from_float(rounded) < value
            or not upper and Fraction.from_float(rounded) > value):
        raise ArithmeticError("Binary64 endpoint conversion was not outward")
    return rounded


class BinaryInterval:
    """Vectorized directed binary64 intervals; no BLAS or vendor reductions.

    Arithmetic contract: IEEE binary64 round-to-nearest for +,-,*,/;
    each primitive result is widened by nextafter.  The sqrt ufunc is
    only a seed: outward squared-endpoint inequalities certify its result.
    A minimum/maximum selects endpoints and performs no numerical sum.
    Underflow is permitted and widened; overflow and invalid operations fail.
    """

    __array_priority__ = 1000

    def __init__(self, lower, upper=None):
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(lower if upper is None else upper, dtype=np.float64)

    @classmethod
    def rational(cls, value: Fraction | int):
        value = Fraction(value)
        return cls(outward_float(value, False), outward_float(value, True))

    @classmethod
    def from_iv(cls, value):
        lower, upper = endpoints(value)
        return cls(outward_float(lower, False), outward_float(upper, True))

    @classmethod
    def vector(cls, values):
        bounds = [endpoints(value) for value in values]
        return cls([outward_float(pair[0], False) for pair in bounds],
                   [outward_float(pair[1], True) for pair in bounds])

    @staticmethod
    def coerce(value):
        if isinstance(value, BinaryInterval):
            return value
        if isinstance(value, (int, Fraction)):
            return BinaryInterval.rational(value)
        raise TypeError("Only explicit interval or exact rational operands are accepted")

    def __getitem__(self, item):
        return BinaryInterval(self.lower[item], self.upper[item])

    def __neg__(self):
        return BinaryInterval(-self.upper, -self.lower)

    def __add__(self, other):
        other = self.coerce(other)
        return BinaryInterval(np.nextafter(self.lower + other.lower, -np.inf),
                              np.nextafter(self.upper + other.upper, np.inf))

    __radd__ = __add__

    def __sub__(self, other):
        return self + -self.coerce(other)

    def __rsub__(self, other):
        return self.coerce(other) - self

    def __mul__(self, other):
        other = self.coerce(other)
        products = [self.lower * other.lower, self.lower * other.upper,
                    self.upper * other.lower, self.upper * other.upper]
        return BinaryInterval(np.nextafter(np.minimum.reduce(products), -np.inf),
                              np.nextafter(np.maximum.reduce(products), np.inf))

    __rmul__ = __mul__

    def reciprocal(self):
        if np.any((self.lower <= 0) & (self.upper >= 0)):
            raise ArithmeticError("Interval denominator contains zero")
        return BinaryInterval(np.nextafter(1.0 / self.upper, -np.inf),
                              np.nextafter(1.0 / self.lower, np.inf))

    def __truediv__(self, other):
        return self * self.coerce(other).reciprocal()

    def __rtruediv__(self, other):
        return self.coerce(other) * self.reciprocal()

    def square(self):
        left, right = self.lower * self.lower, self.upper * self.upper
        crosses_zero = (self.lower <= 0) & (self.upper >= 0)
        lower = np.where(crosses_zero, 0.0,
                         np.maximum(0.0, np.nextafter(np.minimum(left, right), -np.inf)))
        upper = np.nextafter(np.maximum(left, right), np.inf)
        return BinaryInterval(lower, upper)

    def __pow__(self, power):
        if not isinstance(power, int) or power < 0:
            raise TypeError("Only nonnegative integer interval powers are supported")
        result = self.rational(1)
        factor = self
        while power:
            if power & 1:
                result = result * factor
            power >>= 1
            if power:
                factor = factor.square()
        return result

    def sqrt(self):
        if (np.any(~np.isfinite(self.lower)) or np.any(~np.isfinite(self.upper))
                or np.any(self.lower > self.upper) or np.any(self.lower < 0)):
            raise ArithmeticError("Square-root input must be finite, ordered and nonnegative")
        if np.any((self.upper > 0) & (self.upper < np.finfo(np.float64).tiny)):
            raise ArithmeticError("Sqrt verification requires zero or normal positive input")
        lower = np.maximum(0.0, np.nextafter(np.sqrt(self.lower), -np.inf))
        upper = np.where(self.upper == 0, 0.0, np.nextafter(np.sqrt(self.upper), np.inf))
        if (np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper))
                or np.any(lower < 0) or np.any(lower > upper)):
            raise ArithmeticError("Square-root seed must be finite, ordered and nonnegative")
        for _ in range(8):
            bad_lower = (lower != 0) & (np.nextafter(lower * lower, np.inf) > self.lower)
            bad_upper = (self.upper != 0) & (np.nextafter(upper * upper, -np.inf) < self.upper)
            if not np.any(bad_lower | bad_upper):
                return BinaryInterval(lower, upper)
            lower = np.where(bad_lower, np.maximum(0.0, np.nextafter(lower, -np.inf)), lower)
            upper = np.where(bad_upper, np.nextafter(upper, np.inf), upper)
        raise ArithmeticError("Sqrt seed could not be certified by squared endpoints")

    def sum(self):
        lower, upper = self.lower.ravel(), self.upper.ravel()
        while lower.size > 1:
            pairs = lower.size // 2
            next_lower = np.nextafter(lower[:2 * pairs:2] + lower[1:2 * pairs:2], -np.inf)
            next_upper = np.nextafter(upper[:2 * pairs:2] + upper[1:2 * pairs:2], np.inf)
            if lower.size % 2:
                next_lower = np.concatenate((next_lower, lower[-1:]))
                next_upper = np.concatenate((next_upper, upper[-1:]))
            lower, upper = next_lower, next_upper
        return BinaryInterval(lower[0], upper[0])

    def to_iv(self):
        if self.lower.size != 1 or self.upper.size != 1:
            raise ValueError("Only a scalar interval can convert to mpmath.iv")
        return hull(Fraction.from_float(float(self.lower)), Fraction.from_float(float(self.upper)))


def native_kernel(p, q, transverse, mass_squared):
    """T22i removable divided difference, including the p=q diagonal."""
    p_squared, q_squared = p.square(), q.square()
    transverse_squared = transverse.square()
    t_p, t_q = p_squared + transverse_squared, q_squared + transverse_squared
    r_p_squared, r_q_squared = mass_squared + t_p ** 4, mass_squared + t_q ** 4
    r_p, r_q = r_p_squared.sqrt(), r_q_squared.sqrt()
    e_p, e_q = (r_p_squared + t_p).sqrt(), (r_q_squared + t_q).sqrt()
    polynomial = (t_p + t_q) * (t_p.square() + t_q.square())
    d_r = (p + q) * polynomial / (r_p + r_q)
    d_e = (p + q) * (1 + polynomial) / (e_p + e_q)
    denominator = e_p * e_q
    real_difference = (d_r * e_q - r_q * d_e) / denominator
    normal_difference = (e_q - q * d_e) / denominator
    transverse_difference = -(transverse * d_e) / denominator
    return real_difference.square() + normal_difference.square() + transverse_difference.square()


def native_kernel_iv(p, q, transverse, mass_squared):
    t_p, t_q = p ** 2 + transverse ** 2, q ** 2 + transverse ** 2
    r_p, r_q = certified_iv_sqrt(mass_squared + t_p ** 4), certified_iv_sqrt(mass_squared + t_q ** 4)
    e_p = certified_iv_sqrt(mass_squared + t_p + t_p ** 4)
    e_q = certified_iv_sqrt(mass_squared + t_q + t_q ** 4)
    polynomial = (t_p + t_q) * (t_p ** 2 + t_q ** 2)
    d_r = (p + q) * polynomial / (r_p + r_q)
    d_e = (p + q) * (1 + polynomial) / (e_p + e_q)
    denominator = e_p * e_q
    components = ((d_r * e_q - r_q * d_e) / denominator,
                  (e_q - q * d_e) / denominator, -transverse * d_e / denominator)
    return sum((component ** 2 for component in components), interval(0))


def verify_binary64_contract():
    np.seterr(over="raise", invalid="raise", divide="raise", under="ignore")
    library = ctypes.CDLL("libm.so.6")
    library.fegetround.restype = ctypes.c_int
    if library.fegetround() != 0:
        raise ArithmeticError("Host rounding mode is not FE_TONEAREST")
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError("Numerical execution requires niceness at least10")
    if len(os.listdir("/proc/self/task")) > 16:
        raise RuntimeError("Numerical process exceeds the16-thread limit")
    one = np.float64(1)
    half_ulp = np.float64(float.fromhex("0x1p-53"))
    if one + half_ulp != one or one + 3 * half_ulp != np.float64(float.fromhex("0x1.0000000000002p+0")):
        raise ArithmeticError("Binary64 tie rounding check failed")
    smallest = np.nextafter(np.float64(0), np.float64(1))
    minimum_normal = np.float64(float.fromhex("0x1p-1022"))
    subnormal = minimum_normal / np.float64(2)
    if (smallest != np.float64(float.fromhex("0x0.0000000000001p-1022"))
            or subnormal != np.float64(float.fromhex("0x0.8000000000000p-1022"))
            or subnormal * np.float64(2) != minimum_normal
            or smallest * one != smallest
            or smallest + smallest != np.float64(float.fromhex("0x0.0000000000002p-1022"))):
        raise ArithmeticError("Gradual underflow/FTZ/DAZ check failed")
    cases = ((Fraction(0), Fraction(0), Fraction(1, 10), Fraction(1, 10 ** 6)),
             (Fraction(1, 10), Fraction(1, 10), Fraction(1, 10), Fraction(1, 10 ** 6)),
             (Fraction(1), Fraction(1), Fraction(1), Fraction(1, 10 ** 6)),
             (Fraction(96), Fraction(96), Fraction(4), Fraction(1)),
             (Fraction(96), Fraction(-96), Fraction(1, 10), Fraction(1, 2)),
             (Fraction(2), Fraction(-1, 10), Fraction(1, 10), Fraction(1, 10 ** 6)))
    for values in cases:
        binary = native_kernel(*(BinaryInterval.rational(value) for value in values)).to_iv()
        reference = native_kernel_iv(*(interval(value) for value in values))
        binary_lower, binary_upper = endpoints(binary)
        reference_lower, reference_upper = endpoints(reference)
        if binary_lower > reference_lower or binary_upper < reference_upper:
            raise ArithmeticError("Binary64 native-kernel interval did not contain the interval reference")
    return {"FE_TONEAREST": True, "ties_to_even_checks": True,
            "gradual_underflow_and_FTZ_DAZ_off_checks": True,
            "sqrt": "Seed only; every output certified by outward squared-endpoint inequalities",
            "basic_operations": "IEEE binary64; one nextafter outward per +,-,*,/ primitive",
            "reduction": "Pairwise additions, each widened outward; no BLAS or arithmetic vendor reductions",
            "kernel_reference_cases_enclosed": len(cases),
            "reference_cases_are_diagnostics_not_a_replacement_for_the_interval_operation_proof": True}


def legendre_exact(order: int, point: Fraction) -> Fraction:
    before, current = Fraction(1), point
    if order == 0:
        return before
    for degree in range(2, order + 1):
        before, current = current, ((2 * degree - 1) * point * current
                                   - (degree - 1) * before) / degree
    return current


def legendre_interval(order: int, point):
    before, current = interval(1), point
    if order == 0:
        return before, interval(0)
    for degree in range(2, order + 1):
        before, current = current, ((2 * degree - 1) * point * current
                                   - (degree - 1) * before) / degree
    derivative = order * (before - point * current) / (1 - point * point)
    return current, derivative


def gauss_enclosures(order: int, root_bits: int) -> tuple[list, list, dict]:
    seeds, _ = leggauss(order)
    initial_radius = Fraction(1, 1 << 20)
    brackets = []
    for seed in seeds:
        center = Fraction.from_float(float(seed))
        radius = initial_radius
        for _ in range(20):
            left, right = center - radius, center + radius
            f_left, f_right = legendre_exact(order, left), legendre_exact(order, right)
            if -1 < left < right < 1 and f_left * f_right < 0:
                break
            radius *= 2
        else:
            raise ArithmeticError("Could not certify a root bracket")
        brackets.append([left, right, f_left, f_right])
    if any(left[1] >= right[0] for left, right in zip(brackets, brackets[1:])):
        raise ArithmeticError("Root brackets are not disjoint")
    if len(brackets) != order:
        raise ArithmeticError("The sign brackets do not exhaust the polynomial degree")
    # There are order disjoint sign-changing brackets for a polynomial of
    # degree order, so each bracket contains exactly one root and all roots
    # are present.  Bisection below is entirely exact rational arithmetic.
    tolerance = Fraction(1, 1 << root_bits)
    nodes, weights = [], []
    max_width = Fraction(0)
    for left, right, f_left, f_right in brackets:
        while right - left > tolerance:
            middle = (left + right) / 2
            f_middle = legendre_exact(order, middle)
            if f_middle == 0:
                left = right = middle
                break
            if f_left * f_middle < 0:
                right, f_right = middle, f_middle
            else:
                left, f_left = middle, f_middle
        node = hull(left, right)
        _, derivative = legendre_interval(order, node)
        if endpoints(derivative)[0] <= 0 <= endpoints(derivative)[1]:
            raise ArithmeticError("Gauss derivative enclosure contains zero")
        weight = 2 / ((1 - node * node) * derivative * derivative)
        if endpoints(weight)[0] <= 0:
            raise ArithmeticError("Gauss weight is not strictly positive")
        nodes.append(node)
        weights.append(weight)
        max_width = max(max_width, right - left)
    if order % 2 == 0:
        # Exact Legendre parity gives symmetric roots and equal weights.
        # Use the certified positive brackets to enclose both members;
        # this permits exact reflection reduction of the tensor rule.
        for index in range(order // 2):
            nodes[index] = -nodes[order - 1 - index]
            weights[index] = weights[order - 1 - index]
    weight_sum = sum(weights, interval(0))
    if not (endpoints(weight_sum)[0] <= 2 <= endpoints(weight_sum)[1]):
        raise ArithmeticError("Gauss weights do not enclose total measure two")
    return nodes, weights, {
        "order": order, "all_roots_certified_by_disjoint_exact_sign_brackets": True,
        "maximum_root_bracket_width": decimal_endpoint(max_width, 40, True),
        "weight_sum": enclosure(weight_sum),
    }


def mapped_gauss(lower: Fraction, upper: Fraction, nodes: list, weights: list):
    midpoint = interval((lower + upper) / 2)
    half_width = interval((upper - lower) / 2)
    return [midpoint + half_width * node for node in nodes], [half_width * w for w in weights]


def geometric_panels(lower: Fraction, upper: Fraction, ratio: Fraction = Fraction(5, 4)):
    first = lower
    panels = []
    while lower < upper:
        end = min(upper, lower * ratio)
        panels.append((lower, end))
        lower = end
    if (not panels or panels[0][0] != first or panels[-1][1] != upper
            or any(left[1] != right[0] for left, right in zip(panels, panels[1:]))
            or any(not 0 < left < right <= ratio * left for left, right in panels)):
        raise ArithmeticError("Exact rational panels fail the tiling/ratio requirements")
    return panels


def massless_radial(point):
    # This analytic form is regular at the origin and avoids 0/0.
    light = point * (1 + interval(Fraction(16, 3)) * point ** 6) / certified_iv_sqrt(1 + point ** 6)
    grouped = light
    for mass_squared, multiplicity in ((Fraction(1, 2), -2), (Fraction(1), 1)):
        real_squared = interval(mass_squared) + point ** 8
        massive = point ** 2 * (1 + interval(Fraction(16, 3)) * point ** 14 / real_squared)
        massive /= certified_iv_sqrt(real_squared + point ** 2)
        grouped += multiplicity * massive
    return 12 / (PI ** 2) * grouped


def newton_ellipse_bound(lower: Fraction, upper: Fraction):
    maximum = interval(lower + Fraction(9, 8) * (upper - lower))
    if lower == 0:
        light = maximum * (1 + interval(Fraction(16, 3)) * maximum ** 6) / certified_iv_sqrt(1 - maximum ** 6)
        massive = maximum ** 2 * (1 + interval(Fraction(16, 3)) * maximum ** 14
                                 / (interval(Fraction(1, 2)) - maximum ** 8))
        massive /= certified_iv_sqrt(interval(Fraction(1, 2)) - maximum ** 2 - maximum ** 8)
        return 12 / (PI ** 2) * (light + 3 * massive)
    return (4 * 12 / (PI ** 2) * SQRT2 * maximum
            * (1 + interval(Fraction(32, 3)) * maximum ** 6))


def newton_certificate(nodes: list, weights: list, order: int):
    panels = [(Fraction(0), Fraction(1, 10))]
    panels += geometric_panels(Fraction(1, 10), Fraction(4))
    quadrature = interval(0)
    gauss_error = interval(0)
    for lower, upper in panels:
        points, measures = mapped_gauss(lower, upper, nodes, weights)
        quadrature += sum((w * massless_radial(p) for p, w in zip(points, measures)), interval(0))
        bound = newton_ellipse_bound(lower, upper)
        gauss_error += 8 * interval(upper - lower) * bound / (1 << (2 * order))
    finite_lower = endpoints(quadrature)[0] - endpoints(gauss_error)[1]
    finite_upper = endpoints(quadrature)[1] + endpoints(gauss_error)[1]
    limit = interval(4)
    tail = 12 / (PI ** 2) * (interval(Fraction(3, 272)) * limit ** -17
                               + interval(Fraction(5, 11)) * limit ** -11)
    massless = hull(finite_lower, finite_upper + endpoints(tail)[1])
    delta = interval(Fraction(1, 1000))
    mass_loss = 12 / (PI ** 2) * delta ** 2 * (
        (1 + interval(Fraction(16, 3)) * delta ** 6) / 2
        + interval(Fraction(17, 2)) * (LOG1000 + interval(Fraction(1, 3))))
    positive_lower = endpoints(massless)[0] - endpoints(mass_loss)[1]
    positive = hull(positive_lower, endpoints(massless)[1])
    spacing_threshold = interval(Fraction(48, 25)) ** 2 / (2 * PI)
    conservative_spacing_lower = endpoints(massless)[0] - Fraction(1, 10000)
    return {
        "scope": "Continuum full-source Newton coefficient k0/Omega^2; same symmetric rectangle with delta=0",
        "finite_interval": ["0", "4"], "panels": len(panels),
        "panel_tiling": "Exact rational [0,1/10] and geometric panels through4, ratio<=5/4",
        "quadrature_arithmetic_enclosure": enclosure(quadrature),
        "analytic_Gauss_absolute_error_upper": enclosure(gauss_error)["upper"],
        "positive_radial_tail_upper_T22f": enclosure(tail)["upper"],
        "k0_enclosure": enclosure(massless),
        "mass_change_upper_T22f2_at_delta_001": enclosure(mass_loss)["upper"],
        "k_delta_enclosure_uniform_0_lt_delta_le_001": enclosure(positive),
        "native_spacing_condition_threshold": enclosure(spacing_threshold),
        "native_spacing_condition_pass": positive_lower > endpoints(spacing_threshold)[1],
        "T22l_conservative_k0_lower_minus_0001": decimal_endpoint(conservative_spacing_lower, 24, False),
        "T22l_conservative_spacing_condition_pass": conservative_spacing_lower > endpoints(spacing_threshold)[1],
    }, massless, positive


def positive_normal_rule(nodes: list, weights: list):
    boundary = Fraction(1, 80)
    central_points, central_weights = mapped_gauss(-boundary, boundary, nodes, weights)
    points, measures = central_points[len(nodes) // 2:], central_weights[len(nodes) // 2:]
    panels = geometric_panels(boundary, Fraction(96))
    for lower, upper in panels:
        new_points, new_measures = mapped_gauss(lower, upper, nodes, weights)
        points += new_points
        measures += new_measures
    if any(endpoints(point)[0] <= 0 for point in points):
        raise ArithmeticError("Positive normal node enclosure crosses zero")
    return points, measures, 1 + 2 * len(panels)


def native_tensor_rule(nodes: list, weights: list, workers: int):
    normal_points, normal_weights, normal_panel_count = positive_normal_rule(nodes, weights)
    transverse_panels = geometric_panels(Fraction(1, 10), Fraction(1))
    transverse_panels += geometric_panels(Fraction(1), Fraction(4))
    tasks = []
    for lower, upper in transverse_panels:
        points, measures = mapped_gauss(lower, upper, nodes, weights)
        for point, weight in zip(points, measures):
            tasks.append((BinaryInterval.from_iv(point), BinaryInterval.from_iv(weight), lower >= 1))
    p_vector = BinaryInterval.vector(normal_points)
    w_vector = BinaryInterval.vector(normal_weights)
    row, column = np.triu_indices(len(normal_points))
    pair_p, pair_q = p_vector[row], p_vector[column]
    multiplicity = BinaryInterval(np.where(row == column, 2.0, 4.0))
    pair_measure = w_vector[row] * w_vector[column] * multiplicity
    entropy_base = BinaryInterval.from_iv(4 * LOG2)
    gamma = BinaryInterval.from_iv((4 * LOG2 - 2) * (SQRT2 - 1))
    prefactor = BinaryInterval.from_iv(1 / (2 * PI ** 3))
    masses = ((Fraction(1, 10 ** 6), 1), (Fraction(1000001, 2000000), 2), (Fraction(1), 1))

    def node_integral(task):
        if len(os.listdir("/proc/self/task")) > 16:
            raise RuntimeError("Dense numerical process exceeds the16-thread limit")
        transverse, transverse_weight, high_region = task
        total = BinaryInterval.rational(0)
        for mass_squared, packet_multiplicity in masses:
            mass = BinaryInterval.rational(mass_squared)
            kernel = native_kernel(pair_p, pair_q, transverse, mass)
            kernel = kernel + native_kernel(pair_p, -pair_q, transverse, mass)
            total = total + packet_multiplicity * (pair_measure * kernel).sum()
        entropy_weight = entropy_base + gamma if high_region else entropy_base
        return (prefactor * transverse * transverse_weight * entropy_weight * total).to_iv()

    worker_contracts = []
    worker_contract_lock = Lock()

    def initialize_worker():
        # The floating-point and NumPy error environment is thread-local.
        # Validate it in every actual dense worker before evaluating a node.
        contract = verify_binary64_contract()
        with worker_contract_lock:
            worker_contracts.append(contract)

    quadrature = interval(0)
    print(f"Compact tensor: {normal_panel_count} normal panels, {len(normal_points)} positive nodes, "
          f"{len(tasks)} transverse nodes, {workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=workers, initializer=initialize_worker) as executor:
        for number, value in enumerate(executor.map(node_integral, tasks), 1):
            quadrature += value
            if number % len(nodes) == 0 or number == len(tasks):
                print(f"Compact tensor enclosed {number}/{len(tasks)} transverse nodes", flush=True)
    if len(worker_contracts) != workers:
        raise ArithmeticError("Not every declared tensor worker checked its arithmetic contract")
    return quadrature, {"normal_panels_full": normal_panel_count,
                        "normal_positive_nodes": len(normal_points),
                        "normal_triangle_pairs": len(row),
                        "transverse_panels": len(transverse_panels),
                        "transverse_nodes": len(tasks), "worker_threads": workers,
                        "worker_FP_contract_checks": {
                            "workers_checked": len(worker_contracts),
                            "FE_TONEAREST_ties_to_even_gradual_underflow_FTZ_DAZ_off": True,
                            "np_seterr_over_invalid_divide_raise_underflow_ignore_in_every_worker": True,
                            "interval_kernel_reference_cases_enclosed_in_every_worker": 6},
                        "panel_tiling_exact_rational_verified": True,
                        "symmetry": "Exact reflected Gauss nodes; 4 times off-diagonal plus 2 times diagonal for F(p,q)+F(p,-q)"}


def native_certificate(nodes: list, weights: list, order: int, workers: int, massless_newton):
    quadrature, layout = native_tensor_rule(nodes, weights, workers)
    q_min, q_max, limit = interval(Fraction(1, 10)), interval(4), interval(96)
    entropy_base = 4 * LOG2
    gamma = (entropy_base - 2) * (SQRT2 - 1)
    disk_radius = q_min / 1000
    ellipse_bound = (4 / (2 * PI ** 3) * (4 * q_max / 3)
                     * (entropy_base + gamma) * (12 / disk_radius) ** 2)
    analytic_error = (12 * (q_max - q_min) * (2 * limit) ** 2 * ellipse_bound
                      / (3 ** (2 * order)) / (1 - interval(Fraction(1, 3))))
    pilot_lower = endpoints(quadrature)[0] - endpoints(analytic_error)[1]
    if pilot_lower <= 0:
        raise ArithmeticError("Certified compact pilot lower bound is nonpositive")
    delta_squared = interval(Fraction(1, 10 ** 6))
    squared_mass_difference = delta_squared + delta_squared ** 2 / 4
    stability_loss = (2 / PI * (entropy_base + gamma) * 3 / (2 * SQRT2)
                       * LOG40 * squared_mass_difference)
    stable_root = (certified_iv_sqrt(interval(pilot_lower))
                   - certified_iv_sqrt(interval(endpoints(stability_loss)[1])))
    uniform_lower = interval(0) if endpoints(stable_root)[0] <= 0 else stable_root ** 2
    target = PI / 2 * massless_newton
    margin = uniform_lower - interval(endpoints(target)[1])
    ratio = uniform_lower / interval(endpoints(target)[1])
    return {
        "scope": "T22i compact positive native planar lower integral; Omega=m4=1, P=1",
        "mass_squared_pilot": ["0.000001", "0.5000005", "0.5000005", "1"],
        "Q_interval": ["0.1", "4"], "Q_split": "1", "normal_interval": ["-96", "96"],
        "normal_panel_center": ["-0.0125", "0.0125"], "panel_ratio": "1.25",
        "Gauss_order_per_panel": order, "Bernstein_ellipse_parameter": "3",
        "layout": layout,
        "quadrature_arithmetic_enclosure": enclosure(quadrature),
        "single_variable_ellipse_bound_T22j": enclosure(ellipse_bound),
        "analytic_tensor_absolute_error_upper_T22k": enclosure(analytic_error)["upper"],
        "pilot_positive_lower_bound": decimal_endpoint(pilot_lower, 24, False),
        "mass_stability_D_upper_T22g": enclosure(stability_loss)["upper"],
        "positive_native_lower_bound_uniform_0_lt_delta_le_001": enclosure(uniform_lower),
        "matched_target_upper_from_k0": enclosure(target)["upper"],
        "uniform_margin_over_matched_target": enclosure(margin),
        "uniform_ratio_lower_bound": enclosure(ratio),
        "strict_entropy_excess_pass": endpoints(margin)[0] > 0,
        "omissions": "Q outside[0.1,4], normal momenta outside[-96,96], w-1 and excess native kernel are nonnegative and discarded",
        "status": "Executed arithmetic enclosure; independent source/result audit still required before adopting the physics conclusion",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--order", type=int, default=32)
    parser.add_argument("--digits", type=int, default=80)
    parser.add_argument("--root-bits", type=int, default=120)
    parser.add_argument("--compact-order", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--newton-only", action="store_true")
    parser.add_argument("--gpu-uuid", help="Assigned free GPU UUID for the same-process visible CUDA context")
    args = parser.parse_args()
    if args.order < 32 or args.digits < 60 or args.root_bits < 100:
        raise ValueError("Use at least order32, interval60digits, and root100bits")
    if args.compact_order < 24 or args.compact_order % 2 or not 1 <= args.workers <= 4:
        raise ValueError("Compact rule requires even order>=24 and 1..4 workers")
    if not args.newton_only and not args.gpu_uuid:
        raise ValueError("A long compact run requires its assigned free --gpu-uuid")
    execution = {"single_process_pid": os.getpid(), "niceness": os.getpriority(os.PRIO_PROCESS, 0)}
    if args.gpu_uuid:
        execution.update(hold_cuda_slot(args.gpu_uuid))
    iv.dps = args.digits
    global PI, LOG2, SQRT2, LOG40, LOG1000
    PI, LOG2, SQRT2, LOG40, LOG1000 = constant_enclosures()
    start = time.perf_counter()
    print("Verifying binary64 arithmetic and interval-kernel implementation", flush=True)
    binary_contract = verify_binary64_contract()
    print("Certifying Gauss nodes and weights with exact rational brackets", flush=True)
    nodes, weights, gauss_data = gauss_enclosures(args.order, args.root_bits)
    print("Enclosing k0 with reviewed finite Gauss and positive tail bounds", flush=True)
    newton, massless_newton, _ = newton_certificate(nodes, weights, args.order)
    result = {
        "scope": "T10 order-four polar-law arithmetic; reviewed analytic bounds plus directed interval evaluation",
        "execution": execution,
        "arithmetic": {"library": "mpmath.iv", "decimal_precision": args.digits,
                       "nodes_weights": gauss_data,
                       "binary64_contract": binary_contract,
                       "outer_aggregation": "Sequential outward mpmath.iv scalar additions; dense binary64 reductions alone are pairwise",
                       "constants": {"recipe": "Exact rational Machin arctan brackets, positive atanh sums with bounded tails, integer-square sqrt2 brackets",
                                     "pi": enclosure(PI), "log2": enclosure(LOG2), "sqrt2": enclosure(SQRT2),
                                     "log40": enclosure(LOG40), "log1000": enclosure(LOG1000)},
                       "serialization": "Exact dyadic endpoints rounded outward to decimal strings"},
        "newton": newton,
        "native_entropy": {"status": "not executed; awaiting author-approved compact integration error bound"},
        "complete": False,
        "elapsed_seconds": time.perf_counter() - start,
        "caveat": "This Newton arithmetic does not itself certify a cut-entropy mismatch, a stellar state, a sphere, or a temperature.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(newton, indent=2), flush=True)
    (args.output_dir / "uv_native_certificate.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not args.newton_only:
        print("Certifying compact tensor Gauss nodes and weights", flush=True)
        compact_nodes, compact_weights, compact_gauss = gauss_enclosures(args.compact_order, args.root_bits)
        result["arithmetic"]["compact_nodes_weights"] = compact_gauss
        result["native_entropy"] = native_certificate(compact_nodes, compact_weights, args.compact_order,
                                                     args.workers, massless_newton)
        result["complete"] = True
        result["elapsed_seconds"] = time.perf_counter() - start
        result["caveat"] = ("The executed enclosure concerns only the declared uniform native planar cut and mass/spacing scope. "
                            "It supplies no stellar-state, spherical-cut, thermodynamic or general mass-rectangle result. "
                            "Adoption requires independent source/result review.")
    (args.output_dir / "uv_native_certificate.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    summary = json.dumps(result, indent=2, allow_nan=False) + "\n"
    (args.output_dir / "uv_native_certificate.txt").write_text(summary)
    print(summary, end="", flush=True)


if __name__ == "__main__":
    main()
