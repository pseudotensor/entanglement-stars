#!/usr/bin/env python3
"""Fixed r3 original-sine native Hankel pilot, reviewed equations H1--H10.

This is a distinct fallback to the frozen compact-variance source.  It uses
the periodic eight-dimensional two-site symbol through four scalar Fourier
components, a cut-preserving Hardy change of variables, and the H3 moment
recurrence.  A completed lower test can establish an excess only at this
one pilot; a failed test is inconclusive about entropy normalization.

All execution, including arithmetic diagnostics and refinements, is remote
qi2 only.  Host-only imports precede a fresh device query, a visible CUDA
allocation in this process, and the sole-owned-PID check.  No vendor FFT,
entropy eigensolver, SVD, BLAS product, or unenclosed arithmetic reduction
is used.  Gauss-node seeds from the frozen helper are untrusted; exact
sign-changing root brackets, degree exhaustion, and interval weights
supply the quadrature certificate.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from fractions import Fraction
from pathlib import Path

# No numerical worker pool or BLAS work is needed.  These assignments occur
# before NumPy and the frozen arithmetic helper are imported.
for _thread_variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
from mpmath import iv

import remote_runtime
import uv_native_certificate as arithmetic


BI = arithmetic.BinaryInterval
PILOT = "r3-sine-a250-m1000"
ALPHA = Fraction(1, 250)
DELTA = Fraction(1, 1000)
PACKETS = (
    (Fraction(1, 1000000), 1, [1]),
    (Fraction(1000001, 2000000), 2, [2, 3]),
    (Fraction(1), 1, [4]),
)
TANGENT_MENU = tuple(Fraction(1, value) for value in (64, 32, 16, 8, 4))
DISPLAY_DIGITS = 48


def exact_string(value: Fraction | int) -> str:
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def reported(value):
    return arithmetic.enclosure(value, DISPLAY_DIGITS)


def exact_endpoints(value):
    lower, upper = arithmetic.endpoints(value)
    return {"lower": exact_string(lower), "upper": exact_string(upper)}


def upper_report(value: Fraction):
    return arithmetic.decimal_endpoint(value, DISPLAY_DIGITS, True)


def lower_report(value: Fraction):
    return arithmetic.decimal_endpoint(value, DISPLAY_DIGITS, False)


def resource_check(execution):
    count = len(list(Path("/proc/self/task").iterdir()))
    if os.getpriority(os.PRIO_PROCESS, 0) < 10 or count > 16:
        raise RuntimeError("Remote numerical work requires nice>=10 and <=16 total threads")
    execution["maximum_threads_observed"] = max(execution.get("maximum_threads_observed", 0), count)


def log_rational_bounds(value: Fraction, log_two_bounds, terms: int):
    """Exact power-of-two reduction, positive atanh sum, geometric tail."""
    if value <= 0:
        raise ArithmeticError("Certified logarithm requires a positive rational argument")
    power = value.numerator.bit_length() - value.denominator.bit_length()
    scale = Fraction(1 << power) if power >= 0 else Fraction(1, 1 << -power)
    reduced = value / scale
    while reduced < 1:
        reduced *= 2
        power -= 1
    while reduced > 2:
        reduced /= 2
        power += 1
    z = (reduced - 1) / (reduced + 1)
    if not 0 <= z <= Fraction(1, 3):
        raise ArithmeticError("Logarithm reduction left [1,2]")
    z_squared, term = z * z, z
    total = Fraction(0)
    for index in range(terms):
        total += 2 * term / (2 * index + 1)
        term *= z_squared
    tail = 2 * term / ((2 * terms + 1) * (1 - z_squared))
    log2_lower, log2_upper = log_two_bounds
    if power >= 0:
        return total + power * log2_lower, total + tail + power * log2_upper
    return total + power * log2_upper, total + tail + power * log2_lower


def certified_log(value, log_two_bounds, terms):
    lower, upper = arithmetic.endpoints(value)
    result_lower, _ = log_rational_bounds(lower, log_two_bounds, terms)
    _, result_upper = log_rational_bounds(upper, log_two_bounds, terms)
    return arithmetic.hull(result_lower, result_upper)


def tangent_constants(log_two, digits):
    terms = max(120, (3 * digits + 1) // 2)
    log_two_bounds = arithmetic.endpoints(log_two)
    constants = []
    for tangent in TANGENT_MENU:
        t = arithmetic.interval(tangent)
        if tangent == Fraction(1, 4):
            a, b = 8 * log_two - 2, 8 - 16 * log_two
        else:
            r = arithmetic.certified_iv_sqrt(1 - 4 * t)
            y = (1 - r) / 2
            g = -y * certified_log(y, log_two_bounds, terms)
            g -= (1 - y) * certified_log(1 - y, log_two_bounds, terms)
            g_prime = certified_log((1 + r) / (1 - r), log_two_bounds, terms) / r
            a, b = 2 * g / t - g_prime, (t * g_prime - g) / (t * t)
        if arithmetic.endpoints(a)[0] <= 0 or arithmetic.endpoints(b)[1] >= 0:
            raise ArithmeticError("Enclosed tangent constants do not certify A>0 and B<0")
        constants.append((tangent, a, b))
    return constants, {
        "menu": [exact_string(t) for t in TANGENT_MENU],
        "logarithm": "Exact rational endpoint reduction to [1,2]; positive atanh sum plus geometric tail",
        "logarithm_terms": terms,
        "sqrt": "Every interval endpoint checked by exact rational squared-endpoint inequalities",
        "endpoint_t_equals_one_quarter": "A=8log2-2, B=8-16log2; no 0/0 evaluation",
        "constants": [{"t": exact_string(t), "A": reported(a), "B": reported(b)}
                      for t, a, b in constants],
    }


def centers_and_radii(value):
    """Choose binary64 centers and enclose their actual endpoint distances."""
    if (np.any(~np.isfinite(value.lower)) or np.any(~np.isfinite(value.upper))
            or np.any(value.lower > value.upper)):
        raise ArithmeticError("Finite ordered input intervals are required")
    centers = value.lower * np.float64(0.5) + value.upper * np.float64(0.5)
    centers = np.minimum(value.upper, np.maximum(value.lower, centers))
    below = np.nextafter(np.abs(centers - value.lower), np.inf)
    above = np.nextafter(np.abs(value.upper - centers), np.inf)
    radii = np.maximum(below, above)
    if np.any(~np.isfinite(centers)) or np.any(~np.isfinite(radii)):
        raise ArithmeticError("Input centers and error radii must be finite")
    return centers, radii


def root_table(fft_size):
    """Enclose the 2N-th root and its powers with exact quadrant symmetries."""
    root_order = 2 * fft_size
    quarter = root_order // 4
    cosine, sine = arithmetic.interval(0), arithmetic.interval(1)
    for _ in range(root_order.bit_length() - 3):
        previous_cosine = cosine
        cosine = arithmetic.certified_iv_sqrt((1 + previous_cosine) / 2)
        sine = arithmetic.certified_iv_sqrt((1 - previous_cosine) / 2)
    first_cosines, first_sines = [], []
    current_cosine, current_sine = arithmetic.interval(1), arithmetic.interval(0)
    for _ in range(quarter):
        first_cosines.append(current_cosine)
        first_sines.append(current_sine)
        next_cosine = current_cosine * cosine - current_sine * sine
        next_sine = current_sine * cosine + current_cosine * sine
        current_cosine, current_sine = next_cosine, next_sine
    all_cosines, all_sines = [], []
    for quadrant in range(4):
        for c, s in zip(first_cosines, first_sines):
            if quadrant == 0:
                real, imaginary = c, s
            elif quadrant == 1:
                real, imaginary = -s, c
            elif quadrant == 2:
                real, imaginary = -c, -s
            else:
                real, imaginary = s, -c
            all_cosines.append(real)
            all_sines.append(imaginary)
    real_intervals, imaginary_intervals = BI.vector(all_cosines), BI.vector(all_sines)
    real_centers, real_radii = centers_and_radii(real_intervals)
    imaginary_centers, imaginary_radii = centers_and_radii(imaginary_intervals)
    stages = []
    stage_twiddles = []
    maximum_twiddle_error = Fraction(0)
    size = 2
    while size <= fft_size:
        indices = np.arange(size // 2, dtype=np.int64) * (root_order // size)
        real_radius = float(np.max(real_radii[indices]))
        imaginary_radius = float(np.max(imaginary_radii[indices]))
        # Each twiddle is complex.  Adding the two component radii, with an
        # outward addition, bounds its complex-modulus error; max alone does not.
        complex_radii = np.nextafter(real_radii[indices] + imaginary_radii[indices], np.inf)
        error = Fraction.from_float(float(np.max(complex_radii)))
        maximum_twiddle_error = max(maximum_twiddle_error, error)
        stage_twiddles.append((real_centers[indices], -imaginary_centers[indices]))
        stages.append({"size": size, "real_error_upper": upper_report(Fraction.from_float(real_radius)),
                       "imaginary_error_upper": upper_report(Fraction.from_float(imaginary_radius)),
                       "complex_modulus_error_upper": upper_report(error),
                       "complex_modulus_error_exact_upper": exact_string(error)})
        size *= 2
    if maximum_twiddle_error > Fraction(1, 1000000):
        raise ArithmeticError("Actual twiddle error violates H9's ew<=1e-6 prerequisite")
    return (real_intervals[:fft_size], imaginary_intervals[:fft_size], stage_twiddles,
            maximum_twiddle_error, {
                "order": root_order,
                "construction": "Verified positive half-angle roots from cos(pi/2)=0; enclosing complex powers; exact quadrant symmetries",
                "root_real": reported(cosine), "root_imaginary": reported(sine),
                "root_real_exact_endpoints": exact_endpoints(cosine),
                "root_imaginary_exact_endpoints": exact_endpoints(sine),
                "stages": stages,
                "maximum_twiddle_complex_modulus_error_upper": upper_report(maximum_twiddle_error),
                "maximum_twiddle_complex_modulus_error_exact_upper": exact_string(maximum_twiddle_error),
            })


def bit_reversal(size):
    indices = np.arange(size, dtype=np.int64)
    reversed_indices = np.zeros(size, dtype=np.int64)
    remaining = indices.copy()
    for _ in range(size.bit_length() - 1):
        reversed_indices = (reversed_indices << 1) | (remaining & 1)
        remaining >>= 1
    return reversed_indices


def normalized_radix_two(inputs, reversal, twiddles):
    """The actual H9 butterfly, explicit real primitives and /2 every stage.

    The transform is (1/N) sum_j input[j] exp(-2pi i jk/N).  Bit reversal
    and reshapes only permute data.  Multiplications and sums below are
    separate NumPy elementwise primitives, so no complex vendor product,
    dot product, arithmetic reduction, or fused expression is relied on.
    """
    real = inputs[reversal].copy()
    imaginary = np.zeros_like(real)
    for twiddle_real, twiddle_imaginary in twiddles:
        half = len(twiddle_real)
        real_rows = real.reshape(-1, 2 * half, inputs.shape[1])
        imaginary_rows = imaginary.reshape(-1, 2 * half, inputs.shape[1])
        left_real, left_imaginary = real_rows[:, :half].copy(), imaginary_rows[:, :half].copy()
        right_real, right_imaginary = real_rows[:, half:].copy(), imaginary_rows[:, half:].copy()
        wr, wi = twiddle_real[None, :, None], twiddle_imaginary[None, :, None]
        rr, ii = wr * right_real, wi * right_imaginary
        ri, ir = wr * right_imaginary, wi * right_real
        product_real, product_imaginary = rr - ii, ri + ir
        real_rows[:, :half] = (left_real + product_real) * np.float64(0.5)
        real_rows[:, half:] = (left_real - product_real) * np.float64(0.5)
        imaginary_rows[:, :half] = (left_imaginary + product_imaginary) * np.float64(0.5)
        imaginary_rows[:, half:] = (left_imaginary - product_imaginary) * np.float64(0.5)
    if np.any(~np.isfinite(real)) or np.any(~np.isfinite(imaginary)):
        raise ArithmeticError("Normalized transform produced a non-finite center")
    return real, imaginary


def symbol_components(c, u, transverse, mass_squared):
    """H1; the four components encode the actual periodic 8x8 block."""
    alpha, mass = BI.rational(ALPHA), BI.rational(mass_squared)
    s = c.square() + alpha.square() * u.square()
    t = transverse.square() * s + u.square()
    f = mass * (s ** 6) + t ** 6
    e = f + t * (s ** 5)
    root_e = e.sqrt()
    components = (f.sqrt() / root_e, transverse * (s ** 3) / root_e,
                  u * c * s.square() / root_e, alpha * u.square() * s.square() / root_e)
    # The exact constant has zero retained Fourier coefficients.  Removing
    # it improves the floating centers and leaves the H9 |input|<=1 bound.
    return (components[0] - 1, *components[1:])


def alias_allowances(fft_size, blocks):
    denominator = 1 - Fraction(3, 8) ** (fft_size // 250)
    if denominator <= 0:
        raise ArithmeticError("H8 rational alias denominator is not positive")
    return [2 * 10 ** 12 * Fraction(3, 8) ** ((fft_size - index) // 250) / denominator
            for index in range(1, 2 * blocks)]


def fft_error_allowances(input_errors, twiddle_error, fft_size):
    stages = fft_size.bit_length() - 1
    rounding = Fraction(64, 1 << 53)
    b = twiddle_error + rounding
    denominator = 1 - stages * b
    if denominator <= 0:
        raise ArithmeticError("H9 requires J(ew+64u)<1")
    output = []
    for input_error in input_errors:
        propagated_input = input_error / denominator
        propagated_twiddle = stages * twiddle_error / denominator
        propagated_rounding = stages * rounding / denominator
        total = propagated_input + propagated_twiddle + propagated_rounding
        output.append((total, {
            "input_error_upper": upper_report(input_error),
            "input_error_exact_upper": exact_string(input_error),
            "propagated_input_error_upper": upper_report(propagated_input),
            "propagated_twiddle_error_upper": upper_report(propagated_twiddle),
            "propagated_butterfly_roundoff_upper": upper_report(propagated_rounding),
            "H9_total_complex_modulus_error_upper": upper_report(total),
            "H9_total_complex_modulus_error_exact_upper": exact_string(total),
        }))
    return output


def coefficient_rectangles(real, imaginary, coefficient_errors):
    radius_upper = np.asarray([[arithmetic.outward_float(error, True) for error in row]
                               for row in coefficient_errors], dtype=np.float64)
    enclosed_real = BI(np.nextafter(real - radius_upper, -np.inf),
                       np.nextafter(real + radius_upper, np.inf))
    enclosed_imaginary = BI(np.nextafter(imaginary - radius_upper, -np.inf),
                            np.nextafter(imaginary + radius_upper, np.inf))
    return enclosed_real, enclosed_imaginary


def outer_product(real, imaginary, first, second):
    """n_first n_second^dagger, with every primitive outward enclosed."""
    a, b = real[first][:, None], imaginary[first][:, None]
    c, d = real[second][None, :], imaginary[second][None, :]
    return a * c + b * d, b * c - a * d


def moment_recurrence(real, imaginary, blocks):
    """H3 using scalar 4x4 blocks and the reviewed sliding recurrence."""
    weights = np.asarray([min(index, 2 * blocks - index) for index in range(1, 2 * blocks)],
                         dtype=np.float64)[:, None]
    m1 = (2 * BI(weights) * (real.square() + imaginary.square())).sum()
    m2 = BI.rational(0)
    diagonal = np.arange(4, dtype=np.int64)
    first = np.asarray([0, 0, 0, 1, 1, 2], dtype=np.int64)
    second = np.asarray([1, 2, 3, 2, 3, 3], dtype=np.int64)
    for difference in range(blocks):
        a_real, a_imaginary = BI(np.zeros((4, 4))), BI(np.zeros((4, 4)))
        for index in range(blocks):
            term_real, term_imaginary = outer_product(real, imaginary, index, index + difference)
            a_real, a_imaginary = a_real + term_real, a_imaginary + term_imaginary
        pair_weight = 1 if difference == 0 else 2
        for row in range(blocks - difference):
            trace_real, trace_imaginary = a_real[diagonal, diagonal].sum(), a_imaginary[diagonal, diagonal].sum()
            antisymmetric_real = a_real[first, second] - a_real[second, first]
            antisymmetric_imaginary = a_imaginary[first, second] - a_imaginary[second, first]
            summand = trace_real.square() + trace_imaginary.square()
            summand += (antisymmetric_real.square() + antisymmetric_imaginary.square()).sum()
            m2 += Fraction(pair_weight, 2) * summand
            if row + 1 < blocks - difference:
                column = row + difference
                removed_real, removed_imaginary = outer_product(real, imaginary, row, column)
                added_real, added_imaginary = outer_product(real, imaginary, row + blocks, column + blocks)
                a_real = a_real - removed_real + added_real
                a_imaginary = a_imaginary - removed_imaginary + added_imaginary
    # Only the proven positive moments are intersected with [0,infinity).
    # There is no ratio m2/m1 in this implementation.
    result = []
    for value in (m1, m2):
        lower, upper = arithmetic.endpoints(value.to_iv())
        if upper < 0:
            raise ArithmeticError("Moment enclosure contradicts positivity")
        result.append(arithmetic.hull(max(Fraction(0), lower), upper))
    return tuple(result)


def node_moments(transverse, mass_squared, c, u, reversal, twiddles, twiddle_error,
                 aliases, blocks, fft_size):
    components = symbol_components(c, u, BI.from_iv(transverse), mass_squared)
    centers, input_errors = [], []
    for component in components:
        center, radii = centers_and_radii(component)
        centers.append(center)
        input_errors.append(Fraction.from_float(float(np.max(radii))))
    errors = fft_error_allowances(input_errors, twiddle_error, fft_size)
    real, imaginary = normalized_radix_two(np.stack(centers, axis=1), reversal, twiddles)
    real, imaginary = real[1:2 * blocks], imaginary[1:2 * blocks]
    coefficient_errors = [[alias + error[0] for error in errors] for alias in aliases]
    enclosed_real, enclosed_imaginary = coefficient_rectangles(real, imaginary, coefficient_errors)
    moments = moment_recurrence(enclosed_real, enclosed_imaginary, blocks)
    # This separate enclosure exposes moment-operation rounding at the
    # actual dyadic FFT centers.  It is not subtracted from, or used instead
    # of, the complete coefficient-uncertainty enclosure above.
    center_moments = moment_recurrence(BI(real), BI(imaginary), blocks)
    coefficients = []
    for row, index in enumerate(range(1, 2 * blocks)):
        coefficients.append({
            "k": index,
            "components": [{"mu": component,
                            "real": reported(enclosed_real[row, component].to_iv()),
                            "imaginary": reported(enclosed_imaginary[row, component].to_iv()),
                            "added_complex_modulus_error_upper": upper_report(coefficient_errors[row][component])}
                           for component in range(4)],
        })
    return moments, {
        "Q": reported(transverse),
        "Fourier_components": [{"mu": index, **budget} for index, (_, budget) in enumerate(errors)],
        "coefficients": coefficients,
        "moment_enclosures_before_radial_quadrature": {"m1": reported(moments[0]), "m2": reported(moments[1])},
        "moment_arithmetic_at_dyadic_FFT_centers": {"m1": reported(center_moments[0]), "m2": reported(center_moments[1])},
        "moment_arithmetic_contract": "Each scalar/block primitive and sliding update widened outward; pairwise sums widened at every addition",
        "moment_center_diagnostic_usage": "Reported separately; not an additive error bound for the uncertain coefficients and never substituted for the full enclosures",
    }


def gauss_bracket_details(nodes, weights):
    details = []
    previous_upper = Fraction(-1)
    order = len(nodes)
    for index, (node, weight) in enumerate(zip(nodes, weights)):
        lower, upper = arithmetic.endpoints(node)
        if not -1 < lower < upper < 1 or lower <= previous_upper:
            raise ArithmeticError("Actual Gauss node enclosures must be ordered, disjoint, and interior")
        left_sign, right_sign = arithmetic.legendre_exact(order, lower), arithmetic.legendre_exact(order, upper)
        if left_sign * right_sign >= 0:
            raise ArithmeticError("Actual Gauss node enclosure endpoints do not certify a sign-changing root bracket")
        details.append({"index": index, "node": reported(node), "node_exact_endpoints": exact_endpoints(node),
                        "left_polynomial_sign": 1 if left_sign > 0 else -1,
                        "right_polynomial_sign": 1 if right_sign > 0 else -1,
                        "weight": reported(weight), "weight_exact_endpoints": exact_endpoints(weight)})
        previous_upper = upper
    return details


def radial_error(lower, upper, blocks, order, pi, moment):
    bound = (200 if moment == 1 else 5000) * blocks
    return (arithmetic.interval(4 * (upper - lower) / (1 - Fraction(1, 3)))
            * (2 / pi) * arithmetic.interval(Fraction(4, 3) * upper)
            * bound * arithmetic.interval(Fraction(1, 3 ** (2 * order))))


def positive_integral_enclosure(quadrature, analytic_error):
    lower, upper = arithmetic.endpoints(quadrature)
    allowance = arithmetic.endpoints(analytic_error)[1]
    return arithmetic.hull(max(Fraction(0), lower - allowance), upper + allowance)


def panel_tangent_bound(moment1, moment2, constants):
    moment1_lower = arithmetic.endpoints(moment1)[0]
    moment2_upper = arithmetic.endpoints(moment2)[1]
    candidates = []
    best_lower, best_tangent = Fraction(0), None
    for tangent, a, b in constants:
        a_lower, b_lower = arithmetic.endpoints(a)[0], arithmetic.endpoints(b)[0]
        candidate = arithmetic.interval(a_lower) * arithmetic.interval(moment1_lower)
        candidate += arithmetic.interval(b_lower) * arithmetic.interval(moment2_upper)
        lower = arithmetic.endpoints(candidate)[0]
        candidates.append({"t": exact_string(tangent), "lower_bound_evaluation": reported(candidate),
                           "lower_bound": lower_report(lower),
                           "moment_endpoints_used": "M1 lower and M2 upper; A lower positive and B lower negative"})
        if lower > best_lower:
            best_lower, best_tangent = lower, tangent
    return best_lower, {
        "tangent_candidates": candidates,
        "selected_tangent": exact_string(best_tangent) if best_tangent is not None else None,
        "panel_entropy_lower_bound": lower_report(best_lower),
        "selection": "Maximum of already certified constant-panel lower bounds and zero",
    }


def load_newton(path):
    saved = json.loads(path.read_text())
    if saved.get("newton_complete") is not True:
        raise ValueError("The completed reviewed order-six Newton enclosure is required")
    newton = saved["newton"]
    if (newton.get("delta_interval") != "0<delta<=0.001"
            or newton.get("massless_rectangle") != ["0", "0.5", "0.5", "1"]
            or newton.get("positive_family_mass_squared") != ["delta^2", "(1+delta^2)/2", "(1+delta^2)/2", "1"]
            or newton.get("full_source_integrand") != "12/pi^2 p^2 sum_eta (1+12p^22/(x+p^12))/sqrt(x+p^2+p^12)"):
        raise ValueError("Newton law and family do not match this fixed r3 pilot")
    bounds = newton["K3_positive_family_enclosure"]
    lower, upper = Fraction(bounds["lower"]), Fraction(bounds["upper"])
    if not 0 < lower <= upper:
        raise ValueError("Newton enclosure must have finite ordered positive rational endpoints")
    return newton, upper


def save_result(output_dir, result):
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "uv_order6_hankel.json"
    temporary = output_dir / "uv_order6_hankel.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", choices=[PILOT], default=PILOT)
    parser.add_argument("--normal-blocks", type=int, choices=[16, 32, 64], default=16)
    parser.add_argument("--fft-size", type=int, default=16384)
    parser.add_argument("--gauss-order", type=int, choices=[12, 24], default=12)
    parser.add_argument("--digits", type=int, default=80)
    parser.add_argument("--root-bits", type=int, default=120)
    parser.add_argument("--newton-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    if (args.fft_size < 16384 or args.fft_size & (args.fft_size - 1)
            or args.fft_size <= 2 * args.normal_blocks or args.digits < 80 or args.root_bits < 120):
        raise ValueError("Require power-of-two N>=16384, N>2L, interval digits>=80 and root bits>=120")
    if os.environ.get("QI2_GPU_UUID", args.gpu_uuid) != args.gpu_uuid:
        raise RuntimeError("CLI GPU UUID differs from the freshly selected remote device")
    os.environ["QI2_GPU_UUID"] = args.gpu_uuid
    newton, newton_upper = load_newton(args.newton_result)

    # This final free-device query and allocation are after all host-only
    # imports/file reads and immediately before every numerical operation.
    execution = remote_runtime.remote_execution(max_threads=16)
    start = time.perf_counter()
    iv.dps = args.digits
    worker_contract = arithmetic.verify_binary64_contract()
    resource_check(execution)
    pi, log_two, _, _, _ = arithmetic.constant_enclosures()
    constants, tangent_data = tangent_constants(log_two, args.digits)
    nodes, weights, gauss_data = arithmetic.gauss_enclosures(args.gauss_order, args.root_bits)
    gauss_data["actual_root_brackets_and_weights"] = gauss_bracket_details(nodes, weights)
    gauss_data["seed_usage"] = "Frozen NumPy seeds untrusted; exact rational sign brackets exhaust the polynomial degree"
    c, u, twiddles, twiddle_error, root_data = root_table(args.fft_size)
    reversal = bit_reversal(args.fft_size)
    aliases = alias_allowances(args.fft_size, args.normal_blocks)
    panels = arithmetic.geometric_panels(Fraction(1, 10), Fraction(1), Fraction(7, 6))
    panels += arithmetic.geometric_panels(Fraction(1), Fraction(4), Fraction(7, 6))
    if (panels[0][0] != Fraction(1, 10) or panels[-1][1] != 4
            or any(a[1] != b[0] for a, b in zip(panels, panels[1:]))):
        raise ArithmeticError("Combined panels do not tile the exact declared radial interval")
    target = (pi / 2) * arithmetic.interval(newton_upper)
    target_upper = arithmetic.endpoints(target)[1]
    spacing_squared_upper = ALPHA ** 2 * 2 * arithmetic.endpoints(pi)[1] * newton_upper
    if spacing_squared_upper >= Fraction(1, 2500):
        raise ArithmeticError("Imported Newton upper bound fails the fixed pilot's a/lP<0.02 certificate")
    result = {
        "scope": "Fixed r3-sine-a250-m1000 original-sine native planar two-moment lower-bound pilot, H1--H10",
        "execution": execution,
        "law": {
            "pilot": PILOT, "order": 6, "stencil": "original sine", "uniform_P": "1", "Omega": "m4=1",
            "a_Omega": exact_string(ALPHA), "m1_over_Omega": exact_string(DELTA),
            "mass_squared_over_Omega_squared": ["1/1000000", "1000001/2000000", "1000001/2000000", "1"],
            "mass_rectangle": "Fixed symmetric rectangle; all four packet entropies add positively",
            "state": "Free occupied polar sea, empty local number-conserving bosons",
            "spatial_algebra": "Original site CAR; exact normal cells (2j,2j+1), cut between cells -1 and 0",
            "normal_symbol": "C=(I8-sum_mu G_mu n_mu)/2; Tr(G_mu G_nu)=8 delta_mu_nu; H1's four periodic components",
            "Hardy_map": "Fixed within-cell unitaries and g_b with b=(1-alpha)/(1+alpha), preserving both cut subspaces",
            "transverse_measure": "(2/pi) Q dQ, four transverse sine preimages; positive excess Jacobian discarded",
            "extra_valley_factor": 1, "second_interface_divisor": 1,
            "Newton_match": "One unchanged continuum match kOmega/Omega^2=K3; no finite-BZ rematch or fit",
            "omissions": "Q outside[1/10,4], positive excess transverse Jacobian, and Hankel rows/columns outside L are discarded by lower bounds",
        },
        "resolution": {"normal_blocks_L": args.normal_blocks, "fft_size_N": args.fft_size,
                       "Gauss_order": args.gauss_order, "digits": args.digits, "root_bits": args.root_bits,
                       "worker_count": 1, "meaning_of_N": "Fourier sampling of the infinite chain; not a periodic slab"},
        "arithmetic": {
            "frozen_helper_source": "solve/code/uv_native_certificate.py: interval, constants, Gauss, BinaryInterval and FP guards only",
            "worker_FP_contract": worker_contract,
            "worker_checks": "The sole numerical worker checks FE_TONEAREST, ties, gradual underflow, FTZ/DAZ, niceness and threads before work",
            "roots_and_twiddles": root_data, "Gauss_nodes_weights": gauss_data,
            "pi": reported(pi), "log2": reported(log_two),
            "FFT": "Specified normalized radix-two real-primitive butterfly; each stage divides by two",
            "FFT_error": "H9 rational bound split into propagated input, twiddle and butterfly-roundoff terms",
            "complex_error_convention": "Absolute complex-modulus errors; sum of real and imaginary radii, never their maximum",
            "moment_arithmetic": "H3 4x4 outward interval products, sliding recurrence and outward pairwise reductions",
            "outer_aggregation": "Sequential outward mpmath.iv scalar additions; exact dyadic endpoint extraction",
            "serialization": f"Outward decimal endpoints at {DISPLAY_DIGITS} places; exact root bracket endpoints also supplied",
        },
        "normal_Fourier_alias": {
            "H8_sigma": "1/250", "H8_envelope_M": "1000000000000", "rational_base": "3/8",
            "allowances": [{"k": k, "absolute_component_error_upper": upper_report(error),
                            "absolute_component_error_exact_upper": exact_string(error)}
                           for k, error in enumerate(aliases, 1)],
        },
        "entropy_tangents": tangent_data,
        "newton_input": newton,
        "comparison": {"target_enclosure_pi_over_2_K3_upper": reported(target),
                       "target_outward_upper": upper_report(target_upper),
                       "spacing_squared_upper": upper_report(spacing_squared_upper),
                       "fixed_pilot_a_over_lP_below_0p02": True},
        "packets": [], "arithmetic_complete": False, "strict_entropy_excess_pass": False,
        "entropy_normalization_disposition": "Runtime incomplete; no entropy comparison credited",
        "caveat": "A failed lower test is inconclusive about full entropy and is not a normalization pass. No other spacing, mass, stencil, sphere, prepared/interacting state, temperature or thermodynamic entropy is certified.",
    }
    save_result(args.output_dir, result)
    total_lower = arithmetic.interval(0)
    for mass_squared, multiplicity, packet_indices in PACKETS:
        packet = {"mass_squared": exact_string(mass_squared), "packet_indices": packet_indices,
                  "multiplicity": multiplicity, "panels": []}
        result["packets"].append(packet)
        packet_lower = arithmetic.interval(0)
        for panel_index, (lower, upper) in enumerate(panels):
            resource_check(execution)
            points, measures = arithmetic.mapped_gauss(lower, upper, nodes, weights)
            quadratures = [arithmetic.interval(0), arithmetic.interval(0)]
            node_data = []
            for point, measure in zip(points, measures):
                resource_check(execution)
                moments, node = node_moments(point, mass_squared, c, u, reversal, twiddles, twiddle_error,
                                             aliases, args.normal_blocks, args.fft_size)
                node["radial_Gauss_measure"] = reported(measure)
                node_data.append(node)
                prefactor = (2 / pi) * point * measure
                for moment in range(2):
                    quadratures[moment] += prefactor * moments[moment]
            analytic_errors = [radial_error(lower, upper, args.normal_blocks, args.gauss_order, pi, moment)
                               for moment in (1, 2)]
            moment_integrals = [positive_integral_enclosure(value, error)
                                for value, error in zip(quadratures, analytic_errors)]
            bound, selected = panel_tangent_bound(*moment_integrals, constants)
            packet_lower += arithmetic.interval(bound)
            packet["panels"].append({
                "index": panel_index, "Q_interval": [exact_string(lower), exact_string(upper)],
                "quadrature_arithmetic_enclosures_before_H10": {"M1": reported(quadratures[0]), "M2": reported(quadratures[1])},
                "H10_separate_analytic_Gauss_error_upper": {"M1": reported(analytic_errors[0])["upper"],
                                                          "M2": reported(analytic_errors[1])["upper"]},
                "moment_integrals_after_H10": {"M1": reported(moment_integrals[0]), "M2": reported(moment_integrals[1])},
                **selected, "nodes": node_data,
            })
            print(f"Hankel packet {packet_indices}, panel {panel_index + 1}/{len(panels)} enclosed; "
                  f"selected t={selected['selected_tangent']}, lower={selected['panel_entropy_lower_bound']}", flush=True)
            packet["partial_lower_sum_enclosure"] = reported(packet_lower)
            save_result(args.output_dir, result)
        packet["single_packet_lower_sum_enclosure"] = reported(packet_lower)
        packet["multiplicity_weighted_lower_sum_enclosure"] = reported(multiplicity * packet_lower)
        total_lower += multiplicity * packet_lower
    resource_check(execution)
    total_lower_endpoint = arithmetic.endpoints(total_lower)[0]
    margin = total_lower_endpoint - target_upper
    strict_excess = margin > 0
    result["comparison"].update({
        "four_packet_entropy_lower_sum_enclosure": reported(total_lower),
        "four_packet_entropy_certified_lower": lower_report(total_lower_endpoint),
        "lower_excess_margin": lower_report(margin),
        "margin_exact_rational_lower": exact_string(margin),
    })
    result["arithmetic_complete"] = True
    result["strict_entropy_excess_pass"] = strict_excess
    result["entropy_normalization_disposition"] = (
        "Certified strict lower-bound excess at this one fixed native planar pilot"
        if strict_excess else "Inconclusive lower-bound test; entropy normalization is unsettled"
    )
    result["elapsed_seconds"] = time.perf_counter() - start
    save_result(args.output_dir, result)
    print(json.dumps({"arithmetic_complete": True, "strict_entropy_excess_pass": strict_excess,
                      "comparison": result["comparison"], "output": str(args.output_dir / "uv_order6_hankel.json")},
                     indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
