"""Norm-ball arithmetic for the reviewed original-A D finite diagnostic.

All execution, including import checks, belongs on the remote qi2 host.
Centers use binary64 NumPy/CuPy arrays. Every primitive encloses its
roundoff; radii and scalar transcendental bounds use directed Decimal.
The arithmetic contract is ordinary IEEE real additions/multiplications,
with at most 8*k+4 elementary operations per complex length-k dot product.
Fused multiply-add or a different reduction order satisfies this budget.
The absolute underflow allowance also covers flush-to-zero below the
binary64 normal range. No single-precision/TF32 product is permitted.

An unchecked eigensolver, inverse, or Sylvester solver supplies candidates
only. Reconstruction, positivity, and true equation residuals certify
the returned balls. Jets store Taylor coefficients f^(k)/k!, k=0..3.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext, ROUND_CEILING, ROUND_FLOOR
import math

import numpy as np


DECIMAL_DIGITS = 80
XP = np
CERTIFIED = True
EVENTS: dict[str, float | int] = {}
_DOT_KERNEL = None
_NORM_KERNEL = None
U64 = Decimal("1.1102230246251565404236316680908203125e-16")
TINY64 = Decimal.from_float(math.ldexp(1.0, -1022))


def configure(backend="numpy", certified=True, digits=80):
    global XP, CERTIFIED, DECIMAL_DIGITS, EVENTS, _DOT_KERNEL, _NORM_KERNEL
    if backend == "cupy":
        import cupy
        XP = cupy
        if XP.cuda.runtime.getDeviceCount() != 1:
            raise RuntimeError("Expose only the freshly reserved GPU to this process.")
    elif backend == "numpy":
        XP = np
    else:
        raise ValueError("Backend must be numpy or cupy.")
    CERTIFIED = bool(certified)
    DECIMAL_DIGITS = int(digits)
    if DECIMAL_DIGITS < 40:
        raise ValueError("Directed scalar arithmetic requires >=40 digits.")
    EVENTS = {}
    _DOT_KERNEL = None
    _NORM_KERNEL = None
    if np.finfo(np.float64).nmant != 52:
        raise RuntimeError("The binary64 center arithmetic contract failed.")


def host(value):
    return np.asarray(value) if XP is np else XP.asnumpy(value)


def candidate_eigh(value):
    """Unverified candidate on the remote CPU, avoiding cuSolver failures.

    The eigensolver supplies approximate centers only. Every use which
    enters the proof still has its independent algebraic residual and
    positivity/gap bounds; no eigensolver accuracy constant is assumed.
    """
    values, vectors = np.linalg.eigh(host(value))
    record("remote_cpu_candidate_eigensolver_used", 1)
    return XP.asarray(values), XP.asarray(vectors)


def candidate_solve(matrix, rhs):
    record("remote_cpu_candidate_linear_solve_used", 1)
    return XP.asarray(np.linalg.solve(host(matrix), host(rhs)))


def decimal(value):
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, np.integer)):
        return Decimal(int(value))
    return Decimal.from_float(float(value))


def float_up(value):
    value = decimal(value)
    result = float(value)
    if not math.isfinite(result):
        raise ArithmeticError("Nonfinite bound.")
    if Decimal.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def float_down(value):
    value = decimal(value)
    result = float(value)
    if not math.isfinite(result):
        raise ArithmeticError("Nonfinite bound.")
    if Decimal.from_float(result) > value:
        result = math.nextafter(result, -math.inf)
    return result


def add_up(*values):
    with localcontext() as context:
        context.prec = DECIMAL_DIGITS
        context.rounding = ROUND_CEILING
        return float_up(sum((decimal(x) for x in values), Decimal(0)))


def sub_down(first, second):
    with localcontext() as context:
        context.prec = DECIMAL_DIGITS
        context.rounding = ROUND_FLOOR
        return float_down(decimal(first) - decimal(second))


def mul_up(*values):
    with localcontext() as context:
        context.prec = DECIMAL_DIGITS
        context.rounding = ROUND_CEILING
        result = Decimal(1)
        for value in values:
            if value < 0:
                raise ValueError("mul_up is for nonnegative bounds.")
            result *= decimal(value)
        return float_up(result)


def div_up(first, second):
    if first < 0 or second <= 0:
        raise ValueError("div_up requires a nonnegative numerator/positive denominator.")
    with localcontext() as context:
        context.prec = DECIMAL_DIGITS
        context.rounding = ROUND_CEILING
        return float_up(decimal(first) / decimal(second))


def sqrt_up(value):
    with localcontext() as context:
        context.prec = DECIMAL_DIGITS
        # Decimal.sqrt is correctly rounded to nearest; one Decimal ulp
        # covers that rounding independently of the context rounding mode.
        return float_up(decimal(value).sqrt(context=context).next_plus(context))


def sqrt_down(value):
    with localcontext() as context:
        context.prec = DECIMAL_DIGITS
        return float_down(decimal(value).sqrt(context=context).next_minus(context))


def exp_up(value):
    with localcontext() as context:
        context.prec = DECIMAL_DIGITS
        return float_up(decimal(value).exp(context=context).next_plus(context))


def gamma(count, unit=U64):
    with localcontext() as context:
        context.prec = DECIMAL_DIGITS
        context.rounding = ROUND_CEILING
        amount = decimal(count) * unit
        if amount >= 1:
            raise ArithmeticError("Roundoff budget exceeds its domain.")
        return float_up(amount / (1 - amount))


def underflow(count, size=1):
    with localcontext() as context:
        context.prec = DECIMAL_DIGITS
        context.rounding = ROUND_CEILING
        amount = decimal(count) * U64
        if 2 * amount >= 1:
            raise ArithmeticError("Underflow accumulation budget exceeds its domain.")
        # The extra geometric factor also covers later inversion of a
        # relative-error inequality when deriving norm upper bounds.
        return float_up(decimal(count) * decimal(size) * TINY64 / (1 - 2 * amount))


def record(name, value):
    EVENTS[name] = max(EVENTS.get(name, 0), float(value))


def _finite(value):
    if not bool(XP.all(XP.isfinite(value)).item()):
        raise ArithmeticError("Nonfinite center arithmetic.")


def _scalar(value):
    return float(value.item() if hasattr(value, "item") else value)


def complex_product_center(first, second):
    """Four real products and two real additions; no opaque complex mul."""
    first, second = XP.asarray(first), XP.asarray(second)
    return ((first.real * second.real - first.imag * second.imag)
            + 1j * (first.real * second.imag + first.imag * second.real))


def norm_upper(value):
    """A certified operator/Euclidean norm upper bound of an exact center.

    Complex absolute values are bounded by |Re|+|Im|; no libm hypot
    accuracy assumption is used. Matrices use explicit row/column
    reductions and sqrt(one-norm*infinity-norm); vectors use Decimal.
    """
    value = XP.asarray(value, dtype=XP.complex128)
    _finite(value)
    if value.ndim == 0:
        re, im = abs(_scalar(value.real)), abs(_scalar(value.imag))
        return sqrt_up(add_up(mul_up(re, re), mul_up(im, im)))
    if value.ndim != 2 or min(value.shape) == 1:
        # Vectors are short in this calculation. Directed Decimal summation
        # avoids a second opaque floating reduction and sharpens the angle
        # and projected-response bounds.
        entries = host(value).reshape(-1)
        with localcontext() as context:
            context.prec = DECIMAL_DIGITS
            context.rounding = ROUND_CEILING
            upper_square = sum((decimal(x.real) ** 2 + decimal(x.imag) ** 2 for x in entries), Decimal(0))
        return sqrt_up(upper_square)
    rows, columns = value.shape
    row_sums, column_sums = _deterministic_norm_sums(value)
    row = max(host(row_sums).tolist(), default=0.0)
    column = max(host(column_sums).tolist(), default=0.0)
    row = add_up(div_up(row, sub_down(1, gamma(2 * columns + 4))), underflow(2 * columns + 4))
    column = add_up(div_up(column, sub_down(1, gamma(2 * rows + 4))), underflow(2 * rows + 4))
    return sqrt_up(mul_up(row, column))


def absolute_norm_upper(value):
    """Bound ||abs(value)||, needed in componentwise roundoff bounds."""
    value = XP.asarray(value, dtype=XP.complex128)
    if value.ndim == 0:
        return add_up(abs(_scalar(value.real)), abs(_scalar(value.imag)))
    if value.ndim != 2 or min(value.shape) == 1:
        entries = host(value).reshape(-1)
        with localcontext() as context:
            context.prec = DECIMAL_DIGITS
            context.rounding = ROUND_CEILING
            squares = sum(((abs(decimal(x.real)) + abs(decimal(x.imag))) ** 2 for x in entries), Decimal(0))
        return sqrt_up(squares)
    # norm_upper already uses exact |Re|+|Im| row/column majorants.
    return norm_upper(value)


def _deterministic_norm_sums(value):
    """Explicit2k nonnegative additions for every row/column sum."""
    global _NORM_KERNEL
    if value.ndim != 2 or value.dtype != np.dtype(np.complex128):
        raise ValueError("Norm kernels require a two-dimensional complex128 input.")
    rows, columns = value.shape
    if rows == 0 or columns == 0:
        return XP.zeros(rows), XP.zeros(columns)
    if XP is not np:
        if _NORM_KERNEL is None:
            _NORM_KERNEL = XP.RawKernel(r'''
            extern "C" {
            struct complex_pair { double re; double im; };
            __global__ void d_certified_norm_sum(const complex_pair* a,
                double* result, long long rows, long long columns, int by_row) {
                long long index=(long long)blockDim.x*blockIdx.x+threadIdx.x;
                long long count=by_row?rows:columns;
                if (index >= count) return;
                long long inner=by_row?columns:rows;
                double total=0.0;
                for (long long k=0; k<inner; ++k) {
                    long long entry=by_row?index*columns+k:k*columns+index;
                    complex_pair x=a[entry];
                    double re=x.re<0.0?-x.re:x.re;
                    double im=x.im<0.0?-x.im:x.im;
                    total=(total+re)+im;
                }
                result[index]=total;
            }
            }
            ''', "d_certified_norm_sum", options=("--fmad=false", "--prec-div=true", "--prec-sqrt=true"))
        value = XP.ascontiguousarray(value)
        row_result = XP.empty(rows, dtype=XP.float64)
        column_result = XP.empty(columns, dtype=XP.float64)
        _NORM_KERNEL(((rows + 127) // 128,), (128,),
                     (value, row_result, np.int64(rows), np.int64(columns), np.int32(1)))
        _NORM_KERNEL(((columns + 127) // 128,), (128,),
                     (value, column_result, np.int64(rows), np.int64(columns), np.int32(0)))
    else:
        row_result, column_result = np.zeros(rows), np.zeros(columns)
        for k in range(columns):
            row_result = (row_result + np.abs(value.real[:, k])) + np.abs(value.imag[:, k])
        for k in range(rows):
            column_result = (column_result + np.abs(value.real[k, :])) + np.abs(value.imag[k, :])
    record("deterministic_norm_reduction_used", 1)
    return row_result, column_result


def _deterministic_product(first, second):
    """Explicit8k binary64 elementary operations per complex dot.

    Vendor GEMM never supplies a certificate center. CuPy's NVRTC kernel
    is a literal sequential real/imaginary sum compiled without fast
    math or contraction. CuPy's added FTZ option is covered by the
    absolute2^-1022-per-operation underflow allowance. CPU verification uses the same
    sequence of NumPy real ufuncs. Eigensolver/inverse candidates may
    still use vendor libraries because their errors are residual-tested.
    """
    global _DOT_KERNEL
    if (first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[0]
            or first.dtype != np.dtype(np.complex128) or second.dtype != np.dtype(np.complex128)):
        raise ValueError("Dot kernels require matching two-dimensional complex128 inputs.")
    rows, length, columns = first.shape[0], first.shape[1], second.shape[1]
    if not rows or not length or not columns:
        return XP.zeros((rows, columns), dtype=XP.complex128)
    if XP is not np:
        if _DOT_KERNEL is None:
            _DOT_KERNEL = XP.RawKernel(r'''
            extern "C" {
            struct complex_pair { double re; double im; };
            __global__ void d_certified_dot(const complex_pair* a,
                const complex_pair* b, complex_pair* c,
                long long rows, long long inner, long long columns) {
                long long entry=(long long)blockDim.x*blockIdx.x+threadIdx.x;
                if (entry >= rows*columns) return;
                long long row=entry/columns, column=entry%columns;
                double re=0.0, im=0.0;
                for (long long k=0; k<inner; ++k) {
                    complex_pair left=a[row*inner+k];
                    complex_pair right=b[k*columns+column];
                    re=(re+left.re*right.re)-left.im*right.im;
                    im=(im+left.re*right.im)+left.im*right.re;
                }
                c[entry].re=re; c[entry].im=im;
            }
            }
            ''', "d_certified_dot", options=("--fmad=false", "--prec-div=true", "--prec-sqrt=true"))
        first = XP.ascontiguousarray(first)
        second = XP.ascontiguousarray(second)
        answer = XP.empty((rows, columns), dtype=XP.complex128)
        _DOT_KERNEL(((rows * columns + 255) // 256,), (256,),
                    (first, second, answer, np.int64(rows), np.int64(length), np.int64(columns)))
    else:
        real, imag = np.zeros((rows, columns)), np.zeros((rows, columns))
        for k in range(length):
            ar, ai = first.real[:, k:k + 1], first.imag[:, k:k + 1]
            br, bi = second.real[k:k + 1, :], second.imag[k:k + 1, :]
            real = (real + ar * br) - ai * bi
            imag = (imag + ar * bi) + ai * br
        answer = real + 1j * imag
    record("deterministic_binary64_dot_used", 1)
    return answer


def _matmul_center(first, second):
    """Extended-precision verification for narrow matrix/vector products.

    On the qi2 Linux host NumPy longdouble has >=63 fraction bits. The
    casts from binary64 are exact. Four real products form the complex
    result, then an explicitly bounded conversion returns binary64.
    Large matrix products stay on the selected NumPy/CuPy backend.
    """
    narrow = (first.ndim == 2 and second.ndim == 2
              and (first.shape[0] == 1 or second.shape[1] == 1))
    length = first.shape[-1]
    if CERTIFIED and narrow:
        bits = int(np.finfo(np.longdouble).nmant) + 1
        if bits < 64:
            raise RuntimeError("A >=64-bit significand is required for verified matvecs.")
        a, b = host(first), host(second)
        ar, ai = a.real.astype(np.longdouble), a.imag.astype(np.longdouble)
        br, bi = b.real.astype(np.longdouble), b.imag.astype(np.longdouble)
        re = np.zeros((first.shape[0], second.shape[1]), dtype=np.longdouble)
        im = np.zeros_like(re)
        for k in range(length):
            re = (re + ar[:, k:k + 1] * br[k:k + 1, :]) - ai[:, k:k + 1] * bi[k:k + 1, :]
            im = (im + ar[:, k:k + 1] * bi[k:k + 1, :]) + ai[:, k:k + 1] * br[k:k + 1, :]
        value = XP.asarray(np.asarray(re, np.float64) + 1j * np.asarray(im, np.float64))
        with localcontext() as context:
            context.prec = DECIMAL_DIGITS
            long_unit = Decimal(2) ** -bits
        relative = add_up(gamma(8 * length + 4, long_unit), gamma(4))
        record("extended_matvec_significand_bits", bits)
    else:
        value = _deterministic_product(first, second) if CERTIFIED else first @ second
        relative = gamma(8 * length + 4)
    _finite(value)
    error = add_up(mul_up(relative, absolute_norm_upper(first), absolute_norm_upper(second)),
                   underflow(8 * length + 4, max(1, int(value.size))))
    record("largest_matmul_roundoff", error)
    return value, error


@dataclass
class Ball:
    data: object
    radius: float = 0.0
    lower: float | None = None
    __array_priority__ = 20000

    def __post_init__(self):
        self.data = XP.asarray(self.data, dtype=XP.complex128)
        self.radius = float(self.radius) if CERTIFIED else 0.0
        if self.radius < 0 or not math.isfinite(self.radius):
            raise ArithmeticError("Invalid norm radius.")
        _finite(self.data)
        self._norm = None
        self._absolute_norm = None

    @classmethod
    def rational(cls, numerator, denominator=1):
        with localcontext() as context:
            context.prec = DECIMAL_DIGITS
            context.rounding = ROUND_FLOOR
            low = decimal(numerator) / decimal(denominator)
            context.rounding = ROUND_CEILING
            high = decimal(numerator) / decimal(denominator)
            center = float((low + high) / 2)
            error = max(decimal(center) - low, high - decimal(center))
        return cls(center, float_up(error))

    @classmethod
    def interval(cls, low, high):
        low, high = decimal(low), decimal(high)
        if low > high:
            raise ValueError("Invalid interval.")
        with localcontext() as context:
            context.prec = DECIMAL_DIGITS
            center = float((low + high) / 2)
            context.rounding = ROUND_CEILING
            error = max(decimal(center) - low, high - decimal(center))
        return cls(center, float_up(error))

    def bounds(self):
        if self.data.ndim != 0 or abs(_scalar(self.data.imag)) > self.radius:
            raise ValueError("Real scalar ball required.")
        center = _scalar(self.data.real)
        return sub_down(center, self.radius), add_up(center, self.radius)

    def center_norm(self):
        if self._norm is None:
            self._norm = norm_upper(self.data)
        return self._norm

    def absolute_norm(self):
        if self._absolute_norm is None:
            self._absolute_norm = absolute_norm_upper(self.data)
        return self._absolute_norm

    def norm(self):
        return add_up(self.center_norm(), self.radius)

    def coerce(self, other):
        return other if isinstance(other, Ball) else Ball(other)

    def dagger(self):
        return Ball(self.data.conj().T, self.radius, self.lower)

    def hermitian(self):
        return (self + self.dagger()) * Ball.rational(1, 2)

    def __neg__(self):
        return Ball(-self.data, self.radius)

    def __add__(self, other):
        if isinstance(other, Taylor3):
            return NotImplemented
        other = self.coerce(other)
        value = self.data + other.data
        # Scalar broadcasting multiplies the radius by the norm of ones.
        first_gain = norm_upper(XP.ones(value.shape)) if self.data.ndim == 0 and value.ndim else 1.0
        second_gain = norm_upper(XP.ones(value.shape)) if other.data.ndim == 0 and value.ndim else 1.0
        rounding = add_up(mul_up(gamma(2), add_up(mul_up(first_gain, self.absolute_norm()),
                                                   mul_up(second_gain, other.absolute_norm()))),
                          underflow(2, max(1, int(value.size))))
        return Ball(value, add_up(mul_up(first_gain, self.radius), mul_up(second_gain, other.radius), rounding))

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-self.coerce(other))

    def __rsub__(self, other):
        return self.coerce(other) - self

    def __mul__(self, other):
        if isinstance(other, Taylor3):
            return NotImplemented
        other = self.coerce(other)
        value = complex_product_center(self.data, other.data)
        rounding = add_up(mul_up(gamma(8), self.absolute_norm(), other.absolute_norm()),
                          underflow(8, max(1, int(value.size))))
        # Elementwise matrix products are compressions of tensor products.
        error = add_up(mul_up(self.center_norm(), other.radius),
                       mul_up(other.center_norm(), self.radius),
                       mul_up(self.radius, other.radius), rounding)
        return Ball(value, error)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.coerce(other)
        if other.data.ndim:
            raise ValueError("Use inverse() for matrix inverses.")
        return self * other.scalar_inverse()

    def __matmul__(self, other):
        if isinstance(other, Taylor3):
            return NotImplemented
        other = self.coerce(other)
        value, rounding = _matmul_center(self.data, other.data)
        error = add_up(mul_up(self.center_norm(), other.radius),
                       mul_up(other.center_norm(), self.radius),
                       mul_up(self.radius, other.radius), rounding)
        return Ball(value, error)

    def __rmatmul__(self, other):
        return self.coerce(other) @ self

    def scalar_inverse(self):
        low, high = self.bounds()
        if low <= 0 <= high:
            raise ArithmeticError("Scalar inverse crosses zero.")
        with localcontext() as context:
            context.prec = DECIMAL_DIGITS
            context.rounding = ROUND_FLOOR
            lo = Decimal(1) / decimal(high)
            context.rounding = ROUND_CEILING
            hi = Decimal(1) / decimal(low)
        return Ball.interval(lo, hi)

    def scalar_sqrt(self):
        low, high = self.bounds()
        if low < 0:
            raise ArithmeticError("Scalar square root left the positive domain.")
        return Ball.interval(sqrt_down(low), sqrt_up(high))

    def scalar_exp(self):
        low, high = self.bounds()
        with localcontext() as context:
            context.prec = DECIMAL_DIGITS
            lo = decimal(low).exp(context=context).next_minus(context)
            hi = decimal(high).exp(context=context).next_plus(context)
        return Ball.interval(lo, hi)

    def map(self, function, gain=None):
        value = XP.asarray(function(host(self.data)))
        if gain is None:
            if self.data.ndim == 1 and value.ndim == 2 and value.shape == (2 * self.data.size, 2 * self.data.size):
                # The shared diagonal_field repeats entries twice on the
                # diagonal; its Euclidean-to-operator gain is <=1.
                gain = 1.0
            elif self.data.ndim == 2 and value.shape == (2 * self.data.shape[0], self.data.shape[1]):
                # The sole shared embedding is vertical stacking with zeros.
                gain = 1.0
            elif value.shape == self.data.shape:
                raise ValueError("A map needs an explicit certified gain.")
            else:
                raise ValueError("An unsupported map needs a certified gain.")
        # These maps perform only indexing/copying and exact zero insertion.
        return Ball(value, mul_up(gain, self.radius))

    def inverse(self):
        if self.data.ndim == 0:
            return self.scalar_inverse()
        candidate = Ball(XP.asarray(np.linalg.inv(host(self.data))))
        residual = Ball(XP.eye(len(self.data))) - self @ candidate
        amount = residual.norm()
        if amount >= 1:
            raise ArithmeticError("Matrix inverse Neumann residual did not close.")
        error = div_up(mul_up(candidate.norm(), amount), sub_down(1, amount))
        record("largest_inverse_residual", amount)
        return Ball(candidate.data, error)

    def positive_root(self):
        """Residual-certified root, including positivity of the candidate."""
        values, vectors = candidate_eigh(self.data)
        values_host = host(values)
        if float(values_host[0]) <= 0:
            raise ArithmeticError("Root candidate is not positive.")
        q = Ball(vectors)
        identity = Ball(XP.eye(len(values)))
        orthogonality = (q.dagger() @ q - identity).norm()
        if orthogonality >= 1:
            raise ArithmeticError("Root eigenvectors do not certify an invertible basis.")
        reconstruction = q @ Ball(XP.diag(values)) @ q.dagger()
        reconstructed_residual = (self - reconstruction).norm()
        with localcontext() as context:
            context.prec = DECIMAL_DIGITS
            context.rounding = ROUND_FLOOR
            a_lower = float_down(decimal(values_host[0]) * decimal(sub_down(1, orthogonality))
                                 - decimal(reconstructed_residual))
        if self.lower is not None:
            a_lower = max(a_lower, self.lower)
        if a_lower <= 0:
            raise ArithmeticError("The interval root argument is not certified positive.")
        sqrt_values = XP.asarray([float(Ball(x).scalar_sqrt().data.real.item()) for x in values_host])
        sqrt_errors = max(Ball(x).scalar_sqrt().radius for x in values_host)
        reconstructed_root = q @ Ball(XP.diag(sqrt_values), sqrt_errors) @ q.dagger()
        symmetric = reconstructed_root.hermitian()
        center = Ball(symmetric.data)
        with localcontext() as context:
            context.prec = DECIMAL_DIGITS
            context.rounding = ROUND_FLOOR
            candidate_lower = float_down(decimal(sqrt_down(values_host[0])) * decimal(sub_down(1, orthogonality))
                                         - decimal(symmetric.radius))
        if candidate_lower <= 0:
            raise ArithmeticError("The approximate positive root is not certified positive.")
        residual = (center @ center - self).norm()
        # The denominator of an upper error bound must be rounded down.
        with localcontext() as context:
            context.prec = DECIMAL_DIGITS
            context.rounding = ROUND_FLOOR
            denominator = float_down(decimal(candidate_lower) + decimal(sqrt_down(a_lower)))
        error = div_up(residual, denominator)
        record("largest_root_equation_residual", residual)
        record("largest_root_radius", error)
        return Ball(center.data, error, sqrt_down(a_lower))


def positive_solver(matrix: Ball, lower=None):
    if lower is None:
        lower = matrix.lower
    if lower is None or lower <= 0:
        raise ValueError("A proved positive denominator is required.")
    values, vectors = candidate_eigh(matrix.data)
    if float(host(values)[0]) <= 0:
        raise ArithmeticError("Approximate Sylvester denominator is not positive.")
    denominators = values[:, None] + values[None, :]

    def solve(rhs: Ball):
        # Eigensolver arithmetic is only used to form this candidate.
        candidate = vectors @ ((vectors.conj().T @ rhs.data @ vectors) / denominators) @ vectors.conj().T
        candidate = (candidate + candidate.conj().T) * 0.5
        approximate = Ball(candidate)
        residual = matrix @ approximate + approximate @ matrix - rhs
        error = div_up(residual.norm(), 2 * lower)
        record("largest_sylvester_equation_residual", residual.norm())
        record("largest_sylvester_radius", error)
        return Ball(candidate, error)

    return solve


class Taylor3:
    """Single-ruler Taylor coefficients, with optional whole-box balls."""
    __array_priority__ = 30000

    def __init__(self, value, coefficients=None):
        zero = value if isinstance(value, Ball) else Ball(value)
        if coefficients is None:
            self.coefficients = [zero] + [Ball(XP.zeros_like(zero.data)) for _ in range(3)]
        else:
            self.coefficients = [zero] + [x if isinstance(x, Ball) else Ball(x) for x in coefficients]
            if len(self.coefficients) != 4:
                raise ValueError("Taylor3 needs exactly four coefficients.")

    @classmethod
    def from_coefficients(cls, coefficients):
        return cls(coefficients[0], coefficients[1:])

    @property
    def value(self):
        return self.coefficients[0].data

    def derivative(self, order):
        return self.coefficients[order] * math.factorial(order)

    def coerce(self, value):
        return value if isinstance(value, Taylor3) else Taylor3(value)

    def map(self, function, gain=None):
        return Taylor3.from_coefficients([x.map(function, gain) for x in self.coefficients])

    def dagger(self):
        return Taylor3.from_coefficients([x.dagger() for x in self.coefficients])

    def hermitian(self):
        return Taylor3.from_coefficients([x.hermitian() for x in self.coefficients])

    def __neg__(self):
        return Taylor3.from_coefficients([-x for x in self.coefficients])

    def __add__(self, other):
        other = self.coerce(other)
        return Taylor3.from_coefficients([a + b for a, b in zip(self.coefficients, other.coefficients)])

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-self.coerce(other))

    def __rsub__(self, other):
        return self.coerce(other) - self

    def product(self, other, operation):
        other = self.coerce(other)
        out = []
        for degree in range(4):
            terms = [operation(self.coefficients[j], other.coefficients[degree - j]) for j in range(degree + 1)]
            value = terms[0]
            for term in terms[1:]:
                value = value + term
            out.append(value)
        return Taylor3.from_coefficients(out)

    def __mul__(self, other):
        return self.product(other, lambda a, b: a * b)

    __rmul__ = __mul__

    def __matmul__(self, other):
        return self.product(other, lambda a, b: a @ b)

    def __rmatmul__(self, other):
        return self.coerce(other) @ self

    def __truediv__(self, other):
        other = self.coerce(other)
        return self * other.elementwise_inverse()

    def elementwise_inverse(self):
        value = self.coefficients[0]
        if value.data.ndim == 0:
            zeroth = value.scalar_inverse()
        elif value.data.ndim == 1:
            # A Euclidean radius bounds each entry. The exact values are
            # enclosed independently and then enclosed in a vector ball.
            values = host(value.data)
            entry_balls = [Ball.interval(sub_down(float(x.real), value.radius), add_up(float(x.real), value.radius)).scalar_inverse()
                           for x in values]
            zeroth = Ball(XP.asarray([x.data.item() for x in entry_balls]),
                          sqrt_up(add_up(*(mul_up(x.radius, x.radius) for x in entry_balls))))
        else:
            raise ValueError("Elementwise inverse is restricted to real fields/scalars.")
        out = [zeroth]
        for degree in range(1, 4):
            rhs = self.coefficients[1] * out[degree - 1]
            for j in range(2, degree + 1):
                rhs = rhs + self.coefficients[j] * out[degree - j]
            out.append(-(zeroth * rhs))
        return Taylor3.from_coefficients(out)

    def __pow__(self, power):
        if not isinstance(power, int):
            raise ValueError("Only integer scalar/field powers are needed by this original-A gate.")
        if power < 0:
            return self.elementwise_inverse() ** -power
        result = Taylor3(XP.ones_like(self.value))
        for _ in range(power):
            result = result * self
        return result

    def matrix_power(self, power):
        result = Taylor3(XP.eye(len(self.value)))
        for _ in range(power):
            result = result @ self
        return result

    def inverse(self):
        zeroth = self.coefficients[0].inverse()
        out = [zeroth]
        for degree in range(1, 4):
            rhs = self.coefficients[1] @ out[degree - 1]
            for j in range(2, degree + 1):
                rhs = rhs + self.coefficients[j] @ out[degree - j]
            out.append(-(zeroth @ rhs))
        return Taylor3.from_coefficients(out)

    def positive_root(self):
        if self.value.ndim == 0:
            zeroth = self.coefficients[0].scalar_sqrt()
            out = [zeroth]
            for degree in range(1, 4):
                rhs = self.coefficients[degree]
                for j in range(1, degree):
                    rhs = rhs - out[j] * out[degree - j]
                out.append(rhs / (2 * zeroth))
            return Taylor3.from_coefficients(out)
        zeroth = self.coefficients[0].positive_root()
        solve = positive_solver(zeroth, zeroth.lower)
        out = [zeroth]
        for degree in range(1, 4):
            rhs = self.coefficients[degree]
            for j in range(1, degree):
                rhs = rhs - out[j] @ out[degree - j]
            out.append(solve(rhs))
        return Taylor3.from_coefficients(out)

    def exponential(self, tolerance=1e-18, maximum_degree=180, delta=False):
        """Differentiated series with a weighted formal-variable tail.

        For a positive weight w, let sigma=sum ||A_k|| w^k. Every
        coefficient of the omitted exponential is bounded by
        exp(sigma)*sigma^(K+1)/(K+1)! / w^k. This covers k=0..3,
        including the noncommuting derivative orders. Scalar geometry
        uses a separate correctly-rounded exponential recurrence.
        """
        if self.value.ndim == 0:
            a = self.coefficients
            e0 = a[0].scalar_exp()
            out = [e0]
            for degree in range(1, 4):
                rhs = a[1] * out[degree - 1]
                for j in range(2, degree + 1):
                    rhs = rhs + j * a[j] * out[degree - j]
                out.append(rhs / degree)
            answer = Taylor3.from_coefficients(out)
            return answer - 1 if delta else answer
        derivative_sum = add_up(*(x.norm() for x in self.coefficients[1:]))
        weight = min(1.0, div_up(1, add_up(1, derivative_sum)))
        # Choose a dyadic weight <= the computed bound. This makes all
        # weight powers exact, and reduces the derivative majorant.
        weight = 2.0 ** math.floor(math.log2(weight))
        sigma = add_up(*(mul_up(x.norm(), weight ** k) for k, x in enumerate(self.coefficients)))
        identity = Taylor3(XP.eye(len(self.value)))
        term = identity
        result = Taylor3(XP.zeros_like(self.value)) if delta else identity
        degree = 0
        tails = [math.inf] * 4
        while degree < maximum_degree:
            degree += 1
            term = (term @ self) / degree
            result = result + term
            with localcontext() as context:
                context.prec = DECIMAL_DIGITS
                context.rounding = ROUND_CEILING
                tail_base = decimal(exp_up(sigma)) * decimal(sigma) ** (degree + 1) / decimal(math.factorial(degree + 1))
                tails = [float_up(tail_base / decimal(weight) ** k) for k in range(4)]
            if max(tails) < tolerance:
                break
        else:
            raise ArithmeticError("Differentiated exponential tail did not close.")
        out = [Ball(x.data, add_up(x.radius, tail)) for x, tail in zip(result.coefficients, tails)]
        record("largest_exponential_degree", degree)
        record("largest_exponential_tail", max(tails))
        return Taylor3.from_coefficients(out)


def sylvester_taylor(matrix: Taylor3, rhs: Taylor3, lower):
    solve = positive_solver(matrix.coefficients[0], lower)
    out = [solve(rhs.coefficients[0])]
    for degree in range(1, 4):
        target = rhs.coefficients[degree]
        for j in range(1, degree + 1):
            target = target - matrix.coefficients[j] @ out[degree - j] - out[degree - j] @ matrix.coefficients[j]
        out.append(solve(target))
    return Taylor3.from_coefficients(out)


def congruence_from_delta(delta: Taylor3, form: Taylor3):
    """(I+delta) form (I+delta), avoiding a large identity product."""
    return (form + delta @ form + form @ delta + delta @ form @ delta).hermitian()


def trace_block(matrix: Ball, first, size=2):
    diagonal = np.diag(host(matrix.data[first:first + size, first:first + size]))
    with localcontext() as context:
        context.prec = DECIMAL_DIGITS
        context.rounding = ROUND_FLOOR
        real_low = sum((decimal(x.real) for x in diagonal), Decimal(0))
        imag_low = sum((decimal(x.imag) for x in diagonal), Decimal(0))
        context.rounding = ROUND_CEILING
        real_high = sum((decimal(x.real) for x in diagonal), Decimal(0))
        imag_high = sum((decimal(x.imag) for x in diagonal), Decimal(0))
    real, imag = Ball.interval(real_low, real_high), Ball.interval(imag_low, imag_high)
    center = real.data + 1j * imag.data
    error = sqrt_up(add_up(mul_up(real.radius, real.radius), mul_up(imag.radius, imag.radius)))
    return Ball(center, add_up(mul_up(size, matrix.radius), error))


def scalar_real(value: Ball):
    return Ball(value.data.real, add_up(value.radius, abs(_scalar(value.data.imag))))


def scalar_abs_upper(value: Ball):
    return value.norm()


def normalized_vector(candidate):
    """Normalize exact binary64 coordinates using directed Decimal sums."""
    candidate = host(candidate).reshape(-1)
    with localcontext() as context:
        context.prec = DECIMAL_DIGITS
        context.rounding = ROUND_FLOOR
        low = sum((decimal(x.real) ** 2 + decimal(x.imag) ** 2 for x in candidate), Decimal(0))
        context.rounding = ROUND_CEILING
        high = sum((decimal(x.real) ** 2 + decimal(x.imag) ** 2 for x in candidate), Decimal(0))
    norm = Ball.interval(sqrt_down(low), sqrt_up(high))
    return Ball(XP.asarray(candidate[:, None])) / norm


def expectation(vector: Ball, matrix: Ball):
    # Retain the actual center action in the state-error terms. Ha u is
    # small in this boundary-dominated diagnostic even when ||Ha|| is
    # large; replacing it by ||Ha|| would lose the source-scale radius.
    point = Ball(vector.data)
    operator = Ball(matrix.data)
    action = operator @ point
    mean = scalar_real((point.dagger() @ action).map(lambda x: x.reshape(()), gain=1))
    if vector.radius:
        adjoint_action = operator.dagger() @ point
        state_error = add_up(mul_up(add_up(action.norm(), adjoint_action.norm()), vector.radius),
                             mul_up(operator.norm(), vector.radius, vector.radius))
    else:
        state_error = 0.0
    assembly_error = mul_up(matrix.radius, add_up(point.norm(), vector.radius), add_up(point.norm(), vector.radius))
    return Ball(mean.data, add_up(mean.radius, state_error, assembly_error))


def squared_norm(vector: Ball):
    if vector.data.ndim > 2 or (vector.data.ndim == 2 and min(vector.data.shape) != 1):
        raise ValueError("squared_norm requires a vector.")
    entries = host(vector.data).reshape(-1)
    with localcontext() as context:
        context.prec = DECIMAL_DIGITS
        context.rounding = ROUND_FLOOR
        low = sum((decimal(x.real) ** 2 + decimal(x.imag) ** 2 for x in entries), Decimal(0))
        context.rounding = ROUND_CEILING
        high = sum((decimal(x.real) ** 2 + decimal(x.imag) ** 2 for x in entries), Decimal(0))
    point = Ball.interval(low, high)
    error = add_up(mul_up(2, sqrt_up(high), vector.radius), mul_up(vector.radius, vector.radius))
    return Ball(point.data, add_up(point.radius, error))


def validate_eigenpair(hamiltonian: Ball, candidate, second_level_lower):
    u = normalized_vector(candidate)
    rayleigh = expectation(u, hamiltonian)
    rayleigh_low, rayleigh_high = rayleigh.bounds()
    separation = sub_down(second_level_lower, rayleigh_high)
    if separation <= 0:
        raise ArithmeticError("Rayleigh value does not isolate the ground state.")
    residual = hamiltonian @ u - u * rayleigh
    residual_norm = residual.norm()
    correction = div_up(mul_up(residual_norm, residual_norm), separation)
    energy = Ball.interval(sub_down(rayleigh_low, correction), rayleigh_high)
    vector_error = div_up(2 * residual_norm, separation)
    state = Ball(u.data, add_up(u.radius, vector_error))
    return {"state": state, "energy": energy, "rayleigh": rayleigh,
            "eigenpair_residual": residual_norm, "vector_error": vector_error,
            "second_level_separation": separation}


def validate_response(hamiltonian: Ball, vertex: Ball, state: Ball,
                      energy: Ball, candidate, gap):
    """True projected response residual, with the projection errors intact."""
    chi = Ball(XP.asarray(candidate).reshape(-1, 1))
    slope = expectation(state, vertex)
    orthogonality = (state.dagger() @ chi).norm()
    # P(H-eps)chi +(Ha-eps_a)psi; applying P costs <=1 in the
    # exact ground projection. Including the unprojected residual is safe.
    residual = hamiltonian @ chi - chi * energy + vertex @ state - state * slope
    error = add_up(div_up(residual.norm(), gap), orthogonality)
    return Ball(chi.data, error), {"true_response_residual": residual.norm(),
                                   "orthogonality_error": orthogonality,
                                   "response_radius": error}


def verify_backend():
    """Exercise typed kernel paths against exact dyadic reference products.

    This runtime check detects layout, compiler, dtype and dispatch faults.
    The enclosure itself uses the source-level elementary-operation budgets;
    sample agreement is never the error bound or the nonlinear certificate.
    """
    first = np.array([[1 + 0.5j, 2 ** -20 - 1j, -1 + 2j],
                      [2 ** 30 + 1j, -2 ** 30 - 0.5j, 0.25j]], complex)
    second = np.array([[0.5 - 1j, 2 + 0.25j],
                       [0.5 + 1j, 2 - 0.25j],
                       [1 - 0.5j, -1 + 0.5j]], complex)
    checked = Ball(first) @ Ball(second)
    center = host(checked.data)
    errors = []
    with localcontext() as context:
        context.prec = max(100, DECIMAL_DIGITS)
        context.rounding = ROUND_CEILING
        for row in range(first.shape[0]):
            for column in range(second.shape[1]):
                real, imag = Decimal(0), Decimal(0)
                for k in range(first.shape[1]):
                    ar, ai = decimal(first[row, k].real), decimal(first[row, k].imag)
                    br, bi = decimal(second[k, column].real), decimal(second[k, column].imag)
                    real += ar * br - ai * bi
                    imag += ar * bi + ai * br
                errors.append((decimal(center[row, column].real) - real) ** 2
                              + (decimal(center[row, column].imag) - imag) ** 2)
        reference_error = sqrt_up(sum(errors, Decimal(0)))
    if reference_error > checked.radius:
        raise ArithmeticError("Deterministic complex product backend verification failed.")
    vector = Ball(first[:, :1])
    narrow = vector.dagger() @ vector
    with localcontext() as context:
        context.prec = max(100, DECIMAL_DIGITS)
        reference = sum((decimal(x.real) ** 2 + decimal(x.imag) ** 2 for x in first[:, 0]), Decimal(0))
        difference = abs(decimal(host(narrow.data)[0, 0].real) - reference)
    if float_up(difference) > narrow.radius:
        raise ArithmeticError("Explicit longdouble product backend verification failed.")
    matrix = XP.asarray(np.diag([3 + 4j, -2 - 2j]))
    row_sums, column_sums = _deterministic_norm_sums(matrix)
    if host(row_sums).tolist() != [7.0, 4.0] or host(column_sums).tolist() != [7.0, 4.0]:
        raise ArithmeticError("Deterministic row/column reduction verification failed.")
    if norm_upper(matrix) < 5:
        raise ArithmeticError("The deterministic operator norm bound failed.")
    # Exercise the exact production size, including the final partial
    # CUDA block and a noncontiguous transpose. The dyadic sparse matrix
    # times the identity has an exact binary64 result, independently known
    # entry by entry. No vendor product is used for this prerequisite.
    size = 1431
    index = XP.arange(size)
    large = XP.zeros((size, size), dtype=XP.complex128)
    large[index, index] = 1 + 0.5j
    large[index, (index + 1) % size] = 2 ** -20 - 0.25j
    strided = large.T
    large_product = _deterministic_product(strided, XP.eye(size, dtype=XP.complex128))
    _finite(large_product)
    if not bool(XP.all(large_product == strided).item()):
        raise ArithmeticError("The1431 strided/tail-block exact identity product check failed.")
    row_sums, column_sums = _deterministic_norm_sums(strided)
    exact_sum = 1.75 + 2 ** -20
    if not bool(XP.all(row_sums == exact_sum).item()) or not bool(XP.all(column_sums == exact_sum).item()):
        raise ArithmeticError("The1431 strided/tail-block norm reduction check failed.")
    if norm_upper(strided) < exact_sum:
        raise ArithmeticError("The1431 operator-norm bound failed its exact row/column majorant.")
    record("production_shape_kernel_verified", size)
    record("arithmetic_backend_verified", 1)
    return {"passed": True, "backend": "numpy" if XP is np else "cupy",
            "binary64_significand_bits": int(np.finfo(np.float64).nmant) + 1,
            "longdouble_significand_bits": int(np.finfo(np.longdouble).nmant) + 1,
            "complex_dot_operations_per_inner_step": 8,
            "norm_sum_additions_per_entry": 2,
            "production_shape": [size, size],
            "strided_and_partial_block_checks_passed": True,
            "cuda_options": [] if XP is np else ["--fmad=false", "--prec-div=true", "--prec-sqrt=true", "-ftz=true (CuPy-added; bounded absolute underflow)"],
            "exact_reference_product_error": reference_error,
            "product_roundoff_bound": checked.radius}
