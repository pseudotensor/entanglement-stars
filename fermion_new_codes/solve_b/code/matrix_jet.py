"""Matrix derivatives used by the reviewed C finite-sector experiment.

This is implementation support for the prescribed Sylvester/Frechet recipe,
not a new source law. Execute only in the remote qi2 environment.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, expm_frechet


class Jet:
    """Value, first derivatives, and unscaled second derivatives."""

    __array_priority__ = 10000

    def __init__(self, value, first=None, second=None, dimension=4):
        self.value = np.asarray(value, dtype=complex)
        self.dimension = dimension
        self.first = np.broadcast_to(np.zeros((), complex), (dimension,) + self.value.shape) if first is None else np.asarray(first, complex)
        self.second = np.broadcast_to(np.zeros((), complex), (dimension, dimension) + self.value.shape) if second is None else np.asarray(second, complex)
        self.active_first = np.zeros(dimension, bool) if first is None else np.array([bool(np.any(x)) for x in self.first])
        self.active_second = np.zeros((dimension, dimension), bool) if second is None else np.array([[bool(np.any(x)) for x in row] for row in self.second])

    @property
    def is_constant(self):
        return not (np.any(self.active_first) or np.any(self.active_second))

    def constant(self):
        return Jet(self.value, dimension=self.dimension)

    def coerce(self, other):
        return other if isinstance(other, Jet) else Jet(other, dimension=self.dimension)

    def map(self, function):
        """Apply a linear matrix/field map; nonlinear maps have named methods."""
        value = function(self.value)
        if self.is_constant:
            return Jet(value, dimension=self.dimension)
        return Jet(value,
                   np.array([function(x) if self.active_first[a] else np.zeros_like(value) for a, x in enumerate(self.first)]),
                   np.array([[function(x) if self.active_second[a, b] else np.zeros_like(value) for b, x in enumerate(row)]
                             for a, row in enumerate(self.second)]), self.dimension)

    def dagger(self):
        return self.map(lambda x: x.conj().T)

    def hermitian(self):
        return (self + self.dagger()) * 0.5

    def __add__(self, other):
        other = self.coerce(other)
        if self.is_constant and other.is_constant:
            return Jet(self.value + other.value, dimension=self.dimension)
        return Jet(self.value + other.value,
                   np.array([self.first[a] + other.first[a] for a in range(self.dimension)]),
                   np.array([[self.second[a, b] + other.second[a, b] for b in range(self.dimension)]
                             for a in range(self.dimension)]), self.dimension)

    __radd__ = __add__

    def __neg__(self):
        if self.is_constant:
            return Jet(-self.value, dimension=self.dimension)
        return Jet(-self.value, -self.first, -self.second, self.dimension)

    def __sub__(self, other):
        return self + (-self.coerce(other))

    def __rsub__(self, other):
        return self.coerce(other) - self

    def product(self, other, operator):
        other = self.coerce(other)
        value = operator(self.value, other.value)
        if self.is_constant and other.is_constant:
            return Jet(value, dimension=self.dimension)
        first = np.zeros((self.dimension,) + value.shape, complex)
        second = np.zeros((self.dimension, self.dimension) + value.shape, complex)
        for a in range(self.dimension):
            if self.active_first[a]:
                first[a] += operator(self.first[a], other.value)
            if other.active_first[a]:
                first[a] += operator(self.value, other.first[a])
            for b in range(self.dimension):
                if self.active_second[a, b]:
                    second[a, b] += operator(self.second[a, b], other.value)
                if self.active_first[a] and other.active_first[b]:
                    second[a, b] += operator(self.first[a], other.first[b])
                if self.active_first[b] and other.active_first[a]:
                    second[a, b] += operator(self.first[b], other.first[a])
                if other.active_second[a, b]:
                    second[a, b] += operator(self.value, other.second[a, b])
        return Jet(value, first, second, self.dimension)

    def __mul__(self, other):
        return self.product(other, np.multiply)

    __rmul__ = __mul__

    def __matmul__(self, other):
        return self.product(other, np.matmul)

    def __rmatmul__(self, other):
        return self.coerce(other) @ self

    def __truediv__(self, other):
        if isinstance(other, Jet):
            return self * other ** -1
        return self * (1 / other)

    def __pow__(self, power):
        """Elementwise power, used only on scalar fields/weights."""
        value = self.value ** power
        if self.is_constant:
            return Jet(value, dimension=self.dimension)
        slope = power * self.value ** (power - 1)
        curvature = power * (power - 1) * self.value ** (power - 2)
        first = np.array([slope * self.first[a] for a in range(self.dimension)])
        second = np.array([[slope * self.second[a, b] + curvature * self.first[a] * self.first[b]
                            for b in range(self.dimension)] for a in range(self.dimension)])
        return Jet(value, first, second, self.dimension)

    def matrix_power(self, power):
        result = Jet(np.eye(self.value.shape[0]), dimension=self.dimension)
        for _ in range(power):
            result = result @ self
        return result

    def inverse(self):
        value = np.linalg.inv(self.value)
        if self.is_constant:
            return Jet(value, dimension=self.dimension)
        first = np.zeros_like(self.first)
        second = np.zeros_like(self.second)
        for a in range(self.dimension):
            if self.active_first[a]:
                first[a] = -value @ self.first[a] @ value
            for b in range(self.dimension):
                if self.active_first[a] and self.active_first[b]:
                    second[a, b] += value @ self.first[b] @ value @ self.first[a] @ value
                    second[a, b] += value @ self.first[a] @ value @ self.first[b] @ value
                if self.active_second[a, b]:
                    second[a, b] -= value @ self.second[a, b] @ value
        return Jet(value, first, second, self.dimension)

    def positive_root(self):
        values, vectors = np.linalg.eigh(self.value)
        if values[0] <= 0:
            raise ValueError(f"Positive root lost its domain: minimum {values[0]}")
        value = (vectors * np.sqrt(values)) @ vectors.conj().T
        if self.is_constant:
            return Jet(value, dimension=self.dimension)
        solve = positive_sylvester_solver(value)
        first = np.zeros_like(self.first)
        second = np.zeros_like(self.second)
        for a in range(self.dimension):
            if self.active_first[a]:
                first[a] = solve(self.first[a])
        for a in range(self.dimension):
            for b in range(self.dimension):
                if self.active_second[a, b] or (self.active_first[a] and self.active_first[b]):
                    second[a, b] = solve(self.second[a, b] - first[a] @ first[b] - first[b] @ first[a])
        return Jet(value, first, second, self.dimension).hermitian()

    def exponential(self):
        if self.is_constant:
            return Jet(np.exp(self.value) if self.value.ndim == 0 else expm(self.value), dimension=self.dimension)
        if self.value.ndim == 0:
            value = np.exp(self.value)
            return Jet(value, value * self.first,
                       value * (self.second + self.first[:, None] * self.first[None, :]), self.dimension)
        n = self.value.shape[0]
        value = expm(self.value)
        first = np.zeros_like(self.first)
        second = np.zeros_like(self.second)
        active = [bool(np.any(x)) for x in self.first]
        for a in range(self.dimension):
            if active[a]:
                first[a] = expm_frechet(self.value, self.first[a], compute_expm=False, check_finite=False)
            for b in range(a + 1):
                derivative = np.zeros_like(value)
                if np.any(self.second[a, b]):
                    derivative += expm_frechet(self.value, self.second[a, b], compute_expm=False, check_finite=False)
                if active[a] and active[b]:
                    derivative += self.ordered_exponential_second(a, b)
                    derivative += self.ordered_exponential_second(b, a)
                second[a, b] = second[b, a] = derivative
        return Jet(value, first, second, self.dimension)

    def ordered_exponential_second(self, a, b):
        n = self.value.shape[0]
        block = np.zeros((3 * n, 3 * n), complex)
        for k in range(3):
            block[k * n:(k + 1) * n, k * n:(k + 1) * n] = self.value
        block[:n, n:2 * n] = self.first[a]
        block[n:2 * n, 2 * n:] = self.first[b]
        return expm(block)[:n, 2 * n:]


def positive_sylvester_solver(matrix):
    values, vectors = np.linalg.eigh(matrix)
    if values[0] <= 0:
        raise ValueError("Unit-lapse Sylvester denominator is not positive.")
    denominator = values[:, None] + values[None, :]

    def solve(rhs):
        return vectors @ ((vectors.conj().T @ rhs @ vectors) / denominator) @ vectors.conj().T

    return solve


def sylvester_jet(matrix, rhs):
    """Differentiate {matrix,S}=rhs with the perturbed unit-lapse matrix."""
    solve = positive_sylvester_solver(matrix.value)
    value = solve(rhs.value)
    if matrix.is_constant and rhs.is_constant:
        return Jet(value, dimension=matrix.dimension).hermitian()
    first = np.zeros_like(rhs.first)
    second = np.zeros_like(rhs.second)
    active = matrix.active_first | rhs.active_first
    for a in range(matrix.dimension):
        if active[a]:
            derivative = rhs.first[a].copy()
            if matrix.active_first[a]:
                derivative -= matrix.first[a] @ value + value @ matrix.first[a]
            first[a] = solve(derivative)
    for a in range(matrix.dimension):
        for b in range(matrix.dimension):
            if (rhs.active_second[a, b] or matrix.active_second[a, b]
                    or (matrix.active_first[a] and active[b]) or (matrix.active_first[b] and active[a])):
                derivative = rhs.second[a, b].copy()
                if matrix.active_second[a, b]:
                    derivative -= matrix.second[a, b] @ value + value @ matrix.second[a, b]
                if matrix.active_first[a] and active[b]:
                    derivative -= matrix.first[a] @ first[b] + first[b] @ matrix.first[a]
                if matrix.active_first[b] and active[a]:
                    derivative -= matrix.first[b] @ first[a] + first[a] @ matrix.first[b]
                second[a, b] = solve(derivative)
    return Jet(value, first, second, matrix.dimension).hermitian()
