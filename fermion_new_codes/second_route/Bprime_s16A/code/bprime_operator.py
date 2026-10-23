"""Own canonical four-spin cubic links. Import only after CPU admission."""
from __future__ import annotations

import numpy as np
from numba import njit, prange

import hashlib


def canonical_identity(array):
    array = np.ascontiguousarray(array)
    if array.dtype.hasobject:
        raise ValueError('Object arrays are not admitted')
    view = memoryview(array).cast('B')
    sha, md5 = hashlib.sha256(), hashlib.md5()
    for start in range(0, len(view), 8*1024*1024):
        sha.update(view[start:start+8*1024*1024])
        md5.update(view[start:start+8*1024*1024])
    return {'shape': list(array.shape), 'dtype': array.dtype.str, 'traversal': 'logical_C',
            'sha256': sha.hexdigest(), 'md5': md5.hexdigest()}


def require_exact(left, right, label):
    if left.shape != right.shape or left.dtype != right.dtype or not np.array_equal(left, right):
        raise ValueError('Exact logical input association failed: '+label)
    return True


def embed_subcube(columns, H, Hs):
    columns = np.asarray(columns)
    width = 2*Hs+1
    if columns.dtype != np.complex128 or columns.ndim != 2 or columns.shape[0] != 4*width**3 or Hs > H:
        raise ValueError('Invalid spin-major subcube basis shape')
    result = np.zeros((columns.shape[1], 4, 2*H+1, 2*H+1, 2*H+1), dtype=np.complex128)
    sl = slice(H-Hs, H+Hs+1)
    result[:, :, sl, sl, sl] = columns.T.reshape(columns.shape[1], 4, width, width, width)
    return result


def extract_subcube(states, H, Hs):
    states = np.asarray(states)
    if states.ndim != 5 or states.shape[1:] != (4, 2*H+1, 2*H+1, 2*H+1):
        raise ValueError('Wrong full-box spin-major state shape')
    sl = slice(H-Hs, H+Hs+1)
    return states[:, :, sl, sl, sl].reshape(len(states), -1)


def exact_clock_association(times, ratios, L):
    times, ratios = np.asarray(times), np.asarray(ratios)
    if times.ndim != 1 or ratios.shape != times.shape or not len(times):
        raise ValueError('Missing or malformed source clocks')
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(ratios)) or np.any(np.diff(times) <= 0):
        raise ValueError('Nonfinite, duplicate or unordered source clocks')
    if not np.array_equal(times, ratios*L):
        raise ValueError('Source clock differs from exact declared endpoint')
    return float(times[0])


def spin_matrices():
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = np.diag([1., -1.]).astype(np.complex128)
    alpha = np.asarray([np.kron(sx, sigma) for sigma in (sx, sy, sz)])
    beta = np.kron(sz, np.eye(2))
    uc = np.kron(sy, sy)
    return alpha, beta, uc


def antiunitary(states):
    states = np.asarray(states, dtype=np.complex128)
    if states.ndim < 3 or states.shape[1] != 4:
        raise ValueError('Antiunitary expects column,spin,spatial axes')
    result = states[:, [3, 2, 1, 0]].conj().copy()
    result[:, 0] *= -1
    result[:, 3] *= -1
    return result


@njit(parallel=True, cache=True, fastmath=False)
def _apply(states, output, lapse, eta, mass, neighbors, one, two, spin_index, spin_value, hamiltonian_sign):
    count, _, sites = states.shape
    # Each task owns one column's disjoint site tile; local arithmetic is unchanged.
    tile_size = 4096
    tiles_per_column = (sites+tile_size-1)//tile_size
    for tile in prange(count*tiles_per_column):
        column = tile//tiles_per_column
        start = (tile % tiles_per_column)*tile_size
        stop = min(start+tile_size, sites)
        for site in range(start, stop):
            for spin in range(4):
                beta = 1. if spin < 2 else -1.
                value = -1j*mass*lapse[site]*beta*states[column, spin, site]
                for axis in range(3):
                    other_spin = spin_index[axis, spin]
                    phase = spin_value[axis, spin]
                    for side in range(2):
                        near = neighbors[axis, side, site]
                        far = neighbors[axis, side+2, site]
                        if near >= 0:
                            value += one[axis, side, site]*phase*states[column, other_spin, near]
                        if far >= 0:
                            value += two[axis, side, site]*states[column, spin, far]
                output[column, spin, site] = hamiltonian_sign*value-eta[site]*states[column, spin, site]


class NativeOperator:
    """A=-ih-eta, spin-major states (columns,4,n,n,n), open unless fixture periodic."""
    def __init__(self, lapse, flow, eta, mass=0., spacing=1., periodic=False):
        arrays = [np.asarray(value) for value in (lapse, flow, eta)]
        if any(np.iscomplexobj(value) or value.dtype.hasobject or not np.issubdtype(value.dtype, np.number) for value in arrays):
            raise ValueError('Native fields must be real numeric arrays')
        self.lapse, self.flow, self.eta = [np.ascontiguousarray(value, dtype=np.float64) for value in arrays]
        self.shape = self.lapse.shape
        if len(self.shape) != 3 or len(set(self.shape)) != 1 or self.shape[0] < 5:
            raise ValueError('Native operator needs a cubic grid with at least five sites per axis')
        if self.flow.shape != (3,)+self.shape or self.eta.shape != self.shape:
            raise ValueError('Native field shapes differ')
        if not all(np.all(np.isfinite(value)) for value in (self.lapse, self.flow, self.eta)):
            raise ValueError('Nonfinite native fields')
        if np.any(self.lapse <= 0) or np.any(self.eta < 0):
            raise ValueError('Nonpositive lapse or negative sink')
        self.mass, self.spacing, self.periodic = float(mass), float(spacing), bool(periodic)
        if not np.isfinite(self.mass) or not np.isfinite(self.spacing) or self.spacing <= 0:
            raise ValueError('Malformed mass or lattice spacing')
        n, sites = self.shape[0], self.lapse.size
        self.neighbors = np.full((3, 4, sites), -1, dtype=np.int32)
        self.one = np.zeros((3, 2, sites), dtype=np.float64)
        self.two = np.zeros_like(self.one)
        indices = np.arange(sites)
        coordinates = np.indices(self.shape).reshape(3, sites)
        strides = (n*n, n, 1)
        nf, vf = self.lapse.ravel(), self.flow.reshape(3, sites)
        for axis in range(3):
            for distance in (1, 2):
                for side, direction in enumerate((1, -1)):
                    target_coordinate = coordinates[axis]+direction*distance
                    valid = np.ones(sites, dtype=bool) if periodic else (target_coordinate >= 0) & (target_coordinate < n)
                    target_coordinate = target_coordinate % n
                    target = indices+(target_coordinate-coordinates[axis])*strides[axis]
                    slot = side if distance == 1 else side+2
                    self.neighbors[axis, slot, valid] = target[valid]
                    coefficient = self.one if distance == 1 else self.two
                    field = nf if distance == 1 else vf[axis]
                    denominator = (4. if distance == 1 else 8.)*self.spacing
                    coefficient[axis, side, valid] = -direction*(field[valid]+field[target[valid]])/denominator
        alpha, _, _ = spin_matrices()
        self.spin_index = np.argmax(np.abs(alpha), axis=2).astype(np.int64)
        self.spin_value = np.take_along_axis(alpha, self.spin_index[:, :, None], axis=2)[:, :, 0].copy()
        diagonal_magnitude = np.hypot(self.eta.ravel(), self.mass*nf)
        row = diagonal_magnitude.copy()
        column = diagonal_magnitude.copy()
        for axis in range(3):
            for slot in range(4):
                coefficient = self.one[axis, slot] if slot < 2 else self.two[axis, slot-2]
                magnitude = np.abs(coefficient)
                row += magnitude
                valid = self.neighbors[axis, slot] >= 0
                np.add.at(column, self.neighbors[axis, slot, valid], magnitude[valid])
        self.row_absolute_sums, self.column_absolute_sums = row, column
        self.norm_infinity, self.norm_one = float(np.max(row)), float(np.max(column))
        self.B = float(np.sqrt(self.norm_infinity*self.norm_one))
        if not np.isfinite(self.B):
            raise ValueError('Nonfinite assembled operator bound')

    def apply(self, states, out=None, adjoint=False):
        states = np.asarray(states)
        if states.dtype != np.complex128 or states.ndim != 5 or states.shape[1:] != (4,)+self.shape:
            raise ValueError('Native states must be complex128 columns,spin,x,y,z')
        states = np.ascontiguousarray(states)
        if out is None:
            out = np.empty_like(states)
        if out.shape != states.shape or out.dtype != np.complex128 or not out.flags.c_contiguous or np.shares_memory(states, out):
            raise ValueError('Generator output must be a separate canonical complex128 block')
        _apply(states.reshape(len(states), 4, -1), out.reshape(len(states), 4, -1),
               self.lapse.ravel(), self.eta.ravel(), self.mass, self.neighbors, self.one, self.two,
               self.spin_index, self.spin_value, -1. if adjoint else 1.)
        return out

    def h_apply(self, states):
        return 1j*(self.apply(states)+self.eta[None, None]*states)

    def adjoint(self):
        return AdjointGenerator(self)


class AdjointGenerator:
    def __init__(self, parent):
        self.parent, self.eta, self.B = parent, parent.eta, parent.B

    def apply(self, states, out=None):
        return self.parent.apply(states, out, adjoint=True)


class DenseGenerator:
    """Manufactured/reference generator only; no dense physical cube assembly."""
    def __init__(self, h, eta):
        self.h = np.asarray(h, dtype=np.complex128)
        eta = np.asarray(eta)
        self.eta = eta.astype(np.complex128) if eta.ndim == 2 else eta.astype(np.float64)
        sink = self.eta if eta.ndim == 2 else np.diag(self.eta)
        self.matrix = -1j*self.h-sink
        self.B = float(np.sqrt(np.linalg.norm(self.matrix, 1)*np.linalg.norm(self.matrix, np.inf)))

    def apply(self, states, out=None, adjoint=False):
        matrix = self.matrix.conj().T if adjoint else self.matrix
        value = np.asarray(states)@matrix.T
        if out is None:
            return value
        out[...] = value
        return out

    def adjoint(self):
        return AdjointGenerator(self)
