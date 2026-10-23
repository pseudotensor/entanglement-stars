"""Own sine/two-link native Hamiltonian; numerical imports follow CPU admission.

NativeHamiltonian.apply is the internal generator -iH.  The explicit Cube
compatibility adapter instead exposes Cube.apply = H, in native cube layout.
No Numba import or pool occurs until the first matrix-free generator apply.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np


_NATIVE_KERNEL = None


def canonical_identity(array):
    array = np.ascontiguousarray(array)
    if array.dtype.hasobject:
        raise ValueError('Object arrays are not admitted')
    view = memoryview(array).cast('B')
    sha, md5 = hashlib.sha256(), hashlib.md5()
    for start in range(0, len(view), 8*1024*1024):
        sha.update(view[start:start+8*1024*1024])
        md5.update(view[start:start+8*1024*1024])
    return {'shape': list(array.shape), 'dtype': array.dtype.str,
            'traversal': 'logical_C', 'sha256': sha.hexdigest(), 'md5': md5.hexdigest()}


def require_exact(left, right, label):
    if left.shape != right.shape or left.dtype != right.dtype or not np.array_equal(left, right):
        raise ValueError('Exact logical input association failed: '+label)
    return True


def spin_matrices():
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = np.diag([1., -1.]).astype(np.complex128)
    alpha = np.asarray([np.kron(sx, sigma) for sigma in (sx, sy, sz)])
    beta = np.kron(sz, np.eye(2))
    uc = np.kron(sy, sy)
    return alpha, beta, uc


def valley_matrix(nu):
    if tuple(nu) not in tuple(itertools.product((0, 1), repeat=3)):
        raise ValueError('Valley label must contain three bits')
    _, beta, _ = spin_matrices()
    sigma = (np.array([[0, 1], [1, 0]], complex),
             np.array([[0, -1j], [1j, 0]], complex), np.diag([1., -1.]))
    result = np.eye(4, dtype=np.complex128)
    for bit, matrix in zip(nu, sigma):
        if bit:
            result = result@np.kron(np.diag([1., -1.]), matrix)
    return result


def coordinates(H):
    if type(H) is not int or H < 0:
        raise ValueError('Half-side must be a nonnegative integer')
    return (np.indices((2*H+1,)*3, dtype=np.int64)-H).reshape(3, -1).T


def ball_mask(H, radius):
    points = coordinates(H)
    if type(radius) is not int or radius < 0:
        raise ValueError('Integer ball radius required for exact ownership')
    return (np.sum(points*points, axis=1) <= radius*radius).reshape((2*H+1,)*3)


class NativeLayout:
    def __init__(self, n):
        self.n, self.native_size = int(n), 4*int(n)**3
        self.total_dofs, self.cube_shape = self.native_size, (4,)+(int(n),)*3

    def native_view(self, states):
        states = np.asarray(states)
        if states.ndim != 2 or states.shape[1] != self.total_dofs:
            raise ValueError('Wrong flat native state layout')
        return states.reshape((len(states),)+self.cube_shape)

    def pack(self, native, lead=None):
        native = np.asarray(native, dtype=np.complex128)
        if lead is not None or native.ndim != 5 or native.shape[1:] != self.cube_shape:
            raise ValueError('Wrong native cube input shape')
        return np.ascontiguousarray(native.reshape(len(native), self.native_size))

    def metadata(self):
        return {'native_shape': list(self.cube_shape), 'native_size': self.native_size,
                'total_dofs': self.total_dofs, 'order': 'column,spin,x,y,z; z fastest'}


def _kernel():
    global _NATIVE_KERNEL
    if _NATIVE_KERNEL is None:
        from numba import njit, prange

        @njit(parallel=True, cache=True, fastmath=False)
        def apply_native(states, output, lapse, mass, neighbors, one, two, spin_index, spin_value):
            count, _, sites = states.shape
            tiles = (sites+4095)//4096
            for tile in prange(count*tiles):
                column, start = tile//tiles, (tile % tiles)*4096
                for site in range(start, min(start+4096, sites)):
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
                        output[column, spin, site] = value
        _NATIVE_KERNEL = apply_native
    return _NATIVE_KERNEL


class NativeHamiltonian:
    """Internal -iH generator, with direct H access and an exact site compression."""
    def __init__(self, lapse, flow, mass=0., spacing=1., mask=None, periodic=False):
        values = [np.asarray(value) for value in (lapse, flow)]
        if any(value.dtype.hasobject or np.iscomplexobj(value) or
               not np.issubdtype(value.dtype, np.number) for value in values):
            raise ValueError('Native N and vector v must be real numeric arrays')
        self.lapse, self.flow = [np.ascontiguousarray(value, dtype=np.float64) for value in values]
        self.shape = self.lapse.shape
        if len(self.shape) != 3 or len(set(self.shape)) != 1 or self.shape[0] < 5:
            raise ValueError('Native grid must be cubic with at least five sites per axis')
        if self.flow.shape != (3,)+self.shape or not all(np.all(np.isfinite(v)) for v in (self.lapse, self.flow)):
            raise ValueError('Malformed or nonfinite native fields')
        if np.any(self.lapse <= 0):
            raise ValueError('Lapse must be positive')
        self.mass, self.spacing = float(mass), float(spacing)
        if not np.isfinite(self.mass) or not np.isfinite(self.spacing) or self.spacing <= 0:
            raise ValueError('Invalid mass or spacing')
        self.mask = np.ones(self.shape, dtype=bool) if mask is None else np.asarray(mask)
        if self.mask.dtype != np.bool_ or self.mask.shape != self.shape:
            raise ValueError('Native compression requires a Boolean mask on the grid')
        self.mask = self.mask.copy()
        self.layout, self.periodic = NativeLayout(self.shape[0]), bool(periodic)
        self.is_unitary = True
        sites, n = self.lapse.size, self.shape[0]
        self.neighbors = np.full((3, 4, sites), -1, dtype=np.int32)
        self.one = np.zeros((3, 2, sites), dtype=np.float64)
        self.two = np.zeros_like(self.one)
        indices = np.arange(sites)
        coord = np.indices(self.shape).reshape(3, sites)
        nf, vf, kept = self.lapse.ravel(), self.flow.reshape(3, sites), self.mask.ravel()
        for axis, stride in enumerate((n*n, n, 1)):
            for distance in (1, 2):
                for side, direction in enumerate((1, -1)):
                    target_coordinate = coord[axis]+direction*distance
                    valid = np.ones(sites, bool) if periodic else (target_coordinate >= 0) & (target_coordinate < n)
                    target = indices+((target_coordinate % n)-coord[axis])*stride
                    valid &= kept & kept[target]
                    slot = side if distance == 1 else side+2
                    self.neighbors[axis, slot, valid] = target[valid]
                    coefficient = self.one if distance == 1 else self.two
                    field = nf if distance == 1 else vf[axis]
                    coefficient[axis, side, valid] = -direction*(field[valid]+field[target[valid]])/((4. if distance == 1 else 8.)*spacing)
        alpha, _, _ = spin_matrices()
        self.spin_index = np.argmax(np.abs(alpha), axis=2).astype(np.int64)
        self.spin_value = np.take_along_axis(alpha, self.spin_index[:, :, None], axis=2)[:, :, 0].copy()
        self.diagonal_lapse = np.where(kept, nf, 0.)
        row = np.abs(self.mass*self.diagonal_lapse)
        column = row.copy()
        for axis in range(3):
            for slot in range(4):
                coefficient = self.one[axis, slot] if slot < 2 else self.two[axis, slot-2]
                magnitude = np.abs(coefficient)
                row += magnitude
                valid = self.neighbors[axis, slot] >= 0
                np.add.at(column, self.neighbors[axis, slot, valid], magnitude[valid])
        self.norm_one, self.norm_infinity = float(np.max(column)), float(np.max(row))
        self.B = float(np.sqrt(self.norm_one*self.norm_infinity))
        if not np.isfinite(self.B):
            raise ValueError('Nonfinite native norm bound')
        outer = np.any((coord <= 1) | (coord >= n-2), axis=0) & kept
        self.end_collar_indices = np.flatnonzero(np.tile(outer, 4))

    def apply(self, states, out=None):
        states = np.asarray(states)
        if states.dtype != np.complex128 or states.ndim != 2 or states.shape[1] != self.layout.total_dofs:
            raise ValueError('Generator input requires flat complex128 columns')
        if states.strides[1] != states.itemsize:
            states = np.ascontiguousarray(states)
        out = np.empty_like(states) if out is None else out
        if out.dtype != np.complex128 or out.shape != states.shape or out.strides[1] != out.itemsize or np.shares_memory(states, out):
            raise ValueError('Generator output must be a separate block with contiguous inner coordinates')
        _kernel()(states.reshape(len(states), 4, -1), out.reshape(len(states), 4, -1),
                  self.diagonal_lapse, self.mass, self.neighbors, self.one, self.two, self.spin_index, self.spin_value)
        return out

    def h_apply(self, states):
        return 1j*self.apply(states)

    def cube_h_apply(self, states):
        states = np.asarray(states, dtype=np.complex128)
        shape = states.shape
        if shape[-4:] != self.layout.cube_shape:
            raise ValueError('Wrong native cube compatibility layout')
        return self.h_apply(np.ascontiguousarray(states.reshape(-1, self.layout.native_size))).reshape(shape)

    def sparse_block(self, target_dofs, source_dofs):
        """Exact restricted bonds for dense setup; no Numba import or pool."""
        from scipy.sparse import coo_matrix
        rows, cols = np.asarray(target_dofs, np.int64), np.asarray(source_dofs, np.int64)
        if rows.ndim != 1 or cols.ndim != 1 or len(np.unique(rows)) != len(rows) or len(np.unique(cols)) != len(cols):
            raise ValueError('Restricted native coordinates must be unique one-dimensional indices')
        if np.any(rows < 0) or np.any(cols < 0) or np.any(rows >= self.layout.native_size) or np.any(cols >= self.layout.native_size):
            raise ValueError('Restricted native index outside cube')
        lookup = np.full(self.layout.native_size, -1, dtype=np.int64)
        lookup[cols] = np.arange(len(cols))
        sites = self.lapse.size
        spins, points = rows//sites, rows % sites
        rr, cc, vv = [], [], []

        def add(global_columns, values):
            local = lookup[global_columns]
            valid = (local >= 0) & (values != 0)
            rr.append(np.flatnonzero(valid)); cc.append(local[valid]); vv.append(values[valid])

        add(rows, self.mass*self.diagonal_lapse[points]*np.where(spins < 2, 1., -1.))
        for axis in range(3):
            for side in range(2):
                near = self.neighbors[axis, side, points]
                far = self.neighbors[axis, side+2, points]
                phase = self.spin_value[axis, spins]
                add(self.spin_index[axis, spins]*sites+np.maximum(near, 0),
                    np.where(near >= 0, 1j*self.one[axis, side, points]*phase, 0.))
                add(spins*sites+np.maximum(far, 0), np.where(far >= 0, 1j*self.two[axis, side, points], 0.))
        return coo_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))), shape=(len(rows), len(cols))).tocsr()

    def valley_image(self, states, nu):
        native = self.layout.native_view(states)
        points = (np.indices(self.shape)-self.shape[0]//2).reshape(3, -1).T
        parity = np.where((points@np.asarray(nu, np.int64)) % 2, -1., 1.).reshape(self.shape)
        result = np.einsum('ab,cbxyz->caxyz', valley_matrix(nu), native)*parity[None, None]
        return result.reshape(len(states), -1)

    def antiunitary_image(self, states):
        native = self.layout.native_view(states)
        result = native[:, [3, 2, 1, 0]].conj().copy()
        result[:, 0] *= -1; result[:, 3] *= -1
        return result.reshape(len(states), -1)


class Cube:
    """Native compatibility adapter: apply(leading_batch,4,x,y,z) returns H psi."""
    def __init__(self, N, v, eta=None, device='cpu', mass=0., spacing=1., mask=None):
        if device != 'cpu' or eta is not None and np.any(np.asarray(eta) != 0):
            raise ValueError('Unitary Cube compatibility permits CPU and no sink only')
        self.operator = NativeHamiltonian(N, v, mass, spacing, mask)
        self.B, self.layout = self.operator.B, self.operator.layout

    def apply(self, psi):
        return self.operator.cube_h_apply(psi)


class DenseHamiltonian:
    """Independent manufactured matrix wrapper; CAP is explicit when supplied."""
    def __init__(self, matrix, eta=None):
        self.matrix = np.asarray(matrix, dtype=np.complex128)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1] or not np.all(np.isfinite(self.matrix)):
            raise ValueError('Dense fixture requires a finite square Hamiltonian')
        self.eta = np.zeros(len(self.matrix)) if eta is None else np.asarray(eta, dtype=np.float64)
        if self.eta.shape != (len(self.matrix),) or np.any(self.eta < 0) or not np.all(np.isfinite(self.eta)):
            raise ValueError('CAP fixture requires finite nonnegative eta')
        self.generator = -1j*self.matrix-np.diag(self.eta)
        self.B = float(np.sqrt(np.linalg.norm(self.generator, 1)*np.linalg.norm(self.generator, np.inf)))
        self.is_unitary = not np.any(self.eta)

    def apply(self, states, out=None):
        value = np.asarray(states)@self.generator.T
        if out is None:
            return value
        out[...] = value
        return out

    def h_apply(self, states):
        return np.asarray(states)@self.matrix.T

    def energy_balance(self, states, hstates=None):
        h = self.h_apply(states) if hstates is None else hstates
        return -(states*self.eta).conj()@h.T-h.conj()@(states*self.eta).T
