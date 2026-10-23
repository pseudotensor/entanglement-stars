"""Independent angular trace and the variable-coefficient folded paper chain.

This module imports NumPy only. Dense frame setup never creates a Numba pool.
The folded matrix-free kernel imports Numba lazily on its first application.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from lead_native import valley_matrix


_FOLDED_KERNEL = None


def chain_labels(K):
    if type(K) is not int or K < 1:
        raise ValueError('Positive integer angular cutoff required')
    return np.asarray([(*nu, k, m2) for nu in itertools.product((0, 1), repeat=3)
                       for k in range(1, K+1) for m2 in range(-2*k+1, 2*k, 2)], dtype=np.int64)


def frame_labels(K):
    return np.asarray([(*label, sign, spin) for label in chain_labels(K)
                       for sign in (1, -1) for spin in (0, 1)], dtype=np.int64)


def scalar_harmonic(ell, m, points):
    """Condon--Shortley Y_lm from its associated-Legendre recurrence."""
    if type(ell) is not int or type(m) is not int or ell < 0 or abs(m) > ell:
        raise ValueError('Invalid scalar harmonic; zero prefactors must skip it')
    points = np.asarray(points, dtype=np.float64)
    radius = np.linalg.norm(points, axis=1)
    if points.ndim != 2 or points.shape[1] != 3 or np.any(radius <= 0) or not np.all(np.isfinite(points)):
        raise ValueError('Angular samples require finite nonzero Cartesian points')
    if m < 0:
        return (-1 if (-m) % 2 else 1)*scalar_harmonic(ell, -m, points).conj()
    z = points[:, 2]/radius
    sine = np.sqrt(np.maximum(0., 1-z*z))
    pmm = np.ones(len(points))
    for order in range(1, m+1):
        pmm *= -(2*order-1)*sine
    if ell == m:
        value = pmm
    else:
        previous, value = pmm, (2*m+1)*z*pmm
        for degree in range(m+2, ell+1):
            previous, value = value, ((2*degree-1)*z*value-(degree+m-1)*previous)/(degree-m)
    normalization = math.sqrt((2*ell+1)*math.factorial(ell-m)/(4*math.pi*math.factorial(ell+m)))
    return normalization*value*np.exp(1j*m*np.arctan2(points[:, 1], points[:, 0]))


def spinor_harmonic(kappa, m2, points):
    if type(kappa) is not int or not kappa or type(m2) is not int or m2 % 2 == 0 or abs(m2) >= 2*abs(kappa):
        raise ValueError('Invalid signed angular or magnetic label')
    k, j, m = abs(kappa), abs(kappa)-.5, m2/2
    if kappa > 0:
        coefficients = (math.sqrt((j+m)/(2*j)), math.sqrt((j-m)/(2*j)))
        ell = k-1
    else:
        coefficients = (-math.sqrt((j-m+1)/(2*j+2)), math.sqrt((j+m+1)/(2*j+2)))
        ell = k
    result = np.zeros((len(points), 2), dtype=np.complex128)
    for spin, (coefficient, magnetic) in enumerate(zip(coefficients, ((m2-1)//2, (m2+1)//2))):
        if coefficient != 0:
            result[:, spin] = coefficient*scalar_harmonic(ell, int(magnetic), points)
    return result


def gamma_spinor(kappa, m2, radial, points):
    radial = np.asarray(radial, dtype=np.complex128)
    if radial.shape != (2,):
        raise ValueError('Radial spinor needs two entries')
    result = np.empty((len(points), 4), dtype=np.complex128)
    result[:, :2] = radial[0]*spinor_harmonic(kappa, m2, points)
    result[:, 2:] = -radial[1]*spinor_harmonic(-kappa, m2, points)
    return result


def sampled_frame(points, K, columns=None, spacing=1.):
    """Rows are shell-local spin-major; columns have the frozen full label order."""
    raw = np.asarray(points)
    if raw.ndim != 2 or raw.shape[1] != 3 or not np.all(np.isfinite(raw)) or not np.array_equal(raw, np.rint(raw)):
        raise ValueError('Frame samples must be integer native coordinates')
    points = raw.astype(np.int64)
    labels = frame_labels(K)
    columns = np.arange(len(labels)) if columns is None else np.asarray(columns, dtype=np.int64)
    if columns.ndim != 1 or np.any(columns < 0) or np.any(columns >= len(labels)) or len(np.unique(columns)) != len(columns):
        raise ValueError('Invalid prescribed frame-column subset')
    physical = points*float(spacing)
    radius = np.linalg.norm(physical, axis=1)
    if np.any(radius <= 0) or not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('Angular frame requires positive radius and spacing')
    factors = spacing/(math.sqrt(2)*radius)
    result = np.empty((4*len(points), len(columns)), dtype=np.complex128)
    cache, matrices = {}, {}
    for out, index in enumerate(columns):
        label = labels[index]
        nu, k, m2, sign, spin = tuple(label[:3]), int(label[3]), int(label[4]), int(label[5]), int(label[6])
        key = (sign*k, m2, spin)
        if key not in cache:
            cache[key] = gamma_spinor(sign*k, m2, (1., 0.) if spin == 0 else (0., 1.), physical)
        if nu not in matrices:
            matrices[nu] = valley_matrix(nu)
        parity = np.where((points@np.asarray(nu)) % 2, -1., 1.)
        value = (cache[key]@matrices[nu].T)*(factors*parity)[:, None]
        result[:, out] = value.T.reshape(-1)
    return result


def valley_permutation(K, mu):
    labels = frame_labels(K)
    lookup = {tuple(row): index for index, row in enumerate(labels)}
    mu = tuple(mu)
    valley_matrix(mu)  # Validate the label without numerical consequences.
    target, phases = [], []
    for row in labels:
        nu = tuple(int(value) for value in row[:3])
        mapped = tuple(a ^ b for a, b in zip(mu, nu))+tuple(row[3:])
        target.append(lookup[mapped])
        inversions = sum(mu[i]*nu[j] for i in range(3) for j in range(i))
        phases.append(-1. if inversions % 2 else 1.)
    return np.asarray(target, np.int64), np.asarray(phases)


def antiunitary_permutation(K):
    """Author-fixed U_C A* = A C in the complete frozen frame labels."""
    labels = frame_labels(K)
    lookup = {tuple(row): index for index, row in enumerate(labels)}
    target, phases = [], []
    for row in labels:
        mapped = (*row[:4], -int(row[4]), -int(row[5]), 1-int(row[6]))
        target.append(lookup[mapped])
        exponent = (int(row[4])-1)//2
        phases.append(int(row[5])*(-1. if exponent % 2 else 1.))
    return np.asarray(target, np.int64), np.asarray(phases)


def fold_matrix():
    eye, sigma3 = np.eye(2), np.diag([1., -1.])
    return np.block([[eye, eye], [-sigma3, sigma3]]).astype(np.complex128)/math.sqrt(2)


def fold_states(raw):
    raw = np.asarray(raw, dtype=np.complex128)
    if raw.ndim < 2 or raw.shape[-1] != 2 or raw.shape[-2] % 2:
        raise ValueError('Fold needs an even number of two-spin radial sites')
    return raw.reshape(raw.shape[:-2]+(raw.shape[-2]//2, 4))@fold_matrix().T


def unfold_states(folded):
    folded = np.asarray(folded, dtype=np.complex128)
    if folded.ndim < 2 or folded.shape[-1] != 4:
        raise ValueError('Unfold needs four components per cell')
    raw = folded@fold_matrix().conj()
    return raw.reshape(folded.shape[:-2]+(2*folded.shape[-2], 2))


def raw_chain_blocks(k, N, v, mass=0., spacing=1.):
    N, v = np.asarray(N, float), np.asarray(v, float)
    if type(k) is not int or not k or N.ndim != 1 or v.shape != N.shape or not len(N):
        raise ValueError('Raw chain needs signed integer k and radial arrays indexed r=1,...')
    if not np.all(np.isfinite(N)) or not np.all(np.isfinite(v)) or np.any(N <= 0) or not np.isfinite(spacing) or spacing <= 0 or not np.isfinite(mass):
        raise ValueError('Invalid radial fields or lattice constants')
    sigma1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma3 = np.diag([1., -1.])
    r = spacing*np.arange(1, len(N)+1)
    onsite = N[:, None, None]*(mass*sigma3[None]+k/r[:, None, None]*sigma2[None])
    one = (-1j*(N[:-1]+N[1:])/(4*spacing))[:, None, None]*sigma1[None]
    two = (-1j*(v[:-2]+v[2:])/(8*spacing))[:, None, None]*np.eye(2)[None]
    return onsite, one, two


def folded_blocks(K, N, v, first_odd=33, end=None, mass=0., spacing=1.):
    N, v = np.asarray(N), np.asarray(v)
    end = len(N) if end is None else end
    if type(first_odd) is not int or first_odd < 1 or first_odd % 2 != 1 or type(end) is not int or end > len(N) or end < first_odd or end % 2:
        raise ValueError('Folded domain must start odd and end even inside radial fields')
    W = fold_matrix()
    cells = (end-first_odd+1)//2
    onsite, forward = [], []
    for k in range(1, K+1):
        diagonal, one, two = raw_chain_blocks(k, N, v, mass, spacing)
        intra = np.zeros((cells, 4, 4), dtype=np.complex128)
        inter = np.zeros((max(0, cells-1), 4, 4), dtype=np.complex128)
        for cell in range(cells):
            odd = first_odd-1+2*cell
            intra[cell, :2, :2], intra[cell, 2:, 2:] = diagonal[odd], diagonal[odd+1]
            intra[cell, :2, 2:] = one[odd]
            intra[cell, 2:, :2] = one[odd].conj().T
            if cell+1 < cells:
                inter[cell, :2, :2] = two[odd]
                inter[cell, 2:, :2] = one[odd+1]
                inter[cell, 2:, 2:] = two[odd+1]
        onsite.append(W[None]@intra@W.conj().T)
        forward.append(W[None]@inter@W.conj().T)
    return {'onsite': np.asarray(onsite), 'forward': np.asarray(forward),
            'first_odd': first_odd, 'end': end, 'K': K, 'spacing': float(spacing)}


def ideal_cell_block(K, blocks, c, d):
    labels = chain_labels(K)
    result = np.zeros((4*len(labels),)*2, dtype=np.complex128)
    if abs(c-d) > 1:
        return result
    for chain, label in enumerate(labels):
        k = int(label[3])-1
        value = blocks['onsite'][k, c] if c == d else blocks['forward'][k, c] if d == c+1 else blocks['forward'][k, d].conj().T
        sl = slice(4*chain, 4*chain+4)
        result[sl, sl] = value
    return result


def _folded_kernel():
    global _FOLDED_KERNEL
    if _FOLDED_KERNEL is None:
        from numba import njit, prange

        @njit(parallel=True, cache=True, fastmath=False)
        def apply_folded(states, output, onsite, forward, chain_k):
            count, chains, cells, _ = states.shape
            tiles = (cells+63)//64
            for tile in prange(count*chains*tiles):
                column, rest = tile//(chains*tiles), tile % (chains*tiles)
                chain, start = rest//tiles, (rest % tiles)*64
                k = chain_k[chain]
                for cell in range(start, min(start+64, cells)):
                    for spin in range(4):
                        value = 0j
                        for other in range(4):
                            value += onsite[k, cell, spin, other]*states[column, chain, cell, other]
                            if cell+1 < cells:
                                value += forward[k, cell, spin, other]*states[column, chain, cell+1, other]
                            if cell > 0:
                                value += forward[k, cell-1, other, spin].conjugate()*states[column, chain, cell-1, other]
                        output[column, chain, cell, spin] = -1j*value
        _FOLDED_KERNEL = apply_folded
    return _FOLDED_KERNEL


class FoldedLead:
    def __init__(self, K, N, v, first_odd=41, end=None, spacing=1.):
        self.K, self.labels = K, chain_labels(K)
        self.blocks = folded_blocks(K, N, v, first_odd, end, spacing=spacing)
        self.cells = self.blocks['onsite'].shape[1]
        self.chain_k = (self.labels[:, 3]-1).astype(np.int64)
        self.first_odd, self.end = first_odd, self.blocks['end']
        bounds = []
        for k in range(1, K+1):
            onsite, one, two = raw_chain_blocks(k, N, v, spacing=spacing)
            row = np.sum(np.abs(onsite[first_odd-1:self.end]), axis=2)
            for distance, values in ((1, one), (2, two)):
                value = np.sum(np.abs(values[first_odd-1:self.end-distance]), axis=2)
                row[:-distance] += value
                row[distance:] += np.sum(np.abs(values[first_odd-1:self.end-distance]), axis=1)
            bounds.append(float(np.max(row)))
        self.B = max(bounds)

    def apply(self, states, out=None):
        states = np.asarray(states)
        expected = (len(self.labels), self.cells, 4)
        if states.dtype != np.complex128 or states.ndim != 4 or states.shape[1:] != expected:
            raise ValueError('Wrong folded lead state layout')
        out = np.empty_like(states) if out is None else out
        if states.strides[-1] != states.itemsize or states.strides[-2] != 4*states.itemsize:
            raise ValueError('Folded inner cell coordinates must be contiguous')
        if out.shape != states.shape or out.dtype != np.complex128 or out.strides[-1] != out.itemsize or out.strides[-2] != 4*out.itemsize or np.shares_memory(states, out):
            raise ValueError('Folded output must be separate with contiguous inner cells')
        _folded_kernel()(states, out, self.blocks['onsite'], self.blocks['forward'], self.chain_k)
        return out


class RadialHamiltonian:
    """B11 chain, raw site-major spin layout; apply is -iH-eta."""
    def __init__(self, k, N, v, eta=None, mass=0., spacing=1.):
        self.k, self.N, self.v = k, np.asarray(N, float), np.asarray(v, float)
        self.onsite, self.one, self.two = raw_chain_blocks(k, N, v, mass, spacing)
        self.eta = np.zeros(len(self.N)) if eta is None else np.asarray(eta, float)
        if self.eta.shape != self.N.shape or np.any(self.eta < 0) or not np.all(np.isfinite(self.eta)):
            raise ValueError('CAP eta must be real, finite and nonnegative')
        self.is_unitary = not np.any(self.eta)
        row = np.sum(np.abs(-1j*self.onsite-self.eta[:, None, None]*np.eye(2)[None]), axis=2)
        for distance, values in ((1, self.one), (2, self.two)):
            row[:-distance] += np.sum(np.abs(values), axis=2)
            row[distance:] += np.sum(np.abs(values), axis=1)
        self.B = float(np.max(row))
        self.radii = float(spacing)*np.arange(1, len(self.N)+1)
        self.end_collar_indices = np.arange(2*max(0, len(self.N)-2), 2*len(self.N))

    def view(self, states):
        states = np.asarray(states)
        if states.dtype != np.complex128 or states.ndim != 2 or states.shape[1] != 2*len(self.N):
            raise ValueError('Raw radial states require flat complex128 columns, then site/spin')
        return states.reshape(len(states), len(self.N), 2)

    def h_apply(self, states):
        raw = self.view(states)
        result = np.einsum('nij,cnj->cni', self.onsite, raw)
        for distance, values in ((1, self.one), (2, self.two)):
            result[:, :-distance] += np.einsum('nij,cnj->cni', values, raw[:, distance:])
            result[:, distance:] += np.einsum('nji,cnj->cni', values.conj(), raw[:, :-distance])
        return result.reshape(len(states), -1)

    def apply(self, states, out=None):
        value = -1j*self.h_apply(states)-self.view(states).reshape(len(states), -1)*np.repeat(self.eta, 2)[None]
        if out is None:
            return value
        if out.shape != value.shape or out.dtype != np.complex128 or np.shares_memory(states, out):
            raise ValueError('Radial output must be a separate complex128 block')
        out[...] = value
        return out

    def dense_matrix(self, maximum_sites=256):
        if len(self.N) > maximum_sites:
            raise ValueError('Dense chain assembly is restricted to small retained/fixture operators')
        result = np.zeros((2*len(self.N),)*2, dtype=np.complex128)
        for site, value in enumerate(self.onsite):
            result[2*site:2*site+2, 2*site:2*site+2] = value
        for distance, values in ((1, self.one), (2, self.two)):
            for site, value in enumerate(values):
                left, right = slice(2*site, 2*site+2), slice(2*(site+distance), 2*(site+distance)+2)
                result[left, right], result[right, left] = value, value.conj().T
        return result

    def energy_balance(self, states, hstates=None):
        damped = states*np.repeat(self.eta, 2)[None]
        h = self.h_apply(states) if hstates is None else hstates
        return -damped.conj()@h.T-h.conj()@damped.T
