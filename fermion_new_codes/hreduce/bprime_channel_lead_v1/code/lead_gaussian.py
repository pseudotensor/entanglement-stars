"""Own accepted ISO128 Gaussian helpers, reused verbatim for lead source review. No I/O."""
from __future__ import annotations
import numpy as np

def _real_scalar(value, name):
    array = np.asarray(value)
    if array.shape != () or np.iscomplexobj(array) or not np.isfinite(array):
        raise ValueError('Finite real scalar required: ' + name)
    return float(array)

def gaussian_kernels(radius, shell_radius, ell=4.0):
    """Unrenormalized angular Gaussian scalar/vector kernels, with broadcasting."""
    if np.iscomplexobj(radius) or np.iscomplexobj(shell_radius):
        raise ValueError('Real Gaussian coordinates required')
    r, source = np.broadcast_arrays(np.asarray(radius, dtype=np.float64),
                                     np.asarray(shell_radius, dtype=np.float64))
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(source)) or np.any(source < 0):
        raise ValueError('Finite coordinates and nonnegative shell radii required')
    ell = _real_scalar(ell, 'ell')
    if ell <= 0:
        raise ValueError('Positive finite Gaussian width required')
    rr = np.abs(r)
    z = rr * source / ell ** 2
    c = (2 * np.pi * ell ** 2) ** (-1.5)
    small = z < 0.05
    k0 = np.empty(r.shape, dtype=np.float64)
    k1 = np.empty_like(k0)
    zs = z[small]
    z2 = zs * zs
    envelope = c * np.exp(-(rr[small] ** 2 + source[small] ** 2) / (2 * ell ** 2))
    k0[small] = envelope * (1 + z2 / 6 + z2 ** 2 / 120 + z2 ** 3 / 5040
                             + z2 ** 4 / 362880 + z2 ** 5 / 39916800)
    k1[small] = envelope * zs * (1 / 3 + z2 / 30 + z2 ** 2 / 840
                                  + z2 ** 3 / 45360 + z2 ** 4 / 3991680
                                  + z2 ** 5 / 518918400)
    zl = z[~small]
    em = np.exp(-(rr[~small] - source[~small]) ** 2 / (2 * ell ** 2))
    ep = np.exp(-(rr[~small] + source[~small]) ** 2 / (2 * ell ** 2))
    k0[~small] = c * (em - ep) / (2 * zl)
    k1[~small] = c * ((zl - 1) * em + (zl + 1) * ep) / (2 * zl ** 2)
    k1 *= np.sign(r)
    if not np.all(np.isfinite(k0)) or not np.all(np.isfinite(k1)):
        raise ValueError('Nonfinite Gaussian evaluation')
    return k0, k1

def gaussian_centre_coefficients(energy, current, radius, ell=4.0):
    if any(np.iscomplexobj(x) for x in [energy, current, radius]):
        raise ValueError('Real centre source arrays required')
    ell = _real_scalar(ell, 'ell')
    e = np.asarray(energy, dtype=np.float64)
    j = np.asarray(current, dtype=np.float64)
    r = np.asarray(radius, dtype=np.float64)
    if e.ndim != 1 or e.shape != j.shape or e.shape != r.shape:
        raise ValueError('Matching one-dimensional shell arrays required')
    if not np.all(np.isfinite(e)) or not np.all(np.isfinite(j)):
        raise ValueError('Finite signed sources required')
    k0, _ = gaussian_kernels(0.0, r, ell)
    result = {'rho0': float(np.sum(e * k0)),
            'rho2': float(np.sum(e * k0 * (r ** 2 / (6 * ell ** 4) - 1 / (2 * ell ** 2)))),
            'J1': float(np.sum(j * k0 * r / (3 * ell ** 2))),
            'J3': float(np.sum(j * k0 * (r ** 3 / (30 * ell ** 6) - r / (6 * ell ** 4))))}
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite centre coefficients')
    return result
