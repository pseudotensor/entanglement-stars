#!/usr/bin/env python3
"""Core formulas: (1.1), (1.2), (1.10)--(1.14), (2.11)."""
import math
import numpy as np
from scipy.linalg import eigvalsh
from scipy.sparse import coo_matrix

sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]], complex)
sz = np.diag([1., -1.]).astype(complex)
GAMMA = np.array([np.kron(sx, s) for s in (sx, sy, sz)])
BETA = np.kron(sz, np.eye(2))
I4 = np.eye(4, dtype=complex)


def primitive(p):
    p = np.asarray(p, dtype=np.int64)
    g = math.gcd(*map(abs, map(int, p)))
    if not g:
        raise ValueError("Zero normal")
    return p // g


def canonical(p):
    return tuple(sorted(map(abs, primitive(p)), reverse=True))


def egcd(a, b):
    if b == 0:
        return abs(a), (1 if a >= 0 else -1), 0
    g, x, y = egcd(b, a % b)
    return g, y, x - (a // b) * y


def basis(p):
    """Integer unimodular basis, Eq. (1.10); no supercell approximation."""
    p = primitive(p)
    a, b, c = map(int, p)
    g, x, y = egcd(a, b)
    if g == 0:
        B = np.diag([1, 1, c]).astype(np.int64)
    else:
        U = np.array([[x, -b//g, 0],
                      [y,  a//g, 0],
                      [0,      0, 1]], dtype=np.int64)
        gg, s, t = egcd(g, c)
        assert gg == 1
        V = np.array([[s, 0, -c],
                      [0, 1,  0],
                      [t, 0,  g]], dtype=np.int64)
        B = (U @ V)[:, [1, 2, 0]]
    A = np.rint(np.linalg.inv(B)).astype(np.int64)
    assert np.array_equal(A @ B, np.eye(3, dtype=np.int64))
    assert np.array_equal(A[2], p)
    return p, B, A


def h2(x):
    """Eq. (1.2), natural logarithms."""
    x = np.clip(np.asarray(x).real, 0, 1)
    out = np.zeros_like(x)
    ok = (x > 0) & (x < 1)
    out[ok] = -x[ok]*np.log(x[ok]) - (1-x[ok])*np.log1p(-x[ok])
    return out


def spectrum(C, gpu=-1):
    C = (C + C.conj().T)/2
    if gpu < 0:
        return eigvalsh(C, check_finite=False, overwrite_a=True)
    import torch
    with torch.cuda.device(gpu):
        a = torch.as_tensor(C, device=f"cuda:{gpu}")
        x = torch.linalg.eigvalsh(a).cpu().numpy()
        del a
    return x


def entropy(C, gpu=-1):
    return float(h2(spectrum(C, gpu)).sum())


def interval_covariance(p, q, length, nfft, mass=0., twist=.371):
    """Infinite-chain Fourier approximation, Eqs. (1.1), (1.12), (1.14)."""
    p, _, A = basis(p)
    if nfft < 4*length:
        raise ValueError("Use nfft >= 4*length")
    th = 2*np.pi*(np.arange(nfft)+twist)/nfft
    k = np.asarray(q) @ A[:2] + th[:, None]*p
    s = np.sin(k)
    E = np.sqrt(mass*mass + np.sum(s*s, axis=1))
    if E.min() < 1e-14:
        raise ValueError("Quadrature hit a node; change twist/grid")
    H = np.einsum("ki,iab->kab", s, GAMMA) + mass*BETA
    P = .5*(I4[None] - H/E[:, None, None])
    K = np.fft.ifft(P, axis=0)
    d = np.arange(length)[:, None] - np.arange(length)[None, :]
    blocks = K[d % nfft] * np.exp(2j*np.pi*twist*d/nfft)[..., None, None]
    return blocks.transpose(0, 2, 1, 3).reshape(4*length, 4*length)


def chain_generator(p, q, length, mass=0., velocity=None):
    """Open chain with endpoint-averaged range-2 transport, Eq. (2.11)."""
    p, _, A = basis(p)
    n = p / np.linalg.norm(p)
    if velocity is None:
        velocity = np.zeros(length)
    velocity = np.asarray(velocity)
    rr, cc, vv = [], [], []

    def add(l, r, block):
        aa, bb = np.indices((4, 4))
        rr.extend((4*l+aa.ravel()).tolist())
        cc.extend((4*r+bb.ravel()).tolist())
        vv.extend(block.ravel().tolist())

    for l in range(length):
        add(l, l, mass*BETA)
    for i in range(3):
        phase = np.exp(1j*np.dot(q, A[:2, i]))
        for l in range(length):
            r = l + int(p[i])
            if 0 <= r < length:
                T = -.5j*phase*GAMMA[i]
                add(l, r, T)
                add(r, l, T.conj().T)
            r = l + 2*int(p[i])
            if 0 <= r < length:
                vbar = .5*(velocity[l]+velocity[r])
                T = -.25j*n[i]*vbar*phase**2*I4
                add(l, r, T)
                add(r, l, T.conj().T)
    H = coo_matrix((vv, (rr, cc)),
                   shape=(4*length, 4*length)).tocsc()
    H.sum_duplicates()
    return H
