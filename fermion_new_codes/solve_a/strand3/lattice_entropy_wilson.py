#!/usr/bin/env python3
"""Strand 3 part C (scouting, not the model): a copy of the paper's lattice_entropy.interval_covariance with the mass
replaced by M_kappa(k) = q + kappa^-1 sum_i (1 - cos k_i), evaluated on the same k array (spec sec. 5).
Source: full_numerics/orient/lattice_entropy.py (md5 2c64ab5e5d13c5a54860381b8cc495b4); basis, GAMMA, BETA, I4 imported
from it unmodified.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "orient"))
from lattice_entropy import basis, GAMMA, BETA, I4  # noqa: E402


def interval_covariance_wilson(p, q, length, nfft, mass=0., kappa=np.inf, twist=.371):
    """As lattice_entropy.interval_covariance, with mass -> M_kappa(k)."""
    p, _, A = basis(p)
    if nfft < 4*length:
        raise ValueError("Use nfft >= 4*length")
    th = 2*np.pi*(np.arange(nfft)+twist)/nfft
    k = np.asarray(q) @ A[:2] + th[:, None]*p
    s = np.sin(k)
    M = mass + np.sum(1 - np.cos(k), axis=1)/kappa
    E = np.sqrt(M*M + np.sum(s*s, axis=1))
    if E.min() < 1e-14:
        raise ValueError("Quadrature hit a node; change twist/grid")
    H = np.einsum("ki,iab->kab", s, GAMMA) + M[:, None, None]*BETA
    P = .5*(I4[None] - H/E[:, None, None])
    K = np.fft.ifft(P, axis=0)
    d = np.arange(length)[:, None] - np.arange(length)[None, :]
    blocks = K[d % nfft] * np.exp(2j*np.pi*twist*d/nfft)[..., None, None]
    return blocks.transpose(0, 2, 1, 3).reshape(4*length, 4*length)
