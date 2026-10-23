#!/usr/bin/env python
"""Strand 5 J5, the fast route, version 2 (builder analyst): strand5_w0fast_v1's resolvent integral and block-tridiagonal sweeps with the
packets batched: the channel matrices of the four packets differ only in the mass entries of the diagonal 4 x 4 blocks (the couplings to the
next cell carry only the derivative), so one sweep over the cells serves all packets and all t nodes at once (batch = packets x nt).
Returns per packet the same diagonal as v1's d_fast; checked against it channel by channel."""
import math

import numpy as np

import strand5_w0fast_v1 as V1


def channel_parts(r_int, r_half, psi_int, psi_half, j, h, cut=4.0):
    """The mass-free matrix (A' without its mass diagonal) and the per-node inverse rulers, on the channel's cut grid: (A0, pinv, k0)."""
    n = len(r_int)
    Ap, k0 = V1.channel_matrix(r_int, r_half, psi_int, psi_half, 0.0, j, h, cut)
    mm = n - k0
    pinv = np.concatenate([1.0 / psi_int[k0:], 1.0 / psi_half[k0:]])
    return Ap, pinv, k0


def d_fast_multi(A0, pinv, masses, nt=200, span=(1e-4, 1e4)):
    """diag(E_1 - E_0) per packet: A' = A0 + diag(m pinv) for each mass; one batched sweep."""
    mm = A0.shape[0] // 2
    P = len(masses)
    Bd0, Cu = V1.blocks(A0)
    # the mass entries per cell: u_int_c <-> v_int_c (rows/cols int node c), u_half_c <-> v_half_c (half node c)
    mi = pinv[:mm]
    mh = pinv[mm:]
    smin = min(masses) * float(np.min(pinv))
    smax = float(np.max(np.sum(np.abs(A0), axis=1))) + max(masses) * float(np.max(pinv))
    tlo, thi = span[0] * smin, span[1] * smax
    x, w = np.polynomial.legendre.leggauss(nt)
    s = 0.5 * (math.log(thi) - math.log(tlo)) * x + 0.5 * (math.log(thi) + math.log(tlo))
    ws = 0.5 * (math.log(thi) - math.log(tlo)) * w
    t = np.exp(s)
    I4 = np.eye(4)
    mass = np.asarray(masses, float)
    # B[c, p, t] = Bd0[c] + mass blocks - i t I
    Bm = np.zeros((mm, P, 4, 4))
    Bm[:, :, 1, 3] = Bm[:, :, 3, 1] = mi[:, None] * mass[None, :]
    Bm[:, :, 0, 2] = Bm[:, :, 2, 0] = mh[:, None] * mass[None, :]
    B = (Bd0[:, None, None, :, :] + Bm[:, :, None, :, :]) - 1j * t[None, None, :, None, None] * I4
    B = B.reshape(mm, P * nt, 4, 4)
    C = Cu.astype(complex)
    S = np.empty_like(B)
    S[0] = B[0]
    for c in range(1, mm):
        X = np.linalg.solve(S[c - 1], np.broadcast_to(C[c - 1], S[c - 1].shape))
        S[c] = B[c] - C[c - 1].T @ X
    Pm = B[mm - 1].copy()
    Gd = np.empty((mm, P * nt, 4), dtype=complex)
    Gd[mm - 1] = np.diagonal(np.linalg.inv(S[mm - 1]), axis1=-2, axis2=-1)
    for c in range(mm - 2, -1, -1):
        Y = np.linalg.solve(Pm, np.broadcast_to(C[c].T, Pm.shape))
        Pm = B[c] - C[c] @ Y
        Gd[c] = np.diagonal(np.linalg.inv(S[c] + Pm - B[c]), axis1=-2, axis2=-1)
    Gd = Gd.reshape(mm, P, nt, 4)
    wt = (t * t * ws)[None, None, :]
    d_int = (2 / math.pi) * np.sum((Gd[..., 3].imag - Gd[..., 1].imag) * wt, axis=2)     # (mm, P)
    d_half = (2 / math.pi) * np.sum((Gd[..., 2].imag - Gd[..., 0].imag) * wt, axis=2)
    out = []
    for p, m in enumerate(masses):
        Ap = A0.copy()
        idx = np.arange(mm)
        Ap[idx, idx] = m * mi
        Ap[mm + idx, mm + idx] = m * mh
        XtX = np.sum(Ap * Ap, axis=0)
        XXt = np.sum(Ap * Ap, axis=1)
        tail = -(2 / math.pi) * (XtX - XXt) / thi
        out.append(np.concatenate([d_int[:, p], d_half[:, p]]) + tail)
    return out
