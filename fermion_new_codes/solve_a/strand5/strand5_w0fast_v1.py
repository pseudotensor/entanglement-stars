#!/usr/bin/env python
"""Strand 5 J5, the fast route (builder analyst; for the production members, to be validated against strand5_w0radial_v4.py's dense
decompositions channel by channel): the same per-channel diagonal d = diag(E_1 - E_0) of the radial polar operator (version 4's A', the
integrating-factor derivative, the per-channel cut) without an eigendecomposition.

With H = [[0, A'], [A'^T, 0]] (u-space: A''s rows, v-space: its columns), (H^2 + t^2)^-1 = diag((A' A'^T + t^2)^-1, (A'^T A' + t^2)^-1) and
|X| = (2/pi) int_0^inf [1 - t^2 (X^2 + t^2)^-1] dt, so
    d_i = (E_1 - E_0)_ii = (2/pi) int_0^inf t^2 [((A'^T A' + t^2)^-1)_ii - ((A' A'^T + t^2)^-1)_ii] dt = (2/pi) int_0^inf t [Im G_vv,ii - Im G_uu,ii] dt,
G = (H - i t)^-1 (its diagonal is purely imaginary: i t diag((H^2 + t^2)^-1)).  Ordered cell by cell, [u_half_c, u_int_c, v_half_c, v_int_c],
H is block tridiagonal with 4 x 4 blocks (A' couples int node k to half nodes k and k + 1, half node k to int nodes k - 1 and k), so the
diagonal blocks of G follow from the two Schur-complement sweeps, G_cc = (S_c + P_c - B_c)^-1 with S_c = B_c - C_{c-1}^T S_{c-1}^-1 C_{c-1}
(forward) and P_c = B_c - C_c P_{c+1}^-1 C_c^T (backward), B_c = H_cc - i t, C_c = H_{c,c+1} (complex symmetric: transposes, not adjoints),
batched over the t nodes and the packets.  The t integral: Gauss-Legendre in s = log t on [log t_lo, log t_hi] (nt nodes; the integrand
t^3 [...] in s), t_lo = 1e-3 min(m) min(psi)^... and t_hi = 1e3 times the largest row norm, plus the tails: below t_lo the integrand ~ t^2
diag(H^-2) (dropped, of relative order (t_lo/sigma_min)^3), above t_hi ~ [diag_v(A'^T A') - diag_u(A' A'^T)] ... /t^2 handled analytically:
(2/pi) [diag(A'^T A')_ii - diag(A' A'^T)_ii]/t_hi with the sign of the expansion t^2 (X + t^2)^-1 = 1 - X/t^2 + ..."""
import math

import numpy as np


def channel_matrix(r_int, r_half, psi_int, psi_half, m, j, h, cut=4.0, n=None):
    """Version 4's A' for one packet and one j on the cut grid: returns (A', k0)."""
    n = len(r_int)
    K = j + 0.5
    kap = -K
    k0 = max(1, int(math.floor(K / cut))) - 1
    mm = n - k0
    ri, rh = r_int[k0:], r_half[k0:]
    pi_, ph_ = psi_int[k0:], psi_half[k0:]
    D = np.zeros((mm, mm))
    idx = np.arange(mm)
    D[idx, idx] = np.exp(kap * np.log(ri / rh)) / h
    D[idx[1:], idx[:-1]] = -np.exp(kap * np.log(ri[:-1] / rh[1:])) / h
    Ap = np.zeros((2 * mm, 2 * mm))
    Ap[idx, idx] = m / pi_
    Ap[mm + idx, mm + idx] = m / ph_
    Ap[:mm, mm:] = (pi_ ** -3)[:, None] * D.T
    Ap[mm:, :mm] = -(ph_ ** -3)[:, None] * D
    return Ap, k0


def d_dense(Ap):
    mm2 = Ap.shape[0]
    H = np.zeros((2 * mm2, 2 * mm2))
    H[:mm2, mm2:] = Ap
    H[mm2:, :mm2] = Ap.T
    lam, Q = np.linalg.eigh(H)
    pos = lam > 0
    top, bot = Q[:mm2][:, pos], Q[mm2:][:, pos]
    return 2.0 * ((top * top) @ lam[pos] - (bot * bot) @ lam[pos])


def blocks(Ap):
    """The block-tridiagonal form of H: per cell c the 4 x 4 diagonal block and the coupling to cell c + 1.
    Cell order [u_half_c, u_int_c, v_half_c, v_int_c]; A' rows/cols: int nodes 0..mm-1 then half nodes mm..2mm-1."""
    mm = Ap.shape[0] // 2
    Bd = np.zeros((mm, 4, 4))
    Cu = np.zeros((mm - 1, 4, 4))
    # index maps: u_half_c -> A' row mm + c ; u_int_c -> row c ; v_half_c -> col mm + c ; v_int_c -> col c
    for c in range(mm):
        rows = {0: mm + c, 1: c}
        cols = {2: mm + c, 3: c}
        for a, ra in rows.items():
            for b, cb in cols.items():
                Bd[c, a, b] = Ap[ra, cb]
                Bd[c, b, a] = Ap[ra, cb]
        if c + 1 < mm:
            rows2 = {0: mm + c + 1, 1: c + 1}
            cols2 = {2: mm + c + 1, 3: c + 1}
            for a, ra in rows.items():
                for b, cb in cols2.items():
                    Cu[c, a, b] = Ap[ra, cb]
            for a, ra in rows2.items():
                for b, cb in cols.items():
                    Cu[c, b, a] = Ap[ra, cb]
    return Bd, Cu


def d_fast(Ap, nt=240, span=(1e-4, 1e4)):
    """diag(E_1 - E_0) on the 2mm nodes (int then half) by the resolvent integral and the block sweeps."""
    mm = Ap.shape[0] // 2
    Bd, Cu = blocks(Ap)
    smin = float(np.min(np.abs(np.diag(Ap))))
    smax = float(np.max(np.sum(np.abs(Ap), axis=1)))
    tlo, thi = span[0] * smin, span[1] * smax
    x, w = np.polynomial.legendre.leggauss(nt)
    s = 0.5 * (math.log(thi) - math.log(tlo)) * x + 0.5 * (math.log(thi) + math.log(tlo))
    ws = 0.5 * (math.log(thi) - math.log(tlo)) * w
    t = np.exp(s)
    I4 = np.eye(4)
    B = Bd[:, None, :, :] - 1j * t[None, :, None, None] * I4      # (mm, nt, 4, 4)
    C = Cu[:, None, :, :].astype(complex)
    S = np.empty_like(B)
    S[0] = B[0]
    for c in range(1, mm):
        X = np.linalg.solve(S[c - 1], np.broadcast_to(C[c - 1], S[c - 1].shape))
        S[c] = B[c] - np.swapaxes(C[c - 1], -1, -2) @ X
    P = B[mm - 1].copy()
    Gd = np.empty((mm, len(t), 4), dtype=complex)
    Gd[mm - 1] = np.diagonal(np.linalg.inv(S[mm - 1]), axis1=-2, axis2=-1)
    for c in range(mm - 2, -1, -1):
        Y = np.linalg.solve(P, np.broadcast_to(np.swapaxes(C[c], -1, -2), P.shape))
        P = B[c] - C[c] @ Y
        Gd[c] = np.diagonal(np.linalg.inv(S[c] + P - B[c]), axis1=-2, axis2=-1)
    # integrand per node: t [Im G_vv - Im G_uu]; cell c: u_half (0), u_int (1), v_half (2), v_int (3)
    integ_int = t[None, :] * (Gd[:, :, 3].imag - Gd[:, :, 1].imag)
    integ_half = t[None, :] * (Gd[:, :, 2].imag - Gd[:, :, 0].imag)
    # in s: dt = t ds
    d_int = (2 / math.pi) * np.sum(integ_int * (t * ws)[None, :], axis=1)
    d_half = (2 / math.pi) * np.sum(integ_half * (t * ws)[None, :], axis=1)
    # the tail above thi: t^2[(X + t^2)^-1]_ii ~ 1 - X_ii/t^2; the difference v - u -> -(XtX_ii - XXt_ii)/t^2, integral (2/pi) * that / thi
    XtX = np.sum(Ap * Ap, axis=0)      # diag(A'^T A') (v-space)
    XXt = np.sum(Ap * Ap, axis=1)      # diag(A' A'^T) (u-space)
    tail = -(2 / math.pi) * (XtX - XXt) / thi
    d = np.concatenate([d_int, d_half]) + tail
    return d
