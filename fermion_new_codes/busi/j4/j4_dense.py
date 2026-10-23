#!/usr/bin/env python3
"""
j4_dense.py -- dense 4-component control for the J4 ball cut.  Independent of j4lib.py and written
without reference to the reviewer's ball4.py: own Clifford matrices, own ball, own kernel (unwrapped
half-grid phase), own entropy.  Also offers a brute-force Bloch sum (--brute) for tiny boxes.

Model: h(k) = m beta + sum_i alpha_i sin k_i on the antiperiodic even box, k_i = 2 pi (j_i + 1/2)/L;
occupied projector P_-(k) = (1 - h/eps)/2; C(x,y) = (1/L^3) sum_k P_-(k) e^{ik(x-y)};
S = Tr s(C_A) over the site-centred ball |x| < R_eff (all 4 components).
Usage: j4_dense.py --R 6.5 --L 80 [--offset-sixteenths k] [--m 0] [--brute] [--out file.json] [--threads 1]
"""
import argparse
import json
import os
import resource
import sys
import time


def _early_threads():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--threads', type=int, default=1)
    a, _ = p.parse_known_args()
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[v] = str(a.threads)


_early_threads()
import numpy as np                    # noqa: E402
from scipy import special             # noqa: E402


def dirac4():
    """alpha_i = tau_x (x) sigma_i, beta = tau_z (x) 1."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    alpha = [np.kron(sx, s) for s in (sx, sy, sz)]
    beta = np.kron(sz, np.eye(2, dtype=complex))
    return alpha, beta


def ball(R_eff=None, n2max=None):
    """Sites with |x|^2 <= n2max = ceil(R_eff^2) - 1, i.e. |x| < R_eff; centre at the origin."""
    if n2max is None:
        n2max = int(np.ceil(float(R_eff) ** 2)) - 1
    M = int(np.floor(np.sqrt(n2max)))
    ax = np.arange(-M, M + 1)
    g = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), axis=-1).reshape(-1, 3)
    return g[np.sum(g * g, axis=1) <= n2max]


def fourier_tables(L, m):
    """raw_i[n] = (1/L^3) sum_j f_i(k_j) e^{2 pi i j.n/L} for f_i = sin k_i/eps, raw_m likewise for m/eps.
    The kernel is F(n) = raw[n mod L] * exp(i pi (n_1+n_2+n_3)/L) for every integer n (antiperiodic)."""
    k = 2.0 * np.pi * (np.arange(L) + 0.5) / L
    s = np.sin(k)
    S = [s.reshape([L if a == i else 1 for a in range(3)]) for i in range(3)]
    eps = np.sqrt(m * m + S[0] ** 2 + S[1] ** 2 + S[2] ** 2)
    raw = [np.fft.ifftn(np.broadcast_to(S[i] / eps, (L, L, L))) for i in range(3)]
    raw_m = np.fft.ifftn(np.broadcast_to(m / eps, (L, L, L))) if m != 0.0 else None
    return raw, raw_m


def kernel_values(raw, r, L):
    """F(r) for integer displacements r (..., 3): the unwrapped phase carries the antiperiodic sign."""
    r = np.asarray(r)
    idx = r % L
    return raw[idx[..., 0], idx[..., 1], idx[..., 2]] * np.exp(1j * np.pi * np.sum(r, axis=-1) / L)


def covariance_fft(sites, L, m):
    alpha, beta = dirac4()
    n = len(sites)
    raw, raw_m = fourier_tables(L, m)
    r = sites[:, None, :] - sites[None, :, :]
    C = np.zeros((n, 4, n, 4), dtype=complex)
    for i in range(3):
        Fi = kernel_values(raw[i], r, L)
        C -= 0.5 * Fi[:, None, :, None] * alpha[i][None, :, None, :]
    if raw_m is not None:
        Fm = kernel_values(raw_m, r, L)
        C -= 0.5 * Fm[:, None, :, None] * beta[None, :, None, :]
    C = C.reshape(4 * n, 4 * n)
    C[np.arange(4 * n), np.arange(4 * n)] += 0.5
    return C


def covariance_brute(sites, L, m):
    """Explicit Bloch sum; for tiny L only."""
    alpha, beta = dirac4()
    n = len(sites)
    ks = 2.0 * np.pi * (np.arange(L) + 0.5) / L
    r = (sites[:, None, :] - sites[None, :, :]).astype(float)
    C = np.zeros((n, 4, n, 4), dtype=complex)
    I4 = np.eye(4, dtype=complex)
    for k1 in ks:
        for k2 in ks:
            for k3 in ks:
                kv = np.array([k1, k2, k3])
                sv = np.sin(kv)
                eps = np.sqrt(m * m + np.dot(sv, sv))
                h = m * beta + sv[0] * alpha[0] + sv[1] * alpha[1] + sv[2] * alpha[2]
                P = 0.5 * (I4 - h / eps)
                ph = np.exp(1j * (r @ kv))
                C += ph[:, None, :, None] * P[None, :, None, :]
    return (C / L ** 3).reshape(4 * n, 4 * n)


def entropy_of_covariance(C):
    ev = np.linalg.eigvalsh(C)
    x = np.clip(ev, 0.0, 1.0)
    S = float(np.sum(-special.xlogy(x, x) - special.xlog1py(1.0 - x, -x)))
    return S, ev


def dense_entropy(R_eff=None, n2max=None, L=None, m=0.0, brute=False):
    t0 = time.time()
    sites = ball(R_eff, n2max)
    n = len(sites)
    if n2max is None:
        n2max = int(np.ceil(float(R_eff) ** 2)) - 1
    if R_eff is None:
        R_eff = float(np.sqrt(n2max + 0.5))
    if 2.0 * R_eff >= L:
        raise ValueError('the ball must fit in the box: need L > 2 R_eff')
    C = covariance_brute(sites, L, m) if brute else covariance_fft(sites, L, m)
    herm = float(np.abs(C - C.conj().T).max())
    C = 0.5 * (C + C.conj().T)
    S, ev = entropy_of_covariance(C)
    return dict(R_eff=float(R_eff), n2max=int(n2max), L=int(L), m=float(m), n=int(n), S=S,
                S_over_4piR2=S / (4.0 * np.pi * R_eff ** 2), hermiticity=herm,
                trace_minus_2n=float(ev.sum() - 2 * n), ev_min=float(ev.min()), ev_max=float(ev.max()),
                brute=bool(brute), seconds=time.time() - t0,
                peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)


def main():
    ap = argparse.ArgumentParser(description='dense 4-component ball-cut entropy (control)')
    ap.add_argument('--R', type=float, default=None, help='nominal radius; R_eff = R + offset/16')
    ap.add_argument('--offset-sixteenths', type=int, default=0)
    ap.add_argument('--Reff', type=float, default=None)
    ap.add_argument('--n2max', type=int, default=None, help='use the ball |x|^2 <= n2max directly')
    ap.add_argument('--L', type=int, required=True)
    ap.add_argument('--m', type=float, default=0.0)
    ap.add_argument('--brute', action='store_true', help='explicit Bloch sum (tiny L only)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--threads', type=int, default=1)
    args = ap.parse_args()
    R_eff = args.Reff if args.Reff is not None else (None if args.R is None else args.R + args.offset_sixteenths / 16.0)
    if R_eff is None and args.n2max is None:
        ap.error('give --R (with --offset-sixteenths), --Reff or --n2max')
    rec = dense_entropy(R_eff=R_eff, n2max=args.n2max, L=args.L, m=args.m, brute=args.brute)
    rec.update(R_nominal=args.R, offset_sixteenths=args.offset_sixteenths, threads=args.threads,
               numpy=np.__version__)
    print('dense 4-component: R_eff %.6f L %d m %g sites %d  S = %.12f  S/(4 pi R^2) = %.8f  (herm %.1e, Tr-2n %.1e, %.1f s)'
          % (rec['R_eff'], rec['L'], rec['m'], rec['n'], rec['S'], rec['S_over_4piR2'], rec['hermiticity'],
             rec['trace_minus_2n'], rec['seconds']))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as fh:
            json.dump(rec, fh, indent=1)


if __name__ == '__main__':
    main()
