#!/usr/bin/env python3
"""
j4leak_v3b.py -- the window-leakage bound, the periodogram and the joint line fit of J4 version 3b (busi lane, task 1;
busi/notes/J4_v2_B4_v1.md section 1.2; the correction in busi/notes/J4_v2_B4_judgment_v1.md sections 1.2 and 3).

The step function S(R) of the site-centred digital ball deviates from the smooth asymptotic form by a sum of
oscillations a_m sqrt R cos(2 pi sqrt m R + phi_m) at the lattice frequencies sqrt m, m a sum of three squares
(Poisson summation over Z^3 \\ 0; the sqrt R growth is the version-1 measurement, confirmed by the flat rms/sqrt R of
the stage-B windows).  The Hann window of full width W responds to the frequency f with
hat w(f) = sinc(W f)/(1 - W^2 f^2): for W = 2, 0 at the integers and half-integers, 0.0083 at sqrt 2 and sqrt 3,
0.0037 at sqrt 5, below 0.002 beyond.  With the single-ball residual power sum_m a_m^2 / 2 = sigma_1^2 (sigma_1 the
pooled rms of the per-ball residuals divided by sqrt R), Cauchy-Schwarz bounds the leakage of the window average at
the centre R by
    |eps(R)| <= sqrt 2 sigma_1 sqrt R [ (sum_m hat w(sqrt m)^2)^{1/2} + (sum_m hat w'(sqrt m)^2)^{1/2} / (4 pi R) ],
the second term being the sqrt R growth across the window.  The constants are 0.0126 and 0.35 for W = 2.  The bound
needs no resolution of the individual lines, only the measured power and the exact responses; it presumes
frequencies >= 1.

Version 3b, the sublattice test.  The single-frequency periodogram (version 3) reads at any frequency f every line's
amplitude times the spectral window of the sampling, |W(f - f_m)| = |sum_j d_j exp(2 pi i (f - f_m) R_j)| / sum_j d_j.
The balls sample R in 10 clusters of width 2 or 3 spaced 4 to 8 apart, and that window is 0.15 to 0.3 at the offsets
0.12 to 0.13 of sqrt5/2 and sqrt3/2 from f = 1 and about 0.15 at the offset 0.5 of 1/2 from 1: a lattice line of 2.8
reads as a 'sublattice line' of 0.4 to 0.8 (the version-3 gate-B WARNING, and the same readings on synthetic data
without any sublattice line).  The amplitudes are therefore measured by ONE joint least-squares fit of cos and sin at
the lattice frequencies sqrt m, m <= LATTICE_JOINT_MMAX, and the 5 sublattice frequencies over all balls of all radii
(joint_lines): the global baseline separates frequencies that one cluster cannot (their relative phase differs from
cluster to cluster); the largest pairwise correlation and the condition number of the design are returned so that
the conditioning is checked on the data; every amplitude carries a standard error from the residual scatter.  The
single-frequency periodogram is kept for the plot and as the diagnostic of the version-3 readings (spectral_window;
the fit prints the readings predicted from the fitted lattice lines alone).  A sublattice line counts as detected
only above 10 percent of the f = 1 amplitude AND above 3 standard errors (j4_fit_v3b).
"""
import math

import numpy as np

import j4window_v2 as jw

NAMED_M = [('1', 1), ('sqrt2', 2), ('sqrt3', 3), ('2', 4), ('sqrt5', 5), ('sqrt6', 6), ('sqrt8', 8), ('3', 9), ('sqrt10', 10)]
NAMED = [(nm, math.sqrt(m)) for nm, m in NAMED_M]
HALF = [('1/2', 0.5), ('sqrt2/2', 0.5 * math.sqrt(2.0)), ('sqrt3/2', 0.5 * math.sqrt(3.0)),
        ('sqrt5/2', 0.5 * math.sqrt(5.0)), ('sqrt11/2', 0.5 * math.sqrt(11.0))]
LATTICE_JOINT_MMAX = 20


def lattice_frequencies(mmax=400):
    ms = np.array([m for m in range(1, int(mmax) + 1) if jw.is_sum_of_three_squares(m)], dtype=int)
    return ms, np.sqrt(ms.astype(float))


def response_sums(W=2.0, window='hann', mmax=400, h=1e-6):
    """The response hat w and its derivative at every lattice frequency up to sqrt mmax, their sums of squares (with the
    analytic tail of the Hann response, |hat w| <= 1/(pi W f (W^2 f^2 - 1)), beyond mmax), and the largest response."""
    ms, f = lattice_frequencies(mmax)
    w = np.asarray(jw.window_ft(f, W, window), dtype=float)
    dw = (np.asarray(jw.window_ft(f + h, W, window), dtype=float) - np.asarray(jw.window_ft(f - h, W, window), dtype=float)) / (2.0 * h)
    tail_w2 = (5.0 / 3.0) / (math.pi ** 2 * float(W) ** 6) / (4.0 * float(mmax) ** 2) if window == 'hann' else float('nan')
    i = int(np.argmax(np.abs(w)))
    return dict(W=float(W), window=window, ms=ms, f=f, w=w, dw=dw, sum_w2=float(np.sum(w * w)) + (tail_w2 if window == 'hann' else 0.0),
                sum_dw2=float(np.sum(dw * dw)), tail_w2=tail_w2, max_w=float(abs(w[i])), argmax_m=int(ms[i]))


def cs_bound(sigma1, Rc, sums):
    """The Cauchy-Schwarz leakage bound (nats) at the window centre Rc for the pooled single-ball residual sigma1 sqrt R."""
    amp = math.sqrt(2.0) * float(sigma1) * math.sqrt(float(Rc))
    return amp * (math.sqrt(sums['sum_w2']) + math.sqrt(sums['sum_dw2']) / (4.0 * math.pi * float(Rc)))


def periodogram(R, e, weights, freqs):
    """Weighted single-frequency least-squares amplitudes sqrt(a^2 + b^2) of e ~ a cos(2 pi f R) + b sin(2 pi f R).
    Reads every line times the spectral window of the sampling (see spectral_window); a plot and a diagnostic, not the
    measurement of the lines (joint_lines is)."""
    R = np.asarray(R, dtype=float)
    e = np.asarray(e, dtype=float)
    wts = np.asarray(weights, dtype=float)
    out = np.empty(len(freqs))
    for i, f in enumerate(freqs):
        c = np.cos(2.0 * np.pi * f * R)
        s = np.sin(2.0 * np.pi * f * R)
        A = np.column_stack([c, s])
        M = A.T @ (A * wts[:, None])
        rhs = A.T @ (wts * e)
        try:
            coef = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(M, rhs, rcond=None)[0]
        out[i] = math.hypot(coef[0], coef[1])
    return out


def noise_floor(grid, amps, exclude_freqs, halfwidth=0.03):
    """The median periodogram amplitude off the given lines (within halfwidth of none of them)."""
    grid = np.asarray(grid, dtype=float)
    amps = np.asarray(amps, dtype=float)
    mask = np.ones(len(grid), dtype=bool)
    for f in exclude_freqs:
        mask &= np.abs(grid - f) > halfwidth
    return float(np.median(amps[mask])) if mask.any() else float('nan')


def spectral_window(R, weights, deltas):
    """|sum_j w_j exp(2 pi i delta R_j)| / sum_j w_j at each delta: the factor with which the single-frequency periodogram
    reads a line at f at the frequency f + delta (the spectral window of the sampling)."""
    R = np.asarray(R, dtype=float)
    w = np.asarray(weights, dtype=float)
    out = np.empty(len(deltas))
    for i, d in enumerate(deltas):
        out[i] = abs(np.sum(w * np.exp(2j * np.pi * float(d) * R))) / np.sum(w)
    return out


def joint_lines(R, e, weights, freqs):
    """Joint (weighted; None = uniform) least squares of e on cos and sin at all freqs.  Returns the amplitudes, phases
    (e ~ sum A cos(2 pi f R - phase)), standard errors of the amplitudes (delta method from the residual variance, which
    treats the residual as white), the coefficients, the fitted values and residuals, the residual rms, the condition
    number of the column-normalized design and its largest |correlation| between columns of different frequencies."""
    R = np.asarray(R, dtype=float)
    e = np.asarray(e, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    N, n = len(R), len(freqs)
    if N <= 2 * n + 1:
        raise ValueError('joint_lines: %d points for %d columns' % (N, 2 * n))
    w = np.ones(N) if weights is None else np.asarray(weights, dtype=float)
    A = np.empty((N, 2 * n))
    for i, f in enumerate(freqs):
        th = 2.0 * np.pi * f * R
        A[:, 2 * i] = np.cos(th)
        A[:, 2 * i + 1] = np.sin(th)
    sw = np.sqrt(w)
    Aw = A * sw[:, None]
    colnorm = np.linalg.norm(Aw, axis=0)
    An = Aw / colnorm
    cond = float(np.linalg.cond(An))
    coef_n = np.linalg.lstsq(An, e * sw, rcond=None)[0]
    coef = coef_n / colnorm
    fitted = A @ coef
    resid = e - fitted
    dof = N - 2 * n
    s2 = float(np.sum(w * resid ** 2) / dof)
    G = np.linalg.inv(An.T @ An)
    cov = s2 * G / np.outer(colnorm, colnorm)
    a, b = coef[0::2], coef[1::2]
    amps = np.hypot(a, b)
    phases = np.arctan2(b, a)
    se = np.empty(n)
    for i in range(n):
        va, vb, cab = cov[2 * i, 2 * i], cov[2 * i + 1, 2 * i + 1], cov[2 * i, 2 * i + 1]
        if amps[i] > 0.0:
            c, s = a[i] / amps[i], b[i] / amps[i]
            se[i] = math.sqrt(max(c * c * va + s * s * vb + 2.0 * c * s * cab, 0.0))
        else:
            se[i] = math.sqrt(max(0.5 * (va + vb), 0.0))
    D = np.sqrt(np.diag(G))
    C = G / np.outer(D, D)
    np.fill_diagonal(C, 0.0)
    for i in range(n):
        C[2 * i, 2 * i + 1] = 0.0
        C[2 * i + 1, 2 * i] = 0.0
    k = int(np.argmax(np.abs(C)))
    i1, i2 = divmod(k, 2 * n)
    return dict(freqs=freqs, amps=amps, phases=phases, se=se, coef=coef, cov=cov, fitted=fitted, resid=resid, dof=int(dof),
                resid_rms=float(np.sqrt(np.mean(resid ** 2))), s2=s2, cond=cond, max_corr=float(abs(C[i1, i2])),
                corr_pair=(float(freqs[i1 // 2]), float(freqs[i2 // 2])), n=int(N),
                weights=('uniform' if weights is None else 'given'))


def line_model(R, freqs, coef, idx=None):
    """The fitted line model at R restricted to the frequencies with indices idx (all if None)."""
    R = np.asarray(R, dtype=float)
    idx = range(len(freqs)) if idx is None else idx
    out = np.zeros(len(R))
    for i in idx:
        th = 2.0 * np.pi * float(freqs[i]) * R
        out += coef[2 * i] * np.cos(th) + coef[2 * i + 1] * np.sin(th)
    return out


def shift_range_factor(f, span=1.0):
    """The minimal half-range of cos(2 pi f (R + s)) over a shift span, divided by the amplitude: (1 - cos(pi f span))/2
    for f span <= 1, 1 beyond (an arc of length theta <= 2 pi has range at least 1 - cos(theta/2))."""
    th = 2.0 * math.pi * float(f) * float(span)
    if th >= 2.0 * math.pi:
        return 1.0
    return 0.5 * (1.0 - math.cos(0.5 * th))


if __name__ == '__main__':
    s = response_sums()
    print('Hann W=2: (sum_m hat w^2)^1/2 = %.6f (largest |hat w| %.5f at sqrt %d); hat w\'(1) = %.6f; (sum_m hat w\'^2)^1/2 = %.5f; tail %.1e'
          % (math.sqrt(s['sum_w2']), s['max_w'], s['argmax_m'], s['dw'][0], math.sqrt(s['sum_dw2']), s['tail_w2']))
    for Rc in (12, 34, 62):
        print('  bound at sigma_1 = 3.3, R %d: %.3f nats' % (Rc, cs_bound(3.3, Rc, s)))
    for W in (2.0, 4.0):
        print('Hann W=%.0f responses at the sublattice frequencies: %s' % (W, ', '.join('%s %.4f' % (nm, abs(float(jw.window_ft(f, W, 'hann')))) for nm, f in HALF)))
