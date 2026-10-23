#!/usr/bin/env python3
"""
j4leak_v3.py -- the window-leakage bound and the oscillation periodogram of J4 version 3 (busi lane, task 1;
busi/notes/J4_v2_B4_v1.md section 1.2).

The step function S(R) of the site-centred digital ball deviates from the smooth asymptotic form by a sum of
oscillations a_m sqrt R cos(2 pi sqrt m R + phi_m) at the lattice frequencies sqrt m, m a sum of three squares
(Poisson summation over Z^3 \\ 0; the sqrt R growth is the version-1 measurement).  The Hann window of width W = 2
responds to the frequency f with hat w(f) = sinc(2f)/(1 - 4f^2): 0 at the integers and half-integers, 0.0083 at
sqrt 2 and sqrt 3, 0.0037 at sqrt 5, below 0.002 beyond.  With the single-ball residual power
sum_m a_m^2 / 2 = sigma_1^2 (sigma_1 the pooled rms of the per-ball residuals divided by sqrt R), Cauchy-Schwarz
bounds the leakage of the window average at the centre R by
    |eps(R)| <= sqrt 2 sigma_1 sqrt R [ (sum_m hat w(sqrt m)^2)^{1/2} + (sum_m hat w'(sqrt m)^2)^{1/2} / (4 pi R) ],
the second term being the sqrt R growth across the window (the window integral of d cos(2 pi f d) is
hat w'(f)/(2 pi i); d sqrt R/dR = 1/(2R)).  The constants are 0.0126 and 0.35 for W = 2.  The bound needs no
resolution of the individual lines (the per-window harmonic fit of version 2 was ill-posed: lines closer than 1/W
cannot be separated on one window), only the measured power and the exact responses.  It presumes frequencies >= 1;
the sublattice frequencies 1/2, sqrt 2/2, sqrt 3/2, sqrt 5/2, sqrt 11/2 (responses 0.5, 0.22, 0.069, 0.024, 0.008) are
tested by the periodogram of the sqrt R-normalized residuals of all balls of all radii (baseline about 50 lattice
units, resolution 0.02) and, if present above 10 percent of the f = 1 line, added to the bound by the fit.
"""
import math

import numpy as np

import j4window_v2 as jw

NAMED = [('1', 1.0), ('sqrt2', math.sqrt(2.0)), ('sqrt3', math.sqrt(3.0)), ('2', 2.0), ('sqrt5', math.sqrt(5.0)),
         ('sqrt6', math.sqrt(6.0)), ('sqrt8', math.sqrt(8.0)), ('3', 3.0), ('sqrt10', math.sqrt(10.0))]
HALF = [('1/2', 0.5), ('sqrt2/2', 0.5 * math.sqrt(2.0)), ('sqrt3/2', 0.5 * math.sqrt(3.0)),
        ('sqrt5/2', 0.5 * math.sqrt(5.0)), ('sqrt11/2', 0.5 * math.sqrt(11.0))]


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
    """Weighted single-frequency least-squares amplitudes sqrt(a^2 + b^2) of e ~ a cos(2 pi f R) + b sin(2 pi f R)."""
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


if __name__ == '__main__':
    s = response_sums()
    print('Hann W=2: (sum_m hat w^2)^1/2 = %.6f (largest |hat w| %.5f at sqrt %d); hat w\'(1) = %.6f; (sum_m hat w\'^2)^1/2 = %.5f; tail %.1e'
          % (math.sqrt(s['sum_w2']), s['max_w'], s['argmax_m'], s['dw'][0], math.sqrt(s['sum_dw2']), s['tail_w2']))
    for Rc in (12, 34, 62):
        print('  bound at sigma_1 = 3.3, R %d: %.3f nats' % (Rc, cs_bound(3.3, Rc, s)))
