#!/usr/bin/env python3
"""
j4window_v2.py -- digital-ball bookkeeping and radius-window averages for J4 version 2
(busi lane, task 1; specification busi/specs/J4_spec_v2.md, sections 2 and 3).  numpy only.

Facts used (each checked in j4_tests_v2.py):
  * The site-centred digital balls are B_n = {x in Z^3 : |x|^2 <= n} with n a sum of three squares
    (Legendre: n is not of the form 4^a (8 b + 7)).  B_n is the strict ball |x| < R exactly for
    R in (sqrt n, sqrt n+], n+ the next sum of three squares; for any integer n2max the ball
    {|x|^2 <= n2max} equals B_m with m the largest sum of three squares <= n2max (canonical label).
  * The radius-window average of centre R_c and full width W with the Hann weight
    w(d) = (1 + cos(2 pi d/W))/W on |d| < W/2 (integral 1) is the exact functional
    S_bar = int w(R - R_c) S(R) dR = sum_n omega_n S(B_n), omega_n = int_{I_n cap window} w,
    and the regressors of the smooth model are the same integrals of 4 pi R^2, R, ln R, 1, 1/R, ...
    (exact, because the model is linear in its parameters).
  * The response of the average to an oscillation cos(2 pi f R + phi) is hat w(f) = sinc(fW)/(1 - (fW)^2)
    (Hann; sinc x = sin(pi x)/(pi x)): zero at f = 1, 3/2, 2, 5/2, ... for W = 2; 0.0083 at sqrt 2,
    0.0083 at sqrt 3, 0.0037 at sqrt 5, 0.069 at sqrt3/2.  The version-1 estimator (8 offsets
    R + k/16, k odd, equal weights) responds with |sin(pi f)|/(8 |sin(pi f/8)|): zero at integer f not
    divisible by 8, 0.23 at sqrt 2, 0.15 at sqrt 3, 0.11 at sqrt 5, and 1 at f = 8 (aliasing).
"""
import numpy as np

TWO_PI = 2.0 * np.pi
V1_OFFSETS = [-7, -5, -3, -1, 1, 3, 5, 7]
REGRESSORS = ['4piR2', 'R', 'lnR', 'one', 'invR', 'invR2', 'lnR_invR2']


# ----------------------------------------------------------------------------------------------
# sums of three squares, canonical ball labels, radius intervals
# ----------------------------------------------------------------------------------------------
def is_sum_of_three_squares(n):
    """Legendre: n >= 0 is a sum of three squares iff n is not of the form 4^a (8 b + 7)."""
    n = int(n)
    if n < 0:
        return False
    while n and n % 4 == 0:
        n //= 4
    return n % 8 != 7


def is_sum_of_three_squares_bruteforce(n):
    """Direct search a <= b <= c (for the test of the rule)."""
    n = int(n)
    if n < 0:
        return False
    m = int(np.floor(np.sqrt(n))) + 1
    for a in range(m):
        for b in range(a, m):
            c2 = n - a * a - b * b
            if c2 < b * b:
                break
            c = int(round(np.sqrt(c2)))
            if c * c == c2:
                return True
    return False


def next_ball(n):
    """The smallest sum of three squares > n (the next canonical label)."""
    m = int(n) + 1
    while not is_sum_of_three_squares(m):
        m += 1
    return m


def canonical_ball(n2max):
    """The canonical label of {|x|^2 <= n2max}: the largest sum of three squares <= n2max."""
    m = int(n2max)
    if m < 0:
        raise ValueError('n2max must be >= 0')
    while not is_sum_of_three_squares(m):
        m -= 1
    return m


def ball_interval(n):
    """(R_lo, R_hi) = (sqrt n, sqrt n+): B_n is the strict ball |x| < R exactly for R in (R_lo, R_hi]."""
    n = int(n)
    if not is_sum_of_three_squares(n):
        raise ValueError('%d is not a canonical ball label' % n)
    return float(np.sqrt(n)), float(np.sqrt(next_ball(n)))


def ball_at_radius(R):
    """Canonical label of the strict ball |x| < R."""
    return canonical_ball(int(np.ceil(float(R) ** 2)) - 1)


def v1_balls(Rc):
    """The 8 balls of the version-1 protocol at the primary radius Rc (canonical labels)."""
    return [ball_at_radius(float(Rc) + k / 16.0) for k in V1_OFFSETS]


def balls_in_window(Rc, W):
    """All distinct balls whose radius interval meets the open window (Rc - W/2, Rc + W/2),
    increasing; list of (n, R_lo, R_hi)."""
    lo, hi = float(Rc) - 0.5 * float(W), float(Rc) + 0.5 * float(W)
    if lo <= 0.0:
        raise ValueError('the window reaches R <= 0')
    n = ball_at_radius(lo)
    out = []
    while True:
        R_lo, R_hi = ball_interval(n)
        if R_lo >= hi:
            break
        if R_hi > lo:
            out.append((int(n), R_lo, R_hi))
        n = next_ball(n)
    return out


def ball_levels(Rc, W, min_level=3, max_level=7):
    """level[n] = the first dyadic sampling level at which B_n is hit: level l samples the window at
    R = Rc - W/2 + (j + 1/2) 2^-l, j = 0 .. W 2^l - 1.  Balls hit by no level <= max_level get max_level + 1.
    Level 3 contains the 8 version-1 balls (R = Rc +- 1/16, ..., +- 7/16)."""
    level = {int(bb[0]): max_level + 1 for bb in balls_in_window(Rc, W)}
    lo = float(Rc) - 0.5 * float(W)
    for lev in range(min_level, max_level + 1):
        npts = int(round(float(W) * 2 ** lev))
        for j in range(npts):
            Rg = lo + (j + 0.5) / 2.0 ** lev
            nn = ball_at_radius(Rg)
            if nn in level and level[nn] > lev:
                level[nn] = lev
    return level


# ----------------------------------------------------------------------------------------------
# windows
# ----------------------------------------------------------------------------------------------
def window_weight(d, W, window='hann'):
    """The normalized window weight at displacement d from the centre (integral 1)."""
    d = np.asarray(d, dtype=float)
    inside = np.abs(d) < 0.5 * W
    if window == 'hann':
        return np.where(inside, (1.0 + np.cos(TWO_PI * d / W)) / W, 0.0)
    if window == 'rect':
        return np.where(inside, 1.0 / W, 0.0)
    raise ValueError('unknown window ' + str(window))


def window_cdf(d, W, window='hann'):
    """int_{-W/2}^{d} w(t) dt, with d clipped to [-W/2, W/2]."""
    d = np.clip(np.asarray(d, dtype=float), -0.5 * W, 0.5 * W)
    if window == 'hann':
        return (d + 0.5 * W) / W + np.sin(TWO_PI * d / W) / TWO_PI
    if window == 'rect':
        return (d + 0.5 * W) / W
    raise ValueError('unknown window ' + str(window))


def window_ft(f, W, window='hann'):
    """hat w(f) = int w(d) e^{2 pi i f d} dd (real and even): Hann sinc(fW)/(1 - (fW)^2) (1/2 at fW = 1),
    rectangular sinc(fW); numpy's sinc is sin(pi x)/(pi x)."""
    x = np.asarray(f, dtype=float) * float(W)
    s = np.sinc(x)
    if window == 'rect':
        return s
    if window == 'hann':
        den = 1.0 - x * x
        safe = np.abs(den) > 1e-9
        return np.where(safe, s / np.where(safe, den, 1.0), 0.5)
    raise ValueError('unknown window ' + str(window))


def v1_protocol_response(f):
    """|(1/8) sum_{k odd, |k| <= 7} e^{2 pi i f k/16}| = |sin(pi f)| / (8 |sin(pi f/8)|) (1 where f is a multiple of 8)."""
    f = np.asarray(f, dtype=float)
    den = 8.0 * np.abs(np.sin(np.pi * f / 8.0))
    safe = den > 1e-12
    return np.where(safe, np.abs(np.sin(np.pi * f)) / np.where(safe, den, 1.0), 1.0)


# ----------------------------------------------------------------------------------------------
# regressors and their window integrals
# ----------------------------------------------------------------------------------------------
def regressor_values(R):
    """The smooth regressors at R: 4 pi R^2, R, ln R, 1, 1/R, 1/R^2, ln R / R^2; shape (..., 7)."""
    R = np.asarray(R, dtype=float)
    lnR = np.log(R)
    return np.stack([4.0 * np.pi * R * R, R, lnR, np.ones_like(R), 1.0 / R, 1.0 / (R * R), lnR / (R * R)], axis=-1)


_GL = {}


def gauss_legendre(n):
    n = int(n)
    if n not in _GL:
        _GL[n] = np.polynomial.legendre.leggauss(n)
    return _GL[n]


def weighted_interval_integrals(Rc, W, a, b, window='hann', ngl=16):
    """(omega, I) for one ball interval [a, b]: omega = int w(R - Rc) over the part inside the window
    (exact antiderivative), I_j = int w f_j (Gauss-Legendre with ngl nodes on that part)."""
    a2, b2 = max(float(a), float(Rc) - 0.5 * W), min(float(b), float(Rc) + 0.5 * W)
    if b2 <= a2:
        return 0.0, np.zeros(len(REGRESSORS))
    x, wq = gauss_legendre(ngl)
    R = 0.5 * (b2 - a2) * x + 0.5 * (a2 + b2)
    wt = 0.5 * (b2 - a2) * wq * window_weight(R - Rc, W, window)
    omega = float(window_cdf(b2 - Rc, W, window) - window_cdf(a2 - Rc, W, window))
    return omega, wt @ regressor_values(R)


def plain_interval_means(a, b, ngl=16):
    """Unweighted means of the regressors over [a, b] (for synthetic data in the tests)."""
    x, wq = gauss_legendre(ngl)
    R = 0.5 * (float(b) - float(a)) * x + 0.5 * (float(a) + float(b))
    return (0.5 * wq) @ regressor_values(R)


def window_table(Rc, W, window='hann', ngl=16):
    """All balls of the window with their exact weights and weighted regressor means.
    'exact' = sum of the per-ball integrals = the whole-window regressor averages."""
    balls = balls_in_window(Rc, W)
    N = len(balls)
    n = np.array([bb[0] for bb in balls], dtype=np.int64)
    R_lo = np.array([bb[1] for bb in balls], dtype=float)
    R_hi = np.array([bb[2] for bb in balls], dtype=float)
    omega = np.zeros(N)
    integrals = np.zeros((N, len(REGRESSORS)))
    means = np.full((N, len(REGRESSORS)), np.nan)
    for i in range(N):
        om, I = weighted_interval_integrals(Rc, W, R_lo[i], R_hi[i], window, ngl)
        omega[i] = om
        integrals[i] = I
        if om > 0.0:
            means[i] = I / om
    return dict(Rc=float(Rc), W=float(W), window=window, n=n, R_lo=R_lo, R_hi=R_hi, R_mid=0.5 * (R_lo + R_hi),
                omega=omega, means=means, exact=integrals.sum(axis=0), omega_sum=float(omega.sum()))


def window_regressors_direct(Rc, W, window='hann', ngl=400):
    """The whole-window regressor averages by one Gauss-Legendre rule (independent of the ball bookkeeping)."""
    x, wq = gauss_legendre(ngl)
    R = 0.5 * W * x + float(Rc)
    wt = 0.5 * W * wq * window_weight(R - Rc, W, window)
    return wt @ regressor_values(R)
