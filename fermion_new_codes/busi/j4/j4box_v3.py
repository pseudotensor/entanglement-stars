#!/usr/bin/env python3
"""
j4box_v3.py -- the finite-box term of the J4 ball cut, version 3 (busi lane, task 1; lead's task B4; the
derivation is in busi/notes/J4_v2_B4_v1.md section 3).

The antiperiodic massless box of side L changes the entropy of the ball of radius R by
    S_L(B_R) - S_inf(B_R) = A_4 (R/L)^4 + remainder,
    A_4 = (128/15) Z_AP(2),  Z_AP(2) = sum_{nu in Z^3, nu != 0} (-1)^{nu_1+nu_2+nu_3} / |nu|^4
                                     = int_0^inf t [vartheta_4(0, e^{-t})^3 - 1] dt = -3.86316380719658...,
    A_4 = -32.9656644880775...
(the first law of entanglement with the continuum ball modular Hamiltonian, weight 2 pi (R^2 - r^2)/(2R), applied
to the uniform energy-density shift Delta rho = 16 Z_AP(2)/(pi^2 L^4) of 8 massless 4-component Dirac cones in the
antiperiodic box: Delta S = (8 pi^2 R^4/15) Delta rho).  The number is re-derived here by 2 evaluations of Z_AP(2)
(the theta integral, and Ewald splits at 3 values of the split point) and checked against the value on record.
The remainder at finite spacing (the lattice part of the (R/L)^4 coefficient, measured as b_4 - A_4 of about +0.4,
and an effective (R/L)^6 term b_6 of about -14) is measured on the 3-box ladders (4R, 6R, 8R) and bounded by its
spread, not modelled.

Per ball:  S_inf = S(L_max) - A_4 u - [e4_ref u + b6_ref u^{3/2}],   u = (R_vol/L_max)^4,
with e4_ref, b6_ref the medians over the ladder balls of the 3 largest radii that have them, and the bound
box_unc = spread(e4) u + spread(b6) u^{3/2} (about 1e-4 nats at L = 8R, 1e-3 at 4R).
"""
import math
import warnings

import numpy as np
from scipy import integrate

ZAP2_RECORD = -3.8631638071965854864
A4_RECORD = -32.96566448807753
_CACHE = {}


def theta4_direct(t):
    """vartheta_4(0, e^{-t}) = 1 + 2 sum_{n >= 1} (-1)^n e^{-t n^2} (converges for every t > 0; used for t >= 1)."""
    n = np.arange(1, 60, dtype=float)
    return 1.0 + 2.0 * float(np.sum((-1.0) ** n * np.exp(-float(t) * n * n)))


def theta4_modular(t):
    """The same function by its modular transform, sqrt(pi/t) 2 sum_{m >= 0} exp(-pi^2 (m + 1/2)^2 / t) (used for t < 1)."""
    m = np.arange(0, 16, dtype=float) + 0.5
    return math.sqrt(math.pi / float(t)) * 2.0 * float(np.sum(np.exp(-math.pi ** 2 * m * m / float(t))))


def theta4(t):
    t = float(t)
    if t <= 0.0:
        raise ValueError('t must be positive')
    return theta4_direct(t) if t >= 1.0 else theta4_modular(t)


def _integrand(t):
    return t * (theta4(t) ** 3 - 1.0)


def _quad(f, a, b):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        val, err = integrate.quad(f, a, b, epsabs=1e-13, epsrel=1e-13, limit=400)
    return float(val), float(err)


def zeta_ap2_quad():
    """Z_AP(2) = int_0^inf t [vartheta_4(0, e^{-t})^3 - 1] dt."""
    a, ea = _quad(_integrand, 0.0, 1.0)
    b, eb = _quad(_integrand, 1.0, np.inf)
    return a + b, ea + eb


def zeta_ap2_ewald(T=1.0, nmax=9):
    """The same number by an Ewald split at t = T: the direct lattice sum of int_T^inf t e^{-t a} dt = e^{-aT}(aT + 1)/a^2
    (a = |nu|^2, nu != 0) plus the theta integral on (0, T)."""
    ax = np.arange(-int(nmax), int(nmax) + 1)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing='ij')
    a = (X * X + Y * Y + Z * Z).astype(float)
    sgn = 1.0 - 2.0 * ((X + Y + Z) % 2)
    mask = a > 0
    direct = float(np.sum(sgn[mask] * np.exp(-T * a[mask]) * (T * a[mask] + 1.0) / a[mask] ** 2))
    recip, err = _quad(_integrand, 0.0, float(T))
    return direct + recip, err


def a4_from_zeta(z):
    return 128.0 * float(z) / 15.0


def a4_derived(check=True, tol=1e-10):
    """(A_4, Z_AP(2), quadrature error).  With check, the Ewald evaluations at T = 0.6, 1.0, 1.7 must agree with the
    theta integral to tol, and the number must agree with the value on record to tol."""
    key = ('a4', bool(check))
    if key in _CACHE:
        return _CACHE[key]
    z, err = zeta_ap2_quad()
    if check:
        for T in (0.6, 1.0, 1.7):
            ze, _ = zeta_ap2_ewald(T)
            if abs(ze - z) > tol:
                raise ArithmeticError('Z_AP(2): theta integral %.15f, Ewald split T = %.1f gives %.15f' % (z, T, ze))
        if abs(z - ZAP2_RECORD) > tol:
            raise ArithmeticError('Z_AP(2) = %.15f differs from the record %.15f' % (z, ZAP2_RECORD))
    out = (a4_from_zeta(z), z, err)
    _CACHE[key] = out
    return out


# ----------------------------------------------------------------------------------------------
# per-ball treatment
# ----------------------------------------------------------------------------------------------
def radius_group(R, radii, wide_upto):
    best = None
    for Rc in radii:
        W = 3.0 if Rc <= wide_upto else 2.0
        if abs(R - Rc) <= 0.5 * W + 0.1:
            best = Rc
    return best


def three_box_solution(Ls, S, R_vol):
    """(S_inf, b4, b6) of S(L) = S_inf + b4 u + b6 u^{3/2}, u = (R_vol/L)^4 (exact for 3 boxes, least squares beyond)."""
    u = (float(R_vol) / np.asarray(Ls, dtype=float)) ** 4
    X = np.column_stack([np.ones_like(u), u, u ** 1.5])
    return np.linalg.lstsq(X, np.asarray(S, dtype=float), rcond=None)[0]


def box_treatment(byball, radii, wide_upto, A4, fallback=(0.4, -14.0, 0.5, 5.0)):
    """byball: canonical ball -> {L: dict(S, R_vol, n_sites)}.  Returns (ext, box): ext[n] has S_inf, box_unc and the
    ladder measurement where 3 boxes exist; box has the reference remainder and the per-radius ladder summary."""
    ext, ladder = {}, []
    for n in sorted(byball):
        d = byball[n]
        Ls = sorted(d)
        R_vol = float(d[Ls[0]]['R_vol'])
        S = np.array([d[L]['S'] for L in Ls], dtype=float)
        u = (R_vol / np.array(Ls, dtype=float)) ** 4
        info = dict(n=int(n), R_vol=R_vol, n_sites=int(d[Ls[0]]['n_sites']), Ls=[int(L) for L in Ls], S_by_L=S.tolist(),
                    u_by_L=u.tolist(), nbox=len(Ls), S_raw=float(S[-1]), L_raw=int(Ls[-1]), u_raw=float(u[-1]),
                    ratio_raw=float(Ls[-1] / R_vol), group=radius_group(R_vol, radii, wide_upto),
                    S_cont=float(S[-1] - A4 * u[-1]), S_3box=None, b4=None, b6=None, delta_by_L=None)
        if len(Ls) >= 3:
            coef = three_box_solution(Ls, S, R_vol)
            info.update(S_3box=float(coef[0]), b4=float(coef[1]), b6=float(coef[2]),
                        delta_by_L={int(L): float(coef[0] - (SL - A4 * uL)) for L, SL, uL in zip(Ls, S, u)})
            ladder.append(info)
        ext[n] = info
    groups = {}
    for info in ladder:
        groups.setdefault(info['group'], []).append(info)
    per_group = {}
    for g, lst in groups.items():
        e4 = np.array([x['b4'] - A4 for x in lst])
        b6 = np.array([x['b6'] for x in lst])
        dmax = np.array([x['delta_by_L'][max(x['delta_by_L'])] for x in lst])
        per_group[g] = dict(count=len(lst), e4_median=float(np.median(e4)), e4_spread=float(np.max(np.abs(e4 - np.median(e4)))),
                            b6_median=float(np.median(b6)), b6_spread=float(np.max(np.abs(b6 - np.median(b6)))),
                            delta_max_median=float(np.median(dmax)), delta_max_abs=float(np.max(np.abs(dmax))),
                            ratio_max=float(np.median([x['ratio_raw'] for x in lst])))
    eligible = sorted([g for g in per_group if g is not None and per_group[g]['count'] >= 4])[-3:]
    if eligible:
        sel = [x for g in eligible for x in groups[g]]
        e4s = np.array([x['b4'] - A4 for x in sel])
        b6s = np.array([x['b6'] for x in sel])
        e4_ref, b6_ref = float(np.median(e4s)), float(np.median(b6s))
        sp4, sp6 = float(np.max(np.abs(e4s - e4_ref))), float(np.max(np.abs(b6s - b6_ref)))
        source = 'medians over the %d ladder balls of radii %s' % (len(sel), eligible)
    else:
        e4_ref, b6_ref, sp4, sp6 = [float(v) for v in fallback]
        source = 'fallback values (no radius with 4 or more ladder balls)'
    for info in ext.values():
        uu = info['u_raw']
        lat = e4_ref * uu + b6_ref * uu ** 1.5
        info['lat_remainder'] = float(lat)
        info['S_inf'] = float(info['S_cont'] - lat)
        info['box_unc'] = float(sp4 * uu + sp6 * uu ** 1.5)
        info['box_correction'] = float(info['S_inf'] - info['S_raw'])
    corr = {}
    for k in (4, 6, 8):
        uk = 1.0 / float(k) ** 4
        corr['%dR' % k] = dict(continuum=-A4 * uk, lattice=-(e4_ref * uk + b6_ref * uk ** 1.5), bound=sp4 * uk + sp6 * uk ** 1.5)
    box = dict(A4=float(A4), e4_ref=e4_ref, b6_ref=b6_ref, spread4=sp4, spread6=sp6, source=source, per_group=per_group,
               n3=len(ladder), n2=sum(1 for x in ext.values() if x['nbox'] == 2), n1=sum(1 for x in ext.values() if x['nbox'] == 1),
               corrections=corr,
               ladder=[dict(n=x['n'], R_vol=x['R_vol'], group=x['group'], b4=x['b4'], b6=x['b6'], delta_by_L=x['delta_by_L']) for x in ladder])
    return ext, box


if __name__ == '__main__':
    A4, z, err = a4_derived(check=True)
    print('Z_AP(2) = %.16f (quadrature error %.1e); Ewald T = 0.6, 1.0, 1.7: %s' % (z, err, [zeta_ap2_ewald(T)[0] for T in (0.6, 1.0, 1.7)]))
    print('A_4 = 128/15 Z_AP(2) = %.14f (record %.14f)' % (A4, A4_RECORD))
    print('corrections S_inf - S(L) at L = 4R, 6R, 8R (continuum): %.6f %.6f %.6f nats' % (-A4 / 256.0, -A4 / 1296.0, -A4 / 4096.0))
