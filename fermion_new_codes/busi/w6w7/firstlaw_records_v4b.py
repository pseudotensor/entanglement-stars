#!/usr/bin/env python3
"""
firstlaw_records_v4b.py -- FIRSTLAW_records v2.3 (busi lane, W6), version 4b.  Every line of this module is new (v4 W6; v4b where marked);
firstlaw_records_v3b.py is not edited and is imported read-only only for the optional engine route of the kernel.

Version 4b (W6W7_selftest_fix_v1b.md): the quadrature of the fixed-radius integrals C_u(R) and M_quad(R) on the dense grid is raised from version 4's
Simpson rule (4th order; its truncation error h^4 |g'''|/24 at a cut inside the kernel's support is about 1e-6 at h = ell/16, the self-test's S1 finding,
of the order of the representation defect the identity reads) to Boole's rule on the 4-cell panels from the centre with the remaining cells and the
partial cell by the exact integral of the degree-6 local polynomial (6th order; the leading error (2/945) h^6 [g^(5)(R) - g^(5)(0)], about 1e-9 at
h = ell/16; exact on cubic pieces with knots at the panel boundaries, which the engine's spline representations have at multiples of ell = 16 dense cells).
The identity, its four columns, the rows' rate, the full-clock residual and the gate are version 4's, unchanged; a stray factor in dM_from_s_exact is
removed (its value was right).  Outputs firstlaw_records_v4b_<tag>.*.

On the stored per-record arrays of a history whose FIRSTLAW_records_v3b pass exists (the v3b json for the roots, the pairs and the inputs; the v3b dense
cache for the dense N, v, rho_ell, J_ell of every rooted record; the fields npz for the site sources; the handoff npz for the radial stress and the per-site
dust, sea and reference energies) it prints, at every dense root (the outer root R_a of the first record of every included pair; the inner zero beside):

  (A) the exact fixed-radius mass-rate identity in 4 columns.  M(r) = r u^2/(2G) = int_0^r 4 pi s^2 (rho_ell + u J_ell) ds, u = v/N, is the dense
      solution's mass function (the constraint row M' = 4 pi r^2 (rho + u J), M(0) = 0).  With rho_ell = sum_n eps_n K_n, J_ell = sum_n J_n K_n
      (K_n the engine's Gaussian kernel of a shell source at R_n, per unit volume), w_n(r) = int_0^r 4 pi s^2 K_n ds (closed form) and the site charge
      Q_n = eps_n + u(R_n) J_n, the kernel identity M(r) = sum_n Q_n w_n(r) - C_u(r) holds exactly, C_u(r) = int_0^r 4 pi s^2 [K(uJ) - u K(J)] ds
      the commutator of the kernel with the flow (K(uJ) = sum_n u(R_n) J_n K_n).  At a radius r fixed between the records a and b, with
      Q_n = Q^S_n - q^P_n (the state's dust + sea against the continued reference) and chi_n(r) = 1_(R_n <= r):
          Delta M(r) = sum_n chi_n dQ^S_n  +  sum_n (w_n - chi_n) dQ^S_n  -  Delta C_u(r)  -  sum_n w_n dq^P_n
                        [1 flux through the old cut] [2 cut-assignment correction] [3 averaging commutator] [4 continued reference]
      The four sum to sum_n w_n dQ_n - Delta C_u(r) to roundoff by construction; the dense Delta M(r) against that sum is the representation
      defect (the analytic kernel against the engine's spline of the sources), printed.  Beside: the rows' rate 4 pi N R^2 T_ll (T_ll = rho + p_r - 2J)
      at the root of record a, -F/N at record b at the same radius, their midpoint, and the defect Delta M/dt - rows' rate.
  (B) the full-clock residual s_t/N + (1 + s^2)(ln N)' - 4 pi G r (rho + p_r), s = -v/N: the theta-theta evolution row with the two constraint rows
      substituted, zero where the rows hold, the full clock identity of requirements edit 32 where s_t = 0; at a root s = 1 and (ln N)' = 4 pi G R J,
      so it equals s_t/N - 4 pi G R T_ll, the mass-rate defect divided by R N/G (Delta M = (R/G)(Delta s + Delta s^2/2) exactly at fixed R).
  (C) the gate value sigma/R per root, sigma = ell for the engine's Gaussian (sigma |K_t| = sigma/R exactly at a root), inside the working value
      0.1 iff R >= 10 sigma.
The derivations, each term as this code computes it, are in RUN_firstlaw_records_v4b.md.  CPU only; no re-solve; no plotting; no dates.
"""
import os
import sys


def _preparse(argv):  # v4 W6: the thread cap before numpy is imported (the box: at most 16)
    threads = 8
    for i, a in enumerate(argv):
        if a == '--threads' and i + 1 < len(argv):
            try:
                threads = int(argv[i + 1])
            except ValueError:
                pass
        elif a.startswith('--threads='):
            try:
                threads = int(a.split('=', 1)[1])
            except ValueError:
                pass
    threads = max(1, min(threads, 16))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(threads)
    return threads


THREADS = _preparse(sys.argv[1:])

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy.special import erf  # noqa: E402

FOUR_PI = 4.0 * np.pi
NAN = float('nan')
SQ2 = math.sqrt(2.0)
SQ2PI = math.sqrt(2.0 * math.pi)
SQ2OPI = math.sqrt(2.0 / math.pi)
VERSION = 'v4b'


# ----------------------------------------------------------------------------
# small helpers (v4 W6)
# ----------------------------------------------------------------------------
def _float(x):  # v4 W6
    try:
        if x is None:
            return NAN
        return float(x)
    except (TypeError, ValueError):
        return NAN


def _fmt(x, p=6):  # v4 W6
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(xf):
        return 'nan'
    return ('%.' + str(p) + 'g') % xf


def _nan(func, arr):  # v4 W6
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    return float(func(arr)) if arr.size else NAN


def _jsonable(o):  # v4 W6
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if not str(k).startswith('_')}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    if o is None or isinstance(o, (str, int)):
        return o
    return '<%s>' % type(o).__name__


def local_poly(R, y, x0, degree=4):  # v4 W6: the degree-4 local polynomial (v3b's rule): (value, derivative) at x0
    R = np.asarray(R, float)
    y = np.asarray(y, float)
    n = R.size
    npts = min(degree + 1, n)
    degree = npts - 1
    i = int(np.searchsorted(R, x0, side='left'))
    lo = i - npts // 2
    if npts % 2 == 1 and 0 < i < n and (x0 - R[i - 1]) < (R[i] - x0):
        lo -= 1
    lo = max(0, min(n - npts, lo))
    x = R[lo:lo + npts]
    yy = y[lo:lo + npts]
    xc = 0.5 * (x[0] + x[-1])
    sc = max(float(x[-1] - x[0]), 1e-300)
    coef = np.polyfit((x - xc) / sc, yy, degree)
    t0 = (x0 - xc) / sc
    return float(np.polyval(coef, t0)), float(np.polyval(np.polyder(coef), t0)) / sc


def local_poly_integral(R, y, a, b, degree=6):  # v4 W6 (v4b: degree 6 by default): the exact integral from a to b of the local polynomial through the nodes nearest (a + b)/2
    R = np.asarray(R, float)
    y = np.asarray(y, float)
    n = R.size
    npts = min(degree + 1, n)
    degree = npts - 1
    xm = 0.5 * (a + b)
    i = int(np.searchsorted(R, xm, side='left'))
    lo = max(0, min(n - npts, i - npts // 2))
    x = R[lo:lo + npts]
    yy = y[lo:lo + npts]
    xc = 0.5 * (x[0] + x[-1])
    sc = max(float(x[-1] - x[0]), 1e-300)
    coef = np.polyfit((x - xc) / sc, yy, degree)
    P = np.polyint(coef)
    return sc * float(np.polyval(P, (b - xc) / sc) - np.polyval(P, (a - xc) / sc))


def boole_uniform(y, h):  # v4b W6: composite Boole's rule on a uniform grid of 4k + 1 points, (2h/45)(7 y_0 + 32 y_1 + 12 y_2 + 32 y_3 + 7 y_4) per panel
    y = np.asarray(y, float)
    n = y.size
    if n < 5 or (n - 1) % 4 != 0:
        raise ValueError('boole_uniform needs 4k + 1 points (k >= 1), got %d' % n)
    s = 7.0 * (y[0] + y[-1]) + 32.0 * y[1::2].sum() + 12.0 * y[2:-1:4].sum() + 14.0 * y[4:-1:4].sum()
    return float(2.0 * h / 45.0 * s)


def integral_0_to(r, g, Rstar, qdegree=6):
    """int_0^Rstar g dr on the uniform grid r (v4b W6): Boole's rule on the 4-cell panels from the centre to the last panel boundary at or below Rstar, the
    remaining cells (at most 3) and the partial cell [last node <= Rstar, Rstar] by the exact integral of the degree-qdegree local polynomial through the
    nearest nodes.  6th order in the spacing for a smooth integrand (the leading error (2/945) h^6 [g^(5)(b) - g^(5)(0)] of the Boole part; the end cells
    O(h^8)); exact on cubic pieces with knots at the panel boundaries.  Rstar beyond the grid is cut at the last node."""
    r = np.asarray(r, float)
    g = np.asarray(g, float)
    h = float(r[1] - r[0])
    i0 = int(np.searchsorted(r, Rstar, side='right')) - 1
    i0 = max(0, min(r.size - 1, i0))
    nb = 4 * (i0 // 4)
    total = boole_uniform(g[:nb + 1], h) if nb >= 4 else 0.0
    for i in range(nb, i0):
        total += local_poly_integral(r, g, float(r[i]), float(r[i + 1]), qdegree)
    if r[i0] < Rstar <= r[-1]:
        total += local_poly_integral(r, g, float(r[i0]), float(Rstar), qdegree)
    return float(total)


# ----------------------------------------------------------------------------
# the kernel (v4 W6): the engine's Gaussian of standard deviation ell per unit volume, for a shell source at R_n
# ----------------------------------------------------------------------------
def shell_kernel(ell, Rn, s):
    """K_n(s) = (2 pi ell^2)^(-3/2) (ell^2/(2 s Rn)) [exp(-(s - Rn)^2/(2 ell^2)) - exp(-(s + Rn)^2/(2 ell^2))], the angular average of the 3D Gaussian over the shell;
    Rn -> 0 and s -> 0 by their limits; int_0^inf 4 pi s^2 K_n ds = 1; K_n(0) = (2 pi ell^2)^(-3/2) exp(-Rn^2/(2 ell^2)) (the engine's centre coefficient)."""  # v4 W6
    s = np.asarray(s, float)
    ell = float(ell)
    Rn = float(Rn)
    pref = (2.0 * np.pi * ell * ell) ** (-1.5)
    if Rn < 1e-12 * ell:
        return pref * np.exp(-s * s / (2.0 * ell * ell))
    out = np.empty_like(s)
    small = s < 1e-6 * ell
    sb = s[~small]
    out[~small] = pref * (ell * ell / (2.0 * sb * Rn)) * (np.exp(-(sb - Rn) ** 2 / (2.0 * ell * ell)) - np.exp(-(sb + Rn) ** 2 / (2.0 * ell * ell)))
    out[small] = pref * np.exp(-Rn * Rn / (2.0 * ell * ell))
    return out


def cumulative_weight(ell, Rn, R):
    """w_n(R) = int_0^R 4 pi s^2 K_n(s) ds = (1/2)[erf((R - Rn)/(sqrt2 ell)) + erf((R + Rn)/(sqrt2 ell))] - (ell/(sqrt(2 pi) Rn)) [exp(-(R - Rn)^2/(2 ell^2)) - exp(-(R + Rn)^2/(2 ell^2))];
    Rn -> 0: erf(R/(sqrt2 ell)) - sqrt(2/pi) (R/ell) exp(-R^2/(2 ell^2)).  Vectorized over the sites Rn for one radius R."""  # v4 W6
    Rn = np.atleast_1d(np.asarray(Rn, float))
    ell = float(ell)
    R = float(R)
    out = np.empty(Rn.size)
    small = Rn < 1e-7 * ell
    x = R / (SQ2 * ell)
    out[small] = erf(x) - SQ2OPI * (R / ell) * math.exp(-x * x)
    a = Rn[~small]
    out[~small] = 0.5 * (erf((R - a) / (SQ2 * ell)) + erf((R + a) / (SQ2 * ell))) - (ell / (SQ2PI * a)) * (np.exp(-(R - a) ** 2 / (2.0 * ell * ell)) - np.exp(-(R + a) ** 2 / (2.0 * ell * ell)))
    return out


def kernel_matrix(ell, R_sites, r_fine):  # v4 W6: K[n, i] = K_n(r_fine[i]), sites x dense points
    R_sites = np.asarray(R_sites, float)
    r_fine = np.asarray(r_fine, float)
    K = np.empty((R_sites.size, r_fine.size))
    for i, Rn in enumerate(R_sites):
        K[i] = shell_kernel(ell, Rn, r_fine)
    return K


def kernel_matrix_engine(v3b_dir, engine_path, engine_module, R_sites, ell, r_last_node, support_rel, r_fine):
    """The same matrices from the engine's own SmoothedSources5 with a unit source per site (the rho and the J kernels separately); slow (one kernel call per site)."""  # v4 W6
    if v3b_dir and v3b_dir not in sys.path:
        sys.path.insert(0, v3b_dir)
    import firstlaw_records_v3b as V3B  # noqa: E402
    FLD, prov = V3B.import_engine(engine_path, engine_module)
    ell_pass = V3B.ell_for_engine(float(ell))
    R_sites = np.asarray(R_sites, float)
    K = np.zeros((R_sites.size, np.asarray(r_fine).size))
    KJ = np.zeros_like(K)
    unit = np.zeros(R_sites.size)
    for i in range(R_sites.size):
        unit[:] = 0.0
        unit[i] = 1.0
        src = FLD.SmoothedSources5(unit, unit.copy(), R_sites, ell_pass, float(r_last_node), support_rel)
        K[i] = V3B._ev(src.rho, np.asarray(r_fine, float))
        KJ[i] = V3B._ev(src.J, np.asarray(r_fine, float))
    return K, KJ, prov


def commutator_integrand(r, u_dense, J_sites, u_sites, KJ):
    """4 pi r^2 [K(uJ)(r) - u(r) K(J)(r)] with K(uJ) = sum_n u(R_n) J_n K_n and K(J) = sum_n J_n K_n: the integrand of the averaging-commutator term C_u."""  # v4 W6
    r = np.asarray(r, float)
    KuJ = (np.asarray(u_sites, float) * np.asarray(J_sites, float)) @ KJ
    KJs = np.asarray(J_sites, float) @ KJ
    return FOUR_PI * r * r * (KuJ - np.asarray(u_dense, float) * KJs), KJs


# ----------------------------------------------------------------------------
# the four columns, the rows' rate, the full clock, the gate (v4 W6): pure functions on values and arrays
# ----------------------------------------------------------------------------
def four_columns(w, chi, dQ_state, dQ_ref, dCu):
    """The four columns at a fixed radius from the site arrays: w the kernel's cumulative weights, chi the sharp cut, dQ_state = dQ^S, dQ_ref = dq^P
    (so that dQ = dQ_state - dQ_ref), dCu = C_u,b - C_u,a.  Returns the columns, their sum and the representation sum w . dQ - dCu."""  # v4 W6
    w = np.asarray(w, float)
    chi = np.asarray(chi, float)
    dQ_state = np.asarray(dQ_state, float)
    dQ_ref = np.asarray(dQ_ref, float)
    col1 = float(chi @ dQ_state)
    col2 = float((w - chi) @ dQ_state)
    col3 = -float(dCu)
    col4 = -float(w @ dQ_ref)
    rep = float(w @ (dQ_state - dQ_ref)) - float(dCu)
    return {'col1_flux_old_cut': col1, 'col2_cut_assignment': col2, 'col3_averaging_commutator': col3, 'col4_continued_reference': col4,
            'sum4': col1 + col2 + col3 + col4, 'rep': rep, 'closure': col1 + col2 + col3 + col4 - rep}


def rows_rate_values(R, N, v, rho, J, p, G=None):
    """F = 4 pi R^2 [(N^2 + v^2) J + N v (rho + p_r)] (input 05) and the rows' mass rate -F/N at the radius R; T_ll = rho + p_r - 2J and 4 pi N R^2 T_ll,
    which equals -F/N where v = -N."""  # v4 W6
    F = FOUR_PI * R * R * ((N * N + v * v) * J + N * v * (rho + p))
    Tll = rho + p - 2.0 * J
    return {'F': F, 'rate_rows': -F / N if N != 0 else NAN, 'Tll': Tll, 'rate_root_form': FOUR_PI * N * R * R * Tll, 'Nplusv': N + v,
            'rate_root_nostress': FOUR_PI * N * R * R * (rho - 2.0 * J), 'rate_root_stress': FOUR_PI * N * R * R * p}


def full_clock_values(R, dt, G, s_a, s_b, N_a, N_b, lnNp_a, lnNp_b, rho_a, rho_b, p_a, p_b):
    """s_t/N + (1 + s^2)(ln N)' - 4 pi G R (rho + p_r) from the two records' values at the fixed radius R: the midpoint rule (the averages of the two
    records) and the endpoint rule at a; s_t = (s_b - s_a)/dt."""  # v4 W6
    st = (s_b - s_a) / dt
    sm, Nm, lm, rm, pm = 0.5 * (s_a + s_b), 0.5 * (N_a + N_b), 0.5 * (lnNp_a + lnNp_b), 0.5 * (rho_a + rho_b), 0.5 * (p_a + p_b)
    res_mid = st / Nm + (1.0 + sm * sm) * lm - FOUR_PI * G * R * (rm + pm)
    res_a = st / N_a + (1.0 + s_a * s_a) * lnNp_a - FOUR_PI * G * R * (rho_a + p_a)
    return {'s_t': st, 'residual_mid': res_mid, 'residual_a': res_a, 'term_st_over_N_mid': st / Nm, 'term_clock_mid': (1.0 + sm * sm) * lm,
            'term_matter_mid': -FOUR_PI * G * R * (rm + pm)}


def gate_value(sigma, R, working=0.1):  # v4 W6: sigma |K_t| at a root (|K_t| = |v|/(r N) = 1/R there), against the working value
    g = float(sigma) / float(R) if (np.isfinite(R) and R > 0) else NAN
    return {'gate': g, 'inside': bool(g <= working) if np.isfinite(g) else None}


# ----------------------------------------------------------------------------
# the records (v4 W6)
# ----------------------------------------------------------------------------
class RecordStore(object):
    """Per fields row: the dense arrays (the v3b cache), the site sources (the fields npz), the per-site dust/sea/reference split and the stress spline
    (the handoff npz), the kernel sums and the integrands the identity needs."""  # v4 W6

    def __init__(self, cache, F, HO, K, KJ, G, ell, degree, qdegree, pr_scale):
        self.cache, self.F, self.HO, self.K, self.KJ, self.G, self.ell = cache, F, HO, K, KJ, float(G), float(ell)
        self.degree, self.qdegree, self.pr_scale = int(degree), int(qdegree), float(pr_scale)  # v4b: qdegree the end cells' polynomial degree of the quadrature
        self.working = 0.1
        self.r = np.asarray(cache['r_fine'], float)
        self.h = float(self.r[1] - self.r[0])
        self.R_sites = np.asarray(F['R'], float)
        idx = np.rint(self.R_sites / self.h).astype(int)
        ok = (idx >= 0) & (idx < self.r.size)
        ok[ok] = np.abs(self.r[idx[ok]] - self.R_sites[ok]) < 1e-9
        self.site_index = idx if bool(np.all(ok)) else None
        self._recs = {}
        self.hmap = {}
        if HO is not None and HO.get('T') is not None:
            for j, t in enumerate(np.asarray(HO['T'], float).ravel()):
                self.hmap.setdefault(round(float(t), 9), int(j))

    def handoff_row(self, row):
        Tf = np.asarray(self.F['T'], float)
        if row >= Tf.size or not self.hmap:
            return None
        return self.hmap.get(round(float(Tf[row]), 9))

    def get(self, row, T_over_rs):
        if row in self._recs:
            return self._recs[row]
        c = self.cache
        i = c['index'][row]
        r = self.r
        N = np.asarray(c['arrays']['N'][i], float)
        v = np.asarray(c['arrays']['v'][i], float)
        rho = np.asarray(c['arrays']['rho'][i], float)
        J = np.asarray(c['arrays']['J'][i], float)
        with np.errstate(divide='ignore', invalid='ignore'):
            u = np.where(N != 0.0, v / np.where(N != 0.0, N, 1.0), 0.0)
        M = r * u * u / (2.0 * self.G)
        eps = np.asarray(self.F['eps_tot'][row], float)
        Js = np.asarray(self.F['J_tot'][row], float)
        u_sites = u[self.site_index] if self.site_index is not None else np.interp(self.R_sites, r, u)
        rec = {'row': int(row), 'T': float(T_over_rs), 'N': N, 'v': v, 'rho': rho, 'J': J, 'u': u, 's': -u, 'M': M, 'eps': eps, 'J_site': Js, 'u_sites': u_sites,
               'split_eps': 'none', 'split_J': 'none', 'ref_check_max_abs': NAN, 'handoff_row': None}
        j = self.handoff_row(row)
        rec['handoff_row'] = j
        HO = self.HO or {}
        eps_state, eps_ref = eps, np.zeros_like(eps)
        if j is not None:
            if HO.get('eps_dust') is not None and HO.get('eps_sea') is not None:
                eps_state = np.asarray(HO['eps_dust'][j], float) + np.asarray(HO['eps_sea'][j], float)
                eps_ref = eps_state - eps
                rec['split_eps'] = 'eps_dust + eps_sea (handoff) against eps_tot (fields); the reference = their difference'
                if HO.get('eps_R2') is not None:
                    rec['ref_check_max_abs'] = float(np.max(np.abs(eps_ref - np.asarray(HO['eps_R2'][j], float))))
            elif HO.get('eps_R2') is not None:
                eps_ref = np.asarray(HO['eps_R2'][j], float)
                eps_state = eps + eps_ref
                rec['split_eps'] = 'eps_R2 (handoff); the state = eps_tot + eps_R2'
        J_state, J_ref = Js, np.zeros_like(Js)
        if j is not None:
            if HO.get('J_R2') is not None:
                J_ref = np.asarray(HO['J_R2'][j], float)
                J_state = Js + J_ref
                rec['split_J'] = 'J_R2 (handoff)'
            elif HO.get('J_dust') is not None and HO.get('J_sea') is not None:
                J_state = np.asarray(HO['J_dust'][j], float) + np.asarray(HO['J_sea'][j], float)
                J_ref = J_state - Js
                rec['split_J'] = 'J_dust + J_sea (handoff) against J_tot'
        if not np.all(np.isfinite(eps_state)):
            eps_state, eps_ref = eps, np.zeros_like(eps)
            rec['split_eps'] = 'none (the handoff arrays are not finite)'
        if not np.all(np.isfinite(J_state)):
            J_state, J_ref = Js, np.zeros_like(Js)
            rec['split_J'] = 'none (the handoff arrays are not finite)'
        rec.update(eps_state=eps_state, eps_ref=eps_ref, J_state=J_state, J_ref=J_ref)
        Krho = eps @ self.K
        gCu, KJs = commutator_integrand(r, u, Js, u_sites, self.KJ)
        mr = float(np.max(np.abs(rho))) if rho.size else 0.0
        mj = float(np.max(np.abs(J))) if J.size else 0.0
        rec['kernel_rel_rho'] = float(np.max(np.abs(Krho - rho))) / mr if mr > 0 else NAN
        rec['kernel_rel_J'] = float(np.max(np.abs(KJs - J))) / mj if mj > 0 else NAN
        rec['_gCu'] = gCu
        rec['_gM'] = FOUR_PI * r * r * (rho + u * J)
        rec['Q'] = eps + u_sites * Js
        rec['Q_state'] = eps_state + u_sites * J_state
        rec['q_ref'] = eps_ref + u_sites * J_ref
        rec['p_of'] = None
        if j is not None and HO.get('S_r_ell') is not None and HO.get('r_nodes') is not None:
            try:
                from scipy.interpolate import CubicSpline
                xs = np.asarray(HO['r_nodes'], float)
                ys = np.asarray(HO['S_r_ell'][j], float) * self.pr_scale
                cs = CubicSpline(xs, ys, bc_type='not-a-knot', extrapolate=False)

                def p_of(x, cs=cs, xs=xs, ys=ys):
                    xv = float(x)
                    if xv < xs[0]:
                        return float(ys[0])
                    if xv > xs[-1]:
                        return float(ys[-1])
                    return float(cs(xv))
                rec['p_of'] = p_of
            except Exception as exc:
                rec['stress_error'] = '%s: %s' % (type(exc).__name__, exc)
        if rec['p_of'] is None:
            rec['p_of'] = lambda x: NAN
        self._recs[row] = rec
        return rec

    def Cu(self, rec, Rstar):
        return integral_0_to(self.r, rec['_gCu'], Rstar, self.qdegree)

    def Mquad(self, rec, Rstar):
        return integral_0_to(self.r, rec['_gM'], Rstar, self.qdegree)


# ----------------------------------------------------------------------------
# the pair at a fixed radius (v4 W6)
# ----------------------------------------------------------------------------
def pair_identity(Rstar, A, B, store, rs, which):
    r = store.r
    G = store.G
    ell = store.ell
    deg = store.degree
    out = {'which': which, 'R_star': float(Rstar), 'T_a': A['T'], 'T_b': B['T'], 'row_a': A['row'], 'row_b': B['row']}
    dt = (B['T'] - A['T']) * rs
    out['dt'] = dt
    val = {}
    for tag, X in (('a', A), ('b', B)):
        for name in ('N', 'v', 'rho', 'J', 'u', 'M'):
            val[name + tag], der = local_poly(r, X[name], Rstar, deg)
            if name == 'N':
                val['Np' + tag] = der
        val['p' + tag] = X['p_of'](Rstar)
    dM = val['Mb'] - val['Ma']
    out['dM'] = dM
    out['rate'] = dM / dt
    w = cumulative_weight(ell, store.R_sites, Rstar)
    chi = (store.R_sites <= Rstar).astype(float)
    dQ_state = B['Q_state'] - A['Q_state']
    dq_ref = B['q_ref'] - A['q_ref']
    Cua, Cub = store.Cu(A, Rstar), store.Cu(B, Rstar)
    cols = four_columns(w, chi, dQ_state, dq_ref, Cub - Cua)
    out.update(cols)
    out['rep_defect'] = dM - cols['rep']
    du = B['u_sites'] - A['u_sites']
    Jmid_state = 0.5 * (A['J_state'] + B['J_state'])
    out['col1_flow_change_part'] = float(chi @ (du * Jmid_state))
    out['col1_matter_part'] = cols['col1_flux_old_cut'] - out['col1_flow_change_part']
    out['sites_inside'] = int(chi.sum())
    out['w_minus_chi_support'] = int(np.sum(np.abs(w - chi) > 1e-12))
    for tag, X, Cu in (('a', A, Cua), ('b', B, Cub)):
        Mrep = float(w @ X['Q']) - Cu
        Mq = store.Mquad(X, Rstar)
        out['M_ode_' + tag] = val['M' + tag]
        out['M_rep_' + tag] = Mrep
        out['M_quad_' + tag] = Mq
        out['C_u_' + tag] = Cu
        out['M_rep_minus_quad_' + tag] = Mrep - Mq
        out['M_quad_minus_ode_' + tag] = Mq - val['M' + tag]
    ra = rows_rate_values(Rstar, val['Na'], val['va'], val['rhoa'], val['Ja'], val['pa'])
    rb = rows_rate_values(Rstar, val['Nb'], val['vb'], val['rhob'], val['Jb'], val['pb'])
    out['Nplusv_a'] = ra['Nplusv']
    out['Nplusv_b'] = rb['Nplusv']
    out['Tll_a'] = ra['Tll']
    out['Tll_b'] = rb['Tll']
    out['rows_a_root_form'] = ra['rate_root_form']
    out['rows_a'] = ra['rate_rows']
    out['rows_b'] = rb['rate_rows']
    out['rows_mid'] = 0.5 * (ra['rate_rows'] + rb['rate_rows'])
    out['rows_a_nostress'] = ra['rate_root_nostress']
    out['rows_a_stress'] = ra['rate_root_stress']
    out['defect_mid'] = out['rate'] - out['rows_mid']
    out['defect_a'] = out['rate'] - out['rows_a']
    out['defect_mid_rel'] = out['defect_mid'] / abs(out['rows_mid']) if out['rows_mid'] != 0 else NAN
    out['same_sign_rate_rows_mid'] = bool(np.sign(out['rate']) == np.sign(out['rows_mid'])) if (out['rate'] != 0 and out['rows_mid'] != 0) else None
    for tag in ('a', 'b'):
        out['N_' + tag], out['v_' + tag], out['rho_' + tag], out['J_' + tag], out['p_' + tag], out['s_' + tag] = val['N' + tag], val['v' + tag], val['rho' + tag], val['J' + tag], val['p' + tag], -val['u' + tag]
    lnNp = {}
    for tag in ('a', 'b'):
        ut = val['u' + tag]
        lnNp[tag + '_row'] = -FOUR_PI * G * Rstar * val['J' + tag] / ut if ut != 0 else NAN
        lnNp[tag + '_fd'] = val['Np' + tag] / val['N' + tag] if val['N' + tag] != 0 else NAN
    fc = full_clock_values(Rstar, dt, G, -val['ua'], -val['ub'], val['Na'], val['Nb'], lnNp['a_row'], lnNp['b_row'], val['rhoa'], val['rhob'], val['pa'], val['pb'])
    fcf = full_clock_values(Rstar, dt, G, -val['ua'], -val['ub'], val['Na'], val['Nb'], lnNp['a_fd'], lnNp['b_fd'], val['rhoa'], val['rhob'], val['pa'], val['pb'])
    out['lnNp_row_a'], out['lnNp_fd_a'] = lnNp['a_row'], lnNp['a_fd']
    out['s_t'] = fc['s_t']
    out['clock_residual_mid'] = fc['residual_mid']
    out['clock_residual_a'] = fc['residual_a']
    out['clock_residual_mid_fd'] = fcf['residual_mid']
    out['clock_term_st_over_N_mid'] = fc['term_st_over_N_mid']
    out['clock_term_clock_mid'] = fc['term_clock_mid']
    out['clock_term_matter_mid'] = fc['term_matter_mid']
    ds = -val['ub'] + val['ua']
    out['dM_from_s_exact'] = (Rstar / (2.0 * G)) * (val['ub'] * val['ub'] - val['ua'] * val['ua'])  # v4b: M = R s^2/(2G) at fixed R, s^2 = u^2 (the stray factor of v4 removed)
    out['dM_from_s_minus_dM'] = out['dM_from_s_exact'] - dM
    out['clock_residual_a_times_RN_over_G'] = fc['residual_a'] * Rstar * val['Na'] / G
    out['rate_minus_rows_a_minus_ds2_term'] = out['rate'] - ra['rate_rows'] - (Rstar / (2.0 * G)) * ds * ds / dt
    gt = gate_value(ell, Rstar, store.working)
    out['gate'] = gt['gate']
    out['gate_inside'] = gt['inside']
    return out


# ----------------------------------------------------------------------------
# loading (v4 W6)
# ----------------------------------------------------------------------------
def load_cache(path):
    z = np.load(path, allow_pickle=False)
    rows = np.asarray(z['rows'], int)
    out = {'path': path, 'key': str(np.asarray(z['key']).ravel()[0]), 'rows': rows, 'r_fine': np.asarray(z['r_fine'], float), 'arrays': {}, 'index': {int(row): i for i, row in enumerate(rows)}}
    for name in ('N', 'v', 'rho', 'J'):
        if 'arr_' + name not in z.files:
            raise SystemExit('the cache %s has no arr_%s' % (path, name))
        out['arrays'][name] = np.asarray(z['arr_' + name], float)
    return out


def load_fields(path):
    z = np.load(path, allow_pickle=False)
    F = {'path': path, 'keys': list(z.files), 'R': np.asarray(z['R'], float), 'T': np.asarray(z['T'], float), 'eps_tot': np.asarray(z['eps_tot'], float), 'J_tot': np.asarray(z['J_tot'], float)}
    for key in ('eps_pairs', 'J_pairs'):
        F[key] = np.asarray(z[key], float) if key in z.files else None
    return F


def load_handoff(path, n_sites):
    if not path or not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    HO = {'path': path, 'keys': list(z.files), 'T': np.asarray(z['T'], float) if 'T' in z.files else None}
    S = np.asarray(z['S_r_ell'], float) if 'S_r_ell' in z.files else None
    HO['S_r_ell'] = S
    HO['r_nodes'] = None
    if S is not None:
        nn = S.shape[-1]
        for k in ('r_nodes', 'r', 'R'):
            if k in z.files:
                a = np.asarray(z[k], float)
                if a.ndim == 1 and a.size == nn:
                    HO['r_nodes'] = a
                    break
    for key in ('eps_dust', 'eps_sea', 'eps_R2', 'eps_tot', 'J_dust', 'J_sea', 'J_R2', 'J_tot'):
        HO[key] = None
        if key in z.files:
            a = np.asarray(z[key], float)
            if a.ndim == 2 and a.shape[-1] == n_sites:
                HO[key] = a
    HO['flux_candidates'] = [k for k in z.files if 'flux' in k.lower() or k.lower().startswith('f_face')]
    HO['face_flux'] = None
    for k in HO['flux_candidates']:
        a = np.asarray(z[k], float)
        if a.ndim == 2:
            HO['face_flux'] = (k, a)
            break
    return HO


# ----------------------------------------------------------------------------
# the analysis (v4 W6)
# ----------------------------------------------------------------------------
def analyse(res3, store, rs, offsets, phases, do_inner, max_pairs):
    recs_by_T = {round(_float(r.get('T_over_rs')), 9): r for r in res3.get('records', [])}
    pairs3 = list(res3.get('pairs') or [])
    if max_pairs is not None:
        pairs3 = pairs3[:max(0, int(max_pairs))]
    outer, inner, offs = [], [], {o: [] for o in offsets}
    t0 = time.time()
    for n_p, p in enumerate(pairs3):
        Ta, Tb = round(_float(p.get('T_a')), 9), round(_float(p.get('T_b')), 9)
        ra, rb = recs_by_T.get(Ta), recs_by_T.get(Tb)
        if ra is None or rb is None:
            continue
        A = store.get(int(ra['row']), float(ra['T_over_rs']))
        B = store.get(int(rb['row']), float(rb['T_over_rs']))
        Rstar = _float(ra.get('R_dense'))
        if not np.isfinite(Rstar):
            continue
        o = pair_identity(Rstar, A, B, store, rs, 'outer')
        o['R_b'] = _float(rb.get('R_dense'))
        o['Tll_v3b_a'] = _float(ra.get('Tll_R'))
        o['Tll_a_minus_v3b'] = o['Tll_a'] - o['Tll_v3b_a'] if np.isfinite(o['Tll_v3b_a']) else NAN
        o['kernel_rel_rho_a'], o['kernel_rel_J_a'] = A['kernel_rel_rho'], A['kernel_rel_J']
        o['ref_check_max_abs_a'] = A['ref_check_max_abs']
        Rin = _float(ra.get('R_in'))
        o['R_in_a'] = Rin
        gi = gate_value(store.ell, Rin, store.working)
        o['gate_inner'], o['gate_inner_inside'] = gi['gate'], gi['inside']
        if store.HO is not None and store.HO.get('face_flux') is not None and A['handoff_row'] is not None:
            k, arr = store.HO['face_flux']
            grid = None
            if arr.shape[-1] == store.R_sites.size:
                grid = store.R_sites
            elif store.HO.get('r_nodes') is not None and arr.shape[-1] == store.HO['r_nodes'].size:
                grid = store.HO['r_nodes']
            elif store.HO.get('r_nodes') is not None and arr.shape[-1] == store.HO['r_nodes'].size - 1:
                grid = 0.5 * (store.HO['r_nodes'][1:] + store.HO['r_nodes'][:-1])
            if grid is not None:
                o['face_flux_key'] = k
                o['face_flux_old_cut_a_times_dt'] = float(np.interp(Rstar, grid, arr[A['handoff_row']])) * o['dt']
        outer.append(o)
        if do_inner and np.isfinite(Rin) and Rin > store.r[0] + 2 * store.ell:
            inner.append(pair_identity(Rin, A, B, store, rs, 'inner'))
        for off in offsets:
            x = Rstar + off * store.ell
            if store.r[0] + 2 * store.ell < x < store.r[-1] - 2 * store.ell:
                e = pair_identity(x, A, B, store, rs, 'offset_%g' % off)
                e['offset_ell'] = off
                offs[off].append(e)
        if (n_p + 1) % 25 == 0:
            print('[%d/%d] T/r_s %s -> %s: R* %s, dM %s, columns %s %s %s %s, closure %s, rep defect %s, rate %s, rows mid %s, defect %s, clock residual %s, gate %s (%.0f s)' % (
                n_p + 1, len(pairs3), _fmt(o['T_a']), _fmt(o['T_b']), _fmt(Rstar, 7), _fmt(o['dM'], 5), _fmt(o['col1_flux_old_cut'], 4), _fmt(o['col2_cut_assignment'], 4),
                _fmt(o['col3_averaging_commutator'], 4), _fmt(o['col4_continued_reference'], 4), _fmt(o['closure'], 2), _fmt(o['rep_defect'], 2), _fmt(o['rate'], 4),
                _fmt(o['rows_mid'], 4), _fmt(o['defect_mid'], 4), _fmt(o['clock_residual_mid'], 3), _fmt(o['gate'], 3), time.time() - t0), flush=True)
    return outer, inner, offs


def summarize(outer, phases):
    st = {'n_pairs': len(outer)}
    if not outer:
        return st

    def arr(key):
        return np.array([_float(o.get(key)) for o in outer], float)
    for key in ('dM', 'col1_flux_old_cut', 'col2_cut_assignment', 'col3_averaging_commutator', 'col4_continued_reference', 'sum4', 'rep', 'col1_flow_change_part',
                'col1_matter_part', 'rate', 'rows_a', 'rows_b', 'rows_mid', 'defect_mid', 'defect_a', 'rows_a_nostress', 'rows_a_stress'):
        a = arr(key)
        st['sum_' + key] = _nan(np.sum, a)
        st['median_' + key] = _nan(np.median, a)
        st['max_abs_' + key] = _nan(np.max, np.abs(a))
    for key in ('closure', 'rep_defect', 'M_rep_minus_quad_a', 'M_quad_minus_ode_a', 'Nplusv_a', 'Tll_a_minus_v3b', 'dM_from_s_minus_dM', 'kernel_rel_rho_a', 'kernel_rel_J_a', 'ref_check_max_abs_a'):
        a = arr(key)
        st['max_abs_' + key] = _nan(np.max, np.abs(a))
        st['median_abs_' + key] = _nan(np.median, np.abs(a))
    a = arr('rep_defect')
    d = arr('dM')
    with np.errstate(divide='ignore', invalid='ignore'):
        st['max_abs_rep_defect_over_dM'] = _nan(np.max, np.abs(a) / np.abs(d))
        st['median_abs_defect_mid_over_rows_mid'] = _nan(np.median, np.abs(arr('defect_mid')) / np.abs(arr('rows_mid')))
        st['median_abs_defect_mid_over_rate'] = _nan(np.median, np.abs(arr('defect_mid')) / np.abs(arr('rate')))
    ss = [o.get('same_sign_rate_rows_mid') for o in outer if o.get('same_sign_rate_rows_mid') is not None]
    st['frac_same_sign_rate_rows_mid'] = float(np.mean(ss)) if ss else NAN
    st['n_rows_mid_negative'] = int(np.sum(arr('rows_mid') < 0))
    st['n_rate_negative'] = int(np.sum(arr('rate') < 0))
    st['n_Tll_a_negative'] = int(np.sum(arr('Tll_a') < 0))
    for key in ('clock_residual_mid', 'clock_residual_a', 'clock_residual_mid_fd', 'clock_term_st_over_N_mid', 'clock_term_clock_mid', 'clock_term_matter_mid', 's_t'):
        a = arr(key)
        st['median_' + key] = _nan(np.median, a)
        st['median_abs_' + key] = _nan(np.median, np.abs(a))
        st['max_abs_' + key] = _nan(np.max, np.abs(a))
    g = arr('gate')
    st['gate_min'], st['gate_max'], st['gate_median'] = _nan(np.min, g), _nan(np.max, g), _nan(np.median, g)
    st['n_gate_inside'] = int(sum(1 for o in outer if o.get('gate_inside')))
    gi = arr('gate_inner')
    st['gate_inner_min'], st['gate_inner_max'] = _nan(np.min, gi), _nan(np.max, gi)
    st['n_gate_inner_inside'] = int(sum(1 for o in outer if o.get('gate_inner_inside')))
    st['n_inner_roots'] = int(np.sum(np.isfinite(gi)))
    bounds = [-np.inf] + list(phases) + [np.inf]
    st['phases'] = []
    Tb = arr('T_b')
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        mk = (Tb > lo) & (Tb <= hi)
        if not mk.any():
            continue
        ph = {'T_b_from': float(np.min(Tb[mk])), 'T_b_to': float(np.max(Tb[mk])), 'n': int(mk.sum())}
        for key in ('dM', 'col1_flux_old_cut', 'col2_cut_assignment', 'col3_averaging_commutator', 'col4_continued_reference', 'rows_mid', 'defect_mid'):
            ph['sum_' + key] = float(np.nansum(arr(key)[mk]))
        rm = arr('rows_mid')[mk]
        dm = arr('defect_mid')[mk]
        ph['sum_rows_mid_times_dt'] = float(np.nansum(rm * arr('dt')[mk]))
        ph['sum_defect_mid_times_dt'] = float(np.nansum(dm * arr('dt')[mk]))
        st['phases'].append(ph)
    return st


# ----------------------------------------------------------------------------
# writers (v4 W6)
# ----------------------------------------------------------------------------
def _stack(dicts, prefix):
    out = {}
    if not dicts:
        return out
    keys = set()
    for d in dicts:
        keys |= set(d.keys())
    for key in sorted(keys):
        if key.startswith('_'):
            continue
        vals = [d.get(key) for d in dicts]
        if all(x is None or isinstance(x, (bool, int, float, np.bool_, np.integer, np.floating)) for x in vals):
            out[prefix + key] = np.array([NAN if x is None else float(x) for x in vals], float)
    return out


def write_md(path, meta, st, outer, inner, offs, st_in, st_off):
    L = ['# FIRSTLAW_records_v4b (v2.3, busi W6) -- %s' % meta['tag'], '']
    L.append('## Inputs')
    for k in ('v3b_json', 'v3b_cache', 'fields_npz', 'handoff_npz', 'kernel', 'quadrature', 'cache_key'):
        L.append('- %s: `%s`' % (k, meta.get(k)))
    L.append('- fields npz keys: %s' % ', '.join(meta.get('fields_keys') or []))
    L.append('- handoff npz keys: %s' % ', '.join(meta.get('handoff_keys') or []))
    L.append('- G = %s, ell = sigma = %s, r_s = %s, dense spacing %s, %d sites; the record step dt = (T_b - T_a) r_s in lattice units; %d pairs (v3b\'s included pairs), %d with an inner root beside' % (
        _fmt(meta['G']), _fmt(meta['ell']), _fmt(meta['rs']), _fmt(meta['h_fine']), meta['n_sites'], st.get('n_pairs', 0), st_in.get('n_pairs', 0)))
    L.append('- the per-site split: %s; the momentum split: %s; max |reference from the split - eps_R2| %s; the face flux: %s' % (
        meta.get('split_eps'), meta.get('split_J'), _fmt(st.get('max_abs_ref_check_max_abs_a'), 3), meta.get('face_flux_note')))
    L.append('- kernel check (the analytic shell Gaussian against the engine\'s dense rho_ell and J_ell of the cache, max |difference| over the grid relative to the maximum, over the records of the pairs): rho max %s, J max %s' % (
        _fmt(st.get('max_abs_kernel_rel_rho_a'), 3), _fmt(st.get('max_abs_kernel_rel_J_a'), 3)))
    L.append('- the stress p_r: the handoff\'s S_r_ell read as the radial pressure (v2.2\'s four conditions carried: the stress module\'s version, the normalization by the ratio test alone, the companion row by the T key, the between-node remainder); T_ll(R_a) against v3b\'s Tll_R: max |difference| %s' % _fmt(st.get('max_abs_Tll_a_minus_v3b'), 3))
    L.append('')
    L.append('## Definitions (as computed; the derivation in RUN_firstlaw_records_v4b.md)')
    L.append('- M(r) = r u^2/(2G) of the dense solution, u = v/N; Delta M(r) = M_b(r) - M_a(r) at the fixed radius r = R_a (the old root) by the degree-%d local polynomial on the dense grid.' % meta['degree'])
    L.append('- Q_n = eps_n + u(R_n) J_n (eps_tot, J_tot of the fields npz; u the dense flow at the site); Q^S_n from the handoff\'s dust + sea, q^P_n = Q^S_n - Q_n the continued reference; w_n(R) = (1/2)[erf((R - R_n)/(sqrt2 ell)) + erf((R + R_n)/(sqrt2 ell))] - (ell/(sqrt(2 pi) R_n))[e^(-(R - R_n)^2/2ell^2) - e^(-(R + R_n)^2/2ell^2)]; chi_n = 1_(R_n <= R).')
    L.append('- col1 = sum chi dQ^S (the flux through the old cut as the sharp-cut balance of the state\'s charge; its flow-change part sum chi du J^S_mid beside); col2 = sum (w - chi) dQ^S; col3 = -Delta C_u, C_u(R) = int_0^R 4 pi s^2 [K(uJ) - u K(J)] ds (%s); col4 = -sum w dq^P; sum4 = col1 + col2 + col3 + col4; rep = sum w dQ - Delta C_u; closure = sum4 - rep (roundoff); rep defect = Delta M - rep (the engine\'s representation of the sources against the analytic kernel; the quadrature\'s own error below 1e-9).' % meta['quadrature'])
    L.append('- rows: F = 4 pi R^2 [(N^2 + v^2) J + N v (rho + p_r)], the rows\' rate -F/N; at the root of a it equals 4 pi N R^2 T_ll, T_ll = rho + p_r - 2J; rows_b = -F_b/N_b at the same radius; rows_mid their mean; defect = Delta M/dt - rows_mid.')
    L.append('- full clock: s = -v/N; residual = (s_b - s_a)/(dt N) + (1 + s^2)(ln N)\' - 4 pi G R (rho + p_r), (ln N)\' = -4 pi G R J/u (the momentum row; the dense derivative beside), the midpoint rule over the two records and the endpoint rule at a; at the root it equals s_t/N - 4 pi G R T_ll and (R N/G) x residual_a = Delta M/dt - rows_a - (R/2G) Delta s^2/dt.')
    L.append('- gate = ell/R (sigma |K_t| at a root), inside iff <= %s.' % _fmt(meta['working']))
    L.append('')
    L.append('## Headline (outer roots, %d pairs)' % st.get('n_pairs', 0))
    if st.get('n_pairs'):
        L.append('- closure of the 4 columns: max |sum4 - rep| %s (roundoff); representation defect Delta M - rep: max |.| %s, max relative to |Delta M| %s; per record |M_rep - M_quad| max %s, |M_quad - M_ode| max %s (M about %s)' % (
            _fmt(st['max_abs_closure'], 2), _fmt(st['max_abs_rep_defect'], 3), _fmt(st['max_abs_rep_defect_over_dM'], 3), _fmt(st['max_abs_M_rep_minus_quad_a'], 3), _fmt(st['max_abs_M_quad_minus_ode_a'], 3), _fmt(outer[0].get('M_ode_a'), 4)))
        L.append('- sums over the pairs: Delta M %s = col1 %s + col2 %s + col3 %s + col4 %s (+ rep defect); col1\'s flow-change part %s, matter part %s' % (
            _fmt(st['sum_dM']), _fmt(st['sum_col1_flux_old_cut']), _fmt(st['sum_col2_cut_assignment']), _fmt(st['sum_col3_averaging_commutator']), _fmt(st['sum_col4_continued_reference']), _fmt(st['sum_col1_flow_change_part']), _fmt(st['sum_col1_matter_part'])))
        L.append('- the rows\' rate against the records: sum of Delta M/dt %s, sum of rows_mid %s, sum of the defect %s; median |defect|/|rows_mid| %s, median |defect|/|rate| %s; the rate and rows_mid of the same sign in a fraction %s of the pairs; rows_mid negative at %d, the rate negative at %d, T_ll(a) negative at %d of %d' % (
            _fmt(st['sum_rate']), _fmt(st['sum_rows_mid']), _fmt(st['sum_defect_mid']), _fmt(st['median_abs_defect_mid_over_rows_mid'], 3), _fmt(st['median_abs_defect_mid_over_rate'], 3), _fmt(st['frac_same_sign_rate_rows_mid'], 3),
            st['n_rows_mid_negative'], st['n_rate_negative'], st['n_Tll_a_negative'], st['n_pairs']))
        L.append('- the stress\'s share of the rows\' rate at a: sum 4 pi N R^2 (rho - 2J) %s, sum 4 pi N R^2 p_r %s' % (_fmt(st['sum_rows_a_nostress']), _fmt(st['sum_rows_a_stress'])))
        L.append('- full-clock residual at the roots: median %s (|.| median %s, max %s) by the midpoint rule, %s by the endpoint rule, %s with the dense (ln N)\'; its terms (medians): s_t/N %s, (1 + s^2)(ln N)\' %s, -4 pi G R (rho + p_r) %s' % (
            _fmt(st['median_clock_residual_mid'], 3), _fmt(st['median_abs_clock_residual_mid'], 3), _fmt(st['max_abs_clock_residual_mid'], 3), _fmt(st['median_clock_residual_a'], 3), _fmt(st['median_clock_residual_mid_fd'], 3),
            _fmt(st['median_clock_term_st_over_N_mid'], 3), _fmt(st['median_clock_term_clock_mid'], 3), _fmt(st['median_clock_term_matter_mid'], 3)))
        L.append('- the exact relation at fixed R: max |(R/2G)(s_b^2 - s_a^2) - Delta M| %s' % _fmt(st['max_abs_dM_from_s_minus_dM'], 2))
        L.append('- gate sigma/R at the outer roots: %s to %s (median %s), inside the working value at %d of %d; at the inner roots: %s to %s, inside at %d of %d' % (
            _fmt(st['gate_min'], 3), _fmt(st['gate_max'], 3), _fmt(st['gate_median'], 3), st['n_gate_inside'], st['n_pairs'], _fmt(st['gate_inner_min'], 3), _fmt(st['gate_inner_max'], 3), st['n_gate_inner_inside'], st['n_inner_roots']))
        if st.get('phases'):
            L.append('')
            L.append('| phase (T_b) | pairs | sum Delta M | col1 | col2 | col3 | col4 | sum rows_mid x dt | sum defect x dt |')
            L.append('|---|---|---|---|---|---|---|---|---|')
            for ph in st['phases']:
                L.append('| %s to %s | %d | %s | %s | %s | %s | %s | %s | %s |' % (_fmt(ph['T_b_from']), _fmt(ph['T_b_to']), ph['n'], _fmt(ph['sum_dM'], 5), _fmt(ph['sum_col1_flux_old_cut'], 5), _fmt(ph['sum_col2_cut_assignment'], 5),
                                                                                 _fmt(ph['sum_col3_averaging_commutator'], 5), _fmt(ph['sum_col4_continued_reference'], 5), _fmt(ph['sum_rows_mid_times_dt'], 5), _fmt(ph['sum_defect_mid_times_dt'], 5)))
    if st_off:
        L.append('')
        L.append('## Across the root (the fixed radius R_a + offset x ell; medians over the pairs)')
        L.append('| offset (ell) | n | median Delta M/dt | median rows_mid | median defect | median |defect|/|rows_mid| | median clock residual (mid) |')
        L.append('|---|---|---|---|---|---|---|')
        for off in sorted(st_off):
            so = st_off[off]
            L.append('| %s | %d | %s | %s | %s | %s | %s |' % (_fmt(off), so.get('n_pairs', 0), _fmt(so.get('median_rate'), 4), _fmt(so.get('median_rows_mid'), 4), _fmt(so.get('median_defect_mid'), 4),
                                                       _fmt(so.get('median_abs_defect_mid_over_rows_mid'), 3), _fmt(so.get('median_clock_residual_mid'), 3)))
    L.append('')
    L.append('## Per pair at the outer root R_a (the old cut)')
    L.append('| T_a | T_b | R_a | R_b | dM | col1 | col1 flow part | col2 | col3 | col4 | closure | rep defect | dM/dt | rows_a | rows_b | rows_mid | defect | T_ll(a) | clock res mid | clock res a | gate | R_in | gate_in |')
    L.append('|' + '---|' * 23)
    for o in outer:
        L.append('| %s |' % ' | '.join([_fmt(o['T_a']), _fmt(o['T_b']), _fmt(o['R_star'], 8), _fmt(o.get('R_b'), 8), _fmt(o['dM'], 5), _fmt(o['col1_flux_old_cut'], 5), _fmt(o['col1_flow_change_part'], 4), _fmt(o['col2_cut_assignment'], 5),
                                        _fmt(o['col3_averaging_commutator'], 5), _fmt(o['col4_continued_reference'], 5), _fmt(o['closure'], 2), _fmt(o['rep_defect'], 2), _fmt(o['rate'], 5), _fmt(o['rows_a'], 5), _fmt(o['rows_b'], 5),
                                        _fmt(o['rows_mid'], 5), _fmt(o['defect_mid'], 5), _fmt(o['Tll_a'], 4), _fmt(o['clock_residual_mid'], 4), _fmt(o['clock_residual_a'], 4), _fmt(o['gate'], 3), _fmt(o.get('R_in_a'), 5), _fmt(o.get('gate_inner'), 3)]))
    if inner:
        L.append('')
        L.append('## Per pair at the inner root R_in,a (outside the gate; N small)')
        L.append('| T_a | T_b | R_in | dM | col1 | col2 | col3 | col4 | closure | rep defect | dM/dt | rows_a | rows_b | defect | T_ll(a) | clock res mid | gate |')
        L.append('|' + '---|' * 17)
        for o in inner:
            L.append('| %s |' % ' | '.join([_fmt(o['T_a']), _fmt(o['T_b']), _fmt(o['R_star'], 7), _fmt(o['dM'], 5), _fmt(o['col1_flux_old_cut'], 5), _fmt(o['col2_cut_assignment'], 5), _fmt(o['col3_averaging_commutator'], 5), _fmt(o['col4_continued_reference'], 5),
                                            _fmt(o['closure'], 2), _fmt(o['rep_defect'], 2), _fmt(o['rate'], 5), _fmt(o['rows_a'], 5), _fmt(o['rows_b'], 5), _fmt(o['defect_mid'], 5), _fmt(o['Tll_a'], 4), _fmt(o['clock_residual_mid'], 4), _fmt(o['gate'], 3)]))
    L.append('')
    L.append('## Label')
    L.append('records, computed; one history, one averaging scale, one record step, one reference convention (as v2.2); the dense fields of the v3b cache; the identity exact in the kernel representation and closed to roundoff in its four columns, the dense Delta M against it at the representation defect printed (the quadrature of C_u and M_quad 6th order, its own error below 1e-9 at the dense spacing, so the defect reads the engine\'s source representation); column 1 is the sharp-cut balance of the state\'s charge (the flux through the old cut up to the fields\' power on the enclosed charge, which the stored densities do not separate; the flow-change part printed); the rows\' rate carries the stress with v2.2\'s four conditions; the full-clock residual is the same evolution-row defect in another normalization; the gate is sigma/R with sigma = ell. FIRSTLAW_records_v4b on FIRSTLAW_records_v2.2 and FIRSTLAW_rows_v1.')
    with open(path, 'w') as fh:
        fh.write('\n'.join(L) + '\n')


# ----------------------------------------------------------------------------
# main (v4 W6; v4b)
# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description='FIRSTLAW_records_v4b (v2.3, W6): the exact fixed-radius mass-rate identity in 4 columns, the full-clock residual and the gate at every dense root, on a FIRSTLAW_records_v3b pass')
    ap.add_argument('--v3b-json', required=True, help='firstlaw_records_v3b_<tag>.json of the v3b run')
    ap.add_argument('--v3b-cache', default=None, help='firstlaw_dense_v3b_<tag>_cache.npz (default: beside the v3b json)')
    ap.add_argument('--json', default=None, help='collapse_<tag>.json (default: the v3b json\'s inputs)')
    ap.add_argument('--npz', default=None, help='collapse_<tag>_fields.npz (default: the v3b json\'s inputs)')
    ap.add_argument('--handoff', default=None, help='collapse_<tag>_handoff.npz (default: the v3b json\'s stress_meta path, else beside the fields npz)')
    ap.add_argument('--out', default=None, help='output directory (default: the v3b json\'s directory)')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--degree', type=int, default=4, help='degree of the local polynomial for the values and derivatives at a radius (v3b\'s rule)')
    ap.add_argument('--quad-degree', dest='quad_degree', type=int, default=6, help='degree of the local polynomial for the remaining and partial cells of the fixed-radius quadrature (Boole on the 4-cell panels)')
    ap.add_argument('--offsets-ell', dest='offsets_ell', default='-2,-1,-0.5,0.5,1,2')
    ap.add_argument('--phases', default='14.75,17.5', help='T_b boundaries of the phases summed separately (v2.2: the root\'s outward then inward passage, then the late root)')
    ap.add_argument('--no-inner', dest='no_inner', action='store_true')
    ap.add_argument('--max-pairs', dest='max_pairs', type=int, default=None)
    ap.add_argument('--kernel', choices=('analytic', 'engine'), default='analytic', help='analytic: the shell Gaussian in closed form (checked against the cache\'s dense sources); engine: SmoothedSources5 with a unit source per site')
    ap.add_argument('--v3b-dir', dest='v3b_dir', default='<internal>/code')
    ap.add_argument('--engine-path', dest='engine_path', default=None)
    ap.add_argument('--engine-module', dest='engine_module', default=None)
    ap.add_argument('--pr-scale', dest='pr_scale', type=float, default=1.0)
    ap.add_argument('--gate-working-value', dest='working', type=float, default=0.1)
    a = ap.parse_args(argv)
    t0 = time.time()
    with open(a.v3b_json) as fh:
        res3 = json.load(fh)
    tag = a.tag or res3.get('tag') or 'records'
    inputs = res3.get('inputs') or {}
    json_path = a.json or inputs.get('json')
    npz_path = a.npz or inputs.get('npz')
    v3b_dir_out = os.path.dirname(os.path.abspath(a.v3b_json))
    cache_path = a.v3b_cache or os.path.join(v3b_dir_out, 'firstlaw_dense_v3b_%s_cache.npz' % tag)
    handoff = a.handoff or ((res3.get('stress_meta') or {}).get('path')) or (npz_path[:-len('_fields.npz')] + '_handoff.npz' if npz_path and npz_path.endswith('_fields.npz') else None)
    out_dir = a.out or v3b_dir_out
    os.makedirs(out_dir, exist_ok=True)
    G = float(res3['G'])
    rs = float(res3['rs'])
    ell = float(res3['ell'])
    cache = load_cache(cache_path)
    F = load_fields(npz_path)
    HO = load_handoff(handoff, F['R'].size)
    r_fine = cache['r_fine']
    h_fine = float(r_fine[1] - r_fine[0])
    if HO is None:
        print('NOTE: no handoff npz at %s: no stress (the rows\' rate and the full clock get p_r = nan), no state/reference split (column 4 is 0 and column 1 carries the subtracted charge)' % handoff)
    prov = None
    if a.kernel == 'engine':
        sa = (res3.get('provenance') or {}).get('solver_args') or {}
        support_rel = _float((sa.get('support_rel') or {}).get('value')) if isinstance(sa.get('support_rel'), dict) else 1e-14
        if not np.isfinite(support_rel):
            support_rel = 1e-14
        eprov = (res3.get('provenance') or {}).get('engine') or {}
        K, KJ, prov = kernel_matrix_engine(a.v3b_dir, a.engine_path or eprov.get('path') or a.v3b_dir, a.engine_module or eprov.get('module') or 'engine_fields_v16', F['R'], ell,
                                           float(res3['grid']['r_fine_last']), support_rel, r_fine)
    else:
        K = kernel_matrix(ell, F['R'], r_fine)
        KJ = K
    store = RecordStore(cache, F, HO, K, KJ, G, ell, a.degree, a.quad_degree, a.pr_scale)
    store.working = float(a.working)
    offsets = [float(x) for x in a.offsets_ell.split(',') if x.strip()]
    phases = [float(x) for x in a.phases.split(',') if x.strip()]
    quadrature = ("Boole's rule on the dense grid's 4-cell panels from the centre, the remaining cells and the partial cell by the exact integral of the degree-%d local polynomial "
                  "(6th order; the leading error (2/945) h^6 [g^(5)(R) - g^(5)(0)]; exact on cubic pieces with knots at the panel boundaries)" % int(a.quad_degree))
    print('FIRSTLAW_records_v4b %s: v3b json %s (%d records, %d pairs), cache %s (%d rows, key %s...), fields %s, handoff %s; G %s, ell %s, r_s %s, dense spacing %s; kernel %s; quadrature %s (%.1f s so far)' % (
        tag, a.v3b_json, len(res3.get('records', [])), len(res3.get('pairs', [])), cache_path, cache['rows'].size, cache['key'][:12], npz_path, handoff, _fmt(G), _fmt(ell), _fmt(rs), _fmt(h_fine), a.kernel,
        quadrature, time.time() - t0))
    outer, inner, offs = analyse(res3, store, rs, offsets, phases, not a.no_inner, a.max_pairs)
    st = summarize(outer, phases)
    st_in = summarize(inner, phases) if inner else {'n_pairs': 0}
    st_off = {off: summarize(lst, phases) for off, lst in offs.items() if lst}
    first = outer[0] if outer else {}
    any_rec = next(iter(store._recs.values())) if store._recs else {}
    meta = {'tag': tag, 'version': VERSION, 'v3b_json': a.v3b_json, 'v3b_cache': cache_path, 'fields_npz': npz_path, 'handoff_npz': handoff, 'records_json': json_path, 'kernel': a.kernel,
            'quadrature': quadrature, 'degree': int(a.degree), 'quad_degree': int(a.quad_degree), 'cache_key': cache['key'],
            'fields_keys': F['keys'], 'handoff_keys': (HO or {}).get('keys'), 'G': G, 'ell': ell, 'rs': rs, 'h_fine': h_fine, 'n_sites': int(F['R'].size), 'working': float(a.working),
            'split_eps': any_rec.get('split_eps'), 'split_J': any_rec.get('split_J'),
            'face_flux_note': ('key %s read at the old cut, printed beside column 1, not in the identity' % first['face_flux_key']) if first.get('face_flux_key') else 'no face-flux array in the handoff (candidates %s)' % ((HO or {}).get('flux_candidates')),
            'engine_kernel_provenance': prov, 'threads': THREADS, 'options': vars(a)}
    base = os.path.join(out_dir, 'firstlaw_records_v4b_%s' % tag)
    res = {'meta': meta, 'stats_outer': st, 'stats_inner': st_in, 'stats_offsets': {str(k): v for k, v in st_off.items()}, 'pairs_outer': outer, 'pairs_inner': inner,
           'pairs_offsets': {str(k): v for k, v in offs.items()}, 'seconds': time.time() - t0}
    with open(base + '.json', 'w') as fh:
        json.dump(_jsonable(res), fh, indent=1)
    arrays = {'r_fine': r_fine, 'R_sites': F['R']}
    arrays.update(_stack(outer, 'pair_'))
    arrays.update(_stack(inner, 'inner_'))
    for off, lst in offs.items():
        arrays.update(_stack(lst, 'off_%g_' % off))
    arrays['stats_json'] = np.array(json.dumps(_jsonable(st)))
    np.savez(base + '.npz', **arrays)
    write_md(base + '.md', meta, st, outer, inner, offs, st_in, st_off)
    if outer:
        print('V4B CLOSURE: max |sum4 - rep| %s (roundoff expected); representation defect max |dM - rep| %s (max relative to |dM| %s); |M_rep - M_quad| max %s, |M_quad - M_ode| max %s; kernel check rho %s J %s; T_ll(a) vs v3b max %s' % (
            _fmt(st['max_abs_closure'], 2), _fmt(st['max_abs_rep_defect'], 3), _fmt(st['max_abs_rep_defect_over_dM'], 3), _fmt(st['max_abs_M_rep_minus_quad_a'], 3), _fmt(st['max_abs_M_quad_minus_ode_a'], 3),
            _fmt(st['max_abs_kernel_rel_rho_a'], 3), _fmt(st['max_abs_kernel_rel_J_a'], 3), _fmt(st['max_abs_Tll_a_minus_v3b'], 3)))
        print('V4B COLUMNS (sums over %d pairs): dM %s = col1 %s (flow-change part %s) + col2 %s + col3 %s + col4 %s' % (
            st['n_pairs'], _fmt(st['sum_dM']), _fmt(st['sum_col1_flux_old_cut']), _fmt(st['sum_col1_flow_change_part']), _fmt(st['sum_col2_cut_assignment']), _fmt(st['sum_col3_averaging_commutator']), _fmt(st['sum_col4_continued_reference'])))
        print('V4B ROWS: sum dM/dt %s against sum rows_mid %s, sum defect %s; median |defect|/|rows_mid| %s; same sign fraction %s; rows_mid < 0 at %d, rate < 0 at %d, T_ll(a) < 0 at %d of %d; stress share sums: (rho - 2J) %s, p_r %s' % (
            _fmt(st['sum_rate']), _fmt(st['sum_rows_mid']), _fmt(st['sum_defect_mid']), _fmt(st['median_abs_defect_mid_over_rows_mid'], 3), _fmt(st['frac_same_sign_rate_rows_mid'], 3), st['n_rows_mid_negative'], st['n_rate_negative'],
            st['n_Tll_a_negative'], st['n_pairs'], _fmt(st['sum_rows_a_nostress']), _fmt(st['sum_rows_a_stress'])))
        print('V4B FULL CLOCK at the roots: residual median %s (|.| median %s, max %s) midpoint; %s endpoint; terms s_t/N %s, (1 + s^2)(ln N)\' %s, -4 pi G R (rho + p_r) %s; exact relation max %s' % (
            _fmt(st['median_clock_residual_mid'], 3), _fmt(st['median_abs_clock_residual_mid'], 3), _fmt(st['max_abs_clock_residual_mid'], 3), _fmt(st['median_clock_residual_a'], 3), _fmt(st['median_clock_term_st_over_N_mid'], 3),
            _fmt(st['median_clock_term_clock_mid'], 3), _fmt(st['median_clock_term_matter_mid'], 3), _fmt(st['max_abs_dM_from_s_minus_dM'], 2)))
        print('V4B GATE sigma/R: outer %s to %s (median %s), inside at %d of %d; inner %s to %s, inside at %d of %d' % (
            _fmt(st['gate_min'], 3), _fmt(st['gate_max'], 3), _fmt(st['gate_median'], 3), st['n_gate_inside'], st['n_pairs'], _fmt(st['gate_inner_min'], 3), _fmt(st['gate_inner_max'], 3), st['n_gate_inner_inside'], st['n_inner_roots']))
    print('wrote %s.{json,npz,md} in %.1f s' % (base, time.time() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
