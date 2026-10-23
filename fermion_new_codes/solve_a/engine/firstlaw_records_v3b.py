#!/usr/bin/env python3
"""
firstlaw_records_v3b.py -- FIRSTLAW_records_v3b: version 3 with the radial stress read from a companion npz (the engine's handoff,
collapse_<tag>_handoff.npz beside the fields npz, where S_r_ell is already the engine's kernel average on the node grid r_nodes; the
option --pr-key handoff:S_r_ell, --pt-key handoff:S_t_ell for the tangential stress, reported only), its normalization checked against
rho_ell and J_ell (the companion's energy densities on the nodes against the dense rho_ell, N2avg against the dense N^2, the ratios of
the stress to rho_ell and J_ell against the node shell volume), the node-to-dense-grid interpolation by a not-a-knot cubic spline with
its stride-2 error estimate and the T_ll sign check at the nodes bracketing the root, the outward flux F(R) = -4 pi R^2 N_R^2 T_ll of
the continuity identity with the pairwise balance against W_ext/dt, every W variant of the pairs summed, the pairwise tail differences
beside the pair floor, and the bitwise check against a version-2 json matched by T (records by T_over_rs, pairs by (T_a, T_b); the
set-dependent stats and matching compared only when the record counts agree).  Written from firstlaw_records_v3.py by
build_firstlaw_records_v3b.py (exact string replacements); the version-2 computation is not touched.  Outputs firstlaw_records_v3b_<tag>.*
and the cache firstlaw_dense_v3b_<tag>_cache.npz.  Version 3's own description follows.

firstlaw_records_v3.py -- FIRSTLAW_records_v3: version 2 plus, per record with a simple dense root and per pair, the null-null
component T_ll = rho + p_r - 2J at the root with its profile across the root on the dense solution, and the row defect at every
dense root (the final plan's item 6).  Written from firstlaw_records_v2.py by build_firstlaw_records_v3.py (exact string
replacements); the version-2 computation of the three columns, the closure, the floors, the flags and the exclusions is not
touched, so on the same records their values are bitwise those of version 2 (--check-against <version-2 json> verifies this
after writing and prints the BITWISE line).

The version-3 additions (the constants and functions under the version-3 banner after kappa_formula, and the v3_* calls spliced
into the version-2 functions):
  T_ll    rho_ell + p_ell - 2 J_ell at the outer dense root R (exact kernel values, as rho_R and J_R already are) and at the inner
          zero; rho_ell + p_ell + 2 J_ell (the other null direction) and rho_ell - 2 J_ell (without the stress) beside it; the
          profile on the dense grid, its samples at R + k ell and its sign changes in a window around R; per pair the values at
          both roots, their mean and the midpoint fields' value at R_mid.  p_ell is the engine's kernel average (SmoothedSources5,
          the operator that gives rho_ell from eps_tot) of the stored per-site radial stress, whose npz key is found by name or
          given by --pr-key (the md section 'stress_normalization' and --pr-reading / --pr-scale for the reading); when no stress
          is stored, T_ll is left empty and the md says so.
  (e, g)  the two constraint rows' defects e = M' - 4 pi r^2 (rho + u J) and g = N'/N + 4 pi G r J/u at every dense zero of N + v,
          from the arrays the applied identity's defect integral already uses (the 4th-order stencil on the dense grid), relative
          to the rows' own terms; e(R) also read from kappa_fd - kappa_formula; the applied identity's kappa correction -N_R G e/R
          and its column-2 effect.
  --seed-cache     a version-2 dense cache read before this run's own (no re-solve: the dense pass is unchanged);
  --check-against  the comparison with a version-2 json (the BITWISE line and <base>_bitwise.json).
Outputs: firstlaw_records_v3_<tag>.{json,npz,md} (the npz adds rec_Tll_profile, rec_p_profile, rec_Tll_window and the per-record
and per-pair columns), the cache firstlaw_dense_v3_<tag>_cache.npz.  Version 2's own description follows.

The first-law identity of FIRSTLAW_rows_v1 (hreduce) between consecutive stored records that both carry a
simple dense root of N + v, evaluated on the DENSE constraint solution, re-solved per record from the stored
sources with the engine's own solver (version 1 read the stored site arrays with eps_tot, J_tot as the rows'
rho, J; the run showed that reading to be wrong on both counts, see the builder notes and section "rows"):

    delta E  =  (kappa_R / (8 pi G)) delta A  +  W_ext

    E        =  the record's collar charge (key 'E'); the re-solved E_collar = W(r_c)/(2g) printed beside it
    R        =  the outer dense zero of N + v of the re-solved solution (the engine's own trapping_radius on
                the dense functions, an independent brentq scan beside it), checked against the record's
                'trapping_zeros' (count and every value within --zero-tol; reported, never widened); A = 4 pi R^2
    kappa_R  =  d_r(N + v)|_R from the equations the dense functions integrate,
                N' = -4 pi g r J_ell N^2 / v,  v' = (W' - W/r)/(2 r v),  W' = 8 pi g r^2 N^2 rho_ell,
                which at the root (v = -N) equals (N_R/2R)[1 - 8 pi g R^2 (rho_ell(R) - J_ell(R))]; the Richardson
                centred difference of the dense N + v at h = ell/64 and the stored 'kappa_T' are printed beside it
    W_ext    =  4 pi int_R^{r_c} r^2 (N delta rho_ell + v delta J_ell) dr on the dense grid (spacing ell/--fine),
                the Eulerian differences at fixed r, N and v the dense functions, rho_ell and J_ell the engine's
                smoothed (averaged) sources built from the stored eps_tot, J_tot

The rows in the solver's own normalization (engine_fields_v16.integrate_UL.rhs_pos and the final W integration of
solve_constraints_ivp; stated from the code, not assumed):
    U' = 8 pi g r^2 (rho_ell + q J_ell),   (ln N)' = -4 pi g r J_ell / q,   q = -sqrt(U/r) = v/N,
    W' = 8 pi g r^2 N^2 rho_ell with N(r_c) = 1,   v = -sqrt(W/r),   E = W(r_c)/(2g),
so with M = U/(2g) = r u^2/(2g) these are M' = 4 pi r^2 (rho + u J) and N'/N = -4 pi G r J/u with rho = rho_ell,
J = +J_ell, G = g.  rho_ell = SmoothedSources5(eps_tot, J_tot, ...).rho is the Gaussian-kernel average (standard
deviation ell) PER UNIT VOLUME: rho_ell(0) = sum_n eps_tot[n] (2 pi ell^2)^(-3/2) exp(-R_n^2/(2 ell^2))
(centre_coefficients), so eps_tot[n] is the energy of site n, not a density, and int 4 pi r^2 rho_ell dr = sum eps_tot
(the engine's charge_conservation_quadrature, printed per record); J_ell = <P_r> = -J_r of the goal's display (the
module's DICTIONARY line), entering the momentum row with the sign above.

Kept from version 1: the midpoint rule in the record index with the endpoint evaluations as the record-step
discretization error, the second radial rules (Simpson, half grid), the floors from the records' own numbers, the
premise flags and exclusions, the three outputs (.json, .npz, .md), the applied-identity diagnostic with the row
defects retained, the no-work and vacuum comparisons.  New: the reproduction check of the zeros, the row audit on the
dense solution (expected at the solver's tolerance), the comparison of the stored site and applied arrays with the dense
solution at the sites (the root displacement of the kernel-averaged fields, with the kernel's own leading-order
prediction), a per-record cache of the dense results (reruns with other thresholds do not re-solve), forked workers.

CPU only.  The solves run in --workers forked processes (default 1); BLAS/OpenMP threads per process =
--threads // --workers, at most 8 threads in total.  No plotting.  Every threshold is a named argument.  Importable:
solve_record_dense(...), analyse(...), build_fine_grid(...) are used by firstlaw_selftest_v2.py.
"""
import os
import sys


def _preparse(argv):
    def get(name, default, cast):
        val = default
        for i, a in enumerate(argv):
            if a == name and i + 1 < len(argv):
                try:
                    val = cast(argv[i + 1])
                except ValueError:
                    pass
            elif a.startswith(name + '='):
                try:
                    val = cast(a.split('=', 1)[1])
                except ValueError:
                    pass
        return val
    threads = max(1, min(get('--threads', 8, int), 8))
    workers = max(1, min(get('--workers', 1, int), 8))
    per = max(1, threads // workers)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(per)
    return threads, workers, per


THREADS, WORKERS, THREADS_PER_PROCESS = _preparse(sys.argv[1:])

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

FOUR_PI = 4.0 * np.pi
NAN = float('nan')
_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
ARRAY_KEYS = ('N', 'v', 'rho', 'J', 'rho_p', 'J_p', 'N_sites', 'v_sites', 'N_kernel', 'v_kernel')


# ----------------------------------------------------------------------------
# small helpers (version 1's)
# ----------------------------------------------------------------------------
def _float(x):
    try:
        if x is None:
            return NAN
        return float(x)
    except (TypeError, ValueError):
        return NAN


def _tors(rec, rs):
    t = rec.get('T_over_rs')
    if t is None:
        t = rec.get('T')
        return NAN if t is None else float(t) / rs
    return float(t)


def _fmt(x, p=6):
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(xf):
        return 'nan'
    return ('%.' + str(p) + 'g') % xf


def _nan(func, arr):
    arr = np.asarray(arr, float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return NAN
    return float(func(arr[np.isfinite(arr)]))


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if not str(k).startswith('_')}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if o is None or isinstance(o, (str, int)):
        return o
    return '<%s>' % type(o).__name__


def uniform_step(R, rtol):
    d = np.diff(R)
    if d.size == 0:
        return None
    h = float(np.median(d))
    if h <= 0:
        return None
    if np.all(np.abs(d - h) <= rtol * abs(h)):
        return h
    return None


def derivative(y, R, h):
    """d/dr on a grid: 4th-order centred stencil (2nd order at the edges) for a uniform grid, np.gradient otherwise."""
    y = np.asarray(y, dtype=float)
    if h is None or y.size < 5:
        return np.gradient(y, R, edge_order=2 if y.size >= 3 else 1)
    d = np.empty_like(y)
    d[2:-2] = (-y[4:] + 8.0 * y[3:-1] - 8.0 * y[1:-3] + y[:-4]) / (12.0 * h)
    d[0] = (-3.0 * y[0] + 4.0 * y[1] - y[2]) / (2.0 * h)
    d[1] = (y[2] - y[0]) / (2.0 * h)
    d[-2] = (y[-1] - y[-3]) / (2.0 * h)
    d[-1] = (3.0 * y[-1] - 4.0 * y[-2] + y[-3]) / (2.0 * h)
    return d


def local_poly_fit(R, y, x0, degree):
    n = R.size
    npts = degree + 1
    if npts > n:
        npts = n
        degree = n - 1
    i = int(np.searchsorted(R, x0, side='left'))
    lo = i - npts // 2
    if npts % 2 == 1 and 0 < i < n:
        if (x0 - R[i - 1]) < (R[i] - x0):
            lo -= 1
    lo = max(0, min(n - npts, lo))
    x = np.asarray(R[lo:lo + npts], float)
    yy = np.asarray(y[lo:lo + npts], float)
    xc = 0.5 * (x[0] + x[-1])
    sc = max(float(x[-1] - x[0]), 1e-300)
    t = (x - xc) / sc
    coef = np.polyfit(t, yy, degree)
    return coef, xc, sc


def local_poly_eval(coef, xc, sc, x0):
    t0 = (x0 - xc) / sc
    val = float(np.polyval(coef, t0))
    der = float(np.polyval(np.polyder(coef), t0)) / sc
    return val, der


def interp_at(R, y, x0, degree):
    coef, xc, sc = local_poly_fit(R, y, x0, degree)
    return local_poly_eval(coef, xc, sc, x0)


def refine_zero(R, f, i_lo, degree, maxit=80):
    a, b = float(R[i_lo]), float(R[i_lo + 1])
    coef, xc, sc = local_poly_fit(R, f, 0.5 * (a + b), degree)
    fa, _ = local_poly_eval(coef, xc, sc, a)
    fb, _ = local_poly_eval(coef, xc, sc, b)
    if fa == 0.0:
        _, sl = local_poly_eval(coef, xc, sc, a)
        return a, sl, True
    if fb == 0.0:
        _, sl = local_poly_eval(coef, xc, sc, b)
        return b, sl, True
    if (fa > 0) == (fb > 0):
        root = a + (b - a) * float(f[i_lo]) / (float(f[i_lo]) - float(f[i_lo + 1]))
        return root, (float(f[i_lo + 1]) - float(f[i_lo])) / (b - a), False
    lo, hi, flo = a, b, fa
    for _ in range(maxit):
        m = 0.5 * (lo + hi)
        fm, _ = local_poly_eval(coef, xc, sc, m)
        if fm == 0.0:
            lo = hi = m
            break
        if (fm > 0) == (flo > 0):
            lo, flo = m, fm
        else:
            hi = m
        if hi - lo <= 1e-14 * max(1.0, abs(m)):
            break
    root = 0.5 * (lo + hi)
    for _ in range(3):
        fr, dfr = local_poly_eval(coef, xc, sc, root)
        if dfr != 0.0:
            step = fr / dfr
            if abs(step) < (b - a):
                root -= step
    root = min(max(root, a), b)
    _, slope = local_poly_eval(coef, xc, sc, root)
    return root, slope, True


def classify_zeros(R, f, r_min, degree):
    s = np.sign(np.asarray(f, float))
    s = np.where(s == 0, 1.0, s)
    idx = np.where((s[:-1] * s[1:] < 0) & (R[1:] > r_min))[0]
    zeros = []
    for i in idx:
        r0, sl, ok = refine_zero(R, f, int(i), degree)
        zeros.append({'r': float(r0), 'slope': float(sl), 'type': 'outer' if sl > 0 else 'inner',
                      'i_lo': int(i), 'bracket_ok': bool(ok)})
    return zeros


def near_tangencies(R, f, tol, r_min):
    f = np.asarray(f, float)
    out = []
    if f.size < 3:
        return out
    fm, f0, fp = f[:-2], f[1:-1], f[2:]
    is_min = (f0 < fm) & (f0 <= fp) & (f0 > 0) & (f0 < tol)
    is_max = (f0 > fm) & (f0 >= fp) & (f0 < 0) & (-f0 < tol)
    idx = np.where((is_min | is_max) & (R[1:-1] > r_min))[0] + 1
    for i in idx:
        out.append({'r': float(R[i]), 'f': float(f[i])})
    return out


def simpson_uniform(y, h):
    y = np.asarray(y, float)
    n = y.size
    if n < 2:
        return 0.0
    if n == 2:
        return 0.5 * h * float(y[0] + y[1])
    if n % 2 == 1:
        return float(h / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum()))
    s = simpson_uniform(y[:-1], h)
    s += h / 12.0 * float(-y[-3] + 8.0 * y[-2] + 5.0 * y[-1])
    return float(s)


def integral_from(R, f, r_lo, h, degree_lo=3):
    """int_{r_lo}^{R[-1]} f dr with f on the grid: the partial cell [r_lo, next point] by the trapezoid with f(r_lo) from
    the local polynomial; the rest by 3 rules ('trapz', 'simpson' on a uniform grid, 'coarse' = trapezoid on every other
    point).  Returns (dict, i0)."""
    R = np.asarray(R, float)
    f = np.asarray(f, float)
    n = R.size
    i0 = int(np.searchsorted(R, r_lo, side='left'))
    rules = {'trapz': 0.0, 'simpson': 0.0, 'coarse': 0.0}
    if i0 >= n:
        return rules, i0
    part = 0.0
    if R[i0] > r_lo and i0 > 0:
        f_lo, _ = interp_at(R, f, r_lo, degree_lo)
        part = 0.5 * (f_lo + float(f[i0])) * (float(R[i0]) - r_lo)
    x = R[i0:]
    y = f[i0:]
    rules['trapz'] = part + (float(_trapz(y, x)) if x.size >= 2 else 0.0)
    rules['simpson'] = part + simpson_uniform(y, h) if h is not None else NAN
    xc = x[::2]
    yc = y[::2]
    coarse = float(_trapz(yc, xc)) if xc.size >= 2 else 0.0
    if x.size >= 2 and (x.size - 1) % 2 == 1:
        coarse += 0.5 * (float(y[-2]) + float(y[-1])) * (float(x[-1]) - float(x[-2]))
    rules['coarse'] = part + coarse
    return rules, i0


def row_defects(R, N, v, rho, J, G, h):
    """The two rows' defects on arrays with grid derivatives:
       e = M' - 4 pi r^2 (rho + u J),  g = N'/N + 4 pi G r J / u,  u = v/N, M = r u^2 / 2G."""
    with np.errstate(divide='ignore', invalid='ignore'):
        u = v / N
        M = R * u * u / (2.0 * G)
        Mp = derivative(M, R, h)
        Np = derivative(N, R, h)
        e = Mp - FOUR_PI * R * R * (rho + u * J)
        g = Np / N + FOUR_PI * G * R * J / u
    g = np.where(np.isfinite(g), g, np.nan)
    return e, g, M, Mp, Np


def kappa_formula(N_R, R, rho_R, J_R, G):
    return N_R / (2.0 * R) * (1.0 - 8.0 * np.pi * G * R * R * (rho_R - J_R))

# ============================================================================
# version 3 (the final plan's item 6): T_ll = rho + p_r - 2J at the roots and the row defect at every dense root.
# The constants and functions from here to the next dashed banner, and the v3_* calls that build_firstlaw_records_v3.py splices
# into the version-2 functions, are the only additions; no version-2 computation is changed.
# ============================================================================
V3_REC_NAN_KEYS = ('e_root', 'g_root', 'e_root_rel', 'g_root_rel', 'den_e_root', 'den_g_root', 'Np_over_N_root', 'R_in', 'e_root_in',
                   'g_root_in', 'e_root_in_rel', 'g_root_in_rel', 'kappa_corr_root', 'e_root_from_kappa', 'e_root_minus_from_kappa', 'p_R',
                   'p_R_in', 'Tll_R', 'Tll_flipJ_R', 'rho_minus_2J_R', 'rho_plus_2J_R', 'p_over_rho_R', 'Tll_dimless_R', 'Tll_R_in',
                   'Tll_window_min', 'Tll_window_max', 'Tll_window_n_zero_crossings', 'Tll_window_nearest_zero',
                   'Tll_window_nearest_zero_minus_R', 'Tll_grid_min', 'Tll_grid_max', 'sum_P_tot', 'sum_abs_P_tot', 'sum_P_over_sum_eps',
                   'p_quadrature_rel', 'p_kernel_check', 'p_max_abs_over_rho_max_abs', 'stress_seconds', 'Tll_R_over_interp_err',
                   'Tll_nodes_bracket_same_sign', 'flux_F_R', 'flux_F_rc', 'flux_F_R_minus_rc', 'r_collar_dense', 'p_interp_err_half_rel', 'p_interp_err_rel',
                   'p_interp_err_half_root_rel', 'p_interp_err_root_rel', 'p_interp_err_root_abs', 'p_R_method_diff_rel', 'p_extrap_cells', 'companion_row',
                   'companion_T', 'handoff_eps_vs_rho_rel', 'handoff_eps_expected_vs_rho_rel', 'handoff_eps_vs_eps_tot_rel', 'handoff_N2avg_vs_dense_rel',
                   'pt_R', 'pt_over_rho_R', 'pt_over_p_R', 'r_node_below_R', 'r_node_above_R', 'Tll_node_below_R', 'Tll_node_above_R', 'p_R_over_J_R',
                   'p_max_abs_over_J_max_abs', 'p_node_shell_volume_at_R')
V3_THRESHOLD_KEYS = ('pr_key', 'pr_reading', 'pr_scale', 'Tll_window_ell', 'Tll_offsets_ell', 'seed_cache', 'seed_key_policy', 'no_stress', 'pt_key', 'pr_node_reading', 'pr_grid_key',
                     'check_against')
V3_STRESS_NORMS_R = ('pr', 'prr', 'prtot', 'prrtot', 'sigmar', 'sigmarr', 'sigmartot', 'sigmarrtot', 'sr', 'srr', 'srtot', 'srrtot', 'trr',
                     'trrtot', 'stressr', 'stressrr', 'stressrtot', 'stressrrtot', 'pressr', 'pressurer', 'pressrtot', 'pressurertot',
                     'prad', 'pradtot', 'pradial', 'pradialtot')
V3_STRESS_NORMS_T = ('pt', 'ptt', 'pttot', 'ptttot', 'sigmat', 'sigmatt', 'sigmattot', 'sigmatttot', 'stt', 'sttot', 'stttot', 'ttt',
                     'ttttot', 'stresst', 'stresstt', 'stressttot', 'stresstttot', 'ptan', 'ptantot', 'pperp', 'pperptot', 'ptangential',
                     'ptangentialtot')
V3_COMPARE_SECTIONS = ('records', 'pairs', 'excluded_pairs', 'stats', 'grid', 'matching', 'G', 'rs', 'ell', 'stop_E', 'tag', 'record_list_key',
                       'md_every')
V3_COMPARE_EXEMPT = ('solve_seconds', 'seconds', 'solve_seconds_total', 'solve_seconds_median')

# ============================================================================
# version 3b: the radial stress from a companion npz (the engine's handoff), where it is already the engine's kernel average on the
# companion's own node grid; its normalization against rho_ell and J_ell; the node-to-dense-grid interpolation and its error; the
# outward-flux reading of T_ll; the bitwise check against version 2 matched by T.  The constants and functions from here to _v3_norm,
# and the v3b_* calls that build_firstlaw_records_v3b.py splices into the version-3 functions, are the only additions; no version-2
# computation is changed.
# ============================================================================
V3B_STRESS_RECORD_KEYS = ('p_interp_err_half_rel', 'p_interp_err_rel', 'p_interp_err_half_root_rel', 'p_interp_err_root_rel', 'p_interp_err_root_abs',
                          'p_R_method_diff_rel', 'p_extrap_cells', 'companion_row', 'companion_T', 'handoff_eps_combo', 'handoff_eps_vs_rho_rel',
                          'handoff_eps_expected_vs_rho_rel', 'handoff_eps_vs_eps_tot_rel', 'handoff_N2avg_vs_dense_rel', 'pt_R', 'pt_over_rho_R',
                          'pt_over_p_R', 'r_node_below_R', 'r_node_above_R', 'Tll_node_below_R', 'Tll_node_above_R', 'p_R_over_J_R',
                          'p_max_abs_over_J_max_abs', 'p_node_shell_volume_at_R')
V3B_STRESS_STRING_KEYS = ('handoff_eps_combo',)
V3B_REC_STAT_KEYS = ('p_interp_err_half_rel', 'p_interp_err_rel', 'p_interp_err_half_root_rel', 'p_interp_err_root_rel', 'p_R_method_diff_rel',
                     'p_extrap_cells', 'handoff_eps_vs_rho_rel', 'handoff_eps_expected_vs_rho_rel', 'handoff_eps_vs_eps_tot_rel',
                     'handoff_N2avg_vs_dense_rel', 'pt_R', 'pt_over_rho_R', 'pt_over_p_R', 'Tll_node_below_R', 'Tll_node_above_R',
                     'Tll_R_over_interp_err', 'p_R_over_J_R', 'p_max_abs_over_J_max_abs', 'p_node_shell_volume_at_R', 'flux_F_R', 'flux_F_rc',
                     'flux_F_R_minus_rc')
V3B_EPS_COMBOS = (('eps_dust + eps_sea - eps_R2', (('eps_dust', 1.0), ('eps_sea', 1.0), ('eps_R2', -1.0))),
                  ('eps_dust + eps_sea', (('eps_dust', 1.0), ('eps_sea', 1.0))),
                  ('eps_sea - eps_R2', (('eps_sea', 1.0), ('eps_R2', -1.0))),
                  ('eps_dust', (('eps_dust', 1.0),)),
                  ('eps_sea', (('eps_sea', 1.0),)),
                  ('eps_tot', (('eps_tot', 1.0),)),
                  ('rho_ell', (('rho_ell', 1.0),)))
V3B_AUX_KEYS = ('eps_dust', 'eps_sea', 'eps_R2', 'eps_tot', 'rho_ell', 'N2avg')
V3B_INTERP_ORDER = 4


def _v3b_stat(fn, vals):
    """fn over the finite values of vals; NaN when there is none."""
    v = np.array([_float(x) for x in vals], float)
    v = v[np.isfinite(v)]
    return float(fn(v)) if v.size else NAN


def _v3b_fmt(x, p=4):
    v = _float(x)
    return ('%%.%dg' % int(p)) % v if np.isfinite(v) else 'nan'


def _v3b_norm_base(key):
    """_v3_norm with a trailing 'ell' removed (S_r_ell -> 'sr', S_t_ell -> 'st'): the companion's averaged fields carry the kernel scale in their name."""
    n = _v3_norm(key)
    if n.endswith('ell') and len(n) > 3:
        n = n[:-3]
    return n


def _v3b_is_radial(key):
    return _v3b_norm_base(key) in V3_STRESS_NORMS_R and 'pair' not in str(key).lower()


def _v3b_is_tangential(key):
    b = _v3b_norm_base(key)
    return (b in V3_STRESS_NORMS_T or b == 'st') and 'pair' not in str(key).lower()


def _v3b_npz_keys(path):
    try:
        with np.load(path, allow_pickle=False) as z:
            return [str(k) for k in z.files]
    except Exception:
        return []


def v3b_companion_path(F, opts):
    """The companion npz beside the fields npz: collapse_<tag>_handoff.npz for collapse_<tag>_fields.npz (the suffix _fields.npz replaced), else
    collapse_<tag>_handoff.npz with this run's tag in the fields npz directory; None when neither exists."""
    fp = str((F or {}).get('path') or '')
    if not fp:
        return None
    d, b = os.path.dirname(os.path.abspath(fp)), os.path.basename(fp)
    cands = []
    if b.endswith('_fields.npz'):
        cands.append(os.path.join(d, b[:-len('_fields.npz')] + '_handoff.npz'))
    tag = getattr(opts, 'tag', None)
    if tag:
        cands.append(os.path.join(d, 'collapse_%s_handoff.npz' % tag))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def _v3b_parse_spec(spec, F, opts):
    """One --pr-key or --pt-key value.  'none' -> (None, None, 'disabled'); 'handoff:<key>' -> (the companion npz beside the fields npz, key);
    '<path>.npz:<key>' -> (that npz, key); a bare key -> (None, key) when it is among the fields keys (the version-3 reading), else (the companion
    npz, key) when it is there.  The third element is the note."""
    s = str(spec)
    if s.lower() == 'none':
        return None, None, 'disabled'
    if ':' in s:
        left, key = s.rsplit(':', 1)
        if left.lower() == 'handoff':
            path = v3b_companion_path(F, opts)
            if path is None:
                return None, None, "no companion npz (collapse_<tag>_handoff.npz) beside the fields npz for '%s'" % s
        else:
            path = os.path.abspath(left)
            if not os.path.exists(path):
                return None, None, "the npz %s named by '%s' does not exist" % (path, s)
        keys = _v3b_npz_keys(path)
        if key not in keys:
            return None, None, "'%s' is not among the keys %s of %s" % (key, keys, path)
        return path, key, 'named by the option'
    keys = [str(k) for k in ((F or {}).get('keys') or [])]
    if s in keys:
        return None, s, 'named by the option (a fields key)'
    path = v3b_companion_path(F, opts)
    if path is not None and s in _v3b_npz_keys(path):
        return path, s, 'named by the option (found in the companion npz, not among the fields keys)'
    return None, None, "'%s' is neither among the fields keys %s nor in the companion npz %s" % (s, keys, path)


def v3b_locate_stress(F, opts):
    """Where the radial stress is read from: a dict with 'companion' (True when it is in a companion npz on its own node grid), 'path', 'key',
    'note', 'candidates', and the tangential stress 'pt_key', 'pt_path', 'pt_note'.  Order: --pr-key in the forms of _v3b_parse_spec; else the
    version-3 discovery among the fields keys; else the companion npz beside the fields npz, searched by the same names with a trailing _ell allowed
    (S_r_ell).  The tangential stress: --pt-key, else found the same way in the companion when the radial one is there; it is reported, not used."""
    keys = [str(k) for k in ((F or {}).get('keys') or [])]
    out = {'companion': False, 'path': None, 'key': None, 'note': None, 'candidates': [], 'pt_key': None, 'pt_path': None, 'pt_note': None}
    explicit = getattr(opts, 'pr_key', None)
    if explicit is not None:
        path, key, note = _v3b_parse_spec(explicit, F, opts)
        if key is None:
            out['note'] = 'disabled by --pr-key none' if note == 'disabled' else note
        else:
            out.update(path=path, key=key, note=note, companion=path is not None)
            if path is None:
                _k, _n, cands, pt_key = find_stress_key(keys, None)
                out.update(candidates=cands, pt_key=pt_key)
    else:
        key, note, cands, pt_key = find_stress_key(keys, None)
        out.update(key=key, note=note, candidates=cands, pt_key=pt_key)
        if key is None:
            path = v3b_companion_path(F, opts)
            if path is not None:
                ck = _v3b_npz_keys(path)
                rc = [k for k in ck if _v3b_is_radial(k)]
                rc.sort(key=lambda k: (0 if _v3_norm(k).endswith('tot') else 1, ck.index(k)))
                out['candidates'] = rc
                if rc:
                    out.update(path=path, key=rc[0], companion=True,
                               note='found by name in the companion npz %s (no radial stress among the fields keys; other candidates: %s)' % (path, rc[1:]))
                else:
                    out['note'] = '%s; the companion npz %s has no radial stress among its keys %s either' % (note, path, ck)
            else:
                out['note'] = '%s; no companion npz beside the fields npz' % note
    pt_spec = getattr(opts, 'pt_key', None)
    if pt_spec is not None:
        ppath, pkey, pnote = _v3b_parse_spec(pt_spec, F, opts)
        out.update(pt_key=pkey, pt_path=ppath, pt_note=pnote)
    elif out['companion'] and out['path']:
        ck = _v3b_npz_keys(out['path'])
        tc = [k for k in ck if _v3b_is_tangential(k)]
        if tc:
            out.update(pt_key=tc[0], pt_path=out['path'], pt_note='found by name in the companion npz')
    return out


def _v3b_fields_T(F):
    """The fields npz T per row (the records were matched to the rows by it)."""
    t = (F or {}).get('T')
    if t is None:
        try:
            with np.load(F['path'], allow_pickle=False) as z:
                t = z['T'] if 'T' in z.files else None
        except Exception:
            t = None
    return None if t is None else np.asarray(t, float).ravel()


def _v3b_node_grid(z, n_nodes, preferred):
    """The node grid of the companion arrays with n_nodes columns: the key preferred (--pr-grid-key) when it is 1-D of that length (or 2-D with every
    row equal), else r_nodes, r, R, else any key of that form.  Returns (key, increasing grid, the column order that sorts it) or (None, None, None)."""
    names = []
    for k in (preferred, 'r_nodes', 'r', 'R'):
        if k and k in z.files and str(k) not in names:
            names.append(str(k))
    names += [str(k) for k in z.files if str(k) not in names]
    for k in names:
        try:
            a = np.asarray(z[k], float)
        except Exception:
            continue
        if a.ndim == 2 and a.shape[-1] == n_nodes and a.shape[0] >= 1 and np.all(np.isfinite(a)) and np.all(a == a[0]):
            a = a[0]
        if a.ndim == 1 and a.size == n_nodes and np.all(np.isfinite(a)) and np.unique(a).size == a.size:
            order = np.argsort(a)
            return k, a[order], order
    return None, None, None


def v3b_spline(x, y):
    """The not-a-knot cubic spline through (x, y) when scipy is importable, else the local polynomial of degree 3 point by point; a callable that is NaN
    outside [x[0], x[-1]]."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    try:
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(x, y, bc_type='not-a-knot', extrapolate=False)

        def f(q):
            return np.asarray(cs(np.asarray(q, float)), float)
        return f
    except Exception:
        def f(q):
            q = np.atleast_1d(np.asarray(q, float))
            return np.array([_interp_finite(x, y, float(v), 3) if x[0] <= v <= x[-1] else NAN for v in q], float)
        return f


def v3b_interp_error(xs, y, R0, w):
    """The stride-2 test of the node-to-grid interpolation: the spline through every other node, evaluated at the skipped nodes, against the stored
    values there; the maximum |.| relative to max |y| over all skipped nodes (half_rel) and over those within w of R0 (half_root_rel); the estimate
    for the full node density divides by 2^V3B_INTERP_ORDER (fourth order for the cubic spline); root_abs is that estimate in the units of y."""
    out = {'p_interp_err_half_rel': NAN, 'p_interp_err_rel': NAN, 'p_interp_err_half_root_rel': NAN, 'p_interp_err_root_rel': NAN, 'p_interp_err_root_abs': NAN}
    xs = np.asarray(xs, float)
    y = np.asarray(y, float)
    scale = float(np.max(np.abs(y))) if y.size else 0.0
    if xs.size < 8 or scale <= 0.0:
        return out
    f2 = v3b_spline(xs[::2], y[::2])
    xo, yo = xs[1::2], y[1::2]
    pred = f2(xo)
    ok = np.isfinite(pred)
    if not ok.any():
        return out
    err = np.abs(np.where(ok, pred, 0.0) - yo)
    fac = float(2 ** V3B_INTERP_ORDER)
    out['p_interp_err_half_rel'] = float(np.max(err[ok])) / scale
    out['p_interp_err_rel'] = out['p_interp_err_half_rel'] / fac
    if np.isfinite(R0):
        m = ok & (np.abs(xo - R0) <= w)
        if m.any():
            out['p_interp_err_half_root_rel'] = float(np.max(err[m])) / scale
            out['p_interp_err_root_rel'] = out['p_interp_err_half_root_rel'] / fac
            out['p_interp_err_root_abs'] = out['p_interp_err_root_rel'] * scale
    return out


def _v3b_rel_rms(a, b):
    """RMS of a - b over the points where both are finite, relative to the RMS of b there."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return NAN
    den = float(np.sqrt(np.mean(b[m] ** 2)))
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) / den if den > 0 else NAN


def v3b_companion_record(xs, y, yt, d, r_fine, deg, w, h_node, aux_row, eps_tot_site, N_site):
    """One record's radial stress from the companion nodes: y on xs (already the kernel average per unit volume, scaled and read as --pr-node-reading says),
    carried to the dense grid and to the dense zeros by the cubic spline (end values held outside the node range); the stride-2 interpolation error; the
    local polynomial of degree deg at the outer root as the method difference; T_ll at the two nodes bracketing the outer root from the stored node value
    and the dense rho, J interpolated there (no interpolation of the stress); the tangential stress yt at the root when given; the normalization checks:
    the companion's energy densities on the nodes (aux_row: name -> row array) against the dense rho_ell at the nodes (or, per site, against eps_tot),
    N2avg against the dense N^2, and the ratios of the stress to rho_ell and J_ell.  The dict has the keys smooth_stress_record's result has, so that
    v3_record_columns reads it unchanged, plus V3B_STRESS_RECORD_KEYS."""
    t0 = time.time()
    a = d.get('arrays') or {}
    r_fine = np.asarray(r_fine, float)
    rho = np.asarray(a.get('rho'), float)
    J = np.asarray(a.get('J'), float)
    N = np.asarray(a.get('N'), float) if a.get('N') is not None else None
    zeros = [float(zz) for zz in (d.get('zeros_dense') or [])]
    out = {'ok': False, 'error': None, 'seconds': NAN, 'p': None, 'p_at_zeros': None, 'p_at_zeros_source': 'cubic spline of the companion nodes at the root',
           'sum_P': NAN, 'sum_abs_P': NAN, 'quadrature': NAN, 'quadrature_rel': NAN, 'kernel_check': NAN,
           'kernel': 'none: the companion array is already the kernel average; cubic spline from the nodes'}
    xs = np.asarray(xs, float)
    y = np.asarray(y, float)
    if y.shape != xs.shape or not np.all(np.isfinite(y)):
        out['error'] = 'the companion row has shape %s against the node grid %s, or non-finite values' % (list(y.shape), list(xs.shape))
        out['seconds'] = time.time() - t0
        return out
    f = v3b_spline(xs, y)
    p = f(r_fine)
    inside = (r_fine >= xs[0]) & (r_fine <= xs[-1])
    p = np.where(inside, p, np.nan)
    p[r_fine < xs[0]] = y[0]
    p[r_fine > xs[-1]] = y[-1]
    bad = ~np.isfinite(p)
    if bad.any():
        p[bad] = np.interp(r_fine[bad], xs, y)
    out['p'] = p
    out['p_extrap_cells'] = int(np.sum(~inside))
    pz = [float(f(np.array([zz]))[0]) if xs[0] <= zz <= xs[-1] else NAN for zz in zeros]
    out['p_at_zeros'] = pz
    scale = float(np.max(np.abs(y)))
    R0 = zeros[-1] if zeros else NAN
    out.update(v3b_interp_error(xs, y, R0, w))
    if zeros and np.isfinite(pz[-1]) and scale > 0:
        loc = _interp_finite(xs, y, R0, deg)
        out['p_R_method_diff_rel'] = abs(pz[-1] - loc) / scale if np.isfinite(loc) else NAN
    out['sum_P'] = float(_trapz(FOUR_PI * r_fine * r_fine * p, r_fine))
    out['sum_abs_P'] = float(_trapz(FOUR_PI * r_fine * r_fine * np.abs(p), r_fine))
    mJ = float(np.max(np.abs(J))) if J.size else 0.0
    out['p_max_abs_over_J_max_abs'] = scale / mJ if mJ > 0 else NAN
    if zeros and xs[0] < R0 < xs[-1]:
        i = int(np.searchsorted(xs, R0))
        for name, node in (('below', i - 1), ('above', i)):
            r_n = float(xs[node])
            rho_n = _interp_finite(r_fine, rho, r_n, deg)
            J_n = _interp_finite(r_fine, J, r_n, deg)
            out['r_node_%s_R' % name] = r_n
            out['Tll_node_%s_R' % name] = rho_n + float(y[node]) - 2.0 * J_n
        J_R = _interp_finite(r_fine, J, R0, deg)
        out['p_R_over_J_R'] = pz[-1] / J_R if (np.isfinite(J_R) and J_R != 0.0 and np.isfinite(pz[-1])) else NAN
        out['p_node_shell_volume_at_R'] = FOUR_PI * R0 * R0 * float(h_node) if np.isfinite(float(h_node)) else NAN
    if yt is not None:
        yt = np.asarray(yt, float)
        if yt.shape == xs.shape and np.all(np.isfinite(yt)) and zeros and xs[0] <= R0 <= xs[-1]:
            pt_R = float(v3b_spline(xs, yt)(np.array([R0]))[0])
            rho_R = _interp_finite(r_fine, rho, R0, deg)
            out['pt_R'] = pt_R
            out['pt_over_rho_R'] = pt_R / rho_R if (np.isfinite(rho_R) and rho_R != 0.0) else NAN
            out['pt_over_p_R'] = pt_R / pz[-1] if (np.isfinite(pz[-1]) and pz[-1] != 0.0) else NAN
    aux_row = aux_row or {}
    node_shaped = {k: np.asarray(v, float) for k, v in aux_row.items() if v is not None and np.asarray(v).shape == xs.shape}
    site_shaped = {}
    if eps_tot_site is not None:
        es = np.asarray(eps_tot_site, float)
        site_shaped = {k: np.asarray(v, float) for k, v in aux_row.items()
                       if v is not None and np.asarray(v).shape == es.shape and np.asarray(v).shape != xs.shape}
    if node_shaped:
        rho_n = v3b_spline(r_fine, rho)(xs)
        best = (NAN, None)
        for name, terms in V3B_EPS_COMBOS:
            if all(t in node_shaped for t, _c in terms):
                comb = sum(c * node_shaped[t] for t, c in terms)
                rel = _v3b_rel_rms(comb, rho_n)
                if name == V3B_EPS_COMBOS[0][0]:
                    out['handoff_eps_expected_vs_rho_rel'] = rel
                if np.isfinite(rel) and (not np.isfinite(best[0]) or rel < best[0]):
                    best = (rel, name)
        if best[1] is not None:
            out['handoff_eps_combo'] = best[1]
            out['handoff_eps_vs_rho_rel'] = best[0]
        if 'N2avg' in node_shaped and N is not None:
            out['handoff_N2avg_vs_dense_rel'] = _v3b_rel_rms(node_shaped['N2avg'], v3b_spline(r_fine, N)(xs) ** 2)
    elif site_shaped:
        es = np.asarray(eps_tot_site, float)
        best = (NAN, None)
        for name, terms in V3B_EPS_COMBOS:
            if all(t in site_shaped for t, _c in terms):
                comb = sum(c * site_shaped[t] for t, c in terms)
                rel = _v3b_rel_rms(comb, es)
                if np.isfinite(rel) and (not np.isfinite(best[0]) or rel < best[0]):
                    best = (rel, name)
        if best[1] is not None:
            out['handoff_eps_combo'] = best[1] + ' (per site)'
            out['handoff_eps_vs_eps_tot_rel'] = best[0]
        if 'N2avg' in site_shaped and N_site is not None:
            out['handoff_N2avg_vs_dense_rel'] = _v3b_rel_rms(site_shaped['N2avg'], np.asarray(N_site, float) ** 2)
    out['ok'] = True
    out['seconds'] = time.time() - t0
    return out


def v3b_stress_module_note(FLD):
    """Provenance only: the lines of the engine's stress modules (engine_stress_v2.py, then engine_stress_v1.py, beside the engine module) that mention
    S_r, the radial stress, sigma_rr or the pressure; the first few are printed under STRESS so that the sign and normalization of S_r_ell are read from
    the engine's own text.  The files are read as text, not imported."""
    out = []
    try:
        base = getattr(FLD, '__file__', None) or __file__
        code_dir = os.path.dirname(os.path.abspath(base))
    except Exception:
        return out
    for name in ('engine_stress_v2', 'engine_stress_v1'):
        path = os.path.join(code_dir, name + '.py')
        if not os.path.exists(path):
            continue
        try:
            with open(path) as fh:
                text = fh.read()
        except Exception:
            continue
        hits = []
        for ln in text.splitlines():
            s = ln.strip()
            low = s.lower()
            if ('s_r' in low or 'radial stress' in low or 'sigma_rr' in low or 'pressure' in low) and 0 < len(s) <= 220:
                hits.append(s)
        out.append('%s: %d lines mention S_r, the radial stress, sigma_rr or the pressure; the first: %s' % (name, len(hits), ' | '.join(hits[:5]) if hits else 'none'))
    return out


def v3b_stress_pass_companion(FLD, F, opts, dense, rows, idx, r_fine, ell_pass, notes, meta, loc, t0):
    """The stress pass when the radial stress is in a companion npz on its own node grid (the engine's handoff: S_r_ell on r_nodes): no kernel call;
    the companion rows are matched to the records by T (the companion's T against the fields npz T of the record's row; positional when the companion
    has no T); every record with dense arrays gets v3b_companion_record.  Returns ({'meta': ..., 'records': {k: ...}}, seconds)."""
    empty = {'meta': meta, 'records': {}}
    path, key = loc['path'], loc['key']
    reading = str(getattr(opts, 'pr_node_reading', 'density'))
    scale = float(getattr(opts, 'pr_scale', 1.0))
    meta.update(companion=True, path=path, key=key, note=loc.get('note'), candidates=list(loc.get('candidates') or []), pt_key=loc.get('pt_key'),
                pt_path=loc.get('pt_path'), pt_note=loc.get('pt_note'), scale=scale, workers=1, ell_pass=float(ell_pass),
                reading=('node density (per unit volume, as stored)' if reading == 'density' else 'per node: divided by the node shell volume 4 pi r^2 h_node'),
                kernel=("none: '%s' is already the engine's kernel average on the companion's nodes (the _ell of its name); not-a-knot cubic spline from the "
                        "nodes to the dense grid and to the zeros, end values held outside the node range" % key),
                module_note=v3b_stress_module_note(FLD))
    try:
        z = np.load(path, allow_pickle=False)
        P = np.asarray(z[key], float)
        if P.ndim == 1:
            P = P[None, :]
        gkey, xs, order = _v3b_node_grid(z, P.shape[-1], getattr(opts, 'pr_grid_key', 'r_nodes'))
        if xs is None:
            raise ValueError('no 1-D key of length %d (the node grid of %s) in %s' % (P.shape[-1], key, path))
        P = P[:, order]
        Tc = np.asarray(z['T'], float).ravel() if 'T' in z.files else None
        Pt = None
        if loc.get('pt_key'):
            zt = z if (loc.get('pt_path') in (None, path)) else np.load(loc['pt_path'], allow_pickle=False)
            Pt = np.asarray(zt[loc['pt_key']], float)
            if Pt.ndim == 1:
                Pt = Pt[None, :]
            Pt = Pt[:, order] if Pt.shape[-1] == xs.size else None
        aux = {}
        for name in V3B_AUX_KEYS:
            if name in z.files:
                try:
                    arr = np.asarray(z[name], float)
                    if arr.ndim == 1:
                        arr = arr[None, :]
                    aux[name] = arr[:, order] if arr.shape[-1] == xs.size else arr
                except Exception:
                    pass
        for name in ('ell', 'g', 'm', 'tau'):
            if name in z.files:
                try:
                    meta['companion_' + name] = float(np.asarray(z[name], float).ravel()[0])
                except Exception:
                    pass
    except Exception as exc:
        meta['note'] = 'the companion stress could not be read (%s: %s)' % (type(exc).__name__, exc)
        notes.append('STRESS: ' + meta['note'])
        print('STRESS: ' + meta['note'])
        return empty, 0.0
    for name in ('ell', 'g'):
        try:
            if F.get(name) is not None:
                meta['fields_' + name] = float(np.asarray(F[name], float).ravel()[0])
        except Exception:
            pass
    h_node = uniform_step(xs, 1e-9) or float(np.median(np.diff(xs)))
    meta.update(grid_key=gkey, n_nodes=int(xs.size), h_node=float(h_node), node_range=[float(xs[0]), float(xs[-1])], n_rows=int(P.shape[0]),
                shape=list(P.shape))
    if reading != 'density':
        vol = FOUR_PI * xs * xs * float(h_node)
        with np.errstate(divide='ignore', invalid='ignore'):
            P = np.where(vol[None, :] > 0, P / vol[None, :], 0.0)
            if Pt is not None:
                Pt = np.where(vol[None, :] > 0, Pt / vol[None, :], 0.0)
    P = P * scale
    if Pt is not None:
        Pt = Pt * scale
    Tf = _v3b_fields_T(F)
    cmap = {}
    if Tc is not None and Tc.size == P.shape[0]:
        for j, t in enumerate(Tc):
            cmap.setdefault(round(float(t), 9), int(j))
    by_T = bool(cmap) and Tf is not None
    meta['row_matching'] = 'by T (the companion key T against the fields T of the record row)' if by_T else 'positional (companion row = fields row)'
    eps_site = np.asarray(F['eps_tot'], float) if F.get('eps_tot') is not None else None
    N_site = np.asarray(F['N'], float) if F.get('N') is not None else None
    deg = int(opts.interp_degree)
    w = float(opts.Tll_window_ell) * float(ell_pass)
    results = {}
    n_unmatched = 0
    for k in idx:
        d = dense.get(k)
        if d is None or not d.get('ok') or (d.get('arrays') or {}).get('rho') is None:
            continue
        row = int(rows[k])
        if by_T and row < Tf.size:
            j = cmap.get(round(float(Tf[row]), 9))
        else:
            j = row if row < P.shape[0] else None
        if j is None:
            n_unmatched += 1
            results[k] = {'ok': False, 'p': None, 'seconds': NAN, 'row': row,
                          'error': 'no companion row at the record T %s' % (float(Tf[row]) if (Tf is not None and row < Tf.size) else row)}
            continue
        aux_row = {name: (arr[j] if arr.shape[0] == P.shape[0] else None) for name, arr in aux.items()}
        rec = v3b_companion_record(xs, P[j], None if Pt is None else Pt[j], d, r_fine, deg, w, h_node, aux_row,
                                   None if eps_site is None else eps_site[row], None if N_site is None else N_site[row])
        rec['companion_row'] = int(j)
        rec['companion_T'] = float(Tc[j]) if Tc is not None else NAN
        rec['row'] = row
        results[k] = rec
    oks = [r for r in results.values() if r.get('ok')]
    meta['n_records'] = len(results)
    meta['n_ok'] = len(oks)
    meta['n_unmatched'] = n_unmatched
    meta['seconds'] = time.time() - t0
    errs = [r.get('error') for r in results.values() if not r.get('ok')]
    if errs:
        meta['first_error'] = errs[0]
        notes.append('STRESS: %d of %d records have no companion stress (first: %s)' % (len(errs), len(results), errs[0]))
    meta['kernel_check_max'] = NAN
    meta['quadrature_rel_max_abs'] = NAN
    sp = _v3b_stat(np.sum, [r.get('sum_P') for r in oks])
    se = float(np.sum([float(np.sum(eps_site[r['row']])) for r in oks])) if (oks and eps_site is not None) else NAN
    meta['sum_P_over_sum_eps_all_rows'] = sp / se if (np.isfinite(se) and se != 0.0 and np.isfinite(sp)) else NAN
    for key_ in ('p_interp_err_half_rel', 'p_interp_err_rel', 'p_interp_err_half_root_rel', 'p_interp_err_root_rel', 'p_R_method_diff_rel',
                 'handoff_eps_vs_rho_rel', 'handoff_eps_expected_vs_rho_rel', 'handoff_eps_vs_eps_tot_rel', 'handoff_N2avg_vs_dense_rel',
                 'p_max_abs_over_J_max_abs', 'p_node_shell_volume_at_R'):
        vals = [r.get(key_) for r in oks]
        meta[key_ + '_median'] = _v3b_stat(np.median, vals)
        meta[key_ + '_max'] = _v3b_stat(np.max, vals)
    combos = {}
    for r in oks:
        c = r.get('handoff_eps_combo')
        if c:
            combos[str(c)] = combos.get(str(c), 0) + 1
    meta['eps_combo_counts'] = combos
    print('STRESS: companion %s, key %s (%s); node grid %s: %d nodes, spacing %s, range [%s, %s]; rows %s; reading %s, scale %g; %d records (%d ok, %d '
          'unmatched) in %.1f s; companion ell %s g %s, fields ell %s g %s; eps on the nodes against the dense rho_ell: best %s (rel RMS median %s; expected '
          'combination median %s); N2avg against the dense N^2 median %s; stride-2 interpolation error relative to max|p| median %s max %s (estimate at the '
          'node density median %s; at the root median %s); |spline - local polynomial|(R) median %s; tangential key %s' % (
              path, key, meta.get('note'), gkey, int(xs.size), _v3b_fmt(h_node, 4), _v3b_fmt(xs[0], 6), _v3b_fmt(xs[-1], 6), meta['row_matching'], meta['reading'],
              scale, meta['n_records'], meta['n_ok'], n_unmatched, meta['seconds'], _v3b_fmt(meta.get('companion_ell')), _v3b_fmt(meta.get('companion_g')),
              _v3b_fmt(meta.get('fields_ell')), _v3b_fmt(meta.get('fields_g')), combos, _v3b_fmt(meta.get('handoff_eps_vs_rho_rel_median'), 3),
              _v3b_fmt(meta.get('handoff_eps_expected_vs_rho_rel_median'), 3), _v3b_fmt(meta.get('handoff_N2avg_vs_dense_rel_median'), 3),
              _v3b_fmt(meta.get('p_interp_err_half_rel_median'), 3), _v3b_fmt(meta.get('p_interp_err_half_rel_max'), 3),
              _v3b_fmt(meta.get('p_interp_err_rel_median'), 3), _v3b_fmt(meta.get('p_interp_err_root_rel_median'), 3),
              _v3b_fmt(meta.get('p_R_method_diff_rel_median'), 3), meta.get('pt_key') or 'none'))
    return {'meta': meta, 'records': results}, meta['seconds']


def v3b_root_extras(out, stress, G, R0, arrays, r_fine):
    """After T_ll(R): its size against the stress interpolation error at the root, the sign agreement with T_ll at the two nodes bracketing R (where
    the stress is the stored node value), and the outward energy flux of FIRSTLAW_rows_v1's continuity identity,
    F = 4 pi r^2 [(N^2 + v^2) J + N v (rho + p_r)], at the root (where v = -N gives F(R) = -4 pi R^2 N_R^2 T_ll) and at the collar (the last dense
    point, from the dense arrays and the interpolated p_r)."""
    tll = _float(out.get('Tll_R'))
    st = stress or {}
    err = _float(st.get('p_interp_err_root_abs'))
    out['Tll_R_over_interp_err'] = abs(tll) / err if (np.isfinite(err) and err > 0.0 and np.isfinite(tll)) else NAN
    lo, hi = _float(st.get('Tll_node_below_R')), _float(st.get('Tll_node_above_R'))
    if np.isfinite(lo) and np.isfinite(hi) and np.isfinite(tll) and tll != 0.0:
        out['Tll_nodes_bracket_same_sign'] = 1.0 if (np.sign(lo) == np.sign(tll) and np.sign(hi) == np.sign(tll)) else 0.0
    else:
        out['Tll_nodes_bracket_same_sign'] = NAN
    N_R = _float(out.get('N_R'))
    out['flux_F_R'] = -FOUR_PI * R0 * R0 * N_R * N_R * tll if (np.isfinite(N_R) and np.isfinite(tll)) else NAN
    try:
        N, v, rho, J, p = (np.asarray(arrays[key_], float) for key_ in ('N', 'v', 'rho', 'J', 'p'))
        rc = float(np.asarray(r_fine, float)[-1])
        out['flux_F_rc'] = FOUR_PI * rc * rc * ((N[-1] ** 2 + v[-1] ** 2) * J[-1] + N[-1] * v[-1] * (rho[-1] + p[-1]))
        out['r_collar_dense'] = rc
    except Exception:
        out['flux_F_rc'] = NAN
    out['flux_F_R_minus_rc'] = out['flux_F_R'] - out['flux_F_rc'] if np.all(np.isfinite([out['flux_F_R'], out['flux_F_rc']])) else NAN


def v3b_pair_extras(pr, A, B, aA, aB, Rg, sl, Rm, deg):
    """Per pair: the two records' fluxes F(R), F(r_c), F(R) - F(r_c) and their means; F on the midpoint fields at R_mid (the lower limit of the pair's
    W_ext) from the midpoint arrays N, v, rho, J, p; dT in the records' T/r_s.  The time-tangent balance W_ext/dt - [F(R) - F(r_c)], which the
    continuity identity of FIRSTLAW_rows_v1 equates to the weighted continuity-defect integral 4 pi int r^2 (N W_0 + v W_r) dr (plus the record-step
    and moving-lower-limit terms of this finite-record estimator), is formed in the md and console from these with r_s."""
    for key_ in ('flux_F_R', 'flux_F_rc', 'flux_F_R_minus_rc'):
        pr[key_ + '_a'] = _float(A.get(key_))
        pr[key_ + '_b'] = _float(B.get(key_))
        pr[key_ + '_mean'] = 0.5 * (pr[key_ + '_a'] + pr[key_ + '_b'])
    pr['flux_F_Rmid'] = NAN
    try:
        if all(aA.get(key_) is not None and aB.get(key_) is not None for key_ in ('N', 'v', 'rho', 'J', 'p')):
            mid = {key_: 0.5 * (np.asarray(aA[key_], float)[sl] + np.asarray(aB[key_], float)[sl]) for key_ in ('N', 'v', 'rho', 'J', 'p')}
            vals = {key_: _interp_finite(Rg, mid[key_], Rm, deg) for key_ in mid}
            if all(np.isfinite(x) for x in vals.values()):
                pr['flux_F_Rmid'] = FOUR_PI * Rm * Rm * ((vals['N'] ** 2 + vals['v'] ** 2) * vals['J'] + vals['N'] * vals['v'] * (vals['rho'] + vals['p']))
    except Exception:
        pass
    Ta, Tb = _float(pr.get('T_a')), _float(pr.get('T_b'))
    pr['dT_over_rs'] = Tb - Ta if np.all(np.isfinite([Ta, Tb])) else NAN


def v3b_record_stats(st, rooted_ok):
    """The version-3b record-level statistics (over the rooted re-solved records)."""
    for key_ in V3B_REC_STAT_KEYS:
        vals = [r.get(key_) for r in rooted_ok]
        st[key_ + '_median'] = _v3b_stat(np.median, vals)
        st[key_ + '_min'] = _v3b_stat(np.min, vals)
        st[key_ + '_max'] = _v3b_stat(np.max, vals)
        st[key_ + '_max_abs'] = _v3b_stat(np.max, [abs(_float(x)) for x in vals])
        st[key_ + '_abs_median'] = _v3b_stat(np.median, [abs(_float(x)) for x in vals])
    bs = np.array([_float(r.get('Tll_nodes_bracket_same_sign')) for r in rooted_ok], float)
    fin = bs[np.isfinite(bs)]
    st['n_Tll_nodes_bracket_finite'] = int(fin.size)
    st['frac_Tll_nodes_bracket_same_sign'] = float(np.mean(fin > 0.5)) if fin.size else NAN
    ratio = np.array([_float(r.get('Tll_R_over_interp_err')) for r in rooted_ok], float)
    st['n_Tll_R_over_interp_err_finite'] = int(np.sum(np.isfinite(ratio)))
    st['n_Tll_R_within_interp_err'] = int(np.sum(np.isfinite(ratio) & (ratio <= 1.0)))
    combos = {}
    for r in rooted_ok:
        c = r.get('handoff_eps_combo')
        if c:
            combos[str(c)] = combos.get(str(c), 0) + 1
    st['handoff_eps_combo_counts'] = combos


def v3b_flux_balance(res):
    """From the included pairs: the time tangent W_ext/dt (dt the records' T difference, (T_b - T_a) r_s), against the mean over the two records of
    F(R) - F(r_c), and against F on the midpoint fields at R_mid minus F(r_c); their difference is the finite-record estimate of the weighted continuity
    defect integral 4 pi int r^2 (N W_0 + v W_r) dr of FIRSTLAW_rows_v1 (plus this estimator's record-step and moving-lower-limit terms), in the units
    in which the records' T is the time of the continuity rows.  Also the same summed with the weight dt (energy units, beside the summed W_ext)."""
    pairs = res.get('pairs') or []
    rs = _float(res.get('rs'))
    unit_rs = bool(np.isfinite(rs) and rs > 0)
    rate, fm, fmid, dts = [], [], [], []
    for p in pairs:
        W, dT, fR, fc, fM = (_float(p.get(k)) for k in ('W_ext', 'dT_over_rs', 'flux_F_R_mean', 'flux_F_rc_mean', 'flux_F_Rmid'))
        if not np.all(np.isfinite([W, dT, fR, fc])) or dT == 0.0:
            continue
        dt = dT * rs if unit_rs else dT
        rate.append(W / dt)
        fm.append(fR - fc)
        fmid.append((fM - fc) if np.isfinite(fM) else NAN)
        dts.append(dt)
    out = {'n': len(rate), 'unit': 'per unit T (dt = (T_b - T_a) r_s)' if unit_rs else 'per unit T/r_s (r_s not in the result)'}
    if not rate:
        return out
    rate, fm, fmid, dts = (np.array(x, float) for x in (rate, fm, fmid, dts))
    out['sum_W_rate'] = float(np.sum(rate))
    out['sum_flux_mean'] = float(np.sum(fm))
    out['sum_defect_est'] = float(np.sum(rate - fm))
    out['median_abs_W_rate'] = float(np.median(np.abs(rate)))
    out['median_abs_flux_mean'] = float(np.median(np.abs(fm)))
    out['median_abs_defect_est'] = float(np.median(np.abs(rate - fm)))
    out['frac_same_sign'] = float(np.mean(np.sign(rate) == np.sign(fm)))
    nz = rate != 0
    out['median_flux_over_W_rate'] = float(np.median(fm[nz] / rate[nz])) if nz.any() else NAN
    out['sum_W_ext'] = float(np.sum(rate * dts))
    out['sum_flux_mean_dt'] = float(np.sum(fm * dts))
    out['sum_defect_est_dt'] = float(np.sum((rate - fm) * dts))
    ok = np.isfinite(fmid)
    out['n_mid'] = int(np.sum(ok))
    if ok.any():
        out['sum_flux_mid_dt'] = float(np.sum(fmid[ok] * dts[ok]))
        out['sum_defect_est_mid_dt'] = float(np.sum((rate[ok] - fmid[ok]) * dts[ok]))
        out['median_abs_defect_est_mid'] = float(np.median(np.abs(rate[ok] - fmid[ok])))
    return out


def v3b_w_variants(res):
    """Every pair key beginning with W whose values are numbers (W_ext and its variants: the quadrature rules, the endpoint weights, the lower limits
    R_a and R_b, the parts), summed over the included pairs, with the rms per pair, the summed difference from W_ext and the count of pairs with a value."""
    pairs = res.get('pairs') or []
    keys = []
    for p in pairs:
        for k in p.keys():
            k = str(k)
            if k.startswith('W') and k not in keys:
                keys.append(k)
    W0 = np.array([_float(p.get('W_ext')) for p in pairs], float)
    rows = []
    for k in keys:
        vals = np.array([_float(p.get(k)) for p in pairs], float)
        fin = np.isfinite(vals)
        if not fin.any():
            continue
        diff = vals - W0
        dfin = np.isfinite(diff)
        rows.append((k, float(np.sum(vals[fin])), float(np.sqrt(np.mean(vals[fin] ** 2))), float(np.sum(diff[dfin])) if dfin.any() else NAN, int(fin.sum())))
    return rows


def v3b_tail_lines(res):
    """The per-record keys whose name contains 'tail' and whose values are numbers, differenced across each included pair and set beside the pair floor:
    version 2's per-pair floor carries the stopping tolerance and the screen difference only (its tail allowance is in the per-record sigma_E and hence
    in the cumulative endpoint floor), so this is the pairwise tail difference that floor does not contain."""
    recs = res.get('records') or []
    pairs = res.get('pairs') or []
    by_T = {}
    for r in recs:
        t = _float(r.get('T_over_rs'))
        if np.isfinite(t):
            by_T[round(t, 9)] = r
    keys = []
    for r in recs:
        for k in r.keys():
            if 'tail' in str(k).lower() and not str(k).startswith('_') and isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool) and k not in keys:
                keys.append(k)
    out = []
    for k in keys:
        diffs, excess, n_floor = [], 0, 0
        for p in pairs:
            ta, tb = _float(p.get('T_a')), _float(p.get('T_b'))
            if not np.all(np.isfinite([ta, tb])):
                continue
            a, b = by_T.get(round(ta, 9)), by_T.get(round(tb, 9))
            if a is None or b is None:
                continue
            da, db = _float(a.get(k)), _float(b.get(k))
            if not np.all(np.isfinite([da, db])):
                continue
            dd = abs(db - da)
            diffs.append(dd)
            fl = _float(p.get('floor'))
            if not np.isfinite(fl):
                fl = _float(p.get('floor_pair'))
            if np.isfinite(fl):
                n_floor += 1
                if dd > fl:
                    excess += 1
        if diffs:
            out.append('%s: |b - a| median %s, max %s over %d pairs; exceeds the pair floor in %d of the %d pairs with a floor value' % (
                k, _v3b_fmt(np.median(diffs), 3), _v3b_fmt(np.max(diffs), 3), len(diffs), excess, n_floor))
    return out


def v3b_stress_line(meta):
    """The one-line description of a companion stress (the md Inputs line and the V3 STRESS console line)."""
    rng = meta.get('node_range') or [NAN, NAN]
    return ("companion npz %s, key '%s' (%s); node grid '%s': %s nodes at spacing %s, range [%s, %s]; rows matched %s; reading %s, scale %s; no kernel call "
            "(%s); %s of %s records read (%s unmatched) in %s s; companion ell %s, g %s (fields ell %s, g %s); the companion's energy densities on the nodes "
            "against the dense rho_ell: best combination(s) %s, rel RMS median %s (the expected eps_dust + eps_sea - eps_R2: median %s); N2avg against the "
            "dense N^2: rel RMS median %s; stride-2 interpolation error relative to max |p_r|: median %s, max %s (estimate at the node density: median %s; "
            "near the root: median %s); |spline - local polynomial| at R: median %s; tangential stress key %s (%s)" % (
                meta.get('path'), meta.get('key'), meta.get('note'), meta.get('grid_key'), meta.get('n_nodes'), _v3b_fmt(meta.get('h_node'), 4),
                _v3b_fmt(rng[0], 6), _v3b_fmt(rng[1], 6), meta.get('row_matching'), meta.get('reading'), _v3b_fmt(meta.get('scale')), meta.get('kernel'),
                meta.get('n_ok'), meta.get('n_records'), meta.get('n_unmatched'), _v3b_fmt(meta.get('seconds'), 3), _v3b_fmt(meta.get('companion_ell')),
                _v3b_fmt(meta.get('companion_g')), _v3b_fmt(meta.get('fields_ell')), _v3b_fmt(meta.get('fields_g')), meta.get('eps_combo_counts'),
                _v3b_fmt(meta.get('handoff_eps_vs_rho_rel_median'), 3), _v3b_fmt(meta.get('handoff_eps_expected_vs_rho_rel_median'), 3),
                _v3b_fmt(meta.get('handoff_N2avg_vs_dense_rel_median'), 3), _v3b_fmt(meta.get('p_interp_err_half_rel_median'), 3),
                _v3b_fmt(meta.get('p_interp_err_half_rel_max'), 3), _v3b_fmt(meta.get('p_interp_err_rel_median'), 3),
                _v3b_fmt(meta.get('p_interp_err_root_rel_median'), 3), _v3b_fmt(meta.get('p_R_method_diff_rel_median'), 3), meta.get('pt_key') or 'none',
                meta.get('pt_note') or 'not read'))


def v3b_stress_definition(meta):
    """The definitions entry stress_normalization when the stress is a companion array."""
    return ("p_r = the key '%s' of the companion npz %s (%s): the engine's own kernel average per unit volume of the radial stress on the companion's node "
            "grid '%s' (%s nodes at spacing %s; the _ell of its name the averaging scale; companion ell %s, g %s against the fields' %s, %s), read as a "
            "density (%s; scale %s applied), matched to the records by T (%s) and carried to the dense grid and to the dense zeros by the not-a-knot cubic "
            "spline through the nodes (end values held outside the node range). No kernel is applied here: rho_ell and J_ell at the root are the exact kernel "
            "values of the records' sources and p_r(R) is the spline's value, so T_ll(R) carries the stress interpolation error, estimated per record by the "
            "stride-2 test (the spline through every other node against the stored values at the skipped nodes, divided by 2^%d for the full node density: "
            "p_interp_err_rel over the grid, p_interp_err_root_rel within the window around R, p_interp_err_root_abs in the units of p_r; Tll_R_over_interp_err "
            "is |T_ll(R)| over that error) and checked by the local polynomial of the records' degree at R (p_R_method_diff_rel) and by T_ll at the two nodes "
            "bracketing R, where the stress is the stored node value and rho_ell, J_ell are interpolated from the dense grid (Tll_node_below_R, Tll_node_above_R; "
            "Tll_nodes_bracket_same_sign). Normalization against rho_ell and J_ell: the companion's energy densities on the same nodes (eps_dust, eps_sea, "
            "eps_R2, in the combinations tried) against the dense rho_ell at the nodes (handoff_eps_combo, handoff_eps_vs_rho_rel; the expected combination "
            "eps_dust + eps_sea - eps_R2 in handoff_eps_expected_vs_rho_rel), which shows whether the companion's _ell fields are the same kernel averages per "
            "unit volume as the records' rho_ell; the ratios |p_r| max / |rho_ell| max, |p_r| max / |J_ell| max, p_r/rho_ell and p_r/J_ell at R (a per-node "
            "extensive quantity read as a density shows as a ratio of the order of the node shell volume 4 pi R^2 h_node = p_node_shell_volume_at_R; "
            "--pr-node-reading per-node divides it out); N2avg against the dense N^2 at the nodes (handoff_N2avg_vs_dense_rel) as the same-record check. "
            "The sign: the stress is read as the radial pressure p_r = T_ss of the normal frame (FIRSTLAW_rows_v1's P_r); the engine's stress module text is "
            "printed beside (--pr-scale -1 if it defines S_r as -T_rr). The tangential stress %s is read the same way at R and reported only. The flux: "
            "F = 4 pi r^2 [(N^2 + v^2) J + N v (rho + p_r)] of the continuity identity, at the root -4 pi R^2 N_R^2 T_ll (flux_F_R), at the dense collar "
            "from the dense arrays (flux_F_rc); per pair W_ext/dt against the mean F(R) - F(r_c) in the md." % (
                meta.get('key'), meta.get('path'), meta.get('note'), meta.get('grid_key'), meta.get('n_nodes'), _v3b_fmt(meta.get('h_node'), 4),
                _v3b_fmt(meta.get('companion_ell')), _v3b_fmt(meta.get('companion_g')), _v3b_fmt(meta.get('fields_ell')), _v3b_fmt(meta.get('fields_g')),
                meta.get('reading'), _v3b_fmt(meta.get('scale')), meta.get('row_matching'), int(V3B_INTERP_ORDER), meta.get('pt_key') or 'none'))


def v3b_md_lines(res):
    """The md lines of version 3b, placed in the version-3 section: the stress source, normalization and interpolation, the T_ll sign checks, the
    tangential stress, the flux reading of T_ll and the pairwise balance, every W variant of the pairs, the pairwise tail differences."""
    st = res.get('stats') or {}
    meta = res.get('stress_meta') or {}
    thr = res.get('thresholds') or {}

    def s(key_, p=3):
        return _v3b_fmt(st.get(key_), p)

    def f(x, p=4):
        return _v3b_fmt(x, p)
    L = []
    if meta.get('companion'):
        L.append('- **stress source (version 3b)**: %s' % v3b_stress_line(meta))
        for ln in meta.get('module_note') or []:
            L.append('- engine stress module text: %s' % ln)
        L.append("- interpolation of p_r from the companion nodes to the dense grid (not-a-knot cubic spline): stride-2 error relative to max |p_r|: median %s, "
                 "max %s; the estimate at the node density (divided by 2^%d): median %s, max %s; near the outer root: median %s, max %s; |spline - local "
                 "polynomial of degree %s| at R relative to max |p_r|: median %s, max %s; dense cells outside the node range (end values held): median %s; "
                 "|T_ll(R)| over the estimated interpolation error of p_r at R: median %s, min %s; records with |T_ll(R)| within that error: %s of %s; T_ll at "
                 "the two nodes bracketing R (the stored node stress, rho_ell and J_ell interpolated from the dense grid) has the sign of T_ll(R) at both nodes "
                 "in a fraction %s of %s records (T_ll at the node below R: median %s; above: median %s)" % (
                     s('p_interp_err_half_rel_median'), s('p_interp_err_half_rel_max'), int(V3B_INTERP_ORDER), s('p_interp_err_rel_median'),
                     s('p_interp_err_rel_max'), s('p_interp_err_root_rel_median'), s('p_interp_err_root_rel_max'), thr.get('interp_degree'),
                     s('p_R_method_diff_rel_median'), s('p_R_method_diff_rel_max'), s('p_extrap_cells_median'), s('Tll_R_over_interp_err_median'),
                     s('Tll_R_over_interp_err_min'), st.get('n_Tll_R_within_interp_err'), st.get('n_Tll_R_over_interp_err_finite'),
                     s('frac_Tll_nodes_bracket_same_sign'), st.get('n_Tll_nodes_bracket_finite'), s('Tll_node_below_R_median'), s('Tll_node_above_R_median')))
        L.append("- normalization of the stress against rho_ell and J_ell: |p_r| max over the grid / |rho_ell| max: median %s (a per-node extensive quantity read as "
                 "a density would give the node shell volume 4 pi R^2 h_node = %s instead); p_r/rho_ell at R: median %s (min %s, max %s); p_r/J_ell at R: median "
                 "%s; |p_r| max / |J_ell| max: median %s; the companion's energy densities on the nodes against the dense rho_ell at the nodes: best "
                 "combination(s) %s, rel RMS median %s, max %s; the expected combination eps_dust + eps_sea - eps_R2: median %s; N2avg against the dense N^2 at "
                 "the nodes: median %s; companion ell %s and g %s against the fields' %s and %s" % (
                     s('p_max_abs_over_rho_max_abs_median'), s('p_node_shell_volume_at_R_median'), s('p_over_rho_R_median'), s('p_over_rho_R_min'),
                     s('p_over_rho_R_max'), s('p_R_over_J_R_median'), s('p_max_abs_over_J_max_abs_median'), st.get('handoff_eps_combo_counts'),
                     s('handoff_eps_vs_rho_rel_median'), s('handoff_eps_vs_rho_rel_max'), s('handoff_eps_expected_vs_rho_rel_median'),
                     s('handoff_N2avg_vs_dense_rel_median'), _v3b_fmt(meta.get('companion_ell')), _v3b_fmt(meta.get('companion_g')),
                     _v3b_fmt(meta.get('fields_ell')), _v3b_fmt(meta.get('fields_g'))))
        if np.isfinite(_float(st.get('pt_R_median'))):
            L.append("- tangential stress (key %s; reported, not used in T_ll): p_t(R) median %s (min %s, max %s); p_t/p_r at R: median %s; p_t/rho_ell at R: median %s" % (
                meta.get('pt_key'), s('pt_R_median'), s('pt_R_min'), s('pt_R_max'), s('pt_over_p_R_median'), s('pt_over_rho_R_median')))
    fb = v3b_flux_balance(res)
    if fb.get('n'):
        L.append("- flux reading of T_ll (the continuity identity of FIRSTLAW_rows_v1: F = 4 pi r^2 [(N^2 + v^2) J + N v (rho + p_r)], at v = -N equal to "
                 "-4 pi R^2 N_R^2 T_ll): F(R) median %s (min %s, max %s); F(r_c) at the dense collar median %s; per included pair, %s: W_ext/dt summed %s "
                 "against the pair mean of F(R) - F(r_c) summed %s (the same sign in a fraction %s of %d pairs; median per pair |W_ext/dt| %s, "
                 "|F(R) - F(r_c)| %s, median ratio %s); their difference, the finite-record estimate of 4 pi int r^2 (N W_0 + v W_r) dr plus this estimator's "
                 "record-step and moving-lower-limit terms, summed %s, median |.| %s; weighted by dt (energy units, beside the summed W_ext %s): sum F dt %s, "
                 "difference %s; with F on the midpoint fields at R_mid in place of the pair mean (%d pairs): sum F_mid dt %s, difference %s. This assumes "
                 "the records' T is the time of the continuity rows and J the flux density in those units; a unit mismatch shows as a common factor between "
                 "W_ext/dt and F, not as a defect" % (
                     s('flux_F_R_median'), s('flux_F_R_min'), s('flux_F_R_max'), s('flux_F_rc_median'), fb.get('unit'), f(fb.get('sum_W_rate')),
                     f(fb.get('sum_flux_mean')), f(fb.get('frac_same_sign')), int(fb.get('n')), f(fb.get('median_abs_W_rate')), f(fb.get('median_abs_flux_mean')),
                     f(fb.get('median_flux_over_W_rate')), f(fb.get('sum_defect_est')), f(fb.get('median_abs_defect_est')), f(fb.get('sum_W_ext')),
                     f(fb.get('sum_flux_mean_dt')), f(fb.get('sum_defect_est_dt')), int(fb.get('n_mid') or 0), f(fb.get('sum_flux_mid_dt')),
                     f(fb.get('sum_defect_est_mid_dt'))))
    rows = v3b_w_variants(res)
    if rows:
        L.append('- every pair key beginning with W, summed over the included pairs (the quadrature rules, the endpoint weights, the lower limits R_a and R_b, the '
                 'parts, as the pair dict names them): ' + '; '.join('%s sum %s (rms %s, summed difference from W_ext %s, %d pairs)' % (
                     k, f(sm, 6), f(rm, 4), f(dd, 4), n) for k, sm, rm, dd, n in rows))
    tl = v3b_tail_lines(res)
    if tl:
        L.append('- pairwise tail differences (per-record keys naming the tail, |b - a| per included pair, beside the pair floor, which carries the stopping tolerance '
                 'and the screen difference only): ' + '; '.join(tl))
    return L


def v3b_console_lines(res):
    st = res.get('stats') or {}
    meta = res.get('stress_meta') or {}

    def s(key_, p=3):
        return _v3b_fmt(st.get(key_), p)
    L = []
    if meta.get('companion'):
        L.append('V3B NORMALIZATION: |p| max / |rho| max median %s (node shell volume at R %s); p/rho at R median %s (min %s, max %s); p/J at R median %s; companion '
                 'eps on the nodes vs dense rho_ell: %s, rel RMS median %s (expected combination median %s); N2avg vs dense N^2 median %s; companion ell %s, g %s '
                 '(fields %s, %s); rows matched %s; %s of %s records (%s unmatched)' % (
                     s('p_max_abs_over_rho_max_abs_median'), s('p_node_shell_volume_at_R_median'), s('p_over_rho_R_median'), s('p_over_rho_R_min'),
                     s('p_over_rho_R_max'), s('p_R_over_J_R_median'), st.get('handoff_eps_combo_counts'), s('handoff_eps_vs_rho_rel_median'),
                     s('handoff_eps_expected_vs_rho_rel_median'), s('handoff_N2avg_vs_dense_rel_median'), _v3b_fmt(meta.get('companion_ell')),
                     _v3b_fmt(meta.get('companion_g')), _v3b_fmt(meta.get('fields_ell')), _v3b_fmt(meta.get('fields_g')), meta.get('row_matching'),
                     meta.get('n_ok'), meta.get('n_records'), meta.get('n_unmatched')))
        L.append('V3B INTERPOLATION: stride-2 error rel to max|p| median %s, max %s; estimate at the node density median %s (near the root median %s, max %s); '
                 '|spline - local polynomial|(R) median %s; |T_ll(R)| / interpolation error at R median %s, min %s; records with |T_ll(R)| within the error %s of %s; '
                 'T_ll at the bracketing nodes of the sign of T_ll(R): fraction %s of %s' % (
                     s('p_interp_err_half_rel_median'), s('p_interp_err_half_rel_max'), s('p_interp_err_rel_median'), s('p_interp_err_root_rel_median'),
                     s('p_interp_err_root_rel_max'), s('p_R_method_diff_rel_median'), s('Tll_R_over_interp_err_median'), s('Tll_R_over_interp_err_min'),
                     st.get('n_Tll_R_within_interp_err'), st.get('n_Tll_R_over_interp_err_finite'), s('frac_Tll_nodes_bracket_same_sign'),
                     st.get('n_Tll_nodes_bracket_finite')))
        if np.isfinite(_float(st.get('pt_R_median'))):
            L.append('V3B TANGENTIAL: key %s, p_t(R) median %s (min %s, max %s), p_t/p_r at R median %s, p_t/rho at R median %s' % (
                meta.get('pt_key'), s('pt_R_median'), s('pt_R_min'), s('pt_R_max'), s('pt_over_p_R_median'), s('pt_over_rho_R_median')))
        for ln in meta.get('module_note') or []:
            L.append('V3B STRESS MODULE: %s' % ln)
    fb = v3b_flux_balance(res)
    if fb.get('n'):
        L.append('V3B FLUX: F(R) = -4 pi R^2 N_R^2 T_ll median %s (min %s, max %s), F(r_c) median %s; per pair (%s) W_ext/dt summed %s vs mean [F(R) - F(r_c)] summed %s '
                 '(same sign fraction %s of %d; median ratio %s); continuity-defect estimate summed %s, median |.| %s; weighted by dt: sum W_ext %s, sum F dt %s, '
                 'difference %s (with F on the midpoint fields at R_mid: %s)' % (
                     s('flux_F_R_median'), s('flux_F_R_min'), s('flux_F_R_max'), s('flux_F_rc_median'), fb.get('unit'), _v3b_fmt(fb.get('sum_W_rate')),
                     _v3b_fmt(fb.get('sum_flux_mean')), _v3b_fmt(fb.get('frac_same_sign')), int(fb.get('n')), _v3b_fmt(fb.get('median_flux_over_W_rate')),
                     _v3b_fmt(fb.get('sum_defect_est')), _v3b_fmt(fb.get('median_abs_defect_est')), _v3b_fmt(fb.get('sum_W_ext')),
                     _v3b_fmt(fb.get('sum_flux_mean_dt')), _v3b_fmt(fb.get('sum_defect_est_dt')), _v3b_fmt(fb.get('sum_defect_est_mid_dt'))))
    rows = v3b_w_variants(res)
    if rows:
        L.append('V3B W VARIANTS (summed over the included pairs): ' + '; '.join('%s %s' % (k, _v3b_fmt(sm, 6)) for k, sm, rm, dd, n in rows))
    tl = v3b_tail_lines(res)
    if tl:
        L.append('V3B TAIL (pairwise differences beside the pair floor): ' + '; '.join(tl))
    return L


def _v3b_T_key(x):
    if isinstance(x, dict):
        for name in ('T_over_rs', 'T'):
            v = x.get(name)
            if v is not None:
                try:
                    return round(float(v), 9)
                except (TypeError, ValueError):
                    pass
    return None


def _v3b_pair_key(x):
    if isinstance(x, dict):
        try:
            return (round(float(x['T_a']), 9), round(float(x['T_b']), 9))
        except (KeyError, TypeError, ValueError):
            return None
    return None


def compare_json_files_matched(ref_path, new_path, sections=V3_COMPARE_SECTIONS):
    """compare_json_files with the lists matched by T: every record of the reference json is compared with the record of the same T_over_rs in the new
    json, every pair and excluded pair with the one of the same (T_a, T_b); the leaves of each matched element must be present and equal (the new json
    may carry more keys and more elements).  The set-dependent sections stats and matching are compared only when both jsons have the same number of
    records; grid and the scalar sections always.  Returns (n leaves compared, [differences], note)."""
    with open(ref_path) as fh:
        ref = json.load(fh)
    with open(new_path) as fh:
        new = json.load(fh)
    res = {'n': 0, 'bad': []}
    parts = []
    for sec, keyf in (('records', _v3b_T_key), ('pairs', _v3b_pair_key), ('excluded_pairs', _v3b_pair_key)):
        if sec not in sections or sec not in ref:
            continue
        if sec not in new:
            res['bad'].append(('/' + sec, 'present', 'missing'))
            continue
        rl, nl = list(ref[sec] or []), list(new[sec] or [])
        nmap = {}
        for j, x in enumerate(nl):
            kx = keyf(x)
            if kx is not None and kx not in nmap:
                nmap[kx] = j
        matched = 0
        for i, x in enumerate(rl):
            kx = keyf(x)
            if kx is not None:
                j = nmap.get(kx)
            else:
                j = i if i < len(nl) else None
            if j is None:
                res['bad'].append(('/%s[%d]' % (sec, i), 'T %s' % (kx,), 'no element of that T in the new json'))
                continue
            matched += 1
            compare_trees(x, nl[j], '/%s[%d]' % (sec, i), V3_COMPARE_EXEMPT, res)
        parts.append('%s %d of %d matched by T (%d in the new json)' % (sec, matched, len(rl), len(nl)))
    n_ref, n_new = len(ref.get('records') or []), len(new.get('records') or [])
    same_set = n_ref == n_new
    for sec in sections:
        if sec in ('records', 'pairs', 'excluded_pairs') or sec not in ref:
            continue
        if sec in ('stats', 'matching') and not same_set:
            parts.append('%s not compared (set-dependent: %d against %d records)' % (sec, n_ref, n_new))
            continue
        if sec not in new:
            res['bad'].append(('/' + sec, 'present', 'missing'))
            continue
        compare_trees(ref[sec], new[sec], '/' + sec, V3_COMPARE_EXEMPT, res)
    return res['n'], res['bad'], '; '.join(parts)


def _v3_norm(key):
    return re.sub(r'[^a-z0-9]', '', str(key).lower())


def find_stress_key(keys, explicit=None):
    """The fields key of the per-site radial stress: --pr-key when given ('none' disables), else the keys whose name, lowered and
    stripped of separators, is one of V3_STRESS_NORMS_R (P_r_tot, Pr_tot, p_r_tot, sigma_rr_tot, S_rr_tot, T_rr_tot, stress_r_tot,
    P_rad_tot, ... and the same without _tot), the _tot forms first; keys naming the pairs' part are not candidates.  The tangential
    stress key is found the same way and reported only.  Returns (key, note, candidates, pt_key)."""
    keys = [str(k) for k in (keys or [])]
    cands = [k for k in keys if _v3_norm(k) in V3_STRESS_NORMS_R and 'pair' not in k.lower()]
    cands.sort(key=lambda k: (0 if _v3_norm(k).endswith('tot') else 1, keys.index(k)))
    pts = [k for k in keys if _v3_norm(k) in V3_STRESS_NORMS_T and 'pair' not in k.lower()]
    pts.sort(key=lambda k: (0 if _v3_norm(k).endswith('tot') else 1, keys.index(k)))
    pt_key = pts[0] if pts else None
    if explicit is not None:
        if str(explicit).lower() == 'none':
            return None, 'disabled by --pr-key none', cands, pt_key
        if explicit in keys:
            return str(explicit), 'named by --pr-key', cands, pt_key
        return None, "--pr-key '%s' is not among the fields keys %s" % (explicit, keys), cands, pt_key
    if not cands:
        return None, 'no radial stress among the fields keys %s' % keys, cands, pt_key
    note = 'found by name among the fields keys' + ((' (other candidates: %s)' % ', '.join(cands[1:])) if len(cands) > 1 else '')
    return cands[0], note, cands, pt_key


def read_stress(npz_path, key, R, reading, scale, h_site):
    """The stored stress array (rows x sites) in the per-site normalization of eps_tot: 'per-site' as stored; 'density' multiplied by
    the shell volume 4 pi R_n^2 h of each site; then times scale."""
    z = np.load(npz_path, allow_pickle=False)
    P = np.asarray(z[key], float)
    if P.ndim == 1:
        P = P[None, :]
    R = np.asarray(R, float)
    if str(reading) == 'density':
        P = P * (FOUR_PI * R * R * float(h_site))[None, :]
    return P * float(scale)


def smooth_stress_record(FLD, R, P, ell, r_last_node, support_rel, r_fine, zeros):
    """The engine's kernel average per unit volume of one record's per-site stress P: p = SmoothedSources5(P, P, R, ell, r_last_node,
    support_rel).rho on the dense grid and, by the same object, at every dense zero (exact kernel values, as rho_R and J_R are);
    .J of the same call against .rho as the kernel check (0 when the engine's scalar and vector kernels coincide); the quadrature
    int 4 pi r^2 p dr against sum P (0 for a per-site quantity, like the engine's charge_conservation_quadrature for eps_tot)."""
    t0 = time.time()
    R = np.asarray(R, float)
    P = np.asarray(P, float)
    r_fine = np.asarray(r_fine, float)
    zeros = [float(z) for z in (zeros or [])]
    out = {'ok': False, 'error': None, 'seconds': NAN, 'sum_P': float(P.sum()), 'sum_abs_P': float(np.abs(P).sum()), 'p': None,
           'p_at_zeros': None, 'quadrature': NAN, 'quadrature_rel': NAN, 'kernel_check': NAN, 'kernel': None}
    if out['sum_abs_P'] == 0.0:
        out.update(ok=True, p=np.zeros(r_fine.size), p_at_zeros=[0.0] * len(zeros), quadrature=0.0, quadrature_rel=NAN, kernel_check=0.0,
                   kernel='identically zero stress (no kernel call)', seconds=time.time() - t0)
        return out
    if not hasattr(FLD, 'SmoothedSources5'):
        out['error'] = 'the engine has no SmoothedSources5'
        out['seconds'] = time.time() - t0
        return out
    try:
        src = FLD.SmoothedSources5(P, P.copy(), R, ell, float(r_last_node), support_rel)
        p = _ev(src.rho, r_fine)
        pJ = _ev(src.J, r_fine)
        pz = [float(_ev(src.rho, np.array([z]))[0]) for z in zeros]
    except Exception as exc:
        out['error'] = 'SmoothedSources5 on the stress raised %s: %s' % (type(exc).__name__, exc)
        out['traceback'] = traceback.format_exc()
        out['seconds'] = time.time() - t0
        return out
    scale = float(np.max(np.abs(p))) if p.size else 0.0
    out['p'] = p
    out['p_at_zeros'] = pz
    out['kernel_check'] = float(np.max(np.abs(p - pJ))) / scale if scale > 0 else NAN
    out['kernel'] = 'SmoothedSources5(P, P, R, ell, r_last_node, support_rel).rho'
    out['quadrature'] = float(_trapz(FOUR_PI * r_fine * r_fine * p, r_fine))
    out['quadrature_rel'] = out['quadrature'] / out['sum_P'] - 1.0 if out['sum_P'] != 0.0 else NAN
    out['ok'] = True
    out['seconds'] = time.time() - t0
    return out


def _interp_finite(R, y, x0, degree):
    """interp_at's value at x0; NaN when the fit fails or is not finite (a window touching r = 0, where g is undefined)."""
    try:
        val = interp_at(np.asarray(R, float), np.asarray(y, float), float(x0), int(degree))[0]
    except Exception:
        return NAN
    return float(val) if np.isfinite(val) else NAN


def root_defects_from_arrays(R, e, g, N, v, rho, J, Np, G, zeros, degree):
    """The two row defects at every dense zero z of N + v, read from the arrays on the dense grid (the e and g that the applied
    identity's defect integral uses: the 4th-order stencil at the grid spacing) by the local polynomial of the given degree:
    e(z) = M' - 4 pi z^2 (rho + u J) (the energy row), g(z) = N'/N + 4 pi G z J/u (the momentum row); the relative forms divide by
    the rows' own terms at z, 4 pi z^2 (|rho| + |u J|) and |N'/N| + 4 pi G z |J/u|.  Returns a list of dicts in increasing z."""
    R = np.asarray(R, float)
    N = np.asarray(N, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        NpN = np.where(N != 0.0, np.asarray(Np, float) / np.where(N != 0.0, N, 1.0), np.nan)
    out = []
    for z in sorted(float(x) for x in (zeros or [])):
        item = {'r': z, 'type': None, 'e': NAN, 'g': NAN, 'e_rel': NAN, 'g_rel': NAN, 'den_e': NAN, 'den_g': NAN, 'rho': NAN, 'J': NAN,
                'u': NAN, 'NpN': NAN}
        if R.size < 2 or not (R[0] <= z <= R[-1]):
            out.append(item)
            continue
        ez, gz, rz, Jz, Nz, vz, nz = (_interp_finite(R, arr, z, degree) for arr in (e, g, rho, J, N, v, NpN))
        uz = vz / Nz if (np.isfinite(Nz) and Nz != 0.0 and np.isfinite(vz)) else NAN
        den_e = FOUR_PI * z * z * (abs(rz) + abs(uz * Jz)) if np.all(np.isfinite([rz, uz, Jz])) else NAN
        den_g = (abs(nz) + FOUR_PI * G * z * abs(Jz / uz)) if (np.all(np.isfinite([nz, Jz, uz])) and uz != 0.0) else NAN
        item.update(e=ez, g=gz, rho=rz, J=Jz, u=uz, NpN=nz, den_e=den_e, den_g=den_g,
                    e_rel=(ez / den_e) if (np.isfinite(den_e) and den_e > 0 and np.isfinite(ez)) else NAN,
                    g_rel=(gz / den_g) if (np.isfinite(den_g) and den_g > 0 and np.isfinite(gz)) else NAN)
        out.append(item)
    return out


def _v3_offsets(spec):
    out = []
    for s in str(spec).split(','):
        s = s.strip()
        if s:
            try:
                out.append(float(s))
            except ValueError:
                pass
    return out


def v3_record_columns(out, arrays, r_fine, h_fine, G, opts, d, has_root_dense, stress, Np):
    """The version-3 per-record columns, called from analyse_record once the dense arrays and the defects e, g exist: the row defect
    at every dense zero, T_ll = rho + p_r - 2J at the roots and its profile across the outer root."""
    N, v, rho, J, e, g = (np.asarray(arrays[key], float) for key in ('N', 'v', 'rho', 'J', 'e', 'g'))
    r_fine = np.asarray(r_fine, float)
    deg = int(opts.interp_degree)
    ell = float(opts.ell)
    zd = sorted(float(z) for z in (d.get('zeros_dense') or []))
    details = list(d.get('zeros_detail') or [])
    types = {}
    for z in details:
        try:
            types[round(float(z['r']), 10)] = z.get('type')
        except (KeyError, TypeError, ValueError):
            pass
    rd = root_defects_from_arrays(r_fine, e, g, N, v, rho, J, Np, G, zd, deg)
    for i, item in enumerate(rd):
        item['type'] = types.get(round(item['r'], 10)) or ('outer' if i == len(rd) - 1 else 'inner')
    out['root_defects'] = rd
    R0 = _float(out.get('R_dense'))
    rooted = bool(has_root_dense) and np.isfinite(R0) and bool(rd)
    if rooted:
        outer = rd[-1]
        out['e_root'], out['g_root'], out['e_root_rel'], out['g_root_rel'] = outer['e'], outer['g'], outer['e_rel'], outer['g_rel']
        out['den_e_root'], out['den_g_root'], out['Np_over_N_root'] = outer['den_e'], outer['den_g'], outer['NpN']
        if len(rd) >= 2:
            xi = rd[-2]
            out['R_in'], out['e_root_in'], out['g_root_in'] = xi['r'], xi['e'], xi['g']
            out['e_root_in_rel'], out['g_root_in_rel'] = xi['e_rel'], xi['g_rel']
        N_R = _float(out.get('N_R'))
        if np.isfinite(N_R) and N_R != 0.0 and R0 > 0.0:
            out['kappa_corr_root'] = -N_R * G * out['e_root'] / R0 if np.isfinite(out['e_root']) else NAN
            kfd, kfo = _float(out.get('kappa_fd')), _float(out.get('kappa_formula'))
            out['e_root_from_kappa'] = -(kfd - kfo) * R0 / (N_R * G)
            out['e_root_minus_from_kappa'] = out['e_root'] - out['e_root_from_kappa']
    p_arr, p_at = None, None
    if stress is not None:
        if stress.get('ok') and stress.get('p') is not None and np.asarray(stress['p']).size == r_fine.size:
            p_arr = np.asarray(stress['p'], float)
            pz = list(stress.get('p_at_zeros') or [])
            if len(pz) == len(zd):
                p_at = [float(x) for x in pz]
            out['sum_P_tot'] = _float(stress.get('sum_P'))
            out['sum_abs_P_tot'] = _float(stress.get('sum_abs_P'))
            out['p_quadrature_rel'] = _float(stress.get('quadrature_rel'))
            out['p_kernel_check'] = _float(stress.get('kernel_check'))
            out['stress_seconds'] = _float(stress.get('seconds'))
            for key_ in V3B_STRESS_RECORD_KEYS:
                if key_ in stress:
                    out[key_] = stress.get(key_) if key_ in V3B_STRESS_STRING_KEYS else _float(stress.get(key_))
            se = _float((d.get('sources') or {}).get('sum_eps_tot'))
            out['sum_P_over_sum_eps'] = out['sum_P_tot'] / se if (np.isfinite(se) and se != 0.0) else NAN
            mr = float(np.max(np.abs(rho))) if rho.size else 0.0
            out['p_max_abs_over_rho_max_abs'] = float(np.max(np.abs(p_arr))) / mr if mr > 0 else NAN
        else:
            out['stress_error'] = stress.get('error') or 'no stress array on the dense grid'
    if p_arr is not None:
        Tll = rho + p_arr - 2.0 * J
        Tll_flip = rho + p_arr + 2.0 * J
        arrays['p'] = p_arr
        arrays['Tll'] = Tll
        arrays['Tll_flipJ'] = Tll_flip
        out['_p_profile'] = p_arr
        out['_Tll_profile'] = Tll
        out['Tll_grid_min'] = float(np.min(Tll))
        out['Tll_grid_max'] = float(np.max(Tll))
    if rooted:
        rho_R, J_R = _float(out.get('rho_R')), _float(out.get('J_R'))
        out['rho_minus_2J_R'] = rho_R - 2.0 * J_R
        out['rho_plus_2J_R'] = rho_R + 2.0 * J_R
        if p_arr is not None:
            if p_at is not None:
                p_R = p_at[-1]
                out['p_R_source'] = stress.get('p_at_zeros_source') or 'kernel value at the root'
            else:
                p_R = _interp_finite(r_fine, p_arr, R0, deg)
                out['p_R_source'] = 'interpolated from the dense grid'
            out['p_R'] = p_R
            out['Tll_R'] = rho_R + p_R - 2.0 * J_R
            out['Tll_flipJ_R'] = rho_R + p_R + 2.0 * J_R
            out['p_over_rho_R'] = p_R / rho_R if rho_R != 0.0 else NAN
            out['Tll_dimless_R'] = 8.0 * np.pi * G * R0 * R0 * out['Tll_R']
            v3b_root_extras(out, stress, G, R0, arrays, r_fine)
            if len(zd) >= 2 and len(details) == len(zd):
                di = details[-2]
                p_in = p_at[-2] if p_at is not None else _interp_finite(r_fine, p_arr, zd[-2], deg)
                out['p_R_in'] = p_in
                out['Tll_R_in'] = _float(di.get('rho')) + p_in - 2.0 * _float(di.get('J'))
            offsets = _v3_offsets(opts.Tll_offsets_ell)
            out['Tll_offsets_ell'] = offsets

            def at(arr, x):
                return _interp_finite(r_fine, arr, x, deg) if (r_fine[0] <= x <= r_fine[-1]) else NAN
            out['Tll_at_offsets'] = [at(Tll, R0 + o * ell) for o in offsets]
            out['Tll_flipJ_at_offsets'] = [at(Tll_flip, R0 + o * ell) for o in offsets]
            out['rho_at_offsets'] = [at(rho, R0 + o * ell) for o in offsets]
            out['p_at_offsets'] = [at(p_arr, R0 + o * ell) for o in offsets]
            out['J_at_offsets'] = [at(J, R0 + o * ell) for o in offsets]
            w = float(opts.Tll_window_ell) * ell
            n_half = max(1, int(round(w / h_fine)))
            rel = np.arange(-n_half, n_half + 1, dtype=float) * h_fine
            out['_Tll_window'] = np.array([at(Tll, R0 + x) for x in rel], float)
            out['_Tll_window_rel'] = rel
            mask = (r_fine >= R0 - w) & (r_fine <= R0 + w)
            Tw, rw = Tll[mask], r_fine[mask]
            if Tw.size:
                out['Tll_window_min'] = float(np.min(Tw))
                out['Tll_window_max'] = float(np.max(Tw))
                s = np.sign(Tw)
                s = np.where(s == 0, 1.0, s)
                ix = np.flatnonzero(s[:-1] * s[1:] < 0)
                out['Tll_window_n_zero_crossings'] = int(ix.size)
                if ix.size:
                    cr = rw[ix] - Tw[ix] * (rw[ix + 1] - rw[ix]) / (Tw[ix + 1] - Tw[ix])
                    j = int(np.argmin(np.abs(cr - R0)))
                    out['Tll_window_nearest_zero'] = float(cr[j])
                    out['Tll_window_nearest_zero_minus_R'] = float(cr[j] - R0)


def v3_pair_columns(pr, A, B, aA, aB, Rg, sl, Rm, dA, pre, deg):
    """The version-3 per-pair columns, called from analyse_pair after the version-2 columns: T_ll at both roots and their mean, the
    midpoint fields' T_ll at R_mid, the other null direction's form, rho - 2J, the stress ratio, the row defects at both roots and
    the applied identity's kappa correction -N_R G e(R)/R carried into column 2 (col2_defect_corr = (delta A/(8 pi G)) x its mean)."""
    for src, dst in (('Tll_R', 'Tll'), ('Tll_flipJ_R', 'Tll_flipJ'), ('rho_minus_2J_R', 'rho_minus_2J'), ('p_R', 'p'),
                     ('p_over_rho_R', 'p_over_rho'), ('Tll_R_in', 'Tll_in')):
        pr[dst + '_a'] = _float(A.get(src))
        pr[dst + '_b'] = _float(B.get(src))
        pr[dst + '_mean'] = 0.5 * (pr[dst + '_a'] + pr[dst + '_b'])
    for key in ('e_root', 'g_root', 'e_root_rel', 'g_root_rel', 'kappa_corr_root', 'e_root_in', 'g_root_in'):
        pr[key + '_a'] = _float(A.get(key))
        pr[key + '_b'] = _float(B.get(key))
    pr['Tll_mid'] = NAN
    ta, tb = aA.get('Tll'), aB.get('Tll')
    if ta is not None and tb is not None:
        Tm = 0.5 * (np.asarray(ta, float)[sl] + np.asarray(tb, float)[sl])
        pr['Tll_mid'] = _interp_finite(Rg, Tm, Rm, deg)
    pr['col2_defect_corr'] = pre * dA * 0.5 * (pr['kappa_corr_root_a'] + pr['kappa_corr_root_b'])
    c2 = _float(pr.get('col2'))
    pr['col2_defect_corr_over_col2'] = pr['col2_defect_corr'] / c2 if (np.isfinite(c2) and c2 != 0.0) else NAN
    v3b_pair_extras(pr, A, B, aA, aB, Rg, sl, Rm, deg)


def v3_record_stats(st, rooted_ok):
    """The version-3 record-level statistics (over the rooted re-solved records), added to the stats dict."""
    for key in ('Tll_R', 'Tll_flipJ_R', 'rho_minus_2J_R', 'p_R', 'p_over_rho_R', 'Tll_dimless_R', 'Tll_R_in', 'e_root', 'g_root', 'e_root_rel',
                'g_root_rel', 'e_root_in', 'g_root_in', 'e_root_in_rel', 'g_root_in_rel', 'e_root_from_kappa', 'e_root_minus_from_kappa',
                'kappa_corr_root', 'p_quadrature_rel', 'p_kernel_check', 'sum_P_over_sum_eps', 'p_max_abs_over_rho_max_abs', 'Tll_window_min',
                'Tll_window_max', 'Tll_window_n_zero_crossings', 'Tll_window_nearest_zero_minus_R', 'stress_seconds'):
        vals = np.array([_float(r.get(key)) for r in rooted_ok], float)
        st[key + '_median'] = _nan(np.median, vals)
        st[key + '_max'] = _nan(np.max, vals)
        st[key + '_min'] = _nan(np.min, vals)
        st[key + '_max_abs'] = _nan(np.max, np.abs(vals))
        st[key + '_abs_median'] = _nan(np.median, np.abs(vals))
    for key in ('Tll_R', 'Tll_flipJ_R', 'rho_minus_2J_R', 'Tll_R_in'):
        vals = np.array([_float(r.get(key)) for r in rooted_ok], float)
        fin = vals[np.isfinite(vals)]
        st['n_' + key + '_finite'] = int(fin.size)
        st['frac_' + key + '_positive'] = float(np.mean(fin > 0)) if fin.size else NAN
        st['n_' + key + '_negative'] = int(np.sum(fin < 0))
    st['n_Tll_window_with_zero_crossing'] = int(sum(1 for r in rooted_ok if _float(r.get('Tll_window_n_zero_crossings')) > 0))
    st['n_rooted_with_inner_zero'] = int(sum(1 for r in rooted_ok if np.isfinite(_float(r.get('R_in')))))
    st['stress_seconds_total'] = _nan(np.sum, [_float(r.get('stress_seconds')) for r in rooted_ok])
    v3b_record_stats(st, rooted_ok)


def v3_pair_stats(st, pairs):
    """The version-3 pair-level statistics (over the included pairs), added to the stats dict."""
    for key in ('Tll_mean', 'Tll_mid', 'Tll_flipJ_mean', 'rho_minus_2J_mean', 'Tll_in_mean', 'col2_defect_corr', 'col2_defect_corr_over_col2'):
        vals = np.array([_float(p.get(key)) for p in pairs], float)
        st[key + '_median'] = _nan(np.median, vals)
        st[key + '_min'] = _nan(np.min, vals)
        st[key + '_max'] = _nan(np.max, vals)
        st[key + '_max_abs'] = _nan(np.max, np.abs(vals))
        st[key + '_abs_median'] = _nan(np.median, np.abs(vals))
    st['sum_col2_defect_corr'] = _nan(np.sum, [_float(p.get('col2_defect_corr')) for p in pairs])
    for key in ('Tll_mean', 'Tll_mid'):
        vals = np.array([_float(p.get(key)) for p in pairs], float)
        fin = vals[np.isfinite(vals)]
        st['n_' + key + '_finite'] = int(fin.size)
        st['frac_' + key + '_positive'] = float(np.mean(fin > 0)) if fin.size else NAN
    sg = [(np.sign(_float(p.get('Tll_mean'))), np.sign(_float(p.get('dA')))) for p in pairs]
    ok = [(x, y) for x, y in sg if np.isfinite(x) and np.isfinite(y) and x != 0 and y != 0]
    st['frac_Tll_mean_same_sign_as_dA'] = float(np.mean([x == y for x, y in ok])) if ok else NAN


def v3_definitions(opts, stress, h_fine):
    """The version-3 entries of the definitions dict: what the new columns are, as computed."""
    meta = (stress or {}).get('meta') or {}
    key = meta.get('key')
    if key and meta.get('companion'):
        stress_text = v3b_stress_definition(meta)
    elif key:
        stress_text = ("p_r = p_ell, the engine's kernel average per unit volume of the stored fields key '%s' (%s), read as the per-site "
                       "radial stress in the normalization of eps_tot (the stress of site n integrated over its shell: an extensive quantity per "
                       "site, not a density, so that rho_ell, p_ell and J_ell are commensurate; reading '%s', scale %s applied before the average), "
                       "by SmoothedSources5(P, P, R, ell, r_last_node, support_rel).rho, the operator that gives rho_ell from eps_tot (the scalar "
                       "kernel: the engine exposes no tensor kernel for a stress, and whether its vector kernel for J differs from the scalar one is "
                       "measured by p_kernel_check = max |.rho - .J| / max |p| of the same call, per record and at its maximum under Inputs). Checks "
                       "per record: int 4 pi r^2 p_ell dr / sum P - 1 (p_quadrature_rel; 0 expected for a per-site quantity, as the engine's "
                       "charge_conservation_quadrature is for eps_tot), sum P / sum eps_tot, p_ell/rho_ell at the root and max |p_ell| / max |rho_ell| "
                       "over the grid (|p_r| <= rho for matter under the dominant energy condition; a per-site array read as a density, or the "
                       "reverse, shows as a ratio of the order of the shell volume 4 pi R^2 h). --pr-reading density multiplies a stress stored per "
                       "unit volume by that shell volume before the average; --pr-scale -1 turns a stored stress sigma_rr into the pressure "
                       "p_r = -sigma_rr; --pr-key names another key; p_r at the root is the exact kernel value there (p_R_source)."
                       % (key, meta.get('note'), meta.get('reading'), _fmt(meta.get('scale'))))
    else:
        stress_text = ("p_r unavailable: %s. T_ll is not computed; rho_ell - 2 J_ell at the root (rho_minus_2J_R) and rho_ell + 2 J_ell are "
                       "printed in its place; the fields keys are listed under Inputs; --pr-key <key> names the radial stress when one of them "
                       "stores it (read then as eps_tot is: a per-site extensive quantity; --pr-reading density for a stress per unit volume)."
                       % (meta.get('note') or 'no stress pass'))
    return {
        'column_Tll': ("T_ll = rho_ell + p_ell - 2 J_ell at the outer dense root R of each rooted record (Tll_R; Tll_R_in at its inner zero), with "
                       "rho_ell and J_ell the exact kernel values the columns already read there (rho_R, J_R) and p_ell the kernel value of the "
                       "stress at R from the same kind of SmoothedSources5 object. J is the rows' J (J_ell = <P_r> = -J_r of the goal's display, "
                       "the module's DICTIONARY line), so this is the null-null component along the null direction that the sign convention of J "
                       "selects; rho_ell + p_ell + 2 J_ell (Tll_flipJ_R) is the component along the other null direction and is printed beside it, "
                       "as is rho_ell - 2 J_ell alone (rho_minus_2J_R, the part without the stress); 8 pi G R^2 T_ll is the dimensionless form "
                       "(Tll_dimless_R; the row formula's margin is 1 - 8 pi G R^2 (rho - J)). Per pair: the values at both roots (Tll_a, Tll_b), "
                       "their mean (Tll_mean, the midpoint rule in the record index) and the midpoint fields' value interpolated at R_mid "
                       "(Tll_mid). The profile across the root: T_ll on the dense grid (spacing %s) is stored in the npz (rec_Tll_profile with "
                       "rec_p_profile; the dense rho_ell and J_ell are in the dense cache), sampled by the local polynomial at R + (%s) ell "
                       "(Tll_at_offsets, with rho_at_offsets, p_at_offsets, J_at_offsets, Tll_flipJ_at_offsets) and on the window |r - R| <= %g ell "
                       "at the dense spacing (rec_Tll_window with rec_Tll_window_rel), where its minimum, maximum, the number of sign changes on "
                       "the grid and the sign change nearest R are listed (Tll_window_*)."
                       % (_fmt(h_fine), opts.Tll_offsets_ell, float(opts.Tll_window_ell))),
        'column_row_defect_roots': ("The row defect at a dense root is the pair (e, g) of the declared constraint rows of FIRSTLAW_rows_v1, "
                                    "e = M' - 4 pi r^2 (rho + u J) (the energy row; M = r u^2/(2G), u = v/N) and g = N'/N + 4 pi G r J/u (the "
                                    "momentum row), evaluated at every dense zero of N + v, inner and outer (root_defects; e_root, g_root at the "
                                    "outer root R; e_root_in, g_root_in at the inner zero R_in). These, and not the two source continuity residuals "
                                    "W_0, W_r (the other pair, which need dot rho, dot J and the stresses), are what that note calls the row "
                                    "defects: its applied identity retains exactly them (the -2Ge correction of kappa at the root and the integral "
                                    "of N(delta e + g delta M)), and the engine's constraint solver integrates these two rows and no evolution row. "
                                    "The values are read from the arrays e, g on the dense grid (the 4th-order stencil at spacing %s that the "
                                    "applied identity's defect integral uses) by the local polynomial of degree %d at the zero; the relative forms "
                                    "divide by the rows' own terms there, 4 pi z^2 (|rho| + |u J|) and |N'/N| + 4 pi G z |J/u|. Beside e(R): the "
                                    "same defect read from the two kappas already in the columns, e_root_from_kappa = -(kappa_fd - kappa_formula) "
                                    "R/(N_R G), which follows from (N + v)' = N(1 - 2GM')/(2R) at u = -1 with M' the Richardson derivative at "
                                    "h = ell/%g, and the applied identity's correction of kappa, -N_R G e(R)/R (kappa_corr_root), carried into "
                                    "column 2 as col2_defect_corr = (delta A/(8 pi G)) x its pair mean. Expected on the dense solution: the "
                                    "solver's tolerance." % (_fmt(h_fine), int(opts.interp_degree), 1.0 / float(opts.fd_h_frac))),
        'stress_normalization': stress_text,
    }


def v3_stress_line(res):
    meta = res.get('stress_meta') or {}
    if not meta.get('key'):
        return 'unavailable (%s)' % (meta.get('note') or 'no stress pass')
    if meta.get('companion'):
        return v3b_stress_line(meta)
    return ("key '%s' (%s), reading %s, scale %s, kernel %s; %s of %s records smoothed in %s s with %s workers; kernel check "
            "max |.rho - .J|/max|p| = %s; quadrature int 4 pi r^2 p_ell dr / sum P - 1: max |.| %s; sum P / sum eps_tot over all rows %s; "
            "tangential stress key %s (not used)" % (
                meta.get('key'), meta.get('note'), meta.get('reading'), _fmt(meta.get('scale')), meta.get('kernel'), meta.get('n_ok'),
                meta.get('n_records'), _fmt(meta.get('seconds'), 4), meta.get('workers'), _fmt(meta.get('kernel_check_max'), 3),
                _fmt(meta.get('quadrature_rel_max_abs'), 3), _fmt(meta.get('sum_P_over_sum_eps_all_rows'), 4), meta.get('pt_key') or 'none'))


def v3_md_summary(res):
    """The md section of the version-3 columns (definitions and statistics), placed before the closure statistics."""
    st = res.get('stats') or {}
    thr = res.get('thresholds') or {}
    df = res.get('definitions') or {}

    def s(key, p=3):
        return _fmt(st.get(key), p)
    L = ['', '## Version 3 columns: T_ll = rho + p_r - 2J at the roots and the row defect at every dense root']
    for key in ('column_Tll', 'column_row_defect_roots', 'stress_normalization'):
        L.append('- **%s**: %s' % (key, df.get(key)))
    L.append("- T_ll(R) at the outer root over the rooted re-solved records (%s with a value, %s of them negative): median %s, min %s, max %s; "
             "positive in a fraction %s; 8 pi G R^2 T_ll median %s (min %s, max %s); the other null direction rho + p_r + 2J: median %s "
             "(min %s, max %s), positive fraction %s; rho - 2J alone: median %s (min %s, max %s), positive fraction %s; p_r(R) median %s; "
             "p_r/rho at R median %s (min %s, max %s); max |p_r| / max |rho| over the grid, median %s" % (
                 st.get('n_Tll_R_finite'), st.get('n_Tll_R_negative'), s('Tll_R_median'), s('Tll_R_min'), s('Tll_R_max'), s('frac_Tll_R_positive'),
                 s('Tll_dimless_R_median'), s('Tll_dimless_R_min'), s('Tll_dimless_R_max'), s('Tll_flipJ_R_median'), s('Tll_flipJ_R_min'),
                 s('Tll_flipJ_R_max'), s('frac_Tll_flipJ_R_positive'), s('rho_minus_2J_R_median'), s('rho_minus_2J_R_min'), s('rho_minus_2J_R_max'),
                 s('frac_rho_minus_2J_R_positive'), s('p_R_median'), s('p_over_rho_R_median'), s('p_over_rho_R_min'), s('p_over_rho_R_max'),
                 s('p_max_abs_over_rho_max_abs_median')))
    L.extend(v3b_md_lines(res))
    L.append("- across the root, |r - R| <= %s ell on the dense grid: the window minimum of T_ll has median %s (min %s) and the maximum median %s "
             "(max %s); %s records have a sign change of T_ll in the window (nearest sign change minus R: median %s); at the inner zero (%s "
             "records): T_ll median %s (min %s, max %s), positive fraction %s" % (
                 thr.get('Tll_window_ell'), s('Tll_window_min_median'), s('Tll_window_min_min'), s('Tll_window_max_median'), s('Tll_window_max_max'),
                 st.get('n_Tll_window_with_zero_crossing'), s('Tll_window_nearest_zero_minus_R_median'), st.get('n_rooted_with_inner_zero'),
                 s('Tll_R_in_median'), s('Tll_R_in_min'), s('Tll_R_in_max'), s('frac_Tll_R_in_positive')))
    L.append("- the row defect at the outer root (e the energy row, g the momentum row, from the dense arrays at spacing %s): |e(R)| max %s, "
             "relative to 4 pi R^2 (|rho| + |uJ|): median |.| %s, max %s; |g(R)| max %s, relative to |N'/N| + 4 pi G R |J/u|: median |.| %s, "
             "max %s; at the inner zero: e relative median |.| %s (max %s), g relative median |.| %s (max %s); e(R) read from kappa_fd - "
             "kappa_formula against e(R) from the arrays: max |difference| %s; the applied identity's correction of kappa, -N_R G e(R)/R: max |.| %s "
             "(mean kappa_eq %s); carried into column 2 and summed over the included pairs: %s, median |.| per pair %s, relative to column 2 "
             "median |.| %s" % (
                 _fmt((res.get('grid') or {}).get('h_fine')), s('e_root_max_abs'), s('e_root_rel_abs_median'), s('e_root_rel_max_abs'), s('g_root_max_abs'),
                 s('g_root_rel_abs_median'), s('g_root_rel_max_abs'), s('e_root_in_rel_abs_median'), s('e_root_in_rel_max_abs'),
                 s('g_root_in_rel_abs_median'), s('g_root_in_rel_max_abs'), s('e_root_minus_from_kappa_max_abs'), s('kappa_corr_root_max_abs'),
                 s('kappa_eq_mean', 5), s('sum_col2_defect_corr'), s('col2_defect_corr_abs_median'), s('col2_defect_corr_over_col2_abs_median')))
    L.append("- per included pair: T_ll mean of the two roots median %s (min %s, max %s), positive in a fraction %s, of the same sign as delta A in "
             "a fraction %s; the midpoint fields' T_ll at R_mid median %s (positive fraction %s)" % (
                 s('Tll_mean_median'), s('Tll_mean_min'), s('Tll_mean_max'), s('frac_Tll_mean_positive'), s('frac_Tll_mean_same_sign_as_dA'),
                 s('Tll_mid_median'), s('frac_Tll_mid_positive')))
    return L


def v3_md_tables(res):
    """The two md tables of the version-3 columns, each repeating the three columns beside the new ones."""
    thr = res.get('thresholds') or {}
    offsets = _v3_offsets(thr.get('Tll_offsets_ell', ''))
    L = ['', '## Per-pair table, version 3 columns (beside the three columns)', '']
    L.append('| T_a | T_b | dE | k dA/8piG | W_ext | T_ll(a) | T_ll(b) | T_ll(mid) | T_ll+(mean) | rho-2J(mean) | p/rho(a) | p/rho(b) | e(R_a) | e(R_b) | '
             'g(R_a) | g(R_b) | e_rel(R_b) | g_rel(R_b) | col2 defect corr |')
    L.append('|' + '---|' * 19)
    for p in res.get('pairs') or []:
        cells = [_fmt(p.get('T_a')), _fmt(p.get('T_b')), _fmt(p.get('dE'), 5), _fmt(p.get('col2'), 5), _fmt(p.get('W_ext'), 5), _fmt(p.get('Tll_a'), 5),
                 _fmt(p.get('Tll_b'), 5), _fmt(p.get('Tll_mid'), 5), _fmt(p.get('Tll_flipJ_mean'), 5), _fmt(p.get('rho_minus_2J_mean'), 5),
                 _fmt(p.get('p_over_rho_a'), 4), _fmt(p.get('p_over_rho_b'), 4), _fmt(p.get('e_root_a'), 3), _fmt(p.get('e_root_b'), 3),
                 _fmt(p.get('g_root_a'), 3), _fmt(p.get('g_root_b'), 3), _fmt(p.get('e_root_rel_b'), 3), _fmt(p.get('g_root_rel_b'), 3),
                 _fmt(p.get('col2_defect_corr'), 3)]
        L.append('| %s |' % ' | '.join(cells))
    L += ['', '## Per-record table, version 3 columns (records with a stored root)', '']
    off_h = ' | '.join('T_ll(R%+g ell)' % o for o in offsets)
    L.append('| T | R_dense | rho(R) | p_r(R) | J(R) | T_ll(R) | T_ll+(R) | rho-2J(R) | p/rho | 8piGR^2 T_ll | %s | win min | win max | sign changes | '
             'nearest zero - R | R_in | T_ll(R_in) | e(R) | g(R) | e_rel(R) | g_rel(R) | e from kappa | kappa corr | e(R_in) | g(R_in) |' % off_h)
    L.append('|' + '---|' * (24 + len(offsets)))
    for r in res.get('records') or []:
        if not r.get('has_root'):
            continue
        offs = list(r.get('Tll_at_offsets') or [])
        offs += [NAN] * (len(offsets) - len(offs))
        cells = [_fmt(r.get('T_over_rs')), _fmt(r.get('R_dense'), 9), _fmt(r.get('rho_R'), 5), _fmt(r.get('p_R'), 5), _fmt(r.get('J_R'), 5),
                 _fmt(r.get('Tll_R'), 5), _fmt(r.get('Tll_flipJ_R'), 5), _fmt(r.get('rho_minus_2J_R'), 5), _fmt(r.get('p_over_rho_R'), 4),
                 _fmt(r.get('Tll_dimless_R'), 4)]
        cells += [_fmt(x, 4) for x in offs[:len(offsets)]]
        cells += [_fmt(r.get('Tll_window_min'), 4), _fmt(r.get('Tll_window_max'), 4), _fmt(r.get('Tll_window_n_zero_crossings'), 3),
                  _fmt(r.get('Tll_window_nearest_zero_minus_R'), 4), _fmt(r.get('R_in'), 6), _fmt(r.get('Tll_R_in'), 5), _fmt(r.get('e_root'), 3),
                  _fmt(r.get('g_root'), 3), _fmt(r.get('e_root_rel'), 3), _fmt(r.get('g_root_rel'), 3), _fmt(r.get('e_root_from_kappa'), 3),
                  _fmt(r.get('kappa_corr_root'), 3), _fmt(r.get('e_root_in'), 3), _fmt(r.get('g_root_in'), 3)]
        L.append('| %s |' % ' | '.join(cells))
    return L


def v3_output_arrays(res, r_fine):
    """The version-3 arrays of the output npz: the T_ll and p_ell profiles on the dense grid and the T_ll window around the outer root."""
    out = {}
    recs = res.get('records') or []
    n = int(np.asarray(r_fine).size)
    idx = [i for i, r in enumerate(recs) if isinstance(r.get('_Tll_profile'), np.ndarray)]
    if idx:
        prof = np.full((len(idx), n), np.nan)
        pp = np.full((len(idx), n), np.nan)
        for j, i in enumerate(idx):
            a = np.asarray(recs[i]['_Tll_profile'], float)
            prof[j, :min(n, a.size)] = a[:n]
            b = recs[i].get('_p_profile')
            if b is not None:
                b = np.asarray(b, float)
                pp[j, :min(n, b.size)] = b[:n]
        out['rec_profile_index'] = np.array(idx, int)
        out['rec_profile_T_over_rs'] = np.array([_float(recs[i].get('T_over_rs')) for i in idx], float)
        out['rec_Tll_profile'] = prof
        out['rec_p_profile'] = pp
    widx = [i for i, r in enumerate(recs) if isinstance(r.get('_Tll_window'), np.ndarray)]
    if widx:
        L = max(np.asarray(recs[i]['_Tll_window']).size for i in widx)
        win = np.full((len(widx), L), np.nan)
        for j, i in enumerate(widx):
            a = np.asarray(recs[i]['_Tll_window'], float)
            win[j, :a.size] = a
        out['rec_window_index'] = np.array(widx, int)
        out['rec_Tll_window'] = win
        out['rec_Tll_window_rel'] = np.asarray(recs[widx[0]]['_Tll_window_rel'], float)
    return out


def v3_add_arguments(p):
    p.add_argument('--pr-key', dest='pr_key', default=None,
                   help="the radial stress: a fields key (the per-site stress, averaged with the engine's kernel as in version 3), 'handoff:<key>' (the key in the "
                        "companion npz collapse_<tag>_handoff.npz beside the fields npz, already the engine's kernel average on r_nodes, carried to the dense grid "
                        "by a cubic spline), '<path>.npz:<key>' (the same in that npz), or a bare key that is in the companion when it is not among the fields "
                        "keys; default: found by name among the fields keys, then among the companion's (S_r_ell and the names below with _ell); 'none' disables "
                        "p_r. The names: P_r_tot, Pr_tot, p_r_tot, sigma_rr_tot, "
                        "S_rr_tot, T_rr_tot, stress_r_tot, P_rad_tot and the same without _tot; 'none' disables p_r)")
    p.add_argument('--pr-reading', dest='pr_reading', choices=('per-site', 'density'), default='per-site',
                   help="how the stored stress is read before the engine's kernel average: 'per-site' = the stress of site n integrated over its shell (the "
                        "reading of eps_tot: an energy per site, not a density); 'density' = a stress per unit volume at the site, multiplied by the shell "
                        "volume 4 pi R_n^2 h before the average")
    p.add_argument('--pr-scale', dest='pr_scale', type=float, default=1.0,
                   help='the stored stress is multiplied by this before the average (-1 if the stored quantity is the stress sigma_rr = -p_r, not the pressure)')
    p.add_argument('--Tll-window-ell', dest='Tll_window_ell', type=float, default=4.0,
                   help='half width, in units of ell, of the window around the outer root over which the T_ll profile is sampled and its sign changes counted')
    p.add_argument('--Tll-offsets-ell', dest='Tll_offsets_ell', default='-2,-1,-0.5,0,0.5,1,2',
                   help='comma list of offsets from the root, in units of ell, at which T_ll is printed in the md')
    p.add_argument('--seed-cache', dest='seed_cache', default=None,
                   help="a version-2 dense cache (firstlaw_dense_v2_<tag>_cache.npz) read before this run's own: its records are not re-solved (the dense "
                        "pass is unchanged); --seed-key-policy says how its key is treated")
    p.add_argument('--seed-key-policy', dest='seed_key_policy', choices=('any', 'strict'), default='any',
                   help="'any' (default): the seed is accepted when its grid is this run's grid and its records name this run's module and solver arguments "
                        "where they record them (the stored key also hashes the running script, so version 2's key never equals version 3's); 'strict': "
                        "the key must be equal")
    p.add_argument('--no-stress', dest='no_stress', action='store_true', help='skip the stress pass: T_ll is not computed (rho - 2J is still printed)')
    p.add_argument('--pt-key', dest='pt_key', default=None,
                   help="the tangential stress, in the forms of --pr-key (default: found beside the radial one when that is in the companion npz, S_t_ell); read at "
                        "the outer root and reported (pt_R, pt_over_p_R, pt_over_rho_R); not used in T_ll")
    p.add_argument('--pr-node-reading', dest='pr_node_reading', choices=('density', 'per-node'), default='density',
                   help="how a companion (node-grid) stress is read: 'density' = already the kernel average per unit volume (the engine's _ell fields; the "
                        "default); 'per-node' = an extensive quantity per node, divided by the node shell volume 4 pi r_n^2 h_node before use")
    p.add_argument('--pr-grid-key', dest='pr_grid_key', default='r_nodes',
                   help="the companion npz key of the node grid of its stress arrays (default r_nodes; else r, R, else any 1-D key of the matching length)")
    p.add_argument('--check-against', dest='check_against', default=None,
                   help="a version-2 json: after writing, every leaf of its records, pairs, excluded_pairs, stats, grid and matching must be present and equal "
                        "in this run's json (timings exempt); the result is printed as the BITWISE line and written to <base>_bitwise.json")


def v3_seed_cache(opts, own_key, r_fine, sargs, module_name):
    """A version-2 dense cache read before this run's own (the dense pass is unchanged, so its records are this run's).  The stored key
    hashes the engine files and the running script (import_engine lists __main__), so version 2's key never equals version 3's: under
    the policy 'any' the seed is accepted when its grid is this run's grid and its records name this run's module and solver arguments
    where they record them; 'strict' requires the key.  Returns ({row: dense dict}, message)."""
    path = getattr(opts, 'seed_cache', None)
    if not path:
        return {}, 'no seed cache'
    if getattr(opts, 'recompute', False):
        return {}, 'seed cache ignored (--recompute)'
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return {}, 'seed cache %s not found' % path
    policy = getattr(opts, 'seed_key_policy', 'any')
    try:
        z = np.load(path, allow_pickle=False)
        stored = str(np.asarray(z['key']).ravel()[0])
        if stored != own_key and policy == 'strict':
            return {}, 'seed cache key differs (strict policy): ignored'
        rf = np.asarray(z['r_fine'], float)
        if rf.shape != np.asarray(r_fine).shape or not np.allclose(rf, r_fine, rtol=0.0, atol=1e-12):
            return {}, 'seed cache grid differs: ignored'
        rows = np.asarray(z['rows'], int)
        scal = np.asarray(z['scalars'])
        out = {}
        bad = []
        for i, row in enumerate(rows):
            d = json.loads(str(scal[i]))
            eng = d.get('engine') or {}
            if eng.get('module') not in (None, module_name):
                bad.append('module %s' % eng.get('module'))
            for key in ('rtol', 'atol', 'support_rel', 'collar_policy'):
                if key in eng and key in sargs and eng.get(key) is not None:
                    try:
                        same = str(eng[key]) == str(sargs[key]) or float(eng[key]) == float(sargs[key])
                    except (TypeError, ValueError):
                        same = str(eng[key]) == str(sargs[key])
                    if not same:
                        bad.append('%s %s vs %s' % (key, eng[key], sargs[key]))
            arrays = {}
            for name in ARRAY_KEYS:
                akey = 'arr_' + name
                if akey in z.files:
                    a = np.asarray(z[akey][i], float)
                    if np.all(np.isnan(a)):
                        continue
                    arrays[name] = a
            d['arrays'] = arrays if d.get('ok') else {}
            out[int(row)] = d
        if bad:
            return {}, 'seed cache records disagree with this run (%s): ignored' % '; '.join(sorted(set(bad))[:4])
        return out, '%d records read (key %s, policy %s)' % (len(out), 'equal' if stored == own_key else 'differs, accepted', policy)
    except Exception as exc:
        return {}, 'seed cache unreadable (%s: %s): ignored' % (type(exc).__name__, exc)


def run_tasks(worker, tasks, workers, progress=None):
    """compute_dense's pool (fork, serial fallback) for another worker function."""
    results = {}
    t0 = time.time()
    n = len(tasks)
    if workers > 1 and n > 1 and hasattr(os, 'fork'):
        try:
            import multiprocessing
            ctx = multiprocessing.get_context('fork')
            with ctx.Pool(processes=min(workers, n)) as pool:
                for i, (k, d) in enumerate(pool.imap_unordered(worker, tasks, chunksize=1)):
                    results[k] = d
                    if progress:
                        progress(i, n, k, d)
            return results, time.time() - t0
        except Exception as exc:
            print('NOTE: the worker pool failed (%s: %s); the remaining tasks run serially' % (type(exc).__name__, exc))
    for i, task in enumerate(tasks):
        if task[0] in results:
            continue
        k, d = worker(task)
        results[k] = d
        if progress:
            progress(i, n, k, d)
    return results, time.time() - t0


def _v3_stress_worker(task):
    k, row = task
    try:
        d = smooth_stress_record(_G['FLD'], _G['R'], _G['P'][row], _G['ell'], _G['r_last_node'], _G['support_rel'], _G['r_fine'], _G['zeros'].get(k))
    except Exception as exc:
        d = {'ok': False, 'error': 'stress worker exception %s: %s' % (type(exc).__name__, exc), 'traceback': traceback.format_exc(), 'seconds': NAN, 'p': None}
    d['row'] = int(row)
    return k, d


def v3_stress_pass(FLD, F, opts, dense, rows, idx, r_fine, r_last_node, sargs, ell_pass, notes):
    """The stress pass: the radial stress of every record with dense arrays, smoothed with the engine's kernel (pool-parallel, no solves).
    Returns ({'meta': ..., 'records': {k: ...}}, seconds)."""
    t0 = time.time()
    meta = {'key': None, 'reading': str(opts.pr_reading), 'scale': float(opts.pr_scale), 'candidates': [], 'pt_key': None, 'note': None,
            'n_records': 0, 'n_ok': 0, 'seconds': 0.0, 'workers': WORKERS,
            'formula': 'T_ll = rho_ell + p_ell - 2 J_ell, J_ell the rows J (= <P_r> = -J_r of the goal display); rho_ell + p_ell + 2 J_ell beside it',
            'kernel': 'SmoothedSources5(P, P, R, ell, r_last_node, support_rel).rho: the scalar kernel average per unit volume that gives rho_ell from eps_tot'}
    empty = {'meta': meta, 'records': {}}
    if getattr(opts, 'no_stress', False):
        meta['note'] = 'skipped (--no-stress)'
        print('STRESS: skipped (--no-stress); T_ll not computed')
        return empty, 0.0
    loc = v3b_locate_stress(F, opts)
    if loc.get('companion'):
        return v3b_stress_pass_companion(FLD, F, opts, dense, rows, idx, r_fine, ell_pass, notes, meta, loc, t0)
    key, note, cands, pt_key = loc['key'], loc['note'], loc['candidates'], loc['pt_key']
    meta.update(key=key, note=note, candidates=cands, pt_key=pt_key)
    if key is None:
        notes.append('STRESS: %s; p_r unavailable, T_ll not computed (rho_ell - 2 J_ell printed in its place); --pr-key names the key when one of the '
                     'stored fields is the radial stress' % note)
        print('STRESS: %s; T_ll not computed' % note)
        return empty, 0.0
    R = np.asarray(F['R'], float)
    h_site = uniform_step(R, 1e-9) or float(np.median(np.diff(R)))
    try:
        P = read_stress(F['path'], key, R, opts.pr_reading, opts.pr_scale, h_site)
    except Exception as exc:
        meta['note'] = 'the stress could not be read (%s: %s)' % (type(exc).__name__, exc)
        notes.append('STRESS: ' + meta['note'])
        print('STRESS: ' + meta['note'])
        return empty, 0.0
    eps = np.asarray(F['eps_tot'], float)
    if P.shape != eps.shape:
        meta['note'] = "the stress key '%s' has shape %s against eps_tot %s: not read" % (key, list(P.shape), list(eps.shape))
        notes.append('STRESS: ' + meta['note'])
        print('STRESS: ' + meta['note'])
        return empty, 0.0
    meta['shape'] = list(P.shape)
    meta['sum_P_over_sum_eps_all_rows'] = float(P.sum() / eps.sum()) if eps.sum() != 0 else NAN
    tasks, zeros = [], {}
    for k in idx:
        d = dense.get(k)
        if d is None or not d.get('ok') or (d.get('arrays') or {}).get('rho') is None:
            continue
        tasks.append((k, int(rows[k])))
        zeros[k] = list(d.get('zeros_dense') or [])
    _G.update({'FLD': FLD, 'R': R, 'P': P, 'ell': ell_pass, 'r_last_node': float(r_last_node), 'support_rel': sargs.get('support_rel', 1e-14),
               'r_fine': np.asarray(r_fine, float), 'zeros': zeros})
    results, secs = run_tasks(_v3_stress_worker, tasks, WORKERS)
    meta['n_records'] = len(results)
    meta['n_ok'] = int(sum(1 for d in results.values() if d.get('ok')))
    meta['seconds'] = time.time() - t0
    errs = [d.get('error') for d in results.values() if not d.get('ok')]
    if errs:
        meta['first_error'] = errs[0]
        notes.append('STRESS: %d of %d records could not be smoothed (first: %s)' % (len(errs), len(results), errs[0]))
    kc = np.array([_float(d.get('kernel_check')) for d in results.values() if d.get('ok')], float)
    qr = np.array([_float(d.get('quadrature_rel')) for d in results.values() if d.get('ok')], float)
    meta['kernel_check_max'] = _nan(np.max, kc)
    meta['quadrature_rel_max_abs'] = _nan(np.max, np.abs(qr))
    print('STRESS: key %s (%s), reading %s, scale %g: %d records smoothed (%d ok) in %.1f s with %d workers; kernel check max %s; quadrature '
          'int 4 pi r^2 p / sum P - 1 max |.| %s; sum P / sum eps_tot %s' % (
              key, note, opts.pr_reading, float(opts.pr_scale), meta['n_records'], meta['n_ok'], meta['seconds'], WORKERS, _fmt(meta['kernel_check_max'], 3),
              _fmt(meta['quadrature_rel_max_abs'], 3), _fmt(meta['sum_P_over_sum_eps_all_rows'], 4)))
    return {'meta': meta, 'records': results}, meta['seconds']


def v3_console(res, stress_secs):
    st = res.get('stats') or {}

    def s(key, p=3):
        return _fmt(st.get(key), p)
    print('V3 STRESS: %s' % v3_stress_line(res))
    for line in v3b_console_lines(res):
        print(line)
    print('V3 T_ll = rho + p_r - 2J at the outer root: median %s (min %s, max %s), positive fraction %s of %s rooted records; rho + p_r + 2J median %s '
          '(positive fraction %s); rho - 2J median %s; p_r/rho at R median %s (min %s, max %s); per pair T_ll mean median %s, positive fraction %s, same '
          'sign as delta A %s; stress pass %.1f s' % (
              s('Tll_R_median'), s('Tll_R_min'), s('Tll_R_max'), s('frac_Tll_R_positive'), st.get('n_Tll_R_finite'), s('Tll_flipJ_R_median'),
              s('frac_Tll_flipJ_R_positive'), s('rho_minus_2J_R_median'), s('p_over_rho_R_median'), s('p_over_rho_R_min'), s('p_over_rho_R_max'),
              s('Tll_mean_median'), s('frac_Tll_mean_positive'), s('frac_Tll_mean_same_sign_as_dA'), float(stress_secs or 0.0)))
    print('V3 ROW DEFECT at the roots: outer |e| max %s (relative |.| median %s, max %s), |g| max %s (relative |.| median %s, max %s); inner e relative '
          '|.| median %s, g relative |.| median %s; e(R) from kappa_fd - kappa_formula: max |difference| %s; kappa correction -N_R G e/R max |.| %s, its '
          'column-2 effect summed %s' % (
              s('e_root_max_abs'), s('e_root_rel_abs_median'), s('e_root_rel_max_abs'), s('g_root_max_abs'), s('g_root_rel_abs_median'),
              s('g_root_rel_max_abs'), s('e_root_in_rel_abs_median'), s('g_root_in_rel_abs_median'), s('e_root_minus_from_kappa_max_abs'),
              s('kappa_corr_root_max_abs'), s('sum_col2_defect_corr')))


def compare_trees(ref, new, path='', exempt=V3_COMPARE_EXEMPT, out=None):
    """Every leaf of ref must be present and equal in new (new may carry more keys); keys in exempt and keys starting with '_' are not
    compared.  Accumulates into out = {'n': leaves compared, 'bad': [(path, ref value, new value)]} and returns it."""
    if out is None:
        out = {'n': 0, 'bad': []}
    if isinstance(ref, dict):
        if not isinstance(new, dict):
            out['bad'].append((path, 'dict', type(new).__name__))
            return out
        for key, val in ref.items():
            if key in exempt or str(key).startswith('_'):
                continue
            if key not in new:
                out['bad'].append((path + '/' + str(key), 'present', 'missing'))
                continue
            compare_trees(val, new[key], path + '/' + str(key), exempt, out)
        return out
    if isinstance(ref, (list, tuple)):
        if not isinstance(new, (list, tuple)) or len(new) != len(ref):
            out['bad'].append((path, 'list[%d]' % len(ref), ('list[%d]' % len(new)) if isinstance(new, (list, tuple)) else type(new).__name__))
            return out
        for i, (x, y) in enumerate(zip(ref, new)):
            compare_trees(x, y, path + '[%d]' % i, exempt, out)
        return out
    out['n'] += 1
    same = (ref == new) or (ref is None and new is None)
    if not same and isinstance(ref, float) and isinstance(new, float) and np.isnan(ref) and np.isnan(new):
        same = True
    if not same:
        out['bad'].append((path, ref, new))
    return out


def compare_json_files(ref_path, new_path, sections=V3_COMPARE_SECTIONS):
    """The sections of a reference json against another: (n leaves compared, [differences])."""
    with open(ref_path) as fh:
        ref = json.load(fh)
    with open(new_path) as fh:
        new = json.load(fh)
    res = {'n': 0, 'bad': []}
    for sec in sections:
        if sec not in ref:
            continue
        if sec not in new:
            res['bad'].append(('/' + sec, 'present', 'missing'))
            continue
        compare_trees(ref[sec], new[sec], '/' + sec, V3_COMPARE_EXEMPT, res)
    return res['n'], res['bad']


def v3_check_against(opts, new_json):
    path = getattr(opts, 'check_against', None)
    if not path:
        return None
    path = os.path.abspath(path)
    if not os.path.exists(path):
        print('BITWISE: --check-against %s not found' % path)
        return None
    n, bad, mnote = compare_json_files_matched(path, new_json)
    print('BITWISE against %s: %d leaves compared (timings exempt; records, pairs and excluded pairs matched by T: %s), %d differ%s' % (
        path, n, mnote, len(bad), '' if not bad else '; first: ' + '; '.join('%s: %r -> %r' % (p, a, b) for p, a, b in bad[:8])))
    with open(new_json[:-5] + '_bitwise.json', 'w') as fh:
        json.dump({'reference': path, 'new': new_json, 'n_compared': n, 'n_differ': len(bad), 'matching': mnote,
                   'first_differences': [[str(x) for x in b] for b in bad[:50]]}, fh, indent=1)
    return n, bad


def rel_defects(R, e, g, N, v, rho, J, Np, mask):
    """RMS defects relative to the rows' own terms over mask (version 1's normalization)."""
    base = FOUR_PI * R * R
    with np.errstate(divide='ignore', invalid='ignore'):
        u = v / N
        den_e = float(np.sqrt(np.nansum((base * (np.abs(rho) + np.abs(u * J)))[mask] ** 2))) if mask.any() else 0.0
        den_g = float(np.sqrt(np.nansum((Np / N)[mask] ** 2))) if mask.any() else 0.0
        rel_e = float(np.sqrt(np.nansum(e[mask] ** 2))) / den_e if den_e > 0 else NAN
        gg = np.where(np.isfinite(g), g, 0.0)
        rel_g = float(np.sqrt(np.sum(gg[mask] ** 2))) / den_g if den_g > 0 else NAN
    return rel_e, rel_g


# ----------------------------------------------------------------------------
# the engine: read-only import by path, the solve arguments and their provenance
# ----------------------------------------------------------------------------
def sha256_of(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while True:
            b = fh.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def import_engine(path, module_name):
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise SystemExit('--engine-path %s is not a directory' % path)
    sys.dont_write_bytecode = True
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        FLD = importlib.import_module(module_name)
    except Exception as exc:
        raise SystemExit('%s could not be imported from %s: %s: %s' % (module_name, path, type(exc).__name__, exc))
    missing = [n for n in ('solve_constraints_ivp', 'trapping_radius') if not hasattr(FLD, n)]
    if missing:
        raise SystemExit('%s at %s lacks %s' % (module_name, getattr(FLD, '__file__', '?'), ', '.join(missing)))
    files = {}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, '__file__', None)
        if f and os.path.abspath(f).startswith(path + os.sep):
            try:
                files[name] = {'path': os.path.abspath(f), 'bytes': int(os.path.getsize(f)), 'sha256': sha256_of(f)}
            except OSError:
                files[name] = {'path': os.path.abspath(f)}
    prov = {'path': path, 'module': module_name, 'module_file': os.path.abspath(FLD.__file__), 'files': files,
            'read_only': 'imported by sys.path insertion with sys.dont_write_bytecode set; nothing copied or edited',
            'dictionary': str(getattr(FLD, 'DICTIONARY', '')), 'numpy_version': np.__version__}
    try:
        import scipy
        prov['scipy_version'] = scipy.__version__
    except Exception:
        prov['scipy_version'] = None
    try:
        prov['solve_signature'] = str(inspect.signature(FLD.solve_constraints_ivp))
    except (TypeError, ValueError):
        prov['solve_signature'] = None
    return FLD, prov


def read_npz_string(z, key):
    if key not in z.files:
        return None
    try:
        return str(np.asarray(z[key]).ravel()[0])
    except Exception:
        return None


def parse_solver_string(s):
    out = {'raw': s, 'module': None, 'kv': {}}
    if not s:
        return out
    m = re.search(r'engine_fields_v\d+', s)
    if m:
        out['module'] = m.group(0)
    try:
        d = json.loads(s)
        if isinstance(d, dict):
            out['kv'] = {str(k): v for k, v in d.items()}
            if out['module'] is None and isinstance(d.get('module'), str):
                out['module'] = d['module']
            return out
    except Exception:
        pass
    for key in ('rtol', 'atol', 'collar_policy', 'support_rel', 'r0_frac', 'site_fields', 'tol', 'maxit', 'method'):
        m = re.search(r'\b' + key + r'\s*[=:]\s*["\']?([^,;)\s"\']+)', s)
        if m:
            out['kv'][key] = m.group(1)
    return out


ARG_ALIASES = {'rtol': ('rtol', 'ivp_rtol', 'solve_rtol'), 'atol': ('atol', 'ivp_atol', 'solve_atol'),
               'collar_policy': ('collar_policy',), 'support_rel': ('support_rel',), 'r0_frac': ('r0_frac',)}
ARG_CASTS = {'rtol': float, 'atol': float, 'collar_policy': str, 'support_rel': float, 'r0_frac': float}


def first_record_solver_dict(records):
    for rec in records:
        s = rec.get('solver')
        if isinstance(s, dict) and s:
            return s
    return {}


def resolve_engine_module(opts, rec_solver, solver_parsed):
    if opts.engine_module:
        return opts.engine_module, '--engine-module'
    m = rec_solver.get('module') if isinstance(rec_solver, dict) else None
    if isinstance(m, str) and re.match(r'engine_fields_v\d+$', m):
        return m, "the records' solver.module"
    if solver_parsed.get('module'):
        return solver_parsed['module'], "the npz 'solver' string"
    return 'engine_fields_v16', 'default (no module named in the records or the npz)'


def resolve_solver_args(FLD, run_args, rec_solver, solver_parsed, opts):
    try:
        params = inspect.signature(FLD.solve_constraints_ivp).parameters
    except (TypeError, ValueError):
        params = {}
    spec = {}
    for key, cast in ARG_CASTS.items():
        cli = getattr(opts, key, None)
        val, src = None, None
        if cli is not None:
            val, src = cast(cli), '--' + key.replace('_', '-')
        if val is None:
            for name, dct in (("the records' solver dict", rec_solver), ('the run args', run_args)):
                if not isinstance(dct, dict):
                    continue
                for alias in ARG_ALIASES[key]:
                    if dct.get(alias) is None:
                        continue
                    try:
                        val, src = cast(dct[alias]), "%s key '%s'" % (name, alias)
                    except (TypeError, ValueError):
                        continue
                    break
                if val is not None:
                    break
        if val is None and solver_parsed['kv'].get(key) is not None:
            try:
                val, src = cast(solver_parsed['kv'][key]), "the npz 'solver' string"
            except (TypeError, ValueError):
                pass
        if val is None and key == 'collar_policy':
            val, src = 'flag', ("default 'flag' (nowhere stated; the engine default 'fail' refuses a collar_margin deficit, which "
                                "the records carry; --collar-policy overrides)")
        if val is None and key in params and params[key].default is not inspect.Parameter.empty:
            val, src = params[key].default, 'engine default'
        if val is None:
            continue
        passed = (not params) or key in params
        spec[key] = {'value': val, 'source': src, 'passed': bool(passed)}
        if not passed:
            spec[key]['note'] = 'not an argument of this solve_constraints_ivp: not passed'
    sargs = {k: v['value'] for k, v in spec.items() if v['passed']}
    site_fields = None
    for name, dct in (("the records' solver dict", rec_solver), ('the run args', run_args)):
        if isinstance(dct, dict) and dct.get('site_fields') is not None:
            site_fields = (str(dct['site_fields']), name)
            break
    if site_fields is None and solver_parsed['kv'].get('site_fields') is not None:
        site_fields = (str(solver_parsed['kv']['site_fields']), "the npz 'solver' string")
    return sargs, spec, site_fields


def build_fine_grid(FLD, R, ell, fine):
    """The dense grid: spacing ell/fine from 0 to the engine's last node (coarse_grid(R, ell)[-1]); exact multiples."""
    R = np.asarray(R, float)
    note = None
    try:
        r_nodes = np.asarray(FLD.coarse_grid(R, ell), float)
        r_last = float(r_nodes[-1])
    except Exception as exc:
        r_last = float(R[-1])
        note = 'coarse_grid unavailable (%s: %s): the dense grid ends at the last site %s' % (type(exc).__name__, exc, _fmt(r_last))
    h = float(ell) / float(fine)
    n = int(round(r_last / h))
    r_fine = np.arange(n + 1, dtype=float) * h
    return r_fine, r_last, note


# ----------------------------------------------------------------------------
# the dense solution: evaluations without lattice differences
# ----------------------------------------------------------------------------
def _ev(fn, x):
    """fn at the array x with x's shape (a (1, n) dense output or a constant is reshaped / broadcast)."""
    x = np.asarray(x, float)
    y = np.asarray(fn(x), float)
    if y.shape == x.shape:
        return y
    if y.ndim == 2 and y.shape[0] == 1 and y.shape[1] == x.size:
        return y[0].reshape(x.shape)
    if y.size == 1:
        return np.full(x.shape, float(y.ravel()[0]))
    return y.reshape(x.shape)


def richardson_d1(fn, x, h):
    """The engine's residual rule: (4 D(h/2) - D(h))/3 of the centred difference D of a dense function."""
    x = np.asarray(x, float)

    def cd(hh):
        return (_ev(fn, x + hh) - _ev(fn, x - hh)) / (2.0 * hh)
    return (4.0 * cd(0.5 * h) - cd(h)) / 3.0


def richardson_d2(fn, x, h):
    x = np.asarray(x, float)
    f0 = _ev(fn, x)

    def c2(hh):
        return (_ev(fn, x + hh) - 2.0 * f0 + _ev(fn, x - hh)) / (hh * hh)
    return (4.0 * c2(0.5 * h) - c2(h)) / 3.0


def slope_from_equations(r, N, v, W, rho, J, g):
    """d_r(N + v) of the dense solution from the equations it integrates (no difference quotient):
    v = -sqrt(W/r) and W' = 8 pi g r^2 N^2 rho_ell give v' = (W' - W/r)/(2 r v) where W > 0 (v' = 0 where W <= 0, where
    the dense v is 0); (ln N)' = -4 pi g r J_ell/q with q = v/N gives N' = -4 pi g r J N^2/v (0 where v = 0: the
    continued gauge's L' = 0).  Returns (N' + v', N', v')."""
    r = np.asarray(r, float)
    N = np.asarray(N, float)
    v = np.asarray(v, float)
    W = np.asarray(W, float)
    rho = np.asarray(rho, float)
    J = np.asarray(J, float)
    Wp = 8.0 * np.pi * g * r * r * N * N * rho
    ok = (W > 0.0) & (v != 0.0) & (r > 0.0)
    rs = np.where(r > 0.0, r, 1.0)
    vs = np.where(v != 0.0, v, 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        vp = np.where(ok, (Wp - W / rs) / (2.0 * rs * vs), 0.0)
        Np = np.where(v != 0.0, -FOUR_PI * g * r * J * N * N / vs, 0.0)
    return Np + vp, Np, vp


def dense_zero_scan(Nf, vf, r_lo, r_hi, step, xtol=1e-12):
    """An independent scan of the zeros of the dense N + v: sign changes on a grid of the given step refined by brentq."""
    from scipy.optimize import brentq
    n = int(np.floor((r_hi - r_lo) / step + 1e-9))
    if n < 1:
        return []
    rr = r_lo + np.arange(n + 1, dtype=float) * step
    F = _ev(Nf, rr) + _ev(vf, rr)

    def f1(x):
        xa = np.array([x])
        return float(_ev(Nf, xa)[0] + _ev(vf, xa)[0])
    zeros = []
    for i in np.flatnonzero(F[:-1] * F[1:] < 0.0):
        try:
            zeros.append(float(brentq(f1, rr[i], rr[i + 1], xtol=xtol, rtol=1e-14, maxiter=200)))
        except (ValueError, RuntimeError):
            zeros.append(float(rr[i] - F[i] * (rr[i + 1] - rr[i]) / (F[i + 1] - F[i])))
    for i in np.flatnonzero(F == 0.0):
        zeros.append(float(rr[i]))
    return sorted(zeros)


def root_quantities(R0, Nf, vf, Wf, rhof, Jf, g, ell, h_fd):
    """Everything the identity reads at a dense zero R0: the fields, the smoothed sources, the slope by the equations, the
    row formula, the Richardson difference, the second derivative (the kernel-shift prediction)."""
    x = np.array([float(R0)])
    N_R = float(_ev(Nf, x)[0])
    v_R = float(_ev(vf, x)[0])
    W_R = float(_ev(Wf, x)[0])
    rho_R = float(_ev(rhof, x)[0])
    J_R = float(_ev(Jf, x)[0])
    k_eq, Np, vp = slope_from_equations(x, N_R, v_R, W_R, rho_R, J_R, g)
    k_eq, Np, vp = float(k_eq[0]), float(Np[0]), float(vp[0])
    margin = 1.0 - 8.0 * np.pi * g * R0 * R0 * (rho_R - J_R)
    k_formula = N_R / (2.0 * R0) * margin

    def f(y):
        return _ev(Nf, y) + _ev(vf, y)
    k_fd = float(richardson_d1(f, x, h_fd)[0])
    k_fd2 = float(richardson_d1(f, x, 2.0 * h_fd)[0])
    fpp = float(richardson_d2(f, x, ell / 16.0)[0])
    margin_eq = 2.0 * R0 * k_eq / N_R if N_R != 0.0 else NAN
    shift3 = -(ell * ell / 2.0) * (fpp / k_eq + 2.0 / R0) if k_eq != 0.0 else NAN
    shift1 = -(ell * ell / 2.0) * (fpp / k_eq) if k_eq != 0.0 else NAN
    return {'R': float(R0), 'N_R': N_R, 'v_R': v_R, 'q_R': v_R / N_R if N_R != 0.0 else NAN, 'W_R': W_R, 'rho_R': rho_R, 'J_R': J_R,
            'Nplusv_R': N_R + v_R, 'kappa_eq': k_eq, 'Np_R': Np, 'vp_R': vp, 'kappa_formula': k_formula, 'kappa_fd': k_fd,
            'kappa_fd2': k_fd2, 'fpp_R': fpp, 'margin': margin, 'margin_eq': margin_eq, 'shift_pred_3d': shift3,
            'shift_pred_1d': shift1, 'h_fd': float(h_fd)}


def dense_row_audit(r, Nf, vf, Wf, rhof, Jf, g, h):
    """The two rows on the dense functions at the points r, derivatives by the Richardson centred difference at h:
    e = M' - 4 pi r^2 (rho + u J), M = r u^2/(2g), u = v/N;  g_row = N'/N + 4 pi g r J/u; the same with J -> -J for
    contrast; and the engine's W form W' - 8 pi g r^2 N^2 rho.  RMS relative to the rows' own terms, and maxima."""
    r = np.asarray(r, float)
    out = {'n': int(r.size), 'h': float(h)}
    if r.size < 4:
        out.update({'rel_e': NAN, 'rel_g': NAN, 'rel_e_flipJ': NAN, 'rel_g_flipJ': NAN, 'rel_W': NAN, 'max_e_rel': NAN,
                    'max_g_rel': NAN, 'max_abs_g': NAN, 'uJ_over_rho_rms': NAN})
        return out
    N = _ev(Nf, r)
    v = _ev(vf, r)
    rho = _ev(rhof, r)
    J = _ev(Jf, r)

    def Mfun(y):
        return y * (_ev(vf, y) / _ev(Nf, y)) ** 2 / (2.0 * g)
    Mp = richardson_d1(Mfun, r, h)
    Np = richardson_d1(Nf, r, h)
    Wp = richardson_d1(Wf, r, h)
    base = FOUR_PI * r * r
    with np.errstate(divide='ignore', invalid='ignore'):
        u = v / N
        e = Mp - base * (rho + u * J)
        e_flip = Mp - base * (rho - u * J)
        gr = Np / N + FOUR_PI * g * r * J / u
        gr_flip = Np / N - FOUR_PI * g * r * J / u
        termW = 8.0 * np.pi * g * r * r * N * N * rho
    den_e = float(np.sqrt(np.sum((base * (np.abs(rho) + np.abs(u * J))) ** 2)))
    den_g = float(np.sqrt(np.sum((Np / N) ** 2)))
    den_W = float(np.sqrt(np.sum(termW ** 2)))
    gr0 = np.where(np.isfinite(gr), gr, 0.0)
    grf = np.where(np.isfinite(gr_flip), gr_flip, 0.0)
    out['rel_e'] = float(np.sqrt(np.sum(e ** 2))) / den_e if den_e > 0 else NAN
    out['rel_e_flipJ'] = float(np.sqrt(np.sum(e_flip ** 2))) / den_e if den_e > 0 else NAN
    out['rel_g'] = float(np.sqrt(np.sum(gr0 ** 2))) / den_g if den_g > 0 else NAN
    out['rel_g_flipJ'] = float(np.sqrt(np.sum(grf ** 2))) / den_g if den_g > 0 else NAN
    out['rel_W'] = float(np.sqrt(np.sum((Wp - termW) ** 2))) / den_W if den_W > 0 else NAN
    me = float(np.max(base * (np.abs(rho) + np.abs(u * J))))
    out['max_e_rel'] = float(np.max(np.abs(e))) / me if me > 0 else NAN
    mg = float(np.max(np.abs(Np / N)))
    out['max_g_rel'] = float(np.max(np.abs(gr0))) / mg if mg > 0 else NAN
    out['max_abs_g'] = float(np.max(np.abs(gr0)))
    out['max_abs_e'] = float(np.max(np.abs(e)))
    dr = float(np.sqrt(np.sum((base * rho) ** 2)))
    out['uJ_over_rho_rms'] = float(np.sqrt(np.sum((base * u * J) ** 2))) / dr if dr > 0 else NAN
    return out


def engine_scalars(info):
    keys = ('module', 'it', 'converged', 'failed', 'r_collar', 'R_last', 'collar_margin', 'collar_policy', 'support_rel', 'rtol',
            'atol', 'site_fields', 'source_spline_error', 'charge_conservation_quadrature', 'residual_W', 'residual_invN',
            'residual_matching_region', 'residual_ladder', 'E_collar', 'N_min', 'interp_floor_N', 'interp_floor_v', 'centre_flag',
            'centre_series_check', 'centre_series_ratio', 'W_negative', 'W_roundoff_negative_points', 'W_tail', 'W_aux_signed_label')
    out = {}
    for key in keys:
        if key in info:
            out[key] = _jsonable(info[key])
    gauge = info.get('gauge') or {}
    out['gauge'] = {k: _jsonable(gauge.get(k)) for k in ('label', 'intervals', 'min_U_over_U_ref', 'delta_L_bound', 'lapse_unresolved',
                                                         'segments', 'zeros', 'stretches', 'near_tangencies') if k in gauge}
    regime = info.get('regime') or {}
    out['regime'] = {k: _jsonable(regime.get(k)) for k in ('label', 'max_ratio', 'r_at_max', 'intervals_above_1') if k in regime}
    ivp = info.get('ivp') or {}
    out['ivp'] = {k: _jsonable(ivp.get(k)) for k in ('r0', 'nfev', 'segments', 'N_centre', 'lnN_collar') if k in ivp}
    out['n_nodes'] = int(len(info['r_nodes'])) if 'r_nodes' in info else None
    return out


def site_comparisons(R, N_s, v_s, N_site, v_site, N_app, v_app, eps, J, rhof, Jf, g, r_c, R0, cfg, kernel=None):
    """The stored site and applied arrays against the dense solution at the sites: the maxima of the differences, the
    refit roots of N + v (version 1's reading) and the row defects at the sites with the smoothed sources, plus version 1's
    raw reading (eps_tot, J_tot as densities) for the record."""
    out = {}
    R = np.asarray(R, float)
    N_s = np.asarray(N_s, float)
    v_s = np.asarray(v_s, float)
    inner = R < r_c
    h_site = uniform_step(R, 1e-9)
    rho_s = _ev(rhof, R)
    J_s = _ev(Jf, R)
    sets = [('dense_at_sites', N_s, v_s)]
    if N_site is not None and v_site is not None:
        N_site = np.asarray(N_site, float)
        v_site = np.asarray(v_site, float)
        sets.append(('site', N_site, v_site))
        out['max_abs_N_site_minus_dense'] = float(np.max(np.abs(N_site - N_s)[inner])) if inner.any() else NAN
        out['max_abs_v_site_minus_dense'] = float(np.max(np.abs(v_site - v_s)[inner])) if inner.any() else NAN
    if N_app is not None and v_app is not None:
        N_app = np.asarray(N_app, float)
        v_app = np.asarray(v_app, float)
        sets.append(('applied', N_app, v_app))
        out['max_abs_N_applied_minus_dense'] = float(np.max(np.abs(N_app - N_s)[inner])) if inner.any() else NAN
        out['max_abs_v_applied_minus_dense'] = float(np.max(np.abs(v_app - v_s)[inner])) if inner.any() else NAN
    if N_site is not None and N_app is not None:
        out['max_abs_N_site_minus_applied'] = float(np.max(np.abs(N_site - N_app)))
        out['max_abs_v_site_minus_applied'] = float(np.max(np.abs(v_site - v_app)))
    if kernel is not None:
        Nk, vk = np.asarray(kernel[0], float), np.asarray(kernel[1], float)
        sets.append(('kernel', Nk, vk))
        if N_site is not None:
            out['max_abs_N_site_minus_kernel'] = float(np.max(np.abs(N_site - Nk)[inner])) if inner.any() else NAN
            out['max_abs_v_site_minus_kernel'] = float(np.max(np.abs(v_site - vk)[inner])) if inner.any() else NAN
        if N_app is not None:
            out['max_abs_N_applied_minus_kernel'] = float(np.max(np.abs(N_app - Nk)[inner])) if inner.any() else NAN
            out['max_abs_v_applied_minus_kernel'] = float(np.max(np.abs(v_app - vk)[inner])) if inner.any() else NAN
        out['max_abs_N_kernel_minus_dense'] = float(np.max(np.abs(Nk - N_s)[inner])) if inner.any() else NAN
        out['max_abs_v_kernel_minus_dense'] = float(np.max(np.abs(vk - v_s)[inner])) if inner.any() else NAN
    lo = R0 if (R0 is not None and np.isfinite(R0)) else cfg['r_min_zero']
    hs = h_site if h_site is not None else float(np.median(np.diff(R)))
    mask = (R > lo) & (R < r_c - 2.0 * hs)
    for label, NN, vv in sets:
        zs = classify_zeros(R, NN + vv, cfg['r_min_zero'], cfg['interp_degree'])
        outer = max(zs, key=lambda z: z['r']) if zs else None
        out[label + '_root_refit'] = outer['r'] if outer else NAN
        out[label + '_slope_refit'] = outer['slope'] if outer else NAN
        out[label + '_n_zeros_refit'] = len(zs)
        e, gg, M, Mp, Np = row_defects(R, NN, vv, rho_s, J_s, g, h_site)
        re_, rg_ = rel_defects(R, e, gg, NN, vv, rho_s, J_s, Np, mask)
        out[label + '_row_e_rel_ext'] = re_
        out[label + '_row_g_rel_ext'] = rg_
    if N_site is not None and eps is not None and J is not None:
        eps = np.asarray(eps, float)
        J = np.asarray(J, float)
        for lab, sgn in (('', 1.0), ('_flipJ', -1.0)):
            e, gg, M, Mp, Np = row_defects(R, N_site, v_site, eps, sgn * J, g, h_site)
            re_, rg_ = rel_defects(R, e, gg, N_site, v_site, eps, sgn * J, Np, mask)
            out['site_row_e_rel_ext_raw_reading' + lab] = re_
            out['site_row_g_rel_ext_raw_reading' + lab] = rg_
    return out


def solve_record_dense(FLD, R, eps, J, g, ell, sargs, r_fine, cfg, eps_p=None, J_p=None, N_site=None, v_site=None,
                       N_app=None, v_app=None):
    """One record's dense solution with the engine's own solver and everything the identity reads from it.  cfg keys:
    r_min_zero, fd_h_frac, scan_frac, check_applied, pairs_split, interp_degree, r_last_node, tangency_tol_abs."""
    t0 = time.time()
    R = np.asarray(R, float)
    eps = np.asarray(eps, float)
    J = np.asarray(J, float)
    out = {'ok': False, 'failed': None, 'seconds': NAN, 'arrays': {}}
    nz = (eps != 0.0) | (J != 0.0)
    out['sources'] = {'sum_eps_tot': float(eps.sum()), 'sum_abs_J_tot': float(np.abs(J).sum()), 'n_sites_nonzero': int(nz.sum()),
                      'R_last_nonzero': float(R[nz].max()) if nz.any() else NAN}
    try:
        res = FLD.solve_constraints_ivp(eps, J, g, R, ell, **sargs)
    except Exception as exc:
        out['failed'] = "the engine's solve_constraints_ivp raised %s: %s" % (type(exc).__name__, exc)
        out['traceback'] = traceback.format_exc()
        out['seconds'] = time.time() - t0
        return out
    try:
        N_s, v_s, W_nodes, info = res
    except (TypeError, ValueError):
        out['failed'] = 'solve_constraints_ivp returned %s (4 values expected)' % type(res).__name__
        out['seconds'] = time.time() - t0
        return out
    out['engine'] = engine_scalars(info)
    if N_s is None or info.get('failed'):
        out['failed'] = "the engine's solve failed: %s" % (info.get('failed'),)
        out['seconds'] = time.time() - t0
        return out
    for key in ('N_dense', 'v_dense', 'W_field', 'rho_ell', 'J_ell', 'r_collar', 'E_collar'):
        if key not in info:
            out['failed'] = "the solve's info lacks '%s'" % key
            out['seconds'] = time.time() - t0
            return out
    Nf, vf, Wf, rhof, Jf = info['N_dense'], info['v_dense'], info['W_field'], info['rho_ell'], info['J_ell']
    r_c = float(info['r_collar'])
    out['E_collar'] = float(info['E_collar'])
    out['r_collar'] = r_c
    h_fd = float(ell) * float(cfg['fd_h_frac'])
    N_s = np.asarray(N_s, float)
    v_s = np.asarray(v_s, float)

    # the zeros: the engine's own call, and an independent scan
    zeros_eng = None
    try:
        tz = FLD.trapping_radius(N_s, v_s, R, info)
        if isinstance(tz, dict):
            zeros_eng = sorted(float(x) for x in (tz.get('zeros') or []))
            out['tangencies_engine'] = [float(x) for x in (tz.get('tangencies') or [])]
            out['min_Nmv_engine'] = _float(tz.get('min'))
            out['r_min_Nmv_engine'] = _float(tz.get('r_min'))
            out['trapping_extra'] = {k: _jsonable(v) for k, v in tz.items()
                                     if k not in ('zeros', 'tangencies', 'min', 'r_min') and isinstance(v, (int, float, np.integer, np.floating))}
        else:
            out['engine_zeros_error'] = 'trapping_radius returned %s, not a dict' % type(tz).__name__
    except Exception as exc:
        out['engine_zeros_error'] = '%s: %s' % (type(exc).__name__, exc)
    try:
        zeros_scan = dense_zero_scan(Nf, vf, float(cfg['r_min_zero']), r_c, float(ell) * float(cfg['scan_frac']))
    except Exception as exc:
        zeros_scan = None
        out['scan_error'] = '%s: %s' % (type(exc).__name__, exc)
    out['zeros_engine'] = zeros_eng
    out['zeros_scan'] = zeros_scan
    if zeros_eng is not None:
        zeros, src = zeros_eng, "the engine's trapping_radius"
    elif zeros_scan is not None:
        zeros, src = zeros_scan, 'the own scan (engine zeros unavailable)'
    else:
        zeros, src = [], 'none (both unavailable)'
    out['zeros_dense'] = zeros
    out['zeros_source'] = src
    out.setdefault('tangencies_engine', [])

    def fsum(y):
        return _ev(Nf, y) + _ev(vf, y)
    details = []
    for z in zeros:
        x = np.array([z])
        N_z, v_z, W_z, rho_z, J_z = (float(_ev(fn, x)[0]) for fn in (Nf, vf, Wf, rhof, Jf))
        s_eq, Np_z, vp_z = slope_from_equations(x, N_z, v_z, W_z, rho_z, J_z, g)
        s_fd = float(richardson_d1(fsum, x, h_fd)[0])
        details.append({'r': float(z), 'slope_eq': float(s_eq[0]), 'slope_fd': s_fd, 'type': 'outer' if float(s_eq[0]) > 0 else 'inner',
                        'N': N_z, 'v': v_z, 'Nplusv': N_z + v_z, 'rho': rho_z, 'J': J_z})
    out['zeros_detail'] = details
    root = None
    if zeros:
        try:
            root = root_quantities(max(zeros), Nf, vf, Wf, rhof, Jf, g, ell, h_fd)
        except Exception as exc:
            out['root_error'] = '%s: %s' % (type(exc).__name__, exc)
    out['root'] = root

    # the dense functions on the common grid
    arrays = {'N': _ev(Nf, r_fine), 'v': _ev(vf, r_fine), 'rho': _ev(rhof, r_fine), 'J': _ev(Jf, r_fine),
              'N_sites': N_s.copy(), 'v_sites': v_s.copy()}

    # the row audit in the exterior of the outer zero, on the dense functions
    lo = (max(zeros) if zeros else float(cfg['r_min_zero'])) + 2.0 * h_fd
    mask = (r_fine > lo) & (r_fine < r_c - 2.0 * h_fd)
    try:
        out['audit'] = dense_row_audit(r_fine[mask], Nf, vf, Wf, rhof, Jf, g, h_fd)
        out['audit_2h'] = dense_row_audit(r_fine[mask], Nf, vf, Wf, rhof, Jf, g, 2.0 * h_fd)
    except Exception as exc:
        out['audit'] = {'error': '%s: %s' % (type(exc).__name__, exc)}
        out['audit_2h'] = {}

    # the stored site and applied arrays against the dense solution
    kernel = None
    if cfg.get('check_applied') and hasattr(FLD, 'site_fields_kernel_average'):
        try:
            Nk, vk, norms = FLD.site_fields_kernel_average(info, R, ell)
            kernel = (np.asarray(Nk, float), np.asarray(vk, float))
            arrays['N_kernel'] = kernel[0].copy()
            arrays['v_kernel'] = kernel[1].copy()
            out['kernel_norms'] = {'min': float(np.min(norms)), 'max': float(np.max(norms))}
        except Exception as exc:
            out['kernel_average_error'] = '%s: %s' % (type(exc).__name__, exc)
    try:
        out['sites'] = site_comparisons(R, N_s, v_s, N_site, v_site, N_app, v_app, eps, J, rhof, Jf, g, r_c,
                                        root['R'] if root else None, cfg, kernel=kernel)
    except Exception as exc:
        out['sites'] = {'error': '%s: %s' % (type(exc).__name__, exc)}

    # the pairs' part of the sources, smoothed with the engine's own kernels
    if cfg.get('pairs_split') and eps_p is not None and J_p is not None and hasattr(FLD, 'SmoothedSources5'):
        try:
            src_p = FLD.SmoothedSources5(np.asarray(eps_p, float), np.asarray(J_p, float), R, ell, float(cfg['r_last_node']),
                                         sargs.get('support_rel', 1e-14))
            arrays['rho_p'] = _ev(src_p.rho, r_fine)
            arrays['J_p'] = _ev(src_p.J, r_fine)
        except Exception as exc:
            out['pairs_split_error'] = '%s: %s' % (type(exc).__name__, exc)
    out['arrays'] = arrays
    out['ok'] = True
    out['seconds'] = time.time() - t0
    return out


# ----------------------------------------------------------------------------
# the workers and the cache
# ----------------------------------------------------------------------------
_G = {}


def _worker(task):
    k, row = task
    try:
        F = _G['F']

        def pick(key):
            a = F.get(key)
            return None if a is None else a[row]
        d = solve_record_dense(_G['FLD'], F['R'], F['eps_tot'][row], F['J_tot'][row], _G['g'], _G['ell'], _G['sargs'], _G['r_fine'],
                               _G['cfg'], eps_p=pick('eps_pairs'), J_p=pick('J_pairs'), N_site=pick('N_site'), v_site=pick('v_site'),
                               N_app=pick('N_applied'), v_app=pick('v_applied'))
    except Exception as exc:
        d = {'ok': False, 'failed': 'worker exception %s: %s' % (type(exc).__name__, exc), 'traceback': traceback.format_exc(),
             'seconds': NAN, 'arrays': {}}
    d['row'] = int(row)
    return k, d


def compute_dense(FLD, F, g, ell, sargs, r_fine, cfg, tasks, workers, progress=None):
    """tasks: list of (k, row).  Returns ({k: dense dict}, seconds)."""
    _G.update({'FLD': FLD, 'F': F, 'g': g, 'ell': ell, 'sargs': sargs, 'r_fine': r_fine, 'cfg': cfg})
    results = {}
    t0 = time.time()
    n = len(tasks)
    if workers > 1 and n > 1 and hasattr(os, 'fork'):
        try:
            import multiprocessing
            ctx = multiprocessing.get_context('fork')
            with ctx.Pool(processes=min(workers, n)) as pool:
                for i, (k, d) in enumerate(pool.imap_unordered(_worker, tasks, chunksize=1)):
                    results[k] = d
                    if progress:
                        progress(i, n, k, d)
            return results, time.time() - t0
        except Exception as exc:
            print('NOTE: the worker pool failed (%s: %s); the remaining records run serially' % (type(exc).__name__, exc))
    for i, task in enumerate(tasks):
        if task[0] in results:
            continue
        k, d = _worker(task)
        results[k] = d
        if progress:
            progress(i, n, k, d)
    return results, time.time() - t0


def cache_key_of(sargs, engine_prov, cfg, g, ell):
    payload = {'sargs': sargs, 'module': engine_prov.get('module'),
               'files': {k: v.get('sha256') for k, v in (engine_prov.get('files') or {}).items()},
               'h_fine': cfg['h_fine'], 'r_min_zero': cfg['r_min_zero'], 'fd_h_frac': cfg['fd_h_frac'], 'scan_frac': cfg['scan_frac'],
               'pairs_split': cfg['pairs_split'], 'check_applied': cfg['check_applied'], 'interp_degree': cfg['interp_degree'],
               'g': g, 'ell': ell, 'version': 2}
    return hashlib.sha256(json.dumps(_jsonable(payload), sort_keys=True).encode()).hexdigest()


def load_cache(path, key, r_fine):
    if not path or not os.path.exists(path):
        return {}, 'no cache file'
    try:
        z = np.load(path, allow_pickle=False)
        if str(np.asarray(z['key']).ravel()[0]) != key:
            return {}, 'cache key differs (other solver arguments, engine files or grid): ignored'
        rf = np.asarray(z['r_fine'], float)
        if rf.shape != r_fine.shape or not np.allclose(rf, r_fine, rtol=0.0, atol=1e-12):
            return {}, 'cache grid differs: ignored'
        rows = np.asarray(z['rows'], int)
        scal = np.asarray(z['scalars'])
        out = {}
        for i, row in enumerate(rows):
            d = json.loads(str(scal[i]))
            arrays = {}
            for name in ARRAY_KEYS:
                akey = 'arr_' + name
                if akey in z.files:
                    a = np.asarray(z[akey][i], float)
                    if np.all(np.isnan(a)):
                        continue
                    arrays[name] = a
            d['arrays'] = arrays if d.get('ok') else {}
            out[int(row)] = d
        return out, '%d records read from the cache' % len(out)
    except Exception as exc:
        return {}, 'cache unreadable (%s: %s): ignored' % (type(exc).__name__, exc)


def save_cache(path, key, r_fine, dense_by_row):
    rows = sorted(dense_by_row)
    scal = []
    cols = {name: [] for name in ARRAY_KEYS}
    for row in rows:
        d = dense_by_row[row]
        scal.append(json.dumps(_jsonable({k: v for k, v in d.items() if k != 'arrays'})))
        a = d.get('arrays') or {}
        for name in ARRAY_KEYS:
            cols[name].append(None if a.get(name) is None else np.asarray(a[name], float))
    data = {'key': np.array(key), 'r_fine': np.asarray(r_fine, float), 'rows': np.array(rows, int), 'scalars': np.array(scal)}
    for name in ARRAY_KEYS:
        present = [x for x in cols[name] if x is not None]
        if not present:
            continue
        L = present[0].size
        data['arr_' + name] = np.array([x if (x is not None and x.size == L) else np.full(L, np.nan) for x in cols[name]], float)
    tmp = path + '.tmp.npz'
    np.savez(tmp, **data)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------
# options
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description='FIRSTLAW_records_v3b: the first-law identity between consecutive rooted records, on the dense solution re-solved with the engine, plus T_ll = rho + p_r - 2J (p_r from the fields npz or from a companion npz such as the handoff, --pr-key handoff:S_r_ell) and the row defect at every dense root; the bitwise check against a version-2 json matched by T')
    p.add_argument('--json', default=None, help='collapse_<tag>.json (the record history)')
    p.add_argument('--npz', default=None, help='collapse_<tag>_fields.npz (default: --json with .json replaced by _fields.npz)')
    p.add_argument('--out', default=None, help='output directory (default: <dir of --json>/firstlaw_v2)')
    p.add_argument('--tag', default=None, help="output tag (default: the run args' tag, else 'records')")
    p.add_argument('--threads', type=int, default=8, help='BLAS/OpenMP threads in total, capped at 8 (per process: threads // workers)')
    p.add_argument('--workers', type=int, default=1, help='forked worker processes for the re-solves (default 1 = in-process; at most 8)')
    p.add_argument('--engine-path', dest='engine_path', default=None, help="directory of the engine's field modules (default: this script's directory); imported read-only")
    p.add_argument('--engine-module', dest='engine_module', default=None, help="the field module (default: the records' solver.module, else the module named in the npz 'solver' string, else engine_fields_v16)")
    p.add_argument('--rtol', type=float, default=None, help="solve_constraints_ivp rtol (default: the records' solver dict, the run args, the npz 'solver' string, then the engine default)")
    p.add_argument('--atol', type=float, default=None, help='solve_constraints_ivp atol (same lookup)')
    p.add_argument('--collar-policy', dest='collar_policy', default=None, help="solve_constraints_ivp collar_policy (same lookup; 'flag' when nowhere stated)")
    p.add_argument('--support-rel', dest='support_rel', type=float, default=None, help='solve_constraints_ivp support_rel (same lookup)')
    p.add_argument('--r0-frac', dest='r0_frac', type=float, default=None, help='solve_constraints_ivp r0_frac (same lookup)')
    p.add_argument('--G', type=float, default=None, help="Newton constant g (default: npz 'g', else run args 'g')")
    p.add_argument('--rs', type=float, default=None, help="r_s in lattice units (default: run args 'rs')")
    p.add_argument('--ell', type=float, default=None, help="the kernel width ell (default: npz 'ell', else run args 'ell')")
    p.add_argument('--stop-E', dest='stop_E', type=float, default=None, help="the solver's relative E stop (default: run args 'stop_E', else 1e-7); enters the E floor")
    p.add_argument('--T-start', dest='T_start', type=float, default=None, help='first T/r_s considered (default: all records)')
    p.add_argument('--T-end', dest='T_end', type=float, default=None, help='last T/r_s considered (default: all records)')
    p.add_argument('--fine', type=int, default=16, help='the dense grid spacing is ell/fine (the integrals, the midpoint interpolations, the stored profiles)')
    p.add_argument('--fd-h-frac', dest='fd_h_frac', type=float, default=1.0 / 64.0, help="the Richardson centred-difference step on the dense functions as a fraction of ell (the engine's residual rule: ell/64)")
    p.add_argument('--scan-frac', dest='scan_frac', type=float, default=1.0 / 64.0, help="step of the independent zero scan of the dense N + v as a fraction of ell (the engine's trapping_radius resolution)")
    p.add_argument('--zero-tol', dest='zero_tol', type=float, default=1e-3, help="sites: every re-solved zero must agree with the stored trapping_zeros within this (and the counts must agree) for the dense solution to count as the record's; reported, never widened by the script")
    p.add_argument('--E-match-rel', dest='E_match_rel', type=float, default=1e-6, help='|E_collar(re-solved) - E|/|E| above this raises the flag E_mismatch (reported, not excluding by default)')
    p.add_argument('--resolve', choices=('rooted', 'all', 'none'), default='rooted', help='which records to re-solve: the rooted ones and their neighbours (default), all matched records, or none (the cache only)')
    p.add_argument('--max-records', dest='max_records', type=int, default=None, help='re-solve at most this many records (a trial run)')
    p.add_argument('--no-cache', dest='no_cache', action='store_true', help='do not read or write the per-record cache of the dense results')
    p.add_argument('--recompute', action='store_true', help='ignore the cache contents (the cache is rewritten)')
    p.add_argument('--check-applied', dest='check_applied', action='store_true', help="also run the engine's site_fields_kernel_average on each re-solved solution and compare it with the stored site and applied arrays")
    p.add_argument('--no-pairs-split', dest='no_pairs_split', action='store_true', help='skip the smoothing of eps_pairs/J_pairs (the pairs/dust split of W_ext)')
    p.add_argument('--interp-degree', dest='interp_degree', type=int, default=4, help='degree of the local polynomial for interpolations on the grids (degree+1 nearest points)')
    p.add_argument('--root-agree-max', dest='root_agree_max', type=float, default=0.05, help='sites: |R_site_refit - R_dense| above this raises the flag site_root_disagree (reported only)')
    p.add_argument('--r-min-zero', dest='r_min_zero', type=float, default=2.0, help='zeros of N+v below this r are ignored (the centre)')
    p.add_argument('--simple-margin-min', dest='simple_margin_min', type=float, default=0.05, help='s = 2 R kappa_R / N_R = 1 - 8 pi g R^2 (rho_ell - J_ell) must exceed this for a simple root (0 at a double root, 1 in vacuum)')
    p.add_argument('--kappa-min-abs', dest='kappa_min_abs', type=float, default=1e-5, help='the dense slope at the root must exceed this (absolute, lattice units)')
    p.add_argument('--max-inner-zeros', dest='max_inner_zeros', type=int, default=1, help='at most this many inner-type dense zeros (negative slope) beside the single outer zero')
    p.add_argument('--max-dR-per-pair', dest='max_dR_per_pair', type=float, default=4.0, help='sites: |R_b - R_a| above this is a root-branch jump (pair excluded)')
    p.add_argument('--max-gap-factor', dest='max_gap_factor', type=float, default=1.5, help='a pair whose T gap exceeds this times the median gap is non-consecutive (excluded)')
    p.add_argument('--tangency-tol-abs', dest='tangency_tol_abs', type=float, default=1e-4, help='|N+v| at a sign-preserving local extremum of the dense N+v below this is a near tangency (flag)')
    p.add_argument('--regime-label-ok', dest='regime_label_ok', default='in the regime', help="the regime.label that passes")
    p.add_argument('--gauge-label-ok', dest='gauge_label_ok', default='intact', help="the gauge.label that passes")
    p.add_argument('--exclude-on', dest='exclude_on', default='solve_failed,zeros_not_reproduced,not_resolved,not_converged,regime,gauge,tangency_stored,tangency_dense,outer_zero_count,inner_zero_count,stored_zero_count,stored_root_not_outermost,not_simple_margin,slope', help='comma list of record-level flags that exclude a pair (others are reported only)')
    p.add_argument('--exclude-pair-on', dest='exclude_pair_on', default='non_consecutive,root_appearing,root_disappearing,root_jump,collar_changed,no_dense_root', help='comma list of pair-level flags that exclude a pair (zero_count_changed and seam are flag-only by default)')
    p.add_argument('--seam-T-over-rs', dest='seam_T', default='', help='comma list of T/r_s where the history was resumed from another engine version; pairs touching one are flagged')
    p.add_argument('--exclude-seam', dest='exclude_seam', action='store_true', help='exclude pairs touching a seam')
    p.add_argument('--kappa-column', dest='kappa_column', choices=('equations', 'formula', 'fd', 'kappa_T', 'midfields'), default='equations', help='which kappa enters the primary column 2 and residual (all variants are printed)')
    p.add_argument('--E-key', dest='E_key', default='E')
    p.add_argument('--collar-key', dest='collar_key', default='r_collar')
    p.add_argument('--zeros-key', dest='zeros_key', default='trapping_zeros')
    p.add_argument('--tangencies-key', dest='tangencies_key', default='tangencies')
    p.add_argument('--kappaT-key', dest='kappaT_key', default='kappa_T')
    p.add_argument('--floor-N-key', dest='floor_N_key', default='interp_floor_N')
    p.add_argument('--floor-v-key', dest='floor_v_key', default='interp_floor_v')
    p.add_argument('--T-match-tol', dest='T_match_tol', type=float, default=1e-6)
    p.add_argument('--uniform-rtol', dest='uniform_rtol', type=float, default=1e-9)
    p.add_argument('--md-every', dest='md_every', type=int, default=8)
    p.add_argument('--list-no-root-pairs', dest='list_no_root_pairs', action='store_true')
    v3_add_arguments(p)
    return p.parse_args(argv)


def make_opts(**overrides):
    opts = parse_args([])
    for key, val in overrides.items():
        if not hasattr(opts, key):
            raise KeyError('unknown option ' + key)
        setattr(opts, key, val)
    return opts


def fill_defaults(opts, run_args, F, notes):
    if opts.G is None:
        g = F.get('G')
        if g is None or not np.isfinite(_float(g)):
            g = run_args.get('g')
        if g is None:
            raise SystemExit("G not found: pass --G (the npz has no 'g', the run args no 'g')")
        opts.G = float(g)
    if opts.ell is None:
        e = F.get('ell')
        if e is None or not np.isfinite(_float(e)):
            e = run_args.get('ell')
        if e is None:
            raise SystemExit("ell not found: pass --ell (the npz has no 'ell', the run args no 'ell')")
        opts.ell = float(e)
    if opts.rs is None:
        rs = run_args.get('rs')
        if rs is None:
            rs = 1.0
            notes.append("run args have no 'rs': r_s = 1 assumed")
        opts.rs = float(rs)
    if opts.stop_E is None:
        se = run_args.get('stop_E')
        opts.stop_E = float(se) if se is not None else 1e-7
        if se is None:
            notes.append("run args have no 'stop_E': 1e-7 assumed for the E floor")
    if opts.tag is None:
        opts.tag = str(run_args.get('tag') or 'records')
    opts.threads = THREADS
    opts.workers = WORKERS


def worker_cfg(opts, r_last_node, h_fine):
    return {'r_min_zero': float(opts.r_min_zero), 'fd_h_frac': float(opts.fd_h_frac), 'scan_frac': float(opts.scan_frac),
            'check_applied': bool(opts.check_applied), 'pairs_split': not bool(opts.no_pairs_split), 'interp_degree': int(opts.interp_degree),
            'r_last_node': float(r_last_node), 'h_fine': float(h_fine), 'tangency_tol_abs': float(opts.tangency_tol_abs)}


def ell_for_engine(ell):
    return int(round(ell)) if abs(ell - round(ell)) < 1e-12 else float(ell)


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def load_json_records(path):
    with open(path) as fh:
        head = json.load(fh)
    run_args = {}
    records = None
    rec_key = None
    if isinstance(head, dict):
        if isinstance(head.get('args'), dict):
            run_args = head['args']
        for key in ('passes', 'records', 'history', 'members', 'steps'):
            val = head.get(key)
            if isinstance(val, list) and val and all(isinstance(x, dict) for x in val) and any('T_over_rs' in x for x in val):
                records, rec_key = val, key
                break
        if records is None:
            for key, val in head.items():
                if isinstance(val, list) and val and all(isinstance(x, dict) for x in val) and any('T_over_rs' in x for x in val):
                    records, rec_key = val, key
                    break
    elif isinstance(head, list):
        records, rec_key = head, '<top-level list>'
    if records is None:
        keys = list(head.keys()) if isinstance(head, dict) else []
        raise SystemExit('no list of record dicts carrying T_over_rs found in %s; top-level keys: %s' % (path, keys))
    return head, run_args, records, rec_key


def load_fields(path):
    z = np.load(path, allow_pickle=False)
    F = {'keys': list(z.files), 'path': path}
    F['R'] = np.asarray(z['R'], float)
    F['T'] = np.asarray(z['T'], float)
    F['eps_tot'] = np.asarray(z['eps_tot'], float)
    F['J_tot'] = np.asarray(z['J_tot'], float)
    for src, dst in (('eps_pairs', 'eps_pairs'), ('J_pairs', 'J_pairs'), ('N', 'N_site'), ('v', 'v_site'), ('N_applied', 'N_applied'), ('v_applied', 'v_applied')):
        F[dst] = np.asarray(z[src], float) if src in z.files else None
    F['G'] = float(z['g']) if 'g' in z.files else None
    F['ell'] = float(z['ell']) if 'ell' in z.files else None
    F['tau'] = float(z['tau']) if 'tau' in z.files else None
    F['solver_string'] = read_npz_string(z, 'solver')
    F['label'] = read_npz_string(z, 'label')
    return F


def match_rows(records, T_npz, rs, tol):
    T_npz = np.asarray(T_npz, float)
    n = len(records)
    best_rows = np.full(n, -1, dtype=int)
    best_mode = None
    best_n = -1
    for mode in ('T', 'T_over_rs', 'T_over_rs*rs', 'T/rs'):
        cand = np.full(n, -1, dtype=int)
        for k, rec in enumerate(records):
            if mode == 'T':
                t = rec.get('T')
            elif mode == 'T_over_rs':
                t = rec.get('T_over_rs')
            elif mode == 'T_over_rs*rs':
                t = rec.get('T_over_rs')
                t = None if t is None else float(t) * rs
            else:
                t = rec.get('T')
                t = None if t is None else float(t) / rs
            if t is None:
                continue
            t = float(t)
            j = int(np.argmin(np.abs(T_npz - t)))
            if abs(T_npz[j] - t) <= tol * max(1.0, abs(t)):
                cand[k] = j
        n_ok = int((cand >= 0).sum())
        if n_ok > best_n:
            best_n, best_rows, best_mode = n_ok, cand, mode
        if n_ok == n:
            break
    return best_rows, best_mode, n - best_n


# ----------------------------------------------------------------------------
# per record
# ----------------------------------------------------------------------------
REC_NAN_KEYS = ('R_dense', 'R_dense_minus_R_stored', 'max_abs_zero_diff', 'max_abs_scan_minus_engine', 'N_R', 'v_R', 'q_R', 'W_R', 'rho_R', 'J_R',
                'Nplusv_R', 'kappa_eq', 'kappa_formula', 'kappa_fd', 'kappa_fd2', 'Np_R', 'vp_R', 'fpp_R', 'simple_margin', 'simple_margin_eq',
                'shift_pred_3d', 'shift_pred_1d', 'kappa_col', 'kappa_eq_minus_kappa_T', 'kappa_fd_minus_kappa_eq', 'kappa_formula_minus_kappa_eq',
                'kappa_spread', 'N_R_minus_N_t_stored', 'four_GE_kappa', 'A', 'R_over_2GE', 'E_collar', 'E_collar_minus_E', 'E_collar_rel_diff',
                'r_collar', 'row_e_rel_ext', 'row_g_rel_ext', 'row_W_rel_ext', 'row_e_rel_ext_2h', 'row_g_rel_ext_2h', 'row_e_rel_ext_flipJ',
                'row_g_rel_ext_flipJ', 'row_e_max_rel_ext', 'row_g_max_abs_ext', 'uJ_over_rho_rms', 'residual_W', 'residual_invN',
                'source_spline_error', 'charge_conservation_quadrature', 'solve_seconds', 'nfev', 'segments', 'N_centre',
                'interp_floor_N_resolved', 'interp_floor_v_resolved', 'collar_deficit_ell', 'regime_max_ratio',
                'R_site_refit', 'R_applied_refit', 'R_densesites_refit', 'R_kernel_refit', 'R_site_minus_R_dense', 'R_applied_minus_R_dense',
                'R_densesites_refit_minus_R_dense', 'R_kernel_minus_R_dense', 'kappa_site_refit', 'kappa_applied_refit',
                'max_abs_N_site_minus_dense', 'max_abs_v_site_minus_dense', 'max_abs_N_applied_minus_dense', 'max_abs_v_applied_minus_dense',
                'max_abs_N_site_minus_applied', 'max_abs_v_site_minus_applied', 'max_abs_N_site_minus_kernel', 'max_abs_v_site_minus_kernel',
                'max_abs_N_applied_minus_kernel', 'max_abs_v_applied_minus_kernel', 'max_abs_N_kernel_minus_dense', 'max_abs_v_kernel_minus_dense',
                'site_row_e_rel_ext_raw', 'site_row_e_rel_ext_raw_flipJ', 'site_row_g_rel_ext_raw', 'site_row_e_rel_ext', 'site_row_g_rel_ext',
                'applied_row_e_rel_ext', 'applied_row_g_rel_ext', 'dense_at_sites_row_e_rel_ext', 'dense_at_sites_row_g_rel_ext',
                'kernel_row_e_rel_ext', 'kernel_row_g_rel_ext', 'sigma_R', 'sigma_R_stored', 'sigma_R_scan', 'sigma_R_resid') + V3_REC_NAN_KEYS


def analyse_record(k, rec, row, d, r_fine, h_fine, G, stop_E, h_site, opts, stress=None):
    rs = float(opts.rs)
    out = {'k': int(k), 'row': int(row), 'T_over_rs': _tors(rec, rs), 'T': _float(rec.get('T'))}
    for key in REC_NAN_KEYS:
        out[key] = NAN
    # the stored items
    zs = sorted(float(z) for z in (rec.get(opts.zeros_key) or []) if z is not None)
    tg = rec.get(opts.tangencies_key) or []
    out['zeros_stored'] = zs
    out['n_zeros_stored'] = len(zs)
    out['n_tangencies_stored'] = len(tg)
    out['R_stored'] = max(zs) if zs else NAN
    out['kappa_T_stored'] = _float(rec.get(opts.kappaT_key))
    sj = rec.get('surface_jets') or {}
    out['N_t_stored'] = _float(sj.get('N_t')) if isinstance(sj, dict) else NAN
    out['E_stored'] = _float(rec.get(opts.E_key))
    out['r_collar_stored'] = _float(rec.get(opts.collar_key))
    out['converged'] = rec.get('converged')
    reg = rec.get('regime') or {}
    out['regime_label_stored'] = reg.get('label') if isinstance(reg, dict) else None
    out['regime_intervals_stored'] = len(reg.get('intervals_above_1') or []) if isinstance(reg, dict) else 0
    gau = rec.get('gauge') or {}
    out['gauge_label_stored'] = gau.get('label') if isinstance(gau, dict) else None
    ls = rec.get('late_surface') or {}
    out['late_surface_flag'] = bool(ls.get('flag')) if isinstance(ls, dict) else False
    ind = rec.get('indicator') or {}
    cs = (ind.get('collar_screen') or {}) if isinstance(ind, dict) else {}
    out['collar_screen_dE'] = _float(cs.get('delta_E_signed')) if isinstance(cs, dict) else NAN
    out['collar_label'] = ind.get('collar_label') if isinstance(ind, dict) else None
    sol = rec.get('solver') or {}
    wt = (sol.get('W_tail') or {}) if isinstance(sol, dict) else {}
    out['W_tail_dE'] = (_float(wt.get('E_tail_end')) - _float(wt.get('E_collar'))) if isinstance(wt, dict) and wt else NAN
    out['floor_N'] = _float(rec.get(opts.floor_N_key))
    out['floor_v'] = _float(rec.get(opts.floor_v_key))
    has_root = bool(np.isfinite(out['R_stored']))
    out['has_root'] = has_root
    flags = []
    if out['converged'] is False:
        flags.append('not_converged')
    if (out['regime_label_stored'] is not None and out['regime_label_stored'] != opts.regime_label_ok) or out['regime_intervals_stored'] > 0:
        flags.append('regime')
    if out['gauge_label_stored'] is not None and out['gauge_label_stored'] != opts.gauge_label_ok:
        flags.append('gauge')
    if out['n_tangencies_stored'] > 0:
        flags.append('tangency_stored')
    if out['late_surface_flag']:
        flags.append('late_surface')
    if out['collar_label'] is not None and str(out['collar_label']) != 'passes':
        flags.append('collar_' + str(out['collar_label']).replace(' ', '_'))
    E = out['E_stored']
    out['sigma_E_stop'] = stop_E * abs(E) if np.isfinite(E) else NAN
    out['sigma_E_screen'] = abs(out['collar_screen_dE']) if np.isfinite(out['collar_screen_dE']) else 0.0
    out['sigma_E_tail'] = abs(out['W_tail_dE']) if np.isfinite(out['W_tail_dE']) else 0.0
    out['sigma_E'] = _nan(np.max, [out['sigma_E_stop'], out['sigma_E_screen'], out['sigma_E_tail']])
    out['resolved'] = d is not None
    out['solve_ok'] = bool(d.get('ok')) if d is not None else False
    out['zeros_reproduced'] = None
    out['has_root_dense'] = None
    out['zeros_dense'] = None
    out['zero_diffs'] = None
    out['zeros_source'] = None
    if d is None:
        if has_root:
            flags.append('not_resolved')
        out['flags'] = flags
        return out, None
    out['solve_failed'] = d.get('failed')
    out['solve_seconds'] = _float(d.get('seconds'))
    eng = d.get('engine') or {}
    out['E_collar'] = _float(d.get('E_collar'))
    out['r_collar'] = _float(d.get('r_collar'))
    out['engine_module'] = eng.get('module')
    out['residual_W'] = _float(eng.get('residual_W'))
    out['residual_invN'] = _float(eng.get('residual_invN'))
    out['source_spline_error'] = _float(eng.get('source_spline_error'))
    out['charge_conservation_quadrature'] = _float(eng.get('charge_conservation_quadrature'))
    out['interp_floor_N_resolved'] = _float(eng.get('interp_floor_N'))
    out['interp_floor_v_resolved'] = _float(eng.get('interp_floor_v'))
    ivp = eng.get('ivp') or {}
    out['nfev'] = _float(ivp.get('nfev'))
    out['segments'] = _float(ivp.get('segments'))
    out['N_centre'] = _float(ivp.get('N_centre'))
    greg = eng.get('regime') or {}
    ggau = eng.get('gauge') or {}
    out['regime_label_resolved'] = greg.get('label')
    out['regime_intervals_resolved'] = greg.get('intervals_above_1') or []
    out['regime_max_ratio'] = _float(greg.get('max_ratio'))
    out['gauge_label_resolved'] = ggau.get('label')
    out['gauge_intervals_resolved'] = ggau.get('intervals') or []
    cm = eng.get('collar_margin') or {}
    out['collar_deficit_ell'] = _float(cm.get('deficit_ell')) if isinstance(cm, dict) else NAN
    if np.isfinite(out['collar_deficit_ell']) and out['collar_deficit_ell'] > 0:
        flags.append('collar_deficit')
    if not np.isfinite(out['W_tail_dE']):
        wtr = eng.get('W_tail') or {}
        if isinstance(wtr, dict) and wtr:
            out['W_tail_dE'] = _float(wtr.get('E_tail_end')) - _float(wtr.get('E_collar'))
            out['sigma_E_tail'] = abs(out['W_tail_dE']) if np.isfinite(out['W_tail_dE']) else 0.0
            out['sigma_E'] = _nan(np.max, [out['sigma_E_stop'], out['sigma_E_screen'], out['sigma_E_tail']])
    if not d.get('ok'):
        flags.append('solve_failed')
        out['flags'] = flags
        return out, None
    if eng.get('converged') is False and 'not_converged' not in flags:
        flags.append('not_converged')
    if out['regime_intervals_resolved'] and 'regime' not in flags:
        flags.append('regime')
    if out['regime_label_resolved'] is not None and out['regime_label_stored'] is not None and out['regime_label_resolved'] != out['regime_label_stored']:
        flags.append('regime_mismatch')
    if out['gauge_intervals_resolved'] and 'gauge' not in flags:
        flags.append('gauge')
    if out['gauge_label_resolved'] is not None and out['gauge_label_stored'] is not None and out['gauge_label_resolved'] != out['gauge_label_stored']:
        flags.append('gauge_mismatch')
    if np.isfinite(out['r_collar_stored']) and abs(out['r_collar'] - out['r_collar_stored']) > 1e-9:
        flags.append('r_collar_differs')
    if np.isfinite(E) and np.isfinite(out['E_collar']):
        out['E_collar_minus_E'] = out['E_collar'] - E
        out['E_collar_rel_diff'] = out['E_collar_minus_E'] / abs(E) if E != 0 else NAN
        if np.isfinite(out['E_collar_rel_diff']) and abs(out['E_collar_rel_diff']) > opts.E_match_rel:
            flags.append('E_mismatch')
    # the zeros and their reproduction
    zd = sorted(float(z) for z in (d.get('zeros_dense') or []))
    out['zeros_dense'] = zd
    out['n_zeros_dense'] = len(zd)
    out['zeros_source'] = d.get('zeros_source')
    out['tangencies_dense'] = list(d.get('tangencies_engine') or [])
    out['n_tangencies_dense'] = len(out['tangencies_dense'])
    out['zeros_scan'] = list(d.get('zeros_scan') or [])
    details = d.get('zeros_detail') or []
    out['zeros_detail'] = details
    out['n_outer_dense'] = int(sum(1 for z in details if z.get('type') == 'outer'))
    out['n_inner_dense'] = int(sum(1 for z in details if z.get('type') == 'inner'))
    out['n_zeros_diff'] = len(zd) - len(zs)
    if len(zd) == len(zs):
        diffs = [a - b for a, b in zip(zd, zs)]
        out['zero_diffs'] = diffs
        out['max_abs_zero_diff'] = max(abs(x) for x in diffs) if diffs else 0.0
        out['zeros_reproduced'] = bool(out['max_abs_zero_diff'] <= opts.zero_tol)
    else:
        out['zeros_reproduced'] = False
    if not out['zeros_reproduced']:
        flags.append('zeros_not_reproduced')
    if d.get('engine_zeros_error'):
        flags.append('engine_zeros_unavailable')
        out['engine_zeros_error'] = d.get('engine_zeros_error')
    if zd and out['zeros_scan'] and len(out['zeros_scan']) == len(zd):
        out['max_abs_scan_minus_engine'] = max(abs(a - b) for a, b in zip(sorted(out['zeros_scan']), zd))
    elif zd and out['zeros_scan']:
        out['max_abs_scan_minus_engine'] = abs(max(out['zeros_scan']) - max(zd))
        flags.append('scan_zero_count_differs')
    has_root_dense = bool(zd)
    out['has_root_dense'] = has_root_dense
    root = d.get('root')
    st = d.get('sites') or {}
    if has_root_dense and root:
        for key in ('N_R', 'v_R', 'q_R', 'W_R', 'rho_R', 'J_R', 'Nplusv_R', 'kappa_eq', 'Np_R', 'vp_R', 'kappa_formula', 'kappa_fd', 'kappa_fd2', 'fpp_R', 'shift_pred_3d', 'shift_pred_1d'):
            out[key] = _float(root.get(key))
        out['R_dense'] = float(root['R'])
        out['simple_margin'] = _float(root.get('margin'))
        out['simple_margin_eq'] = _float(root.get('margin_eq'))
        out['kappa_col'] = {'equations': out['kappa_eq'], 'formula': out['kappa_formula'], 'fd': out['kappa_fd'],
                            'kappa_T': out['kappa_T_stored'], 'midfields': out['kappa_eq']}[opts.kappa_column]
        out['R_dense_minus_R_stored'] = out['R_dense'] - out['R_stored'] if has_root else NAN
        out['kappa_eq_minus_kappa_T'] = out['kappa_eq'] - out['kappa_T_stored']
        out['kappa_fd_minus_kappa_eq'] = out['kappa_fd'] - out['kappa_eq']
        out['kappa_formula_minus_kappa_eq'] = out['kappa_formula'] - out['kappa_eq']
        out['kappa_spread'] = _nan(np.max, [abs(out['kappa_fd_minus_kappa_eq']), abs(out['kappa_formula_minus_kappa_eq'])])
        out['N_R_minus_N_t_stored'] = out['N_R'] - out['N_t_stored']
        out['four_GE_kappa'] = 4.0 * G * E * out['kappa_eq']
        out['A'] = FOUR_PI * out['R_dense'] ** 2
        out['R_over_2GE'] = out['R_dense'] / (2.0 * G * E) if E != 0 else NAN
        if out['n_outer_dense'] != 1:
            flags.append('outer_zero_count')
        if out['n_inner_dense'] > opts.max_inner_zeros:
            flags.append('inner_zero_count')
        if len(zs) > opts.max_inner_zeros + 1:
            flags.append('stored_zero_count')
        if has_root and abs(out['R_dense_minus_R_stored']) > opts.zero_tol and 'zeros_not_reproduced' not in flags:
            flags.append('stored_root_not_outermost')
        if not (out['simple_margin_eq'] >= opts.simple_margin_min):
            flags.append('not_simple_margin')
        if not (out['kappa_eq'] > opts.kappa_min_abs):
            flags.append('slope')
        if out['n_tangencies_dense'] > 0:
            flags.append('tangency_dense')
        # the site arrays against the dense root
        out['R_site_refit'] = _float(st.get('site_root_refit'))
        out['R_applied_refit'] = _float(st.get('applied_root_refit'))
        out['R_densesites_refit'] = _float(st.get('dense_at_sites_root_refit'))
        out['R_kernel_refit'] = _float(st.get('kernel_root_refit'))
        out['kappa_site_refit'] = _float(st.get('site_slope_refit'))
        out['kappa_applied_refit'] = _float(st.get('applied_slope_refit'))
        out['R_site_minus_R_dense'] = out['R_site_refit'] - out['R_dense']
        out['R_applied_minus_R_dense'] = out['R_applied_refit'] - out['R_dense']
        out['R_densesites_refit_minus_R_dense'] = out['R_densesites_refit'] - out['R_dense']
        out['R_kernel_minus_R_dense'] = out['R_kernel_refit'] - out['R_dense']
        if np.isfinite(out['R_site_minus_R_dense']) and abs(out['R_site_minus_R_dense']) > opts.root_agree_max * h_site:
            flags.append('site_root_disagree')
        out['sigma_R_stored'] = abs(out['R_dense_minus_R_stored']) if has_root else 0.0
        out['sigma_R_scan'] = abs(out['max_abs_scan_minus_engine']) if np.isfinite(out['max_abs_scan_minus_engine']) else 0.0
        out['sigma_R_resid'] = abs(out['Nplusv_R'] / out['kappa_eq']) if (np.isfinite(out['kappa_eq']) and out['kappa_eq'] != 0) else 0.0
        out['sigma_R'] = _nan(np.max, [out['sigma_R_stored'], out['sigma_R_scan'], out['sigma_R_resid']])
    for key in ('max_abs_N_site_minus_dense', 'max_abs_v_site_minus_dense', 'max_abs_N_applied_minus_dense', 'max_abs_v_applied_minus_dense',
                'max_abs_N_site_minus_applied', 'max_abs_v_site_minus_applied', 'max_abs_N_site_minus_kernel', 'max_abs_v_site_minus_kernel',
                'max_abs_N_applied_minus_kernel', 'max_abs_v_applied_minus_kernel', 'max_abs_N_kernel_minus_dense', 'max_abs_v_kernel_minus_dense'):
        out[key] = _float(st.get(key))
    out['site_row_e_rel_ext_raw'] = _float(st.get('site_row_e_rel_ext_raw_reading'))
    out['site_row_e_rel_ext_raw_flipJ'] = _float(st.get('site_row_e_rel_ext_raw_reading_flipJ'))
    out['site_row_g_rel_ext_raw'] = _float(st.get('site_row_g_rel_ext_raw_reading'))
    for lab in ('site', 'applied', 'dense_at_sites', 'kernel'):
        out[lab + '_row_e_rel_ext'] = _float(st.get(lab + '_row_e_rel_ext'))
        out[lab + '_row_g_rel_ext'] = _float(st.get(lab + '_row_g_rel_ext'))
    if d.get('kernel_average_error'):
        out['kernel_average_error'] = d.get('kernel_average_error')
    au = d.get('audit') or {}
    au2 = d.get('audit_2h') or {}
    out['row_e_rel_ext'] = _float(au.get('rel_e'))
    out['row_g_rel_ext'] = _float(au.get('rel_g'))
    out['row_W_rel_ext'] = _float(au.get('rel_W'))
    out['row_e_rel_ext_flipJ'] = _float(au.get('rel_e_flipJ'))
    out['row_g_rel_ext_flipJ'] = _float(au.get('rel_g_flipJ'))
    out['row_e_max_rel_ext'] = _float(au.get('max_e_rel'))
    out['row_g_max_abs_ext'] = _float(au.get('max_abs_g'))
    out['uJ_over_rho_rms'] = _float(au.get('uJ_over_rho_rms'))
    out['row_e_rel_ext_2h'] = _float(au2.get('rel_e'))
    out['row_g_rel_ext_2h'] = _float(au2.get('rel_g'))
    out['audit_n_points'] = au.get('n')
    out['audit_h'] = _float(au.get('h'))
    if d.get('pairs_split_error'):
        out['pairs_split_error'] = d.get('pairs_split_error')
    # the arrays on the dense grid
    a = d.get('arrays') or {}
    arrays = None
    if all(key in a for key in ('N', 'v', 'rho', 'J')):
        N, v, rho, J = (np.asarray(a[key], float) for key in ('N', 'v', 'rho', 'J'))
        e, g, M, Mp, Np = row_defects(r_fine, N, v, rho, J, G, h_fine)
        f = N + v
        nt = near_tangencies(r_fine, f, opts.tangency_tol_abs, opts.r_min_zero)
        out['n_near_tangencies_dense'] = len(nt)
        if nt:
            flags.append('near_tangency_dense')
        arrays = {'N': N, 'v': v, 'rho': rho, 'J': J, 'rho_p': a.get('rho_p'), 'J_p': a.get('J_p'), 'e': e, 'g': g, 'M': M, 'Nplusv': f}
        v3_record_columns(out, arrays, r_fine, h_fine, G, opts, d, has_root_dense, stress, Np)
        out['M_collar_dense'] = float(np.interp(out['r_collar'], r_fine, M)) if np.isfinite(out['r_collar']) else NAN
        out['N_collar_dense'] = float(np.interp(out['r_collar'], r_fine, N)) if np.isfinite(out['r_collar']) else NAN
    else:
        flags.append('solve_failed')
    out['flags'] = flags
    return out, arrays


# ----------------------------------------------------------------------------
# per pair
# ----------------------------------------------------------------------------
def analyse_pair(A, B, aA, aB, r_fine, G, h, opts):
    deg = int(opts.interp_degree)
    pre = 1.0 / (8.0 * np.pi * G)
    r_c = min(float(A['r_collar']), float(B['r_collar']))
    i_c = int(round(r_c / h))
    if i_c >= r_fine.size or abs(r_fine[i_c] - r_c) > 1e-6 * max(1.0, r_c):
        i_c = int(np.searchsorted(r_fine, r_c, side='right')) - 1
    sl = slice(0, i_c + 1)
    Rg = r_fine[sl]
    base = FOUR_PI * Rg * Rg
    NA, vA, rhoA, JA = aA['N'][sl], aA['v'][sl], aA['rho'][sl], aA['J'][sl]
    NB, vB, rhoB, JB = aB['N'][sl], aB['v'][sl], aB['rho'][sl], aB['J'][sl]
    Nm = 0.5 * (NA + NB)
    vm = 0.5 * (vA + vB)
    rhom = 0.5 * (rhoA + rhoB)
    Jm = 0.5 * (JA + JB)
    drho = rhoB - rhoA
    dJ = JB - JA
    RA, RB = float(A['R_dense']), float(B['R_dense'])
    Rm = 0.5 * (RA + RB)
    A_a, A_b = FOUR_PI * RA * RA, FOUR_PI * RB * RB
    dA = A_b - A_a
    EA, EB = float(A['E_stored']), float(B['E_stored'])
    dE = EB - EA
    Em = 0.5 * (EA + EB)
    dE_collar = float(B['E_collar']) - float(A['E_collar'])

    # column 3 and its variants
    f_mid = base * (Nm * drho + vm * dJ)
    W_rules, i0 = integral_from(Rg, f_mid, Rm, h, 3)
    W = W_rules['trapz']
    W_end_a = integral_from(Rg, base * (NA * drho + vA * dJ), Rm, h, 3)[0]['trapz']
    W_end_b = integral_from(Rg, base * (NB * drho + vB * dJ), Rm, h, 3)[0]['trapz']
    W_low_a = integral_from(Rg, f_mid, RA, h, 3)[0]['trapz']
    W_low_b = integral_from(Rg, f_mid, RB, h, 3)[0]['trapz']
    W_Nrho = integral_from(Rg, base * Nm * drho, Rm, h, 3)[0]['trapz']
    W_vJ = W - W_Nrho
    if all(aX.get(key) is not None for aX in (aA, aB) for key in ('rho_p', 'J_p')):
        f_p = base * (Nm * (np.asarray(aB['rho_p'])[sl] - np.asarray(aA['rho_p'])[sl]) + vm * (np.asarray(aB['J_p'])[sl] - np.asarray(aA['J_p'])[sl]))
        W_pairs = integral_from(Rg, f_p, Rm, h, 3)[0]['trapz']
        W_dust = W - W_pairs
    else:
        W_pairs, W_dust = NAN, NAN
    rcs = _float(B.get('r_collar_stored'))
    W_missing = float(f_mid[-1]) * (rcs - r_c) if (np.isfinite(rcs) and rcs > r_c) else 0.0

    # column 2 and its variants
    kA, kB = float(A['kappa_col']), float(B['kappa_col'])
    k_mean = 0.5 * (kA + kB)
    Nm_R = interp_at(Rg, Nm, Rm, deg)[0]
    rhom_R = interp_at(Rg, rhom, Rm, deg)[0]
    Jm_R = interp_at(Rg, Jm, Rm, deg)[0]
    k_mid = kappa_formula(Nm_R, Rm, rhom_R, Jm_R, G)
    kE_mean = 0.5 * (float(A['kappa_eq']) + float(B['kappa_eq']))
    kF_mean = 0.5 * (float(A['kappa_fd']) + float(B['kappa_fd']))
    kR_mean = 0.5 * (float(A['kappa_formula']) + float(B['kappa_formula']))
    kT_a, kT_b = _float(A.get('kappa_T_stored')), _float(B.get('kappa_T_stored'))
    kT_mean = 0.5 * (kT_a + kT_b)
    k_vac = 1.0 / (4.0 * G * Em)
    col2_eq = kE_mean * dA * pre
    col2_fd = kF_mean * dA * pre
    col2_formula = kR_mean * dA * pre
    col2_mid = k_mid * dA * pre
    col2_T = kT_mean * dA * pre
    col2_a = kA * dA * pre
    col2_b = kB * dA * pre
    col2_vac = k_vac * dA * pre
    col2 = {'equations': col2_eq, 'formula': col2_formula, 'fd': col2_fd, 'kappa_T': col2_T, 'midfields': col2_mid}[opts.kappa_column]

    # residuals
    res = dE - col2 - W
    res_eq = dE - col2_eq - W
    res_fd = dE - col2_fd - W
    res_formula = dE - col2_formula - W
    res_mid = dE - col2_mid - W
    res_T = dE - col2_T - W
    res_simpson = dE - col2 - W_rules['simpson']
    res_coarse = dE - col2 - W_rules['coarse']
    res_noW = dE - col2
    res_vac = dE - col2_vac
    res_collarE = dE_collar - col2 - W

    # the applied identity's defect integral (the dense defects by the grid stencil; a diagnostic)
    dM = aB['M'][sl] - aA['M'][sl]
    de = aB['e'][sl] - aA['e'][sl]
    gm = 0.5 * (aA['g'][sl] + aB['g'][sl])
    gm = np.where(np.isfinite(gm), gm, 0.0)
    fD = Nm * (de + gm * dM)
    D = integral_from(Rg, fD, Rm, h, 3)[0]['trapz']
    Nc = 0.5 * (float(NA[-1]) + float(NB[-1]))
    dM_c = float(aB['M'][sl][-1]) - float(aA['M'][sl][-1])
    res_applied = Nc * dM_c - col2_eq - W - D
    res_applied_storedE = Nc * dE - col2_eq - W - D

    # floors (the records' own) and discretization (ours)
    sE_stop_a, sE_stop_b = _float(A.get('sigma_E_stop')), _float(B.get('sigma_E_stop'))
    scr_a = A['collar_screen_dE'] if np.isfinite(_float(A.get('collar_screen_dE'))) else 0.0
    scr_b = B['collar_screen_dE'] if np.isfinite(_float(B.get('collar_screen_dE'))) else 0.0
    sigma_dE = float(np.sqrt(np.nansum([sE_stop_a ** 2, sE_stop_b ** 2, (scr_b - scr_a) ** 2])))
    sigma_dE_undiff = float(np.sqrt(np.nansum([_float(A.get('sigma_E')) ** 2, _float(B.get('sigma_E')) ** 2])))
    sR_a, sR_b = _float(A.get('sigma_R')), _float(B.get('sigma_R'))
    sigma_col2_root = abs(k_mean) * Rm / G * float(np.sqrt(np.nansum([sR_a ** 2, sR_b ** 2])))
    sigma_col2_kappa = abs(dA) * pre * _nan(np.max, [A.get('kappa_spread'), B.get('kappa_spread')])
    floor = float(np.sqrt(np.nansum([sigma_dE ** 2, sigma_col2_root ** 2, sigma_col2_kappa ** 2])))
    disc_kappa = 0.5 * abs(col2_b - col2_a)
    disc_kappa_mid = abs(col2_mid - col2_eq)
    disc_W_rule = _nan(np.max, [abs(W - W_rules['simpson']), abs(W - W_rules['coarse'])])
    disc_W_lambda = 0.5 * abs(W_end_b - W_end_a)
    disc_W_lower = max(abs(W_low_a - W), abs(W_low_b - W))
    disc = float(np.sqrt(np.nansum([disc_kappa ** 2, disc_W_rule ** 2, disc_W_lambda ** 2, W_missing ** 2])))
    pull = res / floor if floor > 0 else NAN
    tot = float(np.sqrt(floor ** 2 + disc ** 2))
    pull_total = res / tot if tot > 0 else NAN

    pr = dict(T_a=A['T_over_rs'], T_b=B['T_over_rs'], T_mid=0.5 * (A['T_over_rs'] + B['T_over_rs']),
              R_a=RA, R_b=RB, R_mid=Rm, dR=RB - RA, A_a=A_a, A_b=A_b, dA=dA, r_c=r_c,
              E_a=EA, E_b=EB, dE=dE, dE_collar=dE_collar, dM_collar_dense=dM_c, N_collar_mean=Nc, E_mid=Em,
              kappa_a=kA, kappa_b=kB, kappa_mean=k_mean, kappa_eq_mean=kE_mean, kappa_fd_mean=kF_mean, kappa_formula_mean=kR_mean,
              kappa_midfields=k_mid, kappa_T_a=kT_a, kappa_T_b=kT_b, kappa_T_mean=kT_mean, kappa_vac=k_vac,
              four_GE_kappa_mean=4.0 * G * Em * kE_mean, kappa_eq_mean_minus_kappa_T_mean=kE_mean - kT_mean,
              col2=col2, col2_eq=col2_eq, col2_fd=col2_fd, col2_formula=col2_formula, col2_midfields=col2_mid, col2_T=col2_T,
              col2_a=col2_a, col2_b=col2_b, col2_vac=col2_vac,
              W_ext=W, W_simpson=W_rules['simpson'], W_coarse=W_rules['coarse'], W_end_a=W_end_a, W_end_b=W_end_b,
              W_low_a=W_low_a, W_low_b=W_low_b, W_pairs=W_pairs, W_dust=W_dust, W_Nrho=W_Nrho, W_vJ=W_vJ,
              W_missing_strip=W_missing, i0=int(i0),
              res=res, res_eq=res_eq, res_fd=res_fd, res_formula=res_formula, res_midfields=res_mid, res_T=res_T,
              res_simpson=res_simpson, res_coarse=res_coarse, res_noW=res_noW, res_vac=res_vac, res_collarE=res_collarE,
              defect_integral=D, res_applied=res_applied, res_applied_storedE=res_applied_storedE,
              sigma_dE=sigma_dE, sigma_dE_undiff=sigma_dE_undiff, sigma_col2_root=sigma_col2_root,
              sigma_col2_kappa=sigma_col2_kappa, floor=floor,
              disc_kappa=disc_kappa, disc_kappa_mid=disc_kappa_mid, disc_W_rule=disc_W_rule,
              disc_W_lambda=disc_W_lambda, disc_W_lower=disc_W_lower, disc=disc, pull=pull, pull_total=pull_total,
              W_over_dE=W / dE if dE != 0 else NAN, col2_over_dE=col2 / dE if dE != 0 else NAN,
              W_pairs_over_W=W_pairs / W if W != 0 else NAN,
              sigma_E_a=_float(A.get('sigma_E')), sigma_E_b=_float(B.get('sigma_E')),
              kappa_column=opts.kappa_column)
    pr['_f_mid'] = f_mid
    pr['_f_defect'] = fD
    v3_pair_columns(pr, A, B, aA, aB, Rg, sl, Rm, dA, pre, deg)
    return pr


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------
def closure_stats(pairs, recs, G):
    st = {'n_pairs_included': len(pairs)}
    rooted = [r for r in recs if r.get('has_root')]
    resolved = [r for r in recs if r.get('resolved')]
    okd = [r for r in resolved if r.get('solve_ok')]
    rooted_ok = [r for r in rooted if r.get('solve_ok') and np.isfinite(_float(r.get('R_dense')))]
    st['n_records_with_root'] = len(rooted)
    st['n_records_resolved'] = len(resolved)
    st['n_solve_failed'] = len(resolved) - len(okd)
    st['n_rooted_resolved_ok'] = len(rooted_ok)
    st['n_zeros_reproduced'] = int(sum(1 for r in okd if r.get('zeros_reproduced')))
    st['n_zeros_not_reproduced'] = int(sum(1 for r in okd if r.get('zeros_reproduced') is False))
    st['n_rooted_reproduced'] = int(sum(1 for r in rooted_ok if r.get('zeros_reproduced')))
    st['solve_seconds_total'] = _nan(np.sum, [_float(r.get('solve_seconds')) for r in resolved])
    st['solve_seconds_median'] = _nan(np.median, [_float(r.get('solve_seconds')) for r in resolved])
    rms = lambda a: np.sqrt(np.mean(a ** 2))  # noqa: E731

    def med_max(key, pool):
        vals = [_float(r.get(key)) for r in pool]
        return _nan(np.median, vals), _nan(np.max, vals), _nan(np.min, vals)
    for key in ('max_abs_zero_diff', 'max_abs_scan_minus_engine', 'E_collar_rel_diff', 'row_e_rel_ext', 'row_g_rel_ext', 'row_W_rel_ext',
                'row_e_rel_ext_2h', 'row_g_rel_ext_2h', 'row_e_rel_ext_flipJ', 'row_g_rel_ext_flipJ', 'row_e_max_rel_ext', 'row_g_max_abs_ext',
                'residual_W', 'residual_invN', 'source_spline_error', 'charge_conservation_quadrature', 'uJ_over_rho_rms',
                'site_row_e_rel_ext_raw', 'site_row_e_rel_ext_raw_flipJ', 'site_row_g_rel_ext_raw', 'site_row_e_rel_ext', 'site_row_g_rel_ext',
                'applied_row_e_rel_ext', 'applied_row_g_rel_ext', 'dense_at_sites_row_e_rel_ext', 'dense_at_sites_row_g_rel_ext',
                'kernel_row_e_rel_ext', 'kernel_row_g_rel_ext', 'max_abs_N_site_minus_dense', 'max_abs_v_site_minus_dense',
                'max_abs_N_applied_minus_dense', 'max_abs_v_applied_minus_dense', 'max_abs_N_site_minus_applied', 'max_abs_v_site_minus_applied',
                'max_abs_N_site_minus_kernel', 'max_abs_v_site_minus_kernel', 'max_abs_N_applied_minus_kernel', 'max_abs_v_applied_minus_kernel',
                'max_abs_N_kernel_minus_dense', 'max_abs_v_kernel_minus_dense', 'R_site_minus_R_dense', 'R_applied_minus_R_dense',
                'R_densesites_refit_minus_R_dense', 'R_kernel_minus_R_dense', 'shift_pred_3d', 'shift_pred_1d', 'R_dense_minus_R_stored',
                'kappa_eq_minus_kappa_T', 'kappa_fd_minus_kappa_eq', 'kappa_formula_minus_kappa_eq', 'kappa_spread', 'N_R_minus_N_t_stored',
                'simple_margin_eq', 'four_GE_kappa', 'R_over_2GE', 'sigma_E', 'sigma_E_stop', 'sigma_E_screen', 'sigma_E_tail', 'sigma_R',
                'sigma_R_stored', 'sigma_R_scan', 'collar_deficit_ell', 'nfev', 'N_centre', 'interp_floor_N_resolved', 'interp_floor_v_resolved'):
        pool = rooted_ok if key not in ('max_abs_zero_diff', 'E_collar_rel_diff', 'row_e_rel_ext', 'row_g_rel_ext') else okd
        m, x, n = med_max(key, pool)
        st[key + '_median'] = m
        st[key + '_max'] = x
        st[key + '_min'] = n
    for key in ('kappa_eq_minus_kappa_T', 'kappa_fd_minus_kappa_eq', 'kappa_formula_minus_kappa_eq', 'R_dense_minus_R_stored', 'E_collar_rel_diff',
                'R_site_minus_R_dense', 'R_applied_minus_R_dense', 'R_densesites_refit_minus_R_dense', 'N_R_minus_N_t_stored'):
        vals = np.array([_float(r.get(key)) for r in rooted_ok])
        st[key + '_max_abs'] = _nan(np.max, np.abs(vals))
        st[key + '_rms'] = _nan(rms, vals)
    if rooted_ok:
        st['kappa_T_mean'] = _nan(np.mean, [_float(r.get('kappa_T_stored')) for r in rooted_ok])
        st['kappa_eq_mean'] = _nan(np.mean, [_float(r.get('kappa_eq')) for r in rooted_ok])
        st['kappa_fd_mean'] = _nan(np.mean, [_float(r.get('kappa_fd')) for r in rooted_ok])
        st['kappa_formula_mean'] = _nan(np.mean, [_float(r.get('kappa_formula')) for r in rooted_ok])
    v3_record_stats(st, rooted_ok)
    if not pairs:
        return st

    def arr(key):
        return np.array([_float(p.get(key)) for p in pairs])

    dE, c2, W, res = arr('dE'), arr('col2'), arr('W_ext'), arr('res')
    floor, disc = arr('floor'), arr('disc')
    for key in ('dE', 'dE_collar', 'col2', 'col2_eq', 'col2_fd', 'col2_formula', 'col2_T', 'col2_midfields', 'col2_vac',
                'W_ext', 'W_simpson', 'W_coarse', 'W_pairs', 'W_dust', 'W_Nrho', 'W_vJ', 'res', 'res_eq', 'res_fd', 'res_formula',
                'res_T', 'res_midfields', 'res_simpson', 'res_coarse', 'res_noW', 'res_vac', 'res_collarE',
                'defect_integral', 'res_applied', 'res_applied_storedE', 'disc', 'dA', 'dR'):
        st['sum_' + key] = _nan(np.sum, arr(key))
    st['E_first'] = float(pairs[0]['E_a'])
    st['E_last'] = float(pairs[-1]['E_b'])
    st['E_last_minus_E_first'] = st['E_last'] - st['E_first']
    st['R_first'] = float(pairs[0]['R_a'])
    st['R_last'] = float(pairs[-1]['R_b'])
    st['T_first'] = float(pairs[0]['T_a'])
    st['T_last'] = float(pairs[-1]['T_b'])
    for key in ('res', 'res_eq', 'res_fd', 'res_formula', 'res_T', 'res_midfields', 'res_simpson', 'res_collarE', 'res_applied', 'res_noW', 'res_vac'):
        a = arr(key)
        st[key + '_mean'] = _nan(np.mean, a)
        st[key + '_rms'] = _nan(rms, a)
        st[key + '_max_abs'] = _nan(np.max, np.abs(a))
        if np.any(np.isfinite(a)):
            st[key + '_max_abs_T_b'] = float(pairs[int(np.nanargmax(np.abs(a)))]['T_b'])
    st['dE_rms'] = _nan(rms, dE)
    st['col2_rms'] = _nan(rms, c2)
    st['W_ext_rms'] = _nan(rms, W)
    st['res_over_dE_rms'] = st['res_rms'] / st['dE_rms'] if st['dE_rms'] > 0 else NAN
    st['res_over_W_rms'] = st['res_rms'] / st['W_ext_rms'] if st['W_ext_rms'] > 0 else NAN
    pull = arr('pull')
    pull_t = arr('pull_total')
    st['pull_rms'] = _nan(rms, pull)
    st['pull_total_rms'] = _nan(rms, pull_t)
    for q in (1, 2, 3):
        st['frac_abs_pull_lt_%d' % q] = _nan(np.mean, (np.abs(pull) < q).astype(float)) if np.any(np.isfinite(pull)) else NAN
        st['frac_abs_pull_total_lt_%d' % q] = _nan(np.mean, (np.abs(pull_t) < q).astype(float)) if np.any(np.isfinite(pull_t)) else NAN
    for key in ('W_over_dE', 'col2_over_dE', 'W_pairs_over_W', 'four_GE_kappa_mean'):
        a = arr(key)
        st[key + '_median'] = _nan(np.median, a)
        st[key + '_min'] = _nan(np.min, a)
        st[key + '_max'] = _nan(np.max, a)
    st['W_ext_abs_median'] = _nan(np.median, np.abs(W))
    st['dE_abs_median'] = _nan(np.median, np.abs(dE))
    st['col2_abs_median'] = _nan(np.median, np.abs(c2))
    st['floor_median'] = _nan(np.median, floor)
    st['disc_median'] = _nan(np.median, disc)
    st['disc_W_rule_median'] = _nan(np.median, arr('disc_W_rule'))
    st['disc_kappa_median'] = _nan(np.median, arr('disc_kappa'))
    st['floor_cum'] = float(np.sqrt(np.nansum([_float(pairs[0]['sigma_E_a']) ** 2, _float(pairs[-1]['sigma_E_b']) ** 2])
                                    + np.nansum(arr('sigma_col2_root') ** 2) + np.nansum(arr('sigma_col2_kappa') ** 2)))
    st['disc_cum'] = _nan(np.sum, disc)
    st['cum_pull'] = st['sum_res'] / st['floor_cum'] if st['floor_cum'] > 0 else NAN
    tot = np.sqrt(st['floor_cum'] ** 2 + st['disc_cum'] ** 2)
    st['cum_pull_total'] = st['sum_res'] / tot if tot > 0 else NAN
    st['sum_W_over_sum_dE'] = st['sum_W_ext'] / st['sum_dE'] if st['sum_dE'] != 0 else NAN
    st['sum_col2_over_sum_dE'] = st['sum_col2'] / st['sum_dE'] if st['sum_dE'] != 0 else NAN
    st['sum_res_over_sum_dE'] = st['sum_res'] / st['sum_dE'] if st['sum_dE'] != 0 else NAN
    v3_pair_stats(st, pairs)
    return st


# ----------------------------------------------------------------------------
# the analysis
# ----------------------------------------------------------------------------
def analyse(records, F, run_args, opts, dense, r_fine, prov=None, notes=None, stress=None):
    t0 = time.time()
    notes = list(notes or [])
    fill_defaults(opts, run_args, F, notes)
    R = np.asarray(F['R'], float)
    r_fine = np.asarray(r_fine, float)
    h_fine = uniform_step(r_fine, opts.uniform_rtol)
    if h_fine is None:
        raise SystemExit('the dense grid is not uniform')
    h_site = uniform_step(R, opts.uniform_rtol)
    h_site_eff = h_site if h_site is not None else float(np.median(np.diff(R)))
    G = float(opts.G)
    rs = float(opts.rs)
    stop_E = float(opts.stop_E)
    T_npz = np.asarray(F['T'], float)
    rows, match_mode, n_unmatched = match_rows(records, T_npz, rs, opts.T_match_tol)
    if n_unmatched:
        notes.append('%d records not matched to a fields row (matching by %s): skipped' % (n_unmatched, match_mode))
    idx = [k for k in range(len(records)) if rows[k] >= 0]
    idx.sort(key=lambda k: _tors(records[k], rs))
    if opts.T_start is not None:
        idx = [k for k in idx if _tors(records[k], rs) >= opts.T_start - 1e-9]
    if opts.T_end is not None:
        idx = [k for k in idx if _tors(records[k], rs) <= opts.T_end + 1e-9]
    if len(idx) < 2:
        raise SystemExit('fewer than 2 analysable records (matched rows within the T window)')
    seams = [float(s) for s in str(opts.seam_T).split(',') if s.strip() != '']

    recs = []
    arr = {}
    for k in idx:
        rec_out, rec_arr = analyse_record(k, records[k], int(rows[k]), dense.get(k), r_fine, h_fine, G, stop_E, h_site_eff, opts,
                                          stress=((stress or {}).get('records') or {}).get(k))
        recs.append(rec_out)
        arr[k] = rec_arr

    exclude_rec = set(s.strip() for s in str(opts.exclude_on).split(',') if s.strip())
    exclude_pair = set(s.strip() for s in str(opts.exclude_pair_on).split(',') if s.strip())
    Ts = np.array([r['T_over_rs'] for r in recs], float)
    gaps = np.diff(Ts)
    gap_med = float(np.median(gaps)) if gaps.size else NAN
    pairs_inc = []
    pairs_exc = []
    n_no_root = 0
    no_root_T = []
    for i in range(len(recs) - 1):
        A, B = recs[i], recs[i + 1]
        pflags = []
        gap = B['T_over_rs'] - A['T_over_rs']
        if np.isfinite(gap_med) and gap > opts.max_gap_factor * gap_med:
            pflags.append('non_consecutive')
        if not A['has_root'] and not B['has_root']:
            n_no_root += 1
            no_root_T.append((A['T_over_rs'], B['T_over_rs']))
            if opts.list_no_root_pairs:
                pairs_exc.append({'i': i, 'T_a': A['T_over_rs'], 'T_b': B['T_over_rs'], 'R_a': None, 'R_b': None,
                                  'reasons': ['no_root'] + pflags, 'other_flags': []})
            continue
        if A['has_root'] and not B['has_root']:
            pflags.append('root_disappearing')
        if B['has_root'] and not A['has_root']:
            pflags.append('root_appearing')
        if A['has_root'] and B['has_root']:
            if abs(B['R_stored'] - A['R_stored']) > opts.max_dR_per_pair * h_site_eff:
                pflags.append('root_jump')
            if A['n_zeros_stored'] != B['n_zeros_stored']:
                pflags.append('zero_count_changed')
            rca, rcb = _float(A.get('r_collar')), _float(B.get('r_collar'))
            if np.isfinite(rca) and np.isfinite(rcb) and abs(rca - rcb) > 1e-9:
                pflags.append('collar_changed')
            if not (np.isfinite(_float(A.get('R_dense'))) and np.isfinite(_float(B.get('R_dense'))) and arr.get(A['k']) is not None and arr.get(B['k']) is not None):
                pflags.append('no_dense_root')
        if any(abs(A['T_over_rs'] - s) < 1e-9 or abs(B['T_over_rs'] - s) < 1e-9 for s in seams):
            pflags.append('seam')
        reasons = [f for f in pflags if f in exclude_pair]
        if 'seam' in pflags and opts.exclude_seam:
            reasons.append('seam')
        reasons += ['A:' + f for f in A['flags'] if f in exclude_rec]
        reasons += ['B:' + f for f in B['flags'] if f in exclude_rec]
        if 'no_dense_root' in pflags and 'no_dense_root' not in reasons:
            reasons.append('no_dense_root')
        if reasons:
            pairs_exc.append({'i': i, 'T_a': A['T_over_rs'], 'T_b': B['T_over_rs'], 'R_a': A['R_stored'], 'R_b': B['R_stored'],
                              'reasons': reasons, 'other_flags': [f for f in pflags if f not in reasons]})
            continue
        pr = analyse_pair(A, B, arr[A['k']], arr[B['k']], r_fine, G, h_fine, opts)
        pr['i'] = i
        pr['flags'] = pflags + ['A:' + f for f in A['flags']] + ['B:' + f for f in B['flags']]
        pairs_inc.append(pr)

    cum = {'dE': 0.0, 'dE_collar': 0.0, 'col2': 0.0, 'W_ext': 0.0, 'res': 0.0, 'defect_integral': 0.0, 'res_applied': 0.0,
           'W_pairs': 0.0, 'W_dust': 0.0, 'res_T': 0.0, 'res_simpson': 0.0, 'res_vac': 0.0, 'res_noW': 0.0, 'res_collarE': 0.0}
    for pr in pairs_inc:
        for key in cum:
            val = _float(pr.get(key))
            cum[key] += val if np.isfinite(val) else 0.0
            pr['cum_' + key] = cum[key]
        pr['E_b_minus_E_first'] = pr['E_b'] - pairs_inc[0]['E_a']

    stats = closure_stats(pairs_inc, recs, G)
    stats['n_pairs_excluded_listed'] = len(pairs_exc)
    stats['n_pairs_no_root'] = n_no_root
    stats['no_root_T_span'] = [no_root_T[0][0], no_root_T[-1][1]] if no_root_T else None
    stats['n_records_analysed'] = len(recs)
    hist = {}
    for pe in pairs_exc:
        for r in pe['reasons']:
            hist[r] = hist.get(r, 0) + 1
    stats['excluded_reason_histogram'] = hist
    not_rep = [{'T_over_rs': r['T_over_rs'], 'zeros_stored': r.get('zeros_stored'), 'zeros_dense': r.get('zeros_dense'),
                'zero_diffs': r.get('zero_diffs'), 'max_abs_zero_diff': r.get('max_abs_zero_diff'), 'solve_failed': r.get('solve_failed'),
                'zeros_source': r.get('zeros_source')} for r in recs if r.get('resolved') and (not r.get('solve_ok') or r.get('zeros_reproduced') is False)]
    stats['records_not_reproduced'] = not_rep
    if stats['n_rooted_resolved_ok'] > 0 and stats['n_rooted_reproduced'] < stats['n_rooted_resolved_ok']:
        notes.append('REPRODUCTION: %d of %d rooted re-solved records do not reproduce their stored trapping_zeros within --zero-tol %g sites (max |difference| over the re-solved records %s); they are excluded; the tolerance is not widened by the script' % (
            stats['n_rooted_resolved_ok'] - stats['n_rooted_reproduced'], stats['n_rooted_resolved_ok'], opts.zero_tol, _fmt(stats.get('max_abs_zero_diff_max'), 3)))
    if stats.get('n_solve_failed', 0) > 0:
        notes.append('%d re-solves failed (see the reproduction table)' % stats['n_solve_failed'])
    thr = {key: getattr(opts, key) for key in ('fine', 'fd_h_frac', 'scan_frac', 'zero_tol', 'E_match_rel', 'resolve', 'max_records', 'check_applied',
                                                'no_pairs_split', 'interp_degree', 'root_agree_max', 'r_min_zero', 'simple_margin_min', 'kappa_min_abs',
                                                'max_inner_zeros', 'max_dR_per_pair', 'max_gap_factor', 'tangency_tol_abs', 'regime_label_ok',
                                                'gauge_label_ok', 'exclude_on', 'exclude_pair_on', 'seam_T', 'exclude_seam', 'T_match_tol',
                                                'uniform_rtol', 'kappa_column', 'T_start', 'T_end', 'workers', 'threads') + V3_THRESHOLD_KEYS}
    rc_vals = [r['r_collar'] for r in recs if np.isfinite(_float(r.get('r_collar')))]
    rc_str = _fmt(rc_vals[0]) if rc_vals else 'not re-solved'
    rows_text = ("the rows the engine integrates (integrate_UL.rhs_pos and the final W integration of solve_constraints_ivp): "
                 "U' = 8 pi g r^2 (rho_ell + q J_ell), (ln N)' = -4 pi g r J_ell/q with q = -sqrt(U/r) = v/N; W' = 8 pi g r^2 N^2 rho_ell with N(r_c) = 1, "
                 "v = -sqrt(W/r), E = W(r_c)/(2g); with M = U/(2g) = r u^2/(2g) these are M' = 4 pi r^2 (rho + u J) and N'/N = -4 pi G r J/u with "
                 "rho = rho_ell, J = +J_ell, G = g. rho_ell = SmoothedSources5(eps_tot, J_tot, ...).rho and J_ell = .J are the Gaussian-kernel "
                 "(standard deviation ell = %s) averages per unit volume of the stored per-site sources: rho_ell(0) = sum_n eps_tot[n] (2 pi ell^2)^(-3/2) "
                 "exp(-R_n^2/(2 ell^2)) (centre_coefficients), int 4 pi r^2 rho_ell dr = sum eps_tot (charge_conservation_quadrature, printed per record); "
                 "eps_tot[n] is the energy of site n, not a density; J_ell = <P_r> = -J_r of the goal's display (the module's DICTIONARY line)." % _fmt(opts.ell))
    definitions = {
        'rows': rows_text,
        'column_1_dE': ("delta E = E(T_b) - E(T_a); E is the record key '%s' (the collar charge at r_collar = %s). The re-solved E_collar = W(r_c)/(2g) of each "
                        "record is printed beside it (E_collar - E; flag E_mismatch above %g relative) and delta E_collar enters res_collarE beside res. G = %s, r_s = %s." % (
                            opts.E_key, rc_str, opts.E_match_rel, _fmt(G), _fmt(rs))),
        'column_2_kappa_dA': ("(kappa_R/(8 pi G)) delta A, A = 4 pi R^2, R = the outer dense zero of N + v of the re-solved solution (the engine's trapping_radius; "
                              "an independent brentq scan at step ell/%g beside it), delta A = A_b - A_a; kappa_R = d_r(N + v)|_R from the equations the dense functions "
                              "integrate (kappa_eq: N' = -4 pi g r J_ell N^2/v, v' = (W' - W/r)/(2 r v), W' = 8 pi g r^2 N^2 rho_ell), which at the root equals the row "
                              "formula (N_R/2R)[1 - 8 pi g R^2 (rho_ell(R) - J_ell(R))] (kappa_formula); the Richardson centred difference of the dense N + v at "
                              "h = ell/%g (kappa_fd) and the stored '%s' beside them; kappa evaluated at both records and averaged (the midpoint rule in the record "
                              "index); the primary column uses kappa = %s; the kappa spread max(|kappa_fd - kappa_eq|, |kappa_formula - kappa_eq|) enters the floor."
                              % (1.0 / opts.scan_frac, 1.0 / opts.fd_h_frac, opts.kappaT_key, opts.kappa_column)),
        'column_3_W_ext': ("W_ext = 4 pi int_{R_mid}^{r_c} r^2 [N_mid (rho_ell,b - rho_ell,a) + v_mid (J_ell,b - J_ell,a)] dr on the dense grid (spacing ell/%d = %s) "
                           "with the Eulerian differences at fixed r, N and v the dense functions, rho_ell and J_ell the engine's smoothed sources of each record, "
                           "N_mid = (N_a + N_b)/2, v_mid likewise, R_mid = (R_a + R_b)/2, r_c = the re-solved collar (equal for both records, else collar_changed); "
                           "trapezoid with the partial cell [R_mid, next point] by the trapezoid on the interpolated integrand; Simpson and the half grid beside as "
                           "the radial discretization error; the endpoint weights (N_a, v_a) and (N_b, v_b) beside as the record-index discretization error; the "
                           "lower limits R_a and R_b beside. Split into the N delta rho and v delta J parts and, when eps_pairs/J_pairs are stored, into the pairs' "
                           "part (eps_pairs, J_pairs smoothed with the engine's own SmoothedSources5) and the dust's part." % (opts.fine, _fmt(h_fine))),
        'residual': ("res = delta E - (kappa/(8 pi G)) delta A - W_ext per pair, cumulated over the included pairs in order; res_collarE the same with delta E_collar; "
                     "res_noW = delta E - (kappa/8 pi G) delta A (the no-work premise); res_vac = delta E - delta A/(32 pi G^2 E_mid) (the vacuum first law)."),
        'applied_identity_diagnostic': ("res_applied = N_c delta M_c - (kappa_eq/(8 pi G)) delta A - W_ext - int_{R_mid}^{r_c} N_mid [(e_b - e_a) + g_mid (M_b - M_a)] dr with "
                                        "M = r u^2/(2G), e = M' - 4 pi r^2 (rho_ell + u J_ell), g = N'/N + 4 pi G r J_ell/u on the dense grid (4th-order stencil at "
                                        "spacing %s), N_c = N at r_c (1 by the solver's normalization): the identity with the row defects retained; on the dense "
                                        "solution the defect integral is at the solver's tolerance and res_applied = res with delta E_collar in place of delta E." % _fmt(h_fine)),
        'row_audit': ("Per record, on the dense functions in the exterior (R + 2h, r_c - 2h), derivatives by the Richardson centred difference at h = ell/%g "
                      "(the engine's residual rule) and at 2h: RMS of e relative to 4 pi r^2 (|rho_ell| + |u J_ell|), RMS of g relative to N'/N, the W form "
                      "W' - 8 pi g r^2 N^2 rho_ell relative to the source term; the same with J_ell -> -J_ell for contrast; the engine's own residual_W and "
                      "residual_invN (max norm over its fine grid). Expected: the solver's tolerance. For the record, the same rows on the stored site arrays with "
                      "version 1's raw reading (eps_tot, J_tot as densities) and with the smoothed sources at the sites, and on the dense solution sampled at the "
                      "sites (the lattice stencil's own error)." % (1.0 / opts.fd_h_frac)),
        'premises': ("A record carries a simple dense root when: its re-solve succeeded and its dense zeros reproduce the stored '%s' (count and every value within "
                     "%g sites); exactly one outer-type dense zero (positive slope) and at most %d inner-type; the stored zero count at most %d; s = 2 R kappa_eq/N_R "
                     ">= %g; kappa_eq > %g; no stored or dense tangency; converged; regime.label == '%s' with no interval above 1 (stored or re-solved); gauge.label "
                     "== '%s' with no continued interval (re-solved). A pair is excluded when either record fails one of the flags in --exclude-on, or when the root "
                     "appears or disappears between the records, the records are not consecutive (gap > %g x median), |R_b - R_a| > %g sites, or the re-solved "
                     "collars differ. Other flags (E_mismatch, site_root_disagree, collar_deficit, seam, ...) are reported."
                     % (opts.zeros_key, opts.zero_tol, opts.max_inner_zeros, opts.max_inner_zeros + 1, opts.simple_margin_min, opts.kappa_min_abs,
                        opts.regime_label_ok, opts.gauge_label_ok, opts.max_gap_factor, opts.max_dR_per_pair)),
        'floors': ("Per record: sigma_E = max(stop_E |E| with stop_E = %g, |indicator.collar_screen.delta_E_signed|, |W_tail.E_tail_end - E_collar|); "
                   "sigma_R = max(|R_dense - R_stored|, |R_scan - R_engine|, |N + v|(R)/kappa_eq) (the record's interp floors no longer enter: R is read on the dense "
                   "functions, not refit from the site arrays). Per pair: sigma_dE = sqrt(stop_E^2 (E_a^2 + E_b^2) + (screen_b - screen_a)^2); "
                   "sigma_col2 = (kappa_mean R_mid/G) sqrt(sigma_R_a^2 + sigma_R_b^2) (+) |delta A|/(8 pi G) x max kappa spread; floor = sqrt(sigma_dE^2 + sigma_col2^2); "
                   "pull = res/floor. Cumulative floor: sqrt(sigma_E_first^2 + sigma_E_last^2 + sum sigma_col2^2)." % stop_E),
        'discretization': ("disc_kappa = |col2(kappa_b) - col2(kappa_a)|/2; disc_W_rule = max(|W_trapz - W_simpson|, |W_trapz - W_half_grid|) on the dense grid; "
                           "disc_W_lambda = |W(N_b, v_b) - W(N_a, v_a)|/2; disc = their quadrature sum; pull_total = res/sqrt(floor^2 + disc^2). "
                           "The midpoint rule is third order per pair in the record step; the cumulative disc is the plain sum."),
        'site_arrays': ("The stored site arrays N, v and N_applied, v_applied are not used by the columns. Per record they are compared with the dense solution at "
                        "the sites (max |N_site - N_dense(R_n)| over r < r_c; site against applied; with --check-applied the engine's site_fields_kernel_average "
                        "of the re-solved solution against both), and version 1's refit of their outermost zero of N + v is printed against the dense root "
                        "(R_site - R_dense; also the refit of the dense solution sampled at the sites, which isolates the lattice interpolation error). The "
                        "leading-order displacement of a simple zero under a Gaussian kernel average of width ell, -(ell^2/2)[(N+v)''/(N+v)' + 2/R] in 3D "
                        "(-(ell^2/2)(N+v)''/(N+v)' for a 1D radial average), evaluated on the dense functions, is printed as shift_pred."),
    }
    definitions.update(v3_definitions(opts, stress, h_fine))
    result = {
        'tag': opts.tag,
        'stress_meta': _jsonable((stress or {}).get('meta') or {}),
        'definitions': definitions,
        'thresholds': thr,
        'provenance': prov or {},
        'grid': {'nsites': int(R.size), 'r_first': float(R[0]), 'r_last': float(R[-1]), 'h_site': h_site_eff, 'uniform_sites': h_site is not None,
                 'ell': float(opts.ell), 'h_fine': h_fine, 'n_fine': int(r_fine.size), 'r_fine_last': float(r_fine[-1]), 'fields_keys': F.get('keys')},
        'matching': {'mode': match_mode, 'n_records_json': len(records), 'n_rows_npz': int(T_npz.size), 'n_unmatched': n_unmatched,
                     'n_analysed': len(recs), 'T_first_analysed': float(recs[0]['T_over_rs']), 'T_last_analysed': float(recs[-1]['T_over_rs']),
                     'gap_median_T_over_rs': gap_med},
        'records': recs,
        'pairs': pairs_inc,
        'excluded_pairs': pairs_exc,
        'stats': stats,
        'notes': notes,
        'G': G, 'rs': rs, 'ell': float(opts.ell), 'stop_E': stop_E,
        'seconds': time.time() - t0,
    }
    return result


# ----------------------------------------------------------------------------
# writers
# ----------------------------------------------------------------------------
def _stack_numeric(dicts, prefix):
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
            out[prefix + key] = np.array([NAN if x is None else float(x) for x in vals], dtype=float)
    return out


def _s(st, key, p=3):
    return _fmt(st.get(key), p)


def write_md(path, res, json_path, npz_path):
    st = res['stats']
    pv = res.get('provenance') or {}
    ep = pv.get('engine') or {}
    L = []
    L.append('# FIRSTLAW_records_v3b -- %s' % res['tag'])
    L.append('')
    L.append('## Inputs')
    L.append('- records: `%s`' % json_path)
    L.append('- fields: `%s` (keys %s)' % (npz_path, ', '.join(res['grid'].get('fields_keys') or [])))
    L.append('- %d records in the json, %d rows in the npz, matching by %s, %d unmatched; %d records analysed from T/r_s %s to %s (median gap %s)' % (
        res['matching']['n_records_json'], res['matching']['n_rows_npz'], res['matching']['mode'], res['matching']['n_unmatched'],
        res['matching']['n_analysed'], _fmt(res['matching']['T_first_analysed']), _fmt(res['matching']['T_last_analysed']), _fmt(res['matching']['gap_median_T_over_rs'])))
    L.append('- site grid: %d sites, r from %s to %s, spacing %s (%s); dense grid: spacing %s, %d points to %s; G = %s, ell = %s, r_s = %s, stop_E = %s' % (
        res['grid']['nsites'], _fmt(res['grid']['r_first']), _fmt(res['grid']['r_last']), _fmt(res['grid']['h_site']),
        'uniform' if res['grid']['uniform_sites'] else 'NOT uniform', _fmt(res['grid']['h_fine']), res['grid']['n_fine'], _fmt(res['grid']['r_fine_last']),
        _fmt(res['G']), _fmt(res['ell']), _fmt(res['rs']), _fmt(res['stop_E'])))
    if ep:
        files = '; '.join('%s `%s` sha256 %s' % (k, v.get('path'), (v.get('sha256') or '')[:16] + '...') for k, v in sorted((ep.get('files') or {}).items()))
        L.append('- engine: module %s (%s), imported read-only from `%s` (%s); numpy %s, scipy %s; files: %s' % (
            ep.get('module'), ep.get('module_source'), ep.get('path'), ep.get('read_only'), ep.get('numpy_version'), ep.get('scipy_version'), files or 'none recorded'))
        L.append('- the engine\'s dictionary line: %s' % (ep.get('dictionary') or '-'))
        L.append('- solve_constraints_ivp%s' % (ep.get('solve_signature') or ''))
    sa = pv.get('solver_args') or {}
    if sa:
        L.append('- solve arguments: ' + '; '.join('%s = %s (%s%s)' % (k, v.get('value'), v.get('source'), '' if v.get('passed') else '; NOT passed: ' + str(v.get('note'))) for k, v in sorted(sa.items())))
    if pv.get('site_fields'):
        L.append('- the run\'s site_fields: %s (%s)' % tuple(pv['site_fields']))
    if pv.get('solver_string') is not None:
        L.append('- the npz \'solver\' string: `%s`' % str(pv.get('solver_string')).replace('`', "'"))
    if pv.get('label') is not None:
        L.append('- the npz \'label\': `%s`' % str(pv.get('label')).replace('`', "'"))
    if pv.get('record_solver_keys'):
        L.append('- the records\' solver dict keys: %s' % ', '.join(pv['record_solver_keys']))
    if pv.get('cache'):
        L.append('- cache: %s' % pv['cache'])
    L.append('- radial stress p_r (version 3): %s' % v3_stress_line(res))
    L.append('- re-solves: %d records (%d rooted), %d failed; %s s in total (median %s s per record) with %s workers' % (
        st.get('n_records_resolved', 0), st.get('n_rooted_resolved_ok', 0) + sum(1 for r in res['records'] if r.get('has_root') and r.get('resolved') and not r.get('solve_ok')),
        st.get('n_solve_failed', 0), _fmt(st.get('solve_seconds_total'), 4), _fmt(st.get('solve_seconds_median'), 3), res['thresholds'].get('workers')))
    for n in res['notes']:
        L.append('- NOTE: %s' % n)
    L.append('')
    L.append('## The rows in the solver\'s own normalization')
    L.append('- ' + res['definitions']['rows'])
    L.append('')
    L.append('## Definitions (as computed)')
    for key in ('column_1_dE', 'column_2_kappa_dA', 'column_3_W_ext', 'residual', 'applied_identity_diagnostic', 'row_audit', 'premises', 'floors', 'discretization', 'site_arrays'):
        L.append('- **%s**: %s' % (key, res['definitions'][key]))
    L.append('')
    L.append('## Reproduction of the dense solution (zeros, E, collar)')
    L.append('- %d records re-solved, %d solves failed; of the %d rooted records re-solved successfully, %d reproduce their stored trapping_zeros within %g sites (zero tolerance, never widened); %d records in total reproduce, %d do not' % (
        st.get('n_records_resolved', 0), st.get('n_solve_failed', 0), st.get('n_rooted_resolved_ok', 0), st.get('n_rooted_reproduced', 0), res['thresholds']['zero_tol'],
        st.get('n_zeros_reproduced', 0), st.get('n_zeros_not_reproduced', 0)))
    L.append('- |z_dense - z_stored| over the re-solved records: median %s, max %s sites; own brentq scan against the engine\'s trapping_radius: max %s sites; R_dense - R_stored (outer root): rms %s, max |.| %s' % (
        _s(st, 'max_abs_zero_diff_median'), _s(st, 'max_abs_zero_diff_max'), _s(st, 'max_abs_scan_minus_engine_max'), _s(st, 'R_dense_minus_R_stored_rms'), _s(st, 'R_dense_minus_R_stored_max_abs')))
    L.append('- E_collar(re-solved) - E (relative): median %s, max |.| %s (flag E_mismatch above %g); kappa_eq - kappa_T: rms %s, max |.| %s; N_R - surface_jets.N_t: max |.| %s; charge conservation quadrature (int 4 pi r^2 rho_ell / sum eps_tot - 1): median %s, max %s; source spline error median %s' % (
        _s(st, 'E_collar_rel_diff_median'), _s(st, 'E_collar_rel_diff_max_abs'), res['thresholds']['E_match_rel'], _s(st, 'kappa_eq_minus_kappa_T_rms'), _s(st, 'kappa_eq_minus_kappa_T_max_abs'),
        _s(st, 'N_R_minus_N_t_stored_max_abs'), _s(st, 'charge_conservation_quadrature_median'), _s(st, 'charge_conservation_quadrature_max'), _s(st, 'source_spline_error_median')))
    L.append('- collar deficit (ell) median %s, max %s; nfev median %s; N_centre median %s; re-solved interp floors median N %s, v %s' % (
        _s(st, 'collar_deficit_ell_median'), _s(st, 'collar_deficit_ell_max'), _s(st, 'nfev_median', 6), _s(st, 'N_centre_median', 6), _s(st, 'interp_floor_N_resolved_median'), _s(st, 'interp_floor_v_resolved_median')))
    nr = st.get('records_not_reproduced') or []
    if nr:
        L.append('')
        L.append('| T | stored zeros | dense zeros | differences (dense - stored) | max abs | solve failure | zeros from |')
        L.append('|---|---|---|---|---|---|---|')
        for r in nr:
            L.append('| %s | %s | %s | %s | %s | %s | %s |' % (_fmt(r['T_over_rs']), json.dumps(_jsonable(r.get('zeros_stored'))), json.dumps(_jsonable(r.get('zeros_dense'))),
                                                       json.dumps(_jsonable(r.get('zero_diffs'))), _fmt(r.get('max_abs_zero_diff'), 3), r.get('solve_failed') or '-', r.get('zeros_source') or '-'))
    L.append('')
    L.append('## Row audit on the dense solution (expected at the solver\'s tolerance)')
    L.append('- exterior RMS defects, Richardson differences of the dense functions at h = ell/%g: e (Hamiltonian row) median %s, max %s; g (momentum row) median %s, max %s; W form median %s, max %s; at 2h: e median %s, g median %s; max-norm e median %s; with J_ell -> -J_ell for contrast: e median %s, g median %s; |uJ|/rho (RMS, exterior) median %s' % (
        1.0 / res['thresholds']['fd_h_frac'], _s(st, 'row_e_rel_ext_median'), _s(st, 'row_e_rel_ext_max'), _s(st, 'row_g_rel_ext_median'), _s(st, 'row_g_rel_ext_max'),
        _s(st, 'row_W_rel_ext_median'), _s(st, 'row_W_rel_ext_max'), _s(st, 'row_e_rel_ext_2h_median'), _s(st, 'row_g_rel_ext_2h_median'), _s(st, 'row_e_max_rel_ext_median'),
        _s(st, 'row_e_rel_ext_flipJ_median'), _s(st, 'row_g_rel_ext_flipJ_median'), _s(st, 'uJ_over_rho_rms_median')))
    L.append('- the engine\'s own residuals (max norm over its fine grid): residual_W median %s, max %s; residual_invN median %s, max %s' % (
        _s(st, 'residual_W_median'), _s(st, 'residual_W_max'), _s(st, 'residual_invN_median'), _s(st, 'residual_invN_max')))
    L.append('- for the record, the same rows at the sites (4th-order stencil, spacing %s): the stored site arrays with version 1\'s raw reading (eps_tot, J_tot as densities): e median %s (J flipped %s), g median %s; the site arrays with the smoothed sources rho_ell(R_n), J_ell(R_n): e median %s, g median %s; the applied arrays: e median %s, g median %s; the dense solution sampled at the sites: e median %s, g median %s (the stencil\'s own error at spacing %s)%s' % (
        _fmt(res['grid']['h_site']), _s(st, 'site_row_e_rel_ext_raw_median'), _s(st, 'site_row_e_rel_ext_raw_flipJ_median'), _s(st, 'site_row_g_rel_ext_raw_median'),
        _s(st, 'site_row_e_rel_ext_median'), _s(st, 'site_row_g_rel_ext_median'), _s(st, 'applied_row_e_rel_ext_median'), _s(st, 'applied_row_g_rel_ext_median'),
        _s(st, 'dense_at_sites_row_e_rel_ext_median'), _s(st, 'dense_at_sites_row_g_rel_ext_median'), _fmt(res['grid']['h_site']),
        ('; the kernel average of the dense solution: e median %s, g median %s' % (_s(st, 'kernel_row_e_rel_ext_median'), _s(st, 'kernel_row_g_rel_ext_median'))) if np.isfinite(_float(st.get('kernel_row_e_rel_ext_median'))) else ''))
    L.append('')
    L.append('## The stored site arrays against the dense solution')
    L.append('- max |N_site - N_dense(R_n)| over r < r_c: median %s, max %s; |v_site - v_dense|: median %s, max %s; N_applied against dense: median %s, max %s; v: median %s, max %s; site against applied: N max %s, v max %s' % (
        _s(st, 'max_abs_N_site_minus_dense_median'), _s(st, 'max_abs_N_site_minus_dense_max'), _s(st, 'max_abs_v_site_minus_dense_median'), _s(st, 'max_abs_v_site_minus_dense_max'),
        _s(st, 'max_abs_N_applied_minus_dense_median'), _s(st, 'max_abs_N_applied_minus_dense_max'), _s(st, 'max_abs_v_applied_minus_dense_median'), _s(st, 'max_abs_v_applied_minus_dense_max'),
        _s(st, 'max_abs_N_site_minus_applied_max'), _s(st, 'max_abs_v_site_minus_applied_max')))
    L.append('- the outermost zero of N + v refit from the arrays (version 1\'s reading) minus the dense root: site arrays median %s (min %s, max %s); applied arrays median %s; the dense solution sampled at the sites median %s (max |.| %s: the lattice interpolation error alone); predicted kernel displacement -(ell^2/2)[(N+v)\'\'/(N+v)\' + 2/R] median %s (1D form median %s)%s' % (
        _s(st, 'R_site_minus_R_dense_median'), _s(st, 'R_site_minus_R_dense_min'), _s(st, 'R_site_minus_R_dense_max'), _s(st, 'R_applied_minus_R_dense_median'),
        _s(st, 'R_densesites_refit_minus_R_dense_median'), _s(st, 'R_densesites_refit_minus_R_dense_max_abs'), _s(st, 'shift_pred_3d_median'), _s(st, 'shift_pred_1d_median'),
        ('; the engine\'s kernel average of the re-solved solution: root minus dense root median %s, |N_site - N_kernel| max %s, |N_applied - N_kernel| max %s, |v_applied - v_kernel| max %s' % (
            _s(st, 'R_kernel_minus_R_dense_median'), _s(st, 'max_abs_N_site_minus_kernel_max'), _s(st, 'max_abs_N_applied_minus_kernel_max'), _s(st, 'max_abs_v_applied_minus_kernel_max')))
        if np.isfinite(_float(st.get('R_kernel_minus_R_dense_median'))) else ' (run with --check-applied for the engine\'s kernel average)'))
    L.append('')
    L.append('## Headline (included pairs, in order)')
    if st.get('n_pairs_included', 0) > 0:
        L.append('- %d pairs included, T/r_s %s to %s; %d pairs listed as excluded; %d pairs with no root at either end%s' % (
            st['n_pairs_included'], _fmt(st['T_first']), _fmt(st['T_last']), st['n_pairs_excluded_listed'], st['n_pairs_no_root'],
            (' (T/r_s %s to %s)' % (_fmt(st['no_root_T_span'][0]), _fmt(st['no_root_T_span'][1]))) if st.get('no_root_T_span') else ''))
        L.append('- sum delta E = %s (E_last - E_first over the included span = %s; E from %s to %s; R from %s to %s); sum delta E_collar (re-solved) = %s' % (
            _fmt(st['sum_dE']), _fmt(st['E_last_minus_E_first']), _fmt(st['E_first'], 9), _fmt(st['E_last'], 9), _fmt(st['R_first']), _fmt(st['R_last']), _fmt(st['sum_dE_collar'])))
        L.append('- sum (kappa/8 pi G) delta A = %s  (kappa column: %s; equations %s; fd %s; formula %s; kappa_T %s; midpoint fields %s; vacuum 1/(4GE) %s)' % (
            _fmt(st['sum_col2']), res['thresholds']['kappa_column'], _fmt(st['sum_col2_eq']), _fmt(st['sum_col2_fd']), _fmt(st['sum_col2_formula']), _fmt(st['sum_col2_T']),
            _fmt(st['sum_col2_midfields']), _fmt(st['sum_col2_vac'])))
        L.append('- sum W_ext = %s  (Simpson %s, half grid %s; N delta rho part %s, v delta J part %s; pairs part %s, dust part %s)' % (
            _fmt(st['sum_W_ext']), _fmt(st['sum_W_simpson']), _fmt(st['sum_W_coarse']), _fmt(st['sum_W_Nrho']), _fmt(st['sum_W_vJ']), _fmt(st['sum_W_pairs']), _fmt(st['sum_W_dust'])))
        L.append('- sum residual = %s; cumulative floor %s (pull %s); cumulative discretization %s (pull_total %s); sum res / sum delta E = %s; with delta E_collar: sum res_collarE = %s' % (
            _fmt(st['sum_res']), _fmt(st['floor_cum']), _fmt(st['cum_pull'], 3), _fmt(st['disc_cum']), _fmt(st['cum_pull_total'], 3), _fmt(st['sum_res_over_sum_dE'], 3), _fmt(st['sum_res_collarE'])))
        L.append('- W_ext against delta E: sum W / sum delta E = %s; per pair W/delta E median %s (min %s, max %s); (kappa/8 pi G) delta A / delta E median %s' % (
            _fmt(st['sum_W_over_sum_dE'], 4), _fmt(st['W_over_dE_median'], 4), _fmt(st['W_over_dE_min'], 4), _fmt(st['W_over_dE_max'], 4), _fmt(st['col2_over_dE_median'], 4)))
        L.append('- the no-work premise: sum (delta E - (kappa/8 pi G) delta A) = %s; the vacuum first law: sum (delta E - delta A/(32 pi G^2 E)) = %s' % (
            _fmt(st['sum_res_noW']), _fmt(st['sum_res_vac'])))
        L.append('- applied identity (row defects retained): sum defect integral = %s, sum res_applied = %s (rms %s, max |.| %s)' % (
            _fmt(st['sum_defect_integral']), _fmt(st['sum_res_applied']), _fmt(st['res_applied_rms']), _fmt(st['res_applied_max_abs'])))
    else:
        L.append('- no pair included; see the excluded pairs and the reproduction section')
    if st.get('n_rooted_resolved_ok', 0) > 0:
        L.append('- kappa: mean kappa_T %s, mean kappa_eq %s, mean kappa_fd %s, mean kappa_formula %s; |kappa_eq - kappa_T| max %s; |kappa_fd - kappa_eq| max %s; |kappa_formula - kappa_eq| max %s' % (
            _fmt(st.get('kappa_T_mean')), _fmt(st.get('kappa_eq_mean')), _fmt(st.get('kappa_fd_mean')), _fmt(st.get('kappa_formula_mean')),
            _s(st, 'kappa_eq_minus_kappa_T_max_abs'), _s(st, 'kappa_fd_minus_kappa_eq_max_abs'), _s(st, 'kappa_formula_minus_kappa_eq_max_abs')))
        L.append('- 4 G E kappa_eq from %s to %s; R/(2GE) from %s to %s; simple margin s = 2 R kappa_eq/N_R from %s to %s' % (
            _fmt(st.get('four_GE_kappa_min'), 4), _fmt(st.get('four_GE_kappa_max'), 4), _fmt(st.get('R_over_2GE_min'), 4), _fmt(st.get('R_over_2GE_max'), 4),
            _fmt(st.get('simple_margin_eq_min'), 4), _fmt(st.get('simple_margin_eq_max'), 4)))
        L.append('- floors (medians over rooted records): sigma_E %s (stop %s, screen %s, tail %s); sigma_R %s sites (stored %s, scan %s)' % (
            _s(st, 'sigma_E_median'), _s(st, 'sigma_E_stop_median'), _s(st, 'sigma_E_screen_median'), _s(st, 'sigma_E_tail_median'),
            _s(st, 'sigma_R_median'), _s(st, 'sigma_R_stored_median'), _s(st, 'sigma_R_scan_median')))
    L.extend(v3_md_summary(res))
    L.append('')
    L.append('## Closure statistics (included pairs)')
    if st.get('n_pairs_included', 0) > 0:
        L.append('| quantity | mean | rms | max abs | at T_b |')
        L.append('|---|---|---|---|---|')
        for key, lab in (('res', 'res (primary kappa)'), ('res_eq', 'res with kappa_eq'), ('res_fd', 'res with kappa_fd'), ('res_formula', 'res with kappa_formula'),
                         ('res_T', 'res with kappa_T'), ('res_midfields', 'res with midpoint-field kappa'), ('res_simpson', 'res with Simpson W'),
                         ('res_collarE', 'res with delta E_collar'), ('res_applied', 'res_applied (defects retained)'),
                         ('res_noW', 'delta E - (kappa/8 pi G) delta A'), ('res_vac', 'vacuum first law residual')):
            L.append('| %s | %s | %s | %s | %s |' % (lab, _fmt(st.get(key + '_mean')), _fmt(st.get(key + '_rms')), _fmt(st.get(key + '_max_abs')), _fmt(st.get(key + '_max_abs_T_b'))))
        L.append('')
        L.append('- rms: delta E %s, (kappa/8 pi G) delta A %s, W_ext %s; res/delta E (rms) %s; res/W (rms) %s' % (
            _fmt(st['dE_rms']), _fmt(st['col2_rms']), _fmt(st['W_ext_rms']), _fmt(st['res_over_dE_rms'], 3), _fmt(st['res_over_W_rms'], 3)))
        L.append('- pull = res/floor: rms %s; fraction |pull| < 1: %s, < 2: %s, < 3: %s; pull_total (floor (+) disc): rms %s; < 1: %s, < 2: %s, < 3: %s' % (
            _fmt(st['pull_rms'], 3), _fmt(st['frac_abs_pull_lt_1'], 3), _fmt(st['frac_abs_pull_lt_2'], 3), _fmt(st['frac_abs_pull_lt_3'], 3),
            _fmt(st['pull_total_rms'], 3), _fmt(st['frac_abs_pull_total_lt_1'], 3), _fmt(st['frac_abs_pull_total_lt_2'], 3), _fmt(st['frac_abs_pull_total_lt_3'], 3)))
        L.append('- medians: |delta E| %s, |(kappa/8 pi G) delta A| %s, |W_ext| %s, floor %s, disc %s (disc_W_rule %s, disc_kappa %s); W_pairs/W median %s' % (
            _fmt(st['dE_abs_median']), _fmt(st['col2_abs_median']), _fmt(st['W_ext_abs_median']), _fmt(st['floor_median']), _fmt(st['disc_median']),
            _fmt(st['disc_W_rule_median']), _fmt(st['disc_kappa_median']), _fmt(st['W_pairs_over_W_median'], 3)))
    L.append('')
    L.append('## Excluded pairs')
    if res['excluded_pairs']:
        L.append('| T_a | T_b | R_a | R_b | reasons | other flags |')
        L.append('|---|---|---|---|---|---|')
        for pe in res['excluded_pairs']:
            L.append('| %s | %s | %s | %s | %s | %s |' % (_fmt(pe['T_a']), _fmt(pe['T_b']), _fmt(pe.get('R_a')), _fmt(pe.get('R_b')),
                                                    '; '.join(pe['reasons']), '; '.join(pe.get('other_flags') or [])))
        L.append('')
        L.append('- reason histogram: %s' % json.dumps(st.get('excluded_reason_histogram', {})))
    else:
        L.append('- none listed (pairs with no root at either end: %d)' % st.get('n_pairs_no_root', 0))
    L.append('')
    L.append('## Per-pair table')
    L.append('kappa column = %s. Columns: delta E; (kappa/8 pi G) delta A; W_ext (trapezoid); W_pairs; res; floor; pull; disc; W/delta E; kappa_eq(b); kappa_T(b); their difference at b; res with delta E_collar; the applied residual; flags.' % res['thresholds']['kappa_column'])
    L.append('')
    L.append('| T_a | T_b | R_mid | dR | dE | k dA/8piG | W_ext | W_pairs | res | floor | pull | disc | W/dE | kappa_eq(b) | kappa_T(b) | k_eq-k_T(b) | res_collarE | res_applied | flags |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for p in res['pairs']:
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            _fmt(p['T_a']), _fmt(p['T_b']), _fmt(p['R_mid'], 7), _fmt(p['dR'], 4), _fmt(p['dE'], 5), _fmt(p['col2'], 5), _fmt(p['W_ext'], 5), _fmt(p['W_pairs'], 4),
            _fmt(p['res'], 4), _fmt(p['floor'], 3), _fmt(p['pull'], 3), _fmt(p['disc'], 3), _fmt(p['W_over_dE'], 4), _fmt(p['kappa_eq_mean'] * 2 - p['kappa_eq_mean'], 6) if False else _fmt(p['kappa_b'], 6),
            _fmt(p['kappa_T_b'], 6), _fmt(p['kappa_b'] - p['kappa_T_b'], 3), _fmt(p['res_collarE'], 4), _fmt(p['res_applied'], 4), '; '.join(p.get('flags') or [])))
    L.append('')
    every = max(1, int(res.get('md_every', 8)))
    L.append('## Cumulative (every %d-th included pair and the last)' % every)
    L.append('')
    L.append('| T_b | sum dE | E_b - E_first | sum dE_collar | sum k dA/8piG | sum W_ext | sum W_pairs | sum res | sum res_T | sum res_collarE | sum res_applied | sum res_noW | sum res_vac |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    npairs = len(res['pairs'])
    for j, p in enumerate(res['pairs']):
        if j % every == every - 1 or j == npairs - 1:
            L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
                _fmt(p['T_b']), _fmt(p['cum_dE'], 6), _fmt(p['E_b_minus_E_first'], 6), _fmt(p['cum_dE_collar'], 6), _fmt(p['cum_col2'], 6), _fmt(p['cum_W_ext'], 6),
                _fmt(p['cum_W_pairs'], 5), _fmt(p['cum_res'], 4), _fmt(p['cum_res_T'], 4), _fmt(p['cum_res_collarE'], 4), _fmt(p['cum_res_applied'], 4),
                _fmt(p['cum_res_noW'], 5), _fmt(p['cum_res_vac'], 5)))
    L.append('')
    L.append('## Per-record table (records with a stored root)')
    L.append('')
    L.append('| T | R_dense | R_dense-R_stored | max|dz| | N_R | rho_ell(R) | J_ell(R) | s | kappa_eq | k_fd-k_eq | k_formula-k_eq | kappa_T | k_eq-k_T | E_collar-E | e_rel_ext | g_rel_ext | R_site-R | R_app-R | shift_pred | sigma_E | seconds | flags |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in res['records']:
        if not r.get('has_root'):
            continue
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            _fmt(r['T_over_rs']), _fmt(r['R_dense'], 9), _fmt(r['R_dense_minus_R_stored'], 3), _fmt(r['max_abs_zero_diff'], 3), _fmt(r['N_R'], 7), _fmt(r['rho_R'], 4), _fmt(r['J_R'], 4),
            _fmt(r['simple_margin_eq'], 5), _fmt(r['kappa_eq'], 7), _fmt(r['kappa_fd_minus_kappa_eq'], 3), _fmt(r['kappa_formula_minus_kappa_eq'], 3), _fmt(r['kappa_T_stored'], 7),
            _fmt(r['kappa_eq_minus_kappa_T'], 3), _fmt(r['E_collar_minus_E'], 3), _fmt(r['row_e_rel_ext'], 3), _fmt(r['row_g_rel_ext'], 3), _fmt(r['R_site_minus_R_dense'], 3),
            _fmt(r['R_applied_minus_R_dense'], 3), _fmt(r['shift_pred_3d'], 3), _fmt(r['sigma_E'], 3), _fmt(r['solve_seconds'], 3), '; '.join(r.get('flags') or [])))
    L.extend(v3_md_tables(res))
    L.append('')
    L.append('## Thresholds (named arguments)')
    for key, val in sorted(res['thresholds'].items()):
        L.append('- %s = %s' % (key, val))
    L.append('')
    L.append('- seconds (analysis after the solves): %s' % _fmt(res['seconds'], 4))
    with open(path, 'w') as fh:
        fh.write('\n'.join(L) + '\n')


def write_outputs(res, r_fine, out_dir, tag, json_path, npz_path, opts):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, 'firstlaw_records_v3b_%s' % tag)
    res['md_every'] = int(opts.md_every)
    res['thresholds']['md_every'] = int(opts.md_every)
    arrays = {'r_fine': np.asarray(r_fine, float)}
    arrays.update(_stack_numeric(res['pairs'], 'pair_'))
    arrays.update(_stack_numeric(res['records'], 'rec_'))
    if res['pairs']:
        nf = r_fine.size
        fm = np.full((len(res['pairs']), nf), np.nan)
        fd = np.full((len(res['pairs']), nf), np.nan)
        for i, p in enumerate(res['pairs']):
            fm[i, :p['_f_mid'].size] = p['_f_mid']
            fd[i, :p['_f_defect'].size] = p['_f_defect']
        arrays['pair_f_mid'] = fm
        arrays['pair_f_defect'] = fd
    arrays.update(v3_output_arrays(res, r_fine))
    arrays['options_json'] = np.array(json.dumps(_jsonable(vars(opts))))
    arrays['stats_json'] = np.array(json.dumps(_jsonable(res['stats'])))
    np.savez(base + '.npz', **arrays)
    res_json = dict(res)
    res_json['inputs'] = {'json': json_path, 'npz': npz_path, 'options': vars(opts)}
    res_json['pairs'] = [{k: v for k, v in p.items() if not k.startswith('_')} for p in res['pairs']]
    with open(base + '.json', 'w') as fh:
        json.dump(_jsonable(res_json), fh, indent=1)
    write_md(base + '.md', res, json_path, npz_path)
    return base


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main(argv=None):
    opts = parse_args(argv)
    if opts.json is None:
        raise SystemExit('--json is required (collapse_<tag>.json)')
    json_path = os.path.abspath(opts.json)
    npz_path = os.path.abspath(opts.npz) if opts.npz else (json_path[:-5] + '_fields.npz' if json_path.endswith('.json') else json_path + '_fields.npz')
    t0 = time.time()
    head, run_args, records, rec_key = load_json_records(json_path)
    F = load_fields(npz_path)
    notes = []
    fill_defaults(opts, run_args, F, notes)
    out_dir = opts.out or os.path.join(os.path.dirname(json_path), 'firstlaw_v2')
    os.makedirs(out_dir, exist_ok=True)
    rec_solver = first_record_solver_dict(records)
    solver_parsed = parse_solver_string(F.get('solver_string'))
    module_name, module_src = resolve_engine_module(opts, rec_solver, solver_parsed)
    engine_path = opts.engine_path or os.path.dirname(os.path.abspath(__file__))
    FLD, eprov = import_engine(engine_path, module_name)
    eprov['module_source'] = module_src
    sargs, aspec, site_fields = resolve_solver_args(FLD, run_args, rec_solver, solver_parsed, opts)
    G = float(opts.G)
    ell = float(opts.ell)
    ell_pass = ell_for_engine(ell)
    R = np.asarray(F['R'], float)
    r_fine, r_last_node, gnote = build_fine_grid(FLD, R, ell_pass, opts.fine)
    if gnote:
        notes.append(gnote)
    h_fine = ell / float(opts.fine)
    cfg = worker_cfg(opts, r_last_node, h_fine)
    rows, match_mode, n_unmatched = match_rows(records, F['T'], float(opts.rs), opts.T_match_tol)
    idx = [k for k in range(len(records)) if rows[k] >= 0]
    idx.sort(key=lambda k: _tors(records[k], float(opts.rs)))
    if opts.T_start is not None:
        idx = [k for k in idx if _tors(records[k], float(opts.rs)) >= opts.T_start - 1e-9]
    if opts.T_end is not None:
        idx = [k for k in idx if _tors(records[k], float(opts.rs)) <= opts.T_end + 1e-9]
    if len(idx) < 2:
        raise SystemExit('fewer than 2 analysable records (matched rows within the T window); matching by %s, %d unmatched' % (match_mode, n_unmatched))
    rooted = set(k for k in idx if (records[k].get(opts.zeros_key) or []))
    if opts.resolve == 'all':
        todo = list(idx)
    elif opts.resolve == 'none':
        todo = []
    else:
        pos = {k: i for i, k in enumerate(idx)}
        sel = set()
        for k in rooted:
            i = pos[k]
            for j in (i - 1, i, i + 1):
                if 0 <= j < len(idx):
                    sel.add(idx[j])
        todo = [k for k in idx if k in sel]
    if opts.max_records is not None:
        todo = todo[:max(0, int(opts.max_records))]
    ckey = cache_key_of(sargs, eprov, cfg, G, ell)
    cache_file = None if opts.no_cache else os.path.join(out_dir, 'firstlaw_dense_v3b_%s_cache.npz' % opts.tag)
    if opts.recompute or cache_file is None:
        cached, cmsg = {}, ('recompute requested' if opts.recompute else 'cache disabled')
    else:
        cached, cmsg = load_cache(cache_file, ckey, r_fine)
    seeded, smsg = v3_seed_cache(opts, ckey, r_fine, sargs, module_name)
    if opts.seed_cache:
        cmsg += '; seed cache %s: %s' % (opts.seed_cache, smsg)
    dense_by_row = dict(seeded)
    dense_by_row.update(cached)
    tasks = [(k, int(rows[k])) for k in todo if int(rows[k]) not in dense_by_row]
    print('FIRSTLAW_records_v3b %s: %d records matched (by %s), %d rooted, %d to re-solve (%d from the cache: %s); engine %s (%s) from %s; solve args %s; workers %d, threads/process %d' % (
        opts.tag, len(idx), match_mode, len(rooted), len(tasks), len(todo) - len(tasks), cmsg, module_name, module_src, eprov['module_file'],
        json.dumps(_jsonable(sargs)), WORKERS, THREADS_PER_PROCESS))
    if F.get('solver_string'):
        print("npz 'solver' string: %s" % F['solver_string'])

    def progress(i, n, k, d):
        rec = records[k]
        zs = sorted(float(z) for z in (rec.get(opts.zeros_key) or []) if z is not None)
        T = _tors(rec, float(opts.rs))
        if not d.get('ok'):
            print('[%d/%d] T/r_s %s: SOLVE FAILED: %s (%.1f s)' % (i + 1, n, _fmt(T), d.get('failed'), _float(d.get('seconds'))))
            return
        zd = sorted(d.get('zeros_dense') or [])
        if len(zd) == len(zs):
            dz = max([abs(a - b) for a, b in zip(zd, zs)] or [0.0])
            rep = 'max|dz| %s' % _fmt(dz, 3)
        else:
            rep = 'COUNT %d vs stored %d' % (len(zd), len(zs))
        root = d.get('root') or {}
        au = d.get('audit') or {}
        print('[%d/%d] T/r_s %s: zeros %s vs stored %s (%s); R %s kappa_eq %s (kappa_T %s, fd-eq %s); E_collar - E %s; audit e %s g %s; %.1f s' % (
            i + 1, n, _fmt(T), json.dumps([round(z, 6) for z in zd]), json.dumps([round(z, 6) for z in zs]), rep, _fmt(root.get('R'), 8), _fmt(root.get('kappa_eq'), 7),
            _fmt(rec.get(opts.kappaT_key), 7), _fmt(_float(root.get('kappa_fd')) - _float(root.get('kappa_eq')), 2), _fmt(_float(d.get('E_collar')) - _float(rec.get(opts.E_key)), 3),
            _fmt(au.get('rel_e'), 2), _fmt(au.get('rel_g'), 2), _float(d.get('seconds'))))

    results, secs = compute_dense(FLD, F, G, ell_pass, sargs, r_fine, cfg, tasks, WORKERS, progress=progress)
    for k, d in results.items():
        dense_by_row[int(d['row'])] = d
    if cache_file is not None and (results or seeded):
        try:
            save_cache(cache_file, ckey, r_fine, dense_by_row)
            cmsg += '; written: %s (%d records)' % (cache_file, len(dense_by_row))
        except Exception as exc:
            notes.append('the cache could not be written (%s: %s)' % (type(exc).__name__, exc))
    dense = {k: dense_by_row[int(rows[k])] for k in idx if int(rows[k]) in dense_by_row}
    stress, stress_secs = v3_stress_pass(FLD, F, opts, dense, rows, idx, r_fine, r_last_node, sargs, ell_pass, notes)
    prov = {'engine': eprov, 'solver_args': aspec, 'site_fields': site_fields, 'solver_string': F.get('solver_string'), 'label': F.get('label'),
            'record_solver_keys': sorted(rec_solver.keys()) if rec_solver else [], 'record_list_key': rec_key, 'cache': cmsg + ('' if opts.no_cache else '; key %s' % ckey[:16]),
            'solve_wall_seconds': secs, 'cfg': cfg}
    res = analyse(records, F, run_args, opts, dense, r_fine, prov=prov, notes=notes, stress=stress)
    res['record_list_key'] = rec_key
    base = write_outputs(res, r_fine, out_dir, opts.tag, json_path, npz_path, opts)
    st = res['stats']
    print('FIRSTLAW_records_v3b %s: %d records analysed, %d with a root, %d re-solved (%d failed), %d rooted reproduce their zeros within %g sites, %d pairs included, %d excluded listed, %d without a root' % (
        opts.tag, st['n_records_analysed'], st['n_records_with_root'], st['n_records_resolved'], st['n_solve_failed'], st['n_rooted_reproduced'], opts.zero_tol,
        st['n_pairs_included'], st['n_pairs_excluded_listed'], st['n_pairs_no_root']))
    print('ROW AUDIT (dense, exterior, Richardson h = ell/%g): e rel RMS median %s max %s; g median %s max %s; W form median %s; J flipped: e %s g %s; engine residual_W median %s, residual_invN median %s' % (
        1.0 / opts.fd_h_frac, _s(st, 'row_e_rel_ext_median'), _s(st, 'row_e_rel_ext_max'), _s(st, 'row_g_rel_ext_median'), _s(st, 'row_g_rel_ext_max'), _s(st, 'row_W_rel_ext_median'),
        _s(st, 'row_e_rel_ext_flipJ_median'), _s(st, 'row_g_rel_ext_flipJ_median'), _s(st, 'residual_W_median'), _s(st, 'residual_invN_median')))
    print('site arrays: the raw reading of version 1 gives e %s (J flipped %s); the smoothed sources at the sites e %s; |N_site - N_dense| max %s, |N_site - N_applied| max %s; R_site - R_dense median %s (predicted kernel shift %s), R(dense at sites) - R_dense max |.| %s' % (
        _s(st, 'site_row_e_rel_ext_raw_median'), _s(st, 'site_row_e_rel_ext_raw_flipJ_median'), _s(st, 'site_row_e_rel_ext_median'), _s(st, 'max_abs_N_site_minus_dense_max'),
        _s(st, 'max_abs_N_site_minus_applied_max'), _s(st, 'R_site_minus_R_dense_median'), _s(st, 'shift_pred_3d_median'), _s(st, 'R_densesites_refit_minus_R_dense_max_abs')))
    if st['n_pairs_included'] > 0:
        print('sum dE %s | sum (kappa/8piG) dA %s | sum W_ext %s | sum res %s (floor %s, disc %s) | sum res_collarE %s | sum res_applied %s' % (
            _fmt(st['sum_dE']), _fmt(st['sum_col2']), _fmt(st['sum_W_ext']), _fmt(st['sum_res']), _fmt(st['floor_cum']), _fmt(st['disc_cum']), _fmt(st['sum_res_collarE']), _fmt(st['sum_res_applied'])))
        print('W/dE: sum ratio %s, median per pair %s | pull rms %s, |pull|<2 fraction %s | kappa_eq - kappa_T max %s, kappa_fd - kappa_eq max %s, E_collar - E rel max %s' % (
            _fmt(st['sum_W_over_sum_dE'], 4), _fmt(st['W_over_dE_median'], 4), _fmt(st['pull_rms'], 3), _fmt(st['frac_abs_pull_lt_2'], 3),
            _s(st, 'kappa_eq_minus_kappa_T_max_abs'), _s(st, 'kappa_fd_minus_kappa_eq_max_abs'), _s(st, 'E_collar_rel_diff_max_abs')))
    for n in res['notes']:
        print('NOTE: ' + n)
    v3_console(res, stress_secs)
    print('wrote %s.{json,npz,md} in %.1f s (solves %.1f s)' % (base, time.time() - t0, secs))
    v3_check_against(opts, base + '.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
