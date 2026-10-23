#!/usr/bin/env python3
"""
firstlaw_records_v1.py -- FIRSTLAW_records_v1 (reorientation 4, C15; solve entry item 8)

The first-law identity of FIRSTLAW_rows_v1 (hreduce), evaluated between
consecutive stored records that both carry a simple dense root of N + v:

    delta E  =  (kappa_R / (8 pi G)) delta A  +  W_ext

    kappa_R  =  (N_R / 2R) [ 1 - 8 pi G R^2 (rho_R - J_R) ]
                (input 02's formula; it equals the dense collar slope d_r(N+v)|_R
                 exactly when the stored fields solve the declared rows)
    A        =  4 pi R^2 at the outer dense root R
    W_ext    =  4 pi int_R^{r_c} r^2 ( N delta rho + v delta J ) dr
                (Eulerian delta at fixed r; the averaged, normal-ordered sources;
                 from the root to the collar r_c where E is read)

Column 1: the change of the collar charge, record key 'E' (the Misner-Sharp
mass M = r u^2 / 2G, u = v/N, at r_collar).  Column 2: the stored dense root (the
outermost entry of 'trapping_zeros') and kappa_R from the stored N, rho, J
interpolated at R; the stored kappa_T is printed beside it with the difference.
Column 3: the stored averaged sources (eps_tot, J_tot) outside the root.

Finite differences between two records are closed with the midpoint rule in the
record index: the fields and kappa averaged over the two records, the lower limit
at the mean root.  The endpoint evaluations (kappa at each record, the sources
weighted with each record's own N, v) and a second radial quadrature rule
(Simpson, and the half grid) are printed as the discretization error.

Diagnostic beside the 3 columns: the applied identity of input 02 with the row
defects retained,
    N_c delta M_c = (kappa_dense/8 pi G) delta A + W_ext + int_R^{r_c} N (delta e + g delta M) dr,
    e = M' - 4 pi r^2 (rho + u J),  g = N'/N + 4 pi G r J / u,
whose residual closes up to the discretization even when the stored fields do
not solve the stored rows exactly.  The row defects also fix the sign convention
of the stored J (the auto J-sign audit) and predict kappa_T - kappa_formula =
-N_R G e_R / R, which is printed as a check.

Outputs in --out: firstlaw_records_<tag>.json, .npz (every per-pair and
per-record array), .md (definitions, row audit, per-record table, per-pair
table, cumulative table, closure statistics, excluded pairs).

No plotting.  Every threshold is a named argument.  CPU only; at most 8 threads.
Also importable: analyse(records, fields, run_args, opts) works on in-memory
records (the self-test uses it).
"""
import os
import sys


def _preparse_threads(argv):
    n = 8
    for i, a in enumerate(argv):
        if a == '--threads' and i + 1 < len(argv):
            try:
                n = int(argv[i + 1])
            except ValueError:
                pass
        elif a.startswith('--threads='):
            try:
                n = int(a.split('=', 1)[1])
            except ValueError:
                pass
    n = max(1, min(n, 8))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(n)
    return n


THREADS = _preparse_threads(sys.argv[1:])

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

FOUR_PI = 4.0 * np.pi
if hasattr(np, 'trapezoid'):
    _trapz = np.trapezoid
else:
    _trapz = np.trapz


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def _float(x):
    try:
        if x is None:
            return float('nan')
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def _tors(rec, rs):
    t = rec.get('T_over_rs')
    if t is None:
        t = rec.get('T')
        return float('nan') if t is None else float(t) / rs
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
        return float('nan')
    return float(func(arr[np.isfinite(arr)]))


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if not str(k).startswith('_')}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    return o


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
    """d/dr on the site grid: 4th-order centred stencil (2nd order at the edges) for a uniform
    grid, np.gradient otherwise."""
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
    """Interpolating polynomial of the given degree through the degree+1 sites nearest x0
    (centred on x0); returns (coef, centre, scale) in the scaled variable t = (x - centre)/scale."""
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
    """(value, d/dr) of the site array y at x0 from the local interpolating polynomial."""
    coef, xc, sc = local_poly_fit(R, y, x0, degree)
    return local_poly_eval(coef, xc, sc, x0)


def refine_zero(R, f, i_lo, degree, maxit=80):
    """f changes sign between sites i_lo and i_lo+1.  Returns (root, slope, bracket_ok) from the
    local interpolating polynomial (bisection on the polynomial inside the bracket, Newton polish)."""
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
    """All sign changes of f = N + v on the site grid above r_min, refined; type 'outer' if the slope
    is positive (trapped inside, untrapped outside), 'inner' if negative."""
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
    """Sign-preserving local extrema of f with |f| < tol above r_min (a double root's grid signature)."""
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
    """Composite Simpson on a uniform grid; an even number of points gets a quadratic end cell."""
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
    """int_{r_lo}^{R[-1]} f dr with f on the sites.  The partial cell [r_lo, next site] by the
    trapezoid with f(r_lo) from the local polynomial; the rest by 3 rules: 'trapz', 'simpson'
    (uniform grid only), 'coarse' (trapezoid on every other site).  Returns (dict, i0)."""
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
    if h is not None:
        rules['simpson'] = part + simpson_uniform(y, h)
    else:
        rules['simpson'] = float('nan')
    xc = x[::2]
    yc = y[::2]
    coarse = float(_trapz(yc, xc)) if xc.size >= 2 else 0.0
    if x.size >= 2 and (x.size - 1) % 2 == 1:
        coarse += 0.5 * (float(y[-2]) + float(y[-1])) * (float(x[-1]) - float(x[-2]))
    rules['coarse'] = part + coarse
    return rules, i0


def row_defects(R, N, v, rho, J, G, h):
    """The two declared rows' defects on the stored arrays:
       e = M' - 4 pi r^2 (rho + u J),  g = N'/N + 4 pi G r J / u,  u = v/N, M = r u^2 / 2G."""
    u = v / N
    M = R * u * u / (2.0 * G)
    Mp = derivative(M, R, h)
    Np = derivative(N, R, h)
    e = Mp - FOUR_PI * R * R * (rho + u * J)
    with np.errstate(divide='ignore', invalid='ignore'):
        g = Np / N + FOUR_PI * G * R * J / u
    g = np.where(np.isfinite(g), g, np.nan)
    return e, g, M, Mp, Np


def kappa_formula(N_R, R, rho_R, J_R, G):
    return N_R / (2.0 * R) * (1.0 - 8.0 * np.pi * G * R * R * (rho_R - J_R))


# ----------------------------------------------------------------------------
# options
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description='FIRSTLAW_records_v1: the first-law identity between consecutive rooted records')
    p.add_argument('--json', default=None, help='collapse_<tag>.json (the record history)')
    p.add_argument('--npz', default=None, help='collapse_<tag>_fields.npz (default: --json with .json replaced by _fields.npz)')
    p.add_argument('--out', default=None, help='output directory (default: <dir of --json>/firstlaw)')
    p.add_argument('--tag', default=None, help='output tag (default: the run args\' tag, else the json basename)')
    p.add_argument('--threads', type=int, default=8, help='BLAS/OpenMP threads, capped at 8')
    p.add_argument('--fields', choices=('dense', 'applied'), default='dense', help="which stored fields carry the root: 'dense' (N, v) or 'applied' (N_applied, v_applied)")
    p.add_argument('--J-sign', dest='J_sign', choices=('auto', '+1', '-1'), default='auto', help="sign s with J = s * J_tot matching input 02's rows; 'auto' decides from the exterior row defects")
    p.add_argument('--kappa-column', dest='kappa_column', choices=('formula', 'kappa_T', 'dense', 'midfields'), default='formula', help='which kappa enters the primary column 2 and residual (all variants are printed)')
    p.add_argument('--G', type=float, default=None, help="Newton constant (default: npz 'g', else run args 'g')")
    p.add_argument('--rs', type=float, default=None, help="r_s in lattice units (default: run args 'rs')")
    p.add_argument('--stop-E', dest='stop_E', type=float, default=None, help="the solver's relative E stop (default: run args 'stop_E', else 1e-7); enters the E floor")
    p.add_argument('--T-start', dest='T_start', type=float, default=None, help='first T/r_s considered (default: all records)')
    p.add_argument('--T-end', dest='T_end', type=float, default=None, help='last T/r_s considered (default: all records)')
    p.add_argument('--interp-degree', dest='interp_degree', type=int, default=4, help='degree of the local polynomial interpolating N, v, rho, J at the root (degree+1 nearest sites)')
    p.add_argument('--interp-degree-check', dest='interp_degree_check', type=int, default=6, help='second degree for the interpolation spread of kappa_formula')
    p.add_argument('--root-window', dest='root_window', type=float, default=3.0, help='sites: the stored root must be reproduced by a grid sign change of N+v within this distance')
    p.add_argument('--root-agree-max', dest='root_agree_max', type=float, default=0.05, help='sites: |R_refit - R_stored| above this raises the flag root_disagree (reported, not excluding by default)')
    p.add_argument('--r-min-zero', dest='r_min_zero', type=float, default=2.0, help='zeros of N+v below this r are ignored (the centre)')
    p.add_argument('--simple-margin-min', dest='simple_margin_min', type=float, default=0.05, help='s = 1 - 8 pi G R^2 (rho_R - J_R) = 2 R kappa_R / N_R must exceed this for a simple root (0 at a double root, 1 in vacuum)')
    p.add_argument('--kappa-min-abs', dest='kappa_min_abs', type=float, default=1e-5, help='the dense slope d_r(N+v) at the root must exceed this (absolute, lattice units)')
    p.add_argument('--max-inner-zeros', dest='max_inner_zeros', type=int, default=1, help='at most this many inner-type zeros (negative slope) beside the single outer zero')
    p.add_argument('--max-dR-per-pair', dest='max_dR_per_pair', type=float, default=4.0, help='sites: |R_b - R_a| above this is a root-branch jump (pair excluded)')
    p.add_argument('--max-gap-factor', dest='max_gap_factor', type=float, default=1.5, help='a pair whose T gap exceeds this times the median gap is non-consecutive (excluded)')
    p.add_argument('--tangency-tol-abs', dest='tangency_tol_abs', type=float, default=1e-4, help='|N+v| at a sign-preserving local extremum below this is a near tangency on the grid (flag)')
    p.add_argument('--regime-label-ok', dest='regime_label_ok', default='in the regime', help="the record's regime.label that passes")
    p.add_argument('--gauge-label-ok', dest='gauge_label_ok', default='intact', help="the record's gauge.label that passes")
    p.add_argument('--exclude-on', dest='exclude_on', default='not_converged,regime,gauge,tangency_stored,outer_zero_count,inner_zero_count,stored_zero_count,stored_root_not_outermost,root_not_on_grid,not_simple_margin,slope', help='comma list of record-level flags that exclude a pair (others are reported only)')
    p.add_argument('--exclude-pair-on', dest='exclude_pair_on', default='non_consecutive,root_appearing,root_disappearing,root_jump', help='comma list of pair-level flags that exclude a pair (zero_count_changed and seam are flag-only by default)')
    p.add_argument('--seam-T-over-rs', dest='seam_T', default='', help='comma list of T/r_s where the history was resumed from another engine version; pairs touching one are flagged')
    p.add_argument('--exclude-seam', dest='exclude_seam', action='store_true', help='exclude pairs touching a seam')
    p.add_argument('--E-key', dest='E_key', default='E', help="record key of the collar charge")
    p.add_argument('--collar-key', dest='collar_key', default='r_collar', help="record key of the collar radius")
    p.add_argument('--zeros-key', dest='zeros_key', default='trapping_zeros', help="record key listing the dense zeros of N+v (outer root = the largest)")
    p.add_argument('--tangencies-key', dest='tangencies_key', default='tangencies', help="record key listing dense tangencies (double roots)")
    p.add_argument('--kappaT-key', dest='kappaT_key', default='kappa_T', help="record key of the stored dense slope at the outer root")
    p.add_argument('--floor-N-key', dest='floor_N_key', default='interp_floor_N', help="record key of the N interpolation floor (absolute)")
    p.add_argument('--floor-v-key', dest='floor_v_key', default='interp_floor_v', help="record key of the v interpolation floor (absolute)")
    p.add_argument('--T-match-tol', dest='T_match_tol', type=float, default=1e-6, help='relative tolerance matching record T to the fields rows')
    p.add_argument('--uniform-rtol', dest='uniform_rtol', type=float, default=1e-9, help='relative tolerance for the site grid to count as uniform')
    p.add_argument('--row-defect-warn', dest='row_defect_warn', type=float, default=0.1, help='relative RMS row defect in the exterior above which the reading of the stored sources is flagged')
    p.add_argument('--J-undetermined-rel', dest='J_undetermined_rel', type=float, default=1e-6, help='|uJ|/rho (RMS, exterior) below which the J sign is undetermined and +1 is kept')
    p.add_argument('--md-every', dest='md_every', type=int, default=8, help='print every n-th pair in the cumulative table (plus the last)')
    p.add_argument('--list-no-root-pairs', dest='list_no_root_pairs', action='store_true', help='also list every pair with no root at either end')
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
    if opts.rs is None:
        rs = run_args.get('rs')
        if rs is None:
            rs = 1.0
            notes.append("run args have no 'rs': r_s = 1 assumed (T/r_s matching uses the records' own T_over_rs)")
        opts.rs = float(rs)
    if opts.stop_E is None:
        se = run_args.get('stop_E')
        opts.stop_E = float(se) if se is not None else 1e-7
        if se is None:
            notes.append("run args have no 'stop_E': 1e-7 assumed for the E floor")
    if opts.tag is None:
        opts.tag = str(run_args.get('tag') or 'records')
    opts.threads = THREADS


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


def load_fields(path, which):
    z = np.load(path, allow_pickle=False)
    F = {'keys': list(z.files), 'path': path}
    F['R'] = np.asarray(z['R'], float)
    F['T'] = np.asarray(z['T'], float)
    if which == 'applied':
        F['N'] = z['N_applied']
        F['v'] = z['v_applied']
        F['N_key'], F['v_key'] = 'N_applied', 'v_applied'
    else:
        F['N'] = z['N']
        F['v'] = z['v']
        F['N_key'], F['v_key'] = 'N', 'v'
    F['rho'] = z['eps_tot']
    F['J'] = z['J_tot']
    F['rho_key'], F['J_key'] = 'eps_tot', 'J_tot'
    F['rho_pairs'] = z['eps_pairs'] if 'eps_pairs' in z.files else None
    F['J_pairs'] = z['J_pairs'] if 'J_pairs' in z.files else None
    F['G'] = float(z['g']) if 'g' in z.files else None
    F['ell'] = int(z['ell']) if 'ell' in z.files else None
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
# the J-sign audit (pass 1)
# ----------------------------------------------------------------------------
def decide_J_sign(records, F, rows, idx, G, h, opts):
    R = np.asarray(F['R'], float)
    base = FOUR_PI * R * R
    tot_e = {1: 0.0, -1: 0.0}
    tot_g = {1: 0.0, -1: 0.0}
    den_e = 0.0
    den_g = 0.0
    den_rho = 0.0
    den_J = 0.0
    n_used = 0
    for k in idx:
        rec = records[k]
        row = int(rows[k])
        zs = rec.get(opts.zeros_key) or []
        zs = [float(z) for z in zs if z is not None]
        if not zs:
            continue
        Rr = max(zs)
        N = np.asarray(F['N'][row], float)
        v = np.asarray(F['v'][row], float)
        rho = np.asarray(F['rho'][row], float)
        J0 = np.asarray(F['J'][row], float)
        mask = R > Rr
        if int(mask.sum()) < 10:
            continue
        u = v / N
        M = R * u * u / (2.0 * G)
        Mp = derivative(M, R, h)
        Np = derivative(N, R, h)
        for s in (1, -1):
            e = Mp - base * (rho + u * s * J0)
            tot_e[s] += float(np.sum(e[mask] ** 2))
            with np.errstate(divide='ignore', invalid='ignore'):
                g = Np / N + FOUR_PI * G * R * s * J0 / u
            g = np.where(np.isfinite(g), g, 0.0)
            tot_g[s] += float(np.sum(g[mask] ** 2))
        den_e += float(np.sum((base * (np.abs(rho) + np.abs(u * J0)))[mask] ** 2))
        den_rho += float(np.sum((base * rho)[mask] ** 2))
        den_J += float(np.sum((base * u * J0)[mask] ** 2))
        den_g += float(np.sum((Np / N)[mask] ** 2))
        n_used += 1
    rel_e = {s: (np.sqrt(tot_e[s] / den_e) if den_e > 0 else float('nan')) for s in (1, -1)}
    rel_g = {s: (np.sqrt(tot_g[s] / den_g) if den_g > 0 else float('nan')) for s in (1, -1)}
    J_over_rho = np.sqrt(den_J / den_rho) if den_rho > 0 else float('nan')
    if opts.J_sign == 'auto':
        if n_used == 0 or not np.isfinite(J_over_rho) or J_over_rho <= opts.J_undetermined_rel:
            sJ = 1
            status = 'auto: undetermined (|uJ| negligible against rho in the exterior, or no rooted record); +1 kept'
        else:
            sJ = 1 if tot_e[1] <= tot_e[-1] else -1
            status = 'auto: the sign with the smaller exterior Hamiltonian-row defect'
    else:
        sJ = int(float(opts.J_sign))
        status = 'forced by --J-sign'
    out = {'sJ': int(sJ), 'status': status, 'n_records_used': n_used,
           'rel_e_plus': float(rel_e[1]), 'rel_e_minus': float(rel_e[-1]),
           'rel_g_plus': float(rel_g[1]), 'rel_g_minus': float(rel_g[-1]),
           'uJ_over_rho_rel_rms': float(J_over_rho),
           'rel_e_chosen': float(rel_e[sJ]), 'rel_g_chosen': float(rel_g[sJ])}
    out['warn'] = bool(np.isfinite(out['rel_e_chosen']) and out['rel_e_chosen'] > opts.row_defect_warn)
    return out


# ----------------------------------------------------------------------------
# per record (pass 2)
# ----------------------------------------------------------------------------
def analyse_record(k, rec, F, row, sJ, G, h, h_eff, stop_E, opts):
    R = np.asarray(F['R'], float)
    N = np.asarray(F['N'][row], float)
    v = np.asarray(F['v'][row], float)
    rho = np.asarray(F['rho'][row], float)
    J = sJ * np.asarray(F['J'][row], float)
    rho_p = None if F.get('rho_pairs') is None else np.asarray(F['rho_pairs'][row], float)
    J_p = None if F.get('J_pairs') is None else sJ * np.asarray(F['J_pairs'][row], float)
    deg = int(opts.interp_degree)
    deg2 = int(opts.interp_degree_check)
    rs = float(opts.rs)
    f = N + v
    out = {'k': int(k), 'row': int(row), 'T_over_rs': _tors(rec, rs), 'T': _float(rec.get('T'))}

    # stored items
    zs = rec.get(opts.zeros_key) or []
    zs = [float(z) for z in zs if z is not None]
    tg = rec.get(opts.tangencies_key) or []
    out['zeros_stored'] = zs
    out['n_zeros_stored'] = len(zs)
    out['n_tangencies_stored'] = len(tg)
    out['R_stored'] = max(zs) if zs else float('nan')
    out['kappa_T_stored'] = _float(rec.get(opts.kappaT_key))
    sj = rec.get('surface_jets') or {}
    out['N_t_stored'] = _float(sj.get('N_t')) if isinstance(sj, dict) else float('nan')
    out['E_stored'] = _float(rec.get(opts.E_key))
    out['r_collar'] = _float(rec.get(opts.collar_key))
    out['converged'] = rec.get('converged')
    reg = rec.get('regime') or {}
    out['regime_label'] = reg.get('label') if isinstance(reg, dict) else None
    out['regime_intervals_above_1'] = len(reg.get('intervals_above_1') or []) if isinstance(reg, dict) else 0
    gau = rec.get('gauge') or {}
    out['gauge_label'] = gau.get('label') if isinstance(gau, dict) else None
    ls = rec.get('late_surface') or {}
    out['late_surface_flag'] = bool(ls.get('flag')) if isinstance(ls, dict) else False
    ind = rec.get('indicator') or {}
    cs = (ind.get('collar_screen') or {}) if isinstance(ind, dict) else {}
    out['collar_screen_dE'] = _float(cs.get('delta_E_signed')) if isinstance(cs, dict) else float('nan')
    out['collar_label'] = ind.get('collar_label') if isinstance(ind, dict) else None
    sol = rec.get('solver') or {}
    wt = (sol.get('W_tail') or {}) if isinstance(sol, dict) else {}
    out['W_tail_dE'] = (_float(wt.get('E_tail_end')) - _float(wt.get('E_collar'))) if isinstance(wt, dict) and wt else float('nan')
    out['floor_N'] = _float(rec.get(opts.floor_N_key))
    out['floor_v'] = _float(rec.get(opts.floor_v_key))

    # fields at the collar; the rows' defects everywhere
    e, g, M, Mp, Np = row_defects(R, N, v, rho, J, G, h)
    u = v / N
    out['N_collar_site'] = float(N[-1])
    out['R_last_site'] = float(R[-1])
    out['M_collar_fields'] = float(M[-1])
    out['M_collar_fields_minus_E'] = out['M_collar_fields'] - out['E_stored']

    # zeros of N + v on the site grid
    zeros = classify_zeros(R, f, opts.r_min_zero, deg)
    out['zeros_grid'] = [{'r': z['r'], 'slope': z['slope'], 'type': z['type']} for z in zeros]
    out['n_outer_grid'] = int(sum(1 for z in zeros if z['type'] == 'outer'))
    out['n_inner_grid'] = int(sum(1 for z in zeros if z['type'] == 'inner'))
    nt = near_tangencies(R, f, opts.tangency_tol_abs, opts.r_min_zero)
    out['near_tangencies_grid'] = nt
    out['n_near_tangencies_grid'] = len(nt)

    has_root = bool(np.isfinite(out['R_stored']))
    out['has_root'] = has_root
    flags = []
    if out['converged'] is False:
        flags.append('not_converged')
    if (out['regime_label'] is not None and out['regime_label'] != opts.regime_label_ok) or out['regime_intervals_above_1'] > 0:
        flags.append('regime')
    if out['gauge_label'] is not None and out['gauge_label'] != opts.gauge_label_ok:
        flags.append('gauge')
    if out['n_tangencies_stored'] > 0:
        flags.append('tangency_stored')
    if out['n_near_tangencies_grid'] > 0:
        flags.append('near_tangency_grid')
    if out['late_surface_flag']:
        flags.append('late_surface')
    if out['collar_label'] is not None and str(out['collar_label']) != 'passes':
        flags.append('collar_' + str(out['collar_label']).replace(' ', '_'))

    nan_keys = ('R_refit', 'kappa_dense', 'N_R', 'v_R', 'Nplusv_at_R_stored', 'slope_at_R_stored', 'rho_R', 'J_R', 'e_at_R',
                'simple_margin', 'kappa_formula', 'kappa_formula_deg_check', 'kappa_formula_interp_spread',
                'kappa_formula_minus_kappa_T', 'kappa_dense_minus_kappa_T', 'kappa_defect_prediction',
                'kappa_prediction_check', 'four_GE_kappa_formula', 'A', 'R_over_2GE', 'sigma_R_interp', 'sigma_R_refit', 'sigma_R')
    for key in nan_keys:
        out[key] = float('nan')
    out['root_type_grid'] = None
    out['outermost_is_stored'] = None

    if has_root:
        Rr = out['R_stored']
        cand = [z for z in zeros if abs(z['r'] - Rr) <= opts.root_window * h_eff]
        if cand:
            zc = min(cand, key=lambda z: abs(z['r'] - Rr))
            out['R_refit'] = zc['r']
            out['kappa_dense'] = zc['slope']
            out['root_type_grid'] = zc['type']
        else:
            flags.append('root_not_on_grid')
        outermost = max(zeros, key=lambda z: z['r']) if zeros else None
        out['outermost_is_stored'] = bool(outermost is not None and abs(outermost['r'] - Rr) <= opts.root_window * h_eff)
        if not out['outermost_is_stored']:
            flags.append('stored_root_not_outermost')
        if out['n_outer_grid'] != 1:
            flags.append('outer_zero_count')
        if out['n_inner_grid'] > opts.max_inner_zeros:
            flags.append('inner_zero_count')
        if out['n_zeros_stored'] > opts.max_inner_zeros + 1:
            flags.append('stored_zero_count')
        if np.isfinite(out['R_refit']) and abs(out['R_refit'] - Rr) > opts.root_agree_max * h_eff:
            flags.append('root_disagree')
        N_R, dN = interp_at(R, N, Rr, deg)
        v_R, dv = interp_at(R, v, Rr, deg)
        rho_R, _ = interp_at(R, rho, Rr, deg)
        J_R, _ = interp_at(R, J, Rr, deg)
        rho_R2, _ = interp_at(R, rho, Rr, deg2)
        J_R2, _ = interp_at(R, J, Rr, deg2)
        e_R, _ = interp_at(R, e, Rr, deg)
        out['N_R'] = N_R
        out['v_R'] = v_R
        out['Nplusv_at_R_stored'] = N_R + v_R
        out['slope_at_R_stored'] = dN + dv
        out['rho_R'] = rho_R
        out['J_R'] = J_R
        out['e_at_R'] = e_R
        margin = 1.0 - 8.0 * np.pi * G * Rr * Rr * (rho_R - J_R)
        margin2 = 1.0 - 8.0 * np.pi * G * Rr * Rr * (rho_R2 - J_R2)
        out['simple_margin'] = margin
        out['kappa_formula'] = N_R / (2.0 * Rr) * margin
        out['kappa_formula_deg_check'] = N_R / (2.0 * Rr) * margin2
        out['kappa_formula_interp_spread'] = abs(out['kappa_formula'] - out['kappa_formula_deg_check'])
        out['kappa_formula_minus_kappa_T'] = out['kappa_formula'] - out['kappa_T_stored']
        out['kappa_dense_minus_kappa_T'] = out['kappa_dense'] - out['kappa_T_stored']
        out['kappa_defect_prediction'] = -N_R * G * e_R / Rr
        out['kappa_prediction_check'] = (out['kappa_T_stored'] - out['kappa_formula']) - out['kappa_defect_prediction']
        out['four_GE_kappa_formula'] = 4.0 * G * out['E_stored'] * out['kappa_formula']
        out['A'] = FOUR_PI * Rr * Rr
        out['R_over_2GE'] = Rr / (2.0 * G * out['E_stored'])
        if margin < opts.simple_margin_min:
            flags.append('not_simple_margin')
        kd = out['kappa_dense']
        if not np.isfinite(kd) or kd <= opts.kappa_min_abs:
            flags.append('slope')
        fN = out['floor_N'] if np.isfinite(out['floor_N']) else 0.0
        fv = out['floor_v'] if np.isfinite(out['floor_v']) else 0.0
        kref = abs(kd) if (np.isfinite(kd) and kd != 0.0) else abs(out['kappa_T_stored'])
        if not (np.isfinite(kref) and kref > 0.0):
            kref = abs(out['kappa_formula'])
        out['sigma_R_interp'] = (fN + fv) / kref if kref > 0.0 else float('nan')
        out['sigma_R_refit'] = abs(out['R_refit'] - Rr) if np.isfinite(out['R_refit']) else 0.0
        out['sigma_R'] = _nan(np.max, [out['sigma_R_interp'], out['sigma_R_refit']])

    E = out['E_stored']
    out['sigma_E_stop'] = stop_E * abs(E) if np.isfinite(E) else float('nan')
    out['sigma_E_screen'] = abs(out['collar_screen_dE']) if np.isfinite(out['collar_screen_dE']) else 0.0
    out['sigma_E_tail'] = abs(out['W_tail_dE']) if np.isfinite(out['W_tail_dE']) else 0.0
    out['sigma_E'] = _nan(np.max, [out['sigma_E_stop'], out['sigma_E_screen'], out['sigma_E_tail']])

    # the exterior row audit for this record
    mask = R > (out['R_stored'] if has_root else opts.r_min_zero)
    base = FOUR_PI * R * R
    den = float(np.sqrt(np.sum((base * (np.abs(rho) + np.abs(u * J)))[mask] ** 2))) if mask.any() else 0.0
    out['row_e_rel_ext'] = float(np.sqrt(np.sum(e[mask] ** 2))) / den if den > 0 else float('nan')
    gg = np.where(np.isfinite(g), g, 0.0)
    deng = float(np.sqrt(np.sum((Np / N)[mask] ** 2))) if mask.any() else 0.0
    out['row_g_rel_ext'] = float(np.sqrt(np.sum(gg[mask] ** 2))) / deng if deng > 0 else float('nan')
    out['flags'] = flags
    arrays = {'N': N, 'v': v, 'rho': rho, 'J': J, 'rho_p': rho_p, 'J_p': J_p, 'e': e, 'g': g, 'M': M, 'Nplusv': f}
    return out, arrays


# ----------------------------------------------------------------------------
# per pair
# ----------------------------------------------------------------------------
def analyse_pair(A, B, aA, aB, R, G, h, h_eff, opts):
    deg = int(opts.interp_degree)
    pre = 1.0 / (8.0 * np.pi * G)
    base = FOUR_PI * R * R
    NA, vA, rhoA, JA = aA['N'], aA['v'], aA['rho'], aA['J']
    NB, vB, rhoB, JB = aB['N'], aB['v'], aB['rho'], aB['J']
    Nm = 0.5 * (NA + NB)
    vm = 0.5 * (vA + vB)
    rhom = 0.5 * (rhoA + rhoB)
    Jm = 0.5 * (JA + JB)
    drho = rhoB - rhoA
    dJ = JB - JA
    RA, RB = float(A['R_stored']), float(B['R_stored'])
    Rm = 0.5 * (RA + RB)
    A_a, A_b = FOUR_PI * RA * RA, FOUR_PI * RB * RB
    dA = A_b - A_a
    EA, EB = float(A['E_stored']), float(B['E_stored'])
    dE = EB - EA
    Em = 0.5 * (EA + EB)

    # column 3 and its variants
    f_mid = base * (Nm * drho + vm * dJ)
    W_rules, i0 = integral_from(R, f_mid, Rm, h, 3)
    W = W_rules['trapz']
    W_end_a = integral_from(R, base * (NA * drho + vA * dJ), Rm, h, 3)[0]['trapz']
    W_end_b = integral_from(R, base * (NB * drho + vB * dJ), Rm, h, 3)[0]['trapz']
    W_low_a = integral_from(R, f_mid, RA, h, 3)[0]['trapz']
    W_low_b = integral_from(R, f_mid, RB, h, 3)[0]['trapz']
    W_Nrho = integral_from(R, base * Nm * drho, Rm, h, 3)[0]['trapz']
    W_vJ = W - W_Nrho
    if all(aX.get(key) is not None for aX in (aA, aB) for key in ('rho_p', 'J_p')):
        f_p = base * (Nm * (aB['rho_p'] - aA['rho_p']) + vm * (aB['J_p'] - aA['J_p']))
        W_pairs = integral_from(R, f_p, Rm, h, 3)[0]['trapz']
        W_dust = W - W_pairs
    else:
        W_pairs, W_dust = float('nan'), float('nan')
    rc = float(B['r_collar']) if np.isfinite(_float(B.get('r_collar'))) else float(R[-1])
    W_missing = float(f_mid[-1]) * (rc - float(R[-1])) if rc > R[-1] else 0.0

    # column 2 and its variants
    kA, kB = float(A['kappa_formula']), float(B['kappa_formula'])
    k_mean = 0.5 * (kA + kB)
    Nm_R = interp_at(R, Nm, Rm, deg)[0]
    rhom_R = interp_at(R, rhom, Rm, deg)[0]
    Jm_R = interp_at(R, Jm, Rm, deg)[0]
    k_mid = kappa_formula(Nm_R, Rm, rhom_R, Jm_R, G)
    kT_a, kT_b = _float(A.get('kappa_T_stored')), _float(B.get('kappa_T_stored'))
    kT_mean = 0.5 * (kT_a + kT_b)
    kd_mean = 0.5 * (_float(A.get('kappa_dense')) + _float(B.get('kappa_dense')))
    k_vac = 1.0 / (4.0 * G * Em)
    col2_formula = k_mean * dA * pre
    col2_mid = k_mid * dA * pre
    col2_T = kT_mean * dA * pre
    col2_dense = kd_mean * dA * pre
    col2_a = kA * dA * pre
    col2_b = kB * dA * pre
    col2_vac = k_vac * dA * pre
    col2 = {'formula': col2_formula, 'kappa_T': col2_T, 'dense': col2_dense, 'midfields': col2_mid}[opts.kappa_column]

    # residuals
    res = dE - col2 - W
    res_formula = dE - col2_formula - W
    res_mid = dE - col2_mid - W
    res_T = dE - col2_T - W
    res_dense = dE - col2_dense - W
    res_simpson = dE - col2 - W_rules['simpson']
    res_coarse = dE - col2 - W_rules['coarse']
    res_noW = dE - col2
    res_vac = dE - col2_vac

    # the applied identity's defect integral
    dM = aB['M'] - aA['M']
    de = aB['e'] - aA['e']
    gm = 0.5 * (aA['g'] + aB['g'])
    gm = np.where(np.isfinite(gm), gm, 0.0)
    fD = Nm * (de + gm * dM)
    D = integral_from(R, fD, Rm, h, 3)[0]['trapz']
    Nc = 0.5 * (float(NA[-1]) + float(NB[-1]))
    dE_fields = float(B['M_collar_fields']) - float(A['M_collar_fields'])
    res_applied = Nc * dE_fields - col2_dense - W - D
    res_applied_storedE = Nc * dE - col2_dense - W - D

    # floors (the records' own) and discretization (ours)
    sE_stop_a, sE_stop_b = _float(A.get('sigma_E_stop')), _float(B.get('sigma_E_stop'))
    scr_a = A['collar_screen_dE'] if np.isfinite(_float(A.get('collar_screen_dE'))) else 0.0
    scr_b = B['collar_screen_dE'] if np.isfinite(_float(B.get('collar_screen_dE'))) else 0.0
    sigma_dE = float(np.sqrt(np.nansum([sE_stop_a ** 2, sE_stop_b ** 2, (scr_b - scr_a) ** 2])))
    sigma_dE_undiff = float(np.sqrt(np.nansum([_float(A.get('sigma_E')) ** 2, _float(B.get('sigma_E')) ** 2])))
    sR_a, sR_b = _float(A.get('sigma_R')), _float(B.get('sigma_R'))
    sigma_col2_root = abs(k_mean) * Rm / G * float(np.sqrt(np.nansum([sR_a ** 2, sR_b ** 2])))
    sigma_col2_kappa = abs(dA) * pre * _nan(np.max, [A.get('kappa_formula_interp_spread'), B.get('kappa_formula_interp_spread')])
    floor = float(np.sqrt(np.nansum([sigma_dE ** 2, sigma_col2_root ** 2, sigma_col2_kappa ** 2])))
    disc_kappa = 0.5 * abs(col2_b - col2_a)
    disc_kappa_mid = abs(col2_mid - col2_formula)
    disc_W_rule = _nan(np.max, [abs(W - W_rules['simpson']), abs(W - W_rules['coarse'])])
    disc_W_lambda = 0.5 * abs(W_end_b - W_end_a)
    disc_W_lower = max(abs(W_low_a - W), abs(W_low_b - W))
    disc = float(np.sqrt(np.nansum([disc_kappa ** 2, disc_W_rule ** 2, disc_W_lambda ** 2, W_missing ** 2])))
    pull = res / floor if floor > 0 else float('nan')
    tot = float(np.sqrt(floor ** 2 + disc ** 2))
    pull_total = res / tot if tot > 0 else float('nan')

    pr = dict(T_a=A['T_over_rs'], T_b=B['T_over_rs'], T_mid=0.5 * (A['T_over_rs'] + B['T_over_rs']),
              R_a=RA, R_b=RB, R_mid=Rm, dR=RB - RA, A_a=A_a, A_b=A_b, dA=dA,
              E_a=EA, E_b=EB, dE=dE, dE_fields=dE_fields, N_collar_mean=Nc, E_mid=Em,
              kappa_a=kA, kappa_b=kB, kappa_mean=k_mean, kappa_midfields=k_mid,
              kappa_T_a=kT_a, kappa_T_b=kT_b, kappa_T_mean=kT_mean, kappa_dense_mean=kd_mean, kappa_vac=k_vac,
              four_GE_kappa_mean=4.0 * G * Em * k_mean, kappa_mean_minus_kappa_T_mean=k_mean - kT_mean,
              col2=col2, col2_formula=col2_formula, col2_midfields=col2_mid, col2_T=col2_T, col2_dense=col2_dense,
              col2_a=col2_a, col2_b=col2_b, col2_vac=col2_vac,
              W_ext=W, W_simpson=W_rules['simpson'], W_coarse=W_rules['coarse'], W_end_a=W_end_a, W_end_b=W_end_b,
              W_low_a=W_low_a, W_low_b=W_low_b, W_pairs=W_pairs, W_dust=W_dust, W_Nrho=W_Nrho, W_vJ=W_vJ,
              W_missing_strip=W_missing, i0=int(i0),
              res=res, res_formula=res_formula, res_midfields=res_mid, res_T=res_T, res_dense=res_dense,
              res_simpson=res_simpson, res_coarse=res_coarse, res_noW=res_noW, res_vac=res_vac,
              defect_integral=D, res_applied=res_applied, res_applied_storedE=res_applied_storedE,
              sigma_dE=sigma_dE, sigma_dE_undiff=sigma_dE_undiff, sigma_col2_root=sigma_col2_root,
              sigma_col2_kappa=sigma_col2_kappa, floor=floor,
              disc_kappa=disc_kappa, disc_kappa_mid=disc_kappa_mid, disc_W_rule=disc_W_rule,
              disc_W_lambda=disc_W_lambda, disc_W_lower=disc_W_lower, disc=disc, pull=pull, pull_total=pull_total,
              W_over_dE=W / dE if dE != 0 else float('nan'), col2_over_dE=col2 / dE if dE != 0 else float('nan'),
              W_pairs_over_W=W_pairs / W if W != 0 else float('nan'),
              sigma_E_a=_float(A.get('sigma_E')), sigma_E_b=_float(B.get('sigma_E')),
              kappa_column=opts.kappa_column)
    pr['_f_mid'] = f_mid
    pr['_f_defect'] = fD
    return pr


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------
def closure_stats(pairs, recs, G):
    st = {'n_pairs_included': len(pairs)}
    rooted = [r for r in recs if r.get('has_root')]
    st['n_records_with_root'] = len(rooted)
    if rooted:
        kf = np.array([_float(r.get('kappa_formula')) for r in rooted])
        kT = np.array([_float(r.get('kappa_T_stored')) for r in rooted])
        kd = np.array([_float(r.get('kappa_dense')) for r in rooted])
        pc = np.array([_float(r.get('kappa_prediction_check')) for r in rooted])
        dR = np.array([_float(r.get('R_refit')) - _float(r.get('R_stored')) for r in rooted])
        dM = np.array([_float(r.get('M_collar_fields_minus_E')) for r in rooted])
        E = np.array([_float(r.get('E_stored')) for r in rooted])
        st['kappa_T_mean'] = _nan(np.mean, kT)
        st['kappa_formula_mean'] = _nan(np.mean, kf)
        st['kappa_dense_mean'] = _nan(np.mean, kd)
        st['kappa_formula_minus_kappa_T_mean'] = _nan(np.mean, kf - kT)
        st['kappa_formula_minus_kappa_T_max_abs'] = _nan(np.max, np.abs(kf - kT))
        st['kappa_dense_minus_kappa_T_max_abs'] = _nan(np.max, np.abs(kd - kT))
        st['kappa_prediction_check_rms'] = _nan(lambda a: np.sqrt(np.mean(a ** 2)), pc)
        st['kappa_prediction_check_max_abs'] = _nan(np.max, np.abs(pc))
        st['R_refit_minus_R_stored_max_abs'] = _nan(np.max, np.abs(dR))
        st['R_refit_minus_R_stored_rms'] = _nan(lambda a: np.sqrt(np.mean(a ** 2)), dR)
        st['M_collar_fields_minus_E_rel_max'] = _nan(np.max, np.abs(dM) / np.abs(E))
        st['Nplusv_at_R_stored_max_abs'] = _nan(np.max, np.abs([_float(r.get('Nplusv_at_R_stored')) for r in rooted]))
        st['simple_margin_min'] = _nan(np.min, [_float(r.get('simple_margin')) for r in rooted])
        st['simple_margin_max'] = _nan(np.max, [_float(r.get('simple_margin')) for r in rooted])
        st['four_GE_kappa_min'] = _nan(np.min, [_float(r.get('four_GE_kappa_formula')) for r in rooted])
        st['four_GE_kappa_max'] = _nan(np.max, [_float(r.get('four_GE_kappa_formula')) for r in rooted])
        st['R_over_2GE_min'] = _nan(np.min, [_float(r.get('R_over_2GE')) for r in rooted])
        st['R_over_2GE_max'] = _nan(np.max, [_float(r.get('R_over_2GE')) for r in rooted])
        st['row_e_rel_ext_median'] = _nan(np.median, [_float(r.get('row_e_rel_ext')) for r in rooted])
        st['row_g_rel_ext_median'] = _nan(np.median, [_float(r.get('row_g_rel_ext')) for r in rooted])
        for key in ('sigma_E', 'sigma_E_stop', 'sigma_E_screen', 'sigma_E_tail', 'sigma_R', 'sigma_R_interp', 'sigma_R_refit', 'floor_N', 'floor_v'):
            st[key + '_median'] = _nan(np.median, [_float(r.get(key)) for r in rooted])
            st[key + '_max'] = _nan(np.max, [_float(r.get(key)) for r in rooted])
    if not pairs:
        return st

    def arr(key):
        return np.array([_float(p.get(key)) for p in pairs])

    dE, c2, W, res = arr('dE'), arr('col2'), arr('W_ext'), arr('res')
    floor, disc = arr('floor'), arr('disc')
    for key in ('dE', 'dE_fields', 'col2', 'col2_formula', 'col2_T', 'col2_dense', 'col2_midfields', 'col2_vac',
                'W_ext', 'W_simpson', 'W_coarse', 'W_pairs', 'W_dust', 'W_Nrho', 'W_vJ', 'res', 'res_formula',
                'res_T', 'res_dense', 'res_midfields', 'res_simpson', 'res_coarse', 'res_noW', 'res_vac',
                'defect_integral', 'res_applied', 'res_applied_storedE', 'disc', 'dA', 'dR'):
        st['sum_' + key] = _nan(np.sum, arr(key))
    st['E_first'] = float(pairs[0]['E_a'])
    st['E_last'] = float(pairs[-1]['E_b'])
    st['E_last_minus_E_first'] = st['E_last'] - st['E_first']
    st['R_first'] = float(pairs[0]['R_a'])
    st['R_last'] = float(pairs[-1]['R_b'])
    st['T_first'] = float(pairs[0]['T_a'])
    st['T_last'] = float(pairs[-1]['T_b'])
    rms = lambda a: np.sqrt(np.mean(a ** 2))  # noqa: E731
    for key in ('res', 'res_T', 'res_dense', 'res_midfields', 'res_simpson', 'res_applied', 'res_noW', 'res_vac'):
        a = arr(key)
        st[key + '_mean'] = _nan(np.mean, a)
        st[key + '_rms'] = _nan(rms, a)
        st[key + '_max_abs'] = _nan(np.max, np.abs(a))
        if np.any(np.isfinite(a)):
            st[key + '_max_abs_T_b'] = float(pairs[int(np.nanargmax(np.abs(a)))]['T_b'])
    st['dE_rms'] = _nan(rms, dE)
    st['col2_rms'] = _nan(rms, c2)
    st['W_ext_rms'] = _nan(rms, W)
    st['res_over_dE_rms'] = st['res_rms'] / st['dE_rms'] if st['dE_rms'] > 0 else float('nan')
    st['res_over_W_rms'] = st['res_rms'] / st['W_ext_rms'] if st['W_ext_rms'] > 0 else float('nan')
    pull = arr('pull')
    pull_t = arr('pull_total')
    st['pull_rms'] = _nan(rms, pull)
    st['pull_total_rms'] = _nan(rms, pull_t)
    for q in (1, 2, 3):
        st['frac_abs_pull_lt_%d' % q] = _nan(np.mean, (np.abs(pull) < q).astype(float)) if np.any(np.isfinite(pull)) else float('nan')
        st['frac_abs_pull_total_lt_%d' % q] = _nan(np.mean, (np.abs(pull_t) < q).astype(float)) if np.any(np.isfinite(pull_t)) else float('nan')
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
    st['floor_cum'] = float(np.sqrt(np.nansum([_float(pairs[0]['sigma_E_a']) ** 2, _float(pairs[-1]['sigma_E_b']) ** 2])
                                    + np.nansum(arr('sigma_col2_root') ** 2) + np.nansum(arr('sigma_col2_kappa') ** 2)))
    st['disc_cum'] = _nan(np.sum, disc)
    st['cum_pull'] = st['sum_res'] / st['floor_cum'] if st['floor_cum'] > 0 else float('nan')
    tot = np.sqrt(st['floor_cum'] ** 2 + st['disc_cum'] ** 2)
    st['cum_pull_total'] = st['sum_res'] / tot if tot > 0 else float('nan')
    st['sum_W_over_sum_dE'] = st['sum_W_ext'] / st['sum_dE'] if st['sum_dE'] != 0 else float('nan')
    st['sum_col2_over_sum_dE'] = st['sum_col2'] / st['sum_dE'] if st['sum_dE'] != 0 else float('nan')
    st['sum_res_over_sum_dE'] = st['sum_res'] / st['sum_dE'] if st['sum_dE'] != 0 else float('nan')
    return st


# ----------------------------------------------------------------------------
# the analysis
# ----------------------------------------------------------------------------
def analyse(records, F, run_args, opts):
    t0 = time.time()
    notes = []
    fill_defaults(opts, run_args, F, notes)
    R = np.asarray(F['R'], float)
    nsites = int(R.size)
    h = uniform_step(R, opts.uniform_rtol)
    h_eff = h if h is not None else float(np.median(np.diff(R)))
    if h is None:
        notes.append('site grid is not uniform (rtol %g): Simpson skipped, derivatives by np.gradient' % opts.uniform_rtol)
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

    audit = decide_J_sign(records, F, rows, idx, G, h, opts)
    sJ = audit['sJ']
    if audit.get('warn'):
        notes.append('ROW AUDIT WARNING: the exterior Hamiltonian-row defect with the chosen J sign is %.3g relative (threshold %.3g): the stored %s/%s may not be the rows\' rho, J in this normalization, or the stored fields are not the ones solved with these sources; the 3 columns are still computed, read the applied residual' % (audit['rel_e_chosen'], opts.row_defect_warn, F.get('rho_key', 'eps_tot'), F.get('J_key', 'J_tot')))

    recs = []
    arr = {}
    for k in idx:
        rec_out, rec_arr = analyse_record(k, records[k], F, int(rows[k]), sJ, G, h, h_eff, stop_E, opts)
        recs.append(rec_out)
        arr[k] = rec_arr

    exclude_rec = set(s.strip() for s in str(opts.exclude_on).split(',') if s.strip())
    exclude_pair = set(s.strip() for s in str(opts.exclude_pair_on).split(',') if s.strip())
    Ts = np.array([r['T_over_rs'] for r in recs], float)
    gaps = np.diff(Ts)
    gap_med = float(np.median(gaps)) if gaps.size else float('nan')
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
            if abs(B['R_stored'] - A['R_stored']) > opts.max_dR_per_pair * h_eff:
                pflags.append('root_jump')
            if A['n_zeros_stored'] != B['n_zeros_stored']:
                pflags.append('zero_count_changed')
        if any(abs(A['T_over_rs'] - s) < 1e-9 or abs(B['T_over_rs'] - s) < 1e-9 for s in seams):
            pflags.append('seam')
        reasons = [f for f in pflags if f in exclude_pair]
        if 'seam' in pflags and opts.exclude_seam:
            reasons.append('seam')
        reasons += ['A:' + f for f in A['flags'] if f in exclude_rec]
        reasons += ['B:' + f for f in B['flags'] if f in exclude_rec]
        if reasons:
            pairs_exc.append({'i': i, 'T_a': A['T_over_rs'], 'T_b': B['T_over_rs'], 'R_a': A['R_stored'], 'R_b': B['R_stored'],
                              'reasons': reasons, 'other_flags': [f for f in pflags if f not in reasons]})
            continue
        pr = analyse_pair(A, B, arr[A['k']], arr[B['k']], R, G, h, h_eff, opts)
        pr['i'] = i
        pr['flags'] = pflags + ['A:' + f for f in A['flags']] + ['B:' + f for f in B['flags']]
        pairs_inc.append(pr)

    cum = {'dE': 0.0, 'col2': 0.0, 'W_ext': 0.0, 'res': 0.0, 'defect_integral': 0.0, 'res_applied': 0.0,
           'W_pairs': 0.0, 'W_dust': 0.0, 'res_T': 0.0, 'res_simpson': 0.0, 'res_vac': 0.0, 'res_noW': 0.0}
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

    thr = {key: getattr(opts, key) for key in ('interp_degree', 'interp_degree_check', 'root_window', 'root_agree_max', 'r_min_zero',
                                                'simple_margin_min', 'kappa_min_abs', 'max_inner_zeros', 'max_dR_per_pair',
                                                'max_gap_factor', 'tangency_tol_abs', 'regime_label_ok', 'gauge_label_ok',
                                                'exclude_on', 'exclude_pair_on', 'seam_T', 'exclude_seam', 'row_defect_warn',
                                                'J_undetermined_rel', 'T_match_tol', 'uniform_rtol', 'kappa_column', 'fields',
                                                'J_sign', 'T_start', 'T_end')}
    rc_vals = [r['r_collar'] for r in recs if np.isfinite(_float(r.get('r_collar')))]
    rc_str = _fmt(rc_vals[0]) if rc_vals else 'not stored'
    definitions = {
        'column_1_dE': ("delta E = E(T_b) - E(T_a); E is the record key '%s' (the collar charge at r_collar = %s; the Misner-Sharp "
                        "mass M = r u^2/(2G), u = v/N, of the stored fields at the last site r = %s is printed beside it as M_collar_fields "
                        "and its difference to E as a check). G = %s, r_s = %s." % (opts.E_key, rc_str, _fmt(R[-1]), _fmt(G), _fmt(rs))),
        'column_2_kappa_dA': ("(kappa_R/(8 pi G)) delta A, A = 4 pi R^2, R = the largest entry of the record key '%s' (the dense outer zero "
                              "of N + v), delta A = A_b - A_a; kappa_R = (N_R/(2R)) [1 - 8 pi G R^2 (rho_R - J_R)] (input 02) with N = '%s', "
                              "rho = '%s', J = %+d * '%s' interpolated at R from the site arrays by the centred polynomial of degree %d "
                              "(degree %d beside, their difference = the interpolation spread); kappa evaluated at both records and averaged "
                              "(the midpoint rule in the record index); the stored '%s' beside it with the difference; the primary column uses "
                              "kappa = %s. kappa_dense = the slope of the stored N + v at the refit grid zero. Predicted kappa_T - kappa_formula "
                              "= -N_R G e_R / R from the Hamiltonian-row defect e at R (input 02's applied kappa)."
                              % (opts.zeros_key, F.get('N_key', 'N'), F.get('rho_key', 'eps_tot'), sJ, F.get('J_key', 'J_tot'),
                                 opts.interp_degree, opts.interp_degree_check, opts.kappaT_key, opts.kappa_column)),
        'column_3_W_ext': ("W_ext = 4 pi int_{R_mid}^{r_last} r^2 [N_mid (rho_b - rho_a) + v_mid (J_b - J_a)] dr with the Eulerian differences "
                           "at fixed r, N_mid = (N_a + N_b)/2, v_mid likewise, R_mid = (R_a + R_b)/2, r_last = %s; trapezoid on the site grid "
                           "(spacing %s) with the partial cell [R_mid, next site] by the trapezoid on the interpolated integrand; Simpson and the "
                           "half grid (every other site) beside as the radial discretization error; the endpoint weights (N_a, v_a) and (N_b, v_b) "
                           "beside as the record-index discretization error; the lower limits R_a and R_b beside. Split into the N delta rho and "
                           "v delta J parts and, when eps_pairs/J_pairs are stored, into the pairs' and the dust's parts."
                           % (_fmt(R[-1]), _fmt(h_eff))),
        'residual': "res = delta E - (kappa/(8 pi G)) delta A - W_ext per pair, cumulated over the included pairs in order; res_noW = delta E - (kappa/8 pi G) delta A (the no-work premise); res_vac = delta E - delta A/(32 pi G^2 E_mid) (the vacuum first law).",
        'applied_identity_diagnostic': ("res_applied = N_c delta M_c - (kappa_dense/(8 pi G)) delta A - W_ext - int_{R_mid}^{r_last} N_mid [(e_b - e_a) + g_mid (M_b - M_a)] dr, "
                                        "e = M' - 4 pi r^2 (rho + u J), g = N'/N + 4 pi G r J/u on the stored arrays (4th-order centred derivatives); "
                                        "N_c = N at the last site; input 02's applied identity, which closes up to the discretization whatever the row defects are."),
        'premises': ("A record carries a simple dense root when: it stores a root; the site arrays reproduce it by a sign change of N+v within %g sites; it is the outermost zero; "
                     "exactly one outer-type zero (positive slope) exists on the grid and at most %d inner-type zeros; the stored zero count is at most %d; "
                     "s = 1 - 8 pi G R^2 (rho_R - J_R) >= %g; the dense slope > %g; no stored tangency; converged; regime.label == '%s' with no interval above 1; "
                     "gauge.label == '%s'. A pair is excluded when either record fails one of the flags in --exclude-on, or when the root appears or disappears "
                     "between the records, the records are not consecutive (gap > %g x median), or |R_b - R_a| > %g sites. Other flags are reported."
                     % (opts.root_window, opts.max_inner_zeros, opts.max_inner_zeros + 1, opts.simple_margin_min, opts.kappa_min_abs,
                        opts.regime_label_ok, opts.gauge_label_ok, opts.max_gap_factor, opts.max_dR_per_pair)),
        'floors': ("Per record: sigma_E = max(stop_E |E| with stop_E = %g, |indicator.collar_screen.delta_E_signed|, |solver.W_tail.E_tail_end - E_collar|); "
                   "sigma_R = max((%s + %s)/|kappa_dense|, |R_refit - R_stored|). Per pair: sigma_dE = sqrt(stop_E^2 (E_a^2 + E_b^2) + (screen_b - screen_a)^2) "
                   "(the collar screen is a slowly varying offset of E and enters the pair differenced; undifferenced in sigma_dE_undiff); "
                   "sigma_col2 = (kappa_mean R_mid / G) sqrt(sigma_R_a^2 + sigma_R_b^2) (+) |delta A|/(8 pi G) x the kappa interpolation spread; "
                   "floor = sqrt(sigma_dE^2 + sigma_col2^2); pull = res/floor. Cumulative floor: sqrt(sigma_E_first^2 + sigma_E_last^2 + sum sigma_col2^2)."
                   % (stop_E, opts.floor_N_key, opts.floor_v_key)),
        'discretization': ("disc_kappa = |col2(kappa_b) - col2(kappa_a)|/2 (endpoint against midpoint); disc_W_rule = max(|W_trapz - W_simpson|, |W_trapz - W_half_grid|); "
                           "disc_W_lambda = |W(N_b, v_b) - W(N_a, v_a)|/2; disc = their quadrature sum (+ the strip beyond the last site to r_collar if any); "
                           "pull_total = res/sqrt(floor^2 + disc^2). The midpoint rule is third order per pair in the record step; the cumulative disc is the plain sum."),
    }
    result = {
        'tag': opts.tag,
        'definitions': definitions,
        'thresholds': thr,
        'grid': {'nsites': nsites, 'r_first': float(R[0]), 'r_last': float(R[-1]), 'h': h, 'h_eff': h_eff, 'uniform': h is not None,
                 'ell_npz': F.get('ell'), 'fields_keys': F.get('keys')},
        'matching': {'mode': match_mode, 'n_records_json': len(records), 'n_rows_npz': int(T_npz.size), 'n_unmatched': n_unmatched,
                     'n_analysed': len(recs), 'T_first_analysed': float(recs[0]['T_over_rs']), 'T_last_analysed': float(recs[-1]['T_over_rs']),
                     'gap_median_T_over_rs': gap_med},
        'audit': audit,
        'records': recs,
        'pairs': pairs_inc,
        'excluded_pairs': pairs_exc,
        'stats': stats,
        'notes': notes,
        'G': G, 'rs': rs, 'stop_E': stop_E, 'sJ': sJ,
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
            out[prefix + key] = np.array([float('nan') if x is None else float(x) for x in vals], dtype=float)
    return out


def write_md(path, res, json_path, npz_path):
    st = res['stats']
    au = res['audit']
    L = []
    L.append('# FIRSTLAW_records_v1 -- %s' % res['tag'])
    L.append('')
    L.append('## Inputs')
    L.append('- records: `%s`' % json_path)
    L.append('- fields: `%s` (keys %s)' % (npz_path, ', '.join(res['grid'].get('fields_keys') or [])))
    L.append('- %d records in the json, %d rows in the npz, matching by %s, %d unmatched; %d records analysed from T/r_s %s to %s (median gap %s)' % (
        res['matching']['n_records_json'], res['matching']['n_rows_npz'], res['matching']['mode'], res['matching']['n_unmatched'],
        res['matching']['n_analysed'], _fmt(res['matching']['T_first_analysed']), _fmt(res['matching']['T_last_analysed']), _fmt(res['matching']['gap_median_T_over_rs'])))
    L.append('- site grid: %d sites, r from %s to %s, spacing %s (%s); G = %s, r_s = %s, stop_E = %s' % (
        res['grid']['nsites'], _fmt(res['grid']['r_first']), _fmt(res['grid']['r_last']), _fmt(res['grid']['h_eff']),
        'uniform' if res['grid']['uniform'] else 'NOT uniform', _fmt(res['G']), _fmt(res['rs']), _fmt(res['stop_E'])))
    L.append('- J sign: J = %+d x J_tot (%s); exterior row defects, relative RMS: e(+) = %s, e(-) = %s; g(+) = %s, g(-) = %s; |uJ|/rho (RMS) = %s; %d rooted records used%s' % (
        au['sJ'], au['status'], _fmt(au['rel_e_plus'], 3), _fmt(au['rel_e_minus'], 3), _fmt(au['rel_g_plus'], 3), _fmt(au['rel_g_minus'], 3),
        _fmt(au['uJ_over_rho_rel_rms'], 3), au['n_records_used'], '; WARNING: defect above --row-defect-warn' if au.get('warn') else ''))
    for n in res['notes']:
        L.append('- NOTE: %s' % n)
    L.append('')
    L.append('## Definitions (as computed)')
    for key in ('column_1_dE', 'column_2_kappa_dA', 'column_3_W_ext', 'residual', 'applied_identity_diagnostic', 'premises', 'floors', 'discretization'):
        L.append('- **%s**: %s' % (key, res['definitions'][key]))
    L.append('')
    L.append('## Headline (included pairs, in order)')
    if st.get('n_pairs_included', 0) > 0:
        L.append('- %d pairs included, T/r_s %s to %s; %d pairs listed as excluded; %d pairs with no root at either end%s' % (
            st['n_pairs_included'], _fmt(st['T_first']), _fmt(st['T_last']), st['n_pairs_excluded_listed'], st['n_pairs_no_root'],
            (' (T/r_s %s to %s)' % (_fmt(st['no_root_T_span'][0]), _fmt(st['no_root_T_span'][1]))) if st.get('no_root_T_span') else ''))
        L.append('- sum delta E = %s (E_last - E_first over the included span = %s; E from %s to %s; R from %s to %s)' % (
            _fmt(st['sum_dE']), _fmt(st['E_last_minus_E_first']), _fmt(st['E_first'], 9), _fmt(st['E_last'], 9), _fmt(st['R_first']), _fmt(st['R_last'])))
        L.append('- sum (kappa/8 pi G) delta A = %s  (kappa column: %s; with kappa_T: %s; with kappa_dense: %s; vacuum 1/(4GE): %s)' % (
            _fmt(st['sum_col2']), res['thresholds']['kappa_column'], _fmt(st['sum_col2_T']), _fmt(st['sum_col2_dense']), _fmt(st['sum_col2_vac'])))
        L.append('- sum W_ext = %s  (Simpson %s, half grid %s; N delta rho part %s, v delta J part %s; pairs part %s, dust part %s)' % (
            _fmt(st['sum_W_ext']), _fmt(st['sum_W_simpson']), _fmt(st['sum_W_coarse']), _fmt(st['sum_W_Nrho']), _fmt(st['sum_W_vJ']),
            _fmt(st['sum_W_pairs']), _fmt(st['sum_W_dust'])))
        L.append('- sum residual = %s; cumulative floor %s (pull %s); cumulative discretization %s (pull_total %s); sum res / sum delta E = %s' % (
            _fmt(st['sum_res']), _fmt(st['floor_cum']), _fmt(st['cum_pull'], 3), _fmt(st['disc_cum']), _fmt(st['cum_pull_total'], 3), _fmt(st['sum_res_over_sum_dE'], 3)))
        L.append('- W_ext against delta E: sum W / sum delta E = %s; per pair W/delta E median %s (min %s, max %s); (kappa/8 pi G) delta A / delta E median %s' % (
            _fmt(st['sum_W_over_sum_dE'], 4), _fmt(st['W_over_dE_median'], 4), _fmt(st['W_over_dE_min'], 4), _fmt(st['W_over_dE_max'], 4), _fmt(st['col2_over_dE_median'], 4)))
        L.append('- the no-work premise: sum (delta E - (kappa/8 pi G) delta A) = %s; the vacuum first law: sum (delta E - delta A/(32 pi G^2 E)) = %s' % (
            _fmt(st['sum_res_noW']), _fmt(st['sum_res_vac'])))
        L.append('- applied identity (row defects retained): sum defect integral = %s, sum res_applied = %s (rms %s, max |.| %s)' % (
            _fmt(st['sum_defect_integral']), _fmt(st['sum_res_applied']), _fmt(st['res_applied_rms']), _fmt(st['res_applied_max_abs'])))
    else:
        L.append('- no pair included; see the excluded pairs')
    if st.get('n_records_with_root', 0) > 0:
        L.append('- kappa: mean kappa_T %s, mean kappa_formula %s, mean kappa_dense %s; |kappa_formula - kappa_T| max %s; |kappa_dense - kappa_T| max %s; '
                 'prediction check (kappa_T - kappa_formula) - (-N_R G e_R/R): rms %s, max %s' % (
                     _fmt(st['kappa_T_mean']), _fmt(st['kappa_formula_mean']), _fmt(st['kappa_dense_mean']), _fmt(st['kappa_formula_minus_kappa_T_max_abs'], 3),
                     _fmt(st['kappa_dense_minus_kappa_T_max_abs'], 3), _fmt(st['kappa_prediction_check_rms'], 3), _fmt(st['kappa_prediction_check_max_abs'], 3)))
        L.append('- 4 G E kappa_formula from %s to %s; R/(2GE) from %s to %s; simple margin s from %s to %s' % (
            _fmt(st['four_GE_kappa_min'], 4), _fmt(st['four_GE_kappa_max'], 4), _fmt(st['R_over_2GE_min'], 4), _fmt(st['R_over_2GE_max'], 4),
            _fmt(st['simple_margin_min'], 4), _fmt(st['simple_margin_max'], 4)))
        L.append('- root checks: |R_refit - R_stored| max %s sites (rms %s); |N+v| at the stored root max %s; |M_collar_fields - E|/E max %s; exterior row defects median e %s, g %s' % (
            _fmt(st['R_refit_minus_R_stored_max_abs'], 3), _fmt(st['R_refit_minus_R_stored_rms'], 3), _fmt(st['Nplusv_at_R_stored_max_abs'], 3),
            _fmt(st['M_collar_fields_minus_E_rel_max'], 3), _fmt(st['row_e_rel_ext_median'], 3), _fmt(st['row_g_rel_ext_median'], 3)))
        L.append('- floors (medians over rooted records): sigma_E %s (stop %s, screen %s, tail %s); %s %s, %s %s; sigma_R %s sites (interp %s, refit %s)' % (
            _fmt(st['sigma_E_median'], 3), _fmt(st['sigma_E_stop_median'], 3), _fmt(st['sigma_E_screen_median'], 3), _fmt(st['sigma_E_tail_median'], 3),
            res['thresholds'].get('kappa_column') and 'floor_N', _fmt(st['floor_N_median'], 3), 'floor_v', _fmt(st['floor_v_median'], 3),
            _fmt(st['sigma_R_median'], 3), _fmt(st['sigma_R_interp_median'], 3), _fmt(st['sigma_R_refit_median'], 3)))
    L.append('')
    L.append('## Closure statistics (included pairs)')
    if st.get('n_pairs_included', 0) > 0:
        L.append('| quantity | mean | rms | max abs | at T_b |')
        L.append('|---|---|---|---|---|')
        for key, lab in (('res', 'res (primary kappa)'), ('res_T', 'res with kappa_T'), ('res_dense', 'res with kappa_dense'),
                         ('res_midfields', 'res with midpoint-field kappa'), ('res_simpson', 'res with Simpson W'),
                         ('res_applied', 'res_applied (defects retained)'), ('res_noW', 'delta E - (kappa/8 pi G) delta A'), ('res_vac', 'vacuum first law residual')):
            L.append('| %s | %s | %s | %s | %s |' % (lab, _fmt(st[key + '_mean']), _fmt(st[key + '_rms']), _fmt(st[key + '_max_abs']), _fmt(st.get(key + '_max_abs_T_b'))))
        L.append('')
        L.append('- rms: delta E %s, (kappa/8 pi G) delta A %s, W_ext %s; res/delta E (rms) %s; res/W (rms) %s' % (
            _fmt(st['dE_rms']), _fmt(st['col2_rms']), _fmt(st['W_ext_rms']), _fmt(st['res_over_dE_rms'], 3), _fmt(st['res_over_W_rms'], 3)))
        L.append('- pull = res/floor: rms %s; fraction |pull| < 1: %s, < 2: %s, < 3: %s; pull_total (floor (+) disc): rms %s; < 1: %s, < 2: %s, < 3: %s' % (
            _fmt(st['pull_rms'], 3), _fmt(st['frac_abs_pull_lt_1'], 3), _fmt(st['frac_abs_pull_lt_2'], 3), _fmt(st['frac_abs_pull_lt_3'], 3),
            _fmt(st['pull_total_rms'], 3), _fmt(st['frac_abs_pull_total_lt_1'], 3), _fmt(st['frac_abs_pull_total_lt_2'], 3), _fmt(st['frac_abs_pull_total_lt_3'], 3)))
        L.append('- medians: |delta E| %s, |(kappa/8 pi G) delta A| %s, |W_ext| %s, floor %s, disc %s; W_pairs/W median %s' % (
            _fmt(st['dE_abs_median']), _fmt(st['col2_abs_median']), _fmt(st['W_ext_abs_median']), _fmt(st['floor_median']), _fmt(st['disc_median']), _fmt(st['W_pairs_over_W_median'], 3)))
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
    L.append('kappa column = %s. Columns: delta E; (kappa/8 pi G) delta A; W_ext (trapezoid); W_pairs; res; floor; pull; disc; W/delta E; kappa_formula(b); kappa_T(b); their difference at b; the applied residual; flags.' % res['thresholds']['kappa_column'])
    L.append('')
    L.append('| T_a | T_b | R_mid | dR | dE | k dA/8piG | W_ext | W_pairs | res | floor | pull | disc | W/dE | kappa_f(b) | kappa_T(b) | k_f-k_T(b) | res_applied | flags |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for p in res['pairs']:
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            _fmt(p['T_a']), _fmt(p['T_b']), _fmt(p['R_mid'], 7), _fmt(p['dR'], 4), _fmt(p['dE'], 5), _fmt(p['col2'], 5), _fmt(p['W_ext'], 5), _fmt(p['W_pairs'], 4),
            _fmt(p['res'], 4), _fmt(p['floor'], 3), _fmt(p['pull'], 3), _fmt(p['disc'], 3), _fmt(p['W_over_dE'], 4), _fmt(p['kappa_b'], 6), _fmt(p['kappa_T_b'], 6),
            _fmt(p['kappa_b'] - p['kappa_T_b'], 3), _fmt(p['res_applied'], 4), '; '.join(p.get('flags') or [])))
    L.append('')
    L.append('## Cumulative (every %d-th included pair and the last)' % max(1, int(res['thresholds'].get('md_every', 8) if 'md_every' in res['thresholds'] else 8)))
    L.append('')
    L.append('| T_b | sum dE | E_b - E_first | sum k dA/8piG | sum W_ext | sum W_pairs | sum res | sum res_T | sum res_applied | sum res_noW | sum res_vac |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    every = max(1, int(res.get('md_every', 8)))
    npairs = len(res['pairs'])
    for j, p in enumerate(res['pairs']):
        if j % every == every - 1 or j == npairs - 1:
            L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
                _fmt(p['T_b']), _fmt(p['cum_dE'], 6), _fmt(p['E_b_minus_E_first'], 6), _fmt(p['cum_col2'], 6), _fmt(p['cum_W_ext'], 6), _fmt(p['cum_W_pairs'], 5),
                _fmt(p['cum_res'], 4), _fmt(p['cum_res_T'], 4), _fmt(p['cum_res_applied'], 4), _fmt(p['cum_res_noW'], 5), _fmt(p['cum_res_vac'], 5)))
    L.append('')
    L.append('## Per-record table (records with a stored root)')
    L.append('')
    L.append('| T | R | R_refit-R | N_R | rho_R | J_R | margin s | kappa_formula | kappa_T | k_f-k_T | pred k_T-k_f | kappa_dense | 4GEk | R/2GE | E | M_fields-E | sigma_E | sigma_R | e_rel_ext | flags |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in res['records']:
        if not r.get('has_root'):
            continue
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            _fmt(r['T_over_rs']), _fmt(r['R_stored'], 8), _fmt(r['R_refit'] - r['R_stored'], 3), _fmt(r['N_R'], 6), _fmt(r['rho_R'], 4), _fmt(r['J_R'], 4),
            _fmt(r['simple_margin'], 5), _fmt(r['kappa_formula'], 6), _fmt(r['kappa_T_stored'], 6), _fmt(r['kappa_formula_minus_kappa_T'], 3),
            _fmt(r['kappa_defect_prediction'], 3), _fmt(r['kappa_dense'], 6), _fmt(r['four_GE_kappa_formula'], 5), _fmt(r['R_over_2GE'], 5),
            _fmt(r['E_stored'], 9), _fmt(r['M_collar_fields_minus_E'], 3), _fmt(r['sigma_E'], 3), _fmt(r['sigma_R'], 3), _fmt(r['row_e_rel_ext'], 3),
            '; '.join(r.get('flags') or [])))
    L.append('')
    L.append('## Thresholds (named arguments)')
    for key, val in sorted(res['thresholds'].items()):
        L.append('- %s = %s' % (key, val))
    L.append('')
    L.append('- seconds: %s' % _fmt(res['seconds'], 4))
    with open(path, 'w') as fh:
        fh.write('\n'.join(L) + '\n')


def write_outputs(res, R, arr, out_dir, tag, json_path, npz_path, opts):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, 'firstlaw_records_%s' % tag)
    res['md_every'] = int(opts.md_every)
    res['thresholds']['md_every'] = int(opts.md_every)
    # npz
    arrays = {'R': np.asarray(R, float)}
    arrays.update(_stack_numeric(res['pairs'], 'pair_'))
    arrays.update(_stack_numeric(res['records'], 'rec_'))
    if res['pairs']:
        arrays['pair_f_mid'] = np.array([p['_f_mid'] for p in res['pairs']], dtype=float)
        arrays['pair_f_defect'] = np.array([p['_f_defect'] for p in res['pairs']], dtype=float)
    if res['records']:
        arrays['rec_e'] = np.array([arr[r['k']]['e'] for r in res['records']], dtype=float)
        arrays['rec_g'] = np.array([arr[r['k']]['g'] for r in res['records']], dtype=float)
        arrays['rec_Nplusv'] = np.array([arr[r['k']]['Nplusv'] for r in res['records']], dtype=float)
        arrays['rec_M'] = np.array([arr[r['k']]['M'] for r in res['records']], dtype=float)
    arrays['options_json'] = np.array(json.dumps(_jsonable(vars(opts))))
    arrays['stats_json'] = np.array(json.dumps(_jsonable(res['stats'])))
    np.savez(base + '.npz', **arrays)
    # json (the big arrays stripped)
    res_json = dict(res)
    res_json['inputs'] = {'json': json_path, 'npz': npz_path, 'options': vars(opts)}
    res_json['pairs'] = [{k: v for k, v in p.items() if not k.startswith('_')} for p in res['pairs']]
    with open(base + '.json', 'w') as fh:
        json.dump(_jsonable(res_json), fh, indent=1)
    write_md(base + '.md', res, json_path, npz_path)
    return base


def main(argv=None):
    opts = parse_args(argv)
    if opts.json is None:
        raise SystemExit('--json is required (collapse_<tag>.json)')
    json_path = os.path.abspath(opts.json)
    if opts.npz is None:
        npz_path = json_path[:-5] + '_fields.npz' if json_path.endswith('.json') else json_path + '_fields.npz'
    else:
        npz_path = os.path.abspath(opts.npz)
    t0 = time.time()
    head, run_args, records, rec_key = load_json_records(json_path)
    F = load_fields(npz_path, opts.fields)
    res = analyse(records, F, run_args, opts)
    res['record_list_key'] = rec_key
    out_dir = opts.out or os.path.join(os.path.dirname(json_path), 'firstlaw')
    # the per-record arrays are needed for the npz: recompute cheaply from the records' rows
    arr = {}
    rows_by_k = {r['k']: r['row'] for r in res['records']}
    sJ = res['sJ']
    h = res['grid']['h']
    for k, row in rows_by_k.items():
        N = np.asarray(F['N'][row], float)
        v = np.asarray(F['v'][row], float)
        rho = np.asarray(F['rho'][row], float)
        J = sJ * np.asarray(F['J'][row], float)
        e, g, M, _, _ = row_defects(np.asarray(F['R'], float), N, v, rho, J, res['G'], h)
        arr[k] = {'e': e, 'g': g, 'M': M, 'Nplusv': N + v}
    base = write_outputs(res, F['R'], arr, out_dir, opts.tag, json_path, npz_path, opts)
    st = res['stats']
    print('FIRSTLAW_records_v1 %s: %d records analysed, %d with a root, %d pairs included, %d excluded listed, %d without a root' % (
        opts.tag, st['n_records_analysed'], st['n_records_with_root'], st['n_pairs_included'], st['n_pairs_excluded_listed'], st['n_pairs_no_root']))
    print('record list key: %s; J sign %+d (%s); row defects e(+) %s e(-) %s' % (rec_key, res['sJ'], res['audit']['status'],
                                                                             _fmt(res['audit']['rel_e_plus'], 3), _fmt(res['audit']['rel_e_minus'], 3)))
    if st['n_pairs_included'] > 0:
        print('sum dE %s | sum (kappa/8piG) dA %s | sum W_ext %s | sum res %s (floor %s, disc %s) | sum res_applied %s' % (
            _fmt(st['sum_dE']), _fmt(st['sum_col2']), _fmt(st['sum_W_ext']), _fmt(st['sum_res']), _fmt(st['floor_cum']), _fmt(st['disc_cum']), _fmt(st['sum_res_applied'])))
        print('W/dE: sum ratio %s, median per pair %s | pull rms %s, |pull|<2 fraction %s | kappa_formula - kappa_T max %s, prediction check rms %s' % (
            _fmt(st['sum_W_over_sum_dE'], 4), _fmt(st['W_over_dE_median'], 4), _fmt(st['pull_rms'], 3), _fmt(st['frac_abs_pull_lt_2'], 3),
            _fmt(st['kappa_formula_minus_kappa_T_max_abs'], 3), _fmt(st['kappa_prediction_check_rms'], 3)))
    for n in res['notes']:
        print('NOTE: ' + n)
    print('wrote %s.{json,npz,md} in %.1f s' % (base, time.time() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
