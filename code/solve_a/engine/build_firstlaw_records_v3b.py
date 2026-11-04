#!/usr/bin/env python3
"""
build_firstlaw_records_v3b.py -- writes firstlaw_records_v3b.py from firstlaw_records_v3.py by exact string replacements.

Version 3 on the sigma/a 2 history found no radial stress among the fields npz keys (T_ll not computed) and a history json grown
from 301 to 321 records (the BITWISE check against version 2 differing by construction).  Version 3b:
  - reads p_r from a companion npz (the engine's handoff, collapse_<tag>_handoff.npz beside the fields npz) where S_r_ell is already
    the engine's kernel average on the companion's node grid r_nodes: --pr-key handoff:S_r_ell (or <path>.npz:S_r_ell, or the bare
    key when it is not among the fields keys); the rows are matched to the records by T; no kernel is applied; the stress is carried
    to the dense grid and to the dense zeros by a not-a-knot cubic spline, with the stride-2 error estimate, the local-polynomial
    method difference at R, and T_ll at the two nodes bracketing R where the stress is the stored node value;
  - checks the normalization of S_r_ell against rho_ell and J_ell: the companion's own energy densities on the nodes against the dense
    rho_ell at the nodes (combinations of eps_dust, eps_sea, eps_R2), N2avg against the dense N^2, the ratios |p| max/|rho| max,
    |p| max/|J| max, p/rho and p/J at R against the node shell volume 4 pi R^2 h_node that a per-node extensive quantity would show
    (--pr-node-reading per-node divides it out; --pr-scale -1 if the engine's stress module defines S_r as -T_rr; the module's own
    text lines about S_r are printed);
  - adds the outward flux F(R) = -4 pi R^2 N_R^2 T_ll of FIRSTLAW_rows_v1's continuity identity, F(r_c) at the dense collar, and per
    pair the balance W_ext/dt against the mean F(R) - F(r_c) (the finite-record estimate of the weighted continuity-defect integral);
    every pair key beginning with W summed (the lower-limit variants included); the pairwise tail differences beside the pair floor;
  - makes --check-against match the records, pairs and excluded pairs by T, so that version 2's 301 records are compared with the same
    records of the 321 (stats and matching are set-dependent and are compared only when the record counts agree).
Nothing in the version-2 computation of the three columns, the floors, the flags and the exclusions is touched.

Each replacement names an OLD text copied verbatim from the firstlaw_records_v3.py that build_firstlaw_records_v3.py writes and the NEW
text that takes its place.  Every OLD must occur exactly once in the source; otherwise nothing is written and the offending replacement
is named.  The result must compile.  The sha256 of the source and of the result are printed.  --smoke imports the written module and
runs engine-free checks of the new parts.

    python3 build_firstlaw_records_v3b.py [--src firstlaw_records_v3.py] [--dst firstlaw_records_v3b.py] [--force] [--smoke]
"""
import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def lines(*xs):
    return '\n'.join(xs)


# ----------------------------------------------------------------------------
# the version-3b additions (spliced after V3_COMPARE_EXEMPT, before _v3_norm, inside the version-3 section)
# ----------------------------------------------------------------------------
V3B_HELPERS = r'''
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
'''

V3B_DOC_HEAD = r"""firstlaw_records_v3b.py -- FIRSTLAW_records_v3b: version 3 with the radial stress read from a companion npz (the engine's handoff,
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

firstlaw_records_v3.py -- FIRSTLAW_records_v3: version 2 plus, per record with a simple dense root and per pair, the null-null"""

REPLACEMENTS = [
    ('docstring header',
     r"firstlaw_records_v3.py -- FIRSTLAW_records_v3: version 2 plus, per record with a simple dense root and per pair, the null-null",
     V3B_DOC_HEAD),
    ('version-3b helpers after V3_COMPARE_EXEMPT',
     r"V3_COMPARE_EXEMPT = ('solve_seconds', 'seconds', 'solve_seconds_total', 'solve_seconds_median')",
     lines(r"V3_COMPARE_EXEMPT = ('solve_seconds', 'seconds', 'solve_seconds_total', 'solve_seconds_median')", "", "") + V3B_HELPERS.strip('\n')),
    ('V3_REC_NAN_KEYS: the version-3b record keys',
     r"'p_quadrature_rel', 'p_kernel_check', 'p_max_abs_over_rho_max_abs', 'stress_seconds')",
     lines(r"'p_quadrature_rel', 'p_kernel_check', 'p_max_abs_over_rho_max_abs', 'stress_seconds', 'Tll_R_over_interp_err',",
           r"                   'Tll_nodes_bracket_same_sign', 'flux_F_R', 'flux_F_rc', 'flux_F_R_minus_rc', 'r_collar_dense', 'p_interp_err_half_rel', 'p_interp_err_rel',",
           r"                   'p_interp_err_half_root_rel', 'p_interp_err_root_rel', 'p_interp_err_root_abs', 'p_R_method_diff_rel', 'p_extrap_cells', 'companion_row',",
           r"                   'companion_T', 'handoff_eps_vs_rho_rel', 'handoff_eps_expected_vs_rho_rel', 'handoff_eps_vs_eps_tot_rel', 'handoff_N2avg_vs_dense_rel',",
           r"                   'pt_R', 'pt_over_rho_R', 'pt_over_p_R', 'r_node_below_R', 'r_node_above_R', 'Tll_node_below_R', 'Tll_node_above_R', 'p_R_over_J_R',",
           r"                   'p_max_abs_over_J_max_abs', 'p_node_shell_volume_at_R')")),
    ('V3_THRESHOLD_KEYS: the version-3b options',
     r"V3_THRESHOLD_KEYS = ('pr_key', 'pr_reading', 'pr_scale', 'Tll_window_ell', 'Tll_offsets_ell', 'seed_cache', 'seed_key_policy', 'no_stress',",
     r"V3_THRESHOLD_KEYS = ('pr_key', 'pr_reading', 'pr_scale', 'Tll_window_ell', 'Tll_offsets_ell', 'seed_cache', 'seed_key_policy', 'no_stress', 'pt_key', 'pr_node_reading', 'pr_grid_key',"),
    ('v3_stress_pass: the companion dispatch',
     lines(r"    key, note, cands, pt_key = find_stress_key(F.get('keys') or [], getattr(opts, 'pr_key', None))",
           r"    meta.update(key=key, note=note, candidates=cands, pt_key=pt_key)"),
     lines(r"    loc = v3b_locate_stress(F, opts)",
           r"    if loc.get('companion'):",
           r"        return v3b_stress_pass_companion(FLD, F, opts, dense, rows, idx, r_fine, ell_pass, notes, meta, loc, t0)",
           r"    key, note, cands, pt_key = loc['key'], loc['note'], loc['candidates'], loc['pt_key']",
           r"    meta.update(key=key, note=note, candidates=cands, pt_key=pt_key)")),
    ('v3_record_columns: the p_R source label',
     r"                out['p_R_source'] = 'kernel value at the root'",
     r"                out['p_R_source'] = stress.get('p_at_zeros_source') or 'kernel value at the root'"),
    ('v3_record_columns: the version-3b stress record keys',
     r"            out['stress_seconds'] = _float(stress.get('seconds'))",
     lines(r"            out['stress_seconds'] = _float(stress.get('seconds'))",
           r"            for key_ in V3B_STRESS_RECORD_KEYS:",
           r"                if key_ in stress:",
           r"                    out[key_] = stress.get(key_) if key_ in V3B_STRESS_STRING_KEYS else _float(stress.get(key_))")),
    ('v3_record_columns: the root extras after T_ll(R)',
     r"            out['Tll_dimless_R'] = 8.0 * np.pi * G * R0 * R0 * out['Tll_R']",
     lines(r"            out['Tll_dimless_R'] = 8.0 * np.pi * G * R0 * R0 * out['Tll_R']",
           r"            v3b_root_extras(out, stress, G, R0, arrays, r_fine)")),
    ('v3_pair_columns: the pair extras',
     r"    pr['col2_defect_corr_over_col2'] = pr['col2_defect_corr'] / c2 if (np.isfinite(c2) and c2 != 0.0) else NAN",
     lines(r"    pr['col2_defect_corr_over_col2'] = pr['col2_defect_corr'] / c2 if (np.isfinite(c2) and c2 != 0.0) else NAN",
           r"    v3b_pair_extras(pr, A, B, aA, aB, Rg, sl, Rm, deg)")),
    ('v3_record_stats: the version-3b statistics',
     r"    st['stress_seconds_total'] = _nan(np.sum, [_float(r.get('stress_seconds')) for r in rooted_ok])",
     lines(r"    st['stress_seconds_total'] = _nan(np.sum, [_float(r.get('stress_seconds')) for r in rooted_ok])",
           r"    v3b_record_stats(st, rooted_ok)")),
    ('v3_definitions: the companion stress text',
     lines(r"    key = meta.get('key')",
           r"    if key:",
           r'''        stress_text = ("p_r = p_ell, the engine's kernel average per unit volume of the stored fields key '%s' (%s), read as the per-site "'''),
     lines(r"    key = meta.get('key')",
           r"    if key and meta.get('companion'):",
           r"        stress_text = v3b_stress_definition(meta)",
           r"    elif key:",
           r'''        stress_text = ("p_r = p_ell, the engine's kernel average per unit volume of the stored fields key '%s' (%s), read as the per-site "''')),
    ('v3_stress_line: the companion line',
     lines(r"        return 'unavailable (%s)' % (meta.get('note') or 'no stress pass')",
           r'''    return ("key '%s' (%s), reading %s, scale %s, kernel %s; %s of %s records smoothed in %s s with %s workers; kernel check "'''),
     lines(r"        return 'unavailable (%s)' % (meta.get('note') or 'no stress pass')",
           r"    if meta.get('companion'):",
           r"        return v3b_stress_line(meta)",
           r'''    return ("key '%s' (%s), reading %s, scale %s, kernel %s; %s of %s records smoothed in %s s with %s workers; kernel check "''')),
    ('v3_md_summary: the version-3b lines',
     r'''    L.append("- across the root, |r - R| <= %s ell on the dense grid: the window minimum of T_ll has median %s (min %s) and the maximum median %s "''',
     lines(r"    L.extend(v3b_md_lines(res))",
           r'''    L.append("- across the root, |r - R| <= %s ell on the dense grid: the window minimum of T_ll has median %s (min %s) and the maximum median %s "''')),
    ('v3_console: the version-3b lines',
     r"    print('V3 STRESS: %s' % v3_stress_line(res))",
     lines(r"    print('V3 STRESS: %s' % v3_stress_line(res))",
           r"    for line in v3b_console_lines(res):",
           r"        print(line)")),
    ('v3_add_arguments: the --pr-key forms',
     r'''                   help="fields key of the per-site radial stress P_r (default: found by name among the npz keys: P_r_tot, Pr_tot, p_r_tot, sigma_rr_tot, "''',
     lines(r'''                   help="the radial stress: a fields key (the per-site stress, averaged with the engine's kernel as in version 3), 'handoff:<key>' (the key in the "''',
           r'''                        "companion npz collapse_<tag>_handoff.npz beside the fields npz, already the engine's kernel average on r_nodes, carried to the dense grid "''',
           r'''                        "by a cubic spline), '<path>.npz:<key>' (the same in that npz), or a bare key that is in the companion when it is not among the fields "''',
           r'''                        "keys; default: found by name among the fields keys, then among the companion's (S_r_ell and the names below with _ell); 'none' disables "''',
           r'''                        "p_r. The names: P_r_tot, Pr_tot, p_r_tot, sigma_rr_tot, "''')),
    ('v3_add_arguments: --pt-key, --pr-node-reading, --pr-grid-key',
     r"    p.add_argument('--no-stress', dest='no_stress', action='store_true', help='skip the stress pass: T_ll is not computed (rho - 2J is still printed)')",
     lines(r"    p.add_argument('--no-stress', dest='no_stress', action='store_true', help='skip the stress pass: T_ll is not computed (rho - 2J is still printed)')",
           r"    p.add_argument('--pt-key', dest='pt_key', default=None,",
           r'''                   help="the tangential stress, in the forms of --pr-key (default: found beside the radial one when that is in the companion npz, S_t_ell); read at "''',
           r'''                        "the outer root and reported (pt_R, pt_over_p_R, pt_over_rho_R); not used in T_ll")''',
           r"    p.add_argument('--pr-node-reading', dest='pr_node_reading', choices=('density', 'per-node'), default='density',",
           r'''                   help="how a companion (node-grid) stress is read: 'density' = already the kernel average per unit volume (the engine's _ell fields; the "''',
           r'''                        "default); 'per-node' = an extensive quantity per node, divided by the node shell volume 4 pi r_n^2 h_node before use")''',
           r"    p.add_argument('--pr-grid-key', dest='pr_grid_key', default='r_nodes',",
           r'''                   help="the companion npz key of the node grid of its stress arrays (default r_nodes; else r, R, else any 1-D key of the matching length)")''')),
    ('v3_check_against: matched by T',
     lines(r"    n, bad = compare_json_files(path, new_json)",
           r"    print('BITWISE against %s: %d leaves of records, pairs, excluded_pairs, stats, grid and matching compared (timings exempt), %d differ%s' % (",
           r"        path, n, len(bad), '' if not bad else '; first: ' + '; '.join('%s: %r -> %r' % (p, a, b) for p, a, b in bad[:8])))",
           r"    with open(new_json[:-5] + '_bitwise.json', 'w') as fh:",
           r"        json.dump({'reference': path, 'new': new_json, 'n_compared': n, 'n_differ': len(bad), 'first_differences': [[str(x) for x in b] for b in bad[:50]]},",
           r"                  fh, indent=1)",
           r"    return n, bad"),
     lines(r"    n, bad, mnote = compare_json_files_matched(path, new_json)",
           r"    print('BITWISE against %s: %d leaves compared (timings exempt; records, pairs and excluded pairs matched by T: %s), %d differ%s' % (",
           r"        path, n, mnote, len(bad), '' if not bad else '; first: ' + '; '.join('%s: %r -> %r' % (p, a, b) for p, a, b in bad[:8])))",
           r"    with open(new_json[:-5] + '_bitwise.json', 'w') as fh:",
           r"        json.dump({'reference': path, 'new': new_json, 'n_compared': n, 'n_differ': len(bad), 'matching': mnote,",
           r"                   'first_differences': [[str(x) for x in b] for b in bad[:50]]}, fh, indent=1)",
           r"    return n, bad")),
    ('write_md: title',
     r"    L.append('# FIRSTLAW_records_v3 -- %s' % res['tag'])",
     r"    L.append('# FIRSTLAW_records_v3b -- %s' % res['tag'])"),
    ('write_outputs: base name',
     r"    base = os.path.join(out_dir, 'firstlaw_records_v3_%s' % tag)",
     r"    base = os.path.join(out_dir, 'firstlaw_records_v3b_%s' % tag)"),
    ('main: the cache file name',
     r"    cache_file = None if opts.no_cache else os.path.join(out_dir, 'firstlaw_dense_v3_%s_cache.npz' % opts.tag)",
     r"    cache_file = None if opts.no_cache else os.path.join(out_dir, 'firstlaw_dense_v3b_%s_cache.npz' % opts.tag)"),
    ('main: the first console line',
     r"    print('FIRSTLAW_records_v3 %s: %d records matched",
     r"    print('FIRSTLAW_records_v3b %s: %d records matched"),
    ('main: the summary console line',
     r"    print('FIRSTLAW_records_v3 %s: %d records analysed",
     r"    print('FIRSTLAW_records_v3b %s: %d records analysed"),
    ('parse_args: description',
     r"    p = argparse.ArgumentParser(description='FIRSTLAW_records_v3: the first-law identity between consecutive rooted records, on the dense solution re-solved with the engine, plus T_ll = rho + p_r - 2J and the row defect at every dense root')",
     r"    p = argparse.ArgumentParser(description='FIRSTLAW_records_v3b: the first-law identity between consecutive rooted records, on the dense solution re-solved with the engine, plus T_ll = rho + p_r - 2J (p_r from the fields npz or from a companion npz such as the handoff, --pr-key handoff:S_r_ell) and the row defect at every dense root; the bitwise check against a version-2 json matched by T')"),
]


def sha256_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def smoke(dst):
    """Imports the written module and exercises what needs no engine and no history: the --pr-key forms and the companion discovery on a small npz pair,
    the node-to-grid interpolation and its error on a Gaussian, the companion stress pass end to end on two synthetic records (the console line included),
    and the matched comparison on two small jsons.  Prints PASS/FAIL lines; returns the number of failures."""
    import argparse as _ap
    import importlib.util
    import json
    import tempfile
    import time
    import traceback
    import numpy as np
    spec = importlib.util.spec_from_file_location('firstlaw_records_v3b_smoke', dst)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    fails = 0

    def check(name, ok, detail=''):
        nonlocal fails
        print('%s  %s%s' % ('PASS' if ok else 'FAIL', name, (': ' + detail) if detail else ''))
        if not ok:
            fails += 1
    tmp = tempfile.mkdtemp(prefix='firstlaw_v3b_smoke_')
    xs = np.arange(1.0, 65.0)
    r_fine = np.arange(0.0, 64.0 + 1e-9, 0.125)
    R0 = 30.4

    def rho_f(r):
        return 3e-4 * np.exp(-(r - 30.0) ** 2 / 8.0) + 1e-5

    def J_f(r):
        return 1e-4 * np.exp(-(r - 31.0) ** 2 / 10.0)

    def p_f(r):
        return 1e-4 * np.exp(-(r - 29.0) ** 2 / 8.0)
    T_f = np.array([8.0, 16.0])
    S_r = np.array([p_f(xs), 1.1 * p_f(xs)])
    S_t = 0.5 * S_r
    eps_dust = np.array([rho_f(xs), rho_f(xs)]) * 0.6
    eps_sea = np.array([rho_f(xs), rho_f(xs)]) * 0.7
    eps_R2 = np.array([rho_f(xs), rho_f(xs)]) * 0.3
    fields_path = os.path.join(tmp, 'collapse_smoke_fields.npz')
    hand_path = os.path.join(tmp, 'collapse_smoke_handoff.npz')
    np.savez(hand_path, T=T_f, r_nodes=xs, S_r_ell=S_r, S_t_ell=S_t, eps_dust=eps_dust, eps_sea=eps_sea, eps_R2=eps_R2, ell=np.array(2.0), g=np.array(0.25))
    sites = np.arange(1.0, 65.0)
    F = {'path': fields_path, 'keys': ['R', 'T', 'eps_tot', 'J_tot'], 'R': sites, 'T': T_f,
         'eps_tot': np.array([rho_f(sites), rho_f(sites)]) * 4 * np.pi * sites ** 2, 'ell': np.array(2.0), 'g': np.array(0.25)}
    opts = _ap.Namespace(pr_key=None, pt_key=None, tag='smoke', pr_grid_key='r_nodes', pr_node_reading='density', pr_scale=1.0, interp_degree=4,
                         Tll_window_ell=4.0)
    try:
        loc = m.v3b_locate_stress(F, opts)
        check('S1 companion discovery (no stress among the fields keys -> the handoff beside the fields npz, S_r_ell and S_t_ell by name)',
              bool(loc.get('companion')) and loc.get('key') == 'S_r_ell' and loc.get('pt_key') == 'S_t_ell',
              str({k: loc.get(k) for k in ('companion', 'key', 'pt_key', 'note')}))
        a = m._v3b_parse_spec('handoff:S_r_ell', F, opts)
        b = m._v3b_parse_spec('handoff:nope', F, opts)
        c = m._v3b_parse_spec('S_r_ell', F, opts)
        check('S2 the --pr-key forms (handoff:<key>, a missing key, a bare companion key)',
              a[0] == hand_path and a[1] == 'S_r_ell' and b[1] is None and c[0] == hand_path and c[1] == 'S_r_ell', '%s | %s | %s' % (a[2], b[2], c[2]))
        arrays = {'rho': rho_f(r_fine), 'J': J_f(r_fine), 'N': np.ones(r_fine.size), 'v': -0.5 * np.ones(r_fine.size)}
        d0 = {'ok': True, 'arrays': dict(arrays), 'zeros_dense': [12.3, R0]}
        d1 = {'ok': True, 'arrays': dict(arrays), 'zeros_dense': [R0]}
        rec = m.v3b_companion_record(xs, S_r[0], S_t[0], d0, r_fine, 4, 8.0, 1.0, {'eps_dust': eps_dust[0], 'eps_sea': eps_sea[0], 'eps_R2': eps_R2[0]}, None, None)
        exact = float(p_f(np.array([R0]))[0])
        err_root = abs(rec['p_at_zeros'][1] - exact) / float(np.max(S_r[0]))
        check('S3 interpolation at the root (cubic spline against the exact Gaussian, within the stride-2 estimate)',
              bool(rec.get('ok')) and err_root < 5e-3 and np.isfinite(rec.get('p_interp_err_root_rel', np.nan)) and err_root <= rec['p_interp_err_half_root_rel'] + 1e-12,
              'rel error at R %.2e, stride-2 at the root %.2e, estimate %.2e, method difference %.2e' % (
                  err_root, rec.get('p_interp_err_half_root_rel'), rec.get('p_interp_err_root_rel'), rec.get('p_R_method_diff_rel')))
        check('S4 the bracketing nodes, the normalization combination and the tangential stress',
              rec.get('r_node_below_R') == 30.0 and rec.get('r_node_above_R') == 31.0 and np.isfinite(rec.get('Tll_node_below_R', np.nan))
              and rec.get('handoff_eps_combo') == 'eps_dust + eps_sea - eps_R2' and rec.get('handoff_eps_vs_rho_rel', 1.0) < 1e-3 and np.isfinite(rec.get('pt_R', np.nan)),
              'combination %s rel RMS %.2e, pt_R %s' % (rec.get('handoff_eps_combo'), rec.get('handoff_eps_vs_rho_rel', np.nan), rec.get('pt_R')))
        meta, notes = {}, []
        res, secs = m.v3b_stress_pass_companion(None, F, opts, {0: d0, 1: d1}, np.array([0, 1]), [0, 1], r_fine, 2.0, notes, meta, loc, time.time())
        check('S5 the companion pass end to end (rows matched by T, the console line printed)',
              res['meta'].get('n_ok') == 2 and res['meta'].get('n_unmatched') == 0 and str(res['meta'].get('row_matching', '')).startswith('by T'),
              'n_ok %s, unmatched %s, %s' % (res['meta'].get('n_ok'), res['meta'].get('n_unmatched'), res['meta'].get('row_matching')))
        line = m.v3b_stress_line(res['meta'])
        defn = m.v3b_stress_definition(res['meta'])
        check('S6 the md line and the definition text format', isinstance(line, str) and 'S_r_ell' in line and isinstance(defn, str) and 'S_r_ell' in defn)
    except Exception as exc:
        traceback.print_exc()
        check('S1-S6 the companion path', False, '%s: %s' % (type(exc).__name__, exc))
    try:
        ref = {'records': [{'T_over_rs': 0.0, 'x': 1.0}, {'T_over_rs': 0.125, 'x': 2.0}], 'pairs': [{'T_a': 0.0, 'T_b': 0.125, 'y': 3.0}], 'excluded_pairs': [],
               'stats': {'n': 2}, 'grid': {'h_fine': 0.125}, 'matching': {'n': 2}}
        new = {'records': [{'T_over_rs': 0.0, 'x': 1.0, 'z': 9.0}, {'T_over_rs': 0.125, 'x': 2.0}, {'T_over_rs': 0.25, 'x': 5.0}],
               'pairs': [{'T_a': 0.0, 'T_b': 0.125, 'y': 3.0}, {'T_a': 0.125, 'T_b': 0.25, 'y': 4.0}], 'excluded_pairs': [], 'stats': {'n': 3},
               'grid': {'h_fine': 0.125}, 'matching': {'n': 3}}
        rp, npth = os.path.join(tmp, 'ref.json'), os.path.join(tmp, 'new.json')
        with open(rp, 'w') as fh:
            json.dump(ref, fh)
        with open(npth, 'w') as fh:
            json.dump(new, fh)
        n, bad, note = m.compare_json_files_matched(rp, npth)
        check('S7 matched comparison on a grown list (8 leaves, stats and matching not compared)', n == 8 and not bad and 'stats not compared' in note,
              'n %d, bad %s, %s' % (n, bad, note))
        new['records'][1]['x'] = 2.5
        with open(npth, 'w') as fh:
            json.dump(new, fh)
        n, bad, note = m.compare_json_files_matched(rp, npth)
        check('S8 matched comparison, one changed leaf found', n == 8 and len(bad) == 1 and bad[0][0] == '/records[1]/x', 'n %d, bad %s' % (n, bad))
        new['records'] = new['records'][:2]
        new['records'][1]['x'] = 2.0
        new['pairs'] = new['pairs'][:1]
        with open(npth, 'w') as fh:
            json.dump(new, fh)
        n, bad, note = m.compare_json_files_matched(rp, npth)
        check('S9 matched comparison on the same set (stats and matching compared, 10 leaves)', n == 10 and not bad and 'not compared' not in note,
              'n %d, bad %s, %s' % (n, bad, note))
    except Exception as exc:
        traceback.print_exc()
        check('S7-S9 the matched comparison', False, '%s: %s' % (type(exc).__name__, exc))
    print('smoke: %d failures (temporary files in %s)' % (fails, tmp))
    return fails


def main(argv=None):
    p = argparse.ArgumentParser(description='write firstlaw_records_v3b.py from firstlaw_records_v3.py by exact string replacements')
    p.add_argument('--src', default=os.path.join(HERE, 'firstlaw_records_v3.py'))
    p.add_argument('--dst', default=os.path.join(HERE, 'firstlaw_records_v3b.py'))
    p.add_argument('--force', action='store_true', help='overwrite a destination that exists with other contents')
    p.add_argument('--smoke', action='store_true', help='after writing (or finding the destination unchanged), import it and run the engine-free checks')
    a = p.parse_args(argv)
    with open(a.src) as fh:
        src = fh.read()
    problems = []
    for name, old, new in REPLACEMENTS:
        n = src.count(old)
        if n != 1:
            problems.append("replacement '%s': the OLD text occurs %d times in %s (exactly 1 required)" % (name, n, a.src))
        if old == new:
            problems.append("replacement '%s': OLD and NEW are identical" % name)
    if problems:
        print('\n'.join(problems))
        raise SystemExit('nothing written: the source is not the firstlaw_records_v3.py that build_firstlaw_records_v3.py writes')
    out = src
    for name, old, new in REPLACEMENTS:
        out = out.replace(old, new, 1)
    try:
        compile(out, a.dst, 'exec')
    except SyntaxError as exc:
        raise SystemExit('the result does not compile: %s' % exc)
    if os.path.exists(a.dst) and not a.force:
        with open(a.dst) as fh:
            cur = fh.read()
        if cur == out:
            print('unchanged: %s already has the built contents (sha256 %s)' % (a.dst, sha256_text(out)))
            return (1 if smoke(a.dst) else 0) if a.smoke else 0
        raise SystemExit('%s exists with other contents; --force overwrites it' % a.dst)
    with open(a.dst, 'w') as fh:
        fh.write(out)
    print('source %s: %d lines, sha256 %s' % (a.src, src.count('\n'), sha256_text(src)))
    print('wrote  %s: %d lines, sha256 %s (%d replacements applied)' % (a.dst, out.count('\n'), sha256_text(out), len(REPLACEMENTS)))
    if a.smoke:
        return 1 if smoke(a.dst) else 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
