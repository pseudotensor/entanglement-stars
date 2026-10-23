#!/usr/bin/env python3
"""
build_firstlaw_records_v3.py -- writes firstlaw_records_v3.py from firstlaw_records_v2.py by exact string replacements.

firstlaw_records_v3.py is version 2 plus, per record with a simple dense root and per pair, the null-null component
T_ll = rho + p_r - 2J at the root (with its profile across the root on the dense solution) and the row defect (e, g) at every
dense root; every new column is written to the json and the md beside the existing three columns; nothing else changes (the
version-2 computation of the three columns, the closure, the floors, the flags and the exclusions is not touched, so on the same
records their values are bitwise those of version 2; the written script checks this itself with --check-against).

Each replacement names an OLD text copied verbatim from firstlaw_records_v2.py and the NEW text that takes its place.  Every OLD
must occur exactly once in the source; otherwise nothing is written and the offending replacement is named.  The result must
compile.  The sha256 of the source and of the result are printed.

    python3 build_firstlaw_records_v3.py [--src firstlaw_records_v2.py] [--dst firstlaw_records_v3.py] [--force]
"""
import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def lines(*xs):
    return '\n'.join(xs)


# ----------------------------------------------------------------------------
# the version-3 additions (spliced after kappa_formula, inside the small-helpers section of version 2)
# ----------------------------------------------------------------------------
V3_HELPERS = r'''
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
                   'p_quadrature_rel', 'p_kernel_check', 'p_max_abs_over_rho_max_abs', 'stress_seconds')
V3_THRESHOLD_KEYS = ('pr_key', 'pr_reading', 'pr_scale', 'Tll_window_ell', 'Tll_offsets_ell', 'seed_cache', 'seed_key_policy', 'no_stress',
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
                out['p_R_source'] = 'kernel value at the root'
            else:
                p_R = _interp_finite(r_fine, p_arr, R0, deg)
                out['p_R_source'] = 'interpolated from the dense grid'
            out['p_R'] = p_R
            out['Tll_R'] = rho_R + p_R - 2.0 * J_R
            out['Tll_flipJ_R'] = rho_R + p_R + 2.0 * J_R
            out['p_over_rho_R'] = p_R / rho_R if rho_R != 0.0 else NAN
            out['Tll_dimless_R'] = 8.0 * np.pi * G * R0 * R0 * out['Tll_R']
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
    if key:
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
                   help="fields key of the per-site radial stress P_r (default: found by name among the npz keys: P_r_tot, Pr_tot, p_r_tot, sigma_rr_tot, "
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
    key, note, cands, pt_key = find_stress_key(F.get('keys') or [], getattr(opts, 'pr_key', None))
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
    n, bad = compare_json_files(path, new_json)
    print('BITWISE against %s: %d leaves of records, pairs, excluded_pairs, stats, grid and matching compared (timings exempt), %d differ%s' % (
        path, n, len(bad), '' if not bad else '; first: ' + '; '.join('%s: %r -> %r' % (p, a, b) for p, a, b in bad[:8])))
    with open(new_json[:-5] + '_bitwise.json', 'w') as fh:
        json.dump({'reference': path, 'new': new_json, 'n_compared': n, 'n_differ': len(bad), 'first_differences': [[str(x) for x in b] for b in bad[:50]]},
                  fh, indent=1)
    return n, bad
'''

V3_DOC = r"""firstlaw_records_v3.py -- FIRSTLAW_records_v3: version 2 plus, per record with a simple dense root and per pair, the null-null
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
and per-pair columns), the cache firstlaw_dense_v3_<tag>_cache.npz.  Version 2's own description follows."""

REPLACEMENTS = [
    ('docstring header',
     r"firstlaw_records_v2.py -- FIRSTLAW_records_v2 (reorientation 4, C15; solve entry item 8)",
     V3_DOC),
    ('version-3 helpers after kappa_formula',
     lines(r"def kappa_formula(N_R, R, rho_R, J_R, G):",
           r"    return N_R / (2.0 * R) * (1.0 - 8.0 * np.pi * G * R * R * (rho_R - J_R))"),
     lines(r"def kappa_formula(N_R, R, rho_R, J_R, G):",
           r"    return N_R / (2.0 * R) * (1.0 - 8.0 * np.pi * G * R * R * (rho_R - J_R))",
           "", "") + V3_HELPERS.strip('\n')),
    ('REC_NAN_KEYS',
     r"'kernel_row_e_rel_ext', 'kernel_row_g_rel_ext', 'sigma_R', 'sigma_R_stored', 'sigma_R_scan', 'sigma_R_resid')",
     r"'kernel_row_e_rel_ext', 'kernel_row_g_rel_ext', 'sigma_R', 'sigma_R_stored', 'sigma_R_scan', 'sigma_R_resid') + V3_REC_NAN_KEYS"),
    ('analyse_record signature',
     r"def analyse_record(k, rec, row, d, r_fine, h_fine, G, stop_E, h_site, opts):",
     r"def analyse_record(k, rec, row, d, r_fine, h_fine, G, stop_E, h_site, opts, stress=None):"),
    ('analyse_record: the version-3 record columns after the arrays',
     r"        arrays = {'N': N, 'v': v, 'rho': rho, 'J': J, 'rho_p': a.get('rho_p'), 'J_p': a.get('J_p'), 'e': e, 'g': g, 'M': M, 'Nplusv': f}",
     lines(r"        arrays = {'N': N, 'v': v, 'rho': rho, 'J': J, 'rho_p': a.get('rho_p'), 'J_p': a.get('J_p'), 'e': e, 'g': g, 'M': M, 'Nplusv': f}",
           r"        v3_record_columns(out, arrays, r_fine, h_fine, G, opts, d, has_root_dense, stress, Np)")),
    ('analyse signature',
     r"def analyse(records, F, run_args, opts, dense, r_fine, prov=None, notes=None):",
     r"def analyse(records, F, run_args, opts, dense, r_fine, prov=None, notes=None, stress=None):"),
    ('analyse: the stress of each record',
     r"        rec_out, rec_arr = analyse_record(k, records[k], int(rows[k]), dense.get(k), r_fine, h_fine, G, stop_E, h_site_eff, opts)",
     lines(r"        rec_out, rec_arr = analyse_record(k, records[k], int(rows[k]), dense.get(k), r_fine, h_fine, G, stop_E, h_site_eff, opts,",
           r"                                          stress=((stress or {}).get('records') or {}).get(k))")),
    ('analyse: the version-3 definitions and the stress meta',
     lines(r'is printed as shift_pred."),',
           r"    }",
           r"    result = {",
           r"        'tag': opts.tag,"),
     lines(r'is printed as shift_pred."),',
           r"    }",
           r"    definitions.update(v3_definitions(opts, stress, h_fine))",
           r"    result = {",
           r"        'tag': opts.tag,",
           r"        'stress_meta': _jsonable((stress or {}).get('meta') or {}),")),
    ('analyse_pair: the version-3 pair columns',
     lines(r"    pr['_f_mid'] = f_mid",
           r"    pr['_f_defect'] = fD",
           r"    return pr"),
     lines(r"    pr['_f_mid'] = f_mid",
           r"    pr['_f_defect'] = fD",
           r"    v3_pair_columns(pr, A, B, aA, aB, Rg, sl, Rm, dA, pre, deg)",
           r"    return pr")),
    ('closure_stats: the version-3 record statistics',
     lines(r"        st['kappa_formula_mean'] = _nan(np.mean, [_float(r.get('kappa_formula')) for r in rooted_ok])",
           r"    if not pairs:",
           r"        return st"),
     lines(r"        st['kappa_formula_mean'] = _nan(np.mean, [_float(r.get('kappa_formula')) for r in rooted_ok])",
           r"    v3_record_stats(st, rooted_ok)",
           r"    if not pairs:",
           r"        return st")),
    ('closure_stats: the version-3 pair statistics',
     lines(r"    st['sum_res_over_sum_dE'] = st['sum_res'] / st['sum_dE'] if st['sum_dE'] != 0 else NAN",
           r"    return st"),
     lines(r"    st['sum_res_over_sum_dE'] = st['sum_res'] / st['sum_dE'] if st['sum_dE'] != 0 else NAN",
           r"    v3_pair_stats(st, pairs)",
           r"    return st")),
    ('write_md: title',
     r"    L.append('# FIRSTLAW_records_v2 -- %s' % res['tag'])",
     r"    L.append('# FIRSTLAW_records_v3 -- %s' % res['tag'])"),
    ('write_md: the stress line under Inputs',
     lines(r"    if pv.get('cache'):",
           r"        L.append('- cache: %s' % pv['cache'])"),
     lines(r"    if pv.get('cache'):",
           r"        L.append('- cache: %s' % pv['cache'])",
           r"    L.append('- radial stress p_r (version 3): %s' % v3_stress_line(res))")),
    ('write_md: the version-3 summary section',
     lines(r"    L.append('')",
           r"    L.append('## Closure statistics (included pairs)')"),
     lines(r"    L.extend(v3_md_summary(res))",
           r"    L.append('')",
           r"    L.append('## Closure statistics (included pairs)')")),
    ('write_md: the version-3 tables',
     lines(r"    L.append('')",
           r"    L.append('## Thresholds (named arguments)')"),
     lines(r"    L.extend(v3_md_tables(res))",
           r"    L.append('')",
           r"    L.append('## Thresholds (named arguments)')")),
    ('write_outputs: base name',
     r"    base = os.path.join(out_dir, 'firstlaw_records_v2_%s' % tag)",
     r"    base = os.path.join(out_dir, 'firstlaw_records_v3_%s' % tag)"),
    ('write_outputs: the version-3 arrays',
     r"    arrays['options_json'] = np.array(json.dumps(_jsonable(vars(opts))))",
     lines(r"    arrays.update(v3_output_arrays(res, r_fine))",
           r"    arrays['options_json'] = np.array(json.dumps(_jsonable(vars(opts))))")),
    ('parse_args: description',
     r"    p = argparse.ArgumentParser(description='FIRSTLAW_records_v2: the first-law identity between consecutive rooted records, on the dense solution re-solved with the engine')",
     r"    p = argparse.ArgumentParser(description='FIRSTLAW_records_v3: the first-law identity between consecutive rooted records, on the dense solution re-solved with the engine, plus T_ll = rho + p_r - 2J and the row defect at every dense root')"),
    ('parse_args: the version-3 arguments',
     lines(r"    p.add_argument('--list-no-root-pairs', dest='list_no_root_pairs', action='store_true')",
           r"    return p.parse_args(argv)"),
     lines(r"    p.add_argument('--list-no-root-pairs', dest='list_no_root_pairs', action='store_true')",
           r"    v3_add_arguments(p)",
           r"    return p.parse_args(argv)")),
    ('analyse: the version-3 thresholds',
     r"'uniform_rtol', 'kappa_column', 'T_start', 'T_end', 'workers', 'threads')}",
     r"'uniform_rtol', 'kappa_column', 'T_start', 'T_end', 'workers', 'threads') + V3_THRESHOLD_KEYS}"),
    ('main: the cache file name',
     r"    cache_file = None if opts.no_cache else os.path.join(out_dir, 'firstlaw_dense_v2_%s_cache.npz' % opts.tag)",
     r"    cache_file = None if opts.no_cache else os.path.join(out_dir, 'firstlaw_dense_v3_%s_cache.npz' % opts.tag)"),
    ('main: the seed cache',
     r"    dense_by_row = dict(cached)",
     lines(r"    seeded, smsg = v3_seed_cache(opts, ckey, r_fine, sargs, module_name)",
           r"    if opts.seed_cache:",
           r"        cmsg += '; seed cache %s: %s' % (opts.seed_cache, smsg)",
           r"    dense_by_row = dict(seeded)",
           r"    dense_by_row.update(cached)")),
    ('main: the first console line',
     r"    print('FIRSTLAW_records_v2 %s: %d records matched",
     r"    print('FIRSTLAW_records_v3 %s: %d records matched"),
    ('main: write the own cache also when seeded',
     r"    if cache_file is not None and results:",
     r"    if cache_file is not None and (results or seeded):"),
    ('main: the stress pass',
     r"    dense = {k: dense_by_row[int(rows[k])] for k in idx if int(rows[k]) in dense_by_row}",
     lines(r"    dense = {k: dense_by_row[int(rows[k])] for k in idx if int(rows[k]) in dense_by_row}",
           r"    stress, stress_secs = v3_stress_pass(FLD, F, opts, dense, rows, idx, r_fine, r_last_node, sargs, ell_pass, notes)")),
    ('main: analyse with the stress',
     r"    res = analyse(records, F, run_args, opts, dense, r_fine, prov=prov, notes=notes)",
     r"    res = analyse(records, F, run_args, opts, dense, r_fine, prov=prov, notes=notes, stress=stress)"),
    ('main: the summary console line',
     r"    print('FIRSTLAW_records_v2 %s: %d records analysed",
     r"    print('FIRSTLAW_records_v3 %s: %d records analysed"),
    ('main: the version-3 console lines and the bitwise check',
     lines(r"    print('wrote %s.{json,npz,md} in %.1f s (solves %.1f s)' % (base, time.time() - t0, secs))",
           r"    return 0"),
     lines(r"    v3_console(res, stress_secs)",
           r"    print('wrote %s.{json,npz,md} in %.1f s (solves %.1f s)' % (base, time.time() - t0, secs))",
           r"    v3_check_against(opts, base + '.json')",
           r"    return 0")),
]


def sha256_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description='write firstlaw_records_v3.py from firstlaw_records_v2.py by exact string replacements')
    p.add_argument('--src', default=os.path.join(HERE, 'firstlaw_records_v2.py'))
    p.add_argument('--dst', default=os.path.join(HERE, 'firstlaw_records_v3.py'))
    p.add_argument('--force', action='store_true', help='overwrite a destination that exists with other contents')
    a = p.parse_args(argv)
    with open(a.src) as fh:
        src = fh.read()
    problems = []
    for name, old, new in REPLACEMENTS:
        n = src.count(old)
        if n != 1:
            problems.append("replacement '%s': the OLD text occurs %d times in %s (exactly 1 required)" % (name, n, a.src))
    if problems:
        print('\n'.join(problems))
        raise SystemExit('nothing written: the source is not the firstlaw_records_v2.py these replacements were copied from')
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
            return 0
        raise SystemExit('%s exists with other contents; --force overwrites it' % a.dst)
    with open(a.dst, 'w') as fh:
        fh.write(out)
    print('source %s: %d lines, sha256 %s' % (a.src, src.count('\n'), sha256_text(src)))
    print('wrote  %s: %d lines, sha256 %s (%d replacements applied)' % (a.dst, out.count('\n'), sha256_text(out), len(REPLACEMENTS)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
