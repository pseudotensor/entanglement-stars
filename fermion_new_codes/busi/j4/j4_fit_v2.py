#!/usr/bin/env python3
"""
j4_fit_v2.py -- the version-2 analysis of the J4 ladder (busi lane, task 1; spec busi/specs/J4_spec_v2.md,
sections 4 and 5).

Steps: (1) read every record (version 1 and 2), identify each ball by its canonical label (the largest sum of
three squares <= n2max) and its box L; (2) extrapolate the finite box per ball (3 boxes: S = S_inf + b4 u +
b6 u^{3/2}, u = (R_vol/L)^4; 2 boxes: the slope; 1 box: the reference b4, b6 from the 3-box balls of the
largest radii); (3) form the Hann radius-window average (full width 2) of each primary radius with the exact
weights and the exact window-averaged regressors, balls not yet computed passing their weight to the nearest
computed ball; (4) fit S_bar = 4 pi eta <R^2> + c1 <R> + c_log <ln R> + c0 (c_log fixed at -44/45) and the
variants of the spec; (5) the oscillation diagnostic (per-ball residual harmonics at the lattice frequencies)
and the window-leakage bound; (6) the error statement.  --selftest runs the pipeline on synthetic data.

Usage:
  j4_fit_v2.py --dirs prod_v1 prod_v2 [--radii 12 ... 62] [--wide-upto 40] [--out J4_fit_v2.json]
               [--sphere-average 0.9192 --sphere-average-err 0.0002] [--no-site-check]
  j4_fit_v2.py --selftest [--osc 3.0] [--seed 1]
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import j4window_v2 as jw  # noqa: E402

C_LOG = -44.0 / 45.0
B4_CONTINUUM = -32.96566448807753
RADII_DEFAULT = [12, 16, 20, 24, 29, 34, 40, 48, 56, 62]
COL = {name: i for i, name in enumerate(jw.REGRESSORS)}
NREG = len(jw.REGRESSORS)
HARMONICS = [('1', 1.0), ('sqrt2', math.sqrt(2.0)), ('sqrt5', math.sqrt(5.0)), ('sqrt3', math.sqrt(3.0)), ('2', 2.0),
             ('sqrt6', math.sqrt(6.0)), ('sqrt3/2', 0.5 * math.sqrt(3.0)), ('sqrt8', math.sqrt(8.0)), ('3', 3.0),
             ('sqrt10', math.sqrt(10.0)), ('sqrt13', math.sqrt(13.0)), ('sqrt11/2', 0.5 * math.sqrt(11.0)),
             ('sqrt11', math.sqrt(11.0)), ('sqrt12', math.sqrt(12.0)), ('sqrt14', math.sqrt(14.0)), ('4', 4.0)]
ENVELOPE_VARIANTS = ['V2a', 'V2b', 'V2c', 'V3', 'V3b', 'V4', 'V4b', 'V8', 'V10']
VARIANT_ORDER = ['V0', 'V1', 'V2a', 'V2b', 'V2c', 'V3', 'V3b', 'V4', 'V4b', 'V5', 'V6', 'V7', 'V8', 'V10']


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ----------------------------------------------------------------------------------------------
# records
# ----------------------------------------------------------------------------------------------
def load_records(dirs, check_sites=True, log=print):
    """(canonical ball, L) -> dict(n, L, S, n_sites, R_vol, file, method); duplicates keep the last and report the disagreement."""
    jl = None
    if check_sites:
        import j4lib as jl
    recs, dup_max, nfiles = {}, 0.0, 0
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, '*.json'))):
            try:
                with open(f) as fh:
                    r = json.load(fh)
            except Exception:
                continue
            if not all(k in r for k in ('S', 'L', 'n2max', 'n')) or r['S'] != r['S']:
                continue
            nfiles += 1
            n_c = jw.canonical_ball(int(r['n2max']))
            L = int(r['L'])
            ns = int(r['n'])
            if check_sites and len(jl.ball_sites(n2max=n_c)) != ns:
                raise ArithmeticError('%s: site count %d disagrees with the canonical ball %d' % (f, ns, n_c))
            rec = dict(n=n_c, L=L, S=float(r['S']), n_sites=ns, R_vol=(3.0 * ns / (4.0 * np.pi)) ** (1.0 / 3.0),
                       file=f, method=r.get('method'))
            key = (n_c, L)
            if key in recs:
                dup_max = max(dup_max, abs(recs[key]['S'] - rec['S']))
            recs[key] = rec
    log('records: %d files, %d distinct (ball, box) cases, largest duplicate disagreement %.2e nats' % (nfiles, len(recs), dup_max))
    return recs


# ----------------------------------------------------------------------------------------------
# finite box
# ----------------------------------------------------------------------------------------------
def radius_group(R, radii, wide_upto):
    best = None
    for Rc in radii:
        W = 3.0 if Rc <= wide_upto else 2.0
        if abs(R - Rc) <= 0.5 * W + 0.1:
            best = Rc
    return best


def box_extrapolate(recs, radii, wide_upto, b_fallback=(-32.6, -13.5)):
    """Per canonical ball: S_inf and the box coefficients; the reference b4, b6 from the 3-box balls of the 3 largest radii."""
    byball = {}
    for (n, L), r in recs.items():
        byball.setdefault(n, {})[L] = r
    ext, three = {}, []
    for n in sorted(byball):
        d = byball[n]
        Ls = sorted(d)
        R_vol = d[Ls[0]]['R_vol']
        S = np.array([d[L]['S'] for L in Ls], dtype=float)
        u = (R_vol / np.array(Ls, dtype=float)) ** 4
        v = u ** 1.5
        info = dict(n=n, R_vol=R_vol, n_sites=d[Ls[0]]['n_sites'], Ls=Ls, S_by_L=S.tolist(), nbox=len(Ls), S_raw=float(S[-1]),
                    L_raw=int(Ls[-1]), group=radius_group(R_vol, radii, wide_upto), S_inf=None, b4=None, b6=None,
                    box_correction=None, box_unc=0.0)
        if len(Ls) >= 3:
            X = np.column_stack([np.ones_like(u), u, v])
            coef = np.linalg.lstsq(X, S, rcond=None)[0]
            info.update(S_inf=float(coef[0]), b4=float(coef[1]), b6=float(coef[2]), box_correction=float(coef[0] - S[-1]))
            three.append(info)
        elif len(Ls) == 2:
            b4e = float((S[1] - S[0]) / (u[1] - u[0]))
            info.update(S_inf=float(S[1] - b4e * u[1]), b4=b4e, box_correction=float(-b4e * u[1]))
        ext[n] = info
    groups = {}
    for info in three:
        groups.setdefault(info['group'], []).append(info)
    per_group = {}
    for g, lst in groups.items():
        b4s = np.array([x['b4'] for x in lst])
        b6s = np.array([x['b6'] for x in lst])
        per_group[g] = dict(count=len(lst), b4_median=float(np.median(b4s)), b4_spread=float(np.max(np.abs(b4s - np.median(b4s)))),
                            b6_median=float(np.median(b6s)), b6_spread=float(np.max(np.abs(b6s - np.median(b6s)))))
    eligible = sorted([g for g in per_group if g is not None and per_group[g]['count'] >= 4])[-3:]
    if eligible:
        sel = [x for g in eligible for x in groups[g]]
        b4s = np.array([x['b4'] for x in sel])
        b6s = np.array([x['b6'] for x in sel])
        b4_ref, b6_ref = float(np.median(b4s)), float(np.median(b6s))
        sp4, sp6 = float(np.max(np.abs(b4s - b4_ref))), float(np.max(np.abs(b6s - b6_ref)))
        source = 'medians over the %d 3-box balls of radii %s' % (len(sel), eligible)
    else:
        b4_ref, b6_ref = float(b_fallback[0]), float(b_fallback[1])
        sp4, sp6 = 0.5, 5.0
        source = 'fallback values (no radius with 4 or more 3-box balls)'
    for info in ext.values():
        if info['nbox'] == 1:
            u1 = (info['R_vol'] / info['L_raw']) ** 4
            v1 = u1 ** 1.5
            info.update(S_inf=info['S_raw'] - b4_ref * u1 - b6_ref * v1, box_correction=-b4_ref * u1 - b6_ref * v1,
                        box_unc=sp4 * u1 + sp6 * v1)
        elif info['nbox'] == 2:
            u0, u1 = [(info['R_vol'] / L) ** 4 for L in info['Ls']]
            info['box_unc'] = abs(b6_ref) * abs(u1 ** 1.5 - u1 * (u1 ** 1.5 - u0 ** 1.5) / (u1 - u0))
    corrections = {'%dR' % k: -b4_ref / float(k) ** 4 - b6_ref / float(k) ** 6 for k in (4, 6, 8)}
    box = dict(b4_ref=b4_ref, b6_ref=b6_ref, spread4=sp4, spread6=sp6, source=source, per_group=per_group,
               n3=len(three), n2=sum(1 for x in ext.values() if x['nbox'] == 2), n1=sum(1 for x in ext.values() if x['nbox'] == 1),
               corrections=corrections, b4_continuum=B4_CONTINUUM)
    return ext, box


# ----------------------------------------------------------------------------------------------
# window averages
# ----------------------------------------------------------------------------------------------
def complete_weights(omega, R_mid, have):
    """Weights of the balls not computed go to the nearest computed ball (in R)."""
    om = np.where(have, omega, 0.0).astype(float)
    idx = np.where(have)[0]
    for i in np.where(~have)[0]:
        j = idx[np.argmin(np.abs(R_mid[idx] - R_mid[i]))]
        om[j] += omega[i]
    return om


def window_rows(ext, radii, W, window, shift=0.0, wide_upto=40, value='S_inf'):
    rows = []
    for Rc in radii:
        tab = jw.window_table(float(Rc) + shift, W, window)
        vals = np.array([(ext[n][value] if (n in ext and ext[n][value] is not None) else np.nan) for n in tab['n']], dtype=float)
        have = ~np.isnan(vals)
        if not have.any():
            continue
        om = complete_weights(tab['omega'], tab['R_mid'], have)
        S = float(np.sum(om[have] * vals[have]))
        Rvol = np.array([ext[n]['R_vol'] for n in tab['n'][have]], dtype=float)
        Xvol = om[have] @ jw.regressor_values(Rvol)
        box_unc = float(np.sum(om[have] * np.array([ext[n]['box_unc'] for n in tab['n'][have]], dtype=float)))
        rows.append(dict(Rc=float(Rc), W=float(W), window=window, shift=float(shift), S=S,
                         X=np.array(tab['exact'], dtype=float), Xvol=np.asarray(Xvol, dtype=float),
                         completeness=float(tab['omega'][have].sum()), n_have=int(have.sum()), n_total=int(len(tab['n'])),
                         box_unc=box_unc,
                         balls=dict(n=tab['n'][have], R_mid=tab['R_mid'][have], S_inf=vals[have], omega=om[have], means=tab['means'][have])))
    return rows


def v1_rows(ext, radii, value='S_inf'):
    """The version-1 estimator: the 8 offset balls with equal weights and point-evaluated regressors."""
    rows = []
    for Rc in radii:
        labels = jw.v1_balls(Rc)
        if not all(n in ext and ext[n][value] is not None for n in labels):
            continue
        Rpts = np.array([Rc + k / 16.0 for k in jw.V1_OFFSETS], dtype=float)
        vals = np.array([ext[n][value] for n in labels], dtype=float)
        Rvol = np.array([ext[n]['R_vol'] for n in labels], dtype=float)
        rows.append(dict(Rc=float(Rc), W=0.875, window='v1', shift=0.0, S=float(vals.mean()),
                         X=jw.regressor_values(Rpts).mean(axis=0), Xvol=jw.regressor_values(Rvol).mean(axis=0),
                         completeness=1.0, n_have=8, n_total=8, box_unc=float(np.mean([ext[n]['box_unc'] for n in labels])),
                         balls=dict(n=np.array(labels), R_mid=Rpts, S_inf=vals, omega=np.full(8, 0.125), means=jw.regressor_values(Rpts))))
    return rows


# ----------------------------------------------------------------------------------------------
# fits
# ----------------------------------------------------------------------------------------------
def fit(rows, cols, fixed_log=True, drop_below=None, weights=None, Xkey='X'):
    rows = sorted([r for r in rows if drop_below is None or r['Rc'] >= drop_below], key=lambda r: r['Rc'])
    p = len(cols)
    if len(rows) < p + 1:
        return None
    Xall = np.array([r[Xkey] for r in rows], dtype=float)
    y = np.array([r['S'] for r in rows], dtype=float)
    if fixed_log:
        y = y - C_LOG * Xall[:, COL['lnR']]
    X = Xall[:, cols]
    w = np.ones(len(y)) if weights is None else np.array([float(weights(r)) for r in rows])
    sw = np.sqrt(w)
    pinv = np.linalg.pinv(X * sw[:, None])
    beta = pinv @ (y * sw)
    resid = y - X @ beta
    dof = len(y) - p
    sigma2 = float(np.sum(w * resid ** 2) / dof) if dof > 0 else float('nan')
    cov = sigma2 * (pinv @ pinv.T)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    row_eta = pinv[0] * sw
    free = (not fixed_log) and (COL['lnR'] in cols)
    return dict(eta=float(beta[0]), se_eta=float(se[0]), beta=beta, se=se, cols=list(cols), names=[jw.REGRESSORS[c] for c in cols],
                fixed_log=bool(fixed_log), c_log=(float(beta[cols.index(COL['lnR'])]) if free else C_LOG),
                se_c_log=(float(se[cols.index(COL['lnR'])]) if free else 0.0), radii=[r['Rc'] for r in rows],
                resid=resid, dof=int(dof), rms=float(np.sqrt(np.mean(resid ** 2))), row_eta=row_eta,
                lnR_bar=Xall[:, COL['lnR']], se_unit=float(np.sqrt(np.sum(row_eta ** 2))), Xkey=Xkey)


def full_beta(res):
    """Coefficient vector over all 7 regressors (zeros where absent; c_log at its slot when fixed)."""
    b = np.zeros(NREG)
    for c, val in zip(res['cols'], res['beta']):
        b[c] = val
    if res['fixed_log']:
        b[COL['lnR']] = C_LOG
    return b


def param(res, name):
    return float(res['beta'][res['names'].index(name)]) if name in res['names'] else None


# ----------------------------------------------------------------------------------------------
# oscillation diagnostic and leakage
# ----------------------------------------------------------------------------------------------
def harmonic_diagnostic(row, beta_full, W_primary=2.0, window_primary='hann', balls_per_harmonic=6):
    b = row['balls']
    e = b['S_inf'] - b['means'] @ beta_full
    R = b['R_mid']
    N = len(R)
    out = dict(ok=False, Rc=row['Rc'], n_balls=int(N), rms_total=float(np.sqrt(np.mean(e ** 2))) if N else float('nan'))
    k = min(len(HARMONICS), N // (2 * balls_per_harmonic))
    if k < 2:
        return out
    freqs = np.array([f for _, f in HARMONICS[:k]])
    names = [nm for nm, _ in HARMONICS[:k]]
    A = np.empty((N, 2 * k))
    for j, f in enumerate(freqs):
        A[:, 2 * j] = np.cos(2.0 * np.pi * f * R)
        A[:, 2 * j + 1] = np.sin(2.0 * np.pi * f * R)
    coef = np.linalg.lstsq(A, e, rcond=None)[0]
    amp = np.sqrt(coef[0::2] ** 2 + coef[1::2] ** 2)
    rem = e - A @ coef
    resp = np.abs(jw.window_ft(freqs, W_primary, window_primary))
    fgrid = np.arange(float(freqs.max()), 12.0, 1e-3)
    resp_tail = float(np.max(np.abs(jw.window_ft(fgrid, W_primary, window_primary))))
    rms_rem = float(np.sqrt(np.mean(rem ** 2)))
    out.update(ok=True, names=names, freqs=freqs, amp=amp, response=resp, rms_remainder=rms_rem, response_tail=resp_tail,
               bound=float(np.sum(amp * resp) + resp_tail * math.sqrt(2.0) * rms_rem), cond=float(np.linalg.cond(A)))
    return out


def shift_test(ext, radii_wide, beta_full, W=2.0, window='hann', wide_upto=40, shifts=None):
    """At the radii with width-3 data: the fixed-log residual of the primary window shifted by s in [-1/2, 1/2]."""
    if shifts is None:
        shifts = np.linspace(-0.5, 0.5, 9)
    out = {}
    for Rc in radii_wide:
        eps = []
        for s in shifts:
            rows = window_rows(ext, [Rc], W, window, shift=float(s), wide_upto=wide_upto)
            if not rows or rows[0]['completeness'] < 0.9999:
                eps.append(np.nan)
            else:
                eps.append(rows[0]['S'] - rows[0]['X'] @ beta_full)
        eps = np.array(eps, dtype=float)
        if np.isnan(eps).any():
            continue
        i0 = int(np.argmin(np.abs(shifts)))
        out[Rc] = dict(shifts=shifts, eps=eps, b=float(np.max(np.abs(eps - eps[i0]))), rms=float(np.std(eps)))
    return out


# ----------------------------------------------------------------------------------------------
# the analysis
# ----------------------------------------------------------------------------------------------
def analyze(recs, radii, wide_upto, log=print, sphere=(0.9192, 0.0002), b_fallback=(-32.6, -13.5)):
    ext, box = box_extrapolate(recs, radii, wide_upto, b_fallback)
    log('\nfinite box: %d balls with 3 boxes, %d with 2, %d with 1' % (box['n3'], box['n2'], box['n1']))
    for g in sorted(k for k in box['per_group'] if k is not None):
        pg = box['per_group'][g]
        log('  R %2d  3-box balls %3d  b4 median %8.3f (spread %.3f)  b6 median %7.2f (spread %.2f)'
            % (g, pg['count'], pg['b4_median'], pg['b4_spread'], pg['b6_median'], pg['b6_spread']))
    log('  reference b4 %.3f +- %.3f, b6 %.2f +- %.2f (%s; continuum b4 %.3f)'
        % (box['b4_ref'], box['spread4'], box['b6_ref'], box['spread6'], box['source'], B4_CONTINUUM))
    log('  corrections at L = 4R, 6R, 8R: %.4f, %.4f, %.4f nats' % tuple(box['corrections'][k] for k in ('4R', '6R', '8R')))

    rows = window_rows(ext, radii, 2.0, 'hann', wide_upto=wide_upto)
    if len(rows) < 4:
        log('fewer than 4 radii with data; no fit')
        return None
    log('\nwindow data: Hann window of full width 2 at each primary radius (exact weights; missing balls pass their weight to the nearest computed ball)')
    log('  R    balls    completeness  S_bar             S_bar/(4 pi <R^2>)  box unc (nats)')
    for r in rows:
        log('  %2.0f  %3d/%3d   %.4f        %15.6f   %.6f            %.4f'
            % (r['Rc'], r['n_have'], r['n_total'], r['completeness'], r['S'], r['S'] / r['X'][0], r['box_unc']))
    rows_rect = window_rows(ext, radii, 2.0, 'rect', wide_upto=wide_upto)
    rows_v1 = v1_rows(ext, radii)
    rows_raw = window_rows(ext, radii, 2.0, 'hann', wide_upto=wide_upto, value='S_raw')

    c = COL
    V = {}
    specs = [('V0', dict(rows=rows, cols=[c['4piR2'], c['R'], c['one']]), 'Hann W=2, c_log fixed, c1 free (primary)'),
             ('V1', dict(rows=rows, cols=[c['4piR2'], c['R'], c['lnR'], c['one']], fixed_log=False), 'c_log free (information)'),
             ('V2a', dict(rows=rows, cols=[c['4piR2'], c['R'], c['one'], c['invR']]), '+ c_-1/R'),
             ('V2b', dict(rows=rows, cols=[c['4piR2'], c['R'], c['one'], c['invR2']]), '+ c_-2/R^2'),
             ('V2c', dict(rows=rows, cols=[c['4piR2'], c['R'], c['one'], c['invR2'], c['lnR_invR2']]), '+ c_-2/R^2 + c_-2L ln R/R^2'),
             ('V3', dict(rows=rows, cols=[c['4piR2'], c['R'], c['one']], drop_below=20), 'drop R < 20'),
             ('V3b', dict(rows=rows, cols=[c['4piR2'], c['R'], c['one']], drop_below=29), 'drop R < 29'),
             ('V4', dict(rows=rows, cols=[c['4piR2'], c['one']]), 'c1 = 0 (complement symmetry)'),
             ('V4b', dict(rows=rows, cols=[c['4piR2'], c['one'], c['invR2']]), 'c1 = 0, + c_-2/R^2'),
             ('V5', dict(rows=rows_rect, cols=[c['4piR2'], c['R'], c['one']]), 'rectangular W=2 (information)'),
             ('V6', dict(rows=rows_v1, cols=[c['4piR2'], c['R'], c['one']]), 'version-1 estimator on the same balls (information)'),
             ('V7', dict(rows=rows_raw, cols=[c['4piR2'], c['R'], c['one']]), 'no box correction (information)'),
             ('V8', dict(rows=rows, cols=[c['4piR2'], c['R'], c['one']], Xkey='Xvol'), 'regressors from the volume radius per ball'),
             ('V10', dict(rows=rows, cols=[c['4piR2'], c['R'], c['one']], weights=lambda r: 1.0 / r['Rc']), 'weights 1/R')]
    for name, kw, note in specs:
        rws = kw.pop('rows')
        res = fit(rws, **kw) if rws else None
        if res is not None:
            res['note'] = note
            V[name] = res
    if 'V0' not in V:
        log('no primary fit')
        return None
    log('\nfits: S_bar = 4 pi eta <R^2> + c1 <R> + c_log <ln R> + c0 (+ terms); c_log = -44/45 unless free')
    for name in VARIANT_ORDER:
        if name not in V:
            continue
        res = V[name]
        c1 = param(res, 'R')
        c0 = param(res, 'one')
        extra = ' '.join('%s %+.3g' % (nm, param(res, nm)) for nm in ('invR', 'invR2', 'lnR_invR2') if nm in res['names'])
        log('  %-4s eta %.6f +- %.6f  c_log %+.4f  c1 %s  c0 %+.3f  %s rms %.4f nats  radii %s  [%s]'
            % (name, res['eta'], res['se_eta'], res['c_log'], ('%+.4f' % c1) if c1 is not None else '   0   ',
               c0 if c0 is not None else float('nan'), extra, res['rms'], [int(x) for x in res['radii']], res['note']))
    V0 = V['V0']
    log('  V0 residuals (nats): ' + ' '.join('%d:%+.3f' % (R, r) for R, r in zip(V0['radii'], V0['resid'])))
    log('  V0 eta standard error per nat of homogeneous per-radius noise: %.3e' % V0['se_unit'])
    if 'V1' in V:
        log('  V1 c_log = %+.3f +- %.3f against the continuum -44/45 = %+.4f (not a test at this lever arm when the error exceeds 1)'
            % (V['V1']['c_log'], V['V1']['se_c_log'], C_LOG))

    beta0 = full_beta(V0)
    radii_wide = [R for R in radii if R <= wide_upto]
    shifts = shift_test(ext, radii_wide, beta0, wide_upto=wide_upto)
    rows_diag = window_rows(ext, radii_wide, 3.0, 'hann', wide_upto=wide_upto) + [r for r in rows if r['Rc'] > wide_upto]
    harm = {r['Rc']: harmonic_diagnostic(r, beta0) for r in rows_diag}
    log('\noscillation diagnostic (per-ball residuals about the V0 smooth model; amplitudes in nats at the lattice frequencies) and leakage')
    log('  R    balls  rms(resid)  A(1)     A(sqrt2) A(sqrt5) A(sqrt3) A(2)     A(sqrt3/2) remainder  harmonic bound  shift-test b  V0 resid')
    b_of = {}
    for r in rows:
        Rc = r['Rc']
        h = harm.get(Rc, dict(ok=False, rms_total=float('nan')))
        sh = shifts.get(Rc)
        b_h = h['bound'] if h.get('ok') else float('nan')
        b_s = sh['b'] if sh else float('nan')
        b_of[Rc] = b_s if sh else b_h
        amps = {nm: a for nm, a in zip(h.get('names', []), h.get('amp', []))} if h.get('ok') else {}

        def A(nm):
            return ('%8.3f' % amps[nm]) if nm in amps else '     -  '
        i0 = V0['radii'].index(Rc) if Rc in V0['radii'] else None
        log('  %2.0f  %4d   %9.3f  %s %s %s %s %s %s  %9.3f  %12.4f   %12s  %8s'
            % (Rc, h.get('n_balls', 0), h['rms_total'], A('1'), A('sqrt2'), A('sqrt5'), A('sqrt3'), A('2'), A('sqrt3/2'),
               h.get('rms_remainder', float('nan')), b_h, ('%.4f' % b_s) if sh else '-', ('%+.4f' % V0['resid'][i0]) if i0 is not None else '-'))
    half_shift_flag = any(h.get('ok') and 'sqrt3/2' in h['names'] and '1' in h['names'] and
                          h['amp'][h['names'].index('sqrt3/2')] > 0.1 * max(h['amp'][h['names'].index('1')], 1e-9) for h in harm.values())
    if half_shift_flag:
        log('  WARNING: a half-shifted frequency (sqrt3/2) carries more than 10 percent of the f = 1 amplitude at some radius: see spec section 2.5 (W = 4 contingency)')

    w = V0['row_eta']
    bvec = np.array([b_of[R] for R in V0['radii']], dtype=float)
    missing = np.isnan(bvec)
    if missing.any():
        log('  no leakage bound at radii %s (too few balls); they enter env_window with the largest available bound' % [int(R) for R, m in zip(V0['radii'], missing) if m])
        bvec[missing] = np.nanmax(bvec) if (~missing).any() else 0.0
    env_window = float(np.sum(np.abs(w) * bvec))
    env_members = {nm: V[nm]['eta'] - V0['eta'] for nm in ENVELOPE_VARIANTS if nm in V}
    env_model = float(max(abs(v) for v in env_members.values())) if env_members else 0.0
    env_argmax = max(env_members, key=lambda k: abs(env_members[k])) if env_members else None
    deta_dclog = float(-np.sum(w * V0['lnR_bar']))
    dlog = abs(deta_dclog) * (44.0 / 45.0)
    two_se = 2.0 * V0['se_eta']
    half = math.sqrt(two_se ** 2 + env_model ** 2 + env_window ** 2 + dlog ** 2)
    eta = V0['eta']
    a_over_lp = 2.0 * math.sqrt(eta)
    log('\nerror statement (spec v2 section 5.3):')
    log('  2 se(V0)                                   %.6f' % two_se)
    log('  model envelope max|eta_v - eta_0| over E   %.6f (from %s)' % (env_model, env_argmax))
    log('  window leakage sum_i |w_i| b_i             %.6f' % env_window)
    log('  log sensitivity 44/45 |d eta/d c_log|      %.6f (d eta/d c_log = %.3e)' % (dlog, deta_dclog))
    log('RESULT: eta = %.5f, half-width %.5f; target 5e-4 %s' % (eta, half, 'MET' if half <= 5e-4 else 'NOT MET'))
    log('        a/l_P = 2 sqrt(eta) = %.5f +- %.5f' % (a_over_lp, half / math.sqrt(eta)))
    if sphere is not None:
        diff = eta - sphere[0]
        comb = math.sqrt(half ** 2 + sphere[1] ** 2)
        log('        sphere average under test: %.5f +- %.5f; difference %+.5f against the combined half-width %.5f: %s'
            % (sphere[0], sphere[1], diff, comb, 'consistent' if abs(diff) <= comb else 'tension'))
    for nm, what in (('V6', 'the version-1 estimator on the same balls'), ('V5', 'the rectangular window'), ('V1', 'the free-log fit')):
        if nm in V:
            log('        %s: eta %.6f (+%.6f)' % (what, V[nm]['eta'], V[nm]['eta'] - eta))
    log('        comparisons: planar [100] 0.78658, [111] 0.92338180595796; J7 mesh 4 0.9145 (labelled extrapolation 0.919)')
    return dict(eta=eta, half_width=half, two_se=two_se, env_model=env_model, env_model_from=env_argmax, env_window=env_window,
                dlog=dlog, deta_dclog=deta_dclog, a_over_lP=a_over_lp, a_over_lP_half_width=half / math.sqrt(eta),
                V=V, rows=rows, rows_rect=rows_rect, rows_v1=rows_v1, box=box, shifts=shifts, harm=harm, b=b_of,
                half_shift_flag=bool(half_shift_flag), ext_summary=[dict(n=x['n'], R_vol=x['R_vol'], nbox=x['nbox'], Ls=x['Ls'], S_inf=x['S_inf'], b4=x['b4'], b6=x['b6'],
                                                                         box_correction=x['box_correction'], box_unc=x['box_unc']) for x in ext.values()])


# ----------------------------------------------------------------------------------------------
# synthetic data and the selftest
# ----------------------------------------------------------------------------------------------
def synthetic_records(radii, wide_upto, eta0=0.919, c0=-5.0, b4=-32.6, b6=-13.5, osc=3.0, noise=0.0, seed=1):
    """Records of the production design (windows of width 3 at R <= wide_upto, 2 above, all at 8R; the version-1 balls
    at 4R, 6R, 8R for R <= 48 and at 4R only above) for S_inf = smooth(interval means) + sqrt(R) sum_f a_f cos(2 pi f R + phi_f) + noise."""
    rng = np.random.default_rng(seed)
    nmax = int(math.ceil((max(radii) + 2.0) ** 2)) + 1
    Mx = int(math.floor(math.sqrt(nmax))) + 1
    ax = np.arange(-Mx, Mx + 1)
    X1, X2, X3 = np.meshgrid(ax, ax, ax, indexing='ij')
    norms = np.sort((X1 ** 2 + X2 ** 2 + X3 ** 2).ravel())
    ms = [m for m in range(1, 65) if jw.is_sum_of_three_squares(m)]
    freqs = np.sqrt(np.array(ms, dtype=float))
    amps = osc * rng.uniform(0.5, 1.5, len(ms)) / np.array(ms, dtype=float) ** 0.75
    phases = rng.uniform(0.0, 2.0 * np.pi, len(ms))
    recs, truth = {}, {}
    for Rc in radii:
        W = 3.0 if Rc <= wide_upto else 2.0
        v1 = set(jw.v1_balls(Rc))
        for n, R_lo, R_hi in jw.balls_in_window(Rc, W):
            if n in truth:
                continue
            ns = int(np.searchsorted(norms, n, side='right'))
            R_vol = (3.0 * ns / (4.0 * np.pi)) ** (1.0 / 3.0)
            means = jw.plain_interval_means(R_lo, R_hi)
            Rm = 0.5 * (R_lo + R_hi)
            smooth = eta0 * means[COL['4piR2']] + C_LOG * means[COL['lnR']] + c0
            oscv = float(np.sqrt(Rm) * np.sum(amps * np.cos(2.0 * np.pi * freqs * Rm + phases))) if osc else 0.0
            S_inf = smooth + oscv + (noise * rng.normal() if noise else 0.0)
            truth[n] = dict(S_inf=S_inf, osc=oscv, smooth=smooth, Rc=Rc)
            boxes = [8 * Rc] if n not in v1 else ([4 * Rc, 6 * Rc, 8 * Rc] if Rc <= 48 else [4 * Rc])
            for L in boxes:
                u = (R_vol / L) ** 4
                recs[(n, L)] = dict(n=n, L=L, S=S_inf + b4 * u + b6 * u ** 1.5, n_sites=ns, R_vol=R_vol, file='synthetic', method='synthetic')
    return recs, truth, dict(eta0=eta0, c0=c0, b4=b4, b6=b6, freqs=freqs, amps=amps, phases=phases)


def selftest(radii=None, wide_upto=40, log=print, seed=1, osc=3.0):
    radii = list(radii or RADII_DEFAULT)
    quiet = lambda *a, **k: None  # noqa: E731
    results = {}
    recs, truth, par = synthetic_records(radii, wide_upto, osc=0.0, seed=seed)
    log('=== selftest A: smooth synthetic data without oscillation (exactness of records, boxes, windows, fit) ===')
    resA = analyze(recs, radii, wide_upto, log=log, sphere=None)
    V0 = resA['V']['V0']
    results['A'] = dict(d_eta=V0['eta'] - par['eta0'], c1=param(V0, 'R'), d_c0=param(V0, 'one') - par['c0'],
                        d_b4=resA['box']['b4_ref'] - par['b4'], d_b6=resA['box']['b6_ref'] - par['b6'], rms=V0['rms'])
    log('selftest A: eta - eta0 %.2e, c1 %.2e, c0 - c0_true %.2e, b4_ref - b4 %.2e, b6_ref - b6 %.2e, rms %.2e nats' % (
        results['A']['d_eta'], results['A']['c1'], results['A']['d_c0'], results['A']['d_b4'], results['A']['d_b6'], results['A']['rms']))

    recs, truth, par = synthetic_records(radii, wide_upto, osc=osc, seed=seed)
    single = np.array([truth[n]['osc'] / math.sqrt(0.5 * sum(jw.ball_interval(n))) for n in truth])
    log('\n=== selftest B: synthetic shell oscillation (single-ball rms about %.2f sqrt R nats) ===' % float(np.sqrt(np.mean(single ** 2))))
    resB = analyze(recs, radii, wide_upto, log=log, sphere=None)
    V0 = resB['V']['V0']
    results['B'] = dict(d_eta_hann=V0['eta'] - par['eta0'], half_width=resB['half_width'], env_window=resB['env_window'], env_model=resB['env_model'],
                        d_eta_v1=(resB['V']['V6']['eta'] - par['eta0']) if 'V6' in resB['V'] else float('nan'),
                        d_eta_rect=(resB['V']['V5']['eta'] - par['eta0']) if 'V5' in resB['V'] else float('nan'),
                        resid=V0['resid'], bounds=[resB['b'][R] for R in V0['radii']], single_rms_per_sqrtR=float(np.sqrt(np.mean(single ** 2))))
    covered = sum(1 for r, b in zip(V0['resid'], results['B']['bounds']) if abs(r) <= b + 1e-12)
    log('selftest B: eta - eta0: Hann exact %.2e (half-width %.2e); version-1 estimator %.2e; rectangular %.2e; leakage bound covers the V0 residual at %d of %d radii'
        % (results['B']['d_eta_hann'], results['B']['half_width'], results['B']['d_eta_v1'], results['B']['d_eta_rect'], covered, len(V0['resid'])))

    keep = set()
    for Rc in radii:
        W = 3.0 if Rc <= wide_upto else 2.0
        lev = jw.ball_levels(Rc, W)
        v1 = set(jw.v1_balls(Rc))
        for (n, L) in recs:
            if n in v1 or lev.get(n, 99) <= 5:
                keep.add((n, L))
    resC = analyze({k: v for k, v in recs.items() if k in keep}, radii, wide_upto, log=quiet, sphere=None)
    results['C'] = dict(d_eta_level5=resC['V']['V0']['eta'] - par['eta0'], completeness=[r['completeness'] for r in resC['rows']])
    log('selftest C: the level-5 sampled windows (completeness %.2f to %.2f): eta - eta0 %.2e'
        % (min(results['C']['completeness']), max(results['C']['completeness']), results['C']['d_eta_level5']))
    return results


def main():
    ap = argparse.ArgumentParser(description='J4 version-2 fit')
    ap.add_argument('--dirs', nargs='*', default=[], help='directories with production records (version 1 and 2)')
    ap.add_argument('--radii', type=int, nargs='*', default=RADII_DEFAULT)
    ap.add_argument('--wide-upto', type=int, default=40)
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-site-check', action='store_true')
    ap.add_argument('--sphere-average', type=float, default=0.9192)
    ap.add_argument('--sphere-average-err', type=float, default=0.0002)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--osc', type=float, default=3.0)
    args = ap.parse_args()
    if args.selftest:
        res = selftest(args.radii, args.wide_upto, seed=args.seed, osc=args.osc)
        if args.out:
            with open(args.out, 'w') as fh:
                json.dump(to_jsonable(res), fh, indent=1)
        return
    if not args.dirs:
        ap.error('give --dirs or --selftest')
    recs = load_records(args.dirs, check_sites=not args.no_site_check)
    res = analyze(recs, args.radii, args.wide_upto, sphere=(args.sphere_average, args.sphere_average_err))
    if res is None:
        sys.exit(1)
    if args.out:
        slim = {k: v for k, v in res.items() if k not in ('rows', 'rows_rect', 'rows_v1')}
        slim['rows'] = [{k: v for k, v in r.items() if k != 'balls'} for r in res['rows']]
        slim['V'] = {nm: {k: v for k, v in r.items() if k not in ('row_eta', 'lnR_bar')} for nm, r in res['V'].items()}
        with open(args.out, 'w') as fh:
            json.dump(to_jsonable(slim), fh, indent=1)


if __name__ == '__main__':
    main()
