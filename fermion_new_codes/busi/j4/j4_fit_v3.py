#!/usr/bin/env python3
"""
j4_fit_v3.py -- the version-3 analysis of the J4 ladder (busi lane, task 1; lead's task B4; the note
busi/notes/J4_v2_B4_v1.md).  Reads every record (version 1, version 2, prefix), identifies each ball by its canonical
label and box, and

  1. removes the finite box per ball with the fixed term A_4 (R_vol/L)^4, A_4 = (128/15) Z_AP(2) = -32.9657 (j4box_v3,
     re-derived at every run), plus the lattice remainder measured on the 3-box ladders (about 1e-4 nats at L = 8R),
     whose spread is the box bound;
  2. forms the Hann window average of full width 2 at every primary radius with exact weights and exact window-averaged
     regressors (j4window_v2); radii whose window is not complete (weight of the computed balls below 0.9999) are
     excluded from the fit unless --allow-incomplete (previews, with the weight-passing rule);
  3. fits the primary model S_bar = 4 pi eta <R^2> + c_log <ln R> + c_0 with c_log = -44/45 and c_1 = 0 (complement
     symmetry), the envelope variants E1 (c_1 free), E2 (+ c_-2/R^2), E3 (+ c_-2/R^2 + c_-2L ln R/R^2), E4 (R >= 20),
     E5 (R >= 29), E6 (volume-radius regressors), E7 (weights 1/R), and the information variants I1 (c_log free),
     I2 (rectangular window), I3/I3b (the version-1 estimator), I4 (no box correction), I5 (A_4 term only),
     I6 (+ c_-1/R), I7 (Hann width 3 where available);
  4. measures the oscillation: the pooled single-ball residual sigma_1 (rms of e_n / sqrt R), the periodogram of the
     normalized residuals of all balls (lattice and sublattice frequencies), the shift test at the radii with width-3
     data, and the Cauchy-Schwarz leakage bound per radius (j4leak_v3), validated against the shift test;
  5. prints the error statement
        half-width = sqrt((2 se)^2 + env_model^2 + env_window^2 + env_box^2 + (44/45 |d eta/d c_log|)^2)
     and the conditional set-membership (LP) interval of eta.
  --selftest runs the pipeline on synthetic data with known answers (A1 exactly representable, A2 the step-function
  construction with its predicted sawtooth leakage, B oscillation of the measured size, B2 a planted sublattice line,
  C partial windows, D the LP interval).

Usage:
  j4_fit_v3.py --dirs prod prod_v2 [--radii 12 ... 62] [--wide-upto 40] [--out fit.json] [--plot PREFIX]
               [--allow-incomplete] [--sphere-average 0.9192 --sphere-average-err 0.0002] [--no-site-check]
  j4_fit_v3.py --selftest [--osc 3.0] [--seed 1] [--out selftest.json]
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
import j4box_v3 as jb     # noqa: E402
import j4leak_v3 as jk    # noqa: E402

C_LOG = -44.0 / 45.0
RADII_DEFAULT = [12, 16, 20, 24, 29, 34, 40, 48, 56, 62]
COL = {name: i for i, name in enumerate(jw.REGRESSORS)}
NREG = len(jw.REGRESSORS)
ENVELOPE = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7']
ORDER = ['P', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'I1', 'I2', 'I3', 'I3b', 'I4', 'I5', 'I6', 'I7']
COMPLETE = 0.9999
TARGET = 5e-4
PLANAR_100 = 0.786583386553
PLANAR_111 = 0.92338180595796
J7_MESH4 = 0.9145


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
    """(canonical ball, L) -> dict(n, L, S, n_sites, R_vol, file, method, assembly); duplicates keep the last and report."""
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
            if not isinstance(r, dict) or not all(k in r for k in ('S', 'L', 'n2max', 'n')) or r['S'] != r['S']:
                continue
            nfiles += 1
            n_c = jw.canonical_ball(int(r['n2max']))
            L, ns = int(r['L']), int(r['n'])
            if check_sites and len(jl.ball_sites(n2max=n_c)) != ns:
                raise ArithmeticError('%s: site count %d disagrees with the canonical ball %d' % (f, ns, n_c))
            rec = dict(n=n_c, L=L, S=float(r['S']), n_sites=ns, R_vol=(3.0 * ns / (4.0 * np.pi)) ** (1.0 / 3.0),
                       file=f, method=r.get('method'), assembly=r.get('assembly', 'direct'))
            key = (n_c, L)
            if key in recs:
                dup_max = max(dup_max, abs(recs[key]['S'] - rec['S']))
            recs[key] = rec
    n_prefix = sum(1 for r in recs.values() if r['assembly'] == 'prefix')
    log('records: %d files, %d distinct (ball, box) cases (%d by the prefix assembly), largest duplicate disagreement %.2e nats'
        % (nfiles, len(recs), n_prefix, dup_max))
    return recs


def by_ball(recs):
    d = {}
    for (n, L), r in recs.items():
        d.setdefault(int(n), {})[int(L)] = r
    return d


# ----------------------------------------------------------------------------------------------
# window averages
# ----------------------------------------------------------------------------------------------
def complete_weights(omega, R_mid, have):
    """Weights of the balls not computed go to the nearest computed ball (in R); previews only."""
    om = np.where(have, omega, 0.0).astype(float)
    idx = np.where(have)[0]
    for i in np.where(~have)[0]:
        j = idx[np.argmin(np.abs(R_mid[idx] - R_mid[i]))]
        om[j] += omega[i]
    return om


def window_rows(ext, radii, W, window, shift=0.0, value='S_inf'):
    rows = []
    for Rc in radii:
        tab = jw.window_table(float(Rc) + shift, W, window)
        vals = np.array([(ext[n][value] if (n in ext and ext[n].get(value) is not None) else np.nan) for n in tab['n']], dtype=float)
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
                         balls=dict(n=tab['n'][have], R_mid=tab['R_mid'][have], R_lo=tab['R_lo'][have], R_hi=tab['R_hi'][have],
                                    S_inf=vals[have], omega=om[have], means=tab['means'][have])))
    return rows


def v1_rows(ext, radii, value='S_inf'):
    """The version-1 estimator: the 8 offset balls with equal weights and point-evaluated regressors."""
    rows = []
    for Rc in radii:
        labels = jw.v1_balls(Rc)
        if not all(n in ext and ext[n].get(value) is not None for n in labels):
            continue
        Rpts = np.array([Rc + k / 16.0 for k in jw.V1_OFFSETS], dtype=float)
        vals = np.array([ext[n][value] for n in labels], dtype=float)
        Rvol = np.array([ext[n]['R_vol'] for n in labels], dtype=float)
        rows.append(dict(Rc=float(Rc), W=0.875, window='v1', shift=0.0, S=float(vals.mean()),
                         X=jw.regressor_values(Rpts).mean(axis=0), Xvol=jw.regressor_values(Rvol).mean(axis=0),
                         completeness=1.0, n_have=8, n_total=8, box_unc=float(np.mean([ext[n]['box_unc'] for n in labels])),
                         balls=dict(n=np.array(labels), R_mid=Rpts, R_lo=Rpts, R_hi=Rpts, S_inf=vals, omega=np.full(8, 0.125),
                                    means=jw.regressor_values(Rpts))))
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
                se_c_log=(float(se[cols.index(COL['lnR'])]) if free else 0.0), radii=[int(round(r['Rc'])) for r in rows],
                resid=resid, dof=int(dof), rms=float(np.sqrt(np.mean(resid ** 2))), row_eta=row_eta,
                lnR_bar=Xall[:, COL['lnR']], se_unit=float(np.sqrt(np.sum(row_eta ** 2))), sum_abs_w=float(np.sum(np.abs(row_eta))),
                Xkey=Xkey, S=np.array([r['S'] for r in rows]), box_unc=np.array([r['box_unc'] for r in rows]))


def full_beta(res):
    b = np.zeros(NREG)
    for c, val in zip(res['cols'], res['beta']):
        b[c] = val
    if res['fixed_log']:
        b[COL['lnR']] = C_LOG
    return b


def param(res, name):
    return float(res['beta'][res['names'].index(name)]) if name in res['names'] else None


# ----------------------------------------------------------------------------------------------
# oscillation: pooled residuals, shift test, periodogram, LP
# ----------------------------------------------------------------------------------------------
def pooled_residuals(ext, radii, wide_upto, beta_full):
    """Per-ball residuals about the model over every computed ball of every radius (width-3 tables where they exist),
    their sqrt R-normalized pooled rms sigma_1 (weights: the interval lengths), and per-window summaries."""
    R_all, e_all, d_all, per = [], [], [], {}
    for Rc in radii:
        W = 3.0 if Rc <= wide_upto else 2.0
        rows = window_rows(ext, [Rc], W, 'hann')
        if not rows:
            continue
        b = rows[0]['balls']
        e = b['S_inf'] - b['means'] @ beta_full
        dl = b['R_hi'] - b['R_lo']
        en = e / np.sqrt(b['R_mid'])
        per[int(Rc)] = dict(n=int(len(e)), W=W, complete=bool(rows[0]['completeness'] >= COMPLETE),
                            rms=float(np.sqrt(np.mean(e ** 2))), rms_norm=float(np.sqrt(np.sum(dl * en ** 2) / np.sum(dl))))
        R_all.append(b['R_mid'])
        e_all.append(e)
        d_all.append(dl)
    if not R_all:
        return None
    R = np.concatenate(R_all)
    e = np.concatenate(e_all)
    d = np.concatenate(d_all)
    en = e / np.sqrt(R)
    return dict(R=R, e=e, d=d, en=en, sigma1=float(np.sqrt(np.sum(d * en ** 2) / np.sum(d))), per_window=per, n=int(len(e)))


def shift_test(ext, radii_wide, beta_full, shifts=None):
    """At the radii with width-3 data: the fixed-log residual of the primary window shifted by s in [-1/2, 1/2]."""
    if shifts is None:
        shifts = np.linspace(-0.5, 0.5, 21)
    out = {}
    for Rc in radii_wide:
        eps = []
        for s in shifts:
            rows = window_rows(ext, [Rc], 2.0, 'hann', shift=float(s))
            if not rows or rows[0]['completeness'] < COMPLETE:
                eps.append(np.nan)
            else:
                eps.append(rows[0]['S'] - rows[0]['X'] @ beta_full)
        eps = np.array(eps, dtype=float)
        if np.isnan(eps).any():
            continue
        i0 = int(np.argmin(np.abs(shifts)))
        out[int(Rc)] = dict(shifts=np.array(shifts), eps=eps, eps0=float(eps[i0]), half_range=float(0.5 * (eps.max() - eps.min())),
                            max_dev=float(np.max(np.abs(eps - eps[i0]))), rms=float(np.std(eps)))
    return out


def periodogram_report(pool):
    freqs_named = [f for _, f in jk.NAMED]
    freqs_half = [f for _, f in jk.HALF]
    A_named = jk.periodogram(pool['R'], pool['en'], pool['d'], freqs_named)
    A_half = jk.periodogram(pool['R'], pool['en'], pool['d'], freqs_half)
    grid = np.arange(0.3, 4.5, 0.005)
    A_grid = jk.periodogram(pool['R'], pool['en'], pool['d'], grid)
    excl = [math.sqrt(m) for m in range(1, 21) if jw.is_sum_of_three_squares(m)] + freqs_half
    return dict(names=[nm for nm, _ in jk.NAMED], freqs=np.array(freqs_named), amps=A_named,
                half_names=[nm for nm, _ in jk.HALF], half_freqs=np.array(freqs_half), half_amps=A_half,
                grid=grid, grid_amps=A_grid, floor=jk.noise_floor(grid, A_grid, excl), A1=float(A_named[0]))


def lp_interval(rows, cols, B, fixed_log=True):
    """The set-membership interval of eta: min and max of eta over {|S_i - model_i| <= B_i} (scipy HiGHS)."""
    try:
        from scipy.optimize import linprog
    except Exception as e:
        return dict(status=-1, message=str(e))
    rows = sorted(rows, key=lambda r: r['Rc'])
    X = np.array([r['X'][cols] for r in rows], dtype=float)
    y = np.array([r['S'] for r in rows], dtype=float)
    if fixed_log:
        y = y - C_LOG * np.array([r['X'][COL['lnR']] for r in rows], dtype=float)
    B = np.maximum(np.asarray(B, dtype=float), 1e-9 * np.abs(y))
    A_ub = np.vstack([X, -X])
    b_ub = np.concatenate([y + B, -(y - B)])
    out = dict(B=B)
    for sign, key in ((1.0, 'min'), (-1.0, 'max')):
        cvec = np.zeros(X.shape[1])
        cvec[0] = sign
        res = linprog(cvec, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * X.shape[1], method='highs')
        if res.status != 0:
            return dict(status=int(res.status), message=str(res.message), B=B)
        out[key] = float(res.x[0])
    out.update(status=0, half_width=0.5 * (out['max'] - out['min']), mid=0.5 * (out['max'] + out['min']))
    return out


# ----------------------------------------------------------------------------------------------
# the analysis
# ----------------------------------------------------------------------------------------------
def analyze(recs, radii, wide_upto, log=print, sphere=(0.9192, 0.0002), A4=None, allow_incomplete=False, plot=None, do_lp=True):
    if A4 is None:
        try:
            A4, zap, zerr = jb.a4_derived(check=True)
            log('finite-box constant: Z_AP(2) = %.15f (theta integral; the Ewald splits agree to 1e-10), A_4 = 128/15 Z_AP(2) = %.12f' % (zap, A4))
        except Exception as e:
            A4 = jb.A4_RECORD
            log('finite-box constant: derivation failed (%s); using the recorded A_4 = %.12f' % (e, A4))
    ext, box = jb.box_treatment(by_ball(recs), radii, wide_upto, A4)
    log('\nfinite box: S_inf = S(L) - A_4 (R_vol/L)^4 - lattice remainder; %d balls with 3 boxes, %d with 2, %d with 1'
        % (box['n3'], box['n2'], box['n1']))
    for g in sorted(k for k in box['per_group'] if k is not None):
        pg = box['per_group'][g]
        log('  R %2d  ladder balls %2d  b4 - A4 %+.3f (spread %.3f)  b6 %+7.2f (spread %.2f)  remainder at L/R %.1f: median %+.2e, max |.| %.2e nats'
            % (g, pg['count'], pg['e4_median'], pg['e4_spread'], pg['b6_median'], pg['b6_spread'], pg['ratio_max'],
               pg['delta_max_median'], pg['delta_max_abs']))
    log('  reference remainder: b4 - A4 = %+.3f +- %.3f, b6 = %+.2f +- %.2f (%s)'
        % (box['e4_ref'], box['spread4'], box['b6_ref'], box['spread6'], box['source']))
    for k in ('4R', '6R', '8R'):
        c = box['corrections'][k]
        log('  per-ball correction S_inf - S(L) at L = %s: continuum %+.4f, lattice %+.1e, bound %.1e nats' % (k, c['continuum'], c['lattice'], c['bound']))

    rows_all = window_rows(ext, radii, 2.0, 'hann')
    incomplete = [int(round(r['Rc'])) for r in rows_all if r['completeness'] < COMPLETE]
    rows = rows_all if allow_incomplete else [r for r in rows_all if r['completeness'] >= COMPLETE]
    radii_fit = [int(round(r['Rc'])) for r in rows]
    log('\nwindow data: Hann window of full width 2 at each primary radius (exact weights, exact regressors)')
    log('  R    balls    completeness  S_bar             S_bar/(4 pi <R^2>)  box bound (nats)  in fit')
    for r in rows_all:
        log('  %2.0f  %3d/%3d   %.4f        %15.6f   %.6f            %.1e           %s'
            % (r['Rc'], r['n_have'], r['n_total'], r['completeness'], r['S'], r['S'] / r['X'][0], r['box_unc'],
               'yes' if int(round(r['Rc'])) in radii_fit else 'no (incomplete)'))
    if incomplete:
        log('  incomplete windows at radii %s%s' % (incomplete, ' (admitted with the weight-passing rule: PREVIEW, not a result)' if allow_incomplete else ' (excluded)'))
    if len(rows) < 3:
        log('fewer than 3 radii in the fit; stop')
        return None
    rows_rect = window_rows(ext, radii_fit, 2.0, 'rect')
    rows_v1 = v1_rows(ext, radii_fit)
    rows_raw = window_rows(ext, radii_fit, 2.0, 'hann', value='S_raw')
    rows_cont = window_rows(ext, radii_fit, 2.0, 'hann', value='S_cont')
    rows_w3 = []
    for Rc, r in zip(radii_fit, rows):
        if Rc <= wide_upto:
            r3 = window_rows(ext, [Rc], 3.0, 'hann')
            if r3 and r3[0]['completeness'] >= COMPLETE:
                rows_w3.append(r3[0])
                continue
        rows_w3.append(r)

    c = COL
    specs = [('P', dict(rows=rows, cols=[c['4piR2'], c['one']]), 'primary: Hann W=2, c_log = -44/45, c1 = 0'),
             ('E1', dict(rows=rows, cols=[c['4piR2'], c['R'], c['one']]), 'c1 free (tests the complement-symmetry premise)'),
             ('E2', dict(rows=rows, cols=[c['4piR2'], c['one'], c['invR2']]), '+ c_-2/R^2'),
             ('E3', dict(rows=rows, cols=[c['4piR2'], c['one'], c['invR2'], c['lnR_invR2']]), '+ c_-2/R^2 + c_-2L ln R/R^2'),
             ('E4', dict(rows=rows, cols=[c['4piR2'], c['one']], drop_below=20), 'drop R < 20'),
             ('E5', dict(rows=rows, cols=[c['4piR2'], c['one']], drop_below=29), 'drop R < 29'),
             ('E6', dict(rows=rows, cols=[c['4piR2'], c['one']], Xkey='Xvol'), 'regressors from the volume radius per ball'),
             ('E7', dict(rows=rows, cols=[c['4piR2'], c['one']], weights=lambda r: 1.0 / r['Rc']), 'weights 1/R'),
             ('I1', dict(rows=rows, cols=[c['4piR2'], c['lnR'], c['one']], fixed_log=False), 'c_log free (information)'),
             ('I2', dict(rows=rows_rect, cols=[c['4piR2'], c['one']]), 'rectangular W=2 (information)'),
             ('I3', dict(rows=rows_v1, cols=[c['4piR2'], c['R'], c['one']]), 'version-1 estimator, c1 free (information)'),
             ('I3b', dict(rows=rows_v1, cols=[c['4piR2'], c['one']]), 'version-1 estimator, c1 = 0 (information)'),
             ('I4', dict(rows=rows_raw, cols=[c['4piR2'], c['one']]), 'no box correction (information)'),
             ('I5', dict(rows=rows_cont, cols=[c['4piR2'], c['one']]), 'box: A_4 term only, no lattice remainder (information)'),
             ('I6', dict(rows=rows, cols=[c['4piR2'], c['one'], c['invR']]), '+ c_-1/R (excluded by the same symmetry as c1; information)'),
             ('I7', dict(rows=rows_w3, cols=[c['4piR2'], c['one']]), 'Hann W=3 at R <= %d, W=2 above (information)' % wide_upto)]
    V = {}
    for name, kw, note in specs:
        rws = kw.pop('rows')
        res = fit(rws, **kw) if rws else None
        if res is not None:
            res['note'] = note
            V[name] = res
    if 'P' not in V:
        log('no primary fit')
        return None
    log('\nfits: S_bar = 4 pi eta <R^2> + c_log <ln R> + c0 (+ terms); c_log = -44/45 unless free; c1 = 0 unless stated')
    for name in ORDER:
        if name not in V:
            continue
        res = V[name]
        c1 = param(res, 'R')
        c0 = param(res, 'one')
        extra = ' '.join('%s %+.3g' % (nm, param(res, nm)) for nm in ('invR', 'invR2', 'lnR_invR2') if nm in res['names'])
        log('  %-4s eta %.6f +- %.6f  c_log %+.4f  c1 %s  c0 %+.3f  %s rms %.4f nats  radii %s  [%s]'
            % (name, res['eta'], res['se_eta'], res['c_log'], ('%+.4f' % c1) if c1 is not None else '   0   ',
               c0 if c0 is not None else float('nan'), extra, res['rms'], res['radii'], res['note']))
    P = V['P']
    log('  P residuals (nats): ' + ' '.join('%d:%+.4f' % (R, r) for R, r in zip(P['radii'], P['resid'])))
    log('  P eta row: se per nat of homogeneous per-radius noise %.3e; sum |w_i| %.3e per nat  (E1: %.3e, %.3e)'
        % (P['se_unit'], P['sum_abs_w'], V['E1']['se_unit'] if 'E1' in V else float('nan'), V['E1']['sum_abs_w'] if 'E1' in V else float('nan')))
    if 'E1' in V:
        log('  E1 c1 = %+.4f +- %.4f: %s' % (param(V['E1'], 'R'), V['E1']['se'][V['E1']['names'].index('R')],
                                             'consistent with 0 within 2 se' if abs(param(V['E1'], 'R')) <= 2 * V['E1']['se'][V['E1']['names'].index('R')] else 'MORE THAN 2 se FROM 0: the premise c1 = 0 is in question'))
    if 'I1' in V:
        log('  I1 c_log = %+.3f +- %.3f against the continuum -44/45 = %+.4f (not a test at this lever arm when the error exceeds 1)'
            % (V['I1']['c_log'], V['I1']['se_c_log'], C_LOG))

    # oscillation
    betaP = full_beta(P)
    pool = pooled_residuals(ext, radii, wide_upto, betaP)
    sums = jk.response_sums(2.0, 'hann')
    log('\noscillation: per-ball residuals about the primary model (every computed ball; width-3 tables at R <= %d)' % wide_upto)
    log('  R    balls  complete  rms (nats)  rms/sqrt(R)')
    for Rc in sorted(pool['per_window']):
        pw = pool['per_window'][Rc]
        log('  %2d   %4d   %-8s  %8.3f     %6.3f' % (Rc, pw['n'], 'yes' if pw['complete'] else 'no', pw['rms'], pw['rms_norm']))
    log('  pooled sigma_1 = rms(e_n / sqrt R_n) over %d balls (weights: the interval lengths) = %.3f (version 1: about 3.3)' % (pool['n'], pool['sigma1']))
    log('  Hann W=2 responses at the lattice frequencies: (sum_m hat w^2)^1/2 = %.5f (largest |hat w| %.4f at sqrt %d), (sum_m hat w\'^2)^1/2 = %.4f'
        % (math.sqrt(sums['sum_w2']), sums['max_w'], sums['argmax_m'], math.sqrt(sums['sum_dw2'])))
    pg = periodogram_report(pool)
    log('  periodogram of e_n / sqrt R over all balls (amplitudes in nats per sqrt R; resolution about %.3f): ' % (1.0 / (pool['R'].max() - pool['R'].min()))
        + '  '.join('%s %.3f' % (nm, a) for nm, a in zip(pg['names'], pg['amps'])) + '  | noise floor off the lines %.3f' % pg['floor'])
    log('  sublattice frequencies: ' + '  '.join('%s %.3f' % (nm, a) for nm, a in zip(pg['half_names'], pg['half_amps']))
        + '  (threshold 10 percent of A(1) = %.3f)' % (0.1 * pg['A1']))
    half_terms = [(nm, f, a) for nm, f, a in zip(pg['half_names'], pg['half_freqs'], pg['half_amps']) if a > 0.1 * pg['A1']]
    half_flag = bool(half_terms)
    if half_flag:
        log('  WARNING: sublattice line(s) %s above 10 percent of the f = 1 line: added to every bound; the W = 4 contingency of the specification applies at R >= 48'
            % [nm for nm, _, _ in half_terms])

    radii_wide = [R for R in radii_fit if R <= wide_upto]
    shifts = shift_test(ext, radii_wide, betaP)
    b_cs, b_half, b = {}, {}, {}
    for Rc in P['radii']:
        b_cs[Rc] = jk.cs_bound(pool['sigma1'], Rc, sums)
        b_half[Rc] = float(sum(a * math.sqrt(Rc) * abs(float(jw.window_ft(f, 2.0, 'hann'))) for _, f, a in half_terms))
        b[Rc] = b_cs[Rc] + b_half[Rc]
    ratios = {Rc: shifts[Rc]['half_range'] / b[Rc] for Rc in shifts if b[Rc] > 0}
    inflate = max([1.0] + list(ratios.values()))
    if inflate > 1.0 + 1e-9:
        log('  WARNING: the shift-test half-range exceeds the Cauchy-Schwarz bound at %s (largest ratio %.2f): every bound is inflated by that ratio'
            % ([Rc for Rc, q in ratios.items() if q > 1.0 + 1e-9], inflate))
        b = {Rc: v * inflate for Rc, v in b.items()}
    log('\nleakage bounds per radius (nats)')
    log('  R    CS bound  sublattice  bound b   shift half-range  shift max dev  P residual  covered')
    for Rc, r in zip(P['radii'], P['resid']):
        sh = shifts.get(Rc)
        log('  %2d   %7.3f   %7.3f    %7.3f   %14s   %12s   %+9.4f   %s'
            % (Rc, b_cs[Rc], b_half[Rc], b[Rc], ('%.4f' % sh['half_range']) if sh else '-', ('%.4f' % sh['max_dev']) if sh else '-',
               r, 'yes' if abs(r) <= b[Rc] + 1e-12 else 'NO'))

    # error statement
    w = P['row_eta']
    bvec = np.array([b[R] for R in P['radii']], dtype=float)
    env_window = float(np.sum(np.abs(w) * bvec))
    env_box = float(np.sum(np.abs(w) * P['box_unc']))
    env_members = {nm: V[nm]['eta'] - P['eta'] for nm in ENVELOPE if nm in V}
    env_model = float(max(abs(v) for v in env_members.values())) if env_members else 0.0
    env_argmax = max(env_members, key=lambda k: abs(env_members[k])) if env_members else None
    deta_dclog = float(-np.sum(w * P['lnR_bar']))
    dlog = abs(deta_dclog) * (44.0 / 45.0)
    two_se = 2.0 * P['se_eta']
    half = math.sqrt(two_se ** 2 + env_model ** 2 + env_window ** 2 + env_box ** 2 + dlog ** 2)
    eta = P['eta']
    log('\nerror statement (note section 2.4):')
    log('  2 se(P)                                     %.6f' % two_se)
    log('  model envelope max|eta_v - eta_P| over E    %.6f (from %s)' % (env_model, env_argmax))
    log('  window leakage sum_i |w_i| b_i              %.6f' % env_window)
    log('  box sum_i |w_i| beta_i                      %.6f' % env_box)
    log('  log sensitivity 44/45 |d eta/d c_log|       %.6f (d eta/d c_log = %.3e)' % (dlog, deta_dclog))
    lp = None
    if do_lp:
        Bvec = bvec + P['box_unc']
        lp = lp_interval(rows, [c['4piR2'], c['one']], Bvec)
        if lp.get('status') == 0:
            lp['influence_bound'] = float(np.sum(np.abs(w) * np.maximum(Bvec, 1e-9 * np.abs(P['S']))))
            lp['contains_P'] = bool(lp['min'] - 1e-12 <= eta <= lp['max'] + 1e-12)
            log('  set-membership interval (LP, B_i = b_i + beta_i; conditional on these measured bounds, not a certificate): [%.6f, %.6f], half-width %.6f (influence bound %.6f)'
                % (lp['min'], lp['max'], lp['half_width'], lp['influence_bound']))
        else:
            log('  set-membership interval: LP %s' % lp.get('message'))
    preview = allow_incomplete and bool(incomplete)
    log('RESULT%s: eta = %.5f, half-width %.5f; target 5e-4 %s%s'
        % (' (PREVIEW)' if preview else '', eta, half, 'MET' if half <= TARGET else 'NOT MET',
           '' if len(P['radii']) == len(radii) else ' (radii %s of %s)' % (P['radii'], list(radii))))
    log('        a/l_P = 2 sqrt(eta) = %.5f +- %.5f' % (2.0 * math.sqrt(eta), half / math.sqrt(eta)))
    if sphere is not None:
        diff = eta - sphere[0]
        comb = math.sqrt(half ** 2 + sphere[1] ** 2)
        log('        sphere average under test (argument; the web value is a hypothesis until the lane\'s planar route lands): %.5f +- %.5f; difference %+.5f against the combined half-width %.5f: %s'
            % (sphere[0], sphere[1], diff, comb, 'consistent' if abs(diff) <= comb else 'tension'))
    for nm, what in (('E1', 'c1 free'), ('I3', 'the version-1 estimator, c1 free'), ('I3b', 'the version-1 estimator, c1 = 0'),
                     ('I2', 'the rectangular window'), ('I1', 'the free-log fit'), ('I4', 'no box correction'), ('I5', 'A_4 term only')):
        if nm in V:
            log('        %s: eta %.6f (%+.6f)' % (what, V[nm]['eta'], V[nm]['eta'] - eta))
    log('        comparisons: planar [100] %.5f, [111] %.14f; J7 mesh 4 %.4f (labelled extrapolation 0.919)' % (PLANAR_100, PLANAR_111, J7_MESH4))
    res = dict(eta=eta, half_width=half, two_se=two_se, env_model=env_model, env_model_from=env_argmax, env_window=env_window, env_box=env_box,
               dlog=dlog, deta_dclog=deta_dclog, a_over_lP=2.0 * math.sqrt(eta), a_over_lP_half_width=half / math.sqrt(eta), target_met=bool(half <= TARGET),
               preview=preview, radii_fit=radii_fit, incomplete=incomplete, V=V, rows=rows, rows_all=rows_all, box=box, ext=ext,
               sigma1=pool['sigma1'], pool=pool, periodogram=pg, shifts=shifts, b=b, b_cs=b_cs, b_half=b_half, inflate=inflate,
               half_flag=half_flag, lp=lp, A4=A4)
    if plot:
        make_plots(res, plot)
    return res


def make_plots(res, prefix):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print('plots skipped (%s)' % e)
        return
    P = res['V']['P']
    R = np.array(P['radii'], dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.errorbar(R, P['resid'], yerr=[res['b'][r] for r in P['radii']], fmt='o', capsize=3, label='primary residual, bar = leakage bound')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('R')
    ax.set_ylabel('nats')
    ax.legend()
    fig.tight_layout()
    fig.savefig(prefix + '_residuals.png', dpi=130)
    plt.close(fig)
    if res['shifts']:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for Rc, sh in sorted(res['shifts'].items()):
            ax.plot(sh['shifts'], sh['eps'], label='R %d' % Rc)
        ax.set_xlabel('window shift s')
        ax.set_ylabel('eps(s) (nats)')
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(prefix + '_shift.png', dpi=130)
        plt.close(fig)
    pg = res['periodogram']
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(pg['grid'], pg['grid_amps'], lw=0.8)
    for f in pg['freqs']:
        ax.axvline(f, color='g', lw=0.5, alpha=0.6)
    for f in pg['half_freqs']:
        ax.axvline(f, color='r', lw=0.5, alpha=0.6, ls='--')
    ax.set_xlabel('frequency (cycles per lattice unit); green = lattice, red dashed = sublattice')
    ax.set_ylabel('amplitude (nats per sqrt R)')
    fig.tight_layout()
    fig.savefig(prefix + '_periodogram.png', dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# synthetic data and the selftest
# ----------------------------------------------------------------------------------------------
def synthetic_records(radii, wide_upto, eta0=0.919, c0=-5.0, A4=jb.A4_RECORD, e4=0.37, b6=-13.5, osc=0.0, mode='interval',
                      half_shift=0.0, seed=1, redo_v1=True):
    """Records of the production design (windows of width 3 at R <= wide_upto, 2 above, every ball at 8R; the version-1
    balls also at 4R and 6R for R <= 48 and at 4R (plus 8R if redo_v1) above) for a smooth model plus an optional shell
    oscillation.  mode 'weighted': the ball value is the Hann W=2 window-weighted interval mean of the smooth model (exactly
    representable by the estimator); 'interval': the unweighted interval mean (the version-2 construction; its sawtooth
    leaks about 0.7 R^-3/2 nats); 'midpoint': the smooth model at the interval midpoint.  The oscillation
    sqrt(R_mid) sum_m a_m cos(2 pi sqrt m R_mid + phi_m) over m <= 64 (amplitudes osc U(0.5, 1.5) m^-0.75) plus,
    with half_shift > 0, a sublattice line at sqrt 3/2 of amplitude half_shift a_1."""
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
    half_f, half_amp, half_phase = 0.5 * math.sqrt(3.0), float(half_shift) * float(amps[0]), rng.uniform(0.0, 2.0 * np.pi)
    recs, truth = {}, {}
    for Rc in radii:
        W = 3.0 if Rc <= wide_upto else 2.0
        tab2 = jw.window_table(Rc, 2.0, 'hann')
        w2 = {int(n): tab2['means'][i] for i, n in enumerate(tab2['n']) if tab2['omega'][i] > 0}
        v1 = set(jw.v1_balls(Rc))
        for n, R_lo, R_hi in jw.balls_in_window(Rc, W):
            if n in truth:
                continue
            ns = int(np.searchsorted(norms, n, side='right'))
            R_vol = (3.0 * ns / (4.0 * np.pi)) ** (1.0 / 3.0)
            plain = jw.plain_interval_means(R_lo, R_hi)
            Rm = 0.5 * (R_lo + R_hi)
            if mode == 'weighted' and n in w2:
                base = w2[n]
            elif mode == 'midpoint':
                base = jw.regressor_values(Rm)
            else:
                base = plain
            smooth = eta0 * base[COL['4piR2']] + C_LOG * base[COL['lnR']] + c0
            oscv = 0.0
            if osc:
                oscv = float(np.sqrt(Rm) * (np.sum(amps * np.cos(2.0 * np.pi * freqs * Rm + phases)) + half_amp * np.cos(2.0 * np.pi * half_f * Rm + half_phase)))
            S_inf = smooth + oscv
            truth[n] = dict(S_inf=S_inf, osc=oscv, smooth=smooth, Rc=Rc, R_lo=R_lo, R_hi=R_hi, R_mid=Rm, plain=plain)
            if n not in v1:
                boxes = [8 * Rc]
            elif Rc <= 48:
                boxes = [4 * Rc, 6 * Rc, 8 * Rc]
            else:
                boxes = [4 * Rc, 8 * Rc] if redo_v1 else [4 * Rc]
            for L in boxes:
                u = (R_vol / L) ** 4
                recs[(n, L)] = dict(n=n, L=L, S=S_inf + (A4 + e4) * u + b6 * u ** 1.5, n_sites=ns, R_vol=R_vol, file='synthetic', method='synthetic')
    par = dict(eta0=eta0, c0=c0, A4=A4, e4=e4, b6=b6, ms=ms, freqs=freqs, amps=amps, phases=phases, half_f=half_f, half_amp=half_amp, mode=mode)
    return recs, truth, par


def predicted_sawtooth(truth, radii, beta_true):
    """The leakage of the interval-mean construction: sum_n omega_n (<f>_{I_n} - <f>_{w,n}) . beta_true per radius."""
    eps = []
    for Rc in radii:
        tab = jw.window_table(Rc, 2.0, 'hann')
        s = 0.0
        for i, n in enumerate(tab['n']):
            if tab['omega'][i] <= 0:
                continue
            s += tab['omega'][i] * float((truth[int(n)]['plain'] - tab['means'][i]) @ beta_true)
        eps.append(s)
    return np.array(eps)


def selftest(radii=None, wide_upto=40, log=print, seed=1, osc=3.0):
    radii = list(radii or RADII_DEFAULT)
    quiet = lambda *a, **k: None  # noqa: E731
    A4, zap, _ = jb.a4_derived(check=True)
    out = {}

    recs, truth, par = synthetic_records(radii, wide_upto, A4=A4, mode='weighted', osc=0.0, seed=seed)
    log('=== selftest A1: exactly representable synthetic data (ball values = Hann W=2 weighted interval means of the smooth model) ===')
    rA1 = analyze(recs, radii, wide_upto, log=log, sphere=None, A4=A4)
    P, E1 = rA1['V']['P'], rA1['V']['E1']
    out['A1'] = dict(d_eta=P['eta'] - par['eta0'], d_c0=param(P, 'one') - par['c0'], d_eta_E1=E1['eta'] - par['eta0'], c1_E1=param(E1, 'R'),
                     d_e4=rA1['box']['e4_ref'] - par['e4'], d_b6=rA1['box']['b6_ref'] - par['b6'],
                     max_box_err=float(max(abs(rA1['ext'][n]['S_inf'] - truth[n]['S_inf']) for n in truth)), rms=P['rms'], lp=rA1['lp'])
    log('selftest A1: eta - eta0 %.2e, c0 - c0_true %.2e; E1: eta - eta0 %.2e, c1 %.2e; box: e4 %.1e, b6 %.1e, max |S_inf - truth| %.1e; P rms %.1e nats; LP [%.9f, %.9f]'
        % (out['A1']['d_eta'], out['A1']['d_c0'], out['A1']['d_eta_E1'], out['A1']['c1_E1'], out['A1']['d_e4'], out['A1']['d_b6'],
           out['A1']['max_box_err'], out['A1']['rms'], rA1['lp'].get('min', float('nan')), rA1['lp'].get('max', float('nan'))))

    recs, truth, par = synthetic_records(radii, wide_upto, A4=A4, mode='interval', osc=0.0, seed=seed)
    log('\n=== selftest A2: the step construction (unweighted interval means) and its predicted sawtooth leakage ===')
    rA2 = analyze(recs, radii, wide_upto, log=log, sphere=None, A4=A4)
    P = rA2['V']['P']
    beta_true = np.zeros(NREG)
    beta_true[COL['4piR2']], beta_true[COL['lnR']], beta_true[COL['one']] = par['eta0'], C_LOG, par['c0']
    eps_pred = predicted_sawtooth(truth, P['radii'], beta_true)
    eps_fit = np.array([r['S'] - r['X'] @ beta_true for r in sorted(rA2['rows'], key=lambda r: r['Rc'])])
    out['A2'] = dict(max_dev=float(np.max(np.abs(eps_fit - eps_pred))), eps_pred=eps_pred, radii=P['radii'], d_eta=P['eta'] - par['eta0'],
                     d_eta_pred=float(np.sum(P['row_eta'] * eps_pred)), d_c0=param(P, 'one') - par['c0'], rms=P['rms'], c1_E1=param(rA2['V']['E1'], 'R'))
    log('selftest A2: predicted sawtooth leakage (nats): ' + ' '.join('%d:%+.4f' % (R, e) for R, e in zip(P['radii'], eps_pred)))
    log('selftest A2: |pipeline deviation - prediction| max %.1e; eta - eta0 %.2e (predicted %.2e); c0 - c0_true %.2e; E1 c1 %.2e (the artifact of the construction, the version-2 failure)'
        % (out['A2']['max_dev'], out['A2']['d_eta'], out['A2']['d_eta_pred'], out['A2']['d_c0'], out['A2']['c1_E1']))

    recs, truth, par = synthetic_records(radii, wide_upto, A4=A4, mode='midpoint', osc=osc, seed=seed)
    single = np.array([truth[n]['osc'] / math.sqrt(truth[n]['R_mid']) for n in truth])
    single_rms = float(np.sqrt(np.mean(single ** 2)))
    log('\n=== selftest B: synthetic shell oscillation at the lattice frequencies (single-ball rms %.2f sqrt R nats) ===' % single_rms)
    rB = analyze(recs, radii, wide_upto, log=log, sphere=None, A4=A4)
    P = rB['V']['P']
    bounds = np.array([rB['b'][R] for R in P['radii']])
    pg = rB['periodogram']
    amps_true = dict(zip(par['ms'], par['amps']))
    out['B'] = dict(d_eta=P['eta'] - par['eta0'], half_width=rB['half_width'], env_window=rB['env_window'], env_model=rB['env_model'], env_box=rB['env_box'],
                    resid=P['resid'], bounds=bounds, covered=int(np.sum(np.abs(P['resid']) <= bounds + 1e-12)), n_radii=len(bounds),
                    sigma1=rB['sigma1'], sigma1_true=single_rms, inflate=rB['inflate'],
                    A_meas={nm: float(a) for nm, a in zip(pg['names'], pg['amps'])},
                    A_true={'1': amps_true[1], 'sqrt2': amps_true[2], 'sqrt3': amps_true[3], '2': amps_true[4], 'sqrt5': amps_true[5]},
                    half_ratio=float(np.max(pg['half_amps']) / pg['A1']), half_flag=rB['half_flag'], lp=rB['lp'],
                    d_eta_E1=rB['V']['E1']['eta'] - par['eta0'], d_eta_v1=(rB['V']['I3b']['eta'] - par['eta0']) if 'I3b' in rB['V'] else float('nan'),
                    d_eta_rect=(rB['V']['I2']['eta'] - par['eta0']) if 'I2' in rB['V'] else float('nan'))
    log('selftest B: eta - eta0: P %.2e (half-width %.2e, env_window %.2e, env_model %.2e), E1 %.2e, version-1 estimator %.2e, rectangular %.2e; '
        'bound covers the P residual at %d of %d radii; shift/CS ratio %.2f; sigma_1 %.3f (true %.3f); A(1) %.2f (true %.2f), A(sqrt2) %.2f (%.2f), A(sqrt3) %.2f (%.2f); sublattice/A(1) %.3f'
        % (out['B']['d_eta'], out['B']['half_width'], out['B']['env_window'], out['B']['env_model'], out['B']['d_eta_E1'], out['B']['d_eta_v1'], out['B']['d_eta_rect'],
           out['B']['covered'], out['B']['n_radii'], out['B']['inflate'], out['B']['sigma1'], single_rms, out['B']['A_meas']['1'], out['B']['A_true']['1'],
           out['B']['A_meas']['sqrt2'], out['B']['A_true']['sqrt2'], out['B']['A_meas']['sqrt3'], out['B']['A_true']['sqrt3'], out['B']['half_ratio']))

    recs2, truth2, par2 = synthetic_records(radii, wide_upto, A4=A4, mode='midpoint', osc=osc, half_shift=0.3, seed=seed)
    rB2 = analyze(recs2, radii, wide_upto, log=quiet, sphere=None, A4=A4)
    P2 = rB2['V']['P']
    bounds2 = np.array([rB2['b'][R] for R in P2['radii']])
    pg2 = rB2['periodogram']
    out['B2'] = dict(half_flag=rB2['half_flag'], half_ratio=float(pg2['half_amps'][pg2['half_names'].index('sqrt3/2')] / pg2['A1']),
                     d_eta=P2['eta'] - par2['eta0'], covered=int(np.sum(np.abs(P2['resid']) <= bounds2 + 1e-12)), n_radii=len(bounds2), half_width=rB2['half_width'])
    log('selftest B2 (sublattice line sqrt3/2 planted at 0.3 A(1)): detected %s with ratio %.3f; eta - eta0 %.2e; bound covers %d of %d; half-width %.2e'
        % (out['B2']['half_flag'], out['B2']['half_ratio'], out['B2']['d_eta'], out['B2']['covered'], out['B2']['n_radii'], out['B2']['half_width']))

    keep = set()
    for Rc in radii:
        W = 3.0 if Rc <= wide_upto else 2.0
        lev = jw.ball_levels(Rc, W)
        v1 = set(jw.v1_balls(Rc))
        for (n, L) in recs:
            if n in v1 or lev.get(n, 99) <= 5:
                keep.add((n, L))
    rC = analyze({k: v for k, v in recs.items() if k in keep}, radii, wide_upto, log=quiet, sphere=None, A4=A4, allow_incomplete=True)
    out['C'] = dict(d_eta=rC['V']['P']['eta'] - par['eta0'], completeness=[r['completeness'] for r in rC['rows']])
    log('selftest C: the level-5 sampled windows (completeness %.2f to %.2f, weight-passing rule): eta - eta0 %.2e'
        % (min(out['C']['completeness']), max(out['C']['completeness']), out['C']['d_eta']))
    out['D'] = dict(A1_lp=out['A1']['lp'], B_lp=out['B']['lp'], eta0=par['eta0'])
    log('selftest D: LP interval A1 [%.9f, %.9f] (half-width %.1e), B [%.6f, %.6f] (half-width %.2e, influence bound %.2e); truth %.6f'
        % (out['A1']['lp']['min'], out['A1']['lp']['max'], out['A1']['lp']['half_width'], out['B']['lp']['min'], out['B']['lp']['max'],
           out['B']['lp']['half_width'], out['B']['lp']['influence_bound'], par['eta0']))
    return out


def main():
    ap = argparse.ArgumentParser(description='J4 version-3 fit')
    ap.add_argument('--dirs', nargs='*', default=[], help='directories with records (version 1, version 2, prefix)')
    ap.add_argument('--radii', type=int, nargs='*', default=RADII_DEFAULT)
    ap.add_argument('--wide-upto', type=int, default=40)
    ap.add_argument('--out', default=None)
    ap.add_argument('--plot', default=None, help='prefix of the PNG files')
    ap.add_argument('--allow-incomplete', action='store_true', help='admit incomplete windows with the weight-passing rule (preview)')
    ap.add_argument('--no-site-check', action='store_true')
    ap.add_argument('--no-lp', action='store_true')
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
    res = analyze(recs, args.radii, args.wide_upto, sphere=(args.sphere_average, args.sphere_average_err),
                  allow_incomplete=args.allow_incomplete, plot=args.plot, do_lp=not args.no_lp)
    if res is None:
        sys.exit(1)
    if args.out:
        slim = {k: v for k, v in res.items() if k not in ('rows', 'rows_all', 'ext', 'pool')}
        slim['rows'] = [{k: v for k, v in r.items() if k != 'balls'} for r in res['rows_all']]
        slim['V'] = {nm: {k: v for k, v in r.items() if k not in ('row_eta', 'lnR_bar')} for nm, r in res['V'].items()}
        slim['pool'] = dict(R=res['pool']['R'], e=res['pool']['e'], d=res['pool']['d'], sigma1=res['pool']['sigma1'], per_window=res['pool']['per_window'])
        slim['ext_summary'] = [dict(n=x['n'], R_vol=x['R_vol'], nbox=x['nbox'], Ls=x['Ls'], S_inf=x['S_inf'], S_cont=x['S_cont'], box_correction=x['box_correction'],
                                    box_unc=x['box_unc'], b4=x['b4'], b6=x['b6'], delta_by_L=x['delta_by_L']) for x in res['ext'].values()]
        with open(args.out, 'w') as fh:
            json.dump(to_jsonable(slim), fh, indent=1)


if __name__ == '__main__':
    main()
