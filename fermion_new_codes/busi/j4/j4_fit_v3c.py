#!/usr/bin/env python3
"""
j4_fit_v3c.py -- the version-3c analysis of the J4 ladder (busi lane, task 1; lead's task B4; the design note
busi/notes/J4_v2_B4_v1.md, the corrections of busi/notes/J4_v2_B4_judgment_v1.md, the G6 adjudication of
busi/notes/J4_v2_B4_judgment_v2.md).

The estimator, the variants P, E1 to E7 and I1 to I7, the bounds, the error statement of design-note section 2.4 and the
LP interval are those of version 3b, unchanged: the same records give the same numbers (j4_tests_v3c.py checks this
against j4_fit_v3b.py on identical synthetic records).  Version 3c adds the diagnostics that adjudicate gate G6 (E1's
c1 = -0.0335 +- 0.0103 at gate C):

  1. the leakage predicted per window from the joint fit of the lattice lines (amplitudes AND phases, m <= 20, plus any
     sublattice line detected by G4) through the exact window weights, lambda = sum_n omega_n sqrt(R_n) l(R_n), with its
     standard error from the line-fit covariance; the unmodelled lines (m > 20) are bounded by the Cauchy-Schwarz tail
     bound with the post-fit residual;
  2. the predicted shift curves lambda(s) beside the measured eps(s) at every radius with width-3 data: the validation of
     the leakage model (rms of the deleaked curve about its mean against the rms of the measured curve about its mean);
  3. the window-level fit, on the deleaked shift residuals y(s) = S(s) - lambda(s) - X(s) beta_I8, of the smooth odd
     pattern r(R_c + s) plus the window responses of per-ball lines at 1/2, sqrt2/2 and sqrt3/2 (cos and sin): the
     smooth coefficient is c1 if a linear term is the cause; the three amplitudes are measured where the Hann window is
     most sensitive to them (responses 0.5, 0.217, 0.069), far below the per-ball threshold of G4; the standard errors
     are inflated by sqrt(n_shift / SHIFT_NEFF) for the correlation of neighbouring shifts; a line above
     WINDOW_LEVEL_SE inflated se enters every bound (amplitude + 2 se) and is subtracted with the lattice lines;
  4. the deleaked variants I8 (c1 = 0), I8c1 (c1 free; the G6 discriminator) and I8w3 (Hann W=3 at R <= wide_upto, c1
     free), the W=3 reading I7c1 (c1 free on the width-3 windows), the I7 residuals printed beside the P residuals;
  5. the reading of G6 by the rule of judgment note v2 section 5: 'aliasing' / 'odd term indicated' / 'inconclusive' /
     'model not validated' / 'not readable';
  6. the error statement of the replacement primary E1 (c1 free; envelope P, E1x2, E1x4 to E1x7) as information.
  The self-test adds the prediction of A1's shift curves from its representation mismatch (the A1 G3 WARNING is the
  construction's, reproduced to roundoff) and self-test B3 (a planted linear term c1 = -0.05 survives the deleaking).

Usage:
  j4_fit_v3c.py --dirs prod prod_v2 [--radii 12 ... 62] [--wide-upto 40] [--out fit.json] [--plot PREFIX]
                [--allow-incomplete] [--sphere-average 0.9192 --sphere-average-err 0.0002] [--no-site-check] [--w4-from 48]
  j4_fit_v3c.py --selftest [--osc 3.0] [--seed 1] [--out selftest.json]
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
import j4leak_v3b as jk   # noqa: E402

C_LOG = -44.0 / 45.0
RADII_DEFAULT = [12, 16, 20, 24, 29, 34, 40, 48, 56, 62]
COL = {name: i for i, name in enumerate(jw.REGRESSORS)}
NREG = len(jw.REGRESSORS)
ENVELOPE = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7']
ENVELOPE_E1 = ['P', 'E1x2', 'E1x4', 'E1x5', 'E1x6', 'E1x7']
ORDER = ['P', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'I1', 'I2', 'I3', 'I3b', 'I4', 'I5', 'I6', 'I7',
         'I7c1', 'I8', 'I8c1', 'I8w3', 'E1x2', 'E1x4', 'E1x5', 'E1x6', 'E1x7']
COMPLETE = 0.9999
TARGET = 5e-4
PLANAR_100 = 0.786583386553
PLANAR_111 = 0.92338180595796
J7_MESH4 = 0.9145
SUBLATTICE_SE = 3.0         # a sublattice line is detected above 10 percent of A(1) AND above this many standard errors
LATTICE_SIGNIFICANCE = 5.0  # the sublattice test applies only when A(1) exceeds this many standard errors
G5_GATE = 3e-4
MAX_COND = 20.0
MAX_CORR = 0.7
# version 3c: the G6 adjudication (judgment note v2 section 5); fixed before the R 56 and 62 data exist
WINDOW_LEVEL_NAMES = ['1/2', 'sqrt2/2', 'sqrt3/2']
WINDOW_LEVEL_FREQS = [0.5, 0.5 * math.sqrt(2.0), 0.5 * math.sqrt(3.0)]
WINDOW_LEVEL_SE = 3.0       # a window-level line is detected above this many INFLATED standard errors
SHIFT_NEFF = 4.0            # effective independent shifts per radius (span 1, correlation length about 0.25)
G6_SHIFT_RATIO = 0.5        # the line model is validated when rms(y - mean) <= this times rms(eps - mean) at every radius
G6_CLEAR_SE = 2.0           # deleaked |c1| <= this many se: the premise c1 = 0 stands (reading 'aliasing')
G6_INDICATED_SE = 3.0       # deleaked |c1| > this many se, same sign, and E1's own c1 > this many se: 'odd term indicated'


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


def window_rows(ext, radii, W, window, shift=0.0, value='S_inf', W_of=None):
    """One row per radius: the window average S with the exact weights (passed weights for incomplete windows), the
    exact regressors X_exact, the passed-weight regressors X_pass = sum_j om_j <f>_{w,j} (equal to X_exact when the
    window is complete), X = X_exact if complete else X_pass, the volume-radius regressors, the completeness, the
    passed weight and its sum of squares (the preview sampling uncertainty), and the per-ball data."""
    rows = []
    for Rc in radii:
        Wc = float(W if W_of is None else W_of(Rc))
        tab = jw.window_table(float(Rc) + shift, Wc, window)
        labels = np.asarray(tab['n'])
        vals = np.array([(ext[n][value] if (n in ext and ext[n].get(value) is not None) else np.nan) for n in labels], dtype=float)
        have = ~np.isnan(vals)
        if not have.any():
            continue
        omega = np.asarray(tab['omega'], dtype=float)
        R_mid = np.asarray(tab['R_mid'], dtype=float)
        means = np.asarray(tab['means'], dtype=float)
        om = complete_weights(omega, R_mid, have)
        S = float(np.sum(om[have] * vals[have]))
        completeness = float(omega[have].sum())
        complete = completeness >= COMPLETE
        X_exact = np.asarray(tab['exact'], dtype=float)
        X_pass = np.asarray(om[have] @ means[have], dtype=float)
        Rvol = np.array([ext[n]['R_vol'] for n in labels[have]], dtype=float)
        Xvol = np.asarray(om[have] @ jw.regressor_values(Rvol), dtype=float)
        box_unc = float(np.sum(om[have] * np.array([ext[n]['box_unc'] for n in labels[have]], dtype=float)))
        rows.append(dict(Rc=float(Rc), W=Wc, window=window, shift=float(shift), S=S,
                         X=(X_exact if complete else X_pass), X_exact=X_exact, X_pass=X_pass, Xvol=Xvol,
                         regressors=('exact' if complete else 'passed'),
                         completeness=completeness, n_have=int(have.sum()), n_total=int(len(labels)),
                         passed_weight=float(omega[~have].sum()), passed_w2=float(np.sum(omega[~have] ** 2)),
                         box_unc=box_unc,
                         balls=dict(n=labels[have], R_mid=R_mid[have], R_lo=np.asarray(tab['R_lo'], dtype=float)[have],
                                    R_hi=np.asarray(tab['R_hi'], dtype=float)[have], S_inf=vals[have], omega=om[have], means=means[have])))
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
                         X=jw.regressor_values(Rpts).mean(axis=0), Xvol=jw.regressor_values(Rvol).mean(axis=0), regressors='v1',
                         completeness=1.0, n_have=8, n_total=8, passed_weight=0.0, passed_w2=0.0,
                         box_unc=float(np.mean([ext[n]['box_unc'] for n in labels])),
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
    return float(res['beta'][res['names'].index(name)]) if (res is not None and name in res['names']) else None


def param_se(res, name):
    return float(res['se'][res['names'].index(name)]) if (res is not None and name in res['names']) else None


# ----------------------------------------------------------------------------------------------
# oscillation: pooled residuals, the predicted leakage, the shift test, the window-level fit, the periodogram, lines, LP
# ----------------------------------------------------------------------------------------------
def pooled_residuals(ext, radii, wide_upto, beta_full, W_of=None):
    """Per-ball residuals about the model over every computed ball of every radius (width-3 tables where they exist),
    their sqrt R-normalized pooled rms sigma_1 (weights: the interval lengths), and per-window summaries."""
    R_all, e_all, d_all, per = [], [], [], {}
    for Rc in radii:
        W = 3.0 if Rc <= wide_upto else (float(W_of(Rc)) if W_of is not None else 2.0)
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


def response_matrix(rows, freqs):
    """G[k, 2i] = sum_n omega_n sqrt(R_n) cos(2 pi f_i R_n), G[k, 2i+1] = ... sin ...: the response of window k (its
    balls and weights) to a per-ball line of unit amplitude times sqrt R at f_i.  The window leakage of a fitted line
    model with coefficients (a_i, b_i) is G @ coef; for a complete Hann window the rows approach sqrt(R_c) times the
    response hat w(f) at the window's phase."""
    freqs = np.asarray(freqs, dtype=float)
    G = np.zeros((len(rows), 2 * len(freqs)))
    for k, r in enumerate(rows):
        b = r['balls']
        R = np.asarray(b['R_mid'], dtype=float)
        om = np.asarray(b['omega'], dtype=float)
        sq = np.sqrt(R)
        for i, f in enumerate(freqs):
            th = 2.0 * np.pi * f * R
            G[k, 2 * i] = np.sum(om * sq * np.cos(th))
            G[k, 2 * i + 1] = np.sum(om * sq * np.sin(th))
    return G


def predicted_leakage(rows, lines):
    """Per row: (lambda, se) of the window leakage of the fitted lines with indices lines['leak_idx'] (the lattice lines
    m <= LATTICE_JOINT_MMAX plus any sublattice line detected by G4), lambda = G @ coef, se from the line-fit covariance."""
    idx = list(lines['leak_idx'])
    cols = [2 * i + k for i in idx for k in (0, 1)]
    G = response_matrix(rows, [float(lines['freqs_all'][i]) for i in idx])
    coef = np.asarray(lines['coef'], dtype=float)[cols]
    cov = np.asarray(lines['cov'], dtype=float)[np.ix_(cols, cols)]
    lam = G @ coef
    var = np.einsum('ij,jk,ik->i', G, cov, G)
    return lam, np.sqrt(np.maximum(var, 0.0))


def deleak_rows(rows, lam, lam_se):
    """Copies of the rows with S replaced by S - lambda (the regressors unchanged)."""
    out = []
    for r, l, s in zip(rows, lam, lam_se):
        d = dict(r)
        d.update(S=float(r['S'] - l), S_orig=float(r['S']), leak=float(l), leak_se=float(s))
        out.append(d)
    return out


def tail_sums(sums, mmax_fit):
    """The response sums restricted to the lattice frequencies beyond sqrt(mmax_fit): the lines the joint fit does not
    model (the analytic tail of j4leak_v3b.response_sums beyond its mmax is included)."""
    mask = np.asarray(sums['ms']) > int(mmax_fit)
    tail = sums.get('tail_w2', 0.0)
    tail = float(tail) if (tail is not None and tail == tail) else 0.0
    return dict(W=sums['W'], window=sums['window'], mmax_fit=int(mmax_fit),
                sum_w2=float(np.sum(np.asarray(sums['w'])[mask] ** 2)) + tail,
                sum_dw2=float(np.sum(np.asarray(sums['dw'])[mask] ** 2)))


def shift_test(ext, radii_wide, beta_full, shifts=None, lines=None, half_freqs=None):
    """At the radii with width-3 data: the fixed-log residual eps(s) of the primary W=2 window shifted by s in
    [-1/2, 1/2] (21 steps) about beta_full; the window averages S(s), the exact regressors X(s) and the responses
    G_half(s) to per-ball lines at half_freqs are kept; with a line model, the predicted leakage lambda(s) and its se."""
    if shifts is None:
        shifts = np.linspace(-0.5, 0.5, 21)
    if half_freqs is None:
        half_freqs = WINDOW_LEVEL_FREQS
    out = {}
    for Rc in radii_wide:
        srows, ok = [], True
        for s in shifts:
            rows = window_rows(ext, [Rc], 2.0, 'hann', shift=float(s))
            if not rows or rows[0]['completeness'] < COMPLETE:
                ok = False
                break
            srows.append(rows[0])
        if not ok:
            continue
        eps = np.array([r['S'] - r['X'] @ beta_full for r in srows], dtype=float)
        i0 = int(np.argmin(np.abs(shifts)))
        d = dict(shifts=np.array(shifts, dtype=float), eps=eps, eps0=float(eps[i0]), half_range=float(0.5 * (eps.max() - eps.min())),
                 max_dev=float(np.max(np.abs(eps - eps[i0]))), rms=float(np.std(eps)),
                 S=np.array([r['S'] for r in srows], dtype=float), X=np.array([r['X'] for r in srows], dtype=float),
                 G_half=response_matrix(srows, half_freqs))
        if lines is not None:
            lam, lam_se = predicted_leakage(srows, lines)
            d.update(lam=lam, lam_se=lam_se, lam0=float(lam[i0]))
        out[int(Rc)] = d
    return out


def shift_model_check(shifts, beta_ref, half_coef=None):
    """y(s) = S(s) - lambda(s) - G_half(s) half_coef - X(s) beta_ref, the deleaked shift residual about the deleaked
    primary; per radius its mean (what a smooth term leaves), its rms about the mean, the rms of the measured eps about
    its mean, and the ratio (the validation of the line model)."""
    out = {}
    for R, sh in shifts.items():
        if 'lam' not in sh:
            continue
        extra = (sh['G_half'] @ np.asarray(half_coef, dtype=float)) if half_coef is not None else 0.0
        y = sh['S'] - sh['lam'] - extra - sh['X'] @ beta_ref
        yc = y - y.mean()
        ec = sh['eps'] - sh['eps'].mean()
        rms_y = float(np.sqrt(np.mean(yc ** 2)))
        rms_e = float(np.sqrt(np.mean(ec ** 2)))
        out[int(R)] = dict(y=y, mean=float(y.mean()), rms=rms_y, eps_rms=rms_e, ratio=(rms_y / rms_e if rms_e > 0 else float('nan')),
                           y0=float(y[int(np.argmin(np.abs(sh['shifts'])))]))
    return out


def shift_limits(shifts, W=2.0):
    """Consistency limits on a sublattice line from the shift test: if the whole half-range h(R) were the line f, its
    amplitude A (nats per sqrt R) obeys h >= c_f |hat w(f)| A sqrt R with c_f the minimal half-range factor of cos over
    the shift span 1 (j4leak_v3b.shift_range_factor).  The minimum over the radii with shift data; not a measurement."""
    out = {}
    for nm, f in jk.HALF:
        resp = abs(float(jw.window_ft(f, W, 'hann')))
        c = jk.shift_range_factor(f, 1.0)
        lim = [sh['half_range'] / (c * resp * math.sqrt(Rc)) for Rc, sh in shifts.items() if c * resp > 0]
        out[nm] = float(min(lim)) if lim else float('nan')
    return out


def window_level_fit(shifts, chk, rows_fit):
    """Least squares on the deleaked shift residuals y(s) at every (R_c, s): the smooth odd pattern r(R_c + s) =
    <R>(s) - (4 pi <R^2>(s), 1) . gamma, gamma the projection of <R> on the primary design over the fit rows (a linear
    term c1 R in the data leaves c1 times this column), and the window responses G_half(s) of per-ball lines at
    WINDOW_LEVEL_FREQS (cos and sin).  Standard errors from the residual scatter, inflated by sqrt(n_shift / SHIFT_NEFF)
    for the correlation of neighbouring shifts; a line is detected above WINDOW_LEVEL_SE inflated se."""
    Rs = sorted(R for R in shifts if R in chk)
    if len(Rs) < 3:
        return None
    XP = np.array([r['X'][[COL['4piR2'], COL['one']]] for r in rows_fit], dtype=float)
    XR = np.array([r['X'][COL['R']] for r in rows_fit], dtype=float)
    gamma = np.linalg.lstsq(XP, XR, rcond=None)[0]
    ys, Ds, pat = [], [], {}
    for R in Rs:
        sh = shifts[R]
        X = sh['X']
        smooth = X[:, COL['R']] - X[:, [COL['4piR2'], COL['one']]] @ gamma
        pat[int(R)] = float(smooth[int(np.argmin(np.abs(sh['shifts'])))])
        ys.append(chk[R]['y'])
        Ds.append(np.column_stack([smooth, sh['G_half']]))
    y = np.concatenate(ys)
    D = np.vstack(Ds)
    N, p = D.shape
    if N <= p + 1:
        return None
    colnorm = np.linalg.norm(D, axis=0)
    cond = float(np.linalg.cond(D / colnorm))
    beta = np.linalg.lstsq(D, y, rcond=None)[0]
    resid = y - D @ beta
    s2 = float(np.sum(resid ** 2) / (N - p))
    cov = s2 * np.linalg.inv(D.T @ D)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    n_shift = len(shifts[Rs[0]]['shifts'])
    infl = math.sqrt(max(n_shift / SHIFT_NEFF, 1.0))
    nf = len(WINDOW_LEVEL_FREQS)
    amps, amp_se, phases = np.zeros(nf), np.zeros(nf), np.zeros(nf)
    for i in range(nf):
        a, b = beta[1 + 2 * i], beta[2 + 2 * i]
        va, vb, cab = cov[1 + 2 * i, 1 + 2 * i], cov[2 + 2 * i, 2 + 2 * i], cov[1 + 2 * i, 2 + 2 * i]
        A = math.hypot(a, b)
        amps[i] = A
        phases[i] = math.atan2(b, a)
        if A > 0.0:
            ca, sb = a / A, b / A
            amp_se[i] = math.sqrt(max(ca * ca * va + sb * sb * vb + 2.0 * ca * sb * cab, 0.0))
        else:
            amp_se[i] = math.sqrt(max(0.5 * (va + vb), 0.0))
    detected = [int(i) for i in range(nf) if amps[i] > WINDOW_LEVEL_SE * infl * amp_se[i]]
    return dict(radii=[int(R) for R in Rs], n=int(N), p=int(p), beta=beta, cov=cov, se=se, infl=float(infl), cond=cond,
                c1=float(beta[0]), c1_se=float(se[0] * infl), c1_se_naive=float(se[0]), amps=amps, amp_se=amp_se * infl,
                amp_se_naive=amp_se, phases=phases, detected=detected, names=list(WINDOW_LEVEL_NAMES),
                freqs=np.array(WINDOW_LEVEL_FREQS), resid_rms=float(np.sqrt(np.mean(resid ** 2))), pattern=pat, gamma=gamma)


def g6_reading(chk, V):
    """The rule of judgment note v2 section 5 applied to the deleaked c1-free fit I8c1 and the shift-curve validation."""
    out = dict(status='not readable', validated=False,
               text='no deleaked fit or no shift data with a line model: G6 is not adjudicated; P stands under the rule of section 2.4')
    if not chk or 'I8c1' not in V or 'E1' not in V:
        return out
    ratios = {int(R): chk[R]['ratio'] for R in chk}
    rmax = float(max(ratios.values()))
    validated = bool(rmax <= G6_SHIFT_RATIO)
    E1, D = V['E1'], V['I8c1']
    c1, se = param(E1, 'R'), param_se(E1, 'R')
    c1d, sed = param(D, 'R'), param_se(D, 'R')
    z = abs(c1) / se if se and se > 0 else float('nan')
    zd = abs(c1d) / sed if sed and sed > 0 else float('nan')
    out.update(ratios=ratios, max_ratio=rmax, validated=validated, c1=c1, se=se, z=z, c1_deleaked=c1d, se_deleaked=sed, z_deleaked=zd)
    if not validated:
        out.update(status='model not validated',
                   text='the line model does not reproduce the shift curves (largest ratio %.2f against %.2f): the deleaked fits are not read; P stands under the rule of section 2.4 and G6 is unresolved' % (rmax, G6_SHIFT_RATIO))
    elif zd <= G6_CLEAR_SE:
        out.update(status='aliasing',
                   text='the line model reproduces the shift curves (largest ratio %.2f) and the deleaked c1 = %+.4f +- %.4f is %.1f se from 0: E1\'s c1 is the aliasing of the shell oscillation over the design radii; the premise c1 = 0 stands, P is the primary and its envelope keeps E1\'s difference as section 2.4 prescribes'
                   % (rmax, c1d, sed, zd))
    elif zd > G6_INDICATED_SE and np.sign(c1d) == np.sign(c1) and z > G6_INDICATED_SE:
        out.update(status='odd term indicated',
                   text='the line model is validated (largest ratio %.2f) and the deleaked c1 = %+.4f +- %.4f is %.1f se from 0 with the sign of E1\'s c1 (%.1f se): a smooth odd term of the window mean is indicated; the premise c1 = 0 is not maintained and the contingency of judgment note v2 section 5 applies (E1 primary; its statement is printed below the error statement)'
                   % (rmax, c1d, sed, zd, z))
    else:
        out.update(status='inconclusive',
                   text='the line model is validated (largest ratio %.2f) but the deleaked c1 = %+.4f +- %.4f (%.1f se from 0) is between the clearing and the indicating thresholds, or E1\'s own c1 is below %.0f se (%.1f): inconclusive; P stands under the rule of section 2.4 and the note of the result says so'
                   % (rmax, c1d, sed, zd, G6_INDICATED_SE, z))
    return out


def error_terms(res, env_etas, b_by_R):
    """The terms of the error statement of design-note section 2.4 for the fit res with the envelope etas env_etas and
    the per-radius bounds b_by_R (already inflated where the shift test demands it)."""
    w = res['row_eta']
    bvec = np.array([b_by_R[R] for R in res['radii']], dtype=float)
    env_window = float(np.sum(np.abs(w) * bvec))
    env_box = float(np.sum(np.abs(w) * res['box_unc']))
    members = {nm: float(e - res['eta']) for nm, e in env_etas.items()}
    env_model = float(max(abs(v) for v in members.values())) if members else 0.0
    env_from = max(members, key=lambda k: abs(members[k])) if members else None
    deta = float(-np.sum(w * res['lnR_bar']))
    dlog = abs(deta) * (44.0 / 45.0)
    two_se = 2.0 * res['se_eta']
    half = math.sqrt(two_se ** 2 + env_model ** 2 + env_window ** 2 + env_box ** 2 + dlog ** 2)
    return dict(eta=res['eta'], two_se=two_se, env_model=env_model, env_model_from=env_from, env_members=members,
                env_window=env_window, env_box=env_box, dlog=dlog, deta_dclog=deta, half=half, sum_abs_w=res['sum_abs_w'])


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


def line_report(pool):
    """The joint fit of the lattice lines (m <= LATTICE_JOINT_MMAX) and the 5 sublattice frequencies over all balls, the
    detection by the rule (10 percent of A(1) and SUBLATTICE_SE se; A(1) itself above LATTICE_SIGNIFICANCE se; design
    well conditioned), the diagnostic of the single-frequency readings (leakage prediction, spectral window), and the
    coefficients and covariance that the leakage prediction of version 3c uses."""
    ms, f_lat = jk.lattice_frequencies(jk.LATTICE_JOINT_MMAX)
    f_half = [f for _, f in jk.HALF]
    names_half = [nm for nm, _ in jk.HALF]
    freqs = list(f_lat) + f_half
    J = jk.joint_lines(pool['R'], pool['en'], None, freqs)
    n_lat = len(f_lat)
    ms_list = [int(m) for m in ms]
    amps_lat, se_lat = J['amps'][:n_lat], J['se'][:n_lat]
    amps_half, se_half = J['amps'][n_lat:], J['se'][n_lat:]
    named = {}
    for nm, m in jk.NAMED_M:
        if m in ms_list:
            i = ms_list.index(m)
            named[nm] = (float(amps_lat[i]), float(se_lat[i]))
    i1 = ms_list.index(1)
    A1, se_A1 = float(amps_lat[i1]), float(se_lat[i1])
    well = bool(J['cond'] < MAX_COND and J['max_corr'] < MAX_CORR)
    applicable = bool(well and A1 > LATTICE_SIGNIFICANCE * se_A1)
    thr = 0.1 * A1
    detected, det_idx = [], []
    for k, (nm, f, a, s) in enumerate(zip(names_half, f_half, amps_half, se_half)):
        if applicable and a > thr and a > SUBLATTICE_SE * s:
            detected.append((nm, float(f), float(a), float(s)))
            det_idx.append(n_lat + k)
    lat_model = jk.line_model(pool['R'], freqs, J['coef'], range(n_lat))
    leak_pred = jk.periodogram(pool['R'], lat_model, pool['d'], f_half)
    specw = jk.spectral_window(pool['R'], pool['d'], [abs(f - 1.0) for f in f_half])
    ratio = amps_half / thr if thr > 0 else np.full(len(f_half), np.nan)
    signif = amps_half / np.where(se_half > 0, se_half, np.nan)
    return dict(ms=ms_list, f_lat=np.asarray(f_lat), amps_lat=amps_lat, se_lat=se_lat, f_half=np.asarray(f_half), half_names=names_half,
                amps_half=amps_half, se_half=se_half, upper_half=amps_half + 2.0 * se_half, A1=A1, se_A1=se_A1, named=named,
                resid_rms=J['resid_rms'], cond=J['cond'], max_corr=J['max_corr'], corr_pair=J['corr_pair'], dof=J['dof'], n=J['n'],
                well_conditioned=well, applicable=applicable, threshold=float(thr), detected=detected, leak_pred=leak_pred,
                spectral_window=specw, ratio=ratio, significance=signif,
                freqs_all=np.asarray(freqs, dtype=float), coef=np.asarray(J['coef'], dtype=float), cov=np.asarray(J['cov'], dtype=float),
                phases=np.asarray(J['phases'], dtype=float), n_lat=int(n_lat), leak_idx=list(range(n_lat)) + det_idx)


def polygon_eta_range(A, c, tol=1e-9):
    """Exact min and max of z[0] over the polygon {z in R^2: A z <= c} by vertex enumeration (the set is bounded here).
    Returns (zmin, zmax, number of feasible vertices) or None if no feasible vertex exists."""
    A = np.asarray(A, dtype=float)
    c = np.asarray(c, dtype=float)
    m = A.shape[0]
    zmin, zmax, nv = np.inf, -np.inf, 0
    for i in range(m):
        for j in range(i + 1, m):
            M = np.array([A[i], A[j]])
            det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
            if abs(det) <= 1e-14 * (np.abs(M).max() ** 2 + 1e-300):
                continue
            z = np.linalg.solve(M, np.array([c[i], c[j]]))
            if np.all(A @ z <= c + tol * (1.0 + np.abs(c))):
                nv += 1
                zmin = min(zmin, float(z[0]))
                zmax = max(zmax, float(z[0]))
    if nv == 0:
        return None
    return zmin, zmax, nv


def eta_range(X, y, B):
    """The set-membership interval of the first coefficient: min and max of beta[0] over {|X beta - y| <= B}.  Posed in
    least-squares-centred, column-scaled variables z = scale (beta - beta_LS): |Xs z - r| <= B.  For 2 parameters the
    exact vertex enumeration is the primary method; scipy's HiGHS runs beside it (and alone beyond 2 parameters).  The
    half-width is at most the influence bound sum_i |w_i| B_i, w the first row of the pseudo-inverse."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    B = np.asarray(B, dtype=float)
    n, p = X.shape
    pinv = np.linalg.pinv(X)
    beta_ls = pinv @ y
    r = y - X @ beta_ls
    scale = np.linalg.norm(X, axis=0)
    Xs = X / scale
    A_ub = np.vstack([Xs, -Xs])
    b_ub = np.concatenate([r + B, -(r - B)])
    out = dict(B=B, eta_ls=float(beta_ls[0]), influence_bound=float(np.sum(np.abs(pinv[0]) * B)), status=None, method=None)
    res_highs = None
    try:
        from scipy.optimize import linprog
        vals, ok = {}, True
        for sign, key in ((1.0, 'min'), (-1.0, 'max')):
            cvec = np.zeros(p)
            cvec[0] = sign
            res = linprog(cvec, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * p, method='highs')
            if res.status != 0:
                ok = False
                out['highs_message'] = str(res.message)
                break
            vals[key] = float(res.x[0])
        if ok:
            res_highs = (float(beta_ls[0] + vals['min'] / scale[0]), float(beta_ls[0] + vals['max'] / scale[0]))
    except Exception as e:
        out['highs_message'] = '%s: %s' % (type(e).__name__, e)
    res_poly = None
    if p == 2:
        pr = polygon_eta_range(A_ub, b_ub)
        if pr is not None:
            res_poly = (float(beta_ls[0] + pr[0] / scale[0]), float(beta_ls[0] + pr[1] / scale[0]))
            out['n_vertices'] = int(pr[2])
    if res_highs is not None and res_poly is not None:
        out['method_agreement'] = float(max(abs(res_highs[0] - res_poly[0]), abs(res_highs[1] - res_poly[1])))
    use = res_poly if res_poly is not None else res_highs
    if use is None:
        out.update(status=2, message='no feasible vertex (2 parameters) or LP failure: %s' % out.get('highs_message'))
        return out
    out.update(status=0, method=('vertex enumeration' if res_poly is not None else 'HiGHS (scaled)'), min=use[0], max=use[1],
               half_width=0.5 * (use[1] - use[0]), mid=0.5 * (use[0] + use[1]), highs=res_highs)
    return out


def lp_interval(rows, cols, B, fixed_log=True):
    """The set-membership interval of eta over the fit rows: min and max of eta over {|S_i - model_i| <= B_i}."""
    rows = sorted(rows, key=lambda r: r['Rc'])
    X = np.array([r['X'][cols] for r in rows], dtype=float)
    y = np.array([r['S'] for r in rows], dtype=float)
    if fixed_log:
        y = y - C_LOG * np.array([r['X'][COL['lnR']] for r in rows], dtype=float)
    B = np.maximum(np.asarray(B, dtype=float), 1e-9 * np.abs(y))
    return eta_range(X, y, B)


# ----------------------------------------------------------------------------------------------
# the analysis
# ----------------------------------------------------------------------------------------------
def analyze(recs, radii, wide_upto, log=print, sphere=(0.9192, 0.0002), A4=None, allow_incomplete=False, plot=None, do_lp=True, w4_from=None):
    def W_of(Rc):
        return 4.0 if (w4_from is not None and Rc >= w4_from) else 2.0

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
        log('  R %2d  ladder balls %2d  b4 - A4 %+.3f (spread %.3f)  b6 %+7.2f (spread %.2f)  S_3box - S_cont at L/R %.1f (= - lattice remainder): median %+.2e, max |.| %.2e nats'
            % (g, pg['count'], pg['e4_median'], pg['e4_spread'], pg['b6_median'], pg['b6_spread'], pg['ratio_max'],
               pg['delta_max_median'], pg['delta_max_abs']))
    log('  reference remainder: b4 - A4 = %+.3f +- %.3f, b6 = %+.2f +- %.2f (%s)'
        % (box['e4_ref'], box['spread4'], box['b6_ref'], box['spread6'], box['source']))
    for k in ('4R', '6R', '8R'):
        cc = box['corrections'][k]
        log('  per-ball correction S_inf - S(L) at L = %s: continuum %+.4f, lattice %+.1e, bound %.1e nats' % (k, cc['continuum'], cc['lattice'], cc['bound']))
    g5 = max([pg['delta_max_abs'] for g, pg in box['per_group'].items() if g is not None]) if any(g is not None for g in box['per_group']) else float('nan')
    log('  G5: largest |S_3box - S_cont| at the largest box over the ladder balls %.1e nats (gate %.0e): %s' % (g5, G5_GATE, 'pass' if g5 <= G5_GATE else 'FAIL'))

    if w4_from is not None:
        log('\n  W = 4 contingency: Hann width 4 at R >= %d, width 2 below' % w4_from)
    rows_all = window_rows(ext, radii, 2.0, 'hann', W_of=W_of)
    incomplete = [int(round(r['Rc'])) for r in rows_all if r['completeness'] < COMPLETE]
    rows = rows_all if allow_incomplete else [r for r in rows_all if r['completeness'] >= COMPLETE]
    radii_fit = [int(round(r['Rc'])) for r in rows]
    log('\nwindow data: Hann window of full width W at each primary radius (exact weights and regressors when complete; passed weights and consistent regressors when not)')
    log('  R    W   balls    completeness  passed weight  S_bar             S_bar/(4 pi <R^2>)  box bound (nats)  regressors  in fit')
    for r in rows_all:
        log('  %2.0f  %3.0f  %3d/%3d   %.4f        %.4f         %15.6f   %.6f            %.1e           %-7s     %s'
            % (r['Rc'], r['W'], r['n_have'], r['n_total'], r['completeness'], r['passed_weight'], r['S'], r['S'] / r['X'][0], r['box_unc'],
               r['regressors'], 'yes' if int(round(r['Rc'])) in radii_fit else 'no (incomplete)'))
    if incomplete:
        log('  incomplete windows at radii %s%s' % (incomplete, ' (admitted with the weight-passing rule: PREVIEW, not a result)' if allow_incomplete else ' (excluded)'))
    if len(rows) < 3:
        log('fewer than 3 radii in the fit; stop')
        return None
    rows_rect = window_rows(ext, radii_fit, 2.0, 'rect', W_of=W_of)
    rows_v1 = v1_rows(ext, radii_fit)
    rows_raw = window_rows(ext, radii_fit, 2.0, 'hann', value='S_raw', W_of=W_of)
    rows_cont = window_rows(ext, radii_fit, 2.0, 'hann', value='S_cont', W_of=W_of)
    rows_w3 = []
    for Rc, r in zip(radii_fit, rows):
        if Rc <= wide_upto:
            r3 = window_rows(ext, [Rc], 3.0, 'hann')
            if r3 and r3[0]['completeness'] >= COMPLETE:
                rows_w3.append(r3[0])
                continue
        rows_w3.append(r)
    rows_w3only = sorted([r for r in rows_w3 if r['W'] == 3.0], key=lambda r: r['Rc'])
    rows_sorted = sorted(rows, key=lambda r: r['Rc'])
    W_rows = {int(round(r['Rc'])): r['W'] for r in rows}
    sums_cache = {}

    def sums_for(W):
        if W not in sums_cache:
            sums_cache[W] = jk.response_sums(W, 'hann')
        return sums_cache[W]

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
             ('I7', dict(rows=rows_w3, cols=[c['4piR2'], c['one']]), 'Hann W=3 at R <= %d, W=2 above (information)' % wide_upto),
             ('I7c1', dict(rows=rows_w3only, cols=[c['4piR2'], c['R'], c['one']]), 'Hann W=3 at R <= %d only, c1 free (information: the W=3 reading of G6)' % wide_upto),
             ('E1x2', dict(rows=rows, cols=[c['4piR2'], c['R'], c['one'], c['invR2']]), 'c1 free + c_-2/R^2 (envelope of the replacement primary E1)'),
             ('E1x4', dict(rows=rows, cols=[c['4piR2'], c['R'], c['one']], drop_below=20), 'c1 free, drop R < 20 (envelope of E1)'),
             ('E1x5', dict(rows=rows, cols=[c['4piR2'], c['R'], c['one']], drop_below=29), 'c1 free, drop R < 29 (envelope of E1)'),
             ('E1x6', dict(rows=rows, cols=[c['4piR2'], c['R'], c['one']], Xkey='Xvol'), 'c1 free, volume-radius regressors (envelope of E1)'),
             ('E1x7', dict(rows=rows, cols=[c['4piR2'], c['R'], c['one']], weights=lambda r: 1.0 / r['Rc']), 'c1 free, weights 1/R (envelope of E1)')]
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
    P = V['P']

    # --- the oscillation measurement and the G6 diagnostics are computed here and printed below in the version-3b order ---
    betaP = full_beta(P)
    pool = pooled_residuals(ext, radii, wide_upto, betaP, W_of=W_of)
    pg = periodogram_report(pool)
    lines, lines_error = None, None
    try:
        lines = line_report(pool)
    except Exception as exc:
        lines_error = '%s: %s' % (type(exc).__name__, exc)
    lines_ok = bool(lines is not None and lines['applicable'])
    radii_wide = [R for R in radii_fit if R <= wide_upto]
    shifts = shift_test(ext, radii_wide, betaP, lines=(lines if lines_ok else None))
    half_coef = np.zeros(2 * len(WINDOW_LEVEL_FREQS))
    rows_dl, rows_w3_dl, wl, chk, g6_error = [], [], None, None, None
    lam_fit = lam_fit_se = lam_w3 = lam_w3_se = None
    G_fit = G_w3 = None
    if lines_ok:
        try:
            G_fit = response_matrix(rows_sorted, WINDOW_LEVEL_FREQS)
            G_w3 = response_matrix(rows_w3only, WINDOW_LEVEL_FREQS) if rows_w3only else np.zeros((0, len(half_coef)))
            lam_fit, lam_fit_se = predicted_leakage(rows_sorted, lines)
            if rows_w3only:
                lam_w3, lam_w3_se = predicted_leakage(rows_w3only, lines)
            else:
                lam_w3, lam_w3_se = np.zeros(0), np.zeros(0)

            def _deleak():
                rd = deleak_rows(rows_sorted, lam_fit + G_fit @ half_coef, lam_fit_se)
                r3 = deleak_rows(rows_w3only, lam_w3 + G_w3 @ half_coef, lam_w3_se) if rows_w3only else []
                return rd, r3

            rows_dl, rows_w3_dl = _deleak()
            I8p = fit(rows_dl, [c['4piR2'], c['one']])
            if I8p is not None and shifts and all('lam' in sh for sh in shifts.values()):
                chk = shift_model_check(shifts, full_beta(I8p), half_coef)
                wl = window_level_fit(shifts, chk, rows_sorted)
                if wl is not None and wl['detected']:
                    for i in wl['detected']:
                        half_coef[2 * i:2 * i + 2] = wl['beta'][1 + 2 * i:3 + 2 * i]
                        cv = wl['cov'][1 + 2 * i:3 + 2 * i, 1 + 2 * i:3 + 2 * i] * wl['infl'] ** 2
                        Gi = G_fit[:, 2 * i:2 * i + 2]
                        lam_fit_se = np.sqrt(lam_fit_se ** 2 + np.einsum('ij,jk,ik->i', Gi, cv, Gi))
                        if rows_w3only:
                            G3i = G_w3[:, 2 * i:2 * i + 2]
                            lam_w3_se = np.sqrt(lam_w3_se ** 2 + np.einsum('ij,jk,ik->i', G3i, cv, G3i))
                    rows_dl, rows_w3_dl = _deleak()
                    I8p = fit(rows_dl, [c['4piR2'], c['one']])
                    if I8p is not None:
                        chk = shift_model_check(shifts, full_beta(I8p), half_coef)
        except Exception as exc:
            g6_error = '%s: %s' % (type(exc).__name__, exc)
            rows_dl, rows_w3_dl, wl, chk = [], [], None, None
    if rows_dl:
        specs_dl = [('I8', dict(rows=rows_dl, cols=[c['4piR2'], c['one']]), 'deleaked primary: S_bar minus the predicted leakage, c1 = 0 (information)'),
                    ('I8c1', dict(rows=rows_dl, cols=[c['4piR2'], c['R'], c['one']]), 'deleaked, c1 free (information: the G6 discriminator)'),
                    ('I8w3', dict(rows=rows_w3_dl, cols=[c['4piR2'], c['R'], c['one']]), 'deleaked Hann W=3 at R <= %d, c1 free (information)' % wide_upto)]
        for name, kw, note in specs_dl:
            rws = kw.pop('rows')
            res = fit(rws, **kw) if rws else None
            if res is not None:
                res['note'] = note
                V[name] = res
    g6 = g6_reading(chk, V)

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
    wP = P['row_eta']
    deta_dclog = float(-np.sum(wP * P['lnR_bar']))
    log('  P residuals (nats): ' + ' '.join('%d:%+.4f' % (R, r) for R, r in zip(P['radii'], P['resid'])))
    log('  P eta row: se per nat of homogeneous per-radius noise %.3e; sum |w_i| %.3e per nat  (E1: %.3e, %.3e)'
        % (P['se_unit'], P['sum_abs_w'], V['E1']['se_unit'] if 'E1' in V else float('nan'), V['E1']['sum_abs_w'] if 'E1' in V else float('nan')))
    if 'E1' in V:
        se_c1 = param_se(V['E1'], 'R')
        log('  E1 c1 = %+.4f +- %.4f: %s (G6)' % (param(V['E1'], 'R'), se_c1,
                                                 'consistent with 0 within 2 se' if abs(param(V['E1'], 'R')) <= 2 * se_c1 else 'MORE THAN 2 se FROM 0: the premise c1 = 0 is in question; read the G6 adjudication below'))
    if 'I1' in V:
        I1 = V['I1']
        dev = abs(I1['c_log'] - C_LOG) / I1['se_c_log'] if I1['se_c_log'] > 0 else float('nan')
        log('  I1 c_log = %+.3f +- %.3f: %.1f se from the continuum -44/45 = %+.4f (information, G7 reads 3 se; eta moves by %.1e per unit of c_log)'
            % (I1['c_log'], I1['se_c_log'], dev, C_LOG, abs(deta_dclog)))

    # oscillation
    sums2 = sums_for(2.0)
    log('\noscillation: per-ball residuals about the primary model (every computed ball; width-3 tables at R <= %d)' % wide_upto)
    log('  R    balls  complete  rms (nats)  rms/sqrt(R)')
    for Rc in sorted(pool['per_window']):
        pw = pool['per_window'][Rc]
        log('  %2d   %4d   %-8s  %8.3f     %6.3f' % (Rc, pw['n'], 'yes' if pw['complete'] else 'no', pw['rms'], pw['rms_norm']))
    log('  pooled sigma_1 = rms(e_n / sqrt R_n) over %d balls (weights: the interval lengths) = %.3f (version 1: about 3.3)' % (pool['n'], pool['sigma1']))
    log('  Hann W=2 responses at the lattice frequencies: (sum_m hat w^2)^1/2 = %.5f (largest |hat w| %.4f at sqrt %d), (sum_m hat w\'^2)^1/2 = %.4f'
        % (math.sqrt(sums2['sum_w2']), sums2['max_w'], sums2['argmax_m'], math.sqrt(sums2['sum_dw2'])))
    if w4_from is not None:
        s4 = sums_for(4.0)
        log('  Hann W=4 responses at the lattice frequencies: (sum_m hat w^2)^1/2 = %.5f, (sum_m hat w\'^2)^1/2 = %.4f; at the sublattice frequencies %s'
            % (math.sqrt(s4['sum_w2']), math.sqrt(s4['sum_dw2']), ', '.join('%s %.4f' % (nm, abs(float(jw.window_ft(f, 4.0, 'hann')))) for nm, f in jk.HALF)))
    log('  single-frequency periodogram of e_n / sqrt R over all balls (interval weights; baseline %.0f lattice units; it reads every line times the spectral window of the clustered sampling, so these numbers carry the cross-leakage of the lines): '
        % (pool['R'].max() - pool['R'].min())
        + '  '.join('%s %.3f' % (nm, a) for nm, a in zip(pg['names'], pg['amps'])) + '  | noise floor off the lines %.3f' % pg['floor'])
    half_terms = []
    if lines is None:
        log('  joint line fit not possible (%s)' % lines_error)
    else:
        log('  joint least-squares fit of %d lattice lines (m <= %d) and the 5 sublattice frequencies over %d balls (uniform weights): residual rms %.3f nats per sqrt R, largest |correlation| between columns %.2f (f %.3f and %.3f), condition number %.1f%s'
            % (len(lines['ms']), jk.LATTICE_JOINT_MMAX, lines['n'], lines['resid_rms'], lines['max_corr'], lines['corr_pair'][0], lines['corr_pair'][1], lines['cond'],
               '' if lines['well_conditioned'] else '  WARNING: ill conditioned (gates %.1f and %.0f): the joint amplitudes are not read' % (MAX_CORR, MAX_COND)))
        log('  joint amplitudes (nats per sqrt R, +- se): ' + '  '.join('%s %.3f+-%.3f' % (nm, a, s) for nm, (a, s) in lines['named'].items()))
        log('  sublattice (joint): ' + '  '.join('%s %.3f+-%.3f' % (nm, a, s) for nm, a, s in zip(lines['half_names'], lines['amps_half'], lines['se_half']))
            + '  | threshold 10 percent of A(1) = %.3f; detection needs the threshold and %.0f se%s'
            % (lines['threshold'], SUBLATTICE_SE, '' if lines['applicable'] else ('; NOT APPLICABLE: A(1) = %.3f against %.0f se = %.3f, or ill conditioned (no lattice oscillation measured)' % (lines['A1'], LATTICE_SIGNIFICANCE, LATTICE_SIGNIFICANCE * lines['se_A1']))))
        log('  single-frequency readings at the sublattice frequencies: data ' + ' '.join('%.3f' % a for a in pg['half_amps'])
            + ' | predicted from the fitted lattice lines alone ' + ' '.join('%.3f' % a for a in lines['leak_pred'])
            + ' | spectral window |W| of the sampling at the offsets from f = 1: ' + ' '.join('%.3f' % a for a in lines['spectral_window']))
        half_terms = lines['detected']
    half_flag = bool(half_terms)
    if half_flag:
        log('  WARNING (G4): sublattice line(s) detected: %s (name, f, amplitude, se): added to every bound with amplitude + 2 se and included in the leakage model; for 1/2 or sqrt2/2 the W = 4 contingency applies at R >= 48 (judgment note v1 section 3.4), for sqrt3/2, sqrt5/2, sqrt11/2 the bound alone'
            % ['%s f %.3f A %.3f se %.3f' % t for t in half_terms])
    elif lines is not None and lines['applicable']:
        try:
            imax = int(np.nanargmax(lines['ratio']))
            smax = int(np.nanargmax(lines['significance']))
            log('  G4: no sublattice line detected (largest amplitude/threshold %.2f at %s; largest significance %.1f se at %s)'
                % (lines['ratio'][imax], lines['half_names'][imax], lines['significance'][smax], lines['half_names'][smax]))
        except Exception:
            log('  G4: no sublattice line detected')
    else:
        log('  G4: not applicable on these data (see above)')

    limits = shift_limits(shifts) if shifts else {nm: float('nan') for nm, _ in jk.HALF}
    if shifts:
        log('  shift-test consistency limits on a sublattice line (if the whole half-range were that line; nats per sqrt R): '
            + '  '.join('%s <= %.3f' % (nm, limits[nm]) for nm, _ in jk.HALF))
    # window-level lines detected in version 3c enter every bound as amplitude + 2 (inflated) se times the window response
    wl_terms = []
    if wl is not None and wl['detected']:
        for i in wl['detected']:
            wl_terms.append((WINDOW_LEVEL_NAMES[i], float(WINDOW_LEVEL_FREQS[i]), float(wl['amps'][i]), float(wl['amp_se'][i])))
    b_cs, b_half, b_allow, b = {}, {}, {}, {}
    for Rc in P['radii']:
        W = W_rows[Rc]
        b_cs[Rc] = jk.cs_bound(pool['sigma1'], Rc, sums_for(W))
        b_half[Rc] = float(sum((a + 2.0 * s) * math.sqrt(Rc) * abs(float(jw.window_ft(f, W, 'hann'))) for _, f, a, s in half_terms + wl_terms))
        if lines is not None and lines['applicable']:
            b_allow[Rc] = float(sum((a + 2.0 * s) * math.sqrt(Rc) * abs(float(jw.window_ft(f, W, 'hann')))
                                    for f, a, s in zip(lines['f_half'], lines['amps_half'], lines['se_half'])))
        else:
            b_allow[Rc] = float('nan')
        b[Rc] = b_cs[Rc] + b_half[Rc]
    ref = {Rc: jk.cs_bound(pool['sigma1'], Rc, sums2) + float(sum((a + 2.0 * s) * math.sqrt(Rc) * abs(float(jw.window_ft(f, 2.0, 'hann'))) for _, f, a, s in half_terms + wl_terms))
           for Rc in shifts}
    ratios = {Rc: shifts[Rc]['half_range'] / ref[Rc] for Rc in shifts if ref[Rc] > 0}
    inflate = max([1.0] + list(ratios.values()))
    if inflate > 1.0 + 1e-9:
        log('  WARNING (G3): the shift-test half-range exceeds the Cauchy-Schwarz bound at %s (largest ratio %.2f): every bound is inflated by that ratio'
            % ([Rc for Rc, q in ratios.items() if q > 1.0 + 1e-9], inflate))
        b = {Rc: v * inflate for Rc, v in b.items()}
    elif shifts:
        log('  G3: shift half-range within the Cauchy-Schwarz bound at %s (largest ratio %.2f)' % (sorted(ratios), max(ratios.values()) if ratios else float('nan')))
    log('\nleakage bounds per radius (nats)')
    log('  R    W   CS bound  sublattice  bound b   allowance (info)  shift half-range  shift max dev  P residual  covered')
    for Rc, rr in zip(P['radii'], P['resid']):
        sh = shifts.get(Rc)
        log('  %2d  %3.0f  %7.3f   %7.3f    %7.3f   %7.3f           %14s   %12s   %+9.4f   %s'
            % (Rc, W_rows[Rc], b_cs[Rc], b_half[Rc], b[Rc], b_allow[Rc], ('%.4f' % sh['half_range']) if sh else '-', ('%.4f' % sh['max_dev']) if sh else '-',
               rr, 'yes' if abs(rr) <= b[Rc] + 1e-12 else 'NO'))

    # --- G6 adjudication (version 3c) ---
    log('\nG6 adjudication (information; the rule of busi/notes/J4_v2_B4_judgment_v2.md section 5): is E1\'s c1 the aliasing of the shell oscillation over the design radii, or a smooth odd term of the window mean?')
    leak_table = []
    if not rows_dl:
        log('  not readable: %s' % (g6_error or ('no validated line model (A(1) below %.0f se, or ill conditioned): the deleaked variants are not formed' % LATTICE_SIGNIFICANCE)))
    else:
        try:
            lk = sorted(((lines['named'][nm][0] * abs(float(jw.window_ft(f, 2.0, 'hann'))), nm) for nm, f in jk.NAMED if nm in lines['named']), reverse=True)
            log('  leaked amplitude per lattice line through the Hann W=2 window, A |hat w| (nats per sqrt R): ' + '  '.join('%s %.4f' % (nm, a) for a, nm in lk[:6])
                + '; the leakage model holds %d lines (%d lattice, %d sublattice detected by G4) with their phases' % (len(lines['leak_idx']), lines['n_lat'], len(lines['leak_idx']) - lines['n_lat']))
            tail2 = tail_sums(sums2, jk.LATTICE_JOINT_MMAX)
            R0, R1 = P['radii'][0], P['radii'][-1]
            log('  unmodelled lattice lines (m > %d): residual %.3f per sqrt R after the line fit; tail response sums (sum hat w^2)^1/2 = %.2e, (sum hat w\'^2)^1/2 = %.3f: Cauchy-Schwarz tail bound %.3f nats at R %d, %.3f at R %d'
                % (jk.LATTICE_JOINT_MMAX, lines['resid_rms'], math.sqrt(tail2['sum_w2']), math.sqrt(tail2['sum_dw2']),
                   jk.cs_bound(lines['resid_rms'], R0, tail2), R0, jk.cs_bound(lines['resid_rms'], R1, tail2), R1))
            lam_tot = lam_fit + G_fit @ half_coef
            lam3_tot = (lam_w3 + G_w3 @ half_coef) if rows_w3only else np.zeros(0)
            c1E1 = param(V['E1'], 'R')
            XP = np.array([r['X'][[c['4piR2'], c['one']]] for r in rows_sorted], dtype=float)
            XR = np.array([r['X'][c['R']] for r in rows_sorted], dtype=float)
            rperp = XR - XP @ np.linalg.lstsq(XP, XR, rcond=None)[0]

            def resid_by(res):
                return dict(zip(res['radii'], res['resid'])) if res is not None else {}
            rI8, rI7, rI7c1, rI8w3 = resid_by(V.get('I8')), resid_by(V.get('I7')), resid_by(V.get('I7c1')), resid_by(V.get('I8w3'))
            lam3_by = {int(round(r['Rc'])): (float(l), float(s)) for r, l, s in zip(rows_w3only, lam3_tot, lam_w3_se)}
            log('  per radius (nats).  W=2: the P residual, the predicted leakage lambda +- se, P - lambda, the residual of the deleaked fit I8 and its tail bound (tail + 2 se of lambda) with the coverage; the pattern c1 r(R) that E1 attributes to a linear term; W=3 at R <= %d (Hann W=3 passes sqrt2 with 0.003 and sqrt3 with 0.0014, 2.8 and 6 times less than W=2, a smooth term unchanged): the I7 residual, lambda at W=3, I7 - lambda, the residuals of I7c1 and I8w3 (c1 free at W=3, original and deleaked)' % wide_upto)
            log('  R    P resid   lambda   +-se    P-lambda  I8 resid  tail bd  cov | E1 c1 r(R) | I7 resid  lambda3   I7-lam3   I7c1 resid  I8w3 resid')
            for i, (Rc, rr) in enumerate(zip(P['radii'], P['resid'])):
                tb = jk.cs_bound(lines['resid_rms'], Rc, tail_sums(sums_for(W_rows[Rc]), jk.LATTICE_JOINT_MMAX)) + 2.0 * float(lam_fit_se[i])
                i8 = rI8.get(Rc, float('nan'))
                entry = dict(R=int(Rc), P_resid=float(rr), lam=float(lam_tot[i]), lam_se=float(lam_fit_se[i]), I8_resid=float(i8), tail_bound=float(tb),
                             pattern=float(c1E1 * rperp[i]) if c1E1 is not None else float('nan'))
                w3 = ''
                if Rc in rI7 and Rc in lam3_by and Rc <= wide_upto:
                    l3, s3 = lam3_by[Rc]
                    w3 = '%+8.4f  %+8.4f  %+8.4f   %s   %s' % (rI7[Rc], l3, rI7[Rc] - l3,
                                                            ('%+8.4f' % rI7c1[Rc]) if Rc in rI7c1 else '    -   ',
                                                            ('%+8.4f' % rI8w3[Rc]) if Rc in rI8w3 else '    -   ')
                    entry.update(I7_resid=float(rI7[Rc]), lam3=l3, lam3_se=s3, I7c1_resid=rI7c1.get(Rc), I8w3_resid=rI8w3.get(Rc))
                leak_table.append(entry)
                log('  %2d  %+8.4f  %+8.4f  %6.4f  %+8.4f  %+8.4f  %7.4f  %-3s | %+9.4f  | %s'
                    % (Rc, rr, lam_tot[i], lam_fit_se[i], rr - lam_tot[i], i8, tb, ('yes' if abs(i8) <= tb + 1e-12 else 'NO') if i8 == i8 else '-',
                       entry['pattern'], w3))
            if chk:
                Rs = sorted(chk)
                log('  shift curves at R %s: rms about the mean of the deleaked curve y(s) = S(s) - lambda(s) - X(s) beta_I8 over that of the measured eps(s): %s; largest %.2f against %.2f: the line model is %s'
                    % (Rs, ' '.join('%.2f' % chk[R]['ratio'] for R in Rs), max(chk[R]['ratio'] for R in Rs), G6_SHIFT_RATIO,
                       'VALIDATED' if max(chk[R]['ratio'] for R in Rs) <= G6_SHIFT_RATIO else 'NOT validated'))
                log('  mean over s of y(s) per radius (the deleaked residual of the s-averaged window, which a smooth term leaves as c1 r(R) and the lattice leakage does not): %s | E1\'s c1 r(R) there: %s'
                    % (' '.join('%d:%+.4f' % (R, chk[R]['mean']) for R in Rs),
                       ' '.join('%d:%+.4f' % (R, c1E1 * rperp[P['radii'].index(R)]) for R in Rs if R in P['radii'])))
            if wl is not None:
                det = ['%s %.4f+-%.4f' % (wl['names'][i], wl['amps'][i], wl['amp_se'][i]) for i in wl['detected']]
                log('  window-level fit on y(s) (%d points at R %s; %d columns: the smooth odd pattern r(R_c + s) and the window responses of per-ball lines at %s, cos and sin; condition number %.1f): smooth coefficient %+.4f +- %.4f (E1: %+.4f +- %.4f); lines %s nats per sqrt R; se inflated by %.2f (%d shifts per radius, %.0f effective); residual rms %.4f; %s'
                    % (wl['n'], wl['radii'], wl['p'], ', '.join(wl['names']), wl['cond'], wl['c1'], wl['c1_se'], c1E1, param_se(V['E1'], 'R'),
                       '  '.join('%s %.4f+-%.4f' % (nm, a, s) for nm, a, s in zip(wl['names'], wl['amps'], wl['amp_se'])), wl['infl'],
                       len(shifts[wl['radii'][0]]['shifts']), SHIFT_NEFF, wl['resid_rms'],
                       ('DETECTED above %.0f inflated se: %s, subtracted with the lattice lines and added to every bound' % (WINDOW_LEVEL_SE, det)) if det else ('no line above %.0f inflated se' % WINDOW_LEVEL_SE)))
            I8, I8c1, I7c1, I8w3 = V.get('I8'), V.get('I8c1'), V.get('I7c1'), V.get('I8w3')

            def c1txt(res):
                return ('c1 = %+.4f +- %.4f (%.1f se from 0)' % (param(res, 'R'), param_se(res, 'R'), abs(param(res, 'R')) / param_se(res, 'R') if param_se(res, 'R') > 0 else float('nan'))) if res is not None else 'not fitted'
            log('  deleaked fits: I8 (c1 = 0) eta %.6f (%+.1e from P), rms %.4f; I8c1 %s, eta %.6f; W=3: I7c1 %s; I8w3 %s'
                % (I8['eta'] if I8 else float('nan'), (I8['eta'] - P['eta']) if I8 else float('nan'), I8['rms'] if I8 else float('nan'),
                   c1txt(I8c1), I8c1['eta'] if I8c1 else float('nan'), c1txt(I7c1), c1txt(I8w3)))
        except Exception as exc:
            log('  the G6 table failed (%s)' % exc)
    log('  G6 reading: %s' % g6['text'])

    # error statement
    termsP = error_terms(P, {nm: V[nm]['eta'] for nm in ENVELOPE if nm in V}, b)
    env_window, env_box, env_model, env_argmax = termsP['env_window'], termsP['env_box'], termsP['env_model'], termsP['env_model_from']
    dlog, two_se, half, eta = termsP['dlog'], termsP['two_se'], termsP['half'], P['eta']
    w = wP
    bvec = np.array([b[R] for R in P['radii']], dtype=float)
    log('\nerror statement (design note section 2.4):')
    log('  2 se(P)                                     %.6f' % two_se)
    log('  model envelope max|eta_v - eta_P| over E    %.6f (from %s)' % (env_model, env_argmax))
    log('  window leakage sum_i |w_i| b_i              %.6f' % env_window)
    log('  box sum_i |w_i| beta_i                      %.6f' % env_box)
    log('  log sensitivity 44/45 |d eta/d c_log|       %.6f (d eta/d c_log = %.3e)' % (dlog, deta_dclog))
    termsE1 = None
    if 'E1' in V:
        try:
            termsE1 = error_terms(V['E1'], {nm: V[nm]['eta'] for nm in ENVELOPE_E1 if nm in V}, b)
            log('  information: the replacement primary E1 (c1 free; adopted only on the reading "odd term indicated" of judgment note v2 section 5) would state eta %.6f with 2 se %.6f, model envelope %.6f (from %s; members %s), window leakage %.6f (sum |w_i| %.3e per nat), box %.6f, log sensitivity %.6f: half-width %.6f, %.1f times the primary\'s'
                % (termsE1['eta'], termsE1['two_se'], termsE1['env_model'], termsE1['env_model_from'], [nm for nm in ENVELOPE_E1 if nm in V],
                   termsE1['env_window'], termsE1['sum_abs_w'], termsE1['env_box'], termsE1['dlog'], termsE1['half'], termsE1['half'] / half if half > 0 else float('nan')))
        except Exception as exc:
            log('  information: the replacement primary E1\'s statement could not be formed (%s)' % exc)
    env_window_allow, half_allow = float('nan'), float('nan')
    if lines is not None and lines['applicable']:
        bvec_allow = np.array([(b_cs[R] + b_allow[R]) * inflate for R in P['radii']], dtype=float)
        env_window_allow = float(np.sum(np.abs(w) * bvec_allow))
        half_allow = math.sqrt(two_se ** 2 + env_model ** 2 + env_window_allow ** 2 + env_box ** 2 + dlog ** 2)
        log('  information: with the 2-se upper limits of all 5 sublattice frequencies in every bound (b_i %.2f to %.2f nats) the leakage term would be %.6f and the half-width %.6f'
            % (bvec_allow.min(), bvec_allow.max(), env_window_allow, half_allow))
    preview = allow_incomplete and bool(incomplete)
    preview_unc = 0.0
    if preview:
        rows_by_R = {int(round(r['Rc'])): r for r in rows}
        err = np.array([math.sqrt(2.0) * pool['sigma1'] * math.sqrt(R) * math.sqrt(rows_by_R[R]['passed_w2']) for R in P['radii']], dtype=float)
        preview_unc = float(np.sqrt(np.sum((w * err) ** 2)))
        log('  PREVIEW sampling uncertainty of eta from the passed weights (sqrt 2 sigma_1 sqrt R (sum_missing omega^2)^1/2 per radius, through the eta row): %.6f; not part of the error statement' % preview_unc)
    lp = None
    if do_lp:
        Bvec = bvec + P['box_unc']
        lp = lp_interval(rows, [c['4piR2'], c['one']], Bvec)
        if lp.get('status') == 0:
            lp['contains_P'] = bool(lp['min'] - 1e-12 <= eta <= lp['max'] + 1e-12)
            log('  set-membership interval (LP by %s, B_i = b_i + beta_i; conditional on these measured bounds, not a certificate): [%.6f, %.6f], half-width %.6f (influence bound %.6f; contains P: %s%s%s)'
                % (lp['method'], lp['min'], lp['max'], lp['half_width'], lp['influence_bound'], lp['contains_P'],
                   ('; HiGHS agrees to %.1e' % lp['method_agreement']) if 'method_agreement' in lp else ('; HiGHS: %s' % lp.get('highs_message', 'not run')),
                   ('; %d vertices' % lp['n_vertices']) if 'n_vertices' in lp else ''))
        else:
            log('  set-membership interval: LP failed (%s)' % lp.get('message'))
    log('RESULT%s: eta = %.5f, half-width %.5f%s; target 5e-4 %s%s'
        % (' (PREVIEW)' if preview else '', eta, half, (' (+- %.5f preview sampling uncertainty)' % preview_unc) if preview else '',
           'MET' if half <= TARGET else 'NOT MET',
           '' if len(P['radii']) == len(radii) else ' (radii %s of %s)' % (P['radii'], list(radii))))
    log('        a/l_P = 2 sqrt(eta) = %.5f +- %.5f' % (2.0 * math.sqrt(eta), half / math.sqrt(eta)))
    if sphere is not None:
        diff = eta - sphere[0]
        comb = math.sqrt(half ** 2 + sphere[1] ** 2)
        log('        sphere average under test (argument; the web value is a hypothesis until the lane\'s planar route lands): %.5f +- %.5f; difference %+.5f against the combined half-width %.5f: %s'
            % (sphere[0], sphere[1], diff, comb, 'consistent' if abs(diff) <= comb else 'tension'))
    for nm, what in (('E1', 'c1 free'), ('I8', 'the deleaked primary'), ('I8c1', 'deleaked, c1 free'), ('I3', 'the version-1 estimator, c1 free'),
                     ('I3b', 'the version-1 estimator, c1 = 0'), ('I2', 'the rectangular window'), ('I1', 'the free-log fit'),
                     ('I4', 'no box correction'), ('I5', 'A_4 term only')):
        if nm in V:
            log('        %s: eta %.6f (%+.6f)' % (what, V[nm]['eta'], V[nm]['eta'] - eta))
    log('        G6 reading: %s' % g6['status'])
    log('        comparisons: planar [100] %.5f, [111] %.14f; J7 mesh 4 %.4f (labelled extrapolation 0.919)' % (PLANAR_100, PLANAR_111, J7_MESH4))
    res = dict(eta=eta, half_width=half, two_se=two_se, env_model=env_model, env_model_from=env_argmax, env_window=env_window, env_box=env_box,
               dlog=dlog, deta_dclog=deta_dclog, a_over_lP=2.0 * math.sqrt(eta), a_over_lP_half_width=half / math.sqrt(eta), target_met=bool(half <= TARGET),
               preview=preview, preview_unc=preview_unc, radii_fit=radii_fit, incomplete=incomplete, V=V, rows=rows, rows_all=rows_all, box=box, ext=ext,
               g5=g5, sigma1=pool['sigma1'], pool=pool, periodogram=pg, lines=lines, shifts=shifts, limits=limits, b=b, b_cs=b_cs, b_half=b_half,
               b_allow=b_allow, env_window_allow=env_window_allow, half_width_allow=half_allow, inflate=inflate, half_flag=half_flag,
               detected=half_terms, lp=lp, A4=A4, w4_from=w4_from,
               g6=g6, P_terms=termsP, E1_terms=termsE1, window_level=wl, shift_check=chk, leak_table=leak_table, half_coef=half_coef,
               wl_terms=wl_terms, g6_error=g6_error)
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
    if 'I8' in res['V']:
        I8 = res['V']['I8']
        ax.plot(I8['radii'], I8['resid'], 'kx', ms=6, label='deleaked residual (I8: predicted lattice leakage removed)')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('R')
    ax.set_ylabel('nats')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(prefix + '_residuals.png', dpi=130)
    plt.close(fig)
    if res['shifts']:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for Rc, sh in sorted(res['shifts'].items()):
            line, = ax.plot(sh['shifts'], sh['eps'], label='R %d' % Rc)
            if 'lam' in sh:
                ax.plot(sh['shifts'], sh['lam'], ls='--', lw=0.8, color=line.get_color())
        ax.set_xlabel('window shift s (solid: measured eps(s); dashed: leakage predicted from the fitted lines)')
        ax.set_ylabel('eps(s) (nats)')
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(prefix + '_shift.png', dpi=130)
        plt.close(fig)
    pg = res['periodogram']
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(pg['grid'], pg['grid_amps'], lw=0.8, label='single-frequency periodogram (leaks through the sampling window)')
    for f in pg['freqs']:
        ax.axvline(f, color='g', lw=0.5, alpha=0.6)
    for f in pg['half_freqs']:
        ax.axvline(f, color='r', lw=0.5, alpha=0.6, ls='--')
    L = res.get('lines')
    if L is not None:
        ax.plot(L['f_lat'], L['amps_lat'], 'g^', ms=4, label='joint fit, lattice lines')
        ax.errorbar(L['f_half'], L['amps_half'], yerr=2.0 * L['se_half'], fmt='rs', ms=4, capsize=2, label='joint fit, sublattice (2 se)')
    ax.set_xlabel('frequency (cycles per lattice unit); green = lattice, red dashed = sublattice')
    ax.set_ylabel('amplitude (nats per sqrt R)')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(prefix + '_periodogram.png', dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# synthetic data and the selftest
# ----------------------------------------------------------------------------------------------
def synthetic_records(radii, wide_upto, eta0=0.919, c0=-5.0, A4=jb.A4_RECORD, e4=0.37, b6=-13.5, osc=0.0, mode='interval',
                      half_shift=0.0, seed=1, redo_v1=True, c1=0.0, osc_mmax=64):
    """Records of the production design (windows of width 3 at R <= wide_upto, 2 above, every ball at 8R; the version-1
    balls also at 4R and 6R for R <= 48 and at 4R (plus 8R if redo_v1) above) for a smooth model plus an optional shell
    oscillation.  mode 'weighted': the ball value is the Hann W=2 window-weighted interval mean of the smooth model (exactly
    representable by the estimator); 'interval': the unweighted interval mean (the version-2 construction; its sawtooth
    leaks about 0.7 R^-3/2 nats); 'midpoint': the smooth model at the interval midpoint.  The oscillation
    sqrt(R_mid) sum_m a_m cos(2 pi sqrt m R_mid + phi_m) over m <= osc_mmax (amplitudes osc U(0.5, 1.5) m^-0.75) plus,
    with half_shift > 0, a sublattice line at sqrt 3/2 of amplitude half_shift a_1.  c1 adds a linear term c1 <R> to
    the smooth model (version 3c; 0 by default, the records then equal those of version 3b for the same arguments)."""
    rng = np.random.default_rng(seed)
    nmax = int(math.ceil((max(radii) + 2.0) ** 2)) + 1
    Mx = int(math.floor(math.sqrt(nmax))) + 1
    ax = np.arange(-Mx, Mx + 1)
    X1, X2, X3 = np.meshgrid(ax, ax, ax, indexing='ij')
    norms = np.sort((X1 ** 2 + X2 ** 2 + X3 ** 2).ravel())
    ms = [m for m in range(1, int(osc_mmax) + 1) if jw.is_sum_of_three_squares(m)]
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
            base = np.asarray(base, dtype=float)
            smooth = eta0 * base[COL['4piR2']] + C_LOG * base[COL['lnR']] + c0 + c1 * base[COL['R']]
            oscv = 0.0
            if osc:
                oscv = float(np.sqrt(Rm) * (np.sum(amps * np.cos(2.0 * np.pi * freqs * Rm + phases)) + half_amp * np.cos(2.0 * np.pi * half_f * Rm + half_phase)))
            S_inf = smooth + oscv
            truth[n] = dict(S_inf=S_inf, osc=oscv, smooth=smooth, Rc=Rc, R_lo=R_lo, R_hi=R_hi, R_mid=Rm, plain=plain, base=base)
            if n not in v1:
                boxes = [8 * Rc]
            elif Rc <= 48:
                boxes = [4 * Rc, 6 * Rc, 8 * Rc]
            else:
                boxes = [4 * Rc, 8 * Rc] if redo_v1 else [4 * Rc]
            for L in boxes:
                u = (R_vol / L) ** 4
                recs[(n, L)] = dict(n=n, L=L, S=S_inf + (A4 + e4) * u + b6 * u ** 1.5, n_sites=ns, R_vol=R_vol, file='synthetic', method='synthetic')
    par = dict(eta0=eta0, c0=c0, c1=c1, A4=A4, e4=e4, b6=b6, ms=ms, freqs=freqs, amps=amps, phases=phases, half_f=half_f, half_amp=half_amp, mode=mode,
               osc_mmax=int(osc_mmax))
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


def predicted_shift_mismatch(truth, Rc, shifts, beta_true, W=2.0):
    """For synthetic data whose ball values are regressor means truth[n]['base'] . beta_true: the shift-test curve that the
    representation mismatch alone produces, sum_n omega_n(s) (base_n - <f>_{w(s),n}) . beta_true (None if a ball of a
    shifted window is not in the truth table)."""
    out = []
    for s in shifts:
        tab = jw.window_table(float(Rc) + float(s), W, 'hann')
        tot = 0.0
        for i, n in enumerate(tab['n']):
            om = float(tab['omega'][i])
            if om <= 0.0:
                continue
            t = truth.get(int(n))
            if t is None:
                return None
            tot += om * float((t['base'] - np.asarray(tab['means'][i], dtype=float)) @ beta_true)
        out.append(tot)
    return np.array(out)


def selftest(radii=None, wide_upto=40, log=print, seed=1, osc=3.0):
    radii = list(radii or RADII_DEFAULT)
    quiet = lambda *a, **k: None  # noqa: E731
    A4, zap, _ = jb.a4_derived(check=True)
    out = {}

    recs, truth, par = synthetic_records(radii, wide_upto, A4=A4, mode='weighted', osc=0.0, seed=seed)
    log('=== selftest A1: exactly representable synthetic data (ball values = Hann W=2 weighted interval means of the smooth model) ===')
    log('(A1 and A2 plant no oscillation: their oscillation blocks, shift tests and leakage bounds measure the representation mismatch of the construction (the per-ball residual is taken against the width-3 table\'s means; the shifted windows see the edge balls with the unshifted weighting), not the estimator, and are not read; the sublattice test and the G6 adjudication are not applicable there by their own rules; the A1 shift curves are predicted from the construction below)')
    rA1 = analyze(recs, radii, wide_upto, log=log, sphere=None, A4=A4)
    P, E1 = rA1['V']['P'], rA1['V']['E1']
    beta_true1 = np.zeros(NREG)
    beta_true1[COL['4piR2']], beta_true1[COL['lnR']], beta_true1[COL['one']] = par['eta0'], C_LOG, par['c0']
    devs = []
    for Rc, sh in rA1['shifts'].items():
        pred = predicted_shift_mismatch(truth, Rc, sh['shifts'], beta_true1)
        if pred is not None:
            devs.append(float(np.max(np.abs(sh['eps'] - pred))))
    out['A1'] = dict(d_eta=P['eta'] - par['eta0'], d_c0=param(P, 'one') - par['c0'], d_eta_E1=E1['eta'] - par['eta0'], c1_E1=param(E1, 'R'),
                     d_e4=rA1['box']['e4_ref'] - par['e4'], d_b6=rA1['box']['b6_ref'] - par['b6'],
                     max_box_err=float(max(abs(rA1['ext'][n]['S_inf'] - truth[n]['S_inf']) for n in truth)), rms=P['rms'], lp=rA1['lp'],
                     shift_mismatch_dev=(max(devs) if devs else float('nan')), n_shift_radii=len(devs))
    log('selftest A1: eta - eta0 %.2e, c0 - c0_true %.2e; E1: eta - eta0 %.2e, c1 %.2e; box: e4 %.1e, b6 %.1e, max |S_inf - truth| %.1e; P rms %.1e nats; LP [%.9f, %.9f] by %s'
        % (out['A1']['d_eta'], out['A1']['d_c0'], out['A1']['d_eta_E1'], out['A1']['c1_E1'], out['A1']['d_e4'], out['A1']['d_b6'],
           out['A1']['max_box_err'], out['A1']['rms'], rA1['lp'].get('min', float('nan')), rA1['lp'].get('max', float('nan')), rA1['lp'].get('method')))
    log('selftest A1: the shift curves equal the construction\'s representation mismatch sum_n omega_n(s) (base_n - <f>_{w(s),n}) . beta_true to %.1e at %d radii: the G3 WARNING above is the construction\'s (the 2 balls straddling each edge of the unshifted window), not the shift test\'s, and is not read'
        % (out['A1']['shift_mismatch_dev'], out['A1']['n_shift_radii']))

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
    log('\n=== selftest B: synthetic shell oscillation at the lattice frequencies (single-ball rms %.2f sqrt R nats; no sublattice line) ===' % single_rms)
    rB = analyze(recs, radii, wide_upto, log=log, sphere=None, A4=A4)
    P = rB['V']['P']
    bounds = np.array([rB['b'][R] for R in P['radii']])
    pg = rB['periodogram']
    L = rB['lines']
    amps_true = dict(zip(par['ms'], par['amps']))
    out['B'] = dict(d_eta=P['eta'] - par['eta0'], half_width=rB['half_width'], env_window=rB['env_window'], env_model=rB['env_model'], env_box=rB['env_box'],
                    resid=P['resid'], bounds=bounds, covered=int(np.sum(np.abs(P['resid']) <= bounds + 1e-12)), n_radii=len(bounds),
                    sigma1=rB['sigma1'], sigma1_true=single_rms, inflate=rB['inflate'],
                    A_meas={nm: float(a) for nm, a in zip(pg['names'], pg['amps'])},
                    A_joint=({nm: v[0] for nm, v in L['named'].items()} if L is not None else {}),
                    se_joint=({nm: v[1] for nm, v in L['named'].items()} if L is not None else {}),
                    A_true={'1': amps_true[1], 'sqrt2': amps_true[2], 'sqrt3': amps_true[3], '2': amps_true[4], 'sqrt5': amps_true[5]},
                    half_ratio_periodogram=float(np.max(pg['half_amps']) / pg['A1']),
                    half_ratio_joint=(float(np.max(L['amps_half']) / L['A1']) if L is not None else float('nan')),
                    half_amps_joint=(L['amps_half'] if L is not None else None), half_se_joint=(L['se_half'] if L is not None else None),
                    half_amps_periodogram=pg['half_amps'], leak_pred=(L['leak_pred'] if L is not None else None),
                    joint_cond=(L['cond'] if L is not None else float('nan')), joint_max_corr=(L['max_corr'] if L is not None else float('nan')),
                    joint_applicable=(bool(L['applicable']) if L is not None else False),
                    half_flag=rB['half_flag'], lp=rB['lp'], g6=rB['g6'],
                    c1_E1=param(rB['V']['E1'], 'R'), se_c1_E1=param_se(rB['V']['E1'], 'R'),
                    c1_I8c1=param(rB['V'].get('I8c1'), 'R'), se_c1_I8c1=param_se(rB['V'].get('I8c1'), 'R'),
                    rms_I8=(rB['V']['I8']['rms'] if 'I8' in rB['V'] else float('nan')),
                    d_eta_E1=rB['V']['E1']['eta'] - par['eta0'], d_eta_v1=(rB['V']['I3b']['eta'] - par['eta0']) if 'I3b' in rB['V'] else float('nan'),
                    d_eta_rect=(rB['V']['I2']['eta'] - par['eta0']) if 'I2' in rB['V'] else float('nan'))
    log('selftest B: eta - eta0: P %.2e (half-width %.2e, env_window %.2e, env_model %.2e), E1 %.2e, version-1 estimator %.2e, rectangular %.2e; '
        'bound covers the P residual at %d of %d radii; shift/CS ratio %.2f; sigma_1 %.3f (true %.3f)'
        % (out['B']['d_eta'], out['B']['half_width'], out['B']['env_window'], out['B']['env_model'], out['B']['d_eta_E1'], out['B']['d_eta_v1'], out['B']['d_eta_rect'],
           out['B']['covered'], out['B']['n_radii'], out['B']['inflate'], out['B']['sigma1'], single_rms))
    if L is not None:
        log('selftest B: joint amplitudes against the planted: ' + '  '.join('%s %.3f+-%.3f (true %.3f)' % (k, out['B']['A_joint'][k], out['B']['se_joint'][k], out['B']['A_true'][k]) for k in ('1', 'sqrt2', 'sqrt3', '2', 'sqrt5'))
            + '; single-frequency periodogram A(sqrt3) %.3f (the version-3 failure); sublattice/A(1): joint %.3f, periodogram %.3f (the version-3 artifact); detected %s; cond %.1f, max corr %.2f'
            % (out['B']['A_meas']['sqrt3'], out['B']['half_ratio_joint'], out['B']['half_ratio_periodogram'], rB['half_flag'], out['B']['joint_cond'], out['B']['joint_max_corr']))
    log('selftest B: G6 adjudication on data with no smooth odd term: E1 c1 %+.4f +- %.4f; deleaked I8c1 c1 %s; deleaked rms %.4f against P\'s %.4f (the lines beyond m = 20 remain); shift-model largest ratio %s; reading: %s'
        % (out['B']['c1_E1'], out['B']['se_c1_E1'], ('%+.4f +- %.4f' % (out['B']['c1_I8c1'], out['B']['se_c1_I8c1'])) if out['B']['c1_I8c1'] is not None else 'not fitted',
           out['B']['rms_I8'], P['rms'], ('%.2f' % rB['g6']['max_ratio']) if 'max_ratio' in rB['g6'] else '-', rB['g6']['status']))

    recs2, truth2, par2 = synthetic_records(radii, wide_upto, A4=A4, mode='midpoint', osc=osc, half_shift=0.3, seed=seed)
    rB2 = analyze(recs2, radii, wide_upto, log=quiet, sphere=None, A4=A4)
    P2 = rB2['V']['P']
    bounds2 = np.array([rB2['b'][R] for R in P2['radii']])
    pg2 = rB2['periodogram']
    L2 = rB2['lines']
    k32 = pg2['half_names'].index('sqrt3/2')
    out['B2'] = dict(half_flag=rB2['half_flag'], half_ratio_periodogram=float(pg2['half_amps'][k32] / pg2['A1']),
                     half_ratio_joint=(float(L2['amps_half'][k32] / L2['A1']) if L2 is not None else float('nan')),
                     half_se_joint=(float(L2['se_half'][k32]) if L2 is not None else float('nan')),
                     detected=rB2['detected'], d_eta=P2['eta'] - par2['eta0'], covered=int(np.sum(np.abs(P2['resid']) <= bounds2 + 1e-12)),
                     n_radii=len(bounds2), half_width=rB2['half_width'], g6=rB2['g6'])
    log('selftest B2 (sublattice line sqrt3/2 planted at 0.3 A(1)): detected %s (%s); joint ratio %.3f +- %.3f (periodogram %.3f); eta - eta0 %.2e; bound covers %d of %d; half-width %.2e; the detected line enters the leakage model: G6 reading %s'
        % (out['B2']['half_flag'], [t[0] for t in out['B2']['detected']], out['B2']['half_ratio_joint'], out['B2']['half_se_joint'] / max(L2['A1'], 1e-300) if L2 is not None else float('nan'),
           out['B2']['half_ratio_periodogram'], out['B2']['d_eta'], out['B2']['covered'], out['B2']['n_radii'], out['B2']['half_width'], rB2['g6']['status']))

    c1_plant = -0.05
    recs3, truth3, par3 = synthetic_records(radii, wide_upto, A4=A4, mode='midpoint', osc=osc, osc_mmax=20, c1=c1_plant, seed=seed)
    rB3 = analyze(recs3, radii, wide_upto, log=quiet, sphere=None, A4=A4)
    P3, E13 = rB3['V']['P'], rB3['V']['E1']
    I8c13 = rB3['V'].get('I8c1')
    wl3 = rB3['window_level']
    out['B3'] = dict(c1_true=c1_plant, c1_E1=param(E13, 'R'), se_c1_E1=param_se(E13, 'R'), c1_I8c1=param(I8c13, 'R'), se_c1_I8c1=param_se(I8c13, 'R'),
                     c1_window_level=(wl3['c1'] if wl3 is not None else float('nan')), se_window_level=(wl3['c1_se'] if wl3 is not None else float('nan')),
                     g6=rB3['g6'], d_eta_P=P3['eta'] - par3['eta0'], d_eta_E1=E13['eta'] - par3['eta0'],
                     d_eta_I8c1=((I8c13['eta'] - par3['eta0']) if I8c13 is not None else float('nan')),
                     half_width_P=rB3['half_width'], half_width_E1=(rB3['E1_terms']['half'] if rB3['E1_terms'] else float('nan')))
    log('selftest B3 (lattice lines m <= 20 at the measured size plus a planted linear term c1 = %.3f): E1 c1 %+.4f +- %.4f; deleaked I8c1 c1 %s; window-level smooth coefficient %+.4f +- %.4f; shift-model largest ratio %s; G6 reading: %s; eta - eta0: P %.2e (the bias of the premise when it is false), E1 %.2e, I8c1 %.2e; half-widths P %.2e, E1 %.2e'
        % (c1_plant, out['B3']['c1_E1'], out['B3']['se_c1_E1'], ('%+.4f +- %.4f' % (out['B3']['c1_I8c1'], out['B3']['se_c1_I8c1'])) if out['B3']['c1_I8c1'] is not None else 'not fitted',
           out['B3']['c1_window_level'], out['B3']['se_window_level'], ('%.2f' % rB3['g6']['max_ratio']) if 'max_ratio' in rB3['g6'] else '-', rB3['g6']['status'],
           out['B3']['d_eta_P'], out['B3']['d_eta_E1'], out['B3']['d_eta_I8c1'], out['B3']['half_width_P'], out['B3']['half_width_E1']))

    keep = set()
    for Rc in radii:
        W = 3.0 if Rc <= wide_upto else 2.0
        lev = jw.ball_levels(Rc, W)
        v1 = set(jw.v1_balls(Rc))
        for (n, Lbox) in recs:
            if n in v1 or lev.get(n, 99) <= 5:
                keep.add((n, Lbox))
    rC = analyze({k: v for k, v in recs.items() if k in keep}, radii, wide_upto, log=quiet, sphere=None, A4=A4, allow_incomplete=True)
    out['C'] = dict(d_eta=rC['V']['P']['eta'] - par['eta0'], completeness=[r['completeness'] for r in rC['rows']], preview_unc=rC['preview_unc'])
    log('selftest C: the level-5 sampled windows (completeness %.2f to %.2f, weight-passing rule with consistent regressors): eta - eta0 %.2e against the preview sampling uncertainty %.2e'
        % (min(out['C']['completeness']), max(out['C']['completeness']), out['C']['d_eta'], out['C']['preview_unc']))
    out['D'] = dict(A1_lp=out['A1']['lp'], B_lp=out['B']['lp'], eta0=par['eta0'])
    log('selftest D: LP interval A1 [%.9f, %.9f] (half-width %.1e, influence bound %.1e, %s), B [%.6f, %.6f] (half-width %.2e, influence bound %.2e, %s); truth %.6f'
        % (out['A1']['lp'].get('min', float('nan')), out['A1']['lp'].get('max', float('nan')), out['A1']['lp'].get('half_width', float('nan')), out['A1']['lp'].get('influence_bound', float('nan')), out['A1']['lp'].get('method'),
           out['B']['lp'].get('min', float('nan')), out['B']['lp'].get('max', float('nan')), out['B']['lp'].get('half_width', float('nan')), out['B']['lp'].get('influence_bound', float('nan')), out['B']['lp'].get('method'), par['eta0']))
    return out


def main():
    ap = argparse.ArgumentParser(description='J4 version-3c fit')
    ap.add_argument('--dirs', nargs='*', default=[], help='directories with records (version 1, version 2, prefix)')
    ap.add_argument('--radii', type=int, nargs='*', default=RADII_DEFAULT)
    ap.add_argument('--wide-upto', type=int, default=40)
    ap.add_argument('--w4-from', type=int, default=None, help='Hann width 4 at R >= this radius (the specification\'s contingency)')
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
                  allow_incomplete=args.allow_incomplete, plot=args.plot, do_lp=not args.no_lp, w4_from=args.w4_from)
    if res is None:
        sys.exit(1)
    if args.out:
        slim = {k: v for k, v in res.items() if k not in ('rows', 'rows_all', 'ext', 'pool')}
        slim['rows'] = [{k: v for k, v in r.items() if k != 'balls'} for r in res['rows_all']]
        slim['V'] = {nm: {k: v for k, v in r.items() if k not in ('row_eta', 'lnR_bar')} for nm, r in res['V'].items()}
        slim['pool'] = dict(R=res['pool']['R'], e=res['pool']['e'], d=res['pool']['d'], sigma1=res['pool']['sigma1'], per_window=res['pool']['per_window'])
        slim['shifts'] = {R: {k: v for k, v in sh.items() if k not in ('X', 'G_half')} for R, sh in res['shifts'].items()}
        slim['ext_summary'] = [dict(n=x['n'], R_vol=x['R_vol'], nbox=x['nbox'], Ls=x['Ls'], S_inf=x['S_inf'], S_cont=x['S_cont'], box_correction=x['box_correction'],
                                    box_unc=x['box_unc'], b4=x['b4'], b6=x['b6'], delta_by_L=x['delta_by_L']) for x in res['ext'].values()]
        with open(args.out, 'w') as fh:
            json.dump(to_jsonable(slim), fh, indent=1)


if __name__ == '__main__':
    main()
