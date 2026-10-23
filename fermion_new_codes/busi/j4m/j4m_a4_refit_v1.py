#!/usr/bin/env python3
"""
j4m_a4_refit_v1.py -- the A_4 second route reanalysed on one rung set (busi lane, J4 massive; the amendment that
J4_massive_judgment_v1.1 section (c) called for).  Reads only the massless records (mu 0: production, ladder,
ladder2 and P22 kinds), prints, writes a json and a tsv table.  No solver is run here.

Data.  For every massless ball with both routes: y = S_L - S_inf (the antiperiodic box minus the box-free
Schwinger-Bessel kernel), u = (R_vol/L)^4, s = sqrt(u) = (R_vol/L)^2, z = y/u (the single-box coefficient).
At R 12, 16, 20, 24, 28 every one of the 8 balls (R + k/16, k = -7, -5, ..., 7) has the rungs L = 6R, 8R, 10R, 12R
(R the window centre); the production box is an extra rung where it differs (L 192 = 16R at R 12 and 9.6R at R 20);
the 4R rung (R 12 to 24) lies below the set and enters only the single-box table and the continuity check.

Per-ball forms, fitted in z with equal weight per rung (y weighted by 1/u, as the judgment prescribes) on the 4 rungs:
  F1  z = b4                   the mean single-box coefficient
  F2  z = b4 + b6 s            the specification's form, y = b4 u + b6 u^{3/2}
  F3  z = b4 + b8 s^2          y = b4 u + b8 u^2 (the second-order, relative-entropy form: the first law gives A_4 u
                               exactly in the continuum, the remainder is second order in the state change, so u^2)
  F4  z = b4 + b6 s + b8 s^2   both terms, 1 degree of freedom: THE PRIMARY (it nests F2 and F3; fixed before the run)
and F2 unweighted in y (the version 1 convention) for comparison.  Per R: median, spread (max |b4 - median|), std;
e_4(R) = median b4(R) - A_4; its R dependence as c/R^2 against a free power c R^-p; the extrapolation to infinite R.

Stated error of the extrapolated b4(inf), fixed before the fit was run (the specification's additive convention):
  2 se(intercept of the primary extrapolation) + envelope + kernel term + floor term, where
  primary extrapolation: the per-R medians of the F4 b4 fitted to b4_inf + c/R^2 over R 12, 16, 20, 24, 28 (OLS, 3 dof);
  envelope: the largest |b4_inf(variant) - b4_inf(primary)| over the pre-registered variants
    V1 free power b4_inf + c R^-p;  V2 drop R 12;  V3 per-ball form F2;  V4 per-ball form F3;
    V5 rungs 8R, 10R, 12R only (F4 exact on 3 rungs);  V6 means instead of medians;
    V7 the 40 balls individually against 1/R_vol^2;  V8 + c1/R (b4_inf + c1/R + c2/R^2);
    V9 the production box added as a fifth rung where it differs (R 12, R 20);
    V10 the global fit z = b4_inf + c/R_vol^2 + b6 s + b8 s^2 on all 160 (ball, rung) points;
  kernel term: max |inf_alt_minus_inf| over the records that carry it (n <= 15,000 sites), divided by the smallest u of
    the rung set (the specification's definition);
  floor term: the arithmetic floor of S_L - S_inf, 4.2e-9 nats (the massive production maximum at the same boxes,
    where the true box term is e^{-19} smaller; judgment 1.1 (a)), divided by the smallest u of the rung set.
The clause "the kernel route reproduces A_4 within its stated error" is met if |b4_inf - A_4| <= that error.
The 1/R^2 form of e_4 is called established if the free power p lies within 2 se of 2 and the residuals of the c/R^2
fit lie within the per-R spreads at every radius.

Usage: j4m_a4_refit_v1.py --records DIR [DIR ...] [--out-json F] [--out-tsv F]
"""
import argparse
import glob
import json
import math
import os

import numpy as np
from scipy.optimize import least_squares

A4 = -32.96566448807753          # j4box_v3.A4_RECORD = (128/15) Z_AP(2), the first route
FLOOR_NATS = 4.2e-9              # the massive production maximum |S_L - S_inf| (judgment 1.1 (a)): the arithmetic floor of y
RUNG_RATIOS = (6, 8, 10, 12)
RADII = (12, 16, 20, 24, 28)
OFFSETS = (-7, -5, -3, -1, 1, 3, 5, 7)
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, '..', '..', 'results', 'j4m'))

PRESCRIPTION = """Stated error of b4(inf), fixed before the fit: 2 se(primary intercept) + envelope + kernel term + floor term.
  primary: per-ball F4 (z = b4 + b6 s + b8 s^2 on the rungs 6R, 8R, 10R, 12R, equal weight per rung), per-R medians,
           fitted to b4_inf + c/R^2 over R 12, 16, 20, 24, 28 (OLS, 3 dof);
  envelope: max |b4_inf(V) - b4_inf(primary)| over V1 free power, V2 drop R 12, V3 form F2, V4 form F3,
           V5 rungs 8R-12R only, V6 means, V7 40 balls vs 1/R_vol^2, V8 + c1/R, V9 production box as a 5th rung, V10 global fit;
  kernel term: max |inf_alt_minus_inf| / u_min;  floor term: 4.2e-9 / u_min.
  Clause met if |b4_inf - A_4| <= stated error.  e_4 ~ 1/R^2 established if |p - 2| <= 2 se_p and the c/R^2 residuals lie
  within the per-R spreads."""


# ----------------------------------------------------------------------------------------------
# records
# ----------------------------------------------------------------------------------------------
def load_records(dirs):
    recs = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, '*.json'))):
            try:
                with open(f) as fh:
                    r = json.load(fh)
                box = r['routes']['box']
                inf = r['routes']['inf']
            except Exception:
                continue
            if float(r['mu']) != 0.0 or 'S' not in inf:
                continue
            checks = r.get('checks') or {}
            R_vol = float(r['R_vol'])
            L = int(r['L'])
            y = float(box['S']) - float(inf['S'])
            u = (R_vol / L) ** 4
            recs.append(dict(id=r.get('id', os.path.basename(f)), kind=r.get('kind'), n=int(r['n2max']), L=L,
                             R=(float(r['window_centre']) if r.get('window_centre') is not None else None),
                             k=r.get('offset_k'), n_sites=int(r['n']), R_vol=R_vol, R_hi=float(r['R_hi']),
                             S_box=float(box['S']), S_inf=float(inf['S']), y=y, u=u, s=math.sqrt(u), z=y / u,
                             ratio=L / float(r['window_centre']) if r.get('window_centre') else None,
                             inf_alt=checks.get('inf_alt_minus_inf')))
    return recs


def group_balls(recs):
    balls = {}
    for r in recs:
        balls.setdefault((r['R'], r['k'], r['n']), []).append(r)
    for key in balls:
        balls[key].sort(key=lambda r: r['L'])
    return balls


# ----------------------------------------------------------------------------------------------
# least squares
# ----------------------------------------------------------------------------------------------
def ols(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    scale = np.abs(X).max(axis=0)
    scale[scale == 0] = 1.0
    Xs = X / scale
    beta_s = np.linalg.lstsq(Xs, y, rcond=None)[0]
    beta = beta_s / scale
    resid = y - X @ beta
    dof = n - p
    if dof > 0:
        s2 = float(resid @ resid) / dof
        cov = s2 * np.linalg.inv(Xs.T @ Xs) / np.outer(scale, scale)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    else:
        se = np.full(p, float('nan'))
    return beta, se, resid, dof


FORMS = {'F1': ('1',), 'F2': ('1', 's'), 'F3': ('1', 's2'), 'F4': ('1', 's', 's2')}


def design(s, cols):
    s = np.asarray(s, dtype=float)
    d = {'1': np.ones_like(s), 's': s, 's2': s * s}
    return np.column_stack([d[c] for c in cols])


def fit_form(form, s, z):
    cols = FORMS[form]
    beta, se, resid, dof = ols(design(s, cols), z)
    out = dict(form=form, b4=float(beta[0]), dof=int(dof), rms=float(np.sqrt(np.mean(resid ** 2))), resid=resid.tolist())
    for c, b in zip(cols, beta):
        if c == 's':
            out['b6'] = float(b)
        elif c == 's2':
            out['b8'] = float(b)
    return out


def fit_f2_unweighted(u, y):
    u = np.asarray(u, dtype=float)
    beta, se, resid, dof = ols(np.column_stack([u, u ** 1.5]), y)
    return dict(form='F2y', b4=float(beta[0]), b6=float(beta[1]), dof=int(dof), rms=float(np.sqrt(np.mean(resid ** 2))))


# ----------------------------------------------------------------------------------------------
# the per-ball table on the common rung set
# ----------------------------------------------------------------------------------------------
def per_ball_table(balls):
    rows = []
    for (R, k, n), lst in sorted(balls.items(), key=lambda t: ((t[0][0] or 0), (t[0][1] if t[0][1] is not None else 99), t[0][2])):
        if R is None or k is None or int(R) not in RADII:
            continue
        byL = {r['L']: r for r in lst}
        rungs = [int(q * R) for q in RUNG_RATIOS]
        if any(L not in byL for L in rungs):
            raise RuntimeError('ball R %s k %s n %d lacks a rung: have %s' % (R, k, n, sorted(byL)))
        rr = [byL[L] for L in rungs]
        s = np.array([r['s'] for r in rr])
        u = np.array([r['u'] for r in rr])
        y = np.array([r['y'] for r in rr])
        z = np.array([r['z'] for r in rr])
        row = dict(R=int(R), k=int(k), n=int(n), n_sites=rr[0]['n_sites'], R_vol=rr[0]['R_vol'], rungs=rungs,
                   u=u.tolist(), z=z.tolist(), y=y.tolist())
        for form in FORMS:
            row[form] = fit_form(form, s, z)
        row['F2y'] = fit_f2_unweighted(u, y)
        # diagnostic, not in the envelope: the exact cubic z = b4 + b6 s + b8 s^2 + b10 s^3 on the 4 rungs (the next term of the
        # series, u^{5/2}); its b4 is the F4 value corrected for the omitted term, and b10 its coefficient
        cub = np.polyfit(s, z, 3)
        row['F5_cubic'] = dict(form='F5', b4=float(cub[3]), b6=float(cub[2]), b8=float(cub[1]), b10=float(cub[0]), dof=0, rms=0.0)
        # rungs 8R, 10R, 12R only (V5): F4 exact, F2 and F3 with 1 dof
        row['F4_8to12'] = fit_form('F4', s[1:], z[1:])
        row['F3_8to12'] = fit_form('F3', s[1:], z[1:])
        row['F2_8to12'] = fit_form('F2', s[1:], z[1:])
        # the production box as a fifth rung where it differs (V9)
        prod = [r for r in lst if r['kind'] == 'prod' and r['L'] not in rungs]
        if prod:
            p = prod[0]
            s5 = np.append(s, p['s'])
            z5 = np.append(z, p['z'])
            row['prod_extra'] = dict(L=p['L'], ratio=p['L'] / R, u=p['u'], z=p['z'])
            row['F4_prod'] = fit_form('F4', s5, z5)
            row['F2_prod'] = fit_form('F2', s5, z5)
            row['F3_prod'] = fit_form('F3', s5, z5)
        else:
            row['prod_extra'] = None
            row['F4_prod'] = row['F4']
            row['F2_prod'] = row['F2']
            row['F3_prod'] = row['F3']
        # continuity with j4box_v3 and the version 1 fits: F2 unweighted in y on the 4R, 6R, 8R ladder where it exists
        L4 = int(4 * R)
        if L4 in byL:
            lad = [byL[L4], byL[int(6 * R)], byL[int(8 * R)]]
            row['F2y_4to8'] = fit_f2_unweighted([r['u'] for r in lad], [r['y'] for r in lad])
            row['z_4R'] = byL[L4]['z']
        else:
            row['F2y_4to8'] = None
            row['z_4R'] = None
        rows.append(row)
    return rows


def aggregate(rows, key, R_list=RADII, stat='median'):
    """Per-R medians (or means), spreads and stds of rows[key]['b4'] (key a form label), with b6 and b8 where present."""
    out = {}
    for R in R_list:
        sel = [r for r in rows if r['R'] == R]
        if not sel:
            continue
        b4 = np.array([r[key]['b4'] for r in sel])
        centre = float(np.median(b4)) if stat == 'median' else float(np.mean(b4))
        entry = dict(R=R, count=len(sel), b4=centre, b4_median=float(np.median(b4)), b4_mean=float(np.mean(b4)),
                     spread=float(np.max(np.abs(b4 - np.median(b4)))), std=float(np.std(b4, ddof=1)) if len(b4) > 1 else 0.0,
                     e4=centre - A4, e4_R2=(centre - A4) * R * R, b4_values=b4.tolist(),
                     rms_median=float(np.median([r[key]['rms'] for r in sel])))
        for c in ('b6', 'b8'):
            if c in sel[0][key]:
                v = np.array([r[key][c] for r in sel])
                entry[c + '_median'] = float(np.median(v))
                entry[c + '_spread'] = float(np.max(np.abs(v - np.median(v))))
        out[R] = entry
    return out


# ----------------------------------------------------------------------------------------------
# the R dependence and the extrapolation
# ----------------------------------------------------------------------------------------------
def extrap_1R2(Rs, b4s):
    Rs = np.asarray(Rs, dtype=float)
    beta, se, resid, dof = ols(np.column_stack([np.ones_like(Rs), 1.0 / Rs ** 2]), b4s)
    return dict(model='b4_inf + c/R^2', b4_inf=float(beta[0]), se=float(se[0]), c=float(beta[1]), c_se=float(se[1]),
                resid=resid.tolist(), dof=int(dof), rms=float(np.sqrt(np.mean(resid ** 2))), radii=list(map(float, Rs)))


def extrap_1R_1R2(Rs, b4s):
    Rs = np.asarray(Rs, dtype=float)
    beta, se, resid, dof = ols(np.column_stack([np.ones_like(Rs), 1.0 / Rs, 1.0 / Rs ** 2]), b4s)
    return dict(model='b4_inf + c1/R + c2/R^2', b4_inf=float(beta[0]), se=float(se[0]), c1=float(beta[1]), c1_se=float(se[1]),
                c2=float(beta[2]), c2_se=float(se[2]), resid=resid.tolist(), dof=int(dof), rms=float(np.sqrt(np.mean(resid ** 2))))


def extrap_power(Rs, b4s, fix_inf=None):
    """b4_inf + c R^-p (3 parameters), or with b4_inf fixed (2 parameters: the e_4 power-law test)."""
    Rs = np.asarray(Rs, dtype=float)
    b4s = np.asarray(b4s, dtype=float)

    def model(th):
        if fix_inf is None:
            return th[0] + th[1] * Rs ** (-th[2])
        return fix_inf + th[0] * Rs ** (-th[1])

    def fun(th):
        return model(th) - b4s

    x0 = [A4, 20.0, 2.0] if fix_inf is None else [20.0, 2.0]
    sol = least_squares(fun, x0, x_scale=[1.0, 10.0, 1.0] if fix_inf is None else [10.0, 1.0], xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=20000)
    th = sol.x
    resid = fun(th)
    dof = len(Rs) - len(th)
    J = sol.jac
    try:
        cov = (float(resid @ resid) / dof) * np.linalg.inv(J.T @ J) if dof > 0 else np.full((len(th), len(th)), float('nan'))
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        se = np.full(len(th), float('nan'))
    if fix_inf is None:
        return dict(model='b4_inf + c R^-p', b4_inf=float(th[0]), se=float(se[0]), c=float(th[1]), c_se=float(se[1]), p=float(th[2]),
                    p_se=float(se[2]), resid=resid.tolist(), dof=int(dof), rms=float(np.sqrt(np.mean(resid ** 2))), success=bool(sol.success))
    return dict(model='A_4 + c R^-p', c=float(th[0]), c_se=float(se[0]), p=float(th[1]), p_se=float(se[1]), resid=resid.tolist(),
                dof=int(dof), rms=float(np.sqrt(np.mean(resid ** 2))), success=bool(sol.success))


def e4_forms(agg):
    """e_4(R) = median b4(R) - A_4 against c/R^2 (1 parameter) and c R^-p (2 parameters); the loglog slope as a cross-check."""
    Rs = np.array([agg[R]['R'] for R in sorted(agg)], dtype=float)
    e4 = np.array([agg[R]['e4'] for R in sorted(agg)])
    spreads = np.array([agg[R]['spread'] for R in sorted(agg)])
    w = Rs ** -2
    c = float(np.sum(w * e4) / np.sum(w * w))
    resid = e4 - c * w
    dof = len(Rs) - 1
    c_se = float(math.sqrt(float(resid @ resid) / dof / np.sum(w * w))) if dof > 0 else float('nan')
    out = dict(radii=Rs.tolist(), e4=e4.tolist(), e4_R2=(e4 * Rs ** 2).tolist(), spreads=spreads.tolist(),
               fit_1R2=dict(c=c, c_se=c_se, resid=resid.tolist(), rms=float(np.sqrt(np.mean(resid ** 2))), dof=dof,
                            within_spreads=bool(np.all(np.abs(resid) <= spreads))))
    out['fit_power'] = extrap_power(Rs, e4 + A4, fix_inf=A4)
    if np.all(e4 > 0):
        beta, se, r2, d2 = ols(np.column_stack([np.ones_like(Rs), np.log(Rs)]), np.log(e4))
        out['loglog'] = dict(p=float(-beta[1]), p_se=float(se[1]), log_c=float(beta[0]))
    p, pse = out['fit_power']['p'], out['fit_power']['p_se']
    out['established_1R2'] = bool(abs(p - 2.0) <= 2.0 * pse and out['fit_1R2']['within_spreads'])
    return out


def global_fit(rows, fix_inf=None):
    """z = b4_inf + c/R_vol^2 + b6 s + b8 s^2 over all (ball, rung) points of the common rung set."""
    Rv, S, Z = [], [], []
    for r in rows:
        for u, z in zip(r['u'], r['z']):
            Rv.append(r['R_vol'])
            S.append(math.sqrt(u))
            Z.append(z)
    Rv, S, Z = np.array(Rv), np.array(S), np.array(Z)
    if fix_inf is None:
        X = np.column_stack([np.ones_like(S), 1.0 / Rv ** 2, S, S * S])
        beta, se, resid, dof = ols(X, Z)
        return dict(model='z = b4_inf + c/R_vol^2 + b6 s + b8 s^2', points=len(Z), b4_inf=float(beta[0]), se=float(se[0]), c=float(beta[1]),
                    c_se=float(se[1]), b6=float(beta[2]), b6_se=float(se[2]), b8=float(beta[3]), b8_se=float(se[3]), dof=int(dof),
                    rms=float(np.sqrt(np.mean(resid ** 2))), max_abs_resid=float(np.max(np.abs(resid))))
    X = np.column_stack([1.0 / Rv ** 2, S, S * S])
    beta, se, resid, dof = ols(X, Z - fix_inf)
    return dict(model='z = A_4 + c/R_vol^2 + b6 s + b8 s^2', points=len(Z), c=float(beta[0]), c_se=float(se[0]), b6=float(beta[1]),
                b6_se=float(se[1]), b8=float(beta[2]), b8_se=float(se[2]), dof=int(dof), rms=float(np.sqrt(np.mean(resid ** 2))),
                max_abs_resid=float(np.max(np.abs(resid))))


def pooled_fits(rows):
    """The judgment's pooled fit z = b4 + b6 sqrt(u) with equal weight per rung on the 160 points, with F3, F4 and the
    version 1 convention (unweighted in y) on the same points.  These mix the radii: they estimate the R-average of b4(R)."""
    S, Z, U, Y = [], [], [], []
    for r in rows:
        for u, z, y in zip(r['u'], r['z'], r['y']):
            S.append(math.sqrt(u))
            Z.append(z)
            U.append(u)
            Y.append(y)
    S, Z, U, Y = map(np.array, (S, Z, U, Y))
    out = dict(points=len(Z))
    for form in ('F2', 'F3', 'F4'):
        beta, se, resid, dof = ols(design(S, FORMS[form]), Z)
        out[form] = dict(b4=float(beta[0]), se_b4=float(se[0]), coef=beta.tolist(), se=se.tolist(), rms=float(np.sqrt(np.mean(resid ** 2))), dof=int(dof))
    beta, se, resid, dof = ols(np.column_stack([U, U ** 1.5]), Y)
    out['F2y'] = dict(b4=float(beta[0]), se_b4=float(se[0]), b6=float(beta[1]), rms_y=float(np.sqrt(np.mean(resid ** 2))), dof=int(dof))
    return out


def single_box_table(recs):
    """z = y/u by R and L/R: median and spread over the 8 offsets; the P22 balls (k None) listed on their own."""
    groups = {}
    for r in recs:
        if r['R'] is None:
            continue
        key = (int(r['R']), '%.1f' % (r['L'] / r['R']), 'p22' if r['k'] is None else 'reg')
        groups.setdefault(key, []).append(r)
    out = []
    for key in sorted(groups, key=lambda t: (t[0], float(t[1]), t[2])):
        lst = groups[key]
        z = np.array([r['z'] for r in lst])
        out.append(dict(R=key[0], L_over_R=float(key[1]), L=lst[0]['L'], balls=key[2], count=len(lst), z_median=float(np.median(z)),
                        z_spread=float(np.max(np.abs(z - np.median(z)))), z_minus_A4=float(np.median(z) - A4),
                        kinds=sorted(set(r['kind'] for r in lst))))
    return out


# ----------------------------------------------------------------------------------------------
# the analysis
# ----------------------------------------------------------------------------------------------
def analyse(recs):
    balls = group_balls(recs)
    rows = per_ball_table(balls)
    res = dict(A4_first_route=A4, records=len(recs), balls=len(rows), rungs=list(RUNG_RATIOS), radii=list(RADII), prescription=PRESCRIPTION)
    res['per_ball'] = rows
    agg = {form: aggregate(rows, form) for form in ('F1', 'F2', 'F3', 'F4', 'F2y', 'F4_8to12', 'F3_8to12', 'F2_8to12', 'F4_prod', 'F5_cubic')}
    agg['F4_mean'] = aggregate(rows, 'F4', stat='mean')
    res['per_radius'] = {k: {str(R): v for R, v in a.items()} for k, a in agg.items()}
    # the e_4 forms on the primary
    res['e4'] = {form: e4_forms(agg[form]) for form in ('F4', 'F2', 'F3')}

    # the extrapolation: primary and the pre-registered variants
    def med(a):
        Rs = [a[R]['R'] for R in sorted(a)]
        return np.array(Rs, dtype=float), np.array([a[R]['b4'] for R in sorted(a)])
    Rs, b4 = med(agg['F4'])
    prim = extrap_1R2(Rs, b4)
    variants = {}
    variants['V1_free_power'] = extrap_power(Rs, b4)
    variants['V2_drop_R12'] = extrap_1R2(Rs[1:], b4[1:])
    variants['V3_form_F2'] = extrap_1R2(*med(agg['F2']))
    variants['V4_form_F3'] = extrap_1R2(*med(agg['F3']))
    variants['V5_rungs_8to12'] = extrap_1R2(*med(agg['F4_8to12']))
    variants['V6_means'] = extrap_1R2(*med(agg['F4_mean']))
    variants['V7_all_balls'] = extrap_1R2(np.array([r['R_vol'] for r in rows]), np.array([r['F4']['b4'] for r in rows]))
    variants['V8_plus_1R'] = extrap_1R_1R2(Rs, b4)
    variants['V9_prod_rung'] = extrap_1R2(*med(agg['F4_prod']))
    variants['V10_global'] = global_fit(rows)
    env = max(abs(v['b4_inf'] - prim['b4_inf']) for v in variants.values())
    env_by = {k: float(v['b4_inf'] - prim['b4_inf']) for k, v in variants.items()}
    env_without_V8 = max(abs(d) for k, d in env_by.items() if not k.startswith('V8'))
    umin = min(min(r['u']) for r in rows)
    alts = [abs(float(r['inf_alt'])) for r in recs if r['inf_alt'] is not None]
    kernel_term = (max(alts) / umin) if alts else float('nan')
    floor_term = FLOOR_NATS / umin
    two_se = 2.0 * prim['se']
    stated = two_se + env + kernel_term + floor_term
    dev = prim['b4_inf'] - A4
    res['extrapolation'] = dict(primary=prim, variants=variants, envelope=env, envelope_by_variant=env_by, envelope_without_V8=env_without_V8,
                                two_se=two_se, kernel_term=kernel_term, kernel_records=len(alts), kernel_max_alt=(max(alts) if alts else None),
                                floor_term=floor_term, floor_nats=FLOOR_NATS, u_min=umin, stated_error=stated, deviation_from_A4=dev,
                                reproduces_A4=bool(abs(dev) <= stated), relative_deviation=dev / abs(A4), relative_stated=stated / abs(A4))
    # diagnostics: the extrapolation offset by offset (the shape scatter's reach into b4_inf), the global fit with A_4 fixed
    byk = {}
    for k in OFFSETS:
        sel = sorted([r for r in rows if r['k'] == k], key=lambda r: r['R'])
        if len(sel) == len(RADII):
            e = extrap_1R2(np.array([r['R'] for r in sel], dtype=float), np.array([r['F4']['b4'] for r in sel]))
            byk[str(k)] = dict(b4_inf=e['b4_inf'], c=e['c'], rms=e['rms'])
    vals = np.array([v['b4_inf'] for v in byk.values()])
    res['by_offset'] = dict(fits=byk, b4_inf_median=float(np.median(vals)), b4_inf_spread=float(np.max(np.abs(vals - np.median(vals)))),
                            b4_inf_std=float(np.std(vals, ddof=1)))
    res['global_fixed_A4'] = global_fit(rows, fix_inf=A4)
    # diagnostic (not in the envelope): the exact cubic's medians extrapolated in 1/R^2, and the F4 residual pattern
    res['diagnostic_cubic'] = dict(extrapolation=extrap_1R2(*med(agg['F5_cubic'])),
                                   b10_median_by_R={str(R): float(np.median([r['F5_cubic']['b10'] for r in rows if r['R'] == R])) for R in RADII},
                                   F4_resid_sign_pattern_count=int(sum(1 for r in rows if np.all(np.sign(r['F4']['resid']) == np.sign(rows[0]['F4']['resid'])))),
                                   F4_rms_min=float(min(r['F4']['rms'] for r in rows)), F4_rms_max=float(max(r['F4']['rms'] for r in rows)))
    res['pooled'] = pooled_fits(rows)
    res['single_box'] = single_box_table(recs)
    # continuity: the 4R, 6R, 8R ladder fitted as j4box_v3 and version 1 did
    cont = {}
    for R in RADII:
        sel = [r for r in rows if r['R'] == R and r['F2y_4to8'] is not None]
        if sel:
            b4s = np.array([r['F2y_4to8']['b4'] for r in sel])
            b6s = np.array([r['F2y_4to8']['b6'] for r in sel])
            cont[str(R)] = dict(b4_median=float(np.median(b4s)), b6_median=float(np.median(b6s)), e4=float(np.median(b4s) - A4), count=len(sel))
    res['continuity_4to8_F2y'] = cont
    return res


# ----------------------------------------------------------------------------------------------
# the report
# ----------------------------------------------------------------------------------------------
def fmt(v, f='%.4f'):
    return (f % v) if v is not None and v == v else 'n/a'


def print_report(res):
    print('J4 massive, A_4 second route refit on one rung set: %d massless records, %d balls at R %s with rungs %s x R' %
          (res['records'], res['balls'], res['radii'], res['rungs']))
    print(res['prescription'])
    print('A_4 (first route, j4box_v3) = %.4f' % A4)
    print()
    print('Per-ball fits (z = y/u, equal weight per rung): per R the median b4 [spread; std], e_4 = median - A_4, e_4 R^2; b6, b8 medians')
    for form, label in (('F1', 'F1 z = b4 (mean single-box coefficient)'), ('F2', 'F2 z = b4 + b6 s  (spec form)'),
                        ('F3', 'F3 z = b4 + b8 s^2 (u^2 form)'), ('F4', 'F4 z = b4 + b6 s + b8 s^2 (PRIMARY)'),
                        ('F2y', 'F2 unweighted in y (version 1 convention; its rms is in y, not z)'), ('F4_8to12', 'F4 on rungs 8R, 10R, 12R (exact)'),
                        ('F3_8to12', 'F3 on rungs 8R, 10R, 12R'), ('F4_prod', 'F4 with the production box as a 5th rung (R 12, 20)'),
                        ('F5_cubic', 'diagnostic: exact cubic z = b4 + b6 s + b8 s^2 + b10 s^3 on the 4 rungs (not in the envelope)')):
        print('  ' + label)
        for R, a in sorted(res['per_radius'][form].items(), key=lambda t: float(t[0])):
            extra = ''
            if 'b6_median' in a:
                extra += '  b6 %+.2f [%.2f]' % (a['b6_median'], a['b6_spread'])
            if 'b8_median' in a:
                extra += '  b8 %+.1f [%.1f]' % (a['b8_median'], a['b8_spread'])
            print('    R %2s: b4 %.4f [%.4f; %.4f]  e_4 %+.4f  e_4 R^2 %+.1f  rms %.1e%s' %
                  (R, a['b4'], a['spread'], a['std'], a['e4'], a['e4_R2'], a['rms_median'], extra))
    print()
    for form in ('F4', 'F2', 'F3'):
        e = res['e4'][form]
        print('e_4(R) forms on %s medians: e_4 = %s; e_4 R^2 = %s' % (form, ['%.4f' % v for v in e['e4']], ['%.1f' % v for v in e['e4_R2']]))
        print('  c/R^2: c = %.2f +- %.2f, residuals %s (rms %.4f), within the per-R spreads: %s' %
              (e['fit_1R2']['c'], e['fit_1R2']['c_se'], ['%+.4f' % v for v in e['fit_1R2']['resid']], e['fit_1R2']['rms'], e['fit_1R2']['within_spreads']))
        fp = e['fit_power']
        print('  c R^-p: p = %.2f +- %.2f, c = %.1f +- %.1f, rms %.4f%s; 1/R^2 established: %s' %
              (fp['p'], fp['p_se'], fp['c'], fp['c_se'], fp['rms'], ('; loglog p = %.2f +- %.2f' % (e['loglog']['p'], e['loglog']['p_se'])) if 'loglog' in e else '',
               e['established_1R2']))
    print()
    x = res['extrapolation']
    p = x['primary']
    print('Extrapolation to infinite R (primary: F4 medians, b4_inf + c/R^2): b4_inf = %.4f, se %.4f, c = %.1f +- %.1f, residuals %s, rms %.4f' %
          (p['b4_inf'], p['se'], p['c'], p['c_se'], ['%+.4f' % v for v in p['resid']], p['rms']))
    for k, v in x['variants'].items():
        extra = ''
        if 'p' in v:
            extra = ', p = %.2f +- %.2f' % (v['p'], v['p_se'])
        if 'c1' in v:
            extra = ', c1 = %.2f +- %.2f, c2 = %.1f +- %.1f' % (v['c1'], v['c1_se'], v['c2'], v['c2_se'])
        if 'b8' in v:
            extra = ', c = %.1f +- %.1f, b6 = %+.2f +- %.2f, b8 = %+.1f +- %.1f, %d points' % (v['c'], v['c_se'], v['b6'], v['b6_se'], v['b8'], v['b8_se'], v['points'])
        print('  %-16s b4_inf = %.4f (se %.4f, rms %.4f)  displacement %+.4f%s' % (k, v['b4_inf'], v['se'], v['rms'], x['envelope_by_variant'][k], extra))
    print('  envelope %.4f (without V8: %.4f); 2 se %.4f; kernel term %.1e (max |inf_alt - inf| %s on %d records, u_min %.2e); floor term %.1e' %
          (x['envelope'], x['envelope_without_V8'], x['two_se'], x['kernel_term'], fmt(x['kernel_max_alt'], '%.1e'), x['kernel_records'], x['u_min'], x['floor_term']))
    print('RESULT: b4(inf) = %.4f +- %.4f (stated error); A_4 = %.4f; deviation %+.4f (%.3f percent); stated error %.3f percent -> %s' %
          (p['b4_inf'], x['stated_error'], A4, x['deviation_from_A4'], 100 * x['relative_deviation'], 100 * x['relative_stated'],
           'the kernel route REPRODUCES A_4 within the stated error' if x['reproduces_A4'] else 'NOT reproduced within the stated error'))
    bo = res['by_offset']
    print('  by offset k (5 radii each): b4_inf median %.4f, spread %.4f, std %.4f: %s' %
          (bo['b4_inf_median'], bo['b4_inf_spread'], bo['b4_inf_std'], {k: '%.4f' % v['b4_inf'] for k, v in bo['fits'].items()}))
    g = res['global_fixed_A4']
    print('  global fit with A_4 fixed: c = %.1f +- %.1f, b6 = %+.2f +- %.2f, b8 = %+.1f +- %.1f, rms %.4f, max |resid| %.4f (%d points)' %
          (g['c'], g['c_se'], g['b6'], g['b6_se'], g['b8'], g['b8_se'], g['rms'], g['max_abs_resid'], g['points']))
    d = res['diagnostic_cubic']
    print('  diagnostic (not in the envelope): the F4 residual has the same sign pattern on %d of %d balls (rms %.1e to %.1e), the next term of the series; '
          'the exact cubic gives b10 medians %s and extrapolates to b4_inf = %.4f (se %.4f, c = %.1f)' %
          (d['F4_resid_sign_pattern_count'], res['balls'], d['F4_rms_min'], d['F4_rms_max'], {R: '%.0f' % v for R, v in d['b10_median_by_R'].items()},
           d['extrapolation']['b4_inf'], d['extrapolation']['se'], d['extrapolation']['c']))
    print()
    po = res['pooled']
    print('Pooled fits on the %d (ball, rung) points (these mix the radii): F2 z = b4 + b6 s: b4 = %.4f +- %.4f, b6 = %+.2f, rms %.4f; '
          'F3: b4 = %.4f +- %.4f, b8 = %+.1f, rms %.4f; F4: b4 = %.4f +- %.4f, b6 = %+.2f, b8 = %+.1f, rms %.4f; version 1 convention (y unweighted): b4 = %.4f +- %.4f, b6 = %+.2f' %
          (po['points'], po['F2']['b4'], po['F2']['se_b4'], po['F2']['coef'][1], po['F2']['rms'], po['F3']['b4'], po['F3']['se_b4'], po['F3']['coef'][1], po['F3']['rms'],
           po['F4']['b4'], po['F4']['se_b4'], po['F4']['coef'][1], po['F4']['coef'][2], po['F4']['rms'], po['F2y']['b4'], po['F2y']['se_b4'], po['F2y']['b6']))
    print('Continuity: the 4R, 6R, 8R ladder fitted as j4box_v3 and version 1 did (y = b4 u + b6 u^1.5, unweighted): %s' %
          {R: 'b4 %.3f b6 %.1f e_4 %+.3f' % (v['b4_median'], v['b6_median'], v['e4']) for R, v in res['continuity_4to8_F2y'].items()})
    print()
    print('Single-box coefficients z = (S_L - S_inf)/u, median [spread] over the balls, z - A_4:')
    for t in res['single_box']:
        print('  R %2d  L %3d  L/R %4.1f  %s  n %d: z = %.4f [%.4f]  z - A_4 %+.4f' % (t['R'], t['L'], t['L_over_R'], t['balls'], t['count'], t['z_median'], t['z_spread'], t['z_minus_A4']))


def write_outputs(res, out_json, out_tsv):
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        tmp = out_json + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(res, fh, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
        os.replace(tmp, out_json)
    if out_tsv:
        os.makedirs(os.path.dirname(os.path.abspath(out_tsv)), exist_ok=True)
        cols = ['R', 'k', 'n', 'n_sites', 'R_vol', 'z_4R', 'z_6R', 'z_8R', 'z_10R', 'z_12R', 'z_prod_extra', 'L_prod_extra',
                'b4_F1', 'b4_F2', 'b6_F2', 'rms_F2', 'b4_F3', 'b8_F3', 'rms_F3', 'b4_F4', 'b6_F4', 'b8_F4', 'rms_F4',
                'b4_F2y', 'b6_F2y', 'b4_F4_8to12', 'b4_F3_8to12', 'b4_F4_prod', 'b4_F2y_4to8', 'b6_F2y_4to8', 'e4_F4']
        with open(out_tsv, 'w') as fh:
            fh.write('# per-ball fits of z = (S_L - S_inf)/u on the rungs 6R, 8R, 10R, 12R, u = (R_vol/L)^4, s = sqrt(u); A_4 = %.14f\n' % A4)
            fh.write('\t'.join(cols) + '\n')
            for r in res['per_ball']:
                pe = r['prod_extra']
                vals = [r['R'], r['k'], r['n'], r['n_sites'], '%.6f' % r['R_vol'], fmt(r['z_4R'], '%.6f')] + ['%.6f' % z for z in r['z']] + \
                       [fmt(pe['z'] if pe else None, '%.6f'), (pe['L'] if pe else 'n/a'),
                        '%.6f' % r['F1']['b4'], '%.6f' % r['F2']['b4'], '%.4f' % r['F2']['b6'], '%.2e' % r['F2']['rms'],
                        '%.6f' % r['F3']['b4'], '%.3f' % r['F3']['b8'], '%.2e' % r['F3']['rms'],
                        '%.6f' % r['F4']['b4'], '%.4f' % r['F4']['b6'], '%.3f' % r['F4']['b8'], '%.2e' % r['F4']['rms'],
                        '%.6f' % r['F2y']['b4'], '%.4f' % r['F2y']['b6'], '%.6f' % r['F4_8to12']['b4'], '%.6f' % r['F3_8to12']['b4'],
                        '%.6f' % r['F4_prod']['b4'], fmt(r['F2y_4to8']['b4'] if r['F2y_4to8'] else None, '%.6f'),
                        fmt(r['F2y_4to8']['b6'] if r['F2y_4to8'] else None, '%.4f'), '%.6f' % (r['F4']['b4'] - A4)]
                fh.write('\t'.join(str(v) for v in vals) + '\n')


def main():
    ap = argparse.ArgumentParser(description='A_4 second route refit on one rung set (reads records only)')
    ap.add_argument('--records', nargs='+', required=True)
    ap.add_argument('--out-json', default=os.path.join(RESULTS, 'J4m_a4_refit_v1.json'))
    ap.add_argument('--out-tsv', default=os.path.join(RESULTS, 'J4m_a4_refit_v1.tsv'))
    args = ap.parse_args()
    recs = load_records(args.records)
    res = analyse(recs)
    print_report(res)
    write_outputs(res, args.out_json, args.out_tsv)
    print('\nwrote %s and %s' % (args.out_json, args.out_tsv))


if __name__ == '__main__':
    main()
