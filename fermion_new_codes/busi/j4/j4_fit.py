#!/usr/bin/env python3
"""
j4_fit.py -- finite-box extrapolation, registry averaging and the fit of eta (spec section 4 and 6).
Reads the production JSONs (R_nominal, offset_sixteenths, L, R_eff, S), extrapolates each (R, k) in
(R_eff/L)^4 [and ^6] where 2 [3] boxes exist, corrects single-box cases with the median b4 of the largest
multi-box radii, averages over the 8 registry offsets, and fits
    S = 4 pi eta R^2 + c_1 R + c_log ln R + c_0 [+ c_{-1}/R]
with c_log fixed at -44/45 (primary) and free, plus the stability variants of the spec.
Usage: j4_fit.py --indir prod [--pattern 'R*_o*_L*.json'] [--out J4_fit_v1.json] [--allow-partial]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

C_LOG = -44.0 / 45.0
ALL_OFFSETS = [-7, -5, -3, -1, 1, 3, 5, 7]


def load(indir, pattern):
    recs = []
    for f in sorted(glob.glob(os.path.join(indir, pattern))):
        with open(f) as fh:
            d = json.load(fh)
        if 'S' not in d or d.get('R_nominal') is None:
            continue
        recs.append(dict(R=int(round(d['R_nominal'])), k=int(d['offset_sixteenths']), L=int(d['L']),
                         Reff=float(d['R_eff']), S=float(d['S']), n=int(d['n']), file=f))
    return recs


def box_extrapolate(recs):
    """Per (R, k): S_inf and b4 from the available boxes.  Returns dict (R,k) -> info."""
    groups = {}
    for r in recs:
        groups.setdefault((r['R'], r['k']), []).append(r)
    out = {}
    for key, lst in groups.items():
        lst = sorted(lst, key=lambda r: r['L'])
        Reff = lst[0]['Reff']
        u = np.array([(Reff / r['L']) ** 4 for r in lst])
        S = np.array([r['S'] for r in lst])
        info = dict(R=key[0], k=key[1], Reff=Reff, Ls=[r['L'] for r in lst], S_by_L=S.tolist(), u=u.tolist())
        if len(lst) >= 3:
            X = np.column_stack([np.ones_like(u), u, u ** 1.5])
            coef = np.linalg.lstsq(X, S, rcond=None)[0]
            X2 = np.column_stack([np.ones_like(u), u])
            coef2 = np.linalg.lstsq(X2, S, rcond=None)[0]
            info.update(S_inf=float(coef[0]), b4=float(coef[1]), b6=float(coef[2]), S_inf_4only=float(coef2[0]), b4_4only=float(coef2[1]),
                        nbox=len(lst), S_raw_4R=float(S[0]))
        elif len(lst) == 2:
            b4 = float((S[1] - S[0]) / (u[1] - u[0]))
            info.update(S_inf=float(S[1] - b4 * u[1]), b4=b4, b6=None, nbox=2, S_raw_4R=float(S[0]))
        else:
            info.update(S_inf=None, b4=None, b6=None, nbox=1, S_raw_4R=float(S[0]))
        out[key] = info
    # reference b4 from the multi-box radii with the largest R (median over their offsets)
    multi = [v for v in out.values() if v['nbox'] >= 2]
    b4_ref, b4_spread = None, None
    if multi:
        Rtop = sorted(set(v['R'] for v in multi))[-3:]
        vals = np.array([v['b4'] for v in multi if v['R'] in Rtop])
        b4_ref = float(np.median(vals))
        b4_spread = float(np.max(np.abs(vals - b4_ref))) if len(vals) > 1 else abs(b4_ref)
    for v in out.values():
        if v['nbox'] == 1:
            u1 = v['u'][0]
            if b4_ref is None:
                v['S_inf'] = v['S_raw_4R']
                v['box_correction'] = 0.0
                v['box_correction_unc'] = 33.0 * u1
            else:
                v['S_inf'] = v['S_raw_4R'] - b4_ref * u1
                v['box_correction'] = -b4_ref * u1
                v['box_correction_unc'] = b4_spread * u1
        else:
            v['box_correction'] = v['S_inf'] - v['S_raw_4R']
            v['box_correction_unc'] = 0.0
    return out, b4_ref, b4_spread


def registry_average(ext, offsets, allow_partial):
    """Per primary R: mean over offsets of S_inf and of the regressors.  Returns sorted list of rows."""
    byR = {}
    for (R, k), v in ext.items():
        if k in offsets:
            byR.setdefault(R, {})[k] = v
    rows = []
    for R in sorted(byR):
        have = sorted(byR[R])
        if len(have) < len(offsets) and not allow_partial:
            print('skipping R=%d: offsets present %s (need %s); use --allow-partial to include' % (R, have, offsets))
            continue
        vs = [byR[R][k] for k in have]
        Reff = np.array([v['Reff'] for v in vs])
        rows.append(dict(R=R, offsets=have, S=float(np.mean([v['S_inf'] for v in vs])),
                         S_raw=float(np.mean([v['S_raw_4R'] for v in vs])),
                         x=[float(np.mean(4 * np.pi * Reff ** 2)), float(np.mean(Reff)), float(np.mean(np.log(Reff))), 1.0,
                            float(np.mean(1.0 / Reff))],
                         box_unc=float(np.mean([v['box_correction_unc'] for v in vs]))))
    return rows


def lsq(X, y, w=None):
    if w is None:
        w = np.ones(len(y))
    Xw = X * np.sqrt(w)[:, None]
    yw = y * np.sqrt(w)
    beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = float(np.sum(w * resid ** 2) / dof) if dof > 0 else float('nan')
    cov = sigma2 * np.linalg.inv(Xw.T @ Xw) if dof > 0 else np.full((X.shape[1], X.shape[1]), np.nan)
    return beta, resid, np.sqrt(np.diag(cov)), dof


def fit(rows, free_log, extra_inv, drop_small=0, use_raw=False, weight=None):
    rows = sorted(rows, key=lambda r: r['R'])[drop_small:]
    Xall = np.array([r['x'] for r in rows])
    y = np.array([r['S_raw'] if use_raw else r['S'] for r in rows])
    cols = [0, 1, 3] + ([4] if extra_inv else [])
    names = ['4pi*eta*R^2', 'c1*R', 'c0'] + (['c_-1/R'] if extra_inv else [])
    if free_log:
        cols.insert(2, 2)
        names.insert(2, 'c_log*lnR')
    else:
        y = y - C_LOG * Xall[:, 2]
    X = Xall[:, cols]
    X[:, 0] = X[:, 0]          # 4 pi R^2 column: beta[0] = eta
    w = None
    if weight == 'R2':
        w = 1.0 / np.array([r['x'][0] for r in rows]) ** 2
    beta, resid, se, dof = lsq(X, y, w)
    out = dict(eta=float(beta[0]), se_eta=float(se[0]), params=dict(zip(names, beta.tolist())), se=dict(zip(names, se.tolist())),
               c_log=(float(beta[2]) if free_log else C_LOG), radii=[r['R'] for r in rows], residuals=resid.tolist(), dof=int(dof),
               rms_resid=float(np.sqrt(np.mean(resid ** 2))))
    return out


def main():
    ap = argparse.ArgumentParser(description='J4 fit')
    ap.add_argument('--indir', default='prod')
    ap.add_argument('--pattern', default='R*_o*_L*.json')
    ap.add_argument('--out', default=None)
    ap.add_argument('--allow-partial', action='store_true')
    args = ap.parse_args()
    recs = load(args.indir, args.pattern)
    if not recs:
        print('no records found in %s' % args.indir)
        sys.exit(1)
    print('records: %d cases, radii %s' % (len(recs), sorted(set(r['R'] for r in recs))))
    ext, b4_ref, b4_spread = box_extrapolate(recs)

    print('\nfinite-box extrapolation per (R, k) [b4 = coefficient of (R_eff/L)^4; continuum -32.97]:')
    for key in sorted(ext):
        v = ext[key]
        if v['nbox'] >= 2:
            print('  R %2d k %+d  L %-14s  b4 %8.3f  b6 %s  S_inf %.9f  correction at 4R %+.4f'
                  % (v['R'], v['k'], v['Ls'], v['b4'], ('%8.2f' % v['b6']) if v['b6'] is not None else '   --   ', v['S_inf'], v['box_correction']))
    single = [v for v in ext.values() if v['nbox'] == 1]
    print('  single-box cases: %d, corrected with b4_ref = %s (spread %s); correction %s nats with uncertainty up to %s'
          % (len(single), b4_ref, b4_spread, ('%.4f' % single[0]['box_correction']) if single else '-',
             ('%.4f' % max(v['box_correction_unc'] for v in single)) if single else '-'))

    rows = registry_average(ext, ALL_OFFSETS, args.allow_partial)
    if len(rows) < 4:
        print('fewer than 4 complete radii; no fit')
        sys.exit(1)
    print('\nregistry-averaged data (mean over offsets of the box-extrapolated S):')
    for r in rows:
        print('  R %2d  offsets %d  S %.6f  S_raw(4R) %.6f  S/(4 pi <R^2>) %.6f' % (r['R'], len(r['offsets']), r['S'], r['S_raw'], r['S'] / r['x'][0]))

    variants = [('V0 fixed log (primary)', dict(free_log=False, extra_inv=False)),
                ('V1 free log', dict(free_log=True, extra_inv=False)),
                ('V2 fixed log + c_-1/R', dict(free_log=False, extra_inv=True)),
                ('V3 fixed log, drop 2 smallest R', dict(free_log=False, extra_inv=False, drop_small=2)),
                ('V4 free log, drop 2 smallest R', dict(free_log=True, extra_inv=False, drop_small=2)),
                ('V7 fixed log, no box correction', dict(free_log=False, extra_inv=False, use_raw=True)),
                ('V8 fixed log, weights 1/R^4', dict(free_log=False, extra_inv=False, weight='R2'))]
    results = {}
    for name, kw in variants:
        results[name] = fit(rows, **kw)
    for name, offs in [('V5 fixed log, offsets {-7,-3,1,5}', [-7, -3, 1, 5]), ('V6 fixed log, offsets {-5,-1,3,7}', [-5, -1, 3, 7])]:
        sub = registry_average(ext, offs, args.allow_partial)
        if len(sub) >= 4:
            results[name] = fit(sub, free_log=False, extra_inv=False)

    print('\nfits (S = 4 pi eta R^2 + c1 R + c_log ln R + c0 [+ c_-1/R]; registry-averaged regressors):')
    primary = results['V0 fixed log (primary)']
    for name, res in results.items():
        print('  %-40s eta %.6f +- %.6f  c_log %+.4f  c1 %+.4f  c0 %+.3f  rms resid %.4f nats  radii %s'
              % (name, res['eta'], res['se_eta'], res['c_log'], res['params']['c1*R'], res['params']['c0'], res['rms_resid'], res['radii']))
    print('  V0 residuals per radius (nats): ' + ' '.join('%d:%+.4f' % (R, r) for R, r in zip(primary['radii'], primary['residuals'])))
    print('  V1 fitted c_log = %+.4f +- %.4f vs continuum -44/45 = %+.4f' % (results['V1 free log']['c_log'], results['V1 free log']['se']['c_log*lnR'], C_LOG))
    envelope = max(abs(res['eta'] - primary['eta']) for name, res in results.items() if name != 'V0 fixed log (primary)')
    half = float(np.sqrt((2 * primary['se_eta']) ** 2 + envelope ** 2))
    eta = primary['eta']
    a_over_lp = 2.0 * np.sqrt(eta)
    print('\nRESULT: eta = %.5f, half-width %.5f (2 se %.5f, envelope %.5f); target 5e-4 %s'
          % (eta, half, 2 * primary['se_eta'], envelope, 'MET' if half <= 5e-4 else 'NOT MET'))
    print('        a/l_P = 2 sqrt(eta) = %.5f +- %.5f' % (a_over_lp, half / np.sqrt(eta)))
    print('        comparisons: web sphere average 0.9192 +- 0.0002 (bounds 0.91905, 0.91930); J7 mesh 4: 0.9145 (extrapolation 0.919); planar [100] 0.78658, [111] 0.92338181')
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(dict(records=len(recs), b4_ref=b4_ref, b4_spread=b4_spread, extrapolation=ext_serializable(ext),
                           rows=rows, fits=results, eta=eta, half_width=half, envelope=envelope, two_se=2 * primary['se_eta'],
                           a_over_lP=a_over_lp, a_over_lP_half_width=half / np.sqrt(eta)), fh, indent=1)


def ext_serializable(ext):
    return [v for _, v in sorted(ext.items())]


if __name__ == '__main__':
    main()
