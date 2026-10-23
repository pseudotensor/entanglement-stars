#!/usr/bin/env python3
"""
j4m_fit_v1.py -- the fit of the massive J4 rerun (busi lane, task 4; spec section 3), reading only the records.

Per radius R: the 8 version-1 balls (R + k/16), their massless S_0 (box value minus the continuum box term
A_4 (R_vol/L)^4), their massive S_0.2 (box value; the massive box term is exponentially negligible and measured),
and the box-free values of both masses.  Fits (ordinary least squares on the registry means; point regressors
averaged over the 8 balls):
  paired P0 (primary):  S_0.2 - S_0 = 4 pi Delta_eta <r^2> + a <r> + (44/45) <ln r> + b + d <1/r^2>, log fixed;
  envelope E: P1 (d = 0), P2 (drop R 12), P3 (drop R 12, 14), P4 (+ e/r), P5 (volume-radius regressors), P6 (box-free);
  information: P7 (log free), P8 (log fixed at 0);
  separate: M0 massive 4 pi eta r^2 + A r + B + D/r^2 (P22 claim 9), M1 (D = 0), M2 (+ free log), M3 (drop R 12);
            Z0 massless lead's model (log fixed at -44/45), Z1 (c_1 = 0).
Half-width of Delta_eta: sqrt((2 se_P0)^2 + env^2).  Acceptance: |Delta_eta - target| <= tol (default -0.0317, 0.003).
Also: the A_4 reading from S_L - S_inf at mu = 0 (section 4.3), the massive box check (section 2), the P22 comparison.
Usage: j4m_fit_v1.py --records DIR [DIR ...] [--radii ...] [--mu 0.2] [--out-json F] [--out-tsv F] | --selftest
"""
import argparse
import glob
import json
import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
J4_DIR = os.path.normpath(os.path.join(HERE, '..', 'j4'))
for _p in (HERE, J4_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import j4window_v2 as jw      # noqa: E402
import j4box_v3 as jb         # noqa: E402

A4 = jb.A4_RECORD
CLOG = -44.0 / 45.0
TARGET_DELTA = -0.0317
TOL_DELTA = 0.003
RADII_DEFAULT = [12, 14, 16, 18, 20, 22, 24, 26, 28]
P22_TABLE = {12: dict(L=192, S1=1574.484054, S0=1629.144659), 16: dict(L=192, S1=2829.054113, S0=2928.140736),
             20: dict(L=192, S1=4450.055876, S0=4606.544637), 24: dict(L=192, S1=6380.492069, S0=6606.229064),
             28: dict(L=224, S1=8715.987087, S0=9024.744103)}
P22_GROUPS = {12: 0.8851736374, 16: 0.8892251529}
P22_SITES_28 = 91965


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
            except Exception:
                continue
            inf = r['routes'].get('inf') or {}
            checks = r.get('checks', {}) or {}
            fbr = r['routes'].get('box_fullblock') or {}
            recs.append(dict(path=f, n=int(r['n2max']), L=int(r['L']), mu=float(r['mu']), R=r.get('window_centre'),
                             k=r.get('offset_k'), kind=r.get('kind'), n_sites=int(r['n']), R_vol=float(r['R_vol']),
                             R_lo=float(r['R_lo']), R_hi=float(r['R_hi']), S_box=float(box['S']),
                             S_inf=(float(inf['S']) if 'S' in inf else None), inf_alt=checks.get('inf_alt_minus_inf'),
                             S_fullblock=fbr.get('S'), fullblock_minus=checks.get('fullblock_minus_sublattice')))
    return recs


def registry_mean(values, weights=None):
    """The registry average: a pure function of the values and the weights (equal weights by default); nothing is cached."""
    v = np.asarray(values, dtype=float)
    if weights is None:
        return float(np.mean(v))
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w * v) / np.sum(w))


def regressors(R):
    R = np.asarray(R, dtype=float)
    return dict(area=4.0 * np.pi * R * R, r=R, one=np.ones_like(R), lnr=np.log(R), invr=1.0 / R, invr2=1.0 / (R * R))


def build_table(recs, radii, mu1):
    by = {}
    for r in recs:
        by.setdefault((r['n'], r['mu']), []).append(r)

    def pick(n, mu):
        lst = by.get((n, mu), [])
        if not lst:
            return None
        prod = [r for r in lst if r['kind'] == 'prod']
        return max(prod or lst, key=lambda r: r['L'])

    table = {}
    for R in radii:
        balls = {}
        for k in jw.V1_OFFSETS:
            n = jw.ball_at_radius(R + k / 16.0)
            b = dict(n=n, R_eff=R + k / 16.0, R_vol=None, S0_box=None, S0_L=None, S0_corr=None, S1_box=None, S1_L=None,
                     S0_inf=None, S1_inf=None, box_corr0=None)
            r0, r1 = pick(n, 0.0), pick(n, mu1)
            if r0:
                u = (r0['R_vol'] / r0['L']) ** 4
                b.update(R_vol=r0['R_vol'], S0_box=r0['S_box'], S0_L=r0['L'], S0_inf=r0['S_inf'], S0_corr=r0['S_box'] - A4 * u, box_corr0=-A4 * u)
            if r1:
                b.update(S1_box=r1['S_box'], S1_L=r1['L'], S1_inf=r1['S_inf'])
                if b['R_vol'] is None:
                    b['R_vol'] = r1['R_vol']
            balls[k] = b
        table[R] = dict(balls=balls)
    return table


def ball_value(b, route, what):
    if route == 'box':
        S0, S1 = b['S0_corr'], b['S1_box']
    else:
        S0, S1 = b['S0_inf'], b['S1_inf']
    if what == 'S0':
        return S0
    if what == 'S1':
        return S1
    if S0 is None or S1 is None:
        return None
    return S1 - S0


def rows_for(table, radii, route, what, rmode='eff'):
    rows = []
    for R in radii:
        vals, Rs = [], []
        for k in sorted(table[R]['balls']):
            b = table[R]['balls'][k]
            v = ball_value(b, route, what)
            if v is None:
                continue
            rr = b['R_eff'] if rmode == 'eff' else b['R_vol']
            if rr is None:
                continue
            vals.append(v)
            Rs.append(rr)
        if not vals:
            continue
        reg = regressors(Rs)
        rows.append(dict(R=R, y=registry_mean(vals), X={c: float(np.mean(reg[c])) for c in reg}, count=len(vals),
                         ball_values=vals, r2_mean=float(np.mean(np.asarray(Rs) ** 2))))
    return rows


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
        se = np.full(p, np.nan)
    return beta, se, resid, dof


def fit_rows(rows, cols, fixed_log=0.0):
    if len(rows) < len(cols):
        return dict(ok=False, reason='fewer radii (%d) than parameters (%d)' % (len(rows), len(cols)), eta=np.nan, se_eta=np.nan, cols=cols)
    y = np.array([r['y'] - fixed_log * r['X']['lnr'] for r in rows])
    X = np.array([[r['X'][c] for c in cols] for r in rows])
    beta, se, resid, dof = ols(X, y)
    return dict(ok=True, cols=cols, fixed_log=fixed_log, beta=beta.tolist(), se=se.tolist(), eta=float(beta[0]),
                se_eta=float(se[0]), resid=resid.tolist(), dof=int(dof), radii=[r['R'] for r in rows],
                rms=float(np.sqrt(np.mean(resid ** 2))), counts=[r['count'] for r in rows])


def fit_variants(table, radii, mu1):
    base = ['area', 'r', 'one', 'invr2']
    rows_box = rows_for(table, radii, 'box', 'D')
    rows_inf = rows_for(table, radii, 'inf', 'D')
    rows_vol = rows_for(table, radii, 'box', 'D', rmode='vol')
    plus = -CLOG
    V = {}
    V['P0'] = dict(desc='paired, log fixed +44/45, d free (primary)', env=False, fit=fit_rows(rows_box, base, plus))
    V['P1'] = dict(desc='paired, d = 0', env=True, fit=fit_rows(rows_box, ['area', 'r', 'one'], plus))
    V['P2'] = dict(desc='paired, drop R 12', env=True, fit=fit_rows([r for r in rows_box if r['R'] > 12], base, plus))
    V['P3'] = dict(desc='paired, drop R 12, 14', env=True, fit=fit_rows([r for r in rows_box if r['R'] > 14], base, plus))
    V['P4'] = dict(desc='paired, + e/r', env=True, fit=fit_rows(rows_box, base + ['invr'], plus))
    V['P5'] = dict(desc='paired, volume-radius regressors', env=True, fit=fit_rows(rows_vol, base, plus))
    V['P6'] = dict(desc='paired, box-free values (both masses)', env=True, fit=fit_rows(rows_inf, base, plus))
    V['P7'] = dict(desc='paired, log free (information)', env=False, fit=fit_rows(rows_box, base + ['lnr'], 0.0))
    V['P8'] = dict(desc='paired, log fixed at 0 (information)', env=False, fit=fit_rows(rows_box, base, 0.0))
    rows_m = rows_for(table, radii, 'box', 'S1')
    rows_z = rows_for(table, radii, 'box', 'S0')
    V['M0'] = dict(desc='massive alone: 4 pi eta r^2 + A r + B + D/r^2', env=False, fit=fit_rows(rows_m, base, 0.0))
    V['M1'] = dict(desc='massive alone, D = 0', env=False, fit=fit_rows(rows_m, ['area', 'r', 'one'], 0.0))
    V['M2'] = dict(desc='massive alone, + free log', env=False, fit=fit_rows(rows_m, base + ['lnr'], 0.0))
    V['M3'] = dict(desc='massive alone, drop R 12', env=False, fit=fit_rows([r for r in rows_m if r['R'] > 12], base, 0.0))
    V['Z0'] = dict(desc="massless alone: lead's model, log fixed -44/45, c_1 free", env=False, fit=fit_rows(rows_z, ['area', 'r', 'one'], CLOG))
    V['Z1'] = dict(desc='massless alone, c_1 = 0', env=False, fit=fit_rows(rows_z, ['area', 'one'], CLOG))
    for key in V:
        V[key]['fit']['variant'] = key
        V[key]['fit']['desc'] = V[key]['desc']
    P0 = V['P0']['fit']
    env = 0.0
    for key, v in V.items():
        if v['env'] and v['fit'].get('ok') and P0.get('ok'):
            env = max(env, abs(v['fit']['eta'] - P0['eta']))
    out = dict(variants={k: v['fit'] for k, v in V.items()}, per_radius=rows_box)
    if P0.get('ok'):
        two_se = 2.0 * P0['se_eta'] if P0['dof'] > 0 else float('nan')
        hw = math.sqrt((two_se if two_se == two_se else 0.0) ** 2 + env ** 2)
        # sensitivity to the fixed log coefficient: Delta_eta changes by -w . <ln r> per unit of c_log
        rows = rows_box
        X = np.array([[r['X'][c] for c in base] for r in rows])
        scale = np.abs(X).max(axis=0)
        Xs = X / scale
        w = np.linalg.pinv(Xs)[0] / scale[0]
        dlog = -float(w @ np.array([r['X']['lnr'] for r in rows]))
        out.update(delta_eta=P0['eta'], two_se=two_se, env=env, half_width=hw, d_delta_eta_d_clog=dlog,
                   log_sensitivity_term=abs(CLOG * dlog), residual_rms=P0['rms'])
    M0, Z0 = V['M0']['fit'], V['Z0']['fit']
    if M0.get('ok') and Z0.get('ok'):
        out['separate_difference'] = M0['eta'] - Z0['eta']
        out['separate_difference_2se'] = 2.0 * math.sqrt((M0['se_eta'] if M0['dof'] > 0 else 0.0) ** 2 + (Z0['se_eta'] if Z0['dof'] > 0 else 0.0) ** 2)
        out['eta_0'] = Z0['eta']
        out['eta_0_2se'] = 2.0 * Z0['se_eta'] if Z0['dof'] > 0 else float('nan')
        out['eta_mu_separate'] = M0['eta']
        if P0.get('ok'):
            out['eta_mu'] = Z0['eta'] + P0['eta']
            out['eta_mu_half_width'] = math.sqrt(out['half_width'] ** 2 + (out['eta_0_2se'] if out['eta_0_2se'] == out['eta_0_2se'] else 0.0) ** 2)
    return out


# ----------------------------------------------------------------------------------------------
# the A_4 reading, the massive box check, the P22 comparison
# ----------------------------------------------------------------------------------------------
def a4_reading(recs):
    pairs = []
    kern = 0.0
    for r in recs:
        if r['mu'] != 0.0 or r['S_inf'] is None:
            continue
        if r['L'] < 4.0 * r['R_hi'] - 1e-9:
            continue
        u = (r['R_vol'] / r['L']) ** 4
        pairs.append(dict(n=r['n'], R=r['R'], L=r['L'], ratio=r['L'] / r['R_vol'], u=u, y=r['S_box'] - r['S_inf'], y_over_u=(r['S_box'] - r['S_inf']) / u))
        if r['inf_alt'] is not None:
            kern = max(kern, abs(float(r['inf_alt'])))
    out = dict(n_pairs=len(pairs), A4_first_route=A4, pairs=pairs)
    if len(pairs) >= 2:
        u = np.array([p['u'] for p in pairs])
        y = np.array([p['y'] for p in pairs])
        X = np.column_stack([u, u ** 1.5])
        beta, se, resid, dof = ols(X, y)
        out.update(b4_pooled=float(beta[0]), b6_pooled=float(beta[1]), se_b4=float(se[0]), se_b6=float(se[1]), dof=int(dof),
                   resid_rms=float(np.sqrt(np.mean(resid ** 2))))
        per = []
        byball = {}
        for p in pairs:
            byball.setdefault(p['n'], []).append(p)
        for n, lst in byball.items():
            if len(lst) >= 3:
                uu = np.array([p['u'] for p in lst])
                yy = np.array([p['y'] for p in lst])
                bb = np.linalg.lstsq(np.column_stack([uu, uu ** 1.5]), yy, rcond=None)[0]
                per.append(dict(n=n, R=lst[0]['R'], b4=float(bb[0]), b6=float(bb[1]), boxes=len(lst)))
        out['per_ball'] = per
        if per:
            b4s = np.array([p['b4'] for p in per])
            out.update(b4_perball_median=float(np.median(b4s)), b4_perball_spread=float(np.max(np.abs(b4s - np.median(b4s)))))
            groups = {}
            for p in per:
                groups.setdefault(p['R'], []).append(p['b4'])
            out['e4_by_radius'] = {str(R): float(np.median(v) - A4) for R, v in sorted(groups.items())}
        umin = float(u.min())
        out['kernel_term'] = kern / umin if umin > 0 else float('nan')
        stated = 2.0 * (out['se_b4'] if dof > 0 else 0.0) + out.get('b4_perball_spread', 0.0) + out['kernel_term']
        out['stated_error'] = stated
        out['deviation_from_A4'] = out['b4_pooled'] - A4
        out['reproduces_A4'] = bool(abs(out['deviation_from_A4']) <= stated)
        out['single_box_by_ratio'] = {}
        for p in pairs:
            key = '%.1f' % p['ratio']
            out['single_box_by_ratio'].setdefault(key, []).append(p['y_over_u'])
        out['single_box_by_ratio'] = {k: dict(count=len(v), median=float(np.median(v)), spread=float(np.max(np.abs(np.array(v) - np.median(v)))))
                                      for k, v in out['single_box_by_ratio'].items()}
    return out


def massive_box_check(recs, mu1):
    rows = []
    for r in recs:
        if r['mu'] != mu1 or r['S_inf'] is None:
            continue
        rows.append(dict(n=r['n'], R=r['R'], L=r['L'], gap=r['L'] - 2.0 * r['R_hi'], y=r['S_box'] - r['S_inf'], kind=r['kind']))
    out = dict(mu=mu1, gamma_arsinh=math.asinh(mu1), rows=rows)
    prod = [abs(x['y']) for x in rows if x['kind'] in ('prod', 'p22')]
    out['production_max_abs'] = max(prod) if prod else None
    out['production_ok'] = (max(prod) < 1e-6) if prod else None
    sel = [x for x in rows if abs(x['y']) > 1e-9]
    if len(sel) >= 2:
        g = np.array([x['gap'] for x in sel])
        ly = np.log(np.array([abs(x['y']) for x in sel]))
        beta, se, resid, dof = ols(np.column_stack([np.ones_like(g), g]), ly)
        out.update(decay_rate=float(-beta[1]), decay_rate_se=float(se[1]), decay_points=len(sel),
                   decay_in_expected_band=bool(math.asinh(mu1) - 3 * (se[1] if dof > 0 else 0.0) <= -beta[1] <= 2 * math.asinh(mu1) + 3 * (se[1] if dof > 0 else 0.0)))
    return out


def p22_comparison(recs, mu1):
    out = {}
    for R, ref in P22_TABLE.items():
        n = R * R
        r0 = [r for r in recs if r['n'] == n and r['mu'] == 0.0 and r['L'] == ref['L']]
        r1 = [r for r in recs if r['n'] == n and r['mu'] == mu1 and r['L'] == ref['L']]
        entry = dict(n=n, L=ref['L'], P22_S0=ref['S0'], P22_S1=ref['S1'])
        if r0:
            entry.update(S0=r0[0]['S_box'], S0_minus_P22=r0[0]['S_box'] - ref['S0'], n_sites=r0[0]['n_sites'])
        if r1:
            entry.update(S1=r1[0]['S_box'], S1_minus_P22=r1[0]['S_box'] - ref['S1'], n_sites=r1[0]['n_sites'])
        if r0 and r1:
            entry['ratio'] = (r1[0]['S_box'] - r0[0]['S_box']) / (4.0 * math.pi * R * R)
            entry['P22_ratio'] = (ref['S1'] - ref['S0']) / (4.0 * math.pi * R * R)
        if R == 28 and (r0 or r1):
            entry['sites_match_P22_91965'] = bool((r0 or r1)[0]['n_sites'] == P22_SITES_28)
        out[str(R)] = entry
    return out


# ----------------------------------------------------------------------------------------------
# the report
# ----------------------------------------------------------------------------------------------
def run_fit(dirs, radii, mu1, target, tol, verbose=True):
    recs = load_records(dirs)
    table = build_table(recs, radii, mu1)
    res = fit_variants(table, radii, mu1)
    res['A4_reading'] = a4_reading(recs)
    res['massive_box_check'] = massive_box_check(recs, mu1)
    res['P22_comparison'] = p22_comparison(recs, mu1)
    res['records'] = len(recs)
    res['radii'] = radii
    res['mu'] = mu1
    res['target'] = target
    res['tol'] = tol
    fb = [abs(r['fullblock_minus']) for r in recs if r['fullblock_minus'] is not None]
    res['fullblock_check_max'] = max(fb) if fb else None
    rows_tab = []
    for R in radii:
        b = table[R]['balls']
        S0 = [x['S0_corr'] for x in b.values() if x['S0_corr'] is not None]
        S1 = [x['S1_box'] for x in b.values() if x['S1_box'] is not None]
        D = [x['S1_box'] - x['S0_corr'] for x in b.values() if x['S0_corr'] is not None and x['S1_box'] is not None]
        r2 = float(np.mean([x['R_eff'] ** 2 for x in b.values()]))
        rows_tab.append(dict(R=R, n0=len(S0), n1=len(S1), S0_mean=(registry_mean(S0) if S0 else None), S1_mean=(registry_mean(S1) if S1 else None),
                             D_mean=(registry_mean(D) if D else None), D_rms_balls=(float(np.std(D)) if D else None),
                             raw_ratio=(registry_mean(D) / (4 * math.pi * r2) if D else None),
                             S1_over_area=(registry_mean(S1) / (4 * math.pi * r2) if S1 else None)))
    res['per_radius_table'] = rows_tab
    if 'delta_eta' in res:
        res['accepted'] = bool(abs(res['delta_eta'] - target) <= tol)
        res['miss'] = res['delta_eta'] - target
    else:
        res['accepted'] = False
    if verbose:
        print_report(res)
    return res


def print_report(res):
    print('J4 massive fit (mu = %g): %d records, radii %s' % (res['mu'], res['records'], res['radii']))
    print('per radius: R  n0 n1  S_0(corr)  S_0.2  D  D_rms(balls)  D/(4pi<r^2>)  S_0.2/(4pi<r^2>)')
    for t in res['per_radius_table']:
        f = lambda v, fmt: (fmt % v) if v is not None else 'n/a'   # noqa: E731
        print('  %2d  %d  %d  %s  %s  %s  %s  %s  %s' % (t['R'], t['n0'], t['n1'], f(t['S0_mean'], '%.6f'), f(t['S1_mean'], '%.6f'),
                                                    f(t['D_mean'], '%.6f'), f(t['D_rms_balls'], '%.4f'), f(t['raw_ratio'], '%.6f'), f(t['S1_over_area'], '%.6f')))
    print('variants (eta or Delta_eta, se, residual rms, radii):')
    for k, v in res['variants'].items():
        if v.get('ok'):
            print('  %-3s %-52s %+.6f  se %.2e  rms %.3f  (%d radii)' % (k, v['desc'], v['eta'], v['se_eta'], v['rms'], len(v['radii'])))
        else:
            print('  %-3s %-52s not fitted: %s' % (k, v['desc'], v.get('reason')))
    if 'delta_eta' in res:
        print('Delta_eta (P0) = %+.6f  half-width %.2e  (2 se %.2e, envelope %.2e); log sensitivity (44/45)|d/dc_log| = %.2e; '
              'P0 residual rms %.3f nats' % (res['delta_eta'], res['half_width'], res['two_se'], res['env'], res['log_sensitivity_term'], res['residual_rms']))
        if 'separate_difference' in res:
            print('separate fits: eta(0) = %.6f (2 se %.2e), eta(mu) = %.6f, difference %+.6f (2 se %.2e); eta(mu) = eta(0) + Delta_eta = %.6f +- %.2e'
                  % (res['eta_0'], res['eta_0_2se'], res['eta_mu_separate'], res['separate_difference'], res['separate_difference_2se'],
                     res['eta_mu'], res['eta_mu_half_width']))
        print('ACCEPTANCE: |Delta_eta - (%+.4f)| = %.2e %s %.3f -> %s' % (res['target'], abs(res['miss']), '<=' if res['accepted'] else '>', res['tol'],
                                                                           'PASS' if res['accepted'] else 'FAIL (attribution route, spec 3.4)'))
    a = res['A4_reading']
    if 'b4_pooled' in a:
        print('A_4 second route: b4 = %.4f +- %.4f (pooled, %d pairs), b6 = %.2f; per-ball median %s spread %s; kernel term %.1e; '
              'deviation from A4 = %.4f (A4 %.4f); stated error %.4f -> %s'
              % (a['b4_pooled'], a['se_b4'], a['n_pairs'], a['b6_pooled'], ('%.4f' % a['b4_perball_median']) if 'b4_perball_median' in a else 'n/a',
                 ('%.4f' % a['b4_perball_spread']) if 'b4_perball_spread' in a else 'n/a', a['kernel_term'], a['deviation_from_A4'], A4, a['stated_error'],
                 'reproduces A_4 within the stated error' if a['reproduces_A4'] else 'the deviation is the finite-R lattice remainder e_4 (by radius: %s)' % a.get('e4_by_radius')))
        print('  single-box readings (S_L - S_inf)/u by L/R_vol: %s' % {k: '%.3f +- %.3f (%d)' % (v['median'], v['spread'], v['count']) for k, v in sorted(a['single_box_by_ratio'].items())})
    else:
        print('A_4 second route: not enough massless pairs with both routes (%d)' % a['n_pairs'])
    m = res['massive_box_check']
    print('massive box check: production max |S_L - S_inf| = %s (%s); decay rate %s (arsinh mu = %.4f, expected band [%.3f, %.3f])'
          % (('%.2e' % m['production_max_abs']) if m['production_max_abs'] is not None else 'n/a',
             'OK' if m.get('production_ok') else 'not OK or no data',
             ('%.3f +- %.3f from %d points' % (m['decay_rate'], m['decay_rate_se'], m['decay_points'])) if 'decay_rate' in m else 'n/a',
             m['gamma_arsinh'], m['gamma_arsinh'], 2 * m['gamma_arsinh']))
    print('P22 comparison (same balls n = R^2, same boxes; P22 is a comparison, not a result):')
    for R, e in res['P22_comparison'].items():
        print('  R %s n %d L %d: S0 - P22 = %s, S0.2 - P22 = %s, ratio %s (P22 %s)'
              % (R, e['n'], e['L'], ('%.2e' % e['S0_minus_P22']) if 'S0_minus_P22' in e else 'n/a', ('%.2e' % e['S1_minus_P22']) if 'S1_minus_P22' in e else 'n/a',
                 ('%.6f' % e['ratio']) if 'ratio' in e else 'n/a', ('%.6f' % e['P22_ratio']) if 'P22_ratio' in e else 'n/a'))
    for R, v in P22_GROUPS.items():
        t = [x for x in res['per_radius_table'] if x['R'] == R]
        if t and t[0]['S1_over_area'] is not None:
            print('  P22 eight-offset group at r %d: S/(4 pi <r^2>) = %.8f, ours %.8f (difference %.2e)' % (R, v, t[0]['S1_over_area'], t[0]['S1_over_area'] - v))
    if res.get('fullblock_check_max') is not None:
        print('massless full-block check: max |S_fullblock - S_sublattice| = %.2e over the recorded cases' % res['fullblock_check_max'])


def write_outputs(res, out_json, out_tsv):
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        tmp = out_json + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(res, fh, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
        os.replace(tmp, out_json)
    if out_tsv:
        os.makedirs(os.path.dirname(os.path.abspath(out_tsv)), exist_ok=True)
        with open(out_tsv, 'w') as fh:
            fh.write('R\tn0\tn1\tS0_corr_mean\tS1_mean\tD_mean\tD_rms_balls\traw_ratio\tS1_over_area\n')
            for t in res['per_radius_table']:
                fh.write('\t'.join(str(t[k]) for k in ('R', 'n0', 'n1', 'S0_mean', 'S1_mean', 'D_mean', 'D_rms_balls', 'raw_ratio', 'S1_over_area')) + '\n')


# ----------------------------------------------------------------------------------------------
# self-test on synthetic records
# ----------------------------------------------------------------------------------------------
def make_synthetic(outdir, amp, seed=1, eta0=0.9191, eta1=0.8874, mu1=0.2, radii=None):
    import j4lib_v2 as jl2
    import j4m_driver_v1 as jdrv
    radii = radii or RADII_DEFAULT
    rng = np.random.default_rng(seed)
    plan = jdrv.build_plan(radii, [0.0, mu1])
    c1, c0 = 0.31, 4.2
    A, B, D = -1.37, 6.8, 25.0
    b6 = -14.0
    gamma = math.asinh(mu1)
    osc = {}
    os.makedirs(outdir, exist_ok=True)
    for c in plan:
        n = c['n']
        if n not in osc:
            Re = (c['R'] + c['k'] / 16.0) if c['k'] is not None else math.sqrt(n)
            o0 = amp * math.sqrt(Re) * rng.standard_normal()
            o1 = 0.97 * o0 + 0.1 * amp * math.sqrt(Re) * rng.standard_normal()
            osc[n] = (Re, o0, o1)
        Re, o0, o1 = osc[n]
        meta, _sites = jl2.ball_radius_data(n)
        u = (meta['R_vol'] / c['L']) ** 4
        if c['mu'] == 0.0:
            smooth = 4 * math.pi * eta0 * Re ** 2 + c1 * Re + CLOG * math.log(Re) + c0
            S_inf = smooth + o0
            S_box = S_inf + A4 * u + b6 * u ** 1.5
        else:
            smooth = 4 * math.pi * eta1 * Re ** 2 + A * Re + B + D / Re ** 2
            S_inf = smooth + o1
            S_box = S_inf + 1e-3 * math.exp(-gamma * (c['L'] - 2.0 * meta['R_hi']))
        rec = dict(version=1, id=c['id'], kind=c['kind'], n2max=n, L=c['L'], mu=c['mu'], window_centre=c['R'], offset_k=c['k'],
                   n=meta['n'], R_vol=meta['R_vol'], R_lo=meta['R_lo'], R_hi=meta['R_hi'],
                   routes=dict(box=dict(S=S_box), inf=dict(S=S_inf)), checks=dict(inf_alt_minus_inf=1e-11))
        with open(os.path.join(outdir, c['id'] + '.json'), 'w') as fh:
            json.dump(rec, fh)
    return dict(eta0=eta0, eta1=eta1, delta=eta1 - eta0, b6=b6, gamma=gamma)


def selftest(verbose=False):
    """Exact recovery on smooth synthetic records (amp 0) and the oscillation case (amp 3); returns a dict of checks."""
    checks = {}
    with tempfile.TemporaryDirectory() as tmp:
        truth = make_synthetic(os.path.join(tmp, 'smooth'), amp=0.0)
        res = run_fit([os.path.join(tmp, 'smooth')], RADII_DEFAULT, 0.2, truth['delta'], TOL_DELTA, verbose=verbose)
        checks['smooth_delta_eta_1e-7'] = abs(res['delta_eta'] - truth['delta']) < 1e-7
        checks['smooth_eta0_1e-7'] = abs(res['eta_0'] - truth['eta0']) < 1e-7
        checks['smooth_eta_mu_separate_1e-7'] = abs(res['eta_mu_separate'] - truth['eta1']) < 1e-7
        checks['smooth_A4_pooled_1e-6'] = abs(res['A4_reading']['b4_pooled'] - A4) < 1e-6
        checks['smooth_b6_1e-3'] = abs(res['A4_reading']['b6_pooled'] - truth['b6']) < 1e-3
        checks['smooth_accepted'] = bool(res['accepted'])
        checks['smooth_decay_rate'] = abs(res['massive_box_check']['decay_rate'] - truth['gamma']) < 0.01
        truth2 = make_synthetic(os.path.join(tmp, 'osc'), amp=3.0, seed=7)
        res2 = run_fit([os.path.join(tmp, 'osc')], RADII_DEFAULT, 0.2, truth2['delta'], TOL_DELTA, verbose=verbose)
        checks['osc_delta_eta_within_2e-3'] = abs(res2['delta_eta'] - truth2['delta']) < 2e-3
        checks['osc_accepted'] = bool(res2['accepted'])
        checks['osc_half_width_positive'] = res2['half_width'] > 0
        checks['osc_error_smooth_case'] = abs(res['delta_eta'] - truth['delta'])
        checks['osc_error_osc_case'] = abs(res2['delta_eta'] - truth2['delta'])
        checks['osc_half_width'] = res2['half_width']
    return checks


def main():
    ap = argparse.ArgumentParser(description='J4 massive fit (reads records only)')
    ap.add_argument('--records', nargs='*', default=[])
    ap.add_argument('--radii', type=int, nargs='+', default=RADII_DEFAULT)
    ap.add_argument('--mu', type=float, default=0.2)
    ap.add_argument('--target', type=float, default=TARGET_DELTA)
    ap.add_argument('--tol', type=float, default=TOL_DELTA)
    ap.add_argument('--out-json', default=None)
    ap.add_argument('--out-tsv', default=None)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        ch = selftest(verbose=False)
        bad = [k for k, v in ch.items() if isinstance(v, bool) and not v]
        for k, v in ch.items():
            print('  selftest %-32s %s' % (k, v))
        print('selftest: %s' % ('PASS' if not bad else 'FAIL ' + str(bad)))
        sys.exit(0 if not bad else 1)
    if not args.records:
        ap.error('give --records DIR (or --selftest)')
    res = run_fit(args.records, args.radii, args.mu, args.target, args.tol)
    write_outputs(res, args.out_json, args.out_tsv)


if __name__ == '__main__':
    main()
