#!/usr/bin/env python3
"""
j4_tests_v3c.py -- the acceptance tests of the version-3c additions to the J4 fit (busi lane, task 1; busi/notes/
J4_v2_B4_judgment_v2.md).  The 13 tests of j4_tests_v3b.py cover the unchanged parts (the box treatment, the response
constants, the Cauchy-Schwarz inequality, the spectral window, the joint line fit, the pipeline A1, A2, B, B2, C, the
LP) and are not repeated; run both.  Here:

  0. regression: on identical synthetic records (self-test B) j4_fit_v3c.analyze reproduces j4_fit_v3b.analyze (eta,
     half-width, every term, the P residuals, the LP interval) to 1e-9, and the synthetic generator with its defaults
     reproduces the version-3b records exactly;
  1. the response matrix of a complete Hann W=2 window: a 1/2 line reads 0.5 sqrt(R_c) at the window's phase, an
     f = 1 line reads 0 (the derivative term only), a sqrt2 line reads 0.0083 sqrt(R_c);
  2. the tail response sums (m > 20) are a small positive part of the full sums;
  3. planted lattice lines m <= 20 at the measured size, no smooth odd term: the predicted leakage reproduces the shift
     curves (largest ratio <= 0.3), the deleaked residuals are below 0.3 of the primary's, the deleaked eta is within
     1e-5 of the truth, the deleaked c1 is below max(3 se, 0.003) and the G6 reading is 'aliasing' or 'inconclusive';
  4. the same lines plus a planted linear term c1 = -0.05: E1 recovers it within 3 se, the deleaked I8c1 within
     max(3 se, 0.005), the window-level smooth coefficient within max(3 se, 0.01), and the reading is 'odd term indicated';
  5. A1 (exactly representable data): the pipeline's shift curves equal the construction's predicted representation
     mismatch to 1e-8 at every radius with shift data;
  6. the replacement primary E1's statement: finite, at least its 2 se, with P among its envelope members.
"""
import math
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import j4_fit_v3c as jf   # noqa: E402
import j4box_v3 as jb     # noqa: E402
import j4leak_v3b as jk   # noqa: E402
import j4window_v2 as jw  # noqa: E402

RADII = list(jf.RADII_DEFAULT)
WIDE = 40
QUIET = lambda *a, **k: None  # noqa: E731
_CACHE = {}


def _A4():
    if 'A4' not in _CACHE:
        _CACHE['A4'] = jb.a4_derived(check=True)[0]
    return _CACHE['A4']


def _run(key, **kw):
    if key not in _CACHE:
        recs, truth, par = jf.synthetic_records(RADII, WIDE, A4=_A4(), **kw)
        res = jf.analyze(recs, RADII, WIDE, log=QUIET, sphere=None, A4=_A4())
        _CACHE[key] = (recs, truth, par, res)
    return _CACHE[key]


def t_regression_v3b():
    import j4_fit_v3b as j3b
    recs, truth, par, rc = _run('B', mode='midpoint', osc=3.0, seed=1)
    recs_b, _, _ = j3b.synthetic_records(RADII, WIDE, A4=_A4(), mode='midpoint', osc=3.0, seed=1)
    assert set(recs_b) == set(recs), 'the synthetic generators produce different cases'
    dS = max(abs(recs_b[k]['S'] - recs[k]['S']) for k in recs)
    assert dS <= 1e-12, 'synthetic records differ from version 3b by %.2e' % dS
    rb = j3b.analyze(recs, RADII, WIDE, log=QUIET, sphere=None, A4=_A4())
    diffs = {}
    for key in ('eta', 'half_width', 'two_se', 'env_model', 'env_window', 'env_box', 'dlog', 'sigma1', 'inflate'):
        diffs[key] = abs(float(rb[key]) - float(rc[key]))
    diffs['P resid'] = float(np.max(np.abs(rb['V']['P']['resid'] - rc['V']['P']['resid'])))
    for nm in ('E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'I1', 'I2', 'I3', 'I3b', 'I4', 'I5', 'I6', 'I7'):
        diffs[nm] = abs(rb['V'][nm]['eta'] - rc['V'][nm]['eta'])
    diffs['lp min'] = abs(rb['lp']['min'] - rc['lp']['min'])
    diffs['lp max'] = abs(rb['lp']['max'] - rc['lp']['max'])
    diffs['b'] = max(abs(rb['b'][R] - rc['b'][R]) for R in rb['b'])
    worst = max(diffs, key=diffs.get)
    assert diffs[worst] <= 1e-9, 'version 3c differs from version 3b in %s by %.2e' % (worst, diffs[worst])
    return ('identical synthetic records (max |dS| %.1e); eta %.6f in both, every term, the 15 variant etas, the P residuals, the bounds and the LP interval agree to %.1e (worst: %s)'
            % (dS, rc['eta'], diffs[worst], worst))


def t_response_matrix():
    Rc = 24
    tab = jw.window_table(Rc, 2.0, 'hann')
    ext = {int(n): dict(S_inf=0.0, R_vol=float(Rm), box_unc=0.0) for n, Rm in zip(tab['n'], tab['R_mid'])}
    rows = jf.window_rows(ext, [Rc], 2.0, 'hann')
    assert rows and rows[0]['completeness'] >= jf.COMPLETE
    G = jf.response_matrix(rows, [0.5, 1.0, math.sqrt(2.0)])[0]
    sq = math.sqrt(Rc)
    half_cos, half_sin = G[0], G[1]
    exp_half = 0.5 * sq * math.cos(math.pi * Rc)
    assert abs(half_cos - exp_half) <= 0.03 and abs(half_sin) <= 0.03, 'f = 1/2 reads (%.4f, %.4f) against (%.4f, 0)' % (half_cos, half_sin, exp_half)
    one = math.hypot(G[2], G[3])
    assert one <= 0.02, 'f = 1 reads %.4f (expected the derivative term only, about %.4f)' % (one, (1.0 / 3.0) / (4.0 * math.pi * sq))
    r2 = math.hypot(G[4], G[5])
    exp_r2 = abs(float(jw.window_ft(math.sqrt(2.0), 2.0, 'hann'))) * sq
    assert abs(r2 - exp_r2) <= 0.01, 'sqrt2 reads %.4f against %.4f' % (r2, exp_r2)
    return 'Hann W=2 at R_c %d: a per-ball 1/2 line reads %.4f (expected %.4f, sin part %.4f), an f = 1 line %.4f (derivative term %.4f), a sqrt2 line %.4f (expected %.4f)' % (
        Rc, half_cos, exp_half, half_sin, one, (1.0 / 3.0) / (4.0 * math.pi * sq), r2, exp_r2)


def t_tail_sums():
    s = jk.response_sums(2.0, 'hann')
    t = jf.tail_sums(s, jk.LATTICE_JOINT_MMAX)
    assert 0.0 < t['sum_w2'] < s['sum_w2'] and 0.0 < t['sum_dw2'] < s['sum_dw2']
    assert 1e-7 <= t['sum_w2'] <= 1e-5, 'tail sum of hat w^2 %.2e' % t['sum_w2']
    b12, b62 = jk.cs_bound(1.83, 12, t), jk.cs_bound(1.83, 62, t)
    assert b12 < 0.05 and b62 < 0.1
    return 'm > %d: (sum hat w^2)^1/2 = %.2e of %.2e, (sum hat w\'^2)^1/2 = %.3f of %.3f; tail bound at sigma 1.83: %.3f nats (R 12), %.3f (R 62)' % (
        jk.LATTICE_JOINT_MMAX, math.sqrt(t['sum_w2']), math.sqrt(s['sum_w2']), math.sqrt(t['sum_dw2']), math.sqrt(s['sum_dw2']), b12, b62)


def t_deleak_planted_lines():
    recs, truth, par, res = _run('L20', mode='midpoint', osc=3.0, osc_mmax=20, c1=0.0, seed=1)
    assert res['lines'] is not None and res['lines']['applicable'], 'no line model'
    chk = res['shift_check']
    assert chk, 'no shift check'
    rmax = max(chk[R]['ratio'] for R in chk)
    assert rmax <= 0.3, 'shift-curve ratio %.2f' % rmax
    P, I8, I8c1 = res['V']['P'], res['V']['I8'], res['V']['I8c1']
    assert I8['rms'] <= 0.3 * P['rms'], 'deleaked rms %.4f against P %.4f' % (I8['rms'], P['rms'])
    assert abs(I8['eta'] - par['eta0']) <= 1e-5, 'deleaked eta error %.1e' % (I8['eta'] - par['eta0'])
    c1d, sed = jf.param(I8c1, 'R'), jf.param_se(I8c1, 'R')
    assert abs(c1d) <= max(3.0 * sed, 0.003), 'deleaked c1 %.4f +- %.4f' % (c1d, sed)
    assert res['g6']['status'] in ('aliasing', 'inconclusive'), 'reading %s' % res['g6']['status']
    wl = res['window_level']
    return ('P rms %.4f, deleaked rms %.4f, deleaked eta error %.1e; shift-curve ratios up to %.2f; E1 c1 %+.4f +- %.4f, deleaked c1 %+.4f +- %.4f; window-level lines %s; reading %s'
            % (P['rms'], I8['rms'], I8['eta'] - par['eta0'], rmax, jf.param(res['V']['E1'], 'R'), jf.param_se(res['V']['E1'], 'R'), c1d, sed,
               ('; '.join('%s %.4f+-%.4f' % (nm, a, s) for nm, a, s in zip(wl['names'], wl['amps'], wl['amp_se'])) if wl is not None else 'none'), res['g6']['status']))


def t_planted_c1():
    c1_true = -0.05
    recs, truth, par, res = _run('C1', mode='midpoint', osc=3.0, osc_mmax=20, c1=c1_true, seed=1)
    E1, I8c1, wl = res['V']['E1'], res['V']['I8c1'], res['window_level']
    c1e, see = jf.param(E1, 'R'), jf.param_se(E1, 'R')
    c1d, sed = jf.param(I8c1, 'R'), jf.param_se(I8c1, 'R')
    assert abs(c1e - c1_true) <= 3.0 * see, 'E1 c1 %.4f +- %.4f against %.3f' % (c1e, see, c1_true)
    assert abs(c1d - c1_true) <= max(3.0 * sed, 0.005), 'deleaked c1 %.4f +- %.4f against %.3f' % (c1d, sed, c1_true)
    assert wl is not None and abs(wl['c1'] - c1_true) <= max(3.0 * wl['c1_se'], 0.01), 'window-level c1 %s' % (wl['c1'] if wl else None)
    assert res['g6']['status'] == 'odd term indicated', 'reading %s: %s' % (res['g6']['status'], res['g6']['text'])
    bias = res['V']['P']['eta'] - par['eta0']
    return ('planted c1 %.3f: E1 %+.4f +- %.4f, deleaked %+.4f +- %.4f, window-level %+.4f +- %.4f; reading %s; the premise\'s bias of P %+.1e against E1\'s error %+.1e; half-widths P %.1e, E1 %.1e'
            % (c1_true, c1e, see, c1d, sed, wl['c1'], wl['c1_se'], res['g6']['status'], bias, E1['eta'] - par['eta0'], res['half_width'], res['E1_terms']['half']))


def t_A1_shift_mismatch():
    recs, truth, par, res = _run('A1', mode='weighted', osc=0.0, seed=1)
    beta_true = np.zeros(jf.NREG)
    beta_true[jf.COL['4piR2']], beta_true[jf.COL['lnR']], beta_true[jf.COL['one']] = par['eta0'], jf.C_LOG, par['c0']
    devs, hr = {}, {}
    for Rc, sh in res['shifts'].items():
        pred = jf.predicted_shift_mismatch(truth, Rc, sh['shifts'], beta_true)
        assert pred is not None
        devs[Rc] = float(np.max(np.abs(sh['eps'] - pred)))
        hr[Rc] = sh['half_range']
    assert devs and max(devs.values()) <= 1e-8, 'A1 shift curves deviate from the construction by %.1e' % max(devs.values())
    assert abs(res['eta'] - par['eta0']) <= 1e-9
    return 'A1: the pipeline\'s shift curves equal the construction\'s representation mismatch to %.1e at R %s; the half-ranges %s nats (about 1/R, the 2 edge balls) are the G3 WARNING of A1, inflation %.2f, eta error %.1e' % (
        max(devs.values()), sorted(devs), ' '.join('%.3f' % hr[R] for R in sorted(hr)), res['inflate'], res['eta'] - par['eta0'])


def t_E1_statement():
    recs, truth, par, res = _run('L20', mode='midpoint', osc=3.0, osc_mmax=20, c1=0.0, seed=1)
    t = res['E1_terms']
    assert t is not None and np.isfinite(t['half']) and t['half'] >= t['two_se']
    assert 'P' in t['env_members'] and abs(t['env_members']['P'] - (res['eta'] - res['V']['E1']['eta'])) <= 1e-12
    assert abs(t['half'] ** 2 - (t['two_se'] ** 2 + t['env_model'] ** 2 + t['env_window'] ** 2 + t['env_box'] ** 2 + t['dlog'] ** 2)) <= 1e-18
    assert t['env_window'] > res['env_window'], 'E1 leakage term %.1e not above P\'s %.1e' % (t['env_window'], res['env_window'])
    return 'E1 statement: half-width %.1e (2 se %.1e, envelope %.1e from %s, leakage %.1e with sum |w_i| %.2e, log %.1e) against P\'s %.1e: a factor %.1f' % (
        t['half'], t['two_se'], t['env_model'], t['env_model_from'], t['env_window'], t['sum_abs_w'], t['dlog'], res['half_width'], t['half'] / res['half_width'])


def main():
    tests = [('regression against j4_fit_v3b on identical synthetic records', t_regression_v3b),
             ('response matrix of a complete Hann W=2 window', t_response_matrix),
             ('tail response sums beyond the fitted lines', t_tail_sums),
             ('deleaking of planted lattice lines (m <= 20): shift curves, residuals, c1, reading', t_deleak_planted_lines),
             ('a planted linear term c1 = -0.05 survives the deleaking and is read as an odd term', t_planted_c1),
             ('A1: the shift curves are the construction\'s representation mismatch', t_A1_shift_mismatch),
             ('the replacement primary E1\'s statement', t_E1_statement)]
    t_all = time.time()
    npass = nfail = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            msg = fn()
            print('[PASS] %s: %s' % (name, msg))
            npass += 1
        except Exception as e:
            print('[FAIL] %s: %s: %s' % (name, type(e).__name__, e))
            traceback.print_exc()
            nfail += 1
        print('      (%s: %.1f s)' % (name, time.time() - t0))
    print('\nSUMMARY: %d PASS, %d FAIL  (%.0f s)' % (npass, nfail, time.time() - t_all))
    sys.exit(1 if nfail else 0)


if __name__ == '__main__':
    main()
