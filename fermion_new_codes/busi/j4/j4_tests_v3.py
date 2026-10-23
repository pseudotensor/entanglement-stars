#!/usr/bin/env python3
"""
j4_tests_v3.py -- acceptance tests of the version-3 code on synthetic data where the answer is known (busi lane,
task 1; busi/notes/J4_v2_B4_v1.md).  The version-1 tests (j4_tests.py, 21 PASS) and the version-2 tests 1 to 8 of
j4_tests_v2.py remain the gate of the single-ball code and the window bookkeeping; the version-2 test 9 (the fit
pipeline) is superseded by the tests here.  numpy and scipy only; no j4lib needed.  All must PASS.
"""
import math
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import j4window_v2 as jw  # noqa: E402
import j4box_v3 as jb     # noqa: E402
import j4leak_v3 as jk    # noqa: E402
import j4_fit_v3 as jf    # noqa: E402

RESULTS = []
_SELF = {}


def run(name, fn):
    t0 = time.time()
    try:
        msg = fn()
        print('[PASS] %s: %s' % (name, msg), flush=True)
        RESULTS.append(True)
    except Exception as e:
        print('[FAIL] %s: %s: %s' % (name, type(e).__name__, e), flush=True)
        traceback.print_exc()
        RESULTS.append(False)
    print('      (%s: %.1f s)' % (name, time.time() - t0), flush=True)


def t_a4():
    for t in (0.7, 1.0, 1.3):
        d = abs(jb.theta4_direct(t) - jb.theta4_modular(t))
        assert d < 1e-13, (t, d)
    A4, z, err = jb.a4_derived(check=True, tol=1e-10)
    ze = [jb.zeta_ap2_ewald(T)[0] for T in (0.6, 1.0, 1.7)]
    assert max(abs(v - z) for v in ze) < 1e-10, (z, ze)
    assert abs(A4 - jb.A4_RECORD) < 1e-10, (A4, jb.A4_RECORD)
    return ('theta representations agree at t = 0.7, 1, 1.3 to 1e-13; Z_AP(2) = %.15f (theta integral, quadrature error %.0e), Ewald T = 0.6/1/1.7 deviations %.1e; '
            'A_4 = 128/15 Z_AP(2) = %.13f (record %.13f); S_inf - S(L) = %.6f, %.6f, %.6f nats at L = 4R, 6R, 8R'
            % (z, err, max(abs(v - z) for v in ze), A4, jb.A4_RECORD, -A4 / 256.0, -A4 / 1296.0, -A4 / 4096.0))


def t_box():
    A4 = jb.A4_RECORD
    e4, b6 = 0.37, -13.5
    labels = jw.v1_balls(12)
    ns0 = int(round(4.0 * np.pi / 3.0 * 12.0 ** 3))

    def rec(L, S, ns):
        R_vol = (3.0 * ns / (4.0 * np.pi)) ** (1.0 / 3.0)
        u = (R_vol / L) ** 4
        return dict(S=S + (A4 + e4) * u + b6 * u ** 1.5, R_vol=R_vol, n_sites=ns)
    byball = {}
    for i, n in enumerate(labels):
        for L in (48, 72, 96):
            byball.setdefault(n, {})[L] = rec(L, 1000.0 + i, ns0 + i)
    n1 = jw.next_ball(max(labels))
    n2 = jw.next_ball(n1)
    byball[n1] = {96: rec(96, 2000.0, ns0)}
    byball[n2] = {48: rec(48, 3000.0, ns0), 96: rec(96, 3000.0, ns0)}
    ext, box = jb.box_treatment(byball, [12], 40, A4)
    assert abs(box['e4_ref'] - e4) < 1e-8 and abs(box['b6_ref'] - b6) < 1e-6, (box['e4_ref'], box['b6_ref'])
    for i, n in enumerate(labels):
        assert abs(ext[n]['S_inf'] - (1000.0 + i)) < 1e-8, ext[n]['S_inf']
        for L, d in ext[n]['delta_by_L'].items():
            u = (ext[n]['R_vol'] / L) ** 4
            assert abs(d - (e4 * u + b6 * u ** 1.5)) < 1e-8, (L, d)
    assert abs(ext[n1]['S_inf'] - 2000.0) < 1e-8 and abs(ext[n2]['S_inf'] - 3000.0) < 1e-8
    assert ext[n1]['box_unc'] < 1e-8 and box['spread4'] < 1e-8
    c = box['corrections']
    assert abs(c['8R']['continuum'] + A4 / 4096.0) < 1e-12
    return ('reference remainder recovered (b4 - A4 = %.6f, b6 = %.6f); S_inf exact for 3-, 2- and 1-box balls; per-ball delta equals the planted remainder; '
            'corrections at 4R/6R/8R: continuum %.4f/%.4f/%.4f, lattice %+.1e/%+.1e/%+.1e nats'
            % (box['e4_ref'], box['b6_ref'], c['4R']['continuum'], c['6R']['continuum'], c['8R']['continuum'], c['4R']['lattice'], c['6R']['lattice'], c['8R']['lattice']))


def t_leak_constants():
    s = jk.response_sums(2.0, 'hann')
    ms = list(s['ms'])
    assert 1.5e-4 < s['sum_w2'] < 1.7e-4, s['sum_w2']
    assert abs(abs(s['w'][ms.index(2)]) - 0.00827) < 2e-4 and abs(abs(s['w'][ms.index(3)]) - 0.00830) < 2e-4
    assert abs(s['w'][ms.index(1)]) < 1e-12 and abs(s['w'][ms.index(4)]) < 1e-12
    assert abs(s['dw'][ms.index(1)] + 1.0 / 3.0) < 1e-5, s['dw'][ms.index(1)]
    assert 0.11 < s['sum_dw2'] < 0.14, s['sum_dw2']
    assert s['tail_w2'] < 1e-8
    b12, b62 = jk.cs_bound(3.3, 12, s), jk.cs_bound(3.3, 62, s)
    assert 0.2 < b12 < 0.3 and 0.4 < b62 < 0.55, (b12, b62)
    return ('Hann W=2: (sum_m hat w^2)^1/2 = %.5f (largest |hat w| %.4f at sqrt %d), hat w\'(1) = %.6f, (sum_m hat w\'^2)^1/2 = %.4f, tail %.1e; '
            'bound at sigma_1 = 3.3: %.3f nats (R 12), %.3f (R 62)' % (math.sqrt(s['sum_w2']), s['max_w'], s['argmax_m'], s['dw'][ms.index(1)], math.sqrt(s['sum_dw2']), s['tail_w2'], b12, b62))


def t_cs_inequality():
    rng = np.random.default_rng(3)
    s = jk.response_sums(2.0, 'hann')
    ms = [m for m in range(1, 65) if jw.is_sum_of_three_squares(m)]
    f = np.sqrt(np.array(ms, dtype=float))
    x, wq = jw.gauss_legendre(600)
    worst, mean_ratio, count = 0.0, 0.0, 0
    for Rc in (12.0, 29.0, 62.0):
        R = Rc + x
        wt = wq * jw.window_weight(x, 2.0, 'hann')
        for _ in range(100):
            a = rng.uniform(0.5, 1.5, len(ms)) / np.array(ms, dtype=float) ** 0.75
            ph = rng.uniform(0.0, 2.0 * np.pi, len(ms))
            oscil = np.sqrt(R) * (np.cos(2.0 * np.pi * f[None, :] * R[:, None] + ph[None, :]) @ a)
            eps = float(wt @ oscil)
            bound = jk.cs_bound(math.sqrt(np.sum(a * a) / 2.0), Rc, s)
            worst = max(worst, abs(eps) / bound)
            mean_ratio += abs(eps) / bound
            count += 1
    assert worst <= 1.0 + 1e-6, worst
    return 'exact window leakage / bound: largest ratio %.3f, mean %.3f over 300 random superpositions at %d lattice frequencies with sqrt R growth, R 12, 29, 62' % (worst, mean_ratio / count, len(ms))


def t_periodogram():
    rng = np.random.default_rng(5)
    R, dl = [], []
    for Rc in jf.RADII_DEFAULT:
        W = 3.0 if Rc <= 40 else 2.0
        for n, lo, hi in jw.balls_in_window(Rc, W):
            R.append(0.5 * (lo + hi))
            dl.append(hi - lo)
    R, dl = np.array(R), np.array(dl)
    lines = [(1.0, 3.0), (math.sqrt(2.0), 1.8), (math.sqrt(3.0), 1.3), (2.0, 1.1), (math.sqrt(5.0), 0.9)]
    e = 0.3 * rng.normal(size=len(R))
    for fq, a in lines:
        e = e + a * np.cos(2.0 * np.pi * fq * R + rng.uniform(0.0, 2.0 * np.pi))
    A = jk.periodogram(R, e, dl, [fq for fq, _ in lines])
    worst = max(abs(A[i] - a) / a for i, (_, a) in enumerate(lines))
    assert worst < 0.1, (A, worst)
    Ah = jk.periodogram(R, e, dl, [fq for _, fq in jk.HALF])
    assert np.max(Ah) < 0.05 * 3.0, Ah
    grid = np.arange(0.3, 4.5, 0.005)
    floor = jk.noise_floor(grid, jk.periodogram(R, e, dl, grid), [fq for fq, _ in lines] + [fq for _, fq in jk.HALF])
    assert floor < 0.1 * 3.0
    return ('%d balls over R %.1f to %.1f (resolution %.3f): planted amplitudes recovered within %.1f percent; sublattice lines at most %.3f (threshold 0.15); noise floor %.3f'
            % (len(R), R.min(), R.max(), 1.0 / (R.max() - R.min()), 100 * worst, float(np.max(Ah)), floor))


def selftest_once():
    if 'res' not in _SELF:
        quiet = lambda *a, **k: None  # noqa: E731
        _SELF['res'] = jf.selftest(log=quiet)
    return _SELF['res']


def t_fit_A1():
    A = selftest_once()['A1']
    assert abs(A['d_eta']) < 1e-9 and abs(A['d_c0']) < 1e-5, A
    assert abs(A['c1_E1']) < 1e-7 and abs(A['d_eta_E1']) < 1e-8, A
    assert abs(A['d_e4']) < 1e-8 and abs(A['d_b6']) < 1e-6 and A['max_box_err'] < 1e-8, A
    assert A['rms'] < 1e-8, A['rms']
    return ('exactly representable data: eta error %.1e, c0 error %.1e, E1 c1 %.1e, box references %.1e/%.1e, max |S_inf - truth| %.1e, rms %.1e nats'
            % (A['d_eta'], A['d_c0'], A['c1_E1'], A['d_e4'], A['d_b6'], A['max_box_err'], A['rms']))


def t_fit_A2():
    A = selftest_once()['A2']
    assert A['max_dev'] < 1e-9, A['max_dev']
    assert abs(A['d_eta']) < 1e-6 and abs(A['d_eta'] - A['d_eta_pred']) < 1e-9, A
    eps = np.array(A['eps_pred'])
    radii = np.array(A['radii'], dtype=float)
    scaled = np.abs(eps) * radii ** 1.5
    return ('the interval-mean construction leaks %+.4f nats at R %d and %+.4f at R %d (|eps| R^1.5 between %.2f and %.2f, the 0.7 R^-3/2 estimate); '
            'the pipeline reproduces the predicted leakage to %.1e; eta error %.1e; the construction\'s c1 is %.1e (the version-2 failure, now stated)'
            % (eps[0], A['radii'][0], eps[-1], A['radii'][-1], scaled.min(), scaled.max(), A['max_dev'], A['d_eta'], A['c1_E1']))


def t_fit_B():
    B = selftest_once()['B']
    assert abs(B['d_eta']) < 1e-4, B['d_eta']
    assert B['covered'] == B['n_radii'], (B['covered'], B['n_radii'])
    assert B['inflate'] <= 1.0 + 1e-9, B['inflate']
    assert B['env_window'] < 5e-4 and B['half_width'] < 5e-4, (B['env_window'], B['half_width'])
    for k in ('1', 'sqrt2', 'sqrt3'):
        assert abs(B['A_meas'][k] - B['A_true'][k]) < 0.15 * B['A_true'][k], (k, B['A_meas'][k], B['A_true'][k])
    assert B['half_ratio'] < 0.05 and not B['half_flag'], B['half_ratio']
    assert abs(B['sigma1'] - B['sigma1_true']) < 0.15 * B['sigma1_true'], (B['sigma1'], B['sigma1_true'])
    return ('oscillation %.2f sqrt R per ball: eta error %.1e (half-width %.1e, env_window %.1e); bound covers the residual at %d of %d radii; shift/CS ratio %.2f; '
            'periodogram A(1) %.2f (true %.2f), A(sqrt2) %.2f (%.2f), A(sqrt3) %.2f (%.2f); sublattice/A(1) %.3f; version-1 estimator error %.1e, rectangular %.1e'
            % (B['sigma1_true'], B['d_eta'], B['half_width'], B['env_window'], B['covered'], B['n_radii'], B['inflate'], B['A_meas']['1'], B['A_true']['1'],
               B['A_meas']['sqrt2'], B['A_true']['sqrt2'], B['A_meas']['sqrt3'], B['A_true']['sqrt3'], B['half_ratio'], B['d_eta_v1'], B['d_eta_rect']))


def t_fit_B2():
    B = selftest_once()['B2']
    assert B['half_flag'], B
    assert 0.15 < B['half_ratio'] < 0.45, B['half_ratio']
    assert B['covered'] >= B['n_radii'] - 1, (B['covered'], B['n_radii'])
    return 'planted sublattice line at 0.3 A(1): detected (ratio %.3f), bound enlarged and covering %d of %d radii; eta error %.1e, half-width %.1e' % (
        B['half_ratio'], B['covered'], B['n_radii'], B['d_eta'], B['half_width'])


def t_fit_C():
    C = selftest_once()['C']
    assert abs(C['d_eta']) < 5e-5, C['d_eta']
    return 'level-5 sampled windows (completeness %.2f to %.2f): eta error %.1e' % (min(C['completeness']), max(C['completeness']), C['d_eta'])


def t_lp():
    D = selftest_once()['D']
    a, b, eta0 = D['A1_lp'], D['B_lp'], D['eta0']
    assert a.get('status') == 0 and a['min'] - 1e-12 <= eta0 <= a['max'] + 1e-12 and a['half_width'] < 1e-8, a
    assert b.get('status') == 0 and b['min'] - 1e-12 <= eta0 <= b['max'] + 1e-12, b
    assert b['half_width'] <= b['influence_bound'] + 1e-12, (b['half_width'], b['influence_bound'])
    return 'A1: interval [%.9f, %.9f] contains the truth, half-width %.1e; B: [%.6f, %.6f] contains the truth, half-width %.2e <= influence bound %.2e' % (
        a['min'], a['max'], a['half_width'], b['min'], b['max'], b['half_width'], b['influence_bound'])


def main():
    t0 = time.time()
    run('A_4 constant (theta integral, Ewald splits, record)', t_a4)
    run('box treatment (A_4 term, ladder remainder, bounds)', t_box)
    run('leakage response constants', t_leak_constants)
    run('Cauchy-Schwarz leakage inequality', t_cs_inequality)
    run('periodogram on the ball sampling', t_periodogram)
    run('fit pipeline A1 (exactly representable)', t_fit_A1)
    run('fit pipeline A2 (step construction, predicted leakage)', t_fit_A2)
    run('fit pipeline B (oscillation, bound, spectrum)', t_fit_B)
    run('fit pipeline B2 (sublattice line detected)', t_fit_B2)
    run('fit pipeline C (partial windows)', t_fit_C)
    run('LP set-membership interval', t_lp)
    print('\nSUMMARY: %d PASS, %d FAIL  (%.0f s)' % (sum(RESULTS), len(RESULTS) - sum(RESULTS), time.time() - t0))
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == '__main__':
    main()
