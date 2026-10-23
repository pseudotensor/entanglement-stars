#!/usr/bin/env python3
"""
j4_tests_v3b.py -- acceptance tests of the version-3b code on synthetic data where the answer is known (busi lane,
task 1; busi/notes/J4_v2_B4_v1.md and the corrections of busi/notes/J4_v2_B4_judgment_v1.md).  The version-1 tests
(j4_tests.py, 21 PASS) and the version-2 tests 1 to 8 remain the gate of the single-ball code and the window
bookkeeping.  What changed from j4_tests_v3.py: the box test's sign convention (delta_by_L = S_3box - S_cont);
the periodogram test reads the joint line fit and demonstrates the single-frequency artifact instead of denying it;
fit test B reads the joint amplitudes; fit test C reads the preview sampling uncertainty; the LP test checks what the
LP guarantees (containment and the influence bound) and adds an exact unit test; 2 tests added (the spectral window,
the window-table consistency).  numpy and scipy only; no j4lib needed.  All 13 must PASS.
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
import j4leak_v3b as jk   # noqa: E402
import j4_fit_v3b as jf   # noqa: E402

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
            # delta_by_L = S_3box - S_cont(L) = -(lattice remainder e4 u + b6 u^1.5): the version-3 test had the sign wrong
            assert abs(d + (e4 * u + b6 * u ** 1.5)) < 1e-8, (L, d, -(e4 * u + b6 * u ** 1.5))
    pg = box['per_group'][12]
    u8 = (ext[labels[0]]['R_vol'] / 96.0) ** 4
    rem8 = e4 * u8 + b6 * u8 ** 1.5
    assert abs(pg['delta_max_median'] + rem8) < 1e-6, (pg['delta_max_median'], -rem8)
    assert abs(ext[n1]['S_inf'] - 2000.0) < 1e-8 and abs(ext[n2]['S_inf'] - 3000.0) < 1e-8
    assert ext[n1]['box_unc'] < 1e-8 and box['spread4'] < 1e-8
    c = box['corrections']
    assert abs(c['8R']['continuum'] + A4 / 4096.0) < 1e-12
    assert abs(c['8R']['lattice'] + (e4 / 4096.0 + b6 / 4096.0 ** 1.5)) < 1e-12
    return ('reference remainder recovered (b4 - A4 = %.6f, b6 = %.6f); S_inf exact for 3-, 2- and 1-box balls; per-ball S_3box - S_cont equals minus the planted remainder '
            '(at 8R: %+.2e nats, i.e. remainder %+.2e); corrections at 4R/6R/8R: continuum %.4f/%.4f/%.4f, lattice %+.1e/%+.1e/%+.1e nats'
            % (box['e4_ref'], box['b6_ref'], pg['delta_max_median'], rem8, c['4R']['continuum'], c['6R']['continuum'], c['8R']['continuum'],
               c['4R']['lattice'], c['6R']['lattice'], c['8R']['lattice']))


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
    r2 = {nm: abs(float(jw.window_ft(f, 2.0, 'hann'))) for nm, f in jk.HALF}
    assert abs(r2['1/2'] - 0.5) < 1e-3 and abs(r2['sqrt2/2'] - 0.217) < 3e-3 and abs(r2['sqrt3/2'] - 0.069) < 2e-3, r2
    try:
        r4 = {nm: abs(float(jw.window_ft(f, 4.0, 'hann'))) for nm, f in jk.HALF}
        r4['1'] = abs(float(jw.window_ft(1.0, 4.0, 'hann')))
        w4_ok = True
    except Exception as e:
        r4, w4_ok = {'error': str(e)}, False
    if w4_ok:
        assert r4['1/2'] < 1e-10 and r4['1'] < 1e-10, r4
        assert abs(r4['sqrt2/2'] - 0.0083) < 6e-4 and abs(r4['sqrt3/2'] - 0.0083) < 6e-4, r4
        w4msg = 'Hann W=4 responses 1/2 %.1e, sqrt2/2 %.4f, sqrt3/2 %.4f, sqrt5/2 %.4f, sqrt11/2 %.4f (the contingency\'s premise)' % (r4['1/2'], r4['sqrt2/2'], r4['sqrt3/2'], r4['sqrt5/2'], r4['sqrt11/2'])
    else:
        w4msg = 'Hann W=4 responses not available (%s): the --w4-from option would need them' % r4['error']
    return ('Hann W=2: (sum_m hat w^2)^1/2 = %.5f (largest |hat w| %.4f at sqrt %d), hat w\'(1) = %.6f, (sum_m hat w\'^2)^1/2 = %.4f, tail %.1e; '
            'bound at sigma_1 = 3.3: %.3f nats (R 12), %.3f (R 62); W=2 sublattice responses 1/2 %.3f, sqrt2/2 %.3f, sqrt3/2 %.3f, sqrt5/2 %.3f, sqrt11/2 %.3f; %s'
            % (math.sqrt(s['sum_w2']), s['max_w'], s['argmax_m'], s['dw'][ms.index(1)], math.sqrt(s['sum_dw2']), s['tail_w2'], b12, b62,
               r2['1/2'], r2['sqrt2/2'], r2['sqrt3/2'], r2['sqrt5/2'], r2['sqrt11/2'], w4msg))


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


def _ball_sampling():
    R, dl = [], []
    for Rc in jf.RADII_DEFAULT:
        W = 3.0 if Rc <= 40 else 2.0
        for n, lo, hi in jw.balls_in_window(Rc, W):
            R.append(0.5 * (lo + hi))
            dl.append(hi - lo)
    return np.array(R), np.array(dl)


def t_spectral_window():
    R, dl = _ball_sampling()
    f_half = [f for _, f in jk.HALF]
    Wd = jk.spectral_window(R, dl, [abs(f - 1.0) for f in f_half])
    e = 3.0 * np.cos(2.0 * np.pi * R + 0.7)
    single = jk.periodogram(R, e, dl, f_half)
    assert np.max(single) > 0.3, single
    ms, f_lat = jk.lattice_frequencies(jk.LATTICE_JOINT_MMAX)
    J = jk.joint_lines(R, e, None, list(f_lat) + f_half)
    n_lat = len(f_lat)
    assert abs(J['amps'][0] - 3.0) < 1e-8, J['amps'][0]
    assert np.max(J['amps'][1:n_lat]) < 1e-8 and np.max(J['amps'][n_lat:]) < 1e-8, (np.max(J['amps'][1:n_lat]), np.max(J['amps'][n_lat:]))
    assert J['cond'] < jf.MAX_COND and J['max_corr'] < jf.MAX_CORR, (J['cond'], J['max_corr'])
    c = [jk.shift_range_factor(f) for f in (0.5, 0.5 * math.sqrt(2.0), 0.5 * math.sqrt(3.0), 1.0, 1.5)]
    assert abs(c[0] - 0.5) < 1e-12 and abs(c[1] - 0.5 * (1 - math.cos(math.pi * 0.5 * math.sqrt(2.0)))) < 1e-12 and c[3] == 1.0 and c[4] == 1.0, c
    return ('%d balls in 10 clusters, baseline %.0f: spectral window |W| at the offsets of the sublattice frequencies from f = 1: %s; a pure f = 1 line of amplitude 3 reads %s at them in the single-frequency periodogram '
            '(the version-3 artifact) and 0 to %.0e in the joint fit of %d lattice + 5 sublattice frequencies (A(1) %.12f; condition number %.1f, largest |correlation| %.2f); shift-range factors 1/2 %.3f, sqrt2/2 %.3f, sqrt3/2 %.3f'
            % (len(R), R.max() - R.min(), ' '.join('%.3f' % v for v in Wd), ' '.join('%.3f' % v for v in single), max(np.max(J['amps'][1:n_lat]), np.max(J['amps'][n_lat:])),
               n_lat, J['amps'][0], J['cond'], J['max_corr'], c[0], c[1], c[2]))


def t_periodogram():
    rng = np.random.default_rng(5)
    R, dl = _ball_sampling()
    lines = [(1.0, 3.0), (math.sqrt(2.0), 1.8), (math.sqrt(3.0), 1.3), (2.0, 1.1), (math.sqrt(5.0), 0.9)]
    e = 0.3 * rng.normal(size=len(R))
    signal = np.zeros(len(R))
    for fq, a in lines:
        signal = signal + a * np.cos(2.0 * np.pi * fq * R + rng.uniform(0.0, 2.0 * np.pi))
    e = e + signal
    A = jk.periodogram(R, e, dl, [fq for fq, _ in lines])
    worst = max(abs(A[i] - a) / a for i, (_, a) in enumerate(lines))
    assert worst < 0.1, (A, worst)
    f_half = [fq for _, fq in jk.HALF]
    Ah_data = jk.periodogram(R, e, dl, f_half)
    Ah_signal = jk.periodogram(R, signal, dl, f_half)
    assert np.max(Ah_data) > 0.15, Ah_data
    assert np.max(np.abs(Ah_data - Ah_signal)) < 0.1, (Ah_data, Ah_signal)
    ms, f_lat = jk.lattice_frequencies(jk.LATTICE_JOINT_MMAX)
    J = jk.joint_lines(R, e, None, list(f_lat) + f_half)
    n_lat = len(f_lat)
    ms_list = [int(m) for m in ms]
    worst_joint, worst_se = 0.0, 0.0
    for fq, a in lines:
        i = ms_list.index(int(round(fq * fq)))
        err = abs(J['amps'][i] - a)
        assert err < max(0.1 * a, 3.0 * J['se'][i]), (ms_list[i], J['amps'][i], a, J['se'][i])
        worst_joint = max(worst_joint, err / a)
        worst_se = max(worst_se, J['se'][i])
    assert np.max(J['amps'][n_lat:]) < 0.15, J['amps'][n_lat:]
    assert J['cond'] < jf.MAX_COND and J['max_corr'] < jf.MAX_CORR, (J['cond'], J['max_corr'])
    grid = np.arange(0.3, 4.5, 0.005)
    floor = jk.noise_floor(grid, jk.periodogram(R, e, dl, grid), [fq for fq, _ in lines] + f_half)
    assert floor < 0.3, floor
    return ('%d balls over R %.1f to %.1f: single-frequency periodogram recovers the 5 planted lines within %.1f percent but reads %s at the sublattice frequencies (the noiseless planted signal alone reads %s: leakage through the sampling window, the version-3 failure); '
            'the joint fit recovers the planted amplitudes within %.1f percent (se up to %.3f) and reads at most %.3f at the sublattice frequencies (threshold 0.15); noise floor %.3f'
            % (len(R), R.min(), R.max(), 100 * worst, ' '.join('%.3f' % v for v in Ah_data), ' '.join('%.3f' % v for v in Ah_signal),
               100 * worst_joint, worst_se, float(np.max(J['amps'][n_lat:])), floor))


def t_window_consistency():
    msgs, worst = [], 0.0
    for Rc, W in ((12, 2.0), (29, 3.0), (62, 2.0)):
        tab = jw.window_table(Rc, W, 'hann')
        om = np.asarray(tab['omega'], dtype=float)
        means = np.asarray(tab['means'], dtype=float)
        exact = np.asarray(tab['exact'], dtype=float)
        assert abs(om.sum() - 1.0) < 1e-10, om.sum()
        d = float(np.max(np.abs(om @ means - exact) / (1.0 + np.abs(exact))))
        assert d < 1e-9, (Rc, W, d)
        worst = max(worst, d)
        msgs.append('R %d W %.0f %.1e' % (Rc, W, d))
    try:
        tab4 = jw.window_table(48, 4.0, 'hann')
        om = np.asarray(tab4['omega'], dtype=float)
        d4 = float(np.max(np.abs(om @ np.asarray(tab4['means'], dtype=float) - np.asarray(tab4['exact'], dtype=float)) / (1.0 + np.abs(np.asarray(tab4['exact'], dtype=float)))))
        w4 = 'W=4 table at R 48: %d balls, weight sum %.6f, consistency %.1e' % (len(om), om.sum(), d4)
    except Exception as e:
        w4 = 'W=4 tables not available (%s: %s); the --w4-from option would need them' % (type(e).__name__, e)
    return 'sum_j omega_j <f>_{w,j} = exact window regressors to %.1e relative (%s): the passed-weight regressors of a preview reduce to the exact ones for a complete window; %s' % (worst, ', '.join(msgs), w4)


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
    assert B['joint_applicable'], ('the joint fit must apply on oscillation data', B['joint_cond'], B['joint_max_corr'])
    for k in ('1', 'sqrt2', 'sqrt3', '2', 'sqrt5'):
        err = abs(B['A_joint'][k] - B['A_true'][k])
        assert err < max(0.15 * B['A_true'][k], 3.0 * B['se_joint'][k]), (k, B['A_joint'][k], B['se_joint'][k], B['A_true'][k])
    assert B['half_ratio_joint'] < 0.1 and not B['half_flag'], (B['half_ratio_joint'], B['half_flag'])
    assert abs(B['sigma1'] - B['sigma1_true']) < 0.15 * B['sigma1_true'], (B['sigma1'], B['sigma1_true'])
    return ('oscillation %.2f sqrt R per ball: eta error %.1e (half-width %.1e, env_window %.1e); bound covers the residual at %d of %d radii; shift/CS ratio %.2f; '
            'joint A(1) %.2f+-%.2f (true %.2f), A(sqrt2) %.2f+-%.2f (%.2f), A(sqrt3) %.2f+-%.2f (%.2f; single-frequency %.2f, the version-3 failure); sublattice/A(1) joint %.3f, single-frequency %.3f (the version-3 artifact); '
            'version-1 estimator error %.1e, rectangular %.1e'
            % (B['sigma1_true'], B['d_eta'], B['half_width'], B['env_window'], B['covered'], B['n_radii'], B['inflate'],
               B['A_joint']['1'], B['se_joint']['1'], B['A_true']['1'], B['A_joint']['sqrt2'], B['se_joint']['sqrt2'], B['A_true']['sqrt2'],
               B['A_joint']['sqrt3'], B['se_joint']['sqrt3'], B['A_true']['sqrt3'], B['A_meas']['sqrt3'], B['half_ratio_joint'], B['half_ratio_periodogram'],
               B['d_eta_v1'], B['d_eta_rect']))


def t_fit_B2():
    B = selftest_once()['B2']
    assert B['half_flag'], B
    assert 'sqrt3/2' in [t[0] for t in B['detected']], B['detected']
    assert 0.2 < B['half_ratio_joint'] < 0.4, B['half_ratio_joint']
    assert B['covered'] >= B['n_radii'] - 1, (B['covered'], B['n_radii'])
    return 'planted sublattice line at 0.3 A(1): detected (joint ratio %.3f; single-frequency %.3f), bound enlarged and covering %d of %d radii; eta error %.1e, half-width %.1e' % (
        B['half_ratio_joint'], B['half_ratio_periodogram'], B['covered'], B['n_radii'], B['d_eta'], B['half_width'])


def t_fit_C():
    C = selftest_once()['C']
    assert 0.0 < C['preview_unc'] < 1e-4, C['preview_unc']
    assert abs(C['d_eta']) <= max(3.0 * C['preview_unc'], 1e-5) and abs(C['d_eta']) < 1.5e-4, (C['d_eta'], C['preview_unc'])
    return 'level-5 sampled windows (completeness %.2f to %.2f, consistent passed-weight regressors): eta error %.1e against the preview sampling uncertainty %.1e (3 times it: %.1e)' % (
        min(C['completeness']), max(C['completeness']), C['d_eta'], C['preview_unc'], 3.0 * C['preview_unc'])


def t_lp():
    A = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    pr = jf.polygon_eta_range(A, np.ones(4))
    assert pr is not None and abs(pr[0] + 1.0) < 1e-12 and abs(pr[1] - 1.0) < 1e-12 and pr[2] == 4, pr
    X = np.array([np.asarray(jw.window_table(Rc, 2.0, 'hann')['exact'], dtype=float) for Rc in jf.RADII_DEFAULT])[:, [jf.COL['4piR2'], jf.COL['one']]]
    beta = np.array([0.919, -5.0])
    y = X @ beta
    r = jf.eta_range(X, y, np.full(len(y), 1e-6))
    assert r['status'] == 0 and r['min'] - 1e-13 <= beta[0] <= r['max'] + 1e-13, r
    assert r['half_width'] <= r['influence_bound'] + 1e-13 and r['influence_bound'] < 1e-10, (r['half_width'], r['influence_bound'])
    if 'method_agreement' in r:
        assert r['method_agreement'] < 1e-9, r['method_agreement']
    D = selftest_once()['D']
    a, b, eta0 = D['A1_lp'], D['B_lp'], D['eta0']
    assert a.get('status') == 0 and a['min'] - 1e-12 <= eta0 <= a['max'] + 1e-12 and a['half_width'] <= a['influence_bound'] + 1e-12, a
    assert b.get('status') == 0 and b['min'] - 1e-12 <= eta0 <= b['max'] + 1e-12 and b['half_width'] <= b['influence_bound'] + 1e-12, b
    return ('known polygon [-1, 1] recovered; exact data with B = 1e-6: interval contains the truth, half-width %.1e <= influence bound %.1e (%s%s); '
            'A1: [%.9f, %.9f] contains the truth, half-width %.1e <= %.1e; B: [%.6f, %.6f] contains the truth, half-width %.2e <= %.2e'
            % (r['half_width'], r['influence_bound'], r['method'], (', HiGHS agrees to %.1e' % r['method_agreement']) if 'method_agreement' in r else (', HiGHS: %s' % r.get('highs_message', 'not run')),
               a['min'], a['max'], a['half_width'], a['influence_bound'], b['min'], b['max'], b['half_width'], b['influence_bound']))


def main():
    t0 = time.time()
    run('A_4 constant (theta integral, Ewald splits, record)', t_a4)
    run('box treatment (A_4 term, ladder remainder, bounds; sign convention S_3box - S_cont)', t_box)
    run('leakage response constants (W=2, W=4)', t_leak_constants)
    run('Cauchy-Schwarz leakage inequality', t_cs_inequality)
    run('spectral window of the ball sampling and the joint line fit', t_spectral_window)
    run('periodogram on the ball sampling (artifact demonstrated) and joint recovery of planted lines', t_periodogram)
    run('window-table consistency (passed-weight regressors)', t_window_consistency)
    run('fit pipeline A1 (exactly representable)', t_fit_A1)
    run('fit pipeline A2 (step construction, predicted leakage)', t_fit_A2)
    run('fit pipeline B (oscillation, bound, joint spectrum)', t_fit_B)
    run('fit pipeline B2 (sublattice line detected by the joint fit)', t_fit_B2)
    run('fit pipeline C (partial windows, preview sampling uncertainty)', t_fit_C)
    run('LP set-membership interval (exact polygon, containment, influence bound)', t_lp)
    print('\nSUMMARY: %d PASS, %d FAIL  (%.0f s)' % (sum(RESULTS), len(RESULTS) - sum(RESULTS), time.time() - t0))
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == '__main__':
    main()
