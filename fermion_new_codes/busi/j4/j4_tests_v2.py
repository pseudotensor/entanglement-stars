#!/usr/bin/env python3
"""
j4_tests_v2.py -- acceptance tests of the version-2 additions (spec busi/specs/J4_spec_v2.md, section 7).
The version-1 tests (j4_tests.py, 21 PASS) remain the gate of the single-ball code; these test the ball
bookkeeping, the windows and their responses, the cached-kernel computation against version 1 and against
the independent dense 4-component code at an 8R box, the finite-box solver, and the whole fit pipeline on
synthetic data.  Run from the code directory with the box's or the pod's python; all must PASS.
"""
import math
import os
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import j4lib as jl           # noqa: E402
import j4lib_v2 as jl2       # noqa: E402
import j4window_v2 as jw     # noqa: E402
import j4_fit_v2 as jf       # noqa: E402

RESULTS = []


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


def t_sums_of_three_squares():
    bad = [n for n in range(0, 3001) if jw.is_sum_of_three_squares(n) != jw.is_sum_of_three_squares_bruteforce(n)]
    assert not bad, 'rule and brute force disagree at %s' % bad[:10]
    dens = sum(jw.is_sum_of_three_squares(n) for n in range(1, 3001)) / 3000.0
    assert abs(dens - 5.0 / 6.0) < 0.01, dens
    for n in range(0, 500):
        if jw.is_sum_of_three_squares(n):
            assert jw.canonical_ball(n) == n
            assert jw.canonical_ball(jw.next_ball(n) - 1) == n
    return 'rule = brute force for n <= 3000; density %.4f (5/6 = %.4f); canonical and next labels consistent' % (dens, 5.0 / 6.0)


def t_balls():
    prev = -1
    for n in range(0, 301):
        sites = jl.ball_sites(n2max=n)
        mx = int(np.max(np.sum(sites * sites, axis=1)))
        assert mx == jw.canonical_ball(n), (n, mx)
        if jw.is_sum_of_three_squares(n):
            assert len(sites) > prev, n
            prev = len(sites)
        else:
            assert len(sites) == prev, n
    for R in np.linspace(3.0, 30.0, 2001):
        n = jw.ball_at_radius(R)
        lo, hi = jw.ball_interval(n)
        assert lo < R <= hi + 1e-12, (R, n, lo, hi)
    for Rc in (12, 40, 62):
        labels = jw.v1_balls(Rc)
        expect = [jw.canonical_ball(int(math.ceil((Rc + k / 16.0) ** 2)) - 1) for k in jw.V1_OFFSETS]
        assert labels == expect and len(set(labels)) == 8
        for W in (2.0, 3.0):
            lev = jw.ball_levels(Rc, W)
            assert all(lev[n] == 3 for n in labels), (Rc, W, [lev[n] for n in labels])
            assert all(3 <= v <= 8 for v in lev.values())
    return 'max |x|^2 = canonical label for n <= 300; strict balls bracketed by their intervals on 2001 radii; version-1 balls at level 3 for W = 2, 3'


def t_window_weights():
    msgs = []
    for Rc, W, window in ((12, 2.0, 'hann'), (12, 3.0, 'hann'), (62, 2.0, 'hann'), (29, 2.0, 'rect'), (40, 3.0, 'rect')):
        tab = jw.window_table(Rc, W, window)
        assert abs(tab['omega_sum'] - 1.0) < 1e-13, tab['omega_sum']
        direct = jw.window_regressors_direct(Rc, W, window, 400)
        rel = np.max(np.abs(tab['exact'] - direct) / np.abs(direct))
        assert rel < 1e-10, rel
        assert abs(tab['exact'][1] - Rc) < 1e-10
        if window == 'hann':
            m2 = W * W * (1.0 / 12.0 - 1.0 / (2.0 * np.pi ** 2))
        else:
            m2 = W * W / 12.0
        assert abs(tab['exact'][0] / (4.0 * np.pi) - (Rc * Rc + m2)) < 1e-9
        nb = len(tab['n'])
        assert abs(nb - (5.0 / 3.0) * Rc * W) / ((5.0 / 3.0) * Rc * W) < 0.15, nb
        msgs.append('(%d, %g, %s): %d balls, sum omega - 1 = %.1e, regressors vs direct %.1e' % (Rc, W, window, nb, tab['omega_sum'] - 1.0, rel))
    return '; '.join(msgs)


def t_responses():
    def numeric(f, W, window):
        x, wq = jw.gauss_legendre(2000)
        d = 0.5 * W * x
        return float(np.sum(0.5 * W * wq * jw.window_weight(d, W, window) * np.cos(2.0 * np.pi * f * d)))
    freqs = [0.5, 0.5 * math.sqrt(3.0), 1.0, math.sqrt(2.0), math.sqrt(3.0), 2.0, math.sqrt(5.0), 3.0]
    worst = 0.0
    table = []
    for f in freqs:
        for W, window in ((2.0, 'hann'), (2.0, 'rect'), (3.0, 'hann')):
            a = float(jw.window_ft(f, W, window))
            b = numeric(f, W, window)
            worst = max(worst, abs(a - b))
        k = np.array(jw.V1_OFFSETS, dtype=float)
        direct = abs(np.sum(np.exp(2j * np.pi * f * k / 16.0))) / 8.0
        d1 = float(jw.v1_protocol_response(f))
        assert abs(direct - d1) < 1e-12, (f, direct, d1)
        table.append('f %.4f: Hann2 %+.4f rect2 %+.4f Hann3 %+.4f v1 %.4f' % (f, jw.window_ft(f, 2.0, 'hann'), jw.window_ft(f, 2.0, 'rect'), jw.window_ft(f, 3.0, 'hann'), d1))
    assert worst < 1e-9, worst
    assert abs(float(jw.v1_protocol_response(8.0)) - 1.0) < 1e-12
    print('       ' + '\n       '.join(table))
    return 'closed-form responses vs numerical integration to %.1e; version-1 response vs direct sum to 1e-12; response 1 at f = 8' % worst


def t_v2_vs_v1():
    out = []
    for R_eff, n, L in ((3.3, 10, 24), (5.1875, 26, 24)):
        r1 = jl.reduced_ball_entropy(R_eff=R_eff, L=L)
        r2 = jl2.reduced_ball_entropy_v2(n, L)
        assert abs(r1['S'] - r2['S']) <= 1e-12, (r1['S'], r2['S'])
        assert r2['n2max'] == n and r2['n'] == r1['n'] and r2['n_e'] == r1['n_e'] and r2['n_o'] == r1['n_o']
        assert abs(r2['R_lo'] - math.sqrt(n)) < 1e-14 and abs(r2['R_hi'] - math.sqrt(jw.next_ball(n))) < 1e-14
        assert r2['sites_hash'] == r1['sites_hash']
        out.append('ball %d L %d: S_v1 %.12f S_v2 %.12f diff %.1e' % (n, L, r1['S'], r2['S'], r1['S'] - r2['S']))
    return '; '.join(out)


def t_kernel_cache():
    d = tempfile.mkdtemp(prefix='j4kc_')
    try:
        cold = jl2.reduced_ball_entropy_v2(26, 40, cache_dir=d)
        path = jl2.kernel_cache_path(d, 40, cold['M_disp'])
        assert cold['kernel_source'] == 'fft' and os.path.exists(path)
        warm = jl2.reduced_ball_entropy_v2(26, 40, cache_dir=d)
        assert warm['kernel_source'] == 'cache'
        assert warm['S'] == cold['S'], (cold['S'], warm['S'])
        g1 = jl2.kernel_table_cached(40, cold['M_disp'], None)[0]
        with np.load(path) as z:
            g2 = z['gtab']
        assert np.array_equal(g1, g2)
        return 'cold (fft) and warm (cache) runs bitwise equal: S = %.12f; cached table identical to a fresh one (M %d)' % (cold['S'], cold['M_disp'])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_dense_8R():
    import j4_dense as jd
    out = []
    for L in (40, 48):
        r2 = jl2.reduced_ball_entropy_v2(26, L)
        d4 = jd.dense_entropy(n2max=26, L=L, m=0.0)
        diff = r2['S'] - d4['S']
        assert abs(diff) <= 1e-9, diff
        out.append('L %d: reduced %.12f dense4 %.12f diff %.1e' % (L, r2['S'], d4['S'], diff))
    return '; '.join(out)


def t_box_solver():
    radii, wide = [12], 40
    b4, b6 = -32.6, -13.5
    labels = jw.v1_balls(12)
    recs = {}
    ns0 = int(round(4.0 * np.pi / 3.0 * 12.0 ** 3))
    for i, n in enumerate(labels):
        ns = ns0 + i
        R_vol = (3.0 * ns / (4.0 * np.pi)) ** (1.0 / 3.0)
        for L in (48, 72, 96):
            u = (R_vol / L) ** 4
            recs[(n, L)] = dict(n=n, L=L, S=1000.0 + i + b4 * u + b6 * u ** 1.5, n_sites=ns, R_vol=R_vol, file='t', method='t')
    n1 = jw.next_ball(max(labels))
    n2 = jw.next_ball(n1)
    R_vol = (3.0 * ns0 / (4.0 * np.pi)) ** (1.0 / 3.0)
    u96 = (R_vol / 96) ** 4
    recs[(n1, 96)] = dict(n=n1, L=96, S=2000.0 + b4 * u96 + b6 * u96 ** 1.5, n_sites=ns0, R_vol=R_vol, file='t', method='t')
    u0, u1 = (R_vol / 48) ** 4, (R_vol / 72) ** 4
    for L, u in ((48, u0), (72, u1)):
        recs[(n2, L)] = dict(n=n2, L=L, S=3000.0 + b4 * u + b6 * u ** 1.5, n_sites=ns0, R_vol=R_vol, file='t', method='t')
    ext, box = jf.box_extrapolate(recs, radii, wide)
    assert abs(box['b4_ref'] - b4) < 1e-8 and abs(box['b6_ref'] - b6) < 1e-6, (box['b4_ref'], box['b6_ref'])
    for i, n in enumerate(labels):
        assert abs(ext[n]['S_inf'] - (1000.0 + i)) < 1e-8
    assert abs(ext[n1]['S_inf'] - 2000.0) < 1e-8
    err2 = ext[n2]['S_inf'] - 3000.0
    factor = (u1 ** 1.5 - u0 ** 1.5) / (u1 - u0)
    predicted = b6 * (u1 ** 1.5 - u1 * factor)
    assert abs(err2 - predicted) < 1e-8 and abs(err2) < 1e-3, (err2, predicted)
    assert abs(ext[n2]['box_unc'] - abs(predicted)) < 1e-8
    assert abs(factor - 0.0710) < 2e-3, factor
    return ('3-box recovery exact (b4 %.6f, b6 %.6f); 1-box correction exact; 2-box error %.2e = b6 term %.2e; '
            'slope factor for boxes 4R, 6R = %.5f' % (box['b4_ref'], box['b6_ref'], err2, predicted, factor))


def t_fit_pipeline():
    quiet = lambda *a, **k: None  # noqa: E731
    res = jf.selftest(log=quiet)
    A, B, C = res['A'], res['B'], res['C']
    assert abs(A['d_eta']) < 1e-6, A
    assert abs(A['c1']) < 1e-4 and abs(A['d_c0']) < 1e-2, A
    assert abs(A['d_b4']) < 1e-6 and abs(A['d_b6']) < 1e-5, A
    assert abs(B['d_eta_hann']) < 1e-4, B['d_eta_hann']
    return ('A (no oscillation): eta error %.1e, c1 %.1e, c0 error %.1e, b4 error %.1e; '
            'B (oscillation %.2f sqrt R nats per ball): Hann exact window eta error %.2e, half-width %.2e, version-1 estimator %.2e, '
            'rectangular %.2e, leakage term %.2e; C (level 5): %.2e'
            % (A['d_eta'], A['c1'], A['d_c0'], A['d_b4'], B['single_rms_per_sqrtR'], B['d_eta_hann'], B['half_width'],
               B['d_eta_v1'], B['d_eta_rect'], B['env_window'], C['d_eta_level5']))


def main():
    t0 = time.time()
    run('sums of three squares', t_sums_of_three_squares)
    run('balls, intervals, levels', t_balls)
    run('window weights and exact regressors', t_window_weights)
    run('window responses', t_responses)
    run('reduced v2 = reduced v1', t_v2_vs_v1)
    run('kernel cache', t_kernel_cache)
    run('8R box against the dense 4-component code', t_dense_8R)
    run('finite-box solver', t_box_solver)
    run('fit pipeline selftest', t_fit_pipeline)
    print('\nSUMMARY: %d PASS, %d FAIL  (%.0f s)' % (sum(RESULTS), len(RESULTS) - sum(RESULTS), time.time() - t0))
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == '__main__':
    main()
