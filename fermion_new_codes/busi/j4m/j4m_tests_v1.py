#!/usr/bin/env python3
"""
j4m_tests_v1.py -- acceptance tests of the massive J4 rerun (busi lane, task 4; spec section 9).  Each test prints
[PASS] or [FAIL] lines; the last line is 'SUMMARY: <passed> passed, <failed> failed'; exit code 1 if any failed.
Usage: j4m_tests_v1.py [--threads 8] [--skip-slow]
"""
import argparse
import inspect
import json
import math
import os
import sys
import tempfile
import traceback


def _early_threads():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--threads', type=int, default=4)
    a, _ = p.parse_known_args()
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[v] = str(a.threads)


_early_threads()
HERE = os.path.dirname(os.path.abspath(__file__))
J4_DIR = os.path.normpath(os.path.join(HERE, '..', 'j4'))
for _p in (HERE, J4_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np                  # noqa: E402
import j4lib as jl                  # noqa: E402
import j4lib_v2 as jl2              # noqa: E402
import j4window_v2 as jw            # noqa: E402
import j4_dense as jd               # noqa: E402
import j4m_lib_v1 as jm             # noqa: E402
import j4m_kernel_v1 as jk          # noqa: E402
import j4m_case_v1 as jc            # noqa: E402
import j4m_driver_v1 as jdrv        # noqa: E402
import j4m_fit_v1 as jf             # noqa: E402

RESULTS = []


def check(name, cond, detail=''):
    cond = bool(cond)
    RESULTS.append(cond)
    print('[%s] %s%s' % ('PASS' if cond else 'FAIL', name, (': ' + detail) if detail else ''), flush=True)


def guarded(name, fn):
    try:
        fn()
    except Exception as e:
        traceback.print_exc()
        check(name + ' (exception)', False, repr(e))


def dense4(n2max, L, mu):
    return jd.dense_entropy(n2max=n2max, L=L, m=mu)['S']


# ----------------------------------------------------------------------------------------------
def t01_kernel():
    L = 16
    k0 = jl.MasslessKernel(L)
    km = jm.MassiveBoxKernel(L, 0.0)
    check('T01a massive box kernel at mu 0 equals j4lib MasslessKernel', np.abs(km.G1 - k0.G1).max() <= 1e-15, 'max diff %.1e' % np.abs(km.G1 - k0.G1).max())
    params = list(inspect.signature(jl.MasslessKernel.__init__).parameters)
    check('T01b j4lib MasslessKernel carries no mass (parameters %s)' % params, params == ['self', 'L'])
    kk = jm.MassiveBoxKernel(L, 0.2)
    gtab, mtab, diag = kk.tables(6)
    check('T01c FFT kernels: kinetic real part %.1e, mass imaginary part %.1e below 1e-13' % (diag['real_part_max'], diag['imag_part_max']),
          max(diag['real_part_max'], diag['imag_part_max']) < 1e-13)
    check('T01d parity structure junk %.1e below 1e-13' % diag['junk'], diag['junk'] < 1e-13)
    ax = np.arange(-6, 7)
    R = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), axis=-1)
    odd = R % 2
    nodd = odd.sum(-1)
    worst = 0.0
    for i in range(3):
        e = np.zeros(3, dtype=np.int64)
        e[i] = 1
        mask = (nodd == 1) & (odd[..., i] == 1)
        diff = 0.5 * (kk.gm(R - e) - kk.gm(R + e))
        worst = max(worst, float(np.abs(diff[mask] - gtab[mask]).max()))
    check('T01e G_i = (Gm(r - e_i) - Gm(r + e_i))/2 on the box (mu 0.2): %.1e' % worst, worst < 1e-14)
    meta, sites, sym = jm.setup_symmetry(10)
    ytabs = jl.signed_tables(k0.staggered_table(sym.M)[0], sym.M, [int(c) for c in np.unique(sym.rep_class[sym.even])])
    K = sym.assemble_blocks(['T1-'], ytabs)['T1-']
    dat = sym.irrep_data('T1-')
    check('T01f j4lib assembles only even x odd blocks (T1- block %s = (m_e, m_o) = %s)' % (K.shape, (dat['m_e'], dat['m_o'])),
          K.shape == (dat['m_e'], dat['m_o']))


def t02_projector():
    L, mu = 8, 0.2
    ker = jm.MassiveBoxKernel(L, mu)
    ax = np.arange(L)
    sites = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), axis=-1).reshape(-1, 3)
    C = jm.dense_scalar_covariance(sites, ker, mu)
    defect = float(np.abs(C @ C - C).max())
    check('T02a pure projector: C_+^2 = C_+ on the whole box L 8 at mu 0.2 (defect %.1e)' % defect, defect < 1e-12)
    Cm = jm.dense_scalar_covariance(sites, ker, mu, drop_mass=True)
    tr = float(np.trace(Cm - Cm @ Cm))
    k = 2 * np.pi * (np.arange(L) + 0.5) / L
    s2 = np.sin(k) ** 2
    eps2 = mu * mu + s2[:, None, None] + s2[None, :, None] + s2[None, None, :]
    pred = float(np.sum(mu * mu / (4.0 * eps2)))
    check('T02b mixed covariance (mass block dropped) fails: Tr C(1-C) = %.10f against sum_k mu^2/(4 eps^2) = %.10f' % (tr, pred),
          abs(tr - pred) < 1e-10 and pred > 1e-3)
    Cz = jm.dense_scalar_covariance(sites, jm.MassiveBoxKernel(L, 0.0), 0.0)
    dz = float(np.abs(Cz @ Cz - Cz).max())
    check('T02c at mu 0 the kinetic-only covariance is a projector (defect %.1e)' % dz, dz < 1e-12)


def t03_massless_limit():
    for n2, L in ((10, 24), (26, 24)):
        rec = jl.reduced_ball_entropy(n2max=n2, L=L)
        meta, sites, sym = jm.setup_symmetry(n2)
        gtab, mtab, diag = jm.MassiveBoxKernel(L, 0.0).tables(sym.M)
        out, _ = jm.massive_entropy_from_tables(sym, gtab, mtab, 0.0)
        d = abs(out['S'] - rec['S'])
        check('T03 full-block code at mu 0 reproduces j4lib (ball %d, L %d): |dS| = %.1e' % (n2, L, d), d < 1e-11)
        dirr = max(abs(a['field_contribution'] - b['field_contribution']) for a, b in zip(out['irreps'], rec['irreps']))
        check('T03 per-irrep agreement (ball %d): %.1e' % (n2, dirr), dirr < 1e-11)


def t04_dense_massive():
    for n2, L, mu in ((4, 32, 0.2), (26, 48, 0.2), (26, 48, 0.7)):
        Sd4 = dense4(n2, L, mu)
        meta, sites, sym = jm.setup_symmetry(n2)
        ker = jm.MassiveBoxKernel(L, mu)
        gtab, mtab, diag = ker.tables(sym.M)
        out, _ = jm.massive_entropy_from_tables(sym, gtab, mtab, mu)
        Ssc, ev, sdef = jm.dense_scalar_entropy(sites, ker, mu)
        check('T04 reduced massive vs independent dense 4-component (ball %d, L %d, mu %g): %.1e' % (n2, L, mu, abs(out['S'] - Sd4)),
              abs(out['S'] - Sd4) < 1e-10)
        check('T04 dense real-symmetric scalar form vs dense 4-component (ball %d, mu %g): %.1e (symmetry defect %.1e)' % (n2, mu, abs(Ssc - Sd4), sdef),
              abs(Ssc - Sd4) < 1e-10 and sdef < 1e-13)
    Sp = dense4(26, 48, 0.2)
    Sm = dense4(26, 48, -0.2)
    check('T04 mass sign irrelevant: |S(+0.2) - S(-0.2)| = %.1e' % abs(Sp - Sm), abs(Sp - Sm) < 1e-10)


def t05_p22_small_controls():
    S = dense4(4, 32, 0.2)
    check('T05a P22 33-site control at box 32 (comparison): ours %.12f, P22 42.21501983935624, diff %.1e' % (S, S - 42.21501983935624),
          abs(S - 42.21501983935624) < 1e-6)
    sites = jl.ball_sites(n2max=4)
    Smix, _, _ = jm.dense_scalar_entropy(sites, jm.MassiveBoxKernel(32, 0.2), 0.2, drop_mass=True)
    check('T05b the shortcut (mass block dropped) reproduces P22 45.523775636496666 (comparison): diff %.1e' % (Smix - 45.523775636496666),
          abs(Smix - 45.523775636496666) < 1e-6)
    S147 = dense4(10, 48, 0.2)
    check('T05c P22 147-site control at box 48 (comparison): ours %.12f, P22 122.41702376522406, diff %.1e' % (S147, S147 - 122.41702376522406),
          abs(S147 - 122.41702376522406) < 1e-6)


def t06_bessel():
    bk = jk.BesselKernel(0.2)
    p22 = {(0, 0, 0): 0.8852782393609306, (2, 0, 0): 0.10105922532490516, (2, 2, 0): 0.036576764327360116,
           (4, 0, 0): 0.02238576734100658, (2, 2, 2): 0.020198366765255986, (12, 4, 0): 0.0004722478996097805}
    pts = np.array(list(p22))
    vals = bk.gm(pts)
    worst = max(abs(v - p22[tuple(p)]) for p, v in zip(pts, vals))
    check('T06a Bessel kernel at mu 0.2 vs P22 values (comparison): max diff %.1e' % worst, worst < 1e-12)
    g1, m1 = bk.tables(20)
    g2, m2 = jk.BesselKernel(0.2, 96, 12, 8).tables(20)
    d = max(np.abs(g1 - g2).max(), np.abs(m1 - m2).max())
    check('T06b node/octave doubling changes the mu 0.2 tables by %.1e' % d, d < 1e-13)
    b0 = jk.BesselKernel(0.0)
    g1z, m1z = b0.tables(20)
    g2z, m2z = jk.BesselKernel(0.0, 96, 12, 8).tables(20)
    dz = max(np.abs(g1z - g2z).max(), np.abs(m1z - m2z).max())
    check('T06b node/octave doubling changes the mu 0 tables by %.1e' % dz, dz < 1e-13)
    gb, mb, diag = jm.MassiveBoxKernel(128, 0.2).tables(20)
    d2 = max(np.abs(gb - g1).max(), np.abs(mb - m1).max())
    check('T06c Bessel tables vs box L 128 at mu 0.2 (image correction ~1e-14): max diff %.1e' % d2, d2 < 1e-12)
    disps = np.array([(0, 0, 0), (2, 0, 0), (2, 2, 0), (2, 2, 2), (4, 0, 0), (6, 2, 0), (4, 4, 2)])
    vals = {}
    for L in (64, 128, 256):
        vals[L] = jm.MassiveBoxKernel(L, 0.0).gm(disps)
    r1 = (4.0 * vals[128] - vals[64]) / 3.0
    r2 = (4.0 * vals[256] - vals[128]) / 3.0
    rich = (16.0 * r2 - r1) / 15.0
    bz = b0.gm(disps)
    dr = float(np.abs(rich - bz).max())
    check('T06d massless Bessel kernel vs Richardson-extrapolated boxes 64/128/256: max diff %.1e (Richardson residual %.1e)' % (dr, float(np.abs(r2 - rich).max())),
          dr < 1e-8)
    odd = np.array([(1, 0, 0), (1, 2, 0), (3, 1, 0), (2, 2, 1)])
    check('T06e Bessel kernel vanishes at odd displacements', np.all(bk.gm(odd) == 0.0))
    sites = jl.ball_sites(n2max=4)
    tk = jm.TableKernel(g1z[20 - 4:20 + 5, 20 - 4:20 + 5, 20 - 4:20 + 5], m1z[20 - 4:20 + 5, 20 - 4:20 + 5, 20 - 4:20 + 5], 4)
    S_inf, _, _ = jm.dense_scalar_entropy(sites, tk, 0.0)
    Rvol = (3.0 * len(sites) / (4.0 * np.pi)) ** (1.0 / 3.0)
    readings = []
    for L in (32, 48, 64):
        S_L, _, _ = jm.dense_scalar_entropy(sites, jm.MassiveBoxKernel(L, 0.0), 0.0)
        readings.append((S_L - S_inf) / (Rvol / L) ** 4)
    med = float(np.median(readings))
    check('T06f box-free massless entropy: single-box readings (S_L - S_inf)/u at L 32/48/64 = %s (A4 %.3f)' % (['%.3f' % x for x in readings], jf.A4),
          all(-50 < x < -20 for x in readings) and max(abs(x - med) for x in readings) < 0.3 * abs(med))


def t07_certified_33():
    lo, hi = 42.21498427187285066, 42.21498427191595159
    bk = jk.BesselKernel(0.2)
    gtab, mtab = bk.tables(4)
    tk = jm.TableKernel(gtab, mtab, 4)
    sites = jl.ball_sites(n2max=4)
    S, ev, sdef = jm.dense_scalar_entropy(sites, tk, 0.2)
    check('T07a 33-site ball at mu 1/5, infinite lattice: S = %.13f in P22 certified [%.12f, %.12f] (+-1e-9)' % (S, lo, hi), lo - 1e-9 <= S <= hi + 1e-9)
    meta, sites2, sym = jm.setup_symmetry(4)
    out, _ = jm.massive_entropy_from_tables(sym, gtab, mtab, 0.2)
    check('T07b reduced massive code with the infinite tables vs dense scalar: %.1e' % abs(out['S'] - S), abs(out['S'] - S) < 1e-10)


def t08_volume_entropy():
    n2, L, mu = 26, 48, 0.2
    meta, sites, sym = jm.setup_symmetry(n2)
    ker = jm.MassiveBoxKernel(L, mu)
    Sok, _, _ = jm.dense_scalar_entropy(sites, ker, mu)
    Smix, _, _ = jm.dense_scalar_entropy(sites, ker, mu, drop_mass=True)
    per_site = (Smix - Sok) / len(sites)
    check('T08a the shortcut at mu 0.2 adds a volume entropy: %.4f nats/site on %d sites (P22: 0.194 asymptotically)' % (per_site, len(sites)), per_site > 0.05)
    d = jl.dense_scalar_entropy(sites, ker)
    S_svd = 4.0 * d['S_scalar_sublattice']
    check('T08b the sublattice SVD on the massive kinetic kernel equals the mixed covariance entropy: %.1e' % abs(S_svd - Smix), abs(S_svd - Smix) < 1e-9)
    gtab, mtab, diag = ker.tables(sym.M)
    outm, _ = jm.massive_entropy_from_tables(sym, gtab, mtab, mu, drop_mass=True)
    check('T08c reduced code with drop_mass reproduces the mixed entropy (test path only): %.1e' % abs(outm['S'] - Smix), abs(outm['S'] - Smix) < 1e-10)


def t09_cache_keys():
    with tempfile.TemporaryDirectory() as tmp:
        g0, m0, d0, s0 = jm.box_tables_cached(16, 0.0, 4, tmp)
        g2, m2, d2, s2 = jm.box_tables_cached(16, 0.2, 4, tmp)
        check('T09a two masses at one (L, M) give two cache files', len([f for f in os.listdir(tmp) if f.startswith('j4m_box')]) == 2 and s0 == 'fft' and s2 == 'fft')
        check('T09a the massless and massive tables differ (%.1e)' % np.abs(g0 - g2).max(), np.abs(g0 - g2).max() > 1e-4)
        g2b, m2b, d2b, s2b = jm.box_tables_cached(16, 0.2, 4, tmp)
        fresh_g, fresh_m, _ = jm.MassiveBoxKernel(16, 0.2).tables(4)
        check('T09a warm read is served from the cache and equals a fresh computation bitwise', s2b == 'cache' and np.array_equal(g2b, fresh_g) and np.array_equal(m2b, fresh_m) and np.array_equal(g2b, g2))
        path = jm.box_cache_path(tmp, 16, 0.2, jm.cache_M(16, 4))
        np.savez(path, gtab=np.zeros((3, 3, 3)), mtab=np.zeros((3, 3, 3)), L=16, mu=0.2, M=jm.cache_M(16, 4), junk=0.0, real_part_max=0.0, imag_part_max=0.0)
        g2c, m2c, d2c, s2c = jm.box_tables_cached(16, 0.2, 4, tmp)
        check('T09a a mismatched-shape cache file is rejected and recomputed', s2c == 'fft' and np.array_equal(g2c, fresh_g))
        gi0, mi0, a0, i0, si0 = jk.inf_tables_cached(0.0, 4, tmp)
        gi2, mi2, a2, i2, si2 = jk.inf_tables_cached(0.2, 4, tmp)
        gi2b, mi2b, a2b, i2b, si2b = jk.inf_tables_cached(0.2, 4, tmp)
        check('T09b infinite-kernel cache keyed by mu: distinct tables (%.1e), bitwise warm read' % np.abs(gi0 - gi2).max(),
              np.abs(gi0 - gi2).max() > 1e-4 and si2b == 'cache' and np.array_equal(gi2b, gi2) and np.array_equal(a2b[0], a2[0]))
        v2dir = os.path.join(tmp, 'v2')
        t1, j1, r1, src1 = jl2.kernel_table_cached(16, 4, v2dir)
        t2, j2, r2, src2 = jl2.kernel_table_cached(16, 4, v2dir)
        t3, j3, r3, src3 = jl2.kernel_table_cached(16, 5, v2dir)
        check('T09c j4lib_v2 kernel cache: cold fft, warm cache bitwise, a different M not served by the (L, M) key',
              src1 == 'fft' and src2 == 'cache' and np.array_equal(t1, t2) and src3 == 'fft' and t3.shape == (11, 11, 11))
        w1 = jw.window_table(12.0, 2.0)
        w2 = jw.window_table(12.0, 2.0)
        w3 = jw.window_table(12.0, 3.0)
        check('T09d j4window_v2 has no persistent cache: weights recomputed per call and change with W (sum %.12f, %.12f)' % (w1['omega_sum'], w3['omega_sum']),
              np.array_equal(w1['omega'], w2['omega']) and (len(w3['omega']) != len(w1['omega']) or not np.array_equal(w1['omega'], w3['omega'])) and abs(w1['omega_sum'] - 1) < 1e-12
              and all(isinstance(k, int) for k in jw._GL))
        ddir = os.path.join(tmp, 'done')
        os.makedirs(ddir)
        for mu in (0.0, 0.2):
            with open(os.path.join(ddir, 'x_mu%s.json' % jm.mu_tag(mu)), 'w') as fh:
                json.dump(dict(n2max=145, L=192, mu=mu, routes=dict(box=dict(S=1.0 + mu))), fh)
        done = jdrv.scan_done(ddir)
        check('T09e the driver done-key includes the mass: %d keys for one (ball, L) at two masses' % len(done), len(done) == 2 and (145, 192, 0.0) in done and (145, 192, 0.2) in done)
        a = jf.registry_mean([1.0, 2.0, 4.0])
        b = jf.registry_mean([1.0, 2.0, 4.0], [1.0, 1.0, 2.0])
        check('T09f the fit registry mean is a pure function of the weights (equal %.4f, weighted %.4f)' % (a, b), abs(a - 7.0 / 3.0) < 1e-15 and abs(b - 11.0 / 4.0) < 1e-15)


def t10_driver_plan():
    plan = jdrv.build_plan([12], [0.0, 0.2])
    kinds = {}
    for c in plan:
        kinds[c['kind']] = kinds.get(c['kind'], 0) + 1
    ids = [c['id'] for c in plan]
    keys = [(c['n'], c['L'], c['mu']) for c in plan]
    check('T10a plan for R 12: %s (expected prod 16, ladder 24, mbox 24, p22 2)' % kinds,
          kinds == dict(prod=16, ladder=24, mbox=24, p22=2) and len(set(ids)) == len(ids) and len(set(keys)) == len(keys))
    check('T10b production boxes 192 (R <= 24) and 224 (R 26, 28)', jdrv.production_box(24) == 192 and jdrv.production_box(26) == 224 and jdrv.production_box(28) == 224)
    full = jdrv.build_plan(jdrv.RADII_DEFAULT, [0.0, 0.2])
    # busi fix: build_plan drops duplicate cases by key; the R 24 ladder's 8R rung (L 192) is the R 24 production box, so its 8 offsets count once (282, not 290)
    check('T10c the full plan has %d cases (144 production, 88 ladder: the R 24 8R rung is the production box, 40 mbox, 10 P22)' % len(full), len(full) == 282)
    n28 = [c for c in full if c['kind'] == 'p22' and c['R'] == 28][0]['n']
    check('T10d the P22 ball at R 28 (n = 784) has 91965 sites', len(jl.ball_sites(n2max=n28)) == 91965)


def t11_fit_selftest():
    ch = jf.selftest(verbose=False)
    for k, v in ch.items():
        if isinstance(v, bool):
            check('T11 fit selftest: ' + k, v)
        else:
            print('       T11 info %s = %.3e' % (k, v))


def t12_case_runner():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'rec')
        cache = os.path.join(tmp, 'cache')
        status, path = jc.run_case(4, 16, 0.2, out, cache, window_centre=2, offset=0, kind='test', threads=1, kernel_check='on')
        with open(path) as fh:
            rec = json.load(fh)
        Sd = dense4(4, 16, 0.2)
        check('T12a case runner (mu 0.2): record written, S_box = dense 4-component to %.1e, saved spectra rebuild it (%.1e)'
              % (abs(rec['routes']['box']['S'] - Sd), abs(rec['checks']['rebuild_box_minus_record'])),
              status == 'done' and abs(rec['routes']['box']['S'] - Sd) < 1e-10 and os.path.exists(path[:-5] + '_eig.npz'))
        check('T12b case runner: box-free route present, |S_box - S_inf| = %.2e, alternate-quadrature sensitivity %.1e'
              % (abs(rec['checks']['box_minus_inf']), abs(rec['checks']['inf_alt_minus_inf'])), 'inf' in rec['routes'] and abs(rec['checks']['inf_alt_minus_inf']) < 1e-9)
        status2, _ = jc.run_case(4, 16, 0.2, out, cache, window_centre=2, offset=0, kind='test', threads=1)
        check('T12c a complete record is skipped on rerun', status2 == 'skipped')
        status3, path3 = jc.run_case(4, 16, 0.0, out, cache, window_centre=2, offset=0, kind='test', threads=1, fullblock_massless='on', kernel_check='on')
        with open(path3) as fh:
            rec3 = json.load(fh)
        Sd0 = dense4(4, 16, 0.0)
        check('T12d case runner (mu 0): j4lib_v2 route = dense to %.1e, full-block check %.1e, inf route present'
              % (abs(rec3['routes']['box']['S'] - Sd0), abs(rec3['checks']['fullblock_minus_sublattice'])),
              abs(rec3['routes']['box']['S'] - Sd0) < 1e-10 and abs(rec3['checks']['fullblock_minus_sublattice']) < 1e-11 and 'inf' in rec3['routes'])
        z = np.load(path3[:-5] + '_eig.npz')
        check('T12e npz carries the singular values of the massless route and the eigenvalues of the full-block check',
              'box_sv_A1+' in z and 'boxfb_eig_A1+' in z and 'inf_sv_A1+' in z)


def t14_massive_box_approach():
    bk = jk.BesselKernel(0.2)
    gtab, mtab = bk.tables(4)
    tk = jm.TableKernel(gtab, mtab, 4)
    sites = jl.ball_sites(n2max=4)
    S_inf, _, _ = jm.dense_scalar_entropy(sites, tk, 0.2)
    diffs = []
    for L in (16, 24, 32):
        S_L, _, _ = jm.dense_scalar_entropy(sites, jm.MassiveBoxKernel(L, 0.2), 0.2)
        diffs.append(abs(S_L - S_inf))
    ok = diffs[0] > diffs[1] > diffs[2] > 0 and diffs[0] / diffs[1] > 2 and diffs[1] / diffs[2] > 2
    check('T14 massive box term falls exponentially at the 33-site ball: |S_L - S_inf| = %s at L 16, 24, 32' % ['%.2e' % d for d in diffs], ok)


def main():
    ap = argparse.ArgumentParser(description='J4 massive rerun: acceptance tests')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--skip-slow', action='store_true', help='skip T06d (box 256) and T11 (fit selftest)')
    args = ap.parse_args()
    print('j4m tests (numpy %s, threads %d)' % (np.__version__, args.threads), flush=True)
    guarded('T01', t01_kernel)
    guarded('T02', t02_projector)
    guarded('T03', t03_massless_limit)
    guarded('T04', t04_dense_massive)
    guarded('T05', t05_p22_small_controls)
    if not args.skip_slow:
        guarded('T06', t06_bessel)
    guarded('T07', t07_certified_33)
    guarded('T08', t08_volume_entropy)
    guarded('T09', t09_cache_keys)
    guarded('T10', t10_driver_plan)
    if not args.skip_slow:
        guarded('T11', t11_fit_selftest)
    guarded('T12', t12_case_runner)
    guarded('T14', t14_massive_box_approach)
    passed = sum(1 for r in RESULTS if r)
    failed = len(RESULTS) - passed
    print('SUMMARY: %d passed, %d failed' % (passed, failed), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
