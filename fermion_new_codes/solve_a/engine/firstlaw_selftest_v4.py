#!/usr/bin/env python3
"""
firstlaw_selftest_v4.py -- the version-3 columns of firstlaw_records_v3.py on a manufactured static pair (two records of the same
sources, so delta E = delta A = W_ext = 0 exactly), with firstlaw_records_v2.py imported beside it for the nothing-else-changed
comparison and for the end-to-end seed.  The checks of the three columns stand in firstlaw_selftest_v3.py on the unchanged
version-2 code; none is repeated here.

Checks (PASS/FAIL per line; exit 1 on any FAIL; a SKIP counts as a failure unless --allow-skip):
  S0  the engine imports read-only from --engine-path; both records modules import
  S1  the static pair: the pair is included, its three columns and residual are exactly 0, the version-3 pair columns are the
      record's values, the midpoint fields' T_ll at R_mid agrees with T_ll(R)
  S2  T_ll at the root: with the manufactured stress P = pf eps_tot the smoothed stress is pf rho_ell on the dense grid (the scalar
      kernel is linear and the records' sources are this kernel), T_ll(R) is assembled from the exact kernel values (expected
      rho_R (1 + pf) - 2 J_R), the other null direction's form differs by 4 J_R, the dense profile equals rho + p - 2J pointwise, the
      samples at the offsets and the window agree with the profile at R, the quadrature int 4 pi r^2 p_ell dr returns pf sum eps_tot
  S3  the row defect at every dense zero (two: inner in the core, outer on the plateau): |e| and |g| relative to the rows' terms are
      within --tol-rows at both; e(R) read from kappa_fd - kappa_formula agrees with e(R) from the arrays; the kappa correction is
      within --tol-rows of kappa
  S4  the response to a manufactured defect: rho -> rho (1 + delta) moves e(z) by -4 pi z^2 delta rho(z) at every zero and leaves g;
      J -> J (1 + delta) moves g(z) by 4 pi G z delta J(z)/u(z) and e(z) by -4 pi z^2 u(z) delta J(z)
  S5  nothing else changed: firstlaw_records_v2.analyse and firstlaw_records_v3.analyse on the same inputs agree on every leaf of
      records, pairs, excluded_pairs, stats, grid and matching
  S6  the stress key discovery (none / by name / --pr-key / the pairs' part excluded) and the loader round trip through the npz layout
  S7  end to end: firstlaw_records_v2.main writes the version-2 outputs and dense cache for the static pair; firstlaw_records_v3.main
      with --seed-cache (no re-solve) and --check-against reports 0 differing leaves; the version-3 json carries the new columns, the
      npz the T_ll profile, the md the version-3 section

CPU only; three engine solves at --n-sites-engine sites (one in-process, two in version 2's main) and two kernel smoothings.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import firstlaw_records_v3 as fl  # noqa: E402  (the version-3 script under test)
import firstlaw_records_v2 as fl2  # noqa: E402  (version 2 beside it)

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

FOUR_PI = 4.0 * np.pi
NAN = float('nan')

# the broad profile of firstlaw_selftest_v3 at lam = 0: a compact core that traps (inner root near r = 20) and a plateau on which
# h = 2 g M_eff - r crosses zero near r = 42 (outer root, simple margin about 0.55); the current J_tot = jf eps_tot
CORE_R0, CORE_W = 20.0, 6.0
E_CORE0 = 80.0
RISE, FALL = (24.0, 32.0), (70.0, 90.0)
A_PLATEAU0 = 1.0
JF0 = 0.10
CENTRE_E, CENTRE_W = 0.02, 12.0


def smoothstep5(x):
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    return x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)


def bump(x):
    x = np.asarray(x, float)
    return np.where(np.abs(x) < 1.0, (1.0 - x * x) ** 4, 0.0)


def plateau(R):
    R = np.asarray(R, float)
    P = np.ones_like(R)
    P = np.where(R <= RISE[0], 0.0, P)
    m = (R > RISE[0]) & (R < RISE[1])
    P = np.where(m, smoothstep5((R - RISE[0]) / (RISE[1] - RISE[0])), P)
    m = (R > FALL[0]) & (R < FALL[1])
    P = np.where(m, 1.0 - smoothstep5((R - FALL[0]) / (FALL[1] - FALL[0])), P)
    P = np.where(R >= FALL[1], 0.0, P)
    return P


def broad_sources(R):
    R = np.asarray(R, float)
    wc = R * R * bump((R - CORE_R0) / CORE_W)
    eps = E_CORE0 * wc / wc.sum() + A_PLATEAU0 * plateau(R) + CENTRE_E * bump(R / CENTRE_W)
    return eps, JF0 * eps


def make_record(lam, E, zeros, kappa_T, N_t, r_collar, extra=None):
    rec = {'T': lam, 'T_over_rs': lam, 'E': float(E), 'r_collar': float(r_collar),
           'trapping_zeros': [float(z) for z in zeros], 'tangencies': [], 'kappa_T': kappa_T,
           'surface_jets': {'N_t': N_t}, 'converged': True,
           'regime': {'label': 'in the regime', 'intervals_above_1': []}, 'gauge': {'label': 'intact'},
           'late_surface': {'flag': False}, 'interp_floor_N': 1e-12, 'interp_floor_v': 1e-12,
           'indicator': {'collar_screen': {'delta_E_signed': 0.0}, 'collar_label': 'passes'},
           'solver': {'W_tail': {'E_tail_end': float(E), 'E_collar': float(E)}}}
    if extra:
        rec.update(extra)
    return rec


def record_from_dense(d, lam):
    eng = d.get('engine') or {}
    root = d.get('root') or {}
    E = d['E_collar']
    extra = {'regime': {'label': (eng.get('regime') or {}).get('label') or 'in the regime',
                        'intervals_above_1': (eng.get('regime') or {}).get('intervals_above_1') or []},
             'gauge': {'label': (eng.get('gauge') or {}).get('label') or 'intact'},
             'interp_floor_N': eng.get('interp_floor_N'), 'interp_floor_v': eng.get('interp_floor_v'),
             'solver': {'W_tail': eng.get('W_tail') or {'E_tail_end': E, 'E_collar': E}, 'module': eng.get('module'),
                        'rtol': eng.get('rtol'), 'atol': eng.get('atol')}}
    return make_record(lam, E, d['zeros_dense'], root.get('kappa_eq'), root.get('N_R'), d['r_collar'], extra=extra)


def save_layout(out_dir, name, records, F, extra_arrays=None):
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, 'collapse_%s.json' % name)
    npath = os.path.join(out_dir, 'collapse_%s_fields.npz' % name)
    with open(jpath, 'w') as fh:
        json.dump({'args': {'g': F['G'], 'rs': 1.0, 'stop_E': 1e-12, 'tag': name, 'ell': F['ell']}, 'passes': fl._jsonable(records)}, fh)
    arrays = dict(R=F['R'], T=F['T'], eps_tot=F['eps_tot'], J_tot=F['J_tot'], g=np.array(F['G']), ell=np.array(int(round(F['ell']))))
    if F.get('N_site') is not None:
        arrays['N'] = F['N_site']
        arrays['v'] = F['v_site']
    for key, val in (extra_arrays or {}).items():
        arrays[key] = val
    np.savez(npath, **arrays)
    return jpath, npath


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='manufactured checks of the version-3 columns of firstlaw_records_v3 on a static pair')
    p.add_argument('--out', default='<internal>/firstlaw_selftest_v4')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--workers', type=int, default=1, help='read at import by the records modules; the mains here run with --workers 1')
    p.add_argument('--engine-path', dest='engine_path', default=None, help="the engine's directory (default: this script's directory)")
    p.add_argument('--engine-module', dest='engine_module', default='engine_fields_v16')
    p.add_argument('--skip-engine', dest='skip_engine', action='store_true')
    p.add_argument('--allow-skip', dest='allow_skip', action='store_true')
    p.add_argument('--G', type=float, default=0.25)
    p.add_argument('--ell', type=float, default=2.0)
    p.add_argument('--fine', type=int, default=16)
    p.add_argument('--n-sites-engine', dest='n_sites_engine', type=int, default=320)
    p.add_argument('--rtol', type=float, default=1e-12)
    p.add_argument('--atol', type=float, default=1e-16)
    p.add_argument('--support-rel', dest='support_rel', type=float, default=1e-14)
    p.add_argument('--pf', type=float, default=0.3, help='the manufactured stress is P = pf eps_tot per site')
    p.add_argument('--delta', type=float, default=1e-3, help='the relative size of the manufactured defect of S4')
    p.add_argument('--tol-rows', dest='tol_rows', type=float, default=1e-6, help='the row defects at the zeros relative to the rows terms (dense arrays, 4th-order stencil at ell/fine)')
    p.add_argument('--tol-linear', dest='tol_linear', type=float, default=1e-9, help='|p_ell/rho_ell - pf| over the dense grid where rho_ell is not negligible')
    p.add_argument('--tol-source-consistency', dest='tol_source_consistency', type=float, default=1e-9, help="|SmoothedSources5(eps, J).rho - the solve's rho_ell| / max |rho_ell|")
    p.add_argument('--tol-assembly', dest='tol_assembly', type=float, default=1e-10, help='|T_ll(R) - (rho_R (1 + pf) - 2 J_R)| / |rho_R|')
    p.add_argument('--tol-interp', dest='tol_interp', type=float, default=1e-5, help='the interpolated T_ll (midpoint fields at R_mid, the offset 0 sample, the window centre) against T_ll(R), relative to max(|T_ll|, |rho_R|)')
    p.add_argument('--tol-quadrature', dest='tol_quadrature', type=float, default=1e-6, help='|int 4 pi r^2 p_ell dr / sum P - 1|')
    p.add_argument('--tol-response', dest='tol_response', type=float, default=1e-6, help='relative error of the predicted response of e, g to the manufactured defect')
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    results = []

    def check(name, ok, detail, skipped=False):
        status = 'SKIP' if skipped else ('PASS' if ok else 'FAIL')
        results.append({'name': name, 'pass': bool(ok) and not skipped, 'skipped': bool(skipped), 'detail': detail})
        print('%s  %s  %s' % (status, name, detail))

    names = ['S1 the static pair: included, the three columns and the residual exactly 0, the version-3 pair columns are the record values, T_ll(mid) agrees',
             'S2 T_ll at the root: p_ell = pf rho_ell (the scalar kernel is linear and is the records source kernel), the assembly from exact kernel values, the other null direction, the profile, the offsets, the window, the quadrature',
             'S3 the row defect at every dense zero (inner and outer) at the solver tolerance; e(R) from kappa_fd - kappa_formula agrees; the kappa correction is tiny',
             'S4 the response of e and g at every zero to a manufactured defect (rho -> rho(1 + delta); J -> J(1 + delta)) is the predicted one',
             'S5 nothing else changed: version 2 and version 3 analyse agree on every leaf of records, pairs, excluded pairs, stats, grid and matching',
             'S6 the stress key discovery and the loader round trip through the npz layout',
             'S7 end to end: version 2 main, then version 3 main with --seed-cache and --check-against: no re-solve, 0 differing leaves, the new columns written']
    G, ell = float(a.G), float(a.ell)
    engine_path = a.engine_path or os.path.dirname(os.path.abspath(__file__))
    FLD = None
    if a.skip_engine:
        check('S0 engine import (and both records modules)', False, 'skipped by --skip-engine', skipped=True)
    else:
        try:
            FLD, eprov = fl.import_engine(engine_path, a.engine_module)
            check('S0 engine import (and both records modules)', True, '%s from %s; records modules %s and %s' % (
                a.engine_module, eprov['module_file'], os.path.abspath(fl.__file__), os.path.abspath(fl2.__file__)))
        except SystemExit as exc:
            check('S0 engine import (and both records modules)', False, str(exc))
    summary_extra = {}
    if FLD is None:
        for nm in names:
            check(nm, False, 'the engine is not available', skipped=a.skip_engine)
    else:
        R_e = np.arange(1, a.n_sites_engine + 1, dtype=float)
        ell_pass = fl.ell_for_engine(ell)
        sargs, aspec, _sf = fl.resolve_solver_args(FLD, {}, {}, {'kv': {}}, fl.make_opts(rtol=a.rtol, atol=a.atol, collar_policy='flag', support_rel=a.support_rel))
        r_fine, r_last, gnote = fl.build_fine_grid(FLD, R_e, ell_pass, a.fine)
        h_fine = ell / float(a.fine)
        cfg = fl.worker_cfg(fl.make_opts(), r_last, h_fine)
        eps, J = broad_sources(R_e)
        P = float(a.pf) * eps
        print('engine solve args %s; dense grid to %s step %s%s; profile: core %g on sites %d..%d, plateau %g per site on [%g, %g], jf %g; stress P = %g eps_tot' % (
            json.dumps(fl._jsonable(sargs)), fl._fmt(r_last), fl._fmt(h_fine), ('; ' + gnote) if gnote else '', E_CORE0, int(CORE_R0 - CORE_W) + 1,
            int(CORE_R0 + CORE_W) - 1, A_PLATEAU0, RISE[1], FALL[0], JF0, a.pf))
        d = fl.solve_record_dense(FLD, R_e, eps, J, G, ell_pass, sargs, r_fine, cfg)
        if not d.get('ok'):
            for nm in names:
                check(nm, False, 'the engine solve failed: %s %s' % (d.get('failed'), str(d.get('traceback', ''))[-600:]))
        else:
            sres = fl.smooth_stress_record(FLD, R_e, P, ell_pass, r_last, a.support_rel, r_fine, d['zeros_dense'])
            rec0 = record_from_dense(d, 0.0)
            rec1 = dict(rec0, T=1.0, T_over_rs=1.0)
            Ns, vs = d['arrays']['N_sites'], d['arrays']['v_sites']
            F = {'R': R_e, 'T': np.array([0.0, 1.0]), 'eps_tot': np.array([eps, eps]), 'J_tot': np.array([J, J]), 'eps_pairs': None, 'J_pairs': None,
                 'N_site': np.array([Ns, Ns]), 'v_site': np.array([vs, vs]), 'N_applied': None, 'v_applied': None, 'G': G, 'ell': ell,
                 'keys': ['R', 'T', 'eps_tot', 'J_tot', 'N', 'v', 'g', 'ell', 'P_r_tot'], 'solver_string': None, 'label': None, 'path': None}
            meta = {'key': 'P_r_tot', 'note': 'manufactured in-process: P = %g eps_tot' % a.pf, 'reading': 'per-site', 'scale': 1.0, 'candidates': ['P_r_tot'],
                    'pt_key': None, 'n_records': 2, 'n_ok': 2 if sres.get('ok') else 0, 'seconds': fl._float(sres.get('seconds')), 'workers': 1,
                    'kernel': sres.get('kernel'), 'kernel_check_max': fl._float(sres.get('kernel_check')),
                    'quadrature_rel_max_abs': abs(fl._float(sres.get('quadrature_rel'))), 'sum_P_over_sum_eps_all_rows': a.pf}
            stress = {'meta': meta, 'records': {0: sres, 1: sres}}
            opts3 = fl.make_opts(G=G, ell=ell, rs=1.0, stop_E=1e-12, tag='static_pair')
            res3 = fl.analyse([rec0, rec1], F, {}, opts3, {0: d, 1: d}, r_fine, stress=stress)
            opts2 = fl2.make_opts(G=G, ell=ell, rs=1.0, stop_E=1e-12, tag='static_pair')
            res2 = fl2.analyse([rec0, rec1], F, {}, opts2, {0: d, 1: d}, r_fine)
            r0 = res3['records'][0]
            pairs = res3['pairs']
            p1 = pairs[0] if pairs else {}
            deg = int(opts3.interp_degree)
            rho = np.asarray(d['arrays']['rho'], float)
            Jd = np.asarray(d['arrays']['J'], float)
            N, v = (np.asarray(d['arrays'][k], float) for k in ('N', 'v'))
            det = list(d.get('zeros_detail') or [])
            zd = sorted(float(z) for z in d['zeros_dense'])
            rho_R, J_R = fl._float(r0.get('rho_R')), fl._float(r0.get('J_R'))
            Tll_R = fl._float(r0.get('Tll_R'))
            eng = d.get('engine') or {}
            print('solve: zeros %s (%s), E_collar %s, r_collar %s, %.1f s; charge conservation quadrature %s, source spline error %s; stress smoothing %s (%s), kernel check %s, quadrature rel %s' % (
                json.dumps([round(z, 6) for z in zd]), d.get('zeros_source'), fl._fmt(d['E_collar'], 9), fl._fmt(d['r_collar']), d['seconds'],
                fl._fmt(eng.get('charge_conservation_quadrature'), 3), fl._fmt(eng.get('source_spline_error'), 3), 'ok' if sres.get('ok') else 'FAILED: %s' % sres.get('error'),
                sres.get('kernel'), fl._fmt(sres.get('kernel_check'), 3), fl._fmt(sres.get('quadrature_rel'), 3)))

            # ---------------- S1 ----------------
            scale_T = max(abs(Tll_R), abs(rho_R)) if np.isfinite(Tll_R) else abs(rho_R)
            zero_cols = {k: p1.get(k) for k in ('dE', 'dA', 'dR', 'col2', 'W_ext', 'res', 'res_collarE', 'res_applied', 'col2_defect_corr')}
            mid_err = abs(fl._float(p1.get('Tll_mid')) - Tll_R) / scale_T if (np.isfinite(Tll_R) and scale_T > 0) else NAN
            ok1 = (len(pairs) == 1 and not res3['excluded_pairs'] and all(val == 0.0 for val in zero_cols.values())
                   and np.isfinite(Tll_R) and p1.get('Tll_a') == Tll_R and p1.get('Tll_b') == Tll_R and p1.get('Tll_mean') == Tll_R
                   and p1.get('e_root_a') == r0.get('e_root') and p1.get('g_root_b') == r0.get('g_root') and np.isfinite(mid_err) and mid_err <= a.tol_interp)
            check(names[0], ok1, 'pairs %d (excluded %s); the zero columns %s; T_ll(R) %s = pair a %s = b %s = mean %s; |T_ll(mid) - T_ll(R)|/scale %s (tol %g); e(R) %s = pair a %s; pair flags %s' % (
                len(pairs), [(e['T_a'], e['reasons']) for e in res3['excluded_pairs']], {k: fl._fmt(val, 3) for k, val in zero_cols.items()}, fl._fmt(Tll_R, 8),
                fl._fmt(p1.get('Tll_a'), 8), fl._fmt(p1.get('Tll_b'), 8), fl._fmt(p1.get('Tll_mean'), 8), fl._fmt(mid_err, 3), a.tol_interp, fl._fmt(r0.get('e_root'), 3),
                fl._fmt(p1.get('e_root_a'), 3), p1.get('flags')))

            # ---------------- S2 ----------------
            p_arr = np.asarray(sres['p'], float) if sres.get('ok') else None
            try:
                src_e = FLD.SmoothedSources5(eps, J, R_e, ell_pass, float(r_last), a.support_rel)
                rho_fresh = fl._ev(src_e.rho, r_fine)
                src_err = None
            except Exception as exc:
                rho_fresh, src_err = None, '%s: %s' % (type(exc).__name__, exc)
            mrho = float(np.max(np.abs(rho)))
            cons = float(np.max(np.abs(rho_fresh - rho))) / mrho if rho_fresh is not None else NAN
            if p_arr is not None and rho_fresh is not None:
                mask = np.abs(rho_fresh) > 1e-6 * mrho
                lin = float(np.max(np.abs(p_arr[mask] / rho_fresh[mask] - a.pf)))
            else:
                lin = NAN
            exp_T = rho_R * (1.0 + a.pf) - 2.0 * J_R
            err_T = abs(Tll_R - exp_T) / abs(rho_R) if rho_R != 0 else NAN
            err_asm = abs(Tll_R - (rho_R + fl._float(r0.get('p_R')) - 2.0 * J_R))
            err_flip = abs(fl._float(r0.get('Tll_flipJ_R')) - Tll_R - 4.0 * J_R) / abs(rho_R) if rho_R != 0 else NAN
            prof = r0.get('_Tll_profile')
            err_prof = float(np.max(np.abs(np.asarray(prof, float) - (rho + p_arr - 2.0 * Jd)))) if (prof is not None and p_arr is not None) else NAN
            offs = list(r0.get('Tll_offsets_ell') or [])
            vals = list(r0.get('Tll_at_offsets') or [])
            err_off = abs(fl._float(vals[offs.index(0.0)]) - Tll_R) / scale_T if (0.0 in offs and len(vals) == len(offs) and scale_T > 0) else NAN
            win, rel = r0.get('_Tll_window'), r0.get('_Tll_window_rel')
            err_win = abs(float(np.asarray(win, float)[int(np.argmin(np.abs(np.asarray(rel, float))))]) - Tll_R) / scale_T if (win is not None and scale_T > 0) else NAN
            quad = abs(fl._float(sres.get('quadrature_rel')))
            if len(det) >= 2 and fl._float(det[-2].get('rho')) != 0:
                exp_in = fl._float(det[-2].get('rho')) * (1.0 + a.pf) - 2.0 * fl._float(det[-2].get('J'))
                err_in = abs(fl._float(r0.get('Tll_R_in')) - exp_in) / abs(fl._float(det[-2].get('rho')))
            else:
                err_in = NAN
            ok2 = (p_arr is not None and np.isfinite(cons) and cons <= a.tol_source_consistency and np.isfinite(lin) and lin <= a.tol_linear
                   and np.isfinite(err_T) and err_T <= a.tol_assembly and err_asm == 0.0 and np.isfinite(err_flip) and err_flip <= 1e-12
                   and err_prof == 0.0 and np.isfinite(err_off) and err_off <= a.tol_interp and np.isfinite(err_win) and err_win <= a.tol_interp
                   and np.isfinite(quad) and quad <= a.tol_quadrature and np.isfinite(err_in) and err_in <= a.tol_assembly
                   and r0.get('p_R_source') == 'kernel value at the root')
            check(names[1], ok2, 'stress smoothing %s; |SmoothedSources5(eps, J).rho - solve rho_ell|/max %s (tol %g)%s; |p/rho - pf| max %s (tol %g); T_ll(R) %s vs rho_R (1 + pf) - 2 J_R %s: rel %s (tol %g), assembly %s, '
                  'other direction minus 4 J_R rel %s; profile - (rho + p - 2J) max %s; offset-0 sample rel %s, window centre rel %s (tol %g); quadrature rel %s (tol %g); inner zero T_ll rel %s; '
                  'p_R source %s; p/rho at R %s; T_ll at offsets %s (ell) = %s; window sign changes %s; kernel check (.rho vs .J of the stress call) %s' % (
                      'ok' if p_arr is not None else 'FAILED: %s' % sres.get('error'), fl._fmt(cons, 3), a.tol_source_consistency, (' (fresh call failed: %s)' % src_err) if src_err else '',
                      fl._fmt(lin, 3), a.tol_linear, fl._fmt(Tll_R, 9), fl._fmt(exp_T, 9), fl._fmt(err_T, 3), a.tol_assembly, fl._fmt(err_asm, 3), fl._fmt(err_flip, 3), fl._fmt(err_prof, 3),
                      fl._fmt(err_off, 3), fl._fmt(err_win, 3), a.tol_interp, fl._fmt(quad, 3), a.tol_quadrature, fl._fmt(err_in, 3), r0.get('p_R_source'), fl._fmt(r0.get('p_over_rho_R'), 6),
                      offs, [fl._fmt(x, 5) for x in vals], r0.get('Tll_window_n_zero_crossings'), fl._fmt(sres.get('kernel_check'), 3)))

            # ---------------- S3 ----------------
            rd = list(r0.get('root_defects') or [])
            types = [x.get('type') for x in rd]
            e_rels = [abs(fl._float(x.get('e_rel'))) for x in rd]
            g_rels = [abs(fl._float(x.get('g_rel'))) for x in rd]
            den_e = fl._float(r0.get('den_e_root'))
            k_eq = fl._float(r0.get('kappa_eq'))
            diff_k = abs(fl._float(r0.get('e_root_minus_from_kappa')))
            corr = abs(fl._float(r0.get('kappa_corr_root')))
            ok3 = (len(rd) == 2 and types == ['inner', 'outer'] and all(np.isfinite(x) and x <= a.tol_rows for x in e_rels + g_rels)
                   and np.isfinite(diff_k) and np.isfinite(den_e) and diff_k <= a.tol_rows * den_e and np.isfinite(corr) and np.isfinite(k_eq) and corr <= a.tol_rows * abs(k_eq)
                   and r0.get('R_in') == rd[0]['r'] and rd[-1]['r'] == r0.get('R_dense'))
            check(names[2], ok3, 'zeros %s types %s; e relative %s, g relative %s (tol %g); e(R) %s, e(R) from kappa_fd - kappa_formula %s, |difference| %s against tol x 4 pi R^2 (|rho| + |uJ|) = %s; '
                  'kappa correction -N_R G e(R)/R %s against tol x kappa_eq = %s; inner zero e %s g %s; R_in %s = %s, R %s = %s' % (
                      json.dumps([round(x['r'], 6) for x in rd]), types, [fl._fmt(x, 3) for x in e_rels], [fl._fmt(x, 3) for x in g_rels], a.tol_rows, fl._fmt(r0.get('e_root'), 3),
                      fl._fmt(r0.get('e_root_from_kappa'), 3), fl._fmt(diff_k, 3), fl._fmt(a.tol_rows * den_e, 3), fl._fmt(corr, 3), fl._fmt(a.tol_rows * abs(k_eq), 3),
                      fl._fmt(r0.get('e_root_in'), 3), fl._fmt(r0.get('g_root_in'), 3), fl._fmt(r0.get('R_in'), 8), fl._fmt(rd[0]['r'] if rd else NAN, 8), fl._fmt(r0.get('R_dense'), 8),
                      fl._fmt(rd[-1]['r'] if rd else NAN, 8)))

            # ---------------- S4 ----------------
            dl = float(a.delta)
            e0, g0, _M0, _Mp0, Np0 = fl.row_defects(r_fine, N, v, rho, Jd, G, h_fine)
            base_rd = fl.root_defects_from_arrays(r_fine, e0, g0, N, v, rho, Jd, Np0, G, zd, deg)
            e1, g1, _M1, _Mp1, Np1 = fl.row_defects(r_fine, N, v, rho * (1.0 + dl), Jd, G, h_fine)
            rd_rho = fl.root_defects_from_arrays(r_fine, e1, g1, N, v, rho * (1.0 + dl), Jd, Np1, G, zd, deg)
            e2, g2, _M2, _Mp2, Np2 = fl.row_defects(r_fine, N, v, rho, Jd * (1.0 + dl), G, h_fine)
            rd_J = fl.root_defects_from_arrays(r_fine, e2, g2, N, v, rho, Jd * (1.0 + dl), Np2, G, zd, deg)
            errs = []
            same_base = all(base_rd[i]['e'] == rd[i]['e'] and base_rd[i]['g'] == rd[i]['g'] for i in range(min(len(rd), len(base_rd)))) and len(rd) == len(base_rd)
            for i, z in enumerate(zd):
                dz = det[i] if i < len(det) else {}
                u_z = fl._float(dz.get('v')) / fl._float(dz.get('N')) if fl._float(dz.get('N')) != 0 else NAN
                pred_e_rho = -FOUR_PI * z * z * dl * fl._float(dz.get('rho'))
                got_e_rho = rd_rho[i]['e'] - base_rd[i]['e']
                pred_e_J = -FOUR_PI * z * z * u_z * dl * fl._float(dz.get('J'))
                got_e_J = rd_J[i]['e'] - base_rd[i]['e']
                pred_g_J = FOUR_PI * G * z * dl * fl._float(dz.get('J')) / u_z if u_z != 0 else NAN
                got_g_J = rd_J[i]['g'] - base_rd[i]['g']
                errs.append({'z': z, 'e_rho': abs(got_e_rho - pred_e_rho) / abs(pred_e_rho) if pred_e_rho != 0 else NAN,
                             'g_rho': abs(rd_rho[i]['g'] - base_rd[i]['g']),
                             'e_J': abs(got_e_J - pred_e_J) / abs(pred_e_J) if pred_e_J != 0 else NAN,
                             'g_J': abs(got_g_J - pred_g_J) / abs(pred_g_J) if pred_g_J != 0 else NAN})
            ok4 = (len(zd) == 2 and same_base and all(np.isfinite(x['e_rho']) and x['e_rho'] <= a.tol_response and x['g_rho'] == 0.0
                                                      and np.isfinite(x['e_J']) and x['e_J'] <= a.tol_response and np.isfinite(x['g_J']) and x['g_J'] <= a.tol_response for x in errs))
            check(names[3], ok4, 'delta %g; per zero (relative errors of the predicted response; tol %g): %s; the base recomputation equals the record values: %s' % (
                dl, a.tol_response, [{k: (fl._fmt(val, 3) if k != 'z' else fl._fmt(val, 6)) for k, val in x.items()} for x in errs], same_base))

            # ---------------- S5 ----------------
            cmp = {'n': 0, 'bad': []}
            for sec in fl.V3_COMPARE_SECTIONS:
                if sec in res2:
                    fl.compare_trees(fl._jsonable(res2[sec]), fl._jsonable(res3.get(sec)), '/' + sec, fl.V3_COMPARE_EXEMPT, cmp)
            new_rec_keys = sorted(set(res3['records'][0].keys()) - set(res2['records'][0].keys()) - set(k for k in res3['records'][0] if k.startswith('_')))
            new_pair_keys = sorted(set(p1.keys()) - set(res2['pairs'][0].keys())) if (pairs and res2['pairs']) else []
            ok5 = cmp['n'] > 100 and not cmp['bad']
            check(names[4], ok5, '%d leaves compared, %d differ%s; version 3 adds %d record keys and %d pair keys (for example %s)' % (
                cmp['n'], len(cmp['bad']), ('; first: ' + '; '.join('%s: %r -> %r' % tuple(b) for b in cmp['bad'][:5])) if cmp['bad'] else '', len(new_rec_keys), len(new_pair_keys),
                new_rec_keys[:6]))

            # ---------------- S6 ----------------
            k_none = fl.find_stress_key(['R', 'T', 'eps_tot', 'J_tot', 'N', 'v', 'g', 'ell'])
            k_name = fl.find_stress_key(['R', 'T', 'eps_tot', 'J_tot', 'Pt_tot', 'P_r_tot', 'N'])
            k_expl = fl.find_stress_key(['R', 'eps_tot', 'sigma_rr', 'other'], 'sigma_rr')
            k_miss = fl.find_stress_key(['R', 'eps_tot'], 'P_r_tot')
            k_off = fl.find_stress_key(['R', 'P_r_tot'], 'none')
            k_pairs = fl.find_stress_key(['P_r_pairs', 'P_r_tot', 'pr'])
            jpath, npath = save_layout(a.out, 'static_pair', [rec0, rec1], F, {'P_r_tot': np.array([P, P])})
            Fl = fl.load_fields(npath)
            k_load = fl.find_stress_key(Fl['keys'])
            Pr = fl.read_stress(npath, 'P_r_tot', R_e, 'per-site', 1.0, 1.0)
            Pd = fl.read_stress(npath, 'P_r_tot', R_e, 'density', 2.0, 1.0)
            ok6 = (k_none[0] is None and k_name[0] == 'P_r_tot' and k_name[3] == 'Pt_tot' and k_expl[0] == 'sigma_rr' and k_miss[0] is None and k_off[0] is None
                   and k_pairs[0] == 'P_r_tot' and 'pr' in k_pairs[2] and 'P_r_pairs' not in k_pairs[2] and k_load[0] == 'P_r_tot'
                   and Pr.shape == (2, R_e.size) and float(np.max(np.abs(Pr - np.array([P, P])))) == 0.0
                   and float(np.max(np.abs(Pd - 2.0 * np.array([P, P]) * (FOUR_PI * R_e * R_e)[None, :]))) == 0.0)
            check(names[5], ok6, 'no stress key -> %s; by name -> %s (tangential %s); --pr-key sigma_rr -> %s; --pr-key missing -> %s (%s); none -> %s; pairs part excluded -> %s (candidates %s); '
                  'loader: keys %s -> %s, per-site read exact %s, density reading = P x 4 pi R^2 h x scale exact %s' % (
                      k_none[0], k_name[0], k_name[3], k_expl[0], k_miss[0], k_miss[1][:60], k_off[0], k_pairs[0], k_pairs[2], Fl['keys'], k_load[0],
                      float(np.max(np.abs(Pr - np.array([P, P])))) == 0.0, float(np.max(np.abs(Pd - 2.0 * np.array([P, P]) * (FOUR_PI * R_e * R_e)[None, :]))) == 0.0))

            # ---------------- S7 ----------------
            out2 = os.path.join(a.out, 'v2')
            out3 = os.path.join(a.out, 'v3')
            common = ['--json', jpath, '--npz', npath, '--engine-path', engine_path, '--engine-module', a.engine_module, '--rtol', repr(a.rtol), '--atol', repr(a.atol),
                      '--support-rel', repr(a.support_rel), '--collar-policy', 'flag', '--threads', str(a.threads), '--workers', '1']
            t7 = time.time()
            rc2, rc3, err7 = None, None, None
            try:
                rc2 = fl2.main(common + ['--out', out2])
            except SystemExit as exc:
                err7 = 'version 2 main: %s' % exc
            t7b = time.time()
            v2_json = os.path.join(out2, 'firstlaw_records_v2_static_pair.json')
            v2_cache = os.path.join(out2, 'firstlaw_dense_v2_static_pair_cache.npz')
            v3_json = os.path.join(out3, 'firstlaw_records_v3_static_pair.json')
            if err7 is None:
                try:
                    rc3 = fl.main(common + ['--out', out3, '--seed-cache', v2_cache, '--check-against', v2_json])
                except SystemExit as exc:
                    err7 = 'version 3 main: %s' % exc
            n_cmp, bad7, J3, seeded, has_prof, has_section, key_ok, new_cols, wall3 = 0, [], {}, False, False, False, False, False, NAN
            if err7 is None and os.path.exists(v2_json) and os.path.exists(v3_json):
                n_cmp, bad7 = fl.compare_json_files(v2_json, v3_json)
                with open(v3_json) as fh:
                    J3 = json.load(fh)
                rooted3 = [r for r in J3.get('records') or [] if r.get('has_root')]
                new_cols = bool(rooted3) and all(r.get('Tll_R') is not None and r.get('e_root') is not None and r.get('g_root') is not None for r in rooted3)
                cache_msg = (J3.get('provenance') or {}).get('cache') or ''
                wall3 = fl._float((J3.get('provenance') or {}).get('solve_wall_seconds'))
                seeded = ('seed cache' in cache_msg) and ('2 records read' in cache_msg) and (J3.get('stats') or {}).get('n_records_resolved') == 2 and np.isfinite(wall3) and wall3 < 1.0
                try:
                    z3 = np.load(os.path.join(out3, 'firstlaw_records_v3_static_pair.npz'), allow_pickle=False)
                    has_prof = 'rec_Tll_profile' in z3.files and 'rec_Tll_window' in z3.files
                except Exception:
                    has_prof = False
                try:
                    with open(os.path.join(out3, 'firstlaw_records_v3_static_pair.md')) as fh:
                        md3 = fh.read()
                    has_section = ('## Version 3 columns' in md3) and ('## Per-pair table, version 3 columns' in md3)
                except Exception:
                    has_section = False
                key_ok = (J3.get('stress_meta') or {}).get('key') == 'P_r_tot'
                summary_extra['v3_stats'] = J3.get('stats')
                summary_extra['v3_stress_meta'] = J3.get('stress_meta')
            ok7 = (err7 is None and rc2 == 0 and rc3 == 0 and n_cmp > 100 and not bad7 and new_cols and seeded and has_prof and has_section and key_ok)
            check(names[6], ok7, 'version 2 main rc %s (%.1f s, 2 solves), version 3 main rc %s (%.1f s, seeded: %s, solve wall %s s)%s; bitwise: %d leaves compared, %d differ%s; new columns on the rooted records %s; '
                  'stress key in the json %s; npz has rec_Tll_profile and rec_Tll_window %s; md has the version-3 sections %s' % (
                      rc2, t7b - t7, rc3, time.time() - t7b, seeded, fl._fmt(wall3, 3), ('; ERROR %s' % err7) if err7 else '', n_cmp, len(bad7),
                      ('; first: ' + '; '.join('%s: %r -> %r' % tuple(b) for b in bad7[:5])) if bad7 else '', new_cols, key_ok, has_prof, has_section))

    n_fail = sum(1 for r in results if not r['pass'] and not r['skipped'])
    n_skip = sum(1 for r in results if r['skipped'])
    summary = {'results': results, 'n_fail': n_fail, 'n_skip': n_skip, 'seconds': time.time() - t0, 'options': vars(a)}
    summary.update(summary_extra)
    with open(os.path.join(a.out, 'firstlaw_selftest_v4.json'), 'w') as fh:
        json.dump(fl._jsonable(summary), fh, indent=1)
    print('firstlaw_selftest_v4: %d checks, %d failed, %d skipped, %.1f s, outputs in %s' % (len(results), n_fail, n_skip, time.time() - t0, a.out))
    if n_fail:
        return 1
    if n_skip and not a.allow_skip:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
