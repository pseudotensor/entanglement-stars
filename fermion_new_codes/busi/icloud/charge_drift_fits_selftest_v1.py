#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""charge_drift_fits_selftest_v1.py -- manufactured checks of charge_drift_fits_v1.py (reads no production data).

1. On E_tau(T) = E_* + c1 tau + c2 tau^2 the 3 forms return E_* exactly for c2 = 0; with c2 != 0 the two-term form
   is exact and the first-order forms err by -8 c2 and -32 c2 (their O(tau^2) terms); the order test returns 2 for
   c2 = 0 and 2(1 + 12x)/(1 + 6x), x = c2/c1, otherwise.
2. The key mapping fails loudly on a missing required key, on a ledger gap, on a wrong tau label, on duplicate clocks.
3. End to end on 3 manufactured histories with a first-order drift: 173 matched clocks, E_* = E0 at every printed
   clock, order test 2, E0 in band, birth clocks interpolated exactly, parts' forms consistent, args check flags.
Run: python -I -B charge_drift_fits_selftest_v1.py
"""
from __future__ import print_function

import json
import math
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I drops the script directory from sys.path
import charge_drift_fits_v1 as cdf  # noqa: E402

N_OK = [0]


def check(cond, msg):
    if not cond:
        raise AssertionError('FAIL: ' + msg)
    N_OK[0] += 1
    print('ok   ' + msg)


def fake_history(tau, T_end_over_rs, c1=0.01, kappa=0.3, m_arg=0.2, drop_key=None, drop_at=None, gap_at=None,
                 duplicate_at=None, tau_label=None):
    """A first-order drift E = E0 - c1 tau T/r_s; a consistent ledger: H_in = 90 - 0.02 T/r_s, c per step
    = kappa tau^2/rs^2 (first order accumulated), b = dH_in step - c; I_E constant; birth at T/r_s 12."""
    rs, E0 = 64.0, 100.0
    n = int(round(T_end_over_rs * rs / tau))
    recs = []
    for i in range(n + 1):
        T = i * tau
        Tr = T / rs
        E = E0 - c1 * tau * Tr
        H = 90.0 - 0.02 * Tr
        rec = {'T': T, 'T_over_rs': Tr, 'E': E, 'E_over_E0': E / E0,
               'trapping_zeros': [] if Tr < 12.0 else [30.0 - 0.1 * (Tr - 12.0)],
               'min_N_minus_absv': 0.5 - Tr / 24.0, 'H_in': H, 'M1_I_E': 1e-4, 'converged': True,
               'engine': 'fake_v%d' % int(tau), 'r_collar': 2048.0, 'M1_parts': {'cut': 2038}}
        if i > 0:
            dH = -0.02 * tau / rs
            c = kappa * tau * tau / (rs * rs)
            dE_dT = -c1 * tau / rs
            dH_dT = -0.02 / rs
            if not (gap_at is not None and abs(Tr - gap_at) < 1e-12):
                rec['M1_balance'] = {'dHin': dH, 'evolution_exact': 0.0, 'field_work': dH - c, 'reference_reset': c,
                                     'closure': 0.0, 'cut_changed': False, 'evolution_trapezoid': 0.0}
            rec['dE_dT'] = dE_dT
            rec['dHin_dT'] = dH_dT
            rec['balance_d_E_minus_Hin_dT'] = dE_dT - dH_dT
        if drop_key is not None and drop_at is not None and abs(Tr - drop_at) < 1e-12:
            rec.pop(drop_key, None)
        recs.append(rec)
        if duplicate_at is not None and abs(Tr - duplicate_at) < 1e-12:
            recs.append(dict(rec))
    return {'script': 'fake_engine.py', 'args': {'tau': tau if tau_label is None else tau_label, 'rs': rs, 'm': m_arg,
                                                 'g': 0.25, 'out': 'dir_%d' % int(tau), 'tag': 'fake'},
            'head': {'E0': E0, 'site_fields': 'fake'}, 'records': recs}


def test_forms():
    Es, c1 = 100.0, 0.37
    E = dict((t, Es + c1 * t) for t in (8, 4, 2))
    f = cdf.richardson(E[8], E[4], E[2])
    for k in cdf.FORMS:
        check(abs(f[k] - Es) < 1e-9, 'form %s returns E_* exactly for a first-order series (%s)' % (k, cdf.FORM_LABEL[k]))
    check(abs(f['order_ratio'] - 2.0) < 1e-9, 'order test returns 2 for a first-order series')
    check(f['spread'] < 1e-9, 'spread vanishes for a first-order series')
    c2 = 0.011
    E = dict((t, Es + c1 * t + c2 * t * t) for t in (8, 4, 2))
    f = cdf.richardson(E[8], E[4], E[2])
    check(abs(f['f2nd'] - Es) < 1e-9, 'two-term form exact with c2 != 0')
    check(abs((f['f42'] - Es) + 8.0 * c2) < 1e-9, '2E_2 - E_4 errs by -8 c2 (its O(tau^2) term)')
    check(abs((f['f84'] - Es) + 32.0 * c2) < 1e-9, '2E_4 - E_8 errs by -32 c2 (its O(tau^2) term)')
    x = c2 / c1
    check(abs(f['order_ratio'] - 2.0 * (1.0 + 12.0 * x) / (1.0 + 6.0 * x)) < 1e-9,
          'order test equals 2(1+12x)/(1+6x), x = c2/c1')
    check(cdf.richardson(None, 1.0, 2.0)['f42'] is None, 'forms return None on a missing input')
    check(cdf.richardson(5.0, 5.0, 5.0)['order_ratio'] is None, 'order test is None when the members agree (0/0)')


def test_mapping_failures():
    try:
        cdf.digest_member(fake_history(8.0, 21.5, drop_key='E', drop_at=1.0), '8', 'fake')
        raise AssertionError('no error on a missing required key')
    except cdf.MappingError as e:
        check('E' in str(e) and 'T/r_s = 1.0' in str(e), 'missing required key E at T/r_s 1.0 fails loudly: %s' % e)
    try:
        cdf.digest_member(fake_history(8.0, 21.5, drop_key='min_N_minus_absv', drop_at=0.0), '8', 'fake')
        raise AssertionError('no error on a missing min_N_minus_absv')
    except cdf.MappingError as e:
        check('min_N_minus_absv' in str(e), 'missing min_N_minus_absv fails loudly')
    try:
        cdf.digest_member(fake_history(4.0, 21.5, gap_at=3.0), '4', 'fake')
        raise AssertionError('no error on a ledger gap')
    except cdf.MappingError as e:
        check('M1_balance' in str(e) and '--allow-gaps' in str(e), 'a gap in M1_balance after its first appearance fails loudly')
    mem = cdf.digest_member(fake_history(4.0, 21.5, gap_at=3.0), '4', 'fake', allow_gaps=True)
    check(mem['coverage']['M1_balance']['n_gaps_after_first'] == 1, '--allow-gaps tolerates and counts the gap')
    try:
        cdf.digest_member(fake_history(8.0, 21.5, tau_label=4.0), '8', 'fake')
        raise AssertionError('no error on a tau label mismatch')
    except cdf.MappingError as e:
        check('does not match' in str(e), 'args.tau against the member label fails loudly')
    try:
        cdf.digest_member(fake_history(8.0, 21.5, duplicate_at=2.0), '8', 'fake')
        raise AssertionError('no error on duplicate clocks')
    except ValueError as e:
        check('duplicate' in str(e), 'duplicate clocks fail loudly')
    try:
        cdf.digest_member({'args': {'tau': 8.0}, 'head': {}, 'records': []}, '8', 'fake')
        raise AssertionError('no error on an empty records list')
    except cdf.MappingError:
        check(True, 'an empty records list fails loudly')
    try:
        cdf.clock_key(0.1)
        raise AssertionError('no error for a clock off the grid')
    except ValueError:
        check(True, 'a clock off the 1/1024 grid is refused')


def test_end_to_end():
    tmp = tempfile.mkdtemp(prefix='cdf_selftest_')
    try:
        paths = {}
        for tau, T_end in ((8.0, 40.0), (4.0, 40.0), (2.0, 21.5)):
            p = os.path.join(tmp, 'fake_tau%d.json' % int(tau))
            h = fake_history(tau, T_end, m_arg=0.2 if tau != 2.0 else 0.3)   # a physics-arg difference to flag
            with open(p, 'w') as f:
                json.dump(h, f)
            paths[str(int(tau))] = p
        out_json, out_md = os.path.join(tmp, 'r.json'), os.path.join(tmp, 'r.md')
        rc = cdf.main(['--tau8', paths['8'], '--tau4', paths['4'], '--tau2', paths['2'],
                       '--out-json', out_json, '--out-md', out_md])
        check(rc == 0, 'main returns 0')
        with open(out_json) as f:
            R = json.load(f)
        check(R['clocks']['n_matched'] == 173, 'matched clocks: 173 (0 to 21.5 by 0.125)')
        check(abs(R['clocks']['matched_last'] - 21.5) < 1e-12, 'last common clock 21.5')
        check(R['clocks']['chosen'] == [11.0, 12.0, 21.5], 'chosen clocks before/formation/last = 11.0, 12.0, 21.5 (got %s)'
              % R['clocks']['chosen'])
        for s in R['chosen_clock_table']:
            for k in cdf.FORMS:
                check(abs(s['E_star'][k] - 100.0) < 1e-9, 'T/r_s %s: %s gives E0 on the manufactured drift' % (s['T_over_rs'], k))
            if s['T_over_rs'] > 0:
                check(abs(s['E_star']['order_ratio'] - 2.0) < 1e-9, 'T/r_s %s: order ratio 2' % s['T_over_rs'])
            check(s['E0_in_band'] is True and s['order_ok'] is True, 'T/r_s %s: E0 in band and order ok' % s['T_over_rs'])
            c = s['parts']['c']['forms']
            check(abs(c['f42']) < 1e-9 and abs(c['order_ratio'] - 2.0) < 1e-9, 'T/r_s %s: accumulated c extrapolates to 0 at order 2' % s['T_over_rs'])
            b = s['parts']['b']['forms']
            check(abs(b['f42'] - (-0.02 * s['T_over_rs'])) < 1e-9, 'T/r_s %s: accumulated b extrapolates to the continuum work' % s['T_over_rs'])
            for m in cdf.MEMBERS:
                check(abs(s['ledger_closure_over_window'][m]) < 1e-9, 'T/r_s %s tau %s: ledger closes over the window' % (s['T_over_rs'], m))
                I = s['parts']['I']['values'][m]
                check(abs(I - 1e-4 * s['T_over_rs'] * 64.0) < 1e-9, 'T/r_s %s tau %s: current integral by trapezoid' % (s['T_over_rs'], m))
        check(R['acceptance']['verdict'] == 'PASS', 'acceptance verdict PASS on the manufactured ladder')
        b = R['birth']
        for m in cdf.MEMBERS:
            check(abs(b['per_member'][m]['interpolated'] - 12.0) < 1e-9 and abs(b['per_member'][m]['grid'] - 12.0) < 1e-9,
                  'tau %s: birth clock 12.0 (grid and interpolated)' % m)
        check(abs(b['forms']['interpolated']['f2nd'] - 12.0) < 1e-9, 'birth clock forms return 12.0')
        pc = R['paper_check']
        check(pc['present'] and abs(pc['recomputed_percent']['8'] - 0.01 * 8 * 21.5) < 1e-9, 'deficit at 21.5 recomputed from E/E0')
        check(abs(pc['ratios_recomputed']['D8/D4'] - 2.0) < 1e-9, 'deficit ratios 2 on the manufactured drift')
        check('m' in R['version_check']['physics_args_differing'], 'a differing physics arg (m) is flagged')
        check(len(R['extension_84_past_last_common']) == 148, '(8,4) extension past 21.5 has 148 clocks (21.625..40 by 0.125)')
        check(abs(R['extension_84_past_last_common'][-1]['f84'] - 100.0) < 1e-9, '(8,4) form at 40 gives E0')
        for m in cdf.MEMBERS:
            c = R['members'][m]['checks']
            check(c['dE_rate_resid_max'] < 1e-9 and c['balance_sum_resid_max'] < 1e-12, 'tau %s: mapping checks tight' % m)
        check(os.path.getsize(out_md) > 2000, 'md written')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    test_forms()
    test_mapping_failures()
    test_end_to_end()
    print('SELFTEST PASS (%d checks)' % N_OK[0])
