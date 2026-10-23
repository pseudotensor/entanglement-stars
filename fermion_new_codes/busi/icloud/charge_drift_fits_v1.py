#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""charge_drift_fits_v1.py -- busi lane, icloud item 10 (P12 claims 8 and 9: the charge-drift fits).

Reads the tau 8, 4, 2 histories READ-ONLY (paths as arguments), finds the matched clocks (common T/r_s) and at
each of them applies the 3 Richardson forms of P12 claim 8,
    X_*(42) = 2 X_2 - X_4,    X_*(84) = 2 X_4 - X_8,    X_*(2nd) = (X_8 - 6 X_4 + 8 X_2) / 3,
to the collar charge E (and E/E0); to the accumulated ledger parts the records carry (P12's a, b, c as
M1_balance.evolution_exact, field_work, reference_reset; P12's -Delta W and bridge only as their sum
Delta(E - H_in) = balance_d_E_minus_Hin_dT * dT; the Killing cut current M1_I_E integrated over T);
to min(N - |v|); and to the birth clock of the first trapping zero.  lambda(T) = E_*/E_8 with the error stated
as the spread of the 3 forms; the order test (X_8 - X_4)/(X_4 - X_2) against 2.  The paper's 9.15, 5.00, 2.44
percent deficits at T/r_s 21.5 are recomputed from E/E0.  Writes one results json and one md (no figure).
Standard library only; CPU; seconds.  Nothing under the input paths is written.

Usage:
  python -I -B charge_drift_fits_v1.py --tau8 A.json --tau4 B.json --tau2 C.json
         [--out-json PATH] [--out-md PATH] [--clocks 10.875,12.125,21.5] [--allow-gaps]
"""
from __future__ import print_function

import argparse
import collections
import datetime
import hashlib
import json
import math
import os
import re
import sys

SCRIPT = 'charge_drift_fits_v1.py'
MEMBERS = ('8', '4', '2')                      # the tau ladder, coarsest first
FORMS = ('f42', 'f84', 'f2nd')
FORM_LABEL = {'f42': '2X_2 - X_4', 'f84': '2X_4 - X_8', 'f2nd': '(X_8 - 6X_4 + 8X_2)/3'}
KEY_SCALE = 1024                               # matched-clock key = round(T/r_s * 1024): exact for multiples of 1/32
DEFAULT_OUT_JSON = '<internal>/charge_drift_fits_v1.json'
DEFAULT_OUT_MD = '<internal>/charge_drift_fits_v1.md'
PAPER_CLOCK = 21.5
PAPER_DEFICITS = {'8': 9.15, '4': 5.00, '2': 2.44}          # item 5 sentence 5 (r_s/a 64, T/r_s 21.5)
PAPER_RATIOS = {'D8/D4': 1.83, 'D4/D2': 2.05}
PAPER_CURRENT_SHARE = {'8': 1.6, '4': 0.2, '2': 0.002}       # percent of the deficit carried by the cut current
P12_ILLUSTRATION = {'lambda': 1.11235, 'E8_over_E0': 0.899}   # P12 claim 9 (full refund, root held)
ORDER_WINDOW = (1.5, 2.5)                      # the order test passes inside 2 +- 25 percent
BAND_TOL_REL = 1e-9                            # roundoff tolerance for "E0 in band"

BOOKKEEPING_ARGS = frozenset(['tau', 'T_over_rs', 'out', 'tag', 'resume_from', 'resume_T_over_rs',
                              'resume_grandparent', 'slices_c128', 'slices_c64', 'slice_dir', 'gpu',
                              'streams', 'benchmark'])
REQUIRED_KEYS = ('T_over_rs', 'E', 'E_over_E0', 'trapping_zeros', 'min_N_minus_absv')
LEDGER_SCALARS = ('H_in', 'M1_I_E')
STEP_KEYS = ('dE_dT', 'dHin_dT', 'balance_d_E_minus_Hin_dT')
BALANCE_KEYS = ('dHin', 'evolution_exact', 'field_work', 'reference_reset', 'closure')
PART_NAMES = ('a', 'b', 'c', 'closure', 'dHin', 'G', 'I', 'dE', 'dHin_direct', 'dG_direct')
PART_LABEL = {
    'a': 'a: sum evolution_exact (frozen-field evolution)',
    'b': 'b: sum field_work',
    'c': 'c: sum reference_reset',
    'closure': 'sum closure (0 by construction)',
    'dHin': 'a+b+c: sum M1_balance.dHin',
    'G': '-Delta W + Delta B: sum balance_d_E_minus_Hin_dT * dT',
    'I': 'current: trapezoid of M1_I_E over T',
    'dE': 'Delta E over the window',
    'dHin_direct': 'H_in(T) - H_in(start)',
    'dG_direct': '(E - H_in)(T) - (E - H_in)(start)',
}
KEY_MAPPING = [
    ('E (the collar charge E_G)', 'E; E_over_E0; E0 in head', 'carried'),
    ('a = <A_Pi(h0), C1 - C0>', 'M1_balance.evolution_exact (evolution_trapezoid: alternative quadrature)',
     'carried per step (records with T > 0)'),
    ('b = <A_Pi(h1 - h0), C1 - R1>', 'M1_balance.field_work (= (Y2 - Y1) - (X2 - X1) of M1_parts; checked)',
     'carried per step'),
    ('c = -<A_Pi(h0), R1 - R0>', 'M1_balance.reference_reset', 'carried per step'),
    ('a + b + c = Delta K_Pi', 'M1_balance.dHin = H_in(T) - H_in(T - tau); M1_balance.closure', 'carried per step'),
    ('Delta W (the geometric work)', 'no key of its own', 'NOT carried'),
    ('Delta B_bridge', 'no key of its own; only the sum -Delta W + Delta B = Delta(E - H_in) '
     '= balance_d_E_minus_Hin_dT * dT', 'NOT carried separately'),
    ('current (the Killing cut current)', 'M1_I_E per record (= M1_parts.I_E_start = M9.collar.I_E_cut); '
     'accumulated by the trapezoid rule over T', 'carried'),
    ('min(N - |v|)', 'min_N_minus_absv', 'carried'),
    ('birth clock', 'first record with trapping_zeros non-empty; first with min_N_minus_absv < 0; '
     'the linear zero crossing of min_N_minus_absv', 'derived'),
]


class MappingError(KeyError):
    """A key the mapping needs is missing: loud, with the member and the clock."""


def need(container, key, member, clock, where='record'):
    if not isinstance(container, dict) or key not in container:
        raise MappingError('tau %s, %s at T/r_s = %s: key %r is missing' % (member, where, clock, key))
    return container[key]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def clock_key(t):
    k = float(t) * KEY_SCALE
    kr = int(round(k))
    if abs(k - kr) > 1e-6:
        raise ValueError('clock T/r_s = %r is not on the 1/%d grid' % (t, KEY_SCALE))
    return kr


def key_to_clock(k):
    return k / float(KEY_SCALE)


def safe_div(a, b):
    try:
        if a is None or b is None:
            return None
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return None
    if b == 0.0 or math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
        return None
    return a / b


def richardson(x8, x4, x2):
    """P12 claim 8: the 3 forms, the band [min, max] and its spread, the differences and the order test."""
    none = dict((k, None) for k in ('f42', 'f84', 'f2nd', 'band_lo', 'band_hi', 'spread', 'd84', 'd42', 'order_ratio'))
    try:
        x8, x4, x2 = float(x8), float(x4), float(x2)
    except (TypeError, ValueError):
        return none
    if any(math.isnan(v) or math.isinf(v) for v in (x8, x4, x2)):
        return none
    f42 = 2.0 * x2 - x4
    f84 = 2.0 * x4 - x8
    f2nd = (x8 - 6.0 * x4 + 8.0 * x2) / 3.0
    lo, hi = min(f42, f84, f2nd), max(f42, f84, f2nd)
    d84, d42 = x8 - x4, x4 - x2
    scale = max(abs(x8), abs(x4), abs(x2), 1e-300)
    ratio = (d84 / d42) if abs(d42) > 1e-13 * scale else None
    return {'f42': f42, 'f84': f84, 'f2nd': f2nd, 'band_lo': lo, 'band_hi': hi, 'spread': hi - lo,
            'd84': d84, 'd42': d42, 'order_ratio': ratio}


def _upd(d, key, val):
    d[key] = val if d.get(key) is None else max(d[key], val)


def digest_member(d, member, path, allow_gaps=False):
    """One history -> flat rows carrying the P12 names; loud on a missing key; coverage of the per-step ledger."""
    args = d.get('args') or {}
    head = d.get('head') or {}
    recs = d.get('records')
    if not isinstance(recs, list) or not recs:
        raise MappingError('tau %s: %s carries no records list' % (member, path))
    tau = args.get('tau')
    try:
        tau_ok = abs(float(tau) - float(member)) < 1e-9
    except (TypeError, ValueError):
        tau_ok = False
    if not tau_ok:
        raise MappingError('tau %s: args.tau = %r in %s does not match the member label' % (member, tau, path))
    rs = float(args.get('rs') or 64.0)
    rows = []
    pat_lz = re.compile(r'from T/rs\s*([0-9]+(?:\.[0-9]+)?)')
    for i, r in enumerate(recs):
        if not isinstance(r, dict):
            raise MappingError('tau %s: record %d is not a dict' % (member, i))
        clock = r.get('T_over_rs')
        row = {'index': i}
        for k in REQUIRED_KEYS:
            row[k] = need(r, k, member, clock)
        row['T_over_rs'] = float(row['T_over_rs'])
        row['key'] = clock_key(row['T_over_rs'])
        row['T'] = float(r['T']) if r.get('T') is not None else row['T_over_rs'] * rs
        row['n_zeros'] = len(row['trapping_zeros'] or [])
        row['R_t'] = r.get('R_t')
        for k in LEDGER_SCALARS:
            row[k] = r.get(k)
        row['converged'] = r.get('converged')
        row['iterations'] = r.get('iterations')
        row['engine'] = r.get('engine')
        row['r_collar'] = r.get('r_collar')
        mp = r.get('M1_parts') or {}
        row['cut'] = mp.get('cut')
        for k in ('X1', 'X2', 'Y1', 'Y2'):
            row[k] = mp.get(k)
        bal = r.get('M1_balance')
        row['has_balance'] = isinstance(bal, dict)
        if row['has_balance']:
            for k in BALANCE_KEYS:
                row['bal_' + k] = need(bal, k, member, clock, where='M1_balance')
            row['bal_cut_changed'] = bool(bal.get('cut_changed'))
            row['bal_evolution_trapezoid'] = bal.get('evolution_trapezoid')
        for k in STEP_KEYS:
            row[k] = r.get(k)
        row['has_step'] = all(row[k] is not None for k in STEP_KEYS)
        row['nsub_max'] = r.get('nsub_max')
        modes, nsub_gt1, meff_gt1 = set(), 0, 0
        cont = r.get('continuation')
        if isinstance(cont, dict):
            for c in cont.values():
                if isinstance(c, dict):
                    modes.add(str(c.get('mode')))
                    if (c.get('nsub') or 1) > 1:
                        nsub_gt1 += 1
                    if (c.get('m_eff') or 1) > 1:
                        meff_gt1 += 1
        row['cont_modes'] = sorted(modes)
        row['cont_nsub_gt1'] = nsub_gt1
        row['cont_meff_gt1'] = meff_gt1
        row['continuation_label'] = r.get('continuation_label')
        rows.append(row)

    keys = [row['key'] for row in rows]
    dups = sorted(k for k, n in collections.Counter(keys).items() if n > 1)
    if dups:
        raise ValueError('tau %s: duplicate clocks T/r_s %s' % (member, [key_to_clock(k) for k in dups[:10]]))
    if keys != sorted(keys):
        raise ValueError('tau %s: the records are not in clock order' % member)
    by_key = dict((row['key'], row) for row in rows)

    # coverage of the ledger keys: carried at all; gaps after the first appearance are loud
    coverage = {}
    presence = {'H_in': [row['H_in'] is not None for row in rows],
                'M1_I_E': [row['M1_I_E'] is not None for row in rows],
                'M1_balance': [row['has_balance'] for row in rows],
                'step_rates': [row['has_step'] for row in rows]}
    for name in ('H_in', 'M1_I_E', 'M1_balance', 'step_rates'):
        pres = presence[name]
        first = next((i for i, p in enumerate(pres) if p), None)
        gaps = [] if first is None else [rows[i]['T_over_rs'] for i in range(first, len(rows)) if not pres[i]]
        coverage[name] = {'carried': first is not None,
                          'first_clock': None if first is None else rows[first]['T_over_rs'],
                          'first_index': first, 'n_gaps_after_first': len(gaps), 'gap_clocks_head': gaps[:10]}
        if gaps and not allow_gaps:
            raise MappingError('tau %s: %s is missing at %d record(s) after its first appearance (T/r_s %s): %s ... '
                               '(pass --allow-gaps to tolerate)' % (member, name, len(gaps), rows[first]['T_over_rs'],
                                                                   gaps[:10]))
    cand = []
    if coverage['M1_balance']['carried']:
        cand.append(max(coverage['M1_balance']['first_index'], 1) - 1)
    if coverage['step_rates']['carried']:
        cand.append(max(coverage['step_rates']['first_index'], 1) - 1)
    for name in LEDGER_SCALARS:
        if coverage[name]['carried']:
            cand.append(coverage[name]['first_index'])
    ledger_start_index = max(cand) if cand else 0

    # per-record consistency of the mapping (maxima of residuals)
    chk = dict((k, None) for k in ('dE_rate_resid_max', 'dHin_rate_resid_max', 'G_rate_resid_max',
                                   'dHin_balance_resid_max', 'balance_sum_resid_max',
                                   'field_work_formula_resid_max'))
    chk['n_balance_records'] = 0
    for i in range(1, len(rows)):
        r, p = rows[i], rows[i - 1]
        dT = r['T'] - p['T']
        if r['has_step']:
            _upd(chk, 'dE_rate_resid_max', abs(r['dE_dT'] * dT - (r['E'] - p['E'])))
            if r['H_in'] is not None and p['H_in'] is not None:
                _upd(chk, 'dHin_rate_resid_max', abs(r['dHin_dT'] * dT - (r['H_in'] - p['H_in'])))
                _upd(chk, 'G_rate_resid_max', abs(r['balance_d_E_minus_Hin_dT'] * dT
                                                  - ((r['E'] - r['H_in']) - (p['E'] - p['H_in']))))
        if r['has_balance']:
            chk['n_balance_records'] += 1
            if r['H_in'] is not None and p['H_in'] is not None:
                _upd(chk, 'dHin_balance_resid_max', abs(r['bal_dHin'] - (r['H_in'] - p['H_in'])))
            _upd(chk, 'balance_sum_resid_max', abs(r['bal_dHin'] - (r['bal_evolution_exact'] + r['bal_field_work']
                                                                    + r['bal_reference_reset'] + r['bal_closure'])))
            if all(r[k] is not None for k in ('X1', 'X2', 'Y1', 'Y2')):
                fw = (r['Y2'] - r['Y1']) - (r['X2'] - r['X1'])
                _upd(chk, 'field_work_formula_resid_max', abs(fw - r['bal_field_work']))

    collar_changes, cut_changes = [], []
    for i in range(1, len(rows)):
        if rows[i]['r_collar'] != rows[i - 1]['r_collar']:
            collar_changes.append({'T_over_rs': rows[i]['T_over_rs'], 'from': rows[i - 1]['r_collar'],
                                   'to': rows[i]['r_collar'], 'dE_record': rows[i]['E'] - rows[i - 1]['E'],
                                   'dE_prev_record': (rows[i - 1]['E'] - rows[i - 2]['E']) if i >= 2 else None})
        if rows[i]['cut'] != rows[i - 1]['cut']:
            cut_changes.append({'T_over_rs': rows[i]['T_over_rs'], 'from': rows[i - 1]['cut'], 'to': rows[i]['cut']})
    n_cut_changed_flags = sum(1 for row in rows if row.get('bal_cut_changed'))
    unconv = [row['T_over_rs'] for row in rows if row['converged'] is False]

    engines = []
    for row in rows:
        e = str(row['engine'])
        if engines and engines[-1]['engine'] == e:
            engines[-1]['to'] = row['T_over_rs']
            engines[-1]['n'] += 1
        else:
            engines.append({'engine': e, 'from': row['T_over_rs'], 'to': row['T_over_rs'], 'n': 1})
    lz = None
    for row in rows:
        mt = pat_lz.search(row.get('continuation_label') or '')
        if mt:
            lz = {'first_record_clock': row['T_over_rs'], 'label_from_T_over_rs': float(mt.group(1)),
                  'label': (row.get('continuation_label') or '')[:200]}
            break

    def _nsub(row):
        return row['cont_nsub_gt1'] > 0 or ((row['nsub_max'] or 1) > 1)

    census = {'records_with_nsub_gt1': sum(1 for row in rows if _nsub(row)),
              'records_with_meff_gt1': sum(1 for row in rows if row['cont_meff_gt1'] > 0),
              'first_clock_nsub_gt1': next((row['T_over_rs'] for row in rows if _nsub(row)), None),
              'first_clock_meff_gt1': next((row['T_over_rs'] for row in rows if row['cont_meff_gt1'] > 0), None),
              'modes': sorted(set(m for row in rows for m in row['cont_modes']))}

    first_rec = recs[0]
    solver = first_rec.get('solver') if isinstance(first_rec.get('solver'), dict) else {}
    first_extract = {'E': first_rec.get('E'), 'E_over_E0': first_rec.get('E_over_E0'),
                     'E_dust_shells': first_rec.get('E_dust_shells'),
                     'min_N_minus_absv': first_rec.get('min_N_minus_absv'), 'N_min': first_rec.get('N_min'),
                     'R_c': first_rec.get('R_c'), 'v_at_Rc': first_rec.get('v_at_Rc'), 'H_in': first_rec.get('H_in'),
                     'M1_I_E': first_rec.get('M1_I_E'), 'r_collar': first_rec.get('r_collar'),
                     'residual_W': solver.get('residual_W'), 'residual_invN': solver.get('residual_invN'),
                     'solver_module': solver.get('module'), 'engine': first_rec.get('engine'),
                     'record_label': first_rec.get('record_label')}
    resume = head.get('resume') if isinstance(head.get('resume'), dict) else None
    head_extract = {'E0': head.get('E0'), 'site_fields': head.get('site_fields'),
                    'label': (head.get('label') or '')[:400], 'label_v19': (head.get('label_v19') or '')[:400] or None,
                    'resume': None if resume is None else dict((k, resume.get(k)) for k in
                                                               ('from', 'script', 'T_over_rs', 'records_replayed',
                                                                'max_eps_rel', 'max_J_rel'))}
    return {'member': member, 'tau': float(tau), 'rs': rs, 'script': d.get('script'), 'args': args,
            'E0': head.get('E0'), 'head': head_extract, 'first_record': first_extract, 'rows': rows,
            'by_key': by_key, 'n_records': len(rows), 'clock_first': rows[0]['T_over_rs'],
            'clock_last': rows[-1]['T_over_rs'],
            'record_spacing': (rows[1]['T_over_rs'] - rows[0]['T_over_rs']) if len(rows) > 1 else None,
            'coverage': coverage, 'ledger_start_index': ledger_start_index, 'checks': chk,
            'collar_changes': collar_changes, 'cut_changes': cut_changes, 'n_cut_changed_flags': n_cut_changed_flags,
            'unconverged_clocks': unconv, 'engines': engines, 'lz_label': lz, 'continuation_census': census}


def accumulate(mem, start_key):
    """Cumulative ledger parts over the member's own records from the record at start_key (exclusive of its step)."""
    rows = mem['rows']
    s = mem['by_key'][start_key]['index']
    r0 = rows[s]
    E_s, H_s = r0['E'], r0['H_in']
    G_s = None if H_s is None else E_s - H_s
    acc = dict((n, 0.0) for n in ('a', 'b', 'c', 'closure', 'dHin', 'G', 'I'))
    missing = {'balance': 0, 'G': 0, 'I': 0}
    out = {start_key: dict(acc, dE=0.0, dHin_direct=0.0 if H_s is not None else None,
                           dG_direct=0.0 if H_s is not None else None, missing=dict(missing))}
    for i in range(s + 1, len(rows)):
        r, p = rows[i], rows[i - 1]
        dT = r['T'] - p['T']
        if r['has_balance']:
            acc['a'] += r['bal_evolution_exact']
            acc['b'] += r['bal_field_work']
            acc['c'] += r['bal_reference_reset']
            acc['closure'] += r['bal_closure']
            acc['dHin'] += r['bal_dHin']
        else:
            missing['balance'] += 1
        if r['balance_d_E_minus_Hin_dT'] is not None:
            acc['G'] += r['balance_d_E_minus_Hin_dT'] * dT
        else:
            missing['G'] += 1
        if r['M1_I_E'] is not None and p['M1_I_E'] is not None:
            acc['I'] += 0.5 * dT * (r['M1_I_E'] + p['M1_I_E'])
        else:
            missing['I'] += 1
        ok_H = H_s is not None and r['H_in'] is not None
        out[r['key']] = dict(acc, dE=r['E'] - E_s,
                             dHin_direct=(r['H_in'] - H_s) if ok_H else None,
                             dG_direct=((r['E'] - r['H_in']) - G_s) if ok_H else None,
                             missing=dict(missing))
    return out


def birth_clocks(mem):
    rows = mem['rows']
    grid = next((r['T_over_rs'] for r in rows if r['n_zeros'] > 0), None)
    first_neg = next((r['T_over_rs'] for r in rows
                      if r['min_N_minus_absv'] is not None and r['min_N_minus_absv'] < 0), None)
    interp = None
    for i in range(1, len(rows)):
        mp, mc = rows[i - 1]['min_N_minus_absv'], rows[i]['min_N_minus_absv']
        if mp is not None and mc is not None and mp >= 0 > mc:
            interp = rows[i - 1]['T_over_rs'] + (rows[i]['T_over_rs'] - rows[i - 1]['T_over_rs']) * mp / (mp - mc)
            break
    R_t_birth = None
    if grid is not None:
        zs = mem['by_key'][clock_key(grid)]['trapping_zeros'] or []
        R_t_birth = max(zs) if zs else None
    return {'grid': grid, 'first_negative': first_neg, 'interpolated': interp,
            'record_spacing': mem['record_spacing'], 'R_t_at_grid_birth': R_t_birth}


def crossing_of_forms(series):
    out = {}
    for f in FORMS:
        prev, found = None, None
        for s in series:
            m = s['min_N_minus_absv']['forms'][f]
            if m is None:
                prev = None
                continue
            if prev is not None and prev[1] >= 0 > m:
                found = prev[0] + (s['T_over_rs'] - prev[0]) * prev[1] / (prev[1] - m)
                break
            prev = (s['T_over_rs'], m)
        out[f] = found
    return out


def choose_clocks(common_keys, births, override=None):
    last = common_keys[-1]
    if override:
        keys = []
        for t in override:
            k = clock_key(t)
            if k not in common_keys:
                raise ValueError('clock T/r_s = %r is not a matched clock' % t)
            keys.append(k)
        return keys, 'override from the command line: %s' % list(override)
    interp = [b['interpolated'] for b in births.values() if b['interpolated'] is not None]
    grid = [b['grid'] for b in births.values() if b['grid'] is not None]
    if len(interp) == 3 and len(grid) == 3:
        earliest, latest = min(interp + grid), max(grid)
        before = max([k for k in common_keys if key_to_clock(k) <= earliest - 1.0] or [common_keys[0]])
        formation = min([k for k in common_keys if key_to_clock(k) >= latest] or [last])
        rule = ('before = the last matched clock at or before (earliest birth %.5f - 1 r_s); formation = the first '
                'matched clock at or after the latest grid birth %.5f (all 3 members carry a surface); '
                'last = the last common clock' % (earliest, latest))
    else:
        before, formation = common_keys[len(common_keys) // 3], common_keys[2 * len(common_keys) // 3]
        rule = 'no birth in every member: thirds of the matched range'
    return [before, formation, last], rule


def compare_args(members):
    allkeys = set()
    for m in MEMBERS:
        allkeys |= set((members[m]['args'] or {}).keys())
    physics, book = {}, {}
    for k in sorted(allkeys):
        vals = dict((m, (members[m]['args'] or {}).get(k, '<absent>')) for m in MEMBERS)
        if len(set(json.dumps(v, sort_keys=True, default=str) for v in vals.values())) > 1:
            (book if k in BOOKKEEPING_ARGS else physics)[k] = vals
    return physics, book


def first_record_differences(members):
    out = {}
    ref = members['4']['first_record']
    for k, v in ref.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            row = {'4': v}
            for m in ('8', '2'):
                o = members[m]['first_record'].get(k)
                row[m + '-4'] = (o - v) if isinstance(o, (int, float)) and not isinstance(o, bool) else None
            out[k] = row
        else:
            out[k] = dict((m, members[m]['first_record'].get(k)) for m in MEMBERS)
    return out


def build_series(members, common_keys, E0, acc_ledger, ledger_start_key, acc_first):
    series = []
    tol = BAND_TOL_REL * abs(E0)
    for k in common_keys:
        T = key_to_clock(k)
        rows = dict((m, members[m]['by_key'][k]) for m in MEMBERS)
        E = dict((m, rows[m]['E']) for m in MEMBERS)
        fE = richardson(E['8'], E['4'], E['2'])
        fR = richardson(rows['8']['E_over_E0'], rows['4']['E_over_E0'], rows['2']['E_over_E0'])
        D = dict((m, 100.0 * (1.0 - E[m] / E0)) for m in MEMBERS)
        ratios = {'D8/D4': safe_div(D['8'], D['4']), 'D4/D2': safe_div(D['4'], D['2'])}
        lam = {}
        for f in FORMS + ('band_lo', 'band_hi'):
            lam[f] = safe_div(fE[f], E['8'])
        lam['spread'] = None if (lam['band_hi'] is None or lam['band_lo'] is None) else lam['band_hi'] - lam['band_lo']
        lam['full_refund'] = safe_div(E0, E['8'])
        refund = dict((f, None if fE[f] is None else 100.0 * (1.0 - fE[f] / E0)) for f in FORMS)
        mN = dict((m, rows[m]['min_N_minus_absv']) for m in MEMBERS)
        fm = richardson(mN['8'], mN['4'], mN['2'])
        in_band = None if fE['band_lo'] is None else (fE['band_lo'] - tol <= E0 <= fE['band_hi'] + tol)
        r = fE['order_ratio']
        order_ok = None if r is None else (ORDER_WINDOW[0] <= r <= ORDER_WINDOW[1])
        entry = {'key': k, 'T_over_rs': T, 'E': E,
                 'E_over_E0': dict((m, rows[m]['E_over_E0']) for m in MEMBERS),
                 'deficit_percent': D, 'deficit_ratios': ratios, 'E_star': fE, 'E_over_E0_star': fR,
                 'E_star_deficit_percent': refund, 'lambda': lam,
                 'min_N_minus_absv': {'values': mN, 'forms': fm},
                 'n_zeros': dict((m, rows[m]['n_zeros']) for m in MEMBERS),
                 'R_t': dict((m, rows[m]['R_t']) for m in MEMBERS),
                 'converged': dict((m, rows[m]['converged']) for m in MEMBERS),
                 'E0_in_band': in_band, 'order_ok': order_ok}
        if acc_ledger is not None and k >= ledger_start_key:
            parts, lin = {}, {}
            for name in PART_NAMES:
                vals = dict((m, acc_ledger[m][k].get(name)) for m in MEMBERS)
                parts[name] = {'values': vals, 'forms': richardson(vals['8'], vals['4'], vals['2'])}
            for m in MEMBERS:
                v = acc_ledger[m][k]
                lin[m] = None if (v.get('dHin') is None or v.get('G') is None) else v['dE'] - (v['dHin'] + v['G'])
            entry['parts'] = parts
            entry['parts_missing'] = dict((m, acc_ledger[m][k]['missing']) for m in MEMBERS)
            entry['ledger_closure_over_window'] = lin
        if acc_first is not None:
            pf = {}
            for name in ('dE', 'dHin_direct', 'dG_direct', 'I'):
                vals = dict((m, acc_first[m][k].get(name)) for m in MEMBERS)
                pf[name] = {'values': vals, 'forms': richardson(vals['8'], vals['4'], vals['2'])}
            entry['parts_from_first_clock'] = pf
        series.append(entry)
    return series


def run(paths, override_clocks=None, allow_gaps=False, out_json=None, out_md=None):
    members = {}
    for m in MEMBERS:
        p = paths[m]
        with open(p, 'r') as f:            # read-only
            d = json.load(f)
        mem = digest_member(d, m, p, allow_gaps=allow_gaps)
        del d
        mem['path'] = p
        mem['sha256'] = sha256_of(p)
        mem['bytes'] = os.path.getsize(p)
        members[m] = mem

    E0s = dict((m, members[m]['E0']) for m in MEMBERS)
    E0_ref = E0s['4'] if E0s['4'] is not None else next((v for v in E0s.values() if v is not None), None)
    if E0_ref is None:
        E0_ref = members['4']['rows'][0]['E']
    E0_spread = max([abs(v - E0_ref) for v in E0s.values() if v is not None] or [0.0])

    physics_diff, bookkeeping_diff = compare_args(members)
    common_keys = sorted(set.intersection(*[set(members[m]['by_key'].keys()) for m in MEMBERS]))
    if not common_keys:
        raise ValueError('the three histories share no clock')

    want = max(members[m]['rows'][members[m]['ledger_start_index']]['key'] for m in MEMBERS)
    later = [k for k in common_keys if k >= want]
    ledger_start_key = later[0] if later else None
    acc_ledger = dict((m, accumulate(members[m], ledger_start_key)) for m in MEMBERS) if ledger_start_key is not None else None
    acc_first = dict((m, accumulate(members[m], common_keys[0])) for m in MEMBERS)
    series = build_series(members, common_keys, E0_ref, acc_ledger, ledger_start_key, acc_first)

    births = dict((m, birth_clocks(members[m])) for m in MEMBERS)
    birth_forms = dict((kind, richardson(births['8'][kind], births['4'][kind], births['2'][kind]))
                       for kind in ('grid', 'first_negative', 'interpolated'))
    birth_from_forms = crossing_of_forms(series)
    chosen_keys, rule = choose_clocks(common_keys, births, override_clocks)
    chosen = [s for s in series if s['key'] in chosen_keys]
    chosen_names = dict(zip(chosen_keys, ('before formation', 'at formation', 'last common clock')))

    # the (8,4) one-form extension past the last common clock
    keys84 = sorted(set(members['8']['by_key']) & set(members['4']['by_key']))
    ext = []
    for k in keys84:
        if k <= common_keys[-1]:
            continue
        r8, r4 = members['8']['by_key'][k], members['4']['by_key'][k]
        f84 = 2.0 * r4['E'] - r8['E']
        D8, D4 = 100.0 * (1.0 - r8['E'] / E0_ref), 100.0 * (1.0 - r4['E'] / E0_ref)
        ext.append({'T_over_rs': key_to_clock(k), 'E_8': r8['E'], 'E_4': r4['E'], 'D_8_percent': D8,
                    'D_4_percent': D4, 'D8_over_D4': safe_div(D8, D4), 'f84': f84,
                    'f84_deficit_percent': 100.0 * (1.0 - f84 / E0_ref), 'lambda_84': f84 / r8['E'],
                    'lambda_full_refund': E0_ref / r8['E']})

    # the paper's deficits at 21.5 and the current's share of the deficit
    paper = {'clock': PAPER_CLOCK, 'present': False}
    kp = clock_key(PAPER_CLOCK)
    if kp in set(common_keys):
        s = next(x for x in series if x['key'] == kp)
        paper['present'] = True
        paper['recomputed_percent'] = dict((m, 100.0 * (1.0 - s['E_over_E0'][m])) for m in MEMBERS)
        paper['recomputed_percent_head_E0'] = s['deficit_percent']
        paper['paper_percent'] = dict(PAPER_DEFICITS)
        paper['difference_percent_points'] = dict((m, paper['recomputed_percent'][m] - PAPER_DEFICITS[m]) for m in MEMBERS)
        paper['ratios_recomputed'] = s['deficit_ratios']
        paper['ratios_paper'] = dict(PAPER_RATIOS)
        share = {}
        for m in MEMBERS:
            I = acc_first[m][kp]['I']
            deficit = E0_ref - s['E'][m]
            share[m] = {'integral_I_E_from_first_clock': I, 'deficit_E0_minus_E': deficit,
                        'share_percent': safe_div(100.0 * I, deficit), 'paper_share_percent': PAPER_CURRENT_SHARE[m]}
        paper['current_share'] = share
        paper['series_entry'] = s

    early = []
    want_T = set(x['T_over_rs'] for x in series[1:7]) | set([1.0, 2.0, 4.0, 8.0])
    for s in series:
        if s['T_over_rs'] in want_T:
            early.append({'T_over_rs': s['T_over_rs'], 'd84': s['E_star']['d84'], 'd42': s['E_star']['d42'],
                          'order_ratio': s['E_star']['order_ratio']})
    late = [s for s in series if s['T_over_rs'] >= 1.0]
    acceptance = {'error_rule': 'the stated error at a clock is the full spread (max - min) of the 3 forms; the band '
                                '[min, max] is what "within the fit\'s stated error" refers to; the quoted central '
                                'value is 2E_2 - E_4 (the two smallest steps)',
                  'reading': 'the paper\'s "halves per halving" law has zero deficit at zero step, i.e. E_* = E0 (P12 '
                             'claim 9\'s full refund): acceptance = E0 in the band and the order test in [1.5, 2.5] '
                             'at each of the 3 clocks',
                  'per_clock': [], 'n_late_clocks': len(late),
                  'late_fraction_E0_in_band': safe_div(sum(1 for s in late if s['E0_in_band']), len(late)),
                  'late_fraction_order_ok': safe_div(sum(1 for s in late if s['order_ok']), len(late))}
    n_pass = 0
    for s in chosen:
        ok = bool(s['E0_in_band']) and bool(s['order_ok'])
        n_pass += int(ok)
        acceptance['per_clock'].append({'T_over_rs': s['T_over_rs'], 'name': chosen_names[s['key']],
                                        'E_star_central_f42': s['E_star']['f42'], 'spread': s['E_star']['spread'],
                                        'band': [s['E_star']['band_lo'], s['E_star']['band_hi']], 'E0': E0_ref,
                                        'E0_in_band': s['E0_in_band'], 'order_ratio': s['E_star']['order_ratio'],
                                        'order_ok': s['order_ok'], 'pass': ok})
    acceptance['n_pass_of_3'] = n_pass
    acceptance['verdict'] = 'PASS' if n_pass == len(chosen) else ('PARTIAL' if n_pass else 'FAIL')

    members_meta = {}
    for m in MEMBERS:
        mm = members[m]
        members_meta[m] = dict((k, mm[k]) for k in ('member', 'tau', 'rs', 'script', 'args', 'E0', 'head',
                                                   'first_record', 'n_records', 'clock_first', 'clock_last',
                                                   'record_spacing', 'coverage', 'checks', 'collar_changes',
                                                   'cut_changes', 'n_cut_changed_flags', 'unconverged_clocks',
                                                   'engines', 'lz_label', 'continuation_census', 'path', 'sha256',
                                                   'bytes'))
        members_meta[m]['ledger_start_clock_member'] = mm['rows'][mm['ledger_start_index']]['T_over_rs']
        members_meta[m]['n_unconverged'] = len(mm['unconverged_clocks'])
    R = {'meta': {'script': SCRIPT, 'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'python': sys.version.split()[0], 'argv': sys.argv, 'out_json': out_json, 'out_md': out_md,
                  'inputs': dict((m, paths[m]) for m in MEMBERS)},
         'protocol': {'forms': FORM_LABEL, 'order_test': '(X_8 - X_4)/(X_4 - X_2) against 2 (4 for pure second order)',
                      'order_window': list(ORDER_WINDOW), 'matched_clock_key': 'round(T/r_s * %d)' % KEY_SCALE},
         'key_mapping': [{'p12': a, 'record': b, 'status': c} for a, b, c in KEY_MAPPING],
         'E0': {'reference': E0_ref, 'per_member_head': E0s, 'max_abs_spread': E0_spread},
         'version_check': {'physics_args_differing': physics_diff, 'bookkeeping_args_differing': bookkeeping_diff,
                           'first_record_differences_vs_tau4': first_record_differences(members),
                           'early_matched_clock_order_test': early},
         'clocks': {'n_matched': len(common_keys), 'matched_first': key_to_clock(common_keys[0]),
                    'matched_last': key_to_clock(common_keys[-1]),
                    'ledger_window_start': None if ledger_start_key is None else key_to_clock(ledger_start_key),
                    'chosen': [key_to_clock(k) for k in chosen_keys], 'chosen_names': list(chosen_names.values()),
                    'rule': rule},
         'birth': {'per_member': births, 'forms': birth_forms, 'from_min_N_minus_absv_forms': birth_from_forms},
         'chosen_clock_table': chosen, 'paper_check': paper, 'extension_84_past_last_common': ext,
         'acceptance': acceptance, 'members': members_meta, 'series': series}
    return R


def sanitize(x):
    if isinstance(x, dict):
        return dict((str(k), sanitize(v)) for k, v in x.items())
    if isinstance(x, (list, tuple)):
        return [sanitize(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted(sanitize(v) for v in x)
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def fnum(x, nd=4):
    if x is None:
        return '-'
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    if isinstance(x, int):
        return str(x)
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(v) or math.isinf(v):
        return str(v)
    return ('%.' + str(nd) + 'f') % v


def md_table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    for r in rows:
        lines.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return lines


def write_json(path, R):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(sanitize(R), f, indent=1)


def write_md(path, R):
    L = []
    w = L.append
    E0 = R['E0']['reference']
    w('# charge_drift_fits_v1 - icloud item 10: the charge-drift fits (P12 claims 8, 9)')
    w('')
    w('Generated %s by %s (busi lane). Inputs read-only. Results json: `%s`.' % (R['meta']['generated_utc'], SCRIPT,
                                                                                R['meta']['out_json']))
    w('')
    w('## 1. Inputs')
    w('')
    rows = []
    for m in MEMBERS:
        mm = R['members'][m]
        eng = ', '.join('%s [%s, %s]' % (e['engine'], fnum(e['from'], 3), fnum(e['to'], 3)) for e in mm['engines'])
        rows.append(['tau %s' % m, '`%s`' % mm['path'], mm['sha256'][:16], mm['script'], eng, mm['n_records'],
                     '%s to %s' % (fnum(mm['clock_first'], 5), fnum(mm['clock_last'], 5)), fnum(mm['record_spacing'], 5)])
    L += md_table(['member', 'file', 'sha256[:16]', 'script', 'record engine [from, to]', 'records', 'T/r_s range',
                   'spacing'], rows)
    w('')
    w('E0 (head, tau 4 member) = %.12f; the 3 heads agree to %.2e.' % (E0, R['E0']['max_abs_spread']))
    rs8 = R['members']['8']['head'].get('resume')
    if rs8:
        w('tau 8 head.resume: from %s (%s) at T/r_s %s, %s records replayed, max_eps_rel %s, max_J_rel %s.' % (
            rs8.get('from'), rs8.get('script'), rs8.get('T_over_rs'), rs8.get('records_replayed'),
            fnum(rs8.get('max_eps_rel'), 2) if rs8.get('max_eps_rel') is None else '%.2e' % rs8['max_eps_rel'],
            '%.2e' % rs8['max_J_rel'] if rs8.get('max_J_rel') is not None else '-'))
    w('')
    w('## 2. Protocol, error rule, acceptance reading (stated before the numbers)')
    w('')
    w('P12 claim 8: at each matched clock (a T/r_s common to the 3 histories), with X_tau the value on the member of '
      'step tau, the forms X_*(42) = 2X_2 - X_4 and X_*(84) = 2X_4 - X_8 (each exact for X_tau = X_* + c1 tau) and '
      'X_*(2nd) = (X_8 - 6X_4 + 8X_2)/3 (exact for X_* + c1 tau + c2 tau^2); the order test (X_8 - X_4)/(X_4 - X_2), '
      '2 for first order and 4 for pure second order; the forms are an extrapolation only inside [%.1f, %.1f]. '
      'Applied to E and E/E0, to the accumulated ledger parts the records carry, to min(N - |v|) and to the birth '
      'clock; not to a newborn radius (an O(tau) field error moves a double root by O(sqrt tau)).' % ORDER_WINDOW)
    w('')
    w('**Error rule.** %s.' % R['acceptance']['error_rule'])
    w('')
    w('**Acceptance reading.** %s. For the parts, 0 in the band of a (frozen-field evolution) and of c (the '
      'reference reset; zero for a commuting continued reference, P12 claim 1) is reported as information, not as '
      'acceptance.' % R['acceptance']['reading'])
    w('')
    w('## 3. Key mapping (P12 names -> record keys)')
    w('')
    L += md_table(['P12', 'record key', 'status'], [[a, b, c] for a, b, c in KEY_MAPPING])
    w('')
    w('Plainly: of b, c, Delta W, current and bridge, the records carry b (field_work), c (reference_reset) and the '
      'current (M1_I_E); they do NOT carry Delta W or the bridge separately, only their sum Delta(E - H_in). '
      'a (evolution_exact) is also carried. Ledger accumulation window: from T/r_s %s (the first matched clock at '
      'which all 3 members carry the per-step keys; per member: %s).' % (
          fnum(R['clocks']['ledger_window_start'], 5),
          ', '.join('tau %s: %s' % (m, fnum(R['members'][m]['ledger_start_clock_member'], 5)) for m in MEMBERS)))
    w('')
    w('## 4. The v25 tau 2 member against the v19 members: does it matter for the fit; what the records settle')
    w('')
    vc = R['version_check']
    w('- Physics args that differ (other than tau): %s' % (json.dumps(vc['physics_args_differing'], default=str)
                                                            if vc['physics_args_differing'] else 'none'))
    w('- Bookkeeping args that differ: %s' % ', '.join(sorted(vc['bookkeeping_args_differing'].keys())))
    for m in MEMBERS:
        mm = R['members'][m]
        c = mm['continuation_census']
        lz = mm['lz_label']
        w('- tau %s: script %s; head site_fields "%s"; continuation modes %s; records with nsub > 1: %d (first %s); '
          'with m_eff > 1: %d (first %s); LZ-convention label first at T/r_s %s (label says from %s); unconverged '
          'records %d%s; collar changes %d%s; cut changes %d; cut_changed flags %d.' % (
              m, mm['script'], mm['head'].get('site_fields'), c['modes'], c['records_with_nsub_gt1'],
              fnum(c['first_clock_nsub_gt1'], 3), c['records_with_meff_gt1'], fnum(c['first_clock_meff_gt1'], 3),
              fnum(None if lz is None else lz['first_record_clock'], 3),
              fnum(None if lz is None else lz['label_from_T_over_rs'], 3), mm['n_unconverged'],
              (' (first %s)' % fnum(mm['unconverged_clocks'][0], 3)) if mm['unconverged_clocks'] else '',
              len(mm['collar_changes']),
              (' (%s)' % '; '.join('T/r_s %s: %s -> %s, dE_record %s' % (fnum(x['T_over_rs'], 3), x['from'], x['to'],
                                                                           fnum(x['dE_record'], 6))
                                   for x in mm['collar_changes'][:4])) if mm['collar_changes'] else '',
              len(mm['cut_changes']), mm['n_cut_changed_flags']))
    w('')
    w('First records (T/r_s 0) against the tau 4 member (differences; strings listed):')
    w('')
    rows = []
    for k, v in vc['first_record_differences_vs_tau4'].items():
        if isinstance(v, dict) and '8-4' in v:
            rows.append([k, fnum(v['4'], 10), '%.3e' % v['8-4'] if v['8-4'] is not None else '-',
                         '%.3e' % v['2-4'] if v['2-4'] is not None else '-'])
        else:
            rows.append([k, str(v.get('4'))[:60], str(v.get('8'))[:60], str(v.get('2'))[:60]])
    L += md_table(['field', 'tau 4', 'tau 8 - tau 4', 'tau 2 - tau 4'], rows)
    w('')
    w('Order test at early matched clocks (one first-order coefficient shared by the 3 members gives 2 from the start):')
    w('')
    L += md_table(['T/r_s', 'E_8 - E_4', 'E_4 - E_2', '(E_8-E_4)/(E_4-E_2)'],
                  [[fnum(e['T_over_rs'], 4), '%.3e' % e['d84'] if e['d84'] is not None else '-',
                    '%.3e' % e['d42'] if e['d42'] is not None else '-', fnum(e['order_ratio'], 3)]
                   for e in vc['early_matched_clock_order_test']])
    w('')
    w('Reading. The forms assume one error expansion X_tau = X_* + c1 tau + c2 tau^2 across the 3 members. A version '
      'offset delta at fixed tau (v25 against v19 at tau 2) enters 2E_2 - E_4 as 2 delta and the two-term form as '
      '8 delta/3, so it matters at the level of the band when |delta| is comparable to the spread. What the records '
      'settle: (i) the T/r_s 0 records agree to the listed differences in E, the fields and the solver residuals, so '
      'the preparation and the static solve are the same; (ii) a different step algorithm can show only in the '
      'drift, and the early order ratios above are the test (2 for a shared first-order coefficient); (iii) the '
      'continuation convention differs (v19: windows and sub-steps in the counted records; v25: one step in every '
      'record): the reference reset c is the part the convention touches, so its order ratio in section 8 is the '
      'sharpest check; (iv) the tau 8 member is itself c17 (v17) to T/r_s 11.0 and v19 after. What the records cannot '
      'settle: a tau 2 member on v19 (or tau 4 on v25) is the only decisive check; absent it, the fit\'s band is the '
      'honest error and the version difference is one of its sources.')
    w('')
    w('## 5. Matched clocks, the 3 printed clocks, the birth clocks')
    w('')
    ck = R['clocks']
    w('Matched clocks: %d, T/r_s %s to %s. Printed clocks: %s (%s). Rule: %s.' % (
        ck['n_matched'], fnum(ck['matched_first'], 5), fnum(ck['matched_last'], 5),
        ', '.join(fnum(t, 5) for t in ck['chosen']), ', '.join(ck['chosen_names']), ck['rule']))
    w('')
    b = R['birth']
    rows = []
    for kind in ('grid', 'first_negative', 'interpolated'):
        f = b['forms'][kind]
        rows.append([kind] + [fnum(b['per_member'][m][kind], 5) for m in MEMBERS]
                    + [fnum(f['f42'], 5), fnum(f['f84'], 5), fnum(f['f2nd'], 5), fnum(f['order_ratio'], 3)])
    rows.append(['R_t at grid birth (not extrapolated)'] + [fnum(b['per_member'][m]['R_t_at_grid_birth'], 4)
                                                             for m in MEMBERS] + ['-', '-', '-', '-'])
    rows.append(['record spacing (quantization)'] + [fnum(b['per_member'][m]['record_spacing'], 5) for m in MEMBERS]
                + ['-', '-', '-', '-'])
    L += md_table(['birth clock T/r_s', 'tau 8', 'tau 4', 'tau 2', 'f42', 'f84', 'f2nd', 'order ratio'], rows)
    w('')
    w('Zero crossing of the Richardson-extrapolated min(N - |v|) series: f42 %s, f84 %s, f2nd %s (T/r_s).' % tuple(
        fnum(b['from_min_N_minus_absv_forms'][f], 5) for f in FORMS))
    w('')
    w('## 6. E at the 3 clocks: the 3 forms, the band, the order test')
    w('')
    rows = []
    for s in R['chosen_clock_table']:
        f = s['E_star']
        rows.append([fnum(s['T_over_rs'], 5)] + [fnum(s['E'][m], 4) for m in MEMBERS]
                    + [fnum(s['deficit_percent'][m], 3) for m in MEMBERS]
                    + [fnum(f['f42'], 4), fnum(f['f84'], 4), fnum(f['f2nd'], 4),
                       '[%s, %s]' % (fnum(f['band_lo'], 4), fnum(f['band_hi'], 4)), fnum(f['spread'], 4)])
    L += md_table(['T/r_s', 'E_8', 'E_4', 'E_2', 'D_8 %', 'D_4 %', 'D_2 %', 'E*(42)', 'E*(84)', 'E*(2nd)', 'band',
                   'spread'], rows)
    w('')
    rows = []
    for s in R['chosen_clock_table']:
        f, d = s['E_star'], s['E_star_deficit_percent']
        rows.append([fnum(s['T_over_rs'], 5), fnum(d['f42'], 3), fnum(d['f84'], 3), fnum(d['f2nd'], 3),
                     fnum(100.0 * f['spread'] / E0 if f['spread'] is not None else None, 3),
                     fnum(f['order_ratio'], 3), fnum(s['deficit_ratios']['D8/D4'], 3),
                     fnum(s['deficit_ratios']['D4/D2'], 3), fnum(s['E0_in_band']), fnum(s['order_ok']),
                     ', '.join('%s:%s' % (m, fnum(s['converged'][m])) for m in MEMBERS)])
    L += md_table(['T/r_s', 'deficit of E*(42) %', 'E*(84) %', 'E*(2nd) %', 'spread/E0 %', 'r=(E8-E4)/(E4-E2)',
                   'D8/D4', 'D4/D2', 'E0 in band', 'order ok', 'converged 8,4,2'], rows)
    w('')
    w('## 7. lambda(T) = E_*/E_8 read against the 10 percent deficit (P12 claim 9, a sensitivity map)')
    w('')
    rows = []
    for s in R['chosen_clock_table']:
        lam = s['lambda']
        lo, hi = lam['band_lo'], lam['band_hi']
        rows.append([fnum(s['T_over_rs'], 5), fnum(s['deficit_percent']['8'], 3), fnum(lam['f42'], 4),
                     fnum(lam['f84'], 4), fnum(lam['f2nd'], 4), '[%s, %s]' % (fnum(lo, 4), fnum(hi, 4)),
                     fnum(lam['full_refund'], 4),
                     '[%s, %s]' % (fnum(lo, 4), fnum(hi, 4)),
                     '[%s, %s]' % (fnum(safe_div(1.0, hi), 4), fnum(safe_div(1.0, lo), 4)),
                     '[%s, %s]' % (fnum(safe_div(1.0, hi * hi) if hi else None, 4), fnum(safe_div(1.0, lo * lo) if lo else None, 4)),
                     '[%s, %s]' % (fnum(lo * lo if lo is not None else None, 4), fnum(hi * hi if hi is not None else None, 4))])
    L += md_table(['T/r_s', 'D_8 %', 'lambda(42)', 'lambda(84)', 'lambda(2nd)', 'lambda band',
                   'full refund E0/E_8', '2GE factor', '1/(4GE) comparator factor', 'luminosity factor',
                   'area factor'], rows)
    w('')
    s_last = R['chosen_clock_table'][-1]
    w('At the last common clock (T/r_s %s) D_8 = %s percent, so a full refund gives lambda = 1/(1 - D_8) = %s; the '
      'measured band is [%s, %s]. P12 claim 9\'s illustration lambda = %.5f belongs to E_8/E0 = %.3f, a deficit the '
      'tau 8 member reaches only past the matched range (its last record: E/E0 = %s at T/r_s %s). R/(2GE) carries '
      'the root\'s own factor zeta, not measured here (a held root gives 1/lambda).' % (
          fnum(s_last['T_over_rs'], 5), fnum(s_last['deficit_percent']['8'], 3), fnum(s_last['lambda']['full_refund'], 4),
          fnum(s_last['lambda']['band_lo'], 4), fnum(s_last['lambda']['band_hi'], 4), P12_ILLUSTRATION['lambda'],
          P12_ILLUSTRATION['E8_over_E0'],
          fnum(R['extension_84_past_last_common'][-1]['E_8'] / E0 if R['extension_84_past_last_common'] else None, 4),
          fnum(R['extension_84_past_last_common'][-1]['T_over_rs'] if R['extension_84_past_last_common'] else None, 3)))
    w('')
    w('## 8. The accumulated ledger parts at the 3 clocks (window from T/r_s %s)' % fnum(ck['ledger_window_start'], 5))
    w('')
    for s in R['chosen_clock_table']:
        w('T/r_s %s (%s):' % (fnum(s['T_over_rs'], 5), chosen_name(R, s)))
        w('')
        if 'parts' not in s:
            w('(before the ledger window; no parts)')
            w('')
            continue
        rows = []
        for name in PART_NAMES:
            p = s['parts'][name]
            f = p['forms']
            zero_in = None if f['band_lo'] is None else (f['band_lo'] <= 0.0 <= f['band_hi'])
            rows.append([PART_LABEL[name]] + [fnum(p['values'][m], 6) for m in MEMBERS]
                        + [fnum(f['f42'], 6), fnum(f['f84'], 6), fnum(f['f2nd'], 6), fnum(f['order_ratio'], 3),
                           fnum(zero_in)])
        L += md_table(['part (accumulated)', 'tau 8', 'tau 4', 'tau 2', 'f42', 'f84', 'f2nd', 'order ratio',
                       '0 in band'], rows)
        w('')
        w('Ledger closure over the window, dE - (a+b+c + G) per member: %s; records lacking a balance in the window: '
          '%s.' % (', '.join('%s: %s' % (m, '%.3e' % s['ledger_closure_over_window'][m]
                                         if s['ledger_closure_over_window'][m] is not None else '-') for m in MEMBERS),
                   ', '.join('%s: %d' % (m, s['parts_missing'][m]['balance']) for m in MEMBERS)))
        w('')
    w('## 9. min(N - |v|) at the 3 clocks')
    w('')
    rows = []
    for s in R['chosen_clock_table']:
        v, f = s['min_N_minus_absv']['values'], s['min_N_minus_absv']['forms']
        rows.append([fnum(s['T_over_rs'], 5)] + [fnum(v[m], 6) for m in MEMBERS]
                    + [fnum(f['f42'], 6), fnum(f['f84'], 6), fnum(f['f2nd'], 6), fnum(f['order_ratio'], 3),
                       ', '.join('%s:%s' % (m, s['n_zeros'][m]) for m in MEMBERS)])
    L += md_table(['T/r_s', 'tau 8', 'tau 4', 'tau 2', 'f42', 'f84', 'f2nd', 'order ratio', 'zeros 8,4,2'], rows)
    w('')
    w('## 10. The paper\'s deficits at T/r_s %s recomputed from E/E0; the cut current\'s share' % fnum(PAPER_CLOCK, 1))
    w('')
    pc = R['paper_check']
    if pc.get('present'):
        rows = [['tau %s' % m, fnum(pc['recomputed_percent'][m], 4), fnum(PAPER_DEFICITS[m], 2),
                 '%+.4f' % pc['difference_percent_points'][m]] for m in MEMBERS]
        L += md_table(['member', 'recomputed 100(1 - E/E0) %', 'paper %', 'difference (points)'], rows)
        w('')
        w('Ratios recomputed: D8/D4 = %s (paper %.2f), D4/D2 = %s (paper %.2f).' % (
            fnum(pc['ratios_recomputed']['D8/D4'], 4), PAPER_RATIOS['D8/D4'],
            fnum(pc['ratios_recomputed']['D4/D2'], 4), PAPER_RATIOS['D4/D2']))
        w('')
        rows = []
        for m in MEMBERS:
            cs = pc['current_share'][m]
            rows.append(['tau %s' % m, '%.4e' % cs['integral_I_E_from_first_clock'], fnum(cs['deficit_E0_minus_E'], 4),
                         fnum(cs['share_percent'], 4), fnum(cs['paper_share_percent'], 3)])
        L += md_table(['member', 'int_0^T M1_I_E dT', 'E0 - E(T)', 'share %', 'sentence 5 share %'], rows)
        w('')
        w('If the recomputed shares do not reproduce 1.6, 0.2, 0.002 percent, the mapping of "current" to M1_I_E (or '
          'its sign or normalization) is the open question, not the fit.')
    else:
        w('T/r_s %s is not a matched clock in these records.' % fnum(PAPER_CLOCK, 1))
    w('')
    w('## 11. The matched-clock series (every 2 r_s and the printed clocks; the full series is in the json)')
    w('')
    rows = []
    chosen_set = set(s['key'] for s in R['chosen_clock_table'])
    for s in R['series']:
        if s['key'] % (2 * KEY_SCALE) == 0 or s['key'] in chosen_set:
            f, lam = s['E_star'], s['lambda']
            rows.append([fnum(s['T_over_rs'], 4)] + [fnum(s['deficit_percent'][m], 3) for m in MEMBERS]
                        + [fnum(s['E_star_deficit_percent']['f42'], 3),
                           '[%s, %s]' % (fnum(100.0 * (1.0 - f['band_hi'] / E0) if f['band_hi'] is not None else None, 3),
                                         fnum(100.0 * (1.0 - f['band_lo'] / E0) if f['band_lo'] is not None else None, 3)),
                           fnum(f['order_ratio'], 3), fnum(lam['f42'], 4),
                           '[%s, %s]' % (fnum(lam['band_lo'], 4), fnum(lam['band_hi'], 4)),
                           fnum(lam['full_refund'], 4), fnum(s['E0_in_band']), fnum(s['order_ok'])])
    L += md_table(['T/r_s', 'D_8 %', 'D_4 %', 'D_2 %', 'deficit of E*(42) %', 'band of E* deficit %', 'r',
                   'lambda(42)', 'lambda band', 'E0/E_8', 'E0 in band', 'order ok'], rows)
    w('')
    if R['extension_84_past_last_common']:
        w('Past the last common clock only the (8,4) pair exists: one form, no band (at T/r_s 21.5 the (8,4) form sat '
          '%s percent of E0 from the (4,2) form).' % fnum(
              100.0 * (s_last['E_star']['f84'] - s_last['E_star']['f42']) / E0, 3))
        w('')
        rows = []
        for e in R['extension_84_past_last_common']:
            if abs(e['T_over_rs'] - round(e['T_over_rs'])) < 1e-9 and int(round(e['T_over_rs'])) % 4 == 0:
                rows.append([fnum(e['T_over_rs'], 3), fnum(e['D_8_percent'], 3), fnum(e['D_4_percent'], 3),
                             fnum(e['D8_over_D4'], 3), fnum(e['f84_deficit_percent'], 3), fnum(e['lambda_84'], 4),
                             fnum(e['lambda_full_refund'], 4)])
        L += md_table(['T/r_s', 'D_8 %', 'D_4 %', 'D8/D4', 'deficit of 2E_4-E_8 %', 'lambda(84)', 'E0/E_8'], rows)
        w('')
    w('## 12. Checks of the mapping (maxima of per-record residuals)')
    w('')
    rows = []
    for m in MEMBERS:
        c = R['members'][m]['checks']
        rows.append(['tau %s' % m, c['n_balance_records']] + [('%.2e' % c[k]) if c[k] is not None else '-' for k in
                                                              ('dE_rate_resid_max', 'dHin_rate_resid_max',
                                                               'G_rate_resid_max', 'dHin_balance_resid_max',
                                                               'balance_sum_resid_max',
                                                               'field_work_formula_resid_max')])
    L += md_table(['member', 'balance records', 'dE_dT*dT - dE', 'dHin_dT*dT - dH_in', 'balance rate*dT - d(E-H_in)',
                   'M1_balance.dHin - dH_in', 'dHin - (a+b+c+closure)', 'field_work - ((Y2-Y1)-(X2-X1))'], rows)
    w('')
    for m in MEMBERS:
        cov = R['members'][m]['coverage']
        w('- tau %s coverage: %s.' % (m, '; '.join('%s: %s' % (k, ('from T/r_s %s, %d gap(s)' % (
            fnum(v['first_clock'], 5), v['n_gaps_after_first'])) if v['carried'] else 'not carried')
                                                   for k, v in cov.items())))
    w('')
    w('## 13. Acceptance')
    w('')
    for a in R['acceptance']['per_clock']:
        w('- T/r_s %s (%s): E_*(42) = %s, spread %s, band [%s, %s]; E0 = %s in band: %s; order ratio %s ok: %s -> %s.' % (
            fnum(a['T_over_rs'], 5), a['name'], fnum(a['E_star_central_f42'], 4), fnum(a['spread'], 4),
            fnum(a['band'][0], 4), fnum(a['band'][1], 4), fnum(a['E0'], 4), fnum(a['E0_in_band']),
            fnum(a['order_ratio'], 3), fnum(a['order_ok']), 'pass' if a['pass'] else 'fail'))
    w('')
    w('Verdict under the stated rule: **%s** (%d of %d clocks). Over the %d matched clocks with T/r_s >= 1: E0 in band '
      'at a fraction %s, order test ok at %s.' % (R['acceptance']['verdict'], R['acceptance']['n_pass_of_3'],
                                                   len(R['acceptance']['per_clock']), R['acceptance']['n_late_clocks'],
                                                   fnum(R['acceptance']['late_fraction_E0_in_band'], 3),
                                                   fnum(R['acceptance']['late_fraction_order_ok'], 3)))
    w('')
    w('## 14. What the result can and cannot say')
    w('')
    w('- Can: the zero-step collar charge E_*(T) at matched coordinate clocks with the spread of 3 forms as its error; '
      'whether the first-order law (zero deficit at zero step) is inside that error; lambda(T) = E_*/E_8, i.e. the '
      'factor by which 2GE(t), the 1/(4GE) comparator (1/lambda), the massless luminosity comparator (lambda^-2) and '
      'charge-radius areas (lambda^2) move if the deficit is refunded; which ledger parts converge with the step and '
      'which carry the first-order drift; the birth clock and min(N - |v|) extrapolated.')
    w('- Cannot: P12 claim 9\'s map is a sensitivity map, not a corrected result (a corrected 2GE needs the zero-step '
      'fields, and the roots move as O(sqrt tau)); no radius, closing time, lifetime or late-time law is extrapolated '
      'or implied (the paper prints none, and sentence 5 prints only the deficit\'s order); the matched clocks are '
      'coordinate clocks, not proper times; the 3 members are not on one engine version (section 4), so a version '
      'offset is a component of the band; the ledger accumulation starts at the window in section 3, not before; '
      'Delta W and the bridge are not separated by the records.')
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')


def chosen_name(R, s):
    try:
        i = [clock_key(t) for t in R['clocks']['chosen']].index(s['key'])
        return R['clocks']['chosen_names'][i]
    except (ValueError, IndexError):
        return ''


def main(argv=None):
    ap = argparse.ArgumentParser(description='icloud item 10: the tau 8/4/2 charge-drift fits (P12 claims 8, 9).')
    ap.add_argument('--tau8', required=True, help='tau 8 history json (read-only)')
    ap.add_argument('--tau4', required=True, help='tau 4 history json (read-only)')
    ap.add_argument('--tau2', required=True, help='tau 2 history json (read-only)')
    ap.add_argument('--out-json', default=DEFAULT_OUT_JSON)
    ap.add_argument('--out-md', default=DEFAULT_OUT_MD)
    ap.add_argument('--clocks', default=None,
                    help='override the 3 printed clocks: comma-separated T/r_s values that are matched clocks')
    ap.add_argument('--allow-gaps', action='store_true',
                    help='tolerate ledger keys missing after their first appearance (default: fail loudly)')
    a = ap.parse_args(argv)
    override = None
    if a.clocks:
        override = [float(x) for x in a.clocks.split(',') if x.strip()]
        if len(override) != 3:
            ap.error('--clocks needs exactly 3 values')
    R = run({'8': a.tau8, '4': a.tau4, '2': a.tau2}, override_clocks=override, allow_gaps=a.allow_gaps,
            out_json=a.out_json, out_md=a.out_md)
    write_json(a.out_json, R)
    write_md(a.out_md, R)
    print('%s: %d matched clocks (T/r_s %s to %s); printed clocks %s; ledger window from %s' % (
        SCRIPT, R['clocks']['n_matched'], fnum(R['clocks']['matched_first'], 5), fnum(R['clocks']['matched_last'], 5),
        [fnum(t, 5) for t in R['clocks']['chosen']], fnum(R['clocks']['ledger_window_start'], 5)))
    for acc in R['acceptance']['per_clock']:
        print('  T/r_s %s: E*(42) %s band [%s, %s] E0 %s in band %s; order ratio %s -> %s' % (
            fnum(acc['T_over_rs'], 5), fnum(acc['E_star_central_f42'], 4), fnum(acc['band'][0], 4),
            fnum(acc['band'][1], 4), fnum(acc['E0'], 4), fnum(acc['E0_in_band']), fnum(acc['order_ratio'], 3),
            'pass' if acc['pass'] else 'fail'))
    pc = R['paper_check']
    if pc.get('present'):
        print('  deficits at T/r_s %.1f recomputed: %s (paper 9.15, 5.00, 2.44)' % (
            PAPER_CLOCK, ', '.join('%.3f' % pc['recomputed_percent'][m] for m in MEMBERS)))
    print('  verdict: %s; wrote %s and %s' % (R['acceptance']['verdict'], a.out_json, a.out_md))
    return 0


if __name__ == '__main__':
    sys.exit(main())
