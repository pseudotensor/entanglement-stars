#!/usr/bin/env python3
"""eta_handoff.py -- the busi lane's hand-off of the directional planar coefficients to the hreduce lane.

Reads the final sweep table (eta_table_v1.tsv format: comment lines beginning with #, one header line,
one row per primitive orbit with columns p1 p2 p3 pnorm mass eta eta_hw eta_lev0 eta_lev1 levels ...;
eta_lev0 and eta_lev1 are the 2 finest levels used), applies the delivered half-width rule of
ETA_judgment_v1.md section 4, substitutes the exact [100] and [111] values, and writes
  * the hand-off table (one row per orbit; the columns the hreduce parser reads keep their names:
    p1 p2 p3 pnorm mass eta eta_hw, with eta_hw the DELIVERED half-width),
  * a short JSON with the exact values, their provenance and the rule's parameters.

The rule (defaults; every factor and floor is an option):
  exact rows [100], [111]: eta = exact value to 12 decimals, eta_hw = 1e-12 (covers the rounding);
  generic rows:            eta_hw = generic_factor * measured half-width (generic_factor = 1);
  family rows (p1 = p2 > p3, the [110]-[111] edge) with 3 levels: max(measured, factor3*|d_last|, floor3)
                           with factor3 = 1, floor3 = 0;
  family rows with 2 levels: max(measured, factor2*|d_last|, floor2) with factor2 = 0, floor2 = 2e-6.
Pure standard library. Run --selftest before use.
"""
import argparse
import json
import math
import os
import re
import statistics
import sys
import tempfile
from decimal import Decimal, ROUND_HALF_EVEN

# ---------------------------------------------------------------------------------------------- exact values
# [100]: ETA_spec_v1.md section 6, 2 mpmath routes (eta_exact_v2.log); [111]: route 1 lies inside the
# hreduce lane's outward-certified interval (web40e_h1_sphere_bounds.tex, eta111_certificate.json).
EXACT = {
    (1, 0, 0): dict(
        eta='0.78658338655313811731',
        route1_1d_25digits='0.78658338655313811731',
        route2_2d_polar_20digits='0.78658338655313811731',
        routes_differ=6.76e-22,
        hreduce_bounded_value='0.7865833865531384',
        status='2 independent evaluations agree to 6.8e-22; the hreduce bounded value agrees to its last '
               'printed digit (3e-16); not an interval certificate'),
    (1, 1, 1): dict(
        eta='0.92338180595796093742',
        route1_1d_25digits='0.92338180595796093742',
        route2_2d_polar_20digits='0.92338180595757305945',
        routes_differ=-3.88e-13,
        hreduce_certified_interval=['0.9233818059579609366', '0.9233818059579609427'],
        status='route 1 lies inside the hreduce certified interval; route 2 agrees only to 3.9e-13 '
               '(its z -> 3 endpoint carries an uncontrolled error at that level)'),
}
B100_INFO = dict(ours_30digits='-0.417518091239732166', hreduce_bounded='-0.4175180912397777254',
                 difference=4.6e-14, note='information only; the massive extension is not part of this hand-off')

MEASURED_RULE = ("levels(h,h/2): eta = eta1 + (eta1-eta0)/10; hw = |eta1-eta0|/4 + chain_err(1.5 cL recompute) "
                 "+ ref_err/(3|p|) + 1e-9; valid if the grid error decreases at least as h^2 between the levels "
                 "(h^4 log h observed in all 34 tested generic orbits; exact [100] and [111] reproduced to 1e-9)")

FAMILY_TEXT = ("p1 = p2 > p3 (canonical orbit), the [110]-[111] edge, [110] included, [111] excluded: the two fast "
               "sines are degenerate, their fully dimerized line psi = pi/2 is a second non-analytic locus the node "
               "subtraction does not treat, and the level-0 grid (23 points) has a grid line on it while levels 1 "
               "and 2 do not; level ratios -12.5 to +14.7 in all 16 tested members")

HW_RE = re.compile(r'^\d\.\d{3}e[+-]\d{2}$')
ETA_RE = re.compile(r'^0\.\d{12}$')


# ---------------------------------------------------------------------------------------------- input
def read_table(path):
    rows = []
    header = None
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.rstrip('\n').split('\t')
            if header is None:
                header = parts
                continue
            d = dict(zip(header, parts))
            p = (int(d['p1']), int(d['p2']), int(d['p3']))
            levels = [int(x) for x in d['levels'].split(',')]
            ratio = None
            r = d.get('ratio_if_3_levels', 'na')
            if r not in ('na', '', None):
                try:
                    ratio = float(r)
                except ValueError:
                    ratio = None
            rows.append(dict(p=p, mass=float(d['mass']), eta=float(d['eta']), hw=float(d['eta_hw']),
                             lev_a=float(d['eta_lev0']), lev_b=float(d['eta_lev1']), levels=levels, ratio=ratio,
                             chain_err=float(d.get('chain_err', 'nan')), ref_err=float(d.get('ref_err', 'nan')),
                             N1=d.get('N1', 'na'), N2=d.get('N2', 'na'), h_lev1=d.get('h_lev1', 'na')))
    if header is None:
        raise SystemExit(f'no header in {path}')
    return rows


def canonical(p):
    return tuple(sorted((abs(int(v)) for v in p), reverse=True))


def is_primitive(p):
    return math.gcd(math.gcd(abs(p[0]), abs(p[1])), abs(p[2])) == 1


def classify(p):
    q = canonical(p)
    if q in EXACT:
        return 'exact'
    if q[0] == q[1] and q[1] > q[2]:
        return 'family'
    return 'generic'


# ---------------------------------------------------------------------------------------------- the rule
def default_params():
    return dict(generic_factor=1.0, factor3=1.0, floor3=0.0, factor2=0.0, floor2=2e-6, exact_hw=1e-12,
                exact_decimals=12)


def exact_decimal_string(value_str, decimals):
    q = Decimal(1).scaleb(-decimals)
    return str(Decimal(value_str).quantize(q, rounding=ROUND_HALF_EVEN))


def apply_rule(row, params):
    p = row['p']
    cls = classify(p)
    d_last = row['lev_b'] - row['lev_a']
    out = dict(row)
    out['class'] = cls
    out['d_last'] = d_last
    out['n_levels'] = len(row['levels'])
    out['eta_computed'] = row['eta']
    out['hw_measured'] = row['hw']
    out['eta_exact'] = 'na'
    if cls == 'exact':
        ex = EXACT[canonical(p)]
        out['eta_str'] = exact_decimal_string(ex['eta'], params['exact_decimals'])
        out['eta_del'] = float(out['eta_str'])
        out['hw_del'] = params['exact_hw']
        out['eta_exact'] = ex['eta']
        out['rule'] = 'exact'
        out['reason'] = (f"exact value (2 routes; see JSON) printed to {params['exact_decimals']} decimals, "
                         f"hw covers the rounding; sweep value {row['eta']:.12f} +/- {row['hw']:.3e} kept in "
                         f"eta_computed/eta_hw_measured")
        out['covers_exact'] = abs(row['eta'] - float(ex['eta'])) <= row['hw']
    elif cls == 'family':
        if out['n_levels'] >= 3:
            hw = max(row['hw'], params['factor3'] * abs(d_last), params['floor3'])
            out['rule'] = 'family3:max(measured,%g*|d_last|,%g)' % (params['factor3'], params['floor3'])
            out['reason'] = ('p1=p2 family, 3 levels: level ratios irregular (ratio %s); the last change '
                             '|d_12| = %.2e is the gauge (covers any ratio >= 2 of the level 1,2,3 sequence)'
                             % ('%.2f' % row['ratio'] if row['ratio'] is not None else 'na', abs(d_last)))
        else:
            hw = max(row['hw'], params['factor2'] * abs(d_last), params['floor2'])
            out['rule'] = 'family2:max(measured,%g*|d_last|,%g)' % (params['factor2'], params['floor2'])
            out['reason'] = ('p1=p2 family, 2 levels: the only level change (%.2e) includes the level-0 grid '
                             'alignment effect and is no error gauge; floor %.1e from the family\'s 3-level evidence'
                             % (d_last, params['floor2']))
        out['eta_str'] = f"{row['eta']:.12f}"
        out['eta_del'] = row['eta']
        out['hw_del'] = hw
        out['covers_exact'] = 'na'
    else:
        out['eta_str'] = f"{row['eta']:.12f}"
        out['eta_del'] = row['eta']
        out['hw_del'] = params['generic_factor'] * row['hw']
        out['rule'] = 'measured' if params['generic_factor'] == 1.0 else 'measured*%g' % params['generic_factor']
        out['reason'] = ('generic: measured rule (h^4 log h law verified at 3 levels in 34 generic orbits, '
                         'ratios 16.0 to 17.3)')
        out['covers_exact'] = 'na'
    # builder's repair: the guard exempts the exact rows, where the exact value (2 routes) replaces the sweep value and
    # its half-width is the rounding of the printed decimals; the sweep value and its measured half-width are kept in
    # eta_computed and eta_hw_measured (the first run raised here at (1, 0, 0)).
    if cls != 'exact' and out['hw_del'] < row['hw'] - 1e-300:
        raise RuntimeError('delivered half-width below the measured one at %s' % (p,))
    return out


# ---------------------------------------------------------------------------------------------- output
def up3(x):
    """Builder's repair after the red team (busi/lead_api/eta_rt1/result.md, section 4): the delivered half-width is
    rounded outward (up) at the last printed digit of the .3e format (4 significant digits) instead of to nearest."""
    import math
    if x <= 0:
        return f"{x:.3e}"
    e = math.floor(math.log10(x)) - 3
    m = math.ceil(x / 10.0 ** e - 1e-9)
    v = m * 10.0 ** e
    if v < x:
        v = (m + 1) * 10.0 ** e
    return f"{v:.3e}"


COLS = ['p1', 'p2', 'p3', 'pnorm', 'mass', 'eta', 'eta_hw', 'eta_hw_measured', 'eta_computed', 'eta_exact',
        'levels', 'n_levels', 'eta_lev_a', 'eta_lev_b', 'd_last', 'ratio', 'class', 'hw_rule', 'reason']


def write_handoff(rows_out, path, params, source):
    with open(path, 'w') as f:
        f.write("# hand-off from the busi lane (task 2) to the hreduce lane (2c enclosure) and the paper lane: "
                "directional planar coefficients eta(n) of the flat massless sine-Dirac sea.\n")
        f.write("# conventions: nats; one 4-component field; one planar face; per unit area a^2; a = 1; ma = 0; "
                "one row per primitive cubic orbit (p1 >= p2 >= p3 >= 0), the 48 images to be generated by the reader.\n")
        f.write(f"# source table: {source}\n")
        f.write(f"# measured half-width (column eta_hw_measured): {MEASURED_RULE}\n")
        f.write("# delivered half-width (column eta_hw, the one to use): exact rows eta = exact value to %d decimals, "
                "eta_hw = %.1e; generic rows %g x measured; family rows with 3 levels max(measured, %g*|d_last|, %g); "
                "family rows with 2 levels max(measured, %g*|d_last|, %g).\n"
                % (params['exact_decimals'], params['exact_hw'], params['generic_factor'], params['factor3'],
                   params['floor3'], params['factor2'], params['floor2']))
        f.write(f"# family: {FAMILY_TEXT}\n")
        f.write("# the half-widths are measured enclosures under the stated premises, not certified intervals; "
                "eta is printed to 12 decimals (an inward error of at most 5e-13, inside the 1e-9 term of the measured rule); eta_hw is rounded outward at its last printed digit (the .3e format).\n")
        f.write("# eta_lev_a, eta_lev_b: the 2 finest grid levels used (levels[-2], levels[-1]); d_last = eta_lev_b - "
                "eta_lev_a; ratio = (change between the 2 coarser levels)/(change between the 2 finer levels) where 3 "
                "levels exist; eta_exact: the exact value to 20 digits for the exact rows.\n")
        f.write('\t'.join(COLS) + '\n')
        for r in sorted(rows_out, key=lambda r: (sum(v * v for v in r['p']), r['p'])):
            p = r['p']
            pn = math.sqrt(sum(v * v for v in p))
            vals = [p[0], p[1], p[2], f"{pn:.12f}", f"{r['mass']:g}", r['eta_str'], up3(r['hw_del']),
                    f"{r['hw_measured']:.3e}", f"{r['eta_computed']:.12f}", r['eta_exact'],
                    ','.join(str(l) for l in r['levels']), r['n_levels'], f"{r['lev_a']:.12f}", f"{r['lev_b']:.12f}",
                    f"{r['d_last']:+.3e}", f"{r['ratio']:.3f}" if r['ratio'] is not None else 'na', r['class'],
                    r['rule'], r['reason']]
            f.write('\t'.join(str(v) for v in vals) + '\n')


def summarize(rows_out):
    out = dict(n_rows=len(rows_out))
    for cls in ('exact', 'family', 'generic'):
        sub = [r for r in rows_out if r['class'] == cls]
        if not sub:
            continue
        hws = [r['hw_del'] for r in sub]
        hwm = [r['hw_measured'] for r in sub]
        out[cls] = dict(n=len(sub), n_3levels=sum(1 for r in sub if r['n_levels'] >= 3),
                        hw_delivered_max=max(hws), hw_delivered_median=statistics.median(hws),
                        hw_measured_max=max(hwm), hw_measured_median=statistics.median(hwm),
                        n_inflated=sum(1 for r in sub if r['hw_del'] > r['hw_measured'] * (1 + 1e-12)))
    return out


def write_json(rows_out, path, params, source, extra=None):
    fam = [dict(p=list(r['p']), n_levels=r['n_levels'], hw_measured=r['hw_measured'], hw_delivered=r['hw_del'],
                d_last=r['d_last'], ratio=r['ratio']) for r in rows_out if r['class'] == 'family']
    exact = {}
    for p, ex in EXACT.items():
        key = ''.join(str(v) for v in p)
        e = dict(ex)
        e['printed_in_table'] = exact_decimal_string(ex['eta'], params['exact_decimals'])
        e['table_hw'] = params['exact_hw']
        row = [r for r in rows_out if canonical(r['p']) == p]
        if row:
            e['sweep_value'] = row[0]['eta_computed']
            e['sweep_hw_measured'] = row[0]['hw_measured']
            e['sweep_minus_exact'] = row[0]['eta_computed'] - float(ex['eta'])
            e['sweep_interval_contains_exact'] = bool(row[0]['covers_exact'])
        exact[key] = e
    exact['b100_info'] = B100_INFO
    doc = dict(lane='busi', task='task 2 hand-off of the directional planar coefficients', source_table=source,
               conventions='nats; one 4-component sine-Dirac field; one face; per unit area a^2; a = 1; ma = 0',
               exact=exact,
               rule=dict(measured=MEASURED_RULE, parameters=params, family_definition=FAMILY_TEXT,
                         statement='exact rows: exact value to 12 decimals with hw 1e-12; generic rows: measured; '
                                   'family rows with 3 levels: max(measured, factor3*|d_last|, floor3); family rows with '
                                   '2 levels: max(measured, factor2*|d_last|, floor2); measured, not certified'),
               summary=summarize(rows_out), family_rows=sorted(fam, key=lambda d: sum(v * v for v in d['p'])))
    if extra:
        doc.update(extra)
    with open(path, 'w') as f:
        json.dump(doc, f, indent=1)


def check_table(rows):
    seen = set()
    for r in rows:
        if r['p'] != canonical(r['p']):
            raise SystemExit('non-canonical orbit %s' % (r['p'],))
        if not is_primitive(r['p']):
            raise SystemExit('non-primitive orbit %s' % (r['p'],))
        key = (r['p'], r['mass'])
        if key in seen:
            raise SystemExit('duplicate row %s' % (key,))
        seen.add(key)
        if len(r['levels']) < 2:
            raise SystemExit('fewer than 2 levels at %s' % (r['p'],))


def run(table, out_tsv, out_json, params, mass=0.0, verbose=True):
    rows = [r for r in read_table(table) if abs(r['mass'] - mass) < 1e-12]
    if not rows:
        raise SystemExit('no rows at mass %g' % mass)
    check_table(rows)
    rows_out = [apply_rule(r, params) for r in rows]
    write_handoff(rows_out, out_tsv, params, os.path.abspath(table))
    write_json(rows_out, out_json, params, os.path.abspath(table))
    if verbose:
        s = summarize(rows_out)
        print(f"{len(rows_out)} orbits -> {out_tsv}, {out_json}")
        for cls in ('exact', 'family', 'generic'):
            if cls in s:
                d = s[cls]
                print(f"  {cls:8s} n={d['n']:3d} (3 levels: {d['n_3levels']:3d}) hw delivered max {d['hw_delivered_max']:.2e} "
                      f"median {d['hw_delivered_median']:.2e}; measured max {d['hw_measured_max']:.2e} median "
                      f"{d['hw_measured_median']:.2e}; inflated {d['n_inflated']}")
        for r in sorted(rows_out, key=lambda r: sum(v * v for v in r['p'])):
            if r['class'] == 'exact':
                print(f"  exact {r['p']}: sweep {r['eta_computed']:.12f} +/- {r['hw_measured']:.1e}, exact {r['eta_exact']}, "
                      f"sweep - exact {r['eta_computed'] - float(r['eta_exact']):+.1e}, covered {r['covers_exact']}; "
                      f"delivered {r['eta_str']} +/- {r['hw_del']:.0e}")
            elif r['hw_del'] > r['hw_measured'] * (1 + 1e-12):
                rat = f"{r['ratio']:.2f}" if r['ratio'] is not None else 'na'
                print(f"  inflated {r['p']}: levels {r['levels']} ratio {rat} d_last {r['d_last']:+.2e} "
                      f"hw {r['hw_measured']:.2e} -> {r['hw_del']:.2e} ({r['rule']})")
    return rows_out


# ---------------------------------------------------------------------------------------------- self-test
SELFTEST_HEADER = ('p1\tp2\tp3\tpnorm\tmass\teta\teta_hw\teta_lev0\teta_lev1\tlevels\th_lev1\tN1\tN2\tcL\tLmin\t'
                   'chain_err\tref_err\tratio_if_3_levels\tseconds\thw_method')
SELFTEST_ROWS = [
    '1\t0\t0\t1.000000000000\t0\t0.786583387002\t4.356e-09\t0.786583372237\t0.786583385660\t0,1,2\t0.02454\t128\t128\t10\t24\t3.24e-13\t1.67e-15\t16.083\t32\tsee header',
    '1\t1\t0\t1.414213562373\t0\t0.869179366913\t8.251e-08\t0.869179725549\t0.869179399516\t0,1,2\t0.02454\t92\t128\t10\t24\t2.17e-13\t1.38e-15\t-12.466\t27\tsee header',
    '1\t1\t1\t1.732050807569\t0\t0.923381804953\t8.633e-09\t0.923381838536\t0.923381808006\t0,1,2\t0.02466\t104\t104\t10\t24\t3.33e-13\t1.38e-15\t16.145\t30\tsee header',
    '7\t1\t1\t7.141428428543\t0\t0.874320722283\t4.253e-08\t0.874320904963\t0.874320738890\t0,1,2\t0.02438\t28\t92\t10\t24\t9.16e-12\t8.86e-16\t16.459\t113\tsee header',
    '7\t7\t1\t9.949874371066\t0\t0.898496525478\t1.487e-07\t0.898497175297\t0.898496584552\t0,1,2\t0.02427\t20\t92\t10\t24\t1.41e-11\t6.76e-16\t2.011\t202\tsee header',
    '5\t5\t2\t7.348469228350\t0\t0.928758865961\t4.722e-08\t0.928759069335\t0.928758884449\t0,1\t0.04829\t14\t46\t10\t24\t3.22e-12\t1.58e-15\tna\t36\tsee header',
    '10\t10\t9\t16.763054614240\t0\t0.929860569810\t3.687e-06\t0.929876790200\t0.929862044391\t0,1\t0.04838\t6\t46\t10\t24\t1.18e-11\t4.25e-16\tna\t130\tsee header',
    '9\t1\t0\t9.055385138137\t0\t0.825890622156\t2.023e-06\t0.825899519056\t0.825891430965\t0,1\t0.04909\t8\t64\t10\t24\t9.05e-12\t5.09e-16\tna\t42\tsee header',
]


def _close(a, b, rel=1e-12):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-300)


def selftest():
    fails = []

    def check(name, ok, detail=''):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
        if not ok:
            fails.append(name)

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, 'eta_table_selftest.tsv')
        with open(src, 'w') as f:
            f.write('# synthetic rows copied from eta_table_v1.tsv\n')
            f.write(SELFTEST_HEADER + '\n')
            f.write('\n'.join(SELFTEST_ROWS) + '\n')
            f.write('# hw_method: trailing comment\n')
        params = default_params()
        out_tsv = os.path.join(tmp, 'handoff.tsv')
        out_json = os.path.join(tmp, 'handoff.json')
        rows_out = run(src, out_tsv, out_json, params, verbose=False)
        by = {r['p']: r for r in rows_out}
        check('row count', len(rows_out) == 8, f'{len(rows_out)}')
        cls = {p: classify(p) for p in by}
        check('classes', cls[(1, 0, 0)] == 'exact' and cls[(1, 1, 1)] == 'exact' and cls[(1, 1, 0)] == 'family'
              and cls[(7, 7, 1)] == 'family' and cls[(5, 5, 2)] == 'family' and cls[(10, 10, 9)] == 'family'
              and cls[(7, 1, 1)] == 'generic' and cls[(9, 1, 0)] == 'generic', str(cls))
        check('family detection on unsorted input', classify((0, 3, 3)) == 'family' and classify((-1, 1, 7)) == 'generic')
        r = by[(1, 0, 0)]
        check('[100] exact substitution', r['eta_str'] == '0.786583386553' and _close(r['hw_del'], 1e-12),
              f"{r['eta_str']} +/- {r['hw_del']:.0e}")
        check('[100] sweep interval covers exact', r['covers_exact'] is True,
              f"sweep - exact {r['eta_computed'] - float(r['eta_exact']):+.2e} vs hw {r['hw_measured']:.2e}")
        r = by[(1, 1, 1)]
        check('[111] exact substitution', r['eta_str'] == '0.923381805958' and _close(r['hw_del'], 1e-12),
              f"{r['eta_str']} +/- {r['hw_del']:.0e}")
        check('[111] sweep interval covers exact', r['covers_exact'] is True,
              f"sweep - exact {r['eta_computed'] - float(r['eta_exact']):+.2e} vs hw {r['hw_measured']:.2e}")
        lo, hi = (Decimal(x) for x in EXACT[(1, 1, 1)]['hreduce_certified_interval'])
        check('[111] route 1 inside the hreduce certificate', lo <= Decimal(EXACT[(1, 1, 1)]['eta']) <= hi)
        check('12-decimal rounding of the exact rows below hw',
              abs(float('0.786583386553') - float(EXACT[(1, 0, 0)]['eta'])) < 1e-12
              and abs(float('0.923381805958') - float(EXACT[(1, 1, 1)]['eta'])) < 1e-12)
        r = by[(1, 1, 0)]
        check('[110] family 3 levels: hw = |d_last|', _close(r['hw_del'], abs(0.869179399516 - 0.869179725549), 1e-9)
              and r['hw_del'] > r['hw_measured'], f"{r['hw_measured']:.3e} -> {r['hw_del']:.3e}")
        r = by[(7, 7, 1)]
        check('(7,7,1) family 3 levels: hw = |d_last|', _close(r['hw_del'], abs(0.898496584552 - 0.898497175297), 1e-9),
              f"{r['hw_measured']:.3e} -> {r['hw_del']:.3e}")
        r = by[(5, 5, 2)]
        check('(5,5,2) family 2 levels: floor binds', _close(r['hw_del'], params['floor2']),
              f"{r['hw_measured']:.3e} -> {r['hw_del']:.3e}")
        r = by[(10, 10, 9)]
        check('(10,10,9) family 2 levels: measured above the floor is kept', _close(r['hw_del'], 3.687e-6),
              f"{r['hw_del']:.3e}")
        check('generic rows unchanged', _close(by[(7, 1, 1)]['hw_del'], 4.253e-8) and _close(by[(9, 1, 0)]['hw_del'], 2.023e-6))
        # builder's repair, matching the guard in apply_rule: the exact rows carry the rounding half-width of the exact value
        check('no delivered half-width below the measured one (exact rows exempt)',
              all(r['hw_del'] >= r['hw_measured'] for r in rows_out if r['class'] != 'exact'))
        check('generic values unchanged', all(_close(r['eta_del'], r['eta_computed']) for r in rows_out if r['class'] != 'exact'))
        # output format checks (the hreduce parser reads p1 p2 p3 pnorm mass eta eta_hw by name)
        with open(out_tsv) as f:
            lines = [l.rstrip('\n') for l in f if not l.startswith('#') and l.strip()]
        hdr = lines[0].split('\t')
        need = ['p1', 'p2', 'p3', 'pnorm', 'mass', 'eta', 'eta_hw']
        check('header carries the hreduce columns by name', all(c in hdr for c in need), str(hdr[:8]))
        data = [dict(zip(hdr, l.split('\t'))) for l in lines[1:]]
        check('eta printed to 12 decimals', all(ETA_RE.match(d['eta']) for d in data), str([d['eta'] for d in data][:3]))
        check('eta_hw printed as 3-digit scientific', all(HW_RE.match(d['eta_hw']) for d in data),
              str([d['eta_hw'] for d in data][:3]))
        check('rows sorted by |p|', [tuple(int(d[c]) for c in ('p1', 'p2', 'p3')) for d in data]
              == sorted((r['p'] for r in rows_out), key=lambda p: (sum(v * v for v in p), p)))
        pn_ok = all(abs(float(d['pnorm']) - math.sqrt(sum(int(d[c]) ** 2 for c in ('p1', 'p2', 'p3')))) < 1e-12 for d in data)
        check('pnorm recomputed from the integers', pn_ok)
        with open(out_json) as f:
            doc = json.load(f)
        check('JSON carries the exact values and the rule',
              doc['exact']['100']['eta'] == EXACT[(1, 0, 0)]['eta'] and doc['exact']['111']['eta'] == EXACT[(1, 1, 1)]['eta']
              and doc['rule']['parameters']['floor2'] == params['floor2'] and len(doc['family_rows']) == 4)
        # sensitivity option: raising the family floors changes family rows only
        p2 = default_params()
        p2['floor2'] = 4e-6
        p2['floor3'] = 4e-6
        rows2 = run(src, os.path.join(tmp, 'h2.tsv'), os.path.join(tmp, 'h2.json'), p2, verbose=False)
        by2 = {r['p']: r for r in rows2}
        check('sensitivity floors raise family rows only',
              _close(by2[(5, 5, 2)]['hw_del'], 4e-6) and _close(by2[(1, 1, 0)]['hw_del'], 4e-6)
              and _close(by2[(7, 1, 1)]['hw_del'], 4.253e-8) and _close(by2[(1, 0, 0)]['hw_del'], 1e-12))
        # duplicate and non-primitive rows are refused
        bad = os.path.join(tmp, 'bad.tsv')
        with open(bad, 'w') as f:
            f.write(SELFTEST_HEADER + '\n' + SELFTEST_ROWS[0] + '\n' + SELFTEST_ROWS[0] + '\n')
        try:
            run(bad, os.path.join(tmp, 'b.tsv'), os.path.join(tmp, 'b.json'), params, verbose=False)
            check('duplicate rows refused', False)
        except SystemExit:
            check('duplicate rows refused', True)
    print(f"{len(SELFTEST_ROWS) and 24 - len(fails)} PASS, {len(fails)} FAIL")
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--table', default='eta_table_v1.tsv', help='the final sweep table (input-4 format)')
    ap.add_argument('--out', default='handoff_eta_table_v1.tsv')
    ap.add_argument('--json', default='handoff_eta_exact_v1.json')
    ap.add_argument('--mass', type=float, default=0.0)
    ap.add_argument('--generic-factor', type=float, default=1.0)
    ap.add_argument('--family-factor3', type=float, default=1.0, help='family rows with 3 levels: factor on |d_last|')
    ap.add_argument('--family-floor3', type=float, default=0.0)
    ap.add_argument('--family-factor2', type=float, default=0.0, help='family rows with 2 levels: factor on |d_last|')
    ap.add_argument('--family-floor2', type=float, default=2e-6)
    ap.add_argument('--exact-hw', type=float, default=1e-12)
    ap.add_argument('--exact-json', default=None, help='optional JSON with keys eta100, eta111 (decimal strings) to override')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if args.exact_json:
        with open(args.exact_json) as f:
            ov = json.load(f)
        if 'eta100' in ov:
            EXACT[(1, 0, 0)]['eta'] = str(ov['eta100'])
        if 'eta111' in ov:
            EXACT[(1, 1, 1)]['eta'] = str(ov['eta111'])
    params = dict(generic_factor=args.generic_factor, factor3=args.family_factor3, floor3=args.family_floor3,
                  factor2=args.family_factor2, floor2=args.family_floor2, exact_hw=args.exact_hw, exact_decimals=12)
    run(args.table, args.out, args.json, params, mass=args.mass)


if __name__ == '__main__':
    main()
