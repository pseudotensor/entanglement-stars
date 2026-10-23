#!/usr/bin/env python3
"""
core_hf_table_v1.py: the CORE_HF table from core_hf_slices_v1.py's outputs (json and npz only; no computation beyond reading and ratios).
One row per slice: history, T/r_s, dtype, the gate label, the field-check label, the core radii with M_H, sqrt(Var_C)/M_H (full) beside
N_<^-1/2 at R_50 and R_in, the signed excess, the Fock term E_F with y = E_F/E_pair (dressed; bare), the evolution rows' residuals at the
core nodes (support max, relative) and at R_50 and R_t, the checks' PASS/FAIL count. The star-mode json (if present) prints beside.
    python3 core_hf_table_v1.py --out <internal>/core_hf_v1 [--md core_hf_table.md]
"""
import argparse
import glob
import json
import math
import os


def f(x, fmt='%.3e'):
    if x is None:
        return '-'
    try:
        return fmt % x
    except (TypeError, ValueError):
        return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='<internal>/core_hf_v1')
    ap.add_argument('--md', default=None)
    a = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(os.path.join(a.out, '*', 'core_hf_*_T*.json'))):
        try:
            J = json.load(open(p))
        except Exception:  # noqa: BLE001
            continue
        if 'variance' not in J:
            continue
        V = J['variance']
        lab50 = 'R_50' if 'R_50' in V else J['radius_labels'][0]
        key = 'full' if 'full' in V[lab50]['variants'] else 'noref'
        v50 = V[lab50]['variants'][key]
        vin = V['R_in']['variants'][key] if 'R_in' in V else None
        ex = J.get('exchange', {})
        ed, eb = ex.get('dressed'), ex.get('bare')
        ev = J.get('evolution_rows') or {}
        s = ev.get('summary') or {}
        ar = ev.get('at_radii') or {}
        radii = ', '.join('%s %.2f (M_H %.3f)' % (l, V[l]['r'], V[l]['M_H']) for l in J['radius_labels'] if l in ('R_in', 'R_50', 'R_t'))
        rows.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            J['history'], f(J['T_over_rs'], '%.2f'), 'c64' if J['complex64'] else 'c128', (J['status'] or {}).get('limit_label'), J['rows']['field_check']['label'],
            radii,
            '%s: %s (N^-1/2 %s; excess %s)' % (lab50, f(v50['sqrt_VarC_over_M']), f(v50['N_r_inv_sqrt']), f(v50['signed_excess'], '%+.3e')),
            ('%s (N^-1/2 %s)' % (f(vin['sqrt_VarC_over_M']), f(vin['N_r_inv_sqrt']))) if vin else '-',
            ('%s / y %s' % (f(ed['E_F']), f(ed['y'], '%.4e'))) if ed else '-',
            ('%s / y %s' % (f(eb['E_F']), f(eb['y'], '%.4e'))) if eb else '-',
            f(eb['E_pair']) if eb else '-',
            ('rr %s, tt %s (floors %s, %s)' % (f(s.get('res_rr_support_max_rel')), f(s.get('res_tt_support_max_rel')), f(s.get('stepping_floor_rr')), f(s.get('stepping_floor_tt'))))
            if s else (ev.get('status') or 'not run'),
            ', '.join('%s rr %s tt %s' % (l, f(ar[l]['rel_rr']), f(ar[l]['rel_tt'])) for l in ('R_50', 'R_t') if l in ar) or '-',
            '%d PASS %d FAIL' % (J.get('n_pass', 0), J.get('n_fail', 0))))
    star = []
    for p in sorted(glob.glob(os.path.join(a.out, 'star', 'core_hf_star_*.json'))):
        J = json.load(open(p))
        V = J['variance']
        ex = J.get('exchange', {})
        star.append('| %s member %d (%s) | %s | %s | %s | regression %s |' % (
            os.path.basename(J['star']), J['member'], J['kernel'], J['rows']['field_check']['label'],
            ', '.join('%s Var_C(full) %s M_H %s' % (l, f(V[l]['variants'].get('full', V[l]['variants']['noref'])['Var_C'], '%.10e'), f(V[l]['M_H'], '%.10f')) for l in V if l != 'R_out'),
            ', '.join('%s E_F %s y %s' % (k, f(x['E_F']), f(x['y'], '%.4e')) for k, x in ex.items()),
            ('PASS' if J['regression']['pass_'] else ('FAIL' if J['regression']['rel'] else 'no reference')) + ' ' + str({k: f(v, '%.1e') for k, v in J['regression']['rel'].items()})))
    L = ['# CORE_HF table (core_hf_slices_v1.py outputs under %s)' % a.out, '',
         'Every core number is past the record\'s own hydrodynamic limit (gate label) and channel-truncated at kappa_sea = 10 in its sea part; '
         'complex64 slices carry the stored floor (sources 6.2e-6 to 8.7e-6 relative). The field-check label says whether the native-kernel re-solve '
         'reproduces the slice\'s applied fields (CORE) or not (SCHEME CONTROL). The evolution rows: engine_rows_v2\'s residuals relative to the '
         'largest term of each row, the stepping floor (second difference over tau) beside. E_F: the Fock term with the dressed (I + B P^A)^-1 B and '
         'the bare B native density-current kernels; y = E_F/E_pair; the small-cluster correlation control (2.7e-4 |E_H|) is quoted in each json.', '',
         '| history | T/r_s | dtype | gate | field check | core radii (M_H) | sqrt(Var_C)/M_H at R_50 | at R_in | E_F dressed / y | E_F bare / y | E_pair (bare) | rows at the core nodes | rows at R_50, R_t | checks |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|'] + rows
    if star:
        L += ['', '## The static star beside (the S10 regression with the compact kernel; the native-kernel star with gauss)', '',
              '| member | field check | Var_C(full), M_H | E_F, y | regression |', '|---|---|---|---|---|'] + star
    txt = '\n'.join(L) + '\n'
    md = a.md or os.path.join(a.out, 'core_hf_table.md')
    with open(md, 'w') as fh:
        fh.write(txt)
    print(txt)


if __name__ == '__main__':
    main()
