#!/usr/bin/env python3
"""
core_hf_full_tables_v1.py: markdown tables for CORE_HF_full_v1.md from core_hf_full_v1.json (core_hf_full_data_v1.py); no computation
beyond formatting and ratios already in the json.
    python3 core_hf_full_tables_v1.py --json .../core_hf_full_v1.json --out .../core_hf_full_v1_tables.md
"""
import argparse
import json


def g(x, fmt='%.3g'):
    return '-' if x is None else (fmt % x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default='<internal>/core_hf_full_v1.json')
    ap.add_argument('--out', default='<internal>/core_hf_full_v1_tables.md')
    a = ap.parse_args()
    R = json.load(open(a.json))
    L = []
    # table 1: variance per clock
    L += ['## T1. Variance (full: dressed, with the reference response) at R_in, R_50, R_t', '',
          '| history | T/r_s | dtype | radius | r/a | M_H | Var_C | sea part Var_C0 | excess | sqrt(Var_C)/M_H | N_<^-1/2 | ratio / N_<^-1/2 | ref. response | floor footprint |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for h, rows in R['slices'].items():
        for r in rows:
            for lab in ('R_in', 'R_50', 'R_t'):
                if lab not in r['radii']:
                    continue
                x = r['radii'][lab]
                L.append('| %s | %g | %s | %s | %.2f | %.2f | %.4g | %.4g | %+.4g | %.4f | %.4f | %.2f | %s | %s |' % (
                    h, r['T_over_rs'], 'c64' if r['complex64'] else 'c128', lab, x['r'], x['M_H'], x['Var_C'], x['Var_C0'], x['excess'],
                    x['sqrt_VarC_over_M'], x['N_inv_sqrt'], x['ratio_over_N'],
                    ('%+.1f%%' % (100 * x['reference_response_rel'])) if x['reference_response_rel'] is not None else '-',
                    g(x['floor_footprint_rel'], '%.1e')))
    # table 2: Fock term
    L += ['', '## T2. Fock term (E_F/\\|E_pair\\|; comparators 1/N_B = 1.5625e-3, monopole bound 5/(4 N_B) = 1.953e-3)', '',
          '| history | T/r_s | E_F dressed | E_pair dressed | E_F/\\|E_pair\\| dressed | E_F bare | E_pair bare | E_F/\\|E_pair\\| bare | dressed/bare E_F - 1 | / (1/N_B) dressed | \\|\\|V-B\\|\\|/\\|\\|B\\|\\| | sigma_min(I+BP^A) |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for h, rows in R['slices'].items():
        for r in rows:
            d, b = r['exchange']['dressed'], r['exchange']['bare']
            L.append('| %s | %g | %.4e | %.3f | %.4e | %.4e | %.3f | %.4e | %+.2f%% | %.3f | %.3f | %.3f |' % (
                h, r['T_over_rs'], d['E_F'], d['E_pair'], d['E_F_over_abs_E_pair'], b['E_F'], b['E_pair'], b['E_F_over_abs_E_pair'],
                100 * r['exchange']['dressed_vs_bare_E_F_rel'], d['E_F_over_abs_E_pair'] / d['inv_NB'], r['kernel']['V_minus_B_rel'], r['kernel']['smin_I_plus_BPA']))
    # table 3: evolution rows
    L += ['', '## T3. Evolution rows (abs(res) / largest term of the row; floor = abs(second difference over tau) on the same scale)', '',
          '| history | T/r_s | nodes | rr max (at r/a) | tt max (at r/a) | rr at R_50 (floor) | tt at R_50 (floor) | rr at R_t (floor) | tt at R_t (floor) | tt nodes above floor | rr nodes above 3x floor | W_0 max / rho scale | W_r max |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for h, rows in R['slices'].items():
        for r in rows:
            e = r['evolution']
            if e['status'] != 'done':
                L.append('| %s | %g | - | %s | | | | | | | | | |' % (h, r['T_over_rs'], 'not computed: no record at T + tau (%s)' % e['times'].get('plus', {}).get('T')))
                continue
            at = e['at']

            def cell(lab, k):
                return ('%.3f (%.3g)' % (at[lab]['rel_' + k], at[lab]['floor_' + k])) if lab in at else '-'
            L.append('| %s | %g | %d | %.2f (%g) | %.2f (%g) | %s | %s | %s | %s | %d/%d | %d/%d | %.3f | %.1e |' % (
                h, r['T_over_rs'], e['n_nodes'], e['rr_max'], e['rr_max_at'], e['tt_max'], e['tt_max_at'], cell('R_50', 'rr'), cell('R_50', 'tt'),
                cell('R_t', 'rr'), cell('R_t', 'tt'), e['node_res_over_floor_tt']['n_above_1'], e['n_nodes'], e['node_res_over_floor_rr']['n_above_3'], e['n_nodes'],
                e['W0_max_abs_inside'] / e['rho_scale'], e['Wr_max_abs_inside']))
    # table 4: slice labels
    L += ['', '## T4. Slice labels', '',
          '| history | T/r_s | dtype | trapping roots (r/a) | R_c | N_min | gate sigma\\|dv\\| (at r/a) | field check (max rel N, v) | checks |', '|---|---|---|---|---|---|---|---|---|']
    for h, rows in R['slices'].items():
        for r in rows:
            z = r['record']['trapping_zeros']
            L.append('| %s | %g | %s | %s | %.2f | %.3f | %.3f (%g) | %s %.1e | %d PASS %d FAIL |' % (
                h, r['T_over_rs'], r['dtype'], ', '.join('%.2f' % x for x in z) if z else 'none', r['record']['R_c'], r['record']['N_min'],
                r['gate']['sigma_abs_dv_max'], r['gate']['at_R'], r['field_check']['label'],
                max(r['field_check']['N_rel_maxdev'], r['field_check']['v_rel_maxdev']), r['n_pass'], r['n_fail']))
    # table 5: FAILs
    L += ['', '## T5. Every FAIL', '', '| history | T/r_s | dtype | check | value | tol | value/tol |', '|---|---|---|---|---|---|---|']
    for f in R['fails']:
        L.append('| %s | %g | %s | %s | %.3e | %.0e | %.2f |' % (f['history'], f['T_over_rs'], f['dtype'], f['name'].replace('|', '\\|'), f['value'], f['tol'], f['over_tol'] or 0))
    L += ['', '## T6. FAIL diagnostics per slice (S3 raw stored columns vs S3c Loewdin state; Q6\'; R_out sea part)', '',
          '| history | T/r_s | dtype | S3 raw (tol) | S3b raw | S3c Loewdin (tol) | S3d Loewdin | stored orthonormality max | floor footprint on Var_C | Q6\' Gram vs direct | R_out Var_C0/Var_C |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for h, rows in R['slices'].items():
        for r in rows:
            d = r['diagnostics']
            ff = d['floor_footprint_rel_range']
            L.append('| %s | %g | %s | %.2e (%.0e) | %.1e | %.1e (%.0e) | %.1e | %.1e | %s | %.2e | %+.2e |' % (
                h, r['T_over_rs'], r['dtype'], d['S3']['value'], d['S3']['tol'], d['S3b']['value'], d['S3c']['value'], d['S3c']['tol'], d['S3d']['value'],
                d['stored_orthonormality_max'], ('%.1e to %.1e' % tuple(ff)) if ff else '-', d["Q6'"]['value'], d['R_out']['full']['Var_C0_over_Var_C']))
    # table 7: delivery check
    L += ['', '## T7. Card prod T16 against the delivery (CORE_HF_driver_v1.md, md5 %s)' % R['T16']['delivery']['md5'][:8], '',
          '| quantity | delivery | card (rounded alike) | card (full) | agrees |', '|---|---|---|---|---|']
    for it in R['T16']['delivery']['items']:
        c = it['card']
        L.append('| %s | %s | %s | %s | %s |' % (it['quantity'].replace('|', '\\|'), it['delivery'], it['card_rounded'],
                                               ', '.join('%.3f' % x for x in c) if isinstance(c, list) else '%.10g' % c, 'yes' if it['agrees'] else 'NO'))
    txt = '\n'.join(L) + '\n'
    open(a.out, 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
