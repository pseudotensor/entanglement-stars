#!/usr/bin/env python3
"""
core_hf_full_tables_v1_1.py: markdown tables for CORE_HF_full_v1.1.md from core_hf_full_v1_1.json (core_hf_full_data_v1_1.py) and, when
present, core_hf_controls_v1_1.json (core_hf_controls_data_v1_1.py); formatting and ratios only.
    python3 core_hf_full_tables_v1_1.py --json .../core_hf_full_v1_1.json [--controls .../core_hf_controls_v1_1.json] --out .../core_hf_full_v1_1_tables.md
"""
import argparse
import json
import math
import os


def g(x, fmt='%.3g'):
    return '-' if x is None else (fmt % x)


def esc(s):
    return s.replace('|', '\\|')


def fourth_secured(x, foot):
    """is the 4th significant figure of x secured against a relative footprint foot (|x| foot < half a unit of the 4th figure)?"""
    if foot is None:
        return 'c128'
    unit = 10.0 ** (math.floor(math.log10(abs(x))) - 3)
    return 'yes' if abs(x) * foot < 0.5 * unit else 'NO'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default='<internal>/core_hf_full_v1_1.json')
    ap.add_argument('--controls', default='<internal>/core_hf_controls_v1_1.json')
    ap.add_argument('--out', default='<internal>/core_hf_full_v1_1_tables.md')
    a = ap.parse_args()
    R = json.load(open(a.json))
    L = []
    L += ['### T1. Variance (full: the linearized dressed vertex with the reference response) at R_in, R_50, R_t', '',
          'M_H is the re-solve mass at the nearest 0.1a grid row (r_eval); M_interp is the same mass function interpolated at the nominal r. '
          'The 4th-figure column says whether the complex64 floor footprint leaves the 4th printed figure of Var_C secured (c128: no footprint, complex128 columns).', '',
          '| history | T/r_s | dtype | radius | r/a | r_eval | M_H | M_interp | Var_C | sea part Var_C0 | excess | sqrt(Var_C)/M_H | N_<^-1/2 | ratio / N_<^-1/2 | ratio with M_interp | ref. response | floor footprint | 4th figure of Var_C |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for h, rows in R['slices'].items():
        for r in rows:
            for lab in ('R_in', 'R_50', 'R_t'):
                if lab not in r['radii']:
                    continue
                x = r['radii'][lab]
                L.append('| %s | %g | %s | %s | %.2f | %.1f | %.2f | %.2f | %.4g | %.4g | %+.4g | %.4f | %.4f | %.2f | %.2f | %s | %s | %s |' % (
                    h, r['T_over_rs'], 'c64' if r['complex64'] else 'c128', lab, x['r'], x['r_eval'], x['M_H'], x['M_interp'], x['Var_C'], x['Var_C0'], x['excess'],
                    x['sqrt_VarC_over_M'], x['N_inv_sqrt'], x['ratio_over_N'], x['ratio_over_N_interp'],
                    ('%+.1f%%' % (100 * x['reference_response_rel'])) if x['reference_response_rel'] is not None else '-',
                    g(x['floor_footprint_rel'], '%.1e'), fourth_secured(x['Var_C'], x['floor_footprint_rel'])))
    L += ['', '### T2. Fock term of the whole slice (E_F/\\|E_pair\\|; comparators, neither a bound on the native signed exchange: 1/N_B = 1.5625e-3 and the positive scalar monopole kernel\'s 5/(4 N_B) = 1.953e-3)', '',
          'Kernel spectra: eig B (bare) and eig V (dressed). ||V - B||/||B|| is a Frobenius ratio; ||P^A B|| (not tabulated per slice, 0.76 on prod T16) is the spectral norm; sigma_min is a computed singular value, not an enclosure.', '',
          '| history | T/r_s | E_F dressed | E_pair dressed | E_F/\\|E_pair\\| dressed | E_F bare | E_pair bare | E_F/\\|E_pair\\| bare | dressed/bare E_F - 1 | / (1/N_B) dressed | 1/N_<(R_50) | 1/N_<(R_t) | eig B | eig V | \\|\\|V-B\\|\\|_F/\\|\\|B\\|\\|_F | sigma_min(I+BP^A) |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for h, rows in R['slices'].items():
        for r in rows:
            d, b, k = r['exchange']['dressed'], r['exchange']['bare'], r['kernel']
            L.append('| %s | %g | %.4e | %.3f | %.4e | %.4e | %.3f | %.4e | %+.2f%% | %.3f | %s | %s | [%.3f, %.3f] | [%.3f, %.3f] | %.3f | %.3f |' % (
                h, r['T_over_rs'], d['E_F'], d['E_pair'], d['E_F_over_abs_E_pair'], b['E_F'], b['E_pair'], b['E_F_over_abs_E_pair'],
                100 * r['exchange']['dressed_vs_bare_E_F_rel'], d['E_F_over_abs_E_pair'] / d['inv_NB'], g(d['inv_N50'], '%.3e'), g(d['inv_Nt'], '%.3e'),
                k['B_eig_min'], k['B_eig_max'], k['V_eig_min'], k['V_eig_max'], k['V_minus_B_rel'], k['smin_I_plus_BPA']))
    L += ['', '### T3. Evolution rows (abs(res) / the largest of the row\'s terms including d_t k; floor = abs(second difference over tau) on the same scale)', '',
          '| history | T/r_s | support nodes | rr max (at r/a) | tt max (at r/a) | rr at R_in (floor) | tt at R_in (floor) | rr at R_50 (floor) | tt at R_50 (floor) | rr at R_t (floor) | tt at R_t (floor) | tt nodes above floor | rr nodes above floor / 3x floor | W_0 max / rho scale (at r/a) | W_0 max beyond r = ell / rho scale | W_r max / J scale (at r/a) |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for h, rows in R['slices'].items():
        for r in rows:
            e = r['evolution']
            if e['status'] != 'done':
                L.append('| %s | %g | - | %s | | | | | | | | | | | | |' % (h, r['T_over_rs'], esc(str(e.get('status_verbatim') or e['status']))))
                continue
            at = e['at']

            def cell(lab, k):
                return ('%.3f (%.3g)' % (at[lab]['rel_' + k], at[lab]['floor_' + k])) if lab in at else '-'
            n = e['n_support_nodes']
            L.append('| %s | %g | %d | %.2f (%g) | %.2f (%g) | %s | %s | %s | %s | %s | %s | %d/%d | %d/%d / %d/%d | %.3f (%g) | %.3f | %.3f (%g) |' % (
                h, r['T_over_rs'], n, e['rr_max'], e['rr_max_at'], e['tt_max'], e['tt_max_at'], cell('R_in', 'rr'), cell('R_in', 'tt'), cell('R_50', 'rr'), cell('R_50', 'tt'),
                cell('R_t', 'rr'), cell('R_t', 'tt'), e['node_res_over_floor_tt']['n_above_1'], n, e['node_res_over_floor_rr']['n_above_1'], n,
                e['node_res_over_floor_rr']['n_above_3'], n, e['W0_max_abs_inside'] / e['rho_scale'], e['W0_max_at_r'], e['W0_max_beyond_one_ell'] / e['rho_scale'],
                e['Wr_max_over_J_scale'], e['Wr_max_at_r']))
    L += ['', '### T4. Slice labels (the record, the state, the reference)', '',
          '| history | T/r_s | dtype | trapping roots (r/a) | R_c | N_min | gate sigma\\|dv\\| (at r/a) | converged, gauge, flags | field check (max rel N, v) | pure (f_min, f_max) | S7b mask == (e<0) | S8 P_ref resolution | S8b \\|\\|(1-P_ref) h_m P_ref\\|\\|_F/\\|\\|h_m\\|\\|_F | checks |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for h, rows in R['slices'].items():
        for r in rows:
            z = r['record']['trapping_zeros']
            p, rf = r['purity'], r['reference']
            L.append('| %s | %g | %s | %s | %.2f | %.3f | %.3f (%g) | %s, %s, %s | %s %.1e | %s (%g, %g) | %s | %.1e | %.1e | %d PASS %d FAIL |' % (
                h, r['T_over_rs'], r['dtype'], ', '.join('%.2f' % x for x in z) if z else 'none', r['record']['R_c'], r['record']['N_min'],
                r['gate']['sigma_abs_dv_max'], r['gate']['at_R'], r['record']['converged'], r['record']['gauge_label'], r['record']['flags'] or 'none',
                r['field_check']['label'], max(r['field_check']['N_rel_maxdev'], r['field_check']['v_rel_maxdev']),
                'yes' if p['all_pure'] else 'NO', p['f_min'], p['f_max'], 'yes' if rf['S7b_mask_equals_neg_all'] else 'NO',
                rf['S8_pref_resolution_max'], rf['S8b_commutator_max'], r['n_pass'], r['n_fail']))
    L += ['', '### T5. Every FAIL', '', '| history | T/r_s | dtype | check | value | tol | value/tol |', '|---|---|---|---|---|---|---|']
    for f in R['fails']:
        L.append('| %s | %g | %s | %s | %.3e | %.0e | %s |' % (f['history'], f['T_over_rs'], f['dtype'], esc(f['name']), f['value'], f['tol'],
                                                             ('%.2f' % f['over_tol']) if f['over_tol'] is not None else '- (tol 0: a count)'))
    L += ['', '### T6. FAIL diagnostics per slice (S3 raw stored columns vs S3c Loewdin state; Q6\'; R_out sea part)', '',
          '| history | T/r_s | dtype | S3 raw (tol) | S3b raw | S3c Loewdin (tol) | S3d Loewdin | stored orthonormality max | floor footprint on Var_C | Q6\' Gram vs direct | R_out Var_C0/Var_C |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for h, rows in R['slices'].items():
        for r in rows:
            d = r['diagnostics']
            ff = d['floor_footprint_rel_range']
            L.append('| %s | %g | %s | %.2e (%.0e) | %.1e | %.1e (%.0e) | %.1e | %.1e | %s | %.2e | %+.2e |' % (
                h, r['T_over_rs'], r['dtype'], d['S3']['value'], d['S3']['tol'], d['S3b']['value'], d['S3c']['value'], d['S3c']['tol'], d['S3d']['value'],
                d['stored_orthonormality_max'], ('%.1e to %.1e' % tuple(ff)) if ff else '-', d["Q6'"]['value'], d['R_out']['full']['Var_C0_over_Var_C']))
    L += ['', '### T7. Card prod T16 against the delivery (CORE_HF_driver_v1.md, md5 %s): 34 comparison rows from a hard-coded list of 33 scalar quotations and 1 range (the prose is not parsed)' % R['T16']['delivery']['md5'][:8], '',
          '| quantity | delivery | card (same decimals) | card (full) | agrees |', '|---|---|---|---|---|']
    for it in R['T16']['delivery']['items']:
        c = it['card']
        L.append('| %s | %s | %s | %s | %s |' % (esc(it['quantity']), it['delivery'], it['card_rounded'],
                                               ', '.join('%.3f' % x for x in c) if isinstance(c, list) else '%.10g' % c, 'yes' if it['agrees'] else 'NO'))
    L += ['', '### T8. s8 against prod at the shared clocks and radius labels (different evolved objects and different geometric radii; no smoothing law follows)', '',
          '| T/r_s | radius | r/a prod | r/a s8 | Var_C0 s8/prod | Var_C s8/prod | sqrt(Var_C)/M_H s8/prod |', '|---|---|---|---|---|---|---|']
    for c in R['s8_vs_prod']:
        L.append('| %g | %s | %.2f | %.2f | %.3f | %.3f | %.3f |' % (c['T'], c['label'], c['r_prod'], c['r_s8'], c['sea_s8_over_prod'], c['VarC_s8_over_prod'], c['rms_s8_over_prod']))
    if a.controls and os.path.exists(a.controls):
        C = json.load(open(a.controls))
        L += ['', '### C1. Exchange blocks (bare kernel B, from the v1.1 CPU reruns; energy = (0,0), mixed = (0,1) + (1,0), current = (1,1))', '',
              '| run | E_F | energy | mixed | current | \\|E_F\\| / sum \\|blocks\\| | E_pair | energy | mixed | current | \\|E_pair\\| / sum \\|blocks\\| | E_F vs card bare (rel) |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for run in C['runs']:
            if 'blocks' not in run:
                continue
            b = run['blocks']
            L.append('| %s | %.6e | %+.4e | %+.4e | %+.4e | %.3g | %.6f | %+.4f | %+.4f | %+.4f | %.3g | %s |' % (
                run['label'], b['E_F_total'], b['E_F']['energy'], b['E_F']['mixed'], b['E_F']['current'], b['cancellation_E_F'],
                b['E_pair_total'], b['E_pair']['energy'], b['E_pair']['mixed'], b['E_pair']['current'], b['cancellation_E_pair'], g(run.get('E_F_bare_vs_card_rel'), '%.1e')))
        if C.get('rounding'):
            rd = C['rounding']
            L += ['', '### C2. Storage-precision control: prod T16 complex128 columns rounded to complex64 (the same layout as the record\'s complex64 slices), run through the same code', '',
                  'Both runs are the v1.1 CPU reruns (no response); "full" uses the card run\'s dressed betas on each run\'s own Grams. Relative difference = |rounded - complex128| / |complex128|.', '',
                  '| quantity | complex128 | rounded to complex64 | relative difference | figures secured |', '|---|---|---|---|---|']
            for it in rd['items']:
                L.append('| %s | %.10g | %.10g | %.1e | %s |' % (esc(it['name']), it['c128'], it['c64r'], it['rel'], it['figures']))
        if C.get('q6'):
            L += ['', '### C3. Q6\' localized (the worst channel, vertex and contraction; v1.1 reruns, same seed as the driver\'s check)', '',
                  '| run | worst rel | kappa | vertex | contraction | abs difference | denominator | denominator is the 1e-6 \\|\\|A Phi\\|\\|^2 floor |', '|---|---|---|---|---|---|---|---|']
            for q in C['q6']:
                L.append('| %s | %.2e | %d | %s | %s | %.2e | %.2e | %s |' % (q['label'], q['rel'], q['kappa'], esc(q['vertex']), q['contraction'], q['abs'], q['den'], q['den_is_floor']))
        if C.get('r_out'):
            L += ['', '### C4. The R_out sea part in residual form (sum_k 16 kappa ||(1 - P_ref) A P_ref columns||^2, nonnegative by construction) beside the Gram value', '',
                  '| run | vertex | Var_C0 residual form | Var_C0 Gram | Var_C | residual-form Var_C0 / Var_C |', '|---|---|---|---|---|---|']
            for q in C['r_out']:
                L.append('| %s | %s | %.3e | %+.3e | %.6g | %.2e |' % (q['label'], q['vertex'], q['res'], q['gram'], q['VarC'], q['ratio']))
        if C.get('dr'):
            L += ['', '### C5. Radial-grid refinement of M_H and of the noref vertex\'s variance (prod T16, v1.1 CPU rerun)', '',
                  '| radius | r/a | dr | r_eval | M_H (grid row) | M_interp | Var_C noref | Var_C0 noref | Var_C noref rel. to dr 0.1 |', '|---|---|---|---|---|---|---|---|---|']
            for q in C['dr']:
                L.append('| %s | %.4f | %g | %.4f | %.4f | %.4f | %.6g | %.6g | %s |' % (q['label'], q['r'], q['dr'], q['r_eval'], q['M_H'], q['M_interp'], q['VarC'], q['VarC0'], g(q.get('rel_to_base'), '%+.2e')))
    txt = '\n'.join(L) + '\n'
    open(a.out, 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
