#!/usr/bin/env python3
"""s46v3_table_v1.py (busi, NOTCOMP 4.6 v3): the tables from the s46v3_mass2_v1 slice jsons. No computation beyond arithmetic on the stored
numbers. Reads <dir>/<hist>_T<TTTT.TT>.json (production), <hist>_T..._fd.json and _dr05.json (prod 15 controls), and CORE_HF_full_v1.2's
table T1 (Var_C, M_H) for the comparison of Var L with route 1. Writes <dir>/s46v3_tables_v1.md and s46v3_summary_v1.json."""
import glob
import json
import math
import os
import sys

d = sys.argv[1]
core = sys.argv[2]
T1 = {}
t = open(core).read()
i = t.find('### T1.')
j = t.find('###', i + 5)
for line in t[i:j].splitlines():
    if not (line.startswith('| prod') or line.startswith('| s8')):
        continue
    c = [x.strip() for x in line.strip().strip('|').split('|')]
    T1[(c[0], float(c[1]), c[3])] = dict(M_H=float(c[6]), Var_C=float(c[8]))

rows = []
for p in sorted(glob.glob(os.path.join(d, '*_T*.json'))):
    b = os.path.basename(p)[:-5]
    parts = b.split('_')
    if len(parts) != 2:
        continue
    J = json.load(open(p))
    rows.append(J)
rows.sort(key=lambda J: (J['history'] != 'prod', J['T_over_rs']))


def f(x, n=4):
    return ('%.' + str(n) + 'g') % x


out = []
summ = []
out.append('### V1. The second-order branch value per slice\n')
out.append('<B> = 4 sum_j <(M_hat_j - M_j)^2>/M_j^2 with the full second-order mass operator on the Hartree Slater state (interval: the four-loop '
           'exchange term enclosed); <B_lin> = 4 sum_j Var L_j/M_j^2 (v2\'s quantity); x* the Cauchy-Schwarz threshold; max q_j = max sqrt(<Q_j^2>)/M_j.\n')
out.append('| slice | radii | <B_lin> | <B> (lower) | <B> (upper) | <B> - <B_lin> | 1 - <B> (upper) | max q_j | x* | CS bound 4 sum (a_j + q_j)^2 |')
out.append('|---|---|---|---|---|---|---|---|---|---|')
for J in rows:
    B = J['B']
    lab = '%s %g' % (J['history'], J['T_over_rs'])
    qmax = max(J['variance'][l]['q'][1] for l in B['radii'])
    out.append('| %s | %s | %.5f | %.6f | %.6f | %+.2e | %.4f | %.2e | %.4f | %.5f |' % (lab, ', '.join(B['radii']), B['B_lin'], B['B'][0], B['B'][1],
                                                                                   B['B'][1] - B['B_lin'], 1 - B['B'][1], qmax, math.floor(B['x_star'] * 1e4) / 1e4, B['CS_bound']))
    summ.append(dict(slice=lab, radii=B['radii'], B_lin=B['B_lin'], B=B['B'], qmax=qmax, x_star=B['x_star'], CS=B['CS_bound']))
out.append('\n### V2. Per radius: the parts of <(M_hat_j - M_j)^2> (all over M_j^2) and Var L against CORE_HF route 1\n')
out.append('| slice | radius | M_j | Var L/M^2 | <LQ+QL>/M^2 | Var Q/M^2 (upper) | <Q>/M | <Q> parts K, dB, R (over M) | rel. second moment (upper) | Var L vs CORE_HF T1 Var_C (route 1, 4 figures) and ROUTE2 (route 2, unrounded, rel) | <Q> two routes (rel) | <LQ+QL> two routes (rel) |')
out.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
for J in rows:
    lab = '%s %g' % (J['history'], J['T_over_rs'])
    for l in J['B']['radii']:
        v = J['variance'][l]
        M = v['M_H']
        c = T1.get((J['history'], float(J['T_over_rs']), l))
        # T1 prints 4 significant figures: within rounding if |diff| <= half a unit of the 4th figure
        half = 0.5 * 10 ** (math.floor(math.log10(abs(c['Var_C']))) - 3) if c else float('nan')
        r2p = '<internal>/%s_T%07.2f.json' % (J['history'], J['T_over_rs'])
        r2 = ''
        if os.path.exists(r2p):
            R2J = json.load(open(r2p))
            vr = R2J['variance'][l]['variants']['full']['Var_C']
            r2 = '; ROUTE2 %.1e' % (abs(v['Var_L'] - vr) / vr)
        cmp_ = ('%s T1 (diff %.1e, half unit %.0e)%s' % ('within' if abs(v['Var_L'] - c['Var_C']) <= half * (1 + 1e-9) else 'OUTSIDE', abs(v['Var_L'] - c['Var_C']), half, r2)) if c else '-'
        mp = v['mean_offset_parts']
        out.append('| %s | %s | %.4f | %.4e | %+.3e | %.3e | %+.3e | %+.2e, %+.2e, %+.2e | %.5e | %s | %.1e | %.1e |' % (
            lab, l, M, v['Var_L'] / M ** 2, v['LQ_plus_QL'] / M ** 2, v['Var_Q'][1] / M ** 2, v['mean_offset'] / M,
            mp['K'] / M, mp['dB'] / M, mp['R'] / M, v['rel_second_moment'][1], cmp_,
            abs(v['mean_offset'] - v['mean_offset_route_trYD']) / abs(v['mean_offset']),
            abs(v['LQ_plus_QL'] - v['LQ_plus_QL_route_theta']) / abs(v['LQ_plus_QL'])))
out.append('\n### V3. Checks per slice\n')
out.append('| slice | K_j vs FD of windows (max rel) | d_w B Richardson (max rel) | R_hm + P^A (rel, channel 1) | Zolotarev degree r (min-max) | P^A matrix vs operator | H(R_out) vs V |')
out.append('|---|---|---|---|---|---|---|')
for J in rows:
    lab = '%s %g' % (J['history'], J['T_over_rs'])
    kf = max(max(J['variance'][l]['K_fd_rel']) for l in J['B']['radii'])
    db = max(J['variance'][l]['dB_fd']['richardson_rel_diff'] for l in J['B']['radii'])
    idc = max(x['rel'] for x in J['rho2_identity_R_hm_plus_PA'])
    rs = [x.get('r') for x in J['rho2'] if x.get('r') is not None]
    out.append('| %s | %.1e | %.1e | %.1e | %s | %.1e | %s |' % (lab, kf, db, idc, ('%d-%d' % (min(rs), max(rs))) if rs else '-',
                                                             J['response']['matrix_vs_exact_operator_rel'], ('%.1e' % J['H_R_out_vs_V_rel']) if J.get('H_R_out_vs_V_rel') is not None else '-'))
    cmpd = [(x['kappa'], x['compare_dd']['rel']) for x in J['rho2'] if 'compare_dd' in x]
    if cmpd:
        out.append('|  (rho\'\' two routes: %s) | | | | | | |' % '; '.join('kappa %d: %s' % (k, ', '.join('%.1e' % y for y in r)) for k, r in cmpd))
ctl = {}
for tag in ('fd', 'dr05'):
    for p in glob.glob(os.path.join(d, '*_%s.json' % tag)):
        ctl[os.path.basename(p)] = json.load(open(p))
if ctl:
    out.append('\n### V4. Controls on prod 15\n')
    for k, J in sorted(ctl.items()):
        B = J['B']
        out.append('- %s: <B_lin> %.6f, <B> [%.6f, %.6f]' % (k, B['B_lin'], B['B'][0], B['B'][1]))
        if 'fd_route' in J:
            for r_ in J['fd_route']['H']:
                out.append('  - H by the full self-consistent finite differences, direction "%s": %s' % (
                    r_['direction'], '; '.join('%s fd %.6e analytic %.6e rel %.1e' % (l, x['fd'], x['analytic'], x['rel']) for l, x in r_['radii'].items())))
            for r_ in J['fd_route']['rho2']:
                out.append('  - beta.rho\'\' alone by finite differences of rho, direction "%s": %s' % (
                    r_['direction'], '; '.join('%s fd %.6e analytic %.6e rel %.1e' % (l, x['fd'], x['analytic'], x['rel']) for l, x in r_['radii'].items())))
open(os.path.join(d, 's46v3_tables_v1.md'), 'w').write('\n'.join(out) + '\n')
json.dump(dict(slices=summ, controls={k: J['B'] for k, J in ctl.items()}), open(os.path.join(d, 's46v3_summary_v1.json'), 'w'), indent=1)
print('\n'.join(out))
