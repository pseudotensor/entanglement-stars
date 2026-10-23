#!/usr/bin/env python3
"""s46v32_table_v1.py (busi #472): the state-leakage tables from the s46v32 slice jsons (no retyping).
Writes <internal>/s46v32_tables_v1.md and s46v32_summary_v1.json."""
import json, os
B = '<internal>/s46v32'
SL = [('prod', t) for t in (12, 15, 16, 20, 24, 25, 28, 30, 35, 40)] + [('s8', t) for t in (12, 13, 14, 16, 20, 25, 30, 40)]
rows = []
for h, t in SL:
    p = os.path.join(B, '%s_T%07.2f_stateleak_v1.json' % (h, t))
    if os.path.exists(p):
        rows.append((h, t, json.load(open(p))))
L = ['### S1. Per slice: the bound, the rate and the one-record loss against the margin', '',
     '| slice | radii | <B> (upper) | margin 1 - <B> | ||phi1||^2 | ||phi2||^2 (upper) | X_lead (exact leading) | X_up (rigorous) | X used = min(X_up, ||phi||^2) | rate bound 2 sqrt(<B> X) | dT_rec | loss bound over one record | leading-part loss over one record | certified horizon t* = margin/rate (rigorous; leading) | retained over one record by the bound? |',
     '|' + '---|' * 15]
for h, t, J in rows:
    L.append('| %s %d | %s | %.6f | %.4f | %.4e | %.3e | %.4e | %.4e | %.4e | %.4e | %.3f | %.3f | %.3f | %.2f; %.2f | %s |' % (
        h, t, ', '.join(J['radii']), J['B_up'], J['margin'], J['phi1_sq'], J['phi2_sq'][1], J['X_lead'], J['X_up'], J['X_eff'], J['rate_up'],
        J['record']['dT'], J['loss_up_one_record'], J['loss_lead_one_record'], J['t_star_up'], J['t_star_lead'],
        'yes' if J['decision_retained_by_bound'] else 'NO (not certified)'))
L += ['', '### S2. Per radius: the exact leading amplitude and the enclosure', '',
      '| slice | radius | M_j | a_j/M_j (exact) | 0ph / 1ph / 2ph parts of a_j^2 | g_x | e_j/a_j | E1 (N_x on phi2) | E2 (T_j on phi1) | E3 (T_j on phi2) | sum |lambda(H_j)| | rank H_j (1e-8) | Var L (this route) vs v3 (rel) |',
      '|' + '---|' * 13]
for h, t, J in rows:
    for l in J['radii']:
        r = J['per_radius'][l]
        p = r['parts']
        L.append('| %s %d | %s | %.4f | %.4e | %.3e / %.3e / %.3e | %.4e | %.3f | %.2e | %.2e | %.2e | %.3e | %d | %.1e |' % (
            h, t, l, r['M'], r['a_rel'], p['zero'], p['one'], p['two'], r['g_x'], r['e_over_a'], r['E1'], r['E2'], r['E3'], r['sum_abs_lambda'],
            r['H_rank_1e8'], abs(r['Var_L_route'] - r['Var_L_v3']) / r['Var_L_v3']))
summ = dict(n=len(rows), all_not_certified=all(not J['decision_retained_by_bound'] for _, _, J in rows),
            min_loss_up=min(J['loss_up_one_record'] for _, _, J in rows), min_loss_lead=min(J['loss_lead_one_record'] for _, _, J in rows),
            max_t_star_up=max(J['t_star_up'] for _, _, J in rows), max_t_star_lead=max(J['t_star_lead'] for _, _, J in rows),
            min_t_star_up=min(J['t_star_up'] for _, _, J in rows), min_t_star_lead=min(J['t_star_lead'] for _, _, J in rows),
            dT=sorted(set(J['record']['dT'] for _, _, J in rows)),
            max_varL_check=max(abs(r['Var_L_route'] - r['Var_L_v3']) / r['Var_L_v3'] for _, _, J in rows for r in J['per_radius'].values()),
            any_retained=[('%s %d' % (h, t)) for h, t, J in rows if J['decision_retained_by_bound']],
            lead_retained=[('%s %d' % (h, t)) for h, t, J in rows if J['loss_lead_one_record'] < J['margin']])
L += ['', '### Summary', '', '```', json.dumps(summ, indent=1), '```']
open(os.path.join(B, 's46v32_tables_v1.md'), 'w').write('\n'.join(L) + '\n')
json.dump(summ, open(os.path.join(B, 's46v32_summary_v1.json'), 'w'), indent=1)
print('\n'.join(L))
