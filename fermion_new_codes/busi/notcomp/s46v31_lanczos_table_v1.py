#!/usr/bin/env python3
"""s46v31_lanczos_table_v1.py (busi #469/#470): the Lanczos sector table from the three slice jsons, plus the dense pilot rows (no retyping).
Writes <internal>/s46v31_lanczos_table_v1.md."""
import json, os
B = '<internal>/s46v31'
L = ['| slice | sector | blocks (kappa: n_b modes, N_b occupied, dust) | n | N | dim D | two-body entries | nnz per M_hat_j | B_max (Lanczos) | Lanczos residual | matvecs | <B>_sec on Hartree | Pi = whole sector | assemble s | Lanczos s | one-op apply s | RSS peak GB | max mean resid | max rel (2nd moment - Var L_act) |',
     '|' + '---|' * 19]
for tag in ('prod_T0015.00', 'prod_T0035.00', 's8_T0014.00'):
    p = os.path.join(B, 'lanczos', '%s_s46v31_leak_lanczos_v1.json' % tag)
    if not os.path.exists(p):
        L.append('| %s | (not run) |' % tag)
        continue
    J = json.load(open(p))
    for r in J['results']:
        P = r['parts']
        bl = '; '.join('%d: %d, %d, %d' % (b['kappa'], b['n_b'], b['N_b'], b['n_dust']) for b in r['blocks'])
        dm = max(abs(x['second_moment_act'] - x['varL_act']) / x['varL_act'] for x in P.values())
        L.append('| %s | %s | %s | %d | %d | %d | %d | %d | %.4e | %.1e | %d | %.3e | %s | %.0f | %.0f | %.2e | %.1f | %.1e | %.1e |' % (
            tag.replace('_T00', ' ').replace('.00', ''), r['sector'], bl, r['n'], r['N'], r['D'], r['two_body_entries'], r['nnz'][0], r['B_max'], r['lanczos_residual'],
            r['n_matvec'], r['B_hartree'], r['Pi_is_whole_sector'], r['t_assemble_s'], r['t_lanczos_s'], r['t_apply_one_op_s'], r['rss_peak_GB'],
            max(x['mean_resid'] for x in P.values()), dm))
J = json.load(open(os.path.join(B, 'leak', 'prod_T0015.00_leak_pilot_v1.json')))
L.append('\nDense pilot (s46v31_leak_pilot_v1, prod 15, single block kappa 1), for the n 16 cross-check: ' + '; '.join(
    'n %d: B_max %.4e' % (r['n'], r['B_max']) for r in J['results']))
open(os.path.join(B, 'lanczos', 's46v31_lanczos_table_v1.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
