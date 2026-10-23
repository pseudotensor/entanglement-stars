#!/usr/bin/env python3
"""s46v31_leak_table_v1.py (busi #467 item 2): the pilot's per-n table from prod_T0015.00_leak_pilot_v1.json (no retyping)."""
import json
J = json.load(open('<internal>/prod_T0015.00_leak_pilot_v1.json'))
L = ['| n | N | sector dim D | nnz (each op) | B_sec min | B_sec max | <B>_sec on Hartree | dim Ran Pi / D | lambda_Pi | E_hat_2 spread on Hartree | E_hat_2 half-range | E_ground - E_H | eigh s | eigh residual | sparse apply s | structure s | ops build s | GPU peak GB | RSS peak GB | vertex identity | max mean resid | max |sec. moment - Var L_act|/Var L_act |',
     '|' + '---|' * 22]
for r in J['results']:
    P = r['parts']
    dm = max(abs(p['second_moment_act'] - p['varL_act']) / p['varL_act'] for p in P.values())
    L.append('| %d | %d | %d | %d | %.3e | %.3e | %.3e | %d / %d | %.3e | %.3e | %.3e | %.3e | %.1f | %.1e | %.2e | %.1f | %.1f | %.2f | %.2f | %.1e | %.1e | %.1e |' % (
        r['n'], r['N'], r['D'], P['R_out']['nnz'], r['B_min'], r['B_max'], r['B_hartree'], r['dim_Pi'], r['D'], r['lambda_Pi'], r['E_spread_on_hartree'],
        r['E_half_range'], r['E_ground_minus_hartree'], r['t_eigh_s'], max(r['eigh_residual_low'], r['eigh_residual_high']), r['t_apply_sparse_s'],
        r['t_structure_s'], r['t_ops_s'], r['gpu_peak_GB'], r['rss_peak_GB'], r['vertex_identity_rel'], max(p['mean_resid'] for p in P.values()), dm))
open('<internal>/s46v31_leak_table_v1.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
