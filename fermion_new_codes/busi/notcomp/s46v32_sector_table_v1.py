#!/usr/bin/env python3
"""s46v32_sector_table_v1.py (busi #472): the sector self-test table from the s46v32_sector_selftest_v1 jsons (no retyping).
Writes <internal>/s46v32_sector_table_v1.md."""
import json, os
B = '<internal>/s46v32'
L = ['### S3. Sector self-test: production operators compressed to the leak pilot sectors (one copy of kappa 1, N = n/2), brute force', '',
     '| slice | n | sector dim | (E_hat_2 - E_H) psi = N_z psi + T_E psi (rel) | ||N_z psi||^2 = g_z (rel) | <psi|T_E psi> | max over radii a_j^2 formula vs brute (rel) | ||phi||^2_sec (1ph + 2ph) | X_sec (brute) | X_lead,sec (formula) | X_sec / X_lead,sec | X_sec / ||phi||^2_sec |',
     '|' + '---|' * 12]
for h, t in (('prod', 15), ('prod', 35), ('s8', 14)):
    J = json.load(open(os.path.join(B, '%s_T%07.2f_sector_selftest_v1.json' % (h, t))))
    for r in J['results']:
        L.append('| %s %d | %d | %d | %.1e | %.1e | %.1e | %.1e | %.4e (%.4e + %.3e) | %.4e | %.4e | %.4f | %.3e |' % (
            h, t, r['n'], r['D'], r['decomposition_rel'], r['g_z_vs_brute_rel'], r['TE_mean_on_psi'],
            max(p['a_sq_rel'] for p in r['per_radius'].values()), r['phi_sq'], r['phi1_sq_brute'], r['phi2_sq_brute'],
            r['X_sec'], r['X_lead_sec'], r['X_sec_over_lead'], r['X_sec_over_phi_sq']))
open(os.path.join(B, 's46v32_sector_table_v1.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
