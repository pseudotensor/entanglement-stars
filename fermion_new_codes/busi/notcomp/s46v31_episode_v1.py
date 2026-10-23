#!/usr/bin/env python3
"""s46v31_episode_v1.py (busi #467): the prod history's records where the continued reference is not the negative set
(mask != e<0, inversions, pockets), per channel; emits <internal>/s46v31_episode_v1.md and .json. Read-only."""
import json, os
P = '<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json'
OUT = '<internal>/s46v31'
J = json.load(open(P))
rows = []
for r in J['records']:
    if r.get('E') is None:
        continue
    npk = r.get('n_pocket') or {}
    bad = (r.get('mask_equals_negative_set') is False) or int(r.get('inversions_total') or 0) > 0 or sum(int(v) for v in npk.values()) > 0
    if not bad:
        continue
    ch = {}
    for k, c in (r.get('continuation') or {}).items():
        if (not c.get('mask_equals_negative_set', True)) or (c.get('inversions') or 0) or (c.get('n_pocket') or 0):
            ch[k] = {x: c.get(x) for x in ('mode', 'inversions', 'n_pocket', 'pockets_occupied_positive', 'holes_empty_negative', 'min_occupied_weight',
                                            'max_empty_weight', 'mask_equals_negative_set', 'verified', 'margin_min')}
    rows.append(dict(T_over_rs=r['T_over_rs'], mask_eq_neg=r.get('mask_equals_negative_set'), inversions=r.get('inversions_total'),
                     n_pocket=sum(int(v) for v in npk.values()), R2_minus_R0_F=r.get('R2_minus_R0_F'), min_occ=r.get('min_occupied_weight'),
                     label=r.get('continuation_label'), channels=ch))
json.dump(rows, open(os.path.join(OUT, 's46v31_episode_v1.json'), 'w'), indent=1, default=str)
L = ['# prod (c19) records with the continued reference off the negative set (s46v31_episode_v1.py)\n',
     '| T/r_s | mask == e<0 | inversions | n_pocket | R2_minus_R0_F | min occ. weight | channels: kappa (pockets occ.+ / holes empty-; inversions) |', '|---|---|---|---|---|---|---|']
for x in rows:
    cs = '; '.join('%s (%s/%s; %s)' % (k, c['pockets_occupied_positive'], c['holes_empty_negative'], c['inversions']) for k, c in x['channels'].items())
    L.append('| %.3f | %s | %s | %d | %.3e | %.6f | %s |' % (x['T_over_rs'], x['mask_eq_neg'], x['inversions'], x['n_pocket'], x['R2_minus_R0_F'] or 0, x['min_occ'] or 1, cs))
open(os.path.join(OUT, 's46v31_episode_v1.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
