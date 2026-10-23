#!/usr/bin/env python3
"""s46v31_fdprobe_v1.py: keys of the s46v3 fd control json and the c17 parent record args (read-only probe)."""
import json
S = json.load(open('<internal>/prod_T0015.00_fd.json'))
print([k for k in S.keys()])
for k in S:
    if 'fd' in k.lower():
        print(k, json.dumps(S[k], default=str)[:2500])
J = json.load(open('<internal>/collapse_c17_rs64_A_R2_tau8_ell4_avg.json'))
print(J['args'])
recs = [r for r in J['records'] if r.get('E') is not None]
print(len(recs), recs[0].get('T_over_rs'), recs[-1].get('T_over_rs'))
print({k: recs[-1].get(k) for k in ('mask_equals_negative_set', 'inversions_total', 'intermediate_weights_total', 'n_pocket', 'n_refill_R0', 'min_occupied_weight', 'continuation_label')})
print({k: recs[0].get(k) for k in ('mask_equals_negative_set', 'inversions_total', 'intermediate_weights_total', 'n_pocket', 'n_refill_R0', 'min_occupied_weight', 'continuation_label')})
