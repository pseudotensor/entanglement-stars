#!/usr/bin/env python3
"""s46v31_coreprobe_v2.py: one CORE_HF route-1 slice json, the radii/diagnostics/field_check/record blocks (read-only probe)."""
import json
f = '<internal>/core_hf_prod_T0015.00.json'
J = json.load(open(f))
print(list(J.keys()))
for k in ('field_check', 'record', 'diagnostics', 'gate', 'fails'):
    print('==', k, json.dumps(J.get(k), default=str)[:3000])
r = J['radii']
print('== radii keys', list(r.keys()))
for l, v in r.items():
    print(l, type(v).__name__, list(v.keys())[:60] if isinstance(v, dict) else v)
    if isinstance(v, dict):
        for kk, vv in v.items():
            print('   ', kk, json.dumps(vv, default=str)[:400])
        break
