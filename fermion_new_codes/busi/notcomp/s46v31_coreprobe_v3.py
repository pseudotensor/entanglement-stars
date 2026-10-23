#!/usr/bin/env python3
"""s46v31_coreprobe_v3.py: CORE_HF route-1 slice json, variance / channels / checks blocks (read-only probe)."""
import json
f = '<internal>/core_hf_prod_T0015.00.json'
J = json.load(open(f))
v = J['variance']
print('variance', type(v).__name__, list(v.keys())[:20] if isinstance(v, dict) else len(v))
if isinstance(v, dict):
    for k, x in list(v.items())[:3]:
        print('  ', k, json.dumps(x, default=str)[:1500])
c = J['channels']
print('channels', type(c).__name__, len(c))
print(json.dumps(c[0] if isinstance(c, list) else list(c.items())[0], default=str)[:1500])
ck = J['checks']
print('checks', type(ck).__name__, len(ck))
for x in (ck if isinstance(ck, list) else ck.values()):
    print('  ', json.dumps(x, default=str)[:300])
print('status', json.dumps(J['status'], default=str)[:800])
