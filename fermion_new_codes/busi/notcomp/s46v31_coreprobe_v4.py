#!/usr/bin/env python3
"""s46v31_coreprobe_v4.py: CORE_HF route-1 variance variants at R_in (names and scalar fields) and the s46v3 json keys (read-only probe)."""
import json
J = json.load(open('<internal>/core_hf_prod_T0015.00.json'))
for nm, x in J['variance']['R_in']['variants'].items():
    print(nm, {k: v for k, v in x.items() if not isinstance(v, (list, dict))})
S = json.load(open('<internal>/prod_T0015.00.json'))
print(S['variance']['R_in']['Var_L'], S['variance']['R_in']['M_H'])
print(json.dumps(S['B']), json.dumps(S['rows'], default=str)[:600])
print(list(S['variance']['R_in'].keys()))
