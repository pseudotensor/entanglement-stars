#!/usr/bin/env python3
"""s46v31_recprobe_v1.py: list the record keys of the two histories and the crossing/occupation fields (read-only probe)."""
import json, sys
import numpy as np
P = {'prod': '<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json',
     's8': '<internal>/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json'}
for h, p in P.items():
    J = json.load(open(p))
    print('==', h, 'top keys', list(J.keys()))
    print('args', {k: J['args'][k] for k in list(J['args'])[:80]})
    recs = [r for r in J['records'] if r.get('E') is not None]
    print('n records', len(recs))
    r = recs[len(recs) // 2]
    for k, v in r.items():
        s = repr(v)
        print('  ', k, type(v).__name__, s[:200])
