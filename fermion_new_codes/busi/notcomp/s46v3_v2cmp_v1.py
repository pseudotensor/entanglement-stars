#!/usr/bin/env python3
"""s46v3_v2cmp_v1.py (busi, NOTCOMP 4.6 v3): v2's <B_lin> (from CORE_HF T1's printed 4-figure Var_C and M_H) against v3's unrounded
<B_lin>, and v3's Var L and M_H rounded to 4 significant figures, as T1 prints them, to show that the rounding accounts for the gap.
Arithmetic on stored numbers only. Usage: s46v3_v2cmp_v1.py <s46v3 dir> <s46 dir>; writes <s46v3 dir>/s46v3_v2cmp_v1.md."""
import glob
import json
import os
import sys

d3, d2 = sys.argv[1], sys.argv[2]
v2 = {(s['history'], float(s['T_rs'])): s['B_lin'] for s in json.load(open(os.path.join(d2, 's46_branch_v1.json')))['slices']}


def r4(x):
    return float('%.4g' % x)


out = ['| slice | v2 <B_lin> (T1 printed digits) | v3 <B_lin> (unrounded) | rel. gap | v3 inputs rounded to 4 figures | rel. to v2 |',
       '|---|---|---|---|---|---|']
for p in sorted(glob.glob(os.path.join(d3, '*_T*.json'))):
    if len(os.path.basename(p)[:-5].split('_')) != 2:
        continue
    J = json.load(open(p))
    key = (J['history'], float(J['T_over_rs']))
    V = J['variance']
    b3 = J['B']['B_lin']
    br = sum(4 * r4(V[l]['Var_L']) / r4(V[l]['M_H']) ** 2 for l in J['B']['radii'])
    b2 = v2[key]
    out.append('| %s %g | %.6f | %.6f | %+.1e | %.6f | %+.1e |' % (key[0], key[1], b2, b3, (b3 - b2) / b2, br, (br - b2) / b2))
hdr = [x for x in out if x.startswith('| slice') or x.startswith('|---')]
body = sorted([x for x in out if x not in hdr], key=lambda s: (not s.startswith('| prod'), float(s.split('|')[1].split()[1])))
txt = '\n'.join(hdr + body) + '\n'
open(os.path.join(d3, 's46v3_v2cmp_v1.md'), 'w').write(txt)
print(txt)
