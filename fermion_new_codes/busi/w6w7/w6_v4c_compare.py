#!/usr/bin/env python3
"""w6_v4c_compare.py: v4c json against v4b json, every common per-pair key: bitwise equal or max |difference| (box CPU)."""
import json, sys
import numpy as np
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))  # v4b, v4c
for blk in ('pairs_outer', 'pairs_inner'):
    A, B = a[blk], b[blk]
    assert len(A) == len(B), (blk, len(A), len(B))
    same, diff = [], {}
    for k in sorted(set(A[0]) & set(B[0])):
        x = [p.get(k) for p in A]; y = [p.get(k) for p in B]
        if x == y:
            same.append(k); continue
        try:
            d = np.nanmax(np.abs(np.array(x, float) - np.array(y, float)))
        except Exception:
            d = 'non-numeric'
        diff[k] = d
    print(blk, len(A), 'pairs;', len(same), 'keys bitwise equal; differing:', {k: ('%.3g' % v if not isinstance(v, str) else v) for k, v in diff.items()})
offa, offb = a['pairs_offsets'], b['pairs_offsets']
nd = set()
for o in offa:
    for p, q in zip(offa[o], offb[o]):
        for k in set(p) & set(q):
            if p[k] != q[k]:
                nd.add(k)
print('offsets: keys differing anywhere:', sorted(nd))
sa, sb = a['stats_outer'], b['stats_outer']
print('stats_outer keys differing:', sorted(k for k in set(sa) & set(sb) if sa[k] != sb[k]))
P, Q = a['pairs_outer'], b['pairs_outer']
print('per pair, outer roots (first 8 and the max |.|): T_a | col2 v4b -> v4c | col4 v4b -> v4c | rep defect v4b -> v4c')
idx = list(range(8))
for i in idx:
    p, q = P[i], Q[i]
    print('%7.4g | %+.6g -> %+.6g | %+.6g -> %+.6g | %+.4g -> %+.3g' % (p['T_a'], p['col2_cut_assignment'], q['col2_cut_assignment'], p['col4_continued_reference'], q['col4_continued_reference'], p['rep_defect'], q['rep_defect']))
for k in ('col2_cut_assignment', 'col4_continued_reference', 'rep', 'rep_defect', 'col1_flux_old_cut', 'col3_averaging_commutator', 'dM'):
    print('sum %s: v4b %.6g, v4c %.6g; max |.| v4b %.4g, v4c %.4g' % (k, sum(p[k] for p in P), sum(q[k] for q in Q), max(abs(p[k]) for p in P), max(abs(q[k]) for q in Q)))
print('inner: max |rep defect| v4b %.3g, v4c %.3g' % (max(abs(p['rep_defect']) for p in a['pairs_inner']), max(abs(q['rep_defect']) for q in b['pairs_inner'])))
