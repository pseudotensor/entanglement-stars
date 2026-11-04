#!/usr/bin/env python3
"""Format compare_v1.json (from evolvedcut_route_compare_v1.py) into markdown table fragments. Box only; reads json, prints md."""
import json, sys

TOL_ID = 2.1e-10
d = json.load(open(sys.argv[1]))
r2 = json.load(open(sys.argv[2]))['rows']   # tables.json rows, for route 2's primary totals
prim = {int(round(r['clock_rs'])): r for r in r2 if int(r['site_shift']) == 0}


def tot(r, key):
    t = r['totals']
    v = t.get(key)
    return v


print('### Clock table (weighted, 16 kappa, kappa <= 10)\n')
print('| T/r_s | dtype | route 1 cut: R/a (source); interior sites | route 2 primary cut: R/a (label); interior sites | route 2 partitions saved | like-for-like partition | route 1 excess (paper range) | route 2 polar excess at that partition | difference (1 - 2) | within 880 x 2.1e-10 = 1.85e-7? | complement-side difference (1 - 2) | within 1.85e-7? |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for c in d['clocks']:
    T = c['T']
    p = prim[T]
    r1cut = '%.4f (%s); %d' % (c['r1_R_cut'], 'record outer trapping zero' if c['r1_status'] == 'root' else '2gE, no root', c['r1_interior_sites'])
    r2cut = '%.4f (%s); %d' % (c['r2_radius'], c['r2_cut_label'], c['r2_primary_interior_sites'])
    if c['matched']:
        W = c['weighted']
        comp1 = sum(ch['weight'] * (ch['C_complement'][0] - ch['held_complement'][0]) for ch in c['channels'])
        comp2 = sum(ch['weight'] * (ch['C_complement'][1] - ch['held_complement'][1]) for ch in c['channels'])
        dc = comp1 - comp2
        print('| %d | %s | %s | %s | %s | shift %+d (%d sites) | %.9f | %.9f | %+.3e | %s | %+.3e | %s |' % (
            T, c['dtype1'], r1cut, r2cut, ','.join(map(str, c['r2_available_interior_sites'])), c['r2_shift'], c['r1_interior_sites'],
            W['excess_polar'][0], W['excess_polar'][1], W['excess_polar'][2], 'yes' if abs(W['excess_polar'][2]) <= 1.848e-7 else 'NO',
            dc, 'yes' if abs(dc) <= 1.848e-7 else 'NO'))
    else:
        print('| %d | %s | %s | %s | %s | none (route 1 at %d sites) | %.9f | not computed by route 2 at %d sites | - | - | - | - |' % (
            T, c['dtype1'], r1cut, r2cut, ','.join(map(str, c['r2_available_interior_sites'])), c['r1_interior_sites'],
            c['r1_weighted']['excess'], c['r1_interior_sites']))

print('\n### Route 2 primary (applied-root) excess beside route 1 (record-root) excess, different cuts, not a comparison\n')
print('| T/r_s | route 1 excess (record root) | route 2 polar excess (applied root, shift 0) | route 2 raw excess (shift 0) |')
print('|---|---|---|---|')
for c in d['clocks']:
    T = c['T']; p = prim[T]
    t = p['totals']
    r1e = c['weighted']['excess_polar'][0] if c['matched'] else c['r1_weighted']['excess']
    print('| %d | %.6f | %s | %s |' % (T, r1e, '%.6f' % t['orthonormal_control_excess'] if t.get('orthonormal_control_excess') is not None else 'unavailable',
                                      '%.6f' % t['raw_excess'] if t.get('raw_excess') is not None else 'unavailable'))

print('\n### Weighted component sums at the matched partitions (route 1, route 2, difference)\n')
print('| T/r_s | S_X(C) r1 | S_X(C) r2 polar | diff | S_X(P_-) r1 | S_X(P_-) r2 | diff | S_X(P_0) r1 | S_X(P_0) r2 | diff | S_X(C) r2 raw | diff r1 - raw |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for c in d['clocks']:
    if not c['matched']:
        continue
    W = c['weighted']
    raw = W['C_raw']
    print('| %d | %.9f | %.9f | %+.2e | %.9f | %.9f | %+.2e | %.9f | %.9f | %+.2e | %s | %s |' % (
        c['T'], W['C_polar'][0], W['C_polar'][1], W['C_polar'][2], W['held'][0], W['held'][1], W['held'][2],
        W['flat'][0], W['flat'][1], W['flat'][2], ('%.9f' % raw[1]) if raw else 'unavailable (gate)', ('%+.2e' % raw[2]) if raw else '-'))

print('\n### Per channel at the matched partitions (unweighted, nats; differences route 1 minus route 2)\n')
print('| T/r_s | kappa | S_X(C) r1 | S_X(C) r2 polar | d ext | d complement | d held (ext; comp) | d flat (ext; comp) | d excess (ext) | d C vs r2 raw | r1 ext-int | r2 polar ext-int | ext within 2.1e-10 | comp within 2.1e-10 |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for c in d['clocks']:
    if not c['matched']:
        continue
    for ch in c['channels']:
        print('| %d | %d | %.10f | %.10f | %+.2e | %+.2e | %+.2e; %+.2e | %+.2e; %+.2e | %+.2e | %s | %+.2e | %+.2e | %s | %s |' % (
            c['T'], ch['kappa'], ch['C'][0], ch['C'][1], ch['C'][2], ch['C_complement'][2], ch['held'][2], ch['held_complement'][2],
            ch['flat'][2], ch['flat_complement'][2], ch['excess_polar'][2],
            ('%+.2e' % ch['C_raw'][2]) if ch['C_raw'][2] is not None else 'unavailable',
            ch['r1_ext_minus_int'], ch['r2_polar_ext_minus_int'],
            'yes' if abs(ch['C'][2]) <= TOL_ID else 'NO',
            'yes' if all(abs(ch[k][2]) <= TOL_ID for k in ('C_complement', 'held_complement', 'flat_complement')) else 'NO'))

print('\n### Maxima over matched clocks\n')
keys = ['C', 'C_complement', 'held', 'held_complement', 'flat', 'flat_complement', 'excess_polar']
for k in keys:
    print('- max |d %s| = %.3e' % (k, max(c['max_abs_channel_diff'][k] for c in d['clocks'] if c['matched'])))
print('- max |r1 ext-int| = %.3e; max |r2 polar ext-int| = %.3e' % (
    max(c['max_abs_r1_ext_minus_int'] for c in d['clocks'] if c['matched']),
    max(c['max_abs_r2_polar_ext_minus_int'] for c in d['clocks'] if c['matched'])))
cr = [c['max_abs_channel_diff']['C_raw'] for c in d['clocks'] if c['matched'] and c['max_abs_channel_diff']['C_raw'] is not None]
print('- max |d C vs r2 raw| = %.3e' % max(cr))
