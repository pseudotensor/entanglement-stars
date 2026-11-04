#!/usr/bin/env python3
"""EVOLVEDCUT route comparison v1 (inbox #248): first route (solve_a EVOLVEDCUT_slices_v1, prod history)
against the second route (solve_b EVOLVEDCUT_secondroute_v1), channel by channel and clock by clock.

Read-only on every input. Pure json arithmetic (no linear algebra). Run on the box:
  nice -n 10 python evolvedcut_route_compare_v1.py --r1-dir <first-route evolvedcut dir> \
      --r2-results <results.json> --r2-tables <tables.json> --out <out.json>

Matching rule: both routes cut X = {sites n > n_in} with both spinor components together, on the same slice
fields (N_applied, v_applied of the stored slice). Route 1's interior site count is n_Xc/2 from its own record
(cut at the record's outer trapping zero, or 2gE at T12). Route 2 saves partitions at its primary interior count
(applied-field root, or 2gE at T12) and at -1/+1 sites. A pair is compared only where the interior site counts
are equal; otherwise the clock is listed as not comparable (different cut).
"""
import argparse, glob, json, os, re, hashlib

CLOCKS = [12, 15, 16, 20, 24, 25, 28, 30, 35, 40]
TOL_ID = 2.1e-10          # route 1's stated per-channel pure-cut identity defect / purity gap bound
SRC_FLOOR64 = 8.7e-6      # route 1's stated complex64 source floor (relative, sources; not propagated to entropy)


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--r1-dir', required=True)
    ap.add_argument('--r2-results', required=True)
    ap.add_argument('--r2-tables', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    r2 = json.load(open(a.r2_results))['rows']
    r2t = json.load(open(a.r2_tables))['rows']
    r2idx = {(int(round(r['clock_rs'])), int(r['kappa'])): r for r in r2}

    out = {'inputs': {}, 'clocks': []}
    out['inputs']['r2_results'] = [a.r2_results, md5(a.r2_results)]
    out['inputs']['r2_tables'] = [a.r2_tables, md5(a.r2_tables)]
    for T in CLOCKS:
        p1 = os.path.join(a.r1_dir, 'evolvedcut_prod_T%07.2f.json' % T)
        d1 = json.load(open(p1))
        out['inputs']['r1_T%d' % T] = [p1, md5(p1)]
        chans1 = {int(c['kappa']): c for c in d1['channels']}
        n_in1 = None
        rows = []
        r2k1 = r2idx[(T, 1)]
        prim = int(r2k1['interior_site_count'])
        avail = {int(c['interior_sites']): int(c['site_shift']) for c in r2k1['cuts']}
        c1k1 = chans1[1]['cuts']['root']
        n_in1 = int(c1k1['n_Xc']) // 2
        rec = {'T': T, 'dtype1': d1['dtype'], 'r1_R_cut': d1['root']['R_cut'], 'r1_status': d1['root']['status'],
               'r1_source': d1['root']['source'], 'r1_applied_outer': d1['root'].get('applied_outer'),
               'r1_interior_sites': n_in1, 'r2_cut_label': r2k1['cut_label'], 'r2_radius': r2k1['radius_over_a'],
               'r2_primary_interior_sites': prim, 'r2_available_interior_sites': sorted(avail),
               'r2_source_dtype': r2k1['source_dtype'], 'r1_slice': d1['slice'], 'r2_source': r2k1['source']}
        if n_in1 not in avail:
            rec['matched'] = False
            rec['note'] = 'route 1 interior %d sites not among route 2 partitions %s' % (n_in1, sorted(avail))
            # still record route-1 sums and route-2 primary sums for the side-by-side listing
            rec['r1_weighted'] = {k: d1['weighted']['root'][k] for k in ('S_X', 'S_X_vac', 'S_X_P0', 'excess')}
            out['clocks'].append(rec)
            continue
        shift = avail[n_in1]
        rec['matched'] = True
        rec['r2_shift'] = shift
        W = {'C_polar': [0.0, 0.0], 'C_raw': [0.0, 0.0], 'held': [0.0, 0.0], 'flat': [0.0, 0.0],
             'excess_polar': [0.0, 0.0], 'excess_raw': [0.0, 0.0]}
        raw_ok_all = True
        for k in range(1, 11):
            c1 = chans1[k]['cuts']['root']
            assert int(c1['n_Xc']) // 2 == n_in1
            r2k = r2idx[(T, k)]
            cut2 = [c for c in r2k['cuts'] if int(c['site_shift']) == shift][0]
            w1 = float(chans1[k]['weight']); w2 = float(r2k['weight'])
            assert w1 == w2 == 16 * k
            s1C = c1['S_X']; s1H = c1['S_X_vac']; s1F = c1['flat']['S_X_P0']; e1 = c1['excess']
            s2P = cut2['orthonormal_control']['physical_exterior_entropy']
            s2R = cut2['raw']['physical_exterior_entropy'] if cut2['raw'].get('global_raw_physical') else None
            s2H = cut2['held']['entropy']; s2F = cut2['flat']['entropy']
            row = {'kappa': k, 'weight': w1,
                   'C': [s1C, s2P, s1C - s2P], 'C_raw': [s1C, s2R, (s1C - s2R) if s2R is not None else None],
                   'held': [s1H, s2H, s1H - s2H], 'flat': [s1F, s2F, s1F - s2F],
                   'excess_polar': [e1, s2P - s2H, e1 - (s2P - s2H)],
                   'excess_raw': [e1, (s2R - s2H) if s2R is not None else None,
                                  (e1 - (s2R - s2H)) if s2R is not None else None],
                   'C_complement': [c1['S_Xc'], cut2['orthonormal_control']['interior']['entropy'],
                                    c1['S_Xc'] - cut2['orthonormal_control']['interior']['entropy']],
                   'held_complement': [c1['S_Xc_vac'], s2H, c1['S_Xc_vac'] - s2H],
                   'flat_complement': [c1['flat']['S_Xc_P0'], s2F, c1['flat']['S_Xc_P0'] - s2F],
                   'r1_ext_minus_int': c1['S_X'] - c1['S_Xc'],
                   'r2_polar_ext_minus_int': s2P - cut2['orthonormal_control']['interior']['entropy'],
                   'r2_raw_ext_minus_int': (s2R - cut2['raw']['interior']['entropy']) if s2R is not None else None,
                   'r1_identity_defect': c1['pure_cut_identity']['defect'], 'r1_purity_gap': c1['purity_gap'],
                   'r2_polar_rounding_tol': cut2['orthonormal_control']['exterior']['rounding_tolerance'],
                   'r2_polar_clip': cut2['orthonormal_control']['exterior']['clipping_displacement']}
            for key in ('C', 'C_complement', 'held', 'held_complement', 'flat', 'flat_complement', 'excess_polar'):
                row[key + '_within_TOL_ID'] = abs(row[key][2]) <= TOL_ID
            rows.append(row)
            W['C_polar'][0] += w1 * s1C; W['C_polar'][1] += w1 * s2P
            W['held'][0] += w1 * s1H; W['held'][1] += w1 * s2H
            W['flat'][0] += w1 * s1F; W['flat'][1] += w1 * s2F
            W['excess_polar'][0] += w1 * e1; W['excess_polar'][1] += w1 * (s2P - s2H)
            if s2R is None:
                raw_ok_all = False
            else:
                W['C_raw'][0] += w1 * s1C; W['C_raw'][1] += w1 * s2R
                W['excess_raw'][0] += w1 * e1; W['excess_raw'][1] += w1 * (s2R - s2H)
        for key in W:
            W[key].append(W[key][0] - W[key][1])
        if not raw_ok_all:
            W['C_raw'] = None; W['excess_raw'] = None
        rec['weighted'] = W
        rec['r1_weighted_stored'] = {k: d1['weighted']['root'][k] for k in ('S_X', 'S_X_vac', 'S_X_P0', 'excess')}
        # route 2's own stored weighted terms at that shift (tables.json) as an arithmetic cross-check
        t2 = [r for r in r2t if int(round(r['clock_rs'])) == T and int(r['site_shift']) == shift][0]
        rec['r2_tables_totals'] = t2['totals']
        rec['channels'] = rows
        rec['max_abs_channel_diff'] = {key: max(abs(r[key][2]) for r in rows) for key in ('C', 'C_complement', 'held', 'held_complement', 'flat', 'flat_complement', 'excess_polar')}
        rec['max_abs_channel_diff']['C_raw'] = max((abs(r['C_raw'][2]) for r in rows if r['C_raw'][2] is not None), default=None)
        rec['max_abs_r1_ext_minus_int'] = max(abs(r['r1_ext_minus_int']) for r in rows)
        rec['max_abs_r2_polar_ext_minus_int'] = max(abs(r['r2_polar_ext_minus_int']) for r in rows)
        out['clocks'].append(rec)
    json.dump(out, open(a.out, 'w'), indent=1)
    for rec in out['clocks']:
        if rec['matched']:
            W = rec['weighted']
            m = rec['max_abs_channel_diff']
            print('T%2d shift %+d n_in %d  excess r1 %.9f r2polar %.9f d %.3e | maxchan C %.2e Ccomp %.2e held %.2e flat %.2e raw %s hc %.2e fc %.2e | ext-int r1 %.2e r2 %.2e'
                  % (rec['T'], rec['r2_shift'], rec['r1_interior_sites'], W['excess_polar'][0], W['excess_polar'][1],
                     W['excess_polar'][2], m['C'], m['C_complement'], m['held'], m['flat'], m['C_raw'], m['held_complement'], m['flat_complement'],
                     rec['max_abs_r1_ext_minus_int'], rec['max_abs_r2_polar_ext_minus_int']))
        else:
            print('T%2d unmatched: %s' % (rec['T'], rec['note']))


if __name__ == '__main__':
    main()
