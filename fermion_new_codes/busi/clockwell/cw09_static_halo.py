"""cw09: #145 static halo reporting scalars, exact rational, recomputed by busi from the admitted kernel
coefficients G0, G1, G2 and compared with icloud's SAVED static clock-well records (n = 2, 3, 7).
Read-only on <internal>/solve_b_bulk. Output: <internal>/cw09_static_halo.json
No propagation, no matrix rebuild: busi's own values are the spec's hand formulas evaluated in Fractions;
the record values are the saved exact strings.
"""
import json, hashlib, collections
from fractions import Fraction as Fr

ROOT = '<internal>/MOCK_filtered_clockwell_v1'
OUT = '<internal>/cw09_static_halo.json'
G0 = Fr(388627261303, 549873254400); G1 = Fr(456290907571, 3115948441600); G2 = Fr(180865117, 983983718400)

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()

def fs(x):
    return {'exact': f'{x.numerator}/{x.denominator}', 'float': float(x)}

busi = {
    'sum_G': G0 + 2*G1 + 2*G2,
    'site0_01_endpoint_copy': (19*G0 + 20*G1 + 3*G2) / 64,
    'site0_01_flat': (9*G0 + 8*G1 + G2) / 16,
    'site0_01_Nminus1_eq_1_rejected': (5*G0 + 6*G1 + G2) / 16,
    'site0_01_halo_removed_rejected': (9*G0 + 6*G1) / 32,
    'dH_dNminus1_site0_01': (G0 + 4*G1 + G2) / 32,
    'schur_R_Wminus1_n2': (2*G0 + 9*G1 + 6*G2) / 32,
    'schur_R_Wminus1_n3_n7': (2*G0 + 9*G1 + 9*G2) / 32,
}
res = {'busi_exact': {k: fs(v) for k, v in busi.items()}, 'records': {}}
for n in (2, 3, 7):
    p = f'{ROOT}/static_n{n}_v1/written_report.md'
    R = {'report': p, 'md5': md5(p), 'joining': {}}
    for line in open(p):
        r = json.loads(line); k = r['kind']; d = r['data']
        if k == 'halo_derivative_before_checks':
            R['schur_max'] = d['schur']['maximum']; R['same_site01_target'] = d['hand'].get('same_site01_target')
            R['schur_target'] = d['hand'].get('schur_target'); R['schur_ties'] = len(d['all_tie_pairs'])
            flat = d['flat_W_minus1']; alloc = d['allocated_Nminus1_Wminus1']
            ratios = collections.Counter(); absvals = collections.Counter()
            for key, v in flat.items():
                fv = Fr(v['re']) + 0  # entries are real rationals (spec sec. 4)
                av = Fr(alloc[key]['re'])
                assert Fr(v.get('im', 0)) == 0 and Fr(alloc[key].get('im', 0)) == 0
                ratios[str(av / fv) if fv != 0 else ('0/0' if av == 0 else 'x/0')] += 1
                absvals[(str(abs(fv)), str(abs(av)))] += 1
            R['centre_minus1_entries'] = len(flat)
            R['allocated_over_flat_ratios'] = dict(ratios)
            R['flat_vs_allocated_abs_values'] = [{'flat': a, 'allocated': b, 'count': c, 'flat_float': float(Fr(a)), 'allocated_float': float(Fr(b))}
                                                 for (a, b), c in sorted(absvals.items(), key=lambda t: -float(Fr(t[0][0])))]
        elif k == 'joining_exact_row_bounds':
            R['joining'][str(d['U'])] = {'Cjoin_squared': d['Cjoin_squared'], 'Hbound': d['Hbound'], 'L_squared': d['L_squared']}
        elif k == 'termination':
            R['termination'] = {kk: d.get(kk) for kk in ('status', 'checks', 'all_pass', 'sources_unchanged', 'wall', 'cpu', 'rss_kib')}
    sch = busi['schur_R_Wminus1_n2'] if n == 2 else busi['schur_R_Wminus1_n3_n7']
    R['schur_equal_busi'] = (Fr(R['schur_max']) == sch)
    R['same_site01_equal_busi'] = (Fr(R['same_site01_target']) == busi['dH_dNminus1_site0_01'])
    res['records'][n] = R
    print(n, R['md5'], R['schur_max'], R['schur_equal_busi'], R['same_site01_equal_busi'], R['allocated_over_flat_ratios'], R.get('termination'))
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps(res['busi_exact'], indent=1))

# --- part 2: saved static checks for the site-0 targets and the n7 full-support (site 3) interior targets ---
import re
pat = re.compile(r'^(site0_target|site0_flat_target|n7_.*full_support|n7_valley|full_support_spin_average|weighted_tangent_.*full_support)')
for n in (2, 3, 7):
    p = f'{ROOT}/static_n{n}_v1/written_report.md'
    rows = []
    for line in open(p):
        r = json.loads(line)
        if r['kind'] == 'check' and pat.search(r['data'].get('name', '')):
            d = r['data']; det = d.get('detail') or {}
            a = det.get('actual'); e = det.get('expected')
            rows.append({'name': d['name'], 'pass': d['pass'], 'exact_equal': det.get('exact_equal'),
                         'actual': a if not isinstance(a, dict) else a.get('re', a), 'expected': e if not isinstance(e, dict) else e.get('re', e)})
    res['records'][n]['target_checks'] = rows
    s0 = [x for x in rows if x['name'] == 'site0_target_(0, 1)']
    res['records'][n]['site0_01_record'] = s0[0]['actual'] if s0 else None
    res['records'][n]['site0_01_equal_busi'] = (Fr(s0[0]['actual']) == busi['site0_01_endpoint_copy']) if s0 else None
    print(n, 'target checks', len(rows), 'all pass', all(x['pass'] for x in rows), 'site0 equal busi', res['records'][n]['site0_01_equal_busi'])
    for x in rows:
        if x['name'].startswith('n7') or 'full_support' in x['name']:
            if x['actual'] not in ('0',):
                print('  ', x['name'], x['actual'], x['pass'])
json.dump(res, open(OUT, 'w'), indent=1)
