"""cw11: on-site baseline clock-weighted V_X probe table (#138 / run spec sec. 1.1, 7.1) from icloud's SAVED
controls report; read-only. Collects every check named weighted_probe_*, local_amputated_*, weighted_tangent*
(excluding display rows), keeps pass flags and the exact actual/target strings.
Output: <internal>/cw11_vertex_probes.json"""
import json, re, hashlib, collections
P = '<internal>/written_report.md'
OUT = '<internal>/cw11_vertex_probes.json'
h = hashlib.md5(open(P, 'rb').read()).hexdigest()
rows = []; npass = collections.Counter(); nfail = collections.Counter()
pat = re.compile(r'^(weighted_probe|local_amputated|weighted_tangent|tangent)')
for line in open(P):
    r = json.loads(line)
    if r.get('kind') != 'check':
        continue
    d = r['data']; nm = d.get('name', '')
    if not pat.search(nm) or 'display' in nm:
        continue
    fam = nm.split('_n')[0]
    (npass if d.get('pass') else nfail)[fam] += 1
    det = d.get('detail') or {}
    a = det.get('actual'); t = det.get('target')
    rows.append({'name': nm, 'pass': d.get('pass'), 'actual': a, 'target': t})
nonzero = [x for x in rows if isinstance(x['actual'], dict) and (x['actual'].get('re') not in (None, '0') or x['actual'].get('im') not in (None, '0'))]
json.dump({'report': P, 'md5': h, 'pass': dict(npass), 'fail': dict(nfail), 'rows': rows, 'nonzero': nonzero}, open(OUT, 'w'), indent=1)
print(P, h, dict(npass), dict(nfail), len(rows), len(nonzero))
for x in nonzero[:200]:
    print(x['name'], x['pass'], json.dumps(x['actual'])[:160], json.dumps(x['target'])[:160])

# --- part 2: the weighted_probe / weighted_tangent RECORDS (carry n and g explicitly) ---
from fractions import Fraction as Fr
recs = []
for line in open(P):
    r = json.loads(line); k = r.get('kind')
    if k not in ('weighted_probe', 'weighted_tangent'):
        continue
    d = r['data']; m = d.get('model', {})
    g = Fr(m.get('g')).limit_denominator(8) if m.get('g') is not None else None
    raw = d.get('raw') or {}; SN = d.get('S_N') or {}; lc = d.get('local_C') or {}
    ident = None
    if raw and SN and lc and g is not None and SN.get('im') == '0' and lc.get('im') == '0':
        ident = (Fr(raw['re']) == g * Fr(SN['re']) * Fr(lc['re'])) and Fr(raw.get('im', '0')) == 0
    nm = d.get('normalized') or {}
    recs.append({'kind': k, 'label': d.get('label'), 'n': m.get('n'), 'g': str(g), 'raw': raw, 'target': d.get('target'),
                 'S_N': SN, 'local_C': lc, 'QH': d.get('QH'), 'QL': d.get('QL'),
                 'normalized': {kk: v for kk, v in nm.items() if not isinstance(v, (dict, list)) or kk in ('re', 'im')},
                 'raw_eq_g_SN_localC': ident, 'keys': sorted(d.keys())})
out = json.load(open(OUT)); out['records'] = recs
json.dump(out, open(OUT, 'w'), indent=1)
print('records', len(recs), 'identity false:', [x['label'] for x in recs if x['raw_eq_g_SN_localC'] is False][:10])
for x in recs:
    if x['raw'] and x['raw'].get('re') not in ('0', None):
        print(x['kind'], x['n'], x['g'], x['label'], x['raw'], x['target'], x['normalized'], x['raw_eq_g_SN_localC'])
    elif x['kind'] == 'weighted_tangent':
        print(x['kind'], x['n'], x['g'], x['label'], x['keys'])
