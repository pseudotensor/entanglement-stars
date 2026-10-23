"""cw13: md5 manifest of every icloud box input that busi's clock-well reads used (reports, npz arrays, time files).
Read-only on <internal>/solve_b_bulk. Output: <internal>/INPUT_MD5SUMS.txt  (md5  path)"""
import json, os, hashlib
ROOT = '<internal>/solve_b_bulk'
EXT = '<internal>/cw05_extract.json'
OUT = '<internal>/INPUT_MD5SUMS.txt'

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()

ext = json.load(open(EXT))
paths = []
for key, P in ext['physical'].items():
    d = f"{ROOT}/{P['top']}/{P['name']}"
    paths.append(P['report'])
    if P.get('time'):
        paths.append(P['time']['file'])
    for c in P['conversion']:
        paths.append(f"{d}/{c['density_array']}")
    for e in P['entropy']:
        if e.get('raw_vector_array'):
            paths.append(f"{d}/{e['raw_vector_array']}")
    if '/even_' in key:
        reg = P['plans']['registered']; pure = P['plans']['pure']; first_op = None
        with open(P['report']) as f:
            for line in f:
                if first_op is None and line.startswith('{"kind":"numerical_action"'):
                    r = json.loads(line)['data']
                    if r.get('kind') == 'operator':
                        first_op = r['hamiltonian']['array']; paths.append(f'{d}/{first_op}')
                if line.startswith('{"kind":"action"'):
                    r = json.loads(line)['data']; m = r['metadata']
                    if m.get('role') == 'primary' and m.get('start') == 0.0 and (
                            (m.get('plan') == reg and m.get('end') in (0.0, 1.0, 2.0, 4.0)) or (m.get('plan') == pure and m.get('end') in (0.0, 1.0))):
                        paths.append(f"{d}/{r['raw_vector']['array']}")
for key, Q in ext['readouts'].items():
    paths.append(Q['report'])
    if Q.get('time'):
        paths.append(Q['time']['file'])
for n in (2, 3, 7):
    paths.append(f'{ROOT}/MOCK_filtered_clockwell_v1/static_n{n}_v1/written_report.md')
paths.append(f'{ROOT}/MOCK_unitary_lead_run_v1/controls_v1/written_report.md')
OV = '<internal>/adapter_overlay_v1'
for gs in ('0', '0p5', '2'):
    for nm in (f'intake_even_n3_g{gs}_v1', f'even_n3_g{gs}_v1'):
        paths += [f'{OV}/upper/{nm}/written_report.md', f'{OV}/upper/{nm}/records.ndjson', f'{OV}/logs/{nm}_time.txt']
    paths.append(f'<internal>/busi_intake_review_n3_g{gs}_v1.md')
paths.append(f'{ROOT}/MOCK_filtered_comparison_readouts_v1/reviews_controls_v1/redteam_filtered_adapter_controls_v1.md')
paths.append(f'{ROOT}/MOCK_filtered_comparison_readouts_v1/source_v1/run_mock_filtered_comparison_readouts_v1.py')
seen = []
for p in paths:
    if p not in seen:
        seen.append(p)
with open(OUT, 'w') as f:
    for p in seen:
        f.write(f'{md5(p)}  {p}\n')
print(len(seen), 'inputs ->', OUT)
