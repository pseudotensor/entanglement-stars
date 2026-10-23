"""cw01: inventory of record kinds in icloud's saved written_report.md (ndjson) files.
Read-only on <internal>/solve_b_bulk. Output: <internal>/cw01_inventory.json"""
import json, sys, os, collections, hashlib
ROOT = '<internal>/solve_b_bulk'
dirs = []
for top in ['MOCK_filtered_physical_v1', 'MOCK_unitary_lead_run_v1']:
    for d in sorted(os.listdir(f'{ROOT}/{top}')):
        p = f'{ROOT}/{top}/{d}/written_report.md'
        if (d.startswith('even_') or d.startswith('odd_')) and os.path.isfile(p):
            dirs.append(p)
out = {}
for p in dirs:
    kinds = collections.Counter(); sub = collections.Counter(); keys = {}
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for line in f:
            h.update(line)
            r = json.loads(line)
            k = r.get('kind'); kinds[k] += 1
            d = r.get('data')
            if isinstance(d, dict):
                for kk in ('name', 'record', 'type', 'readout', 'label', 'phase'):
                    if kk in d and isinstance(d[kk], str):
                        sub[f'{k}:{kk}={d[kk]}'] += 1
                if k not in keys:
                    keys[k] = sorted(d.keys())[:60]
    out[p] = {'md5': h.hexdigest(), 'kinds': dict(kinds), 'sub_top': dict(sub.most_common(80)), 'keys': keys}
    print(p, h.hexdigest(), dict(kinds), flush=True)
os.makedirs('<internal>/clockwell', exist_ok=True)
json.dump(out, open('<internal>/cw01_inventory.json', 'w'), indent=1)
