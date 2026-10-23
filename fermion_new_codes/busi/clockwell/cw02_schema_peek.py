"""cw02: print truncated schema samples (first record of chosen kinds) from one written_report.md.
Usage: cw02_schema_peek.py REPORT kind1,kind2,...  Read-only."""
import json, sys
p, kinds = sys.argv[1], sys.argv[2].split(',')
idx = {k: int(sys.argv[3]) if len(sys.argv) > 3 else 0 for k in kinds}
seen = {k: 0 for k in kinds}
def trunc(o, d=0):
    if isinstance(o, dict):
        return {k: trunc(v, d+1) for k, v in list(o.items())[:80]}
    if isinstance(o, list):
        if len(o) > 6: return [trunc(x, d+1) for x in o[:3]] + [f'...len={len(o)}']
        return [trunc(x, d+1) for x in o]
    if isinstance(o, str) and len(o) > 120: return o[:120] + '...'
    return o
with open(p) as f:
    for line in f:
        r = json.loads(line)
        k = r.get('kind')
        if k in seen:
            if seen[k] == idx[k]:
                print('=====', k); print(json.dumps(trunc(r['data']), indent=1)[:12000])
            seen[k] += 1
