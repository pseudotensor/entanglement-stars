"""cw10: list check names (with detail scalars) in one report, optionally filtered by a regex. Read-only."""
import json, sys, re
pat = re.compile(sys.argv[2]) if len(sys.argv) > 2 else None
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r.get('kind') == 'check':
        d = r['data']; nm = d.get('name', '')
        if pat is None or pat.search(nm):
            det = d.get('detail')
            print(nm, d.get('pass'), json.dumps(det)[:300] if det is not None else '')
