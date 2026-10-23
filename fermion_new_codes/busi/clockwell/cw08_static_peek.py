"""cw08: print top-level scalar fields (<=300 chars) and list/dict sizes of chosen kinds in a static report. Read-only."""
import json, sys
p, kinds = sys.argv[1], sys.argv[2].split(',')
for line in open(p):
    r = json.loads(line); k = r.get('kind')
    if k in kinds:
        d = r['data']; print('=====', k)
        if not isinstance(d, dict):
            print(str(d)[:1500]); continue
        for kk, v in d.items():
            if isinstance(v, (int, float, bool)) or v is None or (isinstance(v, str) and len(v) <= 300):
                print(' ', kk, '=', v)
            elif isinstance(v, dict):
                sc = {a: b for a, b in v.items() if isinstance(b, (int, float, bool)) or (isinstance(b, str) and len(b) <= 200)}
                print(' ', kk, ': dict', len(v), json.dumps(sc)[:1500])
            else:
                print(' ', kk, ':', type(v).__name__, len(v), json.dumps(v)[:400] if isinstance(v, list) else '')
