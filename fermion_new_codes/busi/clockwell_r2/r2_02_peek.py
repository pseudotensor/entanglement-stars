"""Second reader: print the structure of record #idx of a kind, truncating long lists and strings. Read-only."""
import json, sys
path, kind, idx = sys.argv[1], sys.argv[2], int(sys.argv[3])
def trunc(x, d=0):
    if isinstance(x, dict): return {k: trunc(v, d+1) for k, v in x.items()}
    if isinstance(x, list):
        if len(x) > 6: return [trunc(v, d+1) for v in x[:4]] + ["...len=%d" % len(x)]
        return [trunc(v, d+1) for v in x]
    if isinstance(x, str) and len(x) > 90: return x[:60] + "...(%d)" % len(x)
    return x
i = 0
with open(path) as f:
    for line in f:
        if ('"kind":"%s"' % kind) not in line[:80]: continue
        r = json.loads(line)
        if r.get("kind") != kind: continue
        if i == idx:
            print(json.dumps(trunc(r), indent=1)); break
        i += 1
