"""Second reader: list key paths (lists sampled at index 0 only) with scalar samples for record #idx of a kind. Read-only."""
import json, sys
path, kind, idx = sys.argv[1], sys.argv[2], int(sys.argv[3])
def walk(x, p, out):
    if isinstance(x, dict):
        for k, v in x.items(): walk(v, p + "." + str(k), out)
    elif isinstance(x, list):
        out.append("%s  list[%d]" % (p, len(x)))
        if x and not isinstance(x[0], (dict, list)): out.append("%s[0..] = %s" % (p, str(x[:8])[:150]))
        elif x: walk(x[0], p + "[0]", out)
    else:
        s = str(x); out.append("%s = %s" % (p, s[:100]))
i = 0
with open(path) as f:
    for line in f:
        if ('"kind":"%s"' % kind) not in line[:80]: continue
        r = json.loads(line)
        if r.get("kind") != kind: continue
        if i == idx:
            out = []; walk(r, "", out); print("\n".join(out)); break
        i += 1
