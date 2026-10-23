"""Second reader: schema survey of a written_report.md (one JSON per line). Read-only."""
import json, sys, collections
path = sys.argv[1]
maxlen = int(sys.argv[2]) if len(sys.argv) > 2 else 600
kinds = collections.Counter(); first = {}
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except Exception: print("NONJSON", line[:200]); continue
        k = r.get("kind"); kinds[k] += 1
        if k not in first: first[k] = line
for k, c in kinds.items(): print(k, c)
print("----")
for k, l in first.items(): print("##", k, l[:maxlen]); print()
