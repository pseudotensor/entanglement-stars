"""cw04: kind counts of one ndjson report. Read-only."""
import json, sys, collections
c = collections.Counter()
for line in open(sys.argv[1]):
    c[json.loads(line).get('kind')] += 1
print(dict(c))
