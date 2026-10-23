#!/usr/bin/env python
"""s46v32_exhaust_v11.py (busi, inbox #481): from s46v32_tables_v1.md table S1, the time at which the only integrated bound the stored moments give,
<I - Pi>(t) <= (sqrt(<B>_H) + sigma_E t)^2 with sigma_E^2 = ||phi||^2 = ||phi1||^2 + ||phi2||^2 (upper), reaches 1; prints per slice and the range."""
import sys, math
rows = []; inS1 = False
for line in open(sys.argv[1]):
    if line.startswith("### "): inS1 = line.startswith("### S1.")
    if not inS1: continue
    c = [x.strip() for x in line.split("|")]
    if len(c) > 8 and (c[1].startswith("prod ") or c[1].startswith("s8 ")):
        B = float(c[3]); phi = float(c[5]) + float(c[6]); dT = float(c[11])
        t = (1 - math.sqrt(B)) / math.sqrt(phi); rows.append((c[1], B, phi, t, t / dT))
print("| slice | <B>_H (upper) | ||phi||^2 | t_exhaust (time units) | t_exhaust / dT_rec |")
print("|---|---|---|---|---|")
for r in rows: print("| %s | %.6f | %.4f | %.3f | %.3f |" % r)
lo = min(rows, key=lambda r: r[3]); hi = max(rows, key=lambda r: r[3])
print("\nRange over all %d slices: t_exhaust %.2f (%s) to %.2f (%s) time units, %.3f to %.3f of the record step." % (len(rows), lo[3], lo[0], hi[3], hi[0], lo[4], hi[4]))
