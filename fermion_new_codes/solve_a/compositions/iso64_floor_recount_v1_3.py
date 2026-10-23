# COMP4 v1.3: recount table 5b.B under reviewer A's floor-resolution test (solve_cold8 section (2)).
# RESOLVED if |S7(isoB) - S7(B2)| > F_meas(isoB) + F_meas(B2), else "at floor level"; ratio beside;
# "within 10 percent" (of the larger magnitude) only as a descriptive tag. Reads the v1.2 json only.
import json
J = "<internal>/ISO64_light_v1_2.json"
d = json.load(open(J))
print("rule (v1.2):", d["rule"])
ps = d["partial_sum"]
k0 = next(iter(ps)); print("cell keys:", list(ps[k0].keys()))
rows = []
for key, c in ps.items():
    T, cut = key.split("|")
    rows.append((float(T), float(cut), c))
rows.sort()
def get(c, *names):
    for n in names:
        if n in c: return c[n]
    raise KeyError(names)
n_floor = n_res = n_res10 = 0
out = []
for T, cut, c in rows:
    si = get(c, "S_iso", "S7_iso", "iso"); sb = get(c, "S_B2")
    fi = get(c, "F_iso", "F_meas_iso"); fb = get(c, "F_B2")
    diff = abs(si - sb); fl = fi + fb
    res = diff > fl
    w10 = diff <= 0.10 * max(abs(si), abs(sb))
    ratio = si / sb if sb != 0 else float("nan")
    if res:
        n_res += 1; n_res10 += w10
    else:
        n_floor += 1
    out.append((T, cut, si, sb, fl, diff, ratio, res, w10, c.get("agree")))
print(f"cells {len(rows)}: at floor level {n_floor}; resolved {n_res}, of them within 10 percent {n_res10}")
for T, cut, si, sb, fl, diff, ratio, res, w10, ag in out:
    tag = ("RESOLVED" if res else "at floor level") + (", within 10 percent" if (res and w10) else "")
    print(f"T {T:4.0f} cut {cut:2.0f}: iso {si:+.4e} b2 {sb:+.4e} |diff| {diff:.2e} floor {fl:.2e} ratio {ratio:+.3f}  {tag}  [v1.2: {ag}]")
json.dump({"input": J, "rule": "resolved iff |S7 iso - S7 B2| > F_meas iso + F_meas B2; within-10-percent tag descriptive",
           "counts": {"cells": len(rows), "at_floor": n_floor, "resolved": n_res, "resolved_within_10pct": n_res10},
           "cells": [dict(T=o[0], cut=o[1], S_iso=o[2], S_b2=o[3], floor=o[4], absdiff=o[5], ratio=o[6], resolved=o[7], within10=o[8]) for o in out]},
          open("<internal>/ISO64_floor_recount_v1_3.json", "w"), indent=1)
