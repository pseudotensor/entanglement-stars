#!/usr/bin/env python
"""h3_touched_shares_v1.py (busi lane, PERF H3): from a py-spy raw (collapsed) profile of engine_twofield_v9c2 (H3_run's r6, box CPU), the share
of the record samples in the code each v9c4 option touches. Lines are v9c2's (v9c3c's are the same up to light_sector_walk_v9, then +1).
  A1: stacks through band_upper_v36 (every caller: the walk's ladders, reopenings, light_walk_v36).
  B1: stacks through light_sector_walk_v9 at its ladder block (v9c2 lines 1604 to 1621) or its reopening block (1632 to 1636): the jobs the pool
      takes (sparse K(s), Ks @ Ks, band conversion, Cholesky ladders); its parallel ceiling is this share times (1 - 1/min(N, anchors)).
  C1: stacks through _v9c_weights on a V9cPlus (line 865: the Gram and the column selection) and, inside, the D2H line 819 alone (on the box CPU
      there is no device transfer; the share is the host copy of W).
Usage. python h3_touched_shares_v1.py RAW [--drop SUBSTR ...]
"""
import re
import sys

a = sys.argv[1:]; fn = a[0]; drop = []
i = 1
while i < len(a):
    if a[i] == "--drop":
        drop.append(a[i + 1]); i += 2
    else:
        i += 1
walk_re = re.compile(r"light_sector_walk_v9 \(engine_twofield_v9c2\.py:(\d+)\)")
N = 0; Nall = 0
cnt = {"A1 band_upper_v36": 0, "B1 walk ladder 1604-1621": 0, "B1 walk reopen 1632-1636": 0, "B1 both blocks": 0,
       "C1 _v9c_weights:865": 0, "C1 _v9c_gram:819 (D2H line) under _v9c_weights:865": 0, "walk light_sector_walk_v9 (all)": 0,
       "walk host (no eigT)": 0}
for ln in open(fn, errors="replace"):
    ln = ln.rstrip("\n")
    if not ln:
        continue
    stack, _, c = ln.rpartition(" ")
    try:
        c = int(c)
    except ValueError:
        continue
    Nall += c
    if any(d in stack for d in drop):
        continue
    N += c
    if "band_upper_v36 (" in stack:
        cnt["A1 band_upper_v36"] += c
    ls = [int(x) for x in walk_re.findall(stack)]
    if ls:
        cnt["walk light_sector_walk_v9 (all)"] += c
        if "eigT (" not in stack:
            cnt["walk host (no eigT)"] += c
    lad = any(1604 <= x <= 1621 for x in ls); reo = any(1632 <= x <= 1636 for x in ls)
    if lad:
        cnt["B1 walk ladder 1604-1621"] += c
    if reo:
        cnt["B1 walk reopen 1632-1636"] += c
    if lad or reo:
        cnt["B1 both blocks"] += c
    if "_v9c_weights (engine_twofield_v9c2.py:865)" in stack:
        cnt["C1 _v9c_weights:865"] += c
        if "_v9c_gram (engine_twofield_v9c2.py:819)" in stack:
            cnt["C1 _v9c_gram:819 (D2H line) under _v9c_weights:865"] += c
print("file %s: %d of %d samples kept (drop %s)" % (fn, N, Nall, drop))
for k, v in cnt.items():
    print("  %-55s %7d samples  %6.3f%% of the kept record samples" % (k, v, 100.0 * v / max(N, 1)))
