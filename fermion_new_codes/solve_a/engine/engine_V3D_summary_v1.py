#!/usr/bin/env python
"""engine_V3D_summary_v1.py (builder analyst): V3D's records (engine_validate3d_v1/v2 JSONs) in one table: per case and count, the particle
numbers and energies of the open cube and of the chain at the same E_F, the levels below m, and (version-2 records) the 3D degenerate sets
against the chain's multiplets.  Args: <V3D folder> <out.md>."""
import glob
import json
import os
import sys

d, out = sys.argv[1], sys.argv[2]
L = ["# V3D records (open cube, prescribed N = 1 - 0.3 exp(-r^2/2s^2), v = -B (r/s) exp(-r^2/2s^2), m a 0.3; engine_validate3d_v1/v2)", "",
     "| case | script | levels below m (3D) | count | E_F | N_B 3D | N_B chain | E 3D | E chain | 3D sets (energy(multiplicity)) | chain sets |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for f in sorted(glob.glob(os.path.join(d, "v3d_*.json"))):
    D = json.load(open(f))
    tag = os.path.basename(f)[4:-5]
    lv = D.get("positive_levels_below_m", [])
    for c, r in D["results"].items():
        s3 = " ".join("%.4f(%d)" % (x, n) for x, n in r.get("levels_3d_sets", [])[:8]) or "-"
        sc = " ".join("%.4f(%d)" % (x, n) for x, n in r.get("levels_chain_sets", [])[:6]) or "-"
        L.append("| %s | %s | %d | %s | %.5f | %.2f | %.2f | %.4f | %.4f | %s | %s |" % (tag, D["script"].replace("engine_validate3d_", "").replace(".py", ""), len(lv), c,
                                                                                    r["EF"], r["NB_3d"], r["NB_chain"], r["E_3d"], r["E_chain"], s3, sc))
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
