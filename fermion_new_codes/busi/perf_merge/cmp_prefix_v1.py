#!/usr/bin/env python
"""cmp_prefix_v1.py (busi, MERGE-1): bitwise check of a short arm against a longer one at the short arm's records (cmp_rd13m_v1's diff
rule: every key but walls and labels), plus the fields, spectra and handoff npz rows at those clocks (np.array_equal).
    python cmp_prefix_v1.py DIR_A TAG_A DIR_B TAG_B --rs RS --after T"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmp_rd13m_v1 import diff

ap = argparse.ArgumentParser(); ap.add_argument("da"); ap.add_argument("ta"); ap.add_argument("db"); ap.add_argument("tb")
ap.add_argument("--rs", type=float, required=True); ap.add_argument("--after", type=float, required=True)
a = ap.parse_args()
ja = json.load(open(os.path.join(a.da, "collapse_%s.json" % a.ta))); jb = json.load(open(os.path.join(a.db, "collapse_%s.json" % a.tb)))
ra = {round(r["T"] / a.rs, 6): r for r in ja["records"]}; rb = {round(r["T"] / a.rs, 6): r for r in jb["records"]}
ts = sorted(t for t in ra if t > a.after and t in rb)
out = []
for t in ts:
    diff(ra[t], rb[t], "T%.4f" % t, out)
narr = 0
for s in ("_fields", "_spectra", "_handoff"):
    za = np.load(os.path.join(a.da, "collapse_%s%s.npz" % (a.ta, s)), allow_pickle=True); zb = np.load(os.path.join(a.db, "collapse_%s%s.npz" % (a.tb, s)), allow_pickle=True)
    Ta = np.round(np.asarray(za["T"], float) / a.rs, 6); Tb = np.round(np.asarray(zb["T"], float) / a.rs, 6)
    for t in ts:
        ia = np.where(Ta == t)[0]; ib = np.where(Tb == t)[0]
        if len(ia) != 1 or len(ib) != 1:
            out.append("%s T%.4f: rows %d / %d" % (s, t, len(ia), len(ib))); continue
        for k in za.files:
            if k not in zb.files or za[k].ndim == 0 or za[k].shape[0] != len(Ta):
                continue
            x, y = za[k][ia[0]], zb[k][ib[0]]
            narr += 1
            if not np.array_equal(x, y, equal_nan=(x.dtype.kind in "fc")):
                out.append("%s[%s] T%.4f differs" % (s, k, t))
print("cmp_prefix_v1: %s vs %s at %d records %s, %d npz rows: %d differences -> %s" % (a.ta, a.tb, len(ts), ts, narr, len(out), "BITWISE" if not out else "DIFFER"))
for d in out[:20]:
    print("  " + d)
sys.exit(0 if not out else 1)
