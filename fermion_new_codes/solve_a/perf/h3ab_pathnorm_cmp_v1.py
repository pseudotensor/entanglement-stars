"""h3ab_pathnorm_cmp_v1.py (solve 8a7b5274): the records and the fields npz 'label' of two arms compared after replacing each arm's run-dir path
(gate/H3AB_<arm>_<stamp>) by one token; names every key that still differs and prints the raw differing strings once. Usage: A_DIR B_DIR"""
import json, os, sys
import numpy as np
TAG = "C64_s8_ell8_tau2_twofield_v4L"
A, B = [d.rstrip("/") for d in sys.argv[1:3]]
WALL = {"seconds", "timing", "slice_saved"}
def norm(o, d):
    if isinstance(o, dict): return {k: norm(v, d) for k, v in o.items() if k not in WALL and not k.endswith("_s")}
    if isinstance(o, list): return [norm(v, d) for v in o]
    if isinstance(o, str): return o.replace(d, "<RUN>")
    return o
ra = json.load(open(os.path.join(A, "collapse_%s.json" % TAG)))["records"]
rb = json.load(open(os.path.join(B, "collapse_%s.json" % TAG)))["records"]
raw = set(); bad = []
for x, y in zip(ra, rb):
    for k in set(x) | set(y):
        if k in WALL or k.endswith("_s"): continue
        if norm(x.get(k), A) != norm(y.get(k), B): bad.append((x.get("T_over_rs"), k))
        elif x.get(k) != y.get(k): raw.add(k)
print("records", len(ra), len(rb), "differing after path normalisation:", bad[:10], len(bad), "; keys differing only by the run path:", sorted(raw))
la = str(np.load(os.path.join(A, "collapse_%s_fields.npz" % TAG), allow_pickle=True)["label"])
lb = str(np.load(os.path.join(B, "collapse_%s_fields.npz" % TAG), allow_pickle=True)["label"])
print("fields label equal after path normalisation:", la.replace(A, "<RUN>") == lb.replace(B, "<RUN>"), "; raw equal:", la == lb)
x = ra[-1].get("engine_v34"); print("engine_v34 sample A:", str(x)[:300])
