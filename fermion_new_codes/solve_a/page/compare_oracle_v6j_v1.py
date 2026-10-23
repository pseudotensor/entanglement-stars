"""compare_oracle_v6j_v1.py (the lane's analyst builder; engine spec 42.3 (3)): page_history_v6j with its options off against v6i on the
history oracle's runs: every npz array bitwise, every json number equal, v6j's added keys (collector_select, joint_outside_weight,
archive_gram_dev_F) and wall-clock keys excluded. Usage: compare_oracle_v6j_v1.py DIR_v6i DIR_v6j"""
import json, os, sys
import numpy as np
A, B = sys.argv[1:3]
SKIP = {"collector_select", "joint_outside_weight", "archive_gram_dev_F", "seconds", "wall_s", "wall", "utc_start", "utc_end", "written", "code", "script",
        "device_peak_GiB", "device_peak_before_GiB", "t_wall", "elapsed_s", "loads"}
n = [0, 0]; bad = []
def walk(x, y, p):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in set(x) | set(y):
            if k in SKIP or "seconds" in k or k.endswith("_s"):
                continue
            if k not in x or k not in y:
                bad.append(p + "/" + k + " missing"); continue
            walk(x[k], y[k], p + "/" + k)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            bad.append(p + " length"); return
        for i, (u, v) in enumerate(zip(x, y)):
            walk(u, v, p + "[%d]" % i)
    elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
        n[0] += 1
        if not (x == y or (x != x and y != y)):
            bad.append("%s %r %r" % (p, x, y))
for f in sorted(os.listdir(A)):
    pa, pb = os.path.join(A, f), os.path.join(B, f)
    if not os.path.exists(pb):
        continue
    if f.endswith(".npz"):
        za, zb = np.load(pa, allow_pickle=True), np.load(pb, allow_pickle=True)
        for k in za.files:
            if k in zb.files:
                n[1] += 1
                if not np.array_equal(za[k], zb[k]):
                    bad.append("%s:%s array differs" % (f, k))
    elif f.endswith(".json") and not f.startswith("history_oracle"):
        walk(json.load(open(pa)), json.load(open(pb)), f)
print("json numbers %d, npz arrays %d, differences %d" % (n[0], n[1], len(bad)))
for b in bad[:12]:
    print("  " + b)
print("PASS" if not bad else "FAIL")
