"""compare_oracle_v6m_v2.py (solve builder 7; engine spec 48.2 (3)): version 1 with the follow-mode runs' extension/failed_reads excluded, the
oracle's simulated live writer read under a loaded box CPU (a wall-clock retry counter, no computed number; v6l's own gate differed there
identically, 36 and 14 against 0). Version 1: compare_oracle_v6l_v1.py with v6m's added keys (col_diag) excluded, so that v6m with --col-diag
on is compared with v6l on every v6l number. Version 1 of v6l's:
compare_oracle_v6l_v1.py (solve builder 6; engine spec 47.1 (2) (v)): page_history_v6l with --collector-admit velocity against v6k; from compare_oracle_v6k_v1.py: page_history_v6k with option (v) off against v6j on the
history oracle's runs: every npz array bitwise, every json number equal, wall-clock keys and v6k's two new
input flags in settings (collar_split, collar_split_windows) excluded (v6j's added keys compared too). Usage: compare_oracle_v6k_v1.py DIR_v6j DIR_v6k"""
import json, os, sys
import numpy as np
A, B = sys.argv[1:3]
SKIP = {"failed_reads", "col_diag", "collector_admit", "collar_split", "collar_split_windows", "seconds", "wall_s", "wall", "utc_start", "utc_end", "written", "code", "script",
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
