#!/usr/bin/env python
"""compare_v33_gate_v1.py (builder analyst; S11): the gate of engine_collapse_v33 against engine_collapse_v32 with every version 33 option
off: the two runs' record jsons compared key by key (every number bitwise; the keys that hold wall-clock seconds or timings excluded, and
the top-level "script" and "args" entries, which name the version and its added arguments), and every array of the two fields npz files
compared bitwise. Prints the records, the json numbers compared, the fields arrays and their elements compared, the largest absolute
difference over the compared numbers and elements, the differing keys, and PASS or FAIL. Usage: compare_v33_gate_v1.py dirA/tagA dirB/tagB."""
import json
import os
import sys

import numpy as np

SKIP = {"seconds", "timing", "seconds_preparation", "utc_start", "utc_end", "seconds_hooks"}
STAT = {"numbers": 0, "max_abs_diff": 0.0}


def walk(x, y, path, diffs):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k in SKIP:
                continue
            if k not in x or k not in y:
                diffs.append("%s/%s: present in one run only" % (path, k)); continue
            walk(x[k], y[k], "%s/%s" % (path, k), diffs)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            diffs.append("%s: lengths %d and %d" % (path, len(x), len(y))); return
        for i, (p, q) in enumerate(zip(x, y)):
            walk(p, q, "%s[%d]" % (path, i), diffs)
    else:
        num = isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) and not isinstance(y, bool)
        if num:
            STAT["numbers"] += 1
            if x == x and y == y:
                STAT["max_abs_diff"] = max(STAT["max_abs_diff"], abs(float(x) - float(y)))
        same = (x == y) or (isinstance(x, float) and isinstance(y, float) and x != x and y != y)
        if not same:
            diffs.append("%s: %r against %r" % (path, x, y))


def main():
    a, b = sys.argv[1], sys.argv[2]
    da, ta = os.path.split(a); db, tb = os.path.split(b)
    Ja = json.load(open(os.path.join(da, "collapse_%s.json" % ta))); Jb = json.load(open(os.path.join(db, "collapse_%s.json" % tb)))
    diffs = []
    walk({k: v for k, v in Ja.items() if k not in ("script", "args")}, {k: v for k, v in Jb.items() if k not in ("script", "args")}, "", diffs)
    Za = np.load(os.path.join(da, "collapse_%s_fields.npz" % ta), allow_pickle=True); Zb = np.load(os.path.join(db, "collapse_%s_fields.npz" % tb), allow_pickle=True)
    fdiff = []; nel = 0; fmax = 0.0
    for k in sorted(set(Za.files) | set(Zb.files)):
        if k not in Za.files or k not in Zb.files:
            fdiff.append("%s: present in one file only" % k); continue
        x = Za[k]; y = Zb[k]
        if x.dtype == object or y.dtype == object:
            ok = len(x) == len(y) and all(np.array_equal(np.asarray(p), np.asarray(q)) for p, q in zip(x, y))
            for p, q in zip(x, y):
                p = np.asarray(p); q = np.asarray(q)
                if p.shape == q.shape and p.dtype.kind in "fc" and q.dtype.kind in "fc":
                    nel += p.size; fmax = max(fmax, float(np.max(np.abs(p - q))) if p.size else 0.0)
        else:
            ok = x.shape == y.shape and (np.array_equal(x, y) or (x.dtype.kind in "fc" and np.array_equal(x, y, equal_nan=True)))
            if x.shape == y.shape and x.dtype.kind in "fciu" and y.dtype.kind in "fciu":
                nel += x.size
                if x.size:
                    d = np.abs(x.astype(complex) - y.astype(complex))
                    d = d[np.isfinite(d)]
                    fmax = max(fmax, float(d.max()) if d.size else 0.0)
        if not ok:
            fdiff.append(k)
    nf = len(set(Za.files) | set(Zb.files))
    print("records: %d and %d; json numbers compared (wall-clock keys and script/args excluded): %d, differences %d, the largest absolute difference %.3e; fields npz arrays compared %d (elements %d), differing %d, the largest absolute difference %.3e" % (
        len(Ja.get("records", [])), len(Jb.get("records", [])), STAT["numbers"], len(diffs), STAT["max_abs_diff"], nf, nel, len(fdiff), fmax))
    for d in diffs[:40]:
        print("  json: " + d)
    for d in fdiff[:40]:
        print("  fields: " + d)
    print("GATE " + ("PASS" if not diffs and not fdiff else "FAIL"))


if __name__ == "__main__":
    main()
