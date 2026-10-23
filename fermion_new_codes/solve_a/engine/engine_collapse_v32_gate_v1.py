#!/usr/bin/env python
"""engine_collapse_v32_gate_v1.py (builder analyst; engine spec 32.1 (3): "Gate: v32 with its options off on the solver-independence
check's three sigma/a 8 records, bitwise the stored chk31 output"): two engine_collapse runs (dir/tag) compared: the json leaf by leaf
(the key "script" set aside and printed; numbers bitwise as IEEE doubles (two NaN equal), ints, bools, strings and None exactly, a type
change a difference; every differing leaf printed with its path and both values; a leaf is wall-clock when its key is "seconds" or a
"timing" entry ending in "_s"; keys present in one file only listed with their values), and every array of the fields, handoff and spectra
npz files bitwise (dtype, shape and bytes; object arrays element by element; strings exactly). Verdict PASS: no differing json leaf other
than wall-clock, the keys present in one file only all None-valued entries of "args" (version 32's options at their defaults), and every
array of the three npz files bitwise equal. Args: --ref dir/tag --new dir/tag --md --json. Exit 0 on PASS."""
import argparse
import json
import math
import os
import struct
import sys

import numpy as np


def leaf_same(x, y):
    if type(x) is not type(y):
        return False
    if isinstance(x, float):
        if math.isnan(x) and math.isnan(y):
            return True
        return struct.pack("<d", x) == struct.pack("<d", y)
    return x == y


def walk(x, y, keys, diff, only):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y), key=str):
            if k not in x:
                only.append((keys + [k], "new", y[k]))
            elif k not in y:
                only.append((keys + [k], "ref", x[k]))
            else:
                walk(x[k], y[k], keys + [k], diff, only)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            diff.append((keys, "length %d" % len(x), "length %d" % len(y)))
            return
        for i, (u, v) in enumerate(zip(x, y)):
            walk(u, v, keys + [i], diff, only)
    elif not leaf_same(x, y):
        diff.append((keys, x, y))


def path_str(keys):
    return "".join(("[%d]" % k) if isinstance(k, int) else ("." + str(k)) for k in keys)


def is_wall(keys):
    k = keys[-1] if keys else None
    if k == "seconds":
        return True
    return len(keys) >= 2 and keys[-2] == "timing" and isinstance(k, str) and k.endswith("_s")


def arr_same(A, B):
    A = np.asarray(A); B = np.asarray(B)
    if A.dtype != B.dtype or A.shape != B.shape:
        return False
    if A.dtype == object:
        for u, v in zip(A.ravel(), B.ravel()):
            if not arr_same(u, v):
                return False
        return True
    return A.tobytes() == B.tobytes()


def load(run):
    d, tag = os.path.split(run)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = {k: np.load(os.path.join(d, "collapse_%s_%s.npz" % (tag, k)), allow_pickle=True) for k in ("fields", "handoff", "spectra")}
    return J, Z


def main():
    ap = argparse.ArgumentParser()
    for k in ("ref", "new", "md", "json"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    Jr, Zr = load(a.ref); Jn, Zn = load(a.new)
    sr, sn = Jr.pop("script", None), Jn.pop("script", None)
    diff, only = [], []
    walk(Jr, Jn, [], diff, only)
    wall = [q for q in diff if is_wall(q[0])]
    comp = [q for q in diff if not is_wall(q[0])]
    only_ok = [q for q in only if q[0] and q[0][0] == "args" and len(q[0]) == 2 and q[2] is None]
    only_bad = [q for q in only if q not in only_ok]
    nrec = (len(Jr.get("records", [])), len(Jn.get("records", [])))
    npz = {}
    for k in ("fields", "handoff", "spectra"):
        fr, fn = set(Zr[k].files), set(Zn[k].files)
        common = sorted(fr & fn)
        nd = [f for f in common if not arr_same(Zr[k][f], Zn[k][f])]
        npz[k] = {"compared": len(common), "differ": nd, "ref_only": sorted(fr - fn), "new_only": sorted(fn - fr)}
    npz_ok = all(not v["differ"] and not v["ref_only"] and not v["new_only"] for v in npz.values())
    ok = (not comp) and (not only_bad) and npz_ok
    L = ["## engine_collapse_v32's gate: %s against the stored %s (engine_collapse_v32_gate_v1.py; engine spec 32.1 (3))" % (a.new, a.ref), "",
         "- scripts: %s (new) against %s (stored); records %d (new) against %d (stored)" % (sn, sr, nrec[1], nrec[0]),
         "- json: %d differing leaves, %d of them wall-clock (keys %s), %d other" % (
             len(diff), len(wall), ", ".join(sorted(set(str(q[0][-1]) for q in wall))) or "none", len(comp))]
    L += ["  - %s: %r (new) against %r (stored)" % (path_str(q[0]), q[2], q[1]) for q in diff]
    L += ["- json keys present in one file only: %d (%d of them None-valued entries of args)" % (len(only), len(only_ok))]
    L += ["  - %s, in the %s file only: %r" % (path_str(q[0]), q[1], q[2]) for q in only]
    for k in ("fields", "handoff", "spectra"):
        v = npz[k]
        L.append("- %s npz: %d arrays compared bitwise, %d differ%s%s%s" % (
            k, v["compared"], len(v["differ"]), (" (" + ", ".join(v["differ"]) + ")") if v["differ"] else "",
            ("; in the stored file only: " + ", ".join(v["ref_only"])) if v["ref_only"] else "", ("; in the new file only: " + ", ".join(v["new_only"])) if v["new_only"] else ""))
    L += ["- verdict: %s" % ("PASS: bitwise in every computed quantity (every differing json leaf is wall-clock; every array of the three npz files equal bit for bit)"
                             if ok else "FAIL: not bitwise (see the leaves and arrays above)"),
          "- every number above is printed by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump({"ref": a.ref, "new": a.new, "script_ref": sr, "script_new": sn, "records": nrec, "differing_leaves": [[path_str(q[0]), q[1], q[2]] for q in diff],
               "wall_clock_leaves": len(wall), "other_leaves": [[path_str(q[0]), q[1], q[2]] for q in comp],
               "one_file_only": [[path_str(q[0]), q[1], q[2]] for q in only], "npz": npz, "pass": ok}, open(a.json, "w"), indent=1, default=str)
    print("\n".join(L))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
