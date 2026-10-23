#!/usr/bin/env python
"""cmp_rd13m_v1.py (busi, inbox #387): two runs of the v36nq lineage compared for bitwise equality: the json records (every key but the
walls: keys named label, record_label, timing, rd13, rd13m, slice_saved (a path), or containing "seconds", "wall", or ending in "_s"), the head (the same rule),
the fields, spectra and handoff npz (every array, np.array_equal with NaN equal) and any slices of equal name; the args listed where they
differ. Exit 0 when everything compared is equal.
    python cmp_rd13m_v1.py DIR_A TAG_A DIR_B TAG_B [--slices-a DIR --slices-b DIR] [--out JSON]"""
import argparse
import glob
import json
import os
import sys

import numpy as np

SKIP_NAMES = {"label", "record_label", "timing", "rd13", "rd13m", "slice_saved"}   # slice_saved: the path (the arm's out dir); the slices are compared byte for byte


def skip(k):
    return k in SKIP_NAMES or "seconds" in k or "wall" in k or k.endswith("_s")


def diff(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if skip(k):
                continue
            if k not in a or k not in b:
                out.append("%s/%s: only in %s" % (path, k, "A" if k in a else "B"))
                continue
            diff(a[k], b[k], path + "/" + k, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append("%s: length %d vs %d" % (path, len(a), len(b)))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, "%s[%d]" % (path, i), out)
    else:
        if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
            return
        if a != b:
            out.append("%s: %r vs %r" % (path, a, b))


def npz_cmp(fa, fb, out, name):
    za, zb = np.load(fa, allow_pickle=True), np.load(fb, allow_pickle=True)
    ka, kb = set(za.files), set(zb.files)
    if ka != kb:
        out.append("%s: keys differ %s" % (name, sorted(ka ^ kb)))
    n = 0
    for k in sorted(ka & kb):
        x, y = za[k], zb[k]
        if x.shape != y.shape or x.dtype != y.dtype:
            out.append("%s[%s]: shape/dtype %s %s vs %s %s" % (name, k, x.shape, x.dtype, y.shape, y.dtype)); continue
        eq = np.array_equal(x, y, equal_nan=True) if x.dtype.kind in "fc" else np.array_equal(x, y)
        if not eq:
            out.append("%s[%s]: arrays differ" % (name, k))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a"); ap.add_argument("tag_a"); ap.add_argument("dir_b"); ap.add_argument("tag_b")
    ap.add_argument("--slices-a", default=None); ap.add_argument("--slices-b", default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ja = json.load(open(os.path.join(a.dir_a, "collapse_%s.json" % a.tag_a)))
    jb = json.load(open(os.path.join(a.dir_b, "collapse_%s.json" % a.tag_b)))
    out = []
    diff(ja.get("head"), jb.get("head"), "head", out)
    ra, rb = ja["records"], jb["records"]
    diff(ra, rb, "records", out)
    for k in [k for k in ja if k not in ("args", "head", "records", "script")]:
        diff(ja.get(k), jb.get(k), k, out)
    args_diff = {k: (ja["args"].get(k), jb["args"].get(k)) for k in sorted(set(ja["args"]) | set(jb["args"])) if ja["args"].get(k) != jb["args"].get(k)}
    narr = 0
    for s in ("_fields", "_spectra", "_handoff"):
        fa = os.path.join(a.dir_a, "collapse_%s%s.npz" % (a.tag_a, s)); fb = os.path.join(a.dir_b, "collapse_%s%s.npz" % (a.tag_b, s))
        if os.path.exists(fa) and os.path.exists(fb):
            narr += npz_cmp(fa, fb, out, s.strip("_"))
        else:
            out.append("%s: missing (%s %s)" % (s, os.path.exists(fa), os.path.exists(fb)))
    nsl = 0
    if a.slices_a and a.slices_b:
        for fa in sorted(glob.glob(os.path.join(a.slices_a, "*.npz"))):
            T = fa.rsplit("_T", 1)[-1]
            fb = glob.glob(os.path.join(a.slices_b, "*_T" + T))
            if not fb:
                out.append("slice T%s: missing in B" % T); continue
            narr += npz_cmp(fa, fb[0], out, "slice_T" + T); nsl += 1
    res = {"A": [a.dir_a, a.tag_a], "B": [a.dir_b, a.tag_b], "records": [len(ra), len(rb)], "arrays_compared": narr, "slices_compared": nsl,
           "differences": out[:200], "n_differences": len(out), "args_differ": args_diff, "bitwise": len(out) == 0 and len(ra) == len(rb)}
    print("cmp_rd13m_v1: %s vs %s: %d and %d records, %d arrays, %d slices compared; %d differences; args differ in %s -> %s" % (
        a.tag_a, a.tag_b, len(ra), len(rb), narr, nsl, len(out), sorted(args_diff), "BITWISE" if res["bitwise"] else "DIFFER"))
    for d in out[:20]:
        print("  " + d)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=str)
    sys.exit(0 if res["bitwise"] else 1)


if __name__ == "__main__":
    main()
