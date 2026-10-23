"""cmp_v36u_ab.py (busi, PERF W2-P1/W2-P2): the bench_protocol 4.4 rule 1 comparison of two W2 segment runs (A: fixv4c or v36u at the defaults;
B: v36u with --dg-eigh-cache-gib and/or --dg-node-workers) from the same stored slice on the same card, and the gain.
Bitwise on every stored number but the walls: the json records and head compared key by key with engine_collapse_v36k_selftest's skip set
("seconds", "time", "stamp", "date", "host", "pid", "args", "script") plus "timing", "compute_s", "slice_saved", the per-process "v36u_perf" entry and v36/kernel_cache;
every string with its own run directory replaced by <OUT>; the fields, handoff and spectra npz arrays (strings the same way); the slices if any;
the logs with the wall fields masked and the "v36u perf:" worker-start line dropped. Usage: python cmp_v36u_ab.py DIR_A DIR_B TAG [LOG_A LOG_B]
(logs default to DIR/log.txt). Exit 0 on PASS, 1 on FAIL."""
import glob
import json
import math
import os
import re
import sys

import numpy as np

SKIP = ("seconds", "time", "stamp", "date", "host", "pid", "args", "script", "timing", "compute_s", "slice_saved", "v36u_perf", "kernel_cache")   # kernel_cache: v36/kernel_cache, a per-process counter exempt by 4.4 rule 1


def norm(x, out):
    if isinstance(x, str):
        return x.replace(out, "<OUT>")
    if isinstance(x, dict):
        return {k: norm(v, out) for k, v in x.items()}
    if isinstance(x, list):
        return [norm(v, out) for v in x]
    return x


def diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        ka = set(k for k in a if not any(s in k for s in SKIP)); kb = set(k for k in b if not any(s in k for s in SKIP))
        if ka != kb:
            return ["keys at %s: %s" % (path, sorted(ka ^ kb))]
        return [d for k in sorted(ka) for d in diff(a[k], b[k], path + "/" + k)]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return ["length at %s: %d vs %d" % (path, len(a), len(b))]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff(x, y, "%s[%d]" % (path, i))]
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return []
    return [] if a == b else ["%s: %r vs %r" % (path, a, b)]


def arr_diff(A, B, outA, outB):
    d = []
    ka, kb = set(A.files), set(B.files)
    if ka != kb:
        d.append("keys %s" % sorted(ka ^ kb))
    for k in sorted(ka & kb):
        x, y = A[k], B[k]
        if x.dtype.kind in "OUS" or y.dtype.kind in "OUS":
            if repr(norm(x.tolist(), outA)) != repr(norm(y.tolist(), outB)):
                d.append(k)
        elif x.shape != y.shape or x.dtype != y.dtype or not np.array_equal(x, y, equal_nan=(x.dtype.kind in "fc")):
            d.append(k)
    return d


def mask_log(path, out):
    L = []
    for l_ in open(path, errors="replace"):
        if "v36u perf:" in l_ or "Warning" in l_ or "csr = lambda" in l_:
            continue
        l_ = re.sub(r"[0-9.]+ s\b", "W s", l_.replace(out, "<OUT>"))
        l_ = re.sub(r"[0-9.]+ (min|h)\b", r"W \1", l_)
        L.append(l_.rstrip("\n"))
    return L


def main():
    A, B, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    logA = sys.argv[4] if len(sys.argv) > 5 else os.path.join(A, "log.txt")
    logB = sys.argv[5] if len(sys.argv) > 5 else os.path.join(B, "log.txt")
    JA = json.load(open(os.path.join(A, "collapse_%s.json" % tag))); JB = json.load(open(os.path.join(B, "collapse_%s.json" % tag)))
    d = diff(norm(JA["records"], A), norm(JB["records"], B), "records") + diff(norm(JA.get("head"), A), norm(JB.get("head"), B), "head")
    for k in ("fields", "handoff", "spectra"):
        pa, pb = os.path.join(A, "collapse_%s_%s.npz" % (tag, k)), os.path.join(B, "collapse_%s_%s.npz" % (tag, k))
        if os.path.exists(pa) != os.path.exists(pb):
            d.append("%s.npz present in one run only" % k)
        elif os.path.exists(pa):
            d += ["%s.npz: %s" % (k, q) for q in arr_diff(np.load(pa, allow_pickle=True), np.load(pb, allow_pickle=True), A, B)]
    SA = {os.path.basename(f).split("_T")[-1]: f for f in glob.glob(os.path.join(A, "states", "orbitals_%s_T*.npz" % tag))}
    SB = {os.path.basename(f).split("_T")[-1]: f for f in glob.glob(os.path.join(B, "states", "orbitals_%s_T*.npz" % tag))}
    if set(SA) != set(SB):
        d.append("slice sets %s" % sorted(set(SA) ^ set(SB)))
    for k in sorted(set(SA) & set(SB)):
        d += ["slice %s: %s" % (k, q) for q in arr_diff(np.load(SA[k], allow_pickle=True), np.load(SB[k], allow_pickle=True), A, B)]
    if os.path.exists(logA) and os.path.exists(logB):
        la, lb = mask_log(logA, A), mask_log(logB, B)
        if len(la) != len(lb):
            d.append("log lengths %d vs %d (walls masked)" % (len(la), len(lb)))
        d += ["log line %d: %r vs %r" % (i, x[:200], y[:200]) for i, (x, y) in enumerate(zip(la, lb)) if x != y][:5]
    else:
        d.append("a log is missing (%s, %s)" % (logA, logB))
    ok = not d
    print("%s: B %s vs A %s: %s" % ("PASS" if ok else "FAIL", B, A, ("bitwise: %d records, npz and logs (walls masked) identical, %d slices" % (len(JA["records"]), len(SA))) if ok else ("%d differences, the first: %s" % (len(d), [q[:300] for q in d[:8]]))))
    for i, (ra, rb) in enumerate(zip(JA["records"], JB["records"])):
        da, db = ra.get("dg_v36f") or {}, rb.get("dg_v36f") or {}
        if not da:
            continue
        sa, sb = da.get("seconds") or 0.0, db.get("seconds") or 0.0
        subs_a = [round(q.get("seconds") or 0.0, 1) for q in da.get("substeps", [])]; subs_b = [round(q.get("seconds") or 0.0, 1) for q in db.get("substeps", [])]
        p = rb.get("v36u_perf") or {}
        eg = p.get("eigh_cache", {}); nw = p.get("node_workers", {})
        print("record %d T/rs %s: dg step %.1f s (A) -> %.1f s (B), %+.1f %%; sub-steps A %s, B %s; record wall %.1f -> %.1f s; step eigendecompositions %s (requests), solves %s; B cache hits %s misses %s; node workers solves %s fallbacks %s errors %s" % (
            i, ra.get("T_over_rs"), sa, sb, (100.0 * (sb - sa) / sa) if sa else float("nan"), subs_a, subs_b, ra.get("seconds") or 0.0, rb.get("seconds") or 0.0,
            da.get("eigendecompositions"), da.get("solves"), eg.get("hits"), eg.get("misses"), nw.get("solves"), nw.get("fallbacks"), nw.get("errors")))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
