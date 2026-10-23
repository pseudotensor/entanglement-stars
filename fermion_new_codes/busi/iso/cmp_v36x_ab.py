#!/usr/bin/env python
"""cmp_v36x_ab.py (busi, analyst; #299 PERF ISO-A2/A3): one H100 A/B arm against the reference arm, by bench_protocol 4.4 (1).
Bitwise on the records with the walls removed (the v36n self-test's strip: seconds, timing, handoff_s, oracle34_s, propagation_prev_s,
slice_saved, compute_s) and the per-process entries removed (v36x, v36b, v36.kernel_cache), the head, the fields, spectra and handoff npz and
the slices written in both arms (md5); the args may differ only by the arm's options. Prints the per-record walls of both arms and the gain
(T_B/T_A <= 0.90 adopts, 0.90 to 0.97 only with another build, above 0.97 not adopted: bench_protocol 4.4 (3)).
    python cmp_v36x_ab.py --selftest-lib .../engine_collapse_v36n_selftest.py --a DIR_A --b DIR_B [--allow-args pair_workers,nk_jvp_cache_gib,nk_jvp_defer]
DIR_A/DIR_B each hold run/collapse_<tag>.json (+ _fields, _spectra, _handoff npz) and states/ (the slices); one tag per dir.
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import sys


def load(path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec = importlib.util.spec_from_file_location("v36n_selftest_lib", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def drop_proc(o):
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if k in ("v36x", "v36b"):
                continue
            if k == "v36" and isinstance(v, dict):
                v = {kk: vv for kk, vv in v.items() if kk != "kernel_cache"}
            out[k] = drop_proc(v)
        return out
    if isinstance(o, list):
        return [drop_proc(v) for v in o]
    return o


def arm(d):
    js = glob.glob(os.path.join(d, "run", "collapse_*.json"))
    if len(js) != 1:
        raise SystemExit("%s: expected one run json, found %s" % (d, js))
    p = js[0][:-5]
    return {"json": js[0], "fields": p + "_fields.npz", "spectra": p + "_spectra.npz", "handoff": p + "_handoff.npz",
            "slices": sorted(glob.glob(os.path.join(d, "states", "orbitals_*.npz")))}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-lib", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--allow-args", default="pair_workers,nk_jvp_cache_gib,nk_jvp_defer")
    a = ap.parse_args()
    S = load(a.selftest_lib)
    A, B = arm(a.a), arm(a.b)
    JA, JB = json.load(open(A["json"])), json.load(open(B["json"]))
    diffs = []
    ra = [json.dumps(drop_proc(S.strip(q)), sort_keys=True, default=float) for q in JA["records"]]
    rb = [json.dumps(drop_proc(S.strip(q)), sort_keys=True, default=float) for q in JB["records"]]
    if len(ra) != len(rb):
        diffs.append("record count %d against %d" % (len(ra), len(rb)))
    bad = [JA["records"][i].get("T_over_rs") for i in range(min(len(ra), len(rb))) if ra[i] != rb[i]]
    if bad:
        diffs.append("records differ at T/rs %s" % bad)
    if json.dumps(S.strip(JA["head"]), sort_keys=True, default=float) != json.dumps(S.strip(JB["head"]), sort_keys=True, default=float):
        diffs.append("heads differ")
    for key in ("fields", "spectra", "handoff"):
        if os.path.exists(A[key]) and os.path.exists(B[key]):
            S.npz_equal(A[key], B[key], key, diffs)
        else:
            diffs.append("%s missing" % key)
    na = {os.path.basename(p).split("_T")[-1]: p for p in A["slices"]}
    nb = {os.path.basename(p).split("_T")[-1]: p for p in B["slices"]}
    for k in sorted(set(na) | set(nb)):
        if k not in na or k not in nb:
            diffs.append("slice T%s in one arm only" % k)
        elif md5(na[k]) != md5(nb[k]):
            diffs.append("slice T%s md5 differs" % k)
    allow = set(x for x in a.allow_args.split(",") if x)
    ka = set(JA["args"]) ^ set(JB["args"])
    va = {k for k in set(JA["args"]) & set(JB["args"]) if JA["args"][k] != JB["args"][k]} - {"out", "tag", "slice_dir", "resume_run", "resume_slice"}
    if (ka | va) - allow:
        diffs.append("args differ beyond %s: %s" % (sorted(allow), sorted((ka | va) - allow)))
    print("A %s\nB %s" % (A["json"], B["json"]))
    print("slices compared: %s" % (sorted(set(na) & set(nb)) or "none"))
    wa = [q.get("seconds") or 0.0 for q in JA["records"]]
    wb = [q.get("seconds") or 0.0 for q in JB["records"]]
    for q, x, y in zip(JA["records"], wa, wb):
        print("  T/rs %-9s wall A %8.1f s  B %8.1f s  B/A %.3f" % (q.get("T_over_rs"), x, y, (y / x) if x else float("nan")))
    ta, tb = sum(wa), sum(wb)
    r = tb / ta if ta else float("nan")
    print("segment wall A %.1f s, B %.1f s, B/A %.3f (%s)" % (ta, tb, r, "adopt (>= 10 percent)" if r <= 0.90 else ("3 to 10 percent: only with another build" if r <= 0.97 else "below 3 percent: not adopted")))
    for q in JB["records"]:
        for k in ("v36x", "v36b"):
            if k in q:
                print("  B T/rs %s %s %s" % (q.get("T_over_rs"), k, json.dumps(q[k], default=float)))
    print("VERDICT %s" % ("BITWISE" if not diffs else "DIFFERS: " + "; ".join(diffs)))
    raise SystemExit(0 if not diffs else 1)


if __name__ == "__main__":
    main()
