"""cmp_v36w_busi.py (busi, v36w perf self-test, W2-P3; copy of cmp_v36u_busi.py): the v36w runs of selftest_v36w_box.sh against fixv4c's stored fixture
outputs (defaults) and the --streams 1 runs against fixv4c (prim_fs_A of the v36u stage; fx/fz), bitwise by engine_collapse_v36k_selftest's diff
with SKIP + (timing, compute_s, slice_saved, v36u_perf, kernel_cache), run directories and stage roots replaced by placeholders; npz arrays and
slices the same way; logs with the walls masked and the "v36u perf:"/"v36w perf:" lines dropped. Prints PASS/FAIL per comparison."""
import glob
import importlib.util
import json
import os
import re
import sys

import numpy as np

sys.dont_write_bytecode = True
ST = "<internal>/selftest_v36w_cpu"
REF = "<internal>/selftest_v36p_fixv4c_cpu"
spec = importlib.util.spec_from_file_location("T", os.path.join(REF, "code", "engine_collapse_v36k_selftest.py"))
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
T.SKIP = tuple(T.SKIP) + ("timing", "compute_s", "slice_saved", "v36u_perf", "kernel_cache")   # kernel_cache: v36/kernel_cache, the per-process counter 4.4 rule 1 exempts (the workers do the node solves)
ROOTS = (ST, REF, "<internal>/selftest_v36u_cpu", "<internal>/fixture")


def norm_s(s, out):
    s = s.replace(out, "<OUT>")
    for r in ROOTS:
        s = s.replace(r, "<ROOT>")
    s = re.sub(r'"streams": [01]\b', '"streams": <S>', s)   # v36w: the fields npz's "solver" string records the --streams value by construction (REVIEWA_perf_w2_v1 P3 A/B note)
    return s


def norm(x, out):
    if isinstance(x, str):
        return norm_s(x, out)
    if isinstance(x, dict):
        return {k: norm(v, out) for k, v in x.items()}
    if isinstance(x, list):
        return [norm(v, out) for v in x]
    return x


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


def load(out, tag):
    J = json.load(open(os.path.join(out, "collapse_%s.json" % tag)))
    Z = {k: np.load(p, allow_pickle=True) for k in ("fields", "handoff", "spectra") for p in [os.path.join(out, "collapse_%s_%s.npz" % (tag, k))] if os.path.exists(p)}
    S = {os.path.basename(f).split("_T")[-1]: f for f in glob.glob(os.path.join(out, "states", "orbitals_%s_T*.npz" % tag))}
    return J, Z, S


def mask_log(path, out):
    L = []
    for l_ in open(path, errors="replace"):
        if "v36u perf:" in l_ or "v36w perf:" in l_ or "UserWarning" in l_ or "csr = lambda" in l_:
            continue
        l_ = norm_s(l_, out)
        l_ = re.sub(r"[0-9.]+ s\b", "W s", l_)
        l_ = re.sub(r"[0-9.]+ min\b", "W min", l_)
        L.append(l_.rstrip("\n"))
    return L


def compare(name, outA, tagA, outB, tagB, logA=None, logB=None):
    try:
        JA, ZA, SA = load(outA, tagA); JB, ZB, SB = load(outB, tagB)
    except Exception as ex:
        print("FAIL %s: cannot load (%s: %s)" % (name, type(ex).__name__, ex)); return False
    d = T.diff(norm(JA["records"], outA), norm(JB["records"], outB), "records") + T.diff(norm(JA.get("head"), outA), norm(JB.get("head"), outB), "head")
    if set(ZA) != set(ZB):
        d.append("npz sets %s" % sorted(set(ZA) ^ set(ZB)))
    for k in sorted(set(ZA) & set(ZB)):
        d += ["%s.npz: %s" % (k, q) for q in arr_diff(ZA[k], ZB[k], outA, outB)]
    if set(SA) != set(SB):
        d.append("slice sets %s" % sorted(set(SA) ^ set(SB)))
    for k in sorted(set(SA) & set(SB)):
        d += ["slice %s: %s" % (k, q) for q in arr_diff(np.load(SA[k], allow_pickle=True), np.load(SB[k], allow_pickle=True), outA, outB)]
    nlog = ""
    if logA and logB and os.path.exists(logA) and os.path.exists(logB):
        la, lb = mask_log(logA, outA), mask_log(logB, outB)
        bad = [(i, x, y) for i, (x, y) in enumerate(zip(la, lb)) if x != y]
        if len(la) != len(lb):
            d.append("log lengths %d vs %d" % (len(la), len(lb)))
        d += ["log line %d: %r vs %r" % (i, x[:160], y[:160]) for i, x, y in bad[:3]]
        nlog = ", log %d lines (walls masked) identical" % len(la) if (not bad and len(la) == len(lb)) else ""
    ok = not d
    print("%s %s: %s" % ("PASS" if ok else "FAIL", name, ("bitwise: %d records, npz %s, %d slices%s" % (len(JA["records"]), sorted(ZA), len(SA), nlog)) if ok else ("%d differences, the first: %s" % (len(d), [q[:300] for q in d[:5]]))), flush=True)
    return ok


def perf(out, tag):
    try:
        J = json.load(open(os.path.join(out, "collapse_%s.json" % tag)))
    except Exception:
        return
    for i, r in enumerate(J["records"]):
        if "v36u_perf" in r:
            p = r["v36u_perf"]; eg = p["eigh_cache"]; nw = p["node_workers"]
            st = r.get("dg_v36f") or {}
            print("   %s record %d (T/rs %s): eigh cache hits %d misses %d; node workers solves %d fallbacks %d errors %s; step eigendecompositions %s, solves %s, %.1f s" % (
                os.path.relpath(out, ST), i, r.get("T_over_rs"), eg["hits"], eg["misses"], nw["solves"], nw["fallbacks"], nw["errors"], st.get("eigendecompositions"), st.get("solves"), st.get("seconds") or 0.0))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = []
    if which in ("all", "fx"):
        for d, tag in (("f_dg", "fx"), ("k_dg", "fx"), ("f_fr", "fx"), ("k_fr", "fx"), ("fz", "fz"), ("fa", "fa"), ("fs", "fs"), ("vii", "fz")):
            res.append(compare("(default flags) v36w fx/%s vs fixv4c fx/%s" % (d, d), os.path.join(ST, "fx", d), tag, os.path.join(REF, "fx", d), tag, os.path.join(ST, "fx", d, tag + ".log"), os.path.join(REF, "fx", d, tag + ".log")))
    if which in ("all", "chain"):
        U = "<internal>/selftest_v36u_cpu"   # v36u's stage: prim_fs_A there is fixv4c with two sub-steps per record
        res.append(compare("(default flags) v36w prim_vii vs fixv4c prim_vii", os.path.join(ST, "prim_vii"), "fz", os.path.join(REF, "prim_vii"), "fz", os.path.join(ST, "prim_vii", "prim.log"), os.path.join(REF, "prim_vii", "prim.log")))
        res.append(compare("(--streams 1) v36w prim_fs_S1 vs fixv4c prim_fs_A (prim, two sub-steps per record)", os.path.join(ST, "prim_fs_S1"), "fz", os.path.join(U, "prim_fs_A"), "fz", os.path.join(ST, "prim_fs_S1", "prim.log"), os.path.join(U, "prim_fs_A", "prim.log")))
        res.append(compare("(--streams 1 + cache 1 GiB + node workers 3) v36w prim_fs_B vs fixv4c prim_fs_A (prim, two sub-steps per record)", os.path.join(ST, "prim_fs_B"), "fz", os.path.join(U, "prim_fs_A"), "fz", os.path.join(ST, "prim_fs_B", "prim.log"), os.path.join(U, "prim_fs_A", "prim.log")))
        res.append(compare("(--streams 1 + cache 1 GiB + node workers 3) v36w on/fz vs fixv4c fx/fz", os.path.join(ST, "on", "fz"), "fz", os.path.join(REF, "fx", "fz"), "fz", os.path.join(ST, "on", "fz", "fz.log"), os.path.join(REF, "fx", "fz", "fz.log")))
        for d in ("on/fz", "prim_fs_B"):
            perf(os.path.join(ST, d), "fz")
    print("SUMMARY: %d of %d PASS -> %s" % (sum(res), len(res), "PASS" if all(res) else "FAIL"))
