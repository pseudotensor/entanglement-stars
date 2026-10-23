"""cmp_v36v_busi.py (busi, v36v dg self-test; busi/notes/W2_pass12625_ruling_reviewa_v1.md sec. 3): phase 1 of selftest_v36v_box.sh (options off) against
fixv4c's stored fixture outputs, bitwise by engine_collapse_v36k_selftest's diff with SKIP + (timing, compute_s, slice_saved), every string with its own
run directory and the stage root replaced by placeholders; the npz arrays and the slices compared the same way; the logs compared with the wall fields
masked (cmp_v36u_busi.py's machinery). Phase 2 (options on): per run the v36v dg stall lines, the step lines, the piece-count retry lines, STOP and
Traceback, and the records' E/E0 against the options-off reference (gate 1e-10 for the same quadrature, ruling (c); reported only for (d))."""
import glob
import importlib.util
import json
import os
import re
import sys

import numpy as np

sys.dont_write_bytecode = True
ST = "<internal>/selftest_v36v_cpu"
REF = "<internal>/selftest_v36p_fixv4c_cpu"
spec = importlib.util.spec_from_file_location("T", os.path.join(REF, "code", "engine_collapse_v36k_selftest.py"))
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
T.SKIP = tuple(T.SKIP) + ("timing", "compute_s", "slice_saved")
ROOTS = (ST, REF, "<internal>/fixture")


def norm_s(s, out):
    s = s.replace(out, "<OUT>")
    for r in ROOTS:
        s = s.replace(r, "<ROOT>")
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
        if "v36u perf:" in l_ or "UserWarning" in l_ or "csr = lambda" in l_:
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


def records(out, tag):
    try:
        return json.load(open(os.path.join(out, "collapse_%s.json" % tag)))["records"]
    except Exception:
        return None


def report(name, out, tag, log, ref_out, ref_tag, gate):
    """phase 2: the watched lines of one options-on run and its E/E0 against the options-off reference."""
    print("== %s (%s)" % (name, os.path.relpath(out, ST)), flush=True)
    L = open(log, errors="replace").read().splitlines() if os.path.exists(log) else []
    st = [l_ for l_ in L if l_.startswith("  v36v dg ")]
    print("   v36v dg stall lines: %d (stalled %d, tracked-set changed %d)" % (len(st), sum("stalled" in l_ for l_ in st), sum("the tracked set changed" in l_ for l_ in st)))
    for l_ in st[:3] + (st[-1:] if len(st) > 3 else []):
        print("     " + l_.strip()[:260])
    for l_ in L:
        if " v36k chart step to " in l_ or "the next piece count" in l_ or "the deepest level" in l_ or "the sub-step halved" in l_ or "STOP" in l_ or "Traceback" in l_ or "Error" in l_ or "; v36v: " in l_:
            print("     " + norm_s(l_.strip(), out)[:300])
    R = records(out, tag); R0 = records(ref_out, ref_tag)
    ok = True
    if R is None:
        print("   no records json"); ok = gate is None
    else:
        d = []
        for r in R:
            r0 = [q for q in (R0 or []) if q.get("T_over_rs") == r.get("T_over_rs")]
            d.append((r.get("T_over_rs"), r.get("E_over_E0"), (r.get("E_over_E0") - r0[0].get("E_over_E0")) if (r0 and r.get("E_over_E0") is not None) else None))
        print("   records (T/rs, E/E0, minus the options-off reference): " + "; ".join("%s %.13f %s" % (t, e, ("%+.2e" % q) if q is not None else "-") for t, e, q in d))
        for i, r in enumerate(R):
            for s_ in ((r.get("dg_v36f") or {}).get("substeps") or []):
                if s_.get("stall_events") is not None:
                    print("   record %d sub-step %s/%s: converged %s in %s, stall_events %d, beta_final %s, max_change_final %s" % (i, s_.get("sub"), s_.get("nsub"), s_.get("converged"), s_.get("iterations"), len(s_["stall_events"]), s_.get("beta_final"), s_.get("max_change_final")))
        if gate is not None:
            mx = max([abs(q) for _t, _e, q in d if q is not None] or [float("inf")])
            ok = mx <= gate and not any("Traceback" in l_ for l_ in L)
            print("   %s: max |E/E0 difference| %.2e (gate %g)" % ("PASS" if ok else "FAIL", mx, gate))
    return ok


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = []
    if which in ("all", "off"):
        res.append(compare("(options off) v36v prim_vii vs fixv4c prim_vii", os.path.join(ST, "prim_vii"), "fz", os.path.join(REF, "prim_vii"), "fz", os.path.join(ST, "prim_vii", "prim.log"), os.path.join(REF, "prim_vii", "prim.log")))
        for d, tag in (("fs", "fs"), ("fz", "fz")):
            res.append(compare("(options off) v36v off/%s vs fixv4c fx/%s" % (d, d), os.path.join(ST, "off", d), tag, os.path.join(REF, "fx", d), tag, os.path.join(ST, "off", d, tag + ".log"), os.path.join(REF, "fx", d, tag + ".log")))
    if which in ("all", "on"):
        res.append(report("(c1) prim fixture + --dg-stall 1", os.path.join(ST, "on_c1"), "fz", os.path.join(ST, "on_c1", "prim.log"), os.path.join(REF, "prim_vii"), "fz", 1e-10))
        res.append(report("(c2) anderson fs + --dg-stall 1", os.path.join(ST, "on_c2", "fs"), "fs", os.path.join(ST, "on_c2", "fs", "fs.log"), os.path.join(REF, "fx", "fs"), "fs", 1e-10))
        report("(d) fresh prim, --dg-stall 8 --dg-sub-counts 1,3,2 --dg-sub-start 1 --dg-sub-max 2 (another quadrature: no gate; reference fx/fs, two sub-steps)", os.path.join(ST, "on_d", "fz"), "fz", os.path.join(ST, "on_d", "fz", "fz.log"), os.path.join(REF, "fx", "fs"), "fs", None)
        report("(e) prim fixture, --dg-cap 2 --dg-sub-counts 1,2,3 --dg-sub-start 0 --dg-sub-max 2 (the counts' control flow to the STOP)", os.path.join(ST, "on_e"), "fz", os.path.join(ST, "on_e", "prim.log"), os.path.join(REF, "prim_vii"), "fz", None)
    if which in ("all", "stall"):
        res.append(report("(c3) prim fixture, one record, plain --dg-beta 0.25 --dg-stall 2 --dg-cap 80 (selftest_v36v_stall_box.sh)", os.path.join(ST, "on_c3"), "fz", os.path.join(ST, "on_c3", "prim.log"), os.path.join(REF, "prim_vii"), "fz", 1e-10))
        res.append(report("(c4) prim fixture, one record, anderson --dg-beta 0.25 --dg-stall 1 --dg-cap 80 (selftest_v36v_stall_box.sh)", os.path.join(ST, "on_c4"), "fz", os.path.join(ST, "on_c4", "prim.log"), os.path.join(REF, "prim_vii"), "fz", 1e-10))
    print("SUMMARY: %d of %d PASS -> %s" % (sum(res), len(res), "PASS" if all(res) else "FAIL"))
