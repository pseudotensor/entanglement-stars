#!/usr/bin/env python
"""cmp_v9c1c_fixture.py (busi lane builder, 20 Oct; REVIEWA_C64_toy_readout_v1 sec. 4): the stand-in fixture's compare of a v9c1 run (A) against a
v9c1c run (B), as v9c1b's was: engine_twofield_v8_selftest.compare_runs (every record number but the wall seconds, every npz row at the common
clocks, bitwise), then each difference classified. Allowed, and only these: a passage block's "prim" string (assign_primitive_v1d against
assign_primitive_v1f), an entry's "samples" (absent in v9c1, carried since v9c1b), the resume's run-path text ("engine_v34", "label"); in the json
args only prim_module (and anchor_passage_prim, the lz run's primitive name) besides the run paths. PASS when every difference is allowed.

Usage. python cmp_v9c1c_fixture.py A_DIR/TAG B_DIR/TAG
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_twofield_v8_selftest as S8  # noqa: E402

PATH_KEYS = {"out", "tag", "slice_dir", "resume_slice", "resume_run", "resume_from"}
ARG_OK = {"prim_module", "anchor_passage_prim"}


def allowed(d):
    p = str(d[0]); vals = d[2] if len(d) > 2 else None
    if p.endswith("/passage/prim"):
        return vals is None or ("assign_primitive_v1d" in str(vals[0]) and "assign_primitive_v1f" in str(vals[1]))
    if p.endswith("/samples") or "/samples/" in p:
        return True
    if p.endswith("/engine_v34") or p.endswith("/label"):
        return True
    return False


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    A, B = argv[0], argv[1]
    common, diffs = S8.compare_runs(A, B, None)
    ja = json.load(open(os.path.join(os.path.dirname(A), "collapse_%s.json" % os.path.basename(A))))["args"]
    jb = json.load(open(os.path.join(os.path.dirname(B), "collapse_%s.json" % os.path.basename(B))))["args"]
    adiff = sorted(k for k in set(ja) | set(jb) if k not in PATH_KEYS and ja.get(k, "<absent>") != jb.get(k, "<absent>"))
    bad = [d for d in diffs if not allowed(d)]
    bad_args = [k for k in adiff if k not in ARG_OK]
    kinds = {}
    for d in diffs:
        p = str(d[0]); key = "prim" if p.endswith("/passage/prim") else ("samples" if "samples" in p else ("path text" if (p.endswith("/engine_v34") or p.endswith("/label")) else "other"))
        kinds[key] = kinds.get(key, 0) + 1
    ok = len(common) > 0 and not bad and not bad_args
    print("%s v9c1 against v9c1c %s / %s over %d common records: %d differences (%s); args differ in %s (%s); %s" % (
        "PASS" if ok else "FAIL", A, B, len(common), len(diffs), ", ".join("%s %d" % kv for kv in sorted(kinds.items())) or "none",
        adiff or "nothing", {k: (ja.get(k), jb.get(k)) for k in adiff}, ("not allowed: %s" % bad[:6]) if bad or bad_args else "bitwise except the prim string (and the allowed keys)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
