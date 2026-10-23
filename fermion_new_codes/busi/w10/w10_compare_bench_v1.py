#!/usr/bin/env python
"""w10_compare_bench_v1.py (busi W10): the engine's bitwise gate: two collapse_<tag>.json files (and their fields npz) of a --benchmark
run on engine_collapse_v36d and on engine_collapse_v36e with the same arguments, compared value by value except the timing keys
(seconds, timing, wall_s, seconds_preparation, args.tag, args.out, script) and the new v36e keys (w10, end_traction_w10, the massless
ball's *_strict_v33, *_by_velocity, weight_outgoing_second_valley_spinor_minus, convention_w10). Exit 0 when equal.
Usage: qi2 python w10_compare_bench_v1.py A.json B.json"""
import json
import math
import os
import sys

import numpy as np

EXCLUDE = {"seconds", "timing", "wall_s", "seconds_preparation", "tag", "out", "script", "w10", "end_traction_w10",
           "weight_ingoing_spinor_k_negative_strict_v33", "weight_above_quarter_zone_strict_v33", "weight_ingoing_by_velocity",
           "weight_outgoing_by_velocity", "weight_outgoing_second_valley_spinor_minus", "convention_w10"}


def diff(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k in EXCLUDE:
                continue
            if k not in a:
                out.append((path + "/" + str(k), "only in B")); continue
            if k not in b:
                out.append((path + "/" + str(k), "only in A")); continue
            diff(a[k], b[k], path + "/" + str(k), out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, "length %d vs %d" % (len(a), len(b)))); return
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, path + "[%d]" % i, out)
    else:
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return
        if a != b:
            out.append((path, "%r vs %r" % (a, b)))


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    A = json.load(open(sys.argv[1])); B = json.load(open(sys.argv[2]))
    out = []
    diff(A, B, "", out)
    # the first line of the docstring label differs by version only through the new keys; the record_label and label are compared above
    nz = 0
    for p in (sys.argv[1], sys.argv[2]):
        pass
    fa = sys.argv[1].replace(".json", "_fields.npz"); fb = sys.argv[2].replace(".json", "_fields.npz")
    if os.path.exists(fa) and os.path.exists(fb):
        Za = np.load(fa, allow_pickle=True); Zb = np.load(fb, allow_pickle=True)
        for k in ("N", "v", "eps_tot", "J_tot", "eps_pairs", "J_pairs", "N_applied", "v_applied", "N_cinit", "v_cinit"):
            if k in Za.files and k in Zb.files:
                nz += 1
                if not np.array_equal(np.asarray(Za[k]), np.asarray(Zb[k])):
                    out.append(("fields.npz/" + k, "arrays differ (max |d| %.3e)" % float(np.max(np.abs(np.asarray(Za[k], float) - np.asarray(Zb[k], float))))))
    nrec = min(len(A.get("records", [])), len(B.get("records", [])))
    if out:
        print("W10 bench: %d differences over %d records and %d field arrays; the first 20:" % (len(out), nrec, nz))
        for p, d in out[:20]:
            print("  %s: %s" % (p, d))
        sys.exit(1)
    print("W10 bench: %d records and %d field arrays compared: equal except the excluded timing keys and the new v36e keys (bitwise gate PASS)" % (nrec, nz))


if __name__ == "__main__":
    main()
