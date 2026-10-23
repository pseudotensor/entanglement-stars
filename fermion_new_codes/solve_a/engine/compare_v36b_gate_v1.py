#!/usr/bin/env python
"""compare_v36b_gate_v1.py (solve builder 5): the gate of engine_collapse_v36b against engine_collapse_v36 (G1's form, compare_v33_gate_v1):
every json number and every fields-npz array bitwise, the wall-clock keys and script/args excluded as there, and in addition the two
diagnostic entries that differ by construction and carry no computed quantity: "kernel_cache" (the parent's cache counters and its compute
seconds; with --pair-workers the workers' solves are not counted there) and "v36b" (the option's own entry). Then the per-record walls of
both runs (the record's "seconds"), the exact J.v's products and seconds, and the v36b entry. Usage: compare_v36b_gate_v1.py dirA/tagA dirB/tagB."""
import json
import os
import sys

import compare_v33_gate_v1 as G

G.SKIP |= {"kernel_cache", "v36b"}


def walls(path):
    d, t = os.path.split(path)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % t)))
    out = []
    for r in J.get("records", []):
        jv = r.get("jvp_v35") or {}
        out.append((r.get("T_over_rs"), r.get("seconds"), jv.get("products"), jv.get("seconds"), r.get("v36b")))
    return J.get("head", {}).get("seconds"), out


if __name__ == "__main__":
    G.main()
    for lab, p in (("A", sys.argv[1]), ("B", sys.argv[2])):
        hs, rows = walls(p)
        print("run %s %s: preparation %s s" % (lab, p, "%.1f" % hs if isinstance(hs, (int, float)) else hs))
        for T, s, npr, sj, vb in rows:
            print("  T/r_s %.4f: record %.1f s; J.v products %s (cumulative) in %s s%s" % (
                T, s if s is not None else float("nan"), npr, "%.1f" % sj if sj is not None else "-",
                ("; v36b %s" % json.dumps(vb)) if vb else ""))
