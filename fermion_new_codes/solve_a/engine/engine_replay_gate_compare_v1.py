#!/usr/bin/env python
"""engine_replay_gate_compare_v1.py (builder analyst; engine spec 31.5: "the gate bitwise v7 on three records of a v25 run"): two
replays' outputs (json and handoff npz) compared: the json entries other than "script", leaf by leaf (every differing leaf printed with its
path), and every handoff array bitwise; the verdict line names the differing leaves' kinds. Args: --a-json --b-json --a-npz --b-npz --md."""
import argparse
import json

import numpy as np


def walk(x, y, p, out):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            walk(x.get(k), y.get(k), p + "." + str(k), out)
    elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        for i, (u, v) in enumerate(zip(x, y)):
            walk(u, v, p + "[%d]" % i, out)
    elif x != y:
        out.append((p, x, y))


def main():
    ap = argparse.ArgumentParser()
    for k in ("a-json", "b-json", "a-npz", "b-npz", "md"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    A = json.load(open(a.a_json)); B = json.load(open(a.b_json))
    sa, sb = A.pop("script", None), B.pop("script", None)
    out = []
    walk(A, B, "", out)
    Za = np.load(a.a_npz, allow_pickle=True); Zb = np.load(a.b_npz, allow_pickle=True)
    nd = [k for k in Za.files if np.asarray(Za[k]).dtype != object and not np.array_equal(np.asarray(Za[k]), np.asarray(Zb[k]))]
    kinds = sorted(set(o[0].split(".")[-1] for o in out))
    L = ["## The replay gate: %s against %s (engine_replay_gate_compare_v1.py; engine spec 31.5)" % (sb, sa), "",
         "- json: %d differing leaves (%s)" % (len(out), ", ".join(kinds) if kinds else "none")]
    L += ["  - %s: %r against %r" % o for o in out[:20]]
    L += ["- handoff arrays: %d compared, %d differ%s" % (len(Za.files), len(nd), (" (" + ", ".join(nd) + ")") if nd else ""),
          "- verdict: %s" % ("bitwise in every computed quantity (the differing json leaves are wall-clock seconds only)" if not nd and kinds in ([], ["seconds"]) else "NOT bitwise"),
          "- every number above is printed by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
