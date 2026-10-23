#!/usr/bin/env python
"""engine_replay_v9_gate_v1.py (builder analyst; engine spec 37.5 (1): "engine_replay_v9 (new; v8 with per-record hooks, every engine
mapping of v8)"): the gate of version 9 against version 8 on the records both replayed: every numeric field of version 8's json rows and
S1 rows, and every handoff array version 8 writes, compared bitwise (the hooks only add; version 8's quantities keep their code path).
Args: --v8-json --v8-handoff --v9-json --v9-handoff --md. Every number is printed by the script."""
import argparse
import json
import struct

import numpy as np


def leaves(x, path=""):
    if isinstance(x, dict):
        for k in sorted(x):
            yield from leaves(x[k], "%s.%s" % (path, k) if path else k)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, "%s[%d]" % (path, i))
    else:
        yield path, x


def bits_equal(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return struct.pack("<d", a) == struct.pack("<d", b) or (a != a and b != b)
    return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v8-json", required=True); ap.add_argument("--v8-handoff", required=True)
    ap.add_argument("--v9-json", required=True); ap.add_argument("--v9-handoff", required=True); ap.add_argument("--md", required=True)
    a = ap.parse_args()
    j8 = json.load(open(a.v8_json)); j9 = json.load(open(a.v9_json))
    L = ["## engine_replay_v9 against engine_replay_v8 (engine_replay_v9_gate_v1.py): version 8's fields on the records both replayed, bitwise", ""]
    ok = True
    for key in ("rows", "s1"):
        r8 = j8.get(key, []); r9 = j9.get(key, [])
        n = min(len(r8), len(r9))
        tot = same = 0; diff = []
        for i in range(n):
            l9 = dict(leaves(r9[i]))
            for p, v in leaves(r8[i]):
                if p == "seconds" or p.endswith(".seconds"):
                    continue
                tot += 1
                if p in l9 and bits_equal(v, l9[p]):
                    same += 1
                else:
                    diff.append((i, p, v, l9.get(p)))
        ok = ok and not diff and len(r8) == len(r9)
        L.append("- %s: version 8 %d, version 9 %d records; %d of %d fields bitwise equal (the timing fields skipped)%s." % (
            key, len(r8), len(r9), same, tot, "" if not diff else "; differ: " + "; ".join("record %d %s: %r against %r" % d for d in diff[:12])))
    h8 = np.load(a.v8_handoff, allow_pickle=True); h9 = np.load(a.v9_handoff, allow_pickle=True)
    for k in sorted(h8.files):
        if k not in h9.files:
            L.append("- handoff %s: absent in version 9" % k); ok = False; continue
        x8 = np.asarray(h8[k]); x9 = np.asarray(h9[k])
        eq = x8.shape == x9.shape and (x8.tobytes() == x9.tobytes() if x8.dtype == x9.dtype else False)
        ok = ok and eq
        L.append("- handoff %s: shape %s against %s, dtype %s against %s: %s" % (k, x8.shape, x9.shape, x8.dtype, x9.dtype, "bitwise equal" if eq else "DIFFERS"))
    L.append("- version 9's added handoff keys: %s." % ", ".join(sorted(set(h9.files) - set(h8.files))))
    L.append("- the gate: %s." % ("PASS" if ok else "FAIL"))
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
