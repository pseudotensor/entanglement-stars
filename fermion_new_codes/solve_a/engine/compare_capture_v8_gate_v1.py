#!/usr/bin/env python
"""compare_capture_v8_gate_v1.py (builder analyst): the gate of engine_capture_v8 at the production's last minimum (slice 302): its md
against engine_capture_v7's there, every line compared except the title (names the script), the run's length and version 8's line of
channels excluded at launch, which may list only version 6's no-momenta exclusions (none by version 8's one-point guard); the JSONs' per
channel A readings compared too. Prints the verdict. Args: --v7md --v8md --v7json --v8json."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    for k in ("v7md", "v8md", "v7json", "v8json"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    A = open(a.v7md).read().split("\n"); B = open(a.v8md).read().split("\n")
    skip = lambda l: l.startswith("## C-capture (engine_capture_v") or l.startswith("- the run's length:") or l.startswith("- channels excluded at launch")  # noqa: E731
    exc = [l for l in B if l.startswith("- channels excluded at launch")]
    guard = sum(l.count("momentum point(s)") for l in exc)
    A2 = [l for l in A if not skip(l)]; B2 = [l for l in B if not skip(l)]
    diff = [(i, x, y) for i, (x, y) in enumerate(zip(A2, B2)) if x != y]
    print("md lines compared: %d and %d; differing: %d; version 8's guard exclusions listed: %d" % (len(A2), len(B2), len(diff) + abs(len(A2) - len(B2)), guard))
    for i, x, y in diff[:5]:
        print("  line %d:\n    v7 %s\n    v8 %s" % (i, x[:200], y[:200]))
    J7 = json.load(open(a.v7json)); J8 = json.load(open(a.v8json))
    nA, nd = 0, 0
    for r7, r8 in zip(J7["results"], J8["results"]):
        for c7, c8 in zip(r7["channels"], r8["channels"]):
            for s_ in ("plus", "minus"):
                a7, a8 = c7[s_].get("A", {}), c8[s_].get("A", {})
                nA += len(a7)
                nd += sum(1 for k in a7 if a8.get(k) != a7[k]) + abs(len(a7) - len(a8))
    print("json per-channel A readings compared: %d; differing: %d" % (nA, nd))
    ok = (not diff) and len(A2) == len(B2) and guard == 0 and nd == 0
    print("gate: %s" % ("PASS (version 8 equals version 7 at slice 302 to the bit on every compared line and reading)" if ok else "FAIL"))


if __name__ == "__main__":
    main()
