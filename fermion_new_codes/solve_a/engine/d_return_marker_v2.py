#!/usr/bin/env python
"""d_return_marker_v2.py (the lane's analyst builder; engine spec 41.2 (1), the lead's rule verbatim: "the first slice at which the incoming
positive-band energy beyond 24 r_s is positive and exceeds 1 percent of the outgoing at every window size on a slice where the outgoing exceeds
1e-3 (ten times the baseline's largest magnitude; lead's threshold)"): D_return_marker_v2 from engine_massive_channel_D_v5's jsons (a compose;
the v5 marker file a record). Every slice from the marker on is labelled "after the chain end's first return"; the paper's D numbers are those at
or before the slice preceding it. Per slice and window (16, 32, 64): the outgoing and incoming positive-band energy beyond 24 r_s (weighted over
the channels), the ratio, whether the outgoing exceeds 1e-3, whether the incoming is positive and above 1 percent of it, the test at every window.
Every number printed by the script. Usage: --jsons a.json,... --md --out-json."""
import argparse
import json
import os
import time

WINDOWS = ("16", "32", "64")
OUT_MIN = 1e-3
FRAC = 0.01
LABEL = "after the chain end's first return"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsons", required=True); ap.add_argument("--md", required=True); ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    for p in (a.md, a.out_json):
        if os.path.exists(p):
            raise SystemExit("refusing to overwrite %s" % p)
    srcs = [x for x in a.jsons.split(",") if x]
    rows = []
    for p in srcs:
        rows += json.load(open(p))["slices"]
    rows.sort(key=lambda r: (not r["baseline"], r["T_over_rs"]))
    seen = set(); uniq = []
    for r in rows:
        k = (r["baseline"], round(r["T_over_rs"], 6))
        if k not in seen:
            seen.add(k); uniq.append(r)
    rows = uniq
    tests = []
    for r in rows:
        W = r["weighted"]
        per = {}
        for L in WINDOWS:
            o, i = W[L]["E_plus_out"], W[L]["E_plus_in"]
            per[L] = {"out": o, "in": i, "ratio": (i / o) if o else None, "out_above": o > OUT_MIN, "in_test": (i > 0 and o > 0 and i > FRAC * o)}
        meets = all(per[L]["out_above"] and per[L]["in_test"] for L in WINDOWS)
        tests.append((r, per, meets))
    cont = [t for t in tests if not t[0]["baseline"]]
    marker = next((t[0]["T_over_rs"] for t in cont if t[2]), None)
    prev = [t[0]["T_over_rs"] for t in cont if marker is not None and t[0]["T_over_rs"] < marker]
    S = {"script": "d_return_marker_v2.py", "rule": "engine spec 41.2 (1)", "out_min": OUT_MIN, "fraction": FRAC, "sources": srcs, "marker_T_over_rs": marker,
         "last_slice_before": prev[-1] if prev else None, "rows": []}
    T = ["# D_return_marker_v2: the chain end's first return in D's massive channel beyond 24 r_s (engine spec 41.2 (1))", "",
         "Composed %s by code/engine/d_return_marker_v2.py from %s; every number printed by the script. The rule (the lead's, 41.2 (1)): the first slice at which the incoming "
         "positive-band energy beyond 24 r_s is positive and exceeds %g of the outgoing at every window size (16, 32, 64 sites) on a slice where the outgoing exceeds %g. "
         "Weighted over the channels (16 kappa); the production's slices 28 to 40 are the baseline (no marker is drawn there)." % (time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), ", ".join(srcs), FRAC, OUT_MIN), "",
         "- the marker: T/r_s %s; every slice from it on is labelled \"%s\"; the paper's D numbers are those at or before T/r_s %s." % (
             ("%.3f" % marker) if marker is not None else "not met on the composed slices", LABEL, ("%.3f" % prev[-1]) if prev else "-"), "",
         "| T/r_s | source | label | L 16: out; in; ratio | L 32: out; in; ratio | L 64: out; in; ratio | outgoing above 1e-3 at every window | the test at every window |",
         "|---|---|---|---|---|---|---|---|"]
    for r, per, meets in tests:
        lab = LABEL if (marker is not None and not r["baseline"] and r["T_over_rs"] >= marker - 1e-9) else "-"
        cells = ["%+.4e; %+.4e; %s" % (per[L]["out"], per[L]["in"], ("%+.4f" % per[L]["ratio"]) if per[L]["ratio"] is not None else "-") for L in WINDOWS]
        T.append("| %.3f | %s | %s | %s | %s | %s | %s | %s |" % (r["T_over_rs"], "production (baseline)" if r["baseline"] else "continuation", lab, cells[0], cells[1], cells[2],
                                                               "yes" if all(per[L]["out_above"] for L in WINDOWS) else "no", "met" if meets else "not met"))
        S["rows"].append({"T_over_rs": r["T_over_rs"], "baseline": r["baseline"], "label": lab, "windows": per, "meets": meets})
    T += ["", "No mutual information (D15). The v5 engine's own marker file (D_return_marker_v1_*) is a record."]
    open(a.md, "x").write("\n".join(T) + "\n")
    json.dump(S, open(a.out_json, "x"), indent=1, default=float)
    print("wrote %s: marker %s" % (a.md, marker))


if __name__ == "__main__":
    main()
