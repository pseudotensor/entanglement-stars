#!/usr/bin/env python
"""engine_capture_before_arrival_v1.py (builder analyst; record-only; for RELABEL_table_v1.20's row on ITEM5 version 10's capture clause, whose
"0.105" the lead rounded from capture_read_v3_v9_t300_min_final.md's four-decimal 0.1045): per background and probe of a C-capture
version 9 JSON, engine_capture_read_v3's chain-end arrival (the first reading time at which the plain norm beyond R_max, summed as sigma
is, over pi b_c^2(omega, m), exceeds --arrival 0.01) and the smallest sigma/(pi b_c^2) before it, at full precision with its time and
in-transit share, written to --out-json and printed; nothing else. Args: --json --out-json [--arrival 0.01]."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True); ap.add_argument("--out-json", required=True)
    ap.add_argument("--arrival", type=float, default=0.01)
    a = ap.parse_args()
    d = json.load(open(a.json))
    out = {"script": "engine_capture_before_arrival_v1.py", "source": a.json, "arrival_threshold": a.arrival, "rows": []}
    for r in d["results"]:
        nrm = r["norm_ratio"]; steps = r["steps"]
        arr = None
        for i, s in enumerate(steps):
            if r["beyond_Rmax_summed"]["%.4f" % s["T"]] / nrm > a.arrival:
                arr = i
                break
        pre = steps[:arr] if arr is not None else steps
        mn = min(pre, key=lambda s: s["ratio"])
        row = {"background": r["background"], "probe": r["probe"], "e": r["massive"]["e"], "arrival_T_over_rs": steps[arr]["T_over_rs"] if arr is not None else None,
               "smallest_ratio_before": mn["ratio"], "its_T_over_rs": mn["T_over_rs"], "its_transit_share": mn["transit_share"]}
        out["rows"].append(row)
        print("%s | %s: arrival %s; smallest sigma/(pi b_c^2) before it %.10f at %.1f r_s, in-transit share %.10f" % (
            r["background"][:12], r["probe"], row["arrival_T_over_rs"], mn["ratio"], mn["T_over_rs"], mn["transit_share"]))
    json.dump(out, open(a.out_json, "x"), indent=1)


if __name__ == "__main__":
    main()
