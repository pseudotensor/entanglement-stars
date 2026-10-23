#!/usr/bin/env python
"""engine_single_object_B_merge_v1.py (builder analyst): merges B outputs of engine_single_object_B_v1.py that ran the same mode on the same
record file as separate jobs (B.6's frozen records T/r_s 15.0 and 28.0 ran in two CPU jobs under the coordinator's thread cap) into one file
in the layout engine_single_object_B_compose_v2.py and engine_single_object_F_horizon_v4.py read: the first file's metadata, the launches of
all files in order, and a "merged_from" list with each source's path, md5, launches and clocks. Never edits the sources.
Args: --inputs a.json,b.json --out-json."""
import argparse
import hashlib
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True); ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    paths = a.inputs.split(",")
    Js = [json.load(open(p)) for p in paths]
    for k_ in ("mode", "record", "fields", "nsites", "m", "cheb_tol", "R_cut_rs"):
        vals = {json.dumps(J.get(k_)) for J in Js}
        assert len(vals) == 1, "the sources differ in %s: %s" % (k_, vals)
    out = dict(Js[0])
    out["launches"] = [L for J in Js for L in J["launches"]]
    out["utc_start"] = min(J.get("utc_start") or "" for J in Js); out["utc_end"] = max(J.get("utc_end") or "" for J in Js)
    out["merged_from"] = [{"path": p, "md5": hashlib.md5(open(p, "rb").read()).hexdigest(), "launches": ["%.3f/%d" % (L["T_launch_over_rs"], L["kappa"]) for L in J["launches"]],
                           "utc_start": J.get("utc_start"), "utc_end": J.get("utc_end")} for p, J in zip(paths, Js)]
    keys = ["%.3f/%d" % (L["T_launch_over_rs"], L["kappa"]) for L in out["launches"]]
    assert len(keys) == len(set(keys)), "a launch appears twice: %s" % keys
    json.dump(out, open(a.out_json, "w"), default=float)
    print("merged %d launches (%s) from %s into %s" % (len(keys), ", ".join(keys), ", ".join(paths), a.out_json), flush=True)


if __name__ == "__main__":
    main()
