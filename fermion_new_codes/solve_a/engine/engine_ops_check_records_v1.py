#!/usr/bin/env python
"""engine_ops_check_records_v1.py (builder analyst; red team R27c's code points on engine_ops_v2.continue_sea_v2, checked against the
stored records): per run (an engine_collapse_v19 or later json with the per-channel continuation records) and over its native records and
channels, how often each of R27c's cases occurred: ||dh||_G = 0 (tau_LZ infinite: the window then reaches window_max records back instead
of the predecessor), a window longer than one record (m_eff > 1), LZ sub-steps (nsub > 1), a partly occupied block at the occupation cut
(partly_occupied_blocks > 0: the next Weyl test then used the inter-block gap), rotated degenerate blocks (rotated_blocks > 0), an occupied
weight within 1e-9 of 1/2 (min_occupied_weight: an equal-weight tie at the cut), intermediate weights (n_intermediate > 0), the Weyl test
not verified, the window's consistency below 1 - 1e-12; and the records where any of the first four occurred. Args: --runs (comma list of
json paths) --md."""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    L = ["## R27c's cases in the stored continuation records (engine_ops_check_records_v1.py)", "",
         "Counts over (native record, channel) pairs; a run resumed from a v17 run has native records from its resume time on.", "",
         "| run | records, pairs | dH_G = 0 | m_eff > 1 | nsub > 1 | partly occupied block | rotated blocks | weight within 1e-9 of 1/2 | intermediate weights | not verified | window consistency < 1 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    notes = []
    for path in [p for p in a.runs.split(",") if p.strip()]:
        J = json.load(open(path))
        rs = float(J["args"]["rs"])
        c = dict(n=0, z=0, m=0, s=0, p=0, r=0, h=0, i=0, v=0, w=0)
        nrec = 0
        hits = {"dH_G = 0": [], "partly occupied": [], "half tie": [], "rotated": []}
        for r in J["records"]:
            cont = r.get("continuation")
            if r.get("E") is None or not isinstance(cont, dict) or not cont:
                continue
            first = next(iter(cont.values()))
            if not isinstance(first, dict) or "norm_dH_G" not in first:
                continue
            nrec += 1
            t = r["T"] / rs
            for kap, ci in cont.items():
                c["n"] += 1
                if ci.get("norm_dH_G") == 0.0:
                    c["z"] += 1; hits["dH_G = 0"].append((t, kap))
                if (ci.get("m_eff") or 1) > 1:
                    c["m"] += 1
                if (ci.get("nsub") or 1) > 1:
                    c["s"] += 1
                if (ci.get("partly_occupied_blocks") or 0) > 0:
                    c["p"] += 1; hits["partly occupied"].append((t, kap))
                if (ci.get("rotated_blocks") or 0) > 0:
                    c["r"] += 1; hits["rotated"].append((t, kap))
                w = ci.get("min_occupied_weight")
                if w is not None and abs(w - 0.5) < 1e-9:
                    c["h"] += 1; hits["half tie"].append((t, kap))
                if (ci.get("n_intermediate") or 0) > 0:
                    c["i"] += 1
                if ci.get("verified") is False:
                    c["v"] += 1
                wc = ci.get("window_consistency")
                if wc is not None and wc < 1.0 - 1e-12:
                    c["w"] += 1
        L.append("| %s | %d, %d | %d | %d | %d | %d | %d | %d | %d | %d | %d |" % (os.path.basename(path).replace("collapse_", "").replace(".json", ""), nrec, c["n"],
                                                                         c["z"], c["m"], c["s"], c["p"], c["r"], c["h"], c["i"], c["v"], c["w"]))
        for k, v in hits.items():
            if v:
                notes.append("- %s, %s: %d pairs, first at T/rs %.3f (channel %s), last at %.3f." % (os.path.basename(path), k, len(v), v[0][0], v[0][1], v[-1][0]))
    L += [""] + (notes or ["- none of the four cases (dH_G = 0, a partly occupied block, a half tie, a rotated block) occurred in these records."]) + [
        "- every number above is read from the runs' jsons by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
