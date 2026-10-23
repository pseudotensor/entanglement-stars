#!/usr/bin/env python3
"""engine_vacflow_report_v1.py (builder analyst): J-VF's channel convergence from engine_vacflow_v2's record (partial or final): per ramp
t_r, the running shell sums E_raw and E_NO at the probe radii against the channel count, and the weighted per-channel contribution at the
probe radius R_read for the last channels (the tail's decay with kappa). Args: record.json --md out."""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    D = json.load(open(a.record))
    args = D["args"]; rs = args["rs"]
    probes = [r for r in (1.5 * rs, 2 * rs, 4 * rs, 8 * rs, 16 * rs)]
    L = ["## J-VF channel convergence (engine_vacflow_report_v1.py over %s; r_s %g, profile %s, m %g; channels done %s)" % (a.record, rs, args["profile"], args["m"], D.get("kappa_done")), ""]
    for tr, rows in D["running"].items():
        L += ["### t_r = %s r_s: the running sums at the probe radii R/r_s = %s" % (tr, ", ".join("%g" % (p / rs) for p in probes)), "",
              "| channels | E_raw at the probes | E_NO at the probes |", "|---|---|---|"]
        ks = [r["kappa"] for r in rows]
        for kq in sorted(set([k for k in (8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024) if k <= ks[-1]] + [ks[-1]])):
            r = rows[ks.index(kq)] if kq in ks else None
            if r:
                L.append("| %d | %s | %s |" % (kq, ", ".join("%.5e" % x for x in r["E_raw_at"]), ", ".join("%.5e" % x for x in r["E_NO_at"])))
        L.append("")
    pc = D.get("per_channel", [])
    if pc:
        L += ["### The weighted per-channel contribution at R_read (= 16 r_s) for the last 8 channels done", "",
              "| kappa | weight | " + " | ".join("raw t_r %s, NO t_r %s" % (t, t) for t in D["running"].keys()) + " |", "|---|---|" + "---|" * len(D["running"])]
        for p in pc[-8:]:
            cells = []
            for t in D["running"].keys():
                ap_ = p["at_probes"][t]
                cells.append("%.3e, %.3e" % (p["weight"] * ap_["raw_hold"][-1], p["weight"] * ap_["NO_hold"][-1]))
            L.append("| %d | %g | %s |" % (p["kappa"], p["weight"], " | ".join(cells)))
    L += ["", "- every number above is printed by this script from the record named in the title (the per-site density at R_read times the channel weight)."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
