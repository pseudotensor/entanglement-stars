#!/usr/bin/env python
"""d_v5_signed_report_v1.py (the lane's analyst builder; engine spec 40.3 (1), (2) with the lead's instruction pending turn 49: "for the D
marker, report the signed values and the ratio change from baseline without applying a trigger"): from engine_massive_channel_D_v5's jsons,
per slice (clock order) and window size (16, 32, 64 sites), weighted over the channels (16 kappa): the positive band's outgoing and incoming
energy beyond 24 r_s (signed, as computed), their ratio incoming/outgoing, the baseline ratio (the mean over the baseline slices, the
production's T/r_s 28 to 40, where nothing of the formation has reached 32 r_s) and each slice's ratio minus it; the negative band's two parts,
the windowed total and the block energy beside. No marker and no trigger is applied (40.3 (2)'s test waits for turn 49). Every number is
printed by the script. Usage: --jsons a.json,b.json --md --out-json."""
import argparse
import json
import os
import time

WINDOWS = ("16", "32", "64")


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
    base = {L: [r["weighted"][L]["ratio_in_out_plus"] for r in rows if r["baseline"] and r["weighted"][L]["ratio_in_out_plus"] is not None] for L in WINDOWS}
    bmean = {L: (sum(v) / len(v) if v else None) for L, v in base.items()}
    out = {"script": "d_v5_signed_report_v1.py", "sources": srcs, "baseline_ratio_mean": bmean, "baseline_ratios": base, "rows": []}
    T = ["# D version 5: the pairs' positive-band energy beyond 24 r_s by the local group-velocity sign, signed values and the ratio's change from the baseline (no trigger applied)", "",
         "Composed %s by code/engine/d_v5_signed_report_v1.py from %s; every number printed by the script. Weighted over the channels (16 kappa); energies in lattice units; "
         "the ratio is incoming/outgoing on the positive band, signed as computed (a negative incoming part is a deficit of the deviation's in-moving positive-band component); "
         "the baseline is the mean ratio over the production's slices T/r_s 28 to 40. No marker is drawn: 40.3 (2)'s test waits for the lead's turn 49." % (
             time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), ", ".join(srcs)), "",
         "- the baseline ratio (mean over %s): %s" % (", ".join("%.3f" % r["T_over_rs"] for r in rows if r["baseline"]),
                                                       "; ".join("L %s: %.4f (range %.4f to %.4f)" % (L, bmean[L], min(base[L]), max(base[L])) for L in WINDOWS if bmean[L] is not None)), "",
         "| T/r_s | source | L | + out E | + in E | ratio | ratio minus baseline | - out E | - in E | windowed total | block energy | E_out pairs beyond 24 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        W = r["weighted"]
        for L in WINDOWS:
            s = W[L]
            rt = s["ratio_in_out_plus"]
            d = (rt - bmean[L]) if (rt is not None and bmean[L] is not None) else None
            out["rows"].append({"T_over_rs": r["T_over_rs"], "baseline": r["baseline"], "L": int(L), "E_plus_out": s["E_plus_out"], "E_plus_in": s["E_plus_in"],
                                "ratio": rt, "ratio_minus_baseline": d, "E_minus_out": s["E_minus_out"], "E_minus_in": s["E_minus_in"],
                                "windowed_total": s["E_windowed_total"], "block_energy": W["block_energy_Y"], "E_out_pairs_24": W["E_out_pairs_Pm_24"]})
            T.append("| %.3f | %s | %s | %+.4e | %+.4e | %s | %s | %+.4e | %+.4e | %+.4e | %+.4e | %+.4e |" % (
                r["T_over_rs"], "production (baseline)" if r["baseline"] else "continuation", L, s["E_plus_out"], s["E_plus_in"],
                ("%+.4f" % rt) if rt is not None else "-", ("%+.4f" % d) if d is not None else "-", s["E_minus_out"], s["E_minus_in"], s["E_windowed_total"],
                W["block_energy_Y"], W["E_out_pairs_Pm_24"]))
    T += ["", "The windowed total against the block energy Tr[h_YY Delta_Y] measures the local band reading's error (windows and local dispersion), printed, not corrected. No mutual information (D15)."]
    open(a.md, "x").write("\n".join(T) + "\n")
    json.dump(out, open(a.out_json, "x"), indent=1, default=float)
    print("wrote %s" % a.md)


if __name__ == "__main__":
    main()
