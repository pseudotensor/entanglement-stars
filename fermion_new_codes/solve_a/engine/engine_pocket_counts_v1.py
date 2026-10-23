#!/usr/bin/env python
"""engine_pocket_counts_v1.py (builder analyst; engine spec 21.3, BUILDER: "print n_pocket and the inversion count per record from their
json"): per stored engine_collapse_v17 run, per record the pockets summed over the channels (occupied levels above zero plus empty levels
below it, the continuation's own count), the inversions (occupied levels above the lowest empty one: zero whenever the pockets are zero,
since the occupied set is then the negative-energy set; v17 records no separate count), the continuation's subdivision (nsub, the halving
rule) and the trapping radius; the summary per run: the records, the records with nsub above 1 and their T/rs range, the records with a
nonzero pocket count. Args: --runs dir/tag ... --md."""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    L = ["## The r_s 32 runs' pocket counts (engine_pocket_counts_v1.py)", "",
         "| run | records | T/rs range | records with nsub > 1 (halving) | their T/rs range | records with pockets > 0 | trapping surface (T/rs range with R_t) |",
         "|---|---|---|---|---|---|---|"]
    detail = []
    for run in a.runs:
        d, tag = os.path.split(run)
        J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
        recs = [r for r in J["records"] if r.get("E") is not None]
        rs = float(J["args"]["rs"])
        hal = [r for r in recs if (r.get("weyl_summary") or {}).get("nsub_max", 1) > 1]
        pk = [r for r in recs if sum((r.get("n_pocket") or {}).values()) > 0]
        tr = [r["T"] / rs for r in recs if r.get("R_t")]
        L.append("| %s | %d | %.3f to %.3f | %d | %s | %d | %s |" % (
            tag, len(recs), recs[0]["T"] / rs, recs[-1]["T"] / rs, len(hal), ("%.3f to %.3f" % (hal[0]["T"] / rs, hal[-1]["T"] / rs)) if hal else "-", len(pk),
            ("%.3f to %.3f (%d records)" % (min(tr), max(tr), len(tr))) if tr else "none"))
        detail += ["", "### %s: the records with nsub > 1" % tag, "", "| T/rs | nsub max | pockets | R_t | min(N - abs v) |", "|---|---|---|---|---|"]
        for r in hal:
            detail.append("| %.3f | %d | %d | %s | %.4f |" % (r["T"] / rs, (r.get("weyl_summary") or {}).get("nsub_max", 1), sum((r.get("n_pocket") or {}).values()),
                                                        ("%.2f" % r["R_t"]) if r.get("R_t") else "-", r.get("min_N_minus_absv", float("nan"))))
    L += detail
    L += ["", "- the pockets are the continuation's own count per record (summed over the channels); a record with zero pockets has its occupied set equal "
          "to the negative-energy set, so its inversion count is zero too.", "- every number above is read from the runs' json by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:4 + len(a.runs)]))


if __name__ == "__main__":
    main()
