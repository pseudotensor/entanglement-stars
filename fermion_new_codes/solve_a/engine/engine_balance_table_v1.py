#!/usr/bin/env python
"""engine_balance_table_v1.py (builder analyst; the owner's handoffs addendum of 22:05 UTC, item 1, "the integrated collar flux, work and
reference terms beside the Gauss-charge difference"; spec 19.3's balance table): per collapse run (engine_collapse_v10 and later records,
whose M1 split is stored per interval), accumulated from T = 0 at every record: the Gauss-charge difference E(T) - E(0) (the collar charge
W(r_collar)/(2g)); the inside energy H_in = Tr[(C - R2) H_in] at M1's cut (the cut site and whether it moved) and its change; the three parts
of that change, each summed over the intervals: the evolution under the frozen h_m (exact, the record's evolution_exact, i.e. the
integrated cut current through the cut; the trapezoid of the cut current beside it, and Simpson's where recorded), the field work (the change
of h_m at fixed state) and the reference reset (R2's change); the closure; and the difference (E(T) - E(0)) - (H_in(T) - H_in(0)), the part
of the Gauss charge outside the cut or not carried by the inside energy operator. Per record also the pair energy, R_t and min(N - |v|).
Every number is read from the records (no computation beyond the sums). Args: --runs dir/tag ... --md --every (records)."""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--every", type=int, default=8)
    a = ap.parse_args()
    L = ["## The collar balance beside the Gauss-charge difference (engine_balance_table_v1.py)", "",
         "Columns (accumulated from T = 0): dE = E(T) - E(0) (the collar charge); dH_in = H_in(T) - H_in(0) at M1's cut; its parts summed over the "
         "intervals: evolution (the frozen-h_m evolution, the integrated cut current), its trapezoid, field work, reference reset; the closure "
         "(evolution + work + reset - dH_in, summed); dE - dH_in. The cut site and whether it changed are printed.", ""]
    for run in a.runs:
        d, tag = os.path.split(run)
        J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
        rs = float(J["args"]["rs"])
        recs = [r for r in J["records"] if r.get("E") is not None]
        if not recs:
            continue
        E0 = recs[0]["E"]; H0 = recs[0].get("H_in")
        acc = {"evo": 0.0, "trap": 0.0, "simp": 0.0, "nsimp": 0, "work": 0.0, "reset": 0.0, "clos": 0.0, "dH": 0.0, "cut_moves": 0}
        L += ["### %s (%s; r_s %g, tau %g, ell %s, site fields %s; %d records to T/rs %.2f%s)" % (
            tag, J.get("script"), rs, J["args"]["tau"], J["args"]["ell"], J["args"].get("site_fields", "dense"), len(recs), recs[-1]["T"] / rs,
            "; the run stopped: " + J["records"][-1]["failed"] if J["records"][-1].get("failed") else ""), "",
              "| T/rs | E/E0 | dE | dH_in | evolution | trapezoid | field work | reference reset | closure | dE - dH_in | cut site (moves) | pairs | R_t | min(N - abs v) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(recs):
            b = r.get("M1_balance")
            if b:
                acc["evo"] += b["evolution_exact"]; acc["trap"] += b["evolution_trapezoid"]; acc["work"] += b["field_work"]; acc["reset"] += b["reference_reset"]
                acc["clos"] += b["closure"]; acc["dH"] += b["dHin"]; acc["cut_moves"] += int(bool(b.get("cut_changed")))
                if b.get("evolution_simpson") is not None:
                    acc["simp"] += b["evolution_simpson"] - b["evolution_trapezoid"]; acc["nsimp"] += 1
            last = (i == len(recs) - 1)
            if i % max(1, a.every) == 0 or last:
                dE = r["E"] - E0
                dH = (r["H_in"] - H0) if (r.get("H_in") is not None and H0 is not None) else acc["dH"]
                cut = (r.get("M1_parts") or {}).get("cut")
                L.append("| %.3f | %.7f | %.6e | %.6e | %.6e | %.6e | %.6e | %.6e | %.2e | %.6e | %s (%d) | %.4e | %s | %.4f |" % (
                    r["T"] / rs, r["E"] / E0, dE, dH, acc["evo"], acc["trap"], acc["work"], acc["reset"], acc["clos"], dE - dH, cut, acc["cut_moves"],
                    r.get("E_pairs_shells", float("nan")), ("%.3f" % r["R_t"]) if r.get("R_t") else "-", r.get("min_N_minus_absv", float("nan"))))
        L += ["", "- Simpson's correction to the trapezoid on the %d intervals where it was recorded: %.3e (summed)." % (acc["nsimp"], acc["simp"]) if acc["nsimp"] else
              "- no Simpson check recorded in this run.", ""]
    L += ["- every number above is read from the runs' json by the script (the sums are of the stored per-interval terms)."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:6]))


if __name__ == "__main__":
    main()
