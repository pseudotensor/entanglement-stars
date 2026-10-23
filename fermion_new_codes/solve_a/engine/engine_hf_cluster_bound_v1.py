#!/usr/bin/env python
"""engine_hf_cluster_bound_v1.py (builder analyst; for the lead's restatement of the cluster's agreement after red team R24f defect 7): over
the Wick rows of engine_hf_cluster_v3's records, with engine_hf_cluster_scfcheck_v1's corrections (the one row whose SCF never converged takes
the level-shifted value), the largest |(exact - X)/E_pot| for X = Hartree, HF1 and HFsc among the rows whose ACTUAL grav/kin is at most 0.03,
0.1 and 0.2 (the row that attains it printed), the counts, and the Wick y = E_x/E_pot per N_f (its range over the records' cases).
Args: --records --scfcheck --md."""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--scfcheck", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    corr = {}
    for r in json.load(open(a.scfcheck))["rows"]:
        corr[(r["record"], r["ns"], round(r["kappa"], 9), round(r["G"], 12), r["Nf"])] = r
    rows = []
    for f in a.records:
        D = json.load(open(f))
        for c in D["cases"]:
            if c["ordering"] != "wick":
                continue
            for r in c["rows"]:
                k = (os.path.basename(f), c["ns"], round(c["kappa"], 9), round(c["G"], 12), r["Nf"])
                sc = corr.get(k)
                hfsc = r["exact_minus_HFsc_over_Epot"]
                note = ""
                if sc is not None and not sc["v3_converged"]:
                    if sc.get("corrected_ratio") is not None:
                        hfsc = sc["corrected_ratio"]; note = "level-shifted (v3's SCF unconverged)"
                    else:
                        hfsc = None; note = "unconverged, no correction"
                rows.append({"record": os.path.basename(f), "ns": c["ns"], "kappa": c["kappa"], "G": c["G"], "Nf": r["Nf"], "gk": r["grav_over_kin"],
                             "y": r["y"], "H": r["exact_minus_H_over_Epot"], "HF1": r["exact_minus_HF1_over_Epot"], "HFsc": hfsc, "note": note,
                             "converged": (sc["v3_converged"] if sc is not None else None)})
    L = ["## The Wick cluster's agreement by actual grav/kin (engine_hf_cluster_bound_v1.py over %s; corrections from %s)" % (
        ", ".join(os.path.basename(f) for f in a.records), os.path.basename(a.scfcheck)), "",
         "| grav/kin at most | rows | max |(exact - H)/E_pot| | max |(exact - HF1)/E_pot| | max |(exact - HFsc)/E_pot| (row: n_s, kappa, G, N_f, grav/kin) |", "|---|---|---|---|---|"]
    for t in (0.03, 0.1, 0.2):
        sel = [q for q in rows if q["gk"] is not None and q["gk"] <= t]
        if not sel:
            L.append("| %g | 0 | - | - | - |" % t)
            continue
        mH = max(abs(q["H"]) for q in sel); m1 = max(abs(q["HF1"]) for q in sel)
        sc = [q for q in sel if q["HFsc"] is not None]
        qs = max(sc, key=lambda q: abs(q["HFsc"]))
        L.append("| %g | %d | %.3e | %.3e | %.3e (%d, %g, %.6g, %d, %.4f%s) |" % (t, len(sel), mH, m1, abs(qs["HFsc"]), qs["ns"], qs["kappa"], qs["G"], qs["Nf"], qs["gk"],
                                                                                  ("; " + qs["note"]) if qs["note"] else ""))
    L += ["", "### The Wick y = E_x/E_pot (first order, the Hartree orbitals) per N_f over all Wick cases", "", "| N_f | cases | y min | y max |", "|---|---|---|---|"]
    for nf in sorted({q["Nf"] for q in rows}):
        ys = [q["y"] for q in rows if q["Nf"] == nf and q["y"] is not None]
        L.append("| %d | %d | %.4f | %.4f |" % (nf, len(ys), min(ys), max(ys)))
    L += ["", "- rows: %d Wick rows; the unconverged row(s): %s" % (len(rows), "; ".join("%s n_s %d kappa %g G %.6g N_f %d -> %s" % (
        q["record"], q["ns"], q["kappa"], q["G"], q["Nf"], q["note"]) for q in rows if q["note"]) or "none"),
          "- every number above is printed by the script from the records."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
