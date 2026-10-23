#!/usr/bin/env python3
"""obs_plateau_importcheck_v1.py (builder analyst): the PL-ext import and cut controls as a table, printed from obs_plateau_v3/v4 records.
Pairs of records with the same rows (profile A against profile B: exactc against version 2's pgs; smooth against nodes), per row: theta,
c = (1 - theta)/(kappa a)^2, the differences, kappa 2 r_s, R_cut/r_s, w. Args: pairs "A.json:B.json:label" ...; --md out."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+")
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    L = ["## PL-ext controls: the import path and the cut (obs_plateau_importcheck_v1.py)", "",
         "| pair | kappa a | R_cut/r_s | w | r | mu | theta A | theta B | B - A | c A | c B | kappa 2 r_s A | kappa 2 r_s B |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in a.pairs:
        fa, fb, lab = p.split(":")
        A = json.load(open(fa))["rows"]; B = json.load(open(fb))["rows"]
        for ra in A:
            ga = ra["geometry"]
            for rb in B:
                gb = rb["geometry"]
                same = (rb["kappa_a"] == ra["kappa_a"] and rb["r"] == ra["r"] and rb["mu"] == ra["mu"] and gb.get("R_lead_nominal") == ga.get("R_lead_nominal")
                        and abs(gb.get("R1", gb.get("R_cut", 0)) / gb["r_s"] - ga.get("R1", ga.get("R_cut", 0)) / ga["r_s"]) < 1e-9 and gb["w"] == ga["w"])
                if same and ra.get("theta") is not None and rb.get("theta") is not None:
                    ka = ra["kappa_a"]
                    L.append("| %s | %g | %g | %g | %g | %g | %.10f | %.10f | %.2e | %.4f | %.4f | %.9f | %.9f |" % (
                        lab, ka, ga.get("R1", ga.get("R_cut")) / ga["r_s"], ga["w"], ra["r"], ra["mu"], ra["theta"], rb["theta"], rb["theta"] - ra["theta"],
                        (1 - ra["theta"]) / ka ** 2, (1 - rb["theta"]) / ka ** 2, ga.get("kappa_2rs", float("nan")), gb.get("kappa_2rs", float("nan"))))
    L += ["", "- every number above is printed by this script from the named records."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
