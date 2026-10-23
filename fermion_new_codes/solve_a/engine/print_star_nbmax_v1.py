#!/usr/bin/env python
"""print_star_nbmax_v1.py (builder analyst; engine spec 27.4 and 27.7: "the OVPG N_B,max at (0.1, 1/64)" -- "BUILDER prints the exact value
from OVPG/ovpg_families.json, whose grid holds 0.1:0.015625"): the json's maximum carries z_c and M_max but no N_B; engine_star_v8 evaluates
the continuum solver at that z_c (OV.member(m, g, ..., z_max)["N_B"]) and stores it as summary.NB_max_continuum in each family's json. Prints,
per given engine_star json, the family, NB_max_continuum, the OVPG maximum's z_c and M_max, each member's N_B, its fraction of N_B,max, E,
E g^1.5 (m a)^2, min(N - |v|) and passes. Args: --ovpg --star (comma list of jsons) --md."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ovpg", required=True); ap.add_argument("--star", required=True); ap.add_argument("--md", required=True)
    a = ap.parse_args()
    fams = json.load(open(a.ovpg))["families"]
    L = ["# The continuum maximum's N_B per family (print_star_nbmax_v1.py; OVPG %s, engine_star_v8's summary.NB_max_continuum)" % a.ovpg, "",
         "| family (m a, g) | OVPG z_c at the maximum | OVPG M_max | N_B,max (continuum, at that z_c) | member N_B | N_B / N_B,max | E | E g^1.5 (m a)^2 | min(N - abs v) | passes | json |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in a.star.split(","):
        d = json.load(open(f))
        m, g = d["args"]["m"], d["args"]["g"]
        fam = [x for x in fams if abs(x["m"] - m) < 1e-12 and abs(x["g"] - g) < 1e-12]
        mx = fam[0]["maximum"] if fam else {}
        nbmax = d["summary"]["NB_max_continuum"]
        for q in d["members"]:
            L.append("| (%g, %g) | %.6f | %.4f | %.4f | %.3f | %.5f | %s | %s | %s | %s | %s |" % (
                m, g, mx.get("z_c", float("nan")), mx.get("M_max", float("nan")), nbmax, q["N_B"], q["N_B"] / nbmax,
                ("%.6f" % q["E"]) if q.get("E") is not None else "-", ("%.5f" % (q["E"] * g ** 1.5 * m * m)) if q.get("E") is not None else "-",
                ("%.5f" % q["min_N_minus_absv"]) if q.get("min_N_minus_absv") is not None else "-", q.get("passes"), f))
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
