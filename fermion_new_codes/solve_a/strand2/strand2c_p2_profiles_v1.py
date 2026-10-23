#!/usr/bin/env python
"""Strand 2c P2 profiles (builder analyst): pick the 2b solutions at (m, (g_M, g_B), Lambda) = (0.1, (0.2, 1), 1) and
k in {1, 10} (SPEC_strand2c_mismatch.md sec. 1, P2) and write them as the profiles file of strand2c_mismatch_v3.py.
Blocks (builder reading of the 2b spec's sec. 9 ruling): the dressed stellar-type states, ring4's (2,1) and (1,1) and the
cube's (2,0), with the one-particle (1,0) block of each lattice as reference.  Each entry: name, lattice, the 2b case
file, psi, N and the matter source on I.  JSON extraction only."""
import argparse
import json
import os

PICK = {"R4": ("1,0", "2,1", "1,1"), "C8": ("1,0", "2,0")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r4", default="", help="2b R4 run dir (case_R4_*.json)")
    ap.add_argument("--c8", default="", help="2b C8 run dir (case_C8_*.json)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    profs, missing = [], []
    for lat, d in (("R4", a.r4), ("C8", a.c8)):
        if not d:
            continue
        for blk in PICK[lat]:
            for k in (1, 10):
                case = "case_%s_b%s_m0.1_gM0.2_gB1_L1_k%g.json" % (lat, blk.replace(",", ""), k)
                p = os.path.join(d, case)
                if not os.path.exists(p):
                    missing.append(case)
                    continue
                r = json.load(open(p))
                if "psi" not in r:
                    missing.append(case + " (no solution)")
                    continue
                profs.append({"name": "%s_b%s_k%g" % (lat, blk.replace(",", ""), k), "lattice": lat, "case": case,
                              "psi": r["psi"], "N": r["N"], "source": r.get("source")})
    with open(a.out, "w") as fh:
        json.dump(profs, fh, indent=1)
    print("wrote %d profiles to %s; missing: %s" % (len(profs), a.out, ", ".join(missing) if missing else "none"))


if __name__ == "__main__":
    main()
