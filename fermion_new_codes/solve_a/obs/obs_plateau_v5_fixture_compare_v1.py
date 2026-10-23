#!/usr/bin/env python3
"""obs_plateau_v5_fixture_compare_v1.py (builder analyst): the import control's table: obs_plateau_v5 on the smooth-core fixture (dense route at
the given site orders; the spline comparator) against obs_plateau_v4 --profile smooth (the direct evaluation) at the same R_cut, and
obs_plateau_mix_v3 on the fixture against obs_plateau_mix_v2 --profile smooth. Args: --v5 (json, several) --v4 --mix3 --mix2 --md."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5", nargs="+", required=True)
    ap.add_argument("--v4", required=True)
    ap.add_argument("--mix3", required=True)
    ap.add_argument("--mix2", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    B = json.load(open(a.v4))
    ref = {(round(rd["geometry"]["R_cut"]), rd["r"]): rd for rd in B["rows"]}
    L = ["## The dense import's control (obs_plateau_v5 on the smooth-core fixture, R_c = 2 sites, r_s = 64, against the direct evaluation)", "",
         "| file | route | site order | R_cut | r | theta (import) | theta (direct, obs_plateau_v4 smooth) | difference | kappa (import)/kappa (direct) - 1 |", "|---|---|---|---|---|---|---|---|---|"]
    for f in a.v5:
        A = json.load(open(f))
        for rd in A["rows"]:
            g = rd["geometry"]; k = (round(g["Rcut_sites"]), rd["r"]); q = ref[k]
            L.append("| %s | %s | %s | %d | %g | %.12f | %.12f | %.2e | %.2e |" % (f.split("/")[-1], g["source"], A["args"]["site_order"] if "dense" in g["source"] else "-",
                     k[0], k[1], rd["theta"], q["theta"], rd["theta"] - q["theta"], g["kappa"] / q["geometry"]["kappa"] - 1))
        d = A["slices"][0]["diagnostics"]
        L.append("| %s | off-site floor printed (max |s7 - s5| at the half-integer points beyond R_cut = 12) | | | | N %.2e | v %.2e | | |" % (
            f.split("/")[-1], d["offsite_floor_N"], d["offsite_floor_v"]))
    M3 = json.load(open(a.mix3)); M2 = json.load(open(a.mix2))
    r2 = {(round(q["R_a_over_rs"] * 64), q["r"]): q for q in M2["rows"]}
    L += ["", "| R_a | r | branch-to-long-wave |t| (dense import, order %s) | (direct, obs_plateau_mix_v2 smooth) | cross-branch (import) | (direct) |" % M3["args"]["site_order"],
          "|---|---|---|---|---|---|"]
    for q in M3["rows"]:
        p = r2.get((round(q["R_a"]), q["r"]))
        L.append("| %g | %g | %.2e | %s | %.2e | %s |" % (q["R_a"], q["r"], q["branch_to_longwave_max"], ("%.2e" % p["branch_to_longwave_max"]) if p else "-",
                 q["cross_branch_max"], ("%.2e" % p["cross_branch_max"]) if p else "-"))
    L += ["", "- every number above is printed by the script; the fixture's core (2 sites) is under-resolved at unit spacing, which the solved slices",
          "  (smoothed at ell = 4 or 8) are not: the fixture bounds the dense route's off-site error from above."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
