#!/usr/bin/env python
"""item5_halo_share.py (busi readouts #235, item 5): the halo share (M(r_c) - M(R))/E per FIRSTLAW record (the sigma/a 2 history
c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d, FIRSTLAW_records v2.3 = the v4b pass on the v3b pass's records).
M(r) = r u^2/(2G), u = v/N, of the re-solved dense solution (the v3b record's N_R, v_R at the outer dense root R = R_dense); M(r_c) = the
re-solved collar charge E_collar = W(r_c)/(2G) at r_c = r_collar; E = the record's E (the collar charge of the run json). At a root u = -1, so
M(R) = R/(2G) and the share is 1 - R/(2G E_collar); both forms are printed. Records listed: every v3b record with an outer root; the v4b
pairs' first records are marked.
Usage: python item5_halo_share.py <v3b.json> <v4b.json> <out_prefix>
"""
import json
import sys

import numpy as np


def main():
    v3 = json.load(open(sys.argv[1])); v4 = json.load(open(sys.argv[2])); outp = sys.argv[3]
    G = float(v3.get("G", 0.25))
    pairsT = set()
    for key in ("pairs", "pairs_outer", "records_pairs"):
        if isinstance(v4.get(key), list):
            for p in v4[key]:
                if isinstance(p, dict) and "T_a" in p:
                    pairsT.add(round(p["T_a"], 6))
    print("v4b top keys:", list(v4.keys())[:30], "; pairs found:", len(pairsT))
    rows = []
    for r in v3["records"]:
        R = r.get("R_dense")
        if R is None:
            continue
        N, v, Ec = r.get("N_R"), r.get("v_R"), r.get("E_collar")
        E = r.get("E") if r.get("E") is not None else (Ec - r["E_collar_minus_E"] if r.get("E_collar_minus_E") is not None else None)
        u = v / N
        MR = R * u * u / (2 * G)
        rows.append({"T_over_rs": r["T_over_rs"], "R": R, "N_R": N, "v_R": v, "M_R": MR, "M_R_root_form": R / (2 * G), "E_collar": Ec, "E": E,
                     "r_collar": r.get("r_collar"), "halo": (Ec - MR) / E if E else None, "halo_root_form": 1 - R / (2 * G * Ec),
                     "R_over_2GE": r.get("R_over_2GE"), "in_v4b_pairs": round(r["T_over_rs"], 6) in pairsT})
    json.dump(rows, open(outp + ".json", "w"), indent=0)
    L = ["T/rs     R          M(R)        E_collar     E            halo=(M(r_c)-M(R))/E   1-R/(2G E_c)   r_c     v4b pair"]
    for x in rows:
        L.append("%7.3f  %9.4f  %10.5f  %11.6f  %11.6f  %.5f %s              %.5f        %s  %s" % (
            x["T_over_rs"], x["R"], x["M_R"], x["E_collar"], x["E"], x["halo"], "<5%" if x["halo"] < 0.05 else "   ", x["halo_root_form"],
            x["r_collar"], "a" if x["in_v4b_pairs"] else "-"))
    h = np.array([x["halo"] for x in rows]); T = np.array([x["T_over_rs"] for x in rows])
    below = T[h < 0.05]
    L.append("")
    L.append("SUMMARY: %d rooted records %.3f..%.3f; halo share min %.4f (T %.3f) max %.4f (T %.3f) median %.4f; below 5 percent: %d records%s; "
             "max |M(R) - R/(2G)| %.2e; max |halo - (1 - R/2GE_c)| %.2e" % (
                 len(rows), T[0], T[-1], h.min(), T[np.argmin(h)], h.max(), T[np.argmax(h)], np.median(h), len(below),
                 (" (" + ", ".join("%.3f" % t for t in below) + ")") if len(below) else "",
                 max(abs(x["M_R"] - x["M_R_root_form"]) for x in rows), max(abs(x["halo"] - x["halo_root_form"]) for x in rows)))
    open(outp + ".txt", "w").write("\n".join(L) + "\n")
    print(L[-1])


if __name__ == "__main__":
    main()
