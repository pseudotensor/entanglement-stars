#!/usr/bin/env python
"""Strand S diagnostic D-a1's report (SPEC sec. 9.1 outputs; builder analyst): from strandS_dos_v2.py's JSONs (one per size), the tables of
N, N_ns, S_th, E_th and their near-surface (R_* = 2 A_0) and interior (R_* = s) parts per size, marker and packet mass, with all modes and with
the edge pairs excluded; the exterior part (r >= s) = total - interior; the local powers p = d log X/d log(r_h/h) between neighbouring sizes at
the same marker (matched N_s and matched T r_h, since T = T_H = 1/(4 pi r_h)); the partial sums at j_max/4 and j_max/2 as fractions of the
total; the Gaussian correction.  Args: --jsons (comma list) --json --md."""
import argparse
import json
import math


def f(v, d=5):
    if v is None:
        return "-"
    v = float(v)
    return "%.*g" % (d, v) if (v == 0 or 1e-3 <= abs(v) < 1e6) else "%.3e" % v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsons", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    runs = sorted([json.load(open(p)) for p in a.jsons.split(",")], key=lambda d: d["meta"]["r_h"])
    L = ["# Strand S diagnostic D-a1: the sea's density of states at fixed boundary charge on the static chart (strandS_dos_report_v1.py)", "",
         "Background: L (the locked collapse; the boundary charge A_0 fixed, r_h = 4 A_0); T = T_H = 1/(4 pi r_h); h = 1; R_max = %s r_s0. Columns: "
         "the counts below T (N), the thermal entropy S_th and energy E_th of the sea (particles and holes), the near-surface parts (R_* = 2 A_0) "
         "and the interior parts (R_* = s), each with every mode and with the edge pairs excluded (\"no edge\")." % f(runs[0]["meta"]["Rmax_over_rs"]), ""]
    keys = ("N", "S", "S_noedge", "S_ns_2A0", "S_ns_2A0_noedge", "S_ns_s", "S_ns_s_noedge", "E", "E_ns_2A0", "edge_modes")
    out = {"rows": [], "powers": []}
    L += ["| r_h/h | marker | T/r_s0 | N_s | s/A_0 | j_max | m | N | S_th | S_th no edge | S_ns(2A_0) | no edge | S_int (s) | no edge | S_ext (r >= s) no edge | E_th | edge modes | Gaussian corr. |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    table = {}
    for d in runs:
        rh = d["meta"]["r_h"]
        for mk, M in d["markers"].items():
            for mkey, row in M["rows"].items():
                t = row["total"]
                ext = t["S_noedge"] - t["S_ns_s_noedge"]
                rec = {"r_h": rh, "marker": mk, "T_over_rs0": M["T_over_rs0"], "N_s": M["N_s"], "s_over_A0": M["s_over_A0"], "j_max": M["j_max"],
                       "mass": mkey, **{k: t.get(k) for k in keys}, "S_ext_noedge": ext, "gauss": t.get("gauss_correction"),
                       "partial_quarter_S": row["partial"].get("quarter", {}).get("S_noedge"), "partial_half_S": row["partial"].get("half", {}).get("S_noedge")}
                out["rows"].append(rec)
                table[(mk, mkey, rh)] = rec
                L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    f(rh, 4), mk, f(M["T_over_rs0"], 5), f(M["N_s"], 4), f(M["s_over_A0"], 5), f(M["j_max"]), mkey.replace("m=", ""), f(t["N"]),
                    f(t["S"]), f(t["S_noedge"]), f(t["S_ns_2A0"]), f(t["S_ns_2A0_noedge"]), f(t["S_ns_s"]), f(t["S_ns_s_noedge"]), f(ext), f(t["E"], 4),
                    f(t["edge_modes"]), f(t.get("gauss_correction"), 4)))
    L += ["", "## The local powers p = d log X/d log(r_h/h) between neighbouring sizes at the same marker", "",
          "| marker | m | sizes | S_th no edge | S_ns(2A_0) no edge | S_int no edge | S_ext no edge | N |", "|---|---|---|---|---|---|---|---|"]
    rhs = sorted({d["meta"]["r_h"] for d in runs})
    for (mk, mkey) in sorted({(k[0], k[1]) for k in table}):
        for r1, r2 in zip(rhs[:-1], rhs[1:]):
            if (mk, mkey, r1) in table and (mk, mkey, r2) in table:
                x1, x2 = table[(mk, mkey, r1)], table[(mk, mkey, r2)]
                pw = {}
                for k in ("S_noedge", "S_ns_2A0_noedge", "S_ns_s_noedge", "S_ext_noedge", "N"):
                    try:
                        pw[k] = math.log(x2[k] / x1[k]) / math.log(r2 / r1)
                    except (ValueError, ZeroDivisionError, TypeError):
                        pw[k] = None
                out["powers"].append({"marker": mk, "mass": mkey, "r_h": [r1, r2], **pw})
                L.append("| %s | %s | %s to %s | %s | %s | %s | %s | %s |" % (mk, mkey.replace("m=", ""), f(r1, 4), f(r2, 4), f(pw["S_noedge"], 3),
                                                                     f(pw["S_ns_2A0_noedge"], 3), f(pw["S_ns_s_noedge"], 3), f(pw["S_ext_noedge"], 3), f(pw["N"], 3)))
    L += ["", "- The partial sums over j (the share of S_th, no edge, at j_max/4 and j_max/2) are in the JSON's rows; every number above is emitted by the script."]
    json.dump(out, open(a.json, "w"), indent=1, default=float)
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
