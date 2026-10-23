#!/usr/bin/env python
"""Strand 2b sec. 15 report, items 1, 2 and 4 (builder analyst): tables from strand2b_sec15_v1.py (sec15_items12.json) and
strand2b_sec15_item4_v1.py (item4.json).  Numbers emitted, not retyped."""
import argparse
import json

import numpy as np


def f(v, sig=4):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    v = float(v)
    if v == 0:
        return "0"
    return ("%.*e" % (sig - 1, v)) if (abs(v) < 1e-3 or abs(v) >= 1e5) else ("%.*g" % (sig, v))


def table(head, rows):
    esc = lambda c: str(c).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(esc(c) for c in head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items12", required=True)
    ap.add_argument("--item4", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    R = json.load(open(a.items12))
    I4 = json.load(open(a.item4))
    out = []
    rows = []
    for o in R["item1"]:
        rows.append([o["case"], o["denominators"], o["d0"], " ".join(f(x, 10) for x in o["lowest_levels"]), f(o["selected_splitting"], 4),
                     f(o["max_abs_J"], 6), f(o["J_vs_FD"]["1e-05"]["max_abs_diff"], 3), f(o["J_vs_FD"]["1e-05"]["rel"], 3),
                     f(o["J_vs_FD"]["1e-06"]["max_abs_diff"], 3), f(o["J_vs_FD"]["1e-06"]["rel"], 3)])
    out += ["### Sec. 15 item 1: ring4, the pass-1 continuation solution with N_0 raised by 0.01, d_0 kept; the block's lowest levels and the "
            "splitting of the selected d_0; System.J against central differences of the complete F at h = 1e-5 and 1e-6, with the library's "
            "projector denominators (E_0 - w) and with the selected eigenvalues (w_i - w)", "",
            table(["case", "denominators", "d0", "lowest three levels", "selected splitting", "max |J|", "max |J - J_FD| (1e-5)", "relative",
                   "max |J - J_FD| (1e-6)", "relative"], rows), ""]
    I2 = R["item2"]
    out += ["### Sec. 15 item 2: kappa = 1^T K 1 and the free-block factor 2/kappa", "",
            table(["lattice", "kappa", "2/kappa"], [[k_, f(v, 8), f(I2["free_block_factor_2_over_kappa"][k_], 8)] for k_, v in I2["kappa"].items()]), ""]
    rows = []
    seen = set()
    for r in I2["rows"]:
        key = (r["lattice"], r["block"], r["m"], r["gM"], r["gB"], str(r["Lambda"]))
        if key in seen or r["k"] != 1.0:
            continue
        seen.add(key)
        rows.append([r["lattice"], r["block"], f(r["m"]), f(r["gM"]), f(r["gB"]), str(r["Lambda"]), f(r["E"], 6), f(r["dE"], 6), f(r["d2E"], 6),
                     f(r["c1_closed_form"], 6), f(r["c1_fit"], 6), f(r["ten_R10_minus_Rinf"], 6), f(r["hundred_R100_minus_Rinf"], 6),
                     f(r["c1_closed_form"] / r["c1_fit"] - 1, 3) if r["c1_fit"] else "-", f(r["free_check_2E2_over_kappa"], 6)])
    out += ["### Sec. 15 item 2: per pass-1 case (from the k = 1 record's E_block(1), dE/db, d^2E/db^2 at h = 1e-4): the closed form "
            "c_1^cf = [E'^2 + E E'' - 3 E E' + 2 E^2]/kappa beside the report's fitted c_1 and the direct 10 (R(10) - R_inf), 100 (R(100) - R_inf); "
            "the free-block value 2 E^2/kappa", "",
            table(["lattice", "block", "m", "g_M", "g_B", "Lambda", "E", "E'", "E''", "c_1 closed form", "c_1 fitted", "10 (R(10) - R_inf)",
                   "100 (R(100) - R_inf)", "closed/fitted - 1", "2 E^2/kappa"], rows), ""]
    rows = []
    for lab in ("R4", "C8"):
        for k in sorted({r["k"] for r in I2["rows"] if r["lattice"] == lab}):
            vals = [abs(r["R_minus_4pi_k_AmB"]) for r in I2["rows"] if r["lattice"] == lab and r["k"] == k]
            rr = [abs(r["R"]) for r in I2["rows"] if r["lattice"] == lab and r["k"] == k]
            rows.append([lab, f(k), len(vals), f(max(vals), 3), f(np.median(vals), 3), f(max(rr), 4)])
    out += ["### Sec. 15 item 2: the stationarity identity R = 4 pi k (A - B) over every pass-1 row, per lattice and k: the largest and the median "
            "|R - 4 pi k (A - B)|, beside the largest |R|", "",
            table(["lattice", "k", "rows", "max |R - 4 pi k (A - B)|", "median", "max |R|"], rows), ""]
    rows = [[r["E"], f(r["E_value"], 8), r["combination"], f(r["rel"], 3), "yes" if r["below_1e-2"] else "no"] for r in I4["rows"]]
    out += ["### Sec. 15 item 4: the cube's recorded Jacobian difference D (adjoint rows, psi columns; jacgap_v2.json) against E O_free with no "
            "fitted scale: max |D -/+ E O_free|/max |D| (max |D| = %s)" % f(I4["max_abs_D"], 6), "",
            table(["E", "value", "combination", "max |.|/max |D|", "below 1e-2"], rows), ""]
    open(a.md, "w").write("\n".join(out) + "\n")
    print("wrote", a.md)


if __name__ == "__main__":
    main()
