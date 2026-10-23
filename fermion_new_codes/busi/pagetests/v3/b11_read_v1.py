#!/usr/bin/env python
"""b11_read_v1.py (busi; reads run_bprime_b11_v1.py's jsons without re-running them). The run's own 'max over bins' p comparison mixes in
the collection lag of the far port (an amplitude needs about 48 L to reach 50 L, so early bins differ by construction). This reader compares
at the final clock T (and the retained weight, which needs no collection, at every bin):
  dilation:   cap50 against lead, p(T) per eigenvalue and the retained weight;   pass: max relative difference < 2 percent (p > 1e-2)
  lead:       lead2 against lead;                                              pass: < 2 percent
  near zone:  cap2 against cap50 (= the unitary exterior), p(T) and retained weights at 10, 50, 100 L;  'clean' only if < 2 percent.
Writes <out>/b11_read_v1.json and prints a table."""
import argparse
import json
import os

import numpy as np


def rel(a, b, floor=1e-2):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = (np.abs(a) > floor) | (np.abs(b) > floor)
    return float(np.max(np.abs(a[m] - b[m]) / np.maximum(np.abs(b[m]), 1e-300))) if m.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"script": "b11_read_v1.py", "rows": []}
    print("| family | kappa | inputs e/T0 | dilation p(T) rel (cap50 vs lead) | lead2 vs lead | near zone p(T) rel (cap2 vs cap50) | retained(50 L) cap2 / cap50 | retained(100 L) cap2 / cap50 |")
    print("|---|---|---|---|---|---|---|---|")
    for p in a.jsons:
        R = json.load(open(p))
        fam = R["args"]["family"]
        for k, c in R["kappa"].items():
            if "exteriors" not in c:
                continue
            ex = c["exteriors"]; Tk = sorted(ex["lead"]["p_spectrum_by_bin"], key=float)[-1]
            pl, pc, p2, pl2 = (ex[n]["p_spectrum_by_bin"][Tk] for n in ("lead", "cap50", "cap2", "lead2"))
            r50 = {n: np.asarray(ex[n]["retained_at_bins"]["50"]) for n in ("cap2", "cap50", "lead")}
            r100 = {n: np.asarray(ex[n]["retained_at_bins"][Tk]) for n in ("cap2", "cap50", "lead")}
            row = {"file": os.path.basename(p), "family": fam, "kappa": int(k), "e_over_T0": c["e_over_T0"], "T_over_L": float(Tk),
                   "dilation_p_rel": rel(pc, pl), "dilation_retained_rel": rel(r100["cap50"], r100["lead"], 1e-3), "lead_length_p_rel": rel(pl2, pl),
                   "near_zone_p_rel": rel(p2, pc), "retained50_ratio_cap2_over_cap50": [float(x / y) if y > 0 else None for x, y in zip(r50["cap2"], r50["cap50"])],
                   "retained100_ratio_cap2_over_cap50": [float(x / y) if y > 0 else None for x, y in zip(r100["cap2"], r100["cap50"])],
                   "p_T": {"cap2": p2, "cap50": pc, "lead": pl}}
            row["dilation_pass"] = row["dilation_p_rel"] < 0.02 and row["dilation_retained_rel"] < 0.02
            row["lead_length_pass"] = row["lead_length_p_rel"] < 0.02
            row["port2L_clean"] = row["near_zone_p_rel"] < 0.02
            res["rows"].append(row)
            f = lambda xs: ", ".join("%.2g" % x if x is not None else "-" for x in xs[:3])
            print("| %s | %s | %s | %.3g (%s) | %.3g | %.3g (%s) | %s | %s |" % (fam, k, ", ".join("%.1f" % x for x in c["e_over_T0"][:3]), row["dilation_p_rel"],
                  "PASS" if row["dilation_pass"] else "FAIL", row["lead_length_p_rel"], row["near_zone_p_rel"], "clean" if row["port2L_clean"] else "NOT clean",
                  f(row["retained50_ratio_cap2_over_cap50"]), f(row["retained100_ratio_cap2_over_cap50"])))
    res["all_dilation_pass"] = all(r["dilation_pass"] for r in res["rows"])
    res["all_lead_length_pass"] = all(r["lead_length_pass"] for r in res["rows"])
    res["any_port2L_clean"] = any(r["port2L_clean"] for r in res["rows"])
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "b11_read_v1.json"), "w"), indent=1)
    print("dilation all PASS: %s; lead length all PASS: %s; any 2 L port clean: %s" % (res["all_dilation_pass"], res["all_lead_length_pass"], res["any_port2L_clean"]))


if __name__ == "__main__":
    main()
