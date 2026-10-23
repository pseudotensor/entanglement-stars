#!/usr/bin/env python
"""bprime_cardhours_v2.py (busi; the B' v2 s = 16 queue in card-hours, emitted by script for PAGE_Bprime_design_v2.md sec. 6).
Rate: --bench <bench_summary.json> (the measured H100 seconds per column per unit time a at s = 16) or, until that exists, the v1 assumption
(1 ns per amplitude per apply x 12.75 applies per a x 2.13e6 amplitudes = 0.0271 s per column per a; labelled ASSUMED). Columns: the prep
jsons' window counts (supply + exit). Box factor for enlarged exteriors: (2 H' + 1)^3 / 81^3. Queue: reviewer A bpchk sec. 3, reordered, with A32
at sigma 0 (its applied-row baseline) and gated as design v2 sec. 7 says."""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--bench", default="")
    a = ap.parse_args()
    if a.bench and os.path.exists(a.bench):
        rate = json.load(open(a.bench))["seconds_per_column_per_a_s16"]; src = "MEASURED (%s)" % a.bench
    else:
        rate = 1e-9 * 12.75 * 4 * 81 ** 3; src = "ASSUMED (1 ns per amplitude per apply; replaced by the benchmark)"
    cols = {}
    for tag in ("F1_s16_d0", "F2_s16_d0", "A32_s16_d0_sig0"):
        M = json.load(open(os.path.join(a.prep_dir, tag + "_prep.json")))
        cols[tag] = M["inputs"]["n_supply"] + M["inputs"]["n_exit"]
    L = 16.0
    box = lambda half_L: (2 * round(half_L * L) + 1) ** 3 / 81 ** 3
    q = [("1", "F1 d0 to 1e2 L (rate calibration; continued in 3)", cols["F1_s16_d0"], 100, 1.0, 1.0),
         ("2a", "port control eta_max x 2, F1, 1e2 L", cols["F1_s16_d0"], 100, 1.0, 1.0),
         ("2b", "port control onset 2.25 L (half-side 2.75 L), F1, 1e2 L", cols["F1_s16_d0"], 100, box(2.75), 1.0),
         ("2c", "dt = 2 a control, F1, 1e2 L", cols["F1_s16_d0"], 100, 1.0, 15.0 / 12.75),
         ("3", "F1 d0 continued 1e2 -> 1e3 L", cols["F1_s16_d0"], 900, 1.0, 1.0),
         ("4a", "A32 sigma 0 to 1e2 L", cols["A32_s16_d0_sig0"], 100, 1.0, 1.0),
         ("4b", "A32 sigma 0 continued to 1e3 L", cols["A32_s16_d0_sig0"], 900, 1.0, 1.0),
         ("5", "F2 d0 to 1e3 L", cols["F2_s16_d0"], 1000, 1.0, 1.0),
         ("6", "clock scan d = 1/2, 1, 2 on F1, 1e2 L (3 runs)", 3 * cols["F1_s16_d0"], 100, 1.0, 1.0),
         ("7a", "averaging controls sigma/L 1/16 and 1/4 on F1, 1e2 L (2 runs)", 2 * cols["F1_s16_d0"], 100, 1.0, 1.0),
         ("7b", "splice reviewb (box 2.5 L), F1, 1e2 L", cols["F1_s16_d0"], 100, 1.0, 1.0),
         ("7c", "moved splice reviewb_out (onset 2.5 L, half-side 3 L), F1, 1e2 L", cols["F1_s16_d0"], 100, box(3.0), 1.0),
         ("7d", "interface move, F1, 1e2 L", cols["F1_s16_d0"], 100, 1.0, 1.0),
         ("8", "late port control eta_max x 2 on 32 directions to 1e3 L", 32, 1000, 1.0, 1.0)]
    tot = 0.0
    print("rate: %.4g s per column per a (%s); columns: %s" % (rate, src, cols))
    print("| step | run | columns | T/L | box factor | card-hours |")
    print("|---|---|---|---|---|---|")
    rows = []
    for st, lab, c, TL, bf, sf in q:
        h = c * TL * L * rate * bf * sf / 3600
        tot += h
        rows.append({"step": st, "run": lab, "columns": c, "T_over_L": TL, "box_factor": bf, "card_hours": h})
        print("| %s | %s | %d | %g | %.2f | %.3g |" % (st, lab, c, TL, bf, h))
    print("| total | | | | | %.3g |" % tot)
    json.dump({"rate": rate, "rate_source": src, "columns": cols, "rows": rows, "total": tot}, open(os.path.join(a.prep_dir, "cardhours_v2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
