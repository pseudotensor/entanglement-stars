#!/usr/bin/env python
"""bench_compare_v1.py (busi; the B' short H100 benchmark, part 3). Reads the three short runs of run_bench_v1.sh and the apply benchmark:
  * time-step convergence (reviewer B bpchk sec. 2): max |G_dt4 - G_dt2| and max |S_dt4 - S_dt2| over the common clocks (the acceptance target
    for classifying p ~ 1e-3 is 1e-4 absolute);
  * cross-batch coherence (reviewer B bpchk sec. 3): max |G_A - G_B| between the batch-128 and batch-64 runs (states checkpointed, assembled);
  * the measured rate: seconds per column per unit time a = median step / (batch columns x dt), and the card-hours re-emitted from it for the
    B' v2 queue (design v2 sec. 6; window counts from the s = 16 prep jsons, scaled 4 per doubling where unmeasured; the larger boxes scaled
    by their amplitude count, the bandwidth-linear assumption, labelled).
Writes <out>/bench_summary.json and prints the table."""
import argparse
import glob
import json
import math
import os

import numpy as np


def load(out, tag):
    J = json.load(open(os.path.join(out, tag + ".json")))
    Z = np.load(os.path.join(out, tag + ".npz"))
    return J, Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prep-jsons", nargs="*", default=[])
    a = ap.parse_args()
    J4, Z4 = load(a.out, "bench_F1_s16_dt4_T10")
    J2, Z2 = load(a.out, "bench_F1_s16_dt2_T10")
    Jb, Zb = load(a.out, "bench_F1_s16_dt4_T10_bB")
    s = {"device": J4["timing"]["device_name"], "columns": J4["K"]}
    s["dt_convergence_G_max_abs"] = float(np.max(np.abs(Z4["G"] - Z2["G"])))
    n = min(len(Z4["S_diag"]), len(Z2["S_diag"]) // 2)
    s["dt_convergence_note"] = "G at the common clocks; the target for p ~ 1e-3 classification is 1e-4 absolute"
    s["dt_convergence_pass_1e-4"] = s["dt_convergence_G_max_abs"] < 1e-4
    s["cross_batch_G_max_abs"] = float(np.max(np.abs(Z4["G"] - Zb["G"])))
    s["batches"] = [J4["timing"]["col_batch"], Jb["timing"]["col_batch"]]
    s["cross_batch_R_max_abs"] = float(np.max(np.abs(Z4["R"] - Zb["R"])))
    s["cross_batch_pass_1e-12"] = s["cross_batch_G_max_abs"] < 1e-12 and s["cross_batch_R_max_abs"] < 1e-12
    T = J4["timing"]
    rate = T["step_median_s"] / (min(T["col_batch"], J4["K"]) * T["dt"])            # one step advances one batch of columns
    s["seconds_per_column_per_a_s16"] = rate
    s["step_median_s"] = T["step_median_s"]; s["peak_mem_GB"] = T["peak_mem_GB"]
    ap_ = json.load(open(os.path.join(a.out, "bench_apply.json")))
    s["apply_rows"] = ap_["rows"]
    counts = {}
    for p in a.prep_jsons:
        M = json.load(open(p))
        counts[os.path.basename(p)] = M["inputs"]["n_supply"] + M["inputs"]["n_exit"]
    s["window_columns_s16"] = counts
    amps = {16: 4 * 81 ** 3, 32: 4 * 161 ** 3, 64: 4 * 321 ** 3}
    def hours(s_, cols, T_over_L):
        r_ = rate * amps[s_] / amps[16]
        return cols * T_over_L * s_ * r_ / 3600
    K16 = J4["K"]
    rows = []
    for lab, s_, cols, TL in (("F1 d0 1e2 L (rate calibration)", 16, K16, 100), ("F1 d0 1e3 L", 16, K16, 1000),
                              ("one control at 1e2 L", 16, K16, 100), ("s32 full window 1e2 L", 32, 4 * K16, 100),
                              ("s32 full window 1e3 L", 32, 4 * K16, 1000), ("s64 full window 1e2 L", 64, 16 * K16, 100)):
        rows.append({"run": lab, "s": s_, "columns": cols, "T_over_L": TL, "card_hours": hours(s_, cols, TL)})
    s["card_hours"] = rows
    json.dump(s, open(os.path.join(a.out, "bench_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in s.items() if k not in ("apply_rows", "card_hours")}, indent=1))
    print("| run | s | columns | T/L | card-hours (measured rate) |")
    print("|---|---|---|---|---|")
    for r in rows:
        print("| %s | %d | %d | %g | %.3g |" % (r["run"], r["s"], r["columns"], r["T_over_L"], r["card_hours"]))


if __name__ == "__main__":
    main()
