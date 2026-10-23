#!/usr/bin/env python
"""Item 7's data script (SPEC_strand7_page.md sec. 8; builder analyst, from the spec, the lead's draft having stopped before it): reads the
primary run's records (strand7_sea_page_v1.py, s7p_<tag>.json; one per channel or one with all, merged by channel) and writes one .npz with, per output time T/r_s0: dS_rad (dS^(ad) at the
radiation cut 8A(T), summed over the two sectors and over the channels with the weight 2j + 1), S_diag_rad (the exterior modes' diagonal
entropy, the same sums), dS_thr (dS^(ad) at the throat A(T), the same sums), g_h and g_4h (the two-point function across the throat at
d = h and 4h relative to the initial vacuum's at the same cut: the 2j + 1 weighted sums of G_state over those of G_vac0), and the window
T_1, T_2 (from the record's window: N(r_sh) below 0.5 and below h T_H).  Only outputs present in every channel enter.
Version 2 (SPEC_strand7_page.md sec. 12.7): --records-c (optional, a comma list of the comparison C's JSONs) adds T_over_rs0_C and dS_rad_C
(the same sums over C's own outputs present in every C channel) to the same .npz; T_1 and T_2 remain the primary's (L's)."""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="comma list of s7p JSON files (their channels merged)")
    ap.add_argument("--records-c", default=None, help="comma list of the comparison C's s7p JSON files")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Rs = [json.load(open(p)) for p in a.records.split(",")]
    R = Rs[0]
    rs0 = float(R["meta"]["r_s0"])
    h = float(R["meta"]["h"])
    chans = {}
    for R_ in Rs:
        chans.update({float(k): v for k, v in R_["channels"].items()})
    Ts = sorted(set.intersection(*[set(round(q["T_over_rs0"], 9) for q in v) for v in chans.values()]))
    out = {k: [] for k in ("T_over_rs0", "dS_rad", "S_diag_rad", "dS_thr", "g_h", "g_4h", "G_h_state", "G_h_vac0", "G_4h_state", "G_4h_vac0")}
    for T in Ts:
        tot = {k: 0.0 for k in out if k != "T_over_rs0"}
        for j, v in chans.items():
            q = [x for x in v if abs(x["T_over_rs0"] - T) < 1e-9][0]
            w = 2 * j + 1
            for s in q["sectors"].values():
                c = s["cuts"]
                tot["dS_rad"] += w * c["rad"]["dSad"]
                tot["dS_thr"] += w * c["throat"]["dSad"]
                tot["S_diag_rad"] += w * s["spectrum"]["S_diag_rad"]
                d1, d4 = 1 * (h if h >= 1 else 1.0), 4 * (h if h >= 1 else 1.0)
                tot["G_h_state"] += w * c["throat"]["G_state_d%g" % d1]
                tot["G_h_vac0"] += w * c["throat"]["G_vac0_d%g" % d1]
                tot["G_4h_state"] += w * c["throat"]["G_state_d%g" % d4]
                tot["G_4h_vac0"] += w * c["throat"]["G_vac0_d%g" % d4]
        out["T_over_rs0"].append(T)
        for k in ("dS_rad", "S_diag_rad", "dS_thr", "G_h_state", "G_h_vac0", "G_4h_state", "G_4h_vac0"):
            out[k].append(tot[k])
        out["g_h"].append(tot["G_h_state"] / tot["G_h_vac0"] if tot["G_h_vac0"] > 0 else np.nan)
        out["g_4h"].append(tot["G_4h_state"] / tot["G_4h_vac0"] if tot["G_4h_vac0"] > 0 else np.nan)
    extra = {}
    if a.records_c:
        Rc = [json.load(open(p)) for p in a.records_c.split(",")]
        ch_c = {}
        for R_ in Rc:
            ch_c.update({float(k): v for k, v in R_["channels"].items()})
        Tc = sorted(set.intersection(*[set(round(q["T_over_rs0"], 9) for q in v) for v in ch_c.values()]))
        dsc = []
        for T in Tc:
            tot = 0.0
            for j, v in ch_c.items():
                q = [x for x in v if abs(x["T_over_rs0"] - T) < 1e-9][0]
                tot += (2 * j + 1) * sum(s["cuts"]["rad"]["dSad"] for s in q["sectors"].values())
            dsc.append(tot)
        extra = {"T_over_rs0_C": np.array(Tc, float), "dS_rad_C": np.array(dsc, float), "channels_C": np.array(sorted(ch_c))}
    win = R.get("window", {})
    T1 = win.get("T_1")
    T2 = win.get("T_2_1TH")
    np.savez(a.out, **{k: np.array(v, float) for k, v in out.items()}, T_1_over_rs0=np.nan if T1 is None else T1 / rs0,
             T_2_over_rs0=np.nan if T2 is None else T2 / rs0, r_s0=rs0, h=h, channels=np.array(sorted(chans)), **extra)
    print("wrote %s (%d outputs, channels %s)" % (a.out, len(Ts), sorted(chans)))


if __name__ == "__main__":
    main()
