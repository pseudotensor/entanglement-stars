#!/usr/bin/env python
"""figures_data_v1.py (busi lane, inbox #351): the data half of the owner's two clean figures, built to reviewer A's plan
(deliveries/lead_gaps/ReviewA_figures_v1.md). It copies values out of the stored files named in the plan's section 5 and writes
one .npz per figure; nothing is re-summed or recomputed. The plotting scripts (figure_item5_collapse_v3.py, figure_page_v5.py)
load only these .npz files.

Arithmetic done here, and only this (each is stated in the npz under 'notes'):
  - unit conversions: lattice clock T/64 -> T/r_s and lattice radius r/64 -> r/r_s (r_s = 64 a, the runs' nominal value);
    2GE/r_s = (E/E_0) * two_gE0 / 64 with two_gE0 = 59.0689 a (the readout's counts block), i.e. 0.92295 (E/E_0);
    the lowest channel's excess = 16 x the per-chain readout (the channel weight 16 kappa at kappa = 1);
  - N - |v| on the stored nodes (the plan's sign mask N - |v| < 0 is taken by the plotting script at the single level 0);
  - the +-5 r_s mean of the signed collar-flux ratio in Fig. 2(b) (all stored checkpoints with |t - t_i| <= 5);
  - Fig. 1(a)'s grey interval: the c19 record's stored trapping zeros at 15 (copied, /64);
  - the marker position of Fig. 1(a)'s circle: the node lapse N linearly interpolated at the readout's R_t (placement only).
Usage: python figures_data_v1.py OUT_DIR PAGE_FIG_NPZ READOUT_TSV
"""
import hashlib
import json
import os
import sys

import numpy as np

RS = 64.0
FIELDS_C19 = "<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz"
RECORD_C19 = "<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json"
FIELDS_P64 = "<internal>/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40_fields.npz"
LADDER_NPZ = "<internal>/ladders_readout_v1_3.npz"
LADDER_JSON = "<internal>/ladders_readout_v1_3.json"
HOLD_JSON = "<internal>/v6s_B2hold_k1_summary.json"
HOLD_MD = "<internal>/v6s_B2hold_k1_interior.md"
MEMBERS = (("s4", "P64_s4"), ("s2", "P64_s2"), ("s8", "P64_s8"))
CUTS = ("2", "16", "30")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    out_dir, page_npz, tsv = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    src = {p: md5(p) for p in (FIELDS_C19, RECORD_C19, FIELDS_P64, LADDER_NPZ, LADDER_JSON, HOLD_JSON, HOLD_MD, page_npz, tsv)}
    for p, m in src.items():
        print("source %s  %s" % (m, p))

    # ------------------------------------------------------------------ Fig. 1
    Z = np.load(FIELDS_C19, allow_pickle=True)
    Tf = Z["T"] / RS
    rn = np.array(Z["r_nodes"].tolist(), float)
    Nn = np.array(Z["N_nodes"].tolist(), float)
    vn = np.array(Z["v_nodes"].tolist(), float)
    assert np.all(rn == rn[0])
    r = rn[0] / RS
    keep = r <= 2.0 + 1e-12
    F = Nn - np.abs(vn)
    L = np.load(LADDER_NPZ)
    J = json.load(open(LADDER_JSON))
    T = L["P64_s4__T"]
    assert np.allclose(T, Tf, atol=1e-12)
    i0 = int(np.argmin(np.abs(Tf - 0.0)))
    i15 = int(np.argmin(np.abs(Tf - 15.0)))
    assert abs(Tf[i15] - 15.0) < 1e-12 and abs(Tf[i0]) < 1e-12
    two_gE0 = float(J["members"]["P64_s4"]["counts"]["two_gE0"])
    Rt15 = float(L["P64_s4__Rt_rs"][i15])
    circ_N = float(np.interp(Rt15, r, Nn[i15]))
    rec15 = json.load(open(RECORD_C19))["records"][i15]
    assert abs(rec15["T"] / RS - 15.0) < 1e-12
    zeros15 = np.array(rec15["trapping_zeros"], float) / RS
    ep = J["members"]["P64_s4"]["episodes"]
    form = float(J["members"]["P64_s4"]["formation"]["T"])
    closes, reopens = float(ep[0]["closes_at"]), float(ep[1]["first"])
    out1 = {
        "a_r_over_rs": r[keep], "a_N_T0": Nn[i0][keep], "a_absv_T0": np.abs(vn[i0])[keep],
        "a_N_T15": Nn[i15][keep], "a_absv_T15": np.abs(vn[i15])[keep], "a_F_T15": F[i15][keep],
        "a_Rt_T15_over_rs": np.float64(Rt15), "a_zeros_T15_over_rs": zeros15, "a_circle_N": np.float64(circ_N),
        "b_T_over_rs": Tf, "b_r_over_rs": r[keep], "b_N_minus_absv": F[:, keep],
        "b_Rt_over_rs": L["P64_s4__Rt_rs"], "b_2gE_over_rs": L["P64_s4__EE0"] * two_gE0 / RS,
        "b_Rc_over_rs": L["P64_s4__Rc"] / RS, "b_T_readout": T,
        "ev_formation": np.float64(form), "ev_closes": np.float64(closes), "ev_reopens": np.float64(reopens),
    }
    for tag, name in MEMBERS:
        cert = L[name + "__certified"].astype(bool)
        bad = np.where(~cert)[0]
        end = (int(bad[0]) - 1) if len(bad) else len(cert) - 1
        assert end == int(J["members"][name]["cert_end_index"]), (name, end)
        out1["c_T_%s" % tag] = L[name + "__T"][:end + 1]
        out1["c_minNv_%s" % tag] = L[name + "__minNv"][:end + 1]
        out1["c_md5_%s" % tag] = np.array(J["members"][name]["meta"]["md5"])
        out1["c_path_%s" % tag] = np.array(J["members"][name]["meta"]["path"])
    out1["notes"] = np.array(json.dumps({
        "sources": src,
        "a_b_fields": "the production's (c19) node fields N_nodes, v_nodes on r_nodes (spacing 4 a), not the sigma-averaged site fields: "
                      "the node sign of N - |v| agrees with the readout's min(N - |v|) sign at all 321 records, the site grid's does not at 9 "
                      "(16.75-17.25, 18.375-18.75), which would contradict the printed closes 17.375 / reopens 18.375",
        "conversions": "T/r_s = T/64; r/r_s = r/64; 2GE/r_s = (E/E_0) * %.6f / 64; R_c/r_s = R_c/64" % two_gE0,
        "fill_a": "Fig. 1(a) grey interval between the production record's stored trapping zeros at T/r_s 15 (trapping_zeros of the c19 run json, /64)",
        "circle": "Fig. 1(a) circle at r = R_t(15)/r_s from the readout, y = node N linearly interpolated there (placement only)",
        "members_c": "readout members P64_s4 (sigma/a 4), P64_s2 (2), P64_s8 (8); each to its certified prefix end",
    }))
    p1 = os.path.join(out_dir, "fig_item5_collapse_v3_data.npz")
    np.savez_compressed(p1, **out1)

    # checks printed (not used in the figure)
    Z2 = np.load(FIELDS_P64, allow_pickle=True)
    Nn2 = np.array(Z2["N_nodes"].tolist(), float); vn2 = np.array(Z2["v_nodes"].tolist(), float)
    print("check: c19 vs P64_s4 node fields max|dN| %.3e max|dv| %.3e sign-mask cells differing %d"
          % (np.abs(Nn - Nn2).max(), np.abs(vn - vn2).max(), int(((F < 0) != ((Nn2 - np.abs(vn2)) < 0)).sum())))
    ro = L["P64_s4__minNv"]
    print("check: node min(N-|v|) vs readout P64_s4: sign disagreements %d, max |diff| %.3e"
          % (int((np.sign(F.min(1)) != np.sign(ro)).sum()), np.abs(F.min(1) - ro).max()))
    print("fig1(a): stored zeros at 15 (r/r_s):", zeros15)
    print("fig1: R_t(15)/r_s %.4f, R_t/2GE at 15 %.4f, formation %.3f, closes %.3f, reopens %.3f, 2gE0 %.4f"
          % (Rt15, L["P64_s4__Rt_2gE"][i15], form, closes, reopens, two_gE0))
    for tag, name in MEMBERS:
        print("fig1(c): %s %s to T/r_s %.3f, min %.4f" % (tag, name, out1["c_T_%s" % tag][-1], np.nanmin(out1["c_minNv_%s" % tag])))

    # ------------------------------------------------------------------ Fig. 2
    P = np.load(page_npz)
    t = P["clocks"]
    out2 = {"a_t_over_rs": t}
    for c in CUTS:
        out2["a_S_cut%s" % c] = P["SUM_B2_cut" + c]
        out2["a_light_cut%s" % c] = P["light_band_cut" + c]
    rat = P["ratio12_TH"]
    mean = np.full_like(rat, np.nan)
    for i, ti in enumerate(t):
        w = (np.abs(t - ti) <= 5.0 + 1e-9) & np.isfinite(rat)
        if np.isfinite(rat[i]) and w.any():
            mean[i] = rat[w].mean()
    out2["b_t_over_rs"] = t
    out2["b_ratio_signed"] = rat
    out2["b_ratio_mean_pm5"] = mean
    H = json.load(open(HOLD_JSON))
    rows = H["rows"]
    out2["c_t_over_rs"] = np.array([x["t_over_rs"] for x in rows])
    for c in CUTS:
        out2["c_excess16_cut%s" % c] = 16.0 * np.array([x["per_channel"]["1"]["interior"]["cuts"][c]["excess"] for x in rows])
    out2["c_held"] = np.array([bool(x["held"]) for x in rows])
    out2["c_hold_from"] = np.float64(H["hold"]["T_hold"] / RS)
    out2["notes"] = np.array(json.dumps({
        "sources": src,
        "a": "SUM_B2_cut{2,16,30} of PAGE_fig_v4.1_data.npz (the 18-channel weighted sums, copied); light_band_cut* the stored arrival "
             "clocks of the formation's light from 2 and 1.5 x 2gE_f (the present panel's shading edges)",
        "b": "ratio12_TH of PAGE_fig_v4.1_data.npz: L_col (kappa 1, 2, collar 16 r_s) / L_H at T_H = 1/(8 pi G E) of the record's charge; "
             "b_ratio_mean_pm5 the mean of the signed ratio over all finite stored checkpoints within +-5 r_s",
        "c": "16 x per-chain interior excess (kappa 1) at cuts 2, 16, 30 r_s from v6s_B2hold_k1_summary.json rows, every 10 r_s to 500; held from 100",
    }))
    p2 = os.path.join(out_dir, "fig_page_v5_data.npz")
    np.savez_compressed(p2, **out2)

    # cross-check against the readout export (copy check only)
    mx = 0.0; n = 0
    for line in open(tsv).read().splitlines()[1:]:
        f = line.split("\t")
        tt, cut, S = float(f[0]), f[1], float(f[2])
        if cut in CUTS:
            j = int(np.argmin(np.abs(t - tt)))
            if abs(t[j] - tt) < 1e-9:
                mx = max(mx, abs(P["SUM_B2_cut" + cut][j] - S)); n += 1
    print("check: fig npz sums vs readout export tsv: %d pairs (40-100), max |diff| %.3e" % (n, mx))
    for c in CUTS:
        j = int(np.argmin(np.abs(t - 100.0)))
        print("fig2(a): cut %s at %.0f: %.3f nats; light %s" % (c, t[j], out2["a_S_cut%s" % c][j], out2["a_light_cut%s" % c]))
    ok = np.isfinite(rat) & (t >= 10)
    print("fig2(b): |ratio| range on 10-100: %.3e to %.3e; negatives %d; mean>0 points %d"
          % (np.nanmin(np.abs(rat[ok])), np.nanmax(np.abs(rat[ok])), int((rat[ok] < 0).sum()), int((mean[ok] > 0).sum())))
    print("fig2(c): at 500: %s" % ", ".join("%s %.3f" % (c, out2["c_excess16_cut%s" % c][-1]) for c in CUTS))
    for p in (p1, p2):
        print("wrote %s  %s  %d bytes" % (md5(p), p, os.path.getsize(p)))


if __name__ == "__main__":
    main()
