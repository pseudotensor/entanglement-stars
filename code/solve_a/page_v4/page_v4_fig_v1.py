#!/usr/bin/env python
"""page_v4_fig_v1.py (the solve lane; PAGE_solved_history_v4): the figure script. Loads page_v4_data_v1.json and page_v4_data_v1.npz (the producer's
outputs) and draws PAGE_fig_v4.png; it writes the plotted arrays to PAGE_fig_v4_data.npz. Nothing is computed here beyond what is drawn.
Panels: (a) the K = 18 exterior-cut excess of the B2 history at the cuts 16, 24 and 48 r_s against the record clock, with the s8 history's beside to 40 r_s
and the marks (the formation record clock, the formation's light at each cut, arrival + T_d at the 24 r_s cut); (b) the kappa 1, 2 collar flux against
the two comparators, the kappa 3 guard's ratio beside, with the marks at the collar; (c) the channel terms 16 kappa x_kappa against kappa at two pairs with
the probes kappa 24, 32, 48, the propagation floor and the formed-energy terms.
Colours: black, red, blue and tab10 only. Usage: python page_v4_fig_v1.py DATA_JSON DATA_NPZ OUT_PNG OUT_NPZ"""
import json
import sys
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CUTS = ["2", "16", "24", "30", "48"]
COL = {"16": "tab:blue", "24": "black", "48": "tab:red"}


def main():
    p_json, p_npz, p_png, p_out = sys.argv[1:5]
    D = json.load(open(p_json)); Z = np.load(p_npz)
    cl = list(Z["clocks"]); cl8 = list(Z["clocks8"]); ci = {t: i for i, t in enumerate(cl)}
    SUM = Z["SUM_B2"]; SHARE = Z["SHARE_B2"]; SUM8 = Z["SUM_s8"]; SHARE8 = Z["SHARE_s8"]
    RB = {(r["t"], r["cut"]): r for r in D["B2"]["readout"]}; R8 = {(r["t"], r["cut"]): r for r in D["s8"]["readout"]}
    MK = D["marks"]; FL = D["flux"]
    t = np.array(cl); t8 = np.array(cl8)
    sel = t >= 12.0 - 1e-9; sel8 = t8 >= 12.0 - 1e-9
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(7.6, 14.0))
    saved = {"clocks": t, "clocks8": t8}
    # ---------------------------------------------------------------- (a) the excess
    for lab in ("16", "24", "48"):
        j = CUTS.index(lab); col = COL[lab]
        y = SUM[:, j]; sh = SHARE[:, j]
        a1.plot(t[sel], y[sel], "-", color=col, lw=1.1, label="B2, the cut %s r_s (K = 18)" % lab)
        a1.fill_between(t[sel], (y - sh)[sel], (y + sh)[sel], color=col, alpha=0.15, lw=0)
        formed = np.array([(RB.get((tt, lab)) or {}).get("verdict") in ("formed, tail open", "converged") for tt in cl])
        known = np.array([(tt, lab) in RB for tt in cl])
        a1.plot(t[sel & known & formed], y[sel & known & formed], "o", color=col, ms=3.2)
        a1.plot(t[sel & known & ~formed], y[sel & known & ~formed], "o", color=col, ms=3.6, mfc="none")
        y8 = SUM8[:, j]
        a1.plot(t8[sel8], y8[sel8], "--", color=col, lw=0.9, label="s8, the cut %s r_s (K = 18; a different object)" % lab)
        saved["SUM_B2_cut%s" % lab] = y; saved["SHARE_B2_cut%s" % lab] = sh; saved["SUM_s8_cut%s" % lab] = y8; saved["formed_B2_cut%s" % lab] = (known & formed).astype(int)
        e = MK["light_travel"][lab]
        a1.axvspan(e["2.0"]["arrival_record_clock_over_rs"], e["1.5"]["arrival_record_clock_over_rs"], color=col, alpha=0.12, lw=0)
        saved["light_band_cut%s" % lab] = np.array([e["2.0"]["arrival_record_clock_over_rs"], e["1.5"]["arrival_record_clock_over_rs"]])
    Tf = MK["formation_record_clock_over_rs"]
    a1.axvline(Tf, color="k", ls="--", lw=0.9, label="the formation's record clock T/r_s %g" % Tf)
    e24 = MK["light_travel"]["24"]
    a1.axvline(e24["ray_map_end_record_clock_over_rs"]["2.0"], color="k", ls="-.", lw=0.8, label="the formation's light at 24 r_s (from 2 x 2gE_f) + T_d (T_d/r_s %g; the leading term)" % MK["T_d_over_rs"])
    a1.set_yscale("symlog", linthresh=0.02)
    a1.set_ylabel("S = sum_k 16 k [S_X(C) - S_X(P_-(h))] (nats; symlog)", fontsize=8)
    a1.set_xlim(10, 101)
    a1.set_title("\n".join(textwrap.wrap("(a) The exterior-cut excess over the held Killing vacuum on B2 (page_history_v6o, 8 sub-steps, 8192 sites), summed over kappa 1 to 18 with the weights 16 kappa; the band is the exterior common mode's share K c; filled: formed, tail open; open: not formed or within the floor; shaded: the formation's light from 2 and 1.5 x 2gE_f at each cut (a fixed-charge flight comparison). The tail beyond kappa 18 is open at every pair.", 118)), fontsize=6.8)
    a1.legend(fontsize=6.0, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    a1.grid(alpha=0.25)
    # ---------------------------------------------------------------- (b) the flux
    for key, keyLH, col, lab in (("ratio12_kT", "LH12_kT", "tab:red", "L/L_H(kappa_T), kappa 1, 2 (|.|; open: L < 0)"), ("ratio12_TH", "LH12_TH", "tab:blue", "L/L_H(T_H = 1/(8 pi G E)), kappa 1, 2")):
        r = np.array([np.nan if v is None else v for v in FL[key]])
        ok = np.isfinite(r) & (t >= 14.0 - 1e-9) & (r != 0)
        a2.semilogy(t[ok], np.abs(r[ok]), "-", color=col, lw=1.0, label=lab)
        a2.semilogy(t[ok & (r > 0)], np.abs(r[ok & (r > 0)]), "o", color=col, ms=3)
        a2.semilogy(t[ok & (r < 0)], np.abs(r[ok & (r < 0)]), "o", color=col, ms=3.6, mfc="none")
        saved[key] = r
    for key, col, lab in (("ratio123_kT", "tab:red", "with kappa 3 (the truncation guard), L/L_H(kappa_T)"), ("ratio123_TH", "tab:blue", "with kappa 3, L/L_H(T_H)")):
        r = np.array([np.nan if v is None else v for v in FL[key]]); ok = np.isfinite(r) & (t >= 14.0 - 1e-9) & (r != 0)
        a2.semilogy(t[ok], np.abs(r[ok]), ":", color=col, lw=0.9, label=lab); saved[key] = r
    v3 = D["v3_flux"]; t3 = np.array(v3["t_over_rs"]); r3 = np.array([np.nan if v is None else v for v in v3["L_over_LH_kappaT"]]); ok3 = np.isfinite(r3) & (t3 >= 14.0 - 1e-9) & (r3 != 0)
    a2.semilogy(t3[ok3], np.abs(r3[ok3]), "s", color="tab:gray", ms=3, mfc="none", label="version 3's s4 run (v6b, piecewise-constant, r11 record, through 40): L/L_H(kappa_T)")
    saved["v3_t"] = t3; saved["v3_ratio_kT"] = r3
    e16 = MK["light_travel"]["16"]
    a2.axvspan(e16["2.0"]["arrival_record_clock_over_rs"], e16["1.5"]["arrival_record_clock_over_rs"], color="tab:red", alpha=0.12, lw=0, label="the formation's light at the collar 16 r_s (from 2 and 1.5 x 2gE_f)")
    a2.axvline(Tf, color="k", ls="--", lw=0.9)
    hf = MK["half_flow"]
    if hf["through_100_kappa12"]["record_clock_over_rs"] is not None:
        a2.axvline(hf["through_100_kappa12"]["record_clock_over_rs"], color="k", ls="-", lw=0.8, label="the half-collar-flow clock through 100 (kappa 1, 2)")
    if hf["through_40_kappa12"]["record_clock_over_rs"] is not None:
        a2.axvline(hf["through_40_kappa12"]["record_clock_over_rs"], color="tab:gray", ls="-", lw=0.8, label="the half-collar-flow clock through 40 (kappa 1, 2)")
    a2.axvline(e16["ray_map_end_record_clock_over_rs"]["2.0"], color="k", ls="-.", lw=0.8, label="the formation's light at the collar + T_d")
    a2.set_ylabel("|L| / L_H (kappa 1, 2)", fontsize=8)
    a2.set_xlim(10, 101)
    a2.set_title("\n".join(textwrap.wrap("(b) The collar flux at 16 r_s of the kappa 1 and 2 channels (weights 16 and 32) on B2 to 100 r_s: L is the collar energy per coordinate interval Delta T at the collar's recordwise stationary-chart clock, against the two-channel continuum greybody comparators (not an observed thermal flux); the kappa 3 guard's change dotted; the join's departure from the vacuum form grows to 7.2 percent in v at 100 r_s (T5).", 118)), fontsize=6.8)
    a2.legend(fontsize=6.0, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    a2.grid(alpha=0.25, which="both")
    # ---------------------------------------------------------------- (c) the channel terms
    WX = Z["WX_B2"]; WXP = Z["WX_probes"]; W = Z["W"]; ks = np.array(D["B2"]["kappas"]); kp = np.array([24, 32, 48])
    Ek = Z["E_kappa_w_B2"]
    for (tt, lab), col, mk in (((100.0, "24"), "black", "o"), ((80.0, "30"), "tab:red", "s")):
        i = ci[tt]; j = CUTS.index(lab)
        y = WX[:, i, j]; yp = WXP[1:, i, j]
        fm = np.array(D["B2"]["fmeas_w"])[:, i, j]
        a3.semilogy(ks, np.abs(y), "-", color=col, lw=0.9, label="|16 kappa x_kappa| at %g r_s, the cut %s r_s (filled: x > 0; open: x < 0)" % (tt, lab))
        a3.semilogy(ks[y > 0], np.abs(y[y > 0]), mk, color=col, ms=4); a3.semilogy(ks[y < 0], np.abs(y[y < 0]), mk, color=col, ms=4.5, mfc="none")
        a3.semilogy(kp[yp > 0], np.abs(yp[yp > 0]), mk, color=col, ms=5); a3.semilogy(kp[yp < 0], np.abs(yp[yp < 0]), mk, color=col, ms=5.5, mfc="none")
        a3.semilogy(ks, fm, ":", color=col, lw=0.8, label="16 kappa F_meas,kappa (the propagation floor) at %g/%s" % (tt, lab))
        a3.semilogy(ks, Ek[:, i], "--", color=col, lw=0.8, label="16 kappa E_kappa (the formed-energy term) at %g r_s" % tt)
        saved["terms_%g_%s" % (tt, lab)] = y; saved["probe_terms_%g_%s" % (tt, lab)] = yp; saved["fmeas_%g_%s" % (tt, lab)] = fm; saved["Ekappa_%g" % tt] = Ek[:, i]
    a3.axvline(18.5, color="k", ls=":", lw=0.7)
    a3.set_xlabel("kappa (the probes at 24, 32 and 48 are single-channel runs of the same line)", fontsize=8)
    a3.set_ylabel("weighted channel terms (nats; energy in lattice units)", fontsize=7.5)
    a3.set_title("\n".join(textwrap.wrap("(c) The channel terms against kappa at two readout pairs: from kappa 12 the excess terms are the exterior's common mode (flat to about 10 percent, changing sign with the clock) while the formed-energy terms fall; the probes carry the same term to kappa 48, so the sum's tail beyond kappa 18 is open and not geometric.", 118)), fontsize=6.8)
    a3.legend(fontsize=6.0, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    a3.grid(alpha=0.25, which="both")
    a1.set_xlabel("T/r_s (the record clock; the collar's clock t_inf is T + 0.49 to 0.50 r_s)", fontsize=7.5)
    a2.set_xlabel("T/r_s (the record clock)", fontsize=7.5)
    saved["kappas"] = ks; saved["probe_kappas"] = kp
    fig.suptitle("\n".join(textwrap.wrap("The light channel of the extension at first order (D16): the massless 4-component Dirac field of the same lattice operator, propagated at first order on the solved history of the heavy field's collapse (B2: r_s/a 64, m a 0.2, sigma/a 4, tau 8, continued to 100 r_s; the sigma/a 8 history beside as a different object), its sources not fed back; the interior beyond the hydrodynamic gate from T/r_s 11.75, the record's own limit. PAGE_solved_history_v4.", 125)), fontsize=7)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.905, bottom=0.075, hspace=0.62)
    fig.savefig(p_png, dpi=170)
    np.savez_compressed(p_out, **saved)
    print("wrote %s and %s" % (p_png, p_out))


if __name__ == "__main__":
    main()
