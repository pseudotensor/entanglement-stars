#!/usr/bin/env python
"""s16A_read_2b_v1.py (busi; inbox #485 (ii)): emits PAGE_Bprime_s16A_2b_v<N>.md whole. Reads run 2b (sink onset 2.25 L, half-side 2.75 L,
dt 0.5 a, both windows, 256 columns, to 1e2 L) against run 1 (onset 2 L): at 6.93 L against x_full (256 columns), at 69.3 L against run 1's
supply and exit arms (128 + 128) and x_n32 (16 + 16), at 1e2 L against the 1e2 L clock when --clock is given. The onset dependence is a result
of the sink2L model (boundary dependence), reported separately from numerical error, never as an error bar (ReviewA_bpchk2_v1 sec. 1.3).
When --c2 (the 2c dt 0.25 a arm) is given, the onset differences are set against the measured time-step differences at the same clocks,
only to say whether the boundary dependence is resolved above numerical error. Every number below is printed by this script."""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s16A_end_lib_v1 as E  # noqa: E402

L = E.L
DON = 0.25  # onset change, in L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run1", required=True)
    ap.add_argument("--x", required=True)
    ap.add_argument("--b2", required=True, help="box copy of the 2b arm")
    ap.add_argument("--b2tag", default="s16A_2b_F1_onset225_dt0.5_T100_both")
    ap.add_argument("--clock", default="")
    ap.add_argument("--clocktag", default="s16A_1_F1_d0_dt0.5_T100_both")
    ap.add_argument("--c2", default="")
    ap.add_argument("--c2tag", default="s16A_2c_F1_dthalf_dt0.25_T100_both")
    ap.add_argument("--version", default="1")
    ap.add_argument("--prev", default="")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    R1s = E.load_run(a.run1, "s16A_1_F1_d0_dt0.5_T69.32_sup")
    R1e = E.load_run(a.run1, "s16A_1_F1_d0_dt0.5_T69.32_exit")
    Xf = E.load_run(a.x, "s16A_1_F1_d0_dt0.5_T6.932_both")
    Xn = E.load_run(a.x, "s16A_1_F1_d0_dt0.5_T69.32_both_n32")
    B = E.load_run(a.b2, a.b2tag)
    CK = E.load_run(a.clock, a.clocktag) if a.clock else None
    C2 = E.load_run(a.c2, a.c2tag) if a.c2 else None
    ns = B["ns"]; K = B["K"]
    logB = E.parse_log(os.path.join(a.b2, a.b2tag + ".log"))

    out = []; P = out.append
    P("# PAGE B' s = 16, stage A: run 2b (sink onset 2.25 L) read on all columns%s, version %s (busi lane)%s" % (
        " to 1e2 L against the 1e2 L clock" if CK else "", a.version, " [TEST FIXTURE, NOT A RESULT]" if a.test else ""))
    P("")
    P("Emitted whole by busi/code/pagetests/v3/s16A_read_2b_v1.py (helpers s16A_end_lib_v1.py) from md5-checked box copies; no number is retyped.%s Order: inbox #485 (ii). **The onset dependence below is boundary dependence of the sink2L model, a result, not an error bar** (ReviewA_bpchk2_v1 sec. 1.3; ReviewB_s16A_v1 sec. 3: 'measure boundary dependence'). Numerical error is reported separately (sec. 5 and PAGE_Bprime_s16A_2c)." % (
        (" A new file; %s stays as issued." % a.prev) if a.prev else ""))
    P("")
    P("**Labels.** Run 2b: `port=sink2L(onset 2.25 L, sin^2 ramp 0.5 L, eta_max 1/a, vacuum Markov bath)`, box half-side 2.75 L; run 1: `port=sink2L(onset 2 L, sin^2 ramp 0.5 L, eta_max 1/a, vacuum Markov bath)`, half-side 2.5 L. p is the weight a window direction delivers to the sink beyond its onset; held F1 fields (r_s/a 64 reference at T/r_s 30, fails the hydrodynamic gate inside), s = 16, sigma = L/8, d = 0, dt 0.5 a in both. Not far widths, not greybody factors, not emitted radiation, not a Page curve. Joint (supply + exit) entropies need the Nambu form and are not reported.")
    P("")
    # ------------------------------------------------------------------ 1
    P("## 1. Manifest (from the run files and the engine logs)")
    P("")
    P("| arm | role | dt (a) | onset (L) | half-side (L) | prep | columns | clocks (T/L) | steps per column (T/dt) | step median (s) | engine wall (h) | full-span s per column per a | column-a per s | peak GPU (GB) | md5 vs MD5SUMS | npz md5 |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = [("2b", "onset 2.25 L", B, logB)]
    if CK:
        rows.append(("1 (1e2 L clock)", "onset 2 L reference", CK, E.parse_log(os.path.join(a.clock, a.clocktag + ".log"))))
    if C2:
        rows.append(("2c", "dt 0.25 a (numerical bar only)", C2, E.parse_log(os.path.join(a.c2, a.c2tag + ".log"))))
    rows += [("1 x_full", "onset 2 L at 6.93 L", Xf, E.parse_log(os.path.join(a.x, Xf["tag"] + ".log"))),
             ("1 x_n32", "onset 2 L, 16 + 16 to 69.3 L", Xn, E.parse_log(os.path.join(a.x, Xn["tag"] + ".log"))),
             ("1 sup", "onset 2 L, 128 to 69.3 L", R1s, E.parse_log(os.path.join(a.run1, R1s["tag"] + ".log"))),
             ("1 exit", "onset 2 L, 128 to 69.3 L", R1e, E.parse_log(os.path.join(a.run1, R1e["tag"] + ".log")))]
    man = {}
    for nm, role, R, lg in rows:
        m = E.manifest_row(R, lg); man[nm] = m
        P("| %s | %s | %s | %s | %s | %s | %d | %s | %d | %.3f | %.2f | %.4f | %.1f | %.1f | %s | %s |" % (
            nm, role, m["dt"], m["onset"], m["box_L"], m["prep"], m["K"], ", ".join("%.4g" % x for x in m["clocks"]), m["steps"], m["step_median"], m["wall_h"],
            m["rate_full"], m["col_a_per_s"], m["peak_GB"], "OK" if m["md5_ok"] else "NOT LISTED", m["md5_npz"][:8]))
    P("")
    mb = man["2b"]
    P("Run 2b: %d columns (%d supply + %d exit) in batches of %d to T = %.4g L; engine wall %.0f s (%.2f h, the log's 'wrote' line); resumed from %s. Actual full-span cost %.4f s per column per a (step-median rate %.4f; the larger box costs (2.75/2.5)^3 = %.3f in sites). Peak GPU memory %.1f GB." % (
        K, ns, K - ns, mb["col_batch"], mb["clocks"][-1], mb["wall_h"] * 3600, mb["wall_h"], mb["resumed_from"] or "nothing (one session)", mb["rate_full"], mb["rate_step"], (2.75 / 2.5) ** 3, mb["peak_GB"]))
    if logB["clock_done"]:
        P("")
        P("Per-clock engine seconds (2b log): %s." % "; ".join("T/L %.4g at %.0f s" % (t, s) for t, s in logB["clock_done"]))
    P("")
    P("Basis identity (the onset-2.25 prep re-solves the box; the window inputs are compared): max |e_2b - e_x_full| = %.2g, max |qF_2b - qF_x_full| = %.2g. %s" % (
        np.max(np.abs(B["z"]["e"] - Xf["z"]["e"])), np.max(np.abs(B["z"]["qF"] - Xf["z"]["qF"])),
        "The same input columns, so element-by-element comparisons are valid." if np.max(np.abs(B["z"]["qF"] - Xf["z"]["qF"])) < 1e-10 and np.max(np.abs(B["z"]["e"] - Xf["z"]["e"])) < 1e-12
        else "NOT the same input columns: only the ordered spectra below are basis-independent; element rows are not comparable."))
    P("")
    # ------------------------------------------------------------------ 2
    P("## 2. Onset dependence at 6.93 L on all 256 columns (2b against x_full)")
    P("")
    Gb1, Gx1 = E.G_at(B, E.T1), E.G_at(Xf, E.T1)
    c1 = E.cmp_full(Gb1, Gx1, ns)

    def onset_rows(c, ns_, n_, P):
        for nm, dp, pa, pb in (("supply", c["dp_ss"], c["p_ss_a"], c["p_ss_b"]), ("exit", c["dp_ee"], c["p_ee_a"], c["p_ee_b"]), ("joint", c["dp_joint"], c["p_joint_a"], c["p_joint_b"])):
            P("| %s | %d | %.6f | %.6f | %.3g | %.3g | %.3g | %.3g | %.3g |" % (nm, len(dp), pb.sum(), pa.sum(), (pa.sum() - pb.sum()) / pb.sum(), np.max(np.abs(dp)), np.median(dp),
                                                                np.median(dp) / DON, np.max(np.abs(dp)) / DON))
    P("| spectrum | directions | delivered, onset 2 L | onset 2.25 L | relative change | max \\|dp\\| ordered | median dp (signed) | dp/d(onset) per L, median | max \\|dp\\|/d(onset) per L |")
    P("|---|---|---|---|---|---|---|---|---|")
    onset_rows(c1, ns, K, P)
    P("")
    P("Elements: max |dG| supply block %.3g, exit block %.3g, cross block %.3g, all 256 x 256 %.3g; cross-block norm onset 2 L %.4g, onset 2.25 L %.4g." % (
        c1["max_dG_ss"], c1["max_dG_ee"], c1["max_dG_se"], c1["max_dG"], c1["norm_se_b"], c1["norm_se_a"]))
    P("")
    # ------------------------------------------------------------------ 3
    P("## 3. Onset dependence at 69.3 L (2b against run 1's 128 + 128 single-window arms, and the 16 + 16 both-window piece)")
    P("")
    Gb2 = E.G_at(B, E.T2)
    s2 = E.cmp_block(Gb2[:ns, :ns], E.G_at(R1s, E.T2)); e2 = E.cmp_block(Gb2[ns:, ns:], E.G_at(R1e, E.T2))
    P("| block | directions | delivered, onset 2 L | onset 2.25 L | relative change | max \\|dp\\| ordered | median dp (signed) | dp/d(onset) per L, median | max \\|dp\\|/d(onset) per L | max \\|dG\\| element |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for nm, c in (("supply", s2), ("exit", e2)):
        P("| %s | %d | %.6f | %.6f | %.3g | %.3g | %.3g | %.3g | %.3g | %.3g |" % (nm, len(c["dp"]), c["pb"].sum(), c["pa"].sum(), (c["pa"].sum() - c["pb"].sum()) / c["pb"].sum(),
                                                                       np.max(np.abs(c["dp"])), np.median(c["dp"]), np.median(c["dp"]) / DON, np.max(np.abs(c["dp"])) / DON, c["max_dG"]))
    en = Xn["z"]["e"]; eb = B["z"]["e"]; nsn = Xn["ns"]
    idx = []; used = set()
    for x in en:
        cand = [i for i in np.argsort(np.abs(eb - x)) if i not in used]
        idx.append(cand[0]); used.add(cand[0])
    idx = np.array(idx)
    n2 = E.cmp_full(Gb2[np.ix_(idx, idx)], E.G_at(Xn, E.T2), nsn)
    P("| x_n32 subset (joint) | %d | %.6f | %.6f | %.3g | %.3g | %.3g | %.3g | %.3g | %.3g |" % (len(en), n2["p_joint_b"].sum(), n2["p_joint_a"].sum(), (n2["p_joint_a"].sum() - n2["p_joint_b"].sum()) / n2["p_joint_b"].sum(),
                                                                                np.max(np.abs(n2["dp_joint"])), np.median(n2["dp_joint"]), np.median(n2["dp_joint"]) / DON, np.max(np.abs(n2["dp_joint"])) / DON, n2["max_dG"]))
    P("")
    P("x_n32 columns located in 2b by energy (max |e| mismatch %.2g); its cross-block norm onset 2 L %.3g, onset 2.25 L %.3g. The 256-column cross block at 69.3 L with onset 2 L comes only from the 1e2 L clock." % (
        float(np.max(np.abs(eb[idx] - en))), n2["norm_se_b"], n2["norm_se_a"]))
    P("")
    # ------------------------------------------------------------------ 4
    P("## 4. At 1e2 L: 2b's joint 256-column Gram, cross block and spectrum")
    P("")
    P("| T/L | delivered (joint) | \\|\\|G_se\\|\\|_2 | top 3 singular values of G_se | joint min p | count p > 1/2 | 1 - p > 1e-2 | > 1e-3 | > 1e-4 | dark p < 1e-2 | max \\|p_joint - p_union\\| | supply max (1 - p) | exit max (1 - p) | max \\|p_sup - p_exit\\| (mirror) |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, T in enumerate(B["T"]):
        s = E.joint_summary(np.asarray(B["z"]["G"][i]), ns); r = 1 - s["pj"]
        P("| %.4g | %.6f | %.3g | %s | %.6f | %d | %d | %d | %d | %d | %.3g | %.3g | %.3g | %.2g |" % (T / L, s["trace"], s["norm_se"], ", ".join("%.3g" % x for x in s["sv_se"][:3]), s["pj"].min(),
                                                                                (s["pj"] > 0.5).sum(), (r > 1e-2).sum(), (r > 1e-3).sum(), (r > 1e-4).sum(), s["dark"], s["max_pj_minus_pu"],
                                                                                1 - s["ps"].min(), 1 - s["pe"].min(), s["mirror_dp"]))
    P("")
    cK = {}
    if CK:
        P("**Onset dependence against the 1e2 L clock (onset 2 L, same step), all 256 columns, at every common clock:**")
        P("")
        P("| T/L | spectrum | directions | delivered, onset 2 L | onset 2.25 L | relative change | max \\|dp\\| ordered | median dp (signed) | dp/d(onset) per L, median | max \\|dp\\|/d(onset) per L |")
        P("|---|---|---|---|---|---|---|---|---|---|")
        for T in (E.T1, E.T2, E.T3):
            Ga, Gb = E.G_at(B, T), E.G_at(CK, T)
            if Ga is None or Gb is None:
                continue
            c = E.cmp_full(Ga, Gb, ns); cK[T] = c
            for nm, dp, pa, pb in (("supply", c["dp_ss"], c["p_ss_a"], c["p_ss_b"]), ("exit", c["dp_ee"], c["p_ee_a"], c["p_ee_b"]), ("joint", c["dp_joint"], c["p_joint_a"], c["p_joint_b"])):
                P("| %.4g | %s | %d | %.6f | %.6f | %.3g | %.3g | %.3g | %.3g | %.3g |" % (T / L, nm, len(dp), pb.sum(), pa.sum(), (pa.sum() - pb.sum()) / pb.sum(), np.max(np.abs(dp)), np.median(dp),
                                                                          np.median(dp) / DON, np.max(np.abs(dp)) / DON))
        P("")
        P("Elements at 1e2 L: max |dG| all 256 x 256 %.3g; cross-block norm onset 2 L %.4g, onset 2.25 L %.4g." % (cK[E.T3]["max_dG"], cK[E.T3]["norm_se_b"], cK[E.T3]["norm_se_a"]) if E.T3 in cK else "")
    else:
        P("**What waits:** the 1e2 L clock (run 1, onset 2 L, dt 0.5 a, ARM=1, both windows; it runs on qi2_h100_1 after this arm, due about 18:40 UTC). Without it there is no onset-2 L Gram at 1e2 L, nor a 256-column onset-2 L cross block at 69.3 L, so the onset dependence at 1e2 L and of the 256-column cross block is not given here; it comes in version 2 (a new file). The 1e2 L rows above are 2b's own readouts.")
    P("")
    # ------------------------------------------------------------------ 5
    P("## 5. Is the boundary dependence resolved above numerical error? (separately; the dependence is not an error bar)")
    P("")
    if C2:
        P("Numerical bar: the measured ordered |dp| between dt 0.5 a and dt 0.25 a (2c) at the same clock and block, as read in PAGE_Bprime_s16A_2c. Ratio = onset |dp| / step |dp| per ordered direction.")
        P("")
        P("| T/L | spectrum | directions | max onset \\|dp\\| | max step \\|dp\\| | median ratio | directions with onset \\|dp\\| > 10 x step \\|dp\\| |")
        P("|---|---|---|---|---|---|---|")
        refs = {E.T1: ("x_full", Gx1)}
        for T in (E.T1, E.T2, E.T3):
            Gc = E.G_at(C2, T)
            if T == E.T1:
                cs = E.cmp_full(Gc, Gx1, ns); co = c1
                pairs = (("supply", co["dp_ss"], cs["dp_ss"]), ("exit", co["dp_ee"], cs["dp_ee"]), ("joint", co["dp_joint"], cs["dp_joint"]))
            elif T == E.T2:
                ss_ = E.cmp_block(Gc[:ns, :ns], E.G_at(R1s, T)); ee_ = E.cmp_block(Gc[ns:, ns:], E.G_at(R1e, T))
                pairs = (("supply", s2["dp"], ss_["dp"]), ("exit", e2["dp"], ee_["dp"]))
            else:
                if not CK or T not in cK:
                    continue
                cs = E.cmp_full(Gc, E.G_at(CK, T), ns); co = cK[T]
                pairs = (("supply", co["dp_ss"], cs["dp_ss"]), ("exit", co["dp_ee"], cs["dp_ee"]), ("joint", co["dp_joint"], cs["dp_joint"]))
            for nm, do, ds in pairs:
                rat = np.abs(do) / np.maximum(np.abs(ds), 1e-300)
                P("| %.4g | %s | %d | %.3g | %.3g | %.3g | %d |" % (T / L, nm, len(do), np.max(np.abs(do)), np.max(np.abs(ds)), np.median(rat), int((np.abs(do) > 10 * np.abs(ds)).sum())))
        P("")
    else:
        P("The 2c arm (dt 0.25 a) has not landed on the box at the time of this read, so the per-direction comparison waits for it (PAGE_Bprime_s16A_2c and version 2 of this file). For scale, the earlier column-subset step differences are in PAGE_Bprime_s16A_v1.1 sec. 3 (not recomputed or quoted here).")
        P("")
    # ------------------------------------------------------------------ 6
    P("## 6. What this read settles, and what waits")
    P("")
    P("- Settled: the boundary dependence of the delivered weight on all 256 columns at 6.93 L, on all 128 + 128 window columns at 69.3 L, and 2b's own joint spectrum and cross block at every clock to 1e2 L; the actual full-span throughput and peak memory of the larger-box arm.")
    if CK:
        P("- Settled with the clock: the boundary dependence at 1e2 L on all 256 columns, including the cross block.")
    else:
        P("- Waits for the 1e2 L clock (qi2_h100_1, about 18:40 UTC): the boundary dependence at 1e2 L and of the 256-column cross block at 69.3 L and 1e2 L. Version 2 will be a new file.")
    if not C2:
        P("- Waits for 2c (h100-f, about 12:20 UTC): the per-direction comparison of the boundary dependence with the measured step error (sec. 5).")
    P("")
    files = [(B["dir"], B["tag"], B["md5_npz"])] + ([(CK["dir"], CK["tag"], CK["md5_npz"])] if CK else []) + ([(C2["dir"], C2["tag"], C2["md5_npz"])] if C2 else []) + [
        (Xf["dir"], Xf["tag"], Xf["md5_npz"]), (Xn["dir"], Xn["tag"], Xn["md5_npz"]), (R1s["dir"], R1s["tag"], R1s["md5_npz"]), (R1e["dir"], R1e["tag"], R1e["md5_npz"])]
    P("Files (npz md5): %s; helpers s16A_end_lib_v1.py (md5 %s); this reader (md5 %s)." % (
        "; ".join("%s/%s.npz (%s)" % (d, t, m[:8]) for d, t, m in files), E.md5(os.path.join(HERE, "s16A_end_lib_v1.py"))[:8], E.md5(os.path.abspath(__file__))[:8]))
    open(a.out, "w").write("\n".join(out) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
