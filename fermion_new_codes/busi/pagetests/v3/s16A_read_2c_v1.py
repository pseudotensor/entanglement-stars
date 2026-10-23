#!/usr/bin/env python
"""s16A_read_2c_v1.py (busi; inbox #485 (i)): emits PAGE_Bprime_s16A_2c_v<N>.md whole. Reads run 2c (dt 0.25 a, both windows, 256 columns,
to 1e2 L) against run 1 (dt 0.5 a): at 6.93 L against the both-window piece x_full (all 256 columns, s16A_h100x8_v2/g2), at 69.3 L against run 1's
supply and exit arms (128 + 128, s16A_h100x8_v1; no 256-column cross block of run 1 exists there) and the 16 + 16 piece x_n32, and at 1e2 L against
the 1e2 L clock (run 1, dt 0.5 a, ARM=1, both windows) when --clock is given. Adds the independent anchor of second_route (own coupled-Taylor
propagation, 16 fixed directions at 6.93 L). Labels as in PAGE_Bprime_s16A_v1: port=sink2L, declared covariances, no far-zone words.
Every number below is printed by this script."""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s16A_end_lib_v1 as E  # noqa: E402

L = E.L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run1", required=True, help="s16A_h100x8_v1 (run 1 sup and exit arms to 69.32 L)")
    ap.add_argument("--x", required=True, help="s16A_h100x8_v2/g2 (run 1 both windows: x_full to 6.93 L, x_n32 to 69.3 L)")
    ap.add_argument("--c2", required=True, help="box copy of the 2c arm")
    ap.add_argument("--c2tag", default="s16A_2c_F1_dthalf_dt0.25_T100_both")
    ap.add_argument("--clock", default="", help="box copy of the 1e2 L clock arm (absent: the 1e2 L comparison waits)")
    ap.add_argument("--clocktag", default="s16A_1_F1_d0_dt0.5_T100_both")
    ap.add_argument("--lane", required=True, help="second_route own_spectral_v2/own_arrays.npz")
    ap.add_argument("--version", default="1")
    ap.add_argument("--name", default="PAGE_Bprime_s16A_2c_v1.md")
    ap.add_argument("--prev", default="", help="name of the earlier issued version (for v2 and later)")
    ap.add_argument("--test", action="store_true", help="fixture run: marks the file TEST")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    R1s = E.load_run(a.run1, "s16A_1_F1_d0_dt0.5_T69.32_sup")
    R1e = E.load_run(a.run1, "s16A_1_F1_d0_dt0.5_T69.32_exit")
    Xf = E.load_run(a.x, "s16A_1_F1_d0_dt0.5_T6.932_both")
    Xn = E.load_run(a.x, "s16A_1_F1_d0_dt0.5_T69.32_both_n32")
    C = E.load_run(a.c2, a.c2tag)
    CK = E.load_run(a.clock, a.clocktag) if a.clock else None
    cdx = np.load(a.lane)
    ns = C["ns"]; K = C["K"]
    logC = E.parse_log(os.path.join(a.c2, a.c2tag + ".log"))
    logK = E.parse_log(os.path.join(a.clock, a.clocktag + ".log")) if CK else None

    out = []; P = out.append
    P("# PAGE B' s = 16, stage A: run 2c (dt 0.25 a) read on all columns%s, version %s (busi lane)%s" % (
        " to 1e2 L against the 1e2 L clock" if CK else "", a.version, " [TEST FIXTURE, NOT A RESULT]" if a.test else ""))
    P("")
    P("Emitted whole by busi/code/pagetests/v3/s16A_read_2c_v1.py (helpers s16A_end_lib_v1.py) from md5-checked box copies; no number is retyped.%s Order: inbox #485 (i); the reads reviewer B listed (ReviewB_h100f_v1 sec. (c)): full-256-column G and ordered-p timestep differences at 1e2 L, the joint cross block and its spectrum, residuals against numerical error, onset sensitivity separately (in PAGE_Bprime_s16A_2b), actual full-span throughput and peak memory." % (
        (" A new file; %s stays as issued." % a.prev) if a.prev else ""))
    P("")
    P("**Labels (as PAGE_Bprime_s16A_v1).** `port=sink2L(onset 2 L, sin^2 ramp 0.5 L, eta_max 1/a, vacuum Markov bath)`. p is the weight a window direction delivers to the sink beyond 2 L; every readout is a property of h[N, v] on the box with this sink, on the held F1 fields (r_s/a 64 reference at T/r_s 30, fails the hydrodynamic gate inside), s = L/a = 16, sigma = L/8, d = 0. Not far widths, not greybody factors, not emitted radiation, not a Page curve. 'Delivered number' is Tr G. The exit window is the hole (conjugate) channel; joint (supply + exit) entropies need the Nambu form and are not reported.")
    P("")
    # ------------------------------------------------------------------ 1 manifest
    P("## 1. Manifest (from the run files and the engine logs)")
    P("")
    P("| arm | role | dt (a) | onset (L) | columns | clocks (T/L) | steps per column (T/dt) | step median (s) | engine wall (h) | full-span s per column per a | column-a per s | peak GPU (GB) | device | md5 vs MD5SUMS | npz md5 |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = [("2c", "dt 0.25 a control", C, logC)]
    if CK:
        rows.append(("1 (1e2 L clock)", "reference dt 0.5 a", CK, logK))
    rows += [("1 x_full", "reference at 6.93 L", Xf, E.parse_log(os.path.join(a.x, Xf["tag"] + ".log"))),
             ("1 x_n32", "16 + 16 reference to 69.3 L", Xn, E.parse_log(os.path.join(a.x, Xn["tag"] + ".log"))),
             ("1 sup", "128 reference to 69.3 L", R1s, E.parse_log(os.path.join(a.run1, R1s["tag"] + ".log"))),
             ("1 exit", "128 reference to 69.3 L", R1e, E.parse_log(os.path.join(a.run1, R1e["tag"] + ".log")))]
    man = {}
    for nm, role, R, lg in rows:
        m = E.manifest_row(R, lg); man[nm] = m
        P("| %s | %s | %s | %s | %d | %s | %d | %.3f | %.2f | %.4f | %.1f | %.1f | %s | %s | %s |" % (
            nm, role, m["dt"], m["onset"], m["K"], ", ".join("%.4g" % x for x in m["clocks"]), m["steps"], m["step_median"], m["wall_h"], m["rate_full"],
            m["col_a_per_s"], m["peak_GB"], m["device"], "OK" if m["md5_ok"] else "NOT LISTED", m["md5_npz"][:8]))
    P("")
    mc = man["2c"]
    P("Run 2c: %d columns (%d supply + %d exit) in batches of %d, to T = %.4g L = %.0f a, %d steps of %s a; engine wall %.0f s (%.2f h) from the log's 'wrote' line; resumed from %s. Actual full-span cost %.4f s per column per a (wall / (columns x T)); the step-median rate is %.4f. Peak GPU memory %.1f GB (the engine's torch peak, json timing.peak_mem_GB)." % (
        K, ns, K - ns, mc["col_batch"], mc["clocks"][-1], max(C["T"]), mc["steps"], mc["dt"], mc["wall_h"] * 3600, mc["wall_h"], mc["resumed_from"] or "nothing (one session)",
        mc["rate_full"], mc["rate_step"], mc["peak_GB"]))
    if logC["clock_done"]:
        P("")
        P("Per-clock engine seconds (2c log): %s." % "; ".join("T/L %.4g at %.0f s" % (t, s) for t, s in logC["clock_done"]))
    P("")
    P("Basis identity: the same input columns in every arm (max |e_2c - e_ref| over columns: x_full %.2g, run 1 supply %.2g, run 1 exit %.2g; max |qF_2c - qF_x_full| %.2g)." % (
        np.max(np.abs(C["z"]["e"] - Xf["z"]["e"])), np.max(np.abs(C["z"]["e"][:ns] - R1s["z"]["e"])), np.max(np.abs(C["z"]["e"][ns:] - R1e["z"]["e"])),
        np.max(np.abs(C["z"]["qF"] - Xf["z"]["qF"]))))
    P("")
    # ------------------------------------------------------------------ 2 dt bar at 6.93 L
    P("## 2. The time-step difference at 6.93 L on all 256 columns (2c dt 0.25 a against x_full dt 0.5 a)")
    P("")
    Gc1, Gx1 = E.G_at(C, E.T1), E.G_at(Xf, E.T1)
    c1 = E.cmp_full(Gc1, Gx1, ns)
    P("| quantity | supply block (128) | exit block (128) | cross block G_se | joint (256) |")
    P("|---|---|---|---|---|")
    P("| max \\|dG\\| element | %.3g | %.3g | %.3g | %.3g |" % (c1["max_dG_ss"], c1["max_dG_ee"], c1["max_dG_se"], c1["max_dG"]))
    P("| max \\|dp\\| ordered spectrum | %.3g | %.3g | - | %.3g |" % (np.max(np.abs(c1["dp_ss"])), np.max(np.abs(c1["dp_ee"])), np.max(np.abs(c1["dp_joint"]))))
    P("| directions with \\|dp\\| <= 1e-4 | %d of %d | %d of %d | - | %d of %d |" % ((np.abs(c1["dp_ss"]) <= 1e-4).sum(), ns, (np.abs(c1["dp_ee"]) <= 1e-4).sum(), K - ns,
                                                                                 (np.abs(c1["dp_joint"]) <= 1e-4).sum(), K))
    P("| delivered number dt 0.5 / dt 0.25 | %.5f / %.5f | %.5f / %.5f | - | %.5f / %.5f |" % (c1["p_ss_b"].sum(), c1["p_ss_a"].sum(), c1["p_ee_b"].sum(), c1["p_ee_a"].sum(),
                                                                                     c1["p_joint_b"].sum(), c1["p_joint_a"].sum()))
    P("| \\|\\|G_se\\|\\|_2 dt 0.5 / dt 0.25 | - | - | %.4g / %.4g | - |" % (c1["norm_se_b"], c1["norm_se_a"]))
    P("")
    P("The operator-norm difference over all 256 columns is %.3g; the signed median of the ordered joint dp is %.3g (dt 0.25 minus dt 0.5)." % (c1["norm2_dG"], np.median(c1["dp_joint"])))
    P("")
    # ------------------------------------------------------------------ 3 dt bar at 69.3 L
    P("## 3. The time-step difference at 69.3 L (2c against run 1's single-window arms on all 128 + 128 columns, and the 16 + 16 both-window piece)")
    P("")
    Gc2 = E.G_at(C, E.T2)
    s2 = E.cmp_block(Gc2[:ns, :ns], E.G_at(R1s, E.T2)); e2 = E.cmp_block(Gc2[ns:, ns:], E.G_at(R1e, E.T2))
    P("| block | columns | max \\|dG\\| | \\|\\|dG\\|\\|_2 | max \\|d diag G\\| | max \\|dp\\| ordered | median \\|dp\\| | directions with \\|dp\\| <= 1e-4 | delivered dt 0.5 | dt 0.25 |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for nm, c, n in (("supply", s2, ns), ("exit", e2, K - ns)):
        P("| %s | %d | %.3g | %.3g | %.3g | %.3g | %.3g | %d of %d | %.6f | %.6f |" % (nm, n, c["max_dG"], c["norm2_dG"], c["max_ddiag"], np.max(np.abs(c["dp"])), np.median(np.abs(c["dp"])),
                                                                         (np.abs(c["dp"]) <= 1e-4).sum(), n, c["pb"].sum(), c["pa"].sum()))
    # 16 + 16 subset: identify the columns by energy
    en = Xn["z"]["e"]; ec = C["z"]["e"]; nsn = Xn["ns"]
    idx = []
    used = set()
    for x in en:
        cand = [i for i in np.argsort(np.abs(ec - x)) if i not in used]
        idx.append(cand[0]); used.add(cand[0])
    idx = np.array(idx)
    e_match = float(np.max(np.abs(ec[idx] - en)))
    q_match = float(np.max(np.abs(C["z"]["qF"][np.ix_(idx, idx)] - Xn["z"]["qF"])))
    Gsub = Gc2[np.ix_(idx, idx)]; Gn2 = E.G_at(Xn, E.T2)
    n2 = E.cmp_full(Gsub, Gn2, nsn)
    P("| x_n32 subset, all four blocks | %d + %d | %.3g | %.3g | - | %.3g (joint) | %.3g | %d of %d | %.6f | %.6f |" % (
        nsn, len(en) - nsn, n2["max_dG"], n2["norm2_dG"], np.max(np.abs(n2["dp_joint"])), np.median(np.abs(n2["dp_joint"])), (np.abs(n2["dp_joint"]) <= 1e-4).sum(), len(en),
        n2["p_joint_b"].sum(), n2["p_joint_a"].sum()))
    P("")
    P("The x_n32 columns are located in 2c by energy (max |e| mismatch %.2g, max |qF| mismatch %.2g); its cross-block difference is %.3g (cross-block norm dt 0.5 %.3g, dt 0.25 %.3g)." % (
        e_match, q_match, n2["max_dG_se"], n2["norm_se_b"], n2["norm_se_a"]))
    P("")
    worst69 = max(s2["max_dG"], e2["max_dG"], n2["max_dG"])
    worst6 = c1["max_dG"]
    P("Verdict against the 1e-4 target (ReviewA_bpchk2_v1 sec. 4 criterion 2, between dt 0.5 and dt 0.25): at 6.93 L, max |dG| over all 256 x 256 elements = %.3g, **%s**; at 69.3 L, max |dG| over the 128 x 128 supply and exit blocks and the 32 x 32 both-window subset = %.3g, **%s**. The 256-column cross block at 69.3 L has no dt 0.5 reference except from the 1e2 L clock (sec. 5)." % (
        worst6, "MEETS 1e-4" if worst6 <= 1e-4 else "ABOVE 1e-4", worst69, "MEETS 1e-4" if worst69 <= 1e-4 else "ABOVE 1e-4"))
    P("")
    # ------------------------------------------------------------------ 4 lane anchor
    P("## 4. The independent anchor at 6.93 L: three step sizes against second_route's own propagation (16 fixed directions)")
    P("")
    U = cdx["benchmark_U"]; F = cdx["fine_loss_Gram"]; ranks = cdx["benchmark_ranks"]
    A05 = U.conj().T @ E.G_at(R1s, E.T1) @ U
    A025 = U.conj().T @ Gc1[:ns, :ns] @ U
    e05 = float(np.max(np.abs(A05 - F))); e025 = float(np.max(np.abs(A025 - F))); d0 = float(np.max(np.abs(A05 - A025)))
    ev05 = float(np.max(np.abs(np.linalg.eigvalsh(E.herm(A05)) - np.linalg.eigvalsh(E.herm(F)))))
    ev025 = float(np.max(np.abs(np.linalg.eigvalsh(E.herm(A025)) - np.linalg.eigvalsh(E.herm(F)))))
    sgn05 = float(np.median(np.real(np.diag(A05 - F)))); sgn025 = float(np.median(np.real(np.diag(A025 - F))))
    order = math.log(e05 / e025, 2) if e025 > 0 else float("nan")
    P("second_route (review_Bprime_s16A_v1) propagated the 16 source eigendirections of run 1's 6.93 L supply Gram (ranks %s) with its own coupled degree-32 Taylor action, fine/coarse agreement 2.2e-16; its 16 x 16 fine loss Gram F is the reference here (Q - Psi^dag Psi convention; the max |Q - I| difference is 3e-14). U^dag G U from each of our step sizes against F:" % ", ".join(str(int(r)) for r in ranks))
    P("")
    P("| our step | max \\|U^dag G U - F\\| element | max \\|eigenvalue difference\\| | median diagonal difference (signed) |")
    P("|---|---|---|---|")
    P("| dt 0.5 a (run 1) | %.3g | %.3g | %.3g |" % (e05, ev05, sgn05))
    P("| dt 0.25 a (2c) | %.3g | %.3g | %.3g |" % (e025, ev025, sgn025))
    P("")
    P("Difference between our two steps on these directions: %.3g. Observed order from the anchor, log2(err(0.5)/err(0.25)) = %.3g. At first order the dt 0.5 a error is twice the dt 0.5 to dt 0.25 difference, at second order 4/3 of it; the anchor gives err(0.5)/difference = %.3g and err(0.25)/difference = %.3g directly." % (
        d0, order, e05 / d0 if d0 > 0 else float("nan"), e025 / d0 if d0 > 0 else float("nan")))
    step_like = (d0 > 0) and (e025 < e05 / 1.4)
    if step_like:
        fac05, fac025, floor = e05 / d0, e025 / d0, 0.0
        P("")
        P("Rule fixed before the data: the anchor's discrepancy counts as a step-size effect if err(0.25) < err(0.5)/1.4 (observed order above 1/2). It does here, so the numerical bar below is |dp| (dt 0.5 to dt 0.25) times the anchor's ratio for the step read. This transfers a 16-direction, 6.93 L calibration to the other directions and clocks; it is an estimate, not a certificate.")
    else:
        fac05, fac025, floor = 1.0, 1.0, max(e05, e025)
        P("")
        P("Rule fixed before the data: the anchor's discrepancy counts as a step-size effect only if err(0.25) < err(0.5)/1.4. It does not here (halving the step does not shrink the gap to the independent propagation), so the gap is not our step error; the numerical bar below is |dp| (dt 0.5 to dt 0.25) plus a step-independent floor equal to the larger anchor gap, %.3g. The origin of that floor is not identified by this read." % floor)
    P("")
    # ------------------------------------------------------------------ 5 1e2 L
    P("## 5. At 1e2 L: the joint 256-column Gram, its cross block and spectrum")
    P("")
    Gc3 = E.G_at(C, E.T3)
    Js = {"2c dt 0.25": E.joint_summary(Gc3, ns)}
    Gk3 = None
    if CK:
        Gk3 = E.G_at(CK, E.T3); Js["clock dt 0.5"] = E.joint_summary(Gk3, ns)
    P("| arm | delivered number (joint) | \\|\\|G_se\\|\\|_2 | max \\|G_se\\| | top 3 singular values of G_se | joint min p | joint max p | dark count p < 1e-2 | max \\|p_joint - p_union\\| | supply max (1 - p) | exit max (1 - p) | max \\|p_sup - p_exit\\| (mirror) |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for nm, s in Js.items():
        P("| %s | %.6f | %.3g | %.3g | %s | %.6f | %.6f | %d | %.3g | %.3g | %.3g | %.2g |" % (nm, s["trace"], s["norm_se"], s["max_se"], ", ".join("%.3g" % x for x in s["sv_se"][:3]),
                                                                                 s["pj"].min(), s["pj"].max(), s["dark"], s["max_pj_minus_pu"], 1 - s["ps"].min(), 1 - s["pe"].min(), s["mirror_dp"]))
    P("")
    P("Joint spectrum of 2c at the three clocks (the cross block's history):")
    P("")
    P("| T/L | delivered (joint) | \\|\\|G_se\\|\\|_2 | joint min p | count p > 1/2 | count 1 - p > 1e-2 | > 1e-3 | > 1e-4 | max \\|p_joint - p_union\\| |")
    P("|---|---|---|---|---|---|---|---|---|")
    for i, T in enumerate(C["T"]):
        s = E.joint_summary(np.asarray(C["z"]["G"][i]), ns); r = 1 - s["pj"]
        P("| %.4g | %.6f | %.3g | %.6f | %d | %d | %d | %d | %.3g |" % (T / L, s["trace"], s["norm_se"], s["pj"].min(), (s["pj"] > 0.5).sum(), (r > 1e-2).sum(), (r > 1e-3).sum(), (r > 1e-4).sum(), s["max_pj_minus_pu"]))
    P("")
    if CK:
        P("**Time-step differences against the 1e2 L clock, all 256 columns, at every common clock:**")
        P("")
        P("| T/L | max \\|dG\\| (256 x 256) | \\|\\|dG\\|\\|_2 | supply block | exit block | cross block | max \\|dp\\| ordered joint | supply | exit | directions with \\|dp_joint\\| <= 1e-4 |")
        P("|---|---|---|---|---|---|---|---|---|---|")
        cK = {}
        for T in (E.T1, E.T2, E.T3):
            Ga, Gb = E.G_at(C, T), E.G_at(CK, T)
            if Ga is None or Gb is None:
                continue
            c = E.cmp_full(Ga, Gb, ns); cK[T] = c
            P("| %.4g | %.3g | %.3g | %.3g | %.3g | %.3g | %.3g | %.3g | %.3g | %d of %d |" % (T / L, c["max_dG"], c["norm2_dG"], c["max_dG_ss"], c["max_dG_ee"], c["max_dG_se"], np.max(np.abs(c["dp_joint"])),
                                                                               np.max(np.abs(c["dp_ss"])), np.max(np.abs(c["dp_ee"])), (np.abs(c["dp_joint"]) <= 1e-4).sum(), K))
        P("")
        Gk1 = E.G_at(CK, E.T1)
        if Gk1 is not None:
            P("Regression: the clock against x_full at 6.93 L (same step, same columns): max |dG| = %.3g (PASS if below 1e-10: %s)." % (
                np.max(np.abs(Gk1 - Gx1)), "PASS" if np.max(np.abs(Gk1 - Gx1)) < 1e-10 else "FAIL"))
            P("")
        w3 = cK[E.T3]["max_dG"] if E.T3 in cK else float("nan")
        P("Verdict at the endpoint: max |dG| over all 256 x 256 elements at 1e2 L = %.3g, **%s** (1e-4 target)." % (w3, "MEETS 1e-4" if w3 <= 1e-4 else "ABOVE 1e-4"))
    else:
        P("**What waits:** the 1e2 L clock (run 1, dt 0.5 a, ARM=1, both windows, on qi2_h100_1, due about 18:40 UTC). Until it lands there is no dt 0.5 a Gram at 1e2 L, so the full-256-column G and ordered-p timestep differences at 1e2 L, the dt difference of the 256-column cross block at 69.3 L and 1e2 L, and the residual-against-error count at 1e2 L are not given here; they come in version 2 (a new file). The 2c rows above are 2c's own readouts.")
    P("")
    # ------------------------------------------------------------------ 6 residuals vs numerical error
    P("## 6. Residuals against numerical error")
    P("")
    P("Residual r = 1 - p per ordered direction. The bar per ordered direction is |dp| between the two step sizes times the sec. 4 factor (x %.3g for dt 0.5 a, x %.3g for dt 0.25 a) plus the sec. 4 floor (%.3g). 'Resolved' means r larger than ten times that bar." % (fac05, fac025, floor))
    P("")
    P("| T/L | block | step read | directions | max r | median r | count r > 1e-3 | r > 1e-4 | max bar | resolved (r > 10 bar) | r > bar |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for nm, c in (("supply", s2), ("exit", e2)):
        for lab, p, fac in (("dt 0.5 a (run 1)", c["pb"], fac05), ("dt 0.25 a (2c)", c["pa"], fac025)):
            rr = E.residual_vs_bar(p, np.abs(c["dp"]) * fac + floor)
            P("| %.4g | %s | %s | %d | %.3g | %.3g | %d | %d | %.3g | %d | %d |" % (E.T2 / L, nm, lab, rr["n"], rr["max_res"], rr["median_res"], rr["n_res_gt_1e3"], rr["n_res_gt_1e4"], rr["max_bar"], rr["n_resolved_10x"], rr["n_resolved_1x"]))
    if CK and E.T3 in cK:
        c = cK[E.T3]
        for nm, pa, pb, dp in (("supply", c["p_ss_a"], c["p_ss_b"], c["dp_ss"]), ("exit", c["p_ee_a"], c["p_ee_b"], c["dp_ee"]), ("joint", c["p_joint_a"], c["p_joint_b"], c["dp_joint"])):
            for lab, p, fac in (("dt 0.5 a (clock)", pb, fac05), ("dt 0.25 a (2c)", pa, fac025)):
                rr = E.residual_vs_bar(p, np.abs(dp) * fac + floor)
                P("| %.4g | %s | %s | %d | %.3g | %.3g | %d | %d | %.3g | %d | %d |" % (E.T3 / L, nm, lab, rr["n"], rr["max_res"], rr["median_res"], rr["n_res_gt_1e3"], rr["n_res_gt_1e4"], rr["max_bar"], rr["n_resolved_10x"], rr["n_resolved_1x"]))
    else:
        s3 = Js["2c dt 0.25"]
        for nm, p in (("supply", s3["ps"]), ("exit", s3["pe"]), ("joint", s3["pj"])):
            rr = E.residual_vs_bar(p, 0.0)
            P("| %.4g | %s | dt 0.25 a (2c), bar waits for the clock | %d | %.3g | %.3g | %d | %d | - | - | - |" % (E.T3 / L, nm, rr["n"], rr["max_res"], rr["median_res"], rr["n_res_gt_1e3"], rr["n_res_gt_1e4"]))
    P("")
    # ------------------------------------------------------------------ 7 settles / waits
    P("## 7. What this read settles, and what waits")
    P("")
    P("- Settled: the dt 0.5 a production step's difference from dt 0.25 a on all 256 columns at 6.93 L (sec. 2) and on all 128 + 128 window columns at 69.3 L (sec. 3), now replacing the 24-column preview of PAGE_Bprime_s16A_v1.1 sec. 3; the independent anchor's three-step check (sec. 4); 2c's joint spectrum and cross block at every clock including 1e2 L (sec. 5); the actual full-span throughput and peak memory of a 256-column both-window arm at dt 0.25 a (sec. 1).")
    if CK:
        P("- Settled with the clock: the full-256-column G and ordered-p timestep differences at 1e2 L, the cross block's dt difference at 69.3 L and 1e2 L, and the residual-against-error count at 1e2 L (secs. 5 and 6).")
    else:
        P("- Waits for the 1e2 L clock (qi2_h100_1, about 18:40 UTC): the full-256-column G and ordered-p timestep differences at 1e2 L; the 256-column cross block's dt difference at 69.3 L and 1e2 L; the residual-against-error count at 1e2 L. Version 2 will be a new file.")
    P("- Separately (PAGE_Bprime_s16A_2b): the onset sensitivity, reported as boundary dependence of the sink2L model, not as an error bar.")
    P("")
    files = [(C["dir"], C["tag"], C["md5_npz"])] + ([(CK["dir"], CK["tag"], CK["md5_npz"])] if CK else []) + [(Xf["dir"], Xf["tag"], Xf["md5_npz"]), (Xn["dir"], Xn["tag"], Xn["md5_npz"]),
                                                                                                                (R1s["dir"], R1s["tag"], R1s["md5_npz"]), (R1e["dir"], R1e["tag"], R1e["md5_npz"])]
    P("Files (npz md5): %s; lane anchor %s (md5 %s); helpers s16A_end_lib_v1.py (md5 %s); this reader (md5 %s)." % (
        "; ".join("%s/%s.npz (%s)" % (d, t, m[:8]) for d, t, m in files), a.lane, E.md5(a.lane)[:8], E.md5(os.path.join(HERE, "s16A_end_lib_v1.py"))[:8], E.md5(os.path.abspath(__file__))[:8]))
    open(a.out, "w").write("\n".join(out) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
