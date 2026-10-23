#!/usr/bin/env python
"""s16A_gate_v1.py (busi; inbox #488): emits PAGE_Bprime_s16A_gate_v1.md whole, the stage A gate package from the 1e2 L clock arm
(run 1, onset 2 L, dt 0.5 a, both windows, 256 columns) with the union of reviewer A's and reviewer B's gate items 1 to 7 (ReviewA_s16B_v1 sec. 3,
ReviewB_s16B_v1 sec. 3), and the v2 corrections (a) to (g) to PAGE_Bprime_s16A_2c_v1 and PAGE_Bprime_s16A_2b_v1 (those files stay as issued).
Every number is printed by this script from md5-checked box copies. Labels: port=sink2L, declared covariances, block-diagonal records only,
no joint entropy, no far-zone words."""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s16A_end_lib_v2 as E  # noqa: E402

L = E.L
DON = 0.25
CLK = "s16A_1_F1_d0_dt0.5_T100_both"
C2T = "s16A_2c_F1_dthalf_dt0.25_T100_both"
B2T = "s16A_2b_F1_onset225_dt0.5_T100_both"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run1", required=True)
    ap.add_argument("--x", required=True)
    ap.add_argument("--clock", required=True)
    ap.add_argument("--c2", required=True)
    ap.add_argument("--b2", required=True)
    ap.add_argument("--lane", required=True)
    ap.add_argument("--decl", required=True, help="stageA_decl_v2 (declared hole packet, covariances, manifest)")
    ap.add_argument("--prep", required=True, help="kit prep dir: F1_s16_d0_prep.npz, F1_s16_d0_onset225_bb_prep.npz")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    R1s = E.load_run(a.run1, "s16A_1_F1_d0_dt0.5_T69.32_sup")
    R1e = E.load_run(a.run1, "s16A_1_F1_d0_dt0.5_T69.32_exit")
    Xf = E.load_run(a.x, "s16A_1_F1_d0_dt0.5_T6.932_both")
    Xn = E.load_run(a.x, "s16A_1_F1_d0_dt0.5_T69.32_both_n32")
    K1 = E.load_run(a.clock, CLK)
    C2 = E.load_run(a.c2, C2T)
    B2 = E.load_run(a.b2, B2T)
    cdx = np.load(a.lane)
    ns = K1["ns"]; K = K1["K"]
    T1, T2, T3 = E.T1, E.T2, E.T3
    G = lambda R, T: E.G_at(R, T)
    Rr = lambda R, T: (None if E.cidx(R["T"], T) is None else np.asarray(R["z"]["R"][E.cidx(R["T"], T)]))

    out = []; P = out.append
    P("# PAGE B' s = 16, stage A: the gate package from the 1e2 L clock, with the v2 corrections to the 2c and 2b reads, version 1 (busi lane)%s" % (
        " [TEST FIXTURE, NOT A RESULT]" if a.test else ""))
    P("")
    P("Emitted whole by busi/code/pagetests/v3/s16A_gate_v1.py (helpers s16A_end_lib_v1.py, s16A_end_lib_v2.py) from md5-checked box copies; no number is retyped. Orders: inbox #488 (gate items 1 to 7, the union of ReviewA_s16B_v1 sec. 3 and ReviewB_s16B_v1 sec. 3; v2 corrections (a) to (g)). This file is the v2 of PAGE_Bprime_s16A_2c_v1.md and PAGE_Bprime_s16A_2b_v1.md, which stay as issued, and supersedes PAGE_Bprime_s16A_v1.1.md sec. 3's convergence sentence. Nothing here is paper text; nothing prints until the gate review.")
    P("")
    P("**Labels.** `port=sink2L(onset 2 L, sin^2 ramp 0.5 L, eta_max 1/a, vacuum Markov bath)` (clock, 2c); 2b: `port=sink2L(onset 2.25 L, sin^2 ramp 0.5 L, eta_max 1/a, half-side 2.75 L, vacuum Markov bath)`. p is the weight a window direction delivers to the sink beyond the onset; held F1 fields (r_s/a 64 reference at T/r_s 30, fails the hydrodynamic gate inside), s = 16, sigma = L/8, d = 0. Not far widths, not greybody factors, not emitted radiation, not radiation entropy, not a Page curve. Bath-record quantities are block-diagonal under the declared covariances; no joint (supply + exit) entropy is reported (it needs the Nambu form).")
    P("")
    verdict = {}

    # ================================================================== 7 (first: manifest, it binds the inputs)
    P("## 1. Manifest (gate item 7, first part)")
    P("")
    P("| arm | dt (a) | onset (L) | half-side (L) | prep | columns | clocks (T/L) | steps per column | step median (s) | engine wall (h) | full-span s per column per a | peak GPU (GB) | device | md5 vs MD5SUMS | npz md5 |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    man = {}
    for nm, R in (("1 (1e2 L clock)", K1), ("2c", C2), ("2b", B2), ("1 x_full", Xf), ("1 x_n32", Xn), ("1 sup", R1s), ("1 exit", R1e)):
        m = E.manifest_row(R, E.parse_log(os.path.join(R["dir"], R["tag"] + ".log"))); man[nm] = m
        P("| %s | %s | %s | %s | %s | %d | %s | %d | %.3f | %.2f | %.4f | %.1f | %s | %s | %s |" % (nm, m["dt"], m["onset"], m["box_L"], m["prep"], m["K"], ", ".join("%.4g" % x for x in m["clocks"]),
                                                                                       m["steps"], m["step_median"], m["wall_h"], m["rate_full"], m["peak_GB"], m["device"], "OK" if m["md5_ok"] else "NOT LISTED", m["md5_npz"][:8]))
    P("")
    pmK = K1["j"].get("prep_manifest", {}); pmX = Xf["j"].get("prep_manifest", {})
    same_pm = json.dumps({k: v for k, v in pmK.items() if k != "args"}, sort_keys=True) == json.dumps({k: v for k, v in pmX.items() if k != "args"}, sort_keys=True)
    dE = float(np.max(np.abs(K1["z"]["e"] - Xf["z"]["e"]))); dQ = float(np.max(np.abs(K1["z"]["qF"] - Xf["z"]["qF"])))
    dEs = float(np.max(np.abs(K1["z"]["e"][:ns] - R1s["z"]["e"]))); dEe = float(np.max(np.abs(K1["z"]["e"][ns:] - R1e["z"]["e"])))
    pk = os.path.join(a.prep, "F1_s16_d0_prep.npz")
    P("Prep and basis: the clock's prep is %s; its recorded prep manifest equals run 1's (x_full) apart from run arguments: %s (prep npz md5 recorded in both: %s and %s). Basis identity against run 1: max |e_clock - e_x_full| = %.2g, max |qF_clock - qF_x_full| = %.2g, against the single-window arms %.2g (supply) and %.2g (exit). The kit prep on the box, %s, has md5 %s." % (
        man["1 (1e2 L clock)"]["prep"], "YES" if same_pm else "NO", pmK.get("npz_md5", "-")[:8], pmX.get("npz_md5", "-")[:8], dE, dQ, dEs, dEe, pk, E.md5(pk)[:8]))
    P("")
    mk = man["1 (1e2 L clock)"]
    P("Actual cost of the clock: %d columns to %.4g L, engine wall %.2f h, full-span %.4f s per column per a (step-median rate %.4f), peak GPU %.1f GB, %s." % (
        K, mk["clocks"][-1], mk["wall_h"], mk["rate_full"], mk["rate_step"], mk["peak_GB"], mk["device"]))
    P("")

    # ================================================================== 1 regression
    P("## 2. Regression at 1e-10 (gate item 1; FAIL -> HOLD)")
    P("")
    P("| comparison | T/L | columns | max \\|dG\\| | max \\|dR\\| | PASS at 1e-10 |")
    P("|---|---|---|---|---|---|")
    reg = []
    Gk1, Gx1 = G(K1, T1), G(Xf, T1)
    reg.append(("clock vs x_full, all four blocks", T1, "%d" % K, float(np.max(np.abs(Gk1 - Gx1))), float(np.max(np.abs(Rr(K1, T1) - Rr(Xf, T1))))))
    Gk2 = G(K1, T2)
    reg.append(("clock supply block vs run 1 supply arm", T2, "%d" % ns, float(np.max(np.abs(Gk2[:ns, :ns] - G(R1s, T2)))), float(np.max(np.abs(Rr(K1, T2)[:ns, :ns] - Rr(R1s, T2))))))
    reg.append(("clock exit block vs run 1 exit arm", T2, "%d" % (K - ns), float(np.max(np.abs(Gk2[ns:, ns:] - G(R1e, T2)))), float(np.max(np.abs(Rr(K1, T2)[ns:, ns:] - Rr(R1e, T2))))))
    idx, emm = E.locate(K1["z"]["e"], Xn["z"]["e"])
    reg.append(("clock 16 + 16 subset (with its cross block) vs x_n32", T2, "%d + %d" % (Xn["ns"], Xn["K"] - Xn["ns"]), float(np.max(np.abs(Gk2[np.ix_(idx, idx)] - G(Xn, T2)))),
                float(np.max(np.abs(Rr(K1, T2)[np.ix_(idx, idx)] - Rr(Xn, T2))))))
    reg_ok = True
    for nm, T, c, dg, dr in reg:
        ok = dg < 1e-10
        reg_ok &= ok
        P("| %s | %.4g | %s | %.3g | %.3g | %s |" % (nm, T / L, c, dg, dr, "PASS" if ok else "FAIL"))
    P("")
    P("Regression verdict (on G, the certified quantity; R shown alongside): **%s**%s. The x_n32 columns are located in the clock by energy (max mismatch %.2g)." % (
        "PASS" if reg_ok else "FAIL", "" if reg_ok else " -> HOLD (ReviewA_bpchk2_v1 sec. 6)", emm))
    P("")
    verdict["1 regression"] = "PASS" if reg_ok else "FAIL -> HOLD"

    # ================================================================== anchor (correction g, a)
    P("## 3. The two production steps against the independent reference (correction (g); correction (a))")
    P("")
    U = cdx["benchmark_U"]; F = cdx["fine_loss_Gram"]
    A05 = U.conj().T @ G(R1s, T1) @ U; A025 = U.conj().T @ G(C2, T1)[:ns, :ns] @ U
    e05 = float(np.max(np.abs(A05 - F))); e025 = float(np.max(np.abs(A025 - F))); d0 = float(np.max(np.abs(A05 - A025)))
    order = math.log(e05 / e025, 2); fac = e05 / d0
    P("Two production steps, dt 0.5 a (run 1) and dt 0.25 a (2c), each compared with ONE independent reference: second_route's own coupled degree-32 Taylor propagation of 16 fixed directions at 6.93 L (review_Bprime_s16A_v1; its own fine/coarse agreement is at rounding). The reference is not a third step size.")
    P("")
    P("| production step | max \\|U^dag G U - F\\| against the reference |")
    P("|---|---|")
    P("| dt 0.5 a | %.3g |" % e05)
    P("| dt 0.25 a | %.3g |" % e025)
    P("")
    P("Difference between the two steps on these directions %.3g; observed order log2(err(0.5)/err(0.25)) = %.3g; err(0.5)/difference = %.3g. This factor is used below as the bar multiplier (bar = %.3g |dp|). It is a 16-direction, 6.93 L calibration transferred to other directions and clocks: an ESTIMATE, not a certificate, and it gives no absolute late-time bound." % (d0, order, fac, fac))
    P("")
    P("**Correction (a).** PAGE_Bprime_s16A_v1.1 sec. 3 said 'With the ladder's first-order convergence, the dt 0.5 a production error is about this difference.' That is WITHDRAWN: the anchor gives order %.3g, so the dt 0.5 a error is %.3g times the dt 0.5 to dt 0.25 difference (second order: 4/3), not first-order and not equal to it." % (order, fac))
    P("")

    # ================================================================== 2 endpoint dt
    P("## 4. The time step at the endpoint (gate item 2): 2c (dt 0.25 a) against the clock (dt 0.5 a), all 256 columns")
    P("")
    P("| T/L | max \\|dG\\| 256 x 256 | \\|\\|dG\\|\\|_2 | supply block | exit block | cross block (max \\|dG_se\\|) | \\|\\|dG_se\\|\\|_2 | max \\|dp\\| ordered: supply | exit | joint | directions with \\|dp_joint\\| <= 1e-4 | signed median dp_joint (dt 0.25 minus dt 0.5) | max \\|dR\\| (correction (c)) |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    cK = {}
    for T in (T1, T2, T3):
        Ga, Gb = G(C2, T), G(K1, T)
        c = E.cmp_full(Ga, Gb, ns); cK[T] = c
        c["norm2_dse"] = float(np.linalg.norm(Ga[:ns, ns:] - Gb[:ns, ns:], 2))
        dR = float(np.max(np.abs(Rr(C2, T) - Rr(K1, T))))
        c["dR"] = dR
        P("| %.4g | %.3g | %.3g | %.3g | %.3g | %.3g | %.3g | %.3g | %.3g | %.3g | %d of %d | %.3g | %.3g |" % (
            T / L, c["max_dG"], c["norm2_dG"], c["max_dG_ss"], c["max_dG_ee"], c["max_dG_se"], c["norm2_dse"], np.max(np.abs(c["dp_ss"])), np.max(np.abs(c["dp_ee"])),
            np.max(np.abs(c["dp_joint"])), (np.abs(c["dp_joint"]) <= 1e-4).sum(), K, np.median(c["dp_joint"]), dR))
    P("")
    c3 = cK[T3]
    ok_el = c3["max_dG"] <= 1e-4
    ok_sp = max(np.max(np.abs(c3["dp_ss"])), np.max(np.abs(c3["dp_ee"])), np.max(np.abs(c3["dp_joint"]))) <= 1e-4
    ok_all = (np.abs(c3["dp_joint"]) <= 1e-4).sum() == K
    flag = cK[T3]["max_dG_se"] > cK[T2]["max_dG_se"]
    P("Endpoint verdict at 1e2 L: max elementwise |dG| %.3g (%s), max ordered |dp| over blocks and joint %.3g (%s), %d of %d joint directions within 1e-4 (%s). Cross-block dt difference: %.3g at 69.3 L, %.3g at 1e2 L: %s." % (
        c3["max_dG"], "<= 1e-4" if ok_el else "ABOVE 1e-4", max(np.max(np.abs(c3["dp_ss"])), np.max(np.abs(c3["dp_ee"])), np.max(np.abs(c3["dp_joint"]))), "<= 1e-4" if ok_sp else "ABOVE 1e-4",
        (np.abs(c3["dp_joint"]) <= 1e-4).sum(), K, "all" if ok_all else "NOT all", cK[T2]["max_dG_se"], cK[T3]["max_dG_se"],
        "LARGER at 1e2 L than at 69.3 L: a FLAG, not a pass" if flag else "not larger at 1e2 L than at 69.3 L (no flag)"))
    P("")
    P("Correction (c): the kit saves R(T) (the retained Gram); its dt difference is the last column above. Only G is held to the 1e-4 criterion; R is reported, not certified.")
    P("")
    verdict["2 endpoint dt"] = "PASS" if (ok_el and ok_sp and ok_all and not flag) else ("FLAG (cross block grows)" if (ok_el and ok_sp and ok_all) else "ABOVE 1e-4")

    # ================================================================== 3 residuals
    P("## 5. Residuals against numerical error at 1e2 L (gate item 3)")
    P("")
    P("r = 1 - p per ordered direction of the clock (dt 0.5 a); bar = %.3g |dp| per ordered direction (dp: clock against 2c, sec. 4). Resolved: r > bar (and, stricter, r > 10 bar). A threshold crossing (r within bar of 1e-2, 1e-3 or 1e-4) is marked unresolved." % fac)
    P("")
    P("| spectrum | directions | max r | median r | Sum r | r > 1e-2 | r > 1e-3 | 1e-4 < r <= 1e-3 | r > 1e-4 | dark (p < 1e-2) | max bar | resolved r > bar | r > 10 bar | unresolved threshold crossings |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    res = {}
    for nm, p, dp in (("supply", c3["p_ss_b"], c3["dp_ss"]), ("exit", c3["p_ee_b"], c3["dp_ee"]), ("joint", c3["p_joint_b"], c3["dp_joint"])):
        r = 1 - p; bar = fac * np.abs(dp)
        unres = int(sum(((np.abs(r - th) <= bar).sum()) for th in (1e-2, 1e-3, 1e-4)))
        res[nm] = (r, bar)
        P("| %s | %d | %.3g | %.3g | %.4g | %d | %d | %d | %d | %d | %.3g | %d | %d | %d |" % (nm, len(r), r.max(), np.median(r), r.sum(), (r > 1e-2).sum(), (r > 1e-3).sum(), ((r > 1e-4) & (r <= 1e-3)).sum(),
                                                                                (r > 1e-4).sum(), (p < 1e-2).sum(), bar.max(), (r > bar).sum(), (r > 10 * bar).sum(), unres))
    P("")
    P("Joint spectrum and cross block of the clock at every clock (correction (d): the multiplicity of G_se's top singular value):")
    P("")
    P("| T/L | delivered (joint) | Sum (1 - p) joint | joint min p | dark | \\|\\|G_se\\|\\|_2 | its multiplicity | next singular value | max \\|p_joint - p_union\\| | supply max (1 - p) | exit max (1 - p) |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for T in (T1, T2, T3):
        Gk = G(K1, T); s = E.joint_summary(Gk, ns); top, m, nxt, _ = E.top_sv_mult(Gk[:ns, ns:])
        P("| %.4g | %.6f | %.4g | %.6f | %d | %.4g | %d | %.4g | %.3g | %.3g | %.3g |" % (T / L, s["trace"], K - s["trace"], s["pj"].min(), s["dark"], top, m, nxt, s["max_pj_minus_pu"], 1 - s["ps"].min(), 1 - s["pe"].min()))
    P("")
    verdict["3 residuals"] = "reported"

    # ================================================================== 4 onset at equal dt (corrections b, e, f)
    P("## 6. Onset at equal dt (gate item 4; corrections (b), (e), (f)): 2b (onset 2.25 L) against the clock (onset 2 L), both dt 0.5 a")
    P("")
    P("Boundary dependence of the sink2L model, a result, NOT an error bar. **Correction (b):** what PAGE_Bprime_s16A_2b_v1 called dp/d(onset) is the one-sided difference dp/0.25 L over [2, 2.25] L, named so here until a 2.5 L point exists. Caveat (reviewer B): there is no time-step refinement at onset 2.25 L, so the comparison with the onset-2 L step bar below says whether the boundary dependence exceeds the baseline's step sensitivity, not that both boundary models are numerically certified.")
    P("")
    P("| T/L | spectrum | directions | delivered, onset 2 L | onset 2.25 L | Sum (1 - p), onset 2 L | onset 2.25 L | onset-induced Sum (1 - p) | max (1 - p) onset 2 L | onset 2.25 L | max \\|dp\\| ordered | signed median dp | one-sided dp/0.25 L, median |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    cO = {}
    for T in (T1, T2, T3):
        c = E.cmp_full(G(B2, T), G(K1, T), ns); cO[T] = c
        for nm, pa, pb in (("supply", c["p_ss_a"], c["p_ss_b"]), ("exit", c["p_ee_a"], c["p_ee_b"]), ("joint", c["p_joint_a"], c["p_joint_b"])):
            dp = pa - pb
            P("| %.4g | %s | %d | %.6f | %.6f | %.4g | %.4g | %.4g | %.3g | %.3g | %.3g | %.3g | %.3g |" % (T / L, nm, len(pa), pb.sum(), pa.sum(), (1 - pb).sum(), (1 - pa).sum(), (1 - pa).sum() - (1 - pb).sum(),
                                                                                   (1 - pb).max(), (1 - pa).max(), np.max(np.abs(dp)), np.median(dp), np.median(dp) / DON))
    P("")
    sj2 = (1 - cO[T2]["p_joint_a"]).sum() - (1 - cO[T2]["p_joint_b"]).sum(); sj3 = (1 - cO[T3]["p_joint_a"]).sum() - (1 - cO[T3]["p_joint_b"]).sum()
    s2b = (1 - cO[T2]["p_joint_b"]).sum(); s3b = (1 - cO[T3]["p_joint_b"]).sum()
    P("**Correction (f) (reviewer A).** At equal dt the onset-induced undelivered weight Sum(1 - p) (joint) is %.4g at 69.3 L and %.4g at 1e2 L, while the onset-2 L residual Sum(1 - p) drains from %.4g to %.4g. At 1e2 L the onset-induced part is %.3g of the onset-2 L residual and %.3g of the onset-2.25 L residual: the late residual is set by the sink boundary, so no retention statement about the record rests on the sink2L residual." % (
        sj2, sj3, s2b, s3b, sj3 / s3b, sj3 / (1 - cO[T3]["p_joint_a"]).sum()))
    P("")
    P("Cross block at both onsets (correction (d)):")
    P("")
    P("| T/L | \\|\\|G_se\\|\\|_2 onset 2 L | multiplicity | onset 2.25 L | multiplicity | max \\|dG\\| 256 x 256 (onset) |")
    P("|---|---|---|---|---|---|")
    for T in (T1, T2, T3):
        t1, m1, _, _ = E.top_sv_mult(G(K1, T)[:ns, ns:]); t2, m2, _, _ = E.top_sv_mult(G(B2, T)[:ns, ns:])
        P("| %.4g | %.4g | %d | %.4g | %d | %.3g |" % (T / L, t1, m1, t2, m2, cO[T]["max_dG"]))
    P("")
    P("**Correction (e): the per-direction comparison of the boundary dependence with the step bar** (bar = %.3g |dp_dt| per ordered direction, dp_dt = 2c against the clock; ratio = |dp_onset| / bar):" % fac)
    P("")
    P("| T/L | spectrum | directions | ratio min | 10 percent | median | 90 percent | max | directions with ratio > 1 | > 10 |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    ratios = {}
    for T in (T1, T2, T3):
        for nm, ko, kd in (("supply", "dp_ss", "dp_ss"), ("exit", "dp_ee", "dp_ee"), ("joint", "dp_joint", "dp_joint")):
            do = cO[T][ko]; bar = fac * np.abs(cK[T][kd])
            rat = np.where(bar > 0, np.abs(do) / np.where(bar > 0, bar, 1.0), np.inf); ratios[(T, nm)] = (do, bar, rat)
            qq = E.q(rat)
            P("| %.4g | %s | %d | %.3g | %.3g | %.3g | %.3g | %.3g | %d | %d |" % (T / L, nm, len(do), qq[0], qq[1], qq[2], qq[3], qq[4], (rat > 1).sum(), (rat > 10).sum()))
    P("")
    P("The full per-direction listing at 1e2 L (joint, 256 ordered directions) is Appendix A.")
    P("")
    verdict["4 onset"] = "reported (boundary dependence)"

    # ================================================================== 5 reviewer A sec. 5 remainder
    P("## 7. The rest of reviewer A's sec. 5 gate (gate item 5): the clock's block-diagonal readouts")
    P("")
    import pt3b_v2lib as V2
    manK = os.path.join(a.clock, "stageA_manifest_v1.json")
    manD = os.path.join(a.decl, "stageA_manifest_v1.json")
    hp_path = os.path.join(a.decl, "hole_packet_v1.npz")
    mD = json.load(open(manD))
    P("Declarations (made before any read): %s md5 %s; the copy that travelled with the clock arm, %s, md5 %s (%s). Hole packet %s md5 %s, declared md5 %s (%s)." % (
        manD, E.md5(manD)[:8], manK, E.md5(manK)[:8] if os.path.exists(manK) else "-", "identical" if os.path.exists(manK) and E.md5(manK) == E.md5(manD) else "NOT identical or absent",
        hp_path, E.md5(hp_path)[:8], mD["hole_packet"]["md5"][:8], "match" if E.md5(hp_path) == mD["hole_packet"]["md5"] else "MISMATCH"))
    P("")
    jK = K1["j"]
    P("N_w against N^acc_2L(theta), time route (#{p > 1/2} at T_theta = ln 2 L/theta), the clock's supply and exit blocks:")
    P("")
    P("| theta | T_theta/L | supply N_w | supply N^acc_2L | exit N_w | exit N^acc_2L | status |")
    P("|---|---|---|---|---|---|---|")
    for th in (0.1, 0.01, math.log(2) / 100.0, 1e-3):
        Tt = math.log(2) * L / th
        i = E.cidx(K1["T"], Tt)
        if i is not None:
            Gk = np.asarray(K1["z"]["G"][i]); ps_, pe_ = E.eigp(Gk[:ns, :ns]), E.eigp(Gk[ns:, ns:])
            P("| %.4g | %.4g | %d | %d | %d | %d | measured |" % (th, Tt / L, jK["supply"]["N_w"], (ps_ > 0.5).sum(), jK["exit"]["N_w"], (pe_ > 0.5).sum()))
        else:
            Gk = np.asarray(K1["z"]["G"][-1]); ps_, pe_ = E.eigp(Gk[:ns, :ns]), E.eigp(Gk[ns:, ns:])
            P("| %.4g | %.4g | %d | >= %d | %d | >= %d | censored (beyond the run's %.4g L) |" % (th, Tt / L, jK["supply"]["N_w"], (ps_ > 0.5).sum(), jK["exit"]["N_w"], (pe_ > 0.5).sum(), max(K1["T"]) / L))
    P("")
    thc = jK["exit"].get("theta_chi")
    P("H2 at theta(chi): the exit window's theta(chi) = %s, so its transit time T = ln 2 L/theta(chi) = %s L lies beyond the run (1e2 L): H2 at theta(chi) is CENSORED; the hole packet's record is given at the run's clocks below." % (
        ("%.3g" % thc) if thc else "-", ("%.3g" % (math.log(2) / thc)) if thc else "-"))
    P("")
    Cs = np.load(os.path.join(a.decl, "cin_supply_filled_v1.npz"))["C"]; Ch = np.load(os.path.join(a.decl, "cin_exit_hole_v1.npz"))["C"]; Ct = np.load(os.path.join(a.decl, "cin_thermal_T0_v1.npz"))["C"]
    Gl = [(float(t), np.asarray(g_)) for t, g_ in zip(K1["T"], K1["z"]["G"])]; Rl = [np.asarray(r_) for r_ in K1["z"]["R"]]
    Gs = [(t, g_[:ns, :ns]) for t, g_ in Gl]; Rs = [r_[:ns, :ns] for r_ in Rl]
    Ge = [(t, np.conj(g_[ns:, ns:])) for t, g_ in Gl]; Re = [np.conj(r_[ns:, ns:]) for r_ in Rl]

    def table(name, Gl_, Rl_, C_):
        rd = V2.bins_reader(Gl_, C_, R_list=Rl_)
        P("**%s**" % name)
        P("")
        P("| bin end T/L | number delivered (cumulative) | in this bin | S(R_n) | S(b_n) | S(b_n \\| R_(n-1)) | S(retained) | I(retained : record) |")
        P("|---|---|---|---|---|---|---|---|")
        for b, ro in zip(rd["bins"], rd["retained_output"]):
            sb = b.get("S_b_given_R")
            P("| %.4g | %.4f | %.4g | %.4f | %.4f | %s | %.4f | %.4f |" % (b["t"] / L, b["number"], b["number_bin"], b["S_R"], b["S_b"], "-" if sb is None else "%.4g" % sb, ro["S_retained"], ro["I_retained_output"]))
        P("")
        P("Early-late information: %s." % "; ".join("I(R up to %.4g L : rest) = %.4g" % (m_["split_t"] / L, m_["I_early_late"]) for m_ in rd["early_late"]))
        P("")

    P("Bath-record quantities of the sink2L dilation, block-diagonal only (declared covariances from stageA_decl_v2; exit block read against conj(G_ee)); bins (0, 6.93 L], (6.93, 69.3 L], (69.3, 100 L]:")
    P("")
    table("supply_filled (H1), clock, supply block", Gs, Rs, Cs[:ns, :ns].astype(complex))
    table("exit_hole (H2), clock, exit block: one hole in the declared packet (conjugate convention)", Ge, Re, Ch[ns:, ns:].astype(complex))
    table("file:thermal_T0 (optional bracket), clock, supply block", Gs, Rs, Ct[:ns, :ns].astype(complex))
    table("file:thermal_T0, clock, exit block (holes)", Ge, Re, Ct[ns:, ns:].astype(complex))
    P("Cross-block coherence of the declared covariances: max |C_se| supply_filled %.2g, exit_hole %.2g, thermal_T0 %.2g (block-diagonal as declared). No joint entropy is reported: the windows are coupled through G_se (sec. 5), and a joint record of particles and holes needs the Nambu form, which is not written." % (
        np.max(np.abs(Cs[:ns, ns:])), np.max(np.abs(Ch[:ns, ns:])), np.max(np.abs(Ct[:ns, ns:]))))
    P("")
    verdict["5 reviewer A sec. 5"] = "done (H2 at theta(chi) censored)"

    # ================================================================== 6 mirror
    P("## 8. The mirror intertwiner (gate item 6)")
    P("")
    P("second_route's audit (deliveries/second_route/review_Bprime_s16A_v1.md) established the supply/exit equality as an exact antiunitary symmetry J = (tau_y x sigma_y)K of the declared one-particle sink problem, with the intertwiner C = (J F_s)^dag F_e, on run 1's saved Grams at all three of its clocks, and judged a duplicate exit spectral run unnecessary under paired windows, including the eta x 2 change. busi's recheck on the new arms, from the prep bases (component-major J psi = (-psi3*, psi2*, psi1*, -psi0*)), extends it to the cross block: with J^2 = +1, conj(G_se) = C G_se^dag conj(C).")
    P("")
    Cm, un, _ = E.mirror_C(pk)
    pb2 = os.path.join(a.prep, "F1_s16_d0_onset225_bb_prep.npz")
    Cb, unb, _ = E.mirror_C(pb2)
    P("Intertwiner unitarity: max |C^dag C - I| = %.3g (onset-2 L prep, md5 %s), %.3g (onset-2.25 L prep, md5 %s)." % (un, E.md5(pk)[:8], unb, E.md5(pb2)[:8]))
    P("")
    P("| arm | T/L | max \\|G_ee - C^dag conj(G_ss) C\\| | max \\|conj(G_se) - C G_se^dag conj(C)\\| | max \\|p_supply - p_exit\\| |")
    P("|---|---|---|---|---|")
    mir_max = 0.0
    for nm, R, Cx in (("clock", K1, Cm), ("2c", C2, Cm), ("2b", B2, Cb)):
        for i, T in enumerate(R["T"]):
            Gx = np.asarray(R["z"]["G"][i]); rb, rx = E.mirror_residuals(Gx, Cx, ns)
            mir_max = max(mir_max, rb, rx)
            P("| %s | %.4g | %.3g | %.3g | %.2g |" % (nm, T / L, rb, rx, np.max(np.abs(E.eigp(Gx[:ns, :ns]) - E.eigp(Gx[ns:, ns:])))))
    P("")
    P("Status: the intertwiner holds on every new Gram, both blocks and the cross block, to %.3g. Whether this closes the 2a exit-window run (reviewer B's condition) is the models' call; busi reports the residuals only." % mir_max)
    P("")
    verdict["6 mirror"] = "holds to %.2g" % mir_max

    # ================================================================== 7 repricing
    P("## 9. Repricing and the objects of steps 3 and 8 (gate item 7, second part)")
    P("")
    rate = mk["rate_full"]
    h1e3 = 256 * 1000 * L * rate / 3600.0
    h3e2 = 256 * 300 * L * rate / 3600.0
    assumed = 60.0 / 2.0
    P("At the clock's measured full-span rate %.4f s per column per a, the F1 arm to 1e3 L (256 columns, 16000 a) costs %.1f card-h (to 3e2 L: %.1f card-h). ReviewA_bpchk2_v1 sec. 4.3's trigger is 60 card-h (stated there as about twice the assumed price, so the assumed price was about %.0f card-h): the arm is %s the trigger, and the measured/assumed ratio is %.2f. Scaling stage B's assumed 66 card-h by that ratio gives about %.0f card-h (an estimate: stage B's items differ in columns and length)." % (
        rate, h1e3, h3e2, assumed, "OVER" if h1e3 > 60 else "under", h1e3 / assumed, 66 * h1e3 / assumed))
    P("")
    r3 = 1 - c3["p_joint_b"]
    P("The object of steps 3 and 8 (ReviewA_bpchk2_v1 sec. 5; reviewer A s16B sec. 3): at 1e2 L every joint direction has p >= %.6f; directions with 1e-3 < p < 1e-1: %d (supply %d, exit %d); directions with 1e-3 < 1 - p: %d; the onset-induced Sum(1 - p) at 1e2 L is %.3g of the onset-2 L residual (sec. 6). Whether steps 3 and 8 keep an object is for the models to decide on these counts." % (
        c3["p_joint_b"].min(), int(((c3["p_joint_b"] > 1e-3) & (c3["p_joint_b"] < 1e-1)).sum()), int(((c3["p_ss_b"] > 1e-3) & (c3["p_ss_b"] < 1e-1)).sum()),
        int(((c3["p_ee_b"] > 1e-3) & (c3["p_ee_b"] < 1e-1)).sum()), int((r3 > 1e-3).sum()), sj3 / s3b))
    P("")

    # ================================================================== summary
    P("## 10. Gate items at a glance")
    P("")
    P("| item | result |")
    P("|---|---|")
    for k_, v_ in verdict.items():
        P("| %s | %s |" % (k_, v_))
    P("| 7 manifest, repricing | sec. 1 and sec. 9 |")
    P("| v2 corrections | (a) sec. 3; (b) sec. 6; (c) sec. 4; (d) secs. 5 and 6; (e) sec. 6 and Appendix A; (f) sec. 6; (g) sec. 3 |")
    P("")
    P("Nothing prints from this file until the gate review; the models decide the gate.")
    P("")
    # ================================================================== appendix
    P("## Appendix A. Per-direction listing at 1e2 L (joint, ordered by the clock's p)")
    P("")
    P("| rank | p clock (onset 2 L, dt 0.5 a) | p 2c (dt 0.25 a) | p 2b (onset 2.25 L) | 1 - p clock | step bar %.3g \\|dp_dt\\| | dp_onset | \\|dp_onset\\| / bar |" % fac)
    P("|---|---|---|---|---|---|---|---|")
    do, bar, rat = ratios[(T3, "joint")]
    for k_ in range(K):
        P("| %d | %.9f | %.9f | %.9f | %.3g | %.3g | %.3g | %.3g |" % (k_, c3["p_joint_b"][k_], c3["p_joint_a"][k_], cO[T3]["p_joint_a"][k_], 1 - c3["p_joint_b"][k_], bar[k_], do[k_], rat[k_]))
    P("")
    files = [(R["dir"], R["tag"], R["md5_npz"]) for R in (K1, C2, B2, Xf, Xn, R1s, R1e)]
    P("Files (npz md5): %s; lane anchor %s (md5 %s); helpers s16A_end_lib_v1.py (md5 %s), s16A_end_lib_v2.py (md5 %s); this script (md5 %s)." % (
        "; ".join("%s/%s.npz (%s)" % (d, t, m[:8]) for d, t, m in files), a.lane, E.md5(a.lane)[:8], E.md5(os.path.join(HERE, "s16A_end_lib_v1.py"))[:8],
        E.md5(os.path.join(HERE, "s16A_end_lib_v2.py"))[:8], E.md5(os.path.abspath(__file__))[:8]))
    open(a.out, "w").write("\n".join(out) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
