#!/usr/bin/env python
"""Strand 5 J4 stage 2 report (builder analyst): tables from strand5_J4_stage2_v2.py (stage2_<family>.json), the conservative-transport
cross-check strand5_J4_stage2_cons_v1.py (cons_<family>.json), the stage-1 families (lstar_<tag>.json: P14 eq. 21's prediction, the measured
A - B, k_op) and the single-packet moments k(m) at m_s (1 -+ 1e-3) (strand5_kernel_packets_v1.py, packets_C_m<m>_L256.json).
P14 eq. 7: k'(m_s) = [k(m_s (1 + d)) - k(m_s (1 - d))]/(2 m_s d), d = 1e-3; the lead's rule (sec. 10(j)(7)) k_a'(1) = 2 sum_s eta_s m_s k'(m_s)
(the ruler rescales every lattice mass by psi^2) with its leading-log value 4 k_a(0); P14 eq. 10's definition k_a(P) = P^-4 k(m_s P^2) gives
k_a'(1) = -4 k_a(0) + 2 sum_s eta_s m_s k'(m_s) (leading-log value 0); k_a(0) = sum_s eta_s k(m_s) with k(m_s) the mean of the two moments
(O(d^2)); the piece -(k_a'(1)/(4 pi k_op)) I beside the measured A - B, I the stage-2 field integral.  Numbers emitted, not retyped.
(Version 2: several stage-2, cross-check and k-slope folders (comma-separated); P14 eq. 10's definition is the primary (the lead's ruling,
spec sec. 10(k)) and the mass-rescaling-only value is labelled beside it; each mass pair's slope from the longest ring available (the light
packet at m = 0.01 from L = 1024, whose L = 256 wrap enters P14's small k_a'(1) at its own size); family labels carry the stage-1 folder.)
(Version 3: S3 shows both step pairs of strand5_J4_stage2_v6/v7 (the spec's eps = 1e-3, 5e-4 and eps = 1e-2, 5e-3), their counts of negative
eigenvalues, the agreement flag, and the support edge r_cut beside r_s.)
(Version 4, for strand5_J4_stage2_v9.py, the exact second variation (spec sec. 10(m)(5')): S3a the basis (c) eigenvalues at every member
(the three lowest, the count of negative ones, the directions kept from the volume Gram matrix), beside the exact basis (a) values and the
finite-difference basis (a) values of batch 6 where those exist, with the gradient check (dM/dn_c against mu_inf 4 pi r^2 on the matter
nodes); S3b the full matter-node inertia and lowest three eigenvalues at the five members per family with the finite-difference-of-gradient
check, and the inertia's change across the fitted maximum (the T7 statement); S5 (sec. 10(m)(6)) kappa_max and the EOS series truncation
with the inertia at the three most compact members.)
(Version 5: S6 the sec. 10(o) Gauss column from strand5_J4_symgauss_v1.py: per kernel family the stencil's pre-symmetrization asymmetry, the
moment's change, and over the members the largest |(A_vol - A)/A| and |(B_vol - B)/B| before (the stage-1 record) and after (the member
re-solved with the symmetrized kernel), with the largest relative change of A.)
(Version 6: the title line names this file; version 5's printed version 4's name.)"""
import argparse
import glob
import json
import os

import numpy as np


def f(v, sig=4):
    if v is None:
        return "-"
    v = float(v)
    if not np.isfinite(v):
        return str(v)
    if v == 0:
        return "0"
    return ("%.*e" % (sig - 1, v)) if (abs(v) < 1e-3 or abs(v) >= 1e5) else ("%.*g" % (sig, v))


def table(head, rows):
    esc = lambda c: str(c).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(esc(c) for c in head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows])


POINTS = [(0.1, 1.0), (0.1, 0.5), (0.03, 0.3), (0.01, 0.1)]
D = 1e-3


def family_label(meta):
    return "(%g, %g) %s (%g, %g)" % (meta["mlight"], meta["aH"], meta["operator"], meta["couplings"][0], meta["couplings"][1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2", required=True, help="folder with stage2_*.json")
    ap.add_argument("--cons", default=None, help="folder with cons_*.json")
    ap.add_argument("--kprime", required=True, help="folder with packets_C_m*_L256.json")
    ap.add_argument("--fd", default=None, help="folders with batch 6's finite-difference stage-2 files (basis (a))")
    ap.add_argument("--sym", default=None, help="folders with symgauss_*.json (sec. 10(o))")
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    FD = {}
    for folder in (a.fd.split(",") if a.fd else []):
        for p in sorted(glob.glob(os.path.join(folder, "stage2_lstar_*.json"))):
            R6 = json.load(open(p))
            for x in R6["members"]:
                h = x.get("hessian")
                if h and h.get("eigenvalues"):
                    FD[(R6["meta"]["family"], round(x["z_c"], 10))] = h
    S2 = {}
    for folder in a.stage2.split(","):
        for p in sorted(glob.glob(os.path.join(folder, "stage2_lstar_*.json"))):
            R = json.load(open(p))
            fam = json.load(open(R["meta"]["family"]))
            S2[os.path.basename(os.path.dirname(R["meta"]["family"])) + "/" + os.path.basename(p)] = (R, fam)
    # single-packet moments
    km = {}
    for folder in a.kprime.split(","):
        for p in glob.glob(os.path.join(folder, "packets_C_m*_L*.json")):
            P = json.load(open(p))
            km[(P["m"], P["L"])] = P["analysis"]["k_moment"]
    ring = {}
    def kprime(m):
        best = None
        for L in sorted({L for (_, L) in km}, reverse=True):
            ms = [x for (x, LL) in km if LL == L]
            mp = [x for x in ms if abs(x / (m * (1 + D)) - 1) < 1e-12]
            mm = [x for x in ms if abs(x / (m * (1 - D)) - 1) < 1e-12]
            if mp and mm:
                best = (L, mp[0], mm[0])
                break
        assert best is not None, m
        L, mp, mm = best
        ring[m] = L
        return (km[(mp, L)] - km[(mm, L)]) / (mp - mm), 0.5 * (km[(mp, L)] + km[(mm, L)])
    kap = {}
    rows = []
    for (m, H) in POINTS:
        ms = [m, np.sqrt(m * m + H * H), np.sqrt(m * m + H * H), np.sqrt(m * m + 2 * H * H)]
        etas = [1, -1, -1, 1]
        kp = [kprime(x) for x in ms]
        ka0 = sum(e * k[1] for e, k in zip(etas, kp))
        lead = 2 * sum(e * x * k[0] for e, x, k in zip(etas, ms, kp))
        p14 = -4 * ka0 + lead
        kap[(m, H)] = {"lead": lead, "p14": p14, "ka0": ka0}
        rows.append(["%g, %g" % (m, H), " ".join(f(k[0], 6) for k in kp[:1] + kp[1:2] + kp[3:]), " ".join(str(ring[x]) for x in (ms[0], ms[1], ms[3])),
                     f(ka0, 10), f(p14, 4), f(p14 / ka0, 4), f(lead, 8), f(4 * ka0, 8), f(lead / (4 * ka0), 6)])
    out = ["# Strand 5 J4 stage 2 tables (generated by solve/code/strand5/strand5_J4_stage2_report_v6.py)", ""]
    summ = []
    out += ["## P1 P14 eq. 7: the single-packet slopes k'(m_s) (light, heavy, heaviest; central differences at m_s (1 -+ 1e-3) on the ring "
            "length given), k_a(0) = sum eta_s k(m_s) from the same moments, k_a'(1) by P14 eq. 10's definition k_a(P) = P^-4 k(m_s P^2) (the "
            "primary: -4 k_a(0) + 2 sum eta_s m_s k'(m_s)), and beside it, labelled, the mass-rescaling-only value 2 sum eta_s m_s k'(m_s) of sec. "
            "10(j) (superseded by sec. 10(k)) with its leading-log value 4 k_a(0)", "",
            table(["(m a, aH)", "k'(m_s)", "rings L", "k_a(0) (moments)", "k_a'(1), P14 eq. 10 (primary)", "P14's / k_a(0)",
                   "mass rescaling only", "4 k_a(0)", "ratio"], rows), ""]
    # per family summary
    rows = []
    for name, (R, fam) in sorted(S2.items()):
        M = R["meta"]
        mem = R["members"]
        if not mem:
            continue
        L = [x["lock"] for x in mem]
        cl = lambda k: max(abs(x[k]) for x in L)
        rel = max(abs(x["closure_total_with_mu_dNB"] / x["AmB_over_G"]) for x in L if x["AmB_over_G"] != 0)
        hm = [x for x in mem if "hessian_c_exact" in x]
        rows.append([name.split("/")[0] + " " + family_label(M), "%d of %d" % (len(mem), len([d for d in fam["members"] if d.get("newton_ok")])), f(cl("closure_total_with_mu_dNB"), 2), f(rel, 2),
                     f(cl("closure_Rstat_with_mu_dNB"), 2), f(cl("closure_Sinf"), 2), f(cl("closure_Sinf_direct"), 2), f(max(abs(x["dNB"] / x["N_B_base"]) for x in L), 2),
                     f(cl("rows_dot_zprime"), 2), "%d (%s)" % (len(hm), "".join(str(x["hessian_c_exact"]["n_negative"]) for x in hm)), f(M.get("seconds"), 4)])
    out += ["## S1 Per family: members done, the largest closures over the members (Total - mu_inf dN_B/dlam + (A - B)/G absolute and relative to "
            "(A - B)/G; R_stat - 2 pi k B - rows.z' - mu_inf dN_B/dlam; S_inf + 2 pi k A from Total - R_stat and directly), the largest relative "
            "transport change |dN_B/dlam|/N_B, the largest |rows.z'|, the Hessian members done with their counts of negative eigenvalues", "",
            table(["family", "members", "closure T", "closure T/(A-B)/G", "closure R", "closure S", "closure S direct", "|dN_B/dlam|/N_B", "|rows.z'|",
                   "exact Hessian members (basis (c) negatives, member by member)", "seconds"], rows), ""]
    # per member, kernel families
    for name, (R, fam) in sorted(S2.items()):
        M = R["meta"]
        if M["operator"] != "kernel" or not R["members"]:
            continue
        if M["mlight"] == 0.01 and not name.startswith("J4run4/"):
            continue
        fm = {round(d["z_c"], 10): d for d in fam["members"]}
        kp = kap[(M["mlight"], M["aH"])]
        rows = []
        for x in R["members"]:
            L = x["lock"]
            d = fm[round(x["z_c"], 10)]
            Ivac = L["P14_eq7_field_integral"]
            Vl = -kp["lead"] * Ivac / (4 * np.pi * M["k_op"])
            Vp = -kp["p14"] * Ivac / (4 * np.pi * M["k_op"])
            rows.append([f(x["z_c"], 4), f(L["AmB_over_G"], 6), f(L["Total"], 6), f(L["mu_dNB"], 3), f(L["closure_total_with_mu_dNB"], 2),
                         f(L["share_matter_fixed_NB"], 5), f(L["share_vac"], 5), f(L["matter_part_of_AmB_over_A"], 4), f(d["P14_TF_prediction_AmB_over_A"], 4),
                         f(L["vac_part_of_AmB_over_A"], 4), f(x["A"] - x["B"], 6), f(Vp, 4), f(Vl, 4)])
        out += ["## S2 %s: per member -(A - B)/G's negative (A - B)/G, Total, mu_inf dN_B/dlam, the closure, the shares of -(A - B)/G (matter at "
                "fixed N_B; vacuum D_vac + S_inf), the matter part of (A - B)/A beside P14 eq. 21's free-TF prediction, the vacuum part, the "
                "measured A - B beside P14 eq. 7's piece with P14 eq. 10's k_a'(1) (primary) and with the mass-rescaling-only value (labelled)" % name, "",
                table(["z_c", "(A - B)/G", "Total", "mu dN_B/dlam", "closure", "matter share", "vacuum share", "matter part of (A-B)/A",
                       "P14 eq. 21", "vacuum part of (A-B)/A", "A - B", "eq. 7, P14's k_a' (primary)", "eq. 7, mass rescaling only"], rows), ""]
    # Hessian (exact)
    rows, rows_b, rows5 = [], [], []
    for name, (R, fam) in sorted(S2.items()):
        M = R["meta"]
        imax = M.get("index_of_max")
        lab = name.split("/")[0] + " " + family_label(M)
        gmed, gmax = [], []
        for i, x in enumerate(R["members"]):
            hc, ha = x.get("hessian_c_exact"), x.get("hessian_a_exact")
            if not hc:
                continue
            place = "max" if i == imax else ("+%d" % (i - imax) if i > imax else "%d" % (i - imax))
            fd = FD.get((M["family"], round(x["z_c"], 10)))
            g = x["exact"]["gradient_over_mu_volume"]
            gmed.append(g["median_abs_minus_1"])
            gmax.append(g["max_abs_minus_1_excluding_last_node"] if g["max_abs_minus_1_excluding_last_node"] is not None else np.nan)
            rd = None
            if fd and ha:
                rd = float(np.max(np.abs(np.array(fd["eigenvalues"][:3]) - np.array(ha["eigenvalues"][:3])) / np.maximum(np.abs(np.array(ha["eigenvalues"][:3])), 1e-300)))
            rows.append([lab, f(x["z_c"], 4), place, " ".join(f(v, 4) for v in hc["eigenvalues"][:3]), hc["n_negative"], "%d of 12" % hc["kept"],
                         " ".join(f(v, 4) for v in ha["eigenvalues"][:3]), ha["n_negative"], " ".join(f(v, 4) for v in fd["eigenvalues"][:3]) if fd else "-",
                         f(rd, 2), f(g["median_abs_minus_1"], 2), f(g["max_abs_minus_1_excluding_last_node"], 2), f(g["last_node"], 2)])
            fn = x.get("full_node")
            if fn:
                fc = x.get("fd_gradient_check_a", {})
                rows_b.append([lab, f(x["z_c"], 4), place, fn["matter_nodes"], fn["inertia"], " ".join(f(v, 4) for v in fn["lowest3"]), hc["n_negative"],
                               f(fc.get("max_rel_diff_from_exact"), 2), f(fc.get("asymmetry"), 2)])
        fns = [(i, x) for i, x in enumerate(R["members"]) if x.get("full_node")]
        br = M.get("bracketing")
        if br and fns:
            bi = {i: x["full_node"]["inertia"] for i, x in fns}
            last = fns[-1]
            summ.append("S3 (%s): full-node inertia %s at the members %s; across the fitted maximum (z_c %s, members %s and %s) it goes %s to %s; "
                        "the most compact member (z_c %s) has %s; basis (c) negatives member by member %s; the gradient's median |ratio - 1| %s." % (
                            lab, " ".join(str(x["full_node"]["inertia"]) for _, x in fns), " ".join(f(x["z_c"], 4) for _, x in fns), f(M.get("z_c_fitted_max"), 5),
                            f(R["members"][br[0]]["z_c"], 4) if br[0] < len(R["members"]) else "-", f(R["members"][br[1]]["z_c"], 4) if br[1] < len(R["members"]) else "-",
                            bi.get(br[0], "-"), bi.get(br[1], "-"), f(last[1]["z_c"], 4), last[1]["full_node"]["inertia"],
                            "".join(str(x["hessian_c_exact"]["n_negative"]) for x in R["members"] if x.get("hessian_c_exact")), f(max(gmed) if gmed else None, 2)))
        # item (6): the three most compact members
        for x in R["members"][-3:]:
            es = x.get("eos_series", {})
            rows5.append([lab, f(x["z_c"], 4), f(es.get("kappa_max"), 5), es.get("terms_used", "-"), f(es.get("tail_60_119_rel"), 2), f(es.get("term_119_rel"), 2),
                          x.get("hessian_c_exact", {}).get("n_negative", "-"), x["full_node"]["inertia"] if x.get("full_node") else "-"])
    out += ["## S3a The fixed-N_B radial Hessian of M = 4 pi k A by the exact second variation at every member (the member's place relative to the "
            "maximum in the third column): basis (c) (13 density-weighted Chebyshev modes on r < r_s projected onto fixed N_B; the generalized "
            "eigenproblem against the volume Gram matrix, directions below 1e-10 of its largest eigenvalue dropped): the three lowest eigenvalues, "
            "the negative count, the directions kept; the exact basis (a) values (version 7's modes on r < r_cut) beside batch 6's "
            "finite-difference (a) values where they exist, their largest relative difference over the lowest three; the gradient dM/dn_c over "
            "mu_inf 4 pi r^2 on the matter nodes (median and largest |ratio - 1| without the last node, and the last node's)", "",
            table(["family", "z_c", "member", "basis (c) lowest three", "negative", "kept", "basis (a) exact lowest three", "negative",
                   "basis (a) finite differences (batch 6)", "FD/exact largest rel. diff.", "gradient median", "gradient max", "last node"], rows), "",
            "## S3b The full matter-node basis (every node with n_c > 0, the fixed-N_B projection in the volume metric) at five members per family "
            "(the dilute, the mid, the two bracketing the fitted maximum, the most compact): the matter nodes, the inertia, the three lowest "
            "eigenvalues, the basis (c) negative count beside it; the basis (a) Hessian by central differences of the exact gradient against "
            "the exact one (largest relative entry difference) and its asymmetry", "",
            table(["family", "z_c", "member", "matter nodes", "inertia", "lowest three", "basis (c) negatives", "FD-of-gradient vs exact", "FD asymmetry"], rows_b), "",
            "## S5 Sec. 10(m)(6): at the three most compact members of each family kappa_max (the largest nodal kappa = psi^2 q_F), the EOS angular "
            "series' terms used there, the tail sum_{N=60}^{119} c_N kappa_max^2N relative to the 60-term A(kappa_max) and the last extended term's "
            "relative size; the basis (c) negative count and the full-node inertia (at the most compact member)", "",
            table(["family", "z_c", "kappa_max", "terms used", "tail 60-119 (relative)", "term 119 (relative)", "basis (c) negatives", "full-node inertia"], rows5), ""]
    # conservative cross-check
    if a.cons:
        rows = []
        for p in sorted(sum([glob.glob(os.path.join(c, "cons_lstar_*.json")) for c in a.cons.split(",")], [])):
            C = json.load(open(p))
            fam = json.load(open(C["meta"]["family"]))
            for x in C["members"]:
                rows.append([os.path.basename(os.path.dirname(C["meta"]["family"])) + "/" + family_label(fam["meta"]), f(x["z_c"], 4), f(x["dNB"], 2), f(x["closure_total"], 2), f(x["D_mat"], 8), f(x["multiplier_D_mat_fixed_NB"], 8),
                             f(x["diff_D_mat"], 2), f(x["share_matter"], 6), f(x["multiplier_share_matter"], 6), f(x["diff_share_matter"], 2),
                             f(x["diff_vac_piece"], 2), f(x["min_node_increment"], 2)])
        out += ["## S4 The conservative transport (cumulative N(< r) moved, N_B conserved) against the multiplier route at the dilute, maximum and "
                "most compact members: dN_B/dlam (zero by construction), the closure Total + (A - B)/G, D_mat (conservative) against D_mat - mu_inf "
                "dN_B/dlam (multiplier), their difference, the matter shares and their difference, the difference of the vacuum pieces, the "
                "smallest node increment of the transported N(< r)", "",
                table(["family", "z_c", "dN_B/dlam", "closure", "D_mat (cons.)", "D_mat - mu dN_B (mult.)", "difference", "matter share (cons.)",
                       "matter share (mult.)", "difference", "vac piece difference", "min increment"], rows), ""]
    out.append("P14's text (the digest web40_review/NOTES_P01-P20.md, P14 = FRESH1184): \"(A - B)_vac = -(1/4 pi k_inf) Int psi k_a'(psi) grad psi . "
               "grad chi (eq. 7)\" and \"k_a(P) = 3 P^-4 sum eta Int cos^2(ap_1)/sqrt(sum sin^2/a^2 + x_s P^4) (eq. 10), k_a(P) = k_{aP^2}(1)\".")
    if a.sym:
        rows = []
        for p in sorted(sum([glob.glob(os.path.join(fo, "symgauss_*.json")) for fo in a.sym.split(",")], [])):
            G = json.load(open(p))
            m = G["meta"]
            mem = G["members"]
            if not mem:
                continue
            fam = json.load(open(m["family"]))["meta"]
            lab = os.path.basename(os.path.dirname(m["family"])) + " " + family_label(fam)
            gb = max(max(abs(x["gauss_A_before"]), abs(x["gauss_B_before"])) for x in mem)
            ga = max(max(abs(x["gauss_A_after"]), abs(x["gauss_B_after"])) for x in mem)
            da = max(abs(x["dA_rel"]) for x in mem)
            rows.append([lab, m["R"], f(m["asymmetry_max"], 2), f(m["asymmetry_rel_K0"], 2), f(m["k_op_change_rel"], 2), "%d of %d" % (sum(1 for x in mem if x["newton_ok"]), len(mem)),
                         f(gb, 2), f(ga, 2), f(da, 2)])
            summ.append("S6 (%s): stencil asymmetry %s before symmetrization; the largest Gauss discrepancy %s before and %s after (members re-solved: "
                        "largest |dA/A| %s)." % (lab, f(m["asymmetry_max"], 2), f(gb, 2), f(ga, 2), f(da, 2)))
        out += ["## S6 Sec. 10(o): the line kernels symmetrized at the build (K_n <- (K_n + K_-n)/2, the zero sum re-set, k_op the symmetrized "
                "moment): per kernel family the stencil half-width, the pre-symmetrization asymmetry max |K_n - K_-n| (absolute and relative to "
                "|K_0|), the moment's change, the members re-solved, and over the members the largest |(A_vol - A)/A|, |(B_vol - B)/B| before (the "
                "stage-1 record) and after (re-solved with the symmetrized kernel), and the largest |dA/A| of the re-solve", "",
                table(["family", "R", "asymmetry", "relative to |K_0|", "k_op change", "re-solved", "Gauss before", "Gauss after", "max |dA/A|"], rows), ""]
    out = out[:2] + ["## S Summary (emitted)", ""] + ["- " + x for x in summ] + [""] + out[2:]
    open(a.md, "w").write("\n".join(out) + "\n")
    print("\n".join("- " + x for x in summ))
    print("wrote", a.md)


if __name__ == "__main__":
    main()
