#!/usr/bin/env python
"""Strand 0 report script: reads partA.json, ringL.json, cube.json and writes the tables markdown (no physics reading).

Every number in solve/results/RESULT_strand0_source_allocation.md and RESULT_strand0_tables.md comes from here.
"""
import argparse
import json
import os

import numpy as np


def f(v, sig=4):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    v = float(v)
    if v == 0:
        return "0"
    a = abs(v)
    if a < 1e-3 or a >= 1e5:
        return "%.*e" % (sig - 1, v)
    return "%.*g" % (sig, v)


def fl(lst, sig=4):
    return " ".join(f(v, sig) for v in lst)


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fixed_cases(A):
    return [c for c in A["cases"] if not c["bg"].startswith("B1")]


def b1_cases(A):
    return [c for c in A["cases"] if c["bg"].startswith("B1")]


def sector_list(c):
    if c["bg"].startswith("B1"):
        return [str(c["native_sector"])]
    return sorted(c["sectors"])


def controls(A, R, C):
    cs = A["controls"]
    rows = []
    rows.append(["C1 W map ||W h_nat(1) W^dag - h_*||_F", "<= 1e-12",
                 "%s; chosen %s" % (", ".join("%s %s" % (k, f(v)) for k, v in cs["C1_residuals"].items()), cs["C1_W_choice"])])
    rows.append(["C1b Gamma(W) unitary / intertwining (N = 4, 5, 6)", "<= 1e-12",
                 "%s / %s" % (f(max(cs["C1b_unitary_N%d" % N] for N in (4, 5, 6))),
                              f(max(cs["C1b_intertwine_N%d" % N] for N in (4, 5, 6))))])
    rows.append(["C1c n_x1 n_x2 -> n_x+ n_x-", "<= 1e-12", f(max(cs["C1c_quartic_N%d" % N] for N in (4, 5, 6)))])
    cases = A["cases"]
    mx = lambda key: max(c["controls"][key] for c in cases if key in c["controls"])
    rows.append(["C2 ||A - U E0|| / ||U^dag U - 1|| / ||E1 - U E0 U^dag||", "<= 1e-12",
                 "%s / %s / %s" % (f(mx("C2_A_minus_UE0")), f(mx("C2_UdagU_minus_1")), f(mx("C2_E1_minus_UE0Udag")))])
    rows.append(["C2 min eig(A^dag A) over backgrounds", "> 0",
                 f(min(c["controls"]["C2_min_eig_AdagA"] for c in cases))])
    rows.append(["C3 eig(h_*) = +-eps / closed form (uniform) / C3b = eig(h_nat(s)) (uniform)", "<= 1e-12",
                 "%s / %s / %s" % (f(mx("C3_eig_h_vs_pm_eps")), f(mx("C3_eps_closed_form")), f(mx("C3b_eig_h_vs_native")))])
    sec = lambda key: max(s["controls"][key] for c in cases for k, s in c["sectors"].items() if key in s["controls"])
    rows.append(["C4 c-basis form = quasiparticle form of rho^P(free)", "<= 1e-12", f(sec("C4_cbasis_vs_qp"))])
    e4 = max([abs(c["controls"]["C4b_Evac_minus_minusTrB0"]) for c in cases] +
             [abs(r["C4b_Evac_minus_minusTrB0"]) for r in A["euler_fd"]])
    o4 = max([c["controls"]["C4b_overlap_minus_1"] for c in cases] + [r["C4b_overlap_minus_1"] for r in A["euler_fd"]])
    rows.append(["C4b lowest eig of c^dag h_* c in N = 4 + Tr B0 / overlap with Omega - 1 (incl. Euler N)", "<= 1e-12",
                 "%s / %s" % (f(e4), f(o4))])
    rows.append(["C5a <Omega|:V:|Omega> / <1_i|:V:|1_j>", "<= 1e-12",
                 "%s / %s" % (f(mx("C5a_vacuum")), f(mx("C5a_one_excitation")))])
    rows.append(["C5b mixed-class v_IJ", "0", cases[0]["controls"]["C5b_mixed_class"]])
    rows.append(["C5c [S_*, n_exc] / S_* P_<=1 / S_* - S_*^dag", "<= 1e-12",
                 "%s / %s / %s" % (f(sec("C5c_S_nexc_commutator")), f(sec("C5c_S_on_nexc_le1")), f(sec("C5c_S_hermitian")))])
    rows.append(["C5d sum v_IJ a_I^dag a_J - V_* (N = 4, 5, 6)", "<= 1e-12", f(sec("C5d_monomials_minus_Vstar"))])
    rows.append(["C5e H_exc - H_free^exc on n_exc <= 1 / min eig H_exc", "<= 1e-12 / >= -1e-12",
                 "%s / %s" % (f(sec("C5e_Hexc_minus_Hfree_on_nexc_le1")),
                              f(min(s["controls"]["C5e_min_eig_Hexc"] for c in cases for s in c["sectors"].values())))])
    rat = [r["C5e_ratio_U0.01_over_U0.1"] for r in A["contact_scaling"] if r["x"] == "total"]
    rows.append(["C5e ||H_exc - H_free - V_*|| ratio U = 0.01 over U = 0.1", "1/100 within 10 percent",
                 "%s to %s" % (f(min(rat)), f(max(rat)))])
    c6a = max(v for r in A["euler_fd"] for k, v in r.items() if k.startswith("C6_fd_vs_rhoP"))
    c6b = max(v for r in A["euler_fd"] for k, v in r.items() if k.startswith("C6_fd_h1e-3_vs"))
    rows.append(["C6 finite differences in N_x vs rho^P_x (h = 1e-3, 1e-4) / between the two steps", "<= 1e-9 / <= 1e-10",
                 "%s / %s" % (f(c6a), f(c6b))])
    uni = [abs(t["w"]) for c in cases if np.allclose(c["psi"], c["psi"][0]) for t in c["T5_law"]]
    nonu = [abs(c["sum_Nw"] - c["TrL_E1_minus_E0"]) for c in cases if not np.allclose(c["psi"], c["psi"][0])]
    rows.append(["C7 w_x at uniform psi / sum_x N_x w_x - Tr L(E1 - E0) (B2 to B4)", "<= 1e-12",
                 "%s / %s" % (f(max(uni)), f(max(nonu)))])
    rows.append(["C8a native scalar source = frozen-sea midpoint bilinear (s = 1 and nonuniform)", "<= 1e-12",
                 f(sec("C8a_native_identification"))])
    rows.append(["C8b Gamma(W) e^nat_x(1) Gamma(W)^dag = rho^M_x at B0, U = 0", "<= 1e-12", f(sec("C8b_rhoM_vs_mapped_native"))])
    nat = A["native"]
    ok = sum(1 for r in nat if r["ok"])
    rows.append(["C9 native re-solves ok / max residual / min eig S_x / max |E0 - stored E|", "all / <= 1e-11 / > 0 / <= 1e-10",
                 "%d of %d / %s / %s / %s" % (ok, len(nat), f(max(r["final_residual"] for r in nat if r["ok"])),
                                               f(min(min(r["S_min_eig"]) for r in nat if r["ok"])),
                                               f(max(abs(r["E_minus_stored"]) for r in nat if r["ok"] and r["E_minus_stored"] is not None)))])
    c10 = [(k, s["controls"]["C10_cons_part_minus_Vstar"], s["controls"]["C10_number_changing_norm"], s["controls"]["C10_Vstar_norm"])
           for c in cases if c["bg"] == "B0" and c["U"] == 0.1 for k, s in sorted(c["sectors"].items())
           if "C10_cons_part_minus_Vstar" in s["controls"]]
    rows.append(["C10 ||P_cons(H_nat^(m)(1) - H_0)P_cons - V_*|| at B0, U = 0.1 (max over N, m)", "<= 1e-12",
                 f(max(v[1] for v in c10))])
    rows.append(["C11 contact-1 at U = 1e-4, relative difference (max)", "<= 1e-3", f(max(r["rel_diff"] for r in A["C11"]))])
    c12 = max(max(r["C12"].values()) for r in R["records"])
    c12u = max(r["C12_U_eigen"] for r in R["records"])
    rows.append(["C12 ring-L closed forms (max over L, m, elements) / U|p> = e^{i phi_p}|p>", "<= 1e-12",
                 "%s / %s" % (f(c12), f(c12u))])
    return table(["control", "tolerance", "measured"], rows), c10


def sites_of(c):
    return c.get("report_sites", [0])


def t1(A):
    rows = []
    for c in A["cases"]:
        for N in sector_list(c):
            s = c["sectors"][N]
            for x in sites_of(c):
                st = s["sites"][str(x)]
                es = "-"
                if "eS_eig" in s:
                    es = "%s .. %s" % (f(s["eS_eig"]["min"][x]), f(s["eS_eig"]["max"][x]))
                blk = min(v for v in st["rhoP_block_min"].values() if v is not None)
                rows.append([c["bg"], f(c["m"]), f(c["U"]), "%+d" % s["q"], x,
                             "%s .. %s" % (f(st["rhoP_eig"][0]), f(st["rhoP_eig"][1])), f(blk),
                             "%s .. %s" % (f(st["contact_eig"][0]), f(st["contact_eig"][1])),
                             "%s .. %s" % (f(st["rhoM_eig"][0]), f(st["rhoM_eig"][1])),
                             "%s .. %s" % (f(st["enat_eig"][0]), f(st["enat_eig"][1])), es])
    return table(["bg", "m", "U", "q", "x", "rho^P eig", "min over n_exc blocks of rho^P", "contact part eig",
                  "rho^M eig", "e^nat(psi) eig", "e_x(S) eig (B1)"], rows)


def t1_onepart(A):
    rows = []
    for c in A["cases"]:
        if "5" not in c["sectors"] or (c["bg"].startswith("B1") and c["native_sector"] != 5):
            continue
        for x in sites_of(c):
            st = c["sectors"]["5"]["sites"][str(x)]
            ob = c["onebody"][str(x)]
            rows.append([c["bg"], f(c["m"]), f(c["U"]), x, fl(st["onepart_rhoP_eigs"]), fl(st["onepart_rhoM_eigs"]),
                         fl(ob["nat_pp_eigs"]), f(ob["X1_pp_eigs"][-1])])
    return table(["bg", "m", "U", "x", "one-particle block of rho^P_x (q = +1, (1,0))", "of rho^M_x",
                  "native M1_x[h_nat(psi)] on positive states", "psi_x <x|E0|x> (largest X1 pp eig)"], rows)


def t2(A):
    rows = []
    for c in A["cases"]:
        if "5" not in c["sectors"] or (c["bg"].startswith("B1") and c["native_sector"] != 5):
            continue
        for x in sites_of(c):
            st = c["sectors"]["5"]["sites"][str(x)]
            ob = c["onebody"][str(x)]
            w = st["witness"]
            rows.append([c["bg"], f(c["m"]), f(c["U"]), x, f(w["rhoM"]), f(w["energy"]), f(w["rhoP"]),
                         f(ob["nat_pp_eigs"][0]), f(ob["nat_witness_energy"])])
    a = table(["bg", "m", "U", "x", "min <rho^M_x> (one-particle)", "its energy <H_free^exc>", "<rho^P_x> in it",
               "native block min", "native witness energy"], rows)
    rows = []
    for c in A["cases"]:
        if c["bg"] != "B0":
            continue
        for N in sorted(c["sectors"]):
            lf = c["sectors"][N].get("lower_face")
            if lf and c["U"] in (0.0, 0.1):
                rows.append([f(c["m"]), f(c["U"]), "%+d" % c["sectors"][N]["q"], f(lf["min_eig_enat_sector"]),
                             f(lf["min_eig_rhoP_sector"])])
    b = table(["m", "U", "q", "min eig e^nat_x(1) over the sector (all x)", "min eig rho^P_x(psi = 1) (all x)"], rows)
    rows = []
    for c in fixed_cases(A):
        for N, blk in (("4", "1,1"), ("5", "2,1")):
            s = c["sectors"][N]
            for x in sites_of(c):
                r = s["sites"][str(x)]["rhoM_block_min"].get(blk)
                if r:
                    rows.append([c["bg"], f(c["m"]), f(c["U"]), "%+d" % s["q"], "(%s)" % blk, x, f(r["min"]),
                                 f(r["energy_Hfree"]), f(r["energy_Hexc"]), f(r["rhoP_in_state"]), f(r["rhoP0_in_state"])])
    cc = table(["bg", "m", "U", "q", "block", "x", "min eig rho^M_x in block", "<H_free^exc>", "<H_exc>",
                "<rho^P_x> (dressed)", "<rho^P_x(0)>"], rows)
    return a, b, cc


def t3(A):
    rows = []
    for c in A["cases"]:
        for N in sector_list(c):
            s = c["sectors"][N]
            for x in sites_of(c):
                st = s["sites"][str(x)]
                rows.append([c["bg"], f(c["m"]), f(c["U"]), "%+d" % s["q"], x,
                             "%s / %s" % (f(st["Delta_PM"][0]), f(st["Delta_PM"][1])),
                             "%s / %s" % (f(st["Delta_MN"][0]), f(st["Delta_MN"][1])),
                             "%s / %s" % (f(st["Delta_PN"][0]), f(st["Delta_PN"][1]))])
    a = table(["bg", "m", "U", "q", "x", "Delta^PM fro / spec", "Delta^MN fro / spec", "Delta^PN fro / spec"], rows)
    rows = []
    seen = set()
    for c in A["cases"]:
        key = (c["bg"], c["m"])
        if key in seen:
            continue
        seen.add(key)
        for x in sites_of(c):
            ob = c["onebody"][str(x)]
            rows.append([c["bg"], f(c["m"]), x, f(ob["pp"]), f(ob["hh"]), f(ob["anom"]), f(ob["X_anom"])])
    b = table(["bg", "m", "x", "one-body Delta pp (rel)", "hh (rel)", "anomalous (rel)", "anomalous block of X1 (abs)"], rows)
    rows = []
    seen = set()
    for c in A["cases"]:
        if "band_bottom" not in c:
            continue
        key = (c["bg"], c["m"])
        if key in seen:
            continue
        seen.add(key)
        for x in sites_of(c):
            bb = c["band_bottom"][str(x)]
            rows.append([c["bg"], f(c["m"]), x, f(bb["diff_norm"]), f(bb["rhoP_norm"])])
    cc = table(["bg", "m", "x", "||rho^M - rho^P|| on the band-bottom pair", "||rho^P|| on it"], rows)
    return a, b, cc


def t4(A):
    rows = []
    for c in A["cases"]:
        if c["U"] == 0:
            continue
        for N in sector_list(c):
            s = c["sectors"][N]
            for x in sites_of(c):
                for blk, r in s["sites"][str(x)]["T4"].items():
                    rows.append([c["bg"], f(c["m"]), f(c["U"]), "%+d" % s["q"], "(%s)" % blk, x, f(r["E"]),
                                 f(r["law_contact"]), f(r["native_contact"]), f(r["rel_diff_spec"]), f(r["rel_diff_fro"])])
    a = table(["bg", "m", "U", "q", "block", "x", "lowest E in block", "<law contact>", "<native contact>",
               "rel diff in block (spec)", "(fro)"], rows)
    rows = []
    for r in A["contact_scaling"]:
        if r["x"] == "total":
            continue
        rows.append([r["bg"], f(r["m"]), r["N"] - 4, r["x"], f(r["ratio"]), f(r["second_order"])])
    b = table(["bg", "m", "q", "x", "||contact(0.1)|| / ||contact(0.01)||", "||contact(0.1) - 10 contact(0.01)|| / 0.1^2"], rows)
    rows = []
    for c in fixed_cases(A):
        if c["U"] == 0:
            continue
        for N in sorted(c["sectors"]):
            s = c["sectors"][N]
            rows.append([c["bg"], f(c["m"]), f(c["U"]), "%+d" % s["q"], f(s["controls"]["C5e_norm_Hexc_minus_Hfree_minus_Vstar"]),
                         f(s["controls"]["C5e_spec_Hexc_minus_Hfree_minus_Vstar"])])
    cc = table(["bg", "m", "U", "q", "||H_exc - H_free^exc - V_*|| (fro)", "(spec)"], rows)
    return a, b, cc


def t5(A):
    rows = []
    for c in fixed_cases(A):
        if c["U"] != 0:
            continue
        for t in c["T5_law"]:
            rows.append([c["bg"], f(c["m"]), t["x"], f(t["sea"]), f(t["boson"]), f(t["w"])])
    a = table(["bg", "m", "x", "-psi_x <x|E0|x>", "+psi_x <x|E1|x>", "w_x"], rows)
    rows = []
    for c in A["cases"]:
        if c["U"] != 0.1 and not c["bg"].startswith("B1"):
            continue
        vac = c["sectors"].get("4", {}).get("sites", {})
        vr = max(abs(vac[str(x)]["vac_rhoP"]) for x in range(4)) if vac else None
        rows.append([c["bg"], f(c["m"]), f(c["U"]), f(c["sum_Nw"]), f(c["TrL_E1_minus_E0"]), f(vr),
                     f(c["Evac_fermion"]), f(c["Evac_bilinear_lowest"])])
    b = table(["bg", "m", "U", "sum_x N_x w_x", "Tr L(E1 - E0)", "max_x |<Omega|rho^P_x|Omega>|", "-Tr B0",
               "lowest eig c^dag h_* c (N = 4)"], rows)
    rows = []
    for blk in A["T5_native_cnumbers"]:
        for r in blk["rows"]:
            rows.append([f(blk["m"]), f(blk["U"]), r["x"], f(r["c_formula"]), f(r["c_expect"]), f(r["diff"])])
    cc = table(["m", "U", "x", "c_x^nat(1) formula", "<sea_1| unsubtracted |sea_1>", "difference"], rows)
    rows = []
    for v in A["T5_native_vacuum"]:
        rows.append([v["prom"], f(v["m"]), f(v["U"]), fl(v["sbar"][:1]), f(v["E_HnatS_q0"]), f(v["E_Hnatm_sbar_N4"]),
                     f(v["E_Hnatm0_sbar_N4"]), fl(v["vac_enat_per_site"])])
    d = table(["prom", "m", "U", "sbar (q = 0)", "lowest eig H_nat(S), q = 0", "lowest eig H_nat^(m)(sbar), N = 4",
               "same at U = 0", "<Omega_nat(sbar)|e^nat_x(sbar)|Omega_nat(sbar)>, x = 0..3"], rows)
    return a, b, cc, d


def t6(A):
    rows = []
    for c in A["cases"]:
        for N in sector_list(c):
            s = c["sectors"][N]["controls"]
            rows.append([c["bg"], f(c["m"]), f(c["U"]), "%+d" % c["sectors"][N]["q"], f(s["T6_sum_rhoP_minus_Hexc_rel"]),
                         f(s.get("T6_sum_rhoM_minus_Hfree_rel")), f(s["T6_sum_enat_minus_Hnatm_rel"])])
    a = table(["bg", "m", "U", "q", "||sum_x rho^P_x - H_exc|| rel", "||sum_x rho^M_x - H_free^exc|| rel",
               "||sum_x e^nat_x - H_nat^(m)|| rel"], rows)
    rows = []
    for r in A["euler_fd"]:
        rows.append([r["bg"], fl(r["N"]), f(r["euler_rel_N4"]), f(r["euler_rel_N5"]), f(r["euler_rel_N6"])])
    b = table(["psi", "N_x", "||sum N_x rho^P_x - H_exc[N]|| rel, q = 0", "q = +1", "q = +2"], rows)
    return a, b


def t7(A):
    rows = []
    for c in A["cases"]:
        for N in sector_list(c):
            s = c["sectors"][N]
            prot = s["E_Hexc_block_lowest"].get("1,0")
            rows.append([c["bg"], f(c["m"]), f(c["U"]), "%+d" % s["q"], f(s["E_Hexc_lowest"]),
                         s["lowest_state"]["block"], f(prot), f(min(c["eps"])) if prot is not None else "-",
                         f(s["E_Hnatm_lowest"])])
    a = table(["bg", "m", "U", "q", "lowest eig H_exc", "its (n_p,n_h) block", "lowest in (1,0)", "min eps_r",
               "lowest eig H_nat^(m)(psi)"], rows)
    rows = []
    for r in A["native"]:
        if r["ok"]:
            rows.append([r["prom"], "%+d" % r["q"], f(r["m"]), f(r["U"]), f(r["E0"]), r["deg"], f(r["sbar"][0]),
                         "%s .. %s" % (f(min(r["S_min_eig"])), f(max(r["S_max_eig"]))), f(r["E_minus_stored"])])
    b = table(["prom", "q", "m", "U", "lowest eig H_nat(S)", "deg", "sbar", "eig S_x range", "E0 - stored"], rows)
    rows = []
    seen = set()
    for c in A["cases"]:
        key = (c["bg"], c["m"])
        if key in seen:
            continue
        seen.add(key)
        rows.append([c["bg"], f(c["m"]), fl(sorted(c["eps"])), fl(sorted(c["onebody"]["0"]["nat_pos_spectrum"]))])
    cc = table(["bg", "m", "eps_r of the law (N = 1)", "positive spectrum of h_nat(psi)"], rows)
    return a, b, cc


def t8(R):
    rows = []
    for r in R["records"]:
        v0, vp = r["valleys"]["0"], r["valleys"]["pi"]
        rows.append([r["L"], f(r["m"]), f(v0["c1"], 6), f(vp["c1"], 6), f(v0["4m_c2"], 6), f(vp["4m_c2"], 6), f(v0["c3"], 6),
                     f(vp["c3"], 6), f(max(r["C12"].values())), f(r["onepart_X1"]["max"]), f(r["onepart_X1"]["E00"]),
                     r["onepart_X1"]["n_nonzero"], f(r["onepart_M1"]["min"]), f(r["onepart_M1"]["witness_energy"])])
    return table(["L", "m", "c1 (0)", "c1 (pi)", "4m c2 (0)", "4m c2 (pi)", "c3 (0)", "c3 (pi)", "C12 max",
                  "X1 one-particle max", "<0|E|0>", "X1 nonzero eigs", "M1 one-particle min", "M1 witness energy"], rows)


def t8_pairs(R):
    rows = []
    for r in R["records"]:
        if r["L"] not in (16, 1024):
            continue
        for vname in ("0", "pi"):
            d = r["valleys"][vname]["abs_L_Delta_pp"]
            keys = [k for k in ("1,-1", "1,0", "0,1", "2,-2", "3,-3", "1,1") if k in d]
            rows.append([r["L"], f(r["m"]), vname, "; ".join("(%s) %s" % (k, f(d[k])) for k in keys)])
    return table(["L", "m", "valley", "|L Delta_pp(p, p')| for (n - n_valley, n' - n_valley)"], rows)


def t9(A):
    rows = []
    for c in A["cases"]:
        b1 = c["bg"].startswith("B1") and c["m"] == 0.1 and c["U"] == 0.1 and c["native_sector"] in (5, 6)
        b2 = c["bg"] == "B2_e0.2" and c["m"] == 0.1 and c["U"] == 0.1
        if not (b1 or b2):
            continue
        Ns = [str(c["native_sector"])] if b1 else ["5", "6"]
        for N in Ns:
            s = c["sectors"][N]
            for x in sites_of(c):
                st = s["sites"][str(x)]
                lb = st.get("lowblock", {})
                rows.append([c["bg"], "%+d" % s["q"], x, s["lowest_state"]["block"],
                             f(lb.get("Delta_PM")), f(lb.get("Delta_MN")), f(lb.get("contact_diff")), f(lb.get("Delta_PN")),
                             f(st["Delta_PM"][1]), f(st["Delta_MN"][1]), f(st["contact_diff"][1]), f(st["Delta_PN"][1])])
    return table(["bg (m = 0.1, U = 0.1)", "q", "x", "lowest block", "PM (block)", "MN (block)", "contact (block)",
                  "PN (block)", "PM (sector)", "MN (sector)", "contact (sector)", "PN (sector)"], rows)


def partB(C):
    rows = []
    for r in C["records"]:
        for x in r["report_sites"]:
            s = r["sites"][str(x)]
            rows.append([f(r["m"]), r["bg"], "%d %s" % (x, tuple(s["position"])), fl([v for v in s["X1_onepart_eigs"] if abs(v) > 1e-12]),
                         f(s["M1_min"]), f(s["witness_energy"]), f(s["witness_X1"]), f(s["rel_spec_X1_minus_M1"]),
                         "%s / %s" % (f(s["trace_onepart_X1"]), f(s["trace_onepart_M1"])), f(r["sum_X1_minus_h"]),
                         f(r["sum_M1_minus_h"]), f(r["min_eig_AdagA"])])
    return table(["m", "bg", "site", "nonzero one-particle eigs of X1_x", "min one-particle eig of M1_x", "witness energy",
                  "<X1_x> in witness", "||X1 - M1|| rel (spec)", "one-particle traces X1 / M1", "||sum X1 - h_*||",
                  "||sum M1 - h_*||", "min eig A^dag A"], rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partA", required=True)
    ap.add_argument("--ringL", required=True)
    ap.add_argument("--cube", required=True)
    ap.add_argument("--md", required=True)
    args = ap.parse_args()
    A = json.load(open(args.partA))
    R = json.load(open(args.ringL))
    C = json.load(open(args.cube))
    ctab, c10 = controls(A, R, C)
    parts = ["# Strand 0 tables (generated by solve/code/strand0/strand0_report.py; numbers not retyped)", "",
             "partA.json: host %s, python %s, numpy %s, scipy %s, %.0f s (%s to %s UTC); ringL.json %.0f s; cube.json %.1f s." % (
                 A["meta"]["host"], A["meta"]["python"], A["meta"]["numpy"], A["meta"]["scipy"], A["meta"]["seconds"],
                 A["meta"]["start_utc"], A["meta"]["end_utc"], R["meta"]["seconds"], C["meta"]["seconds"]), "",
             "## Controls C1 to C12", "", ctab, "",
             "C10 per sector at B0, U = 0.1: " + "; ".join("q = %+d: %s (number-changing part %s, ||V_*|| %s)" % (int(k) - 4, f(a), f(b), f(v))
                                                       for k, a, b, v in c10), "",
             "## T1 Positivity (extreme eigenvalues; contact part = rho^P_x(U) - rho^P_x(0))", "", t1(A), "",
             "### T1 one-particle blocks (q = +1, (1,0) block; eigenvalues ascending)", "", t1_onepart(A), ""]
    a, b, cc = t2(A)
    parts += ["## T2 The witness (one-particle block, q = +1)", "", a, "", "### T2 lower face at psi = 1 (integ59's single next computation)", "", b, "",
              "### T2 extended search: the (1,1) block of q = 0 and the (2,1) block of q = +1", "", cc, ""]
    a, b, cc = t3(A)
    parts += ["## T3 Operator differences (relative to the same norm of rho^P_x)", "", a, "",
              "### T3 one-body blocks of Delta^PM in the quasiparticle basis (N-independent)", "", b, "",
              "### T3 band-bottom pair (uniform backgrounds)", "", cc, ""]
    a, b, cc = t4(A)
    parts += ["## T4 Contacts in the 2-excitation blocks (lowest state of H_exc in the block)", "", a, "",
              "### T4 scaling of the law's contact part with U", "", b, "", "### T4 totals ||H_exc - H_free^exc - V_*||", "", cc, ""]
    a, b, cc, d = t5(A)
    parts += ["## T5 Reference derivatives", "", "### Law, per site (U-independent; the U = 0 rows)", "", a, "",
              "### Law vacuum", "", b, "", "### Native c-number of the scalar source at s = 1", "", cc, "",
              "### Native vacuum residuals", "", d, ""]
    a, b = t6(A)
    parts += ["## T6 Sum rules (boundary allocation on the closed ring)", "", a, "", "### Euler background N_x = 1 + 0.3 cos(pi x/2)", "", b, ""]
    a, b, cc = t7(A)
    parts += ["## T7 Energies", "", a, "", "### Native re-solves at k = 1 (sec. 2.1; the B1 backgrounds)", "", b, "",
              "### One-particle spectra", "", cc, ""]
    parts += ["## T8 Ring-L one-body scan (psi = 1, N = 1)", "", t8(R), "", "### T8 |L Delta_pp| for listed pairs (L = 16 and 1024)", "",
              t8_pairs(R), ""]
    parts += ["## T9 Attribution summary (relative spectral norms; block = the n_exc block of the lowest state of H_exc)", "", t9(A), ""]
    parts += ["## Part B: open 2 x 2 x 2 cube, one-body (sites: corner 0 and the opposite corner)", "",
              "D_W checks: ||D_W - D_W^dag|| = %s, ||D_W^2 - (3/4) 1|| = %s." % (f(C["checks"]["DW_hermitian"]),
                                                                                  f(C["checks"]["DW_squared_minus_3_4"])), "",
              partB(C), ""]
    with open(args.md, "w") as fh:
        fh.write("\n".join(parts) + "\n")
    print("wrote %s" % args.md)


if __name__ == "__main__":
    main()
