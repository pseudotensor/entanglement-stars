#!/usr/bin/env python3
"""heavycut_twofield_sum_tables_v1.py: markdown tables from heavycut_twofield_sum_v1.json (formatting only)."""
import json, os, sys

D = sys.argv[1] if len(sys.argv) > 1 else "<internal>/HEAVYCUT_v1"
r = json.load(open(os.path.join(D, "heavycut_twofield_sum_v1.json")))
T = r["table"]


def g(x, f="%.6f"):
    return "-" if x is None else (f % x)


def e(x):
    return "-" if x is None else "%.1e" % x


out = []


def table(hh, cut, title):
    rows = [x for x in T if x["heavy_history"] == hh and x["cut"] == cut]
    out.append("\n### %s\n" % title)
    out.append("| T/r_s | heavy excess | F_H | kappa-10 term | heavy dtype; past first return | light clock | light K=18 excess | F_meas | K_c | light verdict | two-field sum | floor (F_H + F_meas) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for x in rows:
        hv, lv = x["heavy"], x["light"]
        hs = "absent" if hv is None else "%+.6f" % hv["excess"]
        out.append("| %.3f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            x["t"], hs, e(hv["F_H"]) if hv else "-", ("%+.2e" % hv["k_last_term"]) if hv else "-",
            ("%s; %s" % (hv["dtype"], "yes (%.1f)" % hv["first_return_rs"] if hv["past_first_return"] else "no (%.1f)" % hv["first_return_rs"])) if hv else "-",
            g(x["light_clock"], "%.3f") if lv else "absent", ("%+.6f" % lv["S"]) if lv else "absent",
            e(lv["F_meas"]) if lv else "-", e(lv["K_c"]) if lv else "-", (lv["verdict"] or "(no readout row)") if lv else "-",
            ("%+.6f" % x["sum"]) if x["sum"] is not None else "-", e(x.get("sum_floor"))))


table("B2", "16", "B2, 16 r_s cut (heavy: page_own_object_v1 on the B2 slices; light: PAGE v4.1; same chain)")
table("B2", "24", "B2, 24 r_s cut (same chain)")
table("prod", "2", "2 r_s cut: heavy on the prod chain (W8, 2048 sites, chain end 32 r_s), light on B2 (CROSS-CHAIN sum)")
table("prod", "16", "16 r_s cut: heavy prod (W8), light B2 (cross-chain; B2's own heavy rows above are the same-chain sum)")
table("prod", "24", "24 r_s cut: heavy prod (W8), light B2 (cross-chain; prod past its first return from 20 r_s)")
for c in ("2", "16", "24"):
    table("s8", c, "s8 history, %s r_s cut (heavy: W8 s8; light: PAGE v4.1 s8; same history by R_t)" % c)

out.append("\n### Check: prod vs B2 heavy excess at the overlap\n")
out.append("| T/r_s | cut | prod (W8) | B2 (page_own_object) | prod - B2 | rel | prod past return; B2 past return | F_H prod; B2 | E/E0 prod; B2 | R_t prod; B2 |")
out.append("|---|---|---|---|---|---|---|---|---|---|")
for x in r["check_prod_vs_B2"]:
    out.append("| %.0f | %s | %+.6e | %+.6e | %+.2e | %s | %s; %s | %s; %s | %.6f; %.6f | %s; %s |" % (
        x["t"], x["cut"], x["prod"], x["B2"], x["diff"], g(x["rel"], "%+.2e"), x["prod_past_return"], x["B2_past_return"],
        e(x["floors_prod_B2"][0]), e(x["floors_prod_B2"][1]), x["E_over_E0_prod_B2"][0], x["E_over_E0_prod_B2"][1],
        g(x["R_t_prod_B2"][0], "%.4f"), g(x["R_t_prod_B2"][1], "%.4f")))
out.append("\nW8 vs page_own_object on prod: max |diff| %.2e over %d clock-cut pairs." % (
    max(abs(x["diff"]) for x in r["check_W8_vs_page_prod"]), len(r["check_W8_vs_page_prod"])))
out.append("B2 slices' record values vs the light's record json (md5 %s): max |dE/E0| %.1e, max |dR_t| %.1e over %d clocks." % (
    r["inputs"]["<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096.json"],
    max(abs(x["dE"]) for x in r["check_B2_slices_vs_record"]),
    max(abs(x["dR_t"]) for x in r["check_B2_slices_vs_record"] if x["dR_t"] is not None), len(r["check_B2_slices_vs_record"])))
out.append("Light readout S vs the SUM array at the same pair: max |diff| %.1e." % r["light_readout_minus_SUM_max"])
s8 = r["check_s8_heavy_vs_light"]
out.append("s8 heavy (W8) vs light s8 R_t at common clocks: " + "; ".join(
    "%.0f: %s vs %s" % (x["t"], g(x["R_t_heavy"], "%.10f"), g(x["R_t_light"], "%.10f")) for x in s8) + ".")
open(os.path.join(D, "heavycut_twofield_sum_v1_tables.md"), "w").write("\n".join(out) + "\n")
print("ok", len(out))
