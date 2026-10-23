#!/usr/bin/env python3
"""heavycut_check_v1_extra.py: the extra checks of the HEAVYCUT v1.1 reading (run after heavycut_check_v1.py).
 (1) full-precision comparison of my recomputed values with the delivery's json (every table row and check);
 (2) dust count beyond each cut at every B2 clock with the heavy excess per weighted particle (the 46 / 62 r_s boundary);
 (3) s8: W8's record E against the light's s8 E (the W8 E_over_E0 is 6e-4 off the light's: which E0?);
 (4) the light's lattice settings (nsites, join) against the 'same chain' label;
 (5) W8 R_abs per cut; the 48 and 30 r_s first-return estimates on the 64 r_s chain from the light side's stored d13 travel times.
Own code; no producer script imported."""
import glob, json, os, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "<internal>/heavycut_v1_1"
ROOT = "<internal>/solve_a_bulk"
logf = open(os.path.join(OUT, "heavycut_check_v1_extra.out"), "w")


def P(*a):
    s = " ".join(str(x) for x in a); print(s); logf.write(s + "\n"); logf.flush()


mine = json.load(open(os.path.join(OUT, "heavycut_check_v1.json")))
dj = json.load(open(ROOT + "/compositions/HEAVYCUT_v1/heavycut_twofield_sum_v1.json"))
L = json.load(open(ROOT + "/page_v4/out_v41/page_v4_data_v1.json"))
W = json.load(open(ROOT + "/slices144/out/heavycut/HEAVYCUT_W8_v1_all.json"))

P("## (1) full-precision comparison with the delivery json")
drows = {(r["heavy_history"], r["t"], r["cut"]): r for r in dj["table"]}
mrows = {(r["hh"], r["t"], r["cut"]): r for r in mine["table"]}
P("  delivery table rows %d (with heavy %d), mine %d" % (len(drows), sum(1 for r in drows.values() if r["heavy"]), len(mrows)))
worst = {}
nabs = 0
for k, d in drows.items():
    if d["heavy"] is None:
        nabs += 1
        assert k not in mrows, k
        continue
    m = mrows[k]
    pairs = [("heavy.excess", d["heavy"]["excess"], m["heavy"]["excess"]), ("heavy.F_H", d["heavy"]["F_H"], m["heavy"]["F_H"]),
             ("heavy.k10", d["heavy"]["k_last_term"], m["heavy"]["k10_term"]), ("heavy.S_X", d["heavy"]["S_X"], m["heavy"]["S_X"]),
             ("heavy.first_return", d["heavy"]["first_return_rs"], m["heavy"]["first_return_rs"])]
    assert d["heavy"]["past_first_return"] == (not m["heavy"]["before_first_return"]) and d["heavy"]["dtype"] == m["heavy"]["dtype"], k
    if (d["light"] is None) != (m["light"] is None):
        raise SystemExit("light presence differs at %s" % (k,))
    if d["light"] is not None:
        pairs += [("light.S", d["light"]["S"], m["light"]["S"]), ("sum", d["sum"], m["sum"])]
        for nm, a, b in (("light.F_meas", d["light"]["F_meas"], m["light"]["F_meas"]), ("light.K_c", d["light"]["K_c"], m["light"]["K_c"]), ("floor", d.get("sum_floor"), m["floor"])):
            assert (a is None) == (b is None), (k, nm, a, b)
            if a is not None:
                pairs.append((nm, a, b))
        assert d["light"]["verdict"] == m["light"]["verdict"] and d["light_clock"] == k[1], k
    for nm, a, b in pairs:
        worst[nm] = max(worst.get(nm, 0.0), abs(float(a) - float(b)))
P("  rows without heavy in the delivery (light-only prod/s8 cuts 30, 48 etc): %d" % nabs)
for nm, v in worst.items():
    P("  max |delivery - mine| %-18s %.3e" % (nm, v))
dch = {(x["t"], x["cut"]): x for x in dj["check_prod_vs_B2"]}
mch = {(x["t"], x["cut"]): x for x in mine["checks"]["prod_vs_B2"]}
P("  check table: %d pairs; max|diff of diff| %.2e; past-return flags equal: %s" % (
    len(dch), max(abs(dch[k]["diff"] - mch[k]["diff"]) for k in dch), all(dch[k]["prod_past_return"] == mch[k]["prod_past"] and dch[k]["B2_past_return"] == mch[k]["B2_past"] for k in dch)))
P("  W8-vs-page: delivery max %.4e mine %.4e; readout-SUM: delivery %.1e mine %.1e" % (
    max(abs(x["diff"]) for x in dj["check_W8_vs_page_prod"]), max(abs(x["diff"]) for x in mine["checks"]["W8_vs_page_prod"]),
    dj["light_readout_minus_SUM_max"], mine["light"]["checks"]["B2"]["readout_minus_SUM_max"]))

P("\n## (2) B2 dust count beyond the cut vs heavy excess (weighted), per clock")
P("  %6s | %12s %10s %10s | %12s %10s %10s" % ("t", "ex16", "N16", "ex/N16", "ex24", "N24", "ex/N24"))
for t in sorted(float(x) for x in mine["meta"]["B2"]):
    r16 = mrows[("B2", t, "16")]["heavy"]; r24 = mrows[("B2", t, "24")]["heavy"]
    P("  %6g | %12.6f %10.4f %10s | %12.6f %10.4f %10s" % (
        t, r16["excess"], r16["dust_N_out"], ("%.1f" % (r16["excess"] / r16["dust_N_out"])) if r16["dust_N_out"] > 1e-6 else "-",
        r24["excess"], r24["dust_N_out"], ("%.1f" % (r24["excess"] / r24["dust_N_out"])) if r24["dust_N_out"] > 1e-6 else "-"))
for c in ("16", "24"):
    ts = sorted(float(x) for x in mine["meta"]["B2"])
    first_ge1 = next(t for t in ts if mrows[("B2", t, c)]["heavy"]["dust_N_out"] >= 1.0)
    first_ge01 = next(t for t in ts if mrows[("B2", t, c)]["heavy"]["dust_N_out"] >= 0.1)
    win = [t for t in ts if mrows[("B2", t, c)]["heavy"]["dust_N_out"] < 0.1 and mrows[("B2", t, c)]["heavy"]["excess"] > 0.5]
    P("  cut %s: first clock with >= 0.1 weighted particle beyond: %g; with >= 1: %g; clocks with excess > 0.5 nats and < 0.1 particle: %s (excess %s)" % (
        c, first_ge01, first_ge1, win, ", ".join("%.3f" % mrows[("B2", t, c)]["heavy"]["excess"] for t in win)))

P("\n## (3) s8: W8 record E vs the light's s8 E")
Ls8 = L["s8"]; E0s8 = Ls8["E"][0]; E0prod = 118.1378622557944
for r in W["rows"]:
    if r["history"] != "s8":
        continue
    t = float(r["T_over_rs"])
    if t in Ls8["clocks"]:
        i = Ls8["clocks"].index(t)
        P("  t %3g: W8 record E %.10f, light E %.10f (|d| %.1e); W8 E_over_E0 %.10f = E/E0_prod %.10f (E0_prod %.6f); light E/E0_s8 %.10f (E0_s8 %.6f)" % (
            t, r["record"]["E"], Ls8["E"][i], abs(r["record"]["E"] - Ls8["E"][i]), r["record"]["E_over_E0"], r["record"]["E"] / E0prod, E0prod, Ls8["E"][i] / E0s8, E0s8))
P("  marks.s8 E0 %.10f, light E[0] %.10f; W8 s8 history_label: %s" % (L["marks"]["s8"]["E0"], E0s8, next(r["history_label"] for r in W["rows"] if r["history"] == "s8")))

P("\n## (4) the light's lattice settings vs the heavy B2 slices")
st = L["settings"]
P("  light v6o settings: nsites %s, join_rs %s, R_cut_rs %s, R_collar_rs %s, T_end_rs %s, record %s, fields %s (key %s), code %s" % (
    st["nsites"], st["join_rs"], st["R_cut_rs"], st["R_collar_rs"], st["T_end_rs"], st["record"], st["fields"], st["fields_key"], st["code"]))
P("  heavy B2 slices: nsites %s, chain_end %s r_s" % (sorted(set(m["nsites"] for m in mine["meta"]["B2"].values())), sorted(set(m["chain_end"] for m in mine["meta"]["B2"].values()))))
P("  light d13 first_return_at_100 (light chain): %s" % {c: round(v["first_return_rs"], 2) for c, v in L["d13"]["first_return_at_100"].items()})
P("  heavy B2 first_return (page_own_object, 64 r_s chain) at 100: 16: %.2f, 24: %.2f" % (mrows[("B2", 100.0, "16")]["heavy"]["first_return_rs"], mrows[("B2", 100.0, "24")]["heavy"]["first_return_rs"]))

P("\n## (5) W8 R_abs per cut; first-return estimates for the 48 and 30 r_s cuts on the 64 r_s chain from the light side's d13 travel times")
P("  W8 R_abs: %s" % sorted(set((k[2], mrows[k]["heavy"]["R_abs"]) for k in mrows if k[0] in ("prod", "s8"))))
P("  B2 page R_abs: %s" % sorted(set((k[2], mrows[k]["heavy"]["R_abs"]) for k in mrows if k[0] == "B2")))
fr = L["d13"]["first_return_at_100"]
# light chain: 8192 sites, end 128 r_s; t_out(x) = travel x -> 128, t_in(x) = 128 -> x. Segment rates between stored cuts:
cuts = [2, 16, 24, 30, 48]
P("  light-chain t_out/t_in per r_s between stored cuts: " + "; ".join("%d-%d: out %.3f in %.3f" % (
    a, b, (fr[str(a)]["t_out_rs"] - fr[str(b)]["t_out_rs"]) / (b - a), (fr[str(a)]["t_in_rs"] - fr[str(b)]["t_in_rs"]) / (b - a)) for a, b in zip(cuts[:-1], cuts[1:])))
# heavy chain (end 64): out from x to 64 = t_out(x) - t_out(64); t_out(64) unknown on the light side: estimate from the 48 -> 128 rates
out48 = fr["48"]["t_out_rs"] / (128 - 48); in48 = fr["48"]["t_in_rs"] / (128 - 48)
for x in (16, 24, 30, 48):
    est = (fr[str(x)]["t_out_rs"] - fr["48"]["t_out_rs"]) + (fr[str(x)]["t_in_rs"] - fr["48"]["t_in_rs"]) + (64 - 48) * (out48 + in48) if x < 48 else (64 - 48) * (out48 + in48)
    P("  64 r_s chain, cut %2d: round trip estimate %.1f r_s (x->48 by the light's stored segments, 48->64 at the 48->128 mean rate %.3f + %.3f per r_s)%s" % (
        x, est, out48, in48, "; page_own_object computed %.2f" % mrows[("B2", 100.0, str(x))]["heavy"]["first_return_rs"] if str(x) in ("16", "24") else ""))
P("  (the delivery's 'about 33 (48 r_s cut) and about 69 (30 r_s cut)' are labelled estimates by the 2(64 - x) + delay rule; 2(64-48) = 32, 2(64-30) = 68)")
