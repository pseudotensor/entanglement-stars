#!/usr/bin/env python
"""engine_single_object_numbers_v1.py (builder analyst; engine spec 38.4 (3): "FILE 4, the single-object numbers template, filled by
engine_single_object_numbers_v1 (new) from the HORIZON summary json; it prints only with the icloud lane's second route"; FILE 4 =
ITEM7_single_object_numbers_v1_template.md, "every bracket a value copied from it, two significant figures unless the json's own rounding
is coarser"): ITEM7_single_object_numbers_v1.md, the template's quoted paragraph with every bracket filled and its condition paragraph
verbatim, under a head that says it prints only with the icloud lane's second route. Sources: HORIZON_single_object_v1_summary.json (B's
columns) and, for A's items, the two A files that summary records among its inputs (A_single_object_v2_summary.json: the rows and the worst
residuals; A_rs64_sigma4.json: the 4-site slab blocks), each md5 checked against the summary's own record before use (the summary itself
holds no A item). The selection rule of every bracket is stated in RULES below and printed in the log with the full-precision value and the
alternatives the template's words leave open (the other kappa, the other sign of ak_0, the other depth); the paragraph's own words fix the
choices: "per radial channel kappa 1" (kappa 1 throughout), "ak_0 pi/2" and "ak_0 2 pi/64" (the positive signs), "23 r_s after launch"
(the annulus column, whose arrival that is; the 2 sigma column arrives at 26), "the wavelength-r_s columns at most" (every wavelength-r_s
column, both signs and depths). Formats: two significant figures (the slab blocks, residuals and currents in exponent form, fractions and
ratios as decimals); the ratios at d 8 are the script's division of the two stored values. Asserted: no bracket left in the paragraph; none
of sec. 4's replaced words (frozen, freez, window, prediction) in the composed file. Refuses to overwrite. Args: --template --horizon
--a-summary --a-sigma4 --out --log."""
import argparse
import hashlib
import json
import math
import os
import re
import sys

REPLACED = ("frozen", "freez", "window", "prediction")
P2, PL = math.pi / 2, 2 * math.pi / 64


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def sig2(x):
    """two significant figures as a decimal, the trailing zero kept (0.99697 -> 1.0, 0.8742 -> 0.87, 4.726 -> 4.7, 0.0072448 -> 0.0072)."""
    r = float("%.2g" % x)
    d = math.floor(math.log10(abs(r))) if r != 0 else 0
    return "%.*f" % (max(0, 1 - d), r)


def e2(x):
    return "%.1e" % x


def e2s(x):
    return "%+.1e" % x


def main():
    ap = argparse.ArgumentParser()
    for k in ("template", "horizon", "a_summary", "a_sigma4", "out", "log"):
        ap.add_argument("--" + k.replace("_", "-"), required=True)
    a = ap.parse_args()
    for p in (a.out, a.log):
        if os.path.exists(p):
            sys.exit("refusing to overwrite %s" % p)
    H = json.load(open(a.horizon))
    rec = H["inputs"]
    for p in (a.a_summary, a.a_sigma4):
        key = [k for k in rec if k.endswith(os.path.basename(p))]
        if len(key) != 1 or rec[key[0]] != md5(p):
            sys.exit("%s is not the file the HORIZON summary records (%s against %s)" % (p, md5(p), rec.get(key[0]) if key else None))
    A = json.load(open(a.a_summary)); S4 = json.load(open(a.a_sigma4))
    B = H["B_columns"]

    def cols(run, mode, T, kap, depth=None, ak=None, sign=None):
        out = []
        for c in B:
            if not (c["run"].startswith(run) and c["mode"] == mode and c["T_launch_over_rs"] == T and c["kappa"] == kap):
                continue
            if depth is not None and c["depth"] != depth:
                continue
            if ak is not None and abs(abs(c["ak0"]) - ak) > 1e-6:
                continue
            if sign is not None and (c["ak0"] > 0) != (sign > 0):
                continue
            out.append(c)
        return out

    def one(lst):
        if len(lst) != 1:
            sys.exit("a selection holds %d columns, not 1" % len(lst))
        return lst[0]

    def arow(T, kap=1):
        r = [x for x in A["rows"] if x["family"] == "rs64_sigma4" and x["kappa"] == kap and x["root"] == "dense" and x["T_over_rs"] == T]
        return one(r)

    def slab(T, d, which, kap=1):
        rr = one([x for x in S4["records"] if x["T_over_rs"] == T])
        ch = one([c for c in rr["channels"] if c["kappa"] == kap])
        return ch["splits"]["dense"]["slabs"]["P07_%d" % d][which]["sigma_max"]

    V, RULE, ALT = {}, {}, {}
    for T in (15.0, 30.0):
        tag = "%d" % T
        for d in (2, 4, 8):
            V["C_%s_d%d" % (tag, d)] = (slab(T, d, "C"), e2)
            V["V_%s_d%d" % (tag, d)] = (slab(T, d, "P_minus"), e2)
            RULE["C_%s_d%d" % (tag, d)] = "A_rs64_sigma4.json, record T/r_s %g, kappa 1, the dense split's slab P07_%d: C.sigma_max (the state)" % (T, d)
            RULE["V_%s_d%d" % (tag, d)] = "the same slab's P_minus.sigma_max (the Killing vacuum of the same held fields; P, the continued reference, %s)" % e2(slab(T, d, "P"))
            ALT["C_%s_d%d" % (tag, d)] = "kappa 2: %s" % e2(slab(T, d, "C", 2))
            ALT["V_%s_d%d" % (tag, d)] = "kappa 2: %s" % e2(slab(T, d, "P_minus", 2))
        V["ratio_%s_d8" % tag] = (slab(T, 8, "C") / slab(T, 8, "P_minus"), sig2)
        RULE["ratio_%s_d8" % tag] = "the script's division of the two stored d 8 values at T/r_s %g (C over P_minus, kappa 1)" % T
        ALT["ratio_%s_d8" % tag] = "kappa 2: %s" % sig2(slab(T, 8, "C", 2) / slab(T, 8, "P_minus", 2))
    V["SO1_max"] = (A["worst_SO1_residual"], e2); RULE["SO1_max"] = "A_single_object_v2_summary.json's worst_SO1_residual over its %d rows (3 families; a bound on the sigma/a 4 kappa 1 rows)" % len(A["rows"])
    V["SX_max"] = (A["worst_S_X_agreement"], e2); RULE["SX_max"] = "its worst_S_X_agreement over the same rows"
    fr = cols("sigma/a 4", "frozen", 15.0, 1)
    V["unacc_max"] = (max(c["positive_control_unaccounted"] for c in fr), e2)
    RULE["unacc_max"] = "HORIZON's B columns, sigma/a 4 (production), held fixed at T/r_s 15, kappa 1: the largest positive_control_unaccounted over its %d columns" % len(fr)
    ALT["unacc_max"] = "kappa 2 at T/r_s 15: %s; over every fixed-field launch of both histories and both kappa (38.4's facts): %s" % (
        e2(max(c["positive_control_unaccounted"] for c in cols("sigma/a 4", "frozen", 15.0, 2))), e2(max(c["positive_control_unaccounted"] for c in B if c["mode"] == "frozen")))
    c_pi2 = one(cols("sigma/a 4", "frozen", 15.0, 1, "annulus middle", P2, +1))
    V["frozen_pi2_beyond16"] = (c_pi2["beyond16_final_before_reflection"], sig2)
    RULE["frozen_pi2_beyond16"] = "the same launch's column at ak_0 +pi/2, the annulus middle (arrival %g r_s after launch, the template's 23): beyond16_final_before_reflection" % c_pi2["arrival_after_launch_rs"]
    o = one(cols("sigma/a 4", "frozen", 15.0, 1, "2 sigma inside R_t", P2, +1))
    ALT["frozen_pi2_beyond16"] = "2 sigma inside R_t: %s (arrival %g); ak_0 -pi/2 annulus: %s" % (sig2(o["beyond16_final_before_reflection"]), o["arrival_after_launch_rs"],
                                                                                                   sig2(one(cols("sigma/a 4", "frozen", 15.0, 1, "annulus middle", P2, -1))["beyond16_final_before_reflection"]))
    c_l = one(cols("sigma/a 4", "frozen", 15.0, 1, "annulus middle", PL, +1))
    if not c_l["omega_bar"] < 0.2:
        sys.exit("the wavelength-r_s column's mean Killing energy is not below m")
    V["frozen_long_bound"] = (c_l["bound_overlap_end"], sig2)
    V["frozen_long_beyond16"] = (c_l["beyond16_final_before_reflection"], sig2)
    RULE["frozen_long_bound"] = "the same launch's column at ak_0 +2pi/64, the annulus middle (omega_bar %.4f, below m 0.2): bound_overlap_end (at launch %.6f)" % (c_l["omega_bar"], c_l["bound_overlap_launch"])
    RULE["frozen_long_beyond16"] = "that column's beyond16_final_before_reflection"
    o2 = one(cols("sigma/a 4", "frozen", 15.0, 1, "2 sigma inside R_t", PL, +1))
    om = [c for c in cols("sigma/a 4", "frozen", 15.0, 1, None, PL, -1)]
    ALT["frozen_long_bound"] = "2 sigma inside R_t: %s; ak_0 -2pi/64 (annulus, 2 sigma): %s" % (sig2(o2["bound_overlap_end"]), ", ".join(sig2(c["bound_overlap_end"]) for c in om))
    ALT["frozen_long_beyond16"] = "2 sigma inside R_t: %s; ak_0 -2pi/64: %s" % (sig2(o2["beyond16_final_before_reflection"]), ", ".join(sig2(c["beyond16_final_before_reflection"]) for c in om))
    h_a = one(cols("sigma/a 4", "history", 15.0, 1, "annulus middle", P2, +1)); h_2 = one(cols("sigma/a 4", "history", 15.0, 1, "2 sigma inside R_t", P2, +1))
    V["hist_annulus_pi2_collar"] = (h_a["end_collar_16_24"], sig2); V["hist_2sig_pi2_collar"] = (h_2["end_collar_16_24"], sig2)
    RULE["hist_annulus_pi2_collar"] = "B's columns on the solved history from T/r_s 15, kappa 1, ak_0 +pi/2, the annulus middle: end_collar_16_24"
    RULE["hist_2sig_pi2_collar"] = "the same at 2 sigma inside R_t"
    ALT["hist_annulus_pi2_collar"] = "kappa 2: %s; ak_0 -pi/4 (a lattice-band column, 38.4's facts' lower end): %s" % (
        sig2(one(cols("sigma/a 4", "history", 15.0, 2, "annulus middle", P2, +1))["end_collar_16_24"]),
        sig2(one(cols("sigma/a 4", "history", 15.0, 1, "annulus middle", math.pi / 4, -1))["end_collar_16_24"]))
    ALT["hist_2sig_pi2_collar"] = "kappa 2: %s; ak_0 -pi/4: %s" % (sig2(one(cols("sigma/a 4", "history", 15.0, 2, "2 sigma inside R_t", P2, +1))["end_collar_16_24"]),
                                                               sig2(one(cols("sigma/a 4", "history", 15.0, 1, "2 sigma inside R_t", math.pi / 4, -1))["end_collar_16_24"]))
    hl = cols("sigma/a 4", "history", 15.0, 1, None, PL)
    V["hist_long_collar_max"] = (max(c["end_collar_16_24"] for c in hl), sig2)
    RULE["hist_long_collar_max"] = "the largest end_collar_16_24 over the %d wavelength-r_s columns (ak_0 +-2pi/64, both depths) on that history, kappa 1" % len(hl)
    ALT["hist_long_collar_max"] = "ak_0 +2pi/64 only: %s; kappa 2: %s" % (sig2(max(c["end_collar_16_24"] for c in hl if c["ak0"] > 0)),
                                                                     sig2(max(c["end_collar_16_24"] for c in cols("sigma/a 4", "history", 15.0, 2, None, PL))))
    h30 = cols("sigma/a 4", "history", 30.0, 1)
    note30 = "the template's \"from T/r_s 30 nothing reaches the collar by 40\": the largest end_collar_16_24 over the %d columns launched at 30 (kappa 1) %s" % (len(h30), e2(max(c["end_collar_16_24"] for c in h30)))
    for T, key in ((15.0, "F_15"), (16.0, "F_16"), (25.0, "F_25")):
        V[key] = (arow(T)["F_D"], e2s)
        RULE[key] = "A_single_object_v2_summary.json, rs64_sigma4, kappa 1, the dense root, T/r_s %g: F_D (F = -2 Im Tr(h_XY D_YX), outward positive; F_C %s)" % (T, e2s(arow(T)["F_C"]))
        ALT[key] = "kappa 2: %s" % e2s(arow(T, 2)["F_D"])
    V["band_dust_15"] = (arow(15.0)["band_occupation"]["dust_ge_pi4"], sig2)
    RULE["band_dust_15"] = "the same row at T/r_s 15: band_occupation dust_ge_pi4 (the dust's interior weight at |ak| at or above pi/4)"
    ALT["band_dust_15"] = "kappa 2: %s; over the rows 15 to 40 (kappa 1): %s to %s" % (
        sig2(arow(15.0, 2)["band_occupation"]["dust_ge_pi4"]), sig2(min(x["band_occupation"]["dust_ge_pi4"] for x in A["rows"] if x["family"] == "rs64_sigma4" and x["kappa"] == 1 and x["root"] == "dense")),
        sig2(max(x["band_occupation"]["dust_ge_pi4"] for x in A["rows"] if x["family"] == "rs64_sigma4" and x["kappa"] == 1 and x["root"] == "dense")))
    s8 = one(cols("sigma/a 8", "history", 15.0, 1, "annulus middle", P2, +1))
    V["s8_annulus_pi2_collar"] = (s8["end_collar_16_24"], sig2)
    RULE["s8_annulus_pi2_collar"] = "B's sigma/a 8 (B.6 scale control) history from T/r_s 15, kappa 1, ak_0 +pi/2, the annulus middle: end_collar_16_24"
    ALT["s8_annulus_pi2_collar"] = "kappa 2: %s; ak_0 -pi/4: %s" % (sig2(one(cols("sigma/a 8", "history", 15.0, 2, "annulus middle", P2, +1))["end_collar_16_24"]),
                                                               sig2(one(cols("sigma/a 8", "history", 15.0, 1, "annulus middle", math.pi / 4, -1))["end_collar_16_24"]))
    # the template
    tp = open(a.template).read().split("\n\n")
    para = [p for p in tp if p.startswith('"')]
    cond = [p for p in tp if p.startswith("Condition:")]
    if len(para) != 1 or len(cond) != 1:
        sys.exit("the template is not in the expected form")
    text = para[0].strip()
    toks = re.findall(r"\[([A-Za-z][A-Za-z0-9_]*)\]", text)
    miss = [t_ for t_ in toks if t_ not in V]
    if miss or sorted(set(toks)) != sorted(V):
        sys.exit("the brackets and the values do not match: missing %s, unused %s" % (miss, sorted(set(V) - set(toks))))
    L = ["## engine_single_object_numbers_v1 (engine spec 38.4 (3)): FILE 4's brackets, each value, its rule and the alternatives the template's words leave open", "",
         "- sources: %s (md5 %s); %s (md5 %s) and %s (md5 %s), each equal to the HORIZON summary's own inputs record" % (a.horizon, md5(a.horizon), a.a_summary, md5(a.a_summary), a.a_sigma4, md5(a.a_sigma4)), "",
         "| bracket | printed | stored value | rule | alternatives (not printed) |", "|---|---|---|---|---|"]
    for t_ in toks:
        val, fmt = V[t_]
        L.append("| [%s] | %s | %.10g | %s | %s |" % (t_, fmt(val), val, RULE[t_].replace("|", "/"), ALT.get(t_, "-").replace("|", "/")))
        text = text.replace("[%s]" % t_, fmt(val), 1)
    if re.search(r"\[[A-Za-z][A-Za-z0-9_]*\]", text):
        sys.exit("a bracket is left")
    head = ("# Item 7, the cross-surface paragraph: the single object's numbers, version 1 (engine spec 38.4 (3); FILE 4 filled by "
            "engine_single_object_numbers_v1 from HORIZON_single_object_v1_summary.json and the two A files it records, every value the stored one at two "
            "significant figures, the rules in the composition's log; it prints only with the icloud lane's second route: the slab blocks' sigma_max at d 2, 4 "
            "and 8 for the state and the Killing vacuum at T/r_s 15 and 30, kappa 1 and 2, evaluated from the stored slices on the box, and the massive "
            "outward branch (strand_R's scope; engine spec 38.4 (3)); until then a record, not a paper text; the lead's text; the builder copies, never edits)")
    body = "\n".join([head, "", text, "", cond[0].strip(), ""])
    low = body.lower()
    hits = {w: low.count(w) for w in REPLACED if w in low}
    L += ["", "- %s" % note30,
          "- for the lead (the coordinator's note, inbox #45 and #46): the paragraph prints \"[band_dust_15] of the dust's interior weight sits at |ak| at or above pi/4, the outward band\"; #46 reads that bin as \"an unsigned windowed momentum share, not a projection onto the guaranteed outward positive intrinsic Dirac branch\" (RELABEL_table_v1.20 carries it); the slab blocks print \"per radial channel kappa 1 and never weighted\" and the entropy routes \"(per radial channel; not the entropy coefficient)\", as the template has them; the two-figure rule prints the lattice-scale column's 0.997 as 1.0",
          "- sec. 4's replaced words (%s) in the composed file: %s; no bracket left" % (", ".join(REPLACED), "none" if not hits else "FOUND %s" % hits)]
    if hits:
        open(a.log, "x").write("\n".join(L) + "\n\nSTOPPED\n")
        sys.exit("sec. 4 words found: %s" % hits)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "x").write(body)
    L += ["- composed: %s" % a.out, "", "### the filled paragraph", "", text]
    open(a.log, "x").write("\n".join(L) + "\n")
    print("wrote %s and %s" % (a.out, a.log))


if __name__ == "__main__":
    main()
