#!/usr/bin/env python
"""engine_single_object_numbers_v2.py (builder analyst, the HORIZON version 2 builder; engine spec 39.5 (2): "FILE 3:
ITEM7_single_object_numbers_v2 as a template filled by engine_single_object_numbers_v2 (new) from the same json: the two-route part printable
with #51's wording; B's transfer and the dust's band as a labelled one-route record paragraph that does not print until version 2's signed
readout and a second route exist"; the brief: "the printable two-route paragraph with #51's wording; the record paragraph labelled, not
printed"). FILE 3 = ITEM7_single_object_numbers_v2_template.md: its head, its two quoted paragraphs (Printable; Record) and its Condition
paragraph. Every bracket is filled from HORIZON_single_object_v1_summary.json (B's columns) and, for A's items, from the two A files that
summary records among its inputs (A_single_object_v2_summary.json: the rows, the worst residuals and the strand_R checks;
A_rs64_sigma4.json: the 4-site slab blocks), each md5 checked against the summary's own record before use (the summary holds no A item),
at two significant figures (exponent form for the slab blocks, residuals and currents, decimals for fractions and ratios), version 1's rules
(engine_single_object_numbers_v1.py) for the brackets the two versions share; the three heldfixed_ brackets are version 1's frozen_ ones
(the json's mode "frozen", the word the template replaces), mapped by name. The second route (#51) is parsed from
review_ITEM7_single_object_numbers_v1.tex (its table of largest singular values, its two ratios, its identity residual and entropy-route
difference), never retyped, and printed in the log beside the first route's values; the template's literal purity defect 7.5e-8 is checked
against the A summary's stored purity maximum for the production family. Output: ITEM7_single_object_numbers_v2.md (the head; the
printable paragraph filled; the record paragraph filled under its own label, a record that does not print; the Condition paragraph
verbatim) and a log with every bracket's value, rule and the alternatives the words leave open. Asserted: no bracket left; none of sec. 4's
replaced words (frozen, freez, window, prediction) in the composed file; "escape" counted and flagged (the lead's text, copied). Refuses to overwrite.
Args: --template --horizon --a-summary --a-sigma4 --review51 --out --log."""
import argparse
import hashlib
import json
import math
import os
import re
import sys

REPLACED = ("frozen", "freez", "window", "prediction")
FLAGGED = ("escape",)                 # 39.5 (1) (x) bars it in HORIZON version 2; the lead's template text is copied, never edited: counted and flagged
P2, PL = math.pi / 2, 2 * math.pi / 64


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def sig2(x):
    r = float("%.2g" % x)
    d = math.floor(math.log10(abs(r))) if r != 0 else 0
    return "%.*f" % (max(0, 1 - d), r)


def e2(x):
    return "%.1e" % x


def e2s(x):
    return "%+.1e" % x


def parse_review(p):
    """#51's second route from its tex: the table rows T & kappa & (C_* block d 2, 4, 8) & (P_K block d 2, 4, 8), the two ratios, the
    largest observed discrepancy, the identity residual and the entropy-route difference."""
    t = open(p).read()
    rows = {}
    for m in re.finditer(r"(\d+)&(\d)&\(([\d.]+),([\d.]+),([\d.]+)\)&\(([\d.]+),([\d.]+),([\d.]+)\)", t):
        T, k = int(m.group(1)), int(m.group(2))
        rows[(T, k)] = {"C": [float(m.group(i)) for i in (3, 4, 5)], "P": [float(m.group(i)) for i in (6, 7, 8)]}
    rat = re.search(r"ratios are\s*\\\(([\d.]+)\\\)\s*and\s*\\\(([\d.]+)\\\)", t)
    num = lambda s: float(s.replace("\\,10^{", "e").replace("}", ""))
    disc = re.search(r"discrepancy is\s*\\\(([\d.]+\\,10\^\{-?\d+\})\\\)", t)
    ident = re.search(r"\\le\s*([\d.]+\\,10\^\{-?\d+\})\\\)", t)
    ent = re.search(r"agree within\s*\\\(([\d.]+\\,10\^\{-?\d+\})\\\)", t)
    return {"rows": {"%d_%d" % k_: v_ for k_, v_ in rows.items()}, "ratios": [float(rat.group(1)), float(rat.group(2))] if rat else None,
            "largest_discrepancy": num(disc.group(1)) if disc else None, "identity_residual": num(ident.group(1)) if ident else None,
            "entropy_routes": num(ent.group(1)) if ent else None}


def main():
    ap = argparse.ArgumentParser()
    for k in ("template", "horizon", "a_summary", "a_sigma4", "review51", "out", "log"):
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
    R2 = parse_review(a.review51)

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
        return one([x for x in A["rows"] if x["family"] == "rs64_sigma4" and x["kappa"] == kap and x["root"] == "dense" and x["T_over_rs"] == T])

    def slab(T, d, which, kap=1):
        rr = one([x for x in S4["records"] if x["T_over_rs"] == T])
        ch = one([c for c in rr["channels"] if c["kappa"] == kap])
        return ch["splits"]["dense"]["slabs"]["P07_%d" % d][which]["sigma_max"]

    V, RULE, ALT = {}, {}, {}
    two = []
    for T in (15.0, 30.0):
        tag = "%d" % T
        r2 = R2["rows"].get("%d_1" % int(T))
        for i, d in enumerate((2, 4, 8)):
            V["C_%s_d%d" % (tag, d)] = (slab(T, d, "C"), e2)
            V["V_%s_d%d" % (tag, d)] = (slab(T, d, "P_minus"), e2)
            RULE["C_%s_d%d" % (tag, d)] = "A_rs64_sigma4.json, record T/r_s %g, kappa 1, the dense split's slab P07_%d: C.sigma_max (the state)" % (T, d)
            RULE["V_%s_d%d" % (tag, d)] = "the same slab's P_minus.sigma_max (the Killing vacuum of the same held fields; P, the continued reference, %s)" % e2(slab(T, d, "P"))
            ALT["C_%s_d%d" % (tag, d)] = "kappa 2: %s" % e2(slab(T, d, "C", 2))
            ALT["V_%s_d%d" % (tag, d)] = "kappa 2: %s" % e2(slab(T, d, "P_minus", 2))
            if r2:
                two.append(("C", T, d, slab(T, d, "C"), r2["C"][i])); two.append(("P_-", T, d, slab(T, d, "P_minus"), r2["P"][i]))
        V["ratio_%s_d8" % tag] = (slab(T, 8, "C") / slab(T, 8, "P_minus"), sig2)
        RULE["ratio_%s_d8" % tag] = "the script's division of the two stored d 8 values at T/r_s %g (C over P_minus, kappa 1)" % T
        ALT["ratio_%s_d8" % tag] = "kappa 2: %s" % sig2(slab(T, 8, "C", 2) / slab(T, 8, "P_minus", 2))
    V["SO1_max"] = (A["worst_SO1_residual"], e2)
    RULE["SO1_max"] = "A_single_object_v2_summary.json's worst_SO1_residual over its %d rows (3 families; version 1's rule, a bound on the production's rows)" % len(A["rows"])
    fam = A["strand_R_checks"]["rs64_sigma4"]
    ALT["SO1_max"] = "the production family alone (strand_R_checks rs64_sigma4 outer): %s; the second route over its 4 cases: %s" % (e2(fam["outer_SO1_residual_max"]), e2(R2["identity_residual"]) if R2["identity_residual"] else "-")
    V["SX_max"] = (A["worst_S_X_agreement"], e2)
    RULE["SX_max"] = "its worst_S_X_agreement over the same rows"
    ALT["SX_max"] = "the production family alone: %s; the second route over its 4 cases: %s" % (e2(fam["outer_SO2_agreement_max"]), e2(R2["entropy_routes"]) if R2["entropy_routes"] else "-")
    purity = fam["stored_purity_max"]
    # the record paragraph (version 1's rules; heldfixed_ = version 1's frozen_)
    c_pi2 = one(cols("sigma/a 4", "frozen", 15.0, 1, "annulus middle", P2, +1))
    V["heldfixed_pi2_beyond16"] = (c_pi2["beyond16_final_before_reflection"], sig2)
    RULE["heldfixed_pi2_beyond16"] = ("version 1's frozen_pi2_beyond16: HORIZON's B columns, sigma/a 4 (production), held fixed (the json's mode \"frozen\") at T/r_s 15, kappa 1, ak_0 +pi/2, "
                                      "the annulus middle: beyond16_final_before_reflection (its arrival %g r_s after launch, the template's 23)" % c_pi2["arrival_after_launch_rs"])
    o = one(cols("sigma/a 4", "frozen", 15.0, 1, "2 sigma inside R_t", P2, +1))
    ALT["heldfixed_pi2_beyond16"] = "2 sigma inside R_t: %s (arrival %g); ak_0 -pi/2 annulus: %s" % (sig2(o["beyond16_final_before_reflection"]), o["arrival_after_launch_rs"],
                                                                                                    sig2(one(cols("sigma/a 4", "frozen", 15.0, 1, "annulus middle", P2, -1))["beyond16_final_before_reflection"]))
    c_l = one(cols("sigma/a 4", "frozen", 15.0, 1, "annulus middle", PL, +1))
    V["heldfixed_long_bound"] = (c_l["bound_overlap_end"], sig2)
    V["heldfixed_long_beyond16"] = (c_l["beyond16_final_before_reflection"], sig2)
    RULE["heldfixed_long_bound"] = ("version 1's frozen_long_bound: the same launch's column at ak_0 +2pi/64, the annulus middle (omega_bar %.4f, its weight above m %.3e): bound_overlap_end "
                                    "(the finite-box classifier's overlap; at launch %.6f)" % (c_l["omega_bar"], c_l["w_above_m"], c_l["bound_overlap_launch"]))
    RULE["heldfixed_long_beyond16"] = "version 1's frozen_long_beyond16: that column's beyond16_final_before_reflection"
    o2 = one(cols("sigma/a 4", "frozen", 15.0, 1, "2 sigma inside R_t", PL, +1))
    om = cols("sigma/a 4", "frozen", 15.0, 1, None, PL, -1)
    ALT["heldfixed_long_bound"] = "2 sigma inside R_t: %s; ak_0 -2pi/64 (annulus, 2 sigma): %s" % (sig2(o2["bound_overlap_end"]), ", ".join(sig2(c["bound_overlap_end"]) for c in om))
    ALT["heldfixed_long_beyond16"] = "2 sigma inside R_t: %s; ak_0 -2pi/64: %s" % (sig2(o2["beyond16_final_before_reflection"]), ", ".join(sig2(c["beyond16_final_before_reflection"]) for c in om))
    h_a = one(cols("sigma/a 4", "history", 15.0, 1, "annulus middle", P2, +1)); h_2 = one(cols("sigma/a 4", "history", 15.0, 1, "2 sigma inside R_t", P2, +1))
    V["hist_annulus_pi2_collar"] = (h_a["end_collar_16_24"], sig2); V["hist_2sig_pi2_collar"] = (h_2["end_collar_16_24"], sig2)
    RULE["hist_annulus_pi2_collar"] = "B's columns on the solved history from T/r_s 15, kappa 1, ak_0 +pi/2, the annulus middle: end_collar_16_24 (at T/r_s 40)"
    RULE["hist_2sig_pi2_collar"] = "the same at 2 sigma inside R_t"
    ALT["hist_annulus_pi2_collar"] = "kappa 2: %s; ak_0 -pi/4: %s" % (sig2(one(cols("sigma/a 4", "history", 15.0, 2, "annulus middle", P2, +1))["end_collar_16_24"]),
                                                                   sig2(one(cols("sigma/a 4", "history", 15.0, 1, "annulus middle", math.pi / 4, -1))["end_collar_16_24"]))
    ALT["hist_2sig_pi2_collar"] = "kappa 2: %s; ak_0 -pi/4: %s" % (sig2(one(cols("sigma/a 4", "history", 15.0, 2, "2 sigma inside R_t", P2, +1))["end_collar_16_24"]),
                                                                sig2(one(cols("sigma/a 4", "history", 15.0, 1, "2 sigma inside R_t", math.pi / 4, -1))["end_collar_16_24"]))
    hl = cols("sigma/a 4", "history", 15.0, 1, None, PL)
    V["hist_long_collar_max"] = (max(c["end_collar_16_24"] for c in hl), sig2)
    RULE["hist_long_collar_max"] = "the largest end_collar_16_24 over the %d wavelength-r_s columns (ak_0 +-2pi/64, both depths) on that history, kappa 1" % len(hl)
    ALT["hist_long_collar_max"] = "ak_0 +2pi/64 only: %s; kappa 2: %s" % (sig2(max(c["end_collar_16_24"] for c in hl if c["ak0"] > 0)),
                                                                     sig2(max(c["end_collar_16_24"] for c in cols("sigma/a 4", "history", 15.0, 2, None, PL))))
    for T, key in ((15.0, "F_15"), (16.0, "F_16"), (25.0, "F_25")):
        V[key] = (arow(T)["F_D"], e2s)
        RULE[key] = "A_single_object_v2_summary.json, rs64_sigma4, kappa 1, the dense root, T/r_s %g: F_D (F = -2 Im Tr(h_XY D_YX), outward positive; F_C %s)" % (T, e2s(arow(T)["F_C"]))
        ALT[key] = "kappa 2: %s" % e2s(arow(T, 2)["F_D"])
    V["band_dust_15"] = (arow(15.0)["band_occupation"]["dust_ge_pi4"], sig2)
    RULE["band_dust_15"] = "the same row at T/r_s 15: band_occupation dust_ge_pi4 (the dust's interior weight at |ak| at or above pi/4, unsigned)"
    ALT["band_dust_15"] = "kappa 2: %s; over the rows 15 to 40 (kappa 1): %s to %s" % (
        sig2(arow(15.0, 2)["band_occupation"]["dust_ge_pi4"]), sig2(min(x["band_occupation"]["dust_ge_pi4"] for x in A["rows"] if x["family"] == "rs64_sigma4" and x["kappa"] == 1 and x["root"] == "dense")),
        sig2(max(x["band_occupation"]["dust_ge_pi4"] for x in A["rows"] if x["family"] == "rs64_sigma4" and x["kappa"] == 1 and x["root"] == "dense")))
    s8 = one(cols("sigma/a 8", "history", 15.0, 1, "annulus middle", P2, +1))
    V["s8_annulus_pi2_collar"] = (s8["end_collar_16_24"], sig2)
    RULE["s8_annulus_pi2_collar"] = "B's sigma/a 8 (B.6 scale control) history from T/r_s 15, kappa 1, ak_0 +pi/2, the annulus middle: end_collar_16_24"
    ALT["s8_annulus_pi2_collar"] = "kappa 2: %s; ak_0 -pi/4: %s" % (sig2(one(cols("sigma/a 8", "history", 15.0, 2, "annulus middle", P2, +1))["end_collar_16_24"]),
                                                               sig2(one(cols("sigma/a 8", "history", 15.0, 1, "annulus middle", math.pi / 4, -1))["end_collar_16_24"]))
    # the template
    tp = open(a.template).read()
    parts = tp.split("\n\n")
    head_t = parts[0]
    pr = [p for p in parts if p.startswith("Printable:")]
    rc = [p for p in parts if p.startswith("Record (one route; not printed):")]
    cond = [p for p in parts if p.startswith("Condition:")]
    if len(pr) != 1 or len(rc) != 1 or len(cond) != 1:
        sys.exit("the template is not in the expected form")
    toks_all = re.findall(r"\[([A-Za-z][A-Za-z0-9_]*)\]", pr[0] + rc[0])
    miss = [t_ for t_ in toks_all if t_ not in V]
    if miss or sorted(set(toks_all)) != sorted(V):
        sys.exit("the brackets and the values do not match: missing %s, unused %s" % (miss, sorted(set(V) - set(toks_all))))
    if "7.5e-8" not in pr[0] or "%.1e" % purity != "7.5e-08":
        sys.exit("the template's purity defect 7.5e-8 against the stored maximum %.6e" % purity)
    L = ["## engine_single_object_numbers_v2 (engine spec 39.5 (2)): FILE 3's brackets, each value, its rule and the alternatives the template's words leave open", "",
         "- sources: %s (md5 %s); %s (md5 %s) and %s (md5 %s), each equal to the HORIZON summary's own inputs record; the second route %s (md5 %s)" % (
             a.horizon, md5(a.horizon), a.a_summary, md5(a.a_summary), a.a_sigma4, md5(a.a_sigma4), a.review51, md5(a.review51)),
         "- the template %s (md5 %s)" % (a.template, md5(a.template)), "",
         "| bracket | printed | stored value | rule | alternatives (not printed) |", "|---|---|---|---|---|"]
    filled = {}
    for nm, para in (("printable", pr[0]), ("record", rc[0])):
        text = para
        for t_ in re.findall(r"\[([A-Za-z][A-Za-z0-9_]*)\]", para):
            val, fmt = V[t_]
            L.append("| [%s] (%s) | %s | %.10g | %s | %s |" % (t_, nm, fmt(val), val, RULE[t_].replace("|", "/"), ALT.get(t_, "-").replace("|", "/")))
            text = text.replace("[%s]" % t_, fmt(val), 1)
        if re.search(r"\[[A-Za-z][A-Za-z0-9_]*\]", text):
            sys.exit("a bracket is left in the %s paragraph" % nm)
        filled[nm] = text
    L += ["", "### the two routes of the printable paragraph (the first: the HORIZON summary's A files; the second: inbox #51's recomputation, parsed from its tex)", "",
          "| block | T/r_s | d | first route | second route | first minus second | printed (both) |", "|---|---|---|---|---|---|---|"]
    for (bl, T, d, x1, x2) in two:
        L.append("| %s | %g | %d | %.10g | %.10g | %+.2e | %s; %s |" % (bl, T, d, x1, x2, x1 - x2, e2(x1), e2(x2)))
    if R2["ratios"]:
        L.append("- the kappa 1, d 8 ratios: first route %.6f and %.6f; second route %.5f and %.5f (printed %s and %s)" % (
            V["ratio_15_d8"][0], V["ratio_30_d8"][0], R2["ratios"][0], R2["ratios"][1], sig2(V["ratio_15_d8"][0]), sig2(V["ratio_30_d8"][0])))
    L.append("- the second route's largest discrepancy over its 72 comparisons %s; its identity residual %s and entropy-route difference %s (the first route's printed bounds %s and %s cover its 4 cases)" % (
        e2(R2["largest_discrepancy"]) if R2["largest_discrepancy"] else "-", e2(R2["identity_residual"]) if R2["identity_residual"] else "-",
        e2(R2["entropy_routes"]) if R2["entropy_routes"] else "-", e2(V["SO1_max"][0]), e2(V["SX_max"][0])))
    L.append("- the template's literal purity defect 7.5e-8 against the A summary's stored purity maximum for the production family %.6e (%s)" % (purity, e2(purity)))
    head = ("# Item 7, the cross-surface paragraph: the single object's numbers, version 2 (engine spec 39.5 (2); FILE 3 filled by "
            "engine_single_object_numbers_v2 from HORIZON_single_object_v1_summary.json and the two A files it records, every value the stored one at two "
            "significant figures, the rules and both routes in the composition's log; the printable paragraph has its two routes (inbox #51 the second) and "
            "prints with #51's wording; the record paragraph has one route, is labelled and does not print until HORIZON version 2's signed readout and a "
            "second route for the transfer exist; the lead's text; the builder copies, never edits)")
    body = "\n".join([head, "", filled["printable"].strip(), "", filled["record"].strip(), "", cond[0].strip(), ""])
    low = body.lower()
    cond_low = cond[0].lower()
    hits_out = {w: low.count(w) for w in REPLACED if low.count(w) > 0}
    flag_where = {nm: filled[nm].lower().count(w) for nm in filled for w in FLAGGED if filled[nm].lower().count(w)}
    L += ["", "- sec. 4's replaced words (%s) in the composed file: %s; no bracket left" % (", ".join(REPLACED), "none" if not hits_out else "FOUND %s" % hits_out),
          "- flagged for the lead (copied, not edited): \"escape\" occurs %s in the filled paragraphs (%s) and %d time(s) in the Condition paragraph, whose list of what is not printed names it; "
          "39.5 (1) (x) bars the word in HORIZON version 2" % (sum(flag_where.values()), ", ".join("%s %d" % (k_, v_) for k_, v_ in flag_where.items()) or "none", cond_low.count("escape"))]
    if hits_out:
        open(a.log, "x").write("\n".join(L) + "\n\nSTOPPED\n")
        sys.exit("words found: %s" % hits_out)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "x").write(body)
    L += ["- composed: %s (md5 %s)" % (a.out, md5(a.out)), "", "### the filled paragraphs", "", filled["printable"], "", filled["record"]]
    open(a.log, "x").write("\n".join(L) + "\n")
    print("wrote %s and %s" % (a.out, a.log))


if __name__ == "__main__":
    main()
