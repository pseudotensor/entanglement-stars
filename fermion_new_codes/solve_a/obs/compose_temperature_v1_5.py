#!/usr/bin/env python
"""compose_temperature_v1_5.py (builder analyst; engine spec 38.2 (1): "FILE 1 amends version 1.4 into TEMPERATURE_solved_exterior_v1.5
by compose_temperature_v1_5 (new): the literal replacements of sec. 4, the comparator wording, D_1 and delta_conn kept separate with
kappa_read unresolved, the kappa_0 clause with its factors, the tau 4 clause, the budget sentence with the counts' basis and both tau 4
conventions, every number printed by the script from the stored rows into the placeholders"): the delivered version 1.4 (its statement
and its condition paragraph) with FILE 1 (TEMPERATURE_solved_exterior_v1.5_amendments.md) applied literally: (a) the replacements of
requirements_new sec. 4 throughout, every old and new string taken from (a)'s own quotes (the four "held fixed ..." items: "frozen" read
"held fixed" in each quoted phrase, as (a) says), the plural "frozen-stage readings" before the singular; (b) to (f) each a single
replacement in the statement, old and new from the paragraph's quotes. Each replacement's count is printed and checked against the
count stated here before composing (the counts the builder read in version 1.4: 3 and 1 for the plural and singular readings, 2 for "of
the freeze", 1 for every other); a count other than that stops the compose. The placeholders are filled from the files FILE 1's condition
names, parsed, never retyped: compose_temperature_v1_4.md ([b_max], [b_min]: the replaced totals as rounded there; [d_max], [d_min]: the
percent of the replaced term d; [n_max], [n_min]: the replaced counts; [t4_min_old], [t4_min_new], [t4_max_old], [t4_max_new]: the tau 4
records' stored and replaced counts at 39.5 and 39.75), cross-checked by recomputing them from the stored J3' budget rows exactly as
compose_temperature_v1_4 did (a difference stops the compose); TEMPERATURE_kappa0_table_v1.md ([kappa0_dev]: "between x and y" with x, y
the smallest and largest kappa_0/kappa_T - 1 over its five records at two significant figures; [f1_max] to [f3_min]: the three factors
at the production's maximum and minimum records, identified by the J3' JSONs' frozen T/r_s, to two decimals); p26_slope_ratios_v1.md's
tau 4 tables at 39.5 and 39.75 ([tau4_kT_min] to [tau4_km_max]: the smallest and largest 2 pi T_fit/kappa_T and 2 pi T_fit/kappa_m over
the eligible rows, omega r_s 0.5 at holds 32, 48 and 64, both m/omega, both records, to two decimals; [tau4_km_slope_39p5],
[tau4_km_slope_39p75]: each record's 4 G E kappa_m to two decimals). The composed file: a head, the statement, and the condition
paragraph (version 1.4's with (a) applied, then "Version 1.5's amendments:" and FILE 1's condition, then the Not printed lists, version
1.4's and FILE 1's). Asserted before writing: no placeholder left in the statement; none of sec. 4's replaced words ("frozen", "freez",
"window", "prediction", any case) anywhere in the composed file; none of the further sec. 4 phrases listed in FORBIDDEN. Refuses to
overwrite. Args: --v14 --amend --log14 --kappa0 --p26 --max-json --min-json --tau4-min-json --tau4-max-json --out --log."""
import argparse
import json
import os
import re
import sys

REPLACED = ("frozen", "freez", "window", "prediction")
FORBIDDEN = ("toy model", "emergent gravity", "derived einstein", "derived from the fermions", "semiclassical", "mean-field", "mean field",
             "discretized", "an event horizon forms", "a singularity forms", "no black object", "obligation", "bonds keep their strength",
             "bonds always cross", "unitary completion", "rectangle", "allocation exponent", "uv complete", "near-maximal", "remnant",
             "page fraction", "page curve", "stationary luminosity", "static star", "exact for irrotational", "must radiate at t_h",
             "set by the masses", "0.99965", "(a/r)^2")
EXPECTED_A = {0: 1, 2: 1, 5: 3, 4: 1, 8: 1, 10: 2, 12: 1, 13: 1, 14: 1, 15: 1, 17: 1, 19: 1, 21: 1, 23: 1}


def rows_read(d):
    return [b for b in d["budget"] if b.get("T_over_T_H") is not None]


def inside(b, total):
    return abs(b["T_over_T_H"] - 1.0) <= total


def rng2(vals):
    lo, hi = "%.2f" % min(vals), "%.2f" % max(vals)
    return lo if lo == hi else "%s to %s" % (lo, hi)


def recompute(max_json, min_json, t4min, t4max):
    """compose_temperature_v1_4's computation, repeated as a cross-check of the values parsed from its log."""
    r = {}
    for name, path, frows in (("max", max_json, (0.5, (48.0, 64.0))), ("min", min_json, (0.25, (48.0, 64.0)))):
        d = json.load(open(path)); fr = d["frozen"]
        dterm = abs(fr["kappa_T"] - fr["kappa_m"]) / fr["kappa_T"]
        rd = rows_read(d)
        sel = [b for b in d["budget"] if b["omega_rs"] == frows[0] and b["hold_over_rs"] in frows[1]]
        new_tot = {id(b): b["total_linear"] - b["shift_a"] + dterm for b in d["budget"]}
        r["b_" + name] = rng2([new_tot[id(b)] for b in sel])
        r["d_" + name] = "%.1f" % (100 * dterm)
        r["n_" + name] = "%d of %d" % (sum(1 for b in rd if inside(b, new_tot[id(b)])), len(rd))
    for name, path in (("t4_min", t4min), ("t4_max", t4max)):
        d = json.load(open(path)); fr = d["frozen"]; dterm = abs(fr["kappa_T"] - fr["kappa_m"]) / fr["kappa_T"]; rd = rows_read(d)
        r[name + "_old"] = "%d of %d" % (sum(1 for b in rd if inside(b, b["total_linear"])), len(rd))
        r[name + "_new"] = "%d of %d" % (sum(1 for b in rd if inside(b, b["total_linear"] - b["shift_a"] + dterm)), len(rd))
    return r


def parse_log14(text):
    r = {}
    secs = re.split(r"\n### ", text)
    for name, key in (("max", "### the maximum"), ("min", "### the minimum")):
        s = [x for x in secs if x.startswith(key[4:])][0]
        r["b_" + name] = re.search(r"replaced [0-9., ]+ \(rounded ([0-9.]+(?: to [0-9.]+)?)\)", s).group(1)
        r["d_" + name] = re.search(r"the replaced term d = \|kappa_T - kappa_m\|/kappa_T = [0-9.]+ \(([0-9.]+) percent\)", s).group(1)
        r["n_" + name] = re.search(r"- readings inside: stored [0-9]+ of [0-9]+; replaced ([0-9]+ of [0-9]+)", s).group(1)
    for name, fn in (("t4_min", "J3P_raymap_v8_tau4_min.json"), ("t4_max", "J3P_raymap_v9_tau4_max.json")):
        m = re.search(re.escape(fn) + r" \(frozen at T/rs [0-9.]+\): d = [0-9.]+; readings inside: stored ([0-9]+ of [0-9]+), replaced ([0-9]+ of [0-9]+)", text)
        r[name + "_old"], r[name + "_new"] = m.group(1), m.group(2)
    return r


def parse_kappa0(text, T_max, T_min):
    rows = [ln for ln in text.split("\n") if ln.startswith("| prod19 ") or ln.startswith("| tau4 ")]
    devs, fac = [], {}
    for ln in rows:
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        devs.append(float(c[12]))
        if c[0] == "prod19":
            fac[c[1]] = [float(x) for x in c[13].split(";")]
    if len(devs) != 5:
        sys.exit("the kappa_0 table does not hold five records: %d" % len(devs))
    r = {"kappa0_dev": "between %.1e and %.1e" % (min(devs), max(devs)), "_devs": devs}
    for name, T in (("max", T_max), ("min", T_min)):
        f = fac["%.4f" % T]
        for i in range(3):
            r["f%d_%s" % (i + 1, name)] = "%.2f" % f[i]
        r["_fac_" + name] = f
    return r


def parse_p26(text):
    r = {}
    kT, km = [], []
    for T, tag in (("39.5000", "39p5"), ("39.7500", "39p75")):
        sec = [x for x in re.split(r"\n### ", text) if x.startswith("the tau 4 record at") and ("frozen at T/rs %s)" % T) in x.split("\n")[0]][0]
        r["tau4_km_slope_" + tag] = "%.2f" % float(re.search(r"4 G E kappa_m ([0-9.]+)\.", sec).group(1))
        r["_slope_" + tag] = float(re.search(r"4 G E kappa_m ([0-9.]+)\.", sec).group(1))
        for ln in sec.split("\n"):
            if not ln.startswith("| ") or ln.startswith("| omega"):
                continue
            c = [x.strip() for x in ln.strip().strip("|").split("|")]
            if len(c) < 7 or c[0] != "0.5" or c[2] not in ("32", "48", "64") or c[5] == "-":
                continue
            kT.append(float(c[5])); km.append(float(c[6]))
    if len(kT) != 12:
        sys.exit("the tau 4 eligible rows are not 12 (two records, three holds, two m/omega): %d" % len(kT))
    r.update({"tau4_kT_min": "%.2f" % min(kT), "tau4_kT_max": "%.2f" % max(kT), "tau4_km_min": "%.2f" % min(km), "tau4_km_max": "%.2f" % max(km),
              "_kT": (min(kT), max(kT)), "_km": (min(km), max(km))})
    return r


def main():
    ap = argparse.ArgumentParser()
    for k in ("v14", "amend", "log14", "kappa0", "p26", "max_json", "min_json", "tau4_min_json", "tau4_max_json", "out", "log"):
        ap.add_argument("--" + k.replace("_", "-"), required=True)
    a = ap.parse_args()
    for p in (a.out, a.log):
        if os.path.exists(p):
            sys.exit("refusing to overwrite %s" % p)
    L = ["## compose_temperature_v1_5 (engine spec 38.2 (1)): FILE 1 applied to version 1.4; the replacement map with its counts; the placeholders and their sources; the checks", ""]
    v14 = open(a.v14).read().split("\n")
    if not (v14[2].startswith('"') and v14[4].startswith("Condition: ")):
        sys.exit("version 1.4 is not in the expected form (head, statement, condition)")
    st, cond = v14[2], v14[4]
    am = open(a.amend).read().split("\n\n")
    par = {p[:3]: p for p in am}
    q = {k: re.findall(r'"([^"]*)"', par[k]) for k in ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")}
    shape = {k: len(v) for k, v in q.items()}
    if shape != {"(a)": 25, "(b)": 2, "(c)": 2, "(d)": 2, "(e)": 6, "(f)": 4}:
        sys.exit("FILE 1's quotes are not as read: %s" % shape)
    qa = q["(a)"]
    prefix = qa[16][:-len(" ...")] if qa[16].endswith(" ...") else None
    if prefix != "held fixed":
        sys.exit("(a)'s held-fixed rule is not as read: %r" % qa[16])
    amap = [(0, qa[0], qa[1]), (2, qa[2], qa[3]), (5, qa[5], qa[7]), (4, qa[4], qa[6]), (8, qa[8], qa[9]), (10, qa[10], qa[11])]
    for i in (12, 13, 14, 15):
        if not qa[i].startswith("frozen "):
            sys.exit("(a)'s quote %d does not start with frozen: %r" % (i, qa[i]))
        amap.append((i, qa[i], prefix + qa[i][len("frozen"):]))
    amap += [(17, qa[17], qa[18]), (19, qa[19], qa[20]), (21, qa[21], qa[22]), (23, qa[23], qa[24])]
    L += ["### (a): the literal replacements throughout (statement and condition), old and new from (a)'s quotes", "",
          "| (a)'s quote | old | new | expected count | count in the statement | count in the condition | total |", "|---|---|---|---|---|---|---|"]
    for i, old, new in amap:
        cs, cc = st.count(old), cond.count(old)
        L.append("| %d | %s | %s | %d | %d | %d | %d |" % (i, old, new, EXPECTED_A[i], cs, cc, cs + cc))
        if cs + cc != EXPECTED_A[i]:
            open(a.log, "x").write("\n".join(L) + "\n\nSTOPPED: (a) quote %d counted %d, expected %d\n" % (i, cs + cc, EXPECTED_A[i]))
            sys.exit("(a) quote %d: count %d, expected %d" % (i, cs + cc, EXPECTED_A[i]))
        st, cond = st.replace(old, new), cond.replace(old, new)
    # the placeholders
    lg = parse_log14(open(a.log14).read())
    rc = recompute(a.max_json, a.min_json, a.tau4_min_json, a.tau4_max_json)
    diff = {k: (lg[k], rc[k]) for k in lg if lg[k] != rc[k]}
    T_max = json.load(open(a.max_json))["frozen"]["T_over_rs"]; T_min = json.load(open(a.min_json))["frozen"]["T_over_rs"]
    k0 = parse_kappa0(open(a.kappa0).read(), T_max, T_min)
    p26 = parse_p26(open(a.p26).read())
    ph = dict(lg); ph.update({k: x for k, x in k0.items() if not k.startswith("_")}); ph.update({k: x for k, x in p26.items() if not k.startswith("_")})
    L += ["", "### the placeholders (each from the file FILE 1's condition names; parsed, never retyped)", "",
          "- from %s: %s" % (a.log14, "; ".join("[%s] %s" % (k, lg[k]) for k in sorted(lg))),
          "- the same recomputed from the stored J3' budget rows (%s, %s, %s, %s) as compose_temperature_v1_4 did: %s" % (
              a.max_json, a.min_json, a.tau4_min_json, a.tau4_max_json, "equal on every placeholder" if not diff else "DIFFERENT: %s" % diff),
          "- from %s: kappa_0/kappa_T - 1 on the five records %s; [kappa0_dev] %s; the production's maximum (frozen T/r_s %.4f) factors %s, its minimum (%.4f) %s; [f1_max] %s, [f2_max] %s, [f3_max] %s, [f1_min] %s, [f2_min] %s, [f3_min] %s" % (
              a.kappa0, ", ".join("%.3e" % x for x in k0["_devs"]), k0["kappa0_dev"], T_max, k0["_fac_max"], T_min, k0["_fac_min"],
              k0["f1_max"], k0["f2_max"], k0["f3_max"], k0["f1_min"], k0["f2_min"], k0["f3_min"]),
          "- from %s (the tau 4 tables at 39.5 and 39.75, the 12 eligible rows at omega r_s 0.5, holds 32, 48, 64, both m/omega): 2 pi T_fit/kappa_T %.4f to %.4f ([tau4_kT_min] %s, [tau4_kT_max] %s); 2 pi T_fit/kappa_m %.4f to %.4f ([tau4_km_min] %s, [tau4_km_max] %s); 4 G E kappa_m %.6f at 39.5 ([tau4_km_slope_39p5] %s) and %.6f at 39.75 ([tau4_km_slope_39p75] %s)" % (
              a.p26, p26["_kT"][0], p26["_kT"][1], p26["tau4_kT_min"], p26["tau4_kT_max"], p26["_km"][0], p26["_km"][1], p26["tau4_km_min"], p26["tau4_km_max"],
              p26["_slope_39p5"], p26["tau4_km_slope_39p5"], p26["_slope_39p75"], p26["tau4_km_slope_39p75"])]
    if diff:
        open(a.log, "x").write("\n".join(L) + "\n\nSTOPPED: the log's values differ from the recomputation\n")
        sys.exit("the log's values differ from the recomputation: %s" % diff)
    # (b) to (f): each old once in the statement
    pairs = [("(b)", q["(b)"][0], q["(b)"][1]), ("(c)", q["(c)"][0], q["(c)"][1]), ("(d)", q["(d)"][0], q["(d)"][1]),
             ("(e) 1", q["(e)"][0], q["(e)"][1]), ("(e) 2", q["(e)"][2], q["(e)"][3]), ("(e) 3", q["(e)"][4], q["(e)"][5]),
             ("(f) 1", q["(f)"][0], q["(f)"][1]), ("(f) 2", q["(f)"][2], q["(f)"][3])]
    L += ["", "### (b) to (f): single replacements in the statement (old and new from each paragraph's quotes; the placeholders filled)", "",
          "| item | old (first 90 characters) | placeholders filled | expected count | count |", "|---|---|---|---|---|"]
    tokens_all = set()
    for lab, old, new in pairs:
        toks = re.findall(r"\[([a-z][a-z0-9_]*)\]", new)
        tokens_all |= set(toks)
        miss = [t_ for t_ in toks if t_ not in ph]
        if miss:
            sys.exit("%s: no value for placeholder(s) %s" % (lab, miss))
        newf = new
        for t_ in toks:
            newf = newf.replace("[%s]" % t_, ph[t_])
        c_ = st.count(old)
        L.append("| %s | %s | %s | 1 | %d |" % (lab, old[:90], ", ".join("[%s] %s" % (t_, ph[t_]) for t_ in toks) or "-", c_))
        if c_ != 1:
            open(a.log, "x").write("\n".join(L) + "\n\nSTOPPED: %s counted %d, expected 1\n" % (lab, c_))
            sys.exit("%s: count %d, expected 1" % (lab, c_))
        st = st.replace(old, newf)
    left = [t_ for t_ in tokens_all if "[%s]" % t_ in st]
    if left:
        sys.exit("placeholders left in the statement: %s" % left)
    # the condition paragraph
    c14, np14 = cond.split(" Not printed: ", 1)
    cf = par["Con"]
    c15, np15 = cf.split(" Not printed: ", 1)
    c15 = c15[len("Condition: "):]
    cond15 = "%s Version 1.5's amendments: %s Not printed: %s From the version 1.5 amendments: %s" % (c14, c15, np14.rstrip(), np15.strip())
    head = ("# The temperature on the solved exterior: the one solved-exterior statement the paper prints, version 1.5 (engine spec 38.2; version 1.4 "
            "with the amendments (a) to (f) of TEMPERATURE_solved_exterior_v1.5_amendments.md, the lead's text of turn 46, composed by compose_temperature_v1_5: "
            "every replacement literal, its count printed in the composition's log; every placeholder filled from the stored files the amendments' condition names; "
            "version 1.4 stays the record; the builder copies, never edits)")
    body = "\n".join([head, "", st, "", cond15, ""])
    low = body.lower()
    hits = {w: low.count(w) for w in REPLACED + FORBIDDEN if w in low}
    L += ["", "### the checks", "",
          "- the statement: no placeholder left (the tokens %s filled)" % ", ".join("[%s]" % t_ for t_ in sorted(tokens_all)),
          "- sec. 4's replaced words (%s) and the further sec. 4 phrases (%s) in the composed file, any case: %s" % (
              ", ".join(REPLACED), ", ".join(FORBIDDEN), "none" if not hits else "FOUND %s" % hits)]
    if hits:
        open(a.log, "x").write("\n".join(L) + "\n\nSTOPPED: sec. 4 words found\n")
        sys.exit("sec. 4 words found: %s" % hits)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "x").write(body)
    L += ["- composed: %s" % a.out, "- every number above is printed by the script from %s, %s, %s and the J3' JSONs; the text from %s and %s." % (
        a.log14, a.kappa0, a.p26, a.v14, a.amend)]
    open(a.log, "x").write("\n".join(L) + "\n")
    print("wrote %s and %s" % (a.out, a.log))


if __name__ == "__main__":
    main()
