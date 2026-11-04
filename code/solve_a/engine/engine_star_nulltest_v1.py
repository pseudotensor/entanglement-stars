#!/usr/bin/env python
"""engine_star_nulltest_v1.py (builder analyst; engine spec 28.4 (d): "the null test E_16 - 5 E_8 + 4 E_4 (zero for a pure ell^2 law)",
"The null test on the existing triples (FILE 2; lead's arithmetic, the script confirms)", "Ruled: the ell^2 fits (the handoff's E_0 values)
are not error bars where the test exceeds 20 percent of the spread; the class-1 error is the spread itself"): reads every family record's
ell_control entries and prints, per entry with converged E at ell 4, 8 and 16: E_4, E_8, E_16, the null test D = E_16 - 5 E_8 + 4 E_4
(E_0 (1 - 5 + 4) + b (256 - 320 + 64) = 0 for E = E_0 + b ell^2), the spread S = E_16 - E_4, the ratio D/S, and whether |D/S| exceeds 0.2
(then the record's fit E_0 + b ell^2 is not an error bar and the spread is the error). Entries with an unconverged ell are listed without a
test. An entry repeated in several records of one family (the same NB_target and the same E at every ell) is printed once with its records
named. --check-statement: the ell-test paragraph of the lead's statement (FILE 2, STAR_family_statement_v3.md: "... -0.489 against the spread
E_16 - E_4 0.443 at (0.2, 1/4) N_B 160 and -0.099 against 0.133 at N_B 112; ...") is parsed (the family carried forward when a value names
none) and every quoted D and S compared with the computed one at the quoted number of decimals. Functions null_entries(records) and
fmt_entry(x) are used by engine_star_handoff_v4. Args: --families "label=json[,json...]" ... --md [--check-statement path]."""
import argparse
import json
import os
import re

ELLS = ("4", "8", "16")
RULE = 0.2


def null_entries(records):
    """records: [(name, record dict)] of one family. Returns the ell_control entries (deduplicated), each with E_4, E_8, E_16, D, S, D/S,
    fails (|D/S| > 0.2) or the unconverged ells."""
    out = []
    for name, R in records:
        for c in R.get("ell_control") or []:
            rows = c.get("rows", {})
            E = {k: rows[k]["E"] for k in rows if not rows[k].get("failed") and rows[k].get("E") is not None}
            bad = sorted((k for k in rows if rows[k].get("failed")), key=float)
            key = (round(float(c.get("NB_target", -1)), 6), tuple(sorted((k, round(v, 12)) for k, v in E.items())), tuple(bad))
            dup = next((x for x in out if x["key"] == key), None)
            if dup:
                dup["records"].append(name)
                continue
            x = {"key": key, "records": [name], "NB_target": float(c.get("NB_target", float("nan"))), "E": E, "unconverged": bad,
                 "fit": c.get("fit_E_a_plus_b_ell2"), "spread_E_record": c.get("spread_E")}
            if all(k in E for k in ELLS):
                D = E["16"] - 5.0 * E["8"] + 4.0 * E["4"]
                S = E["16"] - E["4"]
                x.update({"D": D, "S": S, "ratio": D / S if S != 0 else float("nan"), "fails": abs(D / S) > RULE if S != 0 else True})
            out.append(x)
    return out


def fmt_entry(x):
    """the handoff's table cell: D (D/S of S; the rule's reading)."""
    if "D" not in x:
        return "no triple (ell %s unconverged)" % ", ".join(x["unconverged"]) if x["unconverged"] else "no triple"
    return "%.4f (%.3f of %.4f; %s)" % (x["D"], x["ratio"], x["S"], "fit not an error bar" if x["fails"] else "fit within the rule")


def parse_statement(path):
    """the ell-test paragraph's quoted (D, S, family, N_B) with the quoted decimals."""
    txt = open(path).read()
    para = next((p for p in txt.split("\n\n") if p.startswith("The ell test")), None)
    if para is None:
        return None, []
    pat = re.compile(r"(-?\d+\.\d+) against (?:the spread E_16 - E_4 )?(\d+\.\d+) at (?:\((\d*\.?\d+), 1/(\d+)\) )?(?:N_B )?(\d+(?:\.\d+)?)")
    out, fam = [], None
    for mt in pat.finditer(para):
        D, S, m, gd, nb = mt.groups()
        if m is not None:
            fam = (float(m), 1.0 / float(gd))
        out.append({"D": float(D), "D_dec": len(D.split(".")[1]), "S": float(S), "S_dec": len(S.split(".")[1]), "m": fam[0], "g": fam[1],
                    "N_B": float(nb), "text": mt.group(0)})
    return para, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", required=True, help="label=path1.json[,path2.json]")
    ap.add_argument("--md", required=True)
    ap.add_argument("--check-statement", default=None)
    a = ap.parse_args()
    L = ["## The ell null test on every ell triple (engine_star_nulltest_v1.py)", "",
         "D = E_16 - 5 E_8 + 4 E_4 vanishes for E = E_0 + b ell^2; S = E_16 - E_4 (the spread); the lead's rule (engine spec 28.4): where |D/S| > %.1f "
         "the record's fit E_0 + b ell^2 is not an error bar and the class-1 error is the spread itself." % RULE, ""]
    allx = []
    for spec in a.families:
        label, paths = spec.split("=", 1)
        recs = [(os.path.basename(p), json.load(open(p))) for p in paths.split(",") if os.path.exists(p)]
        missing = [p for p in paths.split(",") if not os.path.exists(p)]
        if not recs:
            L += ["### %s: no record found (%s)" % (label, ", ".join(missing)), ""]
            continue
        m = float(recs[0][1]["args"]["m"]); g = float(recs[0][1]["args"]["g"])
        xs = null_entries(recs)
        L += ["### %s: m a = %g, g = %g (%s)" % (label, m, g, ", ".join(n for n, _ in recs)), ""]
        if not xs:
            L += ["- no ell_control entry in the records.", ""]
            continue
        L += ["| NB_target | E_4 | E_8 | E_16 | D = E_16 - 5E_8 + 4E_4 | S = E_16 - E_4 | D/S | abs(D/S) > 0.2 | the record's fit E_0 (b) | records |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for x in sorted(xs, key=lambda x: x["NB_target"]):
            E = x["E"]
            fx = lambda k: ("%.6f" % E[k]) if k in E else ("unconverged" if k in x["unconverged"] else "-")
            fit = x["fit"]
            if "D" in x:
                L.append("| %.3f | %s | %s | %s | %.6f | %.6f | %.4f | %s | %s | %s |" % (
                    x["NB_target"], fx("4"), fx("8"), fx("16"), x["D"], x["S"], x["ratio"], "yes: fit not an error bar" if x["fails"] else "no",
                    ("%.6f (%.3e)" % (fit[1], fit[0])) if fit else "-", ", ".join(x["records"])))
            else:
                L.append("| %.3f | %s | %s | %s | - | - | - | no triple | %s | %s |" % (
                    x["NB_target"], fx("4"), fx("8"), fx("16"), ("%.6f (%.3e)" % (fit[1], fit[0])) if fit else "-", ", ".join(x["records"])))
            allx.append((m, g, label, x))
        L.append("")
    tri = [t for t in allx if "D" in t[3]]
    L += ["### Summary", "",
          "- triples: %d; failing the rule (abs(D/S) > 0.2): %d; within it: %d; entries without a triple: %d." % (
              len(tri), sum(1 for t in tri if t[3]["fails"]), sum(1 for t in tri if not t[3]["fails"]), len(allx) - len(tri)),
          "- failing: %s." % ("; ".join("(%g, 1/%g) N_B %.3f D/S %.4f" % (t[0], round(1 / t[1]), t[3]["NB_target"], t[3]["ratio"]) for t in tri if t[3]["fails"]) or "none"),
          "- within the rule: %s." % ("; ".join("(%g, 1/%g) N_B %.3f D/S %.4f" % (t[0], round(1 / t[1]), t[3]["NB_target"], t[3]["ratio"]) for t in tri if not t[3]["fails"]) or "none"),
          "- without a triple: %s." % ("; ".join("(%g, 1/%g) N_B %.3f (ell %s unconverged)" % (t[0], round(1 / t[1]), t[3]["NB_target"], ", ".join(t[3]["unconverged"]) or "-") for t in allx if "D" not in t[3]) or "none"),
          ""]
    if a.check_statement:
        para, q = parse_statement(a.check_statement)
        L += ["### The statement's quoted values against the script (%s, the paragraph beginning 'The ell test')" % os.path.basename(a.check_statement), ""]
        if not q:
            L += ["- no quoted value parsed.", ""]
        else:
            L += ["| quoted text | family | N_B | quoted D | computed D | consistent at the quoted decimals | quoted S | computed S | consistent | computed D/S |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            nmis = 0
            for v in q:
                hit = [t[3] for t in tri if abs(t[0] - v["m"]) < 1e-12 and abs(t[1] - v["g"]) < 1e-12 and abs(t[3]["NB_target"] - v["N_B"]) < 0.5]
                if not hit:
                    L.append("| %s | (%g, 1/%g) | %g | %s | no triple found | - | %s | - | - | - |" % (v["text"], v["m"], round(1 / v["g"]), v["N_B"], v["D"], v["S"]))
                    nmis += 1
                    continue
                x = hit[0]
                okD = abs(x["D"] - v["D"]) <= 0.5 * 10 ** (-v["D_dec"]) + 1e-12
                okS = abs(x["S"] - v["S"]) <= 0.5 * 10 ** (-v["S_dec"]) + 1e-12
                nmis += (not okD) + (not okS)
                L.append("| %s | (%g, 1/%g) | %g | %s | %.6f | %s | %s | %.6f | %s | %.4f |" % (
                    v["text"], v["m"], round(1 / v["g"]), v["N_B"], v["D"], x["D"], "yes" if okD else "NO", v["S"], x["S"], "yes" if okS else "NO", x["ratio"]))
            L += ["", "- %d quoted pairs parsed; %d mismatches at the quoted decimals." % (len(q), nmis), ""]
    L += ["- every number above is read from the family records' ell_control entries and computed by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[-12:]))


if __name__ == "__main__":
    main()
