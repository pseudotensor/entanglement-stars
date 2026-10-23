#!/usr/bin/env python
"""fill_item7_paragraph_v2.py (the lane's analyst builder; engine spec 41.4 (2), turn 49's FILE 3): fills the lead's template
ITEM7_solved_history_paragraph_v2_template.md from PAGE_solved_history_v3_summary.json into ITEM7_solved_history_paragraph_v2.md. Every bracket
at a clock the template names is replaced by the value copied from the json (version 2's run v6bA_s4 through T/r_s 40, the run its printed
defaults come from; the controls' ranges over their clean pre-reflection windows, the composer's, which equal #56's pre-marker ranges); the
bracket of the extension to 100 is turn 50's and is kept as it stands; the template's text is copied, never edited; a values table with each
selection follows the paragraph; requirements_new.txt sec. 4's words and the template's 'Not printed' words are checked on the filled paragraph.
Usage: python fill_item7_paragraph_v2.py SUMMARY_JSON TEMPLATE_MD OUT_MD"""
import datetime
import hashlib
import json
import re
import sys


def sig(x, n=2):
    s = "%.*g" % (n, x)
    mant, _, exp = s.partition("e")
    digits = mant.replace("-", "").replace(".", "").lstrip("0")
    if len(digits) < n:
        mant = mant + ("" if "." in mant else ".") + "0" * (n - len(digits))
    return mant + ("e" + exp if exp else "")


def main():
    p_json, p_tpl, p_out = sys.argv[1:4]
    raw = open(p_json, "rb").read(); J = json.loads(raw)
    tpl = open(p_tpl, encoding="utf-8").read()
    m = re.search(r'Printable: "(.*?)"\n', tpl, re.S)
    if not m:
        sys.exit("the template's paragraph was not found")
    par = m.group(1)
    rows = J["A_s4"]["rows"]
    at = lambda t: [r for r in rows if abs(r["t_over_rs"] - t) < 1e-9][0]
    V = []

    def put(br, val, src):
        V.append((br, val, src))
    r14 = at(14.0)
    put("[E_inv(14)/E(14)]", "%s percent" % sig(100.0 * r14["E_inv_weighted"] / r14["E"]), "A's checkpoint 14: E_inv %.6e over E %.6f" % (r14["E_inv_weighted"], r14["E"]))
    res = max(abs(r["energy_identity_residual_weighted"]) for r in rows)
    put("[largest residual]", sig(res), "A's checkpoints 0 to 40: the largest |energy identity residual| (weighted) %.6e" % res)
    mk = J["figure"]["marks"]["v2"]; r40 = at(40.0)
    put("[4.3417e-05]", "%.4e" % (mk["collar_flow_total"] / r40["E"]), "the net signed collar flow through 40 %.6e over E(40) %.6f" % (mk["collar_flow_total"], r40["E"]))
    arr = [r["L_over_LH_kappaT"] for r in rows if 36.0 - 1e-9 <= r["t_over_rs"] <= 40.0 + 1e-9 and r.get("L_over_LH_kappaT") is not None]
    put("[1.4 to 5.1]", "%s to %s" % (sig(min(arr)), sig(max(arr))), "A's checkpoints 36 to 40: L/L_H(kappa_T) %s" % ", ".join("%.4f" % x for x in arr))
    ow = [p["outside_weight_min"] for r in rows for p in r["per_channel"].values() if p.get("outside_weight_min") is not None and p.get("collected")]
    put("[0.9935 to 0.9999]", "%.4f to %.4f" % (min(ow), max(ow)), "the collected modes' outside weight per block, %d blocks with a collection" % len(ow))
    put("[0.00114744]", "%.6g" % r40["S_esc_weighted"], "A's checkpoint 40: S_col (weighted)")
    for n, br in (("c15", "[the pre-reflection range of L/L_H, the hold at 15]"), ("c30", "[the same, the hold at 30]")):
        hr = J["C"][n]["hold_readouts"]
        put(br, "%s to %s" % (sig(hr["L_clean_min"]), sig(hr["L_clean_max"])),
            "%s: L/L_H(kappa_T) over the clean window from T/r_s %s to the reflection clock %.3f: %.6g to %.6g (#56: %s)" % (
                n, hr["clean_window_from"], hr["T_refl"], hr["L_clean_min"], hr["L_clean_max"], json.dumps(J.get("drain_rule_ranges_against_56", {}).get(n))))
    out = par
    for br, val, _ in V:
        if br not in out:
            sys.exit("bracket %s not in the template" % br)
        out = out.replace(br, val, 1)
    left = re.findall(r"(?<!Tr)\[[^\]]*\]", out)     # Tr[...] is notation, not a bracket
    bad = [w for w in ("frozen", "prediction", "escaping entropy", "entirely above 25 T_H", "emission", "radiation", "escape fraction", "emptying time",
                       "mutual information") if out.lower().count(w) > par.lower().count(w)]     # a word the filling introduced (the template's own text is the lead's)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = ["# Item 7's finite-chain paragraph, version 2, filled (engine spec 41.4 (2); turn 49's FILE 3)", "",
         "Filled %s by code/page/fill_item7_paragraph_v2.py from %s (md5 %s) and the template %s; the template's text copied, the brackets replaced by the "
         "values below; prints only in PAGE_solved_history_v3 after the second route has read it." % (now, p_json, hashlib.md5(raw).hexdigest(), p_tpl), "",
         "Printable: \"%s\"" % out, "", "Brackets left for turn 50: %s" % (", ".join(left) if left else "none"), "",
         "Words check (sec. 4 and the template's 'Not printed', on the filled values): %s" % ("clean" if not bad else "FOUND " + ", ".join(bad)), "",
         "| bracket | value | selection |", "|---|---|---|"]
    L += ["| %s | %s | %s |" % (b, v, s) for b, v, s in V]
    open(p_out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("filled %d brackets; left %s; words %s" % (len(V), left, bad or "clean"))
    if bad:
        sys.exit(4)


if __name__ == "__main__":
    main()
