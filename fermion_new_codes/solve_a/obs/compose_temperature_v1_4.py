#!/usr/bin/env python
"""compose_temperature_v1_4.py (builder analyst; engine spec 37.2 (1) and 37.9 S2: "compose_temperature_v1_4 composing
TEMPERATURE_solved_exterior_v1.4 from version 1.3 and FILE 2, the placeholders filled from the stored budget rows, delivered"): the lead's
version 1.3 statement with FILE 2's two amendments (TEMPERATURE_solved_exterior_v1.4_amendments.md) applied verbatim: (a) its insert placed
after its anchor in the last sentence; (b) its three replacements, the budget sums [b_max], [b_min] and the counts [n_max], [n_min] recomputed
from the stored J3' budget rows with the replaced term. The replaced term per record: d = |kappa_T - kappa_m|/kappa_T of the frozen record (the icloud check's form, inbox #34; equal to 1 - kappa_m/kappa_T where kappa_T > kappa_m) (the
applied-field root slope against the dense surface slope, the stored values in the J3' json's frozen entry, cross-checked against
obs_late_phase_v1.json's rows), in place of each row's shift_a; each row's linear total becomes total_linear - shift_a + d; a reading lies
inside when |T/T_H - 1| <= that total (the rule the stored within_budget flags follow, checked first on the stored totals). The budget sums
are the totals at the four frozen-stage readings version 1.3 names (the omega r_s 0.5 packet at holds 48 and 64 r_s at the maximum, the
omega r_s 0.25 packet at holds 48 and 64 at the minimum, both m/omega), rounded to two decimals, one value where they agree, "x to y"
otherwise. Self-checks before composing: version 1.3's own numbers (0.41, 0.84; 14 of 18, 12 of 12) reproduced from the stored rows, the
anchors found exactly once. Reported, not composed (version 1.3 stands verbatim there): the tau 4 records' counts (8 of 9 at 39.5, 3 of 9 at
39.75) under the same replacement. Every number printed by the script; nothing retyped. Args: --v13 --amend --max-json --min-json
--tau4-min-json --tau4-max-json --late-json --out --log."""
import argparse
import json
import os
import re
import sys


def rows_read(d):
    return [b for b in d["budget"] if b.get("T_over_T_H") is not None]


def inside(b, total):
    return abs(b["T_over_T_H"] - 1.0) <= total


def r2(x):
    return "%.2f" % x


def rng(vals):
    a, b = r2(min(vals)), r2(max(vals))
    return a if a == b else "%s to %s" % (a, b)


def main():
    ap = argparse.ArgumentParser()
    for k in ("v13", "amend", "max_json", "min_json", "tau4_min_json", "tau4_max_json", "late_json", "out", "log"):
        ap.add_argument("--" + k.replace("_", "-"), required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("refusing to overwrite %s" % a.out)
    L = ["## compose_temperature_v1_4 (engine spec 37.2 (1)): the replaced averaging term, the recomputed budget sums and counts, the composition's checks", ""]
    late = json.load(open(a.late_json))
    lrows = {r["T_over_rs"]: r for h in late["histories"] if h["label"] == "prod19" for r in h["rows"]}
    rec = {}
    for name, path, frows in (("maximum", a.max_json, (0.5, (48.0, 64.0))), ("minimum", a.min_json, (0.25, (48.0, 64.0)))):
        d = json.load(open(path)); fr = d["frozen"]; g = d["args"]["g"]
        dterm = abs(fr["kappa_T"] - fr["kappa_m"]) / fr["kappa_T"]
        lr = lrows.get(fr["T_over_rs"])
        rd = rows_read(d)
        flags_ok = all(bool(b["within_budget"]) == inside(b, b["total_linear"]) for b in rd)
        old_n = sum(1 for b in rd if inside(b, b["total_linear"]))
        sel = [b for b in d["budget"] if b["omega_rs"] == frows[0] and b["hold_over_rs"] in frows[1]]
        old_b = [b["total_linear"] for b in sel]
        new_tot = {id(b): b["total_linear"] - b["shift_a"] + dterm for b in d["budget"]}
        new_n = sum(1 for b in rd if inside(b, new_tot[id(b)]))
        new_b = [new_tot[id(b)] for b in sel]
        rec[name] = {"d": dterm, "old_b": rng(old_b), "new_b": rng(new_b), "old_n": "%d of %d" % (old_n, len(rd)), "new_n": "%d of %d" % (new_n, len(rd))}
        L += ["### the %s (%s; frozen at T/rs %.4f)" % (name, os.path.basename(path), fr["T_over_rs"]), "",
              "- kappa_T %.6e, kappa_m %.6e (the json's frozen entry); 4 G E kappa_T %.6f, 4 G E kappa_m %.6f (g %.2f, E %.6f); obs_late_phase_v1's row at T/rs %.4f: kappa_T %s, kappa_m %s" % (
                  fr["kappa_T"], fr["kappa_m"], 4 * g * fr["E"] * fr["kappa_T"], 4 * g * fr["E"] * fr["kappa_m"], g, fr["E"], fr["T_over_rs"],
                  ("%.6e" % lr["kappa_T"]) if lr else "-", ("%.6e" % lr["kappa_m"]) if lr and "kappa_m" in lr else "-"),
              "- the replaced term d = |kappa_T - kappa_m|/kappa_T = %.6f (%.1f percent); the stored shift_a it replaces %s" % (dterm, 100 * dterm, ", ".join(sorted(set("%.4e" % b["shift_a"] for b in d["budget"])))),
              "- the stored within_budget flags follow |T/T_H - 1| <= total_linear on every reading row: %s; readings %d" % ("yes" if flags_ok else "NO", len(rd)),
              "- the four frozen-stage readings' linear totals (omega r_s %g, holds %s, both m/omega): stored %s (rounded %s); replaced %s (rounded %s)" % (
                  frows[0], " and ".join("%g" % h for h in frows[1]), ", ".join("%.4f" % x for x in old_b), rng(old_b), ", ".join("%.4f" % x for x in new_b), rng(new_b)),
              "- readings inside: stored %s; replaced %s" % (rec[name]["old_n"], rec[name]["new_n"]), "",
              "| omega r_s | m/omega | hold/r_s | T/T_H | stored total | inside | replaced total | inside |", "|---|---|---|---|---|---|---|---|"]
        for b in rd:
            L.append("| %g | %g | %g | %.5f | %.4f | %s | %.4f | %s |" % (b["omega_rs"], b["m_over_omega"], b["hold_over_rs"], b["T_over_T_H"], b["total_linear"],
                                                                    "yes" if inside(b, b["total_linear"]) else "no", new_tot[id(b)], "yes" if inside(b, new_tot[id(b)]) else "no"))
        L.append("")
        if not flags_ok:
            sys.exit("the stored flags do not follow the rule at the %s" % name)
    ok13 = (rec["maximum"]["old_b"], rec["minimum"]["old_b"], rec["maximum"]["old_n"], rec["minimum"]["old_n"]) == ("0.41", "0.84", "14 of 18", "12 of 12")
    L.append("- version 1.3's numbers reproduced from the stored rows (0.41, 0.84; 14 of 18, 12 of 12): %s" % ("yes" if ok13 else "NO"))
    if not ok13:
        sys.exit("version 1.3's numbers are not reproduced")
    L += ["", "### reported, not composed: the tau 4 records under the same replacement (version 1.3 stands verbatim there)", ""]
    for path in (a.tau4_min_json, a.tau4_max_json):
        d = json.load(open(path)); fr = d["frozen"]; dterm = abs(fr["kappa_T"] - fr["kappa_m"]) / fr["kappa_T"]; rd = rows_read(d)
        L.append("- %s (frozen at T/rs %.4f): d = %.6f; readings inside: stored %d of %d, replaced %d of %d (each row's own stored partner terms kept)" % (
            os.path.basename(path), fr["T_over_rs"], dterm, sum(1 for b in rd if inside(b, b["total_linear"])), len(rd),
            sum(1 for b in rd if inside(b, b["total_linear"] - b["shift_a"] + dterm)), len(rd)))
    v13 = open(a.v13).read().split("\n")
    st = v13[2]; cond = v13[4]
    am = open(a.amend).read().split("\n\n")
    pa = [p for p in am if p.startswith("(a)")][0]; pb = [p for p in am if p.startswith("(b)")][0]; pc = [p for p in am if p.startswith("Condition:")][0]
    qa = re.findall(r'"([^"]*)"', pa); qb = re.findall(r'"([^"]*)"', pb)
    if len(qa) != 2 or len(qb) != 6:
        sys.exit("the amendments' quotes are not as expected: %d, %d" % (len(qa), len(qb)))
    anchor, ins = qa; old1, new1, old2, old3, new2, new3 = qb
    for s_ in (anchor, old1, old2, old3):
        if st.count(s_) != 1:
            sys.exit("anchor not found exactly once: %r" % s_[:80])
    new2f = new2.replace("[b_max]", rec["maximum"]["new_b"]).replace("[b_min]", rec["minimum"]["new_b"])
    new3f = new3.replace("[n_max]", rec["maximum"]["new_n"]).replace("[n_min]", rec["minimum"]["new_n"])
    st4 = st.replace(anchor, anchor + " " + ins).replace(old1, new1).replace(old2, new2f).replace(old3, new3f)
    if "[" in "".join((new2f, new3f)):
        sys.exit("a placeholder is left unfilled")
    c13, np13 = cond.split(" Not printed: ", 1)
    c14, np14 = pc.split(" Not printed: ", 1)
    head = ("# The temperature on the solved exterior: the one solved-exterior statement the paper prints, version 1.4 (engine spec 37.2; version 1.3 "
            "with the two amendments of TEMPERATURE_solved_exterior_v1.4_amendments.md, the lead's text of turn 45, composed by compose_temperature_v1_4: "
            "the insert and the replacements verbatim, the budget sums and counts recomputed from the stored budget rows with the replaced term; version 1.3 "
            "stays the record; the builder copies, never edits)")
    body = [head, "", st4, "",
            "%s Version 1.4's amendments: %s" % (c13, c14[len("Condition: "):]) + " Not printed: " + np13.rstrip() + " From the amendments: " + np14.strip(), ""]
    open(a.out, "w").write("\n".join(body))
    L += ["", "- composed: %s (the insert after its anchor, the three replacements; [b_max] %s, [b_min] %s, [n_max] %s, [n_min] %s)" % (
        a.out, rec["maximum"]["new_b"], rec["minimum"]["new_b"], rec["maximum"]["new_n"], rec["minimum"]["new_n"]),
          "- every number above is printed by the script from %s, %s, %s, %s, %s; the text from %s and %s." % (a.max_json, a.min_json, a.tau4_min_json, a.tau4_max_json, a.late_json, a.v13, a.amend)]
    open(a.log, "w").write("\n".join(L) + "\n")
    print("wrote %s and %s" % (a.out, a.log))


if __name__ == "__main__":
    main()
