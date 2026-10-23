#!/usr/bin/env python
"""compose_relabel_v1_21.py (builder analyst; engine spec 39.5 (3): "RELABEL_table_v1.21 closes v1.20's temperature rows (#47's and
#49's) and the Page v1a rows (version 2 carries them), keeps the HORIZON rows until version 2, adds 39.2 (4)'s row and a row on
ASYMPTOTICS v1.8's six words (closed by v1.9)"; 39.2 (4): "RELABEL_table_v1.21 carries one row on ITEM5 v10's sentence 6 (5 matched
clocks with a zero in one member only at T/r_s 21 read 29 to 40; the criterion line unchanged)"; the lane's instructions of 2026-10-17
from about 00:50 UTC, the last: ASYMPTOTICS v1.9 is held for the lead's turn 48, so version 1.20's row on ASYMPTOTICS v1.8's six words
stays open, everything else as ruled): version 1.20 with the title's source list, the version line and the sources line extended and
four rows added before the sources line: the temperature read as version 1.6 (delivered); version 1.20's three temperature rows closed;
version 1.20's five Page v1a rows closed (version 2 carries them); 39.2 (4)'s row. Every number in the rows is printed from its file
(cmp_v5_rs32_pair_final.md's member table, criterion line and matched-time table); checked, each a stop on failure: the rows to be
closed exist in version 1.20 as counted (3 temperature, 5 Page; the ASYMPTOTICS row (3) row and the HORIZON rows counted and kept); the
temperature rows' quoted stored texts are inside the old strings of the amendments' (a), (b), (d) and absent from version 1.6, whose
composition log prints every count 1; 39.2 (4)'s words read from the spec; item 5 version 10's sentence 6 carries "5 matched clocks holding a zero in one
member only" once and the criterion line's numbers (the formation spread, the largest R_t spread as a percentage at its clock, the
count of matched clocks with a zero in both) equal the pair file's ("unchanged"); the one-member clocks compressed into runs over the
pair file's own matched-time list equal 39.2's Facts; no pipe inside a cell; none of sec. 4's replaced words or further phrases in the
added rows. Refuses to overwrite. Args: --v120 --spec --item5 --pair --temp16 --temp16-log --amend16 --out --log."""
import argparse
import hashlib
import os
import re
import sys

REPLACED = ("frozen", "freez", "window", "prediction")
FORBIDDEN = ("toy model", "emergent gravity", "derived einstein", "derived from the fermions", "semiclassical", "mean-field", "mean field",
             "discretized", "an event horizon forms", "a singularity forms", "no black object", "obligation", "bonds keep their strength",
             "bonds always cross", "unitary completion", "rectangle", "allocation exponent", "uv complete", "near-maximal", "remnant",
             "page fraction", "page curve", "stationary luminosity", "static star", "exact for irrotational", "must radiate at t_h",
             "set by the masses", "0.99965", "(a/r)^2")


def cells(ln):
    return [x.strip() for x in ln.strip().strip("|").split("|")]


def fmt(x):
    return "%g" % x


def main():
    ap = argparse.ArgumentParser()
    for k in ("v120", "spec", "item5", "pair", "temp16", "temp16_log", "amend16", "out", "log"):
        ap.add_argument("--" + k.replace("_", "-"), required=True)
    a = ap.parse_args()
    for p in (a.out, a.log):
        if os.path.exists(p):
            sys.exit("refusing to overwrite %s" % p)
    md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()  # noqa: E731
    L = ["## compose_relabel_v1_21 (engine spec 39.5 (3), 39.2 (4)): the rows, the numbers printed from their files, the checks", "",
         "- inputs: %s" % "; ".join("%s (md5 %s)" % (p, md5(p)) for p in (a.v120, a.spec, a.item5, a.pair, a.temp16, a.temp16_log, a.amend16)), ""]

    def stop(msg):
        open(a.log, "x").write("\n".join(L) + "\n\nSTOPPED: %s\n" % msg)
        sys.exit(msg)

    t = open(a.v120).read()
    trows = [ln for ln in t.split("\n") if ln.startswith("| ") and not ln.startswith("| stored text") and not ln.startswith("|---")]
    temp = [ln for ln in trows if cells(ln)[0].startswith("TEMPERATURE_solved_exterior_v1.5.md (") and cells(ln)[1].endswith("until the lead's version 1.6")]
    page = [ln for ln in trows if cells(ln)[0].startswith("PAGE_solved_history_v1a.md") and "until PAGE_solved_history version 2" in cells(ln)[1]]
    asym = [ln for ln in trows if cells(ln)[0].startswith("ASYMPTOTICS_checks_v1.8.md row (3)'s carried record lines")]
    hor = [ln for ln in trows if cells(ln)[0].startswith("HORIZON_single_object_v1")]
    L.append("- version 1.20's rows: %d in all; on TEMPERATURE_solved_exterior_v1.5.md until the lead's version 1.6: %d; on PAGE_solved_history_v1a.md until its version 2: %d; on ASYMPTOTICS_checks_v1.8.md row (3)'s carried lines (kept open: version 1.9 held for the lead's turn 48): %d; on HORIZON_single_object_v1 (kept until HORIZON version 2): %d" % (
        len(trows), len(temp), len(page), len(asym), len(hor)))
    if (len(temp), len(page), len(asym)) != (3, 5, 1):
        stop("the rows to be closed are not 3, 5 and 1: %d, %d, %d" % (len(temp), len(page), len(asym)))
    # the temperature rows against the amendments and the composed version 1.6
    am = open(a.amend16).read().split("\n\n")
    par = {p_[:3]: p_ for p_ in am}
    olds = {k: re.findall(r'"([^"]*)"', par[k])[0] for k in ("(a)", "(b)", "(d)")}
    v16 = open(a.temp16).read()
    lg16 = open(a.temp16_log).read()
    if "- every count is 1: yes" not in lg16:
        stop("the version 1.6 log does not print every count 1")
    for ln, k in zip(temp, ("(a)", "(b)", "(d)")):
        q = re.findall(r'"([^"]*)"', cells(ln)[0])
        ok = len(q) == 1 and q[0] in olds[k] and v16.count(q[0]) == 0
        L.append("- the temperature row quoting \"%s\": inside the amendments' %s old string, %d times in the composed version 1.6: %s" % (q[0] if q else None, k, v16.count(q[0]) if q else -1, "closed by %s" % k if ok else "NOT AS READ"))
        if not ok:
            stop("a temperature row is not as read: %r" % cells(ln)[0][:120])
    # 39.2 (4)'s words
    sp = open(a.spec).read()
    s392 = sp.split("### 39.2 ", 1)[1].split("\n### ", 1)[0]
    r4 = re.search(r"\(4\) The pair's final reading is a record: (RELABEL_table_v1\.21 carries one row on ITEM5 v10's sentence 6 \((\d+) matched clocks with a zero in one member only at T/r_s (\d+) read (\d+) to (\d+); the criterion line unchanged\)); ITEM5 version 11 composes it with the capture clause when version 11's 300 r_s run lands \(one compose\)\.", s392)
    fx = re.search(r"and (\d+) matched clocks with a zero in one member only \(([^)]*)\)", s392)
    if not r4 or not fx:
        stop("39.2 (4)'s words or 39.2's Facts are not as read")
    L.append("- 39.2 (4), the lead's words: \"%s\"; 39.2's Facts: %s matched clocks with a zero in one member only (%s)" % (r4.group(1), fx.group(1), fx.group(2)))
    # the pair file
    pr = open(a.pair).read()
    mem = {}
    for ln in pr.split("\n"):
        m = re.match(r"^\| (\d) \| (engine_collapse_v\d+\.py) \| (\w+) \| (\d+) \| ([0-9.]+) \| ([0-9.]+) \|$", ln)
        if m:
            mem[m.group(1)] = (m.group(2), m.group(3), int(m.group(4)), m.group(6))
    cr = re.search(r"- 29\.1 across sigma/a 2, 4: formation spread ([0-9.]+) r_s \(sigma 2 at ([0-9.]+), sigma 4 at ([0-9.]+)\); largest R_t spread ([0-9.]+) \(max/min - 1\) over (\d+) matched times with a zero in every member \(at T/rs ([0-9.]+)\); (\d+) matched times with a zero in some but not all members \((.*?)\)\. Formation within", pr)
    if sorted(mem) != ["2", "4"] or not cr:
        stop("the pair file's member table or criterion line is not as read")
    one = [float(x) for x in re.findall(r"([0-9.]+) \(1 of 2\)", cr.group(8))]
    if len(one) != int(cr.group(7)) or cr.group(8).count("(") != len(one):
        stop("the one-member list does not hold its stated count")
    mt = pr.split("## The matched-time table", 1)[1].split("\n\n", 2)[1]
    times = [float(cells(ln)[0]) for ln in mt.split("\n") if ln.startswith("| ") and not ln.startswith("| T/rs") and not ln.startswith("|---")]
    runs, cur = [], []
    for x in times:
        if x in one:
            cur.append(x)
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    comp = ", ".join(fmt(r_[0]) if len(r_) == 1 else "%s to %s" % (fmt(r_[0]), fmt(r_[-1])) for r_ in runs)
    covered = sum(len(r_) for r_ in runs)
    L.append("- %s: members sigma/a 2 (%s, %s, %d records to T/r_s %s) and sigma/a 4 (%s, %s, %d records to T/r_s %s); the criterion line: formation spread %s r_s (%s and %s), largest R_t spread %s at T/rs %s over %s matched times with a zero in every member, %s matched times with a zero in one member only; the matched-time table %d clocks (%s to %s); the one-member clocks as runs of that table: %s (%d clocks)" % (
        a.pair, mem["2"][0], mem["2"][1], mem["2"][2], mem["2"][3], mem["4"][0], mem["4"][1], mem["4"][2], mem["4"][3], cr.group(1), cr.group(2), cr.group(3),
        cr.group(4), cr.group(6), cr.group(5), cr.group(7), len(times), fmt(times[0]), fmt(times[-1]), comp, covered))
    if covered != len(one) or comp != fx.group(2) or cr.group(7) != fx.group(1) or cr.group(7) != r4.group(4) or float(mem["2"][3]) != float(r4.group(5)) or float(mem["4"][3]) != float(r4.group(5)):
        stop("the pair file's one-member clocks %s (%s) differ from 39.2's %s (%s) or the members do not reach %s" % (comp, cr.group(7), fx.group(2), fx.group(1), r4.group(5)))
    L.append("- the runs equal 39.2's Facts (%s) and the count its (4) (%s); both members reach T/r_s %s: yes" % (fx.group(2), r4.group(4), r4.group(5)))
    # item 5 version 10's sentence 6
    it = open(a.item5).read()
    s6 = [ln for ln in it.split("\n") if ln.startswith('6. "')]
    stored = "%s matched clocks holding a zero in one member only" % r4.group(2)
    pct = "%.1f percent" % (100.0 * float(cr.group(4)))
    need = {"the stored clause": stored, "the formation spread": "a spread of %s r_s" % cr.group(1), "the largest R_t spread": "differ by up to %s" % pct,
            "its clock": "at T/r_s %s)" % cr.group(6), "the clocks with a zero in both": "at the %s matched clocks with a zero in both members" % cr.group(5),
            "the members' span": "both members certified through T/r_s %s" % r4.group(3)}
    got = {k: (s6[0].count(v) if len(s6) == 1 else -1) for k, v in need.items()}
    L.append("- %s sentence 6 (one line: %s): %s; the stored clause in the whole file %d" % (a.item5, len(s6) == 1, "; ".join("%s \"%s\" %d" % (k, need[k], got[k]) for k in need), it.count(stored)))
    if any(v != 1 for v in got.values()) or it.count(stored) != 1:
        stop("item 5 version 10's sentence 6 is not as read")
    L.append("- the criterion line unchanged: sentence 6's spread, largest R_t spread (%s from %s), its clock and the count with a zero in both equal the pair file's: yes" % (pct, cr.group(4)))
    rows = [
        "| TEMPERATURE_solved_exterior_v1.5.md | read as TEMPERATURE_solved_exterior_v1.6.md (engine 39.5 (3), turn 47: TEMPERATURE_solved_exterior_v1.6_amendments.md's (a) to (e) applied literally by compose_temperature_v1_6, each count 1; version 1.5 stays the record) |",
        "| version 1.20's three rows on TEMPERATURE_solved_exterior_v1.5.md (inbox #47, #49: the rounded extrema of kappa_0/kappa_T - 1, the source factor beside the other two factors, the condition carried from version 1.4) | closed: TEMPERATURE_solved_exterior_v1.6.md replaces those clauses ((a), (b), (d); engine 39.5 (3)); its (a) and (c) also carry compose_temperature_v1_5.md's two wording points |",
        "| version 1.20's five rows on PAGE_solved_history_v1a.md and ITEM7_fig_page_draft_v1a.png (inbox #48: the band, the work identity and the residuals, the final clock's mode ledger, the archive with the comparators and the inferred nu, the arrival marker and the plotted L) | closed: PAGE_solved_history version 2 carries them (engine 39.1 (3), 39.5 (3): #48's corrections as the Page builder carried them); PAGE_solved_history_v1a.md and its figure read as PAGE_solved_history version 2 and its figure when they land |",
        "| ITEM5_collapse_sentences_v10_text.md, sentence 6 (the r_s/a %s pair, both members through T/r_s %s): \"%s\" | the pair's final reading, a record (engine 39.2 (4)): with both members' records to T/r_s %s (%d and %d records), %s matched clocks hold a zero in one member only (T/r_s %s); the criterion line unchanged (formation spread %s r_s; the largest R_t spread %s at T/r_s %s over the %s matched clocks with a zero in both members) (cmp_v5_rs32_pair_final.md); ITEM5 version 11 composes it with the capture clause when capture version 11's 300 r_s run lands |" % (
            "32", r4.group(3), stored, mem["2"][3], mem["2"][2], mem["4"][2], cr.group(7), comp, cr.group(1), cr.group(4), cr.group(6), cr.group(5)),
    ]
    for r_ in rows:
        if r_.count("|") != 3:
            stop("a pipe inside a cell: %r" % r_[:120])
    lowr = "\n".join(rows).lower()
    hits = {w: lowr.count(w) for w in REPLACED + FORBIDDEN if w in lowr}
    L.append("- the added rows: %d; no pipe inside a cell; sec. 4's replaced words (%s) and further phrases (%s) in them: %s" % (len(rows), ", ".join(REPLACED), ", ".join(FORBIDDEN), "none" if not hits else "FOUND %s" % hits))
    if hits:
        stop("sec. 4 words in the added rows: %s" % hits)
    tm = re.search(r"^# RELABEL_table_v1\.20: (.*?)\(engine spec (.*?); 38\.2 to 38\.7; inbox #45 to #49\)$", t, re.M)
    if not tm:
        stop("version 1.20's title is not as read")
    t = t.replace(tm.group(0), "# RELABEL_table_v1.21: %s(engine spec %s; 38.2 to 38.7; inbox #45 to #49; 39.2 (4), 39.5 (3))" % (tm.group(1), tm.group(2)), 1)
    pm = "Version 1.20 = version 1.19 with "
    if t.count(pm) != 1:
        stop("version 1.20's version line is not as read")
    t = t.replace(pm, "Version 1.21 = version 1.20 with %d rows added (engine 39.2 (4), 39.5 (3), turn 47: the temperature read as version 1.6 and version 1.20's three temperature rows (inbox #47, #49) closed; version 1.20's five PAGE_solved_history_v1a rows (inbox #48) closed, version 2 carrying them; the HORIZON rows (inbox #45, #46) kept until HORIZON version 2; the r_s/a 32 pair's final reading on item 5 version 10's sentence 6; version 1.20's row on ASYMPTOTICS version 1.8's carried row (3) lines kept open, version 1.9 held for the lead's turn 48). %s" % (len(rows), pm), 1)
    si = t.index("\n- Sources: ")
    t = t[:si].rstrip("\n") + "\n" + "\n".join(rows) + "\n" + t[si:]
    sm = re.search(r"\n- Sources: ([^\n]*)", t)
    t = t.replace(sm.group(0), "\n- Sources: " + sm.group(1).rstrip(".") + "; solve/specs/SPEC_final4b_engine.md sec. 39 (39.1 (3), 39.2 (4), 39.5 (3)); deliveries/solve_a/TEMPERATURE_solved_exterior_v1.6_amendments.md, TEMPERATURE_solved_exterior_v1.6.md, ITEM5_collapse_sentences_v10_text.md; solve/results/obs/J3v7/compose_temperature_v1_6.md; solve/results/engine/LADrs32/cmp_v5_rs32_pair_final.md.", 1)
    n0 = len([ln for ln in open(a.v120).read().split("\n") if ln.startswith("| ")])
    n1 = len([ln for ln in t.split("\n") if ln.startswith("| ")])
    if n1 - n0 != len(rows):
        stop("the table lines grew by %d, not %d" % (n1 - n0, len(rows)))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "x").write(t)
    L += ["", "### the added rows", ""] + ["- " + r_.replace("|", "/") for r_ in rows]
    L += ["", "- table lines %d to %d; composed: %s (md5 %s) from %s; every number in the rows printed by the script from %s." % (
        n0, n1, a.out, hashlib.md5(t.encode()).hexdigest(), a.v120, a.pair)]
    open(a.log, "x").write("\n".join(L) + "\n")
    print("wrote %s (%d rows added) and %s" % (a.out, len(rows), a.log))


if __name__ == "__main__":
    main()
