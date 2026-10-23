#!/usr/bin/env python
"""compose_item5_v10.py (builder analyst; engine spec 38.3 (5), 38.5 (4), 38.6 (3) and 38.7: "compose_item5_v10 (FILE 2; the
consolidated text of versions 3 to 10 printed in full with the replacement map, the chain applied by script)"; FILE 2 =
ITEM5_collapse_sentences_v10.md, which holds only the replacements): the consolidated item 5 text under a new name,
ITEM5_collapse_sentences_v10_text.md, built by the chain each version's head states, applied to whole paragraphs and literal strings:
  v3: sentences 1 to 6 and the "Not printed" paragraph.
  v4: sentences 3, 4, 5 replaced by v4's.
  v5: sentences 3, 4 replaced by v5's; sentence 7 appended; its "Added to "Not printed"" paragraph appended.
  v6: sentence 3 replaced by v6's; v6's form for sentence 6 is a form for a later reading (applied in v10 (b), so not printed as a form);
      its "Added to" paragraph appended.
  v7: v6's capture clause, "until version 7 lands the capture clause reads ...", replaced by v7's capture paragraph: the clause is cut
      out of v6's "Added to" line (with its colon) and v7's paragraph becomes the capture element; its "Added to" paragraph appended.
  v8: sentence 3 "from "the sigma/a 2 member" to its end" replaced by v8's quoted text, its condition by v8's; [x], "its last certified
      record, copied by the builder at each composition" from COLLAPSE_ladder_v1.4.md's running member's table (the last T/rs of the
      leading run of records meeting the stop rule; the "two of them through the source continuation" checked against the table);
      its "Added to" paragraph appended.
  v9: the capture element replaced by v9's capture paragraph and its condition paragraph; its "Added to" paragraph appended.
  v10 (FILE 2): (a) the literal replacements on sentences 1 to 5 and 7 and the "Not printed" paragraphs (old and new from (a)'s quotes;
      "the same word elsewhere" = any remaining "frozen", read "held fixed"); (b) sentence 6's last clause (old and new from (b)'s
      quotes, once), (b)'s condition appended to sentence 6's; (c) the capture element replaced in full by (c)'s paragraph (its "(c) "
      dropped) and the condition part of FILE 2's last paragraph, whose "Added to "Not printed"" part is appended to the paragraphs.
Every step's count is printed in the replacement map and checked (1 for every paragraph or clause replaced; (a)'s counts printed as
found, the v9 clause's word noted as gone with the clause (c) replaces); a count other than expected stops the compose. Printed in full
in the log with the map; asserted: no [x] left; none of sec. 4's replaced words ("frozen", "freez", "window", "prediction", any case)
anywhere; none of the further sec. 4 phrases (FORBIDDEN) in the sentences and the capture element (in the "Not printed" paragraphs,
which name what is not printed, any such phrase is listed, not stopped on). Beside, as a reading aid and never an edit: the numbers of
(b) and (c) against their condition files. Refuses to overwrite. Args: --dir (the deliveries folder holding versions 3 to 10)
--ladder (COLLAPSE_ladder_v1.4.md) --rs32 (COLLAPSE_ladder_rs32_v1.md) --cap-final --cap-40 --out --log."""
import argparse
import os
import re
import sys

REPLACED = ("frozen", "freez", "window", "prediction")
FORBIDDEN = ("toy model", "emergent gravity", "derived einstein", "derived from the fermions", "semiclassical", "mean-field", "mean field",
             "discretized", "an event horizon forms", "a singularity forms", "no black object", "obligation", "bonds keep their strength",
             "bonds always cross", "unitary completion", "rectangle", "allocation exponent", "uv complete", "near-maximal", "remnant",
             "page fraction", "page curve", "stationary luminosity", "static star", "exact for irrotational", "must radiate at t_h",
             "set by the masses", "0.99965", "(a/r)^2")


def paras(path):
    return [p for p in open(path).read().split("\n\n") if p.strip()]


def numbered(ps):
    out = {}
    for p in ps:
        m = re.match(r"^(\d)\. ", p)
        if m:
            out[m.group(1)] = p.rstrip("\n")
    return out


def added(ps):
    a = [p.rstrip("\n") for p in ps if p.startswith('Added to "Not printed":')]
    if len(a) != 1:
        sys.exit("expected one Added-to paragraph, found %d" % len(a))
    return a[0]


class Chain:
    def __init__(self, log):
        self.S, self.cap, self.np, self.log = {}, [], [], log

    def note(self, ver, elem, op, count, expected):
        self.log.append("| %s | %s | %s | %s | %s |" % (ver, elem, op, count, expected))
        if expected is not None and count != expected:
            self.log.append("")
            self.log.append("STOPPED: %s %s counted %s, expected %s" % (ver, elem, count, expected))
            raise SystemExit("%s %s: count %s, expected %s" % (ver, elem, count, expected))


def last_certified(ladder_text):
    sec = ladder_text.split("## The sigma/a 2 member on engine_collapse_v32 with --nk-restarts 2", 1)
    if len(sec) != 2:
        sys.exit("COLLAPSE_ladder_v1.4.md: the running member's section not found")
    head = sec[1].split("\n", 1)[0]
    tab = sec[1].split("### The Newton-Krylov member sigma/a 2: its two residuals per record", 1)[1]
    last, n, stage2, broke = None, 0, 0, False
    for ln in tab.split("\n"):
        if not re.match(r"^\| \d", ln):
            if n and not ln.startswith("|"):
                break
            continue
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        n += 1
        if c[1] != "met":
            broke = True
            break
        last = c[0]
        if "stage 2" in c[5]:
            stage2 += 1
    return last, n, stage2, broke, head.strip()


def main():
    ap = argparse.ArgumentParser()
    for k in ("dir", "ladder", "rs32", "cap_final", "cap_40", "out", "log"):
        ap.add_argument("--" + k.replace("_", "-"), required=True)
    a = ap.parse_args()
    for p in (a.out, a.log):
        if os.path.exists(p):
            sys.exit("refusing to overwrite %s" % p)
    V = {v: paras(os.path.join(a.dir, "ITEM5_collapse_sentences_v%d.md" % v)) for v in range(3, 11)}
    M = ["## compose_item5_v10 (engine spec 38.7): the replacement map of the chain versions 3 to 10, with its counts; then the consolidated text in full", "",
         "| version | element | operation | count | expected |", "|---|---|---|---|---|"]
    C = Chain(M)
    # v3
    n3 = numbered(V[3])
    if sorted(n3) != ["1", "2", "3", "4", "5", "6"]:
        sys.exit("version 3's sentences are not 1 to 6: %s" % sorted(n3))
    C.S = dict(n3)
    base = [p.rstrip("\n") for p in V[3] if p.startswith("Not printed:")]
    C.note("v3", "sentences 1 to 6, Not printed", "taken whole", len(n3) + len(base), 7)
    C.np = list(base)
    # v4
    n4 = numbered(V[4])
    C.note("v4", "sentences 3, 4, 5", "replaced whole by version 4's", len(n4), 3)
    for k in ("3", "4", "5"):
        C.S[k] = n4[k]
    # v5
    n5 = numbered(V[5])
    C.note("v5", "sentences 3, 4; 7", "3, 4 replaced whole; 7 appended", len(n5), 3)
    for k in ("3", "4", "7"):
        C.S[k] = n5[k]
    C.np.append(added(V[5]))
    C.note("v5", "Not printed", "its Added-to paragraph appended", 1, 1)
    # v6
    n6 = numbered(V[6])
    C.note("v6", "sentence 3", "replaced whole by version 6's", len(n6), 1)
    C.S["3"] = n6["3"]
    form6 = [p for p in V[6] if p.startswith("6 (the form;")]
    C.note("v6", "sentence 6's form", "a form for the r_s/a 32 reading, applied as version 10 (b); not printed as a form", len(form6), 1)
    C.np.append(added(V[6]))
    C.note("v6", "Not printed", "its Added-to paragraph appended", 1, 1)
    # v7
    cap7 = [p.rstrip("\n") for p in V[7] if p.startswith("The capture clause")]
    m6 = re.search(r': until version 7 lands the capture clause reads "[^"]*"', C.np[2])
    C.note("v7", "version 6's capture clause", 'cut from version 6\'s Added-to line (": until version 7 lands the capture clause reads ..."), version 7\'s capture paragraph the capture element',
           (1 if m6 else 0) + len(cap7) - 1, 1)
    C.np[2] = C.np[2][:m6.start()] + C.np[2][m6.end():]
    C.cap = cap7
    C.np.append(added(V[7]))
    C.note("v7", "Not printed", "its Added-to paragraph appended", 1, 1)
    # v8
    p8 = [p for p in V[8] if p.startswith("Sentence 3, from")][0]
    q8 = re.findall(r'"([^"]*)"', p8)
    anchor, new3 = q8[0], q8[1]
    cond8 = p8.split('"' + new3 + '"', 1)[1].strip()
    s3 = C.S["3"]
    m3 = re.match(r'^3\. "(.*)" (Condition: .*)$', s3, re.S)
    if not m3:
        sys.exit("sentence 3 is not in the form 3. \"...\" Condition: ...")
    sent3, cond3_old = m3.group(1), m3.group(2)
    C.note("v8", "sentence 3", 'from "%s" to its end replaced by version 8\'s quote; its condition by version 8\'s' % anchor, sent3.count(anchor), 1)
    sent3 = sent3[:sent3.index(anchor)] + new3
    lad = open(a.ladder).read()
    xval, nrec, st2, broke, lhead = last_certified(lad)
    C.note("v8", "sentence 3's [x]", "%s (COLLAPSE_ladder_v1.4.md, the running member's table: %d records, every one meeting the stop rule%s, %d through the source continuation's stage 2; %s)" % (
        xval, nrec, "" if not broke else " up to a record that does not", st2, lhead), sent3.count("[x]"), 1)
    if st2 != 2:
        sys.exit("the table's stage-2 records are %d, not the two of version 8's sentence" % st2)
    sent3 = sent3.replace("[x]", xval)
    C.S["3"] = '3. "%s" %s' % (sent3, cond8)
    M.append("| v8 | sentence 3's condition | version 6's (%s) replaced by version 8's | 1 | 1 |" % cond3_old.replace("|", "/"))
    C.np.append(added(V[8]))
    C.note("v8", "Not printed", "its Added-to paragraph appended", 1, 1)
    # v9
    cap9 = [p.rstrip("\n") for p in V[9] if p.startswith("The capture clause") or p.startswith("Condition:")]
    C.note("v9", "the capture element", "replaced by version 9's capture paragraph and its condition paragraph", len(cap9), 2)
    C.cap = cap9
    C.np.append(added(V[9]))
    C.note("v9", "Not printed", "its Added-to paragraph appended", 1, 1)
    # v10 (FILE 2)
    f = {p[:3]: p.rstrip("\n") for p in V[10] if p[:3] in ("(a)", "(b)", "(c)")}
    last = [p.rstrip("\n") for p in V[10] if p.startswith("Condition:")]
    if len(last) != 1 or len(f) != 3:
        sys.exit("FILE 2 is not in the expected form")
    qa = re.findall(r'"([^"]*)"', f["(a)"])
    pairs_a = [(qa[0], qa[1]), (qa[2], qa[3])]
    if qa[4] != "held fixed":
        sys.exit("(a)'s third rule is not as read: %r" % qa[4])
    targets = ["1", "2", "3", "4", "5", "7"]
    for old, new in pairs_a:
        cs = sum(C.S[k].count(old) for k in targets); cn = sum(p.count(old) for p in C.np)
        c9 = sum(p.count(old) for p in C.cap)
        C.note("v10 (a)", "sentences 1 to 5, 7 and Not printed", '"%s" read "%s" (the clause (c) replaces holds it %d time(s), gone with that clause)' % (old, new, c9), cs + cn, None)
        for k in targets:
            C.S[k] = C.S[k].replace(old, new)
        C.np = [p.replace(old, new) for p in C.np]
    cs = sum(C.S[k].count("frozen") for k in targets); cn = sum(p.count("frozen") for p in C.np)
    C.note("v10 (a)", "sentences 1 to 5, 7 and Not printed", '"frozen" elsewhere read "held fixed"', cs + cn, None)
    for k in targets:
        C.S[k] = C.S[k].replace("frozen", "held fixed")
    C.np = [p.replace("frozen", "held fixed") for p in C.np]
    qb = re.findall(r'"([^"]*)"', f["(b)"])
    oldb, newb = qb[0], qb[1]
    condb = f["(b)"].split('"' + newb + '"', 1)[1].strip()
    if not condb.startswith(". Condition:"):
        sys.exit("(b)'s condition is not as read: %r" % condb[:40])
    condb = condb[2:]
    C.note("v10 (b)", "sentence 6's last clause", '"%s" read (b)\'s clause; (b)\'s condition appended to sentence 6\'s' % oldb, C.S["6"].count(oldb), 1)
    C.S["6"] = C.S["6"].replace(oldb, newb) + " " + condb
    cpar = f["(c)"][len("(c) "):]
    capc, addc = last[0].split(' Added to "Not printed": ', 1)
    C.note("v10 (c)", "the capture element", "replaced in full by (c)'s paragraph and FILE 2's condition", len(C.cap), 2)
    C.cap = [cpar, capc]
    C.np.append('Added to "Not printed": ' + addc)
    C.note("v10", "Not printed", "FILE 2's Added-to part appended", 1, 1)
    head = ("# Item 5, the black object of the model: the sentences the collapse histories support at turn 46, version 10, the consolidated text "
            "(versions 3 to 9 as their heads fix them and ITEM5_collapse_sentences_v10.md's (a) to (c), composed by compose_item5_v10, the "
            "replacement map and its counts in its log; sentence 3 in version 8's running form; the lead's text; the builder copies, never edits)")
    order = ["1", "2", "3", "4", "5", "6", "7"]
    body = [head, ""] + sum([[C.S[k], ""] for k in order], []) + sum([[p, ""] for p in C.cap], []) + sum([[p, ""] for p in C.np], [])
    text = "\n".join(body)
    sent_only = "\n".join([re.sub(r" Condition: .*$", "", C.S[k], flags=re.S) for k in order] + [C.cap[0]])
    if "[x]" in sent_only or "[y]" in sent_only:
        sys.exit("[x] or [y] left in a sentence or the capture clause")
    low = text.lower()
    rep = {w: low.count(w) for w in REPLACED if w in low}
    printed = "\n".join([C.S[k] for k in order] + C.cap).lower()
    forb = {w: printed.count(w) for w in FORBIDDEN if w in printed}
    npl = "\n".join(C.np).lower()
    forb_np = {w: npl.count(w) for w in FORBIDDEN if w in npl}
    M += ["", "### the checks", "",
          "- [x] left in the sentences and the capture clause: none (version 8's condition keeps its own description of [x]); sec. 4's replaced words (%s) anywhere in the composed text: %s" % (", ".join(REPLACED), "none" if not rep else "FOUND %s" % rep),
          "- the further sec. 4 phrases (%s) in the sentences and the capture element: %s; in the Not printed paragraphs (which name what is not printed): %s" % (
              ", ".join(FORBIDDEN), "none" if not forb else "FOUND %s" % forb, "none" if not forb_np else "listed: %s" % forb_np)]
    if rep or forb:
        open(a.log, "x").write("\n".join(M) + "\n\nSTOPPED: sec. 4 words found\n")
        sys.exit("sec. 4 words found: %s %s" % (rep, forb))
    # reading aids: (b) and (c) against their condition files (never an edit)
    rs = open(a.rs32).read()
    crit = [ln for ln in rs.split("\n") if ln.startswith("- 29.1 across sigma/a 2, 4:")]
    fin = open(a.cap_final).read(); c40 = open(a.cap_40).read()
    rows = []
    for ln in fin.split("\n"):
        if ln.startswith("| solved slice") or ln.startswith("| control"):
            c = [x.strip() for x in ln.strip().strip("|").split("|")]
            rows.append("%s %s: C_m %s, departure %s percent, arrival %s, smallest before it %s" % (c[0][:7], c[1][-10:], c[3], c[6], c[13], c[14]))
    rows40 = []
    for ln in c40.split("\n"):
        if ln.startswith("| solved slice") or ln.startswith("| control"):
            c = [x.strip() for x in ln.strip().strip("|").split("|")]
            rows40.append("%s %s: %s" % (c[0][:7], c[1][-10:], c[12]))
    M += ["", "### reading aids (the condition files' own lines beside (b) and (c); the lead's numbers are copied, never edited)", "",
          "- %s: %s" % (a.rs32, crit[0] if crit else "(criterion line not found)"),
          "- %s: %s" % (a.cap_final, "; ".join(rows)),
          "- %s (sigma/(pi b_c^2) at the last reading, both phases' files): %s" % (a.cap_40, "; ".join(rows40)),
          "", "### the consolidated text in full", "", text]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "x").write(text)
    open(a.log, "x").write("\n".join(M) + "\n")
    print("wrote %s and %s" % (a.out, a.log))


if __name__ == "__main__":
    main()
