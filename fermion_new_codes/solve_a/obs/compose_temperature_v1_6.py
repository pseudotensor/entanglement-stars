#!/usr/bin/env python
"""compose_temperature_v1_6.py (builder analyst; engine spec 39.5 (3): "FILE 1: TEMPERATURE version 1.6 by compose_temperature_v1_6
(new) from version 1.5, the five literal replacements with counts"; the lane's instruction of 2026-10-17 about 00:50 UTC: "(a) to (e)
applied literally, each count printed and a count other than 1 stopping the compose, the composed file asserted free of sec. 4's words"):
the delivered version 1.5 (head, statement, condition paragraph) with TEMPERATURE_solved_exterior_v1.6_amendments.md applied literally,
every old and new string taken from the amendments' own quotes, never retyped: (a), (b), (c) in the statement; (d) in the condition;
(e) the head's parenthesis from its first quote to its second (both ends included) replaced by its third, and its fourth quote
("version 1.5") in the head's title (the head before that parenthesis) replaced by its fifth. Every count is printed (head, statement,
condition, total) and must be 1; any other count stops the compose. The condition paragraph then carries the amendments' own condition
and Not printed list as versions 1.4 and 1.5 carried theirs ("Version 1.6's amendments: ..." before the Not printed list, "From the
version 1.6 amendments: ..." at its end). Checks printed beside the composition, each a stop on failure: the numbers the lead's new
texts carry against the stored files the amendments' condition names (TEMPERATURE_kappa0_table_v1.md: the five records' kappa_0/kappa_T
- 1 extrema at four significant figures and at two, the production's three factors at its maximum and minimum records to two decimals;
compose_temperature_v1_4.md: the maximum's replaced linear totals, distinct, and their two-decimal roundings apart); the Not printed
items absent from the statement ("sits in the source factor", "to within between") and the replaced clauses of (c) and (d) absent from
the whole file; none of requirements_new sec. 4's replaced words ("frozen", "freez", "window", "prediction", any case) and none of the
further sec. 4 phrases (FORBIDDEN) anywhere in the composed file. Refuses to overwrite. Args: --v15 --amend --kappa0 --log14 --out --log."""
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


def main():
    ap = argparse.ArgumentParser()
    for k in ("v15", "amend", "kappa0", "log14", "out", "log"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    for p in (a.out, a.log):
        if os.path.exists(p):
            sys.exit("refusing to overwrite %s" % p)
    L = ["## compose_temperature_v1_6 (engine spec 39.5 (3)): TEMPERATURE_solved_exterior_v1.6_amendments.md applied to version 1.5; the counts; the checks", "",
         "- inputs: version 1.5 %s (md5 %s); the amendments %s (md5 %s); %s (md5 %s); %s (md5 %s)" % (
             a.v15, hashlib.md5(open(a.v15, "rb").read()).hexdigest(), a.amend, hashlib.md5(open(a.amend, "rb").read()).hexdigest(),
             a.kappa0, hashlib.md5(open(a.kappa0, "rb").read()).hexdigest(), a.log14, hashlib.md5(open(a.log14, "rb").read()).hexdigest()), ""]

    def stop(msg):
        open(a.log, "x").write("\n".join(L) + "\n\nSTOPPED: %s\n" % msg)
        sys.exit(msg)

    v15 = open(a.v15).read().split("\n")
    if not (v15[0].startswith("# ") and v15[1] == "" and v15[2].startswith('"') and v15[3] == "" and v15[4].startswith("Condition: ")):
        stop("version 1.5 is not in the expected form (head, statement, condition)")
    head, st, cond = v15[0], v15[2], v15[4]
    tail = v15[5:]
    am = open(a.amend).read().split("\n\n")
    par = {p[:3]: p for p in am}
    for k in ("(a)", "(b)", "(c)", "(d)", "(e)", "Con"):
        if k not in par:
            stop("the amendments have no paragraph %s" % k)
    q = {k: re.findall(r'"([^"]*)"', par[k]) for k in ("(a)", "(b)", "(c)", "(d)", "(e)")}
    shape = {k: len(v) for k, v in q.items()}
    L.append("- the amendments' quotes per paragraph: %s" % ", ".join("%s %d" % kv for kv in shape.items()))
    if shape != {"(a)": 2, "(b)": 2, "(c)": 2, "(d)": 2, "(e)": 5}:
        stop("the amendments' quotes are not as read: %s" % shape)
    if not par["(d)"].startswith('(d) In the Condition, "'):
        stop("(d) does not name the Condition")
    L += ["", "### the replacements (old and new verbatim from the amendments' quotes)", "",
          "| item | where | old | new | count: head | statement | condition | total |", "|---|---|---|---|---|---|---|---|"]

    def counts(s):
        return head.count(s), st.count(s), cond.count(s)

    # (a), (b), (c): the statement
    for k in ("(a)", "(b)", "(c)"):
        old, new = q[k]
        ch, cs, cc = counts(old)
        L.append("| %s | the statement | %s | %s | %d | %d | %d | %d |" % (k, old, new, ch, cs, cc, ch + cs + cc))
        if cs != 1 or ch + cc != 0:
            stop("%s counted %d (head %d, statement %d, condition %d), expected 1 in the statement" % (k, ch + cs + cc, ch, cs, cc))
        st = st.replace(old, new)
    # (d): the condition
    old, new = q["(d)"]
    ch, cs, cc = counts(old)
    L.append("| (d) | the condition | %s | %s | %d | %d | %d | %d |" % (old, new, ch, cs, cc, ch + cs + cc))
    if cc != 1 or ch + cs != 0:
        stop("(d) counted %d (head %d, statement %d, condition %d), expected 1 in the condition" % (ch + cs + cc, ch, cs, cc))
    cond = cond.replace(old, new)
    # (e): the head's parenthesis, then the head's title
    e_from, e_to, e_new, t_old, t_new = q["(e)"]
    c_from, c_to = counts(e_from), counts(e_to)
    L.append("| (e) 1, the parenthesis' first end | the head | %s | (the span replaced, both ends included) | %d | %d | %d | %d |" % (e_from, c_from[0], c_from[1], c_from[2], sum(c_from)))
    L.append("| (e) 1, the parenthesis' second end | the head | %s | (the span replaced, both ends included) | %d | %d | %d | %d |" % (e_to, c_to[0], c_to[1], c_to[2], sum(c_to)))
    if c_from != (1, 0, 0) or c_to != (1, 0, 0):
        stop("(e)'s span ends counted %s and %s, expected 1 each in the head only" % (c_from, c_to))
    i0 = head.index(e_from); i1 = head.index(e_to) + len(e_to)
    if i1 <= i0:
        stop("(e)'s second end precedes its first")
    span_old = head[i0:i1]
    title = head[:i0]
    ct = title.count(t_old)
    L.append("| (e) 1 | the head | %s | %s | (the span above) | - | - | 1 |" % (span_old, e_new))
    L.append("| (e) 2 | the head's title (the head before the parenthesis) | %s | %s | %d (title) | - | - | %d |" % (t_old, t_new, ct, ct))
    if ct != 1:
        stop("(e)'s title replacement counted %d in the head's title, expected 1" % ct)
    head = title.replace(t_old, t_new) + e_new + head[i1:]
    L.append("- every count is 1: yes (a), (b), (c) in the statement, (d) in the condition, (e)'s two span ends and its title replacement in the head")
    # the condition paragraph carries the amendments' condition and Not printed list (as versions 1.4 and 1.5 carried theirs)
    if cond.count(" Not printed: ") != 1:
        stop("version 1.5's condition does not hold one Not printed list")
    c15, np15 = cond.split(" Not printed: ", 1)
    cf = par["Con"]
    if not cf.startswith("Condition: ") or cf.count(" Not printed: ") != 1:
        stop("the amendments' condition is not as read")
    c16, np16 = cf.split(" Not printed: ", 1)
    c16 = c16[len("Condition: "):]
    cond16 = "%s Version 1.6's amendments: %s Not printed: %s From the version 1.6 amendments: %s" % (c15, c16.strip(), np15.rstrip(), np16.strip())
    L.append("- the condition: version 1.5's with (d) applied, then \"Version 1.6's amendments: \" and the amendments' condition before the Not printed list, and \"From the version 1.6 amendments: \" with the amendments' Not printed list at its end (versions 1.4 and 1.5 carried theirs so)")
    # the checks of the lead's numbers against the stored files
    L += ["", "### the lead's numbers against the stored files the amendments' condition names (a difference stops the compose)", ""]
    k0 = open(a.kappa0).read()
    rows = [ln for ln in k0.split("\n") if ln.startswith("| prod19 ") or ln.startswith("| tau4 ")]
    if len(rows) != 5:
        stop("the kappa_0 table does not hold five records: %d" % len(rows))
    devs, fac = [], {}
    for ln in rows:
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        devs.append(float(c[12]))
        if c[0] == "prod19":
            fac[c[1]] = [float(x) for x in c[13].split(";")]
    if sorted(fac) != ["35.3750", "37.7500"] or "T/r_s 35.375" not in v15[2] or "the last minimum at 37.75" not in v15[2]:
        stop("the production's maximum and minimum records are not 35.375 and 37.75 in both files: %s" % sorted(fac))
    fmax, fmin = fac["35.3750"], fac["37.7500"]
    na = re.search(r"between (\S+) and (\S+) on the five records \(rounded extrema of the table's (\S+) and (\S+),", q["(a)"][1])
    want_a = ("%.1e" % min(devs), "%.1e" % max(devs), "%.3e" % min(devs), "%.3e" % max(devs))
    ok_a = na is not None and na.groups() == want_a
    L.append("- (a): %s's kappa_0/kappa_T - 1 on the five records %s; smallest %.3e, largest %.3e; at two significant figures %s and %s; the new text carries %s: %s" % (
        a.kappa0, ", ".join("%.3e" % x for x in devs), min(devs), max(devs), want_a[0], want_a[1], na.groups() if na else None, "equal" if ok_a else "DIFFERENT"))
    nb = re.search(r"\((\S+) to (\S+), beside 2 G E/R_H (\S+) to (\S+) and N_H (\S+) to (\S+), which also change;", q["(b)"][1])
    want_b = ("%.2f" % fmax[2], "%.2f" % fmin[2], "%.2f" % fmax[0], "%.2f" % fmin[0], "%.2f" % fmax[1], "%.2f" % fmin[1])
    ok_b = nb is not None and nb.groups() == want_b
    L.append("- (b): %s's three factors (2 g E/R_H; N_H; the source factor) at the maximum (T/r_s 35.3750) %s and at the minimum (37.7500) %s; to two decimals: the source factor %s to %s, 2 G E/R_H %s to %s, N_H %s to %s; the new text carries %s: %s" % (
        a.kappa0, "; ".join("%.6f" % x for x in fmax), "; ".join("%.6f" % x for x in fmin), want_b[0], want_b[1], want_b[2], want_b[3], want_b[4], want_b[5],
        nb.groups() if nb else None, "equal" if ok_b else "DIFFERENT"))
    lg = open(a.log14).read()
    smax = [x for x in re.split(r"\n### ", lg) if x.startswith("the maximum")][0]
    rep = re.search(r"; replaced ([0-9., ]+) \(rounded ([0-9.]+(?: to [0-9.]+)?)\)", smax)
    vals = [x.strip() for x in rep.group(1).split(",")]
    distinct = list(dict.fromkeys(vals))
    nc = re.search(r"the maximum, (\S+) and (\S+), rounding apart at the second decimal", q["(c)"][1])
    two = ["%.2f" % float(x) for x in distinct]
    ok_c = nc is not None and list(nc.groups()) == distinct and len(distinct) == 2 and two[0] != two[1] and two[0][:3] == two[1][:3]
    L.append("- (c): %s, the maximum's replaced linear totals %s (distinct: %s; rounded there %s); to two decimals %s, apart at the second decimal; the new text carries %s: %s" % (
        a.log14, ", ".join(vals), " and ".join(distinct), rep.group(2), " and ".join(two), nc.groups() if nc else None, "equal" if ok_c else "DIFFERENT"))
    hdr = k0.split("\n")[0]
    ok_d = ("obs_p26_slope_ratios_v3.py" in hdr) and ("Version 2 (" in hdr) and ("TEMPERATURE_kappa0_table_v1.md" in q["(d)"][1])
    L.append("- (d): %s's header names obs_p26_slope_ratios_v3.py as the printing script and records version 2's display (\"Version 2 (\"): %s" % (a.kappa0, "yes" if ok_d else "NO"))
    if not (ok_a and ok_b and ok_c and ok_d):
        stop("a number of the lead's new texts differs from its stored file")
    body = "\n".join([head, "", st, "", cond16] + tail)
    if not body.endswith("\n"):
        body += "\n"
    # the checks of the composed text
    np_stmt = re.findall(r'"([^"]*)"', np16)
    gone = {"(c)'s replaced clause": q["(c)"][0], "(d)'s replaced clause (the pending clause)": q["(d)"][0], "(a)'s replaced clause": q["(a)"][0], "(b)'s replaced clause": q["(b)"][0]}
    L += ["", "### the composed text's checks", ""]
    L.append("- the amendments' Not printed quotes absent from the statement: %s" % "; ".join("\"%s\" %d in the statement (%d in the whole file: the condition's Not printed list)" % (x, st.count(x), body.count(x)) for x in np_stmt))
    L.append("- the replaced clauses absent from the whole file: %s" % "; ".join("%s %d" % (k, body.count(v)) for k, v in gone.items()))
    if any(st.count(x) for x in np_stmt) or any(body.count(v) for v in gone.values()):
        stop("a Not printed item or a replaced clause remains")
    low = body.lower()
    hits = {w: low.count(w) for w in REPLACED + FORBIDDEN if w in low}
    L.append("- sec. 4's replaced words (%s) and the further sec. 4 phrases (%s) in the composed file, any case: %s" % (", ".join(REPLACED), ", ".join(FORBIDDEN), "none" if not hits else "FOUND %s" % hits))
    if hits:
        stop("sec. 4 words found: %s" % hits)
    L.append("- the composed head: %s" % head)
    changed = [i + 1 for i, (x, y) in enumerate(zip(v15, body.split("\n"))) if x != y]
    L.append("- lines changed against version 1.5: %s (the head, the statement, the condition); lines: version 1.5 %d, version 1.6 %d" % (changed, len(v15), len(body.split("\n"))))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "x").write(body)
    L += ["- composed: %s (md5 %s)" % (a.out, hashlib.md5(body.encode()).hexdigest()),
          "- every number above is printed by the script from %s and %s; the text from %s and %s; nothing retyped." % (a.kappa0, a.log14, a.v15, a.amend)]
    open(a.log, "x").write("\n".join(L) + "\n")
    print("wrote %s and %s" % (a.out, a.log))


if __name__ == "__main__":
    main()
