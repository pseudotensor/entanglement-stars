#!/usr/bin/env python
"""compose_certificate_v1.py (builder analyst; engine spec 29.2 and 29.11: "COLLAPSE_certificate_v1 is delivered when the production's
certificate lands (FILE 4 at its head; the four r_s 32 certificates, the five collar budgets, the two replay tables, the production's
certificate)"): assembles the delivery mechanically, nothing retyped: (1) the lead's statement COLLAPSE_certificate_statement_v1.md
verbatim; (2) the production's certificate (engine_certificate_v2 on COLLAPSE_records_v1.1's data); (3) the four r_s 32 certificates
(tau 16, 8, 4 and reading B at 4 r_s); (4) the five collar budgets from the stored data (engine_collar_budget_v1); (5) from the two replay
v4 tables (the production from T/rs 11.125, the tau 4 control from 0) the section "Engine 28.2 (S1)" with its sums line, verbatim. Each
part under a heading naming its source file. Args: --statement --production --rs32 (comma list) --budgets (comma list) --replays (comma
list) --out."""
import argparse
import os


def section(lines, start_prefix):
    i = [k for k, l in enumerate(lines) if l.startswith(start_prefix)]
    assert len(i) == 1, "%r: %d matches" % (start_prefix, len(i))
    return lines[i[0]:]


def main():
    ap = argparse.ArgumentParser()
    for k in ("statement", "production", "rs32", "budgets", "replays", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    out = [open(a.statement).read().rstrip("\n"), ""]
    out += ["## The production's certificate (%s)" % os.path.basename(a.production), "", open(a.production).read().rstrip("\n"), ""]
    for f in a.rs32.split(","):
        out += ["## The r_s 32 certificate (%s)" % os.path.basename(f), "", open(f).read().rstrip("\n"), ""]
    for f in a.budgets.split(","):
        out += ["## The collar budget from the stored data (%s)" % os.path.basename(f), "", open(f).read().rstrip("\n"), ""]
    for f in a.replays.split(","):
        L = open(f).read().rstrip("\n").split("\n")
        S = section(L, "### Engine 28.2 (S1)")
        out += ["## The replay's S1 columns (%s, its section %r verbatim)" % (os.path.basename(f), S[0][4:]), "", L[0], ""] + S[1:] + [""]
    open(a.out, "w").write("\n".join(out) + "\n")
    print("wrote %s: %d lines" % (a.out, len("\n".join(out).split("\n"))))


if __name__ == "__main__":
    main()
