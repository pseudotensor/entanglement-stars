#!/usr/bin/env python
"""compose_certificate_v3.py (builder analyst; engine spec 32.4 (1): "COLLAPSE_certificate_v1.2.md is composed when the production's and
the two complete members' v3 tables exist, in this order: the production's, the sigma/a 8 member's, the sigma/a 4 member's (the
production's history on the ladder's data file), the r_s 32 set's four, each with its validation line, the sigma/a 2 member's added as
version 1.3 if it completes; its head is these two lines, which the builder copies: ..."): assembles the delivery mechanically, nothing
retyped. (1) The head: the two lines read from the spec file (the quoted text after "its head is these two lines, which the builder
copies: ", split before "The lead's reading"), verbatim. (2) engine_certificate_v3's homogeneous validation (its --validate md),
verbatim, once. (3) Each table in the ruled order under a heading naming its source file, the heading followed by its validation line
(the code that printed the table, engine_certificate_v3, and the validation's largest relative quadrature/direct differences per eps,
read from the validation md), then the table verbatim. Args: --spec --validate --production --members (comma list: sigma/a 8, sigma/a 4)
--rs32 (comma list) --out."""
import argparse
import os
import re

KEY = "its head is these two lines, which the builder copies: \""


def head_lines(spec):
    t = open(spec).read()
    i = t.index("### 32.4")
    j = t.index("### 32.5")
    sec = t[i:j]
    k = sec.index(KEY) + len(KEY)
    m = re.match(r"(.*?)\" \(2\)", sec[k:], re.S)
    assert m, "the head's closing quote not found"
    q = m.group(1)
    p = q.index(" The lead's reading")
    return [q[:p], q[p + 1:]]


def validation_line(vmd):
    L = open(vmd).read().split("\n")
    out = []
    for l in L:
        m = re.match(r"- eps ([0-9.]+): .*the largest relative difference quadrature/direct over the nodes: tt ([^,]+), rr ([^.]+(?:\.[0-9e+-]+)?)\.$", l)
        if m:
            out.append("eps %s: tt %s, rr %s" % m.groups())
    assert out, "no eps lines in %s" % vmd
    return "- its validation line: printed by engine_certificate_v3.py, whose homogeneous validation (%s, above) gives the largest relative quadrature/direct difference over the nodes %s." % (os.path.basename(vmd), "; ".join(out))


def main():
    ap = argparse.ArgumentParser()
    for k in ("spec", "validate", "production", "members", "rs32", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    members = a.members.split(",")
    assert len(members) == 2, "two members: sigma/a 8, sigma/a 4"
    for f in [a.validate, a.production] + members + a.rs32.split(","):
        assert os.path.exists(f), "missing %s" % f
    vl = validation_line(a.validate)
    out = head_lines(a.spec) + [""]
    out += ["## The homogeneous validation of engine_certificate_v3 (%s, verbatim)" % os.path.basename(a.validate), "", open(a.validate).read().rstrip("\n"), ""]
    parts = [("The production's certificate", a.production),
             ("The ladder's sigma/a 8 member's certificate", members[0]),
             ("The ladder's sigma/a 4 member's certificate (the production's history on the ladder's data file)", members[1])]
    parts += [("The r_s 32 certificate", f) for f in a.rs32.split(",")]
    for title, f in parts:
        out += ["## %s (%s)" % (title, os.path.basename(f)), "", vl, "", open(f).read().rstrip("\n"), ""]
    open(a.out, "w").write("\n".join(out) + "\n")
    print("wrote %s: %d lines; head line 1 %d chars, line 2 %d chars" % (a.out, len("\n".join(out).split("\n")), len(out[0]), len(out[1])))


if __name__ == "__main__":
    main()
