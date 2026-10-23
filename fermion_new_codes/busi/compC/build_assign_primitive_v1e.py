#!/usr/bin/env python
"""build_assign_primitive_v1e.py (busi lane builder, 20 Oct; ReviewA_busi_open_q12_v1.md section 1, the late-step toy toy_late_step_v1.py):
assign_primitive_v1e.py over assign_primitive_v1d.py (md5 093e87879198cce4114a6229c1ccb11a) in the lane's patch form: exact replacements, each
OLD occurring once; every new line carries "# v1e"; --reverse restores v1d byte for byte; the source md5 is checked; an output is never overwritten.

The defect (the toy's variant X on v1d): a passage completed inside the chord's last piece has two in-piece samples (the clean point before, s_p,
and s = 1). v1c's anchor sample (the anchor's splitting, appended when fewer than 3 samples and the anchor lies before s_p) was appended BEFORE
v1d's extension test, so whenever the anchor is the chord's start (a lone late passage: every passage whose gap minimum falls late in the step)
the count reached 3 with the far s = 0 sample and the extension to s = 2 s_c - s_p was never taken: the fit went through (0, s_p, 1) (variant X at
level 1: vertex 1.125, minimum 0.01375 A, not bracketed, unresolved; at level 4 (0, 0.9375, 1): s* 0.9348, gap_min^2 0.01755 A, alpha_s 1.82
against the chord's 2.4). The patch moves the anchor sample after the extension: a last-piece passage takes its third sample on the extended chord;
the anchor sample remains the fallback when the extension gives none (refused by the oracle, or the continuation unclear there), as in v1c.
Nothing else changes: passages with a point after them, or whose anchor is s_p, follow v1d's statements exactly.

Usage: python build_assign_primitive_v1e.py assign_primitive_v1d.py [assign_primitive_v1e.py] [--check]
       python build_assign_primitive_v1e.py assign_primitive_v1e.py restored.py --reverse [--against assign_primitive_v1d.py]
"""
import argparse
import hashlib
import os
import sys

MARK = "# v1e"
SOURCE_MD5 = "093e87879198cce4114a6229c1ccb11a"

OLD_DOC = "refuses s > 1 (the bound's jump_fit of zero entered the resolved jumps as zero: RUN_W2_v4 3.0's X at L = 1).\n"
NEW_DOC = OLD_DOC + (
    "v1e over v1d (ReviewA_busi_open_q12_v1 section 1; the late-step toy toy_late_step_v1): the anchor's sample of a passage completed inside   # v1e\n"
    "the chord's last piece is taken after the extension, not before it, so a lone late passage (the anchor at the chord's start) is fitted through   # v1e\n"
    "s_p, 1 and the extended point instead of through s = 0; the anchor's sample stays the fallback when the extension gives no sample.   # v1e\n")
OLD_VER = 'VERSION = "assign_primitive_v1d"\n'
NEW_VER = 'VERSION = "assign_primitive_v1e"   # v1e\n'
OLD_A = ('                if len(set(s_ for s_, _d in samples)) < 3 and ev.get("sample_a") is not None:\n'
         '                    samples.append(ev["sample_a"])\n'
         '                if len(set(s_ for s_, _d in samples)) < 3 and ev["kind"] == "passage" and not (s_now > ev["s_c"]):')
NEW_A = ('                # v1e: the anchor\'s sample (v1c) is appended after the chord\'s extension below, so it never pre-empts it\n'
         '                if len(set(s_ for s_, _d in samples)) < 3 and ev["kind"] == "passage" and not (s_now > ev["s_c"]):')
OLD_B = ('                            ev["flags"].append("the chord\'s extension to s=%.4f: the pair\'s continuation unclear there (both on index %d): the two-sample bound" % (s_x, kx))   # v1d\n')
NEW_B = OLD_B + (
    '                if len(set(s_ for s_, _d in samples)) < 3 and ev.get("sample_a") is not None:   # v1e: v1c\'s anchor sample, the fallback when the extension gave none\n'
    '                    samples.append(ev["sample_a"])   # v1e\n')


class BuildStop(Exception):
    pass


def replacements():
    return [("doc", OLD_DOC, NEW_DOC), ("version", OLD_VER, NEW_VER), ("anchor sample out", OLD_A, NEW_A), ("anchor sample after the extension", OLD_B, NEW_B)]


def apply(text, reps, reverse=False):
    bad = [(n, text.count(nw if reverse else o)) for n, o, nw in reps if text.count(nw if reverse else o) != 1]
    if bad:
        raise BuildStop("; ".join("%s: %s occurs %d time(s), not once" % (n, "NEW" if reverse else "OLD", c) for n, c in bad))
    for n, o, nw in (reversed(reps) if reverse else reps):
        a, b = (nw, o) if reverse else (o, nw)
        text = text.replace(a, b, 1)
    return text


def marks_check(src, out):
    old = set(src.split("\n"))
    return [(q, l) for q, l in enumerate(out.split("\n"), 1) if l.strip() and l not in old and MARK not in l]


def md5(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def read(p):
    with open(p, "rb") as f:
        return f.read().decode("utf-8")


def write_new(p, text):
    if os.path.exists(p):
        raise BuildStop("the output %s exists; it is never overwritten" % p)
    with open(p, "xb") as f:
        f.write(text.encode("utf-8"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="assign_primitive_v1e over assign_primitive_v1d (exact replacements; fails closed)")
    ap.add_argument("source"); ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--check", action="store_true"); ap.add_argument("--reverse", action="store_true"); ap.add_argument("--against", default=None)
    a = ap.parse_args(argv)
    src = read(a.source); reps = replacements()
    try:
        if a.reverse:
            back = apply(src, reps, reverse=True)
            out = a.out or (os.path.splitext(a.source)[0] + ".restored.py")
            write_new(out, back)
            print("reverse: %s (md5 %s) -> %s (md5 %s)" % (a.source, md5(src), out, md5(back)))
            if a.against:
                ok = read(a.against) == back
                print("against %s: %s" % (a.against, "BYTE FOR BYTE" if ok else "DIFFERS"))
                return 0 if ok else 2
            return 0
        if md5(src) != SOURCE_MD5:
            print("STOP: the source md5 is %s, not v1d's %s" % (md5(src), SOURCE_MD5)); return 2
        out_text = apply(src, reps)
    except BuildStop as stop:
        print("STOP: %s; nothing written" % stop); return 2
    marks = marks_check(src, out_text)
    if marks:
        print("STOP: %d new line(s) lack %r: %s" % (len(marks), MARK, marks[:5])); return 2
    if apply(out_text, reps, reverse=True) != src:
        print("STOP: the reverse does not restore v1d byte for byte"); return 2
    compile(out_text, "assign_primitive_v1e.py", "exec")
    print("build: %s (md5 %s) -> v1e (md5 %s); %d replacements; %d lines carry %r; marks ok; reverse byte for byte; compiled" % (
        a.source, md5(src), md5(out_text), len(reps), sum(1 for l in out_text.split("\n") if MARK in l), MARK))
    if a.check:
        print("CHECK PASS (nothing written)"); return 0
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.source)), "assign_primitive_v1e.py")
    try:
        write_new(out, out_text)
    except BuildStop as stop:
        print("STOP: %s" % stop); return 2
    print("written: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
