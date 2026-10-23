#!/usr/bin/env python
"""build_assign_primitive_v1f.py (busi lane builder, 20 Oct; REVIEWA_C64_toy_readout_v1.md section 1, the ruling on the late-step toy):
assign_primitive_v1f.py over assign_primitive_v1e.py (md5 74be7cb13509878d3c5c9b810bc733c8) in the lane's patch form: exact replacements, each
OLD occurring once; every new line carries "# v1f"; --reverse restores v1e byte for byte; the source md5 is checked; an output is never overwritten.

The change (the ruling's text, OLD -> NEW, nothing else besides the VERSION line and one docstring line): v1e's fallback, v1c's anchor sample
appended when the extension gave none, is kept only for a passage with a point after it whose continuation was unclear there (s_now > s_c); a
passage completed inside the chord's last piece whose extension was refused by the oracle or unclear keeps its two samples, and _classify's
existing branch (n_samples == 2 and kind == "passage") hands it to fit_pair_grid: the two-sample grid bound the v1d docstring states (gap_min 0,
the band [0, Delta_grid^2], decided only when the band clears c). A last-piece passage with its extended sample is fitted as on v1e.

Usage: python build_assign_primitive_v1f.py assign_primitive_v1e.py [assign_primitive_v1f.py] [--check]
       python build_assign_primitive_v1f.py assign_primitive_v1f.py restored.py --reverse [--against assign_primitive_v1e.py]
"""
import argparse
import hashlib
import os
import sys

MARK = "# v1f"
SOURCE_MD5 = "74be7cb13509878d3c5c9b810bc733c8"

OLD_DOC = "s_p, 1 and the extended point instead of through s = 0; the anchor's sample stays the fallback when the extension gives no sample.   # v1e\n"
NEW_DOC = OLD_DOC + (
    "v1f over v1e (REVIEWA_C64_toy_readout_v1 section 1): the anchor's sample only for a passage with a point after it; a last-piece passage whose extension gave no sample takes the two-sample grid bound (fit_pair_grid), as v1d states.   # v1f\n")
OLD_VER = 'VERSION = "assign_primitive_v1e"   # v1e\n'
NEW_VER = 'VERSION = "assign_primitive_v1f"   # v1f\n'
OLD_A = ('                if len(set(s_ for s_, _d in samples)) < 3 and ev.get("sample_a") is not None:   # v1e: v1c\'s anchor sample, the fallback when the extension gave none\n'
         '                    samples.append(ev["sample_a"])   # v1e\n')
NEW_A = ('                if len(set(s_ for s_, _d in samples)) < 3 and ev.get("sample_a") is not None and (s_now > ev["s_c"]):   # v1f: v1c\'s anchor sample only for a passage with a point after it whose continuation was unclear there; a passage completed inside the chord\'s last piece whose extension gave no sample keeps its two samples and takes the two-sample grid bound (fit_pair_grid), as the v1d docstring states\n'
         '                    samples.append(ev["sample_a"])   # v1f\n')


class BuildStop(Exception):
    pass


def replacements():
    return [("doc", OLD_DOC, NEW_DOC), ("version", OLD_VER, NEW_VER), ("anchor sample only with a point after the passage", OLD_A, NEW_A)]


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
    ap = argparse.ArgumentParser(description="assign_primitive_v1f over assign_primitive_v1e (exact replacements; fails closed)")
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
            print("STOP: the source md5 is %s, not v1e's %s" % (md5(src), SOURCE_MD5)); return 2
        out_text = apply(src, reps)
    except BuildStop as stop:
        print("STOP: %s; nothing written" % stop); return 2
    marks = marks_check(src, out_text)
    if marks:
        print("STOP: %d new line(s) lack %r: %s" % (len(marks), MARK, marks[:5])); return 2
    if apply(out_text, reps, reverse=True) != src:
        print("STOP: the reverse does not restore v1e byte for byte"); return 2
    compile(out_text, "assign_primitive_v1f.py", "exec")
    print("build: %s (md5 %s) -> v1f (md5 %s); %d replacements; %d lines carry %r; marks ok; reverse byte for byte; compiled" % (
        a.source, md5(src), md5(out_text), len(reps), sum(1 for l in out_text.split("\n") if MARK in l), MARK))
    if a.check:
        print("CHECK PASS (nothing written)"); return 0
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.source)), "assign_primitive_v1f.py")
    try:
        write_new(out, out_text)
    except BuildStop as stop:
        print("STOP: %s" % stop); return 2
    print("written: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
