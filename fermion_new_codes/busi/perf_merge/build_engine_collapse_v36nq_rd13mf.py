#!/usr/bin/env python
"""build_engine_collapse_v36nq_rd13mf.py (busi, analyst; MERGE-1, busi/notes/PERF_MERGE_build_v1.md): the merged ISO Newton engine
engine_collapse_v36nq_rd13mf.py = engine_collapse_v36nq_rd13m.py (md5 758e0e8f...: v36nq 8549c542 + RD-1/RD-3 rd13 blocks + MEM-1 rd13m
blocks) + RD-2's field-route wiring (the 5 blocks of build_engine_collapse_rd2.py, 28 lines, "rd2 fields" on every new line), so that one
engine carries --eigh banded, --jvp sternheimer, --nk-cache-host, --fields and --fields-warm-start, every option off by default.
Blocks B to E are RD-2's text, taken from build_engine_collapse_rd2.edits() (imported from the same directory, read only), with the same
anchors; block A (the docstring's lineage lines) is this build's own, in RD-2's form. The field module is chosen by deployment: the code
dir's engine_fields_v16.py must carry rd2_configure / rd2_stats (engine_fields_v16d 7a69faad, or v16e when RD-2b passes) for a
non-default --fields; at the defaults the engine never calls them, so any v16 content runs (v16c, v16d, v16e).
Rules (checked): the source's md5 begins with 758e0e8f; every OLD anchor occurs exactly once; every line a NEW block adds carries
"rd2 fields"; the reverse pass (NEW -> OLD) restores the source byte for byte, in memory and on the written file; the result compiles;
nothing is overwritten.
    python build_engine_collapse_v36nq_rd13mf.py --src .../engine_collapse_v36nq_rd13m.py [--dst .../engine_collapse_v36nq_rd13mf.py] [--check-only]
    python build_engine_collapse_v36nq_rd13mf.py --reverse --src .../engine_collapse_v36nq_rd13mf.py --dst .../restored.py --expect .../engine_collapse_v36nq_rd13m.py
"""
import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_engine_collapse_rd2 as RD2B   # RD-2's builder (busi/code/perf_rd2/), read only: its edits() text for blocks B to E

MARK = RD2B.MARK   # "rd2 fields"
SRC_MD5_PREFIX = "758e0e8f"
SRC_NAME = "engine_collapse_v36nq_rd13m.py"
OUT_NAME = "engine_collapse_v36nq_rd13mf.py"


def edits(text):
    # RD-2's blocks for a host named like v36nq; block A's anchor is the docstring's first line, which names the host file
    RD2B.HOSTS["v36nq_rd13m"] = {"md5": SRC_MD5_PREFIX, "name": SRC_NAME, "out": OUT_NAME}
    E = RD2B.edits(text, "v36nq_rd13m")
    assert [e[0] for e in E] == ["A lineage", "B settings", "C options", "D configuration", "E record"], [e[0] for e in E]
    oldA = '#!/usr/bin/env python\n"""' + SRC_NAME + " ("
    newA = ('#!/usr/bin/env python\n"""' + OUT_NAME + " (busi, analyst; MERGE-1: RD-2's --fields wiring (build_engine_collapse_rd2.py's blocks B to E) on " + SRC_NAME + " md5 758e0e8f, built by build_engine_collapse_v36nq_rd13mf.py) [rd2 fields]\n"
            "version 36nq_rd13mf = version 36nq_rd13m with RD-2's two options, off by default, so that the defaults are rd13m (and v36nq 8549c542's route) bitwise [rd2 fields]\n"
            "    (the json's args carry the 2 new keys): --fields collocation: every field solve at the engine's tolerance by the code dir's field module's [rd2 fields]\n"
            "    Gauss-Legendre collocation (engine_fields_v16d 7a69faad, or v16e, as engine_fields_v16.py; the ivp route the logged automatic fallback); [rd2 fields]\n"
            "    --fields-warm-start on: its Newton warm-started from the latest solve on the same mesh. With --eigh banded, --jvp sternheimer and [rd2 fields]\n"
            "    --nk-cache-host (rd13, rd13m) every adopted or pending option of the v36nq line is in one engine. Version 36nq_rd13m's docstring follows. [rd2 fields]\n"
            + SRC_NAME + " (")
    assert E[0][1] == oldA
    E[0] = ("A lineage", oldA, newA)
    return E


def md5(b):
    return hashlib.md5(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="engine_collapse_v36nq_rd13m.py (build) or engine_collapse_v36nq_rd13mf.py (--reverse)")
    ap.add_argument("--dst", default=None)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--expect", default=None, help="(reverse) engine_collapse_v36nq_rd13m.py; the blocks are derived from it")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    src = open(a.src, "rb").read()
    text = src.decode("utf-8")
    if a.reverse:
        if not a.expect:
            raise SystemExit("--reverse needs --expect")
        host = open(a.expect, "rb").read()
        if not md5(host).startswith(SRC_MD5_PREFIX):
            raise SystemExit("--expect's md5 %s does not begin with %s" % (md5(host), SRC_MD5_PREFIX))
        E = edits(host.decode("utf-8"))
        out = RD2B.reverse(text, E).encode("utf-8")
        if a.dst and not a.check_only:
            if os.path.exists(a.dst):
                raise SystemExit("%s exists; nothing written (never overwritten)" % a.dst)
            open(a.dst, "wb").write(out)
        ok = out == host
        print("reverse: %s %s (md5 %s); restored equals %s (md5 %s) byte for byte: %s" % (
            "would write" if (a.check_only or not a.dst) else "wrote", a.dst, md5(out), a.expect, md5(host), ok))
        raise SystemExit(0 if ok else 1)
    print("source %s: md5 %s, %d bytes" % (a.src, md5(src), len(src)))
    if not md5(src).startswith(SRC_MD5_PREFIX):
        raise SystemExit("the source's md5 does not begin with %s; nothing written" % SRC_MD5_PREFIX)
    if MARK in text or "RD2F = {" in text or '"--fields"' in text:
        raise SystemExit("the source already carries %r; nothing written" % MARK)
    E = edits(text)
    for name, old, new in E:
        print("  block %-16s OLD occurs %d time(s) (%d -> %d lines)" % (name, text.count(old), len(old.splitlines()), len(new.splitlines())))
    bad, n = RD2B.marker_check(E)
    if bad:
        for name, ln in bad:
            print("  UNMARKED new line in %s: %r" % (name, ln[:160]))
        raise SystemExit("new lines without the marker")
    print("every new line carries %r (%d lines)" % (MARK, n))
    built = RD2B.forward(text, E)
    out = built.encode("utf-8")
    back = RD2B.reverse(built, E).encode("utf-8")
    print("reversible (NEW -> OLD restores the source byte for byte): %s" % (back == src))
    if back != src:
        raise SystemExit("the reverse pass does not restore the source; nothing written")
    compile(built, a.dst or OUT_NAME, "exec")
    if a.dst is None:
        a.dst = os.path.join(os.path.dirname(os.path.abspath(a.src)), OUT_NAME)
    if a.check_only:
        print("check only: %s would be %d bytes, md5 %s" % (a.dst, len(out), md5(out)))
        return
    if os.path.exists(a.dst):
        if open(a.dst, "rb").read() == out:
            print("%s exists and equals the build byte for byte (md5 %s); not rewritten" % (a.dst, md5(out)))
            return
        raise SystemExit("%s exists and differs from the build; nothing written (never overwritten)" % a.dst)
    open(a.dst, "wb").write(out)
    written = open(a.dst, "rb").read()
    if written != out or RD2B.reverse(written.decode("utf-8"), E).encode("utf-8") != src:
        raise SystemExit("the written file does not reverse to the source")
    compile(written.decode("utf-8"), a.dst, "exec")
    print("wrote %s: %d bytes, md5 %s; compiles; the reverse of the written file equals the source byte for byte; lines carrying '%s': %d" % (
        a.dst, len(out), md5(out), MARK, sum(1 for ln in built.splitlines() if MARK in ln)))


if __name__ == "__main__":
    main()
