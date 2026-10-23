#!/usr/bin/env python
"""build_engine_fields_v16c.py (busi, analyst; inbox #343, PERF tracker rows H2-A, H2-B): engine_fields_v16c.py from
engine_fields_v16b.py (md5 d7250a37...) by one marked replacement block: Options A (H2_PW_ONE) and B (H2_LOCAL_SPLINE) ON by default.

v16b reads each switch as  os.environ.get(NAME, "0") == "1"  (unset = off). v16c reads it as  os.environ.get(NAME, "1") != "0"  (unset = on;
NAME=0 turns that option off, giving v16's arithmetic and control flow exactly, as in v16b). Nothing else changes: the A and B code is
v16b's byte for byte, Option C is absent, info["module"] stays "engine_fields_v16" (the bytewise gates compare it).
The two switch lines of v16b's "H2-S" block are replaced by a block opening "# >>> v16c H2-default" and closing "# <<< v16c H2-default";
every new line lies between that pair (checked); the reverse pass (NEW -> OLD) restores v16b byte for byte (in memory and on the written
file; --reverse --expect); it compiles. The file is never overwritten.
    python build_engine_fields_v16c.py --src .../engine_fields_v16b.py [--dst .../engine_fields_v16c.py] [--check-only]
    python build_engine_fields_v16c.py --reverse --src .../engine_fields_v16c.py --dst .../restored.py --expect .../engine_fields_v16b.py
"""
import argparse
import hashlib
import os

MARK = "v16c H2"
SRC_MD5_PREFIX = "d7250a37"


def edits():
    old = ('_H2_PW_ONE = os.environ.get("H2_PW_ONE", "0") == "1"\n'
           '_H2_LOCAL_SPLINE = os.environ.get("H2_LOCAL_SPLINE", "0") == "1"\n')
    new = ('# >>> v16c H2-default: engine_fields_v16c = engine_fields_v16b with Options A and B ON by default (build_engine_fields_v16c.py,\n'
           '# inbox #343; supersedes the "all off by default" note above); H2_PW_ONE=0 / H2_LOCAL_SPLINE=0 turn each off (= v16 exactly)\n'
           '_H2_PW_ONE = os.environ.get("H2_PW_ONE", "1") != "0"\n'
           '_H2_LOCAL_SPLINE = os.environ.get("H2_LOCAL_SPLINE", "1") != "0"\n'
           '# <<< v16c H2-default\n')
    return [("D1 defaults", old, new)]


def md5(b):
    return hashlib.md5(b).hexdigest()


def forward(text, E):
    for name, old, new in E:
        c = text.count(old)
        if c != 1:
            raise SystemExit("edit %s: the OLD anchor occurs %d times (must be 1)" % (name, c))
        text = text.replace(old, new)
    return text


def reverse(text, E):
    for name, old, new in reversed(E):
        c = text.count(new)
        if c != 1:
            raise SystemExit("reverse %s: the NEW block occurs %d times (must be 1)" % (name, c))
        text = text.replace(new, old)
    return text


def marker_check(E):
    """every line a NEW block adds lies between a '# >>> v16c H2-<X>' and a '# <<< v16c H2-<X>' line."""
    bad, n_new = [], 0
    for name, old, new in E:
        inside = False
        for ln in new.splitlines():
            s = ln.strip()
            if s.startswith("# >>> " + MARK):
                inside = True; n_new += 1; continue
            if s.startswith("# <<< " + MARK):
                inside = False; n_new += 1; continue
            if inside or s == "":
                n_new += inside; continue
            bad.append((name, ln))
        if inside:
            bad.append((name, "<block not closed>"))
    return bad, n_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="engine_fields_v16b.py (build) or engine_fields_v16c.py (--reverse)")
    ap.add_argument("--dst", default=None)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--expect", default=None)
    ap.add_argument("--expect-src-md5", default=SRC_MD5_PREFIX)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    E = edits()
    src = open(a.src, "rb").read()
    text = src.decode("utf-8")
    if a.reverse:
        out = reverse(text, E).encode("utf-8")
        if a.dst and not a.check_only:
            if os.path.exists(a.dst):
                raise SystemExit("%s exists; nothing written (never overwritten)" % a.dst)
            open(a.dst, "wb").write(out)
        print("reverse: %s %s (md5 %s)" % ("would write" if (a.check_only or not a.dst) else "wrote", a.dst, md5(out)))
        if a.expect:
            ok = out == open(a.expect, "rb").read()
            print("restored equals %s byte for byte: %s" % (a.expect, ok))
            raise SystemExit(0 if ok else 1)
        return
    print("source %s: md5 %s, %d bytes" % (a.src, md5(src), len(src)))
    if a.expect_src_md5 and not md5(src).startswith(a.expect_src_md5):
        raise SystemExit("the source's md5 does not start with %s: not engine_fields_v16b.py; nothing written" % a.expect_src_md5)
    if MARK in text:
        raise SystemExit("the source already carries %r; nothing written" % MARK)
    for name, old, new in E:
        print("  anchor %-12s occurs %d time(s) in the input (%d -> %d lines)" % (name, text.count(old), len(old.splitlines()), len(new.splitlines())))
    bad, n_new = marker_check(E)
    if bad:
        for name, ln in bad:
            print("  UNMARKED new line in %s: %r" % (name, ln[:160]))
        raise SystemExit("new lines outside the marked blocks")
    print("every new line lies in a '# >>> %s-X' ... '# <<< %s-X' block (%d lines incl. the markers)" % (MARK, MARK, n_new))
    built = forward(text, E)
    out = built.encode("utf-8")
    back = reverse(built, E).encode("utf-8")
    print("reversible (NEW -> OLD restores the input byte for byte): %s" % (back == src))
    if back != src:
        raise SystemExit("the reverse pass does not restore the input; nothing written")
    try:
        compile(built, a.dst or "engine_fields_v16c.py", "exec")
    except SyntaxError as e:
        raise SystemExit("the result does not compile (%s); nothing written" % e)
    if a.dst is None:
        a.dst = os.path.join(os.path.dirname(os.path.abspath(a.src)), "engine_fields_v16c.py")
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
    if written != out or reverse(written.decode("utf-8"), E).encode("utf-8") != src:
        raise SystemExit("the written file does not reverse to the input")
    compile(written.decode("utf-8"), a.dst, "exec")
    print("wrote %s: %d bytes, md5 %s; compiles; the reverse of the written file equals the source byte for byte" % (a.dst, len(out), md5(out)))


if __name__ == "__main__":
    main()
