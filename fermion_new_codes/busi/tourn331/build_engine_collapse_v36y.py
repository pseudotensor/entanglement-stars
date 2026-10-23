#!/usr/bin/env python
"""build_engine_collapse_v36y.py (busi, analyst; #331, TOURNAMENT_reading_v1 sec. 4.1: MERGE the substep patch into the Newton engine lineage):
engine_collapse_v36y.py from engine_collapse_v36q.py (md5 759c526c...) and the piloted patch engine_collapse_v36q_srf.py (md5 77857541..., the solve
lane's tourn/substep, the engine of P2_srf_RUNG2c and P_srf_S4_128) by marked replacement hunks.

What the build is. v36y is v36q_srf under a version letter: the 20 hunks by which v36q_srf differs from v36q (difflib on lines; read from the two
pinned files at build time, never typed by hand) applied to v36q, plus one lineage hunk (A) that puts two "[v36y srf]" lines in front of the srf
docstring's first line. So v36y with its "v36y srf" lines removed (and the opening quotes put back on the srf line) is v36q_srf byte for byte,
and the reverse of every hunk restores v36q byte for byte. --step-refine off (the default) is v36q bitwise (the gate: the solve lane's
selftest_srf_v1b with ENGINE_NEW = v36y, and busi's whole-engine gate in engine_collapse_v36z_selftest.py).
Rules (checked): the source's md5 begins with 759c526c and the patch's with 77857541 (--expect-*-md5); every hunk's OLD lines sit at their recorded
place in v36q; every line a hunk adds or changes carries "srf" (v36q_srf's own marker) or "v36y srf", or is blank, or lies in v36q_srf's fenced
option block (the hunk that opens with "# srf: automatic record-step refinement (--step-refine). Everything below this comment to set_sig_v36c
is the option's"); the reverse pass restores v36q byte for byte (in memory before writing and on the written file); the output compiles; the
written v36y minus the "v36y srf" lines equals v36q_srf. The hunk list is written beside the engine as engine_collapse_v36y_patch.json.
    python build_engine_collapse_v36y.py --src .../engine_collapse_v36q.py --patch .../engine_collapse_v36q_srf.py [--dst .../engine_collapse_v36y.py] [--check-only]
    python build_engine_collapse_v36y.py --reverse --src .../engine_collapse_v36y.py --patch .../engine_collapse_v36q_srf.py --dst .../restored.py --expect .../engine_collapse_v36q.py
"""
import argparse
import difflib
import hashlib
import json
import os
import py_compile

MARK = "v36y srf"
SRC_MD5_PREFIX = "759c526c"
PATCH_MD5_PREFIX = "77857541"
FENCE = "# srf: automatic record-step refinement (--step-refine). Everything below this comment to set_sig_v36c is the option's"

LINEAGE = [
    '"""engine_collapse_v36y.py (busi, analyst; #331, TOURNAMENT_reading_v1 sec. 4.1: the substep patch merged into the Newton engine lineage; built by build_engine_collapse_v36y.py over v36q md5 759c526c from engine_collapse_v36q_srf.py md5 77857541) [v36y srf]\n',
    'version 36y = engine_collapse_v36q_srf.py under its version letter, every line of it below unchanged: --step-refine off (the default) is version 36q bitwise; the Newton members launch it with "--step-refine on --step-refine-max-halvings 3"; a sub-stepped history is flagged per record (srf.tau_in, halvings, grid_record, attempts) and reads "tau 4 a over [T1, T2]"; readers that assume a fixed record step (d1_raymap_data_v1.py line 55, dT = T[1] - T[0]) must read the marker. The srf docstring follows. [v36y srf]\n',
]


def md5(b):
    return hashlib.md5(b).hexdigest()


def hunks_of(q, s):
    """the non-equal opcodes of v36q -> v36q_srf on lines: [(i1, i2, old_lines, new_lines)], i1/i2 the place in v36q."""
    sm = difflib.SequenceMatcher(None, q, s, autojunk=False)
    return [(i1, i2, q[i1:i2], s[j1:j2]) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]


def lineage_hunk(H):
    """hunk A: v36q_srf's first docstring line (the first hunk replaces v36q's line 2 by the srf docstring's 17 lines) gets the two v36y lines in
    front of it and loses its opening quotes, which move to the first v36y line."""
    i1, i2, old, new = H[0]
    if i1 != 1 or not new[0].startswith('"""engine_collapse_v36q_srf.py'):
        raise SystemExit("the first hunk is not the srf docstring's lineage (got v36q line %d: %r)" % (i1 + 1, new[0][:80]))
    return [(i1, i2, old, LINEAGE + [new[0][3:]] + new[1:])] + H[1:]


def apply(q, H):
    out, k = [], 0
    for i1, i2, old, new in H:
        if q[i1:i2] != old:
            raise SystemExit("a hunk's OLD lines are not at v36q line %d" % (i1 + 1))
        out += q[k:i1] + new
        k = i2
    return out + q[k:]


def unapply(y, H):
    """the reverse: the NEW lines of every hunk (at their place in v36y) back to OLD."""
    out, k, off = [], 0, 0
    for i1, i2, old, new in H:
        j1 = i1 + off
        if y[j1:j1 + len(new)] != new:
            raise SystemExit("reverse: a hunk's NEW lines are not at v36y line %d" % (j1 + 1))
        out += y[k:j1] + old
        k = j1 + len(new)
        off += len(new) - (i2 - i1)
    return out + y[k:]


# the one line of v36q_srf outside its fenced block that carries no "srf": the second line of a docstring sentence whose first line does
# ("... the docstring's (c)). srf: the step passed to the convention is the record's own (the"), in the hunk at v36q line 1557
SRF_DOC_CONT = ['    sub-step inside a split interval; a.tau exactly with the option off), so the LZ rate estimate dh/dt is the same at every step."""\n']


def marker_check(H):
    bad, n_marked, n_fenced = [], 0, 0
    for i1, i2, old, new in H:
        olds = set(old) | {ln[3:] for ln in old if ln.startswith('"""')}
        fenced = any(FENCE in ln for ln in new[:3])
        for ln in new:
            if ln in olds or ln.strip() == "":
                continue
            if "srf" in ln or ln in SRF_DOC_CONT:
                n_marked += 1
            elif fenced:
                n_fenced += 1
            else:
                bad.append((i1 + 1, ln.rstrip()))
    return bad, n_marked, n_fenced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="engine_collapse_v36q.py (build) or engine_collapse_v36y.py (--reverse)")
    ap.add_argument("--patch", required=True, help="engine_collapse_v36q_srf.py (md5 77857541): the hunks' source")
    ap.add_argument("--base", default=None, help="(reverse) engine_collapse_v36q.py, from which the hunks are re-derived (default: --expect)")
    ap.add_argument("--dst", default=None)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--expect", default=None, help="(reverse) the file the restored text must equal byte for byte")
    ap.add_argument("--expect-src-md5", default=SRC_MD5_PREFIX)
    ap.add_argument("--expect-patch-md5", default=PATCH_MD5_PREFIX)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    pb = open(a.patch, "rb").read()
    print("patch %s: md5 %s, %d bytes" % (a.patch, md5(pb), len(pb)))
    if a.expect_patch_md5 and not md5(pb).startswith(a.expect_patch_md5):
        raise SystemExit("the patch's md5 does not start with %s: not the piloted engine_collapse_v36q_srf.py; nothing written" % a.expect_patch_md5)
    s = pb.decode("utf-8").splitlines(True)
    if a.reverse:
        base = a.base or a.expect
        if not base:
            raise SystemExit("--reverse needs --base or --expect (v36q, from which the hunks are derived)")
        q = open(base, "rb").read().decode("utf-8").splitlines(True)
        H = lineage_hunk(hunks_of(q, s))
        y = open(a.src, "rb").read().decode("utf-8").splitlines(True)
        out = "".join(unapply(y, H)).encode("utf-8")
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
    src = open(a.src, "rb").read()
    print("source %s: md5 %s, %d bytes" % (a.src, md5(src), len(src)))
    if a.expect_src_md5 and not md5(src).startswith(a.expect_src_md5):
        raise SystemExit("the source's md5 does not start with %s: not engine_collapse_v36q.py; nothing written" % a.expect_src_md5)
    text = src.decode("utf-8")
    if MARK in text or "SRF = {" in text:
        raise SystemExit("the source already carries the srf option or %r; nothing written" % MARK)
    q = text.splitlines(True)
    H = lineage_hunk(hunks_of(q, s))
    for k, (i1, i2, old, new) in enumerate(H):
        print("  hunk %2d at v36q line %5d: %3d -> %3d lines%s" % (k + 1, i1 + 1, i2 - i1, len(new), (" (A lineage)" if k == 0 else (" (the srf option block, fenced)" if any(FENCE in ln for ln in new[:3]) else ""))))
    bad, n_marked, n_fenced = marker_check(H)
    if bad:
        for i_, ln in bad:
            print("  UNMARKED new line in the hunk at v36q line %d: %r" % (i_, ln[:140]))
        raise SystemExit("new lines without a marker")
    print("every new or changed line carries 'srf' or %r (%d lines) or lies in the fenced srf block (%d lines)" % (MARK, n_marked, n_fenced))
    y = apply(q, H)
    built = "".join(y)
    out = built.encode("utf-8")
    back = "".join(unapply(y, H)).encode("utf-8")
    print("reversible (every hunk NEW -> OLD restores v36q byte for byte): %s" % (back == src))
    if back != src:
        raise SystemExit("the reverse pass does not restore the input; nothing written")
    as_srf = "".join(ln for ln in y if MARK not in ln)
    as_srf = as_srf.replace('#!/usr/bin/env python\nengine_collapse_v36q_srf.py', '#!/usr/bin/env python\n"""engine_collapse_v36q_srf.py', 1)
    same_srf = as_srf.encode("utf-8") == pb
    print("v36y minus its %r lines (the opening quotes back on the srf line) equals engine_collapse_v36q_srf.py byte for byte: %s" % (MARK, same_srf))
    if not same_srf:
        raise SystemExit("the build is not v36q_srf plus the lineage lines; nothing written")
    try:
        compile(built, a.dst or "engine_collapse_v36y.py", "exec")
    except SyntaxError as e:
        raise SystemExit("the result does not compile (%s); nothing written" % e)
    if a.dst is None:
        a.dst = os.path.join(os.path.dirname(os.path.abspath(a.src)), "engine_collapse_v36y.py")
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
    if written != out or "".join(unapply(written.decode("utf-8").splitlines(True), H)).encode("utf-8") != src:
        raise SystemExit("the written file does not reverse to the input")
    py_compile.compile(a.dst, doraise=True)
    pj = os.path.join(os.path.dirname(os.path.abspath(a.dst)), "engine_collapse_v36y_patch.json")
    if not os.path.exists(pj):
        json.dump({"src": os.path.basename(a.src), "src_md5": md5(src), "patch": os.path.basename(a.patch), "patch_md5": md5(pb), "dst_md5": md5(out),
                   "hunks": [{"v36q_line": i1 + 1, "old": old, "new": new} for i1, i2, old, new in H]}, open(pj, "w"), indent=1)
    n_lines = sum(1 for ln in built.splitlines() if MARK in ln)
    print("wrote %s: %d bytes, md5 %s; compiles; the reverse of the written file equals the source byte for byte; hunks in %s" % (a.dst, len(out), md5(out), pj))
    print("check: grep -c '%s' %s  ->  %d" % (MARK, a.dst, n_lines))


if __name__ == "__main__":
    main()
