#!/usr/bin/env python
"""build_C_s2_domain_flags_v1b.py (the busi lane, Computation C): writes C_s2_domain_flags_v1b.py and C_s2_domain_flags_selftest_v1b.py from the delivered
C_s2_domain_flags_v1.py and C_s2_domain_flags_selftest_v1.py by exact string replacement. Every old text must occur exactly once in its source; if any
does not, the build stops and writes nothing. The delivered files are not touched.

The one change in the computation is the zero_crossings helper. v1 took np.sign of the samples and so reported an exactly-zero sample twice: its
sign 0 differs from both neighbours' signs, giving 2 entries at its own abscissa (the self-test's check "zero_crossings: a zero value at a point
gives the crossing at that point" failed on the sample (0.0, 1.0), (0.01, 0.0), (0.02, -1.0); 62 of 63 checks passed by the builder's report). The
delivered run fed no exact zero to the helper (the smallest magnitudes -1.1112e-08 and +1.5293e-07), so the delivered outputs are unchanged. v1b
reports an exactly-zero sample once, at its own abscissa, with "exact" True, and interpolates only between adjacent nonzero samples of opposite sign.
The other replacements rename the script, its usage line, its json "script" field, its output file names and the md's title to v1b, so that the
delivered run's files stay as issued, and let the md say which crossings were exact samples. read_flags, the bisection, the summary and the tables
are untouched.

The self-test is pointed at the v1b module and gains 2 checks of the fixed helper: the delivered self-test's 63 checks (counted from its code; the
builder's 62 of 63 plus the failing one) plus 2 give 65 (derived), so its expected last line is EXPECTED_SELFTEST_LINE below. --run-selftest runs it
in --out-dir after the build and compares the last line.

Usage: python build_C_s2_domain_flags_v1b.py [--src-dir DIR] [--out-dir DIR] [--run-selftest]
  (the defaults for both directories: this script's own directory; on the box the delivered code lives in <internal>/code)"""
import argparse
import os
import subprocess
import sys

EXPECTED_SELFTEST_LINE = "SELFTEST PASS: 65 checks, 0 failed"

BUILD_SRC = "C_s2_domain_flags_v1.py"
BUILD_OUT = "C_s2_domain_flags_v1b.py"
SELFTEST_SRC = "C_s2_domain_flags_selftest_v1.py"
SELFTEST_OUT = "C_s2_domain_flags_selftest_v1b.py"

# ------------------------------------------------------------------------------------------------ the replacements on C_s2_domain_flags_v1.py
OLD_DOC_HEAD = r'''C_s2_domain_flags_v1.py (the busi lane, Computation C): calculation (i) of C_s2_fold_reading_v1.2.md, section 3, the domain flags at the scan's
stored points, through C_s2_soft_scan_v1's capture machinery (imported; nothing of the scan copied or edited) and with none of the engine edited.
'''
NEW_DOC_HEAD = r'''C_s2_domain_flags_v1b.py (the busi lane, Computation C): calculation (i) of C_s2_fold_reading_v1.2.md, section 3, the domain flags at the scan's
stored points, through C_s2_soft_scan_v1's capture machinery (imported; nothing of the scan copied or edited) and with none of the engine edited.
v1b: built from C_s2_domain_flags_v1.py by exact replacement (build_C_s2_domain_flags_v1b.py). The one change in the computation is the zero_crossings
helper, which took np.sign of the samples and reported an exactly-zero sample twice (its sign 0 differing from both neighbours'); it now reports such a
sample once, at its own abscissa, and interpolates only between adjacent nonzero samples of opposite sign. The output file names carry v1b; read_flags,
the bisection, the summary and the tables are unchanged.
'''

OLD_USAGE = r'''Usage: cd <internal>/code && python C_s2_domain_flags_v1.py --engine-dir <internal>/code --scan-dir <internal>/soft_scan_v1'''
NEW_USAGE = r'''Usage: cd <internal>/code && python C_s2_domain_flags_v1b.py --engine-dir <internal>/code --scan-dir <internal>/soft_scan_v1'''

OLD_ZC = r'''def zero_crossings(pairs):
    """the zeros of a readout along dN0 by linear interpolation between adjacent points of opposite sign ((dN0, value) pairs, None skipped)."""
    pts = []
    for d, v in pairs:
        if d is None or v is None:
            continue
        try:
            d = float(d); v = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(d) and np.isfinite(v):
            pts.append((d, v))
    pts.sort(key=lambda q: q[0])
    out = []
    for (da, va), (db, vb) in zip(pts, pts[1:]):
        if np.sign(va) != np.sign(vb):
            dz = da + (db - da) * va / (va - vb) if va != vb else da
            out.append({"dN0": float(dz), "between": [da, db], "values": [va, vb]})
    return out
'''
NEW_ZC = r'''def zero_crossings(pairs):
    """the zeros of a readout along dN0 ((dN0, value) pairs; None and non-finite entries skipped): a sample that is exactly 0 is a zero at its own
    abscissa, reported once with "exact" True whatever its neighbours' signs (v1 took np.sign and reported it twice, its sign 0 differing from both
    neighbours'); between adjacent samples that are both nonzero and of opposite sign the zero by linear interpolation ("exact" False); sorted by dN0."""
    pts = []
    for d, v in pairs:
        if d is None or v is None:
            continue
        try:
            d = float(d); v = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(d) and np.isfinite(v):
            pts.append((d, v))
    pts.sort(key=lambda q: q[0])
    out = []
    for d, v in pts:
        if v == 0.0:
            out.append({"dN0": float(d), "between": [d, d], "values": [v, v], "exact": True})
    for (da, va), (db, vb) in zip(pts, pts[1:]):
        if va != 0.0 and vb != 0.0 and (va < 0.0) != (vb < 0.0):
            out.append({"dN0": float(da + (db - da) * va / (va - vb)), "between": [da, db], "values": [va, vb], "exact": False})
    out.sort(key=lambda z: z["dN0"])
    return out
'''

OLD_MD_TITLE = r'''L = ["# C_s2_domain_flags_v1: the domain flags at the scan's stored points (calculation (i) of C_s2_fold_reading_v1.2.md, section 3)", ""]'''
NEW_MD_TITLE = r'''L = ["# C_s2_domain_flags_v1b: the domain flags at the scan's stored points (calculation (i) of C_s2_fold_reading_v1.2.md, section 3)", ""]'''

OLD_MD_ZC = r'''                L.append("- %s crosses 0 at dN0 %s (linear interpolation between evaluated points%s)." % (
                    lab, ", ".join("%+.5f (between %+.5f and %+.5f)" % (z["dN0"], z["between"][0], z["between"][1]) for z in zc) if zc else "nowhere among the evaluated points",'''
NEW_MD_ZC = r'''                L.append("- %s crosses 0 at dN0 %s (linear interpolation between adjacent evaluated points of opposite sign; an exactly-zero sample reported once at its own abscissa%s)." % (
                    lab, ", ".join((("%+.5f (an exactly-zero sample)" % z["dN0"]) if z.get("exact") else ("%+.5f (between %+.5f and %+.5f)" % (z["dN0"], z["between"][0], z["between"][1]))) for z in zc) if zc else "nowhere among the evaluated points",'''

OLD_PATHS = r'''paths = {"json": os.path.join(o.out, "C_s2_domain_flags_v1.json"), "npz": os.path.join(o.out, "C_s2_domain_flags_v1.npz"), "md": os.path.join(o.out, "C_s2_domain_flags_v1.md")}'''
NEW_PATHS = r'''paths = {"json": os.path.join(o.out, "C_s2_domain_flags_v1b.json"), "npz": os.path.join(o.out, "C_s2_domain_flags_v1b.npz"), "md": os.path.join(o.out, "C_s2_domain_flags_v1b.md")}'''

OLD_SCRIPT = r'''RES = {"script": "C_s2_domain_flags_v1.py", "args": vars(o), "per_s": {}, "status": "running"}'''
NEW_SCRIPT = r'''RES = {"script": "C_s2_domain_flags_v1b.py", "args": vars(o), "per_s": {}, "status": "running"}'''

BUILD_REPLACEMENTS = [
    ("the docstring's head (the name and the v1b note)", OLD_DOC_HEAD, NEW_DOC_HEAD),
    ("the usage line", OLD_USAGE, NEW_USAGE),
    ("the zero_crossings helper (the fix)", OLD_ZC, NEW_ZC),
    ("the md's title line", OLD_MD_TITLE, NEW_MD_TITLE),
    ("the md's zero-crossing line (exact samples named)", OLD_MD_ZC, NEW_MD_ZC),
    ("the output file names", OLD_PATHS, NEW_PATHS),
    ("the json's script field", OLD_SCRIPT, NEW_SCRIPT),
]

# ------------------------------------------------------------------------------------------ the replacements on C_s2_domain_flags_selftest_v1.py
OLD_ST_DOC = r'''C_s2_domain_flags_selftest_v1.py: manufactured checks of C_s2_domain_flags_v1 without the engine.'''
NEW_ST_DOC = r'''C_s2_domain_flags_selftest_v1b.py: manufactured checks of C_s2_domain_flags_v1b without the engine (built from C_s2_domain_flags_selftest_v1.py by exact
replacement in build_C_s2_domain_flags_v1b.py; the 2 checks marked v1b test the fixed zero_crossings helper).'''

OLD_ST_USAGE = r'''Usage: python C_s2_domain_flags_selftest_v1.py [--bisection-steps 6]'''
NEW_ST_USAGE = r'''Usage: python C_s2_domain_flags_selftest_v1b.py [--bisection-steps 6]'''

OLD_ST_IMPORT = r'''import C_s2_domain_flags_v1 as DF  # noqa: E402'''
NEW_ST_IMPORT = r'''import C_s2_domain_flags_v1b as DF  # noqa: E402'''

OLD_ST_PRINT_1 = r'''print("C_s2_domain_flags_selftest_v1: flag parsing", flush=True)'''
NEW_ST_PRINT_1 = r'''print("C_s2_domain_flags_selftest_v1b: flag parsing", flush=True)'''
OLD_ST_PRINT_2 = r'''print("C_s2_domain_flags_selftest_v1: the chord bisection", flush=True)'''
NEW_ST_PRINT_2 = r'''print("C_s2_domain_flags_selftest_v1b: the chord bisection", flush=True)'''
OLD_ST_PRINT_3 = r'''print("C_s2_domain_flags_selftest_v1: the helpers and the smoke test", flush=True)'''
NEW_ST_PRINT_3 = r'''print("C_s2_domain_flags_selftest_v1b: the helpers and the smoke test", flush=True)'''

OLD_ST_DESC = r'''description="manufactured checks of C_s2_domain_flags_v1 (no engine)"'''
NEW_ST_DESC = r'''description="manufactured checks of C_s2_domain_flags_v1b (no engine)"'''

OLD_ST_ZC = r'''    zc = DF.zero_crossings([(0.0, 1.0), (0.01, 0.0), (0.02, -1.0)])
    check("zero_crossings: a zero value at a point gives the crossing at that point", len(zc) == 1 and abs(zc[0]["dN0"] - 0.01) <= o.tol, zc)
'''
NEW_ST_ZC = r'''    zc = DF.zero_crossings([(0.0, 1.0), (0.01, 0.0), (0.02, -1.0)])
    check("zero_crossings: a zero value at a point gives the crossing at that point", len(zc) == 1 and abs(zc[0]["dN0"] - 0.01) <= o.tol, zc)
    zc = DF.zero_crossings([(0.0, 1.0), (0.01, 0.0), (0.02, 1.0)])
    check("zero_crossings (v1b): an exactly-zero sample between two values of the same sign is one zero at its own abscissa, marked exact",
          len(zc) == 1 and abs(zc[0]["dN0"] - 0.01) <= o.tol and zc[0].get("exact") is True, zc)
    zc = DF.zero_crossings([(0.0, 0.0), (0.01, 1.0), (0.02, -1.0), (0.03, 0.0)])
    check("zero_crossings (v1b): zero samples at both ends once each and the interpolated crossing between them, in dN0 order",
          len(zc) == 3 and all(abs(z["dN0"] - d) <= o.tol for z, d in zip(zc, (0.0, 0.015, 0.03))) and [bool(z.get("exact")) for z in zc] == [True, False, True], zc)
'''

SELFTEST_REPLACEMENTS = [
    ("the docstring's head", OLD_ST_DOC, NEW_ST_DOC),
    ("the usage line", OLD_ST_USAGE, NEW_ST_USAGE),
    ("the import of the module under test", OLD_ST_IMPORT, NEW_ST_IMPORT),
    ("the first section's print", OLD_ST_PRINT_1, NEW_ST_PRINT_1),
    ("the second section's print", OLD_ST_PRINT_2, NEW_ST_PRINT_2),
    ("the third section's print", OLD_ST_PRINT_3, NEW_ST_PRINT_3),
    ("the argparse description", OLD_ST_DESC, NEW_ST_DESC),
    ("the zero_crossings checks (2 added)", OLD_ST_ZC, NEW_ST_ZC),
]


def apply_replacements(src, repls, name):
    """each old text exactly once, or the build stops; returns the new text and the list of (label, count) applied."""
    out = src
    done = []
    for label, old, new in repls:
        n = out.count(old)
        if n != 1:
            raise SystemExit("%s: the old text for %s occurs %d times, not once; nothing written (the source is not the delivered file, or its whitespace differs)" % (name, label, n))
        out = out.replace(old, new)
        done.append((label, n))
    return out, done


def read(path):
    if not os.path.exists(path):
        raise SystemExit("no %s" % path)
    with open(path) as fh:
        return fh.read()


def main():
    ap = argparse.ArgumentParser(description="builds C_s2_domain_flags_v1b.py and its self-test from the delivered v1 files by exact replacement")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--src-dir", default=here, help="where %s and %s are (default: this script's directory)" % (BUILD_SRC, SELFTEST_SRC))
    ap.add_argument("--out-dir", default=None, help="where %s and %s are written (default: --src-dir)" % (BUILD_OUT, SELFTEST_OUT))
    ap.add_argument("--run-selftest", action="store_true", help="run the v1b self-test after the build and compare its last line with %r" % EXPECTED_SELFTEST_LINE)
    o = ap.parse_args()
    out_dir = o.out_dir or o.src_dir
    os.makedirs(out_dir, exist_ok=True)
    build_src = read(os.path.join(o.src_dir, BUILD_SRC))
    selftest_src = read(os.path.join(o.src_dir, SELFTEST_SRC))
    build_new, build_done = apply_replacements(build_src, BUILD_REPLACEMENTS, BUILD_SRC)
    selftest_new, selftest_done = apply_replacements(selftest_src, SELFTEST_REPLACEMENTS, SELFTEST_SRC)
    # both results must at least compile before anything is written
    compile(build_new, BUILD_OUT, "exec")
    compile(selftest_new, SELFTEST_OUT, "exec")
    p_build = os.path.join(out_dir, BUILD_OUT)
    p_self = os.path.join(out_dir, SELFTEST_OUT)
    with open(p_build, "w") as fh:
        fh.write(build_new)
    with open(p_self, "w") as fh:
        fh.write(selftest_new)
    print("build_C_s2_domain_flags_v1b: %s -> %s (%d replacements, each once): %s" % (BUILD_SRC, p_build, len(build_done), "; ".join(l for l, _ in build_done)))
    print("build_C_s2_domain_flags_v1b: %s -> %s (%d replacements, each once): %s" % (SELFTEST_SRC, p_self, len(selftest_done), "; ".join(l for l, _ in selftest_done)))
    print("build_C_s2_domain_flags_v1b: the self-test's expected last line: %s" % EXPECTED_SELFTEST_LINE)
    if not o.run_selftest:
        print("build_C_s2_domain_flags_v1b: run it with  cd %s && python %s  and file its output beside the run" % (out_dir, SELFTEST_OUT))
        return
    proc = subprocess.run([sys.executable, SELFTEST_OUT], cwd=out_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    sys.stdout.write(proc.stdout)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    last = lines[-1] if lines else ""
    summary = [ln for ln in lines if ln.startswith("SELFTEST ")]
    summary_line = summary[-1] if summary else last
    ok = (proc.returncode == 0 and summary_line == EXPECTED_SELFTEST_LINE)
    print("build_C_s2_domain_flags_v1b: self-test exit %d, summary line %r, expected %r: %s" % (proc.returncode, summary_line, EXPECTED_SELFTEST_LINE, "as expected" if ok else "NOT as expected"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
