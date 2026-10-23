#!/usr/bin/env python
"""Strand 1 baseline: compare the rerun with the stored native-action results (run on the box, in the run directory).

1. The two summary JSONs (native_summary.json from native_report.py, native_kscan_summary.json from
   native_kscan_report.py), rerun against stored/, leaf by leaf: missing paths, non-numeric differences, and the
   largest absolute and relative numeric differences (relative to max(|a|, |b|, 1e-8)).
2. The per-job files: rerun out/jobs/*.json against the stored native_jobs.tar.gz, the same leaf comparison over all
   jobs pooled, skipping timing and host fields.
3. The k-scan raw file out_kscan/native_kscan.json against stored/native_kscan.json.gz (records matched by spec).
4. The generated tables (native_tables.md, RESULTS_native_kscan_tables.md) against stored/, line by line.
Writes compare_s1.json and prints a short report.
"""
import difflib
import glob
import gzip
import json
import math
import os
import tarfile

SKIP = ("seconds", "host", "time", "elapsed", "wall")


def leaves(x, path=""):
    if isinstance(x, dict):
        for k in sorted(x, key=str):
            if any(s in str(k).lower() for s in SKIP):
                continue
            yield from leaves(x[k], "%s/%s" % (path, k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, "%s[%d]" % (path, i))
    else:
        yield path, x


def isnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def compare_dicts(A, B):
    only_a = sorted(set(A) - set(B))
    only_b = sorted(set(B) - set(A))
    nonnum, rows = [], []
    for p in sorted(set(A) & set(B)):
        a, b = A[p], B[p]
        if isnum(a) and isnum(b):
            fa, fb = float(a), float(b)
            if math.isnan(fa) and math.isnan(fb):
                continue
            if math.isnan(fa) or math.isnan(fb) or math.isinf(fa) or math.isinf(fb):
                if fa != fb:
                    nonnum.append((p, a, b))
                continue
            d = abs(fa - fb)
            rows.append((d / max(abs(fa), abs(fb), 1e-8), d, p, fa, fb))
        elif a != b:
            nonnum.append((p, a, b))
    rows.sort(reverse=True)
    return {
        "n_paths_rerun": len(A), "n_paths_stored": len(B), "only_rerun": only_a[:20], "n_only_rerun": len(only_a),
        "only_stored": only_b[:20], "n_only_stored": len(only_b), "n_numeric": len(rows),
        "max_abs": max((r[1] for r in rows), default=0.0), "max_rel": max((r[0] for r in rows), default=0.0),
        "n_rel_gt_1e-6": sum(1 for r in rows if r[0] > 1e-6), "n_rel_gt_1e-9": sum(1 for r in rows if r[0] > 1e-9),
        "worst": [{"path": r[2], "rerun": r[3], "stored": r[4], "abs": r[1], "rel": r[0]} for r in rows[:15]],
        "nonnumeric_diffs": [{"path": p, "rerun": str(a)[:120], "stored": str(b)[:120]} for p, a, b in nonnum[:20]],
        "n_nonnumeric_diffs": len(nonnum),
    }


def flat(obj, prefix=""):
    return {prefix + p: v for p, v in leaves(obj)}


def main():
    out = {}
    for name in ("native_summary.json", "native_kscan_summary.json"):
        out[name] = compare_dicts(flat(json.load(open(name))), flat(json.load(open(os.path.join("stored", name)))))

    # per-job files
    A, B = {}, {}
    for f in sorted(glob.glob(os.path.join("out", "jobs", "*.json"))):
        A.update(flat(json.load(open(f)), os.path.basename(f) + ":"))
    with tarfile.open(os.path.join("stored", "native_jobs.tar.gz")) as tf:
        names = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".json")]
        for m in names:
            B.update(flat(json.load(tf.extractfile(m)), os.path.basename(m.name) + ":"))
    out["jobs"] = compare_dicts(A, B)
    out["jobs"]["n_job_files_rerun"] = len(glob.glob(os.path.join("out", "jobs", "*.json")))
    out["jobs"]["n_job_files_stored"] = len(names)

    # raw k-scan records, matched by spec
    def by_spec(D):
        return {json.dumps(r["spec"]): r for r in D["records"]}
    ka = by_spec(json.load(open(os.path.join("out_kscan", "native_kscan.json"))))
    kb = by_spec(json.load(gzip.open(os.path.join("stored", "native_kscan.json.gz"), "rt")))
    out["kscan_raw"] = compare_dicts(flat(ka), flat(kb))

    # tables
    for name in ("native_tables.md", "RESULTS_native_kscan_tables.md"):
        la = open(name).read().splitlines()
        lb = open(os.path.join("stored", name)).read().splitlines()
        diff = [l for l in difflib.unified_diff(lb, la, "stored", "rerun", n=0, lineterm="") if not l.startswith("@@")]
        out[name] = {"lines_rerun": len(la), "lines_stored": len(lb), "identical": la == lb,
                     "n_diff_lines": sum(1 for l in diff if l[:1] in "+-" and not l.startswith(("+++", "---"))),
                     "first_diffs": diff[:40]}

    with open("compare_s1.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    for k, v in out.items():
        if "n_numeric" in v:
            print("%-28s numeric %7d  max_abs %.3e  max_rel %.3e  rel>1e-6 %d  rel>1e-9 %d  nonnum %d  only_rerun %d  "
                  "only_stored %d" % (k, v["n_numeric"], v["max_abs"], v["max_rel"], v["n_rel_gt_1e-6"],
                                      v["n_rel_gt_1e-9"], v["n_nonnumeric_diffs"], v["n_only_rerun"],
                                      v["n_only_stored"]))
        else:
            print("%-28s identical %s  diff lines %d (rerun %d, stored %d lines)" % (
                k, v["identical"], v["n_diff_lines"], v["lines_rerun"], v["lines_stored"]))


if __name__ == "__main__":
    main()
