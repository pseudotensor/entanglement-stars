#!/usr/bin/env python
"""cmp_v9c2_runs.py (busi lane, PERF C-A1..C-A5): two engine runs compared as the G0'' compares do (engine_twofield_v8_selftest.compare_runs:
every record number but the wall seconds, every npz row at the common clocks, arrays and tensors bitwise), plus the json args (the run paths and
the arm's own --perf options set aside) and, for the B run's log, the v9c2 perf lines (the physical counts).

Usage. python cmp_v9c2_runs.py A_DIR/TAG B_DIR/TAG [--b-log LOG] [--after T] [--args-equal]
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_twofield_v8_selftest as S8  # noqa: E402

PATH_KEYS = {"out", "tag", "slice_dir", "resume_slice", "resume_run", "resume_from"}
PERF_KEYS = {"perf_eig_cache_gib", "perf_dens_memo", "perf_uhx_cache_gib", "perf_writer"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("A"); ap.add_argument("B")
    ap.add_argument("--b-log", default=None); ap.add_argument("--after", type=float, default=None)
    ap.add_argument("--args-equal", action="store_true", help="the args must be equal with only the paths set aside (the defaults arm)")
    a = ap.parse_args()
    common, diffs = S8.compare_runs(a.A, a.B, a.after)
    ja = json.load(open(os.path.join(os.path.dirname(a.A), "collapse_%s.json" % os.path.basename(a.A))))["args"]
    jb = json.load(open(os.path.join(os.path.dirname(a.B), "collapse_%s.json" % os.path.basename(a.B))))["args"]
    skip = PATH_KEYS if a.args_equal else PATH_KEYS | PERF_KEYS
    adiff = sorted(k for k in set(ja) | set(jb) if k not in skip and ja.get(k, "<absent>") != jb.get(k, "<absent>"))
    perf_b = {k: jb[k] for k in sorted(PERF_KEYS) if k in jb}
    ok = (not diffs) and len(common) > 0 and not adiff
    print("%s compare %s against %s over %d common records%s: %d differences%s; args differ in %s; the B run's perf args %s" % (
        "PASS" if ok else "FAIL", a.A, a.B, len(common), (" after T/rs %g" % a.after) if a.after is not None else "", len(diffs),
        (" %s" % diffs[:6]) if diffs else "", adiff if adiff else "nothing", perf_b if perf_b else "none (the defaults)"))
    if a.b_log and os.path.exists(a.b_log):
        tot = {}
        nl = 0
        for line in open(a.b_log, errors="replace"):
            if "v9c2 perf (" in line:
                nl += 1
                for k, v in re.findall(r"(\w+) ([0-9.]+)(?:,|;|$)", line.split("):", 1)[1].strip()):
                    tot[k] = tot.get(k, 0.0) + float(v)
        print("  B log %s: %d v9c2 perf lines; totals %s" % (a.b_log, nl, ", ".join("%s %g" % kv for kv in sorted(tot.items())) if tot else "none"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
