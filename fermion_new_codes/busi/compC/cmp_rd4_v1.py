#!/usr/bin/env python
"""cmp_rd4_v1.py (busi lane, PERF RD-4): two engine runs (the ladder route A, the tabulated route B) read against the budget declared in
busi/notes/PERF_RD4_build_v1.md before the tests (reviewer A redesign sec. 7): every set decision identical, and the values within the budgets.
The records and npz rows at the common clocks are walked as cmp_symc_v1.py walks them (its walk and leaf rules: every bool/int/string
leaf a label, every float leaf a value; the wall seconds and the timing dicts' *_s set aside). Then:
  labels: the declared changes (the Cholesky counters: the keys cholesky, cholesky_calls, walk_cholesky, and the strings that print them,
      compared with the Cholesky numbers masked) are listed apart; every other label difference counts against "every set decision identical";
  values (the budget lines): E/E0 relative 1e-9; the roots trapping_zeros absolute 3.4e-7; min_N_minus_absv absolute 4.6e-9; the site fields
      of the fields npz (N, v, N_applied, v_applied) absolute 1e-7; the largest other value differences printed for information.
Usage. python cmp_rd4_v1.py A_DIR/TAG B_DIR/TAG [--after T] [--strip label,engine_v34] [--top 12]
"""
import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cmp_symc_v1 as CS  # noqa: E402

DECL_KEYS = {"cholesky", "cholesky_calls", "walk_cholesky"}
CHOL = re.compile(r"(Cholesky|cholesky)[ =:]*\d+|\d+ Cholesky")
SITE = ("N", "v", "N_applied", "v_applied")
BUDGET = (("E/E0 relative", "record/E_over_E0", 1, 1e-9), ("roots trapping_zeros absolute (a)", "record/trapping_zeros[]", 0, 3.4e-7),
          ("min(N - |v|) absolute", "record/min_N_minus_absv", 0, 4.6e-9))


def declared(path, kind, v):
    last = path.rsplit("/", 1)[-1]
    last = re.sub(r"\[\d+\]$", "", last)
    if last in DECL_KEYS:
        return True
    if kind == "label" and isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], str):
        return CHOL.sub("Cholesky #", v[0]) == CHOL.sub("Cholesky #", v[1]) and v[0] != v[1]
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("A"); ap.add_argument("B"); ap.add_argument("--after", type=float, default=None)
    ap.add_argument("--strip", default=""); ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    CS.SKIP.update(k for k in a.strip.split(",") if k)
    S8 = CS.S8
    JA, ZA = S8.load_run(a.A); JB, ZB = S8.load_run(a.B)
    rs = float(JA["args"]["rs"])

    def recs(J):
        return {round(r["T"] / rs, 9): r for r in J["records"] if r.get("E") is not None and (a.after is None or r["T"] / rs > a.after + 1e-9)}
    RA, RB = recs(JA), recs(JB)
    common = sorted(set(RA) & set(RB))
    acc = CS.Acc()
    if set(RA) != set(RB):
        acc.labels.append(("records", "clocks", (sorted(set(RA) - set(RB))[:3], sorted(set(RB) - set(RA))[:3])))
    for t in common:
        CS.walk(RA[t], RB[t], "record@%.5f" % t, acc)
    site = {}
    for nm in ZA:
        if nm not in ZB:
            acc.labels.append((nm, "missing npz", None)); continue
        za = np.load(ZA[nm], allow_pickle=True); zb = np.load(ZB[nm], allow_pickle=True)
        TA = np.asarray(za["T"], float) / rs; TB = np.asarray(zb["T"], float) / rs
        for k in za.files:
            if k == "T" or k not in zb.files:
                continue
            x = za[k]; y = zb[k]
            if x.dtype.kind in "SU" or (x.dtype.kind == "O" and x.ndim == 0):
                continue
            if x.ndim >= 1 and x.shape[0] == len(TA) and y.ndim >= 1 and y.shape[0] == len(TB):
                for t in common:
                    ia = int(np.argmin(np.abs(TA - t))); ib = int(np.argmin(np.abs(TB - t)))
                    CS.walk(x[ia], y[ib], "%s/%s@%.5f" % (nm, k, t), acc)
                    if k in SITE and x.dtype.kind == "f":
                        d = float(np.max(np.abs(np.asarray(x[ia], float) - np.asarray(y[ib], float)))) if np.size(x[ia]) else 0.0
                        site[(nm, k)] = max(site.get((nm, k), 0.0), d)
            else:
                CS.walk(x, y, "%s/%s" % (nm, k), acc)
    decl = [q for q in acc.labels if declared(*q)]
    other = [q for q in acc.labels if not declared(*q)]
    lines = []; ok_all = len(common) > 0
    ok = not other
    ok_all &= ok
    lines.append("%s every set decision identical: %d label differences other than the declared Cholesky counters (%d declared: %s) over %d common records%s" % (
        "PASS" if ok else "FAIL", len(other), len(decl), ", ".join(sorted(set(CS.keyname(p) for p, _k, _v in decl)))[:300], len(common),
        (" after T/rs %g" % a.after) if a.after is not None else ""))
    for p, kind, v in other[:30]:
        lines.append("    LABEL %s %s %s" % (p, kind, v))
    for name, key, col, tol in BUDGET:
        q = acc.vals.get(key)
        x = (q[col] if q is not None else 0.0)
        ok = x <= tol; ok_all &= ok
        lines.append("%s %s: max %.3e (budget %.1e)%s" % ("PASS" if ok else "FAIL", name, x, tol, (" at %s" % q[2]) if q is not None else " (identical)"))
    sx = max(site.values()) if site else 0.0
    ok = sx <= 1e-7; ok_all &= ok
    lines.append("%s site fields %s absolute: max %.3e (budget 1.0e-07)%s" % ("PASS" if ok else "FAIL", "/".join(SITE), sx,
                 (" per key %s" % ", ".join("%s/%s %.2e" % (nm, k, v) for (nm, k), v in sorted(site.items()))) if site else " (no site field rows)"))
    top = sorted(acc.vals.items(), key=lambda kv: -kv[1][1])[: a.top]
    lines.append("  %d float keys differ; the largest relative differences (information):" % len(acc.vals))
    for k, (ab, rel, p) in top:
        lines.append("    value %-80s rel %.3e abs %.3e (at %s)" % (k[:80], rel, ab, p[:100]))
    lines.insert(0, "%s cmp_rd4_v1 %s against %s" % ("PASS" if ok_all else "FAIL", a.A, a.B))
    print("\n".join(lines))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
