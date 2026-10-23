#!/usr/bin/env python
"""cmp_symc_v1.py (busi lane, PERF SYM-C1/SYM-C2): two engine runs compared in the tolerance class with exact labels (the reviewer A symmetry
audit's A/B acceptance, sec. 3): every record at the common clocks and every npz row there, walked as engine_twofield_v8_selftest.deep_diff does
(the wall seconds and the timing dicts' *_s set aside), with the leaves split in two:
  labels, required equal: every bool, int and string (masks, n_plus, n_occ, anchors' counts, crossings, swaps, flagged, passage verdicts and
      branches, routes, modes, fallbacks, pieces, Cholesky counts, iterations, eigendecomposition counts), None against a number, the key sets,
      list lengths, and every bool/int npz array (the spectra npz masks) bitwise; except the diagnostic keys DIAG (the sector rotation's own
      count and impurity, which the sector route sets by construction);
  values, reported: every float leaf and float/complex npz array, by its key (the path with the indices dropped): the largest absolute difference
      and the largest relative one, |x - y| / max(|x|, |y|) for scalars and max |x - y| / max |x| for arrays.
Acceptance printed for the named classes (--tol-e, --tol-E): E and E_over_E0 relative, the spectra npz eigenvalue rows relative to their max |e|.

Usage. python cmp_symc_v1.py A_DIR/TAG B_DIR/TAG [--after T] [--tol-e 1e-13] [--tol-E 1e-12] [--top 30] [--strip label,engine_v34]
"""
import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_twofield_v8_selftest as S8  # noqa: E402

SKIP = set(S8.SKIP_KEYS) | {"seconds_sector"}
DIAG = {"rotated_blocks", "sigma_impurity_max"}


def keyname(path):
    return re.sub(r"\[\d+\]", "[]", re.sub(r"^record@[0-9.]+", "record", re.sub(r"@[0-9.]+$", "", path)))


class Acc(object):
    def __init__(self):
        self.labels = []; self.diag = []; self.vals = {}

    def val(self, path, ab, rel):
        k = keyname(path); q = self.vals.get(k)
        if q is None or rel > q[1] or (rel == q[1] and ab > q[0]):
            self.vals[k] = (ab, rel, path)
        elif ab > q[0]:
            self.vals[k] = (ab, max(rel, q[1]), q[2])


def walk(x, y, path, acc):
    last = path.rsplit("/", 1)[-1]
    if isinstance(x, dict) and isinstance(y, dict):
        if path.endswith("timing") or path.endswith("TM") or path.endswith("TML"):
            x = {k: v for k, v in x.items() if not str(k).endswith("_s")}; y = {k: v for k, v in y.items() if not str(k).endswith("_s")}
        if set(x) != set(y):
            acc.labels.append((path, "keys", sorted(str(k) for k in set(x) ^ set(y))))
        for k in x:
            if k in y and k not in SKIP:
                walk(x[k], y[k], path + "/" + str(k), acc)
        return
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        if len(x) != len(y):
            (acc.diag if last in DIAG else acc.labels).append((path, "length", (len(x), len(y))))
            return
        for i, (p, q) in enumerate(zip(x, y)):
            walk(p, q, path + "[%d]" % i, acc)
        return
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        xa = np.asarray(x); ya = np.asarray(y)
        if xa.shape != ya.shape:
            acc.labels.append((path, "shape", (xa.shape, ya.shape))); return
        if xa.dtype.kind == "O" or ya.dtype.kind == "O":
            for i, (p, q) in enumerate(zip(xa.ravel(), ya.ravel())):
                walk(p, q, path + "[%d]" % i, acc)
            return
        if xa.dtype.kind in "SU" or ya.dtype.kind in "SU":
            if not np.array_equal(xa, ya):
                acc.labels.append((path, "text", None))
            return
        if xa.dtype.kind in "fc" or ya.dtype.kind in "fc":
            if xa.size and not np.array_equal(xa, ya, equal_nan=True):
                d = np.abs(xa.astype(complex) - ya.astype(complex)); fin = np.isfinite(d)
                if not np.array_equal(np.isfinite(xa), np.isfinite(ya)):
                    acc.labels.append((path, "nan pattern", None)); return
                ab = float(np.max(d[fin])) if fin.any() else 0.0
                sc = float(np.max(np.abs(xa[np.isfinite(xa)]))) if np.isfinite(xa).any() else 0.0
                acc.val(path, ab, ab / sc if sc > 0 else (0.0 if ab == 0 else float("inf")))
            return
        if not np.array_equal(xa, ya):
            acc.labels.append((path, "array", int(np.sum(xa != ya))))
        return
    if isinstance(x, bool) or isinstance(y, bool) or isinstance(x, (str,)) or isinstance(y, (str,)) or x is None or y is None:
        if x != y and not (isinstance(x, float) and isinstance(y, float) and x != x and y != y):
            (acc.diag if last in DIAG else acc.labels).append((path, "label", (x, y) if not isinstance(x, str) else (str(x)[:120], str(y)[:120])))
        return
    if isinstance(x, (int, np.integer)) and isinstance(y, (int, np.integer)):
        if int(x) != int(y):
            (acc.diag if last in DIAG else acc.labels).append((path, "int", (int(x), int(y))))
        return
    if isinstance(x, (int, float, np.integer, np.floating)) and isinstance(y, (int, float, np.integer, np.floating)):
        fx, fy = float(x), float(y)
        if fx != fy and not (fx != fx and fy != fy):
            if last in DIAG:
                acc.diag.append((path, "float", (fx, fy)))
            ab = abs(fx - fy); den = max(abs(fx), abs(fy))
            acc.val(path, ab, ab / den if den > 0 else 0.0)
        return
    if type(x) != type(y):
        acc.labels.append((path, "type", (type(x).__name__, type(y).__name__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("A"); ap.add_argument("B"); ap.add_argument("--after", type=float, default=None)
    ap.add_argument("--tol-e", type=float, default=1e-13); ap.add_argument("--tol-E", type=float, default=1e-12); ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--strip", default="", help="comma-separated record keys set aside (a resume: label,engine_v34 carry the run paths)")
    a = ap.parse_args()
    SKIP.update(k for k in a.strip.split(",") if k)
    JA, ZA = S8.load_run(a.A); JB, ZB = S8.load_run(a.B)
    rs = float(JA["args"]["rs"])

    def recs(J):
        return {round(r["T"] / rs, 9): r for r in J["records"] if r.get("E") is not None and (a.after is None or r["T"] / rs > a.after + 1e-9)}
    RA, RB = recs(JA), recs(JB)
    common = sorted(set(RA) & set(RB))
    acc = Acc()
    if set(RA) != set(RB):
        acc.labels.append(("records", "clocks", (sorted(set(RA) - set(RB))[:3], sorted(set(RB) - set(RA))[:3])))
    for t in common:
        walk(RA[t], RB[t], "record@%.5f" % t, acc)
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
                    walk(x[ia], y[ib], "%s/%s@%.5f" % (nm, k, t), acc)
            else:
                walk(x, y, "%s/%s" % (nm, k), acc)
    jE = [v for k, v in acc.vals.items() if k in ("record/E", "record/E_over_E0")]
    E_rel = max([v[1] for v in jE] or [0.0])
    e_rows = [v for k, v in acc.vals.items() if k.startswith("spectra/") and re.search(r"/e_", k)]
    e_rel = max([v[1] for v in e_rows] or [0.0])
    ok = (not acc.labels) and len(common) > 0 and E_rel <= a.tol_E and e_rel <= a.tol_e
    print("%s tolerance compare %s against %s over %d common records%s: %d label differences; E and E/E0 max relative %.3e (tol %.0e); spectra eigenvalue rows max relative %.3e (tol %.0e); %d diagnostic differences (%s); %d float keys differ" % (
        "PASS" if ok else "FAIL", a.A, a.B, len(common), (" after T/rs %g" % a.after) if a.after is not None else "", len(acc.labels), E_rel, a.tol_E, e_rel, a.tol_e,
        len(acc.diag), ", ".join(sorted(set(keyname(p) for p, _k, _v in acc.diag)))[:300], len(acc.vals)))
    for p, kind, v in acc.labels[:40]:
        print("  LABEL %s %s %s" % (p, kind, v))
    top = sorted(acc.vals.items(), key=lambda kv: -kv[1][1])[: a.top]
    for k, (ab, rel, p) in top:
        print("  value %-90s rel %.3e abs %.3e (at %s)" % (k[:90], rel, ab, p[:110]))
    for k in ("record/E", "record/E_over_E0"):
        if k in acc.vals:
            print("  named %s rel %.3e abs %.3e" % (k, acc.vals[k][1], acc.vals[k][0]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
