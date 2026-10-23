#!/usr/bin/env python
"""massless_applied_root_v1.py: the applied-root readout of MASSLESS_ball_pathfinder v1.2 section 9.5 (i), on stored files only.

On every record of a run the applied (back-averaged) fields N_applied and v_applied, stored per record in the run's
fields npz, are read on the grid their length matches (the chain sites R, else the field nodes r_nodes, else a provisional
grid 1..nsites a that is named as such); the zeros of N_applied - abs(v_applied) are found by sign change between adjacent
grid points with linear interpolation (the resolution is the grid's; the records' dense roots were found on an ell/64
mesh with bracket refinement) together with the minimum of N_applied - abs(v_applied) and its radius. The dense fields N
and v of the same npz are read the same way (the dense fields sampled on their grid), and the records' own columns
trapping_zeros and min_N_minus_absv are read from the json as the stored dense reference. Nothing is re-run and no
stored file is written. Outputs: a json and a md with, per column, the first and last record with a root, the
uninterrupted stretches, the deepest minimum, the outer root's maximum and the inner root's minimum (in a and in ell),
and the per-record table over the root window, S11's records 1.125 to 1.5 marked c.
usage: python massless_applied_root_v1.py --run DIR/TAG --md OUT.md --out-json OUT.json
       python massless_applied_root_v1.py --selftest
"""
import argparse
import json
import os
import sys

import numpy as np


class _NPZ(dict):
    """a dict standing in for a loaded npz in the self-test (files = keys)"""
    files = property(lambda self: list(self.keys()))


def roots_and_min(r, f):
    """zeros of the grid-sampled f by sign change with linear interpolation; f's minimum and its radius; non-finite samples are left out"""
    ok = np.isfinite(f)
    r, f = np.asarray(r, float)[ok], np.asarray(f, float)[ok]
    if len(f) < 2:
        return [], float("inf"), float("nan")
    s = np.sign(f)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    roots = [float(r[i] - f[i] * (r[i + 1] - r[i]) / (f[i + 1] - f[i])) for i in idx]
    roots += [float(r[i]) for i in np.where(f == 0)[0]]
    k = int(np.argmin(f))
    return sorted(roots), float(f[k]), float(r[k])


def orient(A, nrec, grids):
    """a per-record field array as (nrec, ngrid) with the name of the grid its other axis matches, else None"""
    A = np.asarray(A, float)
    if A.ndim != 2:
        return None
    for name, r in grids.items():
        if A.shape == (nrec, len(r)):
            return A, name
        if A.shape == (len(r), nrec) and len(r) != nrec:
            return A.T, name
    return None


def align(T_rec, T_npz, rs):
    """the npz row of each record and the unit of the npz T: 1 (T/r_s) or rs (T in a)"""
    Tz = np.asarray(T_npz, float).ravel()
    for unit in (1.0, rs):
        t = Tz / unit
        k = np.array([int(np.argmin(np.abs(t - x))) for x in T_rec])
        if np.all(np.abs(t[k] - np.asarray(T_rec)) < 1e-6):
            return k, unit
    sys.exit("the npz T (%d values) matches the records' T_over_rs (%d) neither in r_s nor in a" % (len(Tz), len(T_rec)))


def grids_of(Z, J):
    """the candidate grids of the npz: R and r_nodes when 1-D, else the provisional 1..nsites"""
    G = {n: np.asarray(Z[n], float).ravel() for n in ("R", "r_nodes") if n in Z.files and np.asarray(Z[n]).ndim == 1}
    if "R" not in G and "nsites" in J["args"]:
        G["sites 1..nsites (provisional: the npz R unusable)"] = np.arange(1, int(J["args"]["nsites"]) + 1, dtype=float)
    return G


def _radii(z):
    """the radii in a stored trapping_zeros value: a number, a list of numbers, a list of dicts carrying r, radius or root, a list of lists with the radius first, or a dict carrying roots, r, radii or zeros; anything else is skipped, so an unreadable structure gives no roots (the sample in the output shows it) instead of a crash"""
    if z is None or isinstance(z, bool):
        return []
    if isinstance(z, (int, float)):
        return [float(z)]
    if isinstance(z, dict):
        for key in ("roots", "r", "radii", "zeros"):
            if key in z:
                return _radii(z[key])
        return []
    out = []
    for x in (z if isinstance(z, (list, tuple)) else []):
        if isinstance(x, dict):
            for key in ("r", "radius", "root"):
                if key in x:
                    out += _radii(x[key])
                    break
        elif isinstance(x, (list, tuple)):
            out += _radii(x[0]) if x else []
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            out.append(float(x))
    return out


def stored_roots(rec):
    """the record's dense roots as the json stores them, its minimum and radius; an unreadable trapping_zeros gives no roots"""
    zs = sorted(_radii(rec.get("trapping_zeros")))
    fm, rm = rec.get("min_N_minus_absv"), rec.get("r_min_N_minus_absv")
    return zs, (float(fm) if fm is not None else float("inf")), (float(rm) if rm is not None else float("nan"))


def summary(T, ell, rows):
    """the per-history summary of one root column (rows: roots, fmin, rmin per record)"""
    deep = min(range(len(rows)), key=lambda i: rows[i]["fmin"])
    out = {"records_with_root": 0, "deepest": [rows[deep]["fmin"], T[deep], rows[deep]["rmin"]]}
    has = [i for i, w in enumerate(rows) if w["roots"]]
    if not has:
        return out
    stretches, a, p = [], has[0], has[0]
    for i in has[1:]:
        if i != p + 1:
            stretches.append([T[a], T[p], p - a + 1])
            a = i
        p = i
    stretches.append([T[a], T[p], p - a + 1])
    outer = max(has, key=lambda i: rows[i]["roots"][-1])
    inner = min(has, key=lambda i: rows[i]["roots"][0])
    out.update({"records_with_root": len(has), "first_record": T[has[0]], "last_record": T[has[-1]], "stretches": stretches,
                "outer_root_max": [rows[outer]["roots"][-1], T[outer], rows[outer]["roots"][-1] / ell],
                "inner_root_min": [rows[inner]["roots"][0], T[inner], rows[inner]["roots"][0] / ell]})
    return out


def readout(J, Z, run):
    rs, g, ell = float(J["args"]["rs"]), float(J["args"]["g"]), float(J["args"]["ell"])
    recs = J["records"]
    T = [float(x["T_over_rs"]) for x in recs]
    k, unit = align(T, Z["T"], rs)
    nz = len(np.asarray(Z["T"]).ravel())
    G = grids_of(Z, J)
    cols, notes = {}, ["the npz T read in %s" % ("r_s" if unit == 1.0 else "a")]
    for label, nk, vk in (("applied", "N_applied", "v_applied"), ("dense_at_grid", "N", "v")):
        if nk not in Z.files or vk not in Z.files:
            notes.append("%s skipped: %s or %s not in the npz" % (label, nk, vk))
            continue
        a, b = orient(Z[nk], nz, G), orient(Z[vk], nz, G)
        if a is None or b is None or a[1] != b[1]:
            notes.append("%s skipped: %s %s and %s %s match no grid among %s" % (label, nk, np.shape(Z[nk]), vk, np.shape(Z[vk]), {n: len(r) for n, r in G.items()}))
            continue
        N, v, r = a[0], b[0], G[a[1]]
        rows = [dict(zip(("roots", "fmin", "rmin"), roots_and_min(r, N[k[i]] - np.abs(v[k[i]])))) for i in range(len(T))]
        cols[label] = {"grid": a[1], "points": len(r), "rows": rows, "summary": summary(T, ell, rows)}
    rows = [dict(zip(("roots", "fmin", "rmin"), stored_roots(x))) for x in recs]
    if not any(w["roots"] for w in rows) and any(w["fmin"] < 0 for w in rows):
        notes.append("stored_dense: no root read from trapping_zeros although min_N_minus_absv < 0 on some records; the stored structure (the sample below) is not one stored_roots reads, and the stored column is not to be used until it is")
    cols["stored_dense"] = {"grid": "the records' columns", "points": None, "rows": rows, "summary": summary(T, ell, rows)}
    sample = next((x.get("trapping_zeros") for x in recs if x.get("trapping_zeros")), None)
    return {"script": "massless_applied_root_v1.py", "run": run, "rs": rs, "g": g, "ell": ell, "records": len(T), "T_over_rs": T,
            "columns": cols, "notes": notes, "certified_segment_T_over_rs": ([1.125, 1.5] if abs(rs - 16.0) < 1e-9 else None),
            "stored_trapping_zeros_sample": repr(sample)[:200]}


def fmt_roots(zs):
    return ", ".join("%.3f" % z for z in zs) if zs else "none"


def write_md(out, path):
    T, C, ell, seg = out["T_over_rs"], out["columns"], out["ell"], out["certified_segment_T_over_rs"]
    order = [c for c in ("stored_dense", "dense_at_grid", "applied") if c in C]
    S = {c: C[c]["summary"] for c in order}
    L = ["# The applied-root readout on %s" % out["run"], "",
         "r_s/a %g, g %g, ell %g a, %d records, read by massless_applied_root_v1.py from the stored json and fields npz (nothing re-run). Columns: %s. Notes: %s. The stored trapping_zeros of the first record carrying one: %s."
         % (out["rs"], out["g"], ell, out["records"],
            "; ".join("%s on %s%s" % (c, C[c]["grid"], (" (%d points)" % C[c]["points"]) if C[c]["points"] else "") for c in order),
            "; ".join(out["notes"]), out["stored_trapping_zeros_sample"]),
         "", "| quantity | " + " | ".join(order) + " |", "|---" * (len(order) + 1) + "|"]

    def q(name, fn, needs_root=True):
        return "| %s | %s |" % (name, " | ".join(fn(S[c]) if (S[c]["records_with_root"] or not needs_root) else "none" for c in order))
    L += [q("records with a root", lambda s: "%d" % s["records_with_root"], False),
          q("first root record T/r_s", lambda s: "%.4f" % s["first_record"]),
          q("last root record T/r_s", lambda s: "%.4f" % s["last_record"]),
          q("uninterrupted stretches (first, last, records)", lambda s: "; ".join("%.4f to %.4f (%d)" % tuple(x) for x in s["stretches"])),
          q("deepest min(N - abs(v)) (at T/r_s; at r a)", lambda s: "%.4f (%.4f; %.3f)" % tuple(s["deepest"]), False),
          q("outer root maximum a (at T/r_s; in ell)", lambda s: "%.3f (%.4f; %.2f)" % tuple(s["outer_root_max"])),
          q("inner root minimum a (at T/r_s; in ell)", lambda s: "%.3f (%.4f; %.2f)" % tuple(s["inner_root_min"]))]
    idx = [i for c in order for i in range(len(T)) if C[c]["rows"][i]["roots"]]
    lo, hi = (max(0, min(idx) - 2), min(len(T) - 1, max(idx) + 2)) if idx else (0, min(len(T) - 1, 4))
    L += ["", "Per record over the root window (records %d to %d of 0 to %d; mark c: a record of S11's certified segment %s):" % (lo, hi, len(T) - 1, seg), "",
          "| T/r_s | mark | " + " | ".join("%s roots (a) | %s min (at r a)" % (c, c) for c in order) + " |", "|---" * (2 + 2 * len(order)) + "|"]
    for i in range(lo, hi + 1):
        m = "c" if seg and seg[0] - 1e-9 <= T[i] <= seg[1] + 1e-9 else ""
        L.append("| %.4f | %s | " % (T[i], m) + " | ".join("%s | %.4f (%.3f)" % (fmt_roots(w["roots"]), w["fmin"], w["rmin"]) for w in (C[c]["rows"][i] for c in order)) + " |")
    L += ["", "- the applied and dense-at-grid roots are zeros of N - abs(v) between adjacent grid points by linear interpolation (the grid's resolution); the stored dense column is the records' own; every number above is read or computed by massless_applied_root_v1.py from the files named."]
    open(path, "w").write("\n".join(L) + "\n")


def selftest():
    rs, g, ell = 16.0, 0.25, 2.0
    r = np.arange(0.5, 40.0, 1.0)
    T = np.arange(0.0, 2.0001, 0.125)
    A = np.where((T > 0.5 - 1e-9) & (T < 1.5 + 1e-9), 1.2, 0.8)
    v = -A[:, None] * np.exp(-((r[None, :] - 10.0) ** 2) / 32.0)
    N = np.ones_like(v)
    h = float(np.sqrt(32.0 * np.log(1.2)))
    recs = [{"T_over_rs": float(t), "trapping_zeros": ([10.0 - h, 10.0 + h] if a > 1 else []), "min_N_minus_absv": 1.0 - float(a), "r_min_N_minus_absv": 10.0} for t, a in zip(T, A)]
    J = {"args": {"rs": rs, "g": g, "ell": ell, "nsites": len(r)}, "records": recs}
    Z = _NPZ({"T": T * rs, "R": r, "r_nodes": np.linspace(0.5, 40.0, 7), "N_applied": N, "v_applied": v, "N": N.T, "v": v.T})
    out = readout(J, Z, "selftest")
    assert out["notes"][0].endswith(" a"), out["notes"]
    for c in ("applied", "dense_at_grid"):
        s, col = out["columns"][c]["summary"], out["columns"][c]
        assert col["grid"] == "R" and col["points"] == len(r), (c, col["grid"])
        assert s["records_with_root"] == 9 and abs(s["first_record"] - 0.5) < 1e-9 and abs(s["last_record"] - 1.5) < 1e-9, s
        assert len(s["stretches"]) == 1 and s["stretches"][0][2] == 9, s
        assert abs(s["outer_root_max"][0] - (10.0 + h)) < 0.05 and abs(s["inner_root_min"][0] - (10.0 - h)) < 0.05, s
        assert abs(s["deepest"][0] + 0.2) < 0.02 and abs(s["deepest"][2] - 10.0) < 0.6, s
    s = out["columns"]["stored_dense"]["summary"]
    assert s["records_with_root"] == 9 and abs(s["outer_root_max"][0] - (10.0 + h)) < 1e-9 and abs(s["deepest"][0] + 0.2) < 1e-9, s
    write_md(out, os.devnull)
    print("SELFTEST PASS: 9 of 17 synthetic records carry roots at 10 -/+ %.5f a (found within 0.05 a on the grid), the minimum -0.2 within 0.02, the npz T read in a, the transposed layout oriented, the stored column reproduced, the md composed" % h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--md")
    ap.add_argument("--out-json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.run and a.md and a.out_json):
        ap.error("--run, --md and --out-json are needed (or --selftest)")
    d, tag = os.path.split(a.run.rstrip("/"))
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    out = readout(J, Z, a.run)
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    write_md(out, a.md)
    for c, col in out["columns"].items():
        print("%s: %s" % (c, json.dumps(col["summary"], default=float)))
    print("composed %s (%d records)" % (a.md, out["records"]))


if __name__ == "__main__":
    main()
