#!/usr/bin/env python
"""tolcmp_v1.py (busi lane, reviewer A perf tolerance ruling v1): two engine runs compared by quantity class, independently of the builders'
comparators. Every record at the common clocks after --after, plus the npz rows there (fields, spectra, handoff, s11) and, optionally, one slice.

Classes (the ruling's table):
  energies     E (relative), E_over_E0 (absolute)
  roots        R_t, r_in, R_c, r_min_N_minus_absv, trapping_zeros[*], threshold_radius/* (absolute, lattice units); R_sh exact
  depth        min_N_minus_absv (absolute)
  named        N_min, kappa_T, H_in, E_dust_shells, E_pairs_shells, pairs_inside_Rt, pairs_outside_Rt, misner_sharp_at_Rt, v_at_Rc,
               min_occupied_weight, M1_I_E, R2_minus_R0_F, commutator_h_R2_F, dE_dT, dHin_dT, dRc_dT (relative)
  indicator    indicator/* floats (B_estimate, delta_rtol ...) reported with both values; indicator_pass as a bit
  counters     int leaves whose key is a work counter (nfev, iterations, solves, eigendecompositions, hits, misses, products, pieces,
               probes, eigvals, eigh, calls, evaluations, attempts? no: attempts is a decision) -> reported
  decisions    every other bool / int / str / None leaf, list lengths and key sets -> must be identical
  other floats reported (max abs, max rel per key)
  fields       npz rows: max |d| and max |d| / max |row|
  spectra      masks identical (uint8 rows); e rows max |d| and relative to max |e|; nocc rows
  slice        masks per channel identical; e, densities, dust weights, dg nodes, level

Skipped: walls (seconds, timing/*_s, keys ending _s, slice_saved, start), path strings (label, record_label, engine_v34, engine),
hbar_prop_v36w2, iteration_history (an iterate-level trace whose length is the iteration count; reported through 'iterations').

Usage: tolcmp_v1.py A_DIR B_DIR --tag TAG [--after T] [--rs 64] [--out out.json] [--slice-a P --slice-b P] [--nameA x --nameB y]
       [--prod-json P --prod-npz-dir D] (B taken from a production json + a mirror dir instead of a run dir)
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np

SKIP_KEYS = {"seconds", "timing", "slice_saved", "label", "record_label", "engine_v34", "engine", "hbar_prop_v36w2", "start",
             "iteration_history", "script", "path", "out", "slice_dir", "resume_slice", "resume_run"}
COUNTER_RE = re.compile(r"(^|/)(nfev|solve_nfev|iterations|iteration|eigendecompositions|solves|n_solves|hits|misses|products|pieces|"
                        r"probes|probes_in_crossing|eigvals|eigh|calls|evaluations|n_eval|nit|njev|nlu|computed|evicted|hit|"
                        r"rotated_blocks|light_sector_eigh|ladder_sector_pairs|sectorize_exact|sectorize_rotation_kept|cholesky|"
                        r"cholesky_calls|band_plan_hit|band_plan_rebuild|gram_gathers|gram_d2h_bytes_avoided|kernel_cache_hits|"
                        r"n_products|n_jvp|steps|nsteps|nfev_total|it|eigh_count|eigs|n_eigh|eig_computed|eig_hit|eig_evicted)$")
ROOT_KEYS = {"R_t", "r_in", "R_c", "r_min_N_minus_absv"}
NAMED_KEYS = {"N_min", "kappa_T", "H_in", "E_dust_shells", "E_pairs_shells", "pairs_inside_Rt", "pairs_outside_Rt", "misner_sharp_at_Rt",
              "v_at_Rc", "min_occupied_weight", "M1_I_E", "R_t_over_2gE", "W_over_2g_at_Rt", "R_t_over_2g_MS", "four_gE_kappa_T"}
RESIDUAL_KEYS = {"R2_minus_R0_F", "commutator_h_R2_F", "dE_dT", "dHin_dT", "dRc_dT", "balance_d_E_minus_Hin_dT", "intermediate_weights_total"}
TRACE_RE = re.compile(r"(/history|_hist|node_residuals|/iteration_history)")
DIAG_FLOAT = {"sigma_impurity_max"}


def leaves(x, p=""):
    """(path, value) for every scalar leaf; lists give '#len'."""
    if isinstance(x, dict):
        for k, v in x.items():
            ks = str(k)
            if ks in SKIP_KEYS or ks.endswith("_s"):
                continue
            yield from leaves(v, p + "/" + ks)
    elif isinstance(x, (list, tuple)):
        yield p + "#len", len(x)
        for i, v in enumerate(x):
            yield from leaves(v, p + "[%d]" % i)
    else:
        yield p, x


def keyname(path):
    return re.sub(r"\[\d+\]", "[]", path)


def classify(path, x, y):
    """-> class name for a float leaf."""
    last = path.rsplit("/", 1)[-1]
    last = re.sub(r"\[\d+\]$", "", last)
    top = path.split("/")[1] if path.count("/") >= 1 else ""
    if path == "/E":
        return "E"
    if path == "/E_over_E0":
        return "E_over_E0"
    if path == "/R_sh":
        return "R_sh"
    if path == "/r_min_N_minus_absv":
        return "argmin"
    if path.lstrip("/") in ROOT_KEYS or top in ("trapping_zeros", "threshold_radius"):
        return "roots"
    if path == "/min_N_minus_absv":
        return "depth"
    if path.lstrip("/") in NAMED_KEYS:
        return "named"
    if path.lstrip("/") in RESIDUAL_KEYS:
        return "residual"
    if TRACE_RE.search(path):
        return "trace"
    if top == "indicator":
        return "indicator"
    if last in DIAG_FLOAT:
        return "diag"
    return "other"


class Acc(object):
    def __init__(self):
        self.cls = {}      # class -> list of (abs, rel, path)
        self.other = {}    # keyname -> (abs, rel, path)
        self.decisions = []
        self.counters = []
        self.bits = []
        self.perkey = {}

    def add(self, c, ab, rel, path):
        self.cls.setdefault(c, []).append((ab, rel, path))


def cmp_record(a, b, acc, t):
    La, Lb = dict(leaves(a)), dict(leaves(b))
    for k in sorted(set(La) | set(Lb)):
        x, y = La.get(k, "<absent>"), Lb.get(k, "<absent>")
        p = "%.5f%s" % (t, k)
        kn = keyname(k)
        if x == "<absent>" or y == "<absent>":
            if kn.startswith("/M9") or kn.startswith("/fields_rd2") or kn.startswith("/perf") or TRACE_RE.search(kn):
                acc.counters.append((p, "absent", (x if y == "<absent>" else "<absent>", y if x == "<absent>" else "<absent>")))
            else:
                acc.decisions.append((p, "absent", (str(x)[:60], str(y)[:60])))
            continue
        if isinstance(x, bool) or isinstance(y, bool):
            if x != y:
                if k.endswith("indicator_pass"):
                    acc.bits.append((p, x, y))
                else:
                    acc.decisions.append((p, "bool", (x, y)))
            continue
        if isinstance(x, str) or isinstance(y, str) or x is None or y is None:
            if x != y:
                acc.decisions.append((p, "label", (str(x)[:80], str(y)[:80])))
            continue
        if isinstance(x, int) and isinstance(y, int):
            if x != y:
                if COUNTER_RE.search(kn) or TRACE_RE.search(kn):
                    acc.counters.append((p, "int", (x, y)))
                else:
                    acc.decisions.append((p, "int", (x, y)))
            continue
        fx, fy = float(x), float(y)
        if (math.isnan(fx) and math.isnan(fy)):
            continue
        if math.isnan(fx) != math.isnan(fy) or math.isinf(fx) != math.isinf(fy):
            acc.decisions.append((p, "nan", (fx, fy))); continue
        d = abs(fx - fy)
        den = max(abs(fx), abs(fy))
        rel = d / den if den > 0 else 0.0
        c = classify(k, fx, fy)
        if c == "other":
            q = acc.other.get(kn)
            if q is None or rel > q[1]:
                acc.other[kn] = (d, rel, p, fx, fy)
        elif c == "indicator":
            acc.add("indicator", d, rel, "%s (%.6e vs %.6e)" % (p, fx, fy))
        elif c in ("trace", "residual"):
            acc.add(c, d, rel, "%s (%.3e vs %.3e)" % (p, fx, fy))
        else:
            acc.add(c, d, rel, "%s (%.12g vs %.12g)" % (p, fx, fy))
        if c in ("roots", "named", "argmin"):
            q = acc.perkey.setdefault(c, {}).get(kn)
            if q is None or d > q[0]:
                acc.perkey[c][kn] = (d, rel, p, fx, fy)


def rows_by_T(Z, rs):
    T = np.asarray(Z["T"], float)
    return {round(float(t) / rs, 6): i for i, t in enumerate(T)}


def cmp_rows(ZA, ZB, ts, rs, name, res):
    ia, ib = rows_by_T(ZA, rs), rows_by_T(ZB, rs)
    keys = [k for k in ZA.files if k in ZB.files and k != "T" and ZA[k].ndim >= 1 and ZA[k].shape[0] == len(ia) and ZB[k].shape[0] == len(ib)]
    for k in keys:
        xa, xb = ZA[k], ZB[k]
        if xa.dtype.kind == "O":
            for t in ts:
                if t in ia and t in ib:
                    ra, rb = np.asarray(xa[ia[t]], float), np.asarray(xb[ib[t]], float)
                    if ra.shape != rb.shape:
                        res.setdefault(name + "/" + k, {})[t] = ("shape", ra.shape, rb.shape); continue
                    d = float(np.max(np.abs(ra - rb))) if ra.size else 0.0
                    sc = float(np.max(np.abs(ra))) if ra.size else 0.0
                    res.setdefault(name + "/" + k, {})[t] = (d, d / sc if sc > 0 else 0.0)
            continue
        if xa.dtype.kind in "SU":
            continue
        for t in ts:
            if t in ia and t in ib:
                ra, rb = xa[ia[t]], xb[ib[t]]
                if xa.dtype.kind in "iub":
                    res.setdefault(name + "/" + k, {})[t] = ("mask", int(np.sum(ra != rb)))
                else:
                    ra = np.asarray(ra, float); rb = np.asarray(rb, float)
                    d = float(np.max(np.abs(ra - rb))) if ra.size else 0.0
                    sc = float(np.max(np.abs(ra))) if ra.size else 0.0
                    res.setdefault(name + "/" + k, {})[t] = (d, d / sc if sc > 0 else 0.0)


def slice_cmp(pa, pb):
    A = np.load(pa, allow_pickle=True); B = np.load(pb, allow_pickle=True)
    ks = sorted(int(k.split("_")[1]) for k in A.files if k.startswith("mask_") and k.split("_")[1].isdigit())
    out = {"masks_equal": True, "e": 0.0, "dens_sea": 0.0, "dens_R2": 0.0, "f": 0.0, "nodes": None, "level_equal": None, "channels": len(ks)}
    for k in ks:
        out["masks_equal"] &= bool(np.array_equal(A["mask_%d" % k], B["mask_%d" % k]))
        out["e"] = max(out["e"], float(np.max(np.abs(A["e_%d" % k] - B["e_%d" % k]))))
        for nm, key in (("dens_sea", "sea_%d"), ("dens_R2", "R2_%d")):
            if (key % k) in A.files and (key % k) in B.files:
                da = np.sum(np.abs(A[key % k]) ** 2, axis=1); db = np.sum(np.abs(B[key % k]) ** 2, axis=1)
                out[nm] = max(out[nm], float(np.max(np.abs(da - db))))
        if ("f_%d" % k) in A.files and ("f_%d" % k) in B.files:
            fa, fb = A["f_%d" % k], B["f_%d" % k]
            if fa.shape == fb.shape and fa.size:
                out["f"] = max(out["f"], float(np.max(np.abs(fa - fb))))
            elif fa.shape != fb.shape:
                out["masks_equal"] = False
    if "dg_nodes_N" in A.files and "dg_nodes_N" in B.files:
        out["nodes"] = max(float(np.max(np.abs(A["dg_nodes_N"] - B["dg_nodes_N"]))), float(np.max(np.abs(A["dg_nodes_v"] - B["dg_nodes_v"]))))
    if "dg_level_v36k" in A.files and "dg_level_v36k" in B.files:
        out["level_equal"] = int(A["dg_level_v36k"]) == int(B["dg_level_v36k"])
    # light masks (mask_L*)
    kl = [k for k in A.files if k.startswith("mask_L")]
    for k in kl:
        if k in B.files:
            out["masks_equal"] &= bool(np.array_equal(A[k], B[k]))
    return out


def load_run(d, tag, prod_json=None, prod_dir=None):
    J = json.load(open(prod_json if prod_json else os.path.join(d, "collapse_%s.json" % tag)))
    nd = prod_dir if prod_dir else d
    Z = {}
    for k in ("fields", "spectra", "handoff", "s11"):
        p = os.path.join(nd, "collapse_%s_%s.npz" % (tag, k))
        if os.path.exists(p):
            Z[k] = np.load(p, allow_pickle=True)
    return J, Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("A"); ap.add_argument("B"); ap.add_argument("--tag", required=True); ap.add_argument("--after", type=float, default=None)
    ap.add_argument("--rs", type=float, default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--slice-a", default=None); ap.add_argument("--slice-b", default=None)
    ap.add_argument("--nameA", default="A"); ap.add_argument("--nameB", default="B")
    ap.add_argument("--prod-json", default=None); ap.add_argument("--prod-npz-dir", default=None)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    JA, ZA = load_run(a.A, a.tag)
    JB, ZB = load_run(a.B, a.tag, a.prod_json, a.prod_npz_dir)
    rs = a.rs or float(JA["args"]["rs"])
    args_diff = sorted(k for k in set(JA["args"]) | set(JB["args"]) if JA["args"].get(k) != JB["args"].get(k))
    byT = lambda J: {round(r["T"] / rs, 6): r for r in J["records"] if r.get("E") is not None}
    RA, RB = byT(JA), byT(JB)
    ts = sorted(t for t in set(RA) & set(RB) if a.after is None or t > a.after + 1e-9)
    acc = Acc()
    per = {}
    for t in ts:
        a1 = Acc()
        cmp_record(RA[t], RB[t], a1, t)
        per[t] = {c: max([v[0] for v in L] or [0.0]) for c, L in a1.cls.items()}
        per[t]["E_rel"] = max([v[1] for v in a1.cls.get("E", [])] or [0.0])
        per[t]["named_rel"] = max([v[1] for v in a1.cls.get("named", [])] or [0.0])
        per[t]["n_decisions"] = len(a1.decisions); per[t]["n_counters"] = len(a1.counters); per[t]["bits"] = len(a1.bits)
        for c, L in a1.cls.items():
            acc.cls.setdefault(c, []).extend(L)
        for kn, v in a1.other.items():
            q = acc.other.get(kn)
            if q is None or v[1] > q[1]:
                acc.other[kn] = v
        for c, D in a1.perkey.items():
            for kn, v in D.items():
                q = acc.perkey.setdefault(c, {}).get(kn)
                if q is None or v[0] > q[0]:
                    acc.perkey[c][kn] = v
        acc.decisions.extend(a1.decisions); acc.counters.extend(a1.counters); acc.bits.extend(a1.bits)
    rows = {}
    for k in ("fields", "spectra", "handoff", "s11"):
        if k in ZA and k in ZB:
            cmp_rows(ZA[k], ZB[k], ts, rs, k, rows)
    out = {"A": a.A, "B": a.B, "nameA": a.nameA, "nameB": a.nameB, "records": len(ts), "clocks": ts, "args_differ": args_diff,
           "per_record": {("%.5f" % t): per[t] for t in ts}, "rows": {}, "decisions": acc.decisions, "counters": acc.counters, "bits": acc.bits,
           "other": {k: {"abs": v[0], "rel": v[1], "at": v[2], "x": v[3], "y": v[4]} for k, v in acc.other.items()}}
    print("== %s against %s: %d common records%s; args differ in: %s" % (a.nameA, a.nameB, len(ts), (" after T/rs %g" % a.after) if a.after else "", args_diff or "nothing"))
    print("-- classes (max over the records; where)")
    summ = {}
    for c in ("E", "E_over_E0", "roots", "argmin", "R_sh", "depth", "named", "residual", "indicator", "trace", "diag"):
        L = acc.cls.get(c, [])
        if not L:
            summ[c] = None; print("   %-11s no differences" % c); continue
        ab = max(L, key=lambda v: v[0]); rl = max(L, key=lambda v: v[1])
        summ[c] = {"max_abs": ab[0], "at_abs": ab[2], "max_rel": rl[1], "at_rel": rl[2], "n": len(L)}
        print("   %-11s max abs %.3e at %s; max rel %.3e at %s (%d leaves differ)" % (c, ab[0], ab[2][:70], rl[1], rl[2][:70], len(L)))
    out["summary"] = summ
    out["perkey"] = {c: {k: {"abs": v[0], "rel": v[1], "at": v[2], "x": v[3], "y": v[4]} for k, v in D.items()} for c, D in acc.perkey.items()}
    for c in ("roots", "argmin", "named"):
        for k, v in sorted(acc.perkey.get(c, {}).items(), key=lambda kv: -kv[1][0]):
            print("     %-8s %-40s max abs %.3e rel %.3e at %s" % (c, k, v[0], v[1], v[2][:60]))
    print("-- per record: E/E0 |d|, E rel, roots |d|, depth |d|, named rel, decisions, counters, bits")
    for t in ts:
        p = per[t]
        print("   %.5f  %.2e  %.2e  %.2e  %.2e  %.2e  %d %d %d" % (t, p.get("E_over_E0", 0.0), p.get("E_rel", 0.0), p.get("roots", 0.0), p.get("depth", 0.0), p.get("named_rel", 0.0), p["n_decisions"], p["n_counters"], p["bits"]))
    print("-- decisions differing: %d" % len(acc.decisions))
    byk = {}
    for p, kind, v in acc.decisions:
        byk.setdefault(keyname(p.split(" ", 1)[0][7:]) if False else keyname(re.sub(r"^[0-9.]+", "", p)), []).append((p, kind, v))
    for k, L in sorted(byk.items(), key=lambda kv: -len(kv[1]))[:40]:
        print("   %-70s %d records, e.g. %s %s %s" % (k[:70], len(L), L[0][0][:30], L[0][1], str(L[0][2])[:60]))
    out["decisions_by_key"] = {k: len(L) for k, L in byk.items()}
    print("-- counters differing: %d" % len(acc.counters))
    byk = {}
    for p, kind, v in acc.counters:
        byk.setdefault(keyname(re.sub(r"^[0-9.]+", "", p)), []).append((p, kind, v))
    for k, L in sorted(byk.items(), key=lambda kv: -len(kv[1]))[:30]:
        print("   %-70s %d records, e.g. %s %s" % (k[:70], len(L), L[0][0][:30], str(L[0][2])[:60]))
    out["counters_by_key"] = {k: len(L) for k, L in byk.items()}
    print("-- indicator bits differing: %d" % len(acc.bits))
    for p, x, y in acc.bits[:40]:
        print("   %s %s vs %s" % (p, x, y))
    print("-- other floats (top by relative difference)")
    for k, v in sorted(acc.other.items(), key=lambda kv: -kv[1][1])[: a.top]:
        print("   %-80s rel %.2e abs %.2e (%.6e vs %.6e at %s)" % (k[:80], v[1], v[0], v[3], v[4], v[2][:12]))
    print("-- npz rows (max over records): key: max abs, max rel; masks: rows with any difference")
    for k, D in sorted(rows.items()):
        vals = list(D.values())
        if vals and isinstance(vals[0], tuple) and vals[0][0] == "mask":
            nd = sum(1 for v in vals if v[1] > 0); tot = sum(v[1] for v in vals)
            out["rows"][k] = {"mask_rows_differ": nd, "entries": tot}
            print("   %-28s masks: %d of %d rows differ (%d entries)" % (k, nd, len(vals), tot))
        elif vals and isinstance(vals[0], tuple) and vals[0][0] == "shape":
            out["rows"][k] = {"shape_mismatch": len(vals)}; print("   %-28s shape mismatch" % k)
        else:
            ma = max(v[0] for v in vals); mr = max(v[1] for v in vals); tm = max(D, key=lambda t: D[t][0])
            out["rows"][k] = {"max_abs": ma, "max_rel": mr, "at": tm, "per_record_abs": {("%.5f" % t): D[t][0] for t in D}}
            print("   %-28s abs %.3e rel %.3e (at %.5f)" % (k, ma, mr, tm))
    if a.slice_a and a.slice_b and os.path.exists(a.slice_a) and os.path.exists(a.slice_b):
        s = slice_cmp(a.slice_a, a.slice_b)
        out["slice"] = s
        print("-- slice: masks %s (%d channels), level %s; e %.2e, sea dens %.2e, R2 dens %.2e, dust f %.2e, dg nodes %s" % (
            "equal" if s["masks_equal"] else "DIFFER", s["channels"], s["level_equal"], s["e"], s["dens_sea"], s["dens_R2"], s["f"], ("%.2e" % s["nodes"]) if s["nodes"] is not None else "n/a"))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, default=float)
        print("written %s" % a.out)


if __name__ == "__main__":
    main()
