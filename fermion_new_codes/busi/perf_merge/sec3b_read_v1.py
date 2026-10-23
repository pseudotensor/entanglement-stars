#!/usr/bin/env python
"""sec3b_read_v1.py (busi, MERGE-1): the job (b) entry test of ReviewA_v16e_v1 sec. 3(b), read exactly as written. Two records from
job (b)'s own 11.5 slice: X = the merged engine on v16d with --fields collocation and --eigh banded, D = the same without --eigh banded,
P = job (b)'s own records. PASS = converged in all; decisions and masks identical; counts and kept residuals reported; the field-derived
classes by the table where every arm's kept residual is at or below 1e-8, and by the three-way test where it is not.
Inputs: the three run jsons and spectra npz, the 11.5 slice (its masks), and tolcmp_v2's pairwise outputs (X_vs_D, X_vs_P, D_vs_P) for the
class numbers. The identical class is tolcmp_v2's decisions plus the continuation's per-channel labels and counts read here directly
(mode, nsub, verified, inversions, n_intermediate, n_pocket, pockets_occupied_positive, holes_empty_negative, rotated_blocks,
partly_occupied_blocks, mask_equals_negative_set), since tolcmp_v2 books rotated_blocks as a counter (MERGE-1 notes sec. 4.1).
    python sec3b_read_v1.py --x RUN_DIR:TAG --d RUN_DIR:TAG --p RUN_DIR:TAG --slice SLICE --tolv2 DIR --shared tolcmp_v2.py --rs 128 --records 11.5625,11.625 [--out JSON]
Also printed beside it, not part of the verdict: the field-derived classes X-D against the table's lines (the reading the table would give)."""
import argparse
import json
import os

import numpy as np

import importlib.util


def shared(path):   # tolcmp_v2.py's three_way(), the rule cmp_rd2_ruling_v2 shares (ruling v2 2(c)(2): no below-the-line escape)
    sp = importlib.util.spec_from_file_location("tolcmp_v2_shared", path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m

CONT_KEYS = ("mode", "nsub", "verified", "inversions", "n_intermediate", "n_pocket", "pockets_occupied_positive", "holes_empty_negative",
             "rotated_blocks", "partly_occupied_blocks", "mask_equals_negative_set")
TABLE = {"fields": 1e-7, "rows": 1e-6, "spectra_rel": 1e-8}
KEPT = {"E_over_E0": 3.0e-8, "roots": 3.4e-7, "depth": 4.6e-9}
C = {"E_over_E0": 3.0e-7, "roots": 3.4e-6, "depth": 4.6e-8, "fields": 2.5e-5}
FIELD_KEYS = ("fields/N", "fields/v", "fields/N_applied", "fields/v_applied", "fields/N_nodes", "fields/v_nodes")


def load(spec, rs):
    d, tag = spec.split(":")
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_spectra.npz" % tag))
    return {round(r["T"] / rs, 6): r for r in J["records"]}, Z


def mask_row(Z, t, rs, k):
    i = np.where(np.round(np.asarray(Z["T"], float) / rs, 6) == t)[0]
    return Z["mask_%d" % k][i[0]] if len(i) == 1 else None


def cont(r):
    c = r.get("continuation") or {}
    out = {}
    for ch, e in (c.items() if isinstance(c, dict) else []):
        if isinstance(e, dict):
            out[ch] = {k: e.get(k) for k in CONT_KEYS}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True); ap.add_argument("--d", required=True); ap.add_argument("--p", required=True)
    ap.add_argument("--slice", required=True); ap.add_argument("--tolv2", required=True); ap.add_argument("--rs", type=float, required=True)
    ap.add_argument("--records", required=True); ap.add_argument("--out", default=None)
    ap.add_argument("--shared", required=True, help="tolcmp_v2.py or tolcmp_v3.py (its three_way())")
    ap.add_argument("--prefix", default="v2", help="the pairwise files tolcmp wrote: v2_X_vs_D.json (tolcmp_v2) or v3_... (tolcmp_v3)")
    a = ap.parse_args()
    TW = shared(a.shared).three_way
    ts = [round(float(x), 6) for x in a.records.split(",")]
    R, Z = {}, {}
    for nm, spec in (("X", a.x), ("D", a.d), ("P", a.p)):
        R[nm], Z[nm] = load(spec, a.rs)
    pw = {k: json.load(open(os.path.join(a.tolv2, "%s_%s.json" % (a.prefix, k)))) for k in ("X_vs_D", "X_vs_P", "D_vs_P")}
    S = np.load(a.slice)
    kap = [int(k) for k in S["kappas"]]
    nch = len(kap)
    res = {"records": {}, "verdict": None}
    allok = True
    for t in ts:
        key = "%.5f" % t
        e = {"arms": {}}
        ok = True
        for nm in ("X", "D", "P"):
            r = R[nm].get(t)
            if r is None:
                e["arms"][nm] = "absent"; ok = False; continue
            s = r.get("solve_v32") or {}
            e["arms"][nm] = {"converged": r.get("converged"), "iterations": r.get("iterations"), "evaluations": s.get("evaluations"),
                             "eighs": (r.get("timing") or {}).get("eigh_n"), "kept_residual": s.get("kept_residual"),
                             "confirming_residual": s.get("confirming_residual"), "E_over_E0": r.get("E_over_E0")}
            ok &= r.get("converged") is True
        e["converged_all"] = ok
        # decisions: tolcmp_v2's identical class, pairwise
        dec = {k: pw[k]["per_record"].get(key, {}).get("n_decisions") for k in pw}
        e["decisions_tolcmp_v2"] = dec
        dok = all(v == 0 for v in dec.values())
        # the continuation's labels, read directly
        cx, cd, cp = (cont(R[nm][t]) for nm in ("X", "D", "P"))
        cdiff = [(nm, ch) for nm, cc in (("X-D", (cx, cd)), ("X-P", (cx, cp)), ("D-P", (cd, cp))) for ch in set(cc[0]) | set(cc[1]) if cc[0].get(ch) != cc[1].get(ch)]
        e["continuation_label_differences"] = cdiff
        dok &= not cdiff
        # masks: the spectra rows, pairwise, and against the 11.5 slice's masks
        mdiff, m115 = [], {}
        for k in range(1, nch + 1):
            mx, md, mp = (mask_row(Z[nm], t, a.rs, k) for nm in ("X", "D", "P"))
            for nm, (u, v) in (("X-D", (mx, md)), ("X-P", (mx, mp)), ("D-P", (md, mp))):
                if u is None or v is None or not np.array_equal(u, v):
                    mdiff.append((nm, k))
            m115[k] = bool(mx is not None and np.array_equal(mx, S["mask_%d" % k]))
        e["mask_differences"] = mdiff
        e["masks_equal_the_11.5_slice"] = m115
        dok &= not mdiff
        e["decisions_and_masks_identical"] = dok
        ok &= dok
        # kept lines, X-D, unconditionally
        xd = pw["X_vs_D"]["per_record"].get(key, {})
        kept = {c: (xd.get(c), KEPT[c], (xd.get(c) or 0.0) <= KEPT[c]) for c in KEPT}
        kept["R_sh"] = (xd.get("R_sh"), 0.0, (xd.get("R_sh") or 0.0) == 0.0)
        e["kept_lines_X_D"] = kept
        ok &= all(v[2] for v in kept.values())
        # the field-derived classes
        kr = [e["arms"][nm]["kept_residual"] for nm in ("X", "D", "P") if isinstance(e["arms"][nm], dict)]
        table_mode = all(x is not None and x <= 1e-8 for x in kr)
        e["field_classes_by"] = "table (every kept residual <= 1e-8)" if table_mode else "three-way test (a kept residual > 1e-8: %s)" % ", ".join("%.2e" % x for x in kr)

        def fmax(p):
            return max([p["rows"][k]["per_record_abs"].get(key, 0.0) for k in FIELD_KEYS if k in p["rows"]] or [0.0])

        def rowmax(p):
            return max([v["per_record_abs"].get(key, 0.0) for k, v in p["rows"].items() if k.startswith("handoff/") or k.split("/")[1].startswith(("eps_", "J_"))] or [0.0])

        def specmax(p):
            return max([v["per_record_rel"].get(key, 0.0) for k, v in p["rows"].items() if k.startswith("spectra/e_")] or [0.0])

        def named(p):
            return p["per_record"].get(key, {}).get("named_rel", 0.0)
        cl = {nm: {"fields": fmax(p), "rows": rowmax(p), "spectra_rel": specmax(p), "named_rel": named(p),
                   "E_over_E0": p["per_record"].get(key, {}).get("E_over_E0"), "roots": p["per_record"].get(key, {}).get("roots"),
                   "depth": p["per_record"].get(key, {}).get("depth")} for nm, p in pw.items()}
        e["classes"] = cl
        if table_mode:
            fl = {c: (cl["X_vs_D"][c], TABLE[c], cl["X_vs_D"][c] <= TABLE[c]) for c in TABLE}
            e["field_lines"] = fl
            ok &= all(v[2] for v in fl.values())
        else:
            one = {nm: {c: (cl[nm][c], C[c], (cl[nm][c] or 0.0) <= C[c]) for c in C} for nm in ("X_vs_P", "D_vs_P")}
            two = {}
            for c in ("fields", "rows", "spectra_rel", "named_rel"):
                ok_, ratio_ = TW(cl["X_vs_P"][c] or 0.0, cl["D_vs_P"][c] or 0.0, None)
                two[c] = (cl["X_vs_P"][c], cl["D_vs_P"][c], ok_, ratio_)
            e["three_way"] = {"(1) within C of P": one, "(2) X-P at most twice D-P (tolcmp_v2.three_way, no below-the-line escape)": two,
                              "(3) per call": "RD-1's banded eigh against dense on the ISO_128 matrices, 2n 8192: eigenvalues 7.1e-15, residuals 2.1e-14 ||h||, tracked sets equal (RD-1 notes 3.2); the change in X is --eigh banded only"}
            ok &= all(v[2] for d_ in one.values() for v in d_.values()) and all(v[2] for v in two.values())
            e["beside_not_gating_table_lines_X_D"] = {c: (cl["X_vs_D"][c], TABLE[c], cl["X_vs_D"][c] <= TABLE[c]) for c in TABLE}
        e["record_PASS"] = ok
        allok &= ok
        res["records"][key] = e
    res["verdict"] = "PASS" if allok else "FAIL"
    print(json.dumps(res, indent=1, default=str))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
