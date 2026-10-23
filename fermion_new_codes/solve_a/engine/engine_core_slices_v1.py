#!/usr/bin/env python
"""engine_core_slices_v1.py (builder analyst; engine spec 28.5, S8 of REVIEWA_REVIEW_web40e: "S8 (engine_core_slices_v1; CPU, seconds): per record
and core shell H ell (max |d_r v| ell over the core), min N, the two sides of the band-kinematic law apart (rho_dot; 3 H (rho + p) with p =
(s_r + 2 s_t)/3, H = -d_r v) and at the stored slices the defect -2 H int f eps sum_i (1 - cos 2 a k_i) from the dust's columns, the packing
ratio ell/R_pack; the hydrodynamic claim stops where H ell exceeds 0.1; no saturated core is imposed"). This version: per record of a
COLLAPSE_records data npz with the run's json (the record's core radius R_c): the dense fields re-solved from the stored shell totals by the
run's field module per segment (engine_certificate_v2's route), the core = the coarse nodes r_i = i ell with r_i <= R_c; per record the core's
largest H ell with H = -d_r v (and its node), the core's smallest N, at the innermost core node and the core's median node rho_dot (centred on
the neighbouring records, the solve's smoothed rho) and 3 H (rho + p) with p = (s_r + 2 s_t)/3 from the smoothed stresses (dust plus pairs),
their difference over the larger; the first record where H ell exceeds --stop (0.1) is printed: the hydrodynamic claim stops there. Not in
this version (labelled pending): the band defect from the dust's columns and the packing ratio (they need the stored slices' orbitals:
engine_core_slices_v2). Every number printed by the script. CPU. Args: --data --json --fm-segments --fm-default --stop --md --out-json."""
import argparse
import importlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_rows_v2 as RW  # noqa: E402
import engine_stress_v2 as STR2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--json", required=True); ap.add_argument("--md", required=True); ap.add_argument("--out-json", required=True)
    ap.add_argument("--fm-segments", default="v17 (replayed)=engine_fields_v12;v19 (native)=engine_fields_v14"); ap.add_argument("--fm-default", default="engine_fields_v12")
    ap.add_argument("--stop", type=float, default=0.1); ap.add_argument("--hfrac", type=float, default=32.0)
    a = ap.parse_args()
    Z = np.load(a.data, allow_pickle=True)
    R = np.asarray(Z["R"], float); T = np.asarray(Z["T"], float); g = float(Z["g"]); ell = float(Z["ell"]); rs = float(Z["rs"])
    seg_names = [str(x) for x in Z["segment_names"]] if "segment_names" in Z.files else None
    seg = np.asarray(Z["segment"], int) if "segment" in Z.files else np.zeros(len(T), int)
    fmap = dict(x.split("=", 1) for x in a.fm_segments.split(";"))
    recs = {}
    for f in a.json.split(","):
        for r in json.load(open(f))["records"]:
            if r.get("E") is not None:
                recs.setdefault(round(float(r["T"]), 6), r)
    fms = {}; cache = {}

    def fields(k):
        if k in cache:
            return cache[k]
        name = fmap.get(seg_names[seg[k]], a.fm_default) if seg_names else a.fm_default
        if name not in fms:
            fms[name] = importlib.import_module(name)
        _, _, _, info = fms[name].solve_constraints_ivp(np.asarray(Z["eps_tot"][k], float), np.asarray(Z["J_tot"][k], float), g, R, ell, collar_policy="flag", support_rel=1e-14)
        Sr = np.asarray(Z["S_r_dust"][k], float) + np.asarray(Z["S_r_pairs"][k], float); St = np.asarray(Z["S_t_dust"][k], float) + np.asarray(Z["S_t_pairs"][k], float)
        cache[k] = (info, Sr, St)
        for old in [x for x in cache if x < k - 2]:
            del cache[old]
        return cache[k]

    rows = []; stop_at = None
    for k in range(1, len(T) - 1):
        rec = recs.get(round(T[k], 6), {})
        Rc = rec.get("R_c")
        if Rc is None:
            continue
        info, Sr, St = fields(k)
        if info.get("failed"):
            rows.append({"T_over_rs": T[k] / rs, "failed": True}); continue
        nodes = np.arange(ell, float(Rc) + 1e-9, ell)
        if len(nodes) == 0:
            continue
        N, _, _ = RW.deriv(info["N_dense"], nodes, ell / a.hfrac)
        v, v1, _ = RW.deriv(info["v_dense"], nodes, ell / a.hfrac)
        H = -v1
        Hl = np.abs(H) * ell
        im = int(np.argmax(Hl))
        im_, _, _ = fields(k - 1); ip_, _, _ = fields(k + 1)
        rho = info["rho_ell"](nodes)
        rho_dot = (ip_["rho_ell"](nodes) - im_["rho_ell"](nodes)) / (T[k + 1] - T[k - 1])
        sr, st, _ = STR2.smooth_stress(Sr, St, R, ell, nodes)
        p = (sr + 2.0 * st) / 3.0
        rhs = 3.0 * H * (rho + p)
        pick = [0, len(nodes) // 2]
        side = [(float(nodes[j]), float(rho_dot[j]), float(rhs[j]), float(abs(rho_dot[j] - rhs[j]) / max(abs(rho_dot[j]), abs(rhs[j]), 1e-300))) for j in pick]
        row = {"T_over_rs": T[k] / rs, "R_c": float(Rc), "core_nodes": len(nodes), "H_ell_max": float(Hl[im]), "H_ell_at": float(nodes[im]), "N_min_core": float(np.min(N)),
               "inner": side[0], "median": side[1]}
        rows.append(row)
        if stop_at is None and row["H_ell_max"] > a.stop:
            stop_at = row["T_over_rs"]
    L = ["## The core slices (engine_core_slices_v1.py; engine spec 28.5, S8): %s" % os.path.basename(a.data), "",
         "- the core: the coarse nodes up to the record's stored core radius R_c; H = -d_r v of the dense solve; rho the solve's smoothed density, rho_dot centred on the neighbouring records; p = (s_r + 2 s_t)/3 from the smoothed stresses (dust plus pairs).",
         "- the hydrodynamic claim stops where H ell exceeds %g: the first such record %s." % (a.stop, ("T/rs %.3f" % stop_at) if stop_at is not None else "none"),
         "- not in this version (pending, engine_core_slices_v2 on the stored slices' orbitals): the band defect -2 H int f eps sum_i (1 - cos 2 a k_i) and the packing ratio ell/R_pack.", "",
         "| T/rs | R_c | core nodes | max H ell (at r) | min N (core) | innermost node: r, rho_dot, 3H(rho+p), relative difference | median node: r, rho_dot, 3H(rho+p), relative difference |",
         "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows):
        if r.get("failed"):
            continue
        if i % 4 == 0 or r["H_ell_max"] > a.stop and (i == 0 or rows[i - 1].get("H_ell_max", 0) <= a.stop):
            L.append("| %.3f | %.2f | %d | %.4f (%.0f) | %.5f | %.0f, %+.3e, %+.3e, %.2e | %.0f, %+.3e, %+.3e, %.2e |" % (
                r["T_over_rs"], r["R_c"], r["core_nodes"], r["H_ell_max"], r["H_ell_at"], r["N_min_core"], *r["inner"], *r["median"]))
    L += ["", "- rows every fourth record (and the first above the stop); every number above is printed by the script; the full list in the json."]
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump({"script": "engine_core_slices_v1.py", "data": a.data, "stop": a.stop, "first_stop": stop_at, "rows": rows}, open(a.out_json, "w"), indent=1, default=float)
    print("\n".join(L[:8]))


if __name__ == "__main__":
    main()
