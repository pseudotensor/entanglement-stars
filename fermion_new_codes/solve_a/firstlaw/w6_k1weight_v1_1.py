#!/usr/bin/env python
"""w6_k1weight_v1_1.py (solve lane, W6 v1.1; the cold reading ReviewA_reading_W6_rungs_v1.md R1): the fixed-radius identity's four columns
with the momentum part of the site charge weighted by the engine's own l = 1 kernel K_1 instead of the scalar kernel's weight.

busi's firstlaw_records_v4b.py (unchanged, imported read-only) weights the whole site charge Q_n = eps_n + u(R_n) J_n by
w_n(R) = int_0^R 4 pi s^2 K_0,n(s) ds (cumulative_weight, analytic). The engine smooths eps with K_0 and J with K_1 (engine_fields_v4.kernels;
SmoothedSources5). This script rebuilds, per rung, v4b's engine-route store (its loaders, its RecordStore, its engine kernel matrices K, KJ),
and for every outer and inner pair of the rung's v4b engine-route json recomputes
  w^J_n(R) = int_0^R 4 pi s^2 KJ_n(s) ds   (cumulative Simpson on the dense grid to the last node at or below R, plus the partial cell by the
                                            trapezoid on the linearly interpolated integrand)
  col1 = chi . dQ_state (unchanged), col2 = (w - chi) . dE_state + (w^J - chi) . dP_state, col3 = -(C_u,b - C_u,a) (unchanged, v4b already
  builds C_u with KJ on the engine route), col4 = -(w . dE_ref + w^J . dP_ref), rep = their sum, rep_defect = dM - rep,
  M_rep(x) = w . eps + w^J . (u J) - C_u against v4b's M_quad (the dense quadrature of 4 pi r^2 (rho + u J)),
where E = eps and P = u(R_n) J_n per site (state and reference parts as v4b splits them). Validation printed per rung: (i) with w^J := w the
script reproduces v4b's col2, col4, rep and M_rep to roundoff; (ii) the same cumulative quadrature applied to the engine's K reproduces v4b's
analytic w (the quadrature's own error, against which the K_1 closure is read).
Usage (box CPU, nice 10): w6_k1weight_v1_1.py --rung LABEL=V4B_ENGINE_JSON ... --out-dir DIR"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "<internal>/w6w7")
import firstlaw_records_v4b as V4  # noqa: E402
from scipy.integrate import cumulative_simpson  # noqa: E402


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def cumulative_weights(r, Kmat):
    """W[n, i] = int_0^{r_i} 4 pi s^2 K_n(s) ds (cumulative Simpson from r_0; r_0 is the grid's first node)."""
    g = 4.0 * np.pi * r * r * Kmat
    W = np.zeros_like(g)
    W[:, 1:] = cumulative_simpson(g, x=r, axis=1)
    return W, g


def weight_at(r, W, g, R):
    i0 = int(np.searchsorted(r, R, side="right")) - 1
    i0 = max(0, min(r.size - 2, i0))
    t = (R - r[i0]) / (r[i0 + 1] - r[i0])
    g_R = (1 - t) * g[:, i0] + t * g[:, i0 + 1]
    return W[:, i0] + 0.5 * (g[:, i0] + g_R) * (R - r[i0])


def corrected(pair, A, B, store, WJ, gJ):
    R = float(pair["R_star"])
    w = V4.cumulative_weight(store.ell, store.R_sites, R)
    wJ = weight_at(store.r, WJ, gJ, R)
    chi = (store.R_sites <= R).astype(float)
    uA, uB = A["u_sites"], B["u_sites"]
    dE_s = B["eps_state"] - A["eps_state"]; dP_s = uB * B["J_state"] - uA * A["J_state"]
    dE_r = B["eps_ref"] - A["eps_ref"]; dP_r = uB * B["J_ref"] - uA * A["J_ref"]
    Cua, Cub = store.Cu(A, R), store.Cu(B, R)
    col1 = float(chi @ (dE_s + dP_s))
    col3 = -(Cub - Cua)

    def cols(wm):
        c2 = float((w - chi) @ dE_s + (wm - chi) @ dP_s)
        c4 = -float(w @ dE_r + wm @ dP_r)
        return c2, c4, col1 + c2 + col3 + c4
    c2_old, c4_old, rep_old = cols(w)          # validation: v4b's own columns
    c2, c4, rep = cols(wJ)
    out = {"T_a": pair["T_a"], "T_b": pair["T_b"], "R_star": R, "dM": pair["dM"], "col1": col1, "col2_K1": c2, "col3": col3, "col4_K1": c4, "rep_K1": rep,
           "rep_defect_K1": pair["dM"] - rep, "rep_defect_v4b": pair["rep_defect"],
           "check_col2_v4b": c2_old - pair["col2_cut_assignment"], "check_col4_v4b": c4_old - pair["col4_continued_reference"], "check_rep_v4b": rep_old - pair["rep"],
           "wJ_minus_w_inside_sum": float(np.sum(wJ - w))}
    for tag, X, Cu in (("a", A, Cua), ("b", B, Cub)):
        Mrep_old = float(w @ X["Q"]) - Cu
        Mrep = float(w @ X["eps"] + wJ @ (X["u_sites"] * X["J_site"])) - Cu
        Mq = store.Mquad(X, R)
        out["M_rep_K1_minus_quad_" + tag] = Mrep - Mq
        out["M_rep_v4b_minus_quad_" + tag] = Mrep_old - Mq
        out["check_Mrep_v4b_" + tag] = Mrep_old - pair["M_rep_" + tag]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", action="append", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    res = {"script": os.path.abspath(__file__), "v4b_module": V4.__file__, "v4b_module_md5": md5(V4.__file__), "rungs": {}}
    for spec in a.rung:
        lab, v4 = spec.split("=", 1)
        t0 = time.time()
        J4 = json.load(open(v4)); meta = J4["meta"]; opt = meta["options"]
        res3 = json.load(open(meta["v3b_json"]))
        cache = V4.load_cache(meta["v3b_cache"])
        F = V4.load_fields(meta["fields_npz"])
        HO = V4.load_handoff(meta["handoff_npz"], F["R"].size)
        r = cache["r_fine"]
        sa = (res3.get("provenance") or {}).get("solver_args") or {}
        support_rel = V4._float((sa.get("support_rel") or {}).get("value")) if isinstance(sa.get("support_rel"), dict) else 1e-14
        if not np.isfinite(support_rel):
            support_rel = 1e-14
        eprov = (res3.get("provenance") or {}).get("engine") or {}
        K, KJ, prov = V4.kernel_matrix_engine(opt["v3b_dir"], opt["engine_path"] or eprov.get("path"), opt["engine_module"] or eprov.get("module") or "engine_fields_v16",
                                              F["R"], meta["ell"], float(res3["grid"]["r_fine_last"]), support_rel, r)
        store = V4.RecordStore(cache, F, HO, K, KJ, meta["G"], meta["ell"], meta["degree"], meta["quad_degree"], opt["pr_scale"])
        WJ, gJ = cumulative_weights(r, KJ)
        W0, g0 = cumulative_weights(r, K)
        recs_by_T = {round(float(x["T_over_rs"]), 9): x for x in res3["records"]}
        out = {"v4b_json": v4, "v4b_md5": md5(v4), "ell": meta["ell"], "n_sites": int(F["R"].size), "dense_h": float(r[1] - r[0]), "pairs_outer": [], "pairs_inner": []}
        # the quadrature's own error: the same cumulative rule on K against the analytic w at a few radii
        qerr = []
        for Rt in (10.0, 20.0, 35.0, 50.0, 80.0):
            qerr.append(float(np.max(np.abs(weight_at(r, W0, g0, Rt) - V4.cumulative_weight(meta["ell"], F["R"], Rt)))))
        out["quadrature_check_K0_vs_analytic_max"] = max(qerr)
        tot = WJ[:, -1]
        out["K1_total_weight_at_sites"] = {"%g" % F["R"][n]: float(tot[n]) for n in (0, 1, 2, 5, 10, 20, 40, 100) if n < F["R"].size}
        for which in ("pairs_outer", "pairs_inner"):
            for p in J4.get(which) or []:
                ra = recs_by_T.get(round(float(p["T_a"]), 9)); rb = recs_by_T.get(round(float(p["T_b"]), 9))
                if ra is None or rb is None:
                    continue
                A = store.get(int(ra["row"]), float(ra["T_over_rs"])); B = store.get(int(rb["row"]), float(rb["T_over_rs"]))
                if int(ra["row"]) != int(p["row_a"]) or int(rb["row"]) != int(p["row_b"]):
                    raise SystemExit("row mismatch at %s" % p["T_a"])
                out[which].append(corrected(p, A, B, store, WJ, gJ))
        P = out["pairs_outer"]
        f = lambda k: np.array([q[k] for q in P])
        out["stats_outer"] = {"n": len(P),
                              "validation_max_abs": {k: float(np.max(np.abs(f(k)))) for k in ("check_col2_v4b", "check_col4_v4b", "check_rep_v4b", "check_Mrep_v4b_a", "check_Mrep_v4b_b")},
                              "rep_defect_v4b_max": float(np.max(np.abs(f("rep_defect_v4b")))), "rep_defect_v4b_median": float(np.median(np.abs(f("rep_defect_v4b")))),
                              "rep_defect_K1_max": float(np.max(np.abs(f("rep_defect_K1")))), "rep_defect_K1_median": float(np.median(np.abs(f("rep_defect_K1")))),
                              "M_rep_v4b_minus_quad_max": float(np.max(np.abs(np.r_[f("M_rep_v4b_minus_quad_a"), f("M_rep_v4b_minus_quad_b")]))),
                              "M_rep_K1_minus_quad_max": float(np.max(np.abs(np.r_[f("M_rep_K1_minus_quad_a"), f("M_rep_K1_minus_quad_b")]))),
                              "sum_dM": float(np.sum(f("dM"))), "sum_col1": float(np.sum(f("col1"))), "sum_col2_K1": float(np.sum(f("col2_K1"))),
                              "sum_col3": float(np.sum(f("col3"))), "sum_col4_K1": float(np.sum(f("col4_K1"))), "sum_rep_K1": float(np.sum(f("rep_K1"))),
                              "max_abs_col2_K1": float(np.max(np.abs(f("col2_K1")))), "max_abs_col4_K1": float(np.max(np.abs(f("col4_K1"))))}
        Pi = out["pairs_inner"]
        if Pi:
            g = lambda k: np.array([q[k] for q in Pi])
            out["stats_inner"] = {"n": len(Pi), "rep_defect_K1_max": float(np.max(np.abs(g("rep_defect_K1")))), "rep_defect_K1_median": float(np.median(np.abs(g("rep_defect_K1")))),
                                  "rep_defect_v4b_max": float(np.max(np.abs(g("rep_defect_v4b")))),
                                  "M_rep_K1_minus_quad_max": float(np.max(np.abs(np.r_[g("M_rep_K1_minus_quad_a"), g("M_rep_K1_minus_quad_b")])))}
        out["seconds"] = time.time() - t0
        res["rungs"][lab] = out
        print(lab, json.dumps(out["stats_outer"]), "quadrature check", out["quadrature_check_K0_vs_analytic_max"], "K1 totals", out["K1_total_weight_at_sites"], "%.0f s" % out["seconds"], flush=True)
    jp = os.path.join(a.out_dir, "w6_k1weight_v1_1.json")
    json.dump(res, open(jp, "w"), indent=1)
    print("wrote", jp, md5(jp))


if __name__ == "__main__":
    main()
