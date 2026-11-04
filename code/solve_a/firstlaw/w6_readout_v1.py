#!/usr/bin/env python
"""w6_readout_v1.py (composition builder, the solve lane): the producer of W6_FIRSTLAW_rungs_v1 (W6 of final4/WEB40I_TASKS/busi.md,
FIRSTLAW_records v2.3 on the averaging ladder's rungs at (0.2, 64)). Reads only the busi lane's firstlaw_records_v4b outputs (json) of
each rung and its v3b json, and the ladder readout's certification (ladders_readout_v1_1.json), and saves:
  w6_readout_v1.json: per rung the identity's closure (max |sum4 - rep|), the representation defect (max |dM - rep| and relative), the
    kernel checks, the column sums and medians, the rows' rate against the records' rate (sums, median |defect|/|rows_mid|, the same-sign
    fraction, the signs), the full-clock residual (median, |.| median and max, midpoint and endpoint, its 3 terms), the gate per root
    (outer and inner, range and counts inside 0.1), per phase (the T_b boundaries given), the seam pairs, and the records' certification;
  w6_readout_v1.npz: per rung per pair T_a, T_b, R*, dM, rate, the 4 columns, rep, rows_mid, defect_mid, clock_residual_mid, gate,
    gate_inner.
Usage (box CPU): python w6_readout_v1.py --rung LABEL=SIGMA=V4B_JSON[=V3B_JSON] ... --ladders JSON --phases 14.75,17.5 --seam 18.0 --out-dir DIR"""
import argparse
import hashlib
import json
import os

import numpy as np


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def fl(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def stats(P, phase=None):
    if not P:
        return {"n_pairs": 0}
    a = lambda k: np.array([fl(p.get(k)) for p in P])
    dM, rate, rows, dfm = a("dM"), a("rate"), a("rows_mid"), a("defect_mid")
    cr = a("clock_residual_mid"); cra = a("clock_residual_a")
    out = {"n_pairs": len(P), "T_first": float(np.nanmin(a("T_a"))), "T_last": float(np.nanmax(a("T_b"))),
           "closure_max": float(np.nanmax(np.abs(a("closure")))), "rep_defect_max": float(np.nanmax(np.abs(a("rep_defect")))),
           "rep_defect_rel_max": float(np.nanmax(np.abs(a("rep_defect")) / np.maximum(np.abs(dM), 1e-300))),
           "rep_defect_median": float(np.nanmedian(np.abs(a("rep_defect")))),
           "M_rep_minus_quad_max": float(np.nanmax(np.abs(np.concatenate([a("M_rep_minus_quad_a"), a("M_rep_minus_quad_b")])))),
           "M_quad_minus_ode_max": float(np.nanmax(np.abs(np.concatenate([a("M_quad_minus_ode_a"), a("M_quad_minus_ode_b")])))),
           "sum_dM": float(np.nansum(dM))}
    for k in ("col1_flux_old_cut", "col2_cut_assignment", "col3_averaging_commutator", "col4_continued_reference", "col1_flow_change_part", "rep"):
        v = a(k)
        out["sum_" + k] = float(np.nansum(v)); out["median_" + k] = float(np.nanmedian(v)); out["max_abs_" + k] = float(np.nanmax(np.abs(v)))
    out.update({"sum_rate": float(np.nansum(rate)), "sum_rows_mid": float(np.nansum(rows)), "sum_defect_mid": float(np.nansum(dfm)),
                "median_abs_defect_over_rows": float(np.nanmedian(np.abs(dfm) / np.maximum(np.abs(rows), 1e-300))),
                "same_sign_fraction": float(np.nanmean(np.sign(rate) == np.sign(rows))),
                "rows_negative": int(np.nansum(rows < 0)), "rate_negative": int(np.nansum(rate < 0)), "Tll_a_negative": int(np.nansum(a("Tll_a") < 0)),
                "sum_rows_a_stress": float(np.nansum(a("rows_a_stress"))), "sum_rows_a_nostress": float(np.nansum(a("rows_a_nostress"))),
                "clock_mid_median": float(np.nanmedian(cr)), "clock_mid_abs_median": float(np.nanmedian(np.abs(cr))), "clock_mid_abs_max": float(np.nanmax(np.abs(cr))),
                "clock_a_median": float(np.nanmedian(cra)), "clock_a_abs_max": float(np.nanmax(np.abs(cra))),
                "clock_term_st_over_N_median": float(np.nanmedian(a("clock_term_st_over_N_mid"))),
                "clock_term_clock_median": float(np.nanmedian(a("clock_term_clock_mid"))), "clock_term_matter_median": float(np.nanmedian(a("clock_term_matter_mid"))),
                "exact_relation_max": float(np.nanmax(np.abs(a("dM_from_s_minus_dM")))),
                "gate_min": float(np.nanmin(a("gate"))), "gate_max": float(np.nanmax(a("gate"))), "gate_median": float(np.nanmedian(a("gate"))),
                "gate_inside": int(np.nansum(a("gate") <= 0.1)), "R_min": float(np.nanmin(a("R_star"))), "R_max": float(np.nanmax(a("R_star")))})
    if "gate_inner" in P[0]:
        gi = a("gate_inner")
        out.update({"gate_inner_min": float(np.nanmin(gi)) if np.any(np.isfinite(gi)) else None,
                    "gate_inner_max": float(np.nanmax(gi)) if np.any(np.isfinite(gi)) else None,
                    "gate_inner_inside": int(np.nansum(gi <= 0.1)), "gate_inner_n": int(np.sum(np.isfinite(gi)))})
    kr, kj = a("kernel_rel_rho_a"), a("kernel_rel_J_a")
    if np.any(np.isfinite(kr)):
        out.update({"kernel_rel_rho_max": float(np.nanmax(kr)), "kernel_rel_J_max": float(np.nanmax(kj)), "kernel_rel_J_median": float(np.nanmedian(kj))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", action="append", required=True)
    ap.add_argument("--ladders", required=True)
    ap.add_argument("--phases", default="14.75,17.5")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    LJ = json.load(open(a.ladders))
    ph = [float(x) for x in a.phases.split(",") if x]
    out, arrays = {"rungs": {}, "order": [], "phases": ph}, {}
    for spec in a.rung:
        parts = spec.split("=")
        lab, sig, v4 = parts[0], float(parts[1]), parts[2]
        v3 = parts[3] if len(parts) > 3 else None
        J = json.load(open(v4))
        P = J["pairs_outer"]; Pi = J.get("pairs_inner") or []
        meta = J["meta"]
        r = {"label": lab, "sigma": sig, "v4b_json": v4, "v4b_md5": md5(v4), "v3b_json": v3 or meta.get("v3b_json"),
             "records_json": meta.get("records_json"), "kernel": meta.get("kernel"), "seconds": J.get("seconds"),
             "outer": stats(P), "inner": stats(Pi)}
        if v3 or meta.get("v3b_json"):
            r["v3b_md5"] = md5(v3 or meta["v3b_json"])
        # phases by T_b
        bounds = [-1e9] + ph + [1e9]
        r["phases"] = []
        for i in range(len(bounds) - 1):
            sel = [p for p in P if bounds[i] < fl(p["T_b"]) <= bounds[i + 1]]
            r["phases"].append({"T_b_from": bounds[i], "T_b_to": bounds[i + 1], **stats(sel)})
        # the records' certification from the ladder readout (by the records json path)
        cert = None
        for nm, M in LJ["members"].items():
            if os.path.realpath(M["meta"]["read_from"]) == os.path.realpath(meta.get("records_json") or ""):
                c = M["convergence"]
                cert = {"member": nm, "last_converged": c["last_converged"], "first_noncounting": c["first_noncounting"],
                        "first_noncounting_gate": c["first_noncounting_gate"], "records": c["records"], "conf_fail_count": c.get("conf_fail_count"),
                        "formation": (M["formation"] or {}).get("T"), "formation_class": (M["formation"] or {}).get("class")}
        r["certification"] = cert
        if cert and cert["last_converged"] is not None:
            r["pairs_beyond_certified"] = int(sum(1 for p in P if fl(p["T_b"]) > cert["last_converged"] + 1e-9))
        elif cert:
            r["pairs_beyond_certified"] = len(P)
        out["rungs"][lab] = r
        out["order"].append(lab)
        for k in ("T_a", "T_b", "R_star", "dM", "rate", "col1_flux_old_cut", "col2_cut_assignment", "col3_averaging_commutator",
                  "col4_continued_reference", "rep", "rep_defect", "rows_mid", "defect_mid", "clock_residual_mid", "gate", "gate_inner", "Tll_a"):
            arrays["%s__%s" % (lab, k)] = np.array([fl(p.get(k)) for p in P])
        print(lab, r["outer"]["n_pairs"], r["outer"].get("closure_max"), r["outer"].get("rep_defect_max"), flush=True)
    jp = os.path.join(a.out_dir, "w6_readout_v1.json")
    json.dump(out, open(jp, "w"), indent=1)
    npz = os.path.join(a.out_dir, "w6_readout_v1.npz")
    np.savez_compressed(npz, **arrays)
    with open(os.path.join(a.out_dir, "MANIFEST_w6_readout_v1.md5"), "w") as f:
        for p in (jp, npz):
            f.write("%s  %s\n" % (md5(p), os.path.basename(p)))
    print("wrote", jp)


if __name__ == "__main__":
    main()
