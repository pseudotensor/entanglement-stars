#!/usr/bin/env python
"""w6_readout_v1_1.py (solve lane, W6 v1.1): w6_readout_v1.py's producer (imported; every version-1 statistic computed by the same function)
plus the cold reading's additions (ReviewA_reading_W6_rungs_v1.md):
  R3: the residual against the row's largest term L = 4 pi N_a R^2 max(|rho_a|, |p_a|, |2 J_a|) at the root of a (median |defect|/L, median
      |rows_mid|/L and |rate|/L, the clock residual over the largest of its three midpoint terms with quartiles and the 90th percentile, its
      ratio to the gate squared per pair, the fraction with |defect| > |rows_mid|, the medians of 2J/rho and p_r/rho at a), outer and inner;
  R1: the four columns with the engine's K_1 weight on the momentum part (from w6_k1weight_v1_1.json), per rung and per phase;
  M1: the root-less gaps per rung (the pair index where T_a of a pair is not T_b of the one before), for the figure's line breaks;
  M2: the median relative representation defect beside the maximum; M4: the inner-root block in full.
Usage: as version 1 plus --k1 W6_K1WEIGHT_JSON; outputs w6_readout_v1_1.json and .npz."""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w6_readout_v1 as V1  # noqa: E402

fl = V1.fl


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def r3(P):
    if not P:
        return {"n_pairs": 0}
    a = lambda k: np.array([fl(p.get(k)) for p in P])
    N, R, rho, pr, J = a("N_a"), a("R_star"), a("rho_a"), a("p_a"), a("J_a")
    L = 4 * np.pi * N * R * R * np.nanmax(np.vstack([np.abs(rho), np.abs(pr), np.abs(2 * J)]), axis=0)
    dfm, rows, rate = a("defect_mid"), a("rows_mid"), a("rate")
    cr = a("clock_residual_mid")
    CL = np.nanmax(np.vstack([np.abs(a("clock_term_st_over_N_mid")), np.abs(a("clock_term_clock_mid")), np.abs(a("clock_term_matter_mid"))]), axis=0)
    crel = np.abs(cr) / CL
    gate = a("gate")
    q = lambda x, p: float(np.nanpercentile(x, p))
    return {"n_pairs": len(P), "largest_term_median": float(np.nanmedian(L)),
            "median_abs_defect_over_largest": float(np.nanmedian(np.abs(dfm) / L)),
            "median_abs_rows_over_largest": float(np.nanmedian(np.abs(rows) / L)), "median_abs_rate_over_largest": float(np.nanmedian(np.abs(rate) / L)),
            "median_abs_defect_over_rows": float(np.nanmedian(np.abs(dfm) / np.abs(rows))),
            "frac_defect_gt_rows": float(np.nanmean(np.abs(dfm) > np.abs(rows))),
            "clock_over_largest_median": float(np.nanmedian(crel)), "clock_over_largest_q25": q(crel, 25), "clock_over_largest_q75": q(crel, 75),
            "clock_over_largest_q90": q(crel, 90), "clock_over_largest_over_gate2_median": float(np.nanmedian(crel / gate ** 2)),
            "median_2J_over_rho": float(np.nanmedian(2 * J / rho)), "median_p_over_rho": float(np.nanmedian(pr / rho)),
            "median_Tll_over_rho": float(np.nanmedian((rho + pr - 2 * J) / rho)), "gate_median": float(np.nanmedian(gate)),
            "gate2_median": float(np.nanmedian(gate ** 2)), "Tll_a_positive": int(np.nansum(a("Tll_a") > 0)),
            "R_min": float(np.nanmin(R)), "R_max": float(np.nanmax(R)), "same_sign_fraction": float(np.nanmean(np.sign(rate) == np.sign(rows)))}


def k1stats(P):
    if not P:
        return {"n_pairs": 0}
    a = lambda k: np.array([fl(p.get(k)) for p in P])
    dM = a("dM")
    out = {"n_pairs": len(P), "sum_dM": float(np.sum(dM))}
    for k in ("col1", "col2_K1", "col3", "col4_K1", "rep_K1"):
        out["sum_" + k] = float(np.sum(a(k))); out["max_abs_" + k] = float(np.max(np.abs(a(k))))
    rd = a("rep_defect_K1")
    out.update({"rep_defect_K1_max": float(np.max(np.abs(rd))), "rep_defect_K1_median": float(np.median(np.abs(rd))),
                "rep_defect_K1_rel_median": float(np.median(np.abs(rd) / np.maximum(np.abs(dM), 1e-300))),
                "M_rep_K1_minus_quad_max": float(np.max(np.abs(np.r_[a("M_rep_K1_minus_quad_a"), a("M_rep_K1_minus_quad_b")])))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", action="append", required=True)
    ap.add_argument("--ladders", required=True)
    ap.add_argument("--k1", required=True)
    ap.add_argument("--phases", default="14.75,17.5")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    # version 1's producer, unchanged, into a scratch file; then extended
    tmp = os.path.join(a.out_dir, "_v1_scratch")
    os.makedirs(tmp, exist_ok=True)
    sys.argv = ["w6_readout_v1.py", "--ladders", a.ladders, "--phases", a.phases, "--out-dir", tmp] + sum([["--rung", r] for r in a.rung], [])
    V1.main()
    D = json.load(open(os.path.join(tmp, "w6_readout_v1.json")))
    Z = dict(np.load(os.path.join(tmp, "w6_readout_v1.npz")))
    K1 = json.load(open(a.k1))
    D["k1_json"] = a.k1; D["k1_md5"] = md5(a.k1)
    ph = D["phases"]; bounds = [-1e9] + ph + [1e9]
    for spec in a.rung:
        lab, v4 = spec.split("=")[0], spec.split("=")[2]
        J = json.load(open(v4))
        P = J["pairs_outer"]; Pi = J.get("pairs_inner") or []
        r = D["rungs"][lab]
        r["outer"]["rep_defect_rel_median"] = float(np.nanmedian(np.abs([fl(p["rep_defect"]) for p in P]) / np.maximum(np.abs([fl(p["dM"]) for p in P]), 1e-300)))
        r["R3_outer"] = r3(P); r["R3_inner"] = r3(Pi)
        Ta = np.array([fl(p["T_a"]) for p in P]); Tb = np.array([fl(p["T_b"]) for p in P])
        gaps = [int(i) for i in range(1, len(P)) if abs(Ta[i] - Tb[i - 1]) > 1e-9]
        r["gaps_after_pair"] = gaps
        r["gaps"] = [[float(Tb[i - 1]), float(Ta[i])] for i in gaps]
        Z[lab + "__gap_breaks"] = np.array(gaps, int)
        if r["kernel"] == "engine" and lab in K1["rungs"]:
            KP = K1["rungs"][lab]["pairs_outer"]; KPi = K1["rungs"][lab]["pairs_inner"]
            assert len(KP) == len(P) and all(abs(fl(x["T_a"]) - fl(p["T_a"])) < 1e-9 for x, p in zip(KP, P))
            r["K1_outer"] = k1stats(KP); r["K1_inner"] = k1stats(KPi)
            r["K1_validation"] = K1["rungs"][lab]["stats_outer"]["validation_max_abs"]
            r["K1_quadrature_check"] = K1["rungs"][lab]["quadrature_check_K0_vs_analytic_max"]
            r["K1_total_weight_at_sites"] = K1["rungs"][lab]["K1_total_weight_at_sites"]
            r["K1_phases"] = []
            for i in range(len(bounds) - 1):
                sel = [x for x in KP if bounds[i] < fl(x["T_b"]) <= bounds[i + 1]]
                r["K1_phases"].append({"T_b_from": bounds[i], "T_b_to": bounds[i + 1], **k1stats(sel)})
            for k in ("col2_K1", "col4_K1", "rep_K1", "rep_defect_K1"):
                Z["%s__%s" % (lab, k)] = np.array([fl(x[k]) for x in KP])
        for k in ("N_a", "rho_a", "p_a", "J_a", "clock_term_st_over_N_mid", "clock_term_clock_mid", "clock_term_matter_mid"):
            Z["%s__%s" % (lab, k)] = np.array([fl(p.get(k)) for p in P])
    D["version"] = "v1.1"
    jp = os.path.join(a.out_dir, "w6_readout_v1_1.json")
    json.dump(D, open(jp, "w"), indent=1)
    npz = os.path.join(a.out_dir, "w6_readout_v1_1.npz")
    np.savez_compressed(npz, **Z)
    with open(os.path.join(a.out_dir, "MANIFEST_w6_readout_v1_1.md5"), "w") as f:
        for p in (jp, npz):
            f.write("%s  %s\n" % (md5(p), os.path.basename(p)))
    for lab in D["order"]:
        r = D["rungs"][lab]
        print(lab, "R3 outer", {k: round(v, 4) if isinstance(v, float) else v for k, v in r["R3_outer"].items()})
        print(lab, "R3 inner", {k: round(v, 4) if isinstance(v, float) else v for k, v in r["R3_inner"].items()})
        if "K1_outer" in r:
            print(lab, "K1", r["K1_outer"])
        print(lab, "gaps", r["gaps"])
    print("wrote", jp)


if __name__ == "__main__":
    main()
