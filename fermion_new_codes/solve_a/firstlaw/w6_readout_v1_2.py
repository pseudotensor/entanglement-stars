#!/usr/bin/env python
"""w6_readout_v1_2.py (solve lane, W6_FIRSTLAW_rungs_v1.2): the readout of busi's v4c rerun (firstlaw_records_v4c.py, the K_1 momentum weight;
ready_jobs_v3.md sec. 28) beside the v4b jsons W6 v1 and v1.1 read, and beside v1.1's own K_1-weight reading (w6_k1weight_v1_1.json).
Data only: writes w6_readout_v1_2.json (statistics, checks) and w6_readout_v1_2.npz (per-pair arrays for the figure). No solver is run.
Per member and route: the meta's momentum weight; the closure (max and median |rep defect| per pair, outer and inner pairs and the offsets);
the sums over the outer pairs of Delta M, col1..col4, rep, col2's and col4's momentum parts, v4b beside v4c; an independent key-by-key
comparison of v4c's per-pair keys against the v4b json (bitwise or max difference), beside busi's compare log; the phases (T_b windows
14.75, 17.5) with the K_1 weight; on the engine route, v4c's col2, rep and rep defect against v1.1's own K_1 reading per pair.
Usage (box CPU, nice 10): w6_readout_v1_2.py --v4c-dir <internal>/v4c --v4b-root <internal>/firstlaw
       --k1 <internal>/w6_k1weight_v1_1.json --out-dir DIR"""
import argparse
import hashlib
import json
import os

import numpy as np

TAGS = {"s2": ("c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d", "sigma2_point", 2.0),
        "s4": ("c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40", "sigma4_rung", 4.0),
        "s8": ("c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_to40", "sigma8_rung", 8.0)}
COLS = ["dM", "col1_flux_old_cut", "col2_cut_assignment", "col3_averaging_commutator", "col4_continued_reference", "rep"]
CHANGED = ["col2_cut_assignment", "rep", "rep_defect", "sum4", "M_rep_a", "M_rep_b", "M_rep_minus_quad_a", "M_rep_minus_quad_b", "closure"]
CHANGED_ANALYTIC = CHANGED + ["col3_averaging_commutator", "C_u_a", "C_u_b", "kernel_rel_J_a"]
WINDOWS = [(-np.inf, 14.75), (14.75, 17.5), (17.5, np.inf)]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def arr(pairs, k):
    return np.array([np.nan if p.get(k) is None else float(p[k]) for p in pairs], float)


def numeric_keys(pairs):
    ks = set()
    for k, v in pairs[0].items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            ks.add(k)
    return ks


def key_compare(pc, pb):
    """per-pair numeric keys common to v4c and v4b: bitwise equal (NaN-aware) or the max |difference|"""
    out = {"n_pairs_v4c": len(pc), "n_pairs_v4b": len(pb), "equal": [], "differ": {}}
    if len(pc) != len(pb):
        out["pair_count_mismatch"] = True
        return out
    common = sorted(numeric_keys(pc) & numeric_keys(pb))
    out["only_v4c"] = sorted(numeric_keys(pc) - numeric_keys(pb))
    out["only_v4b"] = sorted(numeric_keys(pb) - numeric_keys(pc))
    for k in common:
        a, b = arr(pc, k), arr(pb, k)
        same = np.array_equal(a, b, equal_nan=True)
        if same:
            out["equal"].append(k)
        else:
            out["differ"][k] = float(np.nanmax(np.abs(a - b)))
    return out


def stats_block(pairs, v4b_pairs):
    s = {"n": len(pairs)}
    rd = arr(pairs, "rep_defect"); rdb = arr(v4b_pairs, "rep_defect")
    dM = arr(pairs, "dM")
    s["rep_defect_v4c_max"] = float(np.max(np.abs(rd))); s["rep_defect_v4c_median"] = float(np.median(np.abs(rd)))
    s["rep_defect_v4b_max"] = float(np.max(np.abs(rdb))); s["rep_defect_v4b_median"] = float(np.median(np.abs(rdb)))
    s["rep_defect_v4c_rel_median"] = float(np.median(np.abs(rd) / np.abs(dM)))
    s["rep_defect_v4c_sum"] = float(np.sum(rd)); s["rep_defect_v4b_sum"] = float(np.sum(rdb))
    s["n_rep_defect_v4c_above_1e-7"] = int(np.sum(np.abs(rd) > 1e-7))
    s["closure_max"] = float(np.max(np.abs(arr(pairs, "closure"))))
    s["M_rep_minus_quad_max_v4c"] = float(np.nanmax(np.abs(np.r_[arr(pairs, "M_rep_minus_quad_a"), arr(pairs, "M_rep_minus_quad_b")])))
    s["M_rep_minus_quad_max_v4b"] = float(np.nanmax(np.abs(np.r_[arr(v4b_pairs, "M_rep_minus_quad_a"), arr(v4b_pairs, "M_rep_minus_quad_b")])))
    for k in COLS:
        s["sum_" + k + "_v4c"] = float(np.sum(arr(pairs, k)))
        s["sum_" + k + "_v4b"] = float(np.sum(arr(v4b_pairs, k)))
        s["maxabs_" + k + "_v4c"] = float(np.max(np.abs(arr(pairs, k))))
    for k in ("col2_momentum_part", "col4_momentum_part"):
        s["sum_" + k] = float(np.sum(arr(pairs, k)))
    s["identity_sum_check"] = float(s["sum_dM_v4c"] - sum(s["sum_" + k + "_v4c"] for k in COLS[1:5]))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4c-dir", required=True)
    ap.add_argument("--v4b-root", required=True)
    ap.add_argument("--k1", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    K1 = json.load(open(a.k1))
    man = {}
    for line in open(os.path.join(a.v4c_dir, "MANIFEST_w6_v4c.md5")):
        h, p = line.split()
        man[p] = h
    out = {"script": os.path.abspath(__file__), "v4c_dir": a.v4c_dir, "k1_json": a.k1, "k1_md5": md5(a.k1), "manifest": {}, "rungs": {}, "order": []}
    for p, h in man.items():
        out["manifest"][p] = {"listed": h, "actual": md5(os.path.join(a.v4c_dir, p))}
    out["status"] = open(os.path.join(a.v4c_dir, "run_w6_v4c.status")).read()
    Z = {}
    for rung in ("s2", "s4", "s8"):
        tag, stem, sig = TAGS[rung]
        for route in ("engine", "analytic"):
            key = "%s_%s" % (rung, route)
            out["order"].append(key)
            pc = os.path.join(a.v4c_dir, key, "firstlaw_records_v4c_%s.json" % tag)
            pbdir = stem + ("_v4b_ek" if route == "engine" else "_v4b")
            pb = os.path.join(a.v4b_root, pbdir, "firstlaw_records_v4b_%s.json" % tag)
            C = json.load(open(pc)); B = json.load(open(pb))
            m = C["meta"]
            R = {"sigma": sig, "route": route, "v4c_json": pc, "v4c_md5": md5(pc), "v4b_json": pb, "v4b_md5": md5(pb),
                 "meta_version": m.get("version"), "meta_kernel": m.get("kernel"), "momentum_weight": m.get("momentum_weight"),
                 "weight_rule_check_K0_vs_closed_form_max": m.get("weight_rule_check_K0_vs_closed_form_max"),
                 "K1_total_weight_at_sites": m.get("K1_total_weight_at_sites"), "split_J": m.get("split_J"), "seconds": C.get("seconds")}
            R["outer"] = stats_block(C["pairs_outer"], B["pairs_outer"])
            R["inner"] = stats_block(C["pairs_inner"], B["pairs_inner"]) if C["pairs_inner"] else None
            offs = {}
            for ok, opairs in C["pairs_offsets"].items():
                bp = B["pairs_offsets"].get(ok, [])
                if opairs:
                    rd = arr(opairs, "rep_defect")
                    offs[ok] = {"n": len(opairs), "rep_defect_v4c_max": float(np.max(np.abs(rd))), "rep_defect_v4c_median": float(np.median(np.abs(rd))),
                                "rep_defect_v4b_max": float(np.max(np.abs(arr(bp, "rep_defect")))) if bp else None}
            R["offsets"] = offs
            # independent key comparison against v4b
            R["compare_outer"] = key_compare(C["pairs_outer"], B["pairs_outer"])
            R["compare_inner"] = key_compare(C["pairs_inner"], B["pairs_inner"]) if C["pairs_inner"] else None
            R["compare_offsets"] = {ok: key_compare(C["pairs_offsets"][ok], B["pairs_offsets"][ok]) for ok in C["pairs_offsets"] if C["pairs_offsets"][ok]}
            allowed = set(CHANGED if route == "engine" else CHANGED_ANALYTIC)
            bad = set(R["compare_outer"]["differ"]) - allowed
            if R["compare_inner"]:
                bad |= set(R["compare_inner"]["differ"]) - allowed
            for ok, cc in R["compare_offsets"].items():
                bad |= set(cc["differ"]) - allowed
            R["unexpected_changed_keys"] = sorted(bad)
            R["expected_but_unchanged_keys"] = sorted(allowed - set(R["compare_outer"]["differ"]))
            # phases (T_b windows) with the K_1 weight and v4b's
            Tb = arr(C["pairs_outer"], "T_b"); ph = []
            for lo, hi in WINDOWS:
                sel = (Tb > lo) & (Tb <= hi)
                row = {"window": [lo if np.isfinite(lo) else None, hi if np.isfinite(hi) else None], "n": int(sel.sum())}
                for k in COLS:
                    row[k] = float(np.sum(arr(C["pairs_outer"], k)[sel]))
                    row[k + "_v4b"] = float(np.sum(arr(B["pairs_outer"], k)[sel]))
                row["col2_momentum_part"] = float(np.sum(arr(C["pairs_outer"], "col2_momentum_part")[sel]))
                row["max_rep_defect_v4c"] = float(np.max(np.abs(arr(C["pairs_outer"], "rep_defect")[sel]))) if sel.any() else None
                row["max_rep_defect_v4b"] = float(np.max(np.abs(arr(B["pairs_outer"], "rep_defect")[sel]))) if sel.any() else None
                ph.append(row)
            R["phases"] = ph
            # v1.1's own K_1 reading (engine route only)
            if route == "engine" and key in K1["rungs"]:
                kk = K1["rungs"][key]
                for blk in ("pairs_outer", "pairs_inner"):
                    P1 = kk[blk]; P4 = C[blk]
                    if not P1:
                        continue
                    t1 = np.array([float(p["T_a"]) for p in P1]); t4 = arr(P4, "T_a")
                    ok = len(t1) == len(t4) and np.array_equal(t1, t4)
                    d = {"aligned": bool(ok), "n": len(P1)}
                    if ok:
                        for k1k, k4k in (("col2_K1", "col2_cut_assignment"), ("rep_K1", "rep"), ("rep_defect_K1", "rep_defect"), ("col4_K1", "col4_continued_reference")):
                            if k1k in P1[0]:
                                v1 = np.array([float(p[k1k]) for p in P1]); v4 = arr(P4, k4k)
                                d["maxabs_" + k1k + "_minus_v4c"] = float(np.max(np.abs(v1 - v4)))
                    R["vs_v1_1_" + blk] = d
            out["rungs"][key] = R
            po = C["pairs_outer"]
            for k in ("T_a", "T_b", "rate", "rows_mid", "clock_residual_mid", "gate", "dM", "col1_flux_old_cut", "col2_cut_assignment",
                      "col2_momentum_part", "col3_averaging_commutator", "col4_continued_reference", "rep", "rep_defect",
                      "col2_cut_assignment_v4b", "rep_defect_v4b"):
                Z[key + "__" + k] = arr(po, k)
            Ta, Tbb = Z[key + "__T_a"], Z[key + "__T_b"]
            Z[key + "__gap_breaks"] = np.array([i for i in range(1, len(Ta)) if Ta[i] != Tbb[i - 1]], int)
            Z[key + "__v4b_rep_defect_json"] = arr(B["pairs_outer"], "rep_defect")
    oj = os.path.join(a.out_dir, "w6_readout_v1_2.json"); on = os.path.join(a.out_dir, "w6_readout_v1_2.npz")
    json.dump(out, open(oj, "w"), indent=1, default=float)
    np.savez(on, **Z)
    # console summary
    print("manifest:", all(v["listed"] == v["actual"] for v in out["manifest"].values()), len(out["manifest"]))
    for key in out["order"]:
        R = out["rungs"][key]; o = R["outer"]
        print("%-12s mw=%s | outer n %d rep defect max %.3g med %.3g (v4b %.3g, %.3g) >1e-7: %d | sums dM %.6g c1 %.6g c2 %.6g (v4b %.6g; mom %.6g) c3 %.6g c4 %.6g (mom %.3g) rep %.6g (v4b %.6g) idchk %.2g | unexpected %s unchanged-expected %s"
              % (key, "K_1" in (R["momentum_weight"] or ""), o["n"], o["rep_defect_v4c_max"], o["rep_defect_v4c_median"], o["rep_defect_v4b_max"], o["rep_defect_v4b_median"],
                 o["n_rep_defect_v4c_above_1e-7"], o["sum_dM_v4c"], o["sum_col1_flux_old_cut_v4c"], o["sum_col2_cut_assignment_v4c"], o["sum_col2_cut_assignment_v4b"],
                 o["sum_col2_momentum_part"], o["sum_col3_averaging_commutator_v4c"], o["sum_col4_continued_reference_v4c"], o["sum_col4_momentum_part"],
                 o["sum_rep_v4c"], o["sum_rep_v4b"], o["identity_sum_check"], R["unexpected_changed_keys"], R["expected_but_unchanged_keys"]))
        if R["inner"]:
            i = R["inner"]
            print("    inner n %d rep defect max %.3g med %.3g (v4b max %.3g) >1e-7: %d" % (i["n"], i["rep_defect_v4c_max"], i["rep_defect_v4c_median"], i["rep_defect_v4b_max"], i["n_rep_defect_v4c_above_1e-7"]))
        print("    offsets max", {k: "%.2g" % v["rep_defect_v4c_max"] for k, v in R["offsets"].items()})
        for blk in ("pairs_outer", "pairs_inner"):
            if ("vs_v1_1_" + blk) in R:
                print("    vs v1.1", blk, R["vs_v1_1_" + blk])
    print("wrote", oj, on)


if __name__ == "__main__":
    main()
