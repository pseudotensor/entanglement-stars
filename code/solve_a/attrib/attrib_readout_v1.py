#!/usr/bin/env python
"""attrib_readout_v1.py (composition builder, the solve lane): the producer of W5_W4_formation_attribution_v1 (W4 and W5 of
final4/WEB40I_TASKS/busi.md). Reads stored files only and saves every number the delivery prints:
  - the formation of each engine history (the production's 3 averaging scales, its tau 4 and tau 2 step controls, the 6 W5 rungs,
    W4 (b) with the pair sources off, W2's conserving-step history) by the ladder reader (ladders_readout_v1.load_member and summarize:
    the first record holding a trapping zero, R_t, R_t/(2 g E), the residual, the certification rule of MASS_ladder_v1);
  - W4 (a), the dust comparator's runs (dust_comparator_v1c outputs: the fold clock and radius over 2gE0 by the integrator's event,
    the first record on the 0.125 grid holding a root with its outer radius over 2gE0, the record before);
  - W5's slice readout (w5_readout_v1.json of the busi lane's w5_readout_v1.py, run on the box: the pair totals per channel, the
    inner-shell share above pi/4 and the 16 kappa E_kappa tail at T/r_s 14);
  - the differences that the attribution table prints (each switch against its reference, the formation clock and radius).
Output: <out>/attrib_readout_v1.json and attrib_readout_v1.npz (the per-record T, E/E0, min(N - |v|), R_t of every history; the
comparator records' depth and outer root), md5 manifest.
Usage (box CPU): python attrib_readout_v1.py --ladders-code DIR --w4a-dir DIR --w5-json FILE --out-dir DIR"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

L = "<internal>/ladders"
PS = "<internal>/pod_sync"


def w5(K, t):
    card = "r1" if t == 8 else "p6"
    tg = "lad_W5_k%d_tau%d_m0p2_rs64_n2048_R04_k%d_tau%d_ell4_prep4_A_R2_avg_plain_v36d" % (K, t, K, t)
    return os.path.join(L, card, "runs", tg, "collapse_%s.json" % tg)


HIST = [
    ("P64_s2", "<internal>/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d.json",
     "production sigma/a 2 (the composite history, Newton across its seams)", "production"),
    ("P64_s4", "<internal>/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40.json",
     "production sigma/a 4 (rung 4, v36d from T = 0)", "production"),
    ("P64_s8", PS + "/pod2_engine/v36d_from0/collapse_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_to40.json",
     "production sigma/a 8 (rung 8, v36d from T = 0)", "production"),
    ("P64_s4_tau4", PS + "/pod1_engine/C3prod19/collapse_c19_rs64_A_R2_tau4_ell4_avg.json", "production sigma/a 4, tau 4 (engine v19)", "step"),
    ("P64_s4_tau2", PS + "/pod1_engine/C3prod25/collapse_c25_rs64_A_R2_tau2_ell4_avg.json", "production sigma/a 4, tau 2 (engine v25)", "step"),
    ("W5_k10_tau8", w5(10, 8), "W5 kappa_sea 10, tau 8", "W5"), ("W5_k14_tau8", w5(14, 8), "W5 kappa_sea 14, tau 8", "W5"),
    ("W5_k20_tau8", w5(20, 8), "W5 kappa_sea 20, tau 8", "W5"), ("W5_k10_tau4", w5(10, 4), "W5 kappa_sea 10, tau 4", "W5"),
    ("W5_k14_tau4", w5(14, 4), "W5 kappa_sea 14, tau 4", "W5"), ("W5_k20_tau4", w5(20, 4), "W5 kappa_sea 20, tau 4", "W5"),
    ("W4b_pairsoff", "<internal>/collapse_w4b_pairsoff_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36g.json",
     "W4 (b): pair sources off (engine v36g), sigma/a 4, tau 8, to 14", "W4b"),
    ("W2_dg", L + "/r2/W2_v36l/collapse_lad_W2_dg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36f.json",
     "W2: the conserving step (dg, v36f to 12.375; the v36l overlap rule from 12.5)", "W4b_conserving"),
]
W4A_RUNS = ["s2_density_continuum_rest_lapse-on_adaptive", "s4_density_continuum_rest_lapse-on_adaptive", "s8_density_continuum_rest_lapse-on_adaptive",
            "s4_density_continuum_rest_lapse-on_frozen-tau8", "s4_density_continuum_rest_lapse-on_frozen-tau4", "s0.25_density_continuum_rest_lapse-on_adaptive",
            "s2_density_sine_rest_lapse-on_adaptive", "s4_density_sine_rest_lapse-on_adaptive", "s8_density_sine_rest_lapse-on_adaptive",
            "s4_radial_continuum_rest_lapse-on_adaptive", "s4_vector_continuum_rest_lapse-on_adaptive", "s4_density_continuum_read_lapse-on_adaptive",
            "s2_density_continuum_rest_lapse-off_adaptive", "s4_density_continuum_rest_lapse-off_adaptive", "s8_density_continuum_rest_lapse-off_adaptive"]


def md5sum(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladders-code", required=True)
    ap.add_argument("--w4a-dir", required=True)
    ap.add_argument("--w5-json", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.ladders_code)
    import ladders_readout_v1 as LR
    os.makedirs(a.out_dir, exist_ok=True)
    H, arrays = {}, {}
    for name, path, label, role in HIST:
        meta, cols, ints, zeros_all = LR.load_member(name, path, label, role, None)
        s = LR.summarize(meta, cols, ints, zeros_all)
        H[name] = {"meta": meta, "formation": s["formation"], "convergence": s["convergence"], "episodes": s["episodes"][:2],
                   "matched_from_formation": s["matched_from_formation"], "counts": {k: s["counts"][k] for k in ("N_B", "E0", "f_kin")}}
        for k in ("T", "EE0", "minNv", "Rt", "res"):
            arrays["%s__%s" % (name, k)] = cols[k]
        arrays["%s__counts" % name] = ints["counts"]
        print(name, (s["formation"] or {}).get("T"), (s["formation"] or {}).get("R_t"), flush=True)
    # W4 (a)
    W4A = {}
    for lab in W4A_RUNS:
        p = os.path.join(a.w4a_dir, lab + ".json")
        if not os.path.exists(p):
            W4A[lab] = {"absent": p}
            continue
        J = json.load(open(p))
        fo, fr, rb = J.get("fold") or {}, J.get("first_record") or {}, J.get("record_before") or {}
        W4A[lab] = {"path": p, "md5": md5sum(p), "spec": J.get("spec"), "E0": J.get("E0"), "two_gE0": J.get("two_gE0"),
                    "fold_T": fo.get("T_over_rs"), "fold_R": fo.get("R"), "fold_R_over_2gE0": fo.get("R_over_2gE0"), "fold_R_over_rs": fo.get("R_over_rs"),
                    "first_record_T": fr.get("T_over_rs"), "first_record_outer": fr.get("outer"), "first_record_outer_over_2gE0": fr.get("outer_over_2gE0"),
                    "first_record_inner": fr.get("inner"), "first_record_depth": fr.get("depth"), "record_before_T": rb.get("T_over_rs"),
                    "record_before_depth": rb.get("depth"), "E_drift_net_max": J.get("E_drift_net_max"), "end_label": J.get("end_label"),
                    "caustic_T": ((J.get("centre") or {}).get("caustic") or {}).get("T_over_rs") if isinstance((J.get("centre") or {}).get("caustic"), dict) else None,
                    "n_pinned_at_first_root": (J.get("centre") or {}).get("n_pinned_at_first_root")}
        recs = J.get("records") or []
        arrays["w4a_%s__T" % lab] = np.array([r.get("T_over_rs", np.nan) for r in recs], float)
        arrays["w4a_%s__depth" % lab] = np.array([np.nan if r.get("depth") is None else r["depth"] for r in recs], float)
    md = sorted(glob.glob(os.path.join(a.w4a_dir, "DUST_comparator_v1c.md")))
    w4a_files = {"md": md[0] if md else None, "md_md5": md5sum(md[0]) if md else None}
    # W5's slice readout
    W5 = json.load(open(a.w5_json))
    w5s = {}
    for r in W5:
        sl = r.get("slice") or {}
        key = "W5_k%d_tau%d" % (int(r["kappa_sea"]), int(r["tau"]))
        ch = sl.get("channels") or {}
        w5s[key] = {"n_pairs_per_channel": {k: c.get("n_pairs") for k, c in ch.items()}, "E_pairs_bare_per_channel": {k: c.get("E_pairs_bare") for k, c in ch.items()},
                    "E_pairs_killing_per_channel": {k: c.get("E_pairs_killing") for k, c in ch.items()},
                    "w_E_maj_pairs_per_channel": {k: c.get("w_E_maj_pairs") for k, c in ch.items()},
                    "w_E_maj_heavy_per_channel": {k: c.get("w_E_maj_heavy") for k, c in ch.items()},
                    "inner_share_per_channel": {k: c.get("inner_share_above_quarter") for k, c in ch.items()},
                    "eig_vs_stored_max": max([c.get("eig_vs_stored_max") or 0 for c in ch.values()] or [0]),
                    "pockets": sum(c.get("pockets_occupied_positive") or 0 for c in ch.values()),
                    **{k: sl.get(k) for k in ("sum_E_pairs_bare", "sum_n_pairs", "sum_E_pairs_killing", "sum_w_E_maj_pairs", "sum_w_E_maj_heavy",
                                              "sum_w_E_killing_excess_heavy", "top_channel_share_w_E_maj_pairs", "above_matched_share_w_E_maj_pairs",
                                              "matched_sum_E_pairs_bare", "matched_sum_w_E_maj_pairs", "inner_shell_particles", "inner_shell_share_above_quarter",
                                              "all_dust_share_above_quarter", "R_50", "eps_pairs_row_sum")},
                    "record_14": r.get("record_14"), "slice_note": r.get("slice_note"), "head": r.get("head")}
    # the differences (switch minus reference) at formation
    def d(a_, b_):
        fa, fb = H[a_]["formation"], H[b_]["formation"]
        if not (fa and fb):
            return None
        return {"reference": b_, "switch": a_, "dT_f": fa["T"] - fb["T"], "dR_t": fa["R_t"] - fb["R_t"], "dR_t_over_2gE": fa["R_t_over_2gE"] - fb["R_t_over_2gE"],
                "both_certified": bool(fa["certified"] and fb["certified"])}
    pairs = [("W5_k14_tau8", "W5_k10_tau8"), ("W5_k20_tau8", "W5_k10_tau8"), ("W5_k14_tau4", "W5_k10_tau4"), ("W5_k20_tau4", "W5_k10_tau4"),
             ("W5_k10_tau4", "W5_k10_tau8"), ("W5_k14_tau4", "W5_k14_tau8"), ("W5_k20_tau4", "W5_k20_tau8"),
             ("W5_k10_tau8", "P64_s4"), ("W4b_pairsoff", "P64_s4"), ("W2_dg", "P64_s4"), ("P64_s4_tau4", "P64_s4"), ("P64_s4_tau2", "P64_s4"),
             ("P64_s2", "P64_s4"), ("P64_s8", "P64_s4")]
    diffs = [x for x in (d(p, q) for p, q in pairs) if x]
    offsets = []
    for nm, lab in (("P64_s2", "s2_density_continuum_rest_lapse-on_adaptive"), ("P64_s4", "s4_density_continuum_rest_lapse-on_adaptive"),
                    ("P64_s8", "s8_density_continuum_rest_lapse-on_adaptive")):
        fo, w = H[nm]["formation"], W4A.get(lab, {})
        if fo and w.get("fold_T") is not None:
            offsets.append({"history": nm, "comparator": lab, "T_f": fo["T"], "R_t_over_2gE": fo["R_t_over_2gE"], "fold_T": w["fold_T"],
                            "fold_R_over_2gE0": w["fold_R_over_2gE0"], "first_record_T": w["first_record_T"],
                            "first_record_outer_over_2gE0": w["first_record_outer_over_2gE0"],
                            "offset_T_vs_fold": fo["T"] - w["fold_T"], "offset_T_vs_first_record": fo["T"] - w["first_record_T"],
                            "offset_ratio_vs_fold": fo["R_t_over_2gE"] - w["fold_R_over_2gE0"],
                            "offset_ratio_vs_first_record": fo["R_t_over_2gE"] - w["first_record_outer_over_2gE0"]})
    out = {"histories": H, "w4a": W4A, "w4a_files": w4a_files, "w5_slice": w5s, "w5_json": a.w5_json, "w5_json_md5": md5sum(a.w5_json),
           "diffs": diffs, "offsets": offsets, "order": [h[0] for h in HIST], "w4a_order": W4A_RUNS}
    jp = os.path.join(a.out_dir, "attrib_readout_v1.json")
    json.dump(out, open(jp, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    npz = os.path.join(a.out_dir, "attrib_readout_v1.npz")
    np.savez_compressed(npz, **arrays)
    with open(os.path.join(a.out_dir, "MANIFEST_attrib_readout_v1.md5"), "w") as f:
        for p in (jp, npz):
            f.write("%s  %s\n" % (md5sum(p), os.path.basename(p)))
    print("wrote", jp, npz)


if __name__ == "__main__":
    main()
