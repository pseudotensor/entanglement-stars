#!/usr/bin/env python
"""cmp_rd2_ruling_v1.py (busi, analyst; PERF RD-2/RD-5 self-test (c), second reading): the replay judged by the gate of
REVIEWA_perf_tolerance_ruling_v1 sec. 4 with the table of its sec. 2.3 (deliveries/lead_gaps), beside the builder's own budget
(cmp_rd2_replay_v1.py). (A) each change arm against the default arm, every class of the table at every record:
  identical: the continuation entries (mode, nsub, verified, inversions, n_intermediate, n_pocket, pockets_occupied_positive,
    holes_empty_negative, rotated_blocks, partly_occupied_blocks, mask_equals_negative_set), n_pocket, n_refill_R0, inversions_total, the
    trapping-zero and tangency counts, late_surface flag, converged/unconverged, the gauge and regime labels, the v36c set entries, each
    solve's status and meets_stop_rule, the srf decisions, R_sh, the spectra masks;
  E/E0 absolute and E relative 3.0e-8; roots (R_t, r_in, R_c, trapping_zeros, threshold_radius) 3.4e-7; min_N_minus_absv 4.6e-9;
  named observables |d| <= 1e-6 |x| + 1e-8; site fields (N, v, N_applied, v_applied, N_nodes, v_nodes) 1e-7; source and handoff rows 1e-6;
  spectra e rows relative to max |e| 1e-8; indicator_pass reported with both values (a flip counts only when a B_estimate lies outside a
  decade of 1e-9); every other difference reported. Also the segment conditions: at least 8 records; a set change in the segment; no
  monotone growth (the last quarter's median E/E0 difference at most 10 times the first quarter's).
(B) each arm against the stored production records within E/E0 3.0e-7, roots 3.4e-6, min 4.6e-8, fields 2.5e-5, the set decisions identical,
  and the same-class test (the change arm's distance at most twice the default arm's, or below the (A) gate).
usage: cmp_rd2_ruling_v1.py --arm I=DIR:TAG --arm C=DIR:TAG [--arm W=DIR:TAG] --rs RS --prod-json J --prod-dir D --prod-tag T --t-min X
       [--slice I=PATH --slice C=PATH ...] --out JSON   (the end slices: masks, f_k identical, e rel 1e-8, sea/R2 densities 1e-9, applied fields 1e-7)"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmp_rd2_replay_v1 import cmp_record, rows_by_T  # noqa: E402

GATE = {"E_over_E0": 3.0e-8, "E": 3.0e-8, "root": 3.4e-7, "min": 4.6e-9, "fields": 1e-7, "rows": 1e-6, "spectra": 1e-8}
FLOORB = {"E_over_E0": 3.0e-7, "root": 3.4e-6, "min": 4.6e-8, "fields": 2.5e-5}
CONT = {"mode", "nsub", "verified", "inversions", "n_intermediate", "n_pocket", "pockets_occupied_positive", "holes_empty_negative",
        "rotated_blocks", "partly_occupied_blocks", "mask_equals_negative_set"}
NAMED = ["N_min", "kappa_T", "H_in", "E_dust_shells", "E_pairs_shells", "pairs_inside_Rt", "pairs_outside_Rt", "misner_sharp_at_Rt", "v_at_Rc",
         "min_occupied_weight", "W_over_2g_at_Rt", "R_t_over_2gE", "four_gE_kappa_T", "R_t_over_2g_MS"]
FIELDS = ("N", "v", "N_applied", "v_applied", "N_nodes", "v_nodes")
HROWS = ("N2avg", "eps_sea", "eps_R2", "eps_dust", "F_face_dust", "fc_dust", "b_dust", "sy_k_dust", "F_face_pairs", "fc_pairs", "b_pairs", "sy_k_pairs",
         "S_r_ell", "S_t_ell", "n_dust")


def identical_key(k):
    m = re.match(r"^/continuation/[^/]+/([^/\[#]+)$", k)
    if m:
        return m.group(1) in CONT
    if re.match(r"^/(n_pocket|n_refill_R0)/[^/]+$", k):
        return True
    if k in ("/inversions_total", "/trapping_zeros#len", "/tangencies#len", "/late_surface/flag", "/converged", "/unconverged", "/gauge/label",
             "/regime/label", "/mask_equals_negative_set"):
        return True
    if k.startswith("/srf/"):
        return True
    # the v36c set entry (correction of 12:2x UTC 21 Oct, to the ruling's text): the record's /v36c/nk_set and record_set, and each solve's
    # kept round, kept set and kept signature, and its per-round sets; NOT the iterate traces (steps[*], their trials and plain_fallback
    # trials, evaluation_sets), which the ruling's table lists as reported, and not the comparisons with the previous record (None after a
    # restart). The first version matched /v36c/...set anywhere and so gated the per-trial "set" counts of the iterate traces.
    if k in ("/v36c/nk_set", "/v36c/record_set"):
        return True
    if re.match(r"^/solve_v32/solves\[\d+\]/v36c/(kept_round|kept_set|kept_sig|sets(\[\d+\])?(#len)?)$", k):
        return True
    if k.startswith("/solve_v32/") and (k.endswith("/status") or k.endswith("meets_stop_rule") or k.endswith("/converged")):
        return True
    return False


def row(Z, name, t, rs):
    ix = rows_by_T(Z, rs)
    if t not in ix or name not in Z.files:
        return None
    return np.asarray(Z[name][ix[t]], float)


def judge(x, y, t, J, Z, rs, cls):
    """x against y at record t; cls 'A' (the table) or 'B' (the floor). Returns (ok, lines, numbers)."""
    fl, disc = cmp_record(J[x][t], J[y][t])
    out = []; ok = True; nums = {}
    g = GATE if cls == "A" else None
    e = fl.get("/E_over_E0", (0.0, 0.0))[0]; nums["E_over_E0"] = e
    lim = GATE["E_over_E0"] if cls == "A" else FLOORB["E_over_E0"]
    ok &= e <= lim; out.append("E/E0 |d| %.2e (<= %.1e)" % (e, lim))
    if cls == "A":
        er = fl.get("/E", (0.0, 0.0))[1]; ok &= er <= GATE["E"]; out.append("E rel %.2e (<= %.1e)" % (er, GATE["E"]))
    rd = max([fl[k][0] for k in fl if k in ("/R_t", "/r_in", "/R_c") or k.startswith("/trapping_zeros[") or k.startswith("/threshold_radius/")] or [0.0])
    nums["root"] = rd
    lim = GATE["root"] if cls == "A" else FLOORB["root"]
    ok &= rd <= lim; out.append("roots |d| %.2e (<= %.1e)" % (rd, lim))
    m = fl.get("/min_N_minus_absv", (0.0, 0.0))[0]; nums["min"] = m
    lim = GATE["min"] if cls == "A" else FLOORB["min"]
    ok &= m <= lim; out.append("min(N-|v|) |d| %.2e (<= %.1e)" % (m, lim))
    rsh = fl.get("/R_sh", (0.0, 0.0))[0]; ok &= rsh == 0.0; out.append("R_sh |d| %.1e (== 0)" % rsh)
    ident = [(k, a_, b_) for k, a_, b_ in disc if identical_key(k)]
    ok &= not ident
    out.append("identical class: %d differences%s" % (len(ident), (" " + "; ".join("%s: %r vs %r" % s for s in ident[:6])) if ident else ""))
    ib = [(k, a_, b_) for k, a_, b_ in disc if k.endswith("indicator_pass")]
    if ib:
        ba = (J[x][t].get("indicator") or {}).get("B_estimate"); bb = (J[y][t].get("indicator") or {}).get("B_estimate")
        hard = any(v is None or v < 1e-10 or v > 1e-8 for v in (ba, bb))
        ok &= not hard
        out.append("indicator_pass flip: %r vs %r (B_estimate %s vs %s): %s" % (ib[0][1], ib[0][2], ba, bb, "a decision difference" if hard else "inside a decade of 1e-9, reported"))
    if cls == "A":
        bad = []
        for k_ in NAMED:
            a_, b_ = J[x][t].get(k_), J[y][t].get(k_)
            if isinstance(a_, (int, float)) and isinstance(b_, (int, float)) and not isinstance(a_, bool):
                if not (abs(a_ - b_) <= 1e-6 * abs(b_) + 1e-8):
                    bad.append((k_, a_, b_))
            elif a_ != b_:
                bad.append((k_, a_, b_))
        ok &= not bad
        worst = max([abs(J[x][t][k_] - J[y][t][k_]) / (1e-6 * abs(J[y][t][k_]) + 1e-8) for k_ in NAMED
                     if isinstance(J[x][t].get(k_), (int, float)) and isinstance(J[y][t].get(k_), (int, float))] or [0.0])
        out.append("named observables: %d outside |d| <= 1e-6 |x| + 1e-8 (largest d / allowance %.2e)%s" % (len(bad), worst, (" " + str(bad[:3])) if bad else ""))
    # fields, rows, spectra
    fmax = 0.0; parts = []
    for f in FIELDS:
        ra, rb = row(Z[x]["fields"], f, t, rs), row(Z[y]["fields"], f, t, rs)
        if ra is None or rb is None:
            continue
        d = float(np.max(np.abs(ra - rb))); fmax = max(fmax, d); parts.append("%s %.1e" % (f, d))
    nums["fields"] = fmax
    lim = GATE["fields"] if cls == "A" else FLOORB["fields"]
    ok &= fmax <= lim; out.append("site fields max %.2e (<= %.0e): %s" % (fmax, lim, ", ".join(parts)))
    if cls == "A":
        rmax = 0.0; rk = None
        for src, names in (("fields", [n for n in Z[x]["fields"].files if n.startswith("eps_") or n.startswith("J_")]), ("handoff", HROWS)):
            for n in names:
                ra, rb = row(Z[x][src], n, t, rs), row(Z[y][src], n, t, rs)
                if ra is None or rb is None:
                    continue
                d = float(np.max(np.abs(ra - rb)))
                if d > rmax:
                    rmax, rk = d, n
        ok &= rmax <= GATE["rows"]; out.append("source and handoff rows max %.2e (%s) (<= 1e-6)" % (rmax, rk))
    sa = rows_by_T(Z[x]["spectra"], rs); sb = rows_by_T(Z[y]["spectra"], rs)
    ks = [k for k in Z[x]["spectra"].files if k.startswith("mask_")]
    mask_ok = all(np.array_equal(Z[x]["spectra"][k][sa[t]], Z[y]["spectra"][k][sb[t]]) for k in ks)
    erel = max(float(np.max(np.abs(Z[x]["spectra"][k.replace("mask", "e")][sa[t]] - Z[y]["spectra"][k.replace("mask", "e")][sb[t]])))
               / max(float(np.max(np.abs(Z[y]["spectra"][k.replace("mask", "e")][sb[t]]))), 1e-300) for k in ks)
    ok &= mask_ok
    if cls == "A":
        ok &= erel <= GATE["spectra"]
    out.append("spectra masks %s; e rows rel %.2e%s" % ("equal" if mask_ok else "DIFFER", erel, " (<= 1e-8)" if cls == "A" else " (reported)"))
    other = [(k, a_, b_) for k, a_, b_ in disc if not identical_key(k) and not k.endswith("indicator_pass")]
    out.append("reported discrete differences: %d%s" % (len(other), (" " + "; ".join("%s: %r vs %r" % s for s in other[:6])) if other else ""))
    out.append("r_min_N_minus_absv |d| %.2e (reported)" % fl.get("/r_min_N_minus_absv", (0.0, 0.0))[0])
    return bool(ok), out, nums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--rs", type=float, required=True)
    ap.add_argument("--prod-json", required=True); ap.add_argument("--prod-dir", required=True); ap.add_argument("--prod-tag", required=True)
    ap.add_argument("--t-min", type=float, default=-1.0)
    ap.add_argument("--slice", action="append", default=[], help="NAME=PATH: the arm's end slice (compared against the first arm's)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = []
    for s in a.arm:
        n, rest = s.split("=", 1); d, t = rest.rsplit(":", 1); arms.append((n, d, t))
    ref = arms[0][0]
    J = {}; Z = {}
    for n, d, t in arms:
        J[n] = {round(r["T_over_rs"], 6): r for r in json.load(open(os.path.join(d, "collapse_%s.json" % t)))["records"] if r["T_over_rs"] > a.t_min + 1e-9}
        Z[n] = {k: np.load(os.path.join(d, "collapse_%s_%s.npz" % (t, k)), allow_pickle=True) for k in ("fields", "spectra", "handoff")}
    J["prod"] = {round(r["T_over_rs"], 6): r for r in json.load(open(a.prod_json))["records"] if r["T_over_rs"] > a.t_min + 1e-9}
    Z["prod"] = {k: np.load(os.path.join(a.prod_dir, "collapse_%s_%s.npz" % (a.prod_tag, k)), allow_pickle=True) for k in ("fields", "spectra", "handoff")}
    ts = sorted(set.intersection(*[set(J[n]) for n, _, _ in arms]))
    rep = {"records": {}, "verdict": {}}
    seteq = all(sorted(J[n]) == sorted(J[ref]) for n, _, _ in arms)
    print("ruling gate (REVIEWA_perf_tolerance_ruling_v1 sec. 4): records %s; record sets equal %s" % (ts, seteq))
    vA = {n: seteq for n, _, _ in arms[1:]}; vB = {n: True for n, _, _ in arms}; dA = {n: [] for n, _, _ in arms[1:]}; dB = {n: {} for n, _, _ in arms}
    for t in ts:
        print("== T/rs %.6f" % t)
        for n, _, _ in arms[1:]:
            ok, lines, nums = judge(n, ref, t, J, Z, a.rs, "A")
            vA[n] &= ok; dA[n].append(nums["E_over_E0"])
            print("  (A) %s vs %s: %s" % (n, ref, "PASS" if ok else "FAIL"))
            for l_ in lines:
                print("       " + l_)
            rep["records"].setdefault("%.6f" % t, {})["A %s" % n] = {"ok": ok, "lines": lines, "nums": nums}
        for n, _, _ in arms:
            if t not in J["prod"]:
                continue
            ok, lines, nums = judge(n, "prod", t, J, Z, a.rs, "B")
            dB[n][t] = nums
            vB[n] &= ok
            print("  (B) %s vs prod: %s" % (n, "PASS" if ok else "FAIL"))
            for l_ in lines:
                print("       " + l_)
            rep["records"].setdefault("%.6f" % t, {})["B %s" % n] = {"ok": ok, "lines": lines, "nums": nums}
    # the same-class test (informative off the W2 lineage) and the segment conditions
    same = {}
    for n, _, _ in arms[1:]:
        res = []
        for t in ts:
            if t in dB[n] and t in dB[ref]:
                for q in ("E_over_E0", "root", "min", "fields"):
                    gate_q = {"E_over_E0": GATE["E_over_E0"], "root": GATE["root"], "min": GATE["min"], "fields": GATE["fields"]}[q]
                    res.append(dB[n][t][q] <= 2.0 * dB[ref][t][q] or dB[n][t][q] <= gate_q)
        same[n] = all(res) if res else None
    k = max(1, len(ts) // 4)
    growth = {n: (float(np.median(dA[n][-k:])), float(np.median(dA[n][:k]))) for n in dA}
    nogrowth = {n: (g_[0] <= 10.0 * g_[1]) if g_[1] > 0 else (g_[0] == 0.0) for n, g_ in growth.items()}
    setchange = any(J[ref][t].get("inversions_total") != J[ref][ts[i - 1]].get("inversions_total") or J[ref][t].get("n_pocket") != J[ref][ts[i - 1]].get("n_pocket")
                    for i, t in enumerate(ts) if i > 0)
    cond8 = len(ts) >= 8
    for n in vA:
        lines_ok = vA[n] and nogrowth[n]
        rep["verdict"][n] = {"A_lines_every_record": "PASS" if vA[n] else "FAIL", "no_monotone_growth": nogrowth[n], "growth_last_first_quarter_median": growth[n],
                             "records": len(ts), "at_least_8_records": cond8, "set_change_in_segment": setchange,
                             "A": "PASS" if (lines_ok and cond8) else "FAIL", "B": "PASS" if vB[n] else "FAIL", "B_default_arm": "PASS" if vB[ref] else "FAIL",
                             "same_class_test": same[n]}
        print("VERDICT %s: (A) lines at every record %s; no monotone growth %s (last/first quarter median E/E0 %.2e / %.2e); records %d (at least 8: %s); "
              "set change in the segment %s; (A) as written %s; (B) %s (the default arm %s); same-class test %s" % (
                  n, rep["verdict"][n]["A_lines_every_record"], nogrowth[n], growth[n][0], growth[n][1], len(ts), cond8, setchange, rep["verdict"][n]["A"],
                  rep["verdict"][n]["B"], rep["verdict"][n]["B_default_arm"], same[n]))
    # the end slice (the table's masks, spectra, slice densities 1e-9, f_k identical, the applied fields 1e-7)
    sl = dict(x.split("=", 1) for x in a.slice)
    if ref in sl:
        S0 = np.load(sl[ref], allow_pickle=False)
        ks = sorted(int(k.split("_")[1]) for k in S0.files if k.startswith("mask_"))
        for n, _, _ in arms[1:]:
            if n not in sl:
                continue
            S1 = np.load(sl[n], allow_pickle=False)
            r_ = {"masks_equal": True, "f_identical": True, "e_rel": 0.0, "dens_sea": 0.0, "dens_R2": 0.0, "dens_dust": 0.0,
                  "applied": max(float(np.max(np.abs(S1["N_applied"] - S0["N_applied"]))), float(np.max(np.abs(S1["v_applied"] - S0["v_applied"]))))}
            for k in ks:
                r_["masks_equal"] &= bool(np.array_equal(S1["mask_%d" % k], S0["mask_%d" % k]))
                r_["f_identical"] &= bool(S1["f_%d" % k].shape == S0["f_%d" % k].shape and S1["f_%d" % k].tobytes() == S0["f_%d" % k].tobytes())
                e0 = S0["e_%d" % k]; r_["e_rel"] = max(r_["e_rel"], float(np.max(np.abs(S1["e_%d" % k] - e0))) / max(float(np.max(np.abs(e0))), 1e-300))
                for nm, key in (("dens_sea", "sea_%d"), ("dens_R2", "R2_%d"), ("dens_dust", "dust_%d")):
                    A_, B_ = S1[key % k], S0[key % k]
                    if A_.shape != B_.shape:
                        r_["masks_equal"] = False; continue
                    if A_.size:
                        r_[nm] = max(r_[nm], float(np.max(np.abs(np.sum(np.abs(A_) ** 2, axis=1) - np.sum(np.abs(B_) ** 2, axis=1)))))
            ok = r_["masks_equal"] and r_["f_identical"] and r_["e_rel"] <= GATE["spectra"] and r_["dens_sea"] <= 1e-9 and r_["dens_R2"] <= 1e-9 and r_["applied"] <= GATE["fields"]
            r_["ok"] = bool(ok)
            rep["slice %s vs %s" % (n, ref)] = r_
            if n in rep["verdict"]:
                rep["verdict"][n]["end_slice"] = "PASS" if ok else "FAIL"
                if not ok:
                    rep["verdict"][n]["A"] = "FAIL"
            print("end slice %s vs %s: %s: masks %s, f_k %s, e rel %.2e (<= 1e-8), sea densities %.2e, R2 densities %.2e (<= 1e-9), dust densities %.2e (reported), applied fields %.2e (<= 1e-7)" % (
                n, ref, "PASS" if ok else "FAIL", "equal" if r_["masks_equal"] else "DIFFER", "identical" if r_["f_identical"] else "DIFFER", r_["e_rel"], r_["dens_sea"], r_["dens_R2"], r_["dens_dust"], r_["applied"]))
        for n in rep["verdict"]:
            print("FINAL %s: (A) %s, (B) %s" % (n, rep["verdict"][n]["A"], rep["verdict"][n]["B"]))
    json.dump(rep, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
