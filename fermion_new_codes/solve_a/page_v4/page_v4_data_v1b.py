#!/usr/bin/env python
"""page_v4_data_v1.py (the solve lane; PAGE_solved_history_v4): the producer. Runs on the box compute-host only (nice 10, at most 16 threads).

It reads stored files only and runs no propagation:
  - the B2 production's channel files kappa 1 to 18 (page_history_v6o, --interp linear --interp-sub 8, Chebyshev 1e-16, 8192 sites, the cuts 2, 16,
    24, 30, 48 r_s, the collar 16 r_s) through <internal>/prod (symlinks into <internal>/runs);
  - the high-kappa probes kappa 24, 32, 48 (ruling2 run (b)), the 16-sub-step control of kappa 18 (ruling2 run (c), partial where not final) and the
    piecewise-constant control kappa 1 to 8 (page_history_v6n) through the post-pass json;
  - the sigma/a 8 history's channels kappa 1 to 18 (the s8 prod folder of this lane: symlinks into <internal>/runs) and its 16-sub-step
    control of kappa 14;
  - W3's two completions (the sources held inside the gate radius, W3fs; the paired averaging widened inside, W3pa), kappa 1 and 2 to 40 r_s;
  - the post-pass (w1_postpass_v1f.py, the busi lane's) outputs: the B2 K = 18 composition from 40 r_s (W1_B2_composed_v3.json), this lane's reruns of
    the same post-pass from 14 r_s on B2 and on s8 (K = 18), and from 40 r_s on s8;
  - the B2 record json (the formation record and the trapped episodes), the sigma/a 8 record json, the version 3 summary (its marks block) and the
    run (a) results (ruling2/runA_v1_results.jsonl).
It writes OUT/page_v4_data_v1.json (every number the tables and the figure print), OUT/page_v4_data_v1.npz (the arrays) and OUT/MANIFEST.md5
(every input file by its resolved path with md5, and the outputs). The table script and the figure script load these and compute nothing new.

Conventions: x_kappa is the unweighted excess S_X(C) - S_X(P_-(h)) of one channel at one clock and cut (the d13 'excess'); the weight is 16 kappa;
the sum S = sum_k 16 k x_k; F_meas per channel is the post-pass's f_meas (2 m h2((T_trunc + T_meas)/m) + round), weighted and summed; c is the largest
|16 kappa x_kappa| over kappa 12 to K at the pair; the share is K c (the post-pass's 'share'); the verdicts are the post-pass's (readout_v1f), copied.
v1b (version 4.1): run (a) rows parsed (the file's lines carry a RESULT prefix); per pair the geometric criterion of W1 ruling v2 section 2(b) on the
stored probe terms with its conditional majorant; the common-mode spread over kappa 12 to 18; the first-order term d = (E_out - flow)/w against E_kappa per
channel; the flight marks at the 2 and 30 r_s radii; version 3's A-to-100 rows at full precision; the source modes' vacuum occupations and average transfers.
Usage: python page_v4_data_v1b.py OUT_DIR"""
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

CODE_D13 = "<internal>/code_d13"
CODE_V6 = "<internal>/code_v6"
PP_DIR = "<internal>/w1"
os.environ.setdefault("W1_CODE_D13", CODE_D13)
os.environ.setdefault("W1_CODE_V6", CODE_V6)
for _p in (CODE_D13, CODE_V6, PP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import page_w1_sector_v1d as W1S  # noqa: E402
import w1_postpass_v1f as PP  # noqa: E402

BULK = "<internal>/page_v4"
B2_PROD = ("<internal>/prod", "v6o_B2")
B2_CTRL = ("<internal>/v6n_B2_pod7_k1to8", "v6n_B2")
PROBES = {24: ("<internal>/v6o_B2r2b_k24", "v6o_B2r2b_k24"),
          32: ("<internal>/v6o_B2r2b_k32", "v6o_B2r2b_k32"),
          48: ("<internal>/v6o_B2r2b_k48", "v6o_B2r2b_k48")}
B2_SUB16 = ("<internal>/v6o_B2r2c_sub16", "v6o_B2r2c_sub16", 18)
S8_PROD = (BULK + "/s8_prod_K18", "v6o_s8")
S8_SUB16 = ("<internal>/v6o_s8r2c_sub16", "v6o_s8r2c_sub16", 14)
W3 = {"held-sources (W3fs)": {1: ("<internal>/v6o_W3fs_k1to1", "v6o_W3fs_k1to1"), 2: ("<internal>/v6o_W3fs_k2to2", "v6o_W3fs_k2to2")},
      "paired-avg (W3pa)": {1: ("<internal>/v6o_W3pa_k1to1", "v6o_W3pa_k1to1"), 2: ("<internal>/v6o_W3pa_k2to2", "v6o_W3pa_k2to2")}}
POSTPASS = {"B2_from40": "<internal>/W1_B2_composed_v3.json",
            "B2_from14": BULK + "/postpass/W1_B2_v1f_K18_from14.json",
            "s8_from40": BULK + "/postpass/W1_s8_v1f_K18_from40.json",
            "s8_from14": BULK + "/postpass/W1_s8_v1f_K18_from14.json"}
RECORD_B2 = "<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096.json"
FIELDS_B2 = "<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096_fields.npz"
RECORD_S8 = "<internal>/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json"
FIELDS_S8 = "<internal>/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg_fields.npz"
V3_SUMMARY = BULK + "/in_v3_summary.json"
RUNA = "<internal>/runA_v1_results.jsonl"
CODE_FILES = [CODE_D13 + "/page_history_v6o.py", CODE_D13 + "/page_w1_sector_v1d.py", CODE_D13 + "/page_d13_readout_v1d.py", PP_DIR + "/w1_postpass_v1f.py",
              "<internal>/job_v6o_r2.sh", "<internal>/runA_v1.py"]

CUTS = ["2", "16", "24", "30", "48"]
CUT_R_ABS = {"2": 128.0, "16": 1024.0, "24": 1536.0, "30": 1920.0, "48": 3072.0}
RS0 = 64.0                                   # r_s,0 in lattice units (r_s/a 64)
G_RUN = 0.25
T_D_OVER_RS = 2.31                           # 06_scaling.tex line 338: 2.31 r_s at q = 1 and r_s/a = 64 (the leading T_d term; the matching constant unfixed)
T_D_SOURCE = "<internal>/06_scaling.tex:338 (2.31 r_s at q = 1 and r_s/a = 64; T_d = (2/3) r_s ln[1/(2 chi q)] + O(r_s))"
NAN = float("nan")
ROWKEYS = ["E", "two_gE", "T_H", "kappa_T", "R_t_record", "min_N_minus_absv", "join_max_dN", "join_max_rel_dv", "t_inf_collar",
           "E_inv_weighted", "E_out_weighted", "collar_flow_acc_weighted", "L_interval", "L_H_kappaT", "L_H_TH", "W_hdot_weighted", "W_acc_interval_weighted",
           "jump_check_weighted", "W_creation", "dE_inv_dt", "band"]
MANIFEST = []


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def note_input(label, path):
    rp = os.path.realpath(path)
    MANIFEST.append({"label": label, "path": path, "realpath": rp, "md5": md5(rp) if os.path.exists(rp) else None, "bytes": os.path.getsize(rp) if os.path.exists(rp) else None})


def fnum(x):
    return NAN if x is None else float(x)


def load(out_dir, tag, k, label):
    ch = PP.load_channel(out_dir, tag, k)
    if ch is None:
        raise SystemExit("missing channel %s/%s kappa %d" % (out_dir, tag, k))
    for suf in ("summary.json", "d13.json"):
        note_input("%s kappa %d %s" % (label, k, suf), os.path.join(out_dir, "%s_k%d_%s" % (tag, k, suf)))
    return ch


def channel_arrays(ch, k, clocks):
    """per clock and cut: x (the unweighted excess), S_X, S_X_vac, f_meas (unweighted), floor_high, m_pairs, resolved; per clock: the row keys (weighted
    where the key says so) and the channel's own E_inv, E_out, collar_flow_acc (unweighted)."""
    n = len(clocks)
    A = {key: np.full((n, len(CUTS)), NAN) for key in ("x", "S_X", "S_X_vac", "fmeas", "floor_high", "floor_low", "m_pairs", "t_X")}
    R = {key: np.full(n, NAN) for key in ROWKEYS + ["E_inv_ch", "E_out_ch", "flow_ch", "W_hdot_ch", "T_meas", "T_cheb", "T_trunc", "n_substeps", "cheb_K_max"]}
    for i, t in enumerate(clocks):
        cl = ch["clocks"].get(t) or {}
        cuts = cl.get("cuts") or {}
        for j, lab in enumerate(CUTS):
            c = cuts.get(lab) or {}
            A["x"][i, j] = fnum(c.get("excess")); A["S_X"][i, j] = fnum(c.get("S_X")); A["S_X_vac"][i, j] = fnum(c.get("S_X_vac"))
            A["floor_high"][i, j] = fnum(c.get("excess_floor_high")); A["floor_low"][i, j] = fnum(c.get("excess_floor_low"))
            A["m_pairs"][i, j] = fnum(c.get("m_pairs")); A["t_X"][i, j] = fnum(c.get("t_X"))
            fm = PP.f_meas(ch, t, lab) if c else None
            A["fmeas"][i, j] = fnum(fm)
        r = ch["rows"].get(t) or {}
        for key in ROWKEYS:
            R[key][i] = fnum(r.get(key))
        pc = ((r.get("per_channel") or {}).get(str(k))) or {}
        R["E_inv_ch"][i] = fnum(pc.get("E_inv")); R["E_out_ch"][i] = fnum(pc.get("E_out")); R["flow_ch"][i] = fnum(pc.get("collar_flow_acc")); R["W_hdot_ch"][i] = fnum(pc.get("W_hdot"))
        fl = ch["floors"].get(t) or {}
        R["T_meas"][i] = fnum(fl.get("T_meas")); R["T_cheb"][i] = fnum(fl.get("T_cheb")); R["T_trunc"][i] = fnum(fl.get("T_trunc")); R["n_substeps"][i] = fnum(fl.get("n_substeps")); R["cheb_K_max"][i] = fnum(fl.get("cheb_K_max"))
    return A, R


def stack(chs, ks, clocks):
    """the channel arrays of a contiguous set stacked over kappa: X[k, i, j] etc., W[k] = 16 kappa."""
    out = {}
    per = {k: channel_arrays(chs[k], k, clocks) for k in ks}
    for key in ("x", "S_X", "S_X_vac", "fmeas", "floor_high", "floor_low", "m_pairs", "t_X"):
        out[key] = np.stack([per[k][0][key] for k in ks])
    for key in per[ks[0]][1]:
        out["row_" + key] = np.stack([per[k][1][key] for k in ks])
    out["W"] = np.array([16.0 * k for k in ks]); out["kappas"] = np.array(ks)
    return out


def sums_K(S):
    """the weighted sums and their errors at every clock and cut for a stacked set (K = the last channel)."""
    W = S["W"][:, None, None]
    WX = W * S["x"]
    SUM = np.nansum(WX, axis=0); SUM[np.isnan(S["x"]).any(axis=0)] = NAN
    FMW = np.nansum(W * S["fmeas"], axis=0); FMW[np.isnan(S["fmeas"]).any(axis=0)] = NAN
    FAW = np.nansum(W * S["floor_high"], axis=0); FAW[np.isnan(S["floor_high"]).any(axis=0)] = NAN
    ks = S["kappas"]; K = int(ks[-1])
    sel = ks >= 12 if K >= 15 else ks >= ks[-min(3, len(ks))]
    C = np.nanmax(np.abs(WX[sel]), axis=0)
    SHARE = len(ks) * C
    partial = np.cumsum(WX, axis=0)
    resolved = np.abs(S["x"]) > S["fmeas"]
    return {"WX": WX, "SUM": SUM, "FMW": FMW, "FAW": FAW, "C": C, "SHARE": SHARE, "partial": partial, "resolved": resolved, "c_kappas": [int(k) for k in ks[sel]]}


def readout_rows(path):
    d = json.load(open(path))
    rows = {}
    for r in d.get("readout_v1f") or []:
        rows[(round(float(r["t"]), 4), str(r["cut"]))] = r
    return d, rows


def light_travel_a(r_sf, R_abs, c):
    """the outgoing light travel from c x r_s,f to R_abs in the vacuum Painleve-Gullstrand exterior at the formation record's charge: 2 r_s,f [F(x1) - F(x0)],
    F(x) = x^2/2 + x + ln(x - 1), x = (R/r_s,f)^(1/2) (the version 3 marks' formula), in lattice units."""
    F = lambda x: x * x / 2.0 + x + math.log(x - 1.0)
    x0 = math.sqrt(c); x1 = math.sqrt(R_abs / r_sf)
    return 2.0 * r_sf * (F(x1) - F(x0)), x1


def episodes(recs):
    ep = []; cur = None
    for r in recs:
        has = r.get("R_t") not in (None, [], 0, 0.0)
        if has and cur is None:
            cur = {"T_start": r["T_over_rs"], "T_end": r["T_over_rs"], "R_t_first": r["R_t"], "R_t_max": r["R_t"], "R_t_min": r["R_t"], "n_records": 1}
        elif has:
            cur["T_end"] = r["T_over_rs"]; cur["R_t_max"] = max(cur["R_t_max"], r["R_t"]); cur["R_t_min"] = min(cur["R_t_min"], r["R_t"]); cur["n_records"] += 1
        elif cur is not None:
            ep.append(cur); cur = None
    if cur is not None:
        ep.append(cur)
    return ep


def tolist(a):
    if isinstance(a, np.ndarray):
        return [tolist(x) for x in a.tolist()] if a.ndim > 1 else [None if (isinstance(x, float) and math.isnan(x)) else x for x in a.tolist()]
    if isinstance(a, float) and math.isnan(a):
        return None
    if isinstance(a, (np.floating, np.integer)):
        v = a.item(); return None if (isinstance(v, float) and math.isnan(v)) else v
    if isinstance(a, dict):
        return {str(k): tolist(v) for k, v in a.items()}
    if isinstance(a, (list, tuple)):
        return [tolist(v) for v in a]
    return a


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else BULK + "/out"
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()
    checks = {}
    # ---------------------------------------------------------------- B2, kappa 1 to 18
    ks = list(range(1, 19))
    B2 = {k: load(B2_PROD[0], B2_PROD[1], k, "B2 production") for k in ks}
    assert all(B2[k]["final"] for k in ks)
    clocks = sorted(set.intersection(*[set(B2[k]["rows"]) for k in ks]))
    SB2 = stack(B2, ks, clocks); KB2 = sums_K(SB2)
    # settings of the production (kappa 1's summary) for the SOURCES
    S1 = json.load(open(os.path.join(B2_PROD[0], "%s_k1_summary.json" % B2_PROD[1])))
    settings = {key: (S1.get("settings") or {}).get(key) for key in ("record", "fields", "fields_key", "nsites", "T_end_rs", "R_cut_rs", "R_collar_rs", "join_rs", "cheb_tol", "ext_every_rs", "window_rs", "extra_ck_rs", "collect_every_rs", "eta", "bin_TH", "soft_max_TH", "transfer_modes")}
    settings["w1"] = {key: (S1.get("w1") or {}).get(key) for key in ("propagation", "interp", "interp_sub", "sector_used")}
    settings["code"] = S1.get("code"); settings["label_run"] = S1.get("label")
    # the post-pass readouts (verdicts): the busi composition from 40 r_s and this lane's rerun from 14 r_s
    PPB40, R40 = readout_rows(POSTPASS["B2_from40"]); PPB14, R14 = readout_rows(POSTPASS["B2_from14"])
    for key in POSTPASS:
        note_input("post-pass " + key, POSTPASS[key])
    read_B2 = {}
    diff_40_14 = []
    for (t, lab), r in R40.items():
        read_B2[(t, lab)] = dict(r, source="B2_from40")
        q = R14.get((t, lab))
        if q is not None and (q["verdict"] != r["verdict"] or abs(q["S"] - r["S"]) > 1e-9 * max(1.0, abs(r["S"]))):
            diff_40_14.append({"t": t, "cut": lab, "verdict_from40": r["verdict"], "verdict_from14": q["verdict"], "S_from40": r["S"], "S_from14": q["S"], "why_from14": q.get("why")})
    for (t, lab), r in R14.items():
        if (t, lab) not in read_B2:
            read_B2[(t, lab)] = dict(r, source="B2_from14")
    checks["B2_verdict_differences_from40_vs_from14_at_t_ge_40"] = diff_40_14
    # my sums against the post-pass rows
    dS = []; dF = []; dC = []; dA = []
    for (t, lab), r in read_B2.items():
        i = clocks.index(t); j = CUTS.index(lab)
        dS.append(abs(KB2["SUM"][i, j] - r["S"])); dF.append(abs(KB2["FMW"][i, j] - r["F_meas"])); dC.append(abs(KB2["C"][i, j] - r["c"])); dA.append(abs(KB2["FAW"][i, j] - r["F_apriori"]))
    checks["B2_sum_vs_postpass_max_abs"] = {"S": max(dS), "F_meas": max(dF), "c": max(dC), "F_apriori": max(dA), "n_pairs": len(dS)}
    # L identity: L_interval(t) = (acc(t) - acc(t_prev)) / (Delta T) per channel (weighted accumulator), Delta T in lattice units
    worst = 0.0
    for a in range(len(ks)):
        acc = SB2["row_collar_flow_acc_weighted"][a]; L = SB2["row_L_interval"][a]
        for i in range(1, len(clocks)):
            dT = (clocks[i] - clocks[i - 1]) * RS0
            Lc = (acc[i] - acc[i - 1]) / dT
            if np.isfinite(L[i]) and abs(L[i]) > 0:
                worst = max(worst, abs(Lc - L[i]) / abs(L[i]))
    checks["L_interval_identity_max_rel"] = worst
    # ---------------------------------------------------------------- the probes kappa 24, 32, 48
    probes = {}
    for K, (d_, t_) in PROBES.items():
        ch = load(d_, t_, K, "probe"); A, R = channel_arrays(ch, K, clocks)
        probes[K] = {"x": A["x"], "fmeas": A["fmeas"], "W": 16.0 * K, "E_kappa_w": R["E_inv_weighted"], "E_out_w": R["E_out_weighted"], "flow_w": R["collar_flow_acc_weighted"], "L": R["L_interval"], "final": ch["final"], "wall_s": ch["wall_s"], "cheb_K_sum": ch["cheb_K_sum"]}
    n = len(clocks)
    x18 = SB2["x"][17] * 288.0
    pr = {"kappas": [18, 24, 32, 48], "WX": np.stack([x18] + [probes[K]["x"] * probes[K]["W"] for K in (24, 32, 48)])}
    with np.errstate(divide="ignore", invalid="ignore"):
        pr["r_a"] = (np.abs(pr["WX"][2]) / np.abs(pr["WX"][1])) ** (1.0 / 8.0)
        pr["r_b"] = (np.abs(pr["WX"][3]) / np.abs(pr["WX"][2])) ** (1.0 / 16.0)
        pr["ratio_24_18"] = pr["WX"][1] / pr["WX"][0]; pr["ratio_32_18"] = pr["WX"][2] / pr["WX"][0]; pr["ratio_48_18"] = pr["WX"][3] / pr["WX"][0]
    sg = np.sign(pr["WX"]); pr["same_sign"] = np.all(sg == sg[0:1], axis=0)
    slope = np.full((n, len(CUTS)), NAN)
    lk = np.log(np.array([18.0, 24.0, 32.0, 48.0]))
    for i in range(n):
        for j in range(len(CUTS)):
            y = np.abs(pr["WX"][:, i, j])
            if pr["same_sign"][i, j] and np.all(y > 0):
                ly = np.log(y); slope[i, j] = np.polyfit(lk, ly, 1)[0]
    pr["slope_log_abs_WX_vs_log_kappa"] = slope
    pr["E_kappa_w"] = {K: probes[K]["E_kappa_w"] for K in (24, 32, 48)}
    pr["resolved"] = np.stack([np.abs(probes[K]["x"]) > probes[K]["fmeas"] for K in (24, 32, 48)])
    pr["fmeas_w"] = np.stack([probes[K]["fmeas"] * probes[K]["W"] for K in (24, 32, 48)])
    # the geometric criterion of W1 ruling v2 section 2(b) on the stored terms at every clock and cut: the four terms (18, 24, 32, 48) share a sign and
    # both per-step ratios are below 1; its conditional majorant |288 x_18| q/(1 - q) with q the larger ratio (never applied: printed as the licensed
    # conditional remainder beside the OPEN status)
    with np.errstate(divide="ignore", invalid="ignore"):
        q_big = np.maximum(pr["r_a"], pr["r_b"])
        met = pr["same_sign"] & (np.abs(pr["WX"][1]) < np.abs(pr["WX"][0])) & (pr["r_a"] < 1.0) & (pr["r_b"] < 1.0)   # the terms fall from kappa 18 through 24, 32, 48 with one sign and both per-step ratios below 1
        cond_tail = np.where(met, np.abs(x18) * q_big / (1.0 - q_big), NAN)
        cond_tail_over_S = cond_tail / np.abs(KB2["SUM"])
    pr["criterion_met"] = met; pr["q_larger"] = q_big; pr["conditional_majorant"] = cond_tail; pr["conditional_majorant_over_S"] = cond_tail_over_S
    pr["selected_5e-4"] = np.abs(x18) > 5e-4
    # the common mode's spread over kappa 12 to 18 at every pair: max/min of |16 kappa x_kappa|
    with np.errstate(divide="ignore", invalid="ignore"):
        hi = np.abs(KB2["WX"][11:]); cm_spread = np.nanmax(hi, axis=0) / np.nanmin(hi, axis=0)
    # the first-order term d = (E_out - flow)/w per channel (unweighted) against E_kappa = E_inv/w at every clock
    with np.errstate(divide="ignore", invalid="ignore"):
        d_term = (SB2["row_E_out_weighted"] - SB2["row_collar_flow_acc_weighted"]) / SB2["W"][:, None]
        E_kappa_unw = SB2["row_E_inv_weighted"] / SB2["W"][:, None]
    # ---------------------------------------------------------------- the 16-sub-step control of kappa 18 (B2)
    ch16 = PP.load_channel(B2_SUB16[0], B2_SUB16[1], B2_SUB16[2])
    for suf in ("summary.json", "d13.json"):
        note_input("B2 sub16 kappa 18 " + suf, os.path.join(B2_SUB16[0], "%s_k%d_%s" % (B2_SUB16[1], B2_SUB16[2], suf)))
    A16, R16 = channel_arrays(ch16, 18, clocks)
    sub16_B2 = {"final": bool(ch16["final"]), "clocks_present": [t for t in clocks if t in ch16["rows"]], "last_clock": max(ch16["rows"]) if ch16["rows"] else None,
                "x_sub16": A16["x"], "x_sub8": SB2["x"][17], "D16_w": 288.0 * (A16["x"] - SB2["x"][17]), "WX18": x18,
                "E_kappa_w_sub16": R16["E_inv_weighted"], "E_kappa_w_sub8": SB2["row_E_inv_weighted"][17],
                "flow_w_sub16": R16["collar_flow_acc_weighted"], "flow_w_sub8": SB2["row_collar_flow_acc_weighted"][17], "n_substeps": R16["n_substeps"], "wall_s": ch16["wall_s"]}
    # ---------------------------------------------------------------- s8, kappa 1 to 18 (the second averaging scale)
    S8 = {k: load(S8_PROD[0], S8_PROD[1], k, "s8 production") for k in ks}
    assert all(S8[k]["final"] for k in ks)
    clocks8 = sorted(set.intersection(*[set(S8[k]["rows"]) for k in ks]))
    SS8 = stack(S8, ks, clocks8); KS8 = sums_K(SS8)
    PPS40, R8_40 = readout_rows(POSTPASS["s8_from40"]); PPS14, R8_14 = readout_rows(POSTPASS["s8_from14"])
    read_s8 = {}
    for (t, lab), r in R8_40.items():
        read_s8[(t, lab)] = dict(r, source="s8_from40")
    for (t, lab), r in R8_14.items():
        if (t, lab) not in read_s8:
            read_s8[(t, lab)] = dict(r, source="s8_from14")
    d8 = [{"t": t, "cut": lab, "verdict_from40": r["verdict"], "verdict_from14": R8_14[(t, lab)]["verdict"]} for (t, lab), r in R8_40.items() if (t, lab) in R8_14 and R8_14[(t, lab)]["verdict"] != r["verdict"]]
    checks["s8_verdict_differences_from40_vs_from14_at_40"] = d8
    dS8 = max(abs(KS8["SUM"][clocks8.index(t), CUTS.index(lab)] - r["S"]) for (t, lab), r in read_s8.items())
    checks["s8_sum_vs_postpass_max_abs_S"] = dS8
    ch814 = PP.load_channel(S8_SUB16[0], S8_SUB16[1], S8_SUB16[2])
    for suf in ("summary.json", "d13.json"):
        note_input("s8 sub16 kappa 14 " + suf, os.path.join(S8_SUB16[0], "%s_k%d_%s" % (S8_SUB16[1], S8_SUB16[2], suf)))
    A814, R814 = channel_arrays(ch814, 14, clocks8)
    sub16_s8 = {"final": bool(ch814["final"]), "x_sub16": A814["x"], "x_sub8": SS8["x"][13], "D16_w": 224.0 * (A814["x"] - SS8["x"][13]), "WX14": 224.0 * SS8["x"][13],
                "E_kappa_w_sub16": R814["E_inv_weighted"], "E_kappa_w_sub8": SS8["row_E_inv_weighted"][13], "flow_w_sub16": R814["collar_flow_acc_weighted"], "flow_w_sub8": SS8["row_collar_flow_acc_weighted"][13], "wall_s": ch814["wall_s"]}
    # ---------------------------------------------------------------- W3's completions, kappa 1 and 2 (to 40 r_s), beside B2's kappa 1 and 2
    comp = {}
    clocks40 = [t for t in clocks if t <= 40.0 + 1e-9]
    for name, chs in [("B2 (the record's own continuation, engine_collapse_v19, 4096 sites)", {1: B2[1], 2: B2[2]})] + [(nm, {k: load(d_, t_, k, "W3 " + nm) for k, (d_, t_) in W3[nm].items()}) for nm in W3]:
        Sx = stack(chs, [1, 2], clocks40); Kx = sums_K(Sx)
        hist = (json.load(open(os.path.join(W3[name][1][0], "%s_k1_summary.json" % W3[name][1][1]))).get("history") if name in W3 else B2[1]["rows"] and json.load(open(os.path.join(B2_PROD[0], "%s_k1_summary.json" % B2_PROD[1]))).get("history")) or {}
        comp[name] = {"SUM12": Kx["SUM"], "FMW12": Kx["FMW"], "WX": Kx["WX"], "E_inv12": Sx["row_E_inv_weighted"].sum(axis=0), "E_out12": Sx["row_E_out_weighted"].sum(axis=0),
                      "flow12": Sx["row_collar_flow_acc_weighted"].sum(axis=0), "L12": Sx["row_L_interval"].sum(axis=0), "LH12_kT": Sx["row_L_H_kappaT"].sum(axis=0), "LH12_TH": Sx["row_L_H_TH"].sum(axis=0),
                      "E": Sx["row_E"][0], "two_gE": Sx["row_two_gE"][0], "R_t": Sx["row_R_t_record"][0], "minNv": Sx["row_min_N_minus_absv"][0], "join_dN": Sx["row_join_max_dN"][0], "join_dv": Sx["row_join_max_rel_dv"][0],
                      "history_label": (hist.get("label") or "")[-900:], "history_sites": hist.get("engine_sites"), "history_records": hist.get("records")}
    names = list(comp)
    spread = {}
    for key in ("SUM12", "E_inv12", "flow12", "E"):
        arr = np.stack([comp[nm][key] for nm in names])
        spread[key + "_max_minus_min_all3"] = np.nanmax(arr, axis=0) - np.nanmin(arr, axis=0)
        arr2 = np.stack([comp[nm][key] for nm in names[1:]])
        spread[key + "_max_minus_min_two_completions"] = np.nanmax(arr2, axis=0) - np.nanmin(arr2, axis=0)
    # ---------------------------------------------------------------- the flux (kappa 1, 2; with kappa 3 the guard; the K = 18 sum, labelled)
    L = SB2["row_L_interval"]; LHk = SB2["row_L_H_kappaT"]; LHt = SB2["row_L_H_TH"]
    with np.errstate(divide="ignore", invalid="ignore"):
        flux = {"clocks": clocks, "t_inf_collar_over_rs": SB2["row_t_inf_collar"][0] / RS0, "E": SB2["row_E"][0], "two_gE": SB2["row_two_gE"][0], "T_H": SB2["row_T_H"][0], "kappa_T": SB2["row_kappa_T"][0],
                "R_t": SB2["row_R_t_record"][0], "minNv": SB2["row_min_N_minus_absv"][0], "join_dN": SB2["row_join_max_dN"][0], "join_dv": SB2["row_join_max_rel_dv"][0],
                "L12": L[0] + L[1], "LH12_kT": LHk[0] + LHk[1], "LH12_TH": LHt[0] + LHt[1], "L123": L[:3].sum(axis=0), "LH123_kT": LHk[:3].sum(axis=0), "LH123_TH": LHt[:3].sum(axis=0),
                "L18": L.sum(axis=0), "LH18_kT": LHk.sum(axis=0), "LH18_TH": LHt.sum(axis=0), "L_per_channel": L, "LH_kT_per_channel": LHk,
                "E_inv12": SB2["row_E_inv_weighted"][:2].sum(axis=0), "E_inv123": SB2["row_E_inv_weighted"][:3].sum(axis=0), "E_inv18": SB2["row_E_inv_weighted"].sum(axis=0),
                "E_inv_per_channel": SB2["row_E_inv_weighted"], "E_out_per_channel": SB2["row_E_out_weighted"], "flow_per_channel": SB2["row_collar_flow_acc_weighted"],
                "flow12": SB2["row_collar_flow_acc_weighted"][:2].sum(axis=0), "flow123": SB2["row_collar_flow_acc_weighted"][:3].sum(axis=0), "flow18": SB2["row_collar_flow_acc_weighted"].sum(axis=0),
                "W_hdot12": SB2["row_W_hdot_weighted"][:2].sum(axis=0), "W_hdot18": SB2["row_W_hdot_weighted"].sum(axis=0),
                "E_out_over_E_kappa": SB2["row_E_out_ch"] / SB2["row_E_inv_ch"]}
        flux["ratio12_kT"] = flux["L12"] / flux["LH12_kT"]; flux["ratio12_TH"] = flux["L12"] / flux["LH12_TH"]
        flux["ratio123_kT"] = flux["L123"] / flux["LH123_kT"]; flux["ratio123_TH"] = flux["L123"] / flux["LH123_TH"]
        flux["ratio18_kT"] = flux["L18"] / flux["LH18_kT"]; flux["ratio18_TH"] = flux["L18"] / flux["LH18_TH"]
        flux["E_rad0_over_E_12"] = flux["flow12"] / flux["E"]; flux["E_rad0_over_E_18"] = flux["flow18"] / flux["E"]
        flux["E_inv12_over_E"] = flux["E_inv12"] / flux["E"]; flux["E_inv18_over_E"] = flux["E_inv18"] / flux["E"]
    # the levels A/(4G) and eta A/a^2 at the record's instantaneous 2gE (the rows' 'levels' block) and the K = 18 sums' ratios to them
    lev_A = np.array([fnum(((B2[1]["rows"].get(t) or {}).get("levels") or {}).get("A_over_4G")) for t in clocks])
    lev_eta = np.array([fnum(((B2[1]["rows"].get(t) or {}).get("levels") or {}).get("eta_A_over_a2")) for t in clocks])
    with np.errstate(divide="ignore", invalid="ignore"):
        levels = {"A_over_4G": lev_A, "eta_A_over_a2": lev_eta, "SUM_over_A_over_4G": KB2["SUM"] / lev_A[:, None], "SUM_over_eta_A_over_a2": KB2["SUM"] / lev_eta[:, None],
                  "definition": "the rows' levels block: A/(4G) = 4 pi (2gE)^2 / (4 G) at the record's instantaneous 2gE with G/a^2 = 1/4, and eta A/a^2 with eta = 0.9192 (the settings' eta), in nats"}
    ipk = np.unravel_index(np.nanargmax(np.abs(KB2["SUM"][[clocks.index(t) for t in clocks if t >= 40.0 - 1e-9]])), KB2["SUM"][[clocks.index(t) for t in clocks if t >= 40.0 - 1e-9]].shape)
    tpk = [t for t in clocks if t >= 40.0 - 1e-9][ipk[0]]; jpk = int(ipk[1])
    levels["largest_abs_sum_from_40"] = {"t": tpk, "cut": CUTS[jpk], "S": float(KB2["SUM"][clocks.index(tpk), jpk]), "A_over_4G": float(lev_A[clocks.index(tpk)]), "eta_A_over_a2": float(lev_eta[clocks.index(tpk)]),
                                         "S_over_A_over_4G": float(KB2["SUM"][clocks.index(tpk), jpk] / lev_A[clocks.index(tpk)]), "S_over_eta_A_over_a2": float(KB2["SUM"][clocks.index(tpk), jpk] / lev_eta[clocks.index(tpk)])}
    # the signed collar split per clock (kappa 1 and 2, weights 16 and 32): the collar_split block of the d13 clocks
    split = {"clocks": [], "E_flux_out_12": [], "E_flux_in_12": [], "E_flux_out_soft_12": [], "E_flux_in_soft_12": [], "E_flux_out_hard_12": [], "E_flux_in_hard_12": [], "E_flux_out_k1": [], "E_flux_in_k1": [], "E_flux_out_k2": [], "E_flux_in_k2": [], "definition": None}
    for t in clocks:
        c1 = ((B2[1]["clocks"].get(t) or {}).get("collar_split")) or {}; c2 = ((B2[2]["clocks"].get(t) or {}).get("collar_split")) or {}
        if not c1 or not c2:
            continue
        split["clocks"].append(t)
        for f in ("E_flux_out", "E_flux_in", "E_flux_out_soft", "E_flux_in_soft", "E_flux_out_hard", "E_flux_in_hard"):
            split[f + "_12"].append(16.0 * fnum(c1.get(f)) + 32.0 * fnum(c2.get(f)))
        split["E_flux_out_k1"].append(16.0 * fnum(c1.get("E_flux_out"))); split["E_flux_in_k1"].append(16.0 * fnum(c1.get("E_flux_in")))
        split["E_flux_out_k2"].append(32.0 * fnum(c2.get("E_flux_out"))); split["E_flux_in_k2"].append(32.0 * fnum(c2.get("E_flux_in")))
        if split["definition"] is None:
            split["definition"] = {k: c1.get(k) for k in ("L", "centre_site", "R_c", "omega_th", "sum_w2", "sites_valid")}
    # the half-flow clocks of the kappa 1, 2 accumulator (linear interpolation between checkpoints; the per-record flows are not stored)
    def half_clock(acc, tt, t_end):
        ie = tt.index(t_end); target = acc[ie] / 2.0
        for i in range(1, ie + 1):
            a0, a1 = acc[i - 1], acc[i]
            if (a0 - target) * (a1 - target) <= 0 and a1 != a0:
                return tt[i - 1] + (tt[i] - tt[i - 1]) * (target - a0) / (a1 - a0), tt[i - 1], tt[i], target
        return None, None, None, target
    hf100 = half_clock(list(flux["flow12"]), clocks, 100.0); hf40 = half_clock(list(flux["flow12"]), clocks, 40.0)
    offset = flux["t_inf_collar_over_rs"] - np.array(clocks)
    # the version 3 summary (the s4 run's flux through 40 under page_history_v6b, the piecewise-constant propagation) beside
    note_input("version 3 summary json", V3_SUMMARY)
    V3 = json.load(open(V3_SUMMARY)); F3 = V3["figure"]; mk3 = (F3.get("marks") or {}).get("v2") or {}
    v3flux = {"t_over_rs": F3["t_over_rs"], "t_inf_over_rs": F3["t_inf_over_rs"], "L_interval": F3["L_interval"], "L_over_LH_kappaT": F3["L_over_LH_kappaT"], "L_over_LH_TH": F3["L_over_LH_TH"], "marks_v2": mk3}
    A100 = V3.get("A_to100") or {}
    v3flux["A_to100"] = {"path": A100.get("path"), "md5": A100.get("md5"), "marker_rs": A100.get("marker_rs"), "label": A100.get("label"),
                         "rows": [{f: r.get(f) for f in ("t_over_rs", "E", "E_inv_weighted", "L_over_LH_kappaT", "L_over_LH_TH", "L_H_kappaT", "L_H_TH", "collar_flow_acc_weighted", "join_max_dN", "join_max_rel_dv", "after_first_return", "t_inf_collar")} for r in (A100.get("rows") or [])]}
    # ---------------------------------------------------------------- the marks
    note_input("B2 record json", RECORD_B2); note_input("B2 fields npz", FIELDS_B2); note_input("s8 record json", RECORD_S8); note_input("s8 fields npz", FIELDS_S8)
    RB = json.load(open(RECORD_B2)); recs = RB["records"]
    first = [r for r in recs if r.get("R_t") not in (None, [], 0, 0.0)][0]
    T_f = float(first["T_over_rs"]); E_f = float(first["E"]); two_gE_f = 2.0 * G_RUN * E_f
    marks = {"formation_record_clock_over_rs": T_f, "formation_R_t_a": float(first["R_t"]), "formation_E": E_f, "two_gE_formation": two_gE_f, "g": G_RUN,
             "episodes_B2": episodes(recs), "E0_B2": float(recs[0]["E"]), "n_records_B2": len(recs),
             "R_t_at": {str(t): {"R_t": r.get("R_t"), "R_t_over_2gE": r.get("R_t_over_2gE"), "E": r["E"], "E_over_E0": r.get("E_over_E0"), "min_N_minus_absv": r.get("min_N_minus_absv")} for t in (40.0, 56.0, 76.0, 100.0) for r in recs if abs(r["T_over_rs"] - t) < 1e-9},
             "T_d_over_rs": T_D_OVER_RS, "T_d_source": T_D_SOURCE, "light_travel": {}, "half_flow": {}, "collar_clock_offset_over_rs": {"min": float(np.nanmin(offset)), "max": float(np.nanmax(offset)), "at_40": float(offset[clocks.index(40.0)]), "at_100": float(offset[clocks.index(100.0)])},
             "v3_marks_v2": mk3}
    for lab, R_abs in (("2", 128.0), ("16", 1024.0), ("24", 1536.0), ("30", 1920.0), ("48", 3072.0)):
        ent = {}
        for c in (2.0, 1.5):
            trav, x1 = light_travel_a(two_gE_f, R_abs, c)
            arrT = T_f + trav / RS0
            i_near = int(np.argmin(np.abs(np.array(clocks) - arrT)))
            ent[str(c)] = {"travel_over_2gE_f": trav / two_gE_f, "travel_over_rs0": trav / RS0, "arrival_record_clock_over_rs": arrT, "arrival_collar_clock_over_rs": arrT + float(offset[i_near]), "x1": x1}
        ent["ray_map_end_record_clock_over_rs"] = {c: ent[c]["arrival_record_clock_over_rs"] + T_D_OVER_RS for c in ("2.0", "1.5")}
        marks["light_travel"][lab] = ent
    checks["light_travel_vs_v3"] = {"this_16_c2": marks["light_travel"]["16"]["2.0"]["travel_over_2gE_f"], "v3_16_c2": (mk3.get("light_travel_over_2gE") or {}).get("2.0"),
                                    "this_16_c1.5": marks["light_travel"]["16"]["1.5"]["travel_over_2gE_f"], "v3_16_c1.5": (mk3.get("light_travel_over_2gE") or {}).get("1.5"), "v3_two_gE_formation": mk3.get("two_gE_formation"), "this_two_gE_formation": two_gE_f}
    marks["half_flow"] = {"through_100_kappa12": {"record_clock_over_rs": hf100[0], "between": [hf100[1], hf100[2]], "half_of_acc": hf100[3], "acc_at_100": float(flux["flow12"][clocks.index(100.0)])},
                          "through_40_kappa12": {"record_clock_over_rs": hf40[0], "between": [hf40[1], hf40[2]], "half_of_acc": hf40[3], "acc_at_40": float(flux["flow12"][clocks.index(40.0)])},
                          "v3_through_40_s4_run": {"record_clock_over_rs": mk3.get("arrival_half_flow_t_over_rs"), "collar_clock_over_rs": mk3.get("arrival_half_flow_t_inf_over_rs"), "collar_flow_total": mk3.get("collar_flow_total")},
                          "definition": "the clock at which the kappa 1, 2 accumulated signed collar flow reaches half of its value at the end clock, by linear interpolation between the stored checkpoints (2 r_s apart); the per-record flows are not stored in the v6o rows, so the clock is bracketed by the two checkpoints named"}
    RS8 = json.load(open(RECORD_S8)); recs8 = RS8["records"]; first8 = [r for r in recs8 if r.get("R_t") not in (None, [], 0, 0.0)][0]
    marks["s8"] = {"formation_record_clock_over_rs": float(first8["T_over_rs"]), "formation_R_t_a": float(first8["R_t"]), "formation_E": float(first8["E"]), "E0": float(recs8[0]["E"]), "n_records": len(recs8), "episodes": episodes(recs8), "last_T_over_rs": float(recs8[-1]["T_over_rs"])}
    # ---------------------------------------------------------------- the piecewise-constant control (v6n, kappa 1 to 8): the record-interpolation spread
    interp = {"clocks": [], "channels": None, "error_prod_minus_ctrl": [], "excess_prod_1to8": [], "excess_ctrl_1to8": [], "E_quench": [], "E_prod_1to8": [], "E_ctrl_1to8": [], "flow_error": []}
    for t in clocks:
        w = (PPB14.get("weighted") or {}).get(str(t)) or {}
        if not w.get("channels"):
            continue
        interp["clocks"].append(t); interp["channels"] = w["channels"]
        interp["error_prod_minus_ctrl"].append([((w.get("cuts") or {}).get(lab) or {}).get("error") for lab in CUTS])
        interp["excess_prod_1to8"].append([((w.get("cuts") or {}).get(lab) or {}).get("excess_prod") for lab in CUTS])
        interp["excess_ctrl_1to8"].append([((w.get("cuts") or {}).get(lab) or {}).get("excess_ctrl") for lab in CUTS])
        interp["E_quench"].append(w.get("E_quench")); interp["E_prod_1to8"].append(w.get("E_prod")); interp["E_ctrl_1to8"].append(w.get("E_ctrl")); interp["flow_error"].append(w.get("flow_error"))
    for k in range(1, 9):
        for suf in ("summary.json", "d13.json"):
            note_input("B2 piecewise-constant control (v6n) kappa %d %s" % (k, suf), os.path.join(B2_CTRL[0], "%s_k%d_%s" % (B2_CTRL[1], k, suf)))
    interp["quench_per_channel"] = PPB14.get("quench_excess_E_ctrl_minus_E_prod")
    # ---------------------------------------------------------------- the D13 blocks of kappa 1 and 2 (weighted sums where additive)
    d13 = {"windows": [], "between": [], "hard": [], "probes": [], "sources": {}, "first_return_at_100": None, "first_return_at_40": None}
    Dj = {k: json.load(open(os.path.join(B2_PROD[0], "%s_k%d_d13.json" % (B2_PROD[1], k)))) for k in (1, 2, 3)}
    wk = {1: 16.0, 2: 32.0, 3: 48.0}
    def wsum(key, items):
        vals = [(wk[k], (it.get(key) if it else None)) for k, it in items]
        return None if any(v is None for _, v in vals) else float(sum(w * v for w, v in vals))
    W1 = {(w["t0_over_rs"], w["t1_over_rs"], w["cut"]): w for w in Dj[1]["windows"]["1"]}; W2 = {(w["t0_over_rs"], w["t1_over_rs"], w["cut"]): w for w in Dj[2]["windows"]["2"]}; W3w = {(w["t0_over_rs"], w["t1_over_rs"], w["cut"]): w for w in Dj[3]["windows"]["3"]}
    for key in sorted(W1):
        w1, w2, w3 = W1[key], W2.get(key), W3w.get(key)
        ent = {"t0": key[0], "t1": key[1], "cut": key[2], "window_rs": w1.get("window_rs"), "kind": w1.get("kind"), "u_start": w1.get("u_start_rs"), "u_end": w1.get("u_end_rs"), "n_records": w1.get("n_records")}
        if key[2] == "16-30":
            ent["inventory_D_between_12"] = wsum("inventory_D_between", [(1, w1), (2, w2)]); ent["inventory_D_between_123"] = wsum("inventory_D_between", [(1, w1), (2, w2), (3, w3)])
            d13["between"].append(ent)
        else:
            for f in ("inventory_D_start", "inventory_D_end", "second_term_D", "crossing_current", "crossing_jumps", "crossing_total", "vacuum_change", "change_C", "telescoping_residual"):
                ent[f + "_12"] = wsum(f, [(1, w1), (2, w2)])
            for f in ("inventory_D_start", "inventory_D_end", "second_term_D", "crossing_current"):
                ent[f + "_123"] = wsum(f, [(1, w1), (2, w2), (3, w3)])
            ent["second_term_D_k1"] = 16.0 * w1["second_term_D"]; ent["second_term_D_k2"] = 32.0 * w2["second_term_D"] if w2 else None
            d13["windows"].append(ent)
    H1 = {(h["t0_over_rs"], h["t1_over_rs"]): h for h in Dj[1]["hard_windows"]["1"]}; H2 = {(h["t0_over_rs"], h["t1_over_rs"]): h for h in Dj[2]["hard_windows"]["2"]}; H3 = {(h["t0_over_rs"], h["t1_over_rs"]): h for h in Dj[3]["hard_windows"]["3"]}
    for key in sorted(H1):
        h1, h2, h3 = H1[key], H2.get(key), H3.get(key)
        ent = {"t0": key[0], "t1": key[1], "window_rs": h1.get("window_rs"), "u_start": h1.get("u_start_rs"), "u_end": h1.get("u_end_rs")}
        for f in ("delta_hard_E_out", "delta_hard_E_in", "delta_soft_E_out", "delta_soft_E_in", "delta_above_E_out", "delta_above_E_in", "delta_hard_N_out", "delta_hard_N_in", "delta_hard_E_out_P", "delta_soft_E_out_P"):
            ent[f + "_12"] = wsum(f, [(1, h1), (2, h2)])
        for f in ("delta_hard_E_out", "delta_soft_E_out"):
            ent[f + "_123"] = wsum(f, [(1, h1), (2, h2), (3, h3)])
        d13["hard"].append(ent)
    P1 = {(p["t_over_rs"], p["cut"]): p for p in Dj[1]["probes"]}; P2 = {(p["t_over_rs"], p["cut"]): p for p in Dj[2]["probes"]}
    for key in sorted(P1):
        p1, p2 = P1[key], P2.get(key)
        ent = {"t": key[0], "cut": key[1], "rank": p1.get("rank"), "before_first_return": p1.get("before_first_return"), "first_return_rs": p1.get("first_return_rs"),
               "L_direct_12": wsum("L_direct", [(1, p1), (2, p2)]), "bound_rank16_trace_norm_k1": (p1.get("trace_norm_J")), "refs": []}
        r2 = {r["t_ref_over_rs"]: r for r in (p2 or {}).get("refs", [])}
        for r1 in p1.get("refs", []):
            rr = r2.get(r1["t_ref_over_rs"])
            ent["refs"].append({"t_ref": r1["t_ref_over_rs"], "L_total_12": wsum("L_total_backpropagated", [(1, r1), (2, rr)]), "L_old_12": wsum("L_old", [(1, r1), (2, rr)]), "L_sea_12": wsum("L_sea", [(1, r1), (2, rr)]), "L_src_12": wsum("L_src", [(1, r1), (2, rr)]),
                                "L_old_k1": 16.0 * r1["L_old"], "L_sea_k1": 16.0 * r1["L_sea"], "L_old_k2": 32.0 * rr["L_old"] if rr else None, "L_sea_k2": 32.0 * rr["L_sea"] if rr else None,
                                "bound_old_pointwise_12": wsum("bound_old_pointwise", [(1, r1), (2, rr)]), "check_L_minus_direct_k1": r1.get("check_L_minus_direct"), "n_source_modes_k1": r1.get("n_source_modes"), "cheb_tail_acc_k1": r1.get("cheb_tail_acc")})
        d13["probes"].append(ent)
    for k in (1, 2):
        d13["sources"][str(k)] = {}
        for t in (40.0, 60.0, 80.0, 100.0):
            cl = Dj[k]["clocks"]["%010.4f" % t]["per_channel"][str(k)]
            d13["sources"][str(k)][str(t)] = {ref: {f: v.get(f) for f in ("n_modes", "t_ref_over_rs", "tau_sum_collar", "tau_max_collar", "occ_inferred_mean_collar", "tau_sum_cut", "tau_max_cut", "occ_inferred_mean_cut", "tau_sum_far", "tau_max_far", "occ_inferred_mean_far", "radii")} for ref, v in (cl.get("sources") or {}).items()}
        d13["sources"][str(k) + "_definition"] = {ref: {f: v.get(f) for f in ("n_modes", "region", "region_R_abs", "weight_min", "t_ref_over_rs", "label", "occ_C_min", "lam_min_kept", "occ_vac_extremes", "n_in_modes", "n_candidates_above")} for ref, v in (Dj[k]["sources"][str(k)] or {}).items()}
    c100 = Dj[1]["clocks"]["%010.4f" % 100.0]["per_channel"]["1"]; c40 = Dj[1]["clocks"]["%010.4f" % 40.0]["per_channel"]["1"]
    d13["first_return_at_100"] = {lab: {f: v.get(f) for f in ("t_out_rs", "t_in_rs", "first_return_from_start_rs", "first_return_formation_rs", "first_return_rs", "before_first_return")} for lab, v in c100["first_return"].items()}
    d13["first_return_label"] = c100["first_return"]["16"]["label"]
    d13["first_return_at_40"] = {lab: v.get("first_return_rs") for lab, v in c40["first_return"].items()}
    d13["hard_threshold_energy_at_40"] = c40.get("hard_threshold_energy"); d13["T_H_ref"] = c40.get("T_H_ref"); d13["collar_split_at_100_k1"] = {f: c100["collar_split"].get(f) for f in ("E_flux_out_soft", "E_flux_out_hard", "E_flux_in_soft", "E_flux_in_hard", "E_flux_out", "E_flux_in", "omega_th", "L")}
    d13["enclosed_at_100_k1"] = {f: c100["enclosed"].get(f) for f in ("R_root", "I_L", "kappa_R_slope", "N_root", "M_root", "I_L_over_M_root", "delta_R_L", "delta_R_L_over_R", "label")}
    d13["windows_definition"] = "stored per channel by page_d13_readout_v1d (the detector-energy ledger at the cuts 16 and 30 r_s over windows of at least 13 r_s: the deviation's Killing-energy inventory beyond the cut at the window's start and end, the crossing of the state's current and of the record jumps, the held vacuum's change; second_term_D = inventory_D_end - inventory_D_start; telescoping_residual = change_C - crossing_total); here the kappa 1 and 2 channels summed with the weights 16 and 32 (kappa 3 beside where marked 123)"
    d13["probes_definition"] = Dj[1]["probes"][0].get("label")
    d13["hard_definition"] = "stored per channel: the exterior Killing shells' energy change per retarded window split by the radial current's sign, above (hard) and below (soft) the threshold 25 T_H^ref (hard_threshold_energy), and above the band top (above); the _P entries are the held vacuum's; weights 16 and 32 summed"
    # ---------------------------------------------------------------- run (a) (the vacuum's cut entropy under a sub-step field shift), as stored
    runA = []
    if os.path.exists(RUNA):
        note_input("run (a) results jsonl", RUNA)
        for line in open(RUNA):
            line = line.strip()
            if line.startswith("RESULT "):
                line = line[7:]
            if line:
                try:
                    runA.append(json.loads(line))
                except ValueError:
                    pass
    # ---------------------------------------------------------------- code and the composed md's
    for p in CODE_FILES:
        note_input("code " + os.path.basename(p), p)
    for p in (B2_PROD[0] + "/../out/W1_B2_composed_v3.md", B2_PROD[0] + "/../compose_manifest.txt", S8_PROD[0] + "_MANIFEST.md5"):
        if os.path.exists(p):
            note_input("composition file " + os.path.basename(p), p)
    note_input("this script", os.path.abspath(__file__))
    # ---------------------------------------------------------------- write
    def read_list(rd):
        return [dict(r, t=t, cut=lab) for (t, lab), r in sorted(rd.items())]
    out = {"script": os.path.abspath(__file__), "version": "v1b (version 4.1)", "cuts": CUTS, "cut_R_abs": CUT_R_ABS, "rs0_a": RS0, "settings": settings,
           "B2": {"clocks": clocks, "kappas": ks, "W": tolist(SB2["W"]), "x": tolist(SB2["x"]), "WX": tolist(KB2["WX"]), "fmeas_w": tolist(SB2["W"][:, None, None] * SB2["fmeas"]), "floor_high_w": tolist(SB2["W"][:, None, None] * SB2["floor_high"]),
                  "SUM": tolist(KB2["SUM"]), "FMW": tolist(KB2["FMW"]), "FAW": tolist(KB2["FAW"]), "C": tolist(KB2["C"]), "SHARE": tolist(KB2["SHARE"]), "partial": tolist(KB2["partial"]), "resolved": tolist(KB2["resolved"].astype(int)), "c_kappas": KB2["c_kappas"],
                  "S_X_raw_w": tolist(np.nansum(SB2["W"][:, None, None] * SB2["S_X"], axis=0)), "S_X_vac_w": tolist(np.nansum(SB2["W"][:, None, None] * SB2["S_X_vac"], axis=0)),
                  "E_kappa_w": tolist(SB2["row_E_inv_weighted"]), "E_out_w": tolist(SB2["row_E_out_weighted"]), "flow_w": tolist(SB2["row_collar_flow_acc_weighted"]), "T_meas": tolist(SB2["row_T_meas"]), "T_cheb": tolist(SB2["row_T_cheb"]), "T_trunc": tolist(SB2["row_T_trunc"]),
                  "readout": read_list(read_B2), "wall_s": {k: B2[k]["wall_s"] for k in ks}, "cheb_K_sum": {k: B2[k]["cheb_K_sum"] for k in ks}, "n_substeps": tolist(SB2["row_n_substeps"][:, -1])},
           "probes": {"kappas": pr["kappas"], "WX": tolist(pr["WX"]), "r_a": tolist(pr["r_a"]), "r_b": tolist(pr["r_b"]), "criterion_met": tolist(pr["criterion_met"].astype(int)), "q_larger": tolist(pr["q_larger"]), "conditional_majorant": tolist(pr["conditional_majorant"]), "conditional_majorant_over_S": tolist(pr["conditional_majorant_over_S"]), "selected_5e-4": tolist(pr["selected_5e-4"].astype(int)), "clocks": clocks, "ratio_24_18": tolist(pr["ratio_24_18"]), "ratio_32_18": tolist(pr["ratio_32_18"]), "ratio_48_18": tolist(pr["ratio_48_18"]),
                      "same_sign": tolist(pr["same_sign"].astype(int)), "slope": tolist(pr["slope_log_abs_WX_vs_log_kappa"]), "E_kappa_w": {str(K): tolist(v) for K, v in pr["E_kappa_w"].items()}, "resolved": tolist(pr["resolved"].astype(int)), "fmeas_w": tolist(pr["fmeas_w"]),
                      "final": {str(K): probes[K]["final"] for K in probes}, "wall_s": {str(K): probes[K]["wall_s"] for K in probes}, "flow_w": {str(K): tolist(probes[K]["flow_w"]) for K in probes}, "E_out_w": {str(K): tolist(probes[K]["E_out_w"]) for K in probes}},
           "common_mode_spread_12_18": tolist(cm_spread), "d_term_unweighted": tolist(d_term), "E_kappa_unweighted": tolist(E_kappa_unw), "sub16_B2": tolist(sub16_B2), "s8": {"clocks": clocks8, "kappas": ks, "x": tolist(SS8["x"]), "WX": tolist(KS8["WX"]), "SUM": tolist(KS8["SUM"]), "FMW": tolist(KS8["FMW"]), "FAW": tolist(KS8["FAW"]), "C": tolist(KS8["C"]), "SHARE": tolist(KS8["SHARE"]), "partial": tolist(KS8["partial"]),
                                                "resolved": tolist(KS8["resolved"].astype(int)), "E_kappa_w": tolist(SS8["row_E_inv_weighted"]), "E_out_w": tolist(SS8["row_E_out_weighted"]), "flow_w": tolist(SS8["row_collar_flow_acc_weighted"]), "E": tolist(SS8["row_E"][0]), "R_t": tolist(SS8["row_R_t_record"][0]), "minNv": tolist(SS8["row_min_N_minus_absv"][0]),
                                                "L_per_channel": tolist(SS8["row_L_interval"]), "LH_kT_per_channel": tolist(SS8["row_L_H_kappaT"]), "LH_TH_per_channel": tolist(SS8["row_L_H_TH"]), "join_dN": tolist(SS8["row_join_max_dN"][0]), "join_dv": tolist(SS8["row_join_max_rel_dv"][0]),
                                                "readout": read_list(read_s8), "wall_s": {k: S8[k]["wall_s"] for k in ks}},
           "sub16_s8": tolist(sub16_s8), "completions": {"clocks": clocks40, "runs": tolist(comp), "spread": tolist(spread)}, "flux": tolist(flux), "v3_flux": v3flux, "levels": tolist(levels), "collar_split": tolist(split), "marks": tolist(marks), "interp_control": interp, "d13": d13, "runA": runA,
           "checks": tolist(checks), "manifest": MANIFEST, "seconds": time.time() - t_start}
    with open(os.path.join(out_dir, "page_v4_data_v1.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(out_dir, "page_v4_data_v1.npz"), clocks=np.array(clocks), clocks8=np.array(clocks8), W=SB2["W"], x_B2=SB2["x"], WX_B2=KB2["WX"], SUM_B2=KB2["SUM"], FMW_B2=KB2["FMW"], FAW_B2=KB2["FAW"], C_B2=KB2["C"], SHARE_B2=KB2["SHARE"], partial_B2=KB2["partial"],
                        WX_probes=pr["WX"], x_s8=SS8["x"], SUM_s8=KS8["SUM"], FMW_s8=KS8["FMW"], C_s8=KS8["C"], SHARE_s8=KS8["SHARE"], D16_B2=sub16_B2["D16_w"], D16_s8=sub16_s8["D16_w"],
                        L12=flux["L12"], LH12_kT=flux["LH12_kT"], LH12_TH=flux["LH12_TH"], L123=flux["L123"], LH123_kT=flux["LH123_kT"], LH123_TH=flux["LH123_TH"], L18=flux["L18"], LH18_kT=flux["LH18_kT"], LH18_TH=flux["LH18_TH"],
                        t_inf_collar_over_rs=flux["t_inf_collar_over_rs"], E_B2=flux["E"], E_inv12=flux["E_inv12"], E_inv18=flux["E_inv18"], flow12=flux["flow12"], flow18=flux["flow18"], join_dv=flux["join_dv"], join_dN=flux["join_dN"],
                        E_kappa_w_B2=SB2["row_E_inv_weighted"], E_kappa_w_s8=SS8["row_E_inv_weighted"])
    with open(os.path.join(out_dir, "MANIFEST.md5"), "w") as f:
        for m in MANIFEST:
            f.write("%s  %s  [%s]\n" % (m["md5"], m["realpath"], m["label"]))
        for name in ("page_v4_data_v1.json", "page_v4_data_v1.npz"):
            f.write("%s  %s  [output]\n" % (md5(os.path.join(out_dir, name)), os.path.join(out_dir, name)))
    print("checks:", json.dumps(tolist(checks), indent=1)[:4000])
    print("written %s/page_v4_data_v1.{json,npz} and MANIFEST.md5 in %.1f s; B2 clocks %d, s8 clocks %d, readout pairs B2 %d, s8 %d; sub16 B2 final %s (last clock %s)" % (out_dir, time.time() - t_start, len(clocks), len(clocks8), len(read_B2), len(read_s8), sub16_B2["final"], sub16_B2["last_clock"]))


if __name__ == "__main__":
    main()
