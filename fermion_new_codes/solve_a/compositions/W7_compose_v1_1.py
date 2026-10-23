#!/usr/bin/env python
"""W7_compose_v1_1.py (the solve lane; COMP2 v1.1 under reviewer A's binding W7 rulings, deliveries/lead_gaps/ReviewA_w7_rulings_v1.md; box CPU,
arithmetic only). Reads W7_v1's ancestry and r11-slice outputs (read-only), W7_v1_1's ext100-slice outputs (48 to 56, re-fetched) and the
records, and writes W7_compose_v1_1.{json,npz} and W7_compose_v1_1_tables.txt in --out-dir. Every count is printed weighted (16 kappa, the
channel multiplicity; ruling 6) with per copy beside it.
A. (ruling 1) The same-set check 15 to 56: each stored slice's dust columns against the propagated formation set at the same clock (column
   overlap |<slice|propagated>|^2, site-weight L1 difference, the propagated weight beyond the slice chain), the precision floor per slice
   (its stored purity defect), and the gate: 1 - overlap and the L1 difference at or below 1e3 x the floor (or 1e-9 for complex128).
B. (ruling 1) The far weight beyond 16 and 24 r_s on the slices 48 to 56, the propagated set beside.
C. (ruling 2) The Killing split of the exterior part beyond 16 r_s on the FULL held operator's spectral projector (horizon3_bins_v1 bins; the
   mean Killing energy above m), the restricted block as the control (the difference per bin), on every slice and on the propagated set at 56.
D. (ruling 3) The dust number current through 16 r_s summed over channels (weighted), at the sampled clocks 48 to 56 on the slices and on
   the propagated set, with the count of propagated records 48 to 56 at which it is outward; per channel signs to the record.
E. (ruling 4) The half-arrival of the whole dust at 16 r_s, referred to the last record (56), labelled still rising.
F. The certified bound at record 15: per column [sqrt(p_late) - sqrt(eps_early)]_+^2 with eps_early the column's weight beyond the dense
   root at 15 and p_late its weight beyond 16 r_s at 56 (both over the column's norm), composed with the multiplicities (ruling 6); two
   routes: the propagated set (the 15 launch) and slice-only (the 15 and 56 stored slices); the upper bound and the propagated inside-origin
   content beside; the applied root beside.
G. (rulings, red-team paragraph) The cancellation check: the (-b, -m] exterior content of the inside and the outside pieces of the formation
   launch against the whole's, on the full projector.
H. (ruling 5) The capacity, record only, no verdict: the minimum over rooted records of (E(t) - R_t(t)/(2G))/(m e_min) and at m, and the
   weight outside the dense root at 15.
I. The ledger and the inside-origin split reprinted weighted from W7_v1's compose json.
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "2"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import W7_common_v1 as C   # noqa: E402

import argparse   # noqa: E402
import json       # noqa: E402

import numpy as np   # noqa: E402

sys.path.insert(0, C.ENGINE_DIR)
import horizon3_bins_v1 as HB   # noqa: E402

M = C.M_MASS; B = HB.band_top(M); G = C.G_NEWTON; RS = 64.0; EMIN = 0.8771179724
MULT = {k: 16.0 * k for k in C.KAPPAS}
SAMPLED = (48.0, 50.0, 52.0, 54.0, 56.0)
TXT = []


def P(s=""):
    TXT.append(s)


def g6(x, nd=6):
    if x is None:
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "nan" if not np.isfinite(xf) else ("%%.%dg" % nd) % xf


def idx(T, t):
    return int(np.argmin(np.abs(np.asarray(T) - t)))


def records():
    tab = {}
    J1 = json.load(open(C.REC1)); J2 = json.load(open(C.REC2)); E0 = float(J1["records"][0]["E"])
    for src, J in ((1, J1), (2, J2)):
        for r in J["records"]:
            t = r.get("T_over_rs")
            if t is None:
                continue
            t = round(float(t), 6)
            if src == 2 and t <= 40.0 + 1e-9 and t in tab:
                continue
            z = [float(x) for x in (r.get("trapping_zeros") or [])]
            tab[t] = {"E": float(r["E"]), "Rt": max(z) if z else None}
    return tab, E0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-dir", default="<internal>/W7_v1")
    ap.add_argument("--out-dir", default="<internal>/W7_v1_1")
    a = ap.parse_args()
    tab, E0 = records()
    OUT = {"script": "W7_compose_v1_1.py", "script_md5": C.md5(os.path.abspath(__file__)), "rulings": "<internal>/ReviewA_w7_rulings_v1.md",
           "m": M, "b": B, "G": G, "multiplicity": "16 kappa", "inputs": {}, "items": {}}
    AJ = {}; AZ = {}
    for k in C.KAPPAS:
        pj = os.path.join(a.v1_dir, "W7_ancestry_k%d.json" % k)
        AJ[k] = json.load(open(pj)); AZ[k] = np.load(pj[:-5] + ".npz")
        OUT["inputs"]["W7_ancestry_k%d.npz" % k] = C.md5(pj[:-5] + ".npz")
    SJ = {}; SZ = {}
    import glob
    for d in [a.v1_dir] + sorted(glob.glob(os.path.join(a.out_dir, "s[0-9]*"))):
        pj = os.path.join(d, "W7_slices.json")
        J = json.load(open(pj)); OUT["inputs"][pj] = C.md5(pj)
        for key, S in J["slices"].items():
            pz = os.path.join(d, "W7_slice_%s.npz" % key)
            if S.get("present") and os.path.exists(pz):
                SJ[S["T"]] = S; SZ[S["T"]] = np.load(pz); OUT["inputs"][pz] = C.md5(pz)
    V1 = json.load(open(os.path.join(a.v1_dir, "W7_compose_v1.json")))
    OUT["inputs"]["W7_compose_v1.json"] = C.md5(os.path.join(a.v1_dir, "W7_compose_v1.json"))
    OUT["slices"] = {"T%06.3f" % t: {kk: S.get(kk) for kk in ("path", "md5", "manifest_md5", "md5_matches_manifest", "dtype", "keys_present", "fields_vs_history_max", "n_sites", "R_max", "roots")} for t, S in SJ.items()}
    Tp = AZ[1]["T12p125_T"]

    # ---------------- A. same-set check 15 .. 56
    A = []
    for t in sorted(SZ):
        if t < 15.0:
            continue
        floor = max(v["stored_purity_defect"] for v in SJ[t]["kappas"].values())
        lim = max(1e3 * floor, 1e-9)
        for k in C.KAPPAS:
            key = "T12p125_state_all_T%06.3f" % t
            Sp = AZ[k][key]; Ds = SZ[t]["k%d_cols" % k]; n = Ds.shape[0] // 2
            ov = np.abs((Ds.conj() * Sp[:2 * n]).sum(axis=0)) ** 2
            l1 = np.abs(C.site_w(Sp[:2 * n], n) - SZ[t]["k%d_sitew" % k]).sum(axis=0)
            tail = C.site_w(Sp, Sp.shape[0] // 2)[n:].sum(axis=0)
            A.append({"T": t, "kappa": k, "dtype": SJ[t]["dtype"], "purity_floor": floor, "limit": lim, "one_minus_overlap_max": float((1 - ov).max()),
                      "site_weight_L1_max": float(l1.max()), "propagated_beyond_slice_chain_max": float(tail.max()),
                      "pass": bool((1 - ov).max() <= lim and l1.max() <= max(lim, 1e-6 if SJ[t]["dtype"] == "complex64" else lim))})
    gateA = all(x["pass"] for x in A if x["T"] >= 48.0)
    OUT["items"]["A_same_set"] = {"rows": A, "gate_48_56_pass": gateA}
    P("## A. Same-set check, stored slices against the propagated formation set (ruling 1)")
    P("| T | dtype | purity floor | max 1 - overlap | max site-weight L1 | propagated weight beyond the slice chain | pass |")
    P("|---|---|---|---|---|---|---|")
    for t in sorted({x["T"] for x in A}):
        r = [x for x in A if x["T"] == t]
        P("| %s | %s | %s | %s | %s | %s | %s |" % (g6(t), r[0]["dtype"], g6(r[0]["purity_floor"], 2), g6(max(x["one_minus_overlap_max"] for x in r), 2), g6(max(x["site_weight_L1_max"] for x in r), 2),
                                                 g6(max(x["propagated_beyond_slice_chain_max"] for x in r), 2), all(x["pass"] for x in r)))
    P("Gate for printing 48 to 56: %s" % gateA)
    P("")

    # ---------------- B, C, D on the slices, and the propagated set
    rowsS = []
    for t in sorted(SZ):
        Zs = SZ[t]; Rs = Zs["R"]
        rec = tab.get(round(t, 6), {})
        cnt = {"b16": [0.0, 0.0], "b24": [0.0, 0.0]}; J16 = [0.0, 0.0]; J16k = []
        full = np.zeros(5); fullw = np.zeros(5); blk = np.zeros(5); blkw = np.zeros(5); thrw = np.zeros(5); mnum = 0.0; mden = 0.0
        for k in C.KAPPAS:
            f = Zs["k%d_f" % k]; sw = Zs["k%d_sitew" % k]
            for cr, c in (("b16", 16.0), ("b24", 24.0)):
                x = float(f @ sw[Rs > c * RS].sum(axis=0)); cnt[cr][0] += x; cnt[cr][1] += MULT[k] * x
            j = float(f @ Zs["k%d_J_16" % k]); J16[0] += j; J16[1] += MULT[k] * j; J16k.append(j)
            if ("k%d_full_e" % k) in Zs.files:
                eA = Zs["k%d_full_e" % k]; wA = Zs["k%d_full_w_ext16" % k] @ f
                b5 = HB.killing_bins_5(eA, wA, M, B); thr = HB.threshold_columns(eA, wA, M, B, 2 * G * rec.get("E", np.nan), float(Rs[-1]))
                eB = Zs["k%d_block16_e" % k]; wB = Zs["k%d_block16_w" % k] @ f; bb = HB.killing_bins_5(eB, wB, M, B)
                full += b5; fullw += MULT[k] * b5; blk += bb; blkw += MULT[k] * bb; thrw += MULT[k] * np.array(thr["threshold_error"]["bins5"])
                nu, de = C.mean_positive(eA, wA[:, None], M); mnum += MULT[k] * float(nu[0]); mden += MULT[k] * float(de[0])
        rowsS.append({"T": t, "source": "slice", "b16_per_copy": cnt["b16"][0], "b16_weighted": cnt["b16"][1], "b24_per_copy": cnt["b24"][0], "b24_weighted": cnt["b24"][1],
                      "J16_per_copy": J16[0], "J16_weighted": J16[1], "J16_per_kappa": J16k, "full_bins_per_copy": full.tolist(), "full_bins_weighted": fullw.tolist(),
                      "block_bins_weighted": blkw.tolist(), "block_minus_full_weighted": (blkw - fullw).tolist(), "threshold_term_weighted": thrw.tolist(),
                      "mean_E_above_m_over_m_weighted": (mnum / mden / M) if mden > 0 else None, "mb_share": (fullw[3] / fullw.sum()) if fullw.sum() > 0 else None})
    # the propagated set at 48 .. 56 and the full projector at 56
    rowsP = []
    for t in SAMPLED:
        j = idx(Tp, t); r = {"T": t, "source": "propagated"}
        for cr in ("b16", "b24"):
            v = [float(AZ[k]["T12p125_f"] @ AZ[k]["T12p125_%s_all" % cr][j]) for k in C.KAPPAS]
            r[cr + "_per_copy"] = sum(v); r[cr + "_weighted"] = sum(MULT[k] * v[k - 1] for k in C.KAPPAS)
        v = [float(AZ[k]["T12p125_f"] @ AZ[k]["T12p125_J16_all"][j]) for k in C.KAPPAS]
        r["J16_per_copy"] = sum(v); r["J16_weighted"] = sum(MULT[k] * v[k - 1] for k in C.KAPPAS); r["J16_per_kappa"] = v
        rowsP.append(r)
    full = np.zeros(5); fullw = np.zeros(5); blkw = np.zeros(5); thrw = np.zeros(5); mnum = 0.0; mden = 0.0
    sets_all = {}
    for k in C.KAPPAS:
        Z = AZ[k]; i0 = 0
        for tg, nm, nc in AJ[k]["held_sets"]:
            sets_all[(k, tg, nm)] = (i0, nc); i0 += nc
        i0, nc = sets_all[(k, "T12p125", "all")]; f = Z["T12p125_f"]
        eA = Z["held_full_e"]; wA = Z["held_full_w_ext16"][:, i0:i0 + nc] @ f; b5 = HB.killing_bins_5(eA, wA, M, B)
        thr = HB.threshold_columns(eA, wA, M, B, 2 * G * tab[56.0]["E"], float(AJ[k]["n_sites"]))
        eB = Z["held_block16_e"]; wB = Z["held_block16_w"][:, i0:i0 + nc] @ f; bb = HB.killing_bins_5(eB, wB, M, B)
        full += b5; fullw += MULT[k] * b5; blkw += MULT[k] * bb; thrw += MULT[k] * np.array(thr["threshold_error"]["bins5"])
        nu, de = C.mean_positive(eA, wA[:, None], M); mnum += MULT[k] * float(nu[0]); mden += MULT[k] * float(de[0])
    rowsP[-1].update({"full_bins_per_copy": full.tolist(), "full_bins_weighted": fullw.tolist(), "block_bins_weighted": blkw.tolist(), "block_minus_full_weighted": (blkw - fullw).tolist(),
                      "threshold_term_weighted": thrw.tolist(), "mean_E_above_m_over_m_weighted": mnum / mden / M, "mb_share": fullw[3] / fullw.sum()})
    # the record count of the summed current, propagated records 48 .. 56
    Jser = sum(MULT[k] * (AZ[k]["T12p125_J16_all"] @ AZ[k]["T12p125_f"]) for k in C.KAPPAS)
    j48 = idx(Tp, 48.0)
    rec_out = int(np.sum(Jser[j48:] > 0)); rec_n = int(len(Jser) - j48)
    OUT["items"]["BCD_far_weight_split_current"] = {"slices": rowsS, "propagated": rowsP, "current_records_48_56_outward": rec_out, "current_records_48_56": rec_n,
                                                    "current_outward_at_sampled_slices": all(r["J16_weighted"] > 0 for r in rowsS if r["T"] in SAMPLED),
                                                    "current_outward_at_sampled_propagated": all(r["J16_weighted"] > 0 for r in rowsP)}
    P("## B, C, D. Far weight, Killing split on the full held projector (block control), current through 16 r_s (weighted; per copy beside)")
    P("| T | source | beyond 16 r_s weighted | per copy | beyond 24 r_s weighted | J16 summed, weighted | full bins weighted (-inf,-b] (-b,-m] (-m,m] (m,b] (b,inf) | (m,b] share | mean E above m (m) | block - full, weighted | threshold term weighted |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in [x for x in rowsS] + rowsP:
        P("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (g6(r["T"]), r["source"], g6(r["b16_weighted"]), g6(r["b16_per_copy"]), g6(r["b24_weighted"]), g6(r["J16_weighted"], 4),
                                                                   [g6(x, 4) for x in r.get("full_bins_weighted", [])], g6(r.get("mb_share"), 5), g6(r.get("mean_E_above_m_over_m_weighted"), 4),
                                                                   [g6(x, 2) for x in r.get("block_minus_full_weighted", [])], [g6(x, 2) for x in r.get("threshold_term_weighted", [])]))
    P("Per-channel J16 at the sampled clocks (slices): " + "; ".join("%s %s" % (g6(r["T"]), [g6(x, 3) for x in r["J16_per_kappa"]]) for r in rowsS if r["T"] in SAMPLED))
    P("Summed current outward at the sampled clocks: slices %s, propagated %s; propagated records 48 to 56 outward %d of %d" % (
        OUT["items"]["BCD_far_weight_split_current"]["current_outward_at_sampled_slices"], OUT["items"]["BCD_far_weight_split_current"]["current_outward_at_sampled_propagated"], rec_out, rec_n))
    P("")

    # ---------------- E. half-arrival of the whole dust, referred to 56
    Fw = sum(MULT[k] * (AZ[k]["T12p125_b16_all"] @ AZ[k]["T12p125_f"]) for k in C.KAPPAS)
    Fc = sum(AZ[k]["T12p125_b16_all"] @ AZ[k]["T12p125_f"] for k in C.KAPPAS)
    T_form = AJ[1]["T_form"]

    def half(F):
        h = 0.5 * F[-1]; j = next(i for i in range(len(F)) if F[i] >= h)
        return float(Tp[j - 1] + (h - F[j - 1]) / (F[j] - F[j - 1]) * (Tp[j] - Tp[j - 1]))
    Th = half(Fw); Thc = half(Fc)
    still = bool(np.all(np.diff(Fw[idx(Tp, 54.0):]) > 0))
    first1 = next((float(Tp[i]) for i in range(len(Fw)) if Fw[i] >= 1.0), None)
    sl_half = None
    OUT["items"]["E_half_arrival"] = {"T_half_weighted": Th, "delay_weighted_rs": Th - T_form, "T_half_per_copy": Thc, "delay_per_copy_rs": Thc - T_form, "reference": "the last record, T/r_s 56",
                                      "still_rising": still, "F56_weighted": float(Fw[-1]), "F_max_weighted": float(Fw.max()), "first_clock_one_fermion_beyond_16rs": first1,
                                      "slices_beyond16_weighted": {g6(r["T"]): r["b16_weighted"] for r in rowsS if r["T"] >= 48}}
    P("## E. Half-arrival of the whole dust at 16 r_s, referred to the last record (still rising: %s)" % still)
    P("weighted: T %s, %s r_s after formation (%s); per copy: T %s (%s r_s); F(56) %s weighted (max %s); one fermion beyond 16 r_s first at %s" % (
        g6(Th), g6(Th - T_form), g6(T_form), g6(Thc), g6(Thc - T_form), g6(Fw[-1]), g6(Fw.max()), g6(first1)))
    P("")

    # ---------------- F. the certified bound at record 15
    Fb = {}
    for cut in ("dense", "applied"):
        lo_w = lo_c = up_w = up_c = bin_w = bin_c = ins_w = ins_c = out_w = out_c = 0.0; per = {}
        for k in C.KAPPAS:
            Z = AZ[k]; f = Z["T15p000_f"]; norm = Z["T15p000_norm"]
            eps = Z["T15p000_beyond_Rt_%s" % cut][0] / norm[0]; p = Z["T15p000_b16_all"][-1] / norm[-1]; bi = Z["T15p000_b16_in_%s" % cut][-1] / norm[-1]
            lo = np.clip(np.sqrt(p) - np.sqrt(eps), 0, None) ** 2; up = np.minimum((np.sqrt(p) + np.sqrt(eps)) ** 2, 1 - eps)
            per[str(k)] = {"eps_early": eps.tolist(), "p_late": p.tolist(), "lower": lo.tolist(), "b_in": bi.tolist(), "upper": up.tolist()}
            lo_c += f @ lo; lo_w += MULT[k] * (f @ lo); up_c += f @ up; up_w += MULT[k] * (f @ up); bin_c += f @ bi; bin_w += MULT[k] * (f @ bi)
            ins = f @ (norm[0] - Z["T15p000_beyond_Rt_%s" % cut][0]); ins_c += ins; ins_w += MULT[k] * ins
            ot = f @ Z["T15p000_beyond_Rt_%s" % cut][0]; out_c += ot; out_w += MULT[k] * ot
        Fb[cut + "_propagated"] = {"lower_per_copy": float(lo_c), "lower_weighted": float(lo_w), "upper_per_copy": float(up_c), "upper_weighted": float(up_w),
                                   "b_in_per_copy": float(bin_c), "b_in_weighted": float(bin_w), "inside_at_15_per_copy": float(ins_c), "inside_at_15_weighted": float(ins_w),
                                   "outside_at_15_per_copy": float(out_c), "outside_at_15_weighted": float(out_w),
                                   "lower_fraction_of_inside_weighted": float(lo_w / ins_w), "lower_fraction_of_inside_per_copy": float(lo_c / ins_c), "per_kappa": per}
        # slice-only: eps from the 15 slice, p_late from the 56 slice (2048-site chain)
        if 15.0 in SZ and 56.0 in SZ:
            R15 = SJ[15.0]["roots"]["%s_outer" % cut]; Rs15 = SZ[15.0]["R"]; Rs56 = SZ[56.0]["R"]
            lo_c2 = lo_w2 = 0.0
            for k in C.KAPPAS:
                f = SZ[15.0]["k%d_f" % k]; s15 = SZ[15.0]["k%d_sitew" % k]; s56 = SZ[56.0]["k%d_sitew" % k]
                eps = s15[Rs15 > R15].sum(axis=0) / s15.sum(axis=0); p = s56[Rs56 > 16 * RS].sum(axis=0) / s56.sum(axis=0)
                lo = np.clip(np.sqrt(p) - np.sqrt(eps), 0, None) ** 2
                lo_c2 += f @ lo; lo_w2 += MULT[k] * (f @ lo)
            Fb[cut + "_slices_15_56"] = {"root_15": R15, "lower_per_copy": float(lo_c2), "lower_weighted": float(lo_w2)}
    OUT["items"]["F_certified_bound_15"] = Fb
    P("## F. Certified bound at record 15 (beyond 16 r_s at 56; per column, composed with 16 kappa)")
    for cut in ("dense", "applied"):
        x = Fb[cut + "_propagated"]; y = Fb.get(cut + "_slices_15_56", {})
        P("%s root: lower bound weighted %s (per copy %s); slice-only route weighted %s (per copy %s); inside the root at 15 weighted %s (per copy %s), so %s of it (per copy %s); outside at 15 weighted %s (per copy %s); upper bound weighted %s; propagated inside-origin content weighted %s (per copy %s)" % (
            cut, g6(x["lower_weighted"]), g6(x["lower_per_copy"]), g6(y.get("lower_weighted")), g6(y.get("lower_per_copy")), g6(x["inside_at_15_weighted"]), g6(x["inside_at_15_per_copy"]),
            g6(x["lower_fraction_of_inside_weighted"], 4), g6(x["lower_fraction_of_inside_per_copy"], 4), g6(x["outside_at_15_weighted"]), g6(x["outside_at_15_per_copy"]), g6(x["upper_weighted"]),
            g6(x["b_in_weighted"]), g6(x["b_in_per_copy"])))
    P("")

    # ---------------- G. the cancellation check on the full projector
    Gc = {}
    for nm in ("all", "in_dense", "out_dense", "in_applied", "out_applied"):
        bw = np.zeros(5); bc = np.zeros(5)
        for k in C.KAPPAS:
            i0, nc = sets_all[(k, "T12p125", nm)]; f = AZ[k]["T12p125_f"]
            b5 = HB.killing_bins_5(AZ[k]["held_full_e"], AZ[k]["held_full_w_ext16"][:, i0:i0 + nc] @ f, M, B); bc += b5; bw += MULT[k] * b5
        Gc[nm] = {"per_copy": bc.tolist(), "weighted": bw.tolist()}
    OUT["items"]["G_cancellation"] = Gc
    P("## G. (-b, -m] exterior content at 56 on the full projector, formation launch: whole %s, inside piece %s, outside piece %s (weighted); per copy %s, %s, %s; applied cut inside %s, outside %s (weighted)" % (
        g6(Gc["all"]["weighted"][1]), g6(Gc["in_dense"]["weighted"][1]), g6(Gc["out_dense"]["weighted"][1]), g6(Gc["all"]["per_copy"][1]), g6(Gc["in_dense"]["per_copy"][1]), g6(Gc["out_dense"]["per_copy"][1]),
        g6(Gc["in_applied"]["weighted"][1]), g6(Gc["out_applied"]["weighted"][1])))
    P("")

    # ---------------- H. capacity, record only
    rooted = [(t, r) for t, r in sorted(tab.items()) if r["Rt"] is not None and t >= T_form - 1e-9 and t <= 56.0 + 1e-9]
    caps = [(t, (r["E"] - r["Rt"] / (2 * G)) / (M * EMIN), (r["E"] - r["Rt"] / (2 * G)) / M) for t, r in rooted]
    tmin = min(caps, key=lambda x: x[1])
    out15 = Fb["dense_propagated"]
    OUT["items"]["H_capacity_record"] = {"min_over_rooted_records_m_emin": tmin[1], "at_m": tmin[2], "at_T": tmin[0], "n_rooted_records": len(caps),
                                         "C15_at_m": (tab[15.0]["E"] - tab[15.0]["Rt"] / (2 * G)) / M, "outside_dense_root_at_15_weighted": out15["outside_at_15_weighted"],
                                         "outside_dense_root_at_15_per_copy": out15["outside_at_15_per_copy"], "verdict": "none printed (ruling 5)"}
    P("## H. Capacity (record only, no verdict): min over %d rooted records of (E - R_t/(2G))/(m e_min) = %s at T %s (at m %s); C(15) at m %s; weight outside the dense root at 15 weighted %s (per copy %s)" % (
        len(caps), g6(tmin[1]), g6(tmin[0]), g6(tmin[2]), g6(OUT["items"]["H_capacity_record"]["C15_at_m"]), g6(out15["outside_at_15_weighted"]), g6(out15["outside_at_15_per_copy"])))
    P("")

    # ---------------- I. ledger and split, weighted, from v1
    L = V1["items"]["1_ledger"]["record_level"]["dense_weighted_16kappa"]
    I4 = V1["items"]["4_inside_origin"]
    OUT["items"]["I_reprinted_weighted"] = {"ledger_dense_weighted": {"sum_T_j": L["sum_T_j_all"], "sum_S_j": L["sum_S_j_all"], "between_slice_clocks": L["between_slice_clocks"], "window": L["window_15p5_16p125"]},
                                            "split_weighted": {k: {cr: v[cr]["weighted_16kappa"] for cr in ("b16", "b24")} for k, v in I4.items()}}
    P("## I. Ledger (dense, weighted): sum T_j %s, sum S_j %s; window %s" % (g6(L["sum_T_j_all"]), g6(L["sum_S_j_all"]), "; ".join("%s->%s T %s S %s" % (g6(w["T_a"]), g6(w["T_b"]), g6(w["T_j"]), g6(w["S_j"])) for w in L["window_15p5_16p125"])))
    P("| interval | p_in start | p_in end | sum T_j | sum S_j |")
    P("|---|---|---|---|---|")
    for x in L["between_slice_clocks"]:
        P("| %s to %s | %s | %s | %s | %s |" % (g6(x["T_a"]), g6(x["T_b"]), g6(x["p_in_a"]), g6(x["p_in_b"]), g6(x["sum_T_j"]), g6(x["sum_S_j"])))
    P("Split at 56 beyond 16 r_s, weighted (all / in / out / cross / lower): " + "; ".join("%s %s/%s/%s/%s/%s" % (k, g6(v["b16"]["weighted_16kappa"]["all"]), g6(v["b16"]["weighted_16kappa"]["in"]), g6(v["b16"]["weighted_16kappa"]["out"]),
                                                                                                                    g6(v["b16"]["weighted_16kappa"]["cross"]), g6(v["b16"]["weighted_16kappa"]["lower"])) for k, v in I4.items()))
    C.dump_json(OUT, os.path.join(a.out_dir, "W7_compose_v1_1.json"))
    C.save_npz({"T": Tp, "F16_whole_weighted": Fw, "F16_whole_per_copy": Fc, "J16_whole_weighted": Jser}, os.path.join(a.out_dir, "W7_compose_v1_1.npz"))
    with open(os.path.join(a.out_dir, "W7_compose_v1_1_tables.txt"), "w") as fh:
        fh.write("\n".join(TXT) + "\n")
    print("\n".join(TXT))


if __name__ == "__main__":
    main()
