#!/usr/bin/env python
"""W7_compose_v1.py (the solve lane; reorientation 6 composition 2, W7's five items; box CPU, arithmetic only). Reads W7_ancestry_v1's and
W7_slices_v1's outputs, the records' E and trapping zeros, DUST_root_history v2's npz (the gate on kappa 1 and 2) and B.5 version 2's columns,
and writes W7_compose_v1.json (every number), W7_compose_v1.npz (the per-record series) and W7_compose_v1_tables.txt (the tables the notes
quote). No engine, no eigendecomposition: the binning is horizon3_bins_v1's on the saved levels and weights.

Definitions (each an existing one; the source named):
1. Same-set fixed-cut ledger (busi W7_dust_v2b_audit_v1b.md section 0; dust_ledger_postpass_v1b.ledger_arrays): the set = one launch's dust
   columns propagated as one set, never re-tagged; p_j = f . ||1_(R < R_j) psi(t_j)||^2; T_j = f . (||1_(R > R_j) psi(t_(j+1))||^2 -
   ||1_(R > R_j) psi(t_j)||^2) the net outward transport through the OLD cut; S_j = f . (||1_(R < R_(j+1)) psi(t_(j+1))||^2 -
   ||1_(R < R_j) psi(t_(j+1))||^2) the sweep of the new mask; p_(j+1) - p_j = -T_j + S_j + D_j, D_j the weighted norm change. Record by
   record on the propagated set from the formation record, and slice to slice on the stored slices (the same formulas with the slices' columns).
2. Killing split on the held exterior block: the exterior part 1_(R > 16 r_s) psi of the dust on the eigenvectors of the held operator's
   restriction to the sites beyond 16 r_s (export_horizon_refcol_v2's block), 5 bins and the threshold columns of horizon3_bins_v1 (buffers 1
   and 2, half-widths with r_s the record's 2GE and R_max the chain's end); beside it the same exterior part on the full held operator's
   eigenvectors (dust_ledger_postpass_v1b), with the extended/bound split of the (m, b] eigenvectors (weight inside R_cut below a threshold).
3. Never-inside capacity at the record's charge (dust_ledger_postpass_v1b.capacity_rows): C(t) = (E(t) - R_t(t)/(2G))/eps_p, eps_p = m and
   1.2 m, beside eps_p = m e_min (P05's cold bound e_min = 0.8771179724), at E_0, and with the largest earlier root in place of R_t(t); against
   the dust count beyond 16 r_s with the multiplicities 16 kappa.
4. Inside-origin split at the root records 12.125 and 15 (DUST v2; the certificate of dust_ledger_postpass_v1b.certificates): the far weight at
   56 = from inside + from outside + cross term, the launch split by the dense root (the applied root beside); per column the same-packet
   certificate [sqrt(p_late) - sqrt(eps_early)]_+^2 and its upper bound min((sqrt p_late + sqrt eps_early)^2, 1 - eps_early).
5. Escape certificate (REVIEWA_REORIENT_6.md 2.6 item 4): the fraction of the dust weight inside the dense root at the formation record that at 56
   lies beyond 16 r_s with Killing energy in (m, b] on the held exterior block, with the outward number current of that content through 16 r_s at
   the sampled clocks 48, 50, 52, 54, 56; the delay the clock from the formation record to the half-arrival at 16 r_s.
Usage (box): nice -n 10 python W7_compose_v1.py --dir <internal>/W7_v1
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

M = C.M_MASS
B = HB.band_top(M)
G = C.G_NEWTON
RS = 64.0
EMIN_P05 = 0.8771179724
SAMPLED = (48.0, 50.0, 52.0, 54.0, 56.0)
SLICE_CLOCKS = (12.125, 15.0, 16.0, 20.0, 24.0, 25.0, 28.0, 30.0, 35.0, 40.0, 48.0, 50.0, 52.0, 54.0, 56.0)
MULT = {k: 16.0 * k for k in C.KAPPAS}
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
    if not np.isfinite(xf):
        return "nan"
    return ("%%.%dg" % nd) % xf


def records_table():
    tab = {}
    J1 = json.load(open(C.REC1)); J2 = json.load(open(C.REC2))
    E0 = float(J1["records"][0]["E"])
    for src, J in ((1, J1), (2, J2)):
        for r in J["records"]:
            t = r.get("T_over_rs")
            if t is None:
                continue
            t = round(float(t), 6)
            if src == 2 and t <= 40.0 + 1e-9 and t in tab:
                continue
            z = [float(x) for x in (r.get("trapping_zeros") or [])]
            tab[t] = {"E": float(r["E"]), "Rt": max(z) if z else None, "Ri": min(z) if len(z) > 1 else None, "src": src}
    return tab, E0


def idx_of(T, t):
    return int(np.argmin(np.abs(np.asarray(T) - t)))


def bins_of(e, w):
    return HB.killing_bins_5(e, w, M, B)


def ledger_from(T, f, norm, beyond, mcs_col):
    p_out = beyond @ f; p_norm = norm @ f; p_in = p_norm - p_out; mcs = np.where(np.isfinite(mcs_col).all(axis=1), mcs_col @ f, np.nan)
    nr = len(T); Tj = np.full(nr, np.nan); Sj = np.full(nr, np.nan); Dj = np.full(nr, np.nan); dp = np.full(nr, np.nan)
    for j in range(1, nr):
        if np.isfinite(p_out[j - 1]) and np.isfinite(p_out[j]) and np.isfinite(mcs[j]):
            Sj[j] = -mcs[j]; Tj[j] = (p_out[j] - mcs[j]) - p_out[j - 1]; dp[j] = p_in[j] - p_in[j - 1]; Dj[j] = dp[j] + Tj[j] - Sj[j]
    return {"T": np.asarray(T), "p_in": p_in, "T_j": Tj, "S_j": Sj, "D_j": Dj, "dp_in": dp, "D_norm": np.r_[np.nan, np.diff(p_norm)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="<internal>/W7_v1")
    a = ap.parse_args()
    d = a.dir
    tab, E0 = records_table()
    OUT = {"script": "W7_compose_v1.py", "script_md5": C.md5(os.path.abspath(__file__)), "common_md5": C.md5(C.__file__), "bins_module_md5": C.md5(HB.__file__),
           "m": M, "b": B, "G": G, "rs": RS, "E0": E0, "multiplicity": "16 kappa (the slices' weights)", "inputs": {}, "gates": {}, "items": {}}
    SER = {}
    AJ = {}; AZ = {}
    for k in C.KAPPAS:
        pj = os.path.join(d, "W7_ancestry_k%d.json" % k)
        AJ[k] = json.load(open(pj)); AZ[k] = np.load(pj[:-5] + ".npz")
        OUT["inputs"]["W7_ancestry_k%d" % k] = {"json_md5": C.md5(pj), "npz_md5": C.md5(pj[:-5] + ".npz")}
    SJ = json.load(open(os.path.join(d, "W7_slices.json")))
    OUT["inputs"]["W7_slices.json"] = C.md5(os.path.join(d, "W7_slices.json"))
    SZ = {}
    for key, S in SJ["slices"].items():
        pz = os.path.join(d, "W7_slice_%s.npz" % key)
        if os.path.exists(pz):
            SZ[S["T"]] = np.load(pz); OUT["inputs"]["W7_slice_%s.npz" % key] = C.md5(pz)
    OUT["slices"] = {key: {kk: S.get(kk) for kk in ("T", "path", "present", "md5", "manifest_md5", "md5_matches_manifest", "bytes", "dtype", "keys_present",
                                                       "fields_vs_history_max", "n_sites", "R_max", "roots")} for key, S in SJ["slices"].items()}
    k_form_T = AJ[1]["T_form"]
    # ---------------- gates
    V = np.load(C.V2_NPZ); Bz = np.load(C.B5_NPZ)
    OUT["inputs"]["dust_root_history_v2.npz"] = C.md5(C.V2_NPZ); OUT["inputs"]["B5v2_cols_prod.npz"] = C.md5(C.B5_NPZ)
    gate = {}
    for k in (1, 2):
        Z = AZ[k]
        for tg in ("T12p125", "T15p000"):
            f = Z["%s_f" % tg]; r = {}
            r["launch_cols_vs_B5"] = float(np.abs(Z["%s_launch_cols" % tg] - Bz["cols_dust_%s_k%d" % (tg, k)]).max())
            for mine, theirs, wt in (("b16_all", "b16_all", False), ("b16_in_dense", "b16_in_dense", False), ("b16_in_applied", "b16_in_applied", False),
                                     ("b24_all", "b24_all", False), ("b16_out_dense", "b16_out_dense", False), ("beyond_Rt_dense", "beyond_Rt_dense", False),
                                     ("beyond_Rt_applied", "beyond_Rt_applied", False), ("norm", "norm", False), ("J16_all", "current_r16", True),
                                     ("J_outer_dense_all", "current_outer_dense", True), ("mcs_col_dense", "moving_cut_step_dense", True), ("Rt_dense", "Rt_dense", None)):
                x = Z["%s_%s" % (tg, mine)]; y = V["%s_k%d_%s" % (tg, k, theirs)]
                if wt is True:
                    x = np.where(np.isfinite(x).all(axis=1), x @ f, np.nan)
                ok = np.isfinite(x) & np.isfinite(y)
                r[mine] = {"max_abs_diff": float(np.abs(x[ok] - y[ok]).max()) if ok.any() else None, "nan_mismatch": int((np.isfinite(x) != np.isfinite(y)).sum())}
            gate["k%d_%s" % (k, tg)] = r
        ww = Z["held_full_w_whole"] if "held_full_w_whole" in Z.files else None
        if ww is not None:
            sets = AJ[k]["held_sets"]; i0 = 0
            for tg, nm, nc in sets:
                if nm == "all":
                    b5 = bins_of(Z["held_full_e"], ww[:, i0:i0 + nc])
                    gate["k%d_%s" % (k, tg)]["bins_end_vs_v2"] = float(np.abs(b5 - V["%s_k%d_bins_end" % (tg, k)]).max())
                i0 += nc
    OUT["gates"]["v2_and_B5"] = gate
    # ---------------- per kappa series, T12p125 (the formation launch) and T15p000
    for k in C.KAPPAS:
        Z = AZ[k]
        for tg in ("T12p125", "T15p000"):
            for nm in Z.files:
                if nm.startswith(tg + "_") and not ("state" in nm or "end_" in nm or "launch_cols" in nm):
                    SER["k%d_%s" % (k, nm)] = Z[nm]
    # norm drift, Chebyshev
    OUT["gates"]["norm_drift_max"] = {"k%d_%s" % (k, tg): float(np.nanmax(np.abs(AZ[k]["%s_norm" % tg] - AZ[k]["%s_norm" % tg][0]))) for k in C.KAPPAS for tg in ("T12p125", "T15p000")}
    OUT["gates"]["cheb_K_max"] = {"k%d" % k: AJ[k].get("cheb_K_max") for k in C.KAPPAS}
    # reproducibility floor: the end weights from the two launches (the same end state up to the propagation of the stored 15 slice)
    OUT["gates"]["end_b16_all_T12_vs_T15"] = {"k%d" % k: float(np.abs(AZ[k]["T12p125_b16_all"][-1] - AZ[k]["T15p000_b16_all"][-1]).max()) for k in C.KAPPAS}

    # ============ item 1: the same-set fixed-cut ledger
    P("## Item 1. Same-set fixed-cut ledger")
    it1 = {"record_level": {}, "slice_level": {}}
    T12 = AZ[1]["T12p125_T"]
    led = {}
    for cut in ("dense", "applied"):
        acc = None
        for k in C.KAPPAS:
            Z = AZ[k]; f = Z["T12p125_f"]
            L = ledger_from(Z["T12p125_T"], f, Z["T12p125_norm"], Z["T12p125_beyond_Rt_%s" % cut], Z["T12p125_mcs_col_%s" % cut])
            led[(cut, k)] = L
            for q in ("p_in", "T_j", "S_j", "D_j"):
                SER["ledger_%s_k%d_%s" % (cut, k, q)] = L[q]
        # per copy summed over kappa, and multiplicity-weighted
        for wname, wt in (("per_copy", {k: 1.0 for k in C.KAPPAS}), ("weighted_16kappa", MULT)):
            Lsum = {q: sum(wt[k] * led[(cut, k)][q] for k in C.KAPPAS) for q in ("p_in", "T_j", "S_j", "D_j")}
            for q in Lsum:
                SER["ledger_%s_%s_%s" % (cut, wname, q)] = Lsum[q]
            rows = []
            for i in range(len(SLICE_CLOCKS) - 1):
                ta, tb = SLICE_CLOCKS[i], SLICE_CLOCKS[i + 1]
                ja, jb = idx_of(T12, ta), idx_of(T12, tb)
                sl = slice(ja + 1, jb + 1)
                fin = np.isfinite(Lsum["T_j"][sl])
                rows.append({"T_a": ta, "T_b": tb, "steps": int(jb - ja), "rooted_steps": int(fin.sum()), "p_in_a": float(Lsum["p_in"][ja]), "p_in_b": float(Lsum["p_in"][jb]),
                             "sum_T_j": float(np.nansum(Lsum["T_j"][sl])), "sum_S_j": float(np.nansum(Lsum["S_j"][sl])), "max_abs_D_j": float(np.nanmax(np.abs(Lsum["D_j"][sl]))) if fin.any() else None,
                             "unrooted_steps": [float(T12[ja + 1 + q]) for q in np.nonzero(~fin)[0]]})
            win = []
            for j in range(idx_of(T12, 15.5) + 1, idx_of(T12, 16.125) + 1):
                win.append({"T_a": float(T12[j - 1]), "T_b": float(T12[j]), "R_a": float(AZ[1]["T12p125_Rt_%s" % cut][j - 1]), "R_b": float(AZ[1]["T12p125_Rt_%s" % cut][j]),
                            "T_j": float(Lsum["T_j"][j]), "S_j": float(Lsum["S_j"][j]), "D_j": float(Lsum["D_j"][j])})
            it1["record_level"]["%s_%s" % (cut, wname)] = {"between_slice_clocks": rows, "window_15p5_16p125": win,
                                                           "max_abs_D_j": float(np.nanmax(np.abs(Lsum["D_j"]))), "n_rooted_steps": int(np.isfinite(Lsum["T_j"]).sum()),
                                                           "sum_T_j_all": float(np.nansum(Lsum["T_j"])), "sum_S_j_all": float(np.nansum(Lsum["S_j"]))}
    # slice to slice on the stored slices (2048-site chain; the propagated set's columns compared)
    sl_clocks = [t for t in (15.0, 16.0, 20.0, 24.0, 25.0, 28.0, 30.0, 35.0, 40.0) if t in SZ]
    srows = []
    for cut in ("dense", "applied"):
        for i in range(len(sl_clocks) - 1):
            ta, tb = sl_clocks[i], sl_clocks[i + 1]
            Ra = SJ["slices"]["T%06.3f" % ta]["roots"]["%s_outer" % cut]; Rb = SJ["slices"]["T%06.3f" % tb]["roots"]["%s_outer" % cut]
            Za, Zb = SZ[ta], SZ[tb]; Rs = Za["R"]
            row = {"cut": cut, "T_a": ta, "T_b": tb, "R_a": Ra, "R_b": Rb, "per_kappa": {}}
            tot = {"p_in_a": 0.0, "p_in_b": 0.0, "T": 0.0, "S": 0.0, "D": 0.0}
            totw = dict((q, 0.0) for q in tot)
            for k in C.KAPPAS:
                f = Za["k%d_f" % k]; swa = Za["k%d_sitew" % k]; swb = Zb["k%d_sitew" % k]
                if Ra is None or Rb is None or swa.shape[1] == 0:
                    continue
                pa = f @ swa[Rs < Ra].sum(axis=0); pb = f @ swb[Rs < Rb].sum(axis=0)
                Tq = f @ (swb[Rs > Ra].sum(axis=0) - swa[Rs > Ra].sum(axis=0))
                Sq = f @ (swb[Rs < Rb].sum(axis=0) - swb[Rs < Ra].sum(axis=0))
                Dq = (pb - pa) + Tq - Sq
                row["per_kappa"][str(k)] = {"p_in_a": float(pa), "p_in_b": float(pb), "T": float(Tq), "S": float(Sq), "D": float(Dq)}
                for q, v_ in (("p_in_a", pa), ("p_in_b", pb), ("T", Tq), ("S", Sq), ("D", Dq)):
                    tot[q] += float(v_); totw[q] += MULT[k] * float(v_)
            row["per_copy"] = tot; row["weighted_16kappa"] = totw
            # the propagated set over the same interval (record level, summed)
            Lp = {q: sum(led[(cut, k)][q] for k in C.KAPPAS) for q in ("T_j", "S_j")}
            ja, jb = idx_of(T12, ta), idx_of(T12, tb)
            row["propagated_sum_T_j"] = float(np.nansum(Lp["T_j"][ja + 1: jb + 1])); row["propagated_sum_S_j"] = float(np.nansum(Lp["S_j"][ja + 1: jb + 1]))
            row["propagated_net_T_fixed_old_cut"] = None
            if Ra is not None:
                # the net transport through the fixed cut R_a over [ta, tb] on the propagated set (the slice formula on the propagated columns)
                net = 0.0
                for k in C.KAPPAS:
                    Z = AZ[k]; f = Z["T12p125_f"]
                    Sa_ = Z["T12p125_state_all_T%06.3f" % ta]; Sb_ = Z["T12p125_state_all_T%06.3f" % tb]
                    Rp = np.arange(1, Sa_.shape[0] // 2 + 1, dtype=float)
                    net += f @ (C.site_w(Sb_, len(Rp))[Rp > Ra].sum(axis=0) - C.site_w(Sa_, len(Rp))[Rp > Ra].sum(axis=0))
                row["propagated_net_T_fixed_old_cut"] = float(net)
            srows.append(row)
    it1["slice_level"] = srows
    # same-set check: the stored slices' dust columns against the propagated set at the same clock
    ss = []
    for t in sl_clocks:
        for k in C.KAPPAS:
            Z = AZ[k]; key = "T12p125_state_all_T%06.3f" % t
            if key not in Z.files:
                continue
            Sp = Z[key]; Ds = SZ[t]["k%d_cols" % k]; n = Ds.shape[0] // 2
            ov = np.abs((Ds.conj() * Sp[:2 * n]).sum(axis=0)) ** 2
            swd = np.abs(C.site_w(Sp[:2 * n], n) - SZ[t]["k%d_sitew" % k]).sum(axis=0)
            tail = C.site_w(Sp, Sp.shape[0] // 2)[n:].sum(axis=0)
            ss.append({"T": t, "kappa": k, "column_overlap_sq_min": float(ov.min()), "column_overlap_sq": ov.tolist(), "site_weight_L1_diff_max": float(swd.max()),
                       "propagated_weight_beyond_slice_chain_max": float(tail.max())})
    it1["same_set_check"] = ss
    OUT["items"]["1_ledger"] = it1
    for cut in ("dense", "applied"):
        r = it1["record_level"]["%s_per_copy" % cut]
        P("Record level, %s cut, per copy summed over kappa 1..5 (16 columns), from the formation record: %d rooted steps, max |D_j| %s; sum T_j %s, sum S_j %s" % (
            cut, r["n_rooted_steps"], g6(r["max_abs_D_j"], 3), g6(r["sum_T_j_all"]), g6(r["sum_S_j_all"])))
        P("| T_a | T_b | steps | rooted | p_in(a) | p_in(b) | sum T_j | sum S_j | max abs D_j | unrooted step ends |")
        P("|---|---|---|---|---|---|---|---|---|---|")
        for x in r["between_slice_clocks"]:
            P("| %s | %s | %d | %d | %s | %s | %s | %s | %s | %s |" % (g6(x["T_a"]), g6(x["T_b"]), x["steps"], x["rooted_steps"], g6(x["p_in_a"]), g6(x["p_in_b"]), g6(x["sum_T_j"]),
                                                                    g6(x["sum_S_j"]), g6(x["max_abs_D_j"], 2), (("%s..%s (%d)" % (g6(x["unrooted_steps"][0]), g6(x["unrooted_steps"][-1]), len(x["unrooted_steps"]))) if x["unrooted_steps"] else "-")))
        P("Window 15.5 to 16.125 (per copy): " + "; ".join("%s->%s R %s->%s T %s S %s" % (g6(w["T_a"]), g6(w["T_b"]), g6(w["R_a"], 6), g6(w["R_b"], 6), g6(w["T_j"]), g6(w["S_j"])) for w in r["window_15p5_16p125"]))
        rw = it1["record_level"]["%s_weighted_16kappa" % cut]
        P("Weighted (16 kappa) sums: T_j %s, S_j %s; window: %s" % (g6(rw["sum_T_j_all"]), g6(rw["sum_S_j_all"]), "; ".join("%s->%s T %s S %s" % (g6(w["T_a"]), g6(w["T_b"]), g6(w["T_j"]), g6(w["S_j"])) for w in rw["window_15p5_16p125"])))
        P("")
    P("Slice to slice (stored slices, per copy summed over kappa; the propagated set's sums over the same interval beside):")
    P("| cut | T_a | T_b | R_a | R_b | p_in(a) | p_in(b) | T | S | D | weighted T | weighted S | propagated sum T_j | propagated sum S_j | propagated net T at R_a |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for x in srows:
        pc = x["per_copy"]; w_ = x["weighted_16kappa"]
        P("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (x["cut"], g6(x["T_a"]), g6(x["T_b"]), g6(x["R_a"], 6), g6(x["R_b"], 6), g6(pc["p_in_a"]), g6(pc["p_in_b"]), g6(pc["T"]),
                                                                                       g6(pc["S"]), g6(pc["D"], 2), g6(w_["T"]), g6(w_["S"]), g6(x["propagated_sum_T_j"]), g6(x["propagated_sum_S_j"]), g6(x["propagated_net_T_fixed_old_cut"])))
    P("Same-set check (stored slice columns against the propagated set): min |<slice|propagated>|^2 per clock: " + "; ".join(
        "%s k%d %s" % (g6(x["T"]), x["kappa"], g6(x["column_overlap_sq_min"], 8)) for x in ss))
    P("")

    # ============ item 2: the Killing split on the held exterior block
    P("## Item 2. Killing split of the dust beyond 16 r_s on the held exterior block")
    it2 = {"slices": {}, "end_56_propagated": {}}
    for t in sorted(SZ):
        Zs = SZ[t]; key = "T%06.3f" % t; rec = SJ["slices"][key]["record"]
        r_s_rec = 2.0 * G * rec["E"]; Rmax = float(Zs["R"][-1])
        ent = {"r_s_record_2GE": r_s_rec, "R_max": Rmax, "per_kappa": {}}
        totB = np.zeros(5); totBw = np.zeros(5); totA = np.zeros(5); blockw = 0.0; blockww = 0.0; mpos_num = 0.0; mpos_den = 0.0
        th_b4_buf = {"1": np.zeros(5), "2": np.zeros(5)}; th_err = np.zeros(5)
        splitA = {}
        for k in C.KAPPAS:
            if ("k%d_block16_e" % k) not in Zs.files:
                continue
            f = Zs["k%d_f" % k]; eB = Zs["k%d_block16_e" % k]; wB = Zs["k%d_block16_w" % k] @ f
            b5 = bins_of(eB, wB); thr = HB.threshold_columns(eB, wB, M, B, r_s_rec, Rmax)
            eA = Zs["k%d_full_e" % k]; wA = Zs["k%d_full_w_ext16" % k] @ f; a5 = bins_of(eA, wA)
            num, den = C.mean_positive(eB, wB[:, None], M)
            ent["per_kappa"][str(k)] = {"block_weight": float(wB.sum()), "bins5_block": b5.tolist(), "bins5_full_alt": a5.tolist(),
                                        "buffer1_bins5": thr["buffers"]["1"]["bins5"], "buffer2_bins5": thr["buffers"]["2"]["bins5"], "threshold_term_bins5": thr["threshold_error"]["bins5"],
                                        "mean_E_above_m_over_m": float(num[0] / den[0] / M) if den[0] > 0 else None}
            totB += b5; totBw += MULT[k] * b5; totA += a5; blockw += float(wB.sum()); blockww += MULT[k] * float(wB.sum())
            th_b4_buf["1"] += np.array(thr["buffers"]["1"]["bins5"]); th_b4_buf["2"] += np.array(thr["buffers"]["2"]["bins5"]); th_err += np.array(thr["threshold_error"]["bins5"])
            mpos_num += MULT[k] * float(num[0]); mpos_den += MULT[k] * float(den[0])
            inband = (HB.bin_index(eA, M, B) == 3)
            for Rc in (2.0, 16.0):
                win = Zs["k%d_full_win_%g" % (k, Rc)]
                for thv in (0.9, 0.99, 0.999):
                    ext = inband & (win < thv)
                    splitA.setdefault((Rc, thv), 0.0)
                    splitA[(Rc, thv)] += float(wA[ext].sum())
        ent.update({"per_copy_block_weight": blockw, "per_copy_bins5_block": totB.tolist(), "weighted_block_weight": blockww, "weighted_bins5_block": totBw.tolist(),
                    "per_copy_bins5_full_alt": totA.tolist(), "per_copy_buffer1_bins5": th_b4_buf["1"].tolist(), "per_copy_buffer2_bins5": th_b4_buf["2"].tolist(),
                    "per_copy_threshold_term_bins5": th_err.tolist(), "weighted_mean_E_above_m_over_m": (mpos_num / mpos_den / M) if mpos_den > 0 else None,
                    "per_copy_full_alt_extended_mb": {"Rcut%g_thr%g" % kk: v for kk, v in splitA.items()}})
        it2["slices"][key] = ent
    # at 56 on the propagated set (the formation launch's whole dust and the 15 launch's)
    for tg in ("T12p125", "T15p000"):
        totB = np.zeros(5); totBw = np.zeros(5); totA = np.zeros(5); bw = 0.0; bww = 0.0; mnum = 0.0; mden = 0.0; terr = np.zeros(5); per = {}
        rec = tab[56.0]; r_s_rec = 2.0 * G * rec["E"]
        for k in C.KAPPAS:
            Z = AZ[k]; J = AJ[k]
            if "held_block16_e" not in Z.files:
                continue
            i0 = 0
            for tg2, nm, nc in J["held_sets"]:
                if tg2 == tg and nm == "all":
                    f = Z["%s_f" % tg]; eB = Z["held_block16_e"]; wB = Z["held_block16_w"][:, i0:i0 + nc] @ f
                    b5 = bins_of(eB, wB); thr = HB.threshold_columns(eB, wB, M, B, r_s_rec, float(J["n_sites"]))
                    eA = Z["held_full_e"]; wA = Z["held_full_w_ext16"][:, i0:i0 + nc] @ f; a5 = bins_of(eA, wA)
                    num, den = C.mean_positive(eB, wB[:, None], M)
                    per[str(k)] = {"block_weight": float(wB.sum()), "bins5_block": b5.tolist(), "bins5_full_alt": a5.tolist(), "threshold_term_bins5": thr["threshold_error"]["bins5"],
                                   "mean_E_above_m_over_m": float(num[0] / den[0] / M) if den[0] > 0 else None}
                    totB += b5; totBw += MULT[k] * b5; totA += a5; bw += float(wB.sum()); bww += MULT[k] * float(wB.sum()); terr += np.array(thr["threshold_error"]["bins5"])
                    mnum += MULT[k] * float(num[0]); mden += MULT[k] * float(den[0])
                i0 += nc
        it2["end_56_propagated"][tg] = {"per_kappa": per, "per_copy_block_weight": bw, "per_copy_bins5_block": totB.tolist(), "weighted_block_weight": bww,
                                        "weighted_bins5_block": totBw.tolist(), "per_copy_bins5_full_alt": totA.tolist(), "per_copy_threshold_term_bins5": terr.tolist(),
                                        "weighted_mean_E_above_m_over_m": (mnum / mden / M) if mden > 0 else None}
    OUT["items"]["2_killing_split"] = it2
    P("| clock | source | block weight per copy | bins (-inf,-b] (-b,-m] (-m,m] (m,b] (b,inf) per copy | threshold term per bin | weighted (16 kappa) block | weighted (m,b] | mean E above m (m), weighted | full-operator alt bins per copy |")
    P("|---|---|---|---|---|---|---|---|---|")
    for key, ent in it2["slices"].items():
        P("| %s | slice | %s | %s | %s | %s | %s | %s | %s |" % (key, g6(ent["per_copy_block_weight"]), [g6(x, 4) for x in ent["per_copy_bins5_block"]], [g6(x, 2) for x in ent["per_copy_threshold_term_bins5"]],
                                                         g6(ent["weighted_block_weight"]), g6(ent["weighted_bins5_block"][3]), g6(ent["weighted_mean_E_above_m_over_m"], 4), [g6(x, 4) for x in ent["per_copy_bins5_full_alt"]]))
    for tg, ent in it2["end_56_propagated"].items():
        P("| 56 | propagated from %s | %s | %s | %s | %s | %s | %s | %s |" % (tg, g6(ent["per_copy_block_weight"]), [g6(x, 4) for x in ent["per_copy_bins5_block"]], [g6(x, 2) for x in ent["per_copy_threshold_term_bins5"]],
                                                                       g6(ent["weighted_block_weight"]), g6(ent["weighted_bins5_block"][3]), g6(ent["weighted_mean_E_above_m_over_m"], 4), [g6(x, 4) for x in ent["per_copy_bins5_full_alt"]]))
    P("")

    # ============ item 3: the never-inside capacity at the record's charge
    P("## Item 3. Never-inside capacity at the record's charge")
    rows3 = []
    Ts_sorted = sorted(tab)
    for t in SLICE_CLOCKS:
        rec = tab.get(round(t, 6))
        if rec is None or rec["Rt"] is None:
            rows3.append({"T": t, "root": None}); continue
        Rmax_hist = max(tab[x]["Rt"] for x in Ts_sorted if x <= t + 1e-9 and tab[x]["Rt"] is not None)
        Mroot = rec["Rt"] / (2 * G); Mmax = Rmax_hist / (2 * G)
        # the count beyond 16 r_s: slices where present, else the propagated formation-launch set
        cnt = 0.0; cntw = 0.0; src = None
        if t in SZ:
            src = "slice"
            for k in C.KAPPAS:
                Zs = SZ[t]; f = Zs["k%d_f" % k]; Rs = Zs["R"]; c = float(f @ Zs["k%d_sitew" % k][Rs > 16 * RS].sum(axis=0)); cnt += c; cntw += MULT[k] * c
        else:
            src = "propagated from 12.125"
            for k in C.KAPPAS:
                Z = AZ[k]; j = idx_of(Z["T12p125_T"], t); c = float(Z["T12p125_f"] @ Z["T12p125_b16_all"][j]); cnt += c; cntw += MULT[k] * c
        row = {"T": t, "E": rec["E"], "E_over_E0": rec["E"] / E0, "R_t": rec["Rt"], "R_t_over_2G": Mroot, "E_minus_root_charge": rec["E"] - Mroot,
               "C_m": (rec["E"] - Mroot) / M, "C_1p2m": (rec["E"] - Mroot) / (1.2 * M), "C_m_emin": (rec["E"] - Mroot) / (M * EMIN_P05),
               "C_m_E0": (E0 - Mroot) / M, "R_max_history": Rmax_hist, "C_m_Rmax": (rec["E"] - Mmax) / M, "C_m_emin_E0_Rmax": (E0 - Mmax) / (M * EMIN_P05),
               "count_beyond_16rs_per_copy": cnt, "count_beyond_16rs_weighted": cntw, "count_source": src}
        row["fits_C_m"] = cntw <= row["C_m"]; row["fits_C_1p2m"] = cntw <= row["C_1p2m"]; row["fits_C_m_Rmax"] = cntw <= row["C_m_Rmax"]; row["fits_C_m_emin_E0_Rmax"] = cntw <= row["C_m_emin_E0_Rmax"]
        rows3.append(row)
    OUT["items"]["3_capacity"] = rows3
    P("| T/r_s | E | E/E_0 | R_t | E - R_t/(2G) | C at m | C at 1.2 m | C at m e_min | C at m with E_0 | R_max so far | C at m with R_max | C at m e_min, E_0, R_max | beyond 16 r_s (16 kappa) | per copy | source | fits C at m |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows3:
        if r.get("root") is None and "E" not in r:
            P("| %s | no dense root at the record |" % g6(r["T"])); continue
        P("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (g6(r["T"]), g6(r["E"], 8), g6(r["E_over_E0"], 4), g6(r["R_t"], 7), g6(r["E_minus_root_charge"]), g6(r["C_m"]), g6(r["C_1p2m"]),
                                                                                          g6(r["C_m_emin"]), g6(r["C_m_E0"]), g6(r["R_max_history"], 6), g6(r["C_m_Rmax"]), g6(r["C_m_emin_E0_Rmax"]),
                                                                                          g6(r["count_beyond_16rs_weighted"]), g6(r["count_beyond_16rs_per_copy"]), r["count_source"], g6(r["fits_C_m"])))
    P("")

    # ============ item 4: the inside-origin split at the root records 12.125 and 15
    P("## Item 4. Inside-origin split at the root records 12.125 and 15 (at 56)")
    it4 = {}
    for tg in ("T12p125", "T15p000"):
        for cut in ("dense", "applied"):
            ent = {"per_kappa": {}}
            for cr in ("b16", "b24"):
                tot = {"all": 0.0, "in": 0.0, "out": 0.0, "cross": 0.0, "lower": 0.0, "upper": 0.0}; totw = dict((q, 0.0) for q in tot); viol = [0, 0]
                for k in C.KAPPAS:
                    Z = AZ[k]; f = Z["%s_f" % tg]; norm = Z["%s_norm" % tg]
                    al = Z["%s_%s_all" % (tg, cr)][-1]; bi = Z["%s_%s_in_%s" % (tg, cr, cut)][-1]; bo = Z["%s_%s_out_%s" % (tg, cr, cut)][-1]
                    eps = Z["%s_beyond_Rt_%s" % (tg, cut)][0] / norm[0]
                    p = al / norm[-1]; sp_ = np.sqrt(np.clip(p, 0, None)); se = np.sqrt(np.clip(eps, 0, None))
                    lo = np.clip(sp_ - se, 0, None) ** 2; up = np.minimum((sp_ + se) ** 2, 1 - eps)
                    q = {"all": f @ al, "in": f @ bi, "out": f @ bo, "cross": f @ (al - bi - bo), "lower": f @ lo, "upper": f @ up}
                    viol[0] += int(np.sum(lo > bi / norm[-1] + 1e-9)); viol[1] += int(np.sum(bi / norm[-1] > up + 1e-9))
                    ent["per_kappa"].setdefault(str(k), {})[cr] = {kk: float(v) for kk, v in q.items()}
                    ent["per_kappa"][str(k)]["eps_early_per_column"] = eps.tolist()
                    for kk in tot:
                        tot[kk] += float(q[kk]); totw[kk] += MULT[k] * float(q[kk])
                ent[cr] = {"per_copy": tot, "weighted_16kappa": totw, "violations_lower_gt_in": viol[0], "violations_in_gt_upper": viol[1]}
            ent["launch_root"] = AJ[1]["launches"][tg]["roots"]["%s_outer" % cut]
            ent["weight_inside_root_at_launch_per_copy"] = float(sum(AZ[k]["%s_f" % tg] @ (AZ[k]["%s_norm" % tg][0] - AZ[k]["%s_beyond_Rt_%s" % (tg, cut)][0]) for k in C.KAPPAS))
            it4["%s_%s" % (tg, cut)] = ent
    OUT["items"]["4_inside_origin"] = it4
    P("| root record | cut | launch root (a) | inside at launch (per copy) | cut at 56 | all | from inside | from outside | cross | certificate lower | upper | weighted all | weighted inside | violations |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, ent in it4.items():
        for cr in ("b16", "b24"):
            x = ent[cr]["per_copy"]; w_ = ent[cr]["weighted_16kappa"]
            P("| %s | %s | %s | %s | %s r_s | %s | %s | %s | %s | %s | %s | %s | %s | %d/%d |" % (key.split("_")[0], key.split("_")[1], g6(ent["launch_root"], 8), g6(ent["weight_inside_root_at_launch_per_copy"]), cr[1:],
                                                                                         g6(x["all"]), g6(x["in"]), g6(x["out"]), g6(x["cross"]), g6(x["lower"]), g6(x["upper"]), g6(w_["all"]), g6(w_["in"]),
                                                                                         ent[cr]["violations_lower_gt_in"], ent[cr]["violations_in_gt_upper"]))
    P("Per kappa (dense, 16 r_s; all / in / out / cross): " + " | ".join(
        "%s k%s %s/%s/%s/%s" % (tg, k, g6(it4["%s_dense" % tg]["per_kappa"][k]["b16"]["all"], 4), g6(it4["%s_dense" % tg]["per_kappa"][k]["b16"]["in"], 4),
                               g6(it4["%s_dense" % tg]["per_kappa"][k]["b16"]["out"], 4), g6(it4["%s_dense" % tg]["per_kappa"][k]["b16"]["cross"], 3)) for tg in ("T12p125", "T15p000") for k in map(str, C.KAPPAS)))
    P("")

    # ============ item 5: the escape certificate and the delay
    P("## Item 5. Escape certificate (formation record 12.125) and delay")
    it5 = {"formation_T": k_form_T}
    for cut in ("dense", "applied"):
        den = 0.0; denw = 0.0; num = 0.0; numw = 0.0; numb = {"1": 0.0, "2": 0.0}; terr = 0.0; nb16 = 0.0; altA = {}; per = {}
        rec = tab[56.0]; r_s_rec = 2.0 * G * rec["E"]
        for k in C.KAPPAS:
            Z = AZ[k]; J = AJ[k]; f = Z["T12p125_f"]
            din = float(f @ (Z["T12p125_norm"][0] - Z["T12p125_beyond_Rt_%s" % cut][0]))
            i0 = 0; nk = None
            for tg2, nm, nc in J["held_sets"]:
                if tg2 == "T12p125" and nm == "in_%s" % cut and "held_block16_e" in Z.files:
                    eB = Z["held_block16_e"]; wB = Z["held_block16_w"][:, i0:i0 + nc] @ f
                    b5 = bins_of(eB, wB); thr = HB.threshold_columns(eB, wB, M, B, r_s_rec, float(J["n_sites"]))
                    nk = float(b5[3]); nb = float(wB.sum())
                    numb["1"] += thr["buffers"]["1"]["bins5"][3]; numb["2"] += thr["buffers"]["2"]["bins5"][3]; terr += thr["threshold_error"]["bins5"][3]
                    eA = Z["held_full_e"]; wA = Z["held_full_w_ext16"][:, i0:i0 + nc] @ f; inband = HB.bin_index(eA, M, B) == 3
                    for Rc in (2.0, 16.0):
                        win = Z["held_full_win_%g" % Rc]
                        for thv in (0.9, 0.99, 0.999):
                            altA.setdefault("Rcut%g_thr%g" % (Rc, thv), 0.0)
                            altA["Rcut%g_thr%g" % (Rc, thv)] += float(wA[inband & (win < thv)].sum())
                    altA.setdefault("full_operator_mb_all", 0.0); altA["full_operator_mb_all"] += float(wA[inband].sum())
                    per[str(k)] = {"inside_at_formation": din, "beyond16_at_56": nb, "mb_block_at_56": nk, "bins5_block": b5.tolist(), "fraction": nk / din if din > 0 else None}
                    nb16 += nb
                i0 += nc
            den += din; denw += MULT[k] * din
            if nk is not None:
                num += nk; numw += MULT[k] * nk
        # the currents of the inside-origin content through 16 r_s at the sampled clocks
        Jt = {}
        for t in SAMPLED + (40.0, 44.0):
            j = idx_of(AZ[1]["T12p125_T"], t)
            Jt[str(t)] = {"per_copy": float(sum(AZ[k]["T12p125_f"] @ AZ[k]["T12p125_J16_in_%s" % cut][j] for k in C.KAPPAS)),
                          "weighted": float(sum(MULT[k] * (AZ[k]["T12p125_f"] @ AZ[k]["T12p125_J16_in_%s" % cut][j]) for k in C.KAPPAS)),
                          "per_kappa": [float(AZ[k]["T12p125_f"] @ AZ[k]["T12p125_J16_in_%s" % cut][j]) for k in C.KAPPAS],
                          "whole_dust_per_copy": float(sum(AZ[k]["T12p125_f"] @ AZ[k]["T12p125_J16_all"][j] for k in C.KAPPAS))}
        jser = sum(AZ[k]["T12p125_J16_in_%s" % cut] @ AZ[k]["T12p125_f"] for k in C.KAPPAS)
        T = AZ[1]["T12p125_T"]; j48 = idx_of(T, 48.0)
        outward_all_sampled = all(Jt[str(t)]["per_copy"] > 0 for t in SAMPLED)
        outward_per_kappa_sampled = {str(k): all(Jt[str(t)]["per_kappa"][k - 1] > 0 for t in SAMPLED) for k in C.KAPPAS}
        # the half-arrival of the inside-origin content at 16 r_s
        F = sum(AZ[k]["T12p125_b16_in_%s" % cut] @ AZ[k]["T12p125_f"] for k in C.KAPPAS)
        Fw = sum(MULT[k] * (AZ[k]["T12p125_b16_in_%s" % cut] @ AZ[k]["T12p125_f"]) for k in C.KAPPAS)
        Fall = sum(AZ[k]["T12p125_b16_all"] @ AZ[k]["T12p125_f"] for k in C.KAPPAS)

        def half(Fs):
            h_ = 0.5 * Fs[-1]; j = next((i for i in range(len(Fs)) if Fs[i] >= h_), None)
            if j is None or j == 0:
                return None
            return float(T[j - 1] + (h_ - Fs[j - 1]) / (Fs[j] - Fs[j - 1]) * (T[j] - T[j - 1]))
        Th = half(F); Thw = half(Fw); Tha = half(Fall)
        cum = np.r_[0.0, np.cumsum(0.5 * (jser[1:] + jser[:-1]) * np.diff(T) * RS)]
        Thc = None
        if cum[-1] > 0:
            hc = 0.5 * cum[-1]; jc = next((i for i in range(len(cum)) if cum[i] >= hc), None)
            if jc:
                Thc = float(T[jc - 1] + (hc - cum[jc - 1]) / (cum[jc] - cum[jc - 1]) * (T[jc] - T[jc - 1]))
        per_k_half = {}
        for k in C.KAPPAS:
            Fk = AZ[k]["T12p125_b16_in_%s" % cut] @ AZ[k]["T12p125_f"]; tk = half(Fk)
            per_k_half[str(k)] = {"T_half": tk, "delay": (tk - k_form_T) if tk else None, "F56": float(Fk[-1]), "Fmax": float(Fk.max()), "T_at_Fmax": float(T[int(np.argmax(Fk))])}
        SER["escape_%s_F_in_per_copy" % cut] = F; SER["escape_%s_J16_in_per_copy" % cut] = jser; SER["escape_%s_cum_J16_in" % cut] = cum; SER["escape_T"] = T
        it5[cut] = {"inside_at_formation_per_copy": den, "inside_at_formation_weighted": denw, "beyond16_at_56_from_inside_per_copy": nb16,
                    "mb_block_at_56_per_copy": num, "mb_block_at_56_weighted": numw,
                    "fraction_per_copy": num / den if den > 0 else None, "fraction_weighted": numw / denw if denw > 0 else None,
                    "fraction_buffer1_per_copy": numb["1"] / den, "fraction_buffer2_per_copy": numb["2"] / den, "threshold_term_per_copy": terr / den,
                    "alt_full_operator_extended_mb_fraction_per_copy": {kk: v / den for kk, v in altA.items()},
                    "per_kappa": per, "currents_J16_inside_origin": Jt, "outward_at_all_sampled_clocks_per_copy": outward_all_sampled,
                    "outward_at_all_sampled_clocks_per_kappa": outward_per_kappa_sampled,
                    "records_48_to_56_outward_per_copy": int(np.sum(jser[j48:] > 0)), "records_48_to_56": int(len(jser) - j48),
                    "half_arrival_T": Th, "delay_rs": (Th - k_form_T) if Th else None, "half_arrival_T_weighted": Thw, "delay_rs_weighted": (Thw - k_form_T) if Thw else None,
                    "half_arrival_cumulative_current_T": Thc, "delay_cumulative_current_rs": (Thc - k_form_T) if Thc else None,
                    "half_arrival_whole_dust_T": Tha, "F56_per_copy": float(F[-1]), "Fmax_per_copy": float(F.max()), "T_at_Fmax": float(T[int(np.argmax(F))]),
                    "cum_current_56_per_copy": float(cum[-1]), "per_kappa_half_arrival": per_k_half}
    OUT["items"]["5_escape"] = it5
    for cut in ("dense", "applied"):
        e = it5[cut]
        P("%s cut: inside at formation per copy %s (weighted %s); from-inside beyond 16 r_s at 56 %s; in (m, b] on the held block %s (weighted %s); fraction per copy %s, weighted %s; buffer 1 %s, buffer 2 %s, threshold term %s" % (
            cut, g6(e["inside_at_formation_per_copy"]), g6(e["inside_at_formation_weighted"]), g6(e["beyond16_at_56_from_inside_per_copy"]), g6(e["mb_block_at_56_per_copy"]), g6(e["mb_block_at_56_weighted"]),
            g6(e["fraction_per_copy"]), g6(e["fraction_weighted"]), g6(e["fraction_buffer1_per_copy"]), g6(e["fraction_buffer2_per_copy"]), g6(e["threshold_term_per_copy"], 2)))
        P("  full-operator alternative (extended (m,b] eigenvectors): " + ", ".join("%s %s" % (kk, g6(v)) for kk, v in e["alt_full_operator_extended_mb_fraction_per_copy"].items()))
        P("  J16 of the inside-origin content at the sampled clocks (per copy): " + ", ".join("%s %s" % (t, g6(v["per_copy"], 4)) for t, v in e["currents_J16_inside_origin"].items())
          + "; outward at all 5: %s; per kappa %s; records 48..56 outward %d of %d" % (e["outward_at_all_sampled_clocks_per_copy"], e["outward_at_all_sampled_clocks_per_kappa"], e["records_48_to_56_outward_per_copy"], e["records_48_to_56"]))
        P("  half-arrival at 16 r_s: T %s (delay %s r_s); weighted %s; by the cumulative current %s; whole dust %s; F(56) %s, max %s at %s" % (
            g6(e["half_arrival_T"]), g6(e["delay_rs"]), g6(e["delay_rs_weighted"]), g6(e["delay_cumulative_current_rs"]), g6(e["half_arrival_whole_dust_T"]), g6(e["F56_per_copy"]), g6(e["Fmax_per_copy"]), g6(e["T_at_Fmax"])))
        P("  per kappa: " + "; ".join("k%s inside %s, mb %s, fraction %s, delay %s" % (k, g6(v["inside_at_formation"], 4), g6(v["mb_block_at_56"], 4), g6(v["fraction"], 4),
                                                                                       g6(e["per_kappa_half_arrival"][k]["delay"], 4)) for k, v in e["per_kappa"].items()))
    P("")
    P("## Gates")
    for kk, v in OUT["gates"].items():
        P("%s: %s" % (kk, json.dumps(v, default=C.jdefault)[:3000]))
    C.dump_json(OUT, os.path.join(d, "W7_compose_v1.json"))
    C.save_npz({k: np.asarray(v, float) for k, v in SER.items()}, os.path.join(d, "W7_compose_v1.npz"))
    with open(os.path.join(d, "W7_compose_v1_tables.txt"), "w") as fh:
        fh.write("\n".join(TXT) + "\n")
    print("\n".join(TXT))


if __name__ == "__main__":
    main()
