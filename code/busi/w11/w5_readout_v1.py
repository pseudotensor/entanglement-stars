#!/usr/bin/env python
"""w5_readout_v1.py (reviewer A, the busi lane; W5 of WEB40I_TASKS/busi.md): the readout of the 6 heavy-sea truncation runs (kappa_sea 10,
14 and 20 at tau 8 and tau 4 on the (0.2, 64) sigma/a 4 member from the common preparation to T/r_s 14), box CPU.

For each run it prints, with the source of every number:
 (1) the formation clock and radius, from the records json (collapse_<tag>.json, "records"): the first accepted record whose
     "trapping_zeros" is non-empty gives "T_over_rs" (the formation clock), "R_t" (the outermost zero, in a and over r_s), "R_t_over_2gE",
     "r_in", "min_N_minus_absv", "E_over_E0", "continuation_label"; the record before it brackets the clock with its "min_N_minus_absv";
     the label "kappa_sea K, tau tau" prints with it (the gate: the formation clock's channel label);
 (2) the pair totals per channel at T/r_s 14, from the complex128 slice states/<tag>/orbitals_<tag>_T0014.00.npz: per channel kappa the
     arrays sea_<kappa> (the propagated sea's columns S), R2_<kappa> (the continued reference's columns), kappas and weights (16 kappa); with
     h the channel's bare chain (engine_vacflow_v1.chain_sparse(R, m, kappa), the hs of Chan.__init__, the operator of dens(ch.h, .)) and
     h_m = (1/2){N, h} + (1/2){v, Q} at the slice's N_applied, v_applied: the energy in the engine's own source convention
     w (Tr[h P_sea] - Tr[h P_R2]) (the per-channel part of the record's E_pairs_shells), the pair count w (n_sea - ||R2^dag S||_F^2), and the
     Killing form w Tr[h_m (P_sea - P_R2)]; beside them the records' "E_pairs_shells", "pairs_inside_Rt", "pairs_outside_Rt" at the formation
     record and at 14 and the fields row's eps_pairs sum at 14 (collapse_<tag>_fields.npz, "T", "eps_pairs"); n_sea per channel prints (W13);
 (3) the inner-shell share above pi/4 at 14: the dust columns dust_<kappa> with occupations f_<kappa> and the weights; the inner shell is
     the dust inside R_50, the radius enclosing half of the dust energy of the fields row at 14 (eps_tot - eps_pairs, the engine's eps_dust);
     the share is the number-weighted fraction of the inner-shell dust (each column restricted to the sites R < R_50) on momenta with
     cos(2 a k) < 0, that is pi/4 < |a k| < 3 pi/4 (the engine's quarter-zone selector |sin a k| > sin(pi/4) of s11's per-orbital readout),
     by the FFT of each column's 2 components, per channel and in total, the whole dust's share beside it;
 (4) the 16 kappa E_kappa tail at 14: per channel E_kappa^pairs = Tr[|h_m| (P_sea - P_R2)^2] (the formed-energy majorant of the sea's
     deviation from its continued reference on the record's clocked operator), E_kappa^heavy = Tr[|h_m| (C - P_-(h_m))^2] with
     C = P_sea + sum_j f_j |Phi_j><Phi_j| the whole heavy one-copy covariance and P_-(h_m) the instantaneous Killing vacuum, and the Killing
     excess Tr[h_m (C - P_-)]; each weighted by 16 kappa; the cumulative sum from the top channel, the top channel's share and the share of
     the channels above 10 (the matched set); the eigendecomposition of h_m is recomputed here (scipy eigh of the 2 nsites dense matrix)
     and checked against the slice's stored e_<kappa> and mask_<kappa> (the mask against the negative set: the pockets);
 (5) the common-preparation check across the runs: the json head's N_B, E0, passes, f_kin; the fields npz's N_cinit, v_cinit against the
     first run's; the T = 0 slice's dust catalog per channel (columns, occupations, the projectors' overlap) where present;
and the cross-run tables per tau: the formation, the pair totals, the inner-shell share, the tail, and the channels 1 to 10 of every run
against the kappa_sea 10 run (the back-reaction of the added channels on the first 10).
Usage (box CPU, conda env qi2):
  python w5_readout_v1.py --code-dir DIR (--root ROOT [--members a,b,...] | --run RUNDIR/TAG:STATESDIR ...) [--out-dir DIR] [--threads 16]
  ROOT holds runs/<tag>/collapse_<tag>.json and collapse_<tag>_fields.npz and states/<tag>/orbitals_<tag>_T0014.00.npz (and _T0000.00.npz),
  the launcher's layout copied from the pods; DIR holds engine_vacflow_v1.py (the chain)."""
import argparse
import json
import math
import os
import sys

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

W5_MEMBERS = [("W5_k10_tau8", 10, 8), ("W5_k14_tau8", 14, 8), ("W5_k20_tau8", 20, 8), ("W5_k10_tau4", 10, 4), ("W5_k14_tau4", 14, 4), ("W5_k20_tau4", 20, 4)]
T_READ = 14.0
MATCHED_K = 10


def tag_of(name, K, tau):
    return "lad_%s_m0p2_rs64_n2048_R04_k%d_tau%d_ell4_prep4_A_R2_avg_plain_v36d" % (name, K, tau)


def rec_at(recs, t, rs, tol=1e-6):
    c = [r for r in recs if abs(r["T"] / rs - t) < tol]
    return c[0] if c else None


def formation(recs, rs):
    for i, r in enumerate(recs):
        if r.get("trapping_zeros"):
            p = recs[i - 1] if i > 0 else None
            return {"T_over_rs": r["T"] / rs, "R_t": r.get("R_t"), "R_t_over_rs": (r["R_t"] / rs) if r.get("R_t") else None, "R_t_over_2gE": r.get("R_t_over_2gE"),
                    "r_in": r.get("r_in"), "zeros": r.get("trapping_zeros"), "min_N_minus_absv": r.get("min_N_minus_absv"), "E_over_E0": r.get("E_over_E0"),
                    "E_pairs_shells": r.get("E_pairs_shells"), "pairs_inside_Rt": r.get("pairs_inside_Rt"), "pairs_outside_Rt": r.get("pairs_outside_Rt"),
                    "previous_T_over_rs": (p["T"] / rs) if p else None, "previous_min_N_minus_absv": (p.get("min_N_minus_absv") if p else None),
                    "continuation_label": r.get("continuation_label")}
    last = recs[-1] if recs else None
    return {"T_over_rs": None, "note": ("no zero of N - |v| on the records to T/r_s %.4f (min(N - |v|) %s at the last record)" % (last["T"] / rs, last.get("min_N_minus_absv"))) if last else "no records"}


def tr_form(A, X):
    """Re Tr[X^dag A X] for a sparse A and dense columns X."""
    return float(np.real(np.sum(np.conj(X) * (A @ X))))


def clocked(hs, Qs, N, v):
    """h_m = (1/2){N, h} + (1/2){v, Q}: Chan.clocked_dense's operator as a sparse matrix."""
    Nd = sp.diags(np.repeat(np.asarray(N, float), 2)); Vd = sp.diags(np.repeat(np.asarray(v, float), 2))
    return (0.5 * (Nd @ hs + hs @ Nd) + 0.5 * (Vd @ Qs + Qs @ Vd)).tocsr()


def quarter_share(cols, f, i_cut, nsites, sel):
    """(above, total): the number-weighted weight of the columns (restricted to the sites below i_cut when given) on momenta with cos(2 a k) < 0."""
    above = tot = 0.0
    for j in range(cols.shape[1]):
        c = cols[:, j].copy()
        if i_cut is not None:
            c[2 * i_cut:] = 0.0
        p2 = np.abs(np.fft.fft(c[0::2])) ** 2 + np.abs(np.fft.fft(c[1::2])) ** 2
        above += f[j] * float(p2[sel].sum()); tot += f[j] * float(p2.sum())
    return above, tot


def slice_analysis(Z, Zf, m, n, code_dir):
    sys.path.insert(0, code_dir)
    import engine_vacflow_v1 as VF
    T = float(Z["T"]); rs = float(Z["rs"])
    N_app = np.asarray(Z["N_applied"], float); v_app = np.asarray(Z["v_applied"], float); R = np.asarray(Z["R"], float)
    kappas = [int(k) for k in np.asarray(Z["kappas"])]; weights = [float(x) for x in np.asarray(Z["weights"])]
    Tf = np.asarray(Zf["T"], float); i_row = int(np.argmin(np.abs(Tf - T)))
    eps_pairs_row = np.asarray(Zf["eps_pairs"][i_row], float); eps_dust_row = np.asarray(Zf["eps_tot"][i_row], float) - eps_pairs_row
    cum = np.cumsum(eps_dust_row)
    i50 = int(min(np.searchsorted(cum, 0.5 * cum[-1]), n - 1)); R50 = float(R[i50])
    p = 2.0 * np.pi * np.fft.fftfreq(n); sel = np.abs(np.sin(p)) > np.sin(np.pi / 4.0)
    out = {"T_over_rs": T / rs, "fields_row_T_over_rs": float(Tf[i_row] / rs), "R_50": R50, "i_50": i50, "eps_pairs_row_sum": float(eps_pairs_row.sum()), "eps_dust_row_sum": float(eps_dust_row.sum()),
           "channels": {}}
    for k, w in zip(kappas, weights):
        hs, Qs = VF.chain_sparse(R, m, float(k)); hs = sp.csr_matrix(hs); Qs = sp.csr_matrix(Qs)
        S = np.asarray(Z["sea_%d" % k], np.complex128); Rf = np.asarray(Z["R2_%d" % k], np.complex128)
        D = np.asarray(Z["dust_%d" % k], np.complex128); f = np.asarray(Z["f_%d" % k], float)
        n_sea = int(S.shape[1])
        c = {"w": w, "n_sea": n_sea, "n_R2": int(Rf.shape[1]), "dust_columns": int(D.shape[1]), "dust_particles": float(w * f.sum()), "dust_f": [float(x) for x in f]}
        c["E_pairs_bare"] = w * (tr_form(hs, S) - tr_form(hs, Rf))
        c["n_pairs"] = w * (n_sea - float(np.linalg.norm(Rf.conj().T @ S, "fro") ** 2))
        hm = clocked(hs, Qs, N_app, v_app)
        c["E_pairs_killing"] = w * (tr_form(hm, S) - tr_form(hm, Rf))
        e, U = sla.eigh(hm.toarray(), driver="evd")
        e_st = np.asarray(Z["e_%d" % k], float)
        c["eig_vs_stored_max"] = float(np.max(np.abs(np.sort(e) - np.sort(e_st)))) if e_st.size == e.size else None
        mask = np.unpackbits(np.asarray(Z["mask_%d" % k], np.uint8))[:2 * n].astype(bool)
        c["mask_occupied"] = int(mask.sum()); c["n_negative"] = int((e < 0).sum())
        c["pockets_occupied_positive"] = int(np.sum(mask & (e > 0))); c["holes_empty_negative"] = int(np.sum(~mask & (e < 0)))
        As = U.conj().T @ S; Bs = U.conj().T @ Rf
        Dm = As @ As.conj().T - Bs @ Bs.conj().T
        ae = np.abs(e)
        c["E_maj_pairs"] = float(np.sum(ae[:, None] * (np.abs(Dm) ** 2)))
        del Bs
        Cm = As @ As.conj().T
        if D.shape[1]:
            Ad = U.conj().T @ D
            Cm = Cm + (Ad * f[None, :]) @ Ad.conj().T
        Cm[np.diag_indices_from(Cm)] -= (e < 0).astype(float)
        c["E_maj_heavy"] = float(np.sum(ae[:, None] * (np.abs(Cm) ** 2)))
        c["E_killing_excess_heavy"] = float(np.sum(e * np.real(np.diag(Cm))))
        del Dm, Cm, As, U
        for key in ("E_maj_pairs", "E_maj_heavy", "E_killing_excess_heavy"):
            c["w_" + key] = w * c[key]
        if D.shape[1]:
            ab_in, tot_in = quarter_share(D, f, i50, n, sel); ab_all, tot_all = quarter_share(D, f, None, n, sel)
            c["inner_above_quarter"] = w * ab_in / n; c["inner_total"] = w * tot_in / n; c["all_above_quarter"] = w * ab_all / n; c["all_total"] = w * tot_all / n
            c["inner_share_above_quarter"] = (ab_in / tot_in) if tot_in > 0 else None; c["all_share_above_quarter"] = (ab_all / tot_all) if tot_all > 0 else None
        else:
            c["inner_above_quarter"] = c["inner_total"] = c["all_above_quarter"] = c["all_total"] = 0.0
            c["inner_share_above_quarter"] = c["all_share_above_quarter"] = None
        out["channels"][k] = c
    ch = out["channels"]
    ks = sorted(ch)
    tot_w = sum(ch[k]["w_E_maj_pairs"] for k in ks)
    out["sum_E_pairs_bare"] = sum(ch[k]["E_pairs_bare"] for k in ks)
    out["sum_n_pairs"] = sum(ch[k]["n_pairs"] for k in ks)
    out["sum_E_pairs_killing"] = sum(ch[k]["E_pairs_killing"] for k in ks)
    out["sum_w_E_maj_pairs"] = tot_w
    out["sum_w_E_maj_heavy"] = sum(ch[k]["w_E_maj_heavy"] for k in ks)
    out["sum_w_E_killing_excess_heavy"] = sum(ch[k]["w_E_killing_excess_heavy"] for k in ks)
    out["top_channel_share_w_E_maj_pairs"] = (ch[ks[-1]]["w_E_maj_pairs"] / tot_w) if tot_w > 0 else None
    out["above_matched_share_w_E_maj_pairs"] = (sum(ch[k]["w_E_maj_pairs"] for k in ks if k > MATCHED_K) / tot_w) if tot_w > 0 else None
    out["matched_sum_E_pairs_bare"] = sum(ch[k]["E_pairs_bare"] for k in ks if k <= MATCHED_K)
    out["matched_sum_w_E_maj_pairs"] = sum(ch[k]["w_E_maj_pairs"] for k in ks if k <= MATCHED_K)
    cumul = 0.0; out["cumulative_from_top_w_E_maj_pairs"] = {}
    for k in reversed(ks):
        cumul += ch[k]["w_E_maj_pairs"]; out["cumulative_from_top_w_E_maj_pairs"][k] = (cumul / tot_w) if tot_w > 0 else None
    it = sum(ch[k]["inner_total"] for k in ks); ia = sum(ch[k]["inner_above_quarter"] for k in ks)
    at = sum(ch[k]["all_total"] for k in ks); aa = sum(ch[k]["all_above_quarter"] for k in ks)
    out["inner_shell_particles"] = it; out["inner_shell_share_above_quarter"] = (ia / it) if it > 0 else None; out["all_dust_share_above_quarter"] = (aa / at) if at > 0 else None
    out["dust_particles"] = at
    return out


def t0_catalog(Z):
    out = {}
    for k in [int(x) for x in np.asarray(Z["kappas"])]:
        D = np.asarray(Z["dust_%d" % k], np.complex128); f = np.asarray(Z["f_%d" % k], float)
        if D.shape[1]:
            out[k] = {"columns": int(D.shape[1]), "f": [float(x) for x in f], "Phi": D}
    return out


def analyse_run(run_dir, tag, states_dir, code_dir, label):
    J = json.load(open(os.path.join(run_dir, "collapse_%s.json" % tag)))
    A = J["args"]; m = float(A["m"]); rs = float(A["rs"]); n = int(A["nsites"]); K = int(A["kappa_sea"]); tau = float(A["tau"])
    recs = [r for r in J["records"] if r.get("E") is not None]
    head = J["head"]
    Zf = np.load(os.path.join(run_dir, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    out = {"label": label, "tag": tag, "kappa_sea": K, "tau": tau, "m": m, "rs": rs, "nsites": n, "records": len(recs), "last_T_over_rs": (recs[-1]["T"] / rs) if recs else None,
           "head": {k_: head.get(k_) for k_ in ("N_B", "E0", "passes", "f_kin", "last_change", "prep_ell", "evolution_ell")},
           "N_cinit": np.asarray(Zf["N_cinit"], float), "v_cinit": np.asarray(Zf["v_cinit"], float), "formation": formation(recs, rs)}
    r14 = rec_at(recs, T_READ, rs)
    out["record_14"] = ({k_: r14.get(k_) for k_ in ("T_over_rs", "E", "E_over_E0", "E_pairs_shells", "E_dust_shells", "pairs_inside_Rt", "pairs_outside_Rt", "R_t", "R_t_over_2gE", "trapping_zeros",
                                                     "min_N_minus_absv", "R_c", "R_sh", "continuation_label", "mask_equals_negative_set")} if r14 else None)
    s14 = os.path.join(states_dir, "orbitals_%s_T%07.2f.npz" % (tag, T_READ))
    if os.path.exists(s14):
        Z = np.load(s14)
        if str(Z["dtype"]) != "complex128":
            out["slice_note"] = "the slice at 14 is %s, not complex128 (the policy): its numbers carry single precision" % Z["dtype"]
        out["slice"] = slice_analysis(Z, Zf, m, n, code_dir)
    else:
        out["slice"] = None; out["slice_note"] = "no slice at T/r_s 14 in %s" % states_dir
    s0 = os.path.join(states_dir, "orbitals_%s_T%07.2f.npz" % (tag, 0.0))
    out["t0_catalog"] = t0_catalog(np.load(s0)) if os.path.exists(s0) else None
    return out


def f3(x, f="%.4f"):
    return "-" if x is None else (f % x)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def render(runs):
    L = ["# w5_readout_v1 (busi W5): the heavy sea's angular truncation on the (0.2, 64) sigma/a 4 member", ""]
    L.append("Sources: (1) the records json; (2) to (4) the complex128 slice at T/r_s 14 with the chain rebuilt from engine_vacflow_v1.chain_sparse and h_m at the slice's applied fields; (5) the heads, the fields npz's C-init fields and the T = 0 slices. Definitions in the docstring.")
    L.append("")
    L.append("## Per run")
    for r in runs:
        L.append("")
        L.append("### %s (kappa_sea %d, tau %g; %s; %d records to T/r_s %s)" % (r["label"], r["kappa_sea"], r["tau"], r["tag"], r["records"], f3(r["last_T_over_rs"], "%.4f")))
        F = r["formation"]
        if F.get("T_over_rs") is not None:
            L.append("- formation (kappa_sea %d, tau %g): T/r_s %.4f, R_t %.3f a = %.4f r_s = %s of the record's 2GE, r_in %s, zeros %s, min(N - |v|) %s (the record before, T/r_s %s: %s), E/E0 %s; continuation %s" % (
                r["kappa_sea"], r["tau"], F["T_over_rs"], F["R_t"], F["R_t_over_rs"], f3(F["R_t_over_2gE"]), f3(F["r_in"], "%.3f"), F["zeros"], f3(F["min_N_minus_absv"], "%.5f"),
                f3(F["previous_T_over_rs"], "%.4f"), f3(F["previous_min_N_minus_absv"], "%.5f"), f3(F["E_over_E0"], "%.6f"), F.get("continuation_label")))
            L.append("- pairs at the formation record (json): E_pairs_shells %s, inside R_t %s, outside %s" % (f3(F["E_pairs_shells"]), f3(F["pairs_inside_Rt"]), f3(F["pairs_outside_Rt"])))
        else:
            L.append("- formation: %s" % F.get("note"))
        R14 = r["record_14"]
        if R14:
            L.append("- the record at 14 (json): E/E0 %s, E_pairs_shells %s (inside R_t %s, outside %s), E_dust_shells %s, R_t %s (%s of 2GE), zeros %s, min(N - |v|) %s, R_c %s, R_sh %s, mask = negative set %s" % (
                f3(R14["E_over_E0"], "%.6f"), f3(R14["E_pairs_shells"]), f3(R14["pairs_inside_Rt"]), f3(R14["pairs_outside_Rt"]), f3(R14["E_dust_shells"]), f3(R14["R_t"], "%.3f"), f3(R14["R_t_over_2gE"]),
                R14["trapping_zeros"], f3(R14["min_N_minus_absv"], "%.5f"), f3(R14["R_c"], "%.2f"), R14["R_sh"], R14["mask_equals_negative_set"]))
        else:
            L.append("- no record at T/r_s 14 in the json")
        S = r.get("slice")
        if S is None:
            L.append("- slice: %s" % r.get("slice_note"))
            continue
        if r.get("slice_note"):
            L.append("- slice note: %s" % r["slice_note"])
        L.append("- slice at T/r_s %.4f (fields row %.4f): R_50 %.2f a (site %d); fields row eps_pairs sum %.6f against the slice's sum of w (Tr[h P_sea] - Tr[h P_R2]) %.6f; pair count %.4f; Killing pair energy %.6f; dust particles %.4f, inner-shell particles %.4f, inner-shell share above pi/4 %s, whole dust %s; weighted majorants: pairs %.6f, heavy %.6f, Killing excess of the heavy state %.6f; top channel's share of 16 kappa E_kappa^pairs %s; share above kappa 10 %s" % (
            S["T_over_rs"], S["fields_row_T_over_rs"], S["R_50"], S["i_50"], S["eps_pairs_row_sum"], S["sum_E_pairs_bare"], S["sum_n_pairs"], S["sum_E_pairs_killing"], S["dust_particles"], S["inner_shell_particles"],
            f3(S["inner_shell_share_above_quarter"]), f3(S["all_dust_share_above_quarter"]), S["sum_w_E_maj_pairs"], S["sum_w_E_maj_heavy"], S["sum_w_E_killing_excess_heavy"],
            f3(S["top_channel_share_w_E_maj_pairs"]), f3(S["above_matched_share_w_E_maj_pairs"])))
        L.append("")
        rows = []
        for k in sorted(S["channels"]):
            c = S["channels"][k]
            rows.append([k, "%g" % c["w"], c["n_sea"], c["dust_columns"], "%.3f" % c["dust_particles"], "%.6f" % c["E_pairs_bare"], "%.5f" % c["n_pairs"], "%.6f" % c["E_pairs_killing"], "%.3e" % c["E_maj_pairs"], "%.4e" % c["w_E_maj_pairs"],
                         f3(S["cumulative_from_top_w_E_maj_pairs"][k]), "%.4e" % c["w_E_maj_heavy"], "%.4e" % c["w_E_killing_excess_heavy"], f3(c["inner_share_above_quarter"]), f3(c["all_share_above_quarter"]),
                         "%d/%d" % (c["pockets_occupied_positive"], c["holes_empty_negative"]), f3(c["eig_vs_stored_max"], "%.1e")])
        L.append(md_table(["kappa", "w", "n_sea", "dust cols", "dust particles", "w(Tr h P_sea - Tr h P_R2)", "pair count", "w Tr h_m(P_sea - P_R2)", "E^pairs_maj (1 copy)", "16k E^pairs_maj", "cumulative from top", "16k E^heavy_maj", "16k Killing excess", "inner share > pi/4", "whole share > pi/4", "pockets/holes", "eig vs stored"], rows))
    L.append("")
    L.append("## Cross-run tables")
    for tau in sorted(set(r["tau"] for r in runs), reverse=True):
        rr = sorted([r for r in runs if r["tau"] == tau], key=lambda q: q["kappa_sea"])
        L.append("")
        L.append("### tau %g" % tau)
        rows = []
        for r in rr:
            F = r["formation"]; S = r.get("slice") or {}; R14 = r.get("record_14") or {}
            rows.append([r["kappa_sea"], f3(F.get("T_over_rs")), f3(F.get("R_t_over_rs")), f3(F.get("R_t_over_2gE")), f3(F.get("E_over_E0"), "%.5f"), f3(F.get("E_pairs_shells")), f3(R14.get("E_pairs_shells")), f3(R14.get("E_over_E0"), "%.5f"),
                         f3(S.get("sum_n_pairs")), f3(S.get("inner_shell_share_above_quarter")), f3(S.get("all_dust_share_above_quarter")), f3(S.get("matched_sum_E_pairs_bare"), "%.6f"), f3(S.get("sum_E_pairs_bare"), "%.6f"),
                         f3(S.get("matched_sum_w_E_maj_pairs"), "%.5e"), f3(S.get("sum_w_E_maj_pairs"), "%.5e"), f3(S.get("top_channel_share_w_E_maj_pairs")), f3(S.get("above_matched_share_w_E_maj_pairs"))])
        L.append(md_table(["kappa_sea", "T_form/r_s", "R_t/r_s", "R_t/2GE", "E/E0 at formation", "E_pairs at formation", "E_pairs at 14", "E/E0 at 14", "pair count at 14", "inner-shell share > pi/4", "whole dust share", "sum_{k<=10} E_pairs bare", "sum all", "sum_{k<=10} 16k E_maj", "sum all 16k E_maj", "top channel share", "share above kappa 10"], rows))
        L.append("")
        L.append("Channels 1 to %d at matched kappa (w(Tr h P_sea - Tr h P_R2) at 14; the kappa_sea 10 run's value first): " % MATCHED_K)
        rows = []
        for k in range(1, MATCHED_K + 1):
            row = [k]
            for r in rr:
                c = ((r.get("slice") or {}).get("channels") or {}).get(k)
                row.append("%.6f / %.4e" % (c["E_pairs_bare"], c["w_E_maj_pairs"]) if c else "-")
            rows.append(row)
        L.append(md_table(["kappa"] + ["kappa_sea %d: E_pairs / 16k E_maj" % r["kappa_sea"] for r in rr], rows))
    L.append("")
    L.append("## The common-preparation check")
    L.append("")
    base = runs[0]
    rows = []
    for r in runs:
        dN = float(np.max(np.abs(r["N_cinit"] - base["N_cinit"]))) if r["N_cinit"].shape == base["N_cinit"].shape else float("nan")
        dv = float(np.max(np.abs(r["v_cinit"] - base["v_cinit"]))) if r["v_cinit"].shape == base["v_cinit"].shape else float("nan")
        cat = r.get("t0_catalog"); bcat = base.get("t0_catalog")
        if cat is not None and bcat is not None:
            parts = []
            for k in sorted(set(cat) | set(bcat)):
                if k not in cat or k not in bcat:
                    parts.append("kappa %d missing on one side" % k); continue
                fa = np.asarray(cat[k]["f"]); fb = np.asarray(bcat[k]["f"])
                df = float(np.max(np.abs(fa - fb))) if fa.shape == fb.shape else float("nan")
                ov = float(np.linalg.norm(cat[k]["Phi"].conj().T @ bcat[k]["Phi"], "fro") ** 2) if cat[k]["Phi"].shape == bcat[k]["Phi"].shape else float("nan")
                parts.append("kappa %d: %d columns, max|df| %.1e, overlap %.6f of %d" % (k, cat[k]["columns"], df, ov, cat[k]["columns"]))
            catalog = "; ".join(parts)
        else:
            catalog = "no T = 0 slice on one side"
        h = r["head"]
        rows.append([r["label"], f3(h.get("N_B"), "%.6f"), f3(h.get("E0"), "%.10f"), h.get("passes"), f3(h.get("f_kin"), "%.8f"), "%.2e" % dN, "%.2e" % dv, catalog])
    L.append(md_table(["run", "N_B", "E0", "passes", "f_kin", "max|dN_cinit|", "max|dv_cinit|", "T = 0 dust catalog against the first run"], rows))
    L.append("")
    L.append("Gate: the preparations are common when E0 and f_kin agree to roundoff, the C-init fields to 1e-10 and the T = 0 catalogs (columns, occupations, projector overlaps equal to the column counts) agree; otherwise the kappa_sea runs do not share C_0 and the formation comparison carries the preparation difference as a label.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="the W5 readout (busi)")
    ap.add_argument("--code-dir", required=True, help="the engine code tree (engine_vacflow_v1.py)")
    ap.add_argument("--root", default=None, help="the launcher's layout: ROOT/runs/<tag>/ and ROOT/states/<tag>/")
    ap.add_argument("--members", default=",".join(m_[0] for m_ in W5_MEMBERS))
    ap.add_argument("--run", action="append", default=[], help="RUNDIR/TAG:STATESDIR (repeatable; overrides --root)")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    for k_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k_, str(a.threads))
    specs = []
    if a.run:
        for s in a.run:
            rt, sd = s.rsplit(":", 1)
            rd, tg = os.path.split(rt)
            specs.append((rd, tg, sd, tg))
    elif a.root:
        want = set(x.strip() for x in a.members.split(",") if x.strip())
        for name, K, tau in W5_MEMBERS:
            if name in want:
                tg = tag_of(name, K, tau)
                specs.append((os.path.join(a.root, "runs", tg), tg, os.path.join(a.root, "states", tg), name))
    else:
        raise SystemExit("w5_readout_v1: give --root or --run")
    runs = []
    for rd, tg, sd, label in specs:
        if not os.path.exists(os.path.join(rd, "collapse_%s.json" % tg)):
            print("%s: no json at %s (skipped)" % (label, rd), flush=True)
            continue
        print("reading %s ..." % label, flush=True)
        runs.append(analyse_run(rd, tg, sd, a.code_dir, label))
    if not runs:
        raise SystemExit("w5_readout_v1: no run read")
    text = render(runs)
    print(text)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "w5_readout_v1.md"), "w") as fh:
        fh.write(text + "\n")

    def strip(o):
        if isinstance(o, dict):
            return {str(k): strip(v) for k, v in o.items() if k not in ("N_cinit", "v_cinit", "Phi")}
        if isinstance(o, list):
            return [strip(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    with open(os.path.join(a.out_dir, "w5_readout_v1.json"), "w") as fh:
        json.dump(strip(runs), fh, indent=1, default=str)
    print("\nwritten: %s and %s" % (os.path.join(a.out_dir, "w5_readout_v1.md"), os.path.join(a.out_dir, "w5_readout_v1.json")))


if __name__ == "__main__":
    main()
