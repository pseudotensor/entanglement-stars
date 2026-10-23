#!/usr/bin/env python
"""page_solved_history_report_v1.py (builder analyst, 2026-10-16; inbox #36 S1 F, with #41): writes PAGE_solved_history_v1.md and its
summary json from the summaries of Computation A (v6bA_s4 on page_history_v6b.py; v6eA_s4tr, the ledger run, on page_history_v6e.py; the
variants v6dA_s4far, v6dA_s4loc3, v6dA_s4k3, v6dA_s4cubic on page_history_v6d.py), the frozen-record controls C (v6eA_c15, v6eA_c30 on
page_history_v6e.py), the history's own trapping
records (extract_history_records_v1.py), and the validation (the box's self-test v6b and history oracles v6b, v6c and v6d, the regression's
summaries), as fetch_page_v6_v5.sh lays them out under results/page/. Every number is copied from those files or computed here from them
(marked "(script)"); nothing is retyped. Post-processing done here (item 9 of the quasi-frozen ruling, which the run does not reduce): the
window-averaged collected occupation per bin over sliding windows of 13 r_s and, in the soft bins (omega <= 5 T_H), the implied partner
occupation nu = n_out/[Gamma_kappa(omega) f_FD(omega/T_m)] with n_out the bin's collected excess occupation over its mode weight, Gamma_kappa
the continuum greybody of page_scatter_v5 at the bin centre and T_m = kappa_T/(2 pi) of the window's last checkpoint (a readout, no fit);
the hard-bin share (omega > 25 T_H) of the collected excess weighted by the bin centre (the overflow bin at its lower edge). No mutual
information is printed (#41 D15). Missing runs print as "not landed".
Usage: python page_solved_history_report_v1.py RESULTS_PAGE_DIR OUT_MD OUT_JSON"""
import datetime
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RUNS = {"s4": "v6bA_runs/v6bA_s4/v6bA_s4_summary.json", "s4tr": "v6bA_runs/v6eA_s4tr/v6eA_s4tr_summary.json",
        "s4far": "v6bA_runs/v6dA_s4far/v6dA_s4far_summary.json", "s4loc3": "v6bA_runs/v6dA_s4loc3/v6dA_s4loc3_summary.json",
        "s4k3": "v6bA_runs/v6dA_s4k3/v6dA_s4k3_summary.json", "s4cubic": "v6bA_runs/v6dA_s4cubic/v6dA_s4cubic_summary.json",
        "c15": "v6bA_runs/v6eA_c15/v6eA_c15_summary.json", "c30": "v6bA_runs/v6eA_c30/v6eA_c30_summary.json"}
VARIANT_LABEL = {"s4tr": "the ledger run: s4's settings continued from its T/r_s 15.0 state (the ledger at the peak)",
                 "s4far": "the far collector: R_cut 48 r_s, the collar 32 r_s", "s4loc3": "eps_loc 1e-3", "s4k3": "channels 1, 2, 3 (kappa 3 the truncation guard)",
                 "s4cubic": "the cubic time interpolation on 8 to 16 r_s, to T/r_s 16"}
VALID = {"selftest_v6b": "v6_validation/selftest_v6b_run1__selftest_results.json", "oracle_v6b": "v6_validation/history_oracle_v6b_run1__history_oracle_v6b_results.json",
         "oracle_v6c": "v6_validation/history_oracle_v6c_run2__history_oracle_v6c_results.json",
         "oracle_v6d": "v6_validation/history_oracle_v6d_run1__history_oracle_v6d_results.json",
         "oracle_v6e": "v6_validation/history_oracle_v6e_run1__history_oracle_v6e_results.json"}
REG = {n: "ctrl32_v5_runs/page/runs/%s/%s_summary.json" % (n, n) for n in ("ctrl32_v5_baseline", "v6reg_R0", "v6reg_R0s", "v6reg_R1", "v6reg_R3",
                                                                          "v6regc_R0", "v6regc_R0s", "v6regc_R1")}
HIST = "v6bA_runs/history_c3prod19_r11_records_v1.json"
TEX_SCALING = "<internal>/06_scaling.tex"
WINDOW_RS = 13.0
SOFT_MAX_TH = 5.0
HARD_MIN_TH = 25.0


def load(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def md5(p):
    try:
        return hashlib.md5(open(p, "rb").read()).hexdigest()
    except OSError:
        return None


def g(x, fmt="%.6e"):
    if x is None:
        return "-"
    try:
        return fmt % x
    except (TypeError, ValueError):
        return str(x)


def pcs(r):
    return [r["per_channel"][k] for k in sorted(r["per_channel"], key=int)]


def a_table(s, L, rs):
    L.append("| T/r_s | t_inf/r_s (collar) | 2gE | R_t (record) | 4gE kappa_T | E_inv | inside the collar | beyond | W (creation) | Tr[h_dot (C - P)] | L (interval) | L_H(kappa_T) | L_H(T_H) | L/L_H(kappa_T) | L/L_H(T_H) | S_esc | band | S_vac (archive) | S_esc - S_vac | S_B | S(M) - S(M') | collected per channel | archive per channel | outside weight min | incoming fraction | E_remove | M_chi | energy identity | collar identity | collar flow (acc.) |")
    L.append("|" + "---|" * 30)
    for r in s["rows"]:
        P = pcs(r)
        L.append("| %.3f | %s | %.4f | %s | %s | %.6e | %.6e | %.6e | %s | %.4e | %s | %s | %s | %s | %s | %.6f | %.2e | %.4e | %.6f | %s | %s | %s | %s | %s | %s | %.6e | %.6e | %.2e | %.2e | %.6e |" % (
            r["t_over_rs"], g(r.get("t_inf_collar") / rs if r.get("t_inf_collar") is not None else None, "%.3f"), r["two_gE"], g(r.get("R_t_record"), "%.3f"),
            g(r.get("four_gE_kappa_T"), "%.4f"), r["E_inv_weighted"], r["E_inv_inside_weighted"], r["E_out_weighted"], g(r.get("W_creation"), "%.4e"),
            r["W_hdot_weighted"], g(r.get("L_interval"), "%.4e"), g(r.get("L_H_kappaT"), "%.4e"), g(r.get("L_H_TH"), "%.4e"), g(r.get("L_over_LH_kappaT"), "%.4f"),
            g(r.get("L_over_LH_TH"), "%.4f"), r["S_esc_weighted"], r["band"], r["S_vac_archive_weighted"], r["S_esc_minus_S_vac_weighted"], g(r.get("S_B_weighted"), "%.4e"),
            g(r.get("S_M_minus_S_Mprime_weighted"), "%.4e"), ", ".join(str(p.get("collected")) for p in P), ", ".join(str(p.get("archive")) for p in P),
            ", ".join(g(p.get("outside_weight_min"), "%.5f") for p in P), ", ".join(g(p.get("incoming_fraction"), "%.3f") for p in P),
            r["E_remove_weighted"], r["M_chi_weighted"], r["energy_identity_residual_weighted"], r["collar_identity_residual_weighted"], r["collar_flow_acc_weighted"]))


def peak_levels(s):
    rows = s["rows"]
    if not rows:
        return None
    pk = max(rows, key=lambda r: r["S_esc_weighted"])
    lv = pk.get("levels") or {}
    return {"S_esc_peak": pk["S_esc_weighted"], "T_over_rs_peak": pk["t_over_rs"], "two_gE_at_peak": pk["two_gE"], "band_at_peak": pk["band"],
            "eta_A_over_a2": lv.get("eta_A_over_a2"), "A_over_4G": lv.get("A_over_4G"),
            "ratio_eta": pk["S_esc_weighted"] / lv["eta_A_over_a2"] if lv.get("eta_A_over_a2") else None,
            "ratio_A4G": pk["S_esc_weighted"] / lv["A_over_4G"] if lv.get("A_over_4G") else None,
            "S_esc_end": rows[-1]["S_esc_weighted"], "T_over_rs_end": rows[-1]["t_over_rs"], "falls_after_peak": rows[-1]["S_esc_weighted"] < pk["S_esc_weighted"]}


def spectral(s):
    """(script) item 9 from the stored per-block bins: per channel and checkpoint, the 13 r_s window's soft bins and the block's hard share."""
    import page_scatter_v5 as SC
    out = {}
    rows = s["rows"]
    gam = {}
    for r in rows:
        for k, pc in r["per_channel"].items():
            ob = out.setdefault(k, [])
            t = r["t"]; rs_k = r["two_gE"]; TH = r["T_H"]
            win = [q for q in rows if t - WINDOW_RS * s["history"]["rs_nominal"] - 1e-9 < q["t"] <= t + 1e-9]
            acc = {}
            for q in win:
                for b in (q["per_channel"].get(k) or {}).get("bins_occupation") or []:
                    key = (b["sign"], b["bin"], b["lo_TH"], b["hi_TH"])
                    e_, w_ = acc.get(key, (0.0, 0.0)); acc[key] = (e_ + b["excess"], w_ + b["weight"])
            kT = r.get("kappa_T")
            Tm = kT / (2.0 * math.pi) if kT else None
            soft = []
            for (sg, bi, lo, hi), (ex, wt) in sorted(acc.items(), key=lambda z: (z[0][0], z[0][2])):
                if hi is None or hi > SOFT_MAX_TH + 1e-12 or wt <= 0:
                    continue
                om = 0.5 * (lo + hi) * TH
                x = om * rs_k
                ck = (int(k), round(x, 12))
                if ck not in gam:
                    gam[ck] = float(SC.gamma_continuum(x, int(k))[0]) if x > 1e-6 else 0.0
                G = gam[ck]
                fFD = 1.0 / (math.exp(min(om / Tm, 700.0)) + 1.0) if Tm else None
                n_out = ex / wt
                nu = (n_out / (G * fFD)) if (fFD and G > 0) else None
                soft.append({"sign": sg, "lo_TH": lo, "hi_TH": hi, "excess": ex, "weight": wt, "n_out": n_out, "Gamma": G, "f_FD_Tm": fFD, "nu": nu})
            hard_num = 0.0; all_num = 0.0
            for b in pc.get("bins_occupation") or []:
                om = (0.5 * (b["lo_TH"] + b["hi_TH"]) if b.get("hi_TH") is not None else b["lo_TH"])
                all_num += om * b["excess"]
                if b["lo_TH"] >= HARD_MIN_TH - 1e-12:
                    hard_num += om * b["excess"]
            ob.append({"T_over_rs": r["t_over_rs"], "window_blocks": len(win), "T_m_over_T_H": (Tm / TH) if Tm else None, "soft": soft,
                       "hard_share": (hard_num / all_num) if all_num else None})
    return out


def variant_diff(a, b):
    ta = {round(r["t_over_rs"], 6): r for r in a["rows"]}; tb = {round(r["t_over_rs"], 6): r for r in b["rows"]}
    com = sorted(set(ta) & set(tb))
    if not com:
        return None
    def mx(key):
        vals = [abs((ta[t].get(key) or 0.0) - (tb[t].get(key) or 0.0)) for t in com]
        return max(vals) if vals else None
    last = com[-1]
    return {"common_checkpoints": len(com), "T_first": com[0], "T_last": last, "max_dS_esc": mx("S_esc_weighted"), "max_dE_inv": mx("E_inv_weighted"),
            "max_dL_interval": mx("L_interval"), "max_dcollar_flow": mx("collar_flow_acc_weighted"),
            "S_esc_at_last": [ta[last]["S_esc_weighted"], tb[last]["S_esc_weighted"]], "E_inv_at_last": [ta[last]["E_inv_weighted"], tb[last]["E_inv_weighted"]]}


def c_block(s):
    fr = [r for r in s["rows"] if r.get("frozen")]
    if not fr:
        return None
    E0 = fr[0]["E_inv_weighted"]; f0 = fr[0]["collar_flow_acc_weighted"]; fe = fr[-1]["collar_flow_acc_weighted"]
    pk = max(fr, key=lambda r: r["S_esc_weighted"])
    late = fr[-10:]
    comp = lambda r: r.get("L_H_kappaT") or r.get("L_H_TH")
    return {"T_freeze_over_rs": fr[0]["t_over_rs"], "duration_over_rs": fr[-1]["t_over_rs"] - fr[0]["t_over_rs"], "frozen_checkpoints": len(fr),
            "E_inv_0": E0, "E_inv_end": fr[-1]["E_inv_weighted"], "collar_flow_over_window": fe - f0, "fraction_leaving_script": (fe - f0) / E0 if E0 else None,
            "E_out_end": fr[-1]["E_out_weighted"], "S_esc_freeze": fr[0]["S_esc_weighted"], "S_esc_peak": pk["S_esc_weighted"], "T_over_rs_peak": pk["t_over_rs"],
            "S_esc_end": fr[-1]["S_esc_weighted"], "end_reason": s.get("end_reason"), "reflection_clock": s.get("reflection_clock"),
            "rows_after_reflection_clock": sum(1 for r in fr if r.get("after_reflection_clock")),
            "late_abs_L_over_comparator": [abs(r["L_interval"]) / comp(r) if (r.get("L_interval") is not None and comp(r)) else None for r in late],
            "end_rule_last": fr[-1].get("end_rule"), "transfer_starts": s.get("transfer_starts"),
            "transfer_end": {k: (tr or {}).get("end") for k, tr in (s.get("transfer") or {}).items()}}


def transfer_block(s):
    tr = s.get("transfer") or {}
    return {"starts": s.get("transfer_starts"), "per_channel": {k: {kk: v.get(kk) for kk in ("t0", "f", "norms", "w_inside_rt", "w_rt_to_cut", "w_beyond_cut", "end")} for k, v in tr.items()}}


def main():
    base, p_md, p_json = sys.argv[1:4]
    S = {n: load(os.path.join(base, p)) for n, p in RUNS.items()}
    V = {n: load(os.path.join(base, p)) for n, p in VALID.items()}
    Rg = {n: load(os.path.join(base, p)) for n, p in REG.items()}
    H = load(os.path.join(base, HIST))
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    s4 = S["s4"]
    rs = (s4 or {}).get("history", {}).get("rs_nominal") or 64.0
    J = {"written": now, "script": "code/page/page_solved_history_report_v1.py", "inputs": {}, "label": (s4 or {}).get("label"), "label_d15": (s4 or {}).get("label_d15")}
    for n, p in list(RUNS.items()) + list(VALID.items()) + list(REG.items()) + [("history_records", HIST)]:
        q = os.path.join(base, p)
        J["inputs"][n] = {"path": "results/page/" + p, "md5": md5(q), "landed": os.path.exists(q), "final": (load(q) or {}).get("final") if p.endswith("_summary.json") else None}
    L = ["# Computation A on the solved history, its frozen-record controls and the validation (S1 revised, inbox #36; with #41)\n",
         "Written %s by code/page/page_solved_history_report_v1.py from the files listed in its summary json (results/page/, fetched by code/page/fetch_page_v6_v5.sh "
         "through its filter: no mutual information, #41 D15). Every number is copied from those files or computed here from them, marked (script).\n" % now]
    lab = (s4 or {}).get("label") or "the massless channel of the lattice Dirac operator on the solved history of the massive collapse, its sources not fed back"
    L.append("Label on every number below: **%s**; the massless channel is a test channel under owner decision D14 (#41 F's words). %s.\n" % (lab, (s4 or {}).get("label_d15") or ""))
    L.append("Inputs: " + "; ".join("%s %s (%s%s)" % (n, v["path"], "md5 %s" % v["md5"] if v["md5"] else "not landed", ", final" if v.get("final") else "") for n, v in J["inputs"].items()) + ".\n")
    # the history
    L.append("## 0. The history (the record json's own trapping readouts)\n")
    if H:
        L.append("C3prod19 r11 (%s, md5 %s): %d records; the formation clock (the first record with a trapping root): T/r_s %s, R_t %s; the trapped episodes: %s.\n" % (
            H["source"], H["source_md5"], H["n_records"], H["formation"]["T_over_rs"], g(H["formation"]["R_t"], "%.4f"),
            "; ".join("T/r_s %s to %s (%d records, R_t first %.4f, largest %.4f)" % (e["T_over_rs_first"], e["T_over_rs_last"], e["records"], e["R_t_first"], e["R_t_max"]) for e in H["episodes"])))
        J["history"] = {"formation": H["formation"], "episodes": H["episodes"], "source_md5": H["source_md5"]}
    # A
    L.append("## 1. A, the primary run v6bA_s4 (page_history_v6b.py; R_cut 24 r_s, the collar 16 r_s, eps_loc 1e-2, collection every 2 r_s): the ledger per checkpoint (readouts 8 and 10)\n")
    if s4:
        st = s4["settings"]
        L.append("Run: final %s; %d checkpoints to T/r_s %s; the chain %s sites; channels %s; the archive %s; end %s.\n" % (
            s4.get("final"), len(s4["rows"]), g(s4["rows"][-1]["t_over_rs"] if s4["rows"] else None, "%.3f"), st.get("nsites"), st.get("channels"),
            "; ".join("kappa %s: %d modes, all-history Gram %.3e, sea unitarity %.3e, Chebyshev order max %s, tail %.3e" % (k, v["modes"], v["gram_all_history_max"], v["sea_unitarity_max"], v["cheb_K_max"], v["cheb_tail_acc"]) for k, v in s4["archive"].items()),
            s4.get("end_reason")))
        a_table(s4, L, rs)
        pl = peak_levels(s4)
        J["A_s4"] = {"final": s4.get("final"), "rows": s4["rows"], "peak": pl}
        if pl:
            L.append("\nReadout 10: the largest S_esc %.6f nats (band %.2e) at T/r_s %.3f (2gE %.4f); against eta A/a^2 %s: %s (script); against A/(4G) %s: %s (script); at the end T/r_s %.3f S_esc %.6f%s.\n" % (
                pl["S_esc_peak"], pl["band_at_peak"], pl["T_over_rs_peak"], pl["two_gE_at_peak"], g(pl["eta_A_over_a2"], "%.1f"), g(pl["ratio_eta"], "%.4e"), g(pl["A_over_4G"], "%.1f"),
                g(pl["ratio_A4G"], "%.4e"), pl["T_over_rs_end"], pl["S_esc_end"], " (below the largest: a fall, labelled by the identifiability statement of web40f insert 16)" if pl["falls_after_peak"] else " (no fall)"))
        # item 9
        try:
            sp = spectral(s4)
        except Exception as ex:                                             # noqa: BLE001
            sp = {"error": "%s: %s" % (type(ex).__name__, ex)}
        J["A_s4"]["spectral_script"] = sp
        L.append("\n### Readout 9 (script, from the stored per-block bins): the soft bins over the 13 r_s window and the hard share (the checkpoints with a trapping root; every checkpoint in the json)\n")
        if "error" in sp:
            L.append("- not computed: %s" % sp["error"])
        else:
            for k, lst in sp.items():
                for e in lst:
                    if e["T_m_over_T_H"] is None:                          # the md prints the trapped checkpoints (T_m defined); the json keeps all
                        continue
                    L.append("- kappa %s, T/r_s %.3f (%d blocks in the window; T_m/T_H %s): soft bins %s; the hard share %s" % (
                        k, e["T_over_rs"], e["window_blocks"], g(e["T_m_over_T_H"], "%.4f"),
                        "; ".join("%s[%g, %g) T_H: n_out %.4e over weight %.3e, Gamma %.4e, nu %s" % ("+" if b["sign"] > 0 else "-", b["lo_TH"], b["hi_TH"], b["n_out"], b["weight"], b["Gamma"], g(b["nu"], "%.4f")) for b in e["soft"]) or "-",
                        g(e["hard_share"], "%.4f")))
    else:
        L.append("not landed.")
    # the ledger
    L.append("\n## 2. The transfer ledger (readout 11) from the ledger run v6eA_s4tr (s4's settings from its T/r_s 15.0 state; the ledger restarts at each new S_esc maximum, the kept one starts at the run's peak)\n")
    tr = S["s4tr"]
    if tr:
        tb = transfer_block(tr)
        J["ledger_s4tr"] = tb
        dd = variant_diff(tr, s4) if s4 else None
        J["ledger_s4tr"]["against_s4"] = dd
        L.append("- final %s; the ledger's starts (T/r_s, weighted S_esc): %s" % (tr.get("final"), ", ".join("%.3f (%.6f)" % (x["t_over_rs"], x["S_esc_weighted"]) for x in (tb["starts"] or [])) or "-"))
        for k, v in tb["per_channel"].items():
            L.append("- kappa %s: start T/r_s %s; f %s; the partners' weights inside r_t %s, r_t to R_cut %s, beyond %s; at the end %s" % (
                k, g(v["t0"] / rs if v.get("t0") is not None else None, "%.3f"), v.get("f"), v.get("w_inside_rt"), v.get("w_rt_to_cut"), v.get("w_beyond_cut"), json.dumps(v.get("end"))))
        if dd:
            L.append("- against s4 at the %d common checkpoints (T/r_s %s to %s): max |dS_esc| %s, max |dE_inv| %s, max |dL| %s, max |d collar flow| %s (the same state code; the ledger does not enter the state)" % (
                dd["common_checkpoints"], dd["T_first"], dd["T_last"], g(dd["max_dS_esc"], "%.3e"), g(dd["max_dE_inv"], "%.3e"), g(dd["max_dL_interval"], "%.3e"), g(dd["max_dcollar_flow"], "%.3e")))
    else:
        L.append("not landed.")
    # the variants
    L.append("\n## 3. A's variants against s4 (both collectors: R_cut 24 and the far collector R_cut 48; eps_loc; the kappa 3 guard; the cubic control)\n")
    J["variants"] = {}
    for n in ("s4far", "s4loc3", "s4k3", "s4cubic"):
        v = S[n]
        if not v or not s4:
            L.append("- %s (%s): not landed" % (n, VARIANT_LABEL[n])); continue
        dd = variant_diff(v, s4)
        pv = peak_levels(v)
        J["variants"][n] = {"final": v.get("final"), "against_s4": dd, "peak": pv, "rows": v["rows"]}
        L.append("- %s (%s): final %s; %s; the largest S_esc %s at T/r_s %s (s4 %s)" % (
            n, VARIANT_LABEL[n], v.get("final"),
            ("at the %d common checkpoints (T/r_s %s to %s) max |dS_esc| %s, max |dE_inv| %s, max |dL| %s; at the last S_esc %s against %s" % (
                dd["common_checkpoints"], dd["T_first"], dd["T_last"], g(dd["max_dS_esc"], "%.4e"), g(dd["max_dE_inv"], "%.4e"), g(dd["max_dL_interval"], "%.4e"),
                g(dd["S_esc_at_last"][0], "%.6f"), g(dd["S_esc_at_last"][1], "%.6f"))) if dd else "no common checkpoint",
            g(pv and pv["S_esc_peak"], "%.6f"), g(pv and pv["T_over_rs_peak"], "%.3f"), g((peak_levels(s4) or {}).get("S_esc_peak"), "%.6f")))
    # C
    L.append("\n## 4. The frozen-record controls C (page_history_v6e.py, the frozen h's eigenbasis; S1's drain rule; the reflection clock)\n")
    J["C"] = {}
    for n in ("c15", "c30"):
        c = S[n]
        if not c:
            L.append("- %s: not landed" % n); continue
        cb = c_block(c)
        J["C"][n] = dict(cb or {}, final=c.get("final"), rows=[r for r in c["rows"] if r.get("frozen")])
        if cb is None:
            L.append("- %s: no frozen checkpoint yet" % n); continue
        rc = cb["reflection_clock"] or {}
        L.append("- %s: final %s; frozen at T/r_s %.3f for %.1f r_s (%d checkpoints); E_inv(0) %.6e, at the end %.6e; the collar flow over the window %.6e, the fraction that leaves %s (script); S_esc at the freeze %.6f, peak %.6f at T/r_s %.3f, end %.6f; the end: %s; the reflection clock %s r_s after the freeze (|v|/N at R_collar %s), %d rows after it; late |L|/comparator %s; the ledger's starts %s, at the end %s" % (
            n, c.get("final"), cb["T_freeze_over_rs"], cb["duration_over_rs"], cb["frozen_checkpoints"], cb["E_inv_0"], cb["E_inv_end"], cb["collar_flow_over_window"],
            g(cb["fraction_leaving_script"], "%.6f"), cb["S_esc_freeze"], cb["S_esc_peak"], cb["T_over_rs_peak"], cb["S_esc_end"], cb["end_reason"], g(rc.get("T_refl_rs"), "%.1f"),
            g(rc.get("s_at_R_collar"), "%.4f"), cb["rows_after_reflection_clock"], ["%.2e" % x if x is not None else "-" for x in cb["late_abs_L_over_comparator"]],
            ", ".join("%.3f" % x["t_over_rs"] for x in (cb["transfer_starts"] or [])) or "-", json.dumps(cb["transfer_end"])))
    # E
    L.append("\n## 5. Validation (S1 E)\n")
    st6 = V["selftest_v6b"]
    if st6:
        np_ = sum(1 for r in st6["results"] if r.get("pass") is True); nf = sum(1 for r in st6["results"] if r.get("pass") is False)
        L.append("- The self-test v6b (code %s): all_pass %s, %d PASS, %d FAIL, %d tests, %.0f s on %s." % (st6.get("code"), st6.get("all_pass"), np_, nf, len(st6["results"]), st6.get("seconds", float("nan")), st6.get("device")))
    for n in ("oracle_v6b", "oracle_v6c", "oracle_v6d", "oracle_v6e"):
        o = V[n]
        if not o:
            L.append("- %s: not landed" % n); continue
        L.append("- The history oracle %s (code %s; main checks all pass %s): %s; the cubic control %s; the frozen control: worst %s%s." % (
            n[-3:], o.get("code"), all(v["pass"] for v in o["checks"].values()), ", ".join("%s %.3e (%g)" % (k, v["worst"], v["criterion"]) for k, v in o["checks"].items()),
            json.dumps({k: o["cubic_control"].get(k) for k in ("t_over_rs", "dS", "dflow", "E_inv_piecewise", "E_inv_cubic")}) if o.get("cubic_control") else "-",
            json.dumps({k: o["frozen_control"]["worst"].get(k) for k in ("S_esc", "state_F", "gram", "collar_flow")}) if o.get("frozen_control") else "-",
            ("; the eigenbasis against the Chebyshev path: %s" % json.dumps({k: o["frozen_eig_vs_cheb"].get(k) for k in ("rows", "S_esc_max_abs", "E_inv_max_abs", "collar_flow_acc_max_abs", "collected_equal")})) if o.get("frozen_eig_vs_cheb") else ""))
    L.append("- The v6 regression on the pilot's prescribed background (stage 0 uncapped; results/page/page_v6_regression_v1.md): " + "; ".join(
        "%s: %s" % (n, ("S_max %.6f at t %.1f r_s1, S_esc at the end %.6f, E_rad at the end %.6e, final %s, code %s" % (
            max(x["S_esc_total"] for x in r_["curve"]), max(r_["curve"], key=lambda x: x["S_esc_total"])["t_over_rs1"], r_["curve"][-1]["S_esc_total"], r_["curve"][-1]["E_rad_stage"],
            r_.get("final"), r_.get("code", "page_recursion_v5.py"))) if r_ and r_.get("curve") else "not landed") for n, r_ in Rg.items()) + ".")
    J["validation"] = {"selftest_v6b": {"all_pass": (st6 or {}).get("all_pass")}, "oracles": {n: (V[n] or {}).get("checks") for n in ("oracle_v6b", "oracle_v6c", "oracle_v6d", "oracle_v6e")},
                       "regression": {n: ({"S_max": max(x["S_esc_total"] for x in r_["curve"]), "final": r_.get("final")} if r_ and r_.get("curve") else None) for n, r_ in Rg.items()}}
    # the figure's series and marks (the figure script plots them; nothing else is computed there)
    if s4 and s4["rows"]:
        rows = s4["rows"]
        tt = [r["t"] for r in rows]; tu = [r.get("t_inf_collar") for r in rows]
        def to_inf(t):
            """(script) the collar's arrival clock of a coordinate clock t, by linear interpolation of the rows' (t, t_inf_collar)."""
            if t is None or t < tt[0] or t > tt[-1]:
                return None
            for i in range(1, len(tt)):
                if t <= tt[i]:
                    f = (t - tt[i - 1]) / (tt[i] - tt[i - 1]) if tt[i] > tt[i - 1] else 0.0
                    return tu[i - 1] + f * (tu[i] - tu[i - 1])
            return tu[-1]
        Td = None; Td_line = None
        try:
            for ln in open(TEX_SCALING, encoding="utf-8"):
                if ln.lstrip().startswith("\\(T_d/r_s="):
                    Td_line = ln.rstrip("\n"); Td = float(ln.split("&")[2].strip())
                    break
        except OSError:
            pass
        T_form = (H or {}).get("formation", {}).get("T") if H else None
        recs = sorted(s4.get("records") or [], key=lambda r: r["t"])
        cum = []; acc = 0.0
        for r in recs:
            acc += r["collar_flow_weighted"]; cum.append((r["t"] + s4["history"]["tau"], acc))
        t_half = None
        if cum and cum[-1][1] > 0:
            half = 0.5 * cum[-1][1]; prev = (recs[0]["t"], 0.0)
            for (t_, c_) in cum:
                if c_ >= half:
                    t_half = prev[0] + (half - prev[1]) / (c_ - prev[1]) * (t_ - prev[0]) if c_ > prev[1] else t_
                    break
                prev = (t_, c_)
        # (script) the formation's light-cone arrival at the collar: the outgoing light travel time from R_e = c 2gE to R_collar in the vacuum
        # Painleve-Gullstrand exterior at the formation record's charge (N = 1, v = -(2gE/R)^(1/2)): 2 r_s [x^2/2 + x + ln(x - 1)] over
        # x = (R/r_s)^(1/2) from c^(1/2) to (R_collar/r_s)^(1/2); c = 2 marked, c = 1.5 printed beside (the choice of the start radius)
        E_form = (H or {}).get("formation", {}).get("E") if H else None
        R_coll_rs = float(s4["settings"].get("R_collar_rs", 16.0))
        def travel(c_):
            F = lambda x: 0.5 * x * x + x + math.log(x - 1.0)
            return 2.0 * (F(math.sqrt(R_coll_rs)) - F(math.sqrt(c_)))
        rs_form = 2.0 * s4["history"]["g"] * E_form if E_form else None
        lc = {c_: (travel(c_) * rs_form) for c_ in (2.0, 1.5)} if rs_form else {}
        t_arr = (T_form + lc[2.0]) if (T_form is not None and lc) else None
        J["figure"] = {"t_over_rs": [r["t_over_rs"] for r in rows], "t_inf_over_rs": [(r.get("t_inf_collar") or 0.0) / rs for r in rows],
                       "S_esc": [r["S_esc_weighted"] for r in rows], "band": [r["band"] for r in rows],
                       "L_interval": [r.get("L_interval") for r in rows], "L_over_LH_kappaT": [r.get("L_over_LH_kappaT") for r in rows],
                       "L_over_LH_TH": [r.get("L_over_LH_TH") for r in rows], "final": s4.get("final"),
                       "marks": {"formation_T_over_rs": (T_form / rs) if T_form is not None else None,
                                 "formation_t_inf_over_rs": (to_inf(T_form) / rs) if (T_form is not None and to_inf(T_form) is not None) else None,
                                 "T_d_over_rs": Td, "T_d_source": "%s: %s (the r_s/a 64 column; plus an unfixed constant of order r_s)" % (TEX_SCALING, Td_line),
                                 "formation_plus_T_d_t_inf_over_rs": (to_inf(T_form + Td * rs) / rs) if (T_form is not None and Td is not None and to_inf(T_form + Td * rs) is not None) else None,
                                 "arrival_t_over_rs": (t_arr / rs) if t_arr is not None else None,
                                 "arrival_t_inf_over_rs": (to_inf(t_arr) / rs) if (t_arr is not None and to_inf(t_arr) is not None) else None,
                                 "arrival_travel_over_2gE": {str(c_): v_ / rs_form for c_, v_ in lc.items()} if lc else None, "two_gE_formation": rs_form,
                                 "arrival_definition": "(script) the formation clock plus the outgoing light travel time from 2 x 2gE to R_collar in the vacuum Painleve-Gullstrand exterior at the formation record's charge, 2 r_s [x^2/2 + x + ln(x - 1)] over x = (R/r_s)^(1/2); the start radius is the builder's choice (1.5 x 2gE printed beside)",
                                 "half_collar_flow_t_over_rs": (t_half / rs) if t_half is not None else None,
                                 "half_collar_flow_t_inf_over_rs": (to_inf(t_half) / rs) if (t_half is not None and to_inf(t_half) is not None) else None,
                                 "half_collar_flow_definition": "(script) the clock at which half of the collar energy flow accumulated over the run has arrived (#41 B.4's arrival-clock definition applied to A's collar flow)",
                                 "collar_flow_total": cum[-1][1] if cum else None},
                       "label": lab + "; the collapse epoch's escaping entropy and flux on the solved history, a test channel under owner decision D14"}
        mk = J["figure"]["marks"]
        L.append("\n## 5b. The figure's marks (ITEM7_fig_page_draft_v1.png)\n")
        L.append("- the formation clock T/r_s %s (t_inf/r_s %s); T_d/r_s %s (%s), T_form + T_d at t_inf/r_s %s; the formation's arrival clock T/r_s %s (t_inf/r_s %s; the travel over 2gE %s; 2gE at the formation %s): %s; the half-collar-flow clock T/r_s %s (t_inf/r_s %s): %s; the collar flow over the run %s." % (
            g(mk["formation_T_over_rs"], "%.3f"), g(mk["formation_t_inf_over_rs"], "%.3f"), g(mk["T_d_over_rs"], "%.2f"), mk["T_d_source"], g(mk["formation_plus_T_d_t_inf_over_rs"], "%.3f"),
            g(mk["arrival_t_over_rs"], "%.3f"), g(mk["arrival_t_inf_over_rs"], "%.3f"), json.dumps(mk["arrival_travel_over_2gE"]), g(mk["two_gE_formation"], "%.4f"), mk["arrival_definition"],
            g(mk["half_collar_flow_t_over_rs"], "%.3f"), g(mk["half_collar_flow_t_inf_over_rs"], "%.3f"), mk["half_collar_flow_definition"], g(mk["collar_flow_total"], "%.6e")))
    L.append("\n## 6. Not in this file\n")
    L.append("- D (the massive channel's own emission from the stored records) and B (the engine continuation to T/r_s 100) are other builders' tasks; A to T/r_s 100 and A on the sigma/a 8 history wait for B's records.")
    L.append("- S1 A.12 (the slab mutual information) is replaced by S4's A.3 (#41 E), another builder's; no mutual information is printed here.")
    with open(p_md, "w") as f:
        f.write("\n".join(L) + "\n")
    with open(p_json, "w") as f:
        json.dump(J, f, indent=1, default=lambda o: float(o) if isinstance(o, (int, float)) else str(o))
    print("wrote %s and %s: %s" % (p_md, p_json, ", ".join("%s %s" % (n, "landed" if v else "-") for n, v in S.items())))


if __name__ == "__main__":
    main()
