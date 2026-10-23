#!/usr/bin/env python
"""page_solved_history_report_v6.py (the lane's analyst builder; engine spec 39.1 (1)): page_solved_history_report_v5.py with one line changed:
A to T/r_s 100 runs on pod 2 beside the sigma/a 2 ladder member (inbox #54), not queued on pod 1 lane 1. Everything else is v5's.

v5's docstring: page_solved_history_report_v5.py (builder analyst, 2026-10-17; engine spec 39.1 (1)): page_solved_history_report_v4.py with A to T/r_s 100's
code named as it runs, page_history_v6i.py (v6g's anchored ledger and continuation, with the partners built on the pure state and the
orthonormal archive and their Gram brought to the identity), in sections 1, 2 and 7. Everything else is v4's.

v4's docstring: page_solved_history_report_v4.py (builder analyst, 2026-10-17; engine spec 39.1 (1)): page_solved_history_report_v3.py with S_esc over each
fixed-field control's clean reading printed in section 4 (its largest and its value at the last checkpoint before the reflection clock):
v3's line gives the largest over the whole hold, which for c15 falls after the reflection clock (34.031771 at T/r_s 293 against 34.002385
at 237 before it). Everything else is v3's.

v3's docstring: page_solved_history_report_v3.py (builder analyst, 2026-10-17; engine spec 39.1 (1), (3)): page_solved_history_report_v2.py with the run
map of A's variants as they ran (s4k3 and s4cubic on page_history_v6f.py, tags v6fA_s4k3 and v6fA_s4cubic, pod 1 lane 5: s4k3's first run
ran out of device memory beside s8, s4cubic moved from the box's A6000; s4far and s4loc3 on page_history_v6d.py as before), the code of each
variant printed in its line, and the lines on A to T/r_s 100 brought to its state (page_history_v6g.py, from T = 0, queued on pod 1 lane 1;
A on the sigma/a 8 history is in this file, section 1b). Everything else is v2's.

v2's docstring: page_solved_history_report_v2.py (builder analyst, 2026-10-16; engine spec 38.1 (2), (3) and 38.0): the composer of PAGE_solved_history
version 2 and its figure's data. It reuses page_solved_history_report_v1.py's readers and tables (imported, unchanged) and adds:
(1) the wording of requirements_new.txt sec. 4 in every printed line (the records held fixed are "the fixed-field controls"; their start "the
    hold's start"; "window" only for a window observable; no "prediction"), with a literal-replacement pass over the composed text as the net
    (its counts printed by the replacing words);
(2) readout 9 with a per-bin resolution flag (unreadable where 2 pi/omega at the bin's lower edge exceeds a quarter of the exterior segment
    beyond R_cut, the chain's end minus R_cut) and kappa 2's soft counts printed beside kappa 1's as the null;
(3) the figure's marks of 38.1 (2) in the json's figure block ("v2"): the axis the collar's clock; the formation's record clock as a reference
    line; the formation's arrival at the collar as the measured half-collar-flow clock, with the light-travel range from 2 and 1.5 times 2gE as
    a band; the ray-map epoch's end at the collar as that arrival plus T_d (the formation plus T_d is a surface clock and is not marked);
(4) the validation of the later files (the history oracles v6d to v6f, the restore diff of 38.1 (3)) and the work accumulated per record where
    a run has it (page_history_v6f runs; none of this version's runs);
(5) a draft label, printed under the title, from --draft-label.
Usage: python page_solved_history_report_v6.py RESULTS_PAGE_DIR OUT_MD OUT_JSON [--draft-label TEXT]"""
import datetime
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import page_solved_history_report_v1 as V1     # noqa: E402  (its readers and tables, unchanged)

RUNS = dict(V1.RUNS)
RUNS["s8"] = "v6bA_runs/v6fA_s8/v6fA_s8_summary.json"                      # A on the sigma/a 8 history (page_history_v6f.py)
RUNS["s4k3"] = "v6bA_runs/v6fA_s4k3/v6fA_s4k3_summary.json"                  # v3: as it ran (page_history_v6f.py, pod 1 lane 5)
RUNS["s4cubic"] = "v6bA_runs/v6fA_s4cubic/v6fA_s4cubic_summary.json"          # v3: as it ran (page_history_v6f.py, pod 1 lane 5)
VALID = dict(V1.VALID)
VALID["oracle_v6f"] = "v6_validation/history_oracle_v6f_run1__history_oracle_v6f_results.json"
RESTORE = "v6_validation/restore_diff_v1_run2__restore_diff_results.json"
REG = dict(V1.REG)
HIST = V1.HIST
TEX_SCALING = V1.TEX_SCALING
g = V1.g
REPL = [("its frozen-record controls", "its fixed-field controls"), ("The frozen-record controls", "The fixed-field controls"),
        ("frozen-record", "fixed-field"), ("the frozen h's eigenbasis", "the held-fixed h's eigenbasis"), ("frozen h", "held-fixed h"),
        ("Frozen", "Fixed-field"), ("frozen", "fixed-field"), ("the freeze", "the hold's start"), ("over the window", "over the run"),
        ("predictions", "statements"), ("prediction", "statement"), ("unitarity defect", "unitarity deviation"), ("defect", "deviation")]


def replace_all(s, counts):
    for a, b in REPL:
        n = s.count(a)
        if n:
            counts[b] = counts.get(b, 0) + n
            s = s.replace(a, b)
    return s


def rename_key(k, counts):
    """the json's keys under the same wording: "frozen" reads "fixed_field", "freeze" reads "hold" (counted)."""
    if isinstance(k, str) and ("frozen" in k or "freeze" in k):
        counts["key renamed"] = counts.get("key renamed", 0) + 1
        return k.replace("frozen", "fixed_field").replace("freeze", "hold")
    return k


def walk(o, counts):
    if isinstance(o, dict):
        return {rename_key(k, counts): walk(v, counts) for k, v in o.items()}
    if isinstance(o, list):
        return [walk(x, counts) for x in o]
    if isinstance(o, str):
        return replace_all(o, counts)
    return o


def spectral_flagged(s):
    """readout 9 (v1's spectral) with each soft bin's resolution flag: readable where 2 pi/(lo T_H) <= (R_end - R_cut)/4 (lo the bin's lower
    edge in T_H; R_end the chain's last site radius, nsites in lattice units; R_cut the run's)."""
    sp = V1.spectral(s)
    st = s["settings"]; rs = s["history"]["rs_nominal"]
    L_ext = float(st["nsites"]) - float(st["R_cut_rs"]) * rs
    TH_of = {round(r["t_over_rs"], 6): r["T_H"] for r in s["rows"]}
    for k, lst in sp.items():
        for e in lst:
            TH = TH_of.get(round(e["T_over_rs"], 6))
            for b in e["soft"]:
                b["readable"] = bool(TH and b["lo_TH"] > 0 and 2.0 * math.pi / (b["lo_TH"] * TH) <= 0.25 * L_ext)
    return sp, L_ext


def readout9_md(sp, L_ext, rs, L):
    L.append("\n### Readout 9 (script, from the stored per-block bins): the soft bins over the 13 r_s window observable, kappa 2 beside kappa 1 as the null, each bin's resolution flag\n")
    L.append("A bin is readable only where 2 pi/omega at its lower edge is at most a quarter of the exterior segment beyond R_cut (%.1f r_s, a quarter %.2f r_s): about 3 T_H and up on this chain (engine spec 38.1 (v)); nu = n_out/[Gamma_kappa f_FD(omega/T_m)] is a readout, never a partner occupation where the bin is unreadable, and kappa 2's counts are the null (its barrier is 1e-4 to 1e-6 as transparent at these omega). nu > 1 fails the sole-filled-partner reading at the implemented bin normalization and is not a physical partner occupation; the hard share is per block (binned), a different temporal selection from the 13 r_s soft window observable (inbox #48 (3)).\n" % (
        L_ext / rs, 0.25 * L_ext / rs))
    k1 = {round(e["T_over_rs"], 6): e for e in sp.get("1", [])}
    k2 = {round(e["T_over_rs"], 6): e for e in sp.get("2", [])}
    L.append("| T/r_s | T_m/T_H | bin (T_H; + particles, - holes) | readable | kappa 1: n_out | weight | Gamma | nu | kappa 2: n_out | weight | Gamma | nu | hard share kappa 1, kappa 2 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for T in sorted(k1):
        e1 = k1[T]; e2 = k2.get(T)
        if e1["T_m_over_T_H"] is None:
            continue
        b2 = {(b["sign"], b["lo_TH"], b["hi_TH"]): b for b in (e2["soft"] if e2 else [])}
        keys = sorted({(b["sign"], b["lo_TH"], b["hi_TH"]) for b in e1["soft"]} | set(b2), key=lambda z: (z[0], z[1]))
        b1 = {(b["sign"], b["lo_TH"], b["hi_TH"]): b for b in e1["soft"]}
        for kk in keys:
            x = b1.get(kk); y = b2.get(kk)
            rd = (x or y or {}).get("readable")
            L.append("| %.3f | %s | %s[%g, %g) | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s, %s |" % (
                T, g(e1["T_m_over_T_H"], "%.4f"), "+" if kk[0] > 0 else "-", kk[1], kk[2], "yes" if rd else "no",
                g(x and x["n_out"], "%.3e"), g(x and x["weight"], "%.3e"), g(x and x["Gamma"], "%.3e"), g(x and x["nu"], "%.4g"),
                g(y and y["n_out"], "%.3e"), g(y and y["weight"], "%.3e"), g(y and y["Gamma"], "%.3e"), g(y and y["nu"], "%.4g"),
                g(e1["hard_share"], "%.4f"), g(e2 and e2["hard_share"], "%.4f")))


def figure_v2(s4, H, rs, J):
    """the figure block with version 1's marks kept and 38.1 (2)'s marks under "v2"."""
    rows = s4["rows"]
    tt = [r["t"] for r in rows]; tu = [r.get("t_inf_collar") for r in rows]

    def to_inf(t):
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
    E_form = (H or {}).get("formation", {}).get("E") if H else None
    R_coll_rs = float(s4["settings"].get("R_collar_rs", 16.0))

    rs_form = 2.0 * s4["history"]["g"] * E_form if E_form else None
    x_up = math.sqrt(R_coll_rs * rs / rs_form) if rs_form else None              # inbox #48 (4): the collar at 16 r_s,0 = 1024a, in units of r_s,f

    def travel(c_):
        F = lambda x: 0.5 * x * x + x + math.log(x - 1.0)
        return 2.0 * (F(x_up) - F(math.sqrt(c_)))
    lc = {c_: (travel(c_) * rs_form) for c_ in (2.0, 1.5)} if rs_form else {}
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
    th_inf = to_inf(t_half) if t_half is not None else None
    band = [to_inf(T_form + lc[c_]) for c_ in (2.0, 1.5)] if (T_form is not None and lc) else None
    v2 = {"axis": "the collar's clock t_inf: the retarded clock at R_collar in each record's instantaneous stationary chart (a changing-charge substitution, not a solved common arrival map; inbox #48 (4)), nominal r_s",
          "L_definition": "L is the collar energy flow over each checkpoint interval divided by its coordinate length Delta T, not by Delta t_inf (inbox #48 (4)); L_H(kappa_T) and L_H(T_H) are comparators (the two-channel continuum greybody over a finite x range), not an observed thermal flux; kappa_T is the record's dense root and slope, which differ from the applied chain root (inbox #48 (3))",
          "formation_record_clock_over_rs": (T_form / rs) if T_form is not None else None,
          "formation_record_clock_definition": "the history's first record with a trapping root (a record clock drawn as a reference line on the collar axis, not a collar clock)",
          "arrival_half_flow_t_over_rs": (t_half / rs) if t_half is not None else None,
          "arrival_half_flow_t_inf_over_rs": (th_inf / rs) if th_inf is not None else None,
          "arrival_definition": "(script) the clock at which half of the signed collar energy flow accumulated over the available run (to its last record) has arrived at R_collar (the sum of the per-record exact flows); it halves the signed integral over this run only (inbox #48 (4))",
          "light_travel_band_t_inf_over_rs": [(b / rs) if b is not None else None for b in band] if band else None,
          "light_travel_band_definition": "(script) the formation's record clock plus the outgoing light travel from 2 and 1.5 x 2gE_f to the collar R_collar = 16 r_s,0 = 1024a in the vacuum Painleve-Gullstrand exterior at the formation record's charge (r_s,f = 2gE_f), 2 r_s,f [x^2/2 + x + ln(x - 1)] over x = (R/r_s,f)^(1/2) from c^(1/2) to (16 x 64/r_s,f)^(1/2) (inbox #48 (4): v1 integrated to 16 r_s,f), on the collar's clock",
          "light_travel_upper_x": x_up,
          "light_travel_over_2gE": {str(c_): v_ / rs_form for c_, v_ in lc.items()} if lc else None, "two_gE_formation": rs_form,
          "T_d_over_rs": Td, "T_d_source": "%s: %s (the r_s/a 64 column; plus an unfixed constant of order r_s; its matching and its frequency and chart premises are imported from the paper's table, not computed here: inbox #48 (4))" % (TEX_SCALING, Td_line),
          "raymap_end_t_inf_over_rs": ((th_inf / rs) + Td) if (th_inf is not None and Td is not None) else None,
          "raymap_end_definition": "the formation's arrival at the collar (the half-collar-flow clock) plus T_d; the formation plus T_d is a surface clock and is not drawn on the collar axis",
          "collar_flow_total": cum[-1][1] if cum else None}
    return {"t_over_rs": [r["t_over_rs"] for r in rows], "t_inf_over_rs": [(r.get("t_inf_collar") or 0.0) / rs for r in rows],
            "S_esc": [r["S_esc_weighted"] for r in rows], "band": [r["band"] for r in rows], "L_interval": [r.get("L_interval") for r in rows],
            "L_over_LH_kappaT": [r.get("L_over_LH_kappaT") for r in rows], "L_over_LH_TH": [r.get("L_over_LH_TH") for r in rows], "final": s4.get("final"),
            "marks": {"v2": v2}, "label": (J.get("label") or "") + "; the collapse epoch's escaping entropy and flux on the solved history, a test channel under owner decision D14"}


CLEAN_FROM = {"c15": 77.0, "c30": 148.0}                                 # the clean segments' starts of engine spec 39.1's reading (T/r_s)


def hold_readouts(c, cb):
    """(script) the hold's own readouts: E_inv's largest deviation from E_inv(0); the interior part's linear rate up to the last checkpoint before
    the reflection clock and its linear extrapolation to empty (untested); L/L_H(kappa_T) every 16 r_s and its range over the clean reading."""
    fr = [r for r in c["rows"] if r.get("frozen")]
    T0 = fr[0]["t_over_rs"]
    T_refl = T0 + ((cb.get("reflection_clock") or {}).get("T_refl_rs") or 1e9)
    E0 = fr[0]["E_inv_weighted"]
    clean = [r for r in fr if r["t_over_rs"] < T_refl]
    last = clean[-1]
    rate = (fr[0]["E_inv_inside_weighted"] - last["E_inv_inside_weighted"]) / max(last["t_over_rs"] - T0, 1e-300)
    every = [(r["t_over_rs"], r.get("L_over_LH_kappaT")) for r in fr if r.get("L_over_LH_kappaT") is not None and abs(((r["t_over_rs"] - T0 - 2.0) / 16.0) - round((r["t_over_rs"] - T0 - 2.0) / 16.0)) < 1e-9]
    name = "c15" if abs(T0 - 15.0) < 1e-9 else ("c30" if abs(T0 - 30.0) < 1e-9 else None)
    t_from = CLEAN_FROM.get(name, T0)
    seg = [r.get("L_over_LH_kappaT") for r in clean if r["t_over_rs"] >= t_from - 1e-9 and r.get("L_over_LH_kappaT") is not None]
    pk = max(clean, key=lambda r: r["S_esc_weighted"])                       # v4: S_esc over the clean reading
    return {"S_clean_peak": pk["S_esc_weighted"], "T_clean_peak": pk["t_over_rs"], "S_clean_end": last["S_esc_weighted"], "rows_after_reflection": len(fr) - len(clean),
            "E_inv_max_dev": max(abs(r["E_inv_weighted"] - E0) for r in fr), "inside_start": fr[0]["E_inv_inside_weighted"], "inside_clean_end": last["E_inv_inside_weighted"],
            "T_clean_end": last["t_over_rs"], "T_refl": T_refl, "inside_rate_per_rs": rate, "linear_empty_rs": (fr[0]["E_inv_inside_weighted"] / rate) if rate > 0 else None,
            "L_over_LH_every16": every, "clean_window_from": t_from, "L_clean_min": min(seg) if seg else None, "L_clean_max": max(seg) if seg else None,
            "E_at_hold": fr[0]["E"], "E_inv0_over_E": E0 / fr[0]["E"]}


def c_line(n, c, cb):
    rc = cb["reflection_clock"] or {}
    return ("- %s: final %s; held fixed at T/r_s %.3f for %.1f r_s (%d fixed-field checkpoints); E_inv at the hold's start %.6e, at the end %.6e; the collar flow over the hold %.6e, the fraction that leaves %s (script); "
            "S_esc at the hold's start %.6f, its largest %.6f at T/r_s %.3f, at the end %.6f; the end: %s; the reflection clock %s r_s after the hold's start (|v|/N at R_collar %s), %d rows after it; "
            "the last 10 rows' |L|/comparator %s; the ledger's restarts %d (the last at T/r_s %s), at the end %s") % (
        n, c.get("final"), cb["T_freeze_over_rs"], cb["duration_over_rs"], cb["frozen_checkpoints"], cb["E_inv_0"], cb["E_inv_end"], cb["collar_flow_over_window"],
        g(cb["fraction_leaving_script"], "%.6f"), cb["S_esc_freeze"], cb["S_esc_peak"], cb["T_over_rs_peak"], cb["S_esc_end"], cb["end_reason"], g(rc.get("T_refl_rs"), "%.1f"),
        g(rc.get("s_at_R_collar"), "%.4f"), cb["rows_after_reflection_clock"], ["%.2e" % x if x is not None else "-" for x in cb["late_abs_L_over_comparator"]],
        len(cb["transfer_starts"] or []), g((cb["transfer_starts"] or [{}])[-1].get("t_over_rs") if cb["transfer_starts"] else None, "%.3f"),
        json.dumps({k: {kk: (v or {}).get(kk) for kk in ("trace_W_R_over_p", "identity_residual_F")} for k, v in (cb["transfer_end"] or {}).items()}))


def main():
    args = sys.argv[1:]
    draft = "A to T/r_s 40 with its fixed-field controls and variants; the extension to 100 pending"   # engine spec 39.1 (3)
    if "--draft-label" in args:
        i = args.index("--draft-label"); draft = args[i + 1]; args = args[:i] + args[i + 2:]
    base, p_md, p_json = args[:3]
    S = {n: V1.load(os.path.join(base, p)) for n, p in RUNS.items()}
    Vd = {n: V1.load(os.path.join(base, p)) for n, p in VALID.items()}
    Rg = {n: V1.load(os.path.join(base, p)) for n, p in REG.items()}
    RD = V1.load(os.path.join(base, RESTORE))
    H = V1.load(os.path.join(base, HIST))
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    s4 = S["s4"]
    rs = (s4 or {}).get("history", {}).get("rs_nominal") or 64.0
    J = {"written": now, "script": "code/page/page_solved_history_report_v6.py (with page_solved_history_report_v1.py's readers)", "inputs": {},
         "label": (s4 or {}).get("label"), "label_d15": (s4 or {}).get("label_d15"), "draft_label": draft}
    for n, p in list(RUNS.items()) + list(VALID.items()) + list(REG.items()) + [("restore_diff", RESTORE), ("history_records", HIST)]:
        q = os.path.join(base, p)
        J["inputs"][n] = {"path": "results/page/" + p, "md5": V1.md5(q), "landed": os.path.exists(q), "final": (V1.load(q) or {}).get("final") if p.endswith("_summary.json") else None}
    L = ["# Computation A on the solved history, its fixed-field controls and the validation: PAGE_solved_history version 2 (S1 revised, inbox #36; with #41; engine spec 38.1)\n"]
    if draft:
        L.append("**%s**\n" % draft)
    L.append("Written %s by code/page/page_solved_history_report_v6.py from the files listed in its summary json (results/page/, fetched through the no-mutual-information filter, #41 D15). Every number is copied from those files or computed here from them, marked (script).\n" % now)
    lab = (s4 or {}).get("label") or "the massless channel of the lattice Dirac operator on the solved history of the massive collapse, its sources not fed back"
    L.append("Label on every number below: **%s**; the massless channel is a test channel under owner decision D14 (#41 F's words). %s.\n" % (lab, (s4 or {}).get("label_d15") or ""))
    L.append("Inputs: " + "; ".join("%s %s (%s%s)" % (n, v["path"], "md5 %s" % v["md5"] if v["md5"] else "not landed", ", final" if v.get("final") else "") for n, v in J["inputs"].items()) + ".\n")
    if H:
        L.append("## 0. The history (the record json's own trapping readouts)\n")
        L.append("C3prod19 r11 (%s, md5 %s): %d records; the formation's record clock (the first record with a trapping root): T/r_s %s, R_t %s; the trapped episodes: %s.\n" % (
            H["source"], H["source_md5"], H["n_records"], H["formation"]["T_over_rs"], g(H["formation"]["R_t"], "%.4f"),
            "; ".join("T/r_s %s to %s (%d records, R_t first %.4f, largest %.4f)" % (e["T_over_rs_first"], e["T_over_rs_last"], e["records"], e["R_t_first"], e["R_t_max"]) for e in H["episodes"])))
        J["history"] = {"formation": H["formation"], "episodes": H["episodes"], "source_md5": H["source_md5"]}
    L.append("## 1. A, the primary run v6bA_s4 (page_history_v6b.py; R_cut 24 r_s, the collar 16 r_s, eps_loc 1e-2, collection every 2 r_s): the ledger per checkpoint (readouts 8 and 10)\n")
    if s4:
        st = s4["settings"]
        L.append("Run: final %s; %d checkpoints to T/r_s %s; the chain %s sites; channels %s; the archive %s; end %s.\n" % (
            s4.get("final"), len(s4["rows"]), g(s4["rows"][-1]["t_over_rs"] if s4["rows"] else None, "%.3f"), st.get("nsites"), st.get("channels"),
            "; ".join("kappa %s: %d modes, all-history Gram %.3e, sea unitarity %.3e, Chebyshev order max %s, tail %.3e" % (k, v["modes"], v["gram_all_history_max"], v["sea_unitarity_max"], v["cheb_K_max"], v["cheb_tail_acc"]) for k, v in s4["archive"].items()),
            s4.get("end_reason")))
        V1.a_table(s4, L, rs)
        last = s4["rows"][-1]
        L.append("\nLabels on the table (icloud's check of v1a, inbox #48): "
                 "(1) band: the propagation's Chebyshev tail plus the sea's unitarity deviation, not a certified entropy error in nats (an entropy error needs the entropy's modulus in the dimension and rank; the collection thresholds have their own selection effects; the small-fixture oracle is evidence, not production certification). "
                 "(2) E_inv uses the instantaneous negative-energy projector P_-(T_k), not an established identification with the continued source reference; W is the finite-interval interior energy derivative plus the collar flux and Tr[h_dot (C - P)] the checkpoint global field-work contraction: their printed values do not establish the matched work identity (field-jump, exterior-work and local-reference terms remain); the energy-identity and collar-identity residual columns are algebraically the same residual at the same clock, not two balance tests. "
                 "The mode/contact split at the last clock T/r_s %.3f: E_remove %.8f beside M_chi %.8f (copied): the mode ledger alone is not positive escaped energy. "
                 "(3) The archive is the velocity-selected collected set, whose modes may return; it is not an asymptotic radiation space.\n" % (last["t_over_rs"], last["E_remove_weighted"], last["M_chi_weighted"]))
        pl = V1.peak_levels(s4)
        J["A_s4"] = {"final": s4.get("final"), "rows": s4["rows"], "peak": pl}
        if pl:
            L.append("\nReadout 10: the largest S_esc %.6f nats (band %.2e) at T/r_s %.3f (2gE %.4f); against eta A/a^2 %s: %s (script); against A/(4G) %s: %s (script); at the end T/r_s %.3f S_esc %.6f%s.\n" % (
                pl["S_esc_peak"], pl["band_at_peak"], pl["T_over_rs_peak"], pl["two_gE_at_peak"], g(pl["eta_A_over_a2"], "%.1f"), g(pl["ratio_eta"], "%.4e"), g(pl["A_over_4G"], "%.1f"),
                g(pl["ratio_A4G"], "%.4e"), pl["T_over_rs_end"], pl["S_esc_end"], " (below the largest)" if pl["falls_after_peak"] else " (no fall)"))
        try:
            sp, L_ext = spectral_flagged(s4)
            J["A_s4"]["spectral_script"] = sp; J["A_s4"]["spectral_L_ext"] = L_ext
            readout9_md(sp, L_ext, rs, L)
        except Exception as ex:                                             # noqa: BLE001
            L.append("\n- readout 9 not computed: %s: %s" % (type(ex).__name__, ex))
        L.append("\nThe geometry's work accumulated per record (engine spec 38.1 (3)): page_history_v6f.py prints it beside the finite-difference W; s4 ran on page_history_v6b.py and carries the checkpoints' trapezoid only (A extended to T/r_s 100 runs on page_history_v6i.py, which keeps every v6f readout).\n")
    else:
        L.append("not landed.")
    # A on the sigma/a 8 history: beside s4, never averaged (#36 A 14)
    s8 = S.get("s8")
    L.append("\n## 1b. A on the sigma/a 8 history (v6fA_s8, page_history_v6f.py; C3lad27 c27 ell8; s4's settings; its own extra checkpoints at T/r_s 15 and 28): a different object, printed beside s4, never averaged with it\n")
    if s8:
        L.append("Run: final %s; %d checkpoints to T/r_s %s; the archive %s; end %s.\n" % (
            s8.get("final"), len(s8["rows"]), g(s8["rows"][-1]["t_over_rs"] if s8["rows"] else None, "%.3f"),
            "; ".join("kappa %s: %d modes, all-history Gram %.3e, sea unitarity %.3e" % (k, v["modes"], v["gram_all_history_max"], v["sea_unitarity_max"]) for k, v in s8["archive"].items()),
            s8.get("end_reason")))
        V1.a_table(s8, L, rs)
        L.append("\nThe work accumulated per record (page_history_v6f.py): W_acc beside the finite-difference W and the checkpoints' trapezoid, with the jump check (the summed jumps against the change of Tr[h C]); W_acc equals dE_inv/dt of the whole chain by construction, so W - W_acc is L minus the change of the energy beyond the collar.\n")
        L.append("| T/r_s | W (finite difference) | W_acc (per record) | the checkpoints' trapezoid of Tr[h_dot (C - P)] | the jump check (weighted) |")
        L.append("|---|---|---|---|---|")
        for r in s8["rows"]:
            L.append("| %.3f | %s | %s | %s | %s |" % (r["t_over_rs"], g(r.get("W_creation"), "%.4e"), g(r.get("W_acc_rate"), "%.4e"), g(r.get("W_hdot_trapezoid"), "%.4e"), g(r.get("jump_check_weighted"), "%.3e")))
        pl8 = V1.peak_levels(s8)
        J["A_s8"] = {"final": s8.get("final"), "rows": s8["rows"], "peak": pl8}
        if pl8:
            L.append("\nThe largest S_esc %.6f nats at T/r_s %.3f (2gE %.4f); at the end T/r_s %.3f S_esc %.6f%s." % (
                pl8["S_esc_peak"], pl8["T_over_rs_peak"], pl8["two_gE_at_peak"], pl8["T_over_rs_end"], pl8["S_esc_end"], " (below the largest)" if pl8["falls_after_peak"] else " (no fall)"))
    else:
        L.append("not landed.")
    L.append("\n## 2. The transfer ledger (readout 11) from the ledger run v6eA_s4tr (s4's settings from its T/r_s 15.0 state; the ledger restarts at each new S_esc maximum, the kept one starts at the run's largest S_esc)\n")
    tr = S["s4tr"]
    if tr:
        tb = V1.transfer_block(tr)
        dd = V1.variant_diff(tr, s4) if s4 else None
        J["ledger_s4tr"] = dict(tb, against_s4=dd)
        L.append("- final %s; the ledger's restarts %d, the last at T/r_s %s" % (tr.get("final"), len(tb["starts"] or []), g((tb["starts"] or [{}])[-1].get("t_over_rs") if tb["starts"] else None, "%.3f")))
        for k, v in tb["per_channel"].items():
            L.append("- kappa %s: start T/r_s %s; f %s; the partners' weights inside r_t %s, r_t to R_cut %s, beyond %s; at the end %s" % (
                k, g(v["t0"] / rs if v.get("t0") is not None else None, "%.3f"), v.get("f"), v.get("w_inside_rt"), v.get("w_rt_to_cut"), v.get("w_beyond_cut"), json.dumps(v.get("end"))))
        if dd:
            L.append("- against s4 at the %d common checkpoints (T/r_s %s to %s): max |dS_esc| %s, max |dE_inv| %s, max |dL| %s, max |d collar flow| %s" % (
                dd["common_checkpoints"], dd["T_first"], dd["T_last"], g(dd["max_dS_esc"], "%.3e"), g(dd["max_dE_inv"], "%.3e"), g(dd["max_dL_interval"], "%.3e"), g(dd["max_dcollar_flow"], "%.3e")))
    else:
        L.append("not landed.")
    L.append("- The reason the kept ledger is empty: S_esc rises at every checkpoint to the run's end, so the peak-restarted ledger restarts at every checkpoint and the one kept starts at the run's last clock; W_R is 0 by construction (a peak-anchored ledger cannot read a monotone rise). Readout 11 needs an anchor at a stated clock: page_history_v6g carries the ledger anchored at the formation's record clock (T/r_s 12.125) and accumulated to every checkpoint, the peak-anchored one beside it, for A extended to T/r_s 100 (engine spec 39.1 (2)); A to 100 runs on page_history_v6i.py, which builds both ledgers' partners on the pure state closest to the propagated one and on an orthonormal archive (the partners otherwise carry the sea's and the archive's Gram deviations divided by f(1 - f), which at the formation's clock reaches 1e-12).")
    L.append("\n## 3. A's variants against s4 (the far collector R_cut 48 beside R_cut 24; eps_loc; the kappa 3 guard; the cubic control)\n")
    J["variants"] = {}
    for n in ("s4far", "s4loc3", "s4k3", "s4cubic"):
        v = S[n]
        if not v or not s4:
            L.append("- %s (%s): not landed" % (n, V1.VARIANT_LABEL[n])); continue
        dd = V1.variant_diff(v, s4); pv = V1.peak_levels(v)
        J["variants"][n] = {"final": v.get("final"), "against_s4": dd, "peak": pv, "rows": v["rows"]}
        L.append("- %s (%s; %s): final %s; %s; the largest S_esc %s at T/r_s %s (s4 %s)" % (
            n, V1.VARIANT_LABEL[n], v.get("code"), v.get("final"),
            ("at the %d common checkpoints (T/r_s %s to %s) max |dS_esc| %s, max |dE_inv| %s, max |dL| %s; at the last S_esc %s against %s" % (
                dd["common_checkpoints"], dd["T_first"], dd["T_last"], g(dd["max_dS_esc"], "%.4e"), g(dd["max_dE_inv"], "%.4e"), g(dd["max_dL_interval"], "%.4e"),
                g(dd["S_esc_at_last"][0], "%.6f"), g(dd["S_esc_at_last"][1], "%.6f"))) if dd else "no common checkpoint",
            g(pv and pv["S_esc_peak"], "%.6f"), g(pv and pv["T_over_rs_peak"], "%.3f"), g((V1.peak_levels(s4) or {}).get("S_esc_peak"), "%.6f")))
    L.append("\n## 4. The fixed-field controls C (page_history_v6e.py: the record's h held fixed from the hold's start, its one eigendecomposition carrying the state by exact phases; S1's drain rule; the reflection clock)\n")
    J["C"] = {}
    for n in ("c15", "c30"):
        c = S[n]
        if not c:
            L.append("- %s: not landed" % n); continue
        cb = V1.c_block(c)
        J["C"][n] = dict(cb or {}, final=c.get("final"), rows=[r for r in c["rows"] if r.get("frozen")])
        L.append(c_line(n, c, cb) if cb else "- %s: no fixed-field checkpoint yet" % n)
        if cb:
            hr = hold_readouts(c, cb)
            J["C"][n]["hold_readouts"] = hr
            L.append("  - E_inv over the hold: at most %.3e from E_inv(0) (conserved, h held fixed); its interior part (inside the collar) %.6f at the hold's start, %.6f at T/r_s %.3f, the last clean checkpoint before the reflection clock T/r_s %.1f: %.3e per r_s (script); a linear extrapolation of that rate empties it in about %s r_s after the hold's start (script; a linear extrapolation, untested; the chain's clean reading ends at the reflection clock)." % (
                hr["E_inv_max_dev"], hr["inside_start"], hr["inside_clean_end"], hr["T_clean_end"], hr["T_refl"], hr["inside_rate_per_rs"], g(hr["linear_empty_rs"], "%.0f")))
            L.append("  - L/L_H(kappa_T) over the clean reading from T/r_s %.0f to the reflection clock %.1f: %s to %s at every checkpoint (script); the hold table below gives every checkpoint." % (
                hr["clean_window_from"], hr["T_refl"], g(hr["L_clean_min"], "%.3g"), g(hr["L_clean_max"], "%.3g")))
            L.append("  - S_esc over the clean reading (the checkpoints before the reflection clock T/r_s %.1f): the largest %.6f at T/r_s %.3f, %.6f at the last clean checkpoint T/r_s %.3f (script); the %d rows after the reflection clock read the chain end's returns as well." % (
                hr["T_refl"], hr["S_clean_peak"], hr["T_clean_peak"], hr["S_clean_end"], hr["T_clean_end"], hr["rows_after_reflection"]))
            L.append("\n  | T/r_s | E_inv | inside the collar | beyond | collar flow (acc.) | L (interval, per Delta T) | L/L_H(kappa_T) | L/L_H(T_H) | S_esc | after the reflection clock |")
            L.append("  |---|---|---|---|---|---|---|---|---|---|")
            for r in [q for q in c["rows"] if q.get("frozen")]:
                L.append("  | %.3f | %.7e | %.7e | %.4e | %.6e | %s | %s | %s | %.6f | %s |" % (r["t_over_rs"], r["E_inv_weighted"], r["E_inv_inside_weighted"], r["E_out_weighted"], r["collar_flow_acc_weighted"],
                         g(r.get("L_interval"), "%.4e"), g(r.get("L_over_LH_kappaT"), "%.4g"), g(r.get("L_over_LH_TH"), "%.4g"), r["S_esc_weighted"], "yes" if r.get("after_reflection_clock") else "no"))
    if S.get("c15") and S.get("c30"):
        L.append("- The rule's entropy clause is not reached within the chain's clean reading on either record; the drains are in progress (engine spec 39.1 (1)).")
    L.append("\n## 5. Validation (S1 E; engine spec 38.1 (1): page_*_v6 to v6e validated for A)\n")
    st6 = Vd.get("selftest_v6b")
    if st6:
        np_ = sum(1 for r in st6["results"] if r.get("pass") is True); nf = sum(1 for r in st6["results"] if r.get("pass") is False)
        L.append("- The self-test v6b (code %s): all_pass %s, %d PASS, %d FAIL, %d tests, %.0f s on %s." % (st6.get("code"), st6.get("all_pass"), np_, nf, len(st6["results"]), st6.get("seconds", float("nan")), st6.get("device")))
    for n in ("oracle_v6b", "oracle_v6c", "oracle_v6d", "oracle_v6e", "oracle_v6f"):
        o = Vd.get(n)
        if not o:
            L.append("- the history oracle %s: not landed" % n[-3:]); continue
        L.append("- The history oracle %s (code %s; main checks all pass %s): %s; the cubic control %s; the fixed-field control's worst %s%s%s." % (
            n[-3:], o.get("code"), all(v["pass"] for v in o["checks"].values()), ", ".join("%s %.3e (%g)" % (k, v["worst"], v["criterion"]) for k, v in o["checks"].items()),
            json.dumps({k: o["cubic_control"].get(k) for k in ("t_over_rs", "dS", "dflow")}) if o.get("cubic_control") else "-",
            json.dumps({k: o["frozen_control"]["worst"].get(k) for k in ("S_esc", "state_F", "gram", "collar_flow")}) if o.get("frozen_control") else "-",
            ("; the eigenbasis against the Chebyshev path: %s" % json.dumps({k: o["frozen_eig_vs_cheb"].get(k) for k in ("rows", "S_esc_max_abs", "E_inv_max_abs", "collected_equal")})) if o.get("frozen_eig_vs_cheb") else "",
            ("; the two-channel ledger run ok %s" % (o.get("two_channel_ledger") or {}).get("ok")) if o.get("two_channel_ledger") else ""))
    L.append("- The v6 regression on the pilot's prescribed background (stage 0 uncapped; results/page/page_v6_regression_v1.md): " + "; ".join(
        "%s: %s" % (n, ("S_max %.6f at t %.1f r_s1, S_esc at the end %.6f, E_rad at the end %.6e, final %s, code %s" % (
            max(x["S_esc_total"] for x in r_["curve"]), max(r_["curve"], key=lambda x: x["S_esc_total"])["t_over_rs1"], r_["curve"][-1]["S_esc_total"], r_["curve"][-1]["E_rad_stage"],
            r_.get("final"), r_.get("code", "page_recursion_v5.py"))) if r_ and r_.get("curve") else "not landed") for n, r_ in Rg.items()) + ".")
    if RD:
        for p in RD["paths"]:
            ar = [r for r in p["after_restore"]]
            miss = [r[0] for r in ar if isinstance(r[1], str) and "missing" in r[1]]
            L.append("- The restore diff (code/page/restore_diff_page_recursion_v1.py, %s, %s): path %s: right after the restore %s; after %d more blocks %d fields differ, S_esc by at most %.3e; %s (a structure difference printed as inf is a cache the code fills on demand)." % (
                RD.get("recursion"), RD.get("case"), p["path"], "; ".join("%s %s" % (r[0], r[1] if isinstance(r[1], str) else "%.3e" % r[1]) for r in ar),
                RD.get("blocks_after"), len(p["after_more_blocks"]), p["S_esc_per_block"]["max_abs_diff_common"] or 0.0,
                ("fields missing after the restore: %s" % ", ".join(miss)) if miss else "no field is missing after the restore"))
    J["validation"] = {"selftest_v6b": {"all_pass": (st6 or {}).get("all_pass")}, "oracles": {n: (Vd.get(n) or {}).get("checks") for n in Vd},
                       "restore_diff": RD, "regression": {n: ({"S_max": max(x["S_esc_total"] for x in r_["curve"]), "final": r_.get("final")} if r_ and r_.get("curve") else None) for n, r_ in Rg.items()}}
    if s4 and s4["rows"]:
        J["figure"] = figure_v2(s4, H, rs, J)
        J["figure"]["draft_label"] = draft
        mk = J["figure"]["marks"]["v2"]
        L.append("\n## 6. The figure's marks (ITEM7_fig_page_draft_v2.png; engine spec 38.1 (2))\n")
        L.append("- the axis: %s; the formation's record clock T/r_s %s (%s); the formation's arrival at the collar: the half-collar-flow clock T/r_s %s, on the collar's clock %s (%s); the light-travel band on the collar's clock %s (%s; over 2gE %s at 2gE %s); the ray-map epoch's end at the collar %s (%s; T_d/r_s %s, %s); the collar flow over the run %s." % (
            mk["axis"], g(mk["formation_record_clock_over_rs"], "%.3f"), mk["formation_record_clock_definition"], g(mk["arrival_half_flow_t_over_rs"], "%.3f"),
            g(mk["arrival_half_flow_t_inf_over_rs"], "%.3f"), mk["arrival_definition"], json.dumps(mk["light_travel_band_t_inf_over_rs"]), mk["light_travel_band_definition"],
            json.dumps(mk["light_travel_over_2gE"]), g(mk["two_gE_formation"], "%.4f"), g(mk["raymap_end_t_inf_over_rs"], "%.3f"), mk["raymap_end_definition"],
            g(mk["T_d_over_rs"], "%.2f"), mk["T_d_source"], g(mk["collar_flow_total"], "%.6e")))
    L.append("\n## 7. Not in this file\n")
    L.append("- D (the massive channel's own emission from the stored records) and B (the engine continuation to T/r_s 100) are other builders' tasks. A to T/r_s 100 (page_history_v6i.py from T = 0 with s4's settings: the production's records to T/r_s 40, B's continuation after it, read as its records land; the transfer ledger anchored at the formation's record clock T/r_s 12.125 beside the peak-anchored one) runs on pod 2 beside the sigma/a 2 ladder member (inbox #54, the owner; moved from pod 1 lane 1's wait) and is version 3's; A on the sigma/a 8 history is section 1b.")
    L.append("- S1 A.12 is replaced by S4's A.3 (#41 E), another builder's; no mutual information is printed here.")
    text = "\n".join(L) + "\n"
    cnt = {}
    text = replace_all(text, cnt)
    text = text.replace("Written %s by code/page/page_solved_history_report_v6.py" % now,
                        "Written %s by code/page/page_solved_history_report_v6.py (the wording net of requirements_new.txt sec. 4, counted by the replacing words: %s)" % (
                            now, "; ".join("\"%s\" %d" % (k, v) for k, v in cnt.items()) or "no replacement"), 1)
    with open(p_md, "w") as f:
        f.write(text)
    cj = {}
    J = walk(J, cj)
    J["wording_replacements"] = {"md": cnt, "json": cj}
    with open(p_json, "w") as f:
        json.dump(J, f, indent=1, default=lambda o: float(o) if isinstance(o, (int, float)) else str(o))
    print("wrote %s and %s: %s; wording replacements md %s, json %s" % (p_md, p_json, ", ".join("%s %s" % (n, "landed" if v else "-") for n, v in S.items()), cnt, cj))


if __name__ == "__main__":
    main()
