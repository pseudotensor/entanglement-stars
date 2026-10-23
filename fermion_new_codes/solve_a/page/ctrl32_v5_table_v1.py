#!/usr/bin/env python
"""ctrl32_v5_table_v1.py (builder analyst, 2026-10-16; SPEC_final4b_engine.md 36.1 "Ruled" (4), (5) and 36.5 item 1).

Writes results/page/ctrl32_v5_table_v1.md, the Page pilot's controls' table (read in turn 45), from the controls' outputs as
fetch_ctrl32_v5_v1.sh copies them into results/page/ctrl32_v5_runs/ (the summaries launch_page_controls_v5_v1.sh mirrors from pod 1's
container disk to <internal>/ctrl32_v5_<name>/, lane 1's status lines) and the pilot's own summary (its stage 0 beside them).
Per control: S_esc(t) at the block times beside the pilot's stage 0, the peak and its time, S_esc at the end, E_rad at the end over
E_init (weighted), E_inv/E per channel, the uncollected weight's maximum; from the summaries where present: the memory rank over the
one-particle dimension per checkpoint (P06's rank over support, the builder's reading: the support dimension taken as 2 nsites) and the
drained share of the formation's inverted population by energy bin (the end spectrum). Every number is printed by this script; R4: the
turnover flags are not printed or cited. Re-run fetch then this script as each control lands.

Usage: python ctrl32_v5_table_v1.py RUNS_DIR PILOT_SUMMARY OUT_MD"""
import datetime
import glob
import json
import os
import re
import sys

ORDER = ["baseline", "ell8", "point", "Rb0_8", "ramp", "n4096", "cut48", "taus2", "coll4", "coll6", "kappa3", "Rb08"]
CHANGED = {"baseline": "none (the pilot's settings, uncapped)", "ell8": "--ell 8", "point": "--fields point", "Rb0_8": "--Rb0 8", "ramp": "--formation ramp",
           "n4096": "--nsites 4096 (the short chain)", "cut48": "--R-cut 48", "taus2": "--tau-s 2", "coll4": "--eps-coll 1e-4", "coll6": "--eps-coll 1e-6",
           "kappa3": "--channels 1,2,3", "Rb08": "--Rb 0.8"}
EDGES = [0, 1, 2, 3, 4, 6, 8, 12, 25, 1e9]


def g(x, fmt="%.4e"):
    if x is None:
        return "-"
    try:
        return fmt % x
    except (TypeError, ValueError):
        return str(x)


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def wts(s):
    return {str(k): 16.0 * float(k) for k in s["settings"]["channel_list"]}


def mi_end(s):
    """(script) the weighted physical and reference-vacuum MI across r_t at the end, per d."""
    rd = s.get("readouts") or {}
    w = wts(s); out = {}
    for k, r in rd.items():
        for c in r.get("cross_surface") or []:
            d = c["d"]; p, v = out.get(d, (0.0, 0.0))
            out[d] = (p + w[k] * c["physical"]["I"], v + w[k] * c["reference_vacuum"]["I"])
    return out


def main():
    runs, p_pilot, p_out = sys.argv[1:4]
    pilot = load(p_pilot)
    status_lines = []
    sp = os.path.join(runs, "h100b_status_ctrl32_lines.txt")
    if os.path.exists(sp):
        status_lines = [ln.strip() for ln in open(sp, encoding="utf-8", errors="replace") if ln.strip()]
    fetched = open(os.path.join(runs, "FETCHED_AT.txt")).read().strip() if os.path.exists(os.path.join(runs, "FETCHED_AT.txt")) else "-"
    S = {}
    for n in ORDER:
        S[n] = load(os.path.join(runs, "page", "runs", "ctrl32_v5_%s" % n, "ctrl32_v5_%s_summary.json" % n))
    starts = {}; exits = {}
    for ln in status_lines:
        m = re.match(r"lane1 page_v5_ctrl32_(\S+) start (\S+) UTC", ln)
        if m:
            starts[m.group(1)] = m.group(2)
        m = re.match(r"lane1 page_v5_ctrl32_(\S+) exit (\S+) (\S+) UTC", ln)
        if m:
            exits[m.group(1)] = (m.group(2), m.group(3))
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = []
    L.append("# The Page pilot's controls at r_s/a 32 (page_*_v5, one stage each, the collection uncapped): the table of engine spec 36.1 (4)\n")
    L.append("Written %s by code/page/ctrl32_v5_table_v1.py from the controls' mirrored outputs (%s; code/page/fetch_ctrl32_v5_v1.sh) and the pilot's summary %s. Every number is printed by the script; \"(script)\" marks a ratio computed here. Launcher: code_v5/launch_page_controls_v5_v1.sh on pod 1 lane 1, one control per queue line in the ruled order, each with the pilot's line and --dE-frac 1.0 --max-stages 1 --t-stage-max 400 --m-cap 16384 --resume.\n" % (now, fetched, p_pilot))
    L.append("Labels (36.1 (5)): the D8 and formation allowances are not bounds as written (icloud review R1 and R2); R3: FIFO pruning by present weight does not certify the all-history algebra (the FIFO exit count per control is printed below); R4: the turnover flags are not certificates and are not printed here; nothing here is a Page turnover.\n")

    # 1 status
    L.append("## 1. The controls: settings, start, end\n")
    L.append("| control | changed option | queue start (UTC) | exit (code, UTC) | final | stop | blocks | stage end | duration/r_s1 | wall s | m_cap_used | max r_before (all blocks) | the cap could bind | validation status |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n in ORDER:
        s = S[n]
        ex = exits.get(n)
        if s is None:
            L.append("| %s | %s | %s | %s | - | - | - | - | - | - | - | - | - | %s |" % (n, CHANGED[n], starts.get(n, "-"), ("%s %s" % ex) if ex else "-",
                                                                                "running or queued (no mirrored summary yet)" if n in starts and not ex else ("not started" if n not in starts else "no summary")))
            continue
        st = (s.get("stages") or [{}])[-1] if s.get("stages") else {}
        caps = sorted({pc.get("m_cap_used") for r in s["curve"] for pc in r["per_channel"].values() if pc.get("m_cap_used") is not None})
        rmax = max((pc.get("r_before") or 0) for r in s["curve"] for pc in r["per_channel"].values()) if s["curve"] else None
        bind = sum(1 for r in s["curve"] for pc in r["per_channel"].values() if (pc.get("r_before") or 0) > (pc.get("m_cap_used") or 0))
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            n, CHANGED[n], starts.get(n, "-"), ("%s %s" % ex) if ex else "-", s.get("final"), s.get("stop_reason"), s.get("n_blocks"), st.get("end", "-"),
            g(st.get("duration_over_rs1"), "%.1f"), g(s.get("wall_s"), "%.0f"), caps, rmax, bind, (s.get("validation") or {}).get("status", "-")))
    L.append("")

    # 2 the readouts
    L.append("## 2. The readouts per control (the pilot's stage 0 first)\n")
    L.append("| run | S_max (nats) | t at S_max /r_s1 | S_esc at the end | t end /r_s1 | E_rad at the end | E_init (weighted) | E_rad/E_init (script) | E_inv/E per channel | uncollected weight max (over the active norm) | L_mean/L_cont (script) | MI across r_t at the end, d 2: physical - vacuum (script) | FIFO exits | direct route max rel | energy identity max rel |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    if pilot:
        c0 = [r for r in pilot["curve"] if r["stage"] == 0]
        m0 = max(c0, key=lambda r: r["S_esc_total"])
        st0 = pilot["stages"][0]
        uc0 = max((pc.get("weight_beyond_cut_uncollected", 0.0) / max(pc.get("active_norm", 1.0) or 1.0, 1.0)) for r in c0 for pc in r["per_channel"].values())
        mi0 = {}
        for k, lst in (st0.get("cross_surface_end") or {}).items():
            for c in lst:
                p, v = mi0.get(c["d"], (0.0, 0.0)); mi0[c["d"]] = (p + 16.0 * float(k) * c["physical"]["I"], v + 16.0 * float(k) * c["reference_vacuum"]["I"])
        L.append("| pilot stage 0 (cap 64, ended reached dE) | %.5f | %.1f | %.5f | %.1f | %.4e | %.4e | %.5f | %s | %.3e | %.2f | %s | %s | %s | %s |" % (
            m0["S_esc_total"], m0["t_over_rs1"], c0[-1]["S_esc_total"], c0[-1]["t_over_rs1"], st0["E_rad"], st0["E_init_weighted"], st0["E_rad"] / st0["E_init_weighted"],
            ", ".join("%s: %.4e" % (k, v["E_inv_over_E"]) for k, v in pilot["inversion"].items()), uc0, st0["L_mean"] / st0["L_cont"],
            g((mi0.get(2, (None, None))[0] - mi0[2][1]) if 2 in mi0 else None, "%.3e"), "-", "stage 0: see the class-D record", "-"))
    for n in ORDER:
        s = S[n]
        if s is None:
            continue
        cv = s["curve"]
        if not cv:
            L.append("| %s | (no block yet) |||||||||||||| " % n)
            continue
        mx = max(cv, key=lambda r: r["S_esc_total"])
        st = s["stages"][-1] if s.get("stages") else None
        E_rad = st["E_rad"] if st else cv[-1]["E_rad_stage"]
        E_init = st["E_init_weighted"] if st else None
        mi = mi_end(s)
        c = s["checks"]
        L.append("| %s | %.5f | %.1f | %.5f | %.1f | %.4e | %s | %s | %s | %.3e | %s | %s | %s | %s | %s |" % (
            n, mx["S_esc_total"], mx["t_over_rs1"], cv[-1]["S_esc_total"], cv[-1]["t_over_rs1"], E_rad, g(E_init, "%.4e"), g(E_rad / E_init if E_init else None, "%.5f"),
            ", ".join("%s: %.4e" % (k, v["E_inv_over_E"]) for k, v in s["inversion"].items()), c.get("uncollected_over_active_max", 0.0),
            g(st["L_mean"] / st["L_cont"] if st else None, "%.2f"), g((mi[2][0] - mi[2][1]) if 2 in mi else None, "%.3e"), c.get("fifo_exits"),
            g(c.get("direct_vs_memory_max_rel"), "%.3e"), g(c.get("energy_identity_max_rel"), "%.3e")))
    L.append("\nE_rad is the memory route's stage emission (weighted); E_init the stage's initial memory energy (weighted), which equals the formation's weighted inversion energy for a one-stage run. Unfinished controls show their latest block.\n")

    # 3 S_esc(t)
    L.append("## 3. S_esc(t) at the block times (nats; t over r_s1; the pilot's stage 0 beside the controls)\n")
    present = [n for n in ORDER if S[n] is not None and S[n]["curve"]]
    times = set()
    if pilot:
        times |= {round(r["t_over_rs1"], 6) for r in pilot["curve"] if r["stage"] == 0}
    for n in present:
        times |= {round(r["t_over_rs1"], 6) for r in S[n]["curve"]}
    col = {}
    if pilot:
        col["pilot stage 0"] = {round(r["t_over_rs1"], 6): r["S_esc_total"] for r in pilot["curve"] if r["stage"] == 0}
    for n in present:
        col[n] = {round(r["t_over_rs1"], 6): r["S_esc_total"] for r in S[n]["curve"]}
    names = list(col.keys())
    if names:
        L.append("| t/r_s1 | " + " | ".join(names) + " |")
        L.append("|---|" + "---|" * len(names))
        for t in sorted(times):
            L.append("| %g | %s |" % (t, " | ".join(g(col[nm].get(t), "%.5f") for nm in names)))
    else:
        L.append("(no control has a block yet)")
    L.append("")

    # 4 P06 and the drained share
    L.append("## 4. From the summaries where present: the memory rank over the one-particle dimension (P06) and the drained share by energy bin\n")
    L.append("P06's occupation histogram per checkpoint is not in the summaries (only the end spectrum of the stage record), so it is not printed. The rank: the memory rank per channel (r_after) over 2 nsites, the builder's reading of P06's support dimension.\n")
    for n in present:
        s = S[n]; cv = s["curve"]; n2 = 2 * int(s["settings"]["nsites"])
        mx = max(cv, key=lambda r: r["S_esc_total"])
        parts = []
        for k in [str(x) for x in s["settings"]["channel_list"]]:
            parts.append("kappa %s: %d -> %d (at S_max) -> %d, over %d: %.4f -> %.4f -> %.4f" % (
                k, cv[0]["per_channel"][k]["r_after"], mx["per_channel"][k]["r_after"], cv[-1]["per_channel"][k]["r_after"], n2,
                cv[0]["per_channel"][k]["r_after"] / n2, mx["per_channel"][k]["r_after"] / n2, cv[-1]["per_channel"][k]["r_after"] / n2))
        L.append("- %s: memory rank first block -> at S_max -> last block: %s." % (n, "; ".join(parts)))
    L.append("\nThe drained share by bin (script): the end spectrum's collected excess occupations (particles plus hole deficits) summed per omega/T_H bin over the formation's inverted population in the same bin; printed only for a final run whose stage record is complete (every collected mode stored, store cap 6000 per channel).\n")
    for n in present:
        s = S[n]
        if not s.get("final"):
            continue
        for k in [str(x) for x in s["settings"]["channel_list"]]:
            rd = (s.get("readouts") or {}).get(k) or {}
            sd = rd.get("stage_direct") or {}
            sp = rd.get("spectrum") or []
            inv = s["inversion"][k]["inverted_by_bin_TH"]
            if not sp or not sd.get("complete"):
                L.append("- %s kappa %s: not printed (spectrum entries %d, stage record complete %s)." % (n, k, len(sp), sd.get("complete")))
                continue
            cells = []
            for i in range(len(EDGES) - 1):
                tot = sum(abs(x[1]) for x in sp if EDGES[i] <= abs(x[0]) < EDGES[i + 1])
                cells.append("[%g,%s) %s/%s = %s" % (EDGES[i], "inf" if EDGES[i + 1] >= 1e9 else "%g" % EDGES[i + 1], g(tot, "%.3e"), g(inv[i], "%.3e"), g(tot / inv[i] if inv[i] > 0 else None, "%.3e")))
            L.append("- %s kappa %s (%d modes): %s." % (n, k, len(sp), "; ".join(cells)))
    L.append("\nLane 1's status lines for the controls (verbatim):\n")
    for ln in status_lines:
        L.append("    " + ln)
    with open(p_out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("wrote %s: %d controls with a summary (%s)" % (p_out, len(present), ", ".join(present)))


if __name__ == "__main__":
    main()
