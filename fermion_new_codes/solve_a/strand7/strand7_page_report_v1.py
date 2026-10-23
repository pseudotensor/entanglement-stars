#!/usr/bin/env python
"""Item 7's report (SPEC_strand7_page.md secs. 6, 7, 9 with sec. 12's amendments; builder analyst): reads the sea runs' records
(strand7_sea_page_v1/v3.py, s7p_<tag>.json) and writes the emitted summary and the tables as markdown, plus a JSON of the tests:
  T7p2  the primary P (L; channels 1/2, 3/2, 5/2), the 2j + 1 weighted sums per output: dS^(ad) and dS^(0) at the radiation cut, S_diag,rad,
        N_ext, dS^(ad) at the throat, the shell cut and the barrier, S'_in at the throat (A2), I_w at the throat, g(h), g(2h), g(4h) at the
        throat, g(4h), g(8h), g(16h) at the radiation cut (ratios of the weighted sums of G), N_s and the shell cut's areal radius;
  T7p1  per channel (compact): dS^(ad) at the four cuts, S'_in at the throat, g(h) at the throat, N_ext, S_diag,rad;
  T7p3  the spectrum per channel at every 4 r_s0 and at T_1, T_2: N_ext, dN_ext/dT, S_diag,rad, S_diag,near, the bins' mean occupations
        (particles, 1 T_H bins over [0, 8] T_H), the upper edge (the largest bin with mean above 1e-3), the raw-odds slope and intercept
        (labelled: contains the transmission; not a temperature), the channel ratios of the bin means;
  T7p4  the window: T_1, T_2(omega) for omega = 1, 2, 4, 8 T_H and N_s at the outputs;
  T7p5  the controls C-pg-1 to C-pg-14 (C-pg-2 needs T = 0 rows, which the runs do not record: stated);
  T7p6  the checks K1, K2, H, R, J against P (j = 1/2 unless stated);
  T7p7  the comparison C (the same sums as T7p2, its own outputs);
  T7p9  L's trajectory: s/A_0, N_s, the shell cut's areal radius, the energy control and the late law (C-pg-14);
  Tests P1 to P4 (sec. 7 as amended by sec. 12.5), applied mechanically as written, with the floor from C-pg-8 and C-pg-9 (the larger of the
  cadence and the h-halving changes of the same quantity, j = 1/2); no reading.
Arguments: --dir (the runs' folder), --f0 (F0's JSON, run separately), --frozen (frozen_schw E8's S7a JSON for C-pg-14), --paper-h2 (a copy of
the paper's lattice_entropy.py for C-pg-6, optional), --md, --out-json."""
import argparse
import glob
import importlib.util
import json
import math
import os

import numpy as np

CUTS = ("throat", "shell", "barrier", "rad")


def f(x, d=4):
    if x is None:
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    x = float(x)
    if not math.isfinite(x):
        return str(x)
    if x == 0:
        return "0"
    if 1e-3 <= abs(x) < 1e5:
        return ("%." + str(d) + "g") % x
    return "%.3e" % x


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows])


def load(p):
    with open(p) as fh:
        return json.load(fh)


def merged(records):
    """channels merged over several JSONs: {j: {T_over_rs0: row}}"""
    ch = {}
    for R in records:
        for k, v in R["channels"].items():
            ch[float(k)] = {round(q["T_over_rs0"], 6): q for q in v}
    return ch


def wsum(ch, T, fn):
    tot = 0.0
    for j, rows in ch.items():
        q = rows.get(T)
        if q is None:
            return None
        tot += (2 * j + 1) * sum(fn(s) for s in q["sectors"].values())
    return tot


def common_T(ch):
    return sorted(set.intersection(*[set(v.keys()) for v in ch.values()])) if ch else []


def gratio(ch, T, cut, d):
    num = wsum(ch, T, lambda s: s["cuts"][cut]["G_state_d%g" % d])
    den = wsum(ch, T, lambda s: s["cuts"][cut]["G_vac0_d%g" % d])
    return (num / den) if (num is not None and den) else None


def sums(ch, h):
    du = h if h >= 1 else 1.0
    out = []
    for T in common_T(ch):
        q0 = list(ch.values())[0][T]
        r = {"T": T, "dS_rad": wsum(ch, T, lambda s: s["cuts"]["rad"]["dSad"]), "dS0_rad": wsum(ch, T, lambda s: s["cuts"]["rad"]["dS0"]),
             "S_diag_rad": wsum(ch, T, lambda s: s["spectrum"]["S_diag_rad"]), "N_ext": wsum(ch, T, lambda s: s["spectrum"]["N_ext"]),
             "dS_thr": wsum(ch, T, lambda s: s["cuts"]["throat"]["dSad"]),
             "dS_shell": wsum(ch, T, lambda s: s["cuts"]["shell"]["dSad"]) if "shell" in q0["sectors"]["alpha"]["cuts"] else None,
             "dS_bar": wsum(ch, T, lambda s: s["cuts"]["barrier"]["dSad"]), "Sde_thr": wsum(ch, T, lambda s: s["cuts"]["throat"]["S_de_int_state"]),
             "MI_thr": wsum(ch, T, lambda s: s["cuts"]["throat"]["MI_state"]),
             "g_h": gratio(ch, T, "throat", 1 * du), "g_2h": gratio(ch, T, "throat", 2 * du), "g_4h": gratio(ch, T, "throat", 4 * du),
             "g_rad_4h": gratio(ch, T, "rad", 4 * du), "g_rad_8h": gratio(ch, T, "rad", 8 * du), "g_rad_16h": gratio(ch, T, "rad", 16 * du),
             "N_s": (q0.get("shell") or {}).get("N_s"), "areal_shell_cut": (q0.get("shell") or {}).get("areal_shell_cut")}
        out.append(r)
    return out


def at(rows, T, key, tol=1e-6):
    for r in rows:
        if abs(r["T"] - T) < tol:
            return r[key]
    return None


def slope(rows, key, a, b):
    sel = [r for r in rows if a - 1e-9 <= r["T"] <= b + 1e-9 and r[key] is not None]
    if len(sel) < 2:
        return None, None
    return (sel[-1][key] - sel[0][key]) / (sel[-1]["T"] - sel[0]["T"]), (sel[0]["T"], sel[-1]["T"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--f0", default=None)
    ap.add_argument("--frozen", default=None)
    ap.add_argument("--paper-h2", default=None)
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    R = {os.path.basename(p)[4:-5]: load(p) for p in sorted(glob.glob(os.path.join(a.dir, "s7p_*.json")))}
    P_tags = [t for t in R if t.startswith("P_j")]
    C_tags = [t for t in R if t.startswith("C_j")]
    L = ["# Item 7's tables (strand7_page_report_v1.py over %s)" % a.dir, ""]
    summ, js = [], {"runs": sorted(R)}
    h = 1.0
    P = merged([R[t] for t in P_tags]) if P_tags else {}
    Prows = sums(P, h) if P else []
    win = R[P_tags[0]]["window"] if P_tags else {}
    T1 = win.get("T_1") / R[P_tags[0]]["meta"]["r_s0"] if win.get("T_1") else None
    T2 = win.get("T_2_1TH") / R[P_tags[0]]["meta"]["r_s0"] if win.get("T_2_1TH") else None
    rs0 = R[P_tags[0]]["meta"]["r_s0"] if P_tags else None
    # ---- T7p2
    keys = ["dS_rad", "dS0_rad", "S_diag_rad", "N_ext", "dS_thr", "dS_shell", "dS_bar", "Sde_thr", "MI_thr", "g_h", "g_2h", "g_4h", "g_rad_4h", "g_rad_8h",
            "g_rad_16h", "N_s", "areal_shell_cut"]
    L += ["## T7p2 The primary P on L: the 2j + 1 weighted sums per output (channels %s)" % sorted(P), "",
          table(["T/r_s0"] + keys, [[f(r["T"], 5)] + [f(r[k]) for k in keys] for r in Prows]), ""]
    js["P_sums"] = Prows
    # ---- tests P1 to P4
    floor = {}
    tests = {}
    if Prows and T1 and T2:
        # the floor (C-pg-8, C-pg-9) on j = 1/2
        P05 = {0.5: P[0.5]} if 0.5 in P else {}
        p05 = sums(P05, h) if P05 else []
        chk = {}
        for t in ("K1", "K2", "H", "R", "J"):
            if t in R:
                chk[t] = sums(merged([R[t]]), R[t]["meta"]["h"])
        for k in ("dS_rad", "dS_thr", "S_diag_rad", "N_ext", "g_h"):
            ch_ = []
            for t in ("K1", "K2"):
                if t in chk:
                    v1, v0 = at(chk[t], 36.0, k), at(p05, 36.0, k)
                    if v1 is not None and v0 is not None:
                        ch_.append(abs(v1 - v0))
            hh = []
            if "H" in chk:
                for Tq in (20.0, 24.0, 30.0, 36.0, 40.0):
                    v1, v0 = at(chk["H"], Tq, k), at(p05, Tq, k)
                    if v1 is not None and v0 is not None:
                        hh.append(abs(v1 - v0))
            floor[k] = {"cadence": max(ch_) if ch_ else None, "h_halving": max(hh) if hh else None,
                        "floor": max([x for x in (max(ch_) if ch_ else None, max(hh) if hh else None) if x is not None], default=None)}
        fl = floor.get("dS_rad", {}).get("floor")
        win_rows = [r for r in Prows if T1 - 1e-9 <= r["T"] <= T2 + 1e-9]
        drops = [win_rows[i]["dS_rad"] - win_rows[i + 1]["dS_rad"] for i in range(len(win_rows) - 1)]
        rise = (at(Prows, win_rows[-1]["T"], "dS_rad") - at(Prows, win_rows[0]["T"], "dS_rad")) if win_rows else None
        ratio_rows = [r for r in Prows if T1 + 2 - 1e-9 <= r["T"] <= T2 + 1e-9 and r["S_diag_rad"]]
        ratios = [r["dS_rad"] / r["S_diag_rad"] for r in ratio_rows]
        P1 = {"window": [win_rows[0]["T"], win_rows[-1]["T"]] if win_rows else None, "largest_drop": max(drops) if drops else None, "floor": fl,
              "nondecreasing_above_floor": (max(drops) <= fl) if (drops and fl is not None) else None, "rise": rise,
              "rise_over_floor": (rise / fl) if (rise is not None and fl) else None,
              "dS_over_Sdiag_range": [min(ratios), max(ratios)] if ratios else None,
              "ratio_in_0.5_1.5": (min(ratios) >= 0.5 and max(ratios) <= 1.5) if ratios else None}
        P1["holds_as_written"] = bool(P1["nondecreasing_above_floor"] and P1["rise_over_floor"] is not None and P1["rise_over_floor"] >= 10 and P1["ratio_in_0.5_1.5"])
        s_early, w_early = slope(Prows, "dS_rad", T1 + 2, T2)
        s_late, w_late = slope(Prows, "dS_rad", T2 + 5, min(T2 + 15, 40.0))
        n_early, _ = slope(Prows, "N_ext", T1 + 2, T2)
        n_late, _ = slope(Prows, "N_ext", T2 + 5, min(T2 + 15, 40.0))
        P2 = {"slope_early": s_early, "window_early": w_early, "slope_late": s_late, "window_late": w_late,
              "slope_condition": (s_late <= s_early / 3) if (s_early is not None and s_late is not None) else None,
              "dNext_dT_early": n_early, "dNext_dT_late": n_late,
              "rate_condition": (abs(n_late) * 3 <= abs(n_early)) if (n_early is not None and n_late is not None) else None}
        P2["holds_as_written_without_the_edge"] = bool(P2["slope_condition"] and P2["rate_condition"])
        a3, b3 = max(T2 + 10, 36.0), 40.0
        s3, w3 = slope(Prows, "dS_thr", a3, b3)
        thr_last = at(Prows, w3[1], "dS_thr") if w3 else None
        thr_rise = (at(Prows, win_rows[-1]["T"], "dS_thr") - at(Prows, win_rows[0]["T"], "dS_thr")) if win_rows else None
        P3 = {"rise_over_window": thr_rise, "late_slope": s3, "late_window": w3, "dS_thr_at_end": thr_last,
              "late_condition": (abs(s3) < 0.05 * abs(thr_last)) if (s3 is not None and thr_last) else None}
        P3["holds_as_written"] = bool(thr_rise is not None and thr_rise > 0 and P3["late_condition"])
        after = [r for r in Prows if r["T"] >= T2 + 10 - 1e-9]
        P4 = {"g_h_max_after": max(r["g_h"] for r in after) if after else None, "g_2h_max_after": max(r["g_2h"] for r in after) if after else None,
              "rad_g_range": [min(min(r[k] for k in ("g_rad_4h", "g_rad_8h", "g_rad_16h")) for r in Prows),
                              max(max(r[k] for k in ("g_rad_4h", "g_rad_8h", "g_rad_16h")) for r in Prows)]}
        P4["holds_as_written"] = bool(after and P4["g_h_max_after"] < 0.1 and P4["g_2h_max_after"] < 0.1 and P4["rad_g_range"][0] >= 0.5 and P4["rad_g_range"][1] <= 2)
        tests = {"T_1": T1, "T_2": T2, "floor": floor, "P1": P1, "P2": P2, "P3": P3, "P4": P4}
        js["tests"] = tests
        L += ["## Tests P1 to P4 (sec. 7 as amended by sec. 12.5), applied mechanically as written (no reading)", "",
              "- The window from L: T_1 = %s r_s0, T_2 = T_2(T_H) = %s r_s0." % (f(T1, 5), f(T2, 5)),
              "- The floors (C-pg-8 cadence at 36 r_s0, C-pg-9 h-halving at 20, 24, 30, 36, 40 r_s0; j = 1/2): " + "; ".join(
                  "%s: cadence %s, h-halving %s, floor %s" % (k, f(v["cadence"], 3), f(v["h_halving"], 3), f(v["floor"], 3)) for k, v in floor.items()) + ".",
              "- P1: over [%s, %s] r_s0 the largest drop of dS_rad between outputs %s against the floor %s; the rise %s (%s floors); dS_rad/S_diag,rad over "
              "[T_1 + 2, T_2] from %s to %s; as written: %s." % (
                  f(P1["window"][0], 4) if P1["window"] else "-", f(P1["window"][1], 4) if P1["window"] else "-", f(P1["largest_drop"], 3), f(fl, 3),
                  f(P1["rise"]), f(P1["rise_over_floor"], 3), f(P1["dS_over_Sdiag_range"][0]) if P1["dS_over_Sdiag_range"] else "-",
                  f(P1["dS_over_Sdiag_range"][1]) if P1["dS_over_Sdiag_range"] else "-", "holds" if P1["holds_as_written"] else "does not hold"),
              "- P2: the mean slope of dS_rad over %s is %s per r_s0 and over %s is %s; dN_ext/dT %s and %s; as written (the slope and rate parts; the "
              "spectral edge in T7p3): %s." % (w_early, f(s_early, 3), w_late, f(s_late, 3), f(n_early, 3), f(n_late, 3),
                                               "hold" if P2["holds_as_written_without_the_edge"] else "do not hold"),
              "- P3: dS_thr rises by %s over the window; its slope over %s is %s per r_s0 against 0.05 dS_thr = %s; as written: %s." % (
                  f(thr_rise), w3, f(s3, 3), f(0.05 * thr_last if thr_last else None, 3), "holds" if P3["holds_as_written"] else "does not hold"),
              "- P4: after T_2 + 10 the largest g(h) and g(2h) at the throat are %s and %s (against 0.1); g at the radiation cut ranges over [%s, %s] "
              "(against [0.5, 2]); as written: %s." % (f(P4["g_h_max_after"], 3), f(P4["g_2h_max_after"], 3), f(P4["rad_g_range"][0], 3),
                                                      f(P4["rad_g_range"][1], 3), "holds" if P4["holds_as_written"] else "does not hold"), ""]
        summ.append("Window T_1 = %s, T_2 = %s r_s0; P1 %s, P2 (slope and rate) %s, P3 %s, P4 %s, as written and applied mechanically." % (
            f(T1, 4), f(T2, 4), "holds" if P1["holds_as_written"] else "does not hold", "hold" if P2["holds_as_written_without_the_edge"] else "do not hold",
            "holds" if P3["holds_as_written"] else "does not hold", "holds" if P4["holds_as_written"] else "does not hold"))
        last = Prows[-1]
        summ.append("At T = %s r_s0 (three channels, 2j + 1 weighted): dS_rad = %s, S_diag,rad = %s, N_ext = %s, dS_thr = %s, g(h) at the throat %s, "
                    "N_s = %s." % (f(last["T"], 4), f(last["dS_rad"]), f(last["S_diag_rad"]), f(last["N_ext"]), f(last["dS_thr"]), f(last["g_h"], 3),
                                   f(last["N_s"], 3)))
    # ---- T7p1
    L += ["## T7p1 Per channel (P): dS^(ad) at the four cuts, S'_in at the throat, g(h) at the throat, N_ext, S_diag,rad (both sectors summed)", ""]
    for j in sorted(P):
        rows = []
        for T in sorted(P[j]):
            q = P[j][T]
            secs = q["sectors"].values()

            def sm(fn):
                return sum(fn(s) for s in secs)
            gh = sm(lambda s: s["cuts"]["throat"]["G_state_d1"]) / sm(lambda s: s["cuts"]["throat"]["G_vac0_d1"])
            rows.append([f(T, 5)] + [f(sm(lambda s, c=c: s["cuts"][c]["dSad"])) for c in CUTS if c in q["sectors"]["alpha"]["cuts"]] +
                        [f(sm(lambda s: s["cuts"]["throat"]["S_de_int_state"])), f(gh, 3), f(sm(lambda s: s["spectrum"]["N_ext"])),
                         f(sm(lambda s: s["spectrum"]["S_diag_rad"]))])
        L += ["### j = %g" % j, "", table(["T/r_s0", "dSad throat", "dSad shell", "dSad barrier", "dSad rad", "S'_in throat", "g(h) throat", "N_ext",
                                          "S_diag,rad"], rows), ""]
    # ---- T7p3 spectrum
    L += ["## T7p3 The spectrum per channel (diagnostic; no temperature is read): every 4 r_s0 and at T_1, T_2", ""]
    for j in sorted(P):
        Ts = sorted(P[j])
        sel = [T for T in Ts if abs(T / 4 - round(T / 4)) < 1e-6 or (T1 and abs(T - T1) < 0.51) or (T2 and abs(T - T2) < 0.51)]
        rows = []
        for T in sel:
            q = P[j][T]
            k = Ts.index(T)
            Ne = sum(s["spectrum"]["N_ext"] for s in q["sectors"].values())
            rate = None
            if k > 0:
                Np = sum(s["spectrum"]["N_ext"] for s in P[j][Ts[k - 1]]["sectors"].values())
                rate = (Ne - Np) / (T - Ts[k - 1])
            sa = q["sectors"]["alpha"]["spectrum"]
            means = [b["particles"]["mean"] for b in sa["bins"]]
            edge = max([i for i, m_ in enumerate(means) if m_ is not None and m_ > 1e-3], default=None)
            ro = sa.get("raw_odds")
            rows.append([f(T, 5), f(Ne), f(rate, 3), f(sum(s["spectrum"]["S_diag_rad"] for s in q["sectors"].values())),
                         f(sum(s["spectrum"]["S_diag_near"] for s in q["sectors"].values())), ", ".join(f(m_, 2) for m_ in means),
                         "-" if edge is None else "%d-%d T_H" % (edge, edge + 1), "%s / %s" % (f(ro["slope_per_TH"], 3), f(ro["intercept"], 3)) if ro else "-"])
        L += ["### j = %g (bins: sector alpha's particle means over [0, 8] T_H in 1 T_H bins)" % j, "",
              table(["T/r_s0", "N_ext", "dN_ext/dT", "S_diag,rad", "S_diag,near", "bin means", "upper edge", "raw odds slope/intercept (contains the transmission)"],
                    rows), ""]
    # ---- T7p4 window
    if P_tags:
        Rp = R[P_tags[0]]
        L += ["## T7p4 The window (L)", "", "- " + ", ".join("%s = %s r_s0" % (k, f(v / rs0 if isinstance(v, (int, float)) and k.startswith("T_") else v, 5))
                                                        for k, v in Rp["window"].items()), ""]
    # ---- T7p5 controls
    L += ["## T7p5 Controls", ""]
    cl = []
    allrows = [(t, j, q) for t, Rr in R.items() for j, v in Rr["channels"].items() for q in v]
    pur = [abs(c["S_int_" + st] - c["S_ext_" + st]) for (t, j, q) in allrows for s in q["sectors"].values() for c in s["cuts"].values()
           for st in ("state", "vac0", "vacad") if ("S_ext_" + st) in c]
    orth = [s["controls"]["orthonormality"] for (t, j, q) in allrows for s in q["sectors"].values()]
    cl.append(["C-pg-1 purity", "max |S_int - S_ext| over the check times, cuts, states, runs %s; max ||Phi^dag Phi - 1|| %s" % (f(max(pur) if pur else None, 3),
                                                                                                                              f(max(orth), 3))])
    cl.append(["C-pg-2 T = 0", "not recorded: the runs' first output is the first output time after 0 (run_channel's outputs are interval ends); at T = 0 "
               "the evolved sea is the initial vacuum by construction"])
    for lab, Rr in (("C-pg-3 F0", load(a.f0) if a.f0 and os.path.exists(a.f0) else None), ("C-pg-4 S0 on L", R.get("S0_L"))):
        if Rr is None:
            cl.append([lab, "missing"])
            continue
        rr = [q for v in Rr["channels"].values() for q in v]
        mx = lambda fn: max(abs(fn(s)) for q in rr for s in q["sectors"].values())
        cl.append([lab, "max |dS0| %s, max |dSad| %s, max S'_in %s, max |g - 1| %s, max N_ext %s (%d outputs)" % (
            f(mx(lambda s: max(abs(c["dS0"]) for c in s["cuts"].values())), 3), f(mx(lambda s: max(abs(c["dSad"]) for c in s["cuts"].values())), 3),
            f(mx(lambda s: max(abs(c["S_de_int_state"]) for c in s["cuts"].values())), 3),
            f(mx(lambda s: max(abs(c[k] - 1) for c in s["cuts"].values() for k in c if k.startswith("g_d") and c[k] is not None)), 3),
            f(mx(lambda s: s["spectrum"]["N_ext"]), 3), len(rr))])
    fr = [s["controls"]["frame_identity_rel"] for (t, j, q) in allrows for s in q["sectors"].values()]
    sv = [c["S_de_int_vacad"] for (t, j, q) in allrows for s in q["sectors"].values() for c in s["cuts"].values()]
    uo = [s["controls"]["U_orthogonality"] for (t, j, q) in allrows if t != "F0" for s in q["sectors"].values()]
    cl.append(["C-pg-5 frame", "max ||V h V^T - sigma (x) B_0||/||B_0|| %s; max S'_in of the adiabatic vacuum %s; max ||U^T U - 1|| %s" % (
        f(max(fr), 3), f(max(sv), 3), f(max(uo) if uo else None, 3))])
    if a.paper_h2 and os.path.exists(a.paper_h2):
        spec = importlib.util.spec_from_file_location("paper_le", a.paper_h2)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        import strand7_sea_page_v3 as SP
        x = np.linspace(1e-6, 1 - 1e-6, 1001)
        cl.append(["C-pg-6 h2", "max |h2 - the paper's lattice_entropy.h2| on 1001 points in (0, 1): %s" % f(float(np.max(np.abs(SP.h2(x) - mod.h2(x)))), 3)])
    else:
        cl.append(["C-pg-6 h2", "the paper's module not given"])
    if Prows:
        c7 = []
        for (t, j, q) in allrows:
            if not t.startswith("P_j"):
                continue
            for s in q["sectors"].values():
                c = s["cuts"]["rad"]
                v = [c["G_vac0_d%g" % d] * d * d for d in (2, 4, 8, 16)]
                c7.append(max(v) / min(v))
        cl.append(["C-pg-7 vacuum correlations", "max over P's outputs and sectors of max/min of G^(0)(d) d^2 over d = 2, 4, 8, 16 h at the radiation cut: %s"
                   % f(max(c7), 3)])
        for k, v in floor.items():
            cl.append(["C-pg-8/9 %s" % k, "cadence change at 36 r_s0 %s; h-halving change (20 to 40 r_s0) %s" % (f(v["cadence"], 3), f(v["h_halving"], 3))])
    cc = [q.get("connection_control") for (t, j, q) in allrows if q.get("connection_control")]
    if cc:
        x = cc[0]
        cl.append(["C-pg-11 connection", "ratio with the connection %s (expected about 1/8); without %s (expected about 1/2); norms %s" % (
            f(x.get("ratio_with"), 4), f(x.get("ratio_without"), 4), json.dumps({k: v for k, v in x.items() if k.startswith("tau")}, default=float))])
    if "J" in R and Prows:
        Jc = merged([R["J"]])
        rows = []
        for T in sorted(next(iter(Jc.values()))):
            jr = wsum(Jc, T, lambda s: s["cuts"]["rad"]["dSad"])
            jt = wsum(Jc, T, lambda s: s["cuts"]["throat"]["dSad"])
            pr, pt = at(Prows, T, "dS_rad"), at(Prows, T, "dS_thr")
            if pr is not None:
                rows.append("T %s: rad %s (%s of j <= 5/2), throat %s (%s)" % (f(T, 4), f(jr, 3), f(jr / pr if pr else None, 3), f(jt, 3),
                                                                             f(jt / pt if pt else None, 3)))
        cl.append(["C-pg-12 j = 7/2 (weight 8)", "; ".join(rows[-4:])])
    nz = [s["controls"]["near_zero_modes"] for (t, j, q) in allrows for s in q["sectors"].values()]
    mo = [s["controls"]["smallest_abs_omega"] for (t, j, q) in allrows for s in q["sectors"].values()]
    cl.append(["C-pg-13 near-zero modes", "largest count of |omega| < 1e-6 per sector and output %d; smallest |omega| %s" % (max(nz), f(min(mo), 3))])
    if P_tags and "locked" in R[P_tags[0]]:
        lk = R[P_tags[0]]["locked"]
        traj = np.array(lk["s_over_A0"])
        Tq, sA = traj[:, 0], traj[:, 1]
        A0 = lk["A0"]
        Rarea = sA * A0 * (1 + 1 / sA) ** 2
        d_ = Rarea - 4 * A0
        rhv = lk["r_h"]
        late = []
        for (t1, t2) in ((14, 18), (18, 22), (22, 26)):
            m = (Tq >= t1) & (Tq <= t2) & (d_ > 0)
            if np.sum(m) >= 3:
                late.append("[%g, %g]: %s" % (t1, t2, f(-np.polyfit(Tq[m] * rs0, np.log(d_[m]), 1)[0] * rhv, 5)))
        fzs = ""
        if a.frozen and os.path.exists(a.frozen):
            FZ = load(a.frozen)
            recs = FZ["run"]["records"]
            Tz = np.array([q["T"] for q in recs])
            dz = np.array([q["R_sh_minus_Rtrap"] for q in recs])
            A0z = FZ["run"]["A0"]
            rhz = 4 * A0z
            mz = (Tz / rhz >= Tz[-1] / rhz - 10) & (dz > 0)
            fzs = "; frozen_schw E8's late lambda r_h %s over its last 10 r_h" % f(-np.polyfit(Tz[mz], np.log(dz[mz]), 1)[0] * rhz, 5)
        cl.append(["C-pg-14 L's trajectory", "energy control max |N W/eps - 1| = %s; lambda r_h of R - 4A_0 by window (r_s0) %s%s" % (
            f(lk["energy_control"]["max_rel_dev_NW"], 3), "; ".join(late), fzs)])
    L += [table(["control", "value"], cl), ""]
    # ---- T7p6 checks
    if Prows:
        p05 = sums({0.5: P[0.5]}, h) if 0.5 in P else []
        rows = []
        for t in ("K1", "K2", "H", "R"):
            if t not in R:
                continue
            cs = sums(merged([R[t]]), R[t]["meta"]["h"])
            for Tq in (20.0, 24.0, 30.0, 36.0, 40.0):
                if at(cs, Tq, "dS_rad") is None or at(p05, Tq, "dS_rad") is None:
                    continue
                rows.append([t, f(Tq, 4)] + ["%s / %s" % (f(at(cs, Tq, k)), f(at(p05, Tq, k))) for k in ("dS_rad", "dS_thr", "S_diag_rad", "N_ext", "g_h")])
            if t == "R":
                w = R[t]["window"]
                rows.append(["R window", "-", "T_1 %s, T_2 %s r_s0 (P: %s, %s)" % (f(w.get("T_1", 0) / R[t]["meta"]["r_s0"], 4), f(w.get("T_2_1TH", 0) / R[t]["meta"]["r_s0"], 4),
                                                                                  f(T1, 4), f(T2, 4)), "-", "-", "-", "-"])
        L += ["## T7p6 The checks against P at j = 1/2 (check / P)", "", table(["run", "T/r_s0", "dS_rad", "dS_thr", "S_diag,rad", "N_ext", "g(h)"], rows), ""]
    # ---- T7p7 C
    if C_tags:
        Cc = merged([R[t] for t in C_tags])
        Crows = sums(Cc, h)
        keysC = ["dS_rad", "dS0_rad", "S_diag_rad", "N_ext", "dS_thr", "dS_shell", "g_h", "g_2h", "g_rad_8h", "N_s", "areal_shell_cut"]
        L += ["## T7p7 The comparison C (the rows' own collapse; channels %s)" % sorted(Cc), "",
              table(["T/r_s0"] + keysC, [[f(r["T"], 5)] + [f(r[k]) for k in keysC] for r in Crows]), ""]
        js["C_sums"] = Crows
        if Crows:
            lc = Crows[-1]
            summ.append("C (channels %s) at T = %s r_s0: dS_rad = %s, dS_thr = %s, N_ext = %s." % (sorted(Cc), f(lc["T"], 4), f(lc["dS_rad"]), f(lc["dS_thr"]),
                                                                                                 f(lc["N_ext"])))
    # ---- T7p9 L
    if P_tags and "locked" in R[P_tags[0]]:
        lk = R[P_tags[0]]["locked"]
        L += ["## T7p9 L's trajectory (every 0.5 r_s0 in the record; shown every 2 r_s0)", "", "- A_0 = %s, r_h = 4 A_0 = %s, R_0 = %s; energy control %s." % (
            f(lk["A0"], 8), f(lk["r_h"], 8), f(lk["R0"], 6), json.dumps(lk["energy_control"], default=float)), "",
              table(["T/r_s0", "s/A_0", "N_s"], [[f(x[0], 4), f(x[1], 8), f(x[2], 4)] for x in lk["s_over_A0"] if abs(x[0] / 2 - round(x[0] / 2)) < 1e-6]), ""]
    L = L[:2] + ["## Emitted summary", ""] + ["- " + s for s in summ] + [""] + L[2:]
    with open(a.md, "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(a.out_json, "w") as fh:
        json.dump(js, fh, indent=1, default=float)
    print("\n".join("- " + s for s in summ))


if __name__ == "__main__":
    main()
