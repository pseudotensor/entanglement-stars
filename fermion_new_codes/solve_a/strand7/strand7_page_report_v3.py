#!/usr/bin/env python
"""Item 7's report, version 2 (SPEC_strand7_page.md secs. 6, 7, 9, 12 and 13; builder analyst): reads the sea runs' records (s7p_<tag>.json
in --dir) and writes the emitted summary, the tables and the tests as markdown, plus a JSON.  Version 2 implements sec. 13.8:
  (1) c_band from F0's record (--f0 required: its modes file next to it), the markers T_1, T_2, T_2p, T_lw on L from the P record's locked
      trajectory and on C from the S7a JSON's N(r_sh) with the stored fields' psi(r_sh) (--c-fields; --c-s7a-json, default C's meta), by
      strand7_page_markers_v1.py;
  (2) the tests' window [T_1, T_w], T_w = min(T_2, T_2p) (the first output >= T_1 to the last output <= T_w; "not testable" if T_w - T_1 < 4 r_s0),
      the late interval [max(T_2, T_2p) + 5, 40] r_s0;
  (3) the floors per quantity and output (sec. 13.4): f_q(T) = the largest available of |q_K1 - q_P|, |q_K2 - q_P| (T <= 36), |q_H - q_B40|,
      |q_B40 - q_P|, all at j = 1/2 (the cuts at the same radii in length units; the records' distance keys are in length units); on P's
      channel sum F_q(T) = f_q(T) |q_P,sum(T)|/|q_P,1/2(T)| (f_q where q_P,1/2(T) = 0); for the ratios g, F_g = f_g; on a difference
      F_q(T_a) + F_q(T_b); on a mean slope (F_q(T_a) + F_q(T_b))/(T_b - T_a); every floor printed beside its test;
  (4) the tests P1 to P4 of sec. 13.5 (mechanical; zero denominators: g where the weighted G_vac0 > 1e-30, the ratio dS_rad/S_diag_rad where
      S_diag_rad > 10 F_S; the excluded counts printed), R1 descriptive, P2a to P2d;
  (5) T7p3: a bin's mean at count >= 1; the upper edge over bins with count >= 2 and mean > 1e-3; the raw-odds line only over bins with count
      >= 2 and mean in [1e-6, 0.5] and only with at least 3 such bins (else "-"), labelled (it contains the transmission); the record's own
      raw-odds line (the page code's [1, 6] T_H rule) is not used;
  (6) T7p6 with B40: C-pg-9 (H against B40: the spacing alone), C-pg-10 (R against B40: r_h/h alone, with the markers' shifts against the
      predicted 2 r_h ln 2), C-pg-15 (B40 against P: the box alone), K1 and K2 against P;
  (7) T7p7: C matched at j = 1/2 against P's j = 1/2, C's channel sum only if C has P's channels, C's markers, the sensitivities and T_valid
      from the part 2 report (--part2-json, optional; else "pending");
  (8) the labels of sec. 13.1 in the table heads and the summary (the record keys unchanged: "throat" is the reference cut r = A_0);
  (9) no "effective transmission" ratio.
Tables kept from version 1: T7p1 (per channel), T7p5 (the controls), T7p9 (L's trajectory).  Arguments: --dir, --f0, --frozen, --paper-h2,
--c-fields, --c-s7a-json, --part2-json, --md, --out-json.
Version 3 (SPEC_strand7_page.md secs. 14.5 to 14.7): the records are selected per run: the repaired s7p_<tag>_r2.json where it exists, else the
run's own record (version 5's runs need no repair); the validation runs (V5_F0, V5_S0) and the smoke runs are not runs of the item; P at
j = 5/2 is the version 5 rerun P5_j2.5 when C-pg-17 holds (the largest |v5 - r2| over the outputs, cuts and sectors of S_int_state, dS^(ad),
dS^(0), MI_state, g and N_ext, S_diag,rad below 1e-6; expected below 1e-8), else the r2 record with both reported and the tests stopped
(sec. 14.7); T7p5 adds C-pg-16 (the repair pass's summaries, repair_*.json) and C-pg-17, the version 5 controls (the largest
vac_projector_diff and the unresolved pairs per run); T7p10 lists the excluded pairs (sec. 14.3) per channel and sector at every 4 r_s0."""
import argparse
import glob
import importlib.util
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand7_page_markers_v1 as MK  # noqa: E402

CUTS = ("throat", "shell", "barrier", "rad")
QS = ("dS_rad", "dS_thr", "S_diag_rad", "N_ext", "g_h", "g_2h", "g_4h", "g_rad_4h", "g_rad_8h", "g_rad_16h")
GQ = ("g_h", "g_2h", "g_4h", "g_rad_4h", "g_rad_8h", "g_rad_16h")
LABELS = [
    "dS_rad: the excess entanglement entropy (over the instantaneous vacuum's) of the radiation region r >= 8A_0 with its complement (an entropy "
    "difference, not the entropy)",
    "S_diag,rad: the diagonal entropy of the exterior eigenmodes (<r>_a > 8A_0) at their occupations: a different object, shown for comparison",
    "dS_ref (record key dS_thr): the excess entanglement entropy across the reference cut r = A_0 (the nominal throat of the exterior form; at "
    "every finite T inside the uniform interior, lapse N_s(T), areal radius A_0 (1 + A_0/s)^2 < 4A_0; the exterior form's lapse zero r = A_0, "
    "areal 4A_0 = 2 G E_bdy, is the limiting reference surface the shell approaches as s - A_0 ~ e^{-T/r_h}; never the trapping surface)",
    "g(d) at the reference cut: the two-point correlation ratio across the reference cut at d = h, 2h, 4h, relative to the initial vacuum; "
    "g_rad(d): the same at the radiation cut",
    "the shell cut r = s(T) + 2h (just outside the material); the barrier cut r = 4A_0; on C the reference cut r = A(T) (the nominal throat of "
    "the exterior form with the instantaneous charge), the shell cut r_p99(T) + 2h, the radiation cut 8A(T)",
    "the primary curve: the probe's excess-entanglement curve along the prescribed locked collapse L (a prescribed probe history: the locked "
    "exterior form with the fixed charge A_0 outside a shell moving on the test characteristic of that exterior, the matched uniform fields "
    "inside; the lock and the history supplied, the probe not acting back); C: the rows' own unlocked collapse (self E16), an approximate "
    "numerical history whose material is sub-cell after about 20 r_s0"]


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
    if num is None or den is None:
        return None
    return (num / den) if den > 1e-30 else float("nan")


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
            v = r[key]
            return None if (v is None or (isinstance(v, float) and not math.isfinite(v))) else v
    return None


def interp_at(rows, T, key):
    """linear interpolation of a quantity between the neighbouring outputs (the markers are not output times; labelled in the output)."""
    pts = [(r["T"], r[key]) for r in rows if r[key] is not None and math.isfinite(r[key])]
    if not pts or T is None or T < pts[0][0] or T > pts[-1][0]:
        return None
    xs, ys = zip(*pts)
    return float(np.interp(T, xs, ys))


def edge_of(q, cmin=2, mmin=1e-3):
    """the spectral upper edge at one output: the largest bin upper limit (T_H) with count >= cmin and mean > mmin over channels' sectors."""
    best = None
    for s in q["sectors"].values():
        for b in s["spectrum"]["bins"]:
            p = b["particles"]
            if p["modes"] >= cmin and p["mean"] is not None and p["mean"] > mmin:
                best = max(best or 0.0, b["omega_hi_TH"])
    return best


def odds_fit(bins):
    xs, ys = [], []
    for b in bins:
        p = b["particles"]
        if p["modes"] >= 2 and p["mean"] is not None and 1e-6 <= p["mean"] <= 0.5:
            xs.append(0.5 * (b["omega_lo_TH"] + b["omega_hi_TH"]))
            ys.append(math.log(1.0 / p["mean"] - 1.0))
    if len(xs) < 3:
        return None
    sl, ic = np.polyfit(xs, ys, 1)
    return {"slope_per_TH": float(sl), "intercept": float(ic), "bins_used": len(xs)}


SKIP = ("V5_F0", "V5_S0")


def c_pg_17(Rv5, Rr2):
    """the largest |v5 - r2| over the common outputs, cuts and sectors (sec. 14.6 (3))."""
    out = {}
    a5 = {round(q["T_over_rs0"], 6): q for v in Rv5["channels"].values() for q in v}
    a2 = {round(q["T_over_rs0"], 6): q for v in Rr2["channels"].values() for q in v}
    for T in sorted(set(a5) & set(a2)):
        for name in a5[T]["sectors"]:
            s5, s2 = a5[T]["sectors"][name], a2[T]["sectors"][name]
            for cname in s5["cuts"]:
                c5, c2 = s5["cuts"][cname], s2["cuts"][cname]
                for k in ("S_int_state", "dSad", "dS0", "MI_state"):
                    out[k] = max(out.get(k, 0.0), abs(c5[k] - c2[k]))
                for k in c5:
                    if k.startswith("g_d") and c5[k] is not None and c2.get(k) is not None:
                        out["g"] = max(out.get("g", 0.0), abs(c5[k] - c2[k]))
            for k in ("N_ext", "S_diag_rad"):
                out[k] = max(out.get(k, 0.0), abs(s5["spectrum"][k] - s2["spectrum"][k]))
    out["outputs"] = len(set(a5) & set(a2))
    return out


def select_records(d):
    """per run: the repaired _r2 record where it exists, else the run's own (sec. 14.5); P at j = 5/2 from the v5 rerun when C-pg-17 holds."""
    files = sorted(glob.glob(os.path.join(d, "s7p_*.json")))
    base = {}
    for p in files:
        t = os.path.basename(p)[4:-5]
        if t.endswith("_r2"):
            base.setdefault(t[:-3], {})["r2"] = p
        else:
            base.setdefault(t, {})["own"] = p
    R, src = {}, {}
    for t, v in base.items():
        if t in SKIP or t.startswith("smoke"):
            continue
        p = v.get("r2") or v.get("own")
        if p is None:
            continue
        R[t] = load(p)
        src[t] = os.path.basename(p)
    if "P5_j2.5" in R and "P_j2.5" in R:
        c17 = c_pg_17(R["P5_j2.5"], R["P_j2.5"])
        c17["holds"] = all(v < 1e-6 for k, v in c17.items() if k != "outputs")
        if c17["holds"]:
            R["P_j2.5"] = R.pop("P5_j2.5")
            src["P_j2.5"] = src.pop("P5_j2.5") + " (the v5 rerun; C-pg-17 holds)"
        else:
            src["P_j2.5"] += " (C-pg-17 does not hold: the v5 rerun reported beside it as P5_j2.5; the tests stop, sec. 14.7)"
        R["_C_pg_17"] = c17
    elif "P5_j2.5" in R:
        R["P_j2.5"] = R.pop("P5_j2.5")
        src["P_j2.5"] = src.pop("P5_j2.5")
    return R, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--f0", required=True)
    ap.add_argument("--frozen", default=None)
    ap.add_argument("--paper-h2", default=None)
    ap.add_argument("--c-fields", default=None, help="the stored fields of C's background (psi(r_sh) for T_2p and T_lw on C)")
    ap.add_argument("--c-s7a-json", default=None, help="C's S7a JSON (default: C's meta s7a_json)")
    ap.add_argument("--part2-json", default=None, help="the part 2 report's JSON (the dt ladder's spreads, the E-ladder, T_valid)")
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    R, src = select_records(a.dir)
    P_tags = sorted(t for t in R if t.startswith("P_j"))
    C_tags = sorted(t for t in R if t.startswith("C_j"))
    if not P_tags:
        raise SystemExit("no P records in %s" % a.dir)
    rs0 = R[P_tags[0]]["meta"]["r_s0"]
    h = float(R[P_tags[0]]["meta"]["h"])
    c_band, cb_det = MK.c_band_from_f0(a.f0, 1.0)
    mkL = MK.markers_L(R[P_tags[0]], c_band)
    T1, T2, T2p = mkL["T_1"], mkL["T_2"], mkL["T_2p"]
    Tw = min(x for x in (T2, T2p) if x is not None) if (T2 is not None or T2p is not None) else None
    late_a = (max(x for x in (T2, T2p) if x is not None) + 5.0) if Tw is not None else None
    P = merged([R[t] for t in P_tags])
    chans = sorted(P)
    chlab = ", ".join("%g" % j for j in chans)
    Prows = sums(P, h)
    p05 = sums({0.5: P[0.5]}, h) if 0.5 in P else []
    c17 = R.pop("_C_pg_17", None)
    js = {"runs": sorted(R), "record_sources": src, "C_pg_17": c17, "c_band": c_band, "c_band_detail": cb_det, "markers_L": mkL, "T_w": Tw, "late_start": late_a, "channels_P": chans}
    summ = []
    L = ["# Item 7's tables (strand7_page_report_v3.py over %s)" % a.dir, "", "## Labels (SPEC_strand7_page.md sec. 13.1)", ""] + ["- " + x for x in LABELS] + [
        "- the reference vacuum: the exact sea projector (1/2)(1 - sgn h) of each sector (sec. 14.2); the wall's edge mode, one orbital per sector with its "
        "two components at the centre and at the wall, contributes ln 2 to every cut's entropy in the state and in both references and cancels in the excess",
        "", "## The records (sec. 14.5: the repaired _r2 records where they exist, version 5's runs as recorded)", ""] + [
        "- %s: %s" % (k, v) for k, v in sorted(src.items())] + [""]
    # ---- the markers
    cbx = ""
    cj = os.path.join(a.dir, "cband_v1.json")
    if os.path.exists(cj):
        CB = load(cj)
        cbx = "; computed on the flat background by strand7_cband_v1.py: " + ", ".join(
            "j = %s: %s" % (j, f(max(v["max_abs_omega_times_h"] for v in d.values()), 6)) for j, d in CB["computed"].items())
        js["cband_computed"] = CB["computed"]
    L += ["## T7p4 The window markers (sec. 13.3; declared markers, none an exhaustion time)", "",
          "- c_band = %s (the largest |omega| of F0's j = 1/2 sector spectrum times h, edge modes below 1e-6 excluded; a declared coefficient)%s." % (
              f(c_band, 6), cbx),
          "- On L (%s; T_H = %s, h = %s): T_1 = %s, T_2 = %s, T_2p = %s, T_lw = %s r_s0 (crossings interpolated log-linearly between the trajectory's "
          "0.5 r_s0 samples); the record's own window (0.1 r_s0 grid): T_1 = %s, T_2 = %s." % (
              mkL["source"], f(mkL["T_H"], 6), f(h, 3), f(T1, 5), f(T2, 5), f(T2p, 5), f(mkL["T_lw"], 5), f(mkL["record_window"]["T_1"], 4),
              f(mkL["record_window"]["T_2"], 4)),
          "- The tests' window: [T_1, T_w], T_w = min(T_2, T_2p) = %s r_s0 (T_w - T_1 = %s r_s0%s); the late interval [%s, 40] r_s0." % (
              f(Tw, 5), f((Tw - T1) if (Tw is not None and T1 is not None) else None, 4),
              "; not testable" if (Tw is None or T1 is None or Tw - T1 < 4) else "", f(late_a, 5)), ""]
    testable = Tw is not None and T1 is not None and Tw - T1 >= 4 and not (c17 is not None and not c17.get("holds"))
    # ---- T7p2
    keys = ["dS_rad", "dS0_rad", "S_diag_rad", "N_ext", "dS_thr", "dS_shell", "dS_bar", "Sde_thr", "MI_thr", "g_h", "g_2h", "g_4h", "g_rad_4h", "g_rad_8h",
            "g_rad_16h", "N_s", "areal_shell_cut"]
    heads = ["dS_rad", "dS0_rad", "S_diag,rad", "N_ext", "dS_ref", "dS_shell", "dS_barrier", "S'_in ref", "I_w ref", "g(h) ref", "g(2h) ref", "g(4h) ref",
             "g_rad(4h)", "g_rad(8h)", "g_rad(16h)", "N_s", "areal radius of the shell cut"]
    L += ["## T7p2 The primary P on L: the 2j + 1 weighted sums per output (channels %s)" % chlab, "",
          table(["T/r_s0"] + heads, [[f(r["T"], 5)] + [f(r[k]) for k in keys] for r in Prows]), ""]
    js["P_sums"] = Prows
    # ---- the floors (sec. 13.4)
    chk = {}
    for t in ("K1", "K2", "H", "B40", "R"):
        if t in R:
            chk[t] = sums(merged([R[t]]), float(R[t]["meta"]["h"]))
    Tp = [r["T"] for r in Prows]
    floors = {}
    for q in QS:
        fq = {}
        for T in Tp:
            c = []
            for t in ("K1", "K2"):
                if t in chk and T <= 36 + 1e-9:
                    v1, v0 = at(chk[t], T, q), at(p05, T, q)
                    if v1 is not None and v0 is not None:
                        c.append(("|%s - P|" % t, abs(v1 - v0)))
            if "H" in chk and "B40" in chk:
                v1, v0 = at(chk["H"], T, q), at(chk["B40"], T, q)
                if v1 is not None and v0 is not None:
                    c.append(("|H - B40|", abs(v1 - v0)))
            if "B40" in chk:
                v1, v0 = at(chk["B40"], T, q), at(p05, T, q)
                if v1 is not None and v0 is not None:
                    c.append(("|B40 - P|", abs(v1 - v0)))
            if not c:
                fq[T] = None
                continue
            fl = max(x[1] for x in c)
            src = max(c, key=lambda x: x[1])[0]
            if q in GQ:
                F = fl
            else:
                v3, v1 = at(Prows, T, q), at(p05, T, q)
                F = fl * abs(v3) / abs(v1) if (v3 is not None and v1) else fl
            fq[T] = {"f": fl, "F": F, "from": src, "available": [x[0] for x in c]}
        floors[q] = fq
    js["floors"] = {q: {str(T): v for T, v in fq.items()} for q, fq in floors.items()}

    def Fq(q, T):
        v = floors.get(q, {}).get(T)
        return None if v is None else v["F"]

    def Fdiff(q, Ta, Tb):
        a_, b_ = Fq(q, Ta), Fq(q, Tb)
        return None if (a_ is None or b_ is None) else a_ + b_

    def mean_slope(rows, q, lo, hi):
        sel = [r for r in rows if lo - 1e-9 <= r["T"] <= hi + 1e-9 and r[q] is not None and math.isfinite(r[q])]
        if len(sel) < 2:
            return None, None, None
        Ta, Tb = sel[0]["T"], sel[-1]["T"]
        s = (sel[-1][q] - sel[0][q]) / (Tb - Ta)
        fl = Fdiff(q, Ta, Tb)
        return s, (Ta, Tb), (fl / (Tb - Ta) if fl is not None else None)
    tests = {"testable": testable}
    if testable:
        win = [r for r in Prows if T1 - 1e-9 <= r["T"] <= Tw + 1e-9]
        Ta, Tb = win[0]["T"], win[-1]["T"]
        # P1
        steps = []
        for r0, r1 in zip(win[:-1], win[1:]):
            dec = r0["dS_rad"] - r1["dS_rad"]
            fl = Fdiff("dS_rad", r0["T"], r1["T"])
            steps.append({"Ta": r0["T"], "Tb": r1["T"], "decrease": dec, "floor": fl, "ok": (dec <= fl) if fl is not None else None})
        rise = win[-1]["dS_rad"] - win[0]["dS_rad"]
        frise = Fdiff("dS_rad", Ta, Tb)
        p1a = None if any(s["ok"] is None for s in steps) else all(s["ok"] for s in steps)
        p1b = (rise >= 10 * frise) if frise is not None else None
        worst = max(steps, key=lambda s: s["decrease"]) if steps else None
        P1 = {"window_outputs": [Ta, Tb], "outputs": len(win), "steps": steps, "a": p1a, "rise": rise, "rise_floor": frise,
              "rise_in_floors": (rise / frise) if frise else None, "b": p1b,
              "holds": None if (p1a is None or p1b is None) else (p1a and p1b)}
        # R1
        rr, excl = [], 0
        for r in Prows:
            if T1 + 2 - 1e-9 <= r["T"] <= Tw + 1e-9:
                FS = Fq("S_diag_rad", r["T"])
                if FS is not None and r["S_diag_rad"] > 10 * FS:
                    rr.append(r["dS_rad"] / r["S_diag_rad"])
                else:
                    excl += 1
        R1 = {"range": [min(rr), max(rr)] if rr else None, "outputs_used": len(rr), "excluded": excl}
        # P2
        s_e, w_e, F_e = mean_slope(Prows, "dS_rad", T1 + 2, Tw)
        s_l, w_l, F_l = mean_slope(Prows, "dS_rad", late_a, 40.0)
        P2a = (s_l <= s_e / 3) if (s_e is not None and s_l is not None) else None
        p2b_testable = (s_e is not None and F_e is not None and s_e >= 10 * F_e)
        if s_l is not None and F_l is not None and s_l < -2 * F_l:
            P2b = "decline"
        elif p2b_testable and s_l is not None and F_l is not None:
            P2b = abs(s_l) <= max(0.1 * s_e, 2 * F_l)
        else:
            P2b = "not testable"
        n_e, nw_e, nF_e = mean_slope(Prows, "N_ext", T1 + 2, Tw)
        n_l, nw_l, nF_l = mean_slope(Prows, "N_ext", late_a, 40.0)
        P2c = (abs(n_l) <= abs(n_e) / 3) if (n_e is not None and n_l is not None) else None
        e_win = [edge_of(P[j][r["T"]]) for r in win for j in chans if r["T"] in P[j]]
        e_aft = [edge_of(P[j][r["T"]]) for r in Prows if r["T"] > T2 + 1e-9 for j in chans if r["T"] in P[j]]
        e_win = [x for x in e_win if x is not None]
        e_aft = [x for x in e_aft if x is not None]
        P2d = (max(e_aft) < max(e_win)) if (e_win and e_aft) else None
        P2 = {"s_early": s_e, "w_early": w_e, "F_early": F_e, "s_late": s_l, "w_late": w_l, "F_late": F_l, "P2a": P2a, "P2b": P2b,
              "P2b_testable": p2b_testable, "N_rate_early": n_e, "N_rate_late": n_l, "P2c": P2c, "edge_max_window": max(e_win) if e_win else None,
              "edge_max_after_T2": max(e_aft) if e_aft else None, "P2d": P2d}
        # P3
        thr_rise = win[-1]["dS_thr"] - win[0]["dS_thr"]
        fthr = Fdiff("dS_thr", Ta, Tb)
        s3, w3, F3 = mean_slope(Prows, "dS_thr", max(T2 + 10, 36.0), 40.0)
        d40 = at(Prows, 40.0, "dS_thr")
        P3a = (thr_rise >= 10 * fthr) if fthr is not None else None
        P3b = (abs(s3) < 0.05 * abs(d40) + 2 * F3) if (s3 is not None and d40 is not None and F3 is not None) else None
        P3 = {"rise": thr_rise, "rise_floor": fthr, "a": P3a, "late_slope": s3, "late_window": w3, "late_floor": F3, "dS_ref_40": d40, "b": P3b,
              "holds": None if (P3a is None or P3b is None) else (P3a and P3b)}
        # P4
        aft = [r for r in Prows if r["T"] >= T2 + 10 - 1e-9]
        p4 = {}
        und = 0
        for q in ("g_h", "g_2h"):
            vals = [(r[q], Fq(q, r["T"])) for r in aft]
            ok = [(v, F) for v, F in vals if v is not None and math.isfinite(v) and F is not None]
            und += len(vals) - len(ok)
            p4[q] = {"max": max(v for v, F in ok) if ok else None, "pass": all(v < 0.1 + F for v, F in ok) if ok else None}
        rad = []
        for r in Prows:
            for q in ("g_rad_4h", "g_rad_8h", "g_rad_16h"):
                v, F = r[q], Fq(q, r["T"])
                if v is None or not math.isfinite(v) or F is None:
                    und += 1
                    continue
                rad.append((v, F))
        p4["rad"] = {"min": min(v for v, F in rad) if rad else None, "max": max(v for v, F in rad) if rad else None,
                     "pass": all(0.5 - F <= v <= 2 + F for v, F in rad) if rad else None}
        P4 = {"after": T2 + 10, "outputs_after": len(aft), "parts": p4, "undefined_or_no_floor": und,
              "holds": None if (not aft or any(p4[k]["pass"] is None for k in ("g_h", "g_2h", "rad"))) else all(p4[k]["pass"] for k in ("g_h", "g_2h", "rad"))}
        tests.update({"P1": P1, "R1": R1, "P2": P2, "P3": P3, "P4": P4})

        def yn(x):
            return "not decidable (a floor or an entry missing)" if x is None else (x if isinstance(x, str) else ("holds" if x else "does not hold"))
        L += ["## Tests P1 to P4 (sec. 13.5), applied mechanically (no reading); every floor printed beside its test", "",
              "- The window's outputs: %s to %s r_s0 (%d outputs); the floors per output in T7p8." % (f(Ta, 5), f(Tb, 5), len(win)),
              "- P1 (a): every decrease of dS_rad between consecutive outputs at most the summed floors: %s (the largest decrease %s between %s and %s "
              "against %s). P1 (b): the rise %s against 10 x (F(first) + F(last)) = %s (%s floors): %s. P1: %s." % (
                  yn(p1a), f(worst["decrease"], 3) if worst else "-", f(worst["Ta"], 4) if worst else "-", f(worst["Tb"], 4) if worst else "-",
                  f(worst["floor"], 3) if worst else "-", f(rise), f(10 * frise if frise is not None else None, 3), f(P1["rise_in_floors"], 3), yn(p1b),
                  yn(P1["holds"])),
              "- R1 (descriptive, not a condition): dS_rad/S_diag,rad over [T_1 + 2, T_w] from %s to %s (%d outputs; %d excluded with S_diag,rad <= 10 F_S)." % (
                  f(R1["range"][0]) if R1["range"] else "-", f(R1["range"][1]) if R1["range"] else "-", R1["outputs_used"], R1["excluded"]),
              "- P2: s_early = %s per r_s0 over %s (floor %s); s_late = %s over %s (floor %s). P2a (s_late <= s_early/3): %s. P2b (|s_late| <= "
              "max(0.1 s_early, 2 F_late), testable if s_early >= 10 F_early): %s. P2c (the mean rate of N_ext, an occupation-change rate in a moving "
              "eigenbasis, not a flux: %s early, %s late; the late at most a third): %s. P2d (the spectral upper edge after T_2, %s T_H, below its "
              "window maximum, %s T_H): %s." % (
                  f(s_e, 3), w_e, f(F_e, 3), f(s_l, 3), w_l, f(F_l, 3), yn(P2a), yn(P2b), f(n_e, 3), f(n_l, 3), yn(P2c), f(P2["edge_max_after_T2"], 3),
                  f(P2["edge_max_window"], 3), yn(P2d)),
              "- P3 (the reference cut): dS_ref rises over the window by %s against 10 floors %s: %s; over %s its mean slope %s against 0.05 |dS_ref(40)| "
              "+ 2 F = %s: %s. P3: %s." % (
                  f(thr_rise), f(10 * fthr if fthr is not None else None, 3), yn(P3a), w3, f(s3, 3),
                  f((0.05 * abs(d40) + 2 * F3) if (d40 is not None and F3 is not None) else None, 3), yn(P3b), yn(P3["holds"])),
              "- P4 (the correlations): over the %d outputs from T_2 + 10 = %s r_s0 the largest g(h) and g(2h) at the reference cut %s and %s against "
              "0.1 + F_g (%s, %s); g_rad(4h, 8h, 16h) over the run within [%s, %s] against [0.5 - F_g, 2 + F_g] (%s); entries undefined or without a "
              "floor: %d. P4: %s." % (
                  len(aft), f(T2 + 10, 4), f(p4["g_h"]["max"], 3), f(p4["g_2h"]["max"], 3), yn(p4["g_h"]["pass"]), yn(p4["g_2h"]["pass"]),
                  f(p4["rad"]["min"], 3), f(p4["rad"]["max"], 3), yn(p4["rad"]["pass"]), und, yn(P4["holds"])), ""]
        summ.append("Window [T_1, T_w] = [%s, %s] r_s0 (T_2 = %s, T_2p = %s); P1 %s, P2a %s, P2b %s, P2c %s, P2d %s, P3 %s, P4 %s, applied mechanically." % (
            f(T1, 4), f(Tw, 4), f(T2, 4), f(T2p, 4), yn(P1["holds"]), yn(P2a), yn(P2b), yn(P2c), yn(P2d), yn(P3["holds"]), yn(P4["holds"])))
    else:
        L += ["## Tests P1 to P4", "", "- The window is not testable (T_w - T_1 < 4 r_s0 or a marker missing).", ""]
        summ.append("The window is not testable (T_1 = %s, T_w = %s r_s0)." % (f(T1, 4), f(Tw, 4)))
    js["tests"] = tests
    last = Prows[-1]
    summ.append("At T = %s r_s0 (channels %s, 2j + 1 weighted): dS_rad = %s, S_diag,rad = %s, N_ext = %s, dS_ref = %s, g(h) at the reference cut %s, "
                "N_s = %s." % (f(last["T"], 4), chlab, f(last["dS_rad"]), f(last["S_diag_rad"]), f(last["N_ext"]), f(last["dS_thr"]), f(last["g_h"], 3),
                               f(last["N_s"], 3)))
    # ---- T7p8 the floors per output
    rows = []
    for T in Tp:
        rows.append([f(T, 5)] + ["%s (%s)" % (f(floors[q][T]["F"], 2), floors[q][T]["from"]) if floors[q].get(T) else "-" for q in ("dS_rad", "dS_thr", "S_diag_rad", "N_ext", "g_h", "g_rad_8h")])
    L += ["## T7p8 The floors per output on P's channel sum (sec. 13.4; the largest contribution named)", "",
          table(["T/r_s0", "F dS_rad", "F dS_ref", "F S_diag,rad", "F N_ext", "F g(h) ref", "F g_rad(8h)"], rows), ""]
    # ---- T7p1
    L += ["## T7p1 Per channel (P): dS^(ad) at the four cuts (the reference cut first), S'_in at the reference cut, g(h) there, N_ext, S_diag,rad", ""]
    for j in chans:
        rows = []
        for T in sorted(P[j]):
            q = P[j][T]
            secs = list(q["sectors"].values())

            def sm(fn):
                return sum(fn(s) for s in secs)
            den = sm(lambda s: s["cuts"]["throat"]["G_vac0_d1"])
            gh = sm(lambda s: s["cuts"]["throat"]["G_state_d1"]) / den if den > 1e-30 else float("nan")
            rows.append([f(T, 5)] + [f(sm(lambda s, c=c: s["cuts"][c]["dSad"])) for c in CUTS if c in q["sectors"]["alpha"]["cuts"]] +
                        [f(sm(lambda s: s["cuts"]["throat"]["S_de_int_state"])), f(gh, 3), f(sm(lambda s: s["spectrum"]["N_ext"])),
                         f(sm(lambda s: s["spectrum"]["S_diag_rad"]))])
        L += ["### j = %g" % j, "", table(["T/r_s0", "dSad ref", "dSad shell", "dSad barrier", "dSad rad", "S'_in ref", "g(h) ref", "N_ext", "S_diag,rad"], rows), ""]
    # ---- T7p3 spectrum
    L += ["## T7p3 The spectrum per channel (diagnostic; no temperature is read): every 4 r_s0 and the outputs nearest the markers", "",
          "- A bin's mean exists at count >= 1; the upper edge uses bins with count >= 2 and mean > 1e-3; the raw-odds line is fitted only over bins with "
          "count >= 2 and mean in [1e-6, 0.5] and only with at least 3 such bins (it contains the transmission; not a temperature).", ""]
    for j in chans:
        Ts = sorted(P[j])
        mk = [x for x in (T1, T2p, T2) if x is not None]
        sel = [T for T in Ts if abs(T / 4 - round(T / 4)) < 1e-6 or any(abs(T - m) < 0.51 for m in mk)]
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
            means = [b["particles"]["mean"] if b["particles"]["modes"] >= 1 else None for b in sa["bins"]]
            counts = [b["particles"]["modes"] for b in sa["bins"]]
            ed = edge_of(q)
            ro = odds_fit(sa["bins"])
            rows.append([f(T, 5), f(Ne), f(rate, 3), f(sum(s["spectrum"]["S_diag_rad"] for s in q["sectors"].values())),
                         f(sum(s["spectrum"]["S_diag_near"] for s in q["sectors"].values())), ", ".join(f(m_, 2) for m_ in means), ", ".join(str(c) for c in counts),
                         "-" if ed is None else "%s T_H" % f(ed, 3), "%s / %s (%d bins)" % (f(ro["slope_per_TH"], 3), f(ro["intercept"], 3), ro["bins_used"]) if ro else "-"])
        L += ["### j = %g (bins: sector alpha's particle means over [0, 8] T_H in 1 T_H bins; the edge over both sectors)" % j, "",
              table(["T/r_s0", "N_ext", "dN_ext/dT (occupation-change rate)", "S_diag,rad", "S_diag,near", "bin means", "bin counts", "upper edge",
                     "raw odds slope/intercept (contains the transmission)"], rows), ""]
    # ---- T7p5 controls (version 1's, with C-pg-15)
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
    for lab, Rr in (("C-pg-3 F0", load(a.f0)), ("C-pg-4 S0 on L", R.get("S0_L"))):
        if Rr is None:
            cl.append([lab, "missing"])
            continue
        rr = [q for v in Rr["channels"].values() for q in v]
        mx = lambda fn: max(abs(fn(s)) for q in rr for s in q["sectors"].values())  # noqa: E731
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
        import strand7_sea_page_v4 as SP
        x = np.linspace(1e-6, 1 - 1e-6, 1001)
        cl.append(["C-pg-6 h2", "max |h2 - the paper's lattice_entropy.h2| on 1001 points in (0, 1): %s" % f(float(np.max(np.abs(SP.h2(x) - mod.h2(x)))), 3)])
    else:
        cl.append(["C-pg-6 h2", "the paper's module not given"])
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
    cc = [q.get("connection_control") for (t, j, q) in allrows if q.get("connection_control")]
    if cc:
        x = cc[0]
        cl.append(["C-pg-11 connection", "ratio with the connection %s (expected about 1/8); without %s (expected about 1/2); norms %s" % (
            f(x.get("ratio_with"), 4), f(x.get("ratio_without"), 4), json.dumps({k: v for k, v in x.items() if k.startswith("tau")}, default=float))])
    if "J" in R:
        Jc = merged([R["J"]])
        jrows = []
        for T in sorted(next(iter(Jc.values()))):
            jr = wsum(Jc, T, lambda s: s["cuts"]["rad"]["dSad"])
            jt = wsum(Jc, T, lambda s: s["cuts"]["throat"]["dSad"])
            pr, pt = at(Prows, T, "dS_rad"), at(Prows, T, "dS_thr")
            if pr is not None:
                jrows.append("T %s: rad %s (%s of the P sum), ref %s (%s)" % (f(T, 4), f(jr, 3), f(jr / pr if pr else None, 3), f(jt, 3),
                                                                            f(jt / pt if pt else None, 3)))
        Jn = sum(s["spectrum"]["N_ext"] for q in R["J"]["channels"]["3.5"][-1:] for s in q["sectors"].values()) if "3.5" in R["J"]["channels"] else None
        cl.append(["C-pg-12 j = 7/2 (weight 8)", "; ".join(jrows[-4:]) + ("; N_ext at its last output %s (the wall's edge modes counted in it; T7p5 C-pg-13)" % f(Jn, 3) if Jn is not None else "")])
    nz = [s["controls"]["near_zero_modes"] for (t, j, q) in allrows for s in q["sectors"].values()]
    mo = [s["controls"]["smallest_abs_omega"] for (t, j, q) in allrows for s in q["sectors"].values()]
    cl.append(["C-pg-13 near-zero modes", "largest count of |omega| < 1e-6 per sector and output %d; smallest |omega| %s" % (max(nz), f(min(mo), 3))])
    lk = R[P_tags[0]].get("locked")
    if lk:
        traj = np.array(lk["s_over_A0"])
        Tq_, sA = traj[:, 0], traj[:, 1]
        A0 = lk["A0"]
        d_ = sA * A0 * (1 + 1 / sA) ** 2 - 4 * A0
        rhv = lk["r_h"]
        late = []
        for (t1, t2) in ((14, 18), (18, 22), (22, 26)):
            m = (Tq_ >= t1) & (Tq_ <= t2) & (d_ > 0)
            if np.sum(m) >= 3:
                late.append("[%g, %g]: %s" % (t1, t2, f(-np.polyfit(Tq_[m] * rs0, np.log(d_[m]), 1)[0] * rhv, 5)))
        fzs = ""
        if a.frozen and os.path.exists(a.frozen):
            FZ = load(a.frozen)
            recs = FZ["run"]["records"]
            Tz = np.array([q["T"] for q in recs])
            dz = np.array([q["R_sh_minus_Rtrap"] for q in recs])
            rhz = 4 * FZ["run"]["A0"]
            mz = (Tz / rhz >= Tz[-1] / rhz - 10) & (dz > 0)
            fzs = "; frozen_schw E8's late lambda r_h %s over its last 10 r_h" % f(-np.polyfit(Tz[mz], np.log(dz[mz]), 1)[0] * rhz, 5)
        cl.append(["C-pg-14 L's trajectory", "energy control max |N W/eps - 1| = %s; lambda r_h of R - 4A_0 by window (r_s0) %s%s" % (
            f(lk["energy_control"]["max_rel_dev_NW"], 3), "; ".join(late), fzs)])
    reps = sorted(glob.glob(os.path.join(a.dir, "repair_*.json")))
    for rp in reps:
        try:
            RS = load(rp)
        except Exception:  # noqa: BLE001
            continue
        for tg, v in RS.items():
            cl.append(["C-pg-16 %s" % tg, "%d entries; %d above 1e-6 (of which %d equal ln 2 within 1e-5); the largest |difference| below that: S_int_vacad %s, "
                       "MI_vacad %s, G_vacad %s, projector %s; T = 0 projector %s" % (
                           v["entries"], v["above_1e-6"], v["ln2_entries"], f(v["max_below"]["d_S_int_vacad"], 3), f(v["max_below"]["d_MI_vacad"], 3),
                           f(v["max_below"]["d_G_vacad_max"], 3), f(v["max_below"]["projector_diff"], 3), json.dumps(v["T0"], default=float))])
    if c17 is not None:
        cl.append(["C-pg-17 (P5_j2.5 v5 against P_j2.5 r2)", "%s over %d outputs: %s" % ("holds" if c17["holds"] else "does not hold", c17["outputs"],
                                                                                      ", ".join("%s %s" % (k, f(v, 3)) for k, v in c17.items() if k not in ("outputs", "holds")))])
    v5c = []
    for t, Rr in R.items():
        vp = [s["controls"].get("vac_projector_diff") for v in Rr["channels"].values() for q in v for s in q["sectors"].values()]
        vp = [x for x in vp if x is not None]
        up = [s["controls"].get("unresolved_pairs") for v in Rr["channels"].values() for q in v for s in q["sectors"].values()]
        up = [x for x in up if x is not None]
        if vp:
            v5c.append("%s %s (unresolved pairs up to %s)" % (t, f(max(vp), 3), max(up) if up else "-"))
    cl.append(["the reference's projector (v5 or r2)", "the largest ||P_eigh - P_polar||_F per run: " + "; ".join(v5c)])
    L += [table(["control", "value"], cl), ""]
    # ---- T7p10 the excluded pairs (sec. 14.3)
    L += ["## T7p10 The excluded pairs (sec. 14.3): per run, channel and sector at every 4 r_s0", ""]
    x10 = []
    for t, Rr in sorted(R.items()):
        for jk, v in Rr["channels"].items():
            for q in v:
                if abs(q["T_over_rs0"] / 4 - round(q["T_over_rs0"] / 4)) > 1e-6:
                    continue
                for name, s_ in q["sectors"].items():
                    ex = s_["spectrum"].get("excluded_pairs") or []
                    lr = s_["spectrum"].get("lowest_resolved_pair")
                    if not ex and not lr:
                        continue
                    exs = "; ".join(", ".join("%s %s" % (k, f(val, 3)) for k, val in e.items() if k != "label") for e in ex) or "-"
                    lrs = ", ".join("%s %s" % (k, f(val, 3)) for k, val in lr.items()) if lr else "-"
                    x10.append([t, jk, name, f(q["T_over_rs0"], 4), exs, lrs])
    L += [table(["run", "j", "sector", "T/r_s0", "excluded pairs", "lowest resolved pair"], x10) if x10 else "- none.", ""]
    # ---- T7p6 the checks (sec. 13.7)
    rows = []

    def cmp_rows(lab, A_rows, B_rows, Ts):
        for Tq in Ts:
            if at(A_rows, Tq, "dS_rad") is None or at(B_rows, Tq, "dS_rad") is None:
                continue
            rows.append([lab, f(Tq, 4)] + ["%s / %s" % (f(at(A_rows, Tq, k)), f(at(B_rows, Tq, k))) for k in ("dS_rad", "dS_thr", "S_diag_rad", "N_ext", "g_h")])
    sample = (20.0, 24.0, 30.0, 36.0, 40.0)
    for t in ("K1", "K2"):
        if t in chk:
            cmp_rows("%s / P (cadence)" % t, chk[t], p05, sample)
    if "B40" in chk:
        cmp_rows("B40 / P (C-pg-15: the box alone)", chk["B40"], p05, sample)
    if "H" in chk and "B40" in chk:
        cmp_rows("H / B40 (C-pg-9: the spacing alone)", chk["H"], chk["B40"], sample)
    if "R" in chk and "B40" in chk:
        cmp_rows("R / B40 (C-pg-10: r_h/h alone, at the same T/r_s0)", chk["R"], chk["B40"], sample)
    extra = []
    if "B40" in chk:
        dif = {k: max((abs(at(chk["B40"], T, k) - at(p05, T, k)) for T in Tp if at(chk["B40"], T, k) is not None and at(p05, T, k) is not None), default=None)
               for k in ("dS_rad", "dS_thr", "S_diag_rad", "N_ext", "g_h")}
        extra.append("C-pg-15 (B40 against P, j = 1/2, every common output): the largest differences " + ", ".join("%s %s" % (k, f(v, 3)) for k, v in dif.items()) + ".")
    if "R" in R and "B40" in R:
        try:
            mR = MK.markers_L(R["R"], c_band)
            mB = MK.markers_L(R["B40"], c_band)
            rh_rs0 = R["B40"]["locked"]["r_h"] / R["B40"]["meta"]["r_s0"]
            sh = {k: (mR[k] - mB[k]) if (mR[k] is not None and mB[k] is not None) else None for k in ("T_1", "T_2", "T_2p", "T_lw")}
            extra.append("C-pg-10's markers: R minus B40 (r_s0): " + ", ".join("%s %s" % (k, f(v, 4)) for k, v in sh.items()) +
                         " against the predicted shift 2 r_h ln 2 = %s r_s0 for the markers set by h T_H (T_2, T_2p, T_lw) and 0 for T_1." % f(2 * rh_rs0 * math.log(2), 4))
            js["markers_R"], js["markers_B40"], js["marker_shifts_R_minus_B40"] = mR, mB, sh
        except Exception as e:  # noqa: BLE001
            extra.append("C-pg-10's markers: not computed (%s)." % e)
    L += ["## T7p6 The checks at j = 1/2 (first / second)", ""] + (["- " + x for x in extra] + [""] if extra else []) + [
        table(["comparison", "T/r_s0", "dS_rad", "dS_ref", "S_diag,rad", "N_ext", "g(h) ref"], rows), ""]
    # ---- T7p7 C
    if C_tags:
        Cc = merged([R[t] for t in C_tags])
        cmeta = R[C_tags[0]]["meta"]
        s7a = a.c_s7a_json or cmeta.get("s7a_json")
        mkC = MK.markers_C(s7a, a.c_fields, c_band, float(cmeta["h"]), float(cmeta["r_s0"])) if s7a and os.path.exists(s7a) else None
        js["markers_C"] = mkC
        C05 = sums({0.5: Cc[0.5]}, h) if 0.5 in Cc else []
        rows = []
        for r in C05:
            pv = at(p05, r["T"], "dS_rad")
            rows.append([f(r["T"], 5), f(r["dS_rad"]), f(pv), f(r["dS_thr"]), f(at(p05, r["T"], "dS_thr")), f(r["S_diag_rad"]), f(r["N_ext"]), f(r["g_h"], 3), f(r["N_s"], 3)])
        cl_ = sorted(Cc)
        L += ["## T7p7 The comparison C, the rows' own unlocked collapse (self E16; an approximate numerical history): matched at j = 1/2", "",
              "- C's channels: %s; C's markers (%s): T_1 = %s, T_2 = %s, T_2p = %s, T_lw = %s r_s0 (N(r_sh) at the last record %s against h T_H %s)." % (
                  ", ".join("%g" % x for x in cl_), (mkC or {}).get("psi_source", "-"), f((mkC or {}).get("T_1"), 5), f((mkC or {}).get("T_2"), 5),
                  f((mkC or {}).get("T_2p"), 5), f((mkC or {}).get("T_lw"), 5), f((mkC or {}).get("N_sh_last"), 4), f((mkC or {}).get("hT_H_last"), 4))]
        if a.part2_json and os.path.exists(a.part2_json):
            P2J = load(a.part2_json)
            sens = P2J.get("sensitivities_for_item7") or {}
            L.append("- C's sensitivities (the part 2 report, SPEC_strand7_collapse.md sec. 11.3): %s." % json.dumps(sens, default=float))
            js["C_sensitivities"] = sens
        else:
            L.append("- C's sensitivities and T_valid: pending (the part 2 report, SPEC_strand7_collapse.md sec. 11.3).")
        L += ["", table(["T/r_s0", "C dS_rad (j = 1/2)", "P dS_rad (j = 1/2)", "C dS_ref (j = 1/2)", "P dS_ref (j = 1/2)", "C S_diag,rad", "C N_ext", "C g(h) ref", "C N_s"], rows), ""]
        js["C_half"] = C05
        if set(cl_) == set(chans):
            Crows = sums(Cc, h)
            js["C_sums"] = Crows
            L += ["### C's channel sum (channels %s, matching P's)" % ", ".join("%g" % x for x in cl_), "",
                  table(["T/r_s0", "dS_rad", "dS_ref", "S_diag,rad", "N_ext", "g(h) ref"], [[f(r["T"], 5), f(r["dS_rad"]), f(r["dS_thr"]), f(r["S_diag_rad"]),
                                                                                         f(r["N_ext"]), f(r["g_h"], 3)] for r in Crows]), ""]
        if C05:
            summ.append("C at j = 1/2 (an approximate numerical history) at T = %s r_s0: dS_rad = %s, dS_ref = %s; C's channels %s." % (
                f(C05[-1]["T"], 4), f(C05[-1]["dS_rad"]), f(C05[-1]["dS_thr"]), ", ".join("%g" % x for x in cl_)))
    # ---- T7p9 L
    if lk:
        L += ["## T7p9 L's trajectory (every 0.5 r_s0 in the record; shown every 2 r_s0)", "", "- A_0 = %s, r_h = 4 A_0 = %s, R_0 = %s; energy control %s." % (
            f(lk["A0"], 8), f(lk["r_h"], 8), f(lk["R0"], 6), json.dumps(lk["energy_control"], default=float)), "",
              table(["T/r_s0", "s/A_0", "N_s", "c_+^(r)(s) = N_s/psi_s^2"], [[f(x[0], 4), f(x[1], 8), f(x[2], 4), f(x[2] / (1 + 1 / x[1]) ** 2, 4)]
                                                                            for x in lk["s_over_A0"] if abs(x[0] / 2 - round(x[0] / 2)) < 1e-6]), ""]
    # markers' values of the curves (labelled interpolations)
    mv = []
    for name, Tm in (("T_1", T1), ("T_2p", T2p), ("T_2", T2), ("T_w", Tw)):
        if Tm is not None:
            mv.append("%s = %s: dS_rad %s, dS_ref %s" % (name, f(Tm, 5), f(interp_at(Prows, Tm, "dS_rad")), f(interp_at(Prows, Tm, "dS_thr"))))
    summ.append("At the markers (linear interpolations between neighbouring outputs, labelled): " + "; ".join(mv) + ".")
    L = L[:2] + ["## Emitted summary", ""] + ["- " + s for s in summ] + [""] + L[2:]
    with open(a.md, "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(a.out_json, "w") as fh:
        json.dump(js, fh, indent=1, default=float)
    print("\n".join("- " + s for s in summ))


if __name__ == "__main__":
    main()
