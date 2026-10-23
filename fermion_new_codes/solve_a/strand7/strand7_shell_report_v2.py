#!/usr/bin/env python
"""Strand 7 S7a revision 2 report (SPEC_strand7_collapse.md sec. 9.2; builder analyst).  Reads <dir>/s7a_<tag>.json (strand7_shell_v2.py or v3)
and, where present, <dir>/s7a_<tag>_fields.npz, and writes the emitted summary and the tables as markdown:
  S2  per run: the setting, the stop, the kill flags (D1, D2 with their times), the end values of the sec. 9.2 monitors;
  TC  per run: the record at the throat crossing r_sh = A (the run's head, linear interpolation in T), or the smallest r_sh/A reached;
  R1  per self-consistent run: the decision rule (i)'s quantities and the rule as written, applied mechanically (no reading): the freeze test
      (c_+^(r)(r_sh) at the last record below 1e-3, D2's threshold, the builder's operational reading of 'c_+ -> 0'; the largest |dA/dT| r_s/A,
      |dB/dT| r_s/B and |d r_sh/dT| r_s/r_sh over the last 5 r_s, differences of consecutive outputs, against 1e-3) and the runaway test
      (r_sh < A reached, c_+ > 0.01 and B/A < 0.3 at the crossing);
  R3  per held-lock run: the decision rule (iii)'s quantities: the late-time fit over the last 10 r_s before the first record with R_sh - 4A <=
      1e-6 r_s (that record excluded), lambda 4A and u_H, against frozen_schw's (--frozen-ref: the late fit of its whole last 10 r_s), and their
      relative differences against 2 percent; the exterior check (the largest |N - (1 - A/r)/(1 + A/r)| beyond r_p99 + 2 w_r over the records,
      against 1e-10);
  P   per self-consistent or held-lock run: A/A(0) - 1 against the lead's withdrawn prediction (both forms, as revision 1's table P) where R_sh
      first falls through 3.5 ... 1.25 r_s;
  L   per run with a fields file (the builder's diagnostic, labelled; not in the spec): at the last four outputs spaced 1 r_s and at the last
      output, the areal radius R = r psi^2 over 4A at the 1st percentile, the median and the 99th percentile radii; the exterior form's
      r (1 + A/r)^2 at the median over 4A; N at the three radii; the smallest dR/dr over the nodes (finite differences) and its radius (a
      minimal surface of the slice where it is not positive);
  T7b1 per run: the sec. 9.2 monitors at every r_s of external clock and at every output of the last r_s;
  C3  step-halving pairs (--c3 label=base:variant,...) over the common outputs: the largest differences of the monitors;
  C4  the largest |Gauss/tail - 1| of A and B per run;
  M   the markers' <N (1 + 2 u^2)> (weighted by nu W/psi, and unweighted) against B/A: the largest |ratio - 1| over the records and at the end."""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand7_shell_report_v1 as R1  # noqa: E402

f, table = R1.f, R1.table


def setting(m):
    s = "%s E0=%g R0=%g r_s case %s h=%g dt=%g r_s Np=%d" % (m["run"], m["E0"], m["R0_over_rs"], m["case"], m["h"], m["dt_over_rs"], m["Np"])
    if m["case"] == "B":
        s += " vperp=%g" % m["vperp"]
    return s


def rates(recs, rs, span=5.0):
    T = np.array([q["T"] for q in recs])
    w = np.nonzero(T >= T[-1] - span * rs)[0]
    if len(w) < 2:
        return None
    out = {}
    for k, lab in (("A", "dA"), ("B", "dB"), ("r_sh", "dr")):
        x = np.array([recs[i][k] for i in w])
        t = T[w]
        d = np.abs(np.diff(x) / np.diff(t)) * rs / np.abs(x[1:])
        out[lab] = float(np.max(d))
    return out


def late_fit_before(recs, rs, frs):
    """the late fit over the last 10 r_s before the first record with R_sh - 4A <= 1e-6 r_s (excluded)"""
    d = np.array([q["R_sh_minus_Rtrap"] for q in recs])
    k = np.nonzero(d <= 1e-6 * rs)[0]
    end = int(k[0]) if len(k) else len(recs)
    sub = recs[:end]
    if len(sub) < 3:
        return {"skip": "fewer than 3 records before the crossing"}
    out = R1.late_fit(sub, rs)
    out["first_nonpositive_T_over_rs"] = float(recs[end]["T_over_rs"]) if end < len(recs) else None
    out["lambda_4A_last_positive"] = out.get("lambda", np.nan) * 4 * sub[-1]["A"] if "lambda" in out else None
    return out


def layers(fp, recs, rs):
    z = np.load(fp)
    r, Tz = z["r"], z["T"]
    byT = {round(float(q["T"]), 6): q for q in recs}
    idx = len(Tz) - 1
    want = [Tz[idx] - k * rs for k in (3, 2, 1)]
    picks = []
    for t in want:
        j = int(np.argmin(np.abs(Tz - t)))
        if abs(Tz[j] - t) < 0.051 * rs and j not in picks:
            picks.append(j)
    picks.append(idx)
    rows = []
    for j in picks:
        q = byT.get(round(float(Tz[j]), 6))
        if q is None:
            continue
        psi = 1 + z["u"][j]
        N = (1 + z["v"][j]) / psi
        R = r * psi ** 2
        A = q["A"]

        def at(x, y):
            return float(np.interp(x, r, y))
        dR = np.diff(R) / np.diff(r)
        rm = 0.5 * (r[1:] + r[:-1])
        i = int(np.argmin(dR))
        rows.append({"T_over_rs": float(Tz[j] / rs), "A": A, "r_sh_over_A": q["r_sh"] / A,
                     "R_p01_4A": at(q["r_p01"], R) / (4 * A), "R_sh_4A": at(q["r_sh"], R) / (4 * A), "R_p99_4A": at(q["r_p99"], R) / (4 * A),
                     "Rext_sh_4A": q["r_sh"] * (1 + A / q["r_sh"]) ** 2 / (4 * A),
                     "N_p01": at(q["r_p01"], N), "N_sh": at(q["r_sh"], N), "N_p99": at(q["r_p99"], N),
                     "min_dRdr": float(dR[i]), "min_dRdr_r_over_A": float(rm[i] / A)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--c3", default="", help="label=base_tag:variant_tag,...")
    ap.add_argument("--frozen-ref", default=None, help="frozen_schw's JSON (revision 1) for rule (iii)")
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    R = {}
    for p in sorted(glob.glob(os.path.join(a.dir, "s7a_*.json"))):
        with open(p) as fh:
            R[os.path.basename(p)[4:-5]] = json.load(fh)
    dyn = {t: r for t, r in R.items() if "run" in r}
    js = {}
    L = ["# Strand 7 S7a revision 2 tables (strand7_shell_report_v2.py over %s)" % a.dir, ""]
    srows, summ = [], []
    for t, r in dyn.items():
        m, q = r["meta"], r["run"]
        rs = m["r_s"]
        e = q["records"][-1]
        k = q["kills"]
        srows.append([t, setting(m), q["stop"], "%s / %s" % (f(k["D1"]["T_over_rs"], 4), f(k["D1"]["A_rel"], 4)) if "D1" in k else "-",
                      "%s / %s" % (f(k["D2"]["T_over_rs"], 4), f(k["D2"]["c_plus_r"], 4)) if "D2" in k else "-",
                      f(e["T_over_rs"], 5), f(e["A_rel"]), f(e["B_over_A"]), f(e["Mclock_rel"] - 1), f(e["D_over_Ebdy"]), f(e["r_sh_over_A"]),
                      f(e["R_sh"] / rs), f(e["R_trap"] / rs), f(e["R_lapse0_ext"] / rs if e["R_lapse0_ext"] else None), f(e["c_plus_r"]), f(e["N_sh"]),
                      "%s (%s)" % (f(e["N_min_nodes"]), f(e["N_min_r"] / e["A"], 4)), f(e.get("lock_factor")), f(m.get("seconds", q["seconds"]), 4)])
        js[t] = {"setting": setting(m), "stop": q["stop"], "kills": k, "end": {kk: e[kk] for kk in (
            "T_over_rs", "A_rel", "B_over_A", "Mclock_rel", "D_over_Ebdy", "r_sh_over_A", "R_sh", "R_trap", "R_lapse0_ext", "c_plus_r", "N_sh", "N_min_nodes",
            "N_min_r")}, "throat_crossing": q.get("throat_crossing")}
    L += ["## S2 Summary per run (the end record; r_s = r_s0 = 2 G E_0)", "",
          table(["tag", "setting", "stop", "D1 T/r_s / A/A(0)-1", "D2 T/r_s / c_+", "T_end/r_s", "A/A(0)-1", "B/A", "M_clock/M_clock(0)-1", "D/E_bdy",
                 "r_sh/A", "R_sh/r_s", "4A/r_s", "(A+B)^2/B / r_s", "c_+^(r)", "N(r_sh)", "smallest N (r/A)", "lock factor", "seconds"], srows), ""]
    trows = []
    for t, r in dyn.items():
        q = r["run"]
        tc = q.get("throat_crossing")
        if tc:
            trows.append([t, f(tc["T_over_rs"], 5), f(tc["c_plus_r"]), f(tc["N_sh"]), f(tc["R_sh_over_rs"]), f(tc["B_over_A"]), f(tc["D_over_Ebdy"])])
        else:
            mn = min(x["r_sh_over_A"] for x in q["records"])
            trows.append([t, "not reached (smallest r_sh/A %s)" % f(mn), "-", "-", "-", "-", "-"])
    L += ["## TC The throat crossing r_sh = A", "", table(["tag", "T/r_s", "c_+^(r)", "N(r_sh)", "R_sh/r_s", "B/A", "D/E_bdy"], trows), ""]
    r1rows = []
    for t, r in dyn.items():
        if r["meta"]["run"] != "self":
            continue
        q, rs = r["run"], r["meta"]["r_s"]
        e = q["records"][-1]
        rt = rates(q["records"], rs)
        tc = q.get("throat_crossing")
        freeze = (e["c_plus_r"] < 1e-3) and rt is not None and max(rt.values()) < 1e-3
        run_away = bool(tc) and tc["c_plus_r"] > 0.01 and tc["B_over_A"] < 0.3
        r1rows.append([t, f(e["T_over_rs"], 5), f(e["c_plus_r"]), f(rt["dA"]) if rt else "-", f(rt["dB"]) if rt else "-", f(rt["dr"]) if rt else "-",
                       str(freeze), str(bool(tc)), f(tc["c_plus_r"]) if tc else "-", f(tc["B_over_A"]) if tc else "-", str(run_away)])
        js[t]["rule_i"] = {"rates_last_5rs": rt, "freeze_as_written": freeze, "runaway_as_written": run_away}
    L += ["## R1 Decision rule (i) as written, applied mechanically (the builder's operational thresholds stated in the header)", "",
          table(["tag", "T_end/r_s", "c_+^(r) at the end", "max |dA/dT| r_s/A (last 5 r_s)", "max |dB/dT| r_s/B", "max |dr_sh/dT| r_s/r_sh",
                 "freeze conditions met", "r_sh < A reached", "c_+ at the crossing", "B/A at the crossing", "runaway conditions met"], r1rows), ""]
    fz_ref = None
    if a.frozen_ref and os.path.exists(a.frozen_ref):
        with open(a.frozen_ref) as fh:
            FR = json.load(fh)
        fz_ref = R1.late_fit(FR["run"]["records"], FR["meta"]["r_s"])
    r3rows = []
    for t, r in dyn.items():
        if r["meta"]["run"] != "heldlock":
            continue
        q, rs = r["run"], r["meta"]["r_s"]
        lf = late_fit_before(q["records"], rs, fz_ref)
        ext = [x["ext_vs_schw_N"] for x in q["records"] if x.get("ext_vs_schw_N") is not None]
        l4 = lf.get("lambda_4A_last_positive")
        uh = lf.get("uH_median")
        dl = (l4 / fz_ref["lambda_4A_end"] - 1) if (fz_ref and l4 is not None and "lambda_4A_end" in fz_ref) else None
        du = (uh / fz_ref["uH_median"] - 1) if (fz_ref and uh is not None and "uH_median" in fz_ref) else None
        r3rows.append([t, "%s to %s" % (f(lf["window_T_over_rs"][0], 4), f(lf["window_T_over_rs"][1], 4)) if "window_T_over_rs" in lf else "-",
                       f(lf.get("first_nonpositive_T_over_rs"), 5), f(l4), f(uh), f(lf.get("uH_last")), f(dl, 3), f(du, 3),
                       str(dl is not None and du is not None and abs(dl) < 0.02 and abs(du) < 0.02), f(max(ext) if ext else None, 3), lf.get("skip", "")])
        js[t]["rule_iii"] = {"late_fit": lf, "frozen_schw_ref": fz_ref, "lambda_rel": dl, "uH_rel": du, "ext_vs_schw_N_max": max(ext) if ext else None}
    L += ["## R3 Decision rule (iii): the held-lock runs' late fit against frozen_schw's (%s), and the exterior check" % (
        "lambda 4A %s, u_H median %s" % (f(fz_ref.get("lambda_4A_end")), f(fz_ref.get("uH_median"))) if fz_ref else "no reference given"), "",
          table(["tag", "window T/r_s", "first R_sh - 4A <= 1e-6 r_s at T/r_s", "lambda 4A (4A at the last positive record)", "u_H median", "u_H last",
                 "lambda 4A / frozen_schw - 1", "u_H / frozen_schw - 1", "both within 2 percent", "max |N - N_schw| beyond r_p99 + 2 w_r", "note"], r3rows), ""]
    prow = []
    for t, r in dyn.items():
        if r["meta"]["run"] not in ("self", "heldlock"):
            continue
        for Lv, Tc, Ar, pr, ps in R1.crossings(r["run"]["records"], r["meta"]["r_s"]):
            prow.append([t, f(Lv), f(Tc, 5), f(Ar), f(pr), f(Ar / pr if (Ar is not None and pr) else None), f(ps), f(Ar / ps if (Ar is not None and ps) else None)])
    L += ["## P The charge against the lead's withdrawn prediction where R_sh first falls through each level", "",
          table(["tag", "R_sh/r_s", "T/r_s", "A/A(0)-1", "G E_0/(2 R_sh) - G E_0/R_0", "ratio", "G E_0/(2 R_sh) - G E_0/(2 R_0)", "ratio"], prow), ""]
    L += ["## L The shell's layers (builder's diagnostic, labelled; from the fields files)", ""]
    for t, r in dyn.items():
        fp = os.path.join(a.dir, "s7a_%s_fields.npz" % t)
        if not os.path.exists(fp):
            continue
        rows = layers(fp, r["run"]["records"], r["meta"]["r_s"])
        js[t]["layers"] = rows
        L += ["### %s" % t, "",
              table(["T/r_s", "r_sh/A", "R(r_p01)/4A", "R(r_sh)/4A", "R(r_p99)/4A", "r_sh (1 + A/r_sh)^2/4A", "N(r_p01)", "N(r_sh)", "N(r_p99)",
                     "smallest dR/dr", "at r/A"],
                    [[f(x["T_over_rs"], 5), f(x["r_sh_over_A"]), f(x["R_p01_4A"], 7), f(x["R_sh_4A"], 7), f(x["R_p99_4A"], 7), f(x["Rext_sh_4A"], 7),
                      f(x["N_p01"]), f(x["N_sh"]), f(x["N_p99"]), f(x["min_dRdr"]), f(x["min_dRdr_r_over_A"], 4)] for x in rows]), ""]
    L += ["## T7b1 Monitors at every r_s of external clock and at every output of the last r_s", ""]
    head = ["T/r_s", "A/A(0)-1", "B/A", "M_clock rel - 1", "D/E_bdy", "r_sh/A", "r_p01/A", "r_p99/A", "R_sh/r_s", "4A/r_s", "(A+B)^2/B / r_s", "B/r_s",
            "N(r_sh)", "c_+^(r)", "N_min (r/A)", "chi<=0 node r", "<N(1+2u^2)>_w/(B/A)", "<u^2>", "psi_in", "N_in", "lock factor", "ext - schw",
            "residual", "outer"]
    for t, r in dyn.items():
        rs = r["meta"]["r_s"]
        recs = r["run"]["records"]
        Tl = recs[-1]["T"]
        sel = [q for q in recs if abs(q["T_over_rs"] - round(q["T_over_rs"])) < 1e-6 or q["T"] >= Tl - rs - 1e-9]
        L += ["### %s (%s)" % (t, setting(r["meta"])), "",
              table(head, [[f(q["T_over_rs"], 5), f(q["A_rel"]), f(q["B_over_A"]), f(q["Mclock_rel"] - 1), f(q["D_over_Ebdy"]), f(q["r_sh_over_A"]),
                            f(q["r_p01_over_A"]), f(q["r_p99_over_A"]), f(q["R_sh"] / rs), f(q["R_trap"] / rs),
                            f(q["R_lapse0_ext"] / rs if q["R_lapse0_ext"] else None), f(q["B"] / rs), f(q["N_sh"]), f(q["c_plus_r"]),
                            "%s (%s)" % (f(q["N_min_nodes"]), f(q["N_min_r"] / q["A"], 4)), f(q["first_chi_nonpositive_node_r"]),
                            f(q["markers_N12u2_weighted"] / q["B_over_A"] if q["B_over_A"] else None), f(q["u2"]), f(q.get("psi_in")), f(q.get("N_in")),
                            f(q.get("lock_factor")), f(q.get("ext_vs_schw_N"), 3), f(q["residual"], 3), f(q["outer"])] for q in sel]), ""]
    c3rows = []
    keys = R1.C3_KEYS + ["B_over_A", "r_sh_over_A", "Mclock_rel"]
    for item in [x for x in a.c3.split(",") if x.strip()]:
        lab, pair = item.split("=")
        b, v = pair.split(":")
        if b not in dyn or v not in dyn:
            c3rows.append([lab, b, v] + ["missing"] * (len(keys) + 1))
            continue
        tb = {round(q["T_over_rs"], 6): q for q in dyn[b]["run"]["records"]}
        tv = {round(q["T_over_rs"], 6): q for q in dyn[v]["run"]["records"]}
        common = sorted(x for x in tb if x in tv)
        row = [lab, b, v, f(len(common))]
        for kk in keys:
            vals = [abs(tb[x][kk] - tv[x][kk]) for x in common if tb[x].get(kk) is not None and tv[x].get(kk) is not None]
            row.append(f(max(vals)) if vals else "-")
        c3rows.append(row)
    L += ["## C3 Step-halving pairs (largest differences over the common outputs)", "", table(["label", "base", "variant", "common outputs"] + keys, c3rows), ""]
    L += ["## C4 Gauss against tail (largest over the records)", "",
          table(["tag", "|A_gauss/A_tail - 1|", "|B_gauss/B_tail - 1|"],
                [[t, f(max(abs(q["gauss_vs_tail_A"]) for q in r["run"]["records"] if q["gauss_vs_tail_A"] is not None)),
                  f(max(abs(q["gauss_vs_tail_B"]) for q in r["run"]["records"] if q["gauss_vs_tail_B"] is not None))] for t, r in dyn.items()]), ""]
    mrows = []
    for t, r in dyn.items():
        recs = r["run"]["records"]
        w = [abs(q["markers_N12u2_weighted"] / q["B_over_A"] - 1) for q in recs if q["B_over_A"]]
        u = [abs(q["markers_N12u2_mean"] / q["B_over_A"] - 1) for q in recs if q["B_over_A"]]
        e = recs[-1]
        mrows.append([t, f(max(w)), f(max(u)), f(e["markers_N12u2_weighted"]), f(e["markers_N12u2_mean"]), f(e["B_over_A"])])
    L += ["## M The markers' <N (1 + 2 u^2)> against B/A (the lead's thin-shell identity)", "",
          table(["tag", "max |weighted/(B/A) - 1|", "max |unweighted/(B/A) - 1|", "weighted at the end", "unweighted at the end", "B/A at the end"], mrows), ""]
    for t, x in js.items():
        e = x["end"]
        s = "%s: stop '%s' at T = %s r_s; A/A(0) - 1 = %s, B/A = %s, r_sh/A = %s, R_sh/4A = %s, c_+^(r) = %s" % (
            t, x["stop"], f(e["T_over_rs"], 5), f(e["A_rel"]), f(e["B_over_A"]), f(e["r_sh_over_A"]), f(e["R_sh"] / e["R_trap"], 7), f(e["c_plus_r"]))
        if "D1" in x["kills"]:
            s += "; D1 at T = %s r_s" % f(x["kills"]["D1"]["T_over_rs"], 4)
        if "D2" in x["kills"]:
            s += "; D2 at T = %s r_s (c_+ %s)" % (f(x["kills"]["D2"]["T_over_rs"], 4), f(x["kills"]["D2"]["c_plus_r"], 4))
        if x.get("throat_crossing"):
            s += "; throat crossing at T = %s r_s with c_+ %s, B/A %s" % (f(x["throat_crossing"]["T_over_rs"], 5), f(x["throat_crossing"]["c_plus_r"]),
                                                                          f(x["throat_crossing"]["B_over_A"]))
        if x.get("layers"):
            y = x["layers"][-1]
            s += "; at the last output R/4A at r_p01, r_sh, r_p99 = %s, %s, %s (exterior form at r_sh %s), smallest dR/dr %s" % (
                f(y["R_p01_4A"], 7), f(y["R_sh_4A"], 7), f(y["R_p99_4A"], 7), f(y["Rext_sh_4A"], 7), f(y["min_dRdr"]))
        summ.append(s + ".")
    L = L[:2] + ["## Emitted summary", ""] + ["- " + s for s in summ] + [""] + L[2:]
    with open(a.md, "w") as fh:
        fh.write("\n".join(L) + "\n")
    if a.out_json:
        with open(a.out_json, "w") as fh:
            json.dump(js, fh, indent=1, default=float)
    print("wrote %s (%d runs)" % (a.md, len(dyn)))
    print("\n".join("- " + s for s in summ))


if __name__ == "__main__":
    main()
