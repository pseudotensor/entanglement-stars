#!/usr/bin/env python
"""Strand 7 S7a report (SPEC_strand7_collapse.md sec. 4 tables T7a1 to T7a4; builder analyst).  Reads <dir>/s7a_<tag>.json (strand7_shell_v1.py)
and writes the emitted summary and the tables as markdown:
  S   per run: the setting, the stop, the kill flags, the end values, the late-time fit (the same numbers as T7a2 and T7a4);
  P   per self-consistent or held run: A/A(0) - 1 against the lead's prediction G E_0/(2 R_sh) - G E_0/R_0 as the spec writes it (which is
      -G E_0/(2 R_0), not 0, at T = 0) and against that prediction minus its T = 0 value, G E_0/(2 R_sh) - G E_0/(2 R_0), where R_sh first
      falls through 3.5, 3, 2.5, 2, 1.5 and 1.25 r_s (linear interpolation between output times);
  T7a1 per run: the monitors at every r_s of external clock and the last record;
  T7a2 per run: the late-time fit over the last 10 r_s of the run: lambda from a least-squares line of ln(R_sh - 4A) against T, against 1/r_s
      with r_s = 4A at the last record (lambda 4A); the same against the exterior form's lapse-zero radius (A + B)^2/B; u_H as the median
      and the last value of -R_dot/F (the middle marker's R_dot, F = 1 - 4A/R_sh) over the window; skipped where R_sh - 4A is not positive;
  T7a3 the controls: C1 (static_check records), C2 (the frozen runs: the largest |sum nu H/its start - 1|, marker energy change, and the
      deviations from the exact static-background law and from GAP_collapse 2.4's form), C3 (per pair base/variant given by --c3: the largest
      difference of each monitor over the common output times up to the base's D1 flag, or over the whole run without one; the resolution
      floor of a monitor is the larger of its two differences; D1 is asserted when |A/A(0) - 1| at the flag exceeds ten times the floor of
      A/A(0) - 1), C4 (the largest |Gauss/tail - 1| of A and B per run), C5 (the vacuum record);
  T7a4 per run: the kill flags with the time and R_sh at which they triggered, and the stop."""
import argparse
import glob
import json
import os

import numpy as np


def f(x, d=6):
    if x is None:
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    x = float(x)
    if x == 0:
        return "0"
    if 1e-3 <= abs(x) < 1e5:
        return ("%." + str(d) + "g") % x
    return "%.4e" % x


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def setting(m):
    s = "%s E0=%g R0=%g r_s case %s h=%g dt=%g r_s" % (m["run"], m["E0"], m["R0_over_rs"], m["case"], m["h"], m["dt_over_rs"])
    if m["run"].startswith("frozen"):
        s += " " + m["integrator"]
    return s


def late_fit(recs, rs):
    T = np.array([q["T"] for q in recs])
    d = np.array([q["R_sh_minus_Rtrap"] for q in recs])
    w = T >= T[-1] - 10 * rs
    out = {"window_T_over_rs": [float(T[w][0] / rs), float(T[-1] / rs)], "n": int(np.sum(w))}
    if np.sum(w) >= 3 and np.all(d[w] > 0):
        sl = np.polyfit(T[w], np.log(d[w]), 1)[0]
        out["lambda"] = -sl
        out["lambda_4A_end"] = -sl * 4 * recs[-1]["A"]
        out["lambda_rs"] = -sl * rs
    else:
        out["skip"] = "R_sh - 4A not positive in the window" if np.sum(w) >= 3 else "fewer than 3 records"
    dl = np.array([q["R_sh"] - q["R_lapse0_ext"] if q["R_lapse0_ext"] is not None else np.nan for q in recs])
    if np.sum(w) >= 3 and np.all(dl[w] > 0):
        sl = np.polyfit(T[w], np.log(dl[w]), 1)[0]
        out["lambda_lapse0"] = -sl
        out["lambda_lapse0_Rl"] = -sl * recs[-1]["R_lapse0_ext"]
    uh = np.array([q["uH_inst"] if q["uH_inst"] is not None else np.nan for q in recs])
    if np.any(np.isfinite(uh[w])):
        out["uH_median"] = float(np.nanmedian(uh[w]))
        out["uH_last"] = float(uh[w][-1])
    return out


def crossings(recs, rs, levels=(3.5, 3.0, 2.5, 2.0, 1.5, 1.25)):
    rows = []
    x = np.array([q["R_sh"] / rs for q in recs])
    for L in levels:
        k = np.nonzero((x[:-1] > L) & (x[1:] <= L))[0]
        if not len(k):
            rows.append((L, None, None, None, None))
            continue
        k = int(k[0])
        t = (x[k] - L) / (x[k] - x[k + 1])
        Ar = recs[k]["A_rel"] + t * (recs[k + 1]["A_rel"] - recs[k]["A_rel"])
        pr = recs[k]["prediction"] + t * (recs[k + 1]["prediction"] - recs[k]["prediction"])
        Tc = recs[k]["T_over_rs"] + t * (recs[k + 1]["T_over_rs"] - recs[k]["T_over_rs"])
        rows.append((L, Tc, Ar, pr, pr - recs[0]["prediction"]))
    return rows


C3_KEYS = ["A_rel", "B_over_A_minus_1", "R_sh", "R_trap", "c_plus_r", "N_sh", "u2", "sum_nu_H_rel"]


def c3_diff(base, var, rs):
    tb = {round(q["T_over_rs"], 6): q for q in base["records"]}
    tv = {round(q["T_over_rs"], 6): q for q in var["records"]}
    tmax = base["kills"]["D1"]["T_over_rs"] if "D1" in base["kills"] else max(tb)
    common = sorted(t for t in tb if t in tv and t <= tmax + 1e-9)
    out = {"n_common": len(common), "up_to_T_over_rs": tmax}
    for k in C3_KEYS:
        vals = [abs(tb[t][k] - tv[t][k]) for t in common if tb[t].get(k) is not None and tv[t].get(k) is not None]
        out[k] = max(vals) if vals else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--c3", default="", help="label=base_tag:variant_tag,...")
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    R = {}
    for p in sorted(glob.glob(os.path.join(a.dir, "s7a_*.json"))):
        with open(p) as fh:
            R[os.path.basename(p)[4:-5]] = json.load(fh)
    dyn = {t: r for t, r in R.items() if "run" in r}
    L = ["# Strand 7 S7a tables (strand7_shell_report_v1.py over %s)" % a.dir, ""]
    rows = []
    fits = {}
    for t, r in dyn.items():
        m, q = r["meta"], r["run"]
        rs = m["r_s"]
        fits[t] = late_fit(q["records"], rs)
        e = q["records"][-1]
        rows.append([t, setting(m), q["stop"], ",".join(sorted(q["kills"])) or "none", f(e["T_over_rs"], 5), f(e["A_rel"]), f(e["prediction"]),
                     f(e["prediction"] - q["records"][0]["prediction"]),
                     f(e["B_over_A_minus_1"]), f(e["R_sh"] / rs), f(e["R_trap"] / rs), f(e["R_sh_minus_Rtrap"] / rs), f(e["c_plus_r"]),
                     f(fits[t].get("lambda_4A_end")), f(fits[t].get("uH_median")), f(q["seconds"], 4)])
    L += ["## S Summary per run (the end record; lambda 4A and u_H from T7a2)", "",
          table(["tag", "setting", "stop", "kills", "T_end/r_s", "A/A(0)-1", "prediction as written", "minus its T = 0 value", "B/A-1", "R_sh/r_s", "4A/r_s",
                 "(R_sh-4A)/r_s", "c_+^(r)",
                 "lambda 4A", "u_H median", "seconds"], rows), ""]
    L += ["## Initial states", "",
          table(["tag", "r_0", "A(0)", "B(0)", "B(0)/A(0)", "4A(0)/r_s", "(A0+B0)^2/B0 / r_s", "secant its", "initial outer/newton", "initial residual"],
                [[t, f(r["run"]["r0"], 9), f(r["run"]["A0"], 9), f(r["run"]["B0"], 9), f(r["run"]["B0_over_A0"], 9), f(r["run"]["R_trap0"] / r["meta"]["r_s"]),
                  f(r["run"]["R_lapse0_ext0"] / r["meta"]["r_s"]), f(r["run"]["secant"]["secant_iterations"]),
                  "%s/%s" % (r["run"]["initial_solve"]["outer"], r["run"]["initial_solve"]["newton"]), f(r["run"]["initial_solve"]["residual"])]
                 for t, r in dyn.items()]), ""]
    L += ["## P The charge against the lead's prediction where R_sh first falls through each level", ""]
    prow = []
    for t, r in dyn.items():
        if r["meta"]["run"] not in ("self", "held"):
            continue
        for Lv, Tc, Ar, pr, ps in crossings(r["run"]["records"], r["meta"]["r_s"]):
            prow.append([t, f(Lv), f(Tc, 5), f(Ar), f(pr), f(Ar / pr if (Ar is not None and pr) else None), f(ps), f(Ar / ps if (Ar is not None and ps) else None)])
    L += [table(["tag", "R_sh/r_s", "T/r_s", "A/A(0)-1", "prediction as written", "ratio", "prediction minus its T = 0 value", "ratio"], prow), ""]
    L += ["## T7a1 Monitors at every r_s of external clock and the last record", ""]
    head = ["T/r_s", "A/A(0)-1", "prediction", "B/A-1", "<u^2>", "R_sh/r_s", "4A/r_s", "(R_sh-4A)/r_s", "R_N0/r_s", "N(r_sh)", "N psi(r_sh)", "N(r_99)",
            "c_+^(r)", "c_+^(R)", "N/N_ext", "chi<=0 node r", "sum nu H rel", "-R_dot/F", "residual", "outer", "newton", "fp"]
    for t, r in dyn.items():
        rs = r["meta"]["r_s"]
        recs = r["run"]["records"]
        sel = [q for q in recs if abs(q["T_over_rs"] - round(q["T_over_rs"])) < 1e-6]
        if recs[-1] is not sel[-1]:
            sel.append(recs[-1])
        L += ["### %s (%s)" % (t, setting(r["meta"])), "",
              table(head, [[f(q["T_over_rs"], 5), f(q["A_rel"]), f(q["prediction"]), f(q["B_over_A_minus_1"]), f(q["u2"]), f(q["R_sh"] / rs),
                            f(q["R_trap"] / rs), f(q["R_sh_minus_Rtrap"] / rs), f(q["R_lapse0_ext"] / rs if q["R_lapse0_ext"] else None), f(q["N_sh"]),
                            f(q["Npsi_sh"]), f(q["N_out"]), f(q["c_plus_r"]), f(q["c_plus_R"]), f(q["N_over_Next"]), f(q["first_chi_nonpositive_node_r"]),
                            f(q["sum_nu_H_rel"]), f(q["uH_inst"]), f(q["residual"]), f(q["outer"]), f(q["newton"]), f(q["fp_iterations"])]
                           for q in sel]), ""]
    L += ["## T7a2 Late-time fits (the last 10 r_s of each run)", "",
          table(["tag", "window T/r_s", "records", "lambda", "lambda 4A(end)", "lambda r_s", "lambda_N0 R_N0", "u_H median", "u_H last", "note"],
                [[t, "%s to %s" % (f(fz["window_T_over_rs"][0], 4), f(fz["window_T_over_rs"][1], 4)), f(fz["n"]), f(fz.get("lambda")),
                  f(fz.get("lambda_4A_end")), f(fz.get("lambda_rs")), f(fz.get("lambda_lapse0_Rl")), f(fz.get("uH_median")), f(fz.get("uH_last")),
                  fz.get("skip", "")] for t, fz in fits.items()]), ""]
    L += ["## T7a3 Controls", "", "### C1 static check", ""]
    c1 = [(t, r["static_check"]) for t, r in R.items() if "static_check" in r]
    if c1:
        keys = [k for k in c1[0][1] if k not in ("resolve_info",)]
        L += [table(["quantity"] + [t for t, _ in c1], [[k] + [f(c[k], 8) for _, c in c1] for k in keys]), ""]
    L += ["### C2 frozen-background runs (largest over the records)", ""]
    c2 = []
    for t, r in dyn.items():
        if not r["meta"]["run"].startswith("frozen"):
            continue
        recs = r["run"]["records"]

        def mx(k):
            v = [abs(q[k]) for q in recs if q.get(k) is not None]
            return max(v) if v else None
        c2.append([t, setting(r["meta"]), f(mx("sum_nu_H_rel")), f(mx("C2_marker_energy_max_rel")), f(mx("C2_law_max_rel")), f(mx("C2_schw_form_max_rel")),
                   f(min(q.get("C2_law_n", 0) for q in recs[1:]) if len(recs) > 1 else None)])
    L += [table(["tag", "setting", "|sum nu H rel|", "marker energy", "static law", "GAP 2.4 form", "fewest markers in the law test"], c2), ""]
    L += ["### C3 step-halving and grid-doubling differences", ""]
    c3rows = []
    floors = {}
    for item in [x for x in a.c3.split(",") if x.strip()]:
        lab, pair = item.split("=")
        b, v = pair.split(":")
        if b not in dyn or v not in dyn:
            c3rows.append([lab, b, v] + ["missing"] * (len(C3_KEYS) + 2))
            continue
        d = c3_diff(dyn[b]["run"], dyn[v]["run"], dyn[b]["meta"]["r_s"])
        for k in C3_KEYS:
            if d[k] is not None:
                floors.setdefault(b, {})[k] = max(floors.get(b, {}).get(k, 0.0), d[k])
        c3rows.append([lab, b, v, f(d["n_common"]), f(d["up_to_T_over_rs"], 5)] + [f(d[k]) for k in C3_KEYS])
    L += [table(["label", "base", "variant", "common outputs", "up to T/r_s"] + C3_KEYS, c3rows), ""]
    d1rows = []
    for b, fl in floors.items():
        q = dyn[b]["run"]
        if "D1" in q["kills"]:
            Ar = q["kills"]["D1"]["A_rel"]
            d1rows.append([b, f(Ar), f(fl.get("A_rel")), str(abs(Ar) > 10 * fl.get("A_rel", np.inf))])
        else:
            d1rows.append([b, "no flag", f(fl.get("A_rel")), "-"])
    L += ["D1 with the floor (the larger of the two differences of A/A(0) - 1):", "",
          table(["base", "A/A(0)-1 at the flag", "floor", "flag above ten floors"], d1rows), ""]
    L += ["### C4 Gauss against tail (largest over the records)", "",
          table(["tag", "|A_gauss/A_tail - 1|", "|B_gauss/B_tail - 1|"],
                [[t, f(max(abs(q["gauss_vs_tail_A"]) for q in r["run"]["records"] if q["gauss_vs_tail_A"] is not None)),
                  f(max(abs(q["gauss_vs_tail_B"]) for q in r["run"]["records"] if q["gauss_vs_tail_B"] is not None))]
                 for t, r in dyn.items() if not r["meta"]["run"].startswith("frozen")]), ""]
    L += ["### C5 vacuum", ""]
    for t, r in R.items():
        if "vacuum" in r:
            L += ["%s: %s" % (t, ", ".join("%s %s" % (k, f(v)) for k, v in r["vacuum"].items())), ""]
    L += ["## T7a4 Kill flags and stops", "",
          table(["tag", "stop", "D1 (T/r_s, R_sh/r_s, A/A(0)-1)", "D2 (T/r_s, R_sh/r_s, c_+^(r))", "steps", "seconds"],
                [[t, r["run"]["stop"], ", ".join(f(x) for x in r["run"]["kills"]["D1"].values()) if "D1" in r["run"]["kills"] else "-",
                  ", ".join(f(x) for x in r["run"]["kills"]["D2"].values()) if "D2" in r["run"]["kills"] else "-",
                  f(r["run"].get("steps")), f(r["run"]["seconds"], 4)] for t, r in dyn.items()]), ""]
    with open(a.md, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote %s (%d runs)" % (a.md, len(R)))


if __name__ == "__main__":
    main()
