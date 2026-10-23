#!/usr/bin/env python
"""Task 2 report, job T2a (SPEC_task2_static_row_diagnostics.md sec. 2; builder analyst): the static-row diagnostics from the stored
records, beside the post-processor's field-based values (task2_postproc_v1.py, self E16, the only run with stored fields among the spec's).

From every run's record (s7a_<tag>.json, the 0.1 r_s outputs): A, B (Gauss), log E_bdy = log 2A, A/A(0) - 1, D/E_bdy = (A - B)/(2A), B/A,
C/A = -B/A, M_clock/E = (1 + B/A)/2, N_sh (the lapse at the median radius), the record's thin-shell identity B/A - <N(1 + 2 u^2)>_{nu W/psi}
(markers_N12u2_weighted), the markers' unweighted mean of u_r^2 (u2), N_min over the nodes, the first node with chi <= 0, the largest row
residual (M7), the barrier rate r(T) = -d log N_min/dT and its integral; at the recorded coordinate passage r_sh = A (interpolated): N_sh,
C/A, M_clock/E, u2, N_* = 1/(3 + 2 u2), against the chi = 3/4 references 2/9, -5/9, 7/9 and the coherent benchmark 225/997, -547/997,
(1 + 547/997)/2, chi = 0.716; the E-ladder spread over E_0 = 8, 16, 32.  From the post-processor (self E16): N_s, chi_kin (the nodal S1-weighted
mean of v^2, I3), N_* = 1/(3 + 2 chi_kin), the shell identity, P07's residual, sigma_min(J), sigma_min(J_C), F_disc (nodal, markers) along the
run and at the passage.  The dt ladder (dt, dt/2, dt/4 of self E16): log E_bdy's drift at the common times and at the passage, the observed
order and the Richardson value with that order.  Args: --runs tag=path,... --dt-ladder path1,path2,path3 --post path --out-json --md."""
import argparse
import json
import math

import numpy as np

COH = {"N": 225 / 997, "C_over_A": -547 / 997, "Mclock_over_E": (1 + 547 / 997) / 2, "chi": 0.716}
REF = {"N": 2 / 9, "C_over_A": -5 / 9, "Mclock_over_E": 7 / 9}


def f(v, d=5):
    if v is None:
        return "-"
    v = float(v)
    return "%.*g" % (d, v) if (v == 0 or 1e-3 <= abs(v) < 1e5) else "%.3e" % v


def series(rec):
    R = rec["run"]["records"]
    T = np.array([q["T_over_rs"] for q in R])
    g = lambda k: np.array([q.get(k) if q.get(k) is not None else np.nan for q in R], float)
    A, B = g("A"), g("B")
    out = {"T": T, "A": A, "B": B, "logE": np.log(2 * A), "A_rel": A / A[0] - 1, "D_over_E": (A - B) / (2 * A), "B_over_A": B / A, "C_over_A": -B / A,
           "Mclock_over_E": (1 + B / A) / 2, "N_sh": g("N_sh"), "ident": B / A - g("markers_N12u2_weighted"), "u2": g("u2"), "N_min": g("N_min_nodes"),
           "chi_neg_r": g("first_chi_nonpositive_node_r"), "residual": g("residual")}
    with np.errstate(all="ignore"):
        r = -np.gradient(np.log(out["N_min"]), T)
    out["barrier_rate"] = r
    out["barrier_int"] = np.concatenate([[0.0], np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(T))])
    return out


def at(T, y, t):
    return float(np.interp(t, T, y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--dt-ladder", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    runs = dict(x.split("=", 1) for x in a.runs.split(","))
    out = {"runs": {}, "throat": {}}
    L = ["# Task 2, job T2a: the static-row diagnostics from the stored records (task2_report_v1.py; SPEC_task2_static_row_diagnostics.md sec. 2)", ""]
    th_rows = []
    for tag, path in runs.items():
        rec = json.load(open(path))
        s = series(rec)
        tc = rec["run"].get("throat_crossing")
        Tp = tc["T_over_rs"] if tc else None
        thr = None
        if Tp is not None:
            thr = {k: at(s["T"], s[k], Tp) for k in ("N_sh", "C_over_A", "Mclock_over_E", "u2", "B_over_A", "D_over_E", "ident", "logE", "A_rel", "N_min")}
            thr["N_star_u2"] = 1 / (3 + 2 * thr["u2"])
            thr["T_over_rs"] = Tp
        summ = {"outputs": int(len(s["T"])), "T_end": float(s["T"][-1]), "max_abs_A_rel": float(np.nanmax(np.abs(s["A_rel"]))),
                "logE_drift_end": float(s["logE"][-1] - s["logE"][0]), "max_abs_ident": float(np.nanmax(np.abs(s["ident"]))),
                "max_residual": float(np.nanmax(s["residual"])), "N_min_min": float(np.nanmin(s["N_min"])), "N_min_end": float(s["N_min"][-1]),
                "barrier_int_end": float(s["barrier_int"][-1]), "chi_nonpos_any": bool(np.any(np.isfinite(s["chi_neg_r"]))),
                "E0": rec["meta"].get("E0"), "dt_over_rs": rec["meta"].get("dt_over_rs"), "R_max_over_rs": rec["meta"].get("Rmax_over_rs")}
        out["runs"][tag] = {"summary": summ, "throat": thr}
        th_rows.append((tag, thr, summ))
    # the E-ladder spread at the passage
    eruns = [t for t in runs if t.startswith("E")]
    spread = {}
    for k in ("N_sh", "B_over_A", "C_over_A", "Mclock_over_E", "u2"):
        vals = [out["runs"][t]["throat"][k] for t in eruns if out["runs"][t]["throat"]]
        spread[k] = float(max(vals) - min(vals)) if vals else None
    out["E_ladder_spread"] = spread
    # post-processor (fields)
    P = json.load(open(a.post))
    ps = P["series"]
    PT = np.array([o["T_over_rs"] for o in ps])
    pth = P["throat"]
    post = {"throat": pth, "max_p07_residual": max(o["p07_residual"] for o in ps), "max_abs_shell_identity": max(abs(o["shell_identity"]) for o in ps),
            "sigma_min_J_min": min(o["sigma_min_J"] for o in ps), "sigma_min_JC_min": min(o["sigma_min_JC"] for o in ps),
            "N_min_min": min(o["N_min"] for o in ps), "barrier_int_end": ps[-1]["barrier_int"],
            "F_nodal_rel_max": max(abs(o["F_nodal_rel"]) for o in ps), "F_markers_rel_max": max(abs(o["F_markers_rel"]) for o in ps if o["F_markers_rel"] is not None),
            "F_nodal_defect_max": max(abs(o["F_nodal_defect"]) for o in ps),
            "dense_checks": [(o["T_over_rs"], o["sigma_min_J"], o.get("sigma_min_J_dense"), o["sigma_min_JC"], o.get("sigma_min_JC_dense")) for o in ps if "sigma_min_J_dense" in o]}
    out["post"] = post
    # dt ladder
    lad = [json.load(open(p)) for p in a.dt_ladder.split(",")]
    ls = [series(r) for r in lad]
    dts = [r["meta"].get("dt_over_rs") for r in lad]
    Tc = ls[0]["T"]
    common = [t for t in (5.0, 10.0, 20.0, 27.0, 40.0, 50.0, 60.0) if t <= min(x["T"][-1] for x in ls)]
    tcs = [r["run"].get("throat_crossing", {}).get("T_over_rs") for r in lad]
    ladrows = []
    for t in common + (["passage"] if all(tcs) else []):
        if t == "passage":
            v = [at(x["T"], x["logE"], tt) - x["logE"][0] for x, tt in zip(ls, tcs)]
        else:
            v = [at(x["T"], x["logE"], t) - x["logE"][0] for x in ls]
        d1, d2 = v[0] - v[1], v[1] - v[2]
        p = math.log(abs(d1 / d2), 2) if d2 != 0 and d1 / d2 > 0 else None
        rich = v[2] - d2 / (2 ** p - 1) if p else None
        ladrows.append({"T": t, "drift": v, "order": p, "richardson": rich, "resolution": abs(d2)})
    out["dt_ladder"] = {"dt_over_rs": dts, "rows": ladrows, "passage_T": tcs}
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    # markdown
    L += ["## Per run (the 0.1 r_s records)", "", "| run | E_0 | dt/r_s | outputs, T_end/r_s | max |A/A(0) - 1| | log E_bdy drift at the end | max |B/A - <N(1+2u^2)>| (record) | "
          "max row residual | min N_min | N_min at the end | int r dT at the end | a node with chi <= 0 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag, thr, sm in th_rows:
        L.append("| %s | %s | %s | %d, %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (tag, f(sm["E0"]), f(sm["dt_over_rs"]), sm["outputs"], f(sm["T_end"]),
                                                                         f(sm["max_abs_A_rel"], 3), f(sm["logE_drift_end"], 4), f(sm["max_abs_ident"], 3),
                                                                         f(sm["max_residual"], 2), f(sm["N_min_min"], 4), f(sm["N_min_end"], 4),
                                                                         f(sm["barrier_int_end"], 4), "yes" if sm["chi_nonpos_any"] else "no"))
    L += ["", "## The throat (the recorded coordinate passage r_sh = A, interpolated)", "",
          "| run | T/r_s | N_sh | u2 (markers, unweighted) | N_* = 1/(3 + 2 u2) | C/A | M_clock/E | B/A | D/E_bdy | identity residual | log E_bdy - log E_bdy(0) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag, thr, sm in th_rows:
        if thr:
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (tag, f(thr["T_over_rs"], 6), f(thr["N_sh"]), f(thr["u2"]), f(thr["N_star_u2"]),
                                                                       f(thr["C_over_A"]), f(thr["Mclock_over_E"]), f(thr["B_over_A"]), f(thr["D_over_E"]),
                                                                       f(thr["ident"], 3), f(thr["logE"] - math.log(2 * json.load(open(runs[tag]))["run"]["records"][0]["A"]), 3)))
    L += ["| the post-processor, self E16 (fields) | %s | N_s %s | chi_kin %s | %s | %s | %s | %s | %s | shell identity %s | - |" % (
        f(pth["T_over_rs"], 6), f(pth["N_s"]), f(pth["chi_kin"]), f(pth["N_star"]), f(pth["C_over_A"]), f(pth["Mclock_over_E"]), f(pth["B_over_A"]),
        f(pth["D_over_E"]), f(pth["shell_identity"], 3)),
          "| chi = 3/4 references | - | %s | 0.75 | %s | %s | %s | - | - | - | - |" % (f(REF["N"]), f(1 / 4.5), f(REF["C_over_A"]), f(REF["Mclock_over_E"])),
          "| the coherent benchmark (release from 4 r_s) | - | %s | %s | %s | %s | %s | %s | - | - | - |" % (
              f(COH["N"]), f(COH["chi"]), f(1 / (3 + 2 * COH["chi"])), f(COH["C_over_A"]), f(COH["Mclock_over_E"]), f(-COH["C_over_A"])),
          "", "- The E-ladder spread at the passage (E_0 = 8, 16, 32): %s." % ", ".join("%s %s" % (k, f(v, 3)) for k, v in spread.items()), "",
          "## The post-processor on self E16's stored fields (task2_postproc_v1.py)", "",
          "- Over the %d outputs: the largest P07 residual |N - N_p07|/|N| %s; the largest |shell identity| %s; the smallest sigma_min(J) %s and sigma_min(J_C) %s "
          "(inverse iteration; the dense svdvals at %d outputs: %s); the smallest N_min %s; int r dT at the end %s; |F_disc/F_disc(0) - 1| at most %s (nodal "
          "sum) and %s (markers' sum); |F_disc,nodal/(4 pi K A_n) - 1| at most %s." % (
              len(ps), f(post["max_p07_residual"], 3), f(post["max_abs_shell_identity"], 3), f(post["sigma_min_J_min"], 4), f(post["sigma_min_JC_min"], 4),
              len(post["dense_checks"]), "; ".join("T %s: %s/%s, %s/%s" % (f(x[0], 4), f(x[1], 10), f(x[2], 10), f(x[3], 10), f(x[4], 10)) for x in post["dense_checks"]),
              f(post["N_min_min"], 4), f(post["barrier_int_end"], 4), f(post["F_nodal_rel_max"], 3), f(post["F_markers_rel_max"], 3), f(post["F_nodal_defect_max"], 3)),
          "- Every 5 r_s: " + "; ".join("T %s: N_s %s, chi_kin %s, N_* %s, B/A %s, N_min %s, sigma_min(J) %s, F_nodal rel %s" % (
              f(o["T_over_rs"], 4), f(o["N_s"], 5), f(o["chi_kin"], 5), f(o["N_star"], 5), f(o["B_over_A"], 5), f(o["N_min"], 4), f(o["sigma_min_J"], 4),
              f(o["F_nodal_rel"], 3)) for o in ps if abs(o["T_over_rs"] / 5 - round(o["T_over_rs"] / 5)) < 1e-6) + ".", "",
          "## The dt ladder of self E16 (dt/r_s = %s): log E_bdy - log E_bdy(0)" % ", ".join(f(d) for d in dts), "",
          "| T/r_s | dt | dt/2 | dt/4 | observed order | Richardson with that order | |dt/2 - dt/4| |", "|---|---|---|---|---|---|---|"]
    for r in ladrows:
        L.append("| %s | %s | %s | %s | %s | %s | %s |" % (r["T"] if r["T"] == "passage" else f(r["T"]), f(r["drift"][0], 4), f(r["drift"][1], 4), f(r["drift"][2], 4),
                                                       f(r["order"], 3), f(r["richardson"], 3), f(r["resolution"], 3)))
    L += ["", "- Every number above is emitted by the script; u2 is the markers' unweighted mean of u_r^2 (the records carry no W-weighted mean); the "
          "field-based chi_kin is the nodal S1-weighted mean of v^2 (I3)."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
