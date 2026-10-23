#!/usr/bin/env python
"""Strand 7 S7a part 2 report, version 2 (SPEC_strand7_collapse.md secs. 10.2 (b), (e), (f) and 11.3; builder analyst).  Reads the S7a JSONs
and writes markdown and a JSON (with "sensitivities_for_item7", which strand7_page_report_v2.py prints in T7p7):
  (a) sec. 11.3 (a), the discrete energy: F_disc(T) := -K sum_edges c_i (u_{i+1} - u_i)(v_{i+1} - v_i) + 4 pi K A_n B_n/R_max + sum nu N W,
      on the shell code's grid (c_i the finite-volume coefficients of its Laplacian, K = 1/(2 pi)), u = psi - 1 and v = chi - 1 from the stored
      fields, A_n = R_max u_n and B_n = -R_max v_n, the boundary term 4 pi K A_n B_n/R_max the exterior form's own contribution beyond the box
      (exact for the 1/r tails: -K Int_R^inf 4 pi r^2 psi' chi' dr), and sum nu N W the record's sum_nu_H (the markers' sum); against
      4 pi K A_tail(T).  By summation by parts with the discrete clock row K L u = -S1/psi (S1 the cloud-in-cell deposit of nu W), F_disc =
      4 pi K A_gauss + (sum_a nu_a N(r_a) W_a - sum_i N_i S1_i) + the residual: the defect is the gather/deposit mismatch.  The calibration on
      C1's static TF member (the shell code's static check rebuilt here: the J1 member at z_c, the quadrature markers, the re-solved fields)
      prints F_disc - 4 pi K A with the markers' sum and with the nodal sum sum_i N_i S1_i (the discrete identity), so the boundary term is
      fixed first; along the run with stored fields (self E16): the per-output defect over 4 pi K A_tail, F_disc/F_disc(0) - 1 beside
      A/A(0) - 1, and the lagged-source residual;
  (b) sec. 11.3 (b), the dt ladder (dt, dt/2, dt/4; the dt/4 rung optional until it lands): at the common outputs A/A(0) - 1, B/A, r_sh/A,
      c_+^(r), N(r_sh), R_sh/4A, the differences between rungs and the observed order p = log2(|q_dt - q_dt/2|/|q_dt/2 - q_dt/4|) at every 5 r_s
      and its median over the outputs where both differences exceed 1e-12; the same at the matched event (the passage record, A/A(0) - 1
      interpolated in T); T_valid := the last output up to which |(A/A(0))_dt - (A/A(0))_dt/2| and |(r_sh/A)_dt - (r_sh/A)_dt/2| both stay below
      0.01 (the validity boundary of C's fields for item 7), and beside it (the builder's addition,
      labelled) the same with the r_sh/A difference relative to (r_sh/A)_dt/2 and with the charge alone;
  (c) sec. 11.3 (c), the E-ladder's spread at the passage (E8, E16, E32; not a pure refinement: w_r/r_s0 changes), reported as such;
  (e) the moved exterior (base:moved), as version 1;
  (f) the record checks, as version 1 (the records hold the last step's convergence flag per output only); C1 and C2 lines as version 1.
Arguments: --ladder (two or three S7a JSONs: dt, dt/2[, dt/4]), --fields (the stored fields of the ladder's base run), --e-ladder (comma list),
--rmax base:moved, --runs, --static, --static-zc (the C1 member's z_c for the calibration), --frozen-pair, --md, --out-json."""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, "..", "strand5"))
import strand7_shell_v4 as SH  # noqa: E402

QK = ("A_rel", "B_over_A", "r_sh_over_A", "c_plus_r", "N_sh", "R_sh_over_4A")


def f(x, d=5):
    if x is None:
        return "-"
    x = float(x)
    if not math.isfinite(x):
        return str(x)
    if x == 0:
        return "0"
    return ("%." + str(d) + "g") % x if 1e-3 <= abs(x) < 1e5 else "%.3e" % x


def load(p):
    with open(p) as fh:
        return json.load(fh)


def qv(rec, k):
    if k == "R_sh_over_4A":
        return rec["R_sh"] / (4 * rec["A"])
    return rec[k]


def crossing(R):
    tc = R["run"].get("throat_crossing")
    if not tc:
        return None
    recs = R["run"]["records"]
    T = np.array([q["T_over_rs"] for q in recs])
    A = np.array([q["A_rel"] for q in recs])
    return {"T": tc["T_over_rs"], "A_rel": float(np.interp(tc["T_over_rs"], T, A)), "B_over_A": tc["B_over_A"], "N": tc["N_sh"], "c_plus": tc["c_plus_r"],
            "R_sh_over_rs": tc.get("R_sh_over_rs")}


def fdisc_fields(G, u, v, sum_nu_H):
    c = G.L.diagonal(1)
    grad = float(np.sum(c * np.diff(u) * np.diff(v)))
    An, Bn = G.Rmax * u[-1], -G.Rmax * v[-1]
    return -SH.K * grad + 4 * np.pi * SH.K * An * Bn / G.Rmax + sum_nu_H, An, Bn


def static_calibration(zc, static_nodes=200, static_rmax=4.0, static_p=12, static_c=12):
    """C1's static TF member rebuilt as in the shell code's static_check, then F_disc with the markers' sum and with the nodal sum."""
    import strand5_star_v4 as STAR
    g = 16.0
    ref = STAR.shoot(zc, g, keep=True)
    rs_ = float(ref["r_s"])
    hh = rs_ / static_nodes
    G = SH.UGrid(int(round(static_rmax * rs_ / hh)) + 1, hh)
    psi, chi = STAR.shoot_profile(ref, np.maximum(G.r, 1e-12))
    N = chi / psi
    mu_inf = float(ref["mu_inf"])
    xr, wr_ = np.polynomial.legendre.leggauss(4)
    xp, wp = np.polynomial.legendre.leggauss(static_p)
    xc, wc = np.polynomial.legendre.leggauss(static_c)
    edges = np.concatenate([[0.0], 0.5 * (G.r[:-1] + G.r[1:]), [G.r[-1]]])
    cells = np.nonzero(edges[:-1] < rs_)[0]
    rq = np.concatenate([0.5 * (edges[i + 1] - edges[i]) * xr + 0.5 * (edges[i + 1] + edges[i]) for i in cells])
    wq = np.concatenate([0.5 * (edges[i + 1] - edges[i]) * wr_ for i in cells])
    psq, chq = STAR.shoot_profile(ref, rq)
    Nq = chq / psq
    kfq, _, eq, Pq, _, _ = STAR.eos(mu_inf / Nq, g)
    keep = kfq > 0
    rq, wq, psq, Nq, kfq = (x[keep] for x in (rq, wq, psq, Nq, kfq))
    vol = 4 * np.pi * rq * rq * wq
    pk = 0.5 * kfq[:, None] * (xp[None, :] + 1)
    wpk = 0.5 * kfq[:, None] * wp[None, :]
    base = vol[:, None, None] * psq[:, None, None] ** 6 * g / (2 * np.pi) ** 3 * 2 * np.pi * (pk * pk * wpk)[:, :, None] * wc[None, None, :]
    Pc = (psq[:, None] ** 2 * pk)[:, :, None] * np.ones_like(xc)[None, None, :]
    R_ = np.broadcast_to(rq[:, None, None], base.shape).ravel()
    P_ = (Pc * xc[None, None, :]).ravel()
    J_ = (rq[:, None, None] * Pc * np.sqrt(1 - xc * xc)[None, None, :]).ravel()
    NU_ = base.ravel()
    u, v, info = SH.quasi_static(G, R_, P_, J_, NU_, psi - 1, chi - 1)
    fi = SH.FieldInterp(G, u, v)
    _, _, Nm, _, _, _, Wm = SH.marker_terms(fi, R_, P_, J_)
    S1, S2 = SH.sources(G, fi, R_, P_, J_, NU_)
    Nn = (1 + v) / (1 + u)
    marker_sum = float(np.sum(NU_ * Nm * Wm))
    nodal_sum = float(np.sum(Nn * S1))
    Fm, An, Bn = fdisc_fields(G, u, v, marker_sum)
    Fn, _, _ = fdisc_fields(G, u, v, nodal_sum)
    E = 4 * np.pi * SH.K * An
    return {"z_c": zc, "n": G.n, "h": hh, "A_n": An, "A_gauss": info["A_gauss"], "B_n": Bn, "4piKA": E, "F_disc_markers": Fm, "F_disc_nodal": Fn,
            "rel_markers": Fm / E - 1, "rel_nodal": Fn / E - 1, "residual": info["residual"], "outer_converged": info["outer_converged"],
            "markers": int(len(R_)), "gather_deposit_mismatch_rel": (marker_sum - nodal_sum) / E}


def ladder_tables(lad):
    """the dt ladder at the common outputs and at the passage."""
    recs = [{round(q["T_over_rs"], 6): q for q in R["run"]["records"]} for R in lad]
    common = sorted(set.intersection(*[set(r) for r in recs]))
    rows, pmed = [], {}
    diffs = {k: [] for k in QK}
    for T in common:
        row = {"T": T}
        for k in QK:
            vals = [qv(r[T], k) for r in recs]
            d1 = vals[0] - vals[1]
            d2 = (vals[1] - vals[2]) if len(vals) > 2 else None
            p = math.log2(abs(d1) / abs(d2)) if (d2 is not None and abs(d1) > 1e-12 and abs(d2) > 1e-12) else None
            row[k] = {"values": vals, "d1": d1, "d2": d2, "p": p}
            diffs[k].append((T, d1, d2, p))
        rows.append(row)
    for k in QK:
        ps = [x[3] for x in diffs[k] if x[3] is not None]
        pmed[k] = float(np.median(ps)) if ps else None
    # T_valid (sec. 11.3 (b), as written) and two labelled variants (the builder's addition)
    def last_ok(cond):
        out = None
        for row in rows:
            if cond(row):
                out = row["T"]
            else:
                break
        return out
    Tv = last_ok(lambda w: abs(w["A_rel"]["d1"]) < 0.01 and abs(w["r_sh_over_A"]["d1"]) < 0.01)
    Tv_rel = last_ok(lambda w: abs(w["A_rel"]["d1"]) < 0.01 and abs(w["r_sh_over_A"]["d1"]) < 0.01 * abs(w["r_sh_over_A"]["values"][1]))
    Tv_A = last_ok(lambda w: abs(w["A_rel"]["d1"]) < 0.01)
    return common, rows, pmed, (Tv, Tv_rel, Tv_A), diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True, help="S7a JSONs: dt, dt/2[, dt/4]")
    ap.add_argument("--fields", default=None, help="the stored fields of the ladder's base run (for F_disc along the run)")
    ap.add_argument("--e-ladder", default="", help="comma list of S7a JSONs (E8, E16, E32)")
    ap.add_argument("--rmax", default="", help="base:moved (R_max 64 and 128 r_s)")
    ap.add_argument("--runs", default="", help="comma list of S7a JSONs for the record checks (f)")
    ap.add_argument("--static", default="", help="comma list of static-check JSONs (revision 1)")
    ap.add_argument("--static-zc", type=float, default=None, help="z_c of C1's static member for the calibration of (a)")
    ap.add_argument("--frozen-pair", default="", help="dop853:verlet frozen_schw JSONs (revision 1)")
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    L = ["# Strand 7 S7a part 2: the discrete energy, the dt ladder with T_valid, the E-ladder, the moved exterior, the record checks "
         "(strand7_part2_report_v2.py)", ""]
    js = {}
    sens = {}
    lad = [load(p) for p in a.ladder.split(",")]
    dts = [R["meta"]["dt_over_rs"] for R in lad]
    # (a) the calibration and F_disc along the run
    if a.static_zc is not None:
        cal = static_calibration(a.static_zc)
        js["static_calibration"] = cal
        L += ["## (a) The discrete energy F_disc: the boundary term calibrated on C1's static TF member (z_c = %s)" % f(a.static_zc, 8), "",
              "- On the static member (n = %d, h = %s r_s units of the member, %d quadrature markers, the re-solved fields; outer loop converged: %s, residual %s): "
              "F_disc/(4 pi K A) - 1 = %s with the nodal sum sum_i N_i S1_i (the discrete identity; the boundary term 4 pi K A_n B_n/R_max) and %s with the "
              "markers' sum sum nu N W; the gather/deposit mismatch (sum nu N W - sum N_i S1_i)/(4 pi K A) = %s; A_n/A_gauss - 1 = %s." % (
                  cal["n"], f(cal["h"], 4), cal["markers"], cal["outer_converged"], f(cal["residual"], 3), f(cal["rel_nodal"], 3), f(cal["rel_markers"], 3),
                  f(cal["gather_deposit_mismatch_rel"], 3), f(cal["A_n"] / cal["A_gauss"] - 1, 3)), ""]
    if a.fields and os.path.exists(a.fields):
        z = np.load(a.fields)
        r, Tz = z["r"], z["T"]
        G = SH.UGrid(len(r), float(r[1] - r[0]))
        byT = {round(float(q["T"]), 6): q for q in lad[0]["run"]["records"]}
        rs = lad[0]["meta"]["r_s"]
        fr = []
        for j in range(len(Tz)):
            q = byT.get(round(float(Tz[j]), 6))
            if q is None:
                continue
            F, An, Bn = fdisc_fields(G, z["u"][j], z["v"][j], q["sum_nu_H"])
            E = 4 * np.pi * SH.K * q["A_tail"]
            fr.append({"T_over_rs": float(Tz[j] / rs), "F_disc": F, "defect_rel": F / E - 1, "A_rel": q["A_rel"], "residual": q.get("residual"),
                       "A_n_over_A_tail": An / q["A_tail"] - 1})
        F0 = fr[0]["F_disc"]
        for x in fr:
            x["F_rel"] = x["F_disc"] / F0 - 1
        js["F_disc"] = fr
        sel = [x for x in fr if abs(x["T_over_rs"] / 5 - round(x["T_over_rs"] / 5)) < 1e-6]
        L += ["## (a) F_disc along %s (stored fields; every 5 r_s)" % lad[0]["meta"]["tag"], "",
              "- Over the %d outputs: the largest |defect|/(4 pi K A_tail) %s; the largest |F_disc/F_disc(0) - 1| %s against the largest |A/A(0) - 1| %s; "
              "the largest lagged-source residual %s." % (len(fr), f(max(abs(x["defect_rel"]) for x in fr), 3), f(max(abs(x["F_rel"]) for x in fr), 3),
                                                          f(max(abs(x["A_rel"]) for x in fr), 3), f(max(x["residual"] or 0 for x in fr), 3)), "",
              "| T/r_s | (F_disc - 4 pi K A_tail)/(4 pi K A_tail) | F_disc/F_disc(0) - 1 | A/A(0) - 1 | residual |", "|---|---|---|---|---|"] + [
            "| %s | %s | %s | %s | %s |" % (f(x["T_over_rs"], 4), f(x["defect_rel"], 3), f(x["F_rel"], 3), f(x["A_rel"], 3), f(x["residual"], 3)) for x in sel] + [""]
        sens["F_disc"] = {"max_abs_defect_rel": max(abs(x["defect_rel"]) for x in fr), "max_abs_F_rel": max(abs(x["F_rel"]) for x in fr)}
    # (b) the ladder
    common, rows, pmed, (Tv, Tv_rel, Tv_A), diffs = ladder_tables(lad)
    cr = [crossing(R) for R in lad]
    L += ["## (b) The dt ladder at E_0 = 16 (self): dt = %s r_s at the common outputs (%d)" % (", ".join(f(x, 3) for x in dts), len(common)), "",
          "- T_valid (the last output up to which |(A/A(0))_dt - (A/A(0))_dt/2| and |(r_sh/A)_dt - (r_sh/A)_dt/2| both stay below 0.01): %s r_s. "
          "Beside it (the builder's addition, labelled): with the r_sh/A difference relative to (r_sh/A)_dt/2, %s r_s; with the charge alone, %s r_s." % (
              f(Tv, 5), f(Tv_rel, 5), f(Tv_A, 5)),
          "- The median observed order over the outputs with both differences above 1e-12: " + (", ".join("%s %s" % (k, f(v, 3)) for k, v in pmed.items())
                                                                                              if len(lad) > 2 else "the dt/4 rung not yet given") + ".", "",
          "| T/r_s | " + " | ".join("%s: dt - dt/2, dt/2 - dt/4 (p)" % k for k in QK) + " |", "|---|" + "---|" * len(QK)]
    for row in rows:
        if abs(row["T"] / 5 - round(row["T"] / 5)) < 1e-6:
            L.append("| %s | " % f(row["T"], 4) + " | ".join("%s, %s (%s)" % (f(row[k]["d1"], 3), f(row[k]["d2"], 3), f(row[k]["p"], 3)) for k in QK) + " |")
    L.append("")
    orders = {}
    prow = []
    for k in ("T", "A_rel", "B_over_A", "N", "c_plus"):
        v = [c[k] if c else None for c in cr]
        d1 = (v[0] - v[1]) if None not in v[:2] else None
        d2 = (v[1] - v[2]) if (len(v) > 2 and None not in v[1:]) else None
        p = math.log2(abs(d1) / abs(d2)) if (d1 and d2) else None
        orders[k] = {"values": v, "d1": d1, "d2": d2, "p": p}
        prow.append("| %s | %s | %s | %s | %s |" % (k, ", ".join(f(x, 8) for x in v), f(d1, 3), f(d2, 3), f(p, 3)))
    L += ["### At the matched event (the passage record r_sh = A)", "", "| quantity | dt, dt/2, dt/4 | dt - dt/2 | dt/2 - dt/4 | observed order |",
          "|---|---|---|---|---|"] + prow + [""]
    big = {k: {"max_abs_dt_minus_dt2_to_T_valid": max((abs(x[1]) for x in diffs[k] if Tv is not None and x[0] <= Tv), default=None),
               "max_abs_dt_minus_dt2_over_run": max(abs(x[1]) for x in diffs[k])} for k in ("A_rel", "B_over_A", "r_sh_over_A")}
    js["ladder"] = {"runs": a.ladder.split(","), "dt_over_rs": dts, "T_valid_over_rs": Tv, "T_valid_relative_r_sh_over_rs": Tv_rel, "T_valid_charge_only_over_rs": Tv_A,
                    "median_order": pmed, "at_passage": orders, "common_outputs": len(common), "spreads": big}
    sens.update({"T_valid_over_rs0": Tv, "T_valid variants (the builder's addition): r_sh/A relative, the charge alone": [Tv_rel, Tv_A],
                 "dt_ladder_spread_common (|q_dt - q_dt/2|, to T_valid and over the run)": big,
                 "dt_ladder_at_passage (dt - dt/2)": {k: orders[k]["d1"] for k in orders}})
    # (c) the E-ladder
    if a.e_ladder:
        el = [load(p) for p in a.e_ladder.split(",")]
        ce = [(R["meta"]["E0"], crossing(R)) for R in el]
        spr = {}
        for k in ("T", "A_rel", "B_over_A", "N", "c_plus"):
            vals = [c[k] for _, c in ce if c]
            spr[k] = (max(vals) - min(vals)) if vals else None
        js["E_ladder"] = {"crossings": {str(E): c for E, c in ce}, "spread": spr}
        sens["E_ladder_spread_at_passage (not a pure refinement)"] = spr
        L += ["## (c) The E-ladder at the passage (E_0 = %s; w_r/r_s0 changes with E_0: not a pure refinement)" % ", ".join(f(E, 3) for E, _ in ce), "",
              "| E_0 | T/r_s0 | A/A(0) - 1 | B/A | N(r_sh) | c_+^(r) |", "|---|---|---|---|---|---|"] + [
            "| %s | %s | %s | %s | %s | %s |" % (f(E, 3), f(c["T"], 6), f(c["A_rel"], 4), f(c["B_over_A"], 6), f(c["N"], 6), f(c["c_plus"], 6)) for E, c in ce if c] + [
            "", "- The spread (largest minus smallest): " + ", ".join("%s %s" % (k, f(v, 3)) for k, v in spr.items()) + ".", ""]
    if a.rmax:
        b, m = (load(p) for p in a.rmax.split(":"))
        tb = {round(q["T_over_rs"], 6): q for q in b["run"]["records"]}
        tm = {round(q["T_over_rs"], 6): q for q in m["run"]["records"]}
        common_r = sorted(t for t in tb if t in tm)
        cb, cm = crossing(b), crossing(m)
        Tc = cb["T"] if cb else np.inf
        dd = {}
        for k in ("A", "B", "B_over_A", "r_sh"):
            pre = [abs(tm[t][k] / tb[t][k] - 1) for t in common_r if t <= Tc]
            post = [abs(tm[t][k] / tb[t][k] - 1) for t in common_r if t > Tc]
            dd[k] = {"before": max(pre) if pre else None, "after": max(post) if post else None}
        js["rmax"] = {"diffs": dd, "crossing_base": cb, "crossing_moved": cm, "R_max": [b["meta"]["Rmax_over_rs"], m["meta"]["Rmax_over_rs"]]}
        L += ["## (e) The exterior moved: R_max = %s r_s against %s r_s (largest relative differences at the %d common outputs)" % (
            f(m["meta"]["Rmax_over_rs"], 4), f(b["meta"]["Rmax_over_rs"], 4), len(common_r)), "",
              "| quantity | before the passage | after |", "|---|---|---|"] + ["| %s | %s | %s |" % (k, f(v["before"], 3), f(v["after"], 3)) for k, v in dd.items()] + [
            "", "- The passage records: base %s; moved %s." % (json.dumps(cb, default=float), json.dumps(cm, default=float)), ""]
    if a.runs:
        rows_ = []
        js["records"] = {}
        for p in a.runs.split(","):
            R = load(p)
            recs = R["run"]["records"]
            nc = sum(1 for q in recs if q.get("outer_converged") is False)
            mr = max(q["residual"] for q in recs if q.get("residual") is not None)
            js["records"][R["meta"]["tag"]] = {"outputs": len(recs), "not_converged": nc, "max_residual": mr}
            rows_.append("| %s | %d | %d | %s |" % (R["meta"]["tag"], len(recs), nc, f(mr, 3)))
        L += ["## (f) Record checks: the outer loop's convergence flag per output (the last step's flag only) and the largest lagged-source residual", "",
              "| run | outputs | outputs not converged | largest residual |", "|---|---|---|---|"] + rows_ + [""]
    if a.static:
        for p in a.static.split(","):
            S = load(p)["static_check"]
            L.append("- C1 static check %s: resolve_A_rel %s, resolve_B_rel %s, resolve outer/newton/residual %s." % (
                p.split("/")[-1], f(S["resolve_A_rel"], 3), f(S["resolve_B_rel"], 3), json.dumps(S["resolve_info"], default=float)))
        L.append("")
    if a.frozen_pair:
        d, v = (load(p) for p in a.frozen_pair.split(":"))
        md = max(abs(q["sum_nu_H_rel"]) for q in d["run"]["records"] if q.get("sum_nu_H_rel") is not None)
        mv = max(abs(q["sum_nu_H_rel"]) for q in v["run"]["records"] if q.get("sum_nu_H_rel") is not None)
        L += ["- C2 energy: the largest |sum nu H rel| of frozen_schw by DOP853 %s and by the run's own stepper %s (the spec's level 1e-8)." % (f(md, 3), f(mv, 3)), ""]
        js["C2"] = {"dop853": md, "stepper": mv}
    js["sensitivities_for_item7"] = sens
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump(js, open(a.out_json, "w"), indent=1, default=float)
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
