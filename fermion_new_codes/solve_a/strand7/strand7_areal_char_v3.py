#!/usr/bin/env python
"""Strand 7 S7a part 2 (d), version 2: the areal characteristics from the stored fields of a self-consistent run, with the terms kept apart,
the time differencing controlled, the material quantiles, the inward branch at each candidate zero and short rays (SPEC_strand7_collapse.md
sec. 11.2; builder analyst).  From <run>_fields.npz (u = psi - 1, v = chi - 1 on the nodes at every output) and the run's JSON (the
quantiles r_p01, r_sh, r_p99 and the charge A per output):
  (a) per node and output t_1 := 2 r psi psi_dot and t_2 := (psi^2 + 2 r psi psi') N psi^-2 (N = chi/psi, psi' from the cubic spline in r with
      zero slope at r = 0 and not-a-knot at the wall, the S7a code's interpolation), c_+^(R) = t_1 + t_2, c_-^(R) = t_1 - t_2; per output the
      count of nodes with both branches negative (the trapped pattern), split into kink nodes (r in [r_p01 - h, r_p99 + h], within one cell of
      the deposited source) and the others;
  (b) psi_dot by three estimates over the outputs: the three-point central difference (version 1's), the five-point central stencil, and the
      backward difference (the run's monitor's form), the outputs 0.1 r_s apart (the dt/2 rung's fields are not stored); at each evaluation
      point the spread (largest minus smallest) of c_+ and of c_- over the three is printed as the differencing error; the both-negative
      counts per estimate;
  (c) evaluation points: the material quantiles r_p01, r_sh, r_p99 with psi' one-sided (a cubic spline through the nodes on one side of r_sh
      only, inner and outer, each evaluated at the point; with fewer than 4 inner nodes the inner side is the two-node difference, with fewer
      than 2 it is absent), and the source-free points r = A, r = (r_p99 + A)/2 when r_p99 < A (after the passage), r = 2A and r = r_p01/2;
  (d) at each candidate zero of c_+^(R) on the nodes (central estimate; a sign change between neighbouring nodes, the zero by linear
      interpolation): c_-^(R) there, the signs of c_+ on the two neighbouring nodes, whether both branches are negative on the inner neighbour,
      and the distance in cells to the nearest kink node (0 inside the kink zone);
  (e) short rays: from r = 2A at T = 20, 26, 28, 34, 40, 50 r_s (the lead's list) and, labelled as the builder's addition, from the fold's
      midpoint (r_p99 + A)/2 at the same times where it exists, both branches dr/dT = +-N psi^-2 on the fields (cubic in r, linear in T between
      outputs) for 4 r_s, R(T) = r psi^2 recorded; per ray the change of R, whether R decreases on the outgoing branch at any step outside the
      kink zone (the fold's signature) and where.
Writes --md (the summary, the table at every r_s and at every output in [26.5, 29.5] r_s, the candidate zeros at those outputs, the rays) and
--out-json (every output).
Version 3 (builder's addition to sec. 11.2 (a), labelled): every node outside the kink zone with both branches negative is listed with r/A, t_1,
t_2, psi and psi_dot r/A_dot (A_dot the central difference of the run's A over the neighbouring outputs; in the vacuum form outside the
material psi = 1 + A/r, so psi_dot r/A_dot = 1 there when the field's change at fixed r is the charge's change), and the md gets one row per
episode (consecutive outputs with such nodes): the T range, the nodes' r/A range and the ranges of t_1, t_2, psi_dot r/A_dot and A_dot."""
import argparse
import json
import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline


def spl(r, y):
    return CubicSpline(r, y, bc_type=((1, 0.0), "not-a-knot"))


def one_sided(r, u, rsh, side):
    """psi' at a point from the nodes on one side of r_sh only: a cubic spline (natural ends) with >= 4 nodes, the two-node difference with
    2 or 3, None with fewer."""
    sel = r < rsh if side == "inner" else r > rsh
    rr, uu = r[sel], u[sel]
    if side == "outer":
        rr, uu = rr[:400], uu[:400]
    if len(rr) >= 4:
        s = CubicSpline(rr, uu, bc_type="natural")
        return lambda x: float(s(x, 1))
    if len(rr) >= 2:
        i0, i1 = (-2, -1) if side == "inner" else (0, 1)
        d = (uu[i1] - uu[i0]) / (rr[i1] - rr[i0])
        return lambda x: float(d)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--ray-times", default="20,26,28,34,40,50")
    ap.add_argument("--ray-length", type=float, default=4.0, help="in r_s")
    a = ap.parse_args()
    z = np.load(a.fields)
    R = json.load(open(a.json))
    rs = R["meta"]["r_s"]
    h = R["meta"]["h"]
    recs = {round(q["T"], 6): q for q in R["run"]["records"]}
    r, T, U, V = z["r"], z["T"], z["u"], z["v"]
    nT = len(T)
    dTs = np.diff(T)
    uniform = bool(np.max(np.abs(dTs - dTs[0])) < 1e-9 * abs(dTs[0]))
    nodes_mask = r >= h
    rn = r[nodes_mask]

    def psidots(k):
        """the three estimates of psi_dot on all nodes at output k (None where the stencil does not fit)."""
        out = {}
        k0, k1 = max(k - 1, 0), min(k + 1, nT - 1)
        out["c3"] = (U[k1] - U[k0]) / (T[k1] - T[k0])
        out["c5"] = ((-U[k + 2] + 8 * U[k + 1] - 8 * U[k - 1] + U[k - 2]) / (12 * dTs[0])) if (uniform and 2 <= k <= nT - 3) else None
        out["bw"] = ((U[k] - U[k - 1]) / (T[k] - T[k - 1])) if k >= 1 else None
        return out

    rows = []
    Arec = np.array([recs[round(float(t), 6)]["A"] if round(float(t), 6) in recs else np.nan for t in T])
    for k in range(nT):
        q = recs.get(round(float(T[k]), 6))
        if q is None:
            continue
        A, rsh, p01, p99 = q["A"], q["r_sh"], q["r_p01"], q["r_p99"]
        k0_, k1_ = max(k - 1, 0), min(k + 1, nT - 1)
        Adot = float((Arec[k1_] - Arec[k0_]) / (T[k1_] - T[k0_]))
        su, sv = spl(r, U[k]), spl(r, V[k])
        pd = psidots(k)
        sdot = {key: (spl(r, val) if val is not None else None) for key, val in pd.items()}
        kink = (rn >= p01 - h) & (rn <= p99 + h)
        ps_n = 1 + U[k][nodes_mask]
        ch_n = 1 + V[k][nodes_mask]
        N_n = ch_n / ps_n
        t2_n = (ps_n * ps_n + 2 * rn * ps_n * su(rn, 1)) * N_n / ps_n ** 2
        counts = {}
        cp_c3 = cm_c3 = None
        for key, val in pd.items():
            if val is None:
                counts[key] = None
                continue
            t1_n = 2 * rn * ps_n * val[nodes_mask]
            cp, cm = t1_n + t2_n, t1_n - t2_n
            both = (cp < 0) & (cm < 0)
            counts[key] = {"both_negative_nonkink": int(np.sum(both & ~kink)), "both_negative_kink": int(np.sum(both & kink)),
                           "c_plus_nonpositive": int(np.sum(cp <= 0))}
            if key == "c3":
                cp_c3, cm_c3, t1_c3 = cp, cm, t1_n
                bnodes = [{"r_over_A": float(rn[i] / A), "t1": float(t1_n[i]), "t2": float(t2_n[i]), "psi": float(ps_n[i]),
                           "psidot_r_over_Adot": (float(val[nodes_mask][i] * rn[i] / Adot) if Adot != 0 else None)}
                          for i in np.nonzero(both & ~kink)[0]]

        def at_point(x, dpsi):
            """c_+-, t_1, t_2 at r = x with the three psi_dot estimates and the given psi'(x)."""
            ps = 1 + float(su(x))
            N = (1 + float(sv(x))) / ps
            t2 = (ps * ps + 2 * x * ps * dpsi) * N / ps ** 2
            res = {}
            for key, s in sdot.items():
                if s is None:
                    continue
                t1 = 2 * x * ps * float(s(x))
                res[key] = (t1 + t2, t1 - t2, t1, t2)
            cps = [v[0] for v in res.values()]
            cms = [v[1] for v in res.values()]
            return {"c_plus": res["c3"][0], "c_minus": res["c3"][1], "t1": res["c3"][2], "t2": res["c3"][3],
                    "spread_c_plus": max(cps) - min(cps), "spread_c_minus": max(cms) - min(cms), "estimates": len(res)}
        pts = {}
        for name, x in (("r_p01", p01), ("r_sh", rsh), ("r_p99", p99)):
            for side in ("inner", "outer"):
                d = one_sided(r, U[k], rsh, side)
                if d is not None and x > 0:
                    pts["%s_%s" % (name, side)] = dict(at_point(x, d(x)), r_over_A=x / A)
        for name, x in (("A", A), ("mid_p99_A", 0.5 * (p99 + A) if p99 < A else None), ("2A", 2 * A), ("half_p01", 0.5 * p01)):
            if x is not None and x >= h:
                pts[name] = dict(at_point(x, float(su(x, 1))), r_over_A=x / A)
        zeros = []
        sg = np.sign(cp_c3)
        idx = np.nonzero(sg[:-1] * sg[1:] < 0)[0]
        kink_idx = np.nonzero(kink)[0]
        for i in idx:
            x0 = float(rn[i] - cp_c3[i] * (rn[i + 1] - rn[i]) / (cp_c3[i + 1] - cp_c3[i]))
            w = (x0 - rn[i]) / (rn[i + 1] - rn[i])
            cm0 = float((1 - w) * cm_c3[i] + w * cm_c3[i + 1])
            dist = 0.0 if (kink[i] or kink[i + 1]) else (float(np.min(np.abs(kink_idx - i))) if len(kink_idx) else None)
            zeros.append({"r": x0, "r_over_A": x0 / A, "c_minus": cm0, "c_plus_inner_node": float(cp_c3[i]), "c_plus_outer_node": float(cp_c3[i + 1]),
                          "both_negative_inner": bool(cp_c3[i] < 0 and cm_c3[i] < 0), "cells_to_kink": dist})
        rows.append({"k": k, "T_over_rs": float(T[k] / rs), "A": A, "r_sh_over_A": rsh / A, "r_p01_over_A": p01 / A, "r_p99_over_A": p99 / A,
                     "points": pts, "counts": counts, "zeros": zeros, "A_dot": Adot, "both_negative_nonkink_nodes": bnodes, "monitor_c_plus_R": q.get("c_plus_R"),
                     "min_c_plus_nonkink": float(np.min(cp_c3[~kink])) if np.any(~kink) else None,
                     "r_over_A_at_min_nonkink": float(rn[~kink][int(np.argmin(cp_c3[~kink]))] / A) if np.any(~kink) else None})
    # (e) rays
    Tq_list = [float(x) for x in a.ray_times.split(",") if x.strip()]
    rec_T = np.array(sorted(recs))
    p01_T = np.array([recs[t]["r_p01"] for t in rec_T])
    p99_T = np.array([recs[t]["r_p99"] for t in rec_T])
    A_T = np.array([recs[t]["A"] for t in rec_T])

    def fields_at(Tv, x):
        j = int(np.clip(np.searchsorted(T, Tv) - 1, 0, nT - 2))
        w = (Tv - T[j]) / (T[j + 1] - T[j])
        uu = (1 - w) * U[j] + w * U[j + 1]
        vv = (1 - w) * V[j] + w * V[j + 1]
        su_, sv_ = spl(r, uu), spl(r, vv)
        return 1 + float(su_(x)), 1 + float(sv_(x))
    rays = []
    for Tq in Tq_list:
        T0 = Tq * rs
        if T0 + a.ray_length * rs > T[-1]:
            continue
        A0 = float(np.interp(T0, rec_T, A_T))
        p99_0 = float(np.interp(T0, rec_T, p99_T))
        starts = [("2A", 2 * A0, "lead's list")]
        if p99_0 < A0:
            starts.append(("mid_p99_A", 0.5 * (p99_0 + A0), "builder's addition"))
        for sname, x0, origin in starts:
            for branch in (+1, -1):
                def rhs(Tv, y, branch=branch):
                    ps, ch = fields_at(Tv, max(y[0], h))
                    return [branch * (ch / ps) / ps ** 2]
                sol = solve_ivp(rhs, (T0, T0 + a.ray_length * rs), [x0], method="RK45", rtol=1e-8, atol=1e-8 * h,
                                t_eval=np.linspace(T0, T0 + a.ray_length * rs, 81))
                xs, Ts_ = sol.y[0], sol.t
                Rs = np.array([x * fields_at(t, max(x, h))[0] ** 2 for t, x in zip(Ts_, xs)])
                kinkz = np.array([(np.interp(t, rec_T, p01_T) - h <= x <= np.interp(t, rec_T, p99_T) + h) for t, x in zip(Ts_, xs)])
                dR = np.diff(Rs)
                dec_out = [(float(Ts_[i + 1] / rs), float(xs[i + 1] / np.interp(Ts_[i + 1], rec_T, A_T))) for i in range(len(dR))
                           if branch > 0 and dR[i] < 0 and not kinkz[i] and not kinkz[i + 1]]
                rays.append({"start": sname, "origin": origin, "T0_over_rs": Tq, "branch": "outgoing" if branch > 0 else "ingoing",
                             "r0_over_A": x0 / A0, "r_end_over_A": float(xs[-1] / np.interp(Ts_[-1], rec_T, A_T)), "R_start_over_4A": float(Rs[0] / (4 * A0)),
                             "R_end_over_4A": float(Rs[-1] / (4 * np.interp(Ts_[-1], rec_T, A_T))), "R_min_over_4A": float(Rs.min() / (4 * A0)),
                             "steps_in_kink": int(np.sum(kinkz)), "outgoing_R_decreasing_outside_kink": len(dec_out),
                             "first_decrease": dec_out[0] if dec_out else None, "status": sol.message})
    tot = {key: sum(1 for q in rows if q["counts"].get(key) and q["counts"][key]["both_negative_nonkink"] > 0) for key in ("c3", "c5", "bw")}
    totk = {key: sum(1 for q in rows if q["counts"].get(key) and q["counts"][key]["both_negative_kink"] > 0) for key in ("c3", "c5", "bw")}
    json.dump({"meta": {"fields": a.fields, "json": a.json, "r_s": rs, "h": h, "uniform_outputs": uniform, "output_spacing_over_rs": float(dTs[0] / rs),
                        "script": "strand7_areal_char_v3.py", "builder": "analyst"},
               "summary": {"outputs": len(rows), "outputs_with_both_negative_nonkink": tot, "outputs_with_both_negative_kink": totk},
               "rows": rows, "rays": rays}, open(a.out_json, "w"), indent=1, default=float)

    def f(x, d=4):
        if x is None:
            return "-"
        x = float(x)
        return ("%." + str(d) + "g") % x if (x == 0 or 1e-3 <= abs(x) < 1e5) else "%.3e" % x

    def pt(q, name):
        p = q["points"].get(name)
        if not p:
            return "-"
        return "%s, %s (+-%s, +-%s)" % (f(p["c_plus"]), f(p["c_minus"]), f(p["spread_c_plus"], 2), f(p["spread_c_minus"], 2))
    sel = [q for q in rows if abs(q["T_over_rs"] - round(q["T_over_rs"])) < 1e-6 or 26.5 - 1e-9 <= q["T_over_rs"] <= 29.5 + 1e-9]
    L = ["# Strand 7 S7a part 2 (d), version 2: the areal characteristics with the terms apart, the differencing controlled, the quantiles, "
         "the candidate zeros and short rays (strand7_areal_char_v3.py; %s)" % a.fields, "",
         "- Outputs: %d, %s r_s apart (uniform: %s); psi_dot by the three-point central difference (c3), the five-point stencil (c5) and the "
         "backward difference (bw); the dt/2 rung's fields are not stored." % (len(rows), f(dTs[0] / rs, 3), uniform),
         "- Outputs with a node where both branches are negative outside the kink zone (r_p01 - h to r_p99 + h): c3 %d, c5 %d, bw %d; inside the "
         "kink zone: c3 %d, c5 %d, bw %d." % (tot["c3"], tot["c5"], tot["bw"], totk["c3"], totk["c5"], totk["bw"]),
         "- Columns: c_+, c_- with the central estimate, (+- the spread of c_+, c_- over the three estimates); at the quantiles with psi' from the "
         "inner and the outer side.", "",
         "| T/r_s | r_sh/A (r_p01/A, r_p99/A) | r_sh, inner | r_sh, outer | r_p01, outer | r_p99, outer | r = A | (r_p99 + A)/2 | r = 2A | r_p01/2 | "
         "both-negative nodes c3/c5/bw (non-kink; kink) | min c_+ off the kink zone (r/A) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for q in sel:
        c = q["counts"]
        bn = "/".join(str(c[k_]["both_negative_nonkink"]) if c.get(k_) else "-" for k_ in ("c3", "c5", "bw"))
        bk = "/".join(str(c[k_]["both_negative_kink"]) if c.get(k_) else "-" for k_ in ("c3", "c5", "bw"))
        L.append("| %s | %s (%s, %s) | %s | %s | %s | %s | %s | %s | %s | %s | %s; %s | %s (%s) |" % (
            f(q["T_over_rs"], 5), f(q["r_sh_over_A"]), f(q["r_p01_over_A"]), f(q["r_p99_over_A"]), pt(q, "r_sh_inner"), pt(q, "r_sh_outer"),
            pt(q, "r_p01_outer"), pt(q, "r_p99_outer"), pt(q, "A"), pt(q, "mid_p99_A"), pt(q, "2A"), pt(q, "half_p01"), bn, bk,
            f(q["min_c_plus_nonkink"]), f(q["r_over_A_at_min_nonkink"], 3)))
    eps, cur = [], []
    for q in rows:
        if q["both_negative_nonkink_nodes"]:
            if cur and q["k"] != cur[-1]["k"] + 1:
                eps.append(cur)
                cur = []
            cur.append(q)
    if cur:
        eps.append(cur)
    L += ["", "## Nodes outside the kink zone with both branches negative (central estimate; builder's addition, version 3): one row per episode", "",
          "| T/r_s (outputs) | nodes' r/A | t_1 | t_2 | psi_dot r/A_dot | A_dot r_s/A |", "|---|---|---|---|---|---|"]
    for e in eps:
        nd = [x for q in e for x in q["both_negative_nonkink_nodes"]]
        rat = [x["psidot_r_over_Adot"] for x in nd if x["psidot_r_over_Adot"] is not None]
        ad = [q["A_dot"] * rs / q["A"] for q in e]
        L.append("| %s to %s (%d) | %s to %s | %s to %s | %s to %s | %s | %s to %s |" % (
            f(e[0]["T_over_rs"], 5), f(e[-1]["T_over_rs"], 5), len(e), f(min(x["r_over_A"] for x in nd)), f(max(x["r_over_A"] for x in nd)),
            f(min(x["t1"] for x in nd)), f(max(x["t1"] for x in nd)), f(min(x["t2"] for x in nd)), f(max(x["t2"] for x in nd)),
            ("%s to %s" % (f(min(rat)), f(max(rat)))) if rat else "-", f(min(ad)), f(max(ad))))
    L += ["", "## Candidate zeros of c_+^(R) on the nodes (central estimate) at the tabulated outputs", "",
          "| T/r_s | zero r/A | c_- there | c_+ inner node, outer node | both negative on the inner node | cells to the kink zone |", "|---|---|---|---|---|---|"]
    for q in sel:
        for zz in q["zeros"]:
            L.append("| %s | %s | %s | %s, %s | %s | %s |" % (f(q["T_over_rs"], 5), f(zz["r_over_A"]), f(zz["c_minus"]), f(zz["c_plus_inner_node"]),
                                                            f(zz["c_plus_outer_node"]), "yes" if zz["both_negative_inner"] else "no", f(zz["cells_to_kink"], 3)))
    L += ["", "## Short rays (dr/dT = +-N psi^-2 on the stored fields, %s r_s)" % f(a.ray_length, 3), "",
          "| start | T0/r_s | branch | r0/A | r_end/A | R/4A start, end, min | steps in the kink zone (of 81) | outgoing steps with R decreasing outside the kink zone (first at T/r_s, r/A) |",
          "|---|---|---|---|---|---|---|---|"]
    for y in rays:
        L.append("| %s (%s) | %s | %s | %s | %s | %s, %s, %s | %d | %d%s |" % (
            y["start"], y["origin"], f(y["T0_over_rs"], 4), y["branch"], f(y["r0_over_A"]), f(y["r_end_over_A"]), f(y["R_start_over_4A"]),
            f(y["R_end_over_4A"]), f(y["R_min_over_4A"]), y["steps_in_kink"], y["outgoing_R_decreasing_outside_kink"],
            (" (%s, %s)" % (f(y["first_decrease"][0], 5), f(y["first_decrease"][1])) if y["first_decrease"] else "")))
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:5]))


if __name__ == "__main__":
    main()
