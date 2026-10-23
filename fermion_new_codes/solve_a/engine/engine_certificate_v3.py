#!/usr/bin/env python
"""engine_certificate_v3.py (builder analyst; engine spec 31.5 and the icloud lane's two normalization corrections of
COLLAPSE_certificate_v1.1 (its status file: "compare lapse times the reconstructed rows with raw direct rows using the raw scales, and
divide the direct tangential row by the lapse in the mass defect before interpreting corrected comparisons")): version 2 with (1) the
agreement |N R_quadrature - res_direct|/|res_direct| and the quadrature column |N R_quadrature|/scale, the reconstructed rows R_t and R_r
(lapse-divided by their construction) multiplied by the lapse N at the node before they meet the raw direct rows, at the raw rows' own
scales (unchanged); (2) the direct mass defect M_dot + 4 pi r^2 [...] - (r^2 v/G) res_tt/N, the direct tangential row divided by the lapse
as the quadrature route's R_t is. With N = 1 (the homogeneous validation) nothing changes. Nothing else changes; version 2's docstring
follows. engine_certificate_v2.py (builder analyst; version 2 = version 1 with the quadrature from the centre by Gauss-Legendre rules of --gl
points on each interval between consecutive nodes (and on [0, r_1]), the integrand evaluated at the rule's points: version 1's trapezoid on
ell/8 carried a relative error 7.8e-3 in R_t at the innermost node of the homogeneous validation (its --validate output; O(h^2/r^2)).
Nothing else changes. Version 1's docstring: engine spec 28.2, S1 of REVIEWA_REVIEW_web40e.md: "engine_certificate_v1 (new; CPU, minutes;
now): per record and coarse node rho_ell and J_ell by the solve's kernel maps ..., s_r, s_t the smoothed stresses (dust plus pairs; the
pairs apart), N, v the dense solution, time derivatives by centred differences of neighbouring records (one-sided at the ends, labelled);
the continuity defects W_0 and W_r, R_t by the quadrature from the regular centre (nodes with v = 0 excluded, listed), R_r from W_0 and R_t
with A = v'/N, B = v/(r N), and the mass-balance defect, all in the review's S1 forms with M = r w^2/(2G), w = v/N. Three evaluations
compared node by node: these quadratures; engine_rows_v2's direct residuals; the covariant form ... Acceptance [chosen before
consequence]: agreement to 1/10 of the residual they report (label B's normalization); otherwise the discrepancy is printed and nothing is
attributed."). This version carries the first two evaluations and the mass balance; the covariant form follows its symbolic derivation
(a later version). The review's forms, implemented as written (G = g = G/a^2, a = 1):
  W_0 = rho_dot + r^-2 [r^2 (v rho + N J)]' + J N' + s_r v' + 2 s_t v/r,
  W_r = J_dot + r^-2 [r^2 (v J + N s_r)]' - 2 N s_t/r + rho N' + J v',
  R_t = (4 pi G/(N r^2 v)) int_0^r s^2 (N W_0 + v W_r) ds,   R_r = ((4 pi G/N) W_0 - (A + B) R_t)/B,  A = v'/N, B = v/(r N),
  the mass-balance defect  M_dot + 4 pi r^2 [v (rho + s_r) + N (1 + w^2) J] - (r^2 v/G) R_t,  M = r w^2/(2G), w = v/N.
The direct residuals are engine_rows_v2's res_rr = d_t k_r - rhs_rr and res_tt = d_t k_t - rhs_tt (k_r = v'/N, k_t = v/(r N)), whose sign
reproduces the review's homogeneous check (N = 1, v = -H r: both rows -H_dot + 4 pi G (rho + p) with the constraint 3 H^2 = 8 pi G rho, and
(4 pi G) W_0 = -3 H R, i.e. 3 H H_dot = 4 pi G rho_dot). Per record: the dense fields re-solved from the stored shell totals by the run's own
field module (per segment), rho_ell and J_ell the solve's smoothed sources (callables), s_r and s_t by engine_stress_v2.smooth_stress from the
stored per-shell stresses of the dust plus the pairs (the pairs' smoothed stress printed apart), spatial derivatives by Richardson-combined
central differences at h = ell/--hfrac (engine_rows_v2.deriv), the quadrature on the grid ell/--qfrac from the centre (the first interval
linear from zero), time derivatives centred on the neighbouring records (one-sided at the run's ends, labelled), the coarse nodes r_i = i
ell up to r_collar - 2 ell as in engine_rows_v2. Normalization (label B's): |res| over max(the largest bracket term, |d_t k|) per node; the
support: rho_ell above 1e-6 of its largest value. The agreement column: |R_quadrature - res_direct| over |res_direct| at the support nodes
(the acceptance 1/10), its largest value and the share of support nodes within it; nodes where |v| < 1e-12 are excluded and listed.
--validate: the homogeneous ball (N = 1, v = -h r, p = 0, rho = 3 h^2/(8 pi g) at every time, the constraint holding): h = h_dust (W_0 = 0,
both residuals 0) and h = (1 + eps) h_dust (W_0 and the residuals nonzero, R = (3/2) h_dust^2 (1 + eps) eps): the two evaluations must agree.
Args: --data (a COLLAPSE_records data npz) --fm-segments --fm-default --collar-policy --support-rel --records (comma list of indices, or
"all") --hfrac --qfrac --md --out-json | --validate --md."""
import argparse
import importlib
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_rows_v2 as RW  # noqa: E402
import engine_stress_v2 as STR2  # noqa: E402


def cum_from_centre(r, f):
    """int_0^r f ds on the grid r (r[0] > 0, f(0) = 0 taken linear on [0, r[0]])."""
    c = np.empty_like(f)
    c[0] = 0.5 * r[0] * f[0]
    c[1:] = c[0] + np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(r))
    return c


def certificate(F_m, F_0, F_p, dtm, dtp, g, nodes, hq, h):
    """F_*: dicts of callables N, v, rho, J, sr, st at the previous, the current and the next time (None at an end); dtm, dtp the time
    steps to them. Returns per-node arrays of the quadrature residuals, the direct residuals, their scales and the mass-balance defect."""
    G = g
    if F_m is not None and F_p is not None:
        tdot = lambda key, r: (F_p[key](r) - F_m[key](r)) / (dtm + dtp); how = "centred"
    elif F_p is not None:
        tdot = lambda key, r: (F_p[key](r) - F_0[key](r)) / dtp; how = "one-sided (forward)"
    else:
        tdot = lambda key, r: (F_0[key](r) - F_m[key](r)) / dtm; how = "one-sided (backward)"
    flux0 = lambda r: r ** 2 * (F_0["v"](r) * F_0["rho"](r) + F_0["N"](r) * F_0["J"](r))
    fluxr = lambda r: r ** 2 * (F_0["v"](r) * F_0["J"](r) + F_0["N"](r) * F_0["sr"](r))

    def W(r):
        r = np.asarray(r, float)
        N, N1, _ = RW.deriv(F_0["N"], r, h)
        v, v1, _ = RW.deriv(F_0["v"], r, h)
        rho = F_0["rho"](r); J = F_0["J"](r); sr = F_0["sr"](r); st = F_0["st"](r)
        _, d_flux0, _ = RW.deriv(flux0, r, h)
        _, d_fluxr, _ = RW.deriv(fluxr, r, h)
        W0 = tdot("rho", r) + d_flux0 / r ** 2 + J * N1 + sr * v1 + 2.0 * st * v / r
        Wr = tdot("J", r) + d_fluxr / r ** 2 - 2.0 * N * st / r + rho * N1 + J * v1
        return W0, Wr, N, v, v1, rho, J, sr
    xg, wg = np.polynomial.legendre.leggauss(int(hq))
    edges = np.concatenate([[0.0], nodes])
    a_ = edges[:-1][:, None]; b_ = edges[1:][:, None]
    pts = (0.5 * (b_ - a_) * xg[None, :] + 0.5 * (b_ + a_)).ravel()
    W0p, Wrp, Np, vp, _, _, _, _ = W(pts)
    integ = (pts ** 2 * (Np * W0p + vp * Wrp)).reshape(len(nodes), len(xg))
    cumn = np.cumsum((integ * wg[None, :]).sum(axis=1) * 0.5 * (edges[1:] - edges[:-1]))
    W0, Wr, N, v, v1, rho, J, sr = W(nodes)
    idx = np.arange(len(nodes))
    Nn, vn, v1n, W0n = N, v, v1, W0
    ok_v = np.abs(vn) >= 1e-12
    Rt_q = np.full(len(nodes), np.nan); Rr_q = np.full(len(nodes), np.nan)
    Rt_q[ok_v] = 4.0 * math.pi * G * cumn[ok_v] / (Nn[ok_v] * nodes[ok_v] ** 2 * vn[ok_v])
    A = v1n / Nn; B = vn / (nodes * Nn)
    Rr_q[ok_v] = (4.0 * math.pi * G * W0n[ok_v] / Nn[ok_v] - (A[ok_v] + B[ok_v]) * Rt_q[ok_v]) / B[ok_v]
    # the direct residuals (engine_rows_v2's brackets; d_t k from the neighbours)
    bb = RW.brackets(F_0["N"], F_0["v"], F_0["rho"](nodes), F_0["sr"](nodes), F_0["st"](nodes), g, nodes, h)
    z = np.zeros(len(nodes))
    kk = {}
    for lab, F in (("m", F_m), ("p", F_p)):
        if F is not None:
            b_ = RW.brackets(F["N"], F["v"], z, z, z, g, nodes, h)
            kk[lab] = (b_["k_r"], b_["k_t"])
    if "m" in kk and "p" in kk:
        dkr = (kk["p"][0] - kk["m"][0]) / (dtm + dtp); dkt = (kk["p"][1] - kk["m"][1]) / (dtm + dtp)
    elif "p" in kk:
        dkr = (kk["p"][0] - bb["k_r"]) / dtp; dkt = (kk["p"][1] - bb["k_t"]) / dtp
    else:
        dkr = (bb["k_r"] - kk["m"][0]) / dtm; dkt = (bb["k_t"] - kk["m"][1]) / dtm
    res_rr = dkr - bb["rhs_rr"]; res_tt = dkt - bb["rhs_tt"]
    sc_rr = np.maximum(bb["terms_rr"], np.abs(dkr)); sc_tt = np.maximum(bb["terms_tt"], np.abs(dkt))
    # the mass balance at the nodes
    Mf = lambda F, r: r * (F["v"](r) / F["N"](r)) ** 2 / (2.0 * G)
    if F_m is not None and F_p is not None:
        Mdot = (Mf(F_p, nodes) - Mf(F_m, nodes)) / (dtm + dtp)
    elif F_p is not None:
        Mdot = (Mf(F_p, nodes) - Mf(F_0, nodes)) / dtp
    else:
        Mdot = (Mf(F_0, nodes) - Mf(F_m, nodes)) / dtm
    wn = vn / Nn
    flux_M = 4.0 * math.pi * nodes ** 2 * (vn * (rho + sr) + Nn * (1.0 + wn ** 2) * J)
    mdef_q = Mdot + flux_M - (nodes ** 2 * vn / G) * Rt_q
    mdef_d = Mdot + flux_M - (nodes ** 2 * vn / G) * res_tt / Nn  # version 3: the direct tangential row divided by the lapse
    msc = np.maximum(np.abs(Mdot), np.abs(flux_M))
    return {"how": how, "Rt_q": Rt_q, "Rr_q": Rr_q, "res_tt": res_tt, "res_rr": res_rr, "sc_tt": sc_tt, "sc_rr": sc_rr, "W0": W0n, "Wr": Wr,
            "excluded_v0": nodes[~ok_v].tolist(), "mdef_q": mdef_q, "mdef_d": mdef_d, "msc": msc, "rho_nodes": rho, "N_nodes": Nn}


def summarize(c, nodes, support):
    out = {}
    for row, rq_, rd, sc in (("tt", c["Rt_q"], c["res_tt"], c["sc_tt"]), ("rr", c["Rr_q"], c["res_rr"], c["sc_rr"])):
        m = support & np.isfinite(rq_)
        if not m.any():
            out[row] = None; continue
        Nq = c["N_nodes"][m] * rq_[m]  # version 3: the lapse times the reconstructed row
        qd = np.abs(rd[m]) / sc[m]; qq = np.abs(Nq) / sc[m]
        agree = np.abs(Nq - rd[m]) / np.maximum(np.abs(rd[m]), 1e-300)
        i = int(np.argmax(qd))
        out[row] = {"direct_max": float(qd.max()), "direct_max_at": float(nodes[m][i]), "quad_max": float(qq.max()),
                    "agree_max": float(agree.max()), "agree_max_at": float(nodes[m][int(np.argmax(agree))]), "share_within_0.1": float(np.mean(agree <= 0.1)),
                    "agree_median": float(np.median(agree))}
    m = support & np.isfinite(c["mdef_q"])
    out["mass"] = ({"defect_q_max": float(np.max(np.abs(c["mdef_q"][m]) / np.maximum(c["msc"][m], 1e-300))),
                    "defect_d_max": float(np.max(np.abs(c["mdef_d"][m]) / np.maximum(c["msc"][m], 1e-300)))} if m.any() else None)
    return out


def validate(a):
    g = 0.25; tc = 100.0; t1 = 40.0; dt = 1e-3
    one = lambda r: np.ones_like(np.asarray(r, float)); zero = lambda r: np.zeros_like(np.asarray(r, float))

    def F(t, eps):
        h = (1.0 + eps) * 2.0 / (3.0 * (tc - t))
        rho = 3.0 * h * h / (8.0 * math.pi * g)
        return {"N": one, "v": (lambda r, h=h: -h * np.asarray(r, float)), "rho": (lambda r, rho=rho: rho * one(r)), "J": zero, "sr": zero, "st": zero}, h, rho
    nodes = np.arange(4.0, 48.0 + 1e-9, 4.0)
    L = ["## engine_certificate_v3 --validate (homogeneous: N = 1, v = -h r, p = 0, rho = 3 h^2/(8 pi g) at every time so the constraint holds; h = (1 + eps) h_dust, h_dust = 2/(3 (t_c - t)); t_c 100, t 40, dt 1e-3)", ""]
    for eps in (0.0, 1e-3, 1e-2, 1e-1):
        (Fm, _, _), (F0, h0, rho0), (Fp, _, _) = F(t1 - dt, eps), F(t1, eps), F(t1 + dt, eps)
        c = certificate(Fm, F0, Fp, dt, dt, g, nodes, 8, 4.0 / 32.0)
        hd = 2.0 / (3.0 * (tc - t1))
        Rpred = 1.5 * hd * hd * (1.0 + eps) * eps
        L.append("- eps %g: W_0 %.6e (predicted -3 h R/(4 pi g) %.6e), W_r max %.1e; R_t quadrature %.8e, direct %.8e; R_r quadrature %.8e, direct %.8e; predicted R = (3/2) h_dust^2 (1 + eps) eps = %.8e; the largest relative difference quadrature/direct over the nodes: tt %.2e, rr %.2e." % (
            eps, float(np.mean(c["W0"])), -3.0 * h0 * Rpred / (4.0 * math.pi * g), float(np.max(np.abs(c["Wr"]))), float(np.nanmean(c["Rt_q"])), float(np.mean(c["res_tt"])),
            float(np.nanmean(c["Rr_q"])), float(np.mean(c["res_rr"])), Rpred,
            float(np.nanmax(np.abs(c["Rt_q"] - c["res_tt"]) / np.maximum(np.abs(c["res_tt"]), 1e-300))) if eps else float("nan"),
            float(np.nanmax(np.abs(c["Rr_q"] - c["res_rr"]) / np.maximum(np.abs(c["res_rr"]), 1e-300))) if eps else float("nan")))
    open(a.md, "w").write("\n".join(L) + "\n"); print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true"); ap.add_argument("--data"); ap.add_argument("--md", required=True); ap.add_argument("--out-json")
    ap.add_argument("--fm-segments", default="v17 (replayed)=engine_fields_v12;v19 (native)=engine_fields_v14")
    ap.add_argument("--fm-default", default="engine_fields_v12")
    ap.add_argument("--collar-policy", default="flag"); ap.add_argument("--support-rel", type=float, default=1e-14)
    ap.add_argument("--records", default="all"); ap.add_argument("--hfrac", type=float, default=32.0); ap.add_argument("--gl", type=int, default=8)
    a = ap.parse_args()
    if a.validate:
        return validate(a)
    Z = np.load(a.data, allow_pickle=True)
    R = np.asarray(Z["R"], float); T = np.asarray(Z["T"], float); g = float(Z["g"]); ell = float(Z["ell"]); rs = float(Z["rs"])
    seg_names = [str(x) for x in Z["segment_names"]] if "segment_names" in Z.files else None
    seg = np.asarray(Z["segment"], int) if "segment" in Z.files else np.zeros(len(T), int)
    fmap = dict(x.split("=", 1) for x in a.fm_segments.split(";")) if a.fm_segments else {}
    fms = {}

    def fm_for(k):
        name = fmap.get(seg_names[seg[k]], a.fm_default) if seg_names else a.fm_default
        if name not in fms:
            fms[name] = importlib.import_module(name)
        return fms[name], name

    cache = {}

    def fields(k):
        if k in cache:
            return cache[k]
        FM, name = fm_for(k)
        e = np.asarray(Z["eps_tot"][k], float); Jt = np.asarray(Z["J_tot"][k], float)
        _, _, _, info = FM.solve_constraints_ivp(e, Jt, g, R, ell, collar_policy=a.collar_policy, support_rel=a.support_rel)
        Sr = np.asarray(Z["S_r_dust"][k], float) + np.asarray(Z["S_r_pairs"][k], float)
        St = np.asarray(Z["S_t_dust"][k], float) + np.asarray(Z["S_t_pairs"][k], float)
        F = {"N": info["N_dense"], "v": info["v_dense"], "rho": info["rho_ell"], "J": info["J_ell"],
             "sr": (lambda r, Sr=Sr, St=St: STR2.smooth_stress(Sr, St, R, ell, np.atleast_1d(np.asarray(r, float)))[0]),
             "st": (lambda r, Sr=Sr, St=St: STR2.smooth_stress(Sr, St, R, ell, np.atleast_1d(np.asarray(r, float)))[1]),
             "info": info, "fm": name, "failed": bool(info.get("failed"))}
        Srp = np.asarray(Z["S_r_pairs"][k], float); Stp = np.asarray(Z["S_t_pairs"][k], float)
        F["pairs_sr"] = (lambda r, Srp=Srp, Stp=Stp: STR2.smooth_stress(Srp, Stp, R, ell, np.atleast_1d(np.asarray(r, float)))[0])
        cache[k] = F
        for old in [x for x in cache if x < k - 2]:
            del cache[old]
        return F

    ks = list(range(len(T))) if a.records == "all" else [int(x) for x in a.records.split(",")]
    rows = []
    t0 = time.time()
    for k in ks:
        F0 = fields(k)
        if F0["failed"]:
            rows.append({"k": k, "T_over_rs": T[k] / rs, "failed": True}); continue
        Fm = fields(k - 1) if k - 1 >= 0 else None
        Fp = fields(k + 1) if k + 1 < len(T) else None
        rc = float(F0["info"]["r_collar"]); nodes = np.arange(ell, rc - 2.0 * ell + 1e-9, ell)
        c = certificate(Fm, F0, Fp, (T[k] - T[k - 1]) if Fm else None, (T[k + 1] - T[k]) if Fp else None, g, nodes, a.gl, ell / a.hfrac)
        rho_n = c["rho_nodes"]; support = np.abs(rho_n) > 1e-6 * max(float(np.max(np.abs(rho_n))), 1e-300)
        s = summarize(c, nodes, support)
        pairs_sr = np.abs(F0["pairs_sr"](nodes)); tot_sr = np.abs(F0["sr"](nodes))
        rows.append({"k": k, "T_over_rs": T[k] / rs, "fm": F0["fm"], "time_derivative": c["how"], "nodes": len(nodes), "support_nodes": int(support.sum()),
                     "excluded_v0": c["excluded_v0"], "summary": s,
                     "pairs_share_of_sr": float(np.max(pairs_sr[support]) / max(float(np.max(tot_sr[support])), 1e-300)) if support.any() else None})
        if len(rows) % 20 == 1:
            print("record %d (T/rs %.3f) done, %.0f s" % (k, T[k] / rs, time.time() - t0), flush=True)
    f = lambda x, fm: (fm % x) if x is not None else "-"
    L = ["## The evolution-row certificate (engine_certificate_v3.py; engine spec 28.2, S1, with the icloud lane's two normalization corrections, engine spec 31.5): the quadrature from the continuity defects, times the lapse, against engine_rows_v2's raw direct residuals; the direct mass defect with the tangential row divided by the lapse", "",
         "- data: %s (g %g, ell %g, r_s %g; %d records); field modules per segment: %s; derivatives at h = ell/%g, the quadrature by %d-point Gauss-Legendre rules per node interval; the support: rho_ell above 1e-6 of its largest value." % (
             os.path.basename(a.data), g, ell, rs, len(T), a.fm_segments if seg_names else a.fm_default, a.hfrac, a.gl),
         "- columns per row (tt, rr): the support maximum of |res|/scale for the direct and the quadrature evaluation; the agreement |N R_quadrature - res_direct|/|res_direct| at the support nodes (version 3: the lapse times the reconstructed row): its largest value (at r), its median and the share of support nodes within the acceptance 1/10; the mass-balance defect over its largest term with R_t from the quadrature and from the direct row; the pairs' share of the smoothed s_r.", "",
         "| T/rs | d_t | tt direct | tt quadrature | tt agreement max (r) | tt median | tt share within 0.1 | rr direct | rr quadrature | rr agreement max (r) | rr median | rr share within 0.1 | mass defect (quad, direct) | pairs' s_r share | excluded v = 0 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("failed"):
            L.append("| %.3f | the solve failed | - | - | - | - | - | - | - | - | - | - | - | - | - |" % r["T_over_rs"]); continue
        s = r["summary"]; tt = s.get("tt") or {}; rr = s.get("rr") or {}; ms = s.get("mass") or {}
        L.append("| %.3f | %s | %s | %s | %s (%s) | %s | %s | %s | %s | %s (%s) | %s | %s | %s, %s | %s | %s |" % (
            r["T_over_rs"], "c" if r["time_derivative"] == "centred" else r["time_derivative"], f(tt.get("direct_max"), "%.3e"), f(tt.get("quad_max"), "%.3e"),
            f(tt.get("agree_max"), "%.2e"), f(tt.get("agree_max_at"), "%.0f"), f(tt.get("agree_median"), "%.2e"), f(tt.get("share_within_0.1"), "%.3f"),
            f(rr.get("direct_max"), "%.3e"), f(rr.get("quad_max"), "%.3e"), f(rr.get("agree_max"), "%.2e"), f(rr.get("agree_max_at"), "%.0f"),
            f(rr.get("agree_median"), "%.2e"), f(rr.get("share_within_0.1"), "%.3f"), f(ms.get("defect_q_max"), "%.2e"), f(ms.get("defect_d_max"), "%.2e"),
            f(r.get("pairs_share_of_sr"), "%.3f"), ",".join("%.0f" % x for x in r["excluded_v0"]) or "-"))
    L += ["", "- every number above is printed by the script; acceptance (engine 28.2): the two evaluations agree to 1/10 of the residual they report; where they do not, the discrepancy stands as printed and nothing is attributed."]
    open(a.md, "w").write("\n".join(L) + "\n")
    if a.out_json:
        json.dump({"script": "engine_certificate_v3.py", "data": a.data, "rows": rows}, open(a.out_json, "w"), indent=1, default=float)
    print("\n".join(L[:6])); print("rows %d, %.0f s" % (len(rows), time.time() - t0))


if __name__ == "__main__":
    main()
