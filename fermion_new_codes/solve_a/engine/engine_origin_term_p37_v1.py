#!/usr/bin/env python
"""engine_origin_term_p37_v1.py (builder analyst; engine spec 36.4 (5): "P37: 20.1's parenthesis 'a local operator on shells n-1 to n+2' is
corrected: on the open chain F_{n+1/2} carries the origin term -Tr[D B_in] at every cut; the builder prints Tr[D B_in] per record on the
production's certified records and S_r with and without it, against the certificate's 1/10 scale; turn 45 rules if it matters").
Record-only (no engine run).

The operators (engine_vacflow_v1.chain_sparse, a = 1): h_bond = sum_i [-(i/2) |i><i+1| sigma_x + h.c.], Q = sum_i [-(i/4) |i><i+2| + h.c.];
on the open chain i[h_bond, Q] = B_in + B_out, B_in = (i sigma_x/8)(|1><0| - |0><1|), B_out = (i sigma_x/8)(|L-2><L-1| - |L-1><L-2|)
(WEB40F P37 eq. 39; the builder re-derived the commutator from chain_sparse's stencils). With D = C - R2 (the dust's columns with their
occupations, the sea's with +1, R2's with -1, channel weights 16 kappa) and u = c_1^dag sigma_x c_0 per column (c_s the site-s spinor):
  b_in = Tr[D B_in] = -(1/4) sum_kappa w_kappa sum_j f_j Im u_j,  b_out likewise with c_{L-2}^dag sigma_x c_{L-1}.
engine_stress_v2's face flux is the prefix flux F_{n+1/2} = 2 Im sum_{s<=n} g_s = -Tr[D i[h_bond, Q_<=n]]; at the last face (Q_<=L-1 = Q)
F_last = -(b_in + b_out) exactly, so every stored record carries b_in + b_out = -F_last in its handoff (dust plus pairs); b_out alone needs
the far-end spinors, which only the stored orbital slices hold: there b_in and b_out are evaluated directly and b_in + b_out is compared with
-F_last of the same record (the identity's check). P37 eq. 40: F^prefix - F^local = -(1/2){Pi_<=n, B_in + B_out}, i.e. -b_in/2 at the cut
n = 0 and -b_in at every cut 1 <= n <= L-3; eq. 41: S_r^local - S_r^prefix = b_in/(4 pi r^2) at a face.
Per certified record of the production (engine_certificate_v3's json: both rows' shares within 1/10 at least 0.800 at three decimals; 281 of
321 by the certificate statement): b_in + b_out (= -F_last), the shell-value shift Delta S_r of the local allocation (b_in taken as -F_last,
b_out's share shown at the slices; the last two faces' b_out terms not applied), smoothed by engine_stress_v2.smooth_stress (which mixes it
into S_t,ell as well); S_r,ell at R_t with (the stored prefix stress) and without (the local stress); the largest |Delta s_r|/|s_r| over the
certificate's retained support nodes; and the certificate's own evaluation (engine_certificate_v3.certificate and summarize, the same
nodes, support, h, Gauss-Legendre rule and field modules) with the stored stress and with the local stress: per row the share within 1/10
(the criterion's quantity), the largest agreement, and the largest |res_local - res_prefix|/|res_prefix| of the direct residual over the
support with the share of support nodes where it exceeds 1/10. Every number is printed by the script; nothing is ruled.
Args: --json --handoff path=Tmin=Tmax (repeated, T/rs) --data --cert --slices (comma list) --fm-segments --fm-default --records --md --out-json."""
import argparse
import importlib
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_certificate_v3 as CERT  # noqa: E402
import engine_stress_v2 as STR2  # noqa: E402


def b_from_slice(path):
    """b_in and b_out from a stored orbital slice (the sea +1, the dust with f, R2 -1; channel weights)."""
    Z = np.load(path, allow_pickle=True)
    w = np.asarray(Z["weights"], float); ks = [int(k) for k in Z["kappas"]]
    b_in = b_out = 0.0
    per = {}
    for wk, k in zip(w, ks):
        bi = bo = 0.0
        for name, f in (("sea", None), ("dust", "f"), ("R2", None)):
            C = np.asarray(Z["%s_%d" % (name, k)])
            if C.shape[1] == 0:
                continue
            fj = np.ones(C.shape[1]) if name == "sea" else (-np.ones(C.shape[1]) if name == "R2" else np.asarray(Z["f_%d" % k], float))
            c0 = C[0:2, :].astype(np.complex128); c1 = C[2:4, :].astype(np.complex128)
            cl2 = C[-4:-2, :].astype(np.complex128); cl1 = C[-2:, :].astype(np.complex128)
            u_in = np.conj(c1[0]) * c0[1] + np.conj(c1[1]) * c0[0]          # c_1^dag sigma_x c_0
            u_out = np.conj(cl2[0]) * cl1[1] + np.conj(cl2[1]) * cl1[0]     # c_{L-2}^dag sigma_x c_{L-1}
            bi += -0.25 * float(np.sum(fj * u_in.imag)); bo += -0.25 * float(np.sum(fj * u_out.imag))
            del C
        per[k] = (wk * bi, wk * bo)
        b_in += wk * bi; b_out += wk * bo
    return {"T": float(Z["T"]), "rs": float(Z["rs"]), "dtype": str(Z["dtype"]), "b_in": b_in, "b_out": b_out, "per_channel": per}


def delta_shell(b, R):
    """the shell-value shift of S_r for the local allocation (P37 eqs. 40, 41): Delta F = b/2 at the face n = 0, b at the faces n >= 1 (the
    last two faces' b_out terms not applied); shells by engine_stress_v2.shell_values' rule (shell 0: half its outer face, S_t unchanged)."""
    R = np.asarray(R, float)
    dF = np.full(len(R), float(b)); dF[0] = 0.5 * b
    Rf = np.concatenate([R[:-1] + 0.5 * np.diff(R), [R[-1] + 0.5]])
    dSf = dF / (4.0 * math.pi * Rf ** 2)
    dS = 0.5 * (np.concatenate([[0.0], dSf[:-1]]) + dSf)
    return dS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--handoff", action="append", required=True, help="path=Tmin=Tmax (T/rs, inclusive)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--cert", required=True)
    ap.add_argument("--slices", default="")
    ap.add_argument("--fm-segments", default="v17 (replayed)=engine_fields_v12;v19 (native)=engine_fields_v14")
    ap.add_argument("--fm-default", default="engine_fields_v12")
    ap.add_argument("--collar-policy", default="flag"); ap.add_argument("--support-rel", type=float, default=1e-14)
    ap.add_argument("--records", default="certified", help="certified | all | comma list of indices")
    ap.add_argument("--hfrac", type=float, default=32.0); ap.add_argument("--gl", type=int, default=8)
    ap.add_argument("--md", required=True); ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    t0 = time.time()
    J = json.load(open(a.json))
    RJ = [r for r in J["records"] if r.get("E") is not None and "T_over_rs" in r]
    Z = np.load(a.data, allow_pickle=True)
    R = np.asarray(Z["R"], float); T = np.asarray(Z["T"], float); g = float(Z["g"]); ell = float(Z["ell"]); rs = float(Z["rs"])
    if len(RJ) != len(T) or float(np.max(np.abs(np.array([r["T"] for r in RJ]) - T))) > 1e-9:
        raise SystemExit("the json's records and the data npz's do not align")
    HO = []
    for spec in a.handoff:
        p, lo, hi = spec.split("=")
        H = np.load(p, allow_pickle=True)
        HO.append((float(lo), float(hi), np.asarray(H["T"], float), np.asarray(H["F_face_dust"], float), np.asarray(H["F_face_pairs"], float), p))

    def F_last(k):
        t = T[k] / rs
        for lo, hi, TH, Fd, Fp, p in HO:
            if lo - 1e-9 <= t <= hi + 1e-9:
                i = int(np.argmin(np.abs(TH - T[k])))
                if abs(TH[i] - T[k]) < 1e-9:
                    return float(Fd[i][-1] + Fp[i][-1]), float(Fd[i][-1]), float(Fp[i][-1]), os.path.basename(p)
        return None, None, None, None
    C = json.load(open(a.cert))
    crow = {int(r["k"]): r for r in C["rows"]}

    def certified(r):
        s = (r or {}).get("summary") or {}
        tt, rr = s.get("tt"), s.get("rr")
        return bool(tt and rr and round(tt["share_within_0.1"], 3) >= 0.8 and round(rr["share_within_0.1"], 3) >= 0.8)
    cert_ks = [k for k in range(len(T)) if certified(crow.get(k))]
    if a.records == "certified":
        ks = cert_ks
    elif a.records == "all":
        ks = list(range(len(T)))
    else:
        ks = [int(x) for x in a.records.split(",")]
    seg_names = [str(x) for x in Z["segment_names"]] if "segment_names" in Z.files else None
    seg = np.asarray(Z["segment"], int) if "segment" in Z.files else np.zeros(len(T), int)
    fmap = dict(x.split("=", 1) for x in a.fm_segments.split(";")) if a.fm_segments else {}
    mods = {}

    def fm_for(k):
        name = fmap.get(seg_names[seg[k]], a.fm_default) if seg_names else a.fm_default
        if name not in mods:
            mods[name] = importlib.import_module(name)
        return mods[name], name
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
             "Sr": Sr, "St": St, "info": info, "fm": name, "failed": bool(info.get("failed"))}
        cache[k] = F
        for old in [x for x in cache if x < k - 2]:
            del cache[old]
        return F
    rows = []
    for n_, k in enumerate(ks):
        Fl, Fd, Fp_, hof = F_last(k)
        row = {"k": k, "T_over_rs": T[k] / rs, "F_last": Fl, "F_last_dust": Fd, "F_last_pairs": Fp_, "handoff": hof, "certified": k in cert_ks}
        if Fl is None:
            row["note"] = "no handoff record"; rows.append(row); continue
        b = -Fl
        row["b_in_plus_b_out"] = b
        F0 = fields(k)
        if F0["failed"]:
            row["note"] = "the re-solve failed"; rows.append(row); continue
        Fm = fields(k - 1) if k - 1 >= 0 else None
        Fpp = fields(k + 1) if k + 1 < len(T) else None
        dS = delta_shell(b, R)
        SrL = F0["Sr"] + dS; St0 = F0["St"]
        F0L = dict(F0)
        F0L["sr"] = (lambda r, SrL=SrL, St0=St0: STR2.smooth_stress(SrL, St0, R, ell, np.atleast_1d(np.asarray(r, float)))[0])
        F0L["st"] = (lambda r, SrL=SrL, St0=St0: STR2.smooth_stress(SrL, St0, R, ell, np.atleast_1d(np.asarray(r, float)))[1])
        rc = float(F0["info"]["r_collar"]); nodes = np.arange(ell, rc - 2.0 * ell + 1e-9, ell)
        dtm = (T[k] - T[k - 1]) if Fm is not None else None; dtp = (T[k + 1] - T[k]) if Fpp is not None else None
        cW = CERT.certificate(Fm, F0, Fpp, dtm, dtp, g, nodes, a.gl, ell / a.hfrac)
        cL = CERT.certificate(Fm, F0L, Fpp, dtm, dtp, g, nodes, a.gl, ell / a.hfrac)
        rho_n = cW["rho_nodes"]; support = np.abs(rho_n) > 1e-6 * max(float(np.max(np.abs(rho_n))), 1e-300)
        sW = CERT.summarize(cW, nodes, support); sL = CERT.summarize(cL, nodes, support)
        srW = np.asarray(F0["sr"](nodes), float); srL = np.asarray(F0L["sr"](nodes), float)
        stW = np.asarray(F0["st"](nodes), float); stL = np.asarray(F0L["st"](nodes), float)
        m = support
        row.update({"nodes": len(nodes), "support_nodes": int(m.sum()), "time_derivative": cW["how"],
                    "dsr_rel_max": float(np.max(np.abs(srL[m] - srW[m]) / np.maximum(np.abs(srW[m]), 1e-300))) if m.any() else None,
                    "dst_rel_max": float(np.max(np.abs(stL[m] - stW[m]) / np.maximum(np.abs(stW[m]), 1e-300))) if m.any() else None,
                    "with": sW, "without": sL, "stored": (crow.get(k) or {}).get("summary")})
        for rw in ("tt", "rr"):
            d = np.abs(cL["res_" + rw][m] - cW["res_" + rw][m]); base = np.abs(cW["res_" + rw][m])
            q = d / np.maximum(base, 1e-300)
            row["dres_rel_max_" + rw] = float(np.max(q)) if m.any() else None
            row["dres_share_over_0.1_" + rw] = float(np.mean(q > 0.1)) if m.any() else None
        Rt = RJ[k].get("R_t")
        if Rt:
            row["R_t"] = Rt
            row["sr_at_Rt_with"] = float(np.asarray(F0["sr"](Rt), float)[0]); row["sr_at_Rt_without"] = float(np.asarray(F0L["sr"](Rt), float)[0])
            row["b_over_4piRt2"] = b / (4.0 * math.pi * Rt ** 2)
        rows.append(row)
        if n_ % 20 == 0:
            print("record %d (T/rs %.3f) done, %.0f s" % (k, T[k] / rs, time.time() - t0), flush=True)
    SL = []
    for p in [x for x in a.slices.split(",") if x.strip()]:
        s = b_from_slice(p)
        k = int(np.argmin(np.abs(T - s["T"])))
        Fl, Fd, Fp_, hof = F_last(k) if abs(T[k] - s["T"]) < 1e-9 else (None, None, None, None)
        s.update({"k": k, "F_last": Fl, "path": os.path.basename(p)})
        SL.append(s)
        print("slice T/rs %.3f done, %.0f s" % (s["T"] / s["rs"], time.time() - t0), flush=True)
    f = lambda x, fm="%.3e": (fm % x) if (x is not None and np.isfinite(x)) else "-"
    L = ["## The origin term of the open-chain prefix flux on the production's certified records (engine_origin_term_p37_v1.py; engine spec 36.4 (5), WEB40F P37 eqs. 39 to 41; record-only)", "",
         "- B_in = (i sigma_x/8)(|1><0| - |0><1|), B_out = (i sigma_x/8)(|L-2><L-1| - |L-1><L-2|): i[h_bond, Q] = B_in + B_out on the open chain (chain_sparse's stencils, a = 1); b = Tr[D B] with D = C - R2 (dust with f, sea +1, R2 -1, weights 16 kappa). The stored prefix flux's last face is F_last = -(b_in + b_out) exactly, so b_in + b_out = -F_last per record from the handoffs (dust plus pairs); b_out alone needs the far-end spinors (the slices below).",
         "- \"with\": the stored stress (engine_stress_v2's prefix flux, the origin term included); \"without\": the local allocation, S_r^local = S_r^prefix + Delta, Delta F = b/2 at the face n = 0 and b at the faces n >= 1 (P37 eqs. 40, 41; b = -F_last; the last two faces' b_out terms not applied), shells by shell_values' rule, smoothed by smooth_stress (the shift enters both s_r and s_t of the tensor smoothing).",
         "- the certificate's evaluation repeated with each stress (engine_certificate_v3.certificate and summarize: nodes ell to r_collar - 2 ell, support rho_ell above 1e-6 of its largest value, h = ell/%g, %d-point Gauss-Legendre, the field module per segment %s): per row the share of support nodes within 1/10 (the criterion, at least 0.800) and the largest agreement; \"res change\" the largest |res_without - res_with|/|res_with| of the direct residual over the support and the share of support nodes where it exceeds 1/10; \"stored\" the certificate json's own share (the with-evaluation must equal it)." % (a.hfrac, a.gl, a.fm_segments),
         "- %d records certified of %d by the rule (both rows' shares at least 0.800 at three decimals); this table: %d records (%s)." % (len(cert_ks), len(T), len(rows), a.records), ""]
    if SL:
        L += ["### At the stored orbital slices (the production's states; the identity's check and b_out's share)", "",
              "| T/rs | slice dtype | b_in (direct) | b_out (direct) | b_in + b_out | -F_last (handoff) | relative difference | b_out/b_in |", "|---|---|---|---|---|---|---|---|"]
        for s in SL:
            tot = s["b_in"] + s["b_out"]
            L.append("| %.3f | %s | %+.10e | %+.4e | %+.10e | %s | %s | %s |" % (s["T"] / s["rs"], s["dtype"], s["b_in"], s["b_out"], tot,
                                                                   ("%+.10e" % -s["F_last"]) if s["F_last"] is not None else "-",
                                                                   f(abs(tot + s["F_last"]) / max(abs(tot), 1e-300) if s["F_last"] is not None else None, "%.2e"),
                                                                   f(s["b_out"] / s["b_in"] if s["b_in"] else None, "%+.2e")))
        L.append("")
    L += ["### Per record", "",
          "| T/rs | b_in + b_out (= -F_last) | dust, pairs parts | R_t | b/(4 pi R_t^2) | s_r,ell(R_t) with | without | max abs(d s_r)/abs(s_r) on support | max abs(d s_t)/abs(s_t) | tt share with / without (stored) | tt res change max, share over 0.1 | rr share with / without (stored) | rr res change max, share over 0.1 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("note"):
            L.append("| %.3f | %s | - | - | - | - | - | - | - | %s | - | - | - |" % (r["T_over_rs"], f(r.get("b_in_plus_b_out"), "%+.4e"), r["note"])); continue
        sw, sl, st_ = r["with"], r["without"], r.get("stored") or {}
        g3 = lambda s, rw: f(((s or {}).get(rw) or {}).get("share_within_0.1"), "%.3f")
        L.append("| %.3f | %+.4e | %+.3e, %+.3e | %s | %s | %s | %s | %s | %s | %s / %s (%s) | %s, %s | %s / %s (%s) | %s, %s |" % (
            r["T_over_rs"], r["b_in_plus_b_out"], r["F_last_dust"] * -1.0, r["F_last_pairs"] * -1.0, f(r.get("R_t"), "%.3f"), f(r.get("b_over_4piRt2"), "%+.3e"),
            f(r.get("sr_at_Rt_with"), "%+.4e"), f(r.get("sr_at_Rt_without"), "%+.4e"), f(r.get("dsr_rel_max"), "%.2e"), f(r.get("dst_rel_max"), "%.2e"),
            g3(sw, "tt"), g3(sl, "tt"), g3(st_, "tt"), f(r.get("dres_rel_max_tt"), "%.2e"), f(r.get("dres_share_over_0.1_tt"), "%.3f"),
            g3(sw, "rr"), g3(sl, "rr"), g3(st_, "rr"), f(r.get("dres_rel_max_rr"), "%.2e"), f(r.get("dres_share_over_0.1_rr"), "%.3f")))
    ok = [r for r in rows if not r.get("note")]
    if ok:
        bw = [abs(r["b_in_plus_b_out"]) for r in ok]
        chg_tt = [r for r in ok if (r["without"].get("tt") or {}).get("share_within_0.1") is not None and round(r["without"]["tt"]["share_within_0.1"], 3) < 0.8]
        chg_rr = [r for r in ok if (r["without"].get("rr") or {}).get("share_within_0.1") is not None and round(r["without"]["rr"]["share_within_0.1"], 3) < 0.8]
        mism = [r for r in ok if r.get("stored") and any(round(((r["with"].get(rw) or {}).get("share_within_0.1") or -1), 3) != round(((r["stored"].get(rw) or {}).get("share_within_0.1") or -1), 3) for rw in ("tt", "rr"))]
        L += ["", "- summary over the %d records evaluated: |b_in + b_out| from %.3e to %.3e; records whose share with the local stress falls below 0.800: tt %d (%s), rr %d (%s); records whose with-evaluation share differs from the stored certificate's at three decimals: %d (%s); the largest direct-residual change over the support %.2e (tt) and %.2e (rr)." % (
            len(ok), min(bw), max(bw), len(chg_tt), ", ".join("%.3f" % r["T_over_rs"] for r in chg_tt[:30]) or "none", len(chg_rr), ", ".join("%.3f" % r["T_over_rs"] for r in chg_rr[:30]) or "none",
            len(mism), ", ".join("%.3f" % r["T_over_rs"] for r in mism[:30]) or "none",
            max(r["dres_rel_max_tt"] for r in ok if r.get("dres_rel_max_tt") is not None), max(r["dres_rel_max_rr"] for r in ok if r.get("dres_rel_max_rr") is not None))]
    L.append("- every number above is printed by the script from the production's json, handoffs, records data npz, certificate json and orbital slices; nothing is attributed or ruled.")
    open(a.md, "w").write("\n".join(L) + "\n")
    if a.out_json:
        json.dump({"rows": rows, "slices": SL}, open(a.out_json, "w"), default=float, indent=1)
    print("\n".join(L[:6])); print("done, %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
