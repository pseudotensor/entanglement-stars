#!/usr/bin/env python
"""gate_triple_read_v1.py (busi lane, GATECOMP kit (ii), inbox #400): the reader of the width triple and its controls, in the order the kit
requires. ReviewA_gatecomp_v1.md sec. 3 (diagnostics 1 to 8) with ReviewB_gatecomp_v1.md's conditions where reviewer A is silent. Box CPU;
reads run jsons and fields npz only; prints every number with its definition; rules nothing (the PASS/FAIL words are the written lines'
arithmetic, for reviewer A to read).

Order (reviewer B, from final4/requirements_new.txt sec. 2: "the ell control is applied first to the normal-ordered vacuum energy in the flow"):
  0. the vacuum ell control FIRST: the sea's own source against its continued reference (the engine's eps_pairs, J_pairs rows: "the shell
     totals of the continued sea against its reference, without the dust", engine v9; E_pairs_shells and pairs_inside_Rt per record),
     across the members at matched clocks: per clock the three values, the pairwise differences and the ratio d(s1,s2)/d(s2,s3) (a sigma^2
     law gives (s2^2 - s1^2)/(s3^2 - s2^2) = 1/4 for 4, 8, 16); printed before any formation number.
  1. certification per member (reviewer B): converged, the kept residual against the paper's ceiling 1e-3 (the engine's --end-above) and reviewer B's
     target 1e-5, over the whole history and over the window [T_birth - 1, end].
  2. charge drift (reviewer A 3.7, reviewer B): max |E/E0 - 1| (E is the collar charge) over the window; lines 1e-2 (reviewer A) and 1e-3 (reviewer B).
  3. birth per member (reviewer A 3.2), bracket and error.
  4. the sigma^2 law (reviewer A 3.3): T_birth and R_birth/2GE0 against (sigma/r_s)^2: the line through the outer two members; the middle
     member's departure against tau/2 (clock, in r_s) and 1 percent (radius, of 2GE0); the extrapolated T_0, R_0; c_T, c_R (c_R printed
     beside the kernel shift 9/4); T_0 within 0.5 r_s and R_0 within 10 percent of every member.
  5. matched-clock radii (reviewer A 3.4) over the common window: R_t per member, pairwise differences, the ratio d(s1,s2)/d(s2,s3) (1/4 within
     a factor 2) and the spread (max - min)/mean (10 percent line).
  6. the gate profile (reviewer A 3.1; gate_common_v1): sigma ||K||_op over r <= 2 R_t per record, T_gate, the window T_gate - T_birth; with
     --Tc SIGMA=T_c the caustic law T_c - (20/3) sigma/r_s printed beside T_gate; the six-term from --fm-dir (the code of record on the nodes r >= sigma gates; r < sigma reported; OPEN without).
  7. the step control (reviewer A 3.6): T_birth(tau/32) - T_birth(tau/16) of the 8a member against |T_birth(4a) - T_birth(8a)|.
  8. two solvers / the independent route (reviewer A 3.8; reviewer B: a second evolution and field solve, not a reread): the plain-mix control
     (--fields ivp) against the Newton member at matched clocks on the record floor (tolcmp_v3 C_LINES: E/E0 3.0e-7, roots 3.4e-6, depth
     4.6e-8, site fields 2.5e-5); birth from each.
  9. the point-sampled control at the root (reviewer A sec. 2 (ii)): R_t and min(N - |v|) averaged against point-sampled at matched clocks,
     beside (sigma/R_t)^2.
  10. reviewer B's tightening check: T_birth and R_birth with the stop rule tightened 100x, against 0.05 r_s and 1 percent.
  11. reviewer B's local re-integration: each re-integrated member's birth (records every tau/32) against its tau/16 interpolation and error.
  Diagnostic 5 (the evolution-row residual at the outer root) is read by the lane's evolution-row readout on the members' handoff rows
  (the readout behind the r_s/a 64 medians 0.17, 0.04, 0.06); this reader does not reimplement it.

  gate_triple_read_v1.py --rs 256 --member 4=DIR/TAG --member 8=DIR/TAG --member 16=DIR/TAG --tolcmp PATH [--step DIR/TAG]
      [--plain DIR/TAG] [--point DIR/TAG] [--point-free SIGMA] [--step3 DIR/TAG] [--step6 DIR/TAG] [--tight DIR/TAG] [--reint SIGMA=DIR/TAG ...] [--Tc SIGMA=T ...] [--fm-dir FIELD_MODULE_DIR]
      [--tau-rs 0.0625] [--out J]
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_common_v1 as GC  # noqa: E402


def kv(s):
    k, v = s.split("=", 1)
    return float(k), v


def record_floor(TC, JA, FA, JB, FB, rs, t_from=None):
    """X against Y on one chain at matched certified clocks, tolcmp_v3 C_LINES; -> (verdict, worst, n, fails)."""
    LINE = TC.C_LINES
    RA, RB = GC.recs_by_T(JA, rs), GC.recs_by_T(JB, rs)
    IA, IB = GC.rows_by_T(FA, rs), GC.rows_by_T(FB, rs)
    ts = [t for t in sorted(set(RA) & set(RB)) if GC.certified(RA[t]) and GC.certified(RB[t]) and (t_from is None or t > t_from + 1e-9)]
    worst = {"E_over_E0": 0.0, "roots": 0.0, "depth": 0.0, "fields": 0.0}; fails = []
    for t in ts:
        acc = TC.Acc(); TC.cmp_record(RA[t], RB[t], acc, t)
        p = {c: max([v[0] for v in L] or [0.0]) for c, L in acc.cls.items()}
        f = 0.0
        if FA is not None and FB is not None and t in IA and t in IB:
            for k in ("N", "v", "N_applied", "v_applied"):
                f = max(f, float(np.max(np.abs(np.asarray(FA[k][IA[t]], float) - np.asarray(FB[k][IB[t]], float)))))
        vals = {"E_over_E0": p.get("E_over_E0", 0.0), "roots": p.get("roots", 0.0), "depth": p.get("depth", 0.0), "fields": f}
        bad = [c for c in vals if vals[c] > LINE[c]]
        for c in vals:
            worst[c] = max(worst[c], vals[c])
        if bad:
            fails.append((t, bad))
    return ("PASS" if (ts and not fails) else ("FAIL" if ts else "NO CLOCKS")), worst, len(ts), fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs", type=float, required=True); ap.add_argument("--member", action="append", required=True)
    ap.add_argument("--tolcmp", required=True)
    for k in ("step", "plain", "point", "tight", "step3", "step6"):
        ap.add_argument("--" + k, default=None)
    ap.add_argument("--point-free", dest="point_free", default=None, help="SIGMA: the free point-sampled re-solve on that member (needs --fm-dir)")
    ap.add_argument("--reint", action="append", default=[]); ap.add_argument("--Tc", action="append", default=[])
    ap.add_argument("--fm-dir", dest="fm_dir", default=None, help="the field module's dir (engine_fields_v16.py = v16d): the six-term on"); ap.add_argument("--tau-rs", dest="tau_rs", type=float, default=0.0625)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sp = importlib.util.spec_from_file_location("tolcmp_v3", a.tolcmp); TC = importlib.util.module_from_spec(sp); sp.loader.exec_module(TC)
    rs = a.rs
    M = {}
    for s in a.member:
        sg, spec = kv(s); d, t = GC.split_spec(spec); J, F = GC.load_run(d, t); M[sg] = {"tag": t, "J": J, "F": F}
    S = sorted(M)
    out = {"members": {str(s): M[s]["tag"] for s in S}}
    print("== gate_triple_read_v1: r_s/a %g; members sigma/a %s" % (rs, ", ".join("%g: %s" % (s, M[s]["tag"]) for s in S)))
    for s in S:
        M[s]["b"] = GC.birth(M[s]["J"], rs)
    # 0. vacuum ell control FIRST
    print("\n-- 0. the vacuum ell control (FIRST; reviewer B/requirements sec. 2): the sea against its continued reference, eps_pairs / J_pairs")
    RB = {s: GC.recs_by_T(M[s]["J"], rs) for s in S}; IB = {s: GC.rows_by_T(M[s]["F"], rs) for s in S}
    common = sorted(set.intersection(*[set(RB[s]) for s in S]))
    pick = [t for t in common if abs(t * 4 - round(t * 4)) < 1e-9]   # every quarter r_s
    vac = []
    print("   T/rs | E_pairs_shells per member | sum eps_pairs on r <= 2R per member | max|J_pairs| | d12/d23 (E_pairs)")
    for t in pick:
        ep = [RB[s][t].get("E_pairs_shells") for s in S]
        se, mj = [], []
        for s in S:
            F = M[s]["F"]
            if F is None or t not in IB[s]:
                se.append(None); mj.append(None); continue
            Rg = np.asarray(F["R"], float)
            rr = 2.0 * float(RB[s][t].get("R_t") or (M[s]["b"] or {}).get("R_birth") or (float(M[s]["J"]["head"].get("R0", 4 * rs))))
            sel = Rg <= rr
            se.append(float(np.sum(np.asarray(F["eps_pairs"][IB[s][t]], float)[sel]))); mj.append(float(np.max(np.abs(np.asarray(F["J_pairs"][IB[s][t]], float)))))
        rat = None
        if len(S) == 3 and None not in ep:
            d12, d23 = ep[1] - ep[0], ep[2] - ep[1]
            rat = (d12 / d23) if d23 != 0 else None
        vac.append((t, ep, se, mj, rat))
        print("   %7.3f | %s | %s | %s | %s" % (t, " ".join(GC.fmt(x, "%.4e") for x in ep), " ".join(GC.fmt(x, "%.4e") for x in se),
                                             " ".join(GC.fmt(x, "%.2e") for x in mj), GC.fmt(rat, "%.3f")))
    if len(S) == 3:
        print("   sigma^2 expectation for d12/d23: %.3f" % ((S[1] ** 2 - S[0] ** 2) / (S[2] ** 2 - S[1] ** 2)))
    out["vacuum"] = vac
    # 1. certification
    print("\n-- 1. certification (reviewer B): kept residual against the ceiling 1e-3 and the target 1e-5")
    cert = {}
    for s in S:
        J = M[s]["J"]; b = M[s]["b"]; tw = (b["T_birth"] - 1.0) if b else None
        kr = [(r["T"] / rs, GC.kept(r)) for r in J["records"] if r.get("E") is not None]
        allk = [k[0] for _, k in kr if k[0] is not None]; win = [k[0] for t, k in kr if k[0] is not None and tw is not None and t >= tw]
        unc = [t for t, k in kr if k[2] is False]
        c = {"n": len(kr), "max": max(allk) if allk else None, "median": float(np.median(allk)) if allk else None,
             "n_above_1e-3": sum(1 for x in allk if x > 1e-3), "n_above_1e-5": sum(1 for x in allk if x > 1e-5),
             "window_max": max(win) if win else None, "window_n_above_1e-5": sum(1 for x in win if x > 1e-5), "unconverged": unc,
             "end": J.get("end_v32")}
        cert[s] = c
        print("   sigma %g: %d records; kept residual max %s median %s; above 1e-3: %d, above 1e-5: %d; window [T_birth-1, end]: max %s, above 1e-5: %d; "
              "unconverged at %s; end %s" % (s, c["n"], GC.fmt(c["max"], "%.2e"), GC.fmt(c["median"], "%.2e"), c["n_above_1e-3"], c["n_above_1e-5"],
                                              GC.fmt(c["window_max"], "%.2e"), c["window_n_above_1e-5"], c["unconverged"] or "none",
                                              (c["end"] or {}).get("T_over_rs")))
    out["cert"] = cert
    # 2. charge drift
    print("\n-- 2. charge drift (reviewer A 3.7: 1 percent; reviewer B: 1e-3): max |E/E0 - 1| over [T_birth - 1, end] and over the whole history")
    drift = {}
    for s in S:
        J = M[s]["J"]; b = M[s]["b"]; tw = (b["T_birth"] - 1.0) if b else 0.0
        e = [(r["T"] / rs, abs(float(r["E_over_E0"]) - 1.0)) for r in J["records"] if r.get("E_over_E0") is not None]
        w = [x for t, x in e if t >= tw]; d_ = {"window": max(w) if w else None, "all": max(x for _, x in e) if e else None}
        drift[s] = d_
        print("   sigma %g: window %s (1e-2 %s, 1e-3 %s); whole history %s" % (s, GC.fmt(d_["window"], "%.3e"),
              "PASS" if (d_["window"] is not None and d_["window"] <= 1e-2) else "FAIL", "PASS" if (d_["window"] is not None and d_["window"] <= 1e-3) else "FAIL",
              GC.fmt(d_["all"], "%.3e")))
    out["drift"] = drift
    # 3. birth
    print("\n-- 3. birth (reviewer A 3.2): the zero of min(N - |v|), linear through the bracketing records")
    for s in S:
        b = M[s]["b"]
        if not b:
            print("   sigma %g: no crossing" % s); continue
        print("   sigma %g: bracket [%.4f, %.4f]; T_birth %.5f (quadratic %s; err %s); R_birth %s a = %s of 2GE0 (%.3f); sigma/R_birth %s" % (
            s, b["T_lo"], b["T_hi"], b["T_birth"], GC.fmt(b["T_quad"], "%.5f"), GC.fmt(b["T_err"], "%.1e"), GC.fmt(b["R_birth"], "%.3f"),
            GC.fmt(b["R_birth_over_2gE"], "%.5f"), b["two_gE0"], GC.fmt((s / b["R_birth"]) if b["R_birth"] else None, "%.4f")))
    out["birth"] = {str(s): M[s]["b"] for s in S}
    # 4. sigma^2 law
    print("\n-- 4. the sigma^2 law (reviewer A 3.3)")
    law = None
    if len(S) == 3 and all(M[s]["b"] for s in S):
        x = [(s / rs) ** 2 for s in S]; Tb = [M[s]["b"]["T_birth"] for s in S]; Rb = [M[s]["b"]["R_birth_over_2gE"] for s in S]
        cT = (Tb[2] - Tb[0]) / (x[2] - x[0]); T0 = Tb[0] - cT * x[0]
        cR = (Rb[2] - Rb[0]) / (x[2] - x[0]); R0 = Rb[0] - cR * x[0]
        dT = Tb[1] - (T0 + cT * x[1]); dR = Rb[1] - (R0 + cR * x[1])
        law = {"c_T": cT, "T_0": T0, "c_R": cR, "R_0": R0, "middle_dT": dT, "middle_dR": dR,
               "middle_clock_ok": abs(dT) < a.tau_rs / 2, "middle_radius_ok": abs(dR) < 0.01,
               "T0_within_0.5": all(abs(T0 - t) <= 0.5 for t in Tb), "R0_within_10pct": all(abs(R0 - r) <= 0.1 * abs(r) for r in Rb)}
        print("   T_birth = T_0 + c_T (sigma/r_s)^2 through sigma %g and %g: T_0 %.5f, c_T %.4f; the middle member departs by %.3e r_s (line tau/2 = %.4f: %s)" % (
            S[0], S[2], T0, cT, dT, a.tau_rs / 2, "PASS" if law["middle_clock_ok"] else "FAIL"))
        print("   R_birth/2GE0 = R_0 + c_R (sigma/r_s)^2: R_0 %.5f, c_R %.4f (the kernel shift 9/4 = 2.25 beside); the middle departs by %.3e of 2GE0 (line 0.01: %s)" % (
            R0, cR, dR, "PASS" if law["middle_radius_ok"] else "FAIL"))
        print("   T_0 within 0.5 r_s of every member: %s; R_0 within 10 percent of every member: %s" % (law["T0_within_0.5"], law["R0_within_10pct"]))
    else:
        print("   not readable (needs three members with a birth)")
    out["law"] = law
    # 5. matched-clock radii
    print("\n-- 5. matched-clock radii (reviewer A 3.4): R_t at the clocks where every member has a root")
    rad = []
    for t in common:
        Rt = [RB[s][t].get("R_t") for s in S]
        if None in Rt:
            continue
        rat = ((Rt[1] - Rt[0]) / (Rt[2] - Rt[1])) if (len(S) == 3 and Rt[2] != Rt[1]) else None
        spread = (max(Rt) - min(Rt)) / np.mean(Rt)
        rad.append((t, Rt, rat, spread))
    for t, Rt, rat, spread in rad[:12] + ([("...",)] if len(rad) > 24 else []) + (rad[-12:] if len(rad) > 24 else rad[12:]):
        if t == "...":
            print("   ..."); continue
        print("   %7.4f  R_t %s  d12/d23 %s  spread %.4f" % (t, " ".join("%.3f" % r for r in Rt), GC.fmt(rat, "%.3f"), spread))
    out["radii"] = rad
    # 6. gate profile
    print("\n-- 6. the gate profile (reviewer A 3.1): sigma ||K||_op over r <= 2R_t (2R_birth before the root), T_gate, the window")
    six = GC.load_sixterm(a.fm_dir)
    Tc = dict(kv(s) for s in a.Tc)
    gates = {}
    for s in S:
        g = GC.gate_profile(M[s]["J"], M[s]["F"], rs, s, M[s]["b"], sixterm=six)
        gates[s] = g
        pred = (float(Tc[s]) - (20.0 / 3.0) * s / rs) if s in Tc else None
        print("   sigma %g: T_gate %s, window %s r_s; caustic law T_c - (20/3) sigma/r_s %s; six-term %s; rows %d" % (
            s, GC.fmt(g["T_gate"], "%.4f"), GC.fmt(g["window"], "%.4f"), GC.fmt(pred, "%.4f"), "code of record, nodes r >= sigma" if g["sixterm"] else "OPEN (no --fm-dir)", len(g["rows"])))
        for row in g["rows"][:: max(1, len(g["rows"]) // 8)]:
            print("      T %.4f r_max %.1f sigma||K|| %.4f at r %.1f sigma|H| %.4f six (nodes r >= sigma) %s at r %s [%s] | r < sigma (reported) %s at r %s; R_t %s" % (
                row[0], row[1], row[2], row[3], row[4], GC.fmt(row[5], "%.4f"), GC.fmt(row[7], "%.1f"), row[8], GC.fmt(row[9], "%.4f"), GC.fmt(row[10], "%.2f"), GC.fmt(row[6], "%.2f")))
    out["gate"] = {str(s): gates[s] for s in S}
    # 7. step control
    print("\n-- 7. the step control (reviewer A 3.6)")
    if a.step:
        d, t = GC.split_spec(a.step); Js, Fs = GC.load_run(d, t); bs = GC.birth(GC.own(Js), rs)
        mid = S[1] if len(S) == 3 else S[0]
        if bs and M[mid]["b"] and all(M[s]["b"] for s in S[:2]):
            dstep = bs["T_birth"] - M[mid]["b"]["T_birth"]; d48 = abs(M[S[0]]["b"]["T_birth"] - M[S[1]]["b"]["T_birth"])
            print("   T_birth(tau/32) - T_birth(tau/16) at sigma %g: %.4e r_s; |T_birth(%g) - T_birth(%g)| = %.4e: %s" % (
                mid, dstep, S[0], S[1], d48, "smaller (PASS)" if abs(dstep) < d48 else "NOT smaller (the triple is redone at tau/32)"))
            print("   R_birth(tau/32) - R_birth(tau/16): %s a" % GC.fmt((bs["R_birth"] - M[mid]["b"]["R_birth"]) if (bs["R_birth"] and M[mid]["b"]["R_birth"]) else None, "%.3e"))
            out["step"] = {"dT": dstep, "d48": d48, "birth_tau32": bs}
        else:
            print("   no birth in the step control or the members")
    else:
        print("   not given")
    # 7b. the step control by local error (ReviewA_resview_v1 "better design")
    print("\n-- 7b. the local step error from the 8a member's T 3 and T 6 slices (two tau/2 records each through the view, against the parent's next record)")
    mid = S[1] if len(S) == 3 else S[0]
    loc = []
    for lab in ("step3", "step6"):
        spec = getattr(a, lab)
        if not spec:
            continue
        d, t = GC.split_spec(spec); Jv, _ = GC.load_run(d, t); Ov = GC.recs_by_T(GC.own(Jv), rs)
        for tt in sorted(Ov):
            if tt in RB[mid]:
                x, y = Ov[tt], RB[mid][tt]
                dm = abs(float(x["min_N_minus_absv"]) - float(y["min_N_minus_absv"])); dE = abs(float(x["E_over_E0"]) - float(y["E_over_E0"]))
                dRc = abs(float(x.get("R_c") or 0) - float(y.get("R_c") or 0))
                loc.append((lab, tt, dm, dE, dRc))
                print("   %s: at T/rs %.4f the tau/2 pair against the parent's step: |d min(N-|v|)| %.3e, |d E/E0| %.3e, |d R_c| %.3e a" % (lab, tt, dm, dE, dRc))
    if loc and M[mid]["b"]:
        b = M[mid]["b"]; tau_rs = a.tau_rs
        nst = b["T_birth"] / tau_rs; dmmax = max(q[2] for q in loc)
        slope = abs(b["m_lo"] - b["m_hi"]) / (b["T_hi"] - b["T_lo"])
        dTb = nst * dmmax / slope if slope > 0 else None
        print("   accumulated (no amplification): %d steps to T_birth x %.3e = %.3e in min(N-|v|); over dm/dT %.3e at birth -> delta T_birth %s r_s (line 0.01: %s; "
              "the single-step error is about twice the tau/2 pair's difference for a first-order step, included as printed, not doubled)" % (
                  nst, dmmax, nst * dmmax, slope, GC.fmt(dTb, "%.3e"), "the restarted control STANDS" if (dTb is not None and dTb < 0.01) else "the fresh 448 records"))
        out["step_local"] = {"rows": loc, "dT_birth_bound": dTb}
    # 8. two solvers
    print("\n-- 8. two solvers / the independent route (reviewer A 3.8; reviewer B): plain mix (+ ivp fields) against the Newton member, record floor")
    if a.plain:
        d, t = GC.split_spec(a.plain); Jp, Fp = GC.load_run(d, t); mid = S[1] if len(S) == 3 else S[0]
        start = (Jp["head"].get("resume_slice_v34") or {}).get("T_over_rs")
        v, w, n, fl = record_floor(TC, Jp, Fp, M[mid]["J"], M[mid]["F"], rs, t_from=start)
        bp = GC.birth(GC.own(Jp), rs)
        print("   plain against sigma %g Newton after T/rs %s: %s over %d clocks; worst %s; lines %s" % (mid, start, v, n, json.dumps(w), json.dumps(TC.C_LINES)))
        if fl:
            print("   failing records (first 8): %s" % fl[:8])
        if bp and M[mid]["b"]:
            print("   birth plain %.5f vs Newton %.5f (dT %.3e); R %s vs %s" % (bp["T_birth"], M[mid]["b"]["T_birth"], bp["T_birth"] - M[mid]["b"]["T_birth"], GC.fmt(bp["R_birth"], "%.3f"), GC.fmt(M[mid]["b"]["R_birth"], "%.3f")))
        out["plain"] = {"verdict": v, "worst": w, "n": n, "fails": fl, "birth": bp}
    else:
        print("   not given")
    # 9. point-sampled
    print("\n-- 9a. the point-sampled control, FREE version (ReviewA_resview_v1): fields re-solved from the stored shell totals, point-sampled at the sites, against the applied (averaged) site fields")
    if a.point_free and six is not None:
        sg = float(a.point_free); b = M[sg]["b"]
        rows = GC.point_free(M[sg]["J"], M[sg]["F"], rs, six, (b["T_birth"] - 1.0) if b else 0.0, 1e9)
        for q in rows:
            print("   %7.4f  min(N-|v|) averaged %.4e at %.1f, point %.4e at %.1f (d %.2e); outer applied zero %s / %s (d %s); max|dN| %.2e max|dv| %.2e" % (
                q[0], q[1], q[2], q[3], q[4], q[3] - q[1], GC.fmt(q[5], "%.3f"), GC.fmt(q[6], "%.3f"), GC.fmt((q[6] - q[5]) if (q[5] is not None and q[6] is not None) else None, "%.2e"), q[7], q[8]))
        if rows:
            dd = max(abs(q[3] - q[1]) for q in rows); dz = max([abs(q[6] - q[5]) for q in rows if q[5] is not None and q[6] is not None] or [0.0])
            print("   largest shift: depth %.2e (floor 4.6e-8), root %.2e a (floor 3.4e-6): %s" % (dd, dz, "ABOVE the floor: the evolving hybrid restart (C_point) is called for" if (dd > 4.6e-8 or dz > 3.4e-6) else "below the floor: no evolving restart"))
            out["point_free"] = rows
    else:
        print("   not given (--point-free SIGMA with --fm-dir)")
    print("\n-- 9. the point-sampled control at the root, evolving hybrid (site fields dense against averaged)")
    if a.point:
        d, t = GC.split_spec(a.point); Jq, Fq = GC.load_run(d, t); mid = S[1] if len(S) == 3 else S[0]
        Rq = GC.recs_by_T(Jq, rs); start = (Jq["head"].get("resume_slice_v34") or {}).get("T_over_rs") or -1
        for tt in sorted(t_ for t_ in Rq if t_ > start + 1e-9 and t_ in RB[mid]):
            x, y = Rq[tt], RB[mid][tt]
            print("   %7.4f  R_t dense %s averaged %s (d %s); min(N-|v|) %s / %s; (sigma/R_t)^2 %s" % (
                tt, GC.fmt(x.get("R_t"), "%.3f"), GC.fmt(y.get("R_t"), "%.3f"), GC.fmt((x["R_t"] - y["R_t"]) if (x.get("R_t") and y.get("R_t")) else None, "%.3e"),
                GC.fmt(x.get("min_N_minus_absv"), "%.4e"), GC.fmt(y.get("min_N_minus_absv"), "%.4e"), GC.fmt((mid / y["R_t"]) ** 2 if y.get("R_t") else None, "%.2e")))
        bq = GC.birth(GC.own(Jq), rs)
        # ReviewA_resview_v1: the hybrid's no-transient check on its first three own records
        Ro = [r for r in GC.own(Jq)["records"] if r.get("E_over_E0") is not None][:3]
        Rm = [r for r in M[mid]["J"]["records"] if r.get("balance_d_E_minus_Hin_dT") is not None and M[mid]["b"] and r["T"] / rs >= M[mid]["b"]["T_birth"] - 1.0]
        if Ro and Rm:
            dE = [abs(r["E_over_E0"] - y["E_over_E0"]) for r, y in zip(Ro[1:], Ro[:-1])]; dEm = float(np.median([abs(r["E_over_E0"] - y["E_over_E0"]) for r, y in zip(Rm[1:], Rm[:-1])]))
            bal = [abs(r.get("balance_d_E_minus_Hin_dT") or 0.0) for r in Ro]; balm = float(np.median([abs(r["balance_d_E_minus_Hin_dT"]) for r in Rm]))
            print("   no-transient check, first 3 own records: |d E/E0| per record %s against the member window's median %.2e; |d(E - H_in)/dT| %s against %.2e: %s "
                  "(the evolution-row residual itself is read by the rows readout)" % (["%.2e" % x for x in dE], dEm, ["%.2e" % x for x in bal], balm,
                  "no transient" if (all(x <= dEm for x in dE) and all(x <= balm for x in bal)) else "TRANSIENT above the median: the hybrid's birth does not count"))
        if bq and M[mid]["b"]:
            print("   birth dense %.5f vs averaged %.5f; R %s vs %s" % (bq["T_birth"], M[mid]["b"]["T_birth"], GC.fmt(bq["R_birth"], "%.3f"), GC.fmt(M[mid]["b"]["R_birth"], "%.3f")))
    else:
        print("   not given")
    # 10. tightening
    print("\n-- 10. reviewer B's tightening check (stop rule 100x tighter): birth shifts against 0.05 r_s and 1 percent")
    if a.tight:
        d, t = GC.split_spec(a.tight); Jt, _ = GC.load_run(d, t); bt = GC.birth(GC.own(Jt), rs); mid = S[1] if len(S) == 3 else S[0]
        if bt and M[mid]["b"]:
            dT = bt["T_birth"] - M[mid]["b"]["T_birth"]; dR = (bt["R_birth"] - M[mid]["b"]["R_birth"]) / M[mid]["b"]["R_birth"]
            print("   dT_birth %.3e r_s (%s), dR_birth/R %.3e (%s)" % (dT, "PASS" if abs(dT) < 0.05 else "FAIL", dR, "PASS" if abs(dR) < 0.01 else "FAIL"))
    else:
        print("   not given")
    # 11. local re-integration
    print("\n-- 11. reviewer B's local re-integration (records every tau/32 from the last slice before birth)")
    for s in a.reint:
        sg, spec = kv(s); d, t = GC.split_spec(spec); Jr, _ = GC.load_run(d, t); br = GC.birth(GC.own(Jr), rs)
        if br and sg in M and M[sg]["b"]:
            b0 = M[sg]["b"]
            print("   sigma %g: T_birth re-integrated %.5f vs interpolated %.5f (d %.3e; the interpolation's err %s); R %s vs %s" % (
                sg, br["T_birth"], b0["T_birth"], br["T_birth"] - b0["T_birth"], GC.fmt(b0["T_err"], "%.1e"), GC.fmt(br["R_birth"], "%.3f"), GC.fmt(b0["R_birth"], "%.3f")))
    if not a.reint:
        print("   not given")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, default=str)
        print("\nwritten %s" % a.out)


if __name__ == "__main__":
    main()
