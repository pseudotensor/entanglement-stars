#!/usr/bin/env python
"""engine_fold_birth_v2.py (builder analyst; engine spec 37.3 Ruled (1): "engine_fold_birth_v2 (new; v1 plus): the column
M(R_t)/E = R_t/(2gE) per trapped record; ell/(2gE) per member at its release charge and per trapped record (c); the roots with their signed
slopes exported as a json beside the md; the production's tau 8, 4, 2 histories and the sigma/a 8 member in the same tables (the builder
states which v1 already covers)"; inbox #31 S3 (a) to (c): the first zero's branch, the birth event by a joint fit of the double root
F = F_r = 0 between the last untrapped and the first trapped record). Record-only.

Version 1 (engine_fold_birth_v1.py) with:
(1) the runs read either from a records data npz (version 1's route: the sigma/a 4 and 8 members, the tau 8 production) or from a run's own
    fields npz and handoff npz (the tau 4 and tau 2 production histories, which have no records data npz): the shell totals eps_tot, J_tot
    and the applied fields from the fields npz, the per-shell stresses from the handoff's face fluxes and centrifugal forces by the records
    composer's own rule (engine_collapse_records_v2.shell_stress: S_r at a shell the mean of its faces, F/(4 pi R_f^2), shell 0 its outer
    face; S_t = f/(8 pi R)), dust plus pairs, so that these runs carry the stresses a records data npz would; the field module by the run's script (v19: engine_fields_v14, v25 and v27: v16) or, for a records data
    npz with segments, by segment (version 1's map); the route's check printed: on the tau 8 production's v19 segment the handoff route's
    stresses against the records data npz's at three records;
(2) per trapped record the columns M(R_t)/E with M = R_t w^2/(2G) on the re-solved dense fields, R_t/(2 g E), and ell/(2 g E); per member
    ell/(2 g E0) at its release charge (the head's E0), for the runs read and for the ladder's other members from their jsons (--extra);
(3) the first zero (outward from the centre) and its branch per record: inner where d(N + v)/dr < 0, outer where > 0;
(4) the birth event by the joint fit of the double root F = N + v = 0, F_r = 0: F(T, r) interpolated in T at every r between the last
    untrapped record and the first trapped one (linear) and through the record before them (quadratic), on the re-solved dense fields on
    the grid r = 0 to --r-fit at spacing ell/64, the grid's minimum refined by bounded minimization of the continuous interpolated profile
    within two cells; the clock where min_r F(T, r) reaches zero by bisection and R_* the minimizing radius there;
    beside version 1's clocks (m interpolated linearly and quadratically);
(5) a json beside the md: per run and per record from the record before the first birth to --t-max, every zero with its signed slope, the
    first zero's branch, M(R_t)/E, R_t/(2gE), ell/(2gE); the events; the joint-fit clocks.
Everything else is version 1's (the growth identity of P20 eq. 16 with the rows' D_M, the kinematic form, D_M direct; M/(r/2G) - 1 at the
roots; P20 eq. 17's two sides). Every number is printed by the script. Args: --run label=json=data_npz (version 1's form) or
label=json=fields_npz=handoff_npz (repeated); --extra label=json (repeated; ell/(2gE0) only); --t-max --r-fit --md --out-json."""
import argparse
import importlib
import json
import math
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_collapse_records_v2 as CR  # noqa: E402
import engine_fold_birth_v1 as V1  # noqa: E402
import engine_rows_v2 as RW  # noqa: E402

SCRIPT_FM = {"engine_collapse_v17": "engine_fields_v12", "engine_collapse_v19": "engine_fields_v14", "engine_collapse_v24": "engine_fields_v16",
             "engine_collapse_v25": "engine_fields_v16", "engine_collapse_v26": "engine_fields_v16", "engine_collapse_v27": "engine_fields_v16",
             "engine_collapse_v29": "engine_fields_v16", "engine_collapse_v30": "engine_fields_v16", "engine_collapse_v31": "engine_fields_v16"}


def shells_from_handoff(H, k, R):
    """the per-shell S_r and S_t (dust plus pairs) from a handoff's face fluxes and centrifugal forces by the records composer's own rule
    (engine_collapse_records_v2.shell_stress: S_r at a shell the mean of its two faces F/(4 pi R_f^2), shell 0 its outer face; S_t =
    f/(8 pi R)), so that a run read this way carries the stresses a records data npz would."""
    Srd, Std = CR.shell_stress(H, k, "dust", R); Srp, Stp = CR.shell_stress(H, k, "pairs", R)
    return Srd + Srp, Std + Stp


class RunArrays:
    """a records-data-npz-like view of a run's fields npz plus handoff npz."""

    def __init__(self, fields_npz, handoff_npz, g, ell, rs):
        Zf = np.load(fields_npz, allow_pickle=True); Zh = np.load(handoff_npz, allow_pickle=True)
        self.files = ["R", "T", "g", "ell", "rs", "eps_tot", "J_tot", "N_applied", "v_applied", "S_r_dust", "S_r_pairs", "S_t_dust", "S_t_pairs"]
        T = np.asarray(Zf["T"], float); Th = np.asarray(Zh["T"], float)
        if len(T) != len(Th) or float(np.max(np.abs(T - Th))) > 1e-9:
            raise SystemExit("the fields npz and the handoff npz do not align (%d against %d)" % (len(T), len(Th)))
        R = np.asarray(Zf["R"], float)
        self.d = {"R": R, "T": T, "g": np.float64(g), "ell": np.float64(ell), "rs": np.float64(rs),
                  "eps_tot": np.asarray(Zf["eps_tot"], float), "J_tot": np.asarray(Zf["J_tot"], float),
                  "N_applied": np.asarray(Zf["N_applied"], float), "v_applied": np.asarray(Zf["v_applied"], float)}
        Hd = {key: np.asarray(Zh[key], float) for key in ("F_face_dust", "F_face_pairs", "fc_dust", "fc_pairs")}
        Sr = np.empty((len(T), len(R))); St = np.empty((len(T), len(R)))
        for k in range(len(T)):
            Sr[k], St[k] = shells_from_handoff(Hd, k, R)
        self.d.update({"S_r_dust": Sr, "S_r_pairs": np.zeros_like(Sr), "S_t_dust": St, "S_t_pairs": np.zeros_like(St)})

    def __getitem__(self, k):
        return self.d[k]


def joint_fit(FO, ks, T, ell, r_fit):
    """the double root F = F_r = 0 between the records ks = (k_before, k0, k1) (k_before may be None): F(T, r) linear in T through
    (k0, k1) and quadratic through (k_before, k0, k1) at every r; the clock where min_r F reaches zero and the minimizing radius."""
    kb, k0, k1 = ks
    rr = np.arange(0.0, r_fit + 1e-9, ell / 64.0)
    Fv = {}; Fc = {}
    for k in [x for x in ks if x is not None]:
        F = FO(k)
        if F["failed"]:
            return {"failed": True}
        Fv[k] = np.asarray(F["N"](rr), float) + np.asarray(F["v"](rr), float)
        Fc[k] = (lambda x, F=F: float(np.asarray(F["N"](np.atleast_1d(x)), float)[0] + np.asarray(F["v"](np.atleast_1d(x)), float)[0]))
    out = {}
    for kind in ("linear", "quadratic"):
        if kind == "linear":
            nodes = [k0, k1]
        else:
            if kb is None:
                out[kind] = None; continue
            nodes = [kb, k0, k1]
        ts = [T[k] for k in nodes]

        def wts(t):
            w = []
            for i, ti in enumerate(ts):
                li = 1.0
                for j, tj in enumerate(ts):
                    if j != i:
                        li *= (t - tj) / (ti - tj)
                w.append(li)
            return w

        def Fat(t):
            return sum(wi * Fv[k] for wi, k in zip(wts(t), nodes))

        def gmin(t):
            """min over r of F(t, r): the grid's minimum refined by bounded minimization of the continuous profile within two cells."""
            f = Fat(t); i = int(np.argmin(f))
            w = wts(t)
            fc = lambda x: sum(wi * Fc[k](x) for wi, k in zip(w, nodes))
            a_, b_ = rr[max(i - 2, 0)], rr[min(i + 2, len(rr) - 1)]
            res = minimize_scalar(fc, bounds=(a_, b_), method="bounded", options={"xatol": 1e-9})
            return float(res.fun), float(res.x), f
        lo, hi = T[k0], T[k1]
        glo, _, _ = gmin(lo); ghi, _, _ = gmin(hi)
        if not (glo > 0.0 > ghi):
            out[kind] = {"bracket": [glo, ghi], "T": None}; continue
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            gm, _, _ = gmin(mid)
            if gm > 0.0:
                lo = mid
            else:
                hi = mid
        tstar = 0.5 * (lo + hi)
        fmin, rstar, _ = gmin(tstar)
        out[kind] = {"T": tstar, "R": rstar, "F_min_at_T": fmin}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label=json=data_npz or label=json=fields_npz=handoff_npz")
    ap.add_argument("--extra", action="append", default=[], help="label=json (ell/(2 g E0) only)")
    ap.add_argument("--fm-segments", default="v17 (replayed)=engine_fields_v12;v19 (native)=engine_fields_v14")
    ap.add_argument("--t-max", type=float, default=20.0)
    ap.add_argument("--r-fit", type=float, default=120.0)
    ap.add_argument("--hfrac", type=float, default=32.0)
    ap.add_argument("--check-route", default=None, help="label=data_npz=handoff_npz=k1,k2,k3: the handoff route's stresses against the data npz's at those records")
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    t0 = time.time()
    L = ["## The fold, the birth event, the roots and the growth identity on the stored records, version 2 (engine_fold_birth_v2.py; engine spec 37.3 (1), inbox #31 S3 (a) to (c), WEB40F P21 (2) and P20; record-only)", "",
         "- version 1's definitions and tables (engine_fold_birth_v1.md) with: per trapped record M(R_t)/E (M = R_t w^2/(2G) on the re-solved dense fields), R_t/(2gE) and ell/(2gE); per record the first zero (outward from the centre) and its branch (inner where d(N + v)/dr < 0, outer where > 0); the birth by the joint fit of the double root F = N + v = 0 = F_r (F interpolated in T at every r on the re-solved dense fields, r from 0 to %g at ell/64: linear between the last untrapped and the first trapped record, quadratic with the record before; the grid's minimum refined on the continuous profile; the clock where min_r F reaches zero by bisection, R_* the minimizing radius) beside version 1's two clocks of m; ell/(2 g E0) per member at its release charge. The tau 4 and tau 2 histories are read from their fields and handoff npz (no records data npz exists for them): the per-shell stresses by the records composer's rule (engine_collapse_records_v2.shell_stress) from the handoff's face fluxes and forces (the route's check below)." % a.r_fit,
         "- the json beside this md holds per run and record every zero with its signed slope, the first zero's branch and the columns above.", ""]
    store = {}
    rel = []
    # the route's check
    if a.check_route:
        lab, dp, hp, ksx = a.check_route.split("=", 3)
        Zd = np.load(dp, allow_pickle=True); Zh = np.load(hp, allow_pickle=True)
        Th = np.asarray(Zh["T"], float); Rh = np.asarray(Zh["R"], float)
        msg = []
        for kk in [int(x) for x in ksx.split(",")]:
            T_ = float(Zd["T"][kk]); i = int(np.argmin(np.abs(Th - T_)))
            if abs(Th[i] - T_) > 1e-9:
                msg.append("T/rs %.3f: no handoff record" % (T_ / float(Zd["rs"]))); continue
            Sr_h, St_h = shells_from_handoff(Zh, i, Rh)
            srd = np.asarray(Zd["S_r_dust"][kk], float) + np.asarray(Zd["S_r_pairs"][kk], float)
            std = np.asarray(Zd["S_t_dust"][kk], float) + np.asarray(Zd["S_t_pairs"][kk], float)
            dr = float(np.max(np.abs(Sr_h - srd))) / max(float(np.max(np.abs(srd))), 1e-300)
            dt = float(np.max(np.abs(St_h - std))) / max(float(np.max(np.abs(std))), 1e-300)
            msg.append("T/rs %.3f: S_r %.2e, S_t %.2e" % (T_ / float(Zd["rs"]), dr, dt))
        L += ["- the handoff route's check (%s, %s against %s): the largest |difference| over the shells relative to the largest |stress|: %s." % (lab, os.path.basename(hp), os.path.basename(dp), "; ".join(msg)), ""]
    for spec in a.run:
        parts = spec.split("=")
        lab, jpath = parts[0], parts[1]
        J, R = V1.load_records(jpath)
        g = float(J["args"]["g"]); ell = float(J["args"]["ell"]); rs = float(J["args"]["rs"]); E0 = float(J["head"]["E0"])
        if len(parts) == 3:
            Z = np.load(parts[2], allow_pickle=True); src = os.path.basename(parts[2])
            fm_default = SCRIPT_FM.get(str(J.get("script", "")).replace(".py", ""), "engine_fields_v16")
        else:
            Z = RunArrays(parts[2], parts[3], g, ell, rs); src = "%s + %s" % (os.path.basename(parts[2]), os.path.basename(parts[3]))
            fm_default = SCRIPT_FM[str(J.get("script", "")).replace(".py", "")]
        FO = V1.FieldsOf(Z, a.fm_segments, fm_default)
        Tz = np.asarray(Z["T"], float)
        if len(Tz) != len(R) or float(np.max(np.abs(Tz / rs - np.array([r["T_over_rs"] for r in R])))) > 1e-9:
            L.append("### %s: the json's records and the npz's do not align (%d against %d); skipped" % (lab, len(R), len(Tz))); continue
        rel.append((lab, ell, E0, g, J.get("script")))
        ev = V1.crossing_events(R, a.t_max)
        T = np.asarray([r["T"] for r in R], float)
        L.append("### %s (%s, tau %g, ell %g; %s; the field module %s; E0 %.11f, ell/(2 g E0) %.6f)" % (
            lab, os.path.basename(jpath), J["args"]["tau"], ell, src, (a.fm_segments if FO.seg_names else fm_default), E0, ell / (2.0 * g * E0)))
        L.append("")
        L.append("| event | between T/rs | m at the two records | T/rs (m linear) | T/rs (m quadratic) | T/rs (joint fit, linear) | (joint fit, quadratic) | R_* (m linear) | R_* (joint, linear) | (joint, quadratic) | zeros after |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        jf = {}
        for e in ev:
            kk = [i for i, r in enumerate(R) if abs(r["T_over_rs"] - e["records"][0]) < 1e-9 or abs(r["T_over_rs"] - e["records"][1]) < 1e-9]
            j = None
            if e["kind"] == "birth" and len(kk) == 2:
                j = joint_fit(FO, (kk[0] - 1 if kk[0] >= 1 else None, kk[0], kk[1]), T, ell, a.r_fit)
                jf[e["records"][0]] = j
            jl = (j or {}).get("linear") or {}; jq = (j or {}).get("quadratic") or {}
            L.append("| %s | %.5f, %.5f | %+.4f, %+.4f | %.5f | %s | %s | %s | %s | %s | %s | %s |" % (
                e["kind"], e["records"][0], e["records"][1], e["m"][0], e["m"][1], e["T_lin"], V1.fmt(e["T_quad"], "%.5f"),
                V1.fmt(jl.get("T") / rs if jl.get("T") is not None else None, "%.5f"), V1.fmt(jq.get("T") / rs if jq.get("T") is not None else None, "%.5f"),
                V1.fmt(e["R_lin"], "%.3f"), V1.fmt(jl.get("R"), "%.3f"), V1.fmt(jq.get("R"), "%.3f"), ", ".join("%.3f" % x for x in e["zeros_after"]) or "none"))
        b0 = next((e for e in ev if e["kind"] == "birth"), None)
        if b0 is None:
            L += ["", "- no birth to T/rs %g" % a.t_max, ""]
            store[lab] = {"events": ev}
            continue
        k_start = next(i for i, r in enumerate(R) if abs(r["T_over_rs"] - b0["records"][0]) < 1e-9)
        ks = [k for k in range(k_start, len(R)) if R[k]["T_over_rs"] <= a.t_max + 1e-9]
        rows = []
        for k in ks:
            r = R[k]; z = V1.zeros_of(r)
            row = {"k": k, "T_over_rs": r["T_over_rs"], "nz": len(z), "zeros": z, "E": r["E"], "min": r["min_N_minus_absv"], "r_min": r["r_min_N_minus_absv"],
                   "stored_Rt_over_2gMS": r.get("R_t_over_2g_MS")}
            if len(z) >= 1 and FO(k)["failed"]:
                row["failed"] = True
            elif len(z) >= 1:
                F0 = FO(k)
                h = ell / a.hfrac
                sl = []
                for zz in z:
                    rz = np.atleast_1d(zz)
                    _, N1, _ = RW.deriv(lambda x: V1.at(F0["N"], x), rz, h)
                    _, v1, _ = RW.deriv(lambda x: V1.at(F0["v"], x), rz, h)
                    sl.append(float((N1 + v1)[0]))
                row["slopes"] = sl
                row["first_zero"] = z[0]; row["first_branch"] = "inner" if sl[0] < 0 else ("outer" if sl[0] > 0 else "tangent")
                Rt = z[-1]
                Nr = float(V1.at(F0["N"], Rt)[0]); vr = float(V1.at(F0["v"], Rt)[0]); w = vr / Nr
                M = Rt * w * w / (2.0 * g)
                row["M_Rt_dev"] = M / (Rt / (2.0 * g)) - 1.0
                row["M_Rt_over_E"] = M / r["E"]; row["Rt_over_2gE"] = Rt / (2.0 * g * r["E"]); row["ell_over_2gE"] = ell / (2.0 * g * r["E"])
                Na = float(np.interp(Rt, FO.R, F0["N_app"])); va = float(np.interp(Rt, FO.R, F0["v_app"]))
                row["M_Rt_applied_dev"] = (va / Na) ** 2 - 1.0
                row["eq17_lhs"] = 2.0 * g * r["E"] - Rt
                row["eq17_rhs"] = 2.0 * g * (r["E"] - M)
                if len(z) >= 2:
                    ri = z[0]
                    Ni = float(V1.at(F0["N"], ri)[0]); vi = float(V1.at(F0["v"], ri)[0]); wi = vi / Ni
                    row["M_rin_dev"] = (ri * wi * wi / (2.0 * g)) / (ri / (2.0 * g)) - 1.0
                for nm, idx in (("Rt", -1), ("rin", 0)):
                    if nm == "rin" and len(z) < 2:
                        continue
                    km, kp = k - 1, k + 1
                    if km < 0 or kp >= len(R) or FO(km)["failed"] or FO(kp)["failed"]:
                        continue
                    zm, zp = V1.zeros_of(R[km]), V1.zeros_of(R[kp])
                    gt = V1.growth_terms(FO, k, km, kp, T, z[idx], a.hfrac)
                    row["growth_" + nm] = gt
                    if len(zm) == len(z) == len(zp):
                        row["dRdT_" + nm] = (zp[idx] - zm[idx]) / (T[kp] - T[km])
            rows.append(row)
        L += ["", "#### %s: the roots, the mass at the roots and the release-charge ratios, per record from the record before the first birth to T/rs %g" % (lab, a.t_max), "",
              "| T/rs | zeros | first zero (branch) | r_in | R_t | min(N - abs v) at r | M(R_t)/E | R_t/(2gE) | ell/(2gE) | M/(r/2G) - 1 at r_in | at R_t | stored R_t/(2 g M_MS) | applied (w^2 - 1) at R_t | 2gE - R_t | 2g [E - M(R_t)] | slopes at the zeros |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for row in rows:
            z = row["zeros"]
            L.append("| %.5f | %d | %s | %s | %s | %+.4f at %.2f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                row["T_over_rs"], row["nz"], ("%.3f (%s)" % (row["first_zero"], row["first_branch"])) if row.get("first_zero") is not None else "-",
                V1.fmt(z[0], "%.3f") if len(z) >= 2 else "-", V1.fmt(z[-1], "%.3f") if z else "-", row["min"], row["r_min"],
                V1.fmt(row.get("M_Rt_over_E"), "%.6f"), V1.fmt(row.get("Rt_over_2gE"), "%.6f"), V1.fmt(row.get("ell_over_2gE"), "%.6f"),
                V1.fmt(row.get("M_rin_dev"), "%.1e"), V1.fmt(row.get("M_Rt_dev"), "%.1e"), V1.fmt(row.get("stored_Rt_over_2gMS"), "%.10f"),
                V1.fmt(row.get("M_Rt_applied_dev"), "%+.4e"), V1.fmt(row.get("eq17_lhs"), "%.4f"), V1.fmt(row.get("eq17_rhs"), "%.4f"),
                ", ".join("%+.3e" % s for s in row.get("slopes", [])) or "-"))
        for nm, title in (("Rt", "R_t (the outer root)"), ("rin", "r_in (the inner root)")):
            L += ["", "#### %s: the growth identity at %s (P20 eq. 16; version 1's columns)" % (lab, title), "",
                  "| T/rs | R | dR/dT (records, centred) | right side with the rows' D_M | kinematic 2G M_t/(1 - 2G M_r) | matter term | defect term | 1 - 2G M_r | D_M (rows) | D_M (direct, M_t + Fl) | rho | J | S | N |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for row in rows:
                gt = row.get("growth_" + nm)
                if not gt:
                    continue
                z = row["zeros"]; Rr = z[-1] if nm == "Rt" else z[0]
                L.append("| %.5f | %.3f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %.6f |" % (
                    row["T_over_rs"], Rr, V1.fmt(row.get("dRdT_" + nm), "%+.4e"), V1.fmt(gt["rhs_rows"], "%+.4e"), V1.fmt(gt["kinematic"], "%+.4e"),
                    V1.fmt(gt["matter"], "%+.4e"), V1.fmt(gt["defect_rows"], "%+.4e"), V1.fmt(gt["one_minus_2GMr"], "%+.4e"), V1.fmt(gt["DM_rows"], "%+.3e"),
                    V1.fmt(gt["DM_direct"], "%+.3e"), V1.fmt(gt["rho"], "%.3e"), V1.fmt(gt["J"], "%+.3e"), V1.fmt(gt["S"], "%+.3e"), gt["N"]))
        L.append("")
        tr = [x for x in rows if x.get("M_Rt_over_E") is not None]
        if tr:
            br = {}
            for x in tr:
                br[x["first_branch"]] = br.get(x["first_branch"], 0) + 1
            L.append("- %s: %d trapped records to T/rs %g; M(R_t)/E from %.6f to %.6f; |M(R_t)/E - R_t/(2gE)| at most %.1e; the first zero's branch: %s." % (
                lab, len(tr), a.t_max, min(x["M_Rt_over_E"] for x in tr), max(x["M_Rt_over_E"] for x in tr),
                max(abs(x["M_Rt_over_E"] - x["Rt_over_2gE"]) for x in tr), ", ".join("%s %d" % (k_, v_) for k_, v_ in sorted(br.items()))))
            L.append("")
        store[lab] = {"events": ev, "joint_fit": jf, "rows": rows, "ell": ell, "E0": E0, "g": g}
        print("%s done (%d records, %.0f s)" % (lab, len(rows), time.time() - t0), flush=True)
    for spec in a.extra:
        lab, jp = spec.split("=", 1)
        J = json.load(open(jp))
        rel.append((lab, float(J["args"]["ell"]), float(J["head"]["E0"]), float(J["args"]["g"]), J.get("script")))
    L += ["### ell/(2 g E0) per member at its release charge", "", "| member | script | ell | E0 | 2 g E0 | ell/(2 g E0) |", "|---|---|---|---|---|---|"]
    for lab, ell, E0, g, sc in rel:
        L.append("| %s | %s | %g | %.11f | %.6f | %.6f |" % (lab, sc, ell, E0, 2.0 * g * E0, ell / (2.0 * g * E0)))
    L += ["", "- every number above is printed by the script from the runs' jsons, records data npz or fields and handoff npz (the dense fields re-solved by each record's own field module); nothing is read or ruled."]
    open(a.md, "w").write("\n".join(L) + "\n")
    if a.out_json:
        slim = {}
        for lab, d in store.items():
            slim[lab] = {"events": d.get("events"), "joint_fit": d.get("joint_fit"), "ell": d.get("ell"), "E0": d.get("E0"), "g": d.get("g"),
                         "records": [{k_: x.get(k_) for k_ in ("T_over_rs", "nz", "zeros", "slopes", "first_zero", "first_branch", "M_Rt_over_E", "Rt_over_2gE",
                                                               "ell_over_2gE", "M_Rt_dev", "M_rin_dev", "E", "min", "r_min", "dRdT_Rt", "dRdT_rin", "growth_Rt", "growth_rin")}
                                     for x in (d.get("rows") or [])]}
        json.dump({"script": "engine_fold_birth_v2.py", "runs": slim, "release_ratios": [{"member": x[0], "ell": x[1], "E0": x[2], "g": x[3], "script": x[4],
                                                                                           "ell_over_2gE0": x[1] / (2.0 * x[3] * x[2])} for x in rel]},
                  open(a.out_json, "w"), default=float, indent=1)
    print("\n".join(L[:6])); print("done, %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
