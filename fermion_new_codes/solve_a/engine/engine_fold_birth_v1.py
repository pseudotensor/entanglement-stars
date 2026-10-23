#!/usr/bin/env python
"""engine_fold_birth_v1.py (builder analyst; engine spec 36.4 (2): "P21 (2) and P20: a record-only script on the stored records of the
ladder's members and the production: the birth event interpolated between records (T_* where min(N - |v|) crosses zero ...), both fold
roots per record, M(R_t, T) = R_t w^2/2G against R_t/2G with w = v/N, and the growth identity's two sides with the rows' mass-balance
defect; a table, nothing printed").

Per run (a collapse run's json, its records data npz with the stored shell totals and per-shell stresses, the field module of each record):
(1) the crossing events of m(T) = min_r (N - |v|) of the dense fields (the record's min_N_minus_absv): wherever m changes sign between two
    successive records, the event clock by linear interpolation between them (the lead's form; P21 eq. 23's event is F = 0 at the interior
    minimum F_r = 0) and by the quadratic through those two records and the one before (after, for a closing), and the radius R_* of the
    minimum (r_min_N_minus_absv) interpolated linearly to the linear clock; a birth is m going negative, a closing m going positive;
(2) per record from the record before the first birth to --t-max: every zero of N + v (the record's trapping_zeros), r_in (the first) and
    R_t (the last) when two or more, their signed slopes d(N + v)/dr from the re-solved dense fields;
(3) at r_in and R_t: M = r w^2/(2G), w = v/N, on the re-solved dense fields, against r/(2G) (M/(r/2G) - 1; zero at a root of N + v up to the
    root's tolerance and the re-solve's reproduction of the record), the record's own R_t/(2 g M_MS) beside it, the same M from the applied
    (site) fields' w linearly interpolated at R_t (the field the Hamiltonian sees; not a root of it), and P20 eq. 17's two sides
    2 g E - R_t and 2 g [E - M(R_t)];
(4) the growth identity, P20 eq. 16: dR/dT = [8 pi G R^2 N (rho - 2 J + S) + 2 G D_M] / (1 - 2 G M_r) at R = R_t and R = r_in, written with w
    exactly: the matter term -2 G Fl, Fl = 4 pi R^2 [v (rho + S) + N (1 + w^2) J] (P20 eq. 13's flux; it equals 8 pi G R^2 N (rho - 2 J + S)
    at w = -1), M_r = 4 pi R^2 (rho + w J), rho and J the solve's smoothed sources rho_ell and J_ell, S the smoothed radial stress s_r of the
    stored per-shell stresses (dust plus pairs) by engine_stress_v2.smooth_stress; D_M the rows' mass-balance defect, from the direct
    tangential evolution row of engine_rows_v2 (res_tt = d_t k_t - rhs_tt, k_t = v/(r N), d_t k_t centred on the neighbouring records at
    fixed r) as engine_certificate_v3 normalizes it, D_M = (R^2 v/G) res_tt/N; beside it the direct D_M = M_t + Fl (M_t centred at fixed r)
    and the kinematic form 2 G M_t/(1 - 2 G M_r) (the implicit-function identity of R = 2 G M(R, T)); the left side dR/dT the centred
    difference of the records' R_t (r_in) over the neighbouring records, printed only where the neighbours hold the same number of zeros.
The dense fields are re-solved from the stored shell totals by the run's own field module (engine_certificate_v3's route: v12 for the
production's replayed v17 segment, v14 for its v19 segment, v16 for engine_collapse_v27 runs); spatial derivatives by engine_rows_v2.deriv at
h = ell/--hfrac. Every number is printed by the script. Args: --run label=json=data_npz (repeated) --fm-segments --fm-default --t-max --md."""
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


def load_records(path):
    J = json.load(open(path))
    R = [r for r in J["records"] if r.get("E") is not None and "T_over_rs" in r]
    return J, R


def zeros_of(r):
    return list(r.get("trapping_zeros") or [])


def crossing_events(R, t_max=None):
    """the sign changes of m(T) = min(N - |v|) between successive records: linear and quadratic clocks, R_* at the linear clock."""
    ev = []
    for i in range(1, len(R)):
        a, b = R[i - 1], R[i]
        if t_max is not None and a["T_over_rs"] > t_max:
            break
        ma, mb = a.get("min_N_minus_absv"), b.get("min_N_minus_absv")
        if ma is None or mb is None or (ma > 0) == (mb > 0):
            continue
        kind = "birth" if ma > 0 >= mb else "closing"
        ta, tb = a["T_over_rs"], b["T_over_rs"]
        s = ma / (ma - mb)
        t_lin = ta + s * (tb - ta)
        ra, rb = a.get("r_min_N_minus_absv"), b.get("r_min_N_minus_absv")
        r_lin = ra + s * (rb - ra) if (ra is not None and rb is not None) else None
        # the quadratic through (i-2, i-1, i) for a birth, (i-1, i, i+1) for a closing (the third point on the side away from the crossing)
        j3 = i - 2 if kind == "birth" else i + 1
        t_q = None
        if 0 <= j3 < len(R) and R[j3].get("min_N_minus_absv") is not None:
            ts = np.array([R[j3]["T_over_rs"], ta, tb]); ms = np.array([R[j3]["min_N_minus_absv"], ma, mb])
            c = np.polyfit(ts, ms, 2)
            roots = [x.real for x in np.roots(c) if abs(x.imag) < 1e-12 and ta - 1e-12 <= x.real <= tb + 1e-12]
            t_q = float(roots[0]) if len(roots) == 1 else None
        ev.append({"kind": kind, "records": (ta, tb), "m": (ma, mb), "r_min": (ra, rb), "T_lin": t_lin, "R_lin": r_lin, "T_quad": t_q,
                   "zeros_after": zeros_of(b)})
    return ev


def first_birth(R):
    ev = crossing_events(R)
    return next((e for e in ev if e["kind"] == "birth"), None)


class FieldsOf:
    """per record the re-solved dense fields (the run's own module), the smoothed sources and the stored per-shell stresses."""

    def __init__(self, Z, fm_segments, fm_default, collar_policy="flag", support_rel=1e-14):
        self.Z = Z
        self.R = np.asarray(Z["R"], float); self.g = float(Z["g"]); self.ell = float(Z["ell"])
        self.seg_names = [str(x) for x in Z["segment_names"]] if "segment_names" in Z.files else None
        self.seg = np.asarray(Z["segment"], int) if "segment" in Z.files else None
        self.fmap = dict(x.split("=", 1) for x in fm_segments.split(";")) if fm_segments else {}
        self.fm_default = fm_default
        self.mods = {}
        self.cache = {}
        self.cp, self.sr = collar_policy, support_rel

    def module(self, k):
        name = self.fmap.get(self.seg_names[self.seg[k]], self.fm_default) if (self.seg_names and self.seg is not None) else self.fm_default
        if name not in self.mods:
            self.mods[name] = importlib.import_module(name)
        return self.mods[name], name

    def __call__(self, k):
        if k in self.cache:
            return self.cache[k]
        FM, name = self.module(k)
        e = np.asarray(self.Z["eps_tot"][k], float); Jt = np.asarray(self.Z["J_tot"][k], float)
        _, _, _, info = FM.solve_constraints_ivp(e, Jt, self.g, self.R, self.ell, collar_policy=self.cp, support_rel=self.sr)
        Sr = np.asarray(self.Z["S_r_dust"][k], float) + np.asarray(self.Z["S_r_pairs"][k], float)
        St = np.asarray(self.Z["S_t_dust"][k], float) + np.asarray(self.Z["S_t_pairs"][k], float)
        F = {"N": info["N_dense"], "v": info["v_dense"], "rho": info["rho_ell"], "J": info["J_ell"], "Sr": Sr, "St": St, "info": info, "fm": name,
             "failed": bool(info.get("failed")), "N_app": np.asarray(self.Z["N_applied"][k], float), "v_app": np.asarray(self.Z["v_applied"][k], float)}
        self.cache[k] = F
        for old in [x for x in self.cache if x < k - 2]:
            del self.cache[old]
        return F


def at(f, r):
    return np.asarray(f(np.atleast_1d(np.asarray(r, float))), float)


def growth_terms(FO, k, km, kp, T, r0, hfrac):
    """P20 eq. 16 at r0 on record k with the neighbours km, kp (centred time derivatives at fixed r)."""
    G = FO.g; ell = FO.ell; R = FO.R
    F0, Fm, Fp = FO(k), FO(km), FO(kp)
    h = ell / hfrac
    r = np.atleast_1d(float(r0))
    N, N1, _ = RW.deriv(lambda x: at(F0["N"], x), r, h)
    v, v1, _ = RW.deriv(lambda x: at(F0["v"], x), r, h)
    w = v / N
    rho = at(F0["rho"], r); J = at(F0["J"], r)
    sr, st, _ = STR2.smooth_stress(F0["Sr"], F0["St"], R, ell, r)
    sr = np.asarray(sr, float); st = np.asarray(st, float)
    Mr = 4.0 * math.pi * r ** 2 * (rho + w * J)
    Fl = 4.0 * math.pi * r ** 2 * (v * (rho + sr) + N * (1.0 + w ** 2) * J)
    dt = T[kp] - T[km]
    Mf = lambda F: r * (at(F["v"], r) / at(F["N"], r)) ** 2 / (2.0 * G)
    Mt = (Mf(Fp) - Mf(Fm)) / dt
    bb = RW.brackets(lambda x: at(F0["N"], x), lambda x: at(F0["v"], x), rho, sr, st, G, r, h)
    z = np.zeros(1)
    kt = lambda F: RW.brackets(lambda x: at(F["N"], x), lambda x: at(F["v"], x), z, z, z, G, r, h)["k_t"]
    dkt = (kt(Fp) - kt(Fm)) / dt
    res_tt = dkt - bb["rhs_tt"]
    DM_rows = r ** 2 * v / G * res_tt / N
    DM_direct = Mt + Fl
    den = 1.0 - 2.0 * G * Mr
    f_r = N1 + v1
    out = {"N": N[0], "w": w[0], "f_r": f_r[0], "one_minus_2GMr": den[0], "matter": (-2.0 * G * Fl / den)[0], "defect_rows": (2.0 * G * DM_rows / den)[0],
           "rhs_rows": ((-2.0 * G * Fl + 2.0 * G * DM_rows) / den)[0], "kinematic": (2.0 * G * Mt / den)[0], "DM_rows": DM_rows[0], "DM_direct": DM_direct[0],
           "M": (r * w ** 2 / (2.0 * G))[0], "rho": rho[0], "J": J[0], "S": sr[0]}
    return out


def fmt(x, f="%.4e"):
    return (f % x) if (x is not None and np.isfinite(x)) else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label=json=data_npz")
    ap.add_argument("--fm-segments", default="v17 (replayed)=engine_fields_v12;v19 (native)=engine_fields_v14")
    ap.add_argument("--fm-default", default="engine_fields_v16")
    ap.add_argument("--t-max", type=float, default=20.0)
    ap.add_argument("--hfrac", type=float, default=32.0)
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    L = ["## The fold, the birth event and the growth identity on the stored records (engine_fold_birth_v1.py; engine spec 36.4 (2), P21 (2) and P20; record-only)", "",
         "- m(T) = min_r (N - |v|) of the dense fields (the record's min_N_minus_absv); a crossing event is a sign change of m between successive records, its clock by linear interpolation (the lead's form) and by the quadratic through the two records and the next one away from the crossing, R_* the radius of the minimum interpolated linearly to the linear clock; the records' spacing bounds the interpolation only by its own width.",
         "- the roots: every zero of N + v (the record's trapping_zeros), r_in the first and R_t the last of two or more; f_r = d(N + v)/dr at each from the re-solved dense fields (negative at an inner root, positive at an outer one).",
         "- M = r w^2/(2G), w = v/N (P20 eq. 3); \"M/(r/2G) - 1\" on the re-solved dense fields at the root; \"stored R_t/(2 g M_MS)\" the record's own; \"applied\" the same M with w from the applied site fields interpolated linearly at R_t (the field the Hamiltonian sees, not a root of it); P20 eq. 17: 2 g E - R_t against 2 g [E - M(R_t)].",
         "- the growth identity (P20 eq. 16, w kept exact): dR/dT = [-2G Fl + 2G D_M]/(1 - 2G M_r), Fl = 4 pi R^2 [v (rho + S) + N (1 + w^2) J] (= -4 pi R^2 N (rho - 2J + S) at w = -1), M_r = 4 pi R^2 (rho + w J); rho, J the solve's rho_ell, J_ell, S the smoothed radial stress (dust plus pairs); D_M the rows' defect (R^2 v/G) res_tt/N from the direct tangential row (engine_rows_v2, d_t k_t centred at fixed r; engine_certificate_v3's normalization); \"kinematic\" 2G M_t/(1 - 2G M_r) with M_t centred at fixed r (the implicit-function identity of R = 2GM(R, T)); D_M direct = M_t + Fl beside the rows' D_M; dR/dT the centred difference of the records' root over the neighbouring records (printed where both neighbours hold the same number of zeros). Lattice units, G = g; T in lattice units for the derivatives, T/r_s in the tables.",
         ""]
    store = {}
    t0 = time.time()
    for spec in a.run:
        lab, jpath, dpath = spec.split("=", 2)
        J, R = load_records(jpath)
        Z = np.load(dpath, allow_pickle=True)
        Tz = np.asarray(Z["T"], float); rs = float(Z["rs"]); g = float(Z["g"])
        FO = FieldsOf(Z, a.fm_segments, a.fm_default)
        if len(Tz) != len(R) or float(np.max(np.abs(Tz / rs - np.array([r["T_over_rs"] for r in R])))) > 1e-9:
            L.append("### %s: the json's records and the data npz's do not align (%d against %d); skipped" % (lab, len(R), len(Tz)))
            continue
        ev = crossing_events(R, a.t_max)
        L.append("### %s (%s; data %s; the field module per record: %s)" % (lab, os.path.basename(jpath), os.path.basename(dpath),
                                                                          a.fm_segments if FO.seg_names else a.fm_default))
        L.append("")
        L.append("| event | between T/rs | m at the two records | T/rs (linear) | T/rs (quadratic) | R_* (linear) | r_min at the two records | zeros after |")
        L.append("|---|---|---|---|---|---|---|---|")
        for e in ev:
            L.append("| %s | %.3f, %.3f | %+.4f, %+.4f | %.4f | %s | %s | %.2f, %.2f | %s |" % (
                e["kind"], e["records"][0], e["records"][1], e["m"][0], e["m"][1], e["T_lin"], fmt(e["T_quad"], "%.4f"), fmt(e["R_lin"], "%.3f"),
                e["r_min"][0], e["r_min"][1], ", ".join("%.3f" % x for x in e["zeros_after"]) or "none"))
        b0 = next((e for e in ev if e["kind"] == "birth"), None)
        if b0 is None:
            L += ["", "- no birth to T/rs %g" % a.t_max, ""]
            store[lab] = {"events": ev}
            continue
        k_start = next(i for i, r in enumerate(R) if abs(r["T_over_rs"] - b0["records"][0]) < 1e-9)
        ks = [k for k in range(k_start, len(R)) if R[k]["T_over_rs"] <= a.t_max + 1e-9]
        T = np.asarray([r["T"] for r in R], float)
        rows = []
        for k in ks:
            r = R[k]; z = zeros_of(r)
            row = {"k": k, "T_over_rs": r["T_over_rs"], "nz": len(z), "zeros": z, "E": r["E"], "min": r["min_N_minus_absv"], "r_min": r["r_min_N_minus_absv"],
                   "stored_Rt_over_2gMS": r.get("R_t_over_2g_MS")}
            if len(z) >= 1 and FO(k)["failed"]:
                row["failed"] = True
            elif len(z) >= 1:
                F0 = FO(k)
                Rt = z[-1]
                Nr = float(at(F0["N"], Rt)[0]); vr = float(at(F0["v"], Rt)[0]); w = vr / Nr
                row["M_Rt_dev"] = (Rt * w * w / (2.0 * g)) / (Rt / (2.0 * g)) - 1.0
                Na = float(np.interp(Rt, FO.R, F0["N_app"])); va = float(np.interp(Rt, FO.R, F0["v_app"]))
                row["M_Rt_applied_dev"] = (va / Na) ** 2 - 1.0
                row["eq17_lhs"] = 2.0 * g * r["E"] - Rt
                row["eq17_rhs"] = 2.0 * g * (r["E"] - Rt * w * w / (2.0 * g))
                if len(z) >= 2:
                    ri = z[0]
                    Ni = float(at(F0["N"], ri)[0]); vi = float(at(F0["v"], ri)[0]); wi = vi / Ni
                    row["M_rin_dev"] = (ri * wi * wi / (2.0 * g)) / (ri / (2.0 * g)) - 1.0
                for nm, idx in (("Rt", -1), ("rin", 0)):
                    if nm == "rin" and len(z) < 2:
                        continue
                    km, kp = k - 1, k + 1
                    if km < 0 or kp >= len(R) or FO(km)["failed"] or FO(kp)["failed"]:
                        continue
                    zm, zp = zeros_of(R[km]), zeros_of(R[kp])
                    gt = growth_terms(FO, k, km, kp, T, z[idx], a.hfrac)
                    row["growth_" + nm] = gt
                    if len(zm) == len(z) == len(zp):
                        row["dRdT_" + nm] = (zp[idx] - zm[idx]) / (T[kp] - T[km])
            rows.append(row)
        L += ["", "#### %s: the roots and the mass at the roots, per record from the record before the first birth to T/rs %g" % (lab, a.t_max), "",
              "| T/rs | zeros | r_in | R_t | min(N - abs v) at r | M/(r/2G) - 1 at r_in | M/(r/2G) - 1 at R_t | stored R_t/(2 g M_MS) | applied (w^2 - 1) at R_t | 2gE - R_t | 2g [E - M(R_t)] | f_r at r_in | f_r at R_t |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for row in rows:
            z = row["zeros"]; gR = row.get("growth_Rt") or {}; gi = row.get("growth_rin") or {}
            L.append("| %.3f | %d | %s | %s | %+.4f at %.2f | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                row["T_over_rs"], row["nz"], fmt(z[0], "%.3f") if len(z) >= 2 else "-", fmt(z[-1], "%.3f") if z else "-", row["min"], row["r_min"],
                fmt(row.get("M_rin_dev"), "%.1e"), fmt(row.get("M_Rt_dev"), "%.1e"), fmt(row.get("stored_Rt_over_2gMS"), "%.10f"),
                fmt(row.get("M_Rt_applied_dev"), "%+.4e"), fmt(row.get("eq17_lhs"), "%.4f"), fmt(row.get("eq17_rhs"), "%.4f"),
                fmt(gi.get("f_r"), "%+.4e"), fmt(gR.get("f_r"), "%+.4e")))
        for nm, title in (("Rt", "R_t (the outer root)"), ("rin", "r_in (the inner root)")):
            L += ["", "#### %s: the growth identity at %s (P20 eq. 16)" % (lab, title), "",
                  "| T/rs | R | dR/dT (records, centred) | right side with the rows' D_M | kinematic 2G M_t/(1 - 2G M_r) | matter term | defect term | 1 - 2G M_r | D_M (rows) | D_M (direct, M_t + Fl) | rho | J | S | N |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for row in rows:
                gt = row.get("growth_" + nm)
                if not gt:
                    continue
                z = row["zeros"]; Rr = z[-1] if nm == "Rt" else z[0]
                L.append("| %.3f | %.3f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %.6f |" % (
                    row["T_over_rs"], Rr, fmt(row.get("dRdT_" + nm), "%+.4e"), fmt(gt["rhs_rows"], "%+.4e"), fmt(gt["kinematic"], "%+.4e"),
                    fmt(gt["matter"], "%+.4e"), fmt(gt["defect_rows"], "%+.4e"), fmt(gt["one_minus_2GMr"], "%+.4e"), fmt(gt["DM_rows"], "%+.3e"),
                    fmt(gt["DM_direct"], "%+.3e"), fmt(gt["rho"], "%.3e"), fmt(gt["J"], "%+.3e"), fmt(gt["S"], "%+.3e"), gt["N"]))
        L.append("")
        store[lab] = {"events": ev, "rows": rows}
        print("%s done (%d records, %.0f s)" % (lab, len(rows), time.time() - t0), flush=True)
    L.append("- every number above is printed by the script from the runs' json records and records data npz (the dense fields re-solved by each record's own field module); nothing is read or ruled here.")
    open(a.md, "w").write("\n".join(L) + "\n")
    if a.out_json:
        json.dump(store, open(a.out_json, "w"), default=float, indent=1)
    print("\n".join(L[:8]))


if __name__ == "__main__":
    main()
