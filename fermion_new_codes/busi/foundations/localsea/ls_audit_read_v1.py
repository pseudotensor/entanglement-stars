#!/usr/bin/env python
"""ls_audit_read_v1.py (busi, inbox #447): reads the ls_chain_audit_v1 outputs and the records they came from (box; post-processing,
field re-solves with the reader's field module engine_fields_v16 7a69faad). Writes <out>/ls_audit_read_v1.json; prints tables.

(1) slice  : per record, the checks (spectrum against the stored one, R2 against the negative projector, band character) and,
             radius window by window, the shell sums of: the production reference's polarization eps_R2 - eps_0 (J likewise); the
             difference R2 - P_neg; the 3D local sea at second order on the same fields; the pairs (sea - R2) against the record's
             stored eps_pairs.
(2) scale  : the chain sea's polarization at lambda = 1, 2, 3 (fields u(r/lambda)), solved per site for e2 + e4/lambda^2 +
             e6/lambda^4: the production (all orders, lambda = 1) against the local sea through second order (e2; reviewer B's candidate)
             and through fourth (e2 + e4; reviewer A's), radius window by window; then reviewer B A, one field solve with the record's
             sources changed by eps_R2 - eps_LS (the reference replaced, the state held): the root, min(N - |v|) and E against the
             record's own solve.
(3) rates  : at every record of the window with an outer root R_t and both neighbours: with w = v/N of the record's dense solve,
             M_X(R) = int_0^R 4 pi r^2 (rho_X + w J_X) dr (rho, J the field module's smoothed sources of the shell totals X), at
             R = R_t fixed: the total rate dM/dt (central difference over the neighbours), the evolution row at the root
             4 pi R^2 N (rho + p_r - 2 J) (p_r the handoff's smoothed radial stress), the residual, the reference's rate -dM_R2/dt,
             each over the row's largest term max(4 pi R^2 N |rho|, 4 pi R^2 N |p_r|, 8 pi R^2 N |J|); medians over the window."""
import argparse
import importlib.util
import json
import math
import os
import sys

import numpy as np

FM_PATH = "<internal>/engine_fields_v16.py"


def field_module():
    sys.dont_write_bytecode = True
    d = os.path.dirname(FM_PATH)
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location("engine_fields_v16_ro", FM_PATH)
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    return M


def windows(Rt, nmax):
    w = [(1, 4), (4, 8), (8, 16), (16, 32)]
    if Rt:
        w += [(32, Rt), (Rt, 2 * Rt), (2 * Rt, 4 * Rt), (4 * Rt, nmax + 1)]
    else:
        w += [(32, 64), (64, 128), (128, 512), (512, nmax + 1)]
    return [(a, b) for a, b in w if a < b]


def wsum(R, x, a, b):
    m = (R >= a) & (R < b)
    return float(np.sum(x[m]))


def record_of(json_path, T_over_rs):
    J = json.load(open(json_path))
    for x in J["records"]:
        if abs(x["T_over_rs"] - T_over_rs) < 1e-9:
            return x, J
    return None, J


def solve(FM, eps, J, g, R, ell):
    _, _, _, info = FM.solve_constraints_ivp(eps, J, g, R, ell, collar_policy="flag", support_rel=1e-14)
    if info.get("failed"):
        return None
    tz = FM.trapping_radius(None, None, R, info)
    return {"info": info, "zeros": [float(z) for z in tz["zeros"]], "min": tz.get("min"), "r_min": tz.get("r_min")}


def E_of(sol, R):
    """the boundary charge from the dense solve: M = r w^2 / (2 g) at the collar, w = v/N."""
    info = sol["info"]
    rc = float(info["r_collar"])
    N = float(info["N_dense"](rc)); v = float(info["v_dense"](rc))
    return rc, N, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="<internal>/audit")
    ap.add_argument("--json4", required=True); ap.add_argument("--fields4", required=True); ap.add_argument("--handoff4", required=True)
    ap.add_argument("--json8", required=True); ap.add_argument("--fields8", required=True); ap.add_argument("--handoff8", required=True)
    ap.add_argument("--out", default="<internal>/localsea")
    ap.add_argument("--parts", default="scan,slice,scale,rates")
    a = ap.parse_args()
    FM = field_module()
    res = {}
    src = {"s4": (a.json4, a.fields4, a.handoff4), "s8": (a.json8, a.fields8, a.handoff8)}
    parts = set(a.parts.split(","))
    # ---------------------------------------------------------------- (0) every record: is the production reference the negative set?
    if "scan" in parts:
        res["scan"] = {}
        for hist in ("s4", "s8"):
            J = json.load(open(src[hist][0]))
            n = len(J["records"]); bad = []; pockets = []; refills = []; inv = []; dR = []
            for x in J["records"]:
                if x.get("mask_equals_negative_set") is False:
                    bad.append(x["T_over_rs"])
                if sum((x.get("n_pocket") or {}).values()) > 0:
                    pockets.append(x["T_over_rs"])
                if sum((x.get("n_refill_R0") or {}).values()) > 0:
                    refills.append(x["T_over_rs"])
                if (x.get("inversions_total") or 0) > 0:
                    inv.append(x["T_over_rs"])
                if x.get("R2_minus_R0_F") is not None:
                    dR.append((x["T_over_rs"], x["R2_minus_R0_F"]))
            dRv = np.array([d[1] for d in dR]) if dR else np.zeros(1)
            res["scan"][hist] = {"records": n, "mask_not_negative_set": bad, "with_pockets": pockets, "with_refills": refills,
                                 "with_inversions": inv, "R2_minus_R0_F_max": float(dRv.max()),
                                 "R2_minus_R0_F_at_max_T": dR[int(np.argmax(dRv))][0] if dR else None}
            print("scan", hist, json.dumps(res["scan"][hist], default=float)[:600])
    # ---------------------------------------------------------------- (1) slices
    if "slice" in parts:
        res["slice"] = {}
        for tag, hist, T in (("s4_T15", "s4", 15.0), ("s4_T16", "s4", 16.0), ("s8_T14", "s8", 14.0), ("s8_T16", "s8", 16.0), ("s4_T07", "s4", 7.0)):
            p = os.path.join(a.dir, "ls_chain_audit_v1_%s.npz" % tag)
            if not os.path.exists(p):
                continue
            Z = np.load(p); M = json.load(open(p.replace(".npz", ".json")))
            rec, Jh = record_of(src[hist][0], T)
            F = np.load(src[hist][1], allow_pickle=True)
            k = int(np.flatnonzero(np.abs(np.asarray(F["T"], float) / 64.0 - T) < 1e-9)[0])
            R = Z["R"]; Rt = rec.get("R_t") if rec else None
            pol_e = Z["eps_R2"] - Z["eps_0"]; pol_J = Z["J_R2"] - Z["J_0"]
            d_neg_e = Z["eps_R2"] - Z["eps_neg"]; d_neg_J = Z["J_R2"] - Z["J_neg"]
            pairs = Z["eps_sea"] - Z["eps_R2"]
            pc = M["per_channel"]
            rows = []
            for (lo, hi) in windows(Rt, int(R[-1])):
                rows.append({"r": [lo, hi], "pol_eps": wsum(R, pol_e, lo, hi), "pol_J": wsum(R, pol_J, lo, hi),
                             "R2_minus_neg_eps": wsum(R, d_neg_e, lo, hi), "R2_minus_neg_J": wsum(R, d_neg_J, lo, hi),
                             "local_sea_3d_eps": wsum(R, Z["rho3d"], lo, hi), "local_sea_3d_J": wsum(R, Z["J3d"], lo, hi),
                             "clock_only_sea_pol_eps": wsum(R, Z["eps_N"] - Z["eps_0"], lo, hi),
                             "sources_eps_tot": wsum(R, np.asarray(F["eps_tot"][k], float), lo, hi)})
            res["slice"][tag] = {
                "T_over_rs": T, "R_t": Rt, "zeros": rec.get("trapping_zeros") if rec else None, "min_N_minus_absv": rec.get("min_N_minus_absv") if rec else None,
                "mask_equals_negative_set": rec.get("mask_equals_negative_set") if rec else None, "n_pocket": rec.get("n_pocket") if rec else None,
                "max_eig_vs_stored": max(v["eig_vs_stored_max"] for v in pc.values() if v["eig_vs_stored_max"] is not None),
                "max_R2_vs_neg_mismatch": max(v["R2_vs_neg_mismatch"] for v in pc.values()),
                "max_comm_h_R2_F": max(v["comm_h_R2_F"] for v in pc.values()),
                "bN_mixed_total": int(sum(v["bN_mixed_count"] for v in pc.values())),
                "neg_vs_band_character_disagree_total": int(sum(v["neg_set_vs_band_character_disagree"] for v in pc.values())),
                "min_abs_e": min(v["min_abs_e"] for v in pc.values()),
                "pairs_vs_stored_eps_pairs_max": float(np.max(np.abs(pairs - np.asarray(F["eps_pairs"][k], float)))),
                "stored_eps_pairs_maxabs": float(np.max(np.abs(np.asarray(F["eps_pairs"][k], float)))),
                "pol_eps_total": float(pol_e.sum()), "pol_J_total": float(pol_J.sum()),
                "local_sea_3d_eps_total": float(Z["rho3d"].sum()), "windows": rows}
            print(tag, json.dumps({k_: v_ for k_, v_ in res["slice"][tag].items() if k_ != "windows"}, default=float))
            for r_ in rows:
                print("   r %s: pol_eps %.4e pol_J %.4e | R2-neg %.2e %.2e | 3D LS2 eps %.4e J %.4e | clock-only %.3e | sources %.4e" % (
                    r_["r"], r_["pol_eps"], r_["pol_J"], r_["R2_minus_neg_eps"], r_["R2_minus_neg_J"], r_["local_sea_3d_eps"],
                    r_["local_sea_3d_J"], r_["clock_only_sea_pol_eps"], r_["sources_eps_tot"]))
    # ---------------------------------------------------------------- (2) scaling
    if "scale" in parts:
        L = {}
        for lam in (1, 2, 3):
            p = os.path.join(a.dir, "ls_chain_audit_v1_s4_T15_lam%d.npz" % lam)
            if os.path.exists(p):
                L[lam] = np.load(p)
        if len(L) == 3:
            R = L[1]["R"]; n = len(R)
            idx = {lam: (lam * R).astype(int) - 1 for lam in L}
            E = np.stack([L[lam]["eps_pol"][idx[lam]] for lam in (1, 2, 3)]); Jp = np.stack([L[lam]["J_pol"][idx[lam]] for lam in (1, 2, 3)])
            A = np.array([[1, 1, 1], [1, 1 / 4, 1 / 16], [1, 1 / 9, 1 / 81]])
            Ai = np.linalg.inv(A)
            e2, e4, e6 = Ai @ E; j2, j4, j6 = Ai @ Jp
            rec, Jh = record_of(a.json4, 15.0)
            Rt = rec["R_t"]
            rows = []
            for (lo, hi) in windows(Rt, int(R[-1])):
                rows.append({"r": [lo, hi], "prod_all_orders": wsum(R, E[0], lo, hi), "e2": wsum(R, e2, lo, hi), "e4": wsum(R, e4, lo, hi),
                             "e6_plus": wsum(R, e6, lo, hi), "prod_minus_LS2": wsum(R, E[0] - e2, lo, hi), "prod_minus_LS4": wsum(R, E[0] - e2 - e4, lo, hi),
                             "J_prod": wsum(R, Jp[0], lo, hi), "J_prod_minus_LS2": wsum(R, Jp[0] - j2, lo, hi), "J_prod_minus_LS4": wsum(R, Jp[0] - j2 - j4, lo, hi)})
            # reviewer B A: one field solve with the reference replaced
            F = np.load(a.fields4, allow_pickle=True)
            k = int(np.flatnonzero(np.abs(np.asarray(F["T"], float) / 64.0 - 15.0) < 1e-9)[0])
            g, ell = float(F["g"]), float(F["ell"]); Rs = np.asarray(F["R"], float)
            eps0, J0 = np.asarray(F["eps_tot"][k], float), np.asarray(F["J_tot"][k], float)
            solves = {}
            for lab, de, dj in (("record", 0 * e2, 0 * j2), ("LS2", E[0] - e2, Jp[0] - j2), ("LS4", E[0] - e2 - e4, Jp[0] - j2 - j4),
                                ("LS2_outside_4a", np.where(R >= 4, E[0] - e2, 0.0), np.where(R >= 4, Jp[0] - j2, 0.0))):
                sol = solve(FM, eps0 + de, J0 + dj, g, Rs, ell)
                if sol is None:
                    solves[lab] = {"failed": True}
                    continue
                rc, Nc, vc = E_of(sol, Rs)
                solves[lab] = {"zeros": sol["zeros"], "min": sol["min"], "r_min": sol["r_min"], "E_collar": rc * (vc / Nc) ** 2 / (2 * g),
                               "d_eps_total": float(np.sum(de)), "d_J_total": float(np.sum(dj))}
            res["scale"] = {"T_over_rs": 15.0, "R_t": Rt, "windows": rows, "solves": solves, "record_E": rec.get("E"),
                            "record_zeros": rec.get("trapping_zeros"), "record_min": rec.get("min_N_minus_absv")}
            print("scale", json.dumps(solves, default=float))
            for r_ in rows:
                print("   r %s: prod %.4e e2 %.4e e4 %.4e e6+ %.3e | prod-LS2 %.3e prod-LS4 %.3e | J prod %.3e -LS2 %.3e -LS4 %.3e" % (
                    r_["r"], r_["prod_all_orders"], r_["e2"], r_["e4"], r_["e6_plus"], r_["prod_minus_LS2"], r_["prod_minus_LS4"],
                    r_["J_prod"], r_["J_prod_minus_LS2"], r_["J_prod_minus_LS4"]))
    # ---------------------------------------------------------------- (3) the reference's rate at the outer root
    if "rates" in parts:
        res["rates"] = {}
        for hist in ("s4", "s8"):
            p = os.path.join(a.dir, "ls_chain_audit_v1_rates_%s.npz" % hist)
            if not os.path.exists(p):
                continue
            Zr = np.load(p); Mr = json.load(open(p.replace(".npz", ".json")))
            jp, fp, hp = src[hist]
            J = json.load(open(jp)); recs = {round(x["T"], 6): x for x in J["records"]}
            F = np.load(fp, allow_pickle=True); H = np.load(hp, allow_pickle=True)
            g, ell = float(F["g"]), float(F["ell"]); Rs = np.asarray(F["R"], float); tau = float(F["tau"])
            Ta = np.asarray(F["T"], float); Th = np.asarray(H["T"], float); rn = np.asarray(H["r_nodes"], float)
            ks = list(Zr["k"]); rows = []
            for i in range(1, len(ks) - 1):
                k = int(ks[i]); km, kp = int(ks[i - 1]), int(ks[i + 1])
                rec = recs.get(round(Ta[k], 6))
                if rec is None or rec.get("R_t") is None or kp - km != 2:
                    continue
                Rt = float(rec["R_t"])
                sol = solve(FM, np.asarray(F["eps_tot"][k], float), np.asarray(F["J_tot"][k], float), g, Rs, ell)
                if sol is None:
                    continue
                Nd, vd = sol["info"]["N_dense"], sol["info"]["v_dense"]
                rr = np.linspace(1e-6, Rt, 4001)
                w = vd(rr) / Nd(rr)
                rmax = float(sol["info"]["r_nodes"][-1]) if "r_nodes" in sol["info"] else float(Rs[-1])
                def mass(eps, Jt):
                    S = FM.SmoothedSources5(np.asarray(eps, float), np.asarray(Jt, float), Rs, ell, rmax, 1e-14)
                    return float(np.trapezoid(4 * math.pi * rr ** 2 * (S.rho(rr) + w * S.J(rr)), rr)), S
                Mt_m, _ = mass(F["eps_tot"][km], F["J_tot"][km]); Mt_p, _ = mass(F["eps_tot"][kp], F["J_tot"][kp])
                _, S0 = mass(F["eps_tot"][k], F["J_tot"][k])
                # the reference's energy from the handoff (the production R2 itself, every record); its momentum from the
                # recomputed negative projector, which is R2 only where the record says mask_equals_negative_set
                def ehand(kk_):
                    j_ = np.flatnonzero(np.abs(Th - Ta[kk_]) < 1e-9)
                    return np.asarray(H["eps_R2"][j_[0]], float)
                ok_J = all(bool(recs.get(round(Ta[x_], 6), {}).get("mask_equals_negative_set")) for x_ in (km, k, kp))
                MR_m, _ = mass(ehand(km), Zr["J_R2"][i - 1]); MR_p, _ = mass(ehand(kp), Zr["J_R2"][i + 1])
                MRe_m, _ = mass(ehand(km), 0 * Zr["J_R2"][i - 1]); MRe_p, _ = mass(ehand(kp), 0 * Zr["J_R2"][i + 1])
                dMt = (Mt_p - Mt_m) / (2 * tau); dMref = -(MR_p - MR_m) / (2 * tau); dMref_e = -(MRe_p - MRe_m) / (2 * tau)
                NR = float(Nd(Rt)); rho = float(S0.rho(Rt)); Jl = float(S0.J(Rt))
                kh = int(np.flatnonzero(np.abs(Th - Ta[k]) < 1e-9)[0])
                pr = float(np.interp(Rt, rn, np.asarray(H["S_r_ell"][kh], float)))
                t1, t2, t3 = 4 * math.pi * Rt ** 2 * NR * rho, 4 * math.pi * Rt ** 2 * NR * pr, 8 * math.pi * Rt ** 2 * NR * Jl
                big = max(abs(t1), abs(t2), abs(t3))
                row_m = t1 + t2 - t3; row_p = t1 + t2 + t3
                rows.append({"T_over_rs": Ta[k] / 64.0, "R_t": Rt, "N_R": NR, "v_R": float(vd(Rt)), "dM_dt": dMt, "row_minus2J": row_m,
                             "row_plus2J": row_p, "largest": big, "resid_minus2J_rel": (dMt - row_m) / big, "resid_plus2J_rel": (dMt - row_p) / big,
                             "ref_rate_rel": dMref / big, "ref_rate": dMref, "ref_rate_energy_only_rel": dMref_e / big,
                             "J_is_R2": ok_J, "mask_neg_set": bool(rec.get("mask_equals_negative_set"))})
            def med(key, sel=None):
                xs = [abs(r_[key]) for r_ in rows if (sel is None or sel(r_))]
                return float(np.median(xs)) if xs else None
            res["rates"][hist] = {"rows": rows, "median_abs_resid_minus2J": med("resid_minus2J_rel"), "median_abs_resid_plus2J": med("resid_plus2J_rel"),
                                  "median_abs_ref_rate_rel": med("ref_rate_rel"),
                                  "median_abs_ref_rate_rel_Jexact": med("ref_rate_rel", lambda r_: r_["J_is_R2"]),
                                  "median_abs_resid_minus2J_rel_Jexact": med("resid_minus2J_rel", lambda r_: r_["J_is_R2"]),
                                  "median_abs_ref_rate_energy_only_rel": med("ref_rate_energy_only_rel"),
                                  "n_rows": len(rows), "n_rows_Jexact": sum(1 for r_ in rows if r_["J_is_R2"]),
                                  "pocket_records_handoff_vs_neg": [(x["T_over_rs"], x.get("handoff_eps_R2_maxdiff")) for x in Mr["rows"] if not x["mask_equals_negative_set"]],
                                  "handoff_check_max": max((x.get("handoff_eps_R2_maxdiff", 0.0) for x in Mr["rows"]), default=None),
                                  "handoff_scale": max((x.get("handoff_eps_R2_maxabs", 0.0) for x in Mr["rows"]), default=None),
                                  "all_mask_equals_negative_set": all(bool(x["mask_equals_negative_set"]) for x in Mr["rows"])}
            print(hist, json.dumps({k_: v_ for k_, v_ in res["rates"][hist].items() if k_ != "rows"}, default=float))
            for r_ in rows:
                print("   T %.4f R_t %.2f | dM/dt %.3e row(-2J) %.3e row(+2J) %.3e big %.3e | resid %.3f / %.3f | ref rate %.3f (E only %.3f) %s" % (
                    r_["T_over_rs"], r_["R_t"], r_["dM_dt"], r_["row_minus2J"], r_["row_plus2J"], r_["largest"], r_["resid_minus2J_rel"],
                    r_["resid_plus2J_rel"], r_["ref_rate_rel"], r_["ref_rate_energy_only_rel"], "" if r_["J_is_R2"] else "(pocket: J from P_neg)"))
    with open(os.path.join(a.out, "ls_audit_read_v1.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
