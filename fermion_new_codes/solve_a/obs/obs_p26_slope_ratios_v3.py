#!/usr/bin/env python
"""obs_p26_slope_ratios_v3.py (builder analyst; version 2 with kappa_0/kappa_T - 1 printed to three significant figures (version 2 printed the ratio to 6 decimals, 1.000000 at every record) and 4 g E kappa_0 printed as its three factors (2 g E/R_H) N_H [1 - 8 pi g R_H^2 (rho_H - J_H)]; nothing else changed; engine spec 37.2 (2): "obs_p26_slope_ratios_v2 (new): per J3' record D_2 and the collar
prediction kappa_0 = (N_H/2R_H)[1 - 8 pi G R_H^2 (rho_H - J_H)] from the solve's smoothed sources rho_ell, J_ell and the dense N at the
dense root (the inputs printed), kappa_0/kappa_T beside"): per J3' record (--rec label:record.json:fields.npz:T/rs, repeatable), the stored
shell totals eps_tot, J_tot of that record re-solved by the production's own constraint solver, engine_fields_v14.solve_constraints_ivp
(the record's g and ell, collar policy and support_rel from the record's args, rtol 1e-12 and atol 1e-16 as engine_collapse_v19.solve),
the dense root R_H = the outermost zero of N - |v| on the dense functions (engine_fields_v14.trapping_radius, as the engine) and the dense
slope kappa_T = d(N + v)/dR there by engine_collapse_v19.surface_jets' Richardson step (h 1/4 and 1/8; its kappa_T part copied here, the
engine itself not imported); gate: R_H and kappa_T reproduce the record json's stored R_t and kappa_T (relative differences printed;
--gate 1e-6). Then, at R_H: N_H = N_dense(R_H), rho_H = rho_ell(R_H), J_H = J_ell(R_H) (the solve's smoothed sources, per unit volume;
J_ell = <P_r> = -J_r of the goal's display, the engine's dictionary), the two source terms 8 pi g R_H^2 rho_H and 8 pi g R_H^2 J_H,
kappa_0 = (N_H/(2 R_H)) [1 - 8 pi g R_H^2 (rho_H - J_H)] (G = g), kappa_0/kappa_T, 4 g E kappa_0, and D_2 = 4 g E kappa_T - 1 with E the
record's stored E. A reading for the lead (turn 46); nothing composed. Every number printed by the script. Args: --rec (repeatable) --md
[--gate 1e-6] [--code-dir]."""
import argparse
import json
import math
import os
import sys


def kappa_T_rich(info, Rt):
    """engine_collapse_v19.surface_jets, its first-derivative part: d(N + v)/dR at R_t by Richardson in h (1/4, 1/8)."""
    Nf = lambda x: float(info["N_dense"](x)[0]); vf = lambda x: float(info["v_dense"](x)[0])  # noqa: E731
    fnv = lambda x: Nf(x) + vf(x)  # noqa: E731

    def d1(F, x, h):
        return (F(x + h) - F(x - h)) / (2 * h)
    a, b = d1(fnv, Rt, 0.25), d1(fnv, Rt, 0.125)
    return (4.0 * b - a) / 3.0, abs(b - a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", action="append", required=True); ap.add_argument("--md", required=True)
    ap.add_argument("--gate", type=float, default=1e-6); ap.add_argument("--code-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__))))
    a = ap.parse_args()
    sys.path.insert(0, a.code_dir)
    import numpy as np
    import engine_fields_v14 as FLD
    L = ["## J3' records: D_2 and the collar prediction kappa_0 (obs_p26_slope_ratios_v3.py; engine spec 37.2 (2))", "",
         "- kappa_0 = (N_H/(2 R_H)) [1 - 8 pi g R_H^2 (rho_H - J_H)] (the lead's formula, G = g), from the solve's smoothed sources rho_ell, J_ell and the dense N at the dense root R_H, re-solved from each record's stored shell totals by engine_fields_v14.solve_constraints_ivp; J_ell = <P_r> = -J_r of the goal's display (the engine's dictionary); kappa_T the dense slope d(N + v)/dR at R_H (the engine's Richardson step); D_2 = 4 g E kappa_T - 1.",
         "- the gate: the re-solved R_H and kappa_T against the record json's stored R_t and kappa_T, relative difference below %g." % a.gate, "",
         "| history | T/r_s | E | R_H (re-solved; stored R_t; rel. diff.) | kappa_T (re-solved; stored; rel. diff.; Richardson spread) | gate | N_H | rho_H | J_H | 8 pi g R_H^2 rho_H | 8 pi g R_H^2 J_H | kappa_0 | kappa_0/kappa_T - 1 | 4 g E kappa_0 = (2 g E/R_H) N_H [1 - 8 pi g R_H^2 (rho_H - J_H)]: the three factors | 4 g E kappa_0 | D_2 = 4 g E kappa_T - 1 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    fails = 0
    for spec in a.rec:
        lab, rj, fz, tr = spec.split(":")
        tr = float(tr)
        d = json.load(open(rj)); ar = d.get("args", {})
        recs = [r for r in d["records"] if abs(r["T_over_rs"] - tr) < 1e-9]
        if len(recs) != 1:
            sys.exit("record T/rs %g: %d matches in %s" % (tr, len(recs), rj))
        r = recs[0]
        Z = np.load(fz, allow_pickle=False)  # only numeric keys are read
        T = np.asarray(Z["T"], float); k = int(np.argmin(np.abs(T - r["T"])))
        if abs(T[k] - r["T"]) > 1e-9:
            sys.exit("the fields file has no slice at T %g" % r["T"])
        g = float(Z["g"]); ell = float(Z["ell"]); R = np.asarray(Z["R"], float)
        Nn, vn, Wn, info = FLD.solve_constraints_ivp(np.asarray(Z["eps_tot"][k], float), np.asarray(Z["J_tot"][k], float), g, R, ell, rtol=1e-12, atol=1e-16,
                                                     collar_policy=ar.get("collar_policy", "flag"), support_rel=ar.get("support_rel", 1e-14))
        if info.get("failed"):
            sys.exit("the solve failed at %s T/rs %g: %s" % (lab, tr, info["failed"]))
        tz = FLD.trapping_radius(Nn, vn, R, info)
        Rt = tz["zeros"][-1] if tz["zeros"] else None
        if Rt is None:
            sys.exit("no dense root at %s T/rs %g" % (lab, tr))
        kT, spr = kappa_T_rich(info, Rt)
        dR = abs(Rt - r["R_t"]) / abs(r["R_t"]); dk = abs(kT - r["kappa_T"]) / abs(r["kappa_T"])
        ok = dR < a.gate and dk < a.gate
        fails += 0 if ok else 1
        NH = float(info["N_dense"](Rt)[0]); rhoH = float(info["rho_ell"](Rt)); JH = float(info["J_ell"](Rt))
        sr = 8.0 * math.pi * g * Rt ** 2 * rhoH; sj = 8.0 * math.pi * g * Rt ** 2 * JH
        k0 = NH / (2.0 * Rt) * (1.0 - (sr - sj))
        E = r["E"]
        L.append("| %s | %.4f | %.6f | %.6f; %.6f; %.2e | %.6e; %.6e; %.2e; %.1e | %s | %.6f | %.6e | %.6e | %.6f | %.6f | %.6e | %.3e | %.6f; %.6f; %.6f | %.6f | %+.6f |" % (
            lab, tr, E, Rt, r["R_t"], dR, kT, r["kappa_T"], dk, spr, "pass" if ok else "FAIL", NH, rhoH, JH, sr, sj, k0, k0 / kT - 1.0,
            2 * g * E / Rt, NH, 1.0 - (sr - sj), 4 * g * E * k0, 4 * g * E * kT - 1.0))
    L += ["", "- gate: %s (%d of %d records)." % ("PASS" if fails == 0 else "FAIL", len(a.rec) - fails, len(a.rec)),
          "- every number above is printed by the script from %s; nothing composed." % "; ".join(s.split(":")[1] + ", " + s.split(":")[2] for s in a.rec)]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("wrote %s; gate %s" % (a.md, "PASS" if fails == 0 else "FAIL"))


if __name__ == "__main__":
    main()
