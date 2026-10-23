#!/usr/bin/env python
"""obs_j3p_scope_v2.py (builder analyst; engine spec 29.4 and the goal's task 3: "the predicted shift (9/4) sigma^2/r_s^2 ... with the
kernel's own sigma^2 = ell^2 + a^2/12 ...; the scope files used the compact kernel's 1.15 and 4.35 and are recomputed as obs_j3p_scope_v2"):
version 1 with sigma^2 = ell^2 + a^2/12 (a = 1), the per-axis variance of this lane's Gaussian kernel of standard deviation ell (measured on
the engine's own kernel: ((A*r^2)_n - R_n^2)/3 = ell^2 exactly, results/engine/PSD/ps_defect_v1_c19_rs64.md), plus the cell's a^2/12; the
compact kernel's ell^2/15 + a^2/12 is printed beside as the value version 1 used. Nothing else changes; version 1's docstring follows.
obs_j3p_scope_v1.py (builder analyst; engine spec 28.3, S2 of REVIEWA_REVIEW_web40e, the parts that stored fields decide: "(a) kappa_m and
r_H,m of the back-averaged N_bar + v_bar at ell 4 (the production's late records) and ell 8 (the control's at the same times) against
2 r_s kappa_m = 1 - (9/4) sigma_ell^2/r_s^2, sigma_ell^2 = ell^2/15 + a^2/12 ...; (b) ... 4 g E d_R(N + v) at R_H - 1 with the full N + v").
Per selected record of a run: E from the record, r_s = 2 g E; the back-averaged fields N_bar, v_bar = the record's applied site fields
(N_applied, v_applied: the kernel-averaged fields of the h_m that propagated the state); r_H,m the outermost root of N_bar + v_bar on the
sites (the last sign change from negative inside to positive outside, by the local quadratic fit through the nearest five sites) and
kappa_m its slope there (the same fit); the prediction 2 r_s kappa_m = 1 - (9/4) sigma^2/r_s^2 and r_H,m/r_s = 1 - (9/4) sigma^2/r_s^2
(the review's 1.5), sigma^2 = ell^2/15 + 1/12 (a = 1); the full (dense) N + v of the record's solve (re-solved from the stored shell totals
by the run's field module) with its outermost root R_H, its slope there and the test 4 g E d_R(N + v)|_H - 1; the differences printed. The
J3' parts (kappa_p from the ray map, T against 1/(8 pi g E) with its bias budget, the lattice-against-continuum endpoint operator on the
identical profile) are read from J3''s outputs when they land. Every number printed by the script. CPU, seconds per record.
Args: --run dir/tag (the run's json and fields npz) --fm (the field module) --times (T/rs list) --md --out-json --label."""
import argparse
import importlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "engine", "code"))


def outer_root(R, f):
    """the outermost negative-to-positive sign change of f on R, refined by a quadratic fit through the nearest five sites; (root, slope)."""
    s = np.nonzero((f[:-1] < 0) & (f[1:] >= 0))[0]
    if len(s) == 0:
        return None, None
    i = int(s[-1])
    lo = max(i - 2, 0); hi = min(i + 3, len(R))
    x = R[lo:hi] - R[i]; c = np.polyfit(x, f[lo:hi], 2)
    roots = np.roots(c); roots = roots[np.isreal(roots)].real
    roots = roots[(roots >= R[i] - R[i] - 1e-9) & (roots <= R[i + 1] - R[i] + 1e-9)]
    if len(roots) == 0:
        r0 = -f[i] * (R[i + 1] - R[i]) / (f[i + 1] - f[i])
    else:
        r0 = float(roots[0])
    return R[i] + r0, float(2 * c[0] * r0 + c[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--fm", required=True); ap.add_argument("--times", default="30,32,34,36,38,39,39.5,40")
    ap.add_argument("--md", required=True); ap.add_argument("--out-json", required=True); ap.add_argument("--label", default="")
    a = ap.parse_args()
    FM = importlib.import_module(a.fm)
    d, tag = os.path.split(a.run)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    ra = J["args"]; g = float(ra["g"]); ell = float(ra["ell"]); rs0 = float(ra["rs"])
    R = np.asarray(Z["R"], float); T = np.asarray(Z["T"], float)
    recs = {round(float(r["T"]), 6): r for r in J["records"] if r.get("E") is not None}
    sig2 = ell * ell + 1.0 / 12.0
    sig2_compact = ell * ell / 15.0 + 1.0 / 12.0
    rows = []
    for tt in [float(x) for x in a.times.split(",")]:
        k = int(np.argmin(np.abs(T / rs0 - tt)))
        if abs(T[k] / rs0 - tt) > 1e-6:
            rows.append({"T_over_rs": tt, "missing": True}); continue
        rec = recs.get(round(T[k], 6))
        E = float(rec["E"]); rs = 2.0 * g * E
        fb = np.asarray(Z["N_applied"][k], float) + np.asarray(Z["v_applied"][k], float)
        rHm, km = outer_root(R, fb)
        _, _, _, info = FM.solve_constraints_ivp(np.asarray(Z["eps_tot"][k], float), np.asarray(Z["J_tot"][k], float), g, R, ell,
                                                 collar_policy=ra.get("collar_policy", "flag"), support_rel=ra.get("support_rel", 1e-14))
        rd = np.linspace(R[0], R[-1], 16 * len(R))
        fd = info["N_dense"](rd) + info["v_dense"](rd)
        rH, kd = outer_root(rd, fd)
        pred = 1.0 - 2.25 * sig2 / rs ** 2
        row = {"T_over_rs": tt, "E": E, "r_s": rs, "sigma2": sig2, "r_Hm": rHm, "kappa_m": km, "r_H_dense": rH, "slope_dense": kd, "prediction": pred,
               "2rs_kappa_m": (2 * rs * km) if km is not None else None, "rHm_over_rs": (rHm / rs) if rHm is not None else None,
               "test_4gE_dN+v_minus_1": (4 * g * E * kd - 1.0) if kd is not None else None, "rHm_over_rH": (rHm / rH) if (rHm and rH) else None}
        rows.append(row)
    f = lambda x, fm: (fm % x) if x is not None else "-"
    L = ["## J3' re-scoped, the stored-field parts (obs_j3p_scope_v2.py; engine spec 29.4, 28.3 (a) and (b)'s dense test): %s %s" % (tag, a.label), "",
         "- the back-averaged N_bar + v_bar = the record's applied site fields; sigma^2 = ell^2 + a^2/12 = %.6f (ell %g; this lane's Gaussian of standard deviation ell; the compact kernel's ell^2/15 + a^2/12, version 1's value, %.6f); the prediction 1 - (9/4) sigma^2/r_s^2 for both 2 r_s kappa_m and r_H,m/r_s (the review's 1.5); the dense N + v of the record's solve (%s)." % (sig2, ell, sig2_compact, a.fm), "",
         "| T/rs | E | r_s = 2gE | r_H,m (averaged) | kappa_m | 2 r_s kappa_m | r_H,m/r_s | prediction | r_H (dense) | d_R(N+v) at r_H (dense) | 4 g E d_R(N+v)|_H - 1 | r_H,m/r_H |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("missing"):
            L.append("| %.3f | no record | - | - | - | - | - | - | - | - | - | - |" % r["T_over_rs"]); continue
        L.append("| %.3f | %.6f | %.4f | %s | %s | %s | %s | %.6f | %s | %s | %s | %s |" % (r["T_over_rs"], r["E"], r["r_s"], f(r["r_Hm"], "%.4f"), f(r["kappa_m"], "%.6e"),
                 f(r["2rs_kappa_m"], "%.6f"), f(r["rHm_over_rs"], "%.6f"), r["prediction"], f(r["r_H_dense"], "%.4f"), f(r["slope_dense"], "%.6e"),
                 f(r["test_4gE_dN+v_minus_1"], "%+.4e"), f(r["rHm_over_rH"], "%.6f")))
    L += ["", "- every number above is printed by the script; a dash marks a record with no trapping root on that field."]
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump({"script": "obs_j3p_scope_v2.py", "sigma2_compact_v1": sig2_compact, "run": a.run, "rows": rows}, open(a.out_json, "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
