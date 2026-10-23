#!/usr/bin/env python
"""engine_ps_defect_v1.py (builder analyst; engine spec 28.5: "GAP13's exact point-sampling defect (s_ell^2/2) a^3 sum_x (rho_x laplacian N
+ J_x . laplacian v) is computed on the production's records (CPU) as the quantity the point-sampled control's E difference is read against
at matched times"). GAP13 (4.4): D_ps = a^3 sum_x [rho_x (N_x - N(x)) + J_x,i (v_x^i - v^i(x))], exact for any fields, N_x and v_x the
back-averaged site fields (the adjoints of the source kernels) and N(x), v(x) the dense fields sampled at the site; (4.5) its leading order
(s_ell^2/2) a^3 sum_x [rho_x lap N(x) + J_x,i lap v^i(x)], s_ell^2 the kernel's second moment per direction, with the radial forms
lap N = N'' + 2N'/r and lap(v r_hat) . r_hat = v'' + 2v'/r - 2v/r^2. Per record of a COLLAPSE_records data npz: the dense fields
re-solved from the stored shell totals by the run's field module per segment (engine_certificate_v2's route), the back-averaged site
fields by that module's site_fields_kernel_average (the engine's own kernels K_0, K_1), the point-sampled fields N_dense(R_n), v_dense(R_n);
the engine's pairing sum_n (N_n eps_n + v_n J_n) with its sources eps_tot, J_tot (the first variation of the boundary charge, the module's
docstring); D_exact = sum_n [eps_n (N_n - N(R_n)) + J_n (v_n - v(R_n))] and its clock and flow parts; D_lead with s_ell^2 measured on the
module's own kernel ((A*r^2)_n - R_n^2)/3 (its median and range printed, beside GAP13's ell^2/15 + a^2/12 at a = 1), the derivatives by
Richardson-combined central differences at ell/32 (sites with R_n below 2 ell/32 left out of D_lead, their share printed); the record's
applied fields against the re-solved back-averaged ones (the solve's consistency). With --ps-json (the point-sampled control's records): at
every common time the two runs' E and their difference E_avg - E_ps beside D_exact. Every number printed by the script. CPU.
Args: --data --json (the run's record jsons, comma list; for E) --ps-json --fm-segments --fm-default --every --md --out-json."""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_rows_v2 as RW  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--json", required=True); ap.add_argument("--ps-json", default=None)
    ap.add_argument("--fm-segments", default="v17 (replayed)=engine_fields_v12;v19 (native)=engine_fields_v14"); ap.add_argument("--fm-default", default="engine_fields_v12")
    ap.add_argument("--every", type=int, default=4); ap.add_argument("--records", type=int, default=None)
    ap.add_argument("--md", required=True); ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    Z = np.load(a.data, allow_pickle=True)
    R = np.asarray(Z["R"], float); T = np.asarray(Z["T"], float); g = float(Z["g"]); ell = float(Z["ell"]); rs = float(Z["rs"])
    seg_names = [str(x) for x in Z["segment_names"]] if "segment_names" in Z.files else None
    seg = np.asarray(Z["segment"], int) if "segment" in Z.files else np.zeros(len(T), int)
    fmap = dict(x.split("=", 1) for x in a.fm_segments.split(";"))
    Erec = {}
    for f in a.json.split(","):
        for r in json.load(open(f))["records"]:
            if r.get("E") is not None:
                Erec.setdefault(round(float(r["T"]), 6), float(r["E"]))
    Eps = {}
    if a.ps_json:
        for r in json.load(open(a.ps_json))["records"]:
            if r.get("E") is not None:
                Eps[round(float(r["T"]), 6)] = float(r["E"])
    fms = {}
    h = ell / 32.0
    s2 = {}
    rows = []; t0 = time.time()
    K = len(T) if a.records is None else min(len(T), a.records)
    for k in range(K):
        name = fmap.get(seg_names[seg[k]], a.fm_default) if seg_names else a.fm_default
        if name not in fms:
            fms[name] = importlib.import_module(name)
            q = fms[name].site_fields_kernel_average({"N_dense": (lambda r_: np.asarray(r_, float) ** 2), "v_dense": (lambda r_: np.zeros_like(np.asarray(r_, float)))}, R, ell)[0]
            s2[name] = (np.asarray(q, float) - R ** 2) / 3.0
        FM = fms[name]
        eps = np.asarray(Z["eps_tot"][k], float); Jt = np.asarray(Z["J_tot"][k], float)
        _, _, _, info = FM.solve_constraints_ivp(eps, Jt, g, R, ell, collar_policy="flag", support_rel=1e-14)
        if info.get("failed"):
            rows.append({"T_over_rs": T[k] / rs, "failed": True}); continue
        Nx, vx = [np.asarray(x, float) for x in FM.site_fields_kernel_average(info, R, ell)[:2]]
        Nd = np.asarray(info["N_dense"](R), float); vd = np.asarray(info["v_dense"](R), float)
        Dc = float(np.sum(eps * (Nx - Nd))); Df = float(np.sum(Jt * (vx - vd)))
        Em = float(np.sum(Nx * eps + vx * Jt))
        ok = R >= 2.0 * h
        _, N1, N2 = RW.deriv(info["N_dense"], R[ok], h)
        v0, v1, v2 = RW.deriv(info["v_dense"], R[ok], h)
        lapN = N2 + 2.0 * N1 / R[ok]; lapv = v2 + 2.0 * v1 / R[ok] - 2.0 * v0 / R[ok] ** 2
        s2m = float(np.median(s2[name]))
        Dl_c = 0.5 * s2m * float(np.sum(eps[ok] * lapN)); Dl_f = 0.5 * s2m * float(np.sum(Jt[ok] * lapv))
        left = float(np.sum(np.abs(eps[~ok]))) / max(float(np.sum(np.abs(eps))), 1e-300)
        dNa = float(np.max(np.abs(np.asarray(Z["N_applied"][k], float) - Nx))); dva = float(np.max(np.abs(np.asarray(Z["v_applied"][k], float) - vx)))
        key = round(float(T[k]), 6)
        row = {"T_over_rs": T[k] / rs, "segment": seg_names[seg[k]] if seg_names else None, "field_module": name, "E": Erec.get(key), "E_ps": Eps.get(key),
               "E_m": Em, "D_exact": Dc + Df, "D_exact_clock": Dc, "D_exact_flow": Df, "D_lead": Dl_c + Dl_f, "D_lead_clock": Dl_c, "D_lead_flow": Dl_f,
               "left_out_eps_share": left, "applied_minus_averaged_N": dNa, "applied_minus_averaged_v": dva}
        rows.append(row)
        if k % 20 == 0:
            print("record %d (T/rs %.3f) %.0f s: D_exact %+.4e, D_lead %+.4e" % (k, T[k] / rs, time.time() - t0, row["D_exact"], row["D_lead"]), flush=True)
    good = [r for r in rows if not r.get("failed")]
    L = ["## GAP13's point-sampling defect on %s (engine_ps_defect_v1.py; engine spec 28.5)" % os.path.basename(a.data), "",
         "- D_exact = sum_n [eps_n (N_n - N(R_n)) + J_n (v_n - v(R_n))] (GAP13 (4.4)) with the engine's back-averaged site fields N_n, v_n (site_fields_kernel_average of the run's field module per segment) and the dense fields at the sites; D_lead = (s_ell^2/2) sum_n [eps_n lap N + J_n lap v] (GAP13 (4.5)); E_m = sum_n (N_n eps_n + v_n J_n); sources eps_tot, J_tot of the record.",
         "- s_ell^2 on the module's own kernel, ((A*r^2)_n - R_n^2)/3 over the sites: " + "; ".join("%s median %.6f, range %.6f to %.6f" % (nm, float(np.median(v_)), float(np.min(v_)), float(np.max(v_))) for nm, v_ in s2.items())
         + "; GAP13's ell^2/15 + a^2/12 at a = 1: %.6f." % (ell * ell / 15.0 + 1.0 / 12.0),
         "- records: %d (%d failed solves); the largest |applied - re-solved back-averaged| over the records: N %.2e, v %.2e; the largest share of |eps| on the sites left out of D_lead: %.2e." % (
             len(rows), len(rows) - len(good), max((r["applied_minus_averaged_N"] for r in good), default=float("nan")), max((r["applied_minus_averaged_v"] for r in good), default=float("nan")),
             max((r["left_out_eps_share"] for r in good), default=float("nan"))), "",
         "| T/rs | E (this run) | E (point-sampled control) | E - E_ps | D_exact (clock, flow) | D_lead (clock, flow) | D_exact/E_m |", "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(good):
        if i % a.every == 0 or r["E_ps"] is not None or i == len(good) - 1:
            L.append("| %.3f | %s | %s | %s | %+.4e (%+.3e, %+.3e) | %+.4e (%+.3e, %+.3e) | %+.3e |" % (
                r["T_over_rs"], ("%.8f" % r["E"]) if r["E"] is not None else "-", ("%.8f" % r["E_ps"]) if r["E_ps"] is not None else "-",
                ("%+.4e" % (r["E"] - r["E_ps"])) if r["E"] is not None and r["E_ps"] is not None else "-",
                r["D_exact"], r["D_exact_clock"], r["D_exact_flow"], r["D_lead"], r["D_lead_clock"], r["D_lead_flow"], r["D_exact"] / r["E_m"] if r["E_m"] else float("nan")))
    L += ["", "- rows every %d-th record, every record with a point-sampled E at the same time, and the last; the full list in the json; every number above is printed by the script; the reading of E - E_ps against D_exact is the lead's." % a.every]
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump({"script": "engine_ps_defect_v1.py", "data": a.data, "s2": {k_: [float(np.median(v_)), float(np.min(v_)), float(np.max(v_))] for k_, v_ in s2.items()}, "rows": rows},
              open(a.out_json, "w"), indent=1, default=float)
    print("\n".join(L[:6]))


if __name__ == "__main__":
    main()
