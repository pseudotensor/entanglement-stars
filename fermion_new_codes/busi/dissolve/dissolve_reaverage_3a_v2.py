#!/usr/bin/env python
"""dissolve_reaverage_3a_v2.py (busi f52b1152, 2026-10-20; DISSOLVE_redteam_reviewa_v1 ruling 3.3 (a)): the fixed-source re-averaging check.
The stored (32, 4) record at T/r_s 1.8125 (the deepest record, index 29 of the kit's ref fields npz): its bare subtracted site sources
eps_tot[29], J_tot[29] (engine_collapse_v36b lines 2032 to 2041, stored at line 2429) are re-solved with the kit's engine_fields_v17
solve_constraints_ivp at sigma/a = ell in (2, 4, 8), the two radial constraints alone (one outward initial-value problem; its
SmoothedSources5 re-averages the same bare source at each ell). Per ell:
  the dense roots and minimum F.trapping_radius(Nn, vn, R, info) (zeros of N - |v| on the dense functions at ell/64);
  the applied fields Na, va = F.site_fields_kernel_average(info, R, ell) and their roots by linear interpolation of Na - |va| on the sites;
  the compactness maxima: max W/(r N^2) on the dense functions (info["W_field"], info["N_dense"] on info["rfine"] inside the collar, the
  engine's compactness_max_2gM_over_r, line 1779) and max (va/Na)^2 on the applied site fields (= W/(r N^2) with W = r v^2);
  the charge F.charge(info, g).
Check at ell 4: the re-solve of the record's own sources must give the stored dense roots 6.163, 27.235 and min -0.1759.
Usage: dissolve_reaverage_3a_v2.py --kit KIT --out DIR [--index 29] [--ells 2,4,8]. CPU, numpy/scipy; seconds to minutes."""
import argparse, json, os, sys, time
import numpy as np


def site_roots(N, v, R):
    f = np.asarray(N, float) - np.abs(np.asarray(v, float))
    z = []
    for i in range(len(R) - 1):
        if f[i] == 0.0:
            z.append(float(R[i]))
        elif f[i] * f[i + 1] < 0.0:
            z.append(float(R[i] - f[i] * (R[i + 1] - R[i]) / (f[i + 1] - f[i])))
    return z, float(f.min()), float(R[int(np.argmin(f))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--index", type=int, default=29); ap.add_argument("--ells", default="2,4,8")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, os.path.join(a.kit, "code"))
    import engine_fields_v17 as F
    bt = "collapse_pair_rs32_ell4a_v36b"
    Z = np.load(os.path.join(a.kit, "ref", bt + "_fields.npz"), allow_pickle=True)
    J = json.load(open(os.path.join(a.kit, "ref", bt + ".json")))
    i = a.index; R = np.asarray(Z["R"], float); g = float(Z["g"]); rs = float(J["args"]["rs"])
    T = float(Z["T"][i]); eps = np.asarray(Z["eps_tot"][i], float); Jt = np.asarray(Z["J_tot"][i], float); N0 = np.asarray(Z["N"][i], float)
    rec = [r for r in J["records"] if abs(r["T_over_rs"] - T / rs) < 1e-9][0]
    L = open(os.path.join(a.out, "dissolve_reaverage_3a_v2.log"), "w")
    pr = lambda s: (L.write(s + "\n"), print(s, flush=True))  # noqa: E731
    pr("dissolve_reaverage_3a_v2: %s fields row %d, T/r_s %.4f, g %.3f, sites %d, stored ell %s; engine_fields_v17 from %s" % (
        bt, i, T / rs, g, len(R), Z["ell"], F.__file__))
    pr("stored record: dense roots %s, min(N-|v|) %.6f, E %.6f, compactness_max %.6f" % (
        ["%.4f" % x for x in rec["trapping_zeros"]], rec["min_N_minus_absv"], rec["E"], rec["s11"]["compactness_max_2gM_over_r"]))
    sz, smin, _ = site_roots(Z["N_applied"][i], Z["v_applied"][i], R)
    pr("stored applied site fields: roots %s, min(Na-|va|) %.6f" % (["%.4f" % x for x in sz], smin))
    SOLVE = dict(rtol=1e-12, atol=1e-16, collar_policy="flag", support_rel=1e-14)
    res = {}
    for ell in [float(x) for x in a.ells.split(",")]:
        t0 = time.time()
        # the lapse passed to the IVP is iterated to the re-solve's own applied lapse (eps, J fixed); one pass if N0 is a seed only
        N0c = N0.copy(); passes = 0; dN1 = float("nan"); dN = float("nan")
        while passes < 30:
            Nn, vn, Wn, info = F.solve_constraints_ivp(eps, Jt, g, R, ell, N0=N0c, **SOLVE)
            passes += 1
            if info.get("failed"):
                break
            Na, va, norms = F.site_fields_kernel_average(info, R, ell)
            dN = float(np.max(np.abs(np.asarray(Na, float) - N0c)))
            if passes == 1:
                dN1 = dN
            if dN < 1e-9:
                break
            N0c = np.asarray(Na, float).copy()
        if info.get("failed"):
            pr("ell %.0f: solve FAILED at pass %d: %s" % (ell, passes, info.get("failed"))); continue
        tz = F.trapping_radius(Nn, vn, R, info)
        az, amin, ar = site_roots(Na, va, R)
        rc = float(info["r_collar"]); rf = np.asarray(info["rfine"], float); rf = rf[(rf > 0) & (rf <= rc)]
        Wf = np.asarray(info["W_field"](rf), float); Nf = np.asarray(info["N_dense"](rf), float)
        cd = Wf / (np.maximum(rf, 1e-300) * np.maximum(Nf, 1e-300) ** 2); jd = int(np.argmax(cd)); jn = int(np.argmin(Nf))
        ca = (np.asarray(va, float) / np.asarray(Na, float)) ** 2; ja = int(np.argmax(ca))
        Nc = float(info["N_dense"](rc)); Wc = float(info["W_field"](rc))
        E = float(F.charge(info, g))
        # the kernel-integrated bare source: the same eps with J = 0 and the lapse 1; equal across ell if the kernel preserves charge
        iz = F.solve_constraints_ivp(eps, np.zeros_like(Jt), g, R, ell, N0=np.ones_like(N0), **SOLVE)[3]
        Ez = float("nan") if iz.get("failed") else float(F.charge(iz, g))
        res[ell] = dict(dense_zeros=tz["zeros"], dense_min=tz["min"], dense_r_min=tz["r_min"], tangencies=tz["tangencies"], applied_zeros=az,
                        applied_min=amin, applied_r_min=ar, compact_dense=float(cd[jd]), compact_dense_r=float(rf[jd]), compact_applied=float(ca[ja]),
                        compact_applied_r=float(R[ja]), charge=E, r_collar=rc, N_collar=Nc, W_collar_over_2g=Wc / (2.0 * g),
                        charge_MS=Wc / (2.0 * g * Nc ** 2), N_dense_min=float(Nf[jn]), N_dense_min_r=float(rf[jn]), charge_J0=Ez,
                        N0_passes=passes, N0_change_first=dN1, N0_change_last=dN, kernel_norms=(float(norms.min()), float(norms.max())),
                        seconds=time.time() - t0)
        x = res[ell]
        pr("ell %.0f: dense roots %s min %.4f at r %.3f | applied roots %s min %.4f at r %.3f | compactness max dense %.4f (r %.2f) applied %.4f (r %.2f) | charge E %.6f | %.0f s" % (
            ell, ["%.3f" % z for z in x["dense_zeros"]], x["dense_min"], x["dense_r_min"], ["%.3f" % z for z in az], amin, ar,
            x["compact_dense"], x["compact_dense_r"], x["compact_applied"], x["compact_applied_r"], E, x["seconds"]))
        pr("ell %.0f: collar r %.2f, N(collar) %.9f, W/2g %.6f, W/(2g N^2) %.6f | N dense min %.4f at r %.2f | charge with J = 0 and N0 = 1: %.6f | N0 passes %d, change after pass 1 %.2e, last %.2e | back-averaging kernel norms %.9f to %.9f" % (
            ell, rc, Nc, x["W_collar_over_2g"], x["charge_MS"], x["N_dense_min"], x["N_dense_min_r"], Ez, passes, dN1, dN, x["kernel_norms"][0], x["kernel_norms"][1]))
    if 4.0 in res:
        x = res[4.0]; z = x["dense_zeros"]
        ok = len(z) == 2 and abs(z[0] - 6.163) < 5e-4 and abs(z[1] - 27.235) < 5e-4 and abs(x["dense_min"] + 0.1759) < 5e-5
        pr("CHECK ell 4 (the stored record's own sources): dense roots %s min %.6f against 6.163, 27.235, -0.1759: %s; against the stored json: %s, %.6f; charge %.6f against the stored E %.6f" % (
            ["%.6f" % v_ for v_ in z], x["dense_min"], "PASS" if ok else "FAIL", ["%.6f" % v_ for v_ in rec["trapping_zeros"]], rec["min_N_minus_absv"], x["charge"], rec["E"]))
    ks = sorted(res); Ez = np.array([res[k]["charge_J0"] for k in ks], float)
    if len(ks) > 1 and np.all(np.isfinite(Ez)):
        rel = float(Ez.max() - Ez.min()) / abs(float(Ez.mean()))
        pr("CHECK kernel charge (the same eps, J = 0, N0 = 1) over ell %s: %s, relative spread %.2e: %s" % (
            ["%.0f" % k for k in ks], ["%.9f" % e_ for e_ in Ez], rel, "PASS" if rel < 1e-8 else "FAIL"))
        pr("CHECK lapse at the collar over ell %s: %s: %s" % (
            ["%.0f" % k for k in ks], ["%.9f" % res[k]["N_collar"] for k in ks], "PASS" if all(abs(res[k]["N_collar"] - 1.0) < 1e-8 for k in ks) else "NOT 1"))
    L.close()
    json.dump({"T_over_rs": T / rs, "index": i, "results": {str(k): v for k, v in res.items()}}, open(os.path.join(a.out, "dissolve_reaverage_3a_v2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
