#!/usr/bin/env python
"""PAGE v4.2 issue 1 (inbox #205): audit every stored exterior spectrum eig_X_{plus,minus}_cut<c> of the B2 production channels kappa 1..18
for eigenvalues outside [0, 1]; for each failure compare with the same checkpoint's interior spectrum (the Gram pairing
Phi_X^dag Phi_X + Phi_Xc^dag Phi_Xc = 1 for orthonormal columns, so eig_X must equal 1 - [eig_Xc, 0...]), the trace, the - sector's
exterior entropy and the stored state diagnostics; then recompute the T/r_s 50 per-channel excesses and the 18-channel sums from the stored
interior spectra (the readout route) and the direct-route variants (exterior clipped / exterior replaced by the paired spectrum).
Stored data only. Usage: python page_v42_exterior_spectra_audit_v1.py PROD_DIR OUT_JSON"""
import sys, json, re, glob, os, hashlib
import numpy as np

PROD = sys.argv[1]; OUT = sys.argv[2]
TOL = 1e-8


def s_bin(x):
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    m = (x > 0) & (x < 1)
    y = x[m]
    out[m] = -y * np.log(y) - (1 - y) * np.log1p(-y)
    return out


def S(x):
    return float(s_bin(x).sum())


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


res = {"prod": PROD, "tol": TOL, "files": {}, "all_out_of_range": [], "failures": [], "scan": {}, "T50": {}}
gmin, gmax, nclip = np.inf, -np.inf, 0
for K in range(1, 19):
    pn = os.path.join(PROD, "v6o_B2_k%d_d13.npz" % K); pj = os.path.join(PROD, "v6o_B2_k%d_d13.json" % K)
    res["files"][K] = {"npz": os.path.realpath(pn), "npz_md5": md5(pn), "json": os.path.realpath(pj), "json_md5": md5(pj)}
    d = np.load(pn); J = json.load(open(pj))
    clocks = {round(v["t_over_rs"], 6): v for v in J["clocks"].values()}
    nspec = 0
    for key in d.files:
        m_ = re.match(r"T(\d+\.\d+)_k(\d+)_eig_X_(plus|minus)_cut(\d+)$", key)
        if not m_:
            continue
        nspec += 1
        T = float(m_.group(1)); sec = m_.group(3); cut = m_.group(4)
        ev = d[key]
        gmin = min(gmin, ev.min()); gmax = max(gmax, ev.max())
        lo = ev[ev < 0]; hi = ev[ev > 1]
        nclip += int(((ev < 0) & (ev >= -TOL)).sum() + ((ev > 1) & (ev <= 1 + TOL)).sum())
        if lo.size or hi.size:
            res["all_out_of_range"].append({"K": K, "T": T, "sector": sec, "cut": cut, "n_below0": int(lo.size), "n_above1": int(hi.size),
                                            "min": float(ev.min()), "max": float(ev.max())})
        if ev.min() < -TOL or ev.max() > 1 + TOL:
            evc = d["T%s_k%d_eig_Xc_%s_cut%s" % (m_.group(1), K, sec, cut)]
            m = ev.size
            paired = np.sort(1.0 - np.concatenate([evc, np.zeros(m - evc.size)]))
            evs = np.sort(ev)
            dev = np.abs(evs - paired)
            pc = clocks[round(T, 6)]["per_channel"][str(K)]
            cj = pc["cuts"][cut]
            above = np.sort(ev[ev > 1 + TOL])[::-1]
            f = {"K": K, "T": T, "sector": sec, "cut": cut, "key": key, "m": int(m), "n_Xc_eigs": int(evc.size),
                 "max": float(ev.max()), "min": float(ev.min()), "dist_above_1": float(ev.max() - 1.0),
                 "eigs_above_1_plus_tol": above[:6].tolist(), "n_above_1_plus_tol": int(above.size),
                 "n_below_0_minus_tol": int((ev < -TOL).sum()),
                 "trace_X": float(ev.sum()), "trace_Xc": float(evc.sum()), "trace_sum": float(ev.sum() + evc.sum()),
                 "N_occ_plus_json": pc.get("trace_norm", {}).get("N_occ_plus"),
                 "pair_dev_max": float(dev.max()), "n_pair_dev_gt_1e-6": int((dev > 1e-6).sum()),
                 "pair_dev_max_excluding_top": float(np.sort(dev)[:-max(1, above.size)].max()),
                 "S_X_stored_this_sector": cj.get("S_X_plus" if sec == "plus" else "S_X_minus"),
                 "S_X_recomputed_raw_in_open_interval": S(ev), "S_X_clipped": S(np.clip(ev, 0, 1)), "S_X_paired": S(paired),
                 "S_Xc_this_sector": cj.get("S_Xc_plus" if sec == "plus" else "S_Xc_minus"),
                 "S_Xc_recomputed": S(evc),
                 "S_X_other_sector": cj.get("S_X_minus" if sec == "plus" else "S_X_plus"),
                 "S_Xc_other_sector": cj.get("S_Xc_minus" if sec == "plus" else "S_Xc_plus"),
                 "purity_gap_plus": cj.get("purity_gap_plus"), "purity_gap_minus": cj.get("purity_gap_minus"),
                 "pairing_defect_state": cj.get("pairing_defect_state"), "route_X_check": cj.get("route_X_check"),
                 "excess_json": cj.get("excess"), "trace_norm_nu": pc.get("trace_norm", {}).get("nu"),
                 "floor_T_meas": pc.get("sector", {}).get("floor", {}).get("T_meas"),
                 "other_cuts_purity_gap": {c_: pc["cuts"][c_].get("purity_gap") for c_ in pc["cuts"]}}
            res["failures"].append(f)
    res["scan"][K] = nspec
    # T/r_s 50: the readout route per channel and cut (interior spectra) against the json excess
    key50 = [k for k in J["clocks"] if abs(J["clocks"][k]["t_over_rs"] - 50.0) < 1e-9][0]
    pc = J["clocks"][key50]["per_channel"][str(K)]
    for cut in ("2", "16", "24", "30", "48"):
        evc = d["T%s_k%d_eig_Xc_plus_cut%s" % (key50, K, cut)]; evv = d["T%s_k%d_eig_Xc_vac_plus_cut%s" % (key50, K, cut)]
        x_read = 2 * S(evc) - 2 * S(evv)
        r = {"x_readout_recomputed": x_read, "excess_json": pc["cuts"][cut]["excess"]}
        kx = "T%s_k%d_eig_X_plus_cut%s" % (key50, K, cut)
        if kx in d.files:
            evx = d[kx]; evxm = d["T%s_k%d_eig_X_minus_cut%s" % (key50, K, cut)]
            evvm = d["T%s_k%d_eig_Xc_vac_minus_cut%s" % (key50, K, cut)]
            m = evx.size
            paired = 1.0 - np.concatenate([evc, np.zeros(m - evc.size)])
            r["x_direct_clipped"] = S(np.clip(evx, 0, 1)) + S(np.clip(evxm, 0, 1)) - S(evv) - S(evvm)
            r["x_direct_paired"] = S(paired) + S(np.clip(evxm, 0, 1)) - S(evv) - S(evvm)
        res["T50"].setdefault(cut, {})[K] = r
res["global_min"] = float(gmin); res["global_max"] = float(gmax); res["n_endpoint_clips_within_tol"] = nclip
for cut, rows in res["T50"].items():
    tot = {}
    for nm in ("x_readout_recomputed", "excess_json", "x_direct_clipped", "x_direct_paired"):
        if all(nm in rows[K] for K in rows):
            tot[nm] = float(sum(16 * K * rows[K][nm] for K in rows))
    rows["SUM16k"] = tot
json.dump(res, open(OUT, "w"), indent=1, default=float)
print("global min %.17g max %.17g clips %d failures %d" % (gmin, gmax, nclip, len(res["failures"])))
for f in res["failures"]:
    print("K %2d T %5.1f %s cut %s max %.17g dist %.3e n_above %d trace_sum %.12f Nocc %s pairdev %.3e (excl top %.3e) S_X(stored) %s S_X(clip) %.10f S_X(paired) %.10f S_Xc %.10f S_X_other %s pg+ %s" % (
        f["K"], f["T"], f["sector"], f["cut"], f["max"], f["dist_above_1"], f["n_above_1_plus_tol"], f["trace_sum"], f["N_occ_plus_json"], f["pair_dev_max"],
        f["pair_dev_max_excluding_top"], f["S_X_stored_this_sector"], f["S_X_clipped"], f["S_X_paired"], f["S_Xc_this_sector"], f["S_X_other_sector"], f["purity_gap_plus"]))
for cut in ("2", "16", "24", "30", "48"):
    print("T50 cut", cut, res["T50"][cut]["SUM16k"])
