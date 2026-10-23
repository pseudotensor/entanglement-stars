#!/usr/bin/env python
"""gate_profiles_v1.py (busi, GATECOMP step (iii); ReviewA_gatecomp_v1 secs. 1, 2(iii), 3.1, 3.2; ReviewB_gatecomp_v1 conditions 1, 2).

DATA script (box CPU, no card, no evolution). For one member's stored records (the run's _fields.npz: the shell totals eps_tot, J_tot per
record) it re-solves the dense fields N, v per record with the field module's own ivp route (engine_fields_v16.solve_constraints_ivp, the
route of the solve lane's engine_core_slices_v3 six-term code), and evaluates on a fine radial grid from the regular centre outwards:

  one-term       sigma |H|, H = -dv/dr                                                     (paper 05_star: the one-term hydrodynamic test)
  operator norm  sigma ||K||_op = sigma max(|k_r|, |k_t|), k_r = v'/N, k_t = v/(N r)       (K's Cartesian eigenvalues: k_r once, k_t twice)
  six-term       sigma Lambda_geom, Lambda_geom = max(|v'/N|, |v/(N r)|, |N'/N|, sqrt|D k_r|, sqrt|D k_t|, sqrt|N''/N|),
                 D = (d_T + v d_r)/N, d_r k_r = (v'' N - v' N')/N^2, d_r k_t = v'/(r N) - v/(r^2 N) - v N'/(r N^2), d_T by the central
                 difference of the re-solved k between the adjacent records (one-sided at the ends, labelled)
                 (definition recovered from solve/code/engine/engine_core_slices_v3.py lines 7, 13-17, 69-78, 187-216; S7 of WEB40F)
  supplementary  the Cartesian-complete extra terms NOT in the six-term definition: sqrt|N'/(N r)| (the lapse Hessian's transverse
                 eigenvalue), sqrt|d_r k_r|, sqrt|d_r k_t|, sqrt|(k_r - k_t)/r| (the angular part of grad K); printed, never gated.

Derivatives: on the field module's dense callables, extended through the centre by parity (N even, v odd: the regular centre), central
differences at h and h/2 Richardson-combined (engine_rows_v2.deriv's rule) at h = sigma/32, and again at h = sigma/64 (the convergence check).
The regular-centre limits are evaluated at r = 0 explicitly: k_t(0) = v'(0)/N(0) = k_r(0), N'/(N r) -> N''(0)/N(0), d_r k(0) = 0.
sigma = ell (the kernel's per-axis standard deviation; the cell-average a^2/12 adds 0.26 percent at ell 4, printed in the analysis).

Zones per record (the maximum of each term and where): 'centroid' [0, R_c] (R_c the record's dust-energy centroid, the paper's domain);
'reviewa' [0, 2 R_t] (reviewer A 3.1; before the first root R_t := the birth radius passed in --R-birth); 'root' [R_t - 3 sigma, R_t + 3 sigma]
(the root's averaging neighbourhood); 'all' [0, max(R_c, 2 R_t)].  Also: the dense min(N - |v|) on the fine grid and its radius, the dense
zeros of N - |v| (the module's trapping_radius), the centre's H(0), and the record's own stored values beside (check).
Output: one JSON (no fields, no slices). Every number printed by the script; nothing is ruled here."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np


def sym(f, parity):
    def g(x):
        x = np.asarray(x, float)
        y = np.asarray(f(np.abs(x)), float)
        return y if parity > 0 else np.sign(x) * y
    return g


def deriv(F, r, h):
    f0 = F(r)
    d1 = lambda hh: (F(r + hh) - F(r - hh)) / (2.0 * hh)  # noqa: E731
    d2 = lambda hh: (F(r + hh) - 2.0 * f0 + F(r - hh)) / (hh * hh)  # noqa: E731
    return f0, (4.0 * d1(0.5 * h) - d1(h)) / 3.0, (4.0 * d2(0.5 * h) - d2(h)) / 3.0


def kfields(Nf, vf, r, h):
    N, N1, N2 = deriv(Nf, r, h)
    v, v1, v2 = deriv(vf, r, h)
    with np.errstate(divide="ignore", invalid="ignore"):
        kr = v1 / N
        kt = np.where(r > 0, v / (r * N), v1 / N)
        kr1 = (v2 * N - v1 * N1) / N ** 2
        kt1 = np.where(r > 0, v1 / (r * N) - v / (r * r * N) - v * N1 / (r * N * N), 0.0)
        kr1 = np.where(r > 0, kr1, 0.0)
        Ntr = np.where(r > 0, N1 / (r * N), N2 / N)
    return {"N": N, "N1": N1, "N2": N2, "v": v, "v1": v1, "v2": v2, "kr": kr, "kt": kt, "kr1": kr1, "kt1": kt1, "Ntr": Ntr}


TERMS6 = ["abs(v'/N)", "abs(v/(N r))", "abs(N'/N)", "sqrt(abs(D k_r))", "sqrt(abs(D k_t))", "sqrt(abs(N''/N))"]
SUPP = ["sqrt(abs(N'/(N r)))", "sqrt(abs(d_r k_r))", "sqrt(abs(d_r k_t))", "sqrt(abs((k_r - k_t)/r))"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--json", required=True); ap.add_argument("--label", required=True)
    ap.add_argument("--fm-dir", required=True); ap.add_argument("--fm", default="engine_fields_v16")
    ap.add_argument("--T-from", type=float, required=True); ap.add_argument("--T-to", type=float, required=True)
    ap.add_argument("--R-birth", type=float, default=None, help="the birth radius (a) used as R_t before the first root")
    ap.add_argument("--grid-frac", type=float, default=8.0, help="grid spacing sigma/grid_frac")
    ap.add_argument("--uncertified", default="", help="comma list of T/r_s whose record is not certified (labelled)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("refusing to overwrite %s" % a.out)
    sys.path.insert(0, a.fm_dir)
    FM = __import__(a.fm)
    J = json.load(open(a.json)); args = J["args"]; rs = float(args["rs"])
    recs = {round(float(r["T"]) / rs, 6): r for r in J["records"] if r.get("E") is not None}
    Z = np.load(a.data, allow_pickle=True)
    Rd = np.asarray(Z["R"], float); Td = np.asarray(Z["T"], float); g = float(Z["g"]); ell = float(Z["ell"])
    sig = ell; unc = set(round(float(x), 6) for x in a.uncertified.split(",") if x)
    sel = [k for k in range(len(Td)) if a.T_from - 1e-9 <= Td[k] / rs <= a.T_to + 1e-9]
    need = sorted(set(sel) | {k - 1 for k in sel if k > 0} | {k + 1 for k in sel if k + 1 < len(Td)})
    t0 = time.time(); sol = {}
    for k in need:
        _, _, _, info = FM.solve_constraints_ivp(np.asarray(Z["eps_tot"][k], float), np.asarray(Z["J_tot"][k], float), g, Rd, ell,
                                                 collar_policy="flag", support_rel=1e-14)
        if info.get("failed"):
            sol[k] = None; print("T/rs %.4f: ivp failed %s" % (Td[k] / rs, info.get("failed")), flush=True); continue
        tz = FM.trapping_radius(None, None, Rd, info)
        sol[k] = {"Nf": sym(info["N_dense"], +1), "vf": sym(info["v_dense"], -1), "zeros": [float(z) for z in tz["zeros"]],
                  "min": tz.get("min"), "r_min": tz.get("r_min")}
        print("solved T/rs %.4f (%.0f s)" % (Td[k] / rs, time.time() - t0), flush=True)
    rows = []
    for k in sel:
        if sol.get(k) is None:
            continue
        T = Td[k] / rs; rec = recs.get(round(T, 6), {})
        Rc = rec.get("R_c"); S = sol[k]; zeros = S["zeros"]; Rt = zeros[-1] if zeros else None
        Rt_eff = Rt if Rt is not None else a.R_birth
        rmax = max(Rc or 0.0, 2.0 * (Rt_eff or 0.0)) + 3.0 * sig
        hg = sig / a.grid_frac
        r = np.arange(0.0, rmax + 1e-9, hg)
        out_h = {}
        for tag, h in (("h32", sig / 32.0), ("h64", sig / 64.0)):
            k0 = kfields(S["Nf"], S["vf"], r, h)
            kp = kfields(sol[k - 1]["Nf"], sol[k - 1]["vf"], r, h) if sol.get(k - 1) else None
            kn = kfields(sol[k + 1]["Nf"], sol[k + 1]["vf"], r, h) if sol.get(k + 1) else None
            if kp is not None and kn is not None:
                dTr = (kn["kr"] - kp["kr"]) / (Td[k + 1] - Td[k - 1]); dTt = (kn["kt"] - kp["kt"]) / (Td[k + 1] - Td[k - 1]); dmode = "central"
            elif kp is not None:
                dTr = (k0["kr"] - kp["kr"]) / (Td[k] - Td[k - 1]); dTt = (k0["kt"] - kp["kt"]) / (Td[k] - Td[k - 1]); dmode = "one-sided (backward)"
            else:
                dTr = (kn["kr"] - k0["kr"]) / (Td[k + 1] - Td[k]); dTt = (kn["kt"] - k0["kt"]) / (Td[k + 1] - Td[k]); dmode = "one-sided (forward)"
            Dkr = (dTr + k0["v"] * k0["kr1"]) / k0["N"]; Dkt = (dTt + k0["v"] * k0["kt1"]) / k0["N"]
            with np.errstate(divide="ignore", invalid="ignore"):
                angK = np.where(r > 0, np.abs((k0["kr"] - k0["kt"]) / np.where(r > 0, r, 1.0)), 0.0)
            T6 = {"abs(v'/N)": np.abs(k0["kr"]), "abs(v/(N r))": np.abs(k0["kt"]), "abs(N'/N)": np.abs(k0["N1"] / k0["N"]),
                  "sqrt(abs(D k_r))": np.sqrt(np.abs(Dkr)), "sqrt(abs(D k_t))": np.sqrt(np.abs(Dkt)), "sqrt(abs(N''/N))": np.sqrt(np.abs(k0["N2"] / k0["N"]))}
            TS = {"sqrt(abs(N'/(N r)))": np.sqrt(np.abs(k0["Ntr"])), "sqrt(abs(d_r k_r))": np.sqrt(np.abs(k0["kr1"])),
                  "sqrt(abs(d_r k_t))": np.sqrt(np.abs(k0["kt1"])), "sqrt(abs((k_r - k_t)/r))": np.sqrt(angK)}
            H = -k0["v1"]
            Kop = np.maximum(np.abs(k0["kr"]), np.abs(k0["kt"]))
            L6 = np.max(np.array([T6[t] for t in TERMS6]), axis=0)
            LS = np.max(np.array([TS[t] for t in SUPP]), axis=0)
            Fm = k0["N"] - np.abs(k0["v"])
            zones = {"centroid": (0.0, Rc), "reviewa": (0.0, 2.0 * Rt_eff if Rt_eff else None),
                     "root": ((Rt_eff - 3.0 * sig, Rt_eff + 3.0 * sig) if Rt_eff else (None, None)),
                     "all": (0.0, max(Rc or 0.0, 2.0 * (Rt_eff or 0.0)))}
            zout = {}
            for zn, (lo, hi) in zones.items():
                if hi is None:
                    zout[zn] = None; continue
                m = (r >= max(lo, 0.0) - 1e-12) & (r <= hi + 1e-12)
                if not np.any(m):
                    zout[zn] = None; continue
                def mx(arr):
                    i = int(np.argmax(np.where(m, arr, -np.inf))); return [float(sig * arr[i]), float(r[i])]
                zz = {"range": [float(max(lo, 0.0)), float(hi)], "n": int(np.sum(m)),
                      "sigma_H": mx(np.abs(H)), "sigma_Kop": mx(Kop), "sigma_six": mx(L6), "sigma_supp": mx(LS),
                      "six_terms": {t: mx(T6[t]) for t in TERMS6}, "supp_terms": {t: mx(TS[t]) for t in SUPP}}
                d6 = TERMS6[int(np.argmax([zz["six_terms"][t][0] for t in TERMS6]))]; zz["six_dominant"] = d6
                zout[zn] = zz
            i0 = int(np.argmin(Fm))
            out_h[tag] = {"dT_mode": dmode, "zones": zout, "centre": {"H0": float(H[0]), "kr0": float(k0["kr"][0]), "kt0": float(k0["kt"][0]),
                          "N0": float(k0["N"][0]), "sigma_six_r0": float(sig * L6[0]), "sigma_six_r0_terms": {t: float(sig * T6[t][0]) for t in TERMS6}},
                          "fine_min_N_minus_absv": [float(Fm[i0]), float(r[i0])],
                          "profile_every_sigma": {"r": r[::int(a.grid_frac)].tolist(), "sigma_H": (sig * np.abs(H))[::int(a.grid_frac)].tolist(),
                                                  "sigma_Kop": (sig * Kop)[::int(a.grid_frac)].tolist(), "sigma_six": (sig * L6)[::int(a.grid_frac)].tolist()}}
        row = {"T_over_rs": float(T), "record_certified": (round(T, 6) not in unc) and bool(rec.get("converged", True)),
               "record": {k_: rec.get(k_) for k_ in ("E", "E_over_E0", "R_t", "r_in", "R_c", "min_N_minus_absv", "r_min_N_minus_absv", "converged")},
               "dense_zeros": zeros, "dense_min": S["min"], "dense_r_min": S["r_min"], "R_t_used": Rt_eff, "R_t_is_birth_proxy": Rt is None,
               "results": out_h}
        rows.append(row)
        h = out_h["h32"]["zones"]
        f = lambda z, q: ("%.4f@%.1f" % tuple(h[z][q])) if h.get(z) else "-"  # noqa: E731
        print("T/rs %.4f  R_t %s R_c %s | centroid: H %s Kop %s six %s | reviewa: H %s Kop %s six %s | root: Kop %s six %s" % (
            T, ("%.2f" % Rt) if Rt else "-", ("%.2f" % Rc) if Rc else "-", f("centroid", "sigma_H"), f("centroid", "sigma_Kop"), f("centroid", "sigma_six"),
            f("reviewa", "sigma_H"), f("reviewa", "sigma_Kop"), f("reviewa", "sigma_six"), f("root", "sigma_Kop"), f("root", "sigma_six")), flush=True)
    meta = {"script": "gate_profiles_v1.py", "label": a.label, "args": vars(a), "rs": rs, "sigma": sig, "ell": ell, "g": g, "E0": J["head"].get("E0"),
            "two_gE0": 2.0 * g * float(J["head"].get("E0")) if J["head"].get("E0") else None, "nsites": len(Rd), "field_module": FM.__file__,
            "terms6": TERMS6, "supp": SUPP, "seconds": time.time() - t0}
    json.dump({"meta": meta, "rows": rows}, open(a.out, "x"), indent=1, default=float)
    print("wrote %s (%.0f s)" % (a.out, time.time() - t0))


if __name__ == "__main__":
    main()
