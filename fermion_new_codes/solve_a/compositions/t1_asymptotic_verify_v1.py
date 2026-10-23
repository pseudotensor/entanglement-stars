#!/usr/bin/env python
"""T1 readout verification (NOTES_T1_asymptotic_readout_v1): an independent recomputation of the headline T1 numbers of
ASYMPTOTIC_tests_v1b (busi fundmarks/t1) from the core_well_v1 per-slice json/npz, and the composition the paper's sentence
"Outside R_50 the law matches the clock's fall to 1 to 3 percent" needs: the exact law's integrated ln N change over
R_50 -> outer root (and R_c -> root, R_50 -> R_90) against the measured one, per rooted slice, both kernels, dense and applied.

Reads only: <internal>/well_<h>_T<T>.{json,npz} and the T1 output json(s).
Numpy only, box CPU. Writes one json + one md to --out.
"""
import argparse, json, math, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--well", default="<internal>/well")
ap.add_argument("--t1", default="<internal>/ASYMPTOTIC_tests_v1b.json")
ap.add_argument("--t1r", default="<internal>/ASYMPTOTIC_tests_v1b.json")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
t0 = time.time()

SL = {"prod": [12, 15, 16, 20, 24, 25, 28, 30, 35, 40], "s8": [12, 13, 14, 16, 20, 25, 30, 40]}


def shell3d(r, Rn, src, ell):
    # written independently from the closed form: a thin shell of total q at radius Rn convolved with a 3D isotropic
    # Gaussian of std ell has density q/(4 pi r Rn) * [G(r-Rn) - G(r+Rn)], G the 1D normal density of std ell.
    out = np.zeros_like(r, dtype=float)
    c = 1.0 / (math.sqrt(2 * math.pi) * ell)
    for Rk, q in zip(Rn, src):
        if q == 0.0:
            continue
        out += q * c * (np.exp(-0.5 * ((r - Rk) / ell) ** 2) - np.exp(-0.5 * ((r + Rk) / ell) ** 2)) / (4 * math.pi * r * Rk)
    return out


def radial1d(r, Rn, src, ell):
    out = np.zeros_like(r, dtype=float)
    c = 1.0 / (math.sqrt(2 * math.pi) * ell)
    for Rk, q in zip(Rn, src):
        if q == 0.0:
            continue
        out += q * c * (np.exp(-0.5 * ((r - Rk) / ell) ** 2) + np.exp(-0.5 * ((r + Rk) / ell) ** 2))
    return out / (4 * math.pi * r ** 2)


def ival(x, R, f):
    return float(np.interp(x, R, f))


def law_integral(R, lnM, half, R1, R2):
    """integral of (lambda-1)/2 d ln M_w from R1 to R2 by the trapezoid on the site grid plus the two end pieces."""
    sel = (R > R1) & (R < R2)
    xs = np.concatenate([[R1], R[sel], [R2]])
    lm = np.interp(xs, R, lnM)
    hf = np.interp(xs, R, half)
    return float(np.sum(0.5 * (hf[1:] + hf[:-1]) * np.diff(lm)))


def fitll(x, y, sel):
    X, Y = np.log(x[sel]), np.log(y[sel])
    A = np.vstack([X, np.ones_like(X)]).T
    c, res, *_ = np.linalg.lstsq(A, Y, rcond=None)
    n = len(X)
    yhat = A @ c
    s2 = np.sum((Y - yhat) ** 2) / max(n - 2, 1)
    se = math.sqrt(s2 / np.sum((X - X.mean()) ** 2))
    return float(c[0]), se, n


T1 = json.load(open(a.t1))["T1"]
T1r = json.load(open(a.t1r))["T1"] if os.path.exists(a.t1r) else {}
rows = []
for h, Ts in SL.items():
    for T in Ts:
        key = "%s_T%07.2f" % (h, T)
        J = json.load(open(os.path.join(a.well, "well_%s.json" % key)))
        Z = np.load(os.path.join(a.well, "well_%s.npz" % key))
        R = Z["R"].astype(float)
        eps, Nd, Na = Z["eps_tot"].astype(float), Z["N_dense_sites"].astype(float), Z["N_applied"].astype(float)
        Md, Ma = Z["M_w_dense"].astype(float), Z["M_w_applied"].astype(float)
        cum, nsite = Z["dust_enclosed"].astype(float), Z["dust_number_site"].astype(float)
        m, ell = float(J["m"]), float(J["sigma_over_a"])
        roots = J.get("roots") or {}
        Ro, Ri = roots.get("dense_outer"), roots.get("dense_inner")
        R50, R90 = (J.get("dust") or {}).get("R_50"), (J.get("dust") or {}).get("R_90")
        Rc = (J.get("record") or {}).get("R_c")
        row = {"key": key, "T": T, "history": h, "ell": ell, "R_inner": Ri, "R_outer": Ro, "R_50": R50, "R_90": R90, "R_c": Rc}
        # (1) window fit, recomputed
        if R50 is not None:
            mask = R <= R50
            sel = mask & (cum >= 1.0)
            sl, se, n = fitll(cum, Nd, sel)
            sla, sea, _ = fitll(cum, Na, sel)
            eps_c = float(eps[mask].sum() / nsite[mask].sum())
            row.update({"fit_slope_dense": sl, "fit_slope_se": se, "fit_n": n, "fit_slope_applied": sla, "eps_c": eps_c, "nu_same_source": (eps_c - m) / (2 * m),
                        "eps_c_over_2m": eps_c / (2 * m)})
            w = ((J.get("core_fit") or {}).get("windows") or {}).get("R50") or {}
            row["fit_slope_dense_json"] = (w.get("fit_dense") or {}).get("slope")
            row["nu_json"] = w.get("exponent_eps_minus_m_over_2m")
        # (2) lambda on both kernels, recomputed
        for kname, kern, T1src in (("shell3d", shell3d, T1), ("radial1d", radial1d, T1r)):
            rho = kern(R, R, eps, ell)
            # normalization on a fine grid from 0
            rf = np.linspace(1e-6, R[-1] + 8 * ell, 40001)
            nf = float(np.trapz(4 * math.pi * rf ** 2 * kern(rf, R, eps, ell), rf)) if kname == "shell3d" and key in ("prod_T0015.00",) else None
            out = {}
            for fl, M, N in (("dense", Md, Nd), ("applied", Ma, Na)):
                with np.errstate(divide="ignore", invalid="ignore"):
                    dM = np.gradient(M, R)
                    lam = 4 * math.pi * R ** 2 * rho / dM
                    half = 0.5 * (lam - 1)
                    lnM, lnN = np.log(M), np.log(N)
                    lnNp = np.gradient(lnN, R)
                    res = lnNp * M - 2 * math.pi * R ** 2 * rho + 0.5 * dM
                    sc = np.maximum(np.maximum(np.abs(lnNp * M), 2 * math.pi * R ** 2 * np.abs(rho)), 0.5 * np.abs(dM))
                o = {}
                named = [("inner root", Ri), ("R_c", Rc), ("R_50", R50), ("R_90", R90), ("outer root", Ro)]
                o["half_at"] = {nm: (ival(x, R, half) if x is not None else None) for nm, x in named}
                if Ro is not None:
                    ins = (R <= Ro) & np.isfinite(res / sc)
                    o["max_rel_residual_inside"] = float(np.max(np.abs((res / sc)[ins])))
                    o["median_rel_residual_inside"] = float(np.median(np.abs((res / sc)[ins])))
                # (3) integrated law vs measured clock over intervals
                ints = {}
                for lab, x1, x2 in (("R_50->outer", R50, Ro), ("R_c->outer", Rc, Ro), ("R_50->R_90", R50, R90), ("R_90->outer", R90, Ro),
                                    ("inner->R_50", Ri, R50), ("1a->R_50", R[0], R50)):
                    if x1 is None or x2 is None or not (x1 < x2):
                        continue
                    dlnN_meas = ival(x2, R, lnN) - ival(x1, R, lnN)
                    dlnN_law = law_integral(R, lnM, half, x1, x2)
                    ints[lab] = {"R1": x1, "R2": x2, "dlnN_measured": dlnN_meas, "dlnN_law": dlnN_law,
                                 "lapse_ratio_law_over_measured": math.exp(dlnN_meas - dlnN_law),  # N(x1)/N(x2) by the law over measured
                                 "rel_err_of_dlnN": (dlnN_law - dlnN_meas) / dlnN_meas if dlnN_meas != 0 else None,
                                 "N1_over_N2_measured": math.exp(-dlnN_meas)}
                o["intervals"] = ints
                # (4) the spread of (lambda-1)/2 over annuli, on 4 a blocks (the stored nodes' spacing): the law's block mean
                # (d ln M_w-weighted) and the measured block secant d ln N/d ln M_w
                spr = {}
                for lab, x1, x2 in (("inner->R_50", Ri, R50), ("R_c->outer", Rc, Ro), ("R_50->outer", R50, Ro), ("inner->outer", Ri, Ro)):
                    if x1 is None or x2 is None or not (x1 < x2):
                        continue
                    edges = np.arange(x1, x2 + 1e-9, 4.0)
                    if edges[-1] < x2 - 1e-9:
                        edges = np.append(edges, x2)
                    lawb, secb = [], []
                    for e1, e2 in zip(edges[:-1], edges[1:]):
                        if e2 - e1 < 1.0:
                            continue
                        dl = ival(e2, R, lnM) - ival(e1, R, lnM)
                        if dl == 0:
                            continue
                        lawb.append(law_integral(R, lnM, half, e1, e2) / dl)
                        secb.append((ival(e2, R, lnN) - ival(e1, R, lnN)) / dl)
                    if lawb:
                        spr[lab] = {"n_blocks": len(lawb), "law_block_half_min": float(np.min(lawb)), "law_block_half_max": float(np.max(lawb)),
                                    "secant_block_min": float(np.min(secb)), "secant_block_max": float(np.max(secb)),
                                    "law_block_half": [float(x) for x in lawb], "secant_block": [float(x) for x in secb]}
                o["spread_4a_blocks"] = spr
                out[fl] = o
            # compare with the T1 json of that kernel at the named radii (dense lambda) and the halo
            cmp_ = {}
            s = T1src.get(key)
            if s:
                for rr in s["radii"]:
                    if rr.get("t1_named") and rr.get("half_dense") is not None:
                        mine = ival(rr["R"], R, 0.5 * (4 * math.pi * R ** 2 * rho / np.gradient(Md, R) - 1))
                        cmp_[rr["name"]] = (rr["half_dense"], mine, mine - rr["half_dense"])
                if s.get("halo", {}).get("available"):
                    cmp_["halo_edge_lapse_law_over_measured"] = s["halo"]["edge_lapse_law_over_measured"]
                cmp_["window_weighted_half"] = (s.get("window") or {}).get("weighted_half_dense")
            out["compare_T1_json"] = cmp_
            if nf is not None:
                out["norm_fine_over_sum"] = nf / float(eps.sum())
            row[kname] = out
        rows.append(row)
        print(key, "done", round(time.time() - t0, 1), flush=True)

json.dump({"code": "t1_asymptotic_verify_v1.py", "rows": rows, "seconds": time.time() - t0}, open(os.path.join(a.out, "t1_verify_v1.json"), "w"), indent=1, default=float)

L = ["# t1_asymptotic_verify_v1 (box CPU; independent recomputation from core_well_v1 per-slice json/npz)", ""]
L.append("## Window fit over R <= R_50 (recomputed vs core_well json) and the same-source exponent")
L.append("| slice | slope dense (recomp) | se | n | json slope | slope applied | eps_c | nu=(eps_c-m)/2m | json nu |")
L.append("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    if "fit_slope_dense" in r:
        L.append("| %s | %.4f | %.4f | %d | %.4f | %.4f | %.5f | %.4f | %.4f |" % (r["key"], r["fit_slope_dense"], r["fit_slope_se"], r["fit_n"], r["fit_slope_dense_json"] or float("nan"),
                                                                               r["fit_slope_applied"], r["eps_c"], r["nu_same_source"], r["nu_json"] or float("nan")))
for kn in ("shell3d", "radial1d"):
    L.append("")
    L.append("## Kernel %s: (lambda-1)/2 at named radii (dense, recomputed; T1 json in brackets) and max/median rel identity residual inside the outer root" % kn)
    L.append("| slice | inner root | R_c | R_50 | R_90 | outer root | max rel res | median rel res |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        o = r[kn]["dense"]; c = r[kn]["compare_T1_json"]
        cells = []
        for nm in ("inner root", "R_c", "R_50", "R_90", "outer root"):
            v = o["half_at"].get(nm)
            j = c.get(nm)
            cells.append("-" if v is None else ("%.4f [%.4f]" % (v, j[0]) if j else "%.4f" % v))
        L.append("| %s | %s | %s | %s |" % (r["key"], " | ".join(cells), "%.3f" % o["max_rel_residual_inside"] if "max_rel_residual_inside" in o else "-",
                                          "%.3f" % o["median_rel_residual_inside"] if "median_rel_residual_inside" in o else "-"))
    L.append("")
    L.append("## Kernel %s: the integrated law against the measured clock (dense; applied in brackets). Entry: N(R1)/N(R2) by the law over measured" % kn)
    L.append("| slice | R_50->outer | R_c->outer | R_50->R_90 | R_90->outer | inner->R_50 | 1a->R_50 | measured N(R_50)/N(outer) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        d, ap_ = r[kn]["dense"]["intervals"], r[kn]["applied"]["intervals"]
        cells = []
        for lab in ("R_50->outer", "R_c->outer", "R_50->R_90", "R_90->outer", "inner->R_50", "1a->R_50"):
            if lab in d:
                cells.append("%.4f [%s]" % (d[lab]["lapse_ratio_law_over_measured"], ("%.4f" % ap_[lab]["lapse_ratio_law_over_measured"]) if lab in ap_ else "-"))
            else:
                cells.append("-")
        mm = d.get("R_50->outer", {}).get("N1_over_N2_measured")
        L.append("| %s | %s | %s |" % (r["key"], " | ".join(cells), "%.4f" % mm if mm else "-"))
L.append("")
L.append("## Kernel shell3d, dense: the spread of (lambda-1)/2 on 4 a blocks (law's block mean min..max; measured block secant min..max)")
L.append("| slice | inner->R_50 | R_c->outer | R_50->outer | inner->outer |")
L.append("|---|---|---|---|---|")
for r in rows:
    s = r["shell3d"]["dense"].get("spread_4a_blocks", {})
    cells = []
    for lab in ("inner->R_50", "R_c->outer", "R_50->outer", "inner->outer"):
        if lab in s:
            q = s[lab]
            cells.append("%.2f..%.2f (sec %.2f..%.2f; %d)" % (q["law_block_half_min"], q["law_block_half_max"], q["secant_block_min"], q["secant_block_max"], q["n_blocks"]))
        else:
            cells.append("-")
    L.append("| %s | %s |" % (r["key"], " | ".join(cells)))
L.append("")
L.append("seconds %.1f" % (time.time() - t0))
open(os.path.join(a.out, "T1_verify_v1.md"), "w").write("\n".join(L) + "\n")
print("wrote", a.out)
