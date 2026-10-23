#!/usr/bin/env python
"""T1 readout v1.2 (lead inbox #266; second_route review_T1_v1): recompute the printed integrated decimals of
NOTES_T1_asymptotic_readout_v1.1 under the v1 code path AND under the endpoint/quadrature conventions the review names,
so every printed decimal can be printed only to the digits the conventions agree on.

Same inputs as t1_asymptotic_verify_v1.py (core_well_v1 per-slice json/npz). Kernels copied verbatim from v1.
Conventions for Q = [N(A)/N(B) by the law] / [N(A)/N(B) measured] on an interval A<B:
  measured endpoint:  "lnlin" = ln N interpolated linearly in r (v1);  "Nlin" = N interpolated linearly in r (review's frozen definition)
  law integral:       "v1"    = trapezoid of (lambda-1)/2 against ln M_w, both interpolated linearly to the endpoints (v1 code path)
                      "FDr"   = trapezoid in r of 2 pi r^2 rho/M_w (endpoint values interpolated linearly) minus 1/2 ln(M_w(B)/M_w(A)),
                                M_w linear at the endpoints (the unsplit identity; the review's FD_r)
Also: the pointwise C1 residual with and without the two array-end sites; the dense-field kink spacing inside 64 a;
the 4 a block spread at prod 16 (complete blocks only, and with the partial remainder).
Numpy only, box CPU, light. Writes t1_verify_v1_2.json + T1_verify_v1_2.md to --out.
"""
import argparse, json, math, os, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--well", default="<internal>/well")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
t0 = time.time()
SL = {"prod": [12, 15, 16, 20, 24, 25, 28, 30, 35, 40], "s8": [12, 13, 14, 16, 20, 25, 30, 40]}


def shell3d(r, Rn, src, ell):
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


def law_v1(R, lnM, half, R1, R2):
    sel = (R > R1) & (R < R2)
    xs = np.concatenate([[R1], R[sel], [R2]])
    return float(np.sum(0.5 * (np.interp(xs, R, half)[1:] + np.interp(xs, R, half)[:-1]) * np.diff(np.interp(xs, R, lnM))))


def law_FDr(R, M, rho, R1, R2):
    sel = (R > R1) & (R < R2)
    xs = np.concatenate([[R1], R[sel], [R2]])
    gs = np.interp(xs, R, 2 * math.pi * R ** 2 * rho / M)
    return float(np.sum(0.5 * (gs[1:] + gs[:-1]) * np.diff(xs))) - 0.5 * math.log(ival(R2, R, M) / ival(R1, R, M))


def kinks(R, f, rmax=64.0, thr=1e-9):
    d2 = f[2:] - 2 * f[1:-1] + f[:-2]
    k = R[1:-1][np.abs(d2) > thr]
    k = k[k <= rmax]
    sp = np.diff(k)
    return {"n": int(len(k)), "spacing_min": float(sp.min()) if len(sp) else None, "spacing_median": float(np.median(sp)) if len(sp) else None,
            "spacing_max": float(sp.max()) if len(sp) else None}


rows = []
for h, Ts in SL.items():
    for T in Ts:
        key = "%s_T%07.2f" % (h, T)
        J = json.load(open(os.path.join(a.well, "well_%s.json" % key)))
        Z = np.load(os.path.join(a.well, "well_%s.npz" % key))
        R = Z["R"].astype(float)
        eps, Nd, Na = Z["eps_tot"].astype(float), Z["N_dense_sites"].astype(float), Z["N_applied"].astype(float)
        Md, Ma = Z["M_w_dense"].astype(float), Z["M_w_applied"].astype(float)
        ell = float(J["sigma_over_a"])
        roots = J.get("roots") or {}
        Ro, Ri = roots.get("dense_outer"), roots.get("dense_inner")
        R50, R90 = (J.get("dust") or {}).get("R_50"), (J.get("dust") or {}).get("R_90")
        Rc = (J.get("record") or {}).get("R_c")
        row = {"key": key, "R_inner": Ri, "R_outer": Ro, "R_50": R50, "R_90": R90, "R_c": Rc,
               "kinks_dense_N": kinks(R, Nd), "kinks_dense_v_proxy_M": kinks(R, Md)}
        for kname, kern in (("shell3d", shell3d), ("radial1d", radial1d)):
            rho = kern(R, R, eps, ell)
            for fl, M, N in (("dense", Md, Nd), ("applied", Ma, Na)):
                with np.errstate(divide="ignore", invalid="ignore"):
                    dM = np.gradient(M, R)
                    half = 0.5 * (4 * math.pi * R ** 2 * rho / dM - 1)
                    lnM, lnN = np.log(M), np.log(N)
                    lnNp = np.gradient(lnN, R)
                    res = lnNp * M - 2 * math.pi * R ** 2 * rho + 0.5 * dM
                    sc = np.maximum(np.maximum(np.abs(lnNp * M), 2 * math.pi * R ** 2 * np.abs(rho)), 0.5 * np.abs(dM))
                o = {}
                if Ro is not None:
                    q = np.abs(res / sc)
                    ins = (R <= Ro) & np.isfinite(q)
                    o["C1_max_all"] = float(q[ins].max()); o["C1_r_at_max_all"] = float(R[ins][np.argmax(q[ins])])
                    ins2 = ins & (np.arange(len(R)) > 0) & (np.arange(len(R)) < len(R) - 1)
                    o["C1_max_no_array_ends"] = float(q[ins2].max()); o["C1_r_at_max_no_ends"] = float(R[ins2][np.argmax(q[ins2])])
                    o["C1_median"] = float(np.median(q[ins]))
                ints = {}
                for lab, x1, x2 in (("R_50->outer", R50, Ro), ("R_c->outer", Rc, Ro), ("R_50->R_90", R50, R90), ("R_90->outer", R90, Ro),
                                    ("inner->R_50", Ri, R50), ("1a->R_50", R[0], R50)):
                    if x1 is None or x2 is None or not (x1 < x2):
                        continue
                    meas = {"lnlin": ival(x2, R, lnN) - ival(x1, R, lnN), "Nlin": math.log(ival(x2, R, N) / ival(x1, R, N))}
                    law = {"v1": law_v1(R, lnM, half, x1, x2), "FDr": law_FDr(R, M, rho, x1, x2)}
                    e = {"R1": x1, "R2": x2, "N1_over_N2_measured_lnlin": math.exp(-meas["lnlin"]), "N1_over_N2_measured_Nlin": math.exp(-meas["Nlin"])}
                    for lk, lv in law.items():
                        for mk, mv in meas.items():
                            e["Q_%s_%s" % (lk, mk)] = math.exp(mv - lv)
                    ints[lab] = e
                o["intervals"] = ints
                if kname == "shell3d" and fl == "dense" and key in ("prod_T0015.00", "prod_T0016.00") and Rc is not None and Ro is not None and Rc < Ro:
                    edges = np.arange(Rc, Ro + 1e-9, 4.0)
                    full = list(zip(edges[:-1], edges[1:]))
                    rem = (edges[-1], Ro) if edges[-1] < Ro - 1e-9 else None
                    bl = {}
                    for nm_, f_ in (("v1", lambda e1, e2: law_v1(R, lnM, half, e1, e2) / (ival(e2, R, lnM) - ival(e1, R, lnM))),
                                    ("FDr", lambda e1, e2: (law_FDr(R, M, rho, e1, e2)) / (ival(e2, R, lnM) - ival(e1, R, lnM)))):
                        pf = [f_(e1, e2) for e1, e2 in full]
                        pa = pf + ([f_(*rem)] if rem and rem[1] - rem[0] >= 1.0 else [])
                        hr = lambda p: 100 * (max(p) - min(p)) / (max(p) + min(p))
                        bl[nm_] = {"complete_blocks": pf, "with_remainder": pa, "halfrange_over_mid_pct_complete": hr(pf), "halfrange_over_mid_pct_with_remainder": hr(pa),
                                   "remainder": rem}
                    o["halo_blocks_4a"] = bl
                row["%s_%s" % (kname, fl)] = o
        rows.append(row)

# all-13 maximum miss per convention
summ = {}
for kf in ("shell3d_dense", "radial1d_dense", "shell3d_applied", "radial1d_applied"):
    for conv in ("Q_v1_lnlin", "Q_v1_Nlin", "Q_FDr_lnlin", "Q_FDr_Nlin"):
        best = (0, None)
        for r in rows:
            e = r[kf]["intervals"].get("R_50->outer")
            if e:
                m = 100 * abs(e[conv] - 1)
                if m > best[0]:
                    best = (m, r["key"])
        summ["%s %s" % (kf, conv)] = best
out = {"code": "t1_asymptotic_verify_v1_2.py", "rows": rows, "max_miss_R50_outer_pct": summ, "seconds": time.time() - t0}
json.dump(out, open(os.path.join(a.out, "t1_verify_v1_2.json"), "w"), indent=1, default=float)

L = ["# t1_asymptotic_verify_v1_2 (box CPU)", "", "## R_50 -> outer root, shell3d dense: Q under four conventions; measured N(R_50)/N(root) both endpoint conventions",
     "| slice | Q v1/lnlin (v1.1 print) | Q v1/Nlin | Q FDr/lnlin | Q FDr/Nlin | meas lnlin | meas Nlin |", "|---|---|---|---|---|---|---|"]
for r in rows:
    e = r["shell3d_dense"]["intervals"].get("R_50->outer")
    if e:
        L.append("| %s | %.6f | %.6f | %.6f | %.6f | %.10f | %.10f |" % (r["key"], e["Q_v1_lnlin"], e["Q_v1_Nlin"], e["Q_FDr_lnlin"], e["Q_FDr_Nlin"],
                                                                     e["N1_over_N2_measured_lnlin"], e["N1_over_N2_measured_Nlin"]))
for lab in ("R_c->outer", "R_50->R_90", "R_90->outer", "inner->R_50", "1a->R_50"):
    L += ["", "## %s, shell3d dense: Q under four conventions" % lab, "| slice | v1/lnlin | v1/Nlin | FDr/lnlin | FDr/Nlin |", "|---|---|---|---|---|"]
    for r in rows:
        e = r["shell3d_dense"]["intervals"].get(lab)
        if e:
            L.append("| %s | %.6f | %.6f | %.6f | %.6f |" % (r["key"], e["Q_v1_lnlin"], e["Q_v1_Nlin"], e["Q_FDr_lnlin"], e["Q_FDr_Nlin"]))
L += ["", "## all-13 maximum 100|Q-1| on R_50 -> outer", "| field/kernel convention | max | slice |", "|---|---|---|"]
for k, (m, s) in summ.items():
    L.append("| %s | %.6f | %s |" % (k, m, s))
L += ["", "## C1 pointwise (shell3d dense): max |res|/scale inside the outer root, with and without the two array-end sites; median",
      "| slice | max all (r) | max no ends (r) | median |", "|---|---|---|---|"]
for r in rows:
    o = r["shell3d_dense"]
    if "C1_max_all" in o:
        L.append("| %s | %.6f (%.0f) | %.6f (%.0f) | %.4f |" % (r["key"], o["C1_max_all"], o["C1_r_at_max_all"], o["C1_max_no_array_ends"], o["C1_r_at_max_no_ends"], o["C1_median"]))
L += ["", "## dense-field kinks inside 64 a (|second difference of N_dense_sites| > 1e-9)", "| slice | n | spacing min/median/max |", "|---|---|---|"]
for r in rows:
    k = r["kinks_dense_N"]
    L.append("| %s | %d | %s/%s/%s |" % (r["key"], k["n"], k["spacing_min"], k["spacing_median"], k["spacing_max"]))
L += ["", "## prod 15/16 halo R_c -> outer root, 4 a blocks of (lambda-1)/2 (shell3d dense)"]
for r in rows:
    b = r["shell3d_dense"].get("halo_blocks_4a")
    if b:
        for nm_, q in b.items():
            L.append("- %s %s: complete %s -> %.4f %%; with remainder %s -> %.4f %%" % (r["key"], nm_, ["%.4f" % x for x in q["complete_blocks"]],
                     q["halfrange_over_mid_pct_complete"], q["remainder"], q["halfrange_over_mid_pct_with_remainder"]))
L += ["", "seconds %.1f" % (time.time() - t0)]
open(os.path.join(a.out, "T1_verify_v1_2.md"), "w").write("\n".join(L) + "\n")
print("wrote", a.out, "%.1f s" % (time.time() - t0))
