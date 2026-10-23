#!/usr/bin/env python
"""Strand 5 J5 report (builder analyst): from strand5_w0radial_v4.py's records of one member (the star series with the flat control and the
dilation family; the Gaussian series with the lattice forms; the per-node densities), and the member's stage-2 record:
(C) controls: the flat operator's max |d_j| and W, the eigenvalue counts, the j-convergence (the last tenth of the j range over the total);
(K) the operator's own k and g_4 from the Gaussians: W_0^(2)(w) = lim W/e^2 (the even part (W(e) + W(-e))/2 over e^2, Richardson in e^2 over the
    three magnitudes, each series), the e^3 term (the odd part over e^3), and -k a(w) - g_4 b(w) = W_0^(2)(w) with a(w) = (3 pi^(3/2)/2) w,
    b(w) = (15 pi^(3/2)/4)/w (f = exp(-r^2/(2 w^2)), C(q) = -(k q^2 + g_4 q^4)) solved at w = 20 and 60; per cone and times 8 against the
    lattice k_op, the continuum per-cone 3/(8 pi^2) sum eta m^2 log m^2 and P14's L/(2 pi^2), L = sum eta log m^2; the lattice forms f C_lat f
    beside 8 W_0^(2)(w);
(W) the star: W_c(e) per L variant, W_0^(2) by the same Richardson, the e^3 term, W_c at e = 1; the bulk values W_bulk = W_c + 4 pi k R (psi(R) -
    s)(s - L(R)) with k the operator's own (K); J5's number W_bulk[star]/W_0^(2)_bulk - 1 (primary and psi-only); and, since W_0 is exactly affine
    in L = N psi, the vacuum-response linearity at the exact L: W_lin = sum_i (L_i - c) x_lin,i with x_lin the linear-response density (the odd
    part of the stored densities over e, Richardson over e = 1e-3, 1e-2), against W_c[star] at the same c; the lattice forms beside 8 times
    the radial ones;
(D) the dilation derivative (spec 10(p'')): dW_c/dlam and dW_bulk/dlam (central differences at 1e-3, 2e-3, Richardson) for the nonlinear W
    and for the quadratic form (e = +-2e-2 even parts; lam = 1 from the star series), 8 times each beside the stage-2 record's frozen-kernel
    D_vac + S_inf (vac_piece) of the member, and the ratio (nonlinear)/(frozen);
(R) the clock-row residual: x(psi)(r) per unit volume (integer nodes, the half-node density averaged onto them), w = psi x(psi) against
    x_lin(r) = C(psi - 1), 8 times each against the lattice row C_lat(psi - 1) on r = 1..J (strand5_lattice_star_v3's operator from the same
    kernel), the residual's integral over the flux int x_lin and its ratio to the member's (A - B)/A (an estimate, no re-solve).
Numbers emitted, not retyped.
(Version 2: the lattice forms are read from the Gaussian record or, if it has none yet, from the star record (both runs compute them); the
k for the wall terms is the e = 1e-2 series' when both widths have it, else the e = 1e-3 series', stated in the summary.)"""
import argparse
import json
import math
import sys
import os
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def f(v, sig=4):
    if v is None:
        return "-"
    v = float(v)
    if v == 0:
        return "0"
    return ("%.*e" % (sig - 1, v)) if (abs(v) < 1e-3 or abs(v) >= 1e5) else ("%.*g" % (sig, v))


def table(head, rows):
    return "\n".join(["| " + " | ".join(str(c) for c in head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def rich3(es, ys):
    """y(e) = y0 + y1 e^2 + y2 e^4 through three points: y0."""
    X = np.array([[1, e * e, e ** 4] for e in es])
    return float(np.linalg.solve(X, np.array(ys))[0])


def rich2(es, ys):
    X = np.array([[1, e * e] for e in es])
    return float(np.linalg.solve(X, np.array(ys))[0])


def series(cfgs, prefix, variant, mags):
    """even part over e^2 and odd part over e^3 per magnitude, and their Richardson limits."""
    ev, od = [], []
    for e in mags:
        p = cfgs.get("%s%+.0e" % (prefix, e))
        m = cfgs.get("%s%+.0e" % (prefix, -e))
        if p is None or m is None:
            return None
        wp, wm = p["L_variants"][variant]["W_c"], m["L_variants"][variant]["W_c"]
        ev.append((wp + wm) / 2 / e ** 2)
        od.append((wp - wm) / 2 / e ** 3)
    return {"even_over_e2": ev, "odd_over_e3": od, "W2": rich3(mags, ev), "W3": rich3(mags, od)}


def corr(cfg, variant, k, L_wall):
    s = cfg.get("s_inf", 1.0)
    R = cfg.get("R_wall")
    return 4 * math.pi * k * R * (cfg["psi_at_wall"] - s) * (s - L_wall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--star", required=True)
    ap.add_argument("--gauss", required=True)
    ap.add_argument("--density", required=True)
    ap.add_argument("--stage2", default=None, help="the member's family stage-2 record (stage2_<tag>.json)")
    ap.add_argument("--kernel", default=None, help="the stage-1 kernel table (for the lattice row)")
    ap.add_argument("--packets", default=None)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    S = json.load(open(a.star))
    G = json.load(open(a.gauss))
    Dn = np.load(a.density)
    M = S["meta"]
    import strand5_w0radial_v4 as W4
    import strand5_lattice_star_v3 as LS
    mem = W4.Member(M["family"], M["profiles"], M["z_c"], M.get("interp", "cubic"))
    masses, etas = M["masses"], M["etas"]
    out = ["# Strand 5 J5 tables (generated by solve/code/strand5/strand5_w0radial_report_v2.py)", ""]
    summ = []
    cf = S["configs"]
    # (C)
    fl = cf.get("flat")
    crow = []
    if fl:
        crow.append(["flat (star L)", f(fl["L_variants"]["star_L"]["W_c"], 2), f(max(fl["max_abs_d_j"]), 2), fl["eigen_count_mismatches"], "-"])
    for k, c in list(cf.items()) + list(G["configs"].items()):
        v = list(c["L_variants"].values())[0]
        crow.append([k, f(v["W_c"], 6), f(max(c["max_abs_d_j"]), 2), c["eigen_count_mismatches"], f(v["last_tenth_over_total"], 2)])
    out += ["## C Controls per configuration: W_c (first L variant), the largest |d_j| over j, the eigenvalue-count mismatches, the last tenth of the "
            "j range over the total (the j-convergence measure)", "", table(["configuration", "W_c", "max |d_j|", "count mismatches", "last tenth/total"], crow), ""]
    if fl:
        summ.append("C: the flat operator's largest |d_j| %s and W %s (exact normality to rounding); eigenvalue-count mismatches %d in all configurations." % (
            f(max(fl["max_abs_d_j"]), 2), f(fl["L_variants"]["star_L"]["W_c"], 2), sum(c["eigen_count_mismatches"] for c in list(cf.values()) + list(G["configs"].values()))))
    # (K)
    gc = G["configs"]
    widths = sorted({c["width"] for c in gc.values()})
    krow, W2w = [], {}
    for w in widths:
        for name, mags in (("spec", (1e-3, 2e-3, 4e-3)), ("big", (1e-2, 2e-2, 4e-2))):
            sr = series(gc, "gauss_w%g_e" % w, "psi_only", mags)
            if sr is None:
                continue
            W2w[(w, name)] = sr["W2"]
            krow.append([w, name, " ".join("%.10e" % x for x in sr["even_over_e2"]), "%.12e" % sr["W2"], "%.6e" % sr["W3"]])
    A = lambda w: 1.5 * math.pi ** 1.5 * w
    B = lambda w: 3.75 * math.pi ** 1.5 / w
    kres = {}
    for name in ("spec", "big"):
        if (widths[0], name) in W2w and (widths[-1], name) in W2w and len(widths) >= 2:
            w1, w2 = widths[0], widths[-1]
            X = np.array([[-A(w1), -B(w1)], [-A(w2), -B(w2)]])
            k, g4 = np.linalg.solve(X, np.array([W2w[(w1, name)], W2w[(w2, name)]]))
            kres[name] = (float(k), float(g4))
    kc = 3 / (8 * math.pi ** 2) * sum(e * m * m * math.log(m * m) for m, e in zip(masses, etas))
    Lsum = sum(e * math.log(m * m) for m, e in zip(masses, etas))
    lat = G.get("lattice") or S.get("lattice") or {}
    out += ["## K The Gaussian profiles (psi = 1 + e f_w, N = 1): the even part over e^2 at the three magnitudes, W_0^(2) by Richardson in e^2, the e^3 "
            "term", "", table(["w", "series", "even/e^2 at the magnitudes", "W_0^(2)", "e^3 term"], krow), ""]
    k2 = []
    for name, (k, g4) in kres.items():
        k2.append([name, "%.10e" % k, "%.10e" % (8 * k), "%.10e" % M["k_op"], "%.6f" % (8 * k / M["k_op"]), "%.10e" % kc, "%.6f" % (k / kc), "%.6e" % g4,
                   "%.6e" % (8 * g4), "%.6e" % (Lsum / (2 * math.pi ** 2))])
    out += ["## K2 The operator's own k and g_4 from the two widths (per cone; times 8 for the lattice), against the lattice k_op, the continuum per-cone "
            "coefficient and P14's L/(2 pi^2)", "", table(["series", "k (per cone)", "8 k", "k_op", "8 k/k_op", "continuum per cone", "k/continuum",
                                                            "g_4 (per cone)", "8 g_4", "L/(2 pi^2)"], k2), ""]
    lrow = [[w, "%.10e" % (8 * W2w[(w, "big")]) if (w, "big") in W2w else "-", "%.10e" % lat["gauss_w%g_fCf" % w] if ("gauss_w%g_fCf" % w) in lat else "-",
             f((8 * W2w[(w, "big")]) / lat["gauss_w%g_fCf" % w] - 1, 3) if ((w, "big") in W2w and ("gauss_w%g_fCf" % w) in lat) else "-"] for w in widths]
    out += ["## K3 8 W_0^(2)(w) (the e = 1e-2 series) against the lattice kernel's f C_lat f on the stage-1 radial lattice", "",
            table(["w", "8 W_0^(2)", "f C_lat f", "ratio - 1"], lrow), ""]
    kk = kres.get("big", kres.get("spec", (M["k_op"] / 8, None)))[0]
    summ.append("K: the wall terms use k from the %s series." % ("e = 1e-2" if "big" in kres else ("e = 1e-3" if "spec" in kres else "lattice k_op/8 (no Gaussian series complete)")))
    if kres:
        summ.append("K: the operator's own coefficient k = %s per cone (8 k = %s against the lattice k_op %s, ratio %s; the continuum per-cone value %s, ratio %s); "
                    "g_4 = %s per cone (8 g_4 = %s; P14's L/(2 pi^2) = %s)." % (
                        "%.8e" % kres.get("big", kres.get("spec"))[0], "%.8e" % (8 * kk), "%.8e" % M["k_op"], "%.6f" % (8 * kk / M["k_op"]), "%.8e" % kc,
                        "%.6f" % (kk / kc), "%.6e" % kres.get("big", kres.get("spec"))[1], "%.6e" % (8 * kres.get("big", kres.get("spec"))[1]),
                        "%.6e" % (Lsum / (2 * math.pi ** 2))))
    # (W)
    st = cf.get("star_e+1e+00")
    wrow = []
    res_star = {}
    if st:
        for variant in ("primary", "psi_only"):
            Wc1 = st["L_variants"][variant]["W_c"]
            Lw1 = st["L_variants"][variant]["c"]
            c1 = corr(st, variant, kk, Lw1)
            for name, mags in (("spec", (1e-3, 2e-3, 4e-3)), ("big", (1e-2, 2e-2, 4e-2))):
                sr = series(cf, "star_e", variant, mags)
                if sr is None:
                    continue
                # the wall term of the e-series at e (exactly quadratic): its e^2 coefficient from the e = 1e-2 configuration
                cp_, cm_ = cf["star_e%+.0e" % mags[0]], cf["star_e%+.0e" % (-mags[0])]
                cq = (corr(cp_, variant, kk, cp_["L_variants"][variant]["c"]) + corr(cm_, variant, kk, cm_["L_variants"][variant]["c"])) / 2 / mags[0] ** 2
                W2b = sr["W2"] + cq
                Wb = Wc1 + c1
                res_star[(variant, name)] = (sr["W2"], W2b, Wc1, Wb)
                wrow.append([variant, name, "%.10e" % sr["W2"], "%.6e" % sr["W3"], "%.10e" % Wc1, "%.6e" % cq, "%.6e" % c1, "%.10e" % W2b, "%.10e" % Wb,
                             "%.6f" % (Wb / W2b - 1), "%.6f" % (Wc1 / sr["W2"] - 1)])
    out += ["## W The star (psi_e = 1 + e (psi - 1), N_e = 1 + e (N - 1); psi-only: N = 1): W_0^(2) by Richardson in e^2 (each series), the e^3 term, W_c at "
            "e = 1, the wall terms (the e^2 coefficient of the series' and at e = 1; k the operator's own), the bulk values and J5's number "
            "W_bulk[star]/W_0^(2)_bulk - 1 (beside the uncorrected ratio)", "",
            table(["L variant", "series", "W_0^(2) (W_c)", "e^3 term", "W_c[star]", "wall term / e^2", "wall term at e = 1", "W_0^(2) bulk", "W_bulk[star]",
                   "J5: bulk ratio - 1", "uncorrected ratio - 1"], wrow), ""]
    # linear response at the exact L, from the densities
    rint, rhalf = Dn["r_int"], Dn["r_half"]
    n = len(rint)
    if "star_density_e+1e+00" in Dn.files:
        dens1 = Dn["star_density_e+1e+00"]
        xl = {}
        for e in (1e-3, 1e-2):
            kp, km = "star_density_e%+.0e" % e, "star_density_e%+.0e" % (-e)
            if kp in Dn.files and km in Dn.files:
                xl[e] = (Dn[kp] - Dn[km]) / (2 * e)
        if len(xl) == 2:
            xlin = (xl[1e-3] * (1e-2) ** 2 - xl[1e-2] * (1e-3) ** 2) / ((1e-2) ** 2 - (1e-3) ** 2)
        else:
            xlin = list(xl.values())[0]
        psi_i, N_i = mem.fields(rint)
        psi_h, N_h = mem.fields(rhalf)
        L = np.concatenate([N_i * psi_i, N_h * psi_h])
        c = float(L[n - 1])
        Wlin = float(np.sum((L - c) * xlin))
        Wnl = float(np.sum((L - c) * dens1))
        Lp = np.concatenate([psi_i, psi_h])
        cp = float(Lp[n - 1])
        Wlin_p = float(np.sum((Lp - cp) * xlin))
        Wnl_p = float(np.sum((Lp - cp) * dens1))
        out += ["## W2 The vacuum response's linearity at the exact L = N psi (W_0 is affine in L): W_lin = sum (L - c) x_lin (the linear-response "
                "density from the e = +-1e-3 and +-1e-2 densities, Richardson) against the nonlinear sum (L - c) d at e = 1 (the same c); likewise with L = psi", "",
                table(["L", "W_lin", "W_nonlinear", "nonlinear/linear - 1", "W_c[star] (record)"],
                      [["N psi", "%.10e" % Wlin, "%.10e" % Wnl, "%.6f" % (Wnl / Wlin - 1), "%.10e" % st["L_variants"]["primary"]["W_c"]],
                       ["psi", "%.10e" % Wlin_p, "%.10e" % Wnl_p, "%.6f" % (Wnl_p / Wlin_p - 1), "%.10e" % st["L_variants"]["psi_only"]["W_c"]]]), ""]
        summ.append("W: at the exact L = N psi the nonlinear vacuum energy over its linear response is %s - 1 (L = psi: %s - 1); J5's specified number "
                    "W_bulk[star]/W_0^(2)_bulk - 1 = %s (primary, e = 1e-2 series) and %s (psi-only)." % (
                        "%.6f" % (Wnl / Wlin), "%.6f" % (Wnl_p / Wlin_p), "%.6f" % (res_star[("primary", "big")][3] / res_star[("primary", "big")][1] - 1)
                        if ("primary", "big") in res_star else "-", "%.6f" % (res_star[("psi_only", "big")][3] / res_star[("psi_only", "big")][1] - 1)
                        if ("psi_only", "big") in res_star else "-"))
        # (R) the clock-row residual
        vol = 4 * math.pi * rint ** 2 * M["h"]
        volh = 4 * math.pi * rhalf ** 2 * M["h"]

        def per_volume(dv):
            di = dv[:n] / vol
            dh = dv[n:] / volh
            dhi = np.empty(n)
            dhi[:-1] = 0.5 * (dh[:-1] + dh[1:])
            dhi[-1] = dh[-1]
            return di + dhi
        x_nl = per_volume(dens1)
        x_l = per_volume(xlin)
        w_nl = psi_i * x_nl
        resid = w_nl - x_l
        J = int(M["J_stage1"])
        Clat = None
        if a.kernel or a.packets:
            stencil, kmom, info = LS.setup_kernel(SimpleNamespace(kernel=a.kernel, packets=a.packets, mlight=M["mlight"], aH=M["aH"], operator="kernel"))
            Rs_ = max(stencil)
            rx = np.arange(1, J + Rs_ + 1, dtype=float)
            ux = np.concatenate([mem.r * (mem.psi - 1), np.full(Rs_, mem.A)])
            Clat = LS.Radial(J + Rs_, stencil).apply(ux, mem.A)[:J]
        flux_lin = float(np.sum(vol * x_l))
        flux_res = float(np.sum(vol * resid))
        AmB = mem.mem.get("measured_AmB_over_A")
        idxs = sorted(set([0, 4, 9, 19, 39, 59, 79, min(n - 1, 99), min(n - 1, 159), n - 1]))
        rrow = [["%.0f" % rint[i], "%.6e" % psi_i[i], "%.6e" % (8 * x_l[i]), "%.6e" % (8 * w_nl[i]), "%.6e" % (8 * resid[i]),
                 ("%.6e" % Clat[i]) if (Clat is not None and i < J and abs(rint[i] - (i + 1)) < 1e-12) else "-"] for i in idxs]
        if Clat is not None and M["h"] == 1.0:
            dl = 8 * x_l[:J] - Clat
            summ.append("R: the operator's linear-response row 8 x_lin against the lattice kernel's row C_lat(psi - 1) on r = 1..J: largest difference %s "
                        "relative to the row's largest value %s." % ("%.4e" % float(np.max(np.abs(dl))), "%.4e" % float(np.max(np.abs(Clat)))))
        out += ["## R The clock row per unit volume (times 8): x_lin = C(psi - 1) (the operator's own linear response), w = psi x(psi) (nonlinear), the residual "
                "w - x_lin, and the lattice kernel's row C_lat(psi - 1) of the stage-1 operator, at sample radii; the residual's integral over the linear "
                "flux int x_lin, and the member's (A - B)/A", "",
                table(["r", "psi", "8 x_lin", "8 w", "8 (w - x_lin)", "C_lat(psi - 1)"], rrow), "",
                "The integrals: int x_lin d^3x = %s, int (w - x_lin) d^3x = %s, their ratio %s; the member's (A - B)/A = %s; the ratio of the residual's "
                "relative integral to it %s (an estimate of the residual's size if fed into the lock, no re-solve)." % (
                    "%.6e" % flux_lin, "%.6e" % flux_res, "%.6e" % (flux_res / flux_lin), f(AmB, 4), f((flux_res / flux_lin) / AmB, 4) if AmB else "-"), ""]
        summ.append("R: the clock-row residual w - x_lin integrates to %s of the linear flux (the member's (A - B)/A = %s); its largest |8 (w - x_lin)| is %s at r = %s." % (
            "%.4e" % (flux_res / flux_lin), f(AmB, 4), "%.4e" % float(np.max(np.abs(8 * resid))), "%.0f" % rint[int(np.argmax(np.abs(resid)))]))
    # lattice forms of the star
    if lat:
        lr = []
        if ("primary", "big") in res_star:
            lr.append(["(p + n) C p (quadratic form)", "%.10e" % (8 * res_star[("primary", "big")][1]), "%.10e" % lat["pCp_plus_nCp"]])
        if ("psi_only", "big") in res_star:
            lr.append(["p C p (psi-only)", "%.10e" % (8 * res_star[("psi_only", "big")][1]), "%.10e" % lat["pCp"]])
        if "star_density_e+1e+00" in Dn.files:
            Wlin_b = Wlin + 4 * math.pi * kk * float(rint[-1]) * (float(psi_i[-1]) - 1) * (1 - c)
            lr.append(["(chi - 1) C (psi - 1) (stage 1's W; the radial W_lin with its wall term)", "%.10e" % (8 * Wlin_b), "%.10e" % lat["chi_minus_1_C_psi_minus_1"]])
        out += ["## L The radial forms (times 8, bulk) against the lattice kernel's on the same profiles", "", table(["form", "8 x radial", "lattice"], lr), ""]
    # (D) dilation
    drow = []
    if "dil_l+1e-03" in cf:
        lam = {}
        for key, c in cf.items():
            if key.startswith("dil_l") and "_e" not in key[5:]:
                lam[c["lam"]] = c
        lam[1.0] = st
        def Wc(c):
            return c["L_variants"]["primary"]["W_c"]
        def Wb(c):
            return Wc(c) + corr(c, "primary", kk, c["L_variants"]["primary"]["c"])
        der = {}
        for nm, fn in (("nonlinear W_c", Wc), ("nonlinear W_bulk", Wb)):
            d1 = (fn(lam[1 + 1e-3]) - fn(lam[1 - 1e-3])) / 2e-3
            d2 = (fn(lam[1 + 2e-3]) - fn(lam[1 - 2e-3])) / 4e-3
            der[nm] = ((4 * d1 - d2) / 3, d1, d2)
        q = {}
        for key, c in cf.items():
            if key.startswith("dil_l") and "_e" in key[5:]:
                q[(round(c["lam"], 12), c["eps"])] = c
        for e in (2e-2,):
            q[(1.0, e)] = cf["star_e%+.0e" % e]
            q[(1.0, -e)] = cf["star_e%+.0e" % (-e)]

        def W2l(lv, bulk):
            p, m_ = q[(round(lv, 12), 2e-2)], q[(round(lv, 12), -2e-2)]
            v = (Wc(p) + Wc(m_)) / 2 / 4e-4
            if bulk:
                v += (corr(p, "primary", kk, p["L_variants"]["primary"]["c"]) + corr(m_, "primary", kk, m_["L_variants"]["primary"]["c"])) / 2 / 4e-4
            return v
        for nm, bulk in (("quadratic form (W_c)", False), ("quadratic form (bulk)", True)):
            d1 = (W2l(1 + 1e-3, bulk) - W2l(1 - 1e-3, bulk)) / 2e-3
            d2 = (W2l(1 + 2e-3, bulk) - W2l(1 - 2e-3, bulk)) / 4e-3
            der[nm] = ((4 * d1 - d2) / 3, d1, d2)
        vac = None
        if a.stage2:
            S2 = json.load(open(a.stage2))
            mm = min(S2["members"], key=lambda x: abs(x["z_c"] - M["z_c"]))
            vac = mm["lock"]["vac_piece"]
        for nm, (dr, d1, d2) in der.items():
            drow.append([nm, "%.10e" % dr, "%.10e" % d1, "%.10e" % d2, "%.10e" % (8 * dr), f(vac, 8), f(8 * dr / vac, 6) if vac else "-"])
        out += ["## D The dilation derivative (psi_lam(r) = lam^-1/2 psi(r/lam), N_lam(r) = N(r/lam)) of the nonlinear W and of the quadratic form, by central "
                "differences at 1e-3 and 2e-3 and Richardson; 8 times each against the frozen lattice kernel's D_vac + S_inf (stage 2, the member)", "",
                table(["quantity", "dW/dlam (Richardson)", "h = 1e-3", "h = 2e-3", "8 dW/dlam", "frozen D_vac + S_inf", "ratio"], drow), ""]
        summ.append("D: the dilation derivative, times 8: nonlinear (bulk) %s, the quadratic form (bulk) %s, against the frozen kernel's D_vac + S_inf %s "
                    "(ratios %s and %s)." % ("%.6e" % (8 * der["nonlinear W_bulk"][0]), "%.6e" % (8 * der["quadratic form (bulk)"][0]), f(vac, 6),
                                             f(8 * der["nonlinear W_bulk"][0] / vac, 4) if vac else "-", f(8 * der["quadratic form (bulk)"][0] / vac, 4) if vac else "-"))
    out = out[:2] + ["## S Summary (emitted)", ""] + ["- " + x for x in summ] + [""] + out[2:]
    open(a.md, "w").write("\n".join(out) + "\n")
    print("\n".join("- " + x for x in summ))
    print("wrote", a.md)


if __name__ == "__main__":
    main()
