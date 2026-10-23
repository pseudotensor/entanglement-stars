#!/usr/bin/env python
"""Strand 5 J5 report, version 3 (builder analyst): from strand5_w0radial_v4.py's dense records of the pilot member (the star series with the
flat control and the dilation family; the Gaussians with the lattice forms; the star's per-node densities), version 6's exact-L linear
response (--lin), the convergence records of versions 5 and 6 (--conv: the fast route at other h, R_max and nt), and the member's stage-2
record.
(C) controls per configuration: W_c, the largest |d_j| over j, the eigenvalue-count mismatches, the last tenth of the j range over the total;
(K) the operator's own k and g_4 from the Gaussians (-k a(w) - g_4 b(w) = W_0^(2)(w), a = (3 pi^(3/2)/2) w, b = (15 pi^(3/2)/4)/w; f = exp(-r^2/
    (2 w^2)); W_0^(2) by Richardson in e^2 over three magnitudes of the even part (W(e) + W(-e))/2), per cone and times 8 against k_op, the
    continuum per-cone 3/(8 pi^2) sum eta m^2 log m^2 and P14's L/(2 pi^2); 8 W_0^(2)(w) against the lattice f C_lat f;
(W) the specification's number: the quadratic form in both deviations, W_0^(2) = lim W(e)/e^2 with psi_e = 1 + e (psi - 1), N_e = 1 + e (N - 1),
    and W_bulk[star]/W_0^(2)_bulk - 1, with the wall terms W_bulk = W_c + 4 pi k R (psi(R) - s_psi)(s_L - L(R)) (k the operator's own);
(V) the vacuum response's linearity at the exact L = N psi (W_0 is affine in L): W_lin = sum (L - c) x_lin with x_lin the linear-response density
    (the odd part of the stored densities over e, Richardson over e = 1e-3 and 1e-2) and, independently, from version 6's configurations
    psi = 1 + e (psi - 1) with the exact L (the odd part over e, Richardson over e = 1e-2 and 2e-2), against the nonlinear W at e = 1, raw and
    with the wall terms (bilinear in the same tails, so they cancel in the difference at leading order);
(L) the radial forms times 8 against the lattice kernel's on the same profiles;
(D) the dilation derivative (psi_lam = lam^-1/2 psi(r/lam), N_lam = N(r/lam)): the nonlinear W and the exact-L linear functional W_lin(lam), raw
    and bulk, by central differences at 1e-3 and 2e-3 and Richardson, 8 times each against the frozen lattice kernel's D_vac + S_inf of the
    member; beside them version 4's quadratic-form dilation values as recorded (N - 1 and psi - 1 scaled separately, not the dilation of a
    covariant functional);
(R) the clock row per unit volume from shell contents (each integer node with half of each neighbouring half node, over 4 pi r^2 h): x_lin, the
    nonlinear x, psi x and psi x_lin, the lattice row C_lat(psi - 1), the ratio 8 x_lin/C_lat, and the bulk integrals over r <= r_J (the stage-1
    grid) of psi (x - x_lin) against that of psi x_lin, with the member's (A - B)/A;
(H) convergence: per setting (the dense route at h = 1, R_max = 4 r_s; the fast route at the same and at nt 300, at h = 0.5, at R_max = 8 r_s)
    W_c[star], the exact-L linear W_lin (version 6's lam = 1 configurations, the e = 1e-2 odd part), the linearity fraction, and the nonlinear
    and linear dilation derivatives (raw and bulk);
(P) the production cost from the measured seconds per configuration: the fast route's time scales as n^2 (channels times nodes), the dense
    route's as about n^4, n = R_max/h; the members of spec 10(p) at R_max = 4 r_s with h scaled so that m_heavy h equals the pilot's.
Numbers emitted, not retyped.
(Version 4, for version 7's production records as well: the linear-response density x_lin from the star's e-series densities where present, else from
the exact-L configurations' densities at lam = 1 (the odd part over e, Richardson over e = 1e-2 and 2e-2); the --lin record may be the star record
itself; the convergence table H reads each record's own exact-L configurations.)
(Version 5: section Q, the radial build-up of the nonlinear-minus-linear W in every record with densities: over the nodes with r <= rho, the
sums of (L - 1)(d - d_lin), of (d - d_lin) (the flux difference), of (L - 1) d_lin and (L - 1) d, and their ratio - 1, at rho = 40, 80 (about
r_s), r_J, 240, 300 and the last node before the wall layer (R_max - 20 h); d_lin from the e = +-1e-2 densities (the odd part over e). With L - 1
in place of L - c the sums do not carry the term (1 - c) times the flux difference, and inside the star they are independent of R_max.)"""
import argparse
import json
import math
import os
import sys
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
    X = np.array([[1, e * e, e ** 4] for e in es])
    return float(np.linalg.solve(X, np.array(ys))[0])


def series(cfgs, prefix, variant, mags):
    ev, od = [], []
    for e in mags:
        p, m = cfgs.get("%s%+.0e" % (prefix, e)), cfgs.get("%s%+.0e" % (prefix, -e))
        if p is None or m is None:
            return None
        wp, wm = p["L_variants"][variant]["W_c"], m["L_variants"][variant]["W_c"]
        ev.append((wp + wm) / 2 / e ** 2)
        od.append((wp - wm) / 2 / e ** 3)
    return {"even_over_e2": ev, "odd_over_e3": od, "W2": rich3(mags, ev), "W3": rich3(mags, od)}


def corr(cfg, variant, k):
    s_psi = cfg.get("s_inf", 1.0)
    s_L = cfg.get("s_L", s_psi)
    return 4 * math.pi * k * cfg["R_wall"] * (cfg["psi_at_wall"] - s_psi) * (s_L - cfg["L_variants"][variant]["c"])


def lin_value(cfgs, lam, variant="primary", bulk=False, k=None, mags=(1e-2, 2e-2)):
    """the exact-L linear functional at lam from the odd parts over e (Richardson over the magnitudes present)."""
    vals = []
    for e in mags:
        p, m = cfgs.get("lin_l%+.0e_e%+.0e" % (lam - 1, e)), cfgs.get("lin_l%+.0e_e%+.0e" % (lam - 1, -e))
        if p is None or m is None:
            continue
        wp, wm = p["L_variants"][variant]["W_c"], m["L_variants"][variant]["W_c"]
        if bulk:
            wp += corr(p, variant, k)
            wm += corr(m, variant, k)
        vals.append((e, (wp - wm) / (2 * e)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0][1]
    (e1, o1), (e2, o2) = vals[:2]
    return (o1 * e2 ** 2 - o2 * e1 ** 2) / (e2 ** 2 - e1 ** 2)


def dil_deriv(fn):
    try:
        d1 = (fn(1 + 1e-3) - fn(1 - 1e-3)) / 2e-3
        d2 = (fn(1 + 2e-3) - fn(1 - 2e-3)) / 4e-3
    except (KeyError, TypeError):
        return None
    return ((4 * d1 - d2) / 3, d1, d2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--star", required=True)
    ap.add_argument("--gauss", required=True)
    ap.add_argument("--density", required=True)
    ap.add_argument("--lin", default=None)
    ap.add_argument("--conv", default="", help="comma-separated records of other settings (label=path)")
    ap.add_argument("--stage2", default=None)
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--packets", default=None)
    ap.add_argument("--production", default="", help="comma-separated family:z_c of the production members (their stage-1 records give r_s, m_l, aH)")
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    S = json.load(open(a.star))
    G = json.load(open(a.gauss))
    Dn = np.load(a.density)
    LN = json.load(open(a.lin)) if a.lin else None
    M = S["meta"]
    import strand5_w0radial_v4 as W4
    import strand5_lattice_star_v3 as LS
    mem = W4.Member(M["family"], M["profiles"], M["z_c"], M.get("interp", "cubic"))
    masses, etas = M["masses"], M["etas"]
    cf = S["configs"]
    gc = {k_: v for k_, v in G["configs"].items() if k_.startswith("gauss")}
    lat = G.get("lattice") or S.get("lattice") or {}
    out = ["# Strand 5 J5 tables (generated by solve/code/strand5/strand5_w0radial_report_v5.py)", ""]
    summ = []
    # (C)
    crow = []
    allc = list(cf.items()) + list(gc.items()) + (list(LN["configs"].items()) if LN else [])
    for k_, c in allc:
        v = list(c["L_variants"].values())[0]
        crow.append([k_, f(v["W_c"], 6), f(max(c["max_abs_d_j"]), 2), c["eigen_count_mismatches"], f(v["last_tenth_over_total"], 2)])
    out += ["## C Controls per configuration (the dense route, h = 1, R_max = 4 r_s): W_c (first L variant), the largest |d_j| over j, the eigenvalue-count "
            "mismatches, the last tenth of the j range over the total", "", table(["configuration", "W_c", "max |d_j|", "count mismatches", "last tenth/total"], crow), ""]
    fl = cf.get("flat")
    if fl:
        summ.append("C: the flat operator's largest |d_j| over all j %s and its W %s with the star's L (exact normality to rounding); eigenvalue-count mismatches %d in %d "
                    "configurations; the star's last tenth of the j range carries %s of W_c (primary)." % (
                        f(max(fl["max_abs_d_j"]), 2), f(fl["L_variants"]["star_L"]["W_c"], 2), sum(c["eigen_count_mismatches"] for _, c in allc), len(allc),
                        f(cf["star_e+1e+00"]["L_variants"]["primary"]["last_tenth_over_total"], 2)))
    # (K)
    widths = sorted({c["width"] for c in gc.values()})
    krow, W2w = [], {}
    for w in widths:
        for name, mags in (("spec", (1e-3, 2e-3, 4e-3)), ("big", (1e-2, 2e-2, 4e-2))):
            sr = series(gc, "gauss_w%g_e" % w, "psi_only", mags)
            if sr is None:
                continue
            W2w[(w, name)] = sr["W2"]
            krow.append([w, name, " ".join("%.10e" % x for x in sr["even_over_e2"]), "%.12e" % sr["W2"], "%.6e" % sr["W3"]])
    A_ = lambda w: 1.5 * math.pi ** 1.5 * w
    B_ = lambda w: 3.75 * math.pi ** 1.5 / w
    kres = {}
    for name in ("spec", "big"):
        if len(widths) >= 2 and (widths[0], name) in W2w and (widths[-1], name) in W2w:
            w1, w2 = widths[0], widths[-1]
            k, g4 = np.linalg.solve(np.array([[-A_(w1), -B_(w1)], [-A_(w2), -B_(w2)]]), np.array([W2w[(w1, name)], W2w[(w2, name)]]))
            kres[name] = (float(k), float(g4))
    kc = 3 / (8 * math.pi ** 2) * sum(e * m * m * math.log(m * m) for m, e in zip(masses, etas))
    Lsum = sum(e * math.log(m * m) for m, e in zip(masses, etas))
    out += ["## K The Gaussian profiles (psi = 1 + e f_w, N = 1): the even part over e^2 at the three magnitudes, W_0^(2) by Richardson in e^2, the e^3 term", "",
            table(["w", "series", "even/e^2 at the magnitudes", "W_0^(2)", "e^3 term"], krow), ""]
    out += ["## K2 The operator's own k and g_4 (per cone; times 8 for the lattice) against the lattice k_op, the continuum per-cone coefficient and P14's L/(2 pi^2)", "",
            table(["series", "k (per cone)", "8 k", "k_op", "8 k/k_op", "continuum per cone", "k/continuum", "g_4 (per cone)", "8 g_4", "L/(2 pi^2)"],
                  [[n_, "%.10e" % k, "%.10e" % (8 * k), "%.10e" % M["k_op"], "%.6f" % (8 * k / M["k_op"]), "%.10e" % kc, "%.6f" % (k / kc), "%.6e" % g4, "%.6e" % (8 * g4),
                    "%.6e" % (Lsum / (2 * math.pi ** 2))] for n_, (k, g4) in kres.items()]), ""]
    lrow = [[w, "%.10e" % (8 * W2w[(w, s_)]), "%.10e" % lat["gauss_w%g_fCf" % w], f((8 * W2w[(w, s_)]) / lat["gauss_w%g_fCf" % w] - 1, 3), s_]
            for w in widths for s_ in ("big", "spec") if (w, s_) in W2w and ("gauss_w%g_fCf" % w) in lat]
    out += ["## K3 8 W_0^(2)(w) against the lattice kernel's f C_lat f on the stage-1 radial lattice", "", table(["w", "8 W_0^(2)", "f C_lat f", "ratio - 1", "series"], lrow), ""]
    kname = "big" if "big" in kres else ("spec" if "spec" in kres else None)
    kk = kres[kname][0] if kname else M["k_op"] / 8
    if kname:
        summ.append("K: k = %s per cone from the %s series (8 k/k_op = %s; k over the continuum per-cone value %s), g_4 = %s per cone (8 g_4 = %s; P14's L/(2 pi^2) = %s); "
                    "the wall terms below use this k." % ("%.8e" % kk, "e = 1e-2" if kname == "big" else "e = 1e-3", "%.6f" % (8 * kk / M["k_op"]), "%.6f" % (kk / kc),
                                                          "%.6e" % kres[kname][1], "%.6e" % (8 * kres[kname][1]), "%.6e" % (Lsum / (2 * math.pi ** 2))))
    # (W)
    st = cf["star_e+1e+00"]
    wrow, res_star = [], {}
    for variant in ("primary", "psi_only"):
        Wc1 = st["L_variants"][variant]["W_c"]
        c1 = corr(st, variant, kk)
        for name, mags in (("spec", (1e-3, 2e-3, 4e-3)), ("big", (1e-2, 2e-2, 4e-2))):
            sr = series(cf, "star_e", variant, mags)
            if sr is None:
                continue
            cp_, cm_ = cf["star_e%+.0e" % mags[0]], cf["star_e%+.0e" % (-mags[0])]
            cq = (corr(cp_, variant, kk) + corr(cm_, variant, kk)) / 2 / mags[0] ** 2
            W2b, Wb = sr["W2"] + cq, Wc1 + c1
            res_star[(variant, name)] = (sr["W2"], W2b, Wc1, Wb)
            wrow.append([variant, name, "%.10e" % sr["W2"], "%.6e" % sr["W3"], "%.10e" % Wc1, "%.6e" % cq, "%.6e" % c1, "%.10e" % W2b, "%.10e" % Wb,
                         "%.6f" % (Wb / W2b - 1), "%.6f" % (Wc1 / sr["W2"] - 1)])
    out += ["## W The specification's number: W_0^(2) (both deviations scaled by e) by Richardson in e^2, the e^3 term, W_c at e = 1, the wall terms, the bulk "
            "values and W_bulk[star]/W_0^(2)_bulk - 1 (beside the uncorrected ratio)", "",
            table(["L variant", "series", "W_0^(2) (W_c)", "e^3 term", "W_c[star]", "wall term / e^2", "wall term at e = 1", "W_0^(2) bulk", "W_bulk[star]",
                   "bulk ratio - 1", "uncorrected ratio - 1"], wrow), ""]
    if ("primary", "big") in res_star:
        summ.append("W: the specification's number W_bulk[star]/W_0^(2)_bulk - 1 = %s (N psi; the quadratic form scales N - 1 and psi - 1 separately, so it omits "
                    "the product (N - 1)(psi - 1) of the exact L) and %s (psi only)." % ("%.6f" % (res_star[("primary", "big")][3] / res_star[("primary", "big")][1] - 1),
                                                                                         "%.6f" % (res_star[("psi_only", "big")][3] / res_star[("psi_only", "big")][1] - 1)))
    # (V) linearity at the exact L
    rint, rhalf = Dn["r_int"], Dn["r_half"]
    n = len(rint)
    dens1 = Dn["star_density_e+1e+00"]
    xl = {e: (Dn["star_density_e%+.0e" % e] - Dn["star_density_e%+.0e" % (-e)]) / (2 * e) for e in (1e-3, 1e-2)
          if ("star_density_e%+.0e" % e) in Dn.files and ("star_density_e%+.0e" % (-e)) in Dn.files}
    xsrc = "the star's e-series densities"
    if not xl:
        xl = {e: (Dn["lin_density_e%+.0e" % e] - Dn["lin_density_e%+.0e" % (-e)]) / (2 * e) for e in (1e-2, 2e-2)
              if ("lin_density_e%+.0e" % e) in Dn.files and ("lin_density_e%+.0e" % (-e)) in Dn.files}
        xsrc = "the exact-L configurations' densities"
    es = sorted(xl)
    xlin = (xl[es[0]] * es[1] ** 2 - xl[es[1]] * es[0] ** 2) / (es[1] ** 2 - es[0] ** 2) if len(es) == 2 else xl[es[0]]
    psi_i, N_i = mem.fields(rint)
    psi_h, N_h = mem.fields(rhalf)
    L = np.concatenate([N_i * psi_i, N_h * psi_h])
    c = float(L[n - 1])
    Wlin_d = float(np.sum((L - c) * xlin))
    Wnl = float(np.sum((L - c) * dens1))
    vrow = [["densities (%s, e = %s)" % (xsrc, ", ".join("%g" % e for e in es)), "%.10e" % Wlin_d, "%.10e" % Wnl, "%.6e" % (Wnl / Wlin_d - 1), "-"]]
    lin1 = None
    if LN:
        lc = LN["configs"]
        lin1 = lin_value(lc, 1.0)
        lin1b = lin_value(lc, 1.0, bulk=True, k=kk)
        Wnlb = st["L_variants"]["primary"]["W_c"] + corr(st, "primary", kk)
        vrow.append(["version 6 configurations (e = 1e-2, 2e-2)", "%.10e" % lin1, "%.10e" % st["L_variants"]["primary"]["W_c"],
                     "%.6e" % (st["L_variants"]["primary"]["W_c"] / lin1 - 1), "%.6e (bulk %.10e against %.10e)" % (Wnlb / lin1b - 1, lin1b, Wnlb)])
        summ.append("V: at the exact L = N psi the nonlinear vacuum energy over its linear response is 1 + %s (raw) and 1 + %s (bulk, wall terms added) from "
                    "version 6's configurations; from the stored densities 1 + %s." % ("%.4e" % (st["L_variants"]["primary"]["W_c"] / lin1 - 1),
                                                                                     "%.4e" % (Wnlb / lin1b - 1), "%.4e" % (Wnl / Wlin_d - 1)))
    out += ["## V The vacuum response's linearity at the exact L = N psi (W_0 is affine in L): the linear functional W_lin against the nonlinear W at e = 1 (the same c)", "",
            table(["W_lin from", "W_lin", "W nonlinear", "nonlinear/linear - 1 (raw)", "bulk"], vrow), ""]
    # (L)
    lr = []
    if ("primary", "big") in res_star:
        lr.append(["(p + n) C p (the quadratic form, bulk)", "%.10e" % (8 * res_star[("primary", "big")][1]), "%.10e" % lat.get("pCp_plus_nCp", float("nan"))])
        lr.append(["p C p (psi only, bulk)", "%.10e" % (8 * res_star[("psi_only", "big")][1]), "%.10e" % lat.get("pCp", float("nan"))])
    if LN:
        lr.append(["(chi - 1) C (psi - 1) (the exact-L linear functional, bulk; stage 1's W)", "%.10e" % (8 * lin1b), "%.10e" % lat.get("chi_minus_1_C_psi_minus_1", float("nan"))])
    out += ["## L The radial forms (times 8) against the lattice kernel's on the same profiles", "", table(["form", "8 x radial", "lattice"], lr), ""]
    # (D)
    vac = None
    if a.stage2:
        S2 = json.load(open(a.stage2))
        mm = min(S2["members"], key=lambda x: abs(x["z_c"] - M["z_c"]))
        vac = mm["lock"]["vac_piece"]
    lam_nl = {cfg["lam"]: cfg for k_, cfg in cf.items() if k_.startswith("dil_l") and "_e" not in k_[5:]}
    lam_nl[1.0] = st
    drow = []
    ders = {}
    for nm, fn in (("nonlinear W (W_c)", lambda l: lam_nl[l]["L_variants"]["primary"]["W_c"]),
                   ("nonlinear W (bulk)", lambda l: lam_nl[l]["L_variants"]["primary"]["W_c"] + corr(lam_nl[l], "primary", kk))):
        ders[nm] = dil_deriv(fn)
    if LN:
        lc = LN["configs"]
        ders["exact-L linear W_lin (W_c)"] = dil_deriv(lambda l: lin_value(lc, l))
        ders["exact-L linear W_lin (bulk)"] = dil_deriv(lambda l: lin_value(lc, l, bulk=True, k=kk))
    q = {(round(cfg["lam"], 12), cfg["eps"]): cfg for k_, cfg in cf.items() if k_.startswith("dil_l") and "_e" in k_[5:]}
    if cf.get("star_e+2e-02") is not None and q:
        q[(1.0, 2e-2)], q[(1.0, -2e-2)] = cf["star_e+2e-02"], cf["star_e-2e-02"]

        def W2l(lv):
            p, m_ = q[(round(lv, 12), 2e-2)], q[(round(lv, 12), -2e-2)]
            return (p["L_variants"]["primary"]["W_c"] + m_["L_variants"]["primary"]["W_c"]) / 2 / 4e-4
        ders["quadratic form, N - 1 and psi - 1 scaled separately (W_c; as recorded)"] = dil_deriv(W2l)
    for nm, dv in ders.items():
        if dv is None:
            continue
        drow.append([nm, "%.10e" % dv[0], "%.10e" % dv[1], "%.10e" % dv[2], "%.10e" % (8 * dv[0]), f(vac, 8), f(8 * dv[0] / vac, 6) if vac else "-"])
    out += ["## D The dilation derivative, by central differences at 1e-3 and 2e-3 and Richardson; 8 times each against the frozen lattice kernel's D_vac + S_inf "
            "(stage 2, the member)", "", table(["quantity", "dW/dlam (Richardson)", "h = 1e-3", "h = 2e-3", "8 dW/dlam", "frozen D_vac + S_inf", "ratio"], drow), ""]
    if ders.get("nonlinear W (bulk)") and ders.get("exact-L linear W_lin (bulk)"):
        summ.append("D: the dilation derivative times 8: the nonlinear W %s (bulk; %s raw), the exact-L linear functional %s (bulk; %s raw), against the frozen "
                    "lattice kernel's D_vac + S_inf %s (ratios %s and %s, bulk)." % (
                        "%.4e" % (8 * ders["nonlinear W (bulk)"][0]), "%.4e" % (8 * ders["nonlinear W (W_c)"][0]), "%.4e" % (8 * ders["exact-L linear W_lin (bulk)"][0]),
                        "%.4e" % (8 * ders["exact-L linear W_lin (W_c)"][0]), f(vac, 6), f(8 * ders["nonlinear W (bulk)"][0] / vac, 4) if vac else "-",
                        f(8 * ders["exact-L linear W_lin (bulk)"][0] / vac, 4) if vac else "-"))
    # (R) the clock row from shell contents
    h = M["h"]

    def shell(dv):
        di, dh = dv[:n], dv[n:]
        cont = di.copy()
        cont[0] += dh[0] + 0.5 * dh[1]
        cont[1:] += 0.5 * dh[1:] + 0.5 * np.concatenate([dh[2:], [0.0]])
        vol = 4 * math.pi * rint ** 2 * h
        vol[0] = 4 * math.pi / 3 * (1.5 * h) ** 3
        return cont / vol, vol
    x_nl, vol = shell(dens1)
    x_l, _ = shell(xlin)
    J = int(M["J_stage1"])
    Clat = None
    if a.kernel or a.packets:
        stencil, kmom, info = LS.setup_kernel(SimpleNamespace(kernel=a.kernel, packets=a.packets, mlight=M["mlight"], aH=M["aH"], operator="kernel"))
        Rs_ = max(stencil)
        ux = np.concatenate([mem.r * (mem.psi - 1), np.full(Rs_, mem.A)])
        Clat = LS.Radial(J + Rs_, stencil).apply(ux, mem.A)[:J]
    idxs = sorted(set([0, 1, 4, 9, 19, 39, 59, 79, min(n - 1, 99), min(n - 1, J - 1)]))
    rrow = [["%.0f" % rint[i], "%.6e" % psi_i[i], "%.6e" % (8 * x_l[i]), "%.6e" % (8 * x_nl[i]), "%.6e" % (8 * psi_i[i] * (x_nl[i] - x_l[i])),
             ("%.6e" % Clat[i]) if (Clat is not None and i < J and h == 1.0) else "-",
             ("%.5f" % (8 * x_l[i] / Clat[i])) if (Clat is not None and i < J and h == 1.0 and Clat[i] != 0) else "-"] for i in idxs]
    inb = rint <= mem.rJ
    I_lin = float(np.sum((vol * psi_i * x_l)[inb]))
    I_res = float(np.sum((vol * psi_i * (x_nl - x_l))[inb]))
    AmB = mem.mem.get("measured_AmB_over_A")
    out += ["## R The clock row per unit volume from shell contents (times 8): x_lin, the nonlinear x, the residual psi (x - x_lin), the lattice kernel's row "
            "C_lat(psi - 1) and the ratio 8 x_lin/C_lat, at sample radii; the bulk integrals over r <= r_J", "",
            table(["r", "psi", "8 x_lin", "8 x", "8 psi (x - x_lin)", "C_lat(psi - 1)", "8 x_lin/C_lat"], rrow), "",
            "Over r <= r_J (%s): int psi x_lin d^3x = %s, int psi (x - x_lin) d^3x = %s, their ratio %s; the member's (A - B)/A = %s." % (
                "%.0f" % mem.rJ, "%.6e" % I_lin, "%.6e" % I_res, "%.6e" % (I_res / I_lin), f(AmB, 4)), ""]
    summ.append("R: the clock row's nonlinear residual psi (x - x_lin) integrates over r <= r_J to %s of the linear row's integral (the member's (A - B)/A = %s); "
                "8 x_lin/C_lat at r = 5 to 60: %s." % ("%.4e" % (I_res / I_lin), f(AmB, 4),
                                                       ", ".join("%.4f" % (8 * x_l[i] / Clat[i]) for i in (4, 9, 19, 39, 59) if Clat is not None and i < J and h == 1.0)))
    # (H) convergence
    hrow = []
    S["meta"]["_path"] = a.star
    sets = [("this record (%s, h = %g, R_max = %.0f)" % (M.get("route", "dense"), M["h"], M["R_max"]), S, LN)]
    for item in [x for x in a.conv.split(",") if x]:
        lab, path = item.split("=", 1)
        if os.path.exists(path):
            Rx = json.load(open(path))
            Rx["meta"]["_path"] = path
            sets.append((lab, Rx, None))
    for lab, R_, LNx in sets:
        cc = dict(R_["configs"])
        if LNx:
            cc.update(LNx["configs"])
        s1 = cc.get("star_e+1e+00")
        if s1 is None:
            continue
        Wc1 = s1["L_variants"]["primary"]["W_c"]
        l1 = lin_value(cc, 1.0, mags=(1e-2,))
        dn = {cfg["lam"]: cfg for k_, cfg in cc.items() if k_.startswith("dil_l") and "_e" not in k_[5:]}
        dn[1.0] = s1
        dnl = dil_deriv(lambda l: dn[l]["L_variants"]["primary"]["W_c"] + corr(dn[l], "primary", kk))
        dln = dil_deriv(lambda l: lin_value(cc, l, bulk=True, k=kk, mags=(1e-2,)))
        hrow.append([lab, R_["meta"]["h"], "%.0f" % R_["meta"]["R_max"], R_["meta"].get("route", "dense"), R_["meta"].get("nt", "-"), "%.10e" % Wc1,
                     "%.10e" % l1 if l1 else "-", "%.4e" % (Wc1 / l1 - 1) if l1 else "-", "%.4e" % (8 * dnl[0]) if dnl else "-", "%.4e" % (8 * dln[0]) if dln else "-",
                     f(max(s1["max_abs_d_j"]), 2)])
    out += ["## H Convergence: per setting W_c[star], the exact-L linear W_lin (the e = 1e-2 odd part), the linearity fraction, and 8 times the dilation "
            "derivatives (bulk) of the nonlinear W and of W_lin", "", table(["setting", "h", "R_max", "route", "nt", "W_c[star]", "W_lin", "W/W_lin - 1",
                                                                             "8 dW/dlam (nonlinear, bulk)", "8 dW_lin/dlam (bulk)", "max |d_j| (star)"], hrow), ""]
    # (Q) the radial build-up of the nonlinear-minus-linear W
    qrow = []
    for lab, R_, _ in sets:
        path = R_["meta"].get("_path")
        if path is None:
            continue
        nzp = path.replace(".json", "_density.npz")
        if not os.path.exists(nzp):
            continue
        Dq = np.load(nzp)
        if not all(k_ in Dq.files for k_ in ("star_density_e+1e+00", "star_density_e+1e-02", "star_density_e-1e-02")):
            continue
        Mq = R_["meta"]
        memq = W4.Member(Mq["family"], Mq["profiles"], Mq["z_c"], Mq.get("interp", "cubic"))
        ri_, rh_ = Dq["r_int"], Dq["r_half"]
        nq = len(ri_)
        pq, Nq = memq.fields(ri_)
        ph2, Nh2 = memq.fields(rh_)
        Lq = np.concatenate([Nq * pq, Nh2 * ph2]) - 1
        d1q = Dq["star_density_e+1e+00"]
        dlq = (Dq["star_density_e+1e-02"] - Dq["star_density_e-1e-02"]) / 2e-2
        rq = np.concatenate([ri_, rh_])
        o = np.argsort(rq)
        rr_ = rq[o]
        c1 = np.cumsum((Lq * (d1q - dlq))[o])
        c2 = np.cumsum((d1q - dlq)[o])
        c3 = np.cumsum((Lq * dlq)[o])
        c4 = np.cumsum((Lq * d1q)[o])
        for rho in (40.0, 80.0, float(memq.rJ), 240.0, 300.0, float(Mq["R_max"]) - 20 * Mq["h"]):
            if rho > rr_[-1] or rho <= 0:
                continue
            k_ = np.searchsorted(rr_, rho, side="right") - 1
            qrow.append([lab, "%.0f" % rho, "%.5e" % c1[k_], "%.5e" % c2[k_], "%.6e" % c3[k_], "%.6e" % c4[k_], "%.4e" % (c4[k_] / c3[k_] - 1)])
    out += ["## Q The radial build-up: over the nodes with r <= rho, the sums of (L - 1)(d - d_lin), of (d - d_lin) (the flux difference), of (L - 1) d_lin "
            "and of (L - 1) d, and the nonlinear/linear ratio - 1 (d_lin from the e = +-1e-2 densities)", "",
            table(["record", "rho", "sum (L-1)(d - d_lin)", "sum (d - d_lin)", "sum (L-1) d_lin", "sum (L-1) d", "ratio - 1"], qrow), ""]
    if qrow:
        star_rows = [r_ for r_ in qrow if r_[1] == "80"]
        summ.append("Q: inside the star (r <= 80) the nonlinear W over its linear response is 1 + %s; beyond it the flux difference changes sign "
                    "and the build-up depends on R_max and h (table Q)." % ("; 1 + ".join("%s (%s)" % (r_[6], r_[0]) for r_ in star_rows)))
    # (P) cost
    prow = []
    tsec = {}
    for lab, R_, _ in sets:
        secs = R_["meta"].get("seconds")
        ncfg = len(R_["configs"])
        if secs and ncfg:
            tsec[(R_["meta"].get("route", "dense"), R_["meta"]["n"])] = (secs / ncfg, R_["meta"].get("workers", 1 if R_["meta"].get("route", "dense") == "dense" else None))
    mh_p = max(masses)
    for item in [x for x in a.production.split(",") if x]:
        fpath, zc_ = item.rsplit(":", 1)
        FR = json.load(open(fpath))
        mm_ = min(FR["members"], key=lambda x: abs(x["z_c"] - float(zc_)))
        ml, aH, rs = float(FR["meta"]["mlight"]), float(FR["meta"]["aH"]), float(mm_["r_s"])
        mhp = math.sqrt(ml * ml + 2 * aH * aH)
        hp = M["h"] * mh_p / mhp
        npd = int(round(4 * rs / hp))
        for (route, n0), (t0, w0) in tsec.items():
            fac = (npd / n0) ** 2 if route == "fast" else (npd / n0) ** 4
            prow.append(["(%g, %g) z_c %s, r_s %.1f" % (ml, aH, zc_, rs), "%.3g" % hp, npd, route, n0, "%.0f" % t0, "%.3g" % (t0 * fac / 3600)])
    out += ["## P Production cost: the members' grid (R_max = 4 r_s, h scaled so that m_heavy h equals the pilot's), n = R_max/h, and the hours per configuration "
            "from the measured seconds per configuration of each route (fast: times (n/n_0)^2 at the same workers; dense: times (n/n_0)^4)", "",
            table(["member family", "h", "n", "route", "n_0 (measured)", "s per configuration (measured)", "hours per configuration (scaled)"], prow), ""]
    out = out[:2] + ["## S Summary (emitted)", ""] + ["- " + x for x in summ] + [""] + out[2:]
    open(a.md, "w").write("\n".join(out) + "\n")
    print("\n".join("- " + x for x in summ))
    print("wrote", a.md)


if __name__ == "__main__":
    main()
