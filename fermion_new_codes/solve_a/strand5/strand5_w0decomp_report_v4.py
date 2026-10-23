#!/usr/bin/env python
"""Strand 5 J5 decomposition report, version 4 (spec 10(p'''''); builder analyst), for the redesigned runs of strand5_w0radial_v9.py (J5decomp2):
Gaussians psi = P + e f_w (N = 1, e = +-1e-2), the rulers P = 1, 1.05, 1.1, 1.2 at h = 1 and 0.5 with the widths fixed in the flat frame (grid widths
w/P^2), P = 1 alone at h = 0.25 and 1.44, the third width w = 300 at P = 1 (h = 1, 0.5).  The grid identity (version 9's docstring; checked here)
W(P, e; m, h, w/P^2) = W(1, e/P; m, P^2 h, w) makes every (h, P) a flat-ruler point at H = P^2 h: k_a(P; h) = k_1(P^2 h), P^4 g_4(P; h) = g_4,1(P^2 h).
Sections (numbers emitted, not retyped):
(A) per (h, P): W_0^(2) = (W(e) + W(-e))/(2 e^2) and c_3 = (W(e) - W(-e))/(2 e^3) per frame width; the two-width (30, 100) solve
    -k a(w_g) - g_4 b(w_g) = W_0^(2) with a(w) = (3/2) pi^(3/2) w, b(w) = (15/4) pi^(3/2)/w at the grid widths w_g; H = P^2 h.
(A3) at P = 1 (h = 1, 0.5) the three-width (30, 100, 300) solve with c(w) = (105/8) pi^(3/2)/w^3 (the q^6 term) against the two-width one.
(I) the identity check: P = 1.2 at h = 1 (grid widths w/1.44) against P = 1 at h = 1.44 (widths w): config by config W(1.2, +-e) against W(1, +-e/1.2),
    and k, g_4 both ways (the e = 1e-2 sets).
(K) k_1(H) over every point; two fits, (i) k_0 + H^2 (c_1 + c_2 ln H) (the a^2 |log a| class) and (ii) k_0 + c_1 H + c_2 H^2, with their largest
    residuals; q_0(h) = 2 H k_1'(H)/k_1(H) at H = h from each fit, from the direct ruler scans at h = 1 and 0.5 (k_a(P) quadratic in P, q_0 = k_a'(1)/k_a(1)),
    and the flat points' finite differences 2 ln(k_1(h_1)/k_1(h_2))/ln(h_1/h_2) at H = (h_1 h_2)^(1/2).
(B) per star record (h): the eq. 7 class with k_a(psi) = k_1(psi^2 h) (fit (i); fit (ii) beside): Dk(psi) = k_1(psi^2 h) - k_1(h),
    W_7(rho) = -int_0^rho 4 pi r^2 Dk psi' chi' dr, S(rho) = 4 pi rho^2 (chi - 1) Dk psi', F(rho) = 4 pi rho^2 Dk psi' against table Q's sums
    (L - 1)(d - d_lin) and (d - d_lin) at rho = 40, 80 (about r_s), r_J, 240, 300 (interior; the exterior parts as the value minus the value at 80);
    the remainder (measured - W_7 - S) over the linear sum (L - 1) d_lin; psi^2 h outside the fitted H range is flagged.
(C) the in-star (rho = 80) measured ratio's h-sequence (h = 1, 0.5, 0.25) with the extrapolation h -> 0 at the observed order and Richardson at
    orders 1 and 2; the same for the remainder.
(D) c_3 at P = 1 and fixed width over h, the local power ln(c_3(h_1)/c_3(h_2))/ln(h_1/h_2), beside eq. 7's cubic -q_0(h) k_1(h) (2/3)^(5/2) a(w)
    (fit (i)); the dense pilot's c_3 at w = 20 and 60 (h = 1, e = 1e-2) beside when given."""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

A_ = lambda w: 1.5 * math.pi ** 1.5 * w
B_ = lambda w: 3.75 * math.pi ** 1.5 / w
C_ = lambda w: 13.125 * math.pi ** 1.5 / w ** 3
C23 = (2.0 / 3.0) ** 2.5
E0 = 1e-2


def f(v, sig=4):
    if v is None:
        return "-"
    v = float(v)
    if v == 0:
        return "0"
    return ("%.*e" % (sig - 1, v)) if (abs(v) < 1e-3 or abs(v) >= 1e5) else ("%.*g" % (sig, v))


def table(head, rows):
    return "\n".join(["| " + " | ".join(str(c) for c in head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def rk(x):
    return round(float(x), 9)


def load(paths):
    """(h, P, w_frame, e) -> (W_c, w_grid, n)"""
    E = {}
    for p in paths:
        R = json.load(open(p))
        h = rk(R["meta"]["h"])
        for key, c in R["configs"].items():
            if not key.startswith("gauss"):
                continue
            P = rk(c.get("ruler", 1.0))
            wg = float(c["width"])
            wf = rk(c.get("frame_width", wg))
            E[(h, P, wf, rk(c["eps"]))] = (c["L_variants"]["psi_only"]["W_c"], wg, c.get("n"))
    return E


def parts(E, h, P, wf, e=E0):
    p, m = E.get((h, P, wf, rk(e))), E.get((h, P, wf, rk(-e)))
    if p is None or m is None:
        return None
    return (p[0] + m[0]) / (2 * e * e), (p[0] - m[0]) / (2 * e ** 3), p[1]


def kfit(E, h, P, wfs):
    rows = [parts(E, h, P, wf) for wf in wfs]
    if any(r is None for r in rows):
        return None
    cols = [A_, B_, C_][:len(wfs)]
    M = np.array([[-c(r[2]) for c in cols] for r in rows])
    return np.linalg.solve(M, np.array([r[0] for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True, help="comma-separated decomposition records (strand5_w0radial_v9.py --what gauss)")
    ap.add_argument("--stars", required=True, help="comma-separated h=path of the star records with densities")
    ap.add_argument("--pilot-gauss", default=None, help="the dense pilot's Gaussian record (w = 20, 60 at h = 1), for c_3 beside")
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    import strand5_w0radial_v4 as W4
    E = load([x for x in a.decomp.split(",") if x])
    hs_all = sorted({k[0] for k in E}, reverse=True)
    out = ["# Strand 5 J5 decomposition tables, redesigned runs (generated by solve/code/strand5/strand5_w0decomp_report_v4.py)", ""]
    summ = []
    # (A) per (h, P)
    arow, K = [], {}
    for h in hs_all:
        for P in sorted({k[1] for k in E if k[0] == h}):
            wfs = sorted({k[2] for k in E if k[0] == h and k[1] == P})
            for wf in wfs:
                pr = parts(E, h, P, wf)
                if pr:
                    arow.append(["%g" % h, "%.2f" % P, "%g" % wf, "%.6g" % pr[2], "%.12e" % pr[0], "%.5e" % pr[1]])
            kg = kfit(E, h, P, [30.0, 100.0])
            if kg is not None:
                K[(h, P)] = (float(kg[0]), float(kg[1]))
    out += ["## A Per grid spacing h and ruler P: the frame width, the grid width, W_0^(2) (e = 1e-2) and the cubic coefficient c_3", "",
            table(["h", "P", "frame w", "grid w", "W_0^(2)", "c_3"], arow), ""]
    krow = [["%g" % h, "%.2f" % P, "%.6g" % (P * P * h), "%.12e" % K[(h, P)][0], "%.8e" % K[(h, P)][1], "%.8e" % (P ** 4 * K[(h, P)][1])]
            for (h, P) in sorted(K, key=lambda x: x[1] * x[1] * x[0])]
    out += ["## A2 The two-width (30, 100) solve per (h, P), ordered by H = P^2 h: k_a(P; h) = k_1(H) and P^4 g_4(P; h) = g_4,1(H)", "",
            table(["h", "P", "H = P^2 h", "k_a(P; h)", "g_4(P; h)", "P^4 g_4(P; h)"], krow), ""]
    # (A3) three widths
    a3 = []
    for h in hs_all:
        k3 = kfit(E, h, 1.0, [30.0, 100.0, 300.0])
        if k3 is not None and (h, 1.0) in K:
            k2 = K[(h, 1.0)]
            a3.append(["%g" % h, "%.12e" % k3[0], "%.8e" % k3[1], "%.6e" % k3[2], "%.12e" % k2[0], "%.8e" % k2[1], "%.3e" % (k2[0] / k3[0] - 1),
                       "%.3e" % (k2[1] / k3[1] - 1)])
            summ.append("A3 (h = %g): the three-width k = %s, g_4 = %s, g_6 = %s; the two-width k and g_4 differ from them by %s and %s (relative)." % (
                h, "%.10e" % k3[0], "%.6e" % k3[1], "%.4e" % k3[2], "%.2e" % (k2[0] / k3[0] - 1), "%.2e" % (k2[1] / k3[1] - 1)))
    if a3:
        out += ["## A3 At P = 1: the three-width (30, 100, 300) solve with the q^6 term against the two-width (30, 100) one", "",
                table(["h", "k (3 widths)", "g_4 (3 widths)", "g_6 (3 widths)", "k (2 widths)", "g_4 (2 widths)", "k_2/k_3 - 1", "g_4,2/g_4,3 - 1"], a3), ""]
    # (I) identity check
    irow = []
    for wf in (30.0, 100.0):
        for s in (1, -1):
            x = E.get((1.0, 1.2, wf, rk(s * E0)))
            y = E.get((1.44, 1.0, wf, rk(s * E0 / 1.2)))
            if x and y:
                irow.append(["%g" % wf, "%+.0e" % (s * E0), "%.15e" % x[0], "%.15e" % y[0], "%.2e" % (x[0] / y[0] - 1), x[2], y[2]])
    if irow:
        out += ["## I The identity check: W(P = 1.2, e; h = 1, grid width w/1.44) against W(1, e/1.2; h = 1.44, width w), config by config", "",
                table(["frame w", "e", "W(1.2, e; h = 1)", "W(1, e/1.2; h = 1.44)", "relative difference", "n (h = 1)", "n (h = 1.44)"], irow), ""]
        line = "I: the grid identity W(1.2, e; h = 1, w/1.44) = W(1, e/1.2; h = 1.44, w) holds config by config to %s (largest relative difference over %d configs)" % (
            f(max(abs(float(r[4])) for r in irow), 2), len(irow))
        if (1.0, 1.2) in K and (1.44, 1.0) in K:
            line += "; k_a(1.2; h = 1) / k_1(1.44) - 1 = %s and 1.2^4 g_4(1.2; h = 1) / g_4,1(1.44) - 1 = %s (both from e = 1e-2)" % (
                "%.2e" % (K[(1.0, 1.2)][0] / K[(1.44, 1.0)][0] - 1), "%.2e" % (1.2 ** 4 * K[(1.0, 1.2)][1] / K[(1.44, 1.0)][1] - 1))
        summ.append(line + ".")
    # (K) k_1(H) fits
    byH = {}
    for (h, P), (k, g) in K.items():
        byH.setdefault(rk(P * P * h), []).append(k)
    Hs = np.array(sorted(byH))
    ks = np.array([np.mean(byH[H]) for H in Hs])
    fits = {}
    if len(Hs) >= 4:
        Xi = np.stack([np.ones_like(Hs), Hs ** 2, Hs ** 2 * np.log(Hs)], 1)
        ci, *_ = np.linalg.lstsq(Xi, ks, rcond=None)
        Xii = np.stack([np.ones_like(Hs), Hs, Hs ** 2], 1)
        cii, *_ = np.linalg.lstsq(Xii, ks, rcond=None)
        fits["i"] = (lambda H, c=ci: c[0] + c[1] * H ** 2 + c[2] * H ** 2 * np.log(H),
                     lambda H, c=ci: 2 * c[1] * H + c[2] * H * (2 * np.log(H) + 1), ci, float(np.max(np.abs(Xi @ ci - ks)) / abs(ci[0])))
        fits["ii"] = (lambda H, c=cii: c[0] + c[1] * H + c[2] * H ** 2, lambda H, c=cii: c[1] + 2 * c[2] * H, cii,
                      float(np.max(np.abs(Xii @ cii - ks)) / abs(cii[0])))
        out += ["## K k_1(H) over every point (duplicates at one H averaged) and the two fits' values", "",
                table(["H", "k_1(H)", "fit (i)", "fit (ii)"], [["%.6g" % H, "%.12e" % k, "%.12e" % fits["i"][0](H), "%.12e" % fits["ii"][0](H)]
                                                                 for H, k in zip(Hs, ks)]), ""]
        summ.append("K: over H = %s to %s (%d points), fit (i) k_0 + H^2 (c_1 + c_2 ln H): k_0 = %s, c_1 = %s, c_2 = %s, largest residual %s of k_0; fit (ii) "
                    "k_0 + c_1 H + c_2 H^2: k_0 = %s, c_1 = %s, c_2 = %s, largest residual %s." % (
                        "%g" % Hs[0], "%g" % Hs[-1], len(Hs), "%.10e" % ci[0], "%.5e" % ci[1], "%.5e" % ci[2], f(fits["i"][3], 2),
                        "%.10e" % cii[0], "%.5e" % cii[1], "%.5e" % cii[2], f(fits["ii"][3], 2)))
    qrow = []
    q0 = {}
    for h in hs_all:
        Ps = sorted(P for (hh, P) in K if hh == h)
        direct = None
        if len(Ps) >= 3 and 1.0 in Ps:
            Pa = np.array(Ps)
            kk = np.array([K[(h, P)][0] for P in Ps])
            X = np.stack([np.ones_like(Pa), Pa - 1, (Pa - 1) ** 2], 1)
            cf, *_ = np.linalg.lstsq(X, kk, rcond=None)
            direct = float(cf[1] / cf[0])
        qi = float(2 * h * fits["i"][1](h) / fits["i"][0](h)) if fits else None
        qii = float(2 * h * fits["ii"][1](h) / fits["ii"][0](h)) if fits else None
        q0[h] = qi
        qrow.append(["%g" % h, f(direct, 5), f(qi, 5), f(qii, 5)])
    flat = sorted([h for (h, P) in K if P == 1.0], reverse=True)
    fd = []
    for h1, h2 in zip(flat[:-1], flat[1:]):
        fd.append(["%g, %g" % (h1, h2), "%.4g" % math.sqrt(h1 * h2), "%.5e" % (2 * math.log(K[(h1, 1.0)][0] / K[(h2, 1.0)][0]) / math.log(h1 / h2))])
    out += ["## Q0 q_0(h) = k_a'(1)/k_a(1): the direct ruler scan (quadratic in P), fit (i) and fit (ii) of k_1(H) (2 H k_1'/k_1 at H = h)", "",
            table(["h", "direct scan", "fit (i)", "fit (ii)"], qrow), "",
            "Finite differences of the flat (P = 1) points, 2 ln(k_1(h_1)/k_1(h_2))/ln(h_1/h_2):", "",
            table(["h_1, h_2", "at H", "q_0"], fd), ""]
    summ.append("Q0: q_0(h) at h = %s: direct scan %s; fit (i) %s; fit (ii) %s; the flat points' finite differences %s." % (
        ", ".join(r[0] for r in qrow), ", ".join(r[1] for r in qrow), ", ".join(r[2] for r in qrow), ", ".join(r[3] for r in qrow),
        ", ".join("%s at H = %s" % (r[2], r[1]) for r in fd)))
    # (B) the star records
    sts = dict(x.split("=", 1) for x in a.stars.split(",") if x)
    brow, b2row, hseq = [], [], {}
    for hl in sorted(sts, key=lambda x: -float(x)):
        if not fits or not os.path.exists(sts[hl]):
            continue
        h = float(hl)
        S = json.load(open(sts[hl]))
        M = S["meta"]
        mem = W4.Member(M["family"], M["profiles"], M["z_c"], M.get("interp", "cubic"))
        Dn = np.load(sts[hl].replace(".json", "_density.npz"))
        ri, rh = Dn["r_int"], Dn["r_half"]
        pi_, Ni = mem.fields(ri)
        ph_, Nh = mem.fields(rh)
        L1 = np.concatenate([Ni * pi_, Nh * ph_]) - 1
        d1 = Dn["star_density_e+1e+00"]
        dl = (Dn["star_density_e+1e-02"] - Dn["star_density_e-1e-02"]) / 2e-2
        rr = np.concatenate([ri, rh])
        o = np.argsort(rr)
        rs_ = rr[o]
        cQ = np.cumsum((L1 * (d1 - dl))[o])
        cF = np.cumsum((d1 - dl)[o])
        cL = np.cumsum((L1 * dl)[o])

        def prof(r):
            r = np.asarray(r, float)
            ins = r <= mem.rJ
            ps = np.where(ins, mem.sp(np.minimum(r, mem.rJ)), 1 + mem.A / r)
            dps = np.where(ins, mem.sp(np.minimum(r, mem.rJ), 1), -mem.A / r ** 2)
            Nn = np.where(ins, mem.sn(np.minimum(r, mem.rJ)), 0.0)
            dNn = np.where(ins, mem.sn(np.minimum(r, mem.rJ), 1), 0.0)
            chi = np.where(ins, Nn * ps, 1 - mem.B / r)
            dchi = np.where(ins, dNn * ps + Nn * dps, mem.B / r ** 2)
            return ps, dps, chi, dchi
        xg, wg = np.polynomial.legendre.leggauss(16)
        psmax = float(np.max(prof(np.linspace(1e-3, mem.rJ, 4001))[0]))
        flag = "" if (h * 1.0 >= Hs[0] - 1e-12 and h * psmax ** 2 <= Hs[-1] + 1e-12) else " (extrapolated: psi^2 h up to %.4g)" % (h * psmax ** 2)
        res = {}
        for fk in ("i", "ii"):
            kf = fits[fk][0]
            Dk = lambda P, kf=kf: kf(P * P * h) - kf(h)

            def W7(rho, Dk=Dk):
                edges = np.linspace(0.0, rho, 401)
                tot = 0.0
                for lo, hi in zip(edges[:-1], edges[1:]):
                    r = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
                    ps, dps, chi, dchi = prof(r)
                    tot += float(np.sum(0.5 * (hi - lo) * wg * 4 * math.pi * r * r * Dk(ps) * dps * dchi))
                return -tot
            for rho in (40.0, 80.0, float(mem.rJ), 240.0, 300.0):
                if rho > rs_[-1]:
                    continue
                kq = np.searchsorted(rs_, rho, side="right") - 1
                ps, dps, chi, dchi = prof(np.array([rho]))
                Fv = float(4 * math.pi * rho * rho * Dk(ps[0]) * dps[0])
                w7 = W7(rho)
                res[(fk, rho)] = (float(cQ[kq]), w7, float((chi[0] - 1) * Fv), float(cF[kq]), Fv, float(cL[kq]))
        for rho in sorted({r for (_, r) in res}):
            mq, w7, s_, mf, Fv, ml = res[("i", rho)]
            pred = w7 + s_
            ext = ["-", "-", "-"]
            if rho > 80.0 and ("i", 80.0) in res:
                m80, w80, s80 = res[("i", 80.0)][:3]
                ext = ["%.5e" % (mq - m80), "%.5e" % (pred - w80 - s80), "%.5e" % (w7 - w80)]
            brow.append(["%g%s" % (h, flag), "%.0f" % rho, "%.5e" % mq, "%.5e" % w7, "%.5e" % s_, "%.5e" % pred, f(mq / pred, 4) if pred else "-",
                         "%.4e" % ((mq - pred) / ml), "%.5e" % mf, "%.5e" % Fv, "%.4e" % (mq / ml)] + ext)
            mq2, w72, s2 = res[("ii", rho)][:3]
            b2row.append(["%g" % h, "%.0f" % rho, "%.5e" % w72, "%.5e" % s2, "%.5e" % (w72 + s2), "%.4e" % ((mq2 - w72 - s2) / ml)])
            if abs(rho - 80.0) < 1e-9:
                hseq[h] = (mq / ml, (mq - pred) / ml, pred / ml, (mq2 - w72 - s2) / ml)
    out += ["## B Table Q's sums against the eq. 7 class with k_a(psi) = k_1(psi^2 h) (fit (i)): the measured sum (L - 1)(d - d_lin), W_7, S, the predicted "
            "W_7 + S, their ratio, the remainder (measured - W_7 - S) over the linear sum (L - 1) d_lin, the measured and predicted flux differences, the "
            "measured ratio, and beyond rho = 80 the exterior parts (the value minus the value at 80)", "",
            table(["h", "rho", "measured", "W_7", "S", "W_7 + S", "measured/predicted", "remainder/linear", "flux measured", "flux predicted F",
                   "measured ratio", "exterior measured", "exterior predicted", "exterior W_7"], brow), "",
            "## B2 The same with fit (ii)", "", table(["h", "rho", "W_7", "S", "W_7 + S", "remainder/linear"], b2row), ""]

    def seq_line(name, vals):
        hs = sorted(vals, reverse=True)
        line = "%s over h = %s: %s" % (name, ", ".join("%g" % h for h in hs), ", ".join("%.4e" % vals[h] for h in hs))
        if len(hs) >= 3:
            x1, x2, x3 = (vals[h] for h in hs[-3:])
            d1, d2 = x1 - x2, x2 - x3
            ratio = d2 / d1 if d1 else None
            line += "; successive differences %s, %s (ratio %s)" % ("%.4e" % d1, "%.4e" % d2, f(ratio, 3))
            if ratio is not None and 0 < ratio < 1:
                line += ", the observed order %s, the extrapolation at it %s" % ("%.3f" % (-math.log(ratio, 2)), "%.4e" % (x3 - d2 * ratio / (1 - ratio)))
            else:
                line += ", no steady ratio (no extrapolation at the observed order)"
            line += "; Richardson from the two finest at orders 1 and 2: %s and %s" % ("%.4e" % (x3 - d2), "%.4e" % (x3 - d2 / 3))
        return line
    if hseq:
        summ.append("C: " + seq_line("the in-star (rho = 80) measured ratio sum (L - 1)(d - d_lin)/sum (L - 1) d_lin", {h: v[0] for h, v in hseq.items()}) + ".")
        summ.append("C: eq. 7's predicted in-star ratio (W_7 + S)/linear with fit (i): %s; " % ", ".join(
            "%.4e (h = %g)" % (hseq[h][2], h) for h in sorted(hseq, reverse=True)) + seq_line("the remainder (fit (i))", {h: v[1] for h, v in hseq.items()}) +
            "; with fit (ii): %s." % ", ".join("%.4e (h = %g)" % (hseq[h][3], h) for h in sorted(hseq, reverse=True)))
    # (D) c_3 over h
    drow = []
    pilot = load([a.pilot_gauss]) if a.pilot_gauss and os.path.exists(a.pilot_gauss) else {}
    for wf in sorted({k[2] for k in E if k[1] == 1.0}):
        seq = [(h, parts(E, h, 1.0, wf)) for h in hs_all]
        seq = [(h, p) for h, p in seq if p is not None]
        for i, (h, p) in enumerate(seq):
            pw = "-"
            if i > 0:
                h0, p0 = seq[i - 1]
                pw = "%.4g" % (math.log(p0[1] / p[1]) / math.log(h0 / h)) if p0[1] / p[1] > 0 else "sign change"
            e7 = -q0[h] * fits["i"][0](h) * C23 * A_(wf) if (fits and q0.get(h) is not None) else None
            drow.append(["%g" % wf, "%g" % h, "%.6e" % p[1], pw, f(e7, 5), "%.12e" % p[0]])
    for wf in sorted({k[2] for k in pilot}):
        p = parts(pilot, 1.0, 1.0, wf)
        if p:
            e7 = -q0[1.0] * fits["i"][0](1.0) * C23 * A_(wf) if (fits and q0.get(1.0) is not None) else None
            drow.append(["%g (dense pilot)" % wf, "1", "%.6e" % p[1], "-", f(e7, 5), "%.12e" % p[0]])
    out += ["## D The cubic coefficient c_3 at P = 1 and fixed width over h, its local power in h against the next coarser h, eq. 7's cubic "
            "-q_0(h) k_1(h) (2/3)^(5/2) a(w) (fit (i)), and W_0^(2) beside", "",
            table(["w", "h", "c_3", "local power in h", "eq. 7 cubic", "W_0^(2)"], drow), ""]
    if drow:
        summ.append("D: c_3 at P = 1: " + "; ".join("w = %s, h = %s: %s (power %s; eq. 7's cubic %s)" % (r[0], r[1], r[2], r[3], r[4]) for r in drow) + ".")
    out = out[:2] + ["## S Summary (emitted)", ""] + ["- " + x for x in summ] + [""] + out[2:]
    open(a.md, "w").write("\n".join(out) + "\n")
    print("\n".join("- " + x for x in summ))
    print("wrote", a.md)


if __name__ == "__main__":
    main()
