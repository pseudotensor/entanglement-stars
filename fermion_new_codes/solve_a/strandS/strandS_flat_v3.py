#!/usr/bin/env python
"""Strand S, job S2 (SPEC_strandS_graded_entropy.md secs. 1 to 3, 5 to 7; builder analyst): the flat [100] graded cut on job S1's table.

The planar coefficient per packet, eta_100(q) = (8/pi^2) Int over the triangle 0 <= k_z <= k_y <= pi/2 of S_ch(sqrt(q^2 + sin^2 k_y + sin^2 k_z))
(the zone average by the 32 copies of the triangle), with S_ch from S1's table (strandS_table_v1.py: the 2-component chain, 200 points per
decade on [0.006, 3], muL >= 24): the cubic spline of S_ch + alpha log mu in log mu (alpha = 2/3 FIXED), and below the table's mu_min the
fixed-alpha fit -alpha log mu + beta + gamma mu^2 log mu + delta mu^2 + eps mu^4 log mu + zeta mu^4 (S1's coefficients); the junction's
continuity is reported.  The quadrature: polar coordinates about the node (0, 0), radial Gauss-Legendre panels log-spaced from --rho-min to
the triangle's edge (pi/2)/cos(phi), --npan panels x --ngl points, angular Gauss-Legendre with --nphi points on [0, pi/4]; the graded sum
sum_s eta_s S_ch(...) and the unsigned sum formed per point; the ladder (npan, ngl, nphi) doubled one at a time; the inner disc rho < mu_min
and the fit-evaluated share reported.

Rectangles (squared masses over M^2, M the heaviest packet): light (0, 1/2, 1/2, 1), j = log 2; r1223 (1, 2, 2, 3)/3, j = (1/3)log(1/3) -
(4/3)log(2/3); q_s = aM sqrt(x_s); eta = (+, -, -, +).  hat k_0 = k_a(1) a^2 by J2e's integral (strand5_J2e_v2.route1, scipy quad in
u = t^(1/2) with the analytic tail, the rectangle's weight in closed form with expm1; route2, mpmath tanh-sinh at 30 digits, as the check).
Outputs per (rectangle, aM): the four eta_100(q_s), eta_* (unsigned), hat S_gr (graded), c_S = hat S_gr/((aM)^2 j), hat k_0 (both routes),
hat k_0/((aM)^2 j) against 3/pi^2, R_gr = hat S_gr/((pi/2) hat k_0), the unsigned ratios 2 pi eta_*/(3 j (aM)^2) and eta_*/((pi/2) hat k_0);
the quadrature ladder; the extrapolations c_S(aM) = c_S + c_2 (aM)^2 log(1/aM) + c_2' (aM)^2 and R_gr(aM) likewise on aM <= --fit-max.
Controls (SPEC sec. 5): C1 eta_100(0) against 0.7865834; C2 eta_100 at q = 0.1 and 0.5 against strand 3's stored A.1 (32,24) records (their
quoted errors); C6 hat k_0/((aM)^2 j) -> 3/pi^2; C7 V1 the exact pair {(+1, q'), (-1, q')} at q' = 0.3 (hat S_gr unchanged to 1e-14,
eta_* + 2 eta_100(0.3), hat k_0 unchanged identically), V2 a block neutral at the J order: the lead's literal V2 (the rectangle's copy at
2 m_s with weights -eta_s/4) has sum eta' m'^2 log m'^2 = -J for the copy, so it is reported as written beside the builder's repair
(labelled): four packets at x' = (1, 2, 3, 4)/4 M^2 with weights c' (c'_1 = 1/4) solving sum c' = sum c' x' = sum c' x' log x' = 0, whose
changes to hat S_gr and hat k_0 are of order (aM)^4 while eta_* changes at order 1; at aM = 0.1 and 0.025; C8 the rectangles' extrapolated
c_S.  Writes --out-json and --md (every number emitted here)."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import math
import sys
import time
from multiprocessing import Pool

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
S5 = os.environ.get("STRAND5_CODE", os.path.join(HERE, "..", "strand5"))
sys.path.insert(0, S5)
import strand5_J2e_v2 as J2E  # noqa: E402

ALPHA = 2.0 / 3.0
ETAS = np.array([1.0, -1.0, -1.0, 1.0])
RECTS = {"light": np.array([0.0, 0.5, 0.5, 1.0]), "r1223": np.array([1.0, 2.0, 2.0, 3.0]) / 3.0}


def jval(x):
    return float(sum(e * (v * math.log(v) if v > 0 else 0.0) for e, v in zip(ETAS, x)))


class Table:
    def __init__(self, T):
        self.mu = np.array(T["mu"])
        self.S = np.array(T["S_ch"])
        fx = T["fit_fixed_alpha"]
        self.c = [fx[k] for k in ("beta", "gamma", "delta", "eps", "zeta")]
        self.mu_min, self.mu_max = float(self.mu.min()), float(self.mu.max())
        lm = np.log(self.mu)
        self.spl = CubicSpline(lm, self.S + ALPHA * lm)

    def fit(self, m):
        b, g, d, e, z = self.c
        lm = np.log(m)
        return -ALPHA * lm + b + g * m * m * lm + d * m * m + e * m ** 4 * lm + z * m ** 4

    def __call__(self, mu):
        mu = np.asarray(mu, float)
        out = np.empty_like(mu)
        lo = mu < self.mu_min
        hi = ~lo
        if np.any(mu[hi] > self.mu_max * (1 + 1e-12)):
            raise ValueError("mu above the table: %g" % mu[hi].max())
        out[hi] = self.spl(np.log(mu[hi])) - ALPHA * np.log(mu[hi])
        out[lo] = self.fit(np.maximum(mu[lo], 1e-300))
        return out


def zone_points(rho_min, npan, ngl, nphi):
    xg, wg = leggauss(ngl)
    xp, wp = leggauss(nphi)
    phi = 0.5 * (math.pi / 4) * (xp + 1)
    wphi = 0.5 * (math.pi / 4) * wp
    K, W = [], []
    for ph, wph in zip(phi, wphi):
        rmax = (math.pi / 2) / math.cos(ph)
        e = np.geomspace(rho_min, rmax, npan + 1)
        r = (0.5 * (e[1:] - e[:-1])[:, None] * xg[None, :] + 0.5 * (e[1:] + e[:-1])[:, None]).ravel()
        wr = (0.5 * (e[1:] - e[:-1])[:, None] * wg[None, :]).ravel()
        K.append(np.stack([r * math.cos(ph), r * math.sin(ph)], 1))
        W.append(wr * r * wph)
    K = np.concatenate(K)
    W = np.concatenate(W) * 8 / math.pi ** 2
    rho = np.hypot(K[:, 0], K[:, 1])
    d2 = np.sin(K[:, 0]) ** 2 + np.sin(K[:, 1]) ** 2
    return d2, W, rho


def packet_values(S, qs, d2, W, rho, mu_min, etas=ETAS):
    per = []
    g = np.zeros_like(d2)
    u = np.zeros_like(d2)
    fit_any = np.zeros(d2.shape, bool)
    for e, q in zip(etas, qs):
        mu = np.sqrt(q * q + d2)
        s = S(mu)
        per.append(float(np.sum(W * s)))
        g += e * s
        u += s
        fit_any |= mu < mu_min
    G = float(np.sum(W * g))
    inner = float(np.sum(W[rho < mu_min] * g[rho < mu_min]))
    fitsh = float(np.sum(W[fit_any] * g[fit_any]))
    return per, G, float(np.sum(W * u)), inner, fitsh


def weight_fn(rect, aM):
    y = (aM ** 2) * RECTS[rect]
    if rect == "light":
        y2 = y[1]
        return lambda t: np.expm1(-t * y2) ** 2, y.tolist()
    y1 = y[0]
    return lambda t: np.exp(-t * y1) * np.expm1(-t * y1) ** 2, y.tolist()


def k0_job(args):
    rect, aM = args
    w, ys = weight_fn(rect, aM)
    r1, e1, tail = J2E.route1(w, ys, ETAS.tolist())
    r2, tail2 = J2E.route2(ys, ETAS.tolist())
    return rect, aM, r1, e1, r2


def k0_list(ys, etas):
    """hat k_0 for an arbitrary inventory (the controls): route 1 with the weight summed term by term (no closed form)."""
    w = lambda t: sum(e * np.exp(-t * y) for e, y in zip(etas, ys))
    return J2E.route1(w, list(ys), list(etas))[0]


def efit(aMs, vals, fmax):
    a = np.array(aMs)
    v = np.array(vals)
    sel = a <= fmax * (1 + 1e-12)
    X = np.stack([np.ones(sel.sum()), a[sel] ** 2 * np.log(1 / a[sel]), a[sel] ** 2], 1)
    c, *_ = np.linalg.lstsq(X, v[sel], rcond=None)
    r = X @ c - v[sel]
    return {"limit": float(c[0]), "c2_log": float(c[1]), "c2": float(c[2]), "max_resid": float(np.max(np.abs(r))), "points": int(sel.sum()),
            "aM_range": [float(a[sel].min()), float(a[sel].max())]}


def stored_A1(folder, q):
    best = None
    for p in glob.glob(os.path.join(folder, "planar_1_0_0_d32_nk24_q*.json")):
        d = json.load(open(p))
        if abs(float(d["mass"]) - q) < 1e-9:
            best = (float(d["eta"]), float(d["estimated_error"]), os.path.basename(p))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--aM", default="1.6,0.8,0.4,0.2,0.1,0.05,0.025,0.0125,0.00625,0.005,0.0025")
    ap.add_argument("--rho-min", type=float, default=1e-6)
    ap.add_argument("--npan", type=int, default=25)
    ap.add_argument("--ngl", type=int, default=8)
    ap.add_argument("--nphi", type=int, default=16)
    ap.add_argument("--fit-max", type=float, default=0.1)
    ap.add_argument("--A1-folder", default="<internal>/planar")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    t0 = time.time()
    T = json.load(open(a.table))
    S = Table(T)
    junction = float(S.fit(S.mu_min) - S.S[0])
    lad = T.get("ladder", [])
    table_err = max([abs(r["diff"]) for r in lad] + [0.0])
    aMs = [float(v) for v in a.aM.split(",")]
    settings = {"base": (a.npan, a.ngl, a.nphi), "npan x2": (2 * a.npan, a.ngl, a.nphi), "ngl x2": (a.npan, 2 * a.ngl, a.nphi),
                "nphi x2": (a.npan, a.ngl, 2 * a.nphi)}
    grids = {k: zone_points(a.rho_min, *v) for k, v in settings.items()}
    d2, W, rho = grids["base"]
    with Pool(a.workers) as pool:
        k0 = {(r, m): (r1, e1, r2) for r, m, r1, e1, r2 in pool.map(k0_job, [(r, m) for r in RECTS for m in aMs], chunksize=1)}
    print("k_0 by both routes (%.0f s)" % (time.time() - t0), flush=True)
    rows = []
    for rect, x in RECTS.items():
        j = jval(x)
        for aM in aMs:
            qs = aM * np.sqrt(x)
            per, G, U, inner, fitsh = packet_values(S, qs, d2, W, rho, S.mu_min)
            ladder = {k: packet_values(S, qs, *grids[k], S.mu_min)[1] for k in settings if k != "base"}
            r1, e1, r2 = k0[(rect, aM)]
            row = {"rect": rect, "x": x.tolist(), "j": j, "aM": aM, "q": qs.tolist(), "eta_per_packet": per, "eta_star": U, "S_gr": G,
                   "c_S": G / (aM * aM * j), "k0_route1": r1, "k0_route1_quad_err": e1, "k0_route2": r2, "k0_route_diff": r1 - r2,
                   "k0_over_aM2j": r1 / (aM * aM * j), "R_gr": G / (0.5 * math.pi * r1), "unsigned_record_form": 2 * math.pi * U / (3 * j * aM * aM),
                   "unsigned_over_pi2_k0": U / (0.5 * math.pi * r1), "inner_disc_share": inner / G if G else None,
                   "fit_evaluated_share": fitsh / G if G else None, "ladder_S_gr": ladder,
                   "ladder_max_rel": max(abs(v / G - 1) for v in ladder.values()) if G else None}
            rows.append(row)
            print("%s aM %g: S_gr %.10e c_S %.8f k0 %.10e (route diff %.1e) R_gr %.8f eta_* %.8f inner %.3f ladder %.1e" % (
                rect, aM, G, row["c_S"], r1, r1 - r2, row["R_gr"], U, row["inner_disc_share"], row["ladder_max_rel"]), flush=True)
    ext = {}
    for rect in RECTS:
        rr = [r for r in rows if r["rect"] == rect]
        ext[rect] = {"c_S": efit([r["aM"] for r in rr], [r["c_S"] for r in rr], a.fit_max),
                     "R_gr": efit([r["aM"] for r in rr], [r["R_gr"] for r in rr], a.fit_max),
                     "k0_over_aM2j": efit([r["aM"] for r in rr], [r["k0_over_aM2j"] for r in rr], a.fit_max)}
    # controls
    eta0 = packet_values(S, [0.0], d2, W, rho, S.mu_min, etas=[1.0])[0][0]
    C2 = []
    for q in (0.1, 0.5):
        ours = packet_values(S, [q], d2, W, rho, S.mu_min, etas=[1.0])[0][0]
        st = stored_A1(a.A1_folder, q)
        C2.append({"q": q, "ours": ours, "stored_eta": st[0] if st else None, "stored_err": st[1] if st else None,
                   "record": st[2] if st else None, "diff": (ours - st[0]) if st else None})
    C7 = []
    for rect in RECTS:
        x = RECTS[rect]
        for aM in (0.1, 0.025):
            qs = aM * np.sqrt(x)
            base = packet_values(S, qs, d2, W, rho, S.mu_min)
            k0b = k0[(rect, aM)][0]
            qp = 0.3
            v1 = packet_values(S, list(qs) + [qp, qp], d2, W, rho, S.mu_min, etas=list(ETAS) + [1.0, -1.0])
            lit_q = list(qs) + list(2 * qs)
            lit_e = list(ETAS) + list(-ETAS / 4)
            lit = packet_values(S, lit_q, d2, W, rho, S.mu_min, etas=lit_e)
            lit_k0 = k0_list([q * q for q in lit_q], lit_e)
            xb = np.array([1.0, 2.0, 3.0, 4.0]) / 4
            A = np.stack([np.ones(4), xb, xb * np.log(xb)])
            c_rest = np.linalg.solve(A[:, 1:], -A[:, 0] * 0.25)
            cb = np.concatenate([[0.25], c_rest])
            qb = aM * np.sqrt(xb)
            rep = packet_values(S, list(qs) + list(qb), d2, W, rho, S.mu_min, etas=list(ETAS) + list(cb))
            rep_k0 = k0_list([q * q for q in list(qs) + list(qb)], list(ETAS) + list(cb))
            C7.append({"rect": rect, "aM": aM, "S_gr": base[1], "eta_star": base[2], "k0": k0b,
                       "V1_dS_gr": v1[1] - base[1], "V1_eta_star_change": v1[2] - base[2], "V1_expected_2eta03": 2 * packet_values(S, [qp], d2, W, rho, S.mu_min, etas=[1.0])[0][0],
                       "V2_literal_copy_J": float(sum(-e / 4 * (4 * v) * math.log(4 * v) for e, v in zip(ETAS, x) if v > 0)),
                       "V2_literal_dS_gr_rel": lit[1] / base[1] - 1, "V2_literal_dk0_rel": lit_k0 / k0b - 1,
                       "V2_repair_weights": cb.tolist(), "V2_repair_moments": [float(A[i] @ cb) for i in range(3)],
                       "V2_repair_dS_gr_rel": rep[1] / base[1] - 1, "V2_repair_dk0_rel": rep_k0 / k0b - 1,
                       "V2_repair_eta_star_change": rep[2] - base[2]})
    out = {"meta": vars(a), "table_meta": T["meta"], "junction_fit_minus_table": junction, "table_ladder_max": table_err,
           "fit_fixed_alpha": T["fit_fixed_alpha"], "fit_free_alpha": T["fit_free_alpha"], "C3": T.get("C3"),
           "quadrature_points_base": int(len(W)), "rows": rows, "extrapolations": ext,
           "C1": {"eta100_0": eta0, "record": 0.7865834, "diff": eta0 - 0.7865834}, "C2": C2, "C7": C7,
           "targets": {"a1_one_over_3pi": 1 / (3 * math.pi), "three_over_pi2": 3 / math.pi ** 2, "two_ninths": 2 / 9},
           "seconds": time.time() - t0}
    json.dump(out, open(a.out_json, "w"), indent=1)

    def f(v, d=6):
        if v is None:
            return "-"
        return "%.*g" % (d, v) if (v == 0 or 1e-3 <= abs(v) < 1e5) else "%.4e" % v
    L = ["# Strand S job S2: the flat [100] graded cut (strandS_flat_v3.py on S1's table; SPEC_strandS_graded_entropy.md secs. 1 to 3, 5 to 7)", "",
         "- Table: %s (S1); the fit-to-table junction at mu_min = %s: %s; the table ladder's largest |S(2L) - S(L)| %s; the fixed-alpha fit's largest "
         "residual %s; quadrature base (npan, ngl, nphi) = (%d, %d, %d) from rho_min = %s, %d points." % (
             os.path.basename(a.table), f(S.mu_min), f(junction, 3), f(table_err, 3), f(T["fit_fixed_alpha"]["max_resid"], 3), a.npan, a.ngl, a.nphi,
             f(a.rho_min), len(W)),
         "- C1: eta_100(0) = %s against the record's 0.7865834 (difference %s; tolerance 2e-5)." % (f(eta0, 10), f(eta0 - 0.7865834, 3)),
         "- C2: " + "; ".join("q = %s: ours %s, strand 3's stored (32,24) %s +- %s (%s), difference %s" % (
             f(c["q"]), f(c["ours"], 10), f(c["stored_eta"], 10), f(c["stored_err"], 3), c["record"], f(c["diff"], 3)) for c in C2) + ".",
         "- C3 (from S1): " + "; ".join("mu %s: S_4 - 2 S_2 = %s" % (f(c["mu"]), f(c["S4_minus_2S2"], 3)) for c in (T.get("C3") or [])) + ".",
         "- C4 (from S1): the free-alpha refit alpha = %s (2/3 + %s)." % (f(T["fit_free_alpha"]["alpha"], 10), f(T["fit_free_alpha"]["alpha"] - ALPHA, 3)), "",
         "| rectangle | aM | eta_100(q_s) per packet | eta_* | hat S_gr | c_S | hat k_0 (route 1) | routes' difference | hat k_0/((aM)^2 j) | R_gr | 2 pi eta_*/(3 j (aM)^2) | eta_*/((pi/2) hat k_0) | inner-disc share | quadrature ladder (largest rel.) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["rect"], f(r["aM"]), ", ".join(f(v, 10) for v in r["eta_per_packet"]), f(r["eta_star"], 9), f(r["S_gr"], 9), f(r["c_S"], 8),
            f(r["k0_route1"], 10), f(r["k0_route_diff"], 2), f(r["k0_over_aM2j"], 8), f(r["R_gr"], 8), f(r["unsigned_record_form"], 5),
            f(r["unsigned_over_pi2_k0"], 5), f(r["inner_disc_share"], 3), f(r["ladder_max_rel"], 2)))
    L += ["", "Extrapolations on aM <= %s (fit X(aM) = X_0 + c (aM)^2 log(1/aM) + c' (aM)^2): " % f(a.fit_max)]
    for rect, e in ext.items():
        L.append("- %s: c_S -> %s (residual %s; 1/(3 pi) = %s), R_gr -> %s (residual %s; 2/9 = %s), hat k_0/((aM)^2 j) -> %s (3/pi^2 = %s)." % (
            rect, f(e["c_S"]["limit"], 9), f(e["c_S"]["max_resid"], 2), f(1 / (3 * math.pi), 9), f(e["R_gr"]["limit"], 9), f(e["R_gr"]["max_resid"], 2),
            f(2 / 9, 9), f(e["k0_over_aM2j"]["limit"], 9), f(3 / math.pi ** 2, 9)))
    L.append("- C8: the rectangles' extrapolated c_S differ by %s (relative %s)." % (
        f(ext["light"]["c_S"]["limit"] - ext["r1223"]["c_S"]["limit"], 3), f(ext["light"]["c_S"]["limit"] / ext["r1223"]["c_S"]["limit"] - 1, 3)))
    L += ["", "C7, the neutral packets (relative changes against the rectangle's own hat S_gr and hat k_0):"]
    for c in C7:
        L.append("- %s, aM = %s: V1 (the pair at q' = 0.3): Delta hat S_gr = %s, Delta eta_* = %s against 2 eta_100(0.3) = %s; V2 as written (the copy at "
                 "2 m_s, weights -eta_s/4; the copy's J is %s, i.e. minus the rectangle's): Delta hat S_gr/hat S_gr = %s, Delta hat k_0/hat k_0 = %s; "
                 "V2 repaired (builder, labelled: x' = (1, 2, 3, 4)/4, weights %s, moments %s): Delta hat S_gr/hat S_gr = %s, Delta hat k_0/hat k_0 = %s, "
                 "Delta eta_* = %s." % (
                     c["rect"], f(c["aM"]), f(c["V1_dS_gr"], 3), f(c["V1_eta_star_change"], 10), f(c["V1_expected_2eta03"], 10), f(c["V2_literal_copy_J"], 6),
                     f(c["V2_literal_dS_gr_rel"], 4), f(c["V2_literal_dk0_rel"], 4), ", ".join(f(w, 6) for w in c["V2_repair_weights"]),
                     ", ".join(f(m, 2) for m in c["V2_repair_moments"]), f(c["V2_repair_dS_gr_rel"], 4), f(c["V2_repair_dk0_rel"], 4),
                     f(c["V2_repair_eta_star_change"], 8)))
    L += ["", "- J = +sum_s eta_s m_s^2 log m_s^2 (> 0); j = J/M^2; every number above is emitted by the script (%.0f s)." % out["seconds"]]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
