#!/usr/bin/env python
"""Strand S, job S2 revision 1b (the controls named by the reviewer B red team of the judgment, reviewb/review_output/reviewb_azure_redteam_strandS_*,
secs. 3 and 7; builder analyst): strandS_flat_v3.py's computation (imported, unchanged) with
(1) the disc below rho_min added: the radial panels now start at --rho-min 1e-12 (version 3 started at 1e-6 and omitted the disc below, the red
    team's sec. 3; the fit form evaluates the integrand there), and the version-3 value at rho_min = 1e-6 beside it;
(2) the shares printed: the inner disc (rho < mu_min), the fit-evaluated share (at least one packet below mu_min) and the all-packets-fit share;
(3) the per-packet coefficients to 13 significant digits;
(4) the table-and-fit variations, each with the full extrapolation on aM <= 0.1, 0.05, 0.025: the table thinned to every second point (100 per
    decade); the fixed-alpha fit refitted on [mu_min, 0.03] and on [mu_min, 0.1]; the matching point moved to mu = 0.01 and to mu = 0.02 (the table
    used only above it, the fit refitted on [match, 0.05] and [match, 0.08]); the spread of the limits over the variations as the systematic error
    of the table and fit route (hat k_0 is the same J2e integral in every variation).
Writes --out-json and --md.  Args as version 3, plus --rho-min (default 1e-12)."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strandS_flat_v3 as V3  # noqa: E402

ALPHA = V3.ALPHA
ETAS = V3.ETAS
RECTS = V3.RECTS


class TableVar:
    """the chain table with a variation: thin (keep every thin-th point), match (use the table only at mu >= match), the fixed-alpha fit
    (beta, gamma, delta, eps, zeta) refitted on [fit_lo, fit_hi] from the kept table points (or the stored S1 fit when refit is None)."""

    def __init__(self, T, thin=1, match=None, refit=None):
        mu = np.array(T["mu"])
        S = np.array(T["S_ch"])
        keep = np.zeros(len(mu), bool)
        keep[::thin] = True
        keep[-1] = True
        if match is not None:
            keep &= mu >= match * (1 - 1e-12)
        self.mu, self.S = mu[keep], S[keep]
        self.mu_min, self.mu_max = float(self.mu.min()), float(self.mu.max())
        if refit is None:
            fx = T["fit_fixed_alpha"]
            self.c = [fx[k] for k in ("beta", "gamma", "delta", "eps", "zeta")]
            self.fit_resid = fx["max_resid"]
        else:
            lo, hi = refit
            sel = (mu >= lo * (1 - 1e-12)) & (mu <= hi * (1 + 1e-12))
            m = mu[sel]
            lm = np.log(m)
            X = np.stack([np.ones_like(m), m * m * lm, m * m, m ** 4 * lm, m ** 4], 1)
            y = S[sel] + ALPHA * lm
            c, *_ = np.linalg.lstsq(X, y, rcond=None)
            self.c = [float(v) for v in c]
            self.fit_resid = float(np.max(np.abs(X @ c - y)))
        lm = np.log(self.mu)
        self.spl = CubicSpline(lm, self.S + ALPHA * lm)
        self.junction = float(self.fit(self.mu_min) - self.S[0])

    def fit(self, m):
        b, g, d, e, z = self.c
        lm = np.log(m)
        return -ALPHA * lm + b + g * m * m * lm + d * m * m + e * m ** 4 * lm + z * m ** 4

    def __call__(self, mu):
        mu = np.asarray(mu, float)
        out = np.empty_like(mu)
        lo = mu < self.mu_min
        hi = ~lo
        out[hi] = self.spl(np.log(mu[hi])) - ALPHA * np.log(mu[hi])
        out[lo] = self.fit(np.maximum(mu[lo], 1e-300))
        return out


def shares(S, qs, d2, W, rho, mu_min):
    g = np.zeros_like(d2)
    fit_any = np.zeros(d2.shape, bool)
    fit_all = np.ones(d2.shape, bool)
    per = []
    for e, q in zip(ETAS, qs):
        mu = np.sqrt(q * q + d2)
        s = S(mu)
        per.append(float(np.sum(W * s)))
        g += e * s
        fit_any |= mu < mu_min
        fit_all &= mu < mu_min
    G = float(np.sum(W * g))
    return per, G, {"inner_disc": float(np.sum(W[rho < mu_min] * g[rho < mu_min])) / G, "fit_any": float(np.sum(W[fit_any] * g[fit_any])) / G,
                    "fit_all": float(np.sum(W[fit_all] * g[fit_all])) / G}


def run_variant(T, var, aMs, grid, k0):
    S = TableVar(T, **var)
    d2, W, rho = grid
    rows = []
    for rect, x in RECTS.items():
        j = V3.jval(x)
        for aM in aMs:
            qs = aM * np.sqrt(x)
            per, G, sh = shares(S, qs, d2, W, rho, S.mu_min)
            rows.append({"rect": rect, "aM": aM, "eta_per_packet": per, "S_gr": G, "c_S": G / (aM * aM * j), "R_gr": G / (0.5 * math.pi * k0[(rect, aM)]),
                         "shares": sh})
    ext = {}
    for rect in RECTS:
        rr = [r for r in rows if r["rect"] == rect]
        ext[rect] = {q: {fm: V3.efit([r["aM"] for r in rr], [r[q] for r in rr], fm)["limit"] for fm in (0.1, 0.05, 0.025)} for q in ("c_S", "R_gr")}
    return {"rows": rows, "ext": ext, "junction": S.junction, "fit_resid": S.fit_resid, "mu_min": S.mu_min, "n_table": int(len(S.mu))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--aM", default="1.6,0.8,0.4,0.2,0.1,0.05,0.025,0.0125,0.00625,0.005,0.0025")
    ap.add_argument("--rho-min", type=float, default=1e-12)
    ap.add_argument("--npan", type=int, default=25)
    ap.add_argument("--ngl", type=int, default=8)
    ap.add_argument("--nphi", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    t0 = time.time()
    T = json.load(open(a.table))
    aMs = [float(v) for v in a.aM.split(",")]
    with Pool(a.workers) as pool:
        k0 = {(r, m): r1 for r, m, r1, e1, r2 in pool.map(V3.k0_job, [(r, m) for r in RECTS for m in aMs], chunksize=1)}
    # panels per decade kept as version 3's (25 panels over [1e-6, edge]) when rho_min is lowered: npan scaled with the decades
    dec3 = math.log10((math.pi / 2) / 1e-6)
    npan = int(round(a.npan * math.log10((math.pi / 2) / a.rho_min) / dec3))
    grid = V3.zone_points(a.rho_min, npan, a.ngl, a.nphi)
    grid6 = V3.zone_points(1e-6, a.npan, a.ngl, a.nphi)
    variants = {"base": {}, "rho_min 1e-6 (version 3)": None, "table thinned x2": {"thin": 2}, "fit on [mu_min, 0.03]": {"refit": (0.006, 0.03)},
                "fit on [mu_min, 0.1]": {"refit": (0.006, 0.1)}, "match at 0.01 (fit [0.01, 0.05])": {"match": 0.01, "refit": (0.01, 0.05)},
                "match at 0.02 (fit [0.02, 0.08])": {"match": 0.02, "refit": (0.02, 0.08)}}
    res = {}
    for name, var in variants.items():
        if var is None:
            res[name] = run_variant(T, {}, aMs, grid6, k0)
        else:
            res[name] = run_variant(T, var, aMs, grid, k0)
        print("%s: light c_S -> %.10f, R_gr -> %.10f; r1223 c_S -> %.10f (%.0f s)" % (
            name, res[name]["ext"]["light"]["c_S"][0.1], res[name]["ext"]["light"]["R_gr"][0.1], res[name]["ext"]["r1223"]["c_S"][0.1], time.time() - t0), flush=True)
    sysd = {}
    for rect in RECTS:
        for q in ("c_S", "R_gr"):
            lims = [v["ext"][rect][q][fm] for k, v in res.items() if k != "rho_min 1e-6 (version 3)" for fm in (0.1, 0.05, 0.025)]
            sysd["%s_%s" % (rect, q)] = {"min": min(lims), "max": max(lims), "spread": max(lims) - min(lims), "base_0.1": res["base"]["ext"][rect][q][0.1]}
    out = {"meta": dict(vars(a), npan_used=npan), "variants": {k: {kk: (vv if kk != "rows" else vv) for kk, vv in v.items()} for k, v in res.items()},
           "systematic": sysd, "targets": {"one_over_3pi": 1 / (3 * math.pi), "two_ninths": 2 / 9}, "seconds": time.time() - t0}
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)

    def f(v, d=6):
        if v is None:
            return "-"
        v = float(v)
        return "%.*g" % (d, v) if (v == 0 or 1e-3 <= abs(v) < 1e5) else "%.6e" % v
    base = res["base"]
    L = ["## Revision 1b: the red team's controls (strandS_flat_v4.py; reviewb/review_output/reviewb_azure_redteam_strandS_20261014_213102)", "",
         "- The radial panels start at rho_min = %s (%d panels; version 3 started at 1e-6 and omitted the disc below it, the red team's sec. 3); the "
         "row values with every packet column to 13 digits, the three shares (the inner disc rho < mu_min; at least one packet evaluated by the fit; "
         "all four packets by the fit):" % (f(a.rho_min, 3), npan), "",
         "| rectangle | aM | eta_100(q_s) per packet | hat S_gr | c_S | R_gr | inner disc | fit (any packet) | fit (all packets) | hat S_gr (rho_min 1e-6) / hat S_gr - 1 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    v3rows = {(r["rect"], r["aM"]): r for r in res["rho_min 1e-6 (version 3)"]["rows"]}
    for r in base["rows"]:
        L.append("| %s | %s | %s | %.13g | %.12f | %.12f | %s | %s | %s | %s |" % (
            r["rect"], f(r["aM"]), ", ".join("%.13f" % v for v in r["eta_per_packet"]), r["S_gr"], r["c_S"], r["R_gr"], f(r["shares"]["inner_disc"], 4),
            f(r["shares"]["fit_any"], 4), f(r["shares"]["fit_all"], 4), f(v3rows[(r["rect"], r["aM"])]["S_gr"] / r["S_gr"] - 1, 3)))
    L += ["", "- The extrapolated limits per variation (the fit X_0 + c (aM)^2 log(1/aM) + c' (aM)^2 on aM <= 0.1, 0.05, 0.025), with each "
          "variation's table size, junction and fit residual:", "",
          "| variation | table points | junction | fit residual | light c_S (0.1, 0.05, 0.025) | light R_gr | r1223 c_S | r1223 R_gr |", "|---|---|---|---|---|---|---|---|"]
    for name, v in res.items():
        L.append("| %s | %d | %s | %s | %s | %s | %s | %s |" % (name, v["n_table"], f(v["junction"], 3), f(v["fit_resid"], 3),
                                                            ", ".join("%.10f" % v["ext"]["light"]["c_S"][fm] for fm in (0.1, 0.05, 0.025)),
                                                            ", ".join("%.10f" % v["ext"]["light"]["R_gr"][fm] for fm in (0.1, 0.05, 0.025)),
                                                            ", ".join("%.10f" % v["ext"]["r1223"]["c_S"][fm] for fm in (0.1, 0.05, 0.025)),
                                                            ", ".join("%.10f" % v["ext"]["r1223"]["R_gr"][fm] for fm in (0.1, 0.05, 0.025))))
    L += ["", "- The systematic spread over the variations with rho_min = %s (every fit range): " % f(a.rho_min, 3) + "; ".join(
        "%s: [%.10f, %.10f], spread %s (%s relative)" % (k, v["min"], v["max"], f(v["spread"], 3), f(v["spread"] / (v["base_0.1"]), 3)) for k, v in sysd.items()) +
          ". Targets: 1/(3 pi) = %.10f, 2/9 = %.10f. Every number above is emitted by the script (%.0f s)." % (1 / (3 * math.pi), 2 / 9, out["seconds"])]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
