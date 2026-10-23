#!/usr/bin/env python
"""Strand S job S3, version 6 (builder analyst): the per-ray inner form, by re-assembly of a version 3 or version 5 run.

Why (builder, labelled): versions 3 and 5 evaluate the inner disc's packets (gap below g_lim) with one inner form, the phi = 0 ladder's fit,
on every ray; that assumes the chain entropy depends on the transverse momentum q only through the gap.  For tilted normals the two ladders
of the record disagree at the g^2 log g order (the (2,2,1) record: gamma -0.2207 at phi = 0 against -0.1419 at phi = pi/4), the order the
graded sum keeps (sum_s eta_s m_s^2 = 0 cancels the g^2 term, not the g^2 log g term).  Version 6 fits a ladder on every quadrature ray (the
run's own phi nodes) and evaluates the inner form on each ray with that ray's coefficients; the outer points and the disc's direct massive
points are read from the run's saved per-point values (<run>_points.npz), unchanged.  The check: re-assembled with the phi = 0 coefficients on
every ray, the rows of the original run are reproduced (printed).

Args: --from-json (comma list: the runs of one normal, e.g. version 3's base and version 5's extension) --from-npz (their _points.npz, same order) --lad-n --workers --out-json.  The ladders are computed once per normal and used for every run given."""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from numpy.polynomial.legendre import leggauss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strandS_sphere_v5 as SV  # noqa: E402


def build_rays(a, g_lim):
    xg, wg = leggauss(a["ngl"])
    xp, wp = leggauss(a["nphi"] // 2)
    nodes = []
    for lo, hi in ((-math.pi / 4, math.pi / 4), (math.pi / 4, 3 * math.pi / 4)):
        for x, w in zip(xp, wp):
            nodes.append((0.5 * (hi - lo) * x + 0.5 * (hi + lo), 0.5 * (hi - lo) * w))

    def gl_panels(lo, hi, npan):
        e = np.geomspace(lo, hi, npan + 1)
        return ((0.5 * (e[1:] - e[:-1])[:, None] * xg[None, :] + 0.5 * (e[1:] + e[:-1])[:, None]).ravel(),
                (0.5 * (e[1:] - e[:-1])[:, None] * wg[None, :]).ravel())
    rays = []
    for phi, wphi in nodes:
        rmax = (math.pi / 2) / max(abs(math.cos(phi)), abs(math.sin(phi)))
        rho_lim = SV.ray_rho_for_gap(phi, g_lim, rmax)
        r_in, w_in = gl_panels(a["rho_min"], rho_lim, a["npan_in"])
        r_out, w_out = gl_panels(rho_lim, rmax, a["npan_out"])
        r = np.concatenate([r_in, r_out])
        wr = np.concatenate([w_in, w_out])
        region = np.concatenate([np.zeros(len(r_in), int), np.ones(len(r_out), int)])
        g0s = np.array([SV.gap0((rr * math.cos(phi), rr * math.sin(phi))) for rr in r])
        xd, wd = leggauss(a["disc_nr"])
        rays.append({"phi": phi, "wphi": wphi, "r": r, "wr": wr, "g0": g0s, "region": region, "rho_lim": rho_lim,
                     "disc_r": 0.5 * rho_lim * (xd + 1), "disc_w": 0.5 * rho_lim * wd})
    return rays


def assemble(rays, coefs, outer, disc, allm, massive, g_lim, pref, aMs):
    idx = {round(m, 15): i for i, m in enumerate(allm)}
    eta = {m: 0.0 for m in allm}
    eta_inner = {m: 0.0 for m in allm}
    for ir, ray in enumerate(rays):
        cin = coefs[ir]
        for ip, (rr, wr, g0, rg) in enumerate(zip(ray["r"], ray["wr"], ray["g0"], ray["region"])):
            w = pref * ray["wphi"] * wr * rr
            if rg == 1:
                vals = outer[(ir, ip)]
                for m in allm:
                    eta[m] += w * vals[idx[round(m, 15)]]
            else:
                for m in allm:
                    if m < g_lim:
                        v = w * float(SV.inner_S(math.sqrt(m * m + g0 * g0), cin))
                        eta[m] += v
                        eta_inner[m] += v
        big = [m for m in massive if m >= g_lim]
        if big:
            for idd, (rr, wd) in enumerate(zip(ray["disc_r"], ray["disc_w"])):
                w = pref * ray["wphi"] * wd * rr
                vals = disc[(ir, idd)]
                for jm, m in enumerate(big):
                    eta[m] += w * vals[jm]
                    eta_inner[m] += w * vals[jm]
    rows = []
    for rect, x in SV.RECTS.items():
        j = float(sum(e * (v * math.log(v) if v > 0 else 0.0) for e, v in zip(SV.ETAS, x)))
        for aM in aMs:
            ms = [round(aM * math.sqrt(v), 15) if v > 0 else 0.0 for v in x]
            per = [eta[m] for m in ms]
            Sgr = float(SV.ETAS @ np.array(per))
            inner = float(SV.ETAS @ np.array([eta_inner[m] for m in ms]))
            rows.append({"rect": rect, "aM": aM, "j": j, "eta_per_packet": per, "eta_star": float(sum(per)), "S_gr": Sgr, "c_S": Sgr / (aM * aM * j),
                         "inner_share": inner / Sgr if Sgr else None})
    return rows, eta[0.0]


def load_run(jpath, npath):
    D = json.load(open(jpath))
    m = D["meta"]
    aMs = [float(v) for v in m["aM"].split(",")]
    massive = sorted({round(aM * math.sqrt(x), 15) for aM in aMs for r in SV.RECTS.values() for x in r if x > 0})
    allm = [0.0] + massive
    Z = np.load(npath)
    assert np.allclose(Z["masses"], np.array(allm)), "the npz's masses differ from the run's"
    outer = {tuple(int(v) for v in k): Z["outer_vals"][i] for i, k in enumerate(Z["outer_keys"])}
    disc = {tuple(int(v) for v in k): Z["disc_vals"][i] for i, k in enumerate(Z["disc_keys"])} if len(Z["disc_keys"]) else {}
    return D, m, aMs, massive, allm, outer, disc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-json", required=True, help="comma list: runs of one normal with the same quadrature (e.g. the base and the extension)")
    ap.add_argument("--from-npz", required=True, help="comma list, the runs' _points.npz in the same order")
    ap.add_argument("--lad-n", type=int, default=32)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--reuse-ladder", default=None, help="a previous version-6 JSON of the same normal: its per-ray ladder points are reused")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    t0 = time.time()
    runs = [load_run(j_, n_) for j_, n_ in zip(a.from_json.split(","), a.from_npz.split(","))]
    D0, m0 = runs[0][0], runs[0][1]
    keys = ("normal", "muL", "Lcap", "pnorm_rule", "nphi", "ngl", "npan_in", "npan_out", "rho_min", "disc_nr", "lad_lo", "lad_hi")
    for r_ in runs[1:]:
        assert all(r_[1].get(k) == m0.get(k) for k in keys), "the runs' quadratures differ"
    p = tuple(int(v) for v in m0["normal"].split(","))
    prule = bool(m0.get("pnorm_rule", False))
    SV.init(p, m0["muL"], m0["Lcap"], prule)
    g_lim = SV.G["muLp"] / m0["Lcap"]
    assert abs(g_lim - D0["g_lim"]) < 1e-12, (g_lim, D0["g_lim"])
    rays = build_rays(m0, g_lim)
    pref = 8.0 / (2 * math.pi) ** 2 / SV.G["norm"]
    gl = np.geomspace(m0["lad_lo"], m0["lad_hi"], a.lad_n)
    if a.reuse_ladder:
        R = json.load(open(a.reuse_ladder))
        assert tuple(R["normal"]) == p and abs(R["g_lim"] - g_lim) < 1e-12, "the reused ladder belongs to another normal or rule"
        lad = [tuple(x) for x in R["ladder"]]
        phis = sorted({x[0] for x in lad})
        assert all(min(abs(ray["phi"] - q) for q in phis) < 1e-12 for ray in rays), "the reused ladder's rays differ"
        lad = [(min(phis, key=lambda q: abs(q - x[0])),) + tuple(x[1:]) for x in lad]
        rays_phi = {min(phis, key=lambda q: abs(q - ray["phi"])): ray for ray in rays}
        for q, ray in rays_phi.items():
            ray["phi"] = q
    else:
        pool = Pool(a.workers, initializer=SV.init, initargs=(p, m0["muL"], m0["Lcap"], prule))
        lad = pool.map(SV.job_ladder, [(ray["phi"], g) for ray in rays for g in gl], chunksize=1)
        pool.close()
    fits = []
    for ray in rays:
        rr = [r for r in lad if r[0] == ray["phi"] and r[2] is not None]
        c, res = SV.fit_inner([r[2] for r in rr], [r[4] for r in rr])
        fits.append({"phi": ray["phi"], "coef": c, "max_resid": res, "points": len(rr)})
    out = {"normal": list(p), "norm": SV.G["norm"], "g_lim": g_lim, "sources": a.from_json.split(","), "ray_fits": fits, "runs": [],
           "ladder": [list(x) for x in lad]}
    rows_all, rows0_all = [], []
    for (D, m, aMs, massive, allm, outer, disc), src in zip(runs, a.from_json.split(",")):
        c0 = D["inner_fits"]["%.6f" % 0.0]["coef"]
        rows0, e0c = assemble(rays, [c0] * len(rays), outer, disc, allm, massive, g_lim, pref, aMs)
        chk = max(abs(r0["S_gr"] / ro["S_gr"] - 1) for r0, ro in zip(rows0, D["rows"]))
        rows, e0 = assemble(rays, [f_["coef"] for f_ in fits], outer, disc, allm, massive, g_lim, pref, aMs)
        out["runs"].append({"source": os.path.basename(src), "aM": aMs, "reassembly_check_max_rel": chk, "eta_massless": e0, "eta_massless_phi0": e0c,
                            "phi0_coef_of_run": c0, "checks": D.get("checks"), **({"massive_check": D["massive_check"]} if "massive_check" in D else {})})
        rows_all += rows
        rows0_all += rows0
        print("%s: re-assembly check (the run's phi = 0 coefficients) max |rel| %.3e" % (os.path.basename(src), chk), flush=True)
    out["rows"] = sorted(rows_all, key=lambda r: (r["rect"], r["aM"]))
    out["rows_phi0"] = sorted(rows0_all, key=lambda r: (r["rect"], r["aM"]))
    out["eta_massless"] = out["runs"][0]["eta_massless"]
    out["checks"] = out["runs"][0]["checks"]
    out["seconds"] = time.time() - t0
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    print("normal %s: gamma by ray %s (%.0f s)" % (p, ["%.4f" % f_["coef"][1] for f_ in fits], time.time() - t0))
    for r, r0 in zip(out["rows"], out["rows_phi0"]):
        print("%s aM %g: c_S per-ray %.9f (phi = 0 form %.9f) inner share %.3f" % (r["rect"], r["aM"], r["c_S"], r0["c_S"], r["inner_share"]))


if __name__ == "__main__":
    main()
