#!/usr/bin/env python
"""Strand S, job S3 (SPEC_strandS_graded_entropy.md sec. 4 and the lead's draft in sec. 8.2, completed by the builder analyst with the plan
put to the lead in turn 10, labelled): the per-normal graded planar coefficient for one primitive normal p, for the spherical cut's
tilted-plane average.

The channel.  For p primitive with the paper's unimodular basis (lattice_entropy.basis: k = q A[:2] + theta p), the transverse momentum q runs
over [0, 2 pi)^2 and the per-boundary channel entropy is S(q; m) = entropy(interval_covariance(p, q, L, 4L, m))/2 (the paper's functions,
unmodified); eta_p(m) = (2 pi)^-2 Int d^2q S(q; m)/|p| (eta.py's normalization).  The 8 nodes project onto the 4 corners {0, pi}^2, two per
corner; the massless channel's gap is gap_0(q) = min_theta (sum_i sin^2 k_i)^(1/2) (a grid minimum refined by Brent), the massive channel's
sqrt(m^2 + gap_0^2).

The corners and the inversion (builder, labelled).  The pi-shifts of k are site-local unitaries (the staggering (-1)^{x_i} times a Clifford
element that flips gamma^i), which map the channel at corner c + dq to the channel at corner c' + dq, so the 4 corners carry the same function
of dq; time reversal maps dq to -dq.  Both are checked pointwise (--checks random dq at m = 0.1: the largest spread over the 4 corners and
under dq -> -dq), and the zone integral is 4 x the corner (0, 0)'s square cell [-pi/2, pi/2)^2, halved by the inversion: phi in [-pi/4, 3pi/4)
in two sectors (the cell's kinks at pi/4), Gauss-Legendre with --nphi/2 points each.  Radial: --npan panels log-spaced from --rho-min to the
cell edge, --ngl points each.

The inner form (sec. 4 (i), builder's cap, labelled).  A single-packet ladder along phi = 0 (and phi = pi/4 as the anisotropy check): gap_0
from --lad-lo to --lad-hi (--lad-n points), L = max(64, ceil(muL/gap_0)), fitted with alpha = 2/3 fixed by beta + gamma g^2 log g + delta g^2
+ eps g^4 log g + zeta g^4 (g the gap).  Inside the disc gap_0 < g_lim = muL/Lcap the packets with gap below g_lim use the inner form
-alpha log gap + beta + gamma gap^2 log gap + delta gap^2 + eps gap^4 log gap + zeta gap^4 on the fine grid; the packets with m >= g_lim are
computed directly on a coarse Gauss rule over the disc (--disc-nr radial points per ray on [0, rho_lim(phi)], the same angular nodes; the
integrand is smooth there).  Outside the disc every packet is computed directly at a common L = min(Lcap, max(64, ceil(muL/gap_0))) (the
graded sum per point).  The masses: the massless packet and, per spacing aM, the light rectangle's aM/sqrt 2, aM and r1223's aM/sqrt 3,
aM sqrt(2/3).

Version 2 (the lead's rulings, SPEC sec. 9.5): the ladder over gap_0 in [0.015, 0.3] with 32 points; the two-gap check: at --checks random
delta q per corner both local minima of sum_i sin^2 k_i(theta) along the chain line (the two Dirac points; sum sin^2 is pi-periodic in theta
for an integer normal, so they are expected equal) and the largest |g_1 - g_2|, the single-gap inner form being used only if it is at the
rounding level; the per-point outer and disc values saved (<out-json>.npz) so that a change of the inner form re-assembles without new chains.
Version 3: the radial panels are aligned with the disc boundary on every ray: --npan-in log panels on [rho_min, rho_lim(phi)] (the inner form
for the packets whose gap can fall below g_lim) and --npan-out log panels on [rho_lim(phi), the cell edge] (direct chains for every packet),
with the coarse Gauss rule on [0, rho_lim(phi)] for the packets with m >= g_lim; versions 1 and 2 split the region at rho_lim inside a panel,
so the outer points of the straddling panel carried the whole panel's weights for part of it (the [100] cross-check against S2 failed at the
1e-4 level for the massive packets, version 1's record sphere_100.json).
Outputs (one JSON per normal): the checks, the ladders and fits, per (rectangle, aM) the four eta_p(q_s), eta_*, hat S_gr, c_S, the inner-disc
share, the counts and timing; --timing N processes N random outer jobs first and prints the projected cost.  CPU workers (--workers)."""
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
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar

ORIENT = os.environ.get("ORIENT_DIR", "<internal>/orient")
sys.path.insert(0, ORIENT)
import lattice_entropy as LE  # noqa: E402

ALPHA = 2.0 / 3.0
ETAS = np.array([1.0, -1.0, -1.0, 1.0])
RECTS = {"light": np.array([0.0, 0.5, 0.5, 1.0]), "r1223": np.array([1.0, 2.0, 2.0, 3.0]) / 3.0}
G = {}


def init(p, muL, Lcap):
    pp, B, A = LE.basis(p)
    G.update(p=tuple(int(v) for v in pp), A=A, norm=float(np.linalg.norm(pp)), muL=muL, Lcap=Lcap)


def gap0(q, nth=4096):
    A, p = G["A"], np.array(G["p"], float)
    k0 = np.asarray(q, float) @ A[:2]
    th = 2 * np.pi * np.arange(nth) / nth
    e2 = np.sum(np.sin(k0[None, :] + th[:, None] * p[None, :]) ** 2, axis=1)
    i = int(np.argmin(e2))
    f = lambda t: float(np.sum(np.sin(k0 + t * p) ** 2))
    r = minimize_scalar(f, bounds=(th[i] - 2 * np.pi / nth, th[i] + 2 * np.pi / nth), method="bounded", options={"xatol": 1e-14})
    return math.sqrt(max(min(r.fun, e2[i]), 0.0))


def two_gaps(q, nth=4096):
    """the two local minima of sum_i sin^2 k_i(theta) along the chain line (the two Dirac points of a corner): the global minimum and the
    minimum on the half-period away from it, each refined by Brent."""
    A, p = G["A"], np.array(G["p"], float)
    k0 = np.asarray(q, float) @ A[:2]
    th = 2 * np.pi * np.arange(nth) / nth
    e2 = np.sum(np.sin(k0[None, :] + th[:, None] * p[None, :]) ** 2, axis=1)
    i = int(np.argmin(e2))
    f = lambda t: float(np.sum(np.sin(k0 + t * p) ** 2))
    g = []
    for c in (th[i], th[i] + np.pi):
        lo, hi = c - 0.25, c + 0.25
        tt = np.linspace(lo, hi, 2001)
        vv = np.array([f(t) for t in tt])
        j = int(np.argmin(vv))
        r = minimize_scalar(f, bounds=(tt[max(j - 1, 0)], tt[min(j + 1, len(tt) - 1)]), method="bounded", options={"xatol": 1e-14})
        g.append(math.sqrt(max(r.fun, 0.0)))
    return g


def job_twogap(dq):
    out = []
    for c in ((0.0, 0.0), (math.pi, 0.0), (0.0, math.pi), (math.pi, math.pi)):
        g1, g2 = two_gaps((c[0] + dq[0], c[1] + dq[1]))
        out.append(abs(g1 - g2) / max(g1, g2, 1e-300))
    return max(out)


def S_channel(q, L, m):
    C = LE.interval_covariance(G["p"], tuple(q), L, 4 * L, mass=float(m))
    return LE.entropy(C, -1) / 2.0


def Lof(gap):
    L = min(G["Lcap"], max(64, int(math.ceil(G["muL"] / gap))))
    return L + (L % 2)


def job_point(args):
    """one quadrature point: S at every mass in the list at a common L set by the smallest gap among them."""
    key, q, masses, g0 = args
    t = time.time()
    gaps = [math.sqrt(m * m + g0 * g0) for m in masses]
    L = Lof(min(gaps))
    return key, [S_channel(q, L, m) for m in masses], L, time.time() - t


def job_check(args):
    dq, m = args
    out = []
    for c in ((0.0, 0.0), (math.pi, 0.0), (0.0, math.pi), (math.pi, math.pi)):
        q = (c[0] + dq[0], c[1] + dq[1])
        out.append(S_channel(q, Lof(math.sqrt(m * m + gap0(q) ** 2)), m))
    qm = (-dq[0], -dq[1])
    out.append(S_channel(qm, Lof(math.sqrt(m * m + gap0(qm) ** 2)), m))
    return dq, out


def ray_rho_for_gap(phi, g, rmax):
    """the radius along phi from corner (0, 0) where gap_0 = g (bisection; gap_0 increases from 0 near the corner)."""
    lo, hi = 0.0, rmax
    if gap0((hi * math.cos(phi), hi * math.sin(phi))) < g:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if gap0((mid * math.cos(phi), mid * math.sin(phi))) < g:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def job_ladder(args):
    phi, g = args
    rmax = (math.pi / 2) / max(abs(math.cos(phi)), abs(math.sin(phi)))
    rho = ray_rho_for_gap(phi, g, rmax)
    if rho is None:
        return phi, g, None, None, None
    q = (rho * math.cos(phi), rho * math.sin(phi))
    gg = gap0(q)
    L = max(64, int(math.ceil(G["muL"] / gg)))
    L += L % 2
    return phi, g, gg, L, S_channel(q, L, 0.0)


def fit_inner(gs, Ss):
    g = np.array(gs)
    y = np.array(Ss) + ALPHA * np.log(g)
    lg = np.log(g)
    X = np.stack([np.ones_like(g), g * g * lg, g * g, g ** 4 * lg, g ** 4], 1)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return [float(v) for v in c], float(np.max(np.abs(X @ c - y)))


def inner_S(gap, c):
    b, gm, d, e, z = c
    lg = np.log(gap)
    return -ALPHA * lg + b + gm * gap * gap * lg + d * gap * gap + e * gap ** 4 * lg + z * gap ** 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", required=True)
    ap.add_argument("--aM", default="0.4,0.2,0.1,0.05")
    ap.add_argument("--npan", type=int, default=25, help="unused from version 3 (see --npan-in, --npan-out)")
    ap.add_argument("--npan-in", type=int, default=20)
    ap.add_argument("--npan-out", type=int, default=8)
    ap.add_argument("--ngl", type=int, default=8)
    ap.add_argument("--nphi", type=int, default=16)
    ap.add_argument("--rho-min", type=float, default=1e-6)
    ap.add_argument("--muL", type=float, default=24.0)
    ap.add_argument("--Lcap", type=int, default=480)
    ap.add_argument("--lad-lo", type=float, default=0.015)
    ap.add_argument("--lad-hi", type=float, default=0.3)
    ap.add_argument("--lad-n", type=int, default=32)
    ap.add_argument("--disc-nr", type=int, default=6)
    ap.add_argument("--checks", type=int, default=32)
    ap.add_argument("--timing", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    t0 = time.time()
    p = tuple(int(v) for v in a.normal.split(","))
    init(p, a.muL, a.Lcap)
    pnorm = G["norm"]
    g_lim = a.muL / a.Lcap
    aMs = [float(v) for v in a.aM.split(",")]
    massive = sorted({round(aM * math.sqrt(x), 15) for aM in aMs for r in RECTS.values() for x in r if x > 0})
    pool = Pool(a.workers, initializer=init, initargs=(p, a.muL, a.Lcap))
    out = {"meta": vars(a), "normal": list(G["p"]), "norm": pnorm, "g_lim": g_lim, "masses": massive}
    # quadrature nodes: half cell, two sectors
    xg, wg = leggauss(a.ngl)
    xp, wp = leggauss(a.nphi // 2)
    nodes = []
    for lo, hi in ((-math.pi / 4, math.pi / 4), (math.pi / 4, 3 * math.pi / 4)):
        for x, w in zip(xp, wp):
            nodes.append((0.5 * (hi - lo) * x + 0.5 * (hi + lo), 0.5 * (hi - lo) * w))
    rays = []

    def gl_panels(lo, hi, npan):
        e = np.geomspace(lo, hi, npan + 1)
        return ((0.5 * (e[1:] - e[:-1])[:, None] * xg[None, :] + 0.5 * (e[1:] + e[:-1])[:, None]).ravel(),
                (0.5 * (e[1:] - e[:-1])[:, None] * wg[None, :]).ravel())
    for phi, wphi in nodes:
        rmax = (math.pi / 2) / max(abs(math.cos(phi)), abs(math.sin(phi)))
        rho_lim = ray_rho_for_gap(phi, g_lim, rmax)
        if rho_lim is None or rho_lim <= a.rho_min:
            raise RuntimeError("the disc boundary is not inside the cell on phi = %g" % phi)
        r_in, w_in = gl_panels(a.rho_min, rho_lim, a.npan_in)
        r_out, w_out = gl_panels(rho_lim, rmax, a.npan_out)
        r = np.concatenate([r_in, r_out])
        wr = np.concatenate([w_in, w_out])
        region = np.concatenate([np.zeros(len(r_in), int), np.ones(len(r_out), int)])
        g0s = np.array([gap0((rr * math.cos(phi), rr * math.sin(phi))) for rr in r])
        rays.append({"phi": phi, "wphi": wphi, "r": r, "wr": wr, "g0": g0s, "region": region, "rho_lim": rho_lim, "rmax": rmax})
    # weight: zone average = 4 corners x 2 (inversion) x (2 pi)^-2 x Int over the half cell, then / |p|
    pref = 8.0 / (2 * math.pi) ** 2 / pnorm
    outer_jobs, disc_jobs = [], []
    for ir, ray in enumerate(rays):
        for ip, (rr, g0, rg) in enumerate(zip(ray["r"], ray["g0"], ray["region"])):
            if rg == 1:
                q = (rr * math.cos(ray["phi"]), rr * math.sin(ray["phi"]))
                outer_jobs.append((("o", ir, ip), q, [0.0] + massive, max(g0, g_lim)))
        if ray["rho_lim"] is not None:
            xd, wd = leggauss(a.disc_nr)
            rd = 0.5 * ray["rho_lim"] * (xd + 1)
            wdd = 0.5 * ray["rho_lim"] * wd
            ray["disc_r"], ray["disc_w"] = rd, wdd
            big = [m for m in massive if m >= g_lim]
            for idd, rr in enumerate(rd):
                q = (rr * math.cos(ray["phi"]), rr * math.sin(ray["phi"]))
                if big:
                    disc_jobs.append((("d", ir, idd), q, big, gap0(q)))
    out["counts"] = {"rays": len(rays), "outer_points": len(outer_jobs), "disc_points": len(disc_jobs),
                     "inner_fine_points": int(sum(int(np.sum(r["region"] == 0)) for r in rays))}
    print("normal %s: %d outer points x %d masses, %d disc points; g_lim %.4f (%.0f s)" % (p, len(outer_jobs), 1 + len(massive), len(disc_jobs), g_lim, time.time() - t0), flush=True)
    if a.timing:
        rng = np.random.default_rng(7)
        sel = [outer_jobs[i] for i in rng.choice(len(outer_jobs), min(a.timing, len(outer_jobs)), replace=False)]
        res = pool.map(job_point, sel, chunksize=1)
        secs = np.array([r[3] for r in res])
        Ls = np.array([r[2] for r in res])
        allL = np.array([Lof(min(math.sqrt(m * m + j[3] ** 2) for m in j[2])) for j in outer_jobs])
        est = float(np.mean(secs / Ls.astype(float) ** 3) * np.sum(allL.astype(float) ** 3))
        out["timing"] = {"sample": len(sel), "mean_s": float(secs.mean()), "projected_outer_cpu_s_cubic": est, "L_hist": np.histogram(allL, bins=[0, 100, 200, 300, 400, 500])[0].tolist()}
        print("timing: %s" % json.dumps(out["timing"]), flush=True)
        json.dump(out, open(a.out_json, "w"), indent=1)
        return
    # checks and ladders
    rng = np.random.default_rng(11)
    dqs = [(float(r * math.cos(t)), float(r * math.sin(t))) for r, t in zip(rng.uniform(0.05, 1.0, a.checks), rng.uniform(0, 2 * math.pi, a.checks))]
    chk = pool.map(job_check, [(dq, 0.1) for dq in dqs], chunksize=1)
    out["checks"] = {"corner_spread_max": max(max(v[:4]) - min(v[:4]) for _, v in chk), "inversion_max": max(abs(v[4] - v[0]) for _, v in chk)}
    dqs2 = [(float(r * math.cos(t)), float(r * math.sin(t))) for r, t in zip(np.geomspace(0.01, 0.5, a.checks), rng.uniform(0, 2 * math.pi, a.checks))]
    out["checks"]["two_gap_rel_max"] = float(max(pool.map(job_twogap, dqs2, chunksize=1)))
    print("checks: %s (%.0f s)" % (json.dumps(out["checks"]), time.time() - t0), flush=True)
    gl = np.geomspace(a.lad_lo, a.lad_hi, a.lad_n)
    lad = pool.map(job_ladder, [(phi, g) for phi in (0.0, math.pi / 4) for g in gl], chunksize=1)
    fits = {}
    for phi in (0.0, math.pi / 4):
        rows = [r for r in lad if r[0] == phi and r[2] is not None]
        c, res = fit_inner([r[2] for r in rows], [r[4] for r in rows])
        fits["%.6f" % phi] = {"coef": c, "max_resid": res, "points": len(rows), "L_max": max(r[3] for r in rows)}
    out["ladder"] = [{"phi": r[0], "g_target": r[1], "g": r[2], "L": r[3], "S": r[4]} for r in lad]
    out["inner_fits"] = fits
    cin = fits["%.6f" % 0.0]["coef"]
    print("ladder fits: %s (%.0f s)" % (json.dumps({k: (v["coef"][:3], v["max_resid"]) for k, v in fits.items()}), time.time() - t0), flush=True)
    # the quadrature
    res_o = {r[0]: r for r in pool.map(job_point, outer_jobs, chunksize=1)}
    print("outer points done (%.0f s)" % (time.time() - t0), flush=True)
    res_d = {r[0]: r for r in pool.map(job_point, disc_jobs, chunksize=1)} if disc_jobs else {}
    print("disc points done (%.0f s)" % (time.time() - t0), flush=True)
    pool.close()
    allm = [0.0] + massive
    idx = {round(m, 15): i for i, m in enumerate(allm)}
    eta = {m: 0.0 for m in allm}
    eta_inner = {m: 0.0 for m in allm}
    for ir, ray in enumerate(rays):
        for ip, (rr, wr, g0, rg) in enumerate(zip(ray["r"], ray["wr"], ray["g0"], ray["region"])):
            w = pref * ray["wphi"] * wr * rr
            if rg == 1:
                vals = res_o[("o", ir, ip)][1]
                for m in allm:
                    eta[m] += w * vals[idx[round(m, 15)]]
            else:
                for m in allm:
                    if m < g_lim:
                        v = w * float(inner_S(math.sqrt(m * m + g0 * g0), cin))
                        eta[m] += v
                        eta_inner[m] += v
        if ray.get("disc_r") is not None:
            big = [m for m in massive if m >= g_lim]
            for idd, (rr, wd) in enumerate(zip(ray["disc_r"], ray["disc_w"])):
                w = pref * ray["wphi"] * wd * rr
                vals = res_d[("d", ir, idd)][1]
                for jm, m in enumerate(big):
                    eta[m] += w * vals[jm]
                    eta_inner[m] += w * vals[jm]
    rows = []
    for rect, x in RECTS.items():
        j = float(sum(e * (v * math.log(v) if v > 0 else 0.0) for e, v in zip(ETAS, x)))
        for aM in aMs:
            ms = [round(aM * math.sqrt(v), 15) if v > 0 else 0.0 for v in x]
            per = [eta[m] for m in ms]
            Sgr = float(ETAS @ np.array(per))
            inner = float(ETAS @ np.array([eta_inner[m] for m in ms]))
            rows.append({"rect": rect, "aM": aM, "j": j, "eta_per_packet": per, "eta_star": float(sum(per)), "S_gr": Sgr,
                         "c_S": Sgr / (aM * aM * j), "inner_share": inner / Sgr if Sgr else None})
            print("%s aM %g: eta %s S_gr %.6e c_S %.6f inner %.3f" % (rect, aM, ["%.8f" % v for v in per], Sgr, Sgr / (aM * aM * j), inner / Sgr), flush=True)
    np.savez_compressed(a.out_json.replace(".json", "_points.npz"),
                        outer_keys=np.array([k_[1:] for k_ in res_o]), outer_vals=np.array([v_[1] for v_ in res_o.values()]),
                        outer_L=np.array([v_[2] for v_ in res_o.values()]),
                        disc_keys=np.array([k_[1:] for k_ in res_d]) if res_d else np.zeros((0, 2)),
                        disc_vals=np.array([v_[1] for v_ in res_d.values()]) if res_d else np.zeros((0, 0)), masses=np.array(allm))
    out["eta_massless"] = eta[0.0]
    out["rows"] = rows
    out["seconds"] = time.time() - t0
    out["outer_cpu_s"] = float(sum(r[3] for r in res_o.values()))
    json.dump(out, open(a.out_json, "w"), indent=1)
    print("done %.0f s (outer cpu %.0f s)" % (out["seconds"], out["outer_cpu_s"]))


if __name__ == "__main__":
    main()
