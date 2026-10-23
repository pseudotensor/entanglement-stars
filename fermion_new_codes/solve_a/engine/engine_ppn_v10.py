#!/usr/bin/env python
"""engine_ppn_v10.py (builder analyst; T7.1 version 10 per SPEC_final4b_engine.md sec. 17.2; the symbols, the solved exterior, the matched-energy
launch and the deflection of engine_ppn_v9 imported unchanged).

(1) Rays: per direction (the three plane normals, theta0 in --theta0) and b/r_g in --b-over-rg, the coefficient C(x, lambda) =
(deflection lattice/geodesic - 1)/lambda at a omega in {0.05, 0.025, 0.0125} (lambda = (a omega)^2), and per (direction, b) the straight-line
extrapolation in lambda to lambda = 0 (C(x, 0) and the slope) with the three-point curvature printed as its check; the convention R_far/b = 2000
stated per row (the anisotropic optical tail's logarithmic displacement, -4 t_31 (r_g/b) ln(R_2/R_1) per the red team, is not removed: t_31 is
printed per row); the fit C_0 + C_1/2 sqrt(r_g/b) + C_1 r_g/b of version 9 repeated on the lambda = 0 values.
(2) The [011] periastron at e = 0.3: m a in {0.1, 0.05, 0.025}, A/r_g in {160, 640, 2560}, the basis e1 = (1, 0, 0), e2 = (0, 1, -1)/sqrt2, the
angular momentum along +(0, 1, 1) (sense +1) or -(0, 1, 1) (sense -1), the apoapsis at the signed angle theta0 from e1 toward e2 (theta0 in
{-pi/4, 0, pi/8, pi/4}); the lattice and the geodesic orbits from the same phase-space point; the estimator: the periapsis events of r.rdot
(the integrator's event root on its dense output), the in-plane angle of each periapsis, every advance listed, the mean over windows of 1, 3
and 7 periods, the coefficient (advance lattice/geodesic - 1)/(m a)^2 per window; the tolerance ladder (rtol 1e-10 and 1e-12, max_step A/20 and
A/40) with the coefficient's spread over it as the paired difference's error; r_min/r_collar per row; T7g's leading closed form
C_011(theta0, e) beside every row (sense +1; the sine terms change sign with the sense).
Args: --star --member --normals --b-over-rg --aomegas --theta0 --Rfar-over-b --ma --A-over-rg --theta0-orbit --e --skip-rays --skip-orbits --out --tag."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys

import numpy as np
from scipy.integrate import quad, solve_ivp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_ppn_v9 as P9  # noqa: E402


def S_k(e, k):
    return quad(lambda f: math.sqrt(1 + e * math.cos(f)) * math.cos(k * f), 0, 2 * math.pi)[0] / (2 * math.pi)


def C011(theta0, e):
    """T7g's leading closed form (reviewb/review_output/reviewb_azure_redteam_T7g_..., sense +1)."""
    q = math.sqrt(1 - e * e)
    h = 1e-5

    def Q(k):
        dS = (S_k(e + h, k) - S_k(e - h, k)) / (2 * h)
        return S_k(e, k) + q * q / e * dS
    return (-0.75 + q * q / (4 * (1 + q) ** 2) * math.cos(2 * theta0) + 3 * q * q * (1 - q) / (8 * (1 + q) ** 3) * math.cos(4 * theta0)
            + math.sqrt(2) * Q(2) * math.sin(2 * theta0) + 1.5 * math.sqrt(2) * Q(4) * math.sin(4 * theta0))


def orbit_start_sense(F, m, a_orb, e, theta0, nrm, e1, sense):
    y = P9.orbit_start(F, m, a_orb, e, theta0, nrm, e1)
    if sense < 0:
        X = y[:3]; p = y[3:]
        rh = X / np.linalg.norm(X)
        pr = float(p @ rh)
        pt = p - pr * rh
        y = np.concatenate([X, -pt + pr * rh])           # the tangential momentum reversed, the radial (zero-radial-velocity) part kept
    return y


def periapsis_series(F, m, a_orb, e, theta0, nrm, e1, lattice, sense, norbits, rtol, max_step_frac):
    y0 = orbit_start_sense(F, m, a_orb, e, theta0, nrm, e1, sense)
    e2 = np.cross(nrm, e1)
    rhs = P9.make_rhs(F, m, lattice)

    def rdot(t, y):
        return float(y[:3] @ rhs(t, y)[:3])
    rdot.direction = 1
    Tk = 2 * math.pi * math.sqrt(a_orb ** 3 / (F.g * F.E)) * (norbits + 1.2)
    sol = solve_ivp(rhs, (0, Tk), y0, events=rdot, rtol=rtol, atol=rtol * 1e-2, method="DOP853", max_step=a_orb * max_step_frac)
    ye = sol.y_events[0]
    ang = np.unwrap(np.arctan2(ye[:, :3] @ e2, ye[:, :3] @ e1)) if len(ye) else np.array([])
    adv = np.diff(ang) if len(ang) > 1 else np.array([])
    rmin = float(np.min(np.linalg.norm(sol.y[:3], axis=0)))
    return {"advances": adv.tolist(), "n": int(len(adv)), "r_min": rmin, "out_of_plane_max": float(np.max(np.abs(sol.y[:3].T @ nrm))),
            "H_drift": abs(P9.H_of(F, sol.y[:, -1], m, lattice) - P9.H_of(F, y0, m, lattice))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--star", required=True)
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--normals", default="001,011,111")
    ap.add_argument("--b-over-rg", type=float, nargs="+", default=[100.0, 200.0, 400.0, 800.0])
    ap.add_argument("--aomegas", type=float, nargs="+", default=[0.05, 0.025, 0.0125])
    ap.add_argument("--theta0", type=float, nargs="+", default=[0.0, math.pi / 8, math.pi / 4])
    ap.add_argument("--Rfar-over-b", type=float, default=2000.0)
    ap.add_argument("--ma", type=float, nargs="+", default=[0.1, 0.05, 0.025])
    ap.add_argument("--A-over-rg", type=float, nargs="+", default=[160.0, 640.0, 2560.0])
    ap.add_argument("--theta0-orbit", type=float, nargs="+", default=[-math.pi / 4, 0.0, math.pi / 8, math.pi / 4])
    ap.add_argument("--e", type=float, default=0.3)
    ap.add_argument("--norbits", type=int, default=7)
    ap.add_argument("--skip-rays", action="store_true")
    ap.add_argument("--skip-orbits", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    D = json.load(open(a.star))
    M = D["members"][a.member]
    R = np.arange(M["nsites"], dtype=float) + (0.5 if D["args"].get("half_offset") else 1.0)
    g, E = M["g"], M["E"]
    F = P9.Fields(R, np.array(M["N"]), np.array(M["v"]), E, g, M["r_collar"])
    rg = 2 * g * E
    Iq = P9.I_quad()
    rec = {"script": "engine_ppn_v10.py", "star": os.path.basename(a.star), "member": a.member, "g": g, "E": E, "r_g": rg, "r_collar": M["r_collar"], "I": Iq,
           "rays": [], "ray_extrapolations": [], "fits": [], "orbits_011": []}
    L = ["# T7.1 version 10 (engine_ppn_v10.py; member %d of %s; r_g %.4f; the collar %.1f; I = %.6f)" % (a.member, os.path.basename(a.star), rg, M["r_collar"], Iq), ""]
    os.makedirs(a.out, exist_ok=True)
    jpath = os.path.join(a.out, "ppn10_%s.json" % a.tag)
    if not a.skip_rays:
        L += ["## Rays: C = (deflection lattice/geodesic - 1)/(a omega)^2 at three a omega, extrapolated linearly in (a omega)^2 to 0 (R_far/b = %g per row)" % a.Rfar_over_b, "",
              "| normal | theta0 | b/r_g | t_31 | C at a omega %s | C(lambda = 0) | slope in lambda | three-point curvature check | H drift (lat, max) |" % ", ".join("%g" % w for w in a.aomegas),
              "|---|---|---|---|---|---|---|---|---|"]
        for ns in a.normals.split(","):
            nrm, e1 = P9.PLANES[ns]
            for th in a.theta0:
                e2 = np.cross(nrm, e1)
                d = math.cos(th) * e1 + math.sin(th) * e2
                h = -math.sin(th) * e1 + math.cos(th) * e2
                c4 = float(np.sum(d ** 4)); t22 = float(np.sum(d ** 2 * h ** 2)); t31 = float(np.sum(d ** 3 * h)); t13 = float(np.sum(d * h ** 3))
                C0 = -c4 / 2 - 2 * t22
                Chalf = -(Iq / 2) * (4 * t31 + t13)
                bs, c0s = [], []
                for bo in a.b_over_rg:
                    cs, drift = [], 0.0
                    for aw in a.aomegas:
                        dl = P9.deflection8(F, bo * rg, aw, th, nrm, e1, True, a.Rfar_over_b)
                        dg = P9.deflection8(F, bo * rg, aw, th, nrm, e1, False, a.Rfar_over_b)
                        coef = (dl["deflection"] / dg["deflection"] - 1) / aw ** 2
                        cs.append(coef); drift = max(drift, dl["H_drift"])
                        rec["rays"].append({"normal": ns, "theta0": th, "b_over_rg": bo, "aomega": aw, "coefficient": coef, "Rfar_over_b": a.Rfar_over_b,
                                            "lattice": dl, "geodesic": dg})
                    lam = np.array([w * w for w in a.aomegas]); cv = np.array(cs)
                    sl, ic = np.polyfit(lam, cv, 1)
                    curv = float(np.polyfit(lam, cv, 2)[0]) if len(lam) >= 3 else None
                    rec["ray_extrapolations"].append({"normal": ns, "theta0": th, "b_over_rg": bo, "C_lambda0": float(ic), "slope": float(sl), "curvature": curv, "t31": t31})
                    bs.append(bo); c0s.append(float(ic))
                    L.append("| %s | %.4f | %g | %.4f | %s | %.5f | %.4f | %s | %.1e |" % (ns, th, bo, t31, ", ".join("%.5f" % x for x in cs), ic, sl,
                                                                                ("%.3g" % curv) if curv is not None else "-", drift))
                    print(L[-1], flush=True)
                    json.dump(rec, open(jpath, "w"), indent=1, default=float)
                x = np.array([math.sqrt(1.0 / b) for b in bs]); y = np.array(c0s)
                C1 = float(np.linalg.lstsq((x ** 2)[:, None], y - C0 - Chalf * x, rcond=None)[0][0])
                res = float(np.max(np.abs(y - C0 - Chalf * x - C1 * x ** 2)))
                A = np.vstack([x, x ** 2]).T
                ch_free, c1_free = np.linalg.lstsq(A, y - C0, rcond=None)[0]
                rec["fits"].append({"normal": ns, "theta0": th, "C0": C0, "C_half_redteam": Chalf, "C1_fit": C1, "residual": res, "C_half_free": float(ch_free),
                                    "C1_free": float(c1_free), "residual_free": float(np.max(np.abs(y - C0 - ch_free * x - c1_free * x ** 2)))})
        L += ["", "### The fits on the lambda = 0 values: C_0 + C_1/2 sqrt(r_g/b) + C_1 r_g/b", "",
              "| normal | theta0 | C_0 | C_1/2 (red team) | C_1 fitted | max residual | C_1/2 free | C_1 (free) | max residual |", "|---|---|---|---|---|---|---|---|---|"]
        for f in rec["fits"]:
            L.append("| %s | %.4f | %.5f | %.5f | %.4f | %.2e | %.5f | %.4f | %.2e |" % (f["normal"], f["theta0"], f["C0"], f["C_half_redteam"], f["C1_fit"], f["residual"],
                     f["C_half_free"], f["C1_free"], f["residual_free"]))
    L += ["", "## The [011] periastron at e = %g: (advance lattice/geodesic - 1)/(m a)^2 per window of 1, 3, 7 periods; the tolerance ladder's spread" % a.e, "",
          "| m a | A/r_g | theta0 | sense | C_011 (T7g, sense +1) | coefficient: 1, 3, 7 periods (rtol 1e-12, A/40) | ladder spread (7 periods) | every advance, lattice minus geodesic (rad) | r_min/r_collar | out of plane |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    nrm, e1 = P9.PLANES["011"]
    ladder = [(1e-12, 1.0 / 40), (1e-12, 1.0 / 20), (1e-10, 1.0 / 40)]
    for ma in ([] if a.skip_orbits else a.ma):
        for A_ in a.A_over_rg:
            for th in a.theta0_orbit:
                for sense in (+1, -1):
                    vals = []
                    main = None
                    for (rt, ms) in ladder:
                        pl = periapsis_series(F, ma, A_ * rg, a.e, th, nrm, e1, True, sense, a.norbits, rt, ms)
                        pg = periapsis_series(F, ma, A_ * rg, a.e, th, nrm, e1, False, sense, a.norbits, rt, ms)
                        n = min(pl["n"], pg["n"])
                        coefs = {}
                        for w in (1, 3, 7):
                            if n >= w:
                                al = float(np.mean(pl["advances"][:w])); ag = float(np.mean(pg["advances"][:w]))
                                coefs[w] = (al / ag - 1) / ma ** 2
                        vals.append(coefs.get(7))
                        if main is None:
                            main = (pl, pg, coefs, n)
                    pl, pg, coefs, n = main
                    spread = float(np.ptp([x for x in vals if x is not None])) if any(x is not None for x in vals) else None
                    diffs = [float(x - y) for x, y in zip(pl["advances"][:n], pg["advances"][:n])]
                    c011 = C011(th, a.e)
                    row = {"ma": ma, "A_over_rg": A_, "theta0": th, "sense": sense, "C011_T7g_sense_plus": c011, "coefficient_windows": coefs,
                           "ladder_7_period_values": vals, "ladder_spread": spread, "advance_differences": diffs, "lattice": pl, "geodesic": pg,
                           "r_min_over_collar": pl["r_min"] / M["r_collar"]}
                    rec["orbits_011"].append(row)
                    L.append("| %g | %g | %+.4f | %+d | %.6f | %s | %s | %s | %.2f | %.1e |" % (ma, A_, th, sense, c011, ", ".join(("%.5f" % coefs[w]) if w in coefs else "-" for w in (1, 3, 7)),
                             ("%.1e" % spread) if spread is not None else "-", ", ".join("%.3e" % x for x in diffs), pl["r_min"] / M["r_collar"], pl["out_of_plane_max"]))
                    print(L[-1], flush=True)
                    json.dump(rec, open(jpath, "w"), indent=1, default=float)
    L += ["", "- every number above is printed by the script; the rays' R_far/b convention is stated per row; C_011 is T7g's leading closed form for sense +1",
          "  (the sine terms change sign with the sense); the ladder is (rtol 1e-12, max_step A/40), (1e-12, A/20), (1e-10, A/40)."]
    json.dump(rec, open(jpath, "w"), indent=1, default=float)
    open(os.path.join(a.out, "ppn10_%s.md" % a.tag), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
