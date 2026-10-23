#!/usr/bin/env python
"""engine_ppn_v7.py (builder analyst; SPEC_final4b_engine.md sec. 14.1's "what remains of task 7": version 7 of engine_ppn_v6.py).

Version 7: (1) the impact-parameter ladder b/r_g in {100, 200, 400} in the three planes (the default; version 6 ran 20 and 50), for the
large-b limits and the r_g/b terms; (2) the delay per normal: the ray's time from the start point to the exit sphere |X| = R_far minus the
same symbol's flat time over the straight chord of the same impact parameter, 2 sqrt(R_far^2 - b_eff^2)/|v_flat| (|v_flat| the symbol's own
flat group speed at the start momentum: 1 for the geodesic), as the ratio lattice/geodesic - 1 per (k a)^2, with the geodesic's excess
printed beside r_g ln(4 R_far^2/b^2) (the leading Shapiro term; the O(r_g) remainder and the exit-point geometry, of order b times the
deflection, are in the excess and the same for both symbols up to the deflection's own correction); (3) the periastron on [111] at the
eccentricities --e-111 (default 0.1, 0.3, 0.5), the other planes at --e as in version 6.  Version 6's docstring follows.

engine_ppn_v6.py (builder analyst; SPEC_final4b_engine.md sec. 13.4 (1), the plane-normal scan of task 7).

Version 6 = engine_ppn_v5 in three dimensions: the declared operator's symbol H_lat = N(r) sqrt(m^2 + sum_i sin^2 p_i) +
u(r) sum_i x_i sin(2 p_i)/2 (u = v/r) and the geodesic H_geo = N sqrt(m^2 + p^2) + u x.p on the solved exterior of a star member (the member's
N, v inside the collar, the analytic exterior beyond it), with the full 3-dimensional Hamilton equations. The rays and orbits start in the
plane of normal n ([001], [011], [111]) with the in-plane axes e1, e2 (e1 the first lattice direction listed for that plane below, e2 = n x e1),
theta0 measured from e1; the cubic symbol need not keep the motion in a tilted plane: the largest out-of-plane coordinate over the run is
printed. Deflection: the lattice ray aimed so that its flat-space group velocity lies along the start direction (40 fixed-point iterations of
the momentum's direction, which may leave the plane for a tilted normal), both rays started from the same point, the (delta b)-normalized ratio as in version 5. Precession: the
geodesic's bound orbit (turning points a (1 - e), a (1 + e)) started at the apoapsis in the plane with the momentum of zero radial velocity,
the lattice symbol from the same point and momentum; the periapsis angle in the plane per orbit over 6 orbits.
Args: --star --member --normals --b-over-rg --k --theta0 --m-orbit --a-over-rg --e --out --tag."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

PLANES = {"001": (np.array([0, 0, 1.0]), np.array([1.0, 0, 0])), "011": (np.array([0, 1.0, 1.0]) / math.sqrt(2), np.array([1.0, 0, 0])),
          "111": (np.array([1.0, 1.0, 1.0]) / math.sqrt(3), np.array([1.0, -1.0, 0]) / math.sqrt(2))}


class Fields:
    def __init__(self, R, N, v, E, g, r_collar):
        self.g, self.E, self.rc = g, E, r_collar
        sel = R <= r_collar
        self.sN = CubicSpline(R[sel], N[sel]); self.sv = CubicSpline(R[sel], v[sel])

    def all(self, r):
        if r < self.rc:
            N, dN, v, dv = float(self.sN(r)), float(self.sN(r, 1)), float(self.sv(r)), float(self.sv(r, 1))
        else:
            c = math.sqrt(2 * self.g * self.E)
            N, dN, v, dv = 1.0, 0.0, -c / math.sqrt(r), 0.5 * c * r ** -1.5
        return N, dN, v / r, (dv * r - v) / (r * r)


class FlatFields:
    """N = 1, v = 0 everywhere (the symbols' flat motion, for the flat chord time)."""
    g, E, rc = 0.0, 0.0, 0.0

    def all(self, r):
        return 1.0, 0.0, 0.0, 0.0


Fflat = FlatFields()


def make_rhs(F, m, lattice):
    def rhs(t, y):
        X = y[:3]; p = y[3:]
        r = float(np.linalg.norm(X))
        N, dN, u, du = F.all(r)
        if lattice:
            s = np.sin(p); c = np.cos(p)
            E = math.sqrt(m * m + float(s @ s))
            q = np.sin(2 * p) / 2; dq = np.cos(2 * p); dE = s * c / E
        else:
            E = math.sqrt(m * m + float(p @ p))
            q = p; dq = np.ones(3); dE = p / E
        Xq = float(X @ q)
        dX = N * dE + u * X * dq
        dp = -(dN * E + du * Xq) * X / r - u * q
        return np.concatenate([dX, dp])
    return rhs


def H_of(F, y, m, lattice):
    X = y[:3]; p = y[3:]
    N, dN, u, du = F.all(float(np.linalg.norm(X)))
    if lattice:
        return N * math.sqrt(m * m + float(np.sum(np.sin(p) ** 2))) + u * float(X @ (np.sin(2 * p) / 2))
    return N * math.sqrt(m * m + float(p @ p)) + u * float(X @ p)


def gv_flat(p, lattice):
    if lattice:
        g = np.sin(2 * p) / 2
    else:
        g = p.copy()
    return g / max(np.linalg.norm(g), 1e-300)


def deflection(F, b, k, theta0, nrm, e1, lattice, R_far_over_b=2000.0, omega=None):
    e2 = np.cross(nrm, e1)
    d = math.cos(theta0) * e1 + math.sin(theta0) * e2
    nperp = -math.sin(theta0) * e1 + math.cos(theta0) * e2
    R_far = R_far_over_b * b
    X0 = -math.sqrt(R_far ** 2 - b * b) * d + b * nperp
    if lattice:
        p = k * d
        for _ in range(40):                                      # aim the flat-space group velocity along d (within the plane)
            g = gv_flat(p, True)
            corr = d - g
            p = k * (p / k + corr)
            p = k * p / np.linalg.norm(p)
    else:
        p = omega * d
    y0 = np.concatenate([X0 * (1 - 1e-12), p])
    v0 = make_rhs(F, 0.0, lattice)(0.0, y0)[:3]
    vh0 = v0 / np.linalg.norm(v0)
    b_eff = float(np.linalg.norm(np.cross(X0, vh0)))
    ev = lambda t, y: float(np.linalg.norm(y[:3])) - R_far
    ev.terminal = True; ev.direction = 1
    sol = solve_ivp(make_rhs(F, 0.0, lattice), (0, 4.0 * R_far / max(np.linalg.norm(v0), 1e-3)), y0, events=ev, rtol=1e-11, atol=1e-12,
                    method="DOP853", max_step=b)
    v1 = make_rhs(F, 0.0, lattice)(0.0, sol.y[:, -1])[:3]
    ang = math.atan2(float(np.linalg.norm(np.cross(vh0, v1))), float(vh0 @ v1))
    tail = (4 * F.g * F.E / b_eff) * (1.0 - math.sqrt(1.0 - (b_eff / R_far) ** 2))
    oop = float(np.max(np.abs(sol.y[:3].T @ nrm)))
    T_exit = float(sol.t_events[0][0]) if len(sol.t_events[0]) else float(sol.t[-1])
    v_flat = float(np.linalg.norm(make_rhs(Fflat, 0.0, lattice)(0.0, y0)[:3]))
    T_flat = 2.0 * math.sqrt(max(R_far ** 2 - b_eff ** 2, 0.0)) / v_flat
    return {"deflection": ang + tail, "b_eff": b_eff, "omega": H_of(F, y0, 0.0, lattice), "out_of_plane_max": oop, "status": int(sol.status),
            "T_exit": T_exit, "v_flat": v_flat, "delay": T_exit - T_flat, "R_far": R_far}


def orbit_start(F, m, a_orb, e, theta0, nrm, e1):
    rp, ra = a_orb * (1 - e), a_orb * (1 + e)
    rg = 2 * F.g * F.E
    A1, A2 = 1 - rg / rp, 1 - rg / ra
    l2 = (A1 - A2) / (A2 / ra ** 2 - A1 / rp ** 2)
    kappa = math.sqrt(l2) * m
    e2 = np.cross(nrm, e1)
    rh = math.cos(theta0) * e1 + math.sin(theta0) * e2
    th = np.cross(nrm, rh)
    X = ra * rh
    pt = kappa / ra
    N, dN, u, du = F.all(ra)
    v = u * ra
    pr = brentq(lambda pr: N * pr / math.sqrt(m * m + pt * pt + pr * pr) + v, -5.0, 5.0, xtol=1e-15)
    return np.concatenate([X, pt * th + pr * rh])


def precession(F, m, a_orb, e, theta0, nrm, e1, lattice, norbits=6):
    y0 = orbit_start(F, m, a_orb, e, theta0, nrm, e1)
    e2 = np.cross(nrm, e1)

    def rdot(t, y):
        dy = make_rhs(F, m, lattice)(t, y)
        return float(y[:3] @ dy[:3])
    rdot.direction = 1
    Tk = 2 * math.pi * math.sqrt(a_orb ** 3 / (F.g * F.E)) * (norbits + 1)
    sol = solve_ivp(make_rhs(F, m, lattice), (0, Tk), y0, events=rdot, rtol=1e-12, atol=1e-12, method="DOP853", max_step=a_orb / 20)
    ye = sol.y_events[0]
    ang = np.unwrap(np.arctan2(ye[:, :3] @ e2, ye[:, :3] @ e1)) if len(ye) else np.array([])
    adv = np.diff(ang) if len(ang) > 1 else np.array([])
    return {"advance_mean": float(np.mean(adv)) if len(adv) else None, "advance_spread": float(np.ptp(adv)) if len(adv) else None,
            "orbits": int(len(ang)), "out_of_plane_max": float(np.max(np.abs(sol.y[:3].T @ nrm))), "p_max": float(np.max(np.linalg.norm(sol.y[3:], axis=0)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--star", required=True)
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--normals", default="001,011,111")
    ap.add_argument("--b-over-rg", type=float, nargs="+", default=[100.0, 200.0, 400.0])
    ap.add_argument("--e-111", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    ap.add_argument("--k", type=float, nargs="+", default=[0.025, 0.05, 0.1])
    ap.add_argument("--theta0", type=float, nargs="+", default=[0.0, math.pi / 8, math.pi / 4])
    ap.add_argument("--m-orbit", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--a-over-rg", type=float, default=40.0)
    ap.add_argument("--e", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    D = json.load(open(a.star))
    M = D["members"][a.member]
    R = np.arange(M["nsites"], dtype=float) + (0.5 if D["args"].get("half_offset") else 1.0)
    g, E = M["g"], M["E"]
    F = Fields(R, np.array(M["N"]), np.array(M["v"]), E, g, M["r_collar"])
    rg = 2 * g * E
    rec = {"script": "engine_ppn_v7.py", "star": os.path.basename(a.star), "member": a.member, "g": g, "E": E, "r_g": rg, "deflection": [], "precession": []}
    L = ["# T7.1 version 7 (engine_ppn_v7.py): the b ladder, the delay per normal, the [111] eccentricities, on the declared symbol (member %d of %s; r_g = 2 g E = %.4f)" % (a.member, os.path.basename(a.star), rg), "",
         "| normal | b/r_g | k a | theta0 | (delta b) lattice/geodesic - 1 | /(k a)^2 | delay lattice/geodesic - 1 | /(k a)^2 | geodesic delay / r_g ln(4 R_far^2/b^2) | out of plane max |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for ns in a.normals.split(","):
        nrm, e1 = PLANES[ns]
        for bo in a.b_over_rg:
            for k in a.k:
                for th in a.theta0:
                    dl = deflection(F, bo * rg, k, th, nrm, e1, True)
                    dg = deflection(F, bo * rg, k, th, nrm, e1, False, omega=dl["omega"])
                    r_ = (dl["deflection"] * dl["b_eff"]) / (dg["deflection"] * dg["b_eff"]) - 1
                    dr_ = dl["delay"] / dg["delay"] - 1
                    shap = dg["delay"] / (rg * math.log(4 * dg["R_far"] ** 2 / dg["b_eff"] ** 2))
                    rec["deflection"].append({"normal": ns, "b_over_rg": bo, "k": k, "theta0": th, "lattice": dl, "geodesic": dg, "ratio_minus_1": r_,
                                              "delay_ratio_minus_1": dr_, "geodesic_delay_over_shapiro": shap})
                    L.append("| %s | %g | %g | %.4f | %.4e | %.4f | %.4e | %.4f | %.5f | %.2e |" % (ns, bo, k, th, r_, r_ / (k * k), dr_, dr_ / (k * k), shap, dl["out_of_plane_max"]))
                    print(L[-1], flush=True)
    L += ["", "| normal | e | m a | theta0 | advance lattice | geodesic | lattice/geodesic - 1 | /(m a)^2 | out of plane max | spread |", "|---|---|---|---|---|---|---|---|---|---|"]
    for ns in a.normals.split(","):
        nrm, e1 = PLANES[ns]
        for ee in (a.e_111 if ns == "111" else [a.e]):
          for mo in a.m_orbit:
            for th in a.theta0:
                pl = precession(F, mo, a.a_over_rg * rg, ee, th, nrm, e1, True)
                pg = precession(F, mo, a.a_over_rg * rg, ee, th, nrm, e1, False)
                r_ = pl["advance_mean"] / pg["advance_mean"] - 1 if (pl["advance_mean"] and pg["advance_mean"]) else None
                rec["precession"].append({"normal": ns, "e": ee, "m": mo, "theta0": th, "lattice": pl, "geodesic": pg, "ratio_minus_1": r_})
                L.append("| %s | %g | %g | %.4f | %s | %s | %s | %s | %.2e | %s |" % (ns, ee, mo, th, pl["advance_mean"], pg["advance_mean"], ("%.4e" % r_) if r_ is not None else "-",
                         ("%.3f" % (r_ / mo ** 2)) if r_ is not None else "-", pl["out_of_plane_max"], pl["advance_spread"]))
                print(L[-1], flush=True)
    L += ["", "- every number above is printed by the script; the geodesic symbol on the exact exterior is the general-relativistic reference."]
    os.makedirs(a.out, exist_ok=True)
    json.dump(rec, open(os.path.join(a.out, "ppn7_%s.json" % a.tag), "w"), indent=1, default=float)
    open(os.path.join(a.out, "ppn7_%s.md" % a.tag), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
