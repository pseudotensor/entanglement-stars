#!/usr/bin/env python
"""engine_ppn_v3.py (builder analyst; version 3 = version 2 with the lattice ray aimed: the momentum's direction is iterated (4 times, flat
space) so that the lattice group velocity points along theta0; version 2 started with p along theta0, whose group velocity is off by
O((ka)^2) sin 4 theta0, which at R_far = 2000 b moved the impact parameter by a factor up to 60 at theta0 = pi/8; also --k accepts the small
carriers for the k -> 0 check).  SPEC_final4b_engine.md sec. 6, T7.1, version 2 of the lead's engine_ppn_v1 as completed by the builder.  Why version 2 (builder finding, labelled): version 1 integrated the radial chain's symbol, whose radial part is the sine
dispersion while its angular part kappa/R is the continuum's; that mixed symbol is not rotation invariant, so even in flat space its rays
bend and its orbits precess, and on the solved exterior its "lattice corrections" grew with the impact parameter (lattice/geodesic - 1 of
0.015 at b = 20 r_g and 0.094 at b = 100 r_g at k a = 0.05, record ppn_t7m0) and reversed the precession at m a = 1: the reduction's own
anisotropy, not the declared operator's.  Version 2 uses the declared operator's symbol in the orbital plane (z = 0; Cartesian momenta
p_x, p_y; the flow v^i = v(r) x^i/r):
    H_lat = N(r) sqrt(m^2 + sin^2 p_x + sin^2 p_y) + u(r) (x sin(2 p_x)/2 + y sin(2 p_y)/2),   u = v/r,
against the geodesic H_geo = N sqrt(m^2 + p^2) + u (x p_x + y p_y) on the same fields (the member's solved N, v inside the collar, the
exact exterior N = 1, v = -sqrt(2 g E/r) beyond it).  The cubic anisotropy enters through the orientation theta0 of the incoming ray or of
the orbit's start relative to the lattice axes (0, pi/8, pi/4).
(1) Deflection: a massless ray enters at |X| = R_far = 2000 b with momentum k (cos theta0, sin theta0) (lattice) or omega (cos, sin)
(geodesic, omega = the lattice ray's Killing energy at the start), the impact parameter b_eff the distance from the origin to the incoming
line of the group velocity; the deflection is the signed angle between the final and the initial group velocities when |X| returns to
R_far, plus the first-order tail beyond R_far on both legs (4 g E/b)(1 - sqrt(1 - b^2/R_far^2)) (labelled; below 1e-6 relative); in flat
space both symbols move in straight lines (constant p), so no flat reference is subtracted.  (2) Precession: the geodesic's bound orbit
with turning points a (1 - e), a (1 + e) (the Schwarzschild conditions for the energy and angular momentum per unit mass), started at the
apoapsis X = r_a (cos theta0, sin theta0) with the momentum of zero radial velocity; the lattice symbol is started from the same (X, p);
the periapses (local minima of r) over 6 orbits, the advance per orbit and its mean.  (3) The radial delay is version 1's (a radial ray
along an axis is the same in both forms).  Writes <out>/ppn2_<tag>.json and .md.  Args: --star --member --out --tag."""
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


class Fields:
    def __init__(self, R, N, v, E, g, r_collar):
        self.g, self.E, self.rc = g, E, r_collar
        sel = R <= r_collar
        self.sN = CubicSpline(R[sel], N[sel]); self.sv = CubicSpline(R[sel], v[sel])

    def all(self, r):
        """N, N', u = v/r, u' at r."""
        if r < self.rc:
            N, dN, v, dv = float(self.sN(r)), float(self.sN(r, 1)), float(self.sv(r)), float(self.sv(r, 1))
        else:
            c = math.sqrt(2 * self.g * self.E)
            N, dN, v, dv = 1.0, 0.0, -c / math.sqrt(r), 0.5 * c * r ** -1.5
        return N, dN, v / r, (dv * r - v) / (r * r)


def make_rhs(F, m, lattice):
    def rhs(t, y):
        x, yy, px, py = y
        r = math.hypot(x, yy)
        N, dN, u, du = F.all(r)
        if lattice:
            sx, sy = math.sin(px), math.sin(py)
            E = math.sqrt(m * m + sx * sx + sy * sy)
            qx, qy = math.sin(2 * px) / 2, math.sin(2 * py) / 2
            dqx, dqy = math.cos(2 * px), math.cos(2 * py)
            dEx, dEy = sx * math.cos(px) / E, sy * math.cos(py) / E
        else:
            E = math.sqrt(m * m + px * px + py * py)
            qx, qy, dqx, dqy = px, py, 1.0, 1.0
            dEx, dEy = px / E, py / E
        Xq = x * qx + yy * qy
        dxdt = N * dEx + u * x * dqx
        dydt = N * dEy + u * yy * dqy
        dpx = -(dN * E + du * Xq) * x / r - u * qx
        dpy = -(dN * E + du * Xq) * yy / r - u * qy
        return [dxdt, dydt, dpx, dpy]
    return rhs


def H_of(F, y, m, lattice):
    x, yy, px, py = y
    N, dN, u, du = F.all(math.hypot(x, yy))
    if lattice:
        return N * math.sqrt(m * m + math.sin(px) ** 2 + math.sin(py) ** 2) + u * (x * math.sin(2 * px) + yy * math.sin(2 * py)) / 2
    return N * math.sqrt(m * m + px * px + py * py) + u * (x * px + yy * py)


def group_velocity(F, y, m, lattice):
    d = make_rhs(F, m, lattice)(0.0, y)
    return np.array(d[:2])


def deflection(F, b, k, theta0, lattice, R_far_over_b=2000.0, omega=None):
    d = np.array([math.cos(theta0), math.sin(theta0)]); n = np.array([-d[1], d[0]])
    R_far = R_far_over_b * b
    X0 = -math.sqrt(R_far ** 2 - b * b) * d + b * n
    p0 = (k if lattice else omega) * d
    if lattice:                                                         # version 3: aim the group velocity along theta0 (flat space)
        ang = theta0
        for _ in range(4):
            px, py = k * math.cos(ang), k * math.sin(ang)
            vg = math.atan2(math.sin(2 * py), math.sin(2 * px))
            ang += theta0 - vg
        p0 = k * np.array([math.cos(ang), math.sin(ang)])
    y0 = [X0[0], X0[1], p0[0], p0[1]]
    v0 = group_velocity(F, y0, 0.0, lattice)
    vhat0 = v0 / np.linalg.norm(v0)
    b_eff = abs(X0[0] * vhat0[1] - X0[1] * vhat0[0])
    ev = lambda t, y: math.hypot(y[0], y[1]) - R_far
    ev.terminal = True; ev.direction = 1
    sol = solve_ivp(make_rhs(F, 0.0, lattice), (0, 4.0 * R_far / max(np.linalg.norm(v0), 1e-3)), [X0[0] * (1 - 1e-12), X0[1] * (1 - 1e-12), p0[0], p0[1]],
                    events=ev, rtol=1e-11, atol=1e-12, method="DOP853", max_step=b)
    v1 = group_velocity(F, sol.y[:, -1], 0.0, lattice)
    ang = math.atan2(vhat0[0] * v1[1] - vhat0[1] * v1[0], vhat0 @ v1)
    tail = (4 * F.g * F.E / b_eff) * (1.0 - math.sqrt(1.0 - (b_eff / R_far) ** 2))
    rmin = float(np.min(np.hypot(sol.y[0], sol.y[1])))
    return {"deflection": abs(ang) + tail, "b_eff": b_eff, "omega": H_of(F, y0, 0.0, lattice), "r_min": rmin, "status": int(sol.status)}


def orbit_start(F, m, a_orb, e, theta0):
    rp, ra = a_orb * (1 - e), a_orb * (1 + e)
    rg = 2 * F.g * F.E
    A1, A2 = 1 - rg / rp, 1 - rg / ra
    l2 = (A1 - A2) / (A2 / ra ** 2 - A1 / rp ** 2)
    kappa = math.sqrt(l2) * m
    rh = np.array([math.cos(theta0), math.sin(theta0)]); th = np.array([-rh[1], rh[0]])
    X = ra * rh
    pt = kappa / ra
    N, dN, u, du = F.all(ra)
    v = u * ra
    f = lambda pr: N * pr / math.sqrt(m * m + pt * pt + pr * pr) + v       # the geodesic's radial velocity
    pr = brentq(f, -5.0, 5.0, xtol=1e-15)
    p = pt * th + pr * rh
    return [X[0], X[1], p[0], p[1]]


def precession(F, m, a_orb, e, theta0, lattice, norbits=6):
    y0 = orbit_start(F, m, a_orb, e, theta0)
    H0 = H_of(F, y0, m, lattice)

    def rdot(t, y):
        dy = make_rhs(F, m, lattice)(t, y)
        return y[0] * dy[0] + y[1] * dy[1]
    rdot.direction = 1                                                  # r' crossing zero upward: a periapsis
    Tk = 2 * math.pi * math.sqrt(a_orb ** 3 / (F.g * F.E)) * (norbits + 1)
    sol = solve_ivp(make_rhs(F, m, lattice), (0, Tk), y0, events=rdot, rtol=1e-12, atol=1e-12, method="DOP853", max_step=a_orb / 20)
    ye = sol.y_events[0]
    ang = np.unwrap(np.arctan2(ye[:, 1], ye[:, 0])) if len(ye) else np.array([])
    adv = np.diff(ang) - 2 * math.pi if len(ang) > 1 else np.array([])
    pmax = float(np.max(np.hypot(sol.y[2], sol.y[3])))
    return {"advance_per_orbit": adv.tolist(), "advance_mean": float(np.mean(adv)) if len(adv) else None, "orbits": int(len(ang)),
            "H_drift": float(abs(H_of(F, sol.y[:, -1], m, lattice) - H0)), "p_max": pmax}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--star", required=True)
    ap.add_argument("--member", type=int, default=None)
    ap.add_argument("--b-over-rg", type=float, nargs="+", default=[20.0, 50.0, 100.0])
    ap.add_argument("--skip-precession", action="store_true")
    ap.add_argument("--k", type=float, nargs="+", default=[0.0125, 0.025, 0.05, 0.1, 0.2, 0.4])
    ap.add_argument("--theta0", type=float, nargs="+", default=[0.0, math.pi / 8, math.pi / 4])
    ap.add_argument("--m-orbit", type=float, nargs="+", default=[0.2, 1.0])
    ap.add_argument("--a-over-rg", type=float, default=40.0)
    ap.add_argument("--e", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    D = json.load(open(a.star))
    mems = D["members"]
    i = a.member if a.member is not None else next(j for j, mm in enumerate(mems) if not mm.get("failed"))
    M = mems[i]
    R = np.arange(M["nsites"], dtype=float) + (0.5 if D["args"].get("half_offset") else 1.0)
    g, E = M["g"], M["E"]
    F = Fields(R, np.array(M["N"]), np.array(M["v"]), E, g, M["r_collar"])
    rg = 2 * g * E
    rec = {"script": "engine_ppn_v3.py", "star": os.path.basename(a.star), "member": i, "g": g, "E": E, "r_g": rg, "deflection": [], "precession": []}
    L = ["# T7.1 version 3 (engine_ppn_v3.py): the declared operator's symbol in the orbital plane on the solved exterior (member %d of %s; g %g, E %.6f, r_g = 2 g E = %.4f)" % (
        i, os.path.basename(a.star), g, E, rg), "",
        "| b/r_g | k a | theta0 | deflection, lattice | geodesic (same b, same omega) | 4 g E/b | geodesic/(4gE/b) - 1 | lattice/geodesic - 1 |", "|---|---|---|---|---|---|---|---|"]
    for bo in a.b_over_rg:
        b = bo * rg
        for k in a.k:
            for th in a.theta0:
                dl = deflection(F, b, k, th, True)
                dg = deflection(F, dl["b_eff"], k, th, False, omega=dl["omega"])
                rec["deflection"].append({"b_over_rg": bo, "k": k, "theta0": th, "lattice": dl, "geodesic": dg})
                L.append("| %g | %g | %.4f | %.8e | %.8e | %.8e | %.4e | %.4e |" % (bo, k, th, dl["deflection"], dg["deflection"], 4 * g * E / dl["b_eff"],
                         dg["deflection"] / (4 * g * E / dl["b_eff"]) - 1, dl["deflection"] / dg["deflection"] - 1))
                print(L[-1], flush=True)
    L += ["", "| m a | a_orb/r_g | e | theta0 | advance per orbit, lattice | geodesic | 6 pi g E/(a (1 - e^2)) | lattice/geodesic - 1 | p_max a | spread of the lattice advance over the orbits |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for mo in ([] if a.skip_precession else a.m_orbit):
        aorb = a.a_over_rg * rg
        for th in a.theta0:
            pl = precession(F, mo, aorb, a.e, th, True); pg = precession(F, mo, aorb, a.e, th, False)
            rec["precession"].append({"m": mo, "a_over_rg": a.a_over_rg, "e": a.e, "theta0": th, "lattice": pl, "geodesic": pg,
                                      "first_order": 6 * math.pi * g * E / (aorb * (1 - a.e ** 2))})
            L.append("| %g | %g | %g | %.4f | %s | %s | %.6e | %s | %.4f | %s |" % (mo, a.a_over_rg, a.e, th,
                     "%.6e" % pl["advance_mean"] if pl["advance_mean"] is not None else "-", "%.6e" % pg["advance_mean"] if pg["advance_mean"] is not None else "-",
                     6 * math.pi * g * E / (aorb * (1 - a.e ** 2)), "%.4e" % (pl["advance_mean"] / pg["advance_mean"] - 1) if (pl["advance_mean"] and pg["advance_mean"]) else "-",
                     pl["p_max"], "%.3e" % float(np.ptp(pl["advance_per_orbit"])) if pl["advance_per_orbit"] else "-"))
            print(L[-1], flush=True)
    L += ["", "- every number above is printed by the script; the geodesic symbol on the exact exterior is the general-relativistic reference; theta0 is the angle to the lattice x axis."]
    os.makedirs(a.out, exist_ok=True)
    json.dump(rec, open(os.path.join(a.out, "ppn3_%s.json" % a.tag), "w"), indent=1, default=float)
    open(os.path.join(a.out, "ppn3_%s.md" % a.tag), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
