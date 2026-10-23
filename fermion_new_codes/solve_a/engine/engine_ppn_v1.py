#!/usr/bin/env python
"""engine_ppn_v1.py (lead's draft, completed by the builder analyst; SPEC_final4b_engine.md sec. 6, T7.1): the matter's post-Newtonian
content on the solved exterior by the symbol's Hamilton equations, H_lat(R, p_r, kappa) = N(R) sqrt(m^2 + sin^2 p_r + (kappa/R)^2) +
v(R) sin(2 p_r)/2 against the geodesic H_geo = N sqrt(m^2 + p_r^2 + kappa^2/R^2) + v p_r (the same solved N, v, which beyond the matter are
N = 1, v = -sqrt(2 g E/R)).  Deflection of a massless ray against 4 g E/b; periastron advance against 6 pi g E/(a_orb (1 - e^2)); each
against (ka)^2 and a/r_min.

Builder's completion (every choice labelled): (1) the fields: a converged member of engine_star_v2/v3 (its N, v on the sites, its charge E)
inside the collar, the exact exterior N = 1, v = -sqrt(2 g E/r) beyond it (the solver writes that exterior, so the two agree at the collar to
rounding; checked and printed); cubic splines of the member's fields inside, analytic derivatives outside.  (2) The deflection: the ray
enters at R_far = 2000 b with radial momentum inward and the Killing energy omega = sin(k) (lattice) or k (geodesic) and kappa = omega b
(b := kappa/omega, the classical impact parameter: the angular term is the continuum's in both symbols); phi' = dH/dkappa = N kappa/(r^2 E)
is integrated with (r, p_r); the run stops when r returns to R_far; the deflection is phi_total - pi + 2 arcsin(b/R_far) (the straight line's
swept angle removed) plus the far tail beyond R_far on both legs at its first-order value 2 (2 g E/b)(1 - sqrt(1 - b^2/R_far^2)) (labelled; it is
below 1e-6 of the deflection at R_far = 2000 b).  The geodesic symbol on the exact exterior is Schwarzschild's null geodesic in
Painleve-Gullstrand time, so its deflection is the exact general-relativistic bending at this b; the lattice minus geodesic difference is
the lattice correction at carrier k.  (3) The radial delay: a radial ray (kappa = 0, massless) from R = 20 r_g to 2 r_g inward and back out
(r_g = 2 g E), the coordinate travel time of the lattice symbol against the geodesic's, and each against flat space (v = 0) with the same
carrier: the gravitational delay ratio lattice/geodesic.  (4) The periastron advance: a bound orbit of mass m with turning points
r_p = a (1 - e), r_a = a (1 + e) for the geodesic (the Schwarzschild effective-potential conditions solved for the energy and angular momentum
per unit mass, then omega = m eps, kappa = m l); the lattice symbol run with the same (omega, kappa) (its turning points shift by O(p^2));
the angle between successive periapses over 6 orbits; reported against 6 pi g E/(a (1 - e^2)) and the momentum scale p_max a at periapsis.
Writes <out>/ppn_<tag>.json and .md.  Args: --star (engine_star JSON) --member (index; default the first converged) --out --tag."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline


class Fields:
    def __init__(self, R, N, v, E, g, r_collar):
        self.g, self.E, self.rc = g, E, r_collar
        sel = R <= r_collar
        self.sN = CubicSpline(R[sel], N[sel]); self.sv = CubicSpline(R[sel], v[sel])
        self.rmin = float(R[0])
        self.collar_jump = (abs(float(self.sN(r_collar)) - 1.0), abs(float(self.sv(r_collar)) + math.sqrt(2 * g * E / r_collar)))

    def N(self, r):
        return float(self.sN(r)) if r < self.rc else 1.0

    def v(self, r):
        return float(self.sv(r)) if r < self.rc else -math.sqrt(2 * self.g * self.E / r)

    def dN(self, r):
        return float(self.sN(r, 1)) if r < self.rc else 0.0

    def dv(self, r):
        return float(self.sv(r, 1)) if r < self.rc else 0.5 * math.sqrt(2 * self.g * self.E) * r ** -1.5


def rhs_factory(F, m, kappa, lattice, flat=False):
    def rhs(t, y):
        r, p, phi = y
        a2 = (kappa / r) ** 2
        N, v, dN, dv = (1.0, 0.0, 0.0, 0.0) if flat else (F.N(r), F.v(r), F.dN(r), F.dv(r))
        if lattice:
            E = math.sqrt(m * m + a2 + math.sin(p) ** 2)
            dr = N * math.sin(p) * math.cos(p) / E + v * math.cos(2 * p)
            dp = -dN * E + N * a2 / (r * E) - dv * math.sin(2 * p) / 2
        else:
            E = math.sqrt(m * m + a2 + p * p)
            dr = N * p / E + v
            dp = -dN * E + N * a2 / (r * E) - dv * p
        dphi = N * kappa / (r * r * E)
        return [dr, dp, dphi]
    return rhs


def H_of(F, r, p, m, kappa, lattice, flat=False):
    N, v = (1.0, 0.0) if flat else (F.N(r), F.v(r))
    a2 = (kappa / r) ** 2
    if lattice:
        return N * math.sqrt(m * m + a2 + math.sin(p) ** 2) + v * math.sin(2 * p) / 2
    return N * math.sqrt(m * m + a2 + p * p) + v * p


def p_inward(F, r, omega, m, kappa, lattice, flat=False):
    """the inward radial momentum at r with H = omega (the root with dr/dt < 0 nearest p = 0 from below)."""
    from scipy.optimize import brentq
    f = lambda p: H_of(F, r, p, m, kappa, lattice, flat) - omega
    hi = 0.0
    lo = -(math.pi / 2 - 1e-9) if lattice else -10.0 * (omega + 1.0)
    return brentq(f, lo, hi - 1e-15, xtol=1e-15, rtol=1e-14, maxiter=500)


def deflection(F, b, k, lattice, R_far_over_b=2000.0):
    omega = math.sin(k) if lattice else k
    kappa = omega * b
    R_far = R_far_over_b * b
    p0 = p_inward(F, R_far, omega, 0.0, kappa, lattice)
    out_ev = lambda t, y: y[0] - R_far
    out_ev.terminal = True; out_ev.direction = 1
    T = 4.0 * R_far / max(omega if not lattice else math.cos(k), 1e-3)
    sol = solve_ivp(rhs_factory(F, 0.0, kappa, lattice), (0, T), [R_far * (1 - 1e-12), p0, 0.0], events=out_ev, rtol=1e-11, atol=1e-12,
                    method="DOP853", max_step=b)
    phi = float(sol.y[2, -1]); rmin = float(sol.y[0].min())
    tail = 2.0 * (2 * F.g * F.E / b) * (1.0 - math.sqrt(1.0 - (b / R_far) ** 2))
    return {"deflection": phi - math.pi + 2.0 * math.asin(b / R_far) + tail, "r_min": rmin, "tail_correction": tail, "status": int(sol.status)}


def radial_delay(F, k, lattice, r_out, r_in):
    omega = math.sin(k) if lattice else k

    def leg(flat):
        p0 = p_inward(F, r_out, omega, 0.0, 0.0, lattice, flat)
        ev = lambda t, y: y[0] - r_in
        ev.terminal = True; ev.direction = -1
        s = solve_ivp(rhs_factory(F, 0.0, 0.0, lattice, flat), (0, 50 * r_out), [r_out, p0, 0.0], events=ev, rtol=1e-11, atol=1e-12, method="DOP853")
        return float(s.t[-1])
    t_g, t_f = leg(False), leg(True)
    return {"t_inward": t_g, "t_flat": t_f, "delay": t_g - t_f}


def orbit_params(F, m, rp, ra):
    """the Schwarzschild bound orbit with turning points rp, ra: eps^2 = (1 - rg/r)(1 + l^2/r^2) at both (rg = 2 g E)."""
    rg = 2 * F.g * F.E
    A1, A2 = 1 - rg / rp, 1 - rg / ra
    l2 = (A1 - A2) / (A2 / ra ** 2 - A1 / rp ** 2)
    eps2 = A1 * (1 + l2 / rp ** 2)
    return math.sqrt(eps2) * m, math.sqrt(l2) * m


def precession(F, m, a_orb, e, lattice, norbits=6):
    rp, ra = a_orb * (1 - e), a_orb * (1 + e)
    omega, kappa = orbit_params(F, m, rp, ra)
    # start at the apoapsis with the momentum where dr/dt = 0 (in PG time p_r is not zero there): solve dH/dp = 0 near p = 0
    from scipy.optimize import brentq
    def drdt(p):
        return rhs_factory(F, m, kappa, lattice)(0.0, [ra, p, 0.0])[0]
    p_lo, p_hi = -0.5, 0.5
    p_a = brentq(drdt, p_lo, p_hi, xtol=1e-15)
    H0 = H_of(F, ra, p_a, m, kappa, lattice)
    peri = lambda t, y: rhs_factory(F, m, kappa, lattice)(t, y)[0]
    peri.direction = 1                                          # dr/dt crossing zero upward: a periapsis
    Tk = 2 * math.pi * math.sqrt(a_orb ** 3 / (F.g * F.E)) * (norbits + 1)
    sol = solve_ivp(rhs_factory(F, m, kappa, lattice), (0, Tk), [ra, p_a, 0.0], events=peri, rtol=1e-12, atol=1e-12, method="DOP853",
                    max_step=a_orb / 20)
    phis = sol.y_events[0][:, 2] if len(sol.t_events[0]) else np.array([])
    rps = sol.y_events[0][:, 0] if len(sol.t_events[0]) else np.array([])
    dphi = np.diff(phis) - 2 * math.pi if len(phis) > 1 else np.array([])
    pmax = float(np.max(np.abs(sol.y[1])))
    return {"omega": omega, "kappa": kappa, "H_at_start": H0, "H_drift": float(abs(H_of(F, sol.y[0, -1], sol.y[1, -1], m, kappa, lattice) - H0)),
            "periapses": rps.tolist(), "advance_per_orbit": dphi.tolist(), "advance_mean": float(np.mean(dphi)) if len(dphi) else None,
            "p_max": pmax, "orbits": int(len(phis))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--star", required=True)
    ap.add_argument("--member", type=int, default=None)
    ap.add_argument("--b-over-rg", type=float, nargs="+", default=[20.0, 50.0, 100.0])
    ap.add_argument("--k", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4])
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
    rec = {"script": "engine_ppn_v1.py", "star": os.path.basename(a.star), "member": i, "g": g, "E": E, "r_g": rg, "r_collar": M["r_collar"],
           "r_star": M.get("r_star"), "collar_jump_N_v": F.collar_jump, "deflection": [], "delay": [], "precession": []}
    L = ["# T7.1 (engine_ppn_v1.py): the matter's symbol on the solved exterior (member %d of %s; g = %g, E = %.6f, r_g = 2 g E = %.4f, r_collar = %g, r_star = %s)" % (
        i, os.path.basename(a.star), g, E, rg, M["r_collar"], M.get("r_star")), "",
        "| b/r_g | k a | deflection, lattice | geodesic (exact GR) | 4 g E/b | geodesic/(4gE/b) - 1 | lattice/geodesic - 1 | r_min/r_g |", "|---|---|---|---|---|---|---|---|"]
    for bo in a.b_over_rg:
        b = bo * rg
        if b < (M.get("r_star") or 0) * 1.5:
            L.append("| %g | - | (b inside 1.5 r_star: skipped) | | | | | |" % bo)
            continue
        for k in a.k:
            dl = deflection(F, b, k, True); dg = deflection(F, b, k, False)
            row = {"b_over_rg": bo, "k": k, "lattice": dl, "geodesic": dg, "newton_4gE_over_b": 4 * g * E / b}
            rec["deflection"].append(row)
            L.append("| %g | %g | %.8e | %.8e | %.8e | %.4e | %.4e | %.3f |" % (bo, k, dl["deflection"], dg["deflection"], 4 * g * E / b,
                     dg["deflection"] / (4 * g * E / b) - 1, dl["deflection"] / dg["deflection"] - 1, dl["r_min"] / rg))
            print(L[-1], flush=True)
    L += ["", "| k a | radial delay 20 r_g -> 2 r_g: lattice | geodesic | lattice/geodesic - 1 |", "|---|---|---|---|"]
    for k in a.k:
        dl = radial_delay(F, k, True, 20 * rg, max(2 * rg, 1.2 * (M.get("r_star") or 0)))
        dg = radial_delay(F, k, False, 20 * rg, max(2 * rg, 1.2 * (M.get("r_star") or 0)))
        rec["delay"].append({"k": k, "lattice": dl, "geodesic": dg})
        L.append("| %g | %.6f | %.6f | %.4e |" % (k, dl["delay"], dg["delay"], dl["delay"] / dg["delay"] - 1 if dg["delay"] else float("nan")))
        print(L[-1], flush=True)
    L += ["", "| m a | a_orb/r_g | e | advance per orbit, lattice | geodesic | 6 pi g E/(a (1 - e^2)) | lattice/geodesic - 1 | p_max a | orbits |", "|---|---|---|---|---|---|---|---|---|"]
    for mo in a.m_orbit:
        aorb = a.a_over_rg * rg
        if aorb * (1 - a.e) < 1.5 * (M.get("r_star") or 0):
            L.append("| %g | %g | %g | (periapsis inside 1.5 r_star: skipped) | | | | | |" % (mo, a.a_over_rg, a.e))
            continue
        pl = precession(F, mo, aorb, a.e, True); pg = precession(F, mo, aorb, a.e, False)
        rec["precession"].append({"m": mo, "a_over_rg": a.a_over_rg, "e": a.e, "lattice": pl, "geodesic": pg, "first_order": 6 * math.pi * g * E / (aorb * (1 - a.e ** 2))})
        L.append("| %g | %g | %g | %s | %s | %.6e | %s | %.4f | %d, %d |" % (mo, a.a_over_rg, a.e, "%.6e" % pl["advance_mean"] if pl["advance_mean"] is not None else "-",
                 "%.6e" % pg["advance_mean"] if pg["advance_mean"] is not None else "-", 6 * math.pi * g * E / (aorb * (1 - a.e ** 2)),
                 "%.4e" % (pl["advance_mean"] / pg["advance_mean"] - 1) if (pl["advance_mean"] and pg["advance_mean"]) else "-", pl["p_max"], pl["orbits"], pg["orbits"]))
        print(L[-1], flush=True)
    L += ["", "- the collar: |N - 1| %.2e and |v + sqrt(2 g E/r)| %.2e at r_collar (the solver's exterior against the analytic one)." % F.collar_jump,
          "- every number above is printed by the script; the geodesic symbol on the exact exterior is the general-relativistic reference."]
    os.makedirs(a.out, exist_ok=True)
    json.dump(rec, open(os.path.join(a.out, "ppn_%s.json" % a.tag), "w"), indent=1, default=float)
    open(os.path.join(a.out, "ppn_%s.md" % a.tag), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
