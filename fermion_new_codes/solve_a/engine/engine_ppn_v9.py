"""engine_ppn_v9.py (builder analyst; version 9 per SPEC_final4b_engine.md sec. 15.3): (1) rays per direction at b/r_g in {100, 200, 400, 800}
(one far-radius setting, R_far/b = 2000, version 8's ladder having agreed to four digits), the fit C_0 + C_1/2 sqrt(r_g/b) + C_1 r_g/b with C_0 fixed
at -c/2 - 2 t_22 and C_1/2 at the red team's -(I/2)(4 t_31 + t_13), I = int z^2 dz/(1 + z^2)^(9/4), printed per direction, C_1 fitted, the
residual; the variant with C_1/2 free; (2) the radial round trip at r_E/r_g in {20, 40, 80} with r_R = 10 r_E on [100], [110], [111]: the factor
against -3c; (3) the [011] orbit at e = 0.3 on the a_orb ladder (40, 160, 640 r_g), theta0 in {0, pi/4}; (4) the controls: the exit events, the
minimum radius against the collar, the orbit windows in orbital periods. Version 8's docstring follows.

engine_ppn_v8.py (builder analyst; version 8 of engine_ppn_v7.py after the reviewer B red team of versions 6 and 7
(reviewb/review_output/redteam_T7v67): (1) both symbols are launched at a MATCHED CONSERVED ENERGY omega with their FULL initial velocity (the
flow term included) along the start direction d (versions 6 and 7 aimed the flat group velocity, which at a point where the inflow is nonzero
tilts the lattice launch at order (k a)^2 sqrt(r_g/R) and produced most of the [111] out-of-plane displacement; the geodesic was launched at
|p| = omega, not at H = omega); the deflection ratio is then the plain angle ratio at matched (omega, d, b) (no b_eff normalization), with an
R_far ladder; the energy drift |H_end - H_start| is printed; (2) the delay's flat time uses the flat group speed at the conserved energy along d
(version 7 used the launch momentum's, which at the inflow differs at order (k a)^2 sqrt(r_g/R) and produced the sqrt(R_far) growth; the
builder's reading of version 7 was wrong) over the chord 2 sqrt(R_far^2 - b^2); (3) the red team's clean delay: the radial round trip between
fixed radii r_E and r_R along a lattice direction (outbound to r_R, a new inbound ray from r_R at the same omega), D = T - 2 (r_R - r_E)/s_omega,
the ratio D_lat/D_geo - 1 per (a omega)^2 printed beside the red team's -3 c with c = sum d_i^4; (4) the red team's asymptotic ray coefficient
-c/2 - 2 t_22 (c = sum d_i^4, t_22 = sum d_i^2 h_i^2, h the impact direction) printed beside each ray row; (5) the [111] periastron on an
a_orb/r_g ladder (--a-111) for the leading -(4/3) c = -2/3 and the red team's first PN correction -(4/3) c (mu/A)(91/(4(1 - e^2)) - 15/4).
Version 7's docstring follows.

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


def launch(F, X0, d, omega, lattice, m=0.0, iters=60):
    """version 8: the momentum at X0 with H(X0, p) = omega and the full velocity dX/dt along d."""
    pdir = d.copy()
    lam = omega
    for _ in range(iters):
        def H_lam(l):
            return H_of(F, np.concatenate([X0, l * pdir]), m, lattice) - omega
        lo, hi = 1e-9, (1.2 if lattice else 10.0 * omega + 1.0)
        while H_lam(hi) < 0 and hi > 1e-6:
            hi *= 0.8
        lam = brentq(H_lam, lo, hi, xtol=1e-15)
        vel = make_rhs(F, m, lattice)(0.0, np.concatenate([X0, lam * pdir]))[:3]
        vh = vel / np.linalg.norm(vel)
        if np.linalg.norm(vh - d) < 1e-14:
            break
        pdir = pdir + (d - vh)
        pdir = pdir / np.linalg.norm(pdir)
    return lam * pdir


def flat_speed_at_energy(omega, d, lattice, m=0.0):
    """the flat group speed along d at energy omega (the flat aiming of d, then the energy root)."""
    if not lattice:
        return 1.0 if m == 0 else math.sqrt(max(omega ** 2 - m * m, 0.0)) / omega
    p = launch(Fflat, np.zeros(3) + 1e9, d, omega, True, m)
    return float(np.linalg.norm(make_rhs(Fflat, m, True)(0.0, np.concatenate([np.zeros(3) + 1e9, p]))[:3]))


def deflection8(F, b, omega, theta0, nrm, e1, lattice, R_far_over_b):
    e2 = np.cross(nrm, e1)
    d = math.cos(theta0) * e1 + math.sin(theta0) * e2
    nperp = -math.sin(theta0) * e1 + math.cos(theta0) * e2
    R_far = R_far_over_b * b
    X0 = (-math.sqrt(R_far ** 2 - b * b) * d + b * nperp) * (1 - 1e-12)
    p = launch(F, X0, d, omega, lattice)
    y0 = np.concatenate([X0, p])
    H0 = H_of(F, y0, 0.0, lattice)
    v0 = make_rhs(F, 0.0, lattice)(0.0, y0)[:3]
    ev = lambda t, y: float(np.linalg.norm(y[:3])) - R_far
    ev.terminal = True; ev.direction = 1
    sol = solve_ivp(make_rhs(F, 0.0, lattice), (0, 4.0 * R_far / max(np.linalg.norm(v0), 1e-3)), y0, events=ev, rtol=1e-11, atol=1e-12,
                    method="DOP853", max_step=b)
    v1 = make_rhs(F, 0.0, lattice)(0.0, sol.y[:, -1])[:3]
    ang = math.atan2(float(np.linalg.norm(np.cross(d, v1))), float(d @ v1))
    tail = (4 * F.g * F.E / b) * (1.0 - math.sqrt(1.0 - (b / R_far) ** 2))
    T_exit = float(sol.t_events[0][0]) if len(sol.t_events[0]) else float(sol.t[-1])
    s_om = flat_speed_at_energy(omega, d, lattice)
    T_flat = 2.0 * math.sqrt(R_far ** 2 - b * b) / s_om
    oop = float(np.max(np.abs(sol.y[:3].T @ nrm)))
    c4 = float(np.sum(d ** 4)); t22 = float(np.sum(d ** 2 * nperp ** 2))
    return {"deflection": ang + tail, "omega": omega, "H_drift": abs(H_of(F, sol.y[:, -1], 0.0, lattice) - H0), "out_of_plane_max": oop,
            "status": int(sol.status), "T_exit": T_exit, "s_omega": s_om, "delay": T_exit - T_flat, "R_far": R_far, "c4": c4, "t22": t22,
            "redteam_coefficient": -c4 / 2 - 2 * t22}


def radial_round_trip(F, rE, rR, dr, omega, lattice):
    """the red team's clean delay: outbound from r_E to r_R along dr, inbound from r_R to r_E at the same omega; D = T - 2 (r_R - r_E)/s_omega."""
    T = 0.0
    for (r0, r1, sgn) in ((rE, rR, +1), (rR, rE, -1)):
        X0 = r0 * dr
        p = launch(F, X0, sgn * dr, omega, lattice)
        y0 = np.concatenate([X0, p])
        ev = lambda t, y, r1=r1: float(np.linalg.norm(y[:3])) - r1
        ev.terminal = True; ev.direction = 1 if sgn > 0 else -1
        v0 = make_rhs(F, 0.0, lattice)(0.0, y0)[:3]
        sol = solve_ivp(make_rhs(F, 0.0, lattice), (0, 4.0 * abs(r1 - r0) / max(np.linalg.norm(v0), 1e-3)), y0, events=ev, rtol=1e-12, atol=1e-12,
                        method="DOP853", max_step=max(abs(r1 - r0) / 200, 1.0))
        T += float(sol.t_events[0][0]) if len(sol.t_events[0]) else float("nan")
    s_om = flat_speed_at_energy(omega, dr, lattice)
    return {"T": T, "s_omega": s_om, "D": T - 2.0 * (rR - rE) / s_om}


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


def I_quad():
    from scipy.integrate import quad
    return quad(lambda z: z * z / (1 + z * z) ** 2.25, -np.inf, np.inf)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--star", required=True)
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--normals", default="001,011,111")
    ap.add_argument("--b-over-rg", type=float, nargs="+", default=[100.0, 200.0, 400.0, 800.0])
    ap.add_argument("--aomega", type=float, default=0.05)
    ap.add_argument("--theta0", type=float, nargs="+", default=[0.0, math.pi / 8, math.pi / 4])
    ap.add_argument("--Rfar-over-b", type=float, default=2000.0)
    ap.add_argument("--rE-over-rg", type=float, nargs="+", default=[20.0, 40.0, 80.0])
    ap.add_argument("--m-orbit", type=float, default=0.1)
    ap.add_argument("--a-011", type=float, nargs="+", default=[40.0, 160.0, 640.0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    D = json.load(open(a.star))
    M = D["members"][a.member]
    R = np.arange(M["nsites"], dtype=float) + (0.5 if D["args"].get("half_offset") else 1.0)
    g, E = M["g"], M["E"]
    F = Fields(R, np.array(M["N"]), np.array(M["v"]), E, g, M["r_collar"])
    rg = 2 * g * E
    Iq = I_quad()
    aw = a.aomega
    rec = {"script": "engine_ppn_v9.py", "star": os.path.basename(a.star), "member": a.member, "g": g, "E": E, "r_g": rg, "r_collar": M["r_collar"], "I": Iq,
           "rays": [], "fits": [], "round_trips": [], "orbits_011": []}
    L = ["# T7.1 version 9 (engine_ppn_v9.py; member %d of %s; r_g %.4f; the collar %.1f; a omega %g; I = %.6f)" % (a.member, os.path.basename(a.star), rg, M["r_collar"], aw, Iq), "",
         "## Rays: (deflection lattice/geodesic - 1)/(a omega)^2 against b, and the fit C_0 + C_1/2 sqrt(r_g/b) + C_1 r_g/b", "",
         "| normal | theta0 | b/r_g | coefficient | min radius / collar | exit status | H drift (lat) |", "|---|---|---|---|---|---|---|"]
    for ns in a.normals.split(","):
        nrm, e1 = PLANES[ns]
        for th in a.theta0:
            e2 = np.cross(nrm, e1)
            d = math.cos(th) * e1 + math.sin(th) * e2
            h = -math.sin(th) * e1 + math.cos(th) * e2
            c4 = float(np.sum(d ** 4)); t22 = float(np.sum(d ** 2 * h ** 2)); t31 = float(np.sum(d ** 3 * h)); t13 = float(np.sum(d * h ** 3))
            C0 = -c4 / 2 - 2 * t22
            Chalf = -(Iq / 2) * (4 * t31 + t13)
            bs, cs = [], []
            for bo in a.b_over_rg:
                dl = deflection8(F, bo * rg, aw, th, nrm, e1, True, a.Rfar_over_b)
                dg = deflection8(F, bo * rg, aw, th, nrm, e1, False, a.Rfar_over_b)
                coef = (dl["deflection"] / dg["deflection"] - 1) / aw ** 2
                bs.append(bo); cs.append(coef)
                rec["rays"].append({"normal": ns, "theta0": th, "b_over_rg": bo, "coefficient": coef, "lattice": dl, "geodesic": dg})
                L.append("| %s | %.4f | %g | %.5f | %.2f | %d | %.1e |" % (ns, th, bo, coef, (bo * rg) / M["r_collar"], dl["status"], dl["H_drift"]))
                print(L[-1], flush=True)
            x = np.array([math.sqrt(1.0 / b) for b in bs]); y = np.array(cs)
            C1 = float(np.linalg.lstsq((x ** 2)[:, None], y - C0 - Chalf * x, rcond=None)[0][0])
            res = float(np.max(np.abs(y - C0 - Chalf * x - C1 * x ** 2)))
            A = np.vstack([x, x ** 2]).T
            ch_free, c1_free = np.linalg.lstsq(A, y - C0, rcond=None)[0]
            res_free = float(np.max(np.abs(y - C0 - ch_free * x - c1_free * x ** 2)))
            rec["fits"].append({"normal": ns, "theta0": th, "c4": c4, "t22": t22, "t31": t31, "t13": t13, "C0": C0, "C_half_redteam": Chalf, "C1_fit": C1,
                                "residual": res, "C_half_free": float(ch_free), "C1_free": float(c1_free), "residual_free": res_free})
    L += ["", "### The fits per direction", "", "| normal | theta0 | C_0 = -c/2 - 2 t_22 | C_1/2 (red team) | C_1 fitted | max residual | C_1/2 free | C_1 (free fit) | max residual |",
          "|---|---|---|---|---|---|---|---|---|"]
    for f in rec["fits"]:
        L.append("| %s | %.4f | %.5f | %.5f | %.4f | %.2e | %.5f | %.4f | %.2e |" % (f["normal"], f["theta0"], f["C0"], f["C_half_redteam"], f["C1_fit"], f["residual"],
                 f["C_half_free"], f["C1_free"], f["residual_free"]))
    L += ["", "## The radial round trip (r_R = 10 r_E): (D_lat/D_geo - 1)/(a omega)^2 against -3c", "",
          "| direction | c | r_E/r_g | D_geo/(2 r_g ln 10) | coefficient | factor against -3c |", "|---|---|---|---|---|---|"]
    for lab, dr in (("[100]", np.array([1.0, 0, 0])), ("[110]", np.array([1.0, 1.0, 0]) / math.sqrt(2)), ("[111]", np.array([1.0, 1.0, 1.0]) / math.sqrt(3))):
        c4 = float(np.sum(dr ** 4))
        for rE in a.rE_over_rg:
            rl = radial_round_trip(F, rE * rg, 10 * rE * rg, dr, aw, True)
            rgeo = radial_round_trip(F, rE * rg, 10 * rE * rg, dr, aw, False)
            q = (rl["D"] / rgeo["D"] - 1) / aw ** 2
            rec["round_trips"].append({"direction": lab, "c4": c4, "rE_over_rg": rE, "coefficient": q, "factor": q / (-3 * c4), "lattice": rl, "geodesic": rgeo})
            L.append("| %s | %.4f | %g | %.5f | %.4f | %.4f |" % (lab, c4, rE, rgeo["D"] / (2 * rg * math.log(10.0)), q, q / (-3 * c4)))
            print(L[-1], flush=True)
    L += ["", "## The [011] periastron at e = 0.3 on the a_orb ladder: (advance lattice/geodesic - 1)/(m a)^2 (m a = %g)" % a.m_orbit, "",
          "| a_orb/r_g | theta0 | coefficient | orbits | window in periods | out of plane max |", "|---|---|---|---|---|---|"]
    nrm, e1 = PLANES["011"]
    for A_ in a.a_011:
        for th in (0.0, math.pi / 4):
            pl = precession(F, a.m_orbit, A_ * rg, 0.3, th, nrm, e1, True)
            pg = precession(F, a.m_orbit, A_ * rg, 0.3, th, nrm, e1, False)
            q = (pl["advance_mean"] / pg["advance_mean"] - 1) / a.m_orbit ** 2 if (pl["advance_mean"] and pg["advance_mean"]) else None
            rec["orbits_011"].append({"a_over_rg": A_, "theta0": th, "coefficient": q, "lattice": pl, "geodesic": pg})
            L.append("| %g | %.4f | %s | %d | 7 | %.2e |" % (A_, th, ("%.5f" % q) if q is not None else "-", pl["orbits"], pl["out_of_plane_max"]))
            print(L[-1], flush=True)
    L += ["", "- every number above is printed by the script; C_0 and C_1/2 are the red team's (reviewb/review_output/redteam_T7v67), printed for comparison."]
    os.makedirs(a.out, exist_ok=True)
    json.dump(rec, open(os.path.join(a.out, "ppn9_%s.json" % a.tag), "w"), indent=1, default=float)
    open(os.path.join(a.out, "ppn9_%s.md" % a.tag), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
