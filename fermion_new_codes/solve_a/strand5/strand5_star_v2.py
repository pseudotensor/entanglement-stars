#!/usr/bin/env python
"""Strand 5 J1 (spec solve/specs/SPEC_strand5_kernel_star.md sec. 3 J1; builder analyst): the continuum control.

The rows of H_* in spherical symmetry at C = k Lap, k = 1/(2 pi G), b d_b e_b -> 0 (LEAD.tex sec. 8.2), with chi = N psi:
    Lap psi = -2 pi G psi^5 e(mu),   Lap chi = 2 pi G chi psi^4 (e + 6 P)(mu),   mu = mu_inf psi/chi
(the TF equilibrium N e'(n) = mu_inf; n = 0 where mu <= m), the free continuum TF EOS with g states per momentum (g = 16: two spin states
on each of eight cones): e = (g/2 pi^2) int_0^kF p^2 sqrt(m^2 + p^2) dp, n = g kF^3/(6 pi^2), P = n mu - e; units G = m = 1, k = 1/(2 pi);
exterior psi = 1 + A/r, chi = 1 - B/r.

Two solvers.
(1) Reference: the same equations as ODEs in r, shot from the centre with psi(0) = chi(0) = 1 and mu_inf = sqrt(1 + z_c^2) (k_F(0) = z_c m),
    DOP853 (rtol 1e-12) to the surface event mu = m, matched to the exact exterior there, and mapped to psi, chi -> 1 at infinity by the
    chart's two scalings (r -> r/alpha^2 with psi -> psi/alpha; N -> N alpha/beta, mu_inf likewise); the integrals ride as ODE components.
(2) The spec's solver: Newton on a radial grid r in [0, R_max], R_max = 50 r_s (r_s of the previous member), n nodes on the map
    r(s) = L s/(1 - beta s) (s uniform, r(0.6) = r_s, r(1) = R_max); finite-volume Laplacian with face areas 4 pi r_i r_{i+1} (exact on
    1 + D/r) and the Robin flux -4 pi R_max (f - 1) at R_max (the exterior 1 + D/r matched analytically); unknowns psi, chi on the grid and
    mu_inf; the extra row mu_inf = N(0) sqrt(1 + z_c^2); continuation in z_c from the dilute member (the initial guess there is the
    reference solution on the grid).
Per member: M = 2A, B, A/B - 1, r_s (isotropic, where mu = m), the areal R_s = r_s psi(r_s)^2, 2GM/R_s, N_B = int n psi^6 d^3x, mu_inf, N(0),
E_canonical = int N psi^6 e - k int grad psi . grad chi (exterior included) against 2A/G, the Komar-ADM pair int psi^5 e, int N psi^5 (e + 6P),
f_s = (r_s - A)/(r_s + A), r_s/A, mu_inf - (r_s - B)/(r_s + A) (the TF surface condition N(r_s) = mu_inf/m read on the exterior); the first law
(dM/dz_c)/(dN_B/dz_c) against mu_inf by centred steps (the grid solver on the member's own grid); the grid solver also sigma_min of the
row-scaled Newton Jacobian at fixed z_c and with the extra row replaced by fixed N_B.  The mass maximum by Brent's method on each solver.
Writes <out>/star_v2.json and <out>/star_v2_profiles.npz.
(version 2: the first face's conductance is pi r_1, the midpoint form 4 pi (r_1/2)^2/r_1, exact on quadratics; version 1's geometric-mean
area 4 pi r_0 r_1 vanished at r_0 = 0 and decoupled the centre node, so its grid Newton did not converge; the other faces keep 4 pi r_i r_{i+1}.)"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.special import binom

K = 1.0 / (2.0 * np.pi)
SER_E = [binom(0.5, j) / (2 * j + 3) for j in range(12)]     # int_0^x t^2 sqrt(1 + t^2) dt = sum c_j x^(2j+3)
SER_P = [binom(-0.5, j) / (2 * j + 5) for j in range(12)]    # int_0^x t^4/sqrt(1 + t^2) dt = sum c_j x^(2j+5)


def eos(mu, g):
    """Free continuum TF, m = 1: (kF, n, e, P, de/dmu, dn/dmu) for mu (array); mu <= 1 is vacuum."""
    mu = np.atleast_1d(np.asarray(mu, float))
    x = np.sqrt(np.clip(mu * mu - 1.0, 0.0, None))
    s = np.sqrt(1 + x * x)
    ash = np.arcsinh(x)
    Ie = (x * (2 * x * x + 1) * s - ash) / 8
    IP = (x * (2 * x * x - 3) * s + 3 * ash) / 8
    sm = x < 0.1
    if np.any(sm):
        xs = x[sm]
        x2 = xs * xs
        Ie[sm] = xs ** 3 * sum(c * x2 ** j for j, c in enumerate(SER_E))
        IP[sm] = xs ** 5 * sum(c * x2 ** j for j, c in enumerate(SER_P))
    e = g / (2 * np.pi ** 2) * Ie
    P = g / (6 * np.pi ** 2) * IP
    n = g * x ** 3 / (6 * np.pi ** 2)
    de = g / (2 * np.pi ** 2) * x * mu * mu
    dn = g / (2 * np.pi ** 2) * x * mu
    vac = mu <= 1.0
    for arr in (e, P, n, de, dn):
        arr[vac] = 0.0
    return x, n, e, P, de, dn


# ---------------------------------------------------------------- reference (shooting)
def shoot(zc, g, rtol=1e-12, atol=1e-15, keep=False):
    mu_inf = np.sqrt(1 + zc * zc)
    _, n0, e0, P0, _, _ = eos(mu_inf, g)
    e0, P0, n0 = float(e0[0]), float(P0[0]), float(n0[0])
    rsc = 1.0 / np.sqrt(2 * np.pi * e0)

    def rhs(r, y):
        psi, dpsi, chi, dchi = y[0], y[1], y[2], y[3]
        _, n, e, P, _, _ = eos(mu_inf * psi / chi, g)
        n, e, P = n[0], e[0], P[0]
        se = psi ** 5 * e
        sc = chi * psi ** 4 * (e + 6 * P)
        w = 4 * np.pi * r * r
        return [dpsi, -2 * dpsi / r - 2 * np.pi * se, dchi, -2 * dchi / r + 2 * np.pi * sc,
                w * se, w * sc, w * n * psi ** 6, w * chi * psi ** 5 * e, w * dpsi * dchi]

    r0 = 1e-6 * rsc
    a2, c2 = -2 * np.pi * e0 / 6, 2 * np.pi * (e0 + 6 * P0) / 6
    v0 = 4 * np.pi / 3 * r0 ** 3
    y0 = [1 + a2 * r0 ** 2, 2 * a2 * r0, 1 + c2 * r0 ** 2, 2 * c2 * r0, v0 * e0, v0 * (e0 + 6 * P0), v0 * n0, v0 * e0, 0.0]

    def surf(r, y):
        return mu_inf * y[0] - y[2]
    surf.terminal, surf.direction = True, -1
    sol = solve_ivp(rhs, (r0, 1e6 * rsc), y0, method="DOP853", rtol=rtol, atol=atol, events=surf, dense_output=keep)
    if sol.status != 1:
        raise RuntimeError("no surface: %s" % sol.message)
    R = float(sol.t_events[0][0])
    y = sol.y_events[0][0]
    psi, dpsi, chi, dchi, IA, IB, IN, IE, IW = y
    a, alpha = -R * R * dpsi, psi + R * dpsi
    b_, beta = R * R * dchi, chi + R * dchi
    c = alpha / beta
    A, B = a * alpha, b_ * alpha ** 2 / beta
    rs = R * alpha ** 2
    Rs = R * psi ** 2
    out = {"z_c": zc, "g": g, "A": A, "B": B, "M": 2 * A, "A_over_B_minus_1": A / B - 1, "r_s": rs, "R_s_areal": Rs,
           "compactness_2GM_over_Rs": 2 * 2 * A / Rs, "N_B": IN, "mu_inf": mu_inf * c, "N0": c,
           "E_canonical": c * (IE - K * (IW - 4 * np.pi * a * b_ / R)), "komar_adm_pair": [alpha * IA, alpha ** 2 * IB / beta],
           "gauss_check": [a / (IA / 2) - 1, b_ / (IB / 2) - 1], "f_s": (rs - A) / (rs + A), "r_s_over_A": rs / A,
           "tf_surface_check": mu_inf * c - (rs - B) / (rs + A), "nfev": int(sol.nfev)}
    out["E_canonical_minus_2A"] = out["E_canonical"] - 2 * A
    out["komar_adm_rel"] = out["komar_adm_pair"][0] / out["komar_adm_pair"][1] - 1
    if keep:
        out["_sol"], out["_scal"] = sol, (alpha, beta, R)
    return out


def shoot_profile(ref, r):
    """psi, chi of the reference on the rescaled radii r (exterior analytic)."""
    sol = ref["_sol"]
    alpha, beta, R = ref["_scal"]
    rr = r / alpha ** 2
    psi, chi = np.empty_like(r), np.empty_like(r)
    ins = rr <= R
    rin = np.clip(rr[ins], sol.t[0], R)
    Y = sol.sol(rin)
    psi[ins], chi[ins] = Y[0] / alpha, Y[2] / beta
    psi[~ins] = 1 + ref["A"] / r[~ins]
    chi[~ins] = 1 - ref["B"] / r[~ins]
    small = r < sol.t[0] * alpha ** 2
    psi[small], chi[small] = 1 / alpha, 1 / beta
    return psi, chi


# ---------------------------------------------------------------- grid solver
class Grid:
    def __init__(self, n, rs, q=50.0, s_in=0.6):
        Rmax = q * rs
        beta = (q * s_in - 1) / (s_in * (q - 1))
        L = Rmax * (1 - beta)
        s = np.linspace(0.0, 1.0, n)
        r = L * s / (1 - beta * s)
        r[-1] = Rmax
        self.n, self.r, self.Rmax, self.rs_design = n, r, Rmax, rs
        self.cond = 4 * np.pi * r[:-1] * r[1:] / (r[1:] - r[:-1])
        self.cond[0] = np.pi * r[1]
        edges = np.concatenate([[0.0], 0.5 * (r[:-1] + r[1:]), [Rmax]])
        self.V = 4 * np.pi / 3 * np.diff(edges ** 3)
        c = self.cond
        main = np.zeros(n)
        main[:-1] -= c
        main[1:] -= c
        main[-1] -= 4 * np.pi * Rmax
        self.L = sp.diags([c, main, c], [-1, 0, 1], shape=(n, n), format="csr")
        self.rob = np.zeros(n)
        self.rob[-1] = 4 * np.pi * Rmax

    def lap(self, f):
        return self.L @ f + self.rob


def residual(G, u, zc, g, NB_target=None):
    n = G.n
    psi, chi, mu_inf = u[:n], u[n:2 * n], u[2 * n]
    mu = mu_inf * psi / chi
    _, nn, e, P, de, dn = eos(mu, g)
    R1 = K * G.lap(psi) + G.V * psi ** 5 * e
    R2 = K * G.lap(chi) - G.V * chi * psi ** 4 * (e + 6 * P)
    if NB_target is None:
        R3 = mu_inf - chi[0] / psi[0] * np.sqrt(1 + zc * zc)
    else:
        R3 = float(np.sum(G.V * nn * psi ** 6)) - NB_target
    return np.concatenate([R1, R2, [R3]]), (mu, nn, e, P, de, dn)


def jacobian(G, u, zc, g, aux, NB_target=None):
    n = G.n
    psi, chi, mu_inf = u[:n], u[n:2 * n], u[2 * n]
    mu, nn, e, P, de, dn = aux
    mpsi, mchi, mmu = mu_inf / chi, -mu_inf * psi / chi ** 2, psi / chi
    V = G.V
    d11 = V * (5 * psi ** 4 * e + psi ** 5 * de * mpsi)
    d12 = V * psi ** 5 * de * mchi
    c1 = V * psi ** 5 * de * mmu
    w = e + 6 * P
    dw = de + 6 * nn
    d21 = -V * chi * (4 * psi ** 3 * w + psi ** 4 * dw * mpsi)
    d22 = -V * (psi ** 4 * w + chi * psi ** 4 * dw * mchi)
    c2 = -V * chi * psi ** 4 * dw * mmu
    if NB_target is None:
        s0 = np.sqrt(1 + zc * zc)
        rpsi, rchi, rmu = np.zeros(n), np.zeros(n), 1.0
        rpsi[0], rchi[0] = chi[0] / psi[0] ** 2 * s0, -s0 / psi[0]
    else:
        rpsi = V * (6 * psi ** 5 * nn + psi ** 6 * dn * mpsi)
        rchi = V * psi ** 6 * dn * mchi
        rmu = float(np.sum(V * psi ** 6 * dn * mmu))
    J = sp.bmat([[K * G.L + sp.diags(d11), sp.diags(d12), sp.csr_matrix(c1[:, None])],
                 [sp.diags(d21), K * G.L + sp.diags(d22), sp.csr_matrix(c2[:, None])],
                 [sp.csr_matrix(rpsi[None, :]), sp.csr_matrix(rchi[None, :]), sp.csr_matrix(np.array([[rmu]]))]], format="csc")
    return J


def newton(G, u0, zc, g, NB_target=None, maxit=60, tol=1e-14):
    u = u0.copy()
    hist = []
    for it in range(maxit):
        F, aux = residual(G, u, zc, g, NB_target)
        J = jacobian(G, u, zc, g, aux, NB_target)
        du = spla.splu(J).solve(-F)
        step = 1.0
        n = G.n
        while True:
            un = u + step * du
            if np.all(un[:2 * n] > 0) and un[2 * n] > 0:
                break
            step /= 2
            if step < 1e-6:
                return u, {"ok": False, "why": "positivity", "hist": hist}
        u = un
        dmax = float(np.max(np.abs(step * du[:2 * n])))
        hist.append(dmax)
        if dmax < tol:
            F, aux = residual(G, u, zc, g, NB_target)
            return u, {"ok": True, "it": it + 1, "hist": hist, "res_scaled": float(np.max(np.abs(F[:2 * n] / np.concatenate([G.V, G.V]))))}
        if it > 8 and dmax > 0.5 * hist[-2] and dmax < 1e-11:
            F, aux = residual(G, u, zc, g, NB_target)
            return u, {"ok": True, "it": it + 1, "hist": hist, "stopped": "floor",
                       "res_scaled": float(np.max(np.abs(F[:2 * n] / np.concatenate([G.V, G.V]))))}
    return u, {"ok": False, "why": "maxit", "hist": hist}


def sigma_min(G, u, zc, g, NB_target=None, iters=80):
    F, aux = residual(G, u, zc, g, NB_target)
    J = jacobian(G, u, zc, g, aux, NB_target)
    n = G.n
    sc = np.concatenate([1 / G.V, 1 / G.V, [1.0]])
    Js = (sp.diags(sc) @ J).tocsc()
    lu = spla.splu(Js)
    v = np.random.default_rng(1).normal(size=2 * n + 1)
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        w = lu.solve(lu.solve(v), trans="T")
        lam = np.linalg.norm(w)
        v = w / lam
    return float(1 / np.sqrt(lam))


def interp_to(G_old, u_old, G_new):
    n0 = G_old.n
    psi_s = CubicSpline(G_old.r, u_old[:n0])
    chi_s = CubicSpline(G_old.r, u_old[n0:2 * n0])
    A = G_old.Rmax * (u_old[n0 - 1] - 1)
    B = G_old.Rmax * (1 - u_old[2 * n0 - 1])
    r = G_new.r
    ins = r <= G_old.Rmax
    psi = np.where(ins, psi_s(np.minimum(r, G_old.Rmax)), 1 + A / np.maximum(r, 1e-300))
    chi = np.where(ins, chi_s(np.minimum(r, G_old.Rmax)), 1 - B / np.maximum(r, 1e-300))
    return np.concatenate([psi, chi, [u_old[2 * n0]]])


def diagnostics(G, u, g):
    n = G.n
    r = G.r
    psi, chi, mu_inf = u[:n], u[n:2 * n], u[2 * n]
    mu = mu_inf * psi / chi
    _, nn, e, P, _, _ = eos(mu, g)
    A = G.Rmax * (psi[-1] - 1)
    B = G.Rmax * (1 - chi[-1])
    gfun = mu - 1.0
    i = int(np.max(np.nonzero(gfun > 0)[0]))
    lo, hi = max(i - 1, 0), min(i + 3, n)
    idx = np.arange(lo, hi)
    rs = float(np.interp(0.0, gfun[i:i + 2][::-1], r[i:i + 2][::-1]))
    if len(idx) == 4:
        cs = CubicSpline(r[idx], gfun[idx])
        roots = [x for x in cs.roots(extrapolate=False) if r[i] <= x <= r[i + 1]]
        if roots:
            rs = float(roots[0])
    psi_rs = float(CubicSpline(r[idx], psi[idx])(rs)) if len(idx) == 4 else float(np.interp(rs, r, psi))
    Rs = rs * psi_rs ** 2
    dpsi = np.diff(psi)
    dchi = np.diff(chi)
    W2 = -K * float(np.sum(G.cond * dpsi * dchi)) + 4 * np.pi * K * A * B / G.Rmax
    Ecan = float(np.sum(G.V * chi * psi ** 5 * e)) + W2
    kom = [float(np.sum(G.V * psi ** 5 * e)), float(np.sum(G.V * chi * psi ** 4 * (e + 6 * P)))]
    # charges read at a second radius, 10 r_s (C3)
    j = int(np.searchsorted(r, 10 * rs))
    A10 = r[j] * (psi[j] - 1)
    B10 = r[j] * (1 - chi[j])
    return {"A": A, "B": B, "M": 2 * A, "A_over_B_minus_1": A / B - 1, "r_s": rs, "R_s_areal": Rs, "compactness_2GM_over_Rs": 4 * A / Rs,
            "N_B": float(np.sum(G.V * nn * psi ** 6)), "mu_inf": mu_inf, "N0": chi[0] / psi[0], "E_canonical": Ecan,
            "E_canonical_minus_2A": Ecan - 2 * A, "komar_adm_pair": kom, "komar_adm_rel": kom[0] / kom[1] - 1,
            "A_vol_minus_A_tail": kom[0] / 2 - A, "f_s": (rs - A) / (rs + A), "r_s_over_A": rs / A,
            "tf_surface_check": mu_inf - (rs - B) / (rs + A), "charges_at_10rs": [A10, B10, A10 / A - 1, B10 / B - 1],
            "R_max_over_r_s": G.Rmax / rs, "nodes_inside": int(i + 1)}


def solve_member(G, u0, zc, g):
    u, info = newton(G, u0, zc, g)
    if info["ok"]:
        return u, info
    return None, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--g", type=float, default=16.0)
    ap.add_argument("--nmembers", type=int, default=60)
    ap.add_argument("--zmin", type=float, default=0.05)
    ap.add_argument("--zmax", type=float, default=3.0)
    ap.add_argument("--grids", default="2000,4000,8000")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    g = a.g
    zs = np.geomspace(a.zmin, a.zmax, a.nmembers)
    res = {"meta": {"g": g, "units": "G = m = 1, k = 1/(2 pi)", "z_c": zs.tolist(), "grids": a.grids, "R_max_over_r_s_design": 50.0,
                    "map": "r(s) = L s/(1 - beta s), r(0.6) = r_s(previous member), r(1) = 50 r_s(previous member)",
                    "M0_mapping": "M_0 = m_P^3/(sqrt(nu) m^2) with nu = g/2; in G = m = 1 units M_0 = (g/2)^(-1/2); lengths in the same unit"},
           "reference": [], "grid": {}, "controls": {}}
    # reference family
    for zc in zs:
        r = shoot(float(zc), g)
        dz = 1e-5 * zc
        rp, rm = shoot(float(zc + dz), g), shoot(float(zc - dz), g)
        r["first_law"] = {"dM_dz": (rp["M"] - rm["M"]) / (2 * dz), "dNB_dz": (rp["N_B"] - rm["N_B"]) / (2 * dz)}
        fl = r["first_law"]
        fl["dM_minus_mu_dNB_rel"] = (fl["dM_dz"] - r["mu_inf"] * fl["dNB_dz"]) / (abs(fl["dM_dz"]) + r["mu_inf"] * abs(fl["dNB_dz"]))
        res["reference"].append(r)
    print("reference family done (%.0f s)" % (time.time() - t0), flush=True)
    # reference maximum, and its tolerance check
    zM = [r["M"] for r in res["reference"]]
    i = int(np.argmax(zM))
    br = (zs[max(i - 1, 0)], zs[i], zs[min(i + 1, len(zs) - 1)])
    opt = minimize_scalar(lambda z: -shoot(z, g)["M"], bracket=br, tol=1e-12)
    rmax = shoot(float(opt.x), g)
    rmax10 = shoot(float(opt.x), g, rtol=1e-10, atol=1e-13)
    sq = np.sqrt(g / 2)
    rmax["in_P16_units"] = {"M_over_M0": rmax["M"] * sq, "R_s_over_R0": rmax["R_s_areal"] * sq, "m_NB_over_M0": rmax["N_B"] * sq,
                            "mu_inf_over_m": rmax["mu_inf"], "compactness": rmax["compactness_2GM_over_Rs"], "z_c": float(opt.x)}
    rmax["rtol_1e-10_differences"] = {k_: rmax10[k_] - rmax[k_] for k_ in ("M", "N_B", "R_s_areal", "mu_inf", "A_over_B_minus_1")}
    res["controls"]["reference_maximum"] = rmax
    res["controls"]["P16"] = {"z_c": 0.8339, "M_over_M0": 0.3843, "R_s_over_R0": 3.357, "compactness": 0.2289, "m_NB_over_M0": 0.3985,
                              "mu_inf_over_m": 0.8781}
    # g = 2 (the OV normalization) for the mapping
    opt2 = minimize_scalar(lambda z: -shoot(z, 2.0)["M"], bracket=br, tol=1e-12)
    r2 = shoot(float(opt2.x), 2.0)
    res["controls"]["reference_maximum_g2"] = {"z_c": float(opt2.x), "M": r2["M"], "R_s_areal": r2["R_s_areal"], "N_B": r2["N_B"],
                                               "mu_inf": r2["mu_inf"], "compactness": r2["compactness_2GM_over_Rs"],
                                               "ratio_M_g16_over_g2_times_sqrt8": rmax["M"] / r2["M"] * np.sqrt(g / 2) if g != 2 else None}
    print("reference maximum z_c %.8f M %.10f (%.0f s)" % (opt.x, rmax["M"], time.time() - t0), flush=True)
    # grid solver: vacuum control (C1)
    n0 = int(a.grids.split(",")[0])
    Gv = Grid(n0, res["reference"][0]["r_s"])
    uv = np.concatenate([1 + 0.01 * np.exp(-(Gv.r / Gv.rs_design) ** 2), 1 - 0.01 * np.exp(-(Gv.r / Gv.rs_design) ** 2), [1.0]])
    uvs, iv = newton(Gv, uv, 0.0, g)
    res["controls"]["vacuum_grid"] = {"ok": iv["ok"], "max_abs_psi_minus_1": float(np.max(np.abs(uvs[:n0] - 1))),
                                      "max_abs_N_minus_1": float(np.max(np.abs(uvs[n0:2 * n0] / uvs[:n0] - 1))),
                                      "A": Gv.Rmax * (uvs[n0 - 1] - 1), "B": Gv.Rmax * (1 - uvs[2 * n0 - 1]), "mu_inf": float(uvs[-1]),
                                      "sigma_min_vacuum": sigma_min(Gv, uvs, 0.0, g)}
    prof = {}
    for ng in [int(x) for x in a.grids.split(",")]:
        rows = []
        ref0 = shoot(float(zs[0]), g, keep=True)
        G = Grid(ng, ref0["r_s"])
        ps, ch = shoot_profile(ref0, G.r)
        u = np.concatenate([ps, ch, [ref0["mu_inf"]]])
        G_prev = G
        for m_i, zc in enumerate(zs):
            rs_prev = rows[-1]["r_s"] if rows else ref0["r_s"]
            G = Grid(ng, rs_prev)
            u0 = interp_to(G_prev, u, G)
            us, info = newton(G, u0, float(zc), g)
            if not info["ok"] and m_i > 0:   # substeps from the previous member
                zprev = zs[m_i - 1]
                uu, ok = u0, True
                for zz in np.linspace(zprev, zc, 9)[1:]:
                    uu, inf2 = newton(G, uu, float(zz), g)
                    if not inf2["ok"]:
                        ok = False
                        break
                if not ok:
                    rows.append({"z_c": float(zc), "failed": True, "info": {k_: v_ for k_, v_ in info.items() if k_ != "hist"}})
                    break
                us, info = uu, inf2
            elif not info["ok"]:
                rows.append({"z_c": float(zc), "failed": True, "info": {k_: v_ for k_, v_ in info.items() if k_ != "hist"}})
                break
            d = diagnostics(G, us, g)
            d.update({"z_c": float(zc), "newton_it": info.get("it"), "res_scaled": info.get("res_scaled")})
            dz = 1e-4 * zc
            up, ip_ = newton(G, us, float(zc + dz), g)
            um, im_ = newton(G, us, float(zc - dz), g)
            if ip_["ok"] and im_["ok"]:
                dp, dm = diagnostics(G, up, g), diagnostics(G, um, g)
                dM, dN = (dp["M"] - dm["M"]) / (2 * dz), (dp["N_B"] - dm["N_B"]) / (2 * dz)
                d["first_law"] = {"dM_dz": dM, "dNB_dz": dN, "dM_minus_mu_dNB_rel": (dM - d["mu_inf"] * dN) / (abs(dM) + d["mu_inf"] * abs(dN))}
            d["sigma_min_fixed_zc"] = sigma_min(G, us, float(zc), g)
            d["sigma_min_fixed_NB"] = sigma_min(G, us, float(zc), g, NB_target=d["N_B"])
            ref = res["reference"][m_i]
            d["minus_reference_rel"] = {k_: d[k_] / ref[k_] - 1 for k_ in ("M", "N_B", "r_s", "R_s_areal", "mu_inf", "N0")}
            rows.append(d)
            if m_i in (0, len(zs) // 2, len(zs) - 1) or abs(zc - 0.8339) < 0.05:
                prof["n%d_z%.4f_r" % (ng, zc)] = G.r
                prof["n%d_z%.4f_psi" % (ng, zc)] = us[:ng]
                prof["n%d_z%.4f_N" % (ng, zc)] = us[ng:2 * ng] / us[:ng]
            u, G_prev = us, G
        # grid maximum (fixed grid built at the reference maximum's r_s; warm start from the nearest member)
        Gm = Grid(ng, rmax["r_s"])
        ok_rows = [r_ for r_ in rows if not r_.get("failed")]
        ref_n = shoot(float(opt.x), g, keep=True)
        ps, ch = shoot_profile(ref_n, Gm.r)
        cache = {"u": np.concatenate([ps, ch, [ref_n["mu_inf"]]])}

        def negM(z):
            uu, inf_ = newton(Gm, cache["u"], float(z), g)
            if not inf_["ok"]:
                return 0.0
            cache["u"] = uu
            return -diagnostics(Gm, uu, g)["M"]
        gmax = None
        if cache["u"] is not None:
            o = minimize_scalar(negM, bracket=(opt.x * 0.97, opt.x, opt.x * 1.03), tol=1e-10)
            uu, _ = newton(Gm, cache["u"], float(o.x), g)
            dd = diagnostics(Gm, uu, g)
            gmax = {"z_c": float(o.x), **dd, "in_P16_units": {"M_over_M0": dd["M"] * sq, "R_s_over_R0": dd["R_s_areal"] * sq,
                                                              "m_NB_over_M0": dd["N_B"] * sq, "mu_inf_over_m": dd["mu_inf"],
                                                              "compactness": dd["compactness_2GM_over_Rs"]}}
        res["grid"][str(ng)] = {"members": rows, "maximum": gmax}
        print("grid %d done: %d members (%.0f s)" % (ng, len(ok_rows), time.time() - t0), flush=True)
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "star_v2.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    np.savez_compressed(os.path.join(a.out, "star_v2_profiles.npz"), **prof)
    print("done %.0f s" % res["meta"]["seconds"], flush=True)


if __name__ == "__main__":
    main()
