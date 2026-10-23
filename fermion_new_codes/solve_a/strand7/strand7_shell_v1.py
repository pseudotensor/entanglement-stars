#!/usr/bin/env python
"""Strand 7 S7a (SPEC_strand7_collapse.md secs. 1 to 3; builder analyst): the classical shell with quasi-static constraints from the covariant rows.

Units (sec. 1.1): k = 1/(2 pi) (G = 1), the marker mass m = 1, the grid spacing h; r_s := 2 G E_0 = 2 E_0.  Fields psi, chi = N psi on the uniform
nodes r_i = i h, i = 0..n-1 (the centre node r_0 = 0 kept, as J1's grid; R_max = (n - 1) h = 64 r_s), unknowns u = psi - 1, v = chi - 1, J1's
finite-volume Laplacian (strand5_star_v4.Grid's conventions on the uniform grid: face conductances 4 pi r_i r_(i+1)/h, the first face pi r_1, cell
volumes from the midpoint edges) with the Robin exterior d + r d' = 0 at R_max.
Rows at each instant (sec. 1.2), with S1_i := sum over markers of nu W w_i and S2_i := sum of nu (-2 N psi^-5 P^2/W) w_i (w_i the marker's
cloud-in-cell weight on node i, linear in r: the node's share of the volume integrals of J_N and J_psi):
    R1 = k L u + S1/psi = 0,     R2 = k L v - (chi S1/psi^2 - S2) = 0      (L the volume-integrated Laplacian; N = chi/psi at the nodes);
charges A = (1/(4 pi k)) sum S1/psi, B = (1/(4 pi k)) sum (chi S1/psi^2 - S2) (Gauss) against the tails A = R_max u(R_max), B = -R_max v(R_max);
E_bdy = 4 pi k A = 2A; R_trap = 4A; beside it the areal radius of the exterior form's lapse zero, (A + B)^2/B (equal to 4A at A = B).
The solve (sec. 1.5; the builder's choice, stated): a lagged source.  The markers' W, N and psi^-5 at the current fields give S1, S2; the rows are
solved by Newton with S1, S2 held (the nodes' 1/psi and chi/psi^2 in the Jacobian); the sources are re-evaluated at the new fields; the outer loop
repeats until the fields change by less than 1e-12 (or stall below 1e-10; at most 60 passes); the residual with the sources at the final fields is
reported (M7).  Fields at the markers by cubic splines of u and v (zero slope at r = 0).
Markers (secs. 1.3, 1.4): positions r, radial momenta p, angular momenta j (conserved), weights nu = E_0/(N_p m); H = N W, W = sqrt(m^2 + psi^-4 P^2),
P^2 = p^2 + j^2/r^2; r_dot = N psi^-4 p/W, p_dot = F := -[N' W + N (psi^-4)' P^2/(2W) - N psi^-4 j^2/(r^3 W)].  The step: Stoermer-Verlet in its
form for a non-separable H (Hairer, Lubich and Wanner, Geometric Numerical Integration, the generalized leapfrog; symplectic, symmetric, second
order), the fields held over the step: p_half = p + (dt/2) F(r, p_half) (implicit), r_new = r + (dt/2) [v(r, p_half) + v(r_new, p_half)]
(implicit), p_new = p_half + (dt/2) F(r_new, p_half); the implicit stages by fixed-point iteration to 1e-14 relative; then the rows are re-solved.
Runs (--run): self (i), the self-consistent quasi-static run; held (iii), the held-charge diagnostic (LABELLED, not the law: each step's sources
rescaled by the common factor A(0)/A_unscaled, iterated with the solve, so A stays at A(0)); the frozen controls (ii), the markers in a static
background made from the t = 0 solution, in three forms because the spec's expectations need the exterior: frozen_schw, psi = 1 + A0/r, chi = 1 -
A0/r at every r (the static exterior of the charge A0, B set to A0: the form in which u_H -> 1, lambda -> 1/(4 A0), R -> 4 A0 and GAP_collapse 2.4's
law hold); frozen_ext, psi = 1 + A0/r, chi = 1 - B0/r with the t = 0 Gauss charges (B0 differs from A0 for dust released from rest; its lapse
vanishes at the areal radius (A0 + B0)^2/B0); frozen_full, the full t = 0 fields (the literal reading; inside its own t = 0 shell the field is
flat, so no freezing is expected); each by DOP853 (rtol 1e-11; the energy control C2) or by the run (i) stepper (--integrator verlet; the
stepper's own check).  C2 per marker: the energy H_i against H_i(0), and the exact law of a static background, R_dot = -(psi^2 + 2 r psi psi')
N psi^-2 sqrt(1 - N^2 (m^2 + j^2/R^2)/eps_i^2) with eps_i = H_i(0) (equal to GAP_collapse 2.4's -F sqrt(1 - F (m^2 + j^2/R^2)/eps^2), F = 1 - 4A/R,
at A = B), against the integrated R_dot, where N^2 > 0.01 and the kinetic fraction exceeds 1e-4.  static_check (C1): the static TF member of J1
at z_c (--zc) sampled by deterministic markers (Gauss-Legendre in r per cell, in |p_prop| over [0, k_F] and in cos theta; g = 16), their binned
sources against psi^6 e and -6 N psi^5 P binned by the same cloud-in-cell weights at the same radial points (like with like: the one-particle
formulas and the momentum sampling), against the finite-volume cell values (the deposition), and the rows re-solved with the marker sources
against J1's fields and charges.  vacuum (C5).
Initial state: N_p markers at the midpoints of N_p equal sub-intervals of [r_0 - w_r, r_0 + w_r], w_r = 2 (the spec's 2h at h = 1, held fixed
when h is halved), at rest (p = 0); case A j = 0, case B j = m v_perp r (v_perp = 0.3; the initial rows then carry the tangential motion's W
and J_psi, stated); r_0 by the secant method so that r_0 psi(r_0)^2 = R_0 on the self-consistent t = 0 fields.
Monitors (sec. 2) every 0.1 r_s of external clock: M1 A, B (Gauss and tail), E_bdy, A/A(0) - 1, B/A - 1, the lead's prediction G E_0/(2 R_sh) -
G E_0/R_0, the mean of u_i^2 (u_i = -r_dot_i psi^2/N); M2 N and N psi at the median radius r_sh and at the 99th-percentile radius; M3 c_+^(r) =
N psi^-2 and c_+^(R) = 2 r psi psi_dot + (psi^2 + 2 r psi psi') N psi^-2 at r_sh (psi_dot from the fields at the previous output), R_sh =
r_sh psi(r_sh)^2, R_trap = 4A, and the middle marker's R_dot and -R_dot/F (F = 1 - 4A/R_sh); M5 the first node with chi <= 0, N(r_sh)/N_ext(r_sh),
N_ext = (1 - B/r)/(1 + A/r); M6 sum nu H and its change; M7 per output interval the most outer passes, Newton iterations, fixed-point
iterations and the largest residual.  Kill flags (sec. 6, without the resolution floor, which the report applies from C3): D1 |A/A(0) - 1| >
0.02 while R_sh > 1.5 r_s, or > 0.05; D2 R_sh < 4A while c_+^(r)(r_sh) > 1e-3.  The run stops at T_end, at R_sh - 4A < 1e-6 r_s, at a Newton
failure, at a marker within h/2 of the centre, at a non-positive lapse at a marker, or (with --stop-at-kill 1) at a kill flag; the default
records the flags and runs on (the floor is known only after C3).  Writes <out>/s7a_<tag>.json (and <out>/s7a_<tag>_fields.npz with
--save-fields 1: u, v at every output time, for S7b)."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

K = 1.0 / (2 * math.pi)
MM = 1.0


class UGrid:
    def __init__(self, n, h):
        r = np.arange(n) * h
        self.n, self.h, self.r, self.Rmax = n, h, r, float(r[-1])
        c = 4 * np.pi * r[:-1] * r[1:] / (r[1:] - r[:-1])
        c[0] = np.pi * r[1]
        edges = np.concatenate([[0.0], 0.5 * (r[:-1] + r[1:]), [r[-1]]])
        self.V = 4 * np.pi / 3 * np.diff(edges ** 3)
        main = np.zeros(n)
        main[:-1] -= c
        main[1:] -= c
        main[-1] -= 4 * np.pi * r[-1]
        self.L = sp.diags([c, main, c], [-1, 0, 1], shape=(n, n), format="csr")


class FieldInterp:
    def __init__(self, G, u, v, ext=None):
        self.G = G
        self.su = CubicSpline(G.r, u, bc_type=((1, 0.0), "not-a-knot"))
        self.sv = CubicSpline(G.r, v, bc_type=((1, 0.0), "not-a-knot"))
        self.A = float(G.Rmax * u[-1])
        self.B = float(-G.Rmax * v[-1])
        self.ext = ext            # (A, B): the exterior form at every r

    def at(self, r):
        r = np.asarray(r, float)
        if self.ext is not None:
            A0, B0 = self.ext
            psi = 1 + A0 / r
            chi = 1 - B0 / r
            dpsi = -A0 / r ** 2
            dchi = B0 / r ** 2
        else:
            ins = r <= self.G.Rmax
            rc = np.minimum(r, self.G.Rmax)
            psi = np.where(ins, 1 + self.su(rc), 1 + self.A / r)
            chi = np.where(ins, 1 + self.sv(rc), 1 - self.B / r)
            dpsi = np.where(ins, self.su(rc, 1), -self.A / r ** 2)
            dchi = np.where(ins, self.sv(rc, 1), self.B / r ** 2)
        N = chi / psi
        dN = (dchi * psi - chi * dpsi) / psi ** 2
        return psi, chi, N, dpsi, dN


def marker_terms(fi, r, p, j):
    psi, chi, N, dpsi, dN = fi.at(r)
    P2 = p * p + (j / r) ** 2
    W = np.sqrt(MM * MM + psi ** -4 * P2)
    return psi, chi, N, dpsi, dN, P2, W


def cic(G, r, vals):
    x = r / G.h
    i = np.clip(np.floor(x).astype(int), 0, G.n - 2)
    w = x - i
    return np.bincount(i, weights=(1 - w) * vals, minlength=G.n) + np.bincount(i + 1, weights=w * vals, minlength=G.n)


def sources(G, fi, r, p, j, nu):
    psi, chi, N, dpsi, dN, P2, W = marker_terms(fi, r, p, j)
    return cic(G, r, nu * W), cic(G, r, nu * (-2 * N * psi ** -5 * P2 / W))


def newton_rows(G, S1, S2, u, v, tol=1e-13, maxit=50):
    n = G.n
    hist = []
    for it in range(maxit):
        psi, chi = 1 + u, 1 + v
        R1 = K * (G.L @ u) + S1 / psi
        R2 = K * (G.L @ v) - (chi * S1 / psi ** 2 - S2)
        J = sp.bmat([[K * G.L + sp.diags(-S1 / psi ** 2), None], [sp.diags(2 * chi * S1 / psi ** 3), K * G.L + sp.diags(-S1 / psi ** 2)]], format="csc")
        d = spla.spsolve(J, -np.concatenate([R1, R2]))
        u, v = u + d[:n], v + d[n:]
        dm = float(np.max(np.abs(d)))
        hist.append(dm)
        if dm < tol or (len(hist) >= 4 and dm < 1e-11 and dm > 0.25 * min(hist[-4:-1])):
            return u, v, it + 1, True
    return u, v, maxit, False


def quasi_static(G, r, p, j, nu, u, v, hold_A=None, fac0=1.0, tol=1e-12, maxout=60):
    """the lagged-source loop; hold_A: the held-charge diagnostic's A (the sources rescaled by the common factor A(0)/A_unscaled)."""
    fac = fac0
    nits, okn, prev, conv = 0, True, None, False
    for outer in range(maxout):
        fi = FieldInterp(G, u, v)
        S1, S2 = sources(G, fi, r, p, j, nu)
        if hold_A is not None:
            Aun = float(np.sum(S1 / (1 + u)) / (4 * np.pi * K))
            if Aun > 0:
                fac = hold_A / Aun
        un, vn, it, okn = newton_rows(G, fac * S1, fac * S2, u, v)
        nits += it
        ch = float(max(np.max(np.abs(un - u)), np.max(np.abs(vn - v))))
        u, v = un, vn
        if ch < tol or (prev is not None and ch < 1e-10 and ch > 0.5 * prev):
            conv = True
            break
        prev = ch
    fi = FieldInterp(G, u, v)
    S1, S2 = sources(G, fi, r, p, j, nu)
    psi, chi = 1 + u, 1 + v
    if hold_A is not None:
        Aun = float(np.sum(S1 / psi) / (4 * np.pi * K))
        fac = hold_A / Aun if Aun > 0 else fac
    S1, S2 = fac * S1, fac * S2
    R1 = K * (G.L @ u) + S1 / psi
    R2 = K * (G.L @ v) - (chi * S1 / psi ** 2 - S2)
    res = float(max(np.max(np.abs(R1 / G.V)), np.max(np.abs(R2 / G.V))))
    Ag = float(np.sum(S1 / psi) / (4 * np.pi * K))
    Bg = float(np.sum(chi * S1 / psi ** 2 - S2) / (4 * np.pi * K))
    return u, v, {"outer": outer + 1, "outer_converged": conv, "newton": nits, "ok": okn, "residual": res, "A_gauss": Ag, "B_gauss": Bg, "factor": fac}


def force(fi, r, p, j):
    psi, chi, N, dpsi, dN, P2, W = marker_terms(fi, r, p, j)
    dpm4 = -4 * psi ** -5 * dpsi
    return -(dN * W + N * dpm4 * P2 / (2 * W) - N * psi ** -4 * j * j / (r ** 3 * W))


def velocity(fi, r, p, j):
    psi, chi, N, dpsi, dN, P2, W = marker_terms(fi, r, p, j)
    return N * psi ** -4 * p / W


def verlet_step(fi, r, p, j, dt, tol=1e-14, maxit=60):
    """the generalized leapfrog for non-separable H, fields held; returns r, p and the two stages' fixed-point counts."""
    ph = p + 0.5 * dt * force(fi, r, p, j)
    for k1 in range(maxit):
        pn = p + 0.5 * dt * force(fi, r, ph, j)
        d = float(np.max(np.abs(pn - ph)))
        ph = pn
        if d <= tol * (1 + float(np.max(np.abs(ph)))):
            break
    v0 = velocity(fi, r, ph, j)
    rn = r + dt * v0
    for k2 in range(maxit):
        rr = r + 0.5 * dt * (v0 + velocity(fi, rn, ph, j))
        d = float(np.max(np.abs(rr - rn)))
        rn = rr
        if d <= tol * (1 + float(np.max(np.abs(rn)))):
            break
    return rn, ph + 0.5 * dt * force(fi, rn, ph, j), max(k1, k2) + 1


def init_markers(Np, r0, wr, case, vperp):
    r = r0 - wr + (np.arange(Np) + 0.5) * (2 * wr / Np)
    p = np.zeros(Np)
    j = np.zeros(Np) if case == "A" else MM * vperp * r
    return r, p, j


def solve_initial(G, E0, R0, wr, Np, case, vperp):
    """r_0 by the secant method so that the self-consistent areal radius r_0 psi(r_0)^2 equals R0 (the shell at rest)."""
    nu = np.full(Np, E0 / (Np * MM))
    st = {"u": np.zeros(G.n), "v": np.zeros(G.n), "info": None}

    def Rof(r0):
        r, p, j = init_markers(Np, r0, wr, case, vperp)
        st["u"], st["v"], st["info"] = quasi_static(G, r, p, j, nu, st["u"], st["v"])
        psi = FieldInterp(G, st["u"], st["v"]).at(np.array([r0]))[0][0]
        return r0 * psi * psi
    a, b = R0 * 0.9, R0 * 0.8
    fa, fb = Rof(a), Rof(b)
    nsec = 0
    for nsec in range(1, 61):
        c = b - (fb - R0) * (b - a) / (fb - fa)
        fc = Rof(c)
        a, fa, b, fb = b, fb, c, fc
        if abs(fc - R0) < 1e-12 * R0:
            break
    r, p, j = init_markers(Np, b, wr, case, vperp)
    return r, p, j, nu, st["u"], st["v"], b, st["info"], {"secant_iterations": nsec, "R_of_r0_minus_R0": fb - R0}


def monitors(G, fi, fi_prev, dT, r, p, j, nu, A0, E0, R0, info, T, rs, eps=None):
    psi_m, chi_m, N_m, dpsi_m, dN_m, P2, W = marker_terms(fi, r, p, j)
    imid = int(np.argsort(r)[len(r) // 2])
    rsh = float(np.median(r))
    rout = float(np.percentile(r, 99))
    rin = float(np.percentile(r, 1))
    ps, ch, Nn, dps, dN = (float(x[0]) for x in fi.at(np.array([rsh])))
    pso, cho, Nno, _, _ = (float(x[0]) for x in fi.at(np.array([rout])))
    Rsh = rsh * ps * ps
    A, B = info["A_gauss"], info["B_gauss"]
    vel = N_m * psi_m ** -4 * p / W
    u2 = float(np.mean((-vel * psi_m ** 2 / N_m) ** 2))
    psidot, psidot_mid = None, 0.0
    if fi_prev is not None and dT > 0:
        psidot = float((ps - fi_prev.at(np.array([rsh]))[0][0]) / dT)
        psidot_mid = float((psi_m[imid] - fi_prev.at(np.array([r[imid]]))[0][0]) / dT)
    cpR = (2 * rsh * ps * psidot if psidot is not None else 0.0) + (ps * ps + 2 * rsh * ps * dps) * Nn / ps ** 2
    Rdot_mid = float((psi_m[imid] ** 2 + 2 * r[imid] * psi_m[imid] * dpsi_m[imid]) * vel[imid] + 2 * r[imid] * psi_m[imid] * psidot_mid)
    F_sh = 1 - 4 * A / Rsh
    chi_nodes = fi.at(np.maximum(G.r, 0.5 * G.h))[1]
    neg = np.nonzero(chi_nodes <= 0)[0]
    Next = (1 - B / rsh) / (1 + A / rsh)
    rec = {"T": T, "T_over_rs": T / rs, "A": A, "B": B, "A_tail": fi.A, "B_tail": fi.B, "E_bdy": 2 * A, "A_rel": A / A0 - 1,
           "B_over_A_minus_1": B / A - 1, "prediction": E0 / (2 * Rsh) - E0 / R0, "u2": u2, "r_sh": rsh, "r_p01": rin, "r_p99": rout, "R_sh": Rsh,
           "R_trap": 4 * A, "R_sh_minus_Rtrap": Rsh - 4 * A, "R_lapse0_ext": (A + B) ** 2 / B if B > 0 else None, "N_sh": Nn, "Npsi_sh": Nn * ps,
           "N_out": Nno, "Npsi_out": Nno * pso, "c_plus_r": Nn / ps ** 2, "c_plus_R": cpR, "psi_dot": psidot, "Rdot_mid": Rdot_mid,
           "uH_inst": -Rdot_mid / F_sh if F_sh != 0 else None, "F_sh": F_sh,
           "first_chi_nonpositive_node_r": float(G.r[neg[0]]) if len(neg) else None, "N_over_Next": Nn / Next if Next != 0 else None,
           "sum_nu_H": float(np.sum(nu * N_m * W)), "outer": info.get("outer"), "newton": info.get("newton"), "residual": info.get("residual"),
           "fp_iterations": info.get("fp"), "outer_converged": info.get("outer_converged"),
           "gauss_vs_tail_A": A / fi.A - 1 if fi.A else None, "gauss_vs_tail_B": B / fi.B - 1 if fi.B else None, "factor": info.get("factor")}
    if eps is not None:
        Rm = r * psi_m ** 2
        dRdr = psi_m ** 2 + 2 * r * psi_m * dpsi_m
        Rdot = dRdr * vel
        arg = 1 - N_m ** 2 * (MM * MM + j * j / Rm ** 2) / eps ** 2
        law = -dRdr * N_m / psi_m ** 2 * np.sqrt(np.maximum(arg, 0.0))
        sel = (N_m ** 2 > 0.01) & (arg > 1e-4)
        Fm = 1 - 4 * A / Rm
        lawS = -Fm * np.sqrt(np.maximum(1 - Fm * (MM * MM + j * j / Rm ** 2) / eps ** 2, 0.0))
        selS = (Fm > 0.01) & (arg > 1e-4)
        rec.update({"C2_law_max_rel": float(np.max(np.abs(Rdot[sel] / law[sel] - 1))) if np.any(sel) else None, "C2_law_n": int(np.sum(sel)),
                    "C2_schw_form_max_rel": float(np.max(np.abs(Rdot[selS] / lawS[selS] - 1))) if np.any(selS) else None,
                    "C2_schw_form_n": int(np.sum(selS)), "C2_marker_energy_max_rel": float(np.max(np.abs(N_m * W / eps - 1)))})
    return rec


def run_dynamic(a, G, E0, R0, case, mode):
    rs = 2 * E0
    r, p, j, nu, u, v, r0, info, sec = solve_initial(G, E0, R0, a.wr, a.Np, case, a.vperp)
    A0, B0 = info["A_gauss"], info["B_gauss"]
    dt = a.dt_over_rs * rs
    nout = max(1, int(round(0.1 * rs / dt)))
    Tend = a.T_over_rs * rs
    head = {"r0": r0, "A0": A0, "B0": B0, "B0_over_A0": B0 / A0, "R_trap0": 4 * A0, "R_lapse0_ext0": (A0 + B0) ** 2 / B0,
            "initial_solve": {k: info[k] for k in ("outer", "newton", "residual", "outer_converged")}, "secant": sec, "dt": dt, "n_out_steps": nout}
    stop, kills, t0 = "T_end", {}, time.time()
    if mode.startswith("frozen"):
        ext = {"frozen_schw": (A0, A0), "frozen_ext": (A0, B0), "frozen_full": None}[mode]
        fz = FieldInterp(G, u, v, ext=ext)
        infoz = {"A_gauss": A0, "B_gauss": A0 if mode == "frozen_schw" else B0, "outer": 0, "newton": 0, "residual": 0.0}
        psi_m, chi_m, N_m, _, _, _, W = marker_terms(fz, r, p, j)
        eps = N_m * W
        recs = [monitors(G, fz, None, 0.0, r, p, j, nu, A0, E0, R0, infoz, 0.0, rs, eps=eps)]
        Np = len(r)
        if a.integrator == "dop853":
            def rhs(t, y):
                return np.concatenate([velocity(fz, y[:Np], y[Np:], j), force(fz, y[:Np], y[Np:], j)])

            def ev_trap(t, y):
                rm = float(np.median(y[:Np]))
                ps = fz.at(np.array([rm]))[0][0]
                return rm * ps * ps - 4 * A0 - 1e-6 * rs
            ev_trap.terminal, ev_trap.direction = True, -1

            def ev_centre(t, y):
                return float(np.min(y[:Np])) - G.h
            ev_centre.terminal, ev_centre.direction = True, -1
            times = np.arange(1, int(round(Tend / (0.1 * rs))) + 1) * 0.1 * rs
            sol = solve_ivp(rhs, (0, times[-1]), np.concatenate([r, p]), method="DOP853", rtol=1e-11, atol=1e-12, t_eval=times,
                            events=[ev_trap, ev_centre])
            for k_, t in enumerate(sol.t):
                recs.append(monitors(G, fz, None, 0.0, sol.y[:Np, k_], sol.y[Np:, k_], j, nu, A0, E0, R0, infoz, float(t), rs, eps=eps))
            if sol.status == 1:
                stop = "R_sh - 4 A0 < 1e-6 r_s" if len(sol.t_events[0]) else "a marker reached r = h"
                ye = sol.y_events[0][0] if len(sol.t_events[0]) else sol.y_events[1][0]
                te = float(sol.t_events[0][0]) if len(sol.t_events[0]) else float(sol.t_events[1][0])
                recs.append(monitors(G, fz, None, 0.0, ye[:Np], ye[Np:], j, nu, A0, E0, R0, infoz, te, rs, eps=eps))
            elif sol.status != 0:
                stop = "integrator: " + sol.message
            head.update({"integrator": "DOP853 rtol 1e-11 atol 1e-12", "nfev": int(sol.nfev)})
        else:
            T, step = 0.0, 0
            while T < Tend - 1e-12:
                r, p, fp = verlet_step(fz, r, p, j, dt)
                T += dt
                step += 1
                if np.min(r) < 0.5 * G.h:
                    stop = "a marker reached r < h/2 at T = %g" % T
                    break
                if step % nout == 0:
                    infoz["fp"] = fp
                    m_ = monitors(G, fz, None, 0.0, r, p, j, nu, A0, E0, R0, infoz, T, rs, eps=eps)
                    recs.append(m_)
                    if m_["R_sh_minus_Rtrap"] < 1e-6 * rs:
                        stop = "R_sh - 4 A0 < 1e-6 r_s"
                        break
            head.update({"integrator": "generalized leapfrog, dt = %g" % dt, "steps": step})
    else:
        fi = FieldInterp(G, u, v)
        info["fp"] = 0
        recs = [monitors(G, fi, None, 0.0, r, p, j, nu, A0, E0, R0, info, 0.0, rs)]
        saved = {"T": [0.0], "u": [u.copy()], "v": [v.copy()]}
        fi_prev, T, step, fac = fi, 0.0, 0, 1.0
        hold = A0 if mode == "held" else None
        acc = {"outer": 0, "newton": 0, "residual": 0.0, "fp": 0}
        while T < Tend - 1e-12:
            r, p, fp = verlet_step(fi, r, p, j, dt)
            T += dt
            step += 1
            if np.min(r) < 0.5 * G.h:
                stop = "a marker reached r < h/2 at T = %g" % T
                break
            u, v, info = quasi_static(G, r, p, j, nu, u, v, hold_A=hold, fac0=fac)
            fac = info["factor"]
            fi = FieldInterp(G, u, v)
            for k_ in ("outer", "newton", "fp"):
                acc[k_] = max(acc[k_], info[k_] if k_ != "fp" else fp)
            acc["residual"] = max(acc["residual"], info["residual"])
            last = not info["ok"] or float(np.min(fi.at(r)[2])) <= 0
            if step % nout == 0 or last:
                info2 = dict(info, **acc)
                m_ = monitors(G, fi, fi_prev, (T - recs[-1]["T"]), r, p, j, nu, A0, E0, R0, info2, T, rs)
                recs.append(m_)
                fi_prev = fi
                acc = {"outer": 0, "newton": 0, "residual": 0.0, "fp": 0}
                if a.save_fields:
                    saved["T"].append(T)
                    saved["u"].append(u.copy())
                    saved["v"].append(v.copy())
                if (abs(m_["A_rel"]) > 0.02 and m_["R_sh"] > 1.5 * rs) or abs(m_["A_rel"]) > 0.05:
                    kills.setdefault("D1", {"T_over_rs": m_["T_over_rs"], "R_sh_over_rs": m_["R_sh"] / rs, "A_rel": m_["A_rel"]})
                if m_["R_sh"] < 4 * m_["A"] and m_["c_plus_r"] > 1e-3:
                    kills.setdefault("D2", {"T_over_rs": m_["T_over_rs"], "R_sh_over_rs": m_["R_sh"] / rs, "c_plus_r": m_["c_plus_r"]})
                if not info["ok"]:
                    stop = "Newton failure at T = %g" % T
                    break
                if last:
                    stop = "a non-positive lapse at a marker at T = %g" % T
                    break
                if m_["R_sh_minus_Rtrap"] < 1e-6 * rs:
                    stop = "R_sh - 4A < 1e-6 r_s"
                    break
                if kills and a.stop_at_kill:
                    stop = "kill " + ",".join(sorted(kills))
                    break
        head["steps"] = step
        if a.save_fields:
            np.savez_compressed(os.path.join(a.out, "s7a_%s_fields.npz" % a.tag), r=G.r, T=np.array(saved["T"]), u=np.array(saved["u"]),
                                v=np.array(saved["v"]))
    H0 = recs[0]["sum_nu_H"]
    for m_ in recs:
        m_["sum_nu_H_rel"] = m_["sum_nu_H"] / H0 - 1
    head.update({"records": recs, "stop": stop, "kills": kills, "seconds": time.time() - t0})
    return head


def static_check(a):
    """C1: the static TF member of J1 at z_c sampled by deterministic markers."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    sys.path.insert(1, os.path.join(here, "..", "strand5"))
    import strand5_star_v4 as STAR
    g = 16.0
    ref = STAR.shoot(a.zc, g, keep=True)
    rs_ = float(ref["r_s"])
    hh = rs_ / a.static_nodes
    G = UGrid(int(round(a.static_rmax * rs_ / hh)) + 1, hh)
    psi, chi = STAR.shoot_profile(ref, np.maximum(G.r, 1e-12))
    N = chi / psi
    mu_inf = float(ref["mu_inf"])
    _, _, e, P, _, _ = STAR.eos(mu_inf / N, g)
    xr, wr_ = np.polynomial.legendre.leggauss(4)
    xp, wp = np.polynomial.legendre.leggauss(a.static_p)
    xc, wc = np.polynomial.legendre.leggauss(a.static_c)
    edges = np.concatenate([[0.0], 0.5 * (G.r[:-1] + G.r[1:]), [G.r[-1]]])
    cells = np.nonzero(edges[:-1] < rs_)[0]
    rq = np.concatenate([0.5 * (edges[i + 1] - edges[i]) * xr + 0.5 * (edges[i + 1] + edges[i]) for i in cells])
    wq = np.concatenate([0.5 * (edges[i + 1] - edges[i]) * wr_ for i in cells])
    psq, chq = STAR.shoot_profile(ref, rq)
    Nq = chq / psq
    kfq, _, eq, Pq, _, _ = STAR.eos(mu_inf / Nq, g)
    keep = kfq > 0
    rq, wq, psq, Nq, kfq, eq, Pq = (x[keep] for x in (rq, wq, psq, Nq, kfq, eq, Pq))
    vol = 4 * np.pi * rq * rq * wq
    pk = 0.5 * kfq[:, None] * (xp[None, :] + 1)                     # (q, p)
    wpk = 0.5 * kfq[:, None] * wp[None, :]
    base = vol[:, None, None] * psq[:, None, None] ** 6 * g / (2 * np.pi) ** 3 * 2 * np.pi * (pk * pk * wpk)[:, :, None] * wc[None, None, :]
    Pc = (psq[:, None] ** 2 * pk)[:, :, None] * np.ones_like(xc)[None, None, :]
    R_ = np.broadcast_to(rq[:, None, None], base.shape).ravel()
    P_ = (Pc * xc[None, None, :]).ravel()
    J_ = (rq[:, None, None] * Pc * np.sqrt(1 - xc * xc)[None, None, :]).ravel()
    NU_ = base.ravel()
    fi = FieldInterp(G, psi - 1, chi - 1)
    S1, S2 = sources(G, fi, R_, P_, J_, NU_)
    ref1 = cic(G, rq, vol * psq ** 6 * eq)
    ref2 = cic(G, rq, vol * (-6 * Nq * psq ** 5 * Pq))
    nz = np.nonzero(ref1 > 0)[0]
    fv1 = G.V * psi ** 6 * e
    fv2 = G.V * (-6 * N * psi ** 5 * P)
    sel = np.nonzero((np.arange(G.n) >= 10) & (G.r < rs_ - 2 * hh))[0]
    u, v, info = quasi_static(G, R_, P_, J_, NU_, psi - 1, chi - 1)
    return {"z_c": a.zc, "r_s": rs_, "h": hh, "n": G.n, "R_max_over_rs": a.static_rmax, "markers": int(len(R_)),
            "like_with_like_max_rel_J_N": float(np.max(np.abs(S1[nz] / ref1[nz] - 1))),
            "like_with_like_max_rel_J_psi": float(np.max(np.abs(S2[nz] / ref2[nz] - 1))),
            "fv_max_rel_J_N_nodes_10_to_surface": float(np.max(np.abs(S1[sel] / fv1[sel] - 1))),
            "fv_max_rel_J_psi_nodes_10_to_surface": float(np.max(np.abs(S2[sel] / fv2[sel] - 1))),
            "total_rel_J_N_vs_quadrature": float(np.sum(S1) / np.sum(vol * psq ** 6 * eq) - 1),
            "total_rel_J_psi_vs_quadrature": float(np.sum(S2) / np.sum(vol * (-6 * Nq * psq ** 5 * Pq)) - 1),
            "total_rel_J_N_vs_fv": float(np.sum(S1) / np.sum(fv1) - 1), "total_rel_J_psi_vs_fv": float(np.sum(S2) / np.sum(fv2) - 1),
            "resolve_max_abs_dpsi": float(np.max(np.abs(u - (psi - 1)))), "resolve_max_abs_dchi": float(np.max(np.abs(v - (chi - 1)))),
            "resolve_A_rel": info["A_gauss"] / float(ref["A"]) - 1, "resolve_B_rel": info["B_gauss"] / float(ref["B"]) - 1,
            "resolve_info": {k: info[k] for k in ("outer", "newton", "residual", "outer_converged")}, "J1_A": float(ref["A"]), "J1_B": float(ref["B"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=["self", "held", "frozen_schw", "frozen_ext", "frozen_full", "static_check", "vacuum"])
    ap.add_argument("--integrator", default="dop853", choices=["dop853", "verlet"], help="the frozen runs only")
    ap.add_argument("--E0", type=float, default=8.0)
    ap.add_argument("--R0-over-rs", type=float, default=4.0)
    ap.add_argument("--case", default="A", choices=["A", "B"])
    ap.add_argument("--vperp", type=float, default=0.3)
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--Rmax-over-rs", type=float, default=64.0)
    ap.add_argument("--Np", type=int, default=10000)
    ap.add_argument("--wr", type=float, default=2.0)
    ap.add_argument("--dt-over-rs", type=float, default=0.01)
    ap.add_argument("--T-over-rs", type=float, default=40.0)
    ap.add_argument("--stop-at-kill", type=int, default=0)
    ap.add_argument("--save-fields", type=int, default=0)
    ap.add_argument("--zc", type=float, default=0.5)
    ap.add_argument("--static-nodes", type=int, default=200)
    ap.add_argument("--static-rmax", type=float, default=4.0)
    ap.add_argument("--static-p", type=int, default=12)
    ap.add_argument("--static-c", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rec = {"meta": dict(vars(a), K=K, m=MM, builder="analyst", script="strand7_shell_v1.py")}
    t0 = time.time()
    if a.run == "static_check":
        rec["static_check"] = static_check(a)
    elif a.run == "vacuum":
        rs = 2 * a.E0
        G = UGrid(int(round(a.Rmax_over_rs * rs / a.h)) + 1, a.h)
        u, v, info = quasi_static(G, np.array([1.0]), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(G.n), np.zeros(G.n))
        rec["vacuum"] = {"max_abs_u": float(np.max(np.abs(u))), "max_abs_v": float(np.max(np.abs(v))), "A_gauss": info["A_gauss"], "B_gauss": info["B_gauss"],
                         "residual": info["residual"]}
    else:
        rs = 2 * a.E0
        G = UGrid(int(round(a.Rmax_over_rs * rs / a.h)) + 1, a.h)
        rec["meta"].update({"n": G.n, "R_max": G.Rmax, "r_s": rs})
        rec["run"] = run_dynamic(a, G, a.E0, a.R0_over_rs * rs, a.case, a.run)
    rec["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "s7a_%s.json" % a.tag), "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    print("done %s %.0f s" % (a.tag, time.time() - t0))


if __name__ == "__main__":
    main()
